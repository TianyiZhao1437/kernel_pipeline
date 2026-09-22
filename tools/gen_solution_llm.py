"""Generate one HCA solution by driving an OpenAI-compatible model through
flashinfer-bench's own KernelGenerator, then land it in this repo's layout.

Usage:
    python3 tools/gen_solution_llm.py --env ./.env.glm-5.1 \
        --author glm-5.1 --src-dir tasks/hca_compress_c128/solutions/glm_5_1
"""
import argparse
import asyncio
import inspect
import json
import os
import pathlib
import re
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# class last so that `--env` values win over anything inherited from the shell.
#
# KernelGenerator comes from the vendored tree, not from the installed package.
# It is a worked example under examples/, and the fork's pyproject declares
# package-data for py.typed and the CUTLASS headers only -- no examples/ -- so
# it is absent from site-packages and cannot be imported from the install. The
# tree it is read from and the library it drives are the same source only by
# construction: `pyproject.toml` pins flashinfer-bench to the fork commit, and
# third_party/VENDOR.md keeps this tree byte-identical to that commit's
# patched state. Verified for the current pin: the four patched files
# (bench/utils.py, evaluators/lowbit.py, eval_config.yaml, data/validate.py)
# compare equal between the two, so the prompt format cannot disagree.
GEN_DIR = REPO / "third_party/flashinfer-bench/examples/kernel_generator"
if not (GEN_DIR / "kernel_generator.py").is_file():
    raise SystemExit(
        f"kernel_generator.py missing from {GEN_DIR}; re-stage the vendored "
        "tree (third_party/VENDOR.md, 'Updating a pin')"
    )
sys.path.insert(0, str(GEN_DIR))
sys.path.insert(0, str(REPO / "tools"))

# A hang inside the generator is otherwise invisible: ptrace is blocked in this
# container, so gdb/py-spy cannot attach. faulthandler dumps every thread's stack
# from inside the process on a timer, which is the only stack we can get.
if os.environ.get("GEN_FAULT_AFTER"):
    import faulthandler
    faulthandler.dump_traceback_later(float(os.environ["GEN_FAULT_AFTER"]), exit=True)

TASK = REPO / "tasks/hca_compress_c128"
DEF_NAME = "hca_compress_c128_h512_r64"
# The root the generator evaluates its rounds against, and copies into scratch.
# It must be the same workload corpus the benchmark later measures, or the
# round-by-round feedback the model optimises against is a different -- and
# smaller -- sweep than the trace it is finally judged on. It was pinned at the
# old 20-workload hca_c128_v4 (nc = 1..1024); the corpus is now 23 workloads
# (nc = 1..8192) and v4 is a strict subset of it, so the default moved to the
# staged root. --root overrides.
ROOT = REPO / "data/trace_sets/hca_compress_c128"


def extract_code(text: str, language: str = "triton") -> str:
    """Pull the kernel source out of a reply that may be mostly prose.

    KernelGenerator's own _clean_generated_code only strips a fence when the
    reply *starts* with one. glm-5.1 does not write that way: it narrates the
    error it is fixing, then opens a fence several lines in, so every round came
    back with the prose and the bare "```python" line still attached and died as
    "unterminated string literal". Each round's failure is therefore reported
    against source that was never code.

    A fenced block is preferred when present. Otherwise the reply is cut from
    its first code-looking line to its last, which is what a model that answers
    with a preamble and no fence leaves behind.
    """
    if not text:
        return text

    m = re.search(r"```(?:python|py|triton)?[ \t]*\n(.*?)```", text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip("\n") + "\n"

    if "```" in text:
        # An unterminated fence (a truncated reply): take everything after it.
        head, _, tail = text.partition("```")
        tail = re.sub(r"^[a-zA-Z0-9_+-]*[ \t]*\n", "", tail, count=1)
        if tail.strip():
            return tail.strip("\n") + "\n"

    lines = text.splitlines()
    code_starts = ("import ", "from ", "def ", "@", "class ", "#")
    first = next((i for i, ln in enumerate(lines) if ln.startswith(code_starts)), 0)
    last = len(lines) - 1
    while last > first and not lines[last].strip():
        last -= 1

    # A model that narrates its way through a fix may abandon one attempt
    # mid-file ("Wait, I realize ... is redundant") and start a second one
    # below. Keeping both leaves prose in the middle of the source and fails as
    # a syntax error at a line that looks unrelated. The first attempt is the
    # one to keep: it is the complete one, and the round's evaluation feedback
    # is what tells the model the second one was ever wanted.
    cut = None
    for i, ln in enumerate(lines[first:last + 1], first):
        st = ln.strip()
        if not st:
            continue
        if st.startswith(("Wait,", "Wait ", "Actually,", "Hmm", "python", "```")):
            cut = i
            break
    if cut is not None:
        tail = "\n".join(lines[cut:last + 1])
        if not re.search(r"^\s*(import |from |def |@|class )", tail, re.M):
            cut = None  # the "prose" was the last thing; keep it for the error
    if cut is not None:
        last = cut - 1
        while last > first and not lines[last].strip():
            last -= 1

    return "\n".join(lines[first:last + 1]).rstrip() + "\n"


def load_env(path: pathlib.Path) -> None:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=pathlib.Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--src-dir", type=pathlib.Path, required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--code-retries", type=int, default=3,
                    help="re-ask for a round's code this many times when the "
                         "gateway returns an empty completion (content is None, "
                         "which kernel_generator.py turns into an AttributeError "
                         "on .strip()). One such drop killed an "
                         "anthropic/claude-opus-5 run at round 3 of 10.")
    ap.add_argument("--stream", action="store_true",
                    help="accumulate a streamed response instead of waiting for "
                         "one buffered body; needed for gateways that hold a long "
                         "reasoning reply open past their own idle timeout")
    ap.add_argument("--no-thinking", action="store_true",
                    help="ask the gateway to skip its reasoning pass (glm-5.1: "
                         "extra_body {\"thinking\": {\"type\": \"disabled\"}}, which "
                         "was measured to take completion reasoning from 149 tokens "
                         "to 0). Trades capability for a much shorter request.")
    ap.add_argument("--root", type=pathlib.Path, default=ROOT,
                    help="TraceSet root the rounds are evaluated against "
                         f"(default: {ROOT.relative_to(REPO)})")
    args = ap.parse_args()

    root = args.root if args.root.is_absolute() else (REPO / args.root)

    load_env(args.env)
    # The env file exports MODEL_NAME; honour it but let --model override.
    model = args.model

    # KernelGenerator builds its own AsyncOpenAI without a timeout, so it
    # inherits the 600s default and dies on kernels that take longer to write
    # out. third_party/ is not ours to edit, so inject the timeout here. The
    # completion call carries no max_tokens either, which is what the gateways
    # themselves otherwise decide; pin it so a runaway generation is bounded.
    import openai

    _real = openai.AsyncOpenAI

    # The client is patched by subclassing rather than by intercepting attribute
    # access. An earlier version hooked __getattr__ on the client and walked
    # chat -> completions -> create, which silently did nothing: `chat` is a
    # cached_property on AsyncOpenAI, so ordinary lookup finds it and
    # __getattr__ is never consulted. The result was that timeout and
    # max_retries applied (they are set in __init__, which is called normally)
    # while every per-call injection -- max_tokens, max_completion_tokens,
    # thinking, and stream -- was discarded, so the driver sent a buffered
    # request with reasoning on and no token ceiling. That is the configuration
    # this file exists to avoid. Overriding create on the real Completions class
    # cannot be bypassed that way, and _assert_patch_applied below re-checks it.
    _inject_stats = {"calls": 0}

    def _inject(kw):
        _inject_stats["calls"] += 1
        # Both names are set: the two gateways disagree on which one bounds the
        # visible output (measured -- glm counts reasoning outside the budget,
        # qwen counts it inside), and a value this high truncates under neither.
        kw.setdefault("max_tokens", args.max_tokens)
        kw.setdefault("max_completion_tokens", args.max_tokens)
        if args.no_thinking:
            body = dict(kw.get("extra_body") or {})
            body.setdefault("thinking", {"type": "disabled"})
            kw["extra_body"] = body
        if args.stream:
            kw["stream"] = True
        return kw

    _real_create = openai.resources.chat.completions.AsyncCompletions.create

    async def _patient_create(self, *a, **kw):
        _inject(kw)
        n = _inject_stats["calls"]
        if kw.get("stream"):
            return await _accumulate(await _real_create(self, *a, **kw), f"#{n}")
        return await _real_create(self, *a, **kw)

    class _PatientAsyncOpenAI(_real):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            kw.setdefault("timeout", args.timeout)
            kw.setdefault("max_retries", args.retries)
            super().__init__(*a, **kw)
            # Replace the already-constructed completions resource, since the
            # cached_property that builds it runs inside the base __init__.
            self.chat.completions = _PatchedCompletions(
                self.chat.completions._client
            )

    _PatchedCompletions = type(
        "_PatchedCompletions",
        (openai.resources.chat.completions.AsyncCompletions,),
        {"create": _patient_create},
    )

    def _assert_patch_applied():
        """Fail loudly rather than fall back to the stalling configuration."""
        probe = _PatientAsyncOpenAI(api_key="unused", base_url="http://127.0.0.1:1/v1")
        assert isinstance(probe.chat.completions, _PatchedCompletions), (
            "completions patch did not apply; the generator would send a buffered "
            "request and this run would stall"
        )
        assert probe.timeout == args.timeout, "timeout patch did not apply"
        return True

    async def _accumulate(stream, _label=""):
        """Turn an async chunk stream into the ChatCompletion the caller wants.

        KernelGenerator reads response.choices[0].message.content, which a
        stream does not carry, so the chunks are concatenated back into one
        object of the shape it expects. We stream because a gateway that
        buffers a long reasoning response can sit silent for >15 minutes and
        drop the connection; with a stream, bytes flow the whole time.
        """
        from openai.types.chat import ChatCompletion

        import time
        t0 = time.time()
        chunks, last = [], None
        async for ch in (await stream if inspect.isawaitable(stream) else stream):
            if not chunks:
                print(f"  [stream{_label}] first chunk at {time.time()-t0:.1f}s", flush=True)
            chunks.append(ch)
            last = ch
        if last is None:
            raise RuntimeError("stream produced no chunks")
        print(f"  [stream{_label}] done: {len(chunks)} chunks in {time.time()-t0:.1f}s", flush=True)

        content, reasoning, tool_calls = [], [], []
        for ch in chunks:
            if not ch.choices:
                continue
            d = ch.choices[0].delta
            if d is None:
                continue
            if getattr(d, "content", None):
                content.append(d.content)
            if getattr(d, "reasoning_content", None):
                reasoning.append(d.reasoning_content)
            for tc in getattr(d, "tool_calls", None) or []:
                tool_calls.append(tc)

        done = chunks[-1]
        usage = next((c.usage for c in reversed(chunks) if getattr(c, "usage", None)), None)
        # content is "" and never None when the reply carried no visible text.
        # A gateway that bills reasoning but emits none of it as content returns
        # chunks with deltas that never populate `content`, and `None` here is
        # not a neutral empty value: kernel_generator.py:448 reads
        # `response.choices[0].message.content.strip()`, so None raises
        # AttributeError and kills the whole run. That is what ended a
        # claude-opus-5 run at round 5 -- 1 streamed chunk in 0.3s, i.e. a
        # dropped request, not a model that declined to answer. The empty
        # string instead reaches _codegen, which is the layer that can retry.
        msg = {
            "role": "assistant",
            "content": "".join(content),
            "reasoning_content": "".join(reasoning) or None,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return ChatCompletion.model_construct(
            id=done.id, choices=[{
                "index": 0,
                "message": msg,
                "finish_reason": done.choices[0].finish_reason if done.choices else "stop",
                "logprobs": None,
            }],
            created=done.created, model=done.model, object="chat.completion",
            usage=usage,
        )

    openai.AsyncOpenAI = _PatientAsyncOpenAI
    _assert_patch_applied()

    from kernel_generator import KernelGenerator
    from flashinfer_bench import TraceSet

    # KernelGenerator evaluates every round against the trace_set it is given and
    # appends the resulting trace there, under its own round-by-round solution
    # name (e.g. "qwen3.8-max_hca_..._triton_optimized_r3_c0"). Handing it the
    # shared dataset root therefore pollutes that root with traces for solutions
    # the dataset does not contain. Give it a copy instead; only the source files
    # we write below come back out.
    scratch = REPO / "data" / "trace_sets" / f"_gen_{args.author}"
    # A scratch copied from a *different* root would evaluate rounds against the
    # wrong workloads and silently keep doing so, since the copy is only made
    # when `definitions/` is missing. Stamp the source root and re-seed when it
    # changes, removing the previous definitions/workloads first: a plain
    # copytree(dirs_exist_ok=True) only adds and overwrites, so a file that
    # shrank -- the 23-line workload sweep replacing a 20-line one -- would keep
    # its old contents and the corpus would look correct while being stale.
    stamp = scratch / ".source_root"
    want = str(root.resolve())
    if stamp.exists() and stamp.read_text().strip() != want:
        print(f"scratch {scratch.name} came from another root; re-seeding", flush=True)
        for sub in ("definitions", "workloads"):
            shutil.rmtree(scratch / sub, ignore_errors=True)
    if not (scratch / "definitions").is_dir():
        shutil.copytree(root, scratch, dirs_exist_ok=True)
    stamp.write_text(want + "\n")
    # Start from an empty trace set each run: an earlier attempt's traces would
    # otherwise be inherited and read back as if this run had produced them.
    for author_dir in (scratch / "traces").glob("*"):
        for f in author_dir.rglob("*.jsonl"):
            f.write_text("")
    trace_set = TraceSet.from_path(str(scratch))
    print(f"generator trace root: {scratch.relative_to(REPO)} (a copy of {root.name})", flush=True)
    definition = trace_set.definitions[DEF_NAME]
    workloads = trace_set.workloads[DEF_NAME]
    print(f"definition {DEF_NAME}: {len(workloads)} workloads", flush=True)

    gen = KernelGenerator(
        model_name=model,
        language="triton",
        target_gpu="H200",
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["BASE_URL"],
        use_ffi=False,
    )

    # KernelGenerator's own fence stripping only fires when the reply begins with
    # a fence, which is not how every model answers. Wrap the per-round code
    # extraction so the source that is actually built and evaluated has had the
    # surrounding prose removed. Applied to the instance, so third_party/ stays
    # as pinned.
    #
    # A gateway can also answer with an empty completion -- content is None,
    # no exception -- which kernel_generator.py:448 turns into
    # "'NoneType' object has no attribute 'strip'". That is not a bad reply to
    # argue with, it is a dropped one, and retrying is the right response: an
    # anthropic/claude-opus-5 run died at round 3 this way after a 216s stall
    # that produced 2 streamed chunks and no content. Retry the same prompt a
    # few times before surfacing it, and never hand None back to the caller,
    # which has no guard for it.
    _raw_codegen = gen._generate_code_from_prompt

    def _as_text(result):
        """Normalise a generation result so .get('cleaned') is never None.

        _codegen tolerates None via `or \"\"`, but kernel_generator.py reads the
        key itself with .strip() on the content field, so the None has to be
        replaced before the result is returned, not after.
        """
        for key in ("cleaned", "code", "raw"):
            if key in result and result[key] is None:
                result[key] = ""
        return result

    async def _codegen(prompt):
        """Ask for a round's code, tolerating a gateway that drops the request.

        Two failure shapes, both seen from these gateways and both meaning "the
        reply was lost", not "the model answered badly":

          - content arrives as None and kernel_generator.py:448 raises
            AttributeError on `.strip()` before returning anything;
          - content arrives empty (the stream carried only reasoning).

        Neither is a model mistake to argue with, so re-ask the same prompt.
        The guard has to wrap the call, not just its result: the AttributeError
        is raised *inside* _raw_codegen, so a caller inspecting the return value
        never runs. That is why an earlier version of this retry never fired.
        """
        last = None
        last_err = None
        for attempt in range(1, args.code_retries + 1):
            try:
                result = _as_text(await _raw_codegen(prompt))
            except AttributeError as e:
                # The None-content path, before _as_text can normalise it.
                last_err = e
                print(f"  empty completion (attempt {attempt}/{args.code_retries}): "
                      f"{e}; retrying", flush=True)
                await asyncio.sleep(min(2 ** attempt, 15))
                continue
            text = result.get("cleaned") or result.get("code") or result.get("raw") or ""
            if text.strip():
                if attempt > 1:
                    print(f"  code attempt {attempt} produced content", flush=True)
                cleaned = extract_code(text, "triton")
                if cleaned != result.get("cleaned"):
                    result["cleaned"] = cleaned
                return result
            last = result
            print(f"  empty completion (attempt {attempt}/{args.code_retries}); retrying",
                  flush=True)
            await asyncio.sleep(min(2 ** attempt, 15))

        if last is None and last_err is not None:
            # Every attempt raised. Re-raise rather than returning a fake empty
            # result: the round should fail with the real cause attached.
            raise last_err
        print("  all attempts returned empty; passing the empty reply through so the "
              "round fails loudly instead of silently", flush=True)
        return _as_text(last if last is not None else {"cleaned": ""})

    gen._generate_code_from_prompt = _codegen

    solution = gen.generate(trace_set=trace_set, definition=definition, gen_rounds=args.rounds)
    print(f"\ngenerated solution {solution.name} (author {solution.author})", flush=True)

    # Land it the way this repo keeps solutions: real source files on disk.
    src = args.src_dir
    if not src.is_absolute():
        src = REPO / src
    src.mkdir(parents=True, exist_ok=True)
    for f in solution.sources:
        out = src / pathlib.Path(f.path).name
        out.write_text(f.content)
        print(f"  wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}", flush=True)

    meta = src / "_generated.json"
    meta.write_text(json.dumps({
        "name": solution.name,
        "author": solution.author,
        "model": model,
        "entry_point": solution.spec.entry_point,
        "target_hardware": list(solution.spec.target_hardware),
        "language": str(solution.spec.language),
        "description": solution.description,
        "rounds": args.rounds,
    }, indent=2) + "\n")
    print(f"  wrote {meta.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
