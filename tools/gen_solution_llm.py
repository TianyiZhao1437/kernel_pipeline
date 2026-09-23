"""Generate one solution for a task by driving an OpenAI-compatible model
through flashinfer-bench's own KernelGenerator, then land it in this repo's
layout.

Usage:
    python3 tools/gen_solution_llm.py --model-name glm-5.1 \
        --task-dir tasks/kda_prefill_h32_d128

The task is addressed by directory, not by module constant. Everything else --
the definition name, the TraceSet root the generation rounds are evaluated
against, the eval config those rounds are scored with, and the directory the
sources land in -- is derived from it, so adding a task requires no edit here.
See the ``Task`` class for the derivation and what each piece is load-bearing
for.

The model is addressed by its key in ``models.yaml`` (tools/model_config.py),
which replaced the per-model ``.env.<model>`` files. The registry carries the
author under which the solution is filed, so the author cannot drift from the
gateway that produced it.
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

# The class comes last so that resolved `models.yaml` values win over anything
# inherited from the shell.
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

DEFAULT_TASK = REPO / "tasks/hca_compress_c128"


class Task:
    """Everything this tool needs to know about a task, derived from its dir.

    These were four module constants pinned to hca_compress_c128
    (docs/architecture.md, gap 2). Deriving them means a second task needs no
    edit here, and -- more usefully -- it means the four cannot disagree with
    each other, which is the failure docs/contracts.md records as "divergent
    generation and measurement roots".

    * ``def_name`` is read off the task directory rather than restated, so it
      cannot drift from the file that defines it.
    * ``root`` is the staged TraceSet root. It must be the same workload corpus
      ``tools/run_benchmark.py`` later measures, or the round-by-round feedback
      the model optimises against is a different -- and usually smaller -- sweep
      than the trace it is finally judged on.
    * ``eval_config`` is the task's own tolerance file, and it is the piece that
      is easiest to omit and most expensive to omit. ``kernel_generator.py:343``
      constructs a bare ``BenchmarkConfig()``, which loads no eval config at all
      and therefore resolves ``required_matched_ratio`` to ``None`` --
      ``compute_error_stats`` reads that as 1.0, i.e. bitwise equality. A
      correct bf16 kernel is then reported to the model as
      INCORRECT_NUMERICAL every round, and it spends its whole budget chasing an
      error that is not there. hca_compress_c128 is immune by accident: patch
      002 routes it to LowBitEvaluator, which carries its own tolerance inside
      the vendored tree. Any task on the default evaluator is not.
      ``main`` rebinds ``kernel_generator.BenchmarkConfig`` to load this file.
    """

    def __init__(self, task_dir: pathlib.Path, root: pathlib.Path | None = None):
        self.dir = task_dir if task_dir.is_absolute() else (REPO / task_dir)
        if not self.dir.is_dir():
            raise SystemExit(f"no task directory at {self.dir}")

        # The one *.json that is not a *.solution.json is the Definition.
        defs = [
            p for p in sorted(self.dir.glob("*.json"))
            if not p.name.endswith(".solution.json")
        ]
        if len(defs) != 1:
            raise SystemExit(
                f"expected exactly one definition json in {self.dir}, found "
                f"{[p.name for p in defs]}"
            )
        self.def_name = defs[0].stem

        import stage_trace_set

        self.root = root or stage_trace_set.default_root(self.dir)
        if not self.root.is_absolute():
            self.root = REPO / self.root
        if not (self.root / "definitions").is_dir():
            raise SystemExit(
                f"{self.root} is not a staged TraceSet root; run "
                f"`python3 tools/stage_trace_set.py {self.dir.relative_to(REPO)}` first"
            )

        cfg = self.dir / "eval_config.yaml"
        self.eval_config = cfg if cfg.is_file() else None

    def rel(self, path: pathlib.Path) -> str:
        return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def _task_dps(task: "Task", language: str = "triton") -> bool:
    """Does this task's `run` write into output tensors, or return them?

    Read from the task's own authored solutions rather than assumed, because
    the repo's two tasks disagree -- and disagree *within* a task. `BuildSpec`
    defaults `destination_passing_style` to True; hca_compress_c128's Triton
    solutions are True while its torch.compile baseline is False, and
    kda_prefill_h32_d128 is False throughout. So the flag has to be read from
    solutions in the language being generated, not from whichever file sorts
    first. Guessing wrong costs a build error per round, before the kernel is
    ever executed.

    Falls back to the library default when a task has no authored solution in
    this language -- there is nothing better to infer from, and the generated
    solution then at least matches what an unconfigured harness expects.
    """
    votes = []
    for p in sorted(task.dir.glob("*.solution.json")):
        try:
            spec = json.loads(p.read_text()).get("spec") or {}
        except (OSError, json.JSONDecodeError):
            continue
        if "destination_passing_style" not in spec:
            continue
        votes.append((spec.get("language"), bool(spec["destination_passing_style"])))
    same = [v for lang, v in votes if lang == language]
    if same:
        return max(set(same), key=same.count)
    return True


# The last finish_reason seen by the stream accumulator. kernel_generator's
# _generate_code_from_prompt returns only {"raw", "cleaned"} and throws the rest
# of the response away, so the one signal that tells a cut-off reply apart from
# a complete one has to be stashed on the way past.
_LAST_FINISH = {"reason": None}


def _is_truncated(text: str, finish_reason=None) -> bool:
    """Was this reply cut off mid-answer, rather than merely wrong?

    The distinction decides whether a round is spent. A model that writes
    complete code with a bug has answered, and the round should score that bug.
    A reply the ceiling severed mid-statement has not answered at all, and
    scoring it measures the gateway -- which is exactly what makes a per-model
    comparison unfair when one route truncates and the others do not.

    Measured on anthropic/claude-opus-5 via commonstack: the reasoning pass eats
    ~64k of the 65536-token cap, so what content survives stops mid-token --
    5291 chars ending at ``b_h = t`` with the ```python fence never closed.

    Two signals, applied in order of how much they actually settle:

      - an odd number of ``` fences: the block was opened and never closed, which
        only happens when the reply stopped early. Conversely a *balanced* fence
        means the model closed its own block, so the reply is complete even if
        finish_reason says "length" -- that combination is the gateway
        contradicting itself, and the code inside is a real result to score.
      - only when the reply carries no fence at all: finish_reason == "length"
        *and* the code does not parse. Length alone is not enough (a reply can
        finish exactly at the cap) and unparseable alone is not enough (that is
        a model writing bad syntax, which is worth scoring).

    Never returns True for a complete-but-broken reply, so a model cannot earn a
    retry by emitting garbage.

    Known limit: an unfenced reply severed at a point that still parses (``b_h =
    t`` is a valid assignment) is indistinguishable from a complete one by text
    alone, so it is not caught. Harmless here -- the route that truncates writes
    fenced code, and the fence rule catches it without needing the parse.
    """
    if not text or not text.strip():
        return False          # empty is the caller's other, already-handled case
    fences = text.count("```")
    if fences:
        return fences % 2 == 1
    if finish_reason == "length":
        import ast
        try:
            ast.parse(extract_code(text, "triton"))
        except (SyntaxError, ValueError):
            return True
    return False


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True,
                    help="key in models.yaml (see `python3 tools/model_config.py "
                         "--list`). Supplies the gateway, the API key and the "
                         "author the solution is filed under.")
    ap.add_argument("--task-dir", type=pathlib.Path, default=DEFAULT_TASK,
                    help="task directory; the definition name, TraceSet root and "
                         "eval config are derived from it "
                         f"(default: {DEFAULT_TASK.relative_to(REPO)})")
    ap.add_argument("--model", default=None,
                    help="override the model id from models.yaml")
    ap.add_argument("--author", default=None,
                    help="override the author from models.yaml")
    ap.add_argument("--src-dir", type=pathlib.Path, default=None,
                    help="where the generated sources land "
                         "(default: <task-dir>/solutions/<author>)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="ceiling sent as both max_tokens and "
                         "max_completion_tokens. Was 16384, which is enough for "
                         "hca_compress_c128 and not for kda_prefill_h32_d128: on "
                         "commonstack, anthropic/claude-opus-5 spent all 16384 on "
                         "reasoning and returned finish_reason=None with no "
                         "content, three times in a row, so the retry could not "
                         "help. A truncated budget looks exactly like a dropped "
                         "reply in the log -- see the 'no content' diagnostic")
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
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="cap the reasoning pass at N tokens, leaving the rest of "
                         "--max-tokens for the answer. Prefer this to "
                         "--no-thinking: it keeps the reasoning that a hard kernel "
                         "needs while guaranteeing the answer has room. Raising "
                         "--max-tokens alone does NOT help, because the reasoning "
                         "expands to fill whatever budget exists -- glm-5.1 spent "
                         "all 65536 tokens on 27013 chunks of reasoning_content and "
                         "was cut off with finish_reason=length before writing a "
                         "line of code. Measured honoured by bigmodel (reasoning "
                         "14939 -> 8349 chars, finish_reason stop, content "
                         "appeared); not verifiable on commonstack, which never "
                         "forwards Anthropic reasoning.")
    ap.add_argument("--root", type=pathlib.Path, default=None,
                    help="TraceSet root the rounds are evaluated against "
                         "(default: the task's staged root)")
    args = ap.parse_args()

    import model_config

    task = Task(args.task_dir, args.root)
    cfg = model_config.resolve(args.model_name)
    # Exports the seven env spellings the downstream consumers read; the
    # vendored KernelGenerator takes LLM_API_KEY / BASE_URL off the environment
    # and is not ours to edit. See tools/model_config.py.
    model_config.apply(cfg)
    model = args.model or cfg.model
    author = args.author or cfg.author

    root = task.root
    src_dir = args.src_dir or (task.dir / "solutions" / author)

    print(f"task       {task.rel(task.dir)}  (definition {task.def_name})", flush=True)
    print(f"model      {args.model_name} -> {model}", flush=True)
    print(f"author     {author}", flush=True)
    print(f"root       {task.rel(root)}", flush=True)
    print(f"src-dir    {task.rel(src_dir if src_dir.is_absolute() else REPO / src_dir)}",
          flush=True)

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
        elif args.thinking_budget:
            # Bound the reasoning pass, not the total. The ceiling above is the
            # sum of the two, so without this the reasoning can consume all of
            # it and the answer is never emitted -- an empty completion that no
            # amount of retrying fixes, because it is deterministic in the
            # prompt, not flaky in the gateway.
            #
            # Two spellings, because the gateways disagree: Anthropic (and
            # bigmodel, which copies it) take a nested `thinking` object, while
            # dashscope documents flat `enable_thinking` / `thinking_budget`.
            # Sending both was measured to be accepted by all three -- every
            # gateway ignores the spelling it does not know rather than
            # rejecting the request -- so one flag covers them all.
            body = dict(kw.get("extra_body") or {})
            body.setdefault("thinking", {"type": "enabled",
                                         "budget_tokens": args.thinking_budget})
            body.setdefault("enable_thinking", True)
            body.setdefault("thinking_budget", args.thinking_budget)
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
        if not content:
            # An empty reply after a long stream is not self-explaining: 180
            # chunks in 210s looks identical to a dropped request in the log
            # above, but one is a model that filled its budget with reasoning
            # and the other is a gateway that hung up. Report which delta keys
            # actually carried data, the finish_reason, and the token counts,
            # so the retry below is debuggable from the log alone.
            import collections
            seen = collections.Counter()
            for ch in chunks:
                if not ch.choices or ch.choices[0].delta is None:
                    continue
                dd = ch.choices[0].delta
                for k, v in (dd.model_dump() if hasattr(dd, "model_dump") else {}).items():
                    if v:
                        seen[k] += 1
            fin = done.choices[0].finish_reason if done.choices else None
            tok = ""
            if usage is not None:
                tok = (f", prompt={getattr(usage, 'prompt_tokens', '?')}"
                       f" completion={getattr(usage, 'completion_tokens', '?')}")
            print(f"  [stream{_label}] no content: finish_reason={fin}, "
                  f"delta keys {dict(seen)}{tok}", flush=True)
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
        # Stash it before the response is handed back: kernel_generator keeps
        # only the text, and _codegen needs this to tell a severed reply from a
        # complete one. See _is_truncated.
        _LAST_FINISH["reason"] = (
            done.choices[0].finish_reason if done.choices else None
        )
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

    # kernel_generator.py:343 constructs a bare `BenchmarkConfig()`, which loads
    # no eval config at all -- not even the bundled one. Every tolerance then
    # falls back to the dataclass default, and `required_matched_ratio` resolves
    # to None, which `compute_error_stats` reads as 1.0: bitwise equality. A
    # bf16 Triton kernel cannot reach that, so every round of every generation
    # would be graded FAILED and the model would spend its whole budget chasing
    # an error the task does not actually require it to remove. (hca_compress is
    # immune only by accident -- patch 002 routes it to LowBitEvaluator, which
    # carries its own hardcoded tolerance and never reads this config.)
    #
    # Rebind the name the generator resolves at call time so the rounds are
    # scored against the same file `run_benchmark.py` and `verify_task.py` use.
    # Rebinding rather than patching keeps third_party/ as pinned.
    if task.eval_config is not None:
        import kernel_generator as _kg
        _task_cfg = str(task.eval_config)
        _real_config_cls = _kg.BenchmarkConfig  # bind before shadowing the name

        def _task_benchmark_config(*a, **kw):
            return _real_config_cls.from_yaml(_task_cfg, *a, **kw)

        _kg.BenchmarkConfig = _task_benchmark_config
        print(f"eval config {task.rel(task.eval_config)} (scoring every round)", flush=True)
    else:
        print("eval config none in task dir; rounds scored at BenchmarkConfig() "
              "defaults, i.e. bitwise equality -- expect every round to FAIL",
              flush=True)

    # KernelGenerator evaluates every round against the trace_set it is given and
    # appends the resulting trace there, under its own round-by-round solution
    # name (e.g. "qwen3.8-max_hca_..._triton_optimized_r3_c0"). Handing it the
    # shared dataset root therefore pollutes that root with traces for solutions
    # the dataset does not contain. Give it a copy instead; only the source files
    # we write below come back out.
    scratch = REPO / "data" / "trace_sets" / f"_gen_{task.def_name}_{author}"
    # A scratch copied from a *different* root would evaluate rounds against the
    # wrong workloads and silently keep doing so, since the copy is only made
    # when `definitions/` is missing. Stamp the source root and re-seed when it
    # changes, removing the previous definitions/workloads first: a plain
    # copytree(dirs_exist_ok=True) only adds and overwrites, so a file that
    # shrank -- the 23-line workload sweep replacing a 20-line one -- would keep
    # its old contents and the corpus would look correct while being stale.
    #
    # The scratch name carries the definition as well as the author because one
    # author now generates for more than one task, and two tasks sharing a
    # scratch would re-seed (and re-copy) on every alternation.
    stamp = scratch / ".source_root"
    want = str(root.resolve())
    if stamp.exists() and stamp.read_text().strip() != want:
        print(f"scratch {scratch.name} came from another root; re-seeding", flush=True)
        for sub in ("definitions", "workloads"):
            shutil.rmtree(scratch / sub, ignore_errors=True)
        (scratch / "blob").unlink(missing_ok=True)
    if not (scratch / "definitions").is_dir():
        # Everything but the blobs, which are symlinked: the KDA corpus alone is
        # 3.1 GB and the generator only ever reads them, so copying one per
        # model buys nothing and costs four copies of the corpus.
        shutil.copytree(root, scratch, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("blob"))
    blob = scratch / "blob"
    if (root / "blob").is_dir() and not blob.exists():
        blob.symlink_to(root / "blob", target_is_directory=True)
    stamp.write_text(want + "\n")
    # Start from an empty trace set each run: an earlier attempt's traces would
    # otherwise be inherited and read back as if this run had produced them.
    for author_dir in (scratch / "traces").glob("*"):
        for f in author_dir.rglob("*.jsonl"):
            f.write_text("")
    trace_set = TraceSet.from_path(str(scratch))
    print(f"generator trace root: {scratch.relative_to(REPO)} (a copy of {root.name})", flush=True)
    definition = trace_set.definitions[task.def_name]
    workloads = trace_set.workloads[task.def_name]
    print(f"definition {task.def_name}: {len(workloads)} workloads", flush=True)

    gen = KernelGenerator(
        model_name=model,
        language="triton",
        target_gpu="H200",
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["BASE_URL"],
        use_ffi=False,
    )

    # The calling convention is the one thing the stock prompt never states and
    # the builder always enforces. `BuildSpec.destination_passing_style` defaults
    # to True, so a generated solution is graded as
    # `run(*inputs, *outputs) -> None`, while the prompt tells the model to
    # "restore devices for outputs" and "return results" -- which reads
    # value-returning. A model that writes the natural signature is rejected
    # before it is ever run:
    #
    #   BuildError: Destination-passing style callable: expected 8 positional
    #   parameters, but signature '(query, key, value, g, beta,
    #   initial_state=None)' is incompatible: too many positional arguments
    #
    # That is a COMPILE_ERROR the model cannot diagnose from the message, and it
    # burns a whole round. Take the convention from the task's own authored
    # solutions -- they are what `verify_task` and `run_benchmark` already
    # accept -- and both tell the model and record it on the spec.
    dps = _task_dps(task)
    _in = list(definition.inputs)
    _out = list(definition.outputs)
    if dps:
        _sig = f"def run({', '.join(_in + _out)}):  # writes into the output tensors, returns None"
    else:
        _sig = (f"def run({', '.join(_in)}):  # returns "
                + (f"({', '.join(_out)})" if len(_out) != 1 else _out[0]))
    _convention = (
        "\n\n## Required entry point signature (non-negotiable)\n\n"
        "The harness calls `run` with exactly these positional arguments, in "
        "this order, and validates the signature before executing anything:\n\n"
        f"```python\n{_sig}\n```\n\n"
        + ("Allocate nothing for the outputs -- they are passed in already "
           "allocated and you must write into them in place.\n"
           if dps else
           "Allocate the outputs yourself and return them; do not take them as "
           "parameters.\n")
        + "Extra parameters are allowed only if they have defaults. A signature "
        "that does not bind those positional arguments is rejected as a build "
        "error before the kernel runs.\n"
    )
    print(f"entry point  {_sig}", flush=True)

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
        # Every prompt -- first round and each optimisation round -- passes
        # through here, so this is the one place the convention has to be added.
        prompt = prompt + _convention
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
            if text.strip() and _is_truncated(result.get("raw") or text,
                                              _LAST_FINISH.get("reason")):
                # Content arrived, but the ceiling cut it mid-answer. Scoring it
                # would spend a round on a COMPILE_ERROR that belongs to the
                # transport, not the model, so re-ask instead -- the reasoning
                # length varies between attempts, and a shorter one leaves room
                # for the whole kernel.
                last = result
                print(f"  truncated reply (attempt {attempt}/{args.code_retries}): "
                      f"{len(text)} chars, finish_reason={_LAST_FINISH.get('reason')}, "
                      f"unterminated code block; retrying", flush=True)
                await asyncio.sleep(min(2 ** attempt, 15))
                continue
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
        print("  all attempts returned empty or truncated; passing the last reply "
              "through so the round fails loudly instead of silently", flush=True)
        return _as_text(last if last is not None else {"cleaned": ""})

    gen._generate_code_from_prompt = _codegen

    # ...and record the same convention on the spec the builder validates
    # against, so the two agree. Without this the flag stays at its True default
    # no matter what the model was told.
    _raw_make_solution = gen._create_solution_from_code

    def _make_solution(*a, **kw):
        sol = _raw_make_solution(*a, **kw)
        if sol.spec.destination_passing_style != dps:
            sol = sol.model_copy(update={
                "spec": sol.spec.model_copy(update={"destination_passing_style": dps}),
            })
        return sol

    gen._create_solution_from_code = _make_solution

    solution = gen.generate(trace_set=trace_set, definition=definition, gen_rounds=args.rounds)
    print(f"\ngenerated solution {solution.name} (author {solution.author})", flush=True)

    # Land it the way this repo keeps solutions: real source files on disk.
    src = src_dir
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
