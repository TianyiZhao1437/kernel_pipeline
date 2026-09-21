"""Generate one HCA solution by driving an OpenAI-compatible model through
flashinfer-bench's own KernelGenerator, then land it in this repo's layout.

Usage:
    python3 tools/gen_solution_llm.py --env ./.env.glm-5.1 \
        --author glm-5.1 --src-dir tasks/hca_compress_c128/solutions/glm_5_1
"""
import argparse
import asyncio
import json
import os
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
GEN_DIR = REPO / "third_party/flashinfer-bench/examples/kernel_generator"
sys.path.insert(0, str(GEN_DIR))
sys.path.insert(0, str(REPO / "tools"))

TASK = REPO / "tasks/hca_compress_c128"
DEF_NAME = "hca_compress_c128_h512_r64"
ROOT = REPO / "data/trace_sets/hca_c128_v4"


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
    args = ap.parse_args()

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

    class _PatientAsyncOpenAI(_real):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            kw.setdefault("timeout", args.timeout)
            kw.setdefault("max_retries", args.retries)
            super().__init__(*a, **kw)

        def __getattr__(self, item):
            # chat -> completions -> create(**_inject)
            if item != "chat":
                return super().__getattr__(item)
            chat = super().__getattr__(item)

            class _Chat:
                def __getattr__(self, sub):
                    if sub != "completions":
                        return getattr(chat, sub)
                    comp = getattr(chat, sub)

                    class _Completions:
                        def __getattr__(self, name):
                            fn = getattr(comp, name)
                            if name != "create":
                                return fn

                            def wrapped(*a, **kw):
                                # Both names are set: the two gateways disagree
                                # on which one bounds the visible output
                                # (measured -- glm counts reasoning outside the
                                # budget, qwen counts it inside), and a value
                                # this high truncates under neither.
                                kw.setdefault("max_tokens", args.max_tokens)
                                kw.setdefault("max_completion_tokens", args.max_tokens)
                                return fn(*a, **kw)

                            return wrapped

                    return _Completions()

            return _Chat()

    openai.AsyncOpenAI = _PatientAsyncOpenAI

    from kernel_generator import KernelGenerator
    from flashinfer_bench import TraceSet

    # KernelGenerator evaluates every round against the trace_set it is given and
    # appends the resulting trace there, under its own round-by-round solution
    # name (e.g. "qwen3.8-max_hca_..._triton_optimized_r3_c0"). Handing it the
    # shared dataset root therefore pollutes that root with traces for solutions
    # the dataset does not contain. Give it a copy instead; only the source files
    # we write below come back out.
    scratch = REPO / "data" / "trace_sets" / f"_gen_{args.author}"
    if not (scratch / "definitions").is_dir():
        shutil.copytree(ROOT, scratch, dirs_exist_ok=True)
    trace_set = TraceSet.from_path(str(scratch))
    print(f"generator trace root: {scratch.relative_to(REPO)} (a copy of {ROOT.name})", flush=True)
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
