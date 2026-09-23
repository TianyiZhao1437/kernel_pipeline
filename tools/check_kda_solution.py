"""Grade a KDA solution source against the Definition's reference on the real corpus.

The benchmark harness will do this eventually, but it does it through staging,
subprocess workers and a spawn-based runner, which is a poor place to find out
that a kernel is wrong. This runs the same comparison in-process: load the
solution's ``run``, load the Definition's reference, walk the blobbed workloads,
and report ``matched_ratio`` per output under the exact rule the library uses.

The tolerance defaults are read from the task's eval_config.yaml rather than
hardcoded, so this cannot drift from what the benchmark will actually enforce.

Timing here is wall-clock around a synchronize, not the harness's trimmed-mean
over trials -- it is a sanity check on the order of magnitude, not a benchmark
result. Read report_traces.py output for that.

    python3 tools/check_kda_solution.py tasks/kda_prefill_h32_d128/solutions/torch_compile/kda_prefill_h32_d128.py
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

import torch
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO / "tools"))


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"_sol_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tolerances(task_dir: pathlib.Path, def_name: str):
    """Read the box the benchmark will grade at, so this tool cannot disagree with it."""
    cfg_path = task_dir / "eval_config.yaml"
    if not cfg_path.exists():
        return 1e-2, 1e-2, 1.0, "library defaults (no eval_config.yaml)"
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    entry = (raw.get("definition_config") or {}).get(def_name, {})
    ratio = entry.get("required_matched_ratio")
    return (entry.get("rtol", 1e-2), entry.get("atol", 1e-2),
            1.0 if ratio is None else ratio, str(cfg_path.relative_to(REPO)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=pathlib.Path, help="solution .py exposing run(...)")
    ap.add_argument("--task-dir", type=pathlib.Path,
                    default=REPO / "tasks" / "kda_prefill_h32_d128")
    ap.add_argument("--entry-point", default="run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shapes", nargs="*", default=None, metavar="BxT")
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions after one warmup")
    args = ap.parse_args()

    from derive_kda_tolerances import load_corpus, matched_ratio
    from check_kda_numerics import load_reference

    task_dir = args.task_dir.resolve()
    def_name = task_dir.name
    reference = load_reference(task_dir / f"{def_name}.json")
    candidate = getattr(load_module(args.source.resolve()), args.entry_point)

    rtol, atol, floor, origin = tolerances(task_dir, def_name)
    print(f"grading {args.source} against {def_name}")
    print(f"tolerance from {origin}: rtol={rtol:g} atol={atol:g} "
          f"required_matched_ratio={floor:g}\n")

    shapes = None
    if args.shapes:
        shapes = {(int(s.split("x")[0]), int(s.split("x")[1])) for s in args.shapes}

    header = (f"    {'B':>2} {'T':>6}  {'out ratio':>10} {'state ratio':>11} "
              f"{'ref ms':>8} {'sol ms':>8} {'x':>6}  verdict")
    print(header)
    print("    " + "-" * (len(header) - 4))

    worst = 1.0
    failures = 0
    counted = 0
    ref_total = 0.0
    sol_total = 0.0
    for axes, inputs in load_corpus(limit=args.limit, only=shapes):
        want_out, want_state = reference(**inputs)
        got_out, got_state = candidate(**inputs)

        if got_out.shape != want_out.shape or got_state.shape != want_state.shape:
            print(f"    {axes['batch_size']:>2} {axes['seq_len']:>6}  SHAPE MISMATCH "
                  f"out {tuple(got_out.shape)} vs {tuple(want_out.shape)}, "
                  f"state {tuple(got_state.shape)} vs {tuple(want_state.shape)}")
            failures += 1
            counted += 1
            continue

        r_out = matched_ratio(got_out, want_out, rtol, atol)
        r_state = matched_ratio(got_state, want_state, rtol, atol)
        low = min(r_out, r_state)
        worst = min(worst, low)
        ok = low >= floor
        failures += not ok
        counted += 1

        def timed(fn):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.reps):
                fn(**inputs)
            torch.cuda.synchronize()
            return (time.perf_counter() - started) / args.reps * 1e3

        ref_ms = timed(reference)
        sol_ms = timed(candidate)
        ref_total += ref_ms
        sol_total += sol_ms

        print(f"    {axes['batch_size']:>2} {axes['seq_len']:>6}  {r_out:>10.6f} "
              f"{r_state:>11.6f} {ref_ms:>8.2f} {sol_ms:>8.2f} "
              f"{ref_ms / sol_ms:>6.2f} {'ok' if ok else '  <-- FAILS'}")

    if counted == 0:
        print("\nno workloads matched -- nothing was graded, which is not a pass")
        return 1
    print(f"\n    {counted} workloads, worst min-ratio {worst:.6f} against floor {floor:g}")
    print(f"    sweep total: reference {ref_total:.0f} ms, solution {sol_total:.0f} ms "
          f"({ref_total / sol_total:.2f}x)")
    if failures:
        print(f"    {failures} workload(s) would fail the benchmark")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
