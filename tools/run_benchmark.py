#!/usr/bin/env python3
"""Benchmark a staged task and dump Traces.

This is the step that turns an authored task into data: it builds every Solution,
runs it against every workload in the sweep, scores it against the Definition's
reference, times both, and writes one Trace per (workload, solution) pair under
``<root>/traces/<author>/<op_type>/<definition>.jsonl``.

Two things about the configuration are easy to get wrong and are handled here:

* The task's ``eval_config.yaml`` must be passed explicitly.
  ``BenchmarkConfig.default()`` loads the copy bundled inside the vendored
  package, whose ``required_matched_ratio`` is 1.0 -- i.e. bitwise equality,
  which no kernel over a 128-token softmax can satisfy (see the derivation in
  tasks/hca_compress_c128/eval_config.yaml). Using the default would report
  INCORRECT_NUMERICAL for a correct solution.
* ``profile_baseline`` lives at the top level of BenchmarkConfig as well as in
  the per-definition layer, and the top level defaults to True regardless, so
  the reference is timed and ``speedup_factor`` is meaningful.

Usage:
    python3 tools/run_benchmark.py tasks/hca_compress_c128
    python3 tools/run_benchmark.py tasks/hca_compress_c128 --solutions foo --resume
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stage_trace_set  # noqa: E402  (needs the path above)

# Import the real package before anything touches fib_shim, which registers a
# stub `flashinfer_bench` so the schema parsers load without the package
# __init__ (tools/fib_shim.py). Benchmark needs the real __init__; both shim
# helpers no-op when the module is already in sys.modules, so importing for
# real first makes the shim defer to it.
#
# flashinfer_bench resolves through the editable install of the vendored tree
# (pyproject.toml's [tool.uv.sources]), so it needs no path plumbing here and
# the runner's spawned workers inherit it without a PYTHONPATH prefix.
#
# There is no evaluator registration here either. hca_compress is routed to
# LowBitEvaluator and given its tolerance inside the vendored tree
# (third_party/patches/002-hca-compress-eval-routing.patch), so the stock
# resolver finds it in the parent and in every worker alike.
import flashinfer_bench  # noqa: E402,F401  (registers the real package)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="TraceSet root (default: data/trace_sets/<task>/)")
    ap.add_argument("--config", type=pathlib.Path, default=None, help="BenchmarkConfig YAML (default: <task>/eval_config.yaml)")
    ap.add_argument("--solutions", nargs="*", default=None, help="only these solution names")
    ap.add_argument("--resume", action="store_true", help="skip (workload, solution) pairs already traced")
    ap.add_argument("--no-stage", action="store_true", help="use the root as-is instead of refreshing it from the task")
    ap.add_argument("--isolated", action="store_true", help="one subprocess per evaluation instead of persistent workers")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    root = args.root or stage_trace_set.default_root(args.task_dir)
    config_path = args.config or args.task_dir / "eval_config.yaml"
    if not config_path.exists():
        raise SystemExit(f"no eval config at {config_path}; pass --config")

    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if not args.no_stage:
        stage_trace_set.stage(args.task_dir, root, quiet=True)
        print(f"staged {args.task_dir} -> {root}")

    from flashinfer_bench.bench import Benchmark, BenchmarkConfig
    from flashinfer_bench.bench.evaluators import resolve_evaluator
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(str(root))
    overrides = {"use_isolated_runner": True} if args.isolated else {}
    if args.solutions:
        overrides["solutions"] = args.solutions
    config = BenchmarkConfig.from_yaml(str(config_path), **overrides)

    for name, defn in trace_set.definitions.items():
        ev = config.resolve_eval_config(defn)
        evaluator = resolve_evaluator(defn)
        print(
            f"\n{name} ({defn.op_type}): rtol={ev.rtol} atol={ev.atol} "
            f"matched>={ev.required_matched_ratio} warmup={ev.warmup_runs} "
            f"iters={ev.iterations} trials={ev.num_trials} baseline={ev.profile_baseline}"
        )
        print(
            f"  {len(trace_set.solutions.get(name, []))} solution(s) x "
            f"{len(trace_set.workloads.get(name, []))} workload(s)"
            f"   evaluator={evaluator.__name__}"
        )

    bench = Benchmark(trace_set, config)
    started = time.time()
    try:
        result = bench.run_all(dump_traces=True, resume=args.resume)
    finally:
        bench.close()
    elapsed = time.time() - started

    print(f"\n=== {sum(len(v) for v in result.traces.values())} trace(s) in {elapsed:.1f}s ===")
    failures = 0
    for def_name, traces in sorted(result.traces.items()):
        for t in sorted(traces, key=lambda t: t.workload.uuid):
            ev = t.evaluation
            axes = " ".join(f"{k}={v}" for k, v in sorted(t.workload.axes.items()))
            if ev.status.value == "PASSED":
                p = ev.performance
                extra = ev.correctness.extra or {}
                ratio = extra.get("matched_ratio")
                ratio_s = f" matched={ratio:.8f}" if ratio is not None else ""
                print(
                    f"  PASSED  {t.solution}  {axes}\n"
                    f"          {p.latency_ms:.4f} ms vs ref {p.reference_latency_ms:.4f} ms "
                    f"= {p.speedup_factor:.2f}x   max_abs={ev.correctness.max_absolute_error:.3e} "
                    f"max_rel={ev.correctness.max_relative_error:.3e}{ratio_s}"
                )
            else:
                failures += 1
                print(f"  {ev.status.value}  {t.solution}  {axes}")
                if ev.log:
                    tail = "\n".join(ev.log.strip().splitlines()[-15:])
                    print("          " + tail.replace("\n", "\n          "))

    print(f"\ntraces written under {root / 'traces'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
