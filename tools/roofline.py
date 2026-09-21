#!/usr/bin/env python3
"""Score a task's traces against the GPU's measured bandwidth, not against eager PyTorch.

Why this exists
---------------

``speedup_factor`` in a Trace is solution latency over the Definition's
reference latency, and the reference is a chain of eager PyTorch ops. For a
memory-bound kernel that makes it a poor objective at both ends of the sweep:

* at small sizes it mostly measures op-dispatch overhead -- one Triton launch
  against fifteen eager launches -- which is the toy signal this project exists
  to avoid;
* at large sizes it saturates, because the reference is bandwidth-bound too, so
  a kernel can be 30x off roofline and still report a respectable 1.2x.

Achieved bandwidth as a fraction of what the device can actually sustain has
neither problem: it is an absolute scale with a known ceiling.

Peak is measured, not taken from a spec sheet: a large device-to-device copy is
the closest cheap proxy for a streaming kernel's ceiling.

Byte accounting
---------------

By default every declared input and output counts once -- correct for a kernel
that streams all of its operands. A task whose kernel only touches part of a
declared tensor (a gathered cache, say) can override this with a
``bytes_model.py`` next to its Definition exporting
``streamed_bytes(definition, axes) -> int``; see
tasks/hca_compress_c128/bytes_model.py for the case that motivated the hook.
Without it the footprint is an over-count, so the reported bandwidth is a lower
bound and the kernel looks better than it is.

Usage:
    python3 tools/roofline.py tasks/hca_compress_c128
    python3 tools/roofline.py tasks/hca_compress_c128 --root data/trace_sets/other
"""

import argparse
import importlib.util
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stage_trace_set  # noqa: E402  (needs the path above)

REPO = stage_trace_set.REPO

ITEMSIZE = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float4_e2m1": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "bool": 1,
}


def declared_bytes(definition, axes) -> int:
    """Every declared operand, counted once. The generic fallback."""
    named = list(zip(definition.inputs.items(), definition.get_input_shapes(axes))) + list(
        zip(definition.outputs.items(), definition.get_output_shapes(axes))
    )
    total = 0
    for (_name, spec), shape in named:
        elems = 1
        for extent in shape:
            elems *= extent
        total += elems * ITEMSIZE[spec.dtype.value]
    return total


def load_bytes_model(task_dir: pathlib.Path):
    path = task_dir / "bytes_model.py"
    if not path.exists():
        return declared_bytes, "declared footprint (no bytes_model.py; over-counts gathered inputs)"
    spec = importlib.util.spec_from_file_location(f"{task_dir.name}_bytes_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.streamed_bytes, f"streamed footprint ({path})"


def measure_peak_bandwidth(gib: float = 0.5, iters: int = 50) -> float:
    """Sustained device-to-device copy bandwidth in GB/s (reads + writes)."""
    import torch

    n = int(gib * 2**30) // 2
    src = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    per_iter = (time.perf_counter() - started) / iters
    return 2 * src.numel() * src.element_size() / 1e9 / per_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="TraceSet root (default: data/trace_sets/<task>/)")
    ap.add_argument("--peak", type=float, default=None, help="GB/s to use instead of measuring")
    args = ap.parse_args()

    root = args.root or stage_trace_set.default_root(args.task_dir)
    sys.path.insert(0, str(REPO / "third_party" / "flashinfer-bench"))
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(str(root))
    bytes_of, model_desc = load_bytes_model(args.task_dir)

    peak = args.peak if args.peak is not None else measure_peak_bandwidth()
    import torch

    print(f"device       {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).multi_processor_count} SMs)")
    print(f"peak         {peak:.0f} GB/s  ({'given' if args.peak else 'measured d2d copy'})")
    print(f"byte model   {model_desc}\n")

    for def_name, traces in sorted(trace_set.traces.items()):
        definition = trace_set.definitions.get(def_name)
        if definition is None:
            print(f"{def_name}: no Definition in {root}, skipping")
            continue
        by_solution = {}
        for t in traces:
            if t.evaluation is None or t.evaluation.performance is None:
                continue
            by_solution.setdefault(t.solution, []).append(t)

        for sol_name, sol_traces in sorted(by_solution.items()):
            print(f"{def_name}  /  {sol_name}")
            print(
                f"  {'axes':<34} {'MB':>8} {'ms':>9} {'GB/s':>8} {'% peak':>7} "
                f"{'roof ms':>8} {'x off':>6} {'vs ref':>7}"
            )
            rows = []
            for t in sol_traces:
                axes = t.workload.axes
                nbytes = bytes_of(definition, axes)
                ms = t.evaluation.performance.latency_ms
                gbs = nbytes / 1e9 / (ms / 1e3)
                roof_ms = nbytes / (peak * 1e9) * 1e3
                rows.append(
                    (
                        sorted(axes.items()),
                        " ".join(f"{k}={v}" for k, v in sorted(axes.items())),
                        nbytes / 1e6,
                        ms,
                        gbs,
                        100 * gbs / peak,
                        roof_ms,
                        ms / roof_ms,
                        t.evaluation.performance.speedup_factor,
                    )
                )
            for r in sorted(rows, key=lambda r: r[2]):
                print(
                    f"  {r[1]:<34} {r[2]:>8.1f} {r[3]:>9.4f} {r[4]:>8.1f} {r[5]:>7.2f} "
                    f"{r[6]:>8.4f} {r[7]:>6.1f} {r[8]:>7.2f}"
                )
            best = max(rows, key=lambda r: r[4])
            print(
                f"\n  best {best[4]:.1f} GB/s = {best[5]:.2f}% of peak at {best[1]} "
                f"({best[7]:.1f}x off roofline)\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
