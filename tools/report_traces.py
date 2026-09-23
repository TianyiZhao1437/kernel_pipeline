#!/usr/bin/env python3
"""Turn a task's traces into a markdown report comparing every model that ran.

This is the second half of stage 2: the models have each produced a solution and
each solution has been benchmarked, and the question is which one won and on
what basis.

Three decisions shape the output.

**Ranking is by achieved rate against a measured ceiling, not by
`speedup_factor`.** ``speedup_factor`` is solution latency over the Definition's
*reference* latency, and the reference is eager PyTorch. At small sizes that
mostly measures op-dispatch overhead -- one Triton launch against fifteen eager
launches -- and at large sizes it saturates, because the reference is
bandwidth-bound too, so a kernel can be 30x off roofline and still report a
respectable 1.2x. An achieved rate over *measured* peak has neither problem: it
is an absolute scale with a known ceiling. ``speedup_factor`` is still printed,
because it is what the Trace records and a reader will look for it, but it is
not what the table is sorted on.

Which rate depends on the task, and the task says so by which model file it
ships:

* ``bytes_model.py`` -> **GB/s against measured d2d copy bandwidth**. The right
  metric for a streaming op. See tools/roofline.py for the byte accounting.
* ``flops_model.py`` -> **TFLOP/s against a measured matmul ceiling**. The right
  metric for an op whose whole design is about converting serial work into
  dense matmuls, where a GB/s figure would score the kernel on the thing it was
  built to stop being limited by.

The matmul ceiling is not a large-GEMM peak. A task's ``flops_model.py`` may
declare ``PEAK_SHAPES`` -- the (m, k, n) tiles the op is actually made of, with
the fraction of FLOPs running at each -- and the ceiling is then measured at
those shapes and combined as a FLOP-weighted **harmonic** mean, which is the
arithmetic that applies when fractions of a fixed workload run at different
rates. For kda_prefill this matters by a factor of seven: its 64x128x128 tiles
sustain ~103 TFLOP/s on an H200 where a square 8192 GEMM sustains 772, and
ranking against 772 would compress every solution into the 5-15% band and hide
the differences the report exists to show.

**Peak is measured, not quoted.** A spec-sheet figure flatters every kernel by
the same factor and makes the percentages meaningless. ``--peak`` overrides it
for reporting on a machine other than the one that ran the traces.

**A model that failed is a row, not an omission.** The plan asks for "如果有模型
生成的kernel不可用导致没落盘等原因，追加说明" -- and the failure modes are not all
the same shape. A model can be absent because it was never run, because its
solution never compiled, because the kernel ran and was wrong, or because it
produced traces for only part of the sweep. Those are four different facts about
four different stages, and collapsing them into "failed" throws away the only
information that would tell you what to fix. The report separates them, and
distinguishes "no solution on disk" (generation failed) from "solution on disk,
no trace" (the benchmark failed) -- which is exactly the "没落盘" case.

Usage:
    python3 tools/report_traces.py tasks/hca_compress_c128
    python3 tools/report_traces.py tasks/kda_prefill_h32_d128 -o docs/reports/kda.md
    python3 tools/report_traces.py tasks/hca_compress_c128 --peak 4218 --expect claude-opus-5 gpt-6-astra
    python3 tools/report_traces.py tasks/kda_prefill_h32_d128 --metric bandwidth
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import inspect
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stage_trace_set  # noqa: E402  (needs the path above)

REPO = pathlib.Path(__file__).resolve().parent.parent

# Solution authors that are part of the task rather than a model under test.
# They still appear in the report -- the point of a baseline is to be compared
# against -- but they are not ranked among the models and their absence is not a
# failure to explain.
TASK_AUTHORS = ("baseline", "tim.zhao")


# ---------------------------------------------------------------------------
# Byte accounting, shared with roofline.py
# ---------------------------------------------------------------------------

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
    """(fn, description, takes_solution).

    Mirrors roofline.py including the signature dispatch, so a task with a
    per-solution byte model -- one that accounts for an fp32 buffer staged
    between two kernels, say -- is reported on the same footing by both tools.
    Always a 3-tuple, unlike roofline.py's fallback branch.
    """
    path = task_dir / "bytes_model.py"
    if not path.exists():
        return declared_bytes, "declared footprint (no bytes_model.py; over-counts gathered inputs)", False
    spec = importlib.util.spec_from_file_location(f"{task_dir.name}_bytes_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.streamed_bytes
    takes_solution = len(inspect.signature(fn).parameters) >= 3
    return fn, f"streamed footprint ({_rel(path)})", takes_solution


def measure_peak_bandwidth(gib: float = 0.5, iters: int = 50) -> Optional[float]:
    """Sustained d2d copy bandwidth in GB/s, or None if there is no GPU here."""
    import time

    try:
        import torch

        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001
        return None

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


def _rel(path: pathlib.Path) -> str:
    """Repo-relative for display, falling back to the path as given.

    A task_dir handed in relative to some other cwd is not an error worth
    raising from a label.
    """
    resolved = path.resolve()
    return str(resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else path)


def load_flops_model(task_dir: pathlib.Path):
    """(fn, description, takes_solution, peak_shapes) or None if the task has no FLOP model.

    Mirrors load_bytes_model. ``peak_shapes`` is the task's ``PEAK_SHAPES`` if it
    declares one -- the tile shapes the op is built from and their FLOP weights,
    used to measure a ceiling the op could actually reach.
    """
    path = task_dir / "flops_model.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"{task_dir.name}_flops_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.compute_flops
    takes_solution = len(inspect.signature(fn).parameters) >= 3
    return (
        fn,
        f"essential matmul FLOPs ({_rel(path)})",
        takes_solution,
        getattr(mod, "PEAK_SHAPES", None),
    )


def measure_peak_matmul(peak_shapes=None, iters: int = 50, batches: int = 2048):
    """Sustained bf16 matmul rate in TFLOP/s, or (None, reason).

    With ``peak_shapes`` the ceiling is measured at the op's own tile shapes and
    combined as a FLOP-weighted harmonic mean: if fraction f_i of a fixed amount
    of work runs at rate r_i then the time is sum(f_i / r_i) and the composite
    rate is its reciprocal. Averaging the rates arithmetically would overstate
    the ceiling, because the slow shapes take disproportionately long.

    Without it, falls back to a large square GEMM -- the hardware ceiling, which
    for an op made of small tiles is an unreachable one. The report says which
    was used, because the difference is a factor of seven on kda_prefill.
    """
    import time

    try:
        import torch

        if not torch.cuda.is_available():
            return None, "no GPU available"
    except Exception:  # noqa: BLE001
        return None, "no GPU available"

    def timed(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / iters

    if not peak_shapes:
        n = 8192
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        rate = 2 * n**3 / timed(lambda: torch.mm(a, b)) / 1e12
        del a, b
        torch.cuda.empty_cache()
        return rate, f"measured large square GEMM ({n}^3 bf16)"

    inverse = 0.0
    measured = []
    for label, m, k, n, weight in peak_shapes:
        x = torch.randn(batches, m, k, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(batches, k, n, device="cuda", dtype=torch.bfloat16)
        rate = 2 * batches * m * k * n / timed(lambda: torch.bmm(x, y)) / 1e12
        del x, y
        torch.cuda.empty_cache()
        measured.append(f"{label} {m}x{k}x{n} {rate:.0f}")
        inverse += weight / rate

    # Weights need not sum to 1 -- a task may exclude non-matmul terms, as
    # kda_prefill excludes its UT transform. Renormalise so the result is a rate
    # for the matmul work that was measured, not a rate diluted by work that has
    # no shape to measure.
    total_weight = sum(w for *_, w in peak_shapes)
    return total_weight / inverse, (
        "measured at the op's own tile shapes, FLOP-weighted harmonic mean of ["
        + "; ".join(measured)
        + f"] TFLOP/s covering {100 * total_weight:.0f}% of FLOPs"
    )


class Metric:
    """What the report ranks on: a work quantity per workload and a rate.

    Exists so that bandwidth and compute are one code path rather than two. The
    tables, the ranking and the JSON all read ``row.work`` and ``row.rate`` and
    get their units from here.
    """

    __slots__ = ("kind", "unit", "work_unit", "work_scale", "work_fmt", "work_of",
                 "takes_solution", "model_desc", "peak_shapes", "rationale")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def rate(self, work: float, ms: float) -> float:
        if ms <= 0:
            return 0.0
        return work / self.work_scale / (ms / 1e3)


def choose_metric(task_dir: pathlib.Path, requested: str) -> Metric:
    """Pick the metric from the task's own model files, or honour an override.

    ``auto`` prefers flops_model.py when one exists: a task only ships one if its
    author decided the op is compute-bound, and that decision belongs with the
    task rather than with whoever runs the report.
    """
    flops = load_flops_model(task_dir)

    if requested == "auto":
        requested = "compute" if flops else "bandwidth"

    if requested == "compute":
        if flops is None:
            raise SystemExit(
                f"--metric compute needs {task_dir / 'flops_model.py'}, which does not exist"
            )
        fn, desc, takes_solution, peak_shapes = flops
        return Metric(
            kind="compute",
            unit="TFLOP/s",
            work_unit="GFLOP",
            work_scale=1e12,       # FLOPs -> TFLOP, so work/scale / s = TFLOP/s
            work_fmt=lambda work: work / 1e9,
            work_of=fn,
            takes_solution=takes_solution,
            model_desc=desc,
            peak_shapes=peak_shapes,
            rationale=(
                "This task ships a `flops_model.py`, so solutions are ranked on **achieved "
                "TFLOP/s against a measured matmul ceiling**. A GB/s figure would score these "
                "kernels on the thing the chunked form, the WY representation and the UT "
                "transform exist to stop them being limited by."
            ),
        )

    fn, desc, takes_solution = load_bytes_model(task_dir)
    return Metric(
        kind="bandwidth",
        unit="GB/s",
        work_unit="MB",
        work_scale=1e9,            # bytes -> GB, so work/scale / s = GB/s
        work_fmt=lambda work: work / 1e6,
        work_of=fn,
        takes_solution=takes_solution,
        model_desc=desc,
        peak_shapes=None,
        rationale=(
            "Solutions are ranked on **achieved bandwidth as a fraction of measured peak**. "
            "The op streams its operands, so bytes moved over time is the quantity with a "
            "known ceiling."
        ),
    )


def measure_peak(metric: Metric):
    """The ceiling for this metric, as (value, source)."""
    if metric.kind == "compute":
        return measure_peak_matmul(metric.peak_shapes)
    peak = measure_peak_bandwidth()
    return peak, "measured d2d copy" if peak else "no GPU available"


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------


class Row:
    """One (solution, workload) measurement.

    ``work`` and ``rate`` are in the Metric's units -- bytes and GB/s, or FLOPs
    and TFLOP/s -- rather than named for either, so the tables do not have to
    know which metric is in play.
    """

    __slots__ = ("author", "solution", "axes", "axes_key", "work", "ms", "rate", "speedup",
                 "matched", "max_abs", "max_rel", "hardware")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def gather(root: pathlib.Path, task_dir: pathlib.Path, metric: Metric):
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(str(root))

    # Solution name -> author, so a trace (which names only the solution) can be
    # attributed. Traces are also filed under an author directory, but the name
    # is what the Trace carries, so go through the Solution.
    author_of: Dict[str, str] = {}
    solutions_present: Dict[str, str] = {}
    for _def, sols in trace_set.solutions.items():
        for s in sols:
            author_of[s.name] = s.author
            solutions_present[s.author] = s.name

    rows: List[Row] = []
    failures: List[dict] = []
    workload_count: Dict[str, int] = {}

    for def_name, traces in sorted(trace_set.traces.items()):
        definition = trace_set.definitions.get(def_name)
        if definition is None:
            continue
        workload_count[def_name] = len(trace_set.workloads.get(def_name, []))
        for t in traces:
            if t.is_workload_trace() or t.evaluation is None:
                continue
            author = author_of.get(t.solution, "?")
            axes = dict(t.workload.axes)
            key = " ".join(f"{k}={v}" for k, v in sorted(axes.items()))
            status = t.evaluation.status.value
            if status != "PASSED" or t.evaluation.performance is None:
                failures.append(
                    {
                        "author": author,
                        "solution": t.solution,
                        "axes": key,
                        "status": status,
                        "log_tail": "\n".join((t.evaluation.log or "").strip().splitlines()[-4:]),
                    }
                )
                continue
            work = (
                metric.work_of(definition, axes, t.solution)
                if metric.takes_solution
                else metric.work_of(definition, axes)
            )
            ms = t.evaluation.performance.latency_ms
            extra = (t.evaluation.correctness.extra or {}) if t.evaluation.correctness else {}
            rows.append(
                Row(
                    author=author,
                    solution=t.solution,
                    axes=axes,
                    axes_key=key,
                    work=work,
                    ms=ms,
                    rate=metric.rate(work, ms),
                    speedup=t.evaluation.performance.speedup_factor,
                    matched=extra.get("matched_ratio"),
                    max_abs=t.evaluation.correctness.max_absolute_error if t.evaluation.correctness else None,
                    max_rel=t.evaluation.correctness.max_relative_error if t.evaluation.correctness else None,
                    hardware=t.evaluation.environment.hardware,
                )
            )

    return trace_set, rows, failures, workload_count, solutions_present


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def fmt(v, spec="", dash="--"):
    if v is None:
        return dash
    return format(v, spec)


def axis_sort_key(k: str):
    """Order workloads by axis value numerically where the values are numbers.

    A lexical sort puts ``max_position=1048576`` before ``max_position=128``,
    which makes a sweep table unreadable. Axis values are not *required* to be
    integers, so anything non-numeric falls back to the string.
    """
    parts = []
    for p in k.split():
        v = p.split("=", 1)[1] if "=" in p else p
        parts.append((0, int(v), "") if v.lstrip("-").isdigit() else (1, 0, v))
    return parts


def render(
    task_dir: pathlib.Path,
    root: pathlib.Path,
    trace_set,
    rows: List[Row],
    failures: List[dict],
    workload_count: Dict[str, int],
    metric: Metric,
    solutions_present: Dict[str, str],
    peak: Optional[float],
    peak_source: str,
    expected: Sequence[str],
) -> str:
    out: List[str] = []
    w = out.append

    hardware = sorted({r.hardware for r in rows if r.hardware})
    n_def = len(trace_set.definitions)
    def_name = next(iter(trace_set.definitions), "?")
    n_workloads = workload_count.get(def_name, 0)

    w(f"# Trace report: {task_dir.name}")
    w("")
    w(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
      f"by `tools/report_traces.py`.")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| task | `{task_dir.relative_to(REPO) if task_dir.is_relative_to(REPO) else task_dir}` |")
    w(f"| trace set | `{root.relative_to(REPO) if root.is_relative_to(REPO) else root}` |")
    w(f"| definition | `{def_name}`{'' if n_def == 1 else f' (+{n_def - 1} more)'} |")
    w(f"| workloads | {n_workloads} |")
    w(f"| hardware | {', '.join(hardware) if hardware else 'not recorded'} |")
    w(f"| measured peak | {fmt(peak, '.1f' if metric.kind == 'compute' else '.0f')} "
      f"{metric.unit} ({peak_source}) |")
    w(f"| work model | {metric.model_desc} |")
    w("")

    # --- ranking ---------------------------------------------------------
    by_author: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_author[r.author].append(r)

    w("## Ranking")
    w("")
    w(metric.rationale)
    w("")
    w("Not ranked on `speedup_factor`. The reference it is measured against is eager PyTorch,")
    w("which at small sizes measures op-dispatch overhead and at large sizes saturates because")
    w("it is limited by the same resource -- a kernel can be 30x off roofline and still report")
    w("a respectable 1.2x. The `speedup` column is kept because the Trace records it, not")
    w("because it is the objective.")
    w("")

    # One decimal for TFLOP/s (values are O(10-100)), none for GB/s (O(1000)).
    rate_fmt = ".1f" if metric.kind == "compute" else ".0f"
    unit = metric.unit

    if peak:
        w(f"| # | model | author | best {unit} | % of peak | at | worst {unit} | "
          f"median {unit} | best speedup | workloads |")
        w("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    else:
        w(f"| # | model | author | best {unit} | at | worst {unit} | median {unit} | "
          f"best speedup | workloads |")
        w("|---|---|---|---:|---|---:|---:|---:|---:|")

    ranked = []
    for author, rs in by_author.items():
        best = max(rs, key=lambda r: r.rate)
        worst = min(rs, key=lambda r: r.rate)
        med = sorted(r.rate for r in rs)[len(rs) // 2]
        ranked.append((best.rate, author, best, worst, med, rs))
    ranked.sort(key=lambda t: -t[0])

    for i, (bg, author, best, worst, med, rs) in enumerate(ranked, 1):
        tag = " *(task baseline)*" if author in TASK_AUTHORS else ""
        cols = [
            str(i),
            f"`{best.solution}`",
            f"{author}{tag}",
            format(bg, rate_fmt),
        ]
        if peak:
            cols.append(f"{100 * bg / peak:.1f}%")
        cols += [
            f"`{best.axes_key}`",
            format(worst.rate, rate_fmt),
            format(med, rate_fmt),
            f"{max(r.speedup for r in rs):.2f}x",
            f"{len(rs)}/{n_workloads}",
        ]
        w("| " + " | ".join(cols) + " |")
    w("")

    if peak and ranked:
        top = ranked[0]
        w(f"Best kernel reaches **{100 * top[0] / peak:.1f}% of the measured "
          f"{format(peak, rate_fmt)} {unit} ceiling** "
          f"({peak / top[0]:.1f}x off roofline) at `{top[2].axes_key}`.")
        if len(ranked) > 1:
            w("")
            w(f"Spread between best and worst model: **{top[0] / ranked[-1][0]:.2f}x** on best-case "
              f"{unit} ({format(top[0], rate_fmt)} vs {format(ranked[-1][0], rate_fmt)}).")
        w("")

    # --- per-workload ----------------------------------------------------
    w(f"## Per-workload {metric.kind} ({unit})")
    w("")
    w(f"One row per **distinct shape**, one column per model. The rate is the "
      f"{metric.work_unit} figure over measured latency; a blank cell means that model has no "
      f"passing trace at that shape.")
    w("")

    authors = [a for _, a, _, _, _, _ in ranked]
    keys = sorted({r.axes_key for r in rows}, key=axis_sort_key)

    # A workload corpus may contain two workloads with identical axes -- different
    # uuids, different input blobs, same shape. They are distinct measurements and
    # must not silently overwrite each other in a shape-keyed table, so collect a
    # list per cell, show the best, and say how many were folded in.
    cell: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
    for r in rows:
        cell[(r.axes_key, r.author)].append(r)
    collisions = sum(len(v) - 1 for v in cell.values())

    w(f"| workload | {metric.work_unit} | " + " | ".join(authors) + " |")
    w("|---|---:|" + "---:|" * len(authors))
    for k in keys:
        any_row = next(r for r in rows if r.axes_key == k)
        cells = []
        best_here = max(
            (max(x.rate for x in cell[(k, a)]) for a in authors if (k, a) in cell), default=0.0
        )
        for a in authors:
            group = cell.get((k, a))
            if not group:
                cells.append("--")
                continue
            g = max(x.rate for x in group)
            txt = f"**{format(g, rate_fmt)}**" if g == best_here else format(g, rate_fmt)
            if len(group) > 1:
                txt += f" *(best of {len(group)})*"
            cells.append(txt)
        work = metric.work_fmt(any_row.work)
        w(f"| `{k}` | {work:.0f} | " if work >= 10 else f"| `{k}` | {work:.2f} | ")
        out[-1] += " | ".join(cells) + " |"
    w("")
    if collisions:
        w(f"> {len(keys)} distinct shapes over {n_workloads} workloads: some workloads share a "
          f"shape and differ only in their input blobs. Cells marked *(best of n)* fold those "
          f"together; the ranking table above counts every workload separately.")
        w("")

    # --- correctness -----------------------------------------------------
    w("## Correctness")
    w("")
    w("Worst observation per model over every passing trace. `matched_ratio` is a **minimum")
    w("across the outputs**, each over its own element count, not pooled -- so a single byte")
    w("of a small output can set the figure.")
    w("")
    w("| model | worst matched_ratio | at | max abs err | max rel err |")
    w("|---|---:|---|---:|---:|")
    for _, author, _, _, _, rs in ranked:
        with_ratio = [r for r in rs if r.matched is not None]
        if with_ratio:
            worst = min(with_ratio, key=lambda r: r.matched)
            w(f"| {author} | {worst.matched:.8f} | `{worst.axes_key}` | "
              f"{max(r.max_abs for r in rs if r.max_abs is not None):.3e} | "
              f"{max(r.max_rel for r in rs if r.max_rel is not None):.3e} |")
        else:
            w(f"| {author} | not recorded | -- | "
              f"{fmt(max((r.max_abs for r in rs if r.max_abs is not None), default=None), '.3e')} | "
              f"{fmt(max((r.max_rel for r in rs if r.max_rel is not None), default=None), '.3e')} |")
    w("")

    # --- what did not land ----------------------------------------------
    w("## Models that did not land")
    w("")

    ran = set(by_author)
    notes: List[str] = []

    # Four distinct stages, four distinct notes. Collapsing them loses the only
    # information that says what to fix.
    for name in expected:
        if name in ran:
            continue
        if name in solutions_present:
            notes.append(
                f"- **{name}** — a solution is staged (`{solutions_present[name]}`) but it has "
                f"**no passing trace**. The kernel exists; the benchmark did not produce a "
                f"measurement for it. Check the failure table below, then "
                f"`tools/run_benchmark.py <task> --solutions {solutions_present[name]}`."
            )
        else:
            notes.append(
                f"- **{name}** — **no solution on disk**. Generation did not produce a kernel "
                f"that could be staged, so nothing reached the benchmark. This is the "
                f"\"没落盘\" case: the model ran but its reply never became an artefact. "
                f"Re-run `tools/gen_solution_llm.py` for this model and read its log."
            )

    # Partial sweeps: present, but not everywhere.
    for _, author, _, _, _, rs in ranked:
        if n_workloads and len(rs) < n_workloads:
            notes.append(
                f"- **{author}** — landed **{len(rs)}/{n_workloads}** workloads. Its ranking "
                f"row is computed over the workloads it completed, so it is not directly "
                f"comparable to a model that completed all of them."
            )

    if failures:
        by_status: Dict[str, List[dict]] = defaultdict(list)
        for f in failures:
            by_status[f["status"]].append(f)
        notes.append("")
        notes.append(f"{len(failures)} failed evaluation(s):")
        notes.append("")
        notes.append("| model | solution | workload | status |")
        notes.append("|---|---|---|---|")
        for status in sorted(by_status):
            for f in by_status[status][:20]:
                notes.append(f"| {f['author']} | `{f['solution']}` | `{f['axes']}` | `{status}` |")
            if len(by_status[status]) > 20:
                notes.append(f"| | | | *(+{len(by_status[status]) - 20} more `{status}`)* |")

        # One log excerpt per distinct (model, status). The table says a kernel
        # failed; only the log says why, and a COMPILE_ERROR that is the same
        # error twenty times over is one fact, not twenty.
        seen: set = set()
        excerpts: List[Tuple[str, str, str]] = []
        for f in failures:
            key = (f["author"], f["status"])
            if key in seen or not f["log_tail"]:
                continue
            seen.add(key)
            excerpts.append((f["author"], f["status"], f["log_tail"]))
        if excerpts:
            notes.append("")
            notes.append("<details><summary>Log excerpts (last lines, one per model &amp; status)</summary>")
            notes.append("")
            for author, status, tail in excerpts:
                notes.append(f"**{author} — `{status}`**")
                notes.append("")
                notes.append("```")
                notes.extend(tail.splitlines())
                notes.append("```")
                notes.append("")
            notes.append("</details>")

    if notes:
        out.extend(notes)
    else:
        w("Every expected model produced a passing trace at every workload. Nothing to report.")
    w("")

    w("---")
    w("")
    w("Regenerate with:")
    w("")
    w("```")
    rel = task_dir.relative_to(REPO) if task_dir.is_relative_to(REPO) else task_dir
    w(f"python3 tools/report_traces.py {rel}"
      + (f" --metric {metric.kind}" if metric.kind == "bandwidth" and metric.peak_shapes is None
         and (task_dir / "flops_model.py").exists() else "")
      + (f" --peak {format(peak, rate_fmt)}" if peak else "")
      + (f" --expect {' '.join(expected)}" if expected else ""))
    w("```")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task_dir", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="TraceSet root")
    ap.add_argument("-o", "--output", type=pathlib.Path, default=None, help="write here instead of stdout")
    ap.add_argument("--peak", type=float, default=None, help="ceiling to use instead of measuring")
    ap.add_argument("--no-measure", action="store_true", help="do not touch the GPU; omit %% of peak")
    ap.add_argument(
        "--metric",
        choices=("auto", "bandwidth", "compute"),
        default="auto",
        help=(
            "auto (default) picks compute when the task ships a flops_model.py, bandwidth "
            "otherwise. Override to see the same traces under the other unit."
        ),
    )
    ap.add_argument(
        "--expect",
        nargs="*",
        default=None,
        metavar="AUTHOR",
        help="models that were asked to run; any that produced no trace gets an explanation",
    )
    ap.add_argument("--json", type=pathlib.Path, default=None, help="also write the raw rows as JSON")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    root = args.root or stage_trace_set.default_root(args.task_dir)
    if not root.exists():
        raise SystemExit(f"no staged root at {root}; run tools/stage_trace_set.py first")

    import flashinfer_bench  # noqa: F401  (registers the real package)

    metric = choose_metric(task_dir, args.metric)

    if args.peak is not None:
        peak, peak_source = args.peak, "given"
    elif args.no_measure:
        peak, peak_source = None, "not measured"
    else:
        peak, peak_source = measure_peak(metric)

    trace_set, rows, failures, wc, sols = gather(root, task_dir, metric)
    if not rows and not failures:
        raise SystemExit(f"no evaluated traces under {root / 'traces'}")

    # Default expectation: every author that has a solution staged but is not
    # part of the task itself. That makes "a model generated a kernel and it
    # never produced a number" visible without having to be told the roster.
    expected = args.expect
    if expected is None:
        expected = sorted(a for a in sols if a not in TASK_AUTHORS)

    md = render(
        task_dir, root, trace_set, rows, failures, wc, metric, sols, peak, peak_source, expected
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"report written to {args.output}  ({len(rows)} measurement(s), {len(failures)} failure(s))")
    else:
        print(md)

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "task": task_dir.name,
                    "metric": metric.kind,
                    "unit": metric.unit,
                    "peak": peak,
                    "peak_source": peak_source,
                    "rows": [{k: getattr(r, k) for k in Row.__slots__} for r in rows],
                    "failures": failures,
                },
                indent=2,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
