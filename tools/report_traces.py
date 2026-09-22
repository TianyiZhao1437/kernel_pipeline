#!/usr/bin/env python3
"""Turn a task's traces into a markdown report comparing every model that ran.

This is the second half of stage 2: the models have each produced a solution and
each solution has been benchmarked, and the question is which one won and on
what basis.

Three decisions shape the output.

**Ranking is by achieved bandwidth, not by speedup_factor.** ``speedup_factor``
is solution latency over the Definition's *reference* latency, and the reference
is eager PyTorch. At small sizes that mostly measures op-dispatch overhead --
one Triton launch against fifteen eager launches -- and at large sizes it
saturates, because the reference is bandwidth-bound too, so a kernel can be 30x
off roofline and still report a respectable 1.2x. Achieved bandwidth over
*measured* peak has neither problem: it is an absolute scale with a known
ceiling. ``speedup_factor`` is still printed, because it is what the Trace
records and a reader will look for it, but it is not what the table is sorted
on. See tools/roofline.py for the byte accounting.

**Peak is measured, not quoted.** A spec-sheet figure flatters every kernel by
the same factor and makes the percentages meaningless. The default is a
device-to-device copy measured at run time; ``--peak`` overrides it for
reporting on a machine other than the one that ran the traces.

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
    python3 tools/report_traces.py tasks/hca_compress_c128 -o docs/reports/hca.md
    python3 tools/report_traces.py tasks/hca_compress_c128 --peak 4218 --expect claude-opus-5 gpt-6-astra
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
    return fn, f"streamed footprint ({path.relative_to(REPO)})", takes_solution


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


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------


class Row:
    """One (solution, workload) measurement."""

    __slots__ = ("author", "solution", "axes", "axes_key", "nbytes", "ms", "gbs", "speedup",
                 "matched", "max_abs", "max_rel", "hardware")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def gather(root: pathlib.Path, task_dir: pathlib.Path, peak: Optional[float]):
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet.from_path(str(root))
    bytes_of, model_desc, takes_solution = load_bytes_model(task_dir)

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
            nbytes = (
                bytes_of(definition, axes, t.solution) if takes_solution else bytes_of(definition, axes)
            )
            ms = t.evaluation.performance.latency_ms
            extra = (t.evaluation.correctness.extra or {}) if t.evaluation.correctness else {}
            rows.append(
                Row(
                    author=author,
                    solution=t.solution,
                    axes=axes,
                    axes_key=key,
                    nbytes=nbytes,
                    ms=ms,
                    gbs=nbytes / 1e9 / (ms / 1e3) if ms > 0 else 0.0,
                    speedup=t.evaluation.performance.speedup_factor,
                    matched=extra.get("matched_ratio"),
                    max_abs=t.evaluation.correctness.max_absolute_error if t.evaluation.correctness else None,
                    max_rel=t.evaluation.correctness.max_relative_error if t.evaluation.correctness else None,
                    hardware=t.evaluation.environment.hardware,
                )
            )

    return trace_set, rows, failures, workload_count, model_desc, solutions_present


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
    model_desc: str,
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
    w(f"| measured peak | {fmt(peak, '.0f')} GB/s ({peak_source}) |")
    w(f"| byte model | {model_desc} |")
    w("")

    # --- ranking ---------------------------------------------------------
    by_author: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_author[r.author].append(r)

    w("## Ranking")
    w("")
    w("Ranked on **best achieved bandwidth as a fraction of measured peak**, not on")
    w("`speedup_factor`. The reference this task's `speedup_factor` is measured against is")
    w("eager PyTorch, which at small sizes measures op-dispatch overhead and at large sizes")
    w("saturates because it is bandwidth-bound too -- a kernel can be 30x off roofline and")
    w("still report a respectable 1.2x. The `speedup` column is kept because the Trace")
    w("records it, not because it is the objective.")
    w("")

    if peak:
        w("| # | model | author | best GB/s | % of peak | at | worst GB/s | median GB/s | best speedup | workloads |")
        w("|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    else:
        w("| # | model | author | best GB/s | at | worst GB/s | median GB/s | best speedup | workloads |")
        w("|---|---|---|---:|---|---:|---:|---:|---:|")

    ranked = []
    for author, rs in by_author.items():
        best = max(rs, key=lambda r: r.gbs)
        worst = min(rs, key=lambda r: r.gbs)
        med = sorted(r.gbs for r in rs)[len(rs) // 2]
        ranked.append((best.gbs, author, best, worst, med, rs))
    ranked.sort(key=lambda t: -t[0])

    for i, (bg, author, best, worst, med, rs) in enumerate(ranked, 1):
        tag = " *(task baseline)*" if author in TASK_AUTHORS else ""
        cols = [
            str(i),
            f"`{best.solution}`",
            f"{author}{tag}",
            f"{bg:.0f}",
        ]
        if peak:
            cols.append(f"{100 * bg / peak:.1f}%")
        cols += [
            f"`{best.axes_key}`",
            f"{worst.gbs:.0f}",
            f"{med:.0f}",
            f"{max(r.speedup for r in rs):.2f}x",
            f"{len(rs)}/{n_workloads}",
        ]
        w("| " + " | ".join(cols) + " |")
    w("")

    if peak and ranked:
        top = ranked[0]
        w(f"Best kernel reaches **{100 * top[0] / peak:.1f}% of the measured {peak:.0f} GB/s peak** "
          f"({peak / top[0]:.1f}x off roofline) at `{top[2].axes_key}`.")
        if len(ranked) > 1:
            w("")
            w(f"Spread between best and worst model: **{top[0] / ranked[-1][0]:.2f}x** on best-case "
              f"bandwidth ({top[0]:.0f} vs {ranked[-1][0]:.0f} GB/s).")
        w("")

    # --- per-workload ----------------------------------------------------
    w("## Per-workload bandwidth (GB/s)")
    w("")
    w("One row per **distinct shape**, one column per model. Bandwidth is the byte footprint")
    w("over measured latency; a blank cell means that model has no passing trace at that shape.")
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

    w("| workload | MB | " + " | ".join(authors) + " |")
    w("|---|---:|" + "---:|" * len(authors))
    for k in keys:
        any_row = next(r for r in rows if r.axes_key == k)
        cells = []
        best_here = max(
            (max(x.gbs for x in cell[(k, a)]) for a in authors if (k, a) in cell), default=0.0
        )
        for a in authors:
            group = cell.get((k, a))
            if not group:
                cells.append("--")
                continue
            g = max(x.gbs for x in group)
            txt = f"**{g:.0f}**" if g == best_here else f"{g:.0f}"
            if len(group) > 1:
                txt += f" *(best of {len(group)})*"
            cells.append(txt)
        mb = any_row.nbytes / 1e6
        w(f"| `{k}` | {mb:.0f} | " if mb >= 10 else f"| `{k}` | {mb:.2f} | ")
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
      + (f" --peak {peak:.0f}" if peak else "")
      + (f" --expect {' '.join(expected)}" if expected else ""))
    w("```")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task_dir", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="TraceSet root")
    ap.add_argument("-o", "--output", type=pathlib.Path, default=None, help="write here instead of stdout")
    ap.add_argument("--peak", type=float, default=None, help="GB/s to use instead of measuring")
    ap.add_argument("--no-measure", action="store_true", help="do not touch the GPU; omit %% of peak")
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

    if args.peak is not None:
        peak, peak_source = args.peak, "given"
    elif args.no_measure:
        peak, peak_source = None, "not measured"
    else:
        peak = measure_peak_bandwidth()
        peak_source = "measured d2d copy" if peak else "no GPU available"

    trace_set, rows, failures, wc, model_desc, sols = gather(root, task_dir, peak)
    if not rows and not failures:
        raise SystemExit(f"no evaluated traces under {root / 'traces'}")

    # Default expectation: every author that has a solution staged but is not
    # part of the task itself. That makes "a model generated a kernel and it
    # never produced a number" visible without having to be told the roster.
    expected = args.expect
    if expected is None:
        expected = sorted(a for a in sols if a not in TASK_AUTHORS)

    md = render(
        task_dir, root, trace_set, rows, failures, wc, model_desc, sols, peak, peak_source, expected
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
                    "peak_gbps": peak,
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
