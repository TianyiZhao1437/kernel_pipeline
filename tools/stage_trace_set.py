#!/usr/bin/env python3
"""Materialise a flat tasks/ directory as a FlashInfer-Trace dataset root.

Two consumers read such a root, and they do NOT agree on how strict it is:

* ``TraceSet.from_path`` rglobs four subdirectories and reads the op_type and
  definition name out of the JSON, so it accepts almost any nesting
  (flashinfer_bench/data/trace_set.py::from_path).
* ``flashinfer_bench.data.validate`` *discovers* the dataset from the paths
  themselves, and silently skips anything whose depth is wrong
  (validate.py:222-281). A flat ``definitions/<definition>.json`` is not an
  error -- it is invisible, and the report says "0 definitions".

So the layout written here is the strict one, which satisfies both::

    <root>/definitions/<op_type>/<definition>.json                      (2 parts)
    <root>/workloads/<op_type>/<definition>.jsonl                       (2 parts)
    <root>/solutions/<author>/<op_type>/<definition>/<solution>.json    (4 parts)
    <root>/traces/<author>/<op_type>/<definition>.jsonl                 (3 parts)

Every path component is also a *field* -- the validator cross-checks op_type,
author, definition and solution name against the JSON -- so the filename of a
solution must be ``<solution.name>.json``, not whatever it was called under
tasks/.

A task under tasks/ is deliberately flat instead: one directory holds the
Definition, its Solutions, the workload sweep, and the eval_config, so the whole
task reads as a unit and diffs as a unit. This script bridges the two, which
keeps the authored layout independent of the dataset's directory convention.

The staged root is a derived artefact and lands under data/ (gitignored) by
default. Files are copied rather than symlinked so the root is a snapshot: a
benchmark run and the traces it dumps refer to exactly the bytes that were
staged. Re-run the script to refresh; existing traces/ are left alone, so a
refresh does not discard measurements.

Blobs
-----

A workload input may be a ``SafetensorsInput``, whose ``path`` is resolved
relative to the *TraceSet root* -- so the blobs are part of the root, not part
of the task. They are not in tasks/ (they are 271 MB of generated tensors, and
tasks/ is checked in), so staging has to find them. ``--blobs`` names the
directory to take them from; by default every ``data/trace_sets/*/blob`` is
searched. A missing blob is a hard error rather than a warning: the workload
would otherwise load fine here and fail at ``gen_inputs`` time, several minutes
into a benchmark.

If the task carries a ``blobs.sha256`` manifest, every staged blob is also
checked against it. That is not belt-and-braces: the fallback search matches on
the root-relative path alone, and blob filenames are workload uuids, which are
stable across regenerations -- so a root built from a different corpus or a
different seed has exactly the paths being looked for and satisfies the lookup
silently.

Usage:
    python3 tools/stage_trace_set.py tasks/hca_compress_c128
    python3 tools/stage_trace_set.py tasks/hca_compress_c128 --root /tmp/ts
"""

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fib_shim  # noqa: E402  (needs the path above)

REPO = fib_shim.REPO


def stage(
    task_dir: pathlib.Path,
    root: pathlib.Path,
    quiet: bool = False,
    blob_src: pathlib.Path | None = None,
) -> pathlib.Path:
    """Copy one task's artefacts into a TraceSet root. Returns the root."""
    data = fib_shim.load_data()
    log = (lambda *a: None) if quiet else print

    definitions = {}
    solutions = []
    for path in sorted(task_dir.glob("*.json")):
        obj = json.loads(path.read_text())
        # A Solution has a `spec`; a Definition does not. Same discriminator
        # tools/validate_task.py uses.
        if "spec" in obj:
            solutions.append((path, data.Solution.model_validate(obj)))
        else:
            defn = data.Definition.model_validate(obj)
            definitions[defn.name] = (path, defn)

    if not definitions:
        raise SystemExit(f"no Definition found in {task_dir}")

    for sub in ("definitions", "solutions", "workloads", "traces"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    blobs: set = set()

    for path, defn in definitions.values():
        dest = root / "definitions" / defn.op_type / f"{defn.name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        log(f"  definition  {defn.name}  -> {dest.relative_to(root)}")

    for path, sol in solutions:
        if sol.definition not in definitions:
            raise SystemExit(
                f"{path.name} references definition '{sol.definition}', "
                f"which is not in {task_dir}"
            )
        op_type = definitions[sol.definition][1].op_type
        # Filename must be the solution's own name: the validator reads it back
        # out of the path and compares it to the `name` field.
        dest = root / "solutions" / sol.author / op_type / sol.definition / f"{sol.name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        log(f"  solution    {sol.name}  -> {dest.relative_to(root)}")

    # Workload files must be named for the definition they belong to and sit
    # under the op_type, because that is where a run dumps its own workloads and
    # where every other tool expects to find them.
    for path in sorted(task_dir.glob("*.jsonl")):
        traces = [
            data.Trace.model_validate(json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not traces:
            continue
        names = {t.definition for t in traces}
        if len(names) != 1:
            raise SystemExit(f"{path.name} mixes definitions {sorted(names)}; split it per definition")
        name = names.pop()
        if name not in definitions:
            raise SystemExit(f"{path.name} is for definition '{name}', which is not in {task_dir}")
        not_workloads = [t.workload.uuid for t in traces if not t.is_workload_trace()]
        if not_workloads:
            raise SystemExit(
                f"{path.name} contains evaluated traces ({len(not_workloads)}), but "
                "workloads/ must hold workload-only traces; from_path asserts this"
            )
        op_type = definitions[name][1].op_type
        dest = root / "workloads" / op_type / f"{name}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        log(f"  workloads   {len(traces):>3} x {name}  -> workloads/{op_type}/{dest.name}")

        # SafetensorsInput.path is relative to the TraceSet root, so the blobs
        # belong to the root and have to be brought along.
        for t in traces:
            for spec in t.workload.inputs.values():
                if getattr(spec, "type", None) == "safetensors":
                    blobs.add(spec.path)

    for rel in sorted(blobs):
        dest = root / rel
        src = _find_blob(rel, blob_src, exclude=root)
        if dest.exists() and src is not None and dest.stat().st_size == src.stat().st_size:
            continue
        if dest.exists():
            continue
        if src is None:
            raise SystemExit(
                f"workload references blob '{rel}', which is not under {root} and was not "
                f"found in {blob_src or 'any data/trace_sets/*/'}. Regenerate it "
                f"(tools/gen_workload_blobs.py --root {root}) or pass --blobs."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    if blobs:
        nbytes = sum((root / r).stat().st_size for r in blobs)
        log(f"  blobs       {len(blobs):>3} safetensors  ({nbytes / 2**20:.1f} MB)")
        _verify_blobs(task_dir, root, blobs, log)

    return root


def _verify_blobs(
    task_dir: pathlib.Path, root: pathlib.Path, blobs: set, log
) -> None:
    """Check the staged blobs against the task's sha256 manifest, if it has one.

    Worth the second of hashing because ``_find_blob`` will take a blob from any
    other root under data/trace_sets/ that happens to have the same relative
    path, first match by sort order wins. Blobs are named by workload uuid and
    the uuids are stable across regenerations, so a root built from an older
    corpus or an older seed has exactly the paths being looked for and satisfies
    the lookup silently. Nothing downstream would notice: SafetensorsInput
    carries no digest, and a Trace names its definition and solution by name
    with no content hash, so the run would simply record new numbers under the
    old workload's identity.
    """
    manifest = task_dir / "blobs.sha256"
    if not manifest.exists():
        return
    import hashlib

    want = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        want[rel] = digest

    bad = []
    for rel in sorted(blobs):
        expected = want.get(rel)
        if expected is None:
            bad.append(f"  {rel}: referenced by a workload but absent from blobs.sha256")
            continue
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != expected:
            bad.append(f"  {rel}\n    expected {expected}\n    got      {got}")
    if bad:
        raise SystemExit(
            f"staged blobs do not match {manifest}:\n" + "\n".join(bad) + "\n\n"
            "Either a stale blob was picked up from another root under "
            "data/trace_sets/ (pass --blobs to name the right one), or the blobs "
            "were regenerated and every recorded trace is now stale. In the "
            "latter case re-run tools/gen_workload_blobs.py, refresh the manifest, "
            "and re-run tools/run_benchmark.py -- do not just update the manifest."
        )
    log(f"  blobs       verified against {manifest.name}")


def _find_blob(
    rel: str, blob_src: pathlib.Path | None, exclude: pathlib.Path
) -> pathlib.Path | None:
    """Locate a blob by its root-relative path.

    Searched in order: an explicit ``--blobs`` root, then every other staged
    TraceSet under data/trace_sets/. Blobs are content-addressed by workload
    uuid, so the same relative path in two roots is the same tensor.
    """
    roots = [blob_src] if blob_src else sorted((REPO / "data" / "trace_sets").glob("*"))
    for r in roots:
        if r is None or r.resolve() == exclude.resolve():
            continue
        cand = r / rel
        if cand.is_file():
            return cand
    return None


def default_root(task_dir: pathlib.Path) -> pathlib.Path:
    return REPO / "data" / "trace_sets" / task_dir.resolve().name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=pathlib.Path)
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="TraceSet root to write (default: data/trace_sets/<task>/)",
    )
    ap.add_argument(
        "--blobs",
        type=pathlib.Path,
        default=None,
        help="TraceSet root to take workload safetensors blobs from "
        "(default: search data/trace_sets/*/)",
    )
    args = ap.parse_args()

    root = args.root or default_root(args.task_dir)
    print(f"staging {args.task_dir} -> {root}")
    stage(args.task_dir, root, blob_src=args.blobs)

    # Prove the result loads through the real loader rather than asserting the
    # layout is right by inspection.
    ts = fib_shim.load_data().TraceSet.from_path(str(root))
    print(
        f"\nTraceSet.from_path ok: {len(ts.definitions)} definition(s), "
        f"{sum(len(v) for v in ts.solutions.values())} solution(s), "
        f"{sum(len(v) for v in ts.workloads.values())} workload(s), "
        f"{sum(len(v) for v in ts.traces.values())} existing trace(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
