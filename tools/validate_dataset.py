#!/usr/bin/env python3
"""Run flashinfer-bench's own dataset validator over a staged TraceSet root.

Why this wrapper exists rather than `flashinfer-bench validate`:

* It resolves the staged root from a task directory, so the two spellings
  ``tasks/hca_compress_c128`` and ``--root data/trace_sets/...`` both work, and
  it turns the report into an exit status.

Note what the validator does *not* do: it discovers the dataset from path depth
and silently skips files at the wrong depth, so a root staged by hand is likely
to report "0 definitions" rather than an error. Stage with
tools/stage_trace_set.py, which writes the layout the validator expects.

Usage:
    python3 tools/validate_dataset.py tasks/hca_compress_c128
    python3 tools/validate_dataset.py tasks/hca_compress_c128 --disable-gpu
    python3 tools/validate_dataset.py --root data/trace_sets/hca_compress_c128
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stage_trace_set  # noqa: E402  (needs the path above)

# flashinfer_bench resolves through the editable install of the vendored tree
# (pyproject.toml's [tool.uv.sources]), so no sys.path or PYTHONPATH plumbing is
# needed here and the validator's subprocesses inherit the same copy.
import flashinfer_bench  # noqa: E402,F401  (registers the real package)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=pathlib.Path, nargs="?", default=None)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="TraceSet root to validate")
    ap.add_argument("--disable-gpu", action="store_true", help="skip the benchmark check")
    ap.add_argument("--checks", default=None, help="comma-separated subset of checks")
    ap.add_argument(
        "--outputs", default="stdout", help="comma-separated: stdout,json,text (default: stdout)"
    )
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()

    if args.root is None:
        if args.task_dir is None:
            raise SystemExit("pass a task_dir or --root")
        args.root = stage_trace_set.default_root(args.task_dir)
    if not args.root.exists():
        raise SystemExit(f"no staged root at {args.root}; run tools/stage_trace_set.py first")

    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    from flashinfer_bench.data.validate import validate_dataset

    report = validate_dataset(
        dataset=str(args.root),
        checks=args.checks.split(",") if args.checks else None,
        disable_gpu=args.disable_gpu,
        outputs=args.outputs.split(",") if args.outputs else ["stdout"],
    )

    statuses = {name: rep.status for name, rep in report.definitions.items()}
    return 1 if any(s == "error" for s in statuses.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
