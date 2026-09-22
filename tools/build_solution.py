#!/usr/bin/env python3
"""Assemble a flashinfer-bench Solution JSON from a directory of source files.

The sources stay on disk as real, runnable, lintable files; this only wraps them
into the ``sources[]`` array. Hand-escaping a Triton kernel into a JSON string
literal is how the earlier reference acquired silent edit failures, so nothing
here is authored inline.

Usage:
    python3 tools/build_solution.py \
        --src-dir tasks/hca_compress_c128/solutions/triton_h200 \
        --out tasks/hca_compress_c128/hca_compress_c128_triton_h200.solution.json \
        --name hca_compress_c128_triton_h200 \
        --definition hca_compress_c128_h512_r64 \
        --entry-point hca_compress_c128.py::run \
        --language triton \
        --target-hardware NVIDIA_H200 \
        --description-file <path>
"""

import argparse
import json
import pathlib

SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--definition", required=True)
    # Hand-authored solutions are credited to the human who wrote them. The
    # seed was previously labelled "claude-opus-5" purely because this default
    # said so, which made a hand-written baseline read as a model attempt.
    ap.add_argument("--author", default="tim.zhao")
    ap.add_argument("--entry-point", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--target-hardware", action="append", required=True)
    ap.add_argument("--dependency", action="append", default=[])
    ap.add_argument(
        "--returns-outputs",
        action="store_true",
        help="Solution returns its outputs instead of writing into caller-allocated tensors.",
    )
    ap.add_argument("--description-file", type=pathlib.Path)
    args = ap.parse_args()

    # Paths are stored relative to --src-dir: the builder materialises a fresh
    # temporary source directory and lays the files out by these paths, so they
    # must not reach outside it.
    sources = []
    for path in sorted(args.src_dir.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        sources.append(
            {
                "path": str(path.relative_to(args.src_dir)),
                "content": path.read_text(),
            }
        )
    if not sources:
        raise SystemExit(f"no source files under {args.src_dir}")

    entry_file = args.entry_point.split("::")[0]
    if entry_file not in {s["path"] for s in sources}:
        raise SystemExit(
            f"entry point file {entry_file!r} is not among the collected sources: "
            + ", ".join(s["path"] for s in sources)
        )

    solution = {
        "name": args.name,
        "definition": args.definition,
        "author": args.author,
        "spec": {
            "language": args.language,
            "target_hardware": args.target_hardware,
            "entry_point": args.entry_point,
            "dependencies": args.dependency,
            "destination_passing_style": not args.returns_outputs,
        },
        "sources": sources,
    }
    if args.description_file:
        solution["description"] = args.description_file.read_text().strip()

    args.out.write_text(json.dumps(solution, indent=2) + "\n")
    total = sum(len(s["content"]) for s in sources)
    print(f"wrote {args.out}  ({len(sources)} source file(s), {total} chars)")


if __name__ == "__main__":
    main()
