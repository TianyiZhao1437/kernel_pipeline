#!/usr/bin/env python3
"""Load pieces of the vendored flashinfer-bench without executing its __init__.

``import flashinfer_bench`` runs a package __init__ that pulls the agent and
bench stacks and ultimately ``flashinfer.testing`` -- a heavyweight CUDA package
that is not installed here and is not needed to type-check a Definition or to
score two tensors. Importing the leaf modules directly keeps the schema parsers
and the scoring function usable on any machine.

This is a read-only view of the pinned tree. Nothing here patches it; see
third_party/VENDOR.md.
"""

import importlib.util
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "third_party" / "flashinfer-bench" / "flashinfer_bench"


def _ensure_namespace(name: str, path: pathlib.Path) -> None:
    """Register a stub package so submodule imports resolve without __init__."""
    if name in sys.modules:
        return
    shim = types.ModuleType(name)
    shim.__path__ = [str(path)]
    sys.modules[name] = shim


def _load(module_name: str, file_path: pathlib.Path, is_package: bool = False):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
        submodule_search_locations=[str(file_path.parent)] if is_package else None,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_data():
    """The Definition / Workload / Solution parsers."""
    if not (PKG / "data" / "__init__.py").exists():
        sys.exit(
            f"missing {PKG / 'data'}.\n"
            "The vendored tree is incomplete -- see third_party/VENDOR.md."
        )
    _ensure_namespace("flashinfer_bench", PKG)
    return _load("flashinfer_bench.data", PKG / "data" / "__init__.py", is_package=True)


def load_bench_utils():
    """``compute_error_stats`` and ``ResolvedEvalConfig``, the real ones.

    Scoring and tolerance resolution are what every measured number in a task's
    eval_config is downstream of, so they are imported rather than reimplemented.
    """
    _ensure_namespace("flashinfer_bench", PKG)
    _ensure_namespace("flashinfer_bench.bench", PKG / "bench")
    config = _load("flashinfer_bench.bench.config", PKG / "bench" / "config.py")
    utils = _load("flashinfer_bench.bench.utils", PKG / "bench" / "utils.py")
    return types.SimpleNamespace(
        compute_error_stats=utils.compute_error_stats,
        ResolvedEvalConfig=config.ResolvedEvalConfig,
    )
