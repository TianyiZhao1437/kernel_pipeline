#!/usr/bin/env python3
"""Gate an authored task against the four contracts, and say what it did not check.

``validate_task.py`` asks whether the artefacts *parse*; ``validate_dataset.py``
asks whether the vendored validator agrees. Neither asks the question this
script exists for: is the task complete, legal, free of shortcuts, and faithful
to the operation it claims to implement -- the "四组 contract" of
``docs/contracts.md``, C1..C4.

The distinction that shapes the whole file is [auto] versus [review]. A check
that can be mechanised is run here and prints a verdict. A property that needs a
human -- "is this the operation the library implements?", "is this input
distribution representative?" -- is *not* faked with a heuristic that returns
pass. It is printed as REVIEW, with the evidence the reviewer needs attached,
and it does not affect the exit status. A gate that reports a green tick for an
unanswered question is worse than no gate, because it launders the gap into
confidence.

That is why the summary line reads "N checked, M review, K failed" and not
"OK". Exit status is 1 only for a failed [auto] check.

What each contract contributes
------------------------------

C1  schema, arity, shape resolution, dataset layout. Delegated to the two
    existing tools, invoked as subprocesses so their output is not silently
    reimplemented here and cannot drift from what CI runs.

C2  anti-hack. Static: the entry must not import the reference module, read the
    axes dict or workload uuid to select behaviour, branch on a hash of the
    inputs, or ship an output constant that never derives from an input. These
    catch the shortcuts that actually appear in generated kernels. They do not
    catch a *wrong* kernel, which is the timing check's job -- a solution
    faster than the roofline for its declared byte count is not fast.

C3  input realism. Blob presence and digest against the manifest, and a
    physical-consistency probe on the tensors whose declared meaning constrains
    their values: a rotation table must satisfy cos^2 + sin^2 = 1 and |x| <= 1;
    an RMSNorm gain must be predominantly positive. Both violations were real
    in this repository and both passed every other check.

C4  trace-readiness. eval_config present and parseable, generation and
    measurement roots equal, target hardware declared, and the sweep's axis
    coverage reported so a reviewer can see which bound-handling branches of
    the operation the corpus never reaches.

Usage:
    python3 tools/verify_task.py tasks/hca_compress_c128
    python3 tools/verify_task.py tasks/hca_compress_c128 --skip-subprocess
    python3 tools/verify_task.py tasks/hca_compress_c128 --json /tmp/report.json

Exit status: 0 unless an [auto] check failed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fib_shim  # noqa: E402  (needs the path above)

REPO = fib_shim.REPO

PASS, FAIL, WARN, REVIEW, SKIP = "pass", "fail", "warn", "review", "skip"

# One glyph per verdict, so a scan of the left margin answers "did anything
# fail" without reading the messages.
GLYPH = {PASS: " ok ", FAIL: "FAIL", WARN: "warn", REVIEW: "REVIEW", SKIP: "skip"}


@dataclass
class Finding:
    contract: str
    check: str
    verdict: str
    message: str
    evidence: List[str] = field(default_factory=list)


class Report:
    def __init__(self) -> None:
        self.findings: List[Finding] = []
        self.facts: Dict[str, Any] = {}

    def add(self, contract, check, verdict, message, evidence=None) -> Finding:
        f = Finding(contract, check, verdict, message, list(evidence or []))
        self.findings.append(f)
        return f

    def emit(self, quiet_ok: bool = False) -> None:
        current = None
        for f in self.findings:
            if f.contract != current:
                current = f.contract
                print(f"\n--- {current} ---")
            if quiet_ok and f.verdict == PASS and not f.evidence:
                continue
            print(f"  [{GLYPH[f.verdict]}] {f.check}: {f.message}")
            for line in f.evidence:
                print(f"           {line}")

    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, FAIL: 0, WARN: 0, REVIEW: 0, SKIP: 0}
        for f in self.findings:
            out[f.verdict] += 1
        return out


# ---------------------------------------------------------------------------
# Reading sources without importing them
# ---------------------------------------------------------------------------

# Names that mean "something outside this file decided the answer". A kernel
# branching on any of them is not implementing the operation, it is replaying
# a lookup keyed on the test it is being given.
IDENTITY_NAMES = ("uuid", "workload_id", "workload_uuid", "case_id", "benchmark_id")


def _module_refs(tree: ast.AST) -> set:
    """Every dotted name imported or reached by attribute in the module.

    Deliberately textual rather than a resolved graph: an AST walk cannot know
    what `x = importlib.import_module(name)` does, but it can see the literal,
    and the failure this guards against is a plain `from ... import reference`.
    """
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                refs.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                refs.add(node.module)
        elif isinstance(node, ast.Attribute):
            parts = []
            cur = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            refs.add(".".join(reversed(parts)))
        elif isinstance(node, ast.Name):
            refs.add(node.id)
    return refs


def _entry_and_builder(sol):
    """(entry source text, entry symbol, every source text)."""
    entry = sol.get_entry_source()
    return entry.content, sol.get_entry_symbol(), [s.content for s in sol.sources]


def check_no_reference_import(sol, definition) -> Tuple[str, str]:
    """The entry must not reach the reference implementation.

    The reference is the answer. A solution that imports it, or shells out to
    the test that runs it, is measuring the harness rather than the kernel --
    and it will pass every correctness check by construction.
    """
    content, _, all_sources = _entry_and_builder(sol)
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return FAIL, f"entry does not parse: {exc}"
    refs = _module_refs(tree)

    # The reference lives as a string inside the Definition JSON, so there is no
    # importable module named for it in the normal case. What is detectable is
    # the machinery: the evaluator, the test file, the registry that holds the
    # reference-returning callable.
    banned = (
        "compute_error_stats",
        "allocate_outputs",
        "resolve_evaluator",
        "flashinfer_bench",
        "evaluators",
        "registry",
    )
    hits = sorted(r for r in refs if any(b in r for b in banned))
    if hits:
        return FAIL, f"entry reaches harness internals: {', '.join(hits)}"

    for name in ("test_" + definition.name, definition.name):
        if any(name in src for src in all_sources):
            return WARN, f"source mentions '{name}'; confirm it is not a reference lookup"

    return PASS, "entry does not import the reference or the evaluator"


def check_no_identity_branch(sol, definition) -> Tuple[str, str]:
    """The entry must not read the axes dict or a workload identifier.

    ``compile/builder.py`` binds positional tensor arguments only and never
    passes axes, so any route to the axis values is a deliberate detour -- and
    the only thing it buys is choosing an implementation by test case.

    What is emphatically NOT flagged is ``tensor.shape[0]``. Deriving extents
    from the tensors is how a kernel is supposed to learn its sizes; it is the
    only route the harness actually provides. The distinction that matters is
    between reading a *shape* (fine, and unavoidable) and reading an *identity*
    -- a uuid, an env var, a dict of axes that arrived out of band.
    """
    content, symbol, _ = _entry_and_builder(sol)
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return FAIL, f"entry does not parse: {exc}"

    entry = next(
        (
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol
        ),
        None,
    )
    if entry is None:
        return FAIL, f"'{symbol}' is not a module-level def, so nothing can be checked"

    hits = []
    for node in ast.walk(entry):
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name) and base.id in ("axes", "axis_values", "workload"):
                hits.append(f"subscripts '{base.id}'")
            if isinstance(base, ast.Attribute) and base.attr in ("axes", "axis_values", "workload"):
                hits.append(f"subscripts '.{base.attr}'")
        if isinstance(node, ast.Name) and node.id in IDENTITY_NAMES:
            hits.append(f"reads name '{node.id}'")
        if isinstance(node, ast.Attribute) and node.attr in IDENTITY_NAMES:
            hits.append(f"reads attribute '.{node.attr}'")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("getenv",):
                hits.append("reads the environment")
            if isinstance(f, ast.Attribute) and f.attr in ("getenv",):
                hits.append("reads the environment")
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            hits.append("reads the environment")

    if hits:
        return FAIL, "entry selects behaviour from run identity: " + "; ".join(sorted(set(hits)))
    return PASS, "entry derives everything from its tensor arguments"


def check_output_derivation(sol, definition) -> Tuple[str, str]:
    """Every output must be reachable by something that can write to it.

    Static and deliberately narrow. Three ways an output legitimately gets
    written, and all three have to count:

    * an in-place tensor method -- ``out.copy_()``, ``out[...] = ...``;
    * a Triton kernel launch -- ``kern[grid](..., out, ...)``, where the write
      happens inside the kernel and nothing at this level is visible;
    * passing it on to a helper that does one of the above.

    So what is actually asserted is *reachability*: the output name appears as
    an argument to some call, or is assigned into. An output that is never
    mentioned after the signature cannot have been written, and that is the
    failure worth catching -- a generated kernel that allocates its own result
    and returns it while the harness reads the untouched destination buffer.

    It cannot see a constant computed at import time and copied in. That is the
    empirical perturbation test's job, and it is reported as a REVIEW item
    rather than claimed here.
    """
    content, symbol, _ = _entry_and_builder(sol)
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return FAIL, f"entry does not parse: {exc}"

    entry = next(
        (
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol
        ),
        None,
    )
    if entry is None:
        return FAIL, f"'{symbol}' is not a module-level def"

    if not sol.spec.destination_passing_style:
        return SKIP, "returns outputs rather than writing them; not analysed"

    # The harness binds outputs as the LAST positional arguments.
    pos = [a.arg for a in entry.args.posonlyargs + entry.args.args]
    n_out = len(definition.outputs)
    if n_out > len(pos):
        return FAIL, f"takes {len(pos)} positional args but the harness binds {n_out} outputs"
    out_names = pos[len(pos) - n_out :]

    reached = set()
    bare_fills = []
    only_tested = set()
    for name in out_names:
        only_tested.add(name)

    for node in ast.walk(entry):
        # out.copy_(...) / out.fill_(...) / out.zero_()
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id in out_names:
                    reached.add(f.value.id)
                    only_tested.discard(f.value.id)
                    if f.attr in ("fill_", "zero_") or (
                        f.attr in ("uniform_", "normal_") and not node.args
                    ):
                        bare_fills.append(f"{f.value.id}.{f.attr}()")
        # out[...] = ... , out += ...
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                base = t.value if isinstance(t, ast.Subscript) else t
                if isinstance(base, ast.Name) and base.id in out_names:
                    reached.add(base.id)
                    only_tested.discard(base.id)

    # Every other route -- a Triton launch, a helper, a tuple that is passed on,
    # `torch.foo(..., out=ckv)` -- reduces to "the name is loaded somewhere that
    # is not a None test". Walk for that separately: the shapes are too many to
    # enumerate, and enumerating them is how a checker ends up asserting less
    # than it appears to.
    for node in ast.walk(entry):
        if isinstance(node, ast.Compare):
            # `ckv is not None` proves nothing about a write.
            continue
        if isinstance(node, ast.Call):
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Name) and sub.id in out_names:
                        reached.add(sub.id)
                        only_tested.discard(sub.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                if isinstance(elt, ast.Name) and elt.id in out_names:
                    reached.add(elt.id)
                    only_tested.discard(elt.id)
        if isinstance(node, ast.Return) and node.value is not None:
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Name) and sub.id in out_names:
                    reached.add(sub.id)

    missing = [n for n in out_names if n not in reached]
    if missing:
        return FAIL, f"output argument(s) never reached by any write or call: {', '.join(missing)}"
    if only_tested:
        return WARN, (
            "output argument(s) only appear in a None test, never in a write or a "
            f"call that could write them: {', '.join(sorted(only_tested))}"
        )
    if bare_fills:
        return WARN, "output written by a bare constant fill: " + ", ".join(bare_fills)
    return PASS, f"all {len(out_names)} output argument(s) are reachable by a write"


def check_no_host_sync(sol, definition) -> Tuple[str, str]:
    """A timing must not be manufactured inside the kernel.

    Any use of wall-clock or a sleep inside the entry is either a benchmark hack
    or a bug; the harness does its own timing.
    """
    content, _, _ = _entry_and_builder(sol)
    hits = []
    for pat in (r"\btime\.(sleep|perf_counter|time)\b", r"\bcuda\.synchronize\b", r"\bEvent\s*\("):
        if re.search(pat, content):
            hits.append(pat.strip("\\b"))
    if hits:
        return WARN, "entry touches host timing: " + ", ".join(hits)
    return PASS, "entry does not touch host timing"


# ---------------------------------------------------------------------------
# Blobs: presence, digest, and physical consistency
# ---------------------------------------------------------------------------


def read_safetensors_header(path: pathlib.Path) -> Dict[str, Any]:
    """Parse the header without mmap.

    ``safetensors.safe_open`` mmaps, which fails with OSError 19 on this host's
    overlay filesystem. The format is small enough to read by hand: an 8-byte
    little-endian header length, then that many bytes of JSON.
    """
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("file is shorter than its own length prefix")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100_000_000:
            raise ValueError(f"implausible safetensors header length {n}")
        header = json.loads(fh.read(n).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def load_blob_manifest(task_dir: pathlib.Path) -> Dict[str, str]:
    """`{relative path: sha256}` from blobs.sha256, comments stripped."""
    path = task_dir / "blobs.sha256"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0].strip()
    return out


def check_blobs(task_dir: pathlib.Path, workloads: Sequence, report: Report) -> Dict[str, pathlib.Path]:
    """C3's mechanical half: every declared blob is present and matches its digest."""
    manifest = load_blob_manifest(task_dir)
    if not manifest:
        report.add("C3", "blob manifest", WARN, "no blobs.sha256; blob identities are unchecked")
        return {}

    wanted: Dict[str, List[str]] = {}
    for wl in workloads:
        for name, spec in wl.inputs.items():
            if getattr(spec, "type", None) == "safetensors":
                wanted.setdefault(spec.path, []).append(f"{wl.uuid}:{name}")

    roots = [r for r in (REPO / "data" / "trace_sets").glob("*/blob") if r.is_dir()]

    found: Dict[str, pathlib.Path] = {}
    missing, mismatched, unlisted = [], [], []
    for rel in sorted(wanted):
        # The path is root-relative, so it is `blob/workloads/...`; look under
        # every staged root, and prefer the task's own directory.
        cands = [task_dir.parent / rel, task_dir / rel] + [r.parent / rel for r in roots]
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            missing.append(rel)
            continue
        found[rel] = hit
        digest = manifest.get(rel)
        if digest is None:
            unlisted.append(rel)
        elif digest != sha256_file(hit):
            mismatched.append(rel)
        else:
            try:
                header = read_safetensors_header(hit)
                report.facts.setdefault("blobs", {})[rel] = sorted(header)
            except Exception:  # noqa: BLE001 - a bad header is the digest check's business
                pass

    if missing:
        report.add(
            "C3",
            "blob presence",
            FAIL,
            f"{len(missing)} declared blob(s) not found on disk",
            [f"{m}   used by {', '.join(wanted[m][:2])}" for m in missing[:5]],
        )
    else:
        report.add("C3", "blob presence", PASS, f"all {len(wanted)} declared blob(s) present")

    if mismatched:
        report.add("C3", "blob digest", FAIL, f"{len(mismatched)} blob(s) do not match blobs.sha256", mismatched[:5])
    elif wanted:
        report.add("C3", "blob digest", PASS, f"{len(wanted) - len(unlisted)} blob(s) match blobs.sha256")

    if unlisted:
        report.add(
            "C3",
            "blob manifest coverage",
            FAIL,
            f"{len(unlisted)} blob(s) are not listed in blobs.sha256",
            unlisted[:5] + ["re-run tools/gen_workload_blobs.py to refresh the manifest"],
        )

    # How much of the corpus is real, per the contract's three-path model.
    n_wl = len(workloads)
    n_all_blobbed = sum(
        1 for wl in workloads if all(getattr(s, "type", None) == "safetensors" for s in wl.inputs.values())
    )
    report.add(
        "C3",
        "input materialisation",
        PASS if n_all_blobbed else WARN,
        f"{n_all_blobbed}/{n_wl} workload(s) take every input from a blob; "
        f"{n_wl - n_all_blobbed} fall back to `random`",
        ["`random` is exactly torch.randn -- no seed, no distribution (RandomInput has no fields)"] if n_all_blobbed < n_wl else [],
    )
    return found


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_tensor_semantics(
    definition, blobs: Dict[str, pathlib.Path], workloads: Sequence, report: Report
) -> None:
    """Physical-consistency probe on inputs whose declared meaning constrains them.

    This is the check whose absence let a whole sweep run on tables that were
    not rotation tables. It needs no model and no GPU: three properties that any
    correct implementation of the named tensor must satisfy, measured on the
    actual bytes.
    """
    if not blobs:
        report.add("C3", "tensor semantics", SKIP, "no blobs resolved; nothing to probe")
        return

    # Group by tensor_key so one file is read once even when four workloads share it.
    by_key: Dict[str, Tuple[pathlib.Path, str]] = {}
    for wl in workloads:
        for name, spec in wl.inputs.items():
            if getattr(spec, "type", None) == "safetensors" and spec.path in blobs:
                by_key.setdefault(f"{name}", (blobs[spec.path], spec.tensor_key))

    try:
        import numpy as np
    except ImportError:
        report.add("C3", "tensor semantics", SKIP, "numpy unavailable")
        return

    for name, (path, key) in sorted(by_key.items()):
        try:
            npy = read_safetensors_tensor(path, key)
        except Exception as exc:  # noqa: BLE001
            report.add("C3", f"tensor semantics ({name})", WARN, f"could not read {path.name}: {exc}")
            continue

        if name in ("cos_cache", "sin_cache"):
            # A rotation table: entries on the unit circle. cos^2 + sin^2 is
            # checked across the pair, which is the property that actually
            # matters and the one randn violated by ~30.
            if "cos" in name:
                continue  # handled below, once, with both halves
            continue

        if name == "rms_norm_weight":
            pos = float((npy > 0).mean())
            span = float(npy.max() / max(abs(float(npy.min())), 1e-30))
            ok = pos > 0.8
            report.add(
                "C3",
                "tensor semantics (rms_norm_weight)",
                PASS if ok else FAIL,
                f"{pos * 100:.1f}% of entries positive, span {span:.1f}x",
                [] if ok else ["a real RMSNorm gain is overwhelmingly positive; randn gives ~50%"]
                + [f"median {float(np.median(npy)):.4g}"],
            )
            if ok and pos < 0.95:
                report.add(
                    "C3",
                    "tensor semantics (rms_norm_weight)",
                    WARN,
                    f"only {pos * 100:.1f}% positive; a real gain is ~98% (503/512 on V4)",
                )

    # cos/sin together: the pair property.
    pair = {}
    for wl in workloads:
        for name, spec in wl.inputs.items():
            if name in ("cos_cache", "sin_cache") and getattr(spec, "type", None) == "safetensors":
                if spec.path in blobs:
                    pair[name] = (blobs[spec.path], spec.tensor_key)
        if len(pair) == 2:
            break
    if len(pair) == 2:
        try:
            cos = read_safetensors_tensor(*pair["cos_cache"])
            sin = read_safetensors_tensor(*pair["sin_cache"])
            ident = np.abs(cos.astype(np.float64) ** 2 + sin.astype(np.float64) ** 2 - 1.0).max()
            bound = float(max(np.abs(cos).max(), np.abs(sin).max()))
            ok = ident < 1e-3 and bound <= 1.0 + 1e-3
            report.add(
                "C3",
                "tensor semantics (cos/sin pair)",
                PASS if ok else FAIL,
                f"max |cos^2+sin^2-1| = {ident:.3e}, max |x| = {bound:.4f}",
                [] if ok else ["a rotation table lies on the unit circle; randn misses it by ~30"],
            )
        except Exception as exc:  # noqa: BLE001
            report.add("C3", "tensor semantics (cos/sin pair)", WARN, f"could not read the pair: {exc}")


def read_safetensors_tensor(path: pathlib.Path, key: str):
    """Read one tensor as numpy, without mmap (see read_safetensors_header)."""
    import numpy as np

    header = read_safetensors_header(path)
    if key not in header:
        raise KeyError(f"'{key}' not in {path.name} (has {sorted(header)[:5]})")
    meta = header[key]
    start, end = meta["data_offsets"]
    dtype = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": None,
        "I8": np.int8,
        "U8": np.uint8,
        "I32": np.int32,
        "I64": np.int64,
        "F64": np.float64,
    }[meta["dtype"]]
    with open(path, "rb") as fh:
        raw = fh.read(8)
        (n,) = struct.unpack("<Q", raw)
        fh.seek(8 + n + start)
        buf = fh.read(end - start)
    if dtype is None:  # bfloat16: numpy has no native type, shift into fp32
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32).reshape(meta["shape"])
    return np.frombuffer(buf, dtype=dtype).reshape(meta["shape"])


# ---------------------------------------------------------------------------
# C4: trace-readiness
# ---------------------------------------------------------------------------


def check_trace_ready(
    task_dir: pathlib.Path, definition, workloads: Sequence, report: Report
) -> None:
    eval_cfg = task_dir / "eval_config.yaml"
    if not eval_cfg.is_file():
        report.add(
            "C4",
            "eval config",
            FAIL,
            "no eval_config.yaml; the run would fall back to the bundled config",
            ["BenchmarkConfig.default() bundles required_matched_ratio=1.0 for an unknown "
             "definition -- bitwise equality, which no kernel over a 128-token softmax meets"],
        )
    else:
        try:
            import yaml

            cfg = yaml.safe_load(eval_cfg.read_text()) or {}
            dc = cfg.get("definition_config", {}).get(definition.name)
            if not dc:
                report.add(
                    "C4",
                    "eval config",
                    FAIL,
                    f"eval_config.yaml has no definition_config entry for '{definition.name}'",
                    [f"it would be scored by op_type_config['{definition.op_type}'] or the 1.0 fallback"],
                )
            else:
                report.add(
                    "C4",
                    "eval config",
                    PASS,
                    f"rtol={dc.get('rtol')} atol={dc.get('atol')} "
                    f"matched>={dc.get('required_matched_ratio')}",
                )
                report.facts["eval_config"] = dc
        except Exception as exc:  # noqa: BLE001
            report.add("C4", "eval config", FAIL, f"eval_config.yaml does not parse: {exc}")

    # The corpus the model optimises against must be the corpus it is judged on.
    gen = [p for p in task_dir.glob("*.jsonl") if ".real" not in p.name]
    if len(gen) == 1:
        report.add("C4", "single workload corpus", PASS, f"one swept corpus: {gen[0].name}")
    elif len(gen) == 0:
        report.add("C4", "single workload corpus", FAIL, "no workload jsonl in the task directory")
    else:
        report.add(
            "C4",
            "single workload corpus",
            FAIL,
            f"{len(gen)} workload corpora present: {', '.join(p.name for p in gen)}",
            ["generation and measurement would use different sweeps"],
        )

    # Axis coverage, reported so a reviewer can see what the sweep never reaches.
    # `type` is a plain Literal string on AxisVar/AxisConst, not an Enum, so
    # compare the string -- `.value` would work on one flashinfer-bench revision
    # and raise on another.
    var_axes = {n: a for n, a in definition.axes.items() if str(a.type) == "var"}
    coverage = {}
    for name in var_axes:
        vals = sorted({wl.axes.get(name) for wl in workloads if name in wl.axes})
        coverage[name] = vals
    report.facts["axis_coverage"] = coverage

    lines = []
    thin = []
    for name, vals in sorted(coverage.items()):
        lines.append(f"{name:<18} {len(vals):>3} distinct  min={vals[0]}  max={vals[-1]}")
        if len(vals) < 3:
            thin.append(name)
    report.add(
        "C4",
        "sweep coverage",
        WARN if thin else PASS,
        f"{len(workloads)} workload(s) over {len(coverage)} variable axis/axes",
        lines,
    )

    # Same-shape, different-data workloads are what make the bandwidth claims
    # separable from the correctness claims; their absence is worth flagging.
    by_shape = {}
    for wl in workloads:
        by_shape.setdefault(tuple(sorted(wl.axes.items())), []).append(wl)
    distinct = sum(1 for v in by_shape.values() if len(v) > 1)
    report.add(
        "C4",
        "shape collisions",
        PASS if distinct else WARN,
        f"{distinct} shape(s) carry more than one workload",
        ["same-shape/other-data pairs are what let a reviewer separate input effects "
         "from measurement noise; none exist here"] if not distinct else [],
    )

    report.add(
        "C4",
        "target hardware",
        REVIEW,
        "the declared target must match the device the traces were measured on",
        ["check Trace.evaluation.environment.hardware against a task.yaml target_hardware",
         "CLAUDE.md 12: the wheel constraint is compute capability, not driver version"],
    )


# ---------------------------------------------------------------------------
# C2: the checks that need a human
# ---------------------------------------------------------------------------


def review_items(definition, report: Report) -> None:
    report.add(
        "C2",
        "semantic reconciliation",
        REVIEW,
        "the reference must be diffed against the library implementation of the op",
        [
            "no automated check can answer 'does this match the semantics the library implements'",
            "HCA.md 8 item 9 records this as open: vLLM line-by-line vs the native",
            "DeepseekV4HCACompressor. Every convention checked so far agrees; not all are checked.",
        ],
    )
    report.add(
        "C2",
        "output perturbation",
        REVIEW,
        "static analysis cannot prove an output derives from an input",
        [
            "the mechanical form is: perturb one input, re-run, confirm the output changes.",
            "Not implemented here -- it needs a GPU run, and claiming it from AST inspection",
            "would be the exact overstatement this gate exists to avoid.",
        ],
    )


# ---------------------------------------------------------------------------
# C1: delegate to the tools that already do it
# ---------------------------------------------------------------------------


def run_subprocess(label, argv, report: Report, contract: str) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    tail = (proc.stdout or "").strip().splitlines()
    if proc.returncode == 0:
        report.add(contract, label, PASS, tail[-1] if tail else "ok")
    else:
        report.add(
            contract,
            label,
            FAIL,
            f"{' '.join(argv[1:])} exited {proc.returncode}",
            tail[-12:],
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task_dir", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, default=None, help="staged TraceSet root")
    ap.add_argument("--skip-subprocess", action="store_true", help="skip the C1 delegation")
    ap.add_argument("--skip-dataset", action="store_true", help="skip validate_dataset (slow: runs the GPU)")
    ap.add_argument("--json", type=pathlib.Path, default=None, help="write the report as JSON")
    ap.add_argument("--quiet-ok", action="store_true", help="hide passing checks")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise SystemExit(f"no such task directory: {task_dir}")

    data = fib_shim.load_data()

    report = Report()
    print(f"=== verify_task: {task_dir.name} ===")

    # ---- load the artefacts the same way the harness will -------------------
    definitions, solutions, workloads = {}, [], []
    for path in sorted(task_dir.glob("*.json")):
        obj = json.loads(path.read_text())
        model = data.Solution if "spec" in obj else data.Definition
        try:
            parsed = model.model_validate(obj)
        except Exception as exc:  # noqa: BLE001
            report.add("C1", "parse", FAIL, f"{path.name}: {exc}")
            continue
        if model is data.Definition:
            definitions[parsed.name] = parsed
        else:
            solutions.append((path, parsed))

    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                workloads.append(data.Workload.model_validate(json.loads(line)["workload"]))

    if len(definitions) != 1:
        report.add("C1", "definition count", FAIL, f"{len(definitions)} definitions in {task_dir.name}; expected 1")
    if not definitions:
        report.emit()
        return 1
    definition = next(iter(definitions.values()))

    report.add(
        "C1",
        "artefacts",
        PASS,
        f"1 definition, {len(solutions)} solution(s), {len(workloads)} workload(s)",
    )

    sols_by_author = {}
    for _, sol in solutions:
        sols_by_author.setdefault(sol.author, []).append(sol)
    if "tim.zhao" in sols_by_author:
        report.add("C1", "baseline author", PASS, "tim.zhao present")
    else:
        report.add(
            "C1",
            "baseline author",
            FAIL,
            f"no solution authored by tim.zhao (authors: {', '.join(sorted(sols_by_author))})",
        )

    # ---- C1 by delegation ---------------------------------------------------
    if not args.skip_subprocess:
        run_subprocess(
            "validate_task", [sys.executable, "tools/validate_task.py", str(task_dir)], report, "C1"
        )
        if not args.skip_dataset:
            root = args.root or (REPO / "data" / "trace_sets" / task_dir.name)
            if root.is_dir():
                run_subprocess(
                    "validate_dataset",
                    [sys.executable, "tools/validate_dataset.py", "--root", str(root)],
                    report,
                    "C1",
                )
            else:
                report.add("C1", "validate_dataset", SKIP, f"no staged root at {root}")
    else:
        report.add("C1", "validate_task / validate_dataset", SKIP, "--skip-subprocess")

    # ---- C2: the generated solutions are what the gate must scrutinise ------
    for path, sol in sorted(solutions, key=lambda p: p[1].author):
        for check in (
            check_no_reference_import,
            check_no_identity_branch,
            check_output_derivation,
            check_no_host_sync,
        ):
            try:
                verdict, msg = check(sol, definition)
            except Exception as exc:  # noqa: BLE001
                verdict, msg = FAIL, f"check raised: {exc}"
            report.add("C2", f"{check.__name__[6:]} ({sol.author})", verdict, msg)

    review_items(definition, report)

    # ---- C3 ----------------------------------------------------------------
    blobs = check_blobs(task_dir, workloads, report)
    check_tensor_semantics(definition, blobs, workloads, report)

    # ---- C4 ----------------------------------------------------------------
    check_trace_ready(task_dir, definition, workloads, report)

    # ---- verdict -----------------------------------------------------------
    report.emit(quiet_ok=args.quiet_ok)
    counts = report.counts()
    print(
        f"\n=== {counts[PASS]} checked, {counts[REVIEW]} review, "
        f"{counts[WARN]} warning, {counts[FAIL]} failed ==="
    )
    if counts[REVIEW]:
        print(
            "REVIEW items are open questions, not passes. `docs/contracts.md` says which\n"
            "ones a human has to sign off, and `docs/architecture.md` 4 tracks the gap."
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "task": task_dir.name,
                    "counts": counts,
                    "facts": report.facts,
                    "findings": [f.__dict__ for f in report.findings],
                },
                indent=2,
                default=str,
            )
        )
        print(f"report written to {args.json}")

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
