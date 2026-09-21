#!/usr/bin/env python3
"""Validate the artefacts under tasks/ against the vendored flashinfer-bench parsers.

Checks the schema with the real pydantic models, plus three things those models
do not: that a Workload's input descriptors match the Definition's inputs, that
both input and OUTPUT shapes resolve (the harness derives output shapes from
input shapes alone), and that the reference and every Solution take the number
of positional arguments the builder will actually bind.

See tools/fib_shim.py for why the parsers are imported the way they are.

Usage:
    python3 tools/validate_task.py tasks/hca_compress_c128
"""

import argparse
import ast
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fib_shim  # noqa: E402  (needs the path above)

REPO = fib_shim.REPO


def check_entry_signature(sol, defn):
    """Cross-check the entry function's arity against the Definition.

    compile/builder.py binds exactly ``len(inputs)`` positional arguments, plus
    ``len(outputs)`` more when destination_passing_style is set, and passes no
    axes. A mismatch is a BuildError at benchmark time, long after authoring, so
    catch it here. Only checked for python/triton, where the entry is a plain
    Python def.
    """
    if sol.spec.language.value not in ("python", "triton"):
        return
    entry = sol.get_entry_source()
    check_arity(
        entry.content,
        sol.get_entry_symbol(),
        len(defn.inputs),
        len(defn.outputs) if sol.spec.destination_passing_style else 0,
        entry.path,
    )


def check_arity(source, symbol, n_inputs, n_outputs, where):
    tree = ast.parse(source)
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol
        ),
        None,
    )
    if fn is None:
        raise ValueError(f"symbol '{symbol}' is not a module-level def in {where}")
    got = len(fn.args.posonlyargs) + len(fn.args.args)
    want = n_inputs + n_outputs
    if got != want:
        raise ValueError(
            f"'{symbol}' in {where} takes {got} positional args, but the harness binds "
            f"{want} ({n_inputs} inputs"
            + (f" + {n_outputs} outputs, destination-passing" if n_outputs else "")
            + "). Axes are NOT passed."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=pathlib.Path)
    args = ap.parse_args()

    data = fib_shim.load_data()
    task = args.task_dir
    failures = 0
    definitions = {}
    solutions = []

    for path in sorted(task.glob("*.json")):
        obj = json.loads(path.read_text())
        model = data.Solution if "spec" in obj else data.Definition
        try:
            parsed = model.model_validate(obj)
            print(f"  ok    {model.__name__:<10} {path.name}  ({parsed.name})")
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            failures += 1
            print(f"  FAIL  {model.__name__:<10} {path.name}\n{exc}")
            continue

        if model is data.Definition:
            definitions[parsed.name] = parsed
            # registry.py wraps `reference` as a pseudo-Solution with
            # destination_passing_style=False and entry main.py::run, so it is
            # bound by the same rule as a real solution.
            try:
                check_arity(parsed.reference, "run", len(parsed.inputs), 0, f"{path.name}:reference")
                print(f"  ok    reference  {path.name}  (run takes {len(parsed.inputs)} inputs)")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  reference  {path.name}\n{exc}")
        else:
            solutions.append((path, parsed))

    # Solutions are checked after the loop so every Definition in the directory
    # is known regardless of filename order.
    for path, sol in solutions:
        defn = definitions.get(sol.definition)
        try:
            if defn is None:
                raise ValueError(
                    f"references definition '{sol.definition}', which is not in {task}"
                )
            check_entry_signature(sol, defn)
            print(f"  ok    entry      {path.name}  ({sol.spec.entry_point})")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  entry      {path.name}\n{exc}")

    for path in sorted(task.glob("*.jsonl")):
        ok = bad = 0
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                wl = data.Workload.model_validate(rec["workload"])
                # The Workload model does not cross-check against a Definition,
                # so do it here: every var axis must be bound, every input must
                # be described, and -- the one that actually bites -- the output
                # shapes must resolve, since the harness derives them from input
                # shapes alone (evaluators/utils.py::allocate_outputs).
                defn = definitions.get(rec.get("definition"))
                if defn is not None:
                    missing = set(defn.inputs) - set(wl.inputs)
                    extra = set(wl.inputs) - set(defn.inputs)
                    if missing or extra:
                        raise ValueError(
                            f"input descriptors mismatch: missing={sorted(missing)} "
                            f"unexpected={sorted(extra)}"
                        )
                    defn.get_input_shapes(wl.axes)
                    defn.get_output_shapes(wl.axes)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                bad += 1
                if bad <= 3:
                    print(f"  FAIL  Workload   {path.name}:{lineno}\n{exc}")
        failures += bad
        print(f"  {'ok  ' if not bad else 'FAIL'}  Workload   {path.name}  ({ok} valid, {bad} rejected)")

    print("\nOK" if not failures else f"\n{failures} artefact(s) rejected")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
