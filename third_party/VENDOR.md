# Upstream vendoring

Two upstream trees live under this directory, both as pinned copies:

| Directory | Upstream | Commit | Commit date |
|---|---|---|---|
| `KernelBench/` | `https://github.com/ScalingIntelligence/KernelBench` | `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` | 2026-03-05 |
| `flashinfer-bench/` | `https://github.com/flashinfer-ai/flashinfer-bench` | `40e6ca7844b514eb4b1c7edba6d6a7377df57870` | 2026-04-30 |

---

## `KernelBench/`

### What upstream provides

Not a dataset alone. The repository is both halves of a working pipeline:

| Half | Path | Contents |
|---|---|---|
| Dataset | `KernelBench/level1..4/` | 100 / 100 / 50 / 20 problems |
| Framework | `src/kernelbench/` | `eval.py`, `timing.py`, `score.py`, `kernel_static_checker.py`, `compile.py`, `dataset.py`, `profile.py` |
| Entry points | `scripts/` | `run_and_check.py` (single problem, `local` or `modal`), `generate_samples.py`, `eval_from_generations.py`, `benchmark_eval_analysis.py`, `generate_baseline_time.py` |
| Unit tests | `src/kernelbench/unit_tests/` | Includes `test_kernels/` with three real reward-hacking specimens |

The task contract is five symbols: `Model`, a module-level shape definition,
`get_inputs()`, `get_init_inputs()`, and the candidate's `ModelNew`.

### Local modifications

One, to the vendored `.gitignore`: upstream's final line is `CLAUDE.md`, which
hides an agent instructions file from version control. That line is removed so
the vendored tree is committed in full. No other file is modified.

`results/timing/` is kept. It carries upstream's measured PyTorch baselines for
H100 (Modal and PCIe/Lambda Labs), which are the denominators the speedup metric
is relative to — without them a locally measured kernel has nothing to be
compared against.

---

## `flashinfer-bench/`

Task format: a **Definition** (op_type, axes, tensor specs, and a plain-PyTorch
`run` reference as the mathematical specification), a **Workload** (concrete
values for the variable axes plus input data descriptors), a **Solution**
(`sources[]`, `spec.language ∈ {python, triton, cpp, cuda}`, `entry_point`,
`destination_passing_style`), and a **Trace** (the evaluation record with
`status`, max relative/absolute error, `latency_ms`, `reference_latency_ms`,
`speedup_factor`, and an environment snapshot).

Schemas are authoritative in `docs/flashinfer-trace/{definition,workload,solution,trace}.mdx`.
Reference parsers live in `flashinfer_bench/data/`. Note that correctness is
graded on **max relative and max absolute error**, not on an allclose boolean,
and that the reference denominator is the unoptimized PyTorch `run` — so
`speedup_factor` is large by construction and is not a hardware-relative
ceiling the way KernelBench's `fast_p` is.

### Local modifications

Two:

1. **Symlinks preserved by hand.** Five upstream entries are symlinks
   (`web/apps/docs/content`, and four brand PNGs under
   `web/packages/ui/src/brand/`). The copy that populated this tree did not
   carry symlinks, so they were recreated with `ln -s` using upstream's exact
   targets and verified against `git ls-files -s`.
2. **`thirdparty/cutlass/` is an empty placeholder.** Upstream declares CUTLASS
   as a git submodule at
   `7817e47154d7869320f3fa6b409ec8c5e5958970`. This repository never vendors a
   submodule, so the tree is absent and
   `thirdparty/cutlass/README.md` records how to fetch it at the pinned commit.
   Nothing under `flashinfer_bench/` imports CUTLASS at module scope.

`web/` (a pnpm/Next.js monorepo, ~1.1 MB of TypeScript) and `.claude/` are kept
byte-for-byte even though no pipeline code reads them. Removing them would be a
second, silent deviation from upstream; keeping them means a tree diff against
the pin is meaningful.

---

## Why a copy and not a submodule

The task format and the evaluation protocol are the binding specification for
what this pipeline emits, and upstream changes both. A submodule tracks whatever
upstream last pushed, so a task that reproduced on Monday can fail to reproduce
on Tuesday without anything in this repository changing. A pinned copy makes the
reference a fact of this repository: results are reproducible, and adopting a
newer upstream is an explicit, reviewable commit.

## Updating a pin

1. Clone upstream at the new commit into a temporary directory.
2. Replace the tree wholesale — never patch a vendored file in place.
3. Recreate any symlinks the copy does not carry, and re-check `git ls-files -s`
   against the clone for mode `120000` entries.
4. Update the table above.
5. Re-run the test suite; a changed result against the reference problems is the
   signal that the format moved.
