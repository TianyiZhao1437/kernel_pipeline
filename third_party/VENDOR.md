# Upstream vendoring

`KernelBench/` under this directory is a copy of the upstream KernelBench
repository.

| | |
|---|---|
| Source | `https://github.com/ScalingIntelligence/KernelBench` |
| Commit | `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` |
| Commit date | 2026-03-05 |
| Subject | update all legacy python commands to UV + document integration (#143) |
| Path copied | repository root |
| Tasks | 270 problem files across four levels |

## What upstream provides

Not a dataset alone. The repository is both halves of a working pipeline:

| Half | Path | Contents |
|---|---|---|
| Dataset | `KernelBench/level1..4/` | 100 / 100 / 50 / 20 problems |
| Framework | `src/kernelbench/` | `eval.py`, `timing.py`, `score.py`, `kernel_static_checker.py`, `compile.py`, `dataset.py`, `profile.py` |
| Entry points | `scripts/` | `run_and_check.py` (single problem, `local` or `modal`), `generate_samples.py`, `eval_from_generations.py`, `benchmark_eval_analysis.py`, `generate_baseline_time.py` |
| Unit tests | `src/kernelbench/unit_tests/` | Includes `test_kernels/` with three real reward-hacking specimens |

The task contract is five symbols: `Model`, a module-level shape definition,
`get_inputs()`, `get_init_inputs()`, and the candidate's `ModelNew`.

## Why a copy and not a submodule

The task format and the evaluation protocol are the binding specification for
what this pipeline emits, and upstream changes both. A submodule tracks whatever
upstream last pushed, so a task that reproduced on Monday can fail to reproduce
on Tuesday without anything in this repository changing. A pinned copy makes the
reference a fact of this repository: results are reproducible, and adopting a
newer upstream is an explicit, reviewable commit.

## Local modifications

One, to the vendored `.gitignore`: upstream's final line is `CLAUDE.md`, which
hides an agent instructions file from version control. That line is removed so
the vendored tree is committed in full. No other file is modified.

`results/timing/` is kept. It carries upstream's measured PyTorch baselines for
H100 (Modal and PCIe/Lambda Labs), which are the denominators the speedup metric
is relative to — without them a locally measured kernel has nothing to be
compared against.

## Updating the pin

1. Clone upstream at the new commit into a temporary directory.
2. Replace the tree wholesale — never patch a vendored file in place.
3. Update the table above.
4. Re-run the test suite; a changed result against the reference problems is the
   signal that the format moved.
