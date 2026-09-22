# kernel_pipeline

Pipeline for producing GPU-kernel engineering tasks for RL/SFT training.

KernelBench is the reference format: a task hands an agent a reference PyTorch
implementation and asks for a faster equivalent, graded on functional
correctness plus a measured performance ratio against a hardware-relative
ceiling.

## Repository layout

| Path | Contents |
|---|---|
| `tasks/` | Authored tasks. Each is a staged TraceSet source: Definition, Solutions, Workloads, `eval_config.yaml`. |
| `tools/` | Staging, benchmarking, validation, and roofline analysis over those tasks. |
| `third_party/` | Pinned upstream sources. Never edited in place — see `third_party/VENDOR.md`. |
| `third_party/patches/` | The local patch series against the vendored flashinfer-bench, replayed by `apply.sh`. |

## Installing

```
uv pip install --no-deps -r pyproject.toml
```

`flashinfer-bench` is **not** resolved from PyPI and is **not** installed from
the vendored tree. `pyproject.toml` binds the name to this project's fork,
pinned to a commit:

```toml
[tool.uv.sources]
flashinfer-bench = { git = "https://github.com/TianyiZhao1437/flashinfer-bench.git", rev = "19acd0df..." }
```

So the install is one fetch of the fork at that revision, and the local changes
land as an ordinary install — no patch step, no editable link, no `PYTHONPATH`
prefix, and nothing for the benchmark runner's worker subprocesses to lose when
they re-import in a fresh interpreter.

`--no-deps` is not optional. This environment pins torch 2.11.0+cu128 and
triton 3.6.0 to match the host driver and the H200's compute capability; a
resolver given a free hand can satisfy `torch>=2.8.0` by replacing that pin with
a CPU or wrong-CUDA wheel, and the failure surfaces much later as "no kernel
image is available for execution on the device". With `--no-deps`, every one of
flashinfer-bench's dependencies is already satisfied and the resolver installs
exactly the one package.

Bumping the pin is an edit to `rev` in `pyproject.toml` plus the table in
`third_party/VENDOR.md`. The pinned revision, not the fork branch, is what is
bound — a branch would reintroduce the drift the vendoring policy rules out.

### Working on flashinfer-bench itself

Clone the fork outside this repository and edit there, then push and bump the
pin:

```
git clone https://github.com/TianyiZhao1437/flashinfer-bench /path/to/checkout
```

The vendored tree under `third_party/flashinfer-bench/` is **not** what the
interpreter imports. It is kept as the reviewable reference for the pinned
commit: `git diff` against it shows exactly what the fork adds on top of
upstream, and `third_party/patches/` replays that difference onto a fresh
upstream checkout when the pin moves.

## Vendoring policy

`third_party/` holds **copies pinned to a commit**, never git submodules and
never live checkouts.

Upstream changes; the format we emit does not. A submodule tracks whatever
upstream last pushed, so a task that reproduced on Monday can fail to reproduce
on Tuesday without anything in this repository changing. A pinned copy makes
the reference format a fact of this repository: results are reproducible, and
moving to a newer upstream is an explicit, reviewable commit.

Local fixes to a vendored tree are legitimate. They live in two places that are
kept in sync: committed on the fork's `hca-integration` branch (which is what
gets installed), and in `third_party/patches/` as a numbered, replayable series
(which is what makes them reviewable and what survives a pin bump). The patch
series is applied to the vendored tree, so the tree and the installed fork
agree; `third_party/VENDOR.md` records the invariant and how to check it.

## Updating a pin

The procedure is in `third_party/VENDOR.md`, which is the authority. In short:
clone upstream at the new commit, replace the tree wholesale, verify nothing was
silently excluded, replay `third_party/patches/apply.sh`, update the pin table,
and re-run the suite — a changed result is the signal that the format moved.
