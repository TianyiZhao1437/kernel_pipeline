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
uv pip install --no-deps -e third_party/flashinfer-bench
```

That editable link is the whole install: it binds the importable name
`flashinfer_bench` and the `flashinfer-bench` CLI to the vendored copy, with
`third_party/patches/` live. Nothing under `tools/` needs a `PYTHONPATH` prefix
or a `sys.path` insert to find it, and neither do the benchmark runner's worker
subprocesses.

`--no-deps` is not optional — see the comment in `pyproject.toml` for why
(a free resolver can satisfy `torch>=2.8.0` by replacing this environment's
CUDA-matched torch pin, and the failure surfaces much later as "no kernel image
is available for execution on the device").

## Vendoring policy

A vendored tree is a **copy pinned to a commit**, never a git submodule and
never a live checkout.

Upstream changes; the format we emit does not. A submodule tracks whatever
upstream last pushed, so a task that reproduced on Monday can fail to reproduce
on Tuesday without anything in this repository changing. A pinned copy makes
the reference format a fact of this repository: results are reproducible, and
moving to a newer upstream is an explicit, reviewable commit.

Local fixes to a vendored tree are legitimate, but they live in
`third_party/patches/` as a replayable series — never as an in-place edit that
the next pin bump would silently revert.

## Updating a pin

The procedure is in `third_party/VENDOR.md`, which is the authority. In short:
clone upstream at the new commit, replace the tree wholesale, verify nothing was
silently excluded, replay `third_party/patches/apply.sh`, update the pin table,
and re-run the suite — a changed result is the signal that the format moved.
