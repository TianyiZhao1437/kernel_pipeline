# kernel_pipeline

Pipeline for producing GPU-kernel engineering tasks for RL/SFT training.

KernelBench is the reference format: a task hands an agent a reference PyTorch
implementation and asks for a faster equivalent, graded on functional
correctness plus a measured performance ratio against a hardware-relative
ceiling.

## Repository layout

| Path | Contents |
|---|---|
| `third_party/` | Pinned upstream sources. Never edited in place — see `third_party/VENDOR.md`. |

## Vendoring policy

A vendored tree is a **copy pinned to a commit**, never a git submodule and
never a live checkout.

Upstream changes; the format we emit does not. A submodule tracks whatever
upstream last pushed, so a task that reproduced on Monday can fail to reproduce
on Tuesday without anything in this repository changing. A pinned copy makes
the reference format a fact of this repository: results are reproducible, and
moving to a newer upstream is an explicit, reviewable commit.

## Updating a pin

1. Clone upstream at the new commit into a temporary directory.
2. Replace the vendored tree wholesale — never patch a vendored file in place.
3. Update the table in `third_party/VENDOR.md`.
4. Re-run the test suite; any change in measured behavior is the signal that the
   format moved.
