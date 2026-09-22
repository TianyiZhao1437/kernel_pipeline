# The contract catalogue

Four contracts, each a property a task either has or does not have. This file is
the authority for what they assert and how they are checked; `architecture.md` §2.2
is the summary.

A contract is written as an **assertion plus its counterexample** — the specific
plausible-looking thing that would violate it. Every counterexample below is one
this repository has actually produced, because a contract whose failure mode is
hypothetical does not get written correctly.

Notation: **[auto]** = mechanisable, becomes a check in `tools/verify_task.py`;
**[review]** = requires human sign-off, pipeline can only assemble the evidence.

---

## C1 — Schema: the task parses and binds

**Asserts.** The Definition, every Workload, and every Solution load under the
real `flashinfer_bench` pydantic models; input and output shapes resolve for
every workload; the reference and each Solution's entry take exactly the number
of positional arguments the builder will bind; the dataset validates against the
vendored validator from a correctly staged root.

**Checked by.** `tools/validate_task.py` (schema + arity + shape resolution, via
`fib_shim`) and `tools/validate_dataset.py` (the vendored validator).

**Counterexamples, all real — `HCA.md` §2, "four schema constraints that are not
in the docs":**

| # | The violation | Why it looks fine |
|---|---|---|
| 1 | `AxisConst.value` must be an **integer**; `rms_norm_eps = 1e-6` as a const axis is rejected | an epsilon is a natural constant to parameterise |
| 2 | **No unsigned dtypes** exist, so UE8M0's on-wire `uint8` is stored as the unbiased exponent in `int8` | `uint8` is what vLLM actually uses |
| 3 | Shape entries must be declared axis **names**, never expressions — `total_tokens // compress_rate` is rejected | the expression is the clearest way to say it |
| 4 | The reference `run` receives **only input tensors, no axes**; a draft taking nine axis params fails at build time | the reference needs `compress_rate` to be correct |

**Trap inside this contract.** The vendored dataset validator *discovers* the
dataset from path depth and **silently skips** files at the wrong depth, so a
hand-staged root reports "0 definitions" rather than an error. Staging must go
through `tools/stage_trace_set.py`, which writes the strict layout. A green
report from a bad root is the failure mode.

---

## C2 — Semantics: it computes the stated op, without a shortcut

**Asserts.** The reference implements the operation the Definition describes, and
it is *reconciled* against an executable implementation rather than a
description. Solutions derive their outputs from their inputs; no solution reads
the reference's output, aliases a constant, or branches on workload identity.

**Checked by.** `[review]` for the reconciliation; `[auto]` for the anti-hack
checks — **which do not exist in this repository in any form today.** This is the
largest concrete gap in the pipeline.

**Why the review half cannot be automated.** `HCA.md` §8 item 9 is exactly this
contract, still open: *"Reconcile the Definition against the now-available native
`DeepseekV4HCACompressor` line by line. Every convention checked so far agrees,
but the reference was written from vLLM and has not been diffed against the
library implementation in full."* No check can answer "does this match the
semantics the library implements" — only a diff against the library can.

**The convention trap this contract exists for.** `HCA.md` §2.3 records that
RoPE's indexing convention was **resolved, not assumed**: entry `c` reads row
`c * compress_rate`, the window's boundary token floored to the window start,
matching vLLM's `(positions // compress_ratio) * compress_ratio`. A one-row-off
choice here produces a kernel that is *entirely plausible*, runs at full speed,
and returns wrong numbers — and every other contract passes it.

**Anti-hack checks worth implementing [auto]:**

1. **Output derivation** — perturb an input, confirm the output changes. Catches
   constants, cached answers, and reference-aliasing.
2. **No reference call** — the reference module must not be imported by a
   Solution.
3. **No workload-identity branch** — the entry must not read a workload UUID,
   shape-derived special case, or the axes dict to select an implementation.
4. **Timing sanity** — a solution faster than the memory roofline for its
   declared byte count is not fast, it is not doing the work. This is the one
   check that catches a *cheating* kernel rather than a *wrong* one, and it falls
   straight out of `tools/roofline.py`.

---

## C3 — Inputs: every input is physically realisable

**Asserts.** For each declared input, either it is backed by a pinned, hashed
blob (`SafetensorsInput`) whose values are consistent with the op, or the task
**declares** the input distribution-insensitive and a reviewer signs that
declaration.

**Checked by.** `[auto]` for blob presence and hash (`blobs.sha256`), and for
internal consistency of constrained values; `[review]` for the declaration.

**The counterexample — `HCA.md` §7.7, and this is the strongest one in the
repository.** Every workload declared `{"type": "random"}`, and `RandomInput`
**has no parameters at all** — not a distribution, not a seed, not a scale. So
"random" means exactly `torch.randn`, and that made:

- `cos_cache`/`sin_cache` **not a rotation table**. Real entries satisfy
  `cos² + sin² = 1` and `|.| ≤ 1`; `N(0,1)` rows reach ±5 and miss the unit
  circle by ~30, inflating the RoPE columns of `ckv`.
- `rms_norm_weight` **half negative with median near zero**, against a real
  tensor that is 503/512 positive and spans 190x. This one matters most: RMSNorm
  divides out the input scale, so this tensor is what sets the per-channel
  magnitudes the FP8 block quantiser has to cover.
- `kv_state`/`score_state` **light-tailed**. Measured on real activations the
  hidden state has **kurtosis 405** and the MLA KV latent **245**, against 3.0
  for a Gaussian — the "massive activation" channels that are the entire reason
  UE8M0 carries a scale per 64 elements rather than per tensor.

The consequence was that over the whole sweep **the UE8M0 exponent only ever
landed in `[-8, -4]` of a representable `[-22, 120]`** — the quantiser's dynamic
range was almost entirely untested and `clamp(min=1e-4)` was never reached.

**Consistency checks worth automating [auto]** are the ones that need no model:
unit-circle for a rotation table, sign/magnitude profile for a norm gain, and —
the one that generalises — **does the sweep reach the op's bound-handling
branches?** For HCA that is the `clamp(min=1e-4)` floor, which the `tiny` stress
workload now drives 98.66% of blocks under. A sweep that never reaches a clamp,
a saturation, or a denormal is not testing the op.

**This contract is independent of the resource question.** "Run the real model"
is *one* way to satisfy C3; the repository's actual answer for HCA is better:
probe the published checkpoint for the four small tensors that are **exact**
(`norm.weight`, `DeepseekV4RotaryEmbedding`, `wgate`+`ape`, `wkv`) and synthesise
only the hidden state, lifting real per-channel marginals from a **cheaper**
model (`DeepSeek-V2-Lite`, 16B) instead of executing V4-Flash (148 GiB) or
V4-Pro (805 GiB). Same fidelity class, two orders of magnitude cheaper.

---

## C4 — Trace-ready: reproducible from pinned sources on this target

**Asserts.** The task can be re-derived byte-for-byte from pinned sources; it
runs on the declared target hardware; `eval_config.yaml` is present and its
tolerances are the ones the run will actually use; and generation and measurement
use the same workload corpus.

**Checked by.** `[auto]`, mostly. New checks; the pieces exist.

**Counterexamples, all real:**

1. **Unpinned weights.** `model_probe.py` pins `MODEL_REVISION` because "a hub
   repo can be force-pushed or re-quantised in place". Every number in the
   repository derives from those bytes. Unpinned, the task is a dependency on a
   mutable third-party artifact *and* its measurements are not reproducible.
2. **Missing `eval_config.yaml`.** `run_benchmark.py`: `BenchmarkConfig.default()`
   bundles `required_matched_ratio: 1.0` — bitwise equality — which no kernel
   over a 128-token softmax can satisfy. Omitting the config yields
   `INCORRECT_NUMERICAL` for a **correct** solution. It is not a lenient default,
   it is a wrong verdict.
3. **Divergent generation and measurement roots.** `gen_solution_llm.py` carries a
   long comment on this: it was pinned at the old 20-workload `hca_c128_v4`, and
   if the model optimises against a smaller sweep than the trace is judged on,
   "the round-by-round feedback the model optimises against is a different — and
   smaller — sweep than the trace it is finally judged on". Both defaults now
   agree, and **nothing asserts it**.
4. **The `benchmark` check's hardcoded config.** `HCA.md` §8 item 8 records that
   the vendored validator's benchmark check carries its own `BenchmarkConfig`, so
   its verdict can disagree with the real run. Either reconcile it or document
   the divergence; do not let two configs disagree silently.
5. **Target hardware mismatch.** Per `CLAUDE.md` §12 the wheel constraint is set
   by **compute capability**, not driver version, and the driver is host-injected
   and must not be replaced. A task whose reference needs kernels the declared
   target cannot execute does not fail loudly — it dies at the first GPU op with
   `no kernel image is available for execution on the device`.

**The `task.yaml` this contract implies:**

```yaml
task: hca_compress_c128
op_type: hca_compress
target_hardware: NVIDIA_H200        # detected at intake, asserted here
measured_peak_gbps: 4218            # d2d copy; NOT the 4800 spec figure
axes: [num_compressed, max_position]
eval_config: eval_config.yaml
sources:
  definition: vllm@<sha>            # what the reference was written from
  reconcile_against: transformers@<sha># DeepseekV4HCACompressor
  weights: {repo: deepseek-ai/DeepSeek-V4-Flash, revision: 60d8d707...}
blob_manifest: blobs.sha256
generation_root: data/trace_sets/hca_compress_c128   # asserted == measurement root
measurement_root: data/trace_sets/hca_compress_c128
```

---

## How the contracts map to the plan's four groups

The plan asks for a baseline that satisfies "四组contract". C1–C4 are those four,
made checkable. The mapping to the plan's own wording:

| Plan's wording | Contract |
|---|---|
| 完整 (complete) | C1 — parses, binds, validates |
| 合法 (legal) | C1 + C2 |
| 无hack (no hack) | C2 — anti-hack checks |
| 符合实际工程语义 (real engineering semantics) | C2 (reconciliation) + C3 (real inputs) |
| ready for trace | C4 |

"无hack" is C2's automated half and is the piece that does not exist yet.
"符合实际工程语义" is split across C2's review half and C3, and the pipeline's
job is to put the open question in front of a reviewer with the evidence
attached — not to report a check count that implies it has been answered.
