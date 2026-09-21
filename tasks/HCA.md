# HCA task — implementation plan

Plan for building a DeepSeek-V4 **HCA** (hierarchical compressed attention,
`compress_ratio = 128`, known as **C128A** throughout vLLM) kernel task that
conforms to the flashinfer-bench format vendored at
`third_party/flashinfer-bench/`.

This document is the handoff for the GPU machine. Steps 1–4 are offline and can
be done without a GPU; step 6 is the hard blocker for measurement.

---

## 0. Boundary: the task is the compressor, not the attention loop

The vLLM call chain is:

```
DeepseekV4Attention.forward  (vllm/models/deepseek_v4/attention.py:464)
  -> forward_mqa
     -> flash_mla_with_kvcache        (decode)
     -> flash_mla_sparse_fwd          (prefill)
```

**HCA has no dedicated attention kernel.** It shares the MLA sparse backends
(FlashMLA / FlashInfer / AITER / XPU) with CSA. The single CSA/HCA fork in
`attention.py:592-616` is whether `self.indexer is None` — CSA (`compress_ratio
= 4`) has an indexer, HCA (`compress_ratio = 128`) does not.

Two pieces are HCA-specific and written in vLLM itself:

| Candidate | Location | Verdict |
|---|---|---|
| **Compressor**: compress → RMSNorm → RoPE → FP8 quant → KV cache write | `common/ops/save_partial_states.py` (`_SAVE_PARTIAL_STATES_KERNEL`, fuses `score += ape[position % 128]`) plus the fused compress/norm/rope/quant/store path in `compressor.py` | **Build this first.** Clean math, has an independent reference implementation, natural `const` axes, real fusion headroom. |
| **C128A index construction** | `sparse_mla.py`, `_BUILD_C128A_TOPK_METADATA_KERNEL` | Defer or drop. It fills a contiguous range (`tl.where(offset < num_compressed, offset, -1)`) after a `block_table` lookup and aligns to `_C128A_TOPK_ALIGNMENT = 128`. Almost pure integer index movement, low arithmetic intensity. |

Rationale for the ordering: the compressor is the only HCA-specific code with
both (a) a mathematical specification clean enough to write an authoritative
`reference` for, and (b) genuine optimization room from fusing four stages.

---

## 1. Choose the op_type

The vendored suite has 35 Definitions across 8 op types: `gemm` ×8,
`gqa_paged` ×4, `gqa_ragged` ×2, `mla_paged` ×2, `rmsnorm` ×6,
`fused_add_rmsnorm` ×3, `sampling` ×9, `moe` ×1. Note that `dsa_paged` already
has a spec and evaluators but **no baselines entry** — the extension point
exists.

Two options:

- **New op_type `hca_compress`** — recommended. The compressor is semantically
  its own thing; forcing it under `mla_paged` makes that op-type's spec
  incoherent.
- Extend `mla_paged` — only worth it to reuse its evaluator, and see step 5 for
  why that is unlikely to be the deciding factor.

Either way, author `docs/op-types/hca-compress.mdx` following the pattern of
`docs/op-types/mla-paged.mdx` and `docs/op-types/dsa-paged.mdx`, the two
closest precedents.

**Constraint:** the vendored tree is read-only reference. Never write the HCA
Definition into `third_party/flashinfer-bench/` — see `third_party/VENDOR.md`,
which fixes the rule that a vendored tree is replaced wholesale on a pin bump,
never patched in place.

---

## 2. Write the Definition

Schema authority: `docs/flashinfer-trace/definition.mdx`; parser:
`flashinfer_bench/data/definition.py`.

A `Definition` carries `name`, `op_type`, `tags`, `description`, `axes`,
`inputs`, `outputs`, `reference`, `constraints`.

### axes

Keep `var` axes few and `const` axes many. The natural HCA split:

- `var`: `total_tokens` (or `num_compressed_groups`)
- `const`: `compress_rate = 128`, `head_dim = 512`, `rope_head_dim = 64`,
  `window = 128`, `page_size`

`compress_rate` **must** be `const`. 4 is CSA, 128 is HCA — that value changing
means a different kernel, and putting it on a `var` axis only pollutes the
problem. The vendored GEMM example does exactly this: `N`/`K` are locked as
`const` and only `M` varies.

Axis objects are `{"type": "const", "value": <int>}` or `{"type": "var"}`.

### inputs / outputs

`TensorSpec` is `{shape: list[str] | None, dtype}`. Axis names appear as shape
entries. Supported dtypes include `float32`, `float16`, `bfloat16`,
`float8_e4m3fn`, `float8_e5m2`, `float4_e2m1`, `int64/32/16/8`, `bool`.

### reference

This is the heaviest part of the Definition: it is the **official mathematical
specification**, not merely "a PyTorch implementation that runs". Two hard
requirements:

1. **No package-level `torch.nn.functional` calls.** Write the computation out
   in explicit steps. `Definition._validate_reference_code` enforces this.
2. **Implement the transformers semantics exactly:**

   ```
   C_i = kv_norm( Σ_{j ∈ window} softmax(gate_j + position_bias)_j ⊙ kv_j )
   ```

   RoPE is applied to the compressed entries at absolute position
   `i * compress_rate + first_window_position`. The causal threshold is
   `causal_threshold = (position_ids + 1) // compress_rate`.

### ⚠️ RoPE convention must be pinned before anything else

The two references disagree:

- **vLLM** applies GPT-J style rotation (`is_neox_style = False`, interleaved
  pairs, **not** split-half), to the **last** `rope_head_dim` elements of
  `head_dim`, using position `(positions // compress_ratio) * compress_ratio`.
- **HF transformers** `rotate_half` is split-half.

If this is not resolved, no candidate can pass the `reference` and the task is
dead on arrival. Pin it explicitly either by writing the convention into
`constraints` as an explicit expression, or by implementing one of the two in
`reference` and stating the choice in `description`.

### constraints

A list of string expressions, following `dsa-paged.mdx` (which pins
`num_index_heads == 64`, `head_dim_with_scale == 132`, and similar).

---

## 3. Design the Workload sweep

`workloads/<name>.jsonl`, one `Workload` per line. Schema:
`docs/flashinfer-trace/workload.mdx`. Each line carries a `uuid`, concrete
integer bindings for every `var` axis, and input descriptors.

Mirror the 43-point sweep in
`examples/ffi/Example-FlashInfer-Trace/workloads/gemm_n4096_k4096.jsonl`: a
regular run over the head of the range, then a tail of deliberately awkward
values. That example sweeps `M` from 256 down to 1 in steps of 8, then appends
7, 35, 15, 70, 972, 2053, 2379, 8192.

For HCA the critical irregularity is **`total_tokens` not divisible by
`compress_rate` (128)** — a partial trailing block is where real
implementations break, so it must appear in the sweep.

Input descriptors: `{"type": "random"}` is sufficient for most entries.
`scalar` (with `value`) and `safetensors` (with `path` + `tensor_key`) are for
injecting fixed scalars or real weights.

---

## 4. Write a Solution

Schema: `docs/flashinfer-trace/solution.mdx`. Fields: `name`, `definition`,
`description`, `author`, `spec`, `sources`.

`spec` carries `language ∈ {python, triton, cpp, cuda}`, `target_hardware`,
`entry_point: "{file}::{func}"`, `destination_passing_style`, `binding`,
`dependencies`. `sources[]` is a list of `{path, content}`.

**Ship a `language: "python"` Solution first.** Its value is proving that the
specification is executable and that the correctness thresholds are sane, and
it costs no GPU time. The Triton/CUDA solution that the task actually exists to
elicit comes after.

---

## 5. Decide whether a custom evaluator is needed

`flashinfer_bench/bench/evaluators/registry.py` defines:

```python
_EVALUATORS = [SamplingEvaluator, LowBitEvaluator,
               DsaSparseAttentionEvaluator, DsaTopkIndexerEvaluator]
_DEFAULT_EVALUATOR = DefaultEvaluator
```

`resolve_evaluator(definition)` calls `can_evaluate(definition)` on each in
turn and **raises if more than one matches**. So if the new Definition's
`op_type` and `tags` do not accidentally trip an existing `can_evaluate`,
`DefaultEvaluator` is available as-is.

`DefaultEvaluator` runs `build_baseline` → `check_correctness` →
`eval_performance`. **Assume it is sufficient**; write a custom evaluator only
if its checks provably do not cover the case.

Two scoring semantics to internalize:

- Correctness is graded on **`max_relative_error` and `max_absolute_error`**,
  not on an allclose boolean. The step-2 RoPE decision surfaces here first.
- The denominator `reference_latency_ms` is the **unoptimized PyTorch `run`**,
  so `speedup_factor = ref_mean_latency_ms / sol_mean_latency_ms` is large by
  construction. **This is not KernelBench's `fast_p` hardware-relative
  ceiling** — do not reason about thresholds as if it were.

---

## 6. GPU path

This host has no GPU: no `/dev/nvidia*`, no `nvidia-smi`, no `nvcc`, and no
torch or triton installed. All correctness and performance measurement must
happen on a GPU machine or through Modal.

`modal` 1.5.5 is installed but **unauthenticated on this host**. Authenticate
(`modal setup`, or provide `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) before the
first Trace is produced.

Steps 2–5 advance offline. Step 6 blocks only the Trace.

---

## 7. Where the HCA task lives in this repository

The top-level `src/`, `tasks/`, and `tests/` directories are currently empty.
Settle before writing code:

- Do Definition and Workload files live under `tasks/<task_id>/,` or are they
  generated artifacts under `data/` (which `.gitignore` already excludes)?
- The vendored `third_party/flashinfer-bench/` tree is **read-only reference**,
  so the HCA files cannot go there.

---

## 8. Shortest viable path

1. Steps 1–3 — offline, plain files.
2. Step 4 — the python Solution, offline.
3. Step 6 — authenticate Modal (or land on the GPU machine).
4. Produce the first Trace.
5. Step 5 — decide on a custom evaluator from what the Trace actually shows.
6. Add the Triton/CUDA Solution.

---

## Appendix: reference facts

### flashinfer-bench objects

| Object | Role |
|---|---|
| **Definition** | `op_type`, `axes`, tensor specs, and a plain-PyTorch `run` string as the mathematical specification |
| **Workload** | concrete values for the `var` axes plus input data descriptors |
| **Solution** | `sources[]`, `spec.language`, `entry_point`, `destination_passing_style` |
| **Trace** | `status ∈ {PASSED, INCORRECT_SHAPE, INCORRECT_NUMERICAL, INCORRECT_DTYPE, RUNTIME_ERROR, COMPILE_ERROR}`, max rel/abs error, `latency_ms`, `reference_latency_ms`, `speedup_factor`, environment snapshot |

### HCA geometry (from the MLA/HCA parameterization)

| Symbol | Value |
|---|---|
| `kv_lora_rank` / `head_dim_ckv` | 512 |
| `qk_rope_head_dim` / `head_dim_kpe` | 64 |
| `qk_nope_head_dim` | 128 |
| `v_head_dim` | 128 |
| `num_kv_heads` | 1 (MQA-style shared) |
| compress rate | 128 |
| sliding window | 128 |

### vLLM compressor facts

- `DeepseekCompressor.forward` is at `vllm/models/deepseek_v4/compressor.py:353`;
  the class is at `:199`.
- `self.overlap = compress_ratio == 4` and `self.coff = 1 + self.overlap`
  (`:238-239`), so HCA has `coff = 1` and CSA has `coff = 2`. The split is
  `kv, score = kv_score.split([coff * head_dim, coff * head_dim], dim=-1)`.
- There is a CUDA early-return guard that fires only for C128A:
  ```python
  if (current_platform.is_cuda()
      and self.head_dim == 512
      and self.compress_ratio == 128
      and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
      and state_metadata.c128_boundary is False):
      return
  ```
- PDL must stay disabled for `_SAVE_PARTIAL_STATES_KERNEL`: the in-tree comment
  records that `launch_pdl=True` caused a read-after-write race and
  non-deterministic output. Any solution that re-enables it reintroduces the
  race.

### C128A index construction (deferred candidate)

- `_C128A_TOPK_ALIGNMENT = 128`; the comment on it reads "FlashMLA decode
  asserts extra_topk % B_TOPK == 0".
- `_build_c128a_metadata` computes
  `active_topk_width = min(max(triton.next_power_of_2(max(cm.max_seq_len // self.compress_ratio, 1)), _C128A_TOPK_ALIGNMENT), self.c128a_max_compressed)`.
- `build_c128a_topk_metadata` docstring: "Single kernel for all C128A tokens
  (decode + prefill). Decode tokens: position → block_table lookup → global
  slot ids + topk_lens. Prefill tokens: position → local indices
  [0, ..., n-1, -1, ...]."
- Consumed by `nvidia/flashmla.py` through
  `attn_metadata.c128a_global_decode_topk_indices`,
  `c128a_decode_topk_lens`, and `c128a_prefill_topk_indices`.
