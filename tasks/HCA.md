# HCA task — design record

The DeepSeek-V4 **HCA** compressor (hierarchical compressed attention,
`compress_ratio = 128`, called **C128A** throughout vLLM) as a flashinfer-bench
task, conforming to the format vendored at `third_party/flashinfer-bench/`.

This started as a forward-looking plan written without a GPU. It is now a record
of what was built, and — where the plan turned out to be wrong — of what the
code actually requires. Sections marked **[corrected]** contradict the original
plan; the correction is stated rather than silently applied, because several of
those errors are the kind that would be made again.

Artefacts:

| File | Role |
|---|---|
| `tasks/hca_compress_c128/hca_compress_c128_h512_r64.json` | Definition |
| `tasks/hca_compress_c128/hca_compress_c128_h512_r64.jsonl` | 17 Workloads |
| `tasks/hca_compress_c128/hca_compress_c128_triton_h200.solution.json` | Triton Solution (generated) |
| `tasks/hca_compress_c128/solutions/triton_h200/` | Solution sources, authoritative |
| `tasks/hca_compress_c128/eval_config.yaml` | Measured tolerances, with derivation |
| `tools/build_hca_task.py` | Generates the Definition + Workloads |
| `tools/build_solution.py` | Wraps a source dir into a Solution JSON |
| `tools/validate_task.py` | Schema + cross-checks against the vendored parsers |
| `tools/check_hca_numerics.py` | Reference/solution/tolerance validation |

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
| **Compressor**: compress → RMSNorm → RoPE → FP8 quant → KV cache write | `common/ops/save_partial_states.py` (`_SAVE_PARTIAL_STATES_KERNEL`, fuses `score += ape[position % 128]`) plus the fused compress/norm/rope/quant/store path in `compressor.py` | **Built.** Clean math, an independent reference implementation to check against, natural `const` axes, real fusion headroom. |
| **C128A index construction** | `sparse_mla.py`, `_BUILD_C128A_TOPK_METADATA_KERNEL` | Deferred. It fills a contiguous range (`tl.where(offset < num_compressed, offset, -1)`) after a `block_table` lookup and aligns to `_C128A_TOPK_ALIGNMENT = 128`. Almost pure integer index movement, low arithmetic intensity. |

One boundary decision worth restating: the APE add (`score += ape[position %
128]`) is fused into `_SAVE_PARTIAL_STATES_KERNEL` **upstream** of this task, so
`score_state` arrives with it already applied. The task takes it as given rather
than recomputing it.

---

## 1. op_type **[corrected]**

The plan claimed the vendored suite contains "35 Definitions across 8 op types".
**It does not.** The pinned tree contains exactly **one** Definition JSON,
`examples/ffi/Example-FlashInfer-Trace/definitions/gemm_n4096_k4096.json`. The
35-Definition figure describes the public flashinfer-bench dataset, which is not
vendored here. What the tree does carry is ten op-type *specifications* under
`docs/op-types/`: `dsa-paged`, `gdn`, `gemm`, `gqa-paged`, `gqa-ragged`,
`mla-paged`, `moe`, `rmsnorm`, `rope`, `sampling` (the plan's list omitted `gdn`
and `rope`). These are prose specs, not Definitions — so there is no local corpus
to pattern-match a new Definition against, and the schema had to be read from the
parsers instead. Doing that turned up four constraints the plan never
anticipated; see §2.

`op_type` is **`hca_compress`**, a new type. The compressor is semantically its
own thing, and forcing it under `mla_paged` would make that op-type's spec
incoherent. An `hca_compress` op-type document, following `mla-paged.mdx` and
`dsa-paged.mdx`, is still unwritten.

**Constraint:** the vendored tree is read-only reference. The HCA Definition is
never written into `third_party/flashinfer-bench/` — see `third_party/VENDOR.md`:
a vendored tree is replaced wholesale on a pin bump, never patched in place.

---

## 2. The Definition

Schema authority: `flashinfer_bench/data/definition.py` (the parser — trust it
over `docs/flashinfer-trace/definition.mdx`, which is less precise).

### Four schema constraints that are not in the docs **[corrected]**

Each of these was discovered by running the real parser, and each had already
been violated by a draft that looked entirely plausible:

1. **`AxisConst.value` must be an integer.** The draft carried
   `rms_norm_eps = 1e-6` as a const axis. Rejected. The epsilon is now inlined
   in the reference as `_RMS_NORM_EPS`, which is where a non-integer constant
   belongs.
2. **There are no unsigned dtypes.** The DType enum (`definition.py:49`) is
   float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, float4_e2m1,
   int64/32/16/8, bool. vLLM stores the UE8M0 scale as a `uint8` holding
   `exponent + 127`; this Definition stores the **unbiased** exponent as `int8`,
   which is the same one byte of store traffic and so preserves the performance
   characteristics the benchmark measures.
3. **Shape entries must be declared axis names, never expressions.**
   `_validate_tensor_axis_references` rejects `total_tokens // compress_rate`.
   This is what forced the input layout change in §2.1.
4. **The reference `run` receives only the input tensors — no axes.**
   `compile/registry.py:175-192` wraps `reference` as a pseudo-Solution with
   `destination_passing_style=False` and entry `main.py::run`, and
   `compile/builder.py:156-173` binds exactly `len(inputs)` positional arguments
   (plus `len(outputs)` for a DPS solution). A draft reference taking nine axis
   parameters would have failed at build time, long after authoring.
   `tools/validate_task.py` now checks this arity statically, for both the
   reference and every Solution.

### 2.1 axes

`var`: `num_compressed`, `max_position`.
`const`: `compress_rate = 128`, `head_dim = 512`, `nope_head_dim = 448`,
`rope_head_dim = 64`, `rope_head_dim_half = 32`, `quant_block = 64`,
`num_quant_blocks = 7`.

`compress_rate` **must** be `const`: 4 is CSA, 128 is HCA, and that value
changing means a different kernel. The vendored GEMM example does the same — `N`
and `K` are locked and only `M` varies.

**The window axis is structural, not cosmetic.** `kv_state` and `score_state`
are `[num_compressed, compress_rate, head_dim]`, not the flat
`[total_tokens, head_dim]` vLLM uses. Two independent reasons:

- The output shapes are `[num_compressed, ...]`, and `evaluators/utils.py:35`
  resolves output shapes **only** from the *input tensor shapes* — never from
  the workload's declared axes. With a flat input, `num_compressed` is not
  derivable from any input shape, and the only way to express it would be the
  shape expression that constraint 3 forbids.
- It costs nothing. For a contiguous buffer the 3-D form is a zero-copy view;
  the bytes are byte-identical to vLLM's flat layout, so the memory traffic a
  solution is measured on is unchanged.

### 2.2 The reference formula **[corrected]**

The plan wrote the compression as

```
C_i = kv_norm( Σ_{j ∈ window} softmax(gate_j + position_bias)_j ⊙ kv_j )
```

**This is wrong in two ways.** There is no separate `gate` or `position_bias`
term — the positional contribution is the APE already folded into `score_state`
upstream (§0). And the softmax is **per column**: it runs over the window axis
independently for each of the 512 feature columns, so the weights are a
`[128, 512]` matrix, not a `[128]` vector. Writing it as a scalar attention
weight per token produces a different operator. The actual computation, per
window:

```
w        = softmax(score, dim=window)          # [128, 512], column-wise
c        = Σ_window (kv * w)                   # [512]
normed   = c * rsqrt(mean(c²) + 1e-6) * g      # RMSNorm over all 512
nope     = bf16(normed[:448])                  # rounded ONCE, feeds both outputs
rope     = rotate(normed[448:])                # fp32, interleaved pairs
ckv      = bf16(concat(nope, rope))
ckv_fp8, ckv_scale = ue8m0_quant(nope)         # 7 blocks of 64
```

Three details that a plausible implementation gets wrong, all of which the
mutation battery (§5) actively rejects:

- **RMSNorm divides by `head_dim` (512), not by the 448 noPE columns.**
- **The rotation reads the fp32 `normed`; only the FP8 path and `ckv`'s noPE
  columns read the bf16-rounded copy.** vLLM
  (`fused_compress_quant_cache.py:314`) reshapes `normed`, not `quant_input`.
  Rotating the rounded copy instead is a real and easy mistake.
- **`ckv` stores the *rotated* suffix**, not the pre-rotation values.

### 2.3 RoPE convention — resolved, not open **[corrected]**

The plan flagged the RoPE convention as an unresolved blocker ("if this is not
resolved… the task is dead on arrival"), on the grounds that vLLM and HF
transformers disagree. **It was never genuinely open.** vLLM is the reference
implementation for this operator, and it settles the question:
`is_neox_style = False` — GPT-J interleaved pairs `(2j, 2j+1)`, **not**
split-half — applied to the **last** 64 of 512 columns, at position
`(positions // compress_ratio) * compress_ratio`. HF's split-half `rotate_half`
simply is not what this kernel computes. The convention is stated in the
Definition's `description` and enforced by the reference; the split-half variant
is one of the mutations the tolerances reject (matched_ratio 0.938).

The one genuine subtlety is the position: entry `c` covers tokens
`[c*128, c*128+127]`, and the angle comes from the boundary token **floored to
the window start**, i.e. row `c*128` — *not* row `c*128+127`. Using the boundary
token itself is also a rejected mutation (0.877).

Note also that `cos_cache`/`sin_cache` here are two tensors of width 32 (one
column per *pair*), whereas vLLM keeps a single `cos_sin_cache` of width 64 with
cos in the first half and sin in the second. Same values, split for clarity.

### 2.4 constraints

16 string expressions, following `dsa-paged.mdx`. They pin the geometry
(`head_dim == nope_head_dim + rope_head_dim`, `num_quant_blocks ==
nope_head_dim // quant_block`, …), every tensor shape, and the one relation
between the two var axes: `max_position > (num_compressed - 1) * compress_rate`.

---

## 3. The Workload sweep **[corrected]**

20 workloads in `hca_compress_c128_h512_r64.jsonl`. Seventeen of them sweep
`num_compressed ∈ {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
768, 1024}` — i.e. 128 to 131072 tokens — with `max_position = num_compressed *
128`. Following the 43-point GEMM sweep's shape: dense at the small end where
launch overhead dominates, geometric through the middle, plus off-power-of-two
points (24, 48, 96, 192, 384, 768) so nothing can quietly assume a power of two.
The remaining three sit at `num_compressed = 64` and vary only the input
*distribution* — see §7.7.

The plan insisted the sweep must include **`total_tokens` not divisible by 128**,
as "where real implementations break". **That is now structurally impossible**,
and deliberately so: the input is `[num_compressed, 128, head_dim]`, so a partial
trailing window cannot be expressed. This is not a coverage regression — it
reflects the operator's real boundary. vLLM's C128A path only runs the
compressor on complete windows, gated by `state_metadata.c128_boundary`; the
partial tail lives in the *caller*, outside this task. Encoding an impossible
input would have tested a case the kernel never sees.

Input descriptors are all `{"type": "random"}`, with one known consequence
recorded in `eval_config.yaml`: random data never drives a 64-element block's
absmax below `1e-4`, so the `clamp(min=1e-4)` guard is never exercised. Closing
that would need a degenerate near-zero `kv_state` workload. Left open
deliberately — the clamp only guards a division by ~0 and cannot produce a
wrong-but-plausible result.

---

## 4. The Solution

`spec.language = "triton"`, `target_hardware = ["NVIDIA_H200"]`,
`entry_point = "hca_compress_c128.py::run"`, `destination_passing_style = true`.

The sources live as real files under `solutions/triton_h200/` and the JSON is
generated from them by `tools/build_solution.py`. This is deliberate: hand-
escaping a Triton kernel into a JSON string literal is how an earlier draft of
the *reference* acquired silent edit failures — edits that appeared to apply and
did nothing. Nothing in this task is authored inline in JSON.

One program per compressed entry, each owning a whole 128×512 window, so the
softmax reduction never crosses a program boundary: no cross-block reduction, no
atomics, no second pass. Affordable precisely because HCA folds 128 tokens into
one entry.

The plan recommended shipping a `language: "python"` Solution first, to prove the
specification executable at no GPU cost. That step was skipped — the Definition's
own `reference` is a plain-PyTorch implementation and is executed directly by
`tools/check_hca_numerics.py`, which serves the same purpose. A separate Python
Solution would be a third copy of the same arithmetic.

---

## 5. Correctness bar, and why it is a ratio

The default evaluator is used as-is. `resolve_evaluator` raises if more than one
evaluator's `can_evaluate` matches, and `hca_compress` trips none of them.

Scoring is `bench/utils.py::compute_error_stats`: elements fail only when
`abs_error > atol` **and** `rel_error > rtol` (an AND), and `matched_ratio` is
the fraction that pass. `required_matched_ratio` **defaults to 1.0** — bitwise
agreement.

**That default is unsatisfiable here**, for a reason intrinsic to the operator
rather than to any kernel: PyTorch and Triton reduce the 128-token softmax in
different orders, disagreeing by ~1e-7 in fp32, and both results are then rounded
to bf16 and e4m3. Near a rounding boundary that 1e-7 flips a full step. Measured:
48 of 262144 `ckv` elements at 65536 tokens (one bf16 ulp each), 1–4 `ckv_fp8`
bytes of 229k–459k (one e4m3 code point each), and `ckv_scale` bitwise exact at
every workload. Demanding equality would reject correct kernels and reject
nothing else.

`eval_config.yaml` therefore ships `rtol = 1e-3`, `atol = 1e-2`,
`required_matched_ratio = 0.999`, with the full derivation in its header. The
numbers are held honest by a **mutation battery**: eight single-token semantic
edits to the reference, each of which the configuration must reject while
accepting the solution.

```
solution worst 0.99998692  >  threshold 0.999  >  best mutation 0.962821
```

Run it with `/venv/tianyi/bin/python3 tools/check_hca_numerics.py --mutations`.
The harness reads its tolerances *from* `eval_config.yaml` rather than repeating
them, so the two cannot drift apart.

Two further properties the harness establishes, neither of which follows from
solution-vs-reference agreement alone:

- The reference matches an **independent transcription of vLLM's fused kernel**,
  written from that source rather than from the reference. This is what makes the
  reference a specification instead of merely an implementation.
- `ckv_fp8`/`ckv_scale` dequantise back to `ckv`'s noPE columns to within **half
  an e4m3 ulp**, checked per element. Note that a flat relative bound is wrong
  here: e4m3's 6.25% half-ulp relative bound holds only for *normals*, and
  elements far below their block's absmax land in the denormal range where the
  relative error is unbounded while the absolute step stays 2⁻⁹.

Finally, on speedup: the denominator `reference_latency_ms` is the unoptimized
PyTorch `run`, so `speedup_factor` is large by construction. **This is not
KernelBench's `fast_p` hardware-relative ceiling** — do not reason about
thresholds as if it were.

---

## 6. Environment **[corrected]**

The plan stated "this host has no GPU: no `/dev/nvidia*`, no `nvidia-smi`, no
`nvcc`, and no torch or triton installed", and made Modal authentication a
blocker for producing any Trace. **All of that is obsolete.** The work was done
on an H200. The one trap worth recording: torch 2.11.0+cu128 and triton 3.6.0
live in **`/venv/tianyi`**, not `/venv/main` and not the system Python. Use
`/venv/tianyi/bin/python3` for anything touching the GPU.

`tools/validate_task.py` runs anywhere — it deliberately loads only
`flashinfer_bench.data`, bypassing the package `__init__` that would pull the
agent and bench stacks and ultimately `flashinfer` itself.

---

## 7. Repository layout **[resolved]**

The plan left open whether Definitions live under `tasks/<task_id>/` or as
generated artifacts under `data/`. They live under **`tasks/<task_id>/`**:
`data/` is gitignored, and these are reviewable source artefacts, not build
output.

That question turned out to matter more than it looked. The `.gitignore` pattern
was an unanchored `data/`, which matches at *any* depth — and it had silently
swallowed three directories inside the vendored tree, including
`flashinfer_bench/data/`, the package holding the very parsers this pipeline
validates against. The vendored tree was committed incomplete and `import
flashinfer_bench` failed. The patterns are now anchored (`/data/`, `/jobs/`,
`/dist/`, `/logs/`) and the three directories restored from the pin, verified
byte-identical. Worth remembering as a class of bug: a gitignore pattern that
silently removes files from a *vendored* tree produces a repository that looks
complete and is not.

---

## 7.5 First Trace, and what it says about the premise **[measured]**

The task now has data. `tools/run_benchmark.py tasks/hca_compress_c128` produced
17 traces, one per workload, all `PASSED`, written to
`data/trace_sets/hca_compress_c128/traces/claude-opus-5/hca_compress/`.

Three things came out of it.

**The premise holds, and is now a number rather than a belief.** This task was
justified on the claim that the HCA compressor has real fusion headroom.
Measured (`tools/roofline.py`), the seed Triton solution peaks at **222.5 GB/s,
5.3% of this H200's sustained 4218 GB/s** — **19x off roofline** on an op that
is pure streaming. The headroom is not a rounding error; it is the whole task.

**The seed solution is correct and slow, which is what a seed should be.** The
compiled kernel reports `n_regs=32, n_spills=946 B/thread`: a 128x512 window
held live in fp32 is 256 KiB per block, the register file cannot hold it, and
the kernel spills 236 KiB per block against the 256 KiB it streams — it nearly
doubles its own traffic. Diagnosis and the obvious first optimisations are in
the solution's `DESCRIPTION.md`.

**`speedup_factor` is not a usable objective on its own.** It is solution
latency over the reference's, and the reference is eager PyTorch, so across this
sweep it measures the wrong thing at both ends:

| num_compressed | 1 | 64 | 256 | 1024 |
|---|---|---|---|---|
| latency (ms) | 0.0960 | 0.1339 | 0.3350 | 1.2286 |
| % of peak BW | 0.07 | 2.99 | 4.78 | 5.21 |
| `speedup_factor` | 2.86x | 2.03x | 1.37x | 1.18x |

At the small end it rewards launch-overhead elimination — 2.86x for moving 0.3
MB in 96 us, one Triton launch against fifteen eager ones, which is precisely
the KernelBench-style signal this project exists to avoid. At the large end it
saturates near 1.2x because the reference is bandwidth-bound too, so a kernel
19x off roofline still looks respectable. Any ranking or reward built on
`speedup_factor` alone would be dominated by the eight smallest workloads.
Achieved bandwidth against measured peak has neither failure mode: absolute
scale, known ceiling. The traces carry latency, so both can be computed from the
same data — `tools/roofline.py` does.

One upstream defect surfaced: `DefaultEvaluator` cannot evaluate *any*
Definition with a float8 output, because it screens outputs with `torch.isinf`,
which has no float8 kernel (`NotImplementedError: "isinf" not implemented for
'Float8_e4m3fn'`, default.py:142). Every workload failed `RUNTIME_ERROR` on a
kernel that was in fact correct. That is not a torch oversight — the `fn` in
e4m3fn means finite-only, so the format has no inf encoding and the answer is
trivially False; `torch.isinf` on `float8_e5m2` works fine.

It was first worked around out-of-tree, in an evaluator registered at runtime.
That kept the pin pristine but left the emitted TraceSet unusable by anyone
running the stock CLI — the registry has no public registration hook, so a
consumer got twenty `RUNTIME_ERROR`s and no hint why. It is now fixed inside the
pin as a replayable patch series (`third_party/patches/`, documented in
`third_party/VENDOR.md`), which also routes `hca_compress` to `LowBitEvaluator`
and registers its measured `required_matched_ratio` in the bundled per-op_type
config. Patch 001 is a plain upstream bug and is worth filing upstream.

---

## 7.6 The baseline, and the three things it changed **[measured]**

`solutions/baseline/` now holds `hca_compress_c128_torch_compile` — the obvious
PyTorch expression of the Definition handed to `torch.compile` with no tuning,
authored as `baseline` so it lands in `solutions/baseline/` and
`traces/baseline/`. It exists because the dataset layout reserves that author for
the incumbent a candidate must beat, and because the `benchmark` check refuses to
run without one.

**Why not vLLM's kernel.** vLLM's C128A path is itself hand-written Triton
(`fused_compress_quant_cache.py`). vLLM is not a dependency here, and a
transcription of that kernel would be a second hand-written Triton solution — a
competitor to the seed, not a baseline. What a practitioner actually reaches for
first, on an op with no library implementation, is `torch.compile`, and "how
much does the compiler already get?" is the question a baseline should answer.
The `vllm_port` in `tools/check_hca_numerics.py` stays what it is: a per-entry
Python loop that exists to prove the reference is a *spec*, not to be fast.

### It reframes the seed solution's performance

| num_compressed | 1 | 64 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| baseline GB/s | 1.5 | 81.3 | 330.7 | 661.3 | **1274.4** |
| seed GB/s | 2.8 | 126.2 | 201.7 | 222.5 | 219.8 |
| baseline % of peak | 0.04 | 1.93 | 7.84 | 15.68 | **30.22** |
| baseline vs seed | 0.53x | 0.64x | 1.64x | 2.97x | **5.80x** |

§7.5 recorded the seed at 1.18x the eager reference at the top of the sweep and
called that flattering. It was: against the compiler the same kernel **loses by
5.8x**. The 19x-off-roofline headroom is real, but 5.8x of it is available
without writing a kernel at all, which is the honest bar. Note the crossover —
the seed wins below num_compressed ≈ 128 and loses above it.

### It exposes a fixed-overhead floor the sweep cannot see past

The baseline's latency is **flat at ~0.20 ms from num_compressed = 2 to 768**
while the bytes moved grow 400-fold. Only at 1024 does streaming dominate. (The
n=1 point is the one shape Inductor compiles statically and the only one below
the floor, at 0.176 ms; every later shape gets the dynamic kernel and its
guards.) The seed Triton kernel's floor is ~0.093 ms, so this is the baseline's
own dispatch/guard cost, not a harness artefact.

Consequence for the task: only the largest one or two workloads measure memory
throughput at all. A candidate faces two unrelated targets — beat ~0.2 ms of
per-call overhead below num_compressed ≈ 768, and beat 1274 GB/s above it. This
is the strongest argument yet that the sweep needs a second free axis (§8, item
6): 17 points along `num_compressed` buy almost nothing once the first ~12 are
all measuring the same constant.

### It is the first thing the tolerances *caught* rather than confirmed

The noPE columns are rounded to bf16 once, and both `ckv` and the FP8 block
scales read that rounded copy (§2.2). Written inline as
`normed[:, :448].to(torch.bfloat16).float()`, **Inductor deletes the
round-trip.** Measured on 2^20 random fp32 values:

```
eager    round-trip vs eager  no-round-trip : 32450/1048576 differ (3.09%)
compiled round-trip vs eager  round-trip    : 32450/1048576 differ (3.09%)
compiled round-trip vs eager no-round-trip  :     0/1048576 differ
```

The compiled form is bit-identical to the form with the cast removed.
`torch._inductor.config.emulate_precision_casts = True` does not prevent it. The
failure is silent where you would look and loud where you would not: `ckv` still
matches (it rounds to bf16 on the way out anyway) while 3.5% of `ckv_fp8` bytes
come out one e4m3 code point off — `matched_ratio = 0.96498`, *inside* the band
§5 reserves for deliberate semantic mutations (worst mutation 0.96282). A
tolerance calibrated to separate one correct kernel from eight mutations
rejected, unprompted, a compiler transformation that reading the Python could not
reveal. Fixed by splitting into two compiled regions so the bf16 tensor is real
storage; worst `matched_ratio` over the sweep is then 0.99999128.

### The dataset now passes all seven checks

`tools/validate_dataset.py tasks/hca_compress_c128` — a wrapper needed because
the evaluator registration must happen at module scope to survive `spawn` into
the benchmark's workers:

```
1 definitions: 1 ok, 0 warning, 0 error
  layout ok | definition ok | workload ok (20 valid) | solution ok (2)
  trace ok  | baseline ok (build ok, trace all passed, 20/20 workloads)
  benchmark ok (baseline: PASS, reference: PASS)
```

Getting `benchmark` to green required one fix of my own making.
`validate.py:1071` hardcodes `BenchmarkConfig(warmup_runs=2, iterations=5,
num_trials=1)` and **never reads a task's `eval_config.yaml`**, so
`required_matched_ratio` resolves from the evaluator's class default.
`HcaCompressEvaluator` had it at 1.0 — upstream's value, chosen so a run that
forgot the task config would fail loudly. That was wrong, not conservative: this
task's own config *derives from measurement* that bitwise equality over a
128-token softmax is unsatisfiable (§5), so 1.0 failed 6 of 17 workloads for the
baseline and would fail any correct kernel. The class default is now 0.999, the
value that derivation arrives at.

---

## 7.7 The workload inputs, and the model they now come from **[measured]**

The sweep's inputs were all `{"type": "random"}`. That is not a tunable knob:
`RandomInput` has no fields beyond `type` — no distribution, no seed, no scale
(`data/workload.py`) — so "random" resolves to exactly `torch.randn`
(`bench/utils.py::_rand_tensor`). For this Definition that was wrong in a way
worth stating precisely, because two of the five inputs were not merely
unrepresentative but *physically impossible*:

- `cos_cache`/`sin_cache` as N(0,1) reach ±5 and miss `cos² + sin² = 1` by ~30.
  A rotation table cannot look like that.
- `rms_norm_weight` as N(0,1) is half negative with a median near zero. The real
  tensor is 503/512 positive and spans 190×. This is the one that matters most:
  RMSNorm divides out the input scale, so it is the gain vector that sets the
  per-channel magnitudes the FP8 block quantiser then has to cover.

### The target model turned out to be public

The working assumption had been that DeepSeek-V4 was unavailable and the inputs
would have to be synthesised. That was wrong. `deepseek-ai/DeepSeek-V4-Flash` is
public and un-gated (148 GiB), and transformers 5.17 ships a *native*
`deepseek_v4` — including `DeepseekV4HCACompressor`, which is the reference this
task was written against. Reading it confirmed every convention the Definition
had to infer from vLLM: interleaved pairs, half-width cos/sin, the **trailing**
64 dims rotated, fp32 rotation, `ape` added to the gate before the softmax.

It also corrected one: the compressor branch uses `compress_rope_theta =
160000` with YaRN factor 16, not the model's main `rope_theta = 10000`, and V4
pins that branch's `attention_factor` to 1.0 so the tables stay on the unit
circle.

Four of the five inputs are therefore now *exact* rather than modelled
(`tools/model_probe.py`). Only the hidden state `h` feeding `wkv`/`wgate` is
synthesised, because a real V4 forward pass would mean implementing its fp4 MoE
(256 experts/layer) — a project, not a step. In its place `h` is lifted from
**real DeepSeek-V2-Lite hidden states**, harvested through transformers' native
`deepseek_v2` (no remote code): each V4 channel takes a real V2-Lite channel's
real time series, so the per-channel marginals and the heavy tails survive
exactly — measured kurtosis **429**, against 3.0 for a Gaussian.

| Definition input | source | exact? |
|---|---|---|
| `rms_norm_weight` | `layers.11.attn.compressor.norm.weight` | yes |
| `cos_cache`/`sin_cache` | `DeepseekV4RotaryEmbedding(..., "compress")` | yes |
| `score_state` | `h @ wgate.T + ape` | weights yes |
| `kv_state` | `h @ wkv.T` | weights yes |

### What it bought: the quantiser's dynamic range

|  | UE8M0 exponents | distinct | `clamp(1e-4)` hit |
|---|---|---|---|
| randn (before) | `[-8, -4]` | 5 | 0/24913 (0.00%) |
| real (after) | `[-22, -7]` | 7 | 443/26257 (1.69%) |

Against a representable `[-22, 120]` this is still one-sided — real activations
do not produce huge-magnitude blocks either — but the span is 15 exponents
instead of 4, and the lower bound is now *exactly* the clamp. That closes §8
item 3: the `clamp(min=1e-4)` path, which no workload had ever executed, is now
driven by a `tiny` stress workload at 98.9% of its blocks, and both solutions
pass it at `matched_ratio = 1.00000000`.

### It also settles whether input values can matter at all

Three stress workloads (`flat`, `peaked`, `tiny`) share `num_compressed = 64`
with the real one and differ *only* in distribution — same shape, same bytes:

| mode | softmax max | e range | triton ms | baseline ms (harness) |
|---|---|---|---|---|
| real | 0.074 | `[-11, -7]` | 0.1338 | 0.2078 |
| flat | 0.008 | `[-11, -8]` | 0.1337 | 0.1202 |
| peaked | 0.841 | `[-11, -8]` | 0.1336 | 0.1166 |
| tiny | 0.074 | `[-22, -21]` | 0.1336 | 0.1203 |

A 100× swing in softmax concentration and a 1e-7 rescale move the Triton
kernel's latency by **0.15%**. The op is a pure streaming reduction with no
data-dependent control flow or addressing, so this is what theory predicts;
having it as a controlled measurement is what licenses the sizing decision
below.

The baseline's column is the interesting one, and it does **not** contradict
that. Its 1.78× spread cannot be a data effect: `flat`, `peaked` and `tiny`
are the three that differ most from each other and they agree to 3%, while the
outlier is `real`, whose data is the *least* unusual of the four. What sets it
apart is not its bytes but its position — it is the ninth workload in the
sweep, the other three are the last. Timing all four inside one process
settles it:

```
fresh process, n=64 first   real 0.0935  flat 0.0911  peaked 0.0896  tiny 0.0911
same, after the 17 shapes   real 0.1047  flat 0.1072  peaked 0.1047  tiny 0.1034
```

Within a process the four agree to 4% in either order, and the whole group
shifts by 13% depending on how many shapes Inductor has already compiled. So
the harness spread measures `torch.compile`'s dispatch/guard state at the
moment an evaluation runs — the same fixed-overhead floor §7.5 found — and not
the input distribution. (Absolute values are lower here than in the harness
because the probe calls `run` directly, with inputs already resident and no
correctness pass.) A candidate kernel that is not a compiled graph, like the
Triton seed, shows none of it.

Across the whole sweep the baseline's best is 1274.4 GB/s (30.22% of peak)
against 1227.6 on random data, and the Triton seed's is 222.5 GB/s (5.28%)
against 222.6 — identical to within noise.

### Sizing

Since `gen_inputs` dispatches per input *name*, blobs and random can be mixed
within one workload. All 20 workloads get real `rms_norm_weight`/`cos_cache`/
`sin_cache`; `kv_state`/`score_state` are blobbed only up to
`num_compressed = 128`. Above that they stay random, because those workloads
exist to measure bandwidth and — per the table above — values cannot change a
bandwidth number. Total 271 MB instead of ~1 GB.

The generator (`tools/gen_workload_blobs.py`) is seeded and never overwrites the
authored sweep in place. This is the point worth keeping: a captured blob is one
sample of one model at one moment, and it cannot say what it covers. A kernel
has to be correct across the distribution, so the input distribution is now a
*declared* dimension of the sweep rather than a hidden constant.

---

## 8. What remains

1. Author `docs/op-types/hca-compress.mdx` (in this repo, not the vendored tree).
2. ~~Produce the first Trace~~ — done, §7.5.
3. ~~Close the `clamp(min=1e-4)` coverage gap~~ — done, §7.7: the `tiny` stress
   workload drives 98.9% of its blocks under the clamp.
4. Optionally revisit C128A index construction (§0) as a second task.
5. Decide whether the small workloads (num_compressed <= 32) stay. They cost
   nothing to run and they do test the partial-window and n=1 edges, but their
   `speedup_factor` is overhead noise (§7.5) and must not feed a ranking.
6. Give the sweep a second free axis. §7.6 makes this concrete rather than
   aesthetic: with `num_compressed` as the only variable, ~12 of the 17
   workloads sit on the baseline's flat overhead floor and measure the same
   constant. Candidates: make `compress_rate` or `head_dim` `var`, or add C64 /
   C256 Definitions.
7. ~~Add a baseline solution~~ — done, §7.6.
8. Report the `torch.isinf` float8 defect (§7.5) and the hardcoded
   `BenchmarkConfig` in `validate.py`'s `benchmark` check (§7.6) upstream.
9. Reconcile the Definition against the now-available native
   `DeepseekV4HCACompressor` (§7.7) line by line. Every convention checked so
   far agrees, but the reference was written from vLLM and has not been diffed
   against the library implementation in full.
10. Consider widening the exponent coverage upward. Real activations reach
    `e = -7`; the representable range runs to 120, and nothing in the sweep
    tests a huge-magnitude block — which is exactly where `rtol = 1e-3` was
    chosen to stay sensitive (`eval_config.yaml`).

---

## Appendix: reference facts

### flashinfer-bench objects

| Object | Role |
|---|---|
| **Definition** | `op_type`, `axes`, tensor specs, and a plain-PyTorch `run` string as the mathematical specification |
| **Workload** | concrete values for the `var` axes plus input data descriptors |
| **Solution** | `sources[]`, `spec.language`, `entry_point`, `destination_passing_style` |
| **Trace** | `status ∈ {PASSED, INCORRECT_SHAPE, INCORRECT_NUMERICAL, INCORRECT_DTYPE, RUNTIME_ERROR, COMPILE_ERROR}`, max rel/abs error, `latency_ms`, `reference_latency_ms`, `speedup_factor`, environment snapshot |

`SupportedLanguages` is `{python, triton, cpp, cuda, tilelang}` — the plan's list
omitted `tilelang`.

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

Within the compressor the 512 columns split as `[noPE 448 | RoPE 64]`, and the
448 noPE columns quantize as 7 blocks of 64.

### UE8M0 FP8 quantization

Per 64-element block: `exponent = ceil(log2(max(absmax, 1e-4) / 448))`, clamped
to `[-127, 127]`; values are scaled by `2^-exponent`, clamped to ±448, and cast
to e4m3. Keep the division inside the log — `log2(x / 448)` is not the same
float as `log2(x) - log2(448)`.

The exponent range is bounded on both sides, which is what makes `rtol = 1e-3`
able to catch a one-off exponent error anywhere in the representable range: the
`1e-4` absmax clamp puts the floor at `ceil(log2(1e-4/448)) = -22`, and bf16
inputs put the ceiling at `ceil(log2(3.39e38/448)) = 120`.

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
  This is the gate that keeps the compressor on complete windows only — the
  reason §3's partial-tail case does not belong to this task.
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
