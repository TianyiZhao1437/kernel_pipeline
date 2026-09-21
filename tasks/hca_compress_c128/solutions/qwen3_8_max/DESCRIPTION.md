Two Triton programs, generated end to end by `qwen3.8-max` through
flashinfer-bench's own `KernelGenerator` (10 rounds, no hand editing).

`_hca_compute_compressed` splits each 128x512 window along the head dimension:
every program owns a (128, BLOCK_D) slab, so the softmax over the 128 rows never
crosses a program boundary. It keeps the intermediate in fp32 (`compressed_fp32`)
rather than bf16, which is what lets the later ue8m0 step see the unrounded
values.

`_hca_finish_kernel` then does the whole second stage in one program per
compressed entry: RMSNorm against the fp32 intermediate, the RoPE rotation on
the 448:512 slice, and the per-64-block fp8 quantisation. The ue8m0 exponent is
extracted by bit manipulation on the fp32 absmax rather than by `frexp`:
`exp_i = unbiased_exp + (mantissa != 0)`, which is round-half-up on the
exponent, and the block is then scaled by `exp2(-exp_i)`.

Autotuning is a fixed heuristic ladder on `num_compressed` (BLOCK_D and
num_warps step at 4/8/16/32/64), not a measured sweep.

The ue8m0 rounding and the fp8 clamp are the two places this can diverge from
the reference; see `eval_config.yaml` for the tolerance that governs them.
