"""Triton implementation of the DeepSeek-V4 HCA (C128A) compressor.

One program per compressed entry: it owns the whole 128 x 512 window, so the
softmax reduction never crosses a program boundary and no atomics are needed.
That is affordable because HCA collapses 128 tokens into one entry, so there are
``num_compressed`` programs and each one has 512 columns of work.

Layout notes, all of which mirror the fused vLLM kernel:

* The 512 columns are ``[noPE(448) | RoPE(64)]``.
* RMSNorm divides by ``HEAD_DIM`` (512), not by 448.
* RoPE is the GPT-J / interleaved-pair convention: column pair ``(2j, 2j+1)``
  rotates by cache entry ``j``, over pairs ``[224, 256)`` of the 512 columns.
  It is NOT split-half, so a HF-style split-half cache produces wrong results.
* The angle comes from the window's boundary token floored to the window start,
  ``(i // 128) * 128`` -- not from the boundary token itself.
* The rotation reads the fp32 normalised vector; only the FP8 quantisation
  reads a bf16-rounded copy. Keeping those separate is what matches vLLM.
* FP8 uses UE8M0 block scales, one exponent per 64 columns, over the 7 noPE
  blocks of 64. Triton requires power-of-two shapes, so the block axis is padded
  to 8 and the padding is masked off; nothing is reshaped to 7 rows.
* ``ckv_scale`` carries the UNBIASED exponent as int8. vLLM stores
  ``exponent + 127`` in a uint8 cache; the Definition has no unsigned dtype, and
  int8 keeps the store one byte wide so the measured traffic is unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hca_compress_c128_kernel(
    kv_ptr,
    score_ptr,
    rms_ptr,
    cos_ptr,
    sin_ptr,
    ckv_ptr,
    ckv_fp8_ptr,
    ckv_scale_ptr,
    num_compressed,
    kv_stride_n,
    kv_stride_w,
    score_stride_n,
    score_stride_w,
    cos_stride,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_PAIRS: tl.constexpr,
    COMPRESS_RATE: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NUM_QUANT_BLOCKS: tl.constexpr,
    NUM_QUANT_BLOCKS_PAD: tl.constexpr,
    RMS_EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    entry = tl.program_id(0)
    if entry >= num_compressed:
        return

    offs_w = tl.arange(0, COMPRESS_RATE)
    offs_d = tl.arange(0, HEAD_DIM)

    # ---- gather the window -------------------------------------------------
    # The window axis is explicit in the input layout, so this program owns
    # exactly kv_state[entry] -- no division, no partial windows.
    kv = tl.load(
        kv_ptr + entry * kv_stride_n + offs_w[:, None] * kv_stride_w + offs_d[None, :]
    ).to(tl.float32)
    score = tl.load(
        score_ptr + entry * score_stride_n + offs_w[:, None] * score_stride_w + offs_d[None, :]
    ).to(tl.float32)

    # ---- softmax over the window axis (dim 0) -----------------------------
    score = score - tl.max(score, axis=0)[None, :]
    weight = tl.exp(score)
    weight = weight / tl.sum(weight, axis=0)[None, :]

    # ---- weighted sum, then RMSNorm over the full head_dim ----------------
    compressed = tl.sum(kv * weight, axis=0)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + RMS_EPS)
    rms_w = tl.load(rms_ptr + offs_d).to(tl.float32)
    normed = compressed * rrms * rms_w

    # ---- FP8 UE8M0 over the noPE part -------------------------------------
    # 448 noPE columns = 7 blocks of 64, one UE8M0 exponent each. `quant_input`
    # is the bf16-rounded vector, and it feeds both the FP8 grid and ckv's noPE
    # columns -- that is what makes ckv_fp8 a deterministic encoding of ckv.
    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    nope = tl.reshape(quant_input, (NUM_QUANT_BLOCKS_PAD, QUANT_BLOCK))
    blk = tl.arange(0, NUM_QUANT_BLOCKS_PAD)
    blk_mask = (blk < NUM_QUANT_BLOCKS)[:, None]
    block_absmax = tl.max(tl.where(blk_mask, tl.abs(nope), 0.0), axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)
    # Keep the division: log2(x) - log2(448) is not the same float as
    # log2(x / 448), and the reference does the latter.
    exponent = tl.ceil(tl.log2(block_absmax / FP8_MAX))
    exponent = tl.maximum(tl.minimum(exponent, 127.0), -127.0)
    scaled = nope * tl.exp2(-exponent)[:, None]
    scaled = tl.maximum(tl.minimum(scaled, FP8_MAX), -FP8_MAX)
    fp8 = scaled.to(tl.float8e4nv)

    offs_2d = blk[:, None] * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)[None, :]
    tl.store(ckv_fp8_ptr + entry * NOPE_DIM + offs_2d, fp8, mask=blk_mask)
    tl.store(
        ckv_scale_ptr + entry * NUM_QUANT_BLOCKS + blk,
        exponent.to(tl.int8),
        mask=blk < NUM_QUANT_BLOCKS,
    )

    # ---- GPT-J RoPE on the last ROPE_DIM columns --------------------------
    # Build the rotation over all 256 column pairs, but only the last 32 (the
    # rope suffix) get cos/sin: for the rest the masked load yields cos = 1,
    # sin = 0, so those pairs pass through unchanged. A single interleave then
    # rebuilds the 512-wide vector for one store.
    all_pairs = tl.reshape(normed, (HEAD_DIM // 2, 2))
    even, odd = tl.split(all_pairs)

    pair_idx = tl.arange(0, HEAD_DIM // 2)
    is_rope = pair_idx >= NOPE_PAIRS
    rope_pair = pair_idx - NOPE_PAIRS

    # Window boundary floored to the window start: entry c reads row
    # c * COMPRESS_RATE, NOT the boundary row c * COMPRESS_RATE + 127.
    position = entry * COMPRESS_RATE + COMPRESS_RATE - 1
    compressed_pos = (position // COMPRESS_RATE) * COMPRESS_RATE

    cs_off = compressed_pos * cos_stride + rope_pair
    cos_v = tl.load(cos_ptr + cs_off, mask=is_rope, other=1.0)
    sin_v = tl.load(sin_ptr + cs_off, mask=is_rope, other=0.0)

    rotated = tl.interleave(even * cos_v - odd * sin_v, odd * cos_v + even * sin_v)
    # ckv's noPE columns come from the bf16-rounded values; only the suffix is
    # freshly rotated fp32.
    out = tl.where(offs_d < NOPE_DIM, quant_input, rotated)
    tl.store(ckv_ptr + entry * HEAD_DIM + offs_d, out.to(tl.bfloat16))


def run(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    ckv: torch.Tensor,
    ckv_fp8: torch.Tensor,
    ckv_scale: torch.Tensor,
) -> None:
    """Destination-passing entry point: compress every window into one entry."""
    num_compressed, compress_rate, head_dim = kv_state.shape
    nope_dim = ckv_fp8.shape[1]
    quant_block = nope_dim // ckv_scale.shape[1]
    num_quant_blocks = ckv_scale.shape[1]

    if num_compressed == 0:
        return

    _hca_compress_c128_kernel[(num_compressed,)](
        kv_state,
        score_state,
        rms_norm_weight,
        cos_cache,
        sin_cache,
        ckv,
        ckv_fp8,
        ckv_scale,
        num_compressed,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state.stride(0),
        score_state.stride(1),
        cos_cache.stride(0),
        HEAD_DIM=head_dim,
        NOPE_DIM=nope_dim,
        NOPE_PAIRS=nope_dim // 2,
        COMPRESS_RATE=compress_rate,
        QUANT_BLOCK=quant_block,
        NUM_QUANT_BLOCKS=num_quant_blocks,
        NUM_QUANT_BLOCKS_PAD=triton.next_power_of_2(num_quant_blocks),
        RMS_EPS=1e-6,
        FP8_MAX=448.0,
        num_warps=8,
    )
