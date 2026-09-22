import torch
import triton
import triton.language as tl
import math

_NOPE_HEAD_DIM = 448
_QUANT_BLOCK = 64
_RMS_NORM_EPS = 1e-6
_FP8_MAX = 448.0
_LOG2_FP8_MAX = math.log2(_FP8_MAX)
_COMPRESS_RATE = 128
_NUM_QUANT_BLOCKS = 7
_ROPE_HEAD_DIM = 64
_ROPE_HEAD_DIM_HALF = 32
_HEAD_DIM = 512


@triton.jit
def hca_compress_c128_h512_r64_kernel(
    kv_state_ptr, score_state_ptr, rms_norm_weight_ptr,
    cos_cache_ptr, sin_cache_ptr,
    ckv_ptr, ckv_fp8_ptr, ckv_scale_ptr,
    num_compressed, max_position,
    stride_kv_c, stride_kv_r, stride_kv_d,
    stride_sc_c, stride_sc_r, stride_sc_d,
    stride_cos_p, stride_cos_d,
    stride_sin_p, stride_sin_d,
    stride_ckv_c, stride_ckv_d,
    stride_ckv_fp8_c, stride_ckv_fp8_d,
    stride_ckv_sc_c, stride_ckv_sc_d,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    COMPRESS_RATE: tl.constexpr, NOPE_HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr, NUM_QUANT_BLOCKS: tl.constexpr,
    LOG2_FP8_MAX: tl.constexpr, FP8_MAX: tl.constexpr,
    RMS_NORM_EPS: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr, ROPE_HEAD_DIM_HALF: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < num_compressed
    m2 = mask_m[:, None]

    compressed_pos = offs_m * COMPRESS_RATE
    pos_ptr = compressed_pos * stride_cos_p
    offs_d2 = tl.arange(0, BLOCK_D2)
    cos_ptrs = pos_ptr[:, None] + offs_d2[None, :] * stride_cos_d
    sin_ptrs = pos_ptr[:, None] + offs_d2[None, :] * stride_sin_d
    cos = tl.load(cos_cache_ptr + cos_ptrs, mask=m2, other=1.0).to(tl.float32)
    sin = tl.load(sin_cache_ptr + sin_ptrs, mask=m2, other=0.0).to(tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    rn = tl.load(rms_norm_weight_ptr + offs_d).to(tl.float32)

    score_max = tl.full([BLOCK_M, BLOCK_D], -float('inf'), dtype=tl.float32)
    for r in range(COMPRESS_RATE):
        score_ptrs = (offs_m[:, None] * stride_sc_c + r * stride_sc_r + offs_d[None, :] * stride_sc_d)
        s = tl.load(score_state_ptr + score_ptrs, mask=m2, other=-float('inf')).to(tl.float32)
        score_max = tl.where(s > score_max, s, score_max)

    sum_w = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    wkv = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for r in range(COMPRESS_RATE):
        score_ptrs = (offs_m[:, None] * stride_sc_c + r * stride_sc_r + offs_d[None, :] * stride_sc_d)
        kv_ptrs = (offs_m[:, None] * stride_kv_c + r * stride_kv_r + offs_d[None, :] * stride_kv_d)
        s = tl.load(score_state_ptr + score_ptrs, mask=m2, other=-float('inf')).to(tl.float32)
        v = tl.load(kv_state_ptr + kv_ptrs, mask=m2, other=0.0).to(tl.float32)
        w = tl.exp(s - score_max)
        sum_w += w
        wkv += w * v

    compressed = wkv / sum_w

    sq = compressed * compressed
    variance = tl.sum(sq, axis=1, keep_dims=True) / BLOCK_D
    inv_rms = tl.rsqrt(variance + RMS_NORM_EPS)
    normed = compressed * inv_rms * rn[None, :]

    nope = normed[:, :NOPE_HEAD_DIM].to(tl.bfloat16).to(tl.float32)

    rope = normed[:, NOPE_HEAD_DIM:]

    rope_even = rope[:, 0::2]
    rope_odd = rope[:, 1::2]
    rot_even = rope_even * cos - rope_odd * sin
    rot_odd = rope_odd * cos + rope_even * sin

    ckv_nope_bf16 = nope.to(tl.bfloat16)
    ckv_rope_even_bf16 = rot_even.to(tl.bfloat16)
    ckv_rope_odd_bf16 = rot_odd.to(tl.bfloat16)

    ckv_nope_ptrs = offs_m[:, None] * stride_ckv_c + offs_d[None, :NOPE_HEAD_DIM] * stride_ckv_d
    tl.store(ckv_ptr + ckv_nope_ptrs, ckv_nope_bf16, mask=m2)

    rope_ckv_base = offs_m[:, None] * stride_ckv_c + NOPE_HEAD_DIM * stride_ckv_d
    ckv_rope_even_ptrs = rope_ckv_base + offs_d2[None, :] * 2 * stride_ckv_d
    ckv_rope_odd_ptrs = rope_ckv_base + (offs_d2[None, :] * 2 + 1) * stride_ckv_d
    tl.store(ckv_ptr + ckv_rope_even_ptrs, ckv_rope_even_bf16, mask=m2)
    tl.store(ckv_ptr + ckv_rope_odd_ptrs, ckv_rope_odd_bf16, mask=m2)

    for qb in range(NUM_QUANT_BLOCKS):
        q_start = qb * QUANT_BLOCK
        nope_block = nope[:, q_start : q_start + QUANT_BLOCK]

        absmax = tl.max(tl.abs(nope_block), axis=1)
        absmax = tl.where(absmax < 1e-4, 1e-4, absmax)

        exponent = tl.ceil(tl.log2(absmax) - LOG2_FP8_MAX)
        exponent = tl.where(exponent < -127.0, -127.0, exponent)
        exponent = tl.where(exponent > 127.0, 127.0, exponent)

        scaled = nope_block * tl.exp2(-exponent[:, None])
        scaled = tl.where(scaled < -FP8_MAX, -FP8_MAX, scaled)
        scaled = tl.where(scaled > FP8_MAX, FP8_MAX, scaled)
        ckv_fp8_block = scaled.to(tl.float8e4nv)

        q_offs = q_start + tl.arange(0, QUANT_BLOCK)
        ckv_fp8_ptrs = offs_m[:, None] * stride_ckv_fp8_c + q_offs[None, :] * stride_ckv_fp8_d
        tl.store(ckv_fp8_ptr + ckv_fp8_ptrs, ckv_fp8_block, mask=m2)

        ckv_scale_ptrs = offs_m * stride_ckv_sc_c + qb * stride_ckv_sc_d
        tl.store(ckv_scale_ptr + ckv_scale_ptrs, exponent.to(tl.int8), mask=mask_m)


def run(kv_state, score_state, rms_norm_weight, cos_cache, sin_cache, ckv, ckv_fp8, ckv_scale):
    orig_device = kv_state.device
    is_cuda_available = torch.cuda.is_available()

    if orig_device == torch.device('cpu'):
        if not is_cuda_available:
            raise RuntimeError("CUDA is not available, but the operation requires a GPU.")
        kv_state = kv_state.cuda()
        score_state = score_state.cuda()
        rms_norm_weight = rms_norm_weight.cuda()
        cos_cache = cos_cache.cuda()
        sin_cache = sin_cache.cuda()
        ckv = ckv.cuda()
        ckv_fp8 = ckv_fp8.cuda()
        ckv_scale = ckv_scale.cuda()
    elif not is_cuda_available or orig_device.type != 'cuda':
        raise RuntimeError("Input tensors must be on a CUDA device, but CUDA is not available or device is not CUDA.")

    num_compressed = kv_state.shape[0]

    if num_compressed == 0:
        if orig_device == torch.device('cpu'):
            return ckv.cpu(), ckv_fp8.cpu(), ckv_scale.cpu()
        return ckv, ckv_fp8, ckv_scale

    BLOCK_M = 8
    BLOCK_D = 512
    BLOCK_D2 = 32

    grid = ((num_compressed + BLOCK_M - 1) // BLOCK_M,)

    hca_compress_c128_h512_r64_kernel[grid](
        kv_state, score_state, rms_norm_weight,
        cos_cache, sin_cache,
        ckv, ckv_fp8, ckv_scale,
        num_compressed, cos_cache.shape[0],
        kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
        score_state.stride(0), score_state.stride(1), score_state.stride(2),
        cos_cache.stride(0), cos_cache.stride(1),
        sin_cache.stride(0), sin_cache.stride(1),
        ckv.stride(0), ckv.stride(1),
        ckv_fp8.stride(0), ckv_fp8.stride(1),
        ckv_scale.stride(0), ckv_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
        BLOCK_D2=BLOCK_D2,
        COMPRESS_RATE=128, NOPE_HEAD_DIM=448,
        QUANT_BLOCK=64, NUM_QUANT_BLOCKS=7,
        LOG2_FP8_MAX=math.log2(448.0), FP8_MAX=448.0,
        RMS_NORM_EPS=1e-6,
        ROPE_HEAD_DIM=64, ROPE_HEAD_DIM_HALF=32,
        HEAD_DIM=512,
    )

    if orig_device == torch.device('cpu'):
        return ckv.cpu(), ckv_fp8.cpu(), ckv_scale.cpu()
    return ckv, ckv_fp8, ckv_scale
