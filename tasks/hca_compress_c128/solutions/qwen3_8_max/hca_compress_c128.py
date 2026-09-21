import torch
import triton
import triton.language as tl

_COMPRESS_RATE = 128
_HEAD_DIM = 512
_NOPE_HEAD_DIM = 448
_ROPE_HEAD_DIM = 64
_ROPE_HALF = 32
_QUANT_BLOCK = 64
_NUM_QUANT_BLOCKS = 7
_NUM_BLOCKS_TOTAL = _HEAD_DIM // _QUANT_BLOCK
_EPS = 1e-6
_FP8_MAX = 448.0


@triton.jit
def _hca_compute_compressed(
    kv_ptr,
    score_ptr,
    out_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATE: tl.constexpr,
):
    pid_c = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    t_off = tl.arange(0, BLOCK_T).to(tl.int64)
    d_off = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)

    window_base = pid_c * (COMPRESS_RATE * HEAD_DIM)

    score_ptrs = (
        score_ptr
        + window_base
        + t_off[:, None] * HEAD_DIM
        + d_off[None, :]
    )
    score = tl.load(score_ptrs, eviction_policy="evict_first").to(tl.float32)

    max_score = tl.max(score, axis=0)
    safe_max = tl.where(max_score == float("-inf"), 0.0, max_score)

    exp_score = tl.exp(score - safe_max[None, :])
    sum_exp = tl.sum(exp_score, axis=0)
    sum_exp = tl.where(sum_exp == 0.0, 1.0, sum_exp)
    inv_sum = 1.0 / sum_exp

    kv_ptrs = (
        kv_ptr
        + window_base
        + t_off[:, None] * HEAD_DIM
        + d_off[None, :]
    )
    kv = tl.load(kv_ptrs, eviction_policy="evict_first").to(tl.float32)

    compressed = tl.sum(kv * exp_score, axis=0) * inv_sum

    out_ptrs = out_ptr + pid_c * HEAD_DIM + d_off
    tl.store(out_ptrs, compressed)


@triton.jit
def _hca_finish_kernel(
    comp_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    ckv_ptr,
    fp8_ptr,
    scale_ptr,
    stride_cos_pos,
    stride_cos_j,
    stride_sin_pos,
    stride_sin_j,
    HEAD_DIM: tl.constexpr,
    NOPE_HEAD_DIM: tl.constexpr,
    ROPE_HALF: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NUM_QUANT_BLOCKS: tl.constexpr,
    NUM_BLOCKS_TOTAL: tl.constexpr,
    COMPRESS_RATE: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_c = tl.program_id(0).to(tl.int64)

    stride_cos_pos = tl.cast(stride_cos_pos, tl.int64)
    stride_cos_j = tl.cast(stride_cos_j, tl.int64)
    stride_sin_pos = tl.cast(stride_sin_pos, tl.int64)
    stride_sin_j = tl.cast(stride_sin_j, tl.int64)

    ckv_dtype = ckv_ptr.dtype.element_ty
    fp8_dtype = fp8_ptr.dtype.element_ty

    d_all = tl.arange(0, HEAD_DIM).to(tl.int64)
    comp_base = comp_ptr + pid_c * HEAD_DIM

    comp_all = tl.load(comp_base + d_all, eviction_policy="evict_first")
    sumsq = tl.sum(comp_all * comp_all, axis=0)
    inv_rms = tl.rsqrt(sumsq / HEAD_DIM + EPS)

    w_all = tl.load(weight_ptr + d_all, eviction_policy="evict_last").to(tl.float32)
    normed_all = comp_all * inv_rms * w_all
    ckv_all = normed_all.to(ckv_dtype)

    ckv_base = ckv_ptr + pid_c * HEAD_DIM
    mask_nope = d_all < NOPE_HEAD_DIM
    tl.store(ckv_base + d_all, ckv_all, mask=mask_nope)

    ckv_2d = tl.reshape(ckv_all, (NUM_BLOCKS_TOTAL, QUANT_BLOCK))
    rounded_2d = ckv_2d.to(tl.float32)

    b = tl.arange(0, NUM_BLOCKS_TOTAL).to(tl.int64)
    j = tl.arange(0, QUANT_BLOCK).to(tl.int64)

    mask_b = b < NUM_QUANT_BLOCKS
    b_store = tl.where(mask_b, b, 0)
    mask_2d = mask_b[:, None] & (j[None, :] < QUANT_BLOCK)

    absmax = tl.max(tl.abs(rounded_2d), axis=1)
    absmax = tl.where(mask_b, tl.maximum(absmax, 1e-4), 1.0)

    y = absmax / FP8_MAX
    bits = tl.cast(y, tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 255
    mantissa = bits & 8388607
    unbiased_exp = biased_exp - 127

    exp_i = tl.where(mantissa == 0, unbiased_exp, unbiased_exp + 1)
    exp_i = tl.minimum(tl.maximum(exp_i, -127), 127)
    exp_i = tl.where(mask_b, exp_i, 0)

    scale_factor = tl.math.exp2(-exp_i.to(tl.float32))
    scaled = rounded_2d * scale_factor[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -FP8_MAX), FP8_MAX)

    fp8_2d = scaled.to(fp8_dtype)
    fp8_base = fp8_ptr + pid_c * NOPE_HEAD_DIM
    fp8_off = b_store[:, None] * QUANT_BLOCK + j[None, :]
    tl.store(fp8_base + fp8_off, fp8_2d, mask=mask_2d)

    scale_base = scale_ptr + pid_c * NUM_QUANT_BLOCKS
    tl.store(scale_base + b_store, exp_i.to(tl.int8), mask=mask_b)

    jr = tl.arange(0, ROPE_HALF).to(tl.int64)
    even_off = NOPE_HEAD_DIM + 2 * jr
    odd_off = even_off + 1

    comp_even = tl.load(comp_base + even_off, eviction_policy="evict_first")
    comp_odd = tl.load(comp_base + odd_off, eviction_policy="evict_first")

    w_even = tl.load(weight_ptr + even_off, eviction_policy="evict_last").to(tl.float32)
    w_odd = tl.load(weight_ptr + odd_off, eviction_policy="evict_last").to(tl.float32)

    even = comp_even * inv_rms * w_even
    odd = comp_odd * inv_rms * w_odd

    pos = pid_c * COMPRESS_RATE
    cos = tl.load(
        cos_ptr + pos * stride_cos_pos + jr * stride_cos_j,
        eviction_policy="evict_first",
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + pos * stride_sin_pos + jr * stride_sin_j,
        eviction_policy="evict_first",
    ).to(tl.float32)

    out_even = even * cos - odd * sin
    out_odd = odd * cos + even * sin

    tl.store(ckv_base + even_off, out_even.to(ckv_dtype))
    tl.store(ckv_base + odd_off, out_odd.to(ckv_dtype))


def run(
    kv_state,
    score_state,
    rms_norm_weight,
    cos_cache,
    sin_cache,
    ckv=None,
    ckv_fp8=None,
    ckv_scale=None,
):
    orig_device = kv_state.device
    out_provided = ckv is not None and ckv_fp8 is not None and ckv_scale is not None
    num_compressed = int(kv_state.shape[0])

    if num_compressed == 0:
        if out_provided:
            return ckv, ckv_fp8, ckv_scale
        return (
            torch.empty((0, _HEAD_DIM), dtype=torch.bfloat16, device=orig_device),
            torch.empty((0, _NOPE_HEAD_DIM), dtype=torch.float8_e4m3fn, device=orig_device),
            torch.empty((0, _NUM_QUANT_BLOCKS), dtype=torch.int8, device=orig_device),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this kernel, but CUDA is not available.")

    if orig_device.type == "cuda":
        device_index = orig_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        target_device = torch.device("cuda", device_index)
    elif orig_device.type == "cpu":
        device_index = None
        if out_provided:
            for t in (ckv, ckv_fp8, ckv_scale):
                if t is not None and t.is_cuda:
                    device_index = t.device.index
                    break
        if device_index is None:
            device_index = torch.cuda.current_device()
        target_device = torch.device("cuda", device_index)
    else:
        raise RuntimeError(f"Unsupported device type: {orig_device.type}")

    kv = kv_state.to(target_device).contiguous()
    score = score_state.to(target_device).contiguous()
    weight = rms_norm_weight.to(target_device).contiguous()
    cos = cos_cache.to(target_device)
    sin = sin_cache.to(target_device)

    def _prepare_output(out, shape, dtype):
        if out is not None and tuple(out.shape) == shape and out.dtype == dtype:
            if out.device == target_device and out.is_contiguous():
                return out, out, False
            return torch.empty(shape, dtype=dtype, device=target_device), out, True
        return torch.empty(shape, dtype=dtype, device=target_device), None, False

    ckv_buf, ckv_orig, ckv_copy = _prepare_output(
        ckv,
        (num_compressed, _HEAD_DIM),
        torch.bfloat16,
    )
    ckv_fp8_buf, ckv_fp8_orig, ckv_fp8_copy = _prepare_output(
        ckv_fp8,
        (num_compressed, _NOPE_HEAD_DIM),
        torch.float8_e4m3fn,
    )
    ckv_scale_buf, ckv_scale_orig, ckv_scale_copy = _prepare_output(
        ckv_scale,
        (num_compressed, _NUM_QUANT_BLOCKS),
        torch.int8,
    )

    if num_compressed >= 64:
        compute_block_d = 64
        compute_num_warps = 8
    elif num_compressed >= 32:
        compute_block_d = 32
        compute_num_warps = 4
    elif num_compressed >= 16:
        compute_block_d = 16
        compute_num_warps = 4
    elif num_compressed >= 8:
        compute_block_d = 16
        compute_num_warps = 2
    elif num_compressed >= 4:
        compute_block_d = 8
        compute_num_warps = 2
    else:
        compute_block_d = 8
        compute_num_warps = 2

    num_d_blocks = _HEAD_DIM // compute_block_d
    compute_grid = (num_compressed, num_d_blocks)

    if num_compressed < 32:
        finish_num_warps = 4
    elif num_compressed < 1024:
        finish_num_warps = 2
    else:
        finish_num_warps = 1

    prev_device = torch.cuda.current_device()
    restore_device = prev_device != target_device.index
    if restore_device:
        torch.cuda.set_device(target_device)

    try:
        compressed_fp32 = torch.empty(
            (num_compressed, _HEAD_DIM),
            dtype=torch.float32,
            device=target_device,
        )

        _hca_compute_compressed[compute_grid](
            kv,
            score,
            compressed_fp32,
            BLOCK_T=_COMPRESS_RATE,
            BLOCK_D=compute_block_d,
            HEAD_DIM=_HEAD_DIM,
            COMPRESS_RATE=_COMPRESS_RATE,
            num_warps=compute_num_warps,
        )

        _hca_finish_kernel[(num_compressed,)](
            compressed_fp32,
            weight,
            cos,
            sin,
            ckv_buf,
            ckv_fp8_buf,
            ckv_scale_buf,
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            HEAD_DIM=_HEAD_DIM,
            NOPE_HEAD_DIM=_NOPE_HEAD_DIM,
            ROPE_HALF=_ROPE_HALF,
            QUANT_BLOCK=_QUANT_BLOCK,
            NUM_QUANT_BLOCKS=_NUM_QUANT_BLOCKS,
            NUM_BLOCKS_TOTAL=_NUM_BLOCKS_TOTAL,
            COMPRESS_RATE=_COMPRESS_RATE,
            EPS=_EPS,
            FP8_MAX=_FP8_MAX,
            num_warps=finish_num_warps,
        )

        if ckv_copy:
            ckv_orig.copy_(ckv_buf)
        if ckv_fp8_copy:
            ckv_fp8_orig.copy_(ckv_fp8_buf)
        if ckv_scale_copy:
            ckv_scale_orig.copy_(ckv_scale_buf)

        ckv_ret = ckv_orig if ckv_orig is not None else ckv_buf
        ckv_fp8_ret = ckv_fp8_orig if ckv_fp8_orig is not None else ckv_fp8_buf
        ckv_scale_ret = ckv_scale_orig if ckv_scale_orig is not None else ckv_scale_buf

        if ckv_orig is None and ckv_ret.device != orig_device:
            ckv_ret = ckv_ret.to(orig_device)
        if ckv_fp8_orig is None and ckv_fp8_ret.device != orig_device:
            ckv_fp8_ret = ckv_fp8_ret.to(orig_device)
        if ckv_scale_orig is None and ckv_scale_ret.device != orig_device:
            ckv_scale_ret = ckv_scale_ret.to(orig_device)

        return ckv_ret, ckv_fp8_ret, ckv_scale_ret
    finally:
        if restore_device:
            torch.cuda.set_device(prev_device)