import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Shape constants pinned by the definition.
# ---------------------------------------------------------------------------
_HEAD_DIM = 512
_NOPE_HEAD_DIM = 448
_ROPE_HEAD_DIM = 64
_ROPE_HALF = 32
_QUANT_BLOCK = 64
_NUM_QUANT_BLOCKS = 7
_COMPRESS_RATE = 128
_RMS_NORM_EPS = 1e-6
_FP8_MAX = 448.0

# ceil(log2(x / 448)) evaluated with exact integer arithmetic.  Writing the
# absmax as sig * 2^(ea - 23) with sig = 2^23 + mant in [2^23, 2^24) and 448 as
# 7 * 2^6, the inequality 7 * 2^(e + 6) >= sig * 2^(ea - 23) becomes
# 7 * 2^(e + 29 - ea) >= sig, whose smallest solution is e = ea - 8 when
# sig <= 7 * 2^21 = 14680064 and e = ea - 7 otherwise.
_CEIL_THRESH = 14680064


@triton.jit
def _hca_compress_c128_kernel(
    kv_ptr,
    score_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    ckv_ptr,
    ckv_fp8_ptr,
    ckv_scale_ptr,
    stride_kv_c,
    stride_kv_t,
    stride_sc_c,
    stride_sc_t,
    stride_cos,
    stride_sin,
    stride_ckv_c,
    stride_fp8_c,
    stride_scale_c,
    COMPRESS_RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_HEAD_DIM: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    ROPE_HALF: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NUM_QUANT_BLOCKS: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    CEIL_THRESH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EVEN_T: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_T)

    # Row base pointers: the column offset is added once, outside the loop.
    kv_base = kv_ptr + pid * stride_kv_c + offs_d[None, :]
    sc_base = score_ptr + pid * stride_sc_c + offs_d[None, :]

    # ---- 1 + 2: online softmax over the window, folded into the weighted sum -
    # The softmax of score_state is per-column (dim=1 of [C, R, D]), so the
    # window never has to be materialised: a running max plus a rescaled
    # numerator/denominator pair is exact and costs one multiply per column per
    # tile to renormalise.
    m = tl.full([HEAD_DIM], -1.0e30, tl.float32)
    s = tl.zeros([HEAD_DIM], tl.float32)
    acc = tl.zeros([HEAD_DIM], tl.float32)

    for t0 in range(0, COMPRESS_RATE, BLOCK_T):
        offs_tt = t0 + offs_t
        row_off = offs_tt[:, None] * stride_kv_t
        sc_off = offs_tt[:, None] * stride_sc_t
        if EVEN_T:
            sc = tl.load(sc_base + sc_off, eviction_policy="evict_first").to(tl.float32)
            kv = tl.load(kv_base + row_off, eviction_policy="evict_first").to(tl.float32)
        else:
            mask2d = (offs_tt < COMPRESS_RATE)[:, None]
            sc = tl.load(
                sc_base + sc_off,
                mask=mask2d,
                other=float("-inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            kv = tl.load(
                kv_base + row_off,
                mask=mask2d,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

        m_new = tl.maximum(m, tl.max(sc, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(sc - m_new[None, :])

        s = s * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p * kv, axis=0)
        m = m_new

    inv_s = 1.0 / tl.maximum(s, 1.0e-30)
    compressed = acc * inv_s

    # ---- 3: RMSNorm over the full head_dim, in fp32 -------------------------
    var = tl.sum(compressed * compressed, axis=0) * (1.0 / HEAD_DIM)
    rstd = tl.rsqrt(var + EPS)
    gain = tl.load(weight_ptr + offs_d).to(tl.float32)
    normed = compressed * (rstd * gain)

    # Split the 512-wide row into 8 blocks of 64: the first 7 cover the noPE
    # part that is FP8-quantised, the 8th is the RoPE suffix.
    x2 = tl.reshape(normed, (HEAD_BLOCKS, QUANT_BLOCK))
    row_ids = tl.arange(0, HEAD_BLOCKS)[:, None]
    col_ids = tl.arange(0, QUANT_BLOCK)[None, :]
    nope_mask = tl.broadcast_to(row_ids < NUM_QUANT_BLOCKS, (HEAD_BLOCKS, QUANT_BLOCK))

    # The kernel rounds the noPE values to bf16 *before* encoding them, which
    # is what makes ckv_fp8 a deterministic encoding of ckv's noPE columns.
    nope_bf = tl.where(nope_mask, x2, 0.0).to(tl.bfloat16).to(tl.float32)

    # ---- 5: UE8M0 block scales over the noPE part ---------------------------
    absmax = tl.max(tl.abs(nope_bf), axis=1)
    absmax = tl.maximum(absmax, 1.0e-4)

    bits = absmax.to(tl.int32, bitcast=True)
    ea = ((bits >> 23) & 0xFF) - 127
    sig = (bits & 0x7FFFFF) | 0x800000  # 2^23 + mantissa, always in [2^23, 2^24)
    e_int = tl.where(sig > CEIL_THRESH, ea - 7, ea - 8)
    e_int = tl.minimum(tl.maximum(e_int, -127), 127)

    # 2 ** (-e) built bit-exactly; the biased exponent 127 - e is in [0, 254].
    scale = ((127 - e_int) << 23).to(tl.float32, bitcast=True)

    scaled = nope_bf * scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -FP8_MAX), FP8_MAX)
    fp8 = scaled.to(tl.float8e4nv)

    tl.store(
        ckv_fp8_ptr + pid * stride_fp8_c + row_ids * QUANT_BLOCK + col_ids,
        fp8,
        mask=nope_mask,
    )

    offs_b = tl.arange(0, HEAD_BLOCKS)
    tl.store(
        ckv_scale_ptr + pid * stride_scale_c + offs_b,
        e_int.to(tl.int8),
        mask=offs_b < NUM_QUANT_BLOCKS,
    )

    # ---- 4: GPT-J interleaved-pair RoPE on the suffix only ------------------
    # The angle comes from the window's boundary token floored to the window
    # start, i.e. cache row pid * compress_rate.
    rope = tl.sum(tl.where(row_ids == NUM_QUANT_BLOCKS, x2, 0.0), axis=0)
    even, odd = tl.split(tl.reshape(rope, (ROPE_HALF, 2)))

    pos = pid * COMPRESS_RATE
    offs_j = tl.arange(0, ROPE_HALF)
    cos = tl.load(cos_ptr + pos * stride_cos + offs_j).to(tl.float32)
    sin = tl.load(sin_ptr + pos * stride_sin + offs_j).to(tl.float32)

    rot_even = even * cos - odd * sin
    rot_odd = odd * cos + even * sin
    rotated = tl.reshape(tl.join(rot_even, rot_odd), (ROPE_HEAD_DIM,))

    # ckv holds the bf16-rounded noPE values and the *rotated* suffix.
    out2 = tl.where(
        nope_mask,
        nope_bf,
        tl.broadcast_to(
            tl.reshape(rotated, (1, ROPE_HEAD_DIM)), (HEAD_BLOCKS, QUANT_BLOCK)
        ),
    )
    tl.store(
        ckv_ptr + pid * stride_ckv_c + offs_d,
        tl.reshape(out2, (HEAD_DIM,)).to(tl.bfloat16),
    )


def _pick_block_t(compress_rate):
    """Largest power-of-two token tile that divides the window size."""
    for bt in (16, 8, 4, 2, 1):
        if compress_rate % bt == 0:
            return bt, True
    return 8, False


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
    """DeepSeek-V4 HCA compressor, compress_rate == 128, head_dim == 512.

    Supports both the functional call (five inputs, three outputs returned) and
    the destination-passing style call (outputs supplied by the caller and
    written in place; they are also returned).
    """
    orig_device = kv_state.device
    if orig_device.type == "cuda":
        device = orig_device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "hca_compress_c128_h512_r64 requires a CUDA device; none is available."
            )
        device = torch.device("cuda")

    kv = kv_state.to(device=device, non_blocking=True)
    sc = score_state.to(device=device, non_blocking=True)
    gain = rms_norm_weight.to(device=device, non_blocking=True)
    cos = cos_cache.to(device=device, non_blocking=True)
    sin = sin_cache.to(device=device, non_blocking=True)

    if kv.stride(-1) != 1:
        kv = kv.contiguous()
    if sc.stride(-1) != 1:
        sc = sc.contiguous()
    if gain.stride(-1) != 1:
        gain = gain.contiguous()
    if cos.stride(-1) != 1:
        cos = cos.contiguous()
    if sin.stride(-1) != 1:
        sin = sin.contiguous()

    num_compressed = kv.shape[0]
    compress_rate = kv.shape[1] if kv.dim() == 3 else _COMPRESS_RATE
    head_dim = kv.shape[-1]
    nope_head_dim = _NOPE_HEAD_DIM
    num_quant_blocks = nope_head_dim // _QUANT_BLOCK

    def _dest(buf, shape, dtype):
        """Pick a GPU-resident, unit-stride destination buffer."""
        if (
            buf is not None
            and buf.device.type == "cuda"
            and buf.dtype == dtype
            and tuple(buf.shape) == tuple(shape)
            and (buf.numel() == 0 or buf.stride(-1) == 1)
        ):
            return buf, True
        return torch.empty(shape, dtype=dtype, device=device), False

    out_ckv, ckv_inplace = _dest(ckv, (num_compressed, head_dim), torch.bfloat16)
    out_fp8, fp8_inplace = _dest(
        ckv_fp8, (num_compressed, nope_head_dim), torch.float8_e4m3fn
    )
    out_scale, scale_inplace = _dest(
        ckv_scale, (num_compressed, num_quant_blocks), torch.int8
    )

    if num_compressed > 0:
        block_t, even_t = _pick_block_t(compress_rate)
        _hca_compress_c128_kernel[(num_compressed,)](
            kv,
            sc,
            gain,
            cos,
            sin,
            out_ckv,
            out_fp8,
            out_scale,
            kv.stride(0),
            kv.stride(1),
            sc.stride(0),
            sc.stride(1),
            cos.stride(0),
            sin.stride(0),
            out_ckv.stride(0),
            out_fp8.stride(0),
            out_scale.stride(0),
            COMPRESS_RATE=compress_rate,
            HEAD_DIM=head_dim,
            NOPE_HEAD_DIM=nope_head_dim,
            ROPE_HEAD_DIM=_ROPE_HEAD_DIM,
            ROPE_HALF=_ROPE_HALF,
            QUANT_BLOCK=_QUANT_BLOCK,
            NUM_QUANT_BLOCKS=num_quant_blocks,
            HEAD_BLOCKS=head_dim // _QUANT_BLOCK,
            CEIL_THRESH=_CEIL_THRESH,
            BLOCK_T=block_t,
            EVEN_T=even_t,
            EPS=_RMS_NORM_EPS,
            FP8_MAX=_FP8_MAX,
            num_warps=8,
            num_stages=3,
        )

    # Write back into caller-provided destinations when they could not be used
    # directly (wrong device / layout / dtype mismatch).
    if ckv is not None and not ckv_inplace:
        ckv.copy_(out_ckv)
        res_ckv = ckv
    else:
        res_ckv = out_ckv
    if ckv_fp8 is not None and not fp8_inplace:
        ckv_fp8.copy_(out_fp8)
        res_fp8 = ckv_fp8
    else:
        res_fp8 = out_fp8
    if ckv_scale is not None and not scale_inplace:
        ckv_scale.copy_(out_scale)
        res_scale = ckv_scale
    else:
        res_scale = out_scale

    if ckv is None and orig_device.type != "cuda":
        res_ckv = res_ckv.to(orig_device)
        res_fp8 = res_fp8.to(orig_device)
        res_scale = res_scale.to(orig_device)

    return res_ckv, res_fp8, res_scale
