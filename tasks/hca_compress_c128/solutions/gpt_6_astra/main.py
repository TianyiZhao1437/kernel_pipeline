import torch
import triton
import triton.language as tl


@triton.jit
def _compress_windows(
    KV,
    SCORE,
    COMPRESSED,
    KV_S0: tl.constexpr,
    KV_S1: tl.constexpr,
    KV_S2: tl.constexpr,
    SCORE_S0: tl.constexpr,
    SCORE_S1: tl.constexpr,
    SCORE_S2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    window = tl.program_id(0)
    tile = tl.program_id(1)

    tokens = tl.arange(0, 128)
    dims = tile * BLOCK_D + tl.arange(0, BLOCK_D)

    scores = tl.load(
        SCORE
        + window * SCORE_S0
        + tokens[:, None] * SCORE_S1
        + dims[None, :] * SCORE_S2
    ).to(tl.float32)

    scores = scores - tl.max(scores, axis=0)[None, :]
    weights = tl.exp(scores)
    weights = weights / tl.sum(weights, axis=0)[None, :]

    values = tl.load(
        KV
        + window * KV_S0
        + tokens[:, None] * KV_S1
        + dims[None, :] * KV_S2
    ).to(tl.float32)

    compressed = tl.sum(values * weights, axis=0)
    tl.store(COMPRESSED + window * 512 + dims, compressed)


@triton.jit
def _normalize_rotate_quantize(
    COMPRESSED,
    GAIN,
    COS,
    SIN,
    CKV,
    FP8,
    SCALE,
    GAIN_S0: tl.constexpr,
    COS_S0: tl.constexpr,
    COS_S1: tl.constexpr,
    SIN_S0: tl.constexpr,
    SIN_S1: tl.constexpr,
    CKV_S0: tl.constexpr,
    CKV_S1: tl.constexpr,
    FP8_S0: tl.constexpr,
    FP8_S1: tl.constexpr,
    SCALE_S0: tl.constexpr,
    SCALE_S1: tl.constexpr,
):
    window = tl.program_id(0)
    dims = tl.arange(0, 512)

    compressed = tl.load(COMPRESSED + window * 512 + dims)
    variance = tl.sum(compressed * compressed, axis=0) * (1.0 / 512.0)
    inverse_rms = tl.rsqrt(variance + 1.0e-6)
    gain = tl.load(GAIN + dims * GAIN_S0).to(tl.float32)
    normed = (compressed * inverse_rms) * gain

    rounded = normed.to(tl.bfloat16).to(tl.float32)

    partner = tl.gather(normed, dims ^ 1, axis=0)
    is_rope = dims >= 448
    pair = tl.maximum((dims - 448) // 2, 0)
    position = window * 128

    cosine = tl.load(
        COS + position * COS_S0 + pair * COS_S1,
        mask=is_rope,
        other=1.0,
    )
    sine = tl.load(
        SIN + position * SIN_S0 + pair * SIN_S1,
        mask=is_rope,
        other=0.0,
    )

    signed_partner = tl.where((dims & 1) == 0, -partner, partner)
    rotated = normed * cosine + signed_partner * sine
    output = tl.where(is_rope, rotated, rounded)
    tl.store(
        CKV + window * CKV_S0 + dims * CKV_S1,
        output.to(tl.bfloat16),
    )

    blocks = tl.reshape(rounded, (8, 64))
    absmax = tl.maximum(tl.max(tl.abs(blocks), axis=1), 1.0e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    exponent = tl.minimum(tl.maximum(exponent, -127.0), 127.0)

    scaled = blocks * tl.exp2(-exponent)[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    encoded = tl.reshape(scaled, (512,)).to(tl.float8e4nv)

    tl.store(
        FP8 + window * FP8_S0 + dims * FP8_S1,
        encoded,
        mask=dims < 448,
    )

    block_ids = tl.arange(0, 8)
    tl.store(
        SCALE + window * SCALE_S0 + block_ids * SCALE_S1,
        exponent.to(tl.int8),
        mask=block_ids < 7,
    )


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
    """Support eight-argument destination passing and five-argument allocation."""
    inputs = (
        kv_state,
        score_state,
        rms_norm_weight,
        cos_cache,
        sin_cache,
    )
    input_names = (
        "kv_state",
        "score_state",
        "rms_norm_weight",
        "cos_cache",
        "sin_cache",
    )
    input_dtypes = (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
    )

    for name, tensor, dtype in zip(input_names, inputs, input_dtypes):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must be on a CPU or CUDA device")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")

    if kv_state.ndim != 3 or tuple(kv_state.shape[1:]) != (128, 512):
        raise ValueError("kv_state must have shape [num_compressed, 128, 512]")
    if score_state.shape != kv_state.shape:
        raise ValueError("score_state must have the same shape as kv_state")
    if tuple(rms_norm_weight.shape) != (512,):
        raise ValueError("rms_norm_weight must have shape [512]")
    if cos_cache.ndim != 2 or cos_cache.shape[1] != 32:
        raise ValueError("cos_cache must have shape [max_position, 32]")
    if sin_cache.shape != cos_cache.shape:
        raise ValueError("sin_cache must have the same shape as cos_cache")

    num_compressed = kv_state.shape[0]
    if num_compressed and cos_cache.shape[0] <= (num_compressed - 1) * 128:
        raise ValueError("RoPE caches must contain row (num_compressed - 1) * 128")

    supplied_outputs = (ckv, ckv_fp8, ckv_scale)
    destination_passing = any(t is not None for t in supplied_outputs)
    if destination_passing and any(t is None for t in supplied_outputs):
        raise ValueError("Provide all three output tensors or omit all three")

    output_shapes = (
        (num_compressed, 512),
        (num_compressed, 448),
        (num_compressed, 7),
    )
    output_dtypes = (torch.bfloat16, torch.float8_e4m3fn, torch.int8)
    output_names = ("ckv", "ckv_fp8", "ckv_scale")

    if destination_passing:
        for name, tensor, shape, dtype in zip(
            output_names, supplied_outputs, output_shapes, output_dtypes
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(f"{name} must be on a CPU or CUDA device")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must have dtype {dtype}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "hca_compress_c128_h512_r64 requires an available CUDA GPU; "
            "CPU tensors also require CUDA for computation."
        )

    original_device = kv_state.device
    if original_device.type == "cuda":
        compute_device = original_device
    else:
        candidates = inputs + tuple(
            t for t in supplied_outputs if t is not None
        )
        compute_device = next(
            (t.device for t in candidates if t.is_cuda),
            torch.device("cuda", torch.cuda.current_device()),
        )

    outputs = (
        supplied_outputs
        if destination_passing
        else tuple(
            torch.empty(shape, dtype=dtype, device=original_device)
            for shape, dtype in zip(output_shapes, output_dtypes)
        )
    )

    if num_compressed == 0:
        return None if destination_passing else outputs

    with torch.cuda.device(compute_device):
        kv, score, gain, cosine, sine = tuple(
            tensor
            if tensor.device == compute_device
            else tensor.cuda(device=compute_device)
            for tensor in inputs
        )

        gpu_outputs = tuple(
            tensor
            if tensor.device == compute_device
            else torch.empty(shape, dtype=dtype, device=compute_device)
            for tensor, shape, dtype in zip(
                outputs, output_shapes, output_dtypes
            )
        )
        gpu_ckv, gpu_fp8, gpu_scale = gpu_outputs

        compressed = torch.empty(
            (num_compressed, 512),
            dtype=torch.float32,
            device=compute_device,
        )

        block_d = 32 if num_compressed < 32 else 64
        _compress_windows[(num_compressed, triton.cdiv(512, block_d))](
            kv,
            score,
            compressed,
            *kv.stride(),
            *score.stride(),
            BLOCK_D=block_d,
            num_warps=4,
            enable_fp_fusion=False,
        )

        _normalize_rotate_quantize[(num_compressed,)](
            compressed,
            gain,
            cosine,
            sine,
            gpu_ckv,
            gpu_fp8,
            gpu_scale,
            gain.stride(0),
            cosine.stride(0),
            cosine.stride(1),
            sine.stride(0),
            sine.stride(1),
            gpu_ckv.stride(0),
            gpu_ckv.stride(1),
            gpu_fp8.stride(0),
            gpu_fp8.stride(1),
            gpu_scale.stride(0),
            gpu_scale.stride(1),
            num_warps=4,
            enable_fp_fusion=False,
        )

        for destination, result in zip(outputs, gpu_outputs):
            if destination is not result:
                destination.copy_(result, non_blocking=False)

    return None if destination_passing else outputs
