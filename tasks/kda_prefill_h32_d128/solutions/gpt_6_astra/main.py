import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_qkg(
    Q, K, G, Prepared,
    T: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    hb = tl.program_id(0)
    times = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.arange(0, 128)
    batch = hb // 32
    head = hb % 32

    src = (
        (batch * T + times[:, None]) * 4096
        + head * 128
        + channels[None, :]
    )
    valid = times[:, None] < T

    q = tl.load(Q + src, mask=valid, other=0).to(tl.float32)
    k = tl.load(K + src, mask=valid, other=0).to(tl.float32)
    g = tl.load(G + src, mask=valid, other=0)

    q_norm = tl.rsqrt(tl.sum(q * q, axis=1) + 1.0e-6)
    k_norm = tl.rsqrt(tl.sum(k * k, axis=1) + 1.0e-6)
    q = (q * q_norm[:, None]) * 0.08838834764831843
    k = k * k_norm[:, None]
    decay = tl.exp(g)

    dst = (hb * T + times[:, None]) * 128 + channels[None, :]
    tl.store(Prepared + dst, q, mask=valid)
    tl.store(Prepared + TOTAL + dst, k, mask=valid)
    tl.store(Prepared + 2 * TOTAL + dst, decay, mask=valid)


@triton.jit
def _kda_recurrent(
    Q, K, V, G, Beta, Initial, Prepared, Out, Final,
    T: tl.constexpr,
    TOTAL: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    USE_PREPARED: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    hb = tl.program_id(0)
    vb = tl.program_id(1)
    batch = hb // 32
    head = hb % 32

    # Keep the reduction dimension contiguous in registers. Each value
    # channel owns an independent fp32 recurrent state vector.
    channels = tl.arange(0, 128)
    values = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    state_offsets = (
        hb * 16384 + channels[None, :] * 128 + values[:, None]
    )

    if HAS_INITIAL:
        state = tl.load(Initial + state_offsets).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_V, 128), tl.float32)

    token_base = batch * T * 4096 + head * 128
    beta_base = batch * T * 32 + head

    for token in tl.range(0, T):
        base = token_base + token * 4096

        if USE_PREPARED:
            prepared_offsets = (hb * T + token) * 128 + channels
            q = tl.load(Prepared + prepared_offsets)
            k = tl.load(Prepared + TOTAL + prepared_offsets)
            decay = tl.load(Prepared + 2 * TOTAL + prepared_offsets)
        else:
            q = tl.load(Q + base + channels).to(tl.float32)
            k = tl.load(K + base + channels).to(tl.float32)
            log_decay = tl.load(G + base + channels)
            q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
            q = q * 0.08838834764831843
            k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)
            decay = tl.exp(log_decay)

        v = tl.load(V + base + values).to(tl.float32)
        beta = tl.load(Beta + beta_base + token * 32).to(tl.float32)

        state = state * decay[None, :]
        prediction = tl.sum(state * k[None, :], axis=1)
        delta = beta * (v - prediction)
        state = state + delta[:, None] * k[None, :]

        output = tl.sum(state * q[None, :], axis=1)
        tl.store(Out + base + values, output)

    tl.store(Final + state_offsets, state)


def run(query, key, value, g, beta, initial_state):
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "g": g,
        "beta": beta,
    }
    if initial_state is not None:
        tensors["initial_state"] = initial_state

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "kda_prefill_h32_d128 requires CUDA, but CUDA is not available"
        )

    if query.ndim != 4:
        raise ValueError("query must have shape [batch_size, seq_len, 32, 128]")

    batch_size, seq_len, num_heads, head_dim = query.shape
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must both be at least 1")
    if num_heads != 32 or head_dim != 128:
        raise ValueError("num_heads must equal 32 and head_dim must equal 128")

    for name, tensor in (("key", key), ("value", value), ("g", g)):
        if tensor.shape != query.shape:
            raise ValueError(f"{name} must have the same shape as query")

    state_shape = (batch_size, 32, 128, 128)
    if tuple(beta.shape) != (batch_size, seq_len, 32):
        raise ValueError("beta must have shape [batch_size, seq_len, 32]")
    if initial_state is not None and tuple(initial_state.shape) != state_shape:
        raise ValueError(
            "initial_state must have shape [batch_size, 32, 128, 128]"
        )

    for name, tensor in tensors.items():
        expected_dtype = (
            torch.float32 if name in ("g", "initial_state") else torch.bfloat16
        )
        if tensor.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype}")

    output_device = query.device
    state_output_device = (
        initial_state.device if initial_state is not None else output_device
    )

    if query.is_cuda:
        compute_device = query.device
    else:
        compute_device = next(
            (tensor.device for tensor in tensors.values() if tensor.is_cuda),
            None,
        )
        if compute_device is None:
            compute_device = torch.device("cuda", torch.cuda.current_device())

    with torch.cuda.device(compute_device):
        query_gpu = query.cuda(device=compute_device).contiguous()
        key_gpu = key.cuda(device=compute_device).contiguous()
        value_gpu = value.cuda(device=compute_device).contiguous()
        g_gpu = g.cuda(device=compute_device).contiguous()
        beta_gpu = beta.cuda(device=compute_device).contiguous()
        state_gpu = (
            initial_state.cuda(device=compute_device).contiguous()
            if initial_state is not None
            else None
        )

        output = torch.empty(
            query.shape, dtype=torch.bfloat16, device=compute_device
        )
        final_state = torch.empty(
            state_shape, dtype=torch.float32, device=compute_device
        )

        total = batch_size * 32 * seq_len * 128
        use_prepared = seq_len >= 32
        if use_prepared:
            prepared = torch.empty(
                (3, total), dtype=torch.float32, device=compute_device
            )
            _prepare_qkg[(batch_size * 32, triton.cdiv(seq_len, 16))](
                query_gpu,
                key_gpu,
                g_gpu,
                prepared,
                T=seq_len,
                TOTAL=total,
                BLOCK_T=16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            prepared = g_gpu

        block_v = 16
        _kda_recurrent[(batch_size * 32, triton.cdiv(128, block_v))](
            query_gpu,
            key_gpu,
            value_gpu,
            g_gpu,
            beta_gpu,
            state_gpu if state_gpu is not None else final_state,
            prepared,
            output,
            final_state,
            T=seq_len,
            TOTAL=total,
            HAS_INITIAL=state_gpu is not None,
            USE_PREPARED=use_prepared,
            BLOCK_V=block_v,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )

        return (
            output.to(device=output_device),
            final_state.to(device=state_output_device),
        )
