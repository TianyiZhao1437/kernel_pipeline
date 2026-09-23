import torch
import triton
import triton.language as tl

_CHUNK = 64
_HEAD_DIM = 128
_NUM_HEADS = 32
_L2_EPS = 1e-6


@triton.jit
def _l2norm_rows(x, eps: tl.constexpr):
    norm = tl.sqrt(tl.sum(x * x, axis=1) + eps)[:, None]
    return x / norm


@triton.jit
def _kda_prefill_kernel(
    Q, K, V, G, BETA, S_IN, OUT, S_OUT,
    seq_len,
    qk_stride_s, qk_stride_t, qk_stride_h,
    g_stride_s, g_stride_t, g_stride_h,
    beta_stride_s, beta_stride_t,
    o_stride_s, o_stride_t, o_stride_h,
    st_stride_s, st_stride_h,
    NUM_CHUNKS,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    L2_EPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // NUM_HEADS
    h = pid % NUM_HEADS

    d_offs = tl.arange(0, D)
    c_offs = tl.arange(0, C)

    q_base = Q + b * qk_stride_s + h * qk_stride_h
    k_base = K + b * qk_stride_s + h * qk_stride_h
    v_base = V + b * qk_stride_s + h * qk_stride_h
    g_base = G + b * g_stride_s + h * g_stride_h
    beta_base = BETA + b * beta_stride_s + h * beta_stride_t

    o_base = OUT + b * o_stride_s + h * o_stride_h
    s_in_base = S_IN + b * st_stride_s + h * st_stride_h
    s_out_base = S_OUT + b * st_stride_s + h * st_stride_h

    state = tl.load(
        s_in_base + d_offs[:, None] * D + d_offs[None, :]
    ).to(tl.float32)

    tri_mask = c_offs[:, None] >= c_offs[None, :]
    strict_mask = c_offs[:, None] > c_offs[None, :]
    eye_mask = c_offs[:, None] == c_offs[None, :]

    for chunk in range(0, NUM_CHUNKS):
        t0 = chunk * C
        t_offs = t0 + c_offs
        valid_t = t_offs < seq_len

        t_load = tl.where(valid_t, t_offs, seq_len - 1).to(tl.int64)

        q_rows = tl.load(
            q_base + t_load[:, None] * qk_stride_t + d_offs[None, :]
        ).to(tl.float32)
        k_rows = tl.load(
            k_base + t_load[:, None] * qk_stride_t + d_offs[None, :]
        ).to(tl.float32)
        v_rows = tl.load(
            v_base + t_load[:, None] * qk_stride_t + d_offs[None, :]
        ).to(tl.float32)
        g_rows = tl.load(
            g_base + t_load[:, None] * g_stride_t + d_offs[None, :]
        ).to(tl.float32)
        beta_row = tl.load(beta_base + t_load).to(tl.float32)

        valid_2d = valid_t[:, None]
        q_rows = tl.where(valid_2d, q_rows, 0.0)
        k_rows = tl.where(valid_2d, k_rows, 0.0)
        v_rows = tl.where(valid_2d, v_rows, 0.0)
        g_rows = tl.where(valid_2d, g_rows, 0.0)
        beta_row = tl.where(valid_t, beta_row, 0.0)

        q_rows = _l2norm_rows(q_rows, L2_EPS) * SCALE
        k_rows = _l2norm_rows(k_rows, L2_EPS)

        gc = tl.cumsum(g_rows, axis=0)

        g_last = tl.sum(tl.where(c_offs[:, None] == (C - 1), gc, 0.0), axis=0)

        v_beta = v_rows * beta_row[:, None]
        k_beta = k_rows * beta_row[:, None]

        decay = tl.exp(
            tl.where(
                strict_mask[:, :, None],
                gc[:, None, :] - gc[None, :, :],
                float("-inf"),
            )
        )

        a = tl.sum(k_beta[:, None, :] * k_rows[None, :, :] * decay, axis=2)
        a = tl.where(strict_mask, -a, 0.0)

        tmat = a
        for j in tl.static_range(1, C):
            arow = tl.sum(tl.where(c_offs[:, None] == j, tmat, 0.0), axis=0)
            prev = tl.where(c_offs[None, :] < j, tmat, 0.0)
            contrib = tl.sum(arow[None, :] * prev, axis=1)
            new_row = arow + contrib
            tmat = tl.where(c_offs[:, None] == j, new_row[None, :], tmat)
        tmat = tl.where(eye_mask, tmat + 1.0, tmat)

        u = tl.dot(tmat, v_beta)
        kd = tl.dot(tmat, k_beta * tl.exp(gc))

        attn_inter = tl.dot(q_rows * tl.exp(gc), state)

        s_decay = tl.sum(
            tl.where(
                tri_mask[:, :, None],
                q_rows[:, None, :] * k_rows[None, :, :] * decay,
                0.0,
            ),
            axis=2,
        )

        v_prime = tl.dot(kd, state)
        v_new = u - v_prime

        out = attn_inter + tl.dot(s_decay, v_new)

        tl.store(
            o_base + t_offs.to(tl.int64)[:, None] * o_stride_t + d_offs[None, :],
            out.to(OUT.dtype.element_ty),
            mask=valid_t[:, None],
        )

        state = (
            state * tl.exp(g_last)[None, :]
            + tl.dot(
                tl.trans(k_rows * tl.exp(g_last[None, :] - gc)),
                v_new,
            )
        )

    tl.store(
        s_out_base + d_offs[:, None] * D + d_offs[None, :],
        state.to(S_OUT.dtype.element_ty),
    )


def run(query, key, value, g, beta, initial_state):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the KDA prefill Triton kernel")

    orig_device = query.device

    query_c = query.cuda()
    key_c = key.cuda()
    value_c = value.cuda()
    g_c = g.cuda()
    beta_c = beta.cuda()

    if query_c.dtype != torch.bfloat16:
        raise TypeError("query/key/value must be bfloat16")
    if key_c.dtype != torch.bfloat16:
        raise TypeError("query/key/value must be bfloat16")
    if value_c.dtype != torch.bfloat16:
        raise TypeError("query/key/value must be bfloat16")
    if g_c.dtype != torch.float32:
        raise TypeError("g must be float32")

    batch_size, seq_len, num_heads, head_dim = query_c.shape
    if num_heads != _NUM_HEADS or head_dim != _HEAD_DIM:
        raise ValueError("kernel requires num_heads=32 and head_dim=128")
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be >= 1")

    if initial_state is None:
        initial_state_c = torch.zeros(
            (batch_size, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM),
            dtype=torch.float32,
            device="cuda",
        )
    else:
        initial_state_c = initial_state.cuda().to(torch.float32)

    query_c = query_c.contiguous()
    key_c = key_c.contiguous()
    value_c = value_c.contiguous()
    g_c = g_c.contiguous()
    beta_c = beta_c.contiguous()
    initial_state_c = initial_state_c.contiguous()

    core_attn_out = torch.empty(
        (batch_size, seq_len, num_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    final_state = torch.empty(
        (batch_size, num_heads, _HEAD_DIM, _HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
    )

    num_chunks = (seq_len + _CHUNK - 1) // _CHUNK
    grid = (batch_size * num_heads,)

    _kda_prefill_kernel[grid](
        query_c, key_c, value_c, g_c, beta_c, initial_state_c,
        core_attn_out, final_state,
        seq_len,
        query_c.stride(0), query_c.stride(1), query_c.stride(2),
        g_c.stride(0), g_c.stride(1), g_c.stride(2),
        beta_c.stride(0), beta_c.stride(1),
        core_attn_out.stride(0), core_attn_out.stride(1), core_attn_out.stride(2),
        initial_state_c.stride(0), initial_state_c.stride(1),
        num_chunks,
        SCALE=1.0 / (head_dim ** 0.5),
        D=head_dim,
        C=_CHUNK,
        L2_EPS=_L2_EPS,
        NUM_HEADS=_NUM_HEADS,
        num_warps=8,
        num_stages=1,
    )

    return core_attn_out.to(orig_device), final_state.to(orig_device)
