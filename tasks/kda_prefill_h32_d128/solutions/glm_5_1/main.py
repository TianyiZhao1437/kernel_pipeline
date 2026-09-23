import torch
import triton
import triton.language as tl
import math

_CHUNK = 64
_HEAD_DIM = 128
_NUM_HEADS = 32
_L2_EPS = 1e-6


@triton.jit
def _wy_fwd_kernel(
    k_beta_ptr, key_ptr, v_beta_ptr, g_cs_ptr,
    vo_ptr, kcd_ptr,
    stride_bh, stride_c, stride_s, stride_d,
    CHUNK: tl.constexpr,
    D: tl.constexpr,
    BD_F: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_bh * stride_bh + pid_c * stride_c
    s_offs = tl.arange(0, CHUNK)

    lt_mask = (s_offs[:, None] > tl.arange(0, CHUNK)[None, :]).to(tl.float32)

    L = tl.zeros([CHUNK, CHUNK], dtype=tl.float32)
    for d in range(D):
        offs = s_offs * stride_s + d * stride_d
        kb_d = tl.load(k_beta_ptr + base + offs)
        ki_d = tl.load(key_ptr + base + offs)
        gc_d = tl.load(g_cs_ptr + base + offs)

        diff = gc_d[:, None] - gc_d[None, :]
        safe_decay = tl.exp(diff * lt_mask) * lt_mask
        L += kb_d[:, None] * ki_d[None, :] * safe_decay

    for d0 in range(0, D, BD_F):
        d_offs = d0 + tl.arange(0, BD_F)
        m = s_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        vb = tl.load(v_beta_ptr + base + m)
        kb = tl.load(k_beta_ptr + base + m)
        gc = tl.load(g_cs_ptr + base + m)

        rhs_k = kb * tl.exp(gc)
        vo = tl.zeros([CHUNK, BD_F], dtype=tl.float32)
        kcd = tl.zeros([CHUNK, BD_F], dtype=tl.float32)

        for i_step in range(CHUNK):
            row_onehot = (s_offs == i_step).to(tl.float32)
            L_row = tl.sum(L * row_onehot[:, None], axis=0)

            cv = tl.sum(L_row[:, None] * vo, axis=0)
            ck = tl.sum(L_row[:, None] * kcd, axis=0)

            step_mask = (s_offs == i_step)[:, None]
            vo = tl.where(step_mask, vb - cv[None, :], vo)
            kcd = tl.where(step_mask, rhs_k - ck[None, :], kcd)

        tl.store(vo_ptr + base + m, vo)
        tl.store(kcd_ptr + base + m, kcd)


def run(query, key, value, g, beta, initial_state):
    orig_device = query.device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but this kernel requires a GPU")

    query = query.cuda()
    key = key.cuda()
    value = value.cuda()
    g = g.cuda()
    beta = beta.cuda()
    if initial_state is not None:
        initial_state = initial_state.cuda()

    out_dtype = query.dtype
    B = query.shape[0]
    S = query.shape[1]
    H = _NUM_HEADS
    D = _HEAD_DIM
    CS = _CHUNK

    q = query.transpose(1, 2).contiguous().to(torch.float32)
    k = key.transpose(1, 2).contiguous().to(torch.float32)
    v = value.transpose(1, 2).contiguous().to(torch.float32)
    g32 = g.transpose(1, 2).contiguous().to(torch.float32)
    b32 = beta.transpose(1, 2).contiguous().to(torch.float32)

    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + _L2_EPS)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + _L2_EPS)
    q = q * (1.0 / math.sqrt(D))

    pad = (CS - S % CS) % CS
    S_pad = S + pad
    if pad > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        g32 = torch.nn.functional.pad(g32, (0, 0, 0, pad))
        b32 = torch.nn.functional.pad(b32, (0, pad))

    k_beta = k * b32.unsqueeze(-1)
    v_beta = v * b32.unsqueeze(-1)

    NC = S_pad // CS
    q = q.reshape(B, H, NC, CS, D)
    k = k.reshape(B, H, NC, CS, D)
    g32 = g32.reshape(B, H, NC, CS, D)
    k_beta = k_beta.reshape(B, H, NC, CS, D)
    v_beta = v_beta.reshape(B, H, NC, CS, D)

    g_cs = g32.cumsum(dim=-2)

    BH = B * H
    k_beta_f = k_beta.reshape(BH, NC, CS, D).contiguous()
    key_f = k.reshape(BH, NC, CS, D).contiguous()
    v_beta_f = v_beta.reshape(BH, NC, CS, D).contiguous()
    g_cs_f = g_cs.reshape(BH, NC, CS, D).contiguous()
    vo_f = torch.empty_like(v_beta_f)
    kcd_f = torch.empty_like(k_beta_f)

    BD_F = 32
    grid = (BH, NC)
    _wy_fwd_kernel[grid](
        k_beta_f, key_f, v_beta_f, g_cs_f,
        vo_f, kcd_f,
        k_beta_f.stride(0), k_beta_f.stride(1),
        k_beta_f.stride(2), k_beta_f.stride(3),
        CHUNK=CS, D=D, BD_F=BD_F,
    )

    value_out = vo_f.reshape(B, H, NC, CS, D)
    k_cumdecay = kcd_f.reshape(B, H, NC, CS, D)

    if initial_state is None:
        state = torch.zeros(B, H, D, D, dtype=torch.float32, device=query.device)
    else:
        state = initial_state.to(torch.float32).clone()

    core_out = torch.zeros(B, H, NC, CS, D, dtype=torch.float32, device=q.device)
    causal_upper = torch.triu(torch.ones(CS, CS, dtype=torch.bool, device=q.device), diagonal=1)
    lt_f = (~causal_upper).to(torch.float32)

    for c in range(NC):
        q_c = q[:, :, c]
        k_c = k[:, :, c]
        g_c = g_cs[:, :, c]
        vo_c = value_out[:, :, c]
        kcd_c = k_cumdecay[:, :, c]

        q_exp_g = q_c * torch.exp(g_c)
        attn_inter = torch.matmul(q_exp_g, state)

        v_prime = torch.matmul(kcd_c, state)
        v_new = vo_c - v_prime

        M = torch.zeros(B, H, CS, CS, dtype=torch.float32, device=q.device)
        BD_M = 32
        for d0 in range(0, D, BD_M):
            d_end = min(d0 + BD_M, D)
            d_sl = slice(d0, d_end)
            q_d = q_c[:, :, :, d_sl]
            k_d = k_c[:, :, :, d_sl]
            g_d = g_c[:, :, :, d_sl]
            diff_g = g_d.unsqueeze(-2) - g_d.unsqueeze(-3)
            safe_diff_g = diff_g * lt_f.unsqueeze(-1)
            decay = torch.exp(safe_diff_g) * lt_f.unsqueeze(-1)
            M = M + (q_d.unsqueeze(-2) * k_d.unsqueeze(-3) * decay).sum(-1)

        intra_out = torch.matmul(M, v_new)
        core_out[:, :, c] = attn_inter + intra_out

        g_last = g_c[:, :, -1, :]
        state = state * torch.exp(g_last).unsqueeze(-1)

        g_last_bc = g_c[:, :, -1:, :]
        k_decay = k_c * torch.exp(g_last_bc - g_c)
        state = state + torch.matmul(k_decay.transpose(-2, -1), v_new)

    core_out = core_out.reshape(B, H, S_pad, D)
    core_out = core_out[:, :, :S, :]
    core_out = core_out.transpose(1, 2).contiguous().to(out_dtype)

    if orig_device.type != 'cuda':
        core_out = core_out.to(orig_device)
        state = state.to(orig_device)

    return core_out, state
