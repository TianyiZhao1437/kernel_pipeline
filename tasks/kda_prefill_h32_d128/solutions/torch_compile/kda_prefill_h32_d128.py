"""The incumbent for kda_prefill_h32_d128: obvious PyTorch, handed to torch.compile.

This is not a kernel. It is the straightforward expression of the Definition,
compiled with no tuning -- no ``mode=``, no ``options=``, no ``dynamic=`` hint --
because that is what someone reaches for before writing a kernel, and a
solution that cannot beat it has not earned its complexity.

Two deviations from the Definition's reference, both deliberate.

**The UT transform is a triangular solve, not a 63-step loop.** The reference
builds the WY matrix by forward substitution, mutating ``attn`` in place across
63 dependent iterations. That is a faithful transcription of the algorithm but
a terrible thing to hand an inductor: the loop would unroll into 63 stages per
chunk, and the outer chunk loop would multiply that by 256 at the longest
workload. The forward substitution computes exactly ``T = (I - A)^-1`` for the
strictly-lower ``A``, which ``torch.linalg.solve_triangular`` does in one call.
Verified rather than assumed: the residual ``||(I - A)T - I||`` is 4e-8
relative to ``|T|`` on random matrices at the scale this op produces.

**The chunk loop stays in eager.** It is sequential and its trip count varies
with seq_len, so compiling it would either unroll to a graph proportional to
the sequence or force a recompile per workload. Everything before it -- the
normalisation, the padding, the cumulative decay, the decay mask, the WY
construction -- is shape-polymorphic and compiles as one region. That split is
also the honest one to benchmark: it is what the incumbent actually looks like.

What this does NOT do is the thing the task is about. It materialises the full
decay mask, ``[B, H, NC, C, C, D]``, which is 50 GiB and 149 ms at the largest
workload, and it round-trips every intermediate through DRAM. A real kernel
keeps the chunk in registers and never writes the mask at all.
"""

import torch
import torch.nn.functional as F

_CHUNK = 64
_L2_EPS = 1e-6


def _l2norm(x):
    """FLA's l2norm: +eps inside the sqrt, not max(norm, eps)."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + _L2_EPS)


def _prepare(query, key, value, g, beta, chunk: int):
    """Everything up to the sequential chunk loop. Shape-polymorphic, compilable."""
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query)
    key = _l2norm(key)
    seq_len = key.shape[2]
    k_head_dim = key.shape[-1]
    scale = 1.0 / (k_head_dim ** 0.5)

    pad_size = (chunk - seq_len % chunk) % chunk
    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]
    g = g.cumsum(dim=-2)

    strict = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=query.device), diagonal=1)
    decay_mask = ((g.unsqueeze(-2) - g.unsqueeze(-3))
                  .masked_fill(strict[..., None], float("-inf")).exp())

    # Strictly lower; the reference reaches the same matrix via masked_fill on
    # the unit-upper mask, which is the same thing written the other way.
    a = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1)
    a = a.tril(-1)

    eye = torch.eye(chunk, dtype=a.dtype, device=a.device).expand_as(a)
    attn = torch.linalg.solve_triangular(eye - a, eye, upper=False, unitriangular=True)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())
    return query, key, value, g, decay_mask, k_cumdecay, pad_size


def run(query, key, value, g, beta, initial_state):
    out_dtype = query.dtype
    batch_size = query.shape[0]
    seq_len = query.shape[1]

    prepared = _compiled_prepare(query, key, value, g, beta, _CHUNK)
    query, key, value, g, decay_mask, k_cumdecay, pad_size = prepared

    num_heads = query.shape[1]
    k_head_dim = query.shape[-1]
    v_head_dim = value.shape[-1]

    if initial_state is None:
        state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                            dtype=torch.float32, device=value.device)
    else:
        state = initial_state.to(torch.float32).clone()

    core_attn_out = torch.zeros_like(value)
    causal = torch.triu(
        torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=1
    )
    for i in range(query.shape[2]):
        q_i = query[:, :, i]; k_i = key[:, :, i]; v_i = value[:, :, i]; g_i = g[:, :, i]
        attn_inter = (q_i * g_i.exp()) @ state
        attn_intra = ((q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i])
                      .sum(dim=-1).masked_fill(causal, 0))
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        state = (state * g_i[:, :, -1].exp().unsqueeze(-1)
                 + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new)

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(out_dtype)
    return core_attn_out, state


_compiled_prepare = torch.compile(_prepare)
