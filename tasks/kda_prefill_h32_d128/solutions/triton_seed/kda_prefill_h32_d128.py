"""A hand-written Triton seed for kda_prefill_h32_d128.

This is the correctness anchor, not the performance target. It is not pipelined,
not autotuned, and it holds no tile in SRAM across kernels. Measured, it reaches
**13.1 TFLOP/s** against the ~103 TFLOP/s composite ceiling in
``flops_model.py`` -- 12.7% of what these tile shapes sustain. It is 24x the
Definition's reference and 13x the ``torch.compile`` baseline, so the bar it
sets is real, but 87% of the achievable rate is still on the table.

The one thing it does do properly is never materialise the decay mask. The
Definition's reference builds ``exp(g_i - g_j)`` as a rank-3
``[B, H, NC, C, C, D]`` tensor, which is 50 GiB and 149 ms at the largest
workload. That tensor is the whole reason this task exists, and every design
decision below follows from refusing to write it.

Three fidelity points
---------------------

**1. The factored decay overflows fp32, and it is not a corner case.**

Refusing the rank-3 mask means factoring ``exp(g_i - g_j) = exp(g_i)exp(-g_j)``
so the ``[C, C]`` build collapses to one matmul. That is exact in real
arithmetic and lethal in fp32: ``g`` is a within-chunk cumsum of negative
numbers, so the column factor grows as ``e^|g|``. Measured over the real corpus
(see ``tools/kda_decay_range_probe.py``), the within-chunk cumsum reaches **-111.5**
and **0.29% of all (chunk, channel) pairs go past fp32's e^88.7 ceiling** -- in
every one of the sixteen workloads, not just the long ones. The naive factored
kernel produces ``0 * inf = nan``.

The fix here is to centre the exponent per channel at the midpoint of the
chunk's cumsum range, ``c_d = (g_0d + g_{C-1,d}) / 2``. Both factors are then
bounded by half the range -- ``e^55.7 = 1.5e24``, comfortably finite -- while
their product is unchanged. Centring is free: it is one subtract on a tile that
is being exponentiated anyway.

The strictly-upper half of the resulting matmul still overflows, because there
``g_i > g_j`` and the true value exceeds 1. Those entries are discarded with
``tl.where``, never multiplied by a zero mask. ``where`` selects; a multiply by
0.0 would turn ``inf`` into ``nan`` and the nan would survive into the output.

**2. Padding with zeros is exactly what the reference does, so the tail is free.**

``seq_len`` is not a multiple of 64 in a third of the sweep. Loading out-of-range
tokens as zeros reproduces the reference's ``F.pad`` semantics in every place it
matters: zero ``beta`` kills the rank-1 update, zero ``g`` makes the pad tokens'
decay exactly 1, and a zero ``v_new`` contributes nothing to the state. No
special-cased tail chunk is needed, only masked loads and masked stores.

**3. The UT transform stays sequential.**

``(I - A)^-1`` for strictly-lower ``A`` could be had in 5 doubling steps via
``prod_k (I + A^(2^k))``, which is all tensor-core work. It is also ~5.2 MFLOP
per chunk against the 0.17 MFLOP the forward substitution actually needs -- it
would add 57% to the op's total FLOPs to save a serial loop. This kernel keeps
the substitution, matching the reference term for term.

That decision is where its time goes, and the amount is worth stating because
``flops_model.py`` counts this term at 1.9% of the op's FLOPs. Measured at
B=1 T=16384 by truncating the loop (``tools/kda_seed_split_probe.py``; the
truncated variants compute the wrong answer and are timing probes only):

    UT steps   1     0.970 ms of _prepare   19.1%
    UT steps   8     1.530 ms               30.1%
    UT steps  32     2.987 ms               58.8%
    UT steps  64     5.087 ms              100.0%

So 81% of ``_prepare``, and ``_prepare`` is 89% of the kernel: **1.9% of the
FLOPs is 72% of the latency.** That is the single largest thing a better kernel
can attack here, and the doubling formulation -- 30x the arithmetic, all of it
on tensor cores, none of it serial -- is the obvious trade to try. This seed
does not try it, deliberately: a correctness anchor should be the transcription.

Structure
---------

Two kernels, split on the only real dependency in the operation.

``_prepare`` is parallel over ``(batch, head, chunk)`` and does everything that
is chunk-local: l2norm, the cumsum, the centred ``[C, C]`` builds, the UT
transform, and the two WY products. ``_scan`` is parallel over
``(batch, head, v-block)`` and sequential over chunks, carrying the ``[D, BV]``
state in registers.

The v-split is for occupancy, not elegance: at ``batch_size=1`` a scan
parallelised only over ``(batch, head)`` would fill 32 of the H200's 132 SMs.
``BV = 32`` gives 128 programs there. It costs a 4x reread of the per-chunk
tiles the scan does not split -- about 1.4 GB over the largest workload, ~0.3 ms
at measured bandwidth -- which is the right trade at these batch sizes and the
wrong one at large ``B``. A tuned kernel would pick ``BV`` from the axes.
"""

import torch
import triton
import triton.language as tl

CHUNK = 64
HEAD_DIM = 128
BV = 32


@triton.jit
def _prepare(
    Q, K, V, G, BETA,
    QG, KE, W, KCD, AIN, GLAST,
    seq_len, num_chunks,
    s_qb, s_qt, s_qh,
    s_bb, s_bt,
    s_ob, s_oh, s_oc,
    s_ab, s_ah, s_ac,
    s_gb, s_gh, s_gc,
    C: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
):
    chunk = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    rows = tl.arange(0, C)
    cols = tl.arange(0, D)
    tok = chunk * C + rows
    live = tok < seq_len

    base = b * s_qb + tok[:, None] * s_qt + h * s_qh + cols[None, :]
    m2 = live[:, None]
    q = tl.load(Q + base, mask=m2, other=0.0).to(tl.float32)
    k = tl.load(K + base, mask=m2, other=0.0).to(tl.float32)
    v = tl.load(V + base, mask=m2, other=0.0).to(tl.float32)
    g = tl.load(G + base, mask=m2, other=0.0).to(tl.float32)
    beta = tl.load(BETA + b * s_bb + tok * s_bt + h, mask=live, other=0.0).to(tl.float32)

    # FLA's l2norm: eps inside the sqrt. A zero pad row stays zero.
    q = q / tl.sqrt(tl.sum(q * q, axis=1)[:, None] + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k, axis=1)[:, None] + 1e-6)
    q = q * (1.0 / tl.sqrt(D * 1.0))

    gc = tl.cumsum(g, axis=0)

    # g <= 0 makes gc monotone decreasing, so row 0 is its max and row C-1 its
    # min. Centring on the midpoint halves the exponent range; see fidelity
    # point 1 -- without this, 0.29% of channels overflow.
    g_first = tl.sum(tl.where(rows[:, None] == 0, gc, 0.0), axis=0)
    g_last = tl.sum(tl.where(rows[:, None] == C - 1, gc, 0.0), axis=0)
    centre = 0.5 * (g_first + g_last)

    k_beta = k * beta[:, None]
    v_beta = v * beta[:, None]

    row_k = (k_beta * tl.exp(gc - centre[None, :])).to(tl.bfloat16)
    row_q = (q * tl.exp(gc - centre[None, :])).to(tl.bfloat16)
    col_k = (k * tl.exp(centre[None, :] - gc)).to(tl.bfloat16)
    col_kT = tl.trans(col_k)

    strictly_lower = rows[:, None] > rows[None, :]
    causal = rows[:, None] >= rows[None, :]

    # The upper half of both products can be inf or nan (fidelity point 1).
    # tl.where discards it; a multiply by a 0.0 mask would not.
    a = tl.where(strictly_lower, -tl.dot(row_k, col_kT), 0.0)
    attn_intra = tl.where(causal, tl.dot(row_q, col_kT), 0.0)

    # UT transform by forward substitution, term for term with the reference.
    # Serial in C, which is the point: see fidelity point 3.
    for i in tl.range(1, C):
        r = tl.sum(tl.where(rows[:, None] == i, a, 0.0), axis=0)
        r_masked = tl.where(rows < i, r, 0.0)
        upd = tl.sum(r_masked[:, None] * a, axis=0)
        new_row = r + tl.where(rows < i, upd, 0.0)
        a = tl.where(rows[:, None] == i, new_row[None, :], a)
    t_mat = (a + tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)).to(tl.bfloat16)

    eg = tl.exp(gc)
    w = tl.dot(t_mat, v_beta.to(tl.bfloat16))
    kcd = tl.dot(t_mat, (k_beta * eg).to(tl.bfloat16))

    obase = b * s_ob + h * s_oh + chunk * s_oc + rows[:, None] * D + cols[None, :]
    tl.store(QG + obase, (q * eg).to(QG.dtype.element_ty))
    tl.store(KE + obase, (k * tl.exp(g_last[None, :] - gc)).to(KE.dtype.element_ty))
    tl.store(W + obase, w.to(W.dtype.element_ty))
    tl.store(KCD + obase, kcd.to(KCD.dtype.element_ty))

    abase = b * s_ab + h * s_ah + chunk * s_ac + rows[:, None] * C + rows[None, :]
    tl.store(AIN + abase, attn_intra.to(AIN.dtype.element_ty))
    tl.store(GLAST + b * s_gb + h * s_gh + chunk * s_gc + cols, g_last)


@triton.jit
def _scan(
    QG, KE, W, KCD, AIN, GLAST, STATE0, OUT, STATE1,
    seq_len, num_chunks,
    s_ob, s_oh, s_oc,
    s_ab, s_ah, s_ac,
    s_gb, s_gh, s_gc,
    s_sb, s_sh, s_sd,
    s_yb, s_yt, s_yh,
    C: tl.constexpr, D: tl.constexpr, H: tl.constexpr, BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    b = bh // H
    h = bh % H

    rows = tl.arange(0, C)
    cols = tl.arange(0, D)
    vcols = iv * BLOCK_V + tl.arange(0, BLOCK_V)

    state = tl.load(STATE0 + b * s_sb + h * s_sh + cols[:, None] * s_sd + vcols[None, :]).to(tl.float32)

    for chunk in tl.range(0, num_chunks):
        obase = b * s_ob + h * s_oh + chunk * s_oc + rows[:, None] * D
        vbase = obase + vcols[None, :]
        qg = tl.load(QG + obase + cols[None, :])
        ke = tl.load(KE + obase + cols[None, :])
        kcd = tl.load(KCD + obase + cols[None, :])
        w = tl.load(W + vbase).to(tl.float32)
        ain = tl.load(AIN + b * s_ab + h * s_ah + chunk * s_ac
                      + rows[:, None] * C + rows[None, :])

        st_b = state.to(tl.bfloat16)
        attn_inter = tl.dot(qg, st_b)
        v_prime = tl.dot(kcd, st_b)
        v_new = w - v_prime
        out = attn_inter + tl.dot(ain, v_new.to(tl.bfloat16))

        tok = chunk * C + rows
        tl.store(OUT + b * s_yb + tok[:, None] * s_yt + h * s_yh + vcols[None, :],
                 out.to(OUT.dtype.element_ty), mask=(tok < seq_len)[:, None])

        g_last = tl.load(GLAST + b * s_gb + h * s_gh + chunk * s_gc + cols)
        state = state * tl.exp(g_last)[:, None] + tl.dot(tl.trans(ke), v_new.to(tl.bfloat16))

    tl.store(STATE1 + b * s_sb + h * s_sh + cols[:, None] * s_sd + vcols[None, :], state)


def run(query, key, value, g, beta, initial_state):
    batch, seq_len, num_heads, head_dim = query.shape
    num_chunks = (seq_len + CHUNK - 1) // CHUNK
    dev = query.device

    shape = (batch, num_heads, num_chunks, CHUNK, head_dim)
    qg = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    ke = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    w = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    kcd = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    ain = torch.empty((batch, num_heads, num_chunks, CHUNK, CHUNK),
                      dtype=torch.bfloat16, device=dev)
    glast = torch.empty((batch, num_heads, num_chunks, head_dim),
                        dtype=torch.float32, device=dev)

    _prepare[(num_chunks, batch * num_heads)](
        query, key, value, g, beta,
        qg, ke, w, kcd, ain, glast,
        seq_len, num_chunks,
        query.stride(0), query.stride(1), query.stride(2),
        beta.stride(0), beta.stride(1),
        qg.stride(0), qg.stride(1), qg.stride(2),
        ain.stride(0), ain.stride(1), ain.stride(2),
        glast.stride(0), glast.stride(1), glast.stride(2),
        C=CHUNK, D=head_dim, H=num_heads, num_warps=4, num_stages=2,
    )

    out = torch.empty((batch, seq_len, num_heads, head_dim),
                      dtype=query.dtype, device=dev)
    final_state = torch.empty_like(initial_state, dtype=torch.float32)

    _scan[(batch * num_heads, head_dim // BV)](
        qg, ke, w, kcd, ain, glast, initial_state.float(), out, final_state,
        seq_len, num_chunks,
        qg.stride(0), qg.stride(1), qg.stride(2),
        ain.stride(0), ain.stride(1), ain.stride(2),
        glast.stride(0), glast.stride(1), glast.stride(2),
        initial_state.stride(0), initial_state.stride(1), initial_state.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        C=CHUNK, D=head_dim, H=num_heads, BLOCK_V=BV, num_warps=4, num_stages=2,
    )
    return out, final_state
