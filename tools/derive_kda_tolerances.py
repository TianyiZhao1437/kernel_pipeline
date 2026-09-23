"""Derive the eval_config numbers for kda_prefill_h32_d128 by measurement.

Mirrors the procedure tasks/hca_compress_c128/eval_config.yaml documents: a
usable tolerance must accept every implementation that is mathematically the
same operator, and reject every implementation that is not. Both halves are
measured on the real blob corpus rather than on fresh synthetic inputs, because
the corpus is what the benchmark will actually run.

Lower bound -- what a correct kernel may differ by
--------------------------------------------------

The reference is a chunked implementation. ``transformers`` also ships an O(T)
recurrent one, ``recurrent_kimi_delta_attention``, which computes the same
function in a completely different order: T sequential rank-1 updates against
the chunked algorithm's blocked matmuls and UT transform. Neither is more
correct than the other, so the disagreement between them is a floor on what any
correct kernel may show, and it is a floor measured against a real second
implementation rather than guessed from an error model.

Upper bound -- what must be rejected
------------------------------------

Nine semantic mutations of the reference, each a plausible way to get KDA
subtly wrong rather than an arbitrary perturbation:

    local       drop the cross-chunk state entirely (chunk-local attention)
    no_l2norm   skip the l2 normalisation of q and k
    no_cumsum   use per-token g instead of its cumulative sum
    no_kbeta    forget the beta weighting on k in the WY construction
    no_vprime   drop the v_prime correction, v_new = v_i
    no_ut       omit the UT transform's forward substitution
    decay_dir   use exp(g) instead of exp(g_last - g) in the state update
    no_tail     process only whole chunks, dropping the ragged tail
    bf16_accum  accumulate the state in bf16 instead of fp32

``bf16_accum`` is the interesting one and it is not a mutation in the same sense
as the other eight: it computes the right function with less precision. Whether
to accept it is a judgement about what this task is for, not something the
measurement settles, so it is reported separately and the decision is recorded
in eval_config.yaml rather than hidden in a threshold.

The mutations are applied by flags on a parameterised copy of the reference,
not by patching its source text. To keep that copy honest, the tool first
asserts that with every flag off it agrees BITWISE with the Definition's
reference on the corpus. A drifted copy would make every mutation measure the
drift instead of the mutation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
TASK = REPO / "tasks" / "kda_prefill_h32_d128"
ROOT = REPO / "data" / "trace_sets" / "kda_prefill_h32_d128"

_CHUNK = 64
_L2_EPS = 1e-6

MUTATIONS = (
    "local", "no_l2norm", "no_cumsum", "no_kbeta", "no_vprime",
    "no_ut", "decay_dir", "no_tail", "bf16_accum", "bf16_matmul",
    "bf16_wy", "bf16_state",
)


def _mm(a, b, bf16):
    """Matmul, optionally with bf16 operands and fp32 accumulation.

    This is what a tensor-core kernel actually does: the inputs arrive as bf16,
    feeding them to the MMA at full fp32 width would throw away the hardware's
    reason for existing, and torch accumulates bf16 MMAs in fp32. The reference
    upcasts everything to fp32 first, so if the tolerance is derived only from
    fp32-vs-fp32 comparisons it can end up rejecting the one implementation
    strategy the task is asking for.
    """
    if not bf16:
        return a @ b
    return (a.bfloat16() @ b.bfloat16()).float()


def _l2norm(x):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + _L2_EPS)


def run_variant(query, key, value, g, beta, initial_state, *, mutation=None):
    """The reference, with one optional semantic mutation.

    Kept line-for-line parallel to the Definition's reference so the diff a
    mutation makes is visible; verified bitwise-identical when mutation is None.
    """
    m = mutation
    out_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    if m != "no_l2norm":
        query = _l2norm(query)
        key = _l2norm(key)
    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (k_head_dim ** 0.5)

    if m == "no_tail":
        # Truncate to whole chunks instead of padding: a kernel that forgets the
        # ragged tail. Output is then short, so pad it back with zeros to keep
        # the comparison shape-compatible -- that is what such a kernel would
        # hand back if it wrote into a preallocated output buffer.
        keep = (seq_len // _CHUNK) * _CHUNK
        pad_size = 0
        query, key, value, g, beta = (
            query[:, :, :keep], key[:, :, :keep], value[:, :, :keep],
            g[:, :, :keep], beta[:, :, :keep],
        )
        dropped = seq_len - keep
        seq_len_eff = keep
    else:
        pad_size = (_CHUNK - seq_len % _CHUNK) % _CHUNK
        dropped = 0
        seq_len_eff = seq_len

    padded_len = seq_len_eff + pad_size
    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key if m == "no_kbeta" else key * beta.unsqueeze(-1)
    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, _CHUNK, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]
    if m != "no_cumsum":
        g = g.cumsum(dim=-2)

    eye_mask = torch.triu(torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=0)
    strict_mask = torch.triu(torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=1)
    decay_mask = ((g.unsqueeze(-2) - g.unsqueeze(-3))
                  .masked_fill(strict_mask[..., None], float("-inf")).exp())
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1)
    attn = attn.masked_fill(eye_mask, 0)
    if m != "no_ut":
        for i in range(1, _CHUNK):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(_CHUNK, dtype=attn.dtype, device=attn.device)
    # Two independent halves, so the precision loss can be attributed rather
    # than guessed: the WY products consume `attn`, whose entries grow through
    # the UT forward substitution, while the state-side products consume the
    # fp32 running state.
    bf16_wy = m in ("bf16_matmul", "bf16_wy")
    bf16_state = m in ("bf16_matmul", "bf16_state")
    value = _mm(attn, v_beta, bf16_wy)
    k_cumdecay = _mm(attn, k_beta * g.exp(), bf16_wy)

    state_dtype = torch.bfloat16 if m == "bf16_accum" else torch.float32
    if initial_state is None:
        state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                            dtype=state_dtype, device=value.device)
    else:
        state = initial_state.to(state_dtype).clone()

    core_attn_out = torch.zeros_like(value)
    causal_mask = torch.triu(torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(padded_len // _CHUNK):
        q_i = query[:, :, i]; k_i = key[:, :, i]; v_i = value[:, :, i]; g_i = g[:, :, i]
        if m == "local":
            # A kernel that never carries state across chunks.
            state = torch.zeros_like(state)
        attn_inter = _mm(q_i * g_i.exp(), state.float(), bf16_state)
        attn_intra = ((q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i])
                      .sum(dim=-1).masked_fill(causal_mask, 0))
        v_prime = _mm(k_cumdecay[:, :, i], state.float(), bf16_state)
        v_new = v_i if m == "no_vprime" else v_i - v_prime
        core_attn_out[:, :, i] = attn_inter + _mm(attn_intra, v_new, bf16_state)
        if m == "decay_dir":
            decay = (g_i).exp()
        else:
            decay = (g_i[:, :, -1:] - g_i).exp()
        update = (state.float() * g_i[:, :, -1].exp().unsqueeze(-1)
                  + _mm((k_i * decay).transpose(-1, -2), v_new, bf16_state))
        state = update.to(state_dtype)

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :seq_len_eff]
    if dropped:
        core_attn_out = F.pad(core_attn_out, (0, 0, 0, dropped))
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(out_dtype)
    return core_attn_out, state.float()


def error_stats(got, want):
    """rel/abs error and matched_ratio the way flashinfer_bench computes them."""
    got = got.float()
    want = want.float()
    diff = (got - want).abs()
    max_abs = float(diff.max())
    max_rel = float((diff / want.abs().clamp_min(1e-6)).max())
    return max_abs, max_rel


def matched_ratio(got, want, rtol, atol):
    """Exactly flashinfer_bench.bench.utils.compute_error_stats.

    Transcribed rather than approximated, because the combination rule is the
    single most consequential line in this file and it is not the intuitive
    one. An element FAILS only if it exceeds BOTH bounds:

        exceeds = (abs_error > atol) & (rel_error > rtol)

    so passing is an OR, not an AND. An earlier version of this function
    required both bounds to pass, which made every tolerance look far tighter
    than it is and made a bf16 tensor-core kernel look unacceptable. Note also
    that rel_error divides by ``|y| + 1e-8``, not by a clamped denominator.
    """
    x = got.float()
    y = want.float()
    abs_error = (x - y).abs()
    rel_error = abs_error / (y.abs() + 1e-8)
    exceeds = (abs_error > atol) & (rel_error > rtol)
    return 1.0 - float(exceeds.sum()) / exceeds.numel()


def load_corpus(limit=None, only=None):
    from safetensors.torch import load_file

    lines = (TASK / "kda_prefill_h32_d128.jsonl").read_text().splitlines()
    traces = [json.loads(line) for line in lines if line.strip()]
    if only:
        traces = [t for t in traces
                  if (t["workload"]["axes"]["batch_size"], t["workload"]["axes"]["seq_len"]) in only]
    if limit:
        traces = traces[:limit]
    for trace in traces:
        workload = trace["workload"]
        descriptors = workload["inputs"]
        paths = {d["path"] for d in descriptors.values()}
        assert len(paths) == 1, "expected one blob per workload"
        blob = load_file(str(ROOT / paths.pop()))
        inputs = {name: blob[d["tensor_key"]].cuda() for name, d in descriptors.items()}
        yield workload["axes"], inputs


def check_fidelity(reference, corpus):
    """The parameterised copy must equal the Definition's reference bitwise."""
    print("=== run_variant(mutation=None) vs the Definition's reference ===")
    worst = 0.0
    for axes, inputs in corpus:
        a_out, a_state = reference(**inputs)
        b_out, b_state = run_variant(**inputs, mutation=None)
        d = max(float((a_out.float() - b_out.float()).abs().max()),
                float((a_state - b_state).abs().max()))
        worst = max(worst, d)
        flag = "" if d == 0.0 else "  <-- DRIFT"
        print(f"    B={axes['batch_size']:>2} T={axes['seq_len']:>6}  max_abs={d:.3e}{flag}")
    print(f"    worst {worst:.3e} -> {'BITWISE IDENTICAL' if worst == 0.0 else 'DRIFTED, mutations are not trustworthy'}")
    return worst == 0.0


def check_recurrent(reference, corpus):
    """Floor: the reference against an independent O(T) implementation."""
    from transformers.models.kimi_linear.modeling_kimi_linear import (
        recurrent_kimi_delta_attention,
    )

    print("\n=== floor: chunked reference vs recurrent_kimi_delta_attention ===")
    print(f"    {'B':>2} {'T':>6}  {'out max_abs':>12} {'out max_rel':>12} "
          f"{'st max_abs':>12} {'st max_rel':>12}")
    worst = {"out_abs": 0.0, "out_rel": 0.0, "st_abs": 0.0, "st_rel": 0.0}
    rows = []
    for axes, inputs in corpus:
        out, state = reference(**inputs)
        r_out, r_state = recurrent_kimi_delta_attention(
            inputs["query"], inputs["key"], inputs["value"],
            inputs["g"], inputs["beta"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        oa, orl = error_stats(out, r_out)
        sa, srl = error_stats(state, r_state)
        worst["out_abs"] = max(worst["out_abs"], oa)
        worst["out_rel"] = max(worst["out_rel"], orl)
        worst["st_abs"] = max(worst["st_abs"], sa)
        worst["st_rel"] = max(worst["st_rel"], srl)
        rows.append((axes, out, r_out, state, r_state))
        print(f"    {axes['batch_size']:>2} {axes['seq_len']:>6}  "
              f"{oa:>12.3e} {orl:>12.3e} {sa:>12.3e} {srl:>12.3e}")
    print(f"    worst: out {worst['out_abs']:.3e} abs / {worst['out_rel']:.3e} rel;  "
          f"state {worst['st_abs']:.3e} abs / {worst['st_rel']:.3e} rel")

    print("\n    matched_ratio of the recurrent implementation at candidate bounds:")
    print(f"      {'rtol':>8} {'atol':>8}  {'worst out':>12} {'worst state':>12}")
    for rtol, atol in ((2e-2, 1e-3), (2e-2, 5e-4), (2e-2, 1e-4), (1e-2, 1e-4)):
        wo = min(matched_ratio(o, r, rtol, atol) for _, o, r, _, _ in rows)
        ws = min(matched_ratio(s, r, rtol, atol) for _, _, _, s, r in rows)
        print(f"      {rtol:>8.0e} {atol:>8.0e}  {wo:>12.6f} {ws:>12.6f}")
    return worst


def check_mutations(reference, corpus, rtol, atol):
    """Ceiling: every mutation must be rejected at the chosen bounds."""
    print(f"\n=== ceiling: mutations at rtol={rtol:.0e} atol={atol:.0e} ===")
    print(f"    {'mutation':<12} {'out ratio':>10} {'state ratio':>12} "
          f"{'out max_rel':>12} {'||d||/||x||':>12}")
    results = {}
    corpus = list(corpus)
    for mutation in MUTATIONS:
        ratios_out, ratios_state, rels, norms = [], [], [], []
        for axes, inputs in corpus:
            want_out, want_state = reference(**inputs)
            got_out, got_state = run_variant(**inputs, mutation=mutation)
            ratios_out.append(matched_ratio(got_out, want_out, rtol, atol))
            ratios_state.append(matched_ratio(got_state, want_state, rtol, atol))
            rels.append(error_stats(got_out, want_out)[1])
            a, b = want_out.float(), got_out.float()
            norms.append(float((a - b).norm() / a.norm()))
        results[mutation] = {
            "out_ratio": min(ratios_out),
            "state_ratio": min(ratios_state),
            "max_rel": max(rels),
            "norm_miss": max(norms),
        }
        r = results[mutation]
        print(f"    {mutation:<12} {r['out_ratio']:>10.6f} {r['state_ratio']:>12.6f} "
              f"{r['max_rel']:>12.3e} {r['norm_miss']:>12.3e}")
    return results


def time_reference(reference, corpus):
    print("\n=== reference latency (sets iterations) ===")
    print(f"    {'B':>2} {'T':>6}  {'ms':>9} {'peak GiB':>9}")
    total = 0.0
    for axes, inputs in corpus:
        torch.cuda.reset_peak_memory_stats()
        reference(**inputs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            reference(**inputs)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - started) / 3 * 1e3
        peak = torch.cuda.max_memory_allocated() / 2**30
        total += ms
        print(f"    {axes['batch_size']:>2} {axes['seq_len']:>6}  {ms:>9.2f} {peak:>9.2f}")
    print(f"    sum over the sweep: {total:.0f} ms per iteration of all workloads")
    return total



def check_separation(reference, corpus, grid=None):
    """Find an (rtol, atol) box that admits precision variants and rejects mutations.

    atol cannot be chosen without knowing the output's scale: core_attn_out is
    small, so an atol that looks tiny in absolute terms can still be a large
    fraction of a typical element, and the library's rule lets an element pass
    on atol ALONE. That is how no_ut scored 0.9994 on the output at atol=1e-2.

    The verdict is per output and a trace fails if any output exceeds, so the
    figure that decides a mutation's fate is the MIN over core_attn_out and
    final_state -- which is what this prints.
    """
    corpus = list(corpus)
    print("\n=== output scale ===")
    for axes, inputs in corpus:
        out, state = reference(**inputs)
        o, s = out.float(), state.float()
        print(f"    B={axes['batch_size']:>2} T={axes['seq_len']:>6}  "
              f"out rms={float(o.pow(2).mean().sqrt()):.4e} absmax={float(o.abs().max()):.4e} | "
              f"state rms={float(s.pow(2).mean().sqrt()):.4e} absmax={float(s.abs().max()):.4e}")

    precision = ("bf16_accum", "bf16_matmul", "bf16_wy", "bf16_state")
    semantic = [m for m in MUTATIONS if m not in precision]

    grid = grid or [(1e-2, 1e-2), (2e-2, 1e-3), (2e-2, 5e-4), (2e-2, 3e-4),
                    (2e-2, 1e-4), (5e-2, 1e-4), (1e-2, 1e-4)]
    want = {}
    for axes, inputs in corpus:
        want[id(inputs)] = reference(**inputs)

    print("\n=== separation: min ratio over BOTH outputs ===")
    print(f"    {'rtol':>7} {'atol':>7} | {'worst precision':>15} | {'best semantic':>13} "
          f"| {'which':>10} | {'verdict':>8}")
    for rtol, atol in grid:
        scores = {}
        for mutation in MUTATIONS:
            worst = 1.0
            for axes, inputs in corpus:
                w_out, w_state = want[id(inputs)]
                g_out, g_state = run_variant(**inputs, mutation=mutation)
                worst = min(worst,
                            matched_ratio(g_out, w_out, rtol, atol),
                            matched_ratio(g_state, w_state, rtol, atol))
            scores[mutation] = worst
        worst_precision = min(scores[m] for m in precision)
        best_semantic = max(scores[m] for m in semantic)
        which = max(semantic, key=lambda m: scores[m])
        # A usable box needs a threshold strictly between the two.
        verdict = "OK" if worst_precision > best_semantic else "NO GAP"
        print(f"    {rtol:>7.0e} {atol:>7.0e} | {worst_precision:>15.6f} | "
              f"{best_semantic:>13.6f} | {which:>10} | {verdict:>8}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N workloads (the recurrent check is O(T) and slow)")
    ap.add_argument("--skip", nargs="*", default=("fidelity", "recurrent", "mutations", "timing"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--only", nargs="*", default=None,
                    help="stages to run: fidelity recurrent mutations separation timing")
    ap.add_argument("--shapes", nargs="*", default=None, metavar="BxT",
                    help="restrict to these workloads, e.g. 1x100 1x3000 2x8192")
    args = ap.parse_args()

    shapes = None
    if args.shapes:
        shapes = set()
        for item in args.shapes:
            b, _, t = item.partition("x")
            shapes.add((int(b), int(t)))

    def corpus():
        return load_corpus(limit=args.limit, only=shapes)

    sys.path.insert(0, str(REPO / "tools"))
    from check_kda_numerics import load_reference
    reference = load_reference(TASK / "kda_prefill_h32_d128.json")

    stages = args.only or ["fidelity", "recurrent", "mutations", "timing"]
    ok = True
    if "fidelity" in stages:
        ok &= check_fidelity(reference, corpus())
        if not ok:
            print("\nrun_variant has drifted from the reference; fix that before reading any "
                  "mutation result below.")
            return 1
    if "recurrent" in stages:
        check_recurrent(reference, corpus())
    if "mutations" in stages:
        check_mutations(reference, corpus(), args.rtol, args.atol)
    if "separation" in stages:
        check_separation(reference, corpus())
    if "timing" in stages:
        time_reference(reference, corpus())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
