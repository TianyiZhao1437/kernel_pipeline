"""Numerical validation for the hca_compress task: reference, solution, tolerances.

Four independent things are established here, in order:

  1. The Definition's reference agrees with an independent transcription of
     vLLM's fused kernel (``vllm_port``), written from that source rather than
     from the reference. This is what makes the reference a *spec* and not just
     an implementation.
  2. The Triton solution agrees with the reference under flashinfer-bench's own
     scoring function (``bench/utils.py::compute_error_stats``, reproduced
     verbatim below).
  3. ckv_fp8 / ckv_scale are a faithful UE8M0 encoding of ckv's noPE part, to
     within half an e4m3 ulp.
  4. The tolerances in eval_config.yaml separate the solution from eight
     targeted semantic mutations of the reference. This is the part that keeps
     the config honest: a tolerance nobody can fail is not a test.

Run with the interpreter that has torch + triton (on this box,
``/venv/tianyi/bin/python3``):

    /venv/tianyi/bin/python3 tools/check_hca_numerics.py
    /venv/tianyi/bin/python3 tools/check_hca_numerics.py --mutations   # (4) too
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import torch
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import fib_shim  # noqa: E402  (needs the path above)

_fib = fib_shim.load_bench_utils()
TASK = REPO / "tasks" / "hca_compress_c128"
DEV = "cuda"
DEF_PATH = TASK / "hca_compress_c128_h512_r64.json"
SOL_PATH = TASK / "solutions" / "triton_h200" / "hca_compress_c128.py"

HEAD_DIM, NOPE, ROPE = 512, 448, 64
CR, QB, EPS, FP8_MAX = 128, 64, 1e-6, 448.0

with open(DEF_PATH) as fh:
    definition = json.load(fh)


ns: dict = {}
exec(compile(definition["reference"], "<reference>", "exec"), ns)
reference_run = ns["run"]

_spec = importlib.util.spec_from_file_location("solution_mod", SOL_PATH)
solution_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solution_mod)


# --------------------------------------------------------------------------
# Independent port of the vLLM kernel.
# --------------------------------------------------------------------------
def vllm_port(kv_state, score_state, rms_weight, cos_cache, sin_cache, total_tokens):
    num_entries = total_tokens // CR
    out = []
    for c in range(num_entries):
        score = score_state[c].to(torch.float32)
        kv = kv_state[c].to(torch.float32)

        score = score - score.amax(dim=0, keepdim=True)
        w = torch.exp(score)
        w = w / w.sum(dim=0, keepdim=True)
        compressed = (kv * w).sum(dim=0)

        var = (compressed * compressed).sum() / HEAD_DIM
        normed = compressed * torch.rsqrt(var + EPS) * rms_weight.to(torch.float32)

        # `quant_input = normed.to(tl.bfloat16).to(tl.float32)`: the kernel
        # rounds once, and both the FP8 block scales and the rotation read it.
        q = normed.to(torch.bfloat16).to(torch.float32)
        nope = q[:NOPE]
        blocks = nope.reshape(NOPE // QB, QB)
        absmax = blocks.abs().amax(dim=1).clamp(min=1e-4)
        exp = torch.ceil(torch.log2(absmax / FP8_MAX))
        fp8 = (blocks * torch.exp2(-exp)[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        fp8 = fp8.reshape(NOPE)
        scale = exp.clamp(min=-127.0, max=127.0).to(torch.int8)

        boundary = (c + 1) * CR - 1
        compressed_pos = (boundary // CR) * CR
        cos = cos_cache[compressed_pos, : ROPE // 2].to(torch.float32)
        sin = sin_cache[compressed_pos, : ROPE // 2].to(torch.float32)
        # fused_compress_quant_cache.py:314 reshapes `normed`, not `quant_input`.
        # The rotation reads fp32; only the FP8 path reads the bf16-rounded copy.
        tail = normed[NOPE:].reshape(ROPE // 2, 2)
        ev = tail[:, 0] * cos - tail[:, 1] * sin
        od = tail[:, 1] * cos + tail[:, 0] * sin
        rot = torch.stack((ev, od), dim=-1).reshape(ROPE)

        ckv = torch.cat((nope, rot)).to(torch.bfloat16)
        out.append((ckv, fp8, scale))
    return out


def make_inputs(total_tokens, max_position, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kv = torch.randn(total_tokens, HEAD_DIM, dtype=torch.bfloat16, device=DEV, generator=g) * 0.5
    score = torch.randn(total_tokens, HEAD_DIM, dtype=torch.bfloat16, device=DEV, generator=g) * 0.5
    # vLLM writes score + ape[position % CR] in the save-partial-states kernel.
    ape = torch.randn(CR, HEAD_DIM, dtype=torch.float32, device=DEV, generator=g) * 0.1
    pos = torch.arange(total_tokens, device=DEV)
    score = (score.to(torch.float32) + ape[pos % CR]).to(torch.bfloat16)
    rms = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=DEV, generator=g) + 0.5
    # A real RoPE table is a unit circle: cos^2 + sin^2 == 1 per entry.
    theta = torch.rand(max_position, ROPE // 2, dtype=torch.float32, device=DEV, generator=g) * 6.28
    # The Definition takes the window axis explicitly. For a contiguous buffer
    # this view is free and the bytes are identical to the flat vLLM layout.
    n = total_tokens // CR
    return (kv.view(n, CR, HEAD_DIM), score.view(n, CR, HEAD_DIM),
            rms, torch.cos(theta), torch.sin(theta))


def call_reference(inputs, total_tokens=None):
    return reference_run(*inputs)


def call_solution(inputs):
    kv = inputs[0]
    n = kv.shape[0]
    ckv = torch.empty((n, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    ckv_fp8 = torch.empty((n, NOPE), dtype=torch.float8_e4m3fn, device=DEV)
    ckv_scale = torch.empty((n, NOPE // QB), dtype=torch.int8, device=DEV)
    solution_mod.run(*inputs, ckv, ckv_fp8, ckv_scale)
    return ckv, ckv_fp8, ckv_scale


FAIL = []

ARGS = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ARGS.add_argument(
    "--mutations",
    action="store_true",
    help="also run the mutation battery that justifies eval_config.yaml (slower)",
)
ARGS = ARGS.parse_args()


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


print("=== shape / dtype contract ===")
for tt in (0, 128, 256, 2048):
    ckv, fp8, scale = call_reference(make_inputs(tt, max(tt, 1)), tt)
    n = tt // CR
    check(
        f"reference total_tokens={tt}",
        ckv.shape == (n, HEAD_DIM) and fp8.shape == (n, NOPE) and scale.shape == (n, NOPE // QB)
        and ckv.dtype == torch.bfloat16 and fp8.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.int8,
        f"{tuple(ckv.shape)} {ckv.dtype}",
    )

print("\n=== Triton solution vs reference, flashinfer-bench scoring ===")

# The tolerances come from eval_config.yaml rather than being repeated here:
# the config is what the harness is actually told to use, so a copy in this file
# would be free to drift away from it and silently validate against the wrong
# bar. Feeding them through the real ResolvedEvalConfig also type-checks the
# keys, so a typo in the YAML is an error here rather than a silently ignored
# setting at benchmark time.
with open(TASK / "eval_config.yaml") as fh:
    _cfg = yaml.safe_load(fh)["definition_config"]["hca_compress_c128_h512_r64"]
EVAL_CFG = _fib.ResolvedEvalConfig(**_cfg)
RTOL, ATOL = EVAL_CFG.rtol, EVAL_CFG.atol
REQUIRED_MATCHED_RATIO = EVAL_CFG.required_matched_ratio


def error_stats(sol, ref, rtol=None, atol=None):
    """Score with flashinfer-bench's OWN function, not a copy of it.

    Every number in eval_config.yaml is downstream of this, so a hand-copied
    reproduction that drifted from upstream would invalidate the whole
    derivation silently. Returns (max_abs, max_rel, matched_ratio); the real
    function's third return value (exceeds_tol) is recomputed by callers against
    REQUIRED_MATCHED_RATIO, so it is dropped here.
    """
    max_abs, max_rel, _exceeds, matched = _fib.compute_error_stats(sol, ref, EVAL_CFG)
    return max_abs, max_rel, matched


print(f"  (rtol={RTOL:g} atol={ATOL:g} required_matched_ratio={REQUIRED_MATCHED_RATIO} "
      "from eval_config.yaml)")
WORST = {}
SWEEP = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072]
for tt in SWEEP:
    inputs = make_inputs(tt, tt)
    r_ckv, r_fp8, r_scale = call_reference(inputs, tt)
    s_ckv, s_fp8, s_scale = call_solution(inputs)
    torch.cuda.synchronize()

    n = tt // CR
    check(
        f"total_tokens={tt} shapes",
        tuple(s_ckv.shape) == (n, HEAD_DIM)
        and tuple(s_fp8.shape) == (n, NOPE)
        and tuple(s_scale.shape) == (n, NOPE // QB),
        f"{tuple(s_ckv.shape)}",
    )

    # ckv_scale is an integer output: it must be exact, no tolerance applies.
    check(f"total_tokens={tt} ckv_scale exact", torch.equal(r_scale, s_scale),
          "exact" if torch.equal(r_scale, s_scale)
          else f"{(r_scale != s_scale).sum().item()} differ")

    for nm, s, r in (("ckv", s_ckv, r_ckv), ("ckv_fp8", s_fp8, r_fp8)):
        mabs, mrel, ratio = error_stats(s, r, RTOL, ATOL)
        WORST[nm] = min(WORST.get(nm, 1.0), ratio)
        bit = torch.equal(s.view(torch.uint8) if s.dtype == torch.float8_e4m3fn else s,
                          r.view(torch.uint8) if r.dtype == torch.float8_e4m3fn else r)
        check(f"total_tokens={tt} {nm} @rtol={RTOL:g}", ratio >= REQUIRED_MATCHED_RATIO,
              f"matched={ratio:.8f} max_abs={mabs:.2e} max_rel={mrel:.2e}"
              + ("  [bitwise exact]" if bit else ""))

print("\n=== reference vs the vLLM kernel port ===")
for tt in (128, 1024, 4096):
    inputs = make_inputs(tt, tt)
    kv, score, rms, cos_c, sin_c = inputs
    r_ckv, r_fp8, r_scale = call_reference(inputs, tt)
    port = vllm_port(kv, score, rms, cos_c, sin_c, tt)
    p_ckv = torch.stack([p[0] for p in port])
    p_fp8 = torch.stack([p[1] for p in port])
    p_scale = torch.stack([p[2] for p in port])
    check(f"total_tokens={tt} ckv matches port", torch.equal(r_ckv, p_ckv),
          "exact" if torch.equal(r_ckv, p_ckv)
          else f"max diff {(r_ckv.to(torch.float32) - p_ckv.to(torch.float32)).abs().max().item():.3e}")
    check(f"total_tokens={tt} fp8 matches port", torch.equal(r_fp8.view(torch.uint8), p_fp8.view(torch.uint8)))
    check(f"total_tokens={tt} scale matches port", torch.equal(r_scale, p_scale))

print("\n=== fp8/scale is a faithful UE8M0 encoding of ckv's noPE part ===")
# e4m3 is lossy BY CONSTRUCTION, so equality is the wrong assertion. The right
# one is half an ulp of e4m3 itself, scaled by the block's UE8M0 exponent.
# e4m3: 3 mantissa bits, min normal 2**-6, denormal step 2**-9. Elements far
# below their block's absmax land in the denormal range, where the *relative*
# error is unbounded but the absolute step stays 2**-9 -- which is why a flat
# 6.25% relative bound does not hold.
for tt in (128, 4096):
    r_ckv, r_fp8, r_scale = call_reference(make_inputs(tt, tt), tt)
    blk_scale = torch.exp2(r_scale.to(torch.float32)).repeat_interleave(QB, dim=1)
    q = r_fp8.to(torch.float32)
    deq = q * blk_scale
    ref = r_ckv[:, :NOPE].to(torch.float32)

    step = torch.where(q.abs() < 2.0**-6, torch.full_like(q, 2.0**-9),
                       torch.exp2(torch.floor(torch.log2(q.abs().clamp(min=2.0**-9))) - 3))
    bound = 0.5 * step * blk_scale
    over = ((deq - ref).abs() > bound * (1 + 1e-6)).sum().item() if tt else 0
    check(f"total_tokens={tt} dequant within half an e4m3 ulp", over == 0,
          f"{over} elements exceed the ulp bound")

print("\n=== worst matched_ratio across the sweep (sets eval_config) ===")
for nm, v in WORST.items():
    print(f"  {nm:8s} {v:.8f}")

# --------------------------------------------------------------------------
# The mutation battery.
#
# A tolerance that nothing can fail is not a test. Each entry below is a
# single-token edit to the reference that changes its semantics in a way a
# plausible kernel could get wrong; the configured tolerance must reject all of
# them while accepting the solution. The gap between the solution's worst score
# and the least-severe mutation's best score is the safety margin quoted in
# eval_config.yaml.
#
# Two known non-detections, both correct behaviour rather than gaps in the bar:
#
#   * removing the softmax max-subtraction is mathematically identical, so it is
#     not in this list;
#   * removing `clamp(min=1e-4)` on the block absmax changes no output, because
#     {"type": "random"} inputs never drive a 64-element block's absmax below
#     1e-4. That is a workload coverage gap, documented in eval_config.yaml.
# --------------------------------------------------------------------------
MUTATIONS = [
    (
        "cos row = boundary token",
        "compressed_pos = (boundary // compress_rate) * compress_rate",
        "compressed_pos = boundary",
    ),
    (
        "split-half RoPE (HF style)",
        "even = pairs[:, :, 0] * cos - pairs[:, :, 1] * sin",
        "even = rope[:, :rope_head_dim_half] * cos - rope[:, rope_head_dim_half:] * sin",
    ),
    (
        "ckv stores pre-rotation tail",
        "ckv = torch.cat((nope, rope), dim=-1)",
        "ckv = torch.cat((nope, normed[:, nope_head_dim:]), dim=-1)",
    ),
    (
        "quant skips the bf16 round",
        "nope = normed[:, :nope_head_dim].to(torch.bfloat16).to(torch.float32)",
        "nope = normed[:, :nope_head_dim]",
    ),
    (
        "cos/sin swapped",
        "cos = cos_cache[compressed_pos].to(torch.float32)\n    "
        "sin = sin_cache[compressed_pos].to(torch.float32)",
        "sin = cos_cache[compressed_pos].to(torch.float32)\n    "
        "cos = sin_cache[compressed_pos].to(torch.float32)",
    ),
    (
        "RMSNorm over 448 not 512",
        "variance = compressed.square().mean(dim=-1, keepdim=True)",
        "variance = compressed[:, :nope_head_dim].square().mean(dim=-1, keepdim=True)",
    ),
    (
        "softmax over the wrong axis",
        "weight = weight / weight.sum(dim=1, keepdim=True)",
        "weight = weight / weight.sum(dim=2, keepdim=True)",
    ),
    (
        "UE8M0 floor instead of ceil",
        "exponent = torch.ceil(",
        "exponent = torch.floor(",
    ),
]

if ARGS.mutations:
    print("\n=== mutation battery (justifies the eval_config tolerances) ===")
    src = definition["reference"]
    inputs = make_inputs(4096, 4096)
    s_out = call_solution(inputs)
    torch.cuda.synchronize()
    best_mutation = 0.0
    for name, old, new in MUTATIONS:
        if old not in src:
            check(f"mutation '{name}' still applies", False, "pattern not found in reference")
            continue
        mns = {}
        exec(compile(src.replace(old, new, 1), f"<mutation:{name}>", "exec"), mns)
        m_out = mns["run"](*inputs)
        ratios = {
            nm: error_stats(s_out[i], m_out[i], RTOL, ATOL)[2]
            for nm, i in (("ckv", 0), ("ckv_fp8", 1), ("ckv_scale", 2))
        }
        worst = min(ratios.values())
        best_mutation = max(best_mutation, worst)
        check(
            f"rejects '{name}'",
            worst < REQUIRED_MATCHED_RATIO,
            "min=%.6f (%s)" % (worst, ", ".join(f"{k}={v:.4f}" for k, v in ratios.items())),
        )
    solution_worst = min(WORST.values())
    print(
        f"\n  solution worst {solution_worst:.8f}  >  threshold "
        f"{REQUIRED_MATCHED_RATIO}  >  best mutation {best_mutation:.6f}"
    )
    check(
        "threshold separates solution from every mutation",
        best_mutation < REQUIRED_MATCHED_RATIO <= solution_worst,
    )
else:
    print("\n  (re-run with --mutations to check the tolerances still reject bad kernels)")

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): {FAIL}")
    sys.exit(1)
print("All checks passed.")
