"""Reference test for the hca_compress_c128_h512_r64 Definition.

The flashinfer-trace dataset expects one of these per definition, at
``tests/references/test_<definition>.py`` relative to the dataset root, and its
job is narrow: establish that the Definition's ``reference`` -- the plain-PyTorch
``run`` that every candidate kernel is scored against -- is *correct*, not merely
self-consistent. A reference nobody checked is an implementation, not a
specification, and the whole benchmark inherits its bugs.

Ground truth
------------

Upstream's convention is FlashInfer first, SGLang as fallback. Neither has this
op: HCA compression is DeepSeek-V4's, and the only published implementation is
vLLM's fused Triton kernel (``fused_compress_quant_cache.py``). So the ground
truth here is ``vllm_port`` below -- an independent transcription written from
that kernel's source rather than from the Definition, which is what makes the
agreement meaningful. Three places where the two could plausibly disagree, and
where the port follows vLLM rather than intuition, are marked inline.

What is asserted
----------------

1. Shape and dtype contract, including the empty case.
2. The reference is *bitwise* equal to the vLLM port on all three outputs. Not
   within a tolerance -- bitwise. Both run the same fp32 arithmetic in the same
   order, so anything less would mean a real semantic difference.
3. ``ckv_fp8``/``ckv_scale`` dequantise back to ``ckv``'s noPE part within half
   an e4m3 ulp, i.e. the pair really is a UE8M0 encoding of that tensor.
4. The configured tolerance rejects each of eight targeted semantic mutations of
   the reference. A tolerance nothing can fail is not a test, and these numbers
   are the derivation behind ``required_matched_ratio`` -- see
   ``tasks/hca_compress_c128/eval_config.yaml``.
5. The task's ``eval_config.yaml`` and the tolerances flashinfer-bench actually
   resolves agree. They live in two files for a reason the config explains, and
   nothing else notices when they drift.

Scoring uses ``flashinfer_bench.bench.utils.compute_error_stats`` itself rather
than a reimplementation: every tolerance in the task is downstream of that
function, so a local copy that drifted from upstream would invalidate the
derivation silently.

The solution kernels are deliberately absent. Solution-vs-reference is what the
benchmark measures (``tools/run_benchmark.py``); duplicating it here would make
this file depend on a Triton build and on a layout that the dataset does not
have.

Runs on CUDA; skipped otherwise. Layout-portable: works both from the flat task
directory and from a staged TraceSet root.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

pytest.importorskip("flashinfer_bench")

from flashinfer_bench.bench import BenchmarkConfig  # noqa: E402
from flashinfer_bench.bench.utils import compute_error_stats  # noqa: E402
from flashinfer_bench.data import Definition  # noqa: E402

# Model constants, from the Definition's const axes.
HEAD_DIM = 512
NOPE = 448  # nope_head_dim
ROPE = 64  # rope_head_dim
CR = 128  # compress_rate
QB = 64  # quant_block_size
EPS = 1e-6
FP8_MAX = 448.0

DEF_NAME = "hca_compress_c128_h512_r64"

# parents[2] is the dataset root in the flashinfer-trace layout
# (<root>/tests/references/<this file>) and the task directory in this
# repository's flat layout (tasks/<task>/tests/references/<this file>). Both are
# supported so the file can be lifted into the dataset unchanged.
BASE = pathlib.Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the reference runs on CUDA"
)


def _load_definition() -> dict:
    candidates = [BASE / f"{DEF_NAME}.json", *BASE.glob(f"definitions/*/{DEF_NAME}.json")]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    raise RuntimeError(f"{DEF_NAME}.json not found under {BASE}")


DEFINITION = _load_definition()


def _compile_reference(source: str):
    ns: dict = {}
    exec(compile(source, f"<{DEF_NAME}.reference>", "exec"), ns)
    return ns["run"]


reference_run = _compile_reference(DEFINITION["reference"])

# Resolve tolerances the way the harness does rather than restating them. This
# reads the bundled per-op_type eval_config, so the numbers under test are the
# ones a kernel will actually be graded against.
EVAL = BenchmarkConfig.default().resolve_eval_config(Definition.model_validate(DEFINITION))


def make_inputs(total_tokens: int, max_position: int, seed: int = 0, device: str = "cuda"):
    """Synthetic inputs with the structure the real ones have.

    Deliberately not ``torch.randn`` throughout: two of these inputs are not
    free-form tensors. ``cos_cache``/``sin_cache`` are a rotation table and must
    lie on the unit circle, and ``score_state`` carries vLLM's absolute
    positional encoding added per within-window position. Getting either wrong
    makes the test pass on inputs the op will never see.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    kv = torch.randn(total_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=g) * 0.5
    score = (
        torch.randn(total_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=g) * 0.5
    )
    # vLLM adds ape[position % compress_rate] in the save-partial-states kernel.
    ape = torch.randn(CR, HEAD_DIM, dtype=torch.float32, device=device, generator=g) * 0.1
    pos = torch.arange(total_tokens, device=device)
    score = (score.to(torch.float32) + ape[pos % CR]).to(torch.bfloat16)
    rms = torch.rand(HEAD_DIM, dtype=torch.bfloat16, device=device, generator=g) + 0.5
    theta = (
        torch.rand(max_position, ROPE // 2, dtype=torch.float32, device=device, generator=g) * 6.28
    )
    # The Definition takes the 128-token window as an explicit axis. For a
    # contiguous buffer this view is free and the bytes match vLLM's flat layout.
    n = total_tokens // CR
    return (
        kv.view(n, CR, HEAD_DIM),
        score.view(n, CR, HEAD_DIM),
        rms,
        torch.cos(theta),
        torch.sin(theta),
    )


@torch.no_grad()
def vllm_port(kv_state, score_state, rms_weight, cos_cache, sin_cache, total_tokens):
    """Independent transcription of vLLM's fused_compress_quant_cache kernel."""
    out = []
    for c in range(total_tokens // CR):
        score = score_state[c].to(torch.float32)
        kv = kv_state[c].to(torch.float32)

        score = score - score.amax(dim=0, keepdim=True)
        w = torch.exp(score)
        w = w / w.sum(dim=0, keepdim=True)
        compressed = (kv * w).sum(dim=0)

        # RMSNorm over all 512 channels, not just the 448 that get quantised.
        var = (compressed * compressed).sum() / HEAD_DIM
        normed = compressed * torch.rsqrt(var + EPS) * rms_weight.to(torch.float32)

        # `quant_input = normed.to(tl.bfloat16).to(tl.float32)`: the kernel rounds
        # to bf16 once, and both the block scales and the stored noPE half read
        # that rounded copy.
        q = normed.to(torch.bfloat16).to(torch.float32)
        nope = q[:NOPE]
        blocks = nope.reshape(NOPE // QB, QB)
        absmax = blocks.abs().amax(dim=1).clamp(min=1e-4)
        exp = torch.ceil(torch.log2(absmax / FP8_MAX))
        fp8 = (
            (blocks * torch.exp2(-exp)[:, None])
            .clamp(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
            .reshape(NOPE)
        )
        scale = exp.clamp(min=-127.0, max=127.0).to(torch.int8)

        # The rotation reads the row of the window's FIRST token, not its last.
        boundary = (c + 1) * CR - 1
        compressed_pos = (boundary // CR) * CR
        cos = cos_cache[compressed_pos, : ROPE // 2].to(torch.float32)
        sin = sin_cache[compressed_pos, : ROPE // 2].to(torch.float32)
        # fused_compress_quant_cache.py:314 reshapes `normed`, not `quant_input`:
        # the rotation reads fp32, and only the FP8 path sees the bf16 round.
        # Interleaved pairs (GPT-NeoX style), not split halves.
        tail = normed[NOPE:].reshape(ROPE // 2, 2)
        even = tail[:, 0] * cos - tail[:, 1] * sin
        odd = tail[:, 1] * cos + tail[:, 0] * sin
        rot = torch.stack((even, odd), dim=-1).reshape(ROPE)

        out.append((torch.cat((nope, rot)).to(torch.bfloat16), fp8, scale))
    return (
        torch.stack([o[0] for o in out]),
        torch.stack([o[1] for o in out]),
        torch.stack([o[2] for o in out]),
    )


def matched_ratio(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of elements within the configured tolerance, per the harness."""
    return compute_error_stats(candidate, reference, EVAL)[3]


# ---------------------------------------------------------------------------
# 1. shape / dtype contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_tokens", [0, 128, 256, 2048])
@torch.no_grad()
def test_shape_and_dtype_contract(total_tokens):
    ckv, fp8, scale = reference_run(*make_inputs(total_tokens, max(total_tokens, 1)))
    n = total_tokens // CR
    assert tuple(ckv.shape) == (n, HEAD_DIM)
    assert tuple(fp8.shape) == (n, NOPE)
    assert tuple(scale.shape) == (n, NOPE // QB)
    assert ckv.dtype == torch.bfloat16
    assert fp8.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.int8


# ---------------------------------------------------------------------------
# 2. the reference IS the vLLM kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_tokens", [128, 1024, 4096])
@torch.no_grad()
def test_reference_matches_vllm_kernel_bitwise(total_tokens):
    inputs = make_inputs(total_tokens, total_tokens)
    r_ckv, r_fp8, r_scale = reference_run(*inputs)
    p_ckv, p_fp8, p_scale = vllm_port(*inputs, total_tokens)

    # Bitwise, not allclose. Both sides do the same fp32 operations in the same
    # order, so any difference at all is a semantic one worth failing on.
    assert torch.equal(r_ckv, p_ckv), (
        "ckv differs from the vLLM port by up to "
        f"{(r_ckv.float() - p_ckv.float()).abs().max().item():.3e}"
    )
    # fp8 has no eq kernel; compare the raw e4m3 code points.
    assert torch.equal(r_fp8.view(torch.uint8), p_fp8.view(torch.uint8))
    assert torch.equal(r_scale, p_scale)


# ---------------------------------------------------------------------------
# 3. fp8 + scale really encode ckv's noPE part
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_tokens", [128, 4096])
@torch.no_grad()
def test_ue8m0_dequantises_within_half_an_e4m3_ulp(total_tokens):
    """e4m3 is lossy by construction, so equality is the wrong assertion.

    The right bound is half an ulp of e4m3 itself, scaled by the block's UE8M0
    exponent. e4m3 has 3 mantissa bits, min normal 2**-6 and denormal step
    2**-9; elements far below their block's absmax land in the denormal range,
    where the *relative* error is unbounded while the absolute step stays 2**-9.
    That is why a flat 6.25% relative bound does not hold here.
    """
    r_ckv, r_fp8, r_scale = reference_run(*make_inputs(total_tokens, total_tokens))
    blk_scale = torch.exp2(r_scale.to(torch.float32)).repeat_interleave(QB, dim=1)
    q = r_fp8.to(torch.float32)
    dequantised = q * blk_scale
    expected = r_ckv[:, :NOPE].to(torch.float32)

    step = torch.where(
        q.abs() < 2.0**-6,
        torch.full_like(q, 2.0**-9),
        torch.exp2(torch.floor(torch.log2(q.abs().clamp(min=2.0**-9))) - 3),
    )
    bound = 0.5 * step * blk_scale
    over = ((dequantised - expected).abs() > bound * (1 + 1e-6)).sum().item()
    assert over == 0, f"{over} elements exceed half an e4m3 ulp"


# ---------------------------------------------------------------------------
# 4. the configured tolerance rejects wrong kernels
# ---------------------------------------------------------------------------
# Each entry is a single edit to the reference that changes its semantics the
# way a plausible kernel could get it wrong. Two known non-detections are absent
# on purpose, both correct rather than gaps: dropping the softmax
# max-subtraction is mathematically identical, and dropping `clamp(min=1e-4)`
# changes no output on these inputs (it takes the `tiny` stress workload to
# reach it -- see eval_config.yaml).
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


@torch.no_grad()
def _mutation_ratio(old: str, new: str) -> float:
    source = DEFINITION["reference"]
    assert old in source, f"mutation pattern no longer present in the reference: {old!r}"
    mutated = _compile_reference(source.replace(old, new, 1))
    inputs = make_inputs(4096, 4096)
    truth = reference_run(*inputs)
    got = mutated(*inputs)
    torch.cuda.synchronize()
    return min(matched_ratio(g, t) for g, t in zip(got, truth))


@pytest.mark.parametrize("name,old,new", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_tolerance_rejects_mutation(name, old, new):
    ratio = _mutation_ratio(old, new)
    assert ratio < EVAL.required_matched_ratio, (
        f"mutation {name!r} scores matched_ratio={ratio:.6f}, which the configured "
        f"threshold {EVAL.required_matched_ratio} accepts -- the tolerance is too loose "
        f"to distinguish it from a correct kernel"
    )


def test_mutations_leave_margin_below_the_threshold():
    """The threshold has to clear the *least* severe mutation, not the average."""
    best = max(_mutation_ratio(old, new) for _, old, new in MUTATIONS)
    assert best < EVAL.required_matched_ratio
    # This runs on one synthetic input; over the 14 blob-backed sweep
    # workloads eval_config.yaml quotes the margin as 0.098609. Either way the
    # assertion below only has to stop the bar from eroding to nothing, which
    # would mean it is riding on the noise floor.
    assert EVAL.required_matched_ratio - best > 0.01, (
        f"only {EVAL.required_matched_ratio - best:.4f} between the threshold and the "
        f"least severe mutation ({best:.6f})"
    )


# ---------------------------------------------------------------------------
# 5. the two copies of the tolerances agree
# ---------------------------------------------------------------------------
def test_task_eval_config_agrees_with_the_resolved_one():
    """The same tolerances live in two files, and nothing else checks them.

    ``BenchmarkConfig.default()`` -- which ``flashinfer-bench validate`` uses and
    cannot be handed a config file -- reads only the bundled per-op_type
    eval_config. The task's own ``eval_config.yaml`` is what
    ``tools/run_benchmark.py`` loads. If they disagree, the validator and the
    runner grade the same kernel against different bars, and the first symptom
    is a kernel that passes one and fails the other for no visible reason.
    """
    task_config = BASE / "eval_config.yaml"
    if not task_config.is_file():
        pytest.skip("no task-local eval_config.yaml in this layout")
    yaml = pytest.importorskip("yaml")
    authored = yaml.safe_load(task_config.read_text())["definition_config"][DEF_NAME]
    for key in ("rtol", "atol", "required_matched_ratio"):
        assert authored[key] == getattr(EVAL, key), (
            f"{key}: eval_config.yaml says {authored[key]}, flashinfer-bench resolves "
            f"{getattr(EVAL, key)}"
        )
