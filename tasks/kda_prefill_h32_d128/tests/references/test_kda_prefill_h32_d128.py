"""Reference test for the kda_prefill_h32_d128 Definition.

The flashinfer-trace dataset expects one of these per definition, at
``tests/references/test_<definition>.py`` relative to the dataset root, and its
job is narrow: establish that the Definition's ``reference`` -- the plain-PyTorch
``run`` that every candidate kernel is scored against -- is *correct*, not merely
self-consistent. A reference nobody checked is an implementation, not a
specification, and the whole benchmark inherits its bugs.

Ground truth
------------

Upstream's convention is FlashInfer first, SGLang as fallback. Neither has this
op. ``transformers`` does: ``recurrent_kimi_delta_attention`` in
``models/kimi_linear/modeling_kimi_linear.py`` is the released model's own O(T)
sequential form -- ``seq_len`` rank-1 updates, one token at a time. The
Definition's reference is the *chunked* form: blocked matmuls, a WY
representation and a UT transform, mathematically the same function reassociated
for throughput.

That makes it a much stronger ground truth than a transcription would be. The
two implementations share no code path, no summation order and no blocking, so
agreement between them is evidence about the operator rather than about a
shared derivation. The chunked form is where every plausible bug lives -- an
off-by-one in the cumsum, a decay applied on the wrong side, a triangular mask
one diagonal out -- and the recurrence has none of that structure to get wrong.

Neither is more correct than the other, so they will not agree bitwise: they do
the same fp32 arithmetic in different orders. What is asserted is that their
disagreement stays inside the floor measured in ``eval_config.yaml``.

What is asserted
----------------

1. Shape and dtype contract, including the two ragged cases -- ``seq_len``
   shorter than a chunk, and ``seq_len`` not a multiple of it.
2. The reference agrees with ``recurrent_kimi_delta_attention`` on both outputs
   to ``matched_ratio == 1.0`` at the configured tolerance, and its pointwise
   error stays inside the floor the tolerance was derived against.
3. ``final_state`` is a usable continuation point: prefilling ``[0, T)`` in one
   call equals prefilling ``[0, T/2)`` and feeding the resulting state into
   ``[T/2, T)``. The Definition claims this in its description and takes
   ``initial_state`` as an input because of it; nothing else in the task checks
   it, and a reference that decayed or wrote the state one step out of phase
   would still satisfy assertion 2 on a single call.
4. The configured tolerance rejects each of nine targeted semantic mutations of
   the reference. A tolerance nothing can fail is not a test.
5. The task's ``eval_config.yaml`` and the tolerances flashinfer-bench actually
   resolves agree. They live in two files for a reason the config explains, and
   nothing else notices when they drift.

Scoring uses ``flashinfer_bench.bench.utils.compute_error_stats`` itself rather
than a reimplementation: every tolerance in the task is downstream of that
function -- in particular of the fact that an element fails only if it breaches
``atol`` AND ``rtol`` -- so a local copy that drifted from upstream would
invalidate the derivation silently.

Inputs are generated here rather than read from the workload corpus, so the file
is self-contained and liftable into the dataset. They are not ``torch.randn``:
``g`` is built as ``-exp(A_log) * softplus(.)`` and ``beta`` as a sigmoid,
because both signs are load-bearing (see the Definition's input descriptions and
``tools/gen_kda_blobs.py``), and q/k/v are passed through SiLU because that is
what precedes this op in the layer. The separation figures quoted in
``eval_config.yaml`` were measured on the real corpus by
``tools/derive_kda_tolerances.py``; the assertions here only have to hold on
inputs of the right shape and sign, which is what makes them cheap enough to run
in CI.

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
import torch.nn.functional as F

pytest.importorskip("flashinfer_bench")

from flashinfer_bench.bench import BenchmarkConfig  # noqa: E402
from flashinfer_bench.bench.utils import compute_error_stats  # noqa: E402
from flashinfer_bench.data import Definition  # noqa: E402

# Model constants, from the Definition's const axes.
NUM_HEADS = 32
HEAD_DIM = 128
CHUNK = 64

DEF_NAME = "kda_prefill_h32_d128"

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

_DEFINITION_OBJ = Definition.model_validate(DEFINITION)
_TASK_CONFIG = BASE / "eval_config.yaml"


def _resolve_eval():
    """Resolve tolerances the way the thing that grades kernels does.

    Order matters and is not the obvious one. ``tools/run_benchmark.py`` loads
    the task's own ``eval_config.yaml``, and so does ``flashinfer-bench
    validate`` (third_party/patches/004-validate-prefers-task-eval-config.patch).
    The file bundled inside the package carries only the per-op_type entries that
    shipped upstream, and there is no ``kda`` entry among them -- so
    ``BenchmarkConfig.default()`` resolves this definition to
    ``required_matched_ratio = None``, which ``compute_error_stats`` reads as 1.0,
    i.e. bitwise equality. Deriving the tolerances under test from the default
    would test a bar no kernel is ever graded against.
    """
    if _TASK_CONFIG.is_file():
        return BenchmarkConfig.from_yaml(str(_TASK_CONFIG)).resolve_eval_config(
            _DEFINITION_OBJ
        )
    return BenchmarkConfig.default().resolve_eval_config(_DEFINITION_OBJ)


EVAL = _resolve_eval()


def make_inputs(batch_size: int, seq_len: int, seed: int = 0, device: str = "cuda") -> dict:
    """Synthetic inputs with the structure the real ones have.

    Three of the six inputs are not free-form tensors, and for two of them the
    difference is a change of operator rather than a distribution shift:

    * ``g`` is ``-exp(A_log) * softplus(x)`` and therefore strictly negative.
      Gaussian ``g`` makes half the channels *grow* under ``exp(cumsum(g))`` and
      the recurrence diverges -- the reference would return inf and every
      assertion below would be vacuous.
    * ``beta`` is a sigmoid and therefore in (0, 1). Outside that range the delta
      rule over- or back-steps instead of interpolating.
    * ``query``/``key``/``value`` leave a depthwise causal conv followed by SiLU,
      so they are floored at -0.2785 and right-skewed. q and k are l2-normalised
      inside the op, which hides the scale change but not the geometry.

    The gate parameters are chosen so the median channel memory lands near the
    corpus's 15 tokens rather than at a value that makes the state trivially
    dead or trivially persistent.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def conv_silu():
        x = torch.randn(
            batch_size, seq_len, NUM_HEADS, HEAD_DIM, device=device, generator=gen
        )
        return F.silu(x * 0.55).to(torch.bfloat16)

    a_log = torch.randn(NUM_HEADS * HEAD_DIM, device=device, generator=gen) * 0.5
    dt = torch.randn(
        batch_size, seq_len, NUM_HEADS, HEAD_DIM, device=device, generator=gen
    )
    g = (-torch.exp(a_log).view(1, 1, NUM_HEADS, HEAD_DIM) * F.softplus(dt - 3.0)).float()

    return {
        "query": conv_silu(),
        "key": conv_silu(),
        "value": conv_silu(),
        "g": g,
        "beta": torch.sigmoid(
            torch.randn(batch_size, seq_len, NUM_HEADS, device=device, generator=gen)
        ).to(torch.bfloat16),
        "initial_state": torch.zeros(
            batch_size, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        ),
    }


def matched_ratio(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of elements within the configured tolerance, per the harness."""
    return compute_error_stats(candidate, reference, EVAL)[3]


def _recurrent(inputs: dict):
    """The released model's own O(T) form, as ground truth."""
    kimi = pytest.importorskip("transformers.models.kimi_linear.modeling_kimi_linear")
    return kimi.recurrent_kimi_delta_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        # The Definition l2-normalises q and k inside the op; the recurrence has
        # to be told to do the same or it is a different function.
        use_qk_l2norm_in_kernel=True,
    )


# ---------------------------------------------------------------------------
# 1. shape / dtype contract
# ---------------------------------------------------------------------------
# 100 is shorter than a chunk and 1000 is not a multiple of one: the reference
# pads to a chunk boundary and trims, and both ragged cases are in the sweep.
@pytest.mark.parametrize("batch_size,seq_len", [(1, 64), (1, 100), (1, 1000), (2, 128)])
@torch.no_grad()
def test_shape_and_dtype_contract(batch_size, seq_len):
    inputs = make_inputs(batch_size, seq_len)
    out, state = reference_run(**inputs)

    assert tuple(out.shape) == (batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    assert tuple(state.shape) == (batch_size, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    assert out.dtype == torch.bfloat16, "out carries the query's dtype"
    assert state.dtype == torch.float32, "the state is fp32 regardless of the io dtype"
    assert torch.isfinite(out.float()).all() and torch.isfinite(state).all()


# ---------------------------------------------------------------------------
# 2. the reference IS the recurrence, reassociated
# ---------------------------------------------------------------------------
# The floor measured over the real corpus (eval_config.yaml) is 2.441e-04 abs on
# the output and 2.593e-06 on the state. These bounds are that, with room: the
# point of the assertion is to catch a structural difference, which lands orders
# of magnitude above, not to police the last fp32 bit.
FLOOR_OUT_ABS = 1e-3
FLOOR_STATE_ABS = 1e-4


@pytest.mark.parametrize("batch_size,seq_len", [(1, 128), (1, 1000), (2, 512)])
@torch.no_grad()
def test_reference_matches_the_recurrent_implementation(batch_size, seq_len):
    inputs = make_inputs(batch_size, seq_len)
    out, state = reference_run(**inputs)
    r_out, r_state = _recurrent(inputs)

    out_abs = (out.float() - r_out.float()).abs().max().item()
    state_abs = (state - r_state.float()).abs().max().item()
    assert out_abs < FLOOR_OUT_ABS, f"out differs from the recurrence by {out_abs:.3e}"
    assert state_abs < FLOOR_STATE_ABS, f"state differs by {state_abs:.3e}"

    # The bar a kernel is actually graded against. Both implementations are
    # correct, so the recurrence has to score a clean 1.0 here -- if it did not,
    # the tolerance would be rejecting correct implementations.
    assert matched_ratio(r_out, out) == 1.0
    assert matched_ratio(r_state.float(), state) == 1.0


# ---------------------------------------------------------------------------
# 3. final_state is a continuation point
# ---------------------------------------------------------------------------
@torch.no_grad()
def test_split_prefill_equals_single_prefill():
    """Two half prefills chained through final_state must equal one whole one.

    This is the Definition's stated reason for taking ``initial_state`` as an
    input and returning the updated state, and it is the only assertion here
    that exercises the state's *phase*: a reference that applied the chunk decay
    one step early or late, or wrote the rank-1 update before decaying rather
    than after, still matches the recurrence on a single call from zeros in some
    of those cases, but cannot survive being cut in half and resumed.

    The split lands on a chunk boundary because the reference pads each call to
    one; an unaligned split point is a different computation, not a bug.
    """
    seq_len, half = 512, 256
    inputs = make_inputs(1, seq_len, seed=7)

    whole_out, whole_state = reference_run(**inputs)

    first = {k: (v[:, :half] if k != "initial_state" else v) for k, v in inputs.items()}
    out_a, state_a = reference_run(**first)
    second = {k: (v[:, half:] if k != "initial_state" else state_a) for k, v in inputs.items()}
    out_b, state_b = reference_run(**second)

    joined = torch.cat((out_a, out_b), dim=1)
    out_abs = (joined.float() - whole_out.float()).abs().max().item()
    state_abs = (state_b - whole_state).abs().max().item()

    assert out_abs < FLOOR_OUT_ABS, (
        f"resuming from final_state changes the output by {out_abs:.3e}; "
        "the state is not a valid continuation point"
    )
    assert state_abs < FLOOR_STATE_ABS, f"final state differs by {state_abs:.3e}"
    assert matched_ratio(joined, whole_out) == 1.0
    assert matched_ratio(state_b, whole_state) == 1.0


# ---------------------------------------------------------------------------
# 4. the configured tolerance rejects wrong kernels
# ---------------------------------------------------------------------------
# Each entry is a single edit to the reference that changes its semantics the
# way a plausible kernel could get it wrong. They are source-level replacements
# rather than a parameterised copy of the reference so that the assertion
# "``old`` is still present" doubles as a drift guard: if the reference is
# rewritten, these fail loudly instead of silently testing nothing.
#
# The twelve variants behind eval_config.yaml's threshold, including the four
# bf16 precision variants that have to be *accepted*, are measured over the real
# corpus by tools/derive_kda_tolerances.py. This list overlaps it but is not the
# same set: these are the mutations expressible as a one-line edit.
#
# Measured here at B=1 T=1024, seed 3, against rtol=2e-2 atol=1e-3 (min over the
# two outputs is what decides a variant, since a trace fails if any output
# exceeds):
#
#     mutation                                    out      state        min
#     ------------------------------------------------------------------------
#     q/k not l2-normalised                  0.001501   0.000000   0.000000
#     per-token g, not the cumsum            0.077453   0.005732   0.005732
#     query not scaled by 1/sqrt(head_dim)   0.021309   1.000000   0.021309
#     state write not decayed to chunk end   0.751537   0.020121   0.020121
#     beta not applied to the key            0.823086   0.202229   0.202229
#     decay_mask transposed                  0.462408   0.968750   0.462408
#     UT transform omitted                   0.992965   0.574390   0.574390
#     intra-chunk excludes current token     0.754367   1.000000   0.754367
#     v_prime correction dropped             0.976886   0.869518   0.869518
#
# Two things in that table are worth keeping. `UT transform omitted` scores
# 0.992965 on core_attn_out -- above the 0.99 threshold -- and is caught only by
# final_state at 0.574390. `query not scaled` and `intra-chunk excludes current
# token` are the mirror image, caught only by the output. Neither output alone
# separates this operator, which is the concrete reason the Definition declares
# both rather than treating the state as an implementation detail.
MUTATIONS = [
    (
        "q/k not l2-normalised",
        "    query = _l2norm(query)\n    key = _l2norm(key)",
        "    query = query * 1.0\n    key = key * 1.0",
    ),
    (
        "per-token g instead of the within-chunk cumsum",
        "    g = g.cumsum(dim=-2)",
        "    g = g * 1.0",
    ),
    (
        # The only entry here that overflows rather than merely differing:
        # reversing the sign makes the within-chunk decay grow as e^|g| and the
        # masked build goes non-finite. Any tolerance rejects it, so this one is
        # a smoke test rather than a measurement of separation.
        "decay_mask transposed (g_j - g_i)",
        "(g.unsqueeze(-2) - g.unsqueeze(-3))",
        "(g.unsqueeze(-3) - g.unsqueeze(-2))",
    ),
    (
        "state write not decayed to the chunk end",
        "(k_i * (g_i[:, :, -1:] - g_i).exp())",
        "(k_i * g_i.exp())",
    ),
    (
        "UT transform omitted (first-order WY)",
        "for i in range(1, _CHUNK):",
        "for i in range(1, 1):",
    ),
    (
        "beta not applied to the key",
        "    k_beta = key * beta.unsqueeze(-1)",
        "    k_beta = key * 1.0",
    ),
    (
        "delta rule drops the v_prime correction",
        "        v_new = v_i - v_prime",
        "        v_new = v_i + 0.0 * v_prime",
    ),
    (
        "intra-chunk attention excludes the current token",
        "    causal_mask = torch.triu(\n"
        "        torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=1\n"
        "    )",
        "    causal_mask = torch.triu(\n"
        "        torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=0\n"
        "    )",
    ),
    (
        "query not scaled by 1/sqrt(head_dim)",
        "    query = F.pad(query, (0, 0, 0, pad_size)) * scale",
        "    query = F.pad(query, (0, 0, 0, pad_size))",
    ),
]


@torch.no_grad()
def _mutation_ratio(old: str, new: str) -> float:
    source = DEFINITION["reference"]
    assert old in source, f"mutation pattern no longer present in the reference: {old!r}"
    mutated = _compile_reference(source.replace(old, new, 1))
    inputs = make_inputs(1, 1024, seed=3)
    truth = reference_run(**inputs)
    got = mutated(**inputs)
    torch.cuda.synchronize()
    # The verdict is per output and a trace fails if any output exceeds, so the
    # figure that decides a variant's fate is the MIN over the two outputs.
    # final_state is by far the more discriminating of them -- see the table in
    # eval_config.yaml, where every semantic mutation scores higher on the
    # output than on the state.
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
    # This runs on one synthetic input at T=1024. Over the real 16-workload
    # sweep eval_config.yaml quotes the least severe semantic mutation at
    # 0.456037 against a threshold of 0.99. Either way the assertion below only
    # has to stop the bar from eroding to nothing, which would mean it is riding
    # on the noise floor.
    assert EVAL.required_matched_ratio - best > 0.01, (
        f"only {EVAL.required_matched_ratio - best:.4f} between the threshold and the "
        f"least severe mutation ({best:.6f})"
    )


# ---------------------------------------------------------------------------
# 5. the copies of the tolerances agree
# ---------------------------------------------------------------------------
def test_task_eval_config_round_trips():
    """The numbers written in eval_config.yaml are the ones that get resolved.

    They pass through ``BenchmarkConfig``'s schema on the way, so a key under
    the wrong nesting level -- ``op_type_config`` instead of
    ``definition_config``, say -- is silently dropped rather than rejected, and
    the task is then graded at the fallback while its file reads as if it were
    not.
    """
    if not _TASK_CONFIG.is_file():
        pytest.skip("no task-local eval_config.yaml in this layout")
    yaml = pytest.importorskip("yaml")
    authored = yaml.safe_load(_TASK_CONFIG.read_text())["definition_config"][DEF_NAME]
    for key in ("rtol", "atol", "required_matched_ratio"):
        assert authored[key] == getattr(EVAL, key), (
            f"{key}: eval_config.yaml says {authored[key]}, flashinfer-bench resolves "
            f"{getattr(EVAL, key)}"
        )


def test_the_bundled_config_does_not_contradict_the_task_one():
    """If the package ever grows a kda entry, it must not disagree with this one.

    Today it has none, which is exactly why patch 004 exists: without the task's
    own file, ``BenchmarkConfig.default()`` resolves this definition to
    rtol=1e-2, atol=1e-2, required_matched_ratio=None -- and ``None`` means
    bitwise equality. eval_config.yaml measures what that setting does: a
    legitimate bf16 tensor-core kernel fails on ``final_state`` while a kernel
    with no UT transform passes on the output. Both halves of that are wrong, in
    opposite directions.

    The assertion is one-sided on purpose. A bundled entry appearing later is
    fine; a bundled entry that *disagrees* means the validator and the runner
    grade the same kernel against different bars, and the first symptom is a
    kernel that passes one and fails the other for no visible reason.
    """
    if not _TASK_CONFIG.is_file():
        pytest.skip("no task-local eval_config.yaml in this layout")
    bundled = BenchmarkConfig.default().resolve_eval_config(_DEFINITION_OBJ)
    if bundled.required_matched_ratio is None:
        pytest.skip(
            "the bundled eval_config has no kda entry; the task's own file is "
            "the only source of these tolerances (patch 004)"
        )
    for key in ("rtol", "atol", "required_matched_ratio"):
        assert getattr(bundled, key) == getattr(EVAL, key), (
            f"{key}: the bundled eval_config resolves {getattr(bundled, key)}, the "
            f"task's own file {getattr(EVAL, key)}"
        )
