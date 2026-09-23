#!/usr/bin/env python3
"""Build the KDA prefill task: Definition, reference, and workload sweep.

Kimi Delta Attention is the linear-attention half of Kimi Linear (arXiv
2510.26692, 3 KDA layers to 1 full MLA layer). It is Gated DeltaNet with the
scalar forget gate replaced by a per-channel one::

    S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T

with ``a_t = exp(g_t)`` in [0,1]^{d_k} rather than a single scalar. The recurrent
form above is the specification; the chunked form this task benchmarks is the
algebraically equivalent rearrangement that makes it a matmul problem -- a WY
representation packs the rank-1 updates and a UT transform replaces the matrix
inverse with a triangular solve.

**This task is the prefill chunk kernel only.** Prefill and decode are the same
operator but opposite problems: the chunk kernel is FLOP-bound (the WY/UT
machinery exists precisely to convert non-matmul work into matmul work) and the
decode recurrence is bandwidth-bound at one token per step. Ranking them on a
single figure of merit would be incoherent, so decode is a separate definition
if it is ever wanted. Solutions here are scored on achieved TFLOP/s against a
measured matmul peak -- see flops_model.py, not bytes_model.py.

Provenance
----------

The reference is a port of ``chunk_kimi_delta_attention`` from
``transformers.models.kimi_linear.modeling_kimi_linear`` (transformers 5.17.0),
which is the pure-PyTorch fallback the released model runs when FLA's fused
``chunk_kda`` is unavailable. It is ported rather than imported so the Definition
is self-contained, and it is checked against ``recurrent_kimi_delta_attention``
-- the O(T) loop form -- by tools/check_kda_numerics.py. Two independent
implementations of the same recurrence agreeing is what makes the reference a
specification rather than just the first thing that was written.

Input structure is not decorative here
--------------------------------------

``g`` is a log decay and the model produces it as ``-exp(A_log) * softplus(...)``,
so it is **strictly negative**; ``beta`` is a sigmoid, so it lies in (0,1). A
workload that fills them with ``torch.randn`` does not merely get the
distribution wrong -- half of ``g`` becomes positive, ``g.cumsum(-2).exp()``
grows without bound, and the state diverges. The op being measured would be a
different op. This is why every workload here carries real tensors rather than
``{"type": "random"}``, and why tools/verify_task.py grew a C3 probe for the
sign of ``g``.
"""

import argparse
import json
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fib_shim  # noqa: E402

REPO = fib_shim.REPO

TASK_NAME = "kda_prefill_h32_d128"
DEF_NAME = "kda_prefill_h32_d128"
OP_TYPE = "kda"

# From KimiLinearConfig (transformers 5.17.0): linear_num_heads=32,
# linear_head_dim=128 for both the key and value head dims. chunk_size=64 is
# fixed in the algorithm, not a tunable -- the UT transform's triangular solve is
# sized by it.
NUM_HEADS = 32
HEAD_DIM = 128
CHUNK = 64


REFERENCE = '''
import torch
import torch.nn.functional as F

_CHUNK = 64
_L2_EPS = 1e-6


def _l2norm(x):
    """FLA's l2norm: +eps inside the sqrt, not max(norm, eps).

    The difference from F.normalize is small but systematic, and the released
    model's numerics were trained against this one.
    """
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + _L2_EPS)


def run(query, key, value, g, beta, initial_state):
    """Kimi Delta Attention, chunked prefill form.

    query/key/value : [batch, seq_len, num_heads, head_dim]
    g               : [batch, seq_len, num_heads, head_dim]  log decay, <= 0
    beta            : [batch, seq_len, num_heads]            in (0, 1)
    initial_state   : [batch, num_heads, head_dim, head_dim] fp32

    Returns (out [batch, seq_len, num_heads, head_dim], final_state).
    """
    out_dtype = query.dtype

    # Everything runs in fp32: the state is a running sum of rank-1 updates over
    # the whole sequence, so bf16 accumulation drifts badly at long context.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    # l2norm before the scale, matching the kernel's internal order.
    query = _l2norm(query)
    key = _l2norm(key)

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (k_head_dim ** 0.5)

    pad_size = (_CHUNK - seq_len % _CHUNK) % _CHUNK
    padded_len = seq_len + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, _CHUNK, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]

    # Per-channel decay accumulates within the chunk. This is the one place KDA
    # differs from Gated DeltaNet, where g is per-token and broadcast over the
    # head dim.
    g = g.cumsum(dim=-2)

    eye_mask = torch.triu(
        torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=0
    )
    strict_mask = torch.triu(
        torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=1
    )
    decay_mask = (
        (g.unsqueeze(-2) - g.unsqueeze(-3))
        .masked_fill(strict_mask[..., None], float("-inf"))
        .exp()
    )

    # WY representation: build (I - tril(K_beta K^T))^{-1} by forward
    # substitution rather than an explicit inverse. This loop is the UT
    # transform, and it is the part a real kernel replaces with a blocked
    # triangular solve.
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1)
    attn = attn.masked_fill(eye_mask, 0)
    for i in range(1, _CHUNK):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(_CHUNK, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=torch.float32, device=value.device,
        )
    else:
        state = initial_state.to(torch.float32).clone()

    core_attn_out = torch.zeros_like(value)
    causal_mask = torch.triu(
        torch.ones(_CHUNK, _CHUNK, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(padded_len // _CHUNK):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        # Contribution of everything before this chunk, carried by the state.
        attn_inter = (q_i * g_i.exp()) @ state
        # Contribution from within the chunk.
        attn_intra = (
            (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i])
            .sum(dim=-1)
            .masked_fill(causal_mask, 0)
        )
        # The delta rule: subtract what the state already predicts.
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        state = (
            state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(out_dtype)

    return core_attn_out, state
'''


DEFINITION = {
    "name": DEF_NAME,
    "description": (
        "Kimi Delta Attention (KDA), chunked prefill form. num_heads = 32, "
        "head_dim = 128 for both key and value, chunk_size = 64, q/k l2-normalised "
        "in-kernel. The linear-attention layer of Kimi Linear (3 KDA : 1 full MLA). "
        "Gated DeltaNet with a per-channel forget gate: "
        "S_t = (I - beta_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T. "
        "Consumes the recurrent state and returns the updated one, so a chunked "
        "prefill can be continued. FLOP-bound, not bandwidth-bound: rank on "
        "achieved TFLOP/s against a measured matmul peak."
    ),
    "op_type": OP_TYPE,
    "tags": [
        "stage:prefill",
        "model:kimi-linear",
        "linear-attention",
        "delta-rule",
        "status:unverified",
    ],
    "axes": {
        "batch_size": {
            "type": "var",
            "description": "Number of sequences prefilled together.",
        },
        "seq_len": {
            "type": "var",
            "description": (
                "Prefill length in tokens. Not required to be a multiple of "
                "chunk_size; the reference pads to a chunk boundary and trims the "
                "output. Unaligned lengths are included in the sweep on purpose, "
                "since a kernel that assumes alignment is wrong on real traffic."
            ),
        },
        "num_heads": {"type": "const", "value": NUM_HEADS},
        "head_dim": {"type": "const", "value": HEAD_DIM},
        "chunk_size": {"type": "const", "value": CHUNK},
    },
    "constraints": [
        f"num_heads == {NUM_HEADS}",
        f"head_dim == {HEAD_DIM}",
        f"chunk_size == {CHUNK}",
        "seq_len >= 1",
        "batch_size >= 1",
    ],
    "inputs": {
        "query": {
            "shape": ["batch_size", "seq_len", "num_heads", "head_dim"],
            "dtype": "bfloat16",
            "description": "Query after the short causal conv. L2-normalised inside the op.",
        },
        "key": {
            "shape": ["batch_size", "seq_len", "num_heads", "head_dim"],
            "dtype": "bfloat16",
            "description": "Key after the short causal conv. L2-normalised inside the op.",
        },
        "value": {
            "shape": ["batch_size", "seq_len", "num_heads", "head_dim"],
            "dtype": "bfloat16",
            "description": "Value after the short causal conv.",
        },
        "g": {
            "shape": ["batch_size", "seq_len", "num_heads", "head_dim"],
            "dtype": "float32",
            "description": (
                "Per-channel log forget gate, produced as -exp(A_log) * softplus(x) "
                "and therefore STRICTLY NEGATIVE. exp(cumsum(g)) is the decay "
                "applied to the state. Positive entries make the recurrence diverge; "
                "this input cannot be filled with randn."
            ),
        },
        "beta": {
            "shape": ["batch_size", "seq_len", "num_heads"],
            "dtype": "bfloat16",
            "description": (
                "Per-head delta-rule step size, a sigmoid output and therefore in "
                "(0, 1). Values outside that range are not a distribution shift, "
                "they are a different update rule."
            ),
        },
        "initial_state": {
            "shape": ["batch_size", "num_heads", "head_dim", "head_dim"],
            "dtype": "float32",
            "description": (
                "Recurrent state carried in. Zeros for a fresh prefill; a measured "
                "state when continuing a chunked prefill."
            ),
        },
    },
    "outputs": {
        "core_attn_out": {
            "shape": ["batch_size", "seq_len", "num_heads", "head_dim"],
            "dtype": "bfloat16",
            "description": "Attention output, before the gated RMSNorm and o_proj.",
        },
        "final_state": {
            "shape": ["batch_size", "num_heads", "head_dim", "head_dim"],
            "dtype": "float32",
            "description": (
                "Recurrent state after the last token. fp32 because it is a running "
                "sum of rank-1 updates over the whole sequence."
            ),
        },
    },
    "reference": REFERENCE,
}


# Prefill lengths. Chunk-aligned powers of two for the scaling curve, plus four
# deliberately unaligned lengths -- a kernel that silently assumes seq_len % 64
# == 0 passes a power-of-two-only sweep and fails in production.
SEQ_LENS = [
    (1, 128), (1, 256), (1, 512), (1, 1024), (1, 2048),
    (1, 4096), (1, 8192), (1, 16384),
    (1, 100), (1, 1000), (1, 3000), (1, 5000),   # unaligned
    (2, 2048), (4, 2048), (8, 1024), (2, 8192),  # batched
]


def build_workloads(seed_ns: str) -> list:
    """One workload trace per (batch_size, seq_len).

    Inputs are left unbound here -- gen_kda_blobs.py fills them in with real
    tensors and rewrites the corpus. A workload that shipped with
    ``{"type": "random"}`` would run, and would be measuring a divergent
    recurrence; see the module docstring.
    """
    ns = uuid.UUID(seed_ns)
    out = []
    for batch_size, seq_len in SEQ_LENS:
        axes = {"batch_size": batch_size, "seq_len": seq_len}
        wid = uuid.uuid5(ns, f"{DEF_NAME}:{batch_size}:{seq_len}")
        out.append(
            {
                "definition": DEF_NAME,
                "workload": {
                    "uuid": str(wid),
                    "axes": axes,
                    "inputs": {
                        name: {"type": "random"}
                        for name in DEFINITION["inputs"]
                    },
                },
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--task-dir",
        type=pathlib.Path,
        default=REPO / "tasks" / TASK_NAME,
        help="where to write the Definition and workload sweep",
    )
    ap.add_argument(
        "--namespace",
        default="6b3f1c2e-0d5a-4a7e-9c11-2f8e5d0a4b76",
        help="uuid5 namespace for stable workload ids",
    )
    args = ap.parse_args()

    task_dir = args.task_dir
    task_dir.mkdir(parents=True, exist_ok=True)

    def_path = task_dir / f"{DEF_NAME}.json"
    def_path.write_text(json.dumps(DEFINITION, indent=2) + "\n")
    print(f"definition -> {def_path.relative_to(REPO)}")

    # Parse it back through the real loader rather than trusting the dict.
    data = fib_shim.load_data()
    definition = data.Definition.model_validate(json.loads(def_path.read_text()))
    print(f"  parsed ok: {len(definition.inputs)} inputs, {len(definition.outputs)} outputs, "
          f"{len(definition.axes)} axes")

    workloads = build_workloads(args.namespace)
    wl_path = task_dir / f"{DEF_NAME}.jsonl"
    wl_path.write_text("\n".join(json.dumps(w) for w in workloads) + "\n")
    print(f"workloads  -> {wl_path.relative_to(REPO)}  ({len(workloads)} workloads)")

    for w in workloads:
        data.Trace.model_validate(w)

    print("\nNEXT: the corpus is still `random`, which for this op is not a weak "
          "input distribution but a different operator (g must be < 0). Run "
          "tools/gen_kda_blobs.py before benchmarking anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
