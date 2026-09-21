#!/usr/bin/env python3
"""Regenerate the HCA C128A Definition JSON.

The reference lives here as a normal Python string so it stays readable and
lintable; hand-escaping a 100-line function into JSON is how the earlier
reference acquired silent edit failures.
"""

import json
import pathlib

REFERENCE = '''
import torch

# Pinned by the Definition's constraints; the harness calls `run` with exactly
# the five inputs and passes no axis values, so anything not derivable from an
# input shape has to be a literal here.
_NOPE_HEAD_DIM = 448
_QUANT_BLOCK = 64
_RMS_NORM_EPS = 1e-6
_FP8_MAX = 448.0


def run(kv_state, score_state, rms_norm_weight, cos_cache, sin_cache):
    """DeepSeek-V4 HCA compressor, compress_rate == 128.

    Each window of ``compress_rate`` consecutive tokens collapses into one KV
    cache entry. The window axis is explicit in the input layout, so entry c is
    simply ``kv_state[c]`` -- for a contiguous ``[num_compressed * compress_rate,
    head_dim]`` buffer, as vLLM keeps it, this 3-D form is a zero-copy view over
    the same bytes.

    1. Softmax across the compress_rate axis of ``score_state``. vLLM loads
       out-of-range window positions as -inf, which for a packed layout is
       exactly the masked-out tail of a partial window.
    2. Weighted sum of ``kv_state`` by those weights.
    3. RMSNorm over all ``head_dim`` elements, in fp32.
    4. GPT-J (interleaved pair) RoPE on the last ``rope_head_dim`` elements.
       This is NOT split-half rotation: element 2j pairs with element 2j+1 and
       both use cache entry j.
    5. UE8M0 block-scaled FP8 over the first ``nope_head_dim`` elements.

    Step 5 reads the bf16-rounded noPE values, matching the kernel, which is
    what makes ckv_fp8 a deterministic encoding of ckv. Step 4 reads the fp32
    values, also matching the kernel (fused_compress_quant_cache.py:314 reshapes
    `normed`, not `quant_input`).
    """
    num_compressed, compress_rate, head_dim = kv_state.shape
    nope_head_dim = _NOPE_HEAD_DIM
    rope_head_dim = head_dim - nope_head_dim
    rope_head_dim_half = rope_head_dim // 2
    num_quant_blocks = nope_head_dim // _QUANT_BLOCK
    dev = kv_state.device

    if num_compressed == 0:
        return (
            torch.empty((0, head_dim), dtype=torch.bfloat16, device=dev),
            torch.empty((0, nope_head_dim), dtype=torch.float8_e4m3fn, device=dev),
            torch.empty((0, num_quant_blocks), dtype=torch.int8, device=dev),
        )

    kv = kv_state.to(torch.float32)
    score = score_state.to(torch.float32)

    # 1 + 2: softmax over the window, then the weighted sum. Subtracting the
    # window max first is a no-op mathematically and keeps exp() in range.
    score = score - score.amax(dim=1, keepdim=True)
    weight = torch.exp(score)
    weight = weight / weight.sum(dim=1, keepdim=True)
    compressed = (kv * weight).sum(dim=1)

    # 3: RMSNorm over the full head_dim, not just the noPE part.
    variance = compressed.square().mean(dim=-1, keepdim=True)
    normed = compressed * torch.rsqrt(variance + _RMS_NORM_EPS)
    normed = normed * rms_norm_weight.to(torch.float32)

    # The kernel rounds to bf16 before encoding -- that is what makes ckv_fp8 a
    # deterministic encoding of ckv, which also stores bf16 -- but it rotates
    # the fp32 values, so the two slices are rounded at different points.
    nope = normed[:, :nope_head_dim].to(torch.bfloat16).to(torch.float32)
    rope = normed[:, nope_head_dim:]

    # 4: GPT-J RoPE, interleaved pairs, on the suffix only. Both the stored
    # suffix and ckv hold the ROTATED values. The angle comes from the window's
    # boundary token floored to the window start: entry c reads cos/sin row
    # ``c * compress_rate``, not row ``c * compress_rate + compress_rate - 1``.
    boundary = (torch.arange(num_compressed, device=dev) + 1) * compress_rate - 1
    compressed_pos = (boundary // compress_rate) * compress_rate
    cos = cos_cache[compressed_pos].to(torch.float32)
    sin = sin_cache[compressed_pos].to(torch.float32)
    pairs = rope.reshape(num_compressed, rope_head_dim_half, 2)
    even = pairs[:, :, 0] * cos - pairs[:, :, 1] * sin
    odd = pairs[:, :, 1] * cos + pairs[:, :, 0] * sin
    rope = torch.stack((even, odd), dim=-1).reshape(num_compressed, rope_head_dim)

    ckv = torch.cat((nope, rope), dim=-1).to(torch.bfloat16)

    # 5: UE8M0 block scales over the noPE part. ckv_scale carries the UNBIASED
    # exponent; vLLM's on-wire cache byte is ``ckv_scale + 127``. The schema has
    # no unsigned dtype, and int8 keeps the output one byte wide, so the store
    # traffic a kernel is measured on is the same as vLLM's.
    blocks = nope.reshape(num_compressed, num_quant_blocks, _QUANT_BLOCK)
    absmax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(absmax / _FP8_MAX)).clamp(min=-127.0, max=127.0)
    scaled = blocks * torch.exp2(-exponent)[:, :, None]
    ckv_fp8 = scaled.reshape(num_compressed, nope_head_dim).clamp(-_FP8_MAX, _FP8_MAX)
    ckv_fp8 = ckv_fp8.to(torch.float8_e4m3fn)
    ckv_scale = exponent.to(torch.int8)

    return ckv, ckv_fp8, ckv_scale
'''.lstrip()

DEFINITION = {
    "name": "hca_compress_c128_h512_r64",
    "description": (
        "DeepSeek-V4 HCA (hierarchical compressed attention) compressor, compress_rate = 128, "
        "head_dim = 512, rope_head_dim = 64. Every window of 128 tokens folds into one KV-cache "
        "entry: softmax over the score state, weighted sum of the kv state, RMSNorm over head_dim, "
        "GPT-J interleaved-pair RoPE on the last 64 elements only (rotated, not passed through), "
        "and UE8M0 block-scaled FP8 encoding of the 448-element noPE part in blocks of 64. Mirrors "
        "the fused vLLM kernel `fused_compress_quant_cache` for the C128A (HCA) path, including its "
        "non-split-half RoPE convention and its boundary-token gate.\n\n"
        "Two deliberate departures from vLLM's in-memory form, both forced by the schema and "
        "neither changing the bytes a kernel moves:\n"
        "  * kv_state/score_state are [num_compressed, compress_rate, head_dim] rather than a flat "
        "[total_tokens, head_dim]. Output shapes are resolved only from input shapes "
        "(get_axes_values_from_inputs), so the compression factor has to be structural; for a "
        "contiguous buffer the 3-D form is a zero-copy view.\n"
        "  * ckv_scale holds the UNBIASED UE8M0 exponent as int8. vLLM stores exponent + 127 as "
        "uint8; the schema has no unsigned dtype, and int8 preserves the one-byte width.\n\n"
        "rms_norm_eps is 1e-6. It is not an axis because axis values must be integers."
    ),
    "op_type": "hca_compress",
    "tags": [
        "stage:prefill",
        "stage:decode",
        "model:deepseek-v4",
        "quantization:float8_e4m3fn",
        "fused",
        "status:draft",
    ],
    "axes": {
        "num_compressed": {
            "type": "var",
            "description": (
                "Number of compressed KV-cache entries, i.e. number of complete 128-token windows. "
                "Derived from kv_state.shape[0]."
            ),
        },
        "max_position": {
            "type": "var",
            "description": (
                "Rows in the RoPE cos/sin tables. Must exceed the boundary position of the last "
                "entry, i.e. max_position > (num_compressed - 1) * compress_rate."
            ),
        },
        "compress_rate": {
            "type": "const",
            "value": 128,
            "description": "Tokens folded into one compressed entry. 4 is CSA, 128 is HCA.",
        },
        "head_dim": {
            "type": "const",
            "value": 512,
            "description": "State width and compressed entry width.",
        },
        "nope_head_dim": {
            "type": "const",
            "value": 448,
            "description": "Size of the non-positional part, quantized to FP8.",
        },
        "rope_head_dim": {
            "type": "const",
            "value": 64,
            "description": "Size of the RoPE-rotated suffix.",
        },
        "rope_head_dim_half": {
            "type": "const",
            "value": 32,
            "description": (
                "Number of RoPE pairs, and therefore the width of one cos/sin cache row under the "
                "interleaved-pair convention."
            ),
        },
        "quant_block": {
            "type": "const",
            "value": 64,
            "description": "Elements per FP8 scale block.",
        },
        "num_quant_blocks": {
            "type": "const",
            "value": 7,
            "description": "nope_head_dim / quant_block = 448 / 64.",
        },
    },
    "constraints": [
        "head_dim == nope_head_dim + rope_head_dim",
        "head_dim == 512",
        "rope_head_dim == 64",
        "rope_head_dim_half == rope_head_dim // 2",
        "compress_rate == 128",
        "quant_block == 64",
        "num_quant_blocks == nope_head_dim // quant_block",
        "nope_head_dim % quant_block == 0",
        "max_position > (num_compressed - 1) * compress_rate",
        "kv_state.shape == [num_compressed, compress_rate, head_dim]",
        "score_state.shape == [num_compressed, compress_rate, head_dim]",
        "ckv.shape == [num_compressed, head_dim]",
        "ckv_fp8.shape == [num_compressed, nope_head_dim]",
        "ckv_scale.shape == [num_compressed, num_quant_blocks]",
        "cos_cache.shape == sin_cache.shape",
        "cos_cache.shape == [max_position, rope_head_dim_half]",
    ],
    "inputs": {
        "kv_state": {
            "shape": ["num_compressed", "compress_rate", "head_dim"],
            "dtype": "bfloat16",
            "description": (
                "Compressor kv state, one row per token, grouped by window. A zero-copy view of "
                "vLLM's contiguous [total_tokens, head_dim] buffer."
            ),
        },
        "score_state": {
            "shape": ["num_compressed", "compress_rate", "head_dim"],
            "dtype": "bfloat16",
            "description": (
                "Compressor score state, already carrying the absolute position embedding. vLLM "
                "fuses `score += ape[position % compress_rate]` into the earlier "
                "_SAVE_PARTIAL_STATES_KERNEL, so the APE add is outside this task's boundary."
            ),
        },
        "rms_norm_weight": {
            "shape": ["head_dim"],
            "dtype": "bfloat16",
            "description": "RMSNorm gain, applied over all head_dim elements.",
        },
        "cos_cache": {
            "shape": ["max_position", "rope_head_dim_half"],
            "dtype": "float32",
            "description": (
                "cos table for the interleaved-pair RoPE, one row per absolute position and one "
                "column per PAIR. Entry c reads row c * compress_rate -- the window's boundary "
                "token floored to the window start, matching "
                "`(positions // compress_ratio) * compress_ratio` in vLLM. Note that vLLM keeps a "
                "single cos_sin_cache of width rope_head_dim with cos in the first half and sin in "
                "the second; this Definition splits it into two tensors."
            ),
        },
        "sin_cache": {
            "shape": ["max_position", "rope_head_dim_half"],
            "dtype": "float32",
            "description": "sin table, same layout and indexing rule as cos_cache.",
        },
    },
    "outputs": {
        "ckv": {
            "shape": ["num_compressed", "head_dim"],
            "dtype": "bfloat16",
            "description": (
                "Compressed KV entries. The first nope_head_dim columns are the bf16-rounded noPE "
                "values that ckv_fp8 encodes; the last rope_head_dim columns hold the ROTATED "
                "suffix, not the pre-rotation values."
            ),
        },
        "ckv_fp8": {
            "shape": ["num_compressed", "nope_head_dim"],
            "dtype": "float8_e4m3fn",
            "description": (
                "UE8M0 block-scaled FP8 encoding of ckv's noPE columns, 64 elements per scale. "
                "e4m3 keeps 3 mantissa bits, so this is lossy by construction: dequantising gives "
                "back ckv only to within half an e4m3 ulp, and further for elements that fall into "
                "the denormal range of their block."
            ),
        },
        "ckv_scale": {
            "shape": ["num_compressed", "num_quant_blocks"],
            "dtype": "int8",
            "description": (
                "Per-block UE8M0 exponent, UNBIASED: dequantise with "
                "`ckv_fp8 * 2 ** ckv_scale`. vLLM's on-wire cache byte is `ckv_scale + 127`; the "
                "schema has no unsigned dtype and int8 keeps the width at one byte. Computed as "
                "`ceil(log2(max(absmax, 1e-4) / 448))`, clamped to [-127, 127]."
            ),
        },
    },
    "reference": REFERENCE,
}

out = pathlib.Path(__file__).resolve().parent.parent / "tasks/hca_compress_c128/hca_compress_c128_h512_r64.json"
out.write_text(json.dumps(DEFINITION, indent=2) + "\n")
print(f"wrote {out} ({out.stat().st_size} bytes)")

# --- workloads -------------------------------------------------------------
# num_compressed is the only real degree of freedom: it is kv_state.shape[0],
# and total_tokens = num_compressed * 128. The sweep is geometric with the
# 1.5x midpoints filled in, from a single window (the degenerate case, where a
# kernel that assumes multiple programs breaks) up to 1024 windows = 128k
# tokens, which is DeepSeek-V4's context length.
COMPRESS_RATE = 128
NUM_COMPRESSED = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

wl_path = out.with_suffix(".jsonl")
with wl_path.open("w") as fh:
    for i, n in enumerate(NUM_COMPRESSED):
        fh.write(
            json.dumps(
                {
                    "definition": DEFINITION["name"],
                    "solution": None,
                    "workload": {
                        "uuid": f"00000000-0000-4128-8000-{i:012x}",
                        # max_position only has to exceed the last entry's
                        # boundary position; sizing it to the token count is
                        # what a real deployment does.
                        "axes": {"num_compressed": n, "max_position": n * COMPRESS_RATE},
                        "inputs": {
                            k: {"type": "random"}
                            for k in DEFINITION["inputs"]
                        },
                    },
                    "evaluation": None,
                }
            )
            + "\n"
        )
print(f"wrote {wl_path} ({len(NUM_COMPRESSED)} workloads, "
      f"num_compressed {NUM_COMPRESSED[0]}..{NUM_COMPRESSED[-1]})")
