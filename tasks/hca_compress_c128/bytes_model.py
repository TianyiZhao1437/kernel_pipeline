"""Bytes the kernel actually moves for one workload of hca_compress_c128.

``tools/roofline.py`` uses this if a task provides it. The generic fallback --
sum every declared input and output tensor -- over-counts badly here, because
``cos_cache`` and ``sin_cache`` are declared over ``max_position`` (the whole
RoPE table, tens of MB) while the kernel reads exactly one row per compressed
token. At num_compressed=1024 the declared footprint is 303.5 MB against 136.5
MB actually streamed, so the generic number would understate achieved bandwidth
by 2.2x and flatter the kernel.

Every other tensor is streamed in full: kv_state and score_state are read once
end to end, and all three outputs are written once.

Two solutions -- ``openai-gpt-6-astra`` and ``qwen3.8-max`` -- split the work
across two kernels and hand each other a ``(num_compressed, head_dim)`` fp32
buffer. Writing it and reading it back is 2 * num_compressed * 512 * 4 bytes of
real DRAM traffic that the declared operands do not describe, so leaving it out
understates achieved bandwidth. It is 1.6% of the total at every point in the
sweep (the buffer scales with num_compressed exactly as the inputs do), which is
small but systematic and always in the flattering direction -- and it is what
the comparison against the single-kernel solutions turns on, so it has to be
counted. ``streamed_bytes`` takes the solution name to add it; solutions that
fuse the two stages into one kernel pass through untouched.
"""

ITEMSIZE = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "bool": 1,
}

# Read one row per compressed token, not the whole table.
GATHERED = {"cos_cache", "sin_cache"}

# Solutions that stage the normalized/rotated row through global memory as fp32
# between a compress kernel and a finish kernel, instead of fusing the two. One
# write plus one read of (num_compressed, head_dim) fp32 per workload.
#
# Named as they appear in `traces/`, e.g.
# "hca_compress_c128_gpt_6_astra_triton_optimized". Matched by substring rather
# than equality: a generation re-run appends its round suffix to the same stem,
# and every round of both runs uses the same two-kernel structure.
STAGED_FP32_SOLUTIONS = ("gpt_6_astra", "qwen3_8_max")
_HEAD_DIM = 512


def _stages_fp32_buffer(solution: str | None) -> bool:
    """True when `solution` is a two-kernel variant that stages fp32 in DRAM.

    The other three -- claude_opus_5, triton_h200 and torch_compile -- fuse the
    two stages, so they move strictly what the Definition declares.
    """
    if not solution:
        return False
    return any(tag in solution for tag in STAGED_FP32_SOLUTIONS)


def streamed_bytes(definition, axes, solution: str | None = None) -> int:
    named = list(zip(definition.inputs.items(), definition.get_input_shapes(axes))) + list(
        zip(definition.outputs.items(), definition.get_output_shapes(axes))
    )
    total = 0
    for (name, spec), shape in named:
        # get_*_shapes resolves const axes from the Definition and var axes from
        # the workload, so shape is fully concrete here.
        rows = axes["num_compressed"] if name in GATHERED else shape[0]
        elems = rows
        for extent in shape[1:]:
            elems *= extent
        total += elems * ITEMSIZE[spec.dtype.value]

    if _stages_fp32_buffer(solution):
        total += 2 * axes["num_compressed"] * _HEAD_DIM * ITEMSIZE["float32"]
    return total
