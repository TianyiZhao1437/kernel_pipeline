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


def streamed_bytes(definition, axes) -> int:
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
    return total
