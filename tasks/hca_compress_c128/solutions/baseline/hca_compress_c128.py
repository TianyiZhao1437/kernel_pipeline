"""Baseline for hca_compress_c128: the eager chain handed to torch.compile.

This is the incumbent a candidate kernel has to beat, and it is deliberately
*not* a hand-written kernel. vLLM's own C128A path is Triton
(`fused_compress_quant_cache.py`), but vLLM is not a dependency of this
benchmark, and transcribing that kernel here would produce a second hand-written
Triton solution -- a competitor to the one already in the trace set, not a
baseline. What a practitioner actually reaches for first, on an op with no
library implementation, is the obvious PyTorch expression plus `torch.compile`.
So that is what this is, and the number it produces answers a question worth
asking: how much of this op does the compiler already get for free?

Answer, on an H200: quite a lot. Inductor fuses the softmax and the weighted sum
into one pass over the bf16 inputs, which is the single thing that matters on a
streaming op -- the Definition's reference instead casts both [n, 128, 512]
inputs to fp32 up front and eats two 268 MB temporaries at num_compressed=1024.

## The one place torch.compile is wrong, and why the split exists

`run` calls two compiled regions rather than one. That is not a performance
choice, it is a correctness one.

The noPE columns are rounded to bf16 *once*, and both `ckv` and the FP8 block
scales read that rounded copy -- that double rounding is what makes `ckv_fp8` a
deterministic encoding of `ckv` rather than an independent quantisation of the
fp32 value. Written inline as `normed[:, :448].to(torch.bfloat16).float()`,
**Inductor deletes it**. Measured on 2^20 random fp32 values (torch 2.11.0+cu128):

    eager   round-trip vs eager   no-round-trip : 32450/1048576 differ (3.09%)
    compiled round-trip vs eager  round-trip    : 32450/1048576 differ (3.09%)
    compiled round-trip vs eager no-round-trip  :     0/1048576 differ  <-- elided

The compiled form is bit-identical to the form with the cast deleted. It is a
silent semantic change: `ckv` still looks fine, because it rounds to bf16 on the
way out anyway, but 3.5% of `ckv_fp8` bytes come out one e4m3 code point off,
which scored matched_ratio 0.965 against the reference -- inside the band
eval_config.yaml reserves for *semantic mutations* (worst mutation: 0.96282).
`torch._inductor.config.emulate_precision_casts = True` does not prevent it.

Splitting at the bf16 tensor fixes it: `_reduce` returns real bf16 storage, so
`_quant`'s `.float()` is a load-and-convert with nothing to fold through. The
extra region reads back n x 448 bf16 -- 0.9 MB at num_compressed=1024, against
the 268 MB the op streams, so it costs nothing measurable.

Semantics are otherwise the Definition's, element for element.
"""

import torch

# Pinned by the Definition's constraints. `run` receives the five inputs and no
# axis values, so anything not derivable from an input shape is a literal.
_NOPE_HEAD_DIM = 448
_QUANT_BLOCK = 64
_RMS_NORM_EPS = 1e-6
_FP8_MAX = 448.0


def _reduce(kv_state, score_state, rms_norm_weight, cos_cache, sin_cache):
    """Softmax-compress, RMSNorm, RoPE. Returns ckv and the bf16 noPE columns."""
    num_compressed, compress_rate, head_dim = kv_state.shape
    rope_head_dim_half = (head_dim - _NOPE_HEAD_DIM) // 2

    # Softmax over the window axis, in fp32, then the weighted sum. Reading the
    # inputs as fp32 op-by-op rather than casting the tensors is what lets this
    # fuse into a single reduction over the bf16 inputs.
    weight = torch.softmax(score_state.float(), dim=1)
    compressed = (kv_state.float() * weight).sum(dim=1)

    # RMSNorm over the full head_dim -- all 512 columns, not just the 448 noPE
    # ones.
    variance = compressed.square().mean(dim=-1, keepdim=True)
    normed = compressed * torch.rsqrt(variance + _RMS_NORM_EPS)
    normed = normed * rms_norm_weight.float()

    # The angle is the window *start* -- floor(boundary / compress_rate) *
    # compress_rate, i.e. row c * compress_rate -- not the boundary token
    # c * compress_rate + compress_rate - 1.
    pos = torch.arange(num_compressed, device=kv_state.device) * compress_rate
    cos = cos_cache[pos].float()
    sin = sin_cache[pos].float()

    # GPT-J / interleaved-pair RoPE on the suffix: column 2j pairs with 2j+1,
    # both using cache entry j. Not split-half. It reads the fp32 `normed`,
    # while the FP8 path below reads the bf16-rounded copy -- keeping those two
    # reads distinct is the point of the split described in the module docstring.
    pairs = normed[:, _NOPE_HEAD_DIM:].reshape(num_compressed, rope_head_dim_half, 2)
    even = pairs[:, :, 0] * cos - pairs[:, :, 1] * sin
    odd = pairs[:, :, 1] * cos + pairs[:, :, 0] * sin
    rope = torch.stack((even, odd), dim=-1).reshape(num_compressed, 2 * rope_head_dim_half)

    nope = normed[:, :_NOPE_HEAD_DIM].to(torch.bfloat16)
    # Equivalent to cat-then-round: `nope` already holds exactly representable
    # bf16 values, so rounding it a second time is the identity.
    ckv = torch.cat((nope, rope.to(torch.bfloat16)), dim=-1)
    return ckv, nope


def _quant(nope):
    """UE8M0 block-scaled FP8 over the noPE columns of the bf16-rounded value."""
    num_compressed, nope_head_dim = nope.shape
    num_quant_blocks = nope_head_dim // _QUANT_BLOCK

    blocks = nope.float().reshape(num_compressed, num_quant_blocks, _QUANT_BLOCK)
    absmax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    # Keep the division inside the log: log2(x / 448) is not the same float as
    # log2(x) - log2(448), and the result is then rounded to an integer, so the
    # difference is a whole factor of two.
    exponent = torch.ceil(torch.log2(absmax / _FP8_MAX)).clamp(min=-127.0, max=127.0)
    scaled = blocks * torch.exp2(-exponent)[:, :, None]
    ckv_fp8 = scaled.reshape(num_compressed, nope_head_dim).clamp(-_FP8_MAX, _FP8_MAX)
    # ckv_scale carries the UNBIASED exponent as int8; vLLM's on-wire cache byte
    # is this value + 127.
    return ckv_fp8.to(torch.float8_e4m3fn), exponent.to(torch.int8)


# Default settings on purpose: the point of a torch.compile baseline is what the
# compiler gives you without tuning, so no mode=, no options=, no dynamic= hint.
# The sweep varies num_compressed only, so Inductor sees one static shape and
# switches itself to a dynamic kernel from the second shape on.
_reduce_c = torch.compile(_reduce)
_quant_c = torch.compile(_quant)


def run(kv_state, score_state, rms_norm_weight, cos_cache, sin_cache):
    """DeepSeek-V4 HCA compressor, compress_rate == 128."""
    num_compressed, _, head_dim = kv_state.shape
    if num_compressed == 0:
        # A zero-sized leading dim is a guard recompile for no work, and the
        # empty outputs are unambiguous.
        dev = kv_state.device
        return (
            torch.empty((0, head_dim), dtype=torch.bfloat16, device=dev),
            torch.empty((0, _NOPE_HEAD_DIM), dtype=torch.float8_e4m3fn, device=dev),
            torch.empty((0, _NOPE_HEAD_DIM // _QUANT_BLOCK), dtype=torch.int8, device=dev),
        )
    ckv, nope = _reduce_c(kv_state, score_state, rms_norm_weight, cos_cache, sin_cache)
    ckv_fp8, ckv_scale = _quant_c(nope)
    return ckv, ckv_fp8, ckv_scale
