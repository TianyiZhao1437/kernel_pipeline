#!/usr/bin/env python3
"""Generate realistic KDA inputs from the released model's own modules.

Why not just pick distributions
-------------------------------

Every one of the six inputs to this op has structure that ``torch.randn``
destroys, and in three cases the destruction is not a distribution shift but a
change of operator:

* ``g`` is a log decay, ``-exp(A_log) * softplus(...)``, so it is strictly
  negative. Gaussian ``g`` makes half the channels *grow* under
  ``exp(cumsum(g))`` and the state diverges.
* ``beta`` is a sigmoid, so it is in (0, 1). Outside that range the delta rule
  over- or back-steps rather than interpolating.
* ``query``/``key``/``value`` come out of a depthwise causal conv followed by
  **SiLU**, so they are floored at -0.2785 and right-skewed (measured skewness
  1.63, mean +0.070 against std 0.295 -- SiLU preserves sign, so the count of
  negatives is still half; what changes is that the negative tail is clipped
  while the positive one is not). This is easy to miss because q and k are
  L2-normalised inside the op, which hides the scale change but not the
  geometry.

One deviation from the init recipe, made deliberately
-----------------------------------------------------

``_init_weights`` applies ``initializer_range = 0.02`` to the depthwise
``conv1d`` along with every other weight. For a 4-tap depthwise filter that is a
~25x attenuation: it drops the pre-activation std to 0.038, which is deep inside
SiLU's linear regime, and the third bullet above stops being true (measured
skewness falls to 0.20 and nothing reaches the -0.2785 floor). A *trained*
depthwise conv has O(1) taps -- it is a short smoothing/shift filter, not a
projection -- so the squashed version is an artifact of step-0 init rather than a
property of the model. PyTorch's own ``nn.Conv1d`` default (kaiming-uniform over
fan_in=4) gives taps of std 0.29 and a pre-activation std of 0.55, which puts
SiLU in the regime it actually operates in. That default is kept and the rest of
the recipe applied as written.

Rather than approximate any of that, this builds the real
``KimiLinearDeltaAttention`` module from the real ``KimiLinearConfig``, applies
the real ``_init_weights`` recipe (``A_log = log(U(1,16))``, ``dt_bias`` =
inverse-softplus of ``LogUniform(1e-3, 1e-1)``, Linears at
``initializer_range``), and reads the tensors off the call site. Weights are
random-initialised rather than the released checkpoint -- the checkpoint is
~48 B parameters and the point here is the *shape* of the inputs, which the init
recipe already fixes -- but every transformation between a hidden state and the
op's arguments is the model's own code, so nothing can drift out of agreement
with it except by transformers changing.

Sanity of the result, measured: ``g`` lands in about (-3, -6e-4) and ``beta`` in
about (0.03, 0.96), matching the paper's intent of per-channel decay close to 1
per token.
"""

from __future__ import annotations

import torch

HIDDEN_SIZE = 2304
NUM_HEADS = 32
HEAD_DIM = 128

INPUT_ORDER = ("query", "key", "value", "g", "beta", "initial_state")


def _build_layer(device: torch.device, seed: int):
    """A single, really-initialised KDA layer."""
    from transformers.models.kimi_linear.configuration_kimi_linear import KimiLinearConfig
    from transformers.models.kimi_linear.modeling_kimi_linear import (
        KimiLinearDeltaAttention,
        KimiLinearPreTrainedModel,
    )

    torch.manual_seed(seed)
    config = KimiLinearConfig(num_hidden_layers=1, layer_types=["linear_attention"])
    layer = KimiLinearDeltaAttention(config, 0)
    # See the module docstring: the model's init recipe squashes the depthwise
    # conv into SiLU's linear regime, which a trained filter is not in. Keep
    # PyTorch's default taps and apply the recipe to everything else.
    conv_taps = layer.conv1d.weight.detach().clone()
    # PreTrainedModel.__init__ only stores the config; it builds no layers, so
    # this is a cheap handle on the model's real init recipe.
    initializer = KimiLinearPreTrainedModel(config)
    for sub in layer.modules():
        initializer._init_weights(sub)
    with torch.no_grad():
        layer.conv1d.weight.copy_(conv_taps)
    return layer.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def _call_site_tensors(layer, hidden_states):
    """Reproduce KimiLinearDeltaAttention.forward up to the KDA call.

    Deliberately mirrors the source line for line (the conv path taken is the
    "simple full prefill" branch, cache_params=None) instead of calling forward,
    because forward consumes the op we are trying to feed.
    """
    from transformers.models.kimi_linear.modeling_kimi_linear import causal_conv1d_fn

    batch_size, seq_len = hidden_states.shape[:2]
    hidden_shape = (batch_size, seq_len, -1, layer.head_dim)

    mixed_qkv = torch.cat(
        [layer.q_proj(hidden_states), layer.k_proj(hidden_states), layer.v_proj(hidden_states)],
        dim=-1,
    ).transpose(1, 2)
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv,
        weight=layer.conv1d.weight.squeeze(1),
        bias=layer.conv1d.bias,
        activation=layer.activation,
    )
    mixed_qkv = mixed_qkv[:, :, -seq_len:]

    query, key, value = torch.split(
        mixed_qkv.transpose(1, 2), [layer.qkv_dim] * 3, dim=-1
    )
    query = query.view(hidden_shape)
    key = key.view(hidden_shape)
    value = value.view(hidden_shape)

    g = layer.forget_gate(hidden_states)
    beta = torch.sigmoid(layer.b_proj(hidden_states))
    return query, key, value, g, beta


@torch.no_grad()
def make_inputs(
    batch_size: int,
    seq_len: int,
    *,
    seed: int,
    device: str | torch.device = "cuda",
    warm_state: bool = False,
    warm_len: int = 512,
    reference=None,
):
    """Return the six inputs, in Definition order, at their declared dtypes.

    ``warm_state`` produces a genuine non-zero ``initial_state`` by running the
    reference over a synthetic prefix of ``warm_len`` tokens, which is what a
    chunked prefill continuation actually carries. Zeros otherwise -- a fresh
    prefill. A state made of ``randn`` would be neither.
    """
    device = torch.device(device)
    layer = _build_layer(device, seed)

    generator = torch.Generator(device=device).manual_seed(seed + 1)
    hidden = torch.randn(
        batch_size, seq_len, HIDDEN_SIZE, device=device, generator=generator
    )
    query, key, value, g, beta = _call_site_tensors(layer, hidden)

    state_shape = (batch_size, NUM_HEADS, HEAD_DIM, HEAD_DIM)
    if warm_state:
        if reference is None:
            raise ValueError("warm_state=True needs the reference `run` to produce a state")
        prefix_hidden = torch.randn(
            batch_size, warm_len, HIDDEN_SIZE, device=device, generator=generator
        )
        pq, pk, pv, pg, pbeta = _call_site_tensors(layer, prefix_hidden)
        zeros = torch.zeros(state_shape, device=device, dtype=torch.float32)
        _, initial_state = reference(
            pq.to(torch.bfloat16),
            pk.to(torch.bfloat16),
            pv.to(torch.bfloat16),
            pg.to(torch.float32),
            pbeta.to(torch.bfloat16),
            zeros,
        )
        initial_state = initial_state.to(torch.float32)
    else:
        initial_state = torch.zeros(state_shape, device=device, dtype=torch.float32)

    return {
        "query": query.to(torch.bfloat16).contiguous(),
        "key": key.to(torch.bfloat16).contiguous(),
        "value": value.to(torch.bfloat16).contiguous(),
        "g": g.to(torch.float32).contiguous(),
        "beta": beta.to(torch.bfloat16).contiguous(),
        "initial_state": initial_state.contiguous(),
    }


def describe(inputs: dict) -> str:
    """One line per tensor, for the provenance record and for eyeballing."""
    lines = []
    for name in INPUT_ORDER:
        t = inputs[name]
        lines.append(
            f"  {name:<14} {str(tuple(t.shape)):<26} {str(t.dtype).replace('torch.',''):<10} "
            f"min={t.float().min():+.4g} max={t.float().max():+.4g} "
            f"mean={t.float().mean():+.4g}"
        )
    return "\n".join(lines)
