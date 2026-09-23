#!/usr/bin/env python3
"""Everything in this repo that reads DeepSeek weights.

The Definition targets DeepSeek-V4's HCA compressor. It turns out that is not a
model we have to approximate: ``deepseek-ai/DeepSeek-V4-Flash`` is public and
un-gated, and transformers 5.17 carries a *native* ``deepseek_v4``
implementation -- including ``DeepseekV4HCACompressor``, which is the reference
this task was written against. So four of the five Definition inputs can be
taken exactly rather than modelled:

    Definition input      where it comes from                        exact?
    rms_norm_weight       layers.<L>.attn.compressor.norm.weight     yes
    cos_cache/sin_cache   DeepseekV4RotaryEmbedding(layer_type=      yes
                          "compress"), theta=160000, YaRN factor 16
    score_state           wgate(h) + ape                             weights yes
    kv_state              wkv(h)                                     weights yes

Only ``h`` -- the hidden state at the compressor's layer -- needs a forward
pass. See :func:`hidden_states` for how that is obtained and
:mod:`tools.gen_workload_blobs` for what is done when it cannot be.

Why V4-Flash and not V4-Pro: same ``head_dim`` (512), ``qk_rope_head_dim`` (64),
``compress_rate`` (128) and ``rms_norm_eps`` (1e-6) as the Definition, at 148 GiB
instead of 805 GiB. ``config.compress_ratios`` marks which layers are HCA
(value 128) rather than CSA (4) or dense (0).

Checkpoint naming
-----------------

The published checkpoint predates the transformers port and uses DeepSeek's own
key names (``layers.11.attn.compressor.wkv.weight``), which do *not* match the
HF module paths (``model.layers.11.self_attn.compressor.kv_proj.weight``) and
have no conversion mapping in the library. The weights here are therefore read
straight out of the safetensors shards by key, which is what we want anyway --
four small tensors out of a 3 GB shard, no model instantiation.

DeepSeek-V2-Lite is retained only as a fallback source of *activation*
statistics (:func:`v2lite_kv_latents`); its MLA geometry matches on
``kv_lora_rank``=512 and ``rms_norm_eps``, so its KV latent is the same kind of
object at 16B scale.
"""

from __future__ import annotations

import functools
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import torch

V4_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
V2_MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"

# Pinned, for the same reason third_party/ is pinned: everything this repository
# measures derives from these weights, and "whatever main pointed at that day"
# is not a provenance. A hub repo can be force-pushed or re-quantised in place.
MODEL_REVISION = {
    V4_MODEL_ID: "60d8d70770c6776ff598c94bb586a859a38244f1",
    V2_MODEL_ID: "604d5664dddd88a0433dbae533b7fe9472482de0",
}


def revision(model_id: str) -> str:
    return MODEL_REVISION[model_id]


# An HCA layer (compress_ratios[11] == 128), far enough up the stack to be
# representative of the steady state rather than of the input embedding.
DEFAULT_LAYER = 11

REPO = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# repo / hub plumbing
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=4)
def _index(model_id: str) -> Dict[str, str]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision(model_id))
    return json.loads(pathlib.Path(p).read_text())["weight_map"]


@functools.lru_cache(maxsize=4)
def config(model_id: str = V4_MODEL_ID) -> Dict:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(model_id, "config.json", revision=revision(model_id))
    return json.loads(pathlib.Path(p).read_text())


def _get_tensors(model_id: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    """Read named tensors by key, downloading only the shards that hold them."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = _index(model_id)
    missing = [k for k in keys if k not in index]
    if missing:
        raise KeyError(f"{model_id}: keys not in index: {missing}")

    by_shard: Dict[str, List[str]] = {}
    for k in keys:
        by_shard.setdefault(index[k], []).append(k)

    out: Dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        path = hf_hub_download(model_id, shard, revision=revision(model_id))
        with safe_open(path, framework="pt") as h:
            for k in shard_keys:
                out[k] = h.get_tensor(k).contiguous()
    return out


def hca_layers(model_id: str = V4_MODEL_ID) -> List[int]:
    """Indices of layers whose attention uses the HCA compressor (rate 128)."""
    ratios = config(model_id)["compress_ratios"]
    return [i for i, r in enumerate(ratios) if r == 128]


# --------------------------------------------------------------------------- #
# exact: the compressor's own weights
# --------------------------------------------------------------------------- #


def compressor_weights(layer: int = DEFAULT_LAYER, model_id: str = V4_MODEL_ID) -> Dict:
    """The four real HCA compressor tensors for one layer.

    ``norm``   [head_dim]                 -> the Definition's rms_norm_weight
    ``ape``    [compress_rate, head_dim]  -> absolute position embedding, added
                                             to the gate before the softmax.
                                             The Definition folds this into
                                             score_state (vLLM adds it in the
                                             preceding kernel), so it belongs on
                                             the generator side, not the op's.
    ``wkv``    [head_dim, hidden_size]    -> kv_state   = h @ wkv.T
    ``wgate``  [head_dim, hidden_size]    -> score_state = h @ wgate.T + ape
    """
    if layer not in hca_layers(model_id):
        raise ValueError(f"layer {layer} is not an HCA layer; try {hca_layers(model_id)[:8]}")
    pre = f"layers.{layer}.attn.compressor"
    keys = [f"{pre}.norm.weight", f"{pre}.ape", f"{pre}.wkv.weight", f"{pre}.wgate.weight"]
    t = _get_tensors(model_id, keys)
    return {
        "norm": t[keys[0]],
        "ape": t[keys[1]],
        "wkv": t[keys[2]],
        "wgate": t[keys[3]],
        "layer": layer,
        "model_id": model_id,
    }


# --------------------------------------------------------------------------- #
# exact: the compress-branch RoPE tables
# --------------------------------------------------------------------------- #


def rope_tables(
    max_position: int, model_id: str = V4_MODEL_ID
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4's *compress-branch* cos/sin tables, in the Definition's layout.

    Built by the library's own ``DeepseekV4RotaryEmbedding`` at
    ``layer_type="compress"`` rather than by a transcription, so the YaRN
    parameters (theta 160000, factor 16, beta 32/1, original context 65536) come
    from the checkpoint's config and cannot drift. V4 pins the compress
    branch's ``attention_factor`` to 1.0 -- the reference does not apply YaRN's
    mscale -- so the tables stay on the unit circle, which is the property
    ``{"type": "random"}`` violates by construction.

    Returns ``(cos, sin)``, each ``[max_position, rope_head_dim_half]`` float32.
    V4's rotary is already interleaved-pair (one entry per pair, no
    ``repeat_interleave`` until ``apply_rotary_pos_emb``), so this is exactly the
    Definition's half-width layout with no slicing.
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    cfg = DeepseekV4Config(**config(model_id))
    rot = DeepseekV4RotaryEmbedding(cfg)
    dummy = torch.zeros(1, 1, cfg.head_dim)
    pos = torch.arange(max_position).unsqueeze(0)
    cos, sin = rot(dummy, position_ids=pos, layer_type="compress")
    return cos[0].float().contiguous(), sin[0].float().contiguous()


# --------------------------------------------------------------------------- #
# activations
# --------------------------------------------------------------------------- #


def v2lite_latents_path(layer: int, num_tokens: int) -> pathlib.Path:
    # The corpus digest is in the filename on purpose. The harvest is a pure
    # function of (model, layer, num_tokens, corpus), and the first three were
    # already in the name; leaving the fourth out is what let a cache written
    # against one corpus be served silently after the corpus changed.
    return (
        REPO
        / "data"
        / "latents"
        / f"v2lite_L{layer}_n{num_tokens}_c{corpus_digest()}.safetensors"
    )


@torch.inference_mode()
def harvest_v2lite(
    layer: int = 13, num_tokens: int = 16384, seq_len: int = 2048
) -> Dict[str, torch.Tensor]:
    """Run DeepSeek-V2-Lite and capture real hidden states and a real KV latent.

    Returns ``{"hidden": [T, 2048], "kv_latent": [T, 512]}``, both bf16 on CPU.

    * ``hidden``    -- the residual-stream input to layer ``layer``'s attention,
      i.e. exactly the ``h`` that an HCA compressor's ``wkv``/``wgate`` would be
      applied to, at 2048 channels instead of V4's 4096.
    * ``kv_latent`` -- the first ``kv_lora_rank`` (512) columns of
      ``kv_a_proj_with_mqa``'s output: the MLA compressed KV vector, which is
      the same *kind* of object the HCA compressor reduces windows of.

    Loaded through the library's native ``deepseek_v2`` architecture, so no
    remote code is executed: transformers 5.17 implements DeepSeek-V2 directly,
    and the checkpoint's ``auto_map`` is ignored when ``trust_remote_code`` is
    not set.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config(V2_MODEL_ID)
    rank = cfg["kv_lora_rank"]

    tok = AutoTokenizer.from_pretrained(V2_MODEL_ID, revision=revision(V2_MODEL_ID))
    ids = tok(corpus_text(), return_tensors="pt").input_ids[0]
    if ids.numel() < num_tokens:
        ids = ids.repeat(-(-num_tokens // max(int(ids.numel()), 1)))
    ids = ids[:num_tokens]

    model = AutoModelForCausalLM.from_pretrained(
        V2_MODEL_ID,
        revision=revision(V2_MODEL_ID),
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    hidden: List[torch.Tensor] = []
    latent: List[torch.Tensor] = []
    attn = model.model.layers[layer].self_attn

    def pre_hook(_mod, args, kwargs):
        h = kwargs.get("hidden_states", args[0] if args else None)
        hidden.append(h.reshape(-1, h.shape[-1]).to("cpu", torch.bfloat16))

    def kv_hook(_mod, _inp, out):
        latent.append(out[..., :rank].reshape(-1, rank).to("cpu", torch.bfloat16))

    handles = [
        attn.register_forward_pre_hook(pre_hook, with_kwargs=True),
        attn.kv_a_proj_with_mqa.register_forward_hook(kv_hook),
    ]
    try:
        for start in range(0, num_tokens, seq_len):
            chunk = ids[start : start + seq_len]
            if chunk.numel() == 0:
                break
            model(chunk.unsqueeze(0).to("cuda"))
    finally:
        for h in handles:
            h.remove()

    out = {
        "hidden": torch.cat(hidden)[:num_tokens].contiguous(),
        "kv_latent": torch.cat(latent)[:num_tokens].contiguous(),
    }
    del model
    torch.cuda.empty_cache()
    return out


def v2lite_activations(layer: int = 13, num_tokens: int = 16384) -> Dict[str, torch.Tensor]:
    """:func:`harvest_v2lite`, memoised on disk under ``data/``.

    The forward pass costs a 29 GB load and a minute of GPU; the result is
    ~80 MB. Everything downstream reads the cache, so regenerating blobs never
    re-runs the model.
    """
    from safetensors.torch import load_file, save_file

    path = v2lite_latents_path(layer, num_tokens)
    if path.exists():
        return load_file(str(path))
    out = harvest_v2lite(layer=layer, num_tokens=num_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(path))
    return out


def channel_profile(x: torch.Tensor) -> Dict[str, float]:
    """Summarise the per-channel scale structure of an activation matrix.

    ``spread`` and ``p99_over_median`` are the numbers that matter downstream:
    they say how far a block-scaled quantiser's per-block exponents must range,
    which is precisely the coverage the random sweep lacks.
    """
    f = x.float()
    sd = f.std(dim=0)
    med = sd.median()
    flat = f.reshape(-1)
    return {
        "channels": int(f.shape[1]),
        "std_median": float(med),
        "std_max": float(sd.max()),
        "std_min": float(sd.min()),
        "spread": float(sd.max() / sd.min()),
        "p99_over_median": float(sd.quantile(0.99) / med),
        "kurtosis": float(((flat - flat.mean()) / flat.std()).pow(4).mean()),
        "massive_frac": float((sd > 10 * med).float().mean()),
    }


CORPUS_PATH = REPO / "tools" / "corpus" / "hca_v1.txt"

# Asserted, not merely recorded. Everything downstream -- the harvested
# activations, the workload blobs built from them, and the traces measured on
# those blobs -- is a pure function of this text, so a silent edit would
# invalidate committed numbers with no other symptom.
CORPUS_SHA256 = "3e3b3cf574217c6921adee3189981483d3e7fe599afea42262318cdf43a46b8f"


def corpus_text() -> str:
    """Real text to drive a forward pass with.

    Activation statistics only mean anything on in-distribution input: random
    token ids give a garbage hidden state and therefore garbage channel scales.
    This is real technical code and prose -- squarely in a code model's training
    distribution -- and needs no dataset download.

    It is a *frozen snapshot*, and that is the point. An earlier version built
    the corpus live from this task's own design write-up plus the installed
    transformers' ``modeling_deepseek_v4`` source. Both drift: the first was a
    document the repository edited constantly, making the workload data a
    function of its own task write-up, and the second changes with a library
    upgrade. The harvest is memoised on disk, so the drift was invisible -- the
    cache here was written at 12:40 and the write-up was edited at 13:51 the
    same day, and nothing said so.

    Provenance of the snapshot: the concatenation of sixteen files from the
    vendored flashinfer-bench tree at the pin recorded in third_party/VENDOR.md
    (four data-schema modules, the bench config, the builder, four trace-format
    docs, five op_type specs, and the README), each prefixed with its path.
    Those particular files were chosen because none of them is touched by
    third_party/patches/, so the snapshot does not move when the patch series
    does. The snapshot is nonetheless the authority; the file list is only how it
    was produced.
    """
    import hashlib

    if not CORPUS_PATH.exists():
        raise RuntimeError(f"corpus snapshot missing: {CORPUS_PATH}")
    text = CORPUS_PATH.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != CORPUS_SHA256:
        raise RuntimeError(
            f"corpus snapshot changed: {CORPUS_PATH}\n"
            f"  expected sha256 {CORPUS_SHA256}\n"
            f"  found    sha256 {digest}\n"
            "Every blob and every recorded trace derives from this text. If the "
            "change is intended, bump CORPUS_SHA256, regenerate the blobs "
            "(tools/gen_workload_blobs.py) and re-run the sweep "
            "(tools/run_benchmark.py) -- the old traces no longer describe the "
            "new inputs."
        )
    return text


def corpus_digest() -> str:
    """Short digest of the corpus, used to key the activation cache."""
    return CORPUS_SHA256[:12]


def describe(t: torch.Tensor) -> str:
    f = t.float()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={f.min():.4g} max={f.max():.4g} mean={f.mean():.4g} "
        f"absmax={f.abs().max():.4g} neg={int((f < 0).sum())}/{f.numel()}"
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the real DeepSeek-V4 HCA compressor.")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument(
        "--harvest", action="store_true", help="also run DeepSeek-V2-Lite and cache activations"
    )
    args = ap.parse_args()

    cfg = config()
    print(f"model {V4_MODEL_ID}")
    print(
        f"  head_dim={cfg['head_dim']} qk_rope_head_dim={cfg['qk_rope_head_dim']} "
        f"rms_norm_eps={cfg['rms_norm_eps']} compress_rope_theta={cfg['compress_rope_theta']}"
    )
    hl = hca_layers()
    print(f"  {len(hl)} HCA layers (rate 128): {hl[:6]} ... {hl[-2:]}")

    w = compressor_weights(args.layer)
    print(f"\ncompressor weights, layer {args.layer}")
    for k in ("norm", "ape", "wkv", "wgate"):
        print(f"  {k:6s} {describe(w[k])}")

    nf = w["norm"].float()
    print(f"\n  norm.weight spread max/|min| = {nf.max() / nf.abs().min():.1f}x")
    print(f"  norm.weight negative channels = {int((nf < 0).sum())}/512 (randn would be ~256)")

    cos, sin = rope_tables(256)
    print(f"\ncompress rope tables (max_position=256)\n  cos {describe(cos)}\n  sin {describe(sin)}")
    print(f"  max |cos^2+sin^2-1| = {(cos.square() + sin.square() - 1).abs().max():.3e}")

    if args.harvest:
        print(f"\nharvesting DeepSeek-V2-Lite activations ...")
        act = v2lite_activations()
        for k, v in act.items():
            print(f"  {k:10s} {describe(v)}")
            prof = channel_profile(v)
            print("             " + "  ".join(f"{a}={b:.4g}" for a, b in prof.items()))
        print(f"  cached at {v2lite_latents_path(13, 16384)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
