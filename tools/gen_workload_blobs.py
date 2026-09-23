#!/usr/bin/env python3
"""Generate realistic workload inputs for tasks/hca_compress_c128 as safetensors blobs.

The problem
-----------

Every workload in the sweep declares ``{"type": "random"}`` for all five inputs,
and ``RandomInput`` has *no parameters* -- not a distribution, not a seed, not a
scale (flashinfer_bench/data/workload.py). So "random" means exactly
``torch.randn`` (bench/utils.py::_rand_tensor). For this Definition that makes
two of the five inputs physically impossible and a third unrepresentative:

* ``cos_cache``/``sin_cache`` as N(0,1) are not a rotation table. Real entries
  satisfy ``cos^2 + sin^2 = 1`` and ``|.| <= 1``; randn ones reach +-5 and miss
  the unit circle by ~30. That inflates the rope columns of ``ckv`` and feeds a
  wrong magnitude distribution into everything downstream.
* ``rms_norm_weight`` as N(0,1) is half negative with a median near zero. The
  real tensor (DeepSeek-V4-Flash, layer 11) is 503/512 positive and spans 190x.
  This one matters most: RMSNorm divides out the input scale, so it is
  ``rms_norm_weight`` that sets the per-channel magnitudes the FP8 block
  quantiser then has to cover.
* ``kv_state``/``score_state`` as N(0,1) are light-tailed. Measured on real
  DeepSeek-V2-Lite activations, the hidden state has **kurtosis 405** and the
  MLA KV latent **245**, against 3.0 for a Gaussian -- the "massive activation"
  channels that every KV-quantisation paper is about, and the reason UE8M0
  carries a scale per 64 elements instead of one per tensor.

The consequence is recorded in eval_config.yaml: over the whole sweep the
UE8M0 exponent only ever lands in ``[-8, -4]``, out of a representable
``[-22, 120]``. The quantiser's dynamic range is almost entirely untested, and
``clamp(min=1e-4)`` is never reached.

What is real here, and what is not
----------------------------------

The Definition's target model is public, so most of this is not modelled at all
(see tools/model_probe.py):

    rms_norm_weight   layers.11.attn.compressor.norm.weight      exact
    cos_cache/sin     DeepseekV4RotaryEmbedding(..., "compress")  exact
    score_state       h @ wgate.T + ape                           weights exact
    kv_state          h @ wkv.T                                   weights exact

Only ``h``, the hidden state feeding the compressor, is synthesised -- running
DeepSeek-V4-Flash would mean implementing its fp4 MoE (256 experts/layer), which
is a project, not a step. Instead ``h`` is lifted from *real* DeepSeek-V2-Lite
hidden states: V4 is 4096 channels and V2-Lite is 2048, so each V4 channel takes
a real V2-Lite channel's real time series, each used twice with the second copy
rolled in time to avoid duplicating a column outright. That preserves the real
per-channel marginals and the heavy tails exactly; what it does not reproduce is
V4's exact cross-channel covariance.

This matters less than it sounds, and deliberately so: the op RMSNorms before it
quantises, so the input's overall scale is divided out and the per-channel
structure that survives into ``ckv_scale`` comes from ``norm.weight`` and the
``wkv`` row norms -- both exact.

Why not simply capture one tensor and ship it
---------------------------------------------

A captured blob is one sample of one model at one moment; it cannot be
resampled and it does not say what it covers. A kernel has to be correct across
the distribution. So the generator is seeded and its provenance recorded, and
alongside the model-derived workloads it emits three *stress* workloads that the
real distribution does not reliably produce (``--stress``):

    flat      near-uniform gate -> softmax close to 1/128, maximal cancellation
    peaked    high-temperature gate -> one token dominates its window
    tiny      kv_state scaled to ~1e-7 -> drives absmax under the 1e-4 clamp,
              which is the only path through the reference that the sweep has
              never executed

Sizing
------

Values cannot affect a performance number for this op: it is a pure streaming
reduction with no data-dependent control flow or addressing, so a candidate
kernel's bandwidth is identical on real and random input. Blobs therefore buy
*correctness* coverage only, and there is no reason to pay for them where the
bytes are largest. Since ``gen_inputs`` dispatches per input name, the two can
be mixed:

* every workload gets real ``rms_norm_weight``/``cos_cache``/``sin_cache``
  (~111 MB total, dominated by the cos/sin tables at the top of the sweep);
* ``kv_state``/``score_state`` are blobbed only for ``num_compressed <= 128``
  (~110 MB); above that they stay random, because those workloads exist to
  measure bandwidth and 780 MB of blob would buy nothing.

Reproducibility
---------------

The blobs are a pure function of four things, all pinned:

    the frozen corpus          tools/corpus/hca_v1.txt, sha256 asserted
    DeepSeek-V2-Lite           a hub revision, for the activation harvest
    DeepSeek-V4-Flash          a hub revision, for wkv/wgate/ape/norm/rope
    workload_seed(uuid)        crc32, stable across processes

Two of those were broken until this was written down. ``hash(uuid)`` is salted
per process, so the "seed" changed on every invocation; and the corpus was built
live from the task's own design write-up, making the workload data a function of
a document the repository was editing. Both are fixed, and
``tasks/hca_compress_c128/blobs.sha256`` records the digest of every blob so a
rebuild can be checked rather than assumed.

Regenerating the blobs invalidates every recorded trace: the traces describe
measurements on specific input bytes. Re-run tools/run_benchmark.py after.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import model_probe as mp  # noqa: E402

REPO = mp.REPO

# Above this many compressed entries, kv_state/score_state stay random: those
# workloads measure bandwidth, and values cannot change a bandwidth number.
BLOB_KV_MAX_NUM_COMPRESSED = 128

SEED = 0x11CA

# The three stress workloads, keyed by uuid: {uuid: (mode, num_compressed)}.
#
# Keyed rather than appended-in-order because the Workload schema has nowhere to
# put a generator mode -- `axes` is Dict[str, int] and there is no free-form
# field -- so once a stress workload is written into the sweep, the jsonl alone
# cannot say it was anything other than `real`. Regenerating from that file
# previously rebuilt all three with the real distribution and no warning, which
# quietly deleted the coverage they exist for (`tiny` is the only input in the
# repository that reaches the reference's clamp(min=1e-4)). The uuid is the
# stable identity, so the mapping lives here.
STRESS_WORKLOADS = {
    "00000000-0000-4128-8001-000000000000": ("flat", 64),
    "00000000-0000-4128-8001-000000000001": ("peaked", 64),
    "00000000-0000-4128-8001-000000000002": ("tiny", 64),
}


def workload_seed(uuid: str) -> int:
    """A stable per-workload seed.

    ``hash(uuid)`` was used here and is wrong: Python salts str hashing per
    process (PYTHONHASHSEED), so every invocation produced different blobs for
    the same workload while the docstring above claimed the generator was
    seeded. ``zlib.crc32`` is specified, stable across processes, versions and
    platforms, and is used here only to spread uuids -- not as a checksum.
    """
    return SEED ^ zlib.crc32(uuid.encode())


def _lift_channels(h: torch.Tensor, out_channels: int, seed: int) -> torch.Tensor:
    """Widen a real activation matrix to ``out_channels`` channels.

    Each output channel is assigned a real input channel's real time series, so
    per-channel marginals -- and therefore the heavy tails that motivate block
    scaling -- are preserved exactly. When a channel must be reused, the copy is
    rolled along time so no two output columns are identical.
    """
    t, c = h.shape
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(c, generator=g).repeat(-(-out_channels // c))[:out_channels]
    reuse = torch.arange(out_channels) // c  # 0 for the first copy, 1 for the second, ...
    cols = []
    for j in range(out_channels):
        col = h[:, pick[j]]
        shift = int(reuse[j]) * (t // max(int(reuse.max()) + 1, 1) // 2 + 1)
        cols.append(torch.roll(col, shifts=shift) if shift else col)
    return torch.stack(cols, dim=1)


class Generator:
    """Builds ``kv_state``/``score_state`` from real V4 weights and real activations."""

    def __init__(self, layer: int = mp.DEFAULT_LAYER, device: str = "cuda"):
        self.device = device
        self.w = mp.compressor_weights(layer)
        self.cfg = mp.config()
        self.hidden_size = self.cfg["hidden_size"]
        self.head_dim = self.cfg["head_dim"]
        self.compress_rate = self.cfg["compress_ratios"][layer]

        act = mp.v2lite_activations()
        self.h_real = act["hidden"].float()
        self.profile = mp.channel_profile(act["hidden"])

        self.wkv = self.w["wkv"].to(device, torch.float32)
        self.wgate = self.w["wgate"].to(device, torch.float32)
        self.ape = self.w["ape"].to(device, torch.float32)

    def hidden_for(self, num_tokens: int, seed: int) -> torch.Tensor:
        """``[num_tokens, hidden_size]`` fp32 of lifted real activations."""
        h = _lift_channels(self.h_real, self.hidden_size, seed)
        if h.shape[0] < num_tokens:
            h = h.repeat(-(-num_tokens // h.shape[0]), 1)
        g = torch.Generator().manual_seed(seed ^ 0x5EED)
        start = int(torch.randint(0, max(h.shape[0] - num_tokens, 1), (1,), generator=g))
        return h[start : start + num_tokens].to(self.device)

    def states(
        self, num_compressed: int, seed: int, mode: str = "real"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(kv_state, score_state)``, both ``[n, compress_rate, head_dim]`` bf16.

        ``score_state`` carries the absolute position embedding already: vLLM
        adds ``ape`` in the kernel *before* this one, so the Definition's
        boundary puts it on this side. ``ape`` is indexed by position within the
        window, which is exactly its ``[compress_rate, head_dim]`` shape.
        """
        n, m = num_compressed, self.compress_rate
        h = self.hidden_for(n * m, seed)

        kv = (h @ self.wkv.T).reshape(n, m, self.head_dim)
        gate = (h @ self.wgate.T).reshape(n, m, self.head_dim) + self.ape

        if mode == "flat":
            # Near-uniform softmax: every token in the window contributes ~1/128,
            # so the weighted sum is a mean over 128 heavy-tailed vectors and
            # cancellation is maximal.
            gate = gate * 0.02
        elif mode == "peaked":
            # One token dominates its window; the reduction degenerates to a
            # near-copy and the block absmaxes follow a single token's outliers.
            gate = gate * 12.0
        elif mode == "tiny":
            # Push every block absmax under the reference's clamp(min=1e-4).
            kv = kv * 1e-7
        elif mode != "real":
            raise ValueError(f"unknown mode {mode!r}")

        return kv.to(torch.bfloat16).cpu(), gate.to(torch.bfloat16).cpu()


def exponent_stats(
    kv: torch.Tensor, score: torch.Tensor, norm_w: torch.Tensor, device: str = "cuda"
) -> Dict[str, float]:
    """UE8M0 exponents the reference would produce for these inputs.

    Mirrors the Definition's reference so the report says what the *task* will
    see, not what an approximation of it would.
    """
    kv = kv.to(device, torch.float32)
    score = score.to(device, torch.float32)
    w = norm_w.to(device, torch.float32)

    weight = torch.softmax(score, dim=1)
    compressed = (kv * weight).sum(dim=1)
    normed = compressed * torch.rsqrt(compressed.square().mean(-1, keepdim=True) + 1e-6) * w
    nope = normed[:, :448].to(torch.bfloat16).float()
    absmax = nope.reshape(nope.shape[0], 7, 64).abs().amax(-1).clamp(min=1e-4)
    e = torch.ceil(torch.log2(absmax / 448.0)).clamp(-127, 127)
    return {
        "e_min": float(e.min()),
        "e_max": float(e.max()),
        "e_unique": int(e.unique().numel()),
        "clamped_frac": float((absmax <= 1e-4).float().mean()),
        "softmax_max": float(weight.amax(dim=1).mean()),
    }


def blob_rel_path(def_name: str, op_type: str, uuid: str) -> str:
    return f"blob/workloads/{op_type}/{def_name}/{def_name}_{uuid}.safetensors"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=pathlib.Path, nargs="?",
                    default=REPO / "tasks" / "hca_compress_c128")
    ap.add_argument("--root", type=pathlib.Path, default=None,
                    help="TraceSet root to write blobs into (default data/trace_sets/<task>)")
    ap.add_argument("--layer", type=int, default=mp.DEFAULT_LAYER)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv-max", type=int, default=BLOB_KV_MAX_NUM_COMPRESSED,
                    help="blob kv_state/score_state up to this num_compressed")
    ap.add_argument("--stress", action="store_true",
                    help="also append the flat/peaked/tiny stress workloads")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="where to write the rewritten sweep (default: <task>/<def>.real.jsonl). "
                         "The authored random sweep is never overwritten in place; promote the "
                         "result by renaming it once you have reviewed the diff.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from safetensors.torch import save_file

    task_dir = args.task_dir.resolve()
    root = (args.root or REPO / "data" / "trace_sets" / task_dir.name).resolve()

    defs = [p for p in sorted(task_dir.glob("*.json")) if "spec" not in json.loads(p.read_text())]
    if len(defs) != 1:
        raise SystemExit(f"expected exactly one Definition in {task_dir}, found {len(defs)}")
    definition = json.loads(defs[0].read_text())
    def_name, op_type = definition["name"], definition["op_type"]

    wl_path = task_dir / f"{def_name}.jsonl"
    traces = [json.loads(l) for l in wl_path.read_text().splitlines() if l.strip()]
    print(f"definition {def_name}  ({op_type})\n{len(traces)} workloads in {wl_path.name}\n")

    gen = Generator(layer=args.layer, device=args.device)
    print(f"real V4 weights from layer {args.layer} of {mp.V4_MODEL_ID}")
    print(f"real activations: {gen.profile}\n")
    norm_w = gen.w["norm"]

    if args.stress:
        have = {tr["workload"]["uuid"] for tr in traces}
        for uuid, (mode, n) in STRESS_WORKLOADS.items():
            if uuid in have:
                continue  # already in the sweep; its mode comes from STRESS_WORKLOADS
            traces.append({
                "definition": def_name, "solution": None,
                "workload": {"uuid": uuid,
                             "axes": {"num_compressed": n, "max_position": n * 128},
                             "inputs": {}, },
                "evaluation": None,
            })

    blob_dir = root / "blob" / "workloads" / op_type / def_name
    if not args.dry_run:
        blob_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    rows: List[str] = []
    print(f"{'n':>5} {'mode':7} {'inputs blobbed':26} {'e_min':>6} {'e_max':>6} "
          f"{'uniq':>5} {'clamp%':>7} {'smax':>6} {'MB':>7}")
    for tr in traces:
        wl = tr["workload"]
        n = wl["axes"]["num_compressed"]
        maxpos = wl["axes"]["max_position"]
        uuid = wl["uuid"]
        mode = STRESS_WORKLOADS.get(uuid, (None,))[0] or "real"

        tensors: Dict[str, torch.Tensor] = {}
        cos, sin = mp.rope_tables(maxpos)
        tensors["rms_norm_weight"] = norm_w
        tensors["cos_cache"] = cos
        tensors["sin_cache"] = sin

        blob_kv = mode != "real" or n <= args.kv_max
        if blob_kv:
            kv, score = gen.states(n, seed=workload_seed(uuid), mode=mode)
            tensors["kv_state"] = kv
            tensors["score_state"] = score
            stats = exponent_stats(kv, score, norm_w, args.device)
        else:
            stats = {"e_min": float("nan"), "e_max": float("nan"), "e_unique": 0,
                     "clamped_frac": float("nan"), "softmax_max": float("nan")}

        rel = blob_rel_path(def_name, op_type, uuid)
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        total_bytes += nbytes

        if not args.dry_run:
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(root / rel))

        wl["inputs"] = {
            name: ({"type": "safetensors", "path": rel, "tensor_key": name}
                   if name in tensors else {"type": "random"})
            for name in definition["inputs"]
        }
        rows.append(json.dumps(tr, separators=(",", ":")))
        print(f"{n:>5} {mode:7} {','.join(sorted(tensors)):26.26} "
              f"{stats['e_min']:>6.0f} {stats['e_max']:>6.0f} {stats['e_unique']:>5} "
              f"{100 * stats['clamped_frac']:>7.2f} {stats['softmax_max']:>6.3f} "
              f"{nbytes / 2**20:>7.1f}")

    print(f"\ntotal blob bytes: {total_bytes / 2**20:.1f} MB")
    if args.dry_run:
        print("(dry run: nothing written)")
        return 0

    out_path = args.out or (task_dir / f"{def_name}.real.jsonl")
    out_path.write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} workloads -> {out_path}")
    print(f"wrote blobs             -> {blob_dir}")
    print(f"\nthe authored sweep {wl_path.name} is untouched; review the diff, then promote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
