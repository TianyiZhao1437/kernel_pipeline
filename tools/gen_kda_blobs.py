#!/usr/bin/env python3
"""Materialise real input tensors for the kda_prefill sweep.

For most tasks a blobbed corpus buys input *realism*. For this one it is closer
to a correctness prerequisite. ``{"type": "random"}`` is exactly ``torch.randn``
-- ``RandomInput`` carries no distribution, no scale and no seed -- and two of
this Definition's six inputs cannot survive that:

* ``g`` is a log decay and must be negative. Gaussian ``g`` makes half the
  channels positive, ``exp(cumsum(g))`` grows without bound over a 16k-token
  sweep, and the recurrence diverges. The measured op would not be KDA.
* ``beta`` is a sigmoid and must lie in (0, 1). Outside that the delta rule
  overshoots instead of interpolating.

``query``/``key``/``value`` are a weaker case and the argument for blobbing them
is not the one it might look like. They would not diverge under randn, and
tasks/hca_compress_c128 leaves its bulk inputs random above ``--kv-max`` for
exactly that reason. I expected real keys to be strongly mutually aligned --
they come out of a SiLU -- and to therefore stress the WY representation in a
way near-orthogonal Gaussian keys would not. Measured, that is mostly false:
mean adjacent-token cosine after l2norm is **+0.053 for real keys against
+0.000 +/- 0.089 for randn**, a real positive bias but well inside the Gaussian
spread. The distributions differ in shape (real keys are SiLU-floored at
-0.2785 and right-skewed, measured skewness 1.62) without differing much in the
quantity the delta rule actually sees.

So they are blobbed for a duller reason: ``g`` and ``beta`` must be blobbed at
every workload regardless, one safetensors file per workload holds all six
inputs, and 3.1 GB of gitignored scratch against 419 GB free is not worth a
second code path, a split manifest, and a footnote in every tolerance
derivation. Uniform is cheaper than clever here.

The blob tree is gitignored; ``blobs.sha256`` is what gets committed.

``initial_state``: part of the sweep carries zeros (a fresh prefill) and part a
genuine state produced by running the reference over a 512-token prefix (a
chunked-prefill continuation). Both are real cases, and having both also closes
an anti-hack gap -- an all-zero state at every workload would let a kernel skip
the ``attn_inter`` and ``v_prime`` products on the first chunk, or detect the
zeros and skip them wholesale, and nothing in the sweep would notice.

On the decay rates  [measured]
------------------------------

``g`` is a per-channel gate, initialised with the Mamba recipe that
``KimiLinearPreTrainedModel._init_weights`` implements (``A ~ U(1, 16)``,
``dt ~ logU(1e-3, 1e-1)``, ``dt_bias = softplus^-1(dt)``), so the per-channel
decay spans three orders of magnitude and any scalar summary of it misleads.
The mean retention over a chunk is ~1e-6, which read alone says the state is
dead before the second chunk and the corpus is degenerate. It is not. Measured
per channel: 12% retain more than half their magnitude across a 64-token chunk,
the median memory length ``1/|g|`` is 15 tokens, p90 is 107 and the slowest
channel runs to 738.

The consequence that matters is the anti-hack one, and it is measured directly
rather than inferred from the gate: running the reference normally against
running it chunk-by-chunk with the state forced to zero -- i.e. exactly what a
kernel that ignores the recurrence computes -- differs by **53% of the output
norm** (``||delta||/||out|| = 0.53`` at T=512, 0.57 at T=2048). That is O(1). No
tolerance this task could reasonably set would admit such a kernel.

The honest limitation is at the other end: because memory lengths are drawn
from an untrained gate, by T=16384 almost no channel retains anything over the
full sequence, so the long workloads measure an op that is *effectively* local
even though every kernel must still do the full quadratic-in-chunk work to
discover that. A trained Kimi Linear checkpoint would presumably have learned
longer memory. This does not affect the FLOP count, the ranking, or the
cross-chunk anti-hack result above, but it does mean the corpus understates how
much genuine long-range mixing a production trace would contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kda_inputs  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent

# Workloads that carry a warmed-up recurrent state rather than zeros. Chosen to
# cover both ends of the sweep and both alignment cases rather than to be a
# tidy half: a long aligned sequence, a long batched one, two unaligned, and a
# short one.
WARM_STATE = {(1, 256), (1, 1024), (1, 8192), (1, 1000), (1, 5000), (2, 2048), (2, 8192)}
WARM_PREFIX = 512


def blob_rel_path(def_name: str, op_type: str, uuid: str) -> str:
    return f"blob/workloads/{op_type}/{def_name}/{def_name}_{uuid}.safetensors"


def rel(path: pathlib.Path) -> pathlib.Path:
    """Display path relative to the repo, tolerating a path given relative to cwd."""
    try:
        return path.resolve().relative_to(REPO)
    except ValueError:
        return path


def stable_seed(base: int, uuid: str) -> int:
    """Per-workload seed that is reproducible across runs.

    ``hash()`` on a str is salted per interpreter process, so using it here
    would make the corpus unreproducible -- rerunning would emit different
    tensors under the same ``--seed`` and silently invalidate blobs.sha256.
    """
    digest = hashlib.sha256(uuid.encode()).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def input_stats(inputs: dict) -> dict:
    """The numbers that justify blobbing, so the run prints its own evidence."""
    import torch

    g = inputs["g"].float()
    beta = inputs["beta"].float()
    key = inputs["key"].float()
    state = inputs["initial_state"].float()

    centred = key - key.mean()
    skew = float((centred**3).mean() / key.std() ** 3)

    # Mean pairwise cosine between l2-normalised keys of adjacent tokens.
    # Printed to be checked against randn's 0.000 +/- 0.089, not because it is
    # large -- see the module docstring; it is +0.05.
    flat = key.reshape(-1, key.shape[-1])[:4096]
    unit = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cosine = float((unit[:-1] * unit[1:]).sum(-1).mean())

    # exp(mean(g)*64) would be ~1e-6 and would read as "the state is dead".
    # g is per-channel and log-uniform-ish, so report the spread instead: the
    # share of channels still carrying half their magnitude across a chunk,
    # and the median memory length in tokens.
    rate = g.mean(dim=1)                                  # [B, H, D]
    retain = float((torch.exp(rate * 64) > 0.5).float().mean())
    tau = (-1.0 / rate.clamp_max(-1e-12)).flatten()
    memory = float(torch.quantile(tau, 0.5))

    return {
        "g_min": float(g.min()),
        "g_max": float(g.max()),
        "g_all_negative": bool((g < 0).all()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "beta_in_unit": bool(((beta > 0) & (beta < 1)).all()),
        "key_skew": skew,
        "key_cosine": cosine,
        "state_absmax": float(state.abs().max()),
        "retain_frac": retain,
        "memory_tokens": memory,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=pathlib.Path, nargs="?",
                    default=REPO / "tasks" / "kda_prefill_h32_d128")
    ap.add_argument("--root", type=pathlib.Path, default=None,
                    help="TraceSet root to write blobs into (default data/trace_sets/<task>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="where to write the rewritten sweep (default <task>/<def>.real.jsonl). "
                         "The authored random sweep is never overwritten in place.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    task_dir = args.task_dir.resolve()
    root = (args.root or REPO / "data" / "trace_sets" / task_dir.name).resolve()

    defs = [p for p in sorted(task_dir.glob("*.json")) if not p.name.endswith(".solution.json")]
    if len(defs) != 1:
        raise SystemExit(f"expected exactly one Definition in {task_dir}, found {len(defs)}")
    definition = json.loads(defs[0].read_text())
    def_name = definition["name"]
    op_type = definition["op_type"]

    # The reference is needed to produce the warm states, and it is the same
    # code the benchmark will grade against.
    namespace: dict = {}
    exec(compile(definition["reference"], "<reference>", "exec"), namespace)
    reference = namespace["run"]

    wl_path = task_dir / f"{def_name}.jsonl"
    traces = [json.loads(line) for line in wl_path.read_text().splitlines() if line.strip()]

    blob_dir = root / pathlib.Path(blob_rel_path(def_name, op_type, "x")).parent
    if not args.dry_run:
        blob_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'B':>3} {'T':>6} {'state':>6} {'g_min':>8} {'g_max':>9} {'beta':>13} "
          f"{'k_skew':>7} {'k_cos':>7} {'retain':>7} {'mem_tok':>8} {'MB':>7}")
    rows = []
    digests = []
    total_bytes = 0
    bad = 0

    for trace in traces:
        workload = trace["workload"]
        axes = workload["axes"]
        batch_size, seq_len = axes["batch_size"], axes["seq_len"]
        uuid = workload["uuid"]
        warm = (batch_size, seq_len) in WARM_STATE

        inputs = kda_inputs.make_inputs(
            batch_size,
            seq_len,
            seed=stable_seed(args.seed, uuid),
            device=args.device,
            warm_state=warm,
            warm_len=WARM_PREFIX,
            reference=reference if warm else None,
        )
        stats = input_stats(inputs)

        # The two invariants this whole file exists to establish. A workload
        # that violates one is not a bad sample, it is the wrong operator, so
        # fail rather than write it.
        if not stats["g_all_negative"] or not stats["beta_in_unit"]:
            bad += 1
            print(f"  !! B={batch_size} T={seq_len}: g_all_negative="
                  f"{stats['g_all_negative']} beta_in_unit={stats['beta_in_unit']}")
            continue

        tensors = {k: v.cpu().contiguous() for k, v in inputs.items()}
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        total_bytes += nbytes

        rel = blob_rel_path(def_name, op_type, uuid)
        if not args.dry_run:
            save_file(tensors, str(root / rel))
            digests.append((rel, hashlib.sha256((root / rel).read_bytes()).hexdigest()))

        workload["inputs"] = {
            name: {"type": "safetensors", "path": rel, "tensor_key": name}
            for name in definition["inputs"]
        }
        rows.append(json.dumps(trace, separators=(",", ":")))

        print(f"{batch_size:>3} {seq_len:>6} {'warm' if warm else 'zeros':>6} "
              f"{stats['g_min']:>8.3f} {stats['g_max']:>9.2e} "
              f"{stats['beta_min']:.3f}-{stats['beta_max']:.3f}  "
              f"{stats['key_skew']:>7.2f} {stats['key_cosine']:>7.3f} "
              f"{100*stats['retain_frac']:>6.1f}% {stats['memory_tokens']:>8.0f} "
              f"{nbytes / 2**20:>7.1f}")
        del inputs, tensors
        torch.cuda.empty_cache()

    print(f"\ntotal blob bytes: {total_bytes / 2**30:.2f} GiB over {len(rows)} workloads")
    if bad:
        print(f"{bad} workload(s) rejected: see above")
        return 1
    if args.dry_run:
        print("(dry run: nothing written)")
        return 0

    out_path = args.out or (task_dir / f"{def_name}.real.jsonl")
    out_path.write_text("\n".join(rows) + "\n")

    sha_path = task_dir / "blobs.sha256"
    sha_path.write_text("".join(f"{d}  {p}\n" for p, d in digests))

    print(f"wrote {len(rows)} workloads -> {rel(out_path)}")
    print(f"wrote blobs             -> {rel(blob_dir)}")
    print(f"wrote digests           -> {rel(sha_path)}")
    if rel(out_path) != rel(wl_path):
        print(f"\nthe authored sweep {wl_path.name} is untouched; review the diff, then promote "
              f"by renaming {rel(out_path).name} over it -- stage_trace_set.py refuses to "
              f"guess between the two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
