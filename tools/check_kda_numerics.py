#!/usr/bin/env python3
"""Cross-check the KDA reference against an independently-derived one.

The Definition's ``reference`` is the specification every solution is graded
against, so "it was ported carefully" is not good enough -- a transposed index or
a dropped decay term produces a reference that is self-consistent, runs fine, and
silently defines the wrong operator. Three checks, each falsifiable:

1. **Port fidelity.** Our reference vs ``chunk_kimi_delta_attention`` from
   transformers, the function it was ported from. Catches transcription errors.
   Should agree to fp32 round-off.

2. **Algorithmic correctness.** Our reference vs
   ``recurrent_kimi_delta_attention``, the O(T) token-at-a-time loop. This is the
   check that matters: the recurrent form *is* the recurrence in the paper, with
   no chunking, no WY representation and no UT transform, so agreement means the
   chunked rearrangement is algebraically right rather than merely reproducible.
   Two implementations sharing a bug here would have to share it through
   completely different code.

3. **State semantics.** run(T) against run(first half) chained into
   run(second half). This is what makes ``initial_state`` and ``final_state``
   meaningful: if the carried state did not mean what the Definition says it
   means, splitting a sequence would change the answer.

Check 2 is also the reason the sequence lengths here are small. The recurrent
form is a Python loop over tokens; at T=4096 it is minutes, and it buys nothing
over T=256 -- a chunking bug shows up as soon as there is more than one chunk.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kda_inputs  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
DEF_PATH = REPO / "tasks" / "kda_prefill_h32_d128" / "kda_prefill_h32_d128.json"


def load_reference(def_path: pathlib.Path):
    """exec the Definition's reference exactly as the runner would."""
    definition = json.loads(def_path.read_text())
    namespace: dict = {}
    exec(compile(definition["reference"], "<reference>", "exec"), namespace)
    return namespace["run"]


def compare(name: str, got, want, *, tol: float) -> bool:
    got = got.float()
    want = want.float()
    diff = (got - want).abs()
    scale = want.abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / scale).max().item()
    ok = max_rel <= tol or max_abs <= tol
    print(
        f"    {name:<16} max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  "
        f"{'ok' if ok else 'FAIL'}"
    )
    return ok


def check_port_fidelity(run, inputs, tol) -> bool:
    """Our reference vs the transformers chunk function it was ported from."""
    from transformers.models.kimi_linear.modeling_kimi_linear import (
        chunk_kimi_delta_attention,
    )

    out, state = run(**inputs)
    up_out, up_state = chunk_kimi_delta_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["g"],
        inputs["beta"],
        chunk_size=64,
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ok = compare("core_attn_out", out, up_out, tol=tol)
    ok &= compare("final_state", state, up_state, tol=tol)
    return ok


def check_recurrent(run, inputs, tol) -> bool:
    """Our reference vs the token-at-a-time recurrence. The real check."""
    from transformers.models.kimi_linear.modeling_kimi_linear import (
        recurrent_kimi_delta_attention,
    )

    out, state = run(**inputs)
    rec_out, rec_state = recurrent_kimi_delta_attention(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["g"],
        inputs["beta"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ok = compare("core_attn_out", out, rec_out, tol=tol)
    ok &= compare("final_state", state, rec_state, tol=tol)
    return ok


def check_state_chaining(run, inputs, tol) -> bool:
    """run(whole) == run(tail, state=run(head).state)."""
    seq_len = inputs["query"].shape[1]
    split = seq_len // 2
    whole_out, whole_state = run(**inputs)

    def slab(lo, hi, state):
        return {
            "query": inputs["query"][:, lo:hi],
            "key": inputs["key"][:, lo:hi],
            "value": inputs["value"][:, lo:hi],
            "g": inputs["g"][:, lo:hi],
            "beta": inputs["beta"][:, lo:hi],
            "initial_state": state,
        }

    head_out, head_state = run(**slab(0, split, inputs["initial_state"]))
    tail_out, tail_state = run(**slab(split, seq_len, head_state))

    chained = torch.cat([head_out, tail_out], dim=1)
    ok = compare("core_attn_out", chained, whole_out, tol=tol)
    ok &= compare("final_state", tail_state, whole_state, tol=tol)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--definition", type=pathlib.Path, default=DEF_PATH)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="kept small: check 2 is an O(T) Python loop and 4 chunks find what 64 would",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tol",
        type=float,
        default=2e-2,
        help=(
            "relative tolerance. Loose on purpose: q/k/v arrive in bf16 (~3 decimal "
            "digits) and the chunked and recurrent forms sum the same terms in "
            "different orders, so exact agreement is not on offer. A wrong operator "
            "misses by O(1), not by 1e-2."
        ),
    )
    ap.add_argument(
        "--unaligned",
        action="store_true",
        help="use seq_len+7 as well, exercising the reference's padding path",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU", file=sys.stderr)
        return 2

    run = load_reference(args.definition)
    print(f"reference loaded from {args.definition.relative_to(REPO)}")

    seq_lens = [args.seq_len] + ([args.seq_len + 7] if args.unaligned else [])
    failures = 0

    for seq_len in seq_lens:
        for warm in (False, True):
            label = f"B={args.batch_size} T={seq_len} initial_state={'warm' if warm else 'zeros'}"
            print(f"\n=== {label} ===")
            inputs = kda_inputs.make_inputs(
                args.batch_size,
                seq_len,
                seed=args.seed,
                warm_state=warm,
                warm_len=128,
                reference=run if warm else None,
            )
            print(kda_inputs.describe(inputs))

            print("  [1] port fidelity vs transformers chunk_kimi_delta_attention")
            if not check_port_fidelity(run, inputs, args.tol):
                failures += 1
            print("  [2] algorithmic correctness vs recurrent_kimi_delta_attention")
            if not check_recurrent(run, inputs, args.tol):
                failures += 1
            print("  [3] state chaining: run(whole) == run(head) -> run(tail)")
            if not check_state_chaining(run, inputs, args.tol):
                failures += 1

    print()
    if failures:
        print(f"{failures} comparison(s) FAILED -- the reference is not trustworthy yet.")
        return 1
    print("all comparisons passed; the reference agrees with an independent recurrence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
