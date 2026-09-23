"""Is the factored intra-chunk decay safe in fp32 on this corpus?

The reference builds `decay_mask[i,j,d] = exp(g[i,d] - g[j,d])` as a rank-3
tensor, which is why it peaks at 50 GiB. A kernel cannot do that; it has to
factor, `exp(g_i) * exp(-g_j)`, so the [C,C] build becomes one matmul:

    A = -(k_beta * exp(g)) @ (k * exp(-g))^T

The factorisation is exact in real arithmetic and dangerous in fp32: `g` is a
within-chunk cumsum of negative numbers, so `exp(-g_j)` grows as e^|g|. If the
cumsum reaches -89 the column factor overflows fp32 (e^88.7 = 3.4e38) while the
row factor has already flushed to zero, and 0 * inf = nan.

The measured per-token `g` bottoms out at -3.837 (verify_task C3). Worst case
over 64 tokens is -245, which would be fatal. The question is whether the worst
case happens: g is per-channel and most channels are near zero.

Report the distribution, not a verdict -- and report it per channel, because
one channel out of 128*32*256 is enough to produce a nan.
"""

import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

CHUNK = 64
FP32_EXP_MAX = 88.72


def main():
    from derive_kda_tolerances import load_corpus

    print(f"{'B':>2} {'T':>6}  {'min cumsum':>11} {'p99.99':>9} {'p99':>8} "
          f"{'median':>8}  {'channels over':>13}")
    worst = 0.0
    total_over = 0
    total_ch = 0
    for axes, inputs in load_corpus():
        g = inputs["g"].float()                      # [B, T, H, D]
        b, t, h, d = g.shape
        pad = (CHUNK - t % CHUNK) % CHUNK
        if pad:
            g = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad))
        # within-chunk cumsum, exactly what the kernel would hold
        gc = g.reshape(b, -1, CHUNK, h, d).cumsum(dim=2)
        floor = gc.amin(dim=2)                       # [B, NC, H, D] most negative per channel
        mn = float(floor.min())
        worst = min(worst, mn)
        over = int((floor < -FP32_EXP_MAX).sum())
        total_over += over
        total_ch += floor.numel()
        q = torch.quantile(floor.flatten().float(), torch.tensor([1e-4, 0.01, 0.5], device=g.device))
        print(f"{b:>2} {t:>6}  {mn:>11.2f} {float(q[0]):>9.2f} {float(q[1]):>8.2f} "
              f"{float(q[2]):>8.3f}  {over:>13d}")

    print(f"\nworst within-chunk cumsum over the corpus: {worst:.2f}")
    print(f"channels whose column factor would overflow fp32: {total_over}/{total_ch} "
          f"({100 * total_over / total_ch:.4f}%)")
    print(f"exp(-{worst:.2f}) = {'inf' if -worst > FP32_EXP_MAX else f'{torch.tensor(-worst).exp().item():.3e}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
