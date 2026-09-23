A hand-written Triton kernel for the chunked KDA prefill, submitted as the task
seed: the reference point a candidate has to beat, chosen to be beatable. It is
not pipelined, not autotuned, holds no tile in SRAM across the two kernels, and
leaves the single largest cost in the op untouched on purpose. What it does do
is never materialise the decay mask, which is the one thing that separates a
kernel from the compiled baseline here.

## Measured: 13.0 TFLOP/s, 12.7% of peak

Full sweep, 16/16 PASSED (`tools/run_benchmark.py`, then
`tools/report_traces.py`; H200, triton 3.6.0, composite ceiling 102.7 TFLOP/s
measured at the op's own tile shapes -- see `flops_model.py`).

    seq_len (B=1)      128     512    2048    8192   16384
    GFLOP             0.58    2.32    9.29   37.17   74.35
    latency (ms)     0.149   0.191   0.730   2.874   5.706
    TFLOP/s           3.91   12.18   12.73   12.94   13.03
    % of peak         3.8%   11.9%   12.4%   12.6%   12.7%
    vs torch.compile  4.2x   13.2x   13.4x   14.1x   14.8x
    vs reference     44.9x   41.2x   25.5x   25.3x   25.9x

Over the whole 16-workload sweep: 28 ms against the compiled baseline's 385 ms
and the reference's 692 ms.

It saturates by `seq_len=512` and then sits flat: at 12.7% of peak the kernel is
not close to any hardware limit, so there is nothing for more work per program to
amortise against. 87% of the achievable rate is on the table, and the next
section says exactly where it went.

## Where the time goes: 1.9% of the FLOPs is 72% of the latency

The UT transform -- `(I - A)^-1` for strictly-lower `A` -- is done by forward
substitution, 63 dependent steps per chunk, term for term with the reference.
`flops_model.py` counts that term at **1.9% of the op's FLOPs**. Measured at
B=1 T=16384 by truncating the loop (`tools/kda_seed_split_probe.py`; the
truncated variants compute the wrong answer and are timing probes only):

    UT steps   1     0.970 ms of _prepare    19.1%
    UT steps   8     1.530 ms                30.1%
    UT steps  32     2.987 ms                58.8%
    UT steps  64     5.087 ms               100.0%

81% of `_prepare`, and `_prepare` is 89% of the kernel: **72% of total latency
for 1.9% of the arithmetic.** The serial loop is also why `_prepare` cannot hide
its own memory traffic -- there is nothing to overlap with 63 rounds of
`tl.where` reductions.

The obvious trade is the doubling formulation, `prod_k (I + A^(2^k))`: 5 tile
matmuls instead of 63 serial steps, ~5.2 MFLOP per chunk against the 0.17 MFLOP
substitution needs. That is 30x the arithmetic for this term (+57% on the op's
total FLOPs) with all of it on tensor cores and none of it serial. This seed
deliberately does not take it -- a correctness anchor should be a transcription
of the reference, and a seed that already made the interesting optimisation
would not be a seed. A candidate that takes it should expect a large win.

## The trap a candidate will hit: fp32 overflow in the factored decay

Refusing the rank-3 mask means factoring `exp(g_i - g_j) = exp(g_i)exp(-g_j)` so
the `[C, C]` build collapses to one matmul. Exact in real arithmetic, lethal in
fp32: `g` is a within-chunk cumsum of negative numbers, so the column factor
grows as `e^|g|`. Measured over the real corpus
(`tools/kda_decay_range_probe.py`):

    worst within-chunk cumsum                     -111.46
    (chunk, channel) pairs past fp32's e^88.7     14809 / 5038080  = 0.2939%
    workloads in which this occurs                16 / 16

Not a long-sequence corner case -- it fires in every workload, including
`seq_len=100`. The naive factored kernel produces `0 * inf = nan`.

The fix here is to centre the exponent per channel at the midpoint of the
chunk's cumsum range, `c_d = (g_0d + g_{C-1,d}) / 2`. Both factors are then
bounded by half the range (`e^55.7 = 1.5e24`, comfortably finite) and their
product is unchanged. Centring costs one subtract on a tile already being
exponentiated.

The strictly-upper half of the product still overflows -- there `g_i > g_j` and
the true value legitimately exceeds 1 -- and is discarded with `tl.where`, never
multiplied by a zero mask. `where` selects; `inf * 0.0` is `nan`, and the nan
would survive into the output.

## Numerics: bf16 matmuls, worst matched_ratio 0.99846649

Every `tl.dot` runs in bf16 with fp32 accumulate; the state carried across chunks
stays fp32 in registers. Worst observation over all 16 workloads is
`matched_ratio = 0.99846649` at `batch_size=2 seq_len=8192`, inside the task's
0.99 floor. The state error is flat in `T` (0.9985-0.9996 across the sweep, not
monotone) because the forget gate damps state error rather than compounding it --
each chunk multiplies the carried state by `exp(g_last) < 1`, so an error
injected at chunk `i` is attenuated, not accumulated. This is the property that
makes a long-sequence tolerance box possible at all on this op.

## Two structural choices

**The `(batch, head, v-block)` split in `_scan`** is for occupancy, not elegance:
at `batch_size=1` a scan parallelised only over `(batch, head)` fills 32 of the
H200's 132 SMs. `BV = 32` gives 128 programs. It costs a 4x reread of the
per-chunk tiles the scan does not split -- about 1.4 GB over the largest
workload, ~0.3 ms at measured bandwidth -- which is the right trade at these
batch sizes and the wrong one at large `B`. A tuned kernel would pick `BV` from
the axes.

**Padding with zeros needs no tail chunk.** `seq_len` is not a multiple of 64 in
a third of the sweep. Loading out-of-range tokens as zeros reproduces the
reference's `F.pad` semantics everywhere it matters: zero `beta` kills the rank-1
update, zero `g` makes the pad tokens' decay exactly 1, and a zero `v_new`
contributes nothing to the state. Masked loads and masked stores are the whole
tail handling.

## What it does not do

No software pipelining, no autotuning (`num_warps=4, num_stages=2` everywhere,
unmeasured), no fusion of `_prepare` into `_scan` -- the six intermediate tensors
go to HBM and come back. Peak allocation is 1.35 GiB at T=16384 against the
compiled baseline's 18.54 GiB, but a fused kernel would need far less than
either.
