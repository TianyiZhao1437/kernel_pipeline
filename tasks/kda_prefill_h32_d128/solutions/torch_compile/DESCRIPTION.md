The incumbent, not a kernel. This is the obvious PyTorch expression of the
Definition handed to `torch.compile` with no tuning -- no `mode=`, no
`options=`, no `dynamic=` hint -- because that is what a practitioner reaches
for first on an op with no library implementation, and because the question a
baseline should answer is "how much of this does the compiler already get?".

On this op the answer is: not much, and for one structural reason.

## Measured: 1.0 TFLOP/s, 1.0% of peak

Full sweep, 16/16 PASSED, `matched_ratio = 1.00000000` on every workload
(`tools/run_benchmark.py`, then `tools/report_traces.py`; H200,
torch 2.11.0+cu128, composite ceiling 102.7 TFLOP/s measured at the op's own
tile shapes, not at a large square GEMM -- see `flops_model.py`).

    seq_len (B=1)      128    1024    4096    8192   16384
    GFLOP             0.58    4.65   18.59   37.17   74.35
    latency (ms)      0.63    4.97   19.78   40.43   84.33
    TFLOP/s           0.93    0.93    0.94    0.92    0.88
    % of peak         0.9%    0.9%    0.9%    0.9%    0.9%
    vs reference    10.65x   2.04x   1.85x   1.80x   1.75x
    vs triton seed   0.24x   0.08x   0.07x   0.07x   0.07x

Over the whole 16-workload sweep: 385 ms against the reference's 692 ms (1.80x)
and the Triton seed's 28 ms.

The rate is flat at ~0.9 TFLOP/s across a 128-fold range of problem size, and
the speedup over the reference decays monotonically from 10.65x to 1.75x as the
sequence grows -- i.e. the entire advantage at small sizes is eliminated
dispatch overhead, and at large sizes the compiled version is limited by the
same resource the eager reference is. That flatness is the finding: this
solution is not compute-bound at any point in the sweep, so its position on a
TFLOP/s ranking is a statement about memory traffic, not about arithmetic.

## What the compiler gets, and the one thing it cannot

Inductor fuses the pointwise chain -- the l2norm, the cumsum's consumers, the
`exp` calls, the masked fills -- and it does cut the reference's memory
footprint substantially. Peak allocation, measured:

    seq_len (B=1)     2048    8192   16384
    this solution   2.43 G  9.29 G  18.54 G
    triton seed     0.20 G  0.69 G   1.35 G
    ratio            12.2x   13.4x    13.7x

(The Definition's reference is worse still, at 50.16 GiB for the largest
workload.) But 18.5 GiB is what is left after the compiler has done its best,
and it is still 13.7x what a real kernel needs, because the decay mask
`exp(g_i - g_j)` is written as a `[B, H, NC, C, C, D]` tensor in the source and
Inductor cannot make it not exist. It can fuse the producer into the consumer;
it cannot restructure the algorithm so the rank-3 intermediate is never formed.
That restructuring -- keeping the chunk in registers and factoring the decay --
is exactly the work this task is asking for, and it is why the gap to a
hand-written kernel here is 12-15x rather than the ~1.2x a compiler usually
leaves.

## Two deliberate deviations from the reference

**The UT transform is a triangular solve.** The reference builds the WY matrix
by forward substitution: 63 dependent in-place iterations per chunk. Compiled,
that unrolls into 63 stages multiplied by 256 chunks at the largest workload.
The substitution computes exactly `T = (I - A)^-1` for strictly-lower `A`, which
`torch.linalg.solve_triangular` does in one call. Verified rather than assumed:
the residual `||(I - A)T - I||` is 4e-8 relative to `|T|` on random matrices at
the scale this op produces, and the solution scores `matched_ratio = 1.000000`
against the reference on all 16 workloads.

**The chunk loop stays in eager.** It is sequential and its trip count varies
with `seq_len`, so compiling it would either unroll to a graph proportional to
the sequence or recompile per workload. Everything before it is
shape-polymorphic and compiles as one region. That split is also the honest one
to benchmark: it is what the incumbent actually looks like when someone writes
it.

## What this baseline does not catch

Unlike `hca_compress_c128`, where `torch.compile` silently deleted a
precision-critical cast and the tolerance design caught it, nothing here is
elided: the op has no double-rounding step to lose. The tolerance box in
`eval_config.yaml` (rtol=2e-2, atol=1e-3, matched>=0.99) is doing its separating
work against semantic mutations and bf16 kernels, not against this solution,
which sits at the ceiling on every workload.
