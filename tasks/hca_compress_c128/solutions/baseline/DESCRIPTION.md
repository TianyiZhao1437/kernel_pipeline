The incumbent, not a kernel. This is the obvious PyTorch expression of the
Definition handed to `torch.compile` with no tuning -- no `mode=`, no
`options=`, no `dynamic=` hint -- because that is what a practitioner reaches
for first on an op with no library implementation, and because the question a
baseline should answer is "how much of this does the compiler already get?".

Choosing it over the alternative is deliberate. vLLM's own C128A path is a
hand-written Triton kernel (`fused_compress_quant_cache.py`), but vLLM is not a
dependency of this benchmark, and transcribing that kernel here would produce a
second hand-written Triton solution -- a competitor to the one already in the
trace set, not a baseline to measure against.

What the compiler gets: Inductor fuses the softmax and the weighted sum into a
single pass over the bf16 inputs. That is the only thing that matters on a
streaming op, and it is exactly what the Definition's reference fails to do --
the reference casts both [n, 128, 512] inputs to fp32 up front and eats two
268 MB temporaries at num_compressed = 1024.

## torch.compile is wrong here by default, and the tolerances catch it

The noPE columns are rounded to bf16 **once**, and both `ckv` and the FP8 block
scales read that rounded copy. That double rounding is what makes `ckv_fp8` a
deterministic encoding of `ckv` rather than an independent quantisation of the
fp32 value -- it is the third of the three fidelity points the Triton solution
also has to respect.

Written inline as `normed[:, :448].to(torch.bfloat16).float()`, **Inductor
deletes it.** Measured on 2^20 random fp32 values (torch 2.11.0+cu128):

    eager    round-trip  vs eager  no-round-trip : 32450/1048576 differ (3.09%)
    compiled round-trip  vs eager  round-trip    : 32450/1048576 differ (3.09%)
    compiled round-trip  vs eager no-round-trip  :     0/1048576 differ

The compiled form is bit-identical to the form with the cast deleted. The
failure is silent in the obvious place and loud in the subtle one: `ckv` still
matches, because it rounds to bf16 on the way out anyway, while 3.5% of
`ckv_fp8` bytes come out one e4m3 code point off. Scored against the reference
that is `matched_ratio = 0.96498` -- *inside* the band eval_config.yaml reserves
for deliberate semantic mutations, whose worst case is 0.96282. Setting
`torch._inductor.config.emulate_precision_casts = True` does not prevent it.

This is worth stating plainly, because it is the first thing in this task that
the tolerance design caught rather than confirmed: a `required_matched_ratio` of
0.999 chosen to separate a correct Triton kernel (worst 0.99998692) from eight
mutations (worst 0.96282) also rejects, unprompted, a compiler-elided cast that
no amount of reading the Python would reveal.

The fix is the two-region split in the source: `_reduce` returns real bf16
storage, so `_quant`'s `.float()` is a load-and-convert with nothing to fold
through. The extra region reads back n x 448 bf16 -- 0.9 MB at
num_compressed = 1024, against the 268 MB the op streams, so it costs nothing
measurable. With the split, the worst `matched_ratio` over the sweep is
0.99999128.

## Measured: 30% of peak, and a flat 0.20 ms floor

Full trace over the 20-workload sweep, all PASSED (`tools/run_benchmark.py`,
then `tools/roofline.py`; H200, torch 2.11.0+cu128, peak 4216 GB/s measured as a
sustained d2d copy). The sweep now runs on model-derived inputs rather than
`torch.randn` -- real DeepSeek-V4 compressor weights and RoPE tables, real
activations lifted from DeepSeek-V2-Lite (see §7.7 of tasks/HCA.md):

    num_compressed        1      64     256     512    1024
    streamed (MB)       0.3    16.9    67.6   135.1   270.2
    latency (ms)     0.1764  0.2078  0.2043  0.2043  0.2120
    GB/s                1.5    81.3   330.7   661.3  1274.4
    % of peak          0.04    1.93    7.84   15.68   30.22
    vs reference       1.54x   1.31x   2.46x   3.93x   6.83x
    vs Triton seed     0.53x   0.64x   1.64x   2.97x   5.80x

The figures are the same as on the previous all-random sweep to within run-to-run
noise (best was 1227.6 GB/s / 29.12%), which is the expected result: three of the
workloads differ *only* in their input distribution, and timing all of them in
one process puts them within 4% of each other (§7.7 of tasks/HCA.md). Values buy
correctness coverage on this op, not performance signal. What does move this
solution is how many shapes Inductor has already compiled -- see the floor below.

Two things in that table are worth reading carefully.

**It is 5.8x the hand-written Triton solution at the largest workload** (1274.4
vs 219.8 GB/s), which is the point of having a baseline at all: the seed kernel
spills 946 B/thread and never gets past 5.3% of peak, so "beats eager PyTorch by
1.18x" was flattering it. Against the compiler it loses by 5.8x.

**The latency is flat at ~0.20 ms from num_compressed = 2 to 768**, while the
bytes moved grow 400-fold. The compiled op is not bandwidth-bound over most of
the sweep -- it is bound by a fixed cost of about 0.2 ms, and only at
num_compressed = 1024 does streaming start to dominate. (num_compressed = 1 is
the one workload Inductor compiles as a static shape, and it is the only point
below the floor, at 0.1764 ms; every later shape gets the dynamic kernel and its
guard overhead.) So the 30% of peak at the top of the sweep is the only figure
here that measures memory throughput at all, and a candidate kernel has roughly
two separate targets: beat 0.2 ms of overhead below num_compressed ~= 768, and
beat 1274 GB/s above it. That split is itself a finding about the workload sweep
-- see §7.6 of tasks/HCA.md.
