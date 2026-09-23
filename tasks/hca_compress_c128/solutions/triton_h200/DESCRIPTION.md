One Triton program per compressed entry. Each program owns a whole 128x512
window, so the softmax reduction stays inside a single program: no cross-block
reduction, no atomics, no second pass. That is affordable precisely because HCA
collapses 128 tokens into one entry, so the grid is num_compressed wide and each
program has 512 columns of work.

Fidelity points that the tolerances are tight enough to enforce:

* RMSNorm divides by head_dim (512), not by the 448 noPE columns.
* RoPE is GPT-J / interleaved-pair, applied to the last 64 columns; the angle
  comes from the window start (i // 128) * 128, not from the boundary token.
* The rotation reads the fp32 normalised vector, while FP8 quantisation and
  ckv's noPE columns read a single bf16-rounded copy of it. Keeping those two
  reads distinct is what makes ckv_fp8 a deterministic encoding of ckv.
* The UE8M0 exponent keeps the division inside the log: log2(x / 448) is not
  bit-identical to log2(x) - log2(448).

Triton requires power-of-two shapes, so the 7 quantisation blocks are padded to
8 on the block axis and masked off at the store; nothing is reshaped to 7 rows.
The RoPE suffix is handled by rotating all 256 column pairs with a masked cos /
sin load that yields (1, 0) outside the suffix, so the pass-through pairs come
out unchanged and the 512-wide result needs one store.

Measured on an H200 (torch 2.11.0+cu128, triton 3.6.0) against the Definition's
reference over num_compressed = 1..1024: ckv matched_ratio 1.00000000
throughout, ckv_fp8 worst 0.99998692, ckv_scale bitwise exact at every
workload. The residual disagreement is one bf16 ulp / one e4m3 code point on
elements sitting on a rounding boundary, caused by PyTorch and Triton reducing
the 128-token softmax in different orders -- see eval_config.yaml.

## Performance: correct, and deliberately far from good

Full trace over the 20-workload sweep, all PASSED (`tools/run_benchmark.py`,
then `tools/roofline.py`). The sweep runs on model-derived inputs -- real
DeepSeek-V4 compressor weights and RoPE tables, real activations lifted from
DeepSeek-V2-Lite (provenance in `tools/gen_workload_blobs.py`):

    num_compressed     1      64     256    1024
    latency (ms)   0.0934  0.1338  0.3349  1.2295
    GB/s              2.8   126.2   201.7   219.8
    % of peak        0.07    2.99    4.78    5.21
    vs reference     2.91x   2.04x   1.50x   1.18x

Peak is 4218 GB/s, measured on this device as a sustained d2d copy. The kernel
tops out at **222.5 GB/s, 5.3% of peak -- 19x off roofline** -- on an op that is
pure streaming: read two [n, 128, 512] bf16 tensors, write ~1 KB per entry.

The cause is in the compiled kernel, not the algorithm:

    num_warps=8  n_regs=32  n_spills=946 B/thread  shared=8192 B

946 bytes spilled per thread over 256 threads is 236 KiB of spill traffic per
block, against the 256 KiB of input the block streams -- the kernel very nearly
doubles its own memory traffic, and the spill loop serialises what should be a
straight stream. A whole 128x512 window held live in fp32 is 256 KiB of working
set per block; the register file cannot hold it, and nothing here tiles the
reduction to avoid having to.

This is the right property for a seed solution. It is correct to the tolerances,
so it anchors the correctness side of the task, and it leaves the entire 19x on
the table for a candidate to win honestly. The obvious first moves -- tile the
compress_rate axis so the live set fits, or use shared memory for the window --
are exactly the kind of work the benchmark is meant to reward.

It also shows why `speedup_factor` alone is a weak objective here: it reads
2.91x at num_compressed=1, where both sides are launch-overhead (the kernel does
0.3 MB of real work in 93 us), and 1.18x at num_compressed=1024, where the
reference is bandwidth-bound too and 19x of headroom is invisible. Judge
candidates on achieved bandwidth.
