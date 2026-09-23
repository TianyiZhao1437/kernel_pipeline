"""Essential FLOPs for one workload of kda_prefill_h32_d128.

The compute-bound analogue of ``bytes_model.py``. ``tools/report_traces.py``
uses this when a task provides it, and ranks on achieved TFLOP/s instead of
achieved GB/s.

Why this task gets a FLOP model at all  [measured]
--------------------------------------------------

A bandwidth model would be misleading here, but the reason is narrower than
"the op is compute-bound", and the naive version of that claim is false. At
B=1, T=4096, H=32 the op does 18.59 GFLOP over 196.2 MiB of declared traffic --
an arithmetic intensity of 90.3 FLOP/byte. The H200's large-GEMM ridge point is
772 TFLOP/s measured (989 spec) over 4218 GB/s measured, i.e. 183 FLOP/byte. So
against the *large-GEMM* peak this op sits below the ridge and a roofline would
call it memory-bound.

That comparison is the wrong one, because the op cannot reach the large-GEMM
peak and not because any kernel is bad. It is built out of 64x128x128 tiles, and
those shapes sustain far less than a square 8192 GEMM does. Measured on this
H200 (bf16, batched over 2048 tiles):

    large square 8192^3          772.4 TFLOP/s
    bmm 64x128x128 (terms 5,7)   114.2
    bmm 128x64x128 (term 9)      104.9
    bmm 64x128x64  (terms 1,6)    84.3
    bmm 64x64x128  (terms 3,4,8)  81.8

Against the rate its own shapes sustain, the ridge falls to 19-27 FLOP/byte and
the op is firmly compute-bound. The 7x gap between 772 and ~100 is not overhead
to be ignored; it *is* the kernel design problem this task poses -- how much of
the large-GEMM rate can be recovered from a pile of small dependent tiles.

The practical consequence is for ``tools/report_traces.py``: it must measure
peak at these shapes, not with a big square GEMM. Ranking against 772 would put
every solution near 5% and compress the differences that matter into noise.
``PEAK_SHAPES`` below carries the shapes and their FLOP weights so the report
can measure a composite ceiling rather than quote one.

What is counted: essential, not issued
--------------------------------------

The count below is the useful matmul work implied by the Definition and the
axes -- an invariant of the problem, independent of how any kernel chooses to
compute it. In particular **masked-out elements are not counted**. Four of the
nine matmul terms are triangular (three strictly lower, the two WY products
unit-lower), and a kernel that issues full 64x64 tiles and throws half away is
doing real work that does not appear here. That is deliberate and it has a
visible consequence: such a kernel cannot approach 100% of peak on those terms,
and the report will say so. The alternative convention -- counting issued
FLOPs -- would hand a kernel credit for work it did not need to do, and would
let one that correctly skips the masked half read as *above* peak.

The elementwise work is not counted either: ``g.cumsum``, the ``exp`` calls, the
masked fills. It is not negligible in the reference (the decay mask alone is
``B*H*T*C*D`` elements, 32 KB per token-head, which is why the reference peaks
at 50 GiB and 149 ms at the largest workload) but it is not matmul work, it is
exactly what a fused kernel is supposed to make disappear into registers, and
folding it into a TFLOP/s figure would inflate every kernel equally while
rewarding none of them. It is reported separately by ``elementwise_elements``
so the gap is visible rather than silently absorbed.

Terms
-----

Per (batch, head, chunk), with C = chunk_size, D = head_dim, and a matmul
costing 2*m*n*k:

===  ==========================================  ===========================
 #   expression in the reference                  FLOPs
===  ==========================================  ===========================
 1   ``attn`` build, k_beta . key * decay_mask    2 * lower(C) * D
 2   UT transform, the sequential i-loop          2 * sum_{i<C} i^2
 3   ``value = attn @ v_beta``                    2 * unit_lower(C) * D
 4   ``k_cumdecay = attn @ (k_beta * exp g)``     2 * unit_lower(C) * D
 5   ``attn_inter = (q * exp g) @ state``         2 * C * D * D
 6   ``attn_intra`` build, q . k * decay_mask     2 * lower(C) * D
 7   ``v_prime = k_cumdecay @ state``             2 * C * D * D
 8   ``attn_intra @ v_new``                       2 * lower(C) * D
 9   state update, ``k^T @ v_new``                2 * D * C * D
===  ==========================================  ===========================

At C=64, D=128 that is 9.075 MFLOP per (batch, head, chunk), or 141803 FLOPs
per (batch, head, token). Term 2 is 1.9% of the total and terms 5/7/9 -- the
three square D x D products, which are the ones with no triangular structure to
exploit -- are 69%.

Term 2 deserves a note. It is the only sequential term: 63 dependent steps, each
a tiny rank-1 update, and it is what the UT transform exists to make cheap
relative to the matrix inverse it replaces. At 1.9% of FLOPs it will not show up
in a roofline, but it can easily dominate latency in a kernel that implements it
naively, and a solution whose achieved TFLOP/s is far below its peers with no
other explanation should be checked here first.
"""

CHUNK = 64
HEAD_DIM = 128

# (m, k, n) of each matmul shape the op is made of, with the fraction of total
# FLOPs that runs at that shape, at C=64 D=128. Measured fractions, from the
# term table in the docstring: terms 5+7 = 46.2%, term 9 = 23.1%,
# terms 1+6 = 11.4%, terms 3+4+8 = 17.5%. Term 2 (the UT transform, 1.9%) is
# deliberately absent -- it is a sequence of 63 dependent rank-1 updates, not a
# matmul, so there is no shape to measure it at and no honest peak to compare
# it against.
#
# tools/report_traces.py measures each and combines them as a FLOP-weighted
# HARMONIC mean, which is the arithmetic that actually applies: if fraction f_i
# of the work runs at rate r_i, the total time is sum(f_i / r_i) and the
# composite rate is its reciprocal, not sum(f_i * r_i). On this H200 that comes
# to ~103 TFLOP/s against the 772 a large square GEMM sustains.
PEAK_SHAPES = (
    # label                              m       k       n     flop_weight
    ("attn_inter / v_prime",       CHUNK,  HEAD_DIM, HEAD_DIM,      0.462),
    ("state update",            HEAD_DIM,     CHUNK, HEAD_DIM,      0.231),
    ("attn / attn_intra build",    CHUNK,  HEAD_DIM,    CHUNK,      0.114),
    ("WY products",                CHUNK,     CHUNK, HEAD_DIM,      0.175),
)


def _lower(n: int) -> int:
    """Strictly-lower-triangular element count."""
    return n * (n - 1) // 2


def _unit_lower(n: int) -> int:
    """Lower-triangular including the diagonal."""
    return n * (n + 1) // 2


def flops_per_chunk(chunk_size: int = CHUNK, head_dim: int = HEAD_DIM) -> int:
    """Essential matmul FLOPs for one (batch, head, chunk). See module docstring."""
    c, d = chunk_size, head_dim
    lower = _lower(c)
    unit = _unit_lower(c)

    # sum_{i=1}^{c-1} i^2, the UT transform's growing rank-1 updates
    ut = (c - 1) * c * (2 * c - 1) // 6

    return (
        2 * lower * d          # 1  attn build
        + 2 * ut               # 2  UT transform
        + 2 * unit * d         # 3  attn @ v_beta
        + 2 * unit * d         # 4  attn @ (k_beta * exp g)
        + 2 * c * d * d        # 5  attn_inter
        + 2 * lower * d        # 6  attn_intra build
        + 2 * c * d * d        # 7  v_prime
        + 2 * lower * d        # 8  attn_intra @ v_new
        + 2 * d * c * d        # 9  state update
    )


def _resolved(definition, axes) -> tuple:
    """(batch_size, seq_len, num_heads, head_dim, chunk_size) from const + var axes.

    Const axes live on the Definition and var axes on the workload, so read both
    rather than assuming the workload carries all five.
    """
    def axis(name, default=None):
        if name in axes:
            return int(axes[name])
        spec = definition.axes.get(name)
        value = getattr(spec, "value", None)
        if value is None:
            if default is None:
                raise KeyError(f"axis {name!r} is neither in the workload nor const")
            return default
        return int(value)

    return (
        axis("batch_size"),
        axis("seq_len"),
        axis("num_heads"),
        axis("head_dim"),
        axis("chunk_size", CHUNK),
    )


def compute_flops(definition, axes, solution: str | None = None) -> int:
    """Essential matmul FLOPs for one workload.

    ``solution`` is accepted and ignored: unlike the HCA byte model, where two
    solutions staged an extra fp32 buffer through DRAM that the declared
    operands did not describe, the FLOP count here is a property of the problem.
    A kernel that does more work than this does not get a bigger numerator for
    it -- that is the point of counting essential work.
    """
    batch_size, seq_len, num_heads, head_dim, chunk_size = _resolved(definition, axes)

    # The reference pads up to a chunk boundary and trims the output, so the
    # work really is done on the padded length. Counting the unpadded length
    # would credit a kernel with tokens it never processed, and at seq_len=100
    # that is a 28% error.
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    return (
        batch_size * num_heads * num_chunks * flops_per_chunk(chunk_size, head_dim)
    )


def elementwise_elements(definition, axes) -> int:
    """Elements touched by the non-matmul work, reported alongside but not in the FLOPs.

    Dominated by the decay mask, ``[B, H, NC, C, C, D]``, which the reference
    materialises in full. A fused kernel never writes it, which is most of the
    gap between the reference's achieved TFLOP/s and a real kernel's.
    """
    batch_size, seq_len, num_heads, head_dim, chunk_size = _resolved(definition, axes)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    padded = num_chunks * chunk_size

    decay_mask = batch_size * num_heads * num_chunks * chunk_size * chunk_size * head_dim
    cumsum = batch_size * num_heads * padded * head_dim
    return decay_mask + cumsum
