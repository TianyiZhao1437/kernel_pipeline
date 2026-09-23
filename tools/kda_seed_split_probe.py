"""Where does the Triton seed's time actually go, and what fraction of peak is it?

The seed's docstring makes two claims that are easy to write and easy to get
wrong: that the sequential UT transform is where its latency goes, and that it
leaves obvious throughput on the table. Both are measurable. Measure them.

Splits the two kernels, times each, and converts to TFLOP/s using the task's own
flops_model so the number is comparable to what report_traces will print.
"""

import importlib.util
import pathlib
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
TASK = REPO / "tasks" / "kda_prefill_h32_d128"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(TASK / "solutions" / "triton_seed"))


def timed(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    import json
    from derive_kda_tolerances import load_corpus
    import kda_prefill_h32_d128 as seed
    from flashinfer_bench.data import Definition

    sys.path.insert(0, str(TASK))
    import flops_model

    definition = Definition.model_validate(
        json.loads((TASK / "kda_prefill_h32_d128.json").read_text()))

    print(f"{'B':>2} {'T':>6}  {'prep ms':>8} {'scan ms':>8} {'total':>8} "
          f"{'prep %':>7}  {'GFLOP':>8} {'TFLOP/s':>8}")
    for axes, inputs in load_corpus(only={(1, 1024), (1, 4096), (1, 16384), (2, 8192)}):
        b, t, h, d = inputs["query"].shape
        nc = (t + 64 - 1) // 64
        shape = (b, h, nc, 64, d)
        bufs = [torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
        ain = torch.empty((b, h, nc, 64, 64), dtype=torch.bfloat16, device="cuda")
        glast = torch.empty((b, h, nc, d), dtype=torch.float32, device="cuda")
        qg, ke, w, kcd = bufs

        q, k, v, g, beta = (inputs[n] for n in ("query", "key", "value", "g", "beta"))

        def prep():
            seed._prepare[(nc, b * h)](
                q, k, v, g, beta, qg, ke, w, kcd, ain, glast, t, nc,
                q.stride(0), q.stride(1), q.stride(2),
                beta.stride(0), beta.stride(1),
                qg.stride(0), qg.stride(1), qg.stride(2),
                ain.stride(0), ain.stride(1), ain.stride(2),
                glast.stride(0), glast.stride(1), glast.stride(2),
                C=64, D=d, H=h, num_warps=4, num_stages=2)

        out = torch.empty((b, t, h, d), dtype=q.dtype, device="cuda")
        st0 = inputs["initial_state"].float()
        st1 = torch.empty_like(st0)

        def scan():
            seed._scan[(b * h, d // seed.BV)](
                qg, ke, w, kcd, ain, glast, st0, out, st1, t, nc,
                qg.stride(0), qg.stride(1), qg.stride(2),
                ain.stride(0), ain.stride(1), ain.stride(2),
                glast.stride(0), glast.stride(1), glast.stride(2),
                st0.stride(0), st0.stride(1), st0.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                C=64, D=d, H=h, BLOCK_V=seed.BV, num_warps=4, num_stages=2)

        prep()
        ms_p = timed(prep)
        ms_s = timed(scan)
        total = timed(lambda: seed.run(**inputs))
        flops = flops_model.compute_flops(definition, axes)
        print(f"{b:>2} {t:>6}  {ms_p:>8.3f} {ms_s:>8.3f} {total:>8.3f} "
              f"{100 * ms_p / (ms_p + ms_s):>7.1f}  {flops / 1e9:>8.2f} "
              f"{flops / (total * 1e-3) / 1e12:>8.2f}")

        if (b, t) == (1, 16384):
            # Attribute _prepare's time to the UT loop, by timing a variant whose
            # loop is truncated. The variant computes the wrong answer; it is a
            # timing probe, not a solution, which is why it is built here by text
            # substitution rather than by putting a knob in the kernel source.
            print("\n  UT loop attribution (B=1 T=16384, WRONG RESULTS, timing only):")
            src = (TASK / "solutions" / "triton_seed" / "kda_prefill_h32_d128.py").read_text()
            for steps in (1, 8, 32, 64):
                # Triton refuses to jit a function that has no source file, so
                # the variant is written to disk rather than exec'd from a string.
                vpath = pathlib.Path(f"/tmp/kda_ut_variant_{steps}.py")
                vpath.write_text(src.replace("for i in tl.range(1, C):",
                                             f"for i in tl.range(1, {steps}):"))
                vspec = importlib.util.spec_from_file_location(f"ut{steps}", vpath)
                mod = importlib.util.module_from_spec(vspec)
                vspec.loader.exec_module(mod)
                kern = mod._prepare

                def prep_v():
                    kern[(nc, b * h)](
                        q, k, v, g, beta, qg, ke, w, kcd, ain, glast, t, nc,
                        q.stride(0), q.stride(1), q.stride(2),
                        beta.stride(0), beta.stride(1),
                        qg.stride(0), qg.stride(1), qg.stride(2),
                        ain.stride(0), ain.stride(1), ain.stride(2),
                        glast.stride(0), glast.stride(1), glast.stride(2),
                        C=64, D=d, H=h, num_warps=4, num_stages=2)

                prep_v()
                ms = timed(prep_v)
                print(f"    UT steps {steps:>3}:  _prepare {ms:>7.3f} ms  "
                      f"({100 * ms / ms_p:>5.1f}% of the real one)")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
