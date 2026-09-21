# thirdparty/cutlass — intentionally empty

Upstream declares CUTLASS as a git submodule:

| | |
|---|---|
| Path | `thirdparty/cutlass` |
| URL | `https://github.com/NVIDIA/cutlass` |
| Commit | `7817e47154d7869320f3fa6b409ec8c5e5958970` |

This repository vendors upstream as a pinned copy and never as a submodule
(see `third_party/VENDOR.md`), so the CUTLASS tree is not checked in. Fetch it
at the pinned commit when a solution needs CUTLASS headers:

```bash
git clone https://github.com/NVIDIA/cutlass /tmp/cutlass
git -C /tmp/cutlass checkout 7817e47154d7869320f3fa6b409ec8c5e5958970
cp -r /tmp/cutlass/* third_party/flashinfer-bench/thirdparty/cutlass/
```

Nothing under `flashinfer_bench/` imports CUTLASS at module scope; only
`language: cuda` solutions built through `flashinfer_bench.compile` do.
