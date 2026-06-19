# mercury_codegen_gpu

Mercury's **GPU backend**. Mercury is a compiler, so the GPU path **emits PTX text** and
**driver-JIT-loads it via `cudarc`** (`cuModuleLoadData` — the NVIDIA driver's built-in PTX→SASS
JIT, which needs no `nvcc`/`ptxas`/`nvrtc`/`clang`), launches over device buffers, and copies back.
This is the GPU analogue of the CPU seam in `mercury_runtime`: a recognizer picks a kernel symbol,
this crate provides the kernel (as PTX) + a host launcher, and the interpreter stays the oracle —
across the CPU/GPU boundary the gate is a **tolerance** differential (`c·√K·ε`), not bit-exact,
because the GPU reassociates/rounds differently.

## Feature gate
All GPU code is behind the **`gpu`** feature (`cudarc` is an optional dep). A plain `cargo test`
compiles an essentially empty crate (no `cudarc`, no GPU needed) so the toolchain-free CPU core is
untouched. Run GPU work with `cargo test -p mercury_codegen_gpu --features gpu`. `cudarc` uses
`dynamic-loading` (dlopens `nvcuda.dll`), so even with the feature on, **building** needs no CUDA
toolkit — only **running** needs the driver + a device.

## Layout
- `src/lib.rs` — crate root; `GPU_ENABLED` const; re-exports behind `#[cfg(feature = "gpu")]`.
- `src/gpu.rs` — host harness: `Gpu` (context + default stream + PTX-module cache), the process-wide
  `gpu()` accessor (a `Mutex<Option<Gpu>>` — `None` means no device → tests *skip*, not fail), and
  typed launch wrappers (`saxpy`, `vadd`, …).
- `src/ptx.rs` — hand-emitted PTX kernel sources (string constants), target `sm_89`.

## Key facts / gotchas
- **One process-wide `Gpu` behind a `Mutex`.** `cargo` runs tests on many threads and a CUDA context
  is current-per-thread; funneling all driver calls through one locked context both serializes them
  and reuses one primary context + JITed modules across the whole run (init/JIT is not free).
- **PTX is JITed once per `key` and cached** in `Gpu::modules`. Use a stable `&'static str` key.
- **Tests skip gracefully without a GPU**: `with_gpu` returns early if `gpu()` is `None`, so the
  suite is green on GPU-less machines even when compiled `--features gpu`.
- Verified box: NVIDIA RTX 4050 Laptop (Ada `sm_89`, 6 GB), driver 592.27, `cudarc` 0.16
  `dynamic-loading`. `nvcc`/`ptxas` absent — the driver JIT is what compiles the PTX.
