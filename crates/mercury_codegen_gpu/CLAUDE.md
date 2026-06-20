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
  `gpu()` accessor (a `Mutex<Option<Gpu>>` — `None` means no device → tests *skip*, not fail), and the
  typed launch wrappers: `saxpy`/`vadd`, `vmath` (activations), `reduce` (sum/dot/max), `gemm_nt`/`_rb`
  (f32, simple + register-blocked), `gemm_nt_f16`/`_bf16` (WMMA tensor core), `gemm_nt_fp8` + `fp8_tile`
  (fp8 mma.sync), `norm` (softmax/LayerNorm/RMSNorm), `conv2d`, `flash_attn`, and `transformer_layer`
  (a whole pre-norm encoder layer, end-to-end GPU-resident — chains the above on device buffers with no
  host round-trip; `TransformerWeights` bundles the six projections).
- `src/ptx.rs` — base PTX (saxpy/vadd/vmath/reduce/simple GEMM), target `sm_89`.
- `src/ptx_gemm.rs` — register-blocked f32 GEMM generator (64×64 tile, 4×4/thread).
- `src/ptx_wmma.rs` — WMMA fp16/bf16 tensor-core GEMM generators (single-tile + fragment-reuse `_mt`).
- `src/ptx_fp8.rs` — fp8 (E4M3) `mma.sync.m16n8k32` tile + tiled GEMM, single-tile and fragment-reuse
  multi-tile (`_mt`, 2×4 16×8 tiles/warp — the fastest tensor-core path; no WMMA fp8 on sm_89, so the
  fragments are hand-placed per the PTX-ISA lane layout) + host-side E4M3 round/widen.
- `src/ptx_norm.rs` — fused row-norm generators (softmax/LayerNorm/RMSNorm, one warp/row, shfl reduce).
- `src/ptx_flash.rs` — fused flash-attention generator (online softmax, warp-per-query-row, D∈{32,64,128}).
- `src/ptx_conv.rs` — direct conv2d (one thread per output element).
- `src/diff.rs` — tolerance harness (`Rng`, `assert_close`/`assert_scalar_close`).

## Key facts / gotchas
- **One process-wide `Gpu` behind a `Mutex`.** `cargo` runs tests on many threads and a CUDA context
  is current-per-thread; funneling all driver calls through one locked context both serializes them
  and reuses one primary context + JITed modules across the whole run (init/JIT is not free).
- **PTX is JITed once per `key` and cached** in `Gpu::modules`. Use a stable `&'static str` key.
- **Tests skip gracefully without a GPU**: `with_gpu` returns early if `gpu()` is `None`, so the
  suite is green on GPU-less machines even when compiled `--features gpu`.
- Verified box: NVIDIA RTX 4050 Laptop (Ada `sm_89`, 6 GB), driver 592.27, `cudarc` 0.16
  `dynamic-loading`. `nvcc`/`ptxas` absent — the driver JIT is what compiles the PTX.
