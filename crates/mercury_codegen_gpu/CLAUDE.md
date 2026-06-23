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
  typed launch wrappers: `saxpy`/`vadd`/`copy` (streaming), `vmath` (activations), `reduce` (sum/dot/max),
  device-property queries `peak_hbm_gbs`/`sm_count` (the M9 bandwidth denominator), `gemm_nt`/`_rb`
  (f32, simple + register-blocked), `gemm_nt_f16`/`_bf16` (WMMA tensor core), `gemm_nt_fp8` + `fp8_tile`
  (fp8 mma.sync), `norm` (softmax/LayerNorm/RMSNorm), `conv2d`, `flash_attn`, and `transformer_layer`
  (a whole pre-norm encoder layer, end-to-end GPU-resident — chains the above on device buffers with no
  host round-trip; `TransformerWeights` bundles the six projections).
- `src/cubin.rs` — persistent **cubin cache** (M10): `ptx_to_cubin` runs the driver's `cuLink*` JIT to
  emit SASS; `Gpu::load_module_cached` caches it on disk (keyed by PTX hash + driver version) so warm
  processes load precompiled cubins via `cuModuleLoad` instead of re-JITing. Graceful fallback to a
  direct PTX JIT on any miss/failure.
- `src/ptx.rs` — base PTX (saxpy/vadd/vmath/reduce/simple GEMM) + `COPY_V4`, the vectorized streaming
  copy (128-bit `ld/st.global.v4` with 4× ILP) that drives the M9 HBM-bandwidth bench; target `sm_89`.
- `src/ptx_gemm.rs` — register-blocked f32 GEMM generator (64×64 tile, 4×4/thread).
- `src/ptx_wmma.rs` — WMMA fp16/bf16 tensor-core GEMM generators: single-tile, fragment-reuse `_mt`,
  shared-memory-staged `_sm` (cooperative CTA tiles, vectorized 128-bit loads), **`cp.async`
  double-buffered** `_sm_db`/`_sm128_db` (pipelined K-loop; `_sm_db` ≈ cuBLAS at 1024³), and a **fused
  activation epilogue** (`Act` enum — relu/silu/gelu applied to the f32 accumulators before the C store;
  the transcendentals reuse `vmath`'s exact SFU formulas → `_sm_db_{relu,silu,gelu}` beat the cuBLAS
  GEMM+activation chain ~1.1–1.4×, the thing cuBLAS can't fuse), plus a **fused bias epilogue**
  `C = act(A·Bᵀ + bias)` (`entry_smem_db`'s `bias` flag → `_sm_db_bias{,_relu,_silu,_gelu}`). The WMMA
  fragment→column map is opaque, so the bias path `wmma.store.d`s each tile into a per-warp SMEM scratch
  (reusing the freed staging buffer) and re-reads by explicit (row,col) to add `bias[col]` — the
  canonical `nn.Linear`/FFN epilogue cuBLAS needs a 2nd kernel for. `gemm_nt_f16` dispatches the plain
  GEMMs by size regime. The pipelined+fused generators are **precision-generic** (`entry_smem_db` keys
  fragment width/mma type off `ty`), so bf16 gets the same `wmma_nt_bf16_sm_db` + `_sm_db_{relu,silu,
  gelu}` (the training-dtype fusion; bias is fp16-only so far). See `entry_smem`/`entry_smem_db`.
- `src/ptx_fp8.rs` — fp8 (E4M3) `mma.sync.m16n8k32` tile + tiled GEMM, single-tile and fragment-reuse
  multi-tile (`_mt`, 2×4 16×8 tiles/warp — the fastest tensor-core path; no WMMA fp8 on sm_89, so the
  fragments are hand-placed per the PTX-ISA lane layout) + host-side E4M3 round/widen.
- `src/ptx_int4.rs` — **W4A16 int4 weight-only decode** (M4 — the LLM-decode workhorse, an *immature*
  GPU-library field so a documented lead). Host group-wise int4 quant (`quantize_weight_symmetric` /
  `quantize_weight_asymmetric` — AWQ/GPTQ zero-point, group=128) into the **Marlin/AWQ-interleaved**
  packed layout (`nibble_pos`), an exact-bit fp16 dequant reference (`dequant_weight`/`reference_w4a16`),
  and the `gemm_nt_w4a16` PTX: a 64×64 SMEM-staged tensor-core tile that reads **packed int4 from global**
  (8 weights/`u32` — 4× the fp16 weight-footprint shrink) and **unpacks to fp16 on the fly** with the
  canonical fast path (one `lop3` extracts a pair into an `f16x2` of `1024+u`, then one `sub.rn.f16x2`
  zero-offset + one `mul.rn.f16x2` scale dequant two weights/op), then runs the *identical* fp16
  `wmma.mma.sync.m16n16k16` as the dense path. Symmetric (offset-binary, `Z=8`) and zero-point (`Z=zero`)
  share one unpack. `w4a16_static_ptx` is the **static-shape** variant (dims baked → ptxas strength-reduces
  the strides). Launchers `gemm_nt_w4a16` / `gemm_nt_w4a16_static` in `gpu.rs`; peer + scoreboard in
  `baselines.rs` (`nvrtc_naive_w4a16`) / `gpu.rs` (`int4_gemm_vs_peers`).
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
