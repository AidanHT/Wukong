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
- `src/ptx_norm.rs` — fused row-norm generators (softmax/LayerNorm/RMSNorm, one warp/row, shfl reduce).
- `src/ptx_flash.rs` — fused flash-attention generator (online softmax, warp-per-query-row, D∈{32,64,128}).
- `src/ptx_conv.rs` — direct conv2d (one thread per output element).
- `src/diff.rs` — tolerance harness (`Rng`, `assert_close`/`assert_scalar_close`).
- `src/lower.rs` — **general MIR→PTX lowering** (Phase 4, `--backend=gpu-native`): lowers an *arbitrary*
  Mercury program to one PTX kernel (SSA→vregs, block params→register copies, control flow→predicated
  `bra`, allocas→a `.local`/`.global` frame, print/assert→a device record buffer replayed host-side).
  `jit_run` tries the **megakernel** first, else the single-thread lowering (`jit_run_single`, the
  universal reference). Matches the interp oracle on **97/97 of `tests/run`** (-O0==-O3). Also owns the
  mega lowering (`emit_mega_ptx`, `FnEmit::lower_mega`) + the cooperative reduce PTX + `emit_chunked_call`.
- `src/fusion.rs` — **op-graph analysis** (no device): `classify_call` (every `mercury_*` symbol →
  `CoopKind`), `mem_tainted` (fixpoint memory-dependence), and `analyze` → the megakernel **safety gate**
  (a program is eligible only if its control flow is *data-independent* — no `CondBr` reads memory — so
  the SPMD cooperative kernel can't diverge and deadlock a `bar.sync`), plus the recognized-op inventory.
- `src/megakernel.rs` — **whole-program cooperative megakernel** (Phase 8 / M13): compile an eligible
  program into ONE `.visible .entry` kernel run by a 256-thread block — the alloca frame is one shared
  `.global` buffer, every store/side-effect is `tid==0`-guarded, and each recognized op runs cooperatively
  (`mrt_sreduce_coop` tree; elementwise/GEMM/norm via `emit_chunked_call`'s per-thread chunked decomposition
  reusing the serial `mrt_*` kernels) bracketed by `bar.sync`. `try_run` gates via `fusion::analyze`,
  launches one block, decodes the same print/exit buffer → byte-identical output, else `Ok(None)` →
  single-thread fallback. Measured same-run (checksum-cross-checked, clock-invariant): cooperative vs
  single-thread reduce ~73× / silu ~107–120× / GEMM ~38×; **M13 one-launch vs the per-op offload chain
  ~285×** (`mega_vs_chain_reduce` — launch-overhead + residency elimination, vs Mercury's own
  `--backend=gpu` model, not a library). The `.local`→`.global` frame is the lever that lets the block
  cooperate (a per-thread `.local` frame can't be work-split); the data-independence gate is what makes
  SPMD barriers deadlock-free.

## Key facts / gotchas
- **One process-wide `Gpu` behind a `Mutex`.** `cargo` runs tests on many threads and a CUDA context
  is current-per-thread; funneling all driver calls through one locked context both serializes them
  and reuses one primary context + JITed modules across the whole run (init/JIT is not free).
- **PTX is JITed once per `key` and cached** in `Gpu::modules`. Use a stable `&'static str` key.
- **Tests skip gracefully without a GPU**: `with_gpu` returns early if `gpu()` is `None`, so the
  suite is green on GPU-less machines even when compiled `--features gpu`.
- Verified box: NVIDIA RTX 4050 Laptop (Ada `sm_89`, 6 GB), driver 592.27, `cudarc` 0.16
  `dynamic-loading`. `nvcc`/`ptxas` absent — the driver JIT is what compiles the PTX.
- **Megakernel chunked-cooperative pointer strides are per-pointer, not per-op.** `mrt_vmath_bf16/f16`
  reads bf16 input (2 B) but writes **f32 output (4 B)** — a uniform stride-2 chunk made the f32 store
  land at a 2-aligned (not 4-aligned) address → `CUDA_ERROR_MISALIGNED_ADDRESS`, which makes the context
  *sticky*-errored so every later op (incl. JIT loads) fails the same way (looks like a cascade; the root
  is one bad access). Encode each pointer's element size separately in `emit_chunked_call`'s strides.
- **A `bar.sync` divergence-deadlock surfaces as a recoverable TDR launch error, not a permanent hang**
  (Windows resets the GPU after ~2 s), so the mega gates are safe to run; the data-independence
  eligibility gate (`fusion::analyze`) is what prevents the divergence in the first place.
