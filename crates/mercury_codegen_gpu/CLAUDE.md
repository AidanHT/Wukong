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
- `src/ptx_wmma.rs` — fp16/bf16 tensor-core GEMM generators. **Large-GEMM path (the dispatched workhorse,
  per the `gemm_nt_f16`/`gemm_nt_bf16` size-regime dispatch):** an **N-stage `cp.async` software pipeline**
  with configurable staged BK and **threadblock rasterization** (`entry_smem_pipe`, WMMA; `entry_mma_pipe`,
  native `mma.sync.m16n8k16` with hand-placed **bank-conflict-free (8-padded) SMEM** fragment loads — the
  route past the ~72% WMMA ceiling). The `PipeCfg`/`PIPE_VARIANTS` table drives the generator, gate
  (`wmma_pipe_matches_reference`), sweep (`gemm_pipe_sweep`), and dispatch from one source. Same-run vs
  cuBLAS on the RTX 4050 (clock-/contention-sensitive — only %-of-peer is reportable): **≤1024³ ~90%**
  (deep BK=16 WMMA pipe `pipe_64_s6`), **2048³ ~90–97%** and **4096³ ~77%** (`mma_*_r16`, padded + r16
  raster; up from a pre-pipeline 79%/66%/32%). 4096³ is HBM-bound — bigger dynamic-SMEM tiles, 2-D raster,
  and *naive* non-padded occupancy were all swept and *lost*; ~77% is near the PTX-`mma.sync` ceiling
  (cuBLAS's remaining edge is SASS-level). **One lever did reclaim a bit at 4096³: `ldmatrix` + XOR-swizzle
  + no-pad** (`entry_mma_pipe`'s `swz` flag → `mma_nt_f16_128_bk32_s2_r16_swz`). Dropping the pad (40→32 KiB
  ⇒ 3 CTAs/SM vs the padded 2) + an XOR swizzle (`chunk ↦ chunk XOR ((row>>1)&3)`) keeps *both* the
  cp.async stores and the `ldmatrix.x4/.x2` gathers conflict-free at that higher occupancy — the extra
  CTAs/SM hide the HBM latency the padded 2-CTA tile can't. Weakly dominant same-run (`mma_swizzle_vs_
  handplaced`, 7 runs @4096³ mean ~1.04×, never < parity, ~72%→~74% of cuBLAS) — so it's **regime-dispatched
  for A+B ≳ 2×L2 only** (the L2-resident 2048³ loses ~0.86× to the occupancy thrash, keeps the padded base).
  NB the *un-swizzled* padded ldmatrix LOSES ~20% (a trap — the swizzle, not ldmatrix alone, is the lever).
  Gated `mma_swizzle_matches_reference` (max_abs ≤ 1.3e-5). Also: single-tile/`_mt`, `_sm`, **`cp.async` double-buffered** `_sm_db`/
  `_sm128_db` (the older 64×64/128×128 staged kernels, now fallbacks), and a **fused
  activation epilogue** (`Act` enum — relu/silu/gelu applied to the f32 accumulators before the C store;
  the transcendentals reuse `vmath`'s exact SFU formulas → `_sm_db_{relu,silu,gelu}` beat the cuBLAS
  GEMM+activation chain ~1.1–1.4×, the thing cuBLAS can't fuse), plus a **fused bias epilogue**
  `C = act(A·Bᵀ + bias)` (`entry_smem_db`'s `bias` flag → `_sm_db_bias{,_relu,_silu,_gelu}`). The WMMA
  fragment→column map is opaque, so the bias path `wmma.store.d`s each tile into a per-warp SMEM scratch
  (reusing the freed staging buffer) and re-reads by explicit (row,col) to add `bias[col]` — the
  canonical `nn.Linear`/FFN epilogue cuBLAS needs a 2nd kernel for. The **same fused epilogue also rides
  the fast `mma.sync` workhorse** (`entry_mma_pipe`'s `act`/`bias` args → `mma_nt_f16_128_bk32_s2_r16_bias
  {,_relu,_silu,_gelu}`): the mma kernel's D-fragment column map is *known*, so `bias[col]` is added to the
  f32 accumulators **register-level** (no SMEM scratch the opaque WMMA path needs) then the activation,
  before the store. This fuses onto the *fastest* GEMM base (not the slower `_sm_db`), so it beats the
  plain-cuBLAS GEMM+epilogue chain **~1.1–1.4× at 512³ and 2048³** (the FFN-relevant regime — large K,N;
  `fused_gemm_bias_act_vs_chain`, same-run interleaved). `entry_mma_pipe`'s third epilogue flag `residual`
  adds a per-element `residual[M,N]` to the post-activation accumulators → `out = act(A·Bᵀ+bias)+residual`
  (`mma_nt_{f16,bf16}_128_bk32_s2_r16_bias_residual` / `gemm_nt_{f16,bf16}_mma_bias_residual`): the
  transformer **down-proj / attention output-proj** sublayer output, folding the bias-add AND the
  skip-connection-add (the two `+residual` points per block) into the GEMM store. The mma-workhorse base
  wins 512³/2048³ but **lost at 1024³ (~0.90×)** — there the workhorse isn't the per-regime GEMM winner
  (the deep WMMA `pipe_64_s6` is, ~90% of cuBLAS vs ~80%), so its GEMM deficit outweighs the saved
  epilogue round-trip. **FIXED by porting the epilogue onto `pipe_64_s6`** (`entry_smem_pipe` gained the
  same `act`/`bias`/`residual` args → `wmma_nt_f16_pipe_64_s6_bias{,_relu,_silu,_gelu,_residual}`): WMMA's
  fragment column map is opaque, so the per-column bias routes through SMEM store-back scratch (the same
  trick `entry_smem_db` uses — `smemA` is free post-K-loop, re-read by explicit (row,col)) and the
  residual seeds the accumulator via `wmma.load.c`. A **size-aware `gemm_nt_f16_linear{,_relu,_silu,
  _gelu}`** dispatcher routes `pipe_64_s6` ≤1024² and the mma workhorse larger, so the fused FFN/Linear/
  down-proj now **beats the cuBLAS chain at EVERY size** (same-run, 3 runs): **1024³ pipe64 1.04–1.07×**
  where the mma workhorse base is 0.89–0.93× (pipe64 ~12200 GFLOP/s vs the workhorse's ~10500); 512³
  1.06–1.72×. Gated by `wmma_pipe64_bias_match_reference_within_tol` (5 variants, max_abs ≤ 2.3e-5).
  This is **fp16-specific**: bf16/fp8 ride only the `mma.sync` workhorse (no deep WMMA pipe), so there is
  no ≤1024³ champion base to fuse onto — the 1024³ GEMM gap is an fp16-internal competition (its WMMA pipe
  vs its mma workhorse) that bf16/fp8 don't have. `gemm_nt_f16` dispatches the plain GEMMs by size regime.
  **All the fused generators are
  precision-generic** (`entry_smem_db` and `entry_mma_pipe` key fragment width / mma type and the bias/act
  epilogue off `ty`), so **bf16 — the training precision — gets the identical fusion suite**: the WMMA
  `wmma_nt_bf16_sm_db{,_relu,_silu,_gelu}` + `_sm_db_bias{,_relu,_silu,_gelu}` *and* the fast-mma
  `mma_nt_bf16_128_bk32_s2_r16_bias{,_relu,_silu,_gelu}` (`gemm_nt_bf16_mma_bias*`). Speed for the bf16
  fast-mma fused path is inherited (Ada runs bf16 and fp16 `mma.sync` at the same TC rate, byte-identical
  generator); it is correctness-gated (`wmma_bf16_mma_bias_match_reference_within_tol`) but has no same-run
  speed headline of its own — cudarc exposes no cuBLAS bf16 peer here. **Gated-FFN (GLU-family) fusion**
  rides the same fast base through a dedicated dual-B generator (`entry_mma_gate`): `out = act(x·Wgᵀ) ⊙
  (x·Wuᵀ)` — the **SwiGLU** (silu) / **GeGLU** (gelu) / bilinear-GLU FFN gate every modern LLM runs, fp16 +
  bf16, five variants each (±per-column gate/up bias). A **128×64** tile holds TWO accumulator sets at the
  same 64 f32 D-regs/thread and three staged tiles (A + Wg + Wu) at the same 40 KiB as the single-B 128×128
  workhorse (register- and SMEM-neutral), so one staged `x` tile feeds both GEMMs (x read **once** via shared
  A fragments) and the gate (activation on the gate branch + the elementwise product) folds into the store.
  cuBLAS structurally needs **three** kernels for this (two GEMMs + an elementwise multiply, both `[M,N]`
  intermediates round-tripped through HBM) — same-run vs that chain (`fused_swiglu_gate_vs_chain`, interleaved
  best-of-6, 3 runs): **512³ ~1.25–1.31×, 1024³ ~1.01–1.10×, 2048³ ~1.01–1.03× faster** (preloading both B
  tiles + interleaving the gate/up mma recovered the ILP — the serialized first form *lost* 0.96–0.97× at
  1024³/2048³). Gated by `swiglu_gate_match_reference_within_tol` (max_abs ≤ 4.3e-4 fp16 / 1.9e-4 bf16). See
  `entry_smem`/`entry_smem_db`/`entry_mma_pipe`/`entry_mma_gate`.
- `src/ptx_fp8.rs` — fp8 (E4M3) `mma.sync.m16n8k32` GEMM (no WMMA fp8 on sm_89, so fragments are
  hand-placed per the PTX-ISA lane layout). **Dispatched large-GEMM path (`gemm_nt_fp8` for %128/%128/%64
  shapes): `fp8_pipe_entry`** — the f16/bf16 mma-pipeline recipe carried to E4M3 (multi-stage cp.async
  staging + threadblock raster + 16-byte-padded conflict-free fragment loads; fp8 is 1 byte/elem so the
  128×128 BK=64 tile is 40 KiB). Same-run: **~1.7–1.9× the old un-staged `_mt`** and **~1.79–1.98× the
  fp16 mma kernel — the Ada 2× fp8-rate realized** (M2; `fp8_pipe_vs_peers`, checksum-cross-checked). Also
  the single-tile and fragment-reuse `_mt` (2×4 16×8 tiles/warp) fallbacks + host E4M3 round/widen. (The
  literal %-of-cuBLASLt-fp8 needs a raw-sys E4M3 peer — cudarc's safe `Matmul` is f32/f16/bf16 only.)
  **Fused epilogue carried to fp8 — the fastest fused inference path:** `fp8_pipe_entry` takes the same
  `act`/`bias` args as `entry_mma_pipe` (the `m16n8k32` D-fragment column map matches `m16n8k16`, so the
  register-level `bias[col]` add + `Act::epilogue` are reused verbatim) → `fp8_gemm_pipe_bias{,_relu,_silu,
  _gelu}` (`gemm_nt_fp8_mma_bias*`), plus the `residual` flag → `fp8_gemm_pipe_bias_residual`
  (`gemm_nt_fp8_mma_bias_residual`, the fastest down-proj / output-proj: `out = x·Wᵀ + bias + residual`,
  residual kept f32). `C = act(x·Wᵀ + bias)` at the Ada 2× fp8 TC rate is the fastest fused Linear/FFN;
  correctness-gated (`fp8_mma_bias{,_residual}_match_reference_within_tol`, max_abs ≤7e-3 vs an
  `act(e4m3-rounded(A·Bᵀ)+bias)[+residual]` f64 ref). Speed inherits the plain fp8-pipe's 2×-fp16 standing
  (no fp8 cuBLAS peer here for a fused-chain ratio). **Gated-FFN gate carried to fp8 too** (`fp8_gate_entry`,
  the dual-B twin of `entry_mma_gate`): `out = act(x·Wgᵀ) ⊙ (x·Wuᵀ)` — the **fastest fused inference gate**
  (SwiGLU/GeGLU at the Ada 2× fp8 rate), `fp8_gemm_pipe_gate_{silu,gelu,glu}{,_bias}` (`gemm_nt_fp8_swiglu`/
  `_geglu`). 128×64 dual-B tile (fp8 = 1 byte/elem ⇒ three staged tiles fit 40 KiB); gated by
  `fp8_swiglu_gate_match_reference_within_tol` (max_abs ≤ 9.6e-2 at K=192 — the single fp8 GEMM's ~1e-2
  precision amplified by the product, honest fp8). Completes fp16/bf16/fp8 parity for the gated FFN.
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
