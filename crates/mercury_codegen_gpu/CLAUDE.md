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
- `src/pool.rs` — **device memory pool** (M7): `DevicePool`, a bump arena over one `cuMemAllocAsync`
  slab. `alloc`/`alloc_zeros` hand out a sub-range by bumping a cursor (no driver call), `reset()`
  reclaims everything O(1), `high_water_bytes()` tracks the peak. Handouts are real `CudaSlice<T>`
  (`leak()`→`upgrade_device_ptr()`), so they feed the existing launchers unchanged; `PoolBuf`
  **leaks-not-frees** on drop (the slab owns the bytes). Kills per-op alloc/free/zeroing in the
  resident loop **and** is the prerequisite for graph capture (a captured region must contain no
  synchronizing alloc). `poison(byte)` dirties the slab so a gate can expose any read-of-uninit.
- `src/graph.rs` — **CUDA graph capture/replay** (M7): `Graph::capture(stream, record)` records a fixed
  launch sequence (raw `cuStreamBeginCapture`/`EndCapture` + `cuGraphInstantiateWithFlags(flags=0)`,
  since cudarc's safe wrapper forces AUTO_FREE), `launch()` replays the whole thing with one
  `cuGraphLaunch`; Drop-correct. Plus `PinnedBuf<T>` (page-locked host staging for async H2D/D2H). Two
  gotchas it works around: cudarc's `default_stream()` is the **un-capturable NULL stream** (capture on
  a dedicated `new_stream()`), and cudarc enables **event tracking by default** → once a 2nd stream
  exists it inserts cross-stream waits capture rejects (build the layer + buffers with event tracking
  disabled so nothing carries events; see `with_event_tracking_disabled` in `gpu.rs` tests).
- `src/lib.rs` — crate root; `GPU_ENABLED` const; re-exports behind `#[cfg(feature = "gpu")]`.
- `src/gpu.rs` — host harness: `Gpu` (context + default stream + PTX-module cache), the process-wide
  `gpu()` accessor (a `Mutex<Option<Gpu>>` — `None` means no device → tests *skip*, not fail), and the
  typed launch wrappers: `saxpy`/`vadd`/`copy` (streaming), `vmath` (activations), `reduce` (sum/dot/max),
  device-property queries `peak_hbm_gbs`/`sm_count` (the M9 bandwidth denominator), `gemm_nt`/`_rb`
  (f32, simple + register-blocked), `gemm_nt_f16`/`_bf16` (WMMA tensor core), `gemm_nt_fp8` + `fp8_tile`
  (fp8 mma.sync), `norm` (softmax/LayerNorm/RMSNorm), `conv2d`, `flash_attn`, and `transformer_layer`
  (a whole pre-norm encoder layer, end-to-end GPU-resident — chains the above on device buffers with no
  host round-trip; `TransformerWeights` bundles the six projections). **M7 additions (append-only):**
  `ResidentLayerF16::forward_device_pooled{,_on}` (the pooled forward — same kernels/configs/dtypes as
  `forward_device`, scratch from a `DevicePool`, result into a caller-owned persistent buffer; `_on`
  takes an explicit capturable stream), and the test-module M7 gates/benches
  (`resident_layer_{pooled,graphed}_matches_eager`, `resident_stack_graphed_matches_eager`,
  `pool_graph_vs_unpooled`, `decode_stack_latency`, `overlap_throughput`,
  `concurrent_forwards_throughput`) + the `forward_stack_pooled_on` / `capture_resident_{layer,stack}`
  helpers that fold a whole N-layer stack into one `cuGraphLaunch`.
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
  (deep BK=16 WMMA pipe `pipe_64_s6`), and at **≥2048³ (A+B ≳ 16 MB)** the no-pad `ldmatrix` + XOR-swizzle
  **w24** workhorse (`entry_mma_pipe`'s `swz` flag → `mma_nt_f16_128_bk32_s2_r16_swz`) is the robust
  same-run winner — **2048³ ~87%** (1.23× over the padded hand-placed base, 87.4% vs 70.9% of cuBLAS) and
  **4096³ ~83%** (1.13× over padded, past the prior ~77% `mma.sync` ceiling). The mechanism is the
  *fragment load*, not occupancy: dropping the pad (40→32 KiB ⇒ 3 CTAs/SM vs the padded 2) + an XOR swizzle
  (`chunk ↦ chunk XOR ((row>>1)&3)`) keeps *both* the cp.async stores and the `ldmatrix.x4/.x2` gathers
  conflict-free at that higher occupancy, and the `ldmatrix` fragment gather beats hand-placed
  `ld.shared.b32` at equal occupancy. The padded `mma_*_r16` (r16 raster) and the `_sm_db`/`_sm128_db`
  cp.async kernels are retained as fallbacks. NB the *un-swizzled* padded ldmatrix LOSES ~20% (a trap —
  the swizzle, not ldmatrix alone, is the lever).
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
  **Static-shape (M1, completing the int4/int8/fp16 set):** `entry_smem(static_dims=…)` bakes M/N/K as
  constants → `wmma_f16_sm_static_ptx` / launcher `gemm_nt_f16_static`; **bit-exact** vs the dynamic `_sm`
  (`f16_static_matches_reference`, 64 & 128 tiles). HONEST: the same-run win is **modest and
  contention-noisy (~1.0–1.2×, `f16_static_vs_dynamic_ab`)** — materially smaller than int8's clean 1.34×
  thin-M, because the fp16 `_sm` hot loop is wmma-dense with little per-iteration integer-stride math to
  constant-fold (the lever pays off most on the quantized dequant/swizzle paths, less on dense fp16).
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
  the single-tile and fragment-reuse `_mt` (2×4 16×8 tiles/warp) fallbacks + host E4M3 round/widen.
  **%-of-cuBLASLt-fp8 now bound (M2):** `baselines::cublaslt_gemm_nt_fp8_e4m3` drives the **raw-sys
  `cublaslt::{sys,result}`** E4M3 `cublasLtMatmul` (cudarc's safe `Matmul` is f32/f16/bf16-only; the
  `cublaslt` cudarc feature dlopens `cublasLt64_12.dll` like the other peers). The fp8 hardware TN
  constraint (`transa=T,transb=N`) == the NT column-major mapping (`Cᵀ=B̌ᵀ·Ǎ`) the f16 peer already uses,
  so it drops in; gated vs the same E4M3-rounded f64 ref (`cublaslt_fp8_matches_reference_within_tol`),
  measured same-run by `fp8_vs_cublaslt_pct` (~70–85% of cuBLASLt; the **absolute % is contention-noisy** —
  the cuBLASLt baseline alone swings ~1.5× run-to-run, so trust the internal A/B). **Regime lever
  (`fp8_pipe_config_sweep_vs_cublaslt` found it):** `gemm_nt_fp8_pipe` now dispatches **M≤2048 to a 64×128
  `fp8_gemm_pipe_m64` tile** (double the CTAs ⇒ higher occupancy), 128×128 for larger M. Clock-cancelling
  internal A/B (`fp8_pipe_m64_vs_default_ab`): **1.07–1.15× over the 128×128 default at M≤2048**, ~0.98× at
  4096³ (so 128×128 stays the large-M tile). Bit-identical accumulation (same codegen, only BM differs) ⇒
  rides the same fp8 tolerance gate (`fp8_pipe_regime_matches_reference`, both entries + the 128∤M path).
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
  the strides). `w4a16_splitk_ptx` is the **decode split-K** variant (`entry_w4a16(splitk=true)` +
  `w4a16_splitk_reduce`): launched `gridDim.z = sk`, each CTA writes a partial to its own M×N plane (C
  rebased by `ctaid.z·M·N`, disjoint — no atomics) and a **fixed-order** reduction kernel sums the planes
  ⇒ deterministic (M12). Measured up to **~6.4×** vs un-split on thin-M / small-N decode (2-CTA grid,
  sk=8; `int4_splitk_occupancy`) — int4 benefits *more* than int8 split-K (per-CTA dequant ⇒ more
  occupancy-starved). Launchers `gemm_nt_w4a16` / `gemm_nt_w4a16_static` / `gemm_nt_w4a16_splitk` in
  `gpu.rs`; peer + scoreboard in `baselines.rs` (`nvrtc_naive_w4a16`) / `gpu.rs` (`int4_gemm_vs_peers`).
- `src/ptx_int8.rs` — **int8 (W8A8) tensor-core GEMM** (M3): `u8` activations × `i8` weights → `i32`,
  `mma.sync.m16n8k32.s32.u8.s8.s32`. Same 8-bit `m16n8k32` fragment layout as fp8 (no WMMA int8 on
  sm_89), so it mirrors `ptx_fp8.rs`: `INT8_TILE` (single hand-placed tile), `int8_gemm_ptx`
  (single-tile/warp), `int8_gemm_mt_ptx` (fragment-reuse 2×4 `_mt`), `gen_int8_smdb` (SMEM-staged +
  `cp.async` double-buffered, BK=32, 64×64 / 128×128 variants), and **`gen_int8_smdb_swz`** — the
  **default** path: `ldmatrix.x4`/`.x2` gathers from XOR-swizzled (conflict-free, BK=64) SMEM, the int8
  port of `ptx_wmma.rs`'s proven fp16 swz kernel (byte-geometry of the 16×32-u8 tile == fp16's 16×16, so
  the swizzle/ldmatrix math is byte-for-byte identical). A `dequant` flag (on either kernel) emits the
  **fused per-channel dequant epilogue** (`…_smdb_deq`/`…_smdb_swz_deq`): `out = f32(Σ u8·i8)·scale[j]`
  folded into the C store — the round-trip cuBLAS int8 (raw `i32` out) structurally can't fuse. Integer
  accumulate is exact mod 2³² ⇒ the gate is **bit-exact** (`==`), stronger than the float tolerance gate.
  Launchers `gemm_nt_int8`/`_smdb`/`_smdb_dequant`/`_splitk` + `int8_tile` in `gpu.rs` (the `_smdb`/
  `_dequant` dispatch prefers swz when K%64==0, falls back to the BK=32 hand-placed kernel otherwise —
  both bit-exact, choice is purely throughput); peers (naive + `dp4a` CUDA-C via NVRTC, cuBLAS int8 IMMA
  `cublasGemmEx`) in `baselines.rs`; `int8_gemm_vs_peers` scoreboard. Standing (same-run, RTX 4050):
  **180–237× vs naive**, **33–58× vs dp4a**; the hand-placed path was **~44/53/52% of cuBLAS** at
  1024/2048/4096, and **`ldmatrix`+swizzle is a further ~1.2–1.5× internal win** (`int8_swz_vs_handplaced`
  same-run A/B, 12/12 across two runs — the lever was conflict-free SMEM, *not* multi-stage depth, which
  is occupancy-bound). For **thin-M / small-N decode** (M,N grid under-fills the SMs), `gen_int8_smdb_swz`
  has a **split-K** mode (`int8_gemm_nt_smdb_swz_sk`, `gridDim.z=sk`, partials folded by
  `red.global.add.u32` — order-independent ⇒ bit-exact *and* deterministic, a float split-K can't be):
  up to **~2.5×** vs un-split when starved (M64 N128 K8192, 2 CTAs → sk=8), marginal once saturated
  (`int8_splitk_occupancy`). **Static-shape (M1, the int4 twin):** `gen_int8_smdb_swz(static_dims=…)` bakes
  M/N/K as constants (ptxas constant-folds the strides + knows the K trip count) → `int8_gemm_smdb_swz_static_ptx`
  / launcher `gemm_nt_int8_static`; **bit-exact** vs the dynamic kernel (`int8_static_matches_reference`,
  64 & 128 tiles), same-run **1.02× square / up to 1.34–1.37× thin-M decode** (`int8_static_vs_dynamic_ab`,
  both kernels raw-loaded — the int4 confound: never compare a raw-load vs a cubin-cached kernel). cuBLAS int8 is s8×s8
  only, so the peer cross-check uses `[0,127]` activations (where u8≡s8); the full-range `[0,255]` gate is
  separate. Owned by the int8 slice (`gpu-int8-gemm`).
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
- `src/autotune.rs` — **Phase 10 per-(op,shape,dtype) autotuner**. **int8 GEMM**: searches `smdb`/`swz` ×
  {64,64}/{128,128} + split-K `swz64` sk∈{2,4,8}, **bit-exact cross-checks** every candidate against the
  first before timing (a disagreement panics — never caches a wrong winner). **W4A16 (int4 decode)**:
  searches split counts sk∈{1,2,4,8} (the un-split kernel vs `gridDim.z=sk` GEMM + reduction, both timed),
  **tolerance** cross-check (fp16 accumulate, not bit-exact) — auto-selects the up-to-6.4× decode split-K.
  Ranks by best-of-N same-run time (the same-family ratio cancels the shared clock → honest under
  contention). `AutotuneCache` is a tiny hand-rolled text file (no serde); `tune_{int8,w4a16}_cached` =
  lookup-or-tune, `launch_{int8,w4a16}_tuned` = run the winner, `revalidate_int8` = regression mode (flag
  a different config now >10% faster; pure unit-tested decision fn). On-device it discriminates
  (int8 64×128×8192→swz64_sk8, 128×128×256→smdb64; w4a16 64×256×1024→w4a16_sk4). The home for sk/tile selection.

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
