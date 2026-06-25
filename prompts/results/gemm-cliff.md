# GEMM cliff — closing the large fp16/bf16 GEMM gap to cuBLAS (RTX 4050, sm_89)

Branch `perf/gpu-gemm-cliff-2` (worktree `Mercury-gemm2`). Goal: ≥90% of cuBLAS @2048³, ≥75% @4096³
(floors, not targets — then push past parity via the fused epilogue cuBLAS can't do).

## Hardware / constraints
- RTX 4050 Laptop, sm_89 Ada: 20 SMs, 6 GB, ~24 MB L2, 192 GB/s HBM, SM boost 3105 MHz (idles 405).
- **Clock locking DENIED** (no admin/persistence on this mobile part) → clocks swing 405→3105 MHz under
  load. *Only tight, same-inner-loop A/B ratios are trustworthy.* Absolute GFLOP/s is meaningless.
- No offline `ptxas` installed (only cuBLAS/nvrtc/nvjitlink redist DLLs). Driver JIT compiles the PTX.
  `pip install nvidia-cuda-nvcc-cu12` would add ptxas — a reserve lever.

## Baseline (fresh same-run, `gemm_pipe_sweep`, 2026-06-25)
4096³ cuBLAS samples were tightly clustered (17.5–18.3 TFLOP/s ⇒ clock stable ⇒ trustworthy):
- **4096³ mma_nt_f16_128_bk32_s2_r16 (padded workhorse) = 73.3% of cuBLAS.** ← primary target
- 2048³ this run was clock-dipped (cuBLAS swung 7.8–14.5 TFLOP/s; workhorse showed an impossible
  5.0 TFLOP/s < its own 4096³ 13.4) → **2048³ must be re-measured clean**. Doc claims ~90–97%.
- 1024³ pipe_64_s6 ~94% (cuBLAS sample throttled → optimistic; parity-preserve only).
- swz (`_swz`, no-pad ldmatrix) path measured separately by `mma_swizzle_vs_handplaced`.

## Binding-constraint hypothesis (4096³)
Memory + doc say the cliff is **latency-bound, not bandwidth-volume-bound**. The workhorse is a 128×128
tile, BK=32, **2-stage** cp.async, 256 threads, **64 f32 accumulators/thread** (very high reg pressure).
- The PTX emits **no `.maxntid`/`.minnctapersm`/`.maxnreg`** → the driver JIT picks registers/occupancy
  blind. With 64 accumulators + frags + scratch the JIT likely allocates ~100 regs/thread ⇒ **only 2
  CTAs/SM = 16 warps = 33% occupancy**. The doc *assumes* the no-pad swz tile gets 3 CTAs/SM from its
  32 KiB SMEM, but never verified/forced it — if it's register-limited it is silently still at 2.
- 2 CTAs × 2-stage pipeline cannot keep enough HBM loads in flight to hide ~500-cycle latency ⇒ the
  cliff. The fix is **more in-flight work**: higher occupancy (force CTAs/SM via launch bounds, cut
  register footprint) and/or deeper pipeline (s3/s4), within the SMEM budget.

## Lever priority (untried axes; all measured via tight A/B)
1. **Launch bounds** `.maxntid`+`.minnctapersm` — force the JIT to fit N CTAs/SM (cap registers). Cheap
   (one directive). Verify achieved CTAs/SM via `occupancy_max_active_blocks_per_multiprocessor`.
2. **Deeper pipeline s2→s3** on the no-pad swz tile (48 KiB static, fits the ≤48 KiB cap exactly).
   More cp.async in flight; 2 CTAs/SM. Trade vs s2's 3 CTAs.
3. **Dynamic SMEM > 48 KiB** (Ada allows ≤100 KiB) → s3/s4 at higher occupancy, or bigger tiles. Needs
   `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_MEMORY_SIZE)`.
4. **Smaller tile (128×64 / 64×128)** → 32 accum/thread → more CTAs/SM (re-test under launch bounds;
   prior sweep rejected it but under the 48 KiB-static / no-launch-bounds constraint).
5. **Offline ptxas** `-O3 --allow-expensive-optimizations --maxrregcount` (install nvcc redist).
6. **Rasterization width** re-tune for 24 MB L2 (already at r16; lower priority).

## Measurement instrument
`gemm_cliff_ab` (new, gpu.rs append-only): for 2048³ & 4096³, one inner loop times {each candidate,
the padded-workhorse baseline, cuBLAS} back-to-back × N rounds, best_of each → reports candidate/cuBLAS,
baseline/cuBLAS, candidate/baseline (clock-cancelling), + achieved CTAs/SM per kernel. Bit-gated:
checksum vs cuBLAS each size; the f64-tolerance gate stays in the existing `*_matches_reference` tests.

## Results log

### Instrument (`gemm_cliff_ab`, gpu.rs) + gate (`gemm_cliff_matches_reference`)
Round-robin best-of-N, **ratio-of-best** (unbiased for identical kernels, unlike min-of-ratio which
amplifies anti-correlated noise). cuBLAS self-noise sentinel confirms the floor: **1.001–1.006×** on
good runs (sub-1%), up to ~0.95 when the clock wanders. Reports %-of-cuBLAS, ×-vs-base, achieved CTAs/SM
(`occupancy_max_active_blocks_per_multiprocessor`). Every candidate checksum-gated vs cuBLAS; all
`CLIFF_VARIANTS` f64-tolerance-gated. The first metric (per-round min-of-ratio) was **broken** —
base-vs-base read 1.04–1.26× — fixed to ratio-of-best.

### Clean baseline (production paths)
- **4096³ swz_s2 = 76.6% of cuBLAS** (the dispatched ≥48 MB path). Padded base = 75%. (The "73%"/"34%"
  in the prompt were the padded path / stale.) The ≥75% floor was *already* met by the swizzle dispatch.
- **2048³**: padded base ~84–87%, swz ~75–85% (regime is padded; noisier to measure).

### Lever 1 — launch bounds (`.minnctapersm`): **LOSES.** Forcing 3 CTAs on the w24 tile (lb3) → 66–72%;
forcing 4 (lb4) → catastrophic 38–45% (register spill). Confirms the research: big-tile GEMM wants LOW
occupancy + deep pipeline, not forced occupancy. Infrastructure kept (`min_ctas` param on `entry_mma_pipe`).

### Lever 2 — pipeline depth s2→s3: **~FLAT/LOSES.** s3 alone 71.7–78% (≈ or < s2); the 48 KiB s3 tile
drops a CTA vs s2's, trading away occupancy. Not the lever on this part.

### Lever 3 — WARP-TILE SHAPE (the research #1 lever): **WINS at 4096³.** On the 128×128 swizzle tile,
shrinking the warp grid 2×4→**2×2 (w22)** or **1×4 (w14)** gives each warp a 64×64 / 128×32 tile = **32
mma/warp** (2× the ILP) AND 128 threads/CTA ⇒ **3 CTAs/SM** (vs 2). Measured same-run (2 runs):
- **w22: 76.6%→81.7–84.3%** (1.067–1.093× base) · **w14: →82.2%** (1.073×) — both 3 CTAs/SM.
- w41 (4×1) / w42 (4×2) do NOT win (warp shape matters — wn≥2 with bigger tm wins).
- Bigger *threadblock* tiles (256×128, 128×256) LOSE (1 CTA/SM). w22+s3 loses (back to 2 CTAs).
**Net: ~+6% @4096³ → ~82% of cuBLAS, bit-gated, reproduced.** Next: register-fragment prefetch (lever ④)
stacked on w22/w14, then wire the winner into the ≥48 MB dispatch.
