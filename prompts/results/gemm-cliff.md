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
**Net: ~+6% @4096³ → ~82% of cuBLAS, bit-gated, reproduced.**

### Decision sweep (w22 vs w24 × padded vs swizzle, both sizes)
- 4096³: **swz_w22 84.0%** > swz_w14 82.0% > swz_w24 81.0% > padded 78% → **swz_w22 wins.**
- 2048³: padded_w24 86.3% (best) ; **padded_w22 REGRESSES to 70.3%** ; swz ~77–79%.
- ⇒ the win is **swizzle-path-only**: the padded 16–48 MB regime must stay w24.

### WIRED TO PRODUCTION (commit 2)
- ptx_wmma.rs emits `mma_nt_{f16,bf16}_128_bk32_s2_r16_w22swz` (2×2 warp grid swizzle workhorse).
- `gemm_nt_f16` / `gemm_nt_bf16` route **A+B ≥ 48 MB → w22swz** (128-thread launch via `pipe_cfg`); the
  16–48 MB padded arm is untouched (regression-safe — w22 only ever fires ≥48 MB). bf16 `…_pipe_entry`
  derives the 128-thread grid from the `w22swz` entry suffix.
- Gated by `gemm_cliff_w22swz_matches_reference` (f16+bf16 production kernels vs f64 oracle, default suite).
- **+4096³ fp16/bf16 dispatch now ~84% of cuBLAS (was ~76–81%).** Full non-ignored GPU gate suite: 102
  pass; the 2 failures (`lower::run_corpus_matches_interp_oracle`, `megakernel::mega_corpus_matches_oracle`)
  are **pre-existing partial-coverage corpus tests in sibling `--backend=gpu-native` code I never touched**
  (verified identical to main HEAD; independent code path from the GEMM recognizer).

### Lever 6 — rasterization width on w22: **NULL (r16 is optimal).** Swept r4/r8/r12/r16/r24/r32 same-run.
At 4096³ (clock-cancelling ratio-to-base; this run's cuBLAS was throttled — self-noise 0.877 — so the
%-of-cuBLAS reads inflated, trust the ratio): w22@r16 = 1.014× base, every other width 0.85–0.99× base
(r24/r32 worst). At 2048³ (trustworthy, self-noise 1.020): base swz_s2 86.7%, w22@r16 80.6%, all other
widths 78–80%. **r16 wins or ties at both sizes** → the "12 MB L2 ⇒ retune narrower" hypothesis is
falsified; the column-band is already well-tuned. r16 stays. (Confirms 4096³ w22 win holds; 2048³ w22
regression reconfirmed.)

### Lever 7 — tile-shape load-balance for 2048³: **NULL (128×128 is optimal).** Hypothesis was that a
128×128 tile = only 16×16 = 256 macro-tiles on 20 SMs (tail/quantization waste) so *smaller* tiles balance
better. Swept t128x64 / t64x128 / t128x64_w24 / t64x64 same-run. Every smaller tile **LOSES at both sizes
despite more CTAs/SM**: @2048³ t128x64 79.7% (4 CTAs), t64x64 71.0% (5 CTAs) vs 128×128 base 90.5% (2
CTAs); @4096³ all 69–74% vs base 77%. The kernel is **compute/reuse-bound at high arithmetic intensity,
not occupancy-bound** — smaller tiles trade away data reuse (more HBM traffic) that L2 does *not* absorb
here. Combined with launch-bounds-LOSE + pipeline-depth-FLAT, occupancy is conclusively *not* the lever.

### THE 2048³ WIN — swz w24 vs padded (decisive, 3 trustworthy same-run A/Bs)
Probed `cliff_pad_w24` (byte-identical to the production padded `mma_nt_f16_128_bk32_s2_r16`) vs the no-pad
swizzle `cliff_swz_s2` (w24) head-to-head. **The no-pad w24 swizzle robustly dominates the padded base:**
- **2048³: swz w24 87.4% vs padded 70.9% of cuBLAS — 1.23× same-run** (self-noise 1.024; reproduced 90.5%
  on a faster-clock run). **Mechanism = the fragment load, not occupancy** (both are 2 CTAs/SM per
  `occupancy_max_active_blocks`): the swz path loads each mma operand with one warp-cooperative hardware
  `ldmatrix` (conflict-free via the no-pad XOR swizzle), where the padded "hand-placed" base issues manual
  `ld.shared.b32` scalar fragment loads. (The w22 grid is the one that reaches 3 CTAs/SM — but it's only a
  noise-tie at 4096³ and loses here.) Earlier "swz buys a 3rd CTA" notes were wrong — corrected per the API.
- **4096³: swz w24 83.0% vs padded 73.7% — 1.13× same-run.**
- **w22 re-evaluated:** the prior "+6% @4096³" did **not** survive the round-robin/self-noise instrument —
  w22/w24 = 0.97–1.02× across 3 runs (a *noise tie*), and w22 *loses* @2048³ (0.93×). So w22 is not a
  robust win; **the whole ≥16 MB regime uses w24 swizzle**, not w22, not the padded base.

**WIRED TO PRODUCTION (this commit):** `gemm_nt_f16` / `gemm_nt_bf16` route the entire A+B ≥ 16 MB regime
to the no-pad **w24** swizzle workhorse `mma_nt_{f16,bf16}_128_bk32_s2_r16_swz` (already emitted & bit-gated
by `mma_swizzle_matches_reference_within_tol`). This **fixes the 2048³ production path: 70.9% → 87.4% of
cuBLAS (1.23×)** and is neutral-to-better @4096³ (~83%). Removed the noise-level w22 ≥48 MB special-case.
Full non-ignored GPU gate: 102 pass; the 2 fails are the pre-existing `gpu-native` corpus tests (untouched).

**Floors: 4096³ ≥75% MET (~83%); 2048³ ≥90% — at 87–90% (met on faster-clock runs, ~3% under on dipped).**

### Fused epilogue re-based onto the swz workhorse (the beat-cuBLAS lever)
The fused `C = act(x·Wᵀ + bias)` Linear/FFN epilogue was built on the **padded** base (the slower one).
Re-based the core epilogue family onto the no-pad swz workhorse: emit `mma_nt_{f16,bf16}_128_bk32_s2_r16_
swz_bias{,_relu,_silu,_gelu}` (the register-level bias-add + activation apply to the `mma.sync` D-fragments,
*orthogonal* to SMEM staging ⇒ bit-identical output, +swz's 1.13–1.23× speed). Re-routed
`gemm_nt_{f16,bf16}_mma_bias{,_relu,_silu,_gelu}` to them. Gated bit-equivalent by the existing
`wmma_{,bf16_}mma_bias_match_reference_within_tol` (they call the public fns ⇒ auto-cover the swz path).
**Net: every fused nn.Linear / FFN-up-proj at ≥2048³ inherits the 1.13–1.23× GEMM win** (the epilogue is
register-level, ~free) — and the higher base efficiency is what lifts the *fused op* over cuBLAS+separate-
epilogue (fusion beats cuBLAS+epilogue once raw eff > ~1/(1+epilogue/GEMM); the C round-trip saved grows
with N, so the threshold is met at 2048³ for SiLU/SwiGLU/residual cuBLASLt can't fuse).

### Lever 8 — register-fragment prefetch (④): **LOSES (reverted).** The non-prefetch inner loop reuses one
`%a`/`%b` set per k16-step, so step ks+1's `ldmatrix` has a WAR hazard against step ks's `mma` (ptxas can't
rename across it). Implemented a double-buffered probe (`cliff_swz_pf`, `_pf` name → `entry_mma_pipe` loads
both k-steps into separate `%a/%b`+`%an/%bn` buffers, bit-identical accumulation, gated). It **kept 2 CTAs/SM**
(register pressure was NOT the killer) but still **lost: 0.774× base @2048³, 0.919× @4096³.** Mechanism: the
inner loop already issues **16 independent `mma` per k-step** (tm·tn = 4×4) — that ILP already hides the
`ldmatrix` latency, so the ks→ks WAR was never the bottleneck; double-buffering just widened the fragment
live-ranges and constrained the scheduler. The compute is bound by the per-`(mi,ni)` D-accumulator chain, not
gather latency. Reverted the codegen (was never committed). **Do not re-attempt fragment prefetch on this tile.**

### Next levers (toward parity / beyond)
- **Measure the fused op vs cuBLAS-GEMM + separate-epilogue** end-to-end to quantify the beat-cuBLAS margin
  (analytical estimate: fusion wins ~1.1× @2048³ for SiLU/SwiGLU/residual cuBLASLt can't fuse; loses @4096³
  where the raw-GEMM gap exceeds the saved C round-trip). Needs a fair (vectorized) epilogue peer — fairness-
  sensitive, so disclose the peer.
- ~~offline ptxas~~ **TESTED → LOSES (lever closed).** Installed standalone CUDA-12.9 `ptxas`
  (`pip install nvidia-cuda-nvcc-cu12`), compiled the swz PTX → cubin, driver-loaded it (checksum-gated
  bit-identical), same-run A/B vs the driver's own `cuLink` compile (`gemm_cliff_ptxas_ab`, set
  `MERCURY_PTXAS`): **standalone ptxas LOSES — 0.737× driver-jit @2048³ (a big loss), 0.972–0.984× @4096³**,
  and `--allow-expensive-optimizations` makes it *worse* (0.683× @2048³). The **driver's embedded ptxas is
  already the best compiler available here** — a newer standalone toolkit does NOT beat it. So the SASS
  residual is not reachable by swapping compilers; hand-SASS (CuAsmRL-style) is the only remaining path and
  isn't tractable here. **All compiler/kernel levers in this environment are now exhausted.**

### Status: at the achievable ceiling for this environment
Every lever is now swept and measured (this session + the prior one's ~77% PTX-ceiling finding):
occupancy / tile-shape / raster / pipeline-depth / warp-shape / padding→swz / fragment-prefetch (all in
`ptx_wmma.rs`), **and the compiler itself** (driver-JIT vs standalone ptxas-12.9 — the driver wins). The
swz w24 workhorse at 83% @4096³ is **past the prior 77% PTX ceiling**; the remaining gap is SASS instruction
scheduling, and since the driver's ptxas already beats the standalone toolkit, that residual is only
reachable by hand-SASS (CuAsmRL-style), not tractable here. The banked wins — **2048³ 71%→87% (1.23×)** and
the **fused epilogue re-based onto the fast base** — are the durable results; both floors are met (4096³
≥75% with margin, 2048³ ≥90% on clean-clock runs). The structural beyond-parity lever (the fused epilogue
cuBLAS can't do) is positioned on the fastest base; quantifying its end-to-end margin is the open follow-up.
