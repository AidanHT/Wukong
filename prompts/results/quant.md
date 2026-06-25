# Quantized-GEMM gap vs cuBLAS IMMA / Transformer Engine (int8 + fp8)

Branch `perf/gpu-quant-2` (worktree `../Mercury-quant2`). Target: mobile RTX 4050, sm_89 (Ada).

## Mission
- int8 (W8A8, u8×i8→i32) tensor-core GEMM: **≥75% of cuBLAS IMMA** (a floor — push to parity), bit-exact.
- fused per-channel **dequant epilogue**: beat the cuBLAS GEMM+dequant *chain* outright (the lever a library can't use).
- fp8 (E4M3/E5M2): honestly characterized vs best buildable peer (cuBLASLt fp8 / Transformer-Engine-class).
- Bit-exact gate for int8 (i32 mod 2³², `==`), tolerance gate for fp8. Gate BEFORE every speed number.
- Same-run, %-of-cuBLAS + ×vs naive/dp4a. ≥3 re-runs or it's a hypothesis. No co-author trailers.

## Starting code (main @ 4d71523)
- int8 default = `gen_int8_smdb_swz` (ldmatrix.x4/.x2 from XOR-swizzled SMEM, **BK=64, 2-stage double-buffer**),
  64×64 (2×2 warps) + 128×128 (4×2 warps) tiles. Plus hand-placed `gen_int8_smdb` (2-buf) and `_ms` (3/4-stage,
  hand-placed `ld.shared`, NOT swizzled). Split-K (`red.global.add.u32`), static-shape, fused dequant all exist.
- Reported standing (per docs, possibly stale): hand-placed `_smdb` ~44/53/52% of cuBLAS @1024/2048/4096;
  swz a further ~1.2–1.5× internal. **swz %-of-cuBLAS not directly headlined → MEASURE FIRST.**

## Key structural observations (pre-measurement)
1. **swz is capped at 2 stages by `assert(2·(bm·bk)+2·(bn·bk) ≤ 48 KiB)` (static SMEM).** 128×128 BK=64 2-buf = 32 KiB.
   cuBLAS/CUTLASS use **opt-in dynamic SMEM up to ~100 KB/SM** for 3–6 stage pipelines. → untried lever:
   dynamic SMEM + multi-stage swizzle (the prior multi-stage attempt was hand-placed + static-SMEM-bound).
2. Per-K-step there are **two `bar.sync`** (post-wait visibility + pre-overwrite fence). A deeper ring can drop
   the second barrier dependency.
3. Epilogue stores i32 one-at-a-time (`st.global.u32`), no vectorized `st.global.v2/v4`. Minor.
4. `_mt`/single paths are the "fast int8" reported in `int8_gemm_vs_peers`, but swz is the actual best — the
   headline test under-reports. Add a `quant_*` test that reports **swz %-of-cuBLAS** directly.

## Levers to evaluate (measure binding constraint first)
- [ ] **L0 BASELINE**: run `int8_smdb_sweep` (swz %-of-cuBLAS) + `int8_swz_vs_handplaced` ×3. Establish truth.
- [ ] **L1 occupancy probe**: regs/thread, SMEM/block, CTAs/SM for swz64/swz128 (from cubin / ptxas). Bound by what?
- [ ] **L2 multi-stage swizzle + dynamic SMEM**: 3–4 stage ring on the ldmatrix+swizzle base (the untried combo).
- [ ] **L3 tile/warp sweep**: 128×256, 256×128, BK=128; warp-tiling variants; pick per-regime.
- [ ] **L4 epilogue**: vectorized i32 stores, register trim.
- [ ] **L5 split-K / stream-K**: thin-M/small-N decode regime (the real quantized-inference shapes).
- [ ] **L6 fp8**: tile/stage sweep on fp8_pipe; vs cuBLASLt fp8 / TE-class peer; E5M2.
- [ ] **L7 fused dequant headline**: GEMM+dequant vs cuBLAS GEMM+dequant chain (Mercury wins by construction).
- [ ] **L8 autotuner**: extend candidate set, keep bit-exact cross-check.

## Measurement protocol
- Release build, `--features gpu`, `--ignored --nocapture`. PATH += main checkout's `tools/cuda-redist` DLL dirs.
- Clock warmup (hammer cuBLAS) before sampling. Interleave variant vs cuBLAS round-by-round (clock cancels).
- Bit-exact checksum cross-check at each shape before timing.

## Results log

### L0 BASELINE (main @4d71523, `int8_smdb_sweep`, RTX 4050, same-run, cuBLAS-interleaved)
Real swz %-of-cuBLAS (the prompt's "44%/53%" were the *hand-placed* `_smdb`, stale):
| size  | best variant | %-of-cuBLAS | vs 75% floor |
|-------|--------------|-------------|--------------|
| 1024³ | smdb64_swz   | **88.1%**   | ✓ above |
| 2048³ | smdb128_swz  | **79.6%**   | ✓ above |
| 4096³ | smdb128_swz  | **62.1%**   | ✗ **BELOW — the gap** |
- Multi-stage on the *hand-placed* path: helps 128-tile @2048³ (s2 69.9%→s3 73.1%) but HURTS @4096³
  (s2 52.4%→s3 46.2%→s4 15.7% collapse) — deeper stages starve occupancy when HBM-bound. Never combined w/ swz.
- swz64 collapses @4096³ (37.9%) — tile too small, no reuse; swz128 holds (62.1%). 4096³ is HBM/L2-bound.
- **Diagnosis: 4096³ is HBM-bound. The int8 swz kernel has NO threadblock rasterization** (plain ctaid.x/y).
  fp16 mma path proved raster takes 4096³ 56%→72% (L2 locality). → port raster to int8 swz = lever #1.
- 2048³ latency-bound: 3-stage swz (128×128 BK=64 = 48 KiB, fits static) = lever #2.

### L3 RESULT — warp-tile lever (the big one). `quant_int8_bigtile_sweep`, same-run interleaved.
The lever is the **warp tile, NOT the CTA tile**. 128×128 CTA with **w2×2 (4 warps ⇒ 64×64 warp tile)**
crushes the shipped w4×2 (8 warps, 32×64 warp tile); bigger CTA tiles (256×128/128×256) LOSE (20 SMs
can't fill them, 1 CTA/SM).
| config | 1024³ | 2048³ | 4096³ |
|---|---|---|---|
| 128×128 w4×2 (shipped, 32×64 warp) | 69.3% | 79.1% | 75.0% |
| **128×128 w2×2 (64×64 warp)** | **84.3%** | **91.7%** | **84.2%** |
| 128×128 w2×2 + raster=8 | 84.3% | **99.6%** | 81.7% |
| 256×128 / 128×256 (bigger CTA) | ~62% | ~67% | ~66% |
- Internal clock-robust ratio w2×2 / w4×2: **1.21× @1024³, 1.30× @2048³, 1.24× @4096³** (cuBLAS swung
  66932–75388 @4096³ same run → trust the internal ratio over the %).
- **Baseline 88/79.6/62% → 84/99.6/84%.** 2048³ near parity; 4096³ +22 pts (floor cleared at all sizes).
  raster=8 helps ONLY @2048³ (L2-edge regime); neutral @1024³, slightly hurts @4096³ → per-size.
### L3 CONFIRMED + SHIPPED — `quant_int8_w64_confirm` ×3 (per-size winner stable across 3 runs):
| size  | winner | run1 | run2 | run3 | vs baseline |
|-------|--------|------|------|------|-------------|
| 1024³ | **64×64 swz** (smdb64_swz) | 89.1% | 86.7% | 85.7% | ~88% (unchanged — already good) |
| 2048³ | **w64** (±raster8) | 127.5%* | 95.2% | 96.5% | **79.6% → ~96% (near parity)** |
| 4096³ | **w64** (±raster8) | 80.8% | 84.1% | 85.6% | **62.1% → ~84% (+22 pts)** |
*127.5% = a cuBLAS contention outlier; true ~95-99%. ALL sizes well above the 75% floor; 2048³ ≈ parity.
- **SHIPPED**: `gemm_nt_int8_smdb` dispatch now: 64×64 swz (<2048²) / **w64** (≥2048², 128-div) / hand-placed
  fallbacks. Autotuner gained `w64` + `w64_r8` candidates (bit-exact cross-checked, ranks 8/shape; picks
  smdb64 for small square, swz64_sk8 for thin decode, w64 for large). raster8 is an autotuner-only refinement
  (regime-narrow ~2048², noisy → not in the static path). bm≠bn double-buffer fix made 256×128/128×256
  buildable (they lose on 20 SMs, kept only as autotuner/experimental candidates).
- COMMIT 2: w64 ship + dispatch + autotuner.
- Next: (a) push 1024³/4096³ toward parity — multistage + dynamic-SMEM (untried across the whole backend);
  (b) fused dequant headline (now lands — GEMM near parity); (c) fp8 characterization; (d) split-K/Stream-K decode.

## Research findings (parallel agents, verified)
**Ada int8 binding constraint (RTX 4050, sm_89):** at 1024³–4096³ a well-tiled int8 GEMM is
**on-chip-datapath / tensor-core-issue bound, NOT HBM-bound** (int8 roofline knee ~374 OP/byte; square
GEMM AI≈(2/3)·n ⇒ all sizes far past the memory wall; DRAM ~97% idle). Ledger from a directly-comparable
Ada study (spatters.ca, RTX 4090): swizzled-SMEM+vec-loads 37%→83% (have it), **register/warp-tile growth
+ bigger CTA tiles 89.5%→100% (THE remaining lever)**, multistage cp.async only **+3%**.
- **#1 lever — bigger tiles.** CUTLASS int8 winners on Ada: **256×128×64 / 128×256×64, 8 warps, warp tile
  64×64**, 128×128×64 (5-stage) mid-fallback. Mercury's 128×128 has a **32×64** warp tile (wm=4,wn=2),
  64×64 has 32×32 — under-reuse. Fix: 256×128 / 128×256 with wm=4,wn=2 ⇒ 64×64 warp tile. Generator
  already supports arbitrary bm/bn/wm/wn. 256×128 BK=64 2-stage = exactly 48 KiB (fits static).
- **Multistage** secondary (~3%); needs **dynamic-SMEM opt-in** (`cudaFuncAttributeMaxDynamicSharedMemorySize`,
  ≤101376 B) for 3+ stages on big tiles (256×128×64 = 24 KB/stage ⇒ 3-stage=72 KB). `cp.async.wait_group(S-1)`.
- cuBLAS COL32/COL32_2R_4R4 interleave edge = only **~19%**, not 2×; do last. Epilogue vec stores ~0 for
  square (skip), matter only on thin-M decode. Ada budget: 64K regs/SM, 255/thread, 48 warps/SM, 100 KB
  carveout, 99 KB max dynamic/block, 48 KB static. RTX 4050 = 20 SMs / 80 TC, L2 ~16 MB, 192 GB/s.
- **Stream-K**: ~ties tuned split-K on square (1.196× only vs naive data-parallel) → a **decode-regime**
  lever, not the square-gap lever. i32 accumulate ⇒ unordered-atomic reduction is bit-exact (edge vs
  cuBLAS's float turnstile). Keep square data-parallel + raster; Stream-K later for thin-M.

## Commit log
(filled as committed)
