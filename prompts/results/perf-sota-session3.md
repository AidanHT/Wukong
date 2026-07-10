# perf-sota campaign — session 3 (2026-07-09/10) measurement ledger

Branch `campaign/perf-sota-2026-07`. Ten-target directive; every number below is from the release
binary through the real pipeline, same-run/adjacent instruments only, thermal protocol observed
(model bench cool-first, roofline-column validity 128–140 valid this session; first run after a
rebuild discarded as warm-up). Baselines re-proven at session start (T0) before any optimization.

## What shipped (commits on the campaign branch)

- Six wave-1 branches merged: `perf/velem-parallel`, `perf/gemm-2d-shared`, `perf/vmath-exp-ldexp`
  (later reverted — see T9), `perf/flash-ws`, `perf/gpu-gemm-4096`, `perf/serving-goodput`.
- `168070c` size-keyed 2D GEMM dispatch (per-block ≥2²⁶ MACs, shared-pack small band, PAR_MIN 2²³,
  task budget 1 Mi).
- `5979a19` revert of the exp ldexp restructure (measured loss both thermal states).
- `a3b3c0e` GPU dispatch wins: v2cs epilogue ≥48 MB GEMM; warp-specialized flash S≥4096 default-on.
- `d017b4a` C-tile prefetch in the GEMM microkernel prologue.

## Per-target honest standings

### T2/T4/T5 — CPU parallel GEMM (the headline CPU work)
- **Shared-pack cooperative packing REFUTED at mid/large** (the session's biggest surprise):
  per-block packing wins BOTH ABBA orderings — 1024³ ~462 vs ~385 GF/s, 512³ ~315 vs ~278, skinny
  FFN shapes up to 1.75× (128×3072×768). Mechanism: per-worker packing warms each worker's own L2
  with exactly the panels it consumes; the shared pack hands consumers panels another core packed
  (through L3) plus a fork-join barrier per K-block. Shared-pack WINS only the small band just
  above the parallel gate (256³: ~209 vs ~185 GF/s with a 1 Mi-MAC task budget; 4 Mi read 146,
  16 Mi read 90). Shipped as the size-keyed dispatch (`SHARED_MAX_MACS = 2²⁶`).
- **vs oneMKL-all (same-run, MKL-anchored):** 512³ 70–74%, 1024³ 75–83% (from the 66–69%
  baseline); 2048³ 88–93% vs a HEALTHY all-threads peer (~449 GF/s) — the same morning read
  176–186% against a degraded-MKL round (218 GF/s); both are real same-run ratios, the range and
  the peer's bimodality are disclosed. 256³ engaged: **209 GF/s = 102–123% of adjacent MKL-all
  readings** (was 39–52% deliberately-serial). Single-core: 97–100% of MKL-1c ≤1024³.
- New-vs-old-binary policy isolation (MKL-anchored): no regression at 512–2048³; skinny-M
  single-block-row policy +16–25% at 128×768×768 / 128×3072×768.

### T7 — large-GEMM single-core tail: **GOAL MET**
C-tile prefetch (first+last line of each of the mr C rows, issued before the K loop; C rows are
ldc apart — no hardware prefetcher tracks that stride). MKL(1c)-anchored ABBA, 4 rounds:
**2048³ ON 96–99% of MKL-1c vs OFF 86–88%** (~+10pp, both orderings; OFF reproduces the recorded
83–86% baseline, pinning the instrument). ≤1024³ a wash (C is cache-resident there). Hint-only —
bits unchanged on every path.

### T1/T3 — model @parallel and vs PyTorch
Three valid rounds (roofline 130–133; interp gate bit-exact; serial==@parallel bit-exact;
all cross-checks ≤2e-6):
- @parallel scaling: **S=128 2.19× → 2.34× → 2.96×** across the session's fixes (velem-parallel
  dispatch + skinny-M policy + per-block default + gate retune); S=512 2.39–2.46×. Goal ~6× NOT
  met — remaining serial term is the attention head loop (wave-2 lever in flight: head-loop
  privatization).
- vs all-threads torch: S=128 **1.08–1.20× behind** (was 1.5–1.9×); S=512 **1.01–1.63× behind** —
  the wide range is the PEER's power-state swing (torch-Tn 443→271 ms across rounds while Mercury
  @parallel held ~440 ms in every round). Key structural finding: **Mercury's parallel path does
  not ride power-state upside** — peers gain ~1.6× from the strong evening state, Mercury ~0 —
  consistent with a sync/serial-fraction bound, not a clock bound. NOT flipped to a win; honest.
- Single-core vs torch-1T @S=512: 1.10–1.16× FASTER (held all rounds).

### T9 — exp vs VML: ldexp lever REFUTED, standing honestly bounded
The wave-1 ldexp restructure (bit-identical by exhaustive 2³² sweep) measured a consistent
**~10–15% throughput LOSS in BOTH thermal states** (interleaved old/new binary ABBA: warm
1.43/1.33/1.23/1.43× slower-than-VML N/O/O/N; hot 1.47/1.38/1.25/1.68×; tanh — which composes
exp8 — confirmed independently at 2.69× vs 2.90× faster-than-VML). The port-rebalance premise
fails physics: throttling scales the clock, which slows p5 exactly as much as p0/p1; the tail's
~+3 net uops lose at any clock. REVERTED (`5979a19`). exp stands at **1.23–1.45× slower than VML
across thermal states** (T0's recorded 1.12–1.18× was a cooler session; the recorded "1.05×
faster cool" from 2026-07-08 did not reproduce at all). ×6-unroll knob: wash. tanh 2.7–2.9×
faster than VML retained. Residual gap is algorithmic (VML's cheaper core); further poly-degree
work is the only remaining lever and is out of this campaign's scope.

### T6 — GPU attention long-S: warp specialization measured, honestly bounded
`flash_ws_vs_mp` (clock-cancelled, median-of-9): ws/ws3 **win only at S=4096, by 4–6%**
(d64 ws 0.944×, d128 ws3 0.950× of mp), tie at S=2048, **lose 10–52% at S≤1024**. The
provisional route (d64 S≥2048 / d128 S≥1024) would have shipped a d128@1024 regression and was
corrected before default-on. Shipped: S≥4096 only, per-d winners, `MERCURY_FLASH_WS=0`
kill-switch. ~1× cuDNN at long S is NOT reachable via this lever on this 20-SM part — the
SFU/serial-softmax structural diagnosis stands.

### T10 — GPU 4096³ GEMM: +2.7% shipped, honest peer landed
Cliff sweep (round-robin best-of-10): only the **v2cs epilogue** (paired-column
`st.global.cs.v2.f32`) beats the swz base at 4096³ — 1.027×, **76.8% of cuBLAS-f16 / 80.4% of
the honest f32-out peer** (the f16-out peer hides ~half of Mercury's f32 C-write traffic; the
new `time_cublas_gemm_nt_f16_f32out` column makes the asymmetry visible). At 2048³ every
candidate LOSES to base (v2cs 0.75× — C is L2-scale there), so the arm keys on A+B ≥ 48 MB.
3-stage pipe, raster re-tunes, launch-bounds: all measured losses, retained bench-only. ~90%
goal not met; residual is the documented SASS-level ceiling.

### T8 — serving goodput: **GOAL MET (doubled)**
Bcap parameterized end-to-end + graph-driven scheduler (bit-identical to eager, FNV-digest gate)
+ first-fit admission + honest static-batching peer + opt-in int8-KV (1.88× smaller cache).
Same-clock interleaved sweep: **Bcap=256 full-fill 85.6× vs fill=1** (the old ceiling was
38–39× at Bcap=64, reproduced today at 38.2×), 33.1k tok/s; real scheduler drain 27.5k tok/s =
**1.14–1.27× vs static batching**. Honest decomposition disclosed: the multiple is
M-amortization; scheduling adds the 1.14–1.27×.

## Instrument notes (for future sessions)
- gemm_scaling's serial column drifts run-to-run (~±20% under changing power states); its
  scaling ratio is reliable, cross-run par-column comparisons are NOT. The MKL-anchored xbench
  matmul ratio is the cross-binary instrument.
- MKL-all and torch-Tn both ride power-state upside ~1.4–1.6×; Mercury @parallel currently does
  not — always report ranges with the peer's state visible (roofline column / MKL-1c column).
- First xbench/gemm_scaling run after a rebuild reads low (cold page cache) — discard.
- GPU cliff: cuBLAS self-noise sentinel read 0.916×@2048³ / 1.032×@4096³ — kernel-vs-kernel
  round-robin ratios are the trustworthy column, %-of-peer carries that noise.
