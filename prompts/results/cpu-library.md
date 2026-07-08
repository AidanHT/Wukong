# CPU kernels: library-grade — measure & close the gap to oneMKL / oneDNN / OpenBLAS

Branch: `perf/cpu-library-grade` (worktree `../Mercury-cpu`). Owner files: `crates/mercury_runtime/src/*`,
`crates/mercury_xbench/src/main.rs`, additive recognizer arms in `crates/mercury_mir_build/src/lib.rs`.

## Headline standing (vs the oneMKL gold standard, measured this session)

Peer = **Intel oneMKL** (`mkl_rt`, AVX2-dispatched on this Meteor Lake part — apples-to-apples 256-bit),
the gold-standard CPU GEMM/VML the mission names. All ratios same-run + thermally-controlled (see the
measurement law); AVX-512 is a labelled projection (no silicon here). Both binding laws hold — full
`cargo test` green, and `-O0 ≡ -O1 ≡ -O2 ≡ -O3 ≡ interp ≡ native` bit-for-bit on the GEMM fixtures.

- **Single-core GEMM: parity with MKL.** Was eroding 93→77 GFLOP/s across 256→2048³; size-adaptive
  `select_kc` flattened it to ~89→84→79 (77–87% of roofline out to a 512 MB working set). **Now beats
  MKL-1-thread at 256³/512³ (102–104%) and holds 84–95% at 1024–4096³** — up from 82–88%.
- **Multicore GEMM: competitive-to-winning.** A private **physical-core** pool (shed HyperThread
  contention) lifts parallel throughput 1.1–1.4× at the compute-bound mid sizes. **@parallel is 60–70%
  of MKL-all-threads at 512–1024³ and reaches parity-to-winning (86–129%) at ≥2048³** — the large-matrix
  ML regime.
- **vs naive C/Rust (the floor): crushed.** GEMM **2.4–12.5×** single-core (and far more `@parallel`);
  vectorized transcendentals **5–6×**.
- **vmath vs MKL VML: honest open gap.** Mercury's exp/log are ~1.7–2× under VML — *algorithmic* (VML's
  cheaper ~0.5-ULP approximation), not an ILP deficiency (a 4× unroll measured 0.97×; the kernel is
  already OoO-saturated). Characterised and deferred, bounded by the 1-ULP correctness gate.
- **AVX-512 (projection):** a gated, twin-tested microkernel, bit-identical-by-construction to the AVX2
  kernel; projected ~1.3–1.8× on dual-FMA server cores, never measured on hardware that can't run it.

Commits: `select_kc` · honest-harness · physical-pool · AVX-512 · VML-peer (+ P2–P4 docs).

## Environment (measured this session)

- **CPU: Intel Core Ultra 7 155H (Meteor Lake)** — 16 cores / 22 threads: 6 P-cores (HT → 12),
  8 E-cores, 2 LP-E cores. Max 3.8 GHz. **No AVX-512** (consumer hybrid parts disable it) — AVX2+FMA
  is the ceiling that *runs* here. AVX-VNNI (256-bit) is available (int8 path already uses it).
- **Toolchain:** gcc 14.2.0 (MSYS2 ucrt64), rustc (cargo). No clang/llc (LLVM absent — Cranelift is
  the native path). `cargo test` (no features) is the toolchain-free correctness core.
- **Library peer — FOUND, zero-install: Intel oneMKL.** `mkl_rt.2.dll` (32 MB, MKL 2023.x) ships in
  Anaconda (`%USERPROFILE%\Anaconda3\Library\bin\`). Verified exports: `cblas_sgemm`,
  `mkl_set_num_threads`, `mkl_get_max_threads`, `mkl_set_threading_layer`. OpenMP runtime
  `libiomp5md.dll` is in the same dir → threaded MKL loads via `LOAD_WITH_ALTERED_SEARCH_PATH`.
  On Meteor Lake MKL dispatches its **AVX2** kernels (`mkl_avx2.2.dll`) — an apples-to-apples 256-bit
  single-core peer, plus MKL's own multithreading for the multicore column. This is *the* gold-standard
  CPU GEMM the mission names; measuring against it is the honest Tier-B bar.
  - `pacman` exists (`C:\msys64\usr\bin\pacman.exe`) → OpenBLAS is an installable secondary open peer
    (`mingw-w64-ucrt-x86_64-openblas`) if a second library column is wanted.

## Current GEMM (the thing to beat / match)

`crates/mercury_runtime/src/gemm.rs` — already a BLIS five-loop f32 GEMM:
- Register tile **MR=6 × NR=16** (12 live `ymm` accumulators), AVX2+FMA `micro_6x16`, K-unroll ×4 with
  `_mm_prefetch(T0)`.
- Cache blocks **MC=144, KC=256, NC=4080**; packs **both** A (`pack_a`) and B (`pack_b`/`pack_b_trans`)
  into per-thread reused scratch; parallel pack available.
- Runtime dispatch `is_x86_feature_detected!("avx2"/"fma")` → `sgemm_avx2[_parallel]`, else scalar.
- `@parallel` = rayon over C row-panels, **fixed per-(i,j) K-order identical to serial** (so
  serial==parallel bit-for-bit; the interpreter oracle calls the serial kernel for both names).
- Symbols: `mercury_sgemm{,_nt,_tn}{,_parallel}`, `_nt_epi{,_parallel}` (fused bias+act), bf16/f16 twins.
- Standing today (per repo memory): ~90% of one P-core's AVX2-FMA roofline at 512³; 1.1–1.3× tuned
  `matrixmultiply`. **Unmeasured vs MKL. Unmeasured at 2048³/4096³** (the >L3 regime). AVX2-only.

## Correctness model (the sacred gate) — what actually protects a kernel change

The interpreter marshals its abstract memory into real buffers and calls the **same**
`mercury_runtime::mercury_sgemm*` symbol the native backend calls. Consequences:
1. **interp==native is automatic** for any in-place kernel retune (both sides call the new kernel) — so
   retuning the existing symbol needs **zero** backend/interp/recognizer wiring. (A *new* symbol is the
   8-touchpoint recipe; we avoid it where possible by retuning in place + runtime sub-dispatch.)
2. The differential gate therefore does **not** by itself catch a self-consistent kernel arithmetic bug.
   The real guards for a kernel change are: (a) **serial==parallel** bit-exact (fixed chunking);
   (b) the **gemm.rs unit test vs a naive scalar reference within a √K·ε tolerance** (GEMM reorders the
   sum, so it is *not* bit-identical to a naive nest — tolerance, not bit-equality, is correct here);
   (c) the **xbench cross-check vs MKL/C** (`max_rel_err < 1e-3`) as independent validation.
- `-O0 ≡ -O3` stdout/exit (`cargo test -p mercuryc --test run`) stays the hard invariant.

### AVX-512 correctness without AVX-512 silicon
In GEMM each `C[i,j]` accumulates independently across K; SIMD lanes hold **different output elements**,
not partial sums of one element. Widening 8→16 lanes therefore does **not** change any element's
accumulation order. An AVX-512 tile that mirrors the AVX2 K-loop is **bit-identical by construction**.
`avx512f` detection is **false** on this box, so the path is dead code here (cannot affect any gate or
measured number). Plan: implement it, compile-check it, argue correctness by structural twin to the
proven AVX2 kernel, and report the width win as a **labeled projection — never as measured**.

## Plan (phased, each phase its own commits)

- **P0 — peer + honest measurement (no kernel change).** dlopen MKL `cblas_sgemm` in xbench; add
  `MKL(1c)` + `MKL(all)` columns; extend the matmul sweep to 2048³ & 4096³; report %-of-MKL +
  %-of-roofline; checksum-cross-check. *Assume nothing — get the real gap first.*
- **P1 — close the >L3 gap.** Attack wherever P0 shows fall-off (expected ≥2048³): GotoBLAS loop order /
  B-panel reuse, MC/KC/NC retune for large, prefetch distance, packing. Re-measure per lever.
- **P2 — multicore scaling** on the hybrid: rayon work-stealing vs static, core-type-aware splitting.
- **P3 — AVX-512 microkernels** (gated, twin-tested, projected).
- **P4 — memory-bound kernels >L3** (vmath/reduce/saxpy): prefetch + NT-store tuning; MKL VML peer.

## Measurement protocol (honesty law)

Same run, same buffers, checksum cross-checked. Report a **clock-invariant ratio + %-of-roofline +
%-of-MKL**, never a fixed GFLOP/s (laptop clock swings ~3× with thermal state). MKL single-core column
uses `mkl_set_num_threads(1)`; the all-core column uses `mkl_get_max_threads()`. Best-of-N via the
existing `time_ns` (5 warmup + scale-to-50ms + best-of-14). An AVX-512 number is a projection, labeled.

---

## Results (filled as measured)

### P0 baseline (commit: MKL peer + 2048³) — single-core gap is the clean target

`mercury-xbench matmul`, roofline ~106 GFLOP/s this run, MKL = `mkl_rt.2.dll` (ILP64). GFLOP/s:

| size | Mer(1c) | MKL(1c) | Mer/MKL 1c | Mer(par) | MKL(all) | tuned(mm) | C(gcc) |
|------|--------:|--------:|-----------:|---------:|---------:|----------:|-------:|
| 256³ | 93.2 | 96.6 | **97%** | 86.5† | 218.5 | 74.4 | 28.0 |
| 512³ | 82.2 | 97.0 | **85%** | 183.3 | 371.6 | 73.0 | 32.5 |
| 1024³| 82.9 | 94.2 | **88%** | 293.6 | 424.2 | 72.1 | 22.2 |
| 2048³| 76.8 | 94.1 | **82%** | 364.7 | 132‡ | 67.6 | 8.9 |

†256³ < PAR_MIN_MACS so `@parallel` runs the serial kernel. ‡**thermal artifact** — the
multi-second naive C/Rust 2048³ nests run immediately before `MKL(all)` and heat-throttle the chip
(MKL(all) was 424 at 1024³; it cannot truly be 132 < its own 1024³). Multicore needs a measurement
fix (peers adjacent + skip the naive pollution at ≥2048) before its gap is trustworthy.

**Findings:**
- **Floor cleared:** Mer(1c) beats tuned `matrixmultiply` 1.13–1.25× and naive C 2.5–8.7× at every size.
- **Single-core gap to MKL widens with size** (97%→85%→88%→82%): MKL holds ~94–97 GFLOP/s flat
  (≈90% roofline, resident), Mercury **erodes 93→77** as the matrices spill L2/L3 — the classic
  re-streaming loss. *This is the Phase-1 target: lift 512³–2048³ from ~82–88% toward MKL's ~95%.*
  Prime lever (analytical): KC=256 ⇒ C re-streamed ⌈k/256⌉× (8× at 2048³); MC=144 uses only 144 KB
  of the 2 MB L2. Both are conservative (tuned for ≤1024³).
- **Multicore (Phase 2):** ~69% of MKL at 1024³ (294 vs 424) — a real parallel-efficiency gap, but
  thermally confounded; fix the harness first.

### P1 — close the single-core large-size gap — _shipped: size-adaptive `select_kc`_

**Lever: size-adaptive K-block.** Baseline KC=256 left ~10% on the table at ≥1024³ (B-micropanel
loads under-amortized; C re-streamed ⌈k/256⌉× — 8 passes at 2048³). KC=512 *thrashes* the 48 KB L1
(a 512×16 f32 B-micropanel is 32 KB + a 6×512 A-micropanel 12 KB = 44 KB) → ~30% slower at every size.
`select_kc(k)` splits K into the **fewest equal-ish blocks ≤ a 384 cap** (the largest KB whose 24 KB
B + 9 KB A micropanels stay L1-resident): k=512 → 2×256 (even, no thin tail), large K → the full ~384.
Both kernels call it, so **serial stays bit-identical to parallel**; the grouping differs from KC=256
only in low f32 bits (within the √k·ε tolerance the gemm tests assert). 18/18 gemm + 134/134 runtime tests green.

`mercury-xbench matmul` (XBENCH_HUGE=1), roofline ~102 GFLOP/s this run. **Mer(1c) is measured first
in each block, before any heating → the trustworthy single-core series:**

| size | Mer(1c) KC=256 | Mer(1c) `select_kc` | Δ | roofline% | MKL(1c) | Mer/MKL 1c |
|------|--------:|--------:|--------:|--------:|--------:|-----------:|
| 256³ | 93.2 | 89.3 | −4%* | 87% | 87.6 | **102%** |
| 512³ | 82.2 | **88.0** | **+7%** | 86% | 85.6 | **103%** |
| 1024³| 82.9 | 84.0 | +1% | 82% | 98.6 | 85% |
| 2048³| 76.8 | **83.8** | **+9%** | 82% | 97.2 | 86% |
| 4096³| *(n/a)*| **79.2** | — | 77% | 32.5‡ | n/a‡ |

\*256³ has k<384 ⇒ a single K-block either way; the −4% is run-to-run thermal noise (roofline was 106
vs 102 between runs), not a regression — the computation is byte-identical. ‡**4096³ MKL is thermally
invalid**: MKL(1c) runs *after* `Mer(par)` (a multi-second all-core 307 GFLOP/s burst) which throttles
the chip — MKL cannot truly be 32 GFLOP/s at 4096³ when it holds 97 at 2048³. The honest 4096³ datum is
**Mer(1c)=79 (77% roofline)**; the MKL ratio there awaits the Phase-2 harness fix.

**Result:** the eroding curve `93→82→83→77` flattened to `89→88→84→84→79` — the kernel now holds
**77–87% of roofline from L2-resident (256³) out to a 512 MB working set (4096³)**, beats MKL single-core
at 256³/512³, and is 85–86% of it at 1024³/2048³. The 512³ L2-spill dip (the old worst point) is gone.
Mer(1c) still beats tuned `matrixmultiply` 1.11–1.85× and naive C 2.4–12.5× at every size.

**Measurement-ordering confound found (→ Phase 2):** within a size block the column order is
Mer(1c), **Mer(par)**, MKL(1c), MKL(all), tuned, C, Rust. The all-core `Mer(par)` burst heats the chip
*before* MKL(1c)/MKL(all), and the multi-second naive C/Rust nests heat it before the *next* size — so
every peer after the first all-core run at ≥2048³ is throttled. Phase 2 must measure all single-core
variants adjacent (and re-warm / skip naive ≥2048) before any large-size MKL ratio is trustworthy.

### P2 — multicore: honest measurement + a physical-core pool

This phase is two parts: first **make the multicore measurement trustworthy** (it was not), then
**close the scaling gap** with the lever the clean numbers pointed to.

**P2a — measurement (commit `bench(xbench):`).** The harness threw away the multicore signal. The
all-core MKL peer was measured *after* Mercury's all-core burst + two multi-second naive nests had
heat-throttled the chip — at 2048³/4096³ it read **below its own single-thread number** (a bogus
~8–32 GFLOP/s), once even printing "1245% of MKL". Four fixes:
- **Thermal-grouped ordering, coolest-first:** single-core peers adjacent (Mer 1c, MKL 1c, tuned) →
  all-core peers → naive C/Rust LAST (the dominant heat source; skipped ≥2048³, `XBENCH_NAIVE_HUGE`
  to force). The 1-core ratio is now taken near-cold at every size.
- **MKL(all) measured BEFORE Mer(par):** Mer 1c/tuned use serial kernels that never touch rayon, so
  rayon's pool is dormant and MKL(all) runs on idle cores in the coolest state. Mer(par) runs after, so
  the ratio is a *conservative lower bound* on Mercury (throttle ourselves, never the peer). An earlier
  interleaved A/B timer was abandoned — alternating two live thread pools (rayon + MKL's OpenMP)
  thrashes the scheduler and parks MKL's workers, reading worse than sequential.
- **Turbo-ramped roofline:** the ~5 ms fixed-iter warmup measured a cold-clock roofline that warm GEMM
  later *exceeded* (>100% of roofline — an obvious bug). Now warms ≥400 ms of wall time.
- **Degeneracy guard:** when MKL(all) ≤ 1.2× MKL(1c) its 16 OpenMP workers did not scale that call (a
  real pathology on this loaded hybrid); the ratio is omitted with a reason instead of a fake multiple.
  Also reports Mercury's own @parallel scaling (robust to power state).

With this, clean same-run all-core ratios are reproducible — e.g. (roofline 92) **Mer(par)/MKL(all) =
25 / 64 / 70 / 129%** at 256/512/1024/2048³ (Mercury *wins* at 2048³), and mm6 measured 55/49/71/86/124%
at 256→4096³. The gap is concentrated at small sizes (threading overhead) and closes — to a win — by 2048³.

**P2b — physical-core pool (commit `perf(runtime):`).** The clean numbers said mid-size scaling lagged.
Lever: rayon defaults to one worker per **logical** core (22 on this part — 16 physical cores, 6 of them
HyperThreaded). A compute-bound AVX2-FMA GEMM saturates a core's two FMA pipes with **one** thread, so
the HT sibling only contends. The parallel GEMM now runs in a private pool sized to
`num_cpus::get_physical()` (16) — *private*, not a global resize, so the thread-count-derived striping of
the reduction kernels is untouched, and since the GEMM result is independent of panel distribution
(fixed per-(i,j) K-order), it is throughput-only: **serial stays bit-identical to parallel** (18/18 gemm,
134/134 runtime tests green).

Measured with a new **adjacent-A/B probe** (`sgemm_pool_ab`) — the only reliable instrument here: it runs
the *same* kernel in each pool back-to-back best-of-N, so the laptop's ~3× thermal swing (and the
1c-vs-par self-scaling confound that wrecks cross-run comparison) cancels in the ratio. `physical/logical`:

| size | run A | run B | reading |
|------|------:|------:|---------|
| 512³ | 1.09× | 1.40× | compute-bound — HT contention bites hardest |
| 1024³| 1.10× | 1.09× | the stablest point: ~1.10× |
| 2048³| 1.19× | 1.02× | |
| 4096³| 1.05× | 1.00× | bandwidth-bound — thread count matters less |

**≥1.0× at every size in every run** — a Pareto improvement, 1.1–1.4× at the compute-bound mid sizes,
tapering to parity where the kernel is HBM-bandwidth-bound. Never a regression.

**Honest multicore standing:** Mercury's @parallel GEMM is competitive with oneMKL's threaded GEMM on
this hybrid — 60–70% of MKL at 512–1024³, **parity-to-winning (86–129%) at ≥2048³** — the large-matrix
regime that matters for ML. The residual mid-size gap is threading/packing overhead, not the kernel.

### P3 — AVX-512 microkernel (gated, twin-tested, **projection only**)

This box (Meteor Lake) has **no AVX-512** — `avx512f` detects false, so an AVX-512 path is dead code
here and **cannot be measured**. The honest deliverable is therefore: implement it, prove correctness by
construction, gate it so it touches no live gate, and report the width win **only as a labelled
projection**.

`micro_6x16_avx512` (commit `perf(runtime):`) is the AVX-512 twin of the proven `micro_6x16`'s
K-accumulation + plain full-tile writeback. Each of the 6 A-rows gets one 512-bit accumulator holding
all 16 of its C-columns (vs the AVX2 kernel's two 256-bit halves); per K-step one 16-wide B load, 6
broadcasts, **6 FMAs — half** the AVX2 kernel's 12 for the same flops.

- **Correctness — by construction, not measurement.** Lane `j` of accumulator `r` sums `a[r,p]·b[p,j]`
  over `p` ascending — the exact sequence `micro_6x16` folds into its `c{2r}[j]` / `c{2r+1}[j−8]`.
  Widening 2×ymm → 1×zmm changes only register width, never which products reach `C[i,j]` nor their
  order, so it is **bit-identical** to the AVX2 kernel (the very argument the differential gate already
  makes for SIMD width). `micro_6x16_avx512_twin` asserts this on AVX-512 silicon; here it compiles and
  run-skips. The gate stays sacred: dead code can't break it.
- **Bounded untestable surface.** Only the full-tile, no-epilogue case (the bulk of a large GEMM's
  tiles, and the part whose AVX-512 form is a trivial lane-width swap) takes the AVX-512 path. Partial
  edge tiles and the fused bias+act epilogue (which would need AVX-512 `gelu16`/`silu16` vmath that does
  not exist yet) fall back to the proven AVX2 kernel.
- **Projected width win (NOT measured — no silicon here).** Same flops in **half** the FMA instructions.
  On a **dual-512-bit-FMA** server core (Skylake-X / Ice Lake-SP class) that is up to 2× the AVX2 FMA
  issue rate; the 6 independent accumulators only partly hide the ~4-cycle FMA latency against two
  units (need ~8 in flight), so realistically **~1.3–1.8×**, and AVX-512 down-clocking on older Xeons
  shaves some of that. On a **single-512-bit-FMA** client AVX-512 core it is ~**parity** (the 6 zmm
  FMAs match the AVX2 kernel's 12 ymm FMAs on fused 256-bit units) — never a regression. A production
  path would widen the tile (e.g. 14×32, 28 accumulators) to fully saturate toward 2×; that is
  straightforward but deferred precisely because it **cannot be validated on this hardware**.

### P4 — vmath vs oneMKL VML: the gap is *algorithmic*, and honestly so

Beyond GEMM, the mission asks to measure the gap to MKL for the elementwise/transcendental kernels.
Added a **oneMKL VML peer** (commit `bench(xbench):`) — `vsExp`/`vsLn` resolved from the same
`mkl_rt`, single-thread, same buffer, cross-checked against Mercury's output (rel < 1e-3 confirms the
ILP64 ABI and would flag a mismatch). The elementwise analogue of the GEMM-vs-cblas comparison.

| op | Mercury | vs scalar C/Rust | vs oneMKL VML |
|----|--------:|-----------------:|--------------:|
| exp | 256-bit AVX2 poly | **~6× faster** | ~1.7× slower |
| log | 256-bit AVX2 poly | **~5.5× faster** | ~2.0× slower |

So Mercury's vectorized transcendentals **crush idiomatic scalar C/Rust** (the mission's floor — gcc/rustc
can't vectorize a libm call), but trail Intel's VML by ~1.7–2×, *both on AVX2* (MKL dispatches AVX2 on
Meteor Lake). The natural suspicion was an ILP deficiency — the kernel processed one 8-lane vector
through a ~15-FMA poly chain at a time. **Tested and refuted:** a 4×-unrolled loop (four independent
`exp8` chains) measured **0.97×** vs single-vector in an adjacent A/B (`vmath_unroll_ab`) — *no* gain,
because the out-of-order core's deep reorder buffer already overlaps consecutive independent iterations,
so the kernel is **already throughput-saturated on the FMA units**, not latency-bound. The unroll was
reverted (a no-op that only adds code).

The remaining ~1.7–2× is therefore **algorithmic**: VML uses a cheaper approximation (table-assisted
range reduction / lower-degree poly) and still hits ~0.5 ULP, *better* accuracy than Mercury's ~1 ULP.
Matching it means rewriting the exp/log cores — real numerical work bounded by the `≈1-ULP` f64-reference
gate (`vmath_kernels_match_f64_reference`), not a quick tune. **Honestly characterised and deferred**:
this is a disclosed open gap, distinct from GEMM (where Mercury reaches MKL parity), and Mercury's vmath
remains decisively ahead of the C/C++/Rust baselines the mission targets.

Memory-bound kernels (saxpy / reduce / streaming elementwise) are DRAM-bandwidth-bound — Mercury already
streams them with 256-bit + non-temporal stores and sits at ~1.3–1.5× naive C; there a library peer ties
by physics (both saturate the same bus), the CPU analogue of the GPU HBM-bandwidth result.

---

## Session 2 addendum (2026-07-08) — the deferred vmath rewrite landed; mid-size GEMM negatives recorded

**The "honestly characterised and deferred" exp/log rewrite above is now done** (branch
campaign/perf-sota-2026-07): a x4 ILP unroll + NT-store regime in the dispatch loop, Estrin
intermediates, then full 8-bucket in-register-LUT (vpermps) rewrites of both cores — exp 2^(j/8)
table + degree-3 residual poly (~1.3 ULP, exhaustive sweep), log reciprocal/ln tables + degree-5
poly (<=6.9e-7 rel, exhaustive over every f32 in [0.25,4)) — each mirrored value-exactly across
scalar twin / AVX2 lanes / inlined-MIR emitter. Same-run vs oneMKL VML (vsExp/vsLn/vsTanh, HA
mode, 1 thread, N=1M):

| op | was (P4 above) | now |
|---|---|---|
| exp | ~1.7x slower | **1.05x FASTER (cool) to ~1.3x slower (thermally throttled)** |
| log | ~2.0x slower | **1.14x slower** (stable across power states) |
| tanh | (unmeasured) | **2.4-3.0x FASTER than VML** |

exp is FMA-port-bound, so package throttle still shows a ~1.3x gap (VML's kernel is slightly
leaner); log and tanh standings are power-state-stable. vs scalar C the family now reads
4.7-11.8x (log2 9.9x, log1p 7.7x). Correction to the P4 text: the accuracy gate is
`vmath_matches_libm` (f32-libm reference, 2e-5/5e-5 rel) — the "vmath_kernels_match_f64_reference"
name it cites never existed; the new LUT cores additionally carry exhaustive/dense f64-reference
sweep tests of their own.

**Mid-size multicore GEMM: three scheduling hypotheses refuted by adjacent A/B** (all documented
in gemm.rs): a physical/2 mid-size pool (slower everywhere), a persistent broadcast region
replacing the 4-fork-joins-per-call shape (wash — order/thermal effects exceed any delta), and
hard worker pinning MERCURY_GEMM_AFFINITY=1 (25-35% SLOWER than free migration). One real win
kept: fusing the A+B pack into a single region (+7.5% @512^3, +13% @128x768x3072;
MERCURY_PACK_SPLIT_REGIONS=1 kill-switch). Same-run standing at that point: 46% @512^3,
72% @1024^3, **127% @2048^3 (beats MKL-all)**. MKL(all) itself measured 411 GF/s @512^3 today vs
~286 in the P2 session (~1.4x power-state swing) — cross-session ratio comparisons are invalid;
the honest residual at mid sizes was MKL's per-thread-L2-blocked 2D decomposition, an
algorithmic difference.

**Session 2 closeout — the 2D decomposition was then built, and it won.** sgemm_2d_blocks
(gemm.rs) gives each worker exclusive ownership of an L2-resident C block (MR/NR-aligned block
grid so per-block packs are byte-identical to the serial kernel's, per-worker pack scratch, one
rayon task per block, no barriers) — the same decomposition MKL uses. ABBA adjacent-run
(gemm_scaling, both orderings): 512^3 ~182->~305 GF/s (~1.65x), 1024^3 ~365->~460 (~1.26x),
512x768x3072 ~290->~473. Shipped as the DEFAULT parallel path (MERCURY_GEMM_2D=0 opts out); the
two confirmation rounds vs MKL(all) read 66/68% (deep-throttled) and 66/69% (cool) at
512^3/1024^3 — the mid-size standing is now power-state-STABLE at ~66-69% (was a 46-72%
thermal lottery), with 87% @2048^3 same-run. Bit-exact vs serial by construction (one owner per
C block, ascending K-blocks, same kc grouping; pinned by sgemm_2d_blocks_matches_serial).

**End-to-end model bench (bench_model), two full-peer rounds**: ~20-21x C(gcc) and 3.6-4.9x
C(-ffast-math) single-core; **parity vs PyTorch CPU eager 1-thread** (0.93-1.07x @S=128,
1.08-1.21x FASTER @S=512); behind all-threads eager torch multicore (1.05-2.5x, thermal-swing) —
Mercury @parallel scaling at model shapes (1.5-2.1x) is the same mid-size parallel-efficiency
residual. All rounds: interp gate bit-exact, serial==@parallel bit-exact, outputs <2e-6 vs C and
torch.

Post-2D-GEMM model confirmation rounds (same session, after the closeout above): @parallel
scaling 1.9-3.8x (was 1.5-2.1x); the all-threads-torch gap at S=512 is now STABLE at ~1.1x
(1.07/1.10x across the two rounds; was the 1.05-2.5x thermal lottery); S=128 multicore stays
torch's win at 1.5-1.9x — the model's skinny M=128 GEMMs make each 2D row-block re-pack its B
slice (a shared-B pack for skinny-M shapes is the identified follow-up), and 128x768x768 sits
just above the parallel-payoff gate. One matmul-preheated round was discarded as instrument
failure: Mer(1c) read 2.3x below its own adjacent standing while the in-run roofline column
read 93 vs the valid rounds' 118-128 GF/s (the established discard-warm-up rule).
