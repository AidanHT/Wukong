# CPU kernels: library-grade — measure & close the gap to oneMKL / oneDNN / OpenBLAS

Branch: `perf/cpu-library-grade` (worktree `../Mercury-cpu`). Owner files: `crates/mercury_runtime/src/*`,
`crates/mercury_xbench/src/main.rs`, additive recognizer arms in `crates/mercury_mir_build/src/lib.rs`.

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

### P1 — close the single-core large-size gap — _in progress_
