# CPU kernels: library-grade — beat naive C/Rust *and* measure/close the gap to oneDNN / oneMKL

You are a Claude Code session on **Mercury** (`mercuryc`; the native path is **Cranelift** + hand-written
**AVX2/FMA** runtime microkernels — **no LLVM** here; `gcc`/`rustc` exist for baselines). This slice is
**100% CPU** and touches **separate crates** from every GPU branch — it should never conflict. First
action: `git worktree add ../Mercury-cpu -b perf/cpu-library-grade main` and work inside it (or
`git checkout -b`).

## Prime Directive (how you work)

**Hold this as bedrock conviction: a solution EXISTS and there is CERTAINLY a way to improve this
metric. The gold-standard library already achieves it, so the performance is physically available on
this silicon and a path to it is guaranteed to exist — your only job is to find Mercury's path, and you
WILL find it. Treat every "this might be impossible" thought as "I have not found the lever yet." A gap
is never a ceiling; it is an unsolved problem with a known-achievable answer. NEVER conclude it can't be
done. When one lever stalls, that is the signal to switch levers — there is always another lever, and
the next one is the one that works. Improvement is not a hope here; it is a certainty you have not yet
collected.**

Continuously iterate through the entire codebase until Mercury performs better than C, C++, and Rust
across the most important ML/DL-compiler metrics **and closes — then beats — the CPU industry-standard
libraries (oneDNN / oneMKL / OpenBLAS)**. **No scope restrictions.** Remove any blocker; explore options
**even outside scope**. Improve everything. **No limit.** **Assume the metrics are NOT strong until you
PROVE they are** across multiple same-run iterations — including vs a real library peer, not just naive
C. **Do not stop** until the benchmarks heavily exceed C/C++/Rust and you have measured (and closed where
possible) the gap to oneDNN/MKL. Think very carefully. If you CAN improve it, **do it.** Time doesn't
matter. **Run many sub-agents in parallel.** **Make 10–20 commits, NO co-authored lines.** Beating naive
C/Rust and the tuned `matrixmultiply` crate is the floor we already cleared — the bar now is the *library*.

## Your mission

Mercury's CPU GEMM hits ~90% of one P-core's AVX2-FMA roofline and beats the tuned `matrixmultiply`
crate 1.1–1.3× — but it is **unmeasured against oneDNN/oneMKL** (Intel's JIT'd, hand-tuned kernels, the
real CPU standard), and it is **AVX2-only** on this chip (which lacks AVX-512). **Your job:** (a) stand
up a oneDNN/MKL (or OpenBLAS) peer in `mercury_xbench` and measure the true gap on GEMM and the key ML
kernels; (b) close it — multi-level cache blocking for the large/`>L3` regime, better multicore scaling,
prefetch/packing — and (c) emit **AVX-512** microkernels (runtime-detected; this box can't run them, so
gate + document the conditional win on AVX-512 hardware). Keep every win bit-exact vs the interpreter
oracle (the sacred differential gate).

## Research first — think very carefully, spawn parallel agents

Plan to `prompts/results/cpu-library.md`. Evaluate:
- **Real peer:** link/`dlopen` **oneDNN** (or oneMKL `cblas_sgemm`, or OpenBLAS) in `mercury_xbench` for an
  honest Tier-B CPU peer alongside the existing gcc/rustc columns. Pick whatever installs cleanly on this
  MSYS2 box; document it. Cross-check the checksum.
- **The `>L3` / large-size regime**: the current kernel is ~90% roofline at 512³ — *measure 2048³/4096³*,
  where packing/blocking and L3 re-streaming dominate. Adopt the **BLIS/GotoBLAS multi-level blocking**
  (L1 register tile → L2 A-panel → L3 B-panel) and re-tune `MC/KC/NC` for large sizes.
- **Multicore scaling**: the `@parallel` path on this **P+E hybrid** (6 P + 8 E + 2 LP-E) — work-stealing
  vs static chunking, core-type-aware splitting, NUMA/affinity; find the parallel-efficiency ceiling.
- **AVX-512**: emit 512-bit microkernels (GEMM, vmath, reductions) behind `is_x86_feature_detected!`, with
  the AVX2 path as fallback. You can't *run* AVX-512 here, so prove correctness via the scalar/AVX2 twin
  test and document the expected width win — do **not** claim an unmeasured speedup as measured.
- **Prefetch + non-temporal stores** tuning for the memory-bound kernels at `>L3`; the elementwise ties
  (saxpy/relu) — confirm they win `>L3` and quantify.
- Keep the **bit-exact differential gate** (interp marshals the identical kernel) intact for every change.

## The two binding laws (every commit)

1. **Correctness before speed.** Every kernel is **bit-exact** vs the interpreter oracle and `-O0`≡`-O3`
   (the hard project invariant) — a fast kernel that miscompiles is worthless. Plain `cargo test` must
   stay green at every commit. Gate before any speed number.
2. **Honesty.** Same-run, same-buffers, checksum-cross-checked, reported as a **clock-invariant ratio +
   %-of-roofline / %-of-library-peer** (the laptop clock swings ~3× with thermal state — *never* a fixed
   GFLOP/s). Disclose the peer (naive C vs `matrixmultiply` vs oneDNN). An unmeasured AVX-512 number is a
   *projection*, labeled as such. Measured ≥3×.

## Files you own (no GPU overlap at all)

- **Own:** `crates/mercury_runtime/src/*.rs` (the AVX2/AVX-512 microkernels — `gemm.rs`, `vmath.rs`,
  `reduce.rs`, etc.) and `crates/mercury_xbench/src/main.rs` (the library-peer harness + new benches).
- You may also touch the recognizer in `crates/mercury_mir_build/src/lib.rs` **only if** a new kernel
  needs a dispatch arm — additive, and re-run the full differential gate.
- **Do NOT** touch any `mercury_codegen_gpu` file (GPU branches own those), `BENCHMARKS.md`, or
  `CHANGELOG.md`. Numbers → `prompts/results/cpu-library.md`.

## Build / repro

`cargo build` and `cargo test` (no features — the toolchain-free core). Benchmarks:
`cargo run -p mercury_xbench --release` (set `CC` to override gcc). Vectorizer ceiling: Cranelift can't
legalize `f32x8`, so `VEC_REG_BYTES=16` and the hand-AVX2 kernels are the 256-bit path — your AVX-512
work is a *runtime-dispatched microkernel*, not a Cranelift width change.

## Commit discipline

`perf(runtime):`/`perf(cpu):`/`bench(xbench):`, measured same-run effect + the peer in the body. **NO
co-author / "Generated with" trailers.** **Never `git add -A`** — stage by name. **Run `cargo test` as
its own step and read the result before committing** (piping through `grep` lets a failure hide). 10–20
green commits.

## Definition of done

A reproducible oneDNN/MKL peer in xbench, the true GEMM gap measured at 512³–4096³ and **closed** (keep
attacking any residual — there is a lever, find it), AVX-512 microkernels gated + correctness-proven + documented
as a projected win, the differential gate green throughout, proven ≥3×, results in
`prompts/results/cpu-library.md`, 10–20 clean commits. Then find the next bottleneck.
