# Metrics that matter — and Mercury's honest standing

This document defines, from first principles, the metrics an ML/DL kernel language and its
compiler must win at, and records Mercury's current standing on each — including the losses.
It is the north star for optimization work: a change that doesn't move one of these metrics
(or protect a gate) is not worth its complexity. Numbers cited here are *recorded ratios* from
`BENCHMARKS.md` and `prompts/results/*`; absolute GFLOP/s are deliberately absent (this
hardware's clock swings ~3× CPU / ~7× GPU — only same-run ratios and %-of-roofline are stable).

## Who the metrics serve

Three real users: (a) **kernel authors** writing custom ops, (b) **model authors** composing,
training, and running models, (c) **deployers** serving inference. Everything below traces to
what one of them pays for.

## Tier 0 — hard gates (not scores; violating one voids every other number)

| Gate | Definition | Enforcement |
|---|---|---|
| G1 backend agreement | interp == native (== GPU within `c·√K·ε`) bit-for-bit on the differential suite | `cargo test` differential tests; `mercury_bench` equivalence gate |
| G2 opt invariance | `-O0` == `-O1/2/3` observable behavior (stdout + exit) on every run fixture | `optimization_is_observationally_invariant` |
| G3 numerical trust | vmath/norm kernels within documented tolerance of an f64 reference (33/36 vmath ops covered) | `vmath_kernels_match_f64_reference`, `norm.rs` f64 gates |
| G4 deterministic parallelism | `@parallel` == serial bit-for-bit (fixed chunking, ordered folds) | differential `@parallel` tests |
| G5 benchmark honesty | measured path = the real pipeline (parse→sema→MIR→opt→Cranelift JIT); recognizers structural, never workload-keyed; strongest-reasonable peer flags; same-run interleaved A/B | fairness audits; cross-checks inside `mercury_xbench` |

The documented exception to G1/G2: reassociated float reductions (vectorized/recognized) make
the *reassociated form* the oracle — both backends execute the identical reassociated IR.

## Tier 1 — dominant metrics (the reason to exist)

**M1. Kernel throughput on the ML op mix, vs the strongest peer available.**
Peers in order of strength: vendor libraries (oneMKL, cuBLAS/cuDNN) > tuned crates
(`matrixmultiply`) > idiomatic C/C++/Rust at `-O3 -march=native` (gcc/g++/rustc).
Current standing (recorded):

- *Compute-bound, CPU*: f32 GEMM ≈102–103% of oneMKL single-core at ≤512³, **85–86% at
  1024–2048³, ~77% roofline at 4096³** (loss); multicore **60–70% of MKL at 512–1024³**
  (loss), 86–129% at ≥2048³. int8 GEMM (VNNI) 1.5–2.5× gcc's own `vpdpbusd` auto-vec.
- *Compute-bound, GPU (RTX 4050, sm_89)*: fp16/bf16 GEMM ~101% cuBLAS ≤1024³, **~87–90% at
  2048³, ~83% at 4096³** (past the prior 77% PTX ceiling; 4096³ a SASS-level loss); int8 GEMM
  **96–105% at 2048³ (beats IMMA), 86–88% at 1024³, ~70% at 4096³** (HBM-bound loss); fused int8
  GEMM+dequant **1.1–2.2×** the cuBLAS chain; int4 W4A16 documented lead (no library peer exists);
  attention **0.37–0.66× cuDNN at S≥2048** (loss), wins fused-RoPE S≤512, causal D=64 S=512
  (beats cuDNN+cutlass), and D=128 ldmatrix S≤1024.
- *Memory-bound, CPU*: streaming elementwise ≈1.1–1.6× C (NT-store dispatch), honest
  physics-ties at L3-resident sizes (relu, biasadd, hadamard); reductions/norms/scans/column
  family 1.4–107× vs scalar-left-by-gcc patterns — but note these are IEEE-serial C baselines
  (see G5 fairness work: a `-ffast-math` C column normalizes the reassociation share).
- *Transcendentals*: 2–13× scalar libm, **but exp ~1.7× and log ~2.0× SLOWER than oneMKL
  VML** (algorithmic; propagates into softmax/GELU/xent when the peer is a library).

**M2. End-to-end model performance.** A compiler is judged on composed graphs, not op zoos:
fusion, no round-trips, layer-stack throughput. GPT-2-class `.mer` models exist and dispatch
to recognized kernels; GPU-resident 12-layer decode runs under CUDA-graph capture.
Standing: the CPU end-to-end model bench (`bench_model`) now runs — a 12-layer GPT-2-class stack
at **~26–32× idiomatic C / ~4.9–9.4× `-ffast-math` C (PRELIMINARY, ratios only)**, with
eager-PyTorch-CPU peer columns (T1/Tn sdpa+manual) wired; GPU training step still loses to eager
PyTorch (GEMM-bound); serving-stack (paged KV, continuous batching) measured GPU-only.

**M3. Multicore scaling** of M1 with G4 preserved. Standing: strong (dot ~7.9×, max ~25×,
GEMM to ~104× vs single-thread C), but until the OpenMP C column lands the multicore rows
have no multithreaded peer — treat them as scaling demonstrations, not peer wins.

## Tier 2 — what makes it a language, not a kernel library

**M4. Compile time.** Standing: the most defensible headline — `compile-vs` (all four
languages as subprocesses, compile-to-object at -O2) shows ~7–12× vs gcc/g++/rustc (measured
~7–9× this session, ~9–12× on faster-clock runs); in-process JIT latency ~0.3–1.5 ms/kernel and ~13 ms for a full GPT-2 block source→optimized
MIR; GPU cold JIT 0.76 ms / warm cubin 0.16 ms (~10⁴–10⁵× vs Triton cold).

**M5. Static shape safety at zero runtime cost.** Tensor shapes are types; static dims unify
by equality, symbolic dims bind (with a `rigid` mode so a callee cannot lie about its return
shape), `?` defers to runtime. Standing: strong and regression-tested (`E05xx` suite), but a
runtime `?` dim does not *execute* (C0001), which limits real serving code — a Tier-1
language gap (see M6).

**M6. Expressiveness for real ML code.** Can a user write a transformer fwd+bwd+train loop in
pure Mercury without escaping? Standing: forward blocks yes (statically-shaped, recognized
idioms); training via `--train` (CLI transform, not in-language); **known blockers**: no heap
allocation / returned tensors, runtime `?` dims don't run, tensors can't be element-generic
(`Tensor[T,M,N]` rejected → dtype kernels duplicated), no fn pointers/closures, no file I/O
(weights must be synthesized). (Multi-file `import a.b` *does* work — it splices items with
cycle/diamond dedup; only aliased/selective `import as` / `import x.{a,b}` stay partial.) These bound
how far "general programs a real user writes" can go today and are first-class improvement
targets, not footnotes.

**M7. Portability.** Same `.mer` → interp (oracle), Cranelift native, GPU offload,
GPU-native (whole-program MIR→PTX megakernel). Standing: real; GPU-native covers a subset
(UNSUPPORTED=skip) — it passes the eligible corpus except **2 documented general-lowering
miscompiles** (`hadamard`, `log_softmax_fused` in `lower.rs`/`megakernel.rs`); the
recognizer-offload GPU path and both CPU backends are unaffected and bit/tolerance-exact.

## Tier 3 — supporting qualities

Stable diagnostic codes with `--explain`; single-binary toolchain (no LLVM needed to build,
test, or run natively); reproducible benches (`mercury_xbench`, `mercury_bench`) with
in-harness cross-checks that can only *fail* Mercury, never inflate it.

## Current top improvement targets (ranked by user impact × measured gap)

1. End-to-end CPU model bench vs strong peers (closes M2's measurement hole).
2. vmath exp/log core vs MKL VML (~1.7–2× loss; propagates into every composed op).
3. Large-GEMM regime vs MKL (≥1024³ single-core 85%, mid-size multicore 60–70%).
4. GPU attention long-S vs cuDNN (0.37–0.66×).
5. Language blockers that gate real programs: runtime `?` dims, heap tensors, dtype-generic
   tensors, imports, file I/O (M6).
6. Decode-path primitives: KV-cache append/decode, top-k/top-p sampling, argsort (CPU).
7. Benchmark defensibility: relaxed-FP and OpenMP C columns, rustc opt-level 3 (G5).
