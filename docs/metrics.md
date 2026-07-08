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

- *Compute-bound, CPU*: f32 GEMM ≈93–104% of oneMKL single-core at ≤1024³, **83–86% at
  2048³+, ~74–77% roofline at 4096³** (loss); multicore **~66–69% of MKL at 512–1024³**
  (was ~46–72% — the BLIS-style **2D block-parallel path** landed 2026-07-08 as the default:
  per-thread L2-resident C blocks, per-worker packing, no barriers; adjacent-run ABBA
  1.26–1.65× over the row-panel path, and the ratio held 66/68% throttled / 66/69% cool —
  power-state-stable at last), 86–129% at ≥2048³ (**127% measured 2026-07-08 — wins**; 87%
  same-run in the 2D confirmation round). Threaded MKL itself swings ~1.4× with this
  laptop's power state — only same-run ratios are comparable. The three cheap scheduling
  levers remain refuted-by-A/B in gemm.rs; the 2D decomposition was the real lever.
  int8 GEMM (VNNI) 1.5–2.5× gcc's own `vpdpbusd` auto-vec.
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
- *Transcendentals*: 4.7–12× scalar libm, and the former ~1.7–2× VML loss is **closed to
  ~1.1–1.3×** (2026-07-08: ×4 ILP unroll + Estrin, then 8-bucket `vpermps`-LUT rewrites of
  both cores — exp `2^(j/8)` table + degree-3 poly at ~1.3 ULP, log reciprocal/ln tables +
  degree-5 poly at ≤6.9e-7 rel, exhaustively swept): same-run, **log 1.14×, exp 1.05×-faster
  (cool) to ~1.3× (thermally throttled — exp is FMA-port-bound so clock starvation still
  shows), and tanh 2.4–3× FASTER than VML**. Composites inherit: log2 9.9×, log1p 7.7× vs C.

**M2. End-to-end model performance.** A compiler is judged on composed graphs, not op zoos:
fusion, no round-trips, layer-stack throughput. GPT-2-class `.mer` models exist and dispatch
to recognized kernels; GPU-resident 12-layer decode runs under CUDA-graph capture.
Standing (2026-07-08, two full-peer rounds, all cross-checks <2e-6 rel, serial==@parallel
bit-exact, in-benchmark interp gate bit-exact): the 12-layer GPT-2-class stack runs
**~19–21× idiomatic C and 3.1–5.1× `-ffast-math` C single-core**; vs **PyTorch CPU eager**
it is **at parity single-thread (0.93–1.27× @S=128; 1.08–1.49× FASTER @S=512)** and behind
all-threads torch multicore — but the 2D block-parallel GEMM (M1) stabilized that gap:
**~1.1× @S=512** (1.07/1.10× in the two post-2D confirmation rounds; was a 1.05–2.52×
thermal lottery) and 1.5–1.9× @S=128, with Mercury's own @parallel scaling at model shapes
up from ~1.5–2.1× to **1.9–3.8×**. (One matmul-preheated round read Mer(1c) 2.3× below its
own adjacent standing and is discarded as instrument failure — its roofline column read 93
vs the 118–128 GF/s of the valid rounds.) GPU training step still loses to eager PyTorch
(GEMM-bound); serving-stack (paged KV, continuous batching) measured GPU-only.

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
(UNSUPPORTED=skip). The previously documented general-lowering gaps are all **fixed**:
`PTX_VELEM` implements the Hadamard/Div binary modes and `PTX_NORM` the log-softmax/L2 ops;
`mrt_sreduce`/`mrt_sreduce_coop` implement the full `mercury_sreduce_f32` op set including
sumabs(9)/absdiff(10) (`parallel_abssum`); float→narrow-int casts saturate (Rust-`as`/Cranelift
semantics, `float_cast_narrow`); and pointer slots in the megakernel's shared frame are stored
unconditionally so SPMD threads no longer dereference a null base (`tensor_1d_kernels@O3`
illegal-address). The `tests/run` corpus gate stands at 193/274 programs matching the interp
oracle at `-O0`==`-O3` (81 honest UNSUPPORTED skips, zero mismatches, zero device faults), and
the megakernel gate at 81 ran / 89 eligible (8 launch-time declines). The corpus gates also
isolate any future device fault: a `CUDA_ERROR_ILLEGAL_ADDRESS` is a **process-fatal sticky**
CUDA error (measured on this driver: `cuDevicePrimaryCtxReset` returns Ok but the re-retain
still errors — only a process restart recovers), so the harness records the root fault on its
own loud ledger, marks the device lost, and reports every later program as NOT RUN — one
faulting program can no longer cascade into ~100 false failures, and skipped programs are never
reported as passed. The recognizer-offload GPU path and both CPU backends are unaffected and
bit/tolerance-exact.

## Tier 3 — supporting qualities

Stable diagnostic codes with `--explain`; single-binary toolchain (no LLVM needed to build,
test, or run natively); reproducible benches (`mercury_xbench`, `mercury_bench`) with
in-harness cross-checks that can only *fail* Mercury, never inflate it.

## Current top improvement targets (ranked by user impact × measured gap)

Closed in the 2026-07-08 round (see M1/M2 for the numbers): the M2 measurement hole (full
peer table incl. eager PyTorch, all gates green), the vmath exp/log-vs-VML loss (~1.7–2× →
~1.1–1.3×, tanh now 2.4–3× faster), the five gpu-native device-kernel/lowering miscompiles +
the corpus-cascade harness hole, and the missing production D=128 tensor-core attention
dispatch. Remaining, ranked:

1. Multicore parallel efficiency at model/mid-GEMM shapes — the honest next bet named here
   last round (MKL-style 2D per-thread C ownership) was built and shipped 2026-07-08: mid-size
   GEMM moved 46–72% → stable 66–69% of threaded MKL, model @parallel scaling ~1.5–2.1× →
   1.9–3.8×, S=512 vs all-threads torch stabilized at ~1.1×. Still the top CPU gap: the
   residual ~30% at 512–1024³, the S=128 model regime (1.5–1.9× behind torch-Tn — skinny
   M=128 GEMMs pack B redundantly across row-blocks in the 2D grid), and the 256³
   deliberate-serial standing (~39% of MKL-all), worth re-probing under the lower-overhead
   2D path.
2. GPU attention long-S vs cuDNN (0.37–0.66×) — occupancy probe says SFU/serial-softmax
   bound; the only live lever is FA2-style warp specialization (heavy).
3. Language blockers that gate real programs: runtime `?` dims, heap tensors, dtype-generic
   tensors, file I/O (M6).
4. Decode-path primitives: KV-cache append/decode, top-k/top-p sampling, argsort (CPU).
5. Large-GEMM single-core tail (~83–86% of MKL 1-core at ≥2048³).
