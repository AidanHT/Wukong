# Metrics that matter — and Wukong's honest standing

This document defines, from first principles, the metrics an ML/DL kernel language and its
compiler must win at, and records Wukong's current standing on each — including the losses.
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
| G1 backend agreement | interp == native (== GPU within `c·√K·ε`) bit-for-bit on the differential suite | `cargo test` differential tests; `wukong_bench` equivalence gate |
| G2 opt invariance | `-O0` == `-O1/2/3` observable behavior (stdout + exit) on every run fixture | `optimization_is_observationally_invariant` |
| G3 numerical trust | vmath/norm kernels within documented tolerance of an f64 reference (33/36 vmath ops covered) | `vmath_kernels_match_f64_reference`, `norm.rs` f64 gates |
| G4 deterministic parallelism | `@parallel` == serial bit-for-bit (fixed chunking, ordered folds) | differential `@parallel` tests |
| G5 benchmark honesty | measured path = the real pipeline (parse→sema→MIR→opt→Cranelift JIT); recognizers structural, never workload-keyed; strongest-reasonable peer flags; same-run interleaved A/B | fairness audits; cross-checks inside `wukong_xbench` |

The documented exception to G1/G2: reassociated float reductions (vectorized/recognized) make
the *reassociated form* the oracle — both backends execute the identical reassociated IR.

## Tier 1 — dominant metrics (the reason to exist)

**M1. Kernel throughput on the ML op mix, vs the strongest peer available.**
Peers in order of strength: vendor libraries (oneMKL, cuBLAS/cuDNN) > tuned crates
(`matrixmultiply`) > idiomatic C/C++/Rust at `-O3 -march=native` (gcc/g++/rustc).
Current standing (recorded):

- *Compute-bound, CPU*: f32 GEMM single-core **~80–104% of oneMKL-1c across sizes, at or above
  parity (≥100%) at 2048³/4096³** (MKL-anchored ABBA, both orderings; the 2026-07-10 C-tile
  prefetch closed the large-matrix writeback-miss tail, the off-switch reproduces the old number).
  Multicore is **size-keyed 2D block-parallel** with **per-BLOCK packing** (each worker's
  "redundant" packing warms its own L2 — it beat the shared-cooperative-pack design in every ABBA
  ordering, the session's headline refutation) and **dynamic block-claiming the 2026-07-11 default**
  (halved the mid-size straggler variance). Standing vs MKL-all, same-run: **512³ ~98–99%, 1024³
  ~104%, 2048³ 91%, 4096³ 93%, 256³ ~94–110%** (the dynamic-claiming default closed the old
  70–83% mid-size gap; against MKL's power-degraded rounds the ratios read far higher — threaded MKL
  itself swings ~1.4–2× with power state, so only same-run ranges are quoted). Skinny transformer
  NT shapes **75–112% of MKL-all** (4/6 at or above parity; worst 128×768·768ᵀ 75–80%,
  overhead-bound at 151 MFLOP). int8 GEMM (VNNI) 1.5–2.5× gcc's own `vpdpbusd` auto-vec.
- *Compute-bound, GPU (RTX 4050, sm_89)*: fp16/bf16 GEMM ~101% cuBLAS ≤1024³, **~87–90% at
  2048³; the 4096³ GEMM ships the v2cs streaming epilogue at 76.8% of cuBLAS-f16 / 80.4% of the
  honest f32-out peer** (a SASS-level loss past the prior 77% PTX ceiling — see below); int8 GEMM
  **96–105% at 2048³ (beats IMMA), 86–88% at 1024³, ~70% at 4096³** (HBM-bound loss); fused int8
  GEMM+dequant **1.1–2.2×** the cuBLAS chain; int4 W4A16 documented lead (no library peer exists);
  attention **0.37–0.66× cuDNN at S≥2048** (loss — structural: SFU/serial-softmax bound on 20
  SMs; the FA2-style warp-specialized kernels were built and measured 2026-07-09 and win only
  **4–6% at S=4096**, now the default route there, tie at 2048, lose at ≤1024 — so the cuDNN
  long-S gap is honestly bounded, not closable by scheduling), wins fused-RoPE S≤512, causal
  D=64 S=512 (beats cuDNN+cutlass), and D=128 ldmatrix S≤1024. The 4096³ GEMM ships the
  **v2cs streaming epilogue** (+2.7%): **76.8% of cuBLAS-f16 / 80.4% of the honest f32-out
  peer** (the f16-out peer hides ~half of Wukong's f32 C-write traffic — both columns now
  printed; the residual is SASS-level).
- *Memory-bound, CPU*: streaming elementwise ≈1.1–1.6× C (NT-store dispatch), honest
  physics-ties at L3-resident sizes (relu, biasadd, hadamard); reductions/norms/scans/column
  family 1.4–107× vs scalar-left-by-gcc patterns — but note these are IEEE-serial C baselines
  (see G5 fairness work: a `-ffast-math` C column normalizes the reassociation share).
- *Transcendentals*: 4.7–12× scalar libm; vs MKL VML the 8-bucket `vpermps`-LUT rewrites
  (2026-07-08) hold **tanh 2.7–2.9× FASTER**, while the honest 2026-07-09 re-measure puts
  **exp at 1.23–1.45× slower and log ~1.25× slower across thermal states** (the recorded
  "exp 1.05× faster / log 1.14× faster" did not reproduce — those were single cool-session
  readings; ranges are the honest form). A bit-identical ldexp restructure of the exp tail
  was built, exhaustively verified, measured a ~10–15% LOSS in both thermal states
  (port-rebalancing cannot beat a clock throttle that slows all ports), and **reverted** —
  the residual exp/log gap is algorithmic (VML's cheaper core), documented, and stable.
  Composites inherit the wins: log2 9.9×, log1p 7.7× vs C.

**M2. End-to-end model performance.** A compiler is judged on composed graphs, not op zoos:
fusion, no round-trips, layer-stack throughput. GPT-2-class `.wk` models exist and dispatch
to recognized kernels; GPU-resident 12-layer decode runs under CUDA-graph capture.
Standing (2026-07-11, three independent same-day rounds — all cross-checks ≤1e-6 rel,
serial==@parallel bit-exact, interp gate bit-exact): the 12-layer GPT-2-class stack runs
**~19–21× idiomatic C and 3.6–4.9× `-ffast-math` C single-core**. The honest PyTorch bar is
**`torch.compile` (TorchInductor max-autotune, fullgraph, warmed; eager does NOT count)**, run
same-run at both `set_num_threads(1)` and all-threads. Single-thread, Wukong **beats
compiled-torch-1T at both S=128 and S=512**. Multicore — after the **`@parallel` head-loop region**
(2026-07-10: an independent-iteration `for` loop with body-local scratch inside an `@parallel` fn
outlines into a `wukong_parallel_for` region; conservative affine-disjointness legality, serial
kernels inside each iteration so serial==parallel stays bit-exact; the model spells its attention
head loop that way naturally) — Wukong @parallel is **1.39–1.81× FASTER than all-threads compiled
torch at S=512 and 1.07–1.43× FASTER at S=128** (it also beats torch's strongest *eager* config,
1.07–1.37× @S=512; isolated per-side probes with torch at its best). Model @parallel scaling
reaches **4.7× on 16 cores** (default 4.44×, best 5.22×), tracking the ~5.0–5.4× same-run hybrid
MKL ceiling with T=1 at serial parity; residual headroom vs 16 physical cores is the parallel-GEMM
grain (M1). GPU training step still loses to eager PyTorch (GEMM-bound); the serving stack's
continuous-batching goodput ceiling: **Bcap=256 full-fill 85.6× vs fill=1** (graph-driven
scheduler bit-identical to eager, **1.14–1.27× over the honest static-batching peer** — the
batching-vs-scheduling decomposition is disclosed), opt-in int8-KV cache **3.88× smaller than
f32** (1.94× than f16).

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
pure Wukong without escaping? Standing: forward blocks yes (statically-shaped, recognized
idioms); training via `--train` (CLI transform, not in-language); **known blockers**: no heap
allocation / returned tensors, runtime `?` dims don't run, tensors can't be element-generic
(`Tensor[T,M,N]` rejected → dtype kernels duplicated), no fn pointers/closures. **File I/O now
runs** (✅): the `read_<T>`/`write_<T>` family (`T` in {`f32`, `f64`, `i32`, `i64`, `i8`, `u8`})
loads and stores headerless raw little-endian blobs and is gated interp == native bit-for-bit, so
weights can be **read off disk rather than synthesized** — and, built on it, **GPT-2 124M inference
now runs end-to-end** (✅). `examples/gpt2_infer.wk` loads the real pretrained OpenAI GPT-2 weights
(124,439,808 f32 parameters, exported from HuggingFace `transformers` by `tools/export_gpt2.py`)
from a flat little-endian blob via `read_f32` (token ids via `read_i32`), runs the full forward
pass — token + positional embeddings, 12 pre-LayerNorm blocks (biased QKV, 12-head causal
attention, tanh-GELU MLP, residuals), final LayerNorm, tied LM head → logits `[5, 50257]` — and its
next-token logits **match HuggingFace's reference to a relative max error of 1.87e-6** (max|Δ| =
2.44e-4 vs max|ref| = 130.28; **argmax next-token = 1757, " John", for the prompt "Hello, my name
is" — matching HF exactly**; verified by `tools/verify_gpt2.py`). This is a **numerical-correctness
/ capability result, inference only — not training, and no speed or throughput claim is made or
implied**. Two honesty caveats: **tokenization is external** (the `.wk` consumes integer token ids
the HuggingFace tokenizer produced — Wukong has no tokenizer), and the full run is **native-only** —
the interpreter cannot hold 124M parameters, so the native Cranelift-JIT backend is what executes it
(a reduced-config twin, `examples/gpt2_infer_small.wk`, runs **bit-identically on interpreter *and*
native** under the CI examples differential + opt-invariance gate).
(Multi-file `import a.b` *does* work — it splices items with
cycle/diamond dedup; only aliased/selective `import as` / `import x.{a,b}` stay partial.) These bound
how far "general programs a real user writes" can go today and are first-class improvement
targets, not footnotes.

**M7. Portability.** Same `.wk` → interp (oracle), Cranelift native, GPU offload,
GPU-native (whole-program MIR→PTX megakernel). Standing: real; GPU-native covers a subset
(UNSUPPORTED=skip). The previously documented general-lowering gaps are all **fixed**:
`PTX_VELEM` implements the Hadamard/Div binary modes and `PTX_NORM` the log-softmax/L2 ops;
`mrt_sreduce`/`mrt_sreduce_coop` implement the full `wukong_sreduce_f32` op set including
sumabs(9)/absdiff(10) (`parallel_abssum`); float→narrow-int casts saturate (Rust-`as`/Cranelift
semantics, `float_cast_narrow`); and pointer slots in the megakernel's shared frame are stored
unconditionally so SPMD threads no longer dereference a null base (`tensor_1d_kernels@O3`
illegal-address). Re-measured 2026-07-30 at HEAD (`WUKONG_GPU_REQUIRED=1`, 2 passed / 0 failed): the
`tests/run` corpus gate stands at **217/332 programs** matching the interp oracle at `-O0`==`-O3`
(**115** honest UNSUPPORTED skips, zero mismatches, zero device faults), and the megakernel gate at
**83 ran / 91 eligible** (8 launch-time declines). Absolute coverage rose 193→217 as the corpus grew
274→332 with the hardening campaign's fixtures, so the *fraction* moved 70.4%→65.4% — the added
fixtures are mostly recognizer/element-type cases gpu-native honestly declines. The corpus gates also
isolate any future device fault: a `CUDA_ERROR_ILLEGAL_ADDRESS` is a **process-fatal sticky**
CUDA error (measured on this driver: `cuDevicePrimaryCtxReset` returns Ok but the re-retain
still errors — only a process restart recovers), so the harness records the root fault on its
own loud ledger, marks the device lost, and reports every later program as NOT RUN — one
faulting program can no longer cascade into ~100 false failures, and skipped programs are never
reported as passed. The recognizer-offload GPU path and both CPU backends are unaffected and
bit/tolerance-exact.

## Tier 3 — supporting qualities

Stable diagnostic codes with `--explain`; single-binary toolchain (no LLVM needed to build,
test, or run natively); reproducible benches (`wukong_xbench`, `wukong_bench`) with
in-harness cross-checks that can only *fail* Wukong, never inflate it.

## Current top improvement targets (ranked by user impact × measured gap)

Closed in the 2026-07-09/10 round (see M1/M2 for the numbers and
`prompts/results/perf-sota-session3.md` for the full ledger): the large-GEMM single-core
tail (C-tile prefetch, 2048³ now at/above MKL-1c parity, see M1), the 256³ deliberate-serial standing
(now engaged at ~94–110% of MKL-all under the dynamic-claiming default), the serving goodput ceiling
(Bcap=256, 85.6× + honest static peer), the S=128 model regime (1.5–1.9× behind eager at campaign
start → now 1.07–1.43× AHEAD of *compiled* torch, see M2), and the honest-instrument holes (f32-out cuBLAS peer column; exp/log/model ranges
re-based on multi-state measurement). Measured-and-bounded rather than closed: GPU long-S
attention (warp specialization built; wins only 4–6% @S=4096 — structural SFU bound), 4096³
GEMM (v2cs +2.7%; ~80% of the honest peer, residual is SASS-level), exp/log vs VML
(algorithmic; the ldexp lever was built, measured a loss both thermal states, reverted).
Remaining, ranked:

1. Multicore parallel-GEMM grain — the head-loop region + dynamic block-claiming landed (model
   @parallel now beats all-threads *compiled* torch at both S=128 and S=512; scaling 4.44–4.70× on
   16 cores). Dynamic claiming closed most of the old mid-size MKL-all gap (512³/1024³ now
   ~98–104%, was 70–83%); the residual is the smallest overhead-bound skinny shape (128×768·768ᵀ
   75–80%) and a clean verified-AC full-cube table (the 2026-07-11 attempt hit battery mid-run).
   Shared-pack, mid-pool, persistent-region, and fork-join alternatives are all measured/refuted
   in gemm.rs — a genuinely new decomposition idea is required for further mid-size gain.
2. Language blockers that gate real programs: runtime `?` dims, heap tensors, dtype-generic
   tensors (M6; **file I/O now runs** — typed read/write blobs, interp == native).
3. Decode-path primitives: KV-cache append/decode, top-k/top-p sampling, argsort (CPU).
