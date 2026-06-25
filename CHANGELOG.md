# Changelog

All notable changes to Mercury are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows semantic versioning.

## [Unreleased]

### Added
- **Front-end**: lexer (with `@attributes` and error recovery), recursive-descent + Pratt parser,
  AST with a pretty-printer, and `--emit=tokens|ast`.
- **Types & semantics**: the shared type vocabulary (`mercury_types`), name resolution, type
  checking, and **compile-time shape checking** for tensors (rank/dimension unification, symbolic
  dims), with errors `E0501`/`E0502`.
- **Middle-end**: block-parameter SSA MIR, a builder, a pretty-printer, and a verifier with a
  `MirLevel` invariant; AST → MIR lowering (alloca-per-local).
- **Optimizer**: a fixpoint pass manager backed by CFG and dominator analyses (Cooper–Harvey–Kennedy
  immediate dominators + dominance frontiers), with whole-program leaf-function `inlining`, `mem2reg`
  (promote scalar slots to block-parameter SSA), `simplify` (constant folding + algebraic identities
  + self-comparison folding), `simplify-cfg` (constant-branch folding + straight-line block merging +
  unreachable-block pruning), `simplify-phis` (dead/trivial block-parameter elimination), `dce`,
  `cse` (dominator-tree value numbering with load forwarding), `dse` (dead-store elimination), and
  `licm` (loop-invariant code motion), wired across `-O0..-O3`. In debug builds the pass manager
  verifies the MIR after every pass. Across the run suite and kernels, `-O3` removes ~48% of IR ops
  (54–60% on the heavy kernels) and runs ~1.5–2.5x faster than `-O0` under the interpreter.
- **Back-ends**: a zero-dependency MIR interpreter (`--run`), a from-scratch **native Cranelift
  backend** (JIT + host object, no LLVM toolchain), and a textual LLVM-IR emitter (`--emit=llvm-ir`,
  plus `--emit=obj|exe` via `clang` when present). The native backend is differentially tested
  against the interpreter bit-for-bit.
- **Matmul → tuned GEMM dispatch**: the compiler recognizes a matmul loop nest — the `ikj` accumulate
  and `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling — and lowers the whole
  nest to a register-blocked (6×16), cache-tiled, packed **AVX2/FMA** GEMM microkernel in the runtime
  (`mercury_sgemm` / `_nt` / `_parallel`). On a Meteor Lake laptop this beats `gcc -O3 -march=native`
  on the naive nest by **~2.4–3.5× single-thread and ~18× parallel** on `C = A·B` (and **~20–75× on
  `nn.Linear`**, where naive C stays latency-bound), the lead growing with matrix size. The serial
  kernel holds ~90–105 GFLOP/s (~80% of one P-core's AVX2-FMA peak; ~105 pinned at 512³); the parallel
  one packs panels across cores — reusing one pack-scratch allocation across all cache blocks instead
  of re-allocating per K-block (≈+45% at 1024³, ~395–434 GFLOP/s) — and skips threading below a work
  threshold. The 6×16 microkernel stores full tiles straight to C and unrolls the K loop ×4. The interpreter calls the identical
  kernel (marshalling its memory) so the oracle stays exact.
- **Auto-vectorization**: straight-line elementwise loops (incl. branchy ones via if-conversion) and
  float **reductions** (reassociated to vector-lane accumulators) lower to SIMD automatically;
  `x + y*z` contracts to a hardware FMA; adjacent same-range loops fuse. Reductions (`dot`, L2 loss)
  run ~2.6–2.8× faster than serial C.
- **Transcendental → 256-bit AVX2 dispatch**: a pure `out[i] = f(x[i])` loop for **35** functions —
  `exp`/`log`/`expm1`/`log1p`/`tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`mish`/`selu`/`tanhshrink`/
  `hardsigmoid`/`hardswish` plus **`softsign`** (bounded poly activation) and **`logsigmoid`** (the stable
  log-sigmoid behind binary-cross-entropy-with-logits / contrastive losses), **`sin`/`cos`/`tan`/`atan`/`asin`/`acos`**
  (RoPE rotary embeddings and the geometry/3D-vision/graphics-ML angle ops), **`erf`** (exact
  BERT/GPT-2 GELU), **`exp2`/`log2`/`exp10`/`log10`** (FlashAttention base-2 softmax, quantization, and
  base-10 decibel/log-scale features), **`cbrt`** (the all-real cube root — LAB color, variance-stabilizing
  transforms — completing the `sqrt`/`rsqrt`/`cbrt` root family), and the full
  **hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`** (`atanh` = the Fisher z-transform; the
  inverse trio powers hyperbolic/Poincaré embeddings and normalizing flows) — lowers to a tuned **256-bit AVX2/FMA runtime
  kernel** (`mercury_vmath_f32`) — the width Cranelift's general vectorizer can't emit (it caps at
  128-bit SSE). `silu` (Llama/SwiGLU) and `gelu` (BERT/GPT-2/ViT) are first-class intrinsics; the
  kernel's per-element op sequence mirrors the inlined Cephes/A&S polynomial, and the interpreter marshals
  through the identical kernel, so the differential oracle stays exact and dispatched/composed forms
  agree. A multi-statement (fusion-merged) body dispatches one kernel call per activation, and an
  `@parallel` activation dispatches each thread's chunk — so it runs multicore × 256-bit. Versus C's
  scalar `libm` (which can't vectorize a loop with a call), the family runs **~4–11.5×
  faster** single-thread (`asinh`/`acosh` and `sin`/`cos` win most — `libm`'s `asinhf`/`sinf`/`cosf`
  are heavier than `expf`), ~28× `@parallel`. The in-process `vmath_throughput` probe (kernel vs the
  scalar `libm`-call loop gcc/rustc are forced to emit) confirms it locally: exp ~5.9×, gelu ~5.5×,
  tan ~3.6×, asin ~5.4×, exp10 ~13×, logsigmoid ~4.6×.
- **Two-arg transcendentals → 256-bit AVX2 dispatch (`pow`/`atan2`/`hypot`)**: `pow(x,y) = exp(y·log(x))`,
  `atan2(y,x)` (the full-circle angle — geometry, robotics, complex argument), and `hypot(a,b)` (the
  overflow-safe 2-norm) get a two-input kernel (`mercury_vmath2_f32`): a `for j { out[j] = f(x[j], y[j]) }`
  loop lowers to the **256-bit** AVX2 kernel (the same width jump that ~doubled the single-arg
  activations), and the three kernels mirror the inlined `emit_pow`/`emit_atan2`/`emit_hypot` op-for-op
  (via the shared exp8/log8/atan8/√), so dispatched == composed and the interpreter marshals through the
  identical kernel — interp == native, -O0 == -O3. A composed/scalar use (or a body that fuses with an
  adjacent store loop) still lowers to the inlined ≈1-ULP poly (128-bit auto-vec). Either way a win over
  C's scalar `powf`/`atan2f`/`hypotf` (a loop with the call won't vectorize).
- **Streaming elementwise → 256-bit AVX2 dispatch**: a recognized streaming map
  (`out[i] = act(a·x[i] (+ b·y[i]) + c)`, incl. ReLU/ReLU6) lowers to `mercury_velem_f32` and a Horner
  polynomial to `mercury_vhorner_f32` — both true 256-bit AVX2/FMA, unrolled, emitting **non-temporal
  stores** once the working set spills L3 (the read-for-ownership-skipping store gcc/rustc won't emit).
  This turns the former memory-bound *ties* into wins: saxpy ~1.3×, poly ~1.2× at `N=2²⁰`, widening to
  ~1.3–1.6× at realistic >L3 tensor sizes. A velem **identity-affine fast path** (skip the wasted
  `fma(1·x+0)` for a bare `relu`/copy) plus gating the software prefetch on the DRAM/non-temporal
  regime removed a ~1.2× `relu` regression at L3-resident sizes (now a clean tie there, ~1.4× at >L3);
  `vhorner` runs **six** independent Horner chains (was four — a 5-deep dependent-FMA chain needs ~8 in
  flight to fill both FMA ports), turning the degree-4 poly tie into a consistent win over gcc's own
  256-bit autovec. The interpreter marshals through the identical kernel, so the oracle stays exact.
- **`@parallel` reduction → multicore reduction kernel**: a reduction loop in a `@parallel` function
  (`s += x[k]*y[k]`, `(x[k]-y[k])²`, or `x[k]`) lowers to a **deterministic multicore reduction
  kernel** (`mercury_sreduce_f32_parallel`: dot/ssd/sum/sumsq) instead of a sequential per-thread
  accumulation. The parallel result is bit-identical to the serial one regardless of core count
  (fixed-size chunks, ascending partial combine), and the interpreter calls the serial form, so the
  differential oracle stays exact. Spreads the stream across cores to aggregate memory bandwidth:
  **dot ~7.9×, ssd ~8.6× faster** than single-threaded C (which stays serial & latency-bound).
- **`@parallel`**: loops execute across CPU cores via a rayon runtime, each per-core chunk itself
  vectorized — ~2.2–8× faster than idiomatic single-threaded C on the (memory-bound) elementwise and
  reduction kernels.
- **Mixed-precision (bf16 *and* f16) → SIMD dispatch**: both `bf16` and `f16` are real 2-byte storage
  (round-to-nearest-even on store/cast, f32 compute; bf16 via inline bit-math, f16 via shared
  `half`-crate shims since Cranelift x64 lacks f16 convert lowering). A full, symmetric op suite over
  `[bf16]`/`[f16]` arrays read through an `as f32` widening cast (lossless: `<<16` for bf16, F16C
  `vcvtph2ps` for f16) with f32 accumulate/compute dispatches to half-precision runtime kernels:
  **`dot`/`sum`** (`mercury_{dot,sum}_{bf16,f16}`, ~3–8× vs C), **`max`/`min`/`absmax`**
  (`mercury_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric int8-quant scale), **streaming
  `axpby`** (`mercury_axpby_{bf16,f16}`, half-in/f32-out), and the **36-op activation set**
  (`mercury_vmath_{bf16,f16}`). Precision-generic recognizers; the interpreter marshals through the
  identical kernel, so native == interp bit-for-bit. C/Rust can vectorize neither a `libm` call nor the
  half→f32 widen, so the gap is structural.
- **Arrays**: fixed-size `[T; N]` run end to end — literal/repeat initializers, indexed load/store
  with a runtime index, and array parameters passed by base pointer (out-params work). Real kernels
  (dot product, SAXPY, a flat GEMM) run on the interpreter.
- **Intrinsics**: `print`/`println` (captured stdout) and `assert` (traps on false).
- **Runtime**: a bump `Arena` allocator and a deterministic `parallel_for`.
- **Diagnostics**: rustc-style renderer, a stable error-code catalog with `--explain <CODE>`, and
  `--error-format=json` (JSON Lines).
- **Tooling & tests**: end-to-end run-suite with `// EXPECT-*` directives, an opt-level differential
  test (`-O0` vs `-O1/-O2/-O3`), per-stage `--emit` smoke tests, the `mercury_bench` harness (IR-op
  reduction + `-O0`-vs-`-O3` interpreter speedup, doubling as an optimizer-equivalence gate over
  heavy kernels in `bench/kernels`), a GitHub Actions CI (fmt + clippy + test on Linux & Windows),
  and the language guide and internals docs.
- **Performance**: the interpreter pools per-call register files and passes block-parameter arguments
  through a reused buffer, roughly halving its wall-clock; large `[v; n]` array initializers lower to
  a fill loop instead of unrolled stores.
- **Embedding-lookup dispatch**: the LLM token-id row gather `out[t,:] = weight[ids[t],:]` (over an
  `i32` index array — the first recognized dispatch with an integer *index input*) is recognized and
  lowered to `mercury_embedding_f32[_parallel]` (a 256-bit row copy; bit-exact data movement, mapped
  across the independent output rows under `@parallel`).
- **2D pooling dispatch**: the idiomatic 5-deep max/avg-pool nest lowers to
  `mercury_{max,avg}pool2d_f32[_parallel]` (the CNN spatial downsampler; `@parallel` across channels;
  bit-exact — max is idempotent, the avg `(dy,dx)` sum order is fixed). Honest sharp edge: gcc
  auto-vectorizes regular-stride (e.g. 2×2/s2) pooling, so single-core is a tie there, not a win.
- **xbench**: a broadcast bias-add (`out[r,c] = x[r,c] + bias[c]`) cross-language row.

### Changed
- A construct lowering cannot yet handle (tensors, SIMD methods, generics, parallel loops) is now a
  hard `error[C0001]` instead of a warning, and the driver refuses to optimize, run, or codegen a
  module whose lowering failed — so the compiler never emits or executes invalid MIR.
- **Global argmax/argmin** (`mercury_argreduce_f32`) gained a 256-bit AVX2 path (4 accumulators × 8
  `f32` value + `i32` index lanes, strict-compare + `blendv`, collapsed through the scalar tie-break).
  It was the one memory-bound reduction lacking one, so it had *lost* to gcc's branch-predicted scalar
  loop by 1.30× — now **~5–9× faster**, bit-identical to the scalar lowest-index result.
- **Column argmax/argmin** (`mercury_colarg{max,min}_i32`) rewritten to a single row-major pass with
  the column range's running best kept L1-resident, instead of re-reading the matrix `cols/8` times
  per 8-column band — a former 0.9–1.6× tie/loss became **2.7–5.3× faster** (bit-exactness unchanged).
- **Fused elementwise chains**: the vectorizer now forwards a just-stored intermediate's register value
  to its consumer in the same fused body, dropping the per-element store+reload round-trip (one fewer
  memory stream per chained op; the intermediate need not touch memory between producer and consumer).

### Notes
- Tensors, SIMD vectors, and the parallel/GPU surface parse and type/shape-check today; full
  execution of those paths and native LLVM linking are in progress.
