# Changelog

All notable changes to Mercury are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows semantic versioning.

## [Unreleased]

### Performance — close-the-NVIDIA-gap campaign (vs the vendor libraries)
- **CPU GEMM to oneMKL parity** (`perf/cpu-library-grade`): size-adaptive `select_kc(k)` + a private
  physical-core rayon pool put single-core GEMM at 102–104% of MKL-1-thread (256/512³) and 84–95%
  (1024–4096³); `@parallel` reaches 86–129% of MKL-all-threads at ≥2048³. MKL cblas/VML peers added to
  `mercury_xbench`. (vmath still loses to MKL VML — algorithmic, documented.)
- **fp16/bf16 large-GEMM cliff vs cuBLAS** (`perf/gpu-gemm-cliff-2`): the no-pad ldmatrix+XOR-swizzle
  workhorse takes 2048³ to ~87–90% and 4096³ to ~83% of cuBLAS (past the prior 77% `mma.sync` ceiling).
- **int8/fp8 GEMM vs cuBLAS IMMA / cuBLASLt** (`perf/gpu-quant-2`): int8 beats IMMA at 2048³
  (~96–105%); the fused int8 GEMM+dequant beats the cuBLAS GEMM+dequant chain 1.1–2.2×; fp8 is 82–151%
  of cuBLASLt. w64 warp-tile + autotuner candidates.
- **Fused attention vs a genuinely-fused FA2-class peer** (`perf/gpu-attention-2`): vs PyTorch SDPA's
  cuDNN / cutlass mem-efficient backends, the fused flash wins fused-RoPE 1.8–5.7× and causal D=64
  1.03–1.16× (beats both) for S≤512; D=128 ldmatrix beats cutlass for S≤1024.
- **Conv vs cuDNN** (`perf/gpu-conv-2`): a real cuDNN-9 peer plus implicit-GEMM + Winograd
  F(2×2,3×3)/F(4×4,3×3) + a fused epilogue + `conv2d_best` per-shape dispatch reach cuDNN
  parity-to-win (1×1 3.5–5.6×, deep-channel 0.93–1.21×).
- **GPU serving stack** (`perf/gpu-serving`): vLLM-style paged KV-cache + Orca continuous batching +
  whole-model decode CUDA graph + int8 KV (3.88× footprint) + a TP partition sim — 39.1× batching
  goodput at fill=64.

All measured same-run (clock-invariant) on a mobile RTX 4050; the documented gaps and measured negative
results are recorded in `prompts/results/`, and every kernel stays gated against the interpreter oracle.

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
- **Tuples, structs, pointers, `loop`, and constant-shape tensors execute**: tuples (`(a, b)`, field
  access/assign `t.0`, heterogeneous padded fields), structs (`struct S { … }`, literals with fields
  in any order, field access/assign), **nested structs** (struct-in-struct to any depth, an aggregate
  field deep-copied from a variable, arrays of structs, tuple-of-struct), pointers/references
  (`&mut x`, `*p` load/store, a pointer threaded through a call — address-taken locals stay in memory,
  so `-O0` == `-O3`), and `loop { … }` with `break`/`continue` all run end-to-end on both the
  interpreter and the native Cranelift backend. Aggregates lower to a flat padded byte buffer with no
  dedicated aggregate MIR type (the local's value *is* its base pointer, like an array; nested fields
  recurse, a non-literal aggregate field is a leaf-precise deep copy). **Constant-shape tensors** also
  run: a `Tensor[f32, R, C]` parameter passes by base pointer and a multi-dimensional index `a[i, j]`
  flattens to a row-major GEP — the shape-typed surface executing, not just shape-checking. A matmul
  written in that tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot-product and accumulate
  spellings, plus the `b[j,k]` `nn.Linear` `A·Bᵀ` form) **dispatches to the same tuned `mercury_sgemm`
  microkernel** as the flat `a[i*K+k]` spelling — a 2-index operand access supplies its row stride
  from the tensor's inner dimension (gated by `tensor_matmul_is_correct`/`tensor_matmul_accumulate_form`).
  Fixtures `tests/run/{tuple,struct,struct_nested,pointer,loop,tensor_add,tensor_matmul}.mer`; the
  aggregate path is differentially gated by `differential_{tuple,struct,nested_struct}` and pointers by
  `differential_pointer` (native vs interpreter, bit-for-bit). By-value aggregate parameters/returns
  (an sret ABI) now lower too (see the language-surface additions below); symbolic-generic tensor
  dimensions remain pending.
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
- **xbench**: a broadcast bias-add (`out[r,c] = x[r,c] + bias[c]`) cross-language row, plus a
  **fused FFN** row (`C = silu(A·Bᵀ)` — the real Dense/SwiGLU layer, matmul + activation folded into
  one `mercury_sgemm_nt_epi` C-write; **~24–26× single-core, ~48–95× `@parallel`** vs C, where C pays
  an un-tiled serial-reduction GEMM + a separate scalar-`expf` silu pass) and an **`argmax@parallel`**
  row (**~17× vs single-threaded C** — the multicore global argmax).
- **`match` expressions** (`tests/run/{match_expr,match_patterns,match_tuple}.mer`): integer/bool
  literal, identifier-binding, and wildcard `_` patterns, **or-patterns** `1 | 2 | 3`, half-open
  `0..10` / inclusive `0..=10` **range** patterns, **enum-variant** patterns `Color::Red` (matched by
  discriminant), and **tuple** patterns `(0, _)` (per-field tests + bindings, nesting, and composition
  like `(0 | 1, y)`), each with an optional `if` guard. The scrutinee is evaluated once and the whole
  `match` lowers to an if-else chain; it works in value and statement position. Gated by
  `differential_match`/`differential_match_patterns`/`differential_tuple_match` (native==interp, -O0..3).
- **C-style enums** (`tests/run/enum_cstyle.mer`): `enum Code { Ok = 10, Err = 20 }`, auto-incrementing
  `enum Color { Red, Green, Blue }` (0,1,2), and continue-after-explicit `enum Step { A = 5, B, C }`
  (5,6,7). A variant *is* its integer discriminant — usable in `let`, `==`, `as i32`, and as a `match`
  pattern. Unit variants only; tuple/struct-payload (tagged-union) variants and enum-payload matching
  remain unimplemented. Gated by `differential_enum`.
- **Top-level `const` usable as a value** (`tests/run/top_level_const.mer`): sema type-checks each
  initializer against its annotation (an unsuffixed literal adapts) and records it; mir_build inlines
  it at every use site — a bare value, in arithmetic, as an array index, as a loop bound, and when one
  `const` references another (recursive inlining). Gated by `differential_top_level_const`.
- **Tuple destructuring in `let`** (`tests/run/let_destructure.mer`): `let (a, b) = …`, nested
  `let ((m, n), o) = …`, a wildcard `let (keep, _) = …`, and destructuring a tuple-returning call
  result — each sub-pattern binds a view into the initialized tuple buffer (value semantics). Gated by
  `differential_let_destructure`.
- **Nested tuple-field access** (`tests/run/nested_tuple_field.mer`): `t.0.1`, `t.0.0.0`, as reads and
  as assignment targets — the parser splits a lexer-glued `N.M` float in field position into two
  consecutive tuple-field accesses. Gated by `differential_nested_tuple_field`.
- **By-value aggregate parameters and returns** (`tests/run/{struct_fn,struct_return}.mer`): a
  tuple/struct passed by value and a `fn … -> Struct` return are modeled with a MIR-level **sret** ABI
  (a hidden leading pointer, a deep-copy into it, a void return; the call site allocates the
  destination and passes it as the hidden first argument), so no aggregate ever rides in a register and
  both backends agree. A struct param resolves to its registry-aware byte-buffer type (passed by base
  pointer), and `p.x` through a `&`/`*mut` reference auto-derefs. Whole-aggregate **assignment**
  (`s = other;`, `*p = Struct{..}`) now deep-copies leaf by leaf (`tests/run/struct_assign.mer`). Gated
  by `differential_struct_across_fns`/`differential_struct_return`/`differential_struct_assign`.
- **Radix integer literals** (`tests/run/radix_literals.mer`): hex `0xFF`, octal `0o17`, binary
  `0b1010` — with `_` digit separators and an optional type suffix — parse to their real value (they
  previously all evaluated to `0`, since `parse_int` kept only the leading run of decimal digits). Gated
  by `differential_radix_literals`.
- **Char literals** (`tests/run/char_literals.mer`): `'A'` lowers to its `u32` Unicode scalar value,
  with the one-character escapes (`\n` `\t` `\\` `\'` `\0`), `\xHH` hex, and `\u{…}` Unicode escapes
  (the lexer's escape scanner now consumes the multi-byte forms). Gated by `differential_char_literals`.
- **Correctness fixes** (interpreter↔native divergences and ICEs removed; each gated bit-for-bit):
  - **`break`/`continue` outside any loop** is now a clean **`E0303`** instead of a backend divergence
    — the lowerer left a fallback `unreachable` the interpreter trapped (exit 1) but the native backend
    turned into a SIGILL (exit 132). Registered in the diagnostic catalog (`--explain E0303`);
    `tests/fail/break_outside_loop.mer`.
  - **`&&` / `||` now short-circuit** — they lowered to a bitwise `And`/`Or` of both eagerly-evaluated
    operands (so a side-effecting or guarded RHS always ran, e.g. `n != 0 && 100/n > 0` divided by
    zero); now lowered to control flow. Gated by `differential_short_circuit`.
  - **`continue` in a range `for`** now runs the loop step (it had branched straight back to the header
    without advancing the index — an infinite loop); a dedicated latch block holds the step, on both the
    sequential and `@parallel` per-thread paths. `tests/run/for_continue.mer` (`differential_for_continue`).
  - **float → narrow-int casts** (`1e30 as i8`) no longer ICE the native backend and now **saturate**:
    Cranelift's `fcvt_to_*_sat` can't target a sub-32-bit result, so the backend converts to `i32`
    saturating then clamps/reduces to the narrow range — Rust `as` semantics, matching the oracle.
    `tests/run/float_cast_narrow.mer`.
  - **Deep recursion** in the interpreter no longer aborts the process: `run_with_output` runs on a
    scoped 512 MiB-stack worker thread, so a deeply recursive program returns instead of overflowing the
    ~8 MiB main stack (which had taken the whole differential gate down). Gated by
    `deep_recursion_does_not_overflow_oracle`.
  - **int → f32 casts above 2⁵³** round in one step in the interpreter oracle — it had double-rounded
    `int → f64 → f32`, disagreeing with native's single `fcvt_from_*` by a full ULP. Gated by
    `differential_int_to_f32_rounding`.
  - **Math intrinsics on integer operands** no longer emit float-op-on-int MIR: `abs`/`round`/`floor`/
    `ceil`/`trunc` are now type-preserving (integer `abs` = `select(x<0, −x, x)`, rounding an integer is
    the identity) and `sqrt`/the transcendentals promote an integer operand to `f32` — native had
    rejected the old MIR while the interpreter ran it lossily. `tests/run/int_math.mer`
    (`differential_int_math_intrinsics`).

### Changed
- **Cast precedence fixed**: `*p as T` now parses as `(*p) as T`, not `*(p as T)` (which had
  mis-typed the deref as a `ptrtoint` then a load). `as` binds looser than `*`/unary, tighter than the
  binary operators.
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
- **`@parallel` global argmax/argmin** now dispatches to the multicore `mercury_argreduce_f32_parallel`.
  It previously fell through to the *serial* kernel (the parallel reduction recognizer handles only
  plain `+`/`fmax`/`fmin` reductions, not the argmax `(value, index)` bookkeeping), so global argmax
  never scaled across cores. The parallel kernel folds the same fixed `RCHUNK` decomposition in
  ascending order, so the index is bit-identical to the serial kernel the interpreter calls — **~17×
  vs single-threaded C** (the bandwidth-limited ~2× scaling over the 8.9× single-core fold).
- **Optimizer compile time ~31% faster** (in-process, 400-function `-O2`: 18.0 ms → 12.5 ms), from two
  output-preserving changes — the MIR is bit-identical to before, verified by `optimization_preserves_
  results`, the native-vs-interpreter differential gate, and the `-O0`-vs-`-O{1,2,3}` invariance gate:
  - **CSE value-numbering key** is now a packed, allocation-free `enum` instead of a `format!` string
    built per pure instruction on every pass (CSE is the costliest pass; the key induces the identical
    equality relation, so value numbering is unchanged).
  - **Fixpoint loop** skips passes already at fixpoint: a pass that ran with no change is not re-run
    until another pass mutates the function, which drops the optimizer's final all-passes no-op
    *confirmation* sweep without changing the sequence of mutations.

### Notes
- **Constant-shape** tensors, **C-style `enum`s**, and **by-value aggregate parameters/returns** now
  execute end-to-end (above); **symbolic-generic** tensor dimensions, SIMD vector *values*, and slices
  `[]T` parse and type/shape-check today but do not yet lower/run, and native LLVM linking is in
  progress.
