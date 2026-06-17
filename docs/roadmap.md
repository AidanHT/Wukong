# Mercury Roadmap & Known Limitations

Mercury is built openly and incrementally. This page is an honest snapshot of what works, what is
checked-but-not-executed, and what is planned — so expectations match reality.

## Works end to end (interpreter `--run`, **and native code** `--backend=native`)

Two execution backends now run the full language and agree bit-for-bit (a differential gate proves
it across opt levels): the zero-dependency tree-walking interpreter (the reference oracle) and a
from-scratch **Cranelift native backend** (JIT for `--run --backend=native`, object/exe via
`--emit=obj|exe`) — **no LLVM toolchain required**. See `BENCHMARKS.md` for cross-language numbers.

- Modules, functions (including recursion and mutual recursion), and direct calls.
- `let`/`let mut`/`const`, shadowing, block-as-expression values.
- Integers (`i8..i64`, `u8..u64`, `usize`/`isize`), `bool`, and floats. `f32` is computed at **`f32`
  precision** (interpreter and native agree exactly); **`bf16` is real 2-byte storage** rounded to
  bf16 (round-to-nearest-even) on store and on `as bf16`, with `f32` compute; `f16` still promotes to
  `f32`; `f64` is full.
- All arithmetic/comparison/bitwise/boolean operators, compound assignment, casts.
- `if`/`else` (statement and value position), `while`, `for … in a..b [step s]`.
- **Fixed-size arrays** `[T; N]`: literal/repeat init, indexed load/store, array parameters passed
  by base pointer (out-params). Real kernels run: dot, SAXPY, GEMM, matmul, ReLU, clamp, transpose.
- **Matmul → GEMM dispatch**: the compiler recognizes a matmul loop nest (the `ikj` accumulate and
  `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling) and lowers the whole nest
  to a tuned register-blocked (6×16), cache-tiled, packed **AVX2/FMA** microkernel in the runtime —
  the way XLA/TVM/oneDNN lower a matmul op. Serial and `@parallel`. Beats gcc/rustc's naive nest
  ~2.4–3.5× single-thread and up to ~13× parallel on `C = A·B` (~19–70× on `nn.Linear`), the lead
  growing with size. Dimensions may be compile-time literals **or runtime values** (function
  params/locals): the recognizer checks strides symbolically, so a general matmul function dispatches
  to the kernel, not just fixed-size benchmark kernels. The interpreter calls the identical kernel
  (marshalling its memory), so the two stay bit-exact.
- **SIMD auto-vectorization**: straight-line elementwise loops (incl. branchy ones via
  if-conversion) lower to 128-bit vector ops, 4×-unrolled, with a scalar remainder — automatically,
  on the native backend. saxpy/poly/relu/relu6 vectorize.
- **FMA contraction**: a float `x + y*z` becomes one fused multiply-add (`Op::Fma`, a hardware
  `vfmadd`), on both the scalar and vector paths; the interpreter mirrors it with `mul_add`, so the
  two backends stay bit-identical.
- **Reduction vectorization**: a float reduction `s = s + x[k]*y[k]` / `s += ..` lowers to
  vector-lane accumulators (independent FMA chains) + a horizontal reduce + scalar remainder, turning
  the latency-bound serial sum into a throughput-bound one. `dot` runs ~2.6× faster than serial C.
  `fmax`/`fmin` reductions (`m = fmax(m, x[i])`, softmax's row-max) vectorize the same way.
- **Transcendental intrinsics**: `sqrt`/`rsqrt` (hardware), `exp`/`log` (≈1-ULP `f32` minimax
  polynomials), `pow` (= `exp(y·log(x))`), `tanh`/`sigmoid` (built on `exp`), and `fmax`/`fmin` — all
  built from primitive ops
  both backends already agree on bit-for-bit, and all **vectorize** in elementwise loops. So softmax,
  layernorm, GELU, SiLU/swish, tanh activations, and **log-softmax / cross-entropy** lower to SIMD
  instead of scalar `libm` calls and run **~2.5–3.5× faster** than gcc/rustc's scalar
  `expf`/`logf`/`tanhf` (`log` shows the largest margin). See
  `tests/run/{transcendental,softmax,layernorm,gelu,activations,log,log_softmax,ffn_block}.mer`.
- **Convolution via im2col + GEMM**: a conv written as an im2col gather followed by a matmul has its
  matmul recognized and dispatched to the tuned GEMM microkernel (the XLA/cuDNN lowering), so Mercury
  runs a 3×3 conv **~6–7× faster** than idiomatic hand-written direct convolution in C. See
  `tests/run/conv_im2col.mer`.
- **Operator fusion**: adjacent same-range elementwise loops (e.g. a linear map then ReLU) fuse into
  one loop when the combined body is dependence-safe; CSE then forwards the intermediate through
  registers rather than memory.
- **`@parallel`** functions execute across CPU cores (rayon runtime); the per-core chunk is itself
  vectorized. The interpreter runs the same range sequentially, so results stay differential-equal.
- Intrinsics `print`/`println`/`assert`.
- The optimizer (`-O0..-O3`), backed by CFG and dominator analyses: whole-program **inlining** of
  leaf functions, **mem2reg** (alloca → SSA), constant folding, algebraic simplification, CFG cleanup
  with block merging, dead/trivial block-parameter elimination, DCE, dominator-tree CSE with load
  forwarding, DSE, and **loop-invariant code motion**. Guarded by an `-O0`-vs-`-O{1,2,3}` differential
  test and post-pass MIR verification; across the run suite and kernels it removes ~48% of IR ops
  (54–60% on the heavy kernels) and runs ~1.5–2.5x faster than `-O0`.

## Checked but not yet executed

- **Shape-typed tensors** `Tensor[f32, M, N]`: parse and pass compile-time shape checking
  (`E0501`/`E0502`), the headline feature — but tensor *operations* are not yet lowered/run.
- **Explicit SIMD vector types** `f32x8` etc. in *source*: parse and type-check; user-written vector
  values are not yet executed. (Loop auto-vectorization above is separate and *does* run.)
- **Attributes** `@simd`/`@tile`/`@align`/`@extern`/`@export`: parse and validate; consumers in
  progress. (`@parallel` now executes — see above.)

## Planned

- Fusing chains *under* `@parallel`, and GEMM **epilogue fusion** (bias + activation folded into the
  microkernel's write-back), so a `linear → bias → relu` runs in one pass.
- 256-bit AVX for the *general* (non-GEMM) vectorizer. Cranelift cannot legalize a 256-bit `f32x8`
  value (verified — pinned as a tripwire test), so the elementwise vectorizer is 128-bit + unrolling;
  the GEMM family already gets true AVX2/FMA via the runtime microkernel. Closing the general case
  needs a raw-AVX emitter or a future Cranelift.
- Execution of explicit `f32x8`-typed values; broader tensor-op lowering (conv, softmax) with fusion.
- Structs/enums, slices, multi-dimensional indexing `a[i, j]`, and a minimal stdlib.
- GPU device codegen (PTX/AMDGPU), autodiff — designed-for, explicitly deferred.

## Known limitations / sharp edges

- `bf16` is now real 2-byte storage at bf16 precision (round-to-nearest-even), f32 compute; `f16`
  still promotes to `f32`. On this AVX2 box (no AVX-512-BF16) a bf16 *GEMM* would widen to f32 and
  match f32 throughput — a memory-footprint feature, not a FLOP/s win — so it is left at the proven
  correctness path (`tests/run/matmul_bf16.mer`), not a tuned bf16 kernel.
- The *general* vectorizer emits 128-bit SIMD (Cranelift's vector ISA rejects 256-bit `f32x8` —
  verified empirically on Cranelift 0.124). Compute-bound *elementwise* kernels therefore use 2× the
  FMA ports they could; 4× unrolling and auto-parallelism recover throughput, and the vectorized
  **transcendentals still beat scalar `libm` ~2.5–3×**. Breaking 256-bit needs a hand-written AVX2
  path (how the GEMM family already gets 256-bit — a true AVX2/FMA runtime microkernel). The loop
  vectorizer assumes distinct array parameters do not alias.
- Array *length* in a type must be an integer literal (symbolic/`const`-expression lengths fall back
  to an opaque pointer), but matmul *dimensions* may be runtime values — a runtime-dimension matmul
  still dispatches to the GEMM kernel.
- No bounds checking on array indexing (manual memory is a decided constraint).
- `mem2reg` promotes only scalar integer/float slots; arrays, pointers, and address-taken locals
  stay in memory (the interpreter and `cse`/`dse` handle those directly).
