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
  precision** (interpreter and native agree exactly); `f16`/`bf16` promote to `f32`; `f64` is full.
- All arithmetic/comparison/bitwise/boolean operators, compound assignment, casts.
- `if`/`else` (statement and value position), `while`, `for … in a..b [step s]`.
- **Fixed-size arrays** `[T; N]`: literal/repeat init, indexed load/store, array parameters passed
  by base pointer (out-params). Real kernels run: dot, SAXPY, GEMM, matmul, ReLU, clamp, transpose.
- **SIMD auto-vectorization**: straight-line elementwise loops (incl. branchy ones via
  if-conversion) lower to 128-bit vector ops, 4×-unrolled, with a scalar remainder — automatically,
  on the native backend. saxpy/poly/relu/relu6/matmul-inner vectorize.
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

- Cache tiling for matmul/GEMM, and fusing chains *under* `@parallel` (fusion and `@parallel`
  compose only loosely today).
- 256-bit AVX codegen (Cranelift is 128-bit only today; AVX throughput is approximated via unrolling).
- Reduction vectorization (`dot` etc.) with horizontal reduce.
- Execution of explicit `f32x8`-typed values; tensor-op lowering with fusion/tiling.
- Structs/enums, slices, multi-dimensional indexing `a[i, j]`, and a minimal stdlib.
- GPU device codegen (PTX/AMDGPU), autodiff — designed-for, explicitly deferred.

## Known limitations / sharp edges

- `f16`/`bf16` are storage types promoted to `f32` at runtime, not yet reduced precision.
- Native SIMD is 128-bit (Cranelift's vector ISA); compute-bound kernels trail gcc's 256-bit AVX on
  a single thread (4× unrolling narrows but does not erase the gap). Auto-parallelism more than makes
  up for it across cores. The loop vectorizer assumes distinct array parameters do not alias.
- Array length must be an integer literal; symbolic/`const`-expression lengths fall back to an opaque
  pointer.
- No bounds checking on array indexing (manual memory is a decided constraint).
- `mem2reg` promotes only scalar integer/float slots; arrays, pointers, and address-taken locals
  stay in memory (the interpreter and `cse`/`dse` handle those directly).
