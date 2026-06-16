# Mercury Roadmap & Known Limitations

Mercury is built openly and incrementally. This page is an honest snapshot of what works, what is
checked-but-not-executed, and what is planned — so expectations match reality.

## Works end to end (interpreter, `--run`)

- Modules, functions (including recursion and mutual recursion), and direct calls.
- `let`/`let mut`/`const`, shadowing, block-as-expression values.
- Integers (`i8..i64`, `u8..u64`, `usize`/`isize`), `bool`, and floats (`f16`/`bf16`/`f32`/`f64`,
  computed in `f64`).
- All arithmetic/comparison/bitwise/boolean operators, compound assignment, casts.
- `if`/`else`, `while`, `for … in a..b [step s]`.
- **Fixed-size arrays** `[T; N]`: literal/repeat init, indexed load/store, array parameters passed
  by base pointer (out-params). Real kernels run: dot product, SAXPY, flat GEMM, transpose, sort.
- Intrinsics `print`/`println`/`assert`.
- The optimizer (`-O0..-O3`), backed by CFG and dominator analyses: **mem2reg** (alloca → SSA),
  constant folding, algebraic simplification, CFG cleanup with block merging, dead/trivial
  block-parameter elimination, DCE, CSE with load forwarding, DSE, and **loop-invariant code
  motion**. Guarded by an `-O0`-vs-`-O{1,2,3}` differential test; on the benchmark kernels it removes
  ~45% of IR ops and runs ~1.5–2x faster than `-O0`.

## Checked but not yet executed

- **Shape-typed tensors** `Tensor[f32, M, N]`: parse and pass compile-time shape checking
  (`E0501`/`E0502`), the headline feature — but tensor *operations* are not yet lowered/run.
- **SIMD vectors** `f32x8` etc.: parse and type-check; vector ops are not yet executed.
- **Attributes** `@simd`/`@tile`/`@parallel`/`@align`/`@extern`/`@export`: parse and validate; their
  optimizer/runtime consumers are in progress.

## Planned

- Lowering and execution of tensor ops with fusion, tiling, and vectorization.
- SIMD vector execution in the interpreter and via LLVM `<N x T>`.
- `@parallel for`/`reduce` wired to the `mercury_runtime` thread pool.
- Native linking of the LLVM backend (textual IR + `clang`) into runnable executables, and
  interpreter-vs-LLVM differential tests.
- Structs/enums, slices, multi-dimensional indexing `a[i, j]`, and a minimal stdlib.
- GPU device codegen (PTX/AMDGPU), autodiff — designed-for, explicitly deferred.

## Known limitations / sharp edges

- Floats are computed in `f64` in the interpreter; `f16`/`bf16` are storage types, not yet reduced
  precision at runtime.
- Array length must be an integer literal; symbolic/`const`-expression lengths fall back to an opaque
  pointer.
- No bounds checking on array indexing (manual memory is a decided constraint).
- `mem2reg` promotes only scalar integer/float slots; arrays, pointers, and address-taken locals
  stay in memory (the interpreter and `cse`/`dse` handle those directly).
