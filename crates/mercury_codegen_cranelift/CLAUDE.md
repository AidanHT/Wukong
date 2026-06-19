# mercury_codegen_cranelift

Native code generation via **Cranelift** — no LLVM toolchain required. Lowers fully-lowered (Low)
MIR to Cranelift IR and either JIT-compiles and runs it in-process (the fast execution path and a
differential peer of the interpreter) or emits a host object file. This is the backend that makes
Mercury's runtime competitive; see `BENCHMARKS.md`.

## Layout
- `src/lib.rs` — entire backend: type/size/condcode mappers, `FnTranslator` (per-function MIR→CLIF),
  module driving (`populate_module`), `JitProgram`/`JitModuleHandle`, `jit_compile`/`jit_run`/
  `jit_module`/`emit_object`, and the runtime `rt_*` symbols.
- `src/backend.rs` — `CraneliftBackend` implementing the `mercury_backend::Backend` trait.
- `src/tests.rs` — correctness + differential tests (native vs interpreter) incl. the vectorizer and
  `@parallel`.
- `src/fuzz.rs` — randomized **full-buffer** differential fuzzer (interp vs native over identical
  random buffers, asserting the whole output buffer is bit-exact): the f32 kernel battery via
  `mercury_interp::run_kernel_f32` + a `f64`-reference value check, and the int8 `u8×i8→i32` GEMM via
  `run_kernel_i8` across the kernel's K-chunk boundaries.

## Key types & entry points
- `jit_run(program, entry, interner) -> (i64, Vec<u8>)` — compile and run `entry`, capturing stdout
  (the native counterpart to `mercury_interp::run_with_output`; used by the differential gate).
- `jit_compile` → `JitProgram` (`run()` captures stdout; `call()` is raw, for timing).
- `jit_module` → `JitModuleHandle::func_ptr(sym)` — raw code pointers for callers that know the ABI
  (e.g. `mercury_xbench` timing a `(*const f32, …)` kernel).
- `emit_object(program, interner) -> Vec<u8>` — host object; the driver links it with a small C
  runtime for `--emit=exe`.
- `FnTranslator` — RPO block walk; `lower_inst` dispatches each `Op` to `builder.ins()`. `cl_type`
  maps `MirType` (incl. `Vec(elem,n)` → `lane.by(n)`); `size_of` scales `gep`/stack slots.

## Connects to
Upstream: `mercury_mir` (`Program`/`Function`/`Op`/`Terminator`/`MirType`), `mercury_span`
(`Interner`/`Symbol`), `mercury_backend` (`Backend`/`Artifact`), `mercury_runtime`
(`mercury_parallel_for` bound as a JIT symbol), and the `cranelift-*` crates (codegen, frontend,
module, jit, object, native). Dev-deps: parser/sema/mir_build/opt/interp (for differential tests).
Downstream: `mercury_driver` (`--backend=native`, `--emit=obj|exe`), `mercury_bench`,
`mercury_xbench`.

## Gotchas
- **PIC differs by product:** the JIT needs `is_pic=false`, the object emitter `true` —
  `make_isa(pic)` is parametrized. JIT entry points must take no parameters.
- **Vectors are 128-bit only.** Cranelift does not legalize `f32x8` ("Unexpected SSA-value type"), so
  the vectorizer (in `mercury_mir_build`) targets `VEC_REG_BYTES=16` and unrolls for throughput.
- **Vector memory is unaligned-safe**, but vector `Select` lowers to `bitselect` after bitcasting the
  compare mask to the value vector type (Cranelift has no scalar-cond vector `select`).
- **Float typing must be consistent in the incoming MIR.** Cranelift's verifier rejects `fadd` on
  mismatched widths; the front-end coerces operands to the result type and the interpreter rounds
  `f32` in `f32`, so the two backends stay bit-identical. Don't emit loosely-typed float MIR.
- **FMA is real here.** `Op::Fma` lowers to `ins().fma` (a hardware `vfmadd` on FMA3 hosts, scalar
  or 128-bit vector). The front-end contracts float `x + y*z` into it; the interpreter mirrors it
  with `mul_add`, and the two agree bit-for-bit (gated by `fma_contraction_is_bit_exact`). This is
  why `mercury_xbench` gives gcc `-ffp-contract=fast` — both sides fuse.
- Runtime symbols (`mercury_rt_print_i64`/`_f64`/`_assert`, `mercury_parallel_for`, the GEMM
  microkernels `mercury_sgemm`/`_parallel`/`_nt`/`_nt_parallel`/`_nt_epi` (the `_nt_epi` fused-epilogue
  one takes a bias pointer + an `act` code: identity/ReLU/GELU/SiLU, bias may be a null pointer), the
  int8 GEMM `mercury_i8gemm_nt`/`_parallel`, the 256-bit elementwise transcendental
  `mercury_vmath_f32` (op codes exp/log/tanh/sigmoid/relu/silu/gelu/elu/leaky_relu/softplus/mish/selu),
  the streaming affine+activation `mercury_velem_f32` (8 args: 3 ptr + i64 + 3 f32 + i64) and Horner
  `mercury_vhorner_f32` (ptr,ptr,i64,ptr,i64), the reduction `mercury_sreduce_f32`/`_parallel`, the
  fused row-wise norm `mercury_norm_f32`/`_parallel` and its affine sibling `mercury_norm_affine_f32`
  (4 ptr + 4 i64; gamma/beta may be a null pointer)) are bound to Rust fns in the JIT and left as
  imports in the object (resolved by the driver's C runtime). A global run lock serialises JIT runs
  that share the stdout-capture buffer.
- **A runtime call may return a value.** Most (`mercury_sgemm*`, `mercury_vmath_f32`) are void, but
  `mercury_sreduce_f32[_parallel]` returns an **f32** — its `lower_call` arm binds the call result
  (`inst_results(call).first().copied()`) instead of returning `None`, and its signature carries a
  `returns` entry. Mirror that when adding any value-returning runtime symbol.
- **GEMM / vmath calls.** The recognizers in `mercury_mir_build` emit `Op::Call` to a `mercury_sgemm*`
  symbol `(a,b,c,m,k,n,beta)`, the int8 `mercury_i8gemm_nt[_parallel] (a,b,c,m,k,n)` (3 ptr + 3 i64,
  void — element type is irrelevant to the ABI), or `mercury_vmath_f32 (x,out,n,op)`; `lower_call`
  recognizes those names (`RT_SGEMM*`, `RT_I8GEMM_NT*`, `RT_VMATH`) and emits a direct call to the
  declared import. When adding a runtime symbol, wire it in *all* of: the `RT_*` const, `RtFuncs` + its
  `declare_function`, the per-func `rt_refs` insert, `lower_call`, and `builder.symbol(...)` in
  **both** `jit_compile` and `jit_module`.
- Div-by-zero is guarded to yield 0 (no trap), matching the interpreter oracle.
