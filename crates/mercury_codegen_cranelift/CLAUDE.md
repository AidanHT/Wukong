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
- Runtime symbols (`mercury_rt_print_i64`/`_f64`/`_assert`, `mercury_parallel_for`) are bound to Rust
  fns in the JIT and left as imports in the object (resolved by the driver's C runtime). A global run
  lock serialises JIT runs that share the stdout-capture buffer.
- Div-by-zero is guarded to yield 0 (no trap), matching the interpreter oracle.
