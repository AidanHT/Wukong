# `src/` module index

| file | contents |
|---|---|
| `lib.rs` | MIR→Cranelift lowering, `rt_*` runtime shims, `jit_run` / `jit_compile`, `emit_object` / `emit_object_timed`, module population, alias-region plumbing, `JitProgram` handle |
| `backend.rs` | `CraneliftBackend` — `Backend` trait impl wrapping `jit_run` (declared seam; live path is direct `jit_run` from `wukong_driver`) |
| `avx2.rs` | P4 raw-AVX2 emitter: `assemble_kernel`, `host_supports_kernels` → `wukong_mir::host_supports_vec_kernels`, Win64 ABI (rcx/rdx/r8), VEX.256 AVX2+FMA3 via `iced-x86` |
| `tests.rs` | JIT smoke tests, interpreter-vs-native differential on stdout/exit code, optimization invariance |
| `fuzz.rs` | Full-buffer kernel fuzzer: random inputs, bit-exact interp vs native over entire output buffers |
| `fuzz_grammar.rs` | Grammar-based whole-program fuzzer: random well-typed `.wk` through parse→sema→MIR→interp/native |
