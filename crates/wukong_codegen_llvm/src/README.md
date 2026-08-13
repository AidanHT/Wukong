# `src/` module index

Single-file crate.

| file | contents |
|---|---|
| `lib.rs` | `LlvmBackend` (`Backend` impl), `emit_llvm_ir`, MIR→LLVM lowering (`emit_function`, phi bridging via `edge_args`, static string literals, per-op lowering) |
