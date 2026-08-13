# wukong_codegen_llvm

Textual LLVM IR emitter for Wukong. **Inspection path only** — not the native fast path (that is
`wukong_codegen_cranelift`). Nothing here assembles, links, or runs code.

## Role in the pipeline

`wukong_driver` calls `emit_llvm_ir` directly when the user passes `--emit=llvm-ir`. The emitted
text goes to stdout; turning it into a binary would require `clang`/`llc` on PATH — an out-of-process
step this crate never performs.

```
MIR (post-opt)
  → emit_llvm_ir(program, interner) → textual .ll module on stdout
```

## Design choices

- **Textual IR, not libLLVM** — emits a `String` of LLVM assembly. No `inkwell`, no `llvm-sys`, no
  build-time LLVM dependency. The same MIR lowering would feed an in-process inkwell builder; this
  crate chose the fragile-free path for the Windows + MinGW host.
- **Phi bridging** — Wukong MIR uses block-parameter SSA (Cranelift/MLIR style); LLVM uses phi
  nodes. The emitter maps each non-entry block parameter to a `phi` fed by predecessor edge args.
- **`LlvmBackend`** — implements the `wukong_backend::Backend` seam, but the live dispatch path is
  `emit_llvm_ir`, not `Backend::compile`.

## Dependencies

| crate | role |
|---|---|
| `wukong_mir` | MIR types and program structure |
| `wukong_span` | symbol interner |
| `wukong_backend` | `Backend` trait + `Artifact` |

Dev-dependencies (`wukong_parser`, `wukong_sema`, `wukong_mir_build`, `wukong_opt`) support the
crate's unit tests only.

## Layout

| path | contents |
|---|---|
| [`src/lib.rs`](src/lib.rs) | entire backend — single-file crate |
