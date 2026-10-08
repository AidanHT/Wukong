# wukong_codegen_llvm

Textual LLVM IR emitter for Wukong. **Inspection path only** — not the native fast path (that is
`wukong_codegen_cranelift`). Nothing here assembles, links, or runs code.

## Role in the pipeline

`try_emit_llvm_ir` returns a diagnostic for unsupported call ABIs; the compatibility
`emit_llvm_ir` wrapper panics on those errors. The emitted
text goes to stdout; turning it into a binary would require `clang`/`llc` on PATH — an out-of-process
step this crate never performs.

```
MIR (post-opt)
  → try_emit_llvm_ir(program, interner) → textual .ll module on stdout
```

## Design choices

- **Textual IR, not libLLVM** — emits a `String` of LLVM assembly. No `inkwell`, no `llvm-sys`, no
  build-time LLVM dependency. The same MIR lowering would feed an in-process inkwell builder; this
  crate chose the fragile-free path for the Windows + MinGW host.
- **Phi bridging** — Wukong MIR uses block-parameter SSA (Cranelift/MLIR style); LLVM uses phi
  nodes. The emitter maps each non-entry block parameter to a `phi` fed by predecessor edge args.
- **`LlvmBackend`** — implements the `wukong_backend::Backend` seam, but the live dispatch path is
  `try_emit_llvm_ir`, not `Backend::compile`.

The module implements the interpreter's semantics: `f16`/`bf16` values compute in `float`
registers and round onto their grid only at constants, casts, loads and stores; booleans are bytes
in memory; float `!=` is `fcmp une`; integer division and shifts carry the oracle's guards.

`print`/`assert` lower to the `wukong_rt_*` ABI the Cranelift object calls, `wukong_*` kernels are
called with the runtime's C ABI (integers widened to `i64`, a literal-zero optional pointer as
`ptr null`), and the general vectorizer's kernels are **defined** in the module from their recipe
(`src/vec_kernels.rs`), so a module references only symbols the `--emit=exe` runtime provides.
Conflicting signatures for one foreign symbol are rejected because a faithful ABI adapter is not
implemented. The module names the host `target triple` and leaves the datalayout to its consumer.

Regression coverage includes exact float encodings, vector negation/complement, identity casts,
math intrinsic results, duplicate-edge phis, integer shift/division guards, and external ABI
diagnostics. To export the narrow test modules for an independent LLVM verifier, set
`WUKONG_LLVM_TEST_OUTPUT` to an output directory before running
`cargo test -p wukong_codegen_llvm --lib closeout_tests::`.

Floating-to-integer conversions use LLVM's saturating intrinsics (NaN becomes zero and
out-of-range values clamp), and FMA uses `llvm.fma` to require a single rounding — on `f32`
registers for `f16`/`bf16` operands, exactly as the interpreter evaluates it.

For independent scalar/vector conversion and exact-FMA execution checks, set
`WUKONG_LLVM_SEMANTICS_OUTPUT` to a dedicated directory, run
`cargo test -p wukong_codegen_llvm --lib semantic_tests::`, then run
`python crates/wukong_codegen_llvm/tools/verify_semantics.py <directory>`.
The script requires an already installed llvmlite and checks actual LLVM JIT results,
including cancellation that distinguishes FMA from separate multiply/add instructions.

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
