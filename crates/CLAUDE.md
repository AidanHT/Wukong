# crates/ — `mercury_` crate workspace

This is the `mercury_` crate workspace; dependencies flow strictly downward (no cycles).

## Crates (dependency order, low-level first)

- [`mercury_span`](mercury_span/CLAUDE.md) — Source spans, source map, and string interning primitives.
- [`mercury_diag`](mercury_diag/CLAUDE.md) — Diagnostic model, error-code catalog, and terminal/JSON renderers.
- [`mercury_lexer`](mercury_lexer/CLAUDE.md) — Source text to flat token stream plus diagnostics.
- [`mercury_ast`](mercury_ast/CLAUDE.md) — Data-only AST: parser builds, sema annotates via NodeId side tables.
- [`mercury_parser`](mercury_parser/CLAUDE.md) — Recursive-descent + Pratt parser: lexer tokens to AST.
- [`mercury_types`](mercury_types/CLAUDE.md) — Type vocabulary: scalars, SIMD vectors, shape-typed tensors, layouts.
- [`mercury_sema`](mercury_sema/CLAUDE.md) — Name resolution, type checking, and tensor shape checking.
- [`mercury_mir`](mercury_mir/CLAUDE.md) — Typed block-structured SSA IR: types, builder, printer, verifier.
- [`mercury_mir_build`](mercury_mir_build/CLAUDE.md) — Lowers type-checked Mercury AST into MIR (alloca-per-local).
- [`mercury_opt`](mercury_opt/CLAUDE.md) — Pass manager, CFG/dominator analyses, and SSA MIR transforms.
- [`mercury_backend`](mercury_backend/CLAUDE.md) — Backend trait + Artifact enum: the MIR-to-execution/emission seam.
- [`mercury_interp`](mercury_interp/CLAUDE.md) — Zero-dep tree-walking MIR interpreter; default backend and differential oracle.
- [`mercury_codegen_cranelift`](mercury_codegen_cranelift/CLAUDE.md) — Native backend via Cranelift (no LLVM): JIT + object/exe.
- [`mercury_codegen_llvm`](mercury_codegen_llvm/CLAUDE.md) — LLVM backend: lowers MIR to textual LLVM IR.
- [`mercury_runtime`](mercury_runtime/CLAUDE.md) — Minimal runtime: bump Arena allocator and sequential parallel_for.
- [`mercury_driver`](mercury_driver/CLAUDE.md) — Orchestrates the compile pipeline; owns SourceMap/Interner and honors --emit.
- [`mercuryc`](mercuryc/CLAUDE.md) — CLI binary parsing args and delegating compilation to `mercury_driver`.
- [`mercury_bench`](mercury_bench/CLAUDE.md) — Benchmark harness: optimizer effectiveness, interpreter timing, equivalence gate.
- [`mercury_xbench`](mercury_xbench/CLAUDE.md) — Cross-language benchmark: Mercury vs C vs Rust (see `BENCHMARKS.md`).

## Dependency edges

- `mercury_span`, `mercury_runtime` — depend on nothing.
- `mercury_diag` -> `mercury_span`.
- `mercury_lexer` -> `mercury_span`, `mercury_diag`.
- `mercury_ast` -> `mercury_span`.
- `mercury_parser` -> `mercury_span`, `mercury_diag`, `mercury_lexer`, `mercury_ast`.
- `mercury_types` -> `mercury_span`.
- `mercury_sema` -> `mercury_ast`, `mercury_types`, `mercury_span`, `mercury_diag`.
- `mercury_mir` -> `mercury_span`, `mercury_types`.
- `mercury_mir_build` -> `mercury_span`, `mercury_diag`, `mercury_ast`, `mercury_types`, `mercury_mir`, `mercury_sema`.
- `mercury_opt` -> `mercury_mir`, `mercury_span`.
- `mercury_backend` -> `mercury_mir`, `mercury_span`.
- `mercury_interp` -> `mercury_mir`, `mercury_span`, `mercury_backend`.
- `mercury_codegen_llvm` -> `mercury_span`, `mercury_mir`, `mercury_backend`.
- `mercury_codegen_cranelift` -> `mercury_span`, `mercury_mir`, `mercury_backend`, `mercury_runtime`, `cranelift-*`.
- `mercury_driver` -> `mercury_span`, `mercury_diag`, `mercury_lexer`, `mercury_ast`, `mercury_parser`, `mercury_sema`, `mercury_mir`, `mercury_mir_build`, `mercury_backend`, `mercury_interp`, `mercury_opt`, `mercury_codegen_llvm`, `mercury_codegen_cranelift`.
- `mercuryc` -> `mercury_driver`.
- `mercury_xbench` -> `mercury_span`, `mercury_parser`, `mercury_sema`, `mercury_mir_build`, `mercury_opt`, `mercury_codegen_cranelift`.
- `mercury_bench` -> `mercury_span`, `mercury_parser`, `mercury_sema`, `mercury_mir`, `mercury_mir_build`, `mercury_opt`, `mercury_interp`.

Each crate dir has its own CLAUDE.md with the per-crate detail.
