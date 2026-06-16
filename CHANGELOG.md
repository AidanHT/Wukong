# Changelog

All notable changes to Mercury are documented here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows semantic versioning.

## [Unreleased]

### Added
- **Front-end**: lexer (with `@attributes` and error recovery), recursive-descent + Pratt parser,
  AST with a pretty-printer, and `--emit=tokens|ast`.
- **Types & semantics**: the shared type vocabulary (`mercury_types`), name resolution, type
  checking, and **compile-time shape checking** for tensors (rank/dimension unification, symbolic
  dims), with errors `E0501`/`E0502`.
- **Middle-end**: block-parameter SSA MIR, a builder, a pretty-printer, and a verifier with a
  `MirLevel` invariant; AST → MIR lowering (alloca-per-local).
- **Optimizer**: a fixpoint pass manager with `simplify` (constant folding + algebraic identities +
  self-comparison folding), `simplify-cfg` (constant-branch folding + unreachable-block pruning),
  `dce`, `cse` (local value numbering with load forwarding), and `dse` (dead-store elimination),
  wired across `-O0..-O3`.
- **Back-ends**: a zero-dependency MIR interpreter (`--run`) and a textual LLVM-IR emitter
  (`--emit=llvm-ir`, plus `--emit=obj|exe` via `clang` when present).
- **Intrinsics**: `print`/`println` (captured stdout) and `assert` (traps on false).
- **Runtime**: a bump `Arena` allocator and a deterministic `parallel_for`.
- **Diagnostics**: rustc-style renderer, a stable error-code catalog with `--explain <CODE>`, and
  `--error-format=json` (JSON Lines).
- **Tooling & tests**: end-to-end run-suite with `// EXPECT-*` directives, an opt-level differential
  test (`-O0` vs `-O1/-O2/-O3`), per-stage `--emit` smoke tests, the `mercury_bench` optimizer
  report, a GitHub Actions CI (fmt + clippy + test on Linux & Windows), and the language guide and
  internals docs.

### Notes
- Tensors, SIMD vectors, and the parallel/GPU surface parse and type/shape-check today; full
  execution of those paths and native LLVM linking are in progress.
