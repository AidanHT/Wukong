# Contributing to Wukong

Thanks for your interest! Wukong is built incrementally, one small, tested change at a time.

## Ground rules

- **Everything builds and tests without LLVM.** The front-end, optimizer, and the MIR interpreter
  must keep working with plain `cargo test`; native codegen is Cranelift (pure Rust, no toolchain).
  The textual LLVM IR backend (`--emit=llvm-ir`) is also pure Rust and always built — `clang`/`llc`
  are only needed to compile the *emitted* IR, never by the build or tests. No backend may become a
  hard toolchain dependency of the default workspace; the GPU backend is the only opt-in one
  (`--features gpu`).
- **Each change is small and self-contained**, with tests, and leaves the tree green.
- **No new warnings.** CI builds with `-D warnings` and runs `cargo fmt --check` and
  `cargo clippy --all-targets`.

## Before you push

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
```

## Where things live

See [`docs/internals.md`](docs/internals.md) for the crate layering and pipeline. In short:

- Front-end: `wukong_lexer`, `wukong_parser`, `wukong_ast`.
- Types & checking: `wukong_types`, `wukong_sema` (including shape checking).
- Middle-end: `wukong_mir`, `wukong_mir_build`, `wukong_opt`.
- Back-ends: `wukong_interp` (default oracle), `wukong_codegen_cranelift` (Cranelift JIT — the
  native fast path), `wukong_codegen_llvm` (textual LLVM IR), `wukong_codegen_gpu` (PTX GPU backend,
  `--features gpu`). Runtime microkernels: `wukong_runtime`.
- Training: `wukong_autodiff` (reverse-mode autodiff, a MIR→MIR transform).
- Driver/CLI: `wukong_driver`, `wukongc`. Harness: `wukong_bench`, `wukong_xbench`.

## Adding a diagnostic

Give it a stable code in the right range (`E01xx` lexer … `E05xx` shapes, `C0xxx` codegen) and add
an entry to `wukong_diag::catalog` so `wukongc --explain <CODE>` documents it.

## Adding an optimizer pass

Implement `wukong_opt::Pass`, register it in `PassManager::standard` at the appropriate `-O` level,
and add a test asserting both an effect (e.g. instruction-count reduction) and that results are
unchanged. The opt-level differential test (`crates/wukongc/tests/run.rs`) will also exercise it.

## Adding a language feature

Prefer to land it end to end: parse → type/shape-check → lower → run via the interpreter, with an
`examples/` or `tests/run/` program that demonstrates it. Update `docs/language-guide.md` and its
maturity legend.
