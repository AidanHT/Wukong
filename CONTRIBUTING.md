# Contributing to Mercury

Thanks for your interest! Mercury is built incrementally, one small, tested change at a time.

## Ground rules

- **Everything builds and tests without LLVM.** The front-end, optimizer, and the MIR interpreter
  must keep working with plain `cargo test`. LLVM lives behind `--features llvm` and may never be a
  hard build dependency of the default workspace.
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

- Front-end: `mercury_lexer`, `mercury_parser`, `mercury_ast`.
- Types & checking: `mercury_types`, `mercury_sema` (including shape checking).
- Middle-end: `mercury_mir`, `mercury_mir_build`, `mercury_opt`.
- Back-ends: `mercury_interp` (default), `mercury_codegen_llvm` (textual IR).
- Driver/CLI: `mercury_driver`, `mercuryc`. Harness: `mercury_bench`.

## Adding a diagnostic

Give it a stable code in the right range (`E01xx` lexer … `E05xx` shapes, `C0xxx` codegen) and add
an entry to `mercury_diag::catalog` so `mercuryc --explain <CODE>` documents it.

## Adding an optimizer pass

Implement `mercury_opt::Pass`, register it in `PassManager::standard` at the appropriate `-O` level,
and add a test asserting both an effect (e.g. instruction-count reduction) and that results are
unchanged. The opt-level differential test (`crates/mercuryc/tests/run.rs`) will also exercise it.

## Adding a language feature

Prefer to land it end to end: parse → type/shape-check → lower → run via the interpreter, with an
`examples/` or `tests/run/` program that demonstrates it. Update `docs/language-guide.md` and its
maturity legend.
