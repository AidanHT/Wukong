# mercuryc

The `mercuryc` command-line binary: parses CLI args into a `mercury_driver::Options` and delegates the whole compile to `mercury_driver::compile`. The user-facing front door at the end of the pipeline.

## Layout
- `src/main.rs` — the entire binary (~230 lines): `USAGE` string, `main`, `run_compile`, `parse_args`, `print_explanation`. No other modules.
- `tests/run.rs` — e2e suite: runs `<repo>/tests/run/*.mer` through the real binary, defaulting to `--run`, checking stdout/exit against in-file `// EXPECT-OUT:` / `// EXPECT-EXIT:` / `// RUN:` directives; also asserts -O1/-O2/-O3 are observationally identical to -O0.
- `tests/fail.rs` — compile-fail suite: drives `<repo>/tests/fail/*.mer` with `--error-format=json --emit=mir`, asserts the `// EXPECT-CODE:` `E…` code appears in stderr JSON and exit is non-zero.
- `tests/emit.rs` — smoke-tests every `--emit` stage: `tokens`,`ast` for all examples + run-suite; `mir-high`,`mir`,`llvm-ir` for run-suite at -O0 and -O2 (checks no verifier "internal compiler error" leaks).

## Key types & entry points
- `main` (`src/main.rs`) — returns `ExitCode`; on parse success calls `run_compile(opts)` and converts its `i32` to the exit code via `as u8`. `Ok(None)` (help/version/explain) exits 0; `Err` prints `error:` + `USAGE` to stderr and exits 2.
- `run_compile` (`src/main.rs`) — runs `mercury_driver::compile` on a dedicated worker thread with a large (256 MiB) stack, so deep-but-bounded nested input is rejected by the parser's `E0209` nesting limit rather than overflowing the default ~1 MB main-thread stack with no diagnostic (mirrors how `rustc` runs its front-end on a dedicated stack).
- `parse_args` (`src/main.rs`) — hand-rolled arg parser (no clap). `Ok(Some(Options))` = compile, `Ok(None)` = printed help/version/explanation so exit 0, `Err(String)` = usage error. All option types (`Options`, `EmitStage`, `ErrorFormat`, `BackendKind`, `GradOptions`, `TrainOpt`) come from `mercury_driver`; `EmitStage::parse` and `mercury_driver::explain` do the real work. Beyond the core flags it fills `--backend=<interp|native|gpu|gpu-native>` plus the autodiff surface — `--emit=grad`, `--grad-of`/`--grad-wrt`, and the `--train`/`--train-steps`/`--train-lr`/`--train-opt`/`--train-seed` family — into `Options::{backend, grad}`.
- `print_explanation` (`src/main.rs`) — backs `--explain <CODE>`; looks the code up via `mercury_driver::explain`, errors on an unknown code.

## Connects to
Upstream (depends on): `mercury_driver` (its ONLY dependency). Downstream: end users / CI / the test fixtures under `<repo>/tests/` and `<repo>/examples/`.

## Gotchas
- Test fixtures live at the REPO ROOT (`<repo>/tests/run`, `/tests/fail`, `<repo>/examples`), NOT under this crate. Tests reach them via `CARGO_MANIFEST_DIR/../../tests/...`.
- Tests spawn the compiled binary via `CARGO_BIN_EXE_mercuryc` (real process), not by calling library functions — they exercise true exit codes and stdout/stderr.
- `--color=auto` (and the default when `--color` is omitted) probes `stderr.is_terminal()`. The default is re-applied in a second pass AFTER the arg loop guarded by `!args.iter().any(|a| a.starts_with("--color="))`, so an explicit `--color=` always wins.
- `emit_explicit` is computed but unused (`let _ = emit_explicit;`); `--run` does not force or clear `opts.emit`. Run-vs-emit mode is decided inside `mercury_driver::compile`, not here.
- `parse_args` is purely syntactic: bad file paths, unknown error codes (beyond `explain` lookup), and all sema/shape errors surface from the driver. Keep flag docs in `USAGE` in sync with `Options`/`EmitStage` when the driver gains options.
