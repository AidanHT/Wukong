# tests — end-to-end / golden harness

`.mer` fixtures driven through the **real** compiled `mercuryc` binary (spawned as a process via
`CARGO_BIN_EXE_mercuryc`, not library calls), so true exit codes, stdout, and stderr are exercised.
The harness code lives in `crates/mercuryc/tests/` (`run.rs`, `fail.rs`, `emit.rs`); the fixtures
live here at the repo root and are reached via `CARGO_MANIFEST_DIR/../../tests/...`.

## Directories
- `run/*.mer` — programs that must compile **and** run successfully. Each is executed (default
  `--run`) and its stdout + exit code checked against in-file directives. Also re-run across opt
  levels for differential opt-invariance, and smoke-emitted through every lowering stage.
- `fail/*.mer` — programs that must be **rejected at compile time** with a specific stable
  diagnostic code (lexer `E01xx`, parser `E02xx`, name-res `E03xx`, types `E04xx`, shapes `E05xx`).

## Directive vocabulary
Directives are leading `//`-comment lines (parsed by prefix; the harness `trim_start`s each line).

`run/` fixtures (parsed in `run.rs::parse_directives`):
- `// RUN: <args>` — full CLI arg list passed to `mercuryc` (replaces, does not append; **default
  `--run`**). e.g. `// RUN: --run`.
- `// EXPECT-EXIT: <n>` — required process exit code (integer). `main`'s return value becomes the
  exit code; a failed `assert` traps with exit `1`.
- `// EXPECT-OUT: <line>` — one expected stdout line, matched **in order**; all `EXPECT-OUT` lines
  together must equal the program's stdout exactly. Omit to skip the stdout check.

`fail/` fixtures (parsed in `fail.rs::expected_code`):
- `// EXPECT-CODE: <Ennnn>` — the stable diagnostic code that must appear in stderr. **Required** —
  a fixture without it panics the suite.

## What each test asserts
- `run.rs::run_suite` — for every `run/*.mer`: spawns with the `RUN:` args, asserts exit ==
  `EXPECT-EXIT` and stdout lines == `EXPECT-OUT` lines.
- `run.rs::optimization_is_observationally_invariant` — **differential hardening**: runs each
  program at `-O0` and at `-O1`/`-O2`/`-O3`; (exit code, stdout) must be **identical** at every
  level. Optimization must never change observable behavior.
- `fail.rs::compile_fail_suite` — drives each `fail/*.mer` with `--error-format=json --emit=mir`
  (so sema runs), asserts the process exits non-zero and stderr contains `"code":"<EXPECT-CODE>"`.
- `emit.rs::frontend_stages_succeed_for_all_sources` — `--emit=tokens` and `--emit=ast` succeed
  with non-empty output for all `examples/` + `run/` programs.
- `emit.rs::lowering_stages_succeed_for_run_suite` — `--emit=mir-high|mir|llvm-ir` at `-O0` and
  `-O2` succeed for every `run/` program with no `internal compiler error` in stderr (guards the
  MIR verifier, which runs under `--emit=mir`, and the LLVM-IR pretty-printer).

## Adding a test case
- **Passing program**: drop `foo.mer` in `tests/run/`, add `// EXPECT-EXIT:` and any `// EXPECT-OUT:`
  lines (and a `// RUN:` line only if you need flags other than `--run`). It is auto-discovered (dir
  scan, sorted) — no registration. It must compile through every `--emit` stage and behave
  identically at all opt levels, else `emit`/invariance tests fail.
- **Compile-fail program**: drop `foo.mer` in `tests/fail/` with a single `// EXPECT-CODE: Ennnn`.

## Running
- All e2e/golden tests: `cargo test -p mercuryc`
- One suite: `cargo test -p mercuryc --test run` (or `--test fail`, `--test emit`)
- One test fn: `cargo test -p mercuryc --test run optimization_is_observationally_invariant`

No LLVM needed: the default interpreter backend drives `--run`, so the whole harness builds and
passes with plain `cargo test`.
