# tests — the `.wk` fixture corpus

**This is the language product surface.** Every fixture here is a user-written Wukong program driven
through the real compiled `wukongc` binary (spawned as a process via `CARGO_BIN_EXE_wukongc`, never
library calls). Exit codes, stdout bytes, and stderr are the observables. The fixtures live here; the
harnesses that consume them live in `crates/wukongc/tests/` and reach this directory as
`CARGO_MANIFEST_DIR/../../tests/…`.

`bench/` and `tools/` drive Rust launchers and measurement scripts — not the language thesis.

## Directories

| Path | Count | Role |
|------|-------|------|
| `run/` | **363** top-level `*.wk` | Must compile **and** execute. Checked against in-file directives, then re-run across backends and opt levels by the differential gates. |
| `run/lib/` | 2 (`mathlib.wk`, `deep.wk`) | Support modules for `run/import_multi.wk`. **Not fixtures** — harnesses use non-recursive `read_dir`. |
| `fail/` | **113** top-level `*.wk` | Must be **rejected at compile time** with a specific stable diagnostic code. Directory is `tests/fail`, **not** `tests/compile-fail`. |
| `fail/lib/` | 2 (`badlib.wk`, `dup.wk`) | Support for import fixtures. **Not fixtures.** |
| `warn/` | 12 top-level `*.wk` | Must **compile** and produce exactly the warnings (lints) their `// EXPECT-WARN:` / `// EXPECT-ERROR:` / `// EXPECT-NONE` lines list, in order. |
| `imports/` | 6 | Flat import-graph fixtures consumed **only** by `crates/wukong_driver/src/loader.rs` unit tests. E2e harnesses ignore this directory. |
| `filetests/` | 37 `*.mir` / `*.wk`, **recursive** (`opt/`, `mir-text/`, `verify/`, `cli/`) | FileCheck-style tests of what the compiler *emits* — chiefly what each optimizer pass does to the MIR. Each file carries its own `RUN:` and `CHECK:` lines. The only directory scanned recursively; an `Inputs/` subdirectory holds files a `RUN:` line reads, not tests. |
| `doc/` | 3 `*.wk` + 1 `*.md` | Doc-comment fixtures for `crates/wukongc/tests/doc.rs`: `geometry.wk` (every documentable construct; doc examples of every kind that pass) with its hand-written `--emit=doc` page `geometry.md`, `doctest_failures.wk` (one doc example per way of failing), and `doctest_broken_file.wk` (a file that does not compile on its own, so none of its examples runs). Formatted by the formatter's corpus test; not run by `run.rs`. |

See each subdirectory's `README.md` for format details and landmines.

## Fixture format — `run/`

Directives are leading `//` comments, matched by `str::strip_prefix` after `trim_start`, anywhere in
the file (`run.rs::parse_directives`). Convention is to put them at the very top, above the `module`
header.

```wk
// RUN: --run
// EXPECT-EXIT: 0
// EXPECT-OUT: 8
// EXPECT-OUT: 0.25
module tests.run.my_fixture

// Prose: what defect this pins, and where the expected values come from.
fn main() -> i32 { … }
```

- `// RUN: <args>` — **replaces** the whole arg list (default `["--run"]`); it does not append.
- `// EXPECT-EXIT: <n>` — required process exit code.
- `// EXPECT-OUT: <line>` — one expected stdout line, in order. **All** lines together must equal stdout exactly.

## Fixture format — `fail/`

```wk
// EXPECT-CODE: E0501
module tests.fail.array_index_oob
```

- `// EXPECT-CODE: <code>` — **required**; exactly **one** per file; the first match wins.
- `// EXPECT-COUNT: <n>` — the build reports exactly `n` errors (error or bug severity, any file).
- `… //~ <code> [<code> …]` at the end of a line — an error with each code has its primary span on
  that line (`//~^` / `//~v`: the line above / below, one line per mark). Once a fixture has any,
  the errors in its own file must be exactly its annotations; an imported file's are only counted.
- Harness drives `--error-format=json --emit=mir` and asserts the first diagnostic is an error with
  `<code>`. `WUKONG_BLESS=1 cargo test -p wukongc --test fail` rewrites every fixture's count and
  `//~` annotations from the real output (never `EXPECT-CODE`) and fails on purpose: review the
  diff, then rerun without it. Every fixture is annotated.
- Never renumber or recycle a code. `E0001` is retired.

## Fixture format — `warn/`

```wk
// FLAGS: -D warnings
// EXPECT-ERROR: W0001
module tests.warn.deny_warnings_flag
```

- `// EXPECT-WARN: <code>` / `// EXPECT-ERROR: <code>` — one per expected diagnostic, in emission
  order (source order); the harness compares the **whole** JSON stream, so an extra warning fails.
- `// EXPECT-NONE` — the program must compile with no diagnostic at all.
- `// FLAGS: <args>` — extra arguments, appended to `--error-format=json --emit=mir`.
- Exit status must be 1 when an `EXPECT-ERROR` is listed, 0 otherwise.

## Fixture format — `filetests/`

```text
; RUN: wukongc --emit=mir --passes=licm %s | filecheck %s
; CHECK-LABEL: fn invariant_product(
; CHECK:       bb0(
; CHECK:         [[AB:v[0-9]+]] = fmul v2, v3
; CHECK:       bb2:
; CHECK-NOT:     fmul v2, v3
fn invariant_product(v0 ptr, v1 ptr, v2 f32, v3 f32, v4 i64) -> void { ... }
```

- `RUN:` — a pipeline run in order with `pipefail`: `wukongc <args>`, `filecheck <file>
  [--check-prefix=P] [--input-file=f] [--implicit-check-not=pat]`, `diff <a> <b>`, `not <cmd>`
  (a clean failure, never a crash); `>`, `<`, `2>`, `2>&1`; `%s` the file, `%S` its directory,
  `%t` a scratch path. A trailing `\` continues onto the next `RUN:` line. Every file needs a
  `RUN:` line that checks something (`filecheck` or `diff`).
- `CHECK:` / `-NEXT:` / `-SAME:` / `-NOT:` / `-DAG:` / `-LABEL:` / `-EMPTY:` / `-COUNT-<n>:` —
  LLVM FileCheck's semantics, with `{{regex}}`, `[[VAR:regex]]`, `[[VAR]]` and `[[@LINE+k]]`.
- `REQUIRES: vec-kernels` (or `windows`, `unix`, `x86_64`) skips the file elsewhere, by name.
- Expectations come from the pass's semantics, then are confirmed — never pasted from output.
  `CONTRIBUTING.md` ("Writing a filetest") has the rules.

## Gates over this corpus

`crates/wukongc/tests/run.rs` — `run_suite`, `optimization_is_observationally_invariant`,
`native_matches_interpreter`, `native_optimization_is_observationally_invariant`.

`crates/wukongc/tests/fail.rs` — `compile_fail_suite`, `lexer_and_parser_diagnostics_are_reachable`,
`every_catalogued_code_is_exercised`, `imported_file_diagnostics_carry_their_own_path`.

`crates/wukongc/tests/warn.rs` — `warn_suite` over `warn/`; `corpus_is_warning_free` lints every
`run/` and `examples/` program (and the modules they import): the corpus is warning-free, so a new
fixture must be too; `an_imported_files_parser_lint_carries_its_own_path` compiles
`imports/lint_semi_root.wk`.

`crates/wukongc/tests/determinism.rs` — fresh-process byte identity for MIR, object, and diagnostic
emission; the MIR and object sweeps also vary the optimizer's thread count (`WUKONG_OPT_THREADS=1`,
automatic, `4`), and an `-O1` arm compares 1 against 4 threads over the corpus.

`crates/wukongc/tests/exe.rs` — AOT linked exe matches `--run` on four curated fixtures.

`crates/wukongc/tests/emit.rs` — frontend and lowering stages succeed over `examples/` + `run/`.

`crates/wukongc/tests/filetests.rs` — every file under `filetests/`, through the real binary and
the in-tree FileCheck (`crates/wukongc/tests/filecheck/`); also the matcher's and the harness's
own unit tests. `WUKONG_FILETESTS=<substring>` runs a subset.

`crates/wukongc/tests/recognize.rs` — the **only** gate that can see a recognizer at all: one witness
fixture per cascade head loses its kernel under `WUKONG_NO_RECOGNIZE`, the `lang_generic_*` pairs keep
exactly one kernel call each (`@generic` is per function), the programs still print the same bytes on
both backends, and autodiff refuses to run under the instrument.

## Corpus-authoring rules (non-negotiable)

1. **Print every lane, never a reduction.** A sum lets a partially-correct buffer pass.
2. **`EXPECT-OUT` values are scalar semantics, computed independently — never recorded compiler output.**
3. **Use unambiguous decimal rendering** on both backends (exact floats or scale-and-truncate to int).
4. **Recognizer fixtures are GATE-BLIND.** Grepping MIR is the only proof:
   `cargo run -q -p wukongc -- --emit=mir -O2 tests/run/transcendental_kernel.wk | grep wukong_`
5. **File-I/O fixtures share the process CWD.** Use unique `io_test_<type>.bin` names.
6. Always declare at least `// EXPECT-EXIT:` — a fixture with neither exit nor stdout checks is vacuous.

## Running

```sh
cargo test -p wukongc                      # every gate over this corpus
cargo test -p wukongc --test run           # opt-invariance / native-agreement
cargo test -p wukongc --test fail          # compile-fail + catalogue coverage
cargo test -p wukongc --test filetests     # what each optimizer pass emits (FileCheck)
cargo run -q -p wukongc -- --run -O2 tests/run/foo.wk
```

No LLVM needed for the default `--backend=interp`. Do not build `--release` while iterating on
correctness.

## LANDMINES

- **`// RUN:` is honoured by `run_suite` ONLY.** Other gates construct their own arg list and ignore the directive.
- **Non-recursive `read_dir` on `extension == "wk"`.** Put imported support modules in `lib/` subdirectories.
- **The compile-fail directory is `tests/fail`, not `tests/compile-fail`.**
- **`cargo test | grep X && git commit` commits on failure** — run tests unpiped.
- **`cargo test` does not build the `gpu` feature.** Both `cargo test` and `cargo check --features gpu --all-targets` are required before commit.
