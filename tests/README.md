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
| `imports/` | 6 | Flat import-graph fixtures consumed **only** by `crates/wukong_driver/src/loader.rs` unit tests. E2e harnesses ignore this directory. |

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
- Harness drives `--error-format=json --emit=mir` and asserts stderr contains `"code":"<code>"`.
- Never renumber or recycle a code. `E0001` is retired.

## Gates over this corpus

`crates/wukongc/tests/run.rs` — `run_suite`, `optimization_is_observationally_invariant`,
`native_matches_interpreter`, `native_optimization_is_observationally_invariant`.

`crates/wukongc/tests/fail.rs` — `compile_fail_suite`, `lexer_and_parser_diagnostics_are_reachable`,
`every_catalogued_code_is_exercised`, `imported_file_diagnostics_carry_their_own_path`.

`crates/wukongc/tests/determinism.rs` — fresh-process byte identity for MIR, object, and diagnostic
emission.

`crates/wukongc/tests/exe.rs` — AOT linked exe matches `--run` on four curated fixtures.

`crates/wukongc/tests/emit.rs` — frontend and lowering stages succeed over `examples/` + `run/`.

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
