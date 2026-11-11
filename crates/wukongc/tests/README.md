# `wukongc` integration tests

End-to-end gates driving the real `wukongc` binary against the repo-root `tests/` corpus. These
prove the compiler works as a user would invoke it — zero mocked stages.

## Primary gates

| file | what it proves |
|---|---|
| `run.rs` | `tests/run/*.wk` compile and run; `optimization_is_observationally_invariant` (`-O0` == `-O1/-O2/-O3` on interp); `native_matches_interpreter` (interp == native at `-O0`/`-O2`); `native_optimization_is_observationally_invariant` |
| `fail.rs` | `tests/fail/*.wk` rejected with pinned `// EXPECT-CODE:`; catalogue coverage both directions (a `W` code counts as exercised by a `tests/warn` fixture) |
| `warn.rs` | `tests/warn/*.wk` compile and produce exactly the lint diagnostics their `// EXPECT-WARN:` / `// EXPECT-ERROR:` / `// EXPECT-NONE` lines list (the whole JSON stream, in order, plus the exit status); `corpus_is_warning_free` (`tests/run` + `examples`) is `#[ignore]`d until the corpus is brought to zero warnings |
| `determinism.rs` | compiler is a pure function across fresh processes — MIR bytes, object bytes, JSON diagnostic order — and of the optimizer's thread count (`WUKONG_OPT_THREADS` 1 / automatic / 4 in the MIR and object sweeps, 1 vs 4 at `-O1`) |
| `exe.rs` | `--emit=exe` AOT path matches `--run` (linked exe vs interpreter oracle) |
| `recognize.rs` | the recognizers are **switchable**: one witness per cascade head loses its kernel under `WUKONG_NO_RECOGNIZE`, `@generic` removes it for one function only (a MIR count over the `lang_generic_*` twin pairs), the program still prints the same bytes on both backends, autodiff refuses to run under the instrument, and a **corpus sweep** holds every `tests/run` program that loses a `wukong_*` call under the instrument to its generic lowering (`-O2` interp, `-O2` native, `-O0` interp), with the dependent-program and distinct-kernel counts pinned as floors; the **emittable-surface census** fails when a `wukong_*` symbol interned anywhere in `wukong_mir_build` is emitted by no `tests/run` program (`NEVER_EMITTED_ALLOWLIST`, empty, holds the reasoned exceptions) |
| `fmt.rs` | `wukongc fmt` through the binary: in-place rewrite then a clean `--check` and unchanged `--run` output, `--check` lists and writes nothing, a syntax error exits 1 and a refusal exits 2 with the file untouched, stdin/stdout, usage (2) and IO (3) exits, a symlinked file rewritten at its target (skipped where links cannot be made), a read-only file refused, permission bits kept (Unix); and the **MIR-identity gate**: `tests/run` (with `lib/`) and `examples/` formatted in a scratch copy, every root program's `--emit=mir-high -O0` (and the examples' `--emit=mir -O2`) byte-identical before and after, exit code included, all mismatches reported at once |
| `filetests.rs` | `tests/filetests/**/*.{mir,wk}` (the one recursive scan): each file's `RUN:` lines (`wukongc ... %s \| filecheck %s`, `not`, `diff`, `%t` scratch files, `REQUIRES:` features) through the real binary, and its `CHECK:` lines matched by the in-tree FileCheck in `filecheck/` (LLVM's `CHECK`/`-NEXT`/`-SAME`/`-NOT`/`-DAG`/`-LABEL`/`-EMPTY`/`-COUNT`, `{{regex}}`, `[[VAR]]` captures; no new dependency). Pins what each optimizer pass does to the MIR, which no differential gate can see; a failure prints the directive, the nearest candidate input line and the whole input. Also the matcher's and the RUN-line shell's unit tests |
| `recognizer_census.rs` | `tests/recognizer-census.txt`: the `wukong_*` calls every `tests/run` fixture emits at `--emit=mir -O2`, byte-pinned (`WUKONG_UPDATE_CENSUS=1` rewrites it and fails on purpose) |
| `parallel_lane.rs` | the **forced-parallel lane**: every census program that dispatches a `_parallel` kernel runs on native with the runtime's fork gates forced open (`WUKONG_PAR_MIN_WORK=1`, `WUKONG_GEMM_PAR_MIN_MACS=1`, `WUKONG_VELEM_PAR_MIN=1`, `WUKONG_PACK_PAR_MIN_KB=0`) and must print exactly what default-gate native and the interpreter print — corpus fixtures are far below the gates, so without it no gate ever ran a `_parallel` kernel's split |

Focused gate for lowering/optimizer/backend work:

```sh
cargo test -p wukongc --test run
```

## Other tests

| file | role |
|---|---|
| `emit.rs` | intermediate `--emit` stages produce expected artifacts |
| `doc.rs` | `--emit=doc` prints `tests/doc/geometry.md` (hand-written) for `tests/doc/geometry.wk`, composes with a stage list and `-o <stem>` (`<stem>.md`), needs a clean parse, and succeeds on every example and `tests/run` program; `--doctest` passes every kind of example in `geometry.wk` (run, `no_run`, `compile_fail,<code>`, own `main`, `ignore`) on interp and native at `-O0`/`-O2`, reports each way an example in `tests/doc/doctest_failures.wk` fails (with the doc comment's line and column for a compile error) and exits 1, runs the examples' doc tests, and refuses the flags it cannot honour |
| `examples.rs` | `examples/*.wk` through the CLI: each declared class holds, interp == native, `-O` invariance, and every runnable example's in-file `EXPECT-EXIT` / `EXPECT-OUT*` header (the `tests/run` grammar via `common/mod.rs`) matches its output |
| `corpus_lint.rs` | static corpus checks: every directive parses, every `EXPECT-OUT-APPROX` tolerance is in `DECLARED_TOLERANCES` with a cited source, the comparator's own mutation controls |
| `diag_json.rs` | `--error-format=json` schema v1 against the real binary: the v0 prefix is unchanged and the v1 fields (end positions, byte range, `suggestions`, `children`, `rendered`, `version`) are present |
| `explain.rs` | `--explain` lists every code, prints each entry, and every catalogue `bad:`/`good:` example is compiled under `--emit=check` (bad yields exactly its code first, good checks clean) |
| `gpt2_config.rs` | GPT-2 config fixture |
| `parallel_frame_budget.rs` | a `@parallel` loop whose body declares 20 MB of scratch (a slice region and an array-only whole-function loop) runs serially on native at `-O0`/`-O2` and prints the hand-derived lanes; outlined, its frame overflowed a 16 MiB runtime worker stack (native-only: the interpreter takes over ten seconds per run at that size) |
| `tensor_parity.rs` | vector-kernel recipe parity (host-gated) |
| `training_data.rs` | `--train-init` / `--train-data` / `--train-save` / `--train-clip` / `--train-warmup` / `--train-schedule` through the binary: hand-derived trajectories and saved weights, interp == native, and the file-shape refusals |
| `train_lm.rs` | `tests/run/train_tiny_lm.wk` (embedding + tanh layer + projection + cross-entropy) trains on both backends at `-O0`/`-O2`, matching an f64 reference trajectory and decreasing strictly |

## GATE-BLIND recognizers

A recognizer fires identically on interp and native at every `-O` level (both call the same
`wukong_runtime` symbol), so `native_matches_interpreter` and `optimization_is_observationally_invariant`
cannot see whether it fired or whether the kernel is wrong.

Proof requires inspecting MIR dispatch:

```sh
cargo run -q -p wukongc -- --emit=mir -O2 tests/run/transcendental_kernel.wk | grep wukong_
```

Fixtures must print **every lane** (never a reduction). Expected values come from scalar loop
semantics, not recorded compiler output.

`recognize.rs` is the automated form of that grep, and the reason it can be automated is that the same
program can now be built *without* the recognizers (`WUKONG_NO_RECOGNIZE`, `@generic`). A new
recognizer arm belongs inside one of the four gated probe blocks in `wukong_mir_build`, or it becomes a
kernel this gate cannot see either.

## Fixture format (`tests/run/`)

```
// RUN: <extra cli args>        # default: --run
// EXPECT-EXIT: <n>
// EXPECT-OUT: <line>           # one per expected stdout line, in order
```

`tests/fail/` uses `// EXPECT-CODE: Ennnn` (directory is `tests/fail`, not `tests/compile-fail`).
