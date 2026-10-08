# `wukongc` integration tests

End-to-end gates driving the real `wukongc` binary against the repo-root `tests/` corpus. These
prove the compiler works as a user would invoke it — zero mocked stages.

## Primary gates

| file | what it proves |
|---|---|
| `run.rs` | `tests/run/*.wk` compile and run; `optimization_is_observationally_invariant` (`-O0` == `-O1/-O2/-O3` on interp); `native_matches_interpreter` (interp == native at `-O0`/`-O2`); `native_optimization_is_observationally_invariant` |
| `fail.rs` | `tests/fail/*.wk` rejected with pinned `// EXPECT-CODE:`; catalogue coverage both directions (a `W` code counts as exercised by a `tests/warn` fixture) |
| `warn.rs` | `tests/warn/*.wk` compile and produce exactly the lint diagnostics their `// EXPECT-WARN:` / `// EXPECT-ERROR:` / `// EXPECT-NONE` lines list (the whole JSON stream, in order, plus the exit status); `corpus_is_warning_free` (`tests/run` + `examples`) is `#[ignore]`d until the corpus is brought to zero warnings |
| `determinism.rs` | compiler is a pure function across fresh processes — MIR bytes, object bytes, JSON diagnostic order |
| `exe.rs` | `--emit=exe` AOT path matches `--run` (linked exe vs interpreter oracle) |
| `recognize.rs` | the recognizers are **switchable**: one witness per cascade head loses its kernel under `WUKONG_NO_RECOGNIZE`, `@generic` removes it for one function only (a MIR count over the `lang_generic_*` twin pairs), the program still prints the same bytes on both backends, autodiff refuses to run under the instrument, and a **corpus sweep** holds every `tests/run` program that loses a `wukong_*` call under the instrument to its generic lowering (`-O2` interp, `-O2` native, `-O0` interp), with the dependent-program and distinct-kernel counts pinned as floors |

Focused gate for lowering/optimizer/backend work:

```sh
cargo test -p wukongc --test run
```

## Other tests

| file | role |
|---|---|
| `emit.rs` | intermediate `--emit` stages produce expected artifacts |
| `examples.rs` | `examples/*.wk` smoke through the CLI |
| `diag_json.rs` | `--error-format=json` schema v1 against the real binary: the v0 prefix is unchanged and the v1 fields (end positions, byte range, `suggestions`, `children`, `rendered`, `version`) are present |
| `explain.rs` | `--explain` lists every code, prints each entry, and every catalogue `bad:`/`good:` example is compiled under `--emit=check` (bad yields exactly its code first, good checks clean) |
| `gpt2_config.rs` | GPT-2 config fixture |
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
