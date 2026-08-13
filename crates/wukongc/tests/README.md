# `wukongc` integration tests

End-to-end gates driving the real `wukongc` binary against the repo-root `tests/` corpus. These
prove the compiler works as a user would invoke it — zero mocked stages.

## Primary gates

| file | what it proves |
|---|---|
| `run.rs` | `tests/run/*.wk` compile and run; `optimization_is_observationally_invariant` (`-O0` == `-O1/-O2/-O3` on interp); `native_matches_interpreter` (interp == native at `-O0`/`-O2`); `native_optimization_is_observationally_invariant` |
| `fail.rs` | `tests/fail/*.wk` rejected with pinned `// EXPECT-CODE:`; catalogue coverage both directions |
| `determinism.rs` | compiler is a pure function across fresh processes — MIR bytes, object bytes, JSON diagnostic order |
| `exe.rs` | `--emit=exe` AOT path matches `--run` (linked exe vs interpreter oracle) |
| `recognize.rs` | the recognizers are **switchable**: one witness per cascade head loses its kernel under `WUKONG_NO_RECOGNIZE`, `@generic` removes it for one function only (a MIR count over the `lang_generic_*` twin pairs), the program still prints the same bytes on both backends, and autodiff refuses to run under the instrument |

Focused gate for lowering/optimizer/backend work:

```sh
cargo test -p wukongc --test run
```

## Other tests

| file | role |
|---|---|
| `emit.rs` | intermediate `--emit` stages produce expected artifacts |
| `examples.rs` | `examples/*.wk` smoke through the CLI |
| `gpt2_config.rs` | GPT-2 config fixture |
| `tensor_parity.rs` | vector-kernel recipe parity (host-gated) |

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
