# wukongc

The Wukong compiler command-line driver. Thin wrapper around `wukong_driver::compile` — parses
argv into `Options`, spawns compilation on a 256 MiB worker thread (deep nesting safety), and maps
exit codes to `ExitCode`.

`USAGE` in `src/main.rs` is the **only** user-facing flag documentation. Do not invent flags; verify
against `main.rs` and `wukong_driver::Options` when in doubt.

## CLI surface (`src/main.rs` → `wukong_driver::Options`)

| flag | notes |
|---|---|
| `--emit=<stage>` | `tokens`, `ast`, `mir-high`, `mir` (alias `mir-low`), `grad`, `llvm-ir`, `obj`, `exe` — **default `exe`** |
| `--run` | compile and execute |
| `--backend=<b>` | `interp`, `native`, `gpu`, `gpu-native` — **default `interp`** |
| `-O0\|-O1\|-O2\|-O3` | **default `-O0`**; `-O3` byte-identical to `-O2` |
| `--grad-of=<fn>` | default `loss` |
| `--grad-wrt=<i,..>` | default: every `Ptr` param; under `--train`, all but the last |
| `--train` | fwd→bwd→optimizer loop |
| `--train-steps=<n>` | default 100 |
| `--train-lr=<f>` | default 0.01 |
| `--train-opt=<sgd\|adamw>` | default sgd |
| `--train-seed=<n>` | deterministic buffer init |
| `-o <path>` | only `--emit=obj` and `--emit=exe` write a file; other stages print to stdout |
| `--color=<auto\|always\|never>` | default auto |
| `--error-format=<human\|json>` | default human |
| `--explain <CODE>` | print extended error explanation, then exit |
| `-h/--help`, `-V/--version` | |

**Rejected combinations** (in `parse_args`, not the driver):

- `--run` with an explicit `--emit=<stage>`
- `-o` with `--run` or with any stage other than `obj`/`exe`

**Exit codes:** `0` OK, `1` COMPILE_ERROR, `2` UNIMPLEMENTED (also usage errors), `3` IO_ERROR.

## Examples

```sh
cargo run -q -p wukongc -- --help
cargo run -q -p wukongc -- --run examples/fib.wk
cargo run -q -p wukongc -- --run --backend=native -O2 tests/run/tensor_matmul.wk
cargo run -q -p wukongc -- --emit=mir -O2 tests/run/transcendental_kernel.wk
```

## Features

```toml
gpu = ["wukong_driver/gpu"]   # enables --backend=gpu and --backend=gpu-native
```

Without `gpu`, those backends error clearly — never silently fall back to CPU.

## Testing

```sh
cargo test -p wukongc --test run      # opt invariance + interp-vs-native differential
cargo test -p wukongc                 # all integration tests
```

See `tests/README.md` for each gate. **Recognized kernels are GATE-BLIND** — no differential gate can
see whether a recognizer fired; prove dispatch with `--emit=mir` and `grep wukong_`.
