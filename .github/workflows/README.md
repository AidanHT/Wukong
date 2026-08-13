# GitHub Actions workflows

This directory contains the repository's CI definitions. Only one workflow is checked in:
[`ci.yml`](ci.yml) (`name: CI`).

## Triggers

- `push` to `main`
- all `pull_request` events

Workflow-wide env: `CARGO_TERM_COLOR: always` and `RUSTFLAGS: -D warnings`.

## `test` — build & test (no LLVM)

Runs on a two-OS matrix (`ubuntu-latest`, `windows-latest`) with `fail-fast: false`.

Steps, in order:

1. **Checkout** (`actions/checkout@v4`)
2. **Rust toolchain** — stable via `dtolnay/rust-toolchain@stable`, with `rustfmt` and `clippy`
3. **Cargo cache** — keyed on `Cargo.lock` and runner OS
4. **`cargo fmt --all -- --check`**
5. **`cargo clippy --workspace --all-targets`** — without `--features gpu`
6. **`cargo test --workspace`** — the default workspace test gate; does **not** enable the `gpu`
   feature, so `wukong_codegen_gpu` and GPU-gated driver code are not compiled here
7. **`cargo test -p wukong_runtime --release --lib`** — release-profile bit-exactness between
   AVX2 kernels and scalar twins (both matrix OSes)
8. **Smoke-run examples** — `wukongc --run` on `examples/fib.wk`, `examples/dot.wk`, and
   `examples/saxpy_array.wk`
9. **Optimizer effectiveness + equivalence** —
   `cargo run -p wukong_bench --release -- tests/run examples bench/kernels`

## `gpu-check` — GPU type-check + clippy + device-free laws (no device)

Runs on `ubuntu-latest` only. Uses a separate cache key (`-gpu` suffix) because the feature graph
differs from the `test` job.

This job exists because `cargo test --workspace` in `test` never passes `--features gpu`. Without
it, the entire `wukong_codegen_gpu` crate (and GPU-gated arms of `wukong_driver` / `wukongc`)
could fail to compile while CI stayed green. No CUDA toolkit or GPU device is required:
`cudarc` is built with dynamic loading.

Steps:

1. **Checkout** and **Rust toolchain** (stable + `clippy`)
2. **Cargo cache** (GPU-specific key)
3. **`cargo check --features gpu --all-targets`**
4. **`cargo clippy --features gpu --all-targets`** — lints the full GPU surface under
   `-D warnings`
5. **`cargo test -p wukong_driver --features gpu --lib`** — device-free dispatch laws and
   routing rules in `gpu_accel::tests`; end-to-end GPU tests skip (not fail) when no device is
   reachable unless `WUKONG_GPU_REQUIRED=1`

Locally, the two-command gate mirrors this split: plain `cargo test` for the default workspace,
then `cargo check --features gpu --all-targets` for the GPU compile surface.
