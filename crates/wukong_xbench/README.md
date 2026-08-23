# wukong_xbench

Cross-language benchmark: Wukong (Cranelift JIT) vs C (gcc) vs Rust (rustc) vs optional peers
(oneMKL, `matrixmultiply`, PyTorch in the `model` section). Measures generated-kernel performance;
it is not the product.

**Critical landmine:** this harness compiles its `.wk` peer programs at **runtime**. A language,
sema, or recognizer change can break every generated kernel without breaking `cargo test`. Smoke
after any language change:

```sh
cargo run -p wukong_xbench --release -- saxpy
```

(Do not rely on `cargo test` alone for xbench health.)

## What it does

For each kernel, builds the same computation several ways and times them through one identical Rust
loop over the same buffers:

- **Wukong:** `.wk` source generated and JIT-compiled in process
- **C/C++:** compiled to a shared library at runtime (`CC`/`CXX` override, default gcc/g++)
- **Rust:** `rustc -C opt-level=3 -C target-cpu=native`

Reports compile time and ns/call per column. A missing toolchain prints `n/a` and continues.

Every battery row additionally reports a **`wk(gen)` recognition-off line** (master plan §7.2): the
same source lowered with every kernel recognizer skipped (`LowerOptions::recognize = false`, the
`WUKONG_NO_RECOGNIZE` / `@generic` lowering), its ratio vs C and vs the recognizer-on column, and
the MIR dispatch census for **both** lowerings — recognized kernels are gate-blind, so the census is
the only proof of what a number measured. The whole run pins its timing thread (`XBENCH_PIN=off`
disables; see the CPU PINNING block in `src/main.rs`).

Optional first CLI argument is a substring filter over kernel/section names (e.g. `saxpy`).

## Sections

| section | module | focus |
|---|---|---|
| elementwise / GEMM / … | `src/main.rs` | recognizer-dialect kernels |
| general code | `src/general.rs` | ML-engineer spellings that may miss recognizers |
| model | `src/model.rs` | 12-layer forward vs PyTorch |

The `general` section prints a **dispatch census** (`grep wukong_` equivalent) because recognized
kernels are GATE-BLIND to `cargo test`.

## Usage

```sh
cargo run -p wukong_xbench --release --             # all kernels
cargo run -p wukong_xbench --release -- saxpy       # smoke filter
cargo run -p wukong_xbench --release -- general     # general-code section only
cargo run -p wukong_xbench --release -- compare     # tests/compare/ timed vs its committed C peers
cargo run -p wukong_xbench --release -- sota-check  # gate: sweep vs bench/sota/FLOOR.md (exit code!)
```

`compare` times timing-scale twins of the `tests/compare/` fixtures (recognition-off, `@generic`,
census-witnessed) against the committed C peers' kernels on the round-interleaved pinned protocol.
`sota-check` is part 11 of the language close-out verification gate: it re-runs the in-scope sweep
(general + compare) and exits non-zero when a ratio falls below its recorded floor in
`bench/sota/FLOOR.md`, when a program leaves the sweep, when a bar peer fails to compile, or when
the run is not evaluable (battery / AC+charging / power-state change / failed oracle). It requires
AC+full and an idle box; `WUKONG_SOTA_FLOOR` overrides the floor-file path.

**Timing requires `--release`.** Peers need gcc/g++/rustc on PATH; Wukong needs no external LLVM.

## Testing

No dedicated `cargo test` suite — the harness is the instrument. Wukong output is checked against
the interpreter oracle (bit-exact) inside each timed row.
