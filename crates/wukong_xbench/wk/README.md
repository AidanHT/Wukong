# `wukong_xbench` Wukong peer sources

`.wk` programs and C/Rust peer sources used by `wukong_xbench`. Unlike `tests/run/`, these are **not**
exercised by `cargo test` — the harness generates and compiles them at runtime.

A language, sema, MIR-lowering, or recognizer change can break every peer here while `cargo test`
stays green. After such changes, smoke:

```sh
cargo run -p wukong_xbench --release -- saxpy
```

## Layout

| directory | consumed by |
|---|---|
| `general/` | `src/general.rs` — structure-tax blocks, focal loss, S6 scan |

The main elementwise/GEMM kernels in `src/main.rs` embed their `.wk` source as string literals
rather than reading files from this tree.
