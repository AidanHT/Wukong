# `wukong_diag` integration tests

| file | role |
|---|---|
| `catalog_coverage.rs` | **U6 gate** — scans every `"Ennnn"` / `"Cnnnn"` literal under `crates/` and asserts catalogue ↔ emit-site agreement in both directions |

Run with `cargo test -p wukong_diag` (not `--lib` alone — these live under `tests/`).
