# `tests/imports/` — loader unit-test fixtures

**6** flat `.wk` files for `crates/wukong_driver/src/loader.rs` unit tests only:

- `cycle_a.wk`, `cycle_b.wk`
- `diamond_root.wk`, `diamond_b.wk`, `diamond_c.wk`, `diamond_d.wk`

No subdirectories. The e2e harnesses in `crates/wukongc/tests/` **ignore** this directory — they
call `load_imports` only through real programs under `run/` and `fail/`.

These are user-written `.wk` programs testing the import graph, but they are **not** part of the
`cargo test -p wukongc` run/fail gates.
