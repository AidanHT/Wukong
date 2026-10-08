# `tests/imports/` — loader unit-test fixtures

**10** flat `.wk` files for `crates/wukong_driver/src/loader.rs` unit tests only:

- `cycle_a.wk`, `cycle_b.wk` — an import cycle (each file loads once)
- `diamond_root.wk`, `diamond_b.wk`, `diamond_c.wk`, `diamond_d.wk` — a diamond (the leaf loads once)
- `ns_a.wk`, `ns_b.wk` — two modules with colliding PRIVATE `helper`s and a colliding public
  `a_val`: per-module namespaces, `pub` enforcement (E0312), glob ambiguity (E0313), `as` aliases
  and `{x, y}` selection
- `ns_leak.wk` — calls a root item it never imported (imports are per file, not transitive)
- `ns_ext.wk` — repeats the root's `extern "C"` declaration (identical declarations merge)

Most tests synthesize the ROOT source in memory (as `tests/imports/virtual_root.wk`, which does not
exist on disk) and import these files from it.

No subdirectories. The e2e harnesses in `crates/wukongc/tests/` **ignore** this directory — they
call `load_imports` only through real programs under `run/` and `fail/`.

These are user-written `.wk` programs testing the import graph, but they are **not** part of the
`cargo test -p wukongc` run/fail gates.
