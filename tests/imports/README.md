# `tests/imports/` — loader unit-test fixtures

**17** flat `.wk` files for `crates/wukong_driver/src/loader.rs` unit tests (and one root that
`crates/wukongc/tests/warn.rs` compiles):

- `cycle_a.wk`, `cycle_b.wk` — an import cycle (each file loads once)
- `diamond_root.wk`, `diamond_b.wk`, `diamond_c.wk`, `diamond_d.wk` — a diamond (the leaf loads once)
- `ns_a.wk`, `ns_b.wk` — two modules with colliding PRIVATE `helper`s and a colliding public
  `a_val`: per-module namespaces, `pub` enforcement (E0312), glob ambiguity (E0313), `as` aliases
  and `{x, y}` selection
- `ns_leak.wk` — calls a root item it never imported (imports are per file, not transitive)
- `ns_ext.wk` — repeats the root's `extern "C"` declaration (identical declarations merge)
- `ns_meth.wk`, `ns_meth_ext.wk` — a type and, in another module, its `impl`: an import used only
  to call a method is not unused (W0008)
- `ns_meth_prelude.wk` — a module that only imports `ns_meth_ext.wk`: its import is kept when ANY
  file calls the method, since an `impl` is program-wide (W0008)
- `ns_shape.wk`, `ns_shape_impl.wk` — a trait and type, and in another module `impl Shape for Sq`:
  an import that only satisfies a `T: Shape` bound is not unused (W0008)
- `lint_semi.wk`, `lint_semi_root.wk` — an imported module whose `let` misses its `;`: the W0030
  must carry the imported file's path; `warn.rs` compiles `lint_semi_root.wk` end to end

Most tests synthesize the ROOT source in memory (as `tests/imports/virtual_root.wk`, which does not
exist on disk) and import these files from it.

No subdirectories. The e2e harnesses in `crates/wukongc/tests/` **ignore** this directory — they
call `load_imports` only through real programs under `run/` and `fail/` — except the one
named test in `warn.rs` that compiles `lint_semi_root.wk`.

These are user-written `.wk` programs testing the import graph, but they are **not** part of the
`cargo test -p wukongc` run/fail gates.
