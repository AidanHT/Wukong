# `tests/fail/lib/` — import support (not fixtures)

Support modules for compile-fail import tests:

- `badlib.wk` — deliberately ill-typed; imported by `fail/import_type_error.wk`. **Keep the offending
  `let` on line 6** — `imported_file_diagnostics_carry_their_own_path` pins that line number.
- `dup.wk` — exports `clash`, which `fail/import_duplicate.wk` both selects and defines (E0300).
- `privlib.wk` — a private `secret` named through an alias by `fail/import_private.wk` (E0312).
- `amb_a.wk`, `amb_b.wk` — both export `pick`, used unqualified by `fail/import_ambiguous.wk`
  (E0313).
- `geom_assoc.wk` — a type `Pt` with an associated `norm`, called bare by
  `fail/import_assoc_fn_bare.wk` (E0320, whose help must say `Pt`, not `lib.geom_assoc.Pt`).

**Not counted as fixtures.** Non-recursive `read_dir` on `tests/fail/` never enters `lib/`.
