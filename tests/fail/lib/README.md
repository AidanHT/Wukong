# `tests/fail/lib/` — import support (not fixtures)

Two support modules for compile-fail import tests:

- `badlib.wk` — deliberately ill-typed; imported by `fail/import_type_error.wk`. **Keep the offending
  `let` on line 6** — `imported_file_diagnostics_carry_their_own_path` pins that line number.
- `dup.wk` — imported twice by `fail/import_duplicate.wk`.

**Not counted as fixtures.** Non-recursive `read_dir` on `tests/fail/` never enters `lib/`.
