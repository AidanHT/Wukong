# `tests/run/lib/` — import support (not fixtures)

Two support modules imported by `run/import_multi.wk`:

- `mathlib.wk`
- `deep.wk`

**Not counted as fixtures.** Every harness scans `tests/run/` with a non-recursive `read_dir`, so
files in `lib/` are never collected. Drop new imported support modules here, not in the parent
directory.
