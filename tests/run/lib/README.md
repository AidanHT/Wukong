# `tests/run/lib/` — import support (not fixtures)

Support modules imported by the multi-file `run/` programs:

- `mathlib.wk`, `deep.wk` — imported by `run/import_multi.wk` (an import cycle between them)
- `geom.wk`, `stats.wk` — imported by `run/import_namespaces.wk` (private same-named `helper`s,
  `as` aliases, a selective import, qualified struct/enum/const access)

Every item another file uses is declared `pub`: each file is its own module and imports are not
transitive.

**Not counted as fixtures.** Every harness scans `tests/run/` with a non-recursive `read_dir`, so
files in `lib/` are never collected. Drop new imported support modules here, not in the parent
directory.
