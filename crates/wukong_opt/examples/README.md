# `wukong_opt` examples

Development tools for loop analysis and canonicalization — not part of the compiler pipeline.

| Example | Purpose |
| --- | --- |
| `loopdump` | Print `loop_info`'s view of one `.wk` file: natural loops, induction variables, trip counts, memory accesses, dependence verdicts. Can filter by function name. |
| `loopstats` | Corpus-wide census of loop structural properties (preheader, latch count, exit edges, trip count) over many `.wk` files or directories. |

Both compile a **single** source file with no `import` resolution (that lives in `wukong_driver`).
Fixtures that `import` support modules are skipped in `loopstats`.

```sh
cargo run -q -p wukong_opt --example loopdump -- tests/run/saxpy.wk [-O<n>] [fn-name-filter]
cargo run -q -p wukong_opt --example loopstats -- [-O<n>] <file-or-dir>...
```

Default optimization level is `-O2` (the level the vectorizer runs at). The same dump is available
inside a normal `wukongc` run via `WUKONG_DUMP_LOOPS=1`; these binaries exist because they can
resolve function *names*, which `optimize` cannot (the interner does not reach it).
