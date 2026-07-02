# mercury_bench

Standalone binary `mercury-bench`: runs the full front-end + optimizer + interpreter on `.mer` files to quantify optimizer effectiveness (`-O0` vs `-O3`) and interpreter timing, doubling as an optimization-equivalence correctness gate. Tooling/CI harness; sits outside the compiler pipeline.

## Layout
- `src/main.rs` — arg parsing, per-file benchmarking, op counting, adaptive timing, report table, and the `-O0`-vs-`-O3` optimization-equivalence gate.
- `src/compile_time.rs` — in-process per-pass compile-time measurement (via `mercury_opt::optimize_timed`).
- `src/compile_vs.rs` — same-run compile-time comparison of `mercuryc` vs `gcc`/`g++`/`rustc`.

## Key types & entry points
- `main` (`src/main.rs`) — collects `.mer` files from CLI dirs (default `tests/run`), sorts, benchmarks each, prints a table + TOTAL/geomean row, and `exit(1)` if any program failed optimization equivalence.
- `bench_one(path)` (`src/main.rs`) — the core: `mercury_parser::parse_module` -> `mercury_sema::check` -> `mercury_mir_build::lower_program`, clone the program, `optimize(_, 0)` and `optimize(_, 3)`, verify both, run both via `mercury_interp::run` on entry `main`, compare results.
- `Outcome` (`src/main.rs`) — `Ran(Res)` / `Skipped(String)` / `Failed(String)`. The Skipped-vs-Failed distinction is the crate's whole point (see Gotchas).
- `Res` — `ops0/ops3` (MIR op counts) and `t0/t3` (per-run `Duration`).
- `count_ops` — total MIR ops = sum over every func's blocks of `insts.len() + 1` (the `+1` counts the block terminator).
- `time_run` — warms up twice, then doubles `reps` until elapsed >= 50ms (cap `reps >= 1<<22`), returns `elapsed / reps`.

## Connects to
Upstream (depends on): `mercury_span`, `mercury_parser`, `mercury_sema`, `mercury_mir`, `mercury_mir_build`, `mercury_opt`, `mercury_interp`. Downstream: none (leaf binary; consumed by CI / `cargo run -p mercury_bench --release -- <dirs>`).

## Gotchas
- Skipped != Failed. Parse/type/shape/lowering errors, and programs that error identically at both `-O0` and `-O3`, are `Skipped` (not regressions). A `Failed` (process exit 1) means the program lowered cleanly but the optimizer broke it: differing results, runs at only one level, or post-opt MIR fails `mercury_mir::verify::verify_function` (treated as failure when its diagnostic vec is non-empty). Do not "fix" a Failed by widening the Skipped net.
- Entry point is hardcoded to the interned symbol `main`; programs without a `main` fn will not run meaningfully.
- Result equality uses `==` on `mercury_interp::run` `Ok` values; the optimizer must be value-exact, not approximate.
- Final speedup is a geometric mean (`exp(mean(ln(speedup)))`), the correct average for ratios — not arithmetic. Per-program `t3` is floored at `1e-12s` to avoid div-by-zero.
- `[[bin]]` name is `mercury-bench` (hyphen); the crate/package is `mercury_bench` (underscore).
- Directories that can't be read produce a `warning:` on stderr and are skipped, not an error.
