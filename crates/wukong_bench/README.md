# wukong_bench

Benchmark harness for **optimizer effectiveness** and **interpreter timing**. Measures the compiler;
it is not the product.

For each `.wk` program it can lower, reports `-O0` vs `-O3`:

- total MIR op count and reduction
- interpreter wall-clock per run (adaptively batched), including `wukong_interp::run`'s fixed setup cost

Every program is also a correctness check: `-O0` and `-O3` must verify, agree on exit code and
stdout, and Cranelift at `-O3` must agree with the interpreter at `-O3`. Failures print `*** FAILED`
and exit 1.

## Usage

```sh
cargo run -p wukong_bench --release -- tests/run examples bench/kernels
```

**Timing numbers require `--release`.** A debug build makes runtime kernels look far slower than
JIT'd loops; the harness warns but still runs the correctness gate.

## Modes (first CLI argument)

| mode | report |
|---|---|
| *(default)* | optimizer effectiveness + execution timing |
| `compile-time` / `--compile-time` / `ctime` | in-process optimizer timing |
| `compile-profile` / `profile` | full-pipeline per-stage breakdown |
| `compile-work` / `work` | **counting** instrument: allocations, reallocations, bytes and optimizer pass invocations per stage and per pass — deterministic, clock-invariant, no `--release` or cooldown needed |
| `spawn-overhead` / `spawn` | in-process vs process-spawn characterization |
| `compile-vs [wukongc-path]` | cross-compiler compile-time comparison |
| `cold-start [--runs=N] [--backend=native\|interp] [--wukongc=PATH] [prog.wk]` | fresh-process compile + first run of a fixed program (a `@parallel` 128^3 GEMM + checksum): `--runs` (10) children of `wukongc --run -O2`, reported as min / median / p95 / max and the first child alone |
| `scaling [--runs=N] [--max-threads=N] [--reps=N] [--wukongc=PATH] [prog.wk]` | one workload (a `@parallel` 512^3 GEMM x `--reps`, 200) in fresh processes at `WUKONG_NUM_THREADS` = 1, 2, 4, .., all: per count min / median wall, speedup over one thread, efficiency |

With no arguments the corpus defaults to `tests/run`. Non-mode arguments are corpus directories.

`cold-start` and `scaling` (P6.10) time whole processes of `target/release/wukongc` (or
`--wukongc=`), because both quantities are per process: the runtime reads its thread count once, and
a cold start is everything a warm in-process loop amortizes away. Every child's exit status and
stdout are checked; a failed child, or a sweep whose output changes with the thread count, exits 1
rather than reporting. The `cold-start:` and `scaling:` summary lines are a stable,
machine-readable contract, pinned by this crate's unit test. The scaling wall includes the serial
compile and start-up, so its efficiency is a whole-process (Amdahl) figure.

`compile-work` is the mode to compare two revisions with: every other mode reports wall time,
which on this host has moved 3.5x on identical inputs from thermal history alone, while a count is
a pure function of (binary, corpus). Each stage is counted three times per file; runs 2 and 3 must
agree (a disagreement is disclosed per file, never hidden), run 1's excess is reported once as
first-call initialisation, and the per-pass attribution is checked byte-identical against
`optimize`. `WUKONG_WORK_RUNS=<n>` raises the repetitions. Diff two runs' per-file tables to see
where work moved. This binary installs a counting `#[global_allocator]` (three thread-local `Cell`
updates per allocation, no atomics) for that mode; the other modes pay that and nothing else.

The optimizer can run a program's independent functions on several threads (`WUKONG_OPT_THREADS`;
the MIR is the same on every count). `compile-work` counts its optimize stage on one thread, since
its counters are per thread. `compile-time`'s headline `opt-O3` column is the production `optimize`
on the automatic count, so it is the wall time a compile pays. Its per-pass share table is measured
on one optimizer thread. On several threads each fixpoint pass would be summed over the workers
while the unroll and SLP buckets stayed wall clock, and the shares would mix the two kinds of time.

## Testing

The measurements themselves have no `cargo test` equivalent; correctness is embedded in each run's
`-O0`/`-O3` equivalence check and in the process modes' exit/stdout checks. `cargo test -p
wukong_bench` covers the pure parts of the process modes: the min/median/p95 statistics, the thread
sweep, option parsing, the pinned report-line spellings, and that both built-in programs still
compile and dispatch `wukong_sgemm_parallel`. It also checks that `compile-time`'s per-pass
attribution runs on one optimizer thread.
