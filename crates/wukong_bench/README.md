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

With no arguments the corpus defaults to `tests/run`. Non-mode arguments are corpus directories.

`compile-work` is the mode to compare two revisions with: every other mode reports wall time,
which on this host has moved 3.5x on identical inputs from thermal history alone, while a count is
a pure function of (binary, corpus). Each stage is counted three times per file; runs 2 and 3 must
agree (a disagreement is disclosed per file, never hidden), run 1's excess is reported once as
first-call initialisation, and the per-pass attribution is checked byte-identical against
`optimize`. `WUKONG_WORK_RUNS=<n>` raises the repetitions. Diff two runs' per-file tables to see
where work moved. This binary installs a counting `#[global_allocator]` (three thread-local `Cell`
updates per allocation, no atomics) for that mode; the other modes pay that and nothing else.

## Testing

This crate has no `cargo test` suite of its own — it is a measurement tool. Correctness is embedded
in each run's `-O0`/`-O3` equivalence check.
