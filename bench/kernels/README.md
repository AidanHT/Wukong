# `bench/kernels/` — harness-sized `.wk` programs

Five `.wk` programs used by `wukong_bench` / `wukong_xbench` at sizes beyond the `tests/run/` gate:

- `collatz.wk`, `fib_rec.wk`, `matmul.wk`, `primes.wk`, `sum_to_n.wk`

These are Wukong source, but they serve **throughput measurement** — the default correctness corpus
is [`tests/run/`](../tests/run/). Changing a kernel here can break `wukong_xbench` without failing
`cargo test`.
