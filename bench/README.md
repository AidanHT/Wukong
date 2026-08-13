# `bench/` — measurement harnesses (not the language)

Rust benches, peer scripts, and raw GPU round logs. **Not the language thesis** — authors write
`.wk` in [`examples/`](../examples/) and [`tests/run/`](../tests/run/). This tree drives
`cargo bench`, `wukong_xbench`, and rented-GPU campaigns.

| Path | What |
|------|------|
| [`kernels/`](kernels/) | `.wk` programs sized for the benchmark harness (heavier than `tests/run/`) |
| [`pytorch/`](pytorch/) | PyTorch peer scripts for cross-framework comparison |
| [`gpu/`](gpu/) | Append-only raw round logs from rented GPUs — measurement evidence, not programs |
| `vecbench.wk`, `vecbench.c`, `vecbench_ab.sh` | Cross-language vector bench harness at this level |

Published numbers in `BENCHMARKS.md` must trace to files under `bench/gpu/<device>/`.
