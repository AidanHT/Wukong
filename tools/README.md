# `tools/` — scripts around the compiler

Helper scripts for exporting data, verifying outputs, and driving measurement. **Not the language
thesis** — the thesis is user-written `.wk` in [`examples/`](../examples/) and [`tests/`](../tests/).

| Script | What |
|--------|------|
| `export_gpt2.py` | Downloads GPT-2 weights, writes `data/gpt2/`, regenerates `examples/gpt2_config.wk` |
| `verify_gpt2.py` | Cross-checks `gpt2_infer.wk` logits against HuggingFace reference |
| `bench_gpt2_torch.py` | PyTorch reference timing for GPT-2 (peer, not a Wukong program) |
| `fa2_sdpa_peer.py` | FlashAttention SDPA peer helper |
| `measure_gpt2.ps1` | Windows timing wrapper for GPT-2 benches |

Subdirectories:

| Path | Role |
|------|------|
| [`cloud/`](cloud/) | Modal launchers for rented-GPU bring-up — Rust/cargo on cloud silicon, not `.wk` authors |
| [`perf/`](perf/) | Local performance measurement helpers |
| [`cuda-redist/`](cuda-redist/) | Vendored CUDA redistributables for peer builds (when present) |

Skipped from documentation: `torch-venv`, `torch-cuda-venv`, `cuda-redist/nvidia*`, `__pycache__`.
