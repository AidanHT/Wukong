# `bench/pytorch/` — PyTorch peer scripts

`transformer_layer_peer.py` — PyTorch reference for transformer-layer timing comparisons against
Wukong's recognized kernels. This is a **framework peer**, not a Wukong program and not part of the
`.wk` product surface ([`examples/`](../examples/), [`tests/run/`](../tests/run/)).

Run from repo root with the project's torch environment; see `BENCHMARKS.md` for the measurement
context (same-run ratios, not absolute TFLOP/s).
