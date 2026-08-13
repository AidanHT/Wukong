# `data/`

Runtime data that is too large or too machine-specific to live in git. The root `.gitignore`
ignores `/data/`; blobs are regenerated locally and never committed.

Subdirectories hold exported model weights and reference tensors consumed by examples and
verification scripts via relative paths from the repo root (for example `data/gpt2/...`).

| Path | Purpose |
|------|---------|
| [`gpt2/`](gpt2/) | GPT-2 124M weights, prompt tokens, and reference logits for the `examples/gpt2*.wk` programs |

See each subdirectory's README for layout, file names, and how to regenerate contents.
