# Wukong examples

Run any example through the interpreter (no LLVM required):

```sh
cargo run -p wukongc -- --run examples/<name>.wk
```

Inspect any pipeline stage with `--emit=tokens|ast|mir-high|mir|llvm-ir`, e.g.:

```sh
cargo run -p wukongc -- --emit=mir -O3 examples/dot.wk
```

## Runnable today (interpreter)

| File                | What it shows                                   | Output            |
|---------------------|-------------------------------------------------|-------------------|
| `hello.wk`         | smallest program with output                    | `42` `42`         |
| `fib.wk`           | recursion, branches, arithmetic                 | `55`              |
| `dot.wk`           | dot product over a fixed-size array             | `120`             |
| `saxpy_array.wk`   | SAXPY `out = a*x + y` over arrays                | `12 24 36 48`     |
| `relu.wk`          | ReLU over an array with an out-parameter        | `0 5 0 0 8 0`     |
| `gemm.wk`          | f32 matmul recognized + dispatched to the tuned AVX2 GEMM | `4 5 10 11` |
| `gpt2.wk`          | GPT-2 decoder block: LayerNorm + multi-head causal attention + GELU MLP + residuals. Readable *shape* spec; measured, it dispatches only the 2 LayerNorms, the 6 non-attention projections and the 2 residual adds — see `gpt2_forward_bench_small.wk` for the full recognized surface | `-1113 1099 677 -264` |
| `llama_block.wk`   | Llama block: RMSNorm + RoPE + multi-head attention + SwiGLU FFN + residuals | `-1227 1204 593 -404` |

The end-to-end test programs under [`../tests/run`](../tests/run) are also runnable and cover
arrays, a flat GEMM, transpose, bubble sort, gcd, casts, floats, and more. Heavier programs for
the benchmark harness live in [`../bench/kernels`](../bench/kernels).

## GPT-2 124M end-to-end inference

The real pretrained OpenAI GPT-2 124M, run as a Wukong program and verified numerically against
HuggingFace. This is a **correctness / capability** result — inference only, external tokenization, no
speed claim (see [../README.md](../README.md) and [../BENCHMARKS.md](../BENCHMARKS.md)).

| File | What it shows | How to run |
|------|---------------|------------|
| `gpt2_infer.wk` | Full GPT-2 124M forward: loads the **real pretrained weights** (124,439,808 params) + prompt token ids from disk via the `read_f32` / `read_i32` file-I/O intrinsics, runs embed → 12 pre-LayerNorm blocks (biased QKV, 12-head causal attention, tanh-GELU MLP, residuals) → final LayerNorm → tied LM head → logits `[5, 50257]`, writes them back, and prints the argmax (**1757 " John"** for *"Hello, my name is"*). **Native-only** — the interpreter cannot hold 124M params. | export the weights (below), then `wukongc --run --backend=native examples/gpt2_infer.wk` from the repo root; check with `python tools/verify_gpt2.py` (rel 1.87e-6) |
| `gpt2_infer_small.wk` | The **same** forward pass at a reduced, interpreter-runnable config (D=64, H=4, DFF=256, SEQ=8, LAYERS=2) with synthetic in-loop weights and no file I/O — the differential gate that runs **bit-identically on interpreter and native**. Like `gpt2_infer.wk` it uses bare const dims, so (measured) it dispatches **no** recognized kernel: it gates the scalar pipeline. | `cargo run -p wukongc -- --run examples/gpt2_infer_small.wk` |
| `gpt2_forward_bench_small.wk` | The same config again, but spelled in the **recognized dispatch forms** (local `let` dims, per-head repack, `gelu()`, gathered γ/β, bare-index LM-head GEMV) — measured: 6 `sgemm_nt_epi`, 3 `norm_affine_f32`, 2 `velem_f32`, 1 each of `sgemm_nt_alpha` / `sgemm_nt` / `norm_f32` / `vmath_f32` / `sgemv`, the identical set `gpt2_forward_bench.wk` emits. This is the differential fixture for the **kernel-call lowering seam**; its output cross-checks against the all-scalar `gpt2_infer_small.wk` (argmax 15, −14352, 9521). | `cargo run -p wukongc -- --run examples/gpt2_forward_bench_small.wk` |
| `gpt2_config.wk` | GPT-2 124M layout constants (dims + flat-blob element offsets) imported by `gpt2_infer.wk`; **generated** by `tools/export_gpt2.py` (the single source of truth for the offset table) — not run directly | imported, not run |

**Obtaining the weight blob.** The weights are large and not committed to the repo. Export them from
HuggingFace `transformers` (needs `torch` + `transformers` + `numpy`):

```sh
python tools/export_gpt2.py
```

This downloads `GPT2LMHeadModel.from_pretrained("gpt2")`, writes the flat little-endian f32 weight blob,
the prompt token ids, and the authoritative HuggingFace reference logits under `<repo>/data/gpt2/`
(override with `$GPT2_DATA_DIR` or a positional argument), and regenerates `examples/gpt2_config.wk`.
Then run `gpt2_infer.wk` on the native backend and verify with `tools/verify_gpt2.py`. Tokenization is
performed by the HuggingFace tokenizer inside the exporter; the `.wk` program only ever consumes the
integer token ids it produces.

`verify_gpt2.py` refuses a `gpt2_wuk_logits.bin` older than an hour (`--max-age-seconds`, or
`--no-age-check` to re-check an archived artifact): `gpt2_infer.wk` exits 1 *without writing* when the
data is unreachable, so a stale artifact would otherwise let the verifier pass a run that never happened.

## Shape-checked, not yet executed

These parse and pass compile-time **shape checking** (the headline feature) but use tensor/SIMD/
parallel constructs that lowering does not handle yet, so compiling or running them reports
`error[C0001]: ... is not yet supported by codegen`. Inspect them with `--emit=ast`:

| File          | What it shows                                         |
|---------------|-------------------------------------------------------|
| `vadd.wk`    | SIMD vector add over shape-typed tensors              |
| `matmul.wk`  | tiled, parallel matmul with `Tensor[f32, M, K]` shapes |
| `softmax.wk` | numerically-stable softmax                            |

Try the shape checker on a mismatch:

```sh
cargo run -p wukongc -- --emit=mir tests/fail/matmul_dim.wk   # error[E0502]
cargo run -p wukongc -- --explain E0502
```
