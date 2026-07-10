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
| `gpt2.wk`          | GPT-2 decoder block: LayerNorm + multi-head causal attention + GELU MLP + residuals (recognized op-forms) | `-1113 1099 677 -264` |
| `llama_block.wk`   | Llama block: RMSNorm + RoPE + multi-head attention + SwiGLU FFN + residuals | `-1227 1204 593 -404` |

The end-to-end test programs under [`../tests/run`](../tests/run) are also runnable and cover
arrays, a flat GEMM, transpose, bubble sort, gcd, casts, floats, and more. Heavier programs for
the benchmark harness live in [`../bench/kernels`](../bench/kernels).

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
