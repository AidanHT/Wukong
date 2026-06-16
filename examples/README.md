# Mercury examples

Run any example through the interpreter (no LLVM required):

```sh
cargo run -p mercuryc -- --run examples/<name>.mer
```

Inspect any pipeline stage with `--emit=tokens|ast|mir-high|mir|llvm-ir`, e.g.:

```sh
cargo run -p mercuryc -- --emit=mir -O3 examples/dot.mer
```

## Runnable today (interpreter)

| File                | What it shows                                   | Output            |
|---------------------|-------------------------------------------------|-------------------|
| `hello.mer`         | smallest program with output                    | `42` `42`         |
| `fib.mer`           | recursion, branches, arithmetic                 | exit code 55      |
| `dot.mer`           | dot product over a fixed-size array             | `120`             |
| `saxpy_array.mer`   | SAXPY `out = a*x + y` over arrays                | `12 24 36 48`     |

The end-to-end test programs under [`../tests/run`](../tests/run) are also runnable and cover
arrays, a flat GEMM, transpose, bubble sort, gcd, casts, floats, and more.

## Shape-checked, not yet executed

These parse and pass compile-time **shape checking** (the headline feature) but use tensor/SIMD/
parallel constructs whose execution paths are still under construction:

| File          | What it shows                                         |
|---------------|-------------------------------------------------------|
| `vadd.mer`    | SIMD vector add over shape-typed tensors              |
| `matmul.mer`  | tiled, parallel matmul with `Tensor[f32, M, K]` shapes |
| `softmax.mer` | numerically-stable softmax                            |

Try the shape checker on a mismatch:

```sh
cargo run -p mercuryc -- --emit=mir tests/fail/matmul_dim.mer   # error[E0502]
cargo run -p mercuryc -- --explain E0502
```
