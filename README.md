# Wukong

**A programming language for ML/DL tensor kernels with compile-time shape safety.**

Wukong (`.wk`) is for writing performance-critical tensor code — fused elementwise loops, tiled
matmuls, reductions, norms, custom ops — in a small explicit language instead of hand-rolled
CUDA or intrinsics in C/C++/Rust. Tensor dimensions live in the type system, so a shape mismatch at
a call site is a compile error (`E0501`/`E0502`), not a wrong result at run time.

No garbage collector, no hidden allocation, no hidden control flow. `wukongc`, its compiler, is pure
Rust and needs no external toolchain: it runs programs on a reference interpreter, a native
[Cranelift](https://cranelift.dev) JIT/AOT backend, or (with `--features gpu`) NVIDIA GPUs via PTX.

## Example

```wukong
fn saxpy<N>(a: f32, x: Tensor[f32, N], y: Tensor[f32, N], mut out: Tensor[f32, N]) {
    @parallel
    for i in 0..N {
        out[i] = a * x[i] + y[i];   // fused multiply-add, vectorized, parallelized
    }
}
```

## Quick start

```sh
cargo build
cargo run -p wukongc -- --run examples/saxpy_array.wk                        # interpreter
cargo run -p wukongc -- --run --backend=native -O2 tests/run/tensor_matmul.wk  # Cranelift JIT
cargo run -p wukongc -- --emit=exe -O2 examples/fib.wk                         # AOT executable
cargo run -p wukongc -- --emit=mir -O2 tests/run/transcendental_kernel.wk      # inspect MIR
cargo run -p wukongc -- --repl                                                 # interactive session
cargo run -p wukongc -- --help                                                 # full CLI
```

## Features

- **Compile-time shape safety** — `Tensor[f32, M, K]` carries its dimensions and layout into every
  call; dimensions may be generic (`fn f<M, N>`) or run-time values.
- **Kernel knobs as language constructs** — `@parallel`, `@tile`, `@noalias`, `@strict_fp`, tensor
  layouts (`.col_major`), checked slices `[]T`, and an explicit allocator and scratch arena.
- **A full systems language** — structs, enums with data, `match` with exhaustiveness checking,
  traits and generics (statically dispatched), closures and function values, `str`, `i128`/`u128`,
  `bf16`/`f16`/`f8` and int4, modules and imports, `extern "C"` interop.
- **Optimizing compiler** — SSA optimizer (inlining, SROA, mem2reg, CSE, DSE, LICM, unrolling,
  SLP), loop auto-vectorization to AVX2, FMA contraction, loop fusion, and multicore `@parallel`.
- **Tuned kernels** — matmul, GEMV, softmax, LayerNorm/RMSNorm, attention, reductions, scans and
  activations dispatch to hand-tuned AVX2/FMA microkernels, either recognized from the loops you
  write or called by name (`matmul`, `softmax`, `sdpa`, …).
- **Autodiff and training** — reverse-mode gradients (`--emit=grad`) and a built-in SGD/AdamW
  training loop (`--train`), on the interpreter, the JIT, or as a standalone executable.
- **Model I/O** — safetensors, `.npy` and GGUF loaders that shape-check every tensor as it loads.
- **Real models** — runs GPT-2 124M inference end-to-end as an ordinary `.wk` program
  ([`examples/gpt2_infer.wk`](examples/gpt2_infer.wk)), matching HuggingFace's logits.
- **Tooling** — source formatter (`wukongc fmt`), REPL, doc comments with Markdown output
  (`--emit=doc`) and runnable doc tests (`--doctest`), DWARF debug info (`-g`), JSON diagnostics,
  and `wukongc --explain <CODE>` for every error code.

## Architecture

```
source.wk
   │  lexer → parser → AST → sema (types + compile-time shape checking)
   │  mir_build (lowering, kernel recognition, vectorization)
   ▼
MIR (block-parameter SSA) → optimizer → autodiff
   ├──► interpreter        reference oracle, no toolchain
   ├──► Cranelift backend  native JIT + object / executable
   ├──► GPU backend        PTX via the CUDA driver   [--features gpu]
   └──► LLVM backend       textual LLVM IR
```

## Building and testing

```sh
cargo build                                  # the whole compiler, no LLVM needed
cargo test --workspace                       # unit, golden, end-to-end and differential tests
cargo check --features gpu --all-targets     # the optional GPU backend
```

## Documentation

- [Language guide](docs/language-guide.md) — the language surface
- [Compiler internals](docs/internals.md) — architecture, MIR, optimizer, testing
- [Roadmap](docs/roadmap.md) — what runs today and what is planned
- [Benchmarks](BENCHMARKS.md) — measurements against C, C++, Rust and PyTorch
- [Contributing](CONTRIBUTING.md)

## License

MIT — see [LICENSE](LICENSE).
