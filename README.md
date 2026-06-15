# Mercury

**A low-level, low-abstraction systems language built for ML/DL compilers and high-performance tensor kernels.**

Mercury is the language you reach for *instead of* C, C++, or Rust when you are writing the
performance-critical core of a machine-learning stack: fused elementwise kernels, tiled matmuls,
attention microkernels, custom ops, and the compiler passes that generate them.

It is **not** a high-level framework. There is no garbage collector, no hidden allocation, and no
hidden control flow. Every byte of memory comes from an allocator you named, every SIMD lane is one
you asked for, and every parallel loop has a schedule you chose.

## Why Mercury

Native code emitted by Mercury goes through the same LLVM backend that Clang and rustc use, so it
*matches* C/C++/Rust on raw throughput. Where Mercury **wins** for the ML/DL niche:

- **Compile-time shape safety.** Tensor shapes live in the type system:
  `Tensor[f32, M, K] * Tensor[f32, K, N] -> Tensor[f32, M, N]`. A shape mismatch is a *type error*,
  not a segfault at 3 a.m. C/C++ cannot express this; Rust cannot express it ergonomically.
- **The knobs kernels need, as language constructs.** SIMD width, data layout (row-/col-major,
  strides, tiling), alignment, arenas, and parallel schedules are first-class — not a soup of
  intrinsics and `#pragma`s.
- **Zero hidden cost.** No GC, no implicit copies of large aggregates, no surprise allocations.
- **Domain-aware optimization.** Mercury keeps tensor and loop operations *structured* in its IR long
  enough to do elementwise **fusion**, cache **tiling**, and **vectorization** that a general-purpose
  C compiler can't see through — then lowers them to tight scalar+SIMD loops.
- **Seamless interop.** A clean C ABI (`@extern("C")` / `@export`) calls into BLAS/cuBLAS and lets
  Mercury kernels be embedded in existing C/C++/CUDA stacks.

## The four signature features

```mercury
// (1) shape-typed tensors  (2) SIMD vectors  (3) explicit memory/layout  (4) parallelism
fn saxpy<N>(a: f32, x: Tensor[f32, N], y: Tensor[f32, N], out: Tensor[f32, N]) {
    @parallel @simd
    for i in 0..N {
        out[i] = a * x[i] + y[i];   // fused multiply-add, vectorized, parallelized
    }
}
```

## Architecture

```
source.mer
   │  lexer → parser → AST
   │  sema  (name resolution, type inference, COMPILE-TIME SHAPE CHECKING)
   ▼
Mercury IR (MIR)         one SSA IR that lowers progressively from "High" to "Low"
   │  optimization passes (fusion, tiling, vectorization, inlining, DCE/CSE, ...)
   ▼
MIR (Low)
   ├──────────────► interpreter   (always available, zero external deps; the reference oracle)
   └──────────────► LLVM backend  (inkwell; native object → linked executable)   [feature = "llvm"]
```

The front-end, optimizer, and a from-scratch **MIR interpreter** build and test with plain
`cargo test` on any machine. LLVM lives behind `--features llvm` and is needed only for native
codegen, so the whole compiler is buildable and testable without an LLVM install.

## Status

Early development — built incrementally and openly. See
[`docs/`](docs/) for the language guide and compiler internals as they land.

## Building

```sh
cargo build                 # the compiler (interpreter backend, no LLVM needed)
cargo test                  # unit + golden + end-to-end tests
cargo run -p mercuryc -- --help
```

Native codegen (optional, requires an LLVM 19 install — see `docs/llvm-setup.md`):

```sh
cargo build --features llvm
```

## License

MIT — see [LICENSE](LICENSE).
