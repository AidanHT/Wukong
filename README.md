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
- **Domain-aware optimization.** Mercury keeps tensor and loop operations *structured* in its IR so
  the compiler can do elementwise **fusion**, cache **tiling**, and **vectorization** a
  general-purpose C compiler can't see through. Those tensor-level passes are planned; the SSA
  scalar/loop optimizer that backs them — inlining, mem2reg, constant folding, CSE, DSE, DCE, and
  loop-invariant code motion — runs today and removes ~48% of IR ops on the benchmark kernels (54–60%
  on the heavy ones), making them ~1.5–2.5x faster under the interpreter.
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

That tensor/`@parallel`/`@simd` form is the target surface (it type- and shape-checks today). The
same kernel over fixed-size arrays **runs today** on the interpreter:

```mercury
fn saxpy(a: f32, x: [f32; 4], y: [f32; 4], out: [f32; 4]) {
    let mut i: i32 = 0;
    while i < 4 {
        out[i] = a * x[i] + y[i];   // arrays pass by reference; `out` is mutated in place
        i = i + 1;
    }
}
```

```sh
mercuryc --run examples/saxpy_array.mer   # 12 24 36 48
mercuryc --run examples/dot.mer           # 120
```

## Architecture

```
source.mer
   │  lexer → parser → AST
   │  sema  (name resolution, type inference, COMPILE-TIME SHAPE CHECKING)
   ▼
Mercury IR (MIR)         one SSA IR that lowers progressively from "High" to "Low"
   │  optimization passes (mem2reg → SSA, const-fold, CSE, DSE, DCE, LICM, simplify-cfg;
   │                       tensor fusion / tiling / vectorization are planned)
   ▼
MIR (Low)
   ├──────────────► interpreter   (always available, zero external deps; the reference oracle)
   └──────────────► LLVM backend  (inkwell; native object → linked executable)   [feature = "llvm"]
```

The front-end, optimizer, and a from-scratch **MIR interpreter** build and test with plain
`cargo test` on any machine. LLVM lives behind `--features llvm` and is needed only for native
codegen, so the whole compiler is buildable and testable without an LLVM install.

## Status

Early development — built incrementally and openly. The full front-end, optimizer, and interpreter
work today; native LLVM codegen and the tensor/SIMD/parallel execution paths are landing
progressively. See the docs:

- [Language guide](docs/language-guide.md) — the language surface, with an honest maturity legend.
- [Compiler internals](docs/internals.md) — architecture, MIR, optimizer, and testing.
- [Roadmap & limitations](docs/roadmap.md) — what runs, what's checked-only, what's planned.
- [LLVM setup](docs/llvm-setup.md) — optional native-codegen toolchain.

## Building

```sh
cargo build                 # the compiler (interpreter backend, no LLVM needed)
cargo test                  # unit + golden + end-to-end tests
cargo run -p mercuryc -- --help
cargo run -p mercuryc -- --run examples/fib.mer
cargo run -p mercury_bench --release -- tests/run examples bench/kernels   # optimizer report
```

Native codegen (optional, requires an LLVM 19 install — see `docs/llvm-setup.md`):

```sh
cargo build --features llvm
```

## License

MIT — see [LICENSE](LICENSE).
