# Mercury

**A low-level, low-abstraction systems language built for ML/DL compilers and high-performance tensor kernels.**

Mercury is the language you reach for *instead of* C, C++, or Rust when you are writing the
performance-critical core of a machine-learning stack: fused elementwise kernels, tiled matmuls,
attention microkernels, custom ops, and the compiler passes that generate them.

It is **not** a high-level framework. There is no garbage collector, no hidden allocation, and no
hidden control flow. Every byte of memory comes from an allocator you named, every SIMD lane is one
you asked for, and every parallel loop has a schedule you chose.

## Why Mercury

Mercury compiles to native code through a **from-scratch [Cranelift](https://cranelift.dev) backend —
no LLVM, no external toolchain**. In a head-to-head cross-language benchmark (same kernel in each
language, one timing harness; see **[BENCHMARKS.md](BENCHMARKS.md)**), Mercury:

- **compiles ~100–190× faster** than gcc/rustc (Cranelift JIT in-process vs spawning a full
  C/Rust+LLVM toolchain) — the metric that dominates real ML edit-run iteration;
- is **faster than C on every single-threaded elementwise kernel** measured, including a **~2.6×** win
  on `dot` from automatic reduction vectorization;
- is **~3–8× faster** than idiomatic single-threaded C once `@parallel` auto-parallelizes and
  vectorizes the loop, and **~2.6×** faster on parallel 512² matmul.

The one honest single-thread loss is dense matmul (~3.2×), because Cranelift emits 128-bit SSE rather
than 256-bit AVX — a deliberate trade for zero-dependency builds and the compile-speed win above.

Where Mercury is built to win for the ML/DL niche:

- **Compile-time shape safety.** Tensor shapes live in the type system:
  `Tensor[f32, M, K] * Tensor[f32, K, N] -> Tensor[f32, M, N]`. A shape mismatch is a *type error*,
  not a segfault at 3 a.m. C/C++ cannot express this; Rust cannot express it ergonomically.
- **The knobs kernels need, as language constructs.** SIMD width, data layout (row-/col-major,
  strides, tiling), alignment, arenas, and parallel schedules are first-class — not a soup of
  intrinsics and `#pragma`s.
- **Zero hidden cost.** No GC, no implicit copies of large aggregates, no surprise allocations.
- **Domain-aware optimization that runs today.** The compiler keeps loops structured in its IR and
  **auto-vectorizes** elementwise loops (incl. branchy ones via if-conversion) and **reductions** to
  128-bit SIMD, contracts `x + y*z` to a **fused multiply-add**, **fuses** adjacent elementwise loops,
  and **auto-parallelizes** `@parallel` loops across cores — things a general-purpose C compiler won't
  do to naively-written source. Underneath, an SSA optimizer (inlining, mem2reg, const-fold, CSE, DSE,
  DCE, LICM) removes ~48% of IR ops on the benchmark kernels (54–60% on the heavy ones). Tensor-level
  tiling and op-graph fusion are still planned.
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
   │  mir_build  (lowering + SIMD auto-vectorization: elementwise, reductions, FMA, fusion)
   ▼
Mercury IR (MIR)         one SSA IR that lowers progressively from "High" to "Low"
   │  optimization passes (mem2reg → SSA, const-fold, CSE, DSE, DCE, LICM, simplify-cfg;
   │                       inlining; tensor-level tiling / op-graph fusion are planned)
   ▼
MIR (Low)
   ├──────────────► interpreter      (always available, zero deps; the reference oracle)
   ├──────────────► Cranelift backend (native JIT + object/exe; NO LLVM — the fast path)
   └──────────────► LLVM backend     (textual IR for external clang/llc)            [feature = "llvm"]
```

The front-end, optimizer, the from-scratch **MIR interpreter**, *and* the **Cranelift native
backend** build and test with plain `cargo test` on any machine — no LLVM, no toolchain. The native
backend JIT-compiles in-process (and emits host objects) and is differentially tested against the
interpreter bit-for-bit. LLVM is an optional *textual-IR* emitter behind `--features llvm`.

## Status

Early development — built incrementally and openly. The full front-end, optimizer, interpreter, and
**native Cranelift backend** work today, including SIMD auto-vectorization (elementwise + reductions),
FMA contraction, loop fusion, and `@parallel` multicore execution over fixed-size-array kernels.
Shape-typed *tensor* operations type-check today but do not yet lower/run. See the docs:

- [Benchmarks](BENCHMARKS.md) — honest cross-language results vs C and Rust, with methodology.
- [Language guide](docs/language-guide.md) — the language surface, with an honest maturity legend.
- [Compiler internals](docs/internals.md) — architecture, MIR, optimizer, the native backend, and testing.
- [Roadmap & limitations](docs/roadmap.md) — what runs, what's checked-only, what's planned.
- [LLVM setup](docs/llvm-setup.md) — optional textual-IR backend (the native path uses Cranelift, no LLVM).

## Building

```sh
cargo build                 # the whole compiler incl. the native Cranelift backend — no LLVM
cargo test                  # unit + golden + end-to-end + differential (interp vs native) tests
cargo run -p mercuryc -- --help
cargo run -p mercuryc -- --run examples/fib.mer
cargo run -p mercury_bench --release -- tests/run examples bench/kernels   # optimizer report
cargo run -p mercury_xbench --release      # cross-language benchmark vs C/Rust (needs gcc/rustc)
```

The native backend (Cranelift) is built in by default and needs no toolchain. The optional LLVM
backend emits textual IR only (for an external `clang`/`llc`) and lives behind a feature flag:

```sh
cargo build --features llvm
```

## License

MIT — see [LICENSE](LICENSE).
