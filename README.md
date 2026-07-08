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

- **compiles fast** — the metric that dominates real ML edit-run iteration. Apples-to-apples
  *compiler-to-object* (`mercuryc --emit=obj -O2` vs `gcc/g++/rustc -O2 -c`, same artifact, same
  machine): **~7–12× faster** (`mercury_bench compile-vs`; ~7–9× measured this run). The larger **~100–680× (geomean ~305×)**
  figure is *time-to-running-code*: Mercury JIT-compiles in-process while C/Rust must spawn a full
  toolchain **and link a shared object** — a real advantage for the JIT/embedding workflow, but not a
  compiler-vs-compiler number, so it is disclosed as such, never as the headline;
- **wins matmul/GEMM**, the flagship ML kernel: the compiler recognizes a matmul nest (incl. the
  `nn.Linear` `A·Bᵀ` form) and dispatches it to a tuned register-blocked, cache-tiled, packed
  **AVX2/FMA** microkernel — **~3–3.6× faster single-thread** at **~110–120 GFLOP/s ≈ 90% of one
  P-core's AVX2-FMA roofline** (and **~1.1–1.3× over the tuned `matrixmultiply` Rust crate**, at
  **oneMKL parity**), and **up to ~18× parallel** on plain `C = A·B`, **up to ~104× on `nn.Linear`**
  (where naive C leaves the reduction latency-bound), the lead *growing with matrix size*. Against
  the honest SOTA bar — **multi-threaded oneMKL** — Mercury's `@parallel` GEMM is **60–70% at
  512–1024³ but reaches parity-to-winning (86–129%) at ≥2048³**, the large-matrix ML regime (the
  residual mid-size gap is cross-core sync/packing overhead on this P+E hybrid, not the kernel);
- **dispatches the whole transformer/training kernel surface** to tuned microkernels, where the win
  over idiomatic C is largest: the **weight-gradient GEMM** `dW=Aᵀ·B` (training backward, A read
  column-strided) **up to ~128× single / ~445× parallel**, the **fused FFN** `silu(A·Bᵀ)` **~24–26×**,
  **RoPE** rotary embedding **~29–54×** (up to **~156× parallel**), **strided column reductions**
  (bias-grad / per-channel quant stats) **~29–50×**, and the training-backward kernels
  (activation/softmax/LayerNorm-RMSNorm backward, cross-entropy) **~3–13×**;
- **wins the transcendental/activation family ~2–13×** (**~28× under `@parallel`**) — the cleanest
  compute-bound win. Mercury dispatches a pure `out[i]=f(x[i])` loop for **35** functions
  (`exp`/`log`/`exp2`/`log2`/`exp10`/`log10`/`cbrt`/`expm1`/`log1p`/`tanh`/`sigmoid`/`gelu`/`silu`/
  `softplus`/`softsign`/`logsigmoid`/`mish`/`sin`/`cos`/`tan`/`atan`/`asin`/`acos`/`erf` plus the
  hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh` — the transformer activations plus **RoPE**'s
  `sin`/`cos`, the inverse trig, the exact-GELU `erf`, the stable `expm1`/`log1p`/`logsigmoid`, and the
  hyperbolic/Poincaré-embedding inverse trio) to a **256-bit AVX2 ≈1-ULP poly kernel**, where gcc/rustc
  call scalar `libm` and **cannot vectorize a loop containing the call**;
- **wins fused row-norms** (`softmax`/`LayerNorm`/`RMSNorm`, incl. the learned-γ/β affine form)
  **~1.9–6.6×** and **convolution** (im2col + GEMM) **~6–7×**;
- **wins int8 `nn.Linear`** (`vpdpbusd`) **~1.5–2.5× single / ~4.6–14.7× parallel**, and runs a full
  **bf16 *and* f16 mixed-precision CPU suite** — `dot` (**~3×**) / `sum` (**~6–8×**), `max`/`min`/`absmax`
  (the symmetric-quant scale), streaming `axpby`, the `nn.Linear` GEMM (**~24–25×**), and the 36-op
  activation set — all half-in/f32-out, where C/Rust can vectorize neither `libm` nor the half→f32
  widen (f16 via the F16C `vcvtph2ps`);
- **wins reductions ~2.6–2.9×** (`dot`, L2 loss) by reassociating the f32 sum across vector lanes,
  which gcc/rustc leave serial — and **~7.9–8.6× under `@parallel`** (up to ~25× for `max`/`absmax`);
- and ships a **GPU backend** (`--features gpu`, NVIDIA RTX 4050; PTX + cudarc driver-JIT, no CUDA
  toolkit): fp16 tensor-core GEMM at **cuBLAS parity (~101%) ≤1024³**, a fused **flash-attention**
  that **beats the genuinely-fused cuDNN + cutlass fMHA in the causal-S≤512 and fused-RoPE regimes**
  (and is 3.6–5× the unfused cuBLAS chain), trailing cuDNN only at long context (S≥2048),
  int8 GEMM **~180–237× naive CUDA-C**, **95.7% of the 192 GB/s HBM peak**, and **0.76 ms cold GPU
  compile vs Triton's 30–120 s**.

The domain-aware paths (GEMM, the `vmath` transcendentals, the `velem` streaming elementwise, the
fused norms) all emit **true 256-bit AVX2/FMA** via hand-written runtime microkernels — the width
Cranelift's *general* vectorizer can't legalize (it caps at 128-bit `f32x4`). So even the
memory-bandwidth-bound elementwise kernels are now small **wins** (saxpy ~1.25–1.45×, poly ~1.1–1.2×,
widening to ~1.3–1.6× at realistic >L3 tensor sizes via non-temporal stores); the one honest **tie** left is
`relu` at an L3-resident size, where both languages are pinned to the same cache bandwidth.

Where Mercury is built to win for the ML/DL niche:

- **Compile-time shape safety.** Tensor shapes live in the type system:
  `Tensor[f32, M, K] * Tensor[f32, K, N] -> Tensor[f32, M, N]`. A shape mismatch is a *type error*,
  not a segfault at 3 a.m. C/C++ cannot express this; Rust cannot express it ergonomically.
- **The knobs kernels need, as language constructs.** SIMD width, data layout (row-/col-major,
  strides, tiling), alignment, arenas, and parallel schedules are first-class — not a soup of
  intrinsics and `#pragma`s.
- **Zero hidden cost.** No GC, no implicit copies of large aggregates, no surprise allocations.
- **Domain-aware optimization that runs today.** The compiler **recognizes a matmul nest** (the
  `ikj` accumulate and `ijk` dot-product forms, incl. `nn.Linear` `A·Bᵀ`) and lowers it to a tuned
  **register-blocked, cache-tiled, packed AVX2/FMA GEMM microkernel** — the way XLA/TVM/oneDNN lower a
  matmul op. It also **auto-vectorizes** elementwise loops (incl. branchy ones via if-conversion) and
  **reductions** to SIMD, contracts `x + y*z` to a **fused multiply-add**, **fuses** adjacent
  elementwise loops, and **auto-parallelizes** `@parallel` loops across cores — things a
  general-purpose C compiler won't do to naively-written source. Underneath, an SSA optimizer
  (inlining, mem2reg, const-fold, CSE, DSE, DCE, LICM) removes ~42% of IR ops on the benchmark kernels
  (~48–54% on the heavy transformer/GEMM kernels). Op-graph fusion across tensor ops is still planned.
- **Interop (planned).** A clean C ABI (`@extern("C")` / `@export`) is designed to call into
  BLAS/cuBLAS and embed Mercury kernels in C/C++/CUDA stacks. The attributes parse and validate
  today; symbol export/import is not yet wired (see the roadmap).

## The four signature features

```mercury
// (1) shape-typed tensors  (2) SIMD vectors  (3) explicit memory/layout  (4) parallelism
fn saxpy<N>(a: f32, x: Tensor[f32, N], y: Tensor[f32, N], mut out: Tensor[f32, N]) {
    @parallel @simd
    for i in 0..N {
        out[i] = a * x[i] + y[i];   // fused multiply-add, vectorized, parallelized
    }
}
```

That tensor/`@parallel`/`@simd` form is the target surface (it type- and shape-checks today). The
same kernel over fixed-size arrays **runs today** on the interpreter:

```mercury
fn saxpy(a: f32, x: [f32; 4], y: [f32; 4], mut out: [f32; 4]) {
    let mut i: i32 = 0;
    while i < 4 {
        out[i] = a * x[i] + y[i];   // arrays pass by reference; `out` is mutated in place
        i = i + 1;
    }
}
```

```sh
mercuryc --run examples/saxpy_array.mer   # -> 12, 24, 36, 48 (one value per line)
mercuryc --run examples/dot.mer           # 120
```

## Architecture

```
source.mer
   │  lexer → parser → AST
   │  sema  (name resolution, type inference, COMPILE-TIME SHAPE CHECKING)
   │  mir_build  (lowering + matmul→GEMM dispatch + SIMD auto-vectorization:
   │              elementwise, reductions, FMA, fusion)
   ▼
Mercury IR (MIR)         one SSA IR that lowers progressively from "High" to "Low"
   │  optimization passes (mem2reg → SSA, const-fold, CSE, DSE, DCE, LICM, simplify-cfg;
   │                       inlining; op-graph fusion across tensor ops is planned)
   ▼
MIR (Low)
   ├──────────────► interpreter      (always available, zero deps; the reference oracle)
   ├──────────────► Cranelift backend (native JIT + object/exe; NO LLVM — the fast path)
   ├──────────────► GPU backend      (PTX + cudarc driver-JIT; --backend=gpu offload + --backend=gpu-native MIR→PTX)   [feature = "gpu"]
   └──────────────► LLVM backend     (textual IR for external clang/llc — always available, no feature flag)
```

The front-end, optimizer, the from-scratch **MIR interpreter**, *and* the **Cranelift native
backend** build and test with plain `cargo test` on any machine — no LLVM, no toolchain. The native
backend JIT-compiles in-process (and emits host objects) and is differentially tested against the
interpreter bit-for-bit. A **GPU backend** (NVIDIA, PTX via the driver JIT — no CUDA toolkit) is
behind `--features gpu`, and LLVM is an optional *textual-IR* emitter exposed via `--emit=llvm-ir` (always built; no feature flag).

## Status

Early development — built incrementally and openly. The full front-end, optimizer, interpreter, and
**native Cranelift backend** work today, including **matmul → tuned AVX2/FMA GEMM dispatch** (serial
and `@parallel`, incl. `nn.Linear` `A·Bᵀ`), SIMD auto-vectorization (elementwise + reductions), FMA
contraction, loop fusion, and `@parallel` multicore execution over fixed-size-array kernels. A
**GPU backend** (`--features gpu`; NVIDIA, PTX via cudarc driver-JIT) adds tensor-core GEMM, fused
flash-attention, norms, and a GPU-resident transformer layer, and **reverse-mode autodiff**
(`mercury_autodiff`, driven from the CLI via `--emit=grad` / `--train`) emits the training backward
pass and runs a fwd→bwd→optimizer (SGD/AdamW) loop — both gated against the interpreter oracle.
Tuples and structs (incl. nested struct-in-struct, **by-value parameters and `-> Struct` returns** via
an sret ABI, nested tuple fields `t.0.1`, and whole-aggregate assignment), pointers/references
(`&mut`/`*p`), and `loop`/`while`/`for` with `break`/`continue` (incl. **labeled loops** `'outer: …`
that a nested `break 'outer` / `continue 'outer` can target) execute end-to-end on both backends — as do
**`match`** (literal / range / or / enum-variant / tuple patterns, with guards), **C-style and
data-carrying (tagged-union) enums**, **slices `[]T`** (fat-pointer views with `.len()`, indexing,
iteration, and array→slice unsizing), top-level **`const`** values, **`let` tuple destructuring**, **radix `0xFF`/`0o17`/`0b1010` and char
`'A'` literals**, and **`"string"` literals** (typed `*u8`, rendered by `print`). And
**constant-shape tensors run** —
a `Tensor[f32, R, C]` parameter passes by base pointer and a multi-dimensional index `a[i, j]`
flattens to a row-major GEP, so the shape-typed surface *executes*, not just shape-checks — and a
matmul written in tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot and accumulate spellings)
dispatches to the same tuned `mercury_sgemm` microkernel as the flat `a[i*K+k]` form. **Symbolic-generic
tensor dimensions execute too** — `fn f<M, N>(t: Tensor[f32, M, N])` runs at any per-call size via
hidden runtime dim params (`tests/run/generic_shape.mer`). See the docs:

- [Benchmarks](BENCHMARKS.md) — honest cross-language results vs C, C++, and Rust, with methodology.
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
cargo run -p mercury_xbench --release      # cross-language benchmark vs C/C++/Rust (needs gcc/g++/rustc)
```

The native backend (Cranelift) is built in by default and needs no toolchain. The optional LLVM
backend emits textual IR only (for an external `clang`/`llc`), exposed via `mercuryc --emit=llvm-ir`
— it is always built and needs no feature flag.

## License

MIT — see [LICENSE](LICENSE).
