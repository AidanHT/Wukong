# Wukong

**A programming language for ML/DL tensor kernels with compile-time shape safety.**

Wukong (`.wk`) is for writing performance-critical tensor code — fused elementwise loops, tiled
matmuls, reductions, norms, custom ops — in a small explicit language rather than hand-rolling
CUDA/CUTLASS or stitching intrinsics in C/C++/Rust. Tensor dimensions live in the type system: a
`Tensor[f32, M, K]` at a call site must match the callee's contract, so shape mistakes are compile
errors (`E0501`/`E0502`), not silent wrong results at runtime.

The product is a **user-written `.wk` program through `wukongc`**. Backends — interpreter,
native [Cranelift](https://cranelift.dev) JIT/AOT, and optional GPU paths — are how that program
runs. Silent pattern recognizers that swap loop nests for tuned microkernels, and hand-written PTX in
the GPU stack, are current implementation, not the thesis.

It is **not** a high-level framework. There is no garbage collector, no hidden allocation, and no
hidden control flow. Every parallel loop has a schedule you chose (`@parallel` today; `@tile` parses
and is discarded).

## What you write

Ordinary loops over `Tensor[f32, M, K, …]` with shape-generic parameters (`M`, `N`, `?`), plus
`@parallel` for multicore execution. There is no whole-tensor `*` operator — matmul is an indexed
loop nest the compiler may recognize and lower. On CPU that means tuned `wukong_runtime`
microkernels behind the scenes; on GPU, `--backend=gpu` offloads five recognized kernel hooks
(`sgemm_nt`, `vmath`, `norm`, `sreduce`, `sgemm_nt_epi` on `wukong_interp::Accelerator` /
`GpuAccel` in `wukong_driver`), while `--backend=gpu-native` lowers MIR→PTX for a growing subset of
programs (257/363 `tests/run` fixtures today — naive helpers, no tensor cores). Flash attention,
conv, and serving benchmarks in the tree are Rust launchers, not `.wk` programs.

```sh
wukongc --run examples/saxpy_array.wk              # interpreter (default)
wukongc --run --backend=native tests/run/tensor_matmul.wk
wukongc --emit=mir -O2 tests/run/transcendental_kernel.wk   # inspect lowered MIR
```

`--run` defaults to the interpreter; add `--backend=native` for the Cranelift JIT. GPU backends
require building with `--features gpu` and a CUDA device. See `wukongc --help` for `--emit`,
`-O0`…`-O3`, `--grad-of` / `--train`, and the rest of the CLI.

## Why Wukong

Wukong compiles through a **from-scratch Cranelift backend — no LLVM, no external toolchain** on the
default path. Cross-language CPU measurements and honest GPU notes live in
**[BENCHMARKS.md](BENCHMARKS.md)** — read that file before quoting any number; this README does
not duplicate them. In summary, Wukong:

- **checks tensor shapes at compile time** (`wukong_sema`) — real semantic analysis, not a kernel
  compiler pretending to be a language;
- **lowers loop nests you write** to MIR, then optionally recognizes matmul, elementwise, reduction,
  and norm patterns for tuned CPU microkernels (`wukong_runtime`) or the five GPU offload hooks;
- **compiles fast** and runs competitive CPU kernels on recognized patterns — see
  [BENCHMARKS.md](BENCHMARKS.md) for methodology, standing index, and per-family numbers;
- **offers optional GPU backends** (`--features gpu`): recognizer offload (`--backend=gpu`) and
  whole-program MIR→PTX (`--backend=gpu-native`). Rust-side flash/conv/serving benches are launchers,
  not `.wk` programs.

## Language features

- **Compile-time shape safety.** Tensor shapes live in the type system: a `Tensor[f32, M, K]`
  parameter carries its dims — and its layout — into the callee contract, so handing a
  `Tensor[f32, 512, 512]` to a `Tensor[f32, 513, 512]` parameter, or returning the wrong shape, is a *type
  error* (E0501/E0502), not a segfault at 3 a.m. There is no whole-tensor `*` operator:
  whole-tensor arithmetic is rejected outright (E0401, "operate on elements in a loop"), and a
  matmul is written as an indexed loop nest — which the compiler then recognizes and dispatches
  (below). C/C++ cannot express this; Rust cannot express it ergonomically.
- **The knobs kernels need, as language constructs.** SIMD vector types (`vec[f32, 8]`), tensor data
  layout declared in the type (`Tensor[f32, 512, 512, .col_major]`, `.tiled(64, 64)`, strided) and
  parallel schedules (`@parallel`) are part of the language rather than a soup of intrinsics and
  `#pragma`s. Honest status: a user-written `vec[f32, 8]` *value* parses and type-checks but does not
  execute yet (`f32x8::load(..)` is a hard `C0001`); `@parallel` is the only attribute with a consumer
  today, and only its bare form — `@simd`/`@tile`/`@align`/`@extern`/`@export` and `@parallel`'s
  `grain =` argument parse and are inert — layout is
  enforced across call boundaries; the default row-major (contiguous) layout and `.col_major` both
  lower multi-dimensional indexing (`.col_major` is byte-identical to `a[i + j*M]`); `.strided` /
  `.tiled(…)` still reject indexing (`C0001`). Arena/scratch/pool allocator selection is not
  implemented (the language's only runtime-sized memory today is `alloc_<T>`/`free`).
- **Zero hidden cost.** No GC, no implicit copies of large aggregates, no surprise allocations.
- **Domain-aware optimization (implementation today).** The compiler **recognizes a matmul nest** (the
  `ikj` accumulate and `ijk` dot-product forms, incl. `nn.Linear` `A·Bᵀ`) and may lower it to a tuned
  **register-blocked, cache-tiled, packed AVX2/FMA GEMM microkernel**. It also **auto-vectorizes**
  elementwise loops (incl. branchy ones via if-conversion) and **reductions** to SIMD, contracts
  `x + y*z` to a **fused multiply-add**, **fuses** adjacent elementwise loops, and
  **auto-parallelizes** `@parallel` loops across cores. Underneath, an SSA optimizer (inlining, mem2reg,
  const-fold, CSE, DSE, DCE, LICM, loop unrolling) runs at `-O1` and above. Op-graph fusion across
  tensor ops is still planned **as a CPU MIR pass**; on GPU, `wukong_codegen_gpu`'s fusion planner and
  `--backend=gpu-native` can compile an eligible whole program into a cooperative megakernel. These
  recognizers are invisible to the user — you still write the loop nest in `.wk`.
- **Interop (planned).** A clean C ABI (`@extern("C")` / `@export`) is designed to call into
  BLAS/cuBLAS and embed Wukong kernels in C/C++/CUDA stacks. The attributes parse today but are
  neither validated nor consumed (an unknown attribute name is accepted silently); symbol
  export/import is not yet wired (see the roadmap).

## Runs the real GPT-2 124M end-to-end

Wukong compiles and runs **GPT-2 124M inference end-to-end as an ordinary `.wk` program**
([`examples/gpt2_infer.wk`](examples/gpt2_infer.wk)) on the native Cranelift-JIT backend. It loads the
**real pretrained OpenAI GPT-2 weights** — all **124,439,808** parameters — from a flat little-endian
f32 blob on disk (exported from HuggingFace `transformers` by
[`tools/export_gpt2.py`](tools/export_gpt2.py)) through the new `read_f32` file-I/O intrinsic, with the
prompt's token ids read via `read_i32`, and runs the full forward pass in Wukong source: token +
learned positional embedding, 12 pre-LayerNorm decoder blocks (biased QKV, 12-head causal attention,
tanh-GELU MLP, residuals), the final LayerNorm, and the tied LM head → logits `[5, 50257]`.

**It matches HuggingFace numerically.** The next-token logits agree with HuggingFace's reference to a
**relative max error of 1.87×10⁻⁶** (`max|Δ| = 2.44×10⁻⁴` against `max|ref| = 130.28`), and the argmax
next token is **1757 (" John")** for the prompt *"Hello, my name is"* — matching HuggingFace exactly.
This is checked by [`tools/verify_gpt2.py`](tools/verify_gpt2.py).

Reproduce it — one export, one run, one check — from the repo root:

```sh
python tools/export_gpt2.py                             # HF GPT-2 → data/gpt2/*.bin (+ reference logits)
wukongc --run --backend=native examples/gpt2_infer.wk   # runs the model, writes logits to disk
python tools/verify_gpt2.py                             # rel 1.87e-6, argmax 1757 → "GPT2 VERIFY PASS"
```

The export directory is resolved as `$GPT2_DATA_DIR`, then an explicit directory argument, then
`data/gpt2` — [`tools/verify_gpt2.py`](tools/verify_gpt2.py) uses the same order, so both tools agree
on any checkout. The exporter also regenerates the committed
[`examples/gpt2_config.wk`](examples/gpt2_config.wk) offset table (the single source of truth for
every weight offset) and writes a `MANIFEST.md` beside the blobs. The verifier treats the logits
artifact's **age** as part of its verdict (`--max-age-seconds`, default 3600) so a check cannot
re-assert a result for a run that never happened; pass `--no-age-check` to re-verify an archived
artifact.

**Honest scope.** This is a **numerical-correctness and capability** result, not a speed claim — **no
throughput comparison against PyTorch was measured**, and none is made here. It is **inference, not
training**. **Tokenization is external**: the `.wk` program consumes integer token ids produced by the
HuggingFace tokenizer; it does not tokenize text itself. The full 124M run is **native-only** — the
interpreter's slot memory cannot hold 124M parameters — so a reduced-config twin
([`examples/gpt2_infer_small.wk`](examples/gpt2_infer_small.wk)) runs the *same* forward pass
**bit-identically on both the interpreter and the native backend**, the CI differential +
opt-invariance gate that keeps the full model honest. (This is a new capability, orthogonal to the
performance results above.)

## The four signature features

```wukong
// (1) shape-typed tensors  (2) SIMD vectors  (3) explicit memory/layout  (4) parallelism
fn saxpy<N>(a: f32, x: Tensor[f32, N], y: Tensor[f32, N], mut out: Tensor[f32, N]) {
    @parallel @simd
    for i in 0..N {
        out[i] = a * x[i] + y[i];   // fused multiply-add, vectorized, parallelized
    }
}
```

That tensor/`@parallel` form **runs today**, on both backends — the constant-shape spelling and the
symbolic-generic `<N>` spelling alike (`tests/run/tensor_matmul.wk`, `tests/run/generic_shape.wk`,
`tests/run/generic_shape_parallel.wk`),
with a fixed-size array decaying to the rank-1 tensor view at the call. (`@simd` is accepted but
inert: loop auto-vectorization is unconditional and needs no attribute.) The same kernel written over
fixed-size arrays runs too:

```wukong
fn saxpy(a: f32, x: [f32; 4], y: [f32; 4], mut out: [f32; 4]) {
    let mut i: i32 = 0;
    while i < 4 {
        out[i] = a * x[i] + y[i];   // arrays pass by reference; `out` is mutated in place
        i = i + 1;
    }
}
```

```sh
wukongc --run examples/saxpy_array.wk   # -> 12, 24, 36, 48 (one value per line)
wukongc --run examples/dot.wk           # 120
```

`--run` defaults to the interpreter; add `--backend=native` for the same output through the Cranelift
JIT — the examples are gated to agree.

## Architecture

```
source.wk
   │  lexer → parser → AST
   │  sema  (name resolution, type inference, COMPILE-TIME SHAPE CHECKING)
   │  mir_build  (lowering + matmul→GEMM dispatch + SIMD auto-vectorization:
   │              elementwise, reductions, FMA, fusion)
   ▼
Wukong IR (MIR)         one block-parameter SSA IR; mir_build emits it scalar-and-low directly —
   │                    there is no second level (`--emit=mir-high` means MIR *before* the
   │                    optimizer, not a different IR level)
   │  optimization passes (mem2reg → SSA, const-fold, CSE, DSE, DCE, LICM, simplify-cfg;
   │                       inlining; partial loop unrolling (-O2, never reassociating); op-graph fusion across tensor ops is planned on the CPU path;
   │                       the GPU backend has it — see below)
   ▼
MIR
   ├──────────────► interpreter      (always available, no toolchain; the reference oracle)
   ├──────────────► Cranelift backend (native JIT + object/exe; NO LLVM — the fast path)
   ├──────────────► GPU backend      (PTX + cudarc driver-JIT; --backend=gpu offload + --backend=gpu-native MIR→PTX)   [feature = "gpu"]
   └──────────────► LLVM backend     (textual IR for external clang/llc — always available, no feature flag)
```

The front-end, optimizer, the from-scratch **MIR interpreter**, *and* the **Cranelift native
backend** build and test with plain `cargo test` on any machine — no LLVM, no toolchain. The native
backend JIT-compiles in-process (and emits host objects) and is differentially tested against the
interpreter bit-for-bit; the interpreter links the same `wukong_runtime` microkernels the native
backend calls, which is what keeps the two bit-exact. One platform caveat, and it is a *performance*
one only: the raw 256-bit AVX2 vec-kernel emitter hardcodes the Win64 argument registers and VEX.256
AVX2+FMA encodings, so it requires an **x86-64 Windows host with AVX2 + FMA**
(`wukong_mir::host_supports_vec_kernels`, which `avx2::host_supports_kernels` delegates to). On any
other host the vectorizer consults that same predicate and never builds a 256-bit recipe in the first
place, so those loops stay on the portable 128-bit CLIF path — the identical lowering
`WUKONG_P4_NO_256=1` forces, and result-identical by construction. Everything still compiles and
runs; only the widest lane is unavailable. The interpreter needs none of this. A **GPU backend** (NVIDIA, PTX via the driver JIT — no CUDA toolkit) is
behind `--features gpu`, and LLVM is an optional *textual-IR* emitter exposed via `--emit=llvm-ir` (always built; no feature flag).

## Status

Early development — built incrementally and openly. The full front-end, optimizer, interpreter, and
**native Cranelift backend** work today, including **matmul → tuned AVX2/FMA GEMM dispatch** (serial
and `@parallel`, incl. `nn.Linear` `A·Bᵀ`), SIMD auto-vectorization (elementwise + reductions), FMA
contraction, loop fusion, and `@parallel` multicore execution over fixed-size-array kernels. An
**optional GPU backend** (`--features gpu`; recognizer offload via `--backend=gpu`, MIR→PTX via
`--backend=gpu-native`) runs a growing subset of `.wk` programs on NVIDIA hardware. **Reverse-mode
autodiff**
(`wukong_autodiff`, driven from the CLI via `--emit=grad` / `--train`) emits the training backward
pass and runs a fwd→bwd→optimizer (SGD/AdamW) loop — both gated against the interpreter oracle.
Tuples and structs (incl. nested struct-in-struct, **by-value parameters and `-> Struct` returns** via
an sret ABI, nested tuple fields `t.0.1`, and whole-aggregate assignment), pointers/references
(`&mut`/`*p`), and `loop`/`while`/`for` with `break`/`continue` (incl. **labeled loops** `'outer: …`
that a nested `break 'outer` / `continue 'outer` can target) execute end-to-end on both backends — as do
**`match`** (literal / range / or / enum-variant / tuple patterns, with guards), **C-style and
data-carrying (tagged-union) enums**, **slices `[]T`** (fat-pointer views with `.len()`, indexing,
iteration, and array→slice unsizing — and, since every kernel operand resolves through one
base-pointer helper, a legal operand to the whole recognized-kernel family: GEMM/GEMV, the fused
norms, the scans, and the streaming elementwise kernels), top-level **`const`** values, **`let` tuple
destructuring**, **radix `0xFF`/`0o17`/`0b1010` and char
`'A'` literals**, and **`"string"` literals** (typed `*u8`, rendered by `print`). And
**constant-shape tensors run** —
a `Tensor[f32, R, C]` parameter passes by base pointer and a multi-dimensional index `a[i, j]`
flattens to a row-major GEP, so the shape-typed surface *executes*, not just shape-checks — and a
matmul written in tensor notation (`c[i,j] = Σ a[i,k]·b[k,j]`, both the dot and accumulate spellings)
dispatches to the same tuned `wukong_sgemm` microkernel as the flat `a[i*K+k]` form. **Symbolic-generic
tensor dimensions execute too** — `fn f<M, N>(t: Tensor[f32, M, N])` runs at any per-call size via
hidden runtime dim params (`tests/run/generic_shape.wk`). See the docs:

- [Benchmarks](BENCHMARKS.md) — honest cross-language results vs C, C++, and Rust, with methodology.
- [Language guide](docs/language-guide.md) — the language surface, with an honest maturity legend.
- [Compiler internals](docs/internals.md) — architecture, MIR, optimizer, the native backend, and testing.
- [Roadmap & limitations](docs/roadmap.md) — what runs, what's checked-only, what's planned.
- [LLVM setup](docs/llvm-setup.md) — optional textual-IR backend (the native path uses Cranelift, no LLVM).
- [Language direction](docs/LANGUAGE_DIRECTION.md) — thesis for generic kernel lowering, not more silent PTX families.
- [Master plan](docs/WUKONG_MASTER_PLAN.md) — north star for the language (including AI editors); GPU SOTA is a chapter, not the product.
- [Docs index](docs/README.md) and [crate READMEs](crates/README.md) — per-folder orientation.

## Building

```sh
cargo build                 # the whole compiler incl. the native Cranelift backend — no LLVM
cargo test                  # unit + golden + end-to-end + differential (interp vs native) tests
cargo check --features gpu --all-targets  # other half of the gate: `cargo test` never builds the GPU
cargo run -p wukongc -- --help
cargo run -p wukongc -- --run examples/fib.wk
cargo run -p wukong_bench --release -- tests/run examples bench/kernels   # optimizer report
cargo run -p wukong_xbench --release      # cross-language benchmark vs C/C++/Rust (needs gcc/g++/rustc)
```

The native backend (Cranelift) is built in by default and needs no toolchain. The optional LLVM
backend emits textual IR only (for an external `clang`/`llc`), exposed via `wukongc --emit=llvm-ir`
— it is always built and needs no feature flag.

The GPU backend is the only opt-in part of the tree, so `cargo test` neither compiles nor runs it:
pair every run with `cargo check --features gpu --all-targets`. With `--features gpu` on a machine
that really has a device, set `WUKONG_GPU_REQUIRED=1` (and `WUKONG_PEER_REQUIRED=1` for the peer
sweeps) so a missing device turns the suite's `[skip]` lines into failures instead of a silently
passing test.

## License

MIT — see [LICENSE](LICENSE).
