# Wukong

**A low-level, low-abstraction systems language built for ML/DL compilers and high-performance tensor kernels.**

Wukong is the language you reach for *instead of* C, C++, or Rust when you are writing the
performance-critical core of a machine-learning stack: fused elementwise kernels, tiled matmuls,
attention microkernels, custom ops, and the compiler passes that generate them.

It is **not** a high-level framework. There is no garbage collector, no hidden allocation, and no
hidden control flow. Every byte of memory comes from an allocator you named, every SIMD lane is one
you asked for, and every parallel loop has a schedule you chose.

## Why Wukong

Wukong compiles to native code through a **from-scratch [Cranelift](https://cranelift.dev) backend —
no LLVM, no external toolchain**. In a head-to-head cross-language benchmark (same kernel in each
language, one timing harness; see **[BENCHMARKS.md](BENCHMARKS.md)**), Wukong:

- **compiles fast** — the metric that dominates real ML edit-run iteration. Apples-to-apples
  *compiler-to-object* (`wukongc --emit=obj -O2` vs `gcc/g++/rustc -O2 -c`, same artifact, same
  machine): **~7–12× faster** (`wukong_bench compile-vs`; ~7–9× measured this run). The larger **~100–680× (geomean ~306×)**
  figure is *time-to-running-code*: Wukong JIT-compiles in-process while C/Rust must spawn a full
  toolchain **and link a shared object** (`-O3 -march=native`) — a real advantage for the JIT/embedding
  workflow, but not a compiler-vs-compiler number, so it is disclosed as such, never as the headline;
- **wins matmul/GEMM**, the flagship ML kernel: the compiler recognizes a matmul nest (incl. the
  `nn.Linear` `A·Bᵀ` form) and dispatches it to a tuned register-blocked, cache-tiled, packed
  **AVX2/FMA** microkernel — **~3–3.6× faster single-thread** at **~110–120 GFLOP/s ≈ 90% of one
  P-core's AVX2-FMA roofline** (and **~1.1–1.3× over the tuned `matrixmultiply` Rust crate**, at
  **~80–104% of oneMKL single-core**), and **up to ~18× parallel** on plain `C = A·B`, **up to ~104× on `nn.Linear`**
  (where naive C leaves the reduction latency-bound), the lead *growing with matrix size*. Against
  the honest SOTA bar — **multi-threaded oneMKL** — Wukong's `@parallel` GEMM runs a
  **size-keyed BLIS/MKL-style 2D block-parallel decomposition** (per-thread L2-resident C blocks;
  per-worker packing — measured the right locality trade in both ABBA orderings — with dynamic
  block-claiming the 2026-07-11 default) and stands, same-run MKL-anchored, at **~98–99% at 512³,
  ~104% at 1024³, 91% at 2048³, 93% at 4096³, and ~94–110% at 256³** (the dynamic-claiming default
  closed the old mid-size gap; against MKL's own power-degraded rounds the ratios read far higher,
  which is why only same-run figures are quoted). On the skinny transformer shapes it holds
  **75–112% of MKL-all**, four of six at or above parity — the sole laggard the 151-MFLOP
  128×768·768ᵀ, overhead-bound where MKL itself scales only ~2.1×. Single-core it is **~80–104% of
  MKL-1c across sizes, at or above parity at 2048³/4096³**. (Disclosure: threaded MKL itself swings
  ~1.4–2× with this laptop's power state, so ratios are same-run only and reported as ranges; the
  residual mid-size gap is parallel-grain scaling, honestly open);
- **dispatches the whole transformer/training kernel surface** to tuned microkernels: the
  **weight-gradient GEMM** `dW=Aᵀ·B` (training backward) **~2.6–9.2× single / ~3.1–13.9× parallel**,
  the **fused FFN** `silu(A·Bᵀ)` **~24–26×**, **RoPE** rotary embedding **~29–54×** (up to **~156×
  parallel**), and the training-backward kernels (activation/softmax/LayerNorm-RMSNorm backward,
  cross-entropy) **~3–13×**. *(Corrected 2026-08-04. The weight-gradient figure previously read "up
  to ~128× single / ~445× parallel"; that was measured against a C peer written `ijk` with **both**
  operands read column-strided, not the natural `kij` nest. **Strided column reductions**, previously
  listed here at ~29–50×, are now measured as **a tie at best and a 1.8× loss** against a peer written row-outer,
  and have been removed from this list. See the [peer-strength
  correction](BENCHMARKS.md#-peer-strength-correction--2026-08-04).)*;
- **wins the transcendental/activation family ~4.7–11.5× vs C** (**~28× under `@parallel`**) — the
  cleanest compute-bound win. Against **Intel oneMKL VML**, the hand-tuned vector-math SOTA, the
  honest standing is narrower than this file claimed on 2026-08-06 and the "caught VML" line is
  **retracted**: at *matched accuracy* (VML's `VML_LA` mode, which measures 2 ULP for exp and 4 ULP
  for log on the timed band — exactly where Wukong measures 2 and 4) **exp and log are a tie**, exp
  reading 1.09–1.20× *slower* and log between 1.09× slower and 1.07× faster at n = 2¹⁶ and 2²⁰,
  over five same-run best-of-40 ABBA rounds. Only at n = 2²³, where Wukong's non-temporal store
  regime engages, does either come near this machine's ~1.4× noise floor — and there only log
  clears it (1.48–1.66×), exp straddling it at 1.37–1.54×. **tanh is the real win at
  2.85–3.24× vs VML's fastest mode** — with the disclosure that Wukong's tanh is 44 ULP against
  VML's ≤1 ULP, so it trades accuracy for speed. Comparisons against VML's *default* HA mode
  (~0.5 ULP) flatter Wukong and are not quoted as wins.
  What **is** solid is the same-binary engineering: the exp/log gap once recorded as algorithmic was
  the dispatch calling its 8-lane kernel through a function pointer, which without crate-wide AVX
  means the `__m256` crosses the call through memory. Inlining it is worth **1.20–1.78×** with a
  `WUKONG_VMATH_FNPTR=1` kill-switch and is **bit-identical on all 2³² f32** — accuracy stays exp
  ≤2 ULP (1.625e-7 rel) and log ≤12 ULP (6.924e-7 rel), both re-verified exhaustively rather than
  sampled. Wukong
  dispatches a pure
  `out[i]=f(x[i])` loop for **35** functions
  (`exp`/`log`/`exp2`/`log2`/`exp10`/`log10`/`cbrt`/`expm1`/`log1p`/`tanh`/`sigmoid`/`gelu`/`silu`/
  `softplus`/`softsign`/`logsigmoid`/`mish`/`sin`/`cos`/`tan`/`atan`/`asin`/`acos`/`erf` plus the
  hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh` — the transformer activations plus **RoPE**'s
  `sin`/`cos`, the inverse trig, the exact-GELU `erf`, the stable `expm1`/`log1p`/`logsigmoid`, and the
  hyperbolic/Poincaré-embedding inverse trio) to a **256-bit AVX2 ≈1-ULP poly kernel**, where gcc/rustc
  call scalar `libm` and **cannot vectorize a loop containing the call**;
- **wins fused row-norms** (`softmax`/`LayerNorm`/`RMSNorm`, incl. the learned-γ/β affine form)
  **~1.9–6.6×**; **convolution** (im2col + GEMM) is **~1.55×**, and a slight loss (1.12×) against the
  same direct-convolution C at `-ffast-math` *(corrected 2026-08-04: the previous ~6–7× was measured
  against a peer whose buffers were not `restrict`-qualified, which cost gcc 4.3× on that kernel)*;
- **runs a full 12-layer GPT-2-class transformer end-to-end** (d=768, 12 heads, causal attention,
  GELU MLP — ordinary Wukong source through the real pipeline, gated bit-exact against the
  interpreter and cross-checked <2e-6 against C and PyTorch outputs): **~19–21× idiomatic C** and
  **3.6–4.9× `-ffast-math` C single-core**. Against the honest PyTorch bar — **`torch.compile`
  (TorchInductor max-autotune, fullgraph, warmed; beating eager does *not* count)** — Wukong is
  **faster at both batch sizes**: single-thread it beats compiled-torch-1T at S=128 and S=512, and
  multicore (since the `@parallel` head-loop region shipped, 2026-07-10 — independent-iteration
  loops with body-local scratch outline to a parallel region, bit-exact by construction) it runs
  **1.39–1.81× faster than all-threads compiled torch at S=512 and 1.07–1.43× faster at S=128**
  (three independent same-day rounds, isolated per-side probes with torch measured at its best; it
  also beats torch's strongest *eager* config, 1.07–1.37× at S=512). Model `@parallel` scaling
  reaches **4.7× on 16 cores** (default 4.44×, best 5.22×) — tracking this hybrid part's ~5–5.4×
  same-run MKL ceiling, with T=1 at serial parity;
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
  (and is 3.6–5× the unfused cuBLAS chain), trailing cuDNN only at long context (S≥2048) — with the
  **D=128 `ldmatrix` kernel (the Llama-class head dim, 1.11–1.20× cutlass mem-efficient at S≤1024)
  now the production D=128 dispatch**,
  int8 GEMM **~180–237× naive CUDA-C**, **95.7% of the 192 GB/s HBM peak**, and **0.76 ms cold GPU
  compile vs Triton's 30–120 s**.

  > **Device scope (2026-08-06, extended 2026-08-09):** every GPU figure in the bullet above was
  > measured on an **NVIDIA RTX 4050 Laptop GPU** (Ada, `sm_89`, **20 SMs**, 6 GB, **~192 GB/s**,
  > power-capped ~30–50 W) under **Windows/WDDM**, with only the peers that box can host — cuBLAS /
  > IMMA / cuBLASLt and cuDNN via the redistributable DLLs, NVRTC-compiled CUDA-C, and PyTorch in
  > **eager** mode (Triton does not install on Windows); **no CUDA toolkit**, so no CUTLASS,
  > FlashAttention, Marlin or vLLM build was possible. Every tile, stage depth and occupancy
  > crossover behind those figures was swept against those twenty SMs. These are properties of that
  > instrument; **do not extrapolate them to datacenter parts.** The datacenter retarget, including
  > re-measurement against the peers a Linux cloud box can build, is `GPU_RETARGET_PLAN.md`.
  >
  > Three of the claims above need more than a scope note, and `BENCHMARKS.md`'s standing index
  > carries them in full: the **PyTorch** comparison is against **eager** and is *not re-earned*
  > against `torch.compile`, which is the real framework bar and is native on Linux; the **flash
  > standings** — the long-context loss *and* the short-context wins — are occupancy verdicts on
  > 20 SMs and may move in **either** direction elsewhere; and the **CUDA-graph** launch-overhead
  > multiples are Windows/WDDM numbers that should be expected to shrink on Linux before the GPU
  > changes at all. "No CUDA toolkit" describes what *Wukong* needs to build and run — it is never a
  > claim that no stronger peer exists.
  >
  > **The CPU figures on this page carry their own open debts** — an end-to-end model ratio that is
  > an upper bound pending re-measurement, ~40 rows whose Rust column needs re-measuring after a
  > `noalias` fix, a C++ column that only ever existed in three sections, and a general-code suite
  > whose tables are superseded with no replacement. `BENCHMARKS.md` opens with a per-family standing
  > index; read it before quoting any number from this README.

The domain-aware paths (GEMM, the `vmath` transcendentals, the `velem` streaming elementwise, the
fused norms) emit **true 256-bit AVX2/FMA** via hand-written runtime microkernels — and the
**general** loop vectorizer reaches the same width by a second route: a vectorizable f32 loop body is
captured as a flat `VecKernel` recipe and assembled into raw 256-bit AVX2 machine code
([`crates/wukong_codegen_cranelift/src/avx2.rs`](crates/wukong_codegen_cranelift/src/avx2.rs)),
sidestepping the 128-bit `f32x4` cap on Cranelift's vector SSA that the CLIF path is stuck behind
(`WUKONG_P4_NO_256=1` forces the 128-bit path — a result-identical A/B knob and kill-switch). So
even the
memory-bandwidth-bound elementwise kernels are now small **wins** (saxpy ~1.25–1.45×, poly ~1.1–1.2×,
widening to ~1.3–1.6× at realistic >L3 tensor sizes via non-temporal stores); the one honest **tie** left is
`relu` at an L3-resident size, where both languages are pinned to the same cache bandwidth.

Where Wukong is built to win for the ML/DL niche:

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
  enforced across call boundaries but only the default row-major (contiguous) layout lowers a
  multi-dimensional index, and arena/scratch/pool allocator selection is not implemented (the
  language's only runtime-sized memory today is `alloc_<T>`/`free`).
- **Zero hidden cost.** No GC, no implicit copies of large aggregates, no surprise allocations.
- **Domain-aware optimization that runs today.** The compiler **recognizes a matmul nest** (the
  `ikj` accumulate and `ijk` dot-product forms, incl. `nn.Linear` `A·Bᵀ`) and lowers it to a tuned
  **register-blocked, cache-tiled, packed AVX2/FMA GEMM microkernel** — the way XLA/TVM/oneDNN lower a
  matmul op. It also **auto-vectorizes** elementwise loops (incl. branchy ones via if-conversion) and
  **reductions** to SIMD, contracts `x + y*z` to a **fused multiply-add**, **fuses** adjacent
  elementwise loops, and **auto-parallelizes** `@parallel` loops across cores — things a
  general-purpose C compiler won't do to naively-written source. Underneath, an SSA optimizer
  (inlining, mem2reg, const-fold, CSE, DSE, DCE, LICM, loop unrolling) removes ~42% of IR ops on the
  benchmark kernels
  (~48–54% on the heavy transformer/GEMM kernels). Op-graph fusion across tensor ops is still planned
  **as a CPU MIR pass**; on the GPU it exists today — `wukong_codegen_gpu`'s fusion planner classifies
  the recognized op graph and `--backend=gpu-native` compiles an eligible whole program into a single
  cooperative megakernel.
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
contraction, loop fusion, and `@parallel` multicore execution over fixed-size-array kernels. A
**GPU backend** (`--features gpu`; NVIDIA, PTX via cudarc driver-JIT) adds tensor-core GEMM, fused
flash-attention, norms, and a GPU-resident transformer layer, and **reverse-mode autodiff**
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
