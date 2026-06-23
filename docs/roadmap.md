# Mercury Roadmap & Known Limitations

Mercury is built openly and incrementally. This page is an honest snapshot of what works, what is
checked-but-not-executed, and what is planned — so expectations match reality.

## Works end to end (interpreter `--run`, **and native code** `--backend=native`)

Two execution backends now run the full language and agree bit-for-bit (a differential gate proves
it across opt levels): the zero-dependency tree-walking interpreter (the reference oracle) and a
from-scratch **Cranelift native backend** (JIT for `--run --backend=native`, object/exe via
`--emit=obj|exe`) — **no LLVM toolchain required**. See `BENCHMARKS.md` for cross-language numbers.

- Modules, functions (including recursion and mutual recursion), and direct calls.
- `let`/`let mut`/`const`, shadowing, block-as-expression values.
- Integers (`i8..i64`, `u8..u64`, `usize`/`isize`), `bool`, and floats. `f32` is computed at **`f32`
  precision** (interpreter and native agree exactly); **both `bf16` and `f16` are real 2-byte storage**
  rounded to that grid (round-to-nearest-even) on store and on the cast, with `f32` compute. bf16 rounds
  with cheap inline bit-math (it's the top 16 bits of an `f32`); f16's IEEE-half layout has no such
  shortcut and Cranelift x64 has no f16 convert lowering, so f16 cast/load/store call shared `half`-crate
  shims (`mercury_f32_to_f16_bits`/`mercury_f16_bits_to_f32`) the interpreter uses too — bit-exact by
  construction. `f64` is full. (The GPU adds tensor-core fp16/bf16/fp8 GEMM on top.)
- **Mixed-precision (bf16 *and* f16) CPU suite → SIMD dispatch**: low-precision `[bf16]`/`[f16]` arrays
  read through an `as f32` widening cast (lossless: `<<16` for bf16, F16C `vcvtph2ps` for f16) with
  **f32 accumulate/compute** — the standard ML contract — are recognized and lowered to half-precision
  runtime kernels. Symmetric across both precisions:
  - **reductions** `dot`/`sum` (`mercury_{dot,sum}_{bf16,f16}`) — ~3.0–3.5× vs C for dot, ~6–8× for
    sum, growing as the working set spills L3 (`tests/run/reduce_{bf16,f16}.mer`);
  - **max/min/absmax** (`mercury_reduce_{bf16,f16}(x,n,op)`) — the per-tensor absmax is the
    symmetric-quantization scale; exact (max/min round nothing) (`reduce_bf16_minmax.mer`);
  - **streaming axpby** `out = a·x + b·y` (`mercury_axpby_{bf16,f16}`), half-in/f32-out, ~1.3× ≫ L3
    (requires two additive terms; a 1-term scale would force a `0*inf` the source lacks);
  - **activations** — the full 36-op transcendental set over a half-precision input
    (`mercury_vmath_{bf16,f16}`, `out[i] = f((x[i] as f32))`).
  The recognizers are precision-generic (`match_lowp_reduction`/`match_lowp_axpby`/`match_vmath_stmt`),
  and the interpreter marshals through the identical kernel, so native == interp bit-for-bit. C/Rust
  can vectorize neither a `libm` call nor the half→f32 widen, so the gap is structural. A half-precision
  *output* (→ ~2× on the streaming ops) needs a narrowing store and is future work.
- All arithmetic/comparison/bitwise/boolean operators, compound assignment, casts.
- `if`/`else` (statement and value position), `while`, `for … in a..b [step s]`.
- **Fixed-size arrays** `[T; N]`: literal/repeat init, indexed load/store, array parameters passed
  by base pointer (out-params). Real kernels run: dot, SAXPY, GEMM, matmul, ReLU, clamp, transpose.
- **Matmul → GEMM dispatch**: the compiler recognizes a matmul loop nest (the `ikj` accumulate and
  `ijk` dot-product forms, including the `nn.Linear` `C = A·Bᵀ` spelling) and lowers the whole nest
  to a tuned register-blocked (6×16), cache-tiled, packed **AVX2/FMA** microkernel in the runtime —
  the way XLA/TVM/oneDNN lower a matmul op. Serial and `@parallel`. Beats gcc/rustc's naive nest
  ~2.4–3.5× single-thread and up to ~13× parallel on `C = A·B` (~19–70× on `nn.Linear`), the lead
  growing with size. Dimensions may be compile-time literals **or runtime values** (function
  params/locals): the recognizer checks strides symbolically, so a general matmul function dispatches
  to the kernel, not just fixed-size benchmark kernels. The two factors may even be the **same array**
  (a Gram matrix `A·Aᵀ`, or self-attention `Q·Kᵀ` sharing a buffer) — both sides are read-only. The
  interpreter calls the identical kernel (marshalling its memory), so the two stay bit-exact. The
  **transposed-A weight-gradient** form `C = Aᵀ·B` (`dW = dYᵀ·X`, A stored `[k,m]` with the
  contraction axis outermost) is recognized too and dispatched to `mercury_sgemm_tn`, which transposes
  A once then reuses the same NN microkernel — so the training backward pass leaves the scalar nest
  (`tests/run/matmul_tn.mer`). The **bf16/f16 mixed-precision** `nn.Linear` (`[bf16]`/`[f16]` inputs
  widened `as f32`, f32 accumulate) likewise dispatches to `mercury_sgemm_{bf16,f16}_nt` — a lossless
  widen prepass then the same tuned kernel — ~25× the idiomatic bf16 C (`tests/run/linear_{bf16,f16}.mer`),
  and its **fused FFN epilogue** (`act(A·Bᵀ + bias)`) folds to `mercury_sgemm_{bf16,f16}_nt_epi` — bias +
  activation in the GEMM writeback for free (`tests/run/linear_{bf16,f16}_ffn.mer`; see epilogue fusion below).
  The mixed-precision **weight-gradient** `C = Aᵀ·B` (`dW = dYᵀ·X`, the training backward) is recognized
  too → `mercury_sgemm_{bf16,f16}_tn` (widen prepass → the f32 `mercury_sgemm_tn`), closing the half
  training-backward gap where the naive nest loses to both the un-vectorizable widen and the column-strided
  A reads (`tests/run/matmul_{bf16,f16}_tn.mer`).
- **Batched matmul → per-head GEMM dispatch**: a matmul nest wrapped in a batch loop, with each index
  carrying a per-batch base offset (`x[h*S*D + i*K + k]` — the shape of **multi-head attention**, one
  matmul per head), also dispatches. The recognizer peels the offset off each flattened index (it
  must be invariant in the matmul's own `i,j,k`) and the kernel call GEPs each base pointer by it, so
  every head runs the tuned microkernel instead of a scalar nest. Both the `Q·Kᵀ` and `P·V` matmuls of
  an MHA forward dispatch (see `tests/run/{batched_matmul,multi_head_attention}.mer`).
- **Matrix transpose → cache-blocked kernel**: the nest `for i { for j { dst[j*R+i] = src[i*C+j] } }`
  dispatches to a `B=32` cache-blocked `mercury_transpose_f32[_parallel]`. The naive transpose writes
  `dst` with stride `R` (a cache miss per element for large `R`) and gcc/rustc do not loop-tile it at
  `-O3`, so the blocked kernel wins ~1.5× single-core / ~9–14× `@parallel` on this memory-bound layout
  op (attention score / weight-layout transposes). A permutation, so bit-exact (`tests/run/transpose_f32.mer`).
  **bf16/f16** transposes dispatch to the same blocked kernel at 16-bit width (`mercury_transpose_u16`, one
  kernel for both — a transpose moves the raw bits) for the half-precision KV/attention layouts (`transpose_bf16.mer`).
- **Column reduction → SIMD colsum kernel**: the column-outer nest `for j { for i { s += x[i*N+j] }; out[j]=s }`
  (the bias gradient `db = Σ_batch dY`, batch sum, reduce-along-axis-0) dispatches to
  `mercury_colsum_f32[_parallel]`, which streams `x` row-major and accumulates eight columns at a time into
  a cache-resident `out[]`. The naive form strides `x` down the rows *and* — verified on the emitted assembly —
  gcc/rustc leave it fully scalar (no `vaddps`), so the kernel wins ~29–47× single-core / ~52–55× `@parallel`.
  Each column sums in `i`-ascending order, so it is bit-exact (`tests/run/colsum.mer`).
- **Transformer building blocks compose**: a transformer FFN (`gelu(x·W1ᵀ)·W2ᵀ`), scaled
  dot-product attention (`softmax(Q·Kᵀ)·V`), **multi-head** attention (the batched per-head form) and
  its **causal** (decoder/autoregressive) variant, 2D convolution (im2col + matmul), and a full
  pre-norm (Llama-style) transformer block all lower with their matmuls dispatched to the GEMM kernel
  and their softmax/GELU/RMSNorm vectorized — and run bit-identically on both backends (see
  `tests/run/{ffn_block,attention,multi_head_attention,causal_attention,conv_im2col,transformer_block,rmsnorm,log_softmax}.mer`).
- **SIMD auto-vectorization**: straight-line elementwise loops (incl. branchy ones via
  if-conversion) lower to 128-bit vector ops, 4×-unrolled, with a scalar remainder — automatically,
  on the native backend. saxpy/poly/relu/relu6 vectorize.
- **FMA contraction**: a float `x + y*z` becomes one fused multiply-add (`Op::Fma`, a hardware
  `vfmadd`), on both the scalar and vector paths; the interpreter mirrors it with `mul_add`, so the
  two backends stay bit-identical.
- **Reduction vectorization**: a float reduction `s = s + x[k]*y[k]` / `s += ..` lowers to
  vector-lane accumulators (independent FMA chains) + a horizontal reduce + scalar remainder, turning
  the latency-bound serial sum into a throughput-bound one. `dot` runs ~2.7× faster than serial C.
  `fmax`/`fmin` reductions (`m = fmax(m, x[i])`, softmax's row-max) vectorize the same way.
- **`@parallel` reduction → multicore reduction kernel**: a reduction loop in a `@parallel` function
  (dot `x[k]*y[k]`, ssd `(x[k]-y[k])²`, or the unary sum `x[k]`) is **dispatched to a deterministic
  multicore reduction kernel** (`mercury_sreduce_f32_parallel`), spreading the stream across cores to
  aggregate memory bandwidth — `dot@parallel` ~7.9×, `ssd@parallel` ~8.6× faster than single-threaded
  C. The parallel sum is bit-identical to the serial one (fixed-size chunks independent of core count
  + ascending partial combine), so the differential oracle holds.
- **Transcendental intrinsics**: `sqrt`/`rsqrt`/`cbrt` (the root family; `cbrt` for LAB color /
  variance-stabilizing transforms), `exp`/`log`/`exp2`/`log2`/`exp10`/`log10` (≈1-ULP `f32`
  minimax polynomials; base-10 for decibel/log-scale features), `expm1`/`log1p` (Kahan-stable `eˣ−1` / `ln(1+x)`, ≈1-ULP near 0),
  `pow` (= `exp(y·log(x))`), `atan2`/`hypot` (the two-arg geometry pair — full-circle angle, overflow-safe
  2-norm; a `for j { out[j]=f(x[j],y[j]) }` loop dispatches to the 256-bit two-input `mercury_vmath2_f32`),
  `erf` (Abramowitz–Stegun, for **exact** GELU
  `0.5·x·(1+erf(x/√2))`), `sin`/`cos` (Cephes minimax + quadrant reduction, for **RoPE** rotary
  position embeddings), `tan`/`atan`/`asin`/`acos` (Cephes; the inverse trig for angle/geometry/3D-vision
  ops), the activation
  family `tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`softsign`/`logsigmoid`/`mish`/`selu`/`tanhshrink`/
  `hardsigmoid`/`hardswish` (`softsign` a bounded poly activation; `logsigmoid` the stable
  BCE-with-logits primitive), the full hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`
  (the inverse trio for hyperbolic/Poincaré embeddings + the Fisher z-transform), and `fmax`/`fmin` —
  all built from primitive ops both backends agree on bit-for-bit. A pure `out[i] = f(x[i])` loop for
  **any of the 35** single-arg transcendentals (incl. the stable `expm1`/`log1p`) is **dispatched
  to a tuned 256-bit AVX2/FMA kernel** (`mercury_vmath_f32`) — the width Cranelift's general (128-bit)
  vectorizer can't reach; a *composed* use auto-vectorizes the inlined poly at 128-bit. So softmax, layernorm,
  GELU (tanh and exact erf), SiLU/swish, ELU, softplus, mish, tanh, RoPE, and **log-softmax /
  cross-entropy** run on SIMD instead of scalar `libm` — **~4–11.5× faster** than gcc/rustc's scalar
  `libm` call (which can't vectorize a loop containing it; ~28× across cores under `@parallel`). See `tests/run/{transcendental,softmax,
  layernorm,gelu,elu,leaky_relu,softplus,mish,activations,log,erf,trig,ihyp,atan,log_softmax,ffn_block}.mer`.
- **Convolution via im2col + GEMM**: a conv written as an im2col gather followed by a matmul has its
  matmul recognized and dispatched to the tuned GEMM microkernel (the XLA/cuDNN lowering), so Mercury
  runs a 3×3 conv **~6–7× faster** than idiomatic hand-written direct convolution in C. See
  `tests/run/conv_im2col.mer`.
- **Operator fusion**: adjacent same-range elementwise loops (e.g. a linear map then ReLU) fuse into
  one loop when the combined body is dependence-safe; CSE then forwards the intermediate through
  registers rather than memory.
- **GEMM epilogue fusion**: a recognized `nn.Linear` matmul (`C = A·Bᵀ`) immediately followed by a
  bias-add / activation loop over `C` (`C[i,j] = act(C[i,j] [+ bias[j]])`) fuses into one
  `mercury_sgemm_nt_epi` call that folds the bias + activation into the microkernel's C-tile
  writeback — so `C` is written once instead of paying a separate read-modify-write pass over it. The
  saving is a fraction of the C-pass traffic, so it grows as K shrinks: ~1.0× at 512³ (compute-bound,
  no harm), ~1.34× at K=64/N=2048, ~1.65× at K=32/N=4096 — exactly the small-K/large-N projections
  (attention-output, down-projection). Serial; the activation set is identity (bias-only), **ReLU,
  GELU, and SiLU** — the transformer FFNs — with **bias optional**, so the bias-free `silu(x·Wᵀ)`
  **SwiGLU** projection (LLaMA/Mistral) fuses too. Both backends call the identical kernel, so it
  stays bit-exact. See `tests/run/{linear_bias_relu,linear_bias_gelu,linear_silu}.mer`. The
  **bf16/f16 mixed-precision** FFN fuses the same way — a half matmul + its bias/activation loop folds
  to `mercury_sgemm_{bf16,f16}_nt_epi` (the widen prepass feeds the identical f32 epilogue), so the
  mixed-precision transformer FFN gets bias + activation for free (`tests/run/linear_{bf16,f16}_ffn.mer`).
  The **residual projection** `x = x + act(x·Wᵀ + bias)` (the transformer skip connection) also folds
  to `mercury_sgemm_nt_epi`, with **beta = 1** so the kernel accumulates `act(x_residual + A·Bᵀ + bias)`
  in its writeback. The accumulate store would otherwise block the matmul recognizer and drop the whole
  nest to a scalar loop, so this recovers the full GEMM dispatch + the fused residual/bias/activation
  with no backend change (`match_matmul_residual`, `tests/run/linear_residual{,_relu}.mer`).
- **`@parallel`** functions execute across CPU cores (rayon runtime); the per-core chunk is itself
  vectorized. The interpreter runs the same range sequentially, so results stay differential-equal.
- Intrinsics `print`/`println`/`assert`.
- The optimizer (`-O0..-O3`), backed by CFG and dominator analyses: whole-program **inlining** of
  leaf functions, **mem2reg** (alloca → SSA), constant folding, algebraic simplification, CFG cleanup
  with block merging, dead/trivial block-parameter elimination, DCE, dominator-tree CSE with load
  forwarding, DSE, and **loop-invariant code motion**. Guarded by an `-O0`-vs-`-O{1,2,3}` differential
  test and post-pass MIR verification; across the run suite and kernels it removes ~48% of IR ops
  (54–60% on the heavy kernels) and runs ~1.5–2.5x faster than `-O0`.

## GPU backend (NVIDIA RTX 4050, behind `--features gpu`)

A third backend, `mercury_codegen_gpu`: being a compiler, it **emits PTX text** and **driver-JIT-loads
it via `cudarc`** (`cuModuleLoadData` — the driver's built-in PTX→SASS JIT, so **no `nvcc`/`ptxas`/CUDA
toolkit** is needed to build or run, only the driver). Every transformer op category is a device
kernel, each gated against a CPU reference by a **tolerance** differential (`c·√K·ε`, deterministic
grids) — the CPU↔GPU analogue of the bit-exact CPU gate. Measured honestly on a power-capped 6 GB
mobile 4050 (see `BENCHMARKS.md`):

- **Tensor-core GEMM** (fp16/bf16/fp8 inputs, f32 accumulate): WMMA `m16n16k16` for fp16/bf16
  (~9–13 TFLOP/s, ~5–6× the f32 path); **fp8 (E4M3)** via hand-laid `mma.sync.m16n8k32` (no WMMA fp8 on
  `sm_89`), validated bit-exact. Its fragment-reuse multi-tile kernel (`fp8_gemm_mt_ptx`, 2×4 block of
  16×8 tiles per warp) is now the **fastest** tensor-core path — ~2.1–2.4× the naive single-tile fp8 and
  ~1.3–2.3× fp16/bf16 in the same run (single-tile retained as the fallback for non-divisible shapes).
- **Fused flash-attention** (online softmax, never materializes the `S×S` scores — the kernel that
  *loses* on CPU): warp-per-query-row, 183→372 GFLOP/s as context grows to 4 K.
- **Fused row norms** (softmax/LayerNorm/RMSNorm, one warp per row), **activations** (SFU), **reductions**
  (deterministic; max bit-exact), **conv2d**, and elementwise.
- **A whole pre-norm transformer layer runs end-to-end GPU-resident** — RMSNorm → QKV → flash-attn →
  output proj → residual → RMSNorm → FFN(SiLU) → residual, all on device buffers with no host round-trip
  between ops, matching a CPU f64 reference to max_rel 2.7e-4 and **deterministic** run-to-run.

**End-to-end `--backend=gpu`.** `mercuryc --features gpu --backend=gpu --run foo.mer` executes the
program through an **offloading interpreter**: the whole program is tree-walked on the CPU (identical
control flow, buffer layout, and every non-kernel op to the oracle), but recognized GEMM / activation
/ reduction / fused-norm calls run on the device via an `Accelerator` seam (`mercury_interp`) the
driver implements with `mercury_codegen_gpu` (`GpuAccel`). With no accelerator — every other caller,
the differential oracle — the path is byte-for-byte unchanged, so the toolchain-free core is
untouched. The CPU↔GPU boundary is tolerance-gated, so a device error surfaces as an error rather than
a silent CPU fallback. Gated by `gpu_backend_*` tests (driver, `--features gpu`): each family runs on
the interp oracle and the GPU over identical buffers and matches within tolerance (GEMM bit-exact;
silu ~5e-7, dot ~7e-7, softmax ~3e-8 abs), asserting the offload actually fired.

Run the kernel suite with `cargo test -p mercury_codegen_gpu --features gpu` (skips cleanly with no
GPU). A full MIR→PTX scalar compiler (so arbitrary, non-recognized kernels run GPU-side too) is the
remaining stretch; today unrecognized ops execute on the CPU within the same offloading run.

## Checked but not yet executed

- **Shape-typed tensors** `Tensor[f32, M, N]`: parse and pass compile-time shape checking
  (`E0501`/`E0502`), the headline feature — but tensor *operations* are not yet lowered/run.
- **Explicit SIMD vector types** `f32x8` etc. in *source*: parse and type-check; user-written vector
  values are not yet executed. (Loop auto-vectorization above is separate and *does* run.)
- **Attributes** `@simd`/`@tile`/`@align`/`@extern`/`@export`: parse and validate; consumers in
  progress. (`@parallel` now executes — see above.)

## Planned

- Fusing chains *under* `@parallel`; a **parallel** fused-epilogue GEMM kernel (the serial one already
  folds bias + ReLU/GELU/SiLU, bias optional, into the microkernel write-back — a multicore
  `mercury_sgemm_nt_epi_parallel` is the remaining step).
- A **GPU backend** (the next major frontier — where flash-attention and large-batch throughput
  actually win). Scoped in `next-steps.md` at the repo root.
- 256-bit AVX for the *general* (non-GEMM) vectorizer. Cranelift cannot legalize a 256-bit `f32x8`
  value (verified — pinned as a tripwire test), so the elementwise vectorizer is 128-bit + unrolling;
  the GEMM family already gets true AVX2/FMA via the runtime microkernel. Closing the general case
  needs a raw-AVX emitter or a future Cranelift.
- Execution of explicit `f32x8`-typed values; broader tensor-op lowering (conv, softmax) with fusion.
- Structs/enums, slices, multi-dimensional indexing `a[i, j]`, and a minimal stdlib.
- GPU device codegen (PTX/AMDGPU), autodiff — designed-for, explicitly deferred.

## Known limitations / sharp edges

- `bf16` **and** `f16` are both real 2-byte storage (round-to-nearest-even), f32 compute, with a full
  symmetric mixed-precision op suite (reductions, max-family, axpby, activations — see above) **and a
  mixed-precision GEMM**: a bf16/f16 `C = A·Bᵀ` `nn.Linear` nest dispatches to `mercury_sgemm_{bf16,
  f16}_nt[_parallel]` — a lossless widen prepass (`<<16` / F16C, ~1/n of the GEMM) feeding the tuned
  AVX2 f32 microkernel (`tests/run/linear_{bf16,f16}.mer`). On this AVX2+F16C box (no AVX-512-BF16) the
  GEMM itself runs in f32, so it is a *footprint* feature on the FLOPs — but it still beats the
  idiomatic bf16 C **~25× single-core** (that C can vectorize neither the inline `bf16→f32` widen nor
  the serial reduction), ~10× vs a hand-optimized widen-then-tile bf16 C. A half-precision *output* on
  the streaming ops (→ ~2×) needs a narrowing store and is still future work.
- The *general* vectorizer emits 128-bit SIMD (Cranelift's vector ISA rejects 256-bit `f32x8` —
  verified empirically on Cranelift 0.124). Compute-bound *elementwise* kernels therefore use 2× the
  FMA ports they could; 4× unrolling and auto-parallelism recover throughput, and the vectorized
  **transcendentals still beat scalar `libm` ~2.5–3×**. Breaking 256-bit needs a hand-written AVX2
  path (how the GEMM family already gets 256-bit — a true AVX2/FMA runtime microkernel). The loop
  vectorizer assumes distinct array parameters do not alias.
- Array *length* in a type must be an integer literal (symbolic/`const`-expression lengths fall back
  to an opaque pointer), but matmul *dimensions* may be runtime values — a runtime-dimension matmul
  still dispatches to the GEMM kernel.
- No bounds checking on array indexing (manual memory is a decided constraint).
- `mem2reg` promotes only scalar integer/float slots; arrays, pointers, and address-taken locals
  stay in memory (the interpreter and `cse`/`dse` handle those directly).
