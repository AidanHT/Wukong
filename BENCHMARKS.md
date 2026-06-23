# Mercury benchmarks — Mercury vs C vs Rust

An **honest** cross-language benchmark. For each kernel the *same* computation is written three ways
— Mercury (compiled to native code by the from-scratch backend, **no LLVM**), C (`gcc -O3
-march=native`), and Rust (`rustc -O -C target-cpu=native`) — and all three are timed through one
identical Rust harness over the same buffers. C and Rust are built to shared libraries and called via
their C ABI; Mercury is JIT-compiled in-process. The harness cross-checks a result checksum across
all three languages, so a miscompiled kernel is caught, not silently mis-measured.

Reproduce:

```sh
cargo run -p mercury_xbench --release      # CC=gcc by default; set CC to override
```

## Test machine & toolchains

- Windows 11, Intel Core Ultra 7 155H (Meteor Lake: 6 P-cores + 8 E-cores + 2 LP-E, 22 threads),
  MSYS2 toolchains. **No LLVM, no MSVC, no AVX-512** (Intel disabled AVX-512 on this consumer part);
  **AVX2/FMA + AVX-VNNI** are present (the int8 GEMM uses `vpdpbusd`, and so does gcc `-march=native`).
- `gcc` 14.2, `rustc` 1.94, Mercury via Cranelift 0.124 (JIT) + AVX2/FMA runtime microkernels.
- Elementwise/reduction kernels: f32 arrays of N = 2²⁰ (1,048,576). Matmul/linear: 256/512/1024 square.

**Variance.** This is a busy hybrid laptop; the all-core and matmul numbers swing run-to-run (P-core
boost, E-core scheduling, thermals). The harness reports the best of many batches (the least-
interfered estimate); the ranges below span several runs. Treat them as representative, not exact —
but the *ratios* (who wins, by roughly how much) are stable.

## How Mercury wins: domain-aware lowering

The headline wins come from a tensor compiler doing what a general C/C++ compiler will not do to
naively-written source:

- **Matmul dispatch.** Mercury's front-end recognizes a matmul loop nest — both the `ikj` accumulate
  form and the textbook `ijk` dot-product form, including the `C = A·Bᵀ` (`nn.Linear`) spelling — and
  lowers the *whole nest* to a tuned **register-blocked, cache-tiled, packed AVX2+FMA GEMM
  microkernel**. This is exactly how XLA/TVM/oneDNN lower a matmul op. gcc/rustc vectorize the inner
  loop but never tile, pack, or register-block, so they fall out of cache as the matrices grow. The
  recognizer also handles the **batched** form (a matmul nest under a batch loop, each index carrying
  a per-batch offset `x[h*S*D + …]`) — so **multi-head attention** dispatches one tuned GEMM per head,
  for both its `Q·Kᵀ` and `P·V` matmuls.
- **int8 quantized `nn.Linear` dispatch.** The quantized-inference GEMM — `u8` activations × `i8`
  weights → an `i32` accumulator (`C = A·Bᵀ`, the QNNPACK/oneDNN layout) — is recognized and lowered
  to an **AVX-VNNI `vpdpbusd`** microkernel, register-blocked four B-rows at a time (and the
  single-threaded path pairs two A-rows into a 2×4 tile, so each B-row load feeds both rows — halving
  B-matrix traffic). It beats gcc `-O3 -march=native` (which also emits `vpdpbusd`) **~1.5–2.5×
  single-core** — the lead widening with size — and ~4.6–14.7× with `@parallel` (clock-state-dependent;
  parallel int8 is bandwidth-bound). Integer math, so the kernel equals the scalar nest
  *bit-for-bit* (i32 add is associative mod 2³²; no reassociation exception).
- **bf16 *and* f16 mixed-precision dispatch (full suite).** Low-precision `[bf16]`/`[f16]` arrays read
  through an `as f32` widening cast (lossless: `<<16` for bf16, F16C `vcvtph2ps` for f16) with f32
  accumulate/compute — the standard ML contract — dispatch to half-precision runtime kernels across the
  whole op surface, *symmetric* for both precisions: **`dot`/`sum`** (`mercury_{dot,sum}_{bf16,f16}`),
  **`max`/`min`/`absmax`** (`mercury_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric-quant
  scale), **streaming `axpby`** (`mercury_axpby_{bf16,f16}`, half-in/f32-out), and the **36-op activation
  set** (`mercury_vmath_{bf16,f16}`). Because half precision moves **half the input bytes** of f32, the
  memory-bound ops are *bandwidth* wins that grow as the data spills cache: **~3.0–3.5× vs C** for dot,
  **~6–8×** for sum; and C/Rust can vectorize neither a `libm` call nor the half→f32 widen, so the
  activation gap is structural. Half storage is bit-exact across backends (f16 via shared `half`-crate
  shims, since Cranelift x64 lacks f16 convert lowering), and both call the identical kernel, so the gate
  stays exact.
- **Reduction vectorization + multicore dispatch.** A naive f32 reduction (`s += x[i]*y[i]`) is one
  FMA down a single dependency chain — latency-bound. Mercury reassociates it across vector lanes ×
  unrolled accumulators (the standard BLAS reduction); gcc/rustc keep it strictly serial without
  `-ffast-math`. A `@parallel` reduction goes further, lowering to a **deterministic multicore
  reduction kernel** (`mercury_sreduce_f32_parallel`: dot/ssd/sum/sumsq folded by `+`, **max/min**
  folded by `fmax`/`fmin`, and **absmax** = `fmax` over `|x|` — the per-tensor max/range/absmax softmax
  stability and dynamic int8 quantization scale-computation (`scale = absmax/127`) need) whose result
  is bit-identical to the serial form regardless of core count
  (fixed-size chunks, ascending partial combine — which holds even for the non-associative `fmax`/
  `fmin`, since both forms evaluate the identical expression tree).
- **Auto-vectorization + fusion** of elementwise loops (incl. branchy ones via if-conversion), `x +
  y*z` → FMA contraction, and adjacent-loop fusion.
- **Vectorized transcendentals → 256-bit AVX2 dispatch.** A pure `out[i] = f(x[i])` loop for
  `exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu` lowers to a tuned **256-bit AVX2/FMA runtime kernel**
  (`mercury_vmath_f32`) running a ≈1-ULP Cephes minimax poly 8 lanes at a time — the width Cranelift's
  general vectorizer cannot emit (it caps at 128-bit SSE). gcc/rustc call scalar `libm`
  `expf`/`logf`/`tanhf` and cannot vectorize a loop containing a call, so the activation family
  (GELU/SiLU/tanh/sigmoid + softmax/log-softmax/cross-entropy) runs **~5–7.5× faster**, and an
  `@parallel` activation dispatches each thread's chunk to the kernel (multicore × 256-bit).
- **Fused row normalizations → single-pass kernel.** The per-token normalizations every transformer
  layer runs — **softmax**, **LayerNorm**, **RMSNorm** — are recognized from their canonical
  multi-pass source and folded into one **`mercury_norm_f32`** call: each row loaded once, 256-bit
  AVX2, the hand-vectorized `exp` for softmax, and the mean / variance / sum-of-squares reductions
  reassociated across 8 lanes. gcc/rustc auto-vectorize the elementwise passes but (without
  `-ffast-math`) keep the float reductions sequential and call scalar `expf` — so this *composes* the
  reduction-vectorization and transcendental wins into the fused op (**~1.9–6.6× faster than C**). The
  **affine** LayerNorm/RMSNorm real models run (a learned per-channel scale γ and shift β) dispatch to
  a sibling **`mercury_norm_affine_f32`** and hold the same win (**~1.7–3.7×**) — γ/β fold into the
  writeback for free.
- **Convolution via im2col + GEMM.** A conv expressed as im2col + matmul has its matmul recognized
  and dispatched to the GEMM microkernel — so Mercury beats hand-written direct convolution ~5.8×,
  the same way XLA/cuDNN lower conv.
- **Auto-parallelization** of `@parallel` loops across all cores, with each per-thread chunk itself
  vectorized.

## Fairness notes

- **FMA:** Mercury contracts `x + y*z` to a fused multiply-add, so gcc is given its **default**
  `-ffp-contract=fast` (both fuse). Idiomatic Rust does not contract unless the author writes
  `f32::mul_add`, so the Rust column reflects rustc's default — a real toolchain-defaults difference,
  surfaced rather than papered over. Mercury and its interpreter oracle agree bit-for-bit (gated).
- **Matmul dispatch is the value proposition, stated plainly.** The C/Rust columns are the *naive
  nest a programmer writes*; Mercury's compiler optimizes it the way a tensor compiler should. The
  win **grows with size** precisely because tiling/packing matters more as the data stops fitting in
  cache — a single size could be a fluke, so a sweep is shown.
- **`nn.Linear` (`C = A·Bᵀ`)** is written the idiomatic way in all three languages: the `ijk`
  dot-product form (`for i,j { s=0; for k s+=a[i,k]*b[j,k]; c=s }`), where A and B are both read
  contiguously. gcc/rustc leave that f32 reduction strictly serial (latency-bound, ~1.5 GFLOP/s),
  while Mercury dispatches to its GEMM. The large ratio is real and is *caused by C's serial
  reduction*; it is not a strided-access strawman.
- **Correctness:** the native backend and the interpreter run the *identical* GEMM kernel (the
  interpreter marshals its memory through the same routine), so the differential oracle stays
  bit-for-bit exact even though the kernel reassociates.

## Results

### Compile time — Mercury wins by 2 orders of magnitude

| | Mercury | C (gcc) | Rust | Mercury speedup |
|---|---|---|---|---|
| any kernel | ~1–2 ms | ~125–245 ms | ~185–210 ms | **~100–260×** (geomean ~135–155×) |

Cranelift JIT compiling in-process vs spawning a full C/Rust+LLVM toolchain is a 1–2 order-of-
magnitude win, every build. For an ML compiler — where edit/recompile/run iteration dominates
developer time — this is the most robust result of all.

### Matmul `C = A·B` — single-core wins, parallel dominates, and the lead grows with size

GFLOP/s (higher is better), naive `ikj` nest in each language:

| size | Mer 1-core | Mer @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 256³ | 80–100 | 86–99 † | 21–41 | 21–42 | **~2.8–4.3×** | **~3.4–4.1×** |
| 512³ | 110–126 | 328–462 | 30–41 | 38–45 | **~3.0–3.1×** | **~9×** |
| 1024³| 102–119 | 437–522 | 22–30 | 25–32 | **~3.4–3.6×** | **~14–18×** |

The single-core kernel now holds **~110–120 GFLOP/s** at 512³–1024³ — ≈90% of one P-core's AVX2-FMA
peak (pinned to a P-core it reaches a stable ~117–126) — while gcc's naive nest falls from ~40 to ~22
GFLOP/s as 1024² spills out of cache, so the **single-thread lead widens with size**. The parallel
kernel reaches **~437–522 GFLOP/s at 1024³** and **~690 at 2048³** (after the `MC=144` cache-block
widening cut the B-panel's L3 re-streaming, and the pack-scratch is reused across blocks rather than
re-allocated per K-block).

† At 256³ the parallel kernel deliberately falls back to the serial one: ~17M MACs is below the
work threshold where cross-core wake/sync pays off on this P+E hybrid, so "@parallel" ≈ single-core
there (a measured fix — naive threading at that size was a net *loss*).

### `nn.Linear` `C = A·Bᵀ` — Mercury dispatches to GEMM; naive C is latency-bound

| size | Mer 1-core | Mer @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 512² | 81–112 | 251–348 | ~4.0–4.8 | ~3.4–4.5 | **~19–25×** | **~52–76×** |
| 1024²| 88–108 | 258–457 | ~3.5–4.4 | ~3.4–4.3 | **~22–26×** | **~67–104×** |

C/Rust leave the idiomatic `ijk` dot-product reduction strictly serial (~4–5 GFLOP/s, latency-bound),
while Mercury recognizes `C = A·Bᵀ` and dispatches to the same packed GEMM — hence the order-of-
magnitude gap (caused by C's serial reduction, not a strided-access strawman; see Fairness notes).

### Convolution — im2col + GEMM vs idiomatic direct conv

A 3×3 conv (Cin=16, 20×20 input → 64 filters → 18×18 output), the way XLA/cuDNN lower it: an im2col
gather builds the `[Cin·K·K, OH·OW]` column matrix, then the conv is a matmul `Y = W · col` that the
recognizer dispatches to the tuned GEMM. C and Rust run the idiomatic **six-deep direct-convolution
nest** (the loop everyone writes by hand).

| kernel | Mer (im2col+GEMM) | C (direct) | Rust (direct) | Mercury vs C |
|--------|-------------------|------------|---------------|--------------|
| conv2d 3×3 | ~30–40 GFLOP/s | ~3.6–6.6 | ~3.7–6.5 | **~6–7× faster** |

Same result (checksum cross-checked). The conv's GEMM is small (M=64, K=144, N=324) so it runs below
the large-matmul peak, but it still beats hand-written direct convolution ~6–7× — the im2col gather
is cheap data movement and the GEMM microkernel does the FLOPs. So Mercury accelerates conv *for
free* through the existing matmul dispatch (`tests/run/conv_im2col.mer`).

### Transcendentals / activations — Mercury dispatches to a 256-bit AVX2 kernel; C calls scalar `libm`

The activation family every transformer runs, and **the cleanest compute-bound win in the suite**.
Mercury recognizes a pure `out[i] = f(x[i])` loop for **35** functions —
`exp`/`log`/`exp2`/`log2`/`exp10`/`log10`/`cbrt`/`expm1`/`log1p`/`tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`softsign`/`logsigmoid`/`mish`/`selu`/`tanhshrink`/`hardsigmoid`/`hardswish`/`sin`/`cos`/`tan`/`atan`/`asin`/`acos`/`erf`
plus the hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh` — and lowers the whole
loop to a **256-bit AVX2/FMA runtime kernel** (`mercury_vmath_f32`) — the same domain-aware dispatch
as matmul→GEMM. The kernel runs a ≈1-ULP Cephes minimax polynomial 8 lanes at a time; gcc/rustc call
scalar `libm` `expf`/`logf`/`tanhf`/`sinf`/`cosf`/`erff` and **cannot vectorize a loop containing a call** (no `libmvec` on
this mingw toolchain), so they stay serial. `silu` (Llama/SwiGLU) and `gelu` (BERT/GPT-2/ViT, tanh
approximation) are first-class intrinsics, as are `elu`, `leaky_relu`, `softplus` (= `ln(1+eˣ)`), and
`mish` (= `x·tanh(softplus)`) — all composing the shared ≈1-ULP `exp`/`log`. **`sin`/`cos`** (the
rotary-position-embedding transcendentals in every modern LLM) and **`erf`** (the original BERT/GPT-2
GELU's core) dispatch to the 256-bit kernel too, as does the full **hyperbolic family** —
`sinh`/`cosh` plus the inverse `asinh`/`acosh`/`atanh` (`atanh` is the Fisher z-transform; the inverse
trio powers hyperbolic/Poincaré embeddings and normalizing flows), each composing the shared `log`/`√`.
The inverse trig **`tan`/`asin`/`acos`** completes the angle/geometry set (3D vision, graphics ML,
NeRF/SLAM poses) by composing `sin`/`cos`/`atan`; **`exp10`/`log10`** add base-10 (decibel/log-scale
features); **`softsign`** (`x/(1+|x|)`, a cheaper bounded activation than `tanh`) and **`logsigmoid`**
(`ln σ(x)`, the stable BCE-with-logits / contrastive-loss primitive) round out the activation set.
The interpreter marshals through the *identical* kernel, so the differential oracle stays exact.

This is the change that took the activations from a ~128-bit ~2.5–3.5× win to the ~5–7.5× range —
**roughly double**, because they are compute-bound (~20 flops/element) and the missing 256 bits were
the ceiling. (A *composed* `exp`/`erf`/`sin`/`cos` — one inside a larger arithmetic expression rather
than a bare `out[i]=f(x[i])` loop — still lowers to the inlined ≈1-ULP poly and auto-vectorizes at
128-bit; the dispatched 256-bit kernel mirrors that poly op-for-op, so the two agree.)

| kernel | Mercury vs C | notes |
|--------|--------------|-------|
| `exp`  | **~5.9–6.8× faster** | `out=exp(x)`; 256-bit AVX2 poly vs scalar `expf` |
| `log`  | **~4.6–7.5× faster** | `out=log(x)`; 256-bit Cephes poly vs scalar `logf` (libm `logf` timing varies run-to-run) |
| `tanh` | **~4.5–6.1× faster** | exp-based, identical algorithm everywhere; only Mercury vectorizes (256-bit) |
| `gelu` | **~5.0–6.5× faster** | tanh-GELU intrinsic → fused 256-bit kernel; C/Rust the same math, scalar |
| `silu` (swish) | **~3.6–5.8× faster** | `silu()` intrinsic (`x·sigmoid(x)`) → fused 256-bit kernel; C/Rust scalar |
| `softplus` | **~6.6× faster** | `ln(1+eˣ)` (exp+log) → fused 256-bit kernel; also ~4× vs Rust |
| `mish` | **~5.7× faster** | `x·tanh(softplus(x))`, three transcendentals — the heaviest, widest gap; ~6× vs Rust |
| `sin` | **~6.0–8.6× faster** | RoPE; 256-bit Cephes `sinf` poly + quadrant reduction vs scalar `sinf` (heavier than `expf`, so the widest single-call gap) |
| `cos` | **~6.0–8.2× faster** | RoPE; the cos branch of the same reduced-argument poly |
| `erf` | **~3.9–4.9× faster** | exact (erf-based) GELU; 256-bit Abramowitz–Stegun poly vs scalar `erff` |
| `asinh` | **~10–11.5× faster** | `sign(x)·log(\|x\|+√(x²+1))` (sign-stable) vs scalar `asinhf` — libm's `asinhf` carries its own reduction over a log, so the widest gap in the suite |
| `acosh` | **~7.6× faster** | `log(x+√(x²−1))`, x≥1, vs scalar `acoshf` |
| `atan` | **~8.9–9.5× faster** | Cephes 3-region reduction + degree-3 poly vs scalar `atanf` (its branchy reduction is heavy per element) |
| `log1p` | **~6.4× faster** | stable `ln(1+x)` (Kahan, one `log` + a divide) vs scalar `log1pf` |
| `expm1` | **~2.0× faster** | stable `eˣ−1` (Kahan) vs scalar `expm1f` — the family's most *modest* win, since the stable form computes **both** `exp` and `log` per element |
| `gelu@parallel` | **~28× faster** | GELU over a large tensor across cores: multicore × 256-bit vs single-thread scalar C |

The full elementwise math suite — `sqrt`/`rsqrt`/`cbrt` (root family), `exp`/`log`/`exp2`/`log2`/`exp10`/`log10` (≈1-ULP minimax polys),
`pow` (= `exp(y·log(x))`), `atan2`/`hypot` (two-arg geometry), `tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`softsign`/`logsigmoid`/`mish`/`selu`/`tanhshrink`/`hardsigmoid`/`hardswish`,
the trig `sin`/`cos`/`tan`/`atan`/`asin`/`acos`, the hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh`, and `fmax`/`fmin` — all vectorize. Every kernel passes the cross-language checksum (the ≈1-ULP poly agrees
with `libm` within tolerance) and
compiles ~100–490× faster. These are compute-bound, so the win is real SIMD throughput, not
bandwidth. An `@parallel` activation dispatches *each thread's chunk* to the kernel, so it runs
multicore × 256-bit. `softmax`/`LayerNorm`/`RMSNorm` are recognized and dispatched to a fused
single-pass kernel (`mercury_norm_f32` — see the section above), and **log-softmax / cross-entropy**
(`exp` + `log`) run as fused vectorized chains (`tests/run/`); a transformer FFN block (two `nn.Linear` matmuls + GELU)
composes the GEMM and activation wins in one function (`tests/run/ffn_block.mer`). All paths are
bit-identical across backends — the dispatched kernel by marshalling, the inlined polys by
construction.

### Fused row normalizations — softmax / LayerNorm / RMSNorm dispatch to one kernel

The per-token normalizations every transformer layer runs. Mercury recognizes the canonical
multi-pass source (softmax's max/exp/sum/normalize; LayerNorm's mean/variance/normalize; RMSNorm's
mean-square/normalize) and folds the whole thing into one **`mercury_norm_f32`** call — or, when the
normalize step carries the learned per-channel scale γ and shift β that **real** transformer
LayerNorm/RMSNorm apply (`(x-μ)·inv·γ + β`), into one **`mercury_norm_affine_f32`** call with γ/β
folded into the same single-pass writeback. Either way: 256-bit AVX2, the hand-vectorized `exp` for
softmax, and the reductions reassociated across 8 lanes. gcc and
rustc at their honest defaults (no `-ffast-math`) auto-vectorize the elementwise passes but keep the
float reductions strictly sequential, and call scalar `libm` `expf` for softmax. All three languages
copy `x`→`out` then normalize in place — identical work, so the copy pass is charged to everyone and
these figures *understate* the per-pass advantage — and the full output buffer is cross-checked
element-by-element (a tight tolerance, since Mercury reassociates the reductions and C does not).

Measured as the **Mercury-vs-C runtime ratio** over feature rows of 768 and 4096 f32. Absolute
ns/call swings ~2× run-to-run with the laptop's clock/thermal state (the whole machine speeds up or
slows down together), so — as with the GEMM — the honest figure is the *ratio*, reported as a range
across runs:

| op | Mercury vs C | where the win comes from |
|----|--------------|--------------------------|
| softmax   | **~4.4–6.6× faster** | vectorized `exp` (gcc's scalar `expf` can't vectorize a loop with a call) + the reassociated sum |
| LayerNorm | **~3.0–3.5× faster** | two reassociated reductions — the mean, then the variance |
| RMSNorm   | **~1.9–2.5× faster** | one reduction (mean-square); the copy/scale elementwise passes, which gcc vectorizes too, dilute it |
| LayerNorm (affine γ, β) | **~2.8–3.7× faster** | the real transformer form `(x-μ)·inv·γ + β` → `mercury_norm_affine_f32`; same reduction win, γ/β fused into the writeback |
| RMSNorm (affine γ) | **~1.7–3.2× faster** | the real transformer form `x·inv·γ` → same affine kernel |

Softmax wins most — the vectorized `exp` dominates, the same effect as the standalone `exp` kernel.
LayerNorm and RMSNorm win on their reassociated reductions (gcc keeps float reductions strictly
sequential without `-ffast-math`), with RMSNorm lowest because it has only one reduction and a larger
share of plain elementwise work. Rust tracks C within a few percent throughout. One row is one
token's hidden vector; the whole `[tokens, hidden]` batch is recognized too (see *Batched RMSNorm*
below). **The affine
variants hold the same win** (~2.8–3.7× LayerNorm, ~1.7–3.2× RMSNorm) — the learned γ/β are a cheap
per-element multiply-add that rides along in the writeback, so the fusion + reduction-vectorization
advantage is unchanged; this is what makes the win apply to the norms real models actually run, not
just the γ=1 idealization. The interpreter marshals the identical kernel so the differential oracle
stays bit-for-bit exact, and the recognizer runs pre-opt so `-O0` == `-O3`.

### Batched norms — the whole `[tokens, hidden]` matrix in one call, and across cores

A single feature row is one token; a transformer normalizes a whole `[tokens, hidden]` activation.
Mercury recognizes the **batched** form — `for r in 0..R { <norm over x[r*C + i]> }`, the row-major
`[R, C]` matrix — for all three norms and folds the entire batch into one `mercury_norm_f32(x, x, R, C,
…)` call (each row a fused single pass), instead of leaving the outer loop to scalar/vectorized code. In
a `@parallel` function the same batch dispatches to **`mercury_norm_f32_parallel`**, mapping the
independent rows across cores. Measured as the **Mercury-vs-C runtime ratio**, serial *and* `@parallel`,
at an L3-resident batch and a batch that spills L3:

| op | 512×768 (1.5 MB, in L3) serial · @parallel | 4096×4096 (64 MB, ≫ L3) serial · @parallel |
|----|-----|-----|
| RMSNorm   | **~2.0×** · ~1.1× | **~2.0×** · **~2.9×** |
| LayerNorm | **~2.5×** · **~1.9×** | **~2.1×** · **~4.0×** |
| softmax   | **~5.6×** · **~6.1×** | **~4.7×** · **~10.2×** |

The **serial** fused single-pass form is an unconditional win for every norm (~2.0–5.6× vs
single-threaded C, fused-vs-multipass; softmax most, on its vectorized `exp` vs scalar `expf`). Whether
`@parallel` *adds* to that depends on where the kernel's bottleneck is:

- **RMSNorm is purely memory-bound** (read twice, write once), so multicore only helps once the batch
  spills L3 (~2.9× at 64 MB); L3-resident it's a wash (~1.1×) — the same working-set rule as the
  non-temporal streaming dispatch.
- **softmax is compute-bound** (the vectorized `exp` per element), so `@parallel` pays off *even
  L3-resident* (~6.1×) and reaches **~10.2×** at scale.
- **LayerNorm sits between** (two reductions + the center/scale): ~1.9× in L3, ~4.0× past it.

Both paths marshal the identical per-row kernel through the interpreter, and the multicore kernel has no
cross-row combine, so the differential oracle stays bit-for-bit exact (the runtime's
`serial_matches_parallel_bit_for_bit` pins the kernel equality; the codegen's
`differential_parallel_batched_norm` pins the end-to-end dispatch; `tests/run/batched_*.mer` and the
fuzzer's `{rmsnorm,layernorm,softmax}_batched` kernels cover all three). The **affine** forms (a learned
per-column γ/β) batch the same way — `mercury_norm_affine_f32[_parallel]`, the real transformer norm.
C/Rust here are the idiomatic single-threaded per-row nested loops at honest defaults.

### int8 quantized `nn.Linear` — `vpdpbusd` register-blocked, beats gcc single-core

Quantized inference runs `nn.Linear` as `C = A·Bᵀ` with **`u8` activations × `i8` weights → an `i32`
accumulator** (the QNNPACK / XNNPACK / oneDNN layout). Mercury recognizes the `ijk` dot-product nest
— `s += (a[..] as i32) * (b[..] as i32)`, A `u8`, B `i8` — and dispatches it to the
**`mercury_i8gemm_nt`** microkernel: AVX-VNNI **`vpdpbusd`** (one instruction folds 32 `u8×i8`
products into the `i32` lanes *and* accumulates), register-blocked four B-rows at a time (the A-row
chunk loaded once and reused across the four dots, four accumulator chains for ILP); the
single-threaded path additionally pairs **two A-rows into a 2×4 register tile**, so each B-row load
feeds both rows and B-matrix traffic is halved (the lift that takes single-core from ~1.3× to
~1.5–2.5×). C and Rust run the idiomatic naive int8 GEMM at `-O3 -march=native` / `-O
-Ctarget-cpu=native` — gcc auto-vectorizes it to `vpdpbusd` as well, so this is an honest
*same-instruction* comparison, not a straw man.

**Because the math is integer, the cross-language check is bit-exact** (not a tolerance): `i32` add is
associative mod 2³² and `vpdpbusd` is non-saturating, so Mercury, C, and Rust must agree on every one
of the `M·N` outputs — and they do.

Reported as int8 **GOP/s** (2 ops per multiply-accumulate) and the clock-invariant Mercury-vs-C ratio
across runs:

| size | Mercury 1-core vs C | Mercury `@parallel` vs C | Rust |
|------|---------------------|--------------------------|------|
| 512×512   | **~1.5–1.7× faster** | **~4.6–8.5× faster** | ~5–10× slower than Mercury 1-core |
| 1024×1024 | **~2.2–2.5× faster** | **~8.0–14.7× faster** | ~8–12× slower than Mercury 1-core |

**Absolute throughput swings ~2–3× with the laptop's clock/thermal state, so the clock-invariant
ratio is the reported quantity** (as with the f32 GEMM's roofline %): a high-clock run measured
Mercury ~290–347 GOP/s single-core against gcc ~133–226, a cooler run roughly half of each — but the
*ratio* held. **The single-core lead widens with size** — the 1024² ratio (~2.2–2.5×) exceeded the
512² ratio (~1.5–1.7×) in every one of the four post-2×4 runs, exactly as with the f32 GEMM, because
the 2×4 tile's halved B-traffic pays off more once B spills out of L2. The `@parallel` form maps
independent rows across cores (rows are independent, so serial == parallel bit-for-bit); its lead over
single-threaded C (~4.6–14.7×) is more clock-sensitive than the single-core figure because parallel
int8 saturates memory bandwidth (which barely scales with clock) while the lone C core is
compute-bound (which does). Rust trails badly: rustc does not auto-vectorize the `u8×i8` widening dot,
so it runs essentially scalar. The interpreter marshals the identical kernel, so the differential
oracle stays bit-for-bit exact, and the recognizer runs pre-opt so `-O0` == `-O3`.

### bf16 / f16 mixed-precision — half storage, f32 accumulate

The ML mixed-precision contract: store activations in `bf16` (half the bytes), accumulate the
reduction in `f32` (full precision). Mercury recognizes a `s += (x[k] as f32) [* (y[k] as f32)]`
loop over `[bf16; _]` arrays and dispatches it to **`mercury_dot_bf16`** / **`mercury_sum_bf16`** —
widen bf16→f32 (a `<<16` bit-extend, F16C-class) and accumulate across 8 f32 lanes. C and Rust run
the idiomatic `<<16` widen + accumulate at honest default flags: gcc/rustc may vectorize the *widen*,
but without `-ffast-math` they keep the f32 reduction **sequential** — the same basis as the f32 `dot`
kernel. All-positive inputs keep the reduction well-conditioned, so the cross-language scalar result
agrees within a tight tolerance (the three reassociate the f32 sum differently).

GB/s (higher is better) — input traffic over `[bf16; N]` arrays:

| op | N=2²⁰ Mer · C · Rust | N=2²⁴ Mer · C · Rust | Mercury vs C |
|----|----------------------|----------------------|--------------|
| dot Σx·y | **10.2** · 3.4 · 3.2 | **10.2** · 2.9 · 3.0 | **~3.0× → 3.5×** |
| sum Σx   | **12.3** · 1.5 · 1.7 | **10.5** · 1.7 · 1.7 | **~8.3× → 6.1×** |

The **dot** win (~3×) tracks the f32 `dot` — Mercury vectorizes the reduction while C/Rust stay serial.
The **sum** win is larger (~6–8×) because C's unary f32 sum is a single dependency chain (pure
latency, no product to fill the pipeline) at ~1.5 GB/s, while Mercury's 8-lane SIMD sum reaches
~12 GB/s. The lead **widens from 2²⁰ to 2²⁴** as the working set spills L3 and the halved byte count
(bf16 vs f32) starts to dominate — the bandwidth payoff of mixed precision. Correctness: bf16 storage
is bit-exact across the interpreter and native backends (`round_to_bf16` emits the identical integer
arithmetic as the interpreter's `round_bf16`), and both call the identical reduction kernel, so the
differential gate stays exact for *fractional, non-bf16-exact* inputs across `-O0`/`-O2`/`-O3`
(`differential_bf16_reduce`); `tests/run/reduce_bf16.mer` pins the e2e value.

The same dispatch is now **symmetric for `f16`** (widened with the F16C `vcvtph2ps` instruction
instead of the bf16 `<<16`; f16 storage is bit-exact via shared `half`-crate shims, since Cranelift
x64 has no f16 convert lowering) and **extended across the op surface** for both precisions:
**`max`/`min`/`absmax`** (`mercury_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric int8
quant scale; exact, since max/min round nothing), **streaming `axpby`** (`out = a·x + b·y`,
half-in/f32-out, ~1.3× ≫ L3), and the **36-op activation set** (`mercury_vmath_{bf16,f16}`, where the
cheap ops gain bandwidth and the transcendentals keep the full libm-vectorization win — C can vectorize
neither the `libm` call nor the half→f32 widen). e2e: `tests/run/{f16,reduce_f16,reduce_bf16_minmax,
vmath_{bf16,f16},axpby_f16}.mer`; all bit-exact interp == native.

### Single-threaded elementwise & reductions

A recognized streaming map (`out[i] = act(a·x[i] (+ b·y[i]) + c)`) dispatches to the **256-bit AVX2
`mercury_velem_f32`** kernel and a Horner polynomial to **`mercury_vhorner_f32`** — both 4×-unrolled,
both emitting **non-temporal stores** once the working set spills L3 (the store path gcc/rustc will
not emit, skipping read-for-ownership traffic). This turns the former bandwidth-bound *ties* into
wins:

| kernel | Mercury vs C | notes |
|--------|--------------|-------|
| saxpy  | **~1.25–1.45× faster** | `velem` 256-bit + non-temporal store (3-stream, spills L3) — Rust ~1.4× behind too |
| relu   | ≈tie (~1.0×) | 2-stream, L3-resident at N=2²⁰ so stores stay cacheable; both are bandwidth-bound (an [identity-affine fast path](crates/mercury_runtime/src/velem.rs) drops the wasted `fma(1·x+0)` so it no longer trails gcc); the win shows at >L3 (~1.4×, below) |
| poly   | **~1.1–1.2× faster** | `vhorner` AVX2 Horner, **six** independent chains (was four — a 5-deep dependent FMA chain needs ~8 in flight to fill both ports); now a consistent win over gcc's own 256-bit autovec where four chains only tied |
| fused linear→relu | ≈tie (±10%, clock-dependent) | matvec-bound (M=1) at the bandwidth wall; Mercury fuses the two source loops |
| dot    | **~2.9× faster** | reduction reassociated to lane accumulators; gcc/rustc stay serial |
| ssd (Σ(x−y)²) | **~2.6–2.9× faster** | same — an L2-loss reduction |

**At real (>L3) activation-tensor sizes the lead widens** — non-temporal stores avoid the RFO traffic
that dominates when nothing fits in cache. At N=2²⁴ (64 MiB/array):

| kernel (N=2²⁴) | Mercury vs C | Mercury vs Rust |
|--------|--------------|--------|
| saxpy  | **~1.4× faster** | ~1.4× |
| residual (x+y) | **~1.3–1.4× faster** | ~1.4× |
| scale (a·x) | **~1.3× faster** | ~1.4× |
| relu | **~1.5–1.6× faster** | ~1.7× |

A `@parallel` reduction goes further: it dispatches to a deterministic multicore reduction kernel
(`mercury_sreduce_f32_parallel`), spreading the stream across cores to reach *aggregate* bandwidth —
see the `dot@parallel`/`ssd@parallel`/`max@parallel` rows below. The same kernel folds **max/min** (by
`fmax`/`fmin`), so a tensor's `[min, max]` range (asymmetric int8 quantization → `scale=(max−min)/255`)
or per-tensor max (softmax stability) computes across cores too.

### Auto-parallel runtime — Mercury heavily exceeds idiomatic single-threaded C/Rust

`@parallel` lowers the loop to a multicore dispatch whose per-thread chunk is itself vectorized (or,
for a reduction, dispatched to the multicore reduction kernel). These kernels are memory-bound, so the
parallel speedup is limited by *aggregate* bandwidth, not core count — still a clear win over
single-threaded C:

| kernel | Mercury vs single-threaded C |
|--------|------------------------------|
| saxpy@parallel | **~2.4–2.7×** (~132 GB/s, near the chip's memory-bandwidth ceiling) |
| poly@parallel  | **~2.2–2.4×** |
| relu6@parallel | **~6.7–8.0×** (nested branch defeats gcc's vectorizer; Mercury if-converts + parallelizes) |
| dot@parallel   | **~7.9×** (~135 GB/s) — reduction across cores; C/Rust keep it serial & latency-bound |
| ssd@parallel   | **~8.6×** (~144 GB/s) — L2-loss reduction across cores |
| max@parallel   | **~25–26×** (~85 GB/s) — per-tensor max (int8-quant range / softmax stability) across cores; C's single-stream float-max chain is especially latency-bound (~3 GB/s) without `-ffast-math` |
| absmax@parallel | **~25×** (~80 GB/s) — per-tensor max\|x\| (symmetric int8-quant scale) across cores; `abs` is free (a bitwise op) so C stays latency-bound like `max` (~3 GB/s) |

## GPU backend (NVIDIA RTX 4050 Laptop, `sm_89`)

Mercury has a **GPU backend** (`mercury_codegen_gpu`, behind `--features gpu`). Being a compiler, it
**emits PTX text** and **driver-JIT-loads it via `cudarc`** (`cuModuleLoadData` — the NVIDIA driver's
built-in PTX→SASS JIT, so **no `nvcc`/`ptxas`/CUDA toolkit** is needed to build or run, only the
driver). Every transformer op category is available as a device kernel, each tolerance-gated against
a CPU reference. The CPU↔GPU gate is a **tolerance** differential (`c·√K·ε`, deterministic grids),
not bit-exactness, because the GPU reassociates and rounds (SFU transcendentals) differently — but it
is checked over the full output, and the GPU reductions/norms are deterministic run-to-run (fixed grid
+ warp-butterfly all-reduce). Measured on a **mobile RTX 4050** (Ada, 6 GB, **power-capped ~30–50 W** —
far below a desktop/datacenter part), so the absolute TFLOP/s are honest for *this* GPU, not a 4090/H100.

**Tensor-core GEMM (fp16/bf16/fp8 inputs, f32 accumulate).** fp16/bf16 use WMMA `m16n16k16`
(fragment-reuse multi-tile); fp8 (E4M3) has no WMMA on `sm_89`, so it is the warp-level
`mma.sync.m16n8k32` with the fragments placed by hand — both a naive single-tile form and a
fragment-reuse multi-tile (`_mt`) form. Absolute throughput **swings ~7× with the laptop's
power/thermal state**, so the table below is *one representative back-to-back run* (every column
measured in the same clock state — only the cross-column ratios within a run are meaningful, never the
absolute GFLOP/s across runs):

| size | f32 reg-blocked | fp16 TC | bf16 TC | fp8 single | **fp8 multi-tile** |
|------|-----------------|---------|---------|------------|--------------------|
| 512³  | 1152 | 7542 | 7621 | 4622 | **9881** |
| 1024³ | 1216 | 7220 | 7875 | 4758 | **10045** |
| 2048³ | 1431 | 8217 | 10564 | 5527 | **13474** |
| 4096³ | —¹ | 5991 | 5410 | 5815 | **13739** |

¹ the 4096³ f32 run hit a clock dip (226 GFLOP/s); omitted to avoid a misleading ratio.

The Ada tensor cores hit **~9–13 TFLOP/s fp16/bf16** in warmer runs — **~5–6× the f32 register-blocked
path** on the same GPU — with f32 accumulation (the mixed-precision contract). The f32-accumulate
tolerance gates pass (fp16 ~2e-3 rel, bf16 ~1e-2 rel, fp8 vs e4m3-rounded inputs ~2e-3 rel, isolating
accumulation error from input rounding). **The fragment-reuse multi-tile fp8 path is now the fastest
tensor-core kernel** (`fp8_gemm_mt_ptx`: each warp computes a 2×4 block of 16×8 tiles, loading each A
fragment once and reusing it across 4 B-tiles and each B fragment across both A-tiles). In the run
above it is **2.1–2.4× the naive single-tile fp8** and **1.3–2.3× fp16/bf16** (the lead widens with
size as the kernel becomes compute- rather than memory-bound) — finally realizing Ada's ~2× fp8
tensor-core rate. The naive single-tile form (one 16×8 tile per warp, re-reading A/B from global each
K-step) is retained as the fallback for shapes the multi-tile block doesn't divide (`M%32≠0` or
`N%32≠0`); it is memory-bound and sits below fp16, as expected. The hard part — the manual `mma.sync`
fp8 fragment layout on `sm_89` — is bit-exact (`max_abs=0` vs an asymmetric e4m3-exact reference), and
the multi-tile correctness gate covers it at 64³ and 128×256×64. SMEM K/V staging for fp8's true peak
remains documented future work.

**Honest peer scoreboard — vs cuBLAS and naive CUDA-C.** A GPU kernel's only meaningful rivals run on
the *same GPU*. Both are now measured here (Phase 0 of the GPU plan): the redistributable **NVRTC** +
**cuBLAS** DLLs `dlopen` like the driver itself, so `cargo test` stays toolkit-free and the peer bench
*skips* (never fails) when they are absent. Two bars:

- **Tier A — naive CUDA-C** compiled at runtime by NVRTC (the idiomatic one-thread-per-output GEMM a
  programmer writes by hand). Beating it is the literal "beat C/C++/Rust **on the GPU**" — the GPU twin
  of Mercury beating scalar CPU-C, via tiling and tensor cores the author never wrote.
- **Tier B — cuBLAS fp16** (`cublasGemmEx`, f32 accumulate): NVIDIA's hand-tuned closed-source gold
  standard. Mercury is reported as a **% of cuBLAS**.

Same-run, same device buffers, checksum-cross-checked, both peers first tolerance-gated against the f64
oracle (a fast-but-wrong kernel never scores). Two back-to-back runs (the mid-size % swings with the
laptop's clock state; the 4096³ cliff does not):

| size | dispatched kernel | % of cuBLAS | × vs naive CUDA-C | vs prior `_sm` default |
|------|-------------------|-------------|-------------------|------------------------|
| 1024³ | `_sm_db` (64-tile + cp.async) | **~101%** | ~98× | **1.16×** |
| 2048³ | `_sm128_db` (128-tile + cp.async) | ~74% | ~71× | **1.15×** |
| 4096³ | `_sm128_db` (128-tile + cp.async) | ~34% | ~47× | ~1.07× |

The honest standing: at L2-resident sizes Mercury's fp16 tensor-core GEMM now **reaches cuBLAS parity
(~101% at 1024³)** — the plan's M1 isolated target (≥95%) is met there — on top of a **wide Tier-A win**
(tens-to-100×+ over the naive hand-written CUDA-C kernel, the same way it beats naive CPU-C). The large
4096³ case is improved but **still short of cuBLAS** (~34%); closing it is ongoing.

**Phase-1 progress — `cp.async` software pipelining.** The shared-memory-staged kernel's `{load-all;
sync; compute-all; sync}` K-loop stalls on global-load latency once the working set spills L2. Two
levers were measured same-run against cuBLAS:
- A **bigger 128×128 CTA tile** (halves redundant inter-CTA global traffic) was **~neutral alone**
  (33.9%→34.6% @4096³) — proof by elimination that the large-GEMM cliff is **latency-, not
  bandwidth-volume-bound**.
- **`cp.async` double-buffering** (prefetch the next A/B tile into the alternate shared buffer while the
  tensor cores consume the current one) is a **large win where the problem is L2-resident** — the 64×64
  pipelined kernel `wmma_nt_f16_sm_db` hits **~101% of cuBLAS at 1024³ (1.16× over the prior `_sm`)** —
  but *regresses* large (eager loads saturate DRAM). Combined with the 128×128 tile,
  `wmma_nt_f16_sm128_db` (the cuBLAS recipe: big tile cuts traffic, pipeline hides the rest) is the
  best large-GEMM path (**1.15× @2048³, ~1.07× @4096³ over `_sm`**).

`gemm_nt_f16` now **dispatches by regime**: the 64-tile pipeline ≤1024², the 128-tile pipeline ≥2048²,
the plain staged 64-tile otherwise — each the measured winner in its range. Remaining levers toward
≥95% at 4096³: deeper (3+-stage) pipelines, `ldmatrix`, swizzled SMEM, warp-tiling, split-K. Reproduce:
`gemm_vs_peers` in `mercury_codegen_gpu` with the CUDA 12.9 redist DLLs on PATH (one-line setup in
`baselines.rs`).

**Beating cuBLAS by fusion (M1-fused).** cuBLAS can only compute `A·Bᵀ`; an activation needs a *second*
kernel that reads C back from HBM, applies the op, and writes it again. Mercury fuses the activation
into the WMMA store epilogue (elementwise on the f32 accumulators, before the C store — `Act` in
`ptx_wmma.rs`), so `act(A·Bᵀ)` is **one kernel that writes C once**. The fused activations are **relu,
silu, and gelu** — silu fuses the SwiGLU FFN up-projection `silu(x·W1ᵀ)` into a single kernel, and the
transcendental ones reuse the exact SFU formulas of the standalone `vmath` kernels so the fused result
equals the unfused one. Measured on resident device buffers (`fused_gemm_activation_vs_chain`, fused
output gated == `act(cuBLAS)`; the chain cost is GEMM+act summed, exact since they are
dependency-serialized; best-of-N timing so the ratio is clock-stable). The relu case in detail:

| size | Mercury fused | Mercury GEMM+relu | cuBLAS GEMM+relu | fusion vs own chain | **fused vs cuBLAS chain** |
|------|---------------|-------------------|------------------|---------------------|---------------------------|
| 512³ | 0.035 ms | 0.077 ms | 0.084 ms | 2.21× | **2.41×** |
| 1024³ | 0.163 ms | 0.199 ms | 0.193 ms | 1.22× | **1.18×** |
| 2048³ | 1.258 ms | 1.579 ms | 1.177 ms | 1.26× | 0.94× |

So **at L2-resident sizes (≤1024³) the single fused kernel beats the cuBLAS GEMM+activation chain
1.18–2.41×** — the first place Mercury is *faster than cuBLAS*, precisely because it does the fusion
cuBLAS structurally cannot. Fusion beats Mercury's own two-kernel chain at **every** size (1.2–2.2×,
biggest where the GEMM is small and the saved relu pass is a larger share). silu and gelu fuse with the
same effect — across 512³–2048³ all three beat the cuBLAS GEMM+activation chain by **~1.1–1.4×** at
boost clock (the margin is the eliminated activation kernel's launch + C round-trip, which cuBLAS
cannot fuse). The **fused bias** epilogue `act(x·Wᵀ + bias)` — the canonical `nn.Linear`/FFN form — is
fused too: because the WMMA fragment→column map is opaque, the bias path `wmma.store.d`s each tile into
a per-warp SMEM scratch and re-reads by explicit (row,col) to add `bias[col]` before the activation
(`_sm_db_bias{,_relu,_silu,_gelu}`; identity-with-bias is the affine Linear). And the recognizer routes
`act(matmul(…) [+ bias])` straight from Mercury source to these fused kernels under `--backend=gpu`, so
the fusion is a **compiler feature, not a host-API call** (gates `gpu_backend_fused_epilogue_matches_interp`
+ `…_bias_…`, each asserting the offload fired).

**Compile latency + cubin cache (M10).** Mercury emits PTX and the driver JITs it to SASS; there is no
30–120 s autotuning compile like Triton/TorchInductor. Measured (`cubin_cache_compile_latency`, RTX
4050): a from-scratch driver JIT of the fp16 WMMA module (43 KB PTX → 26 KB cubin) is **0.76 ms** — so
even *cold*, Mercury's compile is **~4×10⁴–1.6×10⁵× faster** than a Triton/Inductor cold build (M10
asked ≥100×). On top of that, a **persistent cubin cache** (`cubin.rs`: the driver's own `cuLink*` JIT
emits the SASS, keyed by a hash of the PTX + driver version, loaded warm via `cuModuleLoad`) drops the
warm load to **0.16 ms** — 2.0× under the driver's own JIT-cache-warm PTX load and 4.8× under cold, and
unlike the driver's opaque/clearable compute cache it is **portable and deterministic** across process
runs. Every kernel load now flows through it (`Gpu::load_module_cached`), degrading gracefully to a
direct PTX JIT on any cache miss/failure, so it can never break a load that would otherwise succeed.

**Memory-bound bandwidth (M9): ≥90% of peak HBM.** The denominator is honest — the device's *own*
theoretical peak from its memory clock × bus width (`cuDeviceGetAttribute`, the exact `deviceQuery`
formula `2 × clock × busBytes`), reading **192.0 GB/s** on the RTX 4050, dead-on the spec. Measured on
256 MB/array buffers (far past L2, so it's DRAM not cache), best-of-N to capture the un-throttled clock
(the laptop dynamically down-clocks memory — the *same* kernel spanned 63%–96% of peak across runs
purely from clock state, so a single round is meaningless and the best round is the device's true
capability): the **saxpy triad reaches 183.9 GB/s = 95.7% of peak** (M9 met — a memory-bound kernel at
≥90%). The vectorized copy (`COPY_V4`, 128-bit `ld/st.global.v4` with 4× ILP — four independent float4
loads in flight per thread, the memory-level parallelism a 1:1 read/write copy needs to hide latency +
bus turnaround) sits at 168.8 GB/s = 87.9%, the 1:1 mix's turnaround tax. Correctness gates speed: the
copy is checked bitwise-exact first. Reproduce: `… --ignored --nocapture hbm_bandwidth`.

**Internal fp16 WMMA roofline** (retained as a same-run, clock-invariant compute ceiling — one fragment
load then a long `wmma.mma` loop over 4 independent accumulators, ~zero hot-loop memory traffic): the
fp16-mt GEMM sits at ~25–72% of it and fp8-mt at ~75–100% of it across 2048³–4096³. The roofline itself
(~17 TFLOP/s here) is far below the ~80–97 TFLOP/s *rated* dense peak because the 6 GB mobile 4050 is
power-capped — and, as the cuBLAS column shows, it is ≈ the cuBLAS-achievable rate, not a kernel-bounding
wall. Reproduce: `tensorcore_roofline_pct`.

**Flash-attention — register-resident `mma.sync` (the FlashAttention-2 form).** `flash_d64_m` runs the
online-softmax attention on the Ada tensor cores with the output `O`, the running max `m`, and the
denominator `l` held in **registers** across the whole K-loop — *no* per-step SMEM round-trip. It is built
on the hand-placed `mma.sync.m16n8k16` fragment layout (proven bit-exact by `mma_m16n8k16_layout_verifies`)
and exploits the FA2 trick that the `Q·Kᵀ` score accumulator's register layout *is* the `A`-operand layout
that `P·V` needs, so the softmax probabilities are repacked in-register with no SMEM bounce. It supersedes
the prior warp-per-row kernel (372 GFLOP/s @4096) and the WMMA-store-to-SMEM kernels. Single-head `d=64`,
f16 in / f32 out, matching a CPU f64 two-pass-softmax reference to the f16-rounding floor (max_abs ~9e-5).
A `causal` sibling `flash_d64_mc` skips the upper-triangle key blocks and masks the diagonal (gated vs an
independent f64 causal oracle).

The **production** kernel is now `flash_d64_mp` — the same register-resident core with a **`cp.async`
double-buffered K/V shared-memory stage** layered on: each 16-key K+V slab is a contiguous global slab, so
the warp stages it with 16-byte `cp.async.cg` copies and **prefetches block `kb+1` while the tensor cores
consume block `kb`**, then reads the MMA fragments from SMEM. The math and key order are identical to
`flash_d64_m`, so it is bit-identical (the gate cross-checks both); `flash_d64_mpc` is its causal sibling.

Measured same-run on the RTX 4050 (`flash_vs_peers`):

| seq (1 head, D=64) | Mercury fused flash | cuBLAS unfused chain (Tier B) | naive CUDA-C (Tier A) |
|-----|---------------|------------------------|----------------------------|
| 512  | 2245 GF/s | 476 — **4.7× win** | 5 — **442×** |
| 1024 | 6190 GF/s | 1706 — **3.6×** | 15 — **406×** |
| 2048 | 8487 GF/s | 2066 — **4.1×** | 25 — **338×** |
| 4096 | 7676 GF/s | 1524 — **5.0×** | 27 — **284×** |

(Absolute GFLOP/s is clock-bound; the **× ratios are the same-run honest figures**, all measured
back-to-back under one pinned clock after a GEMM-hammer warmup.)

**M6 (beat naive CUDA-C): a wide win at every context length** — re-measured same-run with the production
`flash_d64_mp`, **205–738× single-head and 275–322× at H=12** vs the naive CUDA-C flash (the literal "beat
the hand-written C kernel on the GPU"; absolute GFLOP/s is clock-bound, the ×-vs-naive ratio is the same-run
honest figure). **The `cp.async` pipeline — the lever this section previously flagged as future work — is
now implemented and shipping.** `flash_pipe_vs_mma` times `flash_d64_mp` against `flash_d64_m` back-to-back
under one pinned clock: **0.28× / 0.34× / 0.53× / 0.88× the time single-head at S=512 / 1024 / 2048 / 4096**
(1.14–3.6× faster, largest where the warp was purely latency-bound) and **0.84–0.90× at the H=12 filled
regime** (1.1–1.2×, the per-warp latency bound). **M5 (vs a tensor-core library): Mercury's fused flash is 3.6–5.0× the cuBLAS unfused attention chain**,
same-run, single-head — the gap widening at long S where the materialized S×S scores hurt the chain most.
The chain is the canonical *pre-FlashAttention* attention (Q·Kᵀ and P·V on tensor-core cuBLAS, the S×S
scores spilled to HBM with a softmax between), so the gap **is** the value of fusion, measured against the
gold-standard library for the matmuls — not a strawman. Two honesty caveats: **(1)** this is **not "% of
FA2"** — a genuinely *fused* FA2 kernel (cuDNN / FlashAttention) is faster than the unfused cuBLAS chain, so
beating the chain 3.6–5× is the *library-bar* result, not parity with the best fused kernel; **(2)** a fused
FA2-class CUDA-C peer is **not buildable on this toolkit-free box** — the `nvrtc_wmma_probe` test shows NVRTC
has *no header search path at all* (even `#include <cuda_fp16.h>` fails to open), so `nvcuda::wmma` cannot be
compiled. (An earlier cross-process re-measure against torch SDPA was **uninterpretable** — torch's *own*
throughput swung ~2× run-to-run on this power-capped part, 66% vs 112% on clock alone — which is exactly why
the in-process cuBLAS chain is the reportable Tier-B bar.) The register-resident core was itself **1.5–5.9×
the prior WMMA flash** (`flash_mma_vs_wmma`), and `flash_d64_mp` compounds the `cp.async` win on top
(**1.1–3.6× `flash_d64_m`**, **205–738× naive CUDA-C** = M6).

**Multi-head** (`grid.y = H`, the kernel folds head `ctaid.y`'s `[H,S,D]` base offset into the pointers —
zero extra params, single-head stays `grid.y=1`). At small S one head's `S/16` blocks can't fill 36 SMs
(S=512 → 32 warps total); batching the GPT-2 `H=12` heads does. Same-run (`flash_vs_peers`, so
clock-invariant): multi-head holds a **flat ~860 GFLOP/s across S** while single-head climbs 117→438 at the
same clock — i.e. multi-head is **7.4× / 3.7× / 2.0× the single-head throughput at S=512 / 1024 / 2048**,
and **275–322× a naive multi-head CUDA-C flash**. The flat saturation said the filled GPU was per-warp
**latency**-bound — which the `cp.async` software-pipelined K-loop now attacks directly: the production
`flash_d64_mp` prefetches the next K/V block under the current block's MMA, for a same-run **1.1–1.2×** at
this filled regime (and up to 3.6× single-head, where the latency was unhidden). **Multi-warp-per-CTA K/V
sharing was then tried** (`flash_d64_mp4`/`_mp8`, W warps sharing one cooperatively-staged K/V block;
`flash_mw_vs_mp`) and is **only marginal** — ~6–8% at the H=12 filled regime and a *regression* single-head
— so the kernel is **not** strongly K/V-bandwidth-bound (`mp`'s per-warp pipelining already captures it).
With the cuBLAS unfused chain now the same-run Tier-B bar, Mercury's flash is already **3.6–5.0× the best
library attention buildable on this box** — and there is no *fused* FA2 peer measurable here to set a ≥90%
target against. `ldmatrix` conflict-free fragment loads (the strided V `u16` pairs at a 128-byte SMEM stride
are the prime bank-conflict suspect — a *per-warp throughput* issue, consistent with the not-bandwidth-bound
finding) remain the one untried kernel lever, but it is an uncertain further squeeze with no
locally-measurable FA2 denominator to chase.

**Whole transformer layer, GPU-resident.** A complete pre-norm encoder layer — RMSNorm → Q/K/V
projections → flash-attention → output projection → residual → RMSNorm → FFN (SiLU) → residual —
runs entirely on the device: inputs/weights upload once, every op reads/writes device buffers with no
host round-trip of activations between them, only the final `[S, D]` copies back. It matches a CPU
f64 reference of the same layer to **max_abs 6e-7, max_rel 2.7e-4** (errors barely accumulate across
the 8-op chain) — the payoff of having every transformer op as a device kernel. End-to-end latency
(`d=64`, `dff=256`, full call incl. weights H2D + D2H — a real model keeps weights resident, so this
is conservative): **1.33 ms/layer at S=256, 3.13 at S=512, 6.07 at S=1024** (~160–190 K tokens/s).
The layer is **deterministic** — bit-identical run-to-run, since every kernel uses a fixed grid and
warp-butterfly reductions with no atomics.

**Beating the library call-chain, end-to-end (M13).** The layer above runs f32; its fp16 sibling
(`ResidentLayerF16`) puts every projection on the WMMA tensor cores and folds the two residual adds +
the SiLU into the GEMM epilogues (flash stays f32 for the softmax range). Stacked N-deep,
`ResidentModelF16` runs the whole model GPU-resident — one input H2D, N×(13 launches), one output D2H,
**no per-layer weight re-upload**. The honest external bar is **cuBLAS**: a transformer built on it is a
chain of `cublasGemmEx` calls glued by hand-written norm/attention/activation kernels that **cannot
fuse** the activation or the skip-add into the GEMM. The peer (`CublasChainLayer` / `CublasChainModel`)
is built scrupulously fair — cuBLAS runs f16-in / **f32-out** (`cublasGemmEx`, `CUDA_R_32F` C,
`COMPUTE_32F`) so it pays the **same** casts as Mercury's WMMA (neither pays a post-GEMM cast), and it
reuses Mercury's *identical* norm/flash/cast/SiLU/add kernels, so the only difference measured is the
GEMM + the epilogue fusion. Both stacks are tolerance-gated against the same f64 oracle before any speed
counts. Both also pipeline their kernel chain on one stream: every intermediate is an `alloc_zeros`
(async device memset) rather than an `memcpy_stod(vec![0;n])` (a *synchronous* H2D that used to stall the
stream ~13× per layer and dominated the cost — removing it cut the layer ~3×). Measured same-machine,
same-buffers, same-run, clock-pinned (`d=64`, `dff=256`, resident):

| single layer | Mercury fused | cuBLAS call-chain | **Mercury faster** |
|---|---|---|---|
| S=256  | 0.633 ms/layer | 0.923 ms/layer | **1.46×** |
| S=512  | 0.813 ms/layer | 1.067 ms/layer | **1.31×** |
| S=1024 | 0.745 ms/layer | 1.121 ms/layer | **1.50×** |
| S=2048 | 1.151 ms/layer | 1.173 ms/layer | **1.02×** |
| S=4096 | 1.250 ms/layer | 1.200 ms/layer | 0.96× (≈par) |

Both stacks run the *same* attention (the register-resident flash above), so the measured gap is purely
**GEMM + epilogue fusion** — the per-call dispatch + the separate add/SiLU launches, paid at every layer,
that the resident fused model folds away. Mercury wins **1.31–1.50× through S=1024**; at S≥2048 the layer
becomes attention-bound (the flash is the same on both sides) so the GEMM-fusion edge shrinks to ≈par. The
register-resident flash also collapsed the long-context layer itself: **S=4096 dropped 3.96 → 1.25 ms
(~3.2×)** vs the prior WMMA-flash layer, since attention was ~3.2 ms of the old 3.96. (Decomposed: Mercury's
WMMA GEMM alone, *unfused* with identical glue, is ~par with cuBLAS at these small-K (64/256) shapes —
*regime specific*, cuBLAS still wins the isolated large-K GEMM ≥2048³, the GEMM table above — and the fused
epilogues the add/SiLU cuBLAS structurally can't fold in add ~1.08–1.09×.) The earlier depth sweep (S=512,
pre-register-flash) showed the ratio **widening with depth** 1.20→1.32× as the fixed per-layer overhead
amortises. Run: `… --ignored --nocapture cublas_chain_vs_mercury` / `resident_model_vs_cublas`.

**PyTorch (Tier C) — Mercury now wins at EVERY S, including S=4096.** The same layer in PyTorch
(`bench/pytorch/transformer_layer_peer.py`, fp16 eager — tensor-core matmuls + fused **SDPA flash
attention**, f32 norm/softmax, f64-verified on the same RTX 4050) is **overhead-bound** (~15 eager kernel
launches + Python dispatch per layer), so its time barely tracks the compute. Mercury is **resident and
overhead-free** — pure compute. With the register-resident flash (`flash_d64_m`), one consistent run:

| case | Mercury (register flash) | PyTorch eager | result |
|---|---|---|---|
| single layer, S=256 | 0.63 ms | 2.03 ms | **Mercury 3.2×** |
| single layer, S=512 | 0.81 ms | 2.15 ms | **Mercury 2.7×** |
| single layer, S=1024 | 0.75 ms | 2.94 ms | **Mercury 3.9×** |
| single layer, S=2048 | 1.15 ms | 1.94 ms | **Mercury 1.7×** |
| single layer, S=4096 | 1.25 ms | 1.69 ms | **Mercury 1.35×** |
| stack (S=512, depth 1–8) | ~0.34 ms/layer | ~2.0–2.6 ms/layer | **Mercury ~6–7.8×** |

This **flips the prior crossover**: with the old WMMA flash PyTorch won S=4096 by ~1.75–3.5×; the
register-resident flash collapsed the S=4096 layer **3.96 → 1.25 ms** and Mercury now *leads* there 1.35×.
Note Mercury wins the *layer* at S=4096 even against torch's own fused SDPA flash inside that layer —
because it pays none of torch's launch/dispatch overhead and fuses the epilogues; and its *isolated*
attention is itself **3.6–5.0× the cuBLAS unfused chain** (the table above), so the layer lead rests on a
flash that already beats the gold-standard library attention.

**Real GPT-2 multi-head shape (new).** The table above is a single-head D=64 layer — useful for the
fusion/residency comparison, but not a real transformer. `ResidentLayerF16::new_mha` now runs **true
multi-head attention** at the **GPT-2 shape (D=768, H=12 heads of dh=64)**: the QKV projection's `[S,D]`
output is bridged token-major↔head-major by two memory-bound transpose shims (`ptx::HEAD_TRANSPOSE_PTX`,
with the f32→f16 cast folded into the forward one) around the existing `grid.y=H` tensor-core flash, so the
production flash kernel is untouched. Gated bit-reproducibly against a **per-head** f64 reference at
D=768/H=12/S=512 (`transformer_layer_mha_matches_reference_within_tol`, max_abs 8.05e-3). Same-run
throughput at this real shape (`cublas_chain_vs_mercury_mha_layer_throughput`) — fused multi-head Mercury
vs the **multi-head cuBLAS-chain layer** (attention common to both, so the gap is purely GEMM + epilogue
fusion):

| S | Mercury fused | cuBLAS chain | net | GEMM (Mercury-unfused / cuBLAS) | fusion |
|---|---|---|---|---|---|
| 512  | 1.073 ms | 1.020 ms | chain 1.05× | 0.90× | 1.05× |
| 1024 | 2.402 ms | 2.360 ms | chain 1.02× | 0.90× | 1.09× |
| 2048 | 6.389 ms | 6.422 ms | **Mercury 1.01×** | 0.92× | 1.09× |
| 4096 | 18.26 ms | 17.61 ms | chain 1.04× | 0.93× | 1.03× |

**Honest finding:** at the real GPT-2 shape the fused layer is **~par with the cuBLAS-chain layer
(0.96–1.01×)** — *not* the clearer win the D=64 toy shows. The decomposition says why: Mercury's WMMA GEMM
is **0.90–0.93× of cuBLAS at D=768** (the GEMM cliff — the same large-size gap `gemm_vs_peers` reports,
~34% @4096³), and the fused residual/SiLU epilogues (**1.03–1.09×**, which cuBLAS structurally can't do)
nearly but not fully offset it. **Closing the GEMM cliff flips this to a clear win** — the single
highest-leverage GPU item, exactly what the parallel GEMM-cliff workstream targets. (A cross-process
PyTorch comparison at this shape, like the D=64 table above, remains a documentation follow-up.)

Honest caveats: **eager** PyTorch only — `torch.compile`/Inductor needs Triton, which has no working Windows
install (`torch.compile` raised `Cannot find a working triton installation` here). Cross-process and
**clock-noisy** at this ~1–2 ms scale (the laptop GPU boosts ~7×), so treat the ratios as order-of-magnitude
and the **direction** — Mercury faster at every S, by a margin growing toward small S — as the robust signal.
The clock-invariant backbones are the same-run `flash_vs_peers` (Tier-A 172–305×, Tier-B 59–77% of FA2) and
the `cublas_chain_vs_mercury` same-run layer table above.

**Determinism, every kernel (M12).** Not just the layer: `gpu_kernels_bit_reproducible` asserts every
reduction-bearing family — `gemm_nt_f16` and its `_sm`/`_sm_db` variants, the three fused row norms,
flash-attention, conv2d, and the sum/dot reductions — returns **bit-identical** output across runs on
identical inputs (fixed grids, atomic-free, fixed-order reductions). cuBLAS offers no such contract:
`reproducibility_vs_cublas` shows Mercury's fp16 GEMM bit-identical across three runs *by construction*,
while cuBLAS's reproducibility is incidental — NVIDIA documents none across library versions, GPU
architecture, or its heuristic algorithm/split-K selection. Reproducible-by-default matters for
regression gates, debugging, and regulated training.

**W4A16 int4 weight-only decode (M4) — leading an immature field.** The LLM-decode workhorse: 4-bit
weights (group-wise quantized, per-group fp16 scale + optional AWQ/GPTQ zero-point), fp16 activations.
`gemm_nt_w4a16` reads the **packed int4 weight from global** (8 weights per `u32` — a **4× smaller
weight footprint** than fp16, the bandwidth win that makes decode memory-bound-friendly), **unpacks
int4→fp16 on the fly inside the K-loop** with the canonical Marlin/AWQ fast path (one `lop3` extracts an
interleaved nibble *pair* straight into an `f16x2` of `1024+u`, then a single `sub.rn.f16x2` zero-offset
and `mul.rn.f16x2` scale dequant **two weights per op** — no per-element convert), then runs the
*identical* fp16 `wmma.mma.sync.m16n16k16` tensor-core tile as the dense path. The only deviation from an
*exact* dequant is the same f32-accumulation tolerance the fp16 GEMM carries (~2e-3): the gate
reconstructs the weight bit-for-bit on the CPU (`reference_w4a16`) and matmuls in f64; both the symmetric
(offset-binary, `Z=8`) and asymmetric (`Z=zero[group]`) paths pass, and the kernel is deterministic
run-to-run (M12).

**No robust library int4-decode GEMM is bindable on this box** (cuBLASLt offers no general W4A16 decode),
so M4 is a *documented lead*, stated honestly: the Tier-A peer is a **naive CUDA-C W4A16** kernel (NVRTC,
one thread/output with an on-the-fly unpack). One representative back-to-back run (`int4_gemm_vs_peers`,
clock-warmed, best-of-4, checksum-cross-checked vs the naive peer):

| shape (M×K×N) | Mercury W4A16 | × vs naive CUDA-C | × vs Mercury fp16 (same tile) | weight HBM/pass |
|---------------|--------------:|------------------:|------------------------------:|-----------------|
| 64×4096×4096 (decode) | ~12.6 TFLOP/s | **~180–213×** | **~4.0–4.2×** | int4 8.4 MB vs fp16 33.6 MB (**4×**) |
| 256×2048×2048 | ~14.9 TFLOP/s | ~165× | **~1.46×** | 2.1 vs 8.4 MB |
| 64×2048×2048 | ~8.8 TFLOP/s | ~92× | ~1.00× | 2.1 vs 8.4 MB |
| 512×512×512 | ~9.1 TFLOP/s | ~78× | ~1.00× | 0.13 vs 0.5 MB |

The **4× weight-bandwidth reduction lands in the large-weight decode regime** (64×4096), where decode is
weight-BW-bound and the fp16 path stalls on weight reads — Mercury is ~4× faster there. After the
Marlin/AWQ `lop3` unpack made the dequant nearly free, **W4A16 is ≥ the fp16 path in *every* regime** (no
int4 penalty anywhere; it even *beats* fp16 1.46× at 256×2048 because the 4× smaller weight tile fits L2
far better). **Static-shape specialization** (Mercury's no-library lever — `w4a16_static_ptx` bakes M/N/K
as constants so ptxas strength-reduces the strides to shifts) adds a further **1.05–1.30×** over the
same-loaded dynamic kernel, bit-identical. The remaining headroom is in the thin-M decode shape, which is
occupancy-bound (M=64 → few CTAs); split-K is the identified next lever (its deterministic reduction caps
the net gain).

**Other op categories** (all emit+execute, tolerance-gated on the 4050): elementwise (saxpy/vadd),
deterministic reductions (sum/dot/max — bit-exact for max, tolerance for the f32 sums), activations
(relu/exp/sigmoid/tanh/silu/gelu via SFU), fused row norms (softmax/LayerNorm/RMSNorm, one warp per
row), and direct conv2d. Reproduce: `cargo test -p mercury_codegen_gpu --features gpu` (correctness;
skips cleanly with no GPU) and `… --release -- --ignored --nocapture` (throughput).

**End-to-end `--backend=gpu`.** Beyond the crate's host API, the GPU is wired as a third compiler
backend: `mercuryc --features gpu --backend=gpu --run foo.mer` tree-walks the program on an
*offloading interpreter* and runs recognized GEMM / activation / reduction / fused-norm calls on the
device (an `Accelerator` seam in `mercury_interp` that the driver fills with `mercury_codegen_gpu`).
**Fusion reaches the source level:** a Mercury `act(matmul(x,w))` folds to `mercury_sgemm_nt_epi`,
which the GPU seam routes to the *single fused WMMA kernel* (`gemm_nt_f16_sm_db_{relu,silu,gelu}`, the
one that beats the cuBLAS GEMM+activation chain) — so the fusion win is a compiler feature, not just a
host API call (`gpu_backend_fused_epilogue_matches_interp`, offload-fired + tolerance-gated).
With no accelerator — the differential oracle and every other caller — the interpreter path is
byte-for-byte unchanged, so the toolchain-free core is untouched and the bit-exact CPU gate is intact.
The CPU↔GPU boundary stays a tolerance differential: the `gpu_backend_*` driver tests run each family
on the interp oracle and the GPU over identical buffers and require both that the offload *actually
fired* and that outputs match within `c·√K·ε` (GEMM bit-exact; silu ~5e-7, dot ~7e-7, softmax ~3e-8
abs on this box). A device error surfaces as an error, never a silent CPU fallback.

### M7 runtime — device memory pool + CUDA graphs + multi-stream (decode/small-batch latency)

A resident transformer layer is ~13–16 individual kernel launches, each touching a few KB at decode
sizes; per-op `cuMemAllocAsync`/free and per-kernel `cuLaunchKernel` then dominate. The Phase-7 runtime
removes both — a **device memory pool** (`pool.rs`, a bump arena over one slab: no per-op alloc/free,
no zeroing memset) and **CUDA-graph capture/replay** (`graph.rs`: capture the whole layer once, replay
with one `cuGraphLaunch`). Every optimized path is gated **bit-identical** to the per-op-alloc +
individual-launch baseline (the slab is poisoned `0xFF` first to expose any read-of-uninitialized
scratch) and deterministic across replays; all numbers are **same-run** ratios (the ~7× laptop clock
swing makes absolutes meaningless), reported as the best of ≥3 re-runs.

| Workload (RTX 4050) | eager → pooled | eager → **graphed** | note |
|---|---|---|---|
| 1 layer, decode (S=64, D=64) | ~1.9–2.0× | **~4.5–6.4×** | graph adds ~2.2–3.3× on top of pooling |
| 1 layer, S=512 D=128 | ~1.8–2.1× | ~3.2–3.5× | |
| 1 layer, GPT-2 (S=512 D=768 H=12) | ~1.0× | ~1.1× | compute-bound; little overhead to remove |
| **12-layer model, decode** | — | **~6.5–6.9×** | ~2.5 ms → ~380 µs; 156 launches → **1 cuGraphLaunch** |

The whole-model win **compounds with depth** (more launches folded) while graphed latency scales
linearly at ~32 µs/layer (pure compute + one launch), and the pool footprint stays at **one layer's
232 KiB** for the whole stack (reset between layers). **Multi-stream:** copy/compute overlap with
pinned host staging (`PinnedBuf`, 2 streams, event-ordered) is gated bit-identical to serial but is
~1.0× here — the resident layer is strongly compute-bound, so even GPT-2's 1.5 MiB/item transfer is a
small fraction of compute (the prefill/slow-link lever, not realized on this workload). The lever that
*does* help the underutilized decode regime is **concurrent forwards** — K=4 independent requests on K
streams reclaim idle SMs for **~1.3×** decode-serving throughput (and ~1.0× at the GPT-2 shape, which
already saturates). Reproduce: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored
--nocapture pool_graph_vs_unpooled decode_stack_latency overlap_throughput concurrent_forwards_throughput`.

## Honest summary

- **Compile time:** ~100–260× faster than gcc/rustc (geomean ~135–155×). Robust every run; the metric
  that dominates ML iteration.
- **Matmul / nn.Linear (the flagship ML kernels):** Mercury **wins single-thread (~3–26×) and
  dominates parallel (~9–104×)**, and the lead **grows with matrix size** — the compiler tiles,
  packs, and register-blocks where gcc/rustc leave the naive nest. The single-core GEMM holds
  **~110–120 GFLOP/s** (≈90% of one P-core's AVX2-FMA peak); the parallel one reaches ~520 at 1024³
  and ~690 at 2048³. This is a reversal of the previous honest loss (single-core matmul used to be ~3×
  *behind*). The dispatch also fires on **runtime dimensions**, so the win applies to general matmul
  functions, not only fixed-size kernels.
- **Fused FFN epilogue (`act(x·Wᵀ [+ bias])`):** a `nn.Linear` immediately followed by a
  bias-add/activation loop folds into **one** `mercury_sgemm_nt_epi` call — the bias and activation
  are applied in the GEMM's C-tile writeback, so the M×N output is written **once** instead of streamed
  again by a separate pass. The activation set covers the transformer FFNs: **ReLU, GELU** (BERT/GPT-2,
  matching `vmath`'s ≈1-ULP form) and **SiLU** (the LLaMA gate), with **bias optional** so the bias-free
  `silu(x·Wᵀ)` **SwiGLU** projection fuses too. On top of the GEMM win above, this removes a full
  read-modify-write pass over the activation tensor; both backends call the identical kernel, so the
  fused result is bit-equal to the unfused `matmul → bias → activation`.
- **int8 quantized `nn.Linear` (u8×i8→i32):** dispatched to an **AVX-VNNI `vpdpbusd`** register-blocked
  microkernel (2×4-tiled on the single-threaded path, halving B-matrix traffic), **~1.5–2.5× faster
  than gcc single-core** (gcc `-march=native` uses `vpdpbusd` too, so it's a fair same-instruction
  comparison) — the lead widening with size — and ~4.6–14.7× with `@parallel` (clock-state-dependent).
  Rust runs essentially scalar here (~5–12× behind). Integer math makes the cross-language check
  **bit-exact**, not a tolerance.
- **Transcendentals / activations (exp, log, exp2, log2, exp10, log10, expm1, log1p, tanh, sigmoid, GELU, SiLU, ELU,
  leaky_relu, softplus, softsign, logsigmoid, mish, SELU, tanhshrink, hardsigmoid, hardswish, sin, cos, tan, atan, asin, acos, erf, and the
  cbrt, hyperbolic family sinh, cosh, asinh, acosh, atanh — 35 in all):** **~2–13× faster** than C's scalar `libm` — Mercury dispatches the loop to a **256-bit AVX2
  ≈1-ULP poly kernel** (`mercury_vmath_f32`), where gcc/rustc cannot vectorize a loop with an
  `expf`/`logf`/`tanhf`/`sinf`/`cosf`/`erff`/`asinhf` call. This is the transformer/vision activation family and the cleanest
  compute-bound win (it roughly doubled when the kernel moved from the 128-bit vectorizer to 256-bit).
  `sin`/`cos` (the **RoPE** rotary-embedding transcendentals) win the most (~6–8.6×) — `libm`'s
  `sinf`/`cosf` are heavier than `expf` — and `erf` gives the exact BERT/GPT-2 GELU.
  All of them are first-class intrinsics composing the shared ≈1-ULP `exp`/`log` (e.g.
  `softplus = ln(1+eˣ)`, `mish = x·tanh(softplus)`), so the whole family is exact and bit-identical
  across backends; an `@parallel` activation runs multicore × 256-bit (~28×); log-softmax /
  cross-entropy compose the exp/log win.
- **Fused normalizations (softmax / LayerNorm / RMSNorm):** the per-token transformer norms are
  recognized from their multi-pass source and folded into one `mercury_norm_f32` call (256-bit AVX2 +
  8-lane reassociated reductions + the vectorized `exp`), **~1.9–6.6× faster than C** — softmax most
  (the vectorized `exp` dominates), RMSNorm least (a single reduction, diluted by its elementwise
  passes). The **affine** form real models run (learned per-channel scale γ + shift β) dispatches to
  `mercury_norm_affine_f32` and holds the same **~1.7–3.7×** (γ/β fused into the writeback). Both
  backends marshal the identical kernel, so the differential oracle stays bit-exact.
- **Convolution:** lowered as im2col + GEMM (the XLA/cuDNN strategy), Mercury runs a 3×3 conv
  **~6–7× faster** than the idiomatic hand-written direct-convolution nest in C — the matmul
  recognizer accelerates conv for free.
- **Reductions:** ~2.6–2.8× faster (lane-accumulator reassociation), incl. `fmax`/`fmin` (softmax's
  row-max).
- **Auto-parallel:** ~1.8–7.6× faster than idiomatic single-threaded C across elementwise kernels —
  bounded by aggregate memory bandwidth, not core count (these kernels are memory-bound).
- **Single-thread memory-bound elementwise (saxpy/scale/residual/poly):** now a **win** (~1.1–1.5×
  vs C at N=2²⁰, widening to ~1.3–1.6× at >L3 sizes), where it used to be a tie. A recognized
  streaming map dispatches to the 256-bit AVX2 `mercury_velem_f32` / `mercury_vhorner_f32` kernels
  (4×-unrolled for ILP) which emit **non-temporal stores** once the working set spills L3 — skipping
  the read-for-ownership traffic every cacheable store pays, a store path gcc/rustc don't emit
  automatically. The non-temporal decision keys on the *total* streamed bytes (all live arrays), so a
  cache-resident map keeps its normal store (where a needless `vmovntps` would lose): `relu` at N=2²⁰
  (2-stream, 8 MiB, L3-resident) is a clean tie, and a win at >L3. This is the same play as the GEMM
  and activation families (`mercury_sgemm*`, `mercury_vmath_f32`) — true 256-bit width plus a
  domain-aware store policy the generic 128-bit-only Cranelift vectorizer can't reach.
- **bf16 mixed-precision reductions (bf16 storage, f32 accumulate):** `bf16` is real 2-byte storage
  at bf16 precision (round-to-nearest-even), bit-exact across backends, and a `s += (x[k] as f32)
  [* (y[k] as f32)]` reduction over `[bf16; _]` arrays now **dispatches to a SIMD kernel**
  (`mercury_dot_bf16` / `mercury_sum_bf16`: widen to f32, 8-lane f32 accumulate) — the standard ML
  mixed-precision contract. The payoff is **bandwidth** (bf16 moves half the bytes of f32): bf16 dot
  runs **~3.0–3.5× faster than idiomatic single-threaded C** and sum **~6–8×** (C's unary f32 sum is
  latency-bound), the lead **growing as the working set spills L3** (see the table below). On this
  AVX2 box (no bf16 FMA) a bf16 *GEMM* would widen to f32 and match f32 throughput — a footprint
  feature, not a FLOP/s win — so that path stays at f32; the reduction kernels are where bf16 pays.
  The same dispatch now also covers **bf16 elementwise** (`out[k] = a*(x[k] as f32) + b*(y[k] as
  f32)` → `mercury_axpby_bf16`, bf16 in / f32 out): ~1.3× a plain f32 axpby (8 vs 12 bytes/elem; below
  2× because the f32 output + write-allocate dilute the half-width-input savings — a bf16 output would
  reach ~2× but needs a narrowing-store differential).
- **GPU backend (RTX 4050):** a PTX-emitting, driver-JIT GPU path (no CUDA toolkit) runs every
  transformer op category on-device, tolerance-gated. **Tensor-core GEMM** (fp16/bf16 in, f32
  accumulate) hits **~9–13 TFLOP/s** (clock-dependent) — ~5–6× the f32 path on the same GPU;
  **fp8 (E4M3) via hand-laid `mma.sync` is validated bit-exact**, and its fragment-reuse multi-tile
  kernel is now the **fastest** tensor-core path — ~2.1–2.4× the naive single-tile fp8 and ~1.3–2.3×
  fp16/bf16 in the same run (realizing Ada's ~2× fp8 rate once it's compute-bound). **`cp.async`-pipelined
  register-resident `mma.sync` flash-attention** (online softmax, O/m/l in registers, K/V prefetched into
  double-buffered SMEM under the MMA) runs **1.1–3.6× the un-pipelined kernel same-run**
  (`flash_pipe_vs_mma`), **205–738× a naive CUDA-C flash** (M6), and **3.6–5.0× a cuBLAS unfused attention
  chain** (M5, same-run — the gold-standard *library* bar; a fused FA2-class CUDA-C peer can't be compiled
  here, NVRTC has no headers, and cross-process torch SDPA swings ~2× run-to-run so it isn't reportable);
  a **whole pre-norm transformer layer runs end-to-end GPU-resident**
  (matching a CPU f64 reference to max_rel 2.7e-4, deterministic run-to-run) and now **beats PyTorch eager
  at every S, including S=4096 (1.35×)**. Numbers are honest for a power-capped 6 GB
  mobile GPU, not a datacenter part. The CPU↔GPU differential is a `c·√K·ε` tolerance over the full output.
- **Safety:** Mercury checks tensor **shapes at compile time** (in the type system), a class of bug
  C/C++/Rust-with-raw-pointers cannot catch.

Where Mercury wins is where a tensor compiler should: compile speed, matmul/GEMM throughput (now on
runtime dimensions too), convolution (im2col + GEMM), vectorized transcendentals (the transformer
activation family, including `log` for log-softmax/cross-entropy), fused row normalizations
(softmax/LayerNorm/RMSNorm), bf16 mixed-precision reductions (bandwidth), automatic parallelism,
automatic vectorization (including reductions), automatic fusion, and shape safety — plus a
PTX-emitting **GPU backend** that takes the same ops to the RTX 4050's tensor cores (~12.8 TFLOP/s
fp16/bf16 GEMM, fused flash-attention, a whole layer GPU-resident).
