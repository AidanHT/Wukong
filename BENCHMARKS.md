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
- **bf16 mixed-precision reduction dispatch.** A `s += (x[k] as f32) [* (y[k] as f32)]` reduction over
  `[bf16; _]` arrays (bf16 storage, f32 accumulate — the standard ML contract) is recognized and
  lowered to **`mercury_dot_bf16`** / **`mercury_sum_bf16`** (widen bf16→f32, 8-lane f32 accumulate).
  Because bf16 moves **half the bytes** of f32, this is a *bandwidth* win that grows as the data spills
  cache: **~3.0–3.5× vs C** for dot, **~6–8×** for sum (C's unary f32 sum is latency-bound). bf16
  storage is bit-exact across backends, and both call the identical kernel, so the gate stays exact.
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
Mercury recognizes a pure `out[i] = f(x[i])` loop for
`exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`mish`/`selu`/`tanhshrink`/`hardsigmoid`/`hardswish` and lowers the whole
loop to a **256-bit AVX2/FMA runtime kernel** (`mercury_vmath_f32`) — the same domain-aware dispatch
as matmul→GEMM. The kernel runs a ≈1-ULP Cephes minimax polynomial 8 lanes at a time; gcc/rustc call
scalar `libm` `expf`/`logf`/`tanhf` and **cannot vectorize a loop containing a call** (no `libmvec` on
this mingw toolchain), so they stay serial. `silu` (Llama/SwiGLU) and `gelu` (BERT/GPT-2/ViT, tanh
approximation) are first-class intrinsics, as are `elu`, `leaky_relu`, `softplus` (= `ln(1+eˣ)`), and
`mish` (= `x·tanh(softplus)`) — all composing the shared ≈1-ULP `exp`/`log`. The interpreter marshals
through the *identical* kernel, so the differential oracle stays exact.

This is the change that took the activations from a ~128-bit ~2.5–3.5× win to the ~5–7.5× range —
**roughly double**, because they are compute-bound (~20 flops/element) and the missing 256 bits were
the ceiling. (Composed/scalar `exp`/`erf`/`sin`/`cos` still lower to the inlined ≈1-ULP poly and
auto-vectorize at 128-bit; `erf` gives the exact erf-GELU and `sin`/`cos` give RoPE.)

| kernel | Mercury vs C | notes |
|--------|--------------|-------|
| `exp`  | **~5.9–6.8× faster** | `out=exp(x)`; 256-bit AVX2 poly vs scalar `expf` |
| `log`  | **~4.6–7.5× faster** | `out=log(x)`; 256-bit Cephes poly vs scalar `logf` (libm `logf` timing varies run-to-run) |
| `tanh` | **~4.5–6.1× faster** | exp-based, identical algorithm everywhere; only Mercury vectorizes (256-bit) |
| `gelu` | **~5.0–6.5× faster** | tanh-GELU intrinsic → fused 256-bit kernel; C/Rust the same math, scalar |
| `silu` (swish) | **~3.6–5.8× faster** | `silu()` intrinsic (`x·sigmoid(x)`) → fused 256-bit kernel; C/Rust scalar |
| `softplus` | **~6.6× faster** | `ln(1+eˣ)` (exp+log) → fused 256-bit kernel; also ~4× vs Rust |
| `mish` | **~5.7× faster** | `x·tanh(softplus(x))`, three transcendentals — the heaviest, widest gap; ~6× vs Rust |
| `gelu@parallel` | **~28× faster** | GELU over a large tensor across cores: multicore × 256-bit vs single-thread scalar C |

The full elementwise math suite — `sqrt`/`rsqrt` (hardware), `exp`/`log` (≈1-ULP minimax polys),
`pow` (= `exp(y·log(x))`), `tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`mish`/`selu`/`tanhshrink`/`hardsigmoid`/`hardswish`,
and `fmax`/`fmin` — all vectorize. Every kernel passes the cross-language checksum (the ≈1-ULP poly agrees
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

### bf16 mixed-precision reductions — bf16 storage, f32 accumulate

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

### Single-threaded elementwise & reductions

A recognized streaming map (`out[i] = act(a·x[i] (+ b·y[i]) + c)`) dispatches to the **256-bit AVX2
`mercury_velem_f32`** kernel and a Horner polynomial to **`mercury_vhorner_f32`** — both 4×-unrolled,
both emitting **non-temporal stores** once the working set spills L3 (the store path gcc/rustc will
not emit, skipping read-for-ownership traffic). This turns the former bandwidth-bound *ties* into
wins:

| kernel | Mercury vs C | notes |
|--------|--------------|-------|
| saxpy  | **~1.25–1.45× faster** | `velem` 256-bit + non-temporal store (3-stream, spills L3) — Rust ~1.4× behind too |
| relu   | ≈tie (~1.0×) | 2-stream, L3-resident at N=2²⁰ so stores stay cacheable; the win shows at >L3 (below) |
| poly   | **~1.1–1.2× faster** | `vhorner` 4×-unrolled AVX2 Horner; **ties Rust's autovec** (both ~43 GB/s) |
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

**Fused flash-attention** (online softmax, never materializes the `S×S` scores in HBM — the kernel
that *lost* on CPU, where the tuned GEMM dominates) — warp-per-query-row, `d=64`:

| seq | 512 | 1024 | 2048 | 4096 |
|-----|-----|------|------|------|
| GFLOP/s | 183 | 249 | 301 | **372** |

Arithmetic intensity rises with context, so throughput climbs with `seq`. This is the warp-per-row
baseline (correct, fused, single HBM pass over K/V); it matches a CPU f64 two-pass-softmax reference
to ~9e-4 rel (the `ex2.approx` SFU dominates the error). Shared-memory K/V tiling + tensor-core MMA
for FlashAttention-2 peak is the documented follow-up.

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

**Other op categories** (all emit+execute, tolerance-gated on the 4050): elementwise (saxpy/vadd),
deterministic reductions (sum/dot/max — bit-exact for max, tolerance for the f32 sums), activations
(relu/exp/sigmoid/tanh/silu/gelu via SFU), fused row norms (softmax/LayerNorm/RMSNorm, one warp per
row), and direct conv2d. Reproduce: `cargo test -p mercury_codegen_gpu --features gpu` (correctness;
skips cleanly with no GPU) and `… --release -- --ignored --nocapture` (throughput).

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
- **Transcendentals / activations (exp, log, tanh, sigmoid, GELU, SiLU, ELU, leaky_relu, softplus,
  mish, SELU, tanhshrink, hardsigmoid, hardswish — 15 in all):** **~5–7.5× faster** than C's scalar `libm` — Mercury dispatches the loop to a **256-bit AVX2
  ≈1-ULP poly kernel** (`mercury_vmath_f32`), where gcc/rustc cannot vectorize a loop with an
  `expf`/`logf`/`tanhf` call. This is the transformer/vision activation family and the cleanest
  compute-bound win (it roughly doubled when the kernel moved from the 128-bit vectorizer to 256-bit).
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
- **GPU backend (RTX 4050):** a PTX-emitting, driver-JIT GPU path (no CUDA toolkit) runs every
  transformer op category on-device, tolerance-gated. **Tensor-core GEMM** (fp16/bf16 in, f32
  accumulate) hits **~9–13 TFLOP/s** (clock-dependent) — ~5–6× the f32 path on the same GPU;
  **fp8 (E4M3) via hand-laid `mma.sync` is validated bit-exact**, and its fragment-reuse multi-tile
  kernel is now the **fastest** tensor-core path — ~2.1–2.4× the naive single-tile fp8 and ~1.3–2.3×
  fp16/bf16 in the same run (realizing Ada's ~2× fp8 rate once it's compute-bound). **Fused
  flash-attention** (online softmax, no `S²` scores in HBM) reaches **372 GFLOP/s** at 4 K context;
  and a **whole pre-norm transformer layer runs end-to-end GPU-resident** (matching a CPU f64
  reference to max_rel 2.7e-4, deterministic run-to-run). Numbers are honest for a power-capped 6 GB
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
