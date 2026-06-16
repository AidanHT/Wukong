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
  MSYS2 toolchains. **No LLVM, no MSVC, no AVX-512** (Intel disabled AVX-512 on this consumer part).
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
  loop but never tile, pack, or register-block, so they fall out of cache as the matrices grow.
- **Reduction vectorization.** A naive f32 reduction (`s += x[i]*y[i]`) is one FMA down a single
  dependency chain — latency-bound. Mercury reassociates it across vector lanes × unrolled
  accumulators (the standard BLAS reduction); gcc/rustc keep it strictly serial without `-ffast-math`.
- **Auto-vectorization + fusion** of elementwise loops (incl. branchy ones via if-conversion), `x +
  y*z` → FMA contraction, and adjacent-loop fusion.
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
| any kernel | ~1–3 ms | ~150–370 ms | ~290–340 ms | **~120–230×** |

Cranelift JIT compiling in-process vs spawning a full C/Rust+LLVM toolchain is a 1–2 order-of-
magnitude win, every build. For an ML compiler — where edit/recompile/run iteration dominates
developer time — this is the most robust result of all.

### Matmul `C = A·B` — single-core wins, parallel dominates, and the lead grows with size

GFLOP/s (higher is better), naive `ikj` nest in each language:

| size | Mer 1-core | Mer @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 256³ | 29–55 | 37–60 | 16–24 | 18–23 | **~1.9–2.3×** | **~2.4–2.6×** |
| 512³ | 36–81 | 119–201 | 17–24 | 18–21 | **~2.0–3.3×** | **~6.8–8.3×** |
| 1024³| 41–89 | 145–248 | 11–17 | 12–18 | **~3.9–5.4×** | **~13.6–15×** |

At 1024³ gcc's naive matmul thrashes cache (~11–17 GFLOP/s) while Mercury's packed, tiled kernel
holds ~89 single-core / ~248 parallel — **~5× single-thread, ~15× parallel.**

### `nn.Linear` `C = A·Bᵀ` — Mercury dispatches to GEMM; naive C is latency-bound

| size | Mer 1-core | Mer @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 512² | 38–107 | 105–246 | ~1.9 | ~1.8 | **~20–40×** | **~55–93×** |
| 1024²| 39–99 | 156–293 | ~1.5 | ~1.4 | **~26–44×** | **~100–130×** |

### Single-threaded elementwise & reductions

| kernel | Mercury vs C | notes |
|--------|--------------|-------|
| saxpy  | ≈tie (~1.0×) | memory-bandwidth bound; everyone is at the wall |
| relu   | ≈tie (~1.0×) | memory-bound; vectorized via if-conversion (one load per element) |
| poly   | ≈tie (~1.0×, ±) | compute-bound deg-4 Horner; 128-bit SSE vs gcc's 256-bit AVX |
| fused linear→relu | ~1.2× faster | two source loops Mercury auto-fuses; C/Rust stream the intermediate |
| dot    | **~2.3–2.9× faster** | reduction reassociated to lane accumulators; gcc/rustc stay serial |
| ssd (Σ(x−y)²) | **~2.9–3.7× faster** | same — an L2-loss reduction |

### Auto-parallel runtime — Mercury heavily exceeds idiomatic single-threaded C/Rust

`@parallel` lowers the loop to a multicore dispatch whose per-thread chunk is itself vectorized:

| kernel | Mercury vs single-threaded C |
|--------|------------------------------|
| saxpy@parallel | **~4–5×** |
| poly@parallel  | **~3.5–4×** |
| relu6@parallel | **~10–12×** (nested branch defeats gcc's vectorizer; Mercury if-converts + parallelizes) |

## Honest summary

- **Compile time:** ~120–230× faster than gcc/rustc. Robust every run; the metric that dominates ML
  iteration.
- **Matmul / nn.Linear (the flagship ML kernels):** Mercury **wins single-thread (~2–5×) and
  dominates parallel (~2.4–130×)**, and the lead **grows with matrix size** — the compiler tiles,
  packs, and register-blocks where gcc/rustc leave the naive nest. This is a reversal of the previous
  honest loss (single-core matmul used to be ~3× *behind*).
- **Reductions:** ~2.3–3.7× faster (lane-accumulator reassociation).
- **Auto-parallel:** ~3.5–12× faster than idiomatic single-threaded C across elementwise kernels.
- **Single-thread memory-bound elementwise (saxpy/relu/poly):** a genuine **tie** — these are at the
  DRAM/cache bandwidth wall, where no compiler "heavily exceeds" another. The one structural cause
  left is that the general (non-GEMM) vectorizer emits 128-bit SSE: Cranelift cannot legalize a
  256-bit `f32x8` value (verified, pinned as a tripwire test), so the width-sensitive *general*
  kernels match rather than beat gcc's AVX. The width that matters most — the GEMM family — gets true
  256-bit AVX2/FMA via the runtime microkernel.
- **Safety:** Mercury checks tensor **shapes at compile time** (in the type system), a class of bug
  C/C++/Rust-with-raw-pointers cannot catch.

Where Mercury wins is where a tensor compiler should: compile speed, matmul/GEMM throughput,
automatic parallelism, automatic vectorization (including reductions), automatic fusion, and shape
safety.
