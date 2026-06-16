# Mercury benchmarks — Mercury vs C vs Rust

This is an **honest** cross-language benchmark. For each kernel the *same* computation is written
three ways — Mercury (compiled to native code by the from-scratch Cranelift backend, **no LLVM**),
C (`gcc -O3 -march=native`), and Rust (`rustc -O -C target-cpu=native`) — and all three are timed
through one identical Rust harness over the same buffers. C and Rust are built to shared libraries
and called via their C ABI; Mercury is JIT-compiled in-process.

Reproduce:

```sh
cargo run -p mercury_xbench --release      # CC=gcc by default; set CC to override
```

The harness (`crates/mercury_xbench`) also cross-checks a result checksum across all three
languages, so a miscompiled kernel is caught, not silently mis-measured.

## Test machine & toolchains

- Windows 11, 22 logical CPUs, MSYS2 toolchains. No LLVM, no MSVC.
- `gcc`/`g++` 14.2, `rustc` 1.94, Mercury via Cranelift 0.124 (JIT).
- f32 arrays of N = 2^20 (1,048,576) elements; matmul is 512×512.

## Fairness notes

- **FMA:** Mercury contracts `x + y*z` to a fused multiply-add (one rounding, one `vfmadd`), so gcc
  is given its **default** `-ffp-contract=fast` — the earlier `-ffp-contract=off` was actually
  suppressing C's natural FMA. Both Mercury and gcc-compiled C now fuse. *Idiomatic* Rust does not
  contract unless the author writes `f32::mul_add`, so the Rust column reflects rustc's default (two
  rounded ops). That is a real toolchain-defaults difference, surfaced honestly rather than papered
  over. Mercury and the interpreter agree on the FMA result bit-for-bit (gated in CI).
- The reduction (`dot`) is a strict left-to-right f32 accumulation, so none of the three
  auto-vectorize it (Mercury and C do use a *scalar* FMA per step).
- The single-threaded kernels compare the *same algorithm* in each language (apples to apples).
- The `@parallel` kernels compare Mercury's **automatic** multicore+SIMD lowering against
  *idiomatic, single-threaded* C/Rust — the value proposition of a tensor DSL compiler: you write
  the obvious loop and the compiler parallelizes and vectorizes it. This is called out per row.

## Results

Numbers vary run-to-run on a busy 22-core desktop (the all-core kernels especially); the harness
reports the best of many batches (the least-interfered estimate), and the ranges below span several
runs. Treat them as representative, not exact.

### Compile time — Mercury wins decisively

| kernel        | Mercury | C (gcc)  | Rust     | Mercury speedup |
|---------------|---------|----------|----------|-----------------|
| saxpy         | ~1.7 ms | ~200 ms  | ~220 ms  | **~115×**       |
| relu          | ~0.8 ms | ~150 ms  | ~200 ms  | **~175×**       |
| matmul        | ~1 ms   | ~250 ms  | ~360 ms  | **~250×**       |

Cranelift JIT compiling in-process vs spawning a full C/Rust+LLVM toolchain is a 1–2 order of
magnitude win, every build. For an ML compiler — where edit/recompile/run iteration dominates
developer time — this is the most important and most robust result.

### Single-threaded runtime — competitive (same algorithm)

| kernel | Mercury vs C | notes |
|--------|--------------|-------|
| saxpy  | ~1.0–1.16× slower | memory-bandwidth bound; both ~55–58 GB/s |
| relu   | ~1.0–1.01× (tie) | memory-bound; vectorized via if-conversion |
| poly   | ~1.0–1.03× (tie) | compute-bound; was **5.9× slower** before vectorization+FMA |
| fused linear→relu | ~1.05–1.1× faster | two source loops; Mercury **fuses** them, C/Rust two-pass |
| dot    | ~2.1× slower | strict serial f32 reduction — see below |

`fused linear→relu` writes a linear map to a scratch array then ReLUs it — two loops in every
language. Mercury's compiler fuses them into one pass and keeps the intermediate in registers; the
edge is modest here only because a 4 MiB intermediate still fits in L3 (the win grows when it spills
to RAM). The point is the *automatic* fusion of naively-written ops.

Mercury's vectorizer lifts straight-line elementwise loops to 128-bit SIMD and unrolls 4× so
independent vector chains issue across the core's FP units (recovering AVX-class throughput from SSE
ops), and contracts `a*x + y` to a hardware FMA. With FMA now enabled for both sides, gcc's 256-bit
AVX FMA gives it a slight edge on compute-bound kernels; Mercury's 128-bit FMA + 4× unroll keeps it
within ~1.0–1.16×. Memory-bound kernels are at the bandwidth wall for everyone.

**`dot` is the one clear single-threaded loss (~2.1×).** It is a strict left-to-right f32 reduction:
the accumulator dependency serializes one FMA per element (latency-bound, not throughput-bound), and
Mercury's straight scalar loop sits right at that latency wall while gcc's is tighter. The standard
fix — splitting the sum across several vector-lane accumulators and reducing horizontally at the end
— reassociates the float sum (what every BLAS does); it is the natural next optimization but changes
rounding, so it is intentionally not yet on by default.

### Auto-parallel runtime — Mercury heavily exceeds idiomatic C/Rust

`@parallel` lowers the loop to a multicore (rayon) dispatch whose per-thread chunk is itself
vectorized and unrolled. Versus the idiomatic single-threaded C/Rust kernel:

| kernel          | Mercury        | C (gcc) single-thread | Mercury speedup vs C |
|-----------------|----------------|-----------------------|----------------------|
| saxpy@parallel  | ~126 GB/s      | ~55 GB/s              | **~2.3×**            |
| poly@parallel   | ~101 GB/s      | ~57 GB/s              | **~1.8×**            |
| relu6@parallel  | ~100 GB/s      | ~16 GB/s              | **~6.2×**            |
| matmul 512²     | ~84 GFLOP/s    | ~44 GFLOP/s           | **~1.9×**            |

`relu6 = clamp(x, 0, 6)` is the standout vs C: its nested conditional defeats gcc's auto-vectorizer
(scalar branches, ~16 GB/s), while Mercury if-converts both branches to vector blends and
parallelizes. (Note rustc *does* vectorize relu6 to ~58 GB/s, so vs Rust the parallel edge there is
~1.7×, not 6×.) These rows compare Mercury's one-line `@parallel` annotation against the obvious
single-threaded loop a C/Rust programmer writes.

## Honest summary

- **Compile time:** Mercury is ~115–250× faster to compile. Robust, every run, and the metric that
  dominates real ML iteration time.
- **Single-threaded runtime:** competitive on elementwise kernels (within ~1.0–1.16× of C on the same
  algorithm; *faster* on auto-fused linear→relu) — strong for a from-scratch, LLVM-free backend. Two
  honest losses: the strict serial `dot` reduction (~2.1×) and single-core matmul (~3.3×).
- **Auto-parallel runtime:** Mercury's automatic SIMD+multicore lowering beats idiomatic
  single-threaded C/Rust by ~1.8–6.2× on the kernels here, and parallel matmul (~84 GFLOP/s) beats
  single-threaded C/Rust matmul (~44–46 GFLOP/s) by ~1.9×.
- **Safety:** Mercury checks tensor **shapes at compile time** (in the type system), catching a class
  of errors C/C++/Rust-with-raw-pointers cannot.

Where Mercury does *not* beat C/C++ today: peak **single-thread** throughput on compute-bound kernels
(matmul single-core ~3.3× behind; `dot` ~2.1×). Two causes, both deliberate tradeoffs: Cranelift
emits 128-bit SSE (no 256-bit AVX legalization) and does less instruction scheduling than gcc/LLVM
`-O3`; and strict-reduction semantics keep `dot` serial. Closing the SIMD gap fully would need an
AVX-capable backend, which the project trades away for zero-dependency builds and ~100× faster
compiles. Mercury's wins are where a tensor compiler should win: compile speed, automatic
parallelism, automatic fusion, and shape safety.
