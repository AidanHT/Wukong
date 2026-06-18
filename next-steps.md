# Next steps — scoping a GPU backend for Mercury

> Status: scoping note, not a commitment. Written after the CPU work reached diminishing returns
> (the GEMM microkernel already beats C/C++/Rust and dominates runtime; remaining single-thread CPU
> wins are marginal). This documents *why* a GPU backend is the next real frontier, the realistic
> options given Mercury's constraints, and a phased plan. See `BENCHMARKS.md` for the current CPU
> standing and `docs/internals.md` for the pipeline this would extend.

## Why GPU is the next frontier

The honest CPU picture (measured mid-2026, vs gcc/rustc `-O3 -march=native`): Mercury already wins on
every metric that matters — compile time **84–525×**; matmul **3–3.8× single / 9–16× parallel**
(455 GFLOP/s @1024³), nn.Linear **23× single / up to ~102× parallel**; conv **~6×**; vectorized
transcendentals **5–7.5× single / ~19× parallel**; reductions **2.7× single / 7.9–8.6× parallel**;
elementwise an honest DRAM-bandwidth tie single-thread but **2.2–8× under `@parallel`**. The tuned
AVX2/FMA GEMM microkernel is at ~90% of the single-core roofline. The CPU path is now
*comprehensively* ahead, and three findings cap further CPU work:

- **The last `@parallel` gap is closed.** Reductions were the one kernel class with no multicore path
  (the function-level `@parallel` model can't express a carried-scalar reduction, and it races). They
  now dispatch to a deterministic multicore reduction kernel (`mercury_sreduce_f32_parallel` —
  dot/ssd/sum/sumsq), taking dot from ~2.7× to ~7.9× and ssd to ~8.6× vs C. That was the last
  *substantial* single-machine CPU win; what's left is marginal.
- **Flash-attention is not a single-thread CPU win** (measured ~2× *slower* than the GEMM-dispatch
  path; preserved on branch `experiment/flash-attention-cpu`, see the memory note). The materialized
  attention is compute-bound on the tuned GEMM, so fusion only saves S² scores traffic that isn't the
  bottleneck. Flash-attention's decisive wins are GPU-shaped: HBM-bandwidth- and kernel-launch-bound.
- The remaining CPU items (epilogue fusion beyond Linear, parallel-GEMM tuning, a 256-bit *single-core*
  reduction) are single-digit-percent or hardware-throttle-limited on this 6P+8E hybrid.

The big ML wins — large-batch training/inference throughput, long-context attention, mixed precision
that actually pays (tensor cores) — live on the GPU. That is where a tensor-kernel compiler earns its
keep, and where Mercury's compile-time shape safety + op recognition (matmul→GEMM, conv→im2col,
fusion) would translate into device kernels that rival hand-written CUDA.

## The two hard constraints (and what they imply)

1. **No GPU on the dev box, and no LLVM toolchain** (same physical fact that put the CPU path on
   Cranelift instead of LLVM). So:
   - The GPU path **cannot run or be differentially tested on this machine.** It is a codegen-only
     target here, validated structurally (emitted-kernel snapshot/lint) locally and *executed* on a
     CI runner or cloud box with a real GPU. This mirrors how the textual-LLVM path is emit-only here.
   - PTX-via-LLVM (NVPTX) is off the table for the same reason CPU LLVM is. Generation must avoid the
     LLVM dependency.
2. **The differential gate is the project's correctness invariant.** On CPU it holds bit-for-bit
   because both backends call the *identical* runtime kernel. **That trick does not cross the CPU/GPU
   boundary** — a GPU kernel reassociates differently than the interpreter. So GPU correctness becomes
   a **tolerance-based** differential (native-CPU/interpreter vs GPU, within an f32 epsilon that
   scales with the reduction length), exactly like the existing `sgemm_matches_naive` runtime test —
   not a bit-exact stdout comparison. This must be designed in from day one, not bolted on.

## Backend options

| Option | How | Pros | Cons |
|---|---|---|---|
| **CUDA-C → `nvcc` at runtime** (recommended first) | Emit CUDA C from MIR, shell out to `nvcc` to a cubin/PTX, launch via the driver API (`cudarc`/`cust`) | Mirrors the existing "shell to `clang`" emit path; `nvcc` does the device optimization/regalloc; fastest route to a *running* kernel; reuses our op recognizers as CUDA templates | NVIDIA-only; needs `nvcc` on the runner; textual codegen |
| **PTX directly** | MIR → PTX text | No `nvcc` dep; full control | Must do our own regalloc/scheduling for PTX — very high effort; reinvents what `nvcc`/`ptxas` do |
| **SPIR-V via `wgpu`/Vulkan** | MIR → SPIR-V, run through `wgpu` | Vendor-neutral (NVIDIA/AMD/Intel/Apple), pure-Rust host, runs almost anywhere incl. this box's iGPU | Compute-only ergonomics; no tensor cores; SPIR-V codegen is real work; perf ceiling below CUDA on NV |
| **`rust-gpu` / `cudarc` kernels** | Hand-write kernels in Rust/CUDA, dispatch from recognizers | Leverages existing crates | Less "compiler", more "library"; still NVIDIA for the fast path |

**Recommendation:** start with **CUDA-C → `nvcc`** for the fast path (it's the shortest path to a
device kernel that beats a CPU baseline and reuses our op-recognition as kernel templates), and keep
a **SPIR-V/`wgpu`** path in mind as the portable, runs-on-the-iGPU validation target (it can actually
execute on this box, which is valuable for catching codegen bugs without a discrete GPU). Decide
between them after Phase 0.

## How it slots into the existing pipeline

The seam already exists: `mercury_backend::Backend` is the MIR-to-execution trait the interpreter and
Cranelift implement. A GPU backend is a third implementer plus a host-side launcher:

```
MIR (Low)
  ├─ mercury_interp            (oracle, CPU)
  ├─ mercury_codegen_cranelift (fast CPU, today's default)
  └─ mercury_codegen_gpu       (NEW: MIR → device kernel + host launch stub)
        ├─ kernelize: pick the parallel axis (the @parallel range / matmul grid) → grid/block
        ├─ emit:      MIR ops → CUDA C (scalars, loads, the recognized GEMM/conv/attention templates)
        └─ launch:    host stub allocates device buffers, H2D copy, launch, D2H copy
```

Key reuse: the **recognizers already identify the tensor ops** (matmul→GEMM, batched→per-head,
conv→im2col, the bias/act epilogue, softmax/transcendental loops). On CPU they emit a microkernel
call; on GPU they emit a *device-kernel template* (tiled GEMM with shared-memory staging; fused
attention that — unlike on CPU — genuinely wins by never spilling S² scores to HBM). The shape types
give us static tile/grid bounds for free.

`@parallel` is the natural first kernelization axis: its range maps directly to a 1-D grid, and the
body is already proven data-parallel (the interpreter runs it sequentially as the oracle).

## Correctness strategy

- **Tolerance differential, GPU vs interpreter**, run on a GPU CI runner: per kernel, compare GPU
  output to the interpreter oracle within `tol = c · sqrt(K) · eps` (the `sgemm_matches_naive`
  pattern). This *replaces* the bit-exact stdout gate for the GPU target only — document it as a
  third sanctioned reassociation exception alongside reductions and the shared CPU kernels.
- **Determinism within the GPU path**: fix block/grid sizes and reduction order per kernel so GPU runs
  are reproducible (no atomics-with-nondeterministic-order in reductions unless explicitly tolerated).
  The CPU `mercury_sreduce_f32_parallel` is the precedent: fixed-size chunks independent of core count
  + an ascending partial combine make its result identical regardless of how many lanes ran it — the
  GPU block-reduce should fix its grid/combine order the same way rather than relying on atomic add.
- **Emit-only CI here**: snapshot the generated CUDA/SPIR-V and lint it (compiles under `nvcc
  --ptx`/`spirv-val`) without executing, so this box still guards codegen regressions.

## Phased plan (each phase: a running, measured kernel before the next)

- **Phase 0 — spike & decide (small):** stand up `mercury_codegen_gpu` behind a feature flag; emit a
  trivial elementwise kernel (saxpy) as CUDA C; compile with `nvcc`; launch via `cudarc` on a cloud
  GPU; tolerance-check vs the interpreter. Decide CUDA-C vs SPIR-V from this spike. Deliverable: one
  green GPU kernel + the differential harness.
- **Phase 1 — elementwise + reductions:** map `@parallel` ranges and the vectorizer's elementwise/
  reduction shapes (saxpy, relu, dot, softmax row-ops, the transcendental polynomials) to device
  kernels. These are bandwidth-bound — the GPU win over CPU is large and easy to show honestly. The
  CPU reduction recognizer (`match_reduction_kernel` → `mercury_sreduce_f32_parallel`) already
  identifies dot/ssd/sum and proves them data-parallel; the GPU emits a block-reduce template instead
  of the CPU chunk kernel, reusing the same recognizer and the same fixed-chunk determinism idea.
- **Phase 2 — tiled GEMM:** the matmul recognizer emits a shared-memory-tiled GEMM (register-blocked,
  the GPU analog of the AVX2 microkernel). Target f32 first; this is the headline throughput kernel.
- **Phase 3 — fused attention (the real payoff):** the attention pattern that *lost* on CPU wins here
  — a fused flash-attention kernel that keeps tiles in shared memory and never materializes S² in HBM.
  This is the single most valuable GPU kernel for modern transformers.
- **Phase 4 — mixed precision & tensor cores:** bf16/fp16 inputs with f32 accumulate via WMMA/tensor
  cores — where bf16 finally pays a *FLOP/s* win (it didn't on AVX2), with the shape types guarding
  the tile contracts.
- **Phase 5 — autotuning & multi-GPU:** tile-size search per shape; later, multi-device.

## Effort & risk (honest)

- **Effort:** large — a new backend crate, a host runtime (alloc/copy/launch), a GPU CI runner, and
  per-op device templates. Phase 0–1 is a few focused weeks; a *competitive* GEMM+attention (Phase
  2–3) is the bulk of the work and the part that must be benchmarked against cuBLAS/FlashAttention to
  claim a win honestly (beating *naive* CUDA is easy; beating cuBLAS is the real bar and may not be the
  goal — matching it while keeping shape safety + fusion is already valuable).
- **Risks:** no local GPU (CI dependency); the bit-exact gate must relax to tolerance for GPU (design
  carefully); textual CUDA codegen is verbose; tensor-core kernels are intricate. None are blockers,
  but Phase 0 must de-risk the toolchain (nvcc/driver availability on CI) before committing.

## Recommended first milestone

A single command — `mercuryc --backend=gpu --run examples/saxpy_array.mer` on a GPU runner — that
emits CUDA C, compiles with `nvcc`, launches, and passes a tolerance differential against the
interpreter. That proves the whole seam (kernelize → emit → compile → launch → verify) end-to-end on
the simplest kernel, and everything after is filling in op templates.
