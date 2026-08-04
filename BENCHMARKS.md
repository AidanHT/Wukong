# Wukong benchmarks — Wukong vs C, C++, and Rust

An **honest** cross-language benchmark. For each kernel the *same* computation is written several ways
— Wukong (compiled to native code by the from-scratch backend, **no LLVM**), C (`gcc -O3
-march=native`), and Rust (`rustc -C opt-level=3 -C target-cpu=native`) — and all are timed through one
identical Rust harness over the same buffers. C and Rust are built to shared libraries and called via
their C ABI; Wukong is JIT-compiled in-process. The harness cross-checks a result checksum across every
language, so a miscompiled kernel is caught, not silently mis-measured. The elementwise/reduction
battery additionally times **C++ (`g++ -O3 -march=native -ffp-contract=fast`)**; g++ and gcc share a
backend, so on identical kernel code C++ tracks C to within a few percent — the C ratios below stand
for C++ too (each is cross-checked against C++ as well, and "vs C++" is printed alongside "vs C").

Two **additional C peer columns** normalize the two disclosed baseline asymmetries (see Fairness
notes): **C(fast)** — the same C source recompiled `-O3 -march=native -ffast-math`, so gcc may
reassociate/vectorize float reductions the way Wukong's recognized kernels do — is printed for the
reduction-bearing benches (dot/ssd + their `@parallel` forms, matmul, `nn.Linear`, FFN, the column
reductions/statistics, the norms and their backwards, cross-entropy, row losses); and **C(omp)** —
the same C kernel under `#pragma omp parallel for` (with `reduction` clauses where the loop is a
reduction), compiled `-fopenmp` — is printed for the `@parallel` rows (plus the parallel GEMM peers
in matmul/linear/transpose/colsum) after a one-time runtime probe verifies OpenMP DLLs actually load
and run multi-threaded here. Both are cross-checked at a **looser** magnitude-normalized `1e-2`
tolerance (reassociation legitimately changes results); a column that can't meet it is dropped with
a printed note rather than failing the bench. They are reported *alongside* — never replacing — the
honest-default-flags C column.

Reproduce:

```sh
cargo run -p wukong_xbench --release      # CC=gcc by default; set CC to override
```

> ## ⚠ Peer-strength correction — 2026-08-04
>
> **Several figures previously published in this document were measured against a miswritten C/Rust
> peer, and were wrong by up to 63×.** A deliberately adversarial audit of every peer kernel in
> `crates/wukong_xbench` found four systematic defects, all of which happened to flatter Wukong:
>
> 1. **No `restrict`, anywhere.** Not one of the 43 generated C kernels declared its output buffer
>    non-overlapping with its inputs, so gcc had to assume `out[i] = f(x[i])` might clobber a later
>    `x[j]` and could not vectorize or reorder any nested kernel. Wukong's tensor parameters carry
>    non-overlap in the type system. The comparison was a no-alias compiler against a may-alias one.
> 2. **The column reductions were written column-outer** — `for j { s=0; for i { s += x[i*N+j] } }`
>    — the single worst loop order for a row-major axis-0 reduction, and the *only* order measured.
>    (This document already conceded the row-outer spelling auto-vectorizes, and published the
>    column-outer multiple anyway.)
> 3. **The weight-gradient GEMM peer read BOTH operands column-strided** (`ijk` with `a[k*NS+i] *
>    b[k*NS+j]`), rather than the natural `kij` nest that hoists `a[k*NS+i]`.
> 4. **The transpose peer was unblocked** while Wukong's kernel is 32×32 cache-blocked — the bench
>    measured loop tiling, which Wukong did not invent, not codegen.
>
> All four are fixed. Every peer now carries `__restrict__`, walks memory in the natural order, and
> gets the same algorithmic opportunity as Wukong's kernel; ten reduction-bearing benches that had no
> `C(fast)` [-ffast-math] column now print one. The corrected numbers are throughout this document,
> and the full before/after table is in **[Corrected peer measurements](#corrected-peer-measurements-2026-08-04)**.
> Four regression tests in `crates/wukong_xbench/src/main.rs` now pin the peers so this cannot recur
> silently.
>
> **The headline effect.** The "strided column reduction" family (~29–50× 1-core) is now **a tie at
> best and a 1.8× LOSS at worst** (15 of the 16 measured rows are losses; the >L3 shape is a
> consistent ~1.65× loss across two independent rounds).
> The weight-gradient GEMM (~128× 1-core / ~445× `@parallel`) is now **2.6–9.2× / 3.1–13.9×**.
> The transpose (~1.5×) is now a **tie**. The bf16 reduction family, once given the `-ffast-math`
> column its own rule required, is a **1.1–1.8× loss**. Wins that did *not* move — GEMM, `nn.Linear`,
> the transcendentals, row-argmax, the fused norms — are exactly the ones that were real.

## Test machine & toolchains

- Windows 11, Intel Core Ultra 7 155H (Meteor Lake: 6 P-cores + 8 E-cores + 2 LP-E, 22 threads),
  MSYS2 toolchains. **No LLVM, no MSVC, no AVX-512** (Intel disabled AVX-512 on this consumer part);
  **AVX2/FMA + AVX-VNNI** are present (the int8 GEMM uses `vpdpbusd`, and so does gcc `-march=native`).
- `gcc` 14.2, `rustc` 1.94, Wukong via Cranelift 0.124 (JIT) + AVX2/FMA runtime microkernels.
- **Library/framework peers** (their versions decide how hard the bar is, so they are recorded):
  oneMKL via `mkl_rt.2.dll` (Anaconda, 16 threads max); **PyTorch 2.12.1+cpu** (16 threads) as of the
  2026-07-30 round — note the harness's `detect_torch` selects the *newest* torch among `python` on
  `PATH` and `tools/torch-venv`, so the peer can change without the benchmark changing; the
  `torch.compile` columns are pinned to the ATEN/MKL GEMM backend because Inductor's CPP FP32 GEMM
  template is broken on Windows/MSVC (disclosed in-run by the harness).
- Elementwise/reduction kernels: f32 arrays of N = 2²⁰ (1,048,576). Matmul: 256/512/1024/**2048**
  square by default (4096 additionally under `XBENCH_HUGE`; `XBENCH_MATMUL_SIZES` narrows the sweep);
  `nn.Linear` and the fused FFN: 512/1024; TN weight-gradient: 256/512/1024.

**Variance.** This is a busy hybrid laptop; the all-core and matmul numbers swing run-to-run (P-core
boost, E-core scheduling, thermals). The harness reports the best of many batches (the least-
interfered estimate); the ranges below span several runs. Treat them as representative, not exact —
but the *ratios* (who wins, by roughly how much) are stable.

**Three instrument axes, not one** (learned the hard way on 2026-07-30; each has bitten a round):

1. **Power state** — battery / AC+charging / AC+full are three different machines. The harness reads
   it and labels every run; on battery the measured single-core roofline fell to **105 GFLOP/s** vs
   **131–142** on AC, so *nothing* is reportable there, not even single-core. While charging, all-core
   is capped ~25% (GEMM `@parallel` read **52% of MKL-all @512³** charging vs **97%** at AC+full —
   same binary, same shape) while single-core is unaffected.
2. **Thermal / recent-load history** — independent of (1). The measured roofline is *instantaneous
   boost headroom*: ~138–142 after an idle period, but 91–111 across ten back-to-back probes, and 78
   when probed immediately after a heavy elementwise battery. So the roofline reading is a valid
   *round-validity gate only if the probe starts from a comparable idle state*, and a `% of measured
   roofline` figure inherits that denominator's noise.
3. **Hybrid core placement** — nothing pins the single-threaded work to a P-core, so an unpinned
   single-core row can land on an E-core and read ~2× slow at an otherwise perfect power state (see
   the 2026-07-30 note in the model section: 277.8 ms vs 617.6 ms for identical work). **Single-core
   rows are comparable within a round, not across rounds**, and any ratio whose denominator is a
   single-core column (notably `@parallel` scaling) inherits the problem.

The practical rule this yields, and the one the tables below follow: **report same-run adjacent
peer ratios**; treat absolute GFLOP/s, `% of roofline`, and scaling factors as round-local.

## Headline results

The numbers that matter most are the ones where the *baseline is hard* — so the win reflects real
quality, not just beating textbook code:

- **GEMM is near the silicon limit *and* ahead of a tuned library.** The single-core f32 GEMM holds
  **~104–117 GFLOP/s ≈ 90% of one P-core's AVX2-FMA roofline**, and is **1.1–1.3× faster than the
  tuned `matrixmultiply` Rust crate** — a real hand-optimized peer, not a strawman (`117 vs 89`,
  `115 vs 91`, `104 vs 96` GFLOP/s across 256²–1024²) — with **no LLVM**.
- **Compile time leads decisively, every build.** The headline *compiler-to-compiler* figure is the
  **both-subprocess `compile-vs`** comparison (`cargo run -p wukong_bench --release -- compile-vs`:
  `wukongc --emit=obj -O2` vs `gcc/g++/rustc` compiling bare equivalent kernels to objects, all as
  subprocesses, best-of-N same-run). The xbench per-kernel "compile (ms)" figure — **~100–680×
  faster than C/Rust** (latest full-board geomean ~306×; ~0.3–1.5 ms vs ~125–245 ms) — measures
  **in-process JIT/embedding latency** (Wukong's front-end + Cranelift JIT in-process vs *spawning*
  a toolchain), the right number for JIT-style embedding but not a process-to-process comparison.
- **Geomean across the elementwise/reduction battery: 3.8–4.4× faster than C** (3.7–4.3× vs C++) —
  re-measured 2026-07-30 and now *derived* rather than provisional: it is the geometric mean the
  harness prints over the **29 single-threaded rows**, with the 9 `@parallel` rows kept in a separate
  accumulator (they are all-core-vs-1-thread and must never be pooled into a figure printed under a
  single-threaded heading). Two same-day rounds read **3.76×** and **4.39×**. Reported as a range
  deliberately: the two rounds' measured single-core rooflines were 131 and 78 GFLOP/s, and the
  geomean moved *with* that — so even a ratio of two same-run single-threaded columns is not fully
  clock-invariant here, because Wukong's vectorized kernels and gcc's scalar ones lose throughput at
  different rates as the machine heats. The previously recorded **4.86×** (marked "derivation not
  shown") is therefore not refuted by these rounds; it is plausibly the same quantity measured on a
  cooler machine, and the honest form is the range plus its instrument state.
- **End-to-end: a 12-layer GPT-2-class transformer stack (768/12/3072, S=128/512) runs ~19–21×
  faster than idiomatic C and 3.6–4.9× faster than `-ffast-math` C single-core** (three independent
  same-day rounds — see the [End-to-end model](#end-to-end-model--a-12-layer-gpt-2-class-transformer-stack-cpu-inference)
  section), and **beats compiled PyTorch** (`torch.compile` Inductor max-autotune, fullgraph, warmed):
  single-thread it beats compiled-torch-1T at **both** shapes, and — since the `@parallel` head-loop
  region shipped (2026-07-10) — it runs **1.39–1.81× faster than all-threads compiled torch at S=512
  and 1.07–1.43× faster at S=128** (three rounds; it also beats torch's strongest eager config
  1.07–1.37× @S=512); the whole block compiles in ~3–12 ms vs gcc's ~0.4–0.9 s and the forward is
  interpreter-gated bit-for-bit at a reduced config.

The largest domain-lowering blowouts (each is multicore-vs-1-core, or vs idiomatic scalar source
where gcc/rustc won't vectorize — disclosed per section, never a rigged baseline). **Corrected
2026-08-04** against the fixed peers; the struck values are what this table said before, and the
[correction section](#corrected-peer-measurements-2026-08-04) shows every ratio that moved:

| What | 1-core vs C | `@parallel` vs C | was |
|---|---|---|---|
| Weight-gradient GEMM `dW=Aᵀ·B` | **~2.6–9.2×** | **~3.1–13.9×** | ~~~128× / ~445×~~ (peer read both operands column-strided) |
| `nn.Linear` `C=A·Bᵀ` | ~19–26× | up to **~104×** | unchanged |
| bf16 `nn.Linear` | ~23–28× (**~4.6–5.6× vs C(fast)**) | ~50–91× | ~24–25×; the C(fast) column is new |
| Fused FFN `silu(A·Bᵀ)` (the Dense layer) | ~24–26× | ~48–95× | unchanged |
| RoPE (rotary embedding) | ~29–54× | **~146–156×** | unchanged |
| Reductions / transcendentals / **row**-argmax | ~2.5–9× | ~9–26× | unchanged |
| ~~Strided column reductions (sum/max/absmax)~~ | **tie → 1.8× SLOWER** | ~1.0–1.4× | ~~~29–50× / ~37–107×~~ — **the peer was written column-outer; see below** |

The last row is the single largest correction in this document: what was published as a ~29–50×
single-core win is, against a C peer written the way a competent programmer writes an axis-0
reduction, a tie at best and a 1.8× **loss** at worst. It is kept in the table, struck through,
rather than quietly deleted.

**GPU backend** (`--features gpu`, mobile RTX 4050, same-run clock-invariant ratios — full section
[below](#gpu-backend-nvidia-rtx-4050-laptop-sm_89)): fp16 tensor-core GEMM reaches **cuBLAS parity
(~101%) at ≤1024³**, and the ldmatrix+XOR-swizzle workhorse lifts the large regime to **~87–90% at
2048³** (past the prior 77% PTX ceiling) and, with the shipped **v2cs streaming epilogue**, **76.8%
of cuBLAS-f16 / 80.4% of the honest f32-out peer at 4096³**;
**95.7% of the 192 GB/s HBM hardware peak** on the saxpy triad; the fused flash-attention **beats the genuinely-fused cuDNN + cutlass mem-efficient fMHA** in the
causal-D64-S=512 (**1.03–1.16×**), fused-RoPE-S≤512 (**1.8–5.7×**), and D=128-ldmatrix-S≤1024 (**1.11–1.20×**
vs cutlass-efficient) regimes, is competitive through S≤1024, and **trails cuDNN at long context S≥2048
(0.37–0.66×)** — and is **3.6–5× a cuBLAS *unfused* attention chain** (205–738× naive CUDA-C, the older,
weaker library bar); int8 tensor-core GEMM is **~180–237× naive CUDA-C** and reaches **96–105% of cuBLAS
int8 IMMA @2048³ (beating it), 86–88% @1024³, ~70% @4096³** (the 64×64 warp-tile, ldmatrix + XOR-swizzle),
with the **fused int8 GEMM+dequant 1.1–2.2× the cuBLAS chain**; the **fused GEMM+activation beats the cuBLAS GEMM+act chain
1.18–2.41× at ≤1024³** (the fusion cuBLAS structurally can't express; it washes to a slight loss by 2048³ — 0.94×); and GPU compile is **0.76 ms cold vs
Triton's 30–120 s** (~4×10⁴–1.6×10⁵×).

Every number is gated bit-for-bit (CPU) or to a `c·√K·ε` tolerance (GPU) against the interpreter
oracle — the wins are correct, not miscompiles.

### Close-the-NVIDIA-gap campaign (vs the vendor libraries)

A six-slice campaign (2026-06-25) measured Wukong against the hand-tuned vendor libraries it had not
yet been compared to — oneMKL, cuBLAS, cuBLAS IMMA / cuBLASLt, cuDNN, and a genuinely *fused*
FlashAttention-class peer — and closed or beat them where the measurement is honest. Every standing is
**same-run** (the 4050's ~7× clock swing makes absolute GFLOP/s meaningless; only the ratio and the
win/lose *direction* are stable across ≥3 re-runs). Full per-slice findings, including the **measured
negative results**, live in [`prompts/results/`](prompts/results/).

| Slice | Peer | Standing | Honest residual gap |
|---|---|---|---|
| CPU GEMM | oneMKL | 1-core **~80–104% of MKL-1c across sizes**, at/above parity (≥100%) at 2048³/4096³ (2026-07-10 C-tile microkernel prefetch closed the writeback-miss tail; the `WUKONG_GEMM_PF_C=0` kill-switch reproduces the old 86–88%); `@parallel` via the **size-keyed 2D block-parallel dispatch** (per-block packing mid/large — it beat the shared-cooperative-pack design in both ABBA orderings — shared-pack small band above the 2²³-MAC gate): **~98–99% @512³, ~104% @1024³** of MKL-all, **91% @2048³ / 93% @4096³ vs a healthy peer** (176–186% vs MKL's degraded rounds), **~94–110% @256³** (engaged at 1.7–2× over serial; was deliberately serial at ~39–52%) | large-size `@parallel` a touch behind MKL-all's parallel grain @2048³ (91%; 512³–1024³ at/above parity — MKL-all swings ~1.4–2× with power state, ratios same-run, reported as ranges); Wukong's parallel path holds flat wall-time across power states while peers ride them (sync/serial-fraction bound — head-loop parallelism is the open lever) |
| CPU vmath | oneMKL VML | **tanh 2.7–2.9× FASTER**; exp **1.23–1.45×** and log **~1.25×** slower across thermal states (2026-07-09 re-measure; the single-session "exp 1.05× faster / log 1.14×" readings did not reproduce — ranges are the honest claim) — still down from the former ~1.7–2× loss via the 8-bucket in-register-LUT rewrites (exp ~1.3 ULP, log ≤6.9e-7, exhaustively swept) | residual exp/log gap is algorithmic (VML's cheaper ~0.5-ULP core); a bit-identical ldexp exp-tail restructure measured a ~10–15% LOSS in both thermal states and was reverted — port rebalancing cannot beat a clock throttle |
| CPU end-to-end model (12-layer GPT-2-class) | PyTorch CPU compiled (`torch.compile` max-autotune, fullgraph) + eager + gcc C | **~19–21× C(gcc), 3.6–4.9× C(-ffast-math)** 1-core; vs compiled-torch-1T it wins at **both** shapes; multicore, since the `@parallel` head-loop region (2026-07-10): **1.39–1.81× FASTER than all-threads compiled torch @S=512 and 1.07–1.43× @S=128** (three rounds, roofline-validated; also 1.07–1.37× vs torch's strongest eager config @S=512), scaling to **default 4.44× / best 5.22× (T=16 4.70×)**, **~61–69× C(gcc)** multicore; interp gate bit-exact three ways (interp == native == @parallel), serial==@parallel bit-exact, outputs cross-checked <2e-6 vs C and torch | residual scaling headroom vs 16 physical cores is the parallel-GEMM grain at 2048³/4096³ (91–93% of MKL-all; 512³–1024³ at/above parity) — the open lever |
| fp16/bf16 GEMM | cuBLAS | **~101%** ≤1024³, **~87–90% of cuBLAS-f16** @2048³; @4096³ the **v2cs streaming epilogue** shipped (+2.7%): **76.8% of cuBLAS-f16 / 80.4% of the honest f32-out peer** (new peer column — the f16-out peer hides ~half of Wukong's f32 C-write traffic) | 4096³ residual is SASS-level; 3-stage pipe, raster re-tunes, launch-bounds, and 2048³ epilogue variants all measured losses (kept bench-only) |
| int8/fp8 GEMM | cuBLAS IMMA / cuBLASLt | int8 **96–105%** @2048³ (**beats IMMA**), 86–88% @1024³; **fused GEMM+dequant 1.1–2.2×** the cuBLAS chain; fp8 82–151% of cuBLASLt | int8 ~70% @4096³ (HBM-bound) |
| Attention | cuDNN / cutlass fused fMHA | fused-RoPE **1.8–5.7×** & causal D=64 **1.03–1.16×** (beats both) @S≤512; D=128 ldmatrix beats cutlass @S≤1024 — the production D=128 dispatch; **FA2-style warp-specialized kernels (2-warp named-barrier anti-phase + 3-stage ring) built, gated, and default-routed at S≥4096 where they win 4–6%** (clock-cancelled median-of-9; tie @2048, lose @≤1024 — pins keep the losing regimes unroutable) | non-causal long-S (≥2048) 0.37–0.66× cuDNN stands — warp specialization was the last scheduling lever and moved only the S=4096 point; the loss is structural (SFU/serial-softmax on 20 SMs) |
| Conv | cuDNN-9 | 1×1 **3.5–5.6×**, 3×3/5×5 deep-channel **0.93–1.21×**, Winograd F(4×4,3×3) **1.2–2.2×** over implicit-GEMM | Winograd loses at low channel count; depthwise/dilated not yet covered |
| Serving | (no vLLM/TRT-LLM installable — vs Wukong's own eager + an honest static-batching peer) | **85.6× continuous-batching goodput** vs fill=1 @Bcap=256 (2026-07-10 Bcap parameterization; 33.1k tok/s full-fill, same-clock interleaved; the old 39× was the Bcap=64 ceiling, reproduced at 38.2×), graph-driven scheduler drain 27.5k tok/s = **1.14–1.27× vs static batching** (bit-identical outputs; the batching-vs-scheduling decomposition disclosed); decode CUDA graph 1.07–1.35×; int8 KV **3.88× smaller cache than f32** (1.94× than f16), the opt-in int8-KV storage path | multi-GPU collective (NCCL) unmeasured on one device |

## Scoreboard

Headline ratios vs the **idiomatic** C/Rust baseline (`gcc -O3 -march=native` / `rustc -C opt-level=3
-C target-cpu=native`), single-core and `@parallel`. Ranges span several runs and shapes; see the linked
sections for the full tables, methodology, and caveats. This is an *honest* board — the ties and the
modest wins are listed alongside the blowouts, and every row is gated bit-for-bit against the
interpreter oracle (the cross-language check is exact for the integer/permutation kernels, a tight
tolerance for the reassociated-float ones).

| Kernel family | 1-core vs C | `@parallel` vs C | Why Wukong wins (what gcc/rustc won't do) |
|---|---|---|---|
| **f32 GEMM** (matmul / `nn.Linear`) | ~3–3.6× | up to ~100× | register-block + cache-tile + pack; they vectorize the inner loop but never tile |
| **bf16/f16 GEMM** | ~23–28× (idiomatic) / **~4.6–5.6× vs C(fast)** | ~50–91× | lossless widen-prepass → the tuned f32 microkernel |
| **Fused FFN** (`silu(A·Bᵀ)`, the Dense layer) | ~24–26× | ~48–95× | matmul + activation folded into one C-write; C re-streams C through a separate scalar-`expf` pass |
| **TN weight-gradient** (`dW=dYᵀ·X`) | **~2.6–9.2×** | ~3.1–13.9× | transpose-prepass + the tiled kernel, vs the natural `kij` nest gcc vectorizes but does not tile *(corrected 2026-08-04: was ~128×/~445× against a peer that read both operands column-strided)* |
| **int8 `nn.Linear`** (`vpdpbusd`) | ~1.5–2.5× | ~4.6–14.7× | 2×4 register tile halves B traffic (vs gcc's own `vpdpbusd`) |
| **Column reductions** (sum/max/min/absmax/mean/L2/RMS) | **tie → 1.8× SLOWER** | ~1.0–1.4× | *nothing* — gcc auto-vectorizes the natural row-outer nest and matches or beats the kernel. The old ~29–50× was the peer's column-outer loop order *(corrected 2026-08-04)* |
| **Transpose** (f32) | ≈tie (1.00–1.03×) | ~4.8–7.1× | *nothing single-core* once the peer is blocked too; `@parallel` adds cross-core bandwidth *(corrected 2026-08-04: was ~1.5× against an unblocked peer)* |
| **Fused norms** (softmax/LN/RMS) | ~1.9–6.6× | memory-bound | single-pass fusion + 256-bit `exp`; their float reductions stay sequential |
| **Reductions** (dot / ssd) | ~2.6–2.9× | ~8–26× | lane accumulators; their reduction is a serial `vaddss` chain |
| **Activations** (35-op `vmath`) | ~2–11.5× | ~28× | hand-AVX2 256-bit transcendentals vs scalar libm |
| **Activation backward** (6: silu/gelu/sigmoid/tanh/elu/softplus grad) | **~3–12×** | **~5.5–25×** | the derivative folds a sigmoid/tanh/exp (`expf`) C/Rust keep scalar — the forward lever, applied to training |
| **Softmax backward** (`y·(dy−Σy·dy)`) | ~1.0–2.0× | ~4.5–6.1× | vectorizes the per-row dot's accumulation (modest — they vectorize the apply) |
| **Norm backward** (RMSNorm / LayerNorm grad) | ~2.6–4.0× | ~11–20× | the per-row coupling-term reductions (`Σdy·x̂` etc.) gcc keeps scalar (measured in tests/run; per-size table not reproduced here) |
| **Cross-entropy** (softmax xent fwd / bwd) | ~7.8× / ~13.5× | ~32–64× | the fused `expf` log-partition + gather; C's reduction stays scalar (measured in tests/run; per-size table not reproduced here) |
| **Row losses** (KL-div / entropy / soft-label xent) | ~3.6–7.5× | ~15–39× | the per-row `logf`/`expf` reduction gcc/rustc keep scalar (measured in tests/run; per-size table not reproduced here) |
| **RoPE** (rotary embedding fwd / bwd) | **~29–54×** | **~146–156×** | the per-pair sin/cos — C calls scalar `sincosf`; Wukong one 256-bit `sincos` |
| **Gate** (SwiGLU / GeGLU `act(a)·b`) | ~5–13× | ~13–27× | the gate's silu/gelu folds an `expf` C/Rust keep scalar (measured in tests/run; per-size table not reproduced here) |
| **Argmax/argmin** — global + **row** | **~2.3–4.9×** (row; ~3.3–3.5× vs C(fast)) | ~5.8–12× | the within-row `(value,index)` bookkeeping gcc/rustc won't auto-vectorize even at `-ffast-math` — this one is real |
| **Argmax/argmin** — **column** (axis-0) | **1.5–2.5× SLOWER** | ~1.1–1.3× | *nothing* — the old ~2.7–5.3× was the peer's column-outer scan *(corrected 2026-08-04)* |
| **Scans** (cumsum / cummax / cummin / cumprod) | ~1.3–2.9× (cumsum **1.03–1.42× vs C(fast)**) | ~4.1–12× | the loop-carried `out[i]=⊕(out[i-1],x[i])` won't auto-vectorize; SIMD Hillis-Steele scan, or 4-row-interleaved ILP for cumprod / `lrscan` (cummax/cummin/cumprod bit-exact). Cumsum's `C(fast)` column is new (2026-08-04) and shows that row is ≈ a tie |
| **Streaming elementwise** (saxpy/poly) | ~1.1–1.5× | bandwidth | 256-bit + non-temporal stores once the working set spills L3 |
| relu / fused linear→relu / bias-add | ≈tie | — | already bandwidth-bound; no headroom standalone (won when *fused*) |

> **Reduction-bearing rows** (dot/ssd, `nn.Linear`, FFN, weight-gradient, bf16 GEMM + bf16 dot/sum,
> gemv, scaled GEMM, conv, column reductions, column/row argmax, cumsum, norms and their backwards,
> cross-entropy, row losses): IEEE-serial C baseline; see the **C(fast)** column xbench prints for
> the reassociation-normalized comparison, and the **C(omp)** column on `@parallel` rows for the
> multithreaded-C comparison (Fairness notes below). *Ten of those families had no `C(fast)` column
> at all before 2026-08-04; four of them (gemv, conv, the bf16 reductions, cumsum) turn out to be
> ties or losses once it exists.*

The pattern, restated after the 2026-08-04 peer audit: Wukong exceeds C/Rust where domain knowledge
lets a tensor compiler do what a scalar C compiler won't — **tiling, packing and register-blocking a
GEMM** (the durable 3–26× family), **fusing** a matmul with its epilogue or a norm's passes, and
**256-bit transcendentals** where gcc has only a scalar `libm` call. It does **not** exceed C on
strided reductions, column arg-reductions, transposes, or bandwidth-bound half-precision reductions:
those rows previously read as one-to-two-order-of-magnitude wins, and every one of them was a peer
defect or a withheld `-ffast-math`. On already-bandwidth-bound elementwise work it ties.

**Additional recognized coverage** (correctness-gated, interp == native bit-for-bit): the **embedding
lookup** `out[t,:] = weight[ids[t],:]` (the token-id row gather that is the first layer of every LLM —
`wukong_embedding_f32`, a bandwidth-bound copy single-core, an `@parallel` win across rows) and **2D
max/avg pooling** `wukong_{max,avg}pool2d_f32` (the CNN downsampler — an `@parallel` win; note that
for *regular* strides like 2×2/s2, gcc auto-vectorizes the pooling well, so single-core is a tie/loss
there, not a win — an honest sharp edge). Broadcast **bias-add** `out[r,c] = x[r,c] + bias[c]` is
memory-bound and ties gcc standalone, but is **fused for free** into the GEMM epilogue / norm affine
(where Wukong already wins) — which is how real models use it.

## How Wukong wins: domain-aware lowering

The headline wins come from a tensor compiler doing what a general C/C++ compiler will not do to
naively-written source:

- **Matmul dispatch.** Wukong's front-end recognizes a matmul loop nest — both the `ikj` accumulate
  form and the textbook `ijk` dot-product form, including the `C = A·Bᵀ` (`nn.Linear`) spelling — and
  lowers the *whole nest* to a tuned **register-blocked, cache-tiled, packed AVX2+FMA GEMM
  microkernel**. This is exactly how XLA/TVM/oneDNN lower a matmul op. gcc/rustc vectorize the inner
  loop but never tile, pack, or register-block, so they fall out of cache as the matrices grow. The
  recognizer also handles the **batched** form (a matmul nest under a batch loop, each index carrying
  a per-batch offset `x[h*S*D + …]`) — so **multi-head attention** dispatches one tuned GEMM per head,
  for both its `Q·Kᵀ` and `P·V` matmuls.
- **Transposed-A weight-gradient dispatch (`C = Aᵀ·B`).** The training backward pass needs
  `dW = dYᵀ·X`, where the contraction (batch) axis is the *outer* index of both operands — so A's
  logical `[m,k]` operand is the transpose of its `[k,m]` storage. Wukong recognizes
  `a[k*M+i]·b[k*N+j]` and dispatches to `wukong_sgemm_tn`, which transposes A into scratch once —
  O(m·k), ~1/n of the O(m·n·k) GEMM — then runs the *same* tuned NN microkernel. The result is
  bit-for-bit the kernel the interpreter oracle marshals (no new accumulation order). **The win is
  the tiling, not the transpose**: a competent C programmer writes this nest `kij` (hoisting
  `a[k*M+i]`), which gcc vectorizes fine, and against that peer the dispatch is worth **2.6–9.2×** —
  not the ~128× this document published against an `ijk` nest that read both operands column-strided
  (corrected 2026-08-04).
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
  whole op surface, *symmetric* for both precisions: **`dot`/`sum`** (`wukong_{dot,sum}_{bf16,f16}`),
  **`max`/`min`/`absmax`** (`wukong_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric-quant
  scale), **streaming `axpby`** (`wukong_axpby_{bf16,f16}`, half-in/f32-out), and the **36-op activation
  set** (`wukong_vmath_{bf16,f16}`). Because half precision moves **half the input bytes** of f32, the
  memory-bound ops are *bandwidth* wins that grow as the data spills cache: **~4–7.6× vs C** for dot,
  **~6–8×** for sum — **but 1.1–1.8× SLOWER than the same C at `-ffast-math`**, which is the
  like-for-like basis for a reassociated accumulate and was not measured until 2026-08-04. C/Rust can
  vectorize neither a `libm` call nor the half→f32 widen, so the
  activation gap is structural. Half storage is bit-exact across backends (f16 via shared `half`-crate
  shims, since Cranelift x64 lacks f16 convert lowering), and both call the identical kernel, so the gate
  stays exact. The dispatch now reaches the **mixed-precision GEMM** too: a bf16/f16 `C = A·Bᵀ`
  `nn.Linear` nest folds to `wukong_sgemm_{bf16,f16}_nt` (a lossless widen prepass + the tuned f32
  microkernel), **~23–28× single-core / ~50–91× `@parallel`** vs the idiomatic bf16 C that leaves the
  inline widen + reduction scalar — and **~4.6–5.6× vs the same C at `-ffast-math`**, which is the
  durable part (measured 2026-08-04; the previous "~10× vs a hand-optimized widen-then-tile bf16 C"
  was an estimate, not a measurement).
- **Reduction vectorization + multicore dispatch.** A naive f32 reduction (`s += x[i]*y[i]`) is one
  FMA down a single dependency chain — latency-bound. Wukong reassociates it across vector lanes ×
  unrolled accumulators (the standard BLAS reduction); gcc/rustc keep it strictly serial without
  `-ffast-math`. A `@parallel` reduction goes further, lowering to a **deterministic multicore
  reduction kernel** (`wukong_sreduce_f32_parallel`: dot/ssd/sum/sumsq folded by `+`, **max/min**
  folded by `fmax`/`fmin`, and **absmax** = `fmax` over `|x|` — the per-tensor max/range/absmax softmax
  stability and dynamic int8 quantization scale-computation (`scale = absmax/127`) need) whose result
  is bit-identical to the serial form regardless of core count
  (fixed-size chunks, ascending partial combine — which holds even for the non-associative `fmax`/
  `fmin`, since both forms evaluate the identical expression tree).
- **Auto-vectorization + fusion** of elementwise loops (incl. branchy ones via if-conversion), `x +
  y*z` → FMA contraction, and adjacent-loop fusion.
- **Vectorized transcendentals → 256-bit AVX2 dispatch.** A pure `out[i] = f(x[i])` loop for
  `exp`/`log`/`tanh`/`sigmoid`/`silu`/`gelu` lowers to a tuned **256-bit AVX2/FMA runtime kernel**
  (`wukong_vmath_f32`) running a ≈1-ULP Cephes minimax poly 8 lanes at a time — the width Cranelift's
  general vectorizer cannot emit (it caps at 128-bit SSE). gcc/rustc call scalar `libm`
  `expf`/`logf`/`tanhf` and cannot vectorize a loop containing a call, so the activation family
  (GELU/SiLU/tanh/sigmoid + softmax/log-softmax/cross-entropy) runs **~5–7.5× faster**, and an
  `@parallel` activation dispatches each thread's chunk to the kernel (multicore × 256-bit).
- **Fused row normalizations → single-pass kernel.** The per-token normalizations every transformer
  layer runs — **softmax**, **LayerNorm**, **RMSNorm** — are recognized from their canonical
  multi-pass source and folded into one **`wukong_norm_f32`** call: each row loaded once, 256-bit
  AVX2, the hand-vectorized `exp` for softmax, and the mean / variance / sum-of-squares reductions
  reassociated across 8 lanes. gcc/rustc auto-vectorize the elementwise passes but (without
  `-ffast-math`) keep the float reductions sequential and call scalar `expf` — so this *composes* the
  reduction-vectorization and transcendental wins into the fused op (**~1.9–6.6× faster than C**). The
  **affine** LayerNorm/RMSNorm real models run (a learned per-channel scale γ and shift β) dispatch to
  a sibling **`wukong_norm_affine_f32`** and hold the same win (**~1.7–3.7×**) — γ/β fold into the
  writeback for free.
- **Convolution via im2col + GEMM.** A conv expressed as im2col + matmul has its matmul recognized
  and dispatched to the GEMM microkernel, the same way XLA/cuDNN lower conv — so conv is accelerated
  for free, with no conv-specific kernel. Against a direct-convolution peer that is allowed to keep
  its accumulator in a register (`restrict`), the margin is **1.55×**, and **1.12× slower** at
  `-ffast-math`; the ~6–7× this document published was measured without `restrict` (corrected
  2026-08-04).
- **Auto-parallelization** of `@parallel` loops across all cores, with each per-thread chunk itself
  vectorized.

## Corrected peer measurements (2026-08-04)

The complete before/after board for the peer-strength audit described at the top of this document.

**Method.** Two release binaries were built from the same tree — one at the pre-audit commit (the
strawman peers) and one with the corrections — and run **adjacently, family by family**, so each
before/after pair shares a thermal and power window. This is the only instrument this laptop
supports (see the three-axes note above): the *ratios* are comparable within a pair, the absolute
GB/s and GFLOP/s are not comparable across pairs.

**Instrument state.** AC + charging, battery 11–35%. Per the power-state rule that means
**single-core rows are valid and every all-core row is directional only** (AC+charging caps all-core
at roughly 25%). The single-core rows are where the corrections live, so this does not soften the
conclusion; the `@parallel` columns below should be read as "which side of 1.0", not as a magnitude.
The measured single-core AVX2-FMA roofline drifted 43–120 GFLOP/s across the session, which is the
usual thermal spread and is why nothing here is quoted as an absolute.

Sign convention: **a positive multiple means Wukong is faster**; "slower" is spelled out.

### The four defects and what each cost

**Measuring this correctly is itself a landmine.** A standalone probe that puts the peer kernel and
its caller in ONE translation unit over `static` arrays lets gcc's interprocedural alias analysis
prove non-overlap by itself, and `restrict` then measures as a no-op — the first pass of this audit
made exactly that mistake and briefly recorded `restrict` as worthless for `colsum` and `conv`. The
peers are compiled to a **shared library**, where gcc sees only pointer parameters. Every figure
below is re-derived with the kernels in their **own translation unit**, which is the harness's real
compilation model.

| defect | families affected | isolated cost to the peer (gcc 14.2 `-O3 -march=native -ffp-contract=fast`, own TU, best-of-N, one process) |
|---|---|---|
| no `restrict` on any of 43 C kernels | all | `matmul_tn` 512³ **176.6 → 20.2 ms (8.7×)**; direct conv2d **1.99 → 0.40 ms (5.0×)**; `colsum` col-outer **21.3 → 5.0 ms (4.2×)**; `transpose` **29.5 → 24.3 ms (1.21×)**; `saxpy` **1.33×**; `relu` **0.96×** and `biasadd` **1.01×** — genuinely unaffected, because gcc already versions a flat 1-D map with a runtime alias check |
| column-outer loop order | colsum, colmax/min/absmax, colmean/sumsq/L2/RMS, colargmax/min | colsum 4096×1024 **21.33 → 0.58 ms (36.7× total, of which 8.7× is the loop order on top of `restrict`)**; colmax **34.82 → 0.86 ms (40.6×)**; colmean **33.03 → 1.24 ms (26.7×)**; colargmax **9.87 → 1.57 ms (6.3×)** |
| `ijk` with both operands column-strided | matmul_tn (weight-gradient GEMM) | 512³ **176.6 → 9.70 ms (18.2× total: 8.7× `restrict` × 2.1× loop order)** |
| unblocked transpose vs Wukong's blocked kernel | transpose | 2048² **29.51 → 14.09 ms (2.09× total: 1.21× `restrict` × 1.73× blocking)** |

Every one of these preserves the output exactly: colsum / colmax / colmean sum |Δ| = 0, colargmax 0
mismatched columns, transpose 0 mismatched elements, conv2d sum |Δ| = 0, and `matmul_tn` max |Δ| =
1.5e-5 on values of order 10² (the k-order is unchanged; the residue is gcc's vectorized rounding).

Plus a fifth, of a different kind: **ten reduction-bearing benches had no `C(fast)` column at all**,
so their published multiple was measured only against a C peer forbidden from reassociating — the
exact asymmetry this document's own fairness rule exists to remove (gemv, scaled_gemm, linear_bf16,
conv, colmax/min/absmax, rowarg, colarg, the bf16 dot/sum, cumsum, xent_bwd).

### Every ratio that moved

Single-core, vs the plain (honest-flags) C column unless stated. `~~struck~~` = the previously
published value.

#### Collapsed to a loss

| bench | shape | before | after | after, vs C(fast) | inflation |
|---|---|---|---|---|---|
| colsum | 1024×1024 | ~~77.67×~~ | **1.23× slower** | 1.31× slower | **63×** |
| colsum | 4096×1024 | ~~25.51×~~ | **1.70× slower** | 1.55× slower | 15× |
| colmax | 1024×1024 | ~~22.71×~~ | 1.40× slower | 1.91× slower | 16× |
| colmax | 4096×1024 | ~~31.38×~~ | 1.72× slower | 1.99× slower | 18× |
| colmin | 1024×1024 | ~~65.68×~~ | 1.37× slower | 1.58× slower | 48× |
| colmin | 4096×1024 | ~~23.60×~~ | 1.81× slower | 1.26× slower | 13× |
| colmaxabs | 1024×1024 | ~~43.73×~~ | 1.57× slower | 2.01× slower | 28× |
| colmaxabs | 4096×1024 | ~~43.99×~~ | 1.05× slower | 1.22× slower | 42× |
| colmean | 1024×1024 | ~~48.70×~~ | 1.60× slower | 1.71× slower | 30× |
| colmean | 4096×1024 | ~~23.19×~~ | 1.26× slower | 1.35× slower | 18× |
| colsumsq | 1024×1024 | ~~73.47×~~ | 1.57× slower | 1.69× slower | 47× |
| colsumsq | 4096×1024 | ~~24.38×~~ | 1.46× slower | 1.44× slower | 17× |
| coll2 | 1024×1024 | ~~56.49×~~ | 1.08× slower | 1.14× slower | 52× |
| coll2 | 4096×1024 | ~~26.33×~~ | 1.43× slower | 1.58× slower | 18× |
| colrms | 1024×1024 | ~~49.56×~~ | 1.25× slower | 1.52× slower | 40× |
| colrms | 4096×1024 | ~~35.71×~~ | 1.58× slower | 1.87× slower | 23× |
| colargmax | 1024×1024 | ~~3.09×~~ | 1.49× slower | 1.98× slower | 2.1× |
| colargmax | 4096×1024 | ~~4.10×~~ | 2.47× slower | 2.46× slower | 1.7× |
| colargmin | 1024×1024 | ~~2.69×~~ | 1.68× slower | 1.86× slower | 1.6× |
| colargmin | 4096×1024 | ~~3.48×~~ | 1.71× slower | 1.98× slower | 2.0× |
| transpose | 2048×2048 | ~~2.20×~~ | 1.00× (tie) | — | 2.2× |

The `@parallel` column for these collapses too: colsum/colmax/colmin/colmaxabs/colstat move from
+16× … +55× down to between 2.0× *slower* and 1.4× faster, i.e. **all-core Wukong roughly ties
single-threaded C** on the column-reduction family.

#### Shrank but survived

| bench | shape | before | after | after, vs C(fast) | inflation |
|---|---|---|---|---|---|
| matmul_tn (weight-grad GEMM) | 256³ | ~~43.83×~~ | **2.61×** (2.95× rd 2) | 2.68× | 16.8× |
| matmul_tn | 512³ | ~~50.69×~~ | **2.72×** (2.78× rd 2) | 2.65× | 18.6× |
| matmul_tn | 1024³ | ~~120.34×~~ | **5.28×** (9.16× rd 2) | 5.38× | 22.8× |
| matmul_tn `@parallel` | 1024³ | ~~436.23×~~ | **13.90×** | 14.16× | 31.4× |
| conv2d 3×3 | Cin16 20² → 64@18² | ~~5.90×~~ | **1.55×** | **1.12× slower** | 3.8× |
| transpose | 1024×1024 | ~~1.48×~~ | 1.03× (tie) | — | 1.4× |
| transpose `@parallel` | 2048² | ~~14.83×~~ | 7.05× (1.12× vs C(omp)) | — | 2.1× |
| rowargmax | 1024×1024 | ~~3.35×~~ | 2.45× | 1.92× | 1.4× |
| rowargmin | 4096×1024 | ~~3.45×~~ | 2.29× | 2.08× | 1.5× |
| gemv | 4096×4096 | ~~3.57×~~ | 3.16× | **1.09× slower** | 1.1× |
| gemv | 8192×2048 | ~~4.99×~~ | 3.55× | 1.11× | 1.4× |
| gemv | 16384×1024 | ~~3.58×~~ | 3.01× | **1.01× slower** | 1.2× |
| linear_bf16 | 512² | ~~25.67×~~ | 22.70× | **4.59×** | 1.1× |
| linear_bf16 | 1024² | ~~37.29×~~ | 27.52× | **5.62×** | 1.4× |
| scaled_gemm (Q·Kᵀ·α) | S=512 D=64 | ~~17.82×~~ | 11.26× | **1.19×** | 1.6× |
| scaled_gemm | S=512 D=128 | ~~24.24×~~ | 21.55× | **3.00×** | 1.1× |
| xent_bwd | 1024×1024 | ~~26.18×~~ | 11.31× | 8.65× | 2.3× |
| xent_bwd | 4096×512 | ~~9.78×~~ | 9.54× | 8.37× | 1.0× |
| cumsum | 1024×1024 | ~~1.35×~~ | 1.31× | 1.03× | 1.0× |
| cumsum | 4096×1024 | ~~2.02×~~ | 1.57× | 1.42× | 1.3× |
| bf16 dot | N=2²⁰ | ~~7.50×~~ | 7.64× | **1.36× slower** | — |
| bf16 dot | N=2²⁴ | ~~5.15×~~ | 6.41× | **1.16× slower** | — |

The pattern in the right-hand column is the important one. For `gemv`, `conv`, the `bf16` reductions
and `cumsum`, the plain-C multiple survives the peer fix but the **`C(fast)` column shows the win was
the withheld `-ffast-math`, not the kernel**. For `matmul_tn`, `linear_bf16` and `scaled_gemm` the
C(fast) column is far below the plain one too — those are GEMM-family wins that are real but 3–6×,
not 20–40×.

#### Did not move (the wins that were real)

Measured, not assumed: these were re-run through the identical A/B and their ratios changed only
within the session's run-to-run noise.

- **`nn.Linear` `C=A·Bᵀ`, matmul, fused FFN** — the peers were already the natural contiguous `ijk`
  nest; `restrict` changes nothing there because the accumulator is already a register.
- **Row argmax/argmin** — the peer already walked each row contiguously; 2.3–4.9× survives, and
  **3.3–3.5× vs `C(fast)`**, so gcc genuinely will not vectorize a within-row `(value,index)` scan
  even under relaxed FP.
- **The 35-op transcendental/activation family** — a `libm` call cannot vectorize with or without
  `restrict` or `-ffast-math` on this mingw toolchain (no `libmvec`).
- **The fused norms, RoPE, the norm backwards, row losses, the int8 GEMM.**
- **The flat elementwise rows** (saxpy, relu, poly, hadamard) — `restrict` measurably does nothing
  for a 1-D map: `relu` **0.96×** and `biasadd` **1.01×**, because gcc already versions those loops
  with a runtime alias check. (`saxpy` is the exception at **1.33×**, which is why this is stated per
  kernel rather than as a rule.) Their A/B pairs moved
  1.27× ↔ 2.33× in *both* directions across the session, which is the noise floor at this power
  state, not a peer effect.

### What was NOT re-measured

- **The norm-backward, cross-entropy, gate, row-loss, act-backward and scan families.** Their peers
  changed only by `restrict`, and their A/B pairs at this power state moved in *both* directions by
  up to ~2× (e.g. `xent` 1024×1024 read 3.59× before and 7.72× after; `softmax_bwd` 1024×1024 read
  2.37× before and 3.43× after; `layernorm_bwd` 4096×512 read 3.23× before and 2.24× after). That is
  the noise floor, not a measurement, so **no corrected figure is published for them** — their
  existing numbers stand, with the caveat that they were taken against non-`restrict` peers and
  should be re-derived at AC+full.
- **The end-to-end `model` section** (the 12-layer GPT-2-class stack and its PyTorch peers). It uses
  its own peer sources in `model.rs`, which this audit did not touch; its numbers stand as previously
  published and are **not** covered by the corrections above. Auditing `model.rs`'s peers the same
  way is open work.
- **The whole-suite geomean headline** ("3.8–4.4× faster than C"). It is dominated by the
  elementwise/reduction table, whose peers changed only by `restrict` — measured to be a no-op there
  — so the figure is expected to hold, but it was not re-derived in a single full-suite run at a
  valid power state and should be treated as unconfirmed until it is.
- **The GPU section.** Untouched by this audit (no C/C++/Rust CPU peers involved).
- **Anything at AC+full.** Every number above is AC+charging. Re-running the single-core rows at
  AC+full is expected to move absolutes but not the direction of any conclusion, since every figure
  quoted is a same-run adjacent ratio.

## Fairness notes

- **Aliasing — every C/C++ peer declares its buffers `__restrict__`** (added 2026-08-04; before that
  date, *none* of the 43 generated C kernels did). Without it gcc must assume `out[i] = f(x[i])` may
  clobber a later `x[j]` and cannot vectorize, interchange or reorder a nested kernel at all, while
  Wukong's tensor parameters carry non-overlap in the type system and its recognized kernels are
  hand-written AVX2 code that assumes it. `__restrict__` is the GCC spelling accepted in both C and
  C++ mode, so the g++ column inherits it. Every call site in `wukong_xbench` passes three (or four)
  genuinely distinct allocations, which is what makes the qualifier true rather than merely fast; the
  one bench that passed an aliasing filler pointer was fixed in the same commit.
- **Peer loop order — the peer must walk memory the way a competent programmer would** (audited
  2026-08-04). Four peers failed this and were rewritten: the column reductions (column-outer → row-
  outer), the column argmax (column-outer → row-outer with a running-best vector), the weight-gradient
  GEMM (`ijk` with both operands column-strided → `kij`), and the transpose (unblocked → 32×32
  blocked, matching Wukong's own kernel). Every one of the four failures inflated Wukong's published
  number, by 1.4× to 63×. Four tests in `crates/wukong_xbench/src/main.rs`
  (`every_generated_c_kernel_declares_restrict`, `column_family_peers_stay_row_outer`,
  `matmul_tn_peer_stays_kij`, `transpose_peer_stays_cache_blocked`) now pin all of it, because a
  weakened peer breaks nothing observable — the suite still runs, the cross-language check still
  passes, and the only symptom is a bigger multiple.
- **FMA:** Wukong contracts `x + y*z` to a fused multiply-add, so gcc is given its **default**
  `-ffp-contract=fast` (both fuse). Idiomatic Rust does not contract unless the author writes
  `f32::mul_add`, so the Rust column reflects rustc's default — a real toolchain-defaults difference,
  surfaced rather than papered over. Wukong and its interpreter oracle agree bit-for-bit (gated).
- **`-ffast-math` is withheld from the *primary* C column — and this *inflates* the reduction wins,
  stated plainly — so the harness now also prints a `C(fast)` column that removes the inflation.**
  The primary baselines get `-O3 -march=native` (+ default `-ffp-contract=fast`) but **not**
  `-ffast-math`, so gcc/rustc keep float reductions strictly IEEE-sequential (latency-bound).
  `-ffast-math` lets gcc reassociate and vectorize a `dot`/`ssd` reduction, **narrowing** those
  specific rows (the ~2.6–2.9× single-core `dot`/`ssd`). The transcendental and GEMM wins are largely
  unaffected — a `libm` call can't vectorize with or without it on this mingw toolchain (no
  `libmvec`), and the GEMM win is cache tiling, not reassociation. The primary column keeps honest
  default flags because `-ffast-math` changes C's numerical results, which would break the tight
  cross-language checksum that catches miscompiles — whereas Wukong's reduction reassociation is
  gated bit-for-bit against its own interpreter oracle. The **`C(fast)` peer column** (the same C
  source recompiled with `-O3 -march=native -ffast-math`) is now printed for every reduction-bearing
  bench, cross-checked at a looser magnitude-normalized `1e-2` tolerance (dropped with a printed note
  if it can't meet even that), and the Wukong/C(fast) ratio is reported alongside Wukong/C. So the
  IEEE-serial rows are labeled for what they are, and the reassociation-normalized number sits next
  to them instead of being left to a footnote.
- **The `@parallel` rows also print a multithreaded `C(omp)` peer.** The `@parallel` comparisons are
  by design Wukong-multicore vs *idiomatic single-threaded* C (disclosed on every row) — but a C
  author who cares can write `#pragma omp parallel for`. The harness therefore compiles exactly that
  twin of each `@parallel` kernel (`-fopenmp`, with `reduction(...)` clauses where the loop is a
  reduction, plus `-ffast-math` on the reduction rows — an OpenMP reduction already reassociates; the
  label stays `C(omp)`) and reports Wukong-vs-C(omp): the honest multicore-vs-multicore standing.
  OpenMP support is **probed at runtime** (compile a probe DLL, load it, require ≥2 threads inside a
  parallel region — verified working on this MSYS2 gcc: 22 threads, and the OpenMP dot runs ~3× its
  serial twin); if the probe fails the columns are skipped with a note. The parallel-GEMM peers in
  matmul/linear/transpose/colsum are measured inside the all-core thermal group *before* `Wuk(par)`,
  so any residual heat lands on Wukong, never the peer.
- **Matmul dispatch is the value proposition, stated plainly.** The C/Rust columns are the *naive
  nest a programmer writes*; Wukong's compiler optimizes it the way a tensor compiler should. The
  win **grows with size** precisely because tiling/packing matters more as the data stops fitting in
  cache — a single size could be a fluke, so a sweep is shown.
- **`nn.Linear` (`C = A·Bᵀ`)** is written the idiomatic way in all three languages: the `ijk`
  dot-product form (`for i,j { s=0; for k s+=a[i,k]*b[j,k]; c=s }`), where A and B are both read
  contiguously. gcc/rustc leave that f32 reduction strictly serial (latency-bound, ~1.5 GFLOP/s),
  while Wukong dispatches to its GEMM. The large ratio is real and is *caused by C's serial
  reduction*; it is not a strided-access strawman.
- **Correctness:** the native backend and the interpreter run the *identical* GEMM kernel (the
  interpreter marshals its memory through the same routine), so the differential oracle stays
  bit-for-bit exact even though the kernel reassociates.

## Results

### Compile time — JIT/embedding latency (xbench) and the headline `compile-vs` comparison

Two distinct measurements, labeled for what each is:

**The headline compiler-to-compiler comparison is `compile-vs`** (`cargo run -p wukong_bench
--release -- compile-vs`): `wukongc --emit=obj -O2` timed as a **subprocess** against `gcc -O2 -c`,
`g++ -O2 -c`, and `rustc -O --emit=obj` compiling **bare equivalent kernels** (a plain exported
function per translation unit — no headers, no `main`, matching work across all four languages) to a
native object, best-of-N minimum, same-run. Every column pays process startup, so this is the
apples-to-apples "how fast does each compiler compile the same kernel" figure. Its numbers are
recorded in dedicated benchmarking sessions (this laptop's clock state makes ad-hoc absolute numbers
unreliable); the ratio is a **~7–12× Wukong win (measured ~7–9× this session)** — single-digit-×, not
the 2–3-order figure below.

**The xbench per-kernel "compile (ms)" figure is in-process JIT/embedding latency**, not a
process-to-process compiler comparison:

| | Wukong (in-process JIT) | C (gcc, spawned) | Rust (rustc, spawned) | ratio |
|---|---|---|---|---|
| any kernel | ~0.3–1.5 ms | ~125–245 ms | ~185–250 ms | **~100–680×** (geomean ~306×) |

Wukong's number is its front-end + Cranelift JIT running **inside the harness process**, while the
C/Rust numbers include *spawning* the toolchain — so the ratio measures what a user of Wukong's
embedded/JIT path waits for versus shelling out to a C compiler (the relevant figure for an
ML-compiler REPL/JIT workflow, where this gap is real and structural: no LLVM, no process spawn),
and it *scales with how long the toolchain takes to spawn* (the dominant term — it varies run to
run; the geomean drifts between ~150× and ~310× across sessions, the latest full-board run measuring
**306×**, per-kernel **107–680×**). For the both-subprocess object-to-object comparison, use
`compile-vs` above. For an ML compiler — where edit/recompile/run iteration dominates developer
time — compile latency is the most robust result of all, under either measurement.

The pipeline's own hot stage within the front end is the **optimizer**, but its share is strongly
corpus-dependent and the basis must be named. Re-measured 2026-07-30 (`wukong-bench compile-time`,
release, warm best-of-N, `tests/run` + `examples` + `bench/kernels` = 354 files, 347 measured /
7 skipped): front total 24.98 ms vs optimize-at-`-O3` 39.57 ms → **the optimizer is 61.3% of the
front→`-O2` time** on the full corpus. The often-quoted ~80–85% is a *different basis* — the
400-function synthetic of the session-X1 compile-time work — and the many tiny `tests/run` fixtures
dilute the optimizer here; `docs/compile-floor.md` records the same effect (~72% on the model-kernel
subset vs ~62% full-corpus). Once the Cranelift backend is counted the optimizer is not the dominant
*wall-time* stage at all: the full-pipeline split was **backend 73.4% / optimize 16.3%** on the
2026-07-11 central `compile-profile` run (not re-measured on 2026-07-30 — see
`docs/compile-floor.md` §4). The recognizer sweep and sema are negligible either way.

Two output-preserving changes cut the optimizer **~31%** (in-process, 400-function `-O2`:
**18.0 ms → 12.5 ms**): the CSE value-numbering key became a packed allocation-free `enum` instead of
a `format!` string built per pure instruction, and the fixpoint loop now skips passes already at
fixpoint — dropping the final all-passes no-op *confirmation* sweep without changing the sequence of
mutations. The resulting MIR is bit-identical (the differential and `-O0`≡`-O{1,2,3}` gates both still
pass), so the speedup is free of any correctness cost. That change also moved the ranking it was
derived from: CSE is **no longer the single costliest pass** — on the 2026-07-30 corpus run
`simplify-cfg` (19.8%, 10.22 ms) and `cse` (18.9%, 9.76 ms) are effectively co-leading, ahead of
`simplify-phis` 17.3%, `mem2reg` 14.3%, `licm` 11.5%, `dce` 8.7%, `simplify` 5.9%, and `dse`/`inline`
1.7% each. (Pass *shares* are ratios within one run, so they survive this laptop's clock swing; the
absolute ms do not.)

### End-to-end model — a 12-layer GPT-2-class transformer stack (CPU inference)

`wukong-xbench model` is the culmination benchmark: not one kernel but a **full inference forward**
over a GPT-2 124M-shaped decoder stack — 12 pre-LayerNorm transformer blocks (multi-head causal
attention + GELU MLP, both with residuals) plus the final LayerNorm — at d_model=768, heads=12,
d_ff=3072, seq lengths S=128 and S=512.

**What is measured, exactly.** One Wukong block function (ordinary `.wk` source, adapted from
`examples/gpt2.wk` to the full config) is compiled once through the real pipeline
(parse → sema → mir_build → `-O3` → Cranelift JIT) and called **12× per forward by the harness**
with per-layer weight pointers, ping-ponging two activation buffers — the way a real runtime drives
a layer stack; the same harness loop drives the C implementation, so the layer-loop cost basis is
identical. Scratch buffers are host-allocated and shared between the columns. Attention runs per
head over contiguous extracted slices (Qh/Kh, V transposed) in **both** languages. The benchmark
**scans the optimized MIR and prints the recognized-kernel dispatch set**, making the mechanism
transparent; one block dispatches

```
2x wukong_norm_affine_f32 (LayerNorm1/2)   6x wukong_sgemm_nt (Q/K/V/O/PV/down-proj)
1x wukong_sgemm_nt_alpha  (scaled Q·Kᵀ)    1x wukong_sgemm_nt_epi (fused GELU FFN up-proj)
1x wukong_norm_f32        (masked row softmax, batched)   2x wukong_velem_f32 (residual adds)
```

**Baselines.** The C column is the same computation as one competent hand-written translation unit
(contiguous loops, its own per-head attention, two-pass LayerNorm, tanh-approx GELU with Wukong's
constants) at the suite's standard `gcc -O3 -march=native -ffp-contract=fast`; **C(fast)** is the
identical source at `-O3 -march=native -ffast-math` (the `llama2.c -Ofast` basis, letting gcc
reassociate + vectorize the dot products — the strongest flags-only C). The naive-dot C forward is
tens of seconds per call at S=512, so it is skipped there by default (`XBENCH_MODEL_NAIVE` forces
it), the same rule as the ≥2048³ naive matmuls.

**PyTorch peer (the industry baseline).** When `python` + `torch` import (probed gracefully; a
printed note + `n/a` columns otherwise), the bench adds **PyTorch CPU** — both **eager**
(MKL/oneDNN-backed; `T1`/`Tn`) **and `torch.compile` max-autotune fullgraph** (`T1(comp)`/`Tn(comp)`),
run same-run at `set_num_threads(1)` and all-threads; the **compiled** columns are the honest bar
(ATEN-pinned because Inductor's Windows CPP FP32 GEMM template is broken, disclosed). The harness dumps the *exact* weight and
input buffers every other column reads as little-endian f32 blobs (the config-invariant 12-layer
weights once per run, the per-config input/final-LN blob per S) and generates a self-contained
Python script that rebuilds the identical forward: `F.linear` computes `x·Wᵀ` over the *same*
`[out, in]` row-major weights Wukong/C dot against (the layouts coincide — the bytes are used
as-is), `F.layer_norm` at the same `eps=1e-5`, the same tanh-approx GELU
(`F.gelu(approximate="tanh")` — Wukong's √(2/π)/0.044715 flavor exactly), and multi-head causal
attention two ways: `F.scaled_dot_product_attention(is_causal=True)` (the fused industry path)
**and** a manual matmul+softmax variant. All under `torch.inference_mode()`, float32. Each variant
warms ≥3 forwards then times ≥10 (min + median; the timed loop body is one bare forward — no
per-iteration allocation/IO beyond what eager torch does inside a forward), at
`torch.set_num_threads(1)` (**T1**) and at the default all-threads (**Tn**), reported as the same
ms/forward + tokens/sec rows. Torch is timed in the *same bench invocation* immediately after the
Wukong columns (same-run adjacency); inside the script the single-thread variants run first and
the all-core one last, so multicore heat pollutes no single-thread torch number. Columns:
`T1(sdpa)`, `T1(man)`, `Tn(sdpa)`, and the compiled `T1(comp)`/`Tn(comp)`; the Wukong-vs-Torch ratio lines print alongside the
Wukong-vs-C ones. Disclosed asymmetry: `Tn(sdpa)` is genuinely multicore while the C columns are
single-threaded — the printed ratios name the thread counts. As everywhere in this suite, absolute
ms is clock/thermal-bound, so only the **same-run ratios** are recorded (below); cross-run raw torch ms
is not a stable headline — the same-run interleaved A/B ms below are the honest form.

**Correctness.** Four gates run inside the benchmark: (1) the interpreter oracle executes the
*identical* 12-layer forward (same MIR, same weights, same harness loop) at a reduced config and
must match the JIT **bit-for-bit** — it does; (2) Wukong serial vs `@parallel` final outputs are
**bit-exact** (every dispatched `_parallel` kernel is bit-identical to its serial twin); (3) Wukong
vs C final `[S,768]` outputs agree to a magnitude-normalized `max|Δ|/max|out| < 1e-3` (per-element
relative error is meaningless on LayerNorm-centered near-zero outputs; 12 layers of reassociation +
poly-vs-libm transcendentals compound the honest small differences); (4) Wukong vs torch's SDPA
output agrees under the same magnitude-normalized `1e-3` metric (same GELU flavor and eps, so the
residual is the same reassociation/poly-vs-libm class — no loosening needed), and the script itself
reports its manual-attention-vs-SDPA agreement.

**Measured standings (2026-07-11 — three independent same-day rounds for the compiled-torch bar; the earlier 2026-07-09/10 eager campaign ran five valid full-peer rounds).** Absolute ms
swings ~3× with this laptop's clock state (the valid rounds read roofline 130–141 GFLOP/s; a sixth
round at roofline 61 — 1 a.m. system activity plus a fresh-binary cache scan — was caught by the
validity protocol and discarded), so the **ratios are the metric**, not raw ms. All four correctness
gates passed every round: interp == native bit-exact, serial == `@parallel` bit-exact, and
Wukong-vs-C / Wukong-vs-torch outputs agree to `max|Δ|/max|out|` ≈ 1–2×10⁻⁶ (tolerance 10⁻³).

*Single-core*, the 12-layer stack runs **~19–21× idiomatic C(gcc)** and **3.6–4.9× C(-ffast-math)**,
and **beats compiled PyTorch single-thread at both shapes** — Wukong-1c is faster than
compiled-torch-1T (`torch.compile` max-autotune fullgraph) at S=128 and S=512.

*Multicore*, the campaign's closing move was the **`@parallel` head-loop region** (2026-07-10): an
independent-iteration `for` loop with body-local scratch inside an `@parallel` fn now outlines into a
`wukong_parallel_for` region — conservative affine-disjointness legality, the serial GEMM/norm
kernels running inside each iteration so serial == `@parallel` stays **bit-exact**, autodiff
declining the construct loudly. The model spells its per-head attention loop that way naturally, and
it **flipped the all-threads-torch comparison**:

| S | Wukong `@parallel` vs all-threads compiled torch (`Tn(comp)`) | `@parallel` scaling | vs C(gcc) multicore |
|---|---|---|---|
| 128 | **1.07–1.43× FASTER** (three rounds) | default 4.44× / best 5.22× | ~61–69× |
| 512 | **1.39–1.81× FASTER** (three rounds; the head loop was worth ~1.4×) | 1.00/1.70/2.13/3.00/3.68/4.70× at T=1/2/4/8/12/16 (default 4.44×, best 5.22×, ceiling ~5.0–5.4×) | large |

Before the region, three mid-session rounds had Wukong `@parallel` scaling 2.2–3.0× @S=128 / 2.4×
@S=512 and sitting **1.01–1.63× *behind* all-threads eager torch @S=512** — the wide range being the peer's
power-state swing (eager torch-Tn ran 443→271 ms across rounds while Wukong `@parallel` held ~440 ms in
every round). That exposed the structural finding the region then addressed: Wukong's parallel path
does **not** ride the clock upside peers do, consistent with a sync/serial-fraction bound rather than
a clock bound. The residual multicore headroom vs 16 physical cores now concentrates in the large-size
parallel-GEMM grain (2048³/4096³ 91–93% of MKL-all; 512³–1024³ at/above parity — see the library table above), the one
honestly-open lever. Compiling the whole block takes Wukong **~3–12 ms vs gcc's ~0.4–0.9 s** for the
equivalent TU; tokens/sec = S ÷ ms/forward.

**2026-07-30 re-measurement — two rounds, and a methodological caveat that limits what they prove.**
Two full-suite rounds were run at HEAD. Correctness reproduced exactly in both: the interp gate
(interpreter == native == `@parallel` native) **bit-exact** over all 1024 outputs, serial ==
`@parallel` **bit-exact** at both S, and Wukong-vs-C / vs-C(fast) / vs-torch-eager / vs-torch-compiled
cross-checks at 1.93e-6 / 1.05e-6 / 9.12e-7 / 9.53e-7 against a 1e-3 tolerance. The per-layer
dispatch set printed from the optimized MIR is **identical to the one documented above**, and the
`@parallel` set contains the `wukong_parallel_for` head-loop region — so no recognizer regressed.

The timings, however, exposed a hybrid-CPU effect that the existing protocol does not control for.
Round 1 (AC+charging, measured roofline 131) read Wuk(1c) **277.8 ms** at S=128; Round 2 (AC+full,
roofline probe 78) read **617.6 ms** for the same work — 2.2× slower *on the healthier power state*.
On this 6 P-core + 8 E-core + 2 LP-E part a single-threaded measurement is at the mercy of which core
type the scheduler picks, and nothing in the harness pins it (the note above that "pinned to a P-core
it reaches a stable ~117–126" is the same effect seen from the other side). Consequences, stated
rather than papered over:

- **Single-core rows are only comparable within a round**, and a round's `Wuk(1c)` column should be
  sanity-checked against its own roofline probe before any cross-round claim is made.
- **`@parallel` scaling figures (par ÷ 1c) inherit that noise in their denominator.** Round 2's
  apparent 6.81× (S=128) / 8.42× (S=512) are *inflated by a slow 1c*, not evidence of improved
  scaling, and are deliberately **not** claimed here.
- Peer ratios that do not involve Wukong's own 1c column are the robust ones. At AC+full, Wukong
  `@parallel` measured **1.60× (S=128) and 3.98× (S=512) faster than all-threads compiled torch**,
  and GEMM `@parallel` measured **99% / 97% / 114% / 128% of MKL-all** at 256³/512³/1024³/2048³ —
  at or above the standing band, one round, warm machine.
- Single-thread vs compiled-torch-1T came out **1.07× and 1.28× faster** (S=128/S=512) in Round 1 but
  **1.20× slower** at S=128 in Round 2, tracking that round's degraded 1c. The recorded claim is left
  as it stands; one round cannot overturn it, and one round cannot confirm it either.

**A real compiler finding along the way (since resolved):** in the first preliminary sessions the
`@parallel` column was only *partially* multicore — `wukong_mir_build`'s statement-path matmul
recognizer hardcoded the serial kernel (`lower_for`: `emit_sgemm(&nest, false)`), so inside a
multi-statement `@parallel` function only the batched norms and the fused-GELU FFN GEMM dispatched
`_parallel` kernels while the six plain GEMMs stayed single-threaded, and multicore bursts dragged the
package clock for the still-serial phases (so `Wuk(par)` could read *slower* than `Wuk(1c)`). That
site now passes the function's `@parallel` flag (`emit_sgemm(&nest, self.parallel_fn)`); combined with
the velem-parallel residual-add dispatch and the head-loop region above, the whole block is genuinely
multicore. The parallel GEMM is bit-identical to serial, so the differential gate was never at risk.

GFLOP/s (higher is better), naive `ikj` nest in each language. Measurement ordering is thermal
hygiene: the naive C/Rust (and `C(fast)`) nests are measured in the **same single-core thermal group
as Wuk(1c)** — after the library peers, before any all-core burst — so they are never read
heat-throttled (they previously ran dead-last, after the all-core `Wuk(par)`/`MKL(all)` bursts, which
could understate them); `Wuk(par)` still runs last of all (we throttle ourselves, never a peer).

| size | Wuk 1-core | Wuk @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 256³ | 80–100 | 86–99 † | 21–41 | 21–42 | **~2.8–4.3×** | **~3.4–4.1×** |
| 512³ | 110–126 | 328–462 | 30–41 | 38–45 | **~3.0–3.1×** | **~9×** |
| 1024³| 102–119 | 437–522 | 22–30 | 25–32 | **~3.4–3.6×** | **~14–18×** |

The single-core kernel now holds **~110–120 GFLOP/s** at 512³–1024³ — ≈90% of one P-core's AVX2-FMA
peak (pinned to a P-core it reaches a stable ~117–126) — while gcc's naive nest falls from ~40 to ~22
GFLOP/s as 1024² spills out of cache, so the **single-thread lead widens with size**. Crucially, that
single-core kernel also **beats the tuned `matrixmultiply` Rust crate by ~1.1–1.3×** (117 vs 89, 115
vs 91, 104 vs 96 GFLOP/s at 256²–1024²) — so Wukong is not merely beating naive C; it edges a
dedicated hand-optimized GEMM library while sitting at ~90% of the AVX2-FMA roofline. The parallel
kernel's absolute throughput swings ~3× with power state, so its standing is the **same-run vs-MKL
ratio** in the library table above rather than a fixed GFLOP/s (the pre-2D-path ~437–522 @1024³ /
~690 @2048³ figures predate the current default parallel path); the `MC=144` cache-block widening cut
the B-panel's L3 re-streaming, and the pack-scratch is reused across blocks rather than re-allocated
per K-block.

† At 256³ the parallel kernel deliberately falls back to the serial one: ~17M MACs is below the
work threshold where cross-core wake/sync pays off on this P+E hybrid, so "@parallel" ≈ single-core
there (a measured fix — naive threading at that size was a net *loss*).

*2026-07-08 note*: the `Wuk @parallel` ranges above predate the **2D block-parallel path** (now the
default), which A/B-measured **1.65× @512³ and 1.26× @1024³** over the path that produced them —
absolute GFLOP/s swing ~3× with this laptop's power state, so the ranges are not re-baselined from a
throttled day; the same-run vs-MKL ratios in the library table above are the current standing.
*2026-07-10 update*: that path is now **size-keyed** — per-block packing at mid/large (it beat the
shared-cooperative-pack design in both ABBA orderings — each worker warming its own L2 is the win),
a shared-pack + 1 Mi-MAC-task small band, and the parallel gate lowered 2²⁶→2²³ MACs so 256³ engages
(**~94–110% of MKL-all**, was deliberately serial). A C-tile microkernel prefetch also closed the
single-core writeback tail (**2048³ 86–88% → ≥100% of MKL-1c**, `WUKONG_GEMM_PF_C=0` kill-switch).
*2026-07-11 updates*: the small-size shared-pack band did not survive pool unification — **per-block
packing is the default at every size** (`WUKONG_GEMM_2D_SHARED` keeps the retired shapes measurable);
and the per-block blocks are now handed out by an **atomic claim queue** instead of rayon's static
range-split (`WUKONG_GEMM_DYN=0` opts back) — on this hybrid the OS intermittently strands a worker
for 100–500 ms stretches, and block-granular self-scheduling halves that episode damage (512³ solo
CoV 14→8%, floor +28%) while winning or washing every measured shape (skinny NT table:
75–112% of MKL-all, was 70–96%). See the library table above for current same-run standings.

### `nn.Linear` `C = A·Bᵀ` — Wukong dispatches to GEMM; naive C is latency-bound

| size | Wuk 1-core | Wuk @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 512² | 81–112 | 251–348 | ~4.0–4.8 | ~3.4–4.5 | **~19–25×** | **~52–76×** |
| 1024²| 88–108 | 258–457 | ~3.5–4.4 | ~3.4–4.3 | **~22–26×** | **~67–104×** |

C/Rust leave the idiomatic `ijk` dot-product reduction strictly serial (~4–5 GFLOP/s, latency-bound),
while Wukong recognizes `C = A·Bᵀ` and dispatches to the same packed GEMM — hence the order-of-
magnitude gap (caused by C's serial reduction, not a strided-access strawman; see Fairness notes).
*IEEE-serial C baseline; see the `C(fast)` column for the reassociation-normalized comparison, and
`C(omp)` for the multithreaded-C one.*

**Residual projection (`x = x + act(x·Wᵀ + bias)`) — the transformer skip connection.** The output
projection of every attention/FFN sub-layer adds its result back to its input. Written fused in one
nest, the *accumulate* store `x[i*N+j] = act(x[i*N+j] + dot + bias[j])` (not the bare `x = dot`) would
**block the matmul recognizer entirely** — so the whole nest, GEMM and all, falls to a scalar loop in a
naive compiler. Wukong recognizes it and dispatches to **`wukong_sgemm_nt_epi` with `beta = 1`**: the
kernel accumulates `act(beta·x_old + A·Bᵀ + bias)` = `act(x_residual + A·Bᵀ + bias)` straight in its
C-tile writeback, so it gets the *same* ~19–26× single-core / ~52–104× parallel GEMM dispatch as the
plain `nn.Linear` above, **plus** the residual-add and bias/activation folded in for free (the
fused-epilogue kernel — no new symbol, no separate pass). This reuses the existing `nt_epi` kernel
end-to-end, so it stays bit-exact across backends (`tests/run/linear_residual{,_relu}.wk`).

**Fused FFN `C = silu(A·Bᵀ)` — the complete Dense / SwiGLU layer** (xbench `ffn`). The matmul nest
followed by a `silu` epilogue nest fuses to one `wukong_sgemm_nt_epi` (SILU act, null bias): the
activation is computed in-register on the GEMM's C-tile writeback, so C is written once. The idiomatic
C/Rust do the GEMM, then a *second* full pass that reads C back and applies silu with scalar libm
`expf` (which they cannot vectorize).

| size  | Wuk 1-core | Wuk @parallel | C (gcc) | 1-core vs C | parallel vs C |
|-------|-----------|---------------|---------|-------------|---------------|
| 512²  | ~115 | ~227 | ~4.8 | **~24×** | **~48×** |
| 1024² | ~107 | ~389 | ~4.1 | **~26×** | **~95×** |

Honest accounting: the **bulk** of this ratio is the same serial-reduction-vs-tiled GEMM gap as the
`nn.Linear` row above (~22–26×); the fused vectorized silu is the *incremental* win over C's separate
scalar-`expf` pass — it widens the lead slightly and proves the activation does not erode it (C's `ffn`
GFLOP/s ≈ its plain-`linear` GFLOP/s, so the silu pass is not a strawman). silu(GEMM) is checked over the
whole buffer to a tight tolerance (the GEMM reassociates, silu is poly-vs-libm ~1 ULP).
*IEEE-serial C baseline; see the `C(fast)` column for the reassociation-normalized comparison.*

### Weight-gradient `C = Aᵀ·B` — the training backward GEMM

The backward pass computes `dW = dYᵀ·X`: the contraction (batch) axis is the **outer** index of both
operands, so A is stored `[k,m]`. Wukong recognizes `a[k*M+i]·b[k*N+j]` and dispatches to
`wukong_sgemm_tn`, which transposes A into scratch once (O(m·k), ~1/n of the GEMM) then runs the
*same* tuned NN kernel.

> **Corrected 2026-08-04.** This section previously published **~42–128× single-core and ~44–445×
> `@parallel`**, measured against a C peer written as an `ijk` dot-product nest in which BOTH
> operands are read column-strided (`s += a[k*NS+i]*b[k*NS+j]`) — every k step advancing both
> pointers by NS floats. Nobody writes `C = Aᵀ·B` that way. With A stored `[K,M]`, k is already A's
> *outer* index, so the natural nest is `kij`: hoist `a[k*NS+i]`, then stream `b[k*NS+·]` and
> `c[i*NS+·]` contiguously — exactly what the same file's untransposed `c_matmul` already did.
> Isolated at `-O3 -march=native`, kernels in their own translation unit, 512³: **176.6 ms `ijk` →
> 20.25 ms `ijk` + `restrict` → 9.70 ms `kij` + `restrict`, an 18.2× total handicap.**
> The prose below also used to claim a hand-transposed C would recover only ~4–5 GFLOP/s and leave a
> "durable ~10×"; the measured `kij` peer does substantially better than that, and the durable win is
> 2.6–9.2× (two independent rounds).

Measured with the corrected `kij` + `restrict` peer (2026-08-04, AC+charging so the all-core column
is directional; each pair is same-run adjacent):

| size  | 1-core vs C | 1-core vs C(fast) | `@parallel` vs C | was (1-core / `@parallel`) |
|-------|-------------|-------------------|------------------|-----------------------------|
| 256²  | **2.61×** (2.95× round 2) | 2.68× | 3.11× | ~~43.8× / 50.4×~~ |
| 512²  | **2.72×** (2.78× round 2) | 2.65× | 5.08× | ~~50.7× / 89.4×~~ |
| 1024² | **5.28×** | 5.38× | 13.90× | ~~120.3× / 436.2×~~ |

So the inflation was **16.8–22.8× on the single-core row and 16.2–31.4× on `@parallel`**. Wukong
still wins this family — the transpose-prepass plus a tiled, packed, register-blocked GEMM beats a
vectorized-but-untiled `kij` nest, and the lead grows with size exactly as the cache argument
predicts — it just wins by 2.6–9.2× (both rounds), not by two orders of magnitude. The `C(fast)` column is
essentially identical to the plain one, which is the expected result: the win here is tiling, not
reassociation. (Absolute GFLOP/s is clock-sensitive; the ratio is the stable part.)

### Convolution — im2col + GEMM vs idiomatic direct conv

A 3×3 conv (Cin=16, 20×20 input → 64 filters → 18×18 output), the way XLA/cuDNN lower it: an im2col
gather builds the `[Cin·K·K, OH·OW]` column matrix, then the conv is a matmul `Y = W · col` that the
recognizer dispatches to the tuned GEMM. C and Rust run the idiomatic **six-deep direct-convolution
nest** (the loop everyone writes by hand).

> **Corrected 2026-08-04 — the largest single `restrict` effect in the suite.** This section
> published **~6–7×**. The direct-convolution peer accumulates into a scalar `s` and then stores to
> `output`; without `restrict` gcc must assume that store may alias `input`/`weight`, which forces a
> reload of the whole inner nest's operands on every output pixel. Adding `__restrict__` — changing
> nothing else about the loop — took the C peer from **4.9 to 21.1 GFLOP/s, a 4.3× peer speedup**,
> and the published multiple from 5.90× to **1.55×**. With the (new) `-ffast-math` column the direct
> conv is **faster than Wukong** (1.12×).

| kernel | 1-core vs C | 1-core vs C(fast) | was |
|--------|-------------|-------------------|-----|
| conv2d 3×3 (Cin=16, 20×20 → 64@18×18) | **1.55×** | **1.12× slower** | ~~5.90×~~ |

Same result (checksum cross-checked). The conv's GEMM is small (M=64, K=144, N=324) so it runs below
the large-matmul peak. Against a peer that is allowed to keep its accumulator in a register, the
im2col+GEMM route is a **1.55× win at honest flags and a slight loss once the peer may reassociate**
— it is not the 6–7× this document claimed. What remains true is the structural point: Wukong gets
whatever conv performance it has *for free* through the existing matmul dispatch
(`tests/run/conv_im2col.wk`), with no conv-specific kernel.

### Transcendentals / activations — Wukong dispatches to a 256-bit AVX2 kernel; C calls scalar `libm`

The activation family every transformer runs, and **the cleanest compute-bound win in the suite**.
Wukong recognizes a pure `out[i] = f(x[i])` loop for **35** functions —
`exp`/`log`/`exp2`/`log2`/`exp10`/`log10`/`cbrt`/`expm1`/`log1p`/`tanh`/`sigmoid`/`silu`/`gelu`/`elu`/`leaky_relu`/`softplus`/`softsign`/`logsigmoid`/`mish`/`selu`/`tanhshrink`/`hardsigmoid`/`hardswish`/`sin`/`cos`/`tan`/`atan`/`asin`/`acos`/`erf`
plus the hyperbolic family `sinh`/`cosh`/`asinh`/`acosh`/`atanh` — and lowers the whole
loop to a **256-bit AVX2/FMA runtime kernel** (`wukong_vmath_f32`) — the same domain-aware dispatch
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

| kernel | Wukong vs C | notes |
|--------|--------------|-------|
| `exp`  | **~5.9–6.8× faster** | `out=exp(x)`; 256-bit AVX2 poly vs scalar `expf` |
| `log`  | **~4.6–7.5× faster** | `out=log(x)`; 256-bit Cephes poly vs scalar `logf` (libm `logf` timing varies run-to-run) |
| `tanh` | **~4.5–6.1× faster** | exp-based, identical algorithm everywhere; only Wukong vectorizes (256-bit) |
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
compiles ~260–680× faster (the simplest single-op kernels compile fastest, so the transcendental
rows post the highest compile ratios on the board). These are compute-bound, so the win is real SIMD
throughput, not bandwidth. An `@parallel` activation dispatches *each thread's chunk* to the kernel, so it runs
multicore × 256-bit. `softmax`/`LayerNorm`/`RMSNorm` are recognized and dispatched to a fused
single-pass kernel (`wukong_norm_f32` — see the section above), and **log-softmax / cross-entropy**
(`exp` + `log`) run as fused vectorized chains (`tests/run/`); a transformer FFN block (two `nn.Linear` matmuls + GELU)
composes the GEMM and activation wins in one function (`tests/run/ffn_block.wk`). All paths are
bit-identical across backends — the dispatched kernel by marshalling, the inlined polys by
construction.

### Fused row normalizations — softmax / LayerNorm / RMSNorm dispatch to one kernel

The per-token normalizations every transformer layer runs. Wukong recognizes the canonical
multi-pass source (softmax's max/exp/sum/normalize; LayerNorm's mean/variance/normalize; RMSNorm's
mean-square/normalize) and folds the whole thing into one **`wukong_norm_f32`** call — or, when the
normalize step carries the learned per-channel scale γ and shift β that **real** transformer
LayerNorm/RMSNorm apply (`(x-μ)·inv·γ + β`), into one **`wukong_norm_affine_f32`** call with γ/β
folded into the same single-pass writeback. Either way: 256-bit AVX2, the hand-vectorized `exp` for
softmax, and the reductions reassociated across 8 lanes. gcc and
rustc at their honest defaults (no `-ffast-math`) auto-vectorize the elementwise passes but keep the
float reductions strictly sequential, and call scalar `libm` `expf` for softmax. All three languages
copy `x`→`out` then normalize in place — identical work, so the copy pass is charged to everyone and
these figures *understate* the per-pass advantage — and the full output buffer is cross-checked
element-by-element (a tight tolerance, since Wukong reassociates the reductions and C does not).

Measured as the **Wukong-vs-C runtime ratio** over feature rows of 768 and 4096 f32. Absolute
ns/call swings ~2× run-to-run with the laptop's clock/thermal state (the whole machine speeds up or
slows down together), so — as with the GEMM — the honest figure is the *ratio*, reported as a range
across runs:

| op | Wukong vs C | where the win comes from |
|----|--------------|--------------------------|
| softmax   | **~4.4–6.6× faster** | vectorized `exp` (gcc's scalar `expf` can't vectorize a loop with a call) + the reassociated sum |
| LayerNorm | **~3.0–3.5× faster** | two reassociated reductions — the mean, then the variance |
| RMSNorm   | **~1.9–2.5× faster** | one reduction (mean-square); the copy/scale elementwise passes, which gcc vectorizes too, dilute it |
| LayerNorm (affine γ, β) | **~2.8–3.7× faster** | the real transformer form `(x-μ)·inv·γ + β` → `wukong_norm_affine_f32`; same reduction win, γ/β fused into the writeback |
| RMSNorm (affine γ) | **~1.7–3.2× faster** | the real transformer form `x·inv·γ` → same affine kernel |
| L2 / unit-normalize | **~1.8–2.1× faster** | `x / √(Σx² + eps)` (cosine similarity, normalized embeddings, retrieval keys) — RMSNorm without the mean divisor, the same fused single-pass reduction → `wukong_norm_f32` |

*IEEE-serial C baseline on the reductions; see the `C(fast)` column for the reassociation-normalized
comparison.*

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
Wukong recognizes the **batched** form — `for r in 0..R { <norm over x[r*C + i]> }`, the row-major
`[R, C]` matrix — for all three norms and folds the entire batch into one `wukong_norm_f32(x, x, R, C,
…)` call (each row a fused single pass), instead of leaving the outer loop to scalar/vectorized code. In
a `@parallel` function the same batch dispatches to **`wukong_norm_f32_parallel`**, mapping the
independent rows across cores. Measured as the **Wukong-vs-C runtime ratio**, serial *and* `@parallel`,
at an L3-resident batch and a batch that spills L3:

| op | 512×768 (1.5 MB, in L3) serial · @parallel | 4096×4096 (64 MB, ≫ L3) serial · @parallel |
|----|-----|-----|
| RMSNorm   | **~2.0×** · ~1.1× | **~2.0×** · **~2.9×** |
| LayerNorm | **~2.5×** · **~1.9×** | **~2.1×** · **~4.0×** |
| softmax   | **~5.6×** · **~6.1×** | **~4.7×** · **~10.2×** |

*IEEE-serial C baseline on the reductions; see the `C(fast)` column for the reassociation-normalized
comparison.*

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
`differential_parallel_batched_norm` pins the end-to-end dispatch; `tests/run/batched_*.wk` and the
fuzzer's `{rmsnorm,layernorm,softmax}_batched` kernels cover all three). The **affine** forms (a learned
per-column γ/β) batch the same way — `wukong_norm_affine_f32[_parallel]`, the real transformer norm.
C/Rust here are the idiomatic single-threaded per-row nested loops at honest defaults.

### int8 quantized `nn.Linear` — `vpdpbusd` register-blocked, beats gcc single-core

Quantized inference runs `nn.Linear` as `C = A·Bᵀ` with **`u8` activations × `i8` weights → an `i32`
accumulator** (the QNNPACK / XNNPACK / oneDNN layout). Wukong recognizes the `ijk` dot-product nest
— `s += (a[..] as i32) * (b[..] as i32)`, A `u8`, B `i8` — and dispatches it to the
**`wukong_i8gemm_nt`** microkernel: AVX-VNNI **`vpdpbusd`** (one instruction folds 32 `u8×i8`
products into the `i32` lanes *and* accumulates), register-blocked four B-rows at a time (the A-row
chunk loaded once and reused across the four dots, four accumulator chains for ILP); the
single-threaded path additionally pairs **two A-rows into a 2×4 register tile**, so each B-row load
feeds both rows and B-matrix traffic is halved (the lift that takes single-core from ~1.3× to
~1.5–2.5×). C and Rust run the idiomatic naive int8 GEMM at `-O3 -march=native` / `-O
-Ctarget-cpu=native` — gcc auto-vectorizes it to `vpdpbusd` as well, so this is an honest
*same-instruction* comparison, not a straw man.

**Because the math is integer, the cross-language check is bit-exact** (not a tolerance): `i32` add is
associative mod 2³² and `vpdpbusd` is non-saturating, so Wukong, C, and Rust must agree on every one
of the `M·N` outputs — and they do.

Reported as int8 **GOP/s** (2 ops per multiply-accumulate) and the clock-invariant Wukong-vs-C ratio
across runs:

| size | Wukong 1-core vs C | Wukong `@parallel` vs C | Rust |
|------|---------------------|--------------------------|------|
| 512×512   | **~1.5–1.7× faster** | **~4.6–8.5× faster** | ~5–10× slower than Wukong 1-core |
| 1024×1024 | **~2.2–2.5× faster** | **~8.0–14.7× faster** | ~8–12× slower than Wukong 1-core |

**Absolute throughput swings ~2–3× with the laptop's clock/thermal state, so the clock-invariant
ratio is the reported quantity** (as with the f32 GEMM's roofline %): a high-clock run measured
Wukong ~290–347 GOP/s single-core against gcc ~133–226, a cooler run roughly half of each — but the
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
reduction in `f32` (full precision). Wukong recognizes a `s += (x[k] as f32) [* (y[k] as f32)]`
loop over `[bf16; _]` arrays and dispatches it to **`wukong_dot_bf16`** / **`wukong_sum_bf16`** —
widen bf16→f32 (a `<<16` bit-extend, F16C-class) and accumulate across 8 f32 lanes. C and Rust run
the idiomatic `<<16` widen + accumulate at honest default flags: gcc/rustc may vectorize the *widen*,
but without `-ffast-math` they keep the f32 reduction **sequential** — the same basis as the f32 `dot`
kernel. All-positive inputs keep the reduction well-conditioned, so the cross-language scalar result
agrees within a tight tolerance (the three reassociate the f32 sum differently).

GB/s (higher is better) — input traffic over `[bf16; N]` arrays:

| op | N=2²⁰ Wuk · C · Rust | N=2²⁴ Wuk · C · Rust | Wukong vs C |
|----|----------------------|----------------------|--------------|
| dot Σx·y | **10.2** · 3.4 · 3.2 | **10.2** · 2.9 · 3.0 | **~3.0× → 3.5×** |
| sum Σx   | **12.3** · 1.5 · 1.7 | **10.5** · 1.7 · 1.7 | **~8.3× → 6.1×** |

> **Corrected 2026-08-04 — this family is a LOSS on a like-for-like basis.** This bench had **no
> `C(fast)` column at all** until the peer audit, even though the whole point of its Wukong kernel is
> an 8-lane *reassociated* f32 accumulate — i.e. it published a reassociating kernel against a peer
> forbidden from reassociating, which is precisely the asymmetry the `C(fast)` rule exists to remove.
> With the column present (2026-08-04, same-run adjacent, AC+charging):
>
> | op | N | Wukong vs C | Wukong vs **C(fast)** |
> |---|---|---|---|
> | dot Σx·y | 2²⁰ | 7.6× | **1.36–1.81× slower** |
> | dot Σx·y | 2²⁴ | 6.4× | **1.09–1.16× slower** |
> | sum Σx   | 2²⁰ | ~8× | **1.22× slower** |
>
> The plain-C multiple survives the peer fix (the peer's loop was already the natural one), but it is
> entirely the withheld `-ffast-math`: once gcc is allowed the same reassociation Wukong's kernel
> takes, it wins. The bandwidth argument below is still correct as *physics*; it is not a win over C.

The **dot** win (~3×) tracks the f32 `dot` — Wukong vectorizes the reduction while C/Rust stay serial.
The **sum** win is larger (~6–8×) because C's unary f32 sum is a single dependency chain (pure
latency, no product to fill the pipeline) at ~1.5 GB/s, while Wukong's 8-lane SIMD sum reaches
~12 GB/s. The **dot** lead **widens from 2²⁰ to 2²⁴** as the working set spills L3 and the halved byte count
(bf16 vs f32) starts to dominate — the bandwidth payoff of mixed precision (the sum lead, already
bandwidth-bound, instead narrows ~8.3→6.1× across the same range). Correctness: bf16 storage
is bit-exact across the interpreter and native backends (`round_to_bf16` emits the identical integer
arithmetic as the interpreter's `round_bf16`), and both call the identical reduction kernel, so the
differential gate stays exact for *fractional, non-bf16-exact* inputs across `-O0`/`-O2`/`-O3`
(`differential_bf16_reduce`); `tests/run/reduce_bf16.wk` pins the e2e value.

The same dispatch is now **symmetric for `f16`** (widened with the F16C `vcvtph2ps` instruction
instead of the bf16 `<<16`; f16 storage is bit-exact via shared `half`-crate shims, since Cranelift
x64 has no f16 convert lowering) and **extended across the op surface** for both precisions:
**`max`/`min`/`absmax`** (`wukong_reduce_{bf16,f16}` — the per-tensor absmax is the symmetric int8
quant scale; exact, since max/min round nothing), **streaming `axpby`** (`out = a·x + b·y`,
half-in/f32-out, ~1.3× ≫ L3), and the **36-op activation set** (`wukong_vmath_{bf16,f16}`, where the
cheap ops gain bandwidth and the transcendentals keep the full libm-vectorization win — C can vectorize
neither the `libm` call nor the half→f32 widen). e2e: `tests/run/{f16,reduce_f16,reduce_bf16_minmax,
vmath_{bf16,f16},axpby_f16}.wk`; all bit-exact interp == native.

#### bf16 / f16 `nn.Linear` — the mixed-precision matmul Wukong dispatches, gcc leaves scalar

The dominant modern transformer matmul: `C = A·Bᵀ` with **bf16/f16 inputs and an f32 accumulator**.
Wukong recognizes the half-precision dot-product nest (`s += (a[..] as f32) * (b[..] as f32)` over
`[bf16]`/`[f16]` arrays) and folds it to one **`wukong_sgemm_bf16_nt`** / **`_f16_nt`** call: a
lossless widen prepass (O(m·k + n·k), ~1/n of the GEMM) feeding the *identical* tuned AVX2 f32 GEMM.
C and Rust store bf16 as `uint16_t` and widen each element inline inside the triple loop.

| size  | Wuk 1-core | Wuk @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|-------|-----------|---------------|---------|------|-------------|---------------|
| 512²  | ~52–53 | ~98–103 | ~2.1 | ~2.2 | **~24–25×** | **~47×** |
| 1024² | ~50–55 | ~208–219 | ~2.0 | ~2.1 | **~25×** | **~98–109×** |

*IEEE-serial C baseline; see the `C(fast)` column for the reassociation-normalized comparison.*

> **Corrected 2026-08-04.** The `C(fast)` column was missing here too; with it present the durable
> figure is **4.59× (512²) and 5.62× (1024²)**, against **22.70× / 27.52×** vs the plain-flags peer.
> The paragraph below used to *estimate* the durable win at "~10× even against optimized bf16 C" from
> a hand-argued ~4–5 GFLOP/s baseline; that estimate was never measured, and the measured
> reassociation-normalized answer is 4.6–5.6×.

Two effects compound here and honesty requires separating them. The idiomatic bf16 C falls to ~2–3.4
GFLOP/s because the inline `bf16→f32` widen won't vectorize **and** the dot-product reduction stays
serial (no `-ffast-math`). Allowing gcc the reassociation — the `C(fast)` column — lifts it to ~16
GFLOP/s, and the durable domain-lowering win over that is the measured **4.6–5.6×**; the ~23–28×
headline is versus the code a person actually writes at honest default flags. Wukong's single-core ~52 GFLOP/s is ~73% of the measured AVX2-FMA roofline —
the f32 GEMM efficiency, since after the widen it *is* the f32 kernel. Without this dispatch the half
nest would fall to a scalar widening loop (the `as f32` casts block the f32 matmul recognizer). The
widen is lossless, so the kernel equals the nest under the documented matmul reassociation, bit-for-bit
across backends (`tests/run/linear_{bf16,f16}.wk`; the runtime twin test pins bf16/f16 == f32-on-
widened-operands and serial == parallel). Single-run, clock-sensitive absolute GFLOP/s; the ratio is
the stable part.

**Fused FFN epilogue (`C = act(A·Bᵀ + bias)`) — bias + activation for free.** The mixed-precision
transformer FFN projection — the half matmul immediately followed by its bias-add and an
identity/ReLU/GELU/SiLU activation — folds into one **`wukong_sgemm_{bf16,f16}_nt_epi`** call: the
lossless widen prepass feeds the f32 GEMM, which applies the bias + activation **in its C-tile
writeback** instead of paying a separate read-modify-write pass over C. So the FFN runs at the bf16/f16
GEMM throughput above (the epilogue is O(m·n), ~1/k of the matmul, and adds *no* extra memory pass) —
the bias and activation cost ≈0. This is the same fusion the int8 dequant epilogue exploits and the one
cuBLAS/oneDNN structurally can't express (they emit the GEMM, then a separate bias/activation kernel
over the full output). Like the f32 `nt_epi`, the bias may be null (the bias-free SwiGLU projection).
GELU/SiLU reuse the `vmath` scalar forms, so the fused result equals the unfused matmul → bias →
activation bit-for-bit (`tests/run/linear_{bf16,f16}_ffn.wk`; the runtime twin pins it == the f32
fused FFN on the widened operands across all four activations × bias on/off, serial == parallel).

**Mixed-precision weight-gradient (`C = Aᵀ·B`) — the training backward GEMM, in bf16/f16.** The
`dW = dYᵀ·X` weight gradient (A stored `[k, m]`, the contraction the *outer* index of A's storage) is
recognized in bf16/f16 too — `wukong_sgemm_{bf16,f16}_tn[_parallel]`, the half twin of the f32
`wukong_sgemm_tn`. It widens A and B losslessly then delegates to that exact f32 TN kernel (transpose
A once → the tuned `C = A·B` microkernel), so it is bit-for-bit the f32 TN GEMM on the widened values.
The naive half TN nest loses **twice** in C/Rust — the inline `bf16→f32` widen won't vectorize *and*
A's column-strided reads (the contraction is A's outer storage index) defeat vectorization one cache
line per element — so this compounds the bf16 widen win with the transpose-prepass win the f32
weight-gradient GEMM already documents (`~42–445×` idiomatic C there). `tests/run/matmul_{bf16,f16}_tn.wk`;
the runtime twin pins both precisions == the f32 TN kernel on the widened operands, serial == parallel.

### Matrix transpose — a tie once the peer is blocked too

`dst = srcᵀ` is the memory-bound layout op behind attention score transposes and weight-layout
conversions. Wukong folds the nest to the `B=32` cache-blocked **`wukong_transpose_f32`**, which
keeps a `B×B` tile of both operands L1-resident. Transpose is a *permutation* (no arithmetic), so the
cross-language check is **bit-exact** — a stronger bar than the GEMM tolerance gate.

> **Corrected 2026-08-04.** The C/Rust peers were the *unblocked* `for i { for j { dst[j*R+i] =
> src[i*C+j] } }`, and this section's own explanation of the win was "**gcc/rustc do not loop-tile a
> transpose at `-O3`**". That is true, and it is not a statement about codegen — it is a statement
> about the benchmark's source. Loop blocking a transpose is the textbook optimization; a competent C
> programmer writes the tiles, and comparing a blocked kernel against an unblocked peer measures the
> blocking, which Wukong did not invent here. The peers are now 32×32 blocked, matching Wukong's own
> algorithm. Isolated at `-O3 -march=native`, kernels in their own translation unit, 2048²:
> **29.51 ms naive → 24.31 ms naive + `restrict` → 14.09 ms 32×32-blocked + `restrict` — a 2.09×
> total handicap.**

Measured with the blocked peer (2026-08-04, same-run adjacent, AC+charging — all-core directional):

| size  | 1-core vs C | `@parallel` vs C | `@parallel` vs C(omp) | was (1-core / `@parallel`) |
|-------|-------------|------------------|------------------------|-----------------------------|
| 1024² | **1.03×**, 1.17× round 2 (tie) | 4.77× | 1.00× (tie) | ~~1.48× / 7.74×~~ |
| 2048² | **1.00×**, 1.05× round 2 (tie) | 7.05× | 1.12× | ~~2.20× / 14.83×~~ |

(GB/s = `2·N²·4` bytes moved per call.) The single-core cache-blocking "win" was the peer's missing
blocking: at equal algorithms it is a **dead tie**, and gcc's scalar tile-mover is as good as
Wukong's SIMD one on a latency-bound permutation. What survives is the `@parallel` column — 4.8–7.1×
vs single-threaded C, and 1.00–1.12× vs an all-core OpenMP peer running the *same blocked* nest,
i.e. Wukong's automatic parallelization roughly matches hand-written OpenMP. That is a real but
modest result, and it is a threading result, not a codegen one. `tests/run/transpose_f32.wk`; the
runtime test pins the blocked kernel == the naive transpose exactly (a permutation, serial ==
parallel).

### Column reduction — a published ~29–50× that was the peer's loop order

`out[j] = Σ_i x[i,j]` reduces a `[rows, cols]` matrix down its **outer** axis — the bias gradient
`db = Σ_batch dY`, the batch sum/mean, a reduce-along-axis-0. Wukong folds the nest to
**`wukong_colsum_f32`**, which streams `x` row-major and accumulates eight columns at a time into a
cache-resident `out[]` (`out[j..j+8] += x[i, j..j+8]`). Each `out[j]` sums `x[0,j], x[1,j], …` in
`i`-ascending order — identical to the scalar twin and to the disjoint-stripe `@parallel` form — so
the cross-language check is **bit-exact**.

> **Corrected 2026-08-04 — this is the largest error in this document.** The C/Rust peers were
> written **column-outer**:
>
> ```c
> for (long j = 0; j < N; j++) { float s = 0.0f; for (long i = 0; i < M; i++) s += x[i*N+j]; out[j] = s; }
> ```
>
> For a row-major matrix that is the worst loop order available: the inner loop advances by `N`
> floats per step, so every element costs a fresh cache line, and gcc leaves the strided fold
> entirely scalar. It was also the *only* order measured. The paragraph that used to close this
> section said, in as many words, "the *row*-outer spelling already auto-vectorizes — the
> column-outer form is the gap", and then published the column-outer multiple as Wukong's win.
>
> The natural spelling — the one any competent C programmer writes, and the one Wukong's own kernel
> implements internally — is row-outer:
>
> ```c
> for (long j = 0; j < N; j++) out[j] = 0.0f;
> for (long i = 0; i < M; i++) for (long j = 0; j < N; j++) out[j] += x[i*N+j];
> ```
>
> It folds each column in the identical i-ascending order (so the bit-exact cross-check still
> holds — verified, sum |Δ| = 0) and it auto-vectorizes. Standalone at `-O3 -march=native`,
> kernels in their own translation unit, 4096×1024: **21.33 ms column-outer → 5.03 ms column-outer +
> `restrict` → 0.58 ms row-outer + `restrict` — a 36.7× total handicap.**

Measured with the corrected peer (2026-08-04, same-run adjacent pairs, AC+charging — the all-core
column is directional only). **>1 means Wukong is faster; a negative sign means Wukong is slower:**

| op | size | 1-core vs C | 1-core vs C(fast) | `@parallel` vs C | was (1-core) | inflation |
|----|------|-------------|-------------------|------------------|--------------|-----------|
| sum      | 1024×1024 | **0.81× (1.23× slower)** | 1.31× slower | 1.95× slower | ~~77.7×~~ | **63×** |
| sum      | 4096×1024 | **0.59× (1.70× slower)** | 1.55× slower | 1.15× faster | ~~25.5×~~ | **15×** |
| max      | 1024×1024 | 1.40× slower | 1.91× slower | 1.68× slower | ~~22.7×~~ | 16× |
| max      | 4096×1024 | 1.72× slower | 1.99× slower | 1.25× faster | ~~31.4×~~ | 18× |
| min      | 1024×1024 | 1.37× slower | 1.58× slower | 1.91× slower | ~~65.7×~~ | 48× |
| min      | 4096×1024 | 1.81× slower | 1.26× slower | 1.19× slower | ~~23.6×~~ | 13× |
| abs-max  | 1024×1024 | 1.57× slower | 2.01× slower | 1.63× slower | ~~43.7×~~ | 28× |
| abs-max  | 4096×1024 | 1.05× slower | 1.22× slower | 1.03× slower | ~~44.0×~~ | 42× |
| mean     | 1024×1024 | 1.60× slower | 1.71× slower | 1.71× slower | ~~48.7×~~ | 30× |
| mean     | 4096×1024 | 1.26× slower | 1.35× slower | 1.29× slower | ~~23.2×~~ | 18× |
| sum-sq   | 1024×1024 | 1.57× slower | 1.69× slower | 2.00× slower | ~~73.5×~~ | 47× |
| sum-sq   | 4096×1024 | 1.46× slower | 1.44× slower | 1.15× slower | ~~24.4×~~ | 17× |
| L2       | 1024×1024 | 1.08× slower | 1.14× slower | 4.91× slower | ~~56.5×~~ | 52× |
| L2       | 4096×1024 | 1.43× slower | 1.58× slower | 1.04× slower | ~~26.3×~~ | 18× |
| RMS      | 1024×1024 | 1.25× slower | 1.52× slower | 1.65× slower | ~~49.6×~~ | 40× |
| RMS      | 4096×1024 | 1.58× slower | 1.87× slower | 1.37× faster | ~~35.7×~~ | 23× |

**There is no column-reduction win.** Against a peer written the way the operation is normally
written, `wukong_colsum_f32` and its six siblings land between **a tie and 1.8× slower** than what
gcc emits for the row-outer nest. 15 of the 16 rows above are losses; a second independent A/B round
(same method, later in the session) reproduced the collapse everywhere — colsum 1024×1024 read
**56.78× → 1.51×** and colsum 4096×1024 **43.62× → 1.64× slower** — with `colsum` at the L3-resident
1024² shape the one row whose *sign* moves between rounds (1.23× slower, then 1.51× faster), i.e. a
tie. The >L3 4096×1024 shape is a consistent ~1.65× loss in both rounds.
`@parallel` does not rescue it either: it is between 1.9× slower and 1.4× faster, i.e. all-core
Wukong roughly ties single-threaded C. The
kernels remain correct (`tests/run/colsum.wk` etc. still pin them bit-exact against the scalar
fold, and the cross-language check still passes element-for-element); they are simply not faster
than the compiler on this operation, and the "strided column-outer fold gcc/rustc leave *scalar*"
explanation was a description of the benchmark's own source code.

*This family is now a documented **loss**, and an open optimization lever: the recognizer fires (see
`--emit=mir -O2` for `wukong_colsum_`), so the gap is in the kernel, not in the dispatch.*

### Softmax backward — vectorizing the per-row dot

`dx[r,i] = y[r,i]·(dy[r,i] − Σ_j y[r,j]·dy[r,j])` is the gradient through a row softmax — the backward
pass of every attention block and classification head. Each row is a **dot** `s = Σ y·dy` followed by an
elementwise `y·(dy − s)`. gcc/rustc load and multiply `y·dy` wide but — verified — keep the
**accumulation scalar** (a serial `vaddss` dependency chain, no `ymm` accumulator, because they won't
reassociate the float sum), which is latency-bound (~one add per 4 cycles). Wukong folds the `[R,C]`
nest to **`wukong_softmax_bwd_f32`**, which delegates the dot to the proven bit-exact
`wukong_sreduce_f32` (eight *independent* lane accumulators, no dependency chain) then applies
`y·(dy − s)` 8-wide. This is a **different gap** from the column reductions (the per-row dot, not a
strided access), so the win is more modest — gcc already vectorizes the apply:

| size | Wuk 1-core | Wuk @parallel | C (gcc) | Rust | 1-core vs C | parallel vs C |
|------|-----------|---------------|---------|------|-------------|---------------|
| 1024×1024 | ~45–49 | ~110–124 | ~24 | ~23 | **~1.9–2.0×** | **~4.5–5.2×** |
| 4096×512 | ~22–26 | ~133–135 | ~21–22 | ~21 | **~1.0–1.2×** | **~6.1×** |

(GB/s = `3·R·C·4` — the two reads + one write; higher is better. *IEEE-serial C baseline; see the
`C(fast)` column for the reassociation-normalized comparison.*) Single-core is a modest win-to-tie
(only the dot's accumulation is recovered); `@parallel` (rows across cores) scales to **~5–6×** on top.
The dot reassociates (lane accumulators vs the C baseline's serial chain — the documented
reassociated-reduction exception), so the cross-language check is a magnitude-normalized tolerance, not
bit-exact; the differential gate (interp == native, both running this kernel) *is* bit-exact, and the
runtime test pins the kernel to its delegated-dot reference and serial == parallel. `tests/run/softmax_bwd.wk`.

### Activation backward — the 256-bit transcendental gradient

`dx[i] = dy[i]·act'(x[i])` is the elementwise gradient through an activation — the backward of every
FFN/attention nonlinearity in training. The derivative is itself a **transcendental**: `silu'` and
`sigmoid'` fold a sigmoid, `gelu'` and `tanh'` fold a tanh — each an `expf` that C/Rust call as scalar
`libm` inside the loop, so the loop **cannot vectorize** (exactly the wall the forward activation
dispatch clears). Wukong recognizes `dx[i] = act_backward(x[i], dy[i])` and folds it to one **256-bit**
`wukong_vmath2_f32` call (`act_backward` ∈ {`silu`,`gelu`,`sigmoid`,`tanh`,`elu`,`softplus`}, new
two-input op codes on the same kernel as `pow`/`atan2`/`hypot`), fusing the upstream `dy·` multiply into
the derivative — one pass.

| backward | 1-core vs scalar C | `@parallel` vs C | derivative |
|----------|--------------------|------------------|------------|
| `silu_backward`     | **~5.1–5.5×** | ~9.6–11×   | `s + x·s·(1−s)`, `s=σ(x)` |
| `gelu_backward`     | **~7.3–8.1×** | ~15.6–20×  | tanh-approx `g'`; more transcendental work → wider gap |
| `sigmoid_backward`  | **~5.4–5.8×** | ~9–12×     | `σ(x)·(1−σ(x))` — the logistic gate |
| `tanh_backward`     | **~11–12.5×** | ~21.8–24.6× | `1−tanh²(x)` — the largest; `tanhf` is costly scalar, trivial vectorized |
| `elu_backward`      | **~3.0×**     | ~5.5×      | `x>0 ? 1 : eˣ` — the most modest (only x≤0 folds `exp`) |
| `softplus_backward` | **~5.8×**     | ~8.3×      | `σ(x)` — softplus' is the sigmoid (VAE/flow/Mish nets) |

(N = 2²⁰, the `(x, dy, dx)` three-pointer harness; absolute GB/s swings with the laptop clock, so the
clock-invariant **ratio** is reported.) The derivative is **pure elementwise — no reduction** — so the
kernel is bit-identical lane-for-lane and the differential gate (interp == native, −O0 == −O3) is trivial
(not even the reassociation exception softmax-backward needs); the cross-language check is a
magnitude-normalized tolerance only because the poly sigmoid/tanh differs from C's `libm` by ~1 ULP. The
scalar twin, the AVX2 lanes, and the inlined-MIR fallback share one op sequence, so a dispatched loop, a
`while`-loop fallback, and a standalone call all agree bit-for-bit.
`tests/run/{silu,gelu,gate,elu_softplus}_backward.wk`. (The non-transcendental backwards — `relu`,
`leaky_relu` = a select on `x>0` — are **deliberately not added**: gcc/rustc vectorize those, so Wukong
would only tie. The family is exactly the activations whose *derivative* is transcendental.)

### Argmax / argmin — the classification top-1 the `(value,index)` bookkeeping won't let gcc vectorize

`out = {argmax,argmin}(x)` returning the **index** of the extreme element — the classification head /
greedy-decode top-1, and the per-channel selection in routing/quant. Wukong recognizes two batched
shapes and folds each to one i32-output kernel that tracks 8 `(value, index)` lanes via `_mm256_blendv_ps`
(strict compare → lowest index wins on a tie):

- **Per-row** `out[r] = argmax_j x[r,j]` → `wukong_rowarg{max,min}_i32`. The within-row `(value,index)`
  scan keeps **both** gcc and rustc scalar — and, newly measured, keeps them scalar even at
  `-ffast-math`, so this win is real on a reassociation-normalized basis too. The peer here already
  walked each row contiguously, so the only change was `restrict` + the new `C(fast)` column, and the
  ratio barely moved (2026-08-04, same-run adjacent, AC+charging):

  | shape | argmax 1-core (vs C / vs C(fast)) | argmin 1-core (vs C / vs C(fast)) | `@parallel` vs C |
  |---|---|---|---|
  | 1024×1024  | **2.45× / 1.92×** (was 3.35×) | **3.37× / 3.27×** (was 3.41×) | 8.5–9.2× |
  | 4096×1024  | **4.85× / 3.50×** (was 3.18×) | **2.29× / 2.08×** (was 3.45×) | 5.8–12.1× |

- **Per-column** `out[j] = argmax_i x[i,j]` (axis-0) → `wukong_colarg{max,min}_i32`.
  **Corrected 2026-08-04 — this is now a LOSS.** The C/Rust peers scanned column-outer, reading `x`
  with stride `C`; the natural spelling for an axis-0 arg-reduction is row-outer over a `C`-long
  running-best vector (what NumPy's `argmax(axis=0)` does internally, and what Wukong's own kernel
  does). Isolated at `-O3 -march=native`, kernels in their own TU, 4096×1024 argmax: **9.87 ms
  column-outer vs 1.57 ms row-outer + `restrict` — a 6.3× handicap**, with identical indices on
  every column.

  | shape | argmax 1-core (vs C / vs C(fast)) | argmin 1-core (vs C / vs C(fast)) | was (argmax / argmin) |
  |---|---|---|---|
  | 1024×1024  | **1.49× slower / 1.98× slower** | **1.68× slower / 1.86× slower** | ~~3.09× / 2.69×~~ |
  | 4096×1024  | **2.47× slower / 2.46× slower** | **1.71× slower / 1.98× slower** | ~~4.10× / 3.48×~~ |

  So the single-pass rewrite this section described did happen and did help, but the surviving margin
  over a correctly-written peer is negative: gcc's row-outer arg-scan is 1.5–2.5× faster than the AVX2
  kernel. The prior text's claim that "gcc *does* vectorize the column arg-scan … but Wukong's single
  DRAM pass beats it outright" was measured against the strided peer and does not hold. The
  bit-exactness is unchanged: the strict compare keeps the lowest-row tie-break and rows are scanned
  i-ascending, so the kernel still equals the scalar twin (the cross-check is exact index equality and
  still passes).

- **Global** `out = argmax(x)` over a flat array → `wukong_argreduce_f32`. This was the one memory-bound
  reduction Wukong *lost* (it had no AVX2 path, so gcc's branch-predicted scalar loop won by 1.30×). It
  now folds 32 elements/iteration across **4 AVX2 accumulators** (8 `f32` value + 8 `i32` index lanes
  each, `_mm256_cmp_ps` strict-compare + `blendv`), collapsing the 32 candidates through the same scalar
  tie-break — so it is bit-identical to the scalar form and **~5–9× faster than gcc** (the `(value,index)`
  bookkeeping gcc/rustc won't auto-vectorize), up from a 1.30× loss. Under `@parallel` it now dispatches
  to `wukong_argreduce_f32_parallel` (previously it fell through to the *serial* kernel — the parallel
  reduction recognizer handles only `+`/`fmax`/`fmin`, not the argmax bookkeeping), folding the same
  fixed `RCHUNK` chunks in ascending order so the index stays bit-identical: **~17× vs single-threaded C**
  (memory-bound, so ~2× over the single-core fold rather than linear in cores).

These are the **first recognized kernels with an i32 output buffer** (the interpreter marshals the result
back as an integer, not a float). A per-row/column arg-selection is a deterministic permutation — no
reassociation — so the differential gate is bit-exact and the cross-language check is **exact** (the
output indices match bit-for-bit, reinterpreting the harness's f32 slots as i32), a stronger bar than the
float kernels' tolerance. `tests/run/{rowargmax,colargmax}.wk`.

### Scans — the prefix computations gcc/rustc can't auto-vectorize at all

A scan `out[i] = ⊕(out[i-1], x[i])` is a **loop-carried dependency**: each output needs the previous one,
so gcc `-O3 -march=native` and rustc leave the whole thing scalar (no auto-vectorization is possible from
the naive recurrence). Wukong recognizes the per-row scan and folds it to a **SIMD Hillis-Steele in-lane
scan + per-row carry** (`_mm256_permutevar8x32_ps` cross-lane shift, three shift-`⊕` steps per 8-block,
then a broadcast-carry fold). Measured single-core / `@parallel` vs naive C (C ≡ Rust — both scalar):

| scan | 1024×1024 | 4096×1024 | fold | cross-check |
|------|-----------|-----------|------|-------------|
| **cumsum** (prefix sum) | 1.72× / 6.39× | 1.37× / 7.29× | `+` | tolerance (in-lane tree reassociates) |
| **cummax** (running max) | **2.87× / 11.6×** | **2.28× / 12.1×** | `fmax` | **bit-exact** |
| **cummin** (running min) | **2.88× / 11.8×** | **2.25× / 12.1×** | `fmin` | **bit-exact** |
| **cumprod** (prefix product) | **1.77–1.89× / 6.0×** | **1.77× / 7.0–7.8×** | `*` | **bit-exact** |

`cumprod` (prefix product) uses a *different* lever than the three above: a bare product is not amenable
to the same in-lane reassociation trade-off, so instead of Hillis-Steele it scans **4 independent rows
interleaved** (the `lrscan` lever — four `mul` chains in flight to fill the ports the single serial chain
leaves idle). Because a product is never fused (no `fma` contraction), each row folds strictly
left-to-right == the scalar C/Rust nest, so the cross-language check is **bit-exact**, not toleranced.

`cummax`/`cummin` lead `cumsum` single-core because gcc's scalar `fmax`/`fmin` recurrence pipelines worse
than its dependent add chain — and because max/min **select** an input value (no float arithmetic), the
SIMD tree fold gives *exactly* the left-to-right scan, so the cross-language check is bit-for-bit (no
tolerance). `cumsum`'s in-lane tree reassociates the float add, so it takes the documented reduction
exception (both backends run the identical kernel → interp == native; the cross-check is magnitude-
normalized, since a mean-zero prefix sum is a random walk that crosses zero where a pointwise ratio
divides by ≈0). GB/s = `R·C·4·2` (read x + write out); rows are independent, so `@parallel` is bit-equal
to serial. `tests/run/{cumsum,cummax}.wk`.

**Linear-recurrence / selective scan (SSM / Mamba).** A first-order recurrence `h_t = a_t·h_{t-1} + b_t`
— the state-space step at the heart of Mamba / S4 / RWKV linear attention, and the EMA — is also
loop-carried, so gcc `-O3 -march=native` and rustc emit **one serial `mul`+`add` chain per row**:
*latency-bound* at a few GB/s, far below DRAM. But the **rows are independent**, so Wukong's
`wukong_lrscan_f32` scans **4 rows interleaved**, keeping four chains in flight to fill the ports the
single chain leaves idle (plain `mul`+`add`, deliberately *not* `f32::mul_add` — on a build without the
`fma` target feature the latter lowers to a libm `fmaf` **call** that re-serializes the chain and erases
the ILP: measured `mul_add` 0.8× vs C, plain `mul`+`add` 1.8×). gcc/rustc may not legally re-order an f32
recurrence across rows without `-ffast-math`, so this 4-way ILP is a genuine **single-core win**:

| scan | 1024×1024 | 4096×1024 | cross-check |
|------|-----------|-----------|-------------|
| **lrscan** `h_t = a_t·h_{t-1} + b_t` | **1.75–1.92× / 5.3–7.6×** | **1.56–1.84× / 7.2–7.7×** | tolerance (gcc may fuse to one `fma`; ~1 ULP) |

Single-core (4-row ILP) beats gcc's serial chain; `@parallel` maps independent row *chunks* across cores
on top. Bit-exact across backends — each row stays its own in-order recurrence, so the 4-row interleave
changes only *issue order*, never a row's arithmetic (no reassociation). GB/s = `3·R·C·4` (read a + read b
+ write out). `tests/run/lrscan.wk`.

**Scatter-add / embedding-gradient backward** (`grad_w[ids[t], :] += grad_out[t, :]`, the dual of the
embedding gather, run every LLM training step) is the one place the headline is **correctness, not a
speed ratio**. Single-core is an honest memory-bound tie (both vectorize the contiguous row read-
modify-write). The structural win is **deterministic parallelism**: a C author splits the *tokens* across
threads and lets colliding ids race into shared `grad_w` rows → **atomic** float adds, slow *and*
non-deterministic in float order (low bits wobble run to run). Wukong's `wukong_scatter_add_f32_parallel`
splits the **output rows** instead — each core owns a disjoint `grad_w` range, scans all tokens, accumulates
only its rows → lock-free, race-free, and **bit-identical to serial** regardless of thread count. The
recognizer recovers the real table height `V` from `grad_w`'s sema array length, and folds collisions in
ascending token order (== the scalar nest), so the differential gate is exact. `tests/run/scatter_add.wk`.

### Single-threaded elementwise & reductions

A recognized streaming map (`out[i] = act(a·x[i] (+ b·y[i]) + c)`) dispatches to the **256-bit AVX2
`wukong_velem_f32`** kernel and a Horner polynomial to **`wukong_vhorner_f32`** — both 4×-unrolled,
both emitting **non-temporal stores** once the working set spills L3 (the store path gcc/rustc will
not emit, skipping read-for-ownership traffic). This turns the former bandwidth-bound *ties* into
wins:

| kernel | Wukong vs C | notes |
|--------|--------------|-------|
| saxpy  | **~1.25–1.45× faster** | `velem` 256-bit + non-temporal store (3-stream, spills L3) — Rust ~1.4× behind too |
| relu   | ≈tie (~1.0×) | 2-stream, L3-resident at N=2²⁰ so stores stay cacheable; both are bandwidth-bound (an [identity-affine fast path](crates/wukong_runtime/src/velem.rs) drops the wasted `fma(1·x+0)` so it no longer trails gcc); the win shows at >L3 (~1.4×, below) |
| poly   | **~1.1–1.2× faster** | `vhorner` AVX2 Horner, **six** independent chains (was four — a 5-deep dependent FMA chain needs ~8 in flight to fill both ports); now a consistent win over gcc's own 256-bit autovec where four chains only tied |
| fused linear→relu | ≈tie (±10%, clock-dependent) | matvec-bound (M=1) at the bandwidth wall; Wukong fuses the two source loops |
| dot    | **~2.9× faster** | reduction reassociated to lane accumulators; gcc/rustc stay serial |
| ssd (Σ(x−y)²) | **~2.6–2.9× faster** | same — an L2-loss reduction |

*dot/ssd: IEEE-serial C baseline; see the `C(fast)` column for the reassociation-normalized
comparison.*

**At real (>L3) activation-tensor sizes the lead widens** — non-temporal stores avoid the RFO traffic
that dominates when nothing fits in cache. At N=2²⁴ (64 MiB/array):

| kernel (N=2²⁴) | Wukong vs C | Wukong vs Rust |
|--------|--------------|--------|
| saxpy  | **~1.4× faster** | ~1.4× |
| residual (x+y) | **~1.3–1.4× faster** | ~1.4× |
| scale (a·x) | **~1.3× faster** | ~1.4× |
| relu | **~1.5–1.6× faster** | ~1.7× |

A `@parallel` reduction goes further: it dispatches to a deterministic multicore reduction kernel
(`wukong_sreduce_f32_parallel`), spreading the stream across cores to reach *aggregate* bandwidth —
see the `dot@parallel`/`ssd@parallel`/`max@parallel` rows below. The same kernel folds **max/min** (by
`fmax`/`fmin`), so a tensor's `[min, max]` range (asymmetric int8 quantization → `scale=(max−min)/255`)
or per-tensor max (softmax stability) computes across cores too.

### Auto-parallel runtime — Wukong heavily exceeds idiomatic single-threaded C/Rust

`@parallel` lowers the loop to a multicore dispatch whose per-thread chunk is itself vectorized (or,
for a reduction, dispatched to the multicore reduction kernel). These kernels are memory-bound, so the
parallel speedup is limited by *aggregate* bandwidth, not core count — still a clear win over
single-threaded C:

| kernel | Wukong vs single-threaded C |
|--------|------------------------------|
| saxpy@parallel | **~2.4–2.7×** (~132 GB/s, near the chip's memory-bandwidth ceiling) |
| poly@parallel  | **~2.2–2.4×** |
| relu6@parallel | **~6.7–8.0×** (nested branch defeats gcc's vectorizer; Wukong if-converts + parallelizes) |
| dot@parallel   | **~7.9×** (~135 GB/s) — reduction across cores; C/Rust keep it serial & latency-bound |
| ssd@parallel   | **~8.6×** (~144 GB/s) — L2-loss reduction across cores |
| max@parallel   | **~25–26×** (~85 GB/s) — per-tensor max (int8-quant range / softmax stability) across cores; C's single-stream float-max chain is especially latency-bound (~3 GB/s) without `-ffast-math` |
| absmax@parallel | **~25×** (~80 GB/s) — per-tensor max\|x\| (symmetric int8-quant scale) across cores; `abs` is free (a bitwise op) so C stays latency-bound like `max` (~3 GB/s) |

*All rows compare Wukong-multicore to single-threaded C (disclosed); see the `C(omp)` column for the
multithreaded-C comparison, and — on the reduction rows (dot/ssd/max/absmax/argmax) — the `C(fast)`
column for the reassociation-normalized single-thread baseline.*

## GPU backend (NVIDIA RTX 4050 Laptop, `sm_89`)

Wukong has a **GPU backend** (`wukong_codegen_gpu`, behind `--features gpu`). Being a compiler, it
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

**int8 (W8A8) tensor-core GEMM — the quantized-inference path (M3).** `u8` activations × `i8` weights
→ `i32`, `C = A·Bᵀ` (Wukong's CPU `vpdpbusd` contract, on the GPU). int8 shares fp8's `m16n8k32`
8-bit fragment geometry (no WMMA int8 on `sm_89` either), so the kernels mirror the fp8 stack retyped
`mma.sync.m16n8k32.s32.u8.s8.s32` with `i32` accumulators: a hand-placed single tile, a fragment-reuse
`_mt` (2×4 16×8 tiles/warp), and a **SMEM-staged + `cp.async` double-buffered** kernel (64×64 / 128×128
tiles, fragments loaded from shared via `ld.shared`). Integer accumulate is exact mod 2³², so the gate
is **bit-exact** — the GPU output must *equal* a wrapping-`i32` CPU reference over the full output (`==`,
not a tolerance), a stronger correctness bar than the float kernels. Gated green across single-tile,
`_mt`, and SMEM-staged paths.

Honest same-run scoreboard (`int8_gemm_vs_peers`, RTX 4050, `2·M·N·K` MAC-FLOP, checksum-cross-checked;
peers: **naive** int8 CUDA-C, a strong **`dp4a.u32.s32`** hand-written CUDA-C, and **cuBLAS int8 IMMA**
`cublasGemmEx`; all four first gated to *equal* the `i32` oracle — classic GemmEx int8 is `s8×s8`, so
the peer cross-check uses `[0,127]` activations where `u8≡s8`, the full-range `[0,255]` gate is separate):

| size | dispatched w64 kernel | % of cuBLAS int8 IMMA | × vs naive CUDA-C | × vs dp4a CUDA-C |
|------|-----------------------|-----------------------|-------------------|------------------|
| 1024³ | w64 (64×64 warp-tile, ldmatrix + XOR-swizzle) | **~86–88%** | ~182× | ~34× |
| 2048³ | w64 (64×64 warp-tile, ldmatrix + XOR-swizzle) | **~96–105% (beats IMMA)** | ~237× | ~58× |
| 4096³ | w64 → 128-tile (HBM-bound) | **~70%** | ~228× | ~57× |

(The retired hand-placed single-tile `_smdb` path measured only ~44–53% of IMMA at these sizes —
22.8 / 39.8 / 39.0 TFLOP/s @1024³/2048³/4096³; the shipped 64×64 warp-tile with `ldmatrix` + XOR-swizzled
SMEM is the standing above. Absolute TFLOP/s are clock-bound — only the same-run ratios are stable.)

**M6 is a decisive, sustained win — ~180–237× the naive hand-written int8 CUDA-C and ~34–58× the dp4a
SIMD-int8 kernel** across re-runs (the literal "beat C on the GPU" for the quantized path). The SMEM
pipeline is **1.6–2.1× the `_mt` fragment-reuse path**, and the dispatch is regime-aware (64×64 tile —
2× occupancy — wins small/medium; the 128×128 tile — more reuse — wins at 4096³), the same split the
fp16 GEMM uses. Against cuBLAS the honest standing is now **~86–88% of its int8 IMMA @1024³, ~96–105%
@2048³ (beating IMMA), and ~70% @4096³ (HBM-bound)** — the 64×64 warp-tile with `ldmatrix` + XOR-swizzled
SMEM (the same levers that lifted fp16) closed the gap the prior hand-placed `_smdb` path (~44–53%) left
open; the residual loss is the 4096³ HBM-bound regime.

**Beating cuBLAS by fusion (int8 dequant).** cuBLAS int8 outputs raw `i32`; a real quantized pipeline
then dequantizes, which cuBLAS **cannot fuse** — it needs a *second* kernel that re-reads the whole
`M×N` `i32` matrix from HBM and writes `M×N` f32. Wukong folds the **per-channel dequant**
`out[i,j] = f32(Σ u8·i8)·scale[j]` into the C store (`int8_gemm_nt_smdb_deq`): the `i32→f32` cvt and the
per-column scale happen in registers before the write. It is **exact** vs the CPU reference (`max_abs=0`,
the shared single f32 rounding), and measured same-run (`int8_dequant_fusion`) the f32-output dequant
kernel costs **~0 over the plain `i32` kernel** (−5%, within noise) — i.e. Wukong gets the dequant free,
exactly the HBM round-trip + launch cuBLAS structurally must pay — so against the cuBLAS int8 GEMM +
separate dequant chain, the fused kernel runs **1.1–2.2× same-run** (the competitive GEMM plus the
eliminated HBM round-trip). Reproduce: `int8_gemm_vs_peers` /
`int8_dequant_fusion` in `wukong_codegen_gpu` with the CUDA 12.9 redist DLLs on PATH.

**Honest peer scoreboard — vs cuBLAS and naive CUDA-C.** A GPU kernel's only meaningful rivals run on
the *same GPU*. Both are now measured here (Phase 0 of the GPU plan): the redistributable **NVRTC** +
**cuBLAS** DLLs `dlopen` like the driver itself, so `cargo test` stays toolkit-free and the peer bench
*skips* (never fails) when they are absent. Two bars:

- **Tier A — naive CUDA-C** compiled at runtime by NVRTC (the idiomatic one-thread-per-output GEMM a
  programmer writes by hand). Beating it is the literal "beat C/C++/Rust **on the GPU**" — the GPU twin
  of Wukong beating scalar CPU-C, via tiling and tensor cores the author never wrote.
- **Tier B — cuBLAS fp16** (`cublasGemmEx`, f32 accumulate): NVIDIA's hand-tuned closed-source gold
  standard. Wukong is reported as a **% of cuBLAS**.

Same-run, same device buffers, checksum-cross-checked, both peers first tolerance-gated against the f64
oracle (a fast-but-wrong kernel never scores). Two back-to-back runs (the mid-size % swings with the
laptop's clock state; the 4096³ cliff does not):

| size | dispatched kernel | % of cuBLAS | × vs naive CUDA-C | vs prior `_sm` default |
|------|-------------------|-------------|-------------------|------------------------|
| 1024³ | `_sm_db` (64-tile + cp.async) | **~101%** | ~98× | **1.16×** |
| 2048³ | `_sm128_db` (128-tile + cp.async) | ~74% | ~71× | **1.15×** |
| 4096³ | `_sm128_db` (128-tile + cp.async) | ~34% | ~47× | ~1.07× |

The honest standing: at L2-resident sizes Wukong's fp16 tensor-core GEMM now **reaches cuBLAS parity
(~101% at 1024³)** — the plan's M1 isolated target (≥95%) is met there — on top of a **wide Tier-A win**
(tens-to-100×+ over the naive hand-written CUDA-C kernel, the same way it beats naive CPU-C). The large
regime — **~74% @2048³ / ~34% @4096³ at this cp.async-only milestone** (the table above) — was
subsequently lifted to **~87–90% @2048³** by the `ldmatrix` + XOR-swizzle workhorse and, with the shipped **v2cs streaming epilogue**, to **76.8% of cuBLAS-f16 / 80.4% of the honest f32-out peer @4096³** (the
headline figure, past the prior 77% PTX ceiling); the cp.async table is retained as the Phase-1 record.

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
the plain staged 64-tile otherwise — each the measured winner in its range. The `ldmatrix` +
swizzled-SMEM workhorse (plus the shipped **v2cs streaming epilogue**) has since lifted the large regime to **~87–90% @2048³ / 76.8% of cuBLAS-f16 / 80.4% of the honest f32-out peer @4096³** (headline);
remaining levers toward ≥95% at 4096³ are deeper (3+-stage) pipelines, warp-tiling, and split-K. Reproduce:
`gemm_vs_peers` in `wukong_codegen_gpu` with the CUDA 12.9 redist DLLs on PATH (one-line setup in
`baselines.rs`).

**Beating cuBLAS by fusion (M1-fused).** cuBLAS can only compute `A·Bᵀ`; an activation needs a *second*
kernel that reads C back from HBM, applies the op, and writes it again. Wukong fuses the activation
into the WMMA store epilogue (elementwise on the f32 accumulators, before the C store — `Act` in
`ptx_wmma.rs`), so `act(A·Bᵀ)` is **one kernel that writes C once**. The fused activations are **relu,
silu, and gelu** — silu fuses the SwiGLU FFN up-projection `silu(x·W1ᵀ)` into a single kernel, and the
transcendental ones reuse the exact SFU formulas of the standalone `vmath` kernels so the fused result
equals the unfused one. Measured on resident device buffers (`fused_gemm_activation_vs_chain`, fused
output gated == `act(cuBLAS)`; the chain cost is GEMM+act summed, exact since they are
dependency-serialized; best-of-N timing so the ratio is clock-stable). The relu case in detail:

| size | Wukong fused | Wukong GEMM+relu | cuBLAS GEMM+relu | fusion vs own chain | **fused vs cuBLAS chain** |
|------|---------------|-------------------|------------------|---------------------|---------------------------|
| 512³ | 0.035 ms | 0.077 ms | 0.084 ms | 2.21× | **2.41×** |
| 1024³ | 0.163 ms | 0.199 ms | 0.193 ms | 1.22× | **1.18×** |
| 2048³ | 1.258 ms | 1.579 ms | 1.177 ms | 1.26× | 0.94× |

So **at L2-resident sizes (≤1024³) the single fused kernel beats the cuBLAS GEMM+activation chain
1.18–2.41×** — the first place Wukong is *faster than cuBLAS*, precisely because it does the fusion
cuBLAS structurally cannot. Fusion beats Wukong's own two-kernel chain at **every** size (1.2–2.2×,
biggest where the GEMM is small and the saved relu pass is a larger share). silu and gelu fuse with the
same effect — across 512³–2048³ all three beat the cuBLAS GEMM+activation chain by **~1.1–1.4×** at
boost clock (the margin is the eliminated activation kernel's launch + C round-trip, which cuBLAS
cannot fuse). The **fused bias** epilogue `act(x·Wᵀ + bias)` — the canonical `nn.Linear`/FFN form — is
fused too: because the WMMA fragment→column map is opaque, the bias path `wmma.store.d`s each tile into
a per-warp SMEM scratch and re-reads by explicit (row,col) to add `bias[col]` before the activation
(`_sm_db_bias{,_relu,_silu,_gelu}`; identity-with-bias is the affine Linear). And the recognizer routes
`act(matmul(…) [+ bias])` straight from Wukong source to these fused kernels under `--backend=gpu`, so
the fusion is a **compiler feature, not a host-API call** (gates `gpu_backend_fused_epilogue_matches_interp`
+ `…_bias_…`, each asserting the offload fired).

**Compile latency + cubin cache (M10).** Wukong emits PTX and the driver JITs it to SASS; there is no
30–120 s autotuning compile like Triton/TorchInductor. Measured (`cubin_cache_compile_latency`, RTX
4050): a from-scratch driver JIT of the fp16 WMMA module (43 KB PTX → 26 KB cubin) is **0.76 ms** — so
even *cold*, Wukong's compile is **~4×10⁴–1.6×10⁵× faster** than a Triton/Inductor cold build (M10
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

| seq (1 head, D=64) | Wukong fused flash | cuBLAS unfused chain (Tier B) | naive CUDA-C (Tier A) |
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
regime** (1.1–1.2×, the per-warp latency bound). **M5 (vs the genuinely-fused libraries): against cuDNN and cutlass mem-efficient fMHA** (reached through
PyTorch's SDPA backends, measured same-run and clock-pinned — `prompts/results/attention.md`), Wukong's
flash **wins the causal-D64-S=512 (1.03–1.16×, beating both peers), fused-RoPE-S≤512 (1.8–5.7×), and
D=128-ldmatrix-S≤1024 (1.11–1.20× vs cutlass-efficient) regimes, is competitive through S≤1024, and trails
cuDNN at long context S≥2048 (0.37–0.66×)** — this is the honest fused-peer bar. Against the older, weaker
**cuBLAS *unfused* attention chain** it is **3.6–5.0×** same-run, single-head — the gap widening at long S
where the materialized S×S scores hurt the chain most.
The chain is the canonical *pre-FlashAttention* attention (Q·Kᵀ and P·V on tensor-core cuBLAS, the S×S
scores spilled to HBM with a softmax between), so the gap **is** the value of fusion, measured against the
gold-standard library for the matmuls — not a strawman. Two honesty caveats: **(1)** the fused-peer win is
**regime-specific** — Wukong beats the fused cuDNN/cutlass peers at short/causal/RoPE context but **trails
cuDNN at long S≥2048** (above), so this is not a uniform "% of FA2" parity claim, and beating the *unfused*
cuBLAS chain 3.6–5× is only the weaker library bar; **(2)** a fused
FA2-class CUDA-C peer is **not buildable on this toolkit-free box** — the `nvrtc_wmma_probe` test shows NVRTC
has *no header search path at all* (even `#include <cuda_fp16.h>` fails to open), so `nvcuda::wmma` cannot be
compiled. (An earlier cross-process re-measure against torch SDPA was **uninterpretable** — torch's *own*
throughput swung ~2× run-to-run on this power-capped part, 66% vs 112% on clock alone — which is exactly why
the fused-peer SDPA standing above is taken **same-run and clock-pinned**, not cross-process.) The register-resident core was itself **1.5–5.9×
the prior WMMA flash** (`flash_mma_vs_wmma`), and `flash_d64_mp` compounds the `cp.async` win on top
(**1.1–3.6× `flash_d64_m`**, **205–738× naive CUDA-C** = M6).

**Multi-head** (`grid.y = H`, the kernel folds head `ctaid.y`'s `[H,S,D]` base offset into the pointers —
zero extra params, single-head stays `grid.y=1`). At small S one head's `S/16` blocks can't fill 20 SMs
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
Against the cuBLAS unfused chain Wukong's flash is **3.6–5.0×** same-run, and against the genuinely-fused
cuDNN/cutlass SDPA peers it **wins the short/causal/RoPE regimes and trails only cuDNN at long S≥2048** (the
fused-peer standing above) — the honest bar, even though a hand-written fused FA2-class CUDA-C peer can't be
compiled on this toolkit-free box. `ldmatrix` conflict-free fragment loads (the strided V `u16` pairs at a 128-byte SMEM stride
are the prime bank-conflict suspect — a *per-warp throughput* issue, consistent with the not-bandwidth-bound
finding) and **FA2-style warp specialization** have both since been explored: the warp-specialized
kernels (2-warp named-barrier anti-phase + 3-stage ring) are **built, gated, and default-routed at
S≥4096 where they win 4–6%** (clock-cancelled median-of-9; a tie @2048 and a loss @≤1024, so pins keep
the losing regimes unroutable) — the last scheduling lever, and it moved only the S=4096 point; the
residual long-S loss is structural (SFU/serial-softmax on 20 SMs), with no locally-measurable FA2
denominator to chase.

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
`COMPUTE_32F`) so it pays the **same** casts as Wukong's WMMA (neither pays a post-GEMM cast), and it
reuses Wukong's *identical* norm/flash/cast/SiLU/add kernels, so the only difference measured is the
GEMM + the epilogue fusion. Both stacks are tolerance-gated against the same f64 oracle before any speed
counts. Both also pipeline their kernel chain on one stream: every intermediate is an `alloc_zeros`
(async device memset) rather than an `memcpy_stod(vec![0;n])` (a *synchronous* H2D that used to stall the
stream ~13× per layer and dominated the cost — removing it cut the layer ~3×). Measured same-machine,
same-buffers, same-run, clock-pinned (`d=64`, `dff=256`, resident):

| single layer | Wukong fused | cuBLAS call-chain | **Wukong faster** |
|---|---|---|---|
| S=256  | 0.633 ms/layer | 0.923 ms/layer | **1.46×** |
| S=512  | 0.813 ms/layer | 1.067 ms/layer | **1.31×** |
| S=1024 | 0.745 ms/layer | 1.121 ms/layer | **1.50×** |
| S=2048 | 1.151 ms/layer | 1.173 ms/layer | **1.02×** |
| S=4096 | 1.250 ms/layer | 1.200 ms/layer | 0.96× (≈par) |

Both stacks run the *same* attention (the register-resident flash above), so the measured gap is purely
**GEMM + epilogue fusion** — the per-call dispatch + the separate add/SiLU launches, paid at every layer,
that the resident fused model folds away. Wukong wins **1.31–1.50× through S=1024**; at S≥2048 the layer
becomes attention-bound (the flash is the same on both sides) so the GEMM-fusion edge shrinks to ≈par. The
register-resident flash also collapsed the long-context layer itself: **S=4096 dropped 3.96 → 1.25 ms
(~3.2×)** vs the prior WMMA-flash layer, since attention was ~3.2 ms of the old 3.96. (Decomposed: Wukong's
WMMA GEMM alone, *unfused* with identical glue, is ~par with cuBLAS at these small-K (64/256) shapes —
*regime specific*, cuBLAS still wins the isolated large-K GEMM ≥2048³, the GEMM table above — and the fused
epilogues the add/SiLU cuBLAS structurally can't fold in add ~1.08–1.09×.) The earlier depth sweep (S=512,
pre-register-flash) showed the ratio **widening with depth** 1.20→1.32× as the fixed per-layer overhead
amortises. Run: `… --ignored --nocapture cublas_chain_vs_wukong` / `resident_model_vs_cublas`.

**PyTorch (Tier C) — Wukong now wins at EVERY S, including S=4096.** The same layer in PyTorch
(`bench/pytorch/transformer_layer_peer.py`, fp16 eager — tensor-core matmuls + fused **SDPA flash
attention**, f32 norm/softmax, f64-verified on the same RTX 4050) is **overhead-bound** (~15 eager kernel
launches + Python dispatch per layer), so its time barely tracks the compute. Wukong is **resident and
overhead-free** — pure compute. With the register-resident flash (`flash_d64_m`), one consistent run:

| case | Wukong (register flash) | PyTorch eager | result |
|---|---|---|---|
| single layer, S=256 | 0.63 ms | 2.03 ms | **Wukong 3.2×** |
| single layer, S=512 | 0.81 ms | 2.15 ms | **Wukong 2.7×** |
| single layer, S=1024 | 0.75 ms | 2.94 ms | **Wukong 3.9×** |
| single layer, S=2048 | 1.15 ms | 1.94 ms | **Wukong 1.7×** |
| single layer, S=4096 | 1.25 ms | 1.69 ms | **Wukong 1.35×** |
| stack (S=512, depth 1–8) | ~0.34 ms/layer | ~2.0–2.6 ms/layer | **Wukong ~6–7.8×** |

This **flips the prior crossover**: with the old WMMA flash PyTorch won S=4096 by ~1.75–3.5×; the
register-resident flash collapsed the S=4096 layer **3.96 → 1.25 ms** and Wukong now *leads* there 1.35×.
Note Wukong wins the *layer* at S=4096 even against torch's own fused SDPA flash inside that layer —
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
throughput at this real shape (`cublas_chain_vs_wukong_mha_layer_throughput`) — fused multi-head Wukong
vs the **multi-head cuBLAS-chain layer** (attention common to both, so the gap is purely GEMM + epilogue
fusion):

| S | Wukong fused | cuBLAS chain | net | GEMM (Wukong-unfused / cuBLAS) | fusion |
|---|---|---|---|---|---|
| 512  | 1.073 ms | 1.020 ms | chain 1.05× | 0.90× | 1.05× |
| 1024 | 2.402 ms | 2.360 ms | chain 1.02× | 0.90× | 1.09× |
| 2048 | 6.389 ms | 6.422 ms | **Wukong 1.01×** | 0.92× | 1.09× |
| 4096 | 18.26 ms | 17.61 ms | chain 1.04× | 0.93× | 1.03× |

**Honest finding:** at the real GPT-2 shape the fused layer is **~par with the cuBLAS-chain layer
(0.96–1.01×)** — *not* the clearer win the D=64 toy shows. The decomposition says why: Wukong's WMMA GEMM
is **0.90–0.93× of cuBLAS at D=768** (the residual large-GEMM gap — the same one `gemm_vs_peers` reports,
76.8% of cuBLAS-f16 / 80.4% of the f32-out peer @4096³ after the v2cs epilogue), and the fused residual/SiLU epilogues (**1.03–1.09×**, which cuBLAS structurally can't do)
nearly but not fully offset it. **Closing the remaining large-GEMM gap flips this to a clear win** — the
single highest-leverage GPU item, exactly what the parallel large-GEMM workstream targets. (A cross-process
PyTorch comparison at this shape, like the D=64 table above, remains a documentation follow-up.)

Honest caveats: **eager** PyTorch only — `torch.compile`/Inductor needs Triton, which has no working Windows
install (`torch.compile` raised `Cannot find a working triton installation` here). Cross-process and
**clock-noisy** at this ~1–2 ms scale (the laptop GPU boosts ~7×), so treat the ratios as order-of-magnitude
and the **direction** — Wukong faster at every S, by a margin growing toward small S — as the robust signal.
The clock-invariant backbones are the same-run `flash_vs_peers` (Tier-A 205–738× single-head / 275–322× at H=12 vs naive CUDA-C; Tier-B the
3.6–5.0× cuBLAS-unfused-chain and the fused cuDNN/cutlass SDPA standings above) and the
`cublas_chain_vs_wukong` same-run layer table above.

**Determinism, every kernel (M12).** Not just the layer: `gpu_kernels_bit_reproducible` asserts every
reduction-bearing family — `gemm_nt_f16` and its `_sm`/`_sm_db` variants, the three fused row norms,
flash-attention, conv2d, and the sum/dot reductions — returns **bit-identical** output across runs on
identical inputs (fixed grids, atomic-free, fixed-order reductions). cuBLAS offers no such contract:
`reproducibility_vs_cublas` shows Wukong's fp16 GEMM bit-identical across three runs *by construction*,
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

| shape (M×K×N) | Wukong W4A16 | × vs naive CUDA-C | × vs Wukong fp16 (same tile) | weight HBM/pass |
|---------------|--------------:|------------------:|------------------------------:|-----------------|
| 64×4096×4096 (decode) | ~12.6 TFLOP/s | **~180–213×** | **~4.0–4.2×** | int4 8.4 MB vs fp16 33.6 MB (**4×**) |
| 256×2048×2048 | ~14.9 TFLOP/s | ~165× | **~1.46×** | 2.1 vs 8.4 MB |
| 64×2048×2048 | ~8.8 TFLOP/s | ~92× | ~1.00× | 2.1 vs 8.4 MB |
| 512×512×512 | ~9.1 TFLOP/s | ~78× | ~1.00× | 0.13 vs 0.5 MB |

The **4× weight-bandwidth reduction lands in the large-weight decode regime** (64×4096), where decode is
weight-BW-bound and the fp16 path stalls on weight reads — Wukong is ~4× faster there. After the
Marlin/AWQ `lop3` unpack made the dequant nearly free, **W4A16 is ≥ the fp16 path in *every* regime** (no
int4 penalty anywhere; it even *beats* fp16 1.46× at 256×2048 because the 4× smaller weight tile fits L2
far better). **Static-shape specialization** (Wukong's no-library lever — `w4a16_static_ptx` bakes M/N/K
as constants so ptxas strength-reduces the strides to shifts) adds a further **1.05–1.30×** over the
same-loaded dynamic kernel, bit-identical. The remaining headroom is in the thin-M decode shape, which is
occupancy-bound (M=64 → few CTAs); split-K is the identified next lever (its deterministic reduction caps
the net gain).

**Other op categories** (all emit+execute, tolerance-gated on the 4050): elementwise (saxpy/vadd),
deterministic reductions (sum/dot/max — bit-exact for max, tolerance for the f32 sums), activations
(relu/exp/sigmoid/tanh/silu/gelu via SFU), fused row norms (softmax/LayerNorm/RMSNorm, one warp per
row), and direct conv2d. Reproduce: `cargo test -p wukong_codegen_gpu --features gpu` (correctness;
skips cleanly with no GPU) and `… --release -- --ignored --nocapture` (throughput).

**End-to-end `--backend=gpu`.** Beyond the crate's host API, the GPU is wired as a third compiler
backend: `wukongc --features gpu --backend=gpu --run foo.wk` tree-walks the program on an
*offloading interpreter* and runs recognized GEMM / activation / reduction / fused-norm calls on the
device (an `Accelerator` seam in `wukong_interp` that the driver fills with `wukong_codegen_gpu`).
**Fusion reaches the source level:** a Wukong `act(matmul(x,w))` folds to `wukong_sgemm_nt_epi`,
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
already saturates). Reproduce: `cargo test -p wukong_codegen_gpu --features gpu --release -- --ignored
--nocapture pool_graph_vs_unpooled decode_stack_latency overlap_throughput concurrent_forwards_throughput`.

### End-to-end models — GPT-2 & Llama blocks authored in `.wk` (Phase 9)

Two real transformer blocks now exist as Wukong *source*: `examples/gpt2.wk` (a pre-LayerNorm GPT-2
decoder block — multi-head causal attention + GELU MLP + residuals) and `examples/llama_block.wk` (a
pre-RMSNorm Llama block — RoPE + multi-head causal attention + SwiGLU FFN + residuals). They are
written **entirely in the recognized op-forms**, so the *same source* dispatches to the tuned kernels
on every backend rather than to a naive nest:

| model | recognized dispatch (`--emit=mir`) | oracle gate |
|---|---|---|
| `gpt2.wk`        | 2× `wukong_norm_affine_f32` (LayerNorm) + **6× `wukong_sgemm_nt`** (Q/K/V/O + FFN up/down) + streaming residual `velem` | `--run` **-O0 == -O3**, deterministic |
| `llama_block.wk` | 2× `wukong_norm_affine_f32` (RMSNorm) + **7× `wukong_sgemm_nt`** (Q/K/V/O + SwiGLU gate/up/down) + `silu` vmath + streaming `velem` | `--run` **-O0 == -O3**, deterministic |

(The per-head attention `Q·Kᵀ`/`P·V` matmuls carry a head-column offset, so on the CPU oracle they run
as general nests; on the GPU they are the hand-coded flash kernel inside the resident layer below.)
Both run end-to-end on the interpreter oracle at a small-but-structurally-exact config (S=8, D=64,
H=4, Dff=256 — the 124M / 7B configs are harness-driven, since a `--run` stack cannot hold the
weights) and are differentially gated **`-O0 == -O3`** (deterministic, byte-for-byte).

**Compile latency on a real model (M10).** Best-of-15, release `wukongc`, this RTX-4050 box, full
source → optimized MIR (`wukongc --emit=mir -O2 <model>`):

| model | source → optimized MIR (wall, best-of-15) | vs the JIT field's cold compile |
|---|---|---|
| `gpt2.wk` (200 lines)        | **~12.8 ms** (≈7.5 ms is a fixed process-startup floor → ≈5 ms compile work) | **~2,300–9,400×** faster |
| `llama_block.wk` (195 lines) | **~14.6 ms** (≈7 ms compile over the same floor)                             | **~2,000–8,200×** faster |

The denominator is the **documented Triton / TorchInductor cold-compile of 30–120 s** for a model (one
reported Triton kernel alone: 151 s), which includes the runtime autotuning search Wukong skips
outright — its shapes are compile-time-known (in the type system), so it emits a bespoke kernel with no
search. Wukong compiles a *whole transformer block's definition* source → runnable IR in **single-digit
-to-teens of milliseconds**; the per-kernel PTX→SASS step is then the already-measured driver JIT
(**0.76 ms cold, 0.16 ms warm cubin**, M10 above). This compile-latency gap is the one place an
orders-of-magnitude **absolute** claim is fair, and these authored models confirm it end-to-end.

**GPU-resident execution = the resident-layer benches above.** The math these `.wk` blocks express is
exactly the resident `transformer_layer` / `ResidentLayerF16` benched in this section — RMSNorm →
Q/K/V → flash-attention → O-proj + residual → norm → FFN → residual: M7 reports **1.33 ms/layer @S=256**
(~160–190 K tokens/s) and M13 reports the fp16 fused layer **beating the cuBLAS call-chain at D=64,
~par at the real GPT-2 D=768/H=12 shape**, with the Phase-7 pool+graph runtime folding a 12-layer
decode into **one `cuGraphLaunch` (~6.5–6.9× eager)**.

**Honest status & what is pending (the honesty law).** Authored + oracle-gated + compile-latency-measured
here; the GPU-resident *inference* numbers are the resident-layer benches above (same computation, same
device). **Not yet measured, so not claimed:** the full 124M-GPT-2 / 7B-Llama run driven straight from
these `.wk` files through the general lowerer (recognized-op GPU dispatch is the Phase-4 *perf* tail);
a **training-step** tokens/s vs PyTorch-eager (the autodiff engine is built and integrated — see the
`--train` CLI surface — but this specific benchmark is not yet run); a full-model run through the
**whole-program cooperative megakernel** (the megakernel is built; an end-to-end full-model pass through
it is not yet measured); and a **PyTorch-eager / TensorRT-LLM** same-run peer for the full model. This
integration branch now carries the general MIR→PTX lowerer, the Phase-7 device pool + CUDA graphs, the
conv2d slice, the int8 GEMM stack (the section above), the whole-program megakernel, and the autodiff /
large-GEMM-fusion work — the branches this note earlier listed as pending are now merged; what remains
open is the *end-to-end full-model* measurement, not the per-op kernels.

### GPT-2 124M end-to-end — numerical correctness vs HuggingFace (not a perf benchmark)

**This is a correctness / capability result, not a performance measurement.** No throughput or latency
comparison against PyTorch (or any other runtime) was run for the full model, and none is claimed here —
the numbers below are *numerical agreement with a reference*, and say nothing about speed. (The perf
standings for GPT-2-*shaped* kernels are in the
[End-to-end model](#end-to-end-model--a-12-layer-gpt-2-class-transformer-stack-cpu-inference) section
above and are untouched by this; a full-model throughput peer remains unmeasured, per the pending-work
note above.)

`examples/gpt2_infer.wk` runs **GPT-2 124M inference end-to-end as a Wukong program** on the native
Cranelift-JIT backend: it loads the **real pretrained OpenAI GPT-2 weights** — all **124,439,808**
parameters, a flat little-endian f32 blob exported from HuggingFace `transformers` by
`tools/export_gpt2.py` — through the `read_f32` file-I/O intrinsic, and the prompt's token ids through
`read_i32`, then executes the full forward: token + learned positional embedding, 12 pre-LayerNorm
blocks (biased QKV, 12-head causal attention, tanh-GELU MLP, residuals), the final LayerNorm, and the
tied LM head → logits `[5, 50257]`.

`tools/verify_gpt2.py` compares those logits against HuggingFace's authoritative reference. Writing Δ
for the elementwise difference (wuk − ref) and taking the relative error as `max|Δ| / max|ref|`:

| metric | measured | reference / gate |
|---|---|---|
| relative max error (max abs Δ over max abs ref) | **1.87×10⁻⁶** | gate: rel ≤ 1e-3 |
| absolute max error (max abs Δ) | 2.44×10⁻⁴ | max abs ref = 130.28 |
| argmax next-token (prompt *"Hello, my name is"*) | **1757 (" John")** | HuggingFace: 1757 — exact match |

**Honest scope.** It is **inference, not training**. **Tokenization is external** — the `.wk` program
consumes integer token ids produced by the HuggingFace tokenizer; it does not tokenize text itself. The
full 124M run is **native-only**: the interpreter cannot hold 124M parameters in its slot memory, so the
reduced-config twin `examples/gpt2_infer_small.wk` runs the *identical* forward pass **bit-for-bit on the
interpreter and the native backend** (the CI examples differential + opt-invariance gate) — which is what
proves the full-scale native run executes the pipeline correctly. Reproduce (repo root):
`python tools/export_gpt2.py`, then `wukongc --run --backend=native examples/gpt2_infer.wk`, then
`python tools/verify_gpt2.py`.

## Honest summary

- **Compile time:** the xbench figure — ~100–680× faster than gcc/rustc (latest full-board geomean
  **306×**; drifts ~150–310× with the C/Rust toolchain's spawn time) — is **in-process JIT/embedding
  latency** vs spawning a toolchain; the both-subprocess `compile-vs` mode is the headline
  compiler-to-compiler comparison (a **~7–12× win, measured ~7–9× this session**, on bare equivalent
  kernels). Robust every
  run; the metric that dominates ML iteration.
- **Matmul / nn.Linear (the flagship ML kernels):** Wukong **wins single-thread (~3–26×) and
  dominates parallel (~9–104×)**, and the lead **grows with matrix size** — the compiler tiles,
  packs, and register-blocks where gcc/rustc leave the naive nest. The single-core GEMM holds
  **~110–120 GFLOP/s** (≈90% of one P-core's AVX2-FMA peak); the parallel path holds **91–104% of
  oneMKL-all across 512³–2048³** same-run (a fixed GFLOP/s is meaningless at this ~3× clock swing).
  This is a reversal of the previous honest loss (single-core matmul used to be ~3×
  *behind*). The dispatch also fires on **runtime dimensions**, so the win applies to general matmul
  functions, not only fixed-size kernels.
- **Fused FFN epilogue (`act(x·Wᵀ [+ bias])`):** a `nn.Linear` immediately followed by a
  bias-add/activation loop folds into **one** `wukong_sgemm_nt_epi` call — the bias and activation
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
  cbrt, hyperbolic family sinh, cosh, asinh, acosh, atanh — 35 in all):** **~2–11.5× faster** than C's scalar `libm` — Wukong dispatches the loop to a **256-bit AVX2
  ≈1-ULP poly kernel** (`wukong_vmath_f32`), where gcc/rustc cannot vectorize a loop with an
  `expf`/`logf`/`tanhf`/`sinf`/`cosf`/`erff`/`asinhf` call. This is the transformer/vision activation family and the cleanest
  compute-bound win (it roughly doubled when the kernel moved from the 128-bit vectorizer to 256-bit).
  `sin`/`cos` (the **RoPE** rotary-embedding transcendentals) win the most (~6–8.6×) — `libm`'s
  `sinf`/`cosf` are heavier than `expf` — and `erf` gives the exact BERT/GPT-2 GELU.
  All of them are first-class intrinsics composing the shared ≈1-ULP `exp`/`log` (e.g.
  `softplus = ln(1+eˣ)`, `mish = x·tanh(softplus)`), so the whole family is exact and bit-identical
  across backends; an `@parallel` activation runs multicore × 256-bit (~28×); log-softmax /
  cross-entropy compose the exp/log win.
- **Fused normalizations (softmax / LayerNorm / RMSNorm):** the per-token transformer norms are
  recognized from their multi-pass source and folded into one `wukong_norm_f32` call (256-bit AVX2 +
  8-lane reassociated reductions + the vectorized `exp`), **~1.9–6.6× faster than C** — softmax most
  (the vectorized `exp` dominates), RMSNorm least (a single reduction, diluted by its elementwise
  passes). The **affine** form real models run (learned per-channel scale γ + shift β) dispatches to
  `wukong_norm_affine_f32` and holds the same **~1.7–3.7×** (γ/β fused into the writeback). Both
  backends marshal the identical kernel, so the differential oracle stays bit-exact.
- **Convolution:** lowered as im2col + GEMM (the XLA/cuDNN strategy), Wukong runs a 3×3 conv
  **~1.55× faster** than the hand-written direct-convolution nest in C, and **1.12× slower** than the
  same nest at `-ffast-math` — the matmul recognizer accelerates conv for free, but only modestly.
  *(Corrected 2026-08-04: the published ~6–7× was measured against a peer with no `restrict`, which
  forced gcc to reload the accumulation operands on every output pixel; adding it took the C peer
  from 4.9 to 21.1 GFLOP/s.)*
- **Reductions:** ~2.6–2.9× faster (lane-accumulator reassociation), incl. `fmax`/`fmin` (softmax's
  row-max). IEEE-serial C baseline; see the `C(fast)` column for the reassociation-normalized
  comparison.
- **Auto-parallel:** ~1.8–7.6× faster than idiomatic single-threaded C across elementwise kernels —
  bounded by aggregate memory bandwidth, not core count (these kernels are memory-bound).
- **Single-thread memory-bound elementwise (saxpy/scale/residual/poly):** now a **win** (~1.1–1.5×
  vs C at N=2²⁰, widening to ~1.3–1.6× at >L3 sizes), where it used to be a tie. A recognized
  streaming map dispatches to the 256-bit AVX2 `wukong_velem_f32` / `wukong_vhorner_f32` kernels
  (4×-unrolled for ILP) which emit **non-temporal stores** once the working set spills L3 — skipping
  the read-for-ownership traffic every cacheable store pays, a store path gcc/rustc don't emit
  automatically. The non-temporal decision keys on the *total* streamed bytes (all live arrays), so a
  cache-resident map keeps its normal store (where a needless `vmovntps` would lose): `relu` at N=2²⁰
  (2-stream, 8 MiB, L3-resident) is a clean tie, and a win at >L3. This is the same play as the GEMM
  and activation families (`wukong_sgemm*`, `wukong_vmath_f32`) — true 256-bit width plus a
  domain-aware store policy the generic 128-bit-only Cranelift vectorizer can't reach.
- **bf16 mixed-precision reductions (bf16 storage, f32 accumulate):** `bf16` is real 2-byte storage
  at bf16 precision (round-to-nearest-even), bit-exact across backends, and a `s += (x[k] as f32)
  [* (y[k] as f32)]` reduction over `[bf16; _]` arrays now **dispatches to a SIMD kernel**
  (`wukong_dot_bf16` / `wukong_sum_bf16`: widen to f32, 8-lane f32 accumulate) — the standard ML
  mixed-precision contract. The payoff is **bandwidth** (bf16 moves half the bytes of f32): bf16 dot
  runs **~4–7.6× faster than idiomatic single-threaded C** and sum **~6–8×** (C's unary f32 sum is
  latency-bound) — **but 1.1–1.8× SLOWER than the same C at `-ffast-math`**, which is the
  like-for-like basis here because Wukong's kernel is itself an 8-lane reassociated accumulate. The
  `C(fast)` column was missing from this bench until 2026-08-04; on a reassociation-normalized basis
  the bf16 reduction family is a **loss**, not a win. On this
  AVX2 box (no bf16 FMA) a bf16 *GEMM* would widen to f32 and match f32 throughput — a footprint
  feature, not a FLOP/s win — so that path stays at f32; the reduction kernels are where bf16 pays.
  The same dispatch now also covers **bf16 elementwise** (`out[k] = a*(x[k] as f32) + b*(y[k] as
  f32)` → `wukong_axpby_bf16`, bf16 in / f32 out): ~1.3× a plain f32 axpby (8 vs 12 bytes/elem; below
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
  (`flash_pipe_vs_mma`), **205–738× a naive CUDA-C flash** (M6), **3.6–5.0× a cuBLAS *unfused* attention
  chain** (M5, the older/weaker library bar), and — vs the genuinely-fused cuDNN/cutlass SDPA peers, same-run
  clock-pinned — **winning the causal-S=512 (1.03–1.16×), fused-RoPE-S≤512 (1.8–5.7×) and D=128-ldmatrix-S≤1024
  regimes while trailing cuDNN at long context S≥2048 (0.37–0.66×)** (a hand-written fused FA2-class CUDA-C
  peer can't be built here — NVRTC has no headers);
  a **whole pre-norm transformer layer runs end-to-end GPU-resident**
  (matching a CPU f64 reference to max_rel 2.7e-4, deterministic run-to-run) and now **beats PyTorch eager
  at every S, including S=4096 (1.35×)**. Numbers are honest for a power-capped 6 GB
  mobile GPU, not a datacenter part. The CPU↔GPU differential is a `c·√K·ε` tolerance over the full output.
- **Safety:** Wukong checks tensor **shapes at compile time** (in the type system), a class of bug
  C/C++/Rust-with-raw-pointers cannot catch.

**Where Wukong does NOT win** (measured 2026-08-04 against corrected peers, and listed here because
every one of these was previously published as a win):

- **Column reductions** (sum/max/min/absmax/mean/sum-sq/L2/RMS down axis 0): between **a tie and
  1.8× slower** than gcc on the natural row-outer nest — 15 of the 16 measured rows are losses, and
  the >L3 4096×1024 shape is a consistent ~1.65× loss across two independent rounds. Was ~29–50×.
- **Column argmax/argmin** (axis-0): between a **tie and 2.5× slower** (round 1: 1.49–2.47× slower on all four rows; round 2: 1.05× faster to 1.74× slower). Was ~2.7–5.3×.
- **f32 transpose, single core:** a **tie** (1.00–1.17× across two rounds) once the peer is blocked too. Was ~1.5×.
  (`@parallel` still wins 4.8–7.1× vs 1-thread C, ~1.0–1.1× vs all-core OpenMP.)
- **bf16 reductions vs `C(fast)`:** **1.1–1.8× slower**. The plain-C multiple (~4–7.6×) is entirely
  the withheld `-ffast-math`.
- **GEMV vs `C(fast)`:** a **tie** (1.09× slower to 1.11× faster). The ~3.0–3.6× vs plain C is the
  withheld `-ffast-math`, not the kernel.
- **conv2d vs `C(fast)`:** **1.12× slower**. Was ~6–7× vs plain C without `restrict`.
- **cumsum vs `C(fast)`:** 1.03–1.42×, i.e. within noise of a tie. Was ~1.4–2.0× vs plain C.

Where Wukong wins is where a tensor compiler should: compile speed, matmul/GEMM throughput (now on
runtime dimensions too), convolution (im2col + GEMM), vectorized transcendentals (the transformer
activation family, including `log` for log-softmax/cross-entropy), fused row normalizations
(softmax/LayerNorm/RMSNorm), bf16 mixed-precision reductions (bandwidth), automatic parallelism,
automatic vectorization (including reductions), automatic fusion, and shape safety — plus a
PTX-emitting **GPU backend** that takes the same ops to the RTX 4050's tensor cores (fp16/bf16 GEMM
at ~5–6× the f32 register-blocked path — clock-dependent, cuBLAS parity ≤1024³ — fused
flash-attention, a whole layer GPU-resident).
