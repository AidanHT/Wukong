# Metrics that matter — and Wukong's honest standing

This document defines, from first principles, the metrics an ML/DL kernel language and its
compiler must win at, and records Wukong's current standing on each — including the losses.
It is the north star for optimization work: a change that doesn't move one of these metrics
(or protect a gate) is not worth its complexity. Numbers cited here are *recorded ratios* from
`BENCHMARKS.md` and `prompts/results/*`; absolute GFLOP/s are deliberately absent (this
hardware's clock swings ~3× CPU / ~7× GPU — only same-run ratios and %-of-roofline are stable).

> **Two GPUs now, and they do not mix (2026-08-11).** The **H100 (`sm_90a`)** figures — the
> wgmma-vs-cuBLAS suite under M1, the H100 HBM figure, and the Hopper quantized standings — were
> measured on an **NVIDIA H100 80GB HBM3** (Hopper, 132 SMs, 50 MiB L2, Linux container, cuBLAS with
> **f32 output** as the peer), 2026-08-11, logs `bench/gpu/h100/2026-08-11-h100-w2-r*.log`, section
> `BENCHMARKS.md` → *GPU backend (NVIDIA H100 80GB HBM3, `sm_90a`)*. **Every other GPU figure in this
> document is an RTX 4050 figure** (scope note immediately below). Do not average the two, and do not
> carry a conclusion across: this visit measured that transfer failing outright — the 4050's int8
> tuning, which reaches 96–105% of cuBLAS IMMA at 2048³ *there*, reads **22–52%** of IMMA on the
> H100. **Every H100 figure in this document is ITERATION-grade, not publication-grade, and is
> recorded under that label:** the rounds ran in a Modal container whose user cannot call
> `nvidia-smi -lgc`, and `GPU_RETARGET_PLAN.md` §6.3 rules that "Container rounds are *iteration*
> data; VM rounds are *publication* data" — every round behind these figures records the refusal
> itself (`[clock] lock: refused (…); running unlocked`), and five of them also stamp themselves
> `ITERATION data, not publication data` in an all-caps banner; `BENCHMARKS.md` names which five, and
> why the K-sweep and the four non-GEMM rounds carry only the `[clock]` line. The label here is this
> document's, applied uniformly because the premise every round records is uniform.
> In place of the lock each round recorded its own before/after clocks and **two
> rounds refused themselves** on +6.82% drift; one shape refused on a ±15.47% *peer* floor, and those
> refusals are printed in `BENCHMARKS.md` as results. A drift gate is the weaker instrument, though —
> it sees a clock that *changed*, not one parked at the wrong steady state for the whole round — so
> **treat every H100 percentage below as provisional pending a locked-clock root-VM re-run**, which
> is owed and unscheduled.
>
> **Device scope (2026-08-06, extended 2026-08-09):** every *other* GPU figure in this document (they
> appear under M1, M2, M4, M7 and the improvement targets) was measured on an **NVIDIA RTX 4050 Laptop GPU**
> (Ada, `sm_89`, **20 SMs**, 6 GB, **~192 GB/s**, power-capped ~30–50 W) under **Windows/WDDM**, with
> only the peers that box can host: cuBLAS / IMMA / cuBLASLt and cuDNN via the redistributable DLLs,
> NVRTC-compiled CUDA-C, PyTorch in **eager** mode (Triton does not install on Windows), and **no
> CUDA toolkit** — hence no CUTLASS, FlashAttention, Marlin/Machete or vLLM peer. These are
> properties of that instrument; **do not extrapolate them to datacenter parts.** The datacenter
> retarget, including re-measurement against the peers a Linux cloud box can build, is
> `GPU_RETARGET_PLAN.md`.
>
> **Three GPU families need more than a scope note; `BENCHMARKS.md`'s standing index has the full
> text.** (1) Every **PyTorch** comparison here is against **eager**, which this document's own M2
> rule says does not count — it is **not re-earned** against `torch.compile`, and the tooling to
> build and verify that peer landed 2026-08-09 (`tools/cloud/peers/`). (2) The **CUDA-graph**
> launch-overhead multiples under M2/M7 are **Windows/WDDM** numbers and should be expected to shrink
> on the Linux driver before the GPU changes at all. (3) M1's attention verdict — the long-S cuDNN
> gap called *"structural: SFU/serial-softmax bound on 20 SMs"* and *"honestly bounded, not closable
> by scheduling"* — **is bounded only for 20 SMs.** At 108/132 SMs the premise dissolves and the
> direction is unknown; and the *short*-S wins recorded beside it (fused-RoPE S≤512, causal D=64
> S=512, D=128 ldmatrix S≤1024) rest on 20 SMs being easy to fill and may **invert** on a part a
> small problem cannot fill. The same applies to int8's *"96–105% at 2048³ (beats IMMA)"*: it is an
> occupancy crossover the 64×64 warp tile wins **because** the part has 20 SMs.
>
> **Standing of the CPU figures in this document.** They are recorded ratios from `BENCHMARKS.md`
> and inherit its open debts, which are currently the larger ones: the **M2** end-to-end model
> ratios vs C are an **upper bound** pending re-measurement (the `c_model` peer was not
> `restrict`-qualified until 2026-08-05); every **"vs Rust"** figure on roughly forty rows needs
> re-measurement after the 2026-08-06 `noalias` fix; the **"vs C++"** column only ever existed in
> three sections and is an unproven assertion everywhere else; and the whole **general-code**
> (outside-the-recognizer-dialect) family is **superseded with no replacement published** — its
> tables' timing columns were taken by an instrument with no control column in it. Per-family
> statuses are in `BENCHMARKS.md`'s standing index.
>
> **M7's corpus-coverage counts are stale.** The figures quoted there predate both the corpus growth
> and the 2026-08-09 platform-invariance fix; the live values are the ratcheted floors
> `RUN_CORPUS_COVERAGE_FLOOR` (`crates/wukong_codegen_gpu/src/lower.rs`) and
> `MEGA_CORPUS_COVERAGE_FLOOR` (`megakernel.rs`), which the gates print. Those floors are now
> **host-vectorizer-invariant** — before the fix, Windows and Linux read different coverage from the
> same commit because the Win64-only 256-bit AVX2 recipe suppressed lowering on one of them, so the
> metric conflated "what gpu-native can lower" with "did the host CPU vectorizer fire". Re-run the
> gates rather than reading a count out of prose.

## Who the metrics serve

Three real users: (a) **kernel authors** writing custom ops, (b) **model authors** composing,
training, and running models, (c) **deployers** serving inference. Everything below traces to
what one of them pays for.

## Tier 0 — hard gates (not scores; violating one voids every other number)

| Gate | Definition | Enforcement |
|---|---|---|
| G1 backend agreement | interp == native (== GPU within `c·√K·ε`) bit-for-bit on the differential suite | `cargo test` differential tests; `wukong_bench` equivalence gate |
| G2 opt invariance | `-O0` == `-O1/2/3` observable behavior (stdout + exit) on every run fixture | `optimization_is_observationally_invariant` |
| G3 numerical trust | vmath/norm kernels within documented tolerance of an f64 reference (33/36 vmath ops covered) | `vmath_kernels_match_f64_reference`, `norm.rs` f64 gates |
| G4 deterministic parallelism | `@parallel` == serial bit-for-bit (fixed chunking, ordered folds) | differential `@parallel` tests |
| G5 benchmark honesty | measured path = the real pipeline (parse→sema→MIR→opt→Cranelift JIT); recognizers structural, never workload-keyed; strongest-reasonable peer flags; same-run interleaved A/B | fairness audits; cross-checks inside `wukong_xbench` |

The documented exception to G1/G2: reassociated float reductions (vectorized/recognized) make
the *reassociated form* the oracle — both backends execute the identical reassociated IR.

## Tier 1 — dominant metrics (the reason to exist)

**M1. Kernel throughput on the ML op mix, vs the strongest peer available.**
Peers in order of strength: vendor libraries (oneMKL, cuBLAS/cuDNN) > tuned crates
(`matrixmultiply`) > idiomatic C/C++/Rust at `-O3 -march=native` (gcc/g++/rustc).
Current standing (recorded):

- *Compute-bound, CPU*: f32 GEMM single-core **~80–104% of oneMKL-1c across sizes, at or above
  parity (≥100%) at 2048³/4096³** (MKL-anchored ABBA, both orderings; the 2026-07-10 C-tile
  prefetch closed the large-matrix writeback-miss tail, the off-switch reproduces the old number).
  Multicore is **size-keyed 2D block-parallel** with **per-BLOCK packing** (each worker's
  "redundant" packing warms its own L2 — it beat the shared-cooperative-pack design in every ABBA
  ordering, the session's headline refutation) and **dynamic block-claiming the 2026-07-11 default**
  (halved the mid-size straggler variance). Standing vs MKL-all, same-run: **512³ ~98–99%, 1024³
  ~104%, 2048³ 91%, 4096³ 93%, 256³ ~94–110%** (the dynamic-claiming default closed the old
  70–83% mid-size gap; against MKL's power-degraded rounds the ratios read far higher — threaded MKL
  itself swings ~1.4–2× with power state, so only same-run ranges are quoted). Skinny transformer
  NT shapes **75–112% of MKL-all** (4/6 at or above parity; worst 128×768·768ᵀ 75–80%,
  overhead-bound at 151 MFLOP). int8 GEMM (VNNI) 1.5–2.5× gcc's own `vpdpbusd` auto-vec.
- *Compute-bound, GPU (H100 80GB HBM3, `sm_90a`, 2026-08-11 — **iteration-grade**: container round,
  clocks unlockable, plan §6.3)*: the `wgmma` + TMA GEMM measures
  **95.3% / 88.6% / 92.3% of cuBLAS f16 (f32 out) at 2048³ / 4096³ / 8192³**, **97.3%** on the GPT
  FFN down-projection (4096×1024×4096) and **73–80%** on the wide-N up-projections; bf16 under the
  same rule agrees within ~2 points on every shape that resolved in both. **1024³ f16 is REFUSED**
  — the peer's own twin arms disagreed by ±15.47%, over the ±5% bar, so no number was minted (bf16
  resolved there and reads 49.5%, a 32-CTA problem on 132 SMs). The dominant lever was the fused
  `st.global.v2.f32` epilogue (**+25.2 / +15.3 / +10.1 points** at the three square shapes — the only
  single-axis reading of it, and on the *clustered* arm), moving
  the six resolvable shapes from ≈61% to **≈88%** of cuBLAS; that suite-level move also carries the
  B-multicast cluster at four of the seven shapes, so it is not a single-axis number. A store-elided *diagnostic* arm (writes
  no C, ungateable, read only as a difference against the **clustered** arm it was cut from) sits at
  **114.0 / 101.1 / 101.8%** **at the three square shapes it was run on — and nowhere else**, so the
  mainloop is at or above the peer *there* and what is still owed *there* is epilogue plus wave
  overhead. Where the gap sits on the three `gpt_*` FFN shapes, including the 73.4% worst row, is
  **unmeasured**: no elided arm was ever run on them. **The GEMM
  readings above went through the instrument; the ones that follow did not** — they are single-shot
  rounds with no provenance header, no A/C peer twin, no measured floor, no median-of-5 and no
  publish gate, and a repeat under the instrument is owed. Memory-bound
  on the same part: copy **2922 GB/s = 87.2% of the 3352 GB/s spec peak** (≥90% not met). Quantized:
  the **fused int8 GEMM+dequant beats the cuBLAS GEMM+dequant chain 1.08× at 1024³ and 1.15× at
  2048³** — the first outright peer win on Hopper — and **loses at 4096³ (0.79×)**, because the int8
  GEMM underneath it is itself only **22–52% of cuBLAS IMMA** here: the 4050's tile/occupancy tuning
  does not transfer and Hopper int8 needs its own search. (Both of those are still *same-run*, and
  the int8 arms are bit-for-bit gated against the `i32` oracle before timing; what they lack is the
  twin, the floor and the gate.) Peer-bar fact, not a Wukong number:
  cuBLASLt on this device fuses RELU/GELU/BIAS at **both** f32 and f16 output, so only **SiLU** (and
  residual+activation) is absent from its epilogue enum. Full section and per-round log citations in
  `BENCHMARKS.md`.
- *Compute-bound, GPU (RTX 4050, sm_89)*: fp16/bf16 GEMM ~101% cuBLAS ≤1024³, **~87–90% at
  2048³; the 4096³ GEMM ships the v2cs streaming epilogue at 76.8% of cuBLAS-f16 / 80.4% of the
  honest f32-out peer** (a SASS-level loss past the prior 77% PTX ceiling — see below); int8 GEMM
  **96–105% at 2048³ (beats IMMA), 86–88% at 1024³, ~70% at 4096³** (HBM-bound loss); fused int8
  GEMM+dequant **1.1–2.2×** the cuBLAS chain; int4 W4A16 documented lead over the *naive* Tier-A
  peer — ~~no library peer exists~~ **no library peer is bindable on this toolkit-free box**;
  Marlin/Machete are the real W4A16 bar and are buildable on a Linux box with the toolkit
  (`docs/gpu/derive/D5_peer_builds.md`), so this lead is over a naive kernel, not over the field;
  attention **0.37–0.66× cuDNN at S≥2048** (loss — structural: SFU/serial-softmax bound on 20
  SMs; the FA2-style warp-specialized kernels were built and measured 2026-07-09 and win only
  **4–6% at S=4096**, now the default route there, tie at 2048, lose at ≤1024 — so the cuDNN
  long-S gap is honestly bounded, not closable by scheduling), wins fused-RoPE S≤512, causal
  D=64 S=512 (beats cuDNN+cutlass), and D=128 ldmatrix S≤1024. The 4096³ GEMM ships the
  **v2cs streaming epilogue** (+2.7%): **76.8% of cuBLAS-f16 / 80.4% of the honest f32-out
  peer** (the f16-out peer hides ~half of Wukong's f32 C-write traffic — both columns now
  printed; the residual is SASS-level).
- *Memory-bound, CPU*: streaming elementwise ≈1.1–1.6× C (NT-store dispatch), honest
  physics-ties at L3-resident sizes (relu, biasadd, hadamard); reductions/norms/scans/column
  family 1.4–107× vs scalar-left-by-gcc patterns — but note these are IEEE-serial C baselines
  (see G5 fairness work: a `-ffast-math` C column normalizes the reassociation share).
- *Transcendentals*: 4.7–12× scalar libm. **vs MKL VML the honest standing is a TIE for exp
  and log, and the 2026-08-06 "log 1.17–1.46× / exp 1.08–1.25× faster" reading is retracted.**
  Re-measured the same day against an **accuracy-matched** peer — VML run through
  `vmlSetMode(VML_LA)`, which the probe measures at 2 ULP for exp and 4 ULP for log on the
  timed band, exactly where Wukong measures 2 and 4 — five same-run best-of-40 ABBA rounds in
  one process (AC+charging 68–76%, nothing else running) read **exp 1.09–1.20× SLOWER at 2¹⁶
  and 2²⁰** and **log between 1.09× slower and 1.07× faster**. Only at n = 2²³ does either
  come near the floor, and even there only log clears it (1.48–1.66×) while exp straddles it
  (1.37–1.54×) — and that is Wukong's non-temporal store regime rather than its core. The earlier numbers were against VML's **default HA mode at
  ~0.5 ULP** — a strictly more accurate peer, so not a like-for-like comparison, and the
  mismatch was disclosed only in a source comment. **tanh is the one real win: 2.85–3.24× vs
  VML's fastest mode** (3.93–4.91× vs HA), bought with accuracy — Wukong's tanh is 44 ULP /
  3.5e-6 relative against VML's ≤1 ULP in every mode. The n = 2¹² row is withdrawn outright:
  at 16 KiB/array every arm's best-of-40 minimum is 0.5–1.4 µs against a **measured** 100 ns
  `Instant` granularity, so its "ratios" were ratios of 5-to-14 clock ticks and came out as
  exact small fractions.
  **The 2026-07-09 "the residual is algorithmic" reading was wrong** (it stood about a month,
  not longer), and it is worth recording why the instrument did not catch it: every lever
  tried against this gap — 8-bucket `vpermps` LUTs (2026-07-08, which did close a real
  ~1.7–2× loss), Estrin scheduling, the ×4/×6 ILP unroll, and a bit-identical `ldexp` exp
  tail that measured a ~10–15% loss and was **reverted** — was a change to the *polynomial*,
  while the cost was in the *dispatch*: the loop called its 8-lane kernel through a function
  pointer, and the Windows x64 ABI passes `__m256` in memory, so every 8 lanes paid a spill /
  indirect call / reload and lost the poly constants out of registers. Monomorphizing that
  dispatch is bit-identical on all 2³² f32 and worth **1.20–1.78×** on its own — the
  power-independent internal ratio (`WUKONG_VMATH_FNPTR=1` measures the old spelling in the
  same binary), and the only figure in this bullet that is not a cross-library ratio. The
  two-input kernel (`wukong_vmath2_f32` — activation backward, SwiGLU/GeGLU gates, pow),
  which spilled **two** `__m256` per call, took the same fix for **1.09–1.48×**.
  Accuracy exhaustively re-swept and unmoved: exp 1.625e-7 / ≤2 ULP over 2.24e9 values, log
  6.924e-7 / ≤12 ULP over every positive normal f32. Composites inherit the wins: log2 9.9×,
  log1p 7.7× vs C.
  *Three lessons for the instrument*: (a) a "residual is algorithmic" verdict needs a
  same-binary A/B against the alternative *structure*, not only against alternative math;
  (b) a vs-library ratio is only a claim if the library is in a **matched accuracy mode** —
  ours flipped from "win" to "tie" the moment the peer stopped being asked for 0.5 ULP;
  (c) every vs-library probe should carry a control column whose true ratio is 1.000. Ours
  now does — two of them — and they read **1.06–1.23× at 2²⁰ and once 2.303× at 2¹⁶** on
  identical code, with no power-state change. Nothing under ~1.4× survives that.

**M2. End-to-end model performance.** A compiler is judged on composed graphs, not op zoos:
fusion, no round-trips, layer-stack throughput. GPT-2-class `.wk` models exist and dispatch
to recognized kernels; GPU-resident 12-layer decode runs under CUDA-graph capture.
Standing (2026-07-11, three independent same-day rounds — all cross-checks ≤1e-6 rel,
serial==@parallel bit-exact, interp gate bit-exact): the 12-layer GPT-2-class stack runs
**~19–21× idiomatic C and 3.6–4.9× `-ffast-math` C single-core**. The honest PyTorch bar is
**`torch.compile` (TorchInductor max-autotune, fullgraph, warmed; eager does NOT count)**, run
same-run at both `set_num_threads(1)` and all-threads. Single-thread, Wukong **beats
compiled-torch-1T at both S=128 and S=512**. Multicore — after the **`@parallel` head-loop region**
(2026-07-10: an independent-iteration `for` loop with body-local scratch inside an `@parallel` fn
outlines into a `wukong_parallel_for` region; conservative affine-disjointness legality, serial
kernels inside each iteration so serial==parallel stays bit-exact; the model spells its attention
head loop that way naturally) — Wukong @parallel is **1.39–1.81× FASTER than all-threads compiled
torch at S=512 and 1.07–1.43× FASTER at S=128** (it also beats torch's strongest *eager* config,
1.07–1.37× @S=512; isolated per-side probes with torch at its best). Model @parallel scaling
reaches **4.7× on 16 cores** (default 4.44×, best 5.22×), tracking the ~5.0–5.4× same-run hybrid
MKL ceiling with T=1 at serial parity; residual headroom vs 16 physical cores is the parallel-GEMM
grain (M1). GPU training step still loses to eager PyTorch (GEMM-bound); the serving stack's
continuous-batching goodput ceiling: **Bcap=256 full-fill 85.6× vs fill=1** (graph-driven
scheduler bit-identical to eager, **1.14–1.27× over the honest static-batching peer** — the
batching-vs-scheduling decomposition is disclosed), opt-in int8-KV cache **3.88× smaller than
f32** (1.94× than f16).

**M3. Multicore scaling** of M1 with G4 preserved. Standing: strong (dot ~7.9×, max ~25×,
GEMM to ~104× vs single-thread C), but until the OpenMP C column lands the multicore rows
have no multithreaded peer — treat them as scaling demonstrations, not peer wins.

## Tier 2 — what makes it a language, not a kernel library

**M4. Compile time.** Standing: the most defensible headline — `compile-vs` (all four
languages as subprocesses, compile-to-object at -O2) shows ~7–12× vs gcc/g++/rustc (measured
~7–9× this session, ~9–12× on faster-clock runs); in-process JIT latency ~0.3–1.5 ms/kernel and ~13 ms for a full GPT-2 block source→optimized
MIR; GPU cold JIT 0.76 ms / warm cubin 0.16 ms (~10⁴–10⁵× vs Triton cold).

**M5. Static shape safety at zero runtime cost.** Tensor shapes are types; static dims unify
by equality, symbolic dims bind (with a `rigid` mode so a callee cannot lie about its return
shape), `?` defers to runtime. Standing: strong and regression-tested (`E05xx` suite), but a
runtime `?` dim does not *execute* (C0001), which limits real serving code — a Tier-1
language gap (see M6).

**M6. Expressiveness for real ML code.** Can a user write a transformer fwd+bwd+train loop in
pure Wukong without escaping? Standing: forward blocks yes (statically-shaped, recognized
idioms); training via `--train` (CLI transform, not in-language); **known blockers**: no heap
allocation / returned tensors, runtime `?` dims don't run, tensors can't be element-generic
(`Tensor[T,M,N]` rejected → dtype kernels duplicated), no fn pointers/closures. **File I/O now
runs** (✅): the `read_<T>`/`write_<T>` family (`T` in {`f32`, `f64`, `i32`, `i64`, `i8`, `u8`})
loads and stores headerless raw little-endian blobs and is gated interp == native bit-for-bit, so
weights can be **read off disk rather than synthesized** — and, built on it, **GPT-2 124M inference
now runs end-to-end** (✅). `examples/gpt2_infer.wk` loads the real pretrained OpenAI GPT-2 weights
(124,439,808 f32 parameters, exported from HuggingFace `transformers` by `tools/export_gpt2.py`)
from a flat little-endian blob via `read_f32` (token ids via `read_i32`), runs the full forward
pass — token + positional embeddings, 12 pre-LayerNorm blocks (biased QKV, 12-head causal
attention, tanh-GELU MLP, residuals), final LayerNorm, tied LM head → logits `[5, 50257]` — and its
next-token logits **match HuggingFace's reference to a relative max error of 1.87e-6** (max|Δ| =
2.44e-4 vs max|ref| = 130.28; **argmax next-token = 1757, " John", for the prompt "Hello, my name
is" — matching HF exactly**; verified by `tools/verify_gpt2.py`). This is a **numerical-correctness
/ capability result, inference only — not training, and no speed or throughput claim is made or
implied**. Two honesty caveats: **tokenization is external** (the `.wk` consumes integer token ids
the HuggingFace tokenizer produced — Wukong has no tokenizer), and the full run is **native-only** —
the interpreter cannot hold 124M parameters, so the native Cranelift-JIT backend is what executes it
(a reduced-config twin, `examples/gpt2_infer_small.wk`, runs **bit-identically on interpreter *and*
native** under the CI examples differential + opt-invariance gate).
(Multi-file `import a.b` *does* work — it splices items with
cycle/diamond dedup; only aliased/selective `import as` / `import x.{a,b}` stay partial.) These bound
how far "general programs a real user writes" can go today and are first-class improvement
targets, not footnotes.

**M7. Portability.** Same `.wk` → interp (oracle), Cranelift native, GPU offload,
GPU-native (whole-program MIR→PTX megakernel). Standing: real; GPU-native covers a subset
(UNSUPPORTED=skip). The previously documented general-lowering gaps are all **fixed**:
`PTX_VELEM` implements the Hadamard/Div binary modes and `PTX_NORM` the log-softmax/L2 ops;
`mrt_sreduce`/`mrt_sreduce_coop` implement the full `wukong_sreduce_f32` op set including
sumabs(9)/absdiff(10) (`parallel_abssum`); float→narrow-int casts saturate (Rust-`as`/Cranelift
semantics, `float_cast_narrow`); and pointer slots in the megakernel's shared frame are stored
unconditionally so SPMD threads no longer dereference a null base (`tensor_1d_kernels@O3`
illegal-address). Re-measured 2026-07-30 at HEAD (`WUKONG_GPU_REQUIRED=1`, 2 passed / 0 failed): the
`tests/run` corpus gate stands at **217/332 programs** matching the interp oracle at `-O0`==`-O3`
(**115** honest UNSUPPORTED skips, zero mismatches, zero device faults), and the megakernel gate at
**83 ran / 91 eligible** (8 launch-time declines). Absolute coverage rose 193→217 as the corpus grew
274→332 with the hardening campaign's fixtures, so the *fraction* moved 70.4%→65.4% — the added
fixtures are mostly recognizer/element-type cases gpu-native honestly declines. The corpus gates also
isolate any future device fault: a `CUDA_ERROR_ILLEGAL_ADDRESS` is a **process-fatal sticky**
CUDA error (measured on this driver: `cuDevicePrimaryCtxReset` returns Ok but the re-retain
still errors — only a process restart recovers), so the harness records the root fault on its
own loud ledger, marks the device lost, and reports every later program as NOT RUN — one
faulting program can no longer cascade into ~100 false failures, and skipped programs are never
reported as passed. The recognizer-offload GPU path and both CPU backends are unaffected and
bit/tolerance-exact.

## Tier 3 — supporting qualities

Stable diagnostic codes with `--explain`; single-binary toolchain (no LLVM needed to build,
test, or run natively); reproducible benches (`wukong_xbench`, `wukong_bench`) with
in-harness cross-checks that can only *fail* Wukong, never inflate it.

## Current top improvement targets (ranked by user impact × measured gap)

Closed in the 2026-07-09/10 round (see M1/M2 for the numbers and
`prompts/results/perf-sota-session3.md` for the full ledger): the large-GEMM single-core
tail (C-tile prefetch, 2048³ now at/above MKL-1c parity, see M1), the 256³ deliberate-serial standing
(now engaged at ~94–110% of MKL-all under the dynamic-claiming default), the serving goodput ceiling
(Bcap=256, 85.6× + honest static peer), the S=128 model regime (1.5–1.9× behind eager at campaign
start → now 1.07–1.43× AHEAD of *compiled* torch, see M2), and the honest-instrument holes (f32-out cuBLAS peer column; exp/log/model ranges
re-based on multi-state measurement). Measured-and-bounded
rather than closed: GPU long-S attention (warp specialization built; wins only 4–6% @S=4096 —
structural SFU bound), 4096³ GEMM (v2cs +2.7%; ~80% of the honest peer, residual is SASS-level),
and **exp/log vs VML** — this list called it "algorithmic" until 2026-08-06, when it turned out to
be a function-pointer dispatch boundary worth 1.20–1.78×; but the follow-up accuracy-matched
measurement the same day put exp and log at a **TIE** with VML's `VML_LA` core rather than ahead of
it, so the gap is *smaller and now measured*, not closed (see M1).
Remaining, ranked:

1. Multicore parallel-GEMM grain — the head-loop region + dynamic block-claiming landed (model
   @parallel now beats all-threads *compiled* torch at both S=128 and S=512; scaling 4.44–4.70× on
   16 cores). Dynamic claiming closed most of the old mid-size MKL-all gap (512³/1024³ now
   ~98–104%, was 70–83%); the residual is the smallest overhead-bound skinny shape (128×768·768ᵀ
   75–80%) and a clean verified-AC full-cube table (the 2026-07-11 attempt hit battery mid-run).
   Shared-pack, mid-pool, persistent-region, and fork-join alternatives are all measured/refuted
   in gemm.rs — a genuinely new decomposition idea is required for further mid-size gain.
2. **The H100 GEMM epilogue and wave overhead** — the measured Hopper gap, and the only one whose
   location is *partly* isolated. **Scope the diagnostic to where it ran:** the store-elided arm
   exists at exactly three shapes — sq2048 / sq4096 / sq8192, at 114.0 / 101.1 / 101.8% of cuBLAS —
   and it is the *clustered* `..._s4_mcb2_v2` arm, so it reads as a difference against that arm's own
   87.4 / 88.7 / 92.0%. At sq4096 and sq8192 the clustered arm is what ships and the conclusion
   carries: the mainloop is at or above the peer, so the **11.4 and 7.7 points** still owed on those
   two published rows are epilogue plus wave overhead, not the inner loop. **The two FFN endpoints of
   the old "3–27 points" range —
   gpt_d1024_down at 97.3% and gpt_d4096_up at 73.4% — had no nostore arm run on them at all**
   (`r3-config-sweep.log` sweeps only the three square shapes; the gpt rows exist only in
   `r10-vs-cublas-v2rule.log`, a round with no elided arm), so where *their* gap sits is unmeasured
   and the 73.4% row in particular is the one worth measuring first. Two further measured holes sit
   beside it: **1024³ is unresolvable in f16** against this peer at the ±5% bar (the peer's own floor
   is ±15.47%, so that shape needs a different measurement, not a different kernel), and **Hopper
   int8 is 22–52% of cuBLAS IMMA** because the 4050's tile search does not transfer. Everything past
   that is planning, not measurement — `docs/roadmap.md` and `docs/gpu/derive/`.
3. Language blockers that gate real programs: runtime `?` dims, heap tensors, dtype-generic
   tensors (M6; **file I/O now runs** — typed read/write blobs, interp == native).
4. Decode-path primitives: KV-cache append/decode, top-k/top-p sampling, argsort (CPU).
