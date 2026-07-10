# perf-perfect campaign — session 1 ledger (2026-07-10 →)

Directive: three targets to their defined bars with zero shortfall, proven by honest, durable,
reproducible measurement. This file is the running ledger. Nothing in here is a claim until it has
survived multiple independent AC-state rounds and an active attempt to break it.

## The three bars, operationalized for this machine

Machine: Intel Core Ultra 7 155H (Meteor Lake, 6P + 8E + 2LPE = 16 physical cores, 22 threads),
Windows 11, throttling laptop. AVX2 (MKL dispatches AVX2 here too — apples-to-apples 256-bit).

### Target A — parallel-GEMM grain + near-linear end-to-end scaling
- **A(a)**: Wukong parallel f32 GEMM ≈100% of all-threads oneMKL, same-run MKL-anchored, at
  256³/512³/1024³/2048³ (+4096³), plus the skinny transformer shapes (128×768×768, 128×3072×768).
  MKL-all is the existence proof of what this silicon sustains all-core.
- **A(b)**: end-to-end 12-layer model scaling near-linear *in the physics-honest sense*: on hybrid
  silicon "16× on 16 cores" does not exist — the all-core ceiling is set by heterogeneous cores +
  all-core clock (the machine's own MKL-all/MKL-1c healthy-state ratio ≈5-5.3× is the demonstrated
  GEMM-compute ceiling shape). The A(b) bar: the model's end-to-end scaling curve (T = 1..16, via
  RAYON_NUM_THREADS) must track the FLOP-weighted composition of its kernels' individual ceilings —
  i.e. no sync/serial-fraction shortfall attributable to Wukong. Quantified by the Amdahl inventory
  (prompts/results/serial-fraction.md) + measured curves. Full curve reported, never one point.

### Target B — beat fused torch.compile end-to-end
- Bar: Wukong steady-state end-to-end forward (12-layer GPT-2-124M-class, S=128/512 minimum,
  fp32 CPU) faster than **TorchInductor max-autotune, fully warmed, fullgraph (no breaks),
  strongest available torch (2.12.1+cpu), all-threads AND 1-thread variants**, numerically
  cross-checked, same machine/shape/precision. Eager-torch comparisons are explicitly NOT the bar
  (the 2026-07-09 campaign "flip" was vs eager; docs correctly scope it).
- Regime discipline: torch.compile's compile wall-time is cold-start-regime and reported
  separately; it does not excuse Wukong losing steady-state, nor does Wukong's fast compile count
  toward B.

### Target C — compile time
- C(1): code→object at a justified floor: full per-stage pipeline profile (lex/parse/sema/
  mir_build/opt/codegen/emit), every avoidable cost removed, floor defined from irreducible work +
  strongest comparable toolchain throughput (gcc/g++/rustc measured by compile-vs, historically
  wukongc ~9-12× faster; that ratio is re-proven this campaign, not assumed).
- C(2): in-process vs spawned-toolchain overhead characterized (CLI spawn tax, --emit=exe rustc
  link spawn), winning path selected per scenario, spawn overhead eliminated where in-process wins.

## Measurement law for this campaign
1. **Power state is part of the instrument.** 2026-07-10: discovered the machine ON BATTERY —
   sustained-load numbers read 2-4× worse machine-wide (both Wukong AND torch), while short bursts
   still boost. ALL cross-session comparisons and ALL reportable rounds require AC + healthy
   roofline; every bench header now prints power state (instrument change in flight). Battery-state
   same-run ratios are usable ONLY for development A/B iteration, never for the ledger.
2. Same-run/adjacent A/B only; MKL-anchored for cross-binary GEMM; first post-rebuild run
   discarded; unpiped test gates; release binary only.
3. Every result needs ≥2 independent AC rounds + an active attempt to break it (vary shapes,
   thread counts, ordering) before it may be called met.

## Session log

### 2026-07-10 — setup + environment discoveries (no reportable perf)
- **torch.compile CPU works on this Windows box** (both torches: system 2.12.1+cpu, venv
  2.6.0+cu124). Recipe: vcvars64 (VS 18 Insiders, MSVC 14.50) env + `LIB += <base_python>\libs`
  (else LNK1104 python313.lib) + numpy present. Verified with a toy MLP: compiled==eager to
  2.4e-4 fp32. The model bench's eager-only torch peer is therefore obsolete as the strongest
  baseline; compiled peer integration in flight (branch perf/torch-compile-peer).
- **Battery incident**: model-bench round read Wuk-par S=512 = 967 ms vs session-3 close ~312 ms;
  torch T1 similarly ~4× degraded; roofline decayed 146→101 across minutes; PowerOnline=False.
  Round recorded as INVALID for cross-session comparison. Within-round (same battery state):
  Wuk-par 1.14× faster than eager-Tn @S=128; 1.09× behind @S=512. NOT reportable.
- **Thread-cap instrument**: RAYON_NUM_THREADS=T already caps both the global pool and the GEMM
  pool (gemm_pool declines when physical ≥ current_num_threads) — full scaling curves need no
  runtime change, just per-T re-exec.
- OPEN-ANOMALY: battery round read Wuk(1c) @512³ at 66% of MKL-1c (history: 97-100%). Re-test on
  AC before diagnosing.
- Battery matmul round (invalid for ledger, one sample): Wuk-par/MKL-all 83% @256³, 90% @512³,
  98% @1024³; MKL-all degraded @2048³ (313 GF/s) so the 150% read there is a degraded-peer
  artifact, disclosed.
- Workstreams dispatched on branches: perf/torch-compile-peer (compiled-torch peer + power-state
  header + WUK_ONLY/TORCH_ONLY sweep knobs), perf/gemm-grain (dynamic block scheduling, env-gated,
  bit-exact), perf/compile-floor (compile-profile + spawn-overhead modes + floor draft),
  serial-fraction analysis (read-only report).

### 2026-07-10 (cont.) — analysis results, feasibility probes, first merges
- **Serial-fraction analysis landed** (prompts/results/serial-fraction.md): literal serial code is
  ~1% — the bound is the parallel work's own hybrid ceiling (~9.3 P-equivalents) PLUS ~108-132
  fork-joins/forward bouncing private↔global pools ~6-7×/layer, which parks cores and blocks clock
  ramp (the "no clock upside" mechanism). Ranked levers: persistent hot team (+20-50%, HIGH risk),
  pool unification (+10-20%, LOW), finer skinny-M grid @S=128 (+10-15%), head×row-tile attention
  grain (+5% @S=512), residual fusion + parallel ln_f (+1-3%).
- **torch.compile fullgraph feasibility PROVEN on the exact model** (bench-generated script +
  blobs): 0 graph breaks. Inductor's CPP GEMM template is broken on Windows/MSVC
  (`cpp_CppMicroGemmFP32Vec not found`) → strongest WORKING config = max-autotune with
  `max_autotune_gemm_backends="ATEN"` (MKL GEMMs + Inductor fusion; disclosed, and ATen-MKL is
  torch's strongest CPU GEMM anyway). DEV-GRADE (battery, same-run ratio): compiled/eager =
  **1.86× @S=128**, 1.04× @S=512; correctness 8.7e-7. Implication: the eager-torch "flip" likely
  becomes a ~1.5-1.9× LOSS at S=128 against the true Target-B bar; S=512 roughly carries over.
  compile wall ≈70-90s (cold-start regime, reported separately).
- **Merged perf/gemm-grain** (3 commits, gemm.rs only, gate 500 green): WUKONG_GEMM_DYN atomic
  claim-queue scheduler (default OFF pending A/B), WUKONG_GEMM_TASK_MACS grain sweep,
  WUKONG_GEMM_STEAL_ORDER LPT probe, unconditional BN tail rebalance (1024³ (144,192)→(132,176)).
- **Pool unification shipped** (40d49a7, lever #2): run_on_wuk_pool routes _parallel norms/velem/
  reductions + parallel_for regions onto the physical-core GEMM pool (WUKONG_POOL_UNIFY=0 A/B
  escape). Also fixed a latent init-order bug — the 16 MiB-stack build_global lost the race to
  the first _parallel norm, so region bodies (~1.5 MiB frames) had been running on rayon's
  default 2 MiB stacks. Full 45-suite gate green (unpiped).
- **Merged perf/compile-floor**: `compile-profile` (per-stage pipeline breakdown) +
  `spawn-overhead` (in-process vs CLI spawn vs exe-link) bench modes + docs/compile-floor.md
  draft. PROVISIONAL shape finding (battery, shares only): **codegen+obj ≈74% of code→object**,
  optimize ≈19%, front-end ≈7% — the full-pipeline correction to "optimizer is 80-85%" (that was
  front→O2 only). Fixed ~6.5 KB COFF floor dominates small objects; --emit=exe is ~95% link
  (rustc subprocess). Target C's lever is the BACKEND, not the optimizer.
- Process lesson re-learned the hard way: a background `cargo test 2>&1 | tail` reported tail's
  exit code — the gate was re-run unpiped before committing (the standing rule exists for a
  reason).
