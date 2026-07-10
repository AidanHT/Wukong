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
