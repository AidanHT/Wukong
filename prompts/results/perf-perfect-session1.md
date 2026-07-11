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

### 2026-07-10/11 — AC window #2 (quiet machine: all agents merged, 0 concurrent procs)
- **Measurement-law addition — the model bench's column ORDER favors Wukong for the B bar.** The
  torch columns run LAST in a full model round (after minutes of Wuk + C all-core heat): today
  torch-Tn read 165/226 ms inside the full round vs **78.5/85.1 ms isolated** (two adjacent
  torch-only probes, stable ±0.5%). The prior rounds' same-run comparisons happened to catch
  torch healthy, but the ordering bias is structural — for the B bar, use ISOLATED probes both
  sides (each side coolest) and take torch's BEST observed. Full-round same-run readings remain
  valid for Wukong-internal columns only.
- **S=128 isolated basis (this window, tiled+fast-path binary)**: Wuk par 66.0-70.9 ms; torch
  best Tn(comp) 85.1, Tn(sdpa) 78.5. Conservative (worst-Wuk vs best-torch): **1.20× faster than
  compiled-Tn, 1.11× vs eager-Tn** — consistent with prior rounds' 1.19-1.21×.
- **S=512 isolated basis**: Wuk par 266.1 ms; torch best Tn(comp) 370.2, Tn(sdpa) 288.2 (torch's
  own sdpa-all swung 863↔288 across its two adjacent isolated rounds — instability is torch-side;
  best taken). **1.39× faster than compiled-Tn, 1.08× vs eager-Tn.** NOTE @S=512 eager-sdpa-Tn
  BEATS compiled-Tn (288 vs 370) — torch's compiled path is not its best config here; Wukong
  beats BOTH.
- **Tiling A/B verdict: REFUTED for the model — reverted.** Same-state adjacent binary A/B
  (tiled main vs untiled 78bb719 worktree, AC, quiet), S=512 WUK_ONLY: par 243.7 vs 244.7 ms
  (WASH), 1c 1151.7 vs 1088.0 ms (tiled **5.9% slower** — per-tile kh/vt re-packing + 4× smaller
  GEMM calls). The load-balance hypothesis (48 vs 12 units) did not materialize: region grain was
  NOT the S=512 bottleneck. Also resolved: the earlier scary readings (tiled 1c 2204 ms, "8.28×
  scaling") were HEAT artifacts — 1c cool reads 1088-1152 for both spellings; the ABBA's 4th/5th
  runs caught the machine flipping to battery mid-experiment (par 487.9/480.1), reinforcing the
  power-check-every-round law. Model respelling reverted to the untiled head loop (doc comment
  records the measurement); outliner div/mod legality + differential fixture + skinny bench stay
  (general infrastructure). LESSON: an optimization that only restructures the ITERATION SPACE
  (not the arithmetic or locality of the hot kernels) needs the load-imbalance to actually be the
  bottleneck — measure the wash-vs-win BEFORE landing the respelling next time (a same-state
  two-binary A/B costs one worktree build).

- **Measurement-law addition #2 — CHARGING is a third instrument state.** After the battery flap,
  all-core par readings degraded ~25% (S=512 model par 244 → 297-329 ms across 11 runs) while the
  1c anchor stayed rock-stable (1019-1052 ms, all runs). The per-block flip was exonerated by an
  adjacent `WUKONG_GEMM_2D_SHARED=band` A/B (310.7 vs 310.5 — identical, as predicted by MAC-size
  analysis). Root cause: **battery charging at ~42 W (69% charge)** — the charger budget splits
  between charge and package, capping all-core sustained clocks; single-core fits the remainder.
  Law: check `PowerOnline` AND `Charging`/`ChargeRate` every round; reportable all-core rounds
  need AC + charge ≈ full (or explicit disclosure). This also retro-explains "bimodal MKL" and
  torch's own 863↔288 sdpa-all swing. Also noted: the T-sweep instrument's top point
  (`RAYON_NUM_THREADS=16`) makes gemm_pool decline (16 ≥ 16) → global-pool scheduling instead of
  the private physical-core pool; the default-config run is the honest T=16 point (charging-state
  reads showed no measurable difference, re-check when full).
- **Scaling sweep (CHARGING state, S=512, both orders + default top point)**: T=1 **1.00×/1.03×
  (width-1 fast-path parity confirmed in-curve)**, T=2 1.58×, T=4 2.47×, T=8 3.19-3.36×,
  T=12 3.36-3.51×, T=16 3.19-3.28×, default-pool 3.16-3.52×. Curve valid as SHAPE under charging
  cap; healthy-state absolute curve to be re-taken at full charge (morning cluster: par 244 ms
  ≈ 4.4-4.7×).

- **Target C central run landed (AC, single-core — charging-immune; docs/compile-floor.md tables
  filled).** compile-profile full corpus (296 files): **168.1 ms front→object total ≈ 0.57
  ms/file**; shares codegen+obj 73.4% (Cranelift codegen 91.7% / object-write 8.3%) > optimize
  16.3% > mir_build 6.5% > parse 1.8% ≈ sema 1.7% > lex 0.3%; throughputs lex 746 MB/s (floor),
  parse 24.1 Mtok/s (floor), sema 14.8 Mnode/s, mir_build 4.0 Mnode/s, opt 2.3 Mop/s, backend
  18.4 MB-obj/s. spawn-overhead: warm in-proc 0.35-1.0 ms vs spawned CLI 7.4-8.8 ms (**tax 7-8 ms
  fixed**), --emit=exe 170-190 ms of which link 162-182 (~95% rustc-link). compile-vs (CLI vs
  CLI, same-run): gcc 7.8-8.4×, g++ 7.4-8.7×, rustc 10.2-11.7× — the historical "9-12×" refines
  to **7.4-11.7×** with the disclosure that wukongc's CLI wall is ~90% spawn tax (its in-process
  core is 0.5-1.0 ms and got ~17% faster this campaign); the C(2) verdict + remaining
  driver-front-matter lever are written into the doc. C(1)/C(2) instrumentation and
  characterization: DONE; what remains for Target C is deciding whether the ~7 ms driver tax and
  the Cranelift-internal 73% merit further attack.

### 2026-07-11 — final healthy-state window (AC, charge ≥97%, trickle, quiet) + session close
- **Merged perf/skinny-gemm-grid** (fccfa9d): ONE square candidate (96,96) in the 2D ladder fixes
  the wide-M/narrow-N mis-shaping (512×768 grid 11×8→6×8, B transpose-packs nearly halved).
  Agent's ABBA: 512×768·768ᵀ 75→89% (zero overlap), 512×3072 76→87%; model S=512 ~8% faster;
  cubes provably unchanged (candidate yields <48 blocks at every square — pinned in the policy
  unit test). Skinny-M 2-row split hypothesis REFUTED and documented in-code. My own post-merge
  gate: full cargo test + gpu check green.
- **Target B — THREE independent same-day rounds, all pairings beat compiled torch.** Isolated
  probes per side (the ordering-bias law), per round vs TorchInductor max-autotune fullgraph
  (ATEN-pinned, disclosed) all-threads: S=512 **1.39× / 1.41-1.42× / 1.81×** faster; S=128
  **1.20× / 1.07-1.19× / 1.43×** faster. Vs torch's best EAGER all-threads config: S=512
  1.07-1.37× faster every round; S=128 0.98-1.23× (one worst-vs-best pairing at parity, all
  others faster). Vs compiled 1-thread: Wuk-1c faster at both shapes (S=128 274-304 vs 306-330;
  S=512 1021-1046 vs 1281-1375 healthy reads). Numerics cross-checked every round (compiled vs
  Wukong ≤1e-6 rel; serial==parallel BIT-EXACT). **The Target-B bar is met at both shapes.**
- **Target A(b) — final scaling curve (S=512, healthy state, per-T processes + default)**:
  T=1 **1.00×** (width-1 fast-path parity), T=2 1.70×, T=4 2.13×, T=8 3.00×, T=12 3.68×,
  T=16 **4.70×** (255.4 ms), default pool 4.44× (254.3 ms); a later same-state round read 5.22×
  (thermally-elevated 1c denominator — honest range 4.0-5.2×). Monotone, no T=16 inversion in
  the healthy state (the charging-state inversion was the cap, not structure). Machine ceiling
  shape (MKL-all/MKL-1c same-run) ≈ 5.0-5.4× — the model's curve tracks the FLOP-weighted
  kernel-ceiling composition; no Wukong-side sync/serial shortfall is visible in the curve.
- **Target A(a) — final full table (one conservative-ordering run, Wuk-par measured LAST/hottest
  per size; % of same-run MKL-all)**: 256³ **94%**, 512³ 76%, 1024³ 76%, 2048³ **91%**, 4096³
  **93%**; skinny: 128×768·768ᵀ 70%, 128×768·3072ᵀ 88%, 128×3072·768ᵀ **96%**, 512×768·768ᵀ 79%,
  512×768·3072ᵀ 85%, 512×3072·768ᵀ **92%**. Wuk-1c vs MKL-1c: 80-104% (≥100% at 2048/4096/
  512×768·768/512×768·3072). Mid-size cubes are RUN-VARIABLE in the healthy state (76-97% today
  across runs, 94-101% in earlier windows) — the honest statement is a RANGE, and A(a)'s
  "≈100% at every size" is met at small/large cubes and the best skinny shapes, NOT yet durably
  at 512³-1024³ (open: 76-97%) nor 128×768·768ᵀ (70%, scaling-limited at 151 MFLOP total).
- **Session status vs the three bars**: B MET (multiple rounds, break attempts: thread counts,
  isolated basis, shape variation, ordering discipline). A(b) MET in the physics-honest sense
  (curve tracks machine ceiling; T=1 parity; monotone). A(a) PARTIAL — mid-size cube variance
  and the smallest skinny shape remain the quantified gaps. C characterized to floor with
  instruments + docs (remaining decision recorded). Levers shipped this session: pool
  unification, pfor dynamic claiming, width==1 fast-path, per-block default, (96,96) candidate,
  backend verifier-off + parallel codegen, outliner div/mod legality. Refuted-with-evidence:
  model head-loop tiling (reverted), skinny 2-row split, TASK_MACS coarse grain, shared-pack
  small band (superseded).

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
- **parallel_for dynamic granule claiming shipped** (fe7fd04, WUKONG_PFOR_DYN=0 A/B escape):
  region iterations claimed from a padded atomic counter in ceil(n/(workers·8)) granules — the
  runtime half of the attention-tiling lever. Full gate green.
- **256³ small-band lever discovered — PENDING CLEAN CONFIRMATION.** In the AC window,
  `WUKONG_GEMM_2D_SHARED=0` (per-block packing, bypassing the shared-pack small band) lifted 256³
  Wuk-par from 185-193 GF/s (67-69% of adjacent MKL-all ~275-282) to **272.7 GF/s = 97% of
  adjacent MKL-all 281.7**; 512³ unchanged at 97% (already per-block). Hypothesis: the
  2026-07-09 "shared-pack wins the small band" measurement is obsolete under this session's pool
  unification — SHARED_MAX_MACS=2^26 is now a wrong default. The ABBA closer (shared default,
  re-run) came back CONFOUNDED: machine flipped to BATTERY + two concurrent agent cargo builds;
  MKL-all itself collapsed 282→193 (burst roofline stayed 136 — the all-core interference
  signature). No default flip until one clean AC ABBA confirms. Note shared-band Wuk-par read
  ~185-193 GF/s in every state, healthy or degraded — consistent with the shared path being
  self-limited (caller-thread packing barrier), but that's interpretation, not yet evidence.
- **Merged perf/backend-compile-time** (023b08f + d962001): Target C's backend attack. (1)
  Cranelift IR verifier (check-only pass, default ON upstream) now off in release / on in debug
  tests, WUKONG_CL_VERIFY override — agent-measured geomean **1.20× backend** over the 295-file
  corpus (provisional: battery-window same-run ratios). (2) Parallel per-function codegen
  (rayon map_init, shared read-only ISA+Decls, define in source order via define_function_bytes),
  default ON, WUKONG_PAR_CODEGEN=0 kill-switch: +2% corpus-wide but 15-46% on the 30
  multi-function programs (gpt2 ~20%). (3) emit_object_timed split: codegen 91.6% /
  object-write 8.4% — kills the "shrink the COFF container" lever (<9% ceiling). (4) ISA rebuild
  cost measured 0.3% — caching declined, tried-and-flat. Byte-identity: 296/296 corpus files
  identical old-binary-vs-new (serial AND parallel), plus 3 unit gates. JIT paths stay serial.
  Merge gated by my own unpiped full `cargo test` + `cargo check --features gpu --all-targets`.
- **Merged feat/outliner-divmod-tiling** (51a0f87..cd85919): the @parallel region outliner now
  accepts div/mod-tiled iteration spaces — `hh=t/C; tile=t%C` digit pairs proven disjoint by a
  two-level mixed-radix argument (index terms distributed via flatten_scaled_terms; digit atoms
  incl. flow/scope-tracked derived locals; per-array common RegionSig; H/L slot bounds with i128
  checked arithmetic). Conservative line pinned by decline tests (non-const divisor, div-only/
  mod-only, equal coefficients, overlapping extents). New differential fixture
  tests/run/parallel_divmod_tiling.wk (serial twin == @parallel, interp+native, -O0..-O3);
  autodiff loud-decline still pinned. I re-verified the DivMod bound arithmetic by hand at the
  model's shapes (both exactly tight: 255<256 test cfg, 98303<98304 at S=512 — tiles exactly
  fill their slots). Agent gate + my own post-merge combined gate.
- **Head×row-tile respelling RE-LANDED in the model** (model.rs wk_block): the head loop is now
  ONE flat `for t in 0..h*4` with hh=t/4, tile=t%4 — 48 units @H=12 instead of 12 (load balance
  on 16 workers) and 4× smaller scores scratch (1 MB→256 KB @S=512: L2-resident). Bit-identical
  per-row math (packing, dot order, mask, softmax unchanged) → outputs identical to untiled.
  block_dispatch_sets_pinned now PASSES on the tiled form (ONE wukong_parallel_for, serial
  sgemm_nt/sgemm_nt_alpha/norm_f32 inside, none outside) and interp_gate_bit_exact green.
  Perf A/B vs untiled DEFERRED to the next AC window (battery now).
- **Width==1 serial fast-path shipped** (T=1 parallel-path tax closed): at `RAYON_NUM_THREADS=1`
  the parallel entries used to pay 0.62-0.80× of the serial kernels (2D-block packing walked on
  one worker; pool handoffs with no second core to win). Now `wuk_pool_width() <= 1` routes to
  the serial sibling — bit-identical by the standing serial==parallel law — at 6 sites:
  gemm_dispatch (covers ALL f32 GEMM `_parallel` entries), norm_f32/norm_affine_f32, velem_f32
  (joined its small-N fallback), sreduce/argreduce (joined their 1-chunk fallback).
  `wukong_parallel_for` deliberately NOT fast-pathed: region bodies need the pool's 16 MiB
  worker stacks, and its width-1 shape (one par_iter job) is already minimal. Verified same-run
  (battery-legitimate adjacent A/B): matmul T=1 par/serial 1.08×/1.00×/0.88× @256³/512³/1024³
  (residual = within-run drift), model T=1 par/serial 0.99×, interp gate + serial-vs-parallel
  BIT-EXACT end-to-end. The remaining ~55 `_parallel` entries (bf16/col/row/scan families)
  follow the same recipe if a T=1 read ever shows residual tax — fix-where-measured.
- **Merged perf/compile-floor follow-up** (4e97e92 + 4aa2ea8): char-safe `trunc` (byte-slice
  panic on multibyte filenames fixed), docs reconciled with the backend codegen/object-write
  split; agent's full-corpus PROVISIONAL shares: codegen+obj 76-80% > optimize 12-16% >
  mir_build 5-6% > sema ≈ parse ~1.3% > lex 0.2%; backend split codegen ~92.5% / object-write
  ~7.5% (small-unit floor is codegen-side setup, not the COFF container).
- **Head×row-tile respelling: probe PASSED but the conclusion was WRONG — corrected.** The
  standalone probe (whole `@parallel` fn = single top-level loop) takes the fn-level chunking
  path (`<fn>$par`), which handles div/mod fine and is interp==native exact. But the model's
  MID-FUNCTION loop uses the region outliner (`wukong$par$N`), whose affine-disjointness legality
  DECLINES `hh=t/T; tile=t%T` — the tiled loop then degenerates into a serial t-loop of
  `_parallel` kernels (~7 fork-joins per tile per layer — far WORSE than untiled). Caught by the
  `block_dispatch_sets_pinned` test (the reason that test exists); model.rs tiling reverted.
  Fix in flight: `feat/outliner-divmod-tiling` extends the legality to the 2-coordinate
  (t/C, t%C) bijection decomposition — a general compiler improvement; the model respelling
  re-lands after it. LESSON: the whole-fn and mid-function @parallel paths have DIFFERENT
  legality analyses — a probe must exercise the exact path the real code takes.
