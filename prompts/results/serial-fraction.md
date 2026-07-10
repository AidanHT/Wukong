# Serial-fraction / sync-overhead inventory — end-to-end GPT-2-124M `@parallel` forward

**Target:** `crates/wukong_xbench/src/model.rs` `bench_model` — 12 pre-LN decoder blocks + final
LayerNorm, `d=768, h=12, hd=64, dff=3072`, at `S=128` and `S=512`.
**Question:** why does the `@parallel` forward scale only ~2.6–6× on 16 physical cores, and what
would push it toward linear.

**One-sentence answer (derived below):** the forward is *not* bounded by a classical serial code
fraction (the literal serial code is ~1% of single-core time). It is bounded by (a) the **~3.9×
parallel-GEMM ceiling of this specific 6P+8E+2LP-E hybrid** — which already folds in E-core
heterogeneity, the all-core clock derate, and mid-size memory-bandwidth saturation — and (b) a
**structural clock-ramp loss**: the forward forks and joins ~110–132 parallel regions per forward
across **two different rayon pools**, parking the cores between regions, so the hardware clock
governor never ramps and the parallel path is stuck at a chronically low P-state while single-core
and MKL/torch ride turbo. Everything else (the 12-head attention cap, sub-gate velem at S=128, the
serial final LayerNorm, the copy passes) is a smaller additive tax on top.

---

## 0. The two-tier parallel structure (what actually runs)

The `@parallel` block (`wk_block(cfg, true)`, `model.rs:279–400`) has two tiers:

* **Tier 1 — whole-`[S,D]` recognized ops** dispatch a `_parallel` kernel each. These run on the
  **private GEMM pool** (physical-core count = 16; `gemm.rs:105 gemm_pool()`) for the GEMMs, and on
  the **global rayon pool** (logical-core count ≈ 22, 16 MiB stacks; `lib.rs:326`) for norms/velem.
* **Tier 2 — the per-head attention loop** (`for hh in 0..12`) is outlined by `wukong_mir_build`
  into **one `wukong_parallel_for` region** (`lib.rs:326`), which runs on the **global pool** and
  splits `[0,12)` into `min(current_num_threads, 12) = 12` contiguous chunks of 1 head each
  (`lib.rs:353–364`). Inside the region every per-head kernel is the **serial** twin
  (`wukong_sgemm_nt_alpha`, `wukong_norm_f32`, `wukong_sgemm_nt`) — the region supplies the
  threading across heads, so nesting parallel kernels is neither needed nor allowed
  (pinned by `model.rs:1766–1789 block_dispatch_sets_pinned`).

The printed per-layer dispatch set decodes as:

| Printed | Count | Op in the source | Tier | Pool |
|---|---|---|---|---|
| `norm_affine_f32_parallel` | 2 | LayerNorm1, LayerNorm2 | 1 | global |
| `sgemm_nt_parallel` | 5 | Q, K, V proj; WO proj; W2 down-proj | 1 | private GEMM |
| `sgemm_nt_epi_parallel` | 1 | W1 up-proj + fused GELU | 1 | private GEMM |
| `velem_f32_parallel` | 2 | residual `a=x+attn·Woᵀ`, residual `out=a+ff1·W2ᵀ` | 1 | global |
| `parallel_for` | 1 | the `for hh in 0..12` head region | 2 | global |
| `sgemm_nt_alpha` | 1 | per-head `scores = α·(Qh·Khᵀ)` | **inside region** | serial |
| `norm_f32` | 1 | per-head causal row-softmax | **inside region** | serial |
| `sgemm_nt` | 1 | per-head `ah = scores·Vhᵀ` | **inside region** | serial |

**Resolving the "serial straggler" question in the brief:** the `1x norm_f32`, `1x sgemm_nt`,
`1x sgemm_nt_alpha` are **NOT genuinely serial at top level** — they are the per-head kernels that
run *inside* the outlined `wukong_parallel_for` region (one call site each in the MIR, executed 12×
across the 12 head-chunks). They are threaded across heads by the region. This is confirmed
structurally by the unit test at `model.rs:1766–1789`, which splits the MIR into the
`fn wukong$par$…` region body vs the rest and asserts those three serial symbols appear **only
inside** the region and that no serial `sgemm_nt` appears outside it. So they are correct and fine —
they are per-head serial kernels inside a parallel region, exactly the design.

The genuinely-serial (single-thread) work in the `@parallel` column is instead:

* the **final LayerNorm** (`wk_final_ln`, `model.rs:404`, compiled **without** `@parallel`) — one
  serial `wukong_norm_affine_f32` per forward;
* the two **residual velem adds at S=128 only**, which fall **below their parallel gate** and run
  the serial kernel (see §4);
* per-head glue (Qh/Kh extraction, V-transpose, **causal mask (O(S²) writes/head)**, attn scatter) —
  these are plain, non-recognized loops (velem declines the offset-indexed / conditional forms —
  mir_build `lib.rs:10711-10752, 11673-11681`); they are *inside* the region so they parallelize
  across heads, but each is serial *within* a head and the mask grows as O(S²).

**The two `nrm[i]=x[i]` / `nrm[i]=a[i]` copies are NOT a separate cost.** mir_build recognizes a bare
copy loop as a velem copy (`VE_ID`, a=1,c=0), but the observed dispatch shows exactly **2×** velem
(the two residuals), not 4× — so the copies do not survive as distinct parallel passes: copy
propagation folds the copied buffer into the LayerNorm's reduction input (the norm reads `x`/`a`
directly and writes `nrm`), leaving no separate copy pass in the optimized MIR. (The recognizer would
otherwise emit them; the `2×` count is the authoritative ground truth from the bench printout.)

---

## 1. Per-op inventory (one layer)

Constants: `D=768, HD=64, H=12, Dff=3072`. MACs = `m·n·k`; FLOP = `2·MAC`. Gates: GEMM parallel
`PAR_MIN_MACS = 2^23 = 8.39M` (`gemm.rs:39`); shared-vs-block split `SHARED_MAX_MACS = 2^26 = 67.1M`
(`gemm.rs:183`); velem parallel `velem_par_min = 262 144` elems (`velem.rs:97`); norm parallel = no
size gate, forks over rows unconditionally (`norm.rs:853`).

### S = 128 (per layer). Per-layer FLOP ≈ 1.862 GF; ×12 = **22.35 GF/forward**

(copy `nrm=x`/`nrm=a` folded into the norm input — not a separate row; see §0.)

| # | Op | dispatch | shape (m×n×k or rows×cols) | MACs | out bytes | ~roofline class | est. share of **1-core** layer time |
|---|---|---|---|---|---|---|---|
| 1 | LN1 | norm_affine ∥ (rows=128) | 128×768 | — | 0.39 MB | mem-bound | ~0.4% |
| 2 | Q proj | sgemm_nt ∥ | 128×768×768 | 75.5M | 0.39 MB | compute | 10.8% |
| 3 | K proj | sgemm_nt ∥ | 128×768×768 | 75.5M | 0.39 MB | compute | 10.8% |
| 4 | V proj | sgemm_nt ∥ | 128×768×768 | 75.5M | 0.39 MB | compute | 10.8% |
| 5 | attention (×12 heads, in region) | parallel_for(12); per-head serial | scores 128²·64 + PV 128²·64 per head | 25.2M total | 0.13 MB/head | compute+mem | 2.7% |
| 6 | WO proj | sgemm_nt ∥ | 128×768×768 | 75.5M | 0.39 MB | compute | 10.8% |
| 7 | residual a=x+· | velem **serial** (98 304 < gate) | 98 304 elems | 0.39 MB | mem-bound | ~0.2% |
| 8 | LN2 | norm_affine ∥ | 128×768 | — | 0.39 MB | mem-bound | ~0.4% |
| 9 | W1 up + GELU | sgemm_nt_epi ∥ | 128×3072×768 | 302M | 1.57 MB | compute | 32.4% |
| 10 | W2 down | sgemm_nt ∥ | 128×768×3072 | 302M | 0.39 MB | compute | 32.4% |
| 11 | residual out=a+· | velem **serial** (98 304 < gate) | 98 304 elems | 0.39 MB | mem-bound | ~0.2% |

FLOP split: **projections (Q/K/V/WO) 32.4%, attention 2.7%, MLP 64.9%.**
On one core the model is compute-bound: the six whole-`[S,D]` GEMMs + attention are ≈ 99% of layer
time; norms/velem/copies/glue are ≈ 1% (matches the `model.rs:57` "~1%" note).

### S = 512 (per layer). Per-layer FLOP ≈ 8.053 GF; ×12 = **96.6 GF/forward**

| # | Op | dispatch | MACs | out bytes | est. share of **1-core** layer time |
|---|---|---|---|---|---|
| 1 | LN1 | norm_affine ∥ (rows=512) | — | 1.57 MB | ~0.4% |
| 2–4 | Q/K/V proj | sgemm_nt ∥ | 302M each | 1.57 MB each | 10.0% each |
| 5 | attention (×12 heads, in region) | parallel_for(12); per-head serial | scores 512²·64 + PV 512²·64 per head = 33.6M/head, 403M total | 1.0 MB/head (scores) | **10.0%** |
| 6 | WO proj | sgemm_nt ∥ | 302M | 1.57 MB | 10.0% |
| 7 | residual a=x+· | velem ∥ (393 216 ≥ gate) | 393 216 elems | 1.57 MB | ~0.3% |
| 8 | LN2 | norm_affine ∥ | — | 1.57 MB | ~0.4% |
| 9 | W1 up + GELU | sgemm_nt_epi ∥ | 1.21G | 6.29 MB | 30.0% |
| 10 | W2 down | sgemm_nt ∥ | 1.21G | 1.57 MB | 30.0% |
| 11 | residual out=a+· | velem ∥ (393 216 ≥ gate) | 393 216 elems | 1.57 MB | ~0.3% |

FLOP split: **projections 30.0%, attention 10.0%, MLP 60.0%.** The two shifts vs S=128 that matter:
attention grows 2.7%→10% of FLOPs, and every per-head score matrix is now `512²·4 = 1 MB` (spills
L2), and the per-head serial work grows **as S²**.

---

## 2. Amdahl / hybrid-ceiling model

**Classical Amdahl on the literal serial fraction is the wrong model here.** The serial code is ~1%
of single-core time → naive Amdahl ceiling on 16 cores would be `1/(0.01 + 0.99/16) ≈ 14×`. Observed
is 3–6×. The gap is not a serial *code* fraction; it is that the **parallel work itself does not
scale 16×** on this box. Three physics-level derates fold into the achievable per-component speedup:

* **Core heterogeneity.** 6 P + 8 E + 2 LP-E. Treating a P-core as 1.0, E ≈ 0.35, LP-E ≈ 0.25 FMA
  throughput → aggregate ≈ `6 + 8·0.35 + 2·0.25 = 9.3` P-core-equivalents, i.e. a hard ~9× ceiling
  over one P-core even at perfect balance.
* **All-core clock derate.** The 16-core burst is package-power-limited to a lower clock than the
  single-core turbo baseline (~2.8 vs ~4.5 GHz ⇒ ×0.6). This is *already baked into* the runtime's
  own measured GEMM scaling.
* **Memory-bandwidth saturation** on the mid-size, skinny GEMMs of this model (M=128/512).

The runtime's own adjacent-run measurement is the anchor: **the parallel GEMM tops out at ~3.9× over
single core at 1024³, ~3.1× at 512³** on this 16-physical pool (`gemm.rs:96–104`). The model's GEMMs
are skinnier than square, so use **s_gemm ≈ 3.5× (S=128)** / **≈ 3.9× (S=512)** (S=512 has the finer
2D grid — see §4 — so it balances the hybrid better).

Per-component achievable speedups on this box:

| Component | FLOP share (128 / 512) | speedup s_i | rationale |
|---|---|---|---|
| Projections (Q/K/V/WO) | 32.4% / 30.0% | 3.5 / 3.9 | GEMM ceiling; skinny-M coarser at 128 |
| MLP (up/down) | 64.9% / 60.0% | 3.5 / 3.9 | GEMM ceiling |
| Attention region | 2.7% / 10.0% | ~5 / ~4 | 12 heads on ≤12 cores; E-core straggler grows as S² |

**Predicted overall speedup** = `1 / Σ(f_i / s_i)`:

* **S=128:** `1/(0.324/3.5 + 0.649/3.5 + 0.027/5) = 1/(0.0926+0.1854+0.0054) = 1/0.283 ≈ 3.5×`.
* **S=512:** `1/(0.30/3.9 + 0.60/3.9 + 0.10/4) = 1/(0.0769+0.1538+0.025) = 1/0.256 ≈ 3.9×`,
  minus the S²-straggler + memory drag in attention → effective **~3.3–3.6×**.

**Reconciliation with the observed numbers.**

* **AC rounds (3.1–3.6×):** match the model directly. The parallel path is at the ~3.9× GEMM
  ceiling; single core rides full turbo. This is the honest, steady-state number.
* **Battery round (6.19× / 2.58×):** high variance is expected because battery power management is
  the dominant, uncontrolled variable. When battery caps *single-core* turbo (throttling the 1c
  baseline), the ratio **inflates** toward the true core-count scaling (→ 6.19×) — the parallel path
  is *already* clock-limited so it loses relatively less. A round where fork-join/attention overhead
  and clock-ramp failure dominate reads **low** (→ 2.58×). The 6.19/2.58 spread is itself the
  clearest symptom of the clock-upside asymmetry (§6): the scaling ratio is a function of how much
  turbo the *single-core baseline* is allowed, which the parallel path can never match.
* **Direction S=128 vs S=512** is thermal/battery-state-dependent and should not be over-read:
  S=512 has better GEMM block-balance (finer 2D grid) pulling scaling *up*, but a larger,
  S²-straggler-bound, memory-heavier attention region pulling it *down*. Which wins depends on the
  power state of the round.

---

## 3. Sync / fork-join census

Every `_parallel` kernel call is one fork-join; the head region is one more. Default GEMM path is
`sgemm_2d_blocks` (`gemm.rs:1512`): `pool.install(body)` + one `into_par_iter` over `nbi·nbj` C-tile
tasks — **one fork-join per GEMM call** (the shared-pack path with per-K-block barriers only fires
below `2^26` MACs, which none of the whole-`[S,D]` GEMMs hit here). Norm/velem/head-region fork-join
the **global** pool via `into_par_iter` / `wukong_parallel_for`.

**Fork-joins per layer:**

| | private GEMM pool | global pool | total |
|---|---|---|---|
| GEMMs (Q/K/V/WO/W1/W2) | 6 | — | 6 |
| norms (LN1/LN2) | — | 2 | 2 |
| residual velem | — | 0 (S=128, serial) / 2 (S=512) | 0 / 2 |
| head region parallel_for | — | 1 | 1 |
| **per layer** | 6 | 3 / 5 | **9 (S=128) / 11 (S=512)** |

**Per forward:** `12 × 9 = 108` (S=128) / `12 × 11 = 132` (S=512), plus the serial final LayerNorm.

**Pool ping-pong.** Within each layer the work bounces private↔global ~6–7× (LN→Q, V→head,
head→WO, WO→velem, LN2→W1, W2→velem). Each switch **parks one 16/22-worker pool while it wakes the
other** — ~75–84 park/unpark waves per forward. The private GEMM pool sleeps through every
norm/velem/head phase; the global pool sleeps through every GEMM phase.

**Per-entry cost & total share.** Entry is a parked-pool wake: `install`/`into_par_iter` signals the
sleeping workers, the join waits for the **slowest to arrive** — worse on this hybrid because a
parked E-core is slow to spin up (documented repeatedly, e.g. `gemm.rs:36–38`, `velem.rs:92–95`).
Estimate ~5–15 µs/entry (central ~8 µs). Total fixed fork-join **latency**:
`108 × 8 µs ≈ 0.86 ms` (S=128), `132 × 8 µs ≈ 1.06 ms` (S=512).

**This raw latency is a *minor* term** (~1–2% of a ~30–150 ms parallel forward). Its importance is
**indirect**: the park-between-regions pattern is what prevents the clock from ramping (§6) — that
is the expensive consequence, not the wake latency itself.

---

## 4. Parallel-gate audit (which ops actually go multicore)

| Op | gate | shape at S=128 | S=128 verdict | shape at S=512 | S=512 verdict |
|---|---|---|---|---|---|
| Q/K/V/WO/W1/W2 GEMM | `2^23 = 8.39M` MACs (`gemm.rs:39`) | 75.5M–302M | **∥ correct** (all `2d_blocks`, ≥2^26) | 302M–1.21G | **∥ correct** |
| per-head scores/PV | (serial in region) | 1.05M/head | serial-in-region (correct — heads threaded) | 16.78M/head | serial-in-region (correct) |
| LN1/LN2 norm_affine | **none** — forks over rows (`norm.rs:853`) | 128 rows × 768 | ∥ but marginal: 128 tiny rows on 22 workers, ~1–2 µs/row → fork-join ≈ the work | 512 rows × 768 | ∥ healthy |
| residual velem | `velem_par_min = 262 144` (`velem.rs:97`) | **98 304 < gate → SERIAL** | **below gate** (runs serial twin) | 393 216 ≥ gate | **∥ correct** |
| per-head softmax `norm_f32` | (serial in region) | rows=128,cols=128 | serial-in-region | rows=512,cols=512 | serial-in-region (memory-heavy: 1 MB scores/head) |
| copy `nrm=x`, `nrm=a` | velem-recognized then eliminated | — | **folded into norm input** (obs. 2× velem) | — | **folded into norm input** |
| final LayerNorm | compiled w/o `@parallel` | 128×768 | **serial** | 512×768 | **serial** |

**Gates that are right:** all six whole-`[S,D]` GEMMs (well above 2^23 at both sizes); velem at S=512.
**Gates/decisions that leave performance on the table:**

1. **velem residual runs serial at S=128** (98 304 < 262 144). Two full `[S,D]` serial memory passes
   per layer. Small (~0.2% ×2 ×12) but pure loss.
2. **norm forks unconditionally even at 128 rows.** With 128 short rows on the ~22-worker global
   pool, the fork-join wake can rival the per-row work — a norm-specific version of the same
   park/unpark tax. A row-count gate (skip the fork below, say, ~256 rows) would avoid a cold wake.
3. **The final LayerNorm is unconditionally serial** (compiled w/o `@parallel`) — a memory-bound
   serial pass that scales the forward's floor (the copies, by contrast, are folded into the norm).
4. **Skinny-M coarse GEMM grid at S=128.** `select_2d_block_shape` (`gemm.rs:1449–1473`) takes the
   `m ≤ MC=144` skinny path at S=128: `nbi=1`, all parallelism in N, giving only ~24 column-blocks
   for Q/K/V/WO/MLP-up. 24 tasks on 16 workers → up to 2× tile imbalance and a coarse E-core
   straggler. At S=512 the normal path yields ~88–144 blocks → work-stealing balances the hybrid.
   This is a concrete reason S=128 GEMM scales below S=512 GEMM.

---

## 5. Ranked levers (fix → payoff → risk → bit-exactness)

Payoffs are end-to-end via the §2 model. "Bit-exact" law: serial == parallel must stay bit-for-bit;
any fixed-chunk / row-mapped / single-owner-tile scheme is safe (no accumulation reorder), which all
of these respect.

### Lever 1 — Persistent, hot, spinning worker team; ONE region per forward (the clock lever)
**Fix.** Replace the ~110–132 per-forward fork-joins across two pools with a single persistent team
that stays *hot* (busy-wait, no park) for the whole forward and takes work via an atomic handoff
queue — the pattern MKL/torch use. The building block already exists: the `WUKONG_GEMM_2D=0`
`sgemm_persistent_region` (`gemm.rs:1213`) is a `broadcast` + in-region `Barrier`; generalize it
across norm/velem/attention and hold it open across ops instead of per-call.
**Payoff.** Largest. This is the fix for the clock mystery (§6): keeping cores continuously loaded
lets the governor ramp and *hold* a high P-state, recovering the ~1.2–1.5× effective-clock gap the
parking pattern forfeits ⇒ **~+20–50% end-to-end at both S** (multiplicative on the whole forward),
plus it removes the ~1–2% raw fork-join latency.
**Risk.** HIGH — invasive runtime rework; must keep the barrier/broadcast width discipline
(`gemm.rs:1216–1220`) and the big-stack requirement (`lib.rs:332–349`).
**Bit-exact.** Safe — each kernel keeps its existing fixed decomposition; only *when* workers sleep
changes.

### Lever 2 — Unify onto ONE (physical-core) pool; kill the dual-pool park churn
**Fix.** Route norm/velem/`wukong_parallel_for` through the same private physical-core pool the
GEMMs use (`gemm.rs:105`) via `pool.install`, instead of the global logical-core pool. Eliminates
the ~75–84 private↔global park/unpark waves per forward and stops the two pools from putting each
other to sleep. Cheap, standalone subset of Lever 1.
**Payoff.** **~+10–20%** (removes cross-pool cold-wake churn; keeps one worker set warm across
phases). Also drops HT oversubscription on the norm/velem/attention phases (16 physical vs 22
logical — the same reason the GEMM pool sheds HT).
**Risk.** LOW — mechanical (`install` the existing closures on `gemm_pool()`), no kernel changes.
**Bit-exact.** Safe — pool identity does not affect the row/chunk mapping.

### Lever 3 — Finer / P-core-weighted GEMM grid (raise the ceiling, esp. skinny-M @S=128)
**Fix.** In `select_2d_block_shape`, raise the block count in the skinny-M regime (S=128 gives only
~24 blocks) and, generally, over-decompose so P-cores steal proportionally more than E-cores
(work-stealing already runs over `nbi·nbj` tasks — just supply more, smaller tasks so a fast P-core
grabs several while an E-core finishes one). NB: static affinity *pinning* was measured 25–35%
slower (`gemm.rs:116–124`) — the lever is uneven **work via finer stealable grid**, not pinning.
**Payoff.** The GEMM ceiling governs ~90% of FLOPs. Much of the 3.9× is hardware physics (E-cores +
all-core clock), but the load-imbalance slack — especially the coarse 24-block skinny-M grid at
S=128 — is addressable ⇒ **~+10–15%, concentrated at S=128.**
**Risk.** MEDIUM — a tuning change with adjacent-run A/B needed (the runtime's discipline); risk of
over-packing (redundant per-block packs) if the grid gets too fine.
**Bit-exact.** Safe — each C tile has exactly one owner and a fixed ascending-K order regardless of
tiling (`gemm.rs:1498–1506`, pinned by `sgemm_2d_blocks_matches_serial`).

### Lever 4 — Attention: tile over (head × query-row-block), not just 12 heads
**Fix.** Extend the outliner so the head region is `H × row-blocks` (e.g. 12 heads × 4 query-blocks =
48 tasks) instead of 12. Removes the hard 12-of-16-core cap and the O(S²) E-core straggler (a whole
head currently lands on one core; at S=512 that is a 33.6M-MAC serial chunk that gates the join when
it lands on an E-core).
**Payoff.** Attention is 10% of FLOPs at S=512 and scales only ~4× ⇒ lifting it to ~7× is
**~+5% at S=512** (and removes the S=512-specific scaling drag + variance). Negligible at S=128
(attention is 2.7%).
**Risk.** MEDIUM — the outliner must tile the query dimension and keep per-(head,row-block) scratch
private; softmax/PV are already per-row independent.
**Bit-exact.** Safe — heads and query rows are independent, disjoint output slices
(`attn[i*D + hh*HD + j]`); already the argument the S/@parallel gate rests on (`model.rs:1300`).

### Lever 5 — Fuse residual into GEMM epilogue; parallelize the final LayerNorm
**Fix.** (a) Add a residual-base pointer to the GEMM epilogue (`gemm.rs:290 Epilogue`) so WO- and
W2-projections write `x + C` in one C-writeback — deletes 2 full `[S,D]` velem passes/layer (and, at
S=512, 2 fork-joins/layer). (b) Compile the final LayerNorm with `@parallel` (it's 512 rows at
S=512, currently the only genuinely-serial top-level op). (The `nrm=x`/`nrm=a` copies are already
eliminated by copy-propagation into the norm — nothing to do there; just don't regress it.)
**Payoff.** Memory-bound, small: **~+1–3%** (larger at S=512 where velem is real bandwidth + 2
fork-joins; the final-LN is the forward's serial floor).
**Risk.** LOW–MEDIUM — the epilogue machinery exists; residual add is a single rounding
(`x + C`), identical to velem's `1·x + 1·y`.
**Bit-exact.** Safe — one IEEE add, no reassociation; matches the velem residual bit-for-bit.

---

## 6. The clock-upside mystery — concrete, testable explanations

Symptom: the parallel path does **not** ride power-state upside while MKL/torch do. Candidate
mechanisms, most-likely first, each with a cheap discriminating experiment:

1. **Park-between-regions defeats the clock governor (most likely).** The forward forks/joins
   ~110–132 regions across two pools and the workers **park** between them (§3). The hardware/OS
   clock governor ramps to turbo only under *sustained* load over ~ms; regions here are sub-ms and
   separated by parks, so the cores oscillate wake→ramp-start→park and **never reach a high P-state**
   — a *lower* effective clock than even the steady all-core clock. MKL/torch keep a persistent
   spinning pool that stays hot, so their cores hold turbo.
   *Test:* set `WUKONG_GEMM_2D=0` (persistent broadcast region, `gemm.rs:1213`) and/or pin
   `RAYON_NUM_THREADS` with a busy-wait build, and read per-core MHz (Windows `HWiNFO`/`perfmon`
   effective-frequency counter) during the forward vs during an equal-length MKL run. If Wukong's
   cores sit ~2 GHz while MKL's sit ~3+ GHz, this is it. Also A/B: insert a ~50 ms warm spin
   immediately before the timed region — if the first ~ms speeds up, the governor was cold.

2. **All-core power-limit derate (partly physics).** Even hot, 16 active cores share the package
   power budget and clock below single-core turbo. This is real but is *already* in the measured
   3.9× ceiling; it explains why even a perfect team can't hit 16×, not why we fall short of 3.9×.
   *Test:* raise PL1/PL2 (or plug in AC vs battery) and re-measure — if scaling barely moves, the
   limiter is #1/#4, not the power cap.

3. **E-core critical path pins the join at low clock.** The join always waits for the slowest worker;
   E-cores clock lower *and* compute slower, so the tail of every region runs at E-core frequency.
   Because it is a wait, higher P-core clocks don't shorten it.
   *Test:* run with the global/GEMM pool restricted to the 6 P-cores only (affinity mask) and
   compare — if scaling *improves* despite fewer cores, the E-core tail was the limiter (consistent
   with the finer-grid Lever 3).

4. **Memory-bandwidth-bound phases are clock-invariant.** Norms, velem, softmax, copies, and the
   memory-heavy tail of skinny GEMMs are bandwidth-bound; DRAM/L3 bandwidth does not rise with core
   clock, so those phases show no clock upside by construction.
   *Test:* compare the forward's scaling against a pure-GEMM microbench of the same total FLOPs — if
   the pure GEMM rides clock and the forward doesn't, the non-GEMM bandwidth phases are the
   difference (points at Lever 5 + Lever 2).

5. **Fork-join wake is IPI/scheduler-latency bound (clock-invariant).** The ~5–15 µs/entry wake cost
   is dominated by inter-core signaling + scheduler dispatch, which is ~fixed in wall-clock
   regardless of frequency, so it too shows no clock scaling.
   *Test:* count regions (already known: 108/132) × measured per-entry wake (instrument
   `wukong_parallel_for` entry/exit with `rdtsc`) — if it's <2% of the forward, wake latency is a
   symptom, not the cause, and #1 (clock ramp) is the real lever.

**Distinguishing #1 from #2 is the key call:** #1 is software-addressable (Lever 1/2, keep cores
hot) and likely the bigger, recoverable slice; #2 is closer to hardware physics. The single cheapest
decisive experiment is **experiment #1's effective-frequency read** during Wukong-par vs MKL over an
equal wall-clock window.
