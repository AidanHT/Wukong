# THE WAVE PLAN — Wukong H100 SOTA campaign

**Ranking rule applied:** the gap decomposition's per-shape mechanisms are the ordering authority. Levers are ranked by (measured mechanism they fix) × (shapes it binds) × (cost). Literature levers with no measured mechanism behind them (ping-pong, Stream-K, 2x2x1, 192x256, prefill-flash-vs-FA3) are held out of the plan with explicit trigger conditions.

**THE ONE THING THAT MUST REACH C1 BEFORE THE AXIS FREEZES.** BW_L2 is now MEASURED, not derived: gpt_d1024_down runs 597.6 TFLOP/s through I_cta = 85.33, which is exactly 7.00 TB/s of L2 read — the bottom edge of D1's [6.97, 9.43] band. Re-run every D1 prediction at 7.0, not 8.2. Now solve for the peer: with I_cta = bm·bn/(bm/cm + bn/cn), a **2x1x1 A-multicast gives I_cta 102.4, so cuBLAS's measured 838.7 TFLOP/s at sq4096 would require 8.19 TB/s — above the measured 7.00. It is arithmetically impossible for the in-flight 2x1x1 arm to reach the peer.** A **1x2x1 multicast of B — the 256-wide operand — gives I_cta 128.0**, under which the peer's entire column (838.7 / 864.3 / 816.4) needs only 6.55 / 6.75 / 6.38 TB/s, all comfortably under the measured 7.00. The peer's numbers are fully explained by a 2-CTA multicast on the wide operand at the L2 bandwidth we measured. **Multicast the wider operand. Make 1x2x1 the primary arm and 2x1x1 the control, not the reverse.** T_L2 goes 597 → 896 TFLOP/s (+50%), which lifts every per-shape L2 roof by 1.5x (sq4096 71.2% → 106.8%, sq8192 69.1% → 103.7%, gpt_d4096_up 73.2% → 109.8%) and takes the L2 fill out of the binding position everywhere. C1 lifts the roof; waves 2 and 3 collect the floor. That is the campaign's spine.

---

## DOSSIER AMENDMENTS (2026-08-11) — read these before executing any wave below

C1 LANDED (round 3: 1x2x1 B-multicast measured 73.5% / 82.2% of cuBLAS at sq4096/sq8192; the
spine held). Wave 2B's peer bar is COMPLETE and staged (CUTLASS v4.6.1 sm90a profiler
f16+fp8+int8 368/782/308 kernels; FA2 2.8.3.post1 with the kvcache API; vLLM 0.26.0; peers.json
is the authority). Wave 2A is in flight. Waves 3/4/5 now each have an implementation-grade
dossier in this directory, derived from the measured rounds — **where a dossier and a wave
section below disagree, the dossier wins.** The corrections that change execution:

- **WAVE 3** (`WAVE3_DOSSIER.md`): the lever ranking is now (1) persistence, (2) raster, (3)
  per-shape dispatch — and the mainloop drain is budgeted at ZERO (Fit C prices its cost at one
  ring-stage fill, 9.2%, against at most 8.9% of gain). The raster optimum is `GROUP_M = 16`,
  a TALL group applied to the CLUSTER index (`sqrt(W·BN/BM)`), not the wide group implied
  above. Wave 3's own section overstates: `128x128 @ 2 CTAs/SM` is REFUTED by the ptxas census
  (occupancy binds on the static 168 regs/thread); static register headroom is 2 regs/thread,
  not 8 (the 8 is the `setmaxnreg` view — reconcile both on the census before any epilogue
  work); sq2048 gets exactly zero from this wave; sq1024's lever is a 128x64 tile (+56 pts).
  192x256x64 is DEAD, not held out (CTA-M 192 is not a multiple of the 128-row wgmma M and 512
  threads cap ptxas at 128 registers). Combined dossier projection: suite mean 61.5% → ~80.8%.
- **WAVE 4** (`WAVE4_DOSSIER.md`): the break-even table above is priced at BW = 3.0 TB/s, and
  **HBM bandwidth has never been measured on this campaign's H100** (`hbm_bandwidth` is
  `#[ignore]`d in both device-suite logs) — every fusion break-even is provisional until it
  runs. The "three shapes already at 0.97x" reads as a WIN above; it is a LOSS (no shape clears
  1.0x pre-W3). `gemm_nt_wgmma` has no product call site — the wave must include the
  recognizer→offload wiring or it ships a benchmark, not a product surface. The
  `cublaslt_epilogue_support_matrix` probe (milliseconds, no launch) precedes any f16-out
  engineering. Highest-margin target: `silu(x·Wᵀ+b)` at gpt_d1024_up — derived 1.451x over the
  fused cuBLASLt chain at today's r = 0.82, for five PTX instructions and one register.
- **WAVE 5** (`WAVE5_DOSSIER.md`): the CUTLASS fp8/int8 profiler rebuild listed under Wave 2B
  is DONE — do not re-run it; only the measurements remain. The 16-bit descriptor transfers to
  1-byte operands field-for-field IFF `bk·dtype.size() == 128` (BK 64 → 128); 8-bit wgmma is
  K-major ONLY; the fp8 exact-integer bring-up arm exists only at K ≤ 256 (14-bit product
  accumulation), int8 keeps `==` at every K on the s32 output; two-level fp8 accumulation does
  not fit the 128x256 tile; wgmma has no int4 shape, so Wave 5 cannot produce an int4 headline.

**NEXT H100 VISIT MANIFEST (one container, in this order):** (1) bring-up E/F/G on Wave 2A's
new guard shape; (2) Wave 2A's arms — v2-store / evict-hint A/B/C, the K-sweep, the
epilogue-elided arm; (3) `hbm_bandwidth` — the campaign's FIRST H100 HBM measurement, it prices
every Wave-4 break-even; (4) `cublaslt_epilogue_support_matrix` (ms); (5) the seven-shape
re-measure under the shipped ≥4096-class default (incl. gpt_d1024_up under `w1_s4_mcb2` — a
Wave-4 prerequisite); (6) the ~$0.06 mma.sync fp8/int8/int4 baseline (Wave 5's Hopper
denominator). **Setup demand (Wave 3's): lock the SM clock if the container permits it,
otherwise record `clocks.sm` per arm — the round-3 provenance reads UNKNOWN and that unknown is
the width of the cost model's uncertainty band.**

### WAVE-2 VISIT RESULTS (2026-08-11, ~$1.0 total) — the manifest above RAN; these supersede its predictions

Source: `bench/gpu/h100/2026-08-11-h100-w2-r1..r12*.log` (12 rounds, one container).

**New measured baseline, shipped rule, % of cuBLAS.** f16 (f32 out): sq2048 **95.3** / sq4096
**88.6** / sq8192 **92.3** / gpt_d1024_up **80.3** / gpt_d1024_down **97.3** / gpt_d4096_up
**73.4**. sq1024 is REFUSED — the peer's own dispersion floor is +/-15.47%, wider than any effect
we could claim. bf16 under the transferred rule: 49.5 / 94.0 / 89.0 / 91.9 / 80.0 / 95.4 / 73.2 —
agrees with f16 within ~2 pts everywhere the shapes overlap, so the rule transfers across dtype.
**Suite mean 61.5% → ~87.9%.**

**Mechanism ledger (what moved it, and what did not):**
- **v2 stores: +25.2 / +15.3 / +10.1 pts. SHIPPED, both dtypes** (`fd41ebf`, `a1e3cd8`). This is
  the whole of the jump; the epilogue's issue cost was the binding term, as predicted.
- **Evict hints: a publishable null.** Measured, no effect, recorded as such — do not re-run.
- **nostore diagnostic: 114.0 / 101.1 / 101.8%.** The mainloop is at or ABOVE the peer already.
  Everything still owed is epilogue + wave overhead, not the inner loop.
- **K-sweep: the scalar epilogue is 17.7 us at M=N=2048** (intercepts 22.33 vs 4.68), which prices
  the v2 win directly instead of inferring it from two shapes.
- **`hbm_bandwidth`, the campaign's FIRST H100 run: copy 2922 GB/s = 87.2% of the 3352 GB/s spec
  peak.** W4's provisional 3.0 TB/s denominator was within 3% — the break-even table stands, now
  MEASURED rather than assumed.
- **int8: the Ada tuning does NOT transfer** — 22-52% of IMMA on Hopper. int8 needs its own tile
  search, which is a wave, not an arm.
- **int4: no bindable library peer exists.** Documented as a lead, not a headline (consistent with
  wgmma having no int4 shape).
- **Fused dequant BEATS the cuBLAS chain 1.08-1.15x** — the fusion thesis survives contact.
- **cuBLASLt fuses RELU/GELU/BIAS at BOTH f32 and f16 out** — W4's target #2 premise is REFUTED.
  **SiLU is absent from the enum**, so W4's top target (`silu(x·Wᵀ+b)` at gpt_d1024_up) stands.
- **Instrument:** clock lock was REFUSED in the container (recorded per Wave 3's demand); one bf16
  round SELF-REFUSED on +6.82% SM-clock drift (`r11a`). The refusal machinery works.

**Consequences per wave.**
- **WAVE 3** now starts from **87.9%, not 61.5%**. Its dossier's projections were derived pre-v2 —
  **re-derive every effect size against the new floor before executing.** The lever RANKING is
  unchanged: persistence > raster > per-shape dispatch, mainloop drain still zero.
- **WAVE 4**'s break-evens are now priced by a **measured 2.92 TB/s**, and its epilogue target list
  shrinks to SiLU (+ the fused-dequant win, already 1.08-1.15x over the chain).
- **WAVE 5** splits: **int8 on Hopper is a wave of its own**, not an arm of the 8-bit wave.

---

## WAVE C1 — Cluster axis + config sweep (LANDED 2026-08-10, r3 measured — see amendment) — MUST

**Owner:** one agent, `ptx_wgmma.rs` + `gpu.rs`. No second agent in these files this wave.

**Levers:** TMA multicast cluster (axis per above), stages sweep, wait-depth (`wgmma.wait_group N>0` with a one-stage-lagged empty-barrier arrival), W3C 128x128 arm.

**Predicted effect (restated from the arithmetic):** 1x2x1 → I_cta 85.33 → 128.0, T_L2 597 → 896 TFLOP/s. 2x1x1 → 102.4 / 717 and cannot reach the peer. Wait-depth: bounded at ~2.5% — the marginal k-tile at one wave already costs 1050 tensor-clocks against a 1024 ideal (97.5%), at or above the published 87.9% wgmma issue ceiling. W3C: at sq2048 it has identical wave efficiency to W1 and 1.33x the L2 traffic (ceiling 69.6% vs W1's already-measured 68.2%) — **it cannot win at 2048 and must not be sold as a 2048 lever**; at sq1024 it is the right call and then some (128x256 is capped at 78% of cuBLAS by 32 CTAs; 128x128 tops out at 145%).

**Guards:** G11 (cluster grid divisibility rejected in `validate` at every `WGMMA_BENCH_GRID` point and the bring-up ragged set; `!ptx.contains("multicast")` becomes conditional on `cfg.multicast`, never deleted; a padded grid.x with a live cluster barrier is the deadlock). G3 (`WgmmaCfg::validate` must reject any config whose `name`/`key` is not derivable from its own geometry — this is the sweep's #1 hazard: a `{stages: 5, ..WGMMA_W1}` row that forgets `key` gets the cached 4-stage module back and bills 1000 launches to a config that never ran). G4 (PTX fingerprint asserted on a `Gpu::function` key hit). G18 (restate the wait-depth count law as an ordering law: the empty-barrier arrive must textually follow a `wait_group` whose depth ≥ committed-but-unwaited groups). G12/G17.

**Metered round:** $0.02 CPU census (ptxas over every emitted variant, spill 0 / no C7511 / sm_90a / ASCII) + one H100 visit running bring-up E/F/G **then** the sweep in the same container, one log. **Refusal:** any sweep row whose achieved SMEM/regs do not match its declared config; any cluster arm without the divisibility law.

---

## WAVE 2 — The instrument, the guard, and the two free levers — MUST

**Owner A:** `ptx_wgmma.rs` + `gpu.rs` + `bench_instrument.rs`. **Owner B (no overlap):** `modal_app.py`, `baselines.rs`, `tools/*.py`.

**Levers (A):** epilogue rung 1 — fuse the accumulator pair at +0/+4 into `st.global.v2.f32` (8192 half-empty sectors/CTA → 4096 full); `.L2::evict_first` / `createpolicy` on the C stores and evict-last on the TMA operand loads; the K-sweep arm at fixed M=N=2048 plus an epilogue-elided arm to split F into prologue vs epilogue on one kernel instead of two shapes.

**Levers (B), CPU-only, ~$1.01/hr:** unlock the CUTLASS profiler for fp8/int8 (`cutlass3x_sm90_tensorop_gemm_e4m3_*` and `_s8_s8_s32_*` — the current filter is f16-only); build the cuBLASLt fused-epilogue peer (`CUBLASLT_MATMUL_DESC_EPILOGUE` + `EPILOGUE_BIAS_POINTER` on the raw-sys plan already in `baselines.rs:2935-2999`); torch.compile max-autotune with caches on the Volume; FA2 `flash_attn_with_kvcache` and an FA3 rebuild without the `DISABLE_{PAGEDKV,SPLIT,PACKGQA,VARLEN,FP8}` flags.

**Predicted effect:** v2 store ≈ half the epilogue's issue cost (4.14 → 2.07 us/CTA pessimistic): +3-8% at sq1024/sq2048/gpt_d1024_up, +1-3% elsewhere. Evict-first C targets the gpt_d1024_up/down controlled pair — identical 402.7 MB of L2 request and identical FLOP, 3.80 vs 7.00 TB/s achieved; recovering half the 48.55 us delta is 106.05 → ~85 us, 54.8% → ~65-68%. Zero expected at gpt_d1024_down (at its roof) and sq8192 (C dwarfs L2). Both are advisory-hint levers: **a null result is a publishable result.** The B track buys no speed and deletes claims: bias/relu/gelu becomes a GEMM-parity fight the moment cuBLASLt's 16 fusable epilogues are on the bar.

**Guards — this wave BUILDS the campaign's correctness floor:** G1 (the guard shape is currently ONE CTA, one ring pass, zero ragged edges — replace with grid ≥ 3x3, ktiles = stages+1, one ragged M/N/K, **and tiles > CTAs so G19 is reachable**; 384x768x320 host reference is ~0.3 s). G2 (a second pre-timing arm on pseudorandom f16 against an independent f64 reference at `c·√K·ε` — the exact-integer oracle is invariant under *any* reassociation and is therefore structurally blind to every scheduler change coming in waves 3-5). G8 (two-run bit-identity, on the G2 arm — with exact integers a nondeterministic reduction is still bit-identical). G7 (memset C inside the timed region whenever the schedule requires a zeroed C — 4.4% at 8192², 8% at 4096², delivered free today because the timed loop never re-zeros and never reads back; plus one closed-form readback: every lane is exactly `K·f16(0.01)²`). G16 (contender dispersion as a `FieldVerdict` and a `Refusal` variant — arms A and C are both cuBLAS, so today's 0.03-2.19% floors are the *denominator's* spread). G9 (restate the epilogue law as a transport law over store-class source operands, required by the v2 change). G12/G13/G14/G15/G17 (invert `MUST_TIME_BOX` into a scan; feed the census `wgmma_device_free_modules()`; ban `tcgen05`/`clusterlaunchcontrol` from an sm_90a module; make `ramp_radix` dtype-aware).

**Metered round:** $0.02 CPU census + one H100 visit: bring-up E/F/G on the NEW guard shape, then the v2/evict A/B/C and the K-sweep, one log. **Refusal:** publish nothing structural from any later wave until G1+G2+G8+G7+G16 are green.

---

## WAVE 3 — The schedule: raster, persistence, per-shape tile dispatch — MUST

**Owner:** one agent, `ptx_wgmma.rs` + `gpu.rs`.

**Levers:** grouped threadblock raster over the CTA index (~8 integer ops in the prologue, applied to the **cluster index** with the intra-cluster rank re-added, not to raw `%ctaid`); persistent CTA with a static tile-scheduler loop (`for tile = ctaid; tile < ntiles; tile += gridDim`, per-tile stage/parity/accumulator reset); occupancy-aware tile dispatch (128x128 @ s3 for two CTAs/SM; a per-shape dispatcher, not a global swap).

**Predicted effect — this is the largest measured-mechanism wave.** Raster: gpt_d4096_up DRAM 2.384 → 0.797 GB (2.99x). This shape is a **hard arithmetic blocker** today — matching the peer's 0.6734 ms with the linear order needs 3.54 TB/s against a 3.35 TB/s HBM peak. Post-raster 42.8% → ~70%. sq8192 2.485 → 1.326 GB (1.87x), 58.8% → ~65-75%. sq4096/sq2048/gpt_d1024_*: raster ratio 1.08-1.30 at 25-30% DRAM utilisation — **expect no measurable change and treat any apparent one as noise.** Persistence removes waves × (stage-0 fill + exposed epilogue) = 2.0-5.6 us per wave, plus wave quantization (3.879 → 4 and 15.515 → 16 both cost 3.0%): gpt_d1024_up 54.8% → 59-69%, sq4096 67.5% → ~76%, sq1024 +9-26%. Tile dispatch is the only lever that moves sq1024 at all: 32 CTAs on 132 SMs caps it at 77.7% of cuBLAS with 128x256; 128x128 raises the ceiling to 155%. Precedent: the Act-1 raster measured 56% → 72% of cuBLAS on the 4050.

**Guards:** G5 (raster bijection — a pure-Rust twin asserted bijective over `[0, gx·gy)` for every `gx % group` residue, plus a device gate at ≥ 8x8 CTAs; the classic defect makes f surjective-not-injective, two CTAs write one tile in-bounds and one tile returns as host-pre-zeroed silence). G6 (state the grid law over whatever function the launcher actually passes to `dyn_launch_cfg`, parameterized over `sm_count ∈ {114, 132}` — a stride bug that cancels on SXM does not on PCIe). G19 (`scale-d = %pfirst` is `kt != 0` and has no meaning in a tile loop — assert `mov.u32 %kt,0` count equals tile-loop entries; tile 2 accumulating into tile 1 is an exact integer and invisible to the old oracle). G20 (pin `TmaOobFill::Zero` as a device-free law). G1's corrected shape, G2's random arm, G16 (a work-stealing kernel has real spread).

**Metered round:** $0.02 CPU census + one H100 visit: bring-up E/F/G, raster-off/raster-on/persistent A/B/C at all seven shapes, one log. **Refusal:** any shape whose apparent gain is inside its own dispersion; any raster arm without the bijection twin.

---

## WAVE 4 — The epilogue becomes a product surface — MUST

**Owner:** one agent, `ptx_wgmma.rs` + `gpu.rs`. Peer functions already landed in W2's `baselines.rs`; this wave only wires them.

**Levers:** epilogue rung 3 — SMEM-staged `st.global.v4.f32` / `cp.async.bulk.tensor.2d.global.shared::cta` TMA store; then the Act-1 register epilogue ported to wgmma (bias, relu/silu/gelu, `beta·C` residual, `cvt.rn.f16x2` low-precision store); then RoPE in the QKV-projection epilogue. Emit each variant separately.

**Predicted effect:** epilogue 4.14 → 0.52 us/CTA (8192 → 1024 sector wavefronts). Fusion is where the heavy exceeds live, and the break-even is far below parity: a fused kernel beats the cuBLAS-GEMM-plus-epilogue chain at 56.5% of cuBLAS (gpt_d1024_up), 70.5% (sq2048), 78.6% (sq4096), 79.0% (gpt_d4096_up), 87.7% (sq8192) — **three shapes are already at 0.97x of the chain at today's un-improved GEMM speed.** With C1+W3 in, fused rows publish at 1.14-1.50x. Low-precision output is separately countable: gpt_d4096_up's C write halves (268.4 → 134.2 MB) and deletes our own f32→f16 cast pass — together 0.179 ms = 27% of the peer's GEMM. RoPE: 8·(H+H_kv)·S·D = 83.9 MB = 28 us at Llama-8B S=2048, against 92.8 us of FA3-speed attention (30%), rising to ~50% at S=512.

**Guards:** G9 (the transport law now load-bearing — a TMA-store epilogue emits zero `st.global.f32` and would otherwise delete the law along with both real properties; plus device gates at N%8≠0, e.g. N=bn+1 and bn−3). **G10 is the sharp one:** W1's SMEM map leaves 35,776 free bytes against a 131,072-byte C tile, so a TMA-store epilogue *physically must alias the mainloop ring* — correct in a one-tile kernel, a live race the instant W3's persistence lands, because the producer refills stage 0 for tile t+1 while a consumer stages C out of it. Give the epilogue its own region, add `the_smem_map_is_disjoint`, and teach `smem_bytes()` about it or `dyn_smem_bytes` silently under-requests. Register budget: 128·32 + 256·232 = 63,488 of 65,536 leaves exactly 8 regs/consumer-thread and `setmaxnreg` moves in steps of 8 — bias+act+residual together do not fit. G14 census on every new entry (C7511 is a *silent 2-4x*, not a failure).

**Metered round:** $0.02 CPU census + one H100 visit: bring-up E/F/G, then fused-vs-cuBLASLt-fused-epilogue and fused-vs-chain at real dims (Llama-3-8B, GPT-2, Qwen — never d=64/dff=256), one log. **Refusal:** any bias/relu/gelu row published as a fusion win now that cuBLASLt fuses those 16 epilogues at zero extra traffic.

---

## WAVE 5 — 8-bit wgmma: e4m3 and s8/u8 — SHOULD (largest absolute headroom on the device)

**Owner:** one agent, `ptx_wgmma.rs` + `gpu.rs`.

**Levers:** `WgmmaDtype::{E4M3, E5M2, S8, U8}`; `size()` → 1; dtype-dependent K (32) in `wgmma_per_stage()`; BK 64 → 128 so the stage geometry stays byte-identical at 49,152 B; drop both transpose immediates (TN/K-major only), fp8 tail `p, scaleA, scaleB`, int8 tail `p` alone and **no `.satfinite`**; the 1-byte core-matrix descriptor re-derivation routed through `desc_sweep_candidates()` in the same visit; DeepSeek two-level accumulation (promote to CUDA-core f32 every K=128); 1x128 activation / 128x128 weight block scaling for comparability.

**Predicted effect:** Hopper has no native fp8 `mma`, so `ptx_fp8.rs` runs the whole family at fp16 issue rate. Instruction-issue ratio wgmma-fp8 / mma-f16 = 1417.2 / 490.7 = **2.89x**; int8 wgmma / mma = 1448.7 / 977.9 = **1.48x and bit-exact-preserving** (s32 accumulation, `==` gate survives verbatim). Retyping W1 at its round-1 peak-fraction (57.2%) projects ~1132 TFLOP/s = 97% of the published H100 library fp8 bar and ~2.0x our best measured absolute — and that projection rises with whatever C1+W3+W4 add to the peak-fraction. Block-scale traffic is +3.1% of operand bytes: publish the honest, scaled number.

**Guards:** G15 (dtype-aware `ramp_radix` — bf16's exact-integer limit is 256, eight times tighter than f16's 2048, and e4m3's is far tighter still; enforce `1+(w−1)(1+w+w²) ≤ limit` inside the ladder). **G2 is a hard prerequisite, not a nicety:** Hopper's fp8 tensor core right-shift-aligns and keeps only the top 14 mantissa bits, so the f64-reference tolerance *will* loosen — derive it, never widen it to fit. Capability: `require_fp8` (cc ≥ 8.9) is necessary and NOT sufficient — every 8-bit wgmma launcher must also call `require_sm90a` or the textual fp8 law passes while the module refuses to load. G13/G14 on the new families.

**Metered round:** $0.02 CPU census + one H100 visit that runs the 8-bit descriptor sweep and the correctness arms **first**, then the peer round against cuBLASLt fp8, cuBLAS IMMA, vLLM `cutlass_scaled_mm` and the W2-unlocked CUTLASS profiler — both peer columns published, because a library fp8 bar at 58.7% of peak versus a cuBLAS f16 bar at 87.3% means the choice of peer, not the kernel, decides whether the headline reads 97% or 73%. Precede it with the $0.06 baseline round measuring the *existing* mma.sync fp8/int8/int4 families on H100 so W5's delta has a Hopper denominator, not a 4050 memory. **Refusal:** an int4 headline without the Machete column; any fp8 headline against a per-tensor-scaled peer.

---

## WAVE 6 — Decode and serving: the memory-bound regime — SHOULD (megakernel sub-item EXPLORATORY)

**Owner:** one agent, `paged_attention.rs` + `serving.rs` + `megakernel.rs` + `ptx_norm.rs` + the bench wiring in `gpu.rs`. `ptx_wgmma.rs` untouched this wave.

**Levers:** paged decode — replace scalar `ld.global.u16` with `ld.global.v4.b32` keeping the lane partition a function of the logical position index (assert head_dim % 8), and give a warp `(slot, kv_head)` with a *tiled* q-group instead of `(slot, q_head)`; int8/fp8 KV; norm/softmax family to 128-bit loads with the per-lane fold order preserved, plus the `norm_vs_peers` bench that does not exist; EXPLORATORY: the persistent cooperative decode megakernel over the existing `grid_barrier_ptx` / `plan_grid` / `launch_mega`.

**Predicted effect:** today one warp memory instruction is a 32-way gather across 32 sectors using 2 of every 32 bytes — the exact shape FlashInfer reports at "40%+ bandwidth utilization"; v4 plus QuACK's coalescing prerequisite puts the band at 75-85%, i.e. 1.6-2.0x on the decode-attention term. GQA fixes a hard g-fold read amplification (4x at Llama-3-8B, 8x at 70B) on the single dominant traffic term: batch 16 / L=2048 / 32 layers requires 4.29 GB/step = 1.28 ms at 3.35 TB/s; today we ask for 4x that. Norms: QuACK measures 3.01 TB/s vs torch.compile 1.89 — vectorization is worth ~1.3-1.7x and moves us from below torch.compile to near QuACK; this is credibility, not a headline, and every fused-norm claim divides by it. Megakernel ceiling: 78% of BW vs vLLM/SGLang's ≤50%, ~1.56x, with ~1.3 us of dead time per kernel even under graphs (320 launches/token-step = ~0.42 ms).

**Guards:** `paged_attention_invariant_to_block_layout` (a v4 that straddles a position boundary makes the lane partition layout-dependent and silently breaks it), the f64 decode reference, the accumulator-tiling design step (g=4 at d=128 is 512 registers — impossible without tiling), norm determinism + CPU-oracle gates with `MIN_ELEMS_PER_LANE` and `warps_per_row` **re-derived on H100, never ported from the 20-SM 4050**, G16, G17. A cooperative grid deadlocks the device if not fully resident — `plan_grid` is only as good as the occupancy query.

**Metered round:** $0.02 CPU census + one H100 visit: decode against FA2 `flash_attn_with_kvcache(block_table=)` and vLLM's `benchmark_paged_attention` (labelled a **floor**, since it drives the deprecated v1/v2 op), norms against QuACK and torch.compile, megakernel against our own `step_graphed`. **Refusal:** any serving headline against `paged_attention_v2` alone.

---

## HELD OUT OF THE PLAN (with trigger conditions)

- **2x2x1 cluster** (I_cta 170.7 / T_L2 1195, removes the L2 roof entirely) — held until 1x2x1 is measured. Two published data points argue against it: the H100 worklog found 2x2 *slower* than a 2-tile cluster, and an NVIDIA-staffed thread measured multicast at ~2 TB/s vs ~8 for independent loads. Trigger: 1x2x1 lands positive AND the round shows us still fill-bound.
- **192x256x64 tile** (I_cta 109.71, T_L2 768; +24.6% at sq8192, −12% at gpt_d1024_down) — large, sits exactly on the register cliff CUTLASS itself falls off (32 C7511 warnings at 128x256x64), needs W4's SMEM-staged epilogue to free ~10 registers first, and needs a dispatcher. Trigger: the cluster lands only as A-multicast; unnecessary if 1x2x1 or 2x2x1 lands.
- **Stream-K** — wave quantization is only ~3% once W3's persistence lands; needs a device workspace and a deterministic reduction order or the tolerance gate becomes shape-dependent, and G2+G7 must be green first.
- **Ping-pong** — the arithmetic is decisive before any measurement: 232 regs/thread caps a consumer at one m64n256 tile, so the CTA tile collapses to 64x256 (T_L2 420) — *below* the 566 we already measure at sq4096. Only correct for the skinny-M/decode class, and no shape in the suite is skinny-M. Trigger: a decode row is added.
- **Prefill flash vs FA3** — we are 3-8x behind (D2 predicts 0.12-0.33x) and a 30-50% fusion adder cannot cover that. Do not chase fused-RoPE prefill.
- **Beating Machete on W4A16** — measure only (a `::marlin` round at M=1/16/128). At M≤32 the shape is 100% weight-bandwidth-bound (12.98 MB = 3.87 us caps a perfect M=16 kernel at 208 TFLOP/s; two kernels at 90% of HBM differ by <10%). Beating it at M=128+ needs W5's plumbing plus a weight pre-shuffle. **Expect and publish a loss at M≥128 and a tie at M=1.**

---

## LIST OF TARGETS — where "heavily exceeds SOTA" is credible

| # | Target | Honest peer | Staged-peer status | Credible claim size |
|---|---|---|---|---|
| 1 | **Gated FFN (SwiGLU/GeGLU) fused into the GEMM** | TransformerEngine merged-weight GEMM + separate SwiGLU kernel; torch.compile max-autotune. **cuDNN GEMM+SwiGLU is SM100-only — no library-fused gated FFN exists on H100.** | TE **not staged** (needs a build); torch.compile recipe in D5 2.2, caches must be on the Volume | 1.14x (Llama-8B, K=4096) to ~1.7x (Qwen-0.5B-class). Brackets the published H100 CUTLASS-SM90 measurement of the same fusion: 0.99-2.47x over eager, with torch.compile at only 34-94% of eager. Win shrinks as 1/K — never publish a small-model ratio as a large-model one. |
| 2 | **Low-precision-output + activation epilogue (f16-out fused act/bias)** | cuBLASLt — its epilogue enum has **no** lowp-out-plus-activation and **no** SILU | Raw-sys plan already in `baselines.rs:2935-2999`; the `EPILOGUE` attribute lands in W2 | 0.179 ms on gpt_d4096_up = 27% of the peer's GEMM (halved C write + the deleted f32→f16 cast pass) |
| 3 | **SiLU / swish and residual+activation epilogues** | cuBLASLt's fusable set is exactly 16: RELU/GELU/BIAS/AUX/D*/BGRAD*. No SILU, no residual+act. Peer floor is cuBLAS + a separate kernel, or torch.compile | Same as #2 (free once #2's peer exists) | 8·M·N bytes of chain traffic deleted; break-even is 56.5-87.7% of cuBLAS depending on shape, so these publish as wins *before* GEMM parity |
| 4 | **RoPE fused into the QKV-projection epilogue** | Nothing on H100: FA3 has no in-kernel rotary (vLLM-Ascend patches in a PyTorch-native fallback), cuBLASLt has no RoPE epilogue, cuDNN GEMM+RoPE is SM100-only. Peer = PyTorch-native RoPE + FA2/FA3 | FA3 wheel as currently staged is **unusable** (`DISABLE_{PAGEDKV,SPLIT,PACKGQA,VARLEN,FP8}=TRUE`) — CPU rebuild required, or use FA2 | "RoPE costs the peer 30-50% of its attention step and costs us nothing": 28 us of 92.8 at Llama-8B S=2048, ~50% at S=512. **Frame it that way, never as "we beat FA3".** RoPE-in-GEMM-epilogue and RoPE-in-attention-prologue are mutually exclusive — publish one and say which |
| 5 | **Bit-exact int8 GEMM with fused per-channel dequant, on wgmma** | vLLM `cutlass_scaled_mm` (CUTLASS SM90, fused dequant epilogue) and cuBLAS IMMA. **The repo's "libraries do not offer fused dequant" framing is FALSE on Hopper and must be retired** | vLLM v0.26.0 tree already on the Volume (`benchmark_int8_gemm.py`); CUTLASS s8 profiler unlocked in W2 | 1.48x issue headroom over our current `mma.sync`, and the surviving differentiator is `==` exactness against an i32 oracle — which no library claims |
| 6 | **Paged decode attention: GQA-correct KV reads + int8/fp8 KV** | FA2 `flash_attn_with_kvcache(block_table=)` is the real bar; FlashInfer tensor-core GQA decode is the stretch bar; vLLM `benchmark_paged_attention` drives the **deprecated** v1/v2 op = a floor, label it as such | FA2 via `::build_peers --fa2` (CPU-only, cheap); vLLM scripts already staged; FlashInfer **not staged** | 1.6-2.0x from coalescing (40-50% → 75-85% of HBM) compounded with a 4x (8B) / 8x (70B) read-amplification removal; the fp8-KV prize is measured by vLLM itself at ITL slope 54% of BF16, +14.9% output tok/s |
| 7 | **Persistent decode megakernel** (EXPLORATORY) | vLLM / SGLang on H100 — a full serving-stack build, not a dlopen; internal floor is our own `step_graphed` (already 6.5-6.9x over eager at 12 layers) | **Not staged**, and the most expensive peer on this list | Published ceiling: 78% of BW and ~1 ms/forward (Llama-1B H100) against vLLM 2.5x / SGLang 1.5x slower and both capped at ≤50% BW → ~1.56x on the bandwidth-bound term; Mirage MPK corroborates 1.0-1.7x |

**Explicitly NOT on this list, and say so in the publication:** fp8 GEMM (projects to 97% of the library bar and 73% of DeepGEMM — a large *absolute* gain and a parity-class *claim*); norm/softmax (QuACK at 89.7% of peak is the bar; vectorization reaches "near QuACK", not past it, and its cluster reduction is a capability we do not have above ~65K reduction dims); big square GEMM (cuBLAS is at 84.8-87.3% of peak there, and parity is a C1+W3 outcome — reachable only via the 1x2x1 cluster, not by any non-cluster change).

---

## STANDING RULES FOR EVERY WAVE

1. `wgmma_hopper_bringup` stages E/F/G run in the **same rented container, before** the perf bench, both in one round log — enforce it in `modal_app.py` so "we only paid for the perf round" cannot happen by accident (G17).
2. The exact-integer arm is a permutation *diagnostic*, never the only arm, from W2 onward (G2 + G8 only work together).
3. Every wave's H100 visit is preceded by the $0.02 CPU ptxas census over `wgmma_device_free_modules()` — one enumeration drives the ASCII, `.target`, `.version` and census laws, so a module cannot be inside three of them and outside the fourth (G14).
4. Report the peak-fraction column beside every ratio. On this part the ratio alone misleads in both directions.
5. A null result on an advisory lever (W2's cache hints) is a publishable result. A gain inside the contender's own dispersion is not (G16).