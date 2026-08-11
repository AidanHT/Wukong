# WAVE 3 DOSSIER -- the schedule: raster, persistence, per-shape tile dispatch

**Scope.** Implementation-grade derivation for WAVE 3 of `ACT2_WAVE_PLAN.md` (owner: one agent,
`ptx_wgmma.rs` + `gpu.rs`). Every number below is derived from measurements already in this repo. No
number in this dossier came from a blog, a datasheet headline or a literature ratio; literature is
cited only where it settles the *shape* of a construction (CUTLASS's swizzle discipline, Stream-K's
decomposition), never its size.

**Status of the two inputs this wave inherits.** Wave 2A's vectorized epilogue
(`st.global.v2.f32`) is taken as DONE and is not re-derived. The 1x2x1 B-multicast (`WGMMA_W1_MCB`,
entry `wgmma_nt_f16_128x256x64_s4_mcb2`) is LANDED and measured.

---

## 0. The instrument: measured inputs, and the per-tile cost model everything below uses

### 0.1 Device and code facts (verbatim from the repo's logs and source)

| fact | value | source |
|---|---|---|
| SMs | 132 | `bench/gpu/h100/2026-08-10-h100-act2-r3-bmulticast.log:519` |
| opt-in SMEM / CTA | 232 448 B | same log, line 521 |
| L2 | 52 428 800 B (50.0 MiB) | same log, line 522 |
| sm clock reported | 1980 MHz (before and after, drift +0.00%) | same log, lines 529-531 |
| clock lock | **UNKNOWN (undeclared)** | same log, line 528 |
| `wgmma_nt_f16_128x256x64_s4` | 168 regs whole-CTA, spill_st 0, spill_ld 0, stack 0, 196 672 B dyn SMEM, 384 thr, **CTAs/SM(reg) = 1**, `setmaxnreg` 32p/232c | `bench/gpu/h100/2026-08-10-ptxas-census.log:755` |
| `wgmma_nt_f16_128x128x64_s6` | 168 regs, 0 spill, 196 704 B, 384 thr, CTAs/SM 1, 32p/168c | same census, line 757 |
| `BW_L2` | 7.00 TB/s (measured, not derived) | `ACT2_WAVE_PLAN.md:5` |
| HBM peak | 3.35 TB/s | `ACT2_WAVE_PLAN.md:45` |
| dense f16 tensor peak (the campaign's denominator) | 989 TFLOP/s | reproduces the plan's 57.2% / 84.8% peak-fractions exactly |

**Occupancy is 1 CTA/SM twice over, and both bounds are tight.** SMEM: 196 672 of 232 448 B, so a
second CTA is impossible by 160 896 B. Registers: 384 threads x 168 = 64 512 of 65 536, so a second
CTA is impossible by 63 488. Neither bound can be relaxed by tuning `setmaxnreg`: `setmaxnreg`
redistributes *within* a CTA at run time; occupancy is decided by the **static** per-thread
allocation ptxas chose, which the census reports as 168.

### 0.2 The grid, from the source

`LaunchPlan::grid` (`crates/wukong_codegen_gpu/src/ptx_wgmma.rs:1601`):

```
gx = ceil(N / BN) rounded up to cluster.x     // x indexes N tiles
gy = ceil(M / BM) rounded up to cluster.y     // y indexes M tiles
```

and the *only* place a tile origin is computed is
`crates/wukong_codegen_gpu/src/ptx_wgmma.rs:3436-3437`:

```
    mov.u32 %tmp,%ctaid.x;    mul.lo.s32 %ctan,%tmp,256;
    mov.u32 %tmp,%ctaid.y;    mul.lo.s32 %ctam,%tmp,128;
```

Two instructions. Every raster in this dossier is a replacement for exactly those two lines plus a
host-side grid reshape. Nothing else in the mainloop, the descriptor, the multicast slicing or the
epilogue reads `%ctaid`.

Note the log's shape triple is printed **M x K x N**, not M x N x K; verified against all eight
`wgmma_cluster_multicast_is_exact` rows (r3 log lines 87-97), e.g. `384 x 320 x 256 grid 1x4` is
M=384 (3 M tiles, padded to 4 by the 1x2x1 cluster, 1 pad CTA), N=256 (1 N tile).

### 0.3 Tiles, waves and quantization, per shape

W1 tile is 128x256. `tiles = gx * gy`; a wave is 132 CTAs (1 CTA/SM).

| shape | M,N,K | gy (M tiles) | gx (N tiles) | tiles | tiles/132 | waves | tail wave | wave eff. |
|---|---|---|---|---|---|---|---|---|
| sq1024 | 1024,1024,1024 | 8 | 4 | 32 | 0.242 | 1 | 32/132 | 24.2% |
| sq2048 | 2048,2048,2048 | 16 | 8 | 128 | 0.970 | 1 | 128/132 | 97.0% |
| sq4096 | 4096,4096,4096 | 32 | 16 | 512 | 3.879 | 4 | 116/132 | 96.97% |
| sq8192 | 8192,8192,8192 | 64 | 32 | 2048 | 15.515 | 16 | 68/132 | 96.97% |
| gpt_d1024_up | 4096,4096,1024 | 32 | 16 | 512 | 3.879 | 4 | 116/132 | 96.97% |
| gpt_d1024_down | 4096,1024,4096 | 32 | 4 | 128 | 0.970 | 1 | 128/132 | 97.0% |
| gpt_d4096_up | 4096,16384,4096 | 32 | 64 | 2048 | 15.515 | 16 | 68/132 | 96.97% |

**A closed form worth knowing.** For any power-of-two tile count `T` with `128 <= T <= 4096`,
`ceil(T/132) * 132 = T * 33/32` exactly, so the wave efficiency is exactly **32/33 = 96.97%** and the
quantization loss is exactly **3.03%** -- the plan's "3.879 -> 4 and 15.515 -> 16 both cost 3.0%".
The identity holds because `132 = 4 * 33` and `T/128 - T/132 = T/4224 < 1` iff `T < 4224`. It breaks
at `T = 8192` (98.51%). Every shape in the suite except `sq1024` is inside the identity, so
**wave quantization is a flat 3.03% across the whole suite and is not a per-shape lever.**

**Under a cluster the wave arithmetic is unchanged.** `132 / 2 = 66` clusters resident and
`tiles / 132 = clusters / 66`, so `waves = ceil(clusters/66) = ceil(tiles/132)` for every shape. The
cluster costs no wave. This holds because a 1x2x1 cluster occupies 2 SMs and H100 SXM5's 132 SMs are
66 TPCs of 2 SMs each distributed over 8 GPCs, so no GPC holds an odd SM count and no SM is stranded
by cluster-granular allocation. CUTLASS's own SM90 occupancy heuristic models exactly this
(`max_sm_per_gpc = 18`, `max_cta_occupancy_per_gpc = 18 - (18 % cluster_size)`) -- see
[tile_scheduler_params.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/tile_scheduler_params.h);
with `cluster_size = 2` the modulus is 0 and nothing is lost. **This is a model, not a measurement**;
its falsifier is a cluster arm that loses exactly 1/66 of the device, which the measured mcb2 wins at
sq4096/sq8192 rule out.

### 0.4 THE PER-TILE COST MODEL (the load-bearing derived object)

Every CTA runs exactly one tile, and its cost is

```
    T_tile  =  X  +  n_k * S            n_k = ceil(K / 64) = k-stages per tile
    T_total =  L  +  waves * T_tile     L = per-launch overhead
```

`X` is everything that is not a steady-state k-stage: the ring fill, the `wgmma.wait_group 0` drain
at `CEND`, the 128 predicated `st.global.f32`, and the prologue. Three **independent** fits:

**Fit A -- the B-multicast arm across K (round 3, `L = 2 us`).**
`(222.735-2)/4 = 55.184 = X + 64 S` and `(1546.454-2)/16 = 96.528 = X + 128 S`
-> `S = 0.6460 us`, `X = 13.84 us`.

**Fit B -- the no-cluster arm across K at a FIXED grid (round 1, f16, `L = 2 us`).**
`sq4096` and `gpt_d1024_up` have the identical 512-tile / 4-wave grid and differ only in K (64 vs 16
k-stages), so this fit has no wave term at all:
`(242.873-2)/4 = 60.218 = X + 64 S` and `(106.050-2)/4 = 26.013 = X + 16 S`
-> **`S_nc = 0.7126 us`, `X = 14.61 us`.**

Fit B and Fit A agree on `X` to within 5% from completely disjoint data. Take **`X = 14.6 us`**.

**Fit C -- the ring depth (round 3, s3 vs s4, both mcb2).**
`(230.136-2)/4 = 57.034 = X3 + 64 S3` and `(1637.038-2)/16 = 102.190 = X3 + 128 S3`
-> `S3 = 0.7056 us`, `X3 = 11.88 us`. So

```
    one ring stage of fill = X4 - X3 = 13.84 - 11.88 = 1.96 us
    the 4-stage fill       = 4 * 1.96 = 7.85 us
    X_epi = X - X_fill     = 13.84 - 7.85 = 5.99 us
```

and the shallower ring's per-stage cost is 9.2% higher (0.7056 vs 0.6460) -- exactly what less
latency hiding predicts. **The two halves of `X` fall out of one A/B and each lands on a physical
constant:**

* `X_fill = 1.96 us` per stage moves `132 CTAs x 49 152 B = 6.488 MB` device-wide, i.e.
  **3.31 TB/s = 98.7% of the 3.35 TB/s HBM peak** (2.20 TB/s = 66% if the multicast's unique-byte
  count is used). The ring fill is a **bandwidth event, not a latency event**, and at a wave boundary
  no CTA on the device has anything to overlap it with.
* `X_epi = 5.99 us` writes `132 x 131 072 B = 17.30 MB`, whose pure HBM-write floor is 5.16 us. The
  epilogue is already within **16% of its write-bandwidth floor** at these shapes -- which is the
  quiet finding that Wave 2A's `v2` store has little left to give at sq4096/sq8192 and should be
  claimed at sq1024/sq2048 instead.

**Model accuracy, no-cluster arm, `T_pred = max(T_floor, T_L2, T_DRAM)`** with
`T_floor = waves * (X + n_k * S_nc)`, `T_L2 = L2_read_bytes / 7.00 TB/s`,
`T_DRAM = DRAM_bytes / 3.35 TB/s`:

| shape | T_floor us | T_L2 us | T_DRAM us | pred | measured us | meas/pred |
|---|---|---|---|---|---|---|
| sq1024 | 20.1 | 3.6 | 2.5 | 20.1 | 21.5 | 1.07 |
| sq2048 | 37.4 | 28.8 | 10.0 | 37.4 | 39.09 | 1.045 |
| sq4096 | 240.8 | 230.1 | 68.9 | 240.8 | 243.44 | 1.011 |
| sq8192 | 1693 | 1841 | 742 | 1841 | 2273.5 | **1.235** |
| gpt_d1024_up | 104.0 | 57.5 | 32.2 | 104.0 | 106.05 | 1.020 |
| gpt_d1024_down | 60.2 | 57.5 | 17.6 | 60.2 | 57.50 | 0.955 |
| gpt_d4096_up | 963 | 920 | 712 | 963 | 1572.2 | **1.632** |

`X(sq1024)` is scaled for 32 resident CTAs (`X_fill` is device-bandwidth-limited, so it scales with
resident CTAs: `7.85 * 32/132 = 1.90`, `X = 8.65 us`).

**Five of seven shapes are explained to within 7%. The two that are not -- sq8192 and gpt_d4096_up --
are exactly the two whose linear-order wave footprint exceeds L2.** That is section 1's entire
subject, and the fact that the residual is confined to precisely those two shapes is the strongest
evidence in this dossier that the raster is a real mechanism and not a hope.

`gpt_d1024_down` lands at `T_L2 = 57.5 us` and measures 57.50 us -- 100.0% of the 7.00 TB/s roof.
That shape *is* the `BW_L2` calibration, and the fact that my byte formula reproduces it to three
digits is the check that the formula is the campaign's formula.

### 0.5 Where the mainloop actually stands (input to section 4)

One 128x256x64 k-stage is `128*256*64 = 2.097e6` MAC. At the campaign's 2048 MAC/clk/SM that is
**1024 tensor-clocks**, the plan's own unit. Converting `S`:

| arm | S (us) | clocks @ 1.8288 GHz (the 989 TFLOP/s reference clock) | clocks @ 1980 MHz (reported) | issue efficiency |
|---|---|---|---|---|
| no cluster | 0.7126 | 1303 | 1411 | 78.6% / 72.6% |
| B multicast | 0.6460 | 1181 | 1279 | 86.7% / 80.1% |

**The clock ambiguity is the whole of section 4's uncertainty.** At the reference clock the
B-multicast mainloop is at 86.7%, i.e. already at the plan's published 87.9% wgmma issue ceiling and
the drain is worth ~0. At the reported 1980 MHz it is at 80.1% and the drain is worth up to 9.7% of
the mainloop. The W3 round can settle this for the price of one line: **lock the clock and declare
it**, so `clock lock` stops reading `UNKNOWN`.

### 0.6 Notation used below

`W = 132` (CTAs per wave). `R` = m-tiles spanned by one wave's footprint, `C` = n-tiles, `R*C = W`.
`f(R) = R*BM + C*BN` is the wave's operand *width* in rows; the wave's DRAM footprint is
`f(R) * K * 2` bytes and the per-tile operand traffic is `f(R) * K * 2 / W` bytes.

---

## 1. RASTER / SUPERTILING

### 1.1 The traffic arithmetic, and its optimum

A wave of `W` CTAs covering an `R x C` rectangle of tiles reads `R*BM*K*2` bytes of A and
`C*BN*K*2` bytes of B, and produces `R*C = W` tiles. Minimising `f(R) = R*BM + (W/R)*BN` gives

```
    R* = sqrt(W * BN / BM)        f(R*) = 2 * sqrt(W * BM * BN)
```

For W1 (BM=128, BN=256, W=132): `R* = sqrt(264) = 16.25`, `f(R*) = 4159.5`.

**The optimum is a TALL group, not a wide one, because A rows are half the price of B rows.** The
candidate set in the wave brief (`GROUP_M in {2,4,8}`) is therefore wrong for a 256-wide tile: the
right set is `{8, 16, 32}` and `GROUP_M = 2` is *worse than linear* on every shape in the suite.

| GROUP_M (= R) | f(R) | vs optimum |
|---|---|---|
| 2 | 17 152 | 4.12x |
| 4 | 8 960 | 2.15x |
| 8 | 5 248 | 1.26x |
| **16** | **4 160** | **1.000x** |
| 32 | 5 152 | 1.24x |
| 64 | 8 720 | 2.10x |

The curve is symmetric about `R*` in log space, so `GROUP_M = 16` and `GROUP_M = 32` bracket the
minimum at +0% and +24%. For the square W3C tile (BM=BN=128): `R* = sqrt(132) = 11.49`,
`f(11) = f(12) = 2944` vs `f(R*) = 2941` -- **`GROUP_M = 12`**, and `GROUP_M = 8` costs +6.6%.

### 1.2 Per-shape effect, against the measured `BW_L2 = 7.00 TB/s` and 3.35 TB/s HBM

The linear order gives `C = min(gx, W)` and `R = W/C`. Total DRAM = `tiles * f(R)*K*2/W + M*N*4`.

| shape | gx | linear R | linear f | DRAM linear | DRAM @ G16 | ratio | **wave footprint linear** | **@ G16** |
|---|---|---|---|---|---|---|---|---|
| sq1024 | 4 | one wave | -- | -- | -- | **1.000** | 8.4 MB (0.17x L2) | identity |
| sq2048 | 8 | one wave | -- | -- | -- | **1.000** | 16.8 MB (0.34x L2) | identity |
| sq4096 | 16 | 8.25 | 5152 | 230.8 MB | 199.3 MB | 1.158 | 42.2 MB (0.84x L2) | 34.1 MB (0.68x) |
| sq8192 | 32 | 4.125 | 8720 | **2485.0 MB** | **1325.7 MB** | **1.875** | **142.9 MB (2.86x L2)** | 68.2 MB (1.36x) |
| gpt_d1024_up | 16 | 8.25 | 5152 | 108.0 MB | 100.1 MB | 1.079 | 10.6 MB (0.21x L2) | 8.5 MB |
| gpt_d1024_down | 4 | one wave | -- | -- | -- | **1.000** | 42.0 MB (0.84x L2) | identity |
| gpt_d4096_up | 64 | 2.0625 | 16648 | **2384.4 MB** | **797.1 MB** | **2.991** | **136.4 MB (2.73x L2)** | **34.1 MB (0.68x)** |

The three bolded DRAM figures reproduce `ACT2_WAVE_PLAN.md:45`'s 2.485 / 1.326 / 2.384 / 0.797 GB and
2.99x / 1.87x **to the digit**, from first principles. The model is the plan's model.

### 1.3 THE CRITERION: the raster is an L2-RESIDENCY lever, not a bandwidth-percentage lever

Three of the seven shapes are a single wave (`tiles <= 132`): sq1024, sq2048, gpt_d1024_down. **For
those the raster is the identity map** -- every tile is resident simultaneously, so no permutation of
the launch order can change a single byte of traffic. Do not bench them as raster rows; they are
controls that must come back inside their own dispersion.

Of the four multi-wave shapes, `sq4096` and `gpt_d1024_up` have a linear-order wave footprint that
already fits L2 (42.2 MB and 10.6 MB of 50.0 MiB). The plan's "raster ratio 1.08-1.30 at 25-30% DRAM
utilisation -- expect no measurable change and treat any apparent one as noise" is exactly right, and
section 0.4's table shows why: at those two shapes `meas/pred` is 1.011 and 1.020, i.e. the model has
**no room** for a raster gain.

That leaves two shapes, and the reason they are the two is visible in the reuse ratio
`L2_read_bytes / DRAM_bytes`, which the achieved L2 rate tracks monotonically:

| shape | reuse (linear) | achieved L2 rate | % of the 7.00 TB/s hit rate |
|---|---|---|---|
| gpt_d4096_up | 2.70 | 4.10 TB/s | 58.5% |
| sq8192 | 5.19 | 5.67 TB/s | 81.0% |
| gpt_d1024_down | 6.85 | 7.00 TB/s | 100.0% |
| sq4096 | 6.98 | 6.62 TB/s | 94.6% |

**Criterion:** enable the raster iff `f(R_linear) * K * 2 > L2_bytes`. Post-raster reuse is 8.08
(gpt_d4096_up) and 9.72 (sq8192), both above the 6.85-6.98 that already achieve 94-100% of the hit
rate.

### 1.4 Effect sizes

**gpt_d4096_up is a hard arithmetic blocker today and the raster is what unblocks it.** Matching the
peer's 0.6734 ms with the linear order needs `2.3844 GB / 0.6734 ms = 3.541 TB/s` against a 3.35 TB/s
HBM peak: **impossible, whatever the mainloop does.** Post-raster the same target needs 1.184 TB/s
(35% of peak). Predicted post-raster, mcb2 arm selected:

```
    T_L2   = 2048 * (128+128) * 4096 * 2 / 7.00 TB/s = 613 us
    T_floor= 16 * (13.84 + 64 * 0.646)               = 883 us
    T_DRAM = 0.797 GB / 3.35 TB/s                    = 238 us
    T      = 883 us   ->  0.6734 / 0.883 = 76.3% of cuBLAS  (from 42.8%)
```

**+33.5 points, the single largest derived effect in Wave 3.** The plan's independent estimate for
the same shape ("42.8% -> ~70%") was computed for the *no-cluster* arm; with the cluster the number
is higher because the L2 roof moves out of the way at the same time.

**sq8192 is a partial win.** Post-raster the footprint is still 1.36x L2, so the achieved L2 rate
should move from 79.4% (mcb2, measured) toward the ~95% that reuse >= 7 delivers elsewhere:
`8.590 GB / (7.00 * 0.95) = 1292 us`, against a mainloop floor of ~1532 us -- so at sq8192 **the
raster removes the memory term entirely and leaves the shape floor-bound, worth roughly the gap
between 1546 us measured and 1532 us of floor: ~1%.** Publish sq8192's raster row as *insurance*
(it takes DRAM demand from 1.607 to 0.858 TB/s and de-risks the persistence gain), not as a headline.
This is a **downgrade** of the plan's "sq8192 58.8% -> 65-75%": that estimate was made before the
B-multicast landed and the multicast has already collected most of it (55.2% -> 82.2%).

### 1.5 THE REMAP: exact integer arithmetic, and why it must be division-free

**The correctness constraint comes first.** Under `Multicast::ClusterB` the producer's B copy is
(`ptx_wgmma.rs:3564-3567`)

```
    mul.lo.s32 %tmp2,%crank,{b_box_rows};   add.u32 %tmp2,%tmp2,%ctan;
    cp.async.bulk.tensor.2d...multicast::cluster [%rdB],[%rdTmB,{%tmp,%tmp2}],[%rdBarF],%cmask;
```

so **the two CTAs of a cluster must compute the SAME `%ctan`.** That is the whole correctness law;
their `%ctam` values need only be distinct (adjacency is a locality preference, not a requirement).
A raster applied to the raw `%ctaid` pair can violate it silently -- each rank fetches half of a B
tile the other does not want, and the result is wrong on half the accumulator columns with no error
anywhere. This is why CUTLASS applies its swizzle to the **cluster** index and re-adds
`cta_m_in_cluster` / `cta_n_in_cluster` afterwards
([sm90_tile_scheduler.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp)),
and why `ACT2_WAVE_PLAN.md:43` specifies "applied to the **cluster index** with the intra-cluster
rank re-added, not to raw `%ctaid`".

**The recommended construction: a 3-D grid, and a division-free prologue.**

The general grouped swizzle (Triton's matmul form, and the shape of CUTLASS's) needs two runtime
integer divisions, because both divisors (`GC*gx` and the ragged group height) are launch-dependent.
On PTX that is either two host-passed magic-number pairs -- which breaks the 6-parameter
`PARAM_ORDER` law that `every_wgmma_entry_declares_exactly_the_parameters_the_launcher_pushes`
enforces -- or ~18 instructions of `rcp.approx.f32` + two-sided correction. **Neither is necessary.**
CUDA grids are three-dimensional; put the group structure in the grid itself:

```
    host, for ClusterB with GROUP_M = 2*GC:
        m_tiles  = ceil(M / BM)
        n_tiles  = ceil(N / BN)
        gcy      = ceil(m_tiles / 2)              // m-clusters
        grid = ( x = GC,                          // cluster-row within the group
                 y = 2 * n_tiles,                 // n tile, with the cluster rank in bit 0
                 z = ceil(gcy / GC) )             // group index
        cluster dimension stays (1, 2, 1)         // gridDim.y = 2*n_tiles is even by construction
```

and the whole kernel-side remap, replacing `ptx_wgmma.rs:3436-3437`, is **six instructions and no
division**:

```
    mov.u32  %tmp,%ctaid.z;                  // group index
    mad.lo.s32 %tmp,%tmp,GC,%ctaid.x;        // m-cluster = z*GC + x
    shl.b32  %tmp,%tmp,1;
    add.u32  %tmp,%tmp,%crank;               // m tile   = 2*m_cluster + rank
    mul.lo.s32 %ctam,%tmp,BM;
    mov.u32  %tmp2,%ctaid.y;  shr.u32 %tmp2,%tmp2,1;   // n tile = y >> 1
    mul.lo.s32 %ctan,%tmp2,BN;
```

(`GC` and `BM`/`BN` are emitter constants -- each config already emits its own module -- so
`mad.lo.s32` takes an immediate. `%crank` is already `mov.u32 %crank,%cluster_ctarank` at
`ptx_wgmma.rs:3434`; it is *not* re-derived from a `%ctaid` component, which the source calls out as
"a second spelling of the axis that could disagree with `Multicast::cluster_shape`".)

**Why this delivers the intended footprint.** CTAs dispatch x-fastest, then y, then z. So the order
is: all `GC` cluster-rows of group 0 at column 0 (both ranks), then column 1, ... A wave of 132 CTAs
therefore covers `GC` cluster-rows = `2*GC` m-tiles and `132/(2*GC)` columns. At `GC = 8`
(`GROUP_M = 16`) that is **16 m-tiles x 8.25 n-tiles** -- exactly the `R = 16.25, C = 8.12` optimum of
1.1, and exactly the footprint that produced the 2.99x and 1.87x above.

**Bijection (G5).** The map is a mixed-radix decomposition of `(x, y>>1, z)`, so injectivity is
structural rather than argued: distinct `(z, x)` give distinct m-clusters because `x < GC`, distinct
`y>>1` give distinct n-tiles, and `%crank` splits each cluster into its two m-tiles. The Rust twin
G5 demands still has to exist, but it is asserting a decomposition, not a division-with-remainder
identity -- which is the difference between a proof and a hope.

**The ragged group, and its interaction with the pad-CTA scheme.** `ceil(gcy/GC)*GC - gcy` cluster
rows are surplus. For every shape in the suite `gcy` is 8 or 16 or 32, all multiples of `GC = 8`, so
**the pad is exactly zero on all seven benched shapes**. It is not zero in general (M = 4224 gives
`gcy = 17`, `ceil(17/8)*8 = 24`, a 41% pad), and a pad cluster that ran the full K loop over
zero-filled tiles would be 41% of the device doing nothing. The fix is a **cluster-uniform early
exit**:

```
    // FIRST instruction of the kernel body, BEFORE mbarrier.init and before any cluster barrier
    setp.ge.u32 %p0, m_cluster, gcy;
    @%p0 ret;
```

This is provably deadlock-free *because* `m_cluster = %ctaid.z*GC + %ctaid.x` is a function of
`%ctaid.z` and `%ctaid.x` only -- neither of which varies inside a 1x2x1 cluster -- so **both ranks
of a cluster take the branch together** and no peer is left waiting on a multicast, an `empty[s]`
arrival, or `barrier.cluster.wait`. It must precede `mbarrier.init` and
`fence.mbarrier_init.release.cluster` (`ptx_wgmma.rs:3449-3466`); an exit *after* the first
`barrier.cluster.arrive` is a hang. `gcy` is not a kernel parameter today; derive it on device as
`(ceil(M/BM) + 1) >> 1` from the existing `%M` parameter -- one `add`, one `shr`, no new param, no
change to `PARAM_ORDER`.

The **existing** pad CTA (odd `m_tiles` under ClusterB) is untouched by all of this and must stay
untouched: it has a valid `%ctan`, its `%ctam` overshoots M, TMA zero-fills, the epilogue predicates,
and -- the load-bearing part, documented at `ptx_wgmma.rs:1594-1600` -- **it still issues its
multicast slice**, without which its peer holds half a stale B tile. The raster does not remove that
CTA; it only decides which `(m,n)` it lands on, and by construction it lands on `m_tile = 2*cm + 1`
with the same `%ctan` as its peer. Correct by the same argument as before.

### 1.6 The one-visit A/B sweep row

Emit three modules, differing in exactly one emitter constant (`raster_gc`), and run them at the four
multi-wave shapes plus one single-wave control:

```rust
/// W1 + B multicast, linear order. THE CONTROL: byte-identical to the round-3 row that
/// measured 73.5% / 82.2%, so a drift here invalidates every other row of the visit.
pub const WGMMA_W1_MCB_G1: WgmmaCfg = WgmmaCfg { raster_gc: 1, ..WGMMA_W1_MCB };   // key "..._mcb2"
/// GROUP_M = 16 (GC = 8): f(16) = 4160, the derived optimum for a 128x256 tile on 132 SMs.
pub const WGMMA_W1_MCB_G8: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s4_mcb2_g8",
    key:  "wgmma_nt_f16_128x256x64_s4_mcb2_g8",   // G3: key derivable from geometry
    raster_gc: 8, ..WGMMA_W1_MCB };
/// GROUP_M = 32 (GC = 16): f(32) = 5152, +24% traffic. The BRACKET -- it must land BETWEEN
/// linear and G8, or the mechanism under test is not traffic.
pub const WGMMA_W1_MCB_G16: WgmmaCfg = WgmmaCfg { raster_gc: 16, ..WGMMA_W1_MCB };
```

```
  label            cfg                  why
  mcb_g1_lin       WGMMA_W1_MCB_G1      the control; must reproduce 73.5% / 82.2% within its floor
  mcb_g8           WGMMA_W1_MCB_G8      the derived optimum; predicted +33.5 points at gpt_d4096_up,
                                        ~+1 at sq8192, 0 at sq4096 / gpt_d1024_up, 0 at sq2048
  mcb_g16          WGMMA_W1_MCB_G16     the bracket: +24% traffic vs g8 must show as a partial gain,
                                        not a full one. A g16 that ties g8 refutes the traffic model.

  shapes: gpt_d4096_up (THE row), sq8192, sq4096, gpt_d1024_up, sq2048 (single-wave control: all
          three arms must tie inside dispersion, because the raster is the identity there)
```

**Refusal.** Any raster arm without the bijection twin (G5). Any gain at sq2048, sq1024 or
gpt_d1024_down reported as a raster effect -- the raster is provably the identity on a single-wave
grid, so a difference there is an instrument fault and invalidates the visit.

**Falsifiers, stated in advance.** (i) `mcb_g8` fails to move `gpt_d4096_up` -> the shape is not
L2-footprint-limited and section 0.4's 1.632 residual has another cause. (ii) `mcb_g16` matches
`mcb_g8` -> the effect is not traffic (suspect launch-order coalescing instead). (iii) any arm moves
`sq2048` -> the instrument, not the kernel.

---

## 2. PERSISTENT CTAs AND WAVE QUANTIZATION

### 2.1 Occupancy is 1 CTA/SM, and it is not negotiable at this tile

Confirmed twice over in 0.1, from the ptxas census rather than from arithmetic: SMEM 196 672 of
232 448 B (a second CTA misses by 160 896 B) and registers 384 x 168 = 64 512 of 65 536 (a second
CTA misses by 63 488). `setmaxnreg`'s 32p/232c split is a *within-CTA* redistribution and does not
enter the occupancy calculation; the number that does is the 168 the census reports. **A wave is
therefore exactly 132 CTAs**, and every wave figure below rests on that one census line.

### 2.2 Wave arithmetic, including the clustered launch

A 1x2x1 cluster occupies 2 SMs and both must be resident concurrently, so the scheduler allocates
`132 / 2 = 66` cluster slots. Because `tiles / 132 = clusters / 66` identically,

```
    waves = ceil(clusters / 66) = ceil(tiles / 132)         for every shape in the suite
```

**the cluster costs no wave.** (0.3 gives the SM-to-TPC argument for why no SM is stranded, and names
its falsifier.)

| shape | tiles (128x256) | waves | tail wave CTAs | tail wave util. | quantization loss |
|---|---|---|---|---|---|
| sq1024 | 32 | 1 | 32 | 24.2% | 75.8% (see 3.4) |
| sq2048 | 128 | 1 | 128 | 97.0% | 3.03% |
| sq4096 | 512 | 4 | 116 | 87.9% | 3.03% |
| sq8192 | 2048 | 16 | 68 | 51.5% | 3.03% |
| gpt_d1024_up | 512 | 4 | 116 | 87.9% | 3.03% |
| gpt_d1024_down | 128 | 1 | 128 | 97.0% | 3.03% |
| gpt_d4096_up | 2048 | 16 | 68 | 51.5% | 3.03% |

**The tail-wave utilisation column is a red herring and must not be quoted as a loss.** A tail wave
that runs 68 of 132 CTAs wastes 64 SMs for the duration of *one* tile out of sixteen waves; the loss
that matters is `1 - tiles/(132*waves) = 3.03%`, which is what the last column says. Both the 51.5%
and the 87.9% shapes lose the same 3.03%, which is the point of the closed form in 0.3.

**Persistence does NOT recover the 3.03%.** With `grid = 132` and a static schedule
`tile = ctaid; tile < ntiles; tile += 132`, at sq4096 116 CTAs run 4 tiles and 16 run 3, so the
critical path is still 4 tiles. The imbalance is *identical* to the wave picture; only the transition
between tiles changes. Removing the 3.03% requires splitting a tile's K across CTAs, which is
Stream-K (section 5.3), not persistence. **Anyone who attributes the persistence gain to wave
quantization has mis-attributed it.**

### 2.3 What persistence actually buys: the `X_fill` term, once per tile boundary

From 0.4: `X = 13.84 us` per tile, of which `X_fill = 7.85 us` (four ring stages at 1.96 us each,
running at 3.31 TB/s device-wide -- 98.7% of HBM peak) and `X_epi = 5.99 us`. At a **wave** boundary
today, `X_fill` is unhidable: no CTA anywhere on the device has work to overlap it with, because the
whole device just finished its previous tile.

Under a persistent CTA whose ring is *continuous* across the tile boundary:

* at the end of tile `t`'s k-loop the producer has nothing more to issue for `t`, and stages free as
  the consumers release them, so it immediately begins issuing tile `t+1`'s stage 0..3 copies;
* the consumers meanwhile execute `wgmma.wait_group 0` and the 128 predicated stores -- `X_epi`;
* the overlap window is therefore the last `stages` releases of tile `t` plus `X_epi`:
  `4 x 0.646 + 5.99 = 8.57 us`, against `X_fill = 7.85 us`.

`8.57 >= 7.85`, so **the fill is fully hidden and the saving is the whole of `X_fill` -- 7.85 us per
tile boundary.** The conservative half-overlap figure is 3.93 us; both are given per shape below.

`X_epi` is *not* saved. The consumer cannot issue tile `t+1`'s first `wgmma` until tile `t`'s
accumulators are stored, because `scale-d = 0` on that first instruction overwrites them
(`ptx_wgmma.rs:3618-3621`). The epilogue stays on the critical path.

**Cross-wave interaction, stated so the campaign does not double-count.** Wave 4's SMEM-staged /
TMA-store epilogue attacks `X_epi`; Wave 3's persistence attacks `X_fill`. They are two halves of the
same `X = 13.84 us`, so **their gains are not additive and the combined ceiling is 13.84 us per
boundary, not 13.84 + 7.85.** Worse, a shorter epilogue shrinks the window that hides the fill: at
`X_epi = 1 us` the window is `2.58 + 1.00 = 3.58 us` and only 46% of the fill is hidden. The two
waves must be measured against each other, not stacked on paper.

### 2.4 Per-shape effect

`T_new = T_measured - (waves - 1) * X_fill`, using the arm the section-3 dispatcher selects and the
peer time from the same run.

| shape | arm | waves | boundaries | measured us | peer us | today | full overlap | half overlap |
|---|---|---|---|---|---|---|---|---|
| **gpt_d1024_up** | no cluster | 4 | 3 | 106.05 | 58.11 | 54.8% | 82.50 us -> **70.4%** | 94.3 -> 61.6% |
| **sq4096** | mcb2 | 4 | 3 | 222.74 | 163.82 | 73.5% | 199.19 us -> **82.2%** | 210.9 -> 77.7% |
| **sq8192** | mcb2 | 16 | 15 | 1546.45 | 1271.81 | 82.2% | 1428.7 us -> **89.0%** | 1487.6 -> 85.5% |
| **gpt_d4096_up** | mcb2 + raster | 16 | 15 | 883 (predicted, 1.4) | 673.40 | 76.3% (post-raster) | 765.2 us -> **88.0%** | 824.1 -> 81.7% |
| sq2048 | no cluster | 1 | 0 | 39.09 | 26.27 | 67.2% | **no change** | -- |
| sq1024 | (see 3.4) | 1 | 0 | 21.5 | 7.0 | 32.3% | **no change** | -- |
| gpt_d1024_down | no cluster | 1 | 0 | 57.50 | 44.85 | 78.0% | **no change** | -- |

**`gpt_d1024_up` is the persistence shape.** It has `n_k = 16` k-stages against `X = 13.84 us`, so
`X` is 56% of its per-tile time -- more than three times its share at sq8192. +15.6 points is the
largest single-shape persistence gain in the wave, and it is available with no raster (that shape's
wave footprint is 10.6 MB, deeply L2-resident) and no cluster (54.2% of the L2 roof; see 3.2).

Three of seven shapes get **exactly zero** and must be benched as controls, not as rows.

### 2.5 Static schedule, not an atomic queue -- and the derivation is decisive

The brief offers "an atomic or static schedule". The atomic is wrong here, for a reason specific to
this suite: **every tile costs the same.** All tiles of a launch share `M`, `N`, `K` and therefore
`n_k`, so the only imbalance is the 3.03% quantization -- and a dynamic work queue does not fix
quantization either (132 workers over 512 equal-cost tiles still take 4 rounds, whoever hands them
out). A `red.global.add` per tile therefore buys nothing and costs two real things:

1. a global atomic round-trip on the critical path between tiles, i.e. it eats into the very
   `X_fill` overlap window persistence exists to open; and
2. **a nondeterministic tile-to-CTA mapping, which destroys the raster.** Section 1's entire gain is
   the *order* in which tiles are visited; a work queue that hands them out by arrival order makes
   the wave footprint a race outcome. It also breaks G8's two-run bit-identity as a *diagnostic*
   (the arithmetic stays bit-identical because each tile's reduction order is unchanged, but the
   L2 behaviour and hence the timing stop being reproducible).

**Verdict: static schedule, `cid += n_cluster_slots`. Do not implement the atomic.** Revisit only if
a ragged-K or mixed-shape batched entry is added.

### 2.6 THE DEADLOCK LAW: iterate over CLUSTERS, never over CTAs

This is the hazard of Wave 3 and it fails by hanging rented silicon, not by returning a wrong number.

Under `Multicast::ClusterB`, `empty[s]` is initialised with `cluster_ctas * consumer_wgs = 4`
arrivals and every consumer warpgroup arrives at the barrier of **every** CTA of the cluster through
`mapa` (`ptx_wgmma.rs:3629-3646`), while every producer multicasts into every peer's ring. Now take
the naive persistent loop `tile = ctaid; tile < ntiles; tile += gridDim`: at sq4096 with 512 tiles and
132 CTAs, **116 CTAs run 4 tiles and 16 run 3.** If the two ranks of a cluster land on opposite sides
of that split, then during rank 0's fourth tile rank 1 has already retired:

* rank 0's producer waits forever on `empty[s]` arrivals that rank 1 will never make -- a hang; and
* rank 0's producer multicasts into the shared memory of a CTA that has exited -- undefined.

**The law: the persistent loop is indexed by the CLUSTER, and its stride is the number of cluster
slots.**

```
    n_cluster_slots  = 132 / cluster_ctas          // 66 under ClusterB, 132 with no cluster
    n_cluster_tiles  = ceil(m_tiles / cluster_ctas) * n_tiles      // ClusterB pairs along M
    for (cid = %ctaid.x; cid < n_cluster_tiles; cid += n_cluster_slots) { ... }
```

Both ranks of a cluster share `%ctaid.x` (the cluster varies along `y`), so they iterate the
**identical** sequence and have **identical tile counts** by construction. This is not a check to be
added; it is a shape of loop that makes the check unnecessary, which is the only kind of fix worth
having for a deadlock.

Two consequences worth naming:

* the final `barrier.cluster.arrive.aligned / barrier.cluster.wait.aligned` at `EXIT`
  (`ptx_wgmma.rs:3704-3707`) now executes once per kernel, after the tile loop -- which is correct
  and still required, since a peer may touch this CTA's shared memory until the last tile drains;
* **persistence removes section 1's ragged-group pad entirely.** `cid` enumerates only real
  cluster-tiles, so `ceil(gcy/GC)*GC - gcy` never appears and the cluster-uniform early exit of 1.5
  is not needed on the persistent path. See 2.8 for how the two constructions reconcile.

### 2.7 The per-tile reset checklist, and G19

| register / object | per tile | why |
|---|---|---|
| `%kt` (producer AND consumer, separately) | **RESET to 0** | drives `%pfirst`; see G19 below |
| `%ctam`, `%ctan` | **RECOMPUTE** | the new tile origin |
| `%row0`, `%row1`, `%colb`, `%rdA`, `%rdB` (epilogue) | **RECOMPUTE** | derived from `%ctam`/`%ctan` |
| `%stg` (ring slot) | **CARRY** | resetting it forces a full ring drain per tile and throws away the entire lever |
| `%phf`, `%phe` (phase parities) | **CARRY** | same; and a reset parity against a live barrier is a hang |
| the mbarrier objects | **initialise once**, before the tile loop | re-initialising a barrier a peer may be signalling is the classic cluster race |
| the accumulators `%acc0..127` | no explicit reset | `scale-d = 0` on the first `wgmma` of each tile overwrites them -- provided G19 holds |

**G19, spelled out.** Today `%pfirst` is `setp.ne.u32 %pfirst,%kt,0` at `ptx_wgmma.rs:3605`, and the
first `wgmma` of each stage takes `scale-d = %pfirst` (line 3620). `%kt` is a whole-kernel counter
today because a kernel is one tile. **In a tile loop, a `%kt` that is not reset makes tile 2's first
`wgmma` take `scale-d = 1` and accumulate into tile 1's result.** With the round's exact-integer
operands the sum of two tiles is still an exact integer, so the corruption is a plausible-looking
number rather than a NaN, and it is unreachable on any guard shape with one tile per CTA -- which is
exactly why G1's corrected shape must have `tiles > CTAs`. The plan's law is the right one: **assert
that the count of `mov.u32 %kt,0` in the emitted text equals the number of tile-loop entries** (2
under persistence: one in the producer, one in the consumer).

**A second textual law the tile loop needs.** The tile-index arithmetic is emitted twice -- once in
the producer branch and once in the consumer branch -- and if the two copies ever disagree the
consumer computes with the wrong tile's operands: silently wrong, no hang, and *not* caught by the
epilogue's bounds predicates. Emit both from one `fn tile_index_ptx(&cfg) -> String` and assert
`ptx.matches(&tile_index_ptx(cfg)).count() == 2`. This is the same discipline
`Multicast::cluster_shape` already uses as "one function, so the launch attribute, the
`.reqnctapercluster` directive and the grid's divisibility rounding cannot disagree".

### 2.8 Reconciling the raster construction with the persistent loop

Section 1.5 gave a division-free remap by putting the group structure in a 3-D grid. That trick works
because the non-persistent kernel's tile index *is* `%ctaid`. Under persistence the tile index is a
loop variable, so the swizzle must be computed in-kernel from a flat `cid`, and two runtime integer
divisions come back. **Take them:** they cost ~26 instructions once per tile, against `n_k >= 16`
stages of ~1200 clocks each, i.e. **under 0.06% of a tile**. The exact u32 divmod, no host params, no
`PARAM_ORDER` change:

```
    // q = a / b, r = a % b, exact for a,b < 2^22 (f32 has a 24-bit significand and the
    // two-sided correction closes rcp.approx's 1-ulp error). 13 instructions.
    cvt.rn.f32.u32   %fa,%a;      cvt.rn.f32.u32 %fb,%b;
    rcp.approx.ftz.f32 %fr,%fb;   mul.f32 %fq,%fa,%fr;
    cvt.rzi.u32.f32  %q,%fq;
    mul.lo.s32 %t,%q,%b;          sub.s32 %t,%a,%t;
    setp.lt.s32 %pc,%t,0;         @%pc sub.u32 %q,%q,1;   @%pc add.s32 %t,%t,%b;
    setp.ge.s32 %pc,%t,%b;        @%pc add.u32 %q,%q,1;   @%pc sub.s32 %t,%t,%b;
```

The `a,b < 2^22` precondition is a `WgmmaCfg::validate` obligation, and it is never binding: the
largest `cid` in the suite is `gcy * n_tiles = 16 * 64 = 1024` (gpt_d4096_up), and the family already
declines at `M*N` beyond `u32` for the epilogue's element index (`ptx_wgmma.rs:1537-1542`), which
caps `cid` far below 2^22. **Assert it anyway** -- the bound is what makes the f32 route exact rather
than approximately right.

The grouped swizzle over the cluster index, with `GC = GROUP_M/2` a compile-time constant:

```
    grp   = cid / (GC * n_tiles)                   // divmod #1
    i     = cid % (GC * n_tiles)
    rows  = min(GC, gcy - grp*GC)                  // the last group may be short
    cn    = i / rows                               // divmod #2
    cm    = grp*GC + (i % rows)
    m_tile = 2*cm + %crank        n_tile = cn
```

This is Triton's grouped-M form applied to the cluster index, and it is bijective over
`[0, gcy*n_tiles)` including the short last group: for the last `grp`, `i` ranges over
`[0, rows*n_tiles)` and `(i % rows, i / rows)` covers `[0,rows) x [0,n_tiles)` exactly once. It is
also the shape CUTLASS uses -- swizzle the cluster index, then re-add the intra-cluster offset
(`cta_m_in_cluster` / `cta_n_in_cluster`), see
[sm90_tile_scheduler.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp).

**Recommendation:** land persistence and the raster together, on the flat-`cid` + divmod
construction. Keep 1.5's 3-D-grid form documented as the fallback if persistence slips, since it is
the only way to get the raster without any division at all.

### 2.9 The one-visit A/B sweep row

Three arms, because "persistence helps" and "the *continuous ring* helps" are different claims and
only the second one is the derivation above.

```rust
/// The control: section 1's winner, one tile per CTA. Must reproduce its own row.
pub const WGMMA_W1_MCB_G8: WgmmaCfg = /* section 1.6 */;
/// Persistent, ring RESET at every tile boundary (full drain, %stg and parities reset).
/// This arm isolates CTA dispatch from the fill overlap: if it ties the control, dispatch is
/// worth nothing and the whole gain is the ring, which is what 2.3 predicts.
pub const WGMMA_W1_MCB_G8_PSTOP: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s4_mcb2_g8_pstop",
    key:  "wgmma_nt_f16_128x256x64_s4_mcb2_g8_pstop",
    schedule: Schedule::PersistentDrained, ..WGMMA_W1_MCB_G8 };
/// Persistent with a CONTINUOUS ring across tile boundaries. THE ARM.
pub const WGMMA_W1_MCB_G8_PERSIST: WgmmaCfg = WgmmaCfg {
    name: "wgmma_nt_f16_128x256x64_s4_mcb2_g8_p",
    key:  "wgmma_nt_f16_128x256x64_s4_mcb2_g8_p",
    schedule: Schedule::Persistent, ..WGMMA_W1_MCB_G8 };
```

```
  label        cfg                        why
  g8_1tile     WGMMA_W1_MCB_G8            control; one tile per CTA
  g8_pstop     WGMMA_W1_MCB_G8_PSTOP      persistence WITHOUT the ring carry. Predicted: ties the
                                          control to within dispersion. If it wins, the gain is CTA
                                          dispatch and 2.3's derivation is wrong.
  g8_persist   WGMMA_W1_MCB_G8_PERSIST    the arm. Predicted +15.6 pts at gpt_d1024_up, +8.7 at
                                          sq4096, +6.8 at sq8192, +11.7 at gpt_d4096_up, 0 at sq2048

  shapes: gpt_d1024_up (THE row -- largest predicted delta, and no raster or cluster confound),
          sq4096, sq8192, gpt_d4096_up, sq2048 (single-wave control: ALL THREE arms must tie)
```

**Refusal.** Any persistent arm without the cluster-indexed loop of 2.6 -- a CTA-indexed persistent
loop under a cluster is a hang, and a hang costs the whole visit. Any persistent arm run before G1's
corrected guard shape (`tiles > CTAs`) is green, because G19's corruption is unreachable on a
one-tile-per-CTA guard and would ship undetected.

**Falsifiers.** (i) `g8_pstop` beats the control -> dispatch matters and `X_fill` is not the
mechanism. (ii) `g8_persist` moves `sq2048` -> the instrument. (iii) `g8_persist` gains less than
half the predicted amount at `gpt_d1024_up` -> `X_fill` is not fully hidden, and the next question is
the ring depth (a 3-stage ring has `X_fill = 5.89 us` and a *shorter* overlap requirement).

---

## 3. PER-SHAPE TILE DISPATCH

### 3.1 The three levers act on three different roofs, and that is why a dispatcher is needed

| lever | what it changes | which roof it lowers | zero when |
|---|---|---|---|
| **cluster (1x2x1, B multicast)** | `L2 -> SMEM` bytes: `(BM+BN)` becomes `(BM+BN/2)` per tile, a flat 1.5x at W1 | the **7.00 TB/s L2** roof (`I_cta` 85.33 -> 128.0, 597 -> 896 TFLOP/s) | the shape is not near the L2 roof |
| **raster** | `DRAM -> L2` bytes, by changing the *wave footprint*; L2->SMEM bytes UNCHANGED | the **3.35 TB/s HBM** roof, and L2 residency | the grid is one wave, or the linear footprint already fits L2 |
| **persistence** | nothing about traffic; removes `X_fill` at `waves-1` boundaries | the **mainloop fixed-cost** floor | the grid is one wave |

They are orthogonal in the sense that matters: the cluster does not change the footprint and the
raster does not change the per-tile L2 read. The dispatcher's job is to decide, per shape, which
roofs are actually binding.

### 3.2 THE CLUSTER PREDICATE, and the measured sq2048 preference it must explain

Define the L2-roof fraction of the un-clustered arm:

```
    T_L2_off(M,N,K) = tiles * (BM + BN) * K * 2 / 7.00e12       [seconds]
    f_L2            = T_L2_off / T_predicted_off                 [T_predicted_off from 0.4]
```

Measured, at the three shapes the round-3 sweep covered:

| shape | `T_L2_off` | measured off | `f_L2` | measured cluster effect (s4) | (s3, tight floors) |
|---|---|---|---|---|---|
| sq2048 | 28.8 us | 39.09 us | **0.736** | 39.09 -> 41.38, **-5.9%** | 38.51 -> 41.60, **-8.0%** |
| sq8192 | 1841 us | 2273.5 us | **0.810** | 2273.5 -> 1546.5, **+47.0%** | 1738.0 -> 1637.0, +6.2% |
| sq4096 | 230.1 us | 243.4 us | **0.946** | 243.4 -> 222.7, **+9.3%** | 239.9 -> 230.1, +4.3% |

**The sign of the cluster's effect flips between `f_L2 = 0.736` and `f_L2 = 0.810`.** The `s3` column
is the decisive evidence at sq2048: floors of +/-0.01% (`w1_s3_off`) and +/-0.15% (`w1_s3_mcb2`)
against an 8.0% effect. The `s4` pair (floors +/-3.29% and +/-1.84% against 5.9%) clears too, but
only just; quote the `s3` pair.

**Mechanism -- what the cluster costs when it is not needed.** Comparing each arm against its own
`T_floor` (using `X = 14.61` from Fit B, and `S_mcb2 = 0.634` derived from sq4096-mcb2 alone so that
sq8192-mcb2 is an *independent* prediction):

| arm / shape | `T_floor` | measured | residual |
|---|---|---|---|
| off / sq2048 | 39.41 us | 39.09 us | **0.992** (at the floor) |
| mcb2 / sq2048 | 36.90 us | 41.38 us | **1.121** (+4.5 us per tile) |
| mcb2 / sq4096 | 220.7 us (x4) | 222.7 us | 1.009 (+0.5 us per tile) |
| mcb2 / sq8192 | 1534.1 us (x16) | 1546.5 us | **1.008** (+0.9 us per tile) -- independent |

The clustered arm carries a fixed per-tile cost that is **4.5 us at `n_k = 32` and 0.5-0.9 us at
`n_k = 64/128`: it is progressively hidden as the mainloop lengthens.** Three mechanisms in the
emitted text produce exactly that shape:

1. `empty[s]` is initialised with `cluster_ctas * consumer_wgs = 4` arrivals instead of 2
   (`WgmmaCfg::empty_arrivals`), so a producer may not refill stage `s` until **both** CTAs'
   consumers have released it. The two CTAs' pipelines are lock-stepped at every stage and per-stage
   jitter becomes the max of two SMs rather than one.
2. every stage release now costs `cvta.to.shared` + `cvt.u32.u64` + two rounds of
   `mov`/`mapa.shared::cluster`/`mbarrier.arrive.shared::cluster` instead of one
   `mbarrier.arrive.shared::cta` (`ptx_wgmma.rs:3629-3648`) -- a cross-SM DSMEM barrier round trip,
   ~96 ns/stage at sq2048 (`3.09 us / 32 stages`), which is ~176 clocks and the right order for that
   network.
3. two `barrier.cluster.arrive/wait` rendezvous per kernel (`ptx_wgmma.rs:3472, 3705`).

**The predicate: cluster ON iff `f_L2 >= 0.78`.** The threshold sits inside the measured bracket
`[0.736, 0.810]` and is placed at its low end deliberately: a wrong OFF at sq8192 costs 27 points, a
wrong ON at sq2048 costs 5. Be eager.

**Evaluate it, including two shapes the round-3 sweep never measured:**

| shape | `f_L2` | predicate | status |
|---|---|---|---|
| sq1024 | 0.167 | OFF | untested |
| sq2048 | 0.736 | **OFF** | MEASURED, -8.0% with the cluster |
| gpt_d1024_up | 0.542 | OFF | **untested -- run it** |
| gpt_d4096_up (today) | 0.585 | OFF | untested |
| gpt_d4096_up (post-raster) | **0.955** | **ON** | untested -- the raster *flips* this decision |
| sq8192 | 0.810 | **ON** | MEASURED, +47.0% |
| sq4096 | 0.946 | **ON** | MEASURED, +9.3% |
| gpt_d1024_down | **1.000** | **ON** | **untested -- the predicate's most informative point** |

Two consequences the implementer must not miss:

* **`gpt_d4096_up` flips.** The predicate must be evaluated on the *post-raster, post-persistence*
  predicted time, not on today's measured time. Today that shape is memory-thrashing at 58.5% of the
  L2 hit rate and the cluster is worth nothing; once the raster makes its wave footprint L2-resident
  it lands at 0.955 and the cluster is worth 1.5x of a binding roof. **The dispatcher is a function
  of the *final* configuration, not of a measurement of an earlier one.**
* **`gpt_d1024_down` is the experiment that pins the threshold.** It is a single-wave shape
  (raster = identity, persistence = zero), it sits exactly at the 7.00 TB/s roof by construction --
  it *is* the `BW_L2` calibration -- and it is the only shape where the cluster is the sole variable.
  Predicted: `T_L2_on = 38.35 us` and `T_floor_on = 55.19 us`, so the cluster's saving is capped at
  `57.50 - 55.19 = 2.3 us` against a fixed cost of 0.5-0.9 us at `n_k = 64`: **a +2.5 to +3% win, and
  the smallest ON in the table.** If it loses, the threshold is above 1.0 and the cluster only ever
  pays when it also removes a *multi-wave* memory term.

### 3.3 The emittable tile set, and what the register file actually allows

Hard constraints from the source and the census:

* `Schedule::Cooperative` requires **CTA-M a multiple of 128** (`ptx_wgmma.rs:1249-1250`:
  "cooperative is illegal below CTA-M 128; a 64-row tile needs the pingpong schedule") and
  `bm = 64 * consumer_wgs`. So `bm` is 128 or 256. **`bm = 192` is not emittable** -- see 5.2.
* SMEM: `stages * (bm+bn) * bk * 2 + 16 * stages <= 232 448` for 1 CTA/SM, `<= 116 224` for 2.
* Registers: accumulators per consumer thread are `bn/2` f32. Occupancy is decided by the **static**
  per-thread allocation ptxas chooses, which the census reports as **168** for all three shipped
  rows; 2 CTAs/SM needs `65536 / (2*384) = 85`.

| tile | accs/thread | SMEM/stage | max stages @1 CTA/SM | 2 CTAs/SM? | `I_cta` | L2 roof (no cluster) |
|---|---|---|---|---|---|---|
| 128x256 | 128 | 49 152 B | 4 | no (needs 85 regs vs 128 accs alone) | 85.33 | 597 TFLOP/s |
| 128x128 | 64 | 32 768 B | 6 | **no** -- 64 accs + ~30 addressing > 85, census says 168 | 64.00 | 448 TFLOP/s |
| 128x64 | 32 | 24 576 B | 8 | **yes** at s4 (98 368 B, ~62 regs) | 42.67 | 299 TFLOP/s |

**The plan's "128x128 @ s3 for two CTAs/SM" does not survive the census.** ptxas already allocates
168 registers per thread for that entry with no cap; forcing 85 means fitting 64 live accumulators
plus the descriptor pairs, the two 64-bit stage bases and the loop state into 21 registers. It will
spill, and a mainloop spill is worth far more than the occupancy. **Verify on a CPU before spending a
visit:** re-run the $0.02 census with `--maxrregcount 85` over the 128x128 s3 entry and read
`spill_st`. If it is non-zero the row is dead and the high-occupancy tile is 128x64, not 128x128.

### 3.4 sq1024: the one shape only tile dispatch can move, and 128x128 is not the answer

sq1024 at 128x256 is 32 CTAs on 132 SMs: **24.2% of the device**, and no raster, cluster or
persistence touches it (one wave, `f_L2 = 0.167`, footprint 8.4 MB). The plan's occupancy ceiling
arithmetic, `(tiles/132) * 989 / peer_TFLOPs`, gives 78% for 128x256 and 155% for 128x128 -- correct,
and it is the *occupancy* ceiling, not the achievable number. Working the full model:

| tile | tiles | SM util | `X` (scaled to resident CTAs) | `n_k` | `S` | `T_floor` | `T_L2` | predicted | vs cuBLAS 7.0 us |
|---|---|---|---|---|---|---|---|---|---|
| 128x256 s4 | 32 | 24.2% | 8.65 us | 16 | 0.713 | 20.05 us | 3.6 us | 20.1 us | 32.3% (**measured 21.5 us**) |
| 128x128 s6 | 64 | 48.5% | ~5.8 us | 16 | ~0.337 | 11.2 us | 4.8 us | 11.2 us | **~62%** |
| **128x64 s4** | **128** | **97.0%** | ~5.3 us | 16 | ~0.165 | 7.94 us | 7.19 us | **7.94 us** | **~88%** |

The 128x256 row reproduces the measured 21.5 us to 7%, which is what licenses the other two.
**128x64 is the sq1024 tile: +56 points, the largest single-shape number in this dossier.** It is not
in the plan, and it needs one new thing -- an `m64n64k16` module family (the shape is in the ISA
menu, so `the_shape_menu_is_the_isa_menu` will accept it) -- plus the `f_L2` check, since its
`I_cta = 42.67` puts its L2 roof at 299 TFLOP/s and it must never be dispatched to a large shape.

### 3.5 sq2048: 128x128 is REFUTED there, by measurement

`ACT2_WAVE_PLAN.md:15` and D1 4.5 put W3C (128x128) at sq2048 "because a 256-wide tile quantizes
below `M*N = 4.3e6`". `sq2048` has `M*N = 4.194e6`, just under that literal. The round-3 sweep
measured it anyway, and:

```
    w1_s4_off  (128x256)  39.094 us     w3c_s6_off  (128x128)  43.757 us    -> 128x128 is 11.9% SLOWER
    w1_s3_off  (128x256)  38.512 us     w3c_s6_mcb2 (128x128)  44.609 us    -> 15.8% slower
```

Two mechanisms, both pointing the same way and neither of them quantization: (i) 128x256 at sq2048 is
**128 tiles = one wave at 97.0% efficiency**, so there is no quantization to fix; (ii) 128x128 has
`I_cta = 64.0` against 128x256's 85.33, i.e. **1.33x the L2 traffic** -- exactly the plan's own C1
note. The `M*N >= 4.3e6` literal is the wrong predicate. The right one is **wave efficiency**:

```
    use the WIDEST tile whose wave efficiency  tiles / (132 * ceil(tiles/132))  is >= 0.90
```

which accepts 128x256 at sq2048 (0.970) and rejects it at sq1024 (0.242), reproducing both measured
facts with one rule.

### 3.6 THE DISPATCH TABLE

Four classes, each defined by a predicate computable on the host from `(M, N, K)` alone.

| # | class predicate | shapes | tile | cluster | raster | persistent | derivation |
|---|---|---|---|---|---|---|---|
| **1** | wave eff. of 128x256 `< 0.90` | sq1024 | **narrow until eff >= 0.90**: 128x64 | OFF (`f_L2` 0.167) | n/a (1 wave) | n/a (1 wave) | 3.4 -- occupancy is the only binding constraint; 24.2% -> 97.0% of the device |
| **2** | `waves == 1`, eff `>= 0.90` | sq2048 | 128x256 | **OFF** (`f_L2` 0.736) | identity | zero | 3.5 (tile) + 3.2 (cluster). Both levers are provably zero; only the cluster decision exists here |
| | | gpt_d1024_down | 128x256 | **ON** (`f_L2` 1.000) | identity | zero | 3.2 -- the only single-wave shape at the L2 roof. Predicted +2.5-3%, the table's smallest ON |
| **3** | `waves > 1`, linear footprint `<= L2` | sq4096 | 128x256 | **ON** (0.946) | **OFF** (42.2 MB fits) | **ON** (3 boundaries) | 1.3 + 2.4. Raster provably worthless; persistence +8.7 pts |
| | | gpt_d1024_up | 128x256 | **OFF** (0.542) | **OFF** (10.6 MB) | **ON** (3 boundaries) | 2.4. `n_k = 16` makes `X` 56% of the tile: persistence +15.6 pts, the wave's best per-shape number |
| **4** | `waves > 1`, linear footprint `> L2` | sq8192 | 128x256 | **ON** (0.810) | **ON** (142.9 -> 68.2 MB) | **ON** (15 boundaries) | 1.4 + 2.4. Raster ~1% (the cluster already collected it), persistence +6.8 pts |
| | | gpt_d4096_up | 128x256 | **ON post-raster** (0.585 -> 0.955) | **ON** (136.4 -> 34.1 MB, *fits*) | **ON** (15 boundaries) | 1.4 + 3.2 + 2.4. The only shape where all three fire: 42.8% -> ~76% -> ~88% |

`L2 = 52 428 800 B`. Linear wave footprint `= f(132/min(gx,132)) * K * 2` from 1.2. `f_L2` from 3.2,
**evaluated on the post-raster configuration**.

### 3.7 The dispatch function, as the implementer writes it

A pure function of the shape, device-free, unit-testable without a GPU:

```
    fn dispatch(m, n, k, sm_count = 132) -> &'static WgmmaCfg {
        // 1. tile: widest tile whose wave efficiency clears 0.90
        for (bm, bn) in [(128,256), (128,128), (128,64)] {
            tiles = ceil(m/bm) * ceil(n/bn);
            if tiles as f64 / (sm_count * ceil(tiles/sm_count)) as f64 >= 0.90 { break }
        }
        // 2. raster: only if the linear wave footprint blows L2 (implies waves > 1)
        waves     = ceil(tiles / sm_count);
        gx        = ceil(n/bn);
        r_lin     = sm_count as f64 / min(gx, sm_count) as f64;
        footprint = (r_lin*bm + (sm_count as f64/r_lin)*bn) * k * 2.0;
        raster    = waves > 1 && footprint > L2_BYTES;
        group_m   = round(sqrt(sm_count * bn / bm)) rounded to EVEN;     // 16 at 128x256, 12 at 128x128
        // 3. persistence: any multi-wave grid
        persistent = waves > 1;
        // 4. cluster: the L2-roof fraction of the FINAL configuration
        t_l2   = tiles * (bm+bn) * k * 2 / BW_L2;
        t_fl   = waves * (X + ceil(k/64) * S);                 // X = 14.61 us, S = 0.7126 us at 128x256
        t_dram = dram_bytes(raster, ...) / HBM;
        cluster = t_l2 / max(t_fl, t_l2, t_dram) >= 0.78;
        // 5. the table lookup must land on a REAL emitted module, or decline loudly
        lookup(bm, bn, stages, cluster, raster, persistent, group_m)
    }
```

**Two laws on the dispatcher itself.** (a) It must be a *pure function* with a Rust unit test that
pins its verdict at all seven benched shapes and at the class boundaries (`M*N` just above and just
below each wave-efficiency threshold, `footprint` just above and just below `L2`, `f_L2` just above
and just below 0.78) -- because a dispatcher that silently picks a different module than the one the
round measured is G3's hazard in a new place. (b) Every `(tile, cluster, raster, persistent)`
combination the dispatcher can emit must exist as a shipped `WgmmaCfg` with a key derivable from its
own geometry, or the dispatcher must **decline**, never fall back. A fallback here is how a
publication ends up quoting a configuration that never ran.

### 3.8 The one-visit A/B sweep row

The dispatch decision needs exactly three rows that the previous rounds did not run:

```
  label            cfg                                  shape(s)              why
  mcb_gd1024down   WGMMA_W1_MCB (unchanged)             gpt_d1024_down        THE predicate row: the
                     vs WGMMA_W1 as the control                               only single-wave shape at
                                                                              f_L2 = 1.000, and the only
                                                                              one where the cluster is the
                                                                              sole variable. Predicted
                                                                              +2.5-3%; a loss moves the
                                                                              threshold above 1.0
  mcb_gd1024up     WGMMA_W1_MCB vs WGMMA_W1             gpt_d1024_up          the OFF prediction at
                                                                              f_L2 = 0.542. Predicted a
                                                                              LOSS of 3-8%; a win falsifies
                                                                              the predicate from below
  w3d_s4           128x64 s4 (NEW module family,        sq1024, sq2048        the tile row. Predicted
                     m64n64k16)                                               ~88% at sq1024 (from 32.3%)
                     vs WGMMA_W1 and WGMMA_W3C                                and a LOSS at sq2048, which is
                                                                              what makes 3.5's rule a rule
```

**Refusal.** Publishing any dispatch rule from a table that lacks `gpt_d1024_down` and
`gpt_d1024_up` with both cluster settings -- the threshold currently rests on a single sign flip
between two shapes, and a rule fitted to one crossing is a curve fitted to two points.

**Falsifiers.** (i) `gpt_d1024_down` prefers no cluster -> `f_L2 >= 0.78` is not the predicate and the
real one involves `waves`. (ii) `gpt_d1024_up` prefers the cluster -> the predicate is not `f_L2` at
all. (iii) 128x64 does not beat 128x256 at sq1024 -> occupancy is not sq1024's constraint and
`X`'s CTA-count scaling in 0.4 is wrong.
