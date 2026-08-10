//! Tensor-core GEMM via `wmma` PTX — the FLOP/s headline. On Ada's 4th-gen tensor cores (the box
//! these were developed on) bf16/fp16 multiplies run with **f32 accumulate** (the standard
//! mixed-precision contract), at many times the f32 CUDA-core rate. This is where low precision stops
//! being a footprint trick and buys real throughput.
//!
//! **Target floor.** Every instruction these generators emit — `wmma.load/mma.m16n16k16`,
//! `mma.sync.m16n8k16`, `ldmatrix`, `cp.async`, `lop3` — is defined by the PTX ISA at `sm_80`
//! (including the legacy 8×`.b32` f16 fragment spelling below), and no fp8 converter appears here.
//! So every module in this file is tagged with the [`crate::ptx_target::HDR_SM80`] floor, which
//! driver-JITs on Ampere and every later part; tagging it at the Ada arch would load on zero A100s.
//!
//! Each warp computes a `(16·tm)×(16·tn)` block of C as a `tm×tn` grid of `m16n16k16` WMMA tiles,
//! accumulating over K. The key optimization over one-tile-per-warp is **fragment reuse**: per
//! K-step a warp loads `tm` A-fragments and `tn` B-fragments and issues `tm·tn` MMAs, so each loaded
//! fragment feeds several MMAs (arithmetic intensity ↑). `A·Bᵀ` (nn.Linear): A is M×K row-major,
//! B is N×K row-major; the `Bᵀ` tile is the `.col` layout of B with leading dim K — no host
//! transpose (A loads `.row`, B loads `.col`, both stride K).
//!
//! Requires M, N, K multiples of the tile (16·tm, 16·tn, 16). Inputs are f16/bf16; accumulator and C
//! are f32. f16·f16→f32 and bf16·bf16→f32 are exact per product, so the only deviation from an f64
//! reference is the f32 accumulation order (tolerance-gated).

use crate::gpu::{smem_mode_for, SmemMode, DSMEM_DECL, DSMEM_SYM, STATIC_SMEM_CAP};
use crate::ptx_target::HDR_SM80;
use std::sync::OnceLock;

/// Comma-joined `{%p0,%p1,...}` register vector for a WMMA fragment.
fn veclist(prefix: &str, n: usize) -> String {
    let regs: Vec<String> = (0..n).map(|i| format!("%{prefix}{i}")).collect();
    format!("{{{}}}", regs.join(","))
}

/// Generate a WMMA GEMM entry computing `C = A·Bᵀ`, with each warp owning a `tm×tn` grid of 16×16
/// tiles. `ty` is "f16" or "bf16"; `name` is also used as the unique label tag.
fn entry(name: &str, ty: &str, tm: usize, tn: usize) -> String {
    let mma_ty = if ty == "f16" {
        "f32.f32".to_string()
    } else {
        format!("f32.{ty}.{ty}.f32")
    };
    // a/b fragment register count for m16n16k16: f16 uses the legacy 8×.b32 layout; bf16 uses 4×.b32.
    let nab = if ty == "f16" { 8 } else { 4 };
    let bm = 16 * tm;
    let bn = 16 * tn;

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{{\n"
    );

    // Register declarations.
    s += "    .reg .pred %p0;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%tmp;\n";
    // accumulator fragments c{ti}_{tj}_{r}
    let mut decl_c = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                decl_c += &format!("%c{ti}_{tj}_{r},");
            }
        }
    }
    s += &format!("    .reg .f32 {};\n", decl_c.trim_end_matches(','));
    // a fragments a{ti}_{r}, b fragments b{tj}_{r}
    let mut decl_ab = String::new();
    for ti in 0..tm {
        for r in 0..nab {
            decl_ab += &format!("%a{ti}_{r},");
        }
    }
    for tj in 0..tn {
        for r in 0..nab {
            decl_ab += &format!("%b{tj}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    // pointers
    let mut decl_ptr = String::from("%A,%B,%C,%off,%cptr");
    for ti in 0..tm {
        decl_ptr += &format!(",%aptr{ti}");
    }
    for tj in 0..tn {
        decl_ptr += &format!(",%bptr{tj}");
    }
    s += &format!("    .reg .b64 {decl_ptr};\n");

    // Params + globals.
    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");

    // Zero accumulators.
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                s += &format!("    mov.f32 %c{ti}_{tj}_{r},0f00000000;\n");
            }
        }
    }
    // A/B tile base pointers (byte addresses; 2 bytes/elem).
    for ti in 0..tm {
        s += &format!("    add.u32 %tmp,%baseRow,{};\n", ti * 16);
        s += "    mul.lo.s32 %tmp,%tmp,%K;\n    mul.wide.u32 %off,%tmp,2;\n";
        s += &format!("    add.s64 %aptr{ti},%A,%off;\n");
    }
    for tj in 0..tn {
        s += &format!("    add.u32 %tmp,%baseCol,{};\n", tj * 16);
        s += "    mul.lo.s32 %tmp,%tmp,%K;\n    mul.wide.u32 %off,%tmp,2;\n";
        s += &format!("    add.s64 %bptr{tj},%B,%off;\n");
    }

    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"), nab);
        s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%aptr{ti}], %K;\n");
    }
    for tj in 0..tn {
        let rb = veclist(&format!("b{tj}_"), nab);
        s += &format!("    wmma.load.b.sync.aligned.m16n16k16.col.{ty} {rb}, [%bptr{tj}], %K;\n");
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"), nab);
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"), nab);
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!(
                "    wmma.mma.sync.aligned.row.col.m16n16k16.{mma_ty} {cc}, {ra}, {rb}, {cc};\n"
            );
        }
    }
    for ti in 0..tm {
        s += &format!("    add.s64 %aptr{ti},%aptr{ti},32;\n");
    }
    for tj in 0..tn {
        s += &format!("    add.s64 %bptr{tj},%bptr{tj},32;\n");
    }
    s += &format!("    add.u32 %kt,%kt,16;\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    for ti in 0..tm {
        for tj in 0..tn {
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!("    add.u32 %tmp,%baseRow,{};\n", ti * 16);
            s += "    mul.lo.s32 %tmp,%tmp,%N;\n";
            s += &format!(
                "    add.u32 %tmp,%tmp,%baseCol;\n    add.u32 %tmp,%tmp,{};\n",
                tj * 16
            );
            s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
            s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n");
        }
    }
    s += "    ret;\n}\n";
    s
}

// ---- Shared-memory-staged WMMA GEMM (the `_sm` kernels) -------------------------------------------
// A CTA of SM_WARPS_M×SM_WARPS_N warps cooperatively stages a SM_BM×SM_BK tile of A and a SM_BN×SM_BK
// tile of B into shared memory each K-step, then every warp computes its SM_TM×SM_TN grid of 16×16
// WMMA tiles *out of shared memory*. So each global element is fetched once per CTA tile and reused by
// all warps — the per-warp `_mt` kernel instead reloads overlapping rows/cols straight from global,
// which spills L2 and craters at large N (the measured 2.3× cliff 2048³→4096³). Requires M%SM_BM==0,
// N%SM_BN==0, K%16==0; the dispatcher falls back to `_mt`/single-tile otherwise.
pub const SM_BM: usize = 64;
pub const SM_BN: usize = 64;
pub const SM_BK: usize = 16;
pub const SM_WARPS_M: usize = 2;
pub const SM_WARPS_N: usize = 2;
/// CTA thread count for the 64×64 `_sm` kernel (one warp per (SM_WARPS_M,SM_WARPS_N) cell).
pub const SM_THREADS: usize = SM_WARPS_M * SM_WARPS_N * 32;

// 128×128 CTA tile used by the double-buffered `wmma_nt_f16_sm128_db`: 8 warps (2×4) per CTA, each
// warp owns a 64×32 sub-tile (a 4×2 grid of 16×16 WMMA tiles). Doubling the output tile each way makes
// one CTA do 4× the work off the same-width A/B strips, so redundant inter-CTA global traffic (=
// M·N·K / tile_dim per operand) *halves* vs the 64×64 tile. On its own that barely moved the
// bandwidth-bound large GEMM (measured: ~neutral); combined with cp.async pipelining it becomes the
// best large-GEMM path (the cuBLAS recipe — big tile cuts traffic, pipeline hides what's left).
pub const SM128_BM: usize = 128;
pub const SM128_BN: usize = 128;
pub const SM128_WARPS_M: usize = 2;
pub const SM128_WARPS_N: usize = 4;
pub const SM128_THREADS: usize = SM128_WARPS_M * SM128_WARPS_N * 32;

/// One multi-stage `cp.async` pipeline GEMM config (see [`entry_smem_pipe`]). `name` is both the PTX
/// entry symbol and the bench label; `bm×bn` is the CTA macro-tile, `bk` the staged K-tile width, `wm×wn`
/// the warp grid, `stages` the pipeline depth (SMEM buffers). The generator, the correctness gate, the
/// `gemm_pipe_sweep` bench, and the `gemm_nt_f16` dispatch all iterate this single table — add a row and
/// it is generated, gated, swept, and dispatchable at once.
#[derive(Clone, Copy)]
pub struct PipeCfg {
    pub name: &'static str,
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
    pub stages: usize,
    /// Threadblock-rasterization group width (in N-tiles). `0` = no rasterization (the CTA grid maps
    /// directly to output tiles). `G>0` launches a 1-D grid and remaps each block to a tile via a
    /// column-band-of-`G` order, so co-scheduled CTAs share a compact A/B footprint in L2 — the cuBLAS
    /// trick that cuts effective HBM traffic at L2-thrashing sizes (4096³). Requires `bm`,`bn` powers of 2.
    pub raster: usize,
    /// `true` ⇒ generate the kernel with `mma.sync.m16n8k16` + hand-placed (optionally XOR-swizzled) SMEM
    /// fragment loads ([`entry_mma_pipe`]) instead of the opaque-layout `wmma.load`/`wmma.mma`
    /// ([`entry_smem_pipe`]). Same CTA tiling / pipeline / launch; only the inner tensor-core path differs
    /// — the route past the WMMA ceiling (conflict-free swizzled SMEM the `wmma.load` path cannot express).
    pub mma: bool,
    /// SMEM row padding in f16 elements (`mma` kernels only). `8` makes the b32 fragment loads
    /// bank-conflict-free (wins the L2-resident/compute-bound regime) but adds SMEM ⇒ fewer CTAs/SM; `0`
    /// keeps the conflicts but the smaller footprint fits more CTAs ⇒ more occupancy to hide HBM latency
    /// (can win the HBM-bound regime). Ignored when `mma == false`.
    pub pad: usize,
}

impl PipeCfg {
    pub const fn threads(&self) -> usize {
        self.wm * self.wn * 32
    }
    /// Static SMEM bytes this config needs (`stages·(bm+bn)·(bk+pad)·2`, the padded row stride for mma
    /// kernels; `pad==0` for WMMA gives the plain `bk`); must be ≤ 48 KiB.
    pub const fn smem_bytes(&self) -> usize {
        self.stages * (self.bm + self.bn) * (self.bk + self.pad) * 2
    }
}

/// The fp16 multi-stage-pipeline GEMM variants, **trimmed to the per-regime winners** after sweeping ~20
/// configs against cuBLAS (same-run, full-clock). The binding constraint splits cleanly by whether A+B fit
/// the 24 MB L2:
///   * **L2-resident (≤ ~2048³): a deep BK=16 pipeline wins** — data is hot in L2 so latency is low, and
///     more SMEM buffers keep the tensor cores fed (`pipe_64_s6` ~90% of cuBLAS at 1024³, `pipe_128_s4`
///     ~87–94% at 2048³). Wider BK there only wastes SMEM.
///   * **L2-thrashing (≥ ~4096³): the GEMM is HBM-bound, so threadblock rasterization wins** — banding
///     co-scheduled CTAs into a compact L2 footprint cuts effective traffic. `pipe_128_bk32_s2_r8` reached
///     ~72% of cuBLAS at 4096³ (vs 56% un-rasterized and the BK=16 pipes' ~27% *collapse*). Bigger
///     dynamic-SMEM tiles (256×128, 256×64-bk64) and deeper pipes there *lost* to the occupancy drop.
///
/// `gemm_nt_f16` dispatches among these by working-set size; `gemm_pipe_sweep` re-measures them. (~72% is
/// near the WMMA ceiling on this part; the cuBLAS-class `mma.sync`+`ldmatrix` path is the next lever.)
pub const PIPE_VARIANTS: &[PipeCfg] = &[
    PipeCfg {
        name: "wmma_nt_f16_pipe_64_s6",
        bm: 64,
        bn: 64,
        bk: 16,
        wm: 2,
        wn: 2,
        stages: 6,
        raster: 0,
        mma: false,
        pad: 0,
    }, // 24 KiB — ≤1024³
    PipeCfg {
        name: "wmma_nt_f16_pipe_128_s4",
        bm: 128,
        bn: 128,
        bk: 16,
        wm: 2,
        wn: 4,
        stages: 4,
        raster: 0,
        mma: false,
        pad: 0,
    }, // 32 KiB — ~2048³
    // Spilling champion: a 128×128 BK=32 tile with **threadblock rasterization** (r8). At 4096³ the GEMM
    // is HBM-bound (~2.7 GB of A/B reads ≫ the 134 MB minimum), so banding co-scheduled CTAs into a compact
    // L2 footprint cut effective traffic: 56% → 72% of cuBLAS (a 12-config raster sweep found the optimum
    // broad over r8..r16; depth beyond s2 and the 256×64 tile both lost). %128, K%32.
    // The workhorse for everything ≥ 2048³: `mma.sync.m16n8k16` with hand-placed, **bank-conflict-free**
    // (8-padded SMEM) fragment loads + **wide (r16) threadblock rasterization**. Beats the WMMA path at
    // both regimes — 2048³ ~97% of cuBLAS (≥ the M1 target; padding kills the 4-way fragment-load conflict
    // in the L2-resident/compute-bound regime, the r16 band maximizes L2 reuse) and 4096³ ~72–75%
    // (HBM-bound). The r16 raster beat r8 at both sizes for the mma kernel (a 6-config sweep); the smaller
    // 64-tile and 128×64 tile both lost. %128, K%32.
    PipeCfg {
        name: "mma_nt_f16_128_bk32_s2_r16",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        mma: true,
        pad: 8,
    }, // 40 KiB
];

/// The bf16 large-GEMM workhorse — the bf16 twin of the f16 spilling champion `mma_nt_f16_128_bk32_s2_r16`
/// (`mma.sync.m16n8k16` is precision-generic: bf16 packs into the same 4×b32 A / 2×b32 B fragments, only
/// the mma type tag changes). Generated into the bf16 module; `gemm_nt_bf16` dispatches A+B ≳ L2 to it,
/// replacing the un-staged `_mt` path bf16 large GEMM used before (no pipeline at all). bf16 is the
/// dominant *training* precision, so this carries the cliff fix to training-shaped GEMMs.
pub const PIPE_BF16: PipeCfg = PipeCfg {
    name: "mma_nt_bf16_128_bk32_s2_r16",
    bm: 128,
    bn: 128,
    bk: 32,
    wm: 2,
    wn: 4,
    stages: 2,
    raster: 16,
    mma: true,
    pad: 8,
};

/// Look up a [`PipeCfg`] by its entry name (the `gemm_nt_f16` dispatcher selects variants this way, so a
/// renamed/removed table row fails loudly at the call site rather than silently mis-dispatching).
pub fn pipe_variant(name: &str) -> &'static PipeCfg {
    PIPE_VARIANTS
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("unknown pipe variant {name:?}"))
}

/// **GEMM-cliff experiment candidates** (branch `perf/gpu-gemm-cliff-2`). A self-contained family of
/// large-GEMM workhorse variants — all 128×128 / BK=32 / r16 raster `mma.sync.m16n8k16`, varying only the
/// SMEM layout (8-padded vs no-pad XOR-swizzle), the cp.async pipeline **depth** (`stages`), and the
/// **launch-bounds** (`min_ctas` ⇒ `.minnctapersm`). The research existence-proof (on Ada, 4096³ → 100% of
/// cuBLAS) attributes the climb past ~74% to warp-tile ILP + swizzle + a **3-stage** pipeline (CUTLASS's
/// SM80 floor is 3; the production workhorse is at 2). The no-pad swizzle is what frees the SMEM for s3
/// (s2 padded = 40 KiB, s3 padded = 60 KiB > the 48 KiB static cap; s3 no-pad = 48 KiB exactly). All are
/// numerically identical `C = A·Bᵀ`; gated by `gemm_cliff_matches_reference`, A/B-timed by `gemm_cliff_ab`.
#[derive(Clone, Copy)]
pub struct CliffCfg {
    pub name: &'static str,
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
    pub stages: usize,
    pub raster: usize,
    pub swz: bool,
    pub pad: usize,
    pub min_ctas: usize,
    /// Epilogue global-store mode ([`Store`]) — bit-identical C, scalar vs vectorized `st.global.v2.f32`.
    pub store: Store,
}

impl CliffCfg {
    /// SMEM bytes (`stages·(bm+bn)·(bk+pad)·2`); `pad==0` on the swizzle path.
    pub const fn smem_bytes(&self) -> usize {
        self.stages * (self.bm + self.bn) * (self.bk + self.pad) * 2
    }
    pub const fn threads(&self) -> usize {
        self.wm * self.wn * 32
    }
    /// How this config's SMEM is declared, and therefore how its launch must be sized — static at or
    /// below the PTX ISA's 48 KiB cap, one `.extern` window beyond it. The launch wrapper takes the
    /// byte count from here (`SmemMode::launch_bytes`), never re-derives it.
    pub const fn smem_mode(&self) -> SmemMode {
        smem_mode_for(self.smem_bytes())
    }
}

/// The cliff sweep. `cliff_swz_s2` is byte-identical to the dispatched swizzle workhorse (the A/B base).
/// The rest probe the research's **#1 lever — warp/threadblock tile shape**. The 4096³ win was the warp
/// grid (w22, 3 CTAs/SM). The 2048³ regime is the open floor: a 128×128 tile gives only 16×16 = **256
/// macro-tiles on 20 SMs** (~4 waves at 3 CTAs/SM → tail/quantization waste), and the 16.8 MB working set
/// sits right at the 12 MB L2 edge. *Smaller* tiles double/quadruple the tile count → better SM load
/// balance (the classic small-GEMM lever), trading arithmetic intensity that L2 can absorb at this size.
/// All BK=32 (the swizzle phase is derived for it), r16 raster (swept — optimal at both sizes).
pub const CLIFF_VARIANTS: &[CliffCfg] = &[
    // Three production-shape anchors (all 128×128, BK=32, r16 — the swept-optimal macro-tile/raster).
    //  - `cliff_swz_s2`  : no-pad swizzle, w24 (2×4) — the dispatched ≥48 MB base, the A/B reference.
    //  - `cliff_swz_w22` : no-pad swizzle, w22 (2×2) — the 4096³ warp-tile winner (3 CTAs/SM, +ILP).
    //  - `cliff_pad_w24` : padded (pad=8), w24 — byte-identical to the production `mma_nt_f16_128_bk32_s2_r16`
    //    that the 16–48 MB arm dispatches; here to settle **swz-vs-padded @2048³ same-run** (the open floor).
    CliffCfg {
        name: "cliff_swz_s2",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "cliff_swz_w22",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 2,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "cliff_pad_w24",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: false,
        pad: 8,
        min_ctas: 0,
        store: Store::Scalar,
    },
    // ---- 4096³ re-tune levers (perf/gpu-gemm-4096) — all w24, no-pad swizzle unless noted; auto-gated by
    // `gemm_cliff_matches_reference`, A/B-timed by `gemm_cliff_ab`. The central sweep picks winners. ----
    //  * **3-stage pipeline** (`_s3`): CUTLASS's SM80 floor is 3 stages; the no-pad swizzle frees the SMEM
    //    for it — s3 no-pad = 48 KiB *exactly* (s3 padded = 60 KiB > the 48 KiB static cap). More cp.async
    //    buffers deepen the prefetch window to hide the HBM latency the 4096³ GEMM is bound by.
    CliffCfg {
        name: "cliff_swz_s3",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    //  * **s3 raster-band re-tune** (`_r8`/`_r32`): the rasterization window sets the co-scheduled CTAs'
    //    A/B footprint; re-tune it against the *measured* L2 at 4096³ (the r16 optimum was found at s2).
    CliffCfg {
        name: "cliff_swz_s3_r8",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 8,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "cliff_swz_s3_r32",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 32,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    //  * **launch-bounds** (`_mc3`/`_mc2`): force the occupancy ptxas leaves on the table. The 128×128 tile
    //    carries 64 f32 accumulators/thread, so the JIT defaults to ~2 CTAs/SM (register-bound). `_s2_mc3`
    //    caps registers to fit 3 CTAs/SM on the 32 KiB s2 tile (may spill — that's what the A/B measures);
    //    `_s3_mc2` pins 2 CTAs/SM on the 48 KiB s3 tile (2×48 = 96 KiB ≤ the 100 KiB SM carveout).
    CliffCfg {
        name: "cliff_swz_s2_mc3",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 3,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "cliff_swz_s3_mc2",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 2,
        store: Store::Scalar,
    },
    //  * **epilogue store vectorization** (`_v2`/`_v2cs`): fold each lane's adjacent-column D-fragment pair
    //    into one `st.global.v2.f32` (half the C-write store count); `_v2cs` adds the `.cs` streaming hint so
    //    the write-once C stream does not evict the L2-resident A/B raster band. Bit-identical to `cliff_swz_s2`.
    CliffCfg {
        name: "cliff_swz_s2_v2",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::V2,
    },
    CliffCfg {
        name: "cliff_swz_s2_v2cs",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::V2Cs,
    },
];

/// The SMEM budget the **deep** table is generated against: the *smallest* opt-in carveout among the
/// parts this project targets (Ada, 99 KiB = 101376 B, measured on the dev card; A100 163 KiB, H100
/// 227 KiB). Spelled as a constant rather than probed, so `gemm_deep_ptx()` stays a pure text function
/// with one cached module and its gates run with no device present — and so every row is guaranteed
/// loadable on **every** target, not just the biggest. A device whose ceiling is lower than a row's
/// footprint declines that row at dispatch (`Gpu::smem_budget()`), it does not get a different kernel.
pub const DEEP_SMEM_BUDGET: usize = 101_376;

/// The SMEM budget the **wide** table ([`PIPE_WIDE_VARIANTS`]) is generated against: the smallest opt-in
/// carveout among the **datacenter** parts, **A100's 163 KiB**. Its rows are the CTA tiles whose useful
/// pipeline depths do not fit *any* Ada part (this laptop and the L40S both cap at 99 KiB), so unlike
/// [`DEEP_SMEM_BUDGET`] this budget cannot promise "loadable everywhere" — it promises the next weaker
/// and still checkable thing: **every wide row runs on every datacenter part this project targets**
/// (A100 163 KiB, H100 227 KiB — CUTLASS spells the latter `sm90_smem_capacity_bytes = 232448`).
///
/// It is a *binding* guard, not a rubber stamp: 128x256 bk32 costs 24 KiB/stage, so s6 = 144 KiB passes
/// and s7 = 168 KiB is refused at generation. The device side is already honest without it —
/// `gemm_nt_f16_deep` compares `smem_bytes()` against the probed `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`
/// and returns a capability decline naming both numbers, and the deep gate turns that into a
/// `[skip:capability]` line rather than a pass — so an Ada card refuses these rows loudly.
pub const WIDE_SMEM_BUDGET: usize = 166_912;

/// **The f16 variable-stage grid — the `mma.sync` pipeline past the 48 KiB static wall.**
///
/// The dispatched workhorse is 2-stage and the deepest thing this crate could previously express was
/// `cliff_swz_s3` at 3·(128+128)·32·2 = **48 KiB exactly** — not a tuning verdict, a wall: PTX caps a
/// *static* `.shared` declaration at 48 KiB on every device, so stage 4 had nowhere to live. These rows
/// put the ring in the `.extern` window instead:
///
/// | row | tile | SMEM | form | CTAs/SM here (100 KiB/SM) | A100 (164) | H100 (228) |
/// |---|---|---|---|---|---|---|
/// | `deep_swz_128_s2` | 128×128 | 32 KiB | static (== `cliff_swz_s2`) | 2 (register-bound) | 2 | 2 |
/// | `deep_swz_128_s3` | 128×128 | 48 KiB | static (== `cliff_swz_s3`) | 2 | 2 | 2 |
/// | `deep_swz_128_s4` | 128×128 | 64 KiB | **dynamic** | 1 | 2 | 2 |
/// | `deep_swz_128_s5` | 128×128 | 80 KiB | **dynamic** | 1 | 2 | 2 |
/// | `deep_swz_128x256_s2` | 128×256 | 48 KiB | static | 1 | 3 | 4 |
/// | `deep_swz_128x256_s3` | 128×256 | 72 KiB | **dynamic** | 1 | 2 | 3 |
/// | `deep_swz_128x256_s4` | 128×256 | 96 KiB | **dynamic** | 1 | 1 | 2 |
/// | `deep_swz_128x256_s4_mc1` | 128×256 | 96 KiB | **dynamic** | 1 | 1 | 2 |
/// | `deep_swz_256x128_s2` | 256×128 | 48 KiB | static | 1 | 3 | 4 |
/// | `deep_swz_256x128_s3` | 256×128 | 72 KiB | **dynamic** | 1 | 2 | 3 |
/// | `deep_swz_256x128_s4` | 256×128 | 96 KiB | **dynamic** | 1 | 1 | 2 |
///
/// `_s4` is D1's A2 — "the cheapest possible test of the whole dynamic-SMEM capability", one table row.
///
/// **The 128×256 / 256×128 block is the CTA-tile-width lever** (D1 §2.5). At 128×128 the binding Act-1
/// ceiling on H100 is **L2→SMEM fill bandwidth** (525 TFLOPS, 73% of cuBLAS@4096³) because CTA
/// arithmetic intensity is only `bm·bn/(bm+bn)` = **64 FLOP per byte of SMEM filled**. Doubling either
/// side takes that to **85.33**, which lifts the fill ceiling to ~700 and moves the binding constraint
/// onto the mma.sync **issue** rate at 642 TFLOPS = **90% of cuBLAS** — +17 points of headroom for no
/// new instruction, no `wgmma`, no TMA. The two orientations are not redundant: they share `I_cta` but
/// differ in warp-tile SMEM intensity (`I_wrp` 32.0 for 128×256 vs 25.6 for 256×128, because a B
/// fragment is 2×b32/lane against A's 4), so an A/B between them separates "CTA area" from "the
/// B-fragment read path" and tells Act 2 whether to widen `wgmma` N or M first.
///
/// Both orientations are generator-legal at `wm2 wn4` (`wmr`/`wnc` ∈ {32,64,128}, all ≡ 0 mod 8) and
/// carry **128 f32 accumulators per thread** — `tm·tn·4` with (tm,tn) = (4,8) and (8,4) — twice the
/// 128×128 tile's 64, so the register wall, not SMEM, is what decides whether they hold up
/// (`wide_tile_lattice_is_generator_legal_and_register_bounded` does that arithmetic; the `_mc1` twin
/// exists because `.minnctapersm 1` is what lets ptxas spend up to 255 regs/thread instead of guessing).
/// The rows here are the depths a **99 KiB Ada** part can actually run, so this card gates them on the
/// metal today; [`PIPE_WIDE_VARIANTS`] carries the deeper datacenter-only rings.
///
/// The s2/s3 128×128 rows are the shipped cliff kernels byte for byte
/// (`deep_grid_s2_s3_are_the_shipped_cliff_kernels`), so they are the equivalence anchor: every deep row
/// must agree with them, and with the f64 oracle, at the crate's fp16 tolerance.
pub const PIPE_DEEP_VARIANTS: &[CliffCfg] = &[
    CliffCfg {
        name: "deep_swz_128_s2",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_128_s3",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_128_s4",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 4,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_128_s5",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 5,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    // ---- The CTA-tile-width lever, N-major (128x256). `_s2` is the one row that needs NO dynamic SMEM
    // at all (24 KiB/stage x 2 = 48 KiB exactly, the static ISA cap), so it is the cleanest single-
    // variable A/B in the file: same bk, same warp grid, same depth, same raster as `deep_swz_128_s2`,
    // and only `bn` doubles. `_s4` is the deepest 128x256 ring a 99 KiB Ada carveout can hold.
    CliffCfg {
        name: "deep_swz_128x256_s2",
        bm: 128,
        bn: 256,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_128x256_s3",
        bm: 128,
        bn: 256,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_128x256_s4",
        bm: 128,
        bn: 256,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 4,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    // The launch-bounds twin of the row above, and the ONLY thing on this laptop that can answer D1's
    // open question 3 ("does ptxas spill at 128 accumulators?"). Without a directive ptxas picks the
    // register count from its own occupancy heuristic and cannot see the dynamic SMEM size (a launch-time
    // quantity), so it may cap registers for a residency the window will not permit anyway; with
    // `.minnctapersm 1` it may spend up to 65536/256 = 256 -> the 255/thread ISA limit. Same PTX
    // arithmetic, same output; only the two directives differ.
    CliffCfg {
        name: "deep_swz_128x256_s4_mc1",
        bm: 128,
        bn: 256,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 4,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 1,
        store: Store::Scalar,
    },
    // ---- The same lever, M-major (256x128) — identical SMEM and identical CTA intensity, deliberately
    // different warp-tile intensity (tm=8,tn=4 => I_wrp 25.6 vs the N-major 32.0). Keeping both at every
    // depth is what makes the orientation A/B a controlled experiment rather than two unrelated points.
    CliffCfg {
        name: "deep_swz_256x128_s2",
        bm: 256,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 2,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_256x128_s3",
        bm: 256,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 3,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "deep_swz_256x128_s4",
        bm: 256,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 4,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
];

/// **The datacenter-only wide-tile rings** — the rows of D1 §2.2's feasible lattice that no Ada part can
/// hold, generated against [`WIDE_SMEM_BUDGET`] instead of [`DEEP_SMEM_BUDGET`].
///
/// | row | tile | stages | SMEM | D1 label | CTAs/SM: A100 (163) / H100 (227) |
/// |---|---|---|---|---|---|
/// | `wide_swz_128_s7` | 128×128 | 7 | 112 KiB | A2, taken to the datacenter depth | 1 / 2 |
/// | `wide_swz_128x256_s5_mc1` | 128×256 | 5 | 120 KiB | — (the depth step below A3) | 1 / 1 |
/// | **`wide_swz_128x256_s6_mc1`** | **128×256** | **6** | **144 KiB** | **A3 — the #1-ranked Act-1 lever** | 1 / 1 |
/// | **`wide_swz_256x128_s6_mc1`** | **256×128** | **6** | **144 KiB** | **A4 — the M-major twin** | 1 / 1 |
///
/// A3/A4 are D1 §4.4 verbatim: `bk32 wm2 wn4 s6 r16 swz pad0 min_ctas1`, 256 threads, 128 accumulator
/// registers per thread, one CTA per SM. The predicted payoff is **+18 to +25 points of cuBLAS at
/// 4096³** and the predicted failure mode is equally specific — if A3 loses to the 128×128 base, the
/// 1-CTA/SM occupancy collapse (8 of H100's 64 warp slots) beats the intensity gain and the answer is
/// warp specialisation, i.e. Act 2. Either result settles the question, which is why both orientations
/// are here rather than only the favourite.
///
/// These rows live in the **same PTX module** as [`PIPE_DEEP_VARIANTS`] ([`gemm_deep_ptx`]) on purpose.
/// A dynamic-SMEM entry declares no size in PTX — the window is `.extern` and unsized, and the byte
/// count is launch-time state — so a module carrying a 144 KiB row still `cuModuleLoadData`s on this
/// 99 KiB laptop; only a *launch* would be refused, and `gemm_nt_f16_deep` refuses it before touching
/// the driver. Sharing the module also keeps every textual law that already scans it (the ASCII gate,
/// the `sm_80` floor law, `gpu.rs`'s `.version` floor law over `device_free_modules`) covering these
/// rows for free, instead of creating a second module that sits outside all of them.
pub const PIPE_WIDE_VARIANTS: &[CliffCfg] = &[
    CliffCfg {
        name: "wide_swz_128_s7",
        bm: 128,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 7,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 0,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "wide_swz_128x256_s5_mc1",
        bm: 128,
        bn: 256,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 5,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 1,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "wide_swz_128x256_s6_mc1",
        bm: 128,
        bn: 256,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 6,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 1,
        store: Store::Scalar,
    },
    CliffCfg {
        name: "wide_swz_256x128_s6_mc1",
        bm: 256,
        bn: 128,
        bk: 32,
        wm: 2,
        wn: 4,
        stages: 6,
        raster: 16,
        swz: true,
        pad: 0,
        min_ctas: 1,
        store: Store::Scalar,
    },
];

/// Look up a [`PIPE_DEEP_VARIANTS`] row by entry name (a wrong name is a loud panic at the call site,
/// never a silent mis-dispatch).
pub fn deep_variant(name: &str) -> &'static CliffCfg {
    PIPE_DEEP_VARIANTS
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("unknown deep variant {name:?}"))
}

/// Look up a [`PIPE_WIDE_VARIANTS`] row by entry name. Same contract as [`deep_variant`]; both live in
/// the one [`gemm_deep_ptx`] module, so the row a caller gets back is launchable through
/// `gpu::gemm_nt_f16_deep` unchanged — which declines loudly if the device's opt-in ceiling is smaller.
pub fn wide_variant(name: &str) -> &'static CliffCfg {
    PIPE_WIDE_VARIANTS
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("unknown wide variant {name:?}"))
}

/// Emit the **deep / wide** (variable-stage, variable-tile) f16 module: the module-scope dynamic-SMEM
/// window, then one `mma.sync` entry per [`PIPE_DEEP_VARIANTS`] row and one per [`PIPE_WIDE_VARIANTS`]
/// row. The only difference between the two tables is the budget each is generated against
/// ([`DEEP_SMEM_BUDGET`] = every part, [`WIDE_SMEM_BUDGET`] = datacenter parts only) — the generator,
/// the window, the launch path and every textual law are shared.
///
/// One module, one window, many entries — legal and deliberate: `sharedMemBytes` is a *per-launch*
/// quantity and `cuFuncSetAttribute` is per-`CUfunction`, so each entry sizes its own window
/// independently even though they share the symbol. Static rows keep their own entry-local `.shared`
/// arrays alongside it (statics and the window do not alias; the window simply starts after them).
/// A row bigger than the running device's carveout costs this module nothing at load time — the
/// `.extern` window carries no size in PTX — so the wide rows ride along on an Ada card and are refused
/// only if somebody launches them. Separate from `gemm_cliff_ptx`/`wmma_f16_ptx` so the deep experiments
/// never perturb the dispatched modules or their warm cubins.
pub fn gemm_deep_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        // Declared once, at MODULE scope (inside an entry body it is CUDA_ERROR_INVALID_PTX), and only
        // when some row actually needs it — so a hypothetical all-static table emits historical text.
        if PIPE_DEEP_VARIANTS
            .iter()
            .chain(PIPE_WIDE_VARIANTS)
            .any(|v| v.smem_mode().is_dynamic())
        {
            m += DSMEM_DECL;
        }
        for (table, budget) in [
            (PIPE_DEEP_VARIANTS, DEEP_SMEM_BUDGET),
            (PIPE_WIDE_VARIANTS, WIDE_SMEM_BUDGET),
        ] {
            for v in table {
                m += &entry_mma_pipe_budget(
                    v.name,
                    "f16",
                    v.bm,
                    v.bn,
                    v.bk,
                    v.wm,
                    v.wn,
                    v.stages,
                    v.raster,
                    v.pad,
                    Act::None,
                    false,
                    false,
                    v.swz,
                    v.min_ctas,
                    v.store,
                    budget,
                );
            }
        }
        m
    })
    .as_str()
}

/// Emit the cliff candidate PTX module (separate from `wmma_f16_ptx` so experiments never perturb the
/// dispatched module / its cubin cache). Each [`CliffCfg`] becomes one `mma.sync` entry.
pub fn gemm_cliff_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        for v in CLIFF_VARIANTS {
            m += &entry_mma_pipe(
                v.name,
                "f16",
                v.bm,
                v.bn,
                v.bk,
                v.wm,
                v.wn,
                v.stages,
                v.raster,
                v.pad,
                Act::None,
                false,
                false,
                v.swz,
                v.min_ctas,
                v.store,
            );
        }
        m
    })
    .as_str()
}

/// Generate a shared-memory-staged WMMA GEMM entry computing `C = A·Bᵀ`. `ty` is "f16" or "bf16"; the
/// CTA stages a `bm×SM_BK` tile of A and a `bn×SM_BK` tile of B, with a `warps_m×warps_n` warp grid
/// each owning a `(bm/warps_m)×(bn/warps_n)` sub-tile. `bm`,`bn` must be 16-multiples and the staging
/// requires `bm·SM_BK` and `bn·SM_BK` to be whole multiples of `threads·8` (8 f16 per vectorized load).
fn entry_smem(
    name: &str,
    ty: &str,
    bm: usize,
    bn: usize,
    warps_m: usize,
    warps_n: usize,
    static_dims: Option<(usize, usize, usize)>,
) -> String {
    if let Some((m, n, k)) = static_dims {
        assert!(
            m % bm == 0 && n % bn == 0 && k % SM_BK == 0,
            "{name}: static dims must tile the kernel"
        );
    }
    let mma_ty = if ty == "f16" {
        "f32.f32".to_string()
    } else {
        format!("f32.{ty}.{ty}.f32")
    };
    let nab = if ty == "f16" { 8 } else { 4 };
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m); // 16×16 tiles per warp, M direction
    let tn = bn / (16 * warps_n); // 16×16 tiles per warp, N direction
    let smem_a = bm * SM_BK * 2; // bytes
    let smem_b = bn * SM_BK * 2;
    // 128-bit (8×f16) vectorized global→shared chunks per thread. Alignment holds: each chunk's
    // global index is K(·16)-aligned + kt(·16) + {0,8} ⇒ a multiple of 8 ⇒ 16-byte aligned.
    let a_chunks = bm * SM_BK / (threads * 8);
    let b_chunks = bn * SM_BK / (threads * 8);
    // The whole-multiple staging constraint stated above must be CHECKED, not assumed: the division
    // truncates, so a tile that does not tile the CTA emits a kernel that stages only part of A/B (or,
    // at `*_chunks == 0`, nothing at all) while every `bar.sync`/`wmma.load`/`wmma.mma` stays
    // well-formed — it JITs cleanly and computes C from stale shared memory. `entry_mma_pipe_budget`
    // (the `mma.sync` pipeline every `CliffCfg` row flows through, including the widened 128x256 /
    // 256x128 tiles) now carries the same whole-multiple guard. LANDMINE: the two remaining siblings
    // (`entry_smem_pipe`, `entry_mma_gate`) still assert only the `>= 1` half, so a partial-multiple
    // tile is generatable there; every dispatched config is an exact multiple, but a new row is not
    // checked for it.
    assert!(
        a_chunks >= 1 && a_chunks * threads * 8 == bm * SM_BK,
        "{name}: A tile {bm}x{SM_BK} is not a whole multiple of threads*8 = {} (128-bit vectorized staging)",
        threads * 8
    );
    assert!(
        b_chunks >= 1 && b_chunks * threads * 8 == bn * SM_BK,
        "{name}: B tile {bn}x{SM_BK} is not a whole multiple of threads*8 = {} (128-bit vectorized staging)",
        threads * 8
    );
    let wn_shift = warps_n.trailing_zeros(); // warpId / warps_n  (warps_n a power of two)
    let wm = (16 * tm) as i64; // per-warp rows owned
    let wn = (16 * tn) as i64; // per-warp cols owned

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
    s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
    s += "    .reg .pred %p0;\n";
    // NB: the linear thread id reg is %tix, NOT %tid — %tid is the PTX special register (threadIdx),
    // so a user reg named %tid makes the assembler read `%tid.x` as a video selector and reject it.
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%ldm,%v0,%v1,%v2,%v3;\n";
    let mut decl_c = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                decl_c += &format!("%c{ti}_{tj}_{r},");
            }
        }
    }
    s += &format!("    .reg .f32 {};\n", decl_c.trim_end_matches(','));
    let mut decl_ab = String::new();
    for ti in 0..tm {
        for r in 0..nab {
            decl_ab += &format!("%a{ti}_{r},");
        }
    }
    for tj in 0..tn {
        for r in 0..nab {
            decl_ab += &format!("%b{tj}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%B,%C,%off,%gp,%gptr,%cptr;\n";

    // Static-shape specialization (Wukong's compile-time-shapes lever): bake M/N/K as constants so
    // ptxas constant-folds/strength-reduces the hot-loop strides (`×K`/`×N` → shifts for power-of-two
    // dims) and knows the K-loop trip count. Dynamic loads from params. Identical signature ⇒ the
    // launcher is unchanged; identical arithmetic ⇒ bit-exact vs the dynamic kernel.
    match static_dims {
        Some((m, n, k)) => {
            s += &format!("    mov.u32 %M,{m};\n    mov.u32 %N,{n};\n    mov.u32 %K,{k};\n");
        }
        None => {
            s +=
                "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
        }
    }
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    s += "    mov.u32 %ldm,16;\n"; // SMEM tile leading dim (BK); wmma.load/.store want a reg stride
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n";
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n");
    s += &format!("    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                s += &format!("    mov.f32 %c{ti}_{tj}_{r},0f00000000;\n");
            }
        }
    }

    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    // Cooperative global→shared staging. element e in the BM×BK (resp BN×BK) tile: r=e/BK, c=e%BK; the
    // shared byte offset is just e·2 because r·BK+c == e (BK==16).
    let stage = |g_base: &str, gptr: &str, smem: String, chunks: usize, s: &mut String| {
        for li in 0..chunks {
            // chunk e (each = 8 contiguous f16); flat element = e·8, so row r=e/2, col0=(e&1)·8.
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += "    shr.u32 %r,%e,1;\n    and.b32 %c,%e,1;\n    shl.b32 %c,%c,3;\n";
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kt;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,{gptr},%off;\n");
            *s += "    ld.global.v4.u32 {%v0,%v1,%v2,%v3},[%gptr];\n";
            // shared dest byte = flat·2 = e·16.
            *s += &format!(
                "    mov.u32 %tmp,{smem};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n"
            );
            *s += "    st.shared.v4.u32 [%tmp],{%v0,%v1,%v2,%v3};\n";
        }
    };
    stage("%baseRow", "%A", format!("smemA_{name}"), a_chunks, &mut s);
    stage("%baseCol", "%B", format!("smemB_{name}"), b_chunks, &mut s);
    s += "    bar.sync 0;\n";

    // Each warp loads its fragments from shared (generic addr via cvta.shared) and accumulates.
    for ti in 0..tm {
        s += &format!("    mov.u32 %tmp,smemA_{name};\n");
        s += &format!(
            "    mul.lo.s32 %tmp2,%warpRow,{wm};\n    add.u32 %tmp2,%tmp2,{};\n",
            ti * 16
        );
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let ra = veclist(&format!("a{ti}_"), nab);
        s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%gp], %ldm;\n");
    }
    for tj in 0..tn {
        s += &format!("    mov.u32 %tmp,smemB_{name};\n");
        s += &format!(
            "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
            tj * 16
        );
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let rb = veclist(&format!("b{tj}_"), nab);
        s += &format!("    wmma.load.b.sync.aligned.m16n16k16.col.{ty} {rb}, [%gp], %ldm;\n");
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"), nab);
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"), nab);
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!(
                "    wmma.mma.sync.aligned.row.col.m16n16k16.{mma_ty} {cc}, {ra}, {rb}, {cc};\n"
            );
        }
    }
    s += "    bar.sync 0;\n";
    s += &format!("    add.u32 %kt,%kt,16;\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    for ti in 0..tm {
        for tj in 0..tn {
            // row = baseRow + warpRow·wm + ti·16 ; col = baseCol + warpCol·wn + tj·16
            s += &format!(
                "    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n",
                ti * 16
            );
            s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
            s += &format!(
                "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
                tj * 16
            );
            s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
            s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n");
        }
    }
    s += "    ret;\n}\n";
    s
}

/// Fused activation epilogue applied to the f32 accumulator **before** the C store — the thing cuBLAS
/// structurally cannot do (it only computes `A·Bᵀ`; an activation needs a second kernel that
/// round-trips C through HBM). The activation is elementwise on each accumulator register, so it needs
/// no knowledge of the WMMA fragment's row/col layout.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
    Silu,
    Gelu,
}

impl Act {
    /// PTX applying the activation in place to one f32 accumulator register `reg`. Transcendental
    /// activations use the Ada SFU fast paths (`ex2.approx`/`tanh.approx`/`rcp.approx`) and the **exact
    /// same formulas + constants** as the standalone `ptx::vmath_ptx` kernels, so a fused `silu(A·Bᵀ)`
    /// equals the unfused `silu(gemm)` and inherits its tolerance gate. Scratch lives in `%act0`/`%act1`
    /// (declared by `entry_smem_db`); each accumulator is processed sequentially so the scratch reuses.
    /// `pub(crate)` so the fp8 mma generator (`ptx_fp8::fp8_pipe_entry`) reuses the identical epilogue.
    pub(crate) fn epilogue(self, reg: &str) -> String {
        let hexf = |x: f32| format!("0f{:08X}", x.to_bits());
        match self {
            Act::None => String::new(),
            Act::Relu => format!("    max.f32 {reg},{reg},0f00000000;\n"),
            // silu(x) = x·sigmoid(x) = x / (1 + exp(-x)), exp via 2^(x·log2e).
            Act::Silu => {
                let (nlog2e, one) = (hexf(-std::f32::consts::LOG2_E), hexf(1.0));
                format!(
                    "    mul.f32 %act0,{reg},{nlog2e};\n    ex2.approx.f32 %act0,%act0;\n    \
                     add.f32 %act0,%act0,{one};\n    rcp.approx.f32 %act0,%act0;\n    \
                     mul.f32 {reg},{reg},%act0;\n"
                )
            }
            // gelu(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
            Act::Gelu => {
                let c0 = hexf((2.0f32 / std::f32::consts::PI).sqrt());
                let (c1, one, half) = (hexf(0.044715), hexf(1.0), hexf(0.5));
                format!(
                    "    mul.f32 %act0,{reg},{reg};\n    mul.f32 %act0,%act0,{reg};\n    \
                     fma.rn.f32 %act0,%act0,{c1},{reg};\n    mul.f32 %act0,%act0,{c0};\n    \
                     tanh.approx.f32 %act0,%act0;\n    add.f32 %act0,%act0,{one};\n    \
                     mul.f32 %act1,{reg},{half};\n    mul.f32 {reg},%act0,%act1;\n"
                )
            }
        }
    }
}

/// Generate a **`cp.async` double-buffered** SMEM-staged WMMA GEMM (`_sm_db`). Same CTA tiling as
/// [`entry_smem`], but the K-loop is software-pipelined: each step issues `cp.async` copies that
/// prefetch the *next* A/B tile into the alternate shared buffer **while the tensor cores consume the
/// current one**, then `cp.async.wait_group 1` only blocks on the older (current) copy. This overlaps
/// global-load latency with compute — the lever for the large-GEMM cliff, which the 128×128 experiment
/// showed is latency- not bandwidth-*volume*-bound (cuBLAS hides the same latency with a multi-stage
/// pipeline). Two shared buffers toggle by XOR (the tile size is a power of two). Requires `bm==bn`
/// (one buffer-offset register drives both A and B) and the [`entry_smem`] staging constraints. `act`
/// fuses an activation into the C-store epilogue (`Act::None` is the plain GEMM).
fn entry_smem_db(
    name: &str,
    ty: &str,
    bm: usize,
    bn: usize,
    warps_m: usize,
    warps_n: usize,
    act: Act,
    bias: bool,
    residual: bool,
) -> String {
    assert_eq!(
        bm, bn,
        "the double-buffered kernel uses one buffer-offset reg for A and B"
    );
    let mma_ty = if ty == "f16" {
        "f32.f32".to_string()
    } else {
        format!("f32.{ty}.{ty}.f32")
    };
    let nab = if ty == "f16" { 8 } else { 4 };
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m);
    let tn = bn / (16 * warps_n);
    let tile_bytes = bm * SM_BK * 2; // one A (== one B) tile; a power of two ⇒ XOR toggles the buffer
    debug_assert!(tile_bytes.is_power_of_two());
    if bias {
        // The fused-bias epilogue reuses smemA as a per-warp 16×16 f32 store-back scratch (one tile at a
        // time), so the 2·tile_bytes smemA allocation must hold num_warps disjoint 256-f32 (1 KiB) slots.
        assert!(
            2 * tile_bytes >= warps_m * warps_n * 16 * 16 * 4,
            "fused-bias epilogue scratch (smemA reuse) too small for the warp grid"
        );
    }
    if residual {
        // The residual fuses by *seeding* the f32 accumulator with `residual[tile]` (wmma.load.c) and
        // letting the K-loop add A·Bᵀ on top → out = residual + A·Bᵀ. A post-accumulate bias/activation
        // would then wrongly act on the residual too, so residual is offered only for the plain GEMM.
        assert!(
            matches!(act, Act::None) && !bias,
            "residual epilogue is incompatible with a fused bias/activation"
        );
    }
    let a_chunks = bm * SM_BK / (threads * 8);
    let b_chunks = bn * SM_BK / (threads * 8);
    // Same unchecked truncating division as [`entry_smem`] — a tile that does not tile the CTA emits a
    // kernel whose double-buffered staging is partial (or empty), which JITs and reads stale SMEM.
    assert!(
        a_chunks >= 1 && a_chunks * threads * 8 == bm * SM_BK,
        "{name}: A tile {bm}x{SM_BK} is not a whole multiple of threads*8 = {} (128-bit vectorized staging)",
        threads * 8
    );
    assert!(
        b_chunks >= 1 && b_chunks * threads * 8 == bn * SM_BK,
        "{name}: B tile {bn}x{SM_BK} is not a whole multiple of threads*8 = {} (128-bit vectorized staging)",
        threads * 8
    );
    let wn_shift = warps_n.trailing_zeros();
    let wm = (16 * tm) as i64;
    let wn = (16 * tn) as i64;

    // The fused-bias variant takes an extra `bias[N]` (f32) param read in the store-back epilogue; the
    // residual variant takes a `residual[M,N]` (f32) param used to seed the accumulator (wmma.load.c).
    let bias_param = if bias { ",\n    .param .u64 pBias" } else { "" };
    let resid_param = if residual {
        ",\n    .param .u64 pResidual"
    } else {
        ""
    };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n{{\n"
    );
    s += &format!(
        "    .shared .align 16 .b8 smemA_{name}[{}];\n",
        2 * tile_bytes
    );
    s += &format!(
        "    .shared .align 16 .b8 smemB_{name}[{}];\n",
        2 * tile_bytes
    );
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%ktn,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%ldm,%bufc,%bufp,%v0,%v1,%v2,%v3;\n";
    let mut decl_c = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                decl_c += &format!("%c{ti}_{tj}_{r},");
            }
        }
    }
    // `%act0`/`%act1` are scratch for a fused transcendental epilogue (Act::Silu/Gelu); unused (and
    // dropped by ptxas) for Act::None/Relu.
    s += &format!(
        "    .reg .f32 {},%act0,%act1;\n",
        decl_c.trim_end_matches(',')
    );
    if bias {
        // Fused-bias store-back epilogue scratch: %lane (lane id), %f (flat 0..256 elem), %grow/%gcol
        // (this element's global row/col), %scbase (this warp's SMEM scratch base), %bval/%biasv (the
        // accumulator value + its bias), %Bias (bias base ptr), %scptr (scratch generic / bias ptr).
        s += "    .reg .b32 %lane,%f,%grow,%gcol,%scbase;\n";
        s += "    .reg .f32 %bval,%biasv;\n";
        s += "    .reg .b64 %Bias,%scptr;\n";
    }
    if residual {
        s += "    .reg .b64 %Resid;\n"; // residual[M,N] base pointer (seeds the accumulator)
    }
    let mut decl_ab = String::new();
    for ti in 0..tm {
        for r in 0..nab {
            decl_ab += &format!("%a{ti}_{r},");
        }
    }
    for tj in 0..tn {
        for r in 0..nab {
            decl_ab += &format!("%b{tj}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%B,%C,%off,%gp,%gptr,%cptr;\n";

    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    s += "    mov.u32 %ldm,16;\n";
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n";
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n");
    s += &format!("    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    if residual {
        // Residual fusion: seed each accumulator from residual[tile] with wmma.load.c — the SAME
        // fragment layout the final wmma.store.d uses, so the opaque (lane,reg)→(row,col) map cancels.
        // The K-loop then adds A·Bᵀ on top → out = residual + A·Bᵀ at f32 accumulate, no HBM round-trip
        // for the residual (cuBLAS needs a beta=1 pre-fill or a separate add kernel). Stride = N.
        s += "    ld.param.u64 %Resid,[pResidual];\n    cvta.to.global.u64 %Resid,%Resid;\n";
        for ti in 0..tm {
            for tj in 0..tn {
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!(
                    "    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n",
                    ti * 16
                );
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!(
                    "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
                    tj * 16
                );
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%Resid,%off;\n";
                s +=
                    &format!("    wmma.load.c.sync.aligned.m16n16k16.row.f32 {cc}, [%cptr], %N;\n");
            }
        }
    } else {
        for ti in 0..tm {
            for tj in 0..tn {
                for r in 0..8 {
                    s += &format!("    mov.f32 %c{ti}_{tj}_{r},0f00000000;\n");
                }
            }
        }
    }
    s += "    mov.u32 %bufc,0;\n";
    s += &format!("    mov.u32 %bufp,{tile_bytes};\n");

    // Issue cp.async copies staging the column-`%kcol` A/B tile into the buffer at byte offset `bufoff`.
    // Identical addressing to entry_smem's `stage`, but with cp.async (global→shared, no register hop)
    // and the staged column read from %kcol (so the prefetch can target the *next* tile while %kt lags).
    let stage = |g_base: &str,
                 gbase_ptr: &str,
                 smem: String,
                 bufoff: &str,
                 chunks: usize,
                 s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += "    shr.u32 %r,%e,1;\n    and.b32 %c,%e,1;\n    shl.b32 %c,%c,3;\n";
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n");
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    // Prologue: prefetch tile 0 into buffer 0 and commit it as the first async group.
    s += "    mov.u32 %kcol,0;\n";
    stage(
        "%baseRow",
        "%A",
        format!("smemA_{name}"),
        "%bufc",
        a_chunks,
        &mut s,
    );
    stage(
        "%baseCol",
        "%B",
        format!("smemB_{name}"),
        "%bufc",
        b_chunks,
        &mut s,
    );
    s += "    cp.async.commit_group;\n";

    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    // Prefetch the next tile into the alternate buffer (if any), then wait on the *current* tile only.
    s += "    add.u32 %ktn,%kt,16;\n    setp.lt.u32 %pmore,%ktn,%K;\n";
    s += &format!("    @!%pmore bra LAST_{name};\n");
    s += "    mov.u32 %kcol,%ktn;\n";
    stage(
        "%baseRow",
        "%A",
        format!("smemA_{name}"),
        "%bufp",
        a_chunks,
        &mut s,
    );
    stage(
        "%baseCol",
        "%B",
        format!("smemB_{name}"),
        "%bufp",
        b_chunks,
        &mut s,
    );
    s += "    cp.async.commit_group;\n    cp.async.wait_group 1;\n";
    s += &format!("    bra SYNC_{name};\nLAST_{name}:\n    cp.async.wait_group 0;\nSYNC_{name}:\n");
    s += "    bar.sync 0;\n";
    // Compute the current tile from buffer `%bufc`.
    for ti in 0..tm {
        s += &format!("    mov.u32 %tmp,smemA_{name};\n    add.u32 %tmp,%tmp,%bufc;\n");
        s += &format!(
            "    mul.lo.s32 %tmp2,%warpRow,{wm};\n    add.u32 %tmp2,%tmp2,{};\n",
            ti * 16
        );
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let ra = veclist(&format!("a{ti}_"), nab);
        s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%gp], %ldm;\n");
    }
    for tj in 0..tn {
        s += &format!("    mov.u32 %tmp,smemB_{name};\n    add.u32 %tmp,%tmp,%bufc;\n");
        s += &format!(
            "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
            tj * 16
        );
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let rb = veclist(&format!("b{tj}_"), nab);
        s += &format!("    wmma.load.b.sync.aligned.m16n16k16.col.{ty} {rb}, [%gp], %ldm;\n");
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"), nab);
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"), nab);
            let cc = veclist(&format!("c{ti}_{tj}_"), 8);
            s += &format!(
                "    wmma.mma.sync.aligned.row.col.m16n16k16.{mma_ty} {cc}, {ra}, {rb}, {cc};\n"
            );
        }
    }
    s += "    bar.sync 0;\n"; // all warps done reading %bufc before a later step overwrites it
    s += &format!("    xor.b32 %bufc,%bufc,{tile_bytes};\n    xor.b32 %bufp,%bufp,{tile_bytes};\n");
    s += &format!("    add.u32 %kt,%kt,16;\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    if bias {
        // Fused **bias + activation** epilogue computing `C = act(A·Bᵀ + bias)`. WMMA's f32 fragment →
        // (row,col) mapping is opaque/architecture-defined, so we cannot add a per-column bias to the
        // accumulator *registers* directly (as the elementwise activations do). Instead each warp
        // `wmma.store.d`s its tile into a private 16×16 f32 SMEM scratch (canonical row-major), then
        // every lane re-reads its 8 elements by explicit (row,col), adds `bias[globalCol]`, applies the
        // activation (after the bias, the canonical `act(x·Wᵀ+bias)` order), and writes C. smemA is free
        // here — the K-loop's last `bar.sync` drained the staging — so it is reused as `num_warps`
        // disjoint 1 KiB (256-f32) slots, one per warp.
        s += "    bar.sync 0;\n"; // smemA staging fully consumed before we repurpose it as scratch
        s += "    ld.param.u64 %Bias,[pBias];\n    cvta.to.global.u64 %Bias,%Bias;\n";
        s += "    and.b32 %lane,%tix,31;\n";
        // This warp's scratch slot: smemA + warpId·1024 bytes (1024 = 16·16·4, one f32 tile).
        s += &format!("    mov.u32 %scbase,smemA_{name};\n    mul.lo.s32 %tmp,%warpId,1024;\n    add.u32 %scbase,%scbase,%tmp;\n");
        for ti in 0..tm {
            for tj in 0..tn {
                // Store this 16×16 tile's accumulator to the warp's scratch (row-major, leading dim 16).
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += "    cvt.u64.u32 %scptr,%scbase;\n    cvta.shared.u64 %scptr,%scptr;\n";
                s += &format!(
                    "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%scptr], {cc}, %ldm;\n"
                );
                s += "    bar.sync 0;\n"; // tile fully written to scratch before any lane reads it
                                          // Each lane owns 8 of the 256 elements: flat f = lane + u·32 ⇒ (row f/16, col f%16).
                for u in 0..8 {
                    if u == 0 {
                        s += "    mov.u32 %f,%lane;\n";
                    } else {
                        s += &format!("    add.u32 %f,%lane,{};\n", u * 32);
                    }
                    s += "    shr.u32 %r,%f,4;\n    and.b32 %c,%f,15;\n";
                    // Read scratch[f] (byte offset = scbase + f·4).
                    s += "    shl.b32 %tmp,%f,2;\n    add.u32 %tmp,%tmp,%scbase;\n    ld.shared.f32 %bval,[%tmp];\n";
                    // globalRow = baseRow + warpRow·wm + ti·16 + r
                    s += &format!("    mul.lo.s32 %grow,%warpRow,{wm};\n    add.u32 %grow,%grow,{};\n    add.u32 %grow,%grow,%baseRow;\n    add.u32 %grow,%grow,%r;\n", ti * 16);
                    // globalCol = baseCol + warpCol·wn + tj·16 + c
                    s += &format!("    mul.lo.s32 %gcol,%warpCol,{wn};\n    add.u32 %gcol,%gcol,{};\n    add.u32 %gcol,%gcol,%baseCol;\n    add.u32 %gcol,%gcol,%c;\n", tj * 16);
                    // Add bias[globalCol] (f32), then activate, then store to C[globalRow·N+globalCol].
                    s += "    mul.wide.u32 %off,%gcol,4;\n    add.s64 %scptr,%Bias,%off;\n    ld.global.f32 %biasv,[%scptr];\n    add.f32 %bval,%bval,%biasv;\n";
                    s += &act.epilogue("%bval");
                    s += "    mul.lo.s32 %tmp,%grow,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n    st.global.f32 [%cptr],%bval;\n";
                }
                s += "    bar.sync 0;\n"; // all lanes done reading scratch before the next tile overwrites it
            }
        }
    } else {
        for ti in 0..tm {
            for tj in 0..tn {
                s += &format!(
                    "    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n",
                    ti * 16
                );
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!(
                    "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
                    tj * 16
                );
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
                // Fused epilogue: activate each accumulator register in place before storing C.
                for r in 0..8 {
                    s += &act.epilogue(&format!("%c{ti}_{tj}_{r}"));
                }
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!(
                    "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n"
                );
            }
        }
    }
    s += "    ret;\n}\n";
    s
}

/// Generate an **N-stage `cp.async` software-pipelined** SMEM-staged WMMA GEMM (`_pipe`). This is the
/// large-GEMM-cliff lever. [`entry_smem_db`] is a *2-stage, BK=16* pipeline: it prefetches exactly one
/// 16-wide K-slice ahead, i.e. ~16 K-values, far short of the ~400–800-cycle HBM latency, so at
/// L2-spilling sizes (4096³ ≈ 67 MB ≫ 24 MB L2) the tensor cores starve on global loads and the GEMM
/// collapses to ~30% of cuBLAS. cuBLAS hides that latency with a *deeper* pipeline; this generalizes the
/// recipe along two axes:
///   * **`stages` SMEM buffers (depth ≥ 2):** the K-loop keeps `stages-1` `cp.async` groups in flight,
///     so each tile's global load is launched `stages-1` iterations before it is consumed — a prefetch
///     window deep enough to cover HBM latency. `cp.async.wait_group stages-2` gates on the oldest group.
///   * **`bk` (staged K-tile width, a 16-multiple):** one staged tile feeds `bk/16` WMMA k-steps, so a
///     wider `bk` amortizes the per-tile barrier and load-issue overhead over more tensor-core work.
///
/// **One `bar.sync` per K-iteration** (vs the 2-stage kernel's two): placed right after `wait_group`, it
/// fences both the just-arrived buffer (RAW: producers→consumers) and the about-to-be-overwritten buffer
/// (WAR: this iteration's consumers of buffer `b` finish before iteration `j+1`'s `cp.async` rewrites
/// `b`). The prefetch is *issued after* that barrier and *before* the MMAs, so the async copy flies under
/// the tensor cores. A/B keep independent buffer-offset registers, so `bm != bn` is allowed (unlike the
/// db kernel). Plain GEMM, no fused epilogue — the isolated-vs-cuBLAS path; fusion stays on `_sm_db`.
///
/// Requires `M%bm==0`, `N%bn==0`, `K%bk==0`, `bk%16==0`, `bk/8` a power of two, and `bm·bk`, `bn·bk` each
/// a multiple of `threads·8` (128-bit vectorized staging). Total static SMEM `stages·(bm+bn)·bk·2` must
/// be ≤ 48 KiB (the static-shared cap; beyond it needs dynamic shared + the max-dyn-smem attribute).
fn entry_smem_pipe(
    name: &str,
    ty: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
    act: Act,
    bias: bool,
    residual: bool,
) -> String {
    assert!(
        stages >= 2,
        "the pipeline needs at least 2 stages (1 prefetch in flight)"
    );
    assert!(
        bk.is_multiple_of(16),
        "bk must be a multiple of the WMMA k16 step"
    );
    assert!(
        (bk / 8).is_power_of_two(),
        "bk/8 must be a power of two (shift-based staging address math)"
    );
    assert!(
        raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()),
        "{name}: rasterization needs bm,bn powers of two (tiles_m/n via shift)"
    );
    let mma_ty = if ty == "f16" {
        "f32.f32".to_string()
    } else {
        format!("f32.{ty}.{ty}.f32")
    };
    let nab = if ty == "f16" { 8 } else { 4 };
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m);
    let tn = bn / (16 * warps_n);
    let nks = bk / 16; // WMMA k-steps fed by one staged tile
    let tile_a = bm * bk * 2; // bytes per A buffer
    let tile_b = bn * bk * 2; // bytes per B buffer
    let smem_a = stages * tile_a;
    let smem_b = stages * tile_b;
    assert!(
        smem_a + smem_b <= 48 * 1024,
        "{name}: static SMEM {} B (stages={stages} bk={bk} {bm}x{bn}) exceeds the 48 KiB cap",
        smem_a + smem_b
    );
    if residual {
        // The residual *seeds* the f32 accumulator (`out = residual + A·Bᵀ`, via wmma.load.c), so a
        // post-accumulate activation would wrongly act on the residual too — residual ⇒ no activation.
        // A per-column `bias` (added in the SMEM-scratch epilogue, post-accumulate, pre-store) IS allowed:
        // `out = (residual + A·Bᵀ) + bias` is exactly the transformer down-proj / attention output-proj.
        assert!(
            matches!(act, Act::None),
            "{name}: residual epilogue cannot also activate (act applies to A·Bᵀ+bias, not the residual)"
        );
    }
    if bias {
        // The fused-bias epilogue repurposes `smemA` (drained after the K-loop) as `num_warps` disjoint
        // 16×16 f32 (1 KiB) store-back scratch slots — `smem_a` must hold them all (the deep pipe's
        // multi-stage smemA is ≫ the 4–8 KiB needed, but assert it so a future shrink can't silently clobber).
        assert!(
            smem_a >= warps_m * warps_n * 16 * 16 * 4,
            "{name}: fused-bias store-back scratch (smemA reuse) too small for the warp grid"
        );
    }
    let a_chunks = bm * bk / (threads * 8);
    let b_chunks = bn * bk / (threads * 8);
    assert!(
        a_chunks >= 1 && b_chunks >= 1,
        "{name}: tile too small for one 128-bit chunk per thread"
    );
    let bk_chunks = bk / 8; // 8-element (128-bit) chunks per staged row
    let row_shift = bk_chunks.trailing_zeros(); // flat-chunk e → row = e >> row_shift
    let col_mask = bk_chunks - 1; //              col8 = (e & col_mask) << 3
    let wn_shift = warps_n.trailing_zeros();
    let wm = (16 * tm) as i64;
    let wn = (16 * tn) as i64;

    // The fused-bias variant takes an extra `bias[N]` (f32) param read in the SMEM store-back epilogue;
    // the residual variant takes a `residual[M,N]` (f32) used to seed the accumulator via wmma.load.c.
    let bias_param = if bias { ",\n    .param .u64 pBias" } else { "" };
    let resid_param = if residual {
        ",\n    .param .u64 pResidual"
    } else {
        ""
    };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
    s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%ldm,%bufcA,%bufcB,%bufwA,%bufwB;\n";
    if bias {
        // Store-back epilogue scratch (mirrors `entry_smem_db`): %lane, %f (flat 0..256 elem), %grow/%gcol
        // (this element's global row/col), %scbase (this warp's SMEM scratch base), %scld (=16, the scratch
        // tile leading dim — distinct from the pipe's %ldm=bk), %bval/%biasv, %Bias / %scptr (bias/scratch ptr).
        s += "    .reg .b32 %lane,%f,%grow,%gcol,%scbase,%scld;\n";
        s += "    .reg .f32 %bval,%biasv;\n";
        s += "    .reg .b64 %Bias,%scptr;\n";
    }
    if residual {
        s += "    .reg .b64 %Resid;\n"; // residual[M,N] base pointer (seeds the accumulator via wmma.load.c)
    }
    if raster > 0 {
        s += "    .reg .b32 %lin,%tn,%tm,%gsz,%grp,%rem,%col0,%gw,%trow,%tcol;\n";
    }
    let mut decl_c = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..8 {
                decl_c += &format!("%c{ti}_{tj}_{r},");
            }
        }
    }
    // `%act0`/`%act1` are scratch for a fused transcendental epilogue (Act::Silu/Gelu, applied in the
    // bias store-back); dropped by ptxas when unused (Act::None/Relu, or no bias).
    if bias {
        s += &format!(
            "    .reg .f32 {},%act0,%act1;\n",
            decl_c.trim_end_matches(',')
        );
    } else {
        s += &format!("    .reg .f32 {};\n", decl_c.trim_end_matches(','));
    }
    let mut decl_ab = String::new();
    for ti in 0..tm {
        for r in 0..nab {
            decl_ab += &format!("%a{ti}_{r},");
        }
    }
    for tj in 0..tn {
        for r in 0..nab {
            decl_ab += &format!("%b{tj}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%B,%C,%off,%gp,%gptr,%cptr;\n";

    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    if raster == 0 {
        s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
        s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    } else {
        // Threadblock rasterization (1-D grid): remap the linear block id into a column-band-of-`raster`
        // tile order. tiles_n = N/bn, tiles_m = M/bm (bm,bn powers of two ⇒ shifts). Within a band of
        // `raster` N-tile columns, sweep all M-tile rows before the next band, so the CTAs co-resident on
        // the SMs touch a compact `raster·bn`-wide A/B footprint that stays hot in L2. Edge bands narrower
        // than `raster` are handled by the runtime group width `gw` (one div/rem per CTA, in the prologue).
        let (bn_sh, bm_sh) = (bn.trailing_zeros(), bm.trailing_zeros());
        s += &format!("    mov.u32 %lin,%ctaid.x;\n    shr.u32 %tn,%N,{bn_sh};\n    shr.u32 %tm,%M,{bm_sh};\n");
        s += &format!("    mul.lo.s32 %gsz,%tm,{raster};\n    div.u32 %grp,%lin,%gsz;\n    rem.u32 %rem,%lin,%gsz;\n");
        s += &format!("    mul.lo.s32 %col0,%grp,{raster};\n    sub.u32 %gw,%tn,%col0;\n    min.u32 %gw,%gw,{raster};\n");
        s += "    div.u32 %trow,%rem,%gw;\n    rem.u32 %tcol,%rem,%gw;\n    add.u32 %tcol,%tcol,%col0;\n";
        s += &format!("    mul.lo.s32 %baseRow,%trow,{bm};\n    mul.lo.s32 %baseCol,%tcol,{bn};\n");
    }
    s += &format!("    mov.u32 %ldm,{bk};\n"); // SMEM tile leading dim = bk (wmma stride operand)
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n";
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n");
    s += &format!("    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    if residual {
        // Seed each accumulator from `residual[tile]` with wmma.load.c — the SAME opaque fragment layout
        // the final wmma.store.d uses, so the (lane,reg)→(row,col) map cancels and the K-loop adds A·Bᵀ on
        // top → out = residual + A·Bᵀ at f32 accumulate, no HBM round-trip for the residual. Stride = N.
        s += "    ld.param.u64 %Resid,[pResidual];\n    cvta.to.global.u64 %Resid,%Resid;\n";
        for ti in 0..tm {
            for tj in 0..tn {
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!(
                    "    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n",
                    ti * 16
                );
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!(
                    "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
                    tj * 16
                );
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%Resid,%off;\n";
                s +=
                    &format!("    wmma.load.c.sync.aligned.m16n16k16.row.f32 {cc}, [%cptr], %N;\n");
            }
        }
    } else {
        for ti in 0..tm {
            for tj in 0..tn {
                for r in 0..8 {
                    s += &format!("    mov.f32 %c{ti}_{tj}_{r},0f00000000;\n");
                }
            }
        }
    }

    // cp.async staging of one BM×BK (or BN×BK) tile at K-column `%kcol` into the buffer at byte offset
    // `bufoff` within `smem`. Each thread copies `chunks` 128-bit (8×f16) groups: flat chunk e=tix+li·T,
    // row r=e>>row_shift, col8=(e&col_mask)<<3 (BK row-major), so SMEM dest byte = bufoff + e·16 and the
    // global element is (g_base+r)·K + kcol + col8. 16-byte aligned (every term is an 8-multiple).
    let stage = |g_base: &str,
                 gbase_ptr: &str,
                 smem: &str,
                 bufoff: &str,
                 chunks: usize,
                 s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,3;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n");
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    // Prologue: prefetch tiles 0..stages-2 into buffers 0..stages-2 (one cp.async group each, committed
    // even when guarded off for tiny K so the wait_group accounting stays uniform).
    for st in 0..(stages - 1) {
        s += &format!("    mov.u32 %kcol,{};\n", st * bk);
        s += &format!(
            "    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n",
            st * tile_a,
            st * tile_b
        );
        s += &format!("    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra PRO_{name}_{st};\n");
        stage(
            "%baseRow",
            "%A",
            &format!("smemA_{name}"),
            "%bufwA",
            a_chunks,
            &mut s,
        );
        stage(
            "%baseCol",
            "%B",
            &format!("smemB_{name}"),
            "%bufwB",
            b_chunks,
            &mut s,
        );
        s += &format!("PRO_{name}_{st}:\n    cp.async.commit_group;\n");
    }

    // Compute buffer starts at offset 0; the write (prefetch) buffer trails by stages-1 buffers.
    s += "    mov.u32 %bufcA,0;\n    mov.u32 %bufcB,0;\n";
    s += &format!(
        "    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n",
        (stages - 1) * tile_a,
        (stages - 1) * tile_b
    );
    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    // Drain the oldest in-flight group (the tile we are about to read), then the single fence.
    s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 2);
    // Issue the prefetch for the tile (stages-1) ahead into the write buffer, if it exists; commit.
    s += &format!("    add.u32 %kcol,%kt,{};\n    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra NOPRE_{name};\n", (stages - 1) * bk);
    stage(
        "%baseRow",
        "%A",
        &format!("smemA_{name}"),
        "%bufwA",
        a_chunks,
        &mut s,
    );
    stage(
        "%baseCol",
        "%B",
        &format!("smemB_{name}"),
        "%bufwB",
        b_chunks,
        &mut s,
    );
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");
    // Compute the current tile from buffer `%bufc`: bk/16 WMMA k-steps, fragment reuse across tm×tn.
    for ks in 0..nks {
        for ti in 0..tm {
            s += &format!("    mov.u32 %tmp,smemA_{name};\n    add.u32 %tmp,%tmp,%bufcA;\n");
            s += &format!(
                "    mul.lo.s32 %tmp2,%warpRow,{};\n    add.u32 %tmp2,%tmp2,{};\n",
                wm * bk as i64,
                ti * 16 * bk
            );
            s += &format!("    add.u32 %tmp2,%tmp2,{};\n    shl.b32 %tmp2,%tmp2,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ks * 16);
            s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
            let ra = veclist(&format!("a{ti}_"), nab);
            s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%gp], %ldm;\n");
        }
        for tj in 0..tn {
            s += &format!("    mov.u32 %tmp,smemB_{name};\n    add.u32 %tmp,%tmp,%bufcB;\n");
            s += &format!(
                "    mul.lo.s32 %tmp2,%warpCol,{};\n    add.u32 %tmp2,%tmp2,{};\n",
                wn * bk as i64,
                tj * 16 * bk
            );
            s += &format!("    add.u32 %tmp2,%tmp2,{};\n    shl.b32 %tmp2,%tmp2,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ks * 16);
            s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
            let rb = veclist(&format!("b{tj}_"), nab);
            s += &format!("    wmma.load.b.sync.aligned.m16n16k16.col.{ty} {rb}, [%gp], %ldm;\n");
        }
        for ti in 0..tm {
            let ra = veclist(&format!("a{ti}_"), nab);
            for tj in 0..tn {
                let rb = veclist(&format!("b{tj}_"), nab);
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!(
                    "    wmma.mma.sync.aligned.row.col.m16n16k16.{mma_ty} {cc}, {ra}, {rb}, {cc};\n"
                );
            }
        }
    }
    // Advance compute + write buffers by one tile, wrapping the ring of `stages` buffers.
    s += &format!("    add.u32 %bufcA,%bufcA,{tile_a};\n    setp.ge.u32 %pmore,%bufcA,{smem_a};\n    @%pmore sub.u32 %bufcA,%bufcA,{smem_a};\n");
    s += &format!("    add.u32 %bufcB,%bufcB,{tile_b};\n    setp.ge.u32 %pmore,%bufcB,{smem_b};\n    @%pmore sub.u32 %bufcB,%bufcB,{smem_b};\n");
    s += &format!("    add.u32 %bufwA,%bufwA,{tile_a};\n    setp.ge.u32 %pmore,%bufwA,{smem_a};\n    @%pmore sub.u32 %bufwA,%bufwA,{smem_a};\n");
    s += &format!("    add.u32 %bufwB,%bufwB,{tile_b};\n    setp.ge.u32 %pmore,%bufwB,{smem_b};\n    @%pmore sub.u32 %bufwB,%bufwB,{smem_b};\n");
    s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    if bias {
        // Fused **bias (+ activation)** store-back epilogue (mirrors `entry_smem_db`'s): WMMA's f32
        // fragment→(row,col) map is opaque, so we cannot add a per-column bias to the accumulator
        // *registers* (as the hand-placed mma.sync kernel does). Instead each warp `wmma.store.d`s its
        // tile into a private 16×16 f32 SMEM scratch (canonical row-major), every lane re-reads its 8
        // elements by explicit (row,col), adds `bias[globalCol]`, applies the activation (post-bias, the
        // canonical `act(x·Wᵀ+bias)` order), and writes C. smemA is free here (the pipeline is drained
        // just below) so it is reused as `num_warps` disjoint 1 KiB (256-f32) scratch slots. For the
        // residual variant the accumulator already holds `residual + A·Bᵀ` and act is None, so this
        // computes `residual + A·Bᵀ + bias` — the transformer down-proj / attention output-proj.
        s += "    cp.async.wait_group 0;\n    bar.sync 0;\n"; // drain the pipe + fence before smemA reuse
        s += "    ld.param.u64 %Bias,[pBias];\n    cvta.to.global.u64 %Bias,%Bias;\n";
        s += "    mov.u32 %scld,16;\n"; // scratch tile leading dim (the pipe's %ldm holds bk, not 16)
        s += "    and.b32 %lane,%tix,31;\n";
        // This warp's scratch slot: smemA + warpId·1024 bytes (1024 = 16·16·4, one f32 tile).
        s += &format!("    mov.u32 %scbase,smemA_{name};\n    mul.lo.s32 %tmp,%warpId,1024;\n    add.u32 %scbase,%scbase,%tmp;\n");
        for ti in 0..tm {
            for tj in 0..tn {
                // Store this 16×16 tile's accumulator to the warp's scratch (row-major, leading dim 16).
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += "    cvt.u64.u32 %scptr,%scbase;\n    cvta.shared.u64 %scptr,%scptr;\n";
                s += &format!(
                    "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%scptr], {cc}, %scld;\n"
                );
                s += "    bar.sync 0;\n"; // tile fully written to scratch before any lane reads it
                                          // Each lane owns 8 of the 256 elements: flat f = lane + u·32 ⇒ (row f/16, col f%16).
                for u in 0..8 {
                    if u == 0 {
                        s += "    mov.u32 %f,%lane;\n";
                    } else {
                        s += &format!("    add.u32 %f,%lane,{};\n", u * 32);
                    }
                    s += "    shr.u32 %r,%f,4;\n    and.b32 %c,%f,15;\n";
                    // Read scratch[f] (byte offset = scbase + f·4).
                    s += "    shl.b32 %tmp,%f,2;\n    add.u32 %tmp,%tmp,%scbase;\n    ld.shared.f32 %bval,[%tmp];\n";
                    // globalRow = baseRow + warpRow·wm + ti·16 + r ; globalCol = baseCol + warpCol·wn + tj·16 + c
                    s += &format!("    mul.lo.s32 %grow,%warpRow,{wm};\n    add.u32 %grow,%grow,{};\n    add.u32 %grow,%grow,%baseRow;\n    add.u32 %grow,%grow,%r;\n", ti * 16);
                    s += &format!("    mul.lo.s32 %gcol,%warpCol,{wn};\n    add.u32 %gcol,%gcol,{};\n    add.u32 %gcol,%gcol,%baseCol;\n    add.u32 %gcol,%gcol,%c;\n", tj * 16);
                    // Add bias[globalCol] (f32), then activate, then store to C[globalRow·N + globalCol].
                    s += "    mul.wide.u32 %off,%gcol,4;\n    add.s64 %scptr,%Bias,%off;\n    ld.global.f32 %biasv,[%scptr];\n    add.f32 %bval,%bval,%biasv;\n";
                    s += &act.epilogue("%bval");
                    s += "    mul.lo.s32 %tmp,%grow,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n    st.global.f32 [%cptr],%bval;\n";
                }
                s += "    bar.sync 0;\n"; // all lanes done reading scratch before the next tile overwrites it
            }
        }
    } else {
        for ti in 0..tm {
            for tj in 0..tn {
                s += &format!(
                    "    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n",
                    ti * 16
                );
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!(
                    "    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n",
                    tj * 16
                );
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!(
                    "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n"
                );
            }
        }
    }
    s += "    ret;\n}\n";
    s
}

/// Epilogue global-store mode for the `mma.sync` D-fragments — **bit-identical output**, differing only
/// in the store *instruction*. Each lane's four f32 accumulators land as two adjacent-column pairs (d0,d1
/// at row `grp`; d2,d3 at row `grp+8`), so each pair is contiguous in C and folds into one vector store.
///   * `Scalar` — four `st.global.f32` (the historical epilogue).
///   * `V2`     — two `st.global.v2.f32`: half the store instructions / memory transactions.
///   * `V2Cs`   — two `st.global.cs.v2.f32`; the `.cs` (cache-streaming, evict-first) hint keeps the
///     write-once C stream from evicting the L2-resident A/B raster band it competes with at 4096³.
///
/// The pair base is 8-byte aligned (`gcol` is always even and the dispatched tiles have `N` a 128-multiple
/// ⇒ `(row·N+gcol)·4` is a multiple of 8), so the `v2.f32` stores are legal.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Store {
    Scalar,
    V2,
    V2Cs,
}

/// Generate an **`mma.sync.m16n8k16`** multi-stage `cp.async` GEMM (`_mma`) — the route past the WMMA
/// ceiling. Same CTA tiling, cp.async pipeline, and rasterization as [`entry_smem_pipe`], but the inner
/// tensor-core path uses the native Ada `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` with
/// **hand-placed SMEM fragment loads** instead of the opaque `wmma.load`/`wmma.mma`. The fragment
/// lane→register layout is the PTX-ISA standard (cross-checked against the proven flash kernel):
/// `groupID = lane/4`, `tid = lane%4`; per m16n8k16 tile a lane holds A = 4×b32 (rows {groupID,
/// groupID+8} × k-pairs {2tid, 2tid+8}), B = 2×b32 (row groupID × the same k-pairs), D = 4×f32
/// (rows {groupID, groupID+8} × cols {2tid, 2tid+1}). Loading fragments by explicit address (vs WMMA's
/// hidden pattern) is what lets a later XOR swizzle make the SMEM reads conflict-free.
///
/// Per-warp tile is `(bm/wm)×(bn/wn)` = `tm` m16-blocks × `tn` n8-blocks; accumulators are `tm·tn·4`
/// f32. Requires `M%bm==0`, `N%bn==0`, `K%bk==0`, `bk%16==0`, `bk/8` a power of two, `bm%(16·wm)==0`,
/// `bn%(8·wn)==0`, and the [`entry_smem_pipe`] staging constraints. Total static SMEM ≤ 48 KiB.
#[allow(clippy::too_many_arguments)]
fn entry_mma_pipe(
    name: &str,
    ty: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
    pad: usize,
    act: Act,
    bias: bool,
    residual: bool,
    swz: bool,
    min_ctas: usize,
    store: Store,
) -> String {
    // Every shipped entry spends at most the PTX ISA's static 48 KiB, so it takes the static emission
    // path and its PTX is the historical text to the byte. Only the deliberately-deep variants
    // ([`PIPE_DEEP_VARIANTS`]) pass a device budget and cross into the dynamic window.
    entry_mma_pipe_budget(
        name,
        ty,
        bm,
        bn,
        bk,
        warps_m,
        warps_n,
        stages,
        raster,
        pad,
        act,
        bias,
        residual,
        swz,
        min_ctas,
        store,
        STATIC_SMEM_CAP,
    )
}

/// [`entry_mma_pipe`] with an explicit shared-memory budget — see the `smem_budget` parameter.
#[allow(clippy::too_many_arguments)]
fn entry_mma_pipe_budget(
    name: &str,
    ty: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
    pad: usize,
    act: Act,
    bias: bool,
    residual: bool,
    swz: bool,
    // min_ctas: `0` ⇒ emit no launch-bounds directive (the driver JIT picks registers/occupancy blind,
    // the historical behavior). `>0` ⇒ emit `.maxntid {threads},1,1` + `.minnctapersm {min_ctas}` so
    // ptxas caps the per-thread register count to fit at least `min_ctas` CTAs/SM. The 128×128 tile
    // carries 64 f32 accumulators/thread, so the JIT defaults to ~2 CTAs/SM (register-limited) even
    // when SMEM would allow 3 — forcing the occupancy is the HBM-latency-hiding lever for ≥4096³.
    min_ctas: usize,
    // Epilogue store mode ([`Store`]): scalar `st.global.f32` (historical) vs vectorized `st.global.v2.f32`
    // (± `.cs` streaming hint). Bit-identical output; the `v2` forms halve the C-write store count.
    store: Store,
    // The SMEM ceiling this entry may spend, in bytes — and, via `smem_mode_for`, the choice of
    // *emission form*. `STATIC_SMEM_CAP` (what every historical caller passes) keeps the entry on the
    // static `.shared` path and its PTX byte-identical; a larger budget (the device's
    // `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`) lets a deeper ring be carved out of the module-scope
    // `.extern .shared` window instead. The generator stays a pure text function: the budget is passed
    // in by the dispatch layer, never probed here, so the whole candidate lattice is enumerable and
    // gateable with no device present — and an A100/H100 budget is testable on this laptop.
    smem_budget: usize,
) -> String {
    assert!(stages >= 2, "the pipeline needs at least 2 stages");
    assert!(
        bk.is_multiple_of(16) && (bk / 8).is_power_of_two(),
        "bk must be a 16-multiple with bk/8 a power of two"
    );
    assert!(
        bm.is_multiple_of(16 * warps_m),
        "{name}: bm must be a multiple of 16·warps_m (m16 sub-tiles)"
    );
    assert!(
        bn.is_multiple_of(8 * warps_n),
        "{name}: bn must be a multiple of 8·warps_n (n8 sub-tiles)"
    );
    assert!(
        raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()),
        "{name}: rasterization needs bm,bn powers of two"
    );
    let mma_ty = format!("f32.{ty}.{ty}.f32"); // f16 or bf16, f32 accumulate
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m); // m16 sub-tiles per warp
    let tn = bn / (8 * warps_n); //  n8 sub-tiles per warp
    let nks = bk / 16; // k16 steps fed by one staged tile
    let wmr = bm / warps_m; // per-warp M rows (WM)
    let wnc = bn / warps_n; // per-warp N cols (WN)
                            // SMEM rows padded by `pad` f16 (`ldp` leading dim): a b32 fragment load by lane (grp,tid) hits bank
                            // (grp·ldp/2 + tid) mod 32. pad=8 ⇒ ldp/2 = (bk+8)/2 ≡ 4·(odd), so grp·(ldp/2) spans all eight
                            // multiples of 4 and the 8 grp × 4 tid = 32 lanes hit 32 distinct banks — conflict-free (vs the 4-way
                            // conflict at stride bk where grp and grp+2 alias). pad=0 keeps the conflict but a smaller footprint ⇒
                            // more CTAs/SM. Pad keeps 16-byte alignment for cp.async. K-offsets stay bk-based.
    assert!(
        pad.is_multiple_of(8),
        "{name}: pad must be a multiple of 8 (16-byte cp.async alignment)"
    );
    // The **`swz` (ldmatrix + XOR-swizzle + no-pad)** path drops the padding (smaller footprint ⇒ more
    // CTAs/SM — 3 vs the padded 2 at this tile, the occupancy the HBM-bound 4096³ needs) and instead makes
    // both the cp.async stores and the `ldmatrix` gathers conflict-free via an XOR swizzle on the 16-byte
    // chunk column: chunk_col ↦ chunk_col XOR ((row>>1) & (nc-1)), nc = bk/8 chunks per row. Derived for
    // bk=32 (nc=4): the row stride contributes only row%2 to the bank, so the (row>>1)&3 phase spreads the
    // 8 rows of an ldmatrix matrix across 32 distinct banks. The phase reduces to a per-lane constant on the
    // read side because warpMrow / mi·16 / warpNcol / ni·8 are all ≡ 0 (mod 8) ⇒ vanish under (·>>1)&3.
    let nc = bk / 8; // 16-byte (8×f16) chunks per SMEM row
    let nc_mask = nc - 1;
    if swz {
        assert!(
            bk == 32,
            "{name}: the swz swizzle phase is derived for bk=32 (nc=4)"
        );
        assert!(
            wmr.is_multiple_of(8) && wnc.is_multiple_of(8),
            "{name}: swz needs warp row/col bases ≡ 0 (mod 8)"
        );
    }
    let ldp = if swz { bk } else { bk + pad }; // swz: no pad (the swizzle, not padding, gives conflict-free)
    let tile_a = bm * ldp * 2;
    let tile_b = bn * ldp * 2;
    let smem_a = stages * tile_a;
    let smem_b = stages * tile_b;
    // SMEM(stages) = stages·(bm+bn)·ldp·2 — the family's closed form. The budget is the ceiling; the
    // 48 KiB PTX ISA cap (STATIC_SMEM_CAP) decides the FORM, not the ceiling.
    let mode = smem_mode_for(smem_a + smem_b);
    assert!(
        smem_a + smem_b <= smem_budget,
        "{name}: SMEM {} B (stages={stages}, {bm}x{bn}, ldp={ldp}) exceeds the budget {smem_budget} B",
        smem_a + smem_b
    );
    // Sub-slab alignment inside the shared window: B starts right after the whole A ring, and every
    // `cp.async …,16` / `ldmatrix` into it assumes 16-B alignment. Loud at generation, because a
    // forgotten offset assert is the one way the swizzle/alignment risk escapes silently.
    assert!(
        smem_a.is_multiple_of(16),
        "{name}: B slab offset {smem_a} is not 16-B aligned"
    );
    // The staging loop issues exactly `chunks` 16-byte `cp.async`s per thread, so `bm·bk` and `bn·bk`
    // must be WHOLE multiples of `threads·8`. Checking only `>= 1` (what this generator did, while every
    // config in the tree happened to divide exactly) lets a truncating division through: the kernel then
    // stages a *prefix* of the tile, every `bar.sync`/`ldmatrix`/`mma` stays well-formed, it JITs
    // cleanly, and it computes C partly from stale shared memory — which reads as a tolerance failure,
    // not a codegen bug. [`entry_smem`] has carried the whole-multiple form since its own near-miss and
    // its doc names this generator as the one still missing it. Widening the CTA tile is exactly the
    // change that stops making the division obvious by inspection, so it gets the guard now.
    let a_chunks = bm * bk / (threads * 8);
    let b_chunks = bn * bk / (threads * 8);
    assert!(
        a_chunks >= 1 && a_chunks * threads * 8 == bm * bk,
        "{name}: A tile {bm}x{bk} is not a whole multiple of threads*8 = {} (128-bit vectorized staging)",
        threads * 8
    );
    assert!(
        b_chunks >= 1 && b_chunks * threads * 8 == bn * bk,
        "{name}: B tile {bn}x{bk} is not a whole multiple of threads*8 = {} (128-bit vectorized staging)",
        threads * 8
    );
    let bk_chunks = bk / 8;
    let row_shift = bk_chunks.trailing_zeros();
    let col_mask = bk_chunks - 1;
    let wn_shift = warps_n.trailing_zeros();

    // The fused-bias variant takes a `bias[N]` (f32) param applied per output column in the store
    // epilogue — the canonical `act(A·Bᵀ + bias)` Linear/FFN form. cuBLAS needs a 2nd kernel for it.
    // The fused-residual variant takes a `residual[M,N]` (f32) param added per element AFTER the
    // activation — `out = act(A·Bᵀ + bias) + residual`, the transformer down-proj / attention output-proj
    // sublayer output (the skip connection); fusing it folds the otherwise-separate residual-add kernel.
    let bias_param = if bias { ",\n    .param .u64 pBias" } else { "" };
    let resid_param = if residual {
        ",\n    .param .u64 pResid"
    } else {
        ""
    };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n"
    );
    // Launch-bounds: `.maxntid` declares the exact block size so ptxas sizes occupancy, and
    // `.minnctapersm` forces it to allocate ≤ 64K/(min_ctas·threads) registers/thread so `min_ctas`
    // CTAs co-reside per SM. Without these the JIT under- or over-occupies the register-heavy mma tile.
    // CUTLASS emits the same pair via `__launch_bounds__`. `min_ctas == 0` keeps byte-identical legacy PTX.
    if min_ctas > 0 {
        s += &format!(".maxntid {threads}, 1, 1\n.minnctapersm {min_ctas}\n");
    }
    s += "{\n";
    if !mode.is_dynamic() {
        s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
        s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
    }
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufcA,%bufcB,%bufwA,%bufwB,%lane,%grp,%tg,%tg2,%laneoff,%warpMrow,%warpNcol,%aptr,%bptr,%grow,%gcol;\n";
    // Fused-epilogue scratch: %act0/%act1 for the transcendental activations, %biasv0/%biasv1 for the
    // two bias columns this lane's D fragment spans, %Bias for the bias base pointer.
    if !matches!(act, Act::None) {
        s += "    .reg .f32 %act0,%act1;\n";
    }
    if bias {
        s += "    .reg .f32 %biasv0,%biasv1;\n    .reg .b64 %Bias;\n";
    }
    if residual {
        s += "    .reg .f32 %resv0,%resv1;\n    .reg .b64 %Resid;\n";
    }
    if swz {
        // %phaseA/%phaseB: per-lane swizzle phases; %arowb/%browb: this lane's A/B row byte-base
        // ((warp+lane)·bk·2); %la16=lane>>4 (A x4 chunk selector), %lb8=(lane>>3)&1 (B x2 selector);
        // %swztmp: chunk-offset scratch; %tmp3: staging dest scratch.
        s += "    .reg .b32 %phaseA,%phaseB,%arowb,%browb,%la16,%lb8,%swztmp,%tmp3;\n";
    }
    if raster > 0 {
        s += "    .reg .b32 %lin,%tn,%tm,%gsz,%grpr,%rem,%col0,%gw,%trow,%tcol;\n";
    }
    let mut decl_d = String::new();
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                decl_d += &format!("%d{mi}_{ni}_{r},");
            }
        }
    }
    s += &format!("    .reg .f32 {};\n", decl_d.trim_end_matches(','));
    let mut decl_ab = String::new();
    for mi in 0..tm {
        for r in 0..4 {
            decl_ab += &format!("%a{mi}_{r},");
        }
    }
    for ni in 0..tn {
        for r in 0..2 {
            decl_ab += &format!("%b{ni}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%B,%C,%off,%gptr,%cptr,%cptr2;\n";

    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    if bias {
        s += "    ld.param.u64 %Bias,[pBias];\n    cvta.to.global.u64 %Bias,%Bias;\n";
    }
    if residual {
        s += "    ld.param.u64 %Resid,[pResid];\n    cvta.to.global.u64 %Resid,%Resid;\n";
    }
    if raster == 0 {
        s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
        s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    } else {
        let (bn_sh, bm_sh) = (bn.trailing_zeros(), bm.trailing_zeros());
        s += &format!("    mov.u32 %lin,%ctaid.x;\n    shr.u32 %tn,%N,{bn_sh};\n    shr.u32 %tm,%M,{bm_sh};\n");
        s += &format!("    mul.lo.s32 %gsz,%tm,{raster};\n    div.u32 %grpr,%lin,%gsz;\n    rem.u32 %rem,%lin,%gsz;\n");
        s += &format!("    mul.lo.s32 %col0,%grpr,{raster};\n    sub.u32 %gw,%tn,%col0;\n    min.u32 %gw,%gw,{raster};\n");
        s += "    div.u32 %trow,%rem,%gw;\n    rem.u32 %tcol,%rem,%gw;\n    add.u32 %tcol,%tcol,%col0;\n";
        s += &format!("    mul.lo.s32 %baseRow,%trow,{bm};\n    mul.lo.s32 %baseCol,%tcol,{bn};\n");
    }
    // lane decomposition: grp = lane/4 (0..7), tg = lane%4 (0..3), tg2 = 2·tg.
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += "    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += &format!(
        "    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n",
        warps_n - 1
    );
    s += &format!(
        "    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n"
    );
    // Per-lane SMEM byte offset shared by A and B fragment loads: (grp·ldp + tg2)·2 (padded row stride).
    s += &format!("    mul.lo.s32 %laneoff,%grp,{ldp};\n    add.u32 %laneoff,%laneoff,%tg2;\n    shl.b32 %laneoff,%laneoff,1;\n");
    if swz {
        // A ldmatrix.x4: row R = warpMrow + mi·16 + (lane&15); arowb = (warpMrow + (lane&15))·bk·2 (the
        // mi·16 part is added per sub-tile). phaseA = ((lane&15)>>1)&(nc-1). la16 = lane>>4 (chunk selector).
        s += &format!("    and.b32 %tmp,%lane,15;\n    add.u32 %tmp2,%tmp,%warpMrow;\n    mul.lo.s32 %arowb,%tmp2,{};\n", bk * 2);
        s += &format!("    shr.u32 %tmp2,%tmp,1;\n    and.b32 %phaseA,%tmp2,{nc_mask};\n");
        s += "    shr.u32 %la16,%lane,4;\n";
        // B ldmatrix.x2: row R = warpNcol + ni·8 + (lane&7); browb = (warpNcol + (lane&7))·bk·2.
        // phaseB = ((lane&7)>>1)&(nc-1). lb8 = (lane>>3)&1 (chunk selector).
        s += &format!("    and.b32 %tmp,%lane,7;\n    add.u32 %tmp2,%tmp,%warpNcol;\n    mul.lo.s32 %browb,%tmp2,{};\n", bk * 2);
        s += &format!("    shr.u32 %tmp2,%tmp,1;\n    and.b32 %phaseB,%tmp2,{nc_mask};\n");
        s += "    shr.u32 %tmp,%lane,3;\n    and.b32 %lb8,%tmp,1;\n";
    }
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %d{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }

    // cp.async staging into the SMEM tile. Non-swz: **padded** row-major (`bufoff + row·ldp·2 + col·2`,
    // ldp=bk+pad) — the layout the hand-placed b32 fragment loads read conflict-free. swz: **no-pad +
    // XOR-swizzle** (`bufoff + row·bk·2 + (chunk XOR ((row>>1)&nc_mask))·16`) — the layout the `ldmatrix`
    // gathers read conflict-free, at the higher occupancy the dropped padding buys. Same global read either way.
    // Where each ring lives. Static: its own entry-name-qualified `.shared` array, offset 0 — the
    // historical spelling, emitted verbatim. Dynamic: BOTH rings are windows into the single
    // module-scope `wk_dsmem` (two module-scope externs ALIAS), A at 0 and B at the whole A ring.
    let (sym_a, sym_b, off_b): (String, String, usize) = if mode.is_dynamic() {
        (DSMEM_SYM.to_string(), DSMEM_SYM.to_string(), smem_a)
    } else {
        (format!("smemA_{name}"), format!("smemB_{name}"), 0)
    };
    // `mov.u32 %reg,<window>;` plus the constant slab offset when the two rings share one window.
    // Always through the symbol: the dynamic window does not start at address 0 when an entry also
    // declares statics, so a hardcoded base would silently land in the wrong place.
    let base_into = |reg: &str, sym: &str, off: usize| -> String {
        if off == 0 {
            format!("    mov.u32 {reg},{sym};\n")
        } else {
            format!("    mov.u32 {reg},{sym};\n    add.u32 {reg},{reg},{off};\n")
        }
    };
    let a_base = base_into("%tmp", &sym_a, 0);
    let b_base = base_into("%tmp", &sym_b, off_b);

    // `smem` is the pre-rendered ring base (the `mov`, plus the constant slab offset in the shared
    // window); the static form is the historical single `mov`, byte for byte.
    let stage = |g_base: &str,
                 gbase_ptr: &str,
                 smem: &str,
                 bufoff: &str,
                 chunks: usize,
                 s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,3;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            if swz {
                // chunk_col = %c>>3 (col8→chunk); chunk_swz = chunk XOR ((row>>1)&nc_mask); dest = smem +
                // bufoff + row·bk·2 + chunk_swz·16 — the swizzle the ldmatrix reads invert (write/read agree).
                *s += &format!("    shr.u32 %swztmp,%c,3;\n    shr.u32 %tmp2,%r,1;\n    and.b32 %tmp2,%tmp2,{nc_mask};\n    xor.b32 %swztmp,%swztmp,%tmp2;\n    shl.b32 %swztmp,%swztmp,4;\n");
                *s += smem;
                *s += &format!("    add.u32 %tmp,%tmp,{bufoff};\n    mul.lo.s32 %tmp3,%r,{};\n    add.u32 %tmp,%tmp,%tmp3;\n    add.u32 %tmp,%tmp,%swztmp;\n", bk * 2);
            } else {
                *s += smem;
                *s += &format!("    add.u32 %tmp,%tmp,{bufoff};\n");
                *s += &format!("    mul.lo.s32 %tmp2,%r,{};\n    add.u32 %tmp,%tmp,%tmp2;\n    shl.b32 %tmp2,%c,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ldp * 2);
            }
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    for st in 0..(stages - 1) {
        s += &format!("    mov.u32 %kcol,{};\n", st * bk);
        s += &format!(
            "    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n",
            st * tile_a,
            st * tile_b
        );
        s += &format!("    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra PRO_{name}_{st};\n");
        stage("%baseRow", "%A", &a_base, "%bufwA", a_chunks, &mut s);
        stage("%baseCol", "%B", &b_base, "%bufwB", b_chunks, &mut s);
        s += &format!("PRO_{name}_{st}:\n    cp.async.commit_group;\n");
    }
    s += "    mov.u32 %bufcA,0;\n    mov.u32 %bufcB,0;\n";
    s += &format!(
        "    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n",
        (stages - 1) * tile_a,
        (stages - 1) * tile_b
    );
    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 2);
    s += &format!("    add.u32 %kcol,%kt,{};\n    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra NOPRE_{name};\n", (stages - 1) * bk);
    stage("%baseRow", "%A", &a_base, "%bufwA", a_chunks, &mut s);
    stage("%baseCol", "%B", &b_base, "%bufwB", b_chunks, &mut s);
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");

    // Compute: for each k16 step load the A/B fragments — swz: one `ldmatrix.x4`/`.x2` warp-cooperative
    // gather per sub-tile from the XOR-swizzled (conflict-free, no-pad) SMEM; non-swz: hand-placed
    // `ld.shared.b32` from the padded SMEM — then issue tm·tn `mma.sync` m16n8k16 ops (identical either way).
    for ks in 0..nks {
        if swz {
            // A: aptr = smemA + bufcA + arowb (this lane's row base). chunk_off = ((ks·2 | la16) XOR phaseA)·16
            // (per-ks, per-lane); per mi add mi·16·bk·2 → ldmatrix.x4 {A00,A10,A01,A11} = the mma A regs.
            s += &base_into("%aptr", &sym_a, 0);
            s += "    add.u32 %aptr,%aptr,%bufcA;\n    add.u32 %aptr,%aptr,%arowb;\n";
            s += &format!("    or.b32 %swztmp,%la16,{};\n    xor.b32 %swztmp,%swztmp,%phaseA;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for mi in 0..tm {
                let mibase = mi * 16 * bk * 2;
                s += &format!("    add.u32 %tmp,%aptr,%swztmp;\n    add.u32 %tmp,%tmp,{mibase};\n    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},[%tmp];\n");
            }
            // B: bptr = smemB + bufcB + browb. chunk_off = ((ks·2 | lb8) XOR phaseB)·16; per ni add ni·8·bk·2
            // → ldmatrix.x2 {B0,B1} = the mma B regs.
            s += &base_into("%bptr", &sym_b, off_b);
            s += "    add.u32 %bptr,%bptr,%bufcB;\n    add.u32 %bptr,%bptr,%browb;\n";
            s += &format!("    or.b32 %swztmp,%lb8,{};\n    xor.b32 %swztmp,%swztmp,%phaseB;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for ni in 0..tn {
                let nibase = ni * 8 * bk * 2;
                s += &format!("    add.u32 %tmp,%bptr,%swztmp;\n    add.u32 %tmp,%tmp,{nibase};\n    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%b{ni}_0,%b{ni}_1}},[%tmp];\n");
            }
        } else {
            // A base ptr = smemA + bufcA + (warpMrow·ldp)·2 + laneoff + (ks·16)·2  (ldp = padded row stride).
            s += &base_into("%aptr", &sym_a, 0);
            s += "    add.u32 %aptr,%aptr,%bufcA;\n";
            s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %aptr,%aptr,%tmp;\n");
            s += &format!(
                "    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n",
                ks * 32
            );
            for mi in 0..tm {
                let base = mi * 16 * ldp * 2; // m16-block row offset (bytes, padded stride)
                let r8 = 8 * ldp * 2; // +8 rows
                s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16); // k+8 (not padded)
                s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
                s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
            }
            // B base ptr = smemB + bufcB + (warpNcol·ldp)·2 + laneoff + (ks·16)·2.
            s += &base_into("%bptr", &sym_b, off_b);
            s += "    add.u32 %bptr,%bptr,%bufcB;\n";
            s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %bptr,%bptr,%tmp;\n");
            s += &format!(
                "    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n",
                ks * 32
            );
            for ni in 0..tn {
                let base = ni * 8 * ldp * 2; // n8-block row offset (bytes, padded stride)
                s += &format!("    ld.shared.b32 %b{ni}_0,[%bptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %b{ni}_1,[%bptr+{}];\n", base + 16);
                // k+8 (not padded)
            }
        }
        for mi in 0..tm {
            for ni in 0..tn {
                s += &format!(
                    "    mma.sync.aligned.m16n8k16.row.col.{mma_ty} {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}},{{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},{{%b{ni}_0,%b{ni}_1}},{{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}};\n"
                );
            }
        }
    }
    s += &format!("    add.u32 %bufcA,%bufcA,{tile_a};\n    setp.ge.u32 %pmore,%bufcA,{smem_a};\n    @%pmore sub.u32 %bufcA,%bufcA,{smem_a};\n");
    s += &format!("    add.u32 %bufcB,%bufcB,{tile_b};\n    setp.ge.u32 %pmore,%bufcB,{smem_b};\n    @%pmore sub.u32 %bufcB,%bufcB,{smem_b};\n");
    s += &format!("    add.u32 %bufwA,%bufwA,{tile_a};\n    setp.ge.u32 %pmore,%bufwA,{smem_a};\n    @%pmore sub.u32 %bufwA,%bufwA,{smem_a};\n");
    s += &format!("    add.u32 %bufwB,%bufwB,{tile_b};\n    setp.ge.u32 %pmore,%bufwB,{smem_b};\n    @%pmore sub.u32 %bufwB,%bufwB,{smem_b};\n");
    s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

    // Store: D fragment → C. Lane holds C[grow0..][gcol0..]: d0=(grp,2tg) d1=(grp,2tg+1)
    // d2=(grp+8,2tg) d3=(grp+8,2tg+1), per (mi,ni) sub-tile. Each row's two accumulators are adjacent
    // columns ⇒ contiguous in C, so `Store::V2{,Cs}` folds them into one `st.global.v2.f32` (bit-identical).
    let store_pair = |dst: &str, lo: &str, hi: &str| -> String {
        match store {
            Store::Scalar => {
                format!("    st.global.f32 [{dst}],{lo};\n    st.global.f32 [{dst}+4],{hi};\n")
            }
            Store::V2 => format!("    st.global.v2.f32 [{dst}],{{{lo},{hi}}};\n"),
            Store::V2Cs => format!("    st.global.cs.v2.f32 [{dst}],{{{lo},{hi}}};\n"),
        }
    };
    s += &format!("KEND_{name}:\n");
    for mi in 0..tm {
        for ni in 0..tn {
            s += &format!("    add.u32 %grow,%baseRow,%warpMrow;\n    add.u32 %grow,%grow,{};\n    add.u32 %grow,%grow,%grp;\n", mi * 16);
            s += &format!("    add.u32 %gcol,%baseCol,%warpNcol;\n    add.u32 %gcol,%gcol,{};\n    add.u32 %gcol,%gcol,%tg2;\n", ni * 8);
            // Fused epilogue, applied to the f32 accumulators before the store: bias add then activation,
            // i.e. C = act(A·Bᵀ + bias). The lane's four D regs span two columns — d0,d2 at gcol and
            // d1,d3 at gcol+1 — so bias[gcol]→biasv0 hits d0,d2 and bias[gcol+1]→biasv1 hits d1,d3.
            if bias {
                s += "    mul.wide.u32 %off,%gcol,4;\n    add.s64 %cptr,%Bias,%off;\n";
                s += "    ld.global.f32 %biasv0,[%cptr];\n    ld.global.f32 %biasv1,[%cptr+4];\n";
                s += &format!("    add.f32 %d{mi}_{ni}_0,%d{mi}_{ni}_0,%biasv0;\n    add.f32 %d{mi}_{ni}_1,%d{mi}_{ni}_1,%biasv1;\n");
                s += &format!("    add.f32 %d{mi}_{ni}_2,%d{mi}_{ni}_2,%biasv0;\n    add.f32 %d{mi}_{ni}_3,%d{mi}_{ni}_3,%biasv1;\n");
            }
            if !matches!(act, Act::None) {
                for r in 0..4 {
                    s += &act.epilogue(&format!("%d{mi}_{ni}_{r}"));
                }
            }
            // row grp: C[grow·N+gcol] = d0, [+1] = d1. With residual, add residual[grow,gcol..+1] to the
            // post-activation accumulators (scratch in the still-free %cptr2): out = act(A·Bᵀ+bias)+residual.
            s += "    mul.lo.s32 %tmp,%grow,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
            if residual {
                s += "    add.s64 %cptr2,%Resid,%off;\n    ld.global.f32 %resv0,[%cptr2];\n    ld.global.f32 %resv1,[%cptr2+4];\n";
                s += &format!("    add.f32 %d{mi}_{ni}_0,%d{mi}_{ni}_0,%resv0;\n    add.f32 %d{mi}_{ni}_1,%d{mi}_{ni}_1,%resv1;\n");
            }
            s += &store_pair(
                "%cptr",
                &format!("%d{mi}_{ni}_0"),
                &format!("%d{mi}_{ni}_1"),
            );
            // row grp+8: C[(grow+8)·N+gcol] = d2, [+1] = d3 (residual scratch in the now-free %cptr).
            s += "    add.u32 %tmp,%grow,8;\n    mul.lo.s32 %tmp,%tmp,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr2,%C,%off;\n";
            if residual {
                s += "    add.s64 %cptr,%Resid,%off;\n    ld.global.f32 %resv0,[%cptr];\n    ld.global.f32 %resv1,[%cptr+4];\n";
                s += &format!("    add.f32 %d{mi}_{ni}_2,%d{mi}_{ni}_2,%resv0;\n    add.f32 %d{mi}_{ni}_3,%d{mi}_{ni}_3,%resv1;\n");
            }
            s += &store_pair(
                "%cptr2",
                &format!("%d{mi}_{ni}_2"),
                &format!("%d{mi}_{ni}_3"),
            );
        }
    }
    s += "    ret;\n}\n";
    s
}

/// Generate a **fused gated-FFN ("GLU-family")** `mma.sync.m16n8k16` kernel computing
/// `out = act(x·Wgᵀ [+ bg]) ⊙ (x·Wuᵀ [+ bu])` — the **SwiGLU** (act=silu) / **GeGLU** (act=gelu) /
/// bilinear-GLU (act=none) FFN gate that every modern LLM (Llama, Mistral, Gemma, …) runs. It shares ONE
/// staged A tile (the activations `x`) between **two** B matrices — the gate weight `Wg` and the up
/// weight `Wu` — keeps two accumulator sets, applies the activation to only the gate branch, and folds
/// the elementwise product into the store. cuBLAS structurally needs **three** kernels for this (two
/// GEMMs + an elementwise multiply) and round-trips both `[M,N]` intermediates through HBM; here `x` is
/// read from global **once** (the shared A fragments feed both `mma` chains) and only the gated product
/// touches HBM — a fusion that is *not* PTX-ceiling-bound the way the plain GEMM is.
///
/// Tile is **128×64** (not the workhorse's 128×128): two accumulator sets at 128×64 cost the SAME
/// `tm·tn·4·2 = 64` f32 D-regs/thread as one set at 128×128 (register-neutral), and three staged tiles
/// (A `128×bk` + Wg `64×bk` + Wu `64×bk` = `stages·(bm+2·bn)·ldp·2`) fit the same 40 KiB the single-B
/// 128×128 uses (SMEM-neutral). The two GEMMs share the proven [`entry_mma_pipe`] recipe — `cp.async`
/// pipeline, padded conflict-free SMEM fragment loads, threadblock raster — and the D-fragment column map
/// is identical for both, so the per-column bias add and the SFU activation reuse [`Act::epilogue`]
/// verbatim. Same shape constraints as [`entry_mma_pipe`]; `bias` adds per-column `bg[N]`,`bu[N]` params.
/// `swz` selects the no-pad `ldmatrix`+XOR-swizzle staging (the [`entry_mma_pipe`] technique ported to both
/// gated GEMMs — A is shared, the two B tiles reuse one swizzle derivation; bit-identical output). NOTE:
/// measured a wash-to-loss for this dual-B tile (the single-B win does not transfer); production uses padded.
fn entry_mma_gate(
    name: &str,
    ty: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
    pad: usize,
    gate_act: Act,
    bias: bool,
    swz: bool,
) -> String {
    assert!(stages >= 2, "the pipeline needs at least 2 stages");
    assert!(
        bk.is_multiple_of(16) && (bk / 8).is_power_of_two(),
        "bk must be a 16-multiple with bk/8 a power of two"
    );
    assert!(
        bm.is_multiple_of(16 * warps_m),
        "{name}: bm must be a multiple of 16·warps_m"
    );
    assert!(
        bn.is_multiple_of(8 * warps_n),
        "{name}: bn must be a multiple of 8·warps_n"
    );
    assert!(
        raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()),
        "{name}: rasterization needs bm,bn powers of two"
    );
    assert!(
        pad.is_multiple_of(8),
        "{name}: pad must be a multiple of 8 (16-byte cp.async alignment)"
    );
    let mma_ty = format!("f32.{ty}.{ty}.f32");
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m); // m16 sub-tiles per warp
    let tn = bn / (8 * warps_n); //  n8 sub-tiles per warp
    let nks = bk / 16;
    let wmr = bm / warps_m;
    let wnc = bn / warps_n;
    let nc = bk / 8; // 16-byte (8×f16) chunks per SMEM row — the swz XOR-swizzle modulus
    let nc_mask = nc - 1;
    // The **`swz`** path (ldmatrix + XOR-swizzle + no-pad) ports the single-B workhorse's technique to BOTH
    // gated GEMMs at once: Wg and Wu share x's `[M,K]` A tile and have *identical* `[N,K]` layout, so the
    // A/B swizzle derivation (see [`entry_mma_pipe`], derived for bk=32/nc=4) ports verbatim — one A path
    // feeds both `mma` chains, and the B path runs once per B tile sharing the same `%browb`/`%phaseB`/`%lb8`
    // (only the smem base + ring cursor differ). The register-level epilogue is untouched, so the output is
    // **bit-identical** to the padded gate, and no-pad *shrinks* SMEM (128×64/s2: 40 KiB → 32 KiB).
    // **MEASURED: the single-B 128×128 swz win does NOT transfer to this dual-B 128×64 tile** — same-run vs
    // the padded gate it is a wash-to-loss (geomean ≈1.00×, ~0.89× @2048³; see `fused_swiglu_gate_vs_chain`),
    // because the gate already amortizes A across both GEMMs (load-x-once) so it is far less SMEM-fragment-
    // load-bound (the path `ldmatrix` accelerates), and tn=2 (vs the single-B tn=4) leaves too little mma-ILP
    // to hide the no-pad swizzle's per-ks address arithmetic. Emitted + correctness-gated as a verified
    // alternative; **production routes to the padded base** (see `gemm_nt_f16_swiglu`).
    if swz {
        assert!(
            bk == 32,
            "{name}: the swz swizzle phase is derived for bk=32 (nc=4)"
        );
        assert!(
            wmr.is_multiple_of(8) && wnc.is_multiple_of(8),
            "{name}: swz needs warp row/col bases ≡ 0 (mod 8)"
        );
    }
    let ldp = if swz { bk } else { bk + pad }; // swz: no pad (the swizzle, not padding, gives conflict-free)
    let tile_a = bm * ldp * 2;
    let tile_b = bn * ldp * 2; // one Wg (== one Wu) tile
    let smem_a = stages * tile_a;
    let smem_b = stages * tile_b;
    // Three ring buffers: A + Wg + Wu. 128×64/bk32/pad8/s2 ⇒ 20480 + 2·10240 = 40 KiB (= single-B 128×128).
    assert!(
        smem_a + 2 * smem_b <= 48 * 1024,
        "{name}: static SMEM {} B exceeds 48 KiB",
        smem_a + 2 * smem_b
    );
    let a_chunks = bm * bk / (threads * 8);
    let b_chunks = bn * bk / (threads * 8);
    assert!(
        a_chunks >= 1 && b_chunks >= 1,
        "{name}: tile too small for one 128-bit chunk per thread"
    );
    let bk_chunks = bk / 8;
    let row_shift = bk_chunks.trailing_zeros();
    let col_mask = bk_chunks - 1;
    let wn_shift = warps_n.trailing_zeros();

    // The fused-bias variant takes per-column gate/up biases `bg[N]`,`bu[N]` (f32), added in the store
    // epilogue (the `silu(x·Wg+bg) ⊙ (x·Wu+bu)` form; biasless is the Llama-style no-bias gate).
    let bias_param = if bias {
        ",\n    .param .u64 pBiasG,\n    .param .u64 pBiasU"
    } else {
        ""
    };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pBg,\n    .param .u64 pBu,\n    .param .u64 pC{bias_param}\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
    s += &format!("    .shared .align 16 .b8 smemBg_{name}[{smem_b}];\n");
    s += &format!("    .shared .align 16 .b8 smemBu_{name}[{smem_b}];\n");
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufcA,%bufcBg,%bufcBu,%bufwA,%bufwBg,%bufwBu,%lane,%grp,%tg,%tg2,%laneoff,%warpMrow,%warpNcol,%aptr,%bptr,%grow,%gcol;\n";
    if !matches!(gate_act, Act::None) {
        s += "    .reg .f32 %act0,%act1;\n";
    }
    if bias {
        s += "    .reg .f32 %biasg0,%biasg1,%biasu0,%biasu1;\n    .reg .b64 %BiasG,%BiasU;\n";
    }
    if swz {
        // swz scratch (shared by A and BOTH B tiles): %phaseA/%phaseB per-lane swizzle phases; %arowb/%browb
        // this lane's A/B row byte-base; %la16=lane>>4 (A x4 chunk selector), %lb8=(lane>>3)&1 (B x2 selector);
        // %swztmp: chunk-offset scratch; %tmp3: staging dest scratch.
        s += "    .reg .b32 %phaseA,%phaseB,%arowb,%browb,%la16,%lb8,%swztmp,%tmp3;\n";
    }
    if raster > 0 {
        s += "    .reg .b32 %lin,%tn,%tm,%gsz,%grpr,%rem,%col0,%gw,%trow,%tcol;\n";
    }
    // Two accumulator sets (gate %dg, up %du) — same per-warp tm×tn×4 layout, so they share the lane→
    // (row,col) map and combine register-for-register in the epilogue product.
    let mut decl_d = String::new();
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                decl_d += &format!("%dg{mi}_{ni}_{r},%du{mi}_{ni}_{r},");
            }
        }
    }
    s += &format!("    .reg .f32 {};\n", decl_d.trim_end_matches(','));
    let mut decl_ab = String::new();
    for mi in 0..tm {
        for r in 0..4 {
            decl_ab += &format!("%a{mi}_{r},");
        }
    }
    for ni in 0..tn {
        for r in 0..2 {
            decl_ab += &format!("%bg{ni}_{r},%bu{ni}_{r},");
        }
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%Bg,%Bu,%C,%off,%gptr,%cptr,%cptr2;\n";

    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %Bg,[pBg];\n    ld.param.u64 %Bu,[pBu];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %Bg,%Bg;\n    cvta.to.global.u64 %Bu,%Bu;\n    cvta.to.global.u64 %C,%C;\n";
    if bias {
        s += "    ld.param.u64 %BiasG,[pBiasG];\n    cvta.to.global.u64 %BiasG,%BiasG;\n";
        s += "    ld.param.u64 %BiasU,[pBiasU];\n    cvta.to.global.u64 %BiasU,%BiasU;\n";
    }
    if raster == 0 {
        s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
        s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    } else {
        let (bn_sh, bm_sh) = (bn.trailing_zeros(), bm.trailing_zeros());
        s += &format!("    mov.u32 %lin,%ctaid.x;\n    shr.u32 %tn,%N,{bn_sh};\n    shr.u32 %tm,%M,{bm_sh};\n");
        s += &format!("    mul.lo.s32 %gsz,%tm,{raster};\n    div.u32 %grpr,%lin,%gsz;\n    rem.u32 %rem,%lin,%gsz;\n");
        s += &format!("    mul.lo.s32 %col0,%grpr,{raster};\n    sub.u32 %gw,%tn,%col0;\n    min.u32 %gw,%gw,{raster};\n");
        s += "    div.u32 %trow,%rem,%gw;\n    rem.u32 %tcol,%rem,%gw;\n    add.u32 %tcol,%tcol,%col0;\n";
        s += &format!("    mul.lo.s32 %baseRow,%trow,{bm};\n    mul.lo.s32 %baseCol,%tcol,{bn};\n");
    }
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += "    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n";
    s += &format!(
        "    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n",
        warps_n - 1
    );
    s += &format!(
        "    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n"
    );
    s += &format!("    mul.lo.s32 %laneoff,%grp,{ldp};\n    add.u32 %laneoff,%laneoff,%tg2;\n    shl.b32 %laneoff,%laneoff,1;\n");
    if swz {
        // A ldmatrix.x4: row R = warpMrow + mi·16 + (lane&15); arowb = (warpMrow + (lane&15))·bk·2 (mi·16
        // added per sub-tile). phaseA = ((lane&15)>>1)&(nc-1). la16 = lane>>4 (x4 chunk selector). Shared by
        // the gate and up chains (both read the SAME staged x). B (Wg AND Wu) ldmatrix.x2: row R = warpNcol +
        // ni·8 + (lane&7); browb = (warpNcol + (lane&7))·bk·2; phaseB = ((lane&7)>>1)&(nc-1); lb8 = (lane>>3)&1.
        s += &format!("    and.b32 %tmp,%lane,15;\n    add.u32 %tmp2,%tmp,%warpMrow;\n    mul.lo.s32 %arowb,%tmp2,{};\n", bk * 2);
        s += &format!("    shr.u32 %tmp2,%tmp,1;\n    and.b32 %phaseA,%tmp2,{nc_mask};\n");
        s += "    shr.u32 %la16,%lane,4;\n";
        s += &format!("    and.b32 %tmp,%lane,7;\n    add.u32 %tmp2,%tmp,%warpNcol;\n    mul.lo.s32 %browb,%tmp2,{};\n", bk * 2);
        s += &format!("    shr.u32 %tmp2,%tmp,1;\n    and.b32 %phaseB,%tmp2,{nc_mask};\n");
        s += "    shr.u32 %tmp,%lane,3;\n    and.b32 %lb8,%tmp,1;\n";
    }
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %dg{mi}_{ni}_{r},0f00000000;\n    mov.f32 %du{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }

    // cp.async staging into the SMEM tile — identical for A, Wg, Wu. Non-swz: **padded** row-major
    // (`bufoff + row·ldp·2 + col·2`, the layout the hand-placed b32 fragment loads read conflict-free).
    // swz: **no-pad + XOR-swizzle** (`bufoff + row·bk·2 + (chunk XOR ((row>>1)&nc_mask))·16`, the layout the
    // `ldmatrix` gathers read conflict-free). Same global read either way.
    let stage = |g_base: &str,
                 gbase_ptr: &str,
                 smem: &str,
                 bufoff: &str,
                 chunks: usize,
                 s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,3;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            if swz {
                // chunk_col = %c>>3; chunk_swz = chunk XOR ((row>>1)&nc_mask); dest = smem + bufoff +
                // row·bk·2 + chunk_swz·16 — the swizzle the ldmatrix reads invert (write/read agree).
                *s += &format!("    shr.u32 %swztmp,%c,3;\n    shr.u32 %tmp2,%r,1;\n    and.b32 %tmp2,%tmp2,{nc_mask};\n    xor.b32 %swztmp,%swztmp,%tmp2;\n    shl.b32 %swztmp,%swztmp,4;\n");
                *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    mul.lo.s32 %tmp3,%r,{};\n    add.u32 %tmp,%tmp,%tmp3;\n    add.u32 %tmp,%tmp,%swztmp;\n", bk * 2);
            } else {
                *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n");
                *s += &format!("    mul.lo.s32 %tmp2,%r,{};\n    add.u32 %tmp,%tmp,%tmp2;\n    shl.b32 %tmp2,%c,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ldp * 2);
            }
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    // Prologue: prefetch tiles 0..stages-2 of A, Wg, Wu (one cp.async group per K-tile).
    for st in 0..(stages - 1) {
        s += &format!("    mov.u32 %kcol,{};\n", st * bk);
        s += &format!(
            "    mov.u32 %bufwA,{};\n    mov.u32 %bufwBg,{};\n    mov.u32 %bufwBu,{};\n",
            st * tile_a,
            st * tile_b,
            st * tile_b
        );
        s += &format!("    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra PRO_{name}_{st};\n");
        stage(
            "%baseRow",
            "%A",
            &format!("smemA_{name}"),
            "%bufwA",
            a_chunks,
            &mut s,
        );
        stage(
            "%baseCol",
            "%Bg",
            &format!("smemBg_{name}"),
            "%bufwBg",
            b_chunks,
            &mut s,
        );
        stage(
            "%baseCol",
            "%Bu",
            &format!("smemBu_{name}"),
            "%bufwBu",
            b_chunks,
            &mut s,
        );
        s += &format!("PRO_{name}_{st}:\n    cp.async.commit_group;\n");
    }
    s += "    mov.u32 %bufcA,0;\n    mov.u32 %bufcBg,0;\n    mov.u32 %bufcBu,0;\n";
    s += &format!(
        "    mov.u32 %bufwA,{};\n    mov.u32 %bufwBg,{};\n    mov.u32 %bufwBu,{};\n",
        (stages - 1) * tile_a,
        (stages - 1) * tile_b,
        (stages - 1) * tile_b
    );
    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 2);
    s += &format!("    add.u32 %kcol,%kt,{};\n    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra NOPRE_{name};\n", (stages - 1) * bk);
    stage(
        "%baseRow",
        "%A",
        &format!("smemA_{name}"),
        "%bufwA",
        a_chunks,
        &mut s,
    );
    stage(
        "%baseCol",
        "%Bg",
        &format!("smemBg_{name}"),
        "%bufwBg",
        b_chunks,
        &mut s,
    );
    stage(
        "%baseCol",
        "%Bu",
        &format!("smemBu_{name}"),
        "%bufwBu",
        b_chunks,
        &mut s,
    );
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");

    // Compute: load A fragments ONCE (smem + buffer + warp + lane + ks·16), then issue the gate `mma`s
    // (A×Wg → %dg) and the up `mma`s (A×Wu → %du, A fragments reused) — the load-x-once arithmetic win.
    // Both B tiles are loaded before any mma so the 2·tm·tn mma's (disjoint %dg/%du accumulators ⇒ all
    // mutually independent) form one pipeline-able block. swz: warp-cooperative `ldmatrix` from the
    // XOR-swizzled no-pad SMEM; non-swz: hand-placed `ld.shared.b32` from the padded SMEM (same fragments).
    for ks in 0..nks {
        if swz {
            // A (x): one ldmatrix.x4 per m16 sub-tile; chunk_off = ((ks·2 | la16) XOR phaseA)·16, +mi·16·bk·2.
            s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n    add.u32 %aptr,%aptr,%arowb;\n");
            s += &format!("    or.b32 %swztmp,%la16,{};\n    xor.b32 %swztmp,%swztmp,%phaseA;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for mi in 0..tm {
                let mibase = mi * 16 * bk * 2;
                s += &format!("    add.u32 %tmp,%aptr,%swztmp;\n    add.u32 %tmp,%tmp,{mibase};\n    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},[%tmp];\n");
            }
            // Wg then Wu: one ldmatrix.x2 per n8 sub-tile; shared %browb/%phaseB/%lb8, only smem base + cursor
            // differ. chunk_off = ((ks·2 | lb8) XOR phaseB)·16, +ni·8·bk·2.
            s += &format!("    mov.u32 %bptr,smemBg_{name};\n    add.u32 %bptr,%bptr,%bufcBg;\n    add.u32 %bptr,%bptr,%browb;\n");
            s += &format!("    or.b32 %swztmp,%lb8,{};\n    xor.b32 %swztmp,%swztmp,%phaseB;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for ni in 0..tn {
                let nibase = ni * 8 * bk * 2;
                s += &format!("    add.u32 %tmp,%bptr,%swztmp;\n    add.u32 %tmp,%tmp,{nibase};\n    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%bg{ni}_0,%bg{ni}_1}},[%tmp];\n");
            }
            s += &format!("    mov.u32 %bptr,smemBu_{name};\n    add.u32 %bptr,%bptr,%bufcBu;\n    add.u32 %bptr,%bptr,%browb;\n");
            s += &format!("    or.b32 %swztmp,%lb8,{};\n    xor.b32 %swztmp,%swztmp,%phaseB;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for ni in 0..tn {
                let nibase = ni * 8 * bk * 2;
                s += &format!("    add.u32 %tmp,%bptr,%swztmp;\n    add.u32 %tmp,%tmp,{nibase};\n    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%bu{ni}_0,%bu{ni}_1}},[%tmp];\n");
            }
        } else {
            s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n");
            s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %aptr,%aptr,%tmp;\n");
            s += &format!(
                "    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n",
                ks * 32
            );
            for mi in 0..tm {
                let base = mi * 16 * ldp * 2;
                let r8 = 8 * ldp * 2;
                s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16);
                s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
                s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
            }
            s += &format!("    mov.u32 %bptr,smemBg_{name};\n    add.u32 %bptr,%bptr,%bufcBg;\n");
            s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %bptr,%bptr,%tmp;\n");
            s += &format!(
                "    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n",
                ks * 32
            );
            for ni in 0..tn {
                let base = ni * 8 * ldp * 2;
                s += &format!("    ld.shared.b32 %bg{ni}_0,[%bptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %bg{ni}_1,[%bptr+{}];\n", base + 16);
            }
            s += &format!("    mov.u32 %bptr,smemBu_{name};\n    add.u32 %bptr,%bptr,%bufcBu;\n");
            s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %bptr,%bptr,%tmp;\n");
            s += &format!(
                "    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n",
                ks * 32
            );
            for ni in 0..tn {
                let base = ni * 8 * ldp * 2;
                s += &format!("    ld.shared.b32 %bu{ni}_0,[%bptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %bu{ni}_1,[%bptr+{}];\n", base + 16);
            }
        }
        // Interleave gate and up mma per (mi,ni): adjacent independent ops (different accumulators) give
        // the issue stage maximal ILP. A fragments are shared (loaded once above) → the load-x-once win.
        for mi in 0..tm {
            for ni in 0..tn {
                s += &format!(
                    "    mma.sync.aligned.m16n8k16.row.col.{mma_ty} {{%dg{mi}_{ni}_0,%dg{mi}_{ni}_1,%dg{mi}_{ni}_2,%dg{mi}_{ni}_3}},{{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},{{%bg{ni}_0,%bg{ni}_1}},{{%dg{mi}_{ni}_0,%dg{mi}_{ni}_1,%dg{mi}_{ni}_2,%dg{mi}_{ni}_3}};\n"
                );
                s += &format!(
                    "    mma.sync.aligned.m16n8k16.row.col.{mma_ty} {{%du{mi}_{ni}_0,%du{mi}_{ni}_1,%du{mi}_{ni}_2,%du{mi}_{ni}_3}},{{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},{{%bu{ni}_0,%bu{ni}_1}},{{%du{mi}_{ni}_0,%du{mi}_{ni}_1,%du{mi}_{ni}_2,%du{mi}_{ni}_3}};\n"
                );
            }
        }
    }
    s += &format!("    add.u32 %bufcA,%bufcA,{tile_a};\n    setp.ge.u32 %pmore,%bufcA,{smem_a};\n    @%pmore sub.u32 %bufcA,%bufcA,{smem_a};\n");
    s += &format!("    add.u32 %bufcBg,%bufcBg,{tile_b};\n    setp.ge.u32 %pmore,%bufcBg,{smem_b};\n    @%pmore sub.u32 %bufcBg,%bufcBg,{smem_b};\n");
    s += &format!("    add.u32 %bufcBu,%bufcBu,{tile_b};\n    setp.ge.u32 %pmore,%bufcBu,{smem_b};\n    @%pmore sub.u32 %bufcBu,%bufcBu,{smem_b};\n");
    s += &format!("    add.u32 %bufwA,%bufwA,{tile_a};\n    setp.ge.u32 %pmore,%bufwA,{smem_a};\n    @%pmore sub.u32 %bufwA,%bufwA,{smem_a};\n");
    s += &format!("    add.u32 %bufwBg,%bufwBg,{tile_b};\n    setp.ge.u32 %pmore,%bufwBg,{smem_b};\n    @%pmore sub.u32 %bufwBg,%bufwBg,{smem_b};\n");
    s += &format!("    add.u32 %bufwBu,%bufwBu,{tile_b};\n    setp.ge.u32 %pmore,%bufwBu,{smem_b};\n    @%pmore sub.u32 %bufwBu,%bufwBu,{smem_b};\n");
    s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

    // Epilogue: gate = act(gate [+bg]); up = up [+bu]; out = gate ⊙ up. The lane's 4 D regs span two
    // columns (d0,d2 @ gcol; d1,d3 @ gcol+1) and rows {grp, grp+8} — same map for %dg and %du, so the
    // product is register-for-register and the per-column bias hits the matching pair.
    s += &format!("KEND_{name}:\n");
    for mi in 0..tm {
        for ni in 0..tn {
            s += &format!("    add.u32 %grow,%baseRow,%warpMrow;\n    add.u32 %grow,%grow,{};\n    add.u32 %grow,%grow,%grp;\n", mi * 16);
            s += &format!("    add.u32 %gcol,%baseCol,%warpNcol;\n    add.u32 %gcol,%gcol,{};\n    add.u32 %gcol,%gcol,%tg2;\n", ni * 8);
            if bias {
                s += "    mul.wide.u32 %off,%gcol,4;\n    add.s64 %cptr,%BiasG,%off;\n";
                s += "    ld.global.f32 %biasg0,[%cptr];\n    ld.global.f32 %biasg1,[%cptr+4];\n";
                s += &format!("    add.f32 %dg{mi}_{ni}_0,%dg{mi}_{ni}_0,%biasg0;\n    add.f32 %dg{mi}_{ni}_1,%dg{mi}_{ni}_1,%biasg1;\n");
                s += &format!("    add.f32 %dg{mi}_{ni}_2,%dg{mi}_{ni}_2,%biasg0;\n    add.f32 %dg{mi}_{ni}_3,%dg{mi}_{ni}_3,%biasg1;\n");
                s += "    add.s64 %cptr,%BiasU,%off;\n";
                s += "    ld.global.f32 %biasu0,[%cptr];\n    ld.global.f32 %biasu1,[%cptr+4];\n";
                s += &format!("    add.f32 %du{mi}_{ni}_0,%du{mi}_{ni}_0,%biasu0;\n    add.f32 %du{mi}_{ni}_1,%du{mi}_{ni}_1,%biasu1;\n");
                s += &format!("    add.f32 %du{mi}_{ni}_2,%du{mi}_{ni}_2,%biasu0;\n    add.f32 %du{mi}_{ni}_3,%du{mi}_{ni}_3,%biasu1;\n");
            }
            if !matches!(gate_act, Act::None) {
                for r in 0..4 {
                    s += &gate_act.epilogue(&format!("%dg{mi}_{ni}_{r}"));
                }
            }
            for r in 0..4 {
                s += &format!("    mul.f32 %dg{mi}_{ni}_{r},%dg{mi}_{ni}_{r},%du{mi}_{ni}_{r};\n");
            }
            s += "    mul.lo.s32 %tmp,%grow,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
            s += &format!("    st.global.f32 [%cptr],%dg{mi}_{ni}_0;\n    st.global.f32 [%cptr+4],%dg{mi}_{ni}_1;\n");
            s += "    add.u32 %tmp,%grow,8;\n    mul.lo.s32 %tmp,%tmp,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr2,%C,%off;\n";
            s += &format!("    st.global.f32 [%cptr2],%dg{mi}_{ni}_2;\n    st.global.f32 [%cptr2+4],%dg{mi}_{ni}_3;\n");
        }
    }
    s += "    ret;\n}\n";
    s
}

/// Number of independent accumulator fragments the roofline kernel keeps in flight (ILP to hide MMA
/// latency so the loop measures tensor-core *throughput*, not the dependent-chain latency).
pub const ROOFLINE_ACC: usize = 4;

/// A compute-bound fp16 tensor-core **roofline** kernel: each warp loads ONE A-fragment and ONE
/// B-fragment from global (so ptxas can't constant-fold), then loops `iters` times issuing
/// `ROOFLINE_ACC` independent `wmma.mma`s that reuse those fragments. One load + `iters·ACC` MMAs +
/// one store ⇒ effectively zero memory traffic in the hot loop, so the achieved rate is the practical
/// tensor-core ceiling on this (power-capped) GPU. This is an *internal* same-run ceiling; the real
/// peer is now cuBLAS (see `baselines.rs`), which in practice matches/exceeds this roofline — so treat
/// it as a soft under-estimate, not a hard wall.
/// FLOPs = warps · iters · ACC · (16·16·16·2). Entry `wmma_roofline_f16`; launch block=32 (one warp).
fn roofline_entry() -> String {
    let (ty, nab, nacc) = ("f16", 8usize, ROOFLINE_ACC);
    let mut s = String::new();
    s += ".visible .entry wmma_roofline_f16(\n    .param .u32 pIters,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{\n";
    s += "    .reg .pred %p0;\n    .reg .b32 %iters,%i,%ld;\n";
    let mut decl_c = String::new();
    for j in 0..nacc {
        for r in 0..8 {
            decl_c += &format!("%c{j}_{r},");
        }
    }
    s += &format!("    .reg .f32 {};\n", decl_c.trim_end_matches(','));
    let mut decl_ab = String::new();
    for r in 0..nab {
        decl_ab += &format!("%a{r},");
    }
    for r in 0..nab {
        decl_ab += &format!("%b{r},");
    }
    s += &format!("    .reg .b32 {};\n", decl_ab.trim_end_matches(','));
    s += "    .reg .b64 %A,%B,%C,%off,%cptr;\n";
    s += "    ld.param.u32 %iters,[pIters];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    s += "    mov.u32 %ld,16;\n";
    // Load one A (row) and one B (col) fragment, leading dim 16.
    let ra = veclist("a", nab);
    let rb = veclist("b", nab);
    s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%A], %ld;\n");
    s += &format!("    wmma.load.b.sync.aligned.m16n16k16.col.{ty} {rb}, [%B], %ld;\n");
    for j in 0..nacc {
        for r in 0..8 {
            s += &format!("    mov.f32 %c{j}_{r},0f00000000;\n");
        }
    }
    s += "    mov.u32 %i,0;\nRLOOP:\n    setp.ge.u32 %p0,%i,%iters;\n    @%p0 bra REND;\n";
    for j in 0..nacc {
        let cc = veclist(&format!("c{j}_"), 8);
        s += &format!(
            "    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {cc}, {ra}, {rb}, {cc};\n"
        );
    }
    s += "    add.u32 %i,%i,1;\n    bra RLOOP;\nREND:\n";
    // Fold the ACC accumulators into c0 so none are dead-code-eliminated (keeps every chain live).
    for r in 0..8 {
        for j in 1..nacc {
            s += &format!("    add.f32 %c0_{r},%c0_{r},%c{j}_{r};\n");
        }
    }
    // Store c0 to this warp's own 256-f32 slot (block=32 ⇒ warp id = ctaid.x).
    s += "    mov.u32 %i,%ctaid.x;\n    mul.wide.u32 %off,%i,1024;\n    add.s64 %cptr,%C,%off;\n";
    let cc0 = veclist("c0_", 8);
    s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc0}, %ld;\n");
    s += "    ret;\n}\n";
    s
}

/// **Static-shape fp16 SMEM-staged GEMM** — the M1 compile-time-shapes lever for fp16 (the twin of the
/// int8/int4 static kernels). Bakes M/N/K into the `wmma_nt_f16_sm` tile so ptxas constant-folds the
/// hot-loop strides (`×K`/`×N`) and knows the K trip count; 64×64 (`wmma_nt_f16_sm_static`) or 128×128
/// (`wmma_nt_f16_sm128_static`) per `use_128`. Returns an owned per-shape module (the caller caches it
/// under a shape-keyed key / raw-loads it). Identical codegen to the dynamic `_sm` kernel ⇒ **bit-exact**.
pub fn wmma_f16_sm_static_ptx(m: usize, n: usize, k: usize, use_128: bool) -> String {
    let mut s = String::from(HDR_SM80);
    if use_128 {
        s += &entry_smem(
            "wmma_nt_f16_sm128_static",
            "f16",
            SM128_BM,
            SM128_BN,
            SM128_WARPS_M,
            SM128_WARPS_N,
            Some((m, n, k)),
        );
    } else {
        s += &entry_smem(
            "wmma_nt_f16_sm_static",
            "f16",
            SM_BM,
            SM_BN,
            SM_WARPS_M,
            SM_WARPS_N,
            Some((m, n, k)),
        );
    }
    s
}

/// Entry name for [`wmma_f16_sm_static_ptx`] at the matching `use_128`.
pub fn wmma_f16_sm_static_entry(use_128: bool) -> &'static str {
    if use_128 {
        "wmma_nt_f16_sm128_static"
    } else {
        "wmma_nt_f16_sm_static"
    }
}

/// fp16 tensor-core GEMM module — the whole dispatched fp16 family in one module, in emission order:
/// `wmma_nt_f16` (single 16×16 tile/warp, any 16-multiple dims), `wmma_nt_f16_mt` (2×4 tiles/warp =
/// 32×64, fragment-reuse), the SMEM-staged `_sm`/`_sm128` and their `cp.async` double-buffered `_db`
/// twins, every [`PIPE_VARIANTS`] multi-stage pipe (WMMA or `mma.sync`), the `_swz`/`_w22swz` no-pad
/// swizzle twins of the `mma.sync` workhorse, and the fused-epilogue families (`_bias[_act]`,
/// `_bias_residual`, the `mma_nt_f16_128x64_gate_*` gated-FFN tiles). The
/// `every_dispatched_tensor_core_entry_is_defined` gate pins the names `gpu.rs` looks up.
pub fn wmma_f16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        m += &entry("wmma_nt_f16", "f16", 1, 1);
        m += &entry("wmma_nt_f16_mt", "f16", TM_TILES, TN_TILES);
        m += &entry_smem(
            "wmma_nt_f16_sm",
            "f16",
            SM_BM,
            SM_BN,
            SM_WARPS_M,
            SM_WARPS_N,
            None,
        );
        // Single-buffered 128×128 tile (no cp.async pipeline). At L2-spilling sizes the double-buffered
        // kernels are occupancy-bound and LOSE to the un-pipelined ones (measured: _sm beats _sm_db at
        // 2048³); the big tile halves redundant inter-CTA traffic while single-buffering avoids the
        // pipeline's extra SMEM + bar.syncs — the large-GEMM candidate the clean scoreboard motivates.
        m += &entry_smem(
            "wmma_nt_f16_sm128",
            "f16",
            SM128_BM,
            SM128_BN,
            SM128_WARPS_M,
            SM128_WARPS_N,
            None,
        );
        m += &entry_smem_db(
            "wmma_nt_f16_sm_db",
            "f16",
            SM_BM,
            SM_BN,
            SM_WARPS_M,
            SM_WARPS_N,
            Act::None,
            false,
            false,
        );
        m += &entry_smem_db(
            "wmma_nt_f16_sm128_db",
            "f16",
            SM128_BM,
            SM128_BN,
            SM128_WARPS_M,
            SM128_WARPS_N,
            Act::None,
            false,
            false,
        );
        // Multi-stage `cp.async` pipeline variants — the large-GEMM-cliff lever (deeper prefetch window +
        // wider staged BK than the 2-stage `_sm*_db`). Swept by `gemm_pipe_sweep`; the winner per size is
        // dispatched from `gemm_nt_f16`. All share the precision-generic `entry_smem_pipe` generator.
        for v in PIPE_VARIANTS {
            m += &if v.mma {
                entry_mma_pipe(
                    v.name,
                    "f16",
                    v.bm,
                    v.bn,
                    v.bk,
                    v.wm,
                    v.wn,
                    v.stages,
                    v.raster,
                    v.pad,
                    Act::None,
                    false,
                    false,
                    false,
                    0,
                    Store::Scalar,
                )
            } else {
                entry_smem_pipe(
                    v.name,
                    "f16",
                    v.bm,
                    v.bn,
                    v.bk,
                    v.wm,
                    v.wn,
                    v.stages,
                    v.raster,
                    Act::None,
                    false,
                    false,
                )
            };
        }
        // **`ldmatrix` + XOR-swizzle + no-pad workhorse** (`_swz`) — the full CUTLASS recipe for the
        // HBM-bound 4096³: dropping the padding shrinks the tile (32 KiB ⇒ 3 CTAs/SM vs the padded 40 KiB's
        // 2), and the XOR swizzle keeps both the cp.async stores and the `ldmatrix.x4`/`.x2` gathers
        // conflict-free at that higher occupancy. Numerically identical to the hand-placed workhorse; swept
        // same-run by `mma_swizzle_vs_handplaced` (the padded-`ldmatrix` variant already LOST — this tests
        // whether the proper swizzle+occupancy pairing reclaims it where the biggest deficit lives).
        {
            let wh = pipe_variant("mma_nt_f16_128_bk32_s2_r16");
            m += &entry_mma_pipe(
                &format!("{}_swz", wh.name),
                "f16",
                wh.bm,
                wh.bn,
                wh.bk,
                wh.wm,
                wh.wn,
                wh.stages,
                wh.raster,
                wh.pad,
                Act::None,
                false,
                false,
                true,
                0,
                Store::Scalar,
            );
            // **w22 swizzle workhorse** — the 2×2 warp grid (vs the w24 base's 2×4) gives each warp a 64×64
            // tile = 32 mma/warp (2× the ILP) AND 128 threads/CTA ⇒ 3 CTAs/SM (vs 2). Kept emitted +
            // correctness-gated (`gemm_cliff_w22swz_matches_reference`) and swept by `gemm_cliff_ab`, but
            // **NOT dispatched**: a clean round-robin best-of-N re-measure showed w22 is only a noise-level
            // tie with the w24 swz at 4096³ (0.97–1.02×) and *loses* at 2048³, so `gemm_nt_f16` routes the
            // whole A+B ≥ 16 MB regime to the w24 swz (see the dispatch comment in `gemm_nt_f16`).
            m += &entry_mma_pipe(
                &format!("{}_w22swz", wh.name),
                "f16",
                wh.bm,
                wh.bn,
                wh.bk,
                2,
                2,
                wh.stages,
                wh.raster,
                wh.pad,
                Act::None,
                false,
                false,
                true,
                0,
                Store::Scalar,
            );
        }
        // Fused epilogues on the **deep WMMA pipe `pipe_64_s6`** — the ≤1024³ GEMM champion. The `mma.sync`
        // workhorse the *other* fused epilogues ride is beaten there (`pipe_64_s6` ~90% of cuBLAS vs the
        // workhorse's ~80%), so a fused epilogue built on the workhorse LOSES at 1024³ (its GEMM deficit
        // exceeds the saved round-trip). Built on `pipe_64_s6` instead, the fused FFN/Linear/down-proj WINS
        // there too. WMMA's fragment column map is opaque, so the per-column bias routes through `smemA`
        // scratch (free post-K-loop) re-read by explicit (row,col); the residual seeds via wmma.load.c.
        let p64 = pipe_variant("wmma_nt_f16_pipe_64_s6");
        for (suffix, act) in [
            ("bias", Act::None),
            ("bias_relu", Act::Relu),
            ("bias_silu", Act::Silu),
            ("bias_gelu", Act::Gelu),
        ] {
            m += &entry_smem_pipe(
                &format!("{}_{suffix}", p64.name),
                "f16",
                p64.bm,
                p64.bn,
                p64.bk,
                p64.wm,
                p64.wn,
                p64.stages,
                p64.raster,
                act,
                true,
                false,
            );
        }
        // `out = x·Wᵀ + bias + residual` (down-proj / attention output-proj) on the ≤1024³ champion: the
        // residual seeds the accumulator, the bias adds in the store-back epilogue (no activation).
        m += &entry_smem_pipe(
            &format!("{}_bias_residual", p64.name),
            "f16",
            p64.bm,
            p64.bn,
            p64.bk,
            p64.wm,
            p64.wn,
            p64.stages,
            p64.raster,
            Act::None,
            true,
            true,
        );
        // Fused-epilogue variants on the **fast `mma.sync` workhorse** — the structural beat-cuBLAS lever.
        // `C = act(x·Wᵀ + bias)` is the canonical nn.Linear / FFN epilogue: cuBLAS computes only `x·Wᵀ`, so
        // the bias add + activation need a *second* kernel that round-trips C through HBM. Here they fold
        // into the store, applied register-level to the f32 accumulators (the mma kernel's D-fragment
        // column map is known, so bias[col] is added without any SMEM scratch the WMMA path needs). Built
        // on the r16 mma champion (the fastest base): bias alone (Linear) + bias·{relu,silu,gelu} (FFN).
        let wh = pipe_variant("mma_nt_f16_128_bk32_s2_r16");
        for (suffix, act) in [
            ("bias", Act::None),
            ("bias_relu", Act::Relu),
            ("bias_silu", Act::Silu),
            ("bias_gelu", Act::Gelu),
        ] {
            m += &entry_mma_pipe(
                &format!("{}_{suffix}", wh.name),
                "f16",
                wh.bm,
                wh.bn,
                wh.bk,
                wh.wm,
                wh.wn,
                wh.stages,
                wh.raster,
                wh.pad,
                act,
                true,
                false,
                false,
                0,
                Store::Scalar,
            );
            // The **no-pad swizzle** twin of each fused-epilogue kernel (`..._swz_bias{,_relu,_silu,_gelu}`).
            // The bias-add + activation apply register-level to the `mma.sync` D-fragments — independent of
            // how A/B were staged in SMEM — so the swizzle composes orthogonally and the result is *bit-
            // identical* to the padded epilogue, but it inherits the swizzle GEMM's 1.13–1.23× same-run speed
            // (hardware `ldmatrix` fragment loads vs the padded base's manual `ld.shared.b32`, at equal
            // occupancy). This is the fastest base for the beat-cuBLAS fused Linear/FFN:
            // `gemm_nt_f16_mma_bias{,_relu,_silu,_gelu}` route here.
            m += &entry_mma_pipe(
                &format!("{}_swz_{suffix}", wh.name),
                "f16",
                wh.bm,
                wh.bn,
                wh.bk,
                wh.wm,
                wh.wn,
                wh.stages,
                wh.raster,
                0,
                act,
                true,
                false,
                true,
                0,
                Store::Scalar,
            );
        }
        // Fused **bias + residual** (no activation) on the same fast mma workhorse — the transformer
        // down-proj / attention output-proj sublayer output `out = x·Wᵀ + bias + residual` (the residual
        // added to the post-bias accumulators before the store). This folds BOTH the bias-add and the
        // residual-add kernels a cuBLAS chain runs separately into the GEMM store — the two `+residual`
        // points in every transformer block, the megakernel-beats-call-chain lever on the fastest base.
        m += &entry_mma_pipe(
            &format!("{}_bias_residual", wh.name),
            "f16",
            wh.bm,
            wh.bn,
            wh.bk,
            wh.wm,
            wh.wn,
            wh.stages,
            wh.raster,
            wh.pad,
            Act::None,
            true,
            true,
            false,
            0,
            Store::Scalar,
        );
        // The no-pad swizzle twin — the bias-add + residual-add apply register-level to the `mma.sync`
        // D-fragments (orthogonal to SMEM staging ⇒ bit-identical output) on the faster swz base, so the
        // down-proj / attention output-proj `out = x·Wᵀ + bias + residual` inherits the swz GEMM's
        // 1.13–1.23× same-run speed. `gemm_nt_f16_mma_bias_residual` routes here.
        m += &entry_mma_pipe(
            &format!("{}_swz_bias_residual", wh.name),
            "f16",
            wh.bm,
            wh.bn,
            wh.bk,
            wh.wm,
            wh.wn,
            wh.stages,
            wh.raster,
            0,
            Act::None,
            true,
            true,
            true,
            0,
            Store::Scalar,
        );
        // Fused **gated-FFN (GLU-family)** kernels `out = act(x·Wgᵀ) ⊙ (x·Wuᵀ)` — the SwiGLU/GeGLU gate
        // every modern LLM FFN runs, the fusion cuBLAS needs THREE kernels + two HBM round-trips for. The
        // 128×64 dual-B tile holds two accumulator sets at the SAME 64 D-regs/thread and the SAME 40 KiB
        // SMEM as the single-B 128×128 workhorse (register- and SMEM-neutral), sharing one staged x tile
        // between Wg and Wu. silu→SwiGLU, gelu→GeGLU, none→bilinear GLU; `_bias` adds the per-column biases.
        for (suffix, act, gbias) in [
            ("gate_silu", Act::Silu, false),
            ("gate_gelu", Act::Gelu, false),
            ("gate_glu", Act::None, false),
            ("gate_silu_bias", Act::Silu, true),
            ("gate_gelu_bias", Act::Gelu, true),
        ] {
            m += &entry_mma_gate(
                &format!("mma_nt_f16_128x64_{suffix}"),
                "f16",
                128,
                64,
                32,
                2,
                4,
                2,
                16,
                8,
                act,
                gbias,
                false,
            );
            // The no-pad `ldmatrix`+XOR-swizzle twin (`..._swz`), bit-identical to the padded gate. Emitted +
            // correctness-gated as a verified alternative, but **measured a wash-to-loss for this dual-B tile**
            // (the single-B GEMM-cliff swz win does NOT transfer — see `entry_mma_gate` / the same-run
            // `fused_swiglu_gate_vs_chain`), so `gemm_nt_f16_swiglu`/`_geglu` route to the PADDED base, not here.
            m += &entry_mma_gate(
                &format!("mma_nt_f16_128x64_{suffix}_swz"),
                "f16",
                128,
                64,
                32,
                2,
                4,
                2,
                16,
                8,
                act,
                gbias,
                true,
            );
        }
        // Fused activation epilogues — the beat-cuBLAS lever (cuBLAS can't fuse). 64-tile pipeline.
        // relu/silu/gelu cover the activations the FFN and classic CNN/MLP stacks actually use; silu in
        // particular fuses the SwiGLU FFN up-projection (`silu(x·W1ᵀ)`) into one kernel.
        for (suffix, act) in [
            ("relu", Act::Relu),
            ("silu", Act::Silu),
            ("gelu", Act::Gelu),
        ] {
            m += &entry_smem_db(
                &format!("wmma_nt_f16_sm_db_{suffix}"),
                "f16",
                SM_BM,
                SM_BN,
                SM_WARPS_M,
                SM_WARPS_N,
                act,
                false,
                false,
            );
        }
        // Fused **bias (+ activation)** epilogues `C = act(A·Bᵀ + bias)` — the canonical `nn.Linear`/FFN
        // form. The per-column bias needs the WMMA fragment's column index, so these route the tile
        // through SMEM and re-read by explicit (row,col); `bias` alone (`Act::None`) is affine Linear.
        for (suffix, act) in [
            ("bias", Act::None),
            ("bias_relu", Act::Relu),
            ("bias_silu", Act::Silu),
            ("bias_gelu", Act::Gelu),
        ] {
            m += &entry_smem_db(
                &format!("wmma_nt_f16_sm_db_{suffix}"),
                "f16",
                SM_BM,
                SM_BN,
                SM_WARPS_M,
                SM_WARPS_N,
                act,
                true,
                false,
            );
        }
        // Fused **residual** epilogue `C = A·Bᵀ + residual` — the transformer skip connection (both
        // `x + attn·Woᵀ` and `x + ffn(x)·W2ᵀ`). Seeds the f32 accumulator from the residual via
        // wmma.load.c so the add costs nothing extra; the library call-chain pays a separate add (or a
        // beta=1 pre-fill) launch + HBM round-trip for it. This is an M13 megakernel building block.
        m += &entry_smem_db(
            "wmma_nt_f16_sm_db_residual",
            "f16",
            SM_BM,
            SM_BN,
            SM_WARPS_M,
            SM_WARPS_N,
            Act::None,
            false,
            true,
        );
        m
    })
    .as_str()
}

/// bf16 tensor-core GEMM module: `wmma_nt_bf16` (single tile), `wmma_nt_bf16_mt` (fragment-reuse), the
/// [`PIPE_BF16`] `mma.sync` large-GEMM workhorse plus its `_swz`/`_w22swz` twins and `_bias*`/
/// `_bias_residual` fused epilogues, the `mma_nt_bf16_128x64_gate_*` gated-FFN tiles, the `cp.async`
/// double-buffered `wmma_nt_bf16_sm_db`, and the fused-epilogue `wmma_nt_bf16_sm_db_{relu,silu,gelu}`
/// / `_bias{,_relu,_silu,_gelu}`. Every generator is precision-generic (`entry_smem_db` /
/// `entry_mma_pipe` / `entry_mma_gate` key the fragment width and mma type off `ty`), so bf16 — the
/// dominant *training* precision — gets the same beat-the-cuBLAS-chain fusion as fp16.
pub fn wmma_bf16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        m += &entry("wmma_nt_bf16", "bf16", 1, 1);
        m += &entry("wmma_nt_bf16_mt", "bf16", TM_TILES, TN_TILES);
        // bf16 large-GEMM workhorse (mma.sync + padded conflict-free SMEM + r16 raster) — the cliff fix
        // carried to the training precision; `gemm_nt_bf16` dispatches A+B ≳ L2 here.
        let v = PIPE_BF16;
        m += &entry_mma_pipe(
            v.name,
            "bf16",
            v.bm,
            v.bn,
            v.bk,
            v.wm,
            v.wn,
            v.stages,
            v.raster,
            v.pad,
            Act::None,
            false,
            false,
            false,
            0,
            Store::Scalar,
        );
        // bf16 ldmatrix+XOR-swizzle+no-pad twin (`_swz`) — the HBM-bound-4096³ win carried to the training
        // dtype (the swz path is dtype-agnostic; `gemm_nt_bf16` regime-dispatches it for A+B ≳ 2×L2).
        m += &entry_mma_pipe(
            &format!("{}_swz", v.name),
            "bf16",
            v.bm,
            v.bn,
            v.bk,
            v.wm,
            v.wn,
            v.stages,
            v.raster,
            v.pad,
            Act::None,
            false,
            false,
            true,
            0,
            Store::Scalar,
        );
        // bf16 w22 swizzle twin: 2×2 warp grid = 32 mma/warp + 3 CTAs/SM. Emitted + correctness-gated but
        // **NOT dispatched** — like its fp16 twin, w22 only noise-ties w24 at 4096³ and loses at 2048³, so
        // `gemm_nt_bf16` routes the whole A+B ≥ 16 MB regime to the w24 `_swz` (see the gpu.rs dispatch).
        m += &entry_mma_pipe(
            &format!("{}_w22swz", v.name),
            "bf16",
            v.bm,
            v.bn,
            v.bk,
            2,
            2,
            v.stages,
            v.raster,
            v.pad,
            Act::None,
            false,
            false,
            true,
            0,
            Store::Scalar,
        );
        // Fused-epilogue variants on the **fast bf16 mma workhorse** — the register-level `act(x·Wᵀ+bias)`
        // (bias added to the f32 accumulators via the known D-fragment column map, no SMEM scratch) carried
        // to the training dtype. The bf16 twin of the fp16 `mma_nt_f16_128_bk32_s2_r16_bias*` champions.
        for (suffix, act) in [
            ("bias", Act::None),
            ("bias_relu", Act::Relu),
            ("bias_silu", Act::Silu),
            ("bias_gelu", Act::Gelu),
        ] {
            m += &entry_mma_pipe(
                &format!("{}_{suffix}", v.name),
                "bf16",
                v.bm,
                v.bn,
                v.bk,
                v.wm,
                v.wn,
                v.stages,
                v.raster,
                v.pad,
                act,
                true,
                false,
                false,
                0,
                Store::Scalar,
            );
            // The no-pad swizzle twin (`..._swz_bias*`) — the register-level epilogue composes orthogonally
            // with the SMEM swizzle (bit-identical output) and inherits the swizzle GEMM's 1.13–1.23× speed.
            // `gemm_nt_bf16_mma_bias{,_relu,_silu,_gelu}` route here (the training-dtype fused Linear/FFN base).
            m += &entry_mma_pipe(
                &format!("{}_swz_{suffix}", v.name),
                "bf16",
                v.bm,
                v.bn,
                v.bk,
                v.wm,
                v.wn,
                v.stages,
                v.raster,
                0,
                act,
                true,
                false,
                true,
                0,
                Store::Scalar,
            );
        }
        // bf16 fused bias + residual (training down-proj / output-proj): out = x·Wᵀ + bias + residual.
        m += &entry_mma_pipe(
            &format!("{}_bias_residual", v.name),
            "bf16",
            v.bm,
            v.bn,
            v.bk,
            v.wm,
            v.wn,
            v.stages,
            v.raster,
            v.pad,
            Act::None,
            true,
            true,
            false,
            0,
            Store::Scalar,
        );
        // The no-pad swizzle twin (bit-identical register-level epilogue, faster swz base) — the training
        // down-proj / output-proj inherits the swz GEMM speed. `gemm_nt_bf16_mma_bias_residual` routes here.
        m += &entry_mma_pipe(
            &format!("{}_swz_bias_residual", v.name),
            "bf16",
            v.bm,
            v.bn,
            v.bk,
            v.wm,
            v.wn,
            v.stages,
            v.raster,
            0,
            Act::None,
            true,
            true,
            true,
            0,
            Store::Scalar,
        );
        // bf16 gated-FFN (GLU-family) gate `out = act(x·Wgᵀ) ⊙ (x·Wuᵀ)` — SwiGLU/GeGLU carried to the
        // training dtype (the dual-B generator is precision-generic, keying the mma type off `ty`).
        for (suffix, act, gbias) in [
            ("gate_silu", Act::Silu, false),
            ("gate_gelu", Act::Gelu, false),
            ("gate_glu", Act::None, false),
            ("gate_silu_bias", Act::Silu, true),
            ("gate_gelu_bias", Act::Gelu, true),
        ] {
            m += &entry_mma_gate(
                &format!("mma_nt_bf16_128x64_{suffix}"),
                "bf16",
                128,
                64,
                32,
                2,
                4,
                2,
                16,
                8,
                act,
                gbias,
                false,
            );
            // bf16 no-pad swizzle twin (`..._swz`), bit-identical; correctness-gated alternative. Like fp16,
            // measured a wash-to-loss for this dual-B tile, so `gemm_nt_bf16_swiglu`/`_geglu` use the padded base.
            m += &entry_mma_gate(
                &format!("mma_nt_bf16_128x64_{suffix}_swz"),
                "bf16",
                128,
                64,
                32,
                2,
                4,
                2,
                16,
                8,
                act,
                gbias,
                true,
            );
        }
        m += &entry_smem_db(
            "wmma_nt_bf16_sm_db",
            "bf16",
            SM_BM,
            SM_BN,
            SM_WARPS_M,
            SM_WARPS_N,
            Act::None,
            false,
            false,
        );
        for (suffix, act) in [
            ("relu", Act::Relu),
            ("silu", Act::Silu),
            ("gelu", Act::Gelu),
        ] {
            m += &entry_smem_db(
                &format!("wmma_nt_bf16_sm_db_{suffix}"),
                "bf16",
                SM_BM,
                SM_BN,
                SM_WARPS_M,
                SM_WARPS_N,
                act,
                false,
                false,
            );
        }
        // Fused **bias (+ activation)** epilogues for bf16 too — the bias path acts on the f32
        // accumulator (dtype-independent), so the same SMEM-staged store-back gives the bf16 training
        // dtype the canonical `act(x·Wᵀ + bias)` Linear/FFN fusion. `bias` alone = affine Linear.
        for (suffix, act) in [
            ("bias", Act::None),
            ("bias_relu", Act::Relu),
            ("bias_silu", Act::Silu),
            ("bias_gelu", Act::Gelu),
        ] {
            m += &entry_smem_db(
                &format!("wmma_nt_bf16_sm_db_{suffix}"),
                "bf16",
                SM_BM,
                SM_BN,
                SM_WARPS_M,
                SM_WARPS_N,
                act,
                true,
                false,
            );
        }
        m
    })
    .as_str()
}

/// fp16 tensor-core roofline module — entry `wmma_roofline_f16` (see [`roofline_entry`]).
pub fn roofline_f16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(HDR_SM80);
        m += &roofline_entry();
        m
    })
    .as_str()
}

/// Per-warp tile grid for the multi-tile (fast) kernels: 2 rows × 4 cols of 16×16 tiles = 32×64.
pub const TM_TILES: usize = 2;
pub const TN_TILES: usize = 4;
/// Per-warp output tile dims (the multi-tile kernel requires M%WARP_M==0 and N%WARP_N==0).
pub const WARP_M: usize = 16 * TM_TILES;
pub const WARP_N: usize = 16 * TN_TILES;

#[cfg(test)]
mod tests {
    use super::*;

    /// Split a straight-line PTX instruction into `(opcode, operands)`, dropping the `;` and all
    /// whitespace the two spellings happen to differ in.
    fn split_insn(line: &str) -> (String, Vec<String>) {
        let l = line.trim().trim_end_matches(';');
        let (op, rest) = l.split_once(char::is_whitespace).unwrap_or((l, ""));
        (
            op.to_string(),
            rest.split(',')
                .map(|o| o.trim().to_string())
                .filter(|o| !o.is_empty())
                .collect(),
        )
    }

    /// Canonicalize an activation body so two spellings of the *same dataflow* compare equal:
    /// registers are renamed by role -- `%in` is the live-in value, `%t0..` the scratch registers in
    /// first-definition order, and the destination of the last instruction is `%out` (the standalone
    /// vmath entry leaves its result wherever it lands and stores it, while the fused epilogue writes
    /// back into the accumulator). Immediates (`0f...`) are compared verbatim, so a changed constant
    /// still fails.
    fn canon_act_body(lines: &[&str], live_in: &str) -> Vec<String> {
        let mut map: Vec<(String, String)> = vec![(live_in.to_string(), "%in".to_string())];
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let (op, ops) = split_insn(line);
            let last = i + 1 == lines.len();
            let mut canon_ops = Vec::new();
            for (j, o) in ops.iter().enumerate() {
                if !o.starts_with('%') {
                    canon_ops.push(o.clone());
                } else if last && j == 0 {
                    canon_ops.push("%out".to_string());
                } else if let Some((_, v)) = map.iter().find(|(k, _)| k == o) {
                    canon_ops.push(v.clone());
                } else {
                    let v = format!("%t{}", map.len() - 1);
                    map.push((o.clone(), v.clone()));
                    canon_ops.push(v);
                }
            }
            out.push(format!("{op} {}", canon_ops.join(",")));
        }
        out
    }

    /// The instruction lines of `ptx::vmath_ptx`'s `name` entry between the input load and the output
    /// store, plus the register the store reads (which must be what the last instruction wrote).
    fn vmath_body(name: &str) -> Vec<String> {
        let ptx = crate::ptx::vmath_ptx();
        let at = ptx
            .find(&format!(".visible .entry {name}("))
            .expect("entry exists");
        let body = &ptx[at..];
        let body = &body[..body.find("\nDONE:").expect("entry has a DONE label")];
        let start = body.find("ld.global.f32").expect("entry loads x[i]");
        let tail = &body[start..];
        let lines: Vec<&str> = tail
            .lines()
            .skip(1)
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        let (store, act) = lines.split_last().expect("at least the store");
        let (op, ops) = split_insn(store);
        assert_eq!(
            op, "st.global.f32",
            "{name}: the entry must end in the output store"
        );
        let result = ops[1].clone();
        let (last_op, last_ops) =
            split_insn(act.last().expect("at least one activation instruction"));
        assert_eq!(
            last_ops[0], result,
            "{name}: `{last_op}` must write the register the store reads"
        );
        act.iter().map(|l| l.to_string()).collect()
    }

    /// U4 mirror gate. `Act::epilogue` (the fused GEMM/fp8 epilogue) and `ptx::vmath_ptx`'s standalone
    /// relu/silu/gelu entries are two copies of one semantic rule: the same .wk source reaches EITHER,
    /// depending only on whether the `sgemm_nt_epi` epilogue recognizer fired. The doc comment on
    /// `Act::epilogue` claims they use "the exact same formulas + constants", but nothing enforced it,
    /// so improving one copy in isolation (e.g. moving vmath's gelu to the erf form) would make the
    /// same program produce different numbers depending on the recognizer -- and the fused kernel's
    /// own 5e-2-tolerance gate compares it to a Rust tanh-form reference, so it would still pass.
    /// Compared as canonical dataflow, so a swapped operand, a changed constant or a different opcode
    /// sequence all fail here.
    #[test]
    fn fused_activation_epilogue_matches_the_standalone_vmath_kernel() {
        for (name, act) in [
            ("relu", Act::Relu),
            ("silu", Act::Silu),
            ("gelu", Act::Gelu),
        ] {
            let standalone = vmath_body(name);
            let standalone: Vec<&str> = standalone.iter().map(|s| s.as_str()).collect();
            let fused = act.epilogue("%acc");
            let fused: Vec<&str> = fused
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect();
            assert_eq!(
                canon_act_body(&standalone, "%f1"),
                canon_act_body(&fused, "%acc"),
                "{name}: the fused epilogue and the standalone vmath kernel have drifted -- a program \
                 would compute a different answer depending on whether the epilogue recognizer fired"
            );
        }
        assert_eq!(
            Act::None.epilogue("%acc"),
            "",
            "the plain GEMM must emit no epilogue"
        );
    }

    /// The full text of one `.visible .entry` — from its declaration up to the next entry (or the end
    /// of the module). Entries are emitted back-to-back, so this is an exact slice of what the driver
    /// JIT sees for that kernel, which is what the byte-identity gates compare.
    fn entry_of(ptx: &str, name: &str) -> String {
        let at = ptx
            .find(&format!(".visible .entry {name}("))
            .unwrap_or_else(|| panic!("entry `{name}` is not defined in this module"));
        let rest = &ptx[at..];
        let end = rest[1..]
            .find(".visible .entry ")
            .map_or(rest.len(), |i| i + 1);
        rest[..end].to_string()
    }

    /// FNV-1a 64. A pinned `(len, hash)` pair identifies a PTX string as tightly as the string itself
    /// for regression purposes, and unlike an inline golden copy it costs six lines instead of a
    /// megabyte. Hand-rolled so the pin needs no dev-dependency and cannot drift with one.
    fn fnv1a64(s: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Every `.visible .entry <name>(` defined in `ptx`, in order.
    fn entry_names(ptx: &str) -> Vec<&str> {
        ptx.match_indices(".visible .entry ")
            .map(|(i, m)| {
                let rest = &ptx[i + m.len()..];
                &rest[..rest
                    .find('(')
                    .expect("an entry declaration opens its param list")]
            })
            .collect()
    }

    /// B1: one non-ASCII byte anywhere in an emitted module is a `ptxas fatal` on this box. `Act`'s
    /// doc comments, the assert messages and the CLIFF_VARIANTS prose in this file are full of
    /// `.`/`x`-style math characters sitting right next to the `format!`s that build the kernels, and
    /// a GPU-less `cargo test` never loads a module -- so nothing but this test keeps the emitted
    /// text loadable.
    #[test]
    fn every_tensor_core_module_is_pure_ascii() {
        let modules: [(&str, &str); 6] = [
            ("wmma_f16_ptx", wmma_f16_ptx()),
            ("wmma_bf16_ptx", wmma_bf16_ptx()),
            ("gemm_cliff_ptx", gemm_cliff_ptx()),
            ("gemm_deep_ptx", gemm_deep_ptx()),
            ("roofline_f16_ptx", roofline_f16_ptx()),
            (
                "wmma_f16_sm_static_ptx",
                &wmma_f16_sm_static_ptx(128, 128, 128, false),
            ),
        ];
        for (what, ptx) in modules {
            if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
                panic!(
                    "{what}: PTX must be pure ASCII (ptxas fatal otherwise) -- line {}: {line}",
                    i + 1
                );
            }
            assert_eq!(
                ptx.matches('{').count(),
                ptx.matches('}').count(),
                "{what}: unbalanced braces"
            );
            let names = entry_names(ptx);
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                names.len(),
                "{what}: duplicate .visible .entry name"
            );
        }
    }

    /// **ZERO REGRESSION on the ≤48 KiB path: the deep grid's s2/s3 rows ARE the shipped cliff kernels,
    /// byte for byte.** The dynamic-SMEM migration's whole promise is that adding a budget parameter and
    /// an extern-window arm changes *nothing* about the kernels already measured on this card. Proven,
    /// not sampled: regenerate `deep_swz_128_s2`/`_s3`, rename the entries to `cliff_swz_s2`/`_s3`, and
    /// demand string equality with what `gemm_cliff_ptx` emits for those rows. (`cliff_swz_s3` is
    /// 3·(128+128)·32·2 = 48 KiB *exactly* — the last depth expressible without the window, which is why
    /// it is the anchor.) Byte-identity also keeps the cubin cache, keyed on PTX text, warm.
    #[test]
    fn deep_grid_s2_s3_are_the_shipped_cliff_kernels() {
        let cliff = gemm_cliff_ptx();
        let deep = gemm_deep_ptx();
        for (deep_name, cliff_name) in [
            ("deep_swz_128_s2", "cliff_swz_s2"),
            ("deep_swz_128_s3", "cliff_swz_s3"),
        ] {
            let v = deep_variant(deep_name);
            assert_eq!(
                v.smem_mode(),
                SmemMode::Static,
                "{deep_name} must stay on the static path"
            );
            assert_eq!(
                entry_of(deep, deep_name).replace(deep_name, cliff_name),
                entry_of(cliff, cliff_name),
                "{deep_name}: the budget-parameterized generator no longer reproduces the shipped \
                 `{cliff_name}` byte for byte — the <=48 KiB path is NOT allowed to move"
            );
        }
    }

    /// **EVERY ALREADY-SHIPPED CONFIGURATION STILL EMITS BYTE-IDENTICAL PTX.** The sibling test above
    /// pins two entries against each other *inside this build*, which cannot see a change that moves
    /// both. This one pins the emitted text against values **recorded from the tree before the CTA tile
    /// lattice was widened**, so it is a real before/after comparison rather than a self-consistency
    /// check: `(byte length, FNV-1a 64)` of every dispatched module whole, plus of each pre-existing
    /// [`PIPE_DEEP_VARIANTS`] entry individually (the deep module's *text* legitimately grows as rows
    /// are added, so only its entries can be pinned).
    ///
    /// Why it matters beyond tidiness: the cubin cache is keyed on PTX text, every published A/B number
    /// on this card was taken against these exact strings, and the whole promise of "add lattice rows,
    /// change nothing that ships" is unfalsifiable by inspection — the generator is one 450-line
    /// function shared by ~24 call sites, and a `tm`/`tn`-dependent edit would move dozens of entries at
    /// once while every structural gate stayed green.
    ///
    /// **Baseline provenance:** printed by a throwaway `#[test]` on the pre-change worktree at
    /// `fe1502f` (`cargo test -p wukong_codegen_gpu --features gpu --lib -- --nocapture`). Regenerating
    /// these numbers is how you *lose* the guarantee — a row that changes shipped text must be
    /// justified in the commit body, not re-pinned quietly.
    #[test]
    fn shipped_tensor_core_ptx_is_byte_identical_to_the_pre_widening_baseline() {
        let s64 = wmma_f16_sm_static_ptx(256, 256, 256, false);
        let s128 = wmma_f16_sm_static_ptx(256, 256, 256, true);
        // (what, text, byte length, FNV-1a 64) — whole modules, none of which gained a row.
        let mut modules = 0usize;
        let mut pin = |what: &str, ptx: &str, len: usize, hash: u64| {
            modules += 1;
            assert_eq!(ptx.len(), len, "{what}: PTX length moved");
            assert_eq!(
                fnv1a64(ptx),
                hash,
                "{what}: PTX text moved at unchanged length - a shipped kernel was rewritten"
            );
        };
        let (f16m, bf16m) = (wmma_f16_ptx(), wmma_bf16_ptx());
        let (cliffm, roofm) = (gemm_cliff_ptx(), roofline_f16_ptx());
        pin("wmma_f16_ptx", f16m, 1_326_227, 0x13ea_55d4_469f_0eb6);
        pin("wmma_bf16_ptx", bf16m, 1_053_587, 0x8552_1e9a_c047_0e0e);
        pin("gemm_cliff_ptx", cliffm, 315_085, 0xee75_178c_0983_0e4a);
        pin("roofline_f16_ptx", roofm, 3_868, 0x429a_caaa_8174_d952);
        pin("sm_static_ptx/64", &s64, 7_844, 0x796c_f22f_119f_fb37);
        pin("sm_static_ptx/128", &s128, 12_805, 0xc000_e086_525a_9e5e);
        // The five deep rows that existed before the widening, entry by entry.
        let deep = gemm_deep_ptx();
        let entries: [(&str, usize, u64); 5] = [
            ("deep_swz_128_s2", 28_447, 0x5375_de64_013e_a40b),
            ("deep_swz_128_s3", 31_072, 0xfaf1_82c0_5d1a_0efa),
            ("deep_swz_128_s4", 33_619, 0xc031_3701_f7d7_aef3),
            ("deep_swz_128_s5", 36_251, 0xf638_f1a2_e57d_6b99),
            ("deep_swz_128x256_s3", 53_157, 0x3efc_639e_058f_fdaf),
        ];
        for (name, len, hash) in entries {
            let e = entry_of(deep, name);
            assert_eq!(e.len(), len, "{name}: entry length moved");
            assert_eq!(fnv1a64(&e), hash, "{name}: entry text moved");
        }
        eprintln!(
            "[gate] {modules} shipped modules + {} pre-existing deep entries are byte-identical to \
             the pre-widening baseline",
            entries.len()
        );
    }

    /// **The widened CTA tiles are generator-legal, register-bounded, and actually raise the thing they
    /// were widened for.** Purely arithmetic, so it runs with no device and it runs everywhere.
    ///
    /// Three claims, in the order they can kill the lever:
    ///
    /// 1. **Legality.** Re-derives the generator's own preconditions per row rather than trusting that
    ///    a panic would have fired: `swz` needs `bk == 32` exactly (the XOR phase is derived for
    ///    `nc = bk/8 = 4`) plus `wmr % 8 == 0` and `wnc % 8 == 0` (which is what makes the read-side
    ///    per-lane phase constant — `warpMrow`/`warpNcol` must vanish under `(·>>1) & 3`);
    ///    `bm % (16·wm) == 0`; `bn % (8·wn) == 0`; raster needs `bm`,`bn` powers of two; and both
    ///    staging tiles must be WHOLE multiples of `threads·8`, the truncating division that stages a
    ///    prefix of the tile and reads as a tolerance failure.
    /// 2. **The register wall, which is what D1 §2.3 says decides whether a 256-wide tile survives.**
    ///    `tm·tn·4` f32 accumulators live in registers for the whole mainloop, so a 128×256 or 256×128
    ///    tile carries **128** of them against the shipped tile's 64. Checked against the three hard ISA
    ///    limits (255 regs/thread, 65 536 regs/CTA, and the 64 K register file per SM at the row's
    ///    declared `min_ctas`), with the A/B fragments and a generous scratch allowance included —
    ///    because "it fits" is the entire premise, and Hopper adds **zero** registers over Ada.
    /// 3. **Intensity.** The point of the widening: CTA arithmetic intensity `bm·bn/(bm+bn)`, the FLOP
    ///    per byte of SMEM filled, must strictly exceed the shipped 128×128 tile's 64. Every wide row
    ///    reaches 85.33, which is what moves the H100 Act-1 binding ceiling off L2 fill (525 TFLOPS,
    ///    73% of cuBLAS@4096³) and onto the mma.sync issue rate (642, 90%).
    #[test]
    fn wide_tile_lattice_is_generator_legal_and_register_bounded() {
        /// Scratch registers beyond the accumulators and fragments: pointers, indices, predicates,
        /// swizzle phases and epilogue temporaries. Read off the generator's `.reg` declarations
        /// (~34 named b32/b64/pred plus the raster block) and rounded UP, so the budget is pessimistic.
        const SCRATCH: usize = 48;
        let i_cta = |bm: usize, bn: usize| (bm * bn) as f64 / (bm + bn) as f64;
        let shipped = i_cta(128, 128); // 64.0 — the tile the H100 ceiling of 73% belongs to
        let mut widened = 0usize;
        for (table, budget, label) in [
            (PIPE_DEEP_VARIANTS, DEEP_SMEM_BUDGET, "deep"),
            (PIPE_WIDE_VARIANTS, WIDE_SMEM_BUDGET, "wide"),
        ] {
            for v in table {
                let threads = v.threads();
                let (tm, tn) = (v.bm / (16 * v.wm), v.bn / (8 * v.wn));
                let (wmr, wnc) = (v.bm / v.wm, v.bn / v.wn);
                // 1. legality
                assert!(v.bk % 16 == 0 && (v.bk / 8).is_power_of_two(), "{}", v.name);
                assert!(v.bm % (16 * v.wm) == 0, "{}: bm % 16*wm", v.name);
                assert!(v.bn % (8 * v.wn) == 0, "{}: bn % 8*wn", v.name);
                assert!(v.wn.is_power_of_two(), "{}: warpId>>wn_shift", v.name);
                if v.swz {
                    assert_eq!(
                        v.bk, 32,
                        "{}: the swz XOR phase is derived for bk=32",
                        v.name
                    );
                    assert_eq!(v.pad, 0, "{}: swz forces ldp = bk", v.name);
                    assert_eq!(wmr % 8, 0, "{}: warpMrow must vanish under (>>1)&3", v.name);
                    assert_eq!(wnc % 8, 0, "{}: warpNcol must vanish under (>>1)&3", v.name);
                }
                if v.raster > 0 {
                    assert!(
                        v.bm.is_power_of_two() && v.bn.is_power_of_two(),
                        "{}: raster shifts by bm/bn",
                        v.name
                    );
                }
                for (which, dim) in [("A", v.bm), ("B", v.bn)] {
                    let chunks = dim * v.bk / (threads * 8);
                    assert!(
                        chunks >= 1 && chunks * threads * 8 == dim * v.bk,
                        "{}: {which} tile {dim}x{} is not a whole multiple of threads*8 = {}",
                        v.name,
                        v.bk,
                        threads * 8
                    );
                }
                assert!(
                    v.smem_bytes() <= budget,
                    "{}: {} B exceeds the {label} budget {budget} B",
                    v.name,
                    v.smem_bytes()
                );
                // 2. the register wall
                let accum = tm * tn * 4; // f32 D fragments, live across the whole mainloop
                let frags = tm * 4 + tn * 2; // b32 A (4/lane) + B (2/lane) fragments
                let per_thread = accum + frags + SCRATCH;
                assert!(
                    per_thread <= 255,
                    "{}: ~{per_thread} regs/thread exceeds the 255/thread ISA limit \
                     ({accum} accumulators at tm={tm} tn={tn})",
                    v.name
                );
                assert!(
                    per_thread * threads <= 65_536,
                    "{}: ~{} regs/CTA exceeds the 65536/CTA limit",
                    v.name,
                    per_thread * threads
                );
                let want = v.min_ctas.max(1);
                assert!(
                    per_thread * threads * want <= 65_536,
                    "{}: .minnctapersm {} cannot be met — {} threads x ~{per_thread} regs x {want} \
                     CTAs needs more than the 64K register file per SM, so ptxas would have to spill",
                    v.name,
                    v.min_ctas,
                    threads
                );
                // 3. intensity — the reason the tile was widened at all
                if v.bm.max(v.bn) > 128 {
                    widened += 1;
                    assert!(
                        i_cta(v.bm, v.bn) > shipped,
                        "{}: I_cta {} does not beat the shipped 128x128 tile's {shipped}",
                        v.name,
                        i_cta(v.bm, v.bn)
                    );
                    assert_eq!(
                        accum, 128,
                        "{}: the wide tiles carry 128 accumulators",
                        v.name
                    );
                }
                eprintln!(
                    "  {:<24} {:>3}x{:<3} s{} {:>3} KiB  tm={tm} tn={tn}  accum={accum} \
                     ~{per_thread} regs/thread  I_cta={:.2}",
                    v.name,
                    v.bm,
                    v.bn,
                    v.stages,
                    v.smem_bytes() / 1024,
                    i_cta(v.bm, v.bn)
                );
            }
        }
        assert!(
            widened >= 6,
            "the CTA-tile-width lever needs both orientations at several depths (only {widened} rows)"
        );
        eprintln!(
            "[gate] {widened} CTA tiles wider than 128 are generator-legal, fit 255 regs/thread at \
             128 accumulators, and raise I_cta 64.0 -> 85.33"
        );
    }

    /// **The deep grid's SMEM arithmetic, emission form, and window discipline (no GPU).** The four ways
    /// a dynamic-SMEM kernel goes wrong *silently*, each checked from the emitted text:
    ///   * the closed form `stages·(bm+bn)·(bk+pad)·2` and the 48 KiB boundary that splits the forms;
    ///   * the window is declared **once** and at **module scope** — two module-scope externs ALIAS (so a
    ///     second one would drop the B ring on top of the A ring), and the same line inside an entry body
    ///     is `CUDA_ERROR_INVALID_PTX`;
    ///   * a dynamic entry declares no static `.shared` array of its own (it would eat the same opt-in
    ///     ceiling) and a static entry never touches the window;
    ///   * the B ring starts at the constant `stages·tile_a`, and both rings are reached through the
    ///     SYMBOL — the window base is not 0 when an entry also has statics.
    ///
    /// Pins every [`PIPE_DEEP_VARIANTS`] row inside [`DEEP_SMEM_BUDGET`] (no row born un-loadable on the
    /// smallest target this project ships to) and every [`PIPE_WIDE_VARIANTS`] row inside
    /// [`WIDE_SMEM_BUDGET`] and *outside* the Ada one — a wide row that quietly shrank back under 99 KiB
    /// belongs in the deep table where this card would gate it on the metal, not here where nothing runs
    /// it. Also pins the **launch-bounds** discipline the wide tiles introduced: `min_ctas > 0` emits
    /// exactly `.maxntid {threads},1,1` + `.minnctapersm N`, `min_ctas == 0` emits neither, and the two
    /// are checked per ENTRY, since a module-wide `contains` cannot see a directive landing on the wrong
    /// kernel.
    #[test]
    fn deep_grid_smem_math_and_window_discipline() {
        let deep = gemm_deep_ptx();
        assert_eq!(
            deep.matches(".extern .shared").count(),
            1,
            "exactly ONE window per module"
        );
        let decl = deep.find(".extern .shared").expect("window");
        let first_entry = deep.find(".visible .entry").expect("entry");
        assert!(
            decl < first_entry,
            "the window must be declared at MODULE scope, before any entry"
        );
        // (name, SMEM bytes, dynamic?) — the closed form `stages·(bm+bn)·(bk+pad)·2` spelled out, so an
        // edit to a row's geometry has to be restated here rather than recomputed by the same formula.
        let deep_expect: [(&str, usize, bool); 11] = [
            ("deep_swz_128_s2", 32768, false),
            ("deep_swz_128_s3", 49152, false),
            ("deep_swz_128_s4", 65536, true),
            ("deep_swz_128_s5", 81920, true),
            ("deep_swz_128x256_s2", 49152, false),
            ("deep_swz_128x256_s3", 73728, true),
            ("deep_swz_128x256_s4", 98304, true),
            ("deep_swz_128x256_s4_mc1", 98304, true),
            ("deep_swz_256x128_s2", 49152, false),
            ("deep_swz_256x128_s3", 73728, true),
            ("deep_swz_256x128_s4", 98304, true),
        ];
        let wide_expect: [(&str, usize, bool); 4] = [
            ("wide_swz_128_s7", 114688, true),
            ("wide_swz_128x256_s5_mc1", 122880, true),
            ("wide_swz_128x256_s6_mc1", 147456, true),
            ("wide_swz_256x128_s6_mc1", 147456, true),
        ];
        assert_eq!(PIPE_DEEP_VARIANTS.len(), deep_expect.len());
        assert_eq!(PIPE_WIDE_VARIANTS.len(), wide_expect.len());
        let rows = PIPE_DEEP_VARIANTS
            .iter()
            .zip(deep_expect.iter().map(|e| (*e, DEEP_SMEM_BUDGET, false)))
            .chain(
                PIPE_WIDE_VARIANTS
                    .iter()
                    .zip(wide_expect.iter().map(|e| (*e, WIDE_SMEM_BUDGET, true))),
            );
        for (v, ((name, bytes, dynamic), budget, datacenter_only)) in rows {
            assert_eq!(v.name, name, "grid order");
            assert_eq!(v.smem_bytes(), bytes, "{name}: SMEM closed form");
            assert_eq!(
                v.smem_mode().is_dynamic(),
                dynamic,
                "{name}: emission form at {bytes} B"
            );
            assert_eq!(
                v.smem_mode().launch_bytes(),
                if dynamic { bytes } else { 0 },
                "{name}"
            );
            assert!(
                v.smem_bytes() <= budget,
                "{name}: must fit its table's ceiling {budget} B"
            );
            assert_eq!(
                v.smem_bytes() > DEEP_SMEM_BUDGET,
                datacenter_only,
                "{name}: a row is datacenter-only exactly when it does not fit an Ada carveout \
                 ({DEEP_SMEM_BUDGET} B) — otherwise it belongs in the other table"
            );
            let tile_a = v.bm * (v.bk + v.pad) * 2; // one A buffer of the ring
            let entry = entry_of(deep, name);
            if dynamic {
                assert!(
                    !deep.contains(&format!("smemA_{name}")),
                    "{name}: no statics beside the window"
                );
                assert!(
                    entry.contains(&format!("mov.u32 %bptr,{DSMEM_SYM};")),
                    "{name}: B ring via the symbol"
                );
                assert!(
                    entry.contains(&format!("add.u32 %bptr,%bptr,{};", v.stages * tile_a)),
                    "{name}: the B ring must start after the whole A ring"
                );
            } else {
                assert!(
                    entry.contains(&format!(
                        ".shared .align 16 .b8 smemA_{name}[{}];",
                        v.stages * tile_a
                    )),
                    "{name}: static rows keep their own arrays"
                );
            }
            // Launch bounds: present iff the row asks for them, on this entry and nowhere else.
            let bounds = format!(
                ".maxntid {}, 1, 1\n.minnctapersm {}\n",
                v.threads(),
                v.min_ctas
            );
            assert_eq!(
                entry.contains(&bounds),
                v.min_ctas > 0,
                "{name}: min_ctas={} but the `{}` directives are {}",
                v.min_ctas,
                bounds.trim().replace('\n', " + "),
                if v.min_ctas > 0 { "missing" } else { "present" }
            );
            assert_eq!(
                entry.contains(".minnctapersm"),
                v.min_ctas > 0,
                "{name}: stray launch-bounds directive"
            );
            // Every depth keeps `stages-2` cp.async groups in flight and guards each prologue slab.
            assert!(
                entry.contains(&format!("cp.async.wait_group {};", v.stages - 2)),
                "{name}"
            );
            for st in 0..(v.stages - 1) {
                assert!(
                    entry.contains(&format!("PRO_{name}_{st}:")),
                    "{name}: prologue slab {st} unguarded"
                );
            }
        }
    }

    /// Retarget gate (GPU_RETARGET_PLAN.md §5, Phase 2): every tensor-core module here must open with
    /// the `sm_80` FLOOR header, never the development box's `sm_89`. `wmma.*.m16n16k16`,
    /// `mma.sync.m16n8k16`, `ldmatrix` and `cp.async` are all Ampere-ISA instructions, and PTX is
    /// forward-compatible only — an `sm_89` tag on an Ampere-legal module is a pure loss that fails
    /// `cuModuleLoadData` on every A100 while being completely invisible to the device gates below,
    /// which run on an Ada card that accepts either tag.
    #[test]
    fn every_tensor_core_module_opens_at_the_sm80_floor() {
        let statics = [
            wmma_f16_sm_static_ptx(128, 128, 128, false),
            wmma_f16_sm_static_ptx(128, 128, 128, true),
        ];
        let modules: [(&str, &str); 7] = [
            ("wmma_f16_ptx", wmma_f16_ptx()),
            ("wmma_bf16_ptx", wmma_bf16_ptx()),
            ("gemm_cliff_ptx", gemm_cliff_ptx()),
            ("gemm_deep_ptx", gemm_deep_ptx()),
            ("roofline_f16_ptx", roofline_f16_ptx()),
            ("wmma_f16_sm_static_ptx(64)", &statics[0]),
            ("wmma_f16_sm_static_ptx(128)", &statics[1]),
        ];
        for (what, ptx) in modules {
            assert!(
                ptx.starts_with(HDR_SM80),
                "{what}: must open with ptx_target::HDR_SM80"
            );
            assert!(
                !ptx.contains(crate::ptx_target::TARGET_SM89),
                "{what}: emits no Ada-only instruction, so it must not be tagged sm_89"
            );
        }
    }

    /// The dispatch seam must close: every entry name a host launcher can hand `Gpu::function` has to
    /// be defined in the module it is loaded from. The two sides are separate literals -- e.g.
    /// `gpu::gemm_nt_f16_mma_bias_gelu` asks for "mma_nt_f16_128_bk32_s2_r16_swz_bias_gelu" while the
    /// builder synthesises it as `format!("{}_swz_{suffix}", wh.name)` -- so reordering the builder's
    /// format string compiles clean, passes every GPU-less test, and then fails at runtime with
    /// `DriverError(CUDA_ERROR_NOT_FOUND)` on four dispatched public entry points.
    #[test]
    fn every_dispatched_tensor_core_entry_is_defined() {
        let f16 = wmma_f16_ptx();
        let bf16 = wmma_bf16_ptx();
        let cliff = gemm_cliff_ptx();
        let has = |ptx: &str, n: &str| ptx.contains(&format!(".visible .entry {n}("));

        // Sweep tables shared by the builder and the host (`gemm_nt_f16_pipe` / `gemm_nt_f16_cliff`
        // launch `v.name` straight from these), so a table edit must reach the module.
        for v in PIPE_VARIANTS {
            assert!(
                has(f16, v.name),
                "PIPE_VARIANTS entry `{}` missing from wmma_f16_ptx",
                v.name
            );
        }
        for v in CLIFF_VARIANTS {
            assert!(
                has(cliff, v.name),
                "CLIFF_VARIANTS entry `{}` missing from gemm_cliff_ptx",
                v.name
            );
        }
        let deep = gemm_deep_ptx();
        for v in PIPE_DEEP_VARIANTS {
            assert!(
                has(deep, v.name),
                "PIPE_DEEP_VARIANTS entry `{}` missing from gemm_deep_ptx",
                v.name
            );
            assert_eq!(deep_variant(v.name).name, v.name, "deep_variant round-trip");
        }
        // The wide (datacenter-tile) rows share the one deep module, so `gpu::gemm_nt_f16_deep` can
        // launch them unchanged on a part whose carveout is big enough — but only if the entry the
        // lookup names is actually in the text the module cache loads.
        for v in PIPE_WIDE_VARIANTS {
            assert!(
                has(deep, v.name),
                "PIPE_WIDE_VARIANTS entry `{}` missing from gemm_deep_ptx",
                v.name
            );
            assert_eq!(wide_variant(v.name).name, v.name, "wide_variant round-trip");
        }
        assert!(
            has(bf16, PIPE_BF16.name),
            "PIPE_BF16 entry `{}` missing",
            PIPE_BF16.name
        );
        for use_128 in [false, true] {
            let n = wmma_f16_sm_static_entry(use_128);
            let ptx = wmma_f16_sm_static_ptx(256, 256, 256, use_128);
            assert!(
                has(&ptx, n),
                "static-shape entry `{n}` missing from its own module"
            );
        }

        // The names gpu.rs spells as literals at its dispatch sites.
        let wh = "mma_nt_f16_128_bk32_s2_r16";
        let p64 = "wmma_nt_f16_pipe_64_s6";
        let bwh = PIPE_BF16.name;
        for n in [
            "wmma_nt_f16",
            "wmma_nt_f16_mt",
            "wmma_nt_f16_sm",
            "wmma_nt_f16_sm128",
            "wmma_nt_f16_sm_db",
            "wmma_nt_f16_sm128_db",
            "wmma_nt_f16_sm_db_residual",
            "wmma_nt_f16_sm_db_relu",
            "wmma_nt_f16_sm_db_silu",
            "wmma_nt_f16_sm_db_gelu",
            "wmma_nt_f16_sm_db_bias",
            "wmma_nt_f16_sm_db_bias_relu",
            "wmma_nt_f16_sm_db_bias_silu",
            "wmma_nt_f16_sm_db_bias_gelu",
        ] {
            assert!(
                has(f16, n),
                "dispatched entry `{n}` missing from wmma_f16_ptx"
            );
        }
        for n in [
            "wmma_nt_bf16",
            "wmma_nt_bf16_mt",
            "wmma_nt_bf16_sm_db",
            "wmma_nt_bf16_sm_db_relu",
            "wmma_nt_bf16_sm_db_silu",
            "wmma_nt_bf16_sm_db_gelu",
            "wmma_nt_bf16_sm_db_bias",
            "wmma_nt_bf16_sm_db_bias_relu",
            "wmma_nt_bf16_sm_db_bias_silu",
            "wmma_nt_bf16_sm_db_bias_gelu",
        ] {
            assert!(
                has(bf16, n),
                "dispatched entry `{n}` missing from wmma_bf16_ptx"
            );
        }
        // The fused-epilogue families: `{base}[_swz]_bias[_act]` and `..._bias_residual`.
        for base in [wh, p64] {
            for suffix in [
                "bias",
                "bias_relu",
                "bias_silu",
                "bias_gelu",
                "bias_residual",
            ] {
                assert!(
                    has(f16, &format!("{base}_{suffix}")),
                    "`{base}_{suffix}` missing"
                );
            }
        }
        for suffix in [
            "bias",
            "bias_relu",
            "bias_silu",
            "bias_gelu",
            "bias_residual",
        ] {
            assert!(
                has(f16, &format!("{wh}_swz_{suffix}")),
                "`{wh}_swz_{suffix}` missing"
            );
            assert!(
                has(bf16, &format!("{bwh}_{suffix}")),
                "`{bwh}_{suffix}` missing"
            );
            assert!(
                has(bf16, &format!("{bwh}_swz_{suffix}")),
                "`{bwh}_swz_{suffix}` missing"
            );
        }
        for base in [wh, bwh] {
            let ptx = if base == wh { f16 } else { bf16 };
            for twin in ["_swz", "_w22swz"] {
                assert!(has(ptx, &format!("{base}{twin}")), "`{base}{twin}` missing");
            }
        }
        // The gated-FFN (GLU-family) dual-B tiles, padded base + swizzle twin.
        for ty in ["f16", "bf16"] {
            let ptx = if ty == "f16" { f16 } else { bf16 };
            for g in [
                "gate_silu",
                "gate_gelu",
                "gate_glu",
                "gate_silu_bias",
                "gate_gelu_bias",
            ] {
                assert!(
                    has(ptx, &format!("mma_nt_{ty}_128x64_{g}")),
                    "`mma_nt_{ty}_128x64_{g}` missing"
                );
                assert!(
                    has(ptx, &format!("mma_nt_{ty}_128x64_{g}_swz")),
                    "`mma_nt_{ty}_128x64_{g}_swz` missing"
                );
            }
        }
        assert!(has(roofline_f16_ptx(), "wmma_roofline_f16"));
    }

    /// The staging precondition [`entry_smem`]/[`entry_smem_db`] document is now checked, so the four
    /// production instantiations must satisfy it (each is one 128-bit chunk per thread) and a tile that
    /// does not tile the CTA must be rejected at generation time instead of emitting a kernel with a
    /// partial (or empty) global->shared stage that JITs and reads stale shared memory.
    #[test]
    fn smem_staging_tiles_the_cta_for_every_generated_shape() {
        // Production shapes: 64x64 / 4 warps and 128x128 / 8 warps, both precisions. `a_chunks == 1`
        // each, so the guard changes no emitted byte.
        for (bm, bn, wm, wn) in [
            (SM_BM, SM_BN, SM_WARPS_M, SM_WARPS_N),
            (SM128_BM, SM128_BN, SM128_WARPS_M, SM128_WARPS_N),
        ] {
            let threads = wm * wn * 32;
            assert_eq!(
                bm * SM_BK % (threads * 8),
                0,
                "A tile {bm}x{SM_BK} must tile {threads} threads"
            );
            assert_eq!(
                bn * SM_BK % (threads * 8),
                0,
                "B tile {bn}x{SM_BK} must tile {threads} threads"
            );
            for ty in ["f16", "bf16"] {
                let p = entry_smem("t_sm", ty, bm, bn, wm, wn, None);
                assert_eq!(
                    p.matches("ld.global.v4.u32").count(),
                    2,
                    "one A + one B stage per K step"
                );
                // The double-buffered twin stages with `cp.async` (global->shared, no register hop):
                // one 16-byte copy per chunk, prologue + steady state.
                let d = entry_smem_db("t_db", ty, bm, bn, wm, wn, Act::None, false, false);
                assert!(
                    d.matches("cp.async.cg.shared.global").count() >= 4,
                    "the double-buffered kernel must stage A/B in both the prologue and the K loop"
                );
            }
        }
    }

    /// `bm*SM_BK = 32*16 = 512 < threads*8 = 1024` truncates to ZERO staging chunks: before the guard
    /// this generated a fully well-formed kernel with no global->shared copy at all.
    #[test]
    #[should_panic(expected = "is not a whole multiple of threads*8")]
    fn smem_tile_smaller_than_one_chunk_per_thread_is_rejected() {
        entry_smem("t_bad_sm", "f16", 32, 32, 2, 2, None);
    }

    /// The partial-multiple case (`96*16 = 1536`, `threads*8 = 1024` => one chunk staged, a third of the
    /// tile left stale) is the more dangerous one — it reads as a tolerance failure, not a codegen bug.
    #[test]
    #[should_panic(expected = "is not a whole multiple of threads*8")]
    fn smem_tile_that_only_partly_tiles_the_cta_is_rejected() {
        entry_smem("t_partial_sm", "f16", 96, 96, 2, 2, None);
    }

    /// The double-buffered generator carries the same guard.
    #[test]
    #[should_panic(expected = "is not a whole multiple of threads*8")]
    fn smem_db_tile_smaller_than_one_chunk_per_thread_is_rejected() {
        entry_smem_db("t_bad_db", "f16", 32, 32, 2, 2, Act::None, false, false);
    }
}
