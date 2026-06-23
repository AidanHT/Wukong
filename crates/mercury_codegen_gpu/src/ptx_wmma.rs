//! Tensor-core GEMM via `wmma` PTX — the FLOP/s headline. On Ada (`sm_89`) the 4th-gen tensor cores
//! do bf16/fp16 multiplies with **f32 accumulate** (the standard mixed-precision contract), at many
//! times the f32 CUDA-core rate. This is where low precision stops being a footprint trick and buys
//! real throughput.
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
/// `gemm_nt_f16` dispatches among these by working-set size; `gemm_pipe_sweep` re-measures them. (~72% is
/// near the WMMA ceiling on this part; the cuBLAS-class `mma.sync`+`ldmatrix` path is the next lever.)
pub const PIPE_VARIANTS: &[PipeCfg] = &[
    PipeCfg { name: "wmma_nt_f16_pipe_64_s6", bm: 64, bn: 64, bk: 16, wm: 2, wn: 2, stages: 6, raster: 0, mma: false, pad: 0 }, // 24 KiB — ≤1024³
    PipeCfg { name: "wmma_nt_f16_pipe_128_s4", bm: 128, bn: 128, bk: 16, wm: 2, wn: 4, stages: 4, raster: 0, mma: false, pad: 0 }, // 32 KiB — ~2048³
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
    PipeCfg { name: "mma_nt_f16_128_bk32_s2_r16", bm: 128, bn: 128, bk: 32, wm: 2, wn: 4, stages: 2, raster: 16, mma: true, pad: 8 }, // 40 KiB
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

/// Generate a shared-memory-staged WMMA GEMM entry computing `C = A·Bᵀ`. `ty` is "f16" or "bf16"; the
/// CTA stages a `bm×SM_BK` tile of A and a `bn×SM_BK` tile of B, with a `warps_m×warps_n` warp grid
/// each owning a `(bm/warps_m)×(bn/warps_n)` sub-tile. `bm`,`bn` must be 16-multiples and the staging
/// requires `bm·SM_BK` and `bn·SM_BK` to be whole multiples of `threads·8` (8 f16 per vectorized load).
fn entry_smem(name: &str, ty: &str, bm: usize, bn: usize, warps_m: usize, warps_n: usize) -> String {
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

    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
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
            *s += &format!("    mov.u32 %tmp,{smem};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n");
            *s += "    st.shared.v4.u32 [%tmp],{%v0,%v1,%v2,%v3};\n";
        }
    };
    stage("%baseRow", "%A", format!("smemA_{name}"), a_chunks, &mut s);
    stage("%baseCol", "%B", format!("smemB_{name}"), b_chunks, &mut s);
    s += "    bar.sync 0;\n";

    // Each warp loads its fragments from shared (generic addr via cvta.shared) and accumulates.
    for ti in 0..tm {
        s += &format!("    mov.u32 %tmp,smemA_{name};\n");
        s += &format!("    mul.lo.s32 %tmp2,%warpRow,{wm};\n    add.u32 %tmp2,%tmp2,{};\n", ti * 16);
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let ra = veclist(&format!("a{ti}_"), nab);
        s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%gp], %ldm;\n");
    }
    for tj in 0..tn {
        s += &format!("    mov.u32 %tmp,smemB_{name};\n");
        s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
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
            s += &format!("    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n", ti * 16);
            s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
            s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
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
    assert_eq!(bm, bn, "the double-buffered kernel uses one buffer-offset reg for A and B");
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
    let wn_shift = warps_n.trailing_zeros();
    let wm = (16 * tm) as i64;
    let wn = (16 * tn) as i64;

    // The fused-bias variant takes an extra `bias[N]` (f32) param read in the store-back epilogue; the
    // residual variant takes a `residual[M,N]` (f32) param used to seed the accumulator (wmma.load.c).
    let bias_param = if bias { ",\n    .param .u64 pBias" } else { "" };
    let resid_param = if residual { ",\n    .param .u64 pResidual" } else { "" };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{}];\n", 2 * tile_bytes);
    s += &format!("    .shared .align 16 .b8 smemB_{name}[{}];\n", 2 * tile_bytes);
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
    s += &format!("    .reg .f32 {},%act0,%act1;\n", decl_c.trim_end_matches(','));
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
                s += &format!("    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n", ti * 16);
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%Resid,%off;\n";
                s += &format!("    wmma.load.c.sync.aligned.m16n16k16.row.f32 {cc}, [%cptr], %N;\n");
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
    let stage = |g_base: &str, gbase_ptr: &str, smem: String, bufoff: &str, chunks: usize, s: &mut String| {
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
    stage("%baseRow", "%A", format!("smemA_{name}"), "%bufc", a_chunks, &mut s);
    stage("%baseCol", "%B", format!("smemB_{name}"), "%bufc", b_chunks, &mut s);
    s += "    cp.async.commit_group;\n";

    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    // Prefetch the next tile into the alternate buffer (if any), then wait on the *current* tile only.
    s += "    add.u32 %ktn,%kt,16;\n    setp.lt.u32 %pmore,%ktn,%K;\n";
    s += &format!("    @!%pmore bra LAST_{name};\n");
    s += "    mov.u32 %kcol,%ktn;\n";
    stage("%baseRow", "%A", format!("smemA_{name}"), "%bufp", a_chunks, &mut s);
    stage("%baseCol", "%B", format!("smemB_{name}"), "%bufp", b_chunks, &mut s);
    s += "    cp.async.commit_group;\n    cp.async.wait_group 1;\n";
    s += &format!("    bra SYNC_{name};\nLAST_{name}:\n    cp.async.wait_group 0;\nSYNC_{name}:\n");
    s += "    bar.sync 0;\n";
    // Compute the current tile from buffer `%bufc`.
    for ti in 0..tm {
        s += &format!("    mov.u32 %tmp,smemA_{name};\n    add.u32 %tmp,%tmp,%bufc;\n");
        s += &format!("    mul.lo.s32 %tmp2,%warpRow,{wm};\n    add.u32 %tmp2,%tmp2,{};\n", ti * 16);
        s += "    mul.lo.s32 %tmp2,%tmp2,32;\n    add.u32 %tmp,%tmp,%tmp2;\n";
        s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
        let ra = veclist(&format!("a{ti}_"), nab);
        s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%gp], %ldm;\n");
    }
    for tj in 0..tn {
        s += &format!("    mov.u32 %tmp,smemB_{name};\n    add.u32 %tmp,%tmp,%bufc;\n");
        s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
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
                s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%scptr], {cc}, %ldm;\n");
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
                s += &format!("    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n", ti * 16);
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
                // Fused epilogue: activate each accumulator register in place before storing C.
                for r in 0..8 {
                    s += &act.epilogue(&format!("%c{ti}_{tj}_{r}"));
                }
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n");
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
    assert!(stages >= 2, "the pipeline needs at least 2 stages (1 prefetch in flight)");
    assert!(bk % 16 == 0, "bk must be a multiple of the WMMA k16 step");
    assert!((bk / 8).is_power_of_two(), "bk/8 must be a power of two (shift-based staging address math)");
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
    assert!(a_chunks >= 1 && b_chunks >= 1, "{name}: tile too small for one 128-bit chunk per thread");
    let bk_chunks = bk / 8; // 8-element (128-bit) chunks per staged row
    let row_shift = bk_chunks.trailing_zeros(); // flat-chunk e → row = e >> row_shift
    let col_mask = bk_chunks - 1; //              col8 = (e & col_mask) << 3
    let wn_shift = warps_n.trailing_zeros();
    let wm = (16 * tm) as i64;
    let wn = (16 * tn) as i64;

    // The fused-bias variant takes an extra `bias[N]` (f32) param read in the SMEM store-back epilogue;
    // the residual variant takes a `residual[M,N]` (f32) used to seed the accumulator via wmma.load.c.
    let bias_param = if bias { ",\n    .param .u64 pBias" } else { "" };
    let resid_param = if residual { ",\n    .param .u64 pResidual" } else { "" };
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
        s += &format!("    .reg .f32 {},%act0,%act1;\n", decl_c.trim_end_matches(','));
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
                s += &format!("    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n", ti * 16);
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%Resid,%off;\n";
                s += &format!("    wmma.load.c.sync.aligned.m16n16k16.row.f32 {cc}, [%cptr], %N;\n");
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
    let stage = |g_base: &str, gbase_ptr: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
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
        s += &format!("    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n", st * tile_a, st * tile_b);
        s += &format!("    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra PRO_{name}_{st};\n");
        stage("%baseRow", "%A", &format!("smemA_{name}"), "%bufwA", a_chunks, &mut s);
        stage("%baseCol", "%B", &format!("smemB_{name}"), "%bufwB", b_chunks, &mut s);
        s += &format!("PRO_{name}_{st}:\n    cp.async.commit_group;\n");
    }

    // Compute buffer starts at offset 0; the write (prefetch) buffer trails by stages-1 buffers.
    s += "    mov.u32 %bufcA,0;\n    mov.u32 %bufcB,0;\n";
    s += &format!("    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n", (stages - 1) * tile_a, (stages - 1) * tile_b);
    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    // Drain the oldest in-flight group (the tile we are about to read), then the single fence.
    s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 2);
    // Issue the prefetch for the tile (stages-1) ahead into the write buffer, if it exists; commit.
    s += &format!("    add.u32 %kcol,%kt,{};\n    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra NOPRE_{name};\n", (stages - 1) * bk);
    stage("%baseRow", "%A", &format!("smemA_{name}"), "%bufwA", a_chunks, &mut s);
    stage("%baseCol", "%B", &format!("smemB_{name}"), "%bufwB", b_chunks, &mut s);
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");
    // Compute the current tile from buffer `%bufc`: bk/16 WMMA k-steps, fragment reuse across tm×tn.
    for ks in 0..nks {
        for ti in 0..tm {
            s += &format!("    mov.u32 %tmp,smemA_{name};\n    add.u32 %tmp,%tmp,%bufcA;\n");
            s += &format!("    mul.lo.s32 %tmp2,%warpRow,{};\n    add.u32 %tmp2,%tmp2,{};\n", wm * bk as i64, ti * 16 * bk);
            s += &format!("    add.u32 %tmp2,%tmp2,{};\n    shl.b32 %tmp2,%tmp2,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ks * 16);
            s += "    cvt.u64.u32 %gp,%tmp;\n    cvta.shared.u64 %gp,%gp;\n";
            let ra = veclist(&format!("a{ti}_"), nab);
            s += &format!("    wmma.load.a.sync.aligned.m16n16k16.row.{ty} {ra}, [%gp], %ldm;\n");
        }
        for tj in 0..tn {
            s += &format!("    mov.u32 %tmp,smemB_{name};\n    add.u32 %tmp,%tmp,%bufcB;\n");
            s += &format!("    mul.lo.s32 %tmp2,%warpCol,{};\n    add.u32 %tmp2,%tmp2,{};\n", wn * bk as i64, tj * 16 * bk);
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
                s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%scptr], {cc}, %scld;\n");
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
                s += &format!("    mul.lo.s32 %tmp,%warpRow,{wm};\n    add.u32 %tmp,%tmp,{};\n", ti * 16);
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!("    mul.lo.s32 %tmp2,%warpCol,{wn};\n    add.u32 %tmp2,%tmp2,{};\n", tj * 16);
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
                let cc = veclist(&format!("c{ti}_{tj}_"), 8);
                s += &format!("    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%cptr], {cc}, %N;\n");
            }
        }
    }
    s += "    ret;\n}\n";
    s
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
) -> String {
    assert!(stages >= 2, "the pipeline needs at least 2 stages");
    assert!(bk % 16 == 0 && (bk / 8).is_power_of_two(), "bk must be a 16-multiple with bk/8 a power of two");
    assert!(bm % (16 * warps_m) == 0, "{name}: bm must be a multiple of 16·warps_m (m16 sub-tiles)");
    assert!(bn % (8 * warps_n) == 0, "{name}: bn must be a multiple of 8·warps_n (n8 sub-tiles)");
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
    assert!(pad % 8 == 0, "{name}: pad must be a multiple of 8 (16-byte cp.async alignment)");
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
        assert!(bk == 32, "{name}: the swz swizzle phase is derived for bk=32 (nc=4)");
        assert!(wmr % 8 == 0 && wnc % 8 == 0, "{name}: swz needs warp row/col bases ≡ 0 (mod 8)");
    }
    let ldp = if swz { bk } else { bk + pad }; // swz: no pad (the swizzle, not padding, gives conflict-free)
    let tile_a = bm * ldp * 2;
    let tile_b = bn * ldp * 2;
    let smem_a = stages * tile_a;
    let smem_b = stages * tile_b;
    assert!(smem_a + smem_b <= 48 * 1024, "{name}: static SMEM {} B exceeds 48 KiB", smem_a + smem_b);
    let a_chunks = bm * bk / (threads * 8);
    let b_chunks = bn * bk / (threads * 8);
    assert!(a_chunks >= 1 && b_chunks >= 1, "{name}: tile too small for one 128-bit chunk per thread");
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
    let resid_param = if residual { ",\n    .param .u64 pResid" } else { "" };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
    s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
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
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    s += &format!("    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n");
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
    let stage = |g_base: &str, gbase_ptr: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
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
                *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    mul.lo.s32 %tmp3,%r,{};\n    add.u32 %tmp,%tmp,%tmp3;\n    add.u32 %tmp,%tmp,%swztmp;\n", bk * 2);
            } else {
                *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n");
                *s += &format!("    mul.lo.s32 %tmp2,%r,{};\n    add.u32 %tmp,%tmp,%tmp2;\n    shl.b32 %tmp2,%c,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ldp * 2);
            }
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    for st in 0..(stages - 1) {
        s += &format!("    mov.u32 %kcol,{};\n", st * bk);
        s += &format!("    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n", st * tile_a, st * tile_b);
        s += &format!("    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra PRO_{name}_{st};\n");
        stage("%baseRow", "%A", &format!("smemA_{name}"), "%bufwA", a_chunks, &mut s);
        stage("%baseCol", "%B", &format!("smemB_{name}"), "%bufwB", b_chunks, &mut s);
        s += &format!("PRO_{name}_{st}:\n    cp.async.commit_group;\n");
    }
    s += "    mov.u32 %bufcA,0;\n    mov.u32 %bufcB,0;\n";
    s += &format!("    mov.u32 %bufwA,{};\n    mov.u32 %bufwB,{};\n", (stages - 1) * tile_a, (stages - 1) * tile_b);
    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 2);
    s += &format!("    add.u32 %kcol,%kt,{};\n    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra NOPRE_{name};\n", (stages - 1) * bk);
    stage("%baseRow", "%A", &format!("smemA_{name}"), "%bufwA", a_chunks, &mut s);
    stage("%baseCol", "%B", &format!("smemB_{name}"), "%bufwB", b_chunks, &mut s);
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");

    // Compute: for each k16 step load the A/B fragments — swz: one `ldmatrix.x4`/`.x2` warp-cooperative
    // gather per sub-tile from the XOR-swizzled (conflict-free, no-pad) SMEM; non-swz: hand-placed
    // `ld.shared.b32` from the padded SMEM — then issue tm·tn `mma.sync` m16n8k16 ops (identical either way).
    for ks in 0..nks {
        if swz {
            // A: aptr = smemA + bufcA + arowb (this lane's row base). chunk_off = ((ks·2 | la16) XOR phaseA)·16
            // (per-ks, per-lane); per mi add mi·16·bk·2 → ldmatrix.x4 {A00,A10,A01,A11} = the mma A regs.
            s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n    add.u32 %aptr,%aptr,%arowb;\n");
            s += &format!("    or.b32 %swztmp,%la16,{};\n    xor.b32 %swztmp,%swztmp,%phaseA;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for mi in 0..tm {
                let mibase = mi * 16 * bk * 2;
                s += &format!("    add.u32 %tmp,%aptr,%swztmp;\n    add.u32 %tmp,%tmp,{mibase};\n    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},[%tmp];\n");
            }
            // B: bptr = smemB + bufcB + browb. chunk_off = ((ks·2 | lb8) XOR phaseB)·16; per ni add ni·8·bk·2
            // → ldmatrix.x2 {B0,B1} = the mma B regs.
            s += &format!("    mov.u32 %bptr,smemB_{name};\n    add.u32 %bptr,%bptr,%bufcB;\n    add.u32 %bptr,%bptr,%browb;\n");
            s += &format!("    or.b32 %swztmp,%lb8,{};\n    xor.b32 %swztmp,%swztmp,%phaseB;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
            for ni in 0..tn {
                let nibase = ni * 8 * bk * 2;
                s += &format!("    add.u32 %tmp,%bptr,%swztmp;\n    add.u32 %tmp,%tmp,{nibase};\n    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%b{ni}_0,%b{ni}_1}},[%tmp];\n");
            }
        } else {
            // A base ptr = smemA + bufcA + (warpMrow·ldp)·2 + laneoff + (ks·16)·2  (ldp = padded row stride).
            s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n");
            s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %aptr,%aptr,%tmp;\n");
            s += &format!("    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n", ks * 32);
            for mi in 0..tm {
                let base = mi * 16 * ldp * 2; // m16-block row offset (bytes, padded stride)
                let r8 = 8 * ldp * 2; // +8 rows
                s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16); // k+8 (not padded)
                s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
                s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
            }
            // B base ptr = smemB + bufcB + (warpNcol·ldp)·2 + laneoff + (ks·16)·2.
            s += &format!("    mov.u32 %bptr,smemB_{name};\n    add.u32 %bptr,%bptr,%bufcB;\n");
            s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %bptr,%bptr,%tmp;\n");
            s += &format!("    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n", ks * 32);
            for ni in 0..tn {
                let base = ni * 8 * ldp * 2; // n8-block row offset (bytes, padded stride)
                s += &format!("    ld.shared.b32 %b{ni}_0,[%bptr+{}];\n", base);
                s += &format!("    ld.shared.b32 %b{ni}_1,[%bptr+{}];\n", base + 16); // k+8 (not padded)
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
    // d2=(grp+8,2tg) d3=(grp+8,2tg+1), per (mi,ni) sub-tile.
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
            s += &format!("    st.global.f32 [%cptr],%d{mi}_{ni}_0;\n    st.global.f32 [%cptr+4],%d{mi}_{ni}_1;\n");
            // row grp+8: C[(grow+8)·N+gcol] = d2, [+1] = d3 (residual scratch in the now-free %cptr).
            s += "    add.u32 %tmp,%grow,8;\n    mul.lo.s32 %tmp,%tmp,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr2,%C,%off;\n";
            if residual {
                s += "    add.s64 %cptr,%Resid,%off;\n    ld.global.f32 %resv0,[%cptr];\n    ld.global.f32 %resv1,[%cptr+4];\n";
                s += &format!("    add.f32 %d{mi}_{ni}_2,%d{mi}_{ni}_2,%resv0;\n    add.f32 %d{mi}_{ni}_3,%d{mi}_{ni}_3,%resv1;\n");
            }
            s += &format!("    st.global.f32 [%cptr2],%d{mi}_{ni}_2;\n    st.global.f32 [%cptr2+4],%d{mi}_{ni}_3;\n");
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
) -> String {
    assert!(stages >= 2, "the pipeline needs at least 2 stages");
    assert!(bk % 16 == 0 && (bk / 8).is_power_of_two(), "bk must be a 16-multiple with bk/8 a power of two");
    assert!(bm % (16 * warps_m) == 0, "{name}: bm must be a multiple of 16·warps_m");
    assert!(bn % (8 * warps_n) == 0, "{name}: bn must be a multiple of 8·warps_n");
    assert!(
        raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()),
        "{name}: rasterization needs bm,bn powers of two"
    );
    assert!(pad % 8 == 0, "{name}: pad must be a multiple of 8 (16-byte cp.async alignment)");
    let mma_ty = format!("f32.{ty}.{ty}.f32");
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m); // m16 sub-tiles per warp
    let tn = bn / (8 * warps_n); //  n8 sub-tiles per warp
    let nks = bk / 16;
    let wmr = bm / warps_m;
    let wnc = bn / warps_n;
    let ldp = bk + pad;
    let tile_a = bm * ldp * 2;
    let tile_b = bn * ldp * 2; // one Wg (== one Wu) tile
    let smem_a = stages * tile_a;
    let smem_b = stages * tile_b;
    // Three ring buffers: A + Wg + Wu. 128×64/bk32/pad8/s2 ⇒ 20480 + 2·10240 = 40 KiB (= single-B 128×128).
    assert!(smem_a + 2 * smem_b <= 48 * 1024, "{name}: static SMEM {} B exceeds 48 KiB", smem_a + 2 * smem_b);
    let a_chunks = bm * bk / (threads * 8);
    let b_chunks = bn * bk / (threads * 8);
    assert!(a_chunks >= 1 && b_chunks >= 1, "{name}: tile too small for one 128-bit chunk per thread");
    let bk_chunks = bk / 8;
    let row_shift = bk_chunks.trailing_zeros();
    let col_mask = bk_chunks - 1;
    let wn_shift = warps_n.trailing_zeros();

    // The fused-bias variant takes per-column gate/up biases `bg[N]`,`bu[N]` (f32), added in the store
    // epilogue (the `silu(x·Wg+bg) ⊙ (x·Wu+bu)` form; biasless is the Llama-style no-bias gate).
    let bias_param = if bias { ",\n    .param .u64 pBiasG,\n    .param .u64 pBiasU" } else { "" };
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
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    s += &format!("    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n");
    s += &format!("    mul.lo.s32 %laneoff,%grp,{ldp};\n    add.u32 %laneoff,%laneoff,%tg2;\n    shl.b32 %laneoff,%laneoff,1;\n");
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %dg{mi}_{ni}_{r},0f00000000;\n    mov.f32 %du{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }

    // cp.async staging into padded SMEM (row stride `ldp`) — identical to entry_mma_pipe's `stage`.
    let stage = |g_base: &str, gbase_ptr: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,3;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    mul.wide.u32 %off,%tmp,2;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n");
            *s += &format!("    mul.lo.s32 %tmp2,%r,{};\n    add.u32 %tmp,%tmp,%tmp2;\n    shl.b32 %tmp2,%c,1;\n    add.u32 %tmp,%tmp,%tmp2;\n", ldp * 2);
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    // Prologue: prefetch tiles 0..stages-2 of A, Wg, Wu (one cp.async group per K-tile).
    for st in 0..(stages - 1) {
        s += &format!("    mov.u32 %kcol,{};\n", st * bk);
        s += &format!("    mov.u32 %bufwA,{};\n    mov.u32 %bufwBg,{};\n    mov.u32 %bufwBu,{};\n", st * tile_a, st * tile_b, st * tile_b);
        s += &format!("    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra PRO_{name}_{st};\n");
        stage("%baseRow", "%A", &format!("smemA_{name}"), "%bufwA", a_chunks, &mut s);
        stage("%baseCol", "%Bg", &format!("smemBg_{name}"), "%bufwBg", b_chunks, &mut s);
        stage("%baseCol", "%Bu", &format!("smemBu_{name}"), "%bufwBu", b_chunks, &mut s);
        s += &format!("PRO_{name}_{st}:\n    cp.async.commit_group;\n");
    }
    s += "    mov.u32 %bufcA,0;\n    mov.u32 %bufcBg,0;\n    mov.u32 %bufcBu,0;\n";
    s += &format!("    mov.u32 %bufwA,{};\n    mov.u32 %bufwBg,{};\n    mov.u32 %bufwBu,{};\n", (stages - 1) * tile_a, (stages - 1) * tile_b, (stages - 1) * tile_b);
    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    s += &format!("    cp.async.wait_group {};\n    bar.sync 0;\n", stages - 2);
    s += &format!("    add.u32 %kcol,%kt,{};\n    setp.lt.u32 %pmore,%kcol,%K;\n    @!%pmore bra NOPRE_{name};\n", (stages - 1) * bk);
    stage("%baseRow", "%A", &format!("smemA_{name}"), "%bufwA", a_chunks, &mut s);
    stage("%baseCol", "%Bg", &format!("smemBg_{name}"), "%bufwBg", b_chunks, &mut s);
    stage("%baseCol", "%Bu", &format!("smemBu_{name}"), "%bufwBu", b_chunks, &mut s);
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");

    // Compute: load A fragments ONCE (smem + buffer + warp + lane + ks·16), then issue the gate `mma`s
    // (A×Wg → %dg) and the up `mma`s (A×Wu → %du, A fragments reused) — the load-x-once arithmetic win.
    for ks in 0..nks {
        s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %aptr,%aptr,%tmp;\n");
        s += &format!("    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n", ks * 32);
        for mi in 0..tm {
            let base = mi * 16 * ldp * 2;
            let r8 = 8 * ldp * 2;
            s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16);
            s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
            s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
        }
        // Preload BOTH B tiles' fragments (Wg into %bg, Wu into %bu) before issuing any mma, so the
        // 2·tm·tn mma's below — which write disjoint accumulators (%dg vs %du) and so are all mutually
        // independent — form one pipeline-able block with every operand already in registers (matches the
        // single-B workhorse's clean N-independent-mma schedule; the serialized load/mma/load/mma form
        // left the up mma's waiting on the Wu loads).
        s += &format!("    mov.u32 %bptr,smemBg_{name};\n    add.u32 %bptr,%bptr,%bufcBg;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!("    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n", ks * 32);
        for ni in 0..tn {
            let base = ni * 8 * ldp * 2;
            s += &format!("    ld.shared.b32 %bg{ni}_0,[%bptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %bg{ni}_1,[%bptr+{}];\n", base + 16);
        }
        s += &format!("    mov.u32 %bptr,smemBu_{name};\n    add.u32 %bptr,%bptr,%bufcBu;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    shl.b32 %tmp,%tmp,1;\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!("    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n", ks * 32);
        for ni in 0..tn {
            let base = ni * 8 * ldp * 2;
            s += &format!("    ld.shared.b32 %bu{ni}_0,[%bptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %bu{ni}_1,[%bptr+{}];\n", base + 16);
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

/// fp16 tensor-core GEMM module: `wmma_nt_f16` (single 16×16 tile/warp, any 16-multiple dims) and
/// `wmma_nt_f16_mt` (2×4 tiles/warp = 32×64, fragment-reuse, the fast path for large GEMMs).
pub fn wmma_f16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("wmma_nt_f16", "f16", 1, 1);
        m += &entry("wmma_nt_f16_mt", "f16", TM_TILES, TN_TILES);
        m += &entry_smem("wmma_nt_f16_sm", "f16", SM_BM, SM_BN, SM_WARPS_M, SM_WARPS_N);
        // Single-buffered 128×128 tile (no cp.async pipeline). At L2-spilling sizes the double-buffered
        // kernels are occupancy-bound and LOSE to the un-pipelined ones (measured: _sm beats _sm_db at
        // 2048³); the big tile halves redundant inter-CTA traffic while single-buffering avoids the
        // pipeline's extra SMEM + bar.syncs — the large-GEMM candidate the clean scoreboard motivates.
        m += &entry_smem("wmma_nt_f16_sm128", "f16", SM128_BM, SM128_BN, SM128_WARPS_M, SM128_WARPS_N);
        m += &entry_smem_db("wmma_nt_f16_sm_db", "f16", SM_BM, SM_BN, SM_WARPS_M, SM_WARPS_N, Act::None, false, false);
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
                entry_mma_pipe(v.name, "f16", v.bm, v.bn, v.bk, v.wm, v.wn, v.stages, v.raster, v.pad, Act::None, false, false, false)
            } else {
                entry_smem_pipe(v.name, "f16", v.bm, v.bn, v.bk, v.wm, v.wn, v.stages, v.raster, Act::None, false, false)
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
                wh.bm, wh.bn, wh.bk, wh.wm, wh.wn, wh.stages, wh.raster, wh.pad,
                Act::None, false, false, true,
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
                p64.bm, p64.bn, p64.bk, p64.wm, p64.wn, p64.stages, p64.raster,
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
            p64.bm, p64.bn, p64.bk, p64.wm, p64.wn, p64.stages, p64.raster,
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
        for (suffix, act) in [("bias", Act::None), ("bias_relu", Act::Relu), ("bias_silu", Act::Silu), ("bias_gelu", Act::Gelu)] {
            m += &entry_mma_pipe(
                &format!("{}_{suffix}", wh.name),
                "f16",
                wh.bm, wh.bn, wh.bk, wh.wm, wh.wn, wh.stages, wh.raster, wh.pad,
                act,
                true,
                false,
                false,
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
            wh.bm, wh.bn, wh.bk, wh.wm, wh.wn, wh.stages, wh.raster, wh.pad,
            Act::None,
            true,
            true,
            false,
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
            m += &entry_mma_gate(&format!("mma_nt_f16_128x64_{suffix}"), "f16", 128, 64, 32, 2, 4, 2, 16, 8, act, gbias);
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
/// `cp.async` double-buffered `wmma_nt_bf16_sm_db`, and the fused-epilogue `wmma_nt_bf16_sm_db_{relu,
/// silu,gelu}`. The pipelined + fused generators are precision-generic (`entry_smem_db` keys the
/// fragment width / mma type off `ty`), so bf16 — the dominant *training* precision — gets the same
/// beat-the-cuBLAS-chain fusion as fp16.
pub fn wmma_bf16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("wmma_nt_bf16", "bf16", 1, 1);
        m += &entry("wmma_nt_bf16_mt", "bf16", TM_TILES, TN_TILES);
        // bf16 large-GEMM workhorse (mma.sync + padded conflict-free SMEM + r16 raster) — the cliff fix
        // carried to the training precision; `gemm_nt_bf16` dispatches A+B ≳ L2 here.
        let v = PIPE_BF16;
        m += &entry_mma_pipe(v.name, "bf16", v.bm, v.bn, v.bk, v.wm, v.wn, v.stages, v.raster, v.pad, Act::None, false, false, false);
        // bf16 ldmatrix+XOR-swizzle+no-pad twin (`_swz`) — the HBM-bound-4096³ win carried to the training
        // dtype (the swz path is dtype-agnostic; `gemm_nt_bf16` regime-dispatches it for A+B ≳ 2×L2).
        m += &entry_mma_pipe(&format!("{}_swz", v.name), "bf16", v.bm, v.bn, v.bk, v.wm, v.wn, v.stages, v.raster, v.pad, Act::None, false, false, true);
        // Fused-epilogue variants on the **fast bf16 mma workhorse** — the register-level `act(x·Wᵀ+bias)`
        // (bias added to the f32 accumulators via the known D-fragment column map, no SMEM scratch) carried
        // to the training dtype. The bf16 twin of the fp16 `mma_nt_f16_128_bk32_s2_r16_bias*` champions.
        for (suffix, act) in [("bias", Act::None), ("bias_relu", Act::Relu), ("bias_silu", Act::Silu), ("bias_gelu", Act::Gelu)] {
            m += &entry_mma_pipe(
                &format!("{}_{suffix}", v.name),
                "bf16",
                v.bm, v.bn, v.bk, v.wm, v.wn, v.stages, v.raster, v.pad,
                act,
                true,
                false,
                false,
            );
        }
        // bf16 fused bias + residual (training down-proj / output-proj): out = x·Wᵀ + bias + residual.
        m += &entry_mma_pipe(
            &format!("{}_bias_residual", v.name),
            "bf16",
            v.bm, v.bn, v.bk, v.wm, v.wn, v.stages, v.raster, v.pad,
            Act::None,
            true,
            true,
            false,
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
            m += &entry_mma_gate(&format!("mma_nt_bf16_128x64_{suffix}"), "bf16", 128, 64, 32, 2, 4, 2, 16, 8, act, gbias);
        }
        m += &entry_smem_db("wmma_nt_bf16_sm_db", "bf16", SM_BM, SM_BN, SM_WARPS_M, SM_WARPS_N, Act::None, false, false);
        for (suffix, act) in [("relu", Act::Relu), ("silu", Act::Silu), ("gelu", Act::Gelu)] {
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
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
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
