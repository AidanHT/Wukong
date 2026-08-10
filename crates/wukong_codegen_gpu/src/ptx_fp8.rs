//! **fp8 (E4M3) tensor cores** on Ada (`sm_89`). Unlike fp16/bf16, fp8 has **no WMMA** path on
//! `sm_89` — it is the warp-level `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` only, which
//! requires loading the A/B fragments into registers in the exact per-lane layout the PTX ISA
//! defines (no `wmma.load` to do it for us). This module pins that layout with a single 16×8 output
//! tile (M=16, N=8, K=32 per `mma`) so it can be validated against a CPU reference with **asymmetric,
//! e4m3-exact** data (all-ones would hide a layout bug), and everything else in this file builds on
//! that proven fragment layout: [`fp8_gemm_ptx`] (one 16×8 tile per warp), [`fp8_gemm_mt_ptx`]
//! (fragment-reuse), and [`fp8_pipe_ptx`] (the `cp.async`-pipelined workhorse plus its fused
//! bias/activation/residual and gated-FFN entries). f32 accumulate (the mixed-precision contract).
//!
//! **Target floor: `sm_89`, and it is real.** Every module here issues
//! `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, an instruction that exists nowhere below
//! Ada — so unlike the int8/int4 families (which share the geometry but use Ampere-legal operand
//! types and therefore float down to `sm_80`), this file must stay tagged `sm_89`. The floor is
//! spelled once, via [`crate::ptx_target::HDR_SM89_V84`] / [`crate::ptx_target::TARGET_SM89`], so a
//! `rg sm_89` over the backend shows exactly which families are genuinely Ada-only. The *device*
//! side of that floor is a `cc >= (8,9)` capability gate before dispatch, in `gpu.rs` — not here.

use crate::gpu::{smem_mode_for, SmemMode, DSMEM_DECL, DSMEM_SYM, STATIC_SMEM_CAP};
use crate::ptx_target::HDR_SM89_V84;

/// OCP **E4M3** (1 sign, 4 exp bias 7, 3 mantissa; max normal 448, no Inf) round-to-nearest-even from
/// `f32`, returning the 8 stored bits. Exact for the e4m3-representable values the validation uses;
/// subnormals (|x| below 2⁻⁶) flush toward zero and `-0.0` returns `+0` (the `x == 0.0` early return)
/// — fine here, the test data is normal. KNOWN GAP: the E5M2 twin
/// [`crate::ptx_fp8_train::f32_to_e5m2`] encodes both, because it is gated byte-for-byte against Ada's
/// `cvt.rn.satfinite.e5m2x2.f32`; this encoder has no such device gate, so if an E4M3 tile is ever
/// quantized on-device (`quantize_scaled_e4m3`) and compared against a host-quantized one, these two
/// cases will disagree.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7f;
    }
    let a = a.min(448.0); // saturate to max normal
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127; // unbiased f32 exponent
    let mant = bits & 0x7f_ffff;
    let exp = e + 7; // e4m3 biased exponent
    if exp <= 0 {
        return sign; // flush subnormals to zero (test data avoids this range)
    }
    // round the 23-bit mantissa to 3 bits, ties-to-even
    let shift = 23 - 3;
    let round_bias = (1u32 << (shift - 1)) - 1 + ((mant >> shift) & 1);
    let m3 = (mant + round_bias) >> shift;
    let (exp, m3) = if m3 == 8 { (exp + 1, 0) } else { (exp, m3) }; // mantissa carry
    if exp > 15 {
        return sign | 0x7e; // max normal 448
    }
    sign | ((exp as u8) << 3) | (m3 as u8)
}

/// Widen E4M3 stored bits back to `f32` (the value the tensor core multiplies) — the reference twin
/// of [`f32_to_e4m3`], used by the CPU oracle so it multiplies exactly what the GPU does.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let m = (b & 0x07) as f32;
    if exp == 0 {
        sign * (m / 8.0) * 2f32.powi(-6) // subnormal
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(exp - 7)
    }
}

/// One `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` tile: A is `16×32` e4m3 row-major, B is
/// `32×8` e4m3 column-major (the `.row.col` operand layout), D = A·B is `16×8` f32 row-major. One
/// warp; the per-lane fragment addresses below are the PTX-ISA layout for 8-bit `m16n8k32`
/// (groupID = laneid≫2, threadID-in-group = laneid&3; A packs 4 e4m3 per .b32 register).
///
/// A `const` cannot interpolate [`HDR_SM89_V84`], so the header is spelled literally here and pinned
/// against the constant by `tests::fp8_ptx_is_ascii_and_structural`. `sm_89` is the *true* floor: the
/// `e4m3` `mma` below does not exist on Ampere.
pub const FP8_TILE: &str = r#".version 8.4
.target sm_89
.address_size 64

.visible .entry fp8_tile(
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pC
)
{
    .reg .b32 %lane,%grp,%tg,%a0,%a1,%a2,%a3,%b0,%b1,%off;
    .reg .f32 %d0,%d1,%d2,%d3,%z;
    .reg .b64 %A,%B,%C,%ab,%bb,%cb,%t;

    ld.param.u64 %A,[pA];
    ld.param.u64 %B,[pB];
    ld.param.u64 %C,[pC];
    cvta.to.global.u64 %A,%A;
    cvta.to.global.u64 %B,%B;
    cvta.to.global.u64 %C,%C;

    mov.u32 %lane,%tid.x;
    shr.u32 %grp,%lane,2;        // groupID = laneid >> 2  (0..7)
    and.b32 %tg,%lane,3;         // threadID-in-group = laneid & 3 (0..3)

    // A (16x32 row-major): a0=[grp, tg*4..], a1=[grp+8, ..], a2=[grp,16+tg*4..], a3=[grp+8,16+..]
    mul.lo.s32 %off,%grp,32;
    shl.b32 %tg,%tg,2;           // tg*4 (byte offset of the 4-wide pack)
    add.s32 %off,%off,%tg;
    cvt.u64.u32 %t,%off;
    add.s64 %ab,%A,%t;
    ld.global.b32 %a0,[%ab];
    ld.global.b32 %a1,[%ab+256];     // +8 rows * 32 cols
    ld.global.b32 %a2,[%ab+16];
    ld.global.b32 %a3,[%ab+272];     // +8 rows + 16 cols

    // B (32x8 col-major): b0=[tg*4.., grp], b1=[16+tg*4.., grp]; col grp is contiguous K (stride 1)
    mul.lo.s32 %off,%grp,32;         // column grp starts at grp*32 (col-major, K=32 per column)
    add.s32 %off,%off,%tg;           // + tg*4
    cvt.u64.u32 %t,%off;
    add.s64 %bb,%B,%t;
    ld.global.b32 %b0,[%bb];
    ld.global.b32 %b1,[%bb+16];

    mov.f32 %z,0f00000000;
    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%z,%z,%z,%z};

    // D (16x8 f32 row-major): d0=[grp,tg2], d1=[grp,tg2+1], d2=[grp+8,tg2], d3=[grp+8,tg2+1]
    and.b32 %tg,%lane,3;
    shl.b32 %tg,%tg,1;               // threadID-in-group * 2 (column)
    mul.lo.s32 %off,%grp,8;
    add.s32 %off,%off,%tg;           // (grp*8 + tg*2) elements
    shl.b32 %off,%off,2;             // * 4 bytes
    cvt.u64.u32 %t,%off;
    add.s64 %cb,%C,%t;
    st.global.f32 [%cb],%d0;
    st.global.f32 [%cb+4],%d1;
    st.global.f32 [%cb+256],%d2;     // +8 rows * 8 cols * 4 bytes
    st.global.f32 [%cb+260],%d3;
    ret;
}
"#;

/// Pipelined fp8 GEMM config — the bf16/f16 mma-pipeline recipe carried to E4M3 (the cliff fix for fp8,
/// which otherwise ran on the un-staged global-load `fp8_gemm_nt_mt` path). 128×128 CTA tile, BK=64
/// staged K-slice (2 `m16n8k32` k-steps/tile), 8 warps (2×4), 2-stage `cp.async`, r16 rasterization, and
/// **16-byte SMEM row padding** so the b32 (4×e4m3) fragment loads are bank-conflict-free
/// (grp·(BK+16)/4 mod 32 spans every multiple of 4 ⇒ the 32 lanes hit 32 banks). fp8 SMEM is 1 byte/elem
/// (half fp16's) so the padded 40 KiB tile fits the static-shared cap with room to spare.
pub const FP8_PIPE_BM: usize = 128;
/// Small/mid-M CTA tile height for the `fp8_gemm_pipe_m64` variant — half [`FP8_PIPE_BM`] doubles the
/// CTA count; the higher occupancy wins for M≤2048 (see [`crate::gpu::gemm_nt_fp8_pipe`]'s dispatch).
pub const FP8_PIPE_M64_BM: usize = 64;
pub const FP8_PIPE_BN: usize = 128;
pub const FP8_PIPE_BK: usize = 64;
pub const FP8_PIPE_WM: usize = 2;
pub const FP8_PIPE_WN: usize = 4;
pub const FP8_PIPE_STAGES: usize = 2;
pub const FP8_PIPE_RASTER: usize = 16;
pub const FP8_PIPE_PAD: usize = 16;
/// Threads per CTA for the pipelined fp8 kernel.
pub const FP8_PIPE_THREADS: usize = FP8_PIPE_WM * FP8_PIPE_WN * 32;

/// Comma-joined `{%p0,%p1,...}` register vector.
fn fp8_veclist(prefix: &str, n: usize) -> String {
    let regs: Vec<String> = (0..n).map(|i| format!("%{prefix}{i}")).collect();
    format!("{{{}}}", regs.join(","))
}

/// `add.u32 <reg>,<reg>,<off>;` for a slab's **constant** base inside the shared dynamic window — and
/// the **empty string** at offset 0.
///
/// Emitting nothing at 0 is the whole point: on the static path every ring already sits at offset 0 of
/// its own `.shared` array, so the generated text stays byte-identical to the shipped kernels' (D6 §5.2).
/// Only when both rings are carved out of the one `wk_dsmem` window does the B ring need its base added.
fn fp8_slab_add(reg: &str, off: usize) -> String {
    if off == 0 {
        String::new()
    } else {
        format!("    add.u32 {reg},{reg},{off};\n")
    }
}

/// Generate the **pipelined fp8 (E4M3) GEMM** entry — `mma.sync.m16n8k32` with a multi-stage `cp.async`
/// SMEM pipeline, padded conflict-free fragment loads, and threadblock rasterization. Mirrors the
/// fp16/bf16 `entry_mma_pipe` but for 1-byte e4m3 and the K=32 mma step. `C = A·Bᵀ`, A `[M,K]` / B `[N,K]`
/// row-major, f32 accumulate. Per-warp tile `(bm/wm)×(bn/wn)` = `tm` m16-blocks × `tn` n8-blocks;
/// requires `M%bm==0`, `N%bn==0`, `K%bk==0`, `bk%32==0` with `bk/16` a power of two (shift-based
/// staging address math), `pad%16==0`, `bm%(16·wm)==0`, `bn%(8·wn)==0`, a **power-of-two warp grid**
/// (`warpRow`/`warpCol` are a shift and a mask), and `bm·bk`,`bn·bk` multiples of `threads·16`
/// (128-bit staging). Every one of the *tile-shape*
/// conditions is asserted below, so an illegal tile config panics at generation instead of emitting a
/// silently wrong kernel. The `M`/`N`/`K` divisibility is the CALLER's: they are runtime kernel params
/// (`pM`/`pN`/`pK`), not generator arguments, so nothing here can check them — the launch wrapper must.
///
/// **`smem_budget` (bytes) is the ceiling this entry may spend, and it also selects the emission form**
/// ([`crate::gpu::smem_mode_for`]): at or below the PTX ISA's 48 KiB **static** cap the `.shared` arrays
/// are declared exactly as they always were — **byte-identical bodies**, so every shipped fp8 entry is
/// untouched — and beyond it both rings are carved out of ONE module-scope [`crate::gpu::DSMEM_DECL`]
/// window at constant offsets (A at 0, B at `stages*bm*(bk+pad)`). The generator stays a pure text
/// function: the budget is *passed in* by the dispatch layer from `Gpu::smem_budget()`, never probed
/// here, so the whole stage grid is enumerable off-device and an A100/H100 budget is testable on this
/// laptop. SMEM(stages) = `stages*(bm+bn)*(bk+pad)` — at 1 byte per e4m3 the slabs are half fp16's, which
/// is exactly why pipeline depth is cheaper for fp8 than for any other dtype (D6 4.1, candidate 2).
///
/// The window itself is declared by the MODULE builder ([`fp8_stage_ptx`]), never here: module scope is
/// mandatory, and the identical `.extern .shared` line inside an `.entry` body is `CUDA_ERROR_INVALID_PTX`.
#[allow(clippy::too_many_arguments)]
fn fp8_pipe_entry(
    name: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
    pad: usize,
    act: crate::ptx_wmma::Act,
    bias: bool,
    residual: bool,
    smem_budget: usize,
) -> String {
    use crate::ptx_wmma::Act;
    assert!(
        stages >= 2
            && bk.is_multiple_of(32)
            && (bk / 16).is_power_of_two()
            && pad.is_multiple_of(16)
    );
    assert!(bm.is_multiple_of(16 * warps_m) && bn.is_multiple_of(8 * warps_n));
    assert!(raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()));
    // `warpRow = warpId >> wn_shift` / `warpCol = warpId & (warps_n-1)` only partition the warps when
    // the warp grid is powers of two; with e.g. warps_n=3 the shift is 0, so warpRow is the raw warp id
    // and warps 2.. address rows outside the CTA tile. (The int8 swz twin is immune only by accident —
    // its `a_tile.is_power_of_two()` assert forces a power-of-two `bn`, which `8*3` cannot divide.)
    assert!(
        warps_m.is_power_of_two() && warps_n.is_power_of_two(),
        "{name}: warp grid must be powers of two (warpRow/warpCol are a shift and a mask)"
    );
    let threads = warps_m * warps_n * 32;
    // The doc above states `bm*bk`/`bn*bk` must be multiples of `threads*16`, but `a_chunks`/`b_chunks`
    // are TRUNCATING divisions: a tile that is not a whole multiple stages only part of itself and
    // feeds uninitialised shared memory straight into `mma.sync` — silently wrong C, no diagnostic.
    assert!(
        (bm * bk).is_multiple_of(threads * 16) && (bn * bk).is_multiple_of(threads * 16),
        "{name}: threads*16 must divide the A/B tile bytes (128-bit cp.async staging)"
    );
    let tm = bm / (16 * warps_m);
    let tn = bn / (8 * warps_n);
    let nks = bk / 32; // m16n8k32 k-steps per staged tile
    let ldp = bk + pad; // padded SMEM row stride (bytes; 1 byte/e4m3)
    let (tile_a, tile_b) = (bm * ldp, bn * ldp);
    let (smem_a, smem_b) = (stages * tile_a, stages * tile_b);
    // SMEM(stages) = stages*(bm+bn)*(bk+pad) — the family's closed form. The BUDGET is the ceiling; the
    // 48 KiB PTX ISA static cap (STATIC_SMEM_CAP) decides the emission FORM, not the ceiling.
    let mode = smem_mode_for(smem_a + smem_b);
    assert!(
        smem_a + smem_b <= smem_budget,
        "{name}: fp8 SMEM {} B (stages={stages}, {bm}x{bn}, ldp={ldp}) exceeds the budget {smem_budget} B",
        smem_a + smem_b
    );
    // Where each ring lives. Static: its own entry-name-qualified `.shared` array at offset 0 — the
    // historical spelling, emitted verbatim. Dynamic: BOTH rings are windows into the single
    // module-scope `wk_dsmem` (two module-scope externs ALIAS, measured), A at 0 and B after the A ring.
    let (sym_a, sym_b, off_b): (String, String, usize) = if mode.is_dynamic() {
        // Sub-slab alignment inside the single window: every `cp.async ...,16` and every
        // `ld.shared.b32` into the B ring assumes a 16-B-aligned destination, and the window base
        // itself is only `.align 16`. Loud at generation, because a misaligned slab offset is a
        // silently-wrong kernel rather than a driver error (D6 risk 4).
        assert!(
            smem_a % 16 == 0,
            "{name}: the B ring's window offset {smem_a} B is not 16-B aligned (cp.async ...,16 needs it)"
        );
        (DSMEM_SYM.to_string(), DSMEM_SYM.to_string(), smem_a)
    } else {
        (format!("smemA_{name}"), format!("smemB_{name}"), 0)
    };
    let a_chunks = bm * bk / (threads * 16); // 16-byte (16×e4m3) cp.async chunks
    let b_chunks = bn * bk / (threads * 16);
    assert!(
        a_chunks >= 1 && b_chunks >= 1,
        "{name}: tile too small for one 128-bit chunk/thread"
    );
    let bk_chunks = bk / 16;
    let row_shift = bk_chunks.trailing_zeros();
    let col_mask = bk_chunks - 1;
    let wn_shift = warps_n.trailing_zeros();
    let (wmr, wnc) = (bm / warps_m, bn / warps_n);

    // The fused-bias variant takes a `bias[N]` (f32) param applied per output column in the store
    // epilogue — the canonical `act(A·Bᵀ + bias)` fp8 Linear/FFN form (cuBLAS needs a 2nd kernel for it).
    let bias_param = if bias { ",\n    .param .u64 pBias" } else { "" };
    // Fused-residual: a `residual[M,N]` (f32) added to the post-activation accumulators before the store,
    // out = act(A·Bᵀ + bias) + residual — the fp8 transformer down-proj / output-proj (cf. entry_mma_pipe).
    let resid_param = if residual {
        ",\n    .param .u64 pResid"
    } else {
        ""
    };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n{{\n"
    );
    // A dynamic entry declares NO static `.shared` array: it would count against the same opt-in
    // ceiling the window is sized from, silently shrinking the window the launch may request.
    if !mode.is_dynamic() {
        s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
        s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
    }
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufcA,%bufcB,%bufwA,%bufwB,%lane,%grp,%tg,%tg4,%tg2,%laneoff,%warpMrow,%warpNcol,%aptr,%bptr,%grow,%gcol;\n";
    // Fused-epilogue scratch: %act0/%act1 for the transcendental activations, %biasv0/%biasv1 for the two
    // bias columns this lane's D fragment spans, %Bias for the bias base pointer (cf. entry_mma_pipe).
    if !matches!(act, Act::None) {
        s += "    .reg .f32 %act0,%act1;\n";
    }
    if bias {
        s += "    .reg .f32 %biasv0,%biasv1;\n    .reg .b64 %Bias;\n";
    }
    if residual {
        s += "    .reg .f32 %resv0,%resv1;\n    .reg .b64 %Resid;\n";
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
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += "    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg4,%tg,2;\n    shl.b32 %tg2,%tg,1;\n";
    s += &format!(
        "    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n",
        warps_n - 1
    );
    s += &format!(
        "    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n"
    );
    // per-lane SMEM byte offset shared by A and B fragment loads: grp·ldp + tg·4 (k byte offset).
    s += &format!("    mul.lo.s32 %laneoff,%grp,{ldp};\n    add.u32 %laneoff,%laneoff,%tg4;\n");
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %d{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }

    // cp.async staging into the padded SMEM layout (row stride ldp bytes). 16-byte chunks = 16 e4m3:
    // flat elem = e·16, row r=e>>row_shift, col c=(e&col_mask)·16; global byte = (g_base+r)·K + kcol + c
    // (1 byte/elem), SMEM dest = bufoff + r·ldp + c.
    let stage = |g_base: &str,
                 gbase_ptr: &str,
                 smem: &str,
                 slab_off: usize,
                 bufoff: &str,
                 chunks: usize,
                 s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,4;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    cvt.u64.u32 %off,%tmp;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            // Always through the SYMBOL, never a literal base: the dynamic window does not start at 0
            // (it begins after whatever statics the entry declares — measured 3072 with 3 KiB of them).
            *s += &format!("    mov.u32 %tmp,{smem};\n");
            *s += &fp8_slab_add("%tmp", slab_off);
            *s += &format!("    add.u32 %tmp,%tmp,{bufoff};\n");
            *s += &format!("    mul.lo.s32 %tmp2,%r,{ldp};\n    add.u32 %tmp,%tmp,%tmp2;\n    add.u32 %tmp,%tmp,%c;\n");
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
        stage("%baseRow", "%A", &sym_a, 0, "%bufwA", a_chunks, &mut s);
        stage("%baseCol", "%B", &sym_b, off_b, "%bufwB", b_chunks, &mut s);
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
    stage("%baseRow", "%A", &sym_a, 0, "%bufwA", a_chunks, &mut s);
    stage("%baseCol", "%B", &sym_b, off_b, "%bufwB", b_chunks, &mut s);
    s += &format!("NOPRE_{name}:\n    cp.async.commit_group;\n");

    // Compute: per k32 step, build A/B fragment base ptrs (smem + buffer + warp·ldp + laneoff + ks·32),
    // ld.shared.b32 the hand-placed fragments, issue tm·tn mma.sync m16n8k32.
    for ks in 0..nks {
        s += &format!("    mov.u32 %aptr,{sym_a};\n    add.u32 %aptr,%aptr,%bufcA;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    add.u32 %aptr,%aptr,%tmp;\n");
        s += &format!(
            "    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n",
            ks * 32
        );
        for mi in 0..tm {
            let base = mi * 16 * ldp; // m16-block row offset (bytes, padded stride)
            let r8 = 8 * ldp;
            s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16); // k+16 (second half of the 32-k tile)
            s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
            s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
        }
        s += &format!("    mov.u32 %bptr,{sym_b};\n");
        s += &fp8_slab_add("%bptr", off_b);
        s += "    add.u32 %bptr,%bptr,%bufcB;\n";
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!(
            "    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n",
            ks * 32
        );
        for ni in 0..tn {
            let base = ni * 8 * ldp;
            s += &format!("    ld.shared.b32 %b{ni}_0,[%bptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %b{ni}_1,[%bptr+{}];\n", base + 16);
        }
        for mi in 0..tm {
            for ni in 0..tn {
                let d = fp8_veclist(&format!("d{mi}_{ni}_"), 4);
                let a = fp8_veclist(&format!("a{mi}_"), 4);
                let b = fp8_veclist(&format!("b{ni}_"), 2);
                s += &format!(
                    "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {d},{a},{b},{d};\n"
                );
            }
        }
    }
    s += &format!("    add.u32 %bufcA,%bufcA,{tile_a};\n    setp.ge.u32 %pmore,%bufcA,{smem_a};\n    @%pmore sub.u32 %bufcA,%bufcA,{smem_a};\n");
    s += &format!("    add.u32 %bufcB,%bufcB,{tile_b};\n    setp.ge.u32 %pmore,%bufcB,{smem_b};\n    @%pmore sub.u32 %bufcB,%bufcB,{smem_b};\n");
    s += &format!("    add.u32 %bufwA,%bufwA,{tile_a};\n    setp.ge.u32 %pmore,%bufwA,{smem_a};\n    @%pmore sub.u32 %bufwA,%bufwA,{smem_a};\n");
    s += &format!("    add.u32 %bufwB,%bufwB,{tile_b};\n    setp.ge.u32 %pmore,%bufwB,{smem_b};\n    @%pmore sub.u32 %bufwB,%bufwB,{smem_b};\n");
    s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    for mi in 0..tm {
        for ni in 0..tn {
            s += &format!("    add.u32 %grow,%baseRow,%warpMrow;\n    add.u32 %grow,%grow,{};\n    add.u32 %grow,%grow,%grp;\n", mi * 16);
            s += &format!("    add.u32 %gcol,%baseCol,%warpNcol;\n    add.u32 %gcol,%gcol,{};\n    add.u32 %gcol,%gcol,%tg2;\n", ni * 8);
            // Fused epilogue (identical to entry_mma_pipe — the m16n8k32 D-fragment column map matches
            // m16n8k16): C = act(A·Bᵀ + bias). d0,d2 sit at column gcol, d1,d3 at gcol+1.
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
            s += "    mul.lo.s32 %tmp,%grow,%N;\n    add.u32 %tmp,%tmp,%gcol;\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %cptr,%C,%off;\n";
            if residual {
                s += "    add.s64 %cptr2,%Resid,%off;\n    ld.global.f32 %resv0,[%cptr2];\n    ld.global.f32 %resv1,[%cptr2+4];\n";
                s += &format!("    add.f32 %d{mi}_{ni}_0,%d{mi}_{ni}_0,%resv0;\n    add.f32 %d{mi}_{ni}_1,%d{mi}_{ni}_1,%resv1;\n");
            }
            s += &format!("    st.global.f32 [%cptr],%d{mi}_{ni}_0;\n    st.global.f32 [%cptr+4],%d{mi}_{ni}_1;\n");
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

/// fp8 (E4M3) twin of [`crate::ptx_wmma::entry_mma_gate`] — the fused **gated-FFN** kernel
/// `out = act(x·Wgᵀ [+bg]) ⊙ (x·Wuᵀ [+bu])` (SwiGLU/GeGLU/GLU) on `mma.sync.m16n8k32`. This is the
/// **fastest fused inference gate**: Ada runs fp8 tensor cores at 2× the fp16 rate, and the whole
/// gate — two GEMMs sharing one staged `x` tile, the activation on the gate branch, the elementwise
/// product — lands in one kernel, the three-kernel chain cuBLAS needs collapsed away. The 128×64 dual-B
/// tile is register-neutral (two f32 accumulator sets = the same 64 D-regs as one set at 128×128) and
/// fits 40 KiB easily (fp8 is 1 byte/elem). The `m16n8k32` D-fragment column map matches `m16n8k16`, so
/// the per-column bias add and `Act::epilogue` are reused verbatim; `bias` adds `bg[N]`,`bu[N]`. Mirrors
/// [`fp8_pipe_entry`]'s staging/fragment layout exactly, only with a second B (Wu) and the gate epilogue.
fn fp8_gate_entry(
    name: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
    pad: usize,
    gate_act: crate::ptx_wmma::Act,
    bias: bool,
) -> String {
    use crate::ptx_wmma::Act;
    assert!(
        stages >= 2
            && bk.is_multiple_of(32)
            && (bk / 16).is_power_of_two()
            && pad.is_multiple_of(16)
    );
    assert!(bm.is_multiple_of(16 * warps_m) && bn.is_multiple_of(8 * warps_n));
    assert!(raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()));
    // Same two preconditions as `fp8_pipe_entry` (this generator mirrors its staging/fragment layout
    // verbatim): the warp grid is addressed by shift+mask, and the cp.async chunk counts truncate.
    assert!(
        warps_m.is_power_of_two() && warps_n.is_power_of_two(),
        "{name}: warp grid must be powers of two (warpRow/warpCol are a shift and a mask)"
    );
    let threads = warps_m * warps_n * 32;
    assert!(
        (bm * bk).is_multiple_of(threads * 16) && (bn * bk).is_multiple_of(threads * 16),
        "{name}: threads*16 must divide the A/B tile bytes (128-bit cp.async staging)"
    );
    let tm = bm / (16 * warps_m);
    let tn = bn / (8 * warps_n);
    let nks = bk / 32;
    let ldp = bk + pad; // padded SMEM row stride (bytes; 1 byte/e4m3)
    let (tile_a, tile_b) = (bm * ldp, bn * ldp);
    let (smem_a, smem_b) = (stages * tile_a, stages * tile_b);
    // The gated-FFN family stays entirely on the STATIC path: three rings at 128x64 is 40 KiB, well
    // inside the ISA cap, and no deep-stage candidate in D6 4.1 targets it. Spelled through
    // `STATIC_SMEM_CAP` so the one boundary constant is the one every fp8 generator reads.
    assert!(
        smem_a + 2 * smem_b <= STATIC_SMEM_CAP,
        "{name}: fp8 gate SMEM {} B exceeds the 48 KiB static ISA cap",
        smem_a + 2 * smem_b
    );
    let a_chunks = bm * bk / (threads * 16);
    let b_chunks = bn * bk / (threads * 16);
    assert!(
        a_chunks >= 1 && b_chunks >= 1,
        "{name}: tile too small for one 128-bit chunk/thread"
    );
    let bk_chunks = bk / 16;
    let row_shift = bk_chunks.trailing_zeros();
    let col_mask = bk_chunks - 1;
    let wn_shift = warps_n.trailing_zeros();
    let (wmr, wnc) = (bm / warps_m, bn / warps_n);

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
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufcA,%bufcBg,%bufcBu,%bufwA,%bufwBg,%bufwBu,%lane,%grp,%tg,%tg4,%tg2,%laneoff,%warpMrow,%warpNcol,%aptr,%bptr,%grow,%gcol;\n";
    if !matches!(gate_act, Act::None) {
        s += "    .reg .f32 %act0,%act1;\n";
    }
    if bias {
        s += "    .reg .f32 %biasg0,%biasg1,%biasu0,%biasu1;\n    .reg .b64 %BiasG,%BiasU;\n";
    }
    if raster > 0 {
        s += "    .reg .b32 %lin,%tn,%tm,%gsz,%grpr,%rem,%col0,%gw,%trow,%tcol;\n";
    }
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
    s += "    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg4,%tg,2;\n    shl.b32 %tg2,%tg,1;\n";
    s += &format!(
        "    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n",
        warps_n - 1
    );
    s += &format!(
        "    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n"
    );
    s += &format!("    mul.lo.s32 %laneoff,%grp,{ldp};\n    add.u32 %laneoff,%laneoff,%tg4;\n");
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %dg{mi}_{ni}_{r},0f00000000;\n    mov.f32 %du{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }

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
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,4;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    cvt.u64.u32 %off,%tmp;\n    add.s64 %gptr,{gbase_ptr},%off;\n");
            *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n");
            *s += &format!("    mul.lo.s32 %tmp2,%r,{ldp};\n    add.u32 %tmp,%tmp,%tmp2;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

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

    // Compute: load A fragments ONCE, preload BOTH B tiles (Wg, Wu), then interleave the independent
    // gate/up mma's (the ILP win that flipped the fp16 gate to a win — see entry_mma_gate).
    for ks in 0..nks {
        s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    add.u32 %aptr,%aptr,%tmp;\n");
        s += &format!(
            "    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n",
            ks * 32
        );
        for mi in 0..tm {
            let base = mi * 16 * ldp;
            let r8 = 8 * ldp;
            s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16);
            s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
            s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
        }
        s += &format!("    mov.u32 %bptr,smemBg_{name};\n    add.u32 %bptr,%bptr,%bufcBg;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!(
            "    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n",
            ks * 32
        );
        for ni in 0..tn {
            let base = ni * 8 * ldp;
            s += &format!("    ld.shared.b32 %bg{ni}_0,[%bptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %bg{ni}_1,[%bptr+{}];\n", base + 16);
        }
        s += &format!("    mov.u32 %bptr,smemBu_{name};\n    add.u32 %bptr,%bptr,%bufcBu;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!(
            "    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n",
            ks * 32
        );
        for ni in 0..tn {
            let base = ni * 8 * ldp;
            s += &format!("    ld.shared.b32 %bu{ni}_0,[%bptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %bu{ni}_1,[%bptr+{}];\n", base + 16);
        }
        for mi in 0..tm {
            for ni in 0..tn {
                let dg = fp8_veclist(&format!("dg{mi}_{ni}_"), 4);
                let du = fp8_veclist(&format!("du{mi}_{ni}_"), 4);
                let a = fp8_veclist(&format!("a{mi}_"), 4);
                let bgv = fp8_veclist(&format!("bg{ni}_"), 2);
                let buv = fp8_veclist(&format!("bu{ni}_"), 2);
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {dg},{a},{bgv},{dg};\n");
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {du},{a},{buv},{du};\n");
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

/// Pipelined fp8 GEMM module (see [`fp8_pipe_entry`] / `FP8_PIPE_*`). Twelve entries — the 128×128
/// `fp8_gemm_pipe`, the small/mid-M `fp8_gemm_pipe_m64`, the fused `fp8_gemm_pipe_bias{,_relu,_silu,
/// _gelu}` / `_bias_residual`, and the five `fp8_gemm_pipe_gate_*` dual-B gated-FFN tiles; the exact
/// list and order is pinned by `tests::PIPE_ENTRIES`.
pub fn fp8_pipe_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        use crate::ptx_wmma::Act;
        let mut m = format!("{HDR_SM89_V84}\n");
        m += &fp8_pipe_entry(
            "fp8_gemm_pipe",
            FP8_PIPE_BM,
            FP8_PIPE_BN,
            FP8_PIPE_BK,
            FP8_PIPE_WM,
            FP8_PIPE_WN,
            FP8_PIPE_STAGES,
            FP8_PIPE_RASTER,
            FP8_PIPE_PAD,
            Act::None,
            false,
            false,
            STATIC_SMEM_CAP,
        );
        // **Small/mid-M tile** (`fp8_gemm_pipe_m64`, 64×128) — the regime-aware win the cuBLASLt-fp8
        // sweep found (`fp8_pipe_config_sweep_vs_cublaslt`): halving BM to 64 doubles the CTA count, and
        // the higher occupancy beats the 128×128 default at M≤2048 by a wide same-run margin (e.g.
        // 2048³ ~97% vs ~78% of cuBLASLt) while only tying at 4096³ — so [`crate::gpu::gemm_nt_fp8_pipe`]
        // dispatches here for M≤2048 (and for any M where 128∤M but 64∣M) and to the 128×128 entry above
        // for the larger sizes. Same kernel/codegen, only BM=64 → bit-identical accumulation, so it
        // rides the same E4M3 tolerance gate.
        m += &fp8_pipe_entry(
            "fp8_gemm_pipe_m64",
            FP8_PIPE_M64_BM,
            FP8_PIPE_BN,
            FP8_PIPE_BK,
            FP8_PIPE_WM,
            FP8_PIPE_WN,
            FP8_PIPE_STAGES,
            FP8_PIPE_RASTER,
            FP8_PIPE_PAD,
            Act::None,
            false,
            false,
            STATIC_SMEM_CAP,
        );
        // Fused-epilogue fp8 variants — the beat-cuBLAS fusion carried to the **fastest** precision (Ada
        // 2× TC rate), so `C = act(x·Wᵀ + bias)` fp8 Linear/FFN is the fastest fused inference path. The
        // m16n8k32 D-fragment column map matches m16n8k16, so the register-level bias+act epilogue (no
        // SMEM scratch) is reused verbatim from `entry_mma_pipe` via `Act::epilogue`.
        for (suffix, act) in [
            ("bias", Act::None),
            ("bias_relu", Act::Relu),
            ("bias_silu", Act::Silu),
            ("bias_gelu", Act::Gelu),
        ] {
            m += &fp8_pipe_entry(
                &format!("fp8_gemm_pipe_{suffix}"),
                FP8_PIPE_BM,
                FP8_PIPE_BN,
                FP8_PIPE_BK,
                FP8_PIPE_WM,
                FP8_PIPE_WN,
                FP8_PIPE_STAGES,
                FP8_PIPE_RASTER,
                FP8_PIPE_PAD,
                act,
                true,
                false,
                STATIC_SMEM_CAP,
            );
        }
        // fp8 fused bias + residual (no act) — the fastest down-proj / output-proj: out = x·Wᵀ + bias +
        // residual at the Ada 2× fp8 rate (the residual stream stays f32, fp8 only the GEMM operands).
        m += &fp8_pipe_entry(
            "fp8_gemm_pipe_bias_residual",
            FP8_PIPE_BM,
            FP8_PIPE_BN,
            FP8_PIPE_BK,
            FP8_PIPE_WM,
            FP8_PIPE_WN,
            FP8_PIPE_STAGES,
            FP8_PIPE_RASTER,
            FP8_PIPE_PAD,
            Act::None,
            true,
            true,
            STATIC_SMEM_CAP,
        );
        // Fused **gated-FFN (SwiGLU/GeGLU/GLU)** gate at the Ada 2× fp8 rate — the fastest fused inference
        // gate. `out = act(x·Wgᵀ) ⊙ (x·Wuᵀ)` in one dual-B kernel (128×64, fp8 = 1 byte/elem so three
        // staged tiles fit 40 KiB); the chain cuBLAS needs three kernels for. silu→SwiGLU, gelu→GeGLU,
        // none→bilinear GLU; `_bias` adds the per-column gate/up biases.
        for (suffix, act, gbias) in [
            ("gate_silu", Act::Silu, false),
            ("gate_gelu", Act::Gelu, false),
            ("gate_glu", Act::None, false),
            ("gate_silu_bias", Act::Silu, true),
            ("gate_gelu_bias", Act::Gelu, true),
        ] {
            m += &fp8_gate_entry(
                &format!("fp8_gemm_pipe_{suffix}"),
                128,
                64,
                64,
                2,
                4,
                2,
                16,
                16,
                act,
                gbias,
            );
        }
        m
    })
    .as_str()
}

/// Build the pipelined fp8 GEMM (entry `fp8_gemm_pipe`) at an **arbitrary tile/pipeline config** — the
/// sweep hook for chasing the cuBLASLt-fp8 % (M2). Identical kernel to [`fp8_pipe_ptx`]'s default
/// entry, only with `bm/bn/bk/warps/stages/raster` chosen by the caller (pad fixed at [`FP8_PIPE_PAD`]).
/// Returns an owned module string; the caller must cache it under a *distinct* key (the JIT cache keys
/// by module key, so two configs sharing a key would collide). [`fp8_pipe_entry`]'s asserts enforce the
/// divisibility/SMEM constraints, so an illegal config panics at build rather than miscompiling.
pub fn fp8_pipe_cfg_ptx(
    bm: usize,
    bn: usize,
    bk: usize,
    warps_m: usize,
    warps_n: usize,
    stages: usize,
    raster: usize,
) -> String {
    use crate::ptx_wmma::Act;
    let mut m = format!("{HDR_SM89_V84}\n");
    m += &fp8_pipe_entry(
        "fp8_gemm_pipe",
        bm,
        bn,
        bk,
        warps_m,
        warps_n,
        stages,
        raster,
        FP8_PIPE_PAD,
        Act::None,
        false,
        false,
        // The historical caller: the 48 KiB ISA cap, so every sweep config this builds stays on the
        // static emission path and its PTX is the shipped text to the byte. The deeper rings go through
        // [`fp8_stage_ptx`], which passes the device's opt-in budget instead.
        STATIC_SMEM_CAP,
    );
    m
}

/// **fp8 64×64-warp-tile pipe — the transferred int8 lever (M2, perf/gpu-quant-2).** A 128×128 CTA with
/// `wm=wn=2` ⇒ a **64×64 warp tile** on 4 warps (128 threads), `BK=64`, 2-stage `cp.async`. The int8
/// sweep found that on this 20-SM Ada part the win is the *warp* tile, not the CTA tile: doubling the
/// per-warp A/B fragment reuse (64×64 vs the shipped 64×32 default) lifts throughput at every size. fp8
/// shares the identical `mma.sync.m16n8k32` 8-bit geometry, and the warp-tile sweep
/// ([`crate::gpu::tests`] `quant_fp8_warp_tile_sweep`) confirmed it same-run: 1024/2048/4096³ at
/// 72/132/91% of cuBLASLt vs the 64×32 default's 66/102/86%. Repartitioning *which* warp owns an output
/// element leaves the per-element k=0,32,64,… accumulation order unchanged ⇒ **bit-identical** to the
/// default entry ⇒ rides the same E4M3 tolerance gate. Entry `fp8_gemm_pipe` (launch with 128 threads).
/// Best at 4096³; for M,N≤2048 the 3-stage [`fp8_pipe_w64_s3_ptx`] adds ~10–19 pts.
pub fn fp8_pipe_w64_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| fp8_pipe_cfg_ptx(FP8_PIPE_BM, FP8_PIPE_BN, 64, 2, 2, 2, FP8_PIPE_RASTER))
        .as_str()
}

/// 3-stage (`BK=32`) sibling of [`fp8_pipe_w64_ptx`] — the deeper `cp.async` pipeline that wins at the
/// small/mid square sizes (1024³ 82%, 2048³ 151% of cuBLASLt; ~+10/+19 pts over the 2-stage) but loses
/// the register/SMEM-pressure trade at 4096³ (81% vs 91%), so [`crate::gpu::gemm_nt_fp8_pipe`] routes it
/// only for M,N≤2048. Same `fp8_gemm_pipe` entry / 128-thread launch; same bit-identical K-accumulation.
pub fn fp8_pipe_w64_s3_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| fp8_pipe_cfg_ptx(FP8_PIPE_BM, FP8_PIPE_BN, 32, 2, 2, 3, FP8_PIPE_RASTER))
        .as_str()
}

/// One row of the **fp8 variable-stage grid** — the family's `(tile, warps, depth)` point, its stable
/// module-cache key **and** its PTX entry name in one `&'static str`.
///
/// The name carries the tile *and* the stage count because [`crate::gpu::Gpu::function`] keys the module
/// cache on the key alone and **never re-examines the PTX on a hit**: two depths under one key would
/// silently run the first one's kernel with the second one's launch window — and, worse, inherit its
/// `cuFuncSetAttribute` SMEM ceiling. One row = one key = one entry = one depth.
#[derive(Clone, Copy, Debug)]
pub struct Fp8StageCfg {
    /// PTX entry symbol *and* module-cache key (a `&'static str`, as `Gpu::function` requires).
    pub name: &'static str,
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
    pub stages: usize,
    pub raster: usize,
}

impl Fp8StageCfg {
    /// `stages*(bm+bn)*(bk+pad)` — the family's closed form at 1 byte per e4m3 ([`FP8_PIPE_PAD`] is the
    /// bank-conflict pad every fp8 pipe entry uses).
    pub const fn smem_bytes(&self) -> usize {
        self.stages * (self.bm + self.bn) * (self.bk + FP8_PIPE_PAD)
    }
    pub const fn threads(&self) -> usize {
        self.wm * self.wn * 32
    }
    /// Static at or below the 48 KiB ISA cap, one dynamic window beyond it — the single emission rule.
    pub const fn smem_mode(&self) -> SmemMode {
        smem_mode_for(self.smem_bytes())
    }
    /// Shortest K this depth's prologue fills exactly: the ring stages `stages-1` slabs before the main
    /// loop, so `K = (stages-1)*bk` is the corner where the prologue fills the ring and the loop never
    /// prefetches. Shorter K is *correct* (every prologue slab past K is guarded) but wastes the depth.
    pub const fn min_k(&self) -> usize {
        (self.stages - 1) * self.bk
    }
}

/// **The fp8 deep-stage grid — `mma.sync.m16n8k32` E4M3 rings past the 48 KiB static wall.**
///
/// fp8 is the dtype where pipeline depth is cheapest per byte (1 B/elem, half fp16's slab) and the one
/// whose Ada `mma.sync` programming model *is* Hopper's — so this grid is the campaign's most direct
/// H100-fp8 evidence, and an A100 cannot run a single row of it (no fp8 ISA at cc 8.0).
///
/// The CTAs/SM column is **measured**, not predicted — `cuOccupancyMaxActiveBlocksPerMultiprocessor`
/// on this 4050, printed by `fp8_deep_matches_reference_within_tol` on every run.
///
/// | row | tile | warps | depth | SMEM | form | CTAs/SM here (100 KiB/SM, 1 KiB reserved) |
/// |---|---|---|---|---|---|---|
/// | `fp8_deep_128_s2` | 128x128 | 2x2 | 2 | 40 KiB | static (== the shipped `w64`) | 2 |
/// | `fp8_deep_128_s3` | 128x128 | 2x2 | 3 | 60 KiB | **dynamic** | 1 |
/// | `fp8_deep_128_s4` | 128x128 | 2x2 | 4 | 80 KiB | **dynamic** | 1 |
/// | `fp8_deep_m64_s2` | 64x128 | 2x4 | 2 | 30 KiB | static (== the shipped `_m64`) | 3 |
/// | `fp8_deep_m64_s4` | 64x128 | 2x4 | 4 | 60 KiB | **dynamic** | 1 |
/// | `fp8_deep_m64_s6` | 64x128 | 2x4 | 6 | 90 KiB | **dynamic** | 1 |
///
/// These are exactly D6 4.1's candidate 2 ("fp8 s3/s4 at 60-90 KiB"). **128x128 s5 is deliberately
/// absent**: 5*(128+128)*80 = 102400 B, four bytes past this Ada part's 101376 B opt-in ceiling — it is
/// not a tuning choice but a hard budget miss here, and the same generator produces it unchanged the
/// moment an A100 (163 KiB) or H100 (227 KiB) budget is passed in.
///
/// The `_s2` rows are the shipped kernels under grid names — their PTX is byte-identical bar the entry
/// symbol, which `fp8_deep_grid_s2_rows_are_the_shipped_kernels` proves — so the grid's own correctness
/// gate covers the dispatched pair too, and they are the equivalence anchor every deeper row must match.
///
/// **This grid is a correctness deliverable, not a perf verdict on this card.** Every row past s2 costs
/// occupancy here (60 KiB already pins 1 CTA/SM); the datacenter budgets are what dissolve that. What
/// transfers 100% is the PTX: this is the exact module an H100 will run, validated on the metal at $0.
pub const FP8_DEEP_VARIANTS: &[Fp8StageCfg] = &[
    // 128x128 CTA on the 64x64 warp tile (wm=wn=2, 128 threads) — the measured fp8 dispatch winner.
    Fp8StageCfg {
        name: "fp8_deep_128_s2",
        bm: 128,
        bn: 128,
        bk: FP8_PIPE_BK,
        wm: 2,
        wn: 2,
        stages: 2,
        raster: FP8_PIPE_RASTER,
    }, // 40 KiB static
    Fp8StageCfg {
        name: "fp8_deep_128_s3",
        bm: 128,
        bn: 128,
        bk: FP8_PIPE_BK,
        wm: 2,
        wn: 2,
        stages: 3,
        raster: FP8_PIPE_RASTER,
    }, // 60 KiB dynamic
    Fp8StageCfg {
        name: "fp8_deep_128_s4",
        bm: 128,
        bn: 128,
        bk: FP8_PIPE_BK,
        wm: 2,
        wn: 2,
        stages: 4,
        raster: FP8_PIPE_RASTER,
    }, // 80 KiB dynamic
    // 64x128 small/mid-M tile (wm=2, wn=4, 256 threads) — half the A ring, so it reaches s6 in budget.
    Fp8StageCfg {
        name: "fp8_deep_m64_s2",
        bm: FP8_PIPE_M64_BM,
        bn: 128,
        bk: FP8_PIPE_BK,
        wm: FP8_PIPE_WM,
        wn: FP8_PIPE_WN,
        stages: 2,
        raster: FP8_PIPE_RASTER,
    }, // 30 KiB static
    Fp8StageCfg {
        name: "fp8_deep_m64_s4",
        bm: FP8_PIPE_M64_BM,
        bn: 128,
        bk: FP8_PIPE_BK,
        wm: FP8_PIPE_WM,
        wn: FP8_PIPE_WN,
        stages: 4,
        raster: FP8_PIPE_RASTER,
    }, // 60 KiB dynamic
    Fp8StageCfg {
        name: "fp8_deep_m64_s6",
        bm: FP8_PIPE_M64_BM,
        bn: 128,
        bk: FP8_PIPE_BK,
        wm: FP8_PIPE_WM,
        wn: FP8_PIPE_WN,
        stages: 6,
        raster: FP8_PIPE_RASTER,
    }, // 90 KiB dynamic
];

/// Look up a [`FP8_DEEP_VARIANTS`] row by entry name — a wrong name is a loud panic at the call site,
/// never a silent mis-dispatch.
pub fn fp8_deep_variant(name: &str) -> &'static Fp8StageCfg {
    FP8_DEEP_VARIANTS
        .iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("unknown fp8 deep variant {name:?}"))
}

/// Generate one [`FP8_DEEP_VARIANTS`] row against `smem_budget` bytes (the running device's
/// `Gpu::smem_budget()`, or a target's budget when enumerating off-device). Returns the module and the
/// [`SmemMode`] its launch must honour — `Gpu::function_smem` consumes exactly this pair, so no caller
/// re-derives a byte count the generator already knows.
///
/// One row, one module, one entry: the row's `name` is both the entry symbol and the module-cache key
/// (crate hard rule 4). The emitted text does **not** depend on `smem_budget` — only on the row's own
/// byte count, through [`crate::gpu::smem_mode_for`] — so a bigger card produces byte-identical PTX and
/// the on-disk cubin cache stays warm across devices.
///
/// Panics (loudly, at generation) if the row does not fit `smem_budget`: a decline belongs in the
/// dispatcher, ahead of any load, never in a silently-clamped launch.
pub fn fp8_stage_ptx(v: &Fp8StageCfg, smem_budget: usize) -> (String, SmemMode) {
    use crate::ptx_wmma::Act;
    let mode = v.smem_mode();
    let mut m = format!("{HDR_SM89_V84}\n");
    if mode.is_dynamic() {
        // MODULE SCOPE, not inside the entry — the identical line in an entry body is CUDA_ERROR_INVALID_PTX.
        m += DSMEM_DECL;
    }
    m += &fp8_pipe_entry(
        v.name,
        v.bm,
        v.bn,
        v.bk,
        v.wm,
        v.wn,
        v.stages,
        v.raster,
        FP8_PIPE_PAD,
        Act::None,
        false,
        false,
        smem_budget,
    );
    (m, mode)
}

/// Multi-tile per warp for fp8: `M` direction tiles (each 16 rows) and `N` direction tiles (each 8
/// cols). 2×4 → a 32×32 C block per warp, 8 `mma`s per K-step. Each warp tile size = 16·TM × 8·TN.
pub const FP8_TM: usize = 2;
pub const FP8_TN: usize = 4;

/// **Fragment-reuse fp8 GEMM** — the throughput path. Each warp computes a `FP8_TM×FP8_TN` block of
/// 16×8 tiles, loading each A fragment once and reusing it across all `FP8_TN` B-tiles (and vice
/// versa), so the global-load traffic per `mma` drops by ~`FP8_TN`/`FP8_TM`× and the kernel becomes
/// compute-bound — the lift the naive single-tile [`fp8_gemm_ptx`] lacks. Same E4M3 `mma.sync.m16n8k32`
/// layout, offset by each sub-tile's origin. Entry `fp8_gemm_nt_mt`; requires M%(16·TM)==N%(8·TN)==0.
pub fn fp8_gemm_mt_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        let (tm, tn) = (FP8_TM, FP8_TN);
        let mut s = format!("{HDR_SM89_V84}\n");
        s += ".visible .entry fp8_gemm_nt_mt(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{\n";
        s += "    .reg .pred %p;\n";
        s += "    .reg .b32 %M,%N,%K,%lane,%grp,%tg4,%tg2,%row0,%col0,%k,%tmp;\n";
        // accumulators d[mt][nt][0..3], A frags a[mt][0..3], B frags b[nt][0..1]
        let mut f32regs = String::from("%z");
        for mi in 0..tm {
            for ni in 0..tn {
                for r in 0..4 {
                    f32regs += &format!(",%d{mi}_{ni}_{r}");
                }
            }
        }
        s += &format!("    .reg .f32 {f32regs};\n");
        let mut b32regs = String::new();
        for mi in 0..tm {
            for r in 0..4 {
                b32regs += &format!("%a{mi}_{r},");
            }
        }
        for ni in 0..tn {
            for r in 0..2 {
                b32regs += &format!("%b{ni}_{r},");
            }
        }
        s += &format!("    .reg .b32 {}; \n", b32regs.trim_end_matches(','));
        let mut b64regs = String::from("%A,%B,%C,%t,%kk,%cp");
        for mi in 0..tm {
            b64regs += &format!(",%a0p{mi},%a8p{mi}");
        }
        for ni in 0..tn {
            b64regs += &format!(",%bp{ni}");
        }
        s += &format!("    .reg .b64 {b64regs};\n");

        s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
        s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
        s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
        s += "    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg4,%lane,3;\n";
        s += "    shl.b32 %tg2,%tg4,1;\n    shl.b32 %tg4,%tg4,2;\n";
        s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %row0,%tmp,{};\n", 16 * tm);
        s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %col0,%tmp,{};\n", 8 * tn);

        // per-m-tile A base addresses (rows grp / grp+8), per-n-tile B base addresses
        for mi in 0..tm {
            s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.s32 %tmp,%tmp,%tg4;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %a0p{mi},%A,%t;\n", mi * 16);
            s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.s32 %tmp,%tmp,%tg4;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %a8p{mi},%A,%t;\n", mi * 16 + 8);
        }
        for ni in 0..tn {
            s += &format!("    add.s32 %tmp,%col0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.s32 %tmp,%tmp,%tg4;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %bp{ni},%B,%t;\n", ni * 8);
        }
        // zero accumulators
        for mi in 0..tm {
            for ni in 0..tn {
                for r in 0..4 {
                    s += &format!("    mov.f32 %d{mi}_{ni}_{r},0f00000000;\n");
                }
            }
        }
        s += "    mov.u32 %k,0;\nKLOOP:\n    setp.ge.u32 %p,%k,%K;\n    @%p bra KEND;\n    cvt.u64.u32 %kk,%k;\n";
        // load A frags
        for mi in 0..tm {
            s += &format!("    add.s64 %t,%a0p{mi},%kk;\n    ld.global.b32 %a{mi}_0,[%t];\n    ld.global.b32 %a{mi}_2,[%t+16];\n");
            s += &format!("    add.s64 %t,%a8p{mi},%kk;\n    ld.global.b32 %a{mi}_1,[%t];\n    ld.global.b32 %a{mi}_3,[%t+16];\n");
        }
        // load B frags
        for ni in 0..tn {
            s += &format!("    add.s64 %t,%bp{ni},%kk;\n    ld.global.b32 %b{ni}_0,[%t];\n    ld.global.b32 %b{ni}_1,[%t+16];\n");
        }
        // mma all tiles (A frag reused across N, B frag reused across M)
        for mi in 0..tm {
            for ni in 0..tn {
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n        {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}}, {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}}, {{%b{ni}_0,%b{ni}_1}}, {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}};\n");
            }
        }
        s += "    add.u32 %k,%k,32;\n    bra KLOOP;\nKEND:\n";
        // store each sub-tile
        for mi in 0..tm {
            for ni in 0..tn {
                // C[(row0+mi*16+grp)][col0+ni*8+tg2]
                s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%N;\n    add.s32 %tmp,%tmp,%col0;\n    add.s32 %tmp,%tmp,{};\n    add.s32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %cp,%C,%t;\n", mi * 16, ni * 8);
                s += &format!("    st.global.f32 [%cp],%d{mi}_{ni}_0;\n    st.global.f32 [%cp+4],%d{mi}_{ni}_1;\n");
                s += &format!("    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %cp,%cp,%t;\n    st.global.f32 [%cp],%d{mi}_{ni}_2;\n    st.global.f32 [%cp+4],%d{mi}_{ni}_3;\n");
            }
        }
        s += "    ret;\n}\n";
        s
    })
    .as_str()
}

/// Full **fp8 (E4M3) tensor-core GEMM** `C = A·Bᵀ` (the nn.Linear form): A is `[M,K]` row-major, B is
/// `[N,K]` row-major — which *is* the `K×N` column-major layout the `mma` `.col` operand wants, so
/// `A·Bᵀ` maps straight onto `mma.row.col` with no transpose. Each warp owns a `16×8` C tile and
/// loops K in steps of 32 (the validated `m16n8k32` layout, now offset by the tile origin and the
/// k-step). f32 accumulate. M%16 == N%8 == K%32 == 0. Returns the module (entry `fp8_gemm_nt`).
pub fn fp8_gemm_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        HDR_SM89_V84.to_string()
            + (r#"
.visible .entry fp8_gemm_nt(
    .param .u32 pM,
    .param .u32 pN,
    .param .u32 pK,
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pC
)
{
    .reg .pred %p;
    .reg .b32 %M,%N,%K,%lane,%grp,%tg4,%tg2,%row0,%col0,%k,%tmp;
    .reg .f32 %d0,%d1,%d2,%d3,%z;
    .reg .b32 %a0,%a1,%a2,%a3,%b0,%b1;
    .reg .b64 %A,%B,%C,%a0p,%a8p,%bp,%cp,%t,%kk;

    ld.param.u32 %M,[pM];
    ld.param.u32 %N,[pN];
    ld.param.u32 %K,[pK];
    ld.param.u64 %A,[pA];
    ld.param.u64 %B,[pB];
    ld.param.u64 %C,[pC];
    cvta.to.global.u64 %A,%A;
    cvta.to.global.u64 %B,%B;
    cvta.to.global.u64 %C,%C;

    mov.u32 %lane,%tid.x;
    shr.u32 %grp,%lane,2;
    and.b32 %tg4,%lane,3;
    shl.b32 %tg2,%tg4,1;          // threadID-in-group * 2 (C column within the tile)
    shl.b32 %tg4,%tg4,2;          // threadID-in-group * 4 (K offset of the 4-wide pack)
    mov.u32 %tmp,%ctaid.y;
    mul.lo.s32 %row0,%tmp,16;     // tile row origin
    mov.u32 %tmp,%ctaid.x;
    mul.lo.s32 %col0,%tmp,8;      // tile col origin

    // a0p = A + (row0+grp)*K + tg4 ; a8p = A + (row0+grp+8)*K + tg4
    add.s32 %tmp,%row0,%grp;
    mul.lo.s32 %tmp,%tmp,%K;
    add.s32 %tmp,%tmp,%tg4;
    cvt.u64.u32 %t,%tmp;
    add.s64 %a0p,%A,%t;
    add.s32 %tmp,%row0,%grp;
    add.s32 %tmp,%tmp,8;
    mul.lo.s32 %tmp,%tmp,%K;
    add.s32 %tmp,%tmp,%tg4;
    cvt.u64.u32 %t,%tmp;
    add.s64 %a8p,%A,%t;
    // bp = B + (col0+grp)*K + tg4   (B is [N,K] row-major == KxN col-major)
    add.s32 %tmp,%col0,%grp;
    mul.lo.s32 %tmp,%tmp,%K;
    add.s32 %tmp,%tmp,%tg4;
    cvt.u64.u32 %t,%tmp;
    add.s64 %bp,%B,%t;

    mov.f32 %d0,0f00000000;
    mov.f32 %d1,0f00000000;
    mov.f32 %d2,0f00000000;
    mov.f32 %d3,0f00000000;
    mov.u32 %k,0;
KLOOP:
    setp.ge.u32 %p,%k,%K;
    @%p bra KEND;
    cvt.u64.u32 %kk,%k;
    add.s64 %t,%a0p,%kk;
    ld.global.b32 %a0,[%t];
    ld.global.b32 %a2,[%t+16];
    add.s64 %t,%a8p,%kk;
    ld.global.b32 %a1,[%t];
    ld.global.b32 %a3,[%t+16];
    add.s64 %t,%bp,%kk;
    ld.global.b32 %b0,[%t];
    ld.global.b32 %b1,[%t+16];
    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%d0,%d1,%d2,%d3};
    add.u32 %k,%k,32;
    bra KLOOP;
KEND:
    // C[row0+grp][col0+tg2] etc, row-major [M,N], f32
    add.s32 %tmp,%row0,%grp;
    mul.lo.s32 %tmp,%tmp,%N;
    add.s32 %tmp,%tmp,%col0;
    add.s32 %tmp,%tmp,%tg2;
    shl.b32 %tmp,%tmp,2;
    cvt.u64.u32 %t,%tmp;
    add.s64 %cp,%C,%t;
    st.global.f32 [%cp],%d0;
    st.global.f32 [%cp+4],%d1;
    // +8 rows = +8*N elements * 4 bytes
    mul.lo.s32 %tmp,%N,32;       // 8 rows * N * 4 bytes
    cvt.u64.u32 %t,%tmp;
    add.s64 %cp,%cp,%t;
    st.global.f32 [%cp],%d2;
    st.global.f32 [%cp+4],%d3;
    ret;
}
"#)
    })
    .as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx_target::TARGET_SM89;
    use crate::ptx_wmma::DEEP_SMEM_BUDGET;

    /// One `.visible .entry <name>(...)` lifted out of a (possibly multi-entry) module, header excluded.
    ///
    /// The entry terminator is the column-0 `"\n}\n"`: PTX register vectors (`{%d0,%d1,%d2,%d3}`) only
    /// ever appear mid-line and every instruction line ends in `;`, so the closing brace of an `.entry`
    /// is the sole occurrence of that pattern. This is what makes a byte-identity comparison possible
    /// between a single-entry grid module and one entry of the twelve-entry shipped pipe module.
    fn entry_text(module: &str, name: &str) -> String {
        let start = module
            .find(&format!(".visible .entry {name}("))
            .unwrap_or_else(|| panic!("{name}: not declared in this module"));
        let end = module[start..]
            .find("\n}\n")
            .unwrap_or_else(|| panic!("{name}: entry never closes"))
            + start
            + 3;
        module[start..end].to_string()
    }

    /// The `.visible .entry` names `fp8_pipe_ptx()` must declare, in emission order.
    const PIPE_ENTRIES: [&str; 12] = [
        "fp8_gemm_pipe",
        "fp8_gemm_pipe_m64",
        "fp8_gemm_pipe_bias",
        "fp8_gemm_pipe_bias_relu",
        "fp8_gemm_pipe_bias_silu",
        "fp8_gemm_pipe_bias_gelu",
        "fp8_gemm_pipe_bias_residual",
        "fp8_gemm_pipe_gate_silu",
        "fp8_gemm_pipe_gate_gelu",
        "fp8_gemm_pipe_gate_glu",
        "fp8_gemm_pipe_gate_silu_bias",
        "fp8_gemm_pipe_gate_gelu_bias",
    ];

    /// **§3A P1 — PTX stays ASCII**, plus the structural minimum, over every module this file emits.
    /// A single non-ASCII byte turns all of `fp8_pipe_ptx()`'s entries into a `ptxas fatal` at
    /// driver-JIT time; on a GPU-less box every fp8 test skips, so nothing else keeps this file ASCII.
    /// Pure-CPU.
    #[test]
    fn fp8_ptx_is_ascii_and_structural() {
        let mut modules: Vec<(String, String)> = vec![
            ("fp8_tile", FP8_TILE.to_string()),
            ("fp8_gemm_nt", fp8_gemm_ptx().to_string()),
            ("fp8_gemm_nt_mt", fp8_gemm_mt_ptx().to_string()),
            ("fp8_gemm_pipe", fp8_pipe_ptx().to_string()),
            ("fp8_gemm_pipe(w64)", fp8_pipe_w64_ptx().to_string()),
            ("fp8_gemm_pipe(w64_s3)", fp8_pipe_w64_s3_ptx().to_string()),
            (
                "fp8_gemm_pipe(cfg)",
                fp8_pipe_cfg_ptx(128, 128, 64, 2, 4, 2, 16),
            ),
        ]
        .into_iter()
        .map(|(n, p)| (n.to_string(), p))
        .collect();
        // Every deep-stage row, static and dynamic alike — the `.extern` window is ASCII too, and a row
        // that lost its Ada floor or its E4M3 mma would be a load failure on the very parts this exists
        // to reach. Generated at the deep budget so the >48 KiB rows take the dynamic arm here.
        for v in FP8_DEEP_VARIANTS {
            modules.push((
                format!("deep/{}", v.name),
                fp8_stage_ptx(v, DEEP_SMEM_BUDGET).0,
            ));
        }
        let modules = modules;
        for (label, ptx) in &modules {
            assert!(ptx.is_ascii(), "{label}: PTX must be ASCII");
            // The fp8 floor is GENUINE, not a leftover device tag: the `e4m3` mma asserted below
            // exists nowhere below Ada, so every module here must stay pinned to the family floor
            // `TARGET_SM89` (never to whatever card the round happens to run on).
            assert!(
                ptx.contains(TARGET_SM89),
                "{label}: must carry the fp8 family floor {TARGET_SM89}"
            );
            assert!(
                ptx.starts_with(HDR_SM89_V84),
                "{label}: must open with the routed HDR_SM89_V84 header"
            );
            assert_eq!(
                ptx.matches('{').count(),
                ptx.matches('}').count(),
                "{label}: unbalanced braces"
            );
            assert!(
                ptx.contains("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"),
                "{label}: must issue the E4M3 mma"
            );
        }
        // Every entry of the multi-entry pipe module must actually be declared — the dispatch in
        // `gpu::gemm_nt_fp8_pipe` and the fused-epilogue launchers look these names up by string.
        let pipe = fp8_pipe_ptx();
        for e in PIPE_ENTRIES {
            assert!(
                pipe.contains(&format!(".visible .entry {e}(")),
                "fp8_pipe_ptx: missing entry {e}"
            );
        }
        assert_eq!(
            pipe.matches(".visible .entry ").count(),
            PIPE_ENTRIES.len(),
            "fp8_pipe_ptx entry count changed - update PIPE_ENTRIES"
        );
    }

    /// **The E4M3 host codec is the fp8 oracle, so it must be pinned independently.** Every fp8 gate
    /// quantizes the kernel's inputs with [`f32_to_e4m3`] *and* re-decodes the same bits for the
    /// reference, so a regression in the encoder moves both sides together and no gate notices. These
    /// are hand-derived bit patterns (bias 7, 3 mantissa bits, max normal 448, no Inf).
    #[test]
    fn e4m3_encoding_matches_hand_derived_bits() {
        for (x, bits) in [
            (0.0f32, 0x00u8),
            (-0.0f32, 0x00), // both zeros encode as +0 (the `x == 0.0` early return)
            (1.0, 0x38),     // exp 7 << 3, m3 = 0
            (-1.0, 0xb8),    // sign bit set
            (2.0, 0x40),     // exp 8 << 3
            (448.0, 0x7e),   // max normal: exp 15, m3 = 6
            (1000.0, 0x7e),  // saturates to max normal (E4M3 has no Inf)
            (f32::INFINITY, 0x7e),
            (-f32::INFINITY, 0xfe),
            (2f32.powi(-9), 0x00), // below the subnormal range -> flushed to zero
            (1.0625, 0x38),        // exact tie between m3=0 and m3=1 -> ties-to-even picks 0
            (1.1875, 0x3a),        // exact tie between m3=1 and m3=2 -> ties-to-even picks 2
            (1.99, 0x40),          // rounds up through the mantissa carry into exp+1 (= 2.0)
        ] {
            assert_eq!(f32_to_e4m3(x), bits, "f32_to_e4m3({x}) != {bits:#04x}");
        }
        assert!(
            f32_to_e4m3(f32::NAN) & 0x7f == 0x7f,
            "NaN must encode as the OCP NaN pattern"
        );
        // Every *normal* encoding must survive decode->encode unchanged. Excluded by construction:
        // exp==0 (zero + the subnormals this codec flushes) and 0x7f/0xff (the OCP NaN, which
        // `e4m3_to_f32` decodes as the finite 480 and the encoder then saturates to 448 = 0x7e).
        for b in 0u8..=255 {
            let (exp, m) = ((b >> 3) & 0x0f, b & 0x07);
            if exp == 0 || (exp == 15 && m == 7) {
                continue;
            }
            assert_eq!(
                f32_to_e4m3(e4m3_to_f32(b)),
                b,
                "e4m3 round-trip failed for {b:#04x}"
            );
        }
    }

    /// **The fp8 generators must reject a config they cannot codegen.** `a_chunks`/`b_chunks` are
    /// truncating integer divisions, so a tile whose bytes are not a whole multiple of `threads*16`
    /// silently stages only part of itself and feeds uninitialised shared memory to `mma.sync`; and
    /// `warpRow`/`warpCol` are derived by shift+mask, which is only a partition when the warp grid is
    /// powers of two. Both are documented at the top of `fp8_pipe_entry` and both were unchecked; the
    /// three sibling generators in `ptx_int8.rs` / `ptx_int4.rs` assert them.
    #[test]
    #[should_panic(expected = "must divide the A/B tile bytes")]
    fn pipe_rejects_a_tile_that_threads16_does_not_divide() {
        // bm=96, threads=256: bm*bk = 6144 but threads*16 = 4096, so a_chunks truncates 1.5 -> 1 and
        // a third of every staged A tile would never be written.
        let _ = fp8_pipe_cfg_ptx(96, 128, 64, 2, 4, 2, 0);
    }

    /// Sibling of the above for the warp grid: `warps_n = 3` makes `wn_shift = 0`, so `warpRow` is the
    /// raw warp id and `warpCol` is masked with 2 - warps 2..5 address rows outside the CTA tile.
    #[test]
    #[should_panic(expected = "warp grid must be powers of two")]
    fn pipe_rejects_a_non_power_of_two_warp_grid() {
        let _ = fp8_pipe_cfg_ptx(128, 192, 64, 2, 3, 2, 0);
    }

    /// **ZERO REGRESSION on the <=48 KiB path: the grid's `_s2` rows ARE the shipped kernels, byte for
    /// byte below the entry symbol.** The whole budget migration rests on one promise — that adding a
    /// budget parameter and an extern-window arm changes *nothing* about the kernels this card already
    /// runs and whose cubins are already warm. A behavioural A/B could only sample that; this proves it.
    ///
    /// Both anchors are real dispatch targets: `fp8_deep_128_s2` reproduces `fp8_pipe_w64_ptx()` (the
    /// 64x64-warp-tile entry `gemm_nt_fp8_pipe` routes 128-divisible shapes to) and `fp8_deep_m64_s2`
    /// reproduces the `fp8_gemm_pipe_m64` entry inside the twelve-entry shipped pipe module (the 128-M
    /// fallback). Rename the grid entry back to the shipped symbol and demand string equality.
    #[test]
    fn fp8_deep_grid_s2_rows_are_the_shipped_kernels() {
        for (grid, shipped_name, shipped_module) in [
            (
                "fp8_deep_128_s2",
                "fp8_gemm_pipe",
                fp8_pipe_w64_ptx().to_string(),
            ),
            (
                "fp8_deep_m64_s2",
                "fp8_gemm_pipe_m64",
                fp8_pipe_ptx().to_string(),
            ),
        ] {
            let v = fp8_deep_variant(grid);
            let (ptx, mode) = fp8_stage_ptx(v, DEEP_SMEM_BUDGET);
            assert_eq!(
                mode,
                SmemMode::Static,
                "{grid} is {} B — it must stay on the static path",
                v.smem_bytes()
            );
            assert_eq!(
                entry_text(&ptx, grid).replace(grid, shipped_name),
                entry_text(&shipped_module, shipped_name),
                "{grid}: the budget-parameterized generator no longer reproduces the shipped \
                 `{shipped_name}` byte for byte — the <=48 KiB path is NOT allowed to move"
            );
        }
    }

    /// **The deep grid's SMEM arithmetic, emission form and window discipline (no GPU).** Six things a
    /// wrong dynamic-SMEM fp8 kernel gets wrong *silently*, each checked from the generated text:
    ///   * the closed form `stages*(bm+bn)*(bk+pad)` and the 48 KiB boundary that splits the two forms;
    ///   * `.extern .shared` sits at **module scope** — the identical line inside an entry body is
    ///     `CUDA_ERROR_INVALID_PTX` (measured), so its offset must precede `.visible .entry`;
    ///   * exactly **one** window per module: two module-scope externs ALIAS (measured), so a second
    ///     would put the B ring on top of the A ring and quietly compute garbage;
    ///   * a dynamic entry declares **no** static `.shared` array (it would count against the same
    ///     opt-in ceiling the window is sized from), and a static entry never names the window;
    ///   * the B ring's window offset is the whole A ring, and is 16-B aligned (`cp.async ...,16`);
    ///   * the ring cursor is **add+wrap at every depth**. The two-buffer XOR toggle the int8 family
    ///     started from cycles exactly two buffers, so it would corrupt every stage past the second
    ///     (D6 risk 2); fp8 must never grow one.
    #[test]
    fn fp8_deep_grid_smem_math_and_modes() {
        // (stages, bytes, dynamic) per row, spelled out rather than recomputed — a table that agrees
        // with the formula by construction would notice nothing if both moved together.
        let expect: [(usize, usize, bool); 6] = [
            (2, 40960, false),
            (3, 61440, true),
            (4, 81920, true),
            (2, 30720, false),
            (4, 61440, true),
            (6, 92160, true),
        ];
        assert_eq!(
            FP8_DEEP_VARIANTS.len(),
            expect.len(),
            "grid length changed — update `expect`"
        );
        for (v, (stages, bytes, dynamic)) in FP8_DEEP_VARIANTS.iter().zip(expect) {
            assert_eq!(v.stages, stages, "{}: grid order", v.name);
            assert_eq!(v.smem_bytes(), bytes, "{}: SMEM closed form", v.name);
            assert_eq!(
                v.smem_mode().is_dynamic(),
                dynamic,
                "{}: emission form at {bytes} B",
                v.name
            );
            assert_eq!(
                v.smem_mode().launch_bytes(),
                if dynamic { bytes } else { 0 },
                "{}: a static row must launch with 0 (its tile is already reserved)",
                v.name
            );
            assert!(
                v.smem_bytes() <= DEEP_SMEM_BUDGET,
                "{}: {bytes} B must fit the smallest target's opt-in ceiling",
                v.name
            );
            let (ptx, mode) = fp8_stage_ptx(v, DEEP_SMEM_BUDGET);
            assert_eq!(
                mode,
                v.smem_mode(),
                "{}: generator and table disagree on the form",
                v.name
            );
            assert_eq!(
                ptx.matches(".extern .shared").count(),
                usize::from(dynamic),
                "{}",
                v.name
            );
            let ring_a = v.stages * v.bm * (v.bk + FP8_PIPE_PAD);
            if dynamic {
                let (decl, entry) = (
                    ptx.find(".extern .shared").expect("window"),
                    ptx.find(".visible .entry").expect("entry"),
                );
                assert!(
                    decl < entry,
                    "{}: the window must be declared at MODULE scope",
                    v.name
                );
                assert!(
                    !ptx.contains(&format!(".shared .align 16 .b8 smemA_{}", v.name)),
                    "{}: no statics beside the window",
                    v.name
                );
                // Both rings address the one window; B starts after the whole A ring, 16-B aligned.
                assert!(
                    ptx.contains(&format!("mov.u32 %bptr,{DSMEM_SYM};")),
                    "{}",
                    v.name
                );
                assert!(
                    ptx.contains(&format!("add.u32 %bptr,%bptr,{ring_a};")),
                    "{}: the B ring must start after the whole A ring ({ring_a} B)",
                    v.name
                );
                assert_eq!(
                    ring_a % 16,
                    0,
                    "{}: window offset must be 16-B aligned",
                    v.name
                );
            } else {
                assert!(
                    ptx.contains(&format!(
                        ".shared .align 16 .b8 smemA_{}[{ring_a}];",
                        v.name
                    )),
                    "{}",
                    v.name
                );
                assert!(
                    !ptx.contains(DSMEM_SYM),
                    "{}: a static kernel must not touch the window",
                    v.name
                );
            }
            // Ring cursor: add+wrap at EVERY depth. fp8 never had the XOR toggle, and must not gain one.
            assert!(
                !ptx.contains("xor.b32 %bufcA"),
                "{}: XOR wrap is invalid past 2 buffers",
                v.name
            );
            assert!(
                ptx.contains(&format!("setp.ge.u32 %pmore,%bufcA,{ring_a};")),
                "{}",
                v.name
            );
            // cp.async bookkeeping: the prologue commits `stages-1` groups (one guarded staging block
            // each, the commit deliberately OUTSIDE the guard because `wait_group` counts positionally),
            // and the steady state keeps `stages-2` in flight.
            assert!(
                ptx.contains(&format!("cp.async.wait_group {};", v.stages - 2)),
                "{}",
                v.name
            );
            assert_eq!(
                ptx.matches("setp.lt.u32 %pmore,%kcol,%K;").count(),
                v.stages,
                "{}: one guard per prologue slab plus the steady-state prefetch guard",
                v.name
            );
            assert_eq!(
                ptx.matches("cp.async.commit_group;").count(),
                v.stages,
                "{}: `stages-1` prologue commits + one per loop iteration, all outside their guards",
                v.name
            );
        }
    }

    /// **An over-budget depth is a loud generation failure, never a clamped launch.** The budget is the
    /// device's `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`; a kernel past it cannot run at all, and the driver's
    /// own rejection (`CUDA_ERROR_INVALID_VALUE`, at launch) names neither the kernel nor the ceiling.
    #[test]
    #[should_panic(expected = "exceeds the budget")]
    fn fp8_deep_over_budget_panics_at_generation() {
        // 128x128 s5 = 5*(128+128)*80 = 102400 B — four bytes past this Ada part's 101376 B opt-in, and
        // the reason the shipped grid stops at s4. It is legal on A100/H100 and this same call proves it.
        let over = Fp8StageCfg {
            name: "fp8_deep_probe_s5",
            bm: 128,
            bn: 128,
            bk: FP8_PIPE_BK,
            wm: 2,
            wn: 2,
            stages: 5,
            raster: FP8_PIPE_RASTER,
        };
        assert_eq!(over.smem_bytes(), 102400);
        let _ = fp8_stage_ptx(&over, DEEP_SMEM_BUDGET);
    }
}
