//! **fp8 (E4M3) tensor cores** on Ada (`sm_89`). Unlike fp16/bf16, fp8 has **no WMMA** path on
//! `sm_89` — it is the warp-level `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` only, which
//! requires loading the A/B fragments into registers in the exact per-lane layout the PTX ISA
//! defines (no `wmma.load` to do it for us). This module pins that layout with a single 16×8 output
//! tile (M=16, N=8, K=32 per `mma`) so it can be validated against a CPU reference with **asymmetric,
//! e4m3-exact** data (all-ones would hide a layout bug); the full tiled GEMM builds on it once the
//! core is proven. f32 accumulate (the mixed-precision contract).

/// OCP **E4M3** (1 sign, 4 exp bias 7, 3 mantissa; max normal 448, no Inf) round-to-nearest-even from
/// `f32`, returning the 8 stored bits. Exact for the e4m3-representable values the validation uses;
/// subnormals (|x| below 2⁻⁶) flush toward zero — fine here, the test data is normal.
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

/// Generate the **pipelined fp8 (E4M3) GEMM** entry — `mma.sync.m16n8k32` with a multi-stage `cp.async`
/// SMEM pipeline, padded conflict-free fragment loads, and threadblock rasterization. Mirrors the
/// fp16/bf16 `entry_mma_pipe` but for 1-byte e4m3 and the K=32 mma step. `C = A·Bᵀ`, A `[M,K]` / B `[N,K]`
/// row-major, f32 accumulate. Per-warp tile `(bm/wm)×(bn/wn)` = `tm` m16-blocks × `tn` n8-blocks;
/// requires `M%bm==0`, `N%bn==0`, `K%bk==0`, `bk%32==0`, `bk%16==0` chunking, `bm%(16·wm)==0`,
/// `bn%(8·wn)==0`, and `bm·bk`,`bn·bk` multiples of `threads·16` (128-bit staging). Static SMEM
/// `stages·(bm+bn)·(bk+pad)` ≤ 48 KiB.
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
) -> String {
    use crate::ptx_wmma::Act;
    assert!(stages >= 2 && bk % 32 == 0 && (bk / 16).is_power_of_two() && pad % 16 == 0);
    assert!(bm % (16 * warps_m) == 0 && bn % (8 * warps_n) == 0);
    assert!(raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()));
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m);
    let tn = bn / (8 * warps_n);
    let nks = bk / 32; // m16n8k32 k-steps per staged tile
    let ldp = bk + pad; // padded SMEM row stride (bytes; 1 byte/e4m3)
    let (tile_a, tile_b) = (bm * ldp, bn * ldp);
    let (smem_a, smem_b) = (stages * tile_a, stages * tile_b);
    assert!(smem_a + smem_b <= 48 * 1024, "{name}: fp8 SMEM {} B exceeds 48 KiB", smem_a + smem_b);
    let a_chunks = bm * bk / (threads * 16); // 16-byte (16×e4m3) cp.async chunks
    let b_chunks = bn * bk / (threads * 16);
    assert!(a_chunks >= 1 && b_chunks >= 1, "{name}: tile too small for one 128-bit chunk/thread");
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
    let resid_param = if residual { ",\n    .param .u64 pResid" } else { "" };
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{bias_param}{resid_param}\n)\n{{\n"
    );
    s += &format!("    .shared .align 16 .b8 smemA_{name}[{smem_a}];\n");
    s += &format!("    .shared .align 16 .b8 smemB_{name}[{smem_b}];\n");
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
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    s += &format!("    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n");
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
    let stage = |g_base: &str, gbase_ptr: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
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

    // Compute: per k32 step, build A/B fragment base ptrs (smem + buffer + warp·ldp + laneoff + ks·32),
    // ld.shared.b32 the hand-placed fragments, issue tm·tn mma.sync m16n8k32.
    for ks in 0..nks {
        s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    add.u32 %aptr,%aptr,%tmp;\n");
        s += &format!("    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n", ks * 32);
        for mi in 0..tm {
            let base = mi * 16 * ldp; // m16-block row offset (bytes, padded stride)
            let r8 = 8 * ldp;
            s += &format!("    ld.shared.b32 %a{mi}_0,[%aptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %a{mi}_2,[%aptr+{}];\n", base + 16); // k+16 (second half of the 32-k tile)
            s += &format!("    ld.shared.b32 %a{mi}_1,[%aptr+{}];\n", base + r8);
            s += &format!("    ld.shared.b32 %a{mi}_3,[%aptr+{}];\n", base + r8 + 16);
        }
        s += &format!("    mov.u32 %bptr,smemB_{name};\n    add.u32 %bptr,%bptr,%bufcB;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!("    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n", ks * 32);
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
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {d},{a},{b},{d};\n");
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
    assert!(stages >= 2 && bk % 32 == 0 && (bk / 16).is_power_of_two() && pad % 16 == 0);
    assert!(bm % (16 * warps_m) == 0 && bn % (8 * warps_n) == 0);
    assert!(raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()));
    let threads = warps_m * warps_n * 32;
    let tm = bm / (16 * warps_m);
    let tn = bn / (8 * warps_n);
    let nks = bk / 32;
    let ldp = bk + pad; // padded SMEM row stride (bytes; 1 byte/e4m3)
    let (tile_a, tile_b) = (bm * ldp, bn * ldp);
    let (smem_a, smem_b) = (stages * tile_a, stages * tile_b);
    assert!(smem_a + 2 * smem_b <= 48 * 1024, "{name}: fp8 gate SMEM {} B exceeds 48 KiB", smem_a + 2 * smem_b);
    let a_chunks = bm * bk / (threads * 16);
    let b_chunks = bn * bk / (threads * 16);
    assert!(a_chunks >= 1 && b_chunks >= 1, "{name}: tile too small for one 128-bit chunk/thread");
    let bk_chunks = bk / 16;
    let row_shift = bk_chunks.trailing_zeros();
    let col_mask = bk_chunks - 1;
    let wn_shift = warps_n.trailing_zeros();
    let (wmr, wnc) = (bm / warps_m, bn / warps_n);

    let bias_param = if bias { ",\n    .param .u64 pBiasG,\n    .param .u64 pBiasU" } else { "" };
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
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n", warps_n - 1);
    s += &format!("    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n");
    s += &format!("    mul.lo.s32 %laneoff,%grp,{ldp};\n    add.u32 %laneoff,%laneoff,%tg4;\n");
    for mi in 0..tm {
        for ni in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.f32 %dg{mi}_{ni}_{r},0f00000000;\n    mov.f32 %du{mi}_{ni}_{r},0f00000000;\n");
            }
        }
    }

    let stage = |g_base: &str, gbase_ptr: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
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

    // Compute: load A fragments ONCE, preload BOTH B tiles (Wg, Wu), then interleave the independent
    // gate/up mma's (the ILP win that flipped the fp16 gate to a win — see entry_mma_gate).
    for ks in 0..nks {
        s += &format!("    mov.u32 %aptr,smemA_{name};\n    add.u32 %aptr,%aptr,%bufcA;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpMrow,{ldp};\n    add.u32 %aptr,%aptr,%tmp;\n");
        s += &format!("    add.u32 %aptr,%aptr,%laneoff;\n    add.u32 %aptr,%aptr,{};\n", ks * 32);
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
        s += &format!("    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n", ks * 32);
        for ni in 0..tn {
            let base = ni * 8 * ldp;
            s += &format!("    ld.shared.b32 %bg{ni}_0,[%bptr+{}];\n", base);
            s += &format!("    ld.shared.b32 %bg{ni}_1,[%bptr+{}];\n", base + 16);
        }
        s += &format!("    mov.u32 %bptr,smemBu_{name};\n    add.u32 %bptr,%bptr,%bufcBu;\n");
        s += &format!("    mul.lo.s32 %tmp,%warpNcol,{ldp};\n    add.u32 %bptr,%bptr,%tmp;\n");
        s += &format!("    add.u32 %bptr,%bptr,%laneoff;\n    add.u32 %bptr,%bptr,{};\n", ks * 32);
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

/// Pipelined fp8 GEMM module — entry `fp8_gemm_pipe` (see [`fp8_pipe_entry`] / `FP8_PIPE_*`).
pub fn fp8_pipe_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        use crate::ptx_wmma::Act;
        let mut m = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
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
            m += &fp8_gate_entry(&format!("fp8_gemm_pipe_{suffix}"), 128, 64, 64, 2, 4, 2, 16, 16, act, gbias);
        }
        m
    })
    .as_str()
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
        let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
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
        String::from(
            r#".version 8.4
.target sm_89
.address_size 64

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
"#,
        )
    })
    .as_str()
}
