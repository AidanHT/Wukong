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
