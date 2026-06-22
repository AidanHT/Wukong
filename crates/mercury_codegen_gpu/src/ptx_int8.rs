//! **int8 (W8A8) tensor cores** on Ada (`sm_89`) — the quantized-inference GEMM. Like fp8, int8 has
//! **no WMMA** path on `sm_89`: it is the warp-level `mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32`
//! only, which needs the A/B fragments loaded into registers in the exact per-lane layout the PTX ISA
//! defines for 8-bit `m16n8k32` (no `wmma.load` to do it). That layout is **identical to fp8's** (both
//! are 8-bit operands in the same `m16n8k32` geometry) — so this module mirrors [`crate::ptx_fp8`] tile
//! for tile, swapping the `mma` type to `.s32.u8.s8.s32` and the accumulators from f32 to **s32**.
//!
//! **Semantics** match Mercury's CPU int8 GEMM (the AVX-VNNI `vpdpbusd` quantized `nn.Linear`):
//! **`u8` activations × `i8` weights → `i32`**, `C = A·Bᵀ`. The integer accumulate is associative and
//! exact **mod 2³²** (no rounding, no reassociation error), so unlike the float kernels this path is
//! **bit-exact** against a CPU `i32` reference — a strictly stronger gate. The MMA without `.satfinite`
//! wraps mod 2³², which is exactly what the CPU reference (wrapping `i32` adds) computes.

/// One `mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32` tile: A is `16×32` **u8** row-major, B is
/// `32×8` **i8** column-major (the `.row.col` operand layout), D = A·B is `16×8` **i32** row-major. One
/// warp; the per-lane fragment addresses are the PTX-ISA layout for 8-bit `m16n8k32` (groupID =
/// laneid≫2, threadID-in-group = laneid&3; A/B pack 4 bytes per `.b32` register). Byte-for-byte the
/// same address math as [`crate::ptx_fp8::FP8_TILE`] — only the `mma` type and accumulator/store width
/// differ (s32, but s32 and f32 are both 4 bytes, so even the store strides are unchanged).
pub const INT8_TILE: &str = r#".version 8.4
.target sm_89
.address_size 64

.visible .entry int8_tile(
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pC
)
{
    .reg .b32 %lane,%grp,%tg,%a0,%a1,%a2,%a3,%b0,%b1,%off;
    .reg .b32 %d0,%d1,%d2,%d3,%z;
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

    // A (16x32 row-major, u8): a0=[grp, tg*4..], a1=[grp+8, ..], a2=[grp,16+tg*4..], a3=[grp+8,16+..]
    mul.lo.s32 %off,%grp,32;
    shl.b32 %tg,%tg,2;           // tg*4 (byte offset of the 4-wide pack)
    add.s32 %off,%off,%tg;
    cvt.u64.u32 %t,%off;
    add.s64 %ab,%A,%t;
    ld.global.b32 %a0,[%ab];
    ld.global.b32 %a1,[%ab+256];     // +8 rows * 32 cols
    ld.global.b32 %a2,[%ab+16];
    ld.global.b32 %a3,[%ab+272];     // +8 rows + 16 cols

    // B (32x8 col-major, i8): b0=[tg*4.., grp], b1=[16+tg*4.., grp]; col grp is contiguous K (stride 1)
    mul.lo.s32 %off,%grp,32;         // column grp starts at grp*32 (col-major, K=32 per column)
    add.s32 %off,%off,%tg;           // + tg*4
    cvt.u64.u32 %t,%off;
    add.s64 %bb,%B,%t;
    ld.global.b32 %b0,[%bb];
    ld.global.b32 %b1,[%bb+16];

    mov.u32 %z,0;
    mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32
        {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%z,%z,%z,%z};

    // D (16x8 i32 row-major): d0=[grp,tg2], d1=[grp,tg2+1], d2=[grp+8,tg2], d3=[grp+8,tg2+1]
    and.b32 %tg,%lane,3;
    shl.b32 %tg,%tg,1;               // threadID-in-group * 2 (column)
    mul.lo.s32 %off,%grp,8;
    add.s32 %off,%off,%tg;           // (grp*8 + tg*2) elements
    shl.b32 %off,%off,2;             // * 4 bytes
    cvt.u64.u32 %t,%off;
    add.s64 %cb,%C,%t;
    st.global.u32 [%cb],%d0;
    st.global.u32 [%cb+4],%d1;
    st.global.u32 [%cb+256],%d2;     // +8 rows * 8 cols * 4 bytes
    st.global.u32 [%cb+260],%d3;
    ret;
}
"#;

/// Multi-tile per warp (mirrors fp8's `FP8_TM`/`FP8_TN`): `M` direction tiles (each 16 rows) and `N`
/// direction tiles (each 8 cols). 2×4 → a 32×32 C block per warp, 8 `mma`s per K-step.
pub const INT8_TM: usize = 2;
pub const INT8_TN: usize = 4;

/// **Fragment-reuse int8 GEMM** — the throughput path. Each warp computes an `INT8_TM×INT8_TN` block of
/// 16×8 tiles, loading each A fragment once and reusing it across all `INT8_TN` B-tiles (and each B
/// fragment across all `INT8_TM` A-tiles), so global-load traffic per `mma` drops ~`TN`/`TM`× and the
/// kernel becomes compute-bound — the lift the naive single-tile [`int8_gemm_ptx`] lacks. Same
/// `mma.sync.m16n8k32` 8-bit layout as fp8's `_mt`, retyped `.s32.u8.s8.s32` with s32 accumulators.
/// Entry `int8_gemm_nt_mt`; requires M%(16·TM)==N%(8·TN)==0, K%32==0.
pub fn int8_gemm_mt_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        let (tm, tn) = (INT8_TM, INT8_TN);
        let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
        s += ".visible .entry int8_gemm_nt_mt(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{\n";
        s += "    .reg .pred %p;\n";
        s += "    .reg .b32 %M,%N,%K,%lane,%grp,%tg4,%tg2,%row0,%col0,%k,%tmp;\n";
        // accumulators d[mt][nt][0..3] (s32), A frags a[mt][0..3], B frags b[nt][0..1]
        let mut accregs = String::from("%z");
        for mi in 0..tm {
            for ni in 0..tn {
                for r in 0..4 {
                    accregs += &format!(",%d{mi}_{ni}_{r}");
                }
            }
        }
        s += &format!("    .reg .b32 {accregs};\n");
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
                    s += &format!("    mov.u32 %d{mi}_{ni}_{r},0;\n");
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
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32\n        {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}}, {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}}, {{%b{ni}_0,%b{ni}_1}}, {{%d{mi}_{ni}_0,%d{mi}_{ni}_1,%d{mi}_{ni}_2,%d{mi}_{ni}_3}};\n");
            }
        }
        s += "    add.u32 %k,%k,32;\n    bra KLOOP;\nKEND:\n";
        // store each sub-tile (i32)
        for mi in 0..tm {
            for ni in 0..tn {
                // C[(row0+mi*16+grp)][col0+ni*8+tg2]
                s += &format!("    add.s32 %tmp,%row0,%grp;\n    add.s32 %tmp,%tmp,{};\n    mul.lo.s32 %tmp,%tmp,%N;\n    add.s32 %tmp,%tmp,%col0;\n    add.s32 %tmp,%tmp,{};\n    add.s32 %tmp,%tmp,%tg2;\n    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %cp,%C,%t;\n", mi * 16, ni * 8);
                s += &format!("    st.global.u32 [%cp],%d{mi}_{ni}_0;\n    st.global.u32 [%cp+4],%d{mi}_{ni}_1;\n");
                s += &format!("    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %t,%tmp;\n    add.s64 %cp,%cp,%t;\n    st.global.u32 [%cp],%d{mi}_{ni}_2;\n    st.global.u32 [%cp+4],%d{mi}_{ni}_3;\n");
            }
        }
        s += "    ret;\n}\n";
        s
    })
    .as_str()
}

/// CTA macro-tile for the SMEM-staged kernel: `BM×BN` C block per CTA, `BK` K-slab per pipeline step
/// (one `mma.sync.m16n8k32` K-step). 64×64 with a 32-deep K-slab; `WARPS_M×WARPS_N` warps cooperate.
pub const INT8_BM: usize = 64;
pub const INT8_BN: usize = 64;
pub const INT8_BK: usize = 32;
pub const INT8_WARPS_M: usize = 2;
pub const INT8_WARPS_N: usize = 2;

/// **SMEM-staged + `cp.async` double-buffered int8 GEMM** — the latency-hiding path that closes the gap
/// to cuBLAS at large sizes. A `BM×BN` CTA tile is computed by `WARPS_M×WARPS_N` warps; each K-step the
/// CTA cooperatively `cp.async`-copies the next `A[BM×BK]` and `B[BN×BK]` slabs into the *alternate* of
/// two SMEM buffers **while the tensor cores consume the current one**, then waits only on the current
/// copy (`wait_group 1`). Fragments are loaded from SMEM (`ld.shared.b32`) in the hand-placed
/// `m16n8k32` per-lane layout (the same addressing as the global path, rebased to the SMEM tile). This
/// is the int8 analogue of `ptx_wmma.rs`'s `entry_smem_db`, but the manual `mma.sync` fragments are
/// loaded by explicit `ld.shared` (there is no `wmma.load` for int8). f32→s32 retype, exact mod 2³².
/// Entry `int8_gemm_nt_smdb`; requires M%BM==0, N%BN==0, K%BK==0.
pub fn int8_gemm_smdb_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb(
            "int8_gemm_nt_smdb",
            INT8_BM,
            INT8_BN,
            INT8_WARPS_M,
            INT8_WARPS_N,
        )
    })
    .as_str()
}

/// CTA macro-tile for the **128×128** SMEM-staged kernel — the bigger-tile / higher-reuse variant for
/// large GEMMs (8 warps, `INT8_BK`-deep K-slab). Each warp owns a 32×64 block (TM=2 16-row × TN=8
/// 8-col subtiles, 16 `mma`s/K-step). 256 threads = `tile_bytes/16` cp.async chunks (one per thread).
pub const INT8_BM128: usize = 128;
pub const INT8_BN128: usize = 128;
pub const INT8_WARPS_M128: usize = 4;
pub const INT8_WARPS_N128: usize = 2;

/// **128×128 SMEM-staged + `cp.async` int8 GEMM** (`int8_gemm_nt_smdb128`) — same pipeline as
/// [`int8_gemm_smdb_ptx`] with a larger CTA tile (more A/B reuse per global load → higher arithmetic
/// intensity at large sizes). Requires M%128==0, N%128==0, K%INT8_BK==0.
pub fn int8_gemm_smdb128_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb(
            "int8_gemm_nt_smdb128",
            INT8_BM128,
            INT8_BN128,
            INT8_WARPS_M128,
            INT8_WARPS_N128,
        )
    })
    .as_str()
}

/// Generate an SMEM-staged + `cp.async` double-buffered int8 GEMM entry for a `bm×bn` CTA tile computed
/// by `warps_m×warps_n` warps. `bm`,`bn` must be multiples of `16*warps_m` / `8*warps_n`; the staging
/// assumes `INT8_BK==32` (16-byte chunks = half a K-slab row) and `threads <= bm*BK/16` so every chunk
/// has a thread. Each generated module owns its own `smemA`/`smemB` (no cross-module symbol clash).
fn gen_int8_smdb(name: &str, bm: usize, bn: usize, wm: usize, wn: usize) -> String {
    {
        let bk = INT8_BK;
        let threads = wm * wn * 32;
        let tm = bm / (16 * wm); // 16-row A subtiles per warp
        let tn = bn / (8 * wn); //  8-col B subtiles per warp
        let tile_bytes = bm * bk; // one A (== one B) tile in bytes (u8); a power of two ⇒ XOR toggles
        debug_assert!(tile_bytes.is_power_of_two());
        assert!(
            threads * 16 <= bm * bk && (bm * bk) % (threads * 16) == 0,
            "smdb staging needs threads*16 to divide the tile bytes"
        );
        let a_chunks = bm * bk / (threads * 16); // 16-byte cp.async chunks per thread
        let b_chunks = bn * bk / (threads * 16);
        let wn_shift = wn.trailing_zeros();

        let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
        s += &format!(".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{{\n");
        s += &format!("    .shared .align 16 .b8 smemA[{}];\n", 2 * tile_bytes);
        s += &format!("    .shared .align 16 .b8 smemB[{}];\n", 2 * tile_bytes);
        s += "    .reg .pred %p0,%pmore;\n";
        s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%ktn,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufc,%bufp,%lane,%grp,%tg4,%tg2,%ab,%cc;\n";
        // accumulators d[ti][tj][0..3], A frags a[ti][0..3], B frags b[tj][0..1]
        let mut accregs = String::new();
        for ti in 0..tm {
            for tj in 0..tn {
                for r in 0..4 {
                    accregs += &format!("%d{ti}_{tj}_{r},");
                }
            }
        }
        s += &format!("    .reg .b32 {};\n", accregs.trim_end_matches(','));
        let mut abregs = String::new();
        for ti in 0..tm {
            for r in 0..4 {
                abregs += &format!("%a{ti}_{r},");
            }
        }
        for tj in 0..tn {
            for r in 0..2 {
                abregs += &format!("%b{tj}_{r},");
            }
        }
        s += &format!("    .reg .b32 {};\n", abregs.trim_end_matches(','));
        s += "    .reg .b64 %A,%B,%C,%off,%gptr,%cp;\n";

        s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
        s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
        s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
        s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
        s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
        s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n";
        s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n");
        s += &format!("    and.b32 %warpCol,%warpId,{};\n", wn - 1);
        s += "    and.b32 %lane,%tix,31;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg4,%lane,3;\n";
        s += "    shl.b32 %tg2,%tg4,1;\n    shl.b32 %tg4,%tg4,2;\n";
        // zero accumulators
        for ti in 0..tm {
            for tj in 0..tn {
                for r in 0..4 {
                    s += &format!("    mov.u32 %d{ti}_{tj}_{r},0;\n");
                }
            }
        }
        s += "    mov.u32 %bufc,0;\n";
        s += &format!("    mov.u32 %bufp,{tile_bytes};\n");

        // Stage the `%kcol` A/B slab into the SMEM buffer at byte offset `bufoff` via cp.async (16-byte
        // chunks). Chunk e: row r=e·16/BK, col c=(e·16)%BK within the slab; src 16 bytes are contiguous
        // in the global row (BK=32 ⇒ c∈{0,16}, c+15<32). dst is the shared u32 address smem+bufoff+e·16.
        let stage = |g_base: &str, gptr_base: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
            for li in 0..chunks {
                if li == 0 {
                    *s += "    mov.u32 %e,%tix;\n";
                } else {
                    *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
                }
                // r = e*16/BK, c = (e*16)%BK. For BK=32: r=e>>1, c=(e&1)*16.
                *s += "    shr.u32 %r,%e,1;\n    and.b32 %c,%e,1;\n    shl.b32 %c,%c,4;\n";
                *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
                *s += &format!("    cvt.u64.u32 %off,%tmp;\n    add.s64 %gptr,{gptr_base},%off;\n");
                *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n");
                *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
            }
        };

        // Prologue: prefetch slab 0 into buffer 0.
        s += "    mov.u32 %kcol,0;\n";
        stage("%baseRow", "%A", "smemA", "%bufc", a_chunks, &mut s);
        stage("%baseCol", "%B", "smemB", "%bufc", b_chunks, &mut s);
        s += "    cp.async.commit_group;\n";

        s += "    mov.u32 %kt,0;\n";
        s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
        // Prefetch the next slab into the alternate buffer (if any), then wait on the current slab only.
        s += &format!("    add.u32 %ktn,%kt,{bk};\n    setp.lt.u32 %pmore,%ktn,%K;\n");
        s += &format!("    @!%pmore bra LAST_{name};\n");
        s += "    mov.u32 %kcol,%ktn;\n";
        stage("%baseRow", "%A", "smemA", "%bufp", a_chunks, &mut s);
        stage("%baseCol", "%B", "smemB", "%bufp", b_chunks, &mut s);
        s += "    cp.async.commit_group;\n    cp.async.wait_group 1;\n";
        s += &format!("    bra SYNC_{name};\nLAST_{name}:\n    cp.async.wait_group 0;\nSYNC_{name}:\n");
        s += "    bar.sync 0;\n";

        // Load this warp's A fragments from smemA[bufc] (row-major BM×BK). For subtile ti: tile-row base
        // = warpRow*(16*tm) + ti*16; a0=[row grp], a1=[row grp+8], a2/a3 = +16 cols (the k32 pack).
        for ti in 0..tm {
            s += "    mov.u32 %ab,smemA;\n    add.u32 %ab,%ab,%bufc;\n";
            s += &format!("    mul.lo.s32 %tmp,%warpRow,{};\n    add.u32 %tmp,%tmp,{};\n", 16 * tm, ti * 16);
            s += "    add.u32 %tmp,%tmp,%grp;\n";
            s += &format!("    mul.lo.s32 %tmp,%tmp,{bk};\n    add.u32 %tmp,%tmp,%tg4;\n    add.u32 %ab,%ab,%tmp;\n");
            s += &format!("    ld.shared.b32 %a{ti}_0,[%ab];\n    ld.shared.b32 %a{ti}_2,[%ab+16];\n");
            s += &format!("    ld.shared.b32 %a{ti}_1,[%ab+{}];\n    ld.shared.b32 %a{ti}_3,[%ab+{}];\n", 8 * bk, 8 * bk + 16);
        }
        // Load this warp's B fragments from smemB[bufc] (row-major BN×BK). For subtile tj: tile-col base
        // (n index) = warpCol*(8*tn) + tj*8; b0=[col grp, k tg4], b1=[+16 k].
        for tj in 0..tn {
            s += "    mov.u32 %ab,smemB;\n    add.u32 %ab,%ab,%bufc;\n";
            s += &format!("    mul.lo.s32 %tmp,%warpCol,{};\n    add.u32 %tmp,%tmp,{};\n", 8 * tn, tj * 8);
            s += "    add.u32 %tmp,%tmp,%grp;\n";
            s += &format!("    mul.lo.s32 %tmp,%tmp,{bk};\n    add.u32 %tmp,%tmp,%tg4;\n    add.u32 %ab,%ab,%tmp;\n");
            s += &format!("    ld.shared.b32 %b{tj}_0,[%ab];\n    ld.shared.b32 %b{tj}_1,[%ab+16];\n");
        }
        // mma all subtiles (A frag reused across N, B frag reused across M).
        for ti in 0..tm {
            for tj in 0..tn {
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32\n        {{%d{ti}_{tj}_0,%d{ti}_{tj}_1,%d{ti}_{tj}_2,%d{ti}_{tj}_3}}, {{%a{ti}_0,%a{ti}_1,%a{ti}_2,%a{ti}_3}}, {{%b{tj}_0,%b{tj}_1}}, {{%d{ti}_{tj}_0,%d{ti}_{tj}_1,%d{ti}_{tj}_2,%d{ti}_{tj}_3}};\n");
            }
        }
        s += "    bar.sync 0;\n"; // all warps done reading bufc before a later step overwrites it
        s += &format!("    xor.b32 %bufc,%bufc,{tile_bytes};\n    xor.b32 %bufp,%bufp,{tile_bytes};\n");
        s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

        // Epilogue: store each subtile's 16×8 i32 result. global row = baseRow + warpRow*16tm + ti*16 +
        // grp (d0/d1) / +8 (d2/d3); global col = baseCol + warpCol*8tn + tj*8 + tg2 (d0/d2) / +1 (d1/d3).
        s += &format!("KEND_{name}:\n");
        for ti in 0..tm {
            for tj in 0..tn {
                s += &format!("    mul.lo.s32 %tmp,%warpRow,{};\n    add.u32 %tmp,%tmp,{};\n", 16 * tm, ti * 16);
                s += "    add.u32 %tmp,%tmp,%baseRow;\n    add.u32 %tmp,%tmp,%grp;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
                s += &format!("    mul.lo.s32 %tmp2,%warpCol,{};\n    add.u32 %tmp2,%tmp2,{};\n", 8 * tn, tj * 8);
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp2,%tmp2,%tg2;\n    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%C,%off;\n";
                s += &format!("    st.global.u32 [%cp],%d{ti}_{tj}_0;\n    st.global.u32 [%cp+4],%d{ti}_{tj}_1;\n");
                s += &format!("    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n");
                s += &format!("    st.global.u32 [%cp],%d{ti}_{tj}_2;\n    st.global.u32 [%cp+4],%d{ti}_{tj}_3;\n");
            }
        }
        s += "    ret;\n}\n";
        s
    }
}

/// Full **int8 (W8A8) tensor-core GEMM** `C = A·Bᵀ` (the quantized nn.Linear form): A is `[M,K]` **u8**
/// row-major (activations), B is `[N,K]` **i8** row-major (weights) — which *is* the `K×N` column-major
/// layout the `mma` `.col` operand wants, so `A·Bᵀ` maps straight onto `mma.row.col` with no transpose.
/// Each warp owns a `16×8` C tile and loops K in steps of 32 (the validated `m16n8k32` layout, offset
/// by the tile origin and the k-step). **i32** accumulate, exact mod 2³². M%16 == N%8 == K%32 == 0.
/// Returns the module (entry `int8_gemm_nt`). The structural twin of [`crate::ptx_fp8::fp8_gemm_ptx`].
pub fn int8_gemm_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        String::from(
            r#".version 8.4
.target sm_89
.address_size 64

.visible .entry int8_gemm_nt(
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
    .reg .b32 %d0,%d1,%d2,%d3,%z;
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

    mov.u32 %d0,0;
    mov.u32 %d1,0;
    mov.u32 %d2,0;
    mov.u32 %d3,0;
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
    mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32
        {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%d0,%d1,%d2,%d3};
    add.u32 %k,%k,32;
    bra KLOOP;
KEND:
    // C[row0+grp][col0+tg2] etc, row-major [M,N], i32
    add.s32 %tmp,%row0,%grp;
    mul.lo.s32 %tmp,%tmp,%N;
    add.s32 %tmp,%tmp,%col0;
    add.s32 %tmp,%tmp,%tg2;
    shl.b32 %tmp,%tmp,2;
    cvt.u64.u32 %t,%tmp;
    add.s64 %cp,%C,%t;
    st.global.u32 [%cp],%d0;
    st.global.u32 [%cp+4],%d1;
    // +8 rows = +8*N elements * 4 bytes
    mul.lo.s32 %tmp,%N,32;       // 8 rows * N * 4 bytes
    cvt.u64.u32 %t,%tmp;
    add.s64 %cp,%cp,%t;
    st.global.u32 [%cp],%d2;
    st.global.u32 [%cp+4],%d3;
    ret;
}
"#,
        )
    })
    .as_str()
}
