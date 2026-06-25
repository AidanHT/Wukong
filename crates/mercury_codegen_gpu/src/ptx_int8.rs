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
            false,
        )
    })
    .as_str()
}

/// **SMEM-staged int8 GEMM with a fused per-channel dequant epilogue** (`int8_gemm_nt_smdb_deq`):
/// `out[i,j] = f32(Σ u8·i8) · scale[j]`, the per-output-channel symmetric-quant dequant. The `i32`
/// accumulators are converted to f32 and scaled by a per-column `scale[N]` **in registers, folded into
/// the C store** — so the f32 result lands in one pass with no extra HBM round-trip. This is the
/// epilogue cuBLAS int8 (which outputs raw `i32`) structurally **cannot fuse**: it needs a second
/// dequant kernel reading C back from HBM. Same 64×64 pipeline as [`int8_gemm_smdb_ptx`]; C is f32.
/// Requires M%64==0, N%64==0, K%INT8_BK==0.
pub fn int8_gemm_smdb_deq_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb(
            "int8_gemm_nt_smdb_deq",
            INT8_BM,
            INT8_BN,
            INT8_WARPS_M,
            INT8_WARPS_N,
            true,
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
            false,
        )
    })
    .as_str()
}

/// Generate an SMEM-staged + `cp.async` double-buffered int8 GEMM entry for a `bm×bn` CTA tile computed
/// by `warps_m×warps_n` warps. `bm`,`bn` must be multiples of `16*warps_m` / `8*warps_n`; the staging
/// assumes `INT8_BK==32` (16-byte chunks = half a K-slab row) and `threads <= bm*BK/16` so every chunk
/// has a thread. Each generated module owns its own `smemA`/`smemB` (no cross-module symbol clash).
fn gen_int8_smdb(name: &str, bm: usize, bn: usize, wm: usize, wn: usize, dequant: bool) -> String {
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

        // The fused-dequant variant takes an extra `scale[N]` (f32) per-output-channel scale param.
        let scale_param = if dequant { ",\n    .param .u64 pScale" } else { "" };
        let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
        s += &format!(".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{scale_param}\n)\n{{\n");
        s += &format!("    .shared .align 16 .b8 smemA[{}];\n", 2 * tile_bytes);
        s += &format!("    .shared .align 16 .b8 smemB[{}];\n", 2 * tile_bytes);
        s += "    .reg .pred %p0,%pmore;\n";
        s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%ktn,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufc,%bufp,%lane,%grp,%tg4,%tg2,%ab,%cc;\n";
        if dequant {
            s += "    .reg .b32 %col;\n    .reg .f32 %f0,%f1,%f2,%f3,%sc0,%sc1;\n    .reg .b64 %Scale,%scp;\n";
        }
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
        if dequant {
            s += "    ld.param.u64 %Scale,[pScale];\n    cvta.to.global.u64 %Scale,%Scale;\n";
        }
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
                s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp2,%tmp2,%tg2;\n";
                if dequant {
                    s += "    mov.u32 %col,%tmp2;\n"; // global output column of d0/d2 (d1/d3 = col+1)
                }
                s += "    add.u32 %tmp,%tmp,%tmp2;\n";
                s += "    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%C,%off;\n";
                if dequant {
                    // Fold the per-channel dequant into the store: out = f32(acc)·scale[col]. The i32→f32
                    // cvt and the f32 mul round identically to the CPU reference (acc as f32)·scale[col],
                    // so the result is exact-to-1-rounding (gated at a tight tolerance).
                    s += "    mul.wide.u32 %off,%col,4;\n    add.s64 %scp,%Scale,%off;\n";
                    s += "    ld.global.f32 %sc0,[%scp];\n    ld.global.f32 %sc1,[%scp+4];\n";
                    s += &format!("    cvt.rn.f32.s32 %f0,%d{ti}_{tj}_0;\n    mul.f32 %f0,%f0,%sc0;\n    st.global.f32 [%cp],%f0;\n");
                    s += &format!("    cvt.rn.f32.s32 %f1,%d{ti}_{tj}_1;\n    mul.f32 %f1,%f1,%sc1;\n    st.global.f32 [%cp+4],%f1;\n");
                    s += "    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n";
                    s += &format!("    cvt.rn.f32.s32 %f2,%d{ti}_{tj}_2;\n    mul.f32 %f2,%f2,%sc0;\n    st.global.f32 [%cp],%f2;\n");
                    s += &format!("    cvt.rn.f32.s32 %f3,%d{ti}_{tj}_3;\n    mul.f32 %f3,%f3,%sc1;\n    st.global.f32 [%cp+4],%f3;\n");
                } else {
                    s += &format!("    st.global.u32 [%cp],%d{ti}_{tj}_0;\n    st.global.u32 [%cp+4],%d{ti}_{tj}_1;\n");
                    s += &format!("    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n");
                    s += &format!("    st.global.u32 [%cp],%d{ti}_{tj}_2;\n    st.global.u32 [%cp+4],%d{ti}_{tj}_3;\n");
                }
            }
        }
        s += "    ret;\n}\n";
        s
    }
}

/// **`ldmatrix` + XOR-swizzle int8 GEMM** (`_swz`) — the conflict-free-SMEM analogue of the proven
/// fp16/bf16 `entry_mma_pipe` swz path ported to the 8-bit `m16n8k32` tile. The hand-placed
/// [`gen_int8_smdb`] issues 4+2 `ld.shared.b32` per warp-subtile per K-step from a row-major SMEM tile
/// that carries a **2-way bank conflict** (lanes `grp` and `grp+4` alias the same banks); this variant
/// instead stages each **64-byte SMEM row** (BK=64 u8 ⇒ `nc=4` 16-byte chunks, two `m16n8k32` K-steps
/// per slab) under the XOR swizzle `chunk ↦ chunk XOR ((row>>1)&3)` and gathers the A/B fragments with
/// one warp-cooperative `ldmatrix.x4`/`.x2` from the conflict-free layout (HW-optimized, far fewer SMEM
/// instructions per `mma`). At the byte level the 16×32-u8 A tile is the same 16-row × 32-byte shape as
/// the fp16 16×16 A tile, so the swizzle/ldmatrix math is **byte-for-byte the fp16 derivation** (every
/// `·bk` here equals fp16's `·bk·2` = 64 B/row); only the global `cp.async` stride (×1, u8), the
/// chunk-column shift (`<<4`), and the `mma`/accumulator types (`.s32.u8.s8.s32`, s32) differ. The
/// `ldmatrix` matrices map to the A operand as {m0:r0-7/k0-15, m1:r8-15/k0-15, m2:r0-7/k16-31,
/// m3:r8-15/k16-31} = {a0,a1,a2,a3} — exactly the fp16 register order. Bit-exact mod 2³² vs the CPU i32
/// reference (the swizzle only reorders SMEM; the integer arithmetic is untouched). Entry `name`;
/// requires M%bm==0, N%bn==0, K%64==0, bm%(16·wm)==0, bn%(8·wn)==0, (bm/wm)%8==(bn/wn)%8==0.
#[allow(clippy::too_many_arguments)]
fn gen_int8_smdb_swz(
    name: &str,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    dequant: bool,
    splitk: bool,
    static_dims: Option<(usize, usize, usize)>,
    raster: usize,
) -> String {
    // The shipped 2-stage double-buffer — every existing caller routes here, byte-identical to before the
    // `stages` generalization (the `_impl` `stages==2` branch contains the verbatim XOR-toggle path).
    gen_int8_smdb_swz_impl(name, bm, bn, wm, wn, dequant, splitk, static_dims, raster, 2)
}

/// `stages`-deep generalization of [`gen_int8_smdb_swz`]: `stages==2` is the original XOR double-buffer
/// (bit-identical PTX); `stages>=3` deepens it into a `stages`-buffer `cp.async` SMEM ring that prefetches
/// `stages-1` K-slabs ahead (the fp16/fp8-proven multistage lever, ported onto the swizzled-`ldmatrix`
/// 64×64-warp tile). Same 2-barrier overwrite discipline at any depth; the deeper ring only reorders when
/// each slab is staged, so it stays **bit-exact mod 2³²** vs the i32 oracle. `stages>=3` is supported only
/// on the plain dynamic path (no split-K / static-dims / raster — those each reuse `ctaid`/baked constants
/// the ring prologue does not thread). 3-stage 128×128 BK=64 = exactly 48 KiB static SMEM (the no-carveout max).
#[allow(clippy::too_many_arguments)]
fn gen_int8_smdb_swz_impl(
    name: &str,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    dequant: bool,
    splitk: bool,
    static_dims: Option<(usize, usize, usize)>,
    raster: usize,
    stages: usize,
) -> String {
    let bk = 64usize; // u8 K-slab: nc = bk/16 = 4 chunks/row (reuses the fp16 nc=4 swizzle phase), 2 k32 steps
    let threads = wm * wn * 32;
    let tm = bm / (16 * wm); // 16-row A subtiles per warp
    let tn = bn / (8 * wn); //  8-col B subtiles per warp
    let nks = bk / 32; // m16n8k32 K-steps per staged slab (2)
    let nc = bk / 16; // 16-byte chunks per SMEM row (4)
    let nc_mask = nc - 1;
    let wmr = bm / wm; // per-warp M rows
    let wnc = bn / wn; // per-warp N cols
    // Per-array tile bytes: A is bm·bk, B is bn·bk — DISTINCT when bm≠bn (the 256×128 / 128×256 big tiles).
    // smemA/smemB are sized 2·a_tile / 2·b_tile, and each array's XOR double-buffer toggle uses its OWN
    // tile size (a single bm·bk toggle would overflow smemB when bn<bm). For bm==bn this is the old kernel.
    let a_tile = bm * bk;
    let b_tile = bn * bk;
    let row_shift = (nc as u32).trailing_zeros(); // e>>row_shift = SMEM row (nc chunks per row)
    let col_mask = nc - 1;
    let wn_shift = (wn as u32).trailing_zeros();
    assert!(bk == 64, "{name}: swz swizzle phase is derived for BK=64 (nc=4)");
    assert!(bm % (16 * wm) == 0 && bn % (8 * wn) == 0, "{name}: bm/bn must tile by 16*wm / 8*wn");
    assert!(wmr % 8 == 0 && wnc % 8 == 0, "{name}: swz needs per-warp row/col bases = 0 (mod 8)");
    assert!(a_tile.is_power_of_two() && b_tile.is_power_of_two(), "{name}: A/B tile bytes must be powers of two (XOR double-buffer)");
    assert!(stages >= 2, "{name}: needs >=2 pipeline stages");
    // stages>=3 deepens the ring; it is only wired for the plain dynamic path (the prologue stages absolute
    // K columns 0,bk,…,(stages-2)·bk and the ring advance uses add+wrap, neither of which threads the
    // split-K K-range, the static baked dims, or the rasterized 1-D tile map).
    assert!(
        stages == 2 || (!splitk && static_dims.is_none() && raster == 0),
        "{name}: multistage (stages>=3) only supports the plain dynamic non-raster path"
    );
    assert!(stages * (bm * bk) + stages * (bn * bk) <= 48 * 1024, "{name}: static SMEM exceeds 48 KiB");
    assert!((bm * bk) % (threads * 16) == 0 && (bn * bk) % (threads * 16) == 0, "{name}: threads*16 must divide the tile bytes");
    // split-K folds each CTA's partial product into C by `red.global.add.u32` (deterministic for i32 —
    // integer add commutes, so the result is order-independent and bit-exact, unlike a float reduction).
    // The dequant epilogue can't combine with split-K (it would scale per-partial, not per-total).
    assert!(!(splitk && dequant), "{name}: split-K and the dequant epilogue are mutually exclusive");
    // Static-shape specialization is incompatible with split-K (which derives kslice from runtime
    // gridDim.z and the K param). The static dims must tile the kernel so the baked constants are exact.
    assert!(!(splitk && static_dims.is_some()), "{name}: split-K uses the runtime K param (dynamic dims)");
    if let Some((m, n, k)) = static_dims {
        assert!(m % bm == 0 && n % bn == 0 && k % bk == 0, "{name}: static dims must tile the kernel");
    }
    // Threadblock rasterization (the HBM-bound 4096³ L2 lever, ported from ptx_wmma's mma raster): a 1-D
    // grid is banded into `raster`-wide N-tile columns so co-resident CTAs touch a compact A/B footprint
    // that stays hot in L2. Needs bm,bn powers of two (tile counts via shift) and a 1-D launch (gridDim.x =
    // tiles_m·tiles_n). Orthogonal to the K-loop / fragment math, which derives only from baseRow/baseCol.
    assert!(!(raster > 0 && splitk), "{name}: raster and split-K both use ctaid.x — mutually exclusive");
    assert!(
        raster == 0 || (bm.is_power_of_two() && bn.is_power_of_two()),
        "{name}: raster needs bm,bn powers of two (tile counts via shift)"
    );
    let a_chunks = bm * bk / (threads * 16); // 16-byte cp.async chunks per thread
    let b_chunks = bn * bk / (threads * 16);

    let scale_param = if dequant { ",\n    .param .u64 pScale" } else { "" };
    let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
    s += &format!(".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{scale_param}\n)\n{{\n");
    s += &format!("    .shared .align 16 .b8 smemA[{}];\n", stages * bm * bk);
    s += &format!("    .shared .align 16 .b8 smemB[{}];\n", stages * bn * bk);
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%ktn,%kcol,%tmp,%tmp2,%tmp3,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%bufcA,%bufpA,%bufcB,%bufpB,%lane,%grp,%tg2,%warpMrow,%warpNcol,%aptr,%bptr,%phaseA,%phaseB,%arowb,%browb,%la16,%lb8,%swztmp;\n";
    if splitk {
        s += "    .reg .b32 %kbeg,%kend,%kslice;\n";
    }
    if raster > 0 {
        s += "    .reg .b32 %rlin,%rtn,%rtm,%rgsz,%rgrp,%rrem,%rcol0,%rgw,%rtrow,%rtcol;\n";
    }
    if dequant {
        s += "    .reg .b32 %col;\n    .reg .f32 %f0,%f1,%f2,%f3,%sc0,%sc1;\n    .reg .b64 %Scale,%scp;\n";
    }
    // accumulators d[ti][tj][0..3] (s32), A frags a[ti][0..3], B frags b[tj][0..1]
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

    // Static-shape specialization (Mercury's compile-time-shapes lever, mirroring `entry_w4a16`): bake
    // M/N/K as constants so ptxas constant-folds the hot-loop strides — every `mul.lo.s32 ...,%N` /
    // `...,%K` becomes a constant multiply (strength-reduced to a shift when the dim is a power of two)
    // and the K-loop trip count is known (unrollable). Dynamic loads them from params. The signature is
    // identical either way (M/N/K params remain, just unused), so the launcher is unchanged.
    match static_dims {
        Some((m, n, k)) => {
            s += &format!("    mov.u32 %M,{m};\n    mov.u32 %N,{n};\n    mov.u32 %K,{k};\n");
        }
        None => {
            s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
        }
    }
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";
    if dequant {
        s += "    ld.param.u64 %Scale,[pScale];\n    cvta.to.global.u64 %Scale,%Scale;\n";
    }
    if raster == 0 {
        s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
        s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    } else {
        // 1-D grid → column-banded tile order. tiles_n=N/bn, tiles_m=M/bm; band of `raster` N-columns,
        // sweep all M-rows before the next band (edge bands narrower than `raster` via runtime `rgw`).
        let (bn_sh, bm_sh) = (bn.trailing_zeros(), bm.trailing_zeros());
        s += &format!("    mov.u32 %rlin,%ctaid.x;\n    shr.u32 %rtn,%N,{bn_sh};\n    shr.u32 %rtm,%M,{bm_sh};\n");
        s += &format!("    mul.lo.s32 %rgsz,%rtm,{raster};\n    div.u32 %rgrp,%rlin,%rgsz;\n    rem.u32 %rrem,%rlin,%rgsz;\n");
        s += &format!("    mul.lo.s32 %rcol0,%rgrp,{raster};\n    sub.u32 %rgw,%rtn,%rcol0;\n    min.u32 %rgw,%rgw,{raster};\n");
        s += "    div.u32 %rtrow,%rrem,%rgw;\n    rem.u32 %rtcol,%rrem,%rgw;\n    add.u32 %rtcol,%rtcol,%rcol0;\n";
        s += &format!("    mul.lo.s32 %baseRow,%rtrow,{bm};\n    mul.lo.s32 %baseCol,%rtcol,{bn};\n");
    }
    if splitk {
        // K-split across gridDim.z CTAs: this CTA owns K-range [kbeg, kend). kslice = K/gridDim.z
        // (the host guarantees K % (gridDim.z · 64) == 0, so kslice is a 64-multiple = whole BK slabs).
        s += "    mov.u32 %tmp,%nctaid.z;\n    div.u32 %kslice,%K,%tmp;\n";
        s += "    mov.u32 %tmp,%ctaid.z;\n    mul.lo.s32 %kbeg,%tmp,%kslice;\n    add.u32 %kend,%kbeg,%kslice;\n";
    }
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n    and.b32 %lane,%tix,31;\n";
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n    and.b32 %warpCol,%warpId,{};\n", wn - 1);
    s += "    shr.u32 %grp,%lane,2;\n    and.b32 %tmp,%lane,3;\n    shl.b32 %tg2,%tmp,1;\n";
    s += &format!("    mul.lo.s32 %warpMrow,%warpRow,{wmr};\n    mul.lo.s32 %warpNcol,%warpCol,{wnc};\n");
    // swz per-lane phases / row byte-bases (every `·bk` = fp16 swz's `·bk·2` = 64 B/row). A: ldmatrix.x4
    // row R = warpMrow + mi·16 + (lane&15); arowb = (warpMrow + (lane&15))·bk; phaseA = ((lane&15)>>1)&3;
    // la16 = lane>>4 (the k16/k32 chunk-half selector). B: ldmatrix.x2, row = warpNcol + ni·8 + (lane&7).
    s += &format!("    and.b32 %tmp,%lane,15;\n    add.u32 %tmp2,%tmp,%warpMrow;\n    mul.lo.s32 %arowb,%tmp2,{bk};\n");
    s += &format!("    shr.u32 %tmp2,%tmp,1;\n    and.b32 %phaseA,%tmp2,{nc_mask};\n");
    s += "    shr.u32 %la16,%lane,4;\n";
    s += &format!("    and.b32 %tmp,%lane,7;\n    add.u32 %tmp2,%tmp,%warpNcol;\n    mul.lo.s32 %browb,%tmp2,{bk};\n");
    s += &format!("    shr.u32 %tmp2,%tmp,1;\n    and.b32 %phaseB,%tmp2,{nc_mask};\n");
    s += "    shr.u32 %tmp,%lane,3;\n    and.b32 %lb8,%tmp,1;\n";
    // zero accumulators
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.u32 %d{ti}_{tj}_{r},0;\n");
            }
        }
    }
    // Read pointer = oldest buffer (0); write pointer = newest slot ((stages-1)·tile). For stages==2 the
    // write slot is `tile` — byte-identical to the original double-buffer init.
    s += "    mov.u32 %bufcA,0;\n";
    s += &format!("    mov.u32 %bufpA,{};\n", (stages - 1) * a_tile);
    s += "    mov.u32 %bufcB,0;\n";
    s += &format!("    mov.u32 %bufpB,{};\n", (stages - 1) * b_tile);

    // cp.async staging into the **swizzled** SMEM tile (16-byte chunks). chunk e: r=e>>row_shift,
    // chunk=e&col_mask, byte col c=chunk·16; src is 16 contiguous u8 of global row (g_base+r) at kcol+c;
    // dst = smem+bufoff + r·bk + (chunk XOR ((r>>1)&nc_mask))·16 — the swizzle the ldmatrix read inverts.
    let stage = |g_base: &str, gptr_base: &str, smem: &str, bufoff: &str, chunks: usize, s: &mut String| {
        for li in 0..chunks {
            if li == 0 {
                *s += "    mov.u32 %e,%tix;\n";
            } else {
                *s += &format!("    add.u32 %e,%tix,{};\n", li * threads);
            }
            *s += &format!("    shr.u32 %r,%e,{row_shift};\n    and.b32 %c,%e,{col_mask};\n    shl.b32 %c,%c,4;\n");
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    cvt.u64.u32 %off,%tmp;\n    add.s64 %gptr,{gptr_base},%off;\n");
            // dst = swizzled SMEM byte: chunk = %c>>4; chunk_swz = chunk XOR ((r>>1)&nc_mask).
            *s += &format!("    shr.u32 %swztmp,%c,4;\n    shr.u32 %tmp2,%r,1;\n    and.b32 %tmp2,%tmp2,{nc_mask};\n    xor.b32 %swztmp,%swztmp,%tmp2;\n    shl.b32 %swztmp,%swztmp,4;\n");
            *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    mul.lo.s32 %tmp3,%r,{bk};\n    add.u32 %tmp,%tmp,%tmp3;\n    add.u32 %tmp,%tmp,%swztmp;\n");
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    // Prologue: prefetch this CTA's first slab (kbeg, or 0 without split-K) into buffer 0.
    let kstart = if splitk { "%kbeg" } else { "0" };
    let kstop = if splitk { "%kend" } else { "%K" };
    if stages == 2 {
        s += &format!("    mov.u32 %kcol,{kstart};\n");
        stage("%baseRow", "%A", "smemA", "%bufcA", a_chunks, &mut s);
        stage("%baseCol", "%B", "smemB", "%bufcB", b_chunks, &mut s);
        s += "    cp.async.commit_group;\n";
    } else {
        // Multistage prologue: prefetch slabs 0..stages-2 into buffers 0..stages-2 (stages-1 committed
        // groups). kstart is 0 here (multistage forbids split-K), so kcol = j·bk are absolute K columns.
        for j in 0..(stages - 1) {
            s += &format!("    mov.u32 %kcol,{};\n", j * bk);
            let (offa, offb) = (format!("{}", j * a_tile), format!("{}", j * b_tile));
            stage("%baseRow", "%A", "smemA", &offa, a_chunks, &mut s);
            stage("%baseCol", "%B", "smemB", &offb, b_chunks, &mut s);
            s += "    cp.async.commit_group;\n";
        }
    }

    s += &format!("    mov.u32 %kt,{kstart};\n");
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,{kstop};\n    @%p0 bra KEND_{name};\n");
    if stages == 2 {
        s += &format!("    add.u32 %ktn,%kt,{bk};\n    setp.lt.u32 %pmore,%ktn,{kstop};\n");
        s += &format!("    @!%pmore bra LAST_{name};\n");
        s += "    mov.u32 %kcol,%ktn;\n";
        stage("%baseRow", "%A", "smemA", "%bufpA", a_chunks, &mut s);
        stage("%baseCol", "%B", "smemB", "%bufpB", b_chunks, &mut s);
        s += "    cp.async.commit_group;\n    cp.async.wait_group 1;\n";
        s += &format!("    bra SYNC_{name};\nLAST_{name}:\n    cp.async.wait_group 0;\nSYNC_{name}:\n");
        s += "    bar.sync 0;\n";
    } else {
        // Prefetch slab (kt + (stages-1)·bk) into the write buffer (bufp), if it exists; then keep
        // stages-1 groups in flight so the oldest (bufc) is guaranteed arrived before the compute reads it.
        s += &format!("    add.u32 %ktn,%kt,{};\n    setp.lt.u32 %pmore,%ktn,{kstop};\n", (stages - 1) * bk);
        s += &format!("    @!%pmore bra NOSTAGE_{name};\n");
        s += "    mov.u32 %kcol,%ktn;\n";
        stage("%baseRow", "%A", "smemA", "%bufpA", a_chunks, &mut s);
        stage("%baseCol", "%B", "smemB", "%bufpB", b_chunks, &mut s);
        s += &format!("NOSTAGE_{name}:\n");
        s += &format!("    cp.async.commit_group;\n    cp.async.wait_group {};\n", stages - 1);
        s += "    bar.sync 0;\n";
    }

    // Compute: per k32 step, one warp-cooperative `ldmatrix.x4` (A) / `.x2` (B) per subtile from the
    // XOR-swizzled (conflict-free, no-pad) SMEM, then `tm·tn` `mma.sync.m16n8k32` (A frag reused across N,
    // B across M). chunk_off = ((ks·2 | la16/lb8) XOR phase)·16 selects the k16/k32 half (per-lane const).
    for ks in 0..nks {
        s += "    mov.u32 %aptr,smemA;\n    add.u32 %aptr,%aptr,%bufcA;\n    add.u32 %aptr,%aptr,%arowb;\n";
        s += &format!("    or.b32 %swztmp,%la16,{};\n    xor.b32 %swztmp,%swztmp,%phaseA;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
        for mi in 0..tm {
            let mibase = mi * 16 * bk;
            s += &format!("    add.u32 %tmp,%aptr,%swztmp;\n    add.u32 %tmp,%tmp,{mibase};\n    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%a{mi}_0,%a{mi}_1,%a{mi}_2,%a{mi}_3}},[%tmp];\n");
        }
        s += "    mov.u32 %bptr,smemB;\n    add.u32 %bptr,%bptr,%bufcB;\n    add.u32 %bptr,%bptr,%browb;\n";
        s += &format!("    or.b32 %swztmp,%lb8,{};\n    xor.b32 %swztmp,%swztmp,%phaseB;\n    shl.b32 %swztmp,%swztmp,4;\n", ks * 2);
        for ni in 0..tn {
            let nibase = ni * 8 * bk;
            s += &format!("    add.u32 %tmp,%bptr,%swztmp;\n    add.u32 %tmp,%tmp,{nibase};\n    ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%b{ni}_0,%b{ni}_1}},[%tmp];\n");
        }
        for ti in 0..tm {
            for tj in 0..tn {
                s += &format!("    mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32\n        {{%d{ti}_{tj}_0,%d{ti}_{tj}_1,%d{ti}_{tj}_2,%d{ti}_{tj}_3}}, {{%a{ti}_0,%a{ti}_1,%a{ti}_2,%a{ti}_3}}, {{%b{tj}_0,%b{tj}_1}}, {{%d{ti}_{tj}_0,%d{ti}_{tj}_1,%d{ti}_{tj}_2,%d{ti}_{tj}_3}};\n");
            }
        }
    }
    s += "    bar.sync 0;\n"; // all warps done reading bufcA/bufcB before a later step overwrites them
    if stages == 2 {
        s += &format!("    xor.b32 %bufcA,%bufcA,{a_tile};\n    xor.b32 %bufpA,%bufpA,{a_tile};\n");
        s += &format!("    xor.b32 %bufcB,%bufcB,{b_tile};\n    xor.b32 %bufpB,%bufpB,{b_tile};\n");
    } else {
        // Ring advance: bump each pointer one tile, wrapping at stages·tile (the XOR trick only cycles 2).
        let (a_ring, b_ring) = (stages * a_tile, stages * b_tile);
        s += &format!("    add.u32 %bufcA,%bufcA,{a_tile};\n    setp.ge.u32 %pmore,%bufcA,{a_ring};\n    @%pmore sub.u32 %bufcA,%bufcA,{a_ring};\n");
        s += &format!("    add.u32 %bufpA,%bufpA,{a_tile};\n    setp.ge.u32 %pmore,%bufpA,{a_ring};\n    @%pmore sub.u32 %bufpA,%bufpA,{a_ring};\n");
        s += &format!("    add.u32 %bufcB,%bufcB,{b_tile};\n    setp.ge.u32 %pmore,%bufcB,{b_ring};\n    @%pmore sub.u32 %bufcB,%bufcB,{b_ring};\n");
        s += &format!("    add.u32 %bufpB,%bufpB,{b_tile};\n    setp.ge.u32 %pmore,%bufpB,{b_ring};\n    @%pmore sub.u32 %bufpB,%bufpB,{b_ring};\n");
    }
    s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

    // Epilogue: store each subtile's 16×8 i32 result (D-fragment layout is fixed by the `mma`, identical
    // to the hand-placed kernel). global row = baseRow + warpRow·16tm + ti·16 + grp (d0/d1) / +8 (d2/d3);
    // global col = baseCol + warpCol·8tn + tj·8 + tg2 (d0/d2) / +1 (d1/d3).
    s += &format!("KEND_{name}:\n");
    for ti in 0..tm {
        for tj in 0..tn {
            s += &format!("    mul.lo.s32 %tmp,%warpRow,{};\n    add.u32 %tmp,%tmp,{};\n", 16 * tm, ti * 16);
            s += "    add.u32 %tmp,%tmp,%baseRow;\n    add.u32 %tmp,%tmp,%grp;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
            s += &format!("    mul.lo.s32 %tmp2,%warpCol,{};\n    add.u32 %tmp2,%tmp2,{};\n", 8 * tn, tj * 8);
            s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp2,%tmp2,%tg2;\n";
            if dequant {
                s += "    mov.u32 %col,%tmp2;\n";
            }
            s += "    add.u32 %tmp,%tmp,%tmp2;\n";
            s += "    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%C,%off;\n";
            if dequant {
                s += "    mul.wide.u32 %off,%col,4;\n    add.s64 %scp,%Scale,%off;\n";
                s += "    ld.global.f32 %sc0,[%scp];\n    ld.global.f32 %sc1,[%scp+4];\n";
                s += &format!("    cvt.rn.f32.s32 %f0,%d{ti}_{tj}_0;\n    mul.f32 %f0,%f0,%sc0;\n    st.global.f32 [%cp],%f0;\n");
                s += &format!("    cvt.rn.f32.s32 %f1,%d{ti}_{tj}_1;\n    mul.f32 %f1,%f1,%sc1;\n    st.global.f32 [%cp+4],%f1;\n");
                s += "    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n";
                s += &format!("    cvt.rn.f32.s32 %f2,%d{ti}_{tj}_2;\n    mul.f32 %f2,%f2,%sc0;\n    st.global.f32 [%cp],%f2;\n");
                s += &format!("    cvt.rn.f32.s32 %f3,%d{ti}_{tj}_3;\n    mul.f32 %f3,%f3,%sc1;\n    st.global.f32 [%cp+4],%f3;\n");
            } else {
                // split-K: accumulate each CTA's partial into C by deterministic integer atomic add
                // (i32 add commutes → order-independent, bit-exact); else a plain overwrite store.
                let st = if splitk { "red.global.add.u32" } else { "st.global.u32" };
                s += &format!("    {st} [%cp],%d{ti}_{tj}_0;\n    {st} [%cp+4],%d{ti}_{tj}_1;\n");
                s += "    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n";
                s += &format!("    {st} [%cp],%d{ti}_{tj}_2;\n    {st} [%cp+4],%d{ti}_{tj}_3;\n");
            }
        }
    }
    s += "    ret;\n}\n";
    s
}

/// **64×64 `ldmatrix`+swizzle int8 GEMM** (`int8_gemm_nt_smdb_swz`) — the conflict-free-SMEM candidate
/// for the int8→cuBLAS-IMMA gap. Same CTA tile / warp layout as [`int8_gemm_smdb_ptx`]; BK=64. Bit-exact.
pub fn int8_gemm_smdb_swz_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_smdb_swz", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N, false, false, None, 0)).as_str()
}

/// **64×64 `ldmatrix`+swizzle int8 GEMM with split-K** (`int8_gemm_nt_smdb_swz_sk`) — the thin-M / small-N
/// occupancy lever. Launched with `gridDim.z = sk` K-splits; each CTA computes a partial `C` over its
/// K-range and folds it in by `red.global.add.u32` (integer add commutes ⇒ the sum is order-independent
/// and **bit-exact / deterministic**, the property a float split-K reduction lacks). Fills the GPU when
/// the M,N grid alone leaves SMs idle (decode: tiny M, modest N). Requires C pre-zeroed and
/// K % (sk·64) == 0. Bit-exact vs the i32 oracle for any sk.
pub fn int8_gemm_smdb_swz_splitk_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_smdb_swz_sk", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N, false, true, None, 0)).as_str()
}

/// **64×64 `ldmatrix`+swizzle int8 GEMM with fused per-channel dequant** (`int8_gemm_nt_smdb_swz_deq`) —
/// the swizzle path carrying the cuBLAS-can't-fuse `f32(Σ u8·i8)·scale[j]` epilogue (see
/// [`int8_gemm_smdb_deq_ptx`]). Same dequant store, gated at the f32-scale tolerance.
pub fn int8_gemm_smdb_swz_deq_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_smdb_swz_deq", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N, true, false, None, 0)).as_str()
}

/// **128×128 `ldmatrix`+swizzle int8 GEMM** (`int8_gemm_nt_smdb128_swz`) — the large-tile swizzle
/// candidate (8 warps, BK=64). Same CTA tile as [`int8_gemm_smdb128_ptx`]. Bit-exact.
pub fn int8_gemm_smdb128_swz_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_smdb128_swz", INT8_BM128, INT8_BN128, INT8_WARPS_M128, INT8_WARPS_N128, false, false, None, 0)).as_str()
}

/// **Static-shape `ldmatrix`+swizzle int8 GEMM** — the M1 compile-time-shapes lever. Bakes `M/N/K` into
/// the kernel so ptxas constant-folds/strength-reduces the hot-loop strides and knows the K trip count
/// (the same win `w4a16_static_ptx` lands for int4). Builds the 64×64 (`int8_gemm_nt_smdb_swz_static`) or
/// 128×128 (`int8_gemm_nt_smdb128_swz_static`) entry per `use_128`; returns an owned per-shape module
/// (the caller caches it under a shape-keyed key). Same codegen as the dynamic swz kernel ⇒ **bit-exact**.
pub fn int8_gemm_smdb_swz_static_ptx(m: usize, n: usize, k: usize, use_128: bool) -> String {
    let (name, bm, bn, wm, wn) = if use_128 {
        ("int8_gemm_nt_smdb128_swz_static", INT8_BM128, INT8_BN128, INT8_WARPS_M128, INT8_WARPS_N128)
    } else {
        ("int8_gemm_nt_smdb_swz_static", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N)
    };
    gen_int8_smdb_swz(name, bm, bn, wm, wn, false, false, Some((m, n, k)), 0)
}

/// Entry name for [`int8_gemm_smdb_swz_static_ptx`] at the matching `use_128`.
pub fn int8_gemm_smdb_swz_static_entry(use_128: bool) -> &'static str {
    if use_128 { "int8_gemm_nt_smdb128_swz_static" } else { "int8_gemm_nt_smdb_swz_static" }
}

/// **Threadblock-rasterized `ldmatrix`+swizzle int8 GEMM** — the HBM-bound (4096³) L2-locality lever. The
/// dispatched 4096³ kernel ([`int8_gemm_smdb128_swz_ptx`]) tops out at ~62% of cuBLAS because at that size
/// A+B (32 MiB) overflow L2 and the GEMM is HBM-bandwidth-bound; the naive `ctaid.x/y → tile` map streams
/// a scattered A/B footprint through L2. **Rasterization** bands the 1-D CTA grid into `raster`-wide
/// N-tile columns and sweeps all M-rows within a band before the next, so the SMs' co-resident CTAs reuse
/// a compact `raster·bn`-wide slab of B (and the band's A rows) from L2 instead of HBM — the same lever
/// that took the fp16 `mma` path from ~56% to ~72% at 4096³. **Launch with a 1-D grid**
/// `gridDim.x = (M/bm)·(N/bn)`. `raster` is a power of two (8/16 the usual optima). Bit-exact (rasterizing
/// only permutes which CTA computes which output tile; the per-tile u8×i8→i32 arithmetic is untouched).
pub fn int8_gemm_smdb_swz_raster_ptx(use_128: bool, raster: usize) -> String {
    let (name, bm, bn, wm, wn) = if use_128 {
        ("int8_gemm_nt_smdb128_swz_r", INT8_BM128, INT8_BN128, INT8_WARPS_M128, INT8_WARPS_N128)
    } else {
        ("int8_gemm_nt_smdb_swz_r", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N)
    };
    gen_int8_smdb_swz(name, bm, bn, wm, wn, false, false, None, raster)
}

/// Entry name for [`int8_gemm_smdb_swz_raster_ptx`] at the matching `use_128` (raster-width-independent —
/// the width is baked into the PTX body, so the caller keys the module cache by `(use_128, raster)`).
pub fn int8_gemm_smdb_swz_raster_entry(use_128: bool) -> &'static str {
    if use_128 { "int8_gemm_nt_smdb128_swz_r" } else { "int8_gemm_nt_smdb_swz_r" }
}

/// **Big-tile `ldmatrix`+swizzle int8 GEMM** — the #1 int8→cuBLAS lever (bigger CTA + warp tiles). The
/// shipped swz kernels use 64×64 (a 32×32 per-warp tile) / 128×128 (32×64); CUTLASS's *winning* int8
/// configs on Ada are **256×128 / 128×256 with a 64×64 warp tile** (8 warps, wm=4 wn=2 or wm=2 wn=4) — 2×
/// the per-warp A/B reuse, exactly where the remaining ~2× lives (an Ada study put bigger CTA/warp tiles
/// at 89.5%→100% of cuBLAS, vs ~+3% for multistage depth). The generator is already tile-generic, so this
/// just exposes arbitrary `(bm,bn,wm,wn)` + optional rasterization; the entry name encodes the tile so the
/// module-cache key is unique. Returns `(entry_name, ptx)` (an owned per-tile module). 2-stage BK=64;
/// 256×128 / 128×256 = **exactly 48 KiB static SMEM** (a deeper pipeline needs the dynamic-SMEM path).
/// **Bit-exact** (same per-tile u8×i8→i32 arithmetic; tile/warp shape only changes work assignment).
/// Launch a **2-D grid** `(N/bn, M/bm, 1)` when `raster==0`, else a **1-D grid** `((M/bm)·(N/bn), 1, 1)`.
pub fn int8_gemm_swz_tile_ptx(bm: usize, bn: usize, wm: usize, wn: usize, raster: usize) -> (String, String) {
    let name = if raster > 0 {
        format!("int8_swz_{bm}x{bn}_w{wm}x{wn}_r{raster}")
    } else {
        format!("int8_swz_{bm}x{bn}_w{wm}x{wn}")
    };
    let ptx = gen_int8_smdb_swz(&name, bm, bn, wm, wn, false, false, None, raster);
    (name, ptx)
}

/// **The winning int8 swz config (perf/gpu-quant-2): a 128×128 CTA with a 64×64 warp tile** — 4 warps
/// (`wm=wn=2`) instead of the shipped 8-warp 32×64. Doubling the per-warp tile (`tm=4` 16-row × `tn=8`
/// 8-col = 32 subtiles, a 64×64 warp tile) doubles the A/B fragment reuse per `mma.sync` — the lever the
/// Ada int8 study put at 89.5%→100% of cuBLAS. Measured same-run vs the shipped 8-warp 128×128 (`_smdb128_swz`):
/// **~1.2–1.3× faster** (1024³ 69%→84%, 2048³ 79%→92%, 4096³ 75%→84% of cuBLAS) — bigger CTA tiles
/// (256×128) instead *lose* on the 20-SM 4050 (1 CTA/SM starves latency hiding). 4 warps × 128 accumulator
/// regs keeps ~3 CTAs/SM. Bit-exact (only the warp work split changes). M%128==N%128==0, K%64==0.
pub const INT8_W64_BM: usize = 128;
pub const INT8_W64_BN: usize = 128;
pub const INT8_W64_WARPS_M: usize = 2;
pub const INT8_W64_WARPS_N: usize = 2;

/// 64×64-warp-tile int8 swz GEMM (entry `int8_gemm_nt_w64_swz`) — the new default workhorse. 2-D grid.
pub fn int8_gemm_w64_swz_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_w64_swz", INT8_W64_BM, INT8_W64_BN, INT8_W64_WARPS_M, INT8_W64_WARPS_N, false, false, None, 0)).as_str()
}

/// **3-stage** 64×64-warp-tile int8 swz GEMM (entry `int8_gemm_nt_w64_swz_s3`) — deepens
/// [`int8_gemm_w64_swz_ptx`]'s 2-buffer `cp.async` into a 3-buffer SMEM ring (prefetch 2 K-slabs ahead).
/// At 128×128 BK=64 the ring is 3·(8 KiB+8 KiB) = **exactly 48 KiB** static SMEM — the no-carveout max.
///
/// **MEASURED NEGATIVE for int8 — NOT shipped** (`quant_int8_w64_s3_sweep`, ≥3 same-run passes). The
/// fp8 warp-tile sweep found this same 3-stage pipeline *won* (+10/+19 pts at 1024³/2048³), so it was
/// the obvious int8 lever to try — but for int8 it **loses at every size** (1024³ ~67%, 2048³ ~69–80%,
/// 4096³ ~65% of cuBLAS, vs the 2-stage's ~84/100/72%). int8 runs `mma` at 2× the fp16/fp8 rate, so the
/// 2-stage BK=64 already hides the `cp.async` latency; deepening to 48 KiB SMEM only *cuts occupancy*
/// (fewer CTAs/SM) with no compute-bound payoff — confirming the earlier hand-placed "+3%" / Ada-study
/// read that multistage depth is not the int8 lever (the warp tile was). Retained as the bit-exact
/// validation of the `stages`-general ring (`quant_int8_w64_s3_matches_reference`) and the reproducible
/// record of the dead-end; the dispatch stays on the 2-stage [`int8_gemm_w64_swz_ptx`].
/// **Bit-exact** mod 2³² (a deeper prefetch ring only reorders staging; the i32 mma arithmetic is identical).
pub fn int8_gemm_w64_swz_s3_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz_impl("int8_gemm_nt_w64_swz_s3", INT8_W64_BM, INT8_W64_BN, INT8_W64_WARPS_M, INT8_W64_WARPS_N, false, false, None, 0, 3)).as_str()
}

/// 64×64-warp-tile int8 swz GEMM **with threadblock rasterization** (`raster=8`, entry
/// `int8_gemm_nt_w64_swz_r8`) — the 2048²-regime near-parity config (w2×2 91.7%→**99.6%** of cuBLAS with
/// raster=8). raster is L2-edge-specific (neutral at 1024³, slightly negative at 4096³), so it is
/// dispatched only in the L2-transition regime. **Launch a 1-D grid** `gridDim.x = (M/128)·(N/128)`.
pub fn int8_gemm_w64_swz_r8_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_w64_swz_r8", INT8_W64_BM, INT8_W64_BN, INT8_W64_WARPS_M, INT8_W64_WARPS_N, false, false, None, 8)).as_str()
}

/// 64×64-warp-tile int8 swz GEMM **with the fused per-channel dequant epilogue** (entry
/// `int8_gemm_nt_w64_swz_deq`) — the fast `out = f32(Σ u8·i8)·scale[j]` Linear/inference output stage on
/// the winning warp tile (the cuBLAS-can't-fuse lever, now riding the fastest int8 base). 2-D grid.
pub fn int8_gemm_w64_swz_deq_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| gen_int8_smdb_swz("int8_gemm_nt_w64_swz_deq", INT8_W64_BM, INT8_W64_BN, INT8_W64_WARPS_M, INT8_W64_WARPS_N, true, false, None, 0)).as_str()
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

/// **Multi-stage `cp.async` int8 GEMM** (`stages`-deep, ≥2) — deepens [`int8_gemm_smdb_ptx`]'s 2-buffer
/// double-buffer into a `stages`-buffer SMEM ring that prefetches **`stages-1` K-slabs ahead**, so the
/// `mma.sync` units never stall on the global→shared `cp.async` latency at large K (the canonical
/// large-GEMM lever the 2-buffer scheme leaves on the table — the same multi-stage pipeline that lifted
/// the fp16 cliff). It keeps the **2-barrier** structure of [`gen_int8_smdb`] verbatim (barrier 1 makes
/// the just-arrived slab visible; barrier 2 fences the read of the current buffer before a future
/// prefetch reuses it), so the overwrite ordering is provably safe at *any* depth — only the ring depth
/// and the `cp.async.wait_group` keep-`stages-1`-in-flight count change. Same hand-placed
/// `mma.sync.m16n8k32.s32.u8.s8.s32` fragments, `BK=32` slab, and bit-exact mod-2³² contract.
fn gen_int8_smdb_ms(
    name: &str,
    bm: usize,
    bn: usize,
    wm: usize,
    wn: usize,
    stages: usize,
    dequant: bool,
) -> String {
    assert!(stages >= 2, "multi-stage needs >=2 buffers");
    let bk = INT8_BK;
    let threads = wm * wn * 32;
    let tm = bm / (16 * wm);
    let tn = bn / (8 * wn);
    let tile_bytes = bm * bk;
    let ring = stages * tile_bytes;
    assert!(
        threads * 16 <= bm * bk && (bm * bk) % (threads * 16) == 0,
        "smdb_ms staging needs threads*16 to divide the tile bytes"
    );
    let a_chunks = bm * bk / (threads * 16);
    let b_chunks = bn * bk / (threads * 16);
    let wn_shift = wn.trailing_zeros();

    let scale_param = if dequant { ",\n    .param .u64 pScale" } else { "" };
    let mut s = String::from(".version 8.4\n.target sm_89\n.address_size 64\n\n");
    s += &format!(".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC{scale_param}\n)\n{{\n");
    s += &format!("    .shared .align 16 .b8 smemA[{ring}];\n");
    s += &format!("    .shared .align 16 .b8 smemB[{ring}];\n");
    s += "    .reg .pred %p0,%pmore;\n";
    s += "    .reg .b32 %M,%N,%K,%baseRow,%baseCol,%kt,%ktn,%kcol,%tmp,%tmp2,%tix,%warpId,%warpRow,%warpCol,%e,%r,%c,%roff,%woff,%lane,%grp,%tg4,%tg2,%ab;\n";
    if dequant {
        s += "    .reg .b32 %col;\n    .reg .f32 %f0,%f1,%f2,%f3,%sc0,%sc1;\n    .reg .b64 %Scale,%scp;\n";
    }
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
    if dequant {
        s += "    ld.param.u64 %Scale,[pScale];\n    cvta.to.global.u64 %Scale,%Scale;\n";
    }
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %baseRow,%tmp,{bm};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %baseCol,%tmp,{bn};\n");
    s += "    mov.u32 %tix,%tid.x;\n    shr.u32 %warpId,%tix,5;\n";
    s += &format!("    shr.u32 %warpRow,%warpId,{wn_shift};\n");
    s += &format!("    and.b32 %warpCol,%warpId,{};\n", wn - 1);
    s += "    and.b32 %lane,%tix,31;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg4,%lane,3;\n";
    s += "    shl.b32 %tg2,%tg4,1;\n    shl.b32 %tg4,%tg4,2;\n";
    for ti in 0..tm {
        for tj in 0..tn {
            for r in 0..4 {
                s += &format!("    mov.u32 %d{ti}_{tj}_{r},0;\n");
            }
        }
    }

    // Stage the `%kcol` A/B slab into the SMEM buffer at byte offset `bufoff` via cp.async (16-byte
    // chunks). Identical chunk math to `gen_int8_smdb` (BK=32 => r=e>>1, c=(e&1)*16).
    let stage = |g_base: &str,
                 gptr_base: &str,
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
            *s += "    shr.u32 %r,%e,1;\n    and.b32 %c,%e,1;\n    shl.b32 %c,%c,4;\n";
            *s += &format!("    add.u32 %tmp,{g_base},%r;\n    mul.lo.s32 %tmp,%tmp,%K;\n    add.u32 %tmp,%tmp,%kcol;\n    add.u32 %tmp,%tmp,%c;\n");
            *s += &format!("    cvt.u64.u32 %off,%tmp;\n    add.s64 %gptr,{gptr_base},%off;\n");
            *s += &format!("    mov.u32 %tmp,{smem};\n    add.u32 %tmp,%tmp,{bufoff};\n    shl.b32 %tmp2,%e,4;\n    add.u32 %tmp,%tmp,%tmp2;\n");
            *s += "    cp.async.cg.shared.global [%tmp],[%gptr],16;\n";
        }
    };

    // Prologue: prefetch slabs 0..stages-2 into buffers 0..stages-2 (stages-1 committed groups).
    for j in 0..(stages - 1) {
        s += &format!("    mov.u32 %kcol,{};\n", j * bk);
        let off = format!("{}", j * tile_bytes);
        stage("%baseRow", "%A", "smemA", &off, a_chunks, &mut s);
        stage("%baseCol", "%B", "smemB", &off, b_chunks, &mut s);
        s += "    cp.async.commit_group;\n";
    }
    s += "    mov.u32 %roff,0;\n";
    s += &format!("    mov.u32 %woff,{};\n", (stages - 1) * tile_bytes);

    s += "    mov.u32 %kt,0;\n";
    s += &format!("KLOOP_{name}:\n    setp.ge.u32 %p0,%kt,%K;\n    @%p0 bra KEND_{name};\n");
    // Prefetch slab (kt/bk + stages-1) into the alternate buffer (woff), if it exists.
    s += &format!("    add.u32 %ktn,%kt,{};\n    setp.lt.u32 %pmore,%ktn,%K;\n", (stages - 1) * bk);
    s += &format!("    @!%pmore bra NOSTAGE_{name};\n");
    s += "    mov.u32 %kcol,%ktn;\n";
    stage("%baseRow", "%A", "smemA", "%woff", a_chunks, &mut s);
    stage("%baseCol", "%B", "smemB", "%woff", b_chunks, &mut s);
    s += &format!("NOSTAGE_{name}:\n");
    // Keep stages-1 groups in flight; wait until the current slab (roff) is the oldest-completed.
    s += &format!("    cp.async.commit_group;\n    cp.async.wait_group {};\n", stages - 1);
    s += "    bar.sync 0;\n";

    // Load this warp's A/B fragments from the current buffer (smem[roff]); same layout as gen_int8_smdb.
    for ti in 0..tm {
        s += "    mov.u32 %ab,smemA;\n    add.u32 %ab,%ab,%roff;\n";
        s += &format!("    mul.lo.s32 %tmp,%warpRow,{};\n    add.u32 %tmp,%tmp,{};\n", 16 * tm, ti * 16);
        s += "    add.u32 %tmp,%tmp,%grp;\n";
        s += &format!("    mul.lo.s32 %tmp,%tmp,{bk};\n    add.u32 %tmp,%tmp,%tg4;\n    add.u32 %ab,%ab,%tmp;\n");
        s += &format!("    ld.shared.b32 %a{ti}_0,[%ab];\n    ld.shared.b32 %a{ti}_2,[%ab+16];\n");
        s += &format!("    ld.shared.b32 %a{ti}_1,[%ab+{}];\n    ld.shared.b32 %a{ti}_3,[%ab+{}];\n", 8 * bk, 8 * bk + 16);
    }
    for tj in 0..tn {
        s += "    mov.u32 %ab,smemB;\n    add.u32 %ab,%ab,%roff;\n";
        s += &format!("    mul.lo.s32 %tmp,%warpCol,{};\n    add.u32 %tmp,%tmp,{};\n", 8 * tn, tj * 8);
        s += "    add.u32 %tmp,%tmp,%grp;\n";
        s += &format!("    mul.lo.s32 %tmp,%tmp,{bk};\n    add.u32 %tmp,%tmp,%tg4;\n    add.u32 %ab,%ab,%tmp;\n");
        s += &format!("    ld.shared.b32 %b{tj}_0,[%ab];\n    ld.shared.b32 %b{tj}_1,[%ab+16];\n");
    }
    for ti in 0..tm {
        for tj in 0..tn {
            s += &format!("    mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32\n        {{%d{ti}_{tj}_0,%d{ti}_{tj}_1,%d{ti}_{tj}_2,%d{ti}_{tj}_3}}, {{%a{ti}_0,%a{ti}_1,%a{ti}_2,%a{ti}_3}}, {{%b{tj}_0,%b{tj}_1}}, {{%d{ti}_{tj}_0,%d{ti}_{tj}_1,%d{ti}_{tj}_2,%d{ti}_{tj}_3}};\n");
        }
    }
    s += "    bar.sync 0;\n"; // fence reads of roff before a future prefetch reuses this buffer
    s += &format!("    add.u32 %roff,%roff,{tile_bytes};\n    setp.ge.u32 %pmore,%roff,{ring};\n    @%pmore sub.u32 %roff,%roff,{ring};\n");
    s += &format!("    add.u32 %woff,%woff,{tile_bytes};\n    setp.ge.u32 %pmore,%woff,{ring};\n    @%pmore sub.u32 %woff,%woff,{ring};\n");
    s += &format!("    add.u32 %kt,%kt,{bk};\n    bra KLOOP_{name};\n");

    s += &format!("KEND_{name}:\n");
    for ti in 0..tm {
        for tj in 0..tn {
            s += &format!("    mul.lo.s32 %tmp,%warpRow,{};\n    add.u32 %tmp,%tmp,{};\n", 16 * tm, ti * 16);
            s += "    add.u32 %tmp,%tmp,%baseRow;\n    add.u32 %tmp,%tmp,%grp;\n    mul.lo.s32 %tmp,%tmp,%N;\n";
            s += &format!("    mul.lo.s32 %tmp2,%warpCol,{};\n    add.u32 %tmp2,%tmp2,{};\n", 8 * tn, tj * 8);
            s += "    add.u32 %tmp2,%tmp2,%baseCol;\n    add.u32 %tmp2,%tmp2,%tg2;\n";
            if dequant {
                s += "    mov.u32 %col,%tmp2;\n";
            }
            s += "    add.u32 %tmp,%tmp,%tmp2;\n";
            s += "    shl.b32 %tmp,%tmp,2;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%C,%off;\n";
            if dequant {
                s += "    mul.wide.u32 %off,%col,4;\n    add.s64 %scp,%Scale,%off;\n";
                s += "    ld.global.f32 %sc0,[%scp];\n    ld.global.f32 %sc1,[%scp+4];\n";
                s += &format!("    cvt.rn.f32.s32 %f0,%d{ti}_{tj}_0;\n    mul.f32 %f0,%f0,%sc0;\n    st.global.f32 [%cp],%f0;\n");
                s += &format!("    cvt.rn.f32.s32 %f1,%d{ti}_{tj}_1;\n    mul.f32 %f1,%f1,%sc1;\n    st.global.f32 [%cp+4],%f1;\n");
                s += "    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n";
                s += &format!("    cvt.rn.f32.s32 %f2,%d{ti}_{tj}_2;\n    mul.f32 %f2,%f2,%sc0;\n    st.global.f32 [%cp],%f2;\n");
                s += &format!("    cvt.rn.f32.s32 %f3,%d{ti}_{tj}_3;\n    mul.f32 %f3,%f3,%sc1;\n    st.global.f32 [%cp+4],%f3;\n");
            } else {
                s += &format!("    st.global.u32 [%cp],%d{ti}_{tj}_0;\n    st.global.u32 [%cp+4],%d{ti}_{tj}_1;\n");
                s += &format!("    mul.lo.s32 %tmp,%N,32;\n    cvt.u64.u32 %off,%tmp;\n    add.s64 %cp,%cp,%off;\n");
                s += &format!("    st.global.u32 [%cp],%d{ti}_{tj}_2;\n    st.global.u32 [%cp+4],%d{ti}_{tj}_3;\n");
            }
        }
    }
    s += "    ret;\n}\n";
    s
}

/// **3-stage** `cp.async` int8 GEMM, 64×64 tile (entry `int8_gemm_nt_smdb_s3`) — see [`gen_int8_smdb_ms`].
pub fn int8_gemm_smdb_s3_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb_ms("int8_gemm_nt_smdb_s3", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N, 3, false)
    })
    .as_str()
}

/// **4-stage** `cp.async` int8 GEMM, 64×64 tile (entry `int8_gemm_nt_smdb_s4`) — see [`gen_int8_smdb_ms`].
pub fn int8_gemm_smdb_s4_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb_ms("int8_gemm_nt_smdb_s4", INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N, 4, false)
    })
    .as_str()
}

/// **3-stage** `cp.async` int8 GEMM, 128×128 tile (entry `int8_gemm_nt_smdb128_s3`) — see [`gen_int8_smdb_ms`].
pub fn int8_gemm_smdb128_s3_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb_ms("int8_gemm_nt_smdb128_s3", INT8_BM128, INT8_BN128, INT8_WARPS_M128, INT8_WARPS_N128, 3, false)
    })
    .as_str()
}

/// **4-stage** `cp.async` int8 GEMM, 128×128 tile (entry `int8_gemm_nt_smdb128_s4`) — see [`gen_int8_smdb_ms`].
pub fn int8_gemm_smdb128_s4_ptx() -> &'static str {
    static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        gen_int8_smdb_ms("int8_gemm_nt_smdb128_s4", INT8_BM128, INT8_BN128, INT8_WARPS_M128, INT8_WARPS_N128, 4, false)
    })
    .as_str()
}
