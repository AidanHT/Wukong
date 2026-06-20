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

/// Generate a **`cp.async` double-buffered** SMEM-staged WMMA GEMM (`_sm_db`). Same CTA tiling as
/// [`entry_smem`], but the K-loop is software-pipelined: each step issues `cp.async` copies that
/// prefetch the *next* A/B tile into the alternate shared buffer **while the tensor cores consume the
/// current one**, then `cp.async.wait_group 1` only blocks on the older (current) copy. This overlaps
/// global-load latency with compute — the lever for the large-GEMM cliff, which the 128×128 experiment
/// showed is latency- not bandwidth-*volume*-bound (cuBLAS hides the same latency with a multi-stage
/// pipeline). Two shared buffers toggle by XOR (the tile size is a power of two). Requires `bm==bn`
/// (one buffer-offset register drives both A and B) and the [`entry_smem`] staging constraints.
fn entry_smem_db(name: &str, ty: &str, bm: usize, bn: usize, warps_m: usize, warps_n: usize) -> String {
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
    let a_chunks = bm * SM_BK / (threads * 8);
    let b_chunks = bn * SM_BK / (threads * 8);
    let wn_shift = warps_n.trailing_zeros();
    let wm = (16 * tm) as i64;
    let wn = (16 * tn) as i64;

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{{\n"
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
    s += "    mov.u32 %ldm,16;\n";
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
        m += &entry_smem_db("wmma_nt_f16_sm_db", "f16", SM_BM, SM_BN, SM_WARPS_M, SM_WARPS_N);
        m += &entry_smem_db(
            "wmma_nt_f16_sm128_db",
            "f16",
            SM128_BM,
            SM128_BN,
            SM128_WARPS_M,
            SM128_WARPS_N,
        );
        m
    })
    .as_str()
}

/// bf16 tensor-core GEMM module: `wmma_nt_bf16` + `wmma_nt_bf16_mt`.
pub fn wmma_bf16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("wmma_nt_bf16", "bf16", 1, 1);
        m += &entry("wmma_nt_bf16_mt", "bf16", TM_TILES, TN_TILES);
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
