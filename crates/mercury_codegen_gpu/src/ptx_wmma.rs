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

/// fp16 tensor-core GEMM module: `wmma_nt_f16` (single 16×16 tile/warp, any 16-multiple dims) and
/// `wmma_nt_f16_mt` (2×4 tiles/warp = 32×64, fragment-reuse, the fast path for large GEMMs).
pub fn wmma_f16_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("wmma_nt_f16", "f16", 1, 1);
        m += &entry("wmma_nt_f16_mt", "f16", TM_TILES, TN_TILES);
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

/// Per-warp tile grid for the multi-tile (fast) kernels: 2 rows × 4 cols of 16×16 tiles = 32×64.
pub const TM_TILES: usize = 2;
pub const TN_TILES: usize = 4;
/// Per-warp output tile dims (the multi-tile kernel requires M%WARP_M==0 and N%WARP_N==0).
pub const WARP_M: usize = 16 * TM_TILES;
pub const WARP_N: usize = 16 * TN_TILES;
