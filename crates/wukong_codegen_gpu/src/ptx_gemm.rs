//! A **PTX generator** for the register-blocked f32 GEMM — Wukong emitting PTX from Rust, which is
//! the natural way to write a register-tiled kernel (lots of unrolled loads/FMAs). This is the GPU
//! analogue of how the AVX2 microkernel register-blocks the C tile.
//!
//! Shape: each 16×16 thread block computes a 64×64 C tile; each thread owns a 4×4 micro-tile (16
//! accumulators in registers). A `64×16` A-panel and a `16×64` B-panel are staged in shared memory
//! per K-step (BK=16); the inner product reads them from shared into registers and issues 16 FMAs
//! per K. Out-of-range threads load zeros and skip the C store, so ragged M/N/K work. Two entries:
//! `gemm_nt_rb` (`C = A·Bᵀ`, nn.Linear) and `gemm_nn_rb` (`C = A·B`).

use std::sync::OnceLock;

const BM: u32 = 64; // C tile rows per block
const BN: u32 = 64; // C tile cols per block
const BK: u32 = 16; // K step
const TM: u32 = 4; // rows per thread
const TN: u32 = 4; // cols per thread
const THREADS: u32 = (BM / TM) * (BN / TN); // 256

/// `0f` + the f32 hex of `0.0` (kept local so this file is self-contained).
const F0: &str = "0f00000000";

/// Generate one register-blocked GEMM entry. `transposed` selects the B index expression:
/// `A·Bᵀ` (B is N×K, index `col*K + k`) vs `A·B` (B is K×N, index `k*N + col`). `tag` is appended
/// to every label so the two entries in one module don't collide (PTX labels are module-scoped).
fn entry(name: &str, transposed: bool, tag: &str) -> String {
    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pM,\n    .param .u32 pN,\n    .param .u32 pK,\n    .param .u64 pA,\n    .param .u64 pB,\n    .param .u64 pC\n)\n{{\n"
    );
    // Registers.
    s += "    .reg .pred %p0, %p1, %p2;\n";
    s += "    .reg .f32 %acc<16>;\n";
    s += "    .reg .f32 %a0,%a1,%a2,%a3,%b0,%b1,%b2,%b3,%v;\n";
    // NB: do not name a register `%tid`/`%ntid`/`%ctaid`/`%nctaid` — those are PTX special regs;
    // the linear thread id is `%lin`.
    s += "    .reg .b32 %M,%N,%K,%tx,%ty,%lin,%bm,%bn,%kt,%ntiles,%rit,%rit16,%cit,%e,%ii,%kk,%jj,%kc,%row,%col,%idx,%tmp;\n";
    s += "    .reg .b64 %A,%B,%C,%off,%addr,%aa,%bb,%sA,%sB;\n";
    s += &format!("    .shared .align 4 .b8 As[{}];\n", BM * BK * 4);
    s += &format!("    .shared .align 4 .b8 Bs[{}];\n", BK * BN * 4);

    // Params + global pointers.
    s += "    ld.param.u32 %M,[pM];\n    ld.param.u32 %N,[pN];\n    ld.param.u32 %K,[pK];\n";
    s += "    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n";
    s += "    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n";

    // Thread/block indices.
    s += "    mov.u32 %tx,%tid.x;\n    mov.u32 %ty,%tid.y;\n    mad.lo.s32 %lin,%ty,16,%tx;\n";
    s += &format!("    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %bm,%tmp,{BM};\n");
    s += &format!("    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %bn,%tmp,{BN};\n");
    s += &format!("    mul.lo.s32 %rit,%ty,{TM};\n"); // row-in-tile = ty*TM
    s += &format!("    mul.lo.s32 %cit,%tx,{TN};\n"); // col-in-tile = tx*TN
    s += &format!("    mul.lo.s32 %rit16,%rit,{BK};\n"); // rit*BK
    s += "    mov.u64 %sA,As;\n    mov.u64 %sB,Bs;\n";
    for i in 0..(TM * TN) {
        s += &format!("    mov.f32 %acc{i},{F0};\n");
    }
    s += &format!(
        "    add.u32 %tmp,%K,{};\n    shr.u32 %ntiles,%tmp,4;\n",
        BK - 1
    );
    s += "    mov.u32 %kt,0;\n";

    s += &format!("KLOOP_{tag}:\n    setp.ge.u32 %p0,%kt,%ntiles;\n    @%p0 bra KEND_{tag};\n");

    // Cooperative load of the A-panel (BM×BK = 1024 elems, 256 threads → 4 each).
    let per = (BM * BK) / THREADS;
    for p in 0..per {
        let off = p * THREADS;
        s += &format!("    add.u32 %e,%lin,{off};\n");
        s += "    shr.u32 %ii,%e,4;\n    and.b32 %kk,%e,15;\n"; // ii=e/16, kk=e%16
        s += "    add.u32 %row,%bm,%ii;\n    mad.lo.s32 %kc,%kt,16,%kk;\n";
        s += &format!("    mov.f32 %v,{F0};\n");
        s += "    setp.lt.u32 %p1,%row,%M;\n    setp.lt.u32 %p2,%kc,%K;\n    and.pred %p1,%p1,%p2;\n";
        s += &format!("    @!%p1 bra SKA_{tag}_{p};\n");
        s += "    mad.lo.s32 %idx,%row,%K,%kc;\n    mul.wide.u32 %off,%idx,4;\n    add.s64 %addr,%A,%off;\n    ld.global.f32 %v,[%addr];\n";
        s += &format!("SKA_{tag}_{p}:\n");
        s += "    mul.wide.u32 %off,%e,4;\n    add.s64 %addr,%sA,%off;\n    st.shared.f32 [%addr],%v;\n";
    }
    // Cooperative load of the B-panel (BK×BN = 1024 elems → 4 each).
    for p in 0..per {
        let off = p * THREADS;
        s += &format!("    add.u32 %e,%lin,{off};\n");
        s += "    shr.u32 %kk,%e,6;\n    and.b32 %jj,%e,63;\n"; // kk=e/64, j=e%64
        s += "    add.u32 %col,%bn,%jj;\n    mad.lo.s32 %kc,%kt,16,%kk;\n";
        s += &format!("    mov.f32 %v,{F0};\n");
        s += "    setp.lt.u32 %p1,%col,%N;\n    setp.lt.u32 %p2,%kc,%K;\n    and.pred %p1,%p1,%p2;\n";
        s += &format!("    @!%p1 bra SKB_{tag}_{p};\n");
        if transposed {
            s += "    mad.lo.s32 %idx,%col,%K,%kc;\n"; // B[col*K + k]
        } else {
            s += "    mad.lo.s32 %idx,%kc,%N,%col;\n"; // B[k*N + col]
        }
        s += "    mul.wide.u32 %off,%idx,4;\n    add.s64 %addr,%B,%off;\n    ld.global.f32 %v,[%addr];\n";
        s += &format!("SKB_{tag}_{p}:\n");
        s += "    mul.wide.u32 %off,%e,4;\n    add.s64 %addr,%sB,%off;\n    st.shared.f32 [%addr],%v;\n";
    }
    s += "    bar.sync 0;\n";

    // Inner product: unrolled over kk = 0..BK.
    for kk in 0..BK {
        // a-base = sA + (rit16 + kk)*4 ; a_i at +i*BK*4
        s += &format!("    add.u32 %idx,%rit16,{kk};\n    mul.wide.u32 %off,%idx,4;\n    add.s64 %aa,%sA,%off;\n");
        for i in 0..TM {
            s += &format!("    ld.shared.f32 %a{i},[%aa+{}];\n", i * BK * 4);
        }
        // b-base = sB + (kk*BN + cit)*4 ; b_j at +j*4
        s += &format!(
            "    add.u32 %idx,%cit,{};\n    mul.wide.u32 %off,%idx,4;\n    add.s64 %bb,%sB,%off;\n",
            kk * BN
        );
        for j in 0..TN {
            s += &format!("    ld.shared.f32 %b{j},[%bb+{}];\n", j * 4);
        }
        for i in 0..TM {
            for j in 0..TN {
                let acc = i * TN + j;
                s += &format!("    fma.rn.f32 %acc{acc},%a{i},%b{j},%acc{acc};\n");
            }
        }
    }
    s += &format!("    bar.sync 0;\n    add.u32 %kt,%kt,1;\n    bra KLOOP_{tag};\n");

    // Store the 4×4 micro-tile, guarded.
    s += &format!("KEND_{tag}:\n");
    for i in 0..TM {
        for j in 0..TN {
            let acc = i * TN + j;
            s += &format!("    add.u32 %row,%bm,%rit;\n    add.u32 %row,%row,{i};\n");
            s += &format!("    add.u32 %col,%bn,%cit;\n    add.u32 %col,%col,{j};\n");
            s += "    setp.lt.u32 %p1,%row,%M;\n    setp.lt.u32 %p2,%col,%N;\n    and.pred %p1,%p1,%p2;\n";
            s += &format!("    @!%p1 bra ST_{tag}_{acc};\n");
            s += "    mad.lo.s32 %idx,%row,%N,%col;\n    mul.wide.u32 %off,%idx,4;\n    add.s64 %addr,%C,%off;\n";
            s += &format!("    st.global.f32 [%addr],%acc{acc};\n");
            s += &format!("ST_{tag}_{acc}:\n");
        }
    }
    s += "    ret;\n}\n";
    s
}

/// The register-blocked GEMM module (both entries), generated once and cached.
pub fn gemm_rb_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        m += &entry("gemm_nt_rb", true, "nt");
        m += &entry("gemm_nn_rb", false, "nn");
        m
    })
    .as_str()
}

/// Block tile dims, exported so the host can compute the grid.
pub const TILE_M: u32 = BM;
pub const TILE_N: u32 = BN;
