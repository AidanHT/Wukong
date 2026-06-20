//! Fused **flash-attention** on the GPU — the kernel that *lost* on CPU (≈2× slower than the
//! GEMM-dispatch path on AVX2, because materialized attention is compute-bound on the tuned GEMM and
//! the S² scores traffic isn't the bottleneck) but is GPU-shaped: it never materializes S = Q·Kᵀ in
//! HBM, using the online-softmax recurrence to stream K/V once.
//!
//! Layout: single head, Q/K/V/O are `[S, D]` row-major, `D` a multiple of 32. **One warp per query
//! row**; the 32 lanes split the head dim (lane `t` owns d ∈ {t, t+32, …}, i.e. `R = D/32` values).
//! For each key j: each lane computes its partial of Q[i]·K[j], a warp butterfly all-reduce gives
//! the full score, then the online-softmax update rescales the running denominator `l` and the
//! per-lane output accumulators `acc[r]`. `R` is unrolled at PTX-gen time (acc/q in registers), so a
//! kernel is generated per supported D. exp uses `ex2.approx`; tolerance-gated vs a CPU f64 reference.

use std::sync::OnceLock;

/// Generate a flash-attention kernel for head dim `d` (must be a multiple of 32). Name = `flash_d{d}`.
fn entry(d: usize) -> String {
    assert!(d % 32 == 0, "D must be a multiple of 32");
    let r = d / 32; // values per lane
    let log2e = format!("0f{:08X}", std::f32::consts::LOG2_E.to_bits());
    let name = format!("flash_d{d}");

    let mut s = String::new();
    s += &format!(
        ".visible .entry {name}(\n    .param .u32 pS,\n    .param .f32 pScale,\n    .param .u64 pQ,\n    .param .u64 pK,\n    .param .u64 pV,\n    .param .u64 pO\n)\n{{\n"
    );
    s += "    .reg .pred %p0;\n";
    // scalars + per-lane register arrays
    let mut fregs = String::from("%scale,%m,%l,%s,%sfull,%mnew,%corr,%pp,%partial,%vv,%kv,%rt");
    for i in 0..r {
        fregs += &format!(",%q{i},%acc{i}");
    }
    s += &format!("    .reg .f32 {fregs};\n");
    s += "    .reg .b32 %S,%row,%lane,%j,%tmp;\n";
    s += "    .reg .b64 %Q,%K,%V,%O,%qbase,%kbase,%vbase,%obase,%laneoff,%off;\n";

    s += "    ld.param.u32 %S,[pS];\n    ld.param.f32 %scale,[pScale];\n";
    s += "    ld.param.u64 %Q,[pQ];\n    ld.param.u64 %K,[pK];\n    ld.param.u64 %V,[pV];\n    ld.param.u64 %O,[pO];\n";
    s += "    cvta.to.global.u64 %Q,%Q;\n    cvta.to.global.u64 %K,%K;\n    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %O,%O;\n";
    s += &format!(
        "    mov.u32 %row,%ctaid.x;\n    setp.ge.u32 %p0,%row,%S;\n    @%p0 bra RET_{name};\n"
    );
    s += "    mov.u32 %lane,%tid.x;\n    mul.wide.u32 %laneoff,%lane,4;\n";
    // qbase = Q + row*D*4 ; preload q[r]
    s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %qbase,%Q,%off;\n    add.s64 %qbase,%qbase,%laneoff;\n");
    for i in 0..r {
        s += &format!("    ld.global.f32 %q{i},[%qbase+{}];\n", i * 128);
    }
    // obase = O + row*D*4
    s += &format!("    mul.lo.s32 %tmp,%row,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %obase,%O,%off;\n    add.s64 %obase,%obase,%laneoff;\n");
    // init running state
    s += "    mov.f32 %m,0fFF800000;\n    mov.f32 %l,0f00000000;\n";
    for i in 0..r {
        s += &format!("    mov.f32 %acc{i},0f00000000;\n");
    }

    // for j in 0..S
    s += "    mov.u32 %j,0;\n";
    s += &format!("J_{name}:\n    setp.ge.u32 %p0,%j,%S;\n    @%p0 bra DONE_{name};\n");
    // kbase = K + j*D*4 + laneoff
    s += &format!("    mul.lo.s32 %tmp,%j,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %kbase,%K,%off;\n    add.s64 %kbase,%kbase,%laneoff;\n");
    // partial = sum_r q[r]*K[j][lane+32r]
    s += "    mov.f32 %partial,0f00000000;\n";
    for i in 0..r {
        s += &format!(
            "    ld.global.f32 %kv,[%kbase+{}];\n    fma.rn.f32 %partial,%q{i},%kv,%partial;\n",
            i * 128
        );
    }
    // warp all-reduce add -> sfull
    s += "    mov.f32 %sfull,%partial;\n";
    for off in [16, 8, 4, 2, 1] {
        s += &format!("    shfl.sync.bfly.b32 %rt,%sfull,{off},0x1f,0xffffffff;\n    add.f32 %sfull,%sfull,%rt;\n");
    }
    // s = sfull*scale ; online softmax update
    s += "    mul.f32 %s,%sfull,%scale;\n";
    s += "    max.f32 %mnew,%m,%s;\n";
    // corr = exp(m - mnew) ; pp = exp(s - mnew)
    s += &format!("    sub.f32 %corr,%m,%mnew;\n    mul.f32 %corr,%corr,{log2e};\n    ex2.approx.f32 %corr,%corr;\n");
    s += &format!(
        "    sub.f32 %pp,%s,%mnew;\n    mul.f32 %pp,%pp,{log2e};\n    ex2.approx.f32 %pp,%pp;\n"
    );
    // l = l*corr + pp
    s += "    fma.rn.f32 %l,%l,%corr,%pp;\n";
    // vbase = V + j*D*4 + laneoff ; acc[r] = acc[r]*corr + pp*V[j][lane+32r]
    s += &format!("    mul.lo.s32 %tmp,%j,{d};\n    mul.wide.u32 %off,%tmp,4;\n    add.s64 %vbase,%V,%off;\n    add.s64 %vbase,%vbase,%laneoff;\n");
    for i in 0..r {
        s += &format!("    ld.global.f32 %vv,[%vbase+{}];\n    mul.f32 %acc{i},%acc{i},%corr;\n    fma.rn.f32 %acc{i},%pp,%vv,%acc{i};\n", i * 128);
    }
    s += "    mov.f32 %m,%mnew;\n";
    s += &format!("    add.u32 %j,%j,1;\n    bra J_{name};\n");

    // O[row][lane+32r] = acc[r] / l
    s += &format!("DONE_{name}:\n");
    for i in 0..r {
        s += &format!(
            "    div.rn.f32 %acc{i},%acc{i},%l;\n    st.global.f32 [%obase+{}],%acc{i};\n",
            i * 128
        );
    }
    s += &format!("RET_{name}:\n    ret;\n}}\n");
    s
}

/// Flash-attention module. Generates a kernel per supported head dim.
pub fn flash_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");
        for &d in &SUPPORTED_D {
            m += &entry(d);
        }
        m
    })
    .as_str()
}

/// Head dims with a generated kernel (the common transformer values).
pub const SUPPORTED_D: [usize; 3] = [32, 64, 128];
