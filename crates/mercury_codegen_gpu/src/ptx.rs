//! Hand-emitted PTX kernels. Mercury is a compiler, so its GPU "backend" emits PTX text directly;
//! the NVIDIA driver JIT-compiles PTX→SASS at load (`cuModuleLoadData`), needing no `nvcc`/`ptxas`.
//!
//! These are the GPU analogue of the AVX2 microkernels in `mercury_runtime`: each recognized op
//! (saxpy, elementwise, reduction, GEMM, …) gets a kernel the host launches over device buffers.
//! Target is `sm_89` (Ada / RTX 4050). Kept deliberately simple and correct first; tiling and
//! tensor-core variants are added in later phases.

/// `y[i] = a*x[i] + y[i]` (the canonical Phase-0 spike). `a` is fused via `fma.rn` so a CPU
/// reference using `f32::mul_add` matches bit-for-bit.
pub const SAXPY: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry saxpy(
    .param .u32 n,
    .param .f32 a,
    .param .u64 x,
    .param .u64 y
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<6>;

    ld.param.u32    %r1, [n];
    ld.param.f32    %f1, [a];
    ld.param.u64    %rd1, [x];
    ld.param.u64    %rd2, [y];

    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %ctaid.x;
    mov.u32     %r4, %tid.x;
    mad.lo.s32  %r2, %r3, %r2, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra    DONE;

    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    mul.wide.u32 %rd3, %r2, 4;
    add.s64     %rd4, %rd1, %rd3;
    add.s64     %rd5, %rd2, %rd3;

    ld.global.f32   %f2, [%rd4];
    ld.global.f32   %f3, [%rd5];
    fma.rn.f32      %f2, %f1, %f2, %f3;
    st.global.f32   [%rd5], %f2;

DONE:
    ret;
}
"#;

/// `out[i] = x[i] + y[i]` — exact IEEE add, matches a CPU reference bit-for-bit.
pub const VADD: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry vadd(
    .param .u32 n,
    .param .u64 x,
    .param .u64 y,
    .param .u64 out
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<8>;

    ld.param.u32    %r1, [n];
    ld.param.u64    %rd1, [x];
    ld.param.u64    %rd2, [y];
    ld.param.u64    %rd3, [out];

    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %ctaid.x;
    mov.u32     %r4, %tid.x;
    mad.lo.s32  %r2, %r3, %r2, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra    DONE;

    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;
    mul.wide.u32 %rd4, %r2, 4;
    add.s64     %rd5, %rd1, %rd4;
    add.s64     %rd6, %rd2, %rd4;
    add.s64     %rd7, %rd3, %rd4;

    ld.global.f32   %f1, [%rd5];
    ld.global.f32   %f2, [%rd6];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rd7], %f3;

DONE:
    ret;
}
"#;

use std::sync::OnceLock;

/// `0f` + the IEEE-754 f32 hex of `x` — a PTX f32 immediate (`mov.f32 %f, 0f3F800000`). Built from
/// `f32::to_bits` so the constant is *exactly* what Rust uses (no hand-transcribed-hex mistakes).
fn hexf(x: f32) -> String {
    format!("0f{:08X}", x.to_bits())
}

/// The elementwise activation module (`out[i] = f(x[i])`) — the GPU analogue of `mercury_vmath_f32`.
/// One `.visible .entry` per activation; the host loads the module once (keyed `"vmath"`) and picks
/// the entry by name. Transcendentals use the Ada SFU fast paths (`ex2.approx`, `tanh.approx`,
/// `rcp.approx`) — a few ULP off the CPU Cephes polys, so this path is tolerance-gated, not
/// bit-exact. The exact `f(x)` forms mirror `mercury_runtime::vmath` (`tanh1`, `sigmoid1`, `gelu1`).
pub fn vmath_ptx() -> &'static str {
    static PTX: OnceLock<String> = OnceLock::new();
    PTX.get_or_init(|| {
        let log2e = hexf(std::f32::consts::LOG2_E); // 1.442695…
        let neg_log2e = hexf(-std::f32::consts::LOG2_E);
        let one = hexf(1.0);
        let half = hexf(0.5);
        let c0 = hexf((2.0f32 / std::f32::consts::PI).sqrt()); // gelu sqrt(2/pi)
        let c1 = hexf(0.044715); // gelu cubic coeff

        // Per-entry prologue: i = blockIdx*blockDim + tid; bounds-check; load x[i] into %f1.
        let entry = |name: &str, body: &str| -> String {
            format!(
                r#"
.visible .entry {name}(
    .param .u32 n,
    .param .u64 x,
    .param .u64 out
)
{{
    .reg .pred  %p<2>;
    .reg .f32   %f<8>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<6>;

    ld.param.u32    %r1, [n];
    ld.param.u64    %rd1, [x];
    ld.param.u64    %rd2, [out];
    mov.u32     %r2, %ntid.x;
    mov.u32     %r3, %ctaid.x;
    mov.u32     %r4, %tid.x;
    mad.lo.s32  %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra    DONE;
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64     %rd4, %rd1, %rd3;
    add.s64     %rd5, %rd2, %rd3;
    ld.global.f32   %f1, [%rd4];
{body}
DONE:
    ret;
}}
"#
            )
        };

        let mut m = String::from(".version 7.8\n.target sm_89\n.address_size 64\n");

        // relu(x) = max(x, 0)
        m += &entry(
            "relu",
            "    max.f32 %f1, %f1, 0f00000000;\n    st.global.f32 [%rd5], %f1;",
        );
        // exp(x) = 2^(x·log2 e)
        m += &entry(
            "exp",
            &format!(
                "    mul.f32 %f1, %f1, {log2e};\n    ex2.approx.f32 %f1, %f1;\n    st.global.f32 [%rd5], %f1;"
            ),
        );
        // sigmoid(x) = 1/(1+exp(-x))
        m += &entry(
            "sigmoid",
            &format!(
                "    mul.f32 %f2, %f1, {neg_log2e};\n    ex2.approx.f32 %f2, %f2;\n    add.f32 %f2, %f2, {one};\n    rcp.approx.f32 %f2, %f2;\n    st.global.f32 [%rd5], %f2;"
            ),
        );
        // tanh(x) — hardware approximation
        m += &entry(
            "tanh",
            "    tanh.approx.f32 %f1, %f1;\n    st.global.f32 [%rd5], %f1;",
        );
        // silu(x) = x·sigmoid(x)
        m += &entry(
            "silu",
            &format!(
                "    mul.f32 %f2, %f1, {neg_log2e};\n    ex2.approx.f32 %f2, %f2;\n    add.f32 %f2, %f2, {one};\n    rcp.approx.f32 %f2, %f2;\n    mul.f32 %f1, %f1, %f2;\n    st.global.f32 [%rd5], %f1;"
            ),
        );
        // gelu(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
        m += &entry(
            "gelu",
            &format!(
                "    mul.f32 %f2, %f1, %f1;\n    mul.f32 %f2, %f2, %f1;\n    fma.rn.f32 %f2, %f2, {c1}, %f1;\n    mul.f32 %f2, %f2, {c0};\n    tanh.approx.f32 %f2, %f2;\n    add.f32 %f2, %f2, {one};\n    mul.f32 %f3, %f1, {half};\n    mul.f32 %f2, %f2, %f3;\n    st.global.f32 [%rd5], %f2;"
            ),
        );
        m
    })
    .as_str()
}

/// Deterministic block reductions — the GPU analogue of `mercury_sreduce_f32`. A **fixed** grid of
/// `GRID` blocks × `BLOCK` threads grid-strides the input into per-thread accumulators, tree-reduces
/// each block in shared memory, and writes one partial per block. The host then sums the `GRID`
/// partials in ascending block order. Because the grid is fixed (independent of occupancy) and the
/// tree + host combine are fixed-order, the result is **identical run-to-run** — determinism rests
/// on the fixed decomposition + ascending combine, not associativity (same idea as the CPU kernel).
/// `BLOCK` must stay 256 (the `sdata[1024]` static shared array and the unrolled tree assume it).
pub const REDUCE: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry reduce_sum(
    .param .u32 n,
    .param .u64 x,
    .param .u64 partials
)
{
    .reg .pred  %p<3>;
    .reg .f32   %f<6>;
    .reg .b32   %r<12>;
    .reg .b64   %rd<10>;
    .shared .align 4 .b8 sdata[1024];

    ld.param.u32 %r1, [n];
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [partials];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.s32 %r6, %r4, %r3, %r2;
    mul.lo.s32 %r7, %r5, %r3;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r8, %r6;
LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra ENDLOOP;
    mul.wide.u32 %rd3, %r8, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    add.f32 %f1, %f1, %f2;
    add.u32 %r8, %r8, %r7;
    bra LOOP;
ENDLOOP:
    mul.wide.u32 %rd5, %r2, 4;
    mov.u64 %rd6, sdata;
    add.s64 %rd6, %rd6, %rd5;
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;

    shr.u32 %r9, %r3, 1;
TREE:
    setp.eq.u32 %p2, %r9, 0;
    @%p2 bra DONE_TREE;
    setp.ge.u32 %p1, %r2, %r9;
    @%p1 bra SKIP;
    add.u32 %r10, %r2, %r9;
    mul.wide.u32 %rd7, %r10, 4;
    mov.u64 %rd8, sdata;
    add.s64 %rd8, %rd8, %rd7;
    ld.shared.f32 %f3, [%rd8];
    ld.shared.f32 %f4, [%rd6];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd6], %f4;
SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    bra TREE;
DONE_TREE:
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra END;
    mov.u64 %rd6, sdata;
    ld.shared.f32 %f5, [%rd6];
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd2, %rd2, %rd3;
    st.global.f32 [%rd2], %f5;
END:
    ret;
}

.visible .entry reduce_dot(
    .param .u32 n,
    .param .u64 x,
    .param .u64 y,
    .param .u64 partials
)
{
    .reg .pred  %p<3>;
    .reg .f32   %f<7>;
    .reg .b32   %r<12>;
    .reg .b64   %rd<12>;
    .shared .align 4 .b8 sdata[1024];

    ld.param.u32 %r1, [n];
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd9, [y];
    ld.param.u64 %rd2, [partials];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd9, %rd9;
    cvta.to.global.u64 %rd2, %rd2;

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.s32 %r6, %r4, %r3, %r2;
    mul.lo.s32 %r7, %r5, %r3;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r8, %r6;
LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra ENDLOOP;
    mul.wide.u32 %rd3, %r8, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd10, %rd9, %rd3;
    ld.global.f32 %f2, [%rd4];
    ld.global.f32 %f6, [%rd10];
    fma.rn.f32 %f1, %f2, %f6, %f1;
    add.u32 %r8, %r8, %r7;
    bra LOOP;
ENDLOOP:
    mul.wide.u32 %rd5, %r2, 4;
    mov.u64 %rd6, sdata;
    add.s64 %rd6, %rd6, %rd5;
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;

    shr.u32 %r9, %r3, 1;
TREE:
    setp.eq.u32 %p2, %r9, 0;
    @%p2 bra DONE_TREE;
    setp.ge.u32 %p1, %r2, %r9;
    @%p1 bra SKIP;
    add.u32 %r10, %r2, %r9;
    mul.wide.u32 %rd7, %r10, 4;
    mov.u64 %rd8, sdata;
    add.s64 %rd8, %rd8, %rd7;
    ld.shared.f32 %f3, [%rd8];
    ld.shared.f32 %f4, [%rd6];
    add.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd6], %f4;
SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    bra TREE;
DONE_TREE:
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra END;
    mov.u64 %rd6, sdata;
    ld.shared.f32 %f5, [%rd6];
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd2, %rd2, %rd3;
    st.global.f32 [%rd2], %f5;
END:
    ret;
}

.visible .entry reduce_max(
    .param .u32 n,
    .param .u64 x,
    .param .u64 partials
)
{
    .reg .pred  %p<3>;
    .reg .f32   %f<6>;
    .reg .b32   %r<12>;
    .reg .b64   %rd<10>;
    .shared .align 4 .b8 sdata[1024];

    ld.param.u32 %r1, [n];
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [partials];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %nctaid.x;
    mad.lo.s32 %r6, %r4, %r3, %r2;
    mul.lo.s32 %r7, %r5, %r3;

    mov.f32 %f1, 0fFF800000;
    mov.u32 %r8, %r6;
LOOP:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra ENDLOOP;
    mul.wide.u32 %rd3, %r8, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    max.f32 %f1, %f1, %f2;
    add.u32 %r8, %r8, %r7;
    bra LOOP;
ENDLOOP:
    mul.wide.u32 %rd5, %r2, 4;
    mov.u64 %rd6, sdata;
    add.s64 %rd6, %rd6, %rd5;
    st.shared.f32 [%rd6], %f1;
    bar.sync 0;

    shr.u32 %r9, %r3, 1;
TREE:
    setp.eq.u32 %p2, %r9, 0;
    @%p2 bra DONE_TREE;
    setp.ge.u32 %p1, %r2, %r9;
    @%p1 bra SKIP;
    add.u32 %r10, %r2, %r9;
    mul.wide.u32 %rd7, %r10, 4;
    mov.u64 %rd8, sdata;
    add.s64 %rd8, %rd8, %rd7;
    ld.shared.f32 %f3, [%rd8];
    ld.shared.f32 %f4, [%rd6];
    max.f32 %f4, %f4, %f3;
    st.shared.f32 [%rd6], %f4;
SKIP:
    bar.sync 0;
    shr.u32 %r9, %r9, 1;
    bra TREE;
DONE_TREE:
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra END;
    mov.u64 %rd6, sdata;
    ld.shared.f32 %f5, [%rd6];
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd2, %rd2, %rd3;
    st.global.f32 [%rd2], %f5;
END:
    ret;
}
"#;
