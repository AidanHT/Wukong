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

/// `dst_f16[i] = (f16)src_f32[i]` — round-to-nearest-even narrowing (`cvt.rn.f16.f32`), matching
/// `half::f16::from_f32` bit-for-bit. The glue that lets a resident fp16 pipeline chain an f32-output
/// stage (e.g. RMSNorm) into an f16-input WMMA GEMM on-device, with no host round-trip. `src` is N×f32,
/// `dst` is N×f16 (2 bytes/elem); one thread per element, grid-stride not needed (host sizes the grid).
pub const CAST_F32_F16: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry cast_f32_f16(
    .param .u32 n,
    .param .u64 src,
    .param .u64 dst
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<2>;
    .reg .b16   %h<2>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<6>;

    ld.param.u32    %r1, [n];
    ld.param.u64    %rd1, [src];
    ld.param.u64    %rd2, [dst];

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
    ld.global.f32   %f1, [%rd4];
    cvt.rn.f16.f32  %h1, %f1;
    mul.wide.u32 %rd5, %r2, 2;
    add.s64     %rd4, %rd2, %rd5;
    st.global.b16   [%rd4], %h1;

DONE:
    ret;
}
"#;

/// `dst[i] = src[i]` — a pure streaming **copy**, the canonical memory-bandwidth kernel (milestone
/// M9). Vectorized 128-bit access (`ld.global.v4.f32` / `st.global.v4.f32` = 4 floats/op) with **4×
/// ILP**: each thread issues four *independent* float4 loads (distinct registers + grid-stride-spaced
/// addresses) before any store, so four memory requests are in flight per thread. That extra
/// memory-level parallelism is what saturates HBM on a 1:1 read/write copy — with one float4 per
/// thread there are only `len/4` threads, too few outstanding requests to hide DRAM latency + bus
/// turnaround (measured ~84%); 4-wide clears 90%. Every warp's access stays coalesced (32 threads ×
/// 16 B = one 512-B segment) because the four groups are each grid-stride-spaced. `pn4` is the float4
/// count (`len/4`); the host guarantees `len % 4 == 0` and 16-B alignment (cudaMalloc gives ≥256 B).
/// The grid-stride GRID/TAIL split covers any `n4` for any launch size — each thread copies exactly
/// its `{i, i+S, i+2S, …}` positions, four at a time while they fit, one at a time for the remainder.
/// No arithmetic between load and store: this measures DRAM throughput, the traffic of any elementwise
/// pass (`2·len` floats moved).
pub const COPY_V4: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry copy_v4(
    .param .u32 pn4,
    .param .u64 psrc,
    .param .u64 pdst
)
{
    .reg .pred %p;
    .reg .b32  %n4, %tix, %ntx, %cta, %ncta, %i, %i1, %i2, %i3, %S, %S4;
    .reg .b64  %src, %dst, %o, %a, %b, %o1, %a1, %b1, %o2, %a2, %b2, %o3, %a3, %b3;
    .reg .f32  %x0, %x1, %x2, %x3, %y0, %y1, %y2, %y3;
    .reg .f32  %z0, %z1, %z2, %z3, %w0, %w1, %w2, %w3;

    ld.param.u32 %n4, [pn4];
    ld.param.u64 %src, [psrc];
    ld.param.u64 %dst, [pdst];
    cvta.to.global.u64 %src, %src;
    cvta.to.global.u64 %dst, %dst;

    mov.u32 %tix, %tid.x;
    mov.u32 %ntx, %ntid.x;
    mov.u32 %cta, %ctaid.x;
    mov.u32 %ncta, %nctaid.x;
    mad.lo.s32 %i, %cta, %ntx, %tix;
    mul.lo.s32 %S, %ncta, %ntx;
    shl.b32 %S4, %S, 2;

GRID:
    add.u32 %i1, %i, %S;
    add.u32 %i2, %i1, %S;
    add.u32 %i3, %i2, %S;
    setp.ge.u32 %p, %i3, %n4;
    @%p bra TAIL;
    mul.wide.u32 %o, %i, 16;
    add.s64 %a, %src, %o;
    add.s64 %b, %dst, %o;
    mul.wide.u32 %o1, %i1, 16;
    add.s64 %a1, %src, %o1;
    add.s64 %b1, %dst, %o1;
    mul.wide.u32 %o2, %i2, 16;
    add.s64 %a2, %src, %o2;
    add.s64 %b2, %dst, %o2;
    mul.wide.u32 %o3, %i3, 16;
    add.s64 %a3, %src, %o3;
    add.s64 %b3, %dst, %o3;
    ld.global.v4.f32 {%x0, %x1, %x2, %x3}, [%a];
    ld.global.v4.f32 {%y0, %y1, %y2, %y3}, [%a1];
    ld.global.v4.f32 {%z0, %z1, %z2, %z3}, [%a2];
    ld.global.v4.f32 {%w0, %w1, %w2, %w3}, [%a3];
    st.global.v4.f32 [%b], {%x0, %x1, %x2, %x3};
    st.global.v4.f32 [%b1], {%y0, %y1, %y2, %y3};
    st.global.v4.f32 [%b2], {%z0, %z1, %z2, %z3};
    st.global.v4.f32 [%b3], {%w0, %w1, %w2, %w3};
    add.u32 %i, %i, %S4;
    bra GRID;
TAIL:
    setp.ge.u32 %p, %i, %n4;
    @%p bra DONE;
    mul.wide.u32 %o, %i, 16;
    add.s64 %a, %src, %o;
    add.s64 %b, %dst, %o;
    ld.global.v4.f32 {%x0, %x1, %x2, %x3}, [%a];
    st.global.v4.f32 [%b], {%x0, %x1, %x2, %x3};
    add.u32 %i, %i, %S;
    bra TAIL;
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

/// Tiled shared-memory f32 GEMM — the GPU analogue of the AVX2 `mercury_sgemm` microkernel. Two
/// entries: `gemm_nn` (`C = A·B`) and `gemm_nt` (`C = A·Bᵀ`, the nn.Linear spelling). 16×16 thread
/// blocks each compute a 16×16 C tile; A and B tiles are staged in shared memory (`As`/`Bs`) and the
/// `TILE=16` inner product runs from there, so each global element is loaded once per tile instead of
/// once per MAC. One C element per thread (no register blocking yet) — a clear win over a naive nest;
/// the register-blocked / vectorized version that chases cuBLAS comes next. Uses **named PTX
/// registers** for legibility. Out-of-range threads load zeros (keeping `bar.sync` uniform) and skip
/// the C store, so ragged M/N/K are handled. `A,B,C` are f32; `fma.rn` accumulation.
pub const GEMM: &str = r#"
.version 7.8
.target sm_89
.address_size 64

.visible .entry gemm_nt(
    .param .u32 pM,
    .param .u32 pN,
    .param .u32 pK,
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pC
)
{
    .reg .pred %p0, %p1, %p2, %p3;
    .reg .f32 %acc, %a, %b, %va, %vb;
    .reg .b32 %M, %N, %K, %tx, %ty, %col, %row, %t, %ntiles, %kA, %kB, %kk, %tmp, %tmp2;
    .reg .b64 %A, %B, %C, %off, %addr, %sA, %sB, %sAt, %sBt;
    .shared .align 4 .b8 As[1024];
    .shared .align 4 .b8 Bs[1024];

    ld.param.u32 %M, [pM];
    ld.param.u32 %N, [pN];
    ld.param.u32 %K, [pK];
    ld.param.u64 %A, [pA];
    ld.param.u64 %B, [pB];
    ld.param.u64 %C, [pC];
    cvta.to.global.u64 %A, %A;
    cvta.to.global.u64 %B, %B;
    cvta.to.global.u64 %C, %C;

    mov.u32 %tx, %tid.x;
    mov.u32 %ty, %tid.y;
    mov.u32 %tmp, %ctaid.x;
    mad.lo.s32 %col, %tmp, 16, %tx;
    mov.u32 %tmp, %ctaid.y;
    mad.lo.s32 %row, %tmp, 16, %ty;

    mov.f32 %acc, 0f00000000;
    add.u32 %tmp, %K, 15;
    shr.u32 %ntiles, %tmp, 4;

    mov.u64 %sA, As;
    mov.u64 %sB, Bs;
    mad.lo.s32 %tmp, %ty, 16, %tx;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %sAt, %sA, %off;
    add.s64 %sBt, %sB, %off;

    mov.u32 %t, 0;
TILELOOP:
    setp.ge.u32 %p0, %t, %ntiles;
    @%p0 bra ENDTILES;

    mad.lo.s32 %kA, %t, 16, %tx;
    mov.f32 %va, 0f00000000;
    setp.ge.u32 %p1, %row, %M;
    setp.ge.u32 %p2, %kA, %K;
    or.pred %p3, %p1, %p2;
    @%p3 bra STOREA;
    mad.lo.s32 %tmp, %row, %K, %kA;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %A, %off;
    ld.global.f32 %va, [%addr];
STOREA:
    st.shared.f32 [%sAt], %va;

    mad.lo.s32 %kB, %t, 16, %ty;
    mov.f32 %vb, 0f00000000;
    setp.ge.u32 %p1, %col, %N;
    setp.ge.u32 %p2, %kB, %K;
    or.pred %p3, %p1, %p2;
    @%p3 bra STOREB;
    mad.lo.s32 %tmp, %col, %K, %kB;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %B, %off;
    ld.global.f32 %vb, [%addr];
STOREB:
    st.shared.f32 [%sBt], %vb;

    bar.sync 0;

    mov.u32 %kk, 0;
INNER:
    setp.ge.u32 %p0, %kk, 16;
    @%p0 bra ENDINNER;
    mad.lo.s32 %tmp, %ty, 16, %kk;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %sA, %off;
    ld.shared.f32 %a, [%addr];
    mad.lo.s32 %tmp2, %kk, 16, %tx;
    mul.wide.u32 %off, %tmp2, 4;
    add.s64 %addr, %sB, %off;
    ld.shared.f32 %b, [%addr];
    fma.rn.f32 %acc, %a, %b, %acc;
    add.u32 %kk, %kk, 1;
    bra INNER;
ENDINNER:
    bar.sync 0;
    add.u32 %t, %t, 1;
    bra TILELOOP;
ENDTILES:
    setp.ge.u32 %p1, %row, %M;
    setp.ge.u32 %p2, %col, %N;
    or.pred %p3, %p1, %p2;
    @%p3 bra DONE;
    mad.lo.s32 %tmp, %row, %N, %col;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %C, %off;
    st.global.f32 [%addr], %acc;
DONE:
    ret;
}

.visible .entry gemm_nn(
    .param .u32 pM,
    .param .u32 pN,
    .param .u32 pK,
    .param .u64 pA,
    .param .u64 pB,
    .param .u64 pC
)
{
    .reg .pred %p0, %p1, %p2, %p3;
    .reg .f32 %acc, %a, %b, %va, %vb;
    .reg .b32 %M, %N, %K, %tx, %ty, %col, %row, %t, %ntiles, %kA, %kB, %kk, %tmp, %tmp2;
    .reg .b64 %A, %B, %C, %off, %addr, %sA, %sB, %sAt, %sBt;
    .shared .align 4 .b8 As[1024];
    .shared .align 4 .b8 Bs[1024];

    ld.param.u32 %M, [pM];
    ld.param.u32 %N, [pN];
    ld.param.u32 %K, [pK];
    ld.param.u64 %A, [pA];
    ld.param.u64 %B, [pB];
    ld.param.u64 %C, [pC];
    cvta.to.global.u64 %A, %A;
    cvta.to.global.u64 %B, %B;
    cvta.to.global.u64 %C, %C;

    mov.u32 %tx, %tid.x;
    mov.u32 %ty, %tid.y;
    mov.u32 %tmp, %ctaid.x;
    mad.lo.s32 %col, %tmp, 16, %tx;
    mov.u32 %tmp, %ctaid.y;
    mad.lo.s32 %row, %tmp, 16, %ty;

    mov.f32 %acc, 0f00000000;
    add.u32 %tmp, %K, 15;
    shr.u32 %ntiles, %tmp, 4;

    mov.u64 %sA, As;
    mov.u64 %sB, Bs;
    mad.lo.s32 %tmp, %ty, 16, %tx;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %sAt, %sA, %off;
    add.s64 %sBt, %sB, %off;

    mov.u32 %t, 0;
TILELOOP2:
    setp.ge.u32 %p0, %t, %ntiles;
    @%p0 bra ENDTILES2;

    mad.lo.s32 %kA, %t, 16, %tx;
    mov.f32 %va, 0f00000000;
    setp.ge.u32 %p1, %row, %M;
    setp.ge.u32 %p2, %kA, %K;
    or.pred %p3, %p1, %p2;
    @%p3 bra STOREA2;
    mad.lo.s32 %tmp, %row, %K, %kA;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %A, %off;
    ld.global.f32 %va, [%addr];
STOREA2:
    st.shared.f32 [%sAt], %va;

    mad.lo.s32 %kB, %t, 16, %ty;
    mov.f32 %vb, 0f00000000;
    setp.ge.u32 %p1, %col, %N;
    setp.ge.u32 %p2, %kB, %K;
    or.pred %p3, %p1, %p2;
    @%p3 bra STOREB2;
    mad.lo.s32 %tmp, %kB, %N, %col;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %B, %off;
    ld.global.f32 %vb, [%addr];
STOREB2:
    st.shared.f32 [%sBt], %vb;

    bar.sync 0;

    mov.u32 %kk, 0;
INNER2:
    setp.ge.u32 %p0, %kk, 16;
    @%p0 bra ENDINNER2;
    mad.lo.s32 %tmp, %ty, 16, %kk;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %sA, %off;
    ld.shared.f32 %a, [%addr];
    mad.lo.s32 %tmp2, %kk, 16, %tx;
    mul.wide.u32 %off, %tmp2, 4;
    add.s64 %addr, %sB, %off;
    ld.shared.f32 %b, [%addr];
    fma.rn.f32 %acc, %a, %b, %acc;
    add.u32 %kk, %kk, 1;
    bra INNER2;
ENDINNER2:
    bar.sync 0;
    add.u32 %t, %t, 1;
    bra TILELOOP2;
ENDTILES2:
    setp.ge.u32 %p1, %row, %M;
    setp.ge.u32 %p2, %col, %N;
    or.pred %p3, %p1, %p2;
    @%p3 bra DONE2;
    mad.lo.s32 %tmp, %row, %N, %col;
    mul.wide.u32 %off, %tmp, 4;
    add.s64 %addr, %C, %off;
    st.global.f32 [%addr], %acc;
DONE2:
    ret;
}
"#;
