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
