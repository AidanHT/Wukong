//! The GPU implementation of [`mercury_interp::Accelerator`] — the bridge that makes `--backend=gpu`
//! run recognized kernel calls on the device while the rest of the program is still tree-walked by
//! the interpreter (so control flow, buffer layout, and every non-kernel op are bit-identical to the
//! CPU oracle; only the recognized GEMM/norm/activation/reduction calls move to the GPU).
//!
//! Behind the driver's `gpu` feature, so the default toolchain-free build never compiles `cudarc`.
//! A GPU error is surfaced as `Some(Err(..))`, never `None`: declining (`None`) means "fall back to
//! the CPU kernel", and silently running on the CPU when the user asked for `--backend=gpu` would be
//! dishonest. `None` is reserved for shapes/ops the GPU wrappers structurally don't cover.

use mercury_codegen_gpu::Gpu;
use mercury_interp::Accelerator;

/// Wraps the process-wide [`Gpu`] context and routes recognized kernels to its launch wrappers.
/// `calls` counts kernels that actually ran on the device — the tolerance test asserts it is `> 0`
/// so a silent CPU fallback can never masquerade as a passing GPU run.
pub struct GpuAccel<'g> {
    pub gpu: &'g mut Gpu,
    pub calls: u32,
}

impl<'g> GpuAccel<'g> {
    /// Construct over a bound GPU context with a zeroed offload counter.
    pub fn new(gpu: &'g mut Gpu) -> Self {
        GpuAccel { gpu, calls: 0 }
    }
}

impl Accelerator for GpuAccel<'_> {
    fn sgemm_nt(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<Result<(), String>> {
        self.calls += 1;
        Some(
            mercury_codegen_gpu::gpu::gemm_nt(self.gpu, a, b, m, k, n)
                .map(|out| c.copy_from_slice(&out))
                .map_err(|e| format!("GPU gemm_nt failed: {e:?}")),
        )
    }

    fn vmath(&mut self, op: i64, x: &[f32], out: &mut [f32]) -> Option<Result<(), String>> {
        // The GPU vmath kernel only implements a subset of the activation op codes; an unsupported
        // one would panic, so guard with `vmath_supported` and decline (CPU fallback) otherwise.
        if !mercury_codegen_gpu::gpu::vmath_supported(op) {
            return None;
        }
        self.calls += 1;
        Some(
            mercury_codegen_gpu::gpu::vmath(self.gpu, op, x)
                .map(|res| out.copy_from_slice(&res))
                .map_err(|e| format!("GPU vmath failed: {e:?}")),
        )
    }

    fn norm(
        &mut self,
        op: i64,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Option<Result<(), String>> {
        self.calls += 1;
        Some(
            mercury_codegen_gpu::gpu::norm(self.gpu, op, x, rows, cols, eps)
                .map(|res| out.copy_from_slice(&res))
                .map_err(|e| format!("GPU norm failed: {e:?}")),
        )
    }

    fn sreduce(&mut self, op: i64, x: &[f32], y: &[f32]) -> Option<Result<f32, String>> {
        // GPU reduce covers sum/dot/max; decline the rest so the CPU kernel handles them.
        if !mercury_codegen_gpu::gpu::reduce_supported(op) {
            return None;
        }
        let yopt = if mercury_codegen_gpu::gpu::reduce_needs_y(op) {
            Some(y)
        } else {
            None
        };
        self.calls += 1;
        Some(
            mercury_codegen_gpu::gpu::reduce(self.gpu, op, x, yopt)
                .map_err(|e| format!("GPU reduce failed: {e:?}")),
        )
    }
}
