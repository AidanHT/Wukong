//! Host-side GPU harness built on `cudarc` (dynamic-loaded NVIDIA driver). Owns the device context,
//! a default stream, and a PTX-module cache; provides typed launch wrappers over device buffers.
//!
//! All driver work funnels through one process-wide [`Gpu`] behind a `Mutex` (acquired via [`gpu`]).
//! That (a) reuses a single primary context + JITed modules across every test/bench instead of
//! re-initializing per call, and (b) serializes driver calls, since `cargo` runs tests on many
//! threads and a CUDA context is current-per-thread.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};

/// A live CUDA device + stream + a cache of JIT-loaded PTX modules (keyed by a stable string).
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
}

impl Gpu {
    fn new() -> Result<Self, DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            modules: HashMap::new(),
        })
    }

    /// Human-readable device name, e.g. "NVIDIA GeForce RTX 4050 Laptop GPU".
    pub fn device_name(&self) -> String {
        self.ctx
            .name()
            .unwrap_or_else(|_| "<unknown CUDA device>".into())
    }

    /// Load (JIT) `ptx` once under `key`, caching the module, and return the named entry function.
    /// The driver compiles PTX→SASS internally, so no `ptxas` is required.
    pub fn function(
        &mut self,
        key: &'static str,
        ptx: &str,
        name: &str,
    ) -> Result<CudaFunction, DriverError> {
        if !self.modules.contains_key(key) {
            let module = self.ctx.load_module(ptx.into())?;
            self.modules.insert(key, module);
        }
        self.modules[key].load_function(name)
    }
}

static GPU: OnceLock<Mutex<Option<Gpu>>> = OnceLock::new();

/// Acquire the process-wide GPU. The inner `Option` is `None` when no CUDA device is reachable
/// (no driver / no GPU): callers should treat that as "skip" rather than fail, so the suite still
/// passes on machines without a GPU even with `--features gpu` compiled in.
pub fn gpu() -> MutexGuard<'static, Option<Gpu>> {
    GPU.get_or_init(|| Mutex::new(Gpu::new().ok()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// True iff a CUDA device is reachable in this process.
pub fn available() -> bool {
    gpu().is_some()
}

/// `y := a*x + y`, computed on the GPU. `a` is fused (`fma.rn`), so a CPU reference using
/// `f32::mul_add(a, x, y)` agrees bit-for-bit.
pub fn saxpy(g: &mut Gpu, a: f32, x: &[f32], y: &mut [f32]) -> Result<(), DriverError> {
    assert_eq!(x.len(), y.len(), "saxpy: x and y must be equal length");
    let n = x.len() as u32;
    let f = g.function("saxpy", crate::ptx::SAXPY, "saxpy")?;
    let x_d = g.stream.memcpy_stod(x)?;
    let mut y_d = g.stream.memcpy_stod(y)?;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&n).arg(&a).arg(&x_d).arg(&mut y_d);
    unsafe { b.launch(cfg)? };
    let out = g.stream.memcpy_dtov(&y_d)?;
    y.copy_from_slice(&out);
    Ok(())
}

/// `out := x + y`, computed on the GPU. Exact IEEE add — matches a CPU reference bit-for-bit.
pub fn vadd(g: &mut Gpu, x: &[f32], y: &[f32], out: &mut [f32]) -> Result<(), DriverError> {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());
    let n = x.len() as u32;
    let f = g.function("vadd", crate::ptx::VADD, "vadd")?;
    let x_d = g.stream.memcpy_stod(x)?;
    let y_d = g.stream.memcpy_stod(y)?;
    let mut out_d = g.stream.memcpy_stod(out)?;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&n).arg(&x_d).arg(&y_d).arg(&mut out_d);
    unsafe { b.launch(cfg)? };
    let res = g.stream.memcpy_dtov(&out_d)?;
    out.copy_from_slice(&res);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `body` with the shared GPU, or skip (printing why) if none is present.
    fn with_gpu(name: &str, body: impl FnOnce(&mut Gpu)) {
        let mut guard = gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => eprintln!("[skip] {name}: no CUDA device reachable"),
        }
    }

    #[test]
    fn saxpy_bit_exact_on_gpu() {
        with_gpu("saxpy_bit_exact_on_gpu", |g| {
            eprintln!("device: {}", g.device_name());
            let n = 4096usize;
            let a = 2.5f32;
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 7.0).collect();
            let y0: Vec<f32> = (0..n).map(|i| (i as f32) * -0.03 + 1.0).collect();
            let mut y = y0.clone();
            saxpy(g, a, &x, &mut y).unwrap();
            for i in 0..n {
                let expect = a.mul_add(x[i], y0[i]);
                assert_eq!(y[i].to_bits(), expect.to_bits(), "saxpy lane {i}");
            }
        });
    }

    #[test]
    fn vadd_bit_exact_on_gpu() {
        with_gpu("vadd_bit_exact_on_gpu", |g| {
            let n = 1000usize; // deliberately not a multiple of the block size
            let x: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5).collect();
            let y: Vec<f32> = (0..n).map(|i| (i as f32) * -0.25).collect();
            let mut out = vec![0.0f32; n];
            vadd(g, &x, &y, &mut out).unwrap();
            for i in 0..n {
                assert_eq!(out[i].to_bits(), (x[i] + y[i]).to_bits(), "vadd lane {i}");
            }
        });
    }
}
