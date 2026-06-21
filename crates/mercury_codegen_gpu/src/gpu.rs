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
    sys, CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;

/// A live CUDA device + stream + a cache of JIT-loaded PTX modules (keyed by a stable string).
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    /// Installed driver version — part of the on-disk cubin cache key (a cubin is driver-ABI specific).
    driver_tag: i32,
}

impl Gpu {
    fn new() -> Result<Self, DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            modules: HashMap::new(),
            driver_tag: crate::cubin::driver_version(),
        })
    }

    /// Human-readable device name, e.g. "NVIDIA GeForce RTX 4050 Laptop GPU".
    pub fn device_name(&self) -> String {
        self.ctx
            .name()
            .unwrap_or_else(|_| "<unknown CUDA device>".into())
    }

    /// Load `ptx` once under `key`, caching the module in-process, and return the named entry
    /// function. The first load consults the persistent **cubin cache** (M10): a warm process loads
    /// precompiled SASS instead of re-JITing the PTX. The driver compiles PTX→SASS internally, so no
    /// external `ptxas` is required either way.
    pub fn function(
        &mut self,
        key: &'static str,
        ptx: &str,
        name: &str,
    ) -> Result<CudaFunction, DriverError> {
        if !self.modules.contains_key(key) {
            let module = self.load_module_cached(ptx)?;
            self.modules.insert(key, module);
        }
        self.modules[key].load_function(name)
    }

    /// Load a module for `ptx`, preferring a cached cubin over a fresh JIT. Warm path: a previously
    /// cached cubin loads via `cuModuleLoad` (no compilation). Cold path: compile to a cubin once,
    /// persist it, and load that. Every cubin-route failure (no linker, unwritable cache, a stale or
    /// driver-incompatible cubin) degrades to the proven direct-PTX JIT — so caching is a pure
    /// optimization that can never break a load that would otherwise succeed.
    fn load_module_cached(&self, ptx: &str) -> Result<Arc<CudaModule>, DriverError> {
        let path = crate::cubin::cache_path(ptx, self.driver_tag);
        if path.exists() {
            if let Ok(m) = self.ctx.load_module(Ptx::from_file(&path)) {
                return Ok(m); // warm: loaded precompiled SASS, no JIT
            }
            let _ = std::fs::remove_file(&path); // stale/incompatible → drop and recompile
        }
        // The raw cuLink compile needs a context current on this thread (cudarc's load_module binds
        // itself, but ptx_to_cubin does not) — otherwise it fails and we'd silently skip caching.
        let _ = self.ctx.bind_to_thread();
        if let Ok(cubin) = crate::cubin::ptx_to_cubin(ptx) {
            if crate::cubin::write_atomic(&path, &cubin).is_ok() {
                if let Ok(m) = self.ctx.load_module(Ptx::from_file(&path)) {
                    return Ok(m); // cold: compiled, cached, and loaded the cubin (one compile)
                }
            }
        }
        self.ctx.load_module(ptx.into()) // fallback: direct PTX JIT (the original path)
    }

    /// Query an integer device attribute (e.g. memory clock, bus width, SM count). Device-level, so it
    /// needs no current context — `cuInit` has already run by the time a `Gpu` exists. `None` if the
    /// driver call fails.
    fn device_attr(&self, attr: sys::CUdevice_attribute) -> Option<i32> {
        let mut dev: sys::CUdevice = 0;
        let mut val: i32 = 0;
        unsafe {
            sys::cuDeviceGet(&mut dev, 0).result().ok()?;
            sys::cuDeviceGetAttribute(&mut val, attr, dev).result().ok()?;
        }
        Some(val)
    }

    /// Number of streaming multiprocessors (falls back to a plausible 20 if unqueryable) — used to size
    /// a grid that saturates the device for memory-bound (grid-stride) kernels.
    pub fn sm_count(&self) -> i32 {
        self.device_attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(20)
    }

    /// **Theoretical peak HBM bandwidth, GB/s** — the honest M9 denominator. Uses the exact formula
    /// NVIDIA's own `deviceQuery` prints: `2 × memClock × (busWidth/8)` (the ×2 is DDR; for GDDR6 the
    /// reported "memory clock" already folds in the per-pin multiplier, so this matches the spec
    /// sheet). `MEMORY_CLOCK_RATE` is kHz, `GLOBAL_MEMORY_BUS_WIDTH` is bits. `None` if unqueryable.
    pub fn peak_hbm_gbs(&self) -> Option<f64> {
        let clk = self.device_attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?;
        let bus =
            self.device_attr(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
        if clk <= 0 || bus <= 0 {
            return None;
        }
        Some(2.0 * (clk as f64 * 1e3) * (bus as f64 / 8.0) / 1e9)
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

/// Grid for the memory-bound copy: one thread per **four** float4s, matching the kernel's 4× ILP (see
/// `ptx::COPY_V4`). Fewer threads, but each keeps four independent loads in flight — the memory-level
/// parallelism that saturates HBM on a 1:1 copy. The grid-stride loop makes correctness independent of
/// this exact count, so the only effect of the `/4` is to route the bulk of the work through the
/// 4-wide fast path (a one-float4-per-thread grid measured ~84% of peak; 4-wide clears 90%).
fn stream_cfg(_g: &Gpu, n4: u32) -> LaunchConfig {
    LaunchConfig::for_num_elems(n4.div_ceil(4))
}

/// `out := x`, a pure streaming copy on the GPU — the canonical HBM-bandwidth kernel (see
/// `ptx::COPY_V4`). Exact bitwise copy. `x.len()` must be a multiple of 4 (128-bit vectorized access).
pub fn copy(g: &mut Gpu, x: &[f32]) -> Result<Vec<f32>, DriverError> {
    assert_eq!(x.len() % 4, 0, "copy: len must be a multiple of 4 (v4 access)");
    let n4 = (x.len() / 4) as u32;
    let f = g.function("copy_v4", crate::ptx::COPY_V4, "copy_v4")?;
    let x_d = g.stream.memcpy_stod(x)?;
    let mut out_d = g.stream.memcpy_stod(&vec![0f32; x.len()])?;
    let cfg = stream_cfg(g, n4);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&n4).arg(&x_d).arg(&mut out_d);
    unsafe { b.launch(cfg)? };
    g.stream.memcpy_dtov(&out_d)
}

/// Narrow `x` (f32) to f16 on the GPU and widen back to f32 — round-trips through the `cast_f32_f16`
/// kernel (round-to-nearest-even, matching `half::f16::from_f32` bit-for-bit). This is the glue a
/// resident fp16 pipeline needs at an f32→f16 stage boundary (e.g. RMSNorm's f32 output feeding an
/// f16-input WMMA GEMM, with no host round-trip — see the resident FFN). Gate-able as a round-trip.
pub fn cast_f32_to_f16(g: &mut Gpu, x: &[f32]) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    let f = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
    let x_d = g.stream.memcpy_stod(x)?;
    let mut out_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.0); x.len()])?;
    let n = x.len() as u32;
    let mut b = g.stream.launch_builder(&f);
    b.arg(&n).arg(&x_d).arg(&mut out_d);
    unsafe { b.launch(LaunchConfig::for_num_elems(n))? };
    let hd: Vec<f16> = g.stream.memcpy_dtov(&out_d)?;
    Ok(hd.iter().map(|h| h.to_f32()).collect())
}

/// Apply an elementwise activation `out[i] = f(x[i])` on the GPU — the GPU twin of
/// `mercury_vmath_f32`. `op` is a `VM_*` code (exp/relu/sigmoid/tanh/silu/gelu so far). Transcendental
/// ops use SFU approximations, so the result is tolerance-close to the CPU kernel, not bit-exact.
pub fn vmath(g: &mut Gpu, op: i64, x: &[f32]) -> Result<Vec<f32>, DriverError> {
    let entry = vmath_entry(op);
    let n = x.len() as u32;
    let f = g.function("vmath", crate::ptx::vmath_ptx(), entry)?;
    let x_d = g.stream.memcpy_stod(x)?;
    let mut out_d = g.stream.memcpy_stod(&vec![0f32; x.len()])?;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&n).arg(&x_d).arg(&mut out_d);
    unsafe { b.launch(cfg)? };
    g.stream.memcpy_dtov(&out_d)
}

fn vmath_entry(op: i64) -> &'static str {
    use mercury_runtime::{VM_EXP, VM_GELU, VM_RELU, VM_SIGMOID, VM_SILU, VM_TANH};
    match op {
        x if x == VM_RELU => "relu",
        x if x == VM_EXP => "exp",
        x if x == VM_SIGMOID => "sigmoid",
        x if x == VM_TANH => "tanh",
        x if x == VM_SILU => "silu",
        x if x == VM_GELU => "gelu",
        _ => panic!("vmath op {op} not implemented on GPU yet"),
    }
}

/// Whether [`vmath`] has a GPU kernel for activation op `op`. The `--backend=gpu` offload checks this
/// and falls back to the CPU kernel for activations not yet on the GPU (instead of panicking).
pub fn vmath_supported(op: i64) -> bool {
    use mercury_runtime::{VM_EXP, VM_GELU, VM_RELU, VM_SIGMOID, VM_SILU, VM_TANH};
    op == VM_RELU
        || op == VM_EXP
        || op == VM_SIGMOID
        || op == VM_TANH
        || op == VM_SILU
        || op == VM_GELU
}

/// Fixed reduction grid — see `ptx::REDUCE`. The grid is independent of input size and occupancy, so
/// the GPU result is identical run-to-run (determinism by fixed decomposition, not associativity).
pub const RED_GRID: u32 = 256;
pub const RED_BLOCK: u32 = 256;
/// `max(x[i])` reduction op-code (mirrors `mercury_runtime::reduce::RED_MAX`, which isn't re-exported).
pub const RED_MAX: i64 = 4;

/// Deterministic GPU reduction — the GPU twin of `mercury_sreduce_f32`. `op` is `RED_SUM` /
/// `RED_DOT` (needs `y`) / [`RED_MAX`]. Blocks tree-reduce in shared memory; the `RED_GRID` partials
/// are combined on the host in ascending block order.
pub fn reduce(g: &mut Gpu, op: i64, x: &[f32], y: Option<&[f32]>) -> Result<f32, DriverError> {
    use mercury_runtime::{RED_DOT, RED_SUM};
    let entry = match op {
        v if v == RED_SUM => "reduce_sum",
        v if v == RED_DOT => "reduce_dot",
        v if v == RED_MAX => "reduce_max",
        _ => panic!("reduce op {op} not implemented on GPU yet"),
    };
    let n = x.len() as u32;
    let f = g.function("reduce", crate::ptx::REDUCE, entry)?;
    let x_d = g.stream.memcpy_stod(x)?;
    let y_d = match y {
        Some(y) => Some(g.stream.memcpy_stod(y)?),
        None => None,
    };
    let mut partials_d = g.stream.memcpy_stod(&vec![0f32; RED_GRID as usize])?;
    let cfg = LaunchConfig {
        grid_dim: (RED_GRID, 1, 1),
        block_dim: (RED_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = g.stream.launch_builder(&f);
    b.arg(&n).arg(&x_d);
    if let Some(ref yd) = y_d {
        b.arg(yd);
    }
    b.arg(&mut partials_d);
    unsafe { b.launch(cfg)? };
    let partials = g.stream.memcpy_dtov(&partials_d)?;
    let is_max = op == RED_MAX;
    let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
    for &p in &partials {
        acc = if is_max { acc.max(p) } else { acc + p };
    }
    Ok(acc)
}

/// Whether [`reduce`] has a GPU kernel for reduction op `op` (sum/dot/max so far). The
/// `--backend=gpu` offload checks this and falls back to the CPU kernel for the rest.
pub fn reduce_supported(op: i64) -> bool {
    use mercury_runtime::{RED_DOT, RED_SUM};
    op == RED_SUM || op == RED_DOT || op == RED_MAX
}

/// Whether reduction op `op` consumes the second operand `y` (only the dot product does).
pub fn reduce_needs_y(op: i64) -> bool {
    op == mercury_runtime::RED_DOT
}

/// Launch config for the 16×16-tiled GEMM: one 16×16 thread block per 16×16 C tile.
fn gemm_cfg(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

/// `C = A·Bᵀ` on the GPU (the nn.Linear spelling): `A` is `m×k`, `B` is `n×k`, `C` is `m×n`. GPU
/// twin of `mercury_sgemm_nt`. Tolerance-gated (the GPU reduces K in a different order).
pub fn gemm_nt(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(a.len(), m * k, "A must be m×k");
    assert_eq!(b.len(), n * k, "B must be n×k (A·Bᵀ)");
    let f = g.function("gemm", crate::ptx::GEMM, "gemm_nt")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(gemm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// Launch config for the register-blocked GEMM: a 16×16 block computes a 64×64 C tile.
fn gemm_rb_cfg(m: usize, n: usize) -> LaunchConfig {
    use crate::ptx_gemm::{TILE_M, TILE_N};
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(TILE_N), (m as u32).div_ceil(TILE_M), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

/// Register-blocked `C = A·Bᵀ` (nn.Linear): the faster GEMM (64×64 tile, 4×4 per thread). Same
/// math/contract as [`gemm_nt`]; tolerance-gated.
pub fn gemm_nt_rb(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let f = g.function("gemm_rb", crate::ptx_gemm::gemm_rb_ptx(), "gemm_nt_rb")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(gemm_rb_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·B` on the GPU: `A` is `m×k`, `B` is `k×n`, `C` is `m×n`. GPU twin of `mercury_sgemm`.
pub fn gemm_nn(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(a.len(), m * k, "A must be m×k");
    assert_eq!(b.len(), k * n, "B must be k×n (A·B)");
    let f = g.function("gemm", crate::ptx::GEMM, "gemm_nn")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(gemm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// Pick the WMMA entry + launch config: the fragment-reuse multi-tile kernel (`<base>_mt`, one warp
/// per WARP_M×WARP_N block) when M and N are multiples of the warp tile, else the single-16×16-tile
/// kernel (`<base>`, any 16-multiple). One warp (32 threads) per block either way.
fn wmma_pick(base: &str, m: usize, n: usize) -> (String, LaunchConfig) {
    use crate::ptx_wmma::{WARP_M, WARP_N};
    let block_dim = (32, 1, 1);
    if m % WARP_M == 0 && n % WARP_N == 0 {
        (
            format!("{base}_mt"),
            LaunchConfig {
                grid_dim: ((n / WARP_N) as u32, (m / WARP_M) as u32, 1),
                block_dim,
                shared_mem_bytes: 0,
            },
        )
    } else {
        (
            base.to_string(),
            LaunchConfig {
                grid_dim: (n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1),
                block_dim,
                shared_mem_bytes: 0,
            },
        )
    }
}

/// Tensor-core `C = A·Bᵀ` in **fp16 inputs with f32 accumulate** (the mixed-precision contract).
/// `A` (m×k) and `B` (n×k) arrive as f32 and are rounded to f16 on the host; `C` is f32. Requires
/// m, n, k to be multiples of 16. The FLOP/s headline — bf16/fp16 on the Ada tensor cores.
pub fn gemm_nt_f16(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM128_BM, SM128_BN, SM_BM, SM_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % 16 == 0 && n % 16 == 0 && k % 16 == 0,
        "WMMA requires 16-multiple dims"
    );
    // Regime-aware dispatch among the SMEM-staged kernels (all identical numerics; chosen by same-run,
    // best-of-N measurement on the RTX 4050). The binding constraint flips with the A+B working set vs
    // L2 (~24 MB on this part):
    //   • L2-resident (≤1024³, ~4 MB): cp.async pipelining wins — the 64×64 double-buffered kernel
    //     reaches ~99% of cuBLAS at 1024³.
    //   • Spills L2 (≳24 MB, e.g. 4096³ ≈ 67 MB): the GEMM goes occupancy-bound and the pipeline's
    //     extra SMEM + bar.syncs make it LOSE — the single-buffered 128 tile (big tile halves redundant
    //     inter-CTA traffic, no pipeline overhead) is best (4096³: 35% of cuBLAS vs db's 28%, same-run).
    //   • In between (~2048³, L2-fitting): the 128 double-buffered pipeline still edges ahead.
    let ws_bytes = (m * k + n * k) * 2; // fp16 A+B working set (bytes)
    if m <= 1024 && n <= 1024 && m % SM_BM == 0 && n % SM_BN == 0 {
        return gemm_nt_f16_sm_db(g, a, b, m, k, n);
    }
    if ws_bytes >= 24 * 1024 * 1024 && m % SM128_BM == 0 && n % SM128_BN == 0 {
        return gemm_nt_f16_sm128(g, a, b, m, k, n);
    }
    if m >= 2048 && n >= 2048 && m % SM128_BM == 0 && n % SM128_BN == 0 {
        return gemm_nt_f16_sm128_db(g, a, b, m, k, n);
    }
    if m % SM_BM == 0 && n % SM_BN == 0 {
        return gemm_nt_f16_sm(g, a, b, m, k, n);
    }
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let (entry, cfg) = wmma_pick("wmma_nt_f16", m, n);
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), &entry)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// Launch config for the shared-memory-staged WMMA kernels (`*_sm`): `SM_THREADS` threads per CTA,
/// one CTA per `SM_BM×SM_BN` output tile. Applicable when `M%SM_BM==0 && N%SM_BN==0 && K%16==0`.
fn wmma_sm_cfg(m: usize, n: usize) -> LaunchConfig {
    use crate::ptx_wmma::{SM_BM, SM_BN, SM_THREADS};
    LaunchConfig {
        grid_dim: ((n / SM_BN) as u32, (m / SM_BM) as u32, 1),
        block_dim: (SM_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `C = A·Bᵀ` (fp16-in, f32-out) via the **shared-memory-staged** kernel `wmma_nt_f16_sm` — a CTA of
/// warps cooperatively stages A/B tiles into shared memory and reuses them, instead of each warp
/// re-streaming overlapping rows/cols from global (the `_mt` path). Requires `M%SM_BM==0`,
/// `N%SM_BN==0`, `K%16==0`. The lever for the 4096³ cliff; tolerance-gated like the other GEMMs.
pub fn gemm_nt_f16_sm(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "wmma_nt_f16_sm requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·Bᵀ` (fp16-in, f32-out) via the **single-buffered 128×128** SMEM-staged kernel
/// `wmma_nt_f16_sm128` — the big-tile, no-cp.async large-GEMM path. The clean scoreboard shows the
/// double-buffered kernels are occupancy-bound and lose to the un-pipelined ones once A/B spill L2;
/// the 128 tile halves redundant inter-CTA traffic (vs the 64 `_sm`) while single-buffering keeps the
/// SMEM footprint and bar.sync count low. Requires `M%128==0`, `N%128==0`, `K%16==0`; tolerance-gated.
pub fn gemm_nt_f16_sm128(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM128_BM, SM128_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % SM128_BM == 0 && n % SM128_BN == 0 && k % 16 == 0,
        "wmma_nt_f16_sm128 requires M%{SM128_BM}==0, N%{SM128_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm128")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(wmma_sm128_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// Launch config for the 128×128 SMEM-staged kernel (`wmma_nt_f16_sm128_db`): `SM128_THREADS` threads
/// per CTA, one CTA per `SM128_BM×SM128_BN` output tile. Applies when `M%128==0 && N%128==0 && K%16==0`.
fn wmma_sm128_cfg(m: usize, n: usize) -> LaunchConfig {
    use crate::ptx_wmma::{SM128_BM, SM128_BN, SM128_THREADS};
    LaunchConfig {
        grid_dim: ((n / SM128_BN) as u32, (m / SM128_BM) as u32, 1),
        block_dim: (SM128_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `C = A·Bᵀ` (fp16-in, f32-out) via the **`cp.async` double-buffered** SMEM-staged kernel
/// `wmma_nt_f16_sm_db` — same 64×64 CTA tile as [`gemm_nt_f16_sm`], but the K-loop prefetches the next
/// A/B tile into the alternate shared buffer while the tensor cores consume the current one, hiding
/// global-load latency. The lever for the large-GEMM cliff (latency-, not bandwidth-volume-bound).
/// Requires `M%SM_BM==0`, `N%SM_BN==0`, `K%16==0`; tolerance-gated like the other GEMMs.
pub fn gemm_nt_f16_sm_db(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "wmma_nt_f16_sm_db requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm_db")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = act(A·Bᵀ)` (fp16-in, f32-out) in **one kernel** via a `wmma_nt_f16_sm_db_*` entry — the cp.async
/// double-buffered GEMM with the activation fused into the C-store epilogue. This is the lever cuBLAS
/// cannot match: it only computes `A·Bᵀ`, so a cuBLAS pipeline must launch a *second* kernel that reads
/// C back from HBM, applies the activation, and writes it again. The fused kernel writes C exactly once.
/// Requires `M%SM_BM==0`, `N%SM_BN==0`, `K%16==0`; tolerance-gated against `act(A·Bᵀ)`.
fn gemm_nt_f16_fused(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "{entry} requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), entry)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = relu(A·Bᵀ)` fused in one kernel (see [`gemm_nt_f16_fused`]).
pub fn gemm_nt_f16_sm_db_relu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused(g, a, b, m, k, n, "wmma_nt_f16_sm_db_relu")
}

/// `C = silu(A·Bᵀ)` fused in one kernel — the SwiGLU/SiLU FFN up-projection (see [`gemm_nt_f16_fused`]).
pub fn gemm_nt_f16_sm_db_silu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused(g, a, b, m, k, n, "wmma_nt_f16_sm_db_silu")
}

/// `C = gelu(A·Bᵀ)` fused in one kernel — the GELU FFN/MLP activation (see [`gemm_nt_f16_fused`]).
pub fn gemm_nt_f16_sm_db_gelu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused(g, a, b, m, k, n, "wmma_nt_f16_sm_db_gelu")
}

/// `C = A·Bᵀ + residual` (fp16-in, f32-out; `residual[M·N]` is f32) in **one kernel** via
/// `wmma_nt_f16_sm_db_residual` — the transformer skip connection fused into the GEMM. The kernel seeds
/// each accumulator with `residual` (wmma.load.c, the inverse of the store-d fragment layout) and the
/// K-loop adds A·Bᵀ on top, so the residual add is free at f32 accumulate and never round-trips through
/// HBM. The library call-chain instead pays a separate add kernel (or a beta=1 C pre-fill) for it — an
/// M13 megakernel building block. Requires `M%SM_BM==0`, `N%SM_BN==0`, `K%16==0`; tolerance-gated.
pub fn gemm_nt_f16_sm_db_residual(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(residual.len(), m * n, "residual must be M×N");
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "wmma_nt_f16_sm_db_residual requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f =
        g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm_db_residual")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let resid_d = g.stream.memcpy_stod(residual)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&resid_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = act(A·Bᵀ + bias)` (fp16-in, f32-out; `bias[N]` is f32) in **one kernel** via a
/// `wmma_nt_f16_sm_db_bias*` entry — the canonical `nn.Linear`/FFN epilogue. A per-column bias needs the
/// WMMA fragment's column index, which is opaque, so the kernel stores each tile to SMEM and re-reads it
/// by explicit (row,col) to add `bias` (then the activation) before writing C. cuBLAS needs a *second*
/// kernel to apply bias+activation (round-tripping C through HBM); this folds both into the GEMM store.
/// Requires `M%SM_BM==0`, `N%SM_BN==0`, `K%16==0`; tolerance-gated against `act(A·Bᵀ + bias)`.
fn gemm_nt_f16_fused_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "{entry} requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), entry)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d)
        .arg(&bias_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·Bᵀ + bias` fused (affine Linear, no activation) — see [`gemm_nt_f16_fused_bias`].
pub fn gemm_nt_f16_sm_db_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_f16_sm_db_bias")
}

/// `C = relu(A·Bᵀ + bias)` fused — the canonical Linear+ReLU (see [`gemm_nt_f16_fused_bias`]).
pub fn gemm_nt_f16_sm_db_bias_relu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_f16_sm_db_bias_relu")
}

/// `C = silu(A·Bᵀ + bias)` fused — SiLU/SwiGLU up-projection with bias (see [`gemm_nt_f16_fused_bias`]).
pub fn gemm_nt_f16_sm_db_bias_silu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_f16_sm_db_bias_silu")
}

/// `C = gelu(A·Bᵀ + bias)` fused — the canonical transformer FFN first layer (see [`gemm_nt_f16_fused_bias`]).
pub fn gemm_nt_f16_sm_db_bias_gelu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_f16_sm_db_bias_gelu")
}

/// `C = A·Bᵀ` (fp16-in, f32-out) via the **128×128 CTA-tile + `cp.async` double-buffered** kernel
/// `wmma_nt_f16_sm128_db` — the cuBLAS recipe: a big tile cuts redundant inter-CTA global traffic
/// *and* software pipelining hides what's left, the combination aimed at the large-GEMM regime where
/// neither lever alone sufficed. Requires `M%128==0`, `N%128==0`, `K%16==0`; tolerance-gated.
pub fn gemm_nt_f16_sm128_db(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM128_BM, SM128_BN};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % SM128_BM == 0 && n % SM128_BN == 0 && k % 16 == 0,
        "wmma_nt_f16_sm128_db requires M%{SM128_BM}==0, N%{SM128_BN}==0, K%16==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm128_db")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(wmma_sm128_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// Tensor-core `C = A·Bᵀ` in **bf16 inputs with f32 accumulate**. Same contract as [`gemm_nt_f16`]
/// but bf16 (wider range, fewer mantissa bits) — the precision modern transformers train in.
pub fn gemm_nt_bf16(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use half::bf16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % 16 == 0 && n % 16 == 0 && k % 16 == 0,
        "WMMA requires 16-multiple dims"
    );
    let a16: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
    let b16: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
    let (entry, cfg) = wmma_pick("wmma_nt_bf16", m, n);
    let f = g.function("wmma_bf16", crate::ptx_wmma::wmma_bf16_ptx(), &entry)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = act(A·Bᵀ)` in **bf16 inputs / f32 accumulate**, fused in one cp.async-pipelined WMMA kernel —
/// the bf16 twin of [`gemm_nt_f16_fused`], for the precision transformers train in. `entry` is a
/// `wmma_nt_bf16_sm_db_*` name. Requires `M%SM_BM==0`, `N%SM_BN==0`, `K%16==0`; tolerance-gated.
fn gemm_nt_bf16_fused(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::bf16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "{entry} requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
    let b16: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
    let f = g.function("wmma_bf16", crate::ptx_wmma::wmma_bf16_ptx(), entry)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = relu(A·Bᵀ)` fused, bf16 inputs (see [`gemm_nt_bf16_fused`]).
pub fn gemm_nt_bf16_sm_db_relu(g: &mut Gpu, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused(g, a, b, m, k, n, "wmma_nt_bf16_sm_db_relu")
}
/// `C = silu(A·Bᵀ)` fused, bf16 inputs — the SwiGLU FFN up-projection (see [`gemm_nt_bf16_fused`]).
pub fn gemm_nt_bf16_sm_db_silu(g: &mut Gpu, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused(g, a, b, m, k, n, "wmma_nt_bf16_sm_db_silu")
}
/// `C = gelu(A·Bᵀ)` fused, bf16 inputs (see [`gemm_nt_bf16_fused`]).
pub fn gemm_nt_bf16_sm_db_gelu(g: &mut Gpu, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused(g, a, b, m, k, n, "wmma_nt_bf16_sm_db_gelu")
}

/// `C = act(A·Bᵀ + bias)` in **bf16 inputs / f32 accumulate**, fused — the bf16 twin of
/// [`gemm_nt_f16_fused_bias`] (the bias epilogue acts on the f32 accumulator, so it is identical across
/// dtypes; only the input quantization differs). `entry` is a `wmma_nt_bf16_sm_db_bias*` name.
fn gemm_nt_bf16_fused_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM_BM, SM_BN};
    use half::bf16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert!(
        m % SM_BM == 0 && n % SM_BN == 0 && k % 16 == 0,
        "{entry} requires M%{SM_BM}==0, N%{SM_BN}==0, K%16==0"
    );
    let a16: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
    let b16: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
    let f = g.function("wmma_bf16", crate::ptx_wmma::wmma_bf16_ptx(), entry)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d);
    unsafe { bld.launch(wmma_sm_cfg(m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·Bᵀ + bias` fused, bf16 inputs (affine Linear) — see [`gemm_nt_bf16_fused_bias`].
pub fn gemm_nt_bf16_sm_db_bias(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_bf16_sm_db_bias")
}
/// `C = relu(A·Bᵀ + bias)` fused, bf16 inputs (see [`gemm_nt_bf16_fused_bias`]).
pub fn gemm_nt_bf16_sm_db_bias_relu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_bf16_sm_db_bias_relu")
}
/// `C = silu(A·Bᵀ + bias)` fused, bf16 inputs (see [`gemm_nt_bf16_fused_bias`]).
pub fn gemm_nt_bf16_sm_db_bias_silu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_bf16_sm_db_bias_silu")
}
/// `C = gelu(A·Bᵀ + bias)` fused, bf16 inputs (see [`gemm_nt_bf16_fused_bias`]).
pub fn gemm_nt_bf16_sm_db_bias_gelu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_fused_bias(g, a, b, bias, m, k, n, "wmma_nt_bf16_sm_db_bias_gelu")
}

/// Measure the **fp16 tensor-core roofline** in FLOP/s: `warps` warps each issue `iters·ROOFLINE_ACC`
/// `wmma.mma`s on register-resident fragments (one global load, no hot-loop memory traffic), so the
/// achieved rate is the practical TC ceiling on this GPU. Best-of-`reps` to ride out clock dips. This
/// is an *internal* ceiling; the real peer is cuBLAS (`baselines.rs` / `gemm_vs_peers`), which matches
/// or exceeds it — so it is a soft under-estimate. See [`crate::ptx_wmma::roofline_entry`].
pub fn wmma_roofline_f16(
    g: &mut Gpu,
    iters: u32,
    warps: u32,
    reps: usize,
) -> Result<f64, DriverError> {
    use half::f16;
    let f = g.function(
        "wmma_roofline_f16",
        crate::ptx_wmma::roofline_f16_ptx(),
        "wmma_roofline_f16",
    )?;
    let a: Vec<f16> = vec![f16::from_f32(0.01); 256];
    let b: Vec<f16> = vec![f16::from_f32(0.01); 256];
    let a_d = g.stream.memcpy_stod(&a)?;
    let b_d = g.stream.memcpy_stod(&b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; warps as usize * 256])?;
    let cfg = LaunchConfig {
        grid_dim: (warps, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch = |g: &Gpu, c_d: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), DriverError> {
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&iters).arg(&a_d).arg(&b_d).arg(c_d);
        unsafe { bld.launch(cfg) }.map(|_| ())
    };
    launch(g, &mut c_d)?; // warm up (JIT + clocks)
    g.stream.synchronize()?;
    let flop = warps as f64
        * iters as f64
        * crate::ptx_wmma::ROOFLINE_ACC as f64
        * (16.0 * 16.0 * 16.0 * 2.0);
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t0 = std::time::Instant::now();
        launch(g, &mut c_d)?;
        g.stream.synchronize()?;
        best = best.min(t0.elapsed().as_secs_f64());
    }
    Ok(flop / best)
}

/// Fused row-wise normalization on the GPU — the GPU twin of `mercury_norm_f32`. `op` is a `NORM_*`
/// code (softmax / layernorm / rmsnorm); `x` is `rows×cols` row-major. One warp per row; the row
/// reductions are warp-butterfly all-reduces (deterministic order). Tolerance-gated.
pub fn norm(
    g: &mut Gpu,
    op: i64,
    x: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>, DriverError> {
    use mercury_runtime::{NORM_LAYERNORM, NORM_RMSNORM, NORM_SOFTMAX};
    assert_eq!(x.len(), rows * cols);
    let entry = match op {
        v if v == NORM_SOFTMAX => "softmax",
        v if v == NORM_LAYERNORM => "layernorm",
        v if v == NORM_RMSNORM => "rmsnorm",
        _ => panic!("norm op {op} not implemented on GPU yet"),
    };
    let f = g.function("norm", crate::ptx_norm::norm_ptx(), entry)?;
    let x_d = g.stream.memcpy_stod(x)?;
    let mut out_d = g.stream.memcpy_stod(&vec![0f32; x.len()])?;
    let (r, c) = (rows as u32, cols as u32);
    let cfg = LaunchConfig {
        grid_dim: (r, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&r).arg(&c).arg(&eps).arg(&x_d).arg(&mut out_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&out_d)
}

/// Fused **flash-attention** on the GPU: `O = softmax(scale · Q·Kᵀ) · V`, single head, Q/K/V/O all
/// `[seq, d]` row-major. Never materializes the `seq×seq` score matrix — the online-softmax
/// recurrence streams K/V once (one warp per query row). `d` must be one of [`ptx_flash::SUPPORTED_D`]
/// (32/64/128). Tolerance-gated against a full-softmax CPU reference. The marquee GPU kernel: the
/// fused form that *lost* on CPU (where the tuned GEMM dominates) wins here by halving HBM traffic.
pub fn flash_attn(
    g: &mut Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(q.len(), seq * d, "Q must be seq×d");
    assert_eq!(k.len(), seq * d, "K must be seq×d");
    assert_eq!(v.len(), seq * d, "V must be seq×d");
    assert!(
        crate::ptx_flash::SUPPORTED_D.contains(&d),
        "flash_attn: head dim {d} has no generated kernel (supported: {:?})",
        crate::ptx_flash::SUPPORTED_D
    );
    let entry = format!("flash_d{d}");
    let f = g.function("flash", crate::ptx_flash::flash_ptx(), &entry)?;
    let q_d = g.stream.memcpy_stod(q)?;
    let k_d = g.stream.memcpy_stod(k)?;
    let v_d = g.stream.memcpy_stod(v)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; seq * d])?;
    let s = seq as u32;
    let cfg = LaunchConfig {
        grid_dim: (s, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&s)
        .arg(&scale)
        .arg(&q_d)
        .arg(&k_d)
        .arg(&v_d)
        .arg(&mut o_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&o_d)
}

/// Direct 2D **convolution** on the GPU (single batch, stride 1, no padding): input `x` is `[C,H,W]`,
/// weights `w` are `[K,C,R,S]`, output is `[K,P,Q]` with `P=H-R+1`, `Q=W-S+1` — the valid
/// cross-correlation deep-learning calls conv2d. One thread per output element. Tolerance-gated
/// (the GPU `fma`-accumulates the C·R·S window in a different order than a serial reference).
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    g: &mut Gpu,
    x: &[f32],
    w: &[f32],
    c: usize,
    h: usize,
    width: usize,
    k: usize,
    r: usize,
    s: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(x.len(), c * h * width, "X must be C×H×W");
    assert_eq!(w.len(), k * c * r * s, "W must be K×C×R×S");
    assert!(h >= r && width >= s, "kernel larger than input");
    let p = h - r + 1;
    let q = width - s + 1;
    let total = (k * p * q) as u32;
    let f = g.function("conv2d", crate::ptx_conv::CONV2D, "conv2d")?;
    let x_d = g.stream.memcpy_stod(x)?;
    let w_d = g.stream.memcpy_stod(w)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; k * p * q])?;
    let dims = [c, h, width, k, r, s, p, q].map(|v| v as u32);
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    for d in &dims {
        bld.arg(d);
    }
    bld.arg(&x_d).arg(&w_d).arg(&mut o_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&o_d)
}

/// **Fused FFN block** `out = x + SiLU(RMSNorm(x)·W1ᵀ)·W2ᵀ` (the SiLU MLP), GPU-**resident**: `x` and
/// the weights upload once, every op runs on device buffers, only the `[S,D]` output copies back. The
/// two epilogues a GEMM library cannot fuse are folded into the projections — the SiLU into the
/// up-projection's WMMA store, the residual into the down-projection's accumulator (`wmma.load.c`):
///
/// ```text
///   h2  = RMSNorm(x)                    f32 [S,D]     ptx_norm rmsnorm
///   h2' = (f16) h2                      cast          cast_f32_f16   (f32→f16 stage boundary)
///   f1a = SiLU(h2'·W1ᵀ)                 f32 [S,Dff]   wmma_nt_f16_sm_db_silu   (GEMM+act fused)
///   f1a'= (f16) f1a                     cast
///   out = (f1a'·W2ᵀ) + x                f32 [S,D]     wmma_nt_f16_sm_db_residual (GEMM+residual fused)
/// ```
///
/// A library FFN is `norm, GEMM, SiLU, GEMM, add` — 5 launches with the SiLU and residual each a kernel
/// round-tripping `[S,Dff]`/`[S,D]` through HBM. This folds both into the GEMMs (the casts are cheap
/// memory passes). `W1` is `[Dff,D]`, `W2` is `[D,Dff]`; requires `S%64==0, D%64==0, Dff%64==0` (the
/// WMMA-staged tiles). fp16-GEMM precision ⇒ tolerance-gated against the f64 reference of the same FFN.
pub fn ffn_fused(
    g: &mut Gpu,
    x: &[f32],
    w1: &[f32],
    w2: &[f32],
    s: usize,
    d: usize,
    dff: usize,
) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    assert_eq!(x.len(), s * d, "x must be S×D");
    assert_eq!(w1.len(), dff * d, "w1 must be Dff×D");
    assert_eq!(w2.len(), d * dff, "w2 must be D×Dff");
    assert!(
        s % 64 == 0 && d % 64 == 0 && dff % 64 == 0,
        "ffn_fused needs S,D,Dff multiples of 64 (WMMA-staged tiles)"
    );
    // Preload every kernel once (the only &mut g use); afterwards device work runs through the cloned
    // stream Arc with no further borrow of g — same pattern as transformer_layer.
    let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
    let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
    let ptx = crate::ptx_wmma::wmma_f16_ptx();
    let f_silu = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_silu")?;
    let f_resid = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_residual")?;
    let stream = g.stream.clone();

    let x_d = stream.memcpy_stod(x)?;
    let w1_16: Vec<f16> = w1.iter().map(|&v| f16::from_f32(v)).collect();
    let w2_16: Vec<f16> = w2.iter().map(|&v| f16::from_f32(v)).collect();
    let w1_d = stream.memcpy_stod(&w1_16)?;
    let w2_d = stream.memcpy_stod(&w2_16)?;
    let eps = 1e-5f32;

    // device f32 → device f16 narrowing (the stage-boundary cast).
    let cast = |src: &cudarc::driver::CudaSlice<f32>, n: usize| -> Result<_, DriverError> {
        let mut dst = stream.memcpy_stod(&vec![f16::from_f32(0.0); n])?;
        let nn = n as u32;
        let mut b = stream.launch_builder(&f_cast);
        b.arg(&nn).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    };

    // h2 = RMSNorm(x)  (one warp per row).
    let mut h2 = stream.memcpy_stod(&vec![0f32; s * d])?;
    {
        let (r, c) = (s as u32, d as u32);
        let mut b = stream.launch_builder(&f_norm);
        b.arg(&r).arg(&c).arg(&eps).arg(&x_d).arg(&mut h2);
        let cfg = LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg)? };
    }
    let h2_16 = cast(&h2, s * d)?;

    // f1a = SiLU(h2·W1ᵀ):  A = h2 [S,D], B = W1 [Dff,D]  ⇒ C [S,Dff].
    let mut f1a = stream.memcpy_stod(&vec![0f32; s * dff])?;
    {
        let (mm, nn, kk) = (s as u32, dff as u32, d as u32);
        let mut b = stream.launch_builder(&f_silu);
        b.arg(&mm).arg(&nn).arg(&kk).arg(&h2_16).arg(&w1_d).arg(&mut f1a);
        unsafe { b.launch(wmma_sm_cfg(s, dff))? };
    }
    let f1a_16 = cast(&f1a, s * dff)?;

    // out = (f1a·W2ᵀ) + x:  A = f1a [S,Dff], B = W2 [D,Dff] ⇒ C [S,D], residual = x.
    let mut out = stream.memcpy_stod(&vec![0f32; s * d])?;
    {
        let (mm, nn, kk) = (s as u32, d as u32, dff as u32);
        let mut b = stream.launch_builder(&f_resid);
        b.arg(&mm).arg(&nn).arg(&kk).arg(&f1a_16).arg(&w2_d).arg(&mut out).arg(&x_d);
        unsafe { b.launch(wmma_sm_cfg(s, d))? };
    }
    stream.memcpy_dtov(&out)
}

/// The learned weights of one transformer layer (all `A·Bᵀ` projections): attention `Wq/Wk/Wv/Wo`
/// are `[D, D]`, the FFN is up `W1` `[Dff, D]` then down `W2` `[D, Dff]`.
pub struct TransformerWeights<'a> {
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub w1: &'a [f32],
    pub w2: &'a [f32],
}

/// One **pre-norm transformer encoder layer, end-to-end GPU-resident**. The input `x` (`[S, D]`) and
/// all weights are uploaded to the device **once**; every op then runs on device buffers with **no
/// host round-trip between ops**, and only the final `[S, D]` output is copied back. The sequence
/// chains the kernels this crate already ships:
///
/// ```text
///   h1 = RMSNorm(x)                              (ptx_norm)
///   Q,K,V = h1·Wqᵀ, h1·Wkᵀ, h1·Wvᵀ              (register-blocked GEMM ×3)
///   A  = FlashAttention(Q, K, V, 1/√D)           (ptx_flash, fused — no S² scores in HBM)
///   x  = x + A·Woᵀ                               (GEMM + residual add)
///   h2 = RMSNorm(x)                              (ptx_norm)
///   x  = x + SiLU(h2·W1ᵀ)·W2ᵀ                    (GEMM, vmath SiLU, GEMM, residual add)
/// ```
///
/// Single head; `d` must be a flash-supported head dim (32/64/128). Tolerance-gated against a CPU f64
/// reference of the same layer. This is the "whole layer stays resident on the GPU" milestone — the
/// payoff of having every transformer op available as a device kernel.
pub fn transformer_layer(
    g: &mut Gpu,
    x: &[f32],
    w: &TransformerWeights,
    s: usize,
    d: usize,
    dff: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(x.len(), s * d, "x must be S×D");
    for (name, wt, len) in [
        ("wq", w.wq, d * d),
        ("wk", w.wk, d * d),
        ("wv", w.wv, d * d),
        ("wo", w.wo, d * d),
        ("w1", w.w1, dff * d),
        ("w2", w.w2, d * dff),
    ] {
        assert_eq!(wt.len(), len, "{name} wrong size");
    }
    assert!(
        crate::ptx_flash::SUPPORTED_D.contains(&d),
        "transformer_layer: head dim {d} unsupported by flash (need {:?})",
        crate::ptx_flash::SUPPORTED_D
    );

    // Preload every kernel once — the only `&mut g` use; the returned handles are owned, so after
    // this the device work runs through `stream` (a cloned `Arc`) with no further borrow of `g`.
    let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
    let f_gemm = g.function("gemm_rb", crate::ptx_gemm::gemm_rb_ptx(), "gemm_nt_rb")?;
    let f_flash = g.function(
        "flash",
        crate::ptx_flash::flash_ptx(),
        &format!("flash_d{d}"),
    )?;
    let f_vadd = g.function("vadd", crate::ptx::VADD, "vadd")?;
    let f_silu = g.function("vmath", crate::ptx::vmath_ptx(), "silu")?;
    let stream = g.stream.clone();

    // Upload input + weights once.
    let x_d = stream.memcpy_stod(x)?;
    let wq = stream.memcpy_stod(w.wq)?;
    let wk = stream.memcpy_stod(w.wk)?;
    let wv = stream.memcpy_stod(w.wv)?;
    let wo = stream.memcpy_stod(w.wo)?;
    let w1 = stream.memcpy_stod(w.w1)?;
    let w2 = stream.memcpy_stod(w.w2)?;

    let eps = 1e-5f32;
    let norm_cfg = LaunchConfig {
        grid_dim: (s as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // RMSNorm a `[rows, d]` device buffer into a fresh one.
    let norm = |src: &cudarc::driver::CudaSlice<f32>, rows: usize| -> Result<_, DriverError> {
        let mut out = stream.memcpy_stod(&vec![0f32; rows * d])?;
        let (r, c) = (rows as u32, d as u32);
        let mut bld = stream.launch_builder(&f_norm);
        bld.arg(&r).arg(&c).arg(&eps).arg(src).arg(&mut out);
        unsafe { bld.launch(norm_cfg)? };
        Ok(out)
    };
    // C = A·Bᵀ, A `[m,k]`, B `[n,k]` → C `[m,n]`, all device buffers.
    let gemm = |a: &cudarc::driver::CudaSlice<f32>,
                b: &cudarc::driver::CudaSlice<f32>,
                m: usize,
                k: usize,
                n: usize|
     -> Result<_, DriverError> {
        let mut c = stream.memcpy_stod(&vec![0f32; m * n])?;
        let (mm, nn, kk) = (m as u32, n as u32, k as u32);
        let mut bld = stream.launch_builder(&f_gemm);
        bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut c);
        unsafe { bld.launch(gemm_rb_cfg(m, n))? };
        Ok(c)
    };
    // out = a + b (residual), element count `n`.
    let add = |a: &cudarc::driver::CudaSlice<f32>,
               b: &cudarc::driver::CudaSlice<f32>,
               n: usize|
     -> Result<_, DriverError> {
        let mut out = stream.memcpy_stod(&vec![0f32; n])?;
        let nn = n as u32;
        let mut bld = stream.launch_builder(&f_vadd);
        bld.arg(&nn).arg(a).arg(b).arg(&mut out);
        unsafe { bld.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(out)
    };

    // --- attention block ---
    let h1 = norm(&x_d, s)?;
    let q = gemm(&h1, &wq, s, d, d)?;
    let k = gemm(&h1, &wk, s, d, d)?;
    let v = gemm(&h1, &wv, s, d, d)?;
    // fused flash-attention over the device Q/K/V (one warp per query row).
    let mut attn = stream.memcpy_stod(&vec![0f32; s * d])?;
    {
        let scale = 1.0f32 / (d as f32).sqrt();
        let ss = s as u32;
        let flash_cfg = LaunchConfig {
            grid_dim: (s as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bld = stream.launch_builder(&f_flash);
        bld.arg(&ss)
            .arg(&scale)
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&mut attn);
        unsafe { bld.launch(flash_cfg)? };
    }
    let o = gemm(&attn, &wo, s, d, d)?;
    let x = add(&x_d, &o, s * d)?; // residual 1

    // --- FFN block ---
    let h2 = norm(&x, s)?;
    let f1 = gemm(&h2, &w1, s, d, dff)?; // [S, Dff]
                                         // SiLU(f1) → a fresh device buffer (out[i]=f(x[i]) needs no aliasing, so no host round-trip).
    let mut f1act = stream.memcpy_stod(&vec![0f32; s * dff])?;
    {
        let n = (s * dff) as u32;
        let mut bld = stream.launch_builder(&f_silu);
        bld.arg(&n).arg(&f1).arg(&mut f1act);
        unsafe { bld.launch(LaunchConfig::for_num_elems(n))? };
    }
    let f2 = gemm(&f1act, &w2, s, dff, d)?; // [S, D]
    let x = add(&x, &f2, s * d)?; // residual 2

    stream.memcpy_dtov(&x)
}

/// One **pre-norm transformer layer on the fp16 tensor cores** — the fast, fused sibling of
/// [`transformer_layer`]. Same math (`x + Attn(RMSNorm(x))·Woᵀ`, then `x + SiLU(RMSNorm(x)·W1ᵀ)·W2ᵀ`),
/// but every `A·Bᵀ` projection runs on the WMMA tensor cores (fp16 in, f32 accumulate) instead of the
/// scalar register-blocked f32 GEMM, and the two epilogues a GEMM library *cannot* fuse are folded into
/// the projections that produce them:
///
/// ```text
///   h1   = RMSNorm(x)                      f32 [S,D]    ptx_norm
///   Q,K,V= (f16 h1)·{Wq,Wk,Wv}ᵀ           f32 [S,D]    wmma_nt_f16_sm_db          (tensor-core ×3)
///   A    = FlashAttention(Q,K,V,1/√D)      f32 [S,D]    ptx_flash (f32 online softmax)
///   x    = (f16 A)·Woᵀ + x                 f32 [S,D]    wmma_nt_f16_sm_db_residual (residual FUSED)
///   h2   = RMSNorm(x)                      f32 [S,D]    ptx_norm
///   f1   = SiLU((f16 h2)·W1ᵀ)              f32 [S,Dff]  wmma_nt_f16_sm_db_silu     (SiLU FUSED)
///   x    = (f16 f1)·W2ᵀ + x                f32 [S,D]    wmma_nt_f16_sm_db_residual (residual FUSED)
/// ```
///
/// The weights upload **once** (pre-narrowed to f16) and the whole layer stays GPU-resident — only the
/// `[S,D]` output copies back. The f32↔f16 stage boundaries (norm/flash run f32; the GEMMs want f16
/// fragments) are cheap `cast_f32_f16` memory passes. Flash stays f32 because its online softmax needs
/// the dynamic range. `Wq/Wk/Wv/Wo` are `[D,D]`, `W1` is `[Dff,D]`, `W2` is `[D,Dff]`; requires
/// `S%64==0, D%64==0, Dff%64==0` (the WMMA tile) and `D` a flash head dim (64/128). fp16-GEMM precision
/// ⇒ tolerance-gated against the f64 reference of the same layer with f16-rounded GEMM inputs.
pub struct ResidentLayerF16 {
    stream: Arc<CudaStream>,
    f_norm: CudaFunction,
    f_cast: CudaFunction,
    f_flash: CudaFunction,
    f_gemm: CudaFunction,
    f_silu: CudaFunction,
    f_resid: CudaFunction,
    /// Standalone SiLU (`vmath`) + residual-add (`vadd`) kernels — used only by the unfused reference
    /// path [`forward_device_unfused`](Self::forward_device_unfused), to measure what the fused
    /// epilogues buy. The fused [`forward_device`](Self::forward_device) never launches them.
    f_act: CudaFunction,
    f_vadd: CudaFunction,
    wq: cudarc::driver::CudaSlice<half::f16>,
    wk: cudarc::driver::CudaSlice<half::f16>,
    wv: cudarc::driver::CudaSlice<half::f16>,
    wo: cudarc::driver::CudaSlice<half::f16>,
    w1: cudarc::driver::CudaSlice<half::f16>,
    w2: cudarc::driver::CudaSlice<half::f16>,
    s: usize,
    d: usize,
    dff: usize,
    eps: f32,
}

impl ResidentLayerF16 {
    /// Upload the weights (narrowed to f16) and preload the kernels — the one-time, `&mut Gpu` setup.
    /// The three WMMA entries (`_sm_db`, `_sm_db_silu`, `_sm_db_residual`) share one JITed module.
    pub fn new(
        g: &mut Gpu,
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
    ) -> Result<Self, DriverError> {
        use half::f16;
        for (name, wt, len) in [
            ("wq", w.wq, d * d),
            ("wk", w.wk, d * d),
            ("wv", w.wv, d * d),
            ("wo", w.wo, d * d),
            ("w1", w.w1, dff * d),
            ("w2", w.w2, d * dff),
        ] {
            assert_eq!(wt.len(), len, "{name} wrong size");
        }
        assert!(
            s % 64 == 0 && d % 64 == 0 && dff % 64 == 0,
            "ResidentLayerF16 needs S,D,Dff multiples of 64 (WMMA-staged tiles)"
        );
        assert!(
            crate::ptx_flash::SUPPORTED_D.contains(&d),
            "ResidentLayerF16: head dim {d} unsupported by flash (need {:?})",
            crate::ptx_flash::SUPPORTED_D
        );
        let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
        let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
        let f_flash = g.function("flash", crate::ptx_flash::flash_ptx(), &format!("flash_d{d}"))?;
        let ptx = crate::ptx_wmma::wmma_f16_ptx();
        let f_gemm = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db")?;
        let f_silu = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_silu")?;
        let f_resid = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_residual")?;
        let f_act = g.function("vmath", crate::ptx::vmath_ptx(), "silu")?;
        let f_vadd = g.function("vadd", crate::ptx::VADD, "vadd")?;
        let stream = g.stream.clone();
        let to16 = |wt: &[f32]| -> Vec<f16> { wt.iter().map(|&v| f16::from_f32(v)).collect() };
        let wq = stream.memcpy_stod(&to16(w.wq))?;
        let wk = stream.memcpy_stod(&to16(w.wk))?;
        let wv = stream.memcpy_stod(&to16(w.wv))?;
        let wo = stream.memcpy_stod(&to16(w.wo))?;
        let w1 = stream.memcpy_stod(&to16(w.w1))?;
        let w2 = stream.memcpy_stod(&to16(w.w2))?;
        Ok(Self {
            stream,
            f_norm,
            f_cast,
            f_flash,
            f_gemm,
            f_silu,
            f_resid,
            f_act,
            f_vadd,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            s,
            d,
            dff,
            eps: 1e-5,
        })
    }

    /// Run the layer on a **resident** `[S,D]` f32 activation buffer, returning a fresh resident `[S,D]`
    /// f32 buffer — the pure on-device kernel chain, no host transfer. This is what stacks N-deep into a
    /// whole-model forward pass: one layer's output is the next layer's input, never leaving the GPU.
    pub fn forward_device(
        &self,
        x_d: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<cudarc::driver::CudaSlice<f32>, DriverError> {
        use half::f16;
        let stream = &self.stream;
        let (s, d, dff, eps) = (self.s, self.d, self.dff, self.eps);
        let norm_cfg = LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

        // RMSNorm a `[rows, d]` f32 buffer into a fresh f32 buffer (one warp per row).
        let norm = |src: &cudarc::driver::CudaSlice<f32>, rows: usize| -> Result<_, DriverError> {
            let mut out = stream.alloc_zeros::<f32>(rows * d)?;
            let (r, c) = (rows as u32, d as u32);
            let mut bld = stream.launch_builder(&self.f_norm);
            bld.arg(&r).arg(&c).arg(&eps).arg(src).arg(&mut out);
            unsafe { bld.launch(norm_cfg)? };
            Ok(out)
        };
        // device f32 -> device f16 narrowing (the stage boundary between norm/flash and the WMMA GEMMs).
        let cast = |src: &cudarc::driver::CudaSlice<f32>, n: usize| -> Result<cudarc::driver::CudaSlice<f16>, DriverError> {
            let mut dst = stream.alloc_zeros::<f16>(n)?;
            let nn = n as u32;
            let mut b = stream.launch_builder(&self.f_cast);
            b.arg(&nn).arg(src).arg(&mut dst);
            unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
            Ok(dst)
        };
        // C = A*Bᵀ (fp16-in, f32-out) via a chosen WMMA entry (`f_gemm` plain, or `f_silu` SiLU-fused).
        let gemm16 = |f: &cudarc::driver::CudaFunction,
                      a: &cudarc::driver::CudaSlice<f16>,
                      b: &cudarc::driver::CudaSlice<f16>,
                      m: usize,
                      k: usize,
                      n: usize|
         -> Result<_, DriverError> {
            let mut c = stream.alloc_zeros::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut c);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };
        // C = A*Bᵀ + residual (fp16-in; f32 residual seeded via wmma.load.c) — the skip connection fused
        // into the GEMM accumulator (no separate add kernel, no HBM round-trip).
        let resid_gemm = |a: &cudarc::driver::CudaSlice<f16>,
                          b: &cudarc::driver::CudaSlice<f16>,
                          residual: &cudarc::driver::CudaSlice<f32>,
                          m: usize,
                          k: usize,
                          n: usize|
         -> Result<_, DriverError> {
            let mut c = stream.alloc_zeros::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(&self.f_resid);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut c).arg(residual);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };

        // --- attention: fp16 tensor-core Q/K/V/O projections, f32 flash, residual fused into O ---
        let h1 = norm(x_d, s)?;
        let h1_16 = cast(&h1, s * d)?;
        let q = gemm16(&self.f_gemm, &h1_16, &self.wq, s, d, d)?;
        let k = gemm16(&self.f_gemm, &h1_16, &self.wk, s, d, d)?;
        let v = gemm16(&self.f_gemm, &h1_16, &self.wv, s, d, d)?;
        let mut attn = stream.alloc_zeros::<f32>(s * d)?;
        {
            let scale = 1.0f32 / (d as f32).sqrt();
            let ss = s as u32;
            let flash_cfg = LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            let mut bld = stream.launch_builder(&self.f_flash);
            bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut attn);
            unsafe { bld.launch(flash_cfg)? };
        }
        let attn_16 = cast(&attn, s * d)?;
        let x1 = resid_gemm(&attn_16, &self.wo, x_d, s, d, d)?; // x + A*Woᵀ  (residual 1, fused)

        // --- FFN: RMSNorm -> SiLU up-projection (fused) -> down-projection with residual (fused) ---
        let h2 = norm(&x1, s)?;
        let h2_16 = cast(&h2, s * d)?;
        let f1 = gemm16(&self.f_silu, &h2_16, &self.w1, s, d, dff)?; // SiLU(h2*W1ᵀ) [S,Dff], act fused
        let f1_16 = cast(&f1, s * dff)?;
        let out = resid_gemm(&f1_16, &self.w2, &x1, s, dff, d)?; // x1 + f1*W2ᵀ  (residual 2, fused)
        Ok(out)
    }

    /// **Unfused reference path** — the same layer with NO fused epilogues: a plain WMMA GEMM for every
    /// projection, then a SEPARATE `vadd` for each residual and a SEPARATE `silu` (`vmath`) for the
    /// activation. Numerically equal to [`forward_device`](Self::forward_device) up to f32-accumulation
    /// order (the SiLU input is an exact f32 store/reload; only the residual add order differs: `x+Σ` vs
    /// `Σ+x`), but it launches **3 extra kernels** (2 residual adds + 1 SiLU) that each round-trip a
    /// `[S,D]`/`[S,Dff]` tensor through HBM. Exists only to measure what the fused epilogues buy at the
    /// full-layer level (`layer_fusion_vs_unfused_throughput`) — it is not the production path.
    pub fn forward_device_unfused(
        &self,
        x_d: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<cudarc::driver::CudaSlice<f32>, DriverError> {
        use half::f16;
        let stream = &self.stream;
        let (s, d, dff, eps) = (self.s, self.d, self.dff, self.eps);
        let norm_cfg = LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

        let norm = |src: &cudarc::driver::CudaSlice<f32>, rows: usize| -> Result<_, DriverError> {
            let mut out = stream.alloc_zeros::<f32>(rows * d)?;
            let (r, c) = (rows as u32, d as u32);
            let mut bld = stream.launch_builder(&self.f_norm);
            bld.arg(&r).arg(&c).arg(&eps).arg(src).arg(&mut out);
            unsafe { bld.launch(norm_cfg)? };
            Ok(out)
        };
        let cast = |src: &cudarc::driver::CudaSlice<f32>, n: usize| -> Result<cudarc::driver::CudaSlice<f16>, DriverError> {
            let mut dst = stream.alloc_zeros::<f16>(n)?;
            let nn = n as u32;
            let mut b = stream.launch_builder(&self.f_cast);
            b.arg(&nn).arg(src).arg(&mut dst);
            unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
            Ok(dst)
        };
        // plain C = A*Bᵀ, no epilogue (always self.f_gemm) — the down/out projections do NOT fold residual.
        let gemm = |a: &cudarc::driver::CudaSlice<f16>, b: &cudarc::driver::CudaSlice<f16>, m: usize, k: usize, n: usize| -> Result<_, DriverError> {
            let mut c = stream.alloc_zeros::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(&self.f_gemm);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut c);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };
        // separate residual add `out = a + b` (the kernel the fused residual GEMM folds away).
        let vadd = |a: &cudarc::driver::CudaSlice<f32>, b: &cudarc::driver::CudaSlice<f32>, n: usize| -> Result<_, DriverError> {
            let mut out = stream.alloc_zeros::<f32>(n)?;
            let nn = n as u32;
            let mut bld = stream.launch_builder(&self.f_vadd);
            bld.arg(&nn).arg(a).arg(b).arg(&mut out);
            unsafe { bld.launch(LaunchConfig::for_num_elems(nn))? };
            Ok(out)
        };
        // separate SiLU activation (the kernel the fused SiLU GEMM folds away).
        let silu = |src: &cudarc::driver::CudaSlice<f32>, n: usize| -> Result<_, DriverError> {
            let mut out = stream.alloc_zeros::<f32>(n)?;
            let nn = n as u32;
            let mut bld = stream.launch_builder(&self.f_act);
            bld.arg(&nn).arg(src).arg(&mut out);
            unsafe { bld.launch(LaunchConfig::for_num_elems(nn))? };
            Ok(out)
        };

        // attention: plain projections, flash, then a SEPARATE residual add.
        let h1 = norm(x_d, s)?;
        let h1_16 = cast(&h1, s * d)?;
        let q = gemm(&h1_16, &self.wq, s, d, d)?;
        let k = gemm(&h1_16, &self.wk, s, d, d)?;
        let v = gemm(&h1_16, &self.wv, s, d, d)?;
        let mut attn = stream.alloc_zeros::<f32>(s * d)?;
        {
            let scale = 1.0f32 / (d as f32).sqrt();
            let ss = s as u32;
            let flash_cfg = LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            let mut bld = stream.launch_builder(&self.f_flash);
            bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut attn);
            unsafe { bld.launch(flash_cfg)? };
        }
        let attn_16 = cast(&attn, s * d)?;
        let o = gemm(&attn_16, &self.wo, s, d, d)?;
        let x1 = vadd(x_d, &o, s * d)?; // separate residual 1

        // FFN: plain up-proj, SEPARATE SiLU, plain down-proj, SEPARATE residual add.
        let h2 = norm(&x1, s)?;
        let h2_16 = cast(&h2, s * d)?;
        let f1 = gemm(&h2_16, &self.w1, s, d, dff)?;
        let f1act = silu(&f1, s * dff)?;
        let f1act_16 = cast(&f1act, s * dff)?;
        let f2 = gemm(&f1act_16, &self.w2, s, dff, d)?;
        let out = vadd(&x1, &f2, s * d)?; // separate residual 2
        Ok(out)
    }

    /// One-shot host call: upload `x` (`[S,D]` f32), run [`forward_device`](Self::forward_device), copy
    /// the `[S,D]` result back. The transfer-inclusive convenience path; a real stack keeps both ends
    /// resident and calls `forward_device` directly.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, DriverError> {
        assert_eq!(x.len(), self.s * self.d, "x must be S×D");
        let x_d = self.stream.memcpy_stod(x)?;
        let out = self.forward_device(&x_d)?;
        self.stream.memcpy_dtov(&out)
    }
}

/// One **pre-norm transformer layer on the fp16 tensor cores** (one-shot): builds a [`ResidentLayerF16`]
/// (weights narrowed to f16 and uploaded once) and runs one [`forward`](ResidentLayerF16::forward). The
/// fast, fused sibling of the f32 [`transformer_layer`]; see [`ResidentLayerF16`] for the kernel chain
/// and shape constraints. fp16-GEMM precision ⇒ tolerance-gated against the same layer's f64 reference
/// with f16-rounded GEMM inputs. For a resident stack (no per-call weight upload), build the layer once
/// and call [`forward_device`](ResidentLayerF16::forward_device).
pub fn transformer_layer_f16(
    g: &mut Gpu,
    x: &[f32],
    w: &TransformerWeights,
    s: usize,
    d: usize,
    dff: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(x.len(), s * d, "x must be S×D");
    ResidentLayerF16::new(g, w, s, d, dff)?.forward(x)
}

/// A **stack of N [`ResidentLayerF16`] layers, whole-model GPU-resident** — the M13 shape. Every layer's
/// weights upload once at construction; [`forward`](Self::forward) uploads the input `[S,D]` **once**,
/// runs all N layers via [`forward_device`](ResidentLayerF16::forward_device) so each layer's output is
/// the next layer's input **without ever leaving the GPU**, and copies the final `[S,D]` back **once**.
/// So an N-layer forward is `N×(13 launches)` with exactly one H2D + one D2H — no per-layer weight
/// re-upload, no intermediate host round-trip. The fixed input/output transfer amortizes over the N
/// layers, so the per-layer cost falls toward the pure resident compute as depth grows (the residency
/// win that a per-call library chain, re-staging weights and activations through HBM, cannot capture).
pub struct ResidentModelF16 {
    stream: Arc<CudaStream>,
    layers: Vec<ResidentLayerF16>,
    s: usize,
    d: usize,
}

impl ResidentModelF16 {
    /// Build the N resident layers (one [`ResidentLayerF16::new`] per weight set) — all share the one
    /// device stream, so the whole stack enqueues in order on a single timeline.
    pub fn new(
        g: &mut Gpu,
        weights: &[TransformerWeights],
        s: usize,
        d: usize,
        dff: usize,
    ) -> Result<Self, DriverError> {
        assert!(!weights.is_empty(), "model needs at least one layer");
        let mut layers = Vec::with_capacity(weights.len());
        for w in weights {
            layers.push(ResidentLayerF16::new(g, w, s, d, dff)?);
        }
        let stream = g.stream.clone();
        Ok(Self { stream, layers, s, d })
    }

    /// Number of transformer layers in the stack.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Run the whole stack on a **resident** `[S,D]` buffer, returning the resident final output — the
    /// pure on-device N-layer chain, no host transfer. Each layer consumes the prior layer's device
    /// output directly.
    pub fn forward_device(
        &self,
        x_d: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<cudarc::driver::CudaSlice<f32>, DriverError> {
        let mut cur = self.layers[0].forward_device(x_d)?;
        for layer in &self.layers[1..] {
            cur = layer.forward_device(&cur)?;
        }
        Ok(cur)
    }

    /// One-shot host call: upload `x` once, run the whole resident stack, copy the final `[S,D]` back
    /// once — the whole-model forward with a single H2D at the front and a single D2H at the end.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, DriverError> {
        assert_eq!(x.len(), self.s * self.d, "x must be S×D");
        let x_d = self.stream.memcpy_stod(x)?;
        let out = self.forward_device(&x_d)?;
        self.stream.memcpy_dtov(&out)
    }
}

/// One **fp8 (E4M3) tensor-core tile** `D = A·B` via `mma.sync.m16n8k32` (Ada has no WMMA fp8): `a`
/// is `16×32` row-major, `b_col` is `32×8` **column-major** (the `.col` operand), both arrive as f32
/// and are rounded to E4M3 on the host; `D` is `16×8` f32 (the mixed-precision accumulate). Validates
/// the manual fragment layout — the core a full fp8 GEMM would tile over.
pub fn fp8_tile(g: &mut Gpu, a: &[f32], b_col: &[f32]) -> Result<Vec<f32>, DriverError> {
    assert_eq!(a.len(), 16 * 32, "A must be 16×32");
    assert_eq!(b_col.len(), 32 * 8, "B must be 32×8 (column-major)");
    let a8: Vec<u8> = a.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
    let b8: Vec<u8> = b_col
        .iter()
        .map(|&x| crate::ptx_fp8::f32_to_e4m3(x))
        .collect();
    let f = g.function("fp8_tile", crate::ptx_fp8::FP8_TILE, "fp8_tile")?;
    let a_d = g.stream.memcpy_stod(&a8)?;
    let b_d = g.stream.memcpy_stod(&b8)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; 16 * 8])?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// Full **fp8 (E4M3) tensor-core `C = A·Bᵀ`** (nn.Linear): `A` (m×k) and `B` (n×k) arrive as f32 and
/// are rounded to E4M3 on the host; `C` is f32 (mixed-precision accumulate). Each warp computes a
/// 16×8 tile via `mma.sync.m16n8k32`. Requires m%16==0, n%8==0, k%32==0. The lowest-precision /
/// highest-throughput tensor-core path on Ada — 2× the fp16 rate at peak.
pub fn gemm_nt_fp8(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % 16 == 0 && n % 8 == 0 && k % 32 == 0,
        "fp8 GEMM needs M%16==0, N%8==0, K%32==0"
    );
    let a8: Vec<u8> = a.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
    let b8: Vec<u8> = b.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
    // Fragment-reuse multi-tile kernel when the block divides evenly (the fast path), else single-tile.
    use crate::ptx_fp8::{FP8_TM, FP8_TN};
    let (f, cfg) = if m % (16 * FP8_TM) == 0 && n % (8 * FP8_TN) == 0 {
        (
            g.function(
                "fp8_gemm_mt",
                crate::ptx_fp8::fp8_gemm_mt_ptx(),
                "fp8_gemm_nt_mt",
            )?,
            LaunchConfig {
                grid_dim: ((n / (8 * FP8_TN)) as u32, (m / (16 * FP8_TM)) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        (
            g.function("fp8_gemm", crate::ptx_fp8::fp8_gemm_ptx(), "fp8_gemm_nt")?,
            LaunchConfig {
                grid_dim: ((n / 8) as u32, (m / 16) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    };
    let a_d = g.stream.memcpy_stod(&a8)?;
    let b_d = g.stream.memcpy_stod(&b8)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

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

    #[test]
    fn copy_v4_is_bit_exact() {
        with_gpu("copy_v4", |g| {
            // Several multiples of 4, incl. ones smaller than and far larger than one wave, to exercise
            // the grid-stride loop (each thread copies many float4s) and the device-saturating grid cap.
            for &n in &[4usize, 4096, 1 << 20] {
                let x: Vec<f32> =
                    (0..n).map(|i| f32::from_bits(0xCAFE_0000 ^ i as u32)).collect();
                let out = copy(g, &x).unwrap();
                assert_eq!(out.len(), n);
                for i in 0..n {
                    assert_eq!(out[i].to_bits(), x[i].to_bits(), "copy lane {i} (n={n})");
                }
            }
        });
    }

    /// The f32→f16 device cast (`cast_f32_f16`, `cvt.rn.f16.f32`) must round-to-nearest-even exactly
    /// like `half::f16::from_f32` — bit-for-bit, so a resident fp16 pipeline's on-device narrowing is
    /// identical to the host-side rounding the GEMM wrappers do. Covers normals plus edges: ±0, f16-max
    /// (65504), a subnormal-ish 1e-5, and 2049/−2049 (round-half-to-even ties at the f16 mantissa step).
    #[test]
    fn cast_f32_to_f16_round_trips_bit_exact() {
        use half::f16;
        with_gpu("cast_f16", |g| {
            let mut rng = crate::diff::Rng::new(0xCA57);
            let mut x = rng.vec(4096, -3.0, 3.0);
            x.extend_from_slice(&[0.0, -0.0, 1.0, 65504.0, 1e-5, 0.5, 2049.0, -2049.0]);
            let got = cast_f32_to_f16(g, &x).unwrap();
            for (i, (&xi, &gi)) in x.iter().zip(got.iter()).enumerate() {
                let want = f16::from_f32(xi).to_f32();
                assert_eq!(gi.to_bits(), want.to_bits(), "cast lane {i}: x={xi}");
            }
        });
    }

    fn cpu_vmath(op: i64, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        unsafe {
            mercury_runtime::mercury_vmath_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i64, op)
        };
        out
    }

    #[test]
    fn vmath_activations_match_cpu_oracle_within_tol() {
        use mercury_runtime::{VM_EXP, VM_GELU, VM_RELU, VM_SIGMOID, VM_SILU, VM_TANH};
        with_gpu("vmath_activations", |g| {
            let mut rng = crate::diff::Rng::new(0xA1);
            let n = 50_000usize; // not a multiple of any block size
            let x = rng.vec(n, -10.0, 10.0);
            // (op, label, abs_tol, rel_tol). relu is exact; transcendentals use SFU approximations.
            let cases: &[(i64, &str, f64, f64)] = &[
                (VM_RELU, "relu", 0.0, 0.0),
                (VM_EXP, "exp", 1e-3, 5e-4),
                (VM_SIGMOID, "sigmoid", 5e-4, 1e-3),
                (VM_TANH, "tanh", 5e-4, 1e-3),
                (VM_SILU, "silu", 1e-3, 1e-3),
                (VM_GELU, "gelu", 1e-3, 1e-3),
            ];
            for &(op, label, abs_tol, rel_tol) in cases {
                let got = vmath(g, op, &x).unwrap();
                let oracle = cpu_vmath(op, &x);
                let s = crate::diff::assert_close(label, &got, &oracle, abs_tol, rel_tol);
                eprintln!(
                    "vmath {label:8}: max_abs={:.3e} max_rel={:.3e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    #[test]
    fn reductions_match_reference_within_tol_and_are_deterministic() {
        use mercury_runtime::{mercury_sreduce_f32, RED_DOT, RED_SUM};
        with_gpu("reductions", |g| {
            let mut rng = crate::diff::Rng::new(0xBEEF);
            let n = 1 << 20; // 1,048,576 elements
                             // Positive inputs keep sum/dot well-conditioned (no catastrophic cancellation), so a
                             // relative tolerance is meaningful.
            let x = rng.vec(n, 0.0, 1.0);
            let y = rng.vec(n, 0.0, 1.0);

            // --- sum ---
            let gpu_sum = reduce(g, RED_SUM, &x, None).unwrap();
            let ref_sum: f64 = x.iter().map(|&v| v as f64).sum();
            let cpu_sum = unsafe { mercury_sreduce_f32(x.as_ptr(), x.as_ptr(), n as i64, RED_SUM) };
            let rel = crate::diff::assert_scalar_close("sum vs f64", gpu_sum, ref_sum, 1e-1, 1e-3);
            eprintln!("reduce sum: gpu={gpu_sum} cpu={cpu_sum} ref={ref_sum:.6} rel={rel:.3e}");

            // --- dot ---
            let gpu_dot = reduce(g, RED_DOT, &x, Some(&y)).unwrap();
            let ref_dot: f64 = x.iter().zip(&y).map(|(&a, &b)| a as f64 * b as f64).sum();
            let rel = crate::diff::assert_scalar_close("dot vs f64", gpu_dot, ref_dot, 1e-1, 1e-3);
            eprintln!("reduce dot: gpu={gpu_dot} ref={ref_dot:.6} rel={rel:.3e}");

            // --- max (order-independent and exact: GPU == true max bit-for-bit) ---
            let xs = rng.vec(n, -5.0, 5.0);
            let gpu_max = reduce(g, RED_MAX, &xs, None).unwrap();
            let ref_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                gpu_max.to_bits(),
                ref_max.to_bits(),
                "gpu max must be exact"
            );

            // --- determinism: same inputs → identical bits across runs ---
            let again = reduce(g, RED_SUM, &x, None).unwrap();
            assert_eq!(
                gpu_sum.to_bits(),
                again.to_bits(),
                "reduction must be deterministic"
            );
        });
    }

    /// f64 reference for `C = A·Bᵀ` (`A` m×k, `B` n×k).
    fn ref_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += a[i * k + kk] as f64 * b[j * k + kk] as f64;
                }
                c[i * n + j] = acc as f32;
            }
        }
        c
    }

    /// f64 reference for `C = A·B` (`A` m×k, `B` k×n).
    fn ref_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
                }
                c[i * n + j] = acc as f32;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_reference_within_tol() {
        with_gpu("gemm", |g| {
            let mut rng = crate::diff::Rng::new(0x6E33);
            // include ragged (non-multiple-of-16) M/N/K to exercise the bounds guards
            let shapes = [(64usize, 64usize, 64usize), (100, 80, 48), (128, 256, 192)];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let bt = rng.vec(n * k, -1.0, 1.0); // n×k for A·Bᵀ
                let bn = rng.vec(k * n, -1.0, 1.0); // k×n for A·B

                // c·√K·ε bound; abs cushion for tiles where the true value is ~0
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(1e-4);

                let c_nt = gemm_nt(g, &a, &bt, m, k, n).unwrap();
                let r_nt = ref_nt(&a, &bt, m, k, n);
                let s = crate::diff::assert_close(
                    &format!("gemm_nt {m}x{k}x{n}"),
                    &c_nt,
                    &r_nt,
                    1e-3,
                    rel,
                );
                eprintln!(
                    "gemm_nt {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );

                let c_nn = gemm_nn(g, &a, &bn, m, k, n).unwrap();
                let r_nn = ref_nn(&a, &bn, m, k, n);
                crate::diff::assert_close(&format!("gemm_nn {m}x{k}x{n}"), &c_nn, &r_nn, 1e-3, rel);
            }
        });
    }

    #[test]
    fn gemm_rb_matches_reference_within_tol() {
        with_gpu("gemm_rb", |g| {
            let mut rng = crate::diff::Rng::new(0x9C17);
            let shapes = [(64usize, 64usize, 64usize), (100, 80, 48), (130, 200, 70)];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let bt = rng.vec(n * k, -1.0, 1.0);
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(1e-4);
                let c = gemm_nt_rb(g, &a, &bt, m, k, n).unwrap();
                let r = ref_nt(&a, &bt, m, k, n);
                let s = crate::diff::assert_close(
                    &format!("gemm_nt_rb {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-3,
                    rel,
                );
                eprintln!(
                    "gemm_nt_rb {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// f64 reference for `C = A·Bᵀ` with each input first rounded by `round` (to match what the
    /// tensor-core kernel actually multiplies: f16/bf16 inputs). Isolates the GEMM accumulation error
    /// from the input-precision loss.
    fn ref_nt_rounded(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        round: impl Fn(f32) -> f32,
    ) -> Vec<f32> {
        let ar: Vec<f64> = a.iter().map(|&x| round(x) as f64).collect();
        let br: Vec<f64> = b.iter().map(|&x| round(x) as f64).collect();
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += ar[i * k + kk] * br[j * k + kk];
                }
                c[i * n + j] = acc as f32;
            }
        }
        c
    }

    #[test]
    fn wmma_tensorcore_matches_reference_within_tol() {
        use half::{bf16, f16};
        with_gpu("wmma", |g| {
            let mut rng = crate::diff::Rng::new(0x7C0DE);
            let shapes = [(16usize, 16usize, 16usize), (64, 64, 64), (256, 128, 512)];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);

                // f16 inputs: products are exact in f32, so only f32 accumulation deviates.
                let c16 = gemm_nt_f16(g, &a, &b, m, k, n).unwrap();
                let r16 = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let s = crate::diff::assert_close(
                    &format!("wmma_f16 {m}x{k}x{n}"),
                    &c16,
                    &r16,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16  {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );

                // bf16 inputs: fewer mantissa bits → looser tolerance, but still exact products.
                let cb = gemm_nt_bf16(g, &a, &b, m, k, n).unwrap();
                let rb = ref_nt_rounded(&a, &b, m, k, n, |x| bf16::from_f32(x).to_f32());
                let s = crate::diff::assert_close(
                    &format!("wmma_bf16 {m}x{k}x{n}"),
                    &cb,
                    &rb,
                    2e-2,
                    1e-2,
                );
                eprintln!(
                    "wmma_bf16 {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The shared-memory-staged fp16 GEMM (`wmma_nt_f16_sm`) must match the f16-rounded f64 reference,
    /// to the same tolerance as the `_mt` path — it computes the identical math, only with A/B tiles
    /// routed through shared memory and warps cooperating per CTA. Shapes exercise the SM_BM/SM_BN
    /// divisibility plus a rectangular case (stresses the cooperative-load and store indexing).
    #[test]
    fn wmma_sm_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm", |g| {
            let mut rng = crate::diff::Rng::new(0x5EED);
            let shapes = [
                (64usize, 64usize, 64usize),
                (128, 128, 128),
                (128, 80, 192),
                (256, 128, 512),
            ];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = gemm_nt_f16_sm(g, &a, &b, m, k, n).unwrap();
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_sm {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16_sm {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The 128×128 cp.async double-buffered kernel (`wmma_nt_f16_sm128_db`) must match the f16-rounded
    /// f64 reference to the same tolerance as the 64×64 paths — identical math, only a bigger
    /// cooperative tile, an 8-warp (2×4) grid, and the pipelined K-loop. Shapes exercise the M%128/N%128
    /// divisibility plus rectangular K and N (stresses the 256-thread vectorized staging, the per-warp
    /// 4×2 store indexing, and the pipeline prologue/drain at small and large K).
    #[test]
    fn wmma_sm128_db_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm128_db", |g| {
            let mut rng = crate::diff::Rng::new(0x5E12);
            let shapes = [
                (128usize, 16usize, 128usize),
                (128, 128, 128),
                (256, 256, 256),
                (256, 128, 512),
                (384, 160, 256),
                (512, 80, 128),
            ];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let c = gemm_nt_f16_sm128_db(g, &a, &b, m, k, n).unwrap();
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_sm128_db {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16_sm128_db {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The single-buffered 128×128 kernel (`wmma_nt_f16_sm128`) must match the f16-rounded f64 reference
    /// to the same tolerance as the other staged paths — identical math, the 128 cooperative tile and
    /// 8-warp (2×4) grid of the `_sm128_db` kernel but without the cp.async pipeline. Same shapes as the
    /// double-buffered 128 gate (M%128/N%128 divisibility, rectangular K and N).
    #[test]
    fn wmma_sm128_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm128", |g| {
            let mut rng = crate::diff::Rng::new(0x5E13);
            let shapes = [
                (128usize, 16usize, 128usize),
                (128, 128, 128),
                (256, 256, 256),
                (256, 128, 512),
                (384, 160, 256),
                (512, 80, 128),
            ];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let c = gemm_nt_f16_sm128(g, &a, &b, m, k, n).unwrap();
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_sm128 {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16_sm128 {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The fused-residual kernel (`wmma_nt_f16_sm_db_residual`) must match `A·Bᵀ + residual` on the
    /// f16-rounded inputs. Seeding the accumulator via wmma.load.c must add the residual *exactly* — the
    /// load.c and store.d fragment layouts are inverse, so the opaque (lane,reg)→(row,col) map cancels; a
    /// wrong layout would scatter the O(1) residual to the wrong element and blow the abs tolerance.
    #[test]
    fn wmma_sm_db_residual_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm_db_residual", |g| {
            let mut rng = crate::diff::Rng::new(0x6E51);
            let shapes = [
                (64usize, 16usize, 64usize),
                (64, 128, 128),
                (128, 256, 256),
                (256, 128, 192),
            ];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let resid = rng.vec(m * n, -1.0, 1.0);
                let mut r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                for (ri, &res) in r.iter_mut().zip(resid.iter()) {
                    *ri += res;
                }
                let c = gemm_nt_f16_sm_db_residual(g, &a, &b, &resid, m, k, n).unwrap();
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_sm_db_residual {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16_sm_db_residual {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The `cp.async` double-buffered kernel (`wmma_nt_f16_sm_db`) must match the f16-rounded f64
    /// reference to the same tolerance as the `_sm` path — identical math and tiling, only the K-loop
    /// is software-pipelined (prefetch next tile via cp.async while computing the current). Shapes
    /// stress the pipeline prologue/steady-state/drain: K=16 (single tile, no prefetch), K=64/256
    /// (many steps), plus rectangular cases.
    #[test]
    fn wmma_sm_db_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm_db", |g| {
            let mut rng = crate::diff::Rng::new(0x0DB1);
            let shapes = [
                (64usize, 16usize, 64usize),
                (64, 64, 64),
                (128, 256, 128),
                (128, 80, 192),
                (256, 128, 512),
            ];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = gemm_nt_f16_sm_db(g, &a, &b, m, k, n).unwrap();
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_sm_db {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16_sm_db {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The **fused** `relu(A·Bᵀ)` kernel must equal relu applied to the same f16-rounded f64 reference
    /// — i.e. the epilogue activates the accumulator with no effect on the GEMM math. relu is exact
    /// (`max(x,0)`), so it neither tightens nor loosens the GEMM's accumulation tolerance; a missing or
    /// misplaced epilogue (e.g. negative outputs surviving) fails immediately.
    #[test]
    fn wmma_sm_db_relu_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm_db_relu", |g| {
            let mut rng = crate::diff::Rng::new(0x0DB2);
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 256, 128), (256, 128, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = gemm_nt_f16_sm_db_relu(g, &a, &b, m, k, n).unwrap();
                let mut r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                for v in &mut r {
                    *v = v.max(0.0); // fused relu
                }
                assert!(c.iter().all(|&v| v >= 0.0), "relu output must be non-negative");
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_sm_db_relu {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!(
                    "wmma_f16_sm_db_relu {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The **fused** `silu(A·Bᵀ)` and `gelu(A·Bᵀ)` kernels must equal the exact activation applied to the
    /// f16-rounded reference GEMM. Unlike relu these use the Ada SFU (`ex2.approx`/`tanh.approx`/`rcp`),
    /// so the gate is a tolerance one — but it inherits the same bounds as the standalone vmath silu/gelu
    /// (the epilogue is byte-identical PTX), and a misplaced/missing epilogue still fails wide. silu is
    /// the SwiGLU FFN up-projection fused into one kernel — the M13 megakernel-vs-call-chain lever.
    #[test]
    fn wmma_sm_db_silu_gelu_match_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm_db_silu_gelu", |g| {
            let mut rng = crate::diff::Rng::new(0x5170);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 256, 128), (256, 128, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let base = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                for (name, got, refv) in [
                    (
                        "silu",
                        gemm_nt_f16_sm_db_silu(g, &a, &b, m, k, n).unwrap(),
                        base.iter().map(|&x| silu(x)).collect::<Vec<_>>(),
                    ),
                    (
                        "gelu",
                        gemm_nt_f16_sm_db_gelu(g, &a, &b, m, k, n).unwrap(),
                        base.iter().map(|&x| gelu(x)).collect::<Vec<_>>(),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_f16_sm_db_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        1e-2,
                    );
                    eprintln!(
                        "wmma_f16_sm_db_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The fused **bias (+ activation)** kernels `C = act(A·Bᵀ + bias)` — the canonical `nn.Linear`/FFN
    /// epilogue. Unlike the activation-only fused path (which acts on accumulator *registers*), a
    /// per-column bias needs the WMMA fragment's column index, so these route each tile through SMEM and
    /// re-read it by explicit (row,col). Each output must equal the activation of (f16-rounded reference
    /// GEMM + bias[col]); a wrong (row,col)→bias map, a dropped SMEM barrier, or a clobbered scratch slot
    /// scrambles outputs by O(|bias|) ≫ tol and fails wide. Bias is added *before* the activation (the
    /// `act(x·Wᵀ+bias)` order). The identity case (`bias`) is affine Linear; the rest are Linear+act.
    #[test]
    fn wmma_sm_db_bias_match_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_sm_db_bias", |g| {
            let mut rng = crate::diff::Rng::new(0xB1A5);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 256, 128), (256, 128, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                // Reference: (f16-rounded GEMM) + bias[col], then the activation — per the kernel order.
                let base = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let with_bias = |act: &dyn Fn(f32) -> f32| -> Vec<f32> {
                    let mut r = base.clone();
                    for i in 0..m {
                        for j in 0..n {
                            r[i * n + j] = act(r[i * n + j] + bias[j]);
                        }
                    }
                    r
                };
                let id = |x: f32| x;
                for (name, got, refv) in [
                    (
                        "bias",
                        gemm_nt_f16_sm_db_bias(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&id),
                    ),
                    (
                        "bias_relu",
                        gemm_nt_f16_sm_db_bias_relu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&|x| x.max(0.0)),
                    ),
                    (
                        "bias_silu",
                        gemm_nt_f16_sm_db_bias_silu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&silu),
                    ),
                    (
                        "bias_gelu",
                        gemm_nt_f16_sm_db_bias_gelu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&gelu),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_f16_sm_db_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        1e-2,
                    );
                    eprintln!(
                        "wmma_f16_sm_db_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The **bf16** fused epilogues (`wmma_nt_bf16_sm_db_{relu,silu,gelu}`) — the same precision-generic
    /// pipelined+fused kernel exercised in the dtype transformers *train* in. bf16's wider exponent /
    /// fewer mantissa bits don't change the epilogue (it acts on the f32 accumulator); each fused output
    /// must equal the activation of the bf16-rounded reference GEMM. This also gates the bf16 `_sm_db`
    /// path itself (new — bf16 previously had only the single-tile and `_mt` variants).
    #[test]
    fn wmma_bf16_sm_db_fused_match_reference_within_tol() {
        use half::bf16;
        with_gpu("wmma_bf16_sm_db_fused", |g| {
            let mut rng = crate::diff::Rng::new(0xBF16);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 256, 128), (256, 128, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let base = ref_nt_rounded(&a, &b, m, k, n, |x| bf16::from_f32(x).to_f32());
                for (name, got, refv) in [
                    (
                        "relu",
                        gemm_nt_bf16_sm_db_relu(g, &a, &b, m, k, n).unwrap(),
                        base.iter().map(|&x| x.max(0.0)).collect::<Vec<_>>(),
                    ),
                    (
                        "silu",
                        gemm_nt_bf16_sm_db_silu(g, &a, &b, m, k, n).unwrap(),
                        base.iter().map(|&x| silu(x)).collect::<Vec<_>>(),
                    ),
                    (
                        "gelu",
                        gemm_nt_bf16_sm_db_gelu(g, &a, &b, m, k, n).unwrap(),
                        base.iter().map(|&x| gelu(x)).collect::<Vec<_>>(),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_bf16_sm_db_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        2e-2,
                    );
                    eprintln!(
                        "wmma_bf16_sm_db_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The **bf16** fused bias kernels `C = act(A·Bᵀ + bias)` — the bf16 (training-dtype) twin of
    /// `wmma_sm_db_bias_match_reference_within_tol`. The bias epilogue acts on the f32 accumulator
    /// (dtype-independent), so this also confirms the bf16 `_sm_db_bias*` entries are wired and the
    /// per-column bias maps correctly under the bf16 input quantization.
    #[test]
    fn wmma_bf16_sm_db_bias_match_reference_within_tol() {
        use half::bf16;
        with_gpu("wmma_bf16_sm_db_bias", |g| {
            let mut rng = crate::diff::Rng::new(0xBB1A);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 256, 128), (256, 128, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                let base = ref_nt_rounded(&a, &b, m, k, n, |x| bf16::from_f32(x).to_f32());
                let with_bias = |act: &dyn Fn(f32) -> f32| -> Vec<f32> {
                    let mut r = base.clone();
                    for i in 0..m {
                        for j in 0..n {
                            r[i * n + j] = act(r[i * n + j] + bias[j]);
                        }
                    }
                    r
                };
                let id = |x: f32| x;
                for (name, got, refv) in [
                    ("bias", gemm_nt_bf16_sm_db_bias(g, &a, &b, &bias, m, k, n).unwrap(), with_bias(&id)),
                    (
                        "bias_relu",
                        gemm_nt_bf16_sm_db_bias_relu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&|x| x.max(0.0)),
                    ),
                    (
                        "bias_silu",
                        gemm_nt_bf16_sm_db_bias_silu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&silu),
                    ),
                    (
                        "bias_gelu",
                        gemm_nt_bf16_sm_db_bias_gelu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&gelu),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_bf16_sm_db_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        2e-2,
                    );
                    eprintln!(
                        "wmma_bf16_sm_db_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// Diagnostic: JIT a PTX module with the driver's error-log buffer attached and print it. The
    /// plain `load_module` path only surfaces `CUDA_ERROR_INVALID_PTX` with no detail; this prints
    /// `ptxas`'s actual line/error, which is how every hand-written PTX kernel here gets debugged.
    /// Point it at whichever module you're bringing up. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu jit_log -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic; prints the driver JIT log for a PTX module"]
    fn jit_log() {
        with_gpu("jitlog", |g| {
            use cudarc::driver::sys;
            g.ctx.bind_to_thread().unwrap();
            let ptx = crate::ptx_wmma::wmma_f16_ptx(); // ← swap in the module under test
            let ptx_c = std::ffi::CString::new(ptx).unwrap();
            let mut log = vec![0u8; 32768];
            let mut opts = [
                sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
                sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            ];
            let mut vals: [*mut std::ffi::c_void; 2] =
                [log.as_mut_ptr() as *mut _, log.len() as *mut _];
            let mut module: sys::CUmodule = std::ptr::null_mut();
            let res = unsafe {
                sys::cuModuleLoadDataEx(
                    &mut module,
                    ptx_c.as_ptr() as *const _,
                    2,
                    opts.as_mut_ptr(),
                    vals.as_mut_ptr(),
                )
            };
            let s = String::from_utf8_lossy(&log);
            eprintln!("=== JIT result {:?} ===\n{}", res, s.trim_end_matches('\0'));
        });
    }

    /// The cubin cache route (M10): `ptx_to_cubin` must emit a real SASS cubin (ELF), and loading it
    /// back via `Ptx::from_file` must yield a module whose entries resolve — i.e. a warm process skips
    /// the PTX JIT entirely. (Execution equivalence of the cached path is covered by the whole suite,
    /// which loads every kernel through `load_module_cached`; a second test-binary run exercises the
    /// warm branch end-to-end.)
    #[test]
    fn cubin_cache_roundtrips_to_loadable_sass() {
        with_gpu("cubin_roundtrip", |g| {
            let _ = g.ctx.bind_to_thread();
            let ptx = crate::ptx_wmma::wmma_f16_ptx();
            let cubin = crate::cubin::ptx_to_cubin(ptx).expect("driver should link PTX→cubin");
            assert!(
                cubin.len() > 64 && &cubin[..4] == b"\x7fELF",
                "expected an ELF SASS cubin, got {} bytes",
                cubin.len()
            );
            let tmp = std::env::temp_dir()
                .join(format!("mercury_cubin_roundtrip_{}.cubin", std::process::id()));
            crate::cubin::write_atomic(&tmp, &cubin).unwrap();
            let m = g
                .ctx
                .load_module(Ptx::from_file(&tmp))
                .expect("a cached cubin must load without JIT");
            for entry in ["wmma_nt_f16_sm", "wmma_nt_f16_sm_db", "wmma_nt_f16_sm128_db"] {
                m.load_function(entry)
                    .unwrap_or_else(|_| panic!("entry {entry} missing from the cubin"));
            }
            let _ = std::fs::remove_file(&tmp);
        });
    }

    /// M10 latency: a warm cubin load (precompiled SASS) vs a from-scratch PTX JIT, same module,
    /// same process. Reports the true-cold first JIT, the driver's own JIT-cache-warm PTX load, and
    /// Mercury's cubin load — the last is fast *and* portable/deterministic (independent of the
    /// driver's opaque, clearable compute cache). The cold figure also documents that even a
    /// from-scratch driver JIT is orders of magnitude under Triton/Inductor's 30–120 s cold autotune.
    #[test]
    #[ignore = "latency bench; run explicitly"]
    fn cubin_cache_compile_latency() {
        use std::time::Instant;
        with_gpu("cubin_latency", |g| {
            let _ = g.ctx.bind_to_thread();
            let ptx = crate::ptx_wmma::wmma_f16_ptx();
            let cubin = crate::cubin::ptx_to_cubin(ptx).expect("link PTX→cubin");
            let tmp = std::env::temp_dir()
                .join(format!("mercury_cubin_lat_{}.cubin", std::process::id()));
            crate::cubin::write_atomic(&tmp, &cubin).unwrap();

            // First load = true cold for this process (may populate the driver's own JIT cache).
            let t0 = Instant::now();
            let _ = g.ctx.load_module(Ptx::from_src(ptx)).unwrap();
            let cold_first = t0.elapsed().as_secs_f64();
            // Best-of-N: PTX load (driver-JIT-cache warm) vs cubin load (Mercury cache).
            let mut ptx_warm = f64::INFINITY;
            let mut cubin_warm = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                let _ = g.ctx.load_module(Ptx::from_src(ptx)).unwrap();
                ptx_warm = ptx_warm.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                let _ = g.ctx.load_module(Ptx::from_file(&tmp)).unwrap();
                cubin_warm = cubin_warm.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "module load latency — wmma_f16 ({} B PTX → {} B cubin):",
                ptx.len(),
                cubin.len()
            );
            eprintln!("  cold PTX JIT (first this process) : {:>7.2} ms", cold_first * 1e3);
            eprintln!("  PTX load, driver-cache warm       : {:>7.2} ms  (best/10)", ptx_warm * 1e3);
            eprintln!(
                "  cubin load (Mercury cache)        : {:>7.2} ms  (best/10) | {:.1}× vs cold-first | {:.1}× vs PTX-warm",
                cubin_warm * 1e3,
                cold_first / cubin_warm,
                ptx_warm / cubin_warm
            );
            eprintln!(
                "  vs Triton/Inductor cold 30–120 s  : even the cold PTX JIT is ~{:.0}×–{:.0}× faster (M10 ≥100×)",
                30.0 / cold_first,
                120.0 / cold_first
            );
            let _ = std::fs::remove_file(&tmp);
        });
    }

    fn cpu_norm(op: i64, x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        unsafe {
            mercury_runtime::mercury_norm_f32(
                x.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                cols as i64,
                eps.to_bits() as i64,
                op,
            )
        };
        out
    }

    #[test]
    fn norms_match_cpu_oracle_within_tol() {
        use mercury_runtime::{NORM_LAYERNORM, NORM_RMSNORM, NORM_SOFTMAX};
        with_gpu("norms", |g| {
            let mut rng = crate::diff::Rng::new(0x5037);
            let (rows, cols) = (128usize, 1024usize);
            let x = rng.vec(rows * cols, -3.0, 3.0);
            let eps = 1e-5f32;
            for (op, label) in [
                (NORM_SOFTMAX, "softmax"),
                (NORM_LAYERNORM, "layernorm"),
                (NORM_RMSNORM, "rmsnorm"),
            ] {
                let got = norm(g, op, &x, rows, cols, eps).unwrap();
                let oracle = cpu_norm(op, &x, rows, cols, eps);
                let s = crate::diff::assert_close(label, &got, &oracle, 1e-4, 1e-3);
                eprintln!(
                    "norm {label:10}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// f64 reference for single-head attention `O = softmax(scale·Q·Kᵀ)·V`, all `[seq, d]`. The
    /// non-flash (materialized, two-pass softmax) form: an independent oracle for the fused kernel.
    fn ref_attn(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize, scale: f32) -> Vec<f32> {
        let mut o = vec![0.0f32; seq * d];
        for i in 0..seq {
            // scores[j] = scale · Q[i]·K[j]
            let mut scores = vec![0.0f64; seq];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for t in 0..d {
                    acc += q[i * d + t] as f64 * k[j * d + t] as f64;
                }
                *sc = acc * scale as f64;
            }
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut l = 0.0f64;
            for sc in &mut scores {
                *sc = (*sc - m).exp();
                l += *sc;
            }
            for t in 0..d {
                let mut acc = 0.0f64;
                for (j, &p) in scores.iter().enumerate() {
                    acc += p * v[j * d + t] as f64;
                }
                o[i * d + t] = (acc / l) as f32;
            }
        }
        o
    }

    #[test]
    fn flash_attention_matches_reference_within_tol() {
        with_gpu("flash_attention", |g| {
            let mut rng = crate::diff::Rng::new(0xF1A54);
            // ragged seq (not a multiple of any block) across the supported head dims
            for (seq, d) in [(128usize, 32usize), (200, 64), (96, 128)] {
                let q = rng.vec(seq * d, -1.0, 1.0);
                let k = rng.vec(seq * d, -1.0, 1.0);
                let v = rng.vec(seq * d, -1.0, 1.0);
                let scale = 1.0 / (d as f32).sqrt();
                let got = flash_attn(g, &q, &k, &v, seq, d, scale).unwrap();
                let oracle = ref_attn(&q, &k, &v, seq, d, scale);
                let s = crate::diff::assert_close(
                    &format!("flash s={seq} d={d}"),
                    &got,
                    &oracle,
                    1e-3,
                    3e-3,
                );
                eprintln!(
                    "flash s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// Flash-attention throughput at increasing context length, kernel-resident (no per-iter copies).
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn flash_throughput() {
        with_gpu("flash_throughput", |g| {
            let mut rng = crate::diff::Rng::new(11);
            let d = 64usize;
            let f = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64")
                .unwrap();
            for seq in [512usize, 1024, 2048, 4096] {
                let q = rng.vec(seq * d, -1.0, 1.0);
                let k = rng.vec(seq * d, -1.0, 1.0);
                let v = rng.vec(seq * d, -1.0, 1.0);
                let q_d = g.stream.memcpy_stod(&q).unwrap();
                let k_d = g.stream.memcpy_stod(&k).unwrap();
                let v_d = g.stream.memcpy_stod(&v).unwrap();
                let mut o_d = g.stream.memcpy_stod(&vec![0f32; seq * d]).unwrap();
                let scale = 1.0f32 / (d as f32).sqrt();
                let s = seq as u32;
                let cfg = LaunchConfig {
                    grid_dim: (s, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let launch = |o_d: &mut cudarc::driver::CudaSlice<f32>| {
                    let mut bld = g.stream.launch_builder(&f);
                    bld.arg(&s)
                        .arg(&scale)
                        .arg(&q_d)
                        .arg(&k_d)
                        .arg(&v_d)
                        .arg(o_d);
                    unsafe { bld.launch(cfg).unwrap() };
                };
                launch(&mut o_d);
                g.stream.synchronize().unwrap();
                let iters = 20;
                let t0 = Instant::now();
                for _ in 0..iters {
                    launch(&mut o_d);
                }
                g.stream.synchronize().unwrap();
                let spi = t0.elapsed().as_secs_f64() / iters as f64;
                // attention FLOPs ≈ QKᵀ (2·s²·d) + P·V (2·s²·d) = 4·s²·d
                let flop = 4.0 * (seq as f64) * (seq as f64) * (d as f64);
                eprintln!(
                    "flash s={seq} d={d}: {:.2} ms/iter, {:.0} GFLOP/s (fused, no s² scores in HBM)",
                    spi * 1e3,
                    flop / spi / 1e9
                );
            }
        });
    }

    /// f64 reference for direct conv2d (single batch, stride 1, no padding): `X[C,H,W]`, `W[K,C,R,S]`
    /// → `O[K,P,Q]`, `P=H-R+1`, `Q=W-S+1`. The independent oracle for the GPU kernel.
    #[allow(clippy::too_many_arguments)]
    fn ref_conv2d(
        x: &[f32],
        w: &[f32],
        c: usize,
        h: usize,
        width: usize,
        k: usize,
        r: usize,
        s: usize,
    ) -> Vec<f32> {
        let (p, q) = (h - r + 1, width - s + 1);
        let mut o = vec![0.0f32; k * p * q];
        for kk in 0..k {
            for pp in 0..p {
                for qq in 0..q {
                    let mut acc = 0.0f64;
                    for cc in 0..c {
                        for rr in 0..r {
                            for ss in 0..s {
                                let (ih, iw) = (pp + rr, qq + ss);
                                acc += x[(cc * h + ih) * width + iw] as f64
                                    * w[((kk * c + cc) * r + rr) * s + ss] as f64;
                            }
                        }
                    }
                    o[(kk * p + pp) * q + qq] = acc as f32;
                }
            }
        }
        o
    }

    #[test]
    fn conv2d_matches_reference_within_tol() {
        with_gpu("conv2d", |g| {
            let mut rng = crate::diff::Rng::new(0xC0FFEE);
            // (C, H, W, K, R, S) — a 3×3 over 3 channels, and a 5×5 over 16 channels
            let cases = [
                (3usize, 16usize, 16usize, 8usize, 3usize, 3usize),
                (16, 32, 32, 4, 5, 5),
            ];
            for (c, h, width, k, r, s) in cases {
                let x = rng.vec(c * h * width, -1.0, 1.0);
                let w = rng.vec(k * c * r * s, -1.0, 1.0);
                let got = conv2d(g, &x, &w, c, h, width, k, r, s).unwrap();
                let oracle = ref_conv2d(&x, &w, c, h, width, k, r, s);
                // c·√(R·S)·ε accumulation bound, abs cushion for near-zero outputs
                let rel = ((8.0 * ((c * r * s) as f64).sqrt()) * f32::EPSILON as f64).max(1e-4);
                let st = crate::diff::assert_close(
                    &format!("conv2d C{c} {h}x{width} K{k} {r}x{s}"),
                    &got,
                    &oracle,
                    1e-4,
                    rel,
                );
                eprintln!(
                    "conv2d C{c} {h}x{width} K{k} {r}x{s}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// f64 RMSNorm reference over `[rows, cols]`: `out = x / sqrt(mean(x²) + eps)` per row — the
    /// non-affine form the GPU `rmsnorm` kernel computes.
    fn ref_rmsnorm(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let mut o = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mut ms = 0.0f64;
            for i in 0..cols {
                let v = x[r * cols + i] as f64;
                ms += v * v;
            }
            let inv = 1.0 / (ms / cols as f64 + eps as f64).sqrt();
            for i in 0..cols {
                o[r * cols + i] = (x[r * cols + i] as f64 * inv) as f32;
            }
        }
        o
    }

    /// f64 SiLU `x·σ(x)`.
    fn ref_silu(x: f32) -> f32 {
        let x = x as f64;
        (x / (1.0 + (-x).exp())) as f32
    }

    /// CPU f64 reference for the whole pre-norm transformer layer — the independent oracle for
    /// [`transformer_layer`], composing the existing `ref_rmsnorm` / `ref_nt` / `ref_attn` pieces.
    fn ref_transformer_layer(
        x: &[f32],
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
    ) -> Vec<f32> {
        let eps = 1e-5f32;
        let h1 = ref_rmsnorm(x, s, d, eps);
        let q = ref_nt(&h1, w.wq, s, d, d);
        let k = ref_nt(&h1, w.wk, s, d, d);
        let v = ref_nt(&h1, w.wv, s, d, d);
        let a = ref_attn(&q, &k, &v, s, d, 1.0 / (d as f32).sqrt());
        let o = ref_nt(&a, w.wo, s, d, d);
        let x1: Vec<f32> = x.iter().zip(&o).map(|(&a, &b)| a + b).collect();
        let h2 = ref_rmsnorm(&x1, s, d, eps);
        let f1 = ref_nt(&h2, w.w1, s, d, dff);
        let f1act: Vec<f32> = f1.iter().map(|&z| ref_silu(z)).collect();
        let f2 = ref_nt(&f1act, w.w2, s, dff, d);
        x1.iter().zip(&f2).map(|(&a, &b)| a + b).collect()
    }

    /// CPU f64 reference for [`transformer_layer_f16`] — identical to [`ref_transformer_layer`] but
    /// every projection rounds its inputs to f16 first (matching the kernel's `cast`+WMMA f16
    /// fragments). RMSNorm, flash, and the two residual adds stay full-precision (the kernel runs those
    /// f32 / f32-accumulate), so only the four GEMM input quantizations differ from the f32 oracle.
    fn ref_transformer_layer_f16(
        x: &[f32],
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
    ) -> Vec<f32> {
        let eps = 1e-5f32;
        let round = |v: f32| half::f16::from_f32(v).to_f32();
        let h1 = ref_rmsnorm(x, s, d, eps);
        let q = ref_nt_rounded(&h1, w.wq, s, d, d, round);
        let k = ref_nt_rounded(&h1, w.wk, s, d, d, round);
        let v = ref_nt_rounded(&h1, w.wv, s, d, d, round);
        let a = ref_attn(&q, &k, &v, s, d, 1.0 / (d as f32).sqrt());
        let o = ref_nt_rounded(&a, w.wo, s, d, d, round);
        let x1: Vec<f32> = x.iter().zip(&o).map(|(&a, &b)| a + b).collect();
        let h2 = ref_rmsnorm(&x1, s, d, eps);
        let f1 = ref_nt_rounded(&h2, w.w1, s, d, dff, round);
        let f1act: Vec<f32> = f1.iter().map(|&z| ref_silu(z)).collect();
        let f2 = ref_nt_rounded(&f1act, w.w2, s, dff, d, round);
        x1.iter().zip(&f2).map(|(&a, &b)| a + b).collect()
    }

    /// CPU f64 reference for [`ResidentModelF16`] — apply the per-layer f16 reference N times, each
    /// layer's output feeding the next (exactly what the resident stack does on the device).
    fn ref_model_f16(
        x: &[f32],
        weights: &[TransformerWeights],
        s: usize,
        d: usize,
        dff: usize,
    ) -> Vec<f32> {
        let mut cur = x.to_vec();
        for w in weights {
            cur = ref_transformer_layer_f16(&cur, w, s, d, dff);
        }
        cur
    }

    /// CPU f64 reference for [`ffn_fused`] — RMSNorm, then the two projections with **f16-rounded
    /// inputs** (matching the kernel's cast + WMMA f16 fragments) and SiLU between, then the f32
    /// residual added in full precision (the kernel seeds it via wmma.load.c, also f32).
    fn ref_ffn(x: &[f32], w1: &[f32], w2: &[f32], s: usize, d: usize, dff: usize) -> Vec<f32> {
        let eps = 1e-5f32;
        let round = |v: f32| half::f16::from_f32(v).to_f32();
        let h2 = ref_rmsnorm(x, s, d, eps);
        let f1 = ref_nt_rounded(&h2, w1, s, d, dff, round);
        let f1a: Vec<f32> = f1.iter().map(|&z| ref_silu(z)).collect();
        let f2 = ref_nt_rounded(&f1a, w2, s, dff, d, round);
        x.iter().zip(&f2).map(|(&a, &b)| a + b).collect()
    }

    /// [`ffn_fused`] (GPU-resident, fp16 fused-epilogue FFN) vs its f64 reference. Small weights keep
    /// activations O(1) (RMSNorm gives unit-RMS rows) so the fp16 error stays well inside tolerance.
    /// Also checks determinism (fixed grids + warp-butterfly reductions ⇒ bit-identical run-to-run).
    #[test]
    fn ffn_fused_matches_reference_within_tol() {
        with_gpu("ffn_fused", |g| {
            let mut rng = crate::diff::Rng::new(0x55FF);
            let (s, d, dff) = (128usize, 64usize, 256usize);
            let x = rng.vec(s * d, -1.0, 1.0);
            let w1 = rng.vec(dff * d, -0.1, 0.1);
            let w2 = rng.vec(d * dff, -0.1, 0.1);
            let got = ffn_fused(g, &x, &w1, &w2, s, d, dff).unwrap();
            let oracle = ref_ffn(&x, &w1, &w2, s, d, dff);
            let st = crate::diff::assert_close("ffn_fused", &got, &oracle, 3e-2, 3e-2);
            eprintln!(
                "ffn_fused S={s} D={d} Dff={dff} (GPU-resident, 5 launches): max_abs={:.2e} max_rel={:.2e}",
                st.max_abs, st.max_rel
            );
            let again = ffn_fused(g, &x, &w1, &w2, s, d, dff).unwrap();
            let bits = |v: &[f32]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&again), "ffn_fused must be bit-reproducible run-to-run");
        });
    }

    #[test]
    fn transformer_layer_matches_reference_within_tol() {
        with_gpu("transformer_layer", |g| {
            let mut rng = crate::diff::Rng::new(0x7A11);
            let (s, d, dff) = (96usize, 64usize, 256usize);
            let x = rng.vec(s * d, -1.0, 1.0);
            // Small weights keep activations O(1) (RMSNorm makes rows unit-RMS), so the layer is
            // well-conditioned and the cross-check is a tight tolerance.
            let wq = rng.vec(d * d, -0.1, 0.1);
            let wk = rng.vec(d * d, -0.1, 0.1);
            let wv = rng.vec(d * d, -0.1, 0.1);
            let wo = rng.vec(d * d, -0.1, 0.1);
            let w1 = rng.vec(dff * d, -0.1, 0.1);
            let w2 = rng.vec(d * dff, -0.1, 0.1);
            let w = TransformerWeights {
                wq: &wq,
                wk: &wk,
                wv: &wv,
                wo: &wo,
                w1: &w1,
                w2: &w2,
            };
            let got = transformer_layer(g, &x, &w, s, d, dff).unwrap();
            let oracle = ref_transformer_layer(&x, &w, s, d, dff);
            let st = crate::diff::assert_close("transformer_layer", &got, &oracle, 2e-2, 2e-2);
            eprintln!(
                "transformer_layer S={s} D={d} Dff={dff} (GPU-resident): max_abs={:.2e} max_rel={:.2e}",
                st.max_abs, st.max_rel
            );
            // Determinism: identical inputs → bit-identical output (every kernel uses a fixed grid +
            // warp-butterfly reductions, no atomics), so the layer is reproducible run-to-run.
            let again = transformer_layer(g, &x, &w, s, d, dff).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&got),
                bits(&again),
                "GPU-resident transformer layer must be deterministic"
            );
        });
    }

    /// [`transformer_layer_f16`] (the fp16 tensor-core, fused-epilogue layer) vs its f64 reference with
    /// f16-rounded GEMM inputs. Small weights keep activations O(1) (RMSNorm gives unit-RMS rows) so the
    /// fp16 error across the four projections + flash + two residuals stays well inside tolerance. Also
    /// asserts run-to-run bit-reproducibility (fixed grids + warp-butterfly reductions, no atomics).
    #[test]
    fn transformer_layer_f16_matches_reference_within_tol() {
        with_gpu("transformer_layer_f16", |g| {
            let mut rng = crate::diff::Rng::new(0x7A13);
            let (s, d, dff) = (128usize, 64usize, 256usize); // S,D,Dff all %64 for the WMMA tile
            let x = rng.vec(s * d, -1.0, 1.0);
            let wq = rng.vec(d * d, -0.1, 0.1);
            let wk = rng.vec(d * d, -0.1, 0.1);
            let wv = rng.vec(d * d, -0.1, 0.1);
            let wo = rng.vec(d * d, -0.1, 0.1);
            let w1 = rng.vec(dff * d, -0.1, 0.1);
            let w2 = rng.vec(d * dff, -0.1, 0.1);
            let w = TransformerWeights {
                wq: &wq,
                wk: &wk,
                wv: &wv,
                wo: &wo,
                w1: &w1,
                w2: &w2,
            };
            let got = transformer_layer_f16(g, &x, &w, s, d, dff).unwrap();
            let oracle = ref_transformer_layer_f16(&x, &w, s, d, dff);
            let st = crate::diff::assert_close("transformer_layer_f16", &got, &oracle, 5e-2, 5e-2);
            eprintln!(
                "transformer_layer_f16 S={s} D={d} Dff={dff} (fp16 tensor-core, fused residual+SiLU, 13 launches): max_abs={:.2e} max_rel={:.2e}",
                st.max_abs, st.max_rel
            );
            let again = transformer_layer_f16(g, &x, &w, s, d, dff).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&got),
                bits(&again),
                "fp16 GPU-resident transformer layer must be deterministic"
            );
        });
    }

    /// [`ResidentModelF16`] (a stack of N fp16 layers, whole-model GPU-resident) vs its f64 reference
    /// (the per-layer f16 reference applied N times). Each layer's RMSNorm re-normalizes the residual
    /// stream, so the fp16 error stays bounded across depth rather than compounding — a 4-layer stack is
    /// still a tight tolerance. Also asserts the whole model is bit-reproducible run-to-run.
    #[test]
    fn resident_model_f16_matches_reference_within_tol() {
        with_gpu("resident_model_f16", |g| {
            let mut rng = crate::diff::Rng::new(0x7A14);
            let (s, d, dff, depth) = (128usize, 64usize, 256usize, 4usize);
            let x = rng.vec(s * d, -1.0, 1.0);
            // N independent layers' weights (owned), then borrowed into TransformerWeights.
            let wdata: Vec<[Vec<f32>; 6]> = (0..depth)
                .map(|_| {
                    [
                        rng.vec(d * d, -0.1, 0.1),
                        rng.vec(d * d, -0.1, 0.1),
                        rng.vec(d * d, -0.1, 0.1),
                        rng.vec(d * d, -0.1, 0.1),
                        rng.vec(dff * d, -0.1, 0.1),
                        rng.vec(d * dff, -0.1, 0.1),
                    ]
                })
                .collect();
            let weights: Vec<TransformerWeights> = wdata
                .iter()
                .map(|wl| TransformerWeights {
                    wq: wl[0].as_slice(),
                    wk: wl[1].as_slice(),
                    wv: wl[2].as_slice(),
                    wo: wl[3].as_slice(),
                    w1: wl[4].as_slice(),
                    w2: wl[5].as_slice(),
                })
                .collect();
            let model = ResidentModelF16::new(g, &weights, s, d, dff).unwrap();
            let got = model.forward(&x).unwrap();
            let oracle = ref_model_f16(&x, &weights, s, d, dff);
            // Tight ABSOLUTE bound (the residual stream is O(1) and RMSNorm bounds error growth, so 4
            // layers land at ~5e-4 max_abs); rel is a loose fallback only for near-zero elements where a
            // tiny abs diff blows up the ratio.
            let st = crate::diff::assert_close("resident_model_f16", &got, &oracle, 5e-3, 1e-1);
            eprintln!(
                "resident_model_f16 depth={depth} S={s} D={d} Dff={dff} (whole-model GPU-resident, {}×13 launches, 1 H2D + 1 D2H): max_abs={:.2e} max_rel={:.2e}",
                depth, st.max_abs, st.max_rel
            );
            let again = model.forward(&x).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&again), "resident model must be deterministic run-to-run");
        });
    }

    /// **M12 — every GPU kernel is bit-identical run-to-run.** Mercury's kernels use fixed grids and
    /// atomic-free, fixed-order reductions (warp-butterfly shuffles / a fixed ascending host combine),
    /// so identical inputs yield identical *bits* every time — a contract cuBLAS does not offer (its
    /// heuristically-selected algorithms and split-K atomic accumulation are reproducible only
    /// incidentally, never guaranteed across shapes, library versions, or GPU architecture).
    /// Reproducibility is load-bearing for regression gates, debugging, and regulated training. Here we
    /// assert it directly across every kernel family that performs a reduction — the only place
    /// nondeterminism could creep in; elementwise kernels are included for completeness. (The whole
    /// fused layer is separately covered by `transformer_layer_matches_reference_within_tol`, and the
    /// reductions by `reductions_match_reference_within_tol_and_are_deterministic`.)
    #[test]
    fn gpu_kernels_bit_reproducible() {
        use mercury_runtime::{NORM_LAYERNORM, NORM_RMSNORM, NORM_SOFTMAX, RED_DOT, RED_SUM, VM_GELU};
        with_gpu("bit_reproducible", |g| {
            let mut rng = crate::diff::Rng::new(0xD37E);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            // Run a kernel twice on identical inputs; the two outputs must be bitwise identical.
            macro_rules! twice_eq {
                ($label:literal, $call:expr) => {{
                    let a = $call;
                    let b = $call;
                    assert_eq!(bits(&a), bits(&b), concat!($label, " must be bit-reproducible"));
                }};
            }

            // GEMM — every WMMA variant (each C element accumulates K in a fixed per-thread order).
            // 64-aligned M,N and 16-aligned K so the SMEM-staged variants accept the shape.
            let (m, k, n) = (128usize, 96usize, 128usize);
            let a = rng.vec(m * k, -1.0, 1.0);
            let bmat = rng.vec(n * k, -1.0, 1.0);
            twice_eq!("gemm_nt_f16", gemm_nt_f16(g, &a, &bmat, m, k, n).unwrap());
            twice_eq!("gemm_nt_f16_sm", gemm_nt_f16_sm(g, &a, &bmat, m, k, n).unwrap());
            twice_eq!("gemm_nt_f16_sm_db", gemm_nt_f16_sm_db(g, &a, &bmat, m, k, n).unwrap());

            // Fused row norms — one warp/row, shfl-butterfly reduction over a fixed lane order.
            let (rows, cols) = (40usize, 128usize);
            let xn = rng.vec(rows * cols, -3.0, 3.0);
            for op in [NORM_SOFTMAX, NORM_LAYERNORM, NORM_RMSNORM] {
                twice_eq!("norm", norm(g, op, &xn, rows, cols, 1e-5).unwrap());
            }

            // Flash-attention — online softmax streamed over K/V in fixed tile order.
            let (seq, d) = (64usize, 64usize);
            let q = rng.vec(seq * d, -1.0, 1.0);
            let kk = rng.vec(seq * d, -1.0, 1.0);
            let vv = rng.vec(seq * d, -1.0, 1.0);
            twice_eq!("flash_attn", flash_attn(g, &q, &kk, &vv, seq, d, 0.125).unwrap());

            // Conv2d — one thread/output, fixed C·R·S fma order.
            let (c, h, wd, kc, r, s) = (3usize, 16usize, 16usize, 4usize, 3usize, 3usize);
            let xc = rng.vec(c * h * wd, -1.0, 1.0);
            let wc = rng.vec(kc * c * r * s, -1.0, 1.0);
            twice_eq!("conv2d", conv2d(g, &xc, &wc, c, h, wd, kc, r, s).unwrap());

            // Reductions — fixed grid + fixed ascending host combine.
            let xr = rng.vec(1 << 16, 0.0, 1.0);
            let yr = rng.vec(1 << 16, 0.0, 1.0);
            twice_eq!("reduce_sum", vec![reduce(g, RED_SUM, &xr, None).unwrap()]);
            twice_eq!("reduce_dot", vec![reduce(g, RED_DOT, &xr, Some(&yr)).unwrap()]);

            // Elementwise — trivially deterministic, included for completeness.
            twice_eq!("vmath_gelu", vmath(g, VM_GELU, &xr).unwrap());
            twice_eq!("copy", copy(g, &xr).unwrap());

            eprintln!("M12: all GPU kernel families bit-identical run-to-run ✓");
        });
    }

    /// **M12 logged win — Mercury is reproducible where cuBLAS makes no such promise.** Asserts
    /// Mercury's fp16 GEMM is bit-identical across three runs (guaranteed by construction), then logs
    /// whether cuBLAS is too. cuBLAS may *happen* to be bit-stable for a fixed shape/version on a fixed
    /// device, but NVIDIA documents no reproducibility contract across library versions, GPU
    /// architectures, or its heuristic algorithm selection — so the *guarantee*, not the incidental
    /// match, is the win. Needs the CUDA redist DLLs on PATH (see `gemm_vs_peers`); skips otherwise.
    #[test]
    #[ignore = "needs CUDA redist DLLs on PATH; run explicitly"]
    fn reproducibility_vs_cublas() {
        use crate::baselines::{cublas_gemm_nt_f16, peer_env_hint, peers_available};
        with_gpu("repro_vs_cublas", |g| {
            if !peers_available(g) {
                eprintln!("[skip] reproducibility_vs_cublas: cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            let mut rng = crate::diff::Rng::new(0xC0FFEE);
            let (m, k, n) = (1024usize, 1024usize, 1024usize);
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

            let mer: Vec<_> = (0..3).map(|_| gemm_nt_f16_sm_db(g, &a, &b, m, k, n).unwrap()).collect();
            let mer_stable = bits(&mer[0]) == bits(&mer[1]) && bits(&mer[1]) == bits(&mer[2]);
            assert!(mer_stable, "Mercury GEMM must be bit-identical run-to-run");

            let cub: Vec<_> = (0..3).map(|_| cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap()).collect();
            let cub_stable = bits(&cub[0]) == bits(&cub[1]) && bits(&cub[1]) == bits(&cub[2]);

            eprintln!("reproducibility @ {m}³ fp16 (3 runs, same buffers):");
            eprintln!("  Mercury: bit-identical = {mer_stable}  (GUARANTEED — fixed grid, no atomics, fixed K order)");
            eprintln!("  cuBLAS : bit-identical = {cub_stable}  (incidental — no cross-version/arch/heuristic contract)");
        });
    }

    /// Latency of the GPU transformer layer, **fp16 tensor-core fused vs f32 scalar chain** — both
    /// Mercury, same machine, same buffers, measured same-run (an internal optimized-vs-naive speedup,
    /// NOT a cross-vendor claim). Three figures: (1) f32 full call and (2) fp16 full call both pay the
    /// per-call weights-H2D + result-D2H; (3) fp16 **resident** times only the on-device kernel chain
    /// from a [`ResidentLayerF16`] whose weights uploaded once — the real serving cost (a deployed model
    /// never re-uploads weights). The fp16 path runs every projection on the WMMA tensor cores and folds
    /// the two residual adds + the SiLU into the GEMMs; the f32 path is the register-blocked scalar GEMM
    /// chain with separate add/SiLU launches. The laptop GPU clock is warmed first and each figure is
    /// `best_of(ROUNDS)` (min time = peak-clock sample), or the ~7× boost ramp would corrupt the ratio.
    /// Each size is also correctness-gated against the f64 oracle before its speed is reported.
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn transformer_layer_throughput() {
        with_gpu("transformer_layer_throughput", |g| {
            let mut rng = crate::diff::Rng::new(0x7A12);
            let (d, dff) = (64usize, 256usize);

            // Pin the laptop GPU clock with a sustained fp16-GEMM burst. The clock boosts ~7× under load
            // and DECAYS in the gaps between measurements, so a single up-front warmup is not enough over
            // a multi-second bench — re-pinning immediately before each timing keeps all three (f32 /
            // fp16 full / fp16 resident) in the same clock regime, the only way the ratios are honest
            // (the "GPU clock" lesson: on this part only same-regime ratios are meaningful).
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin_clock {
                () => {{
                    for _ in 0..40 {
                        gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
                    }
                }};
            }
            // Heavy initial boost from cold (the clock ramps over hundreds of ms of sustained load); the
            // per-measurement pins then bridge the short host gaps so all three stay in one clock regime.
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            for s in [256usize, 512, 1024] {
                let x = rng.vec(s * d, -1.0, 1.0);
                let wq = rng.vec(d * d, -0.1, 0.1);
                let wk = rng.vec(d * d, -0.1, 0.1);
                let wv = rng.vec(d * d, -0.1, 0.1);
                let wo = rng.vec(d * d, -0.1, 0.1);
                let w1 = rng.vec(dff * d, -0.1, 0.1);
                let w2 = rng.vec(d * dff, -0.1, 0.1);
                let w = TransformerWeights {
                    wq: &wq,
                    wk: &wk,
                    wv: &wv,
                    wo: &wo,
                    w1: &w1,
                    w2: &w2,
                };
                // JIT + module-cache warmup for both paths, then correctness-gate the fp16 path at this
                // shape (speed is only reported for an output that matches the f64 oracle).
                transformer_layer(g, &x, &w, s, d, dff).unwrap();
                let got16 = transformer_layer_f16(g, &x, &w, s, d, dff).unwrap();
                let oracle = ref_transformer_layer_f16(&x, &w, s, d, dff);
                crate::diff::assert_close(&format!("tl_f16 S={s}"), &got16, &oracle, 5e-2, 5e-2);

                // Build the resident fp16 layer ONCE (weights → f16, uploaded once) and pin x on-device;
                // the struct owns a cloned stream Arc, not a borrow of `g`, so the full-call timings below
                // (which take `&mut g`) still compile while it is alive.
                let layer = ResidentLayerF16::new(g, &w, s, d, dff).unwrap();
                let x_d = g.stream.memcpy_stod(&x).unwrap();
                layer.forward_device(&x_d).unwrap(); // warm

                let iters = 60;
                pin_clock!();
                let t_f32 = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        transformer_layer(g, &x, &w, s, d, dff).unwrap();
                    }
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                pin_clock!();
                let t_f16 = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        transformer_layer_f16(g, &x, &w, s, d, dff).unwrap();
                    }
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                // Resident per-token compute: only the on-device kernel chain (weights AND x already
                // resident, output stays resident). The launches are async, so sync once before stopping
                // the clock. This is the real serving cost — a deployed model never re-uploads weights.
                pin_clock!();
                let t_res = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = layer.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                eprintln!(
                    "transformer_layer S={s} D={d} Dff={dff} (same-run, both Mercury):\n  \
                     f32  scalar chain,    full call : {:.3} ms, {:>7.0} tok/s\n  \
                     fp16 tensor-core fused, full call : {:.3} ms, {:>7.0} tok/s  → {:.2}× vs f32\n  \
                     fp16 tensor-core fused, RESIDENT  : {:.3} ms, {:>7.0} tok/s  → {:.2}× vs f32 (weights+x resident, no D2H)",
                    t_f32 * 1e3,
                    s as f64 / t_f32,
                    t_f16 * 1e3,
                    s as f64 / t_f16,
                    t_f32 / t_f16,
                    t_res * 1e3,
                    s as f64 / t_res,
                    t_f32 / t_res,
                );
            }
        });
    }

    /// **Whole-model GPU residency scales linearly and amortizes transfer** — [`ResidentModelF16`] across
    /// depths 1/2/4/8. Each depth reports the resident forward (the whole N-layer chain on device, one
    /// sync) as total ms, ms/layer, and tok/s, plus the full-call total (one H2D + one D2H for the entire
    /// stack, weights already resident). The point: resident **ms/layer is flat** across depth (no
    /// per-layer host overhead — pure on-device chaining), and the full-call's fixed activation transfer
    /// amortizes, so full→resident converges as depth grows. Clock re-pinned before each measurement.
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn resident_model_throughput() {
        with_gpu("resident_model_throughput", |g| {
            let mut rng = crate::diff::Rng::new(0x7A15);
            let (s, d, dff) = (512usize, 64usize, 256usize);
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin_clock {
                () => {{
                    for _ in 0..40 {
                        gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
                    }
                }};
            }
            // Heavy initial boost from cold: the laptop GPU clock ramps over hundreds of ms of sustained
            // load, so a handful of GEMMs won't do it (a too-light warmup left depth=1 reading ~8× slow).
            // Once boosted, the measurements themselves sustain it and the per-measurement pins bridge the
            // short host gaps.
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            for depth in [1usize, 2, 4, 8] {
                let wdata: Vec<[Vec<f32>; 6]> = (0..depth)
                    .map(|_| {
                        [
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(dff * d, -0.1, 0.1),
                            rng.vec(d * dff, -0.1, 0.1),
                        ]
                    })
                    .collect();
                let weights: Vec<TransformerWeights> = wdata
                    .iter()
                    .map(|wl| TransformerWeights {
                        wq: wl[0].as_slice(),
                        wk: wl[1].as_slice(),
                        wv: wl[2].as_slice(),
                        wo: wl[3].as_slice(),
                        w1: wl[4].as_slice(),
                        w2: wl[5].as_slice(),
                    })
                    .collect();
                let x = rng.vec(s * d, -1.0, 1.0);
                let model = ResidentModelF16::new(g, &weights, s, d, dff).unwrap();
                model.forward(&x).unwrap(); // warm (JIT + caches)
                let x_d = g.stream.memcpy_stod(&x).unwrap();
                model.forward_device(&x_d).unwrap();

                let iters = 40;
                pin_clock!();
                let t_full = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        model.forward(&x).unwrap();
                    }
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                pin_clock!();
                let t_res = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = model.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let dl = depth as f64;
                eprintln!(
                    "resident_model depth={depth} S={s} D={d} Dff={dff}: \
                     full {:.3} ms ({:.3} ms/layer) | resident {:.3} ms ({:.3} ms/layer, {:.0} tok/s)",
                    t_full * 1e3,
                    t_full * 1e3 / dl,
                    t_res * 1e3,
                    t_res * 1e3 / dl,
                    s as f64 / t_res,
                );
            }
        });
    }

    /// **M13 milestone — Mercury's whole-model resident stack vs the cuBLAS call-chain stack, end-to-end
    /// across depth** (the prompt's "beat the library-call-chain stack end-to-end", the multi-layer
    /// completion of the single-layer `cublas_chain_vs_mercury_layer_throughput`). At each depth N both run
    /// the SAME N weight sets: [`ResidentModelF16`] (whole model GPU-resident, fused epilogues) vs
    /// [`CublasChainModel`](crate::baselines::CublasChainModel) (N cuBLAS-chain layers — cuBLAS GEMM +
    /// separate add/SiLU, **identical** norm/flash/cast glue). Resident `forward_device` timed same-run
    /// (the one H2D/D2H amortizes away), reported as ms/layer so depth-invariance and the per-layer gap
    /// (paid N times) are both visible. Both stacks are cross-checked against the f64 `ref_model_f16`
    /// oracle end-to-end before any speed is reported (the fp16 error stays bounded across depth — each
    /// layer's RMSNorm re-normalizes the residual stream). Clock warmed + `best_of`. Needs the CUDA redist
    /// DLLs on PATH (see `gemm_vs_peers`); skips otherwise. Run: `cargo test -p mercury_codegen_gpu
    /// --features gpu --release -- --ignored --nocapture resident_model_vs_cublas`.
    #[test]
    #[ignore = "needs CUDA redist DLLs on PATH; throughput bench; run explicitly"]
    fn resident_model_vs_cublas_chain_throughput() {
        use crate::baselines::{peer_env_hint, peers_available, CublasChainModel};
        with_gpu("resident_model_vs_cublas", |g| {
            if !peers_available(g) {
                eprintln!(
                    "[skip] resident_model_vs_cublas_chain_throughput: cuBLAS not loadable.\n{}",
                    peer_env_hint()
                );
                return;
            }
            let mut rng = crate::diff::Rng::new(0x0CB3);
            let (s, d, dff) = (512usize, 64usize, 256usize);
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin_clock {
                () => {{
                    for _ in 0..40 {
                        gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
                    }
                }};
            }
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            for depth in [1usize, 2, 4, 8] {
                let wdata: Vec<[Vec<f32>; 6]> = (0..depth)
                    .map(|_| {
                        [
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(d * d, -0.1, 0.1),
                            rng.vec(dff * d, -0.1, 0.1),
                            rng.vec(d * dff, -0.1, 0.1),
                        ]
                    })
                    .collect();
                let weights: Vec<TransformerWeights> = wdata
                    .iter()
                    .map(|wl| TransformerWeights {
                        wq: wl[0].as_slice(),
                        wk: wl[1].as_slice(),
                        wv: wl[2].as_slice(),
                        wo: wl[3].as_slice(),
                        w1: wl[4].as_slice(),
                        w2: wl[5].as_slice(),
                    })
                    .collect();
                let x = rng.vec(s * d, -1.0, 1.0);
                let mer = ResidentModelF16::new(g, &weights, s, d, dff).unwrap();
                let chain = CublasChainModel::new(g, &weights, s, d, dff).unwrap();

                // Correctness before speed: both stacks match the f64 oracle end-to-end at this depth.
                let oracle = ref_model_f16(&x, &weights, s, d, dff);
                let mer_out = mer.forward(&x).unwrap();
                let chain_out = chain.forward(&x).unwrap();
                crate::diff::assert_close(
                    &format!("Mercury model depth={depth}"),
                    &mer_out,
                    &oracle,
                    5e-3,
                    1e-1,
                );
                crate::diff::assert_close(
                    &format!("cuBLAS chain model depth={depth}"),
                    &chain_out,
                    &oracle,
                    5e-3,
                    1e-1,
                );

                let x_d = g.stream.memcpy_stod(&x).unwrap();
                let iters = 40;
                pin_clock!();
                let t_mer = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = mer.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                pin_clock!();
                let t_chain = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = chain.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let dl = depth as f64;
                eprintln!(
                    "M13 stack depth={depth} S={s} D={d} Dff={dff} (resident, same-run): \
                     Mercury {:.3} ms ({:.3} ms/layer, {:.0} tok/s) | cuBLAS chain {:.3} ms ({:.3} ms/layer) \
                     || Mercury {:.2}× faster end-to-end",
                    t_mer * 1e3,
                    t_mer * 1e3 / dl,
                    s as f64 / t_mer,
                    t_chain * 1e3,
                    t_chain * 1e3 / dl,
                    t_chain / t_mer,
                );
            }
        });
    }

    /// **Diagnostic: which kernel makes the layer scale with S** (the PyTorch gap, M13). The fp16 fused
    /// layer's per-layer time grows ~linearly with S while PyTorch eager's is flat — so *something* in the
    /// layer scales badly. This times the kernels **resident** (buffers uploaded once, the kernel
    /// re-launched in a tight loop, a single trailing sync — so it isolates *pure kernel compute*, no
    /// per-call H2D/D2H), across S: the **flash attention** (one warp per query row, serial over keys —
    /// `grid=(S,1,1)`, `block=32`, so O(S) latency/row *and* one warp per CTA = poor occupancy) vs a
    /// representative **WMMA GEMM** (the FFN up-projection `S×64×256`, throughput-bound). Steeper flash
    /// scaling ⇒ the attention kernel is the lever to match PyTorch's tiled SDPA; comparable scaling ⇒ the
    /// gap is broad small-op/occupancy overhead and the lever is the megakernel (fewer launches). Run:
    /// `… --ignored --nocapture flash_vs_gemm_scaling`.
    #[test]
    #[ignore = "diagnostic bench; run explicitly"]
    fn flash_vs_gemm_scaling() {
        use half::f16;
        with_gpu("flash_vs_gemm_scaling", |g| {
            let mut rng = crate::diff::Rng::new(0x0CB4);
            let (d, dff) = (64usize, 256usize);
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin {
                () => {{
                    for _ in 0..40 {
                        gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
                    }
                }};
            }
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            let f_flash = g.function("flash", crate::ptx_flash::flash_ptx(), "flash_d64").unwrap();
            let f_gemm = g
                .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm_db")
                .unwrap();
            let to16 = |v: &[f32]| -> Vec<f16> { v.iter().map(|&x| f16::from_f32(x)).collect() };
            for s in [256usize, 512, 1024] {
                // resident buffers — uploaded once, reused every launch (no per-call transfer/alloc).
                let q = g.stream.memcpy_stod(&rng.vec(s * d, -1.0, 1.0)).unwrap();
                let k = g.stream.memcpy_stod(&rng.vec(s * d, -1.0, 1.0)).unwrap();
                let v = g.stream.memcpy_stod(&rng.vec(s * d, -1.0, 1.0)).unwrap();
                let mut o = g.stream.memcpy_stod(&vec![0f32; s * d]).unwrap();
                let a16 = g.stream.memcpy_stod(&to16(&rng.vec(s * d, -1.0, 1.0))).unwrap();
                let b16 = g.stream.memcpy_stod(&to16(&rng.vec(dff * d, -1.0, 1.0))).unwrap();
                let mut c = g.stream.memcpy_stod(&vec![0f32; s * dff]).unwrap();
                let scale = 1.0f32 / (d as f32).sqrt();
                let ss = s as u32;
                let flash_cfg = LaunchConfig { grid_dim: (ss, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
                pin!();
                let t_flash = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_flash);
                        bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                        unsafe { bld.launch(flash_cfg).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                let (mm, nn, kk) = (s as u32, dff as u32, d as u32);
                let gemm_cfg = wmma_sm_cfg(s, dff);
                pin!();
                let t_gemm = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_gemm);
                        bld.arg(&mm).arg(&nn).arg(&kk).arg(&a16).arg(&b16).arg(&mut c);
                        unsafe { bld.launch(gemm_cfg).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                eprintln!(
                    "S={s} (resident, pure kernel): flash {:.4} ms | WMMA gemm S×{d}×{dff} {:.4} ms | flash/gemm {:.1}×",
                    t_flash * 1e3,
                    t_gemm * 1e3,
                    t_flash / t_gemm,
                );
            }
        });
    }

    /// **What fusion buys at the full-layer level** — the same [`ResidentLayerF16`], resident, timed two
    /// ways same-run: `forward_device` (the 2 residual adds + SiLU folded into the GEMMs) vs
    /// `forward_device_unfused` (identical WMMA GEMMs but 3 separate `vadd`/`silu` launches, each
    /// round-tripping a tensor through HBM — the naive call-chain structure). Same GEMM kernels both
    /// sides, so this isolates the FUSION win (not GEMM quality): the canonical fused-vs-unfused report.
    /// Both paths are first cross-checked against the f64 oracle (correctness before speed); clock is
    /// re-pinned before each measurement.
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn layer_fusion_vs_unfused_throughput() {
        with_gpu("layer_fusion_vs_unfused", |g| {
            let mut rng = crate::diff::Rng::new(0x7A16);
            let (d, dff) = (64usize, 256usize);
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin_clock {
                () => {{
                    for _ in 0..40 {
                        gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
                    }
                }};
            }
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            for s in [256usize, 512, 1024] {
                let x = rng.vec(s * d, -1.0, 1.0);
                let wq = rng.vec(d * d, -0.1, 0.1);
                let wk = rng.vec(d * d, -0.1, 0.1);
                let wv = rng.vec(d * d, -0.1, 0.1);
                let wo = rng.vec(d * d, -0.1, 0.1);
                let w1 = rng.vec(dff * d, -0.1, 0.1);
                let w2 = rng.vec(d * dff, -0.1, 0.1);
                let w = TransformerWeights {
                    wq: &wq,
                    wk: &wk,
                    wv: &wv,
                    wo: &wo,
                    w1: &w1,
                    w2: &w2,
                };
                let layer = ResidentLayerF16::new(g, &w, s, d, dff).unwrap();
                let x_d = g.stream.memcpy_stod(&x).unwrap();
                // Correctness before speed: BOTH paths must match the f64 oracle at this shape.
                let oracle = ref_transformer_layer_f16(&x, &w, s, d, dff);
                let fused_out = g.stream.memcpy_dtov(&layer.forward_device(&x_d).unwrap()).unwrap();
                let unfused_out = g.stream.memcpy_dtov(&layer.forward_device_unfused(&x_d).unwrap()).unwrap();
                crate::diff::assert_close(&format!("fused S={s}"), &fused_out, &oracle, 5e-2, 5e-2);
                crate::diff::assert_close(&format!("unfused S={s}"), &unfused_out, &oracle, 5e-2, 5e-2);

                let iters = 60;
                pin_clock!();
                let t_fused = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = layer.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                pin_clock!();
                let t_unf = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = layer.forward_device_unfused(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                eprintln!(
                    "layer_fusion S={s} D={d} Dff={dff} (resident, same-run): \
                     fused {:.3} ms ({:.0} tok/s) | unfused {:.3} ms | fusion {:.2}× (folds 2 residual adds + 1 SiLU)",
                    t_fused * 1e3,
                    s as f64 / t_fused,
                    t_unf * 1e3,
                    t_unf / t_fused,
                );
            }
        });
    }

    /// **Correctness gate for the Tier-B cuBLAS call-chain peer** ([`CublasChainLayer`]) — law #1, before
    /// any speed number. (a) the f16-in/f32-out `cublasGemmEx` primitive vs the f64 reference catches a
    /// transpose/dtype slip in isolation; (b) the whole cuBLAS chain layer vs the **same** f64 oracle the
    /// fused Mercury layer uses; (c) a cross-check that the cuBLAS chain and Mercury's fused WMMA layer
    /// compute the same function — two independent GEMM implementations consuming bit-identical f16 inputs
    /// (the cast kernel is shared) and differing only in f32 accumulation order, hence the √K·ε tolerance.
    /// Needs the CUDA redist DLLs on PATH (see `gemm_vs_peers`); skips otherwise.
    #[test]
    #[ignore = "needs CUDA redist DLLs on PATH; run explicitly"]
    fn cublas_chain_layer_matches_reference_within_tol() {
        use crate::baselines::{
            cublas_gemm_nt_f16_f32out, peer_env_hint, peers_available, CublasChainLayer,
        };
        with_gpu("cublas_chain_layer", |g| {
            if !peers_available(g) {
                eprintln!(
                    "[skip] cublas_chain_layer_matches_reference: cuBLAS not loadable.\n{}",
                    peer_env_hint()
                );
                return;
            }
            let mut rng = crate::diff::Rng::new(0x0CB1);
            let round = |v: f32| half::f16::from_f32(v).to_f32();

            // (a) the f16-in/f32-out GEMM primitive in isolation vs the f64 reference.
            {
                let (m, k, n) = (128usize, 64usize, 64usize);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let got = cublas_gemm_nt_f16_f32out(g, &a, &b, m, k, n).unwrap();
                let oracle = ref_nt_rounded(&a, &b, m, k, n, round);
                let st = crate::diff::assert_close("cublas_f16_f32out", &got, &oracle, 2e-2, 2e-2);
                eprintln!(
                    "cublasGemmEx f16-in/f32-out {m}×{n}×{k}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }

            // (b)+(c) the whole cuBLAS call-chain layer.
            let (s, d, dff) = (128usize, 64usize, 256usize); // S,D,Dff %64 to match the WMMA peer's tiles
            let x = rng.vec(s * d, -1.0, 1.0);
            let wq = rng.vec(d * d, -0.1, 0.1);
            let wk = rng.vec(d * d, -0.1, 0.1);
            let wv = rng.vec(d * d, -0.1, 0.1);
            let wo = rng.vec(d * d, -0.1, 0.1);
            let w1 = rng.vec(dff * d, -0.1, 0.1);
            let w2 = rng.vec(d * dff, -0.1, 0.1);
            let w = TransformerWeights { wq: &wq, wk: &wk, wv: &wv, wo: &wo, w1: &w1, w2: &w2 };

            let chain_out = CublasChainLayer::new(g, &w, s, d, dff).unwrap().forward(&x).unwrap();
            let oracle = ref_transformer_layer_f16(&x, &w, s, d, dff);
            let st = crate::diff::assert_close("cublas_chain", &chain_out, &oracle, 5e-2, 5e-2);

            let mer_out = ResidentLayerF16::new(g, &w, s, d, dff).unwrap().forward(&x).unwrap();
            let st2 =
                crate::diff::assert_close("cublas_chain vs Mercury-fused", &chain_out, &mer_out, 5e-2, 5e-2);

            eprintln!(
                "cuBLAS call-chain layer S={s} D={d} Dff={dff}: vs f64 oracle max_abs={:.2e} max_rel={:.2e} | vs Mercury-fused max_abs={:.2e} max_rel={:.2e}",
                st.max_abs, st.max_rel, st2.max_abs, st2.max_rel
            );
        });
    }

    /// **M13 milestone bench — Mercury's fused fp16 stack vs the cuBLAS GEMM call-chain** (the literal
    /// "library call-chain" the milestone must beat), Tier B, same machine, same buffers, same-run. Three
    /// resident timings per S: Mercury *fused* ([`ResidentLayerF16::forward_device`] — WMMA with the two
    /// residual adds & SiLU folded into the GEMMs), Mercury *unfused*
    /// ([`ResidentLayerF16::forward_device_unfused`] — WMMA with separate add/SiLU launches), and the
    /// *cuBLAS chain* ([`CublasChainLayer`] — cuBLAS GEMM with separate add/SiLU, **identical** norm/flash/
    /// cast glue). The decomposition is the point: **Mercury-unfused vs cuBLAS** isolates GEMM quality (same
    /// glue, only the GEMM differs) and **fused vs unfused** isolates the epilogue fusion — together they
    /// explain the **fused vs cuBLAS** headline. Both projection paths are f16-in/f32-out, so neither pays
    /// a post-GEMM cast. Laptop clock warmed + each figure `best_of(ROUNDS)` (the ~7× boost ramp would
    /// corrupt the ratio otherwise). Correctness-gated against the f64 oracle first. Needs the CUDA redist
    /// DLLs on PATH (see `gemm_vs_peers`); skips otherwise. Run: `cargo test -p mercury_codegen_gpu
    /// --features gpu --release -- --ignored --nocapture cublas_chain_vs_mercury`.
    #[test]
    #[ignore = "needs CUDA redist DLLs on PATH; throughput bench; run explicitly"]
    fn cublas_chain_vs_mercury_layer_throughput() {
        use crate::baselines::{peer_env_hint, peers_available, CublasChainLayer};
        with_gpu("cublas_chain_vs_mercury", |g| {
            if !peers_available(g) {
                eprintln!(
                    "[skip] cublas_chain_vs_mercury_layer_throughput: cuBLAS not loadable.\n{}",
                    peer_env_hint()
                );
                return;
            }
            let mut rng = crate::diff::Rng::new(0x0CB2);
            let (d, dff) = (64usize, 256usize);
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin_clock {
                () => {{
                    for _ in 0..40 {
                        gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
                    }
                }};
            }
            // Heavy initial warmup: a light pin does not boost a cold laptop GPU (see the M13 harness note).
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            for s in [256usize, 512, 1024] {
                let x = rng.vec(s * d, -1.0, 1.0);
                let wq = rng.vec(d * d, -0.1, 0.1);
                let wk = rng.vec(d * d, -0.1, 0.1);
                let wv = rng.vec(d * d, -0.1, 0.1);
                let wo = rng.vec(d * d, -0.1, 0.1);
                let w1 = rng.vec(dff * d, -0.1, 0.1);
                let w2 = rng.vec(d * dff, -0.1, 0.1);
                let w = TransformerWeights { wq: &wq, wk: &wk, wv: &wv, wo: &wo, w1: &w1, w2: &w2 };
                let mer = ResidentLayerF16::new(g, &w, s, d, dff).unwrap();
                let chain = CublasChainLayer::new(g, &w, s, d, dff).unwrap();
                let x_d = g.stream.memcpy_stod(&x).unwrap();

                // Correctness before speed: both stacks must match the f64 oracle at this shape.
                let oracle = ref_transformer_layer_f16(&x, &w, s, d, dff);
                let mer_out = g.stream.memcpy_dtov(&mer.forward_device(&x_d).unwrap()).unwrap();
                let chain_out = g.stream.memcpy_dtov(&chain.forward_device(&x_d).unwrap()).unwrap();
                crate::diff::assert_close(&format!("Mercury fused S={s}"), &mer_out, &oracle, 5e-2, 5e-2);
                crate::diff::assert_close(&format!("cuBLAS chain S={s}"), &chain_out, &oracle, 5e-2, 5e-2);

                let iters = 60;
                pin_clock!();
                let t_fused = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = mer.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                pin_clock!();
                let t_unf = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = mer.forward_device_unfused(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                pin_clock!();
                let t_chain = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = chain.forward_device(&x_d).unwrap();
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });

                // Honest headline either direction: report whichever stack is faster, with the magnitude.
                let (hdl_lbl, hdl) = if t_chain >= t_fused {
                    ("Mercury fused faster", t_chain / t_fused)
                } else {
                    ("cuBLAS chain faster", t_fused / t_chain)
                };
                eprintln!(
                    "M13 layer S={s} D={d} Dff={dff} (resident, same-run): \
                     Mercury fused {:.3} ms ({:.0} tok/s) | Mercury unfused {:.3} ms | cuBLAS chain {:.3} ms \
                     || {hdl_lbl} {hdl:.2}× | GEMM-quality (Mercury-unfused vs cuBLAS) {:.2}× | fusion {:.2}×",
                    t_fused * 1e3,
                    s as f64 / t_fused,
                    t_unf * 1e3,
                    t_chain * 1e3,
                    t_chain / t_unf,
                    t_unf / t_fused,
                );
            }
        });
    }

    #[test]
    fn fp8_tensorcore_tile_matches_reference() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_tile", |g| {
            // Asymmetric, e4m3-exact integer data — a layout bug can't hide (all-ones would).
            let a: Vec<f32> = (0..16 * 32)
                .map(|t| {
                    let (i, k) = (t / 32, t % 32);
                    ((i + 2 * k) % 7) as f32
                })
                .collect();
            // B column-major: b_col[j*32 + k] = B[k][j].
            let b_col: Vec<f32> = (0..32 * 8)
                .map(|t| {
                    let (j, k) = (t / 32, t % 32);
                    ((3 * k + j) % 5) as f32
                })
                .collect();
            let got = fp8_tile(g, &a, &b_col).unwrap();
            // reference: C[i][j] = Σ_k e4m3(A[i][k])·e4m3(B[k][j]) (exact at these integer magnitudes)
            let mut refc = vec![0.0f32; 16 * 8];
            for i in 0..16 {
                for j in 0..8 {
                    let mut acc = 0.0f64;
                    for k in 0..32 {
                        let av = e4m3_to_f32(f32_to_e4m3(a[i * 32 + k])) as f64;
                        let bv = e4m3_to_f32(f32_to_e4m3(b_col[j * 32 + k])) as f64;
                        acc += av * bv;
                    }
                    refc[i * 8 + j] = acc as f32;
                }
            }
            let st = crate::diff::assert_close("fp8_tile", &got, &refc, 1e-3, 1e-3);
            eprintln!(
                "fp8_tile m16n8k32 (E4M3 tensor core): max_abs={:.2e} max_rel={:.2e}",
                st.max_abs, st.max_rel
            );
        });
    }

    #[test]
    fn fp8_gemm_matches_reference_within_tol() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_gemm", |g| {
            let mut rng = crate::diff::Rng::new(0x00F8);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            // e4m3 products are exact in f32 (4 sig bits × 4 = 8 ≤ 23), so vs an e4m3-rounded-input
            // reference only the f32 accumulation reassociates → a tight tolerance like the WMMA case.
            for (m, k, n) in [(16usize, 32usize, 8usize), (64, 64, 64), (128, 256, 64)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = gemm_nt_fp8(g, &a, &b, m, k, n).unwrap();
                let r = ref_nt_rounded(&a, &b, m, k, n, round);
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                let st =
                    crate::diff::assert_close(&format!("fp8_gemm {m}x{k}x{n}"), &c, &r, 1e-2, rel);
                eprintln!(
                    "fp8_gemm {m}x{k}x{n} (E4M3 tensor core): max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// Time `iters` resident launches of a GEMM `(M,N,K, A,B,C)` kernel; returns seconds/iter.
    fn time_gemm(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<f32>,
        b_d: &cudarc::driver::CudaSlice<f32>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(b_d).arg(c_d);
            unsafe { bld.launch(cfg).unwrap() };
        };
        launch(c_d);
        g.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(c_d);
        }
        g.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Kernel-resident throughput (no per-iter H2D/D2H): upload once, launch many, sync once.
    /// Run with `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn gemm_throughput() {
        with_gpu("gemm_throughput", |g| {
            let mut rng = crate::diff::Rng::new(1);
            for sz in [512usize, 1024, 2048] {
                let (m, k, n) = (sz, sz, sz);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let dims = (m as u32, n as u32, k as u32);
                let flop = 2.0 * (m as f64) * (k as f64) * (n as f64);

                let f_simple = g.function("gemm", crate::ptx::GEMM, "gemm_nt").unwrap();
                let s0 = time_gemm(g, &f_simple, gemm_cfg(m, n), dims, &a_d, &b_d, &mut c_d, 30);

                let f_rb = g
                    .function("gemm_rb", crate::ptx_gemm::gemm_rb_ptx(), "gemm_nt_rb")
                    .unwrap();
                let s1 = time_gemm(g, &f_rb, gemm_rb_cfg(m, n), dims, &a_d, &b_d, &mut c_d, 30);

                eprintln!(
                    "gemm_nt {m}³: simple {:.0} GFLOP/s ({:.2} ms) | reg-blocked {:.0} GFLOP/s ({:.2} ms)  → {:.2}× ",
                    flop / s0 / 1e9,
                    s0 * 1e3,
                    flop / s1 / 1e9,
                    s1 * 1e3,
                    s0 / s1
                );
            }
        });
    }

    /// Tensor-core throughput: f32 register-blocked vs fp16/bf16 WMMA (f32 accumulate), kernel-resident.
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn tensorcore_throughput() {
        use half::{bf16, f16};
        with_gpu("tensorcore_throughput", |g| {
            let mut rng = crate::diff::Rng::new(7);
            for sz in [512usize, 1024, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let flop = 2.0 * (m as f64) * (k as f64) * (n as f64);
                let dims = (m as u32, n as u32, k as u32);

                // f32 register-blocked (reference)
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let f_rb = g
                    .function("gemm_rb", crate::ptx_gemm::gemm_rb_ptx(), "gemm_nt_rb")
                    .unwrap();
                let s_rb = time_gemm(g, &f_rb, gemm_rb_cfg(m, n), dims, &a_d, &b_d, &mut c_d, 30);

                // fp16 WMMA
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a16_d = g.stream.memcpy_stod(&a16).unwrap();
                let b16_d = g.stream.memcpy_stod(&b16).unwrap();
                let (e16, c16) = wmma_pick("wmma_nt_f16", m, n);
                let f_f16 = g
                    .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), &e16)
                    .unwrap();
                let s_f16 = time_wmma(g, &f_f16, c16, dims, &a16_d, &b16_d, &mut c_d, 50);

                // bf16 WMMA
                let ab: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
                let bb: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
                let ab_d = g.stream.memcpy_stod(&ab).unwrap();
                let bb_d = g.stream.memcpy_stod(&bb).unwrap();
                let (eb, cb) = wmma_pick("wmma_nt_bf16", m, n);
                let f_bf16 = g
                    .function("wmma_bf16", crate::ptx_wmma::wmma_bf16_ptx(), &eb)
                    .unwrap();
                let s_bf16 = time_wmma(g, &f_bf16, cb, dims, &ab_d, &bb_d, &mut c_d, 50);

                // fp8 (E4M3) mma.sync — Ada's lowest-precision / highest-throughput tensor-core path.
                // Measure single-tile and fragment-reuse multi-tile back-to-back so the A/B shares the
                // GPU's clock state (absolute GFLOP/s swings ~7× with boost on this power-capped mobile
                // part, so only a same-run ratio is meaningful).
                let a8: Vec<u8> = a.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
                let b8: Vec<u8> = b.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
                let a8_d = g.stream.memcpy_stod(&a8).unwrap();
                let b8_d = g.stream.memcpy_stod(&b8).unwrap();
                let f_fp8 = g
                    .function("fp8_gemm", crate::ptx_fp8::fp8_gemm_ptx(), "fp8_gemm_nt")
                    .unwrap();
                let cfp8 = LaunchConfig {
                    grid_dim: ((n / 8) as u32, (m / 16) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_fp8 = time_wmma(g, &f_fp8, cfp8, dims, &a8_d, &b8_d, &mut c_d, 50);

                use crate::ptx_fp8::{FP8_TM, FP8_TN};
                let f_fp8m = g
                    .function(
                        "fp8_gemm_mt",
                        crate::ptx_fp8::fp8_gemm_mt_ptx(),
                        "fp8_gemm_nt_mt",
                    )
                    .unwrap();
                let cfp8m = LaunchConfig {
                    grid_dim: ((n / (8 * FP8_TN)) as u32, (m / (16 * FP8_TM)) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_fp8m = time_wmma(g, &f_fp8m, cfp8m, dims, &a8_d, &b8_d, &mut c_d, 50);

                eprintln!(
                    "{m}³: f32-rb {:.0} | f16-TC {:.0} ({:.1}×) | bf16-TC {:.0} ({:.1}×) | fp8-TC {:.0} ({:.1}×) | fp8-mt {:.0} ({:.1}×)  GFLOP/s",
                    flop / s_rb / 1e9,
                    flop / s_f16 / 1e9,
                    s_rb / s_f16,
                    flop / s_bf16 / 1e9,
                    s_rb / s_bf16,
                    flop / s_fp8 / 1e9,
                    s_rb / s_fp8,
                    flop / s_fp8m / 1e9,
                    s_rb / s_fp8m,
                );
            }
        });
    }

    /// Report the real tensor-core GEMM as a **% of the measured fp16 roofline** — the honest
    /// substitute for a cuBLAS comparison (no CUDA toolkit on this box ⇒ no cuBLAS to measure
    /// against). Roofline and GEMMs are timed in the SAME run so the ratio is clock-invariant
    /// (absolutes swing ~7× with boost). fp8-mt is shown vs the *fp16* roofline, so >100% is expected
    /// and correct — Ada's fp8 tensor-core rate is ~2× fp16's, i.e. fp8's own ceiling is ~2× higher.
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn tensorcore_roofline_pct() {
        use crate::ptx_fp8::{FP8_TM, FP8_TN};
        use half::f16;
        with_gpu("tensorcore_roofline_pct", |g| {
            let roof = wmma_roofline_f16(g, 4096, 2048, 5).unwrap();
            let mut rng = crate::diff::Rng::new(11);
            for sz in [2048usize, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let flop = 2.0 * m as f64 * k as f64 * n as f64;
                let dims = (m as u32, n as u32, k as u32);
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();

                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a16_d = g.stream.memcpy_stod(&a16).unwrap();
                let b16_d = g.stream.memcpy_stod(&b16).unwrap();
                let (e16, c16) = wmma_pick("wmma_nt_f16", m, n);
                let f_f16 = g
                    .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), &e16)
                    .unwrap();
                let s_f16 = time_wmma(g, &f_f16, c16, dims, &a16_d, &b16_d, &mut c_d, 50);

                let a8: Vec<u8> = a.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
                let b8: Vec<u8> = b.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
                let a8_d = g.stream.memcpy_stod(&a8).unwrap();
                let b8_d = g.stream.memcpy_stod(&b8).unwrap();
                let f_fp8m = g
                    .function(
                        "fp8_gemm_mt",
                        crate::ptx_fp8::fp8_gemm_mt_ptx(),
                        "fp8_gemm_nt_mt",
                    )
                    .unwrap();
                let cfp8m = LaunchConfig {
                    grid_dim: ((n / (8 * FP8_TN)) as u32, (m / (16 * FP8_TM)) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_fp8m = time_wmma(g, &f_fp8m, cfp8m, dims, &a8_d, &b8_d, &mut c_d, 50);

                eprintln!(
                    "{m}³: fp16 roofline {:.0} GFLOP/s | fp16-mt {:.0} ({:.0}% of fp16 roof) | \
                     fp8-mt {:.0} ({:.0}% of fp16 roof)",
                    roof / 1e9,
                    flop / s_f16 / 1e9,
                    100.0 * (flop / s_f16) / roof,
                    flop / s_fp8m / 1e9,
                    100.0 * (flop / s_fp8m) / roof,
                );
            }
        });
    }

    /// **The honest three-tier GPU scoreboard for fp16 GEMM** — the Phase-0 payoff. Mercury's
    /// tensor-core GEMM measured *same-run, same buffers* against the only peers that mean anything on
    /// a GPU: NVIDIA's hand-tuned **cuBLAS** (Tier B, the gold standard — Mercury is reported as a % of
    /// it) and a **naive CUDA-C** kernel compiled by NVRTC (Tier A — the literal "beat the C a
    /// programmer writes," the GPU twin of beating scalar CPU-C). No Mercury-GPU-vs-CPU-C comparison
    /// appears anywhere; that would only prove a GPU beats a CPU (GPU plan §1).
    ///
    /// Correctness gates speed (the first law): both peers are first cross-checked against the f64 CPU
    /// reference at a small shape, and at each timing shape all three implementations' output checksums
    /// must agree — a fast-but-wrong kernel never scores. Requires the redist DLLs on PATH; skips
    /// (never fails) if they are absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture gemm_vs_peers`
    /// with `tools/cuda-redist/nvidia/{cuda_nvrtc,cublas}/bin` prepended to PATH.
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn gemm_vs_peers() {
        use crate::baselines::{
            cublas_gemm_nt_f16, gemm_flop, nvrtc_naive_gemm_nt, peer_env_hint, peers_available,
            time_cublas_gemm_nt_f16, time_nvrtc_naive_gemm_nt,
        };
        use half::f16;
        with_gpu("gemm_vs_peers", |g| {
            if !peers_available(g) {
                eprintln!("[skip] gemm_vs_peers: cuBLAS/NVRTC not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());

            // --- Correctness first: both peers must match the f64 oracle at a small shape. ---
            let mut rng = crate::diff::Rng::new(0x9E3D);
            for (m, k, n) in [(256usize, 256usize, 256usize), (250, 260, 270)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0); // n×k for C = A·Bᵀ
                let r = ref_nt(&a, &b, m, k, n);
                // cuBLAS here is f16-in/f16-out with f32 accumulate, so its error vs the f64 oracle is
                // dominated by ~2^-11 fp16 *quantization* of inputs/output (roughly flat in K, since the
                // sum itself is f32), not the √K accumulation growth — a ~2% rel / 5e-2 abs bound passes
                // a correct fp16 GEMM comfortably yet still fails a transpose/index slip (off by ~100%).
                let cub = cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap();
                crate::diff::assert_close(&format!("cuBLAS fp16 {m}x{k}x{n}"), &cub, &r, 5e-2, 2e-2);
                let naive = nvrtc_naive_gemm_nt(g, &a, &b, m, k, n).unwrap();
                let rel_f32 = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(1e-4);
                crate::diff::assert_close(&format!("naive CUDA-C {m}x{k}x{n}"), &naive, &r, 1e-3, rel_f32);
            }
            eprintln!("[gate] cuBLAS + naive CUDA-C both match the f64 oracle ✓");

            // --- Clock warmup + peak-clock sampling (honesty law: same-run, peak-vs-peak ratios). ---
            // A mobile GPU boosts its clock as sustained load ramps, so the FIRST size otherwise reads at a
            // cold clock: pre-warmup, 1024³ cuBLAS clocked 8× slower than 2048³ (1825 vs 15847 GFLOP/s)
            // purely from that, and the roofline (sampled first, cold) read 10× under cuBLAS. Hammer a
            // large GEMM until the clock settles, then sample every kernel with `best_of` (min time =
            // peak-clock sample) so each size's %-of-cuBLAS is clock-invariant. cf. the HBM bench's best_bw.
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            const ROUNDS: usize = 4;

            // --- Speed: same-run fp16 GEMM, Mercury vs cuBLAS (Tier B) vs naive CUDA-C (Tier A). ---
            let roof = wmma_roofline_f16(g, 4096, 2048, 30).unwrap();
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);

                // Mercury fp16 WMMA fragment-reuse path (same kernel as tensorcore_roofline_pct).
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a16_d = g.stream.memcpy_stod(&a16).unwrap();
                let b16_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let (e16, c16) = wmma_pick("wmma_nt_f16", m, n);
                let f16f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), &e16).unwrap();
                let s_mt = best_of(ROUNDS, || time_wmma(g, &f16f, c16, dims, &a16_d, &b16_d, &mut c_d, 50));
                // Mercury SMEM-staged kernel — the Phase-1 lever (CTA-cooperative shared-memory tiles).
                let f_sm = g
                    .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm")
                    .unwrap();
                let s_sm =
                    best_of(ROUNDS, || time_wmma(g, &f_sm, wmma_sm_cfg(m, n), dims, &a16_d, &b16_d, &mut c_d, 50));
                // Mercury cp.async double-buffered 64×64 kernel — overlap next-tile load with compute.
                let f_db = g
                    .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm_db")
                    .unwrap();
                let s_db =
                    best_of(ROUNDS, || time_wmma(g, &f_db, wmma_sm_cfg(m, n), dims, &a16_d, &b16_d, &mut c_d, 50));
                // Mercury 128×128 + cp.async double-buffered — the cuBLAS recipe (big tile + pipeline).
                let f_sm128 = g
                    .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm128_db")
                    .unwrap();
                let s_sm128 = best_of(ROUNDS, || {
                    time_wmma(g, &f_sm128, wmma_sm128_cfg(m, n), dims, &a16_d, &b16_d, &mut c_d, 50)
                });
                // Single-buffered 128 tile — same big tile, no cp.async pipeline (the occupancy-bound
                // large-GEMM candidate: the clean scoreboard shows pipelining loses once A/B spill L2).
                let f_sm128s = g
                    .function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm128")
                    .unwrap();
                let s_sm128s = best_of(ROUNDS, || {
                    time_wmma(g, &f_sm128s, wmma_sm128_cfg(m, n), dims, &a16_d, &b16_d, &mut c_d, 50)
                });

                // Peers (same buffers' worth of work). Naive is slow → fewer iters, still per-iter time.
                let s_cub = best_of(ROUNDS, || time_cublas_gemm_nt_f16(g, m, k, n, 50).unwrap());
                let naive_iters = if sz >= 4096 { 3 } else { 10 };
                let s_naive = time_nvrtc_naive_gemm_nt(g, m, k, n, naive_iters).unwrap();

                // Checksum cross-check at this shape: all paths must compute the same matrix.
                let csum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
                let cs_mt = csum(&gemm_nt_f16(g, &a, &b, m, k, n).unwrap());
                let cs_sm = csum(&gemm_nt_f16_sm(g, &a, &b, m, k, n).unwrap());
                let cs_sm128 = csum(&gemm_nt_f16_sm128_db(g, &a, &b, m, k, n).unwrap());
                let cs_sm128s = csum(&gemm_nt_f16_sm128(g, &a, &b, m, k, n).unwrap());
                let cs_db = csum(&gemm_nt_f16_sm_db(g, &a, &b, m, k, n).unwrap());
                let cs_c = csum(&cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap());
                let cs_n = csum(&nvrtc_naive_gemm_nt(g, &a, &b, m, k, n).unwrap());
                let agree = |x: f64, y: f64| (x - y).abs() / y.max(1.0) < 3e-2;
                assert!(
                    agree(cs_mt, cs_n) && agree(cs_sm, cs_n) && agree(cs_sm128, cs_n)
                        && agree(cs_sm128s, cs_n) && agree(cs_db, cs_n) && agree(cs_c, cs_n),
                    "{sz}³ checksum disagreement: mt={cs_mt:.3e} sm={cs_sm:.3e} sm128={cs_sm128:.3e} sm128s={cs_sm128s:.3e} db={cs_db:.3e} cublas={cs_c:.3e} naive={cs_n:.3e}"
                );

                let (g_mt, g_sm, g_sm128, g_sm128s, g_db, g_cub, g_naive) = (
                    flop / s_mt, flop / s_sm, flop / s_sm128, flop / s_sm128s,
                    flop / s_db, flop / s_cub, flop / s_naive,
                );
                eprintln!(
                    "\n{sz}³ fp16 GEMM (same-run):\n  \
                     Mercury _mt      : {:>7.0} GFLOP/s  | {:>5.1}% of cuBLAS\n  \
                     Mercury _sm        : {:>7.0} GFLOP/s  | {:>5.1}% of cuBLAS | {:>5.2}× vs _mt\n  \
                     Mercury _sm_db     : {:>7.0} GFLOP/s  | {:>5.1}% of cuBLAS | {:>5.2}× vs _sm | {:>5.1}× vs naive\n  \
                     Mercury _sm128     : {:>7.0} GFLOP/s  | {:>5.1}% of cuBLAS | {:>5.2}× vs _sm\n  \
                     Mercury _sm128db   : {:>7.0} GFLOP/s  | {:>5.1}% of cuBLAS | {:>5.2}× vs _sm\n  \
                     cuBLAS fp16      : {:>7.0} GFLOP/s  | gold standard\n  \
                     naive CUDA-C     : {:>7.0} GFLOP/s  | Tier-A baseline\n  \
                     fp16 roofline    : {:>7.0} GFLOP/s  | _sm128 {:>4.1}% / _sm128db {:>4.1}% / cuBLAS {:>4.1}% of roof",
                    g_mt / 1e9, 100.0 * g_mt / g_cub,
                    g_sm / 1e9, 100.0 * g_sm / g_cub, g_sm / g_mt,
                    g_db / 1e9, 100.0 * g_db / g_cub, g_db / g_sm, g_db / g_naive,
                    g_sm128s / 1e9, 100.0 * g_sm128s / g_cub, g_sm128s / g_sm,
                    g_sm128 / 1e9, 100.0 * g_sm128 / g_cub, g_sm128 / g_sm,
                    g_cub / 1e9,
                    g_naive / 1e9,
                    roof / 1e9, 100.0 * g_sm128s / roof, 100.0 * g_sm128 / roof, 100.0 * g_cub / roof,
                );
            }
        });
    }

    /// **The beat-cuBLAS lever: fusion**, across the activations a real network uses (relu / **silu**,
    /// the SwiGLU FFN gate / gelu). `act(A·Bᵀ)` as a single fused kernel vs the two-kernel call chains
    /// cuBLAS forces (GEMM writes C to HBM, a second kernel reads it back, applies the activation, writes
    /// it again). cuBLAS *cannot* fuse, so its pipeline always pays that extra C round-trip; the fused
    /// kernel writes C once — the same megakernel-beats-call-chain principle as M13. Times are on
    /// resident device buffers; the chain cost is GEMM + act summed (dependency-serialized — the
    /// activation reads the GEMM's output — so the sum is the true chain time). Correctness gates speed:
    /// the fused output must equal `act(cuBLAS GEMM)` at every shape. Redist DLLs on PATH (see
    /// `gemm_vs_peers`).
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn fused_gemm_activation_vs_chain() {
        use crate::baselines::{
            cublas_gemm_nt_f16, gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_f16,
        };
        use half::f16;
        with_gpu("fused_act", |g| {
            if !peers_available(g) {
                eprintln!("[skip] fused_gemm_activation_vs_chain: cuBLAS/NVRTC not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            let mut rng = crate::diff::Rng::new(0xF0ED);
            // Warm the clock first (cf. gemm_vs_peers): without it the GEMM/cuBLAS times (measured at the
            // top of each size block, cold) are compared against fused times measured later (warm), and
            // the laptop GPU's load-ramp clock boost inflates the ratio — a cold-vs-warm artifact, not an
            // honest peak-vs-peak fusion win. best_of(5) alone can't fix a monotonic within-block ramp.
            for _ in 0..30 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            for sz in [512usize, 1024, 2048] {
                let (m, k, n) = (sz, sz, sz);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let dims = (m as u32, n as u32, k as u32);
                let ptx = crate::ptx_wmma::wmma_f16_ptx();
                let flop = gemm_flop(m, n, k);

                // GEMM-only + cuBLAS times are shared across activations (the activation cost adds on).
                let cub_out = cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap();
                let f_gemm = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db").unwrap();
                // best-of-N (min = least clock-throttled) so the fused/chain ratio is peer-vs-peer at
                // peak clock and stable run-to-run, not a single throttle-skewed sample.
                let t_gemm = best_of(5, || time_wmma(g, &f_gemm, wmma_sm_cfg(m, n), dims, &a_d, &b_d, &mut c_d, 30));
                let t_cub = best_of(5, || time_cublas_gemm_nt_f16(g, m, k, n, 30).unwrap());
                eprintln!("\n{sz}³ act(A·Bᵀ) (same-run, on-device; gemm {:.3} ms, cuBLAS {:.3} ms):", t_gemm * 1e3, t_cub * 1e3);

                for act in ["relu", "silu", "gelu"] {
                    // Correctness: the fused kernel must equal the activation applied to cuBLAS's GEMM.
                    let fused = match act {
                        "relu" => gemm_nt_f16_sm_db_relu(g, &a, &b, m, k, n),
                        "silu" => gemm_nt_f16_sm_db_silu(g, &a, &b, m, k, n),
                        _ => gemm_nt_f16_sm_db_gelu(g, &a, &b, m, k, n),
                    }
                    .unwrap();
                    let refout: Vec<f32> = cub_out
                        .iter()
                        .map(|&x| match act {
                            "relu" => x.max(0.0),
                            "silu" => silu(x),
                            _ => gelu(x),
                        })
                        .collect();
                    crate::diff::assert_close(&format!("fused {act} vs cuBLAS chain {sz}³"), &fused, &refout, 5e-2, 2e-2);

                    // Timing: fused one-kernel vs the activation kernel that a call-chain adds.
                    let entry = format!("wmma_nt_f16_sm_db_{act}");
                    let f_fused = g.function("wmma_f16", ptx, &entry).unwrap();
                    let f_act = g.function("vmath", crate::ptx::vmath_ptx(), act).unwrap();
                    let t_fused = best_of(5, || time_wmma(g, &f_fused, wmma_sm_cfg(m, n), dims, &a_d, &b_d, &mut c_d, 30));
                    let t_act = best_of(5, || time_vmath(g, &f_act, m * n, 30));
                    let (mer_chain, cub_chain) = (t_gemm + t_act, t_cub + t_act);
                    eprintln!(
                        "  {act}: fused {:>7.3} ms ({:>6.0} GFLOP/s) | Mercury chain {:>7.3} ms ({:>4.2}× slower) | cuBLAS chain {:>7.3} ms (fused {:>4.2}× faster)",
                        t_fused * 1e3, flop / t_fused / 1e9,
                        mer_chain * 1e3, mer_chain / t_fused,
                        cub_chain * 1e3, cub_chain / t_fused,
                    );
                }

                // --- Residual fusion: out = A·Bᵀ + x (the transformer skip connection). Fused via
                // wmma.load.c in ONE kernel vs the call-chain a GEMM library writes: the GEMM, then a
                // separate vadd that reads C back from HBM, adds x, and writes out. (cuBLAS's beta=1 can
                // fold a C += into the GEMM, but still needs C pre-loaded with x — an extra copy; the
                // honest, simplest peer is gemm + vadd.) Same megakernel-beats-call-chain principle (M13).
                let resid = rng.vec(m * n, -1.0, 1.0);
                let fused_r = gemm_nt_f16_sm_db_residual(g, &a, &b, &resid, m, k, n).unwrap();
                let ref_r: Vec<f32> =
                    cub_out.iter().zip(resid.iter()).map(|(&c, &x)| c + x).collect();
                crate::diff::assert_close(
                    &format!("fused residual vs cuBLAS+add {sz}³"),
                    &fused_r,
                    &ref_r,
                    5e-2,
                    2e-2,
                );
                let resid_d = g.stream.memcpy_stod(&resid).unwrap();
                let f_resid = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_residual").unwrap();
                let f_vadd = g.function("vadd", crate::ptx::VADD, "vadd").unwrap();
                let t_fused_r = best_of(5, || {
                    time_wmma_residual(g, &f_resid, wmma_sm_cfg(m, n), dims, &a_d, &b_d, &mut c_d, &resid_d, 30)
                });
                let t_add = best_of(5, || time_vadd(g, &f_vadd, m * n, 30));
                let (mer_chain_r, cub_chain_r) = (t_gemm + t_add, t_cub + t_add);
                eprintln!(
                    "  residual: fused {:>7.3} ms ({:>6.0} GFLOP/s) | Mercury chain {:>7.3} ms ({:>4.2}× slower) | cuBLAS chain {:>7.3} ms (fused {:>4.2}× faster)",
                    t_fused_r * 1e3, flop / t_fused_r / 1e9,
                    mer_chain_r * 1e3, mer_chain_r / t_fused_r,
                    cub_chain_r * 1e3, cub_chain_r / t_fused_r,
                );
            }
        });
    }

    /// **M13 — the megakernel beats the call-chain, at the FFN-block level.** Times the resident
    /// [`ffn_fused`] (5 launches: norm, cast, SiLU-GEMM, cast, residual-GEMM — the SiLU folded into the
    /// up-proj store, the residual into the down-proj accumulator) against the unfused chain the same
    /// kernels run with the epilogues SPLIT OUT (7 launches: norm, cast, GEMM, SiLU, cast, GEMM, add —
    /// the SiLU and add are two extra kernels, each round-tripping [S,Dff]/[S,D] through HBM). Identical
    /// GEMMs, identical buffers, fully resident (uploaded once); the only difference is the two fused
    /// epilogues, so the ratio isolates the fusion value (2 launches + 2 HBM round-trips). best_of after
    /// a clock warmup. Correctness gates speed: the two chains' outputs must agree (checksum).
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn ffn_fused_vs_chain_throughput() {
        use half::f16;
        with_gpu("ffn_throughput", |g| {
            let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm").unwrap();
            let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16").unwrap();
            let ptx = crate::ptx_wmma::wmma_f16_ptx();
            let f_silu_gemm = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_silu").unwrap();
            let f_resid_gemm = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db_residual").unwrap();
            let f_gemm = g.function("wmma_f16", ptx, "wmma_nt_f16_sm_db").unwrap();
            let f_siluv = g.function("vmath", crate::ptx::vmath_ptx(), "silu").unwrap();
            let f_vadd = g.function("vadd", crate::ptx::VADD, "vadd").unwrap();
            let stream = g.stream.clone();
            let eps = 1e-5f32;
            let mut rng = crate::diff::Rng::new(0xFF11);

            // Clock warmup so fused/unfused are sampled at the same peak clock (cf. gemm_vs_peers).
            {
                use crate::baselines::{peers_available, time_cublas_gemm_nt_f16};
                if peers_available(g) {
                    for _ in 0..30 {
                        let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
                    }
                }
            }
            const ITERS: usize = 50;

            for &(s, d, dff) in &[(128usize, 64usize, 256usize), (512, 256, 1024)] {
                let x = rng.vec(s * d, -1.0, 1.0);
                let w1_f32 = rng.vec(dff * d, -0.1, 0.1);
                let w2_f32 = rng.vec(d * dff, -0.1, 0.1);
                let w1: Vec<f16> = w1_f32.iter().map(|&v| f16::from_f32(v)).collect();
                let w2: Vec<f16> = w2_f32.iter().map(|&v| f16::from_f32(v)).collect();
                let x_d = stream.memcpy_stod(&x).unwrap();
                let w1_d = stream.memcpy_stod(&w1).unwrap();
                let w2_d = stream.memcpy_stod(&w2).unwrap();
                let mut h2 = stream.memcpy_stod(&vec![0f32; s * d]).unwrap();
                let mut h2_16 = stream.memcpy_stod(&vec![f16::from_f32(0.0); s * d]).unwrap();
                let mut t_dff = stream.memcpy_stod(&vec![0f32; s * dff]).unwrap();
                let mut t_dff_b = stream.memcpy_stod(&vec![0f32; s * dff]).unwrap();
                let mut t_dff16 = stream.memcpy_stod(&vec![f16::from_f32(0.0); s * dff]).unwrap();
                let mut t_d = stream.memcpy_stod(&vec![0f32; s * d]).unwrap();
                let mut out = stream.memcpy_stod(&vec![0f32; s * d]).unwrap();
                let norm_cfg =
                    LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
                let cfg1 = wmma_sm_cfg(s, dff); // up-proj  C[S,Dff]
                let cfg2 = wmma_sm_cfg(s, d); // down-proj C[S,D]
                let (mm1, nn1, kk1) = (s as u32, dff as u32, d as u32);
                let (mm2, nn2, kk2) = (s as u32, d as u32, dff as u32);
                let n_sd = (s * d) as u32;
                let n_sdff = (s * dff) as u32;
                let cast_sd = LaunchConfig::for_num_elems(n_sd);
                let cast_sdff = LaunchConfig::for_num_elems(n_sdff);

                // Fused: norm → cast → SiLU-GEMM → cast → residual-GEMM (5 launches/iter).
                let t_fused = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..ITERS {
                        let mut b = stream.launch_builder(&f_norm);
                        b.arg(&mm2).arg(&nn2).arg(&eps).arg(&x_d).arg(&mut h2); // mm2=S, nn2=D
                        unsafe { b.launch(norm_cfg).unwrap() };
                        let mut b = stream.launch_builder(&f_cast);
                        b.arg(&n_sd).arg(&h2).arg(&mut h2_16);
                        unsafe { b.launch(cast_sd).unwrap() };
                        let mut b = stream.launch_builder(&f_silu_gemm);
                        b.arg(&mm1).arg(&nn1).arg(&kk1).arg(&h2_16).arg(&w1_d).arg(&mut t_dff);
                        unsafe { b.launch(cfg1).unwrap() };
                        let mut b = stream.launch_builder(&f_cast);
                        b.arg(&n_sdff).arg(&t_dff).arg(&mut t_dff16);
                        unsafe { b.launch(cast_sdff).unwrap() };
                        let mut b = stream.launch_builder(&f_resid_gemm);
                        b.arg(&mm2).arg(&nn2).arg(&kk2).arg(&t_dff16).arg(&w2_d).arg(&mut out).arg(&x_d);
                        unsafe { b.launch(cfg2).unwrap() };
                    }
                    stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / ITERS as f64
                });

                // Unfused: norm → cast → GEMM → SiLU → cast → GEMM → add (7 launches/iter).
                let t_chain = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..ITERS {
                        let mut b = stream.launch_builder(&f_norm);
                        b.arg(&mm2).arg(&nn2).arg(&eps).arg(&x_d).arg(&mut h2);
                        unsafe { b.launch(norm_cfg).unwrap() };
                        let mut b = stream.launch_builder(&f_cast);
                        b.arg(&n_sd).arg(&h2).arg(&mut h2_16);
                        unsafe { b.launch(cast_sd).unwrap() };
                        let mut b = stream.launch_builder(&f_gemm);
                        b.arg(&mm1).arg(&nn1).arg(&kk1).arg(&h2_16).arg(&w1_d).arg(&mut t_dff);
                        unsafe { b.launch(cfg1).unwrap() };
                        let mut b = stream.launch_builder(&f_siluv);
                        b.arg(&n_sdff).arg(&t_dff).arg(&mut t_dff_b);
                        unsafe { b.launch(cast_sdff).unwrap() };
                        let mut b = stream.launch_builder(&f_cast);
                        b.arg(&n_sdff).arg(&t_dff_b).arg(&mut t_dff16);
                        unsafe { b.launch(cast_sdff).unwrap() };
                        let mut b = stream.launch_builder(&f_gemm);
                        b.arg(&mm2).arg(&nn2).arg(&kk2).arg(&t_dff16).arg(&w2_d).arg(&mut t_d);
                        unsafe { b.launch(cfg2).unwrap() };
                        let mut b = stream.launch_builder(&f_vadd);
                        b.arg(&n_sd).arg(&t_d).arg(&x_d).arg(&mut out);
                        unsafe { b.launch(cast_sd).unwrap() };
                    }
                    stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / ITERS as f64
                });

                // Correctness gates speed: the unfused timing loop ran last, so `out` holds its result —
                // verify it against the f64 FFN reference (the fused path is gated separately by
                // ffn_fused_matches_reference_within_tol; both share the same WMMA GEMMs).
                let oracle = ref_ffn(&x, &w1_f32, &w2_f32, s, d, dff);
                let chain_out = stream.memcpy_dtov(&out).unwrap();
                crate::diff::assert_close(
                    &format!("FFN chain {s}x{d}x{dff}"),
                    &chain_out,
                    &oracle,
                    3e-2,
                    3e-2,
                );

                let flop = 4.0 * s as f64 * d as f64 * dff as f64; // two GEMMs
                eprintln!(
                    "FFN {s}x{d}x{dff} (resident, same-run): fused {:>7.3} ms ({:>6.0} GFLOP/s, 5 launches) | unfused chain {:>7.3} ms (7 launches) | fused {:>4.2}× faster",
                    t_fused * 1e3, flop / t_fused / 1e9, t_chain * 1e3, t_chain / t_fused,
                );
            }
        });
    }

    /// **M9 — memory-bound kernels at ≥90% of peak HBM bandwidth.** Times three pure-streaming kernels
    /// on resident device buffers (no host round-trip) against the *theoretical* peak derived from the
    /// device's own clock + bus width (`peak_hbm_gbs`, the same formula NVIDIA's `deviceQuery` prints) —
    /// not a soft empirical roofline. The working set (256 MB/array) dwarfs L2, so this is DRAM, not
    /// cache. Correctness gates speed: the copy is bit-checked first. The headline is the vectorized
    /// **copy** (read+write, the traffic of any elementwise op); saxpy (triad, 3 streams) and the
    /// read-only reduction are honest context. Achieved GB/s = bytes_moved / time.
    #[test]
    #[ignore = "throughput bench; run explicitly on a CUDA box"]
    fn hbm_bandwidth() {
        with_gpu("hbm_bandwidth", |g| {
            eprintln!("device: {}", g.device_name());
            let peak = g.peak_hbm_gbs();
            match peak {
                Some(p) => eprintln!("theoretical peak HBM: {p:.1} GB/s  ({} SMs)", g.sm_count()),
                None => eprintln!("theoretical peak HBM: <unqueryable>"),
            }

            // Correctness first — a fast copy that corrupts data scores nothing.
            let mut rng = crate::diff::Rng::new(0xB17D);
            let probe = rng.vec(8192, -1.0, 1.0);
            assert_eq!(copy(g, &probe).unwrap(), probe, "copy_v4 must be exact");

            let n = 64usize << 20; // 67.1M floats = 256 MB/array — far past L2
            let n4 = (n / 4) as u32;
            let nn = n as u32;
            let pct = |gbs: f64| {
                peak.map(|p| format!("{:>5.1}% of peak", 100.0 * gbs / p))
                    .unwrap_or_else(|| "   —".into())
            };
            let src = g.stream.memcpy_stod(&vec![1.0f32; n]).unwrap();
            let mut dst = g.stream.memcpy_stod(&vec![0f32; n]).unwrap();

            // --- copy: dst = src. Moves 2N floats (read + write) — the hardest 1:1 mix. ---
            let f_copy = g.function("copy_v4", crate::ptx::COPY_V4, "copy_v4").unwrap();
            let cfg_copy = stream_cfg(g, n4);
            let bw_copy = best_bw(g, 2.0 * n as f64 * 4.0, || {
                let mut b = g.stream.launch_builder(&f_copy);
                b.arg(&n4).arg(&src).arg(&mut dst);
                unsafe { b.launch(cfg_copy).unwrap() };
            });
            eprintln!("  copy   (2N rw): {bw_copy:>6.1} GB/s  {}", pct(bw_copy));

            // --- saxpy triad: y = a·x + y. Reads x and y, writes y → 3N floats. ---
            let a = 2.0f32;
            let f_saxpy = g.function("saxpy", crate::ptx::SAXPY, "saxpy").unwrap();
            let cfg_elem = LaunchConfig::for_num_elems(nn);
            let bw_saxpy = best_bw(g, 3.0 * n as f64 * 4.0, || {
                let mut b = g.stream.launch_builder(&f_saxpy);
                b.arg(&nn).arg(&a).arg(&src).arg(&mut dst);
                unsafe { b.launch(cfg_elem).unwrap() };
            });
            eprintln!("  saxpy  (3N tr): {bw_saxpy:>6.1} GB/s  {}", pct(bw_saxpy));

            // --- reduce_sum: read N floats (partials write is negligible). ---
            let mut partials = g.stream.memcpy_stod(&vec![0f32; RED_GRID as usize]).unwrap();
            let f_red = g.function("reduce", crate::ptx::REDUCE, "reduce_sum").unwrap();
            let cfg_red = LaunchConfig {
                grid_dim: (RED_GRID, 1, 1),
                block_dim: (RED_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let bw_red = best_bw(g, n as f64 * 4.0, || {
                let mut b = g.stream.launch_builder(&f_red);
                b.arg(&nn).arg(&src).arg(&mut partials);
                unsafe { b.launch(cfg_red).unwrap() };
            });
            eprintln!("  reduce (1N r ): {bw_red:>6.1} GB/s  {}", pct(bw_red));

            let best = bw_copy.max(bw_saxpy).max(bw_red);
            if let Some(p) = peak {
                eprintln!(
                    "\nbest streaming kernel: {best:.1} GB/s = {:.1}% of {p:.0} GB/s peak \
                     → M9 (≥90%): {}",
                    100.0 * best / p,
                    if best >= 0.90 * p {
                        "MET ✓"
                    } else {
                        "not yet (laptop GPU may be memory-clock throttled this run)"
                    }
                );
            }
        });
    }

    /// Best-of-N streaming bandwidth (GB/s) for a `launch` closure that moves `bytes` of HBM traffic per
    /// call. A laptop GPU dynamically down-clocks its memory to save power, so a single timed round can
    /// land in any clock state (the same kernel here measured 95% and 63% of peak on two runs purely
    /// from the clock). We warm up to coax the boost clock, then report the **fastest** of several
    /// rounds — the least-throttled one, i.e. the device's peak capability, which is what "% of the
    /// max-clock theoretical peak" is asking for.
    fn best_bw(g: &Gpu, bytes: f64, mut launch: impl FnMut()) -> f64 {
        const WARMUP: usize = 60;
        const ROUNDS: usize = 12;
        const ITERS: usize = 20;
        for _ in 0..WARMUP {
            launch();
        }
        g.stream.synchronize().unwrap();
        let mut best = 0.0f64;
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                launch();
            }
            g.stream.synchronize().unwrap();
            best = best.max(bytes / (t0.elapsed().as_secs_f64() / ITERS as f64) / 1e9);
        }
        best
    }

    /// The roofline microbench must JIT and run, returning a positive, plausible TC rate (sanity that
    /// the compute-bound kernel isn't dead-code-eliminated to ~0 or mis-issued). Not a tolerance gate.
    #[test]
    fn roofline_kernel_runs() {
        with_gpu("roofline_kernel_runs", |g| {
            let r = wmma_roofline_f16(g, 256, 256, 2).unwrap();
            assert!(
                r > 1.0e11 && r < 5.0e14,
                "fp16 roofline {r:.3e} FLOP/s implausible (DCE'd or mis-measured?)"
            );
            eprintln!("fp16 TC roofline (small probe): {:.0} GFLOP/s", r / 1e9);
        });
    }

    /// Time WMMA launches with half-precision inputs (a/b, f16 or bf16) and an f32 C; seconds/iter.
    fn time_wmma<T: cudarc::driver::DeviceRepr>(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<T>,
        b_d: &cudarc::driver::CudaSlice<T>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(b_d).arg(c_d);
            unsafe { bld.launch(cfg).unwrap() };
        };
        launch(c_d);
        g.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(c_d);
        }
        g.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Per-iter device time of the fused-residual GEMM (`wmma_nt_f16_sm_db_residual`), which takes the
    /// `residual[M,N]` (f32) as a 7th param after C. Same warmup+loop shape as [`time_wmma`].
    fn time_wmma_residual<T: cudarc::driver::DeviceRepr>(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<T>,
        b_d: &cudarc::driver::CudaSlice<T>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        resid_d: &cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(b_d).arg(c_d).arg(resid_d);
            unsafe { bld.launch(cfg).unwrap() };
        };
        launch(c_d);
        g.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(c_d);
        }
        g.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Per-iter device time of one `vadd` (`out = x + y`, the residual add a call-chain pays as a
    /// separate kernel) over `n` f32 — 3N traffic (read x, read y, write out), on resident buffers.
    /// `f` is the prefetched `vadd` entry.
    fn time_vadd(g: &Gpu, f: &cudarc::driver::CudaFunction, n: usize, iters: usize) -> f64 {
        let nn = n as u32;
        let x_d = g.stream.memcpy_stod(&vec![0.5f32; n]).unwrap();
        let y_d = g.stream.memcpy_stod(&vec![0.25f32; n]).unwrap();
        let mut out_d = g.stream.memcpy_stod(&vec![0f32; n]).unwrap();
        let cfg = LaunchConfig::for_num_elems(nn);
        let launch = |out_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut b = g.stream.launch_builder(f);
            b.arg(&nn).arg(&x_d).arg(&y_d).arg(out_d);
            unsafe { b.launch(cfg).unwrap() };
        };
        launch(&mut out_d);
        g.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(&mut out_d);
        }
        g.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Minimum per-iter time over `rounds` measurements — the least clock-throttled (peak-clock) sample.
    /// Used to make same-run fused/chain ratios stable and clock-invariant (a laptop GPU's dynamic clock
    /// otherwise skews any single timing; see the bandwidth bench's `best_bw`).
    fn best_of(rounds: usize, mut f: impl FnMut() -> f64) -> f64 {
        (0..rounds).map(|_| f()).fold(f64::INFINITY, f64::min)
    }

    /// Per-iter device time of one elementwise `vmath` pass over `n` f32 (read x, write out — exactly
    /// the HBM round-trip a non-fused GEMM+activation chain pays and a fused epilogue avoids). `f` is any
    /// prefetched single-input activation entry (relu/silu/gelu/…); the cost is the 2N traffic, ~flat
    /// across activations.
    fn time_vmath(g: &Gpu, f: &cudarc::driver::CudaFunction, n: usize, iters: usize) -> f64 {
        let nn = n as u32;
        let x_d = g.stream.memcpy_stod(&vec![0.5f32; n]).unwrap();
        let mut out_d = g.stream.memcpy_stod(&vec![0f32; n]).unwrap();
        let cfg = LaunchConfig::for_num_elems(nn);
        let launch = |out_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut b = g.stream.launch_builder(f);
            b.arg(&nn).arg(&x_d).arg(out_d);
            unsafe { b.launch(cfg).unwrap() };
        };
        launch(&mut out_d);
        g.stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(&mut out_d);
        }
        g.stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() / iters as f64
    }
}
