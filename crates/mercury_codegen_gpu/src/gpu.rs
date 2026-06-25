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
    // Regime-aware dispatch among the multi-stage `cp.async` pipeline kernels (all numerically identical;
    // winners picked by `gemm_pipe_sweep`, same-run vs cuBLAS at full clock). The binding constraint flips
    // with the A+B working set vs the 24 MB L2 (see `PIPE_VARIANTS`):
    //   • L2-resident: a DEEP BK=16 pipeline wins (latency is low; depth keeps the tensor cores fed) —
    //     `pipe_64_s6` ≤1024³ (~90% of cuBLAS), `pipe_128_s4` ~2048³ (~94%).
    //   • Larger (≥ ~2048³, A+B ≳ L2): the `mma.sync.m16n8k16` kernel with conflict-free padded SMEM and
    //     threadblock rasterization wins both the L2-resident (2048³ ~92%, padding-bound) and the
    //     HBM-bound (4096³ ~74%, raster-bound) sub-regimes — `mma_nt_f16_128_bk32_s2_r8`.
    // Anything not matching a pipeline variant's divisibility falls through to the older SMEM kernels.
    use crate::ptx_wmma::pipe_variant;
    let ws_bytes = (m * k + n * k) * 2; // fp16 A+B working set (bytes)
    if ws_bytes >= 16 * 1024 * 1024 && m % 128 == 0 && n % 128 == 0 && k % 32 == 0 {
        let wh = pipe_variant("mma_nt_f16_128_bk32_s2_r16");
        // **Large regime (A+B ≥ 16 MB, ≥2048³):** the no-pad `ldmatrix`+XOR-swizzle **w24** workhorse is the
        // robust same-run winner — it beats the padded hand-placed base **1.23× @2048³ (87.4% vs 70.9% of
        // cuBLAS) and 1.13× @4096³**. Dropping the padding buys a 3rd CTA/SM to hide HBM latency, and the
        // swizzle keeps the `ldmatrix` gathers conflict-free without it. A clean re-measure on the
        // round-robin best-of-N / self-noise-sentinel instrument (`gemm_cliff_ab`) showed the w22 2×2 warp
        // grid is only a *noise-level* tie with w24 at 4096³ (0.97–1.02× across runs) and *loses* at 2048³,
        // so the whole regime uses w24 (not w22, not the padded base). Bit-gated by
        // `mma_swizzle_matches_reference_within_tol`.
        let swz_w24 = crate::ptx_wmma::PipeCfg { name: "mma_nt_f16_128_bk32_s2_r16_swz", pad: 0, ..*wh };
        return gemm_nt_f16_pipe(g, a, b, m, k, n, &swz_w24);
    }
    if m <= 1024 && n <= 1024 && m % SM_BM == 0 && n % SM_BN == 0 {
        return gemm_nt_f16_pipe(g, a, b, m, k, n, pipe_variant("wmma_nt_f16_pipe_64_s6"));
    }
    if m % SM128_BM == 0 && n % SM128_BN == 0 {
        return gemm_nt_f16_pipe(g, a, b, m, k, n, pipe_variant("wmma_nt_f16_pipe_128_s4"));
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

/// **Static-shape fp16 SMEM-staged GEMM** `C = A·Bᵀ` — the M1 compile-time-shapes lever for fp16 (the
/// twin of [`gemm_nt_w4a16_static`] / [`gemm_nt_int8_static`]). Bakes M/N/K into the `wmma_nt_f16_sm`
/// kernel ([`crate::ptx_wmma::wmma_f16_sm_static_ptx`]) so ptxas constant-folds the hot-loop strides and
/// knows the K trip count; picks the 64×64 / 128×128 tile by the large-size regime rule. **Bit-exact**
/// vs the dynamic `_sm` kernel (identical codegen, only the dims are constants); per-shape PTX built +
/// raw-loaded here. Requires M%bm==0, N%bn==0, K%16==0.
pub fn gemm_nt_f16_static(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::{SM128_BM, SM128_BN, SM128_THREADS, SM_BM, SM_BN, SM_THREADS};
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(k % 16 == 0, "gemm_nt_f16_static requires K%16==0");
    let use_128 = m >= 4096 && n >= 4096 && m % SM128_BM == 0 && n % SM128_BN == 0;
    let (bm, bn, threads) = if use_128 {
        (SM128_BM, SM128_BN, SM128_THREADS)
    } else {
        (SM_BM, SM_BN, SM_THREADS)
    };
    assert!(m % bm == 0 && n % bn == 0, "gemm_nt_f16_static requires M%{bm}==0, N%{bn}==0");
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let ptx = crate::ptx_wmma::wmma_f16_sm_static_ptx(m, n, k, use_128);
    let module = g.ctx.load_module(ptx.as_str().into())?;
    let f = module.load_function(crate::ptx_wmma::wmma_f16_sm_static_entry(use_128))?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let cfg = LaunchConfig {
        grid_dim: ((n / bn) as u32, (m / bm) as u32, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
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

/// Launch config for a multi-stage `cp.async` pipeline variant (`entry_smem_pipe`): `v.threads()` per
/// CTA. Without rasterization the grid is 2-D (one CTA per `bm×bn` output tile, mapped from ctaid.x/y);
/// with `v.raster>0` the kernel remaps a **1-D** grid of `tiles_m·tiles_n` blocks into an L2-friendly
/// tile order itself, so the grid must be 1-D. Applies when `M%bm==0 && N%bn==0 && K%bk==0`.
fn pipe_cfg(v: &crate::ptx_wmma::PipeCfg, m: usize, n: usize) -> LaunchConfig {
    let grid_dim = if v.raster > 0 {
        (((m / v.bm) * (n / v.bn)) as u32, 1, 1)
    } else {
        ((n / v.bn) as u32, (m / v.bm) as u32, 1)
    };
    LaunchConfig { grid_dim, block_dim: (v.threads() as u32, 1, 1), shared_mem_bytes: 0 }
}

/// `C = A·Bᵀ` (fp16-in, f32-out) via a **multi-stage `cp.async` pipeline** variant `v` — the
/// large-GEMM-cliff path. `v` selects the macro-tile / staged-BK / pipeline-depth (see
/// [`crate::ptx_wmma::PipeCfg`] and [`crate::ptx_wmma::PIPE_VARIANTS`]); all share the precision-generic
/// `entry_smem_pipe` generator and are numerically identical to the other WMMA GEMMs (f32 accumulate).
/// Requires `M%v.bm==0`, `N%v.bn==0`, `K%v.bk==0`; tolerance-gated.
pub fn gemm_nt_f16_pipe(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    v: &crate::ptx_wmma::PipeCfg,
) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % v.bm == 0 && n % v.bn == 0 && k % v.bk == 0,
        "{} requires M%{}==0, N%{}==0, K%{}==0",
        v.name, v.bm, v.bn, v.bk
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), v.name)?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(pipe_cfg(v, m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// The fp16 `mma.sync` workhorse config (`mma_nt_f16_128_bk32_s2_r16`) — the fastest large-GEMM base, and
/// thus the one the fused `act(A·Bᵀ+bias)` epilogues build on (`gemm_nt_f16_pipe_fused_bias`).
fn mma_workhorse() -> &'static crate::ptx_wmma::PipeCfg {
    crate::ptx_wmma::pipe_variant("mma_nt_f16_128_bk32_s2_r16")
}

/// `C = act(A·Bᵀ + bias)` fused into the **fast `mma.sync` workhorse** store epilogue (the r16-raster
/// pipeline `mma_nt_f16_128_bk32_s2_r16`, the fastest large-GEMM base). `entry` selects the variant
/// (`..._bias{,_relu,_silu,_gelu}`). This is the canonical nn.Linear / FFN epilogue, the thing cuBLAS
/// structurally cannot do (it computes only `A·Bᵀ`, so the bias add + activation need a *second* kernel
/// that round-trips C through HBM) — fusing it onto the fastest GEMM base is the beat-cuBLAS lever on
/// real transformer workloads. Bias is applied per output column to the f32 accumulators before the
/// store, then the activation, register-level (no SMEM scratch the WMMA bias path needs). Requires
/// `M%128==0`, `N%128==0`, `K%32==0`; tolerance-gated against an `act(A·Bᵀ+bias)` f64 reference.
fn gemm_nt_f16_pipe_fused_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_pipe_fused_bias_v(g, a, b, bias, m, k, n, mma_workhorse(), entry)
}

/// `C = act(A·Bᵀ + bias)` fused into an arbitrary pipeline variant `v`'s store epilogue. The workhorse
/// (`mma_nt_f16_128_bk32_s2_r16`) base wins ≥2048³ via the register-level epilogue; the **deep WMMA pipe
/// `wmma_nt_f16_pipe_64_s6`** base wins ≤1024³ via the SMEM store-back epilogue (it is the ≤1024³ GEMM
/// champion, where the workhorse loses). `v` supplies the launch config + divisibility; `entry` the
/// fused variant name (`{v.name}_bias{,_relu,_silu,_gelu}`). Tolerance-gated vs an `act(A·Bᵀ+bias)` f64 ref.
fn gemm_nt_f16_pipe_fused_bias_v(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    v: &crate::ptx_wmma::PipeCfg,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert!(
        m % v.bm == 0 && n % v.bn == 0 && k % v.bk == 0,
        "{entry} requires M%{}==0, N%{}==0, K%{}==0",
        v.bm, v.bn, v.bk
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
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d);
    unsafe { bld.launch(pipe_cfg(v, m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// The deep WMMA pipe `wmma_nt_f16_pipe_64_s6` — fp16's **≤1024³ GEMM champion** (~90% of cuBLAS, vs the
/// `mma.sync` workhorse's ~80% there). The fused-epilogue base that wins the 1024³ regime the workhorse loses.
fn pipe64() -> &'static crate::ptx_wmma::PipeCfg {
    crate::ptx_wmma::pipe_variant("wmma_nt_f16_pipe_64_s6")
}

/// `C = A·Bᵀ + bias` (affine Linear, no activation) fused into the fast mma workhorse — see
/// [`gemm_nt_f16_pipe_fused_bias`]. The fp16 fast-path twin of [`gemm_nt_f16_sm_db_bias`] (slow WMMA base).
pub fn gemm_nt_f16_mma_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_f16_128_bk32_s2_r16_bias")
}

/// `C = relu(A·Bᵀ + bias)` fused into the fast mma workhorse — Linear+ReLU (see [`gemm_nt_f16_pipe_fused_bias`]).
pub fn gemm_nt_f16_mma_bias_relu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_f16_128_bk32_s2_r16_bias_relu")
}

/// `C = silu(A·Bᵀ + bias)` fused into the fast mma workhorse — SiLU FFN (see [`gemm_nt_f16_pipe_fused_bias`]).
pub fn gemm_nt_f16_mma_bias_silu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_f16_128_bk32_s2_r16_bias_silu")
}

/// `C = gelu(A·Bᵀ + bias)` fused into the fast mma workhorse — the canonical transformer FFN first layer
/// (BERT/GPT-2 style; see [`gemm_nt_f16_pipe_fused_bias`]).
pub fn gemm_nt_f16_mma_bias_gelu(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_f16_128_bk32_s2_r16_bias_gelu")
}

/// `C = A·Bᵀ + bias + residual` fused into the fast mma workhorse (`mma_nt_f16_128_bk32_s2_r16_bias_
/// residual`) — the transformer **down-proj / attention output-proj** sublayer output: the bias-add AND
/// the residual (skip-connection) add both fold into the GEMM store (the residual added to the post-bias
/// f32 accumulators, addressed identically to the C store), so the two HBM-round-tripping kernels a
/// cuBLAS chain runs after the GEMM collapse into the GEMM. `residual` is the [M,N] skip tensor. Requires
/// `M%128==0`, `N%128==0`, `K%32==0`; tolerance-gated vs an `(A·Bᵀ + bias) + residual` f64 reference.
pub fn gemm_nt_f16_mma_bias_residual(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    let wh = mma_workhorse();
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert_eq!(residual.len(), m * n, "residual must have length M·N");
    assert!(
        m % wh.bm == 0 && n % wh.bn == 0 && k % wh.bk == 0,
        "mma_nt_f16_128_bk32_s2_r16_bias_residual requires M%{}==0, N%{}==0, K%{}==0",
        wh.bm, wh.bn, wh.bk
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "mma_nt_f16_128_bk32_s2_r16_bias_residual")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let resid_d = g.stream.memcpy_stod(residual)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d).arg(&resid_d);
    unsafe { bld.launch(pipe_cfg(wh, m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·Bᵀ + bias + residual` fused into the **deep WMMA pipe `pipe_64_s6`** (`wmma_nt_f16_pipe_64_s6_
/// bias_residual`) — the ≤1024³ twin of [`gemm_nt_f16_mma_bias_residual`]. The down-proj / attention
/// output-proj at the size where the workhorse base loses: the residual seeds the f32 accumulator
/// (wmma.load.c), the per-column bias adds in the SMEM store-back epilogue (no activation). Requires
/// `M%64==0`, `N%64==0`, `K%16==0`; tolerance-gated vs an `(A·Bᵀ + bias) + residual` f64 reference.
pub fn gemm_nt_f16_pipe64_bias_residual(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    let v = pipe64();
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert_eq!(residual.len(), m * n, "residual must have length M·N");
    assert!(
        m % v.bm == 0 && n % v.bn == 0 && k % v.bk == 0,
        "wmma_nt_f16_pipe_64_s6_bias_residual requires M%{}==0, N%{}==0, K%{}==0",
        v.bm, v.bn, v.bk
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_pipe_64_s6_bias_residual")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let resid_d = g.stream.memcpy_stod(residual)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d).arg(&resid_d);
    unsafe { bld.launch(pipe_cfg(v, m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// **Size-aware `C = act(A·Bᵀ + bias)`** — the `nn.Linear`(+activation) entry point that routes to the
/// fastest fused base per regime: the deep WMMA pipe `pipe_64_s6` ≤1024³ (where the `mma.sync` workhorse
/// base would LOSE the saved-round-trip back to its GEMM deficit) and the `mma.sync` workhorse larger.
/// This is the lever that makes the fused FFN/Linear beat the cuBLAS GEMM+epilogue chain at **every** size,
/// not just 512³/2048³. `act` ∈ {None, ReLU, SiLU, GELU}. Requires (≤1024) `M,N%64==0, K%16==0` or
/// (larger) `M,N%128==0, K%32==0`. Tolerance-gated vs an `act(A·Bᵀ+bias)` f64 reference.
fn gemm_nt_f16_linear_dispatch(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    pipe_entry: &'static str,
    mma: fn(&mut Gpu, &[f32], &[f32], &[f32], usize, usize, usize) -> Result<Vec<f32>, DriverError>,
) -> Result<Vec<f32>, DriverError> {
    if m <= 1024 && n <= 1024 && m % 64 == 0 && n % 64 == 0 && k % 16 == 0 {
        gemm_nt_f16_pipe_fused_bias_v(g, a, b, bias, m, k, n, pipe64(), pipe_entry)
    } else {
        mma(g, a, b, bias, m, k, n)
    }
}

/// `C = A·Bᵀ + bias` (affine `nn.Linear`), size-aware (see [`gemm_nt_f16_linear_dispatch`]).
pub fn gemm_nt_f16_linear(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_linear_dispatch(g, a, b, bias, m, k, n, "wmma_nt_f16_pipe_64_s6_bias", gemm_nt_f16_mma_bias)
}

/// `C = relu(A·Bᵀ + bias)` (Linear+ReLU), size-aware (see [`gemm_nt_f16_linear_dispatch`]).
pub fn gemm_nt_f16_linear_relu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_linear_dispatch(g, a, b, bias, m, k, n, "wmma_nt_f16_pipe_64_s6_bias_relu", gemm_nt_f16_mma_bias_relu)
}

/// `C = silu(A·Bᵀ + bias)` (Linear+SiLU FFN), size-aware (see [`gemm_nt_f16_linear_dispatch`]).
pub fn gemm_nt_f16_linear_silu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_linear_dispatch(g, a, b, bias, m, k, n, "wmma_nt_f16_pipe_64_s6_bias_silu", gemm_nt_f16_mma_bias_silu)
}

/// `C = gelu(A·Bᵀ + bias)` (Linear+GELU FFN), size-aware (see [`gemm_nt_f16_linear_dispatch`]).
pub fn gemm_nt_f16_linear_gelu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_linear_dispatch(g, a, b, bias, m, k, n, "wmma_nt_f16_pipe_64_s6_bias_gelu", gemm_nt_f16_mma_bias_gelu)
}

/// Launch config for the **128×64 dual-B gated-FFN** kernel ([`crate::ptx_wmma::entry_mma_gate`], raster=16,
/// 256 threads): a 1-D grid of `(M/128)·(N/64)` blocks the kernel itself rasterizes into an L2-friendly
/// tile order (`raster=16`), matching the workhorse's [`pipe_cfg`] shape but for the 128×64 gate tile.
fn gate_cfg(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((m / 128) * (n / 64)) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `out = act(x·Wgᵀ [+ bg]) ⊙ (x·Wuᵀ [+ bu])` — the fused **SwiGLU / GeGLU** FFN gate on the 128×64
/// dual-B `mma.sync` kernel ([`crate::ptx_wmma::entry_mma_gate`], fp16). `x` is `[M,K]` activations, the
/// gate/up weights `Wg`,`Wu` are `[N,K]` (the `A·Bᵀ` Linear layout — no host transpose), output `[M,N]`.
/// One staged `x` tile feeds both GEMMs (read once) and the gate fuses into the store, so cuBLAS's
/// three-kernel chain (two GEMMs + an elementwise multiply, both `[M,N]` intermediates round-tripped
/// through HBM) collapses to one kernel — a fusion cuBLAS structurally cannot do. `entry` selects
/// silu/gelu/glu and the bias variant; `bias` carries `(bg[N], bu[N])` for the `*_bias` entries. Requires
/// `M%128==0`, `N%64==0`, `K%32==0`; tolerance-gated vs an `act(x·Wgᵀ+bg)⊙(x·Wuᵀ+bu)` f64 reference.
fn gemm_nt_f16_gate(
    g: &mut Gpu,
    x: &[f32],
    wg: &[f32],
    wu: &[f32],
    bias: Option<(&[f32], &[f32])>,
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use half::f16;
    assert_eq!(x.len(), m * k);
    assert_eq!(wg.len(), n * k);
    assert_eq!(wu.len(), n * k);
    assert!(m % 128 == 0 && n % 64 == 0 && k % 32 == 0, "{entry} requires M%128==0, N%64==0, K%32==0");
    let x16: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
    let wg16: Vec<f16> = wg.iter().map(|&v| f16::from_f32(v)).collect();
    let wu16: Vec<f16> = wu.iter().map(|&v| f16::from_f32(v)).collect();
    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), entry)?;
    let x_d = g.stream.memcpy_stod(&x16)?;
    let wg_d = g.stream.memcpy_stod(&wg16)?;
    let wu_d = g.stream.memcpy_stod(&wu16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&x_d).arg(&wg_d).arg(&wu_d).arg(&mut c_d);
    // The `*_bias` entries take two extra params (pBiasG, pBiasU); the device buffers must outlive launch.
    let (bg_d, bu_d);
    if let Some((bg, bu)) = bias {
        assert_eq!(bg.len(), n, "gate bias bg must have length N");
        assert_eq!(bu.len(), n, "up bias bu must have length N");
        bg_d = g.stream.memcpy_stod(bg)?;
        bu_d = g.stream.memcpy_stod(bu)?;
        bld.arg(&bg_d).arg(&bu_d);
        unsafe { bld.launch(gate_cfg(m, n))? };
    } else {
        unsafe { bld.launch(gate_cfg(m, n))? };
    }
    g.stream.memcpy_dtov(&c_d)
}

/// bf16 twin of [`gemm_nt_f16_gate`] — the gated-FFN gate carried to the training dtype (the dual-B
/// generator is precision-generic). Entries are `mma_nt_bf16_128x64_gate_*`.
fn gemm_nt_bf16_gate(
    g: &mut Gpu,
    x: &[f32],
    wg: &[f32],
    wu: &[f32],
    bias: Option<(&[f32], &[f32])>,
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use half::bf16;
    assert_eq!(x.len(), m * k);
    assert_eq!(wg.len(), n * k);
    assert_eq!(wu.len(), n * k);
    assert!(m % 128 == 0 && n % 64 == 0 && k % 32 == 0, "{entry} requires M%128==0, N%64==0, K%32==0");
    let xb: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    let wgb: Vec<bf16> = wg.iter().map(|&v| bf16::from_f32(v)).collect();
    let wub: Vec<bf16> = wu.iter().map(|&v| bf16::from_f32(v)).collect();
    let f = g.function("wmma_bf16", crate::ptx_wmma::wmma_bf16_ptx(), entry)?;
    let x_d = g.stream.memcpy_stod(&xb)?;
    let wg_d = g.stream.memcpy_stod(&wgb)?;
    let wu_d = g.stream.memcpy_stod(&wub)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&x_d).arg(&wg_d).arg(&wu_d).arg(&mut c_d);
    let (bg_d, bu_d);
    if let Some((bg, bu)) = bias {
        assert_eq!(bg.len(), n, "gate bias bg must have length N");
        assert_eq!(bu.len(), n, "up bias bu must have length N");
        bg_d = g.stream.memcpy_stod(bg)?;
        bu_d = g.stream.memcpy_stod(bu)?;
        bld.arg(&bg_d).arg(&bu_d);
        unsafe { bld.launch(gate_cfg(m, n))? };
    } else {
        unsafe { bld.launch(gate_cfg(m, n))? };
    }
    g.stream.memcpy_dtov(&c_d)
}

/// Fused **SwiGLU** FFN gate (fp16): `silu(x·Wgᵀ) ⊙ (x·Wuᵀ)` — the Llama/Mistral/Gemma FFN gate, one kernel.
pub fn gemm_nt_f16_swiglu(g: &mut Gpu, x: &[f32], wg: &[f32], wu: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_gate(g, x, wg, wu, None, m, k, n, "mma_nt_f16_128x64_gate_silu")
}

/// Fused **GeGLU** FFN gate (fp16): `gelu(x·Wgᵀ) ⊙ (x·Wuᵀ)` (the GLU-with-GELU FFN gate).
pub fn gemm_nt_f16_geglu(g: &mut Gpu, x: &[f32], wg: &[f32], wu: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_f16_gate(g, x, wg, wu, None, m, k, n, "mma_nt_f16_128x64_gate_gelu")
}

/// Fused **SwiGLU** FFN gate (bf16, the training dtype): `silu(x·Wgᵀ) ⊙ (x·Wuᵀ)`.
pub fn gemm_nt_bf16_swiglu(g: &mut Gpu, x: &[f32], wg: &[f32], wu: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_gate(g, x, wg, wu, None, m, k, n, "mma_nt_bf16_128x64_gate_silu")
}

/// Fused **GeGLU** FFN gate (bf16): `gelu(x·Wgᵀ) ⊙ (x·Wuᵀ)`.
pub fn gemm_nt_bf16_geglu(g: &mut Gpu, x: &[f32], wg: &[f32], wu: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_gate(g, x, wg, wu, None, m, k, n, "mma_nt_bf16_128x64_gate_gelu")
}

/// `C = A·Bᵀ + bias + residual` fused into the fast **bf16** mma workhorse — the training-dtype twin of
/// [`gemm_nt_f16_mma_bias_residual`] (the down-proj / output-proj sublayer output).
pub fn gemm_nt_bf16_mma_bias_residual(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::PIPE_BF16;
    use half::bf16;
    let v = &PIPE_BF16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert_eq!(residual.len(), m * n, "residual must have length M·N");
    assert!(
        m % v.bm == 0 && n % v.bn == 0 && k % v.bk == 0,
        "mma_nt_bf16_128_bk32_s2_r16_bias_residual requires M%{}==0, N%{}==0, K%{}==0",
        v.bm, v.bn, v.bk
    );
    let a16: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
    let b16: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
    let f = g.function("wmma_bf16", crate::ptx_wmma::wmma_bf16_ptx(), "mma_nt_bf16_128_bk32_s2_r16_bias_residual")?;
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let resid_d = g.stream.memcpy_stod(residual)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d).arg(&resid_d);
    unsafe { bld.launch(pipe_cfg(v, m, n))? };
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
    // Large bf16 GEMM (A+B ≳ L2): the mma.sync workhorse, the bf16 twin of the f16 dispatch — the cliff
    // fix for the training precision, which otherwise fell through to the un-staged `_mt` path below.
    let ws_bytes = (m * k + n * k) * 2;
    if ws_bytes >= 16 * 1024 * 1024 && m % 128 == 0 && n % 128 == 0 && k % 32 == 0 {
        // Large regime (A+B ≥ 16 MB, ≥2048³): the no-pad ldmatrix+XOR-swizzle **w24** twin is the robust
        // same-run winner over the padded base (1.23× @2048³, 1.13× @4096³ — measured on fp16; bf16 shares
        // the byte-identical `mma.sync` geometry). The w22 2×2 grid was only a noise-tie at 4096³ and lost
        // at 2048³, so the whole regime uses w24. Bit-gated by `mma_swizzle_matches_reference_within_tol`.
        return gemm_nt_bf16_pipe_entry(g, a, b, m, k, n, "mma_nt_bf16_128_bk32_s2_r16_swz");
    }
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

/// `C = A·Bᵀ` (bf16-in, f32-out) via the bf16 `mma.sync` large-GEMM kernel [`crate::ptx_wmma::PIPE_BF16`]
/// — the bf16 twin of [`gemm_nt_f16_pipe`] (padded conflict-free SMEM + r16 rasterization). Requires
/// `M%128==0`, `N%128==0`, `K%32==0`; numerically identical to the other bf16 GEMMs (f32 accumulate),
/// tolerance-gated. Static 40 KiB SMEM ⇒ no dynamic-shared opt-in needed.
pub fn gemm_nt_bf16_pipe(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_pipe_entry(g, a, b, m, k, n, crate::ptx_wmma::PIPE_BF16.name)
}

/// `C = A·Bᵀ` (bf16) via a named `mma.sync` workhorse `entry` launched with the [`crate::ptx_wmma::PIPE_BF16`]
/// config (same tile/raster/threads). `entry` is `mma_nt_bf16_128_bk32_s2_r16` (padded hand-placed) or
/// `…_swz` (the no-pad ldmatrix+XOR-swizzle twin that wins the HBM-bound 4096³); `gemm_nt_bf16` picks by regime.
fn gemm_nt_bf16_pipe_entry(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::PIPE_BF16;
    use half::bf16;
    let v = &PIPE_BF16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % v.bm == 0 && n % v.bn == 0 && k % v.bk == 0,
        "{} requires M%{}==0, N%{}==0, K%{}==0",
        v.name, v.bm, v.bn, v.bk
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
    // The w22 swizzle workhorse is a 2×2 warp grid (128 threads); every other entry is the PIPE_BF16 w24
    // geometry (256 threads). Derive the launch from the entry so the w22 cliff kernel gets the right grid.
    let launch_v = if entry.ends_with("w22swz") {
        crate::ptx_wmma::PipeCfg { wm: 2, wn: 2, ..*v }
    } else {
        *v
    };
    unsafe { bld.launch(pipe_cfg(&launch_v, m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = act(A·Bᵀ + bias)` fused into the **fast bf16 `mma.sync` workhorse** store epilogue
/// (`mma_nt_bf16_128_bk32_s2_r16`, the bf16 large-GEMM champion) — the training-dtype twin of
/// [`gemm_nt_f16_pipe_fused_bias`]. `entry` selects the variant (`..._bias{,_relu,_silu,_gelu}`). The
/// bias is added to the f32 accumulators register-level (known D-fragment column map) then the
/// activation, before the store — the canonical Linear/FFN epilogue cuBLAS needs a 2nd kernel for.
/// Requires `M%128==0`, `N%128==0`, `K%32==0`; tolerance-gated.
fn gemm_nt_bf16_pipe_fused_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_wmma::PIPE_BF16;
    use half::bf16;
    let v = &PIPE_BF16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert!(
        m % v.bm == 0 && n % v.bn == 0 && k % v.bk == 0,
        "{entry} requires M%{}==0, N%{}==0, K%{}==0",
        v.bm, v.bn, v.bk
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
    unsafe { bld.launch(pipe_cfg(v, m, n))? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·Bᵀ + bias` fused into the fast bf16 mma workhorse (affine Linear) — see [`gemm_nt_bf16_pipe_fused_bias`].
pub fn gemm_nt_bf16_mma_bias(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_bf16_128_bk32_s2_r16_bias")
}
/// `C = relu(A·Bᵀ + bias)` fused into the fast bf16 mma workhorse (see [`gemm_nt_bf16_pipe_fused_bias`]).
pub fn gemm_nt_bf16_mma_bias_relu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_bf16_128_bk32_s2_r16_bias_relu")
}
/// `C = silu(A·Bᵀ + bias)` fused into the fast bf16 mma workhorse (see [`gemm_nt_bf16_pipe_fused_bias`]).
pub fn gemm_nt_bf16_mma_bias_silu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_bf16_128_bk32_s2_r16_bias_silu")
}
/// `C = gelu(A·Bᵀ + bias)` fused into the fast bf16 mma workhorse (see [`gemm_nt_bf16_pipe_fused_bias`]).
pub fn gemm_nt_bf16_mma_bias_gelu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_bf16_pipe_fused_bias(g, a, b, bias, m, k, n, "mma_nt_bf16_128_bk32_s2_r16_bias_gelu")
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

/// Pick the flash kernel **entry name and matched launch config** for sequence length `seq`, head dim
/// `d`. `seq >= ptx_flash::FLASH_TILE_MIN` selects the SMEM key-block-tiled kernel (`flash_d{d}_t`,
/// [`ptx_flash::FLASH_TWARPS`] warps/CTA — `W×` less L2 traffic, the long-sequence lever); shorter
/// sequences use the lower-overhead untiled kernel (`flash_d{d}`, [`ptx_flash::FLASH_WARPS`] warps/CTA).
/// Returning name+config together guarantees the block dim always matches the chosen kernel's `W`. The
/// two kernels are bit-identical, so the choice is purely performance. SMEM is static ⇒ `shared_mem_bytes`
/// stays 0. Every flash launch site uses this (struct sites resolve it once at construction).
pub(crate) fn flash_plan(d: usize, seq: usize) -> (String, LaunchConfig) {
    flash_plan_forced(d, seq, seq >= crate::ptx_flash::FLASH_TILE_MIN)
}

/// As [`flash_plan`] but with an explicit `tiled` choice — used by the correctness gate to exercise
/// *both* kernels at the same (ragged) sizes regardless of the dispatch crossover.
pub(crate) fn flash_plan_forced(d: usize, seq: usize, tiled: bool) -> (String, LaunchConfig) {
    let w = if tiled {
        crate::ptx_flash::FLASH_TWARPS
    } else {
        crate::ptx_flash::FLASH_WARPS
    };
    let name = if tiled {
        format!("flash_d{d}_t")
    } else {
        format!("flash_d{d}")
    };
    let cfg = LaunchConfig {
        grid_dim: ((seq as u32).div_ceil(w), 1, 1),
        block_dim: (32 * w, 1, 1),
        shared_mem_bytes: 0,
    };
    (name, cfg)
}

/// Whether [`ResidentLayerF16`] dispatches the **tensor-core flash** (`flash_d64_w`) for this shape:
/// `D == 64`, `S` a multiple of 16, and `S >= 512`. Below 512 the kernel's `S/16` warps can't fill the
/// SM and the tiled f32 flash wins (measured in `flash_tiled_vs_untiled`); at/above it the WMMA `Q·Kᵀ`
/// + `P·V` wins, the margin growing with S (0.92× the tiled @512 → 0.63× @4096). The WMMA path casts
/// Q/K/V to f16 (the tensor-core dtype) — an extra 3 cheap cast launches that the win pays back.
pub(crate) fn wmma_flash_applies(d: usize, s: usize) -> bool {
    d == 64 && s % 16 == 0 && s >= 512
}

/// Tensor-core flash entry name: the **`cp.async`-pipelined register-resident `mma.sync` kernel**
/// `flash_d64_mp` — O/m/l in registers (no SMEM round-trip) *and* the K/V key blocks `cp.async`-staged
/// into double-buffered shared memory so block `kb+1` prefetches under block `kb`'s tensor-core compute.
/// It supersedes `flash_d64_m` (which loaded K/V straight from global and stalled per-block on that
/// latency): the same-run A/B `flash_pipe_vs_mma` measures `mp/m` 0.28×→0.88× single-head S=512→4096
/// (1.14–3.6×, latency-dominated at small S) and 0.84–0.90× at the H=12 GPU-filled regime (the per-warp
/// latency bound). Bit-identical math to `flash_d64_m`, so the gate cross-checks both. Only needs
/// `S % 16 == 0`, which [`wmma_flash_applies`] guarantees; static 16 KB SMEM (no launch param). Shares
/// [`wmma_flash_cfg`] (grid `S/16`, one warp/CTA). `flash_d64_m`/`_w`/`_w4` are retained for the A/B bench.
pub(crate) fn wmma_flash_entry(_s: usize) -> &'static str {
    "flash_d64_mp"
}

/// Launch config for the tensor-core flash kernels (`flash_d64_w`/`_w4`): one warp per 16-query-row block.
pub(crate) fn wmma_flash_cfg(s: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((s / 16) as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Fused **flash-attention** on the GPU: `O = softmax(scale · Q·Kᵀ) · V`, single head, Q/K/V/O all
/// `[seq, d]` row-major. Never materializes the `seq×seq` score matrix — the online-softmax recurrence
/// streams K/V once. `d` must be one of [`ptx_flash::SUPPORTED_D`] (32/64/128). The kernel is chosen by
/// [`flash_plan`] (untiled at short S, SMEM key-block-tiled at long S). Tolerance-gated against a
/// full-softmax CPU reference. The marquee GPU kernel: the fused form that *lost* on CPU (where the
/// tuned GEMM dominates) wins here by never spilling the scores to HBM.
pub fn flash_attn(
    g: &mut Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
) -> Result<Vec<f32>, DriverError> {
    let (entry, cfg) = flash_plan(d, seq);
    flash_attn_run(g, q, k, v, seq, d, scale, &entry, cfg)
}

/// Launch a specific flash kernel (`entry`/`cfg` from [`flash_plan`] or [`flash_plan_forced`]) over
/// host Q/K/V and copy O back. The seam the correctness gate uses to drive *both* kernels at one shape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attn_run(
    g: &mut Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
    entry: &str,
    cfg: LaunchConfig,
) -> Result<Vec<f32>, DriverError> {
    assert_eq!(q.len(), seq * d, "Q must be seq×d");
    assert_eq!(k.len(), seq * d, "K must be seq×d");
    assert_eq!(v.len(), seq * d, "V must be seq×d");
    assert!(
        crate::ptx_flash::SUPPORTED_D.contains(&d),
        "flash_attn: head dim {d} has no generated kernel (supported: {:?})",
        crate::ptx_flash::SUPPORTED_D
    );
    let f = g.function("flash", crate::ptx_flash::flash_ptx(), entry)?;
    let q_d = g.stream.memcpy_stod(q)?;
    let k_d = g.stream.memcpy_stod(k)?;
    let v_d = g.stream.memcpy_stod(v)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; seq * d])?;
    let s = seq as u32;
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

/// Launch grid/block for the SMEM-tiled conv: one CTA per `TILE_P×TILE_Q` output tile per channel `k`
/// (`grid = (ceil(Q/TQ), ceil(P/TP), K)`, `block = (TQ, TP, 1)`). Shared mem is the kernel's own static
/// `.shared` array, so `shared_mem_bytes = 0`. Shared by the launcher and the `conv_vs_peers` bench.
pub(crate) fn conv_tiled_cfg(h: usize, width: usize, k: usize, r: usize, s: usize) -> LaunchConfig {
    use crate::ptx_conv::{kblock, TILE_P, TILE_Q};
    let (p, q) = (h - r + 1, width - s + 1);
    LaunchConfig {
        grid_dim: (
            (q as u32).div_ceil(TILE_Q as u32),
            (p as u32).div_ceil(TILE_P as u32),
            (k / kblock(k)) as u32,
        ),
        block_dim: (TILE_Q as u32, TILE_P as u32, 1),
        shared_mem_bytes: 0,
    }
}

/// 2D **convolution** on the GPU (single batch, stride 1, no padding): input `x` is `[C,H,W]`,
/// weights `w` are `[K,C,R,S]`, output is `[K,P,Q]` with `P=H-R+1`, `Q=W-S+1` — the valid
/// cross-correlation deep-learning calls conv2d. Dispatches the **SMEM-tiled, static-shape-specialized**
/// generator ([`crate::ptx_conv::conv2d_ptx`]) when the staged halo fits shared memory, else falls back
/// to the naive one-thread-per-output kernel. Tolerance-gated (the GPU `fma`-accumulates the C·R·S
/// window in a different order than a serial reference).
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
    let x_d = g.stream.memcpy_stod(x)?;
    let w_d = g.stream.memcpy_stod(w)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; k * p * q])?;

    if crate::ptx_conv::tiled_applies(c, h, width, k, r, s) {
        // Shape-specialized PTX: cache by PTX hash (the in-process module map keys by &'static str,
        // which would alias different shapes), then launch the tiled grid.
        let ptx = crate::ptx_conv::conv2d_ptx(c, h, width, k, r, s);
        let module = g.load_module_cached(&ptx)?;
        let f = module.load_function("conv2d")?;
        let cfg = conv_tiled_cfg(h, width, k, r, s);
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&x_d).arg(&w_d).arg(&mut o_d);
        unsafe { bld.launch(cfg)? };
    } else {
        let total = (k * p * q) as u32;
        let f = g.function("conv2d_naive", crate::ptx_conv::CONV2D, "conv2d")?;
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
    }
    g.stream.memcpy_dtov(&o_d)
}

/// Launch grid/block for the fp16 tensor-core implicit-GEMM conv: one warp per CTA owns a
/// `WMMA_BM×WMMA_BN` output tile (`grid = (ceil(N/BN), ceil(M/BM), 1)`, `block = (32,1,1)`), with
/// `M=K`, `N=P*Q`. Shared by the launcher and the `conv_vs_peers` bench.
pub(crate) fn conv_wmma_cfg(h: usize, width: usize, k: usize, r: usize, s: usize) -> LaunchConfig {
    use crate::ptx_conv::{WMMA_BM, WMMA_BN, WMMA_THREADS};
    let (p, q) = (h - r + 1, width - s + 1);
    let (m, n) = (k, p * q);
    LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(WMMA_BN as u32),
            (m as u32).div_ceil(WMMA_BM as u32),
            1,
        ),
        block_dim: (WMMA_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// **fp16 tensor-core implicit-GEMM** conv2d (single batch, stride 1, no padding). Same contract as
/// [`conv2d`] but the multiplies run on the tensor cores in fp16 with f32 accumulate (so `X`,`W` are
/// rounded to f16 on the host — the price the tensor-core path pays), staging the weights and an
/// on-the-fly im2col of `X` through shared memory ([`crate::ptx_conv::conv_wmma_ptx`]). Tolerance-gated
/// at fp16 precision against the f64 reference.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_wmma(
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
    use half::f16;
    assert_eq!(x.len(), c * h * width, "X must be C×H×W");
    assert_eq!(w.len(), k * c * r * s, "W must be K×C×R×S");
    assert!(h >= r && width >= s, "kernel larger than input");
    let p = h - r + 1;
    let q = width - s + 1;
    let ptx = crate::ptx_conv::conv_wmma_ptx(c, h, width, k, r, s);
    let module = g.load_module_cached(&ptx)?;
    let f = module.load_function("conv2d_wmma")?;
    let x16: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
    let w16: Vec<f16> = w.iter().map(|&v| f16::from_f32(v)).collect();
    let x_d = g.stream.memcpy_stod(&x16)?;
    let w_d = g.stream.memcpy_stod(&w16)?;
    let mut o_d = g.stream.alloc_zeros::<f32>(k * p * q)?;
    let cfg = conv_wmma_cfg(h, width, k, r, s);
    let mut bld = g.stream.launch_builder(&f);
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
    let (flash_name, flash_cfg) = flash_plan(d, s);
    let f_flash = g.function("flash", crate::ptx_flash::flash_ptx(), &flash_name)?;
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
    /// Launch config matched to `f_flash`'s kernel (untiled vs tiled), resolved once by [`flash_plan`]
    /// at construction since the sequence length is fixed for a resident layer.
    flash_cfg: LaunchConfig,
    /// The tensor-core flash kernel (`flash_d64_w`) + its launch config — `Some` when
    /// [`wmma_flash_applies`] (D=64, S≥512, S%16==0). When set, attention runs on the tensor cores
    /// (Q/K/V cast to f16) instead of `f_flash`; the cast is `f_cast`.
    f_flash_w: Option<(CudaFunction, LaunchConfig)>,
    f_gemm: CudaFunction,
    f_silu: CudaFunction,
    f_resid: CudaFunction,
    /// Standalone SiLU (`vmath`) + residual-add (`vadd`) kernels — used only by the unfused reference
    /// path [`forward_device_unfused`](Self::forward_device_unfused), to measure what the fused
    /// epilogues buy. The fused [`forward_device`](Self::forward_device) never launches them.
    f_act: CudaFunction,
    f_vadd: CudaFunction,
    /// Multi-head layout shims (`ptx::HEAD_TRANSPOSE_PTX`): forward f32 `[S,H·dh]`→f16 `[H,S,dh]` (cast
    /// folded in) and inverse f32 `[H,S,dh]`→f32 `[S,H·dh]`. Launched only when `heads > 1`.
    f_qkv_trans: CudaFunction,
    f_attn_trans: CudaFunction,
    wq: cudarc::driver::CudaSlice<half::f16>,
    wk: cudarc::driver::CudaSlice<half::f16>,
    wv: cudarc::driver::CudaSlice<half::f16>,
    wo: cudarc::driver::CudaSlice<half::f16>,
    w1: cudarc::driver::CudaSlice<half::f16>,
    w2: cudarc::driver::CudaSlice<half::f16>,
    s: usize,
    d: usize,
    /// Attention heads and per-head dim `dh = d / heads` (the flash head dim). `heads == 1` is the
    /// original single-head layer (`dh == d`); `heads > 1` runs the tensor-core flash per head.
    heads: usize,
    dh: usize,
    dff: usize,
    eps: f32,
}

impl ResidentLayerF16 {
    /// Upload the weights (narrowed to f16) and preload the kernels — the one-time, `&mut Gpu` setup.
    /// The three WMMA entries (`_sm_db`, `_sm_db_silu`, `_sm_db_residual`) share one JITed module.
    /// Single-head layer — the original API, unchanged. Delegates to [`new_mha`](Self::new_mha) with
    /// `heads = 1` (`dh == d`), preserving every existing caller and the original attention path.
    pub fn new(
        g: &mut Gpu,
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
    ) -> Result<Self, DriverError> {
        Self::new_mha(g, w, s, d, dff, 1)
    }

    /// **Multi-head** pre-norm layer: `heads` attention heads of `dh = d / heads`. The QKV/O projections
    /// are still the full `[D,D]` GEMMs (heads are a reinterpretation of the `D` columns); only attention
    /// runs per-head. Multi-head (`heads > 1`) requires the **tensor-core flash** (`dh == 64`, `S ≥ 512`,
    /// `S % 16 == 0`): that kernel carries the `grid.y = head` offset (`hoff = ctaid.y·S·dh`), while the
    /// f32 fallback flash is single-head only. `heads == 1` is the original single-head layer (`dh == d`,
    /// any supported flash head dim). Q/K/V are bridged token-major↔head-major by the
    /// [`HEAD_TRANSPOSE_PTX`](crate::ptx::HEAD_TRANSPOSE_PTX) shims, so the flash kernel stays untouched.
    pub fn new_mha(
        g: &mut Gpu,
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
        heads: usize,
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
        assert!(heads >= 1 && d % heads == 0, "d={d} must be divisible by heads={heads}");
        let dh = d / heads;
        assert!(
            crate::ptx_flash::SUPPORTED_D.contains(&dh),
            "ResidentLayerF16: head dim dh={dh} (d={d}/heads={heads}) unsupported by flash (need {:?})",
            crate::ptx_flash::SUPPORTED_D
        );
        assert!(
            heads == 1 || wmma_flash_applies(dh, s),
            "multi-head (heads={heads}) needs the tensor-core flash: dh must be 64, S>=512, S%16==0 (got dh={dh}, S={s})"
        );
        let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
        let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
        let f_qkv_trans = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "cast_transpose_qkv")?;
        let f_attn_trans = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "transpose_attn_out")?;
        let (flash_name, flash_cfg) = flash_plan(dh, s);
        let f_flash = g.function("flash", crate::ptx_flash::flash_ptx(), &flash_name)?;
        let f_flash_w = if wmma_flash_applies(dh, s) {
            let f = g.function("flash", crate::ptx_flash::flash_ptx(), wmma_flash_entry(s))?;
            Some((f, wmma_flash_cfg(s)))
        } else {
            None
        };
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
            flash_cfg,
            f_flash_w,
            f_gemm,
            f_silu,
            f_resid,
            f_act,
            f_vadd,
            f_qkv_trans,
            f_attn_trans,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            s,
            d,
            heads,
            dh,
            dff,
            eps: 1e-5,
        })
    }

    /// Narrow a device `[n]` f32 buffer to f16 (the tensor-core flash input dtype) via `f_cast`.
    fn cast16(
        &self,
        src: &cudarc::driver::CudaSlice<f32>,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<half::f16>, DriverError> {
        let mut dst = self.stream.alloc_zeros::<half::f16>(n)?;
        let nn = n as u32;
        let mut b = self.stream.launch_builder(&self.f_cast);
        b.arg(&nn).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// f32 `[S, H·dh]` (token-major) → f16 `[H, S, dh]` (head-major) — the layout the tensor-core flash
    /// reads, with the f32→f16 narrowing folded in (`cast_transpose_qkv`). One pass for each of Q/K/V.
    fn cast_transpose(
        &self,
        src: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<cudarc::driver::CudaSlice<half::f16>, DriverError> {
        let n = self.s * self.d;
        let mut dst = self.stream.alloc_zeros::<half::f16>(n)?;
        let (nn, dd, dhh, sdh) =
            (n as u32, self.d as u32, self.dh as u32, (self.s * self.dh) as u32);
        let mut b = self.stream.launch_builder(&self.f_qkv_trans);
        b.arg(&nn).arg(&dd).arg(&dhh).arg(&sdh).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// f32 `[H, S, dh]` (the flash output, head-major) → f32 `[S, H·dh]` (token-major) — the layout the
    /// O-projection GEMM consumes (`transpose_attn_out`).
    fn transpose_back(
        &self,
        src: &cudarc::driver::CudaSlice<f32>,
    ) -> Result<cudarc::driver::CudaSlice<f32>, DriverError> {
        let n = self.s * self.d;
        let mut dst = self.stream.alloc_zeros::<f32>(n)?;
        let (nn, dd, dhh, sdh) =
            (n as u32, self.d as u32, self.dh as u32, (self.s * self.dh) as u32);
        let mut b = self.stream.launch_builder(&self.f_attn_trans);
        b.arg(&nn).arg(&dd).arg(&dhh).arg(&sdh).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// `O = softmax(scale·Q·Kᵀ)·V` over the resident f32 Q/K/V `[S,D]`, returning a fresh f32 `[S,D]`;
    /// `scale = 1/√dh`. **Single-head** (`heads == 1`): the original path — cast Q/K/V to f16 for the
    /// tensor-core flash, or run the f32 flash directly. **Multi-head** (`heads > 1`): cast-transpose
    /// Q/K/V to head-major f16 `[H,S,dh]`, run the tensor-core flash with `grid.y = heads` (each head an
    /// independent CTA column via the kernel's `hoff = ctaid.y·S·dh`), then transpose the `[H,S,dh]`
    /// output back to `[S,D]`. The seam both forward paths share.
    fn run_attn(
        &self,
        q: &cudarc::driver::CudaSlice<f32>,
        k: &cudarc::driver::CudaSlice<f32>,
        v: &cudarc::driver::CudaSlice<f32>,
        s: usize,
        _d: usize,
    ) -> Result<cudarc::driver::CudaSlice<f32>, DriverError> {
        let scale = 1.0f32 / (self.dh as f32).sqrt();
        let ss = s as u32;
        if self.heads == 1 {
            let mut attn = self.stream.alloc_zeros::<f32>(s * self.d)?;
            if let Some((f_w, cfg_w)) = &self.f_flash_w {
                let q16 = self.cast16(q, s * self.d)?;
                let k16 = self.cast16(k, s * self.d)?;
                let v16 = self.cast16(v, s * self.d)?;
                let mut bld = self.stream.launch_builder(f_w);
                bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut attn);
                unsafe { bld.launch(*cfg_w)? };
            } else {
                let mut bld = self.stream.launch_builder(&self.f_flash);
                bld.arg(&ss).arg(&scale).arg(q).arg(k).arg(v).arg(&mut attn);
                unsafe { bld.launch(self.flash_cfg)? };
            }
            Ok(attn)
        } else {
            // Multi-head: the WMMA flash is guaranteed present (asserted in new_mha). Q/K/V are
            // cast-transposed to head-major f16, flashed with grid.y = heads, then transposed back.
            let (f_w, _) = self
                .f_flash_w
                .as_ref()
                .expect("multi-head requires the tensor-core flash");
            let q_hsd = self.cast_transpose(q)?;
            let k_hsd = self.cast_transpose(k)?;
            let v_hsd = self.cast_transpose(v)?;
            let mut attn_hsd = self.stream.alloc_zeros::<f32>(s * self.d)?;
            let cfg = LaunchConfig {
                grid_dim: ((s / 16) as u32, self.heads as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = self.stream.launch_builder(f_w);
            bld.arg(&ss).arg(&scale).arg(&q_hsd).arg(&k_hsd).arg(&v_hsd).arg(&mut attn_hsd);
            unsafe { bld.launch(cfg)? };
            self.transpose_back(&attn_hsd)
        }
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
        let attn = self.run_attn(&q, &k, &v, s, d)?;
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
        let attn = self.run_attn(&q, &k, &v, s, d)?;
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
    // Pipelined fp8 path (cp.async SMEM staging + padded conflict-free fragments + raster) — the cliff fix
    // carrying the f16/bf16 mma-pipeline recipe to E4M3; takes over once the tile divides the shape.
    use crate::ptx_fp8::{FP8_PIPE_BK, FP8_PIPE_BM, FP8_PIPE_BN};
    if m % FP8_PIPE_BM == 0 && n % FP8_PIPE_BN == 0 && k % FP8_PIPE_BK == 0 {
        return gemm_nt_fp8_pipe(g, a, b, m, k, n);
    }
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

/// **W4A16 weight-only int4 decode** `C = A·dequant(W)ᵀ` (the LLM-decode workhorse). `A` (`[M,K]`)
/// arrives f32 and is rounded to f16; `W` is the group-wise int4-quantized weight `[N,K]` ([`QuantWeight`]
/// from [`crate::ptx_int4`]) — packed 4-bit values, per-group fp16 scales, optional integer zero-points.
/// The kernel reads the **packed int4 weight (4-bit/weight — 4× the fp16 footprint shrink), unpacks it to
/// fp16 on the fly, and runs the identical fp16 tensor-core MMA**; `C` is `[M,N]` f32. Requires
/// `M%64==0`, `N%64==0`, `K%GROUP_SIZE==0`, and `qw.group == GROUP_SIZE` (the kernel bakes the group
/// size). Dispatches the symmetric (`gemm_nt_w4a16`) or asymmetric/zero-point (`gemm_nt_w4a16_z`) entry.
pub fn gemm_nt_w4a16(
    g: &mut Gpu,
    a: &[f32],
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_int4::{GROUP_SIZE, W4_BM, W4_BN, W4_THREADS};
    use half::f16;
    assert_eq!(a.len(), m * k, "A must be M*K");
    assert_eq!(qw.n, n, "weight N mismatch");
    assert_eq!(qw.k, k, "weight K mismatch");
    assert_eq!(qw.group, GROUP_SIZE, "kernel bakes group={GROUP_SIZE}");
    assert!(
        m % W4_BM == 0 && n % W4_BN == 0 && k % GROUP_SIZE == 0,
        "gemm_nt_w4a16 requires M%{W4_BM}==0, N%{W4_BN}==0, K%{GROUP_SIZE}==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let a_d = g.stream.memcpy_stod(&a16)?;
    let bq_d = g.stream.memcpy_stod(&qw.packed)?;
    let scl_d = g.stream.memcpy_stod(&qw.scales)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let cfg = LaunchConfig {
        grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, 1),
        block_dim: (W4_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    match &qw.zeros {
        None => {
            let f = g.function("w4a16", crate::ptx_int4::w4a16_ptx(), "gemm_nt_w4a16")?;
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d);
            unsafe { bld.launch(cfg)? };
        }
        Some(zeros) => {
            let z_d = g.stream.memcpy_stod(zeros)?;
            let f = g.function("w4a16", crate::ptx_int4::w4a16_ptx(), "gemm_nt_w4a16_z")?;
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d).arg(&z_d);
            unsafe { bld.launch(cfg)? };
        }
    }
    g.stream.memcpy_dtov(&c_d)
}

/// **W4A16 with split-K** for the thin-M / small-N decode regime — the dominant LLM-inference shape
/// (tiny M, large K), where the M,N grid alone leaves the SMs idle. `sk` K-splits each compute a partial
/// f32 tile into their **own plane** of an `sk·M·N` buffer (the GEMM CTAs write disjoint planes — no
/// atomics), then a fixed-order reduction kernel sums the planes into C, so the result is **bit-identical
/// run-to-run** (M12) — a float `atomicAdd` split-K could not be. Numerically equals [`gemm_nt_w4a16`]
/// within the fp16-accumulate tolerance (the partials are the same products, only the K-partition
/// differs). Symmetric (no zero-point) path. Requires M%64==0, N%64==0, **K % (sk·128) == 0**, `sk ≥ 1`.
pub fn gemm_nt_w4a16_splitk(
    g: &mut Gpu,
    a: &[f32],
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
    sk: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_int4::{GROUP_SIZE, W4_BM, W4_BN, W4_THREADS};
    use half::f16;
    assert_eq!(a.len(), m * k, "A must be M*K");
    assert_eq!(qw.n, n, "weight N mismatch");
    assert_eq!(qw.k, k, "weight K mismatch");
    assert_eq!(qw.group, GROUP_SIZE, "kernel bakes group={GROUP_SIZE}");
    assert!(qw.zeros.is_none(), "w4a16 split-K is the symmetric path (no zero-point)");
    assert!(sk >= 1, "split count must be >= 1");
    assert!(
        m % W4_BM == 0 && n % W4_BN == 0 && k % (sk * GROUP_SIZE) == 0,
        "gemm_nt_w4a16_splitk requires M%{W4_BM}==0, N%{W4_BN}==0, K%(sk*{GROUP_SIZE})==0"
    );
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let a_d = g.stream.memcpy_stod(&a16)?;
    let bq_d = g.stream.memcpy_stod(&qw.packed)?;
    let scl_d = g.stream.memcpy_stod(&qw.scales)?;
    let mut part_d = g.stream.memcpy_stod(&vec![0f32; sk * m * n])?; // sk disjoint partial planes
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    // split-K GEMM: gridDim.z = sk, each CTA writes its own M×N plane of `part_d`.
    let f = g.function("w4a16_sk", crate::ptx_int4::w4a16_splitk_ptx(), "gemm_nt_w4a16_sk")?;
    let cfg = LaunchConfig {
        grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, sk as u32),
        block_dim: (W4_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    {
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut part_d);
        unsafe { bld.launch(cfg)? };
    }
    // deterministic fixed-order reduction of the sk planes → final C (same cached module).
    let red = g.function("w4a16_sk", crate::ptx_int4::w4a16_splitk_ptx(), "w4a16_splitk_reduce")?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mn, skk) = ((m * n) as u32, sk as u32);
    let rcfg = LaunchConfig { grid_dim: (256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    {
        let mut bld = g.stream.launch_builder(&red);
        bld.arg(&mn).arg(&skk).arg(&part_d).arg(&mut c_d);
        unsafe { bld.launch(rcfg)? };
    }
    g.stream.memcpy_dtov(&c_d)
}

/// **Static-shape-specialized** W4A16 (Mercury's no-library lever, §1A.2): JIT-load a kernel with M/N/K
/// **baked as compile-time constants** for this exact shape, then launch it. Numerically identical to
/// [`gemm_nt_w4a16`] (gated against the same f64 reference), but ptxas strength-reduces the baked strides
/// (`×K`, `×N`, `K/8`, `K/group`) to shifts/immediates — the runtime-multiply overhead a library, which
/// never sees the shape at compile time, cannot remove. Loads a fresh module per call here (the per-shape
/// compile is the static-shape tradeoff; the persistent cubin cache (M10) amortizes it across runs).
pub fn gemm_nt_w4a16_static(
    g: &mut Gpu,
    a: &[f32],
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_int4::{GROUP_SIZE, W4_BM, W4_BN, W4_THREADS};
    use half::f16;
    assert_eq!(a.len(), m * k, "A must be M*K");
    assert_eq!(qw.n, n, "weight N mismatch");
    assert_eq!(qw.k, k, "weight K mismatch");
    assert_eq!(qw.group, GROUP_SIZE, "kernel bakes group={GROUP_SIZE}");
    assert!(
        m % W4_BM == 0 && n % W4_BN == 0 && k % GROUP_SIZE == 0,
        "gemm_nt_w4a16_static requires M%{W4_BM}==0, N%{W4_BN}==0, K%{GROUP_SIZE}==0"
    );
    let zero_point = qw.zeros.is_some();
    let ptx = crate::ptx_int4::w4a16_static_ptx(m, n, k, zero_point);
    let module = g.ctx.load_module(ptx.as_str().into())?;
    let f = module.load_function(crate::ptx_int4::w4a16_static_entry(zero_point))?;
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let a_d = g.stream.memcpy_stod(&a16)?;
    let bq_d = g.stream.memcpy_stod(&qw.packed)?;
    let scl_d = g.stream.memcpy_stod(&qw.scales)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let cfg = LaunchConfig {
        grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, 1),
        block_dim: (W4_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    match &qw.zeros {
        None => {
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d);
            unsafe { bld.launch(cfg)? };
        }
        Some(zeros) => {
            let z_d = g.stream.memcpy_stod(zeros)?;
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d).arg(&z_d);
            unsafe { bld.launch(cfg)? };
        }
    }
    g.stream.memcpy_dtov(&c_d)
}

// ===================================================================================================
// M7 runtime (Phase 7): pooled resident layer. New `impl` block, append-only — it reuses the struct's
// already-preloaded kernels and uploaded weights and only changes *where* the per-op scratch lives
// (a `DevicePool` bump arena instead of per-op `alloc_zeros`). The op sequence, kernels, launch
// configs and dtypes are identical to `forward_device`, so the result is bit-identical (gated) while
// the steady-state inner loop issues zero device alloc/free. This is also the alloc-free body that
// CUDA-graph capture records as pure launches (see `crate::graph`).
// ===================================================================================================
impl ResidentLayerF16 {
    /// [`forward_device`](Self::forward_device) with every intermediate sub-allocated from `pool` and
    /// the `[S,D]` result written into the caller-owned **persistent** `out` (which, like `x_d`, lives
    /// outside the pool so a per-iteration [`reset`](crate::pool::DevicePool::reset) never clobbers it —
    /// the ping-pong a decode loop needs). Does **not** reset the pool; the caller owns the arena's
    /// lifecycle. Every pooled buffer is a full-overwrite output, so the uninitialized
    /// [`alloc`](crate::pool::DevicePool::alloc) fast path is used throughout — the
    /// `resident_layer_pooled_matches_eager` gate proves it by poisoning the slab first.
    pub fn forward_device_pooled(
        &self,
        pool: &mut crate::pool::DevicePool,
        x_d: &cudarc::driver::CudaSlice<f32>,
        out: &mut cudarc::driver::CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        let stream = self.stream.clone();
        self.forward_device_pooled_on(&stream, pool, x_d, out)
    }

    /// As [`forward_device_pooled`](Self::forward_device_pooled) but issuing every launch on an
    /// explicit `stream` (which may differ from the layer's own NULL default stream). The pool's
    /// uninitialized `alloc` path issues no stream work, so *only* these launches land on `stream` —
    /// exactly what [`crate::graph::Graph::capture`] needs: a capturable (non-NULL) stream carrying
    /// nothing but the layer's launches.
    pub fn forward_device_pooled_on(
        &self,
        stream: &Arc<CudaStream>,
        pool: &mut crate::pool::DevicePool,
        x_d: &cudarc::driver::CudaSlice<f32>,
        out: &mut cudarc::driver::CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        use crate::pool::{DevicePool, PoolBuf};
        use cudarc::driver::{CudaSlice, CudaFunction};
        use half::f16;
        let (s, d, dff, eps) = (self.s, self.d, self.dff, self.eps);
        assert_eq!(x_d.len(), s * d, "x_d must be S*D");
        assert_eq!(out.len(), s * d, "out must be S*D");
        let norm_cfg = LaunchConfig { grid_dim: (s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };

        // Pooled equivalents of forward_device's closures. `pool` is threaded as a parameter (not
        // captured) so several pool buffers can be live at once without aliasing a single `&mut`.
        let norm = |pool: &mut DevicePool, src: &CudaSlice<f32>, rows: usize| -> Result<PoolBuf<f32>, DriverError> {
            let mut o = pool.alloc::<f32>(rows * d)?;
            let (r, c) = (rows as u32, d as u32);
            let mut b = stream.launch_builder(&self.f_norm);
            b.arg(&r).arg(&c).arg(&eps).arg(src).arg(&mut *o);
            unsafe { b.launch(norm_cfg)? };
            Ok(o)
        };
        let cast = |pool: &mut DevicePool, src: &CudaSlice<f32>, n: usize| -> Result<PoolBuf<f16>, DriverError> {
            let mut dst = pool.alloc::<f16>(n)?;
            let nn = n as u32;
            let mut b = stream.launch_builder(&self.f_cast);
            b.arg(&nn).arg(src).arg(&mut *dst);
            unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
            Ok(dst)
        };
        let gemm16 = |pool: &mut DevicePool, f: &CudaFunction, a: &CudaSlice<f16>, b: &CudaSlice<f16>, m: usize, k: usize, n: usize| -> Result<PoolBuf<f32>, DriverError> {
            let mut c = pool.alloc::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut *c);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };
        let resid_gemm = |pool: &mut DevicePool, a: &CudaSlice<f16>, b: &CudaSlice<f16>, residual: &CudaSlice<f32>, m: usize, k: usize, n: usize| -> Result<PoolBuf<f32>, DriverError> {
            let mut c = pool.alloc::<f32>(m * n)?;
            let (mm, nn, kk) = (m as u32, n as u32, k as u32);
            let mut bld = stream.launch_builder(&self.f_resid);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a).arg(b).arg(&mut *c).arg(residual);
            unsafe { bld.launch(wmma_sm_cfg(m, n))? };
            Ok(c)
        };

        // --- attention: fp16 tensor-core Q/K/V/O projections, flash, residual fused into O ---
        let h1 = norm(pool, x_d, s)?;
        let h1_16 = cast(pool, &h1, s * d)?;
        let q = gemm16(pool, &self.f_gemm, &h1_16, &self.wq, s, d, d)?;
        let k = gemm16(pool, &self.f_gemm, &h1_16, &self.wk, s, d, d)?;
        let v = gemm16(pool, &self.f_gemm, &h1_16, &self.wv, s, d, d)?;

        // Pooled attention seam — mirrors `run_attn` exactly (single-head f32/tensor-core flash, or
        // multi-head cast-transpose → tensor-core flash with grid.y=heads → transpose back).
        let attn: PoolBuf<f32> = {
            let scale = 1.0f32 / (self.dh as f32).sqrt();
            let ss = s as u32;
            if self.heads == 1 {
                let mut attn = pool.alloc::<f32>(s * d)?;
                if let Some((f_w, cfg_w)) = &self.f_flash_w {
                    let q16 = cast(pool, &q, s * d)?;
                    let k16 = cast(pool, &k, s * d)?;
                    let v16 = cast(pool, &v, s * d)?;
                    let mut bld = stream.launch_builder(f_w);
                    bld.arg(&ss).arg(&scale).arg(&*q16).arg(&*k16).arg(&*v16).arg(&mut *attn);
                    unsafe { bld.launch(*cfg_w)? };
                } else {
                    let mut bld = stream.launch_builder(&self.f_flash);
                    bld.arg(&ss).arg(&scale).arg(&*q).arg(&*k).arg(&*v).arg(&mut *attn);
                    unsafe { bld.launch(self.flash_cfg)? };
                }
                attn
            } else {
                let (f_w, _) = self.f_flash_w.as_ref().expect("multi-head requires the tensor-core flash");
                let cast_transpose = |pool: &mut DevicePool, src: &CudaSlice<f32>| -> Result<PoolBuf<f16>, DriverError> {
                    let n = s * d;
                    let mut dst = pool.alloc::<f16>(n)?;
                    let (nn, dd, dhh, sdh) = (n as u32, d as u32, self.dh as u32, (s * self.dh) as u32);
                    let mut b = stream.launch_builder(&self.f_qkv_trans);
                    b.arg(&nn).arg(&dd).arg(&dhh).arg(&sdh).arg(src).arg(&mut *dst);
                    unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
                    Ok(dst)
                };
                let q_hsd = cast_transpose(pool, &q)?;
                let k_hsd = cast_transpose(pool, &k)?;
                let v_hsd = cast_transpose(pool, &v)?;
                let mut attn_hsd = pool.alloc::<f32>(s * d)?;
                let cfg = LaunchConfig { grid_dim: ((s / 16) as u32, self.heads as u32, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
                let mut bld = stream.launch_builder(f_w);
                bld.arg(&ss).arg(&scale).arg(&*q_hsd).arg(&*k_hsd).arg(&*v_hsd).arg(&mut *attn_hsd);
                unsafe { bld.launch(cfg)? };
                // transpose the [H,S,dh] flash output back to token-major [S,H·dh].
                let mut dst = pool.alloc::<f32>(s * d)?;
                let (nn, dd, dhh, sdh) = ((s * d) as u32, d as u32, self.dh as u32, (s * self.dh) as u32);
                let mut b = stream.launch_builder(&self.f_attn_trans);
                b.arg(&nn).arg(&dd).arg(&dhh).arg(&sdh).arg(&*attn_hsd).arg(&mut *dst);
                unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
                dst
            }
        };

        let attn_16 = cast(pool, &attn, s * d)?;
        let x1 = resid_gemm(pool, &attn_16, &self.wo, x_d, s, d, d)?; // x + A·Woᵀ (residual 1, fused)

        // --- FFN: RMSNorm → SiLU up-projection (fused) → down-projection with residual (fused) ---
        let h2 = norm(pool, &x1, s)?;
        let h2_16 = cast(pool, &h2, s * d)?;
        let f1 = gemm16(pool, &self.f_silu, &h2_16, &self.w1, s, d, dff)?; // SiLU(h2·W1ᵀ), act fused
        let f1_16 = cast(pool, &f1, s * dff)?;
        // residual 2 into the persistent `out`: out = f1·W2ᵀ + x1 (fused), no pool buffer for the result.
        {
            let (mm, nn, kk) = (s as u32, d as u32, dff as u32);
            let mut bld = stream.launch_builder(&self.f_resid);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(&*f1_16).arg(&self.w2).arg(&mut *out).arg(&*x1);
            unsafe { bld.launch(wmma_sm_cfg(s, d))? };
        }
        Ok(())
    }
}

/// `C = A·Bᵀ` (fp8 E4M3 in, f32 out) via the **pipelined** fp8 kernel (`fp8_gemm_pipe`, see
/// [`crate::ptx_fp8::fp8_pipe_entry`]) — cp.async SMEM staging + padded conflict-free `m16n8k32` fragment
/// loads + r16 rasterization, the E4M3 twin of the f16/bf16 mma-pipeline workhorse. Requires
/// `M%128==0`, `N%128==0`, `K%64==0`; numerically identical to the other fp8 GEMMs (f32 accumulate),
/// tolerance-gated. 1-D rasterized grid; static 40 KiB SMEM.
pub fn gemm_nt_fp8_pipe(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_fp8::{
        f32_to_e4m3, FP8_PIPE_BK, FP8_PIPE_BM, FP8_PIPE_BN, FP8_PIPE_M64_BM, FP8_PIPE_THREADS,
    };
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    // Regime-aware tile (see `fp8_pipe_config_sweep_vs_cublaslt`): the 64×128 `_m64` entry's higher
    // occupancy wins for M≤2048 by a wide same-run margin and is the only legal entry when 128∤M but
    // 64∣M; the 128×128 entry ties/wins for the larger M. Both share N%128==0, K%64==0; the chosen tile
    // additionally needs M%BM==0. Same codegen, only BM differs ⇒ bit-identical accumulation.
    let use_m64 = m % FP8_PIPE_BM != 0 || m <= 2048;
    let (bm, entry) = if use_m64 {
        (FP8_PIPE_M64_BM, "fp8_gemm_pipe_m64")
    } else {
        (FP8_PIPE_BM, "fp8_gemm_pipe")
    };
    assert!(
        m % bm == 0 && n % FP8_PIPE_BN == 0 && k % FP8_PIPE_BK == 0,
        "fp8_gemm_pipe requires M%{bm}==0, N%{FP8_PIPE_BN}==0, K%{FP8_PIPE_BK}==0"
    );
    let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
    let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
    let f = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), entry)?;
    let a_d = g.stream.memcpy_stod(&a8)?;
    let b_d = g.stream.memcpy_stod(&b8)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let cfg = LaunchConfig {
        grid_dim: (((m / bm) * (n / FP8_PIPE_BN)) as u32, 1, 1), // 1-D rasterized grid
        block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = act(A·Bᵀ + bias)` (fp8 E4M3 in, f32 out) fused into the pipelined fp8 workhorse store epilogue —
/// the **fastest fused inference path** (Ada runs fp8 `mma.sync` at 2× the fp16 TC rate). `entry` selects
/// the variant (`fp8_gemm_pipe_bias{,_relu,_silu,_gelu}`); bias is added to the f32 accumulators
/// register-level (the m16n8k32 D-fragment column map matches m16n8k16) then the activation, before the
/// store — the canonical fp8 Linear/FFN epilogue cuBLAS needs a 2nd kernel for. Requires `M%128==0`,
/// `N%128==0`, `K%64==0`; tolerance-gated against an `act(e4m3-rounded(A·Bᵀ)+bias)` f64 reference.
fn gemm_nt_fp8_pipe_fused_bias(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_fp8::{f32_to_e4m3, FP8_PIPE_BK, FP8_PIPE_BM, FP8_PIPE_BN, FP8_PIPE_THREADS};
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert!(
        m % FP8_PIPE_BM == 0 && n % FP8_PIPE_BN == 0 && k % FP8_PIPE_BK == 0,
        "{entry} requires M%{FP8_PIPE_BM}==0, N%{FP8_PIPE_BN}==0, K%{FP8_PIPE_BK}==0"
    );
    let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
    let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
    let f = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), entry)?;
    let a_d = g.stream.memcpy_stod(&a8)?;
    let b_d = g.stream.memcpy_stod(&b8)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let cfg = LaunchConfig {
        grid_dim: (((m / FP8_PIPE_BM) * (n / FP8_PIPE_BN)) as u32, 1, 1), // 1-D rasterized grid
        block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// `C = A·Bᵀ + bias` fused, fp8 inputs (affine Linear) — see [`gemm_nt_fp8_pipe_fused_bias`].
pub fn gemm_nt_fp8_mma_bias(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_fp8_pipe_fused_bias(g, a, b, bias, m, k, n, "fp8_gemm_pipe_bias")
}
/// `C = relu(A·Bᵀ + bias)` fused, fp8 inputs (see [`gemm_nt_fp8_pipe_fused_bias`]).
pub fn gemm_nt_fp8_mma_bias_relu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_fp8_pipe_fused_bias(g, a, b, bias, m, k, n, "fp8_gemm_pipe_bias_relu")
}
/// `C = silu(A·Bᵀ + bias)` fused, fp8 inputs (see [`gemm_nt_fp8_pipe_fused_bias`]).
pub fn gemm_nt_fp8_mma_bias_silu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_fp8_pipe_fused_bias(g, a, b, bias, m, k, n, "fp8_gemm_pipe_bias_silu")
}
/// `C = gelu(A·Bᵀ + bias)` fused, fp8 inputs (see [`gemm_nt_fp8_pipe_fused_bias`]).
pub fn gemm_nt_fp8_mma_bias_gelu(g: &mut Gpu, a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_fp8_pipe_fused_bias(g, a, b, bias, m, k, n, "fp8_gemm_pipe_bias_gelu")
}

/// `C = A·Bᵀ + bias + residual` fused, fp8 inputs (E4M3 in, f32 out) — the **fastest down-proj /
/// attention output-proj** (Ada 2× fp8 TC rate); the residual stays f32 (the residual stream), only the
/// GEMM operands are fp8. Bias and residual both fold into the workhorse store; `residual` is the [M,N]
/// skip tensor. Requires `M%128==0`, `N%128==0`, `K%64==0`; tolerance-gated.
pub fn gemm_nt_fp8_mma_bias_residual(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_fp8::{f32_to_e4m3, FP8_PIPE_BK, FP8_PIPE_BM, FP8_PIPE_BN, FP8_PIPE_THREADS};
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(bias.len(), n, "bias must have length N");
    assert_eq!(residual.len(), m * n, "residual must have length M·N");
    assert!(
        m % FP8_PIPE_BM == 0 && n % FP8_PIPE_BN == 0 && k % FP8_PIPE_BK == 0,
        "fp8_gemm_pipe_bias_residual requires M%{FP8_PIPE_BM}==0, N%{FP8_PIPE_BN}==0, K%{FP8_PIPE_BK}==0"
    );
    let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
    let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
    let f = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), "fp8_gemm_pipe_bias_residual")?;
    let a_d = g.stream.memcpy_stod(&a8)?;
    let b_d = g.stream.memcpy_stod(&b8)?;
    let bias_d = g.stream.memcpy_stod(bias)?;
    let resid_d = g.stream.memcpy_stod(residual)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let cfg = LaunchConfig {
        grid_dim: (((m / FP8_PIPE_BM) * (n / FP8_PIPE_BN)) as u32, 1, 1),
        block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d).arg(&bias_d).arg(&resid_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// `out = act(x·Wgᵀ [+bg]) ⊙ (x·Wuᵀ [+bu])` — the fused **SwiGLU/GeGLU** FFN gate on the fp8 dual-B
/// kernel (`fp8_gate_entry`), the **fastest fused inference gate** (Ada 2× fp8 TC rate). x,Wg,Wu round
/// to E4M3, output is f32; one staged x tile feeds both GEMMs (x read once) and the gate fuses into the
/// store — the three-kernel chain cuBLAS needs (two GEMMs + an elementwise multiply, both [M,N]
/// intermediates round-tripped) collapsed to one. `entry` selects silu/gelu/glu ± bias; `bias` carries
/// (bg[N], bu[N]). Requires M%128==0, N%64==0, K%64==0; tolerance-gated vs an
/// act(e4m3-rounded(x·Wgᵀ)+bg) ⊙ (e4m3-rounded(x·Wuᵀ)+bu) f64 reference.
fn gemm_nt_fp8_gate(
    g: &mut Gpu,
    x: &[f32],
    wg: &[f32],
    wu: &[f32],
    bias: Option<(&[f32], &[f32])>,
    m: usize,
    k: usize,
    n: usize,
    entry: &'static str,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_fp8::f32_to_e4m3;
    assert_eq!(x.len(), m * k);
    assert_eq!(wg.len(), n * k);
    assert_eq!(wu.len(), n * k);
    assert!(m % 128 == 0 && n % 64 == 0 && k % 64 == 0, "{entry} requires M%128==0, N%64==0, K%64==0");
    let x8: Vec<u8> = x.iter().map(|&v| f32_to_e4m3(v)).collect();
    let wg8: Vec<u8> = wg.iter().map(|&v| f32_to_e4m3(v)).collect();
    let wu8: Vec<u8> = wu.iter().map(|&v| f32_to_e4m3(v)).collect();
    let f = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), entry)?;
    let x_d = g.stream.memcpy_stod(&x8)?;
    let wg_d = g.stream.memcpy_stod(&wg8)?;
    let wu_d = g.stream.memcpy_stod(&wu8)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let cfg = LaunchConfig {
        grid_dim: (((m / 128) * (n / 64)) as u32, 1, 1), // 1-D rasterized grid (128×64 dual-B tile)
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&x_d).arg(&wg_d).arg(&wu_d).arg(&mut c_d);
    let (bg_d, bu_d);
    if let Some((bg, bu)) = bias {
        assert_eq!(bg.len(), n, "gate bias bg must have length N");
        assert_eq!(bu.len(), n, "up bias bu must have length N");
        bg_d = g.stream.memcpy_stod(bg)?;
        bu_d = g.stream.memcpy_stod(bu)?;
        bld.arg(&bg_d).arg(&bu_d);
        unsafe { bld.launch(cfg)? };
    } else {
        unsafe { bld.launch(cfg)? };
    }
    g.stream.memcpy_dtov(&c_d)
}

/// Fused **SwiGLU** FFN gate (fp8 E4M3 — the fastest fused inference gate): `silu(x·Wgᵀ) ⊙ (x·Wuᵀ)`.
pub fn gemm_nt_fp8_swiglu(g: &mut Gpu, x: &[f32], wg: &[f32], wu: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_fp8_gate(g, x, wg, wu, None, m, k, n, "fp8_gemm_pipe_gate_silu")
}

/// Fused **GeGLU** FFN gate (fp8 E4M3): `gelu(x·Wgᵀ) ⊙ (x·Wuᵀ)`.
pub fn gemm_nt_fp8_geglu(g: &mut Gpu, x: &[f32], wg: &[f32], wu: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, DriverError> {
    gemm_nt_fp8_gate(g, x, wg, wu, None, m, k, n, "fp8_gemm_pipe_gate_gelu")
}

// ---------------------------------------------------------------------------------------------------
// int8 (W8A8) tensor-core GEMM (M3) — `u8` activations × `i8` weights → `i32`, the quantized nn.Linear.
// Mirrors the fp8 launchers ([`fp8_tile`]/[`gemm_nt_fp8`]); int8 shares fp8's `m16n8k32` 8-bit fragment
// layout, retyped `.s32.u8.s8.s32`. The integer accumulate is exact mod 2³² → these are **bit-exact**
// against a CPU `i32` reference, a stronger gate than the float kernels. See [`crate::ptx_int8`].
// ---------------------------------------------------------------------------------------------------

/// One **int8 (W8A8) tensor-core tile** `D = A·B` via `mma.sync.m16n8k32.s32.u8.s8.s32` (Ada has no
/// WMMA int8, same as fp8): `a` is `16×32` **u8** row-major, `b_col` is `32×8` **i8** column-major (the
/// `.col` operand); `D` is `16×8` **i32**. Validates the manual fragment layout — the core a full int8
/// GEMM tiles over. The int8 twin of [`fp8_tile`].
pub fn int8_tile(g: &mut Gpu, a: &[u8], b_col: &[i8]) -> Result<Vec<i32>, DriverError> {
    assert_eq!(a.len(), 16 * 32, "A must be 16×32 (u8)");
    assert_eq!(b_col.len(), 32 * 8, "B must be 32×8 column-major (i8)");
    let f = g.function("int8_tile", crate::ptx_int8::INT8_TILE, "int8_tile")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b_col)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; 16 * 8])?;
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

/// Full **int8 (W8A8) tensor-core `C = A·Bᵀ`** (quantized nn.Linear): `A` (m×k) is **u8** activations,
/// `B` (n×k) is **i8** weights, `C` is **i32** (exact mod 2³² accumulate — bit-exact, no tolerance).
/// Each warp computes a 16×8 tile via `mma.sync.m16n8k32`; the fragment-reuse multi-tile kernel runs
/// when the block divides evenly (the fast path), else the single-tile kernel. Requires m%16==0,
/// n%8==0, k%32==0. Ada's int8 tensor cores run at ~4× the fp16 rate — the lowest-precision inference
/// path. The int8 twin of [`gemm_nt_fp8`].
pub fn gemm_nt_int8(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, DriverError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % 16 == 0 && n % 8 == 0 && k % 32 == 0,
        "int8 GEMM needs M%16==0, N%8==0, K%32==0"
    );
    use crate::ptx_int8::{INT8_TM, INT8_TN};
    let (f, cfg) = if m % (16 * INT8_TM) == 0 && n % (8 * INT8_TN) == 0 {
        (
            g.function(
                "int8_gemm_mt",
                crate::ptx_int8::int8_gemm_mt_ptx(),
                "int8_gemm_nt_mt",
            )?,
            LaunchConfig {
                grid_dim: ((n / (8 * INT8_TN)) as u32, (m / (16 * INT8_TM)) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    } else {
        (
            g.function("int8_gemm", crate::ptx_int8::int8_gemm_ptx(), "int8_gemm_nt")?,
            LaunchConfig {
                grid_dim: ((n / 8) as u32, (m / 16) as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
        )
    };
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
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

/// Launch config for an SMEM-staged + `cp.async` int8 kernel with a `bm×bn` CTA tile and `warps` warps.
fn int8_smdb_cfg(m: usize, n: usize, bm: usize, bn: usize, warps: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n / bn) as u32, (m / bm) as u32, 1),
        block_dim: ((warps * 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// **SMEM-staged + `cp.async` double-buffered int8 GEMM** `C = A·Bᵀ` — the latency-hiding path
/// (cooperative CTA tiles, software-pipelined K-loop), **regime-aware** between two tiles (same split
/// the fp16 `_sm_db`/`_sm128_db` dispatch uses). The **64×64** tile (4 warps) has ~2× the occupancy of
/// the 128×128 variant (128 threads / 32 accumulators vs 256 / 64) and wins while the kernel is
/// latency-bound (small/medium sizes); the **128×128** tile ([`crate::ptx_int8::int8_gemm_smdb128_ptx`])
/// has higher A/B reuse per global load and wins once reuse-bound (large sizes). Measured same-run:
/// 64×64 leads at 1024³/2048³ (~44%/53% of cuBLAS), 128×128 leads at 4096³ (~52% vs ~40%), so the
/// dispatch picks 128×128 when M,N are both ≥ 4096 (and 128-divisible), else 64×64. Same `u8`×`i8`→`i32`
/// bit-exact contract as [`gemm_nt_int8`]. Requires M%64==0, N%64==0, K%INT8_BK==0.
pub fn gemm_nt_int8_smdb(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, DriverError> {
    use crate::ptx_int8::{
        INT8_BK, INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_WARPS_M, INT8_WARPS_M128,
        INT8_WARPS_N, INT8_WARPS_N128,
    };
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % INT8_BM == 0 && n % INT8_BN == 0 && k % INT8_BK == 0,
        "int8 smdb GEMM needs M%{INT8_BM}==0, N%{INT8_BN}==0, K%{INT8_BK}==0"
    );
    // Reuse-bound at large sizes → the bigger 128×128 tile; latency-bound below → the 64×64 tile.
    // The `ldmatrix` + XOR-swizzle (conflict-free SMEM, BK=64) path is a same-run ~1.2-1.5× internal
    // win over the hand-placed fragment loads at every measured size/tile (`int8_swz_vs_handplaced`), so
    // it is the default whenever K%64==0; a K that is only a 32-multiple falls back to the hand-placed
    // BK=32 kernel. Both are bit-exact vs the same i32 oracle, so the choice is purely throughput. The
    // 64×64 tile wins while latency-bound (small/medium), the 128×128 once reuse-bound (≥4096²).
    let use_128 = m >= 4096 && n >= 4096 && m % INT8_BM128 == 0 && n % INT8_BN128 == 0;
    let swz = k % 64 == 0;
    let (ptx, entry, bm, bn, warps): (&'static str, &'static str, usize, usize, usize) =
        match (use_128, swz) {
            (true, true) => (
                crate::ptx_int8::int8_gemm_smdb128_swz_ptx(),
                "int8_gemm_nt_smdb128_swz",
                INT8_BM128,
                INT8_BN128,
                INT8_WARPS_M128 * INT8_WARPS_N128,
            ),
            (true, false) => (
                crate::ptx_int8::int8_gemm_smdb128_ptx(),
                "int8_gemm_nt_smdb128",
                INT8_BM128,
                INT8_BN128,
                INT8_WARPS_M128 * INT8_WARPS_N128,
            ),
            (false, true) => (
                crate::ptx_int8::int8_gemm_smdb_swz_ptx(),
                "int8_gemm_nt_smdb_swz",
                INT8_BM,
                INT8_BN,
                INT8_WARPS_M * INT8_WARPS_N,
            ),
            (false, false) => (
                crate::ptx_int8::int8_gemm_smdb_ptx(),
                "int8_gemm_nt_smdb",
                INT8_BM,
                INT8_BN,
                INT8_WARPS_M * INT8_WARPS_N,
            ),
        };
    let f = g.function(entry, ptx, entry)?;
    let cfg = int8_smdb_cfg(m, n, bm, bn, warps);
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
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

/// **Static-shape int8 (W8A8) GEMM** `C = A·Bᵀ` — the M1 compile-time-shapes lever for int8 (the twin
/// of [`gemm_nt_w4a16_static`]). Bakes M/N/K into the `ldmatrix`+swizzle kernel
/// ([`crate::ptx_int8::int8_gemm_smdb_swz_static_ptx`]) so ptxas constant-folds the hot-loop strides and
/// knows the K trip count; picks the 64×64 / 128×128 swz tile by the same regime rule as
/// [`gemm_nt_int8_smdb`]. **Bit-exact** vs the dynamic kernel (identical codegen, only the dims are
/// constants), so it gates against the same i32 oracle. The per-shape PTX is built + raw-loaded here
/// (not `g.function`-cached). Requires M%bm==0, N%bn==0, K%64==0 (the swz BK).
pub fn gemm_nt_int8_static(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, DriverError> {
    use crate::ptx_int8::{
        INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_WARPS_M, INT8_WARPS_M128, INT8_WARPS_N,
        INT8_WARPS_N128,
    };
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(k % 64 == 0, "gemm_nt_int8_static requires K%64==0 (the swz BK)");
    // Same regime rule as the dynamic swz dispatch: the 128×128 tile once reuse-bound (≥4096²), else 64×64.
    let use_128 = m >= 4096 && n >= 4096 && m % INT8_BM128 == 0 && n % INT8_BN128 == 0;
    let (bm, bn, warps) = if use_128 {
        (INT8_BM128, INT8_BN128, INT8_WARPS_M128 * INT8_WARPS_N128)
    } else {
        (INT8_BM, INT8_BN, INT8_WARPS_M * INT8_WARPS_N)
    };
    assert!(m % bm == 0 && n % bn == 0, "gemm_nt_int8_static requires M%{bm}==0, N%{bn}==0");
    let ptx = crate::ptx_int8::int8_gemm_smdb_swz_static_ptx(m, n, k, use_128);
    let module = g.ctx.load_module(ptx.as_str().into())?;
    let f = module.load_function(crate::ptx_int8::int8_gemm_smdb_swz_static_entry(use_128))?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let cfg = int8_smdb_cfg(m, n, bm, bn, warps);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// **int8 (W8A8) GEMM with explicit split-K** `C = A·Bᵀ` for the thin-M / small-N decode regime, where
/// the M,N grid alone leaves SMs idle. `sk` K-splits each compute a partial product and fold it into C
/// by `red.global.add.u32` — **bit-exact and deterministic** (integer add commutes, unlike a float
/// reduction), so this matches [`gemm_nt_int8`] exactly. Measured same-run (RTX 4050): up to ~2.5× over
/// the un-split swz kernel when the base grid is severely under-filled (M64 N128 K8192, 2 CTAs → sk=8);
/// ~1.9× at 8 CTAs / large K; marginal once the base grid already saturates, and over-splitting past
/// saturation regresses (the `red.add` traffic). The caller (or the Phase-10 autotuner) picks `sk` per
/// shape — rule of thumb `sk ≈ target_ctas / base_ctas`, clamped so each split keeps a few BK=64 slabs.
/// Requires M%64==0, N%64==0, **K % (sk·64) == 0**, `sk ≥ 1`.
pub fn gemm_nt_int8_splitk(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    sk: usize,
) -> Result<Vec<i32>, DriverError> {
    use crate::ptx_int8::{INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N};
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(sk >= 1, "split-K count must be >= 1");
    assert!(
        m % INT8_BM == 0 && n % INT8_BN == 0 && k % (sk * 64) == 0,
        "int8 split-K GEMM needs M%{INT8_BM}==0, N%{INT8_BN}==0, K%(sk*64)==0"
    );
    let f = g.function(
        "int8_gemm_nt_smdb_swz_sk",
        crate::ptx_int8::int8_gemm_smdb_swz_splitk_ptx(),
        "int8_gemm_nt_smdb_swz_sk",
    )?;
    let mut cfg = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, INT8_WARPS_M * INT8_WARPS_N);
    cfg.grid_dim.2 = sk as u32; // gridDim.z K-splits, each folding a partial by red.global.add.u32
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?; // split-K accumulates → C must start zeroed
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// **int8 GEMM + fused per-channel dequant** `out[i,j] = f32(Σ u8·i8) · scale[j]` → **f32** output, in
/// one pass (the SMEM-staged 64×64 kernel with the dequant epilogue folded into the C store). `scale`
/// is the per-output-channel `[N]` f32 scale (symmetric quant). The HBM round-trip cuBLAS int8 needs
/// (separate i32→f32 dequant kernel) is eliminated — the cuBLAS-can't-fuse *beat* lever. Requires
/// M%64==0, N%64==0, K%INT8_BK==0.
pub fn gemm_nt_int8_smdb_dequant(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    scale: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_int8::{INT8_BK, INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N};
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(scale.len(), n, "per-channel scale must be length N");
    assert!(
        m % INT8_BM == 0 && n % INT8_BN == 0 && k % INT8_BK == 0,
        "int8 smdb-dequant GEMM needs M%{INT8_BM}==0, N%{INT8_BN}==0, K%{INT8_BK}==0"
    );
    // Same ldmatrix+swizzle win as the plain int8 GEMM, carried to the fused-dequant epilogue: prefer
    // the conflict-free `_swz_deq` kernel when K%64==0, fall back to the hand-placed BK=32 deq otherwise.
    // Both fold the identical `f32(acc)·scale[j]` store, so the result is unchanged — purely throughput.
    let (ptx, entry) = if k % 64 == 0 {
        (crate::ptx_int8::int8_gemm_smdb_swz_deq_ptx(), "int8_gemm_nt_smdb_swz_deq")
    } else {
        (crate::ptx_int8::int8_gemm_smdb_deq_ptx(), "int8_gemm_nt_smdb_deq")
    };
    let f = g.function(entry, ptx, entry)?;
    let cfg = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, INT8_WARPS_M * INT8_WARPS_N);
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let scale_d = g.stream.memcpy_stod(scale)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm)
        .arg(&nn)
        .arg(&kk)
        .arg(&a_d)
        .arg(&b_d)
        .arg(&mut c_d)
        .arg(&scale_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// **fp8 backward GEMM** `C = dY·Wᵀ` — E5M2 gradient × E4M3 weight, f32 out (the `dX = dY·W` /
/// `dW = dYᵀ·X` building block of an fp8 training step; Session I composes the backward from it). `dy`
/// (`[M,K]`) rounds to E5M2 (wide-range gradient format), `w` (`[N,K]`) to E4M3; both upload as 1-byte
/// fp8 and the tensor core decodes them in hardware. Requires M%(16·FP8_TM)==0, N%(8·FP8_TN)==0,
/// K%32==0. Tolerance-gated vs an f64 reference that decodes the same bits (`fp8_bwd_gemm_matches_reference`).
pub fn gemm_nt_fp8_bwd(
    g: &mut Gpu,
    dy: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DriverError> {
    use crate::ptx_fp8::{FP8_TM, FP8_TN};
    assert_eq!(dy.len(), m * k);
    assert_eq!(w.len(), n * k);
    assert!(
        m % (16 * FP8_TM) == 0 && n % (8 * FP8_TN) == 0 && k % 32 == 0,
        "fp8 backward GEMM needs M%{}==0, N%{}==0, K%32==0",
        16 * FP8_TM,
        8 * FP8_TN
    );
    let a8: Vec<u8> = dy.iter().map(|&x| crate::ptx_fp8_train::f32_to_e5m2(x)).collect();
    let b8: Vec<u8> = w.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
    let f = g.function("fp8_bwd_gemm", crate::ptx_fp8_train::fp8_bwd_gemm_ptx(), "fp8_bwd_gemm_nt")?;
    let a_d = g.stream.memcpy_stod(&a8)?;
    let b_d = g.stream.memcpy_stod(&b8)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let cfg = LaunchConfig {
        grid_dim: ((n / (8 * FP8_TN)) as u32, (m / (16 * FP8_TM)) as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&c_d)
}

/// **Per-tensor `amax`** = `maxᵢ |x[i]|` on-device — the delayed-scaling calibration statistic for fp8
/// training. Launches [`crate::ptx_fp8_train::AMAX_PTX`] over a fixed grid (each thread grid-strides its
/// share, one partial per thread) and takes the host max over the partials. **Deterministic** (fixed
/// grid + exact `max`, no atomics — M12); the result is bit-identical run-to-run and exactly equals the
/// CPU max-abs. (The host final-reduce over `≤1024·256` partials is negligible; a fully device-resident
/// two-level reduce is a later refinement.)
pub fn amax_f32(g: &mut Gpu, x: &[f32]) -> Result<f32, DriverError> {
    let n = x.len();
    if n == 0 {
        return Ok(0.0);
    }
    let threads = 256usize;
    let blocks = n.div_ceil(threads).clamp(1, 1024);
    let total = blocks * threads;
    let x_d = g.stream.memcpy_stod(x)?;
    let mut p_d = g.stream.memcpy_stod(&vec![0f32; total])?;
    let f = g.function("amax_f32", crate::ptx_fp8_train::AMAX_PTX, "amax_f32")?;
    let nn = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&nn).arg(&x_d).arg(&mut p_d);
    unsafe { bld.launch(cfg)? };
    let partials = g.stream.memcpy_dtov(&p_d)?;
    Ok(partials.iter().fold(0f32, |m, &v| m.max(v)))
}

/// **Device delayed-scaling quantize** `out[i] = fp8(x[i]·recip)` — E5M2 if `e5m2` else E4M3 — keeping
/// the whole fp8 cast **on-GPU** via Ada's hardware `cvt.rn.satfinite.e{5m2,4m3}x2.f32` (no host
/// round-trip), the residency the fp8 training step needs. `x.len()` must be even (the converter packs
/// two f32 per `cvt`). Gated to land within one fp8 ULP of the true scaled value
/// (`fp8_device_quantize_within_ulp`).
pub fn quantize_scaled_fp8(
    g: &mut Gpu,
    x: &[f32],
    recip: f32,
    e5m2: bool,
) -> Result<Vec<u8>, DriverError> {
    let n = x.len();
    assert!(n % 2 == 0, "device fp8 quantize needs an even element count");
    if n == 0 {
        return Ok(Vec::new());
    }
    let (ptx, entry) = if e5m2 {
        (crate::ptx_fp8_train::quantize_scaled_e5m2_ptx(), "quantize_scaled_e5m2")
    } else {
        (crate::ptx_fp8_train::quantize_scaled_e4m3_ptx(), "quantize_scaled_e4m3")
    };
    let x_d = g.stream.memcpy_stod(x)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0u8; n])?;
    let f = g.function(entry, ptx, entry)?;
    let threads = 256usize;
    let blocks = (n / 2).div_ceil(threads).clamp(1, 1024);
    let nn = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&nn).arg(&x_d).arg(&mut o_d).arg(&recip);
    unsafe { bld.launch(cfg)? };
    g.stream.memcpy_dtov(&o_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// **Device fp8 quantize gate (first law).** The hw-`cvt`-based delayed-scaling quantize must land
    /// each value within one fp8 ULP of `x·recip` after dequant+unscale (E5M2 ≤ ~1/8 relative, E4M3 ≤
    /// ~1/16) — a round-to-nearest quantizer's guarantee, checked on-device over a sign/scale-mixed
    /// tensor whose magnitude exceeds E4M3's range (so delayed scaling is exercised).
    #[test]
    fn fp8_device_quantize_within_ulp() {
        use crate::ptx_fp8::e4m3_to_f32;
        use crate::ptx_fp8_train::{delayed_scale_recip, e5m2_to_f32, E4M3_MAX, E5M2_MAX};
        with_gpu("fp8_device_quantize", |g| {
            let mut rng = crate::diff::Rng::new(0xF8D);
            let x = rng.vec(4096, -100.0, 100.0);
            let amax = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let r5 = delayed_scale_recip(amax, E5M2_MAX);
            let q5 = quantize_scaled_fp8(g, &x, r5, true).unwrap();
            for (i, &v) in x.iter().enumerate() {
                let rec = e5m2_to_f32(q5[i]) / r5;
                assert!((rec - v).abs() <= v.abs() * 0.13 + 1e-3, "e5m2 dev quant {v} -> {rec}");
            }
            let r4 = delayed_scale_recip(amax, E4M3_MAX);
            let q4 = quantize_scaled_fp8(g, &x, r4, false).unwrap();
            for (i, &v) in x.iter().enumerate() {
                let rec = e4m3_to_f32(q4[i]) / r4;
                assert!((rec - v).abs() <= v.abs() * 0.07 + 1e-3, "e4m3 dev quant {v} -> {rec}");
            }
            eprintln!("[gate] device fp8 quantize (hw cvt e5m2x2/e4m3x2) within ULP ✓");
        });
    }

    /// **amax gate (first law, exact).** The device per-tensor amax must **equal** the CPU max-abs —
    /// `max` reassociates without rounding, so this is a bit-exact `==` check, stronger than a tolerance.
    /// Sizes include `n < blockDim`, a non-grid-aligned `n`, and a large multi-block tensor.
    #[test]
    fn amax_matches_reference() {
        with_gpu("amax", |g| {
            let mut rng = crate::diff::Rng::new(0xAA);
            for n in [1usize, 255, 257, 4096, 100_003] {
                let x = rng.vec(n, -50.0, 50.0);
                let want = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let got = amax_f32(g, &x).unwrap();
                assert_eq!(got, want, "amax n={n}");
            }
            eprintln!("[gate] device amax == CPU max-abs (exact) ✓");
        });
    }
    /// **fp8 backward GEMM gate (first law).** `gemm_nt_fp8_bwd` (E5M2·E4M3, f32 accumulate) must match
    /// an f64 reference that decodes the *same* e5m2/e4m3 bits. The E5M2(3 sig-bit)×E4M3(4 sig-bit)
    /// product is **exact in f32** (7 ≤ 24 bits), and the quantization rounding is matched on both sides,
    /// so the only residual is the f32 accumulation reassociating vs f64 — the **same** honest
    /// `c·√K·ε` bound the E4M3 forward gate uses (`fp8_gemm_matches_reference_within_tol`), **not** an
    /// fp8-slack fudge. Data in [-1,1] (matching the forward gate's fixture); the wide-range benefit of
    /// E5M2 is exercised by the host round-trip test, not this accumulation check.
    #[test]
    fn fp8_bwd_gemm_matches_reference() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        use crate::ptx_fp8_train::{e5m2_to_f32, f32_to_e5m2};
        with_gpu("fp8_bwd_gemm", |g| {
            let mut rng = crate::diff::Rng::new(0xB17D);
            for (m, k, n) in [(32usize, 64usize, 32usize), (64, 96, 64), (96, 128, 32)] {
                let dy = rng.vec(m * k, -1.0, 1.0);
                let w = rng.vec(n * k, -1.0, 1.0);
                let mut want = vec![0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0f64;
                        for kk in 0..k {
                            acc += e5m2_to_f32(f32_to_e5m2(dy[i * k + kk])) as f64
                                * e4m3_to_f32(f32_to_e4m3(w[j * k + kk])) as f64;
                        }
                        want[i * n + j] = acc as f32;
                    }
                }
                let got = gemm_nt_fp8_bwd(g, &dy, &w, m, k, n).unwrap();
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                let st = crate::diff::assert_close(&format!("fp8_bwd {m}x{k}x{n}"), &got, &want, 1e-2, rel);
                eprintln!("fp8_bwd {m}x{k}x{n} (E5M2·E4M3): max_abs={:.2e} max_rel={:.2e}", st.max_abs, st.max_rel);
            }
            eprintln!("[gate] fp8 backward GEMM (E5M2·E4M3) matches f64 reference ✓");
        });
    }

    /// Run `body` with the shared GPU, or skip (printing why) if none is present.
    fn with_gpu(name: &str, body: impl FnOnce(&mut Gpu)) {
        let mut guard = gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => eprintln!("[skip] {name}: no CUDA device reachable"),
        }
    }

    /// **W4A16 correctness gate (the first law).** The int4-decode kernel must reproduce — within the
    /// fp16-accumulate tolerance — an *exact* f64 dequant reference: dequantize the int4 weights with the
    /// same group scales/zero-points on the CPU ([`crate::ptx_int4::reference_w4a16`]) and matmul in f64.
    /// The only legitimate error is the tensor cores' f32 accumulation order (the weight itself is
    /// reconstructed bit-for-bit), so the bound is the same `1e-2 abs / 2e-3 rel` the dense fp16 GEMM
    /// carries — *not* a quantization fudge. Both the symmetric (signed) and asymmetric (zero-point)
    /// paths are gated; a final repeat-run asserts byte-identical output (M12 determinism: fixed grid,
    /// no atomics). A kernel that dequantizes wrong fails here before any speed number is taken.
    #[test]
    fn int4_gemm_matches_reference() {
        use crate::ptx_int4::{
            quantize_weight_asymmetric, quantize_weight_symmetric, reference_w4a16, GROUP_SIZE,
        };
        with_gpu("int4_w4a16", |g| {
            let mut rng = crate::diff::Rng::new(0x174A);
            // (M,N,K): M%64==0, N%64==0, K%128==0. Square-ish + decode-like (small M, big K) + rectangular.
            let shapes = [
                (64usize, 64usize, 128usize),
                (64, 128, 256),
                (128, 256, 256),
                (192, 64, 128),
                (64, 192, 512),
                (256, 128, 384),
            ];
            for (idx, (m, n, k)) in shapes.into_iter().enumerate() {
                let a = rng.vec(m * k, -1.0, 1.0);
                let w = rng.vec(n * k, -0.8, 0.8); // weight [N,K]

                // Symmetric (signed int4).
                let qw = quantize_weight_symmetric(&w, n, k, GROUP_SIZE);
                let c = gemm_nt_w4a16(g, &a, &qw, m, k, n).unwrap();
                let r = reference_w4a16(&a, &qw, m);
                let s = crate::diff::assert_close(
                    &format!("w4a16 sym {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                eprintln!("w4a16 sym  {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);

                // Asymmetric (per-group zero-point) — the AWQ/GPTQ form.
                let qwz = quantize_weight_asymmetric(&w, n, k, GROUP_SIZE);
                let cz = gemm_nt_w4a16(g, &a, &qwz, m, k, n).unwrap();
                let rz = reference_w4a16(&a, &qwz, m);
                let sz = crate::diff::assert_close(
                    &format!("w4a16 asym {m}x{k}x{n}"),
                    &cz,
                    &rz,
                    1e-2,
                    2e-3,
                );
                eprintln!("w4a16 asym {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", sz.max_abs, sz.max_rel);

                // M12 determinism: a second launch is byte-identical (fixed grid, no nondeterministic atomics).
                let c2 = gemm_nt_w4a16(g, &a, &qw, m, k, n).unwrap();
                assert!(
                    c.iter().zip(&c2).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "w4a16 {m}x{k}x{n} not deterministic run-to-run"
                );

                // Static-shape specialization must be bit-identical to the dynamic kernel (same math,
                // only baked constants). Gated on the first 2 shapes (each JIT-compiles a per-shape
                // module) for both the symmetric and zero-point paths.
                if idx < 2 {
                    let cs = gemm_nt_w4a16_static(g, &a, &qw, m, k, n).unwrap();
                    assert!(
                        c.iter().zip(&cs).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "w4a16 static {m}x{k}x{n} (sym) differs from the dynamic kernel"
                    );
                    let csz = gemm_nt_w4a16_static(g, &a, &qwz, m, k, n).unwrap();
                    assert!(
                        cz.iter().zip(&csz).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "w4a16 static {m}x{k}x{n} (asym) differs from the dynamic kernel"
                    );
                    eprintln!("w4a16 static {m}x{k}x{n}: bit-identical to dynamic (sym + asym) ✓");
                }
            }
        });
    }

    /// **W4A16 split-K gate (first law: tolerance + determinism).** The decode-regime split-K path
    /// (`gemm_nt_w4a16_splitk`: `gridDim.z = sk` disjoint partial planes + a fixed-order reduction kernel)
    /// must match the exact f64 dequant reference within the *same* fp16-accumulate tolerance the dense
    /// W4A16 carries (`1e-2 abs / 2e-3 rel`) — the products are identical, only the K-partition and the
    /// f32 reduction differ — AND be **byte-identical run-to-run**: the reduction sums the planes in fixed
    /// z-order, so unlike a float `atomicAdd` split-K it is deterministic (M12). Shapes are decode-like
    /// (small M, large K) with K % (sk·128) == 0 so each split is whole quant groups.
    #[test]
    fn int4_splitk_matches_reference() {
        use crate::ptx_int4::{quantize_weight_symmetric, reference_w4a16, GROUP_SIZE};
        with_gpu("int4_splitk", |g| {
            let mut rng = crate::diff::Rng::new(0x4517);
            for (m, n, k, sk) in [(64usize, 64usize, 256usize, 2usize), (64, 128, 512, 4), (128, 64, 1024, 8)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let w = rng.vec(n * k, -0.8, 0.8);
                let qw = quantize_weight_symmetric(&w, n, k, GROUP_SIZE);
                let r = reference_w4a16(&a, &qw, m);
                let c = gemm_nt_w4a16_splitk(g, &a, &qw, m, k, n, sk).unwrap();
                let s = crate::diff::assert_close(
                    &format!("w4a16 splitk {m}x{k}x{n} sk={sk}"),
                    &c,
                    &r,
                    1e-2,
                    2e-3,
                );
                // M12 determinism: fixed grid + fixed-order plane reduction ⇒ byte-identical.
                let c2 = gemm_nt_w4a16_splitk(g, &a, &qw, m, k, n, sk).unwrap();
                assert!(
                    c.iter().zip(&c2).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "w4a16 splitk {m}x{k}x{n} sk={sk} not deterministic run-to-run"
                );
                eprintln!("w4a16 splitk {m}x{k}x{n} sk={sk}: max_abs={:.2e} max_rel={:.2e}; deterministic ✓", s.max_abs, s.max_rel);
            }
        });
    }

    /// **W4A16 split-K decode occupancy bench (contention-robust internal A/B).** Times the un-split
    /// W4A16 GEMM (sk=1) against the split-K path (`gridDim.z=sk` GEMM **+ the reduction kernel**, both
    /// counted) for decode-like shapes (small M, small-ish N, large K) where the M,N grid alone leaves the
    /// SMs idle. Device-only timing (inputs uploaded once), best-of-N — the internal ratio cancels the
    /// shared clock. The reduction overhead is included, so this is the honest end-to-end split-K speedup.
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int4_splitk_occupancy`
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn int4_splitk_occupancy() {
        use crate::ptx_int4::{quantize_weight_symmetric, GROUP_SIZE, W4_BM, W4_BN, W4_THREADS};
        use half::f16;
        with_gpu("int4_splitk_occupancy", |g| {
            eprintln!("device: {}", g.device_name());
            let mut rng = crate::diff::Rng::new(0x4D0DE);
            const ROUNDS: usize = 8;
            const ITERS: usize = 50;
            let f_base = g.function("w4a16", crate::ptx_int4::w4a16_ptx(), "gemm_nt_w4a16").unwrap();
            let f_sk = g.function("w4a16_sk", crate::ptx_int4::w4a16_splitk_ptx(), "gemm_nt_w4a16_sk").unwrap();
            let f_red = g.function("w4a16_sk", crate::ptx_int4::w4a16_splitk_ptx(), "w4a16_splitk_reduce").unwrap();
            for (m, n, k) in [(64usize, 256usize, 4096usize), (64, 512, 4096), (128, 256, 8192), (64, 128, 8192)] {
                let flop = 2.0 * m as f64 * n as f64 * k as f64;
                let a = rng.vec(m * k, -1.0, 1.0);
                let w = rng.vec(n * k, -0.8, 0.8);
                let qw = quantize_weight_symmetric(&w, n, k, GROUP_SIZE);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let bq_d = g.stream.memcpy_stod(&qw.packed).unwrap();
                let scl_d = g.stream.memcpy_stod(&qw.scales).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let mut part_d = g.stream.memcpy_stod(&vec![0f32; 8 * m * n]).unwrap();
                let (mm, nn, kk) = (m as u32, n as u32, k as u32);
                let base_ctas = (n / W4_BN) * (m / W4_BM);
                eprintln!("\nM{m} N{n} K{k} (base grid = {base_ctas} CTAs):");
                let cfg0 = LaunchConfig { grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, 1), block_dim: (W4_THREADS as u32, 1, 1), shared_mem_bytes: 0 };
                {
                    let mut b = g.stream.launch_builder(&f_base);
                    b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d);
                    unsafe { b.launch(cfg0).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let mut base = f64::INFINITY;
                for _ in 0..ROUNDS {
                    let t0 = Instant::now();
                    for _ in 0..ITERS {
                        let mut b = g.stream.launch_builder(&f_base);
                        b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut c_d);
                        unsafe { b.launch(cfg0).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    base = base.min(t0.elapsed().as_secs_f64() / ITERS as f64);
                }
                eprintln!("  sk=1 (base): {:>7.0} GFLOP/s", flop / base / 1e9);
                for sk in [2usize, 4, 8] {
                    if k % (sk * GROUP_SIZE) != 0 {
                        continue;
                    }
                    let cfg_sk = LaunchConfig { grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, sk as u32), block_dim: (W4_THREADS as u32, 1, 1), shared_mem_bytes: 0 };
                    let rcfg = LaunchConfig { grid_dim: (256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let (mn, skk) = ((m * n) as u32, sk as u32);
                    let mut split = f64::INFINITY;
                    for _ in 0..ROUNDS {
                        let t0 = Instant::now();
                        for _ in 0..ITERS {
                            {
                                let mut b = g.stream.launch_builder(&f_sk);
                                b.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&bq_d).arg(&scl_d).arg(&mut part_d);
                                unsafe { b.launch(cfg_sk).unwrap() };
                            }
                            {
                                let mut b = g.stream.launch_builder(&f_red);
                                b.arg(&mn).arg(&skk).arg(&part_d).arg(&mut c_d);
                                unsafe { b.launch(rcfg).unwrap() };
                            }
                        }
                        g.stream.synchronize().unwrap();
                        split = split.min(t0.elapsed().as_secs_f64() / ITERS as f64);
                    }
                    eprintln!("  sk={sk} ({:>4} CTAs + reduce): {:>7.0} GFLOP/s  → {:.3}× base", base_ctas * sk, flop / split / 1e9, base / split);
                }
            }
        });
    }

    /// Diagnostic: print the driver JIT error log for the W4A16 module (`ptx_int4::w4a16_ptx`) — the
    /// `ptxas` line/error behind a bare `CUDA_ERROR_INVALID_PTX`. Also writes the PTX to a temp file.
    /// `cargo test -p mercury_codegen_gpu --features gpu int4_jit_log -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic; prints the driver JIT log for the W4A16 PTX module"]
    fn int4_jit_log() {
        with_gpu("int4_jitlog", |g| {
            use cudarc::driver::sys;
            g.ctx.bind_to_thread().unwrap();
            let ptx = crate::ptx_int4::w4a16_ptx();
            let dump = std::env::temp_dir().join("mercury_w4a16.ptx");
            let _ = std::fs::write(&dump, ptx);
            eprintln!("wrote PTX to {}", dump.display());
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

    /// Every multi-stage `cp.async` pipeline variant (`entry_smem_pipe`, the large-GEMM-cliff lever) must
    /// match the f16-rounded f64 reference to the same tolerance as the other WMMA GEMMs — they compute
    /// identical math, only with a deeper SMEM ring and (some) a wider staged BK. Per-variant shapes
    /// (divisible by that variant's bm/bn/bk) deliberately stress the corners the pipeline math is most
    /// likely to get wrong: a **single K-tile** (the prologue guards every prefetch but tile 0), a couple
    /// K-tiles, a **deep steady-state K with more tiles than stages** (the ring-buffer offset wrap), and a
    /// rectangular multi-CTA shape (per-warp store indexing + 256/128-thread vectorized staging).
    #[test]
    fn wmma_pipe_matches_reference_within_tol() {
        use crate::ptx_wmma::PIPE_VARIANTS;
        use half::f16;
        with_gpu("wmma_pipe", |g| {
            let mut rng = crate::diff::Rng::new(0x9176);
            for v in PIPE_VARIANTS {
                assert!(
                    v.smem_bytes() <= 48 * 1024,
                    "{}: {}B static SMEM exceeds 48 KiB",
                    v.name,
                    v.smem_bytes()
                );
                let shapes = [
                    (v.bm, v.bk, v.bn),                          // 1 CTA, 1 K-tile (full prologue guard)
                    (v.bm, v.bk * 2, v.bn),                      // 1 CTA, 2 K-tiles
                    (2 * v.bm, v.bk * (v.stages + 3), 2 * v.bn), // 4 CTAs, ring wrap (K-tiles > stages)
                    (v.bm, v.bk * (v.stages + 1), 3 * v.bn),     // rectangular, multi-tile
                ];
                for (m, k, n) in shapes {
                    let a = rng.vec(m * k, -1.0, 1.0);
                    let b = rng.vec(n * k, -1.0, 1.0);
                    let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                    let c = gemm_nt_f16_pipe(g, &a, &b, m, k, n, v).unwrap();
                    let s = crate::diff::assert_close(&format!("{} {m}x{k}x{n}", v.name), &c, &r, 1e-2, 2e-3);
                    eprintln!(
                        "{} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e} (smem={}B)",
                        v.name,
                        s.max_abs,
                        s.max_rel,
                        v.smem_bytes()
                    );
                }
            }
        });
    }

    /// The **`ldmatrix` + XOR-swizzle + no-pad workhorse** (`mma_nt_f16_128_bk32_s2_r16_swz`) must match the
    /// f16-rounded f64 oracle bit-equivalently to the hand-placed/padded workhorse — same `mma.sync`, but the
    /// SMEM is laid out with the XOR swizzle (`chunk ↦ chunk XOR ((row>>1)&3)`, no padding) that the
    /// `ldmatrix.x4`/`.x2` gathers read conflict-free. This gate proves the **staging swizzle and the read
    /// swizzle invert each other** (write/read agree) and the ldmatrix lane→operand contract holds, before
    /// any speed claim. Shapes hit a single K-tile (prologue guard), two K-tiles, a 4-CTA ring wrap, and a
    /// rectangular multi-tile — the same coverage as the hand-placed pipe gate.
    #[test]
    fn mma_swizzle_matches_reference_within_tol() {
        use crate::ptx_wmma::{pipe_variant, PipeCfg};
        use half::f16;
        with_gpu("mma_swizzle", |g| {
            let mut rng = crate::diff::Rng::new(0x5712_BEEF);
            let wh = *pipe_variant("mma_nt_f16_128_bk32_s2_r16");
            let swz = PipeCfg { name: "mma_nt_f16_128_bk32_s2_r16_swz", pad: 0, ..wh };
            for (m, k, n) in [
                (128usize, 32usize, 128usize),
                (128, 64, 128),
                (256, 32 * (wh.stages + 3), 256),
                (128, 32 * 3, 384),
            ] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let c = gemm_nt_f16_pipe(g, &a, &b, m, k, n, &swz).unwrap();
                let s = crate::diff::assert_close(&format!("swz {m}x{k}x{n}"), &c, &r, 1e-2, 2e-3);
                eprintln!("mma_swz {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);
            }
            // bf16 swizzle twin — the same swz path keyed to bf16 (precision-generic); the HBM-bound-4096³
            // win carried to the training dtype. bf16-rounded reference, the wider bf16 tolerance.
            for (m, k, n) in [(128usize, 32usize, 128usize), (256, 160, 256), (128, 96, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| half::bf16::from_f32(x).to_f32());
                let c = gemm_nt_bf16_pipe_entry(g, &a, &b, m, k, n, "mma_nt_bf16_128_bk32_s2_r16_swz").unwrap();
                let s = crate::diff::assert_close(&format!("bf16 swz {m}x{k}x{n}"), &c, &r, 5e-2, 2e-2);
                eprintln!("mma_bf16_swz {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);
            }
        });
    }

    /// The bf16 `mma.sync` large-GEMM kernel (`PIPE_BF16`, dispatched by `gemm_nt_bf16` for A+B ≳ L2) must
    /// match the bf16-rounded f64 reference. Same generator as the f16 mma kernel (precision-generic), so
    /// this confirms the bf16 fragment/mma-type tag and the padded staging↔load consistency at the wider
    /// bf16 tolerance. Shapes hit the %128/%32 divisibility, a single K-tile (prologue guard), the ring
    /// wrap, and a rectangular multi-CTA case.
    #[test]
    fn wmma_bf16_pipe_matches_reference_within_tol() {
        use half::bf16;
        with_gpu("wmma_bf16_pipe", |g| {
            let mut rng = crate::diff::Rng::new(0xB16E);
            let shapes = [(128usize, 32usize, 128usize), (128, 64, 128), (256, 256, 256), (128, 160, 384)];
            for (m, k, n) in shapes {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| bf16::from_f32(x).to_f32());
                let c = gemm_nt_bf16_pipe(g, &a, &b, m, k, n).unwrap();
                let s = crate::diff::assert_close(&format!("bf16 mma {m}x{k}x{n}"), &c, &r, 2e-2, 1e-2);
                eprintln!("bf16 mma {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);
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

    /// The fused `C = act(A·Bᵀ + bias)` epilogues on the **fast `mma.sync` workhorse**
    /// (`gemm_nt_f16_mma_bias{,_relu,_silu,_gelu}`) — the same affine-Linear/FFN forms as
    /// `wmma_sm_db_bias_match_reference_within_tol`, but on the r16-raster mma base (the fastest large-GEMM
    /// path). This is the gate that lets the beat-cuBLAS fused bench trust the fast path: the kernel adds
    /// `bias[col]` to the f32 accumulators (known D-fragment column map) then the activation, so each
    /// output must equal `act(f16-rounded(A·Bᵀ) + bias)`. Workhorse shape constraints: M%128, N%128, K%32.
    #[test]
    fn wmma_mma_bias_match_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_mma_bias", |g| {
            let mut rng = crate::diff::Rng::new(0x3B1A);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 256, 512)] {
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
                        gemm_nt_f16_mma_bias(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&id),
                    ),
                    (
                        "bias_relu",
                        gemm_nt_f16_mma_bias_relu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&|x| x.max(0.0)),
                    ),
                    (
                        "bias_silu",
                        gemm_nt_f16_mma_bias_silu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&silu),
                    ),
                    (
                        "bias_gelu",
                        gemm_nt_f16_mma_bias_gelu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&gelu),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_f16_mma_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        1e-2,
                    );
                    eprintln!(
                        "wmma_f16_mma_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The fused `C = act(A·Bᵀ + bias)` and `A·Bᵀ + bias + residual` epilogues on the **deep WMMA pipe
    /// `pipe_64_s6`** (`wmma_nt_f16_pipe_64_s6_bias{,_relu,_silu,_gelu,_residual}`) — the ≤1024³ GEMM
    /// champion. Same affine-Linear/FFN/down-proj forms as `wmma_mma_bias_match_reference_within_tol`, but
    /// the per-column bias routes through SMEM store-back scratch (WMMA's opaque fragment column map) rather
    /// than the mma kernel's register-level add — so this gate proves that **different** epilogue mechanism
    /// equals `act(f16-rounded(A·Bᵀ) + bias)` (and `+ residual`). `pipe_64_s6` shape constraints: M,N%64, K%16.
    #[test]
    fn wmma_pipe64_bias_match_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_pipe64_bias", |g| {
            let mut rng = crate::diff::Rng::new(0x9164);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            // All ≤1024 and 64-/16-divisible (the pipe_64_s6 regime the size-aware Linear dispatch picks).
            for (m, k, n) in [(64usize, 64usize, 128usize), (128, 256, 256), (512, 128, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                let resid = rng.vec(m * n, -1.0, 1.0);
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
                // Residual reference: (A·Bᵀ + bias) + residual, no activation.
                let mut with_resid = base.clone();
                for i in 0..m {
                    for j in 0..n {
                        with_resid[i * n + j] += bias[j] + resid[i * n + j];
                    }
                }
                let id = |x: f32| x;
                let p = pipe64();
                for (name, got, refv) in [
                    (
                        "bias",
                        gemm_nt_f16_pipe_fused_bias_v(g, &a, &b, &bias, m, k, n, p, "wmma_nt_f16_pipe_64_s6_bias").unwrap(),
                        with_bias(&id),
                    ),
                    (
                        "bias_relu",
                        gemm_nt_f16_pipe_fused_bias_v(g, &a, &b, &bias, m, k, n, p, "wmma_nt_f16_pipe_64_s6_bias_relu").unwrap(),
                        with_bias(&|x| x.max(0.0)),
                    ),
                    (
                        "bias_silu",
                        gemm_nt_f16_pipe_fused_bias_v(g, &a, &b, &bias, m, k, n, p, "wmma_nt_f16_pipe_64_s6_bias_silu").unwrap(),
                        with_bias(&silu),
                    ),
                    (
                        "bias_gelu",
                        gemm_nt_f16_pipe_fused_bias_v(g, &a, &b, &bias, m, k, n, p, "wmma_nt_f16_pipe_64_s6_bias_gelu").unwrap(),
                        with_bias(&gelu),
                    ),
                    (
                        "bias_residual",
                        gemm_nt_f16_pipe64_bias_residual(g, &a, &b, &bias, &resid, m, k, n).unwrap(),
                        with_resid.clone(),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_f16_pipe64_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        1e-2,
                    );
                    eprintln!(
                        "wmma_f16_pipe64_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The fused `C = act(A·Bᵀ + bias)` epilogues on the **fast bf16 `mma.sync` workhorse**
    /// (`gemm_nt_bf16_mma_bias{,_relu,_silu,_gelu}`) — the training-dtype twin of
    /// `wmma_mma_bias_match_reference_within_tol`. The register-level bias epilogue acts on the f32
    /// accumulator (dtype-independent), so each output must equal `act(bf16-rounded(A·Bᵀ) + bias)`; bf16's
    /// wider exponent / fewer mantissa bits only change the input quantization (looser tolerance), not the
    /// per-column map or the SFU activation formulas. Workhorse shape constraints: M%128, N%128, K%32.
    #[test]
    fn wmma_bf16_mma_bias_match_reference_within_tol() {
        use half::bf16;
        with_gpu("wmma_bf16_mma_bias", |g| {
            let mut rng = crate::diff::Rng::new(0x3BF1);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 256, 512)] {
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
                    (
                        "bias",
                        gemm_nt_bf16_mma_bias(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&id),
                    ),
                    (
                        "bias_relu",
                        gemm_nt_bf16_mma_bias_relu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&|x| x.max(0.0)),
                    ),
                    (
                        "bias_silu",
                        gemm_nt_bf16_mma_bias_silu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&silu),
                    ),
                    (
                        "bias_gelu",
                        gemm_nt_bf16_mma_bias_gelu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&gelu),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("wmma_bf16_mma_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        5e-2,
                        2e-2,
                    );
                    eprintln!(
                        "wmma_bf16_mma_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The fused `out = A·Bᵀ + bias + residual` epilogue on the fast mma workhorse (fp16 **and** bf16) —
    /// the transformer down-proj / attention output-proj sublayer output. The residual is added to the
    /// post-bias f32 accumulators (addressed per-element identically to the C store), so each output must
    /// equal `(rounded(A·Bᵀ) + bias[col]) + residual[i]`. This gates BOTH the per-column bias map and the
    /// per-element residual addressing under the mma D-fragment layout. M%128, N%128, K%32.
    #[test]
    fn wmma_mma_bias_residual_match_reference_within_tol() {
        use half::{bf16, f16};
        with_gpu("wmma_mma_bias_residual", |g| {
            let mut rng = crate::diff::Rng::new(0x3B1A_5E51);
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 256, 512)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                let resid = rng.vec(m * n, -1.0, 1.0);
                // Reference: (rounded GEMM) + bias[col] + residual[i] — per the kernel order.
                let reference = |round: &dyn Fn(f32) -> f32| -> Vec<f32> {
                    let mut r = ref_nt_rounded(&a, &b, m, k, n, round);
                    for i in 0..m {
                        for j in 0..n {
                            r[i * n + j] += bias[j] + resid[i * n + j];
                        }
                    }
                    r
                };
                let f16r = reference(&|x| f16::from_f32(x).to_f32());
                let got_f16 = gemm_nt_f16_mma_bias_residual(g, &a, &b, &bias, &resid, m, k, n).unwrap();
                let s = crate::diff::assert_close(
                    &format!("wmma_f16_mma_bias_residual {m}x{k}x{n}"),
                    &got_f16,
                    &f16r,
                    5e-2,
                    1e-2,
                );
                eprintln!(
                    "wmma_f16_mma_bias_residual {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );

                let bf16r = reference(&|x| bf16::from_f32(x).to_f32());
                let got_bf16 = gemm_nt_bf16_mma_bias_residual(g, &a, &b, &bias, &resid, m, k, n).unwrap();
                let s = crate::diff::assert_close(
                    &format!("wmma_bf16_mma_bias_residual {m}x{k}x{n}"),
                    &got_bf16,
                    &bf16r,
                    5e-2,
                    2e-2,
                );
                eprintln!(
                    "wmma_bf16_mma_bias_residual {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
                );
            }
        });
    }

    /// The fused **gated-FFN (SwiGLU/GeGLU/GLU)** kernel `out = act(x·Wgᵀ [+bg]) ⊙ (x·Wuᵀ [+bu])` must
    /// equal the f64 reference computed from the dtype-rounded inputs: `act(round(x·Wgᵀ)[+bg])` times
    /// `round(x·Wuᵀ)[+bu]`. This is the one binding correctness law for the dual-B gate kernel — two
    /// GEMMs sharing one staged A tile, two accumulator sets, the activation on only the gate branch, the
    /// elementwise product fused into the store. The tolerance is looser than a plain GEMM's because the
    /// product of two ~√K-magnitude factors **compounds** their relative errors (and the SFU silu/gelu
    /// approx adds its own ε); the `OR` semantics let large-magnitude lanes pass on relative error and
    /// near-zero lanes on absolute. Exercised for all five variants (silu/gelu/glu, ± bias) in fp16 + bf16.
    #[test]
    fn swiglu_gate_match_reference_within_tol() {
        use half::{bf16, f16};
        with_gpu("swiglu_gate", |g| {
            let mut rng = crate::diff::Rng::new(0x5_71_6C_55);
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            let id = |x: f32| x;
            // M%128, N%64, K%32 — incl. a rectangular case (N=320 stresses the raster edge band).
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 256, 320)] {
                let x = rng.vec(m * k, -1.0, 1.0);
                let wg = rng.vec(n * k, -1.0, 1.0);
                let wu = rng.vec(n * k, -1.0, 1.0);
                let bg = rng.vec(n, -0.5, 0.5);
                let bu = rng.vec(n, -0.5, 0.5);
                // out[i,j] = act(round(x·Wgᵀ)[i,j] + (bg[j] if bias)) · (round(x·Wuᵀ)[i,j] + (bu[j] if bias))
                let gate_ref = |round: &dyn Fn(f32) -> f32, act: &dyn Fn(f32) -> f32, with_bias: bool| -> Vec<f32> {
                    let gp = ref_nt_rounded(&x, &wg, m, k, n, round);
                    let up = ref_nt_rounded(&x, &wu, m, k, n, round);
                    let mut out = vec![0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            let (mut gv, mut uv) = (gp[i * n + j], up[i * n + j]);
                            if with_bias {
                                gv += bg[j];
                                uv += bu[j];
                            }
                            out[i * n + j] = act(gv) * uv;
                        }
                    }
                    out
                };
                // fp16 (inference): all five gate variants.
                let f16r = |x: f32| f16::from_f32(x).to_f32();
                let f16_cases: [(&'static str, &dyn Fn(f32) -> f32, bool); 5] = [
                    ("mma_nt_f16_128x64_gate_silu", &silu, false),
                    ("mma_nt_f16_128x64_gate_gelu", &gelu, false),
                    ("mma_nt_f16_128x64_gate_glu", &id, false),
                    ("mma_nt_f16_128x64_gate_silu_bias", &silu, true),
                    ("mma_nt_f16_128x64_gate_gelu_bias", &gelu, true),
                ];
                for (entry, act, wb) in f16_cases {
                    let bias = if wb { Some((bg.as_slice(), bu.as_slice())) } else { None };
                    let got = gemm_nt_f16_gate(g, &x, &wg, &wu, bias, m, k, n, entry).unwrap();
                    let refv = gate_ref(&f16r, act, wb);
                    // Measured max_abs ≤ 4.3e-4 over these shapes; 5e-3 keeps ~10× margin (every lane
                    // passes on abs), with rel as a secondary guard for any future large-K shape.
                    let s = crate::diff::assert_close(&format!("{entry} {m}x{k}x{n}"), &got, &refv, 5e-3, 2e-2);
                    eprintln!("{entry} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);
                }
                // bf16 (training): the precision-generic twin.
                let bf16r = |x: f32| bf16::from_f32(x).to_f32();
                let bf16_cases: [(&'static str, &dyn Fn(f32) -> f32, bool); 5] = [
                    ("mma_nt_bf16_128x64_gate_silu", &silu, false),
                    ("mma_nt_bf16_128x64_gate_gelu", &gelu, false),
                    ("mma_nt_bf16_128x64_gate_glu", &id, false),
                    ("mma_nt_bf16_128x64_gate_silu_bias", &silu, true),
                    ("mma_nt_bf16_128x64_gate_gelu_bias", &gelu, true),
                ];
                for (entry, act, wb) in bf16_cases {
                    let bias = if wb { Some((bg.as_slice(), bu.as_slice())) } else { None };
                    let got = gemm_nt_bf16_gate(g, &x, &wg, &wu, bias, m, k, n, entry).unwrap();
                    let refv = gate_ref(&bf16r, act, wb);
                    // Measured max_abs ≤ 1.9e-4; 1e-2 keeps wide margin (every lane passes on abs).
                    let s = crate::diff::assert_close(&format!("{entry} {m}x{k}x{n}"), &got, &refv, 1e-2, 3e-2);
                    eprintln!("{entry} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);
                }
            }
        });
    }

    /// The **fp8** fused gated-FFN gate (`fp8_gemm_pipe_gate_*`, the fastest fused inference gate at the
    /// Ada 2× rate) must equal the f64 reference from the e4m3-rounded inputs:
    /// `act(round(x·Wgᵀ)[+bg]) ⊙ (round(x·Wuᵀ)[+bu])`. As with fp16/bf16 the e4m3 rounding is applied in
    /// BOTH the kernel and the reference, so the residual deviation is only the f32-vs-f64 accumulation
    /// order (NOT the fp8 rounding) — the comparison stays tight. All five variants (silu/gelu/glu, ±
    /// bias) in fp8. Constraints: M%128, N%64, K%64.
    #[test]
    fn fp8_swiglu_gate_match_reference_within_tol() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_swiglu_gate", |g| {
            let mut rng = crate::diff::Rng::new(0xF8_5A_7E);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            let id = |x: f32| x;
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 192, 384)] {
                let x = rng.vec(m * k, -1.0, 1.0);
                let wg = rng.vec(n * k, -1.0, 1.0);
                let wu = rng.vec(n * k, -1.0, 1.0);
                let bg = rng.vec(n, -0.5, 0.5);
                let bu = rng.vec(n, -0.5, 0.5);
                let gate_ref = |act: &dyn Fn(f32) -> f32, wb: bool| -> Vec<f32> {
                    let gp = ref_nt_rounded(&x, &wg, m, k, n, round);
                    let up = ref_nt_rounded(&x, &wu, m, k, n, round);
                    let mut out = vec![0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            let (mut gv, mut uv) = (gp[i * n + j], up[i * n + j]);
                            if wb {
                                gv += bg[j];
                                uv += bu[j];
                            }
                            out[i * n + j] = act(gv) * uv;
                        }
                    }
                    out
                };
                let cases: [(&'static str, &dyn Fn(f32) -> f32, bool); 5] = [
                    ("fp8_gemm_pipe_gate_silu", &silu, false),
                    ("fp8_gemm_pipe_gate_gelu", &gelu, false),
                    ("fp8_gemm_pipe_gate_glu", &id, false),
                    ("fp8_gemm_pipe_gate_silu_bias", &silu, true),
                    ("fp8_gemm_pipe_gate_gelu_bias", &gelu, true),
                ];
                for (entry, act, wb) in cases {
                    let bias = if wb { Some((bg.as_slice(), bu.as_slice())) } else { None };
                    let got = gemm_nt_fp8_gate(g, &x, &wg, &wu, bias, m, k, n, entry).unwrap();
                    let refv = gate_ref(act, wb);
                    // fp8 e4m3 is ~3-mantissa-bit, so the single GEMM is ~1e-2 accurate (cf. the fp8 bias
                    // gate) and the **product of two** GEMMs amplifies that: measured max_abs ≤ 9.6e-2 at
                    // K=192. 1.5e-1 keeps every lane passing on abs alone (a real bug = ~tens, so still a
                    // meaningful gate); rel is a secondary guard. This is fp8's honest precision, not slack.
                    let s = crate::diff::assert_close(&format!("{entry} {m}x{k}x{n}"), &got, &refv, 1.5e-1, 6e-2);
                    eprintln!("{entry} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", s.max_abs, s.max_rel);
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

    /// f64 reference for **causal** single-head attention: query `i` attends only to keys `j ≤ i`
    /// (`scores[j>i] = -∞` → `exp = 0`). The independent oracle for `flash_d{d}_mc`.
    fn ref_attn_causal(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize, scale: f32) -> Vec<f32> {
        let mut o = vec![0.0f32; seq * d];
        for i in 0..seq {
            let mut scores = vec![f64::NEG_INFINITY; seq];
            for (j, sc) in scores.iter_mut().enumerate().take(i + 1) {
                let mut acc = 0.0f64;
                for t in 0..d {
                    acc += q[i * d + t] as f64 * k[j * d + t] as f64;
                }
                *sc = acc * scale as f64;
            }
            let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut l = 0.0f64;
            for sc in &mut scores {
                *sc = (*sc - m).exp(); // masked (−∞) keys → 0
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
            // Gate BOTH kernels (untiled + SMEM key-block-tiled) at every shape, regardless of the
            // dispatch crossover — they must agree with the f64 oracle (and, being identical arithmetic,
            // with each other). Ragged seq across the supported head dims: (130,64)/(35,32) are *not*
            // multiples of FLASH_TWARPS(8), so the tiled kernel's final CTA has inactive warps — this
            // exercises the no-deadlock guard (inactive warps still stage K/V + hit both bar.syncs); all
            // are ragged in the key block BK=1024/D, exercising the partial-tail cooperative load. The
            // (768,64) case is at/above FLASH_TILE_MIN so it also confirms the *production* dispatch
            // (`flash_attn` → tiled) is correct at scale.
            for (seq, d) in [
                (128usize, 32usize),
                (200, 64),
                (96, 128),
                (130, 64),
                (35, 32),
                (768, 64),
            ] {
                let q = rng.vec(seq * d, -1.0, 1.0);
                let k = rng.vec(seq * d, -1.0, 1.0);
                let v = rng.vec(seq * d, -1.0, 1.0);
                let scale = 1.0 / (d as f32).sqrt();
                let oracle = ref_attn(&q, &k, &v, seq, d, scale);
                for tiled in [false, true] {
                    let (entry, cfg) = flash_plan_forced(d, seq, tiled);
                    let got =
                        flash_attn_run(g, &q, &k, &v, seq, d, scale, &entry, cfg).unwrap();
                    let s = crate::diff::assert_close(
                        &format!("flash s={seq} d={d} [{entry}]"),
                        &got,
                        &oracle,
                        1e-3,
                        3e-3,
                    );
                    eprintln!(
                        "flash s={seq} d={d} [{entry}]: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// **Standalone `mma.sync.m16n8k16` fp16 fragment-layout probe** — the de-risking step for a
    /// register-resident flash (where O lives in registers, not SMEM). Unlike WMMA (opaque fragments),
    /// `mma.sync` requires the A/B/C fragments hand-placed in the *exact* PTX-ISA per-lane layout, and a
    /// wrong layout JITs fine while mis-addressing → scattered O(1) error. This computes one
    /// `D[16×8] = A[16×16]·B[16×8]` with small integer inputs (f16-exact) and checks it bit-against a CPU
    /// matmul, so the layout is *proven* before any kernel is built on it. Layout (groupID=lane»2,
    /// tg=lane&3, tg2=tg·2): A.row a0/a1/a2/a3 = rows {grp,grp+8}×cols {tg2..,tg2+8..}; B.col b0/b1 = col
    /// grp, K-rows {tg2,tg2+8}; D.f32 d0..d3 = rows {grp,grp+8}×cols {tg2,tg2+1}.
    #[test]
    fn mma_m16n8k16_layout_verifies() {
        use half::f16;
        const MMA_TEST_PTX: &str = "\
.version 7.8\n.target sm_89\n.address_size 64\n\
.visible .entry mma_test(.param .u64 pA, .param .u64 pB, .param .u64 pC)\n{\n\
    .reg .b32 %lane,%grp,%tg,%tg2,%r,%c;\n\
    .reg .b32 %a0,%a1,%a2,%a3,%b0,%b1;\n\
    .reg .f32 %d0,%d1,%d2,%d3;\n\
    .reg .b64 %A,%B,%C,%p,%off;\n\
    ld.param.u64 %A,[pA];\n    ld.param.u64 %B,[pB];\n    ld.param.u64 %C,[pC];\n\
    cvta.to.global.u64 %A,%A;\n    cvta.to.global.u64 %B,%B;\n    cvta.to.global.u64 %C,%C;\n\
    mov.u32 %lane,%tid.x;\n    shr.u32 %grp,%lane,2;\n    and.b32 %tg,%lane,3;\n    shl.b32 %tg2,%tg,1;\n\
    mul.lo.s32 %r,%grp,16;\n    add.s32 %r,%r,%tg2;\n    mul.wide.u32 %off,%r,2;\n    add.s64 %p,%A,%off;\n    ld.global.b32 %a0,[%p];\n\
    add.s32 %r,%grp,8;\n    mul.lo.s32 %r,%r,16;\n    add.s32 %r,%r,%tg2;\n    mul.wide.u32 %off,%r,2;\n    add.s64 %p,%A,%off;\n    ld.global.b32 %a1,[%p];\n\
    mul.lo.s32 %r,%grp,16;\n    add.s32 %r,%r,%tg2;\n    add.s32 %r,%r,8;\n    mul.wide.u32 %off,%r,2;\n    add.s64 %p,%A,%off;\n    ld.global.b32 %a2,[%p];\n\
    add.s32 %r,%grp,8;\n    mul.lo.s32 %r,%r,16;\n    add.s32 %r,%r,%tg2;\n    add.s32 %r,%r,8;\n    mul.wide.u32 %off,%r,2;\n    add.s64 %p,%A,%off;\n    ld.global.b32 %a3,[%p];\n\
    mul.lo.s32 %c,%grp,16;\n    add.s32 %c,%c,%tg2;\n    mul.wide.u32 %off,%c,2;\n    add.s64 %p,%B,%off;\n    ld.global.b32 %b0,[%p];\n\
    mul.lo.s32 %c,%grp,16;\n    add.s32 %c,%c,%tg2;\n    add.s32 %c,%c,8;\n    mul.wide.u32 %off,%c,2;\n    add.s64 %p,%B,%off;\n    ld.global.b32 %b1,[%p];\n\
    mov.f32 %d0,0f00000000;\n    mov.f32 %d1,0f00000000;\n    mov.f32 %d2,0f00000000;\n    mov.f32 %d3,0f00000000;\n\
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%d0,%d1,%d2,%d3},{%a0,%a1,%a2,%a3},{%b0,%b1},{%d0,%d1,%d2,%d3};\n\
    mul.lo.s32 %r,%grp,8;\n    add.s32 %r,%r,%tg2;\n    mul.wide.u32 %off,%r,4;\n    add.s64 %p,%C,%off;\n    st.global.f32 [%p],%d0;\n    st.global.f32 [%p+4],%d1;\n\
    add.s32 %r,%grp,8;\n    mul.lo.s32 %r,%r,8;\n    add.s32 %r,%r,%tg2;\n    mul.wide.u32 %off,%r,4;\n    add.s64 %p,%C,%off;\n    st.global.f32 [%p],%d2;\n    st.global.f32 [%p+4],%d3;\n\
    ret;\n}\n";
        with_gpu("mma_test", |g| {
            // A 16×16 row-major; B 16×8 stored col-major (b_mem[n*16+k] = B[k][n]); small ints (f16-exact).
            let mut a = vec![0f32; 16 * 16];
            for i in 0..16 {
                for k in 0..16 {
                    a[i * 16 + k] = ((i + k) % 5) as f32;
                }
            }
            let mut b = vec![0f32; 16 * 8];
            for k in 0..16 {
                for n in 0..8 {
                    b[n * 16 + k] = ((2 * k + n) % 4) as f32;
                }
            }
            let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
            let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
            let a_d = g.stream.memcpy_stod(&a16).unwrap();
            let b_d = g.stream.memcpy_stod(&b16).unwrap();
            let mut c_d = g.stream.alloc_zeros::<f32>(16 * 8).unwrap();
            let f = g.function("mma_test", MMA_TEST_PTX, "mma_test").unwrap();
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = g.stream.launch_builder(&f);
            bld.arg(&a_d).arg(&b_d).arg(&mut c_d);
            unsafe { bld.launch(cfg).unwrap() };
            let got = g.stream.memcpy_dtov(&c_d).unwrap();
            // reference C[i][n] = Σ_k A[i][k]·B[k][n]
            let mut refc = vec![0f32; 16 * 8];
            for i in 0..16 {
                for n in 0..8 {
                    let mut acc = 0f32;
                    for k in 0..16 {
                        acc += a[i * 16 + k] * b[n * 16 + k];
                    }
                    refc[i * 8 + n] = acc;
                }
            }
            let st = crate::diff::assert_close("mma_m16n8k16", &got, &refc, 1e-3, 1e-3);
            eprintln!("mma.sync.m16n8k16 fp16 layout VERIFIED: max_abs={:.2e}", st.max_abs);
        });
    }

    /// Gate for the **tensor-core (WMMA) flash** kernel `flash_d64_w` (experiment). Q/K/V are f16 (the
    /// tensor-core dtype), so the reference is `ref_attn` over the *same* f16-rounded-back-to-f32 inputs
    /// — isolating the kernel's compute error (f32 accumulation order + the f16 P round-trip), not the
    /// input rounding. The tolerance is looser than the f32 kernels (f16 in), but a mis-mapped WMMA
    /// fragment would scatter O(1) error, which this still catches. `S % 16 == 0` (no ragged key tail).
    #[test]
    fn wmma_flash_matches_reference_within_tol() {
        use half::f16;
        with_gpu("wmma_flash", |g| {
            let mut rng = crate::diff::Rng::new(0x3FA);
            let d = 64usize;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let back = |x: &[f16]| -> Vec<f32> { x.iter().map(|&v| v.to_f32()).collect() };
            for seq in [16usize, 64, 256, 512] {
                let q16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let k16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let v16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let scale = 1.0f32 / (d as f32).sqrt();
                let q_d = g.stream.memcpy_stod(&q16).unwrap();
                let k_d = g.stream.memcpy_stod(&k16).unwrap();
                let v_d = g.stream.memcpy_stod(&v16).unwrap();
                let mut o_d = g.stream.alloc_zeros::<f32>(seq * d).unwrap();
                let f = g
                    .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_w")
                    .unwrap();
                let s32 = seq as u32;
                let cfg = LaunchConfig {
                    grid_dim: ((seq / 16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&s32)
                    .arg(&scale)
                    .arg(&q_d)
                    .arg(&k_d)
                    .arg(&v_d)
                    .arg(&mut o_d);
                unsafe { bld.launch(cfg).unwrap() };
                let got = g.stream.memcpy_dtov(&o_d).unwrap();
                let oracle = ref_attn(&back(&q16), &back(&k16), &back(&v16), seq, d, scale);
                // Tight enough to catch any WMMA fragment-layout regression (which scatters O(0.1+)),
                // loose enough for the f16 input round-trip (observed max_abs ~1e-4 on [-1,1] inputs).
                let st = crate::diff::assert_close(
                    &format!("wmma flash s={seq}"),
                    &got,
                    &oracle,
                    2e-3,
                    2e-2,
                );
                eprintln!(
                    "wmma flash s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
                // The wide-key-tile kernel (64 keys/softmax step) — same online-softmax math, ~2× faster
                // (the long-context lever). It needs S % 64 == 0 (no ragged key tail); the layer's WMMA
                // path is always %64. Must hold to the same tolerance as the 16-key kernel.
                if seq % (16 * crate::ptx_flash::WMMA_FLASH_NKB) == 0 {
                    let mut ow_d = g.stream.alloc_zeros::<f32>(seq * d).unwrap();
                    let fw = g
                        .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_w4")
                        .unwrap();
                    let mut bld = g.stream.launch_builder(&fw);
                    bld.arg(&s32)
                        .arg(&scale)
                        .arg(&q_d)
                        .arg(&k_d)
                        .arg(&v_d)
                        .arg(&mut ow_d);
                    unsafe { bld.launch(cfg).unwrap() };
                    let gotw = g.stream.memcpy_dtov(&ow_d).unwrap();
                    let stw = crate::diff::assert_close(
                        &format!("wmma4 flash s={seq}"),
                        &gotw,
                        &oracle,
                        2e-3,
                        2e-2,
                    );
                    eprintln!(
                        "wmma4 flash s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                        stw.max_abs, stw.max_rel
                    );
                }
            }
        });
    }

    /// Gate for the **register-resident `mma.sync` flash** kernel `flash_d64_m` — the FA2 form with
    /// O/m/l in registers (no SMEM round-trip), hand-placed `mma.sync.m16n8k16` per the layout
    /// `mma_m16n8k16_layout_verifies` proves. Same f16-in setup as `wmma_flash_matches_reference_within_tol`
    /// (reference = `ref_attn` over the f16-rounded inputs, isolating the kernel's compute error). A
    /// mis-placed fragment in the QKᵀ→softmax→PV register dataflow scatters O(0.1+); this catches it.
    /// `S % 16 == 0` (the kernel's query-block stride).
    #[test]
    fn mma_reg_flash_matches_reference_within_tol() {
        use half::f16;
        with_gpu("mma_reg_flash", |g| {
            let mut rng = crate::diff::Rng::new(0x9D2);
            let d = 64usize;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let back = |x: &[f16]| -> Vec<f32> { x.iter().map(|&v| v.to_f32()).collect() };
            for seq in [16usize, 64, 256, 512] {
                let q16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let k16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let v16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let scale = 1.0f32 / (d as f32).sqrt();
                let q_d = g.stream.memcpy_stod(&q16).unwrap();
                let k_d = g.stream.memcpy_stod(&k16).unwrap();
                let v_d = g.stream.memcpy_stod(&v16).unwrap();
                let mut o_d = g.stream.alloc_zeros::<f32>(seq * d).unwrap();
                let f = g
                    .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_m")
                    .unwrap();
                let s32 = seq as u32;
                let cfg = LaunchConfig {
                    grid_dim: ((seq / 16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&s32)
                    .arg(&scale)
                    .arg(&q_d)
                    .arg(&k_d)
                    .arg(&v_d)
                    .arg(&mut o_d);
                unsafe { bld.launch(cfg).unwrap() };
                let got = g.stream.memcpy_dtov(&o_d).unwrap();
                let oracle = ref_attn(&back(&q16), &back(&k16), &back(&v16), seq, d, scale);
                let st = crate::diff::assert_close(
                    &format!("mma_reg flash s={seq}"),
                    &got,
                    &oracle,
                    2e-3,
                    2e-2,
                );
                eprintln!(
                    "mma_reg flash s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// Gate for the **causal** register-resident flash `flash_d64_mc`: query `i` attends only to keys
    /// `j ≤ i`. Same f16-in setup as the non-causal gate, but vs `ref_attn_causal`. Exercises both the
    /// diagonal-block per-score mask and the upper-block skip (the loop stops at `kb==row`); a wrong mask
    /// or an off-by-one in the key/query index comparison shifts the causal boundary and the gate fails.
    #[test]
    fn mma_reg_flash_causal_matches_reference_within_tol() {
        use half::f16;
        with_gpu("mma_reg_flash_causal", |g| {
            let mut rng = crate::diff::Rng::new(0xCA05A1);
            let d = 64usize;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let back = |x: &[f16]| -> Vec<f32> { x.iter().map(|&v| v.to_f32()).collect() };
            for seq in [16usize, 64, 256, 512] {
                let q16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let k16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let v16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let scale = 1.0f32 / (d as f32).sqrt();
                let q_d = g.stream.memcpy_stod(&q16).unwrap();
                let k_d = g.stream.memcpy_stod(&k16).unwrap();
                let v_d = g.stream.memcpy_stod(&v16).unwrap();
                let mut o_d = g.stream.alloc_zeros::<f32>(seq * d).unwrap();
                let f = g
                    .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_mc")
                    .unwrap();
                let s32 = seq as u32;
                let cfg = LaunchConfig {
                    grid_dim: ((seq / 16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&s32)
                    .arg(&scale)
                    .arg(&q_d)
                    .arg(&k_d)
                    .arg(&v_d)
                    .arg(&mut o_d);
                unsafe { bld.launch(cfg).unwrap() };
                let got = g.stream.memcpy_dtov(&o_d).unwrap();
                let oracle = ref_attn_causal(&back(&q16), &back(&k16), &back(&v16), seq, d, scale);
                let st = crate::diff::assert_close(
                    &format!("mma_reg causal flash s={seq}"),
                    &got,
                    &oracle,
                    2e-3,
                    2e-2,
                );
                eprintln!(
                    "mma_reg causal flash s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// **Multi-head** gate for `flash_d64_m`: launch `grid.y = H` heads over a `[H,S,D]` buffer; each
    /// head's `[S,D]` slice must match the single-head f64 oracle. Confirms the `ctaid.y · S · D`
    /// head-base-offset addressing — the occupancy lever that fills the GPU at small S (where one head's
    /// `S/16` blocks can't). H heads share nothing, so head `h`'s output depends only on its own slice.
    #[test]
    fn mma_reg_flash_multihead_matches_reference_within_tol() {
        use half::f16;
        with_gpu("mma_reg_flash_mh", |g| {
            let mut rng = crate::diff::Rng::new(0x4EAD5);
            let d = 64usize;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let back = |x: &[f16]| -> Vec<f32> { x.iter().map(|&v| v.to_f32()).collect() };
            for (heads, seq) in [(4usize, 64usize), (12, 128), (8, 256)] {
                let n = heads * seq * d;
                let q16 = to16(&rng.vec(n, -1.0, 1.0));
                let k16 = to16(&rng.vec(n, -1.0, 1.0));
                let v16 = to16(&rng.vec(n, -1.0, 1.0));
                let scale = 1.0f32 / (d as f32).sqrt();
                let q_d = g.stream.memcpy_stod(&q16).unwrap();
                let k_d = g.stream.memcpy_stod(&k16).unwrap();
                let v_d = g.stream.memcpy_stod(&v16).unwrap();
                let mut o_d = g.stream.alloc_zeros::<f32>(n).unwrap();
                let f = g
                    .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_m")
                    .unwrap();
                let s32 = seq as u32;
                let cfg = LaunchConfig {
                    grid_dim: ((seq / 16) as u32, heads as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&s32).arg(&scale).arg(&q_d).arg(&k_d).arg(&v_d).arg(&mut o_d);
                unsafe { bld.launch(cfg).unwrap() };
                let got = g.stream.memcpy_dtov(&o_d).unwrap();
                let (qf, kf, vf) = (back(&q16), back(&k16), back(&v16));
                let mut max_abs = 0f64;
                for h in 0..heads {
                    let sl = h * seq * d..(h + 1) * seq * d;
                    let oracle = ref_attn(&qf[sl.clone()], &kf[sl.clone()], &vf[sl.clone()], seq, d, scale);
                    let st = crate::diff::assert_close(
                        &format!("mh flash H={heads} s={seq} h={h}"),
                        &got[sl],
                        &oracle,
                        2e-3,
                        2e-2,
                    );
                    max_abs = max_abs.max(st.max_abs);
                }
                eprintln!("mh flash H={heads} s={seq} d={d}: max_abs={max_abs:.2e}");
            }
        });
    }

    /// Gate for the **`cp.async`-pipelined, SMEM-staged** register-resident flash kernels `flash_d64_mp`
    /// (non-causal) and `flash_d64_mpc` (causal). Same f16-in setup and tolerance as
    /// `mma_reg_flash_matches_reference_within_tol`; the pipelined kernel does identical math to
    /// `flash_d64_m` but reads K/V from `cp.async`-staged shared memory with a double-buffered prefetch,
    /// so a wrong SMEM address, a buffer-swap bug, or a missed `cp.async.wait_group`/`bar.sync` would
    /// scatter O — this catches it. Exercises S=16 (single block, no prefetch) through S=512 (the
    /// steady-state pipeline), both query-block parities.
    #[test]
    fn mma_pipe_flash_matches_reference_within_tol() {
        use half::f16;
        with_gpu("mma_pipe_flash", |g| {
            let mut rng = crate::diff::Rng::new(0x9117E);
            let d = 64usize;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let back = |x: &[f16]| -> Vec<f32> { x.iter().map(|&v| v.to_f32()).collect() };
            for seq in [16usize, 64, 256, 512] {
                let q16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let k16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let v16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let scale = 1.0f32 / (d as f32).sqrt();
                let (qf, kf, vf) = (back(&q16), back(&k16), back(&v16));
                let q_d = g.stream.memcpy_stod(&q16).unwrap();
                let k_d = g.stream.memcpy_stod(&k16).unwrap();
                let v_d = g.stream.memcpy_stod(&v16).unwrap();
                let s32 = seq as u32;
                let cfg = LaunchConfig {
                    grid_dim: ((seq / 16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                for (entry, causal) in [("flash_d64_mp", false), ("flash_d64_mpc", true)] {
                    let mut o_d = g.stream.alloc_zeros::<f32>(seq * d).unwrap();
                    let f = g
                        .function("flash", crate::ptx_flash::flash_ptx(), entry)
                        .unwrap();
                    let mut bld = g.stream.launch_builder(&f);
                    bld.arg(&s32).arg(&scale).arg(&q_d).arg(&k_d).arg(&v_d).arg(&mut o_d);
                    unsafe { bld.launch(cfg).unwrap() };
                    let got = g.stream.memcpy_dtov(&o_d).unwrap();
                    let oracle = if causal {
                        ref_attn_causal(&qf, &kf, &vf, seq, d, scale)
                    } else {
                        ref_attn(&qf, &kf, &vf, seq, d, scale)
                    };
                    let st = crate::diff::assert_close(
                        &format!("{entry} s={seq}"),
                        &got,
                        &oracle,
                        2e-3,
                        2e-2,
                    );
                    eprintln!(
                        "{entry} s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                        st.max_abs, st.max_rel
                    );
                }
            }
        });
    }

    /// Gate for the **multi-warp-CTA** pipelined flash `flash_d64_mp4` / `flash_d64_mp8` (W warps share
    /// one CTA and its cooperatively-staged K/V block; each warp owns its own 16-query-row block). Same
    /// math as `flash_d64_mp` so it matches `ref_attn`; exercises the ragged final CTA (S=16 with W=4/8 →
    /// only warp 0 active, the rest stage + barrier but skip compute) and the steady CTA (S=512). A
    /// buffer-swap race, a missing barrier, or a botched active-warp predicate scatters O — caught here.
    #[test]
    fn mma_pipe_mw_flash_matches_reference_within_tol() {
        use half::f16;
        with_gpu("mma_pipe_mw_flash", |g| {
            let mut rng = crate::diff::Rng::new(0x3FA57);
            let d = 64usize;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let back = |x: &[f16]| -> Vec<f32> { x.iter().map(|&v| v.to_f32()).collect() };
            for seq in [16usize, 64, 256, 512] {
                let q16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let k16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let v16 = to16(&rng.vec(seq * d, -1.0, 1.0));
                let scale = 1.0f32 / (d as f32).sqrt();
                let (qf, kf, vf) = (back(&q16), back(&k16), back(&v16));
                let oracle = ref_attn(&qf, &kf, &vf, seq, d, scale);
                let q_d = g.stream.memcpy_stod(&q16).unwrap();
                let k_d = g.stream.memcpy_stod(&k16).unwrap();
                let v_d = g.stream.memcpy_stod(&v16).unwrap();
                let s32 = seq as u32;
                for (warps, entry) in [(4u32, "flash_d64_mp4"), (8, "flash_d64_mp8")] {
                    let mut o_d = g.stream.alloc_zeros::<f32>(seq * d).unwrap();
                    let f = g.function("flash", crate::ptx_flash::flash_ptx(), entry).unwrap();
                    let blocks = (seq / 16) as u32;
                    let cfg = LaunchConfig {
                        grid_dim: (blocks.div_ceil(warps), 1, 1),
                        block_dim: (32 * warps, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut bld = g.stream.launch_builder(&f);
                    bld.arg(&s32).arg(&scale).arg(&q_d).arg(&k_d).arg(&v_d).arg(&mut o_d);
                    unsafe { bld.launch(cfg).unwrap() };
                    let got = g.stream.memcpy_dtov(&o_d).unwrap();
                    let st = crate::diff::assert_close(
                        &format!("{entry} s={seq}"),
                        &got,
                        &oracle,
                        2e-3,
                        2e-2,
                    );
                    eprintln!(
                        "{entry} s={seq} d={d}: max_abs={:.2e} max_rel={:.2e}",
                        st.max_abs, st.max_rel
                    );
                }
            }
        });
    }

    /// **Same-process A/B: the `cp.async`-pipelined flash (`flash_d64_mp`) vs the global-load
    /// register-resident flash (`flash_d64_m`)** — single-head and multi-head (H=12, the GPU-filled
    /// regime where attention is per-warp latency-bound and the prefetch should pay off). One pinned
    /// clock (the only honest comparison; ~7x laptop clock swing). Both kernels are bit-identical in
    /// math, so it first asserts they agree, then reports `mp/m` (below 1.0 ⇒ the pipeline wins) and
    /// each kernel's GFLOP/s (`4·S²·D·H`). Run:
    /// `… --features gpu --release -- --ignored --nocapture flash_pipe_vs_mma`.
    #[test]
    #[ignore = "tuning bench; run explicitly"]
    fn flash_pipe_vs_mma() {
        with_gpu("flash_pipe_vs_mma", |g| {
            let mut rng = crate::diff::Rng::new(0x717E5);
            let d = 64usize;
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
            let f_m = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_m")
                .unwrap();
            let f_mp = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_mp")
                .unwrap();
            let to16 =
                |x: &[f32]| -> Vec<half::f16> { x.iter().map(|&v| half::f16::from_f32(v)).collect() };
            // (heads, [seq lengths]) — single-head sweep, then the H=12 GPU-filled sweep.
            for (heads, seqs) in [(1usize, &[512usize, 1024, 2048, 4096][..]), (12, &[512, 1024, 2048][..])] {
                for &s in seqs {
                    let n = heads * s * d;
                    let qf = rng.vec(n, -1.0, 1.0);
                    let kf = rng.vec(n, -1.0, 1.0);
                    let vf = rng.vec(n, -1.0, 1.0);
                    let q16 = g.stream.memcpy_stod(&to16(&qf)).unwrap();
                    let k16 = g.stream.memcpy_stod(&to16(&kf)).unwrap();
                    let v16 = g.stream.memcpy_stod(&to16(&vf)).unwrap();
                    let mut o = g.stream.memcpy_stod(&vec![0f32; n]).unwrap();
                    let scale = 1.0f32 / (d as f32).sqrt();
                    let ss = s as u32;
                    let cfg = LaunchConfig {
                        grid_dim: ((s / 16) as u32, heads as u32, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // correctness: m and mp must agree (bit-identical math, SMEM vs global load only).
                    {
                        let mut b = g.stream.launch_builder(&f_m);
                        b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                        unsafe { b.launch(cfg).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    let out_m = g.stream.memcpy_dtov(&o).unwrap();
                    {
                        let mut b = g.stream.launch_builder(&f_mp);
                        b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                        unsafe { b.launch(cfg).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    let out_mp = g.stream.memcpy_dtov(&o).unwrap();
                    let dmax = out_m
                        .iter()
                        .zip(&out_mp)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(dmax < 5e-3, "H={heads} S={s}: mp vs m disagree, max_abs={dmax:.2e}");

                    pin!();
                    let t_m = best_of(5, || {
                        let t0 = Instant::now();
                        for _ in 0..100 {
                            let mut b = g.stream.launch_builder(&f_m);
                            b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                            unsafe { b.launch(cfg).unwrap() };
                        }
                        g.stream.synchronize().unwrap();
                        t0.elapsed().as_secs_f64() / 100.0
                    });
                    pin!();
                    let t_mp = best_of(5, || {
                        let t0 = Instant::now();
                        for _ in 0..100 {
                            let mut b = g.stream.launch_builder(&f_mp);
                            b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                            unsafe { b.launch(cfg).unwrap() };
                        }
                        g.stream.synchronize().unwrap();
                        t0.elapsed().as_secs_f64() / 100.0
                    });
                    let gf = 4.0 * (s as f64) * (s as f64) * (d as f64) * (heads as f64);
                    eprintln!(
                        "H={heads:>2} S={s:>4}: m {:.4} ({:>6.0} GFLOP/s) | mp {:.4} ({:>6.0} GFLOP/s) || mp/m {:.2}x",
                        t_m * 1e3,
                        gf / t_m / 1e9,
                        t_mp * 1e3,
                        gf / t_mp / 1e9,
                        t_mp / t_m,
                    );
                }
            }
        });
    }

    /// **Same-process A/B: multi-warp-CTA flash (`flash_d64_mp4` / `_mp8`) vs the 1-warp pipelined
    /// `flash_d64_mp`** — does sharing one cooperatively-staged K/V block across W warps (cutting K/V
    /// global traffic W×) beat the 1-warp kernel once `cp.async` has hidden the latency? One pinned clock;
    /// `mpW/mp` below 1.0 ⇒ the multi-warp kernel wins. Asserts all three agree first. Single-head and
    /// the H=12 filled regime. Run: `… --features gpu --release -- --ignored --nocapture flash_mw_vs_mp`.
    #[test]
    #[ignore = "tuning bench; run explicitly"]
    fn flash_mw_vs_mp() {
        with_gpu("flash_mw_vs_mp", |g| {
            let mut rng = crate::diff::Rng::new(0x317E5);
            let d = 64usize;
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            macro_rules! pin {
                () => {{ for _ in 0..40 { gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap(); } }};
            }
            for _ in 0..1500 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            let to16 = |x: &[f32]| -> Vec<half::f16> { x.iter().map(|&v| half::f16::from_f32(v)).collect() };
            // (entry, warps): warps=1 is the 1-warp baseline (block 32, grid.x = S/16).
            let variants = [("flash_d64_mp", 1u32), ("flash_d64_mp4", 4), ("flash_d64_mp8", 8)];
            let funcs: Vec<_> = variants
                .iter()
                .map(|(e, _)| g.function("flash", crate::ptx_flash::flash_ptx(), e).unwrap())
                .collect();
            let cfg_for = |w: u32, s: usize, heads: u32| LaunchConfig {
                grid_dim: (((s / 16) as u32).div_ceil(w), heads, 1),
                block_dim: (32 * w, 1, 1),
                shared_mem_bytes: 0,
            };
            for (heads, seqs) in [(1u32, &[512usize, 1024, 2048, 4096][..]), (12, &[512, 1024, 2048][..])] {
                for &s in seqs {
                    let n = heads as usize * s * d;
                    let qf = rng.vec(n, -1.0, 1.0);
                    let kf = rng.vec(n, -1.0, 1.0);
                    let vf = rng.vec(n, -1.0, 1.0);
                    let q16 = g.stream.memcpy_stod(&to16(&qf)).unwrap();
                    let k16 = g.stream.memcpy_stod(&to16(&kf)).unwrap();
                    let v16 = g.stream.memcpy_stod(&to16(&vf)).unwrap();
                    let mut o = g.stream.memcpy_stod(&vec![0f32; n]).unwrap();
                    let scale = 1.0f32 / (d as f32).sqrt();
                    let ss = s as u32;
                    // launch helper as a macro (no closure capturing g across pin!()).
                    macro_rules! launch {
                        ($i:expr, $w:expr) => {{
                            let mut b = g.stream.launch_builder(&funcs[$i]);
                            b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                            unsafe { b.launch(cfg_for($w, s, heads)).unwrap() };
                        }};
                    }
                    // baseline = mp (1 warp); all variants must agree with it.
                    launch!(0, 1);
                    g.stream.synchronize().unwrap();
                    let base = g.stream.memcpy_dtov(&o).unwrap();
                    let mut times = vec![];
                    for (i, (_, w)) in variants.iter().enumerate() {
                        let w = *w;
                        launch!(i, w);
                        g.stream.synchronize().unwrap();
                        let out = g.stream.memcpy_dtov(&o).unwrap();
                        let dmax = base.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                        assert!(dmax < 5e-3, "H={heads} S={s} {}: disagree max_abs={dmax:.2e}", variants[i].0);
                        pin!();
                        let t = best_of(5, || {
                            let t0 = Instant::now();
                            for _ in 0..100 { launch!(i, w); }
                            g.stream.synchronize().unwrap();
                            t0.elapsed().as_secs_f64() / 100.0
                        });
                        times.push(t);
                    }
                    let gf = 4.0 * (s as f64) * (s as f64) * (d as f64) * (heads as f64);
                    eprintln!(
                        "H={heads:>2} S={s:>4}: mp {:.4} ({:>5.0}) | mp4 {:.4} ({:>5.0}) {:.2}x | mp8 {:.4} ({:>5.0}) {:.2}x  [ms (GFLOP/s) mpW/mp]",
                        times[0] * 1e3, gf / times[0] / 1e9,
                        times[1] * 1e3, gf / times[1] / 1e9, times[1] / times[0],
                        times[2] * 1e3, gf / times[2] / 1e9, times[2] / times[0],
                    );
                }
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
                // production dispatch: untiled below FLASH_TILE_MIN, SMEM key-block-tiled at/above it.
                let (fname, cfg) = flash_plan(d, seq);
                let f = g
                    .function("flash", crate::ptx_flash::flash_ptx(), &fname)
                    .unwrap();
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
                    "flash s={seq} d={d} [{fname}]: {:.2} ms/iter, {:.0} GFLOP/s (fused, no s² scores in HBM)",
                    spi * 1e3,
                    flop / spi / 1e9
                );
            }
        });
    }

    /// **Same-process A/B of the two flash kernels** — untiled `flash_d64` vs SMEM key-block-tiled
    /// `flash_d64_t` — across sequence length under one pinned clock. This is the honest measurement
    /// behind [`ptx_flash::FLASH_TILE_MIN`]: cross-*run* flash comparisons are corrupted by the ~7×
    /// laptop-GPU clock swing, so untiled and tiled are timed back-to-back in the SAME process with a
    /// re-pin before each `best_of(5)`. Buffers are resident (uploaded once). The two kernels are
    /// bit-identical by construction, so the bench first *asserts* their outputs agree at every S
    /// (covering S far past the correctness gate's sizes), then reports `tiled/untiled` — the crossover
    /// is where that ratio drops below 1.0. Run: `… --ignored --nocapture flash_tiled_vs_untiled`.
    #[test]
    #[ignore = "tuning bench; run explicitly"]
    fn flash_tiled_vs_untiled() {
        with_gpu("flash_tiled_vs_untiled", |g| {
            let mut rng = crate::diff::Rng::new(0x7117ED);
            let d = 64usize;
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
            let f_unt = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64")
                .unwrap();
            let f_til = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_t")
                .unwrap();
            // The tensor-core kernels (f16 in). Same attention, WMMA `Q·Kᵀ` + `P·V`; `_w` stages 16 keys
            // per softmax step, `_w4` stages 64 (4× fewer serial KB iterations — the long-context lever).
            let f_wmma = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_w")
                .unwrap();
            let f_wmma4 = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_w4")
                .unwrap();
            let to16 = |x: &[f32]| -> Vec<half::f16> {
                x.iter().map(|&v| half::f16::from_f32(v)).collect()
            };
            for s in [256usize, 384, 512, 768, 1024, 1536, 2048, 4096] {
                let qf = rng.vec(s * d, -1.0, 1.0);
                let kf = rng.vec(s * d, -1.0, 1.0);
                let vf = rng.vec(s * d, -1.0, 1.0);
                let q = g.stream.memcpy_stod(&qf).unwrap();
                let k = g.stream.memcpy_stod(&kf).unwrap();
                let v = g.stream.memcpy_stod(&vf).unwrap();
                let q16 = g.stream.memcpy_stod(&to16(&qf)).unwrap();
                let k16 = g.stream.memcpy_stod(&to16(&kf)).unwrap();
                let v16 = g.stream.memcpy_stod(&to16(&vf)).unwrap();
                let mut o = g.stream.memcpy_stod(&vec![0f32; s * d]).unwrap();
                let scale = 1.0f32 / (d as f32).sqrt();
                let ss = s as u32;
                let (_, cfg_unt) = flash_plan_forced(d, s, false);
                let (_, cfg_til) = flash_plan_forced(d, s, true);
                let cfg_w = LaunchConfig {
                    grid_dim: ((s / 16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };

                // Correctness at scale: untiled and tiled differ only in data movement ⇒ identical output.
                {
                    let mut bld = g.stream.launch_builder(&f_unt);
                    bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                    unsafe { bld.launch(cfg_unt).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_unt = g.stream.memcpy_dtov(&o).unwrap();
                {
                    let mut bld = g.stream.launch_builder(&f_til);
                    bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                    unsafe { bld.launch(cfg_til).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_til = g.stream.memcpy_dtov(&o).unwrap();
                let maxdiff = out_unt
                    .iter()
                    .zip(&out_til)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    maxdiff < 1e-4,
                    "S={s}: tiled vs untiled disagree, max_abs={maxdiff:.2e}"
                );
                // The wide WMMA kernel (f16 in) must match the f32 reference within the f16 round-trip
                // tolerance (every A/B size is %64, so the 64-key tile has no ragged tail). A
                // fragment-layout or indexing bug would scatter O(0.1+); the online-softmax regrouping is
                // exact, so the only honest deviation is f16 input quantization (~1e-3 here).
                {
                    let mut bld = g.stream.launch_builder(&f_wmma4);
                    bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                    unsafe { bld.launch(cfg_w).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_w4 = g.stream.memcpy_dtov(&o).unwrap();
                let w4diff = out_unt
                    .iter()
                    .zip(&out_w4)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    w4diff < 5e-2,
                    "S={s}: wide WMMA flash vs f32 reference disagree, max_abs={w4diff:.2e}"
                );

                pin!();
                let t_unt = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_unt);
                        bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                        unsafe { bld.launch(cfg_unt).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                pin!();
                let t_til = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_til);
                        bld.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                        unsafe { bld.launch(cfg_til).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                pin!();
                let t_w = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_wmma);
                        bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                        unsafe { bld.launch(cfg_w).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                pin!();
                let t_w4 = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_wmma4);
                        bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                        unsafe { bld.launch(cfg_w).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                eprintln!(
                    "S={s:>4}: untiled {:.4} | tiled {:.4} | wmma {:.4} | wmma4 {:.4} ms || tiled/untiled {:.2}× | wmma/tiled {:.2}× | wmma4/wmma {:.2}× | wmma4/tiled {:.2}×",
                    t_unt * 1e3,
                    t_til * 1e3,
                    t_w * 1e3,
                    t_w4 * 1e3,
                    t_til / t_unt,
                    t_w / t_til,
                    t_w4 / t_w,
                    t_w4 / t_til,
                );
            }
        });
    }

    /// **Same-process A/B: register-resident `mma.sync` flash (`flash_d64_m`) vs the SMEM-round-trip
    /// WMMA flash (`flash_d64_w4`) and the f32 tiled flash (`flash_d64_t`)** across sequence length under
    /// one pinned clock — the only honest flash comparison (a cross-*run* one is corrupted by the ~7×
    /// laptop-GPU clock swing). All single-head, D=64, 1 warp/CTA. Asserts `flash_d64_m` agrees with the
    /// f32 reference at every S (far past the gate's sizes), then reports `m/w4` — below 1.0 means the
    /// register-resident form wins. Run: `… --features gpu --release -- --ignored --nocapture flash_mma_vs_wmma`.
    #[test]
    #[ignore = "tuning bench; run explicitly"]
    fn flash_mma_vs_wmma() {
        with_gpu("flash_mma_vs_wmma", |g| {
            let mut rng = crate::diff::Rng::new(0x3E9157);
            let d = 64usize;
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
            let f_til = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_t")
                .unwrap();
            let f_w4 = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_w4")
                .unwrap();
            let f_m = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_m")
                .unwrap();
            let to16 =
                |x: &[f32]| -> Vec<half::f16> { x.iter().map(|&v| half::f16::from_f32(v)).collect() };
            for s in [256usize, 512, 1024, 2048, 4096] {
                let qf = rng.vec(s * d, -1.0, 1.0);
                let kf = rng.vec(s * d, -1.0, 1.0);
                let vf = rng.vec(s * d, -1.0, 1.0);
                let q = g.stream.memcpy_stod(&qf).unwrap();
                let k = g.stream.memcpy_stod(&kf).unwrap();
                let v = g.stream.memcpy_stod(&vf).unwrap();
                let q16 = g.stream.memcpy_stod(&to16(&qf)).unwrap();
                let k16 = g.stream.memcpy_stod(&to16(&kf)).unwrap();
                let v16 = g.stream.memcpy_stod(&to16(&vf)).unwrap();
                let mut o = g.stream.memcpy_stod(&vec![0f32; s * d]).unwrap();
                let scale = 1.0f32 / (d as f32).sqrt();
                let ss = s as u32;
                let (_, cfg_til) = flash_plan_forced(d, s, true);
                let cfg_w = LaunchConfig {
                    grid_dim: ((s / 16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // f32 reference (tiled), then assert the register-resident mma flash agrees (f16-in tol).
                {
                    let mut b = g.stream.launch_builder(&f_til);
                    b.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                    unsafe { b.launch(cfg_til).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_ref = g.stream.memcpy_dtov(&o).unwrap();
                {
                    let mut b = g.stream.launch_builder(&f_m);
                    b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                    unsafe { b.launch(cfg_w).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_m = g.stream.memcpy_dtov(&o).unwrap();
                let mdiff = out_ref
                    .iter()
                    .zip(&out_m)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(mdiff < 5e-2, "S={s}: mma flash vs f32 ref disagree, max_abs={mdiff:.2e}");

                pin!();
                let t_til = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut b = g.stream.launch_builder(&f_til);
                        b.arg(&ss).arg(&scale).arg(&q).arg(&k).arg(&v).arg(&mut o);
                        unsafe { b.launch(cfg_til).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                pin!();
                let t_w4 = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut b = g.stream.launch_builder(&f_w4);
                        b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                        unsafe { b.launch(cfg_w).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                pin!();
                let t_m = best_of(5, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut b = g.stream.launch_builder(&f_m);
                        b.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o);
                        unsafe { b.launch(cfg_w).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                eprintln!(
                    "S={s:>4}: tiled {:.4} | wmma4 {:.4} | mma {:.4} ms || m/w4 {:.2}× | m/tiled {:.2}×",
                    t_til * 1e3,
                    t_w4 * 1e3,
                    t_m * 1e3,
                    t_m / t_w4,
                    t_m / t_til,
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
            // (C, H, W, K, R, S) — sweep the tiled generator: partial tiles (P,Q not a tile multiple),
            // an exact single full tile (P=Q=16), a 1×1 conv (halo == tile), and a larger multi-tile
            // image so the grid spans many CTAs.
            let cases = [
                (3usize, 16usize, 16usize, 8usize, 3usize, 3usize), // P=Q=14 (partial tile)
                (16, 32, 32, 4, 5, 5),                              // P=Q=28 (partial tile)
                (8, 18, 18, 12, 3, 3),                              // P=Q=16 (one full tile)
                (32, 28, 28, 16, 1, 1),                             // 1×1 conv (halo == tile)
                (8, 64, 64, 16, 3, 3),                              // P=Q=62 (16 tiles, multi-CTA)
                (4, 24, 40, 6, 3, 5),                               // non-square image + kernel
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

    #[test]
    fn conv2d_wmma_matches_reference_within_tol() {
        with_gpu("conv2d_wmma", |g| {
            let mut rng = crate::diff::Rng::new(0x3CA1B0);
            // (C,H,W,K,R,S) — exercise M/N/GK that are NOT tile multiples (guards must zero-pad):
            // K not %32, P*Q not %32, C*R*S not %16, plus a 1×1 and a clean shape.
            let cases = [
                (3usize, 32usize, 32usize, 16usize, 3usize, 3usize), // GK=27, N=900 (both non-mult)
                (16, 28, 28, 32, 3, 3),                              // GK=144, N=676
                (8, 16, 16, 48, 5, 5),                               // K=48, GK=200, N=144
                (32, 14, 14, 64, 1, 1),                              // 1×1: GK=32, N=196
                (4, 24, 24, 24, 3, 3),                               // K=24 (not %32), GK=36
            ];
            for (c, h, width, k, r, s) in cases {
                if !crate::ptx_conv::wmma_applies(c, h, width, k, r, s) {
                    continue;
                }
                let x = rng.vec(c * h * width, -1.0, 1.0);
                let w = rng.vec(k * c * r * s, -1.0, 1.0);
                let got = conv2d_wmma(g, &x, &w, c, h, width, k, r, s).unwrap();
                let oracle = ref_conv2d(&x, &w, c, h, width, k, r, s);
                // fp16 inputs: relative error ~ 2^-10 per element, grows with the C·R·S reduction.
                let rel = ((4.0 * ((c * r * s) as f64).sqrt()) * (2f64).powi(-10)).max(2e-2);
                let st = crate::diff::assert_close(
                    &format!("conv2d_wmma C{c} {h}x{width} K{k} {r}x{s}"),
                    &got,
                    &oracle,
                    5e-2,
                    rel,
                );
                eprintln!(
                    "conv2d_wmma C{c} {h}x{width} K{k} {r}x{s}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// **M6 for conv2d** — Mercury's fp16 tensor-core implicit-GEMM conv (and the f32 SMEM-tiled conv)
    /// vs the **naive CUDA-C conv** a programmer writes first (one thread per output, the whole `c,r,s`
    /// window streamed from global), all JIT-loaded through the same driver and timed **same-run** over
    /// identical buffers. Correctness gates speed: the naive peer is cross-checked against the f64
    /// oracle, and both Mercury kernels are checksum-cross-checked against the peer at every shape
    /// before any ratio counts. Needs the CUDA redist DLLs on PATH (see `gemm_vs_peers` /
    /// `peer_env_hint`).
    ///
    /// **cuDNN (Tier-B gold standard) status:** *not bound here.* `cudarc`'s cuDNN module needs the
    /// cuDNN redist (separate from the NVRTC/cuBLAS wheels these benches already dlopen) and a fragile
    /// descriptor-graph setup; rather than fake a peer, the honest headline is the **wide Tier-A win
    /// over naive CUDA-C** (1.3–5.8× same-run across these shapes, ≥3 reruns). Binding cuDNN to report
    /// a % is the next peer-side TODO.
    #[test]
    #[ignore = "throughput bench; needs CUDA NVRTC redist DLLs on PATH; run explicitly"]
    fn conv_vs_peers() {
        use crate::baselines::{
            conv_flop, nvrtc_naive_conv, peer_env_hint, peers_available, time_nvrtc_naive_conv,
        };
        with_gpu("conv_vs_peers", |g| {
            if !peers_available(g) {
                eprintln!("[skip] conv_vs_peers: NVRTC not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let mut rng = crate::diff::Rng::new(0xC04F1E);

            // --- Correctness first: naive CUDA-C conv matches the f64 oracle on small shapes. ---
            for (c, h, wd, k, r, s) in [(3usize, 16usize, 16usize, 8usize, 3usize, 3usize), (16, 24, 24, 8, 5, 5)] {
                let x = rng.vec(c * h * wd, -1.0, 1.0);
                let w = rng.vec(k * c * r * s, -1.0, 1.0);
                let naive = nvrtc_naive_conv(g, &x, &w, c, h, wd, k, r, s).unwrap();
                let oracle = ref_conv2d(&x, &w, c, h, wd, k, r, s);
                let rel = ((8.0 * ((c * r * s) as f64).sqrt()) * f32::EPSILON as f64).max(1e-4);
                crate::diff::assert_close(&format!("naive conv C{c} {r}x{s}"), &naive, &oracle, 1e-4, rel);
            }
            eprintln!("[gate] naive CUDA-C conv matches the f64 oracle ✓");

            // --- Clock warmup (peak-vs-peak, same-run): hammer a GEMM until the mobile clock settles. ---
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            for _ in 0..40 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            // DL-style conv shapes (C, H, W, K, R, S): a ResNet-ish stack of 3x3 layers + a 5x5.
            let cases = [
                (3usize, 64usize, 64usize, 64usize, 3usize, 3usize),
                (64, 56, 56, 64, 3, 3),
                (128, 28, 28, 128, 3, 3),
                (256, 14, 14, 256, 3, 3),
                (32, 32, 32, 32, 5, 5),
            ];
            use half::f16;
            let to16 = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            let csum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
            for (c, h, wd, k, r, s) in cases {
                assert!(crate::ptx_conv::tiled_applies(c, h, wd, k, r, s), "shape not tiled");
                assert!(crate::ptx_conv::wmma_applies(c, h, wd, k, r, s), "shape not wmma");
                let (p, q) = (h - r + 1, wd - s + 1);
                let x = rng.vec(c * h * wd, -1.0, 1.0);
                let w = rng.vec(k * c * r * s, -1.0, 1.0);
                let naive = nvrtc_naive_conv(g, &x, &w, c, h, wd, k, r, s).unwrap();
                let cs_n = csum(&naive);

                // --- Mercury f32 SMEM-tiled conv (resident; module loaded once) ---
                let ptx_t = crate::ptx_conv::conv2d_ptx(c, h, wd, k, r, s);
                let mod_t = g.load_module_cached(&ptx_t).unwrap();
                let f_t = mod_t.load_function("conv2d").unwrap();
                let cfg_t = conv_tiled_cfg(h, wd, k, r, s);
                let xt_d = g.stream.memcpy_stod(&x).unwrap();
                let wt_d = g.stream.memcpy_stod(&w).unwrap();
                let mut ot_d = g.stream.alloc_zeros::<f32>(k * p * q).unwrap();
                let launch_t = |g: &Gpu, o: &mut cudarc::driver::CudaSlice<f32>| {
                    let mut b = g.stream.launch_builder(&f_t);
                    b.arg(&xt_d).arg(&wt_d).arg(o);
                    unsafe { b.launch(cfg_t).unwrap() };
                };
                launch_t(g, &mut ot_d);
                g.stream.synchronize().unwrap();
                let cs_t = csum(&g.stream.memcpy_dtov(&ot_d).unwrap());
                assert!((cs_t - cs_n).abs() / cs_n.max(1.0) < 2e-2, "tiled checksum: t={cs_t:.3e} n={cs_n:.3e}");

                // --- Mercury fp16 tensor-core implicit-GEMM conv (resident; f16 X/W) ---
                let ptx_w = crate::ptx_conv::conv_wmma_ptx(c, h, wd, k, r, s);
                let mod_w = g.load_module_cached(&ptx_w).unwrap();
                let f_w = mod_w.load_function("conv2d_wmma").unwrap();
                let cfg_w = conv_wmma_cfg(h, wd, k, r, s);
                let xw_d = g.stream.memcpy_stod(&to16(&x)).unwrap();
                let ww_d = g.stream.memcpy_stod(&to16(&w)).unwrap();
                let mut ow_d = g.stream.alloc_zeros::<f32>(k * p * q).unwrap();
                let launch_w = |g: &Gpu, o: &mut cudarc::driver::CudaSlice<f32>| {
                    let mut b = g.stream.launch_builder(&f_w);
                    b.arg(&xw_d).arg(&ww_d).arg(o);
                    unsafe { b.launch(cfg_w).unwrap() };
                };
                launch_w(g, &mut ow_d);
                g.stream.synchronize().unwrap();
                let cs_w = csum(&g.stream.memcpy_dtov(&ow_d).unwrap());
                assert!((cs_w - cs_n).abs() / cs_n.max(1.0) < 6e-2, "wmma checksum: w={cs_w:.3e} n={cs_n:.3e}");

                // Speed, same-run.
                let iters = 50usize;
                let t_t = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters { launch_t(g, &mut ot_d); }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let t_w = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..iters { launch_w(g, &mut ow_d); }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / iters as f64
                });
                let naive_iters = if c >= 128 { 10 } else { 30 };
                let t_n = time_nvrtc_naive_conv(g, c, h, wd, k, r, s, naive_iters).unwrap();

                let flop = conv_flop(c, h, wd, k, r, s);
                let (g_t, g_w, g_n) = (flop / t_t, flop / t_w, flop / t_n);
                eprintln!(
                    "C{c:>3} {h}x{wd} K{k:>3} {r}x{s}: tiled {:>6.0} GF ({:>4.1}×) | WMMA {:.4} ms {:>6.0} GF ({:>4.1}×) | naive {:.4} ms {:>5.0} GF",
                    g_t / 1e9,
                    g_t / g_n,
                    t_w * 1e3,
                    g_w / 1e9,
                    g_w / g_n,
                    t_n * 1e3,
                    g_n / 1e9,
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

    /// CPU f64 reference for the **multi-head** fp16 layer — identical to [`ref_transformer_layer_f16`]
    /// but attention runs per-head over the `D` columns split into `heads` blocks of `dh = d/heads`:
    /// each head gathers `q/k/v[:, h·dh .. (h+1)·dh]` into a contiguous `[S,dh]`, runs the single-head
    /// `ref_attn` at `scale = 1/√dh`, and scatters the result back into `a[:, h·dh ..]`. `heads == 1`
    /// reduces exactly to [`ref_transformer_layer_f16`].
    fn ref_transformer_layer_f16_mha(
        x: &[f32],
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
        heads: usize,
    ) -> Vec<f32> {
        let eps = 1e-5f32;
        let round = |v: f32| half::f16::from_f32(v).to_f32();
        let dh = d / heads;
        let h1 = ref_rmsnorm(x, s, d, eps);
        let q = ref_nt_rounded(&h1, w.wq, s, d, d, round);
        let k = ref_nt_rounded(&h1, w.wk, s, d, d, round);
        let v = ref_nt_rounded(&h1, w.wv, s, d, d, round);
        let mut a = vec![0.0f32; s * d];
        let scale = 1.0 / (dh as f32).sqrt();
        for head in 0..heads {
            let (mut qh, mut kh, mut vh) =
                (vec![0.0f32; s * dh], vec![0.0f32; s * dh], vec![0.0f32; s * dh]);
            for row in 0..s {
                for i in 0..dh {
                    qh[row * dh + i] = q[row * d + head * dh + i];
                    kh[row * dh + i] = k[row * d + head * dh + i];
                    vh[row * dh + i] = v[row * d + head * dh + i];
                }
            }
            let ah = ref_attn(&qh, &kh, &vh, s, dh, scale);
            for row in 0..s {
                for i in 0..dh {
                    a[row * d + head * dh + i] = ah[row * dh + i];
                }
            }
        }
        let o = ref_nt_rounded(&a, w.wo, s, d, d, round);
        let x1: Vec<f32> = x.iter().zip(&o).map(|(&p, &q)| p + q).collect();
        let h2 = ref_rmsnorm(&x1, s, d, eps);
        let f1 = ref_nt_rounded(&h2, w.w1, s, d, dff, round);
        let f1act: Vec<f32> = f1.iter().map(|&z| ref_silu(z)).collect();
        let f2 = ref_nt_rounded(&f1act, w.w2, s, dff, d, round);
        x1.iter().zip(&f2).map(|(&p, &q)| p + q).collect()
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

    /// **Multi-head layer gate** — `ResidentLayerF16::new_mha` at the real GPT-2 shape (D=768, **H=12**
    /// heads of dh=64) vs the per-head f64 reference. Exercises the full multi-head path: QKV projections,
    /// the `cast_transpose_qkv` token→head-major shim, the `grid.y=12` tensor-core flash, the
    /// `transpose_attn_out` shim back, and the fused O-proj/FFN. S=512 (the tensor-core flash minimum).
    /// Tolerance is the f16-GEMM band; also re-asserts run-to-run determinism (fixed grids, no atomics).
    #[test]
    fn transformer_layer_mha_matches_reference_within_tol() {
        with_gpu("transformer_layer_mha", |g| {
            let mut rng = crate::diff::Rng::new(0x6457A);
            let (s, d, dff, heads) = (512usize, 768usize, 1024usize, 12usize); // GPT-2: 12 heads × 64
            let x = rng.vec(s * d, -1.0, 1.0);
            let wq = rng.vec(d * d, -0.1, 0.1);
            let wk = rng.vec(d * d, -0.1, 0.1);
            let wv = rng.vec(d * d, -0.1, 0.1);
            let wo = rng.vec(d * d, -0.1, 0.1);
            let w1 = rng.vec(dff * d, -0.1, 0.1);
            let w2 = rng.vec(d * dff, -0.1, 0.1);
            let w = TransformerWeights { wq: &wq, wk: &wk, wv: &wv, wo: &wo, w1: &w1, w2: &w2 };
            let got = ResidentLayerF16::new_mha(g, &w, s, d, dff, heads).unwrap().forward(&x).unwrap();
            let oracle = ref_transformer_layer_f16_mha(&x, &w, s, d, dff, heads);
            let st = crate::diff::assert_close("transformer_layer_mha", &got, &oracle, 3e-2, 3e-2);
            eprintln!(
                "transformer_layer_mha S={s} D={d} H={heads} dh={} Dff={dff}: max_abs={:.2e} max_rel={:.2e}",
                d / heads,
                st.max_abs,
                st.max_rel
            );
            let again = ResidentLayerF16::new_mha(g, &w, s, d, dff, heads).unwrap().forward(&x).unwrap();
            let bits = |v: &[f32]| v.iter().map(|z| z.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&again), "multi-head layer must be deterministic");
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
            let (d, dff) = (64usize, 256usize);
            // S=128 exercises the f32 flash path; S=512 trips `wmma_flash_applies` so the layer runs the
            // tensor-core flash (Q/K/V cast to f16) — both must match the same f64 layer reference.
            for s in [128usize, 512] {
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
                let st =
                    crate::diff::assert_close("transformer_layer_f16", &got, &oracle, 5e-2, 5e-2);
                let flash = if wmma_flash_applies(d, s) { "wmma" } else { "f32" };
                eprintln!(
                    "transformer_layer_f16 S={s} D={d} Dff={dff} [{flash} flash]: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
                let again = transformer_layer_f16(g, &x, &w, s, d, dff).unwrap();
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&got),
                    bits(&again),
                    "fp16 GPU-resident transformer layer must be deterministic"
                );
            }
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

            // Conv2d (fp16 tensor-core implicit GEMM) — fixed grid, no atomics ⇒ bit-reproducible.
            let (c2, h2, w2, k2, r2, s2) = (16usize, 16usize, 16usize, 32usize, 3usize, 3usize);
            let xw = rng.vec(c2 * h2 * w2, -1.0, 1.0);
            let ww = rng.vec(k2 * c2 * r2 * s2, -1.0, 1.0);
            twice_eq!("conv2d_wmma", conv2d_wmma(g, &xw, &ww, c2, h2, w2, k2, r2, s2).unwrap());

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
                let (fname, flash_cfg) = flash_plan(d, s);
                let f_flash = g
                    .function("flash", crate::ptx_flash::flash_ptx(), &fname)
                    .unwrap();
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
                    "S={s} (resident, pure kernel): flash [{fname}] {:.4} ms | WMMA gemm S×{d}×{dff} {:.4} ms | flash/gemm {:.1}×",
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

            // 2048/4096 are the long-context regime: single-head attention is O(S²·D) and dominates
            // the O(S·D·Dff) FFN (~8× the flops at S=4096), so the per-layer ratio here is governed by
            // the flash kernel — the regime the WMMA flash was built for. Attention is COMMON to both
            // stacks (same kernel), so the cuBLAS edge narrows toward 1.0 as S grows; the real
            // long-context test is the torch harness (separate flash kernels). These absolute ms/layer
            // feed MERCURY_MS_PER_LAYER in bench/pytorch/transformer_layer_peer.py.
            for s in [256usize, 512, 1024, 2048, 4096] {
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

    /// **M13 at the real GPT-2 shape: multi-head, D=768, H=12, Dff=3072.** The single-head
    /// `cublas_chain_vs_mercury_layer_throughput` runs a D=64 toy; this is a genuine transformer layer.
    /// Mercury's fused multi-head [`ResidentLayerF16`] vs the **multi-head** cuBLAS-chain layer
    /// ([`CublasChainLayer::new_mha`]) — attention is *common* to both (the same `grid.y=H` flash +
    /// transpose shims), so the same-run gap is purely (cuBLAS-vs-WMMA GEMM) + (the residual/SiLU epilogue
    /// fusion cuBLAS can't do). Correctness gates speed: both match the per-head f64 reference first.
    /// Same-run only (clock pinned by a GEMM hammer + `best_of`). Needs the cuBLAS redist DLL on PATH;
    /// skips if absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture cublas_chain_vs_mercury_mha`.
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn cublas_chain_vs_mercury_mha_layer_throughput() {
        use crate::baselines::{peer_env_hint, peers_available, CublasChainLayer};
        with_gpu("cublas_chain_vs_mercury_mha", |g| {
            if !peers_available(g) {
                eprintln!(
                    "[skip] cublas_chain_vs_mercury_mha_layer_throughput: cuBLAS not loadable.\n{}",
                    peer_env_hint()
                );
                return;
            }
            let mut rng = crate::diff::Rng::new(0x60C2);
            let (d, dff, heads) = (768usize, 3072usize, 12usize); // GPT-2 base: 12 heads × 64, 4× FFN
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

            for s in [512usize, 1024, 2048, 4096] {
                let x = rng.vec(s * d, -1.0, 1.0);
                let wq = rng.vec(d * d, -0.1, 0.1);
                let wk = rng.vec(d * d, -0.1, 0.1);
                let wv = rng.vec(d * d, -0.1, 0.1);
                let wo = rng.vec(d * d, -0.1, 0.1);
                let w1 = rng.vec(dff * d, -0.1, 0.1);
                let w2 = rng.vec(d * dff, -0.1, 0.1);
                let w = TransformerWeights { wq: &wq, wk: &wk, wv: &wv, wo: &wo, w1: &w1, w2: &w2 };
                let mer = ResidentLayerF16::new_mha(g, &w, s, d, dff, heads).unwrap();
                let chain = CublasChainLayer::new_mha(g, &w, s, d, dff, heads).unwrap();
                let x_d = g.stream.memcpy_stod(&x).unwrap();

                // Correctness before speed: both stacks vs the per-head f64 oracle at the real shape.
                let oracle = ref_transformer_layer_f16_mha(&x, &w, s, d, dff, heads);
                let mer_out = g.stream.memcpy_dtov(&mer.forward_device(&x_d).unwrap()).unwrap();
                let chain_out = g.stream.memcpy_dtov(&chain.forward_device(&x_d).unwrap()).unwrap();
                crate::diff::assert_close(&format!("Mercury fused MHA S={s}"), &mer_out, &oracle, 5e-2, 5e-2);
                crate::diff::assert_close(&format!("cuBLAS chain MHA S={s}"), &chain_out, &oracle, 5e-2, 5e-2);

                let iters = if s >= 2048 { 20 } else { 50 };
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

                let (hdl_lbl, hdl) = if t_chain >= t_fused {
                    ("Mercury fused faster", t_chain / t_fused)
                } else {
                    ("cuBLAS chain faster", t_fused / t_chain)
                };
                eprintln!(
                    "M13 GPT-2 layer S={s} D={d} H={heads} Dff={dff} (resident, same-run): \
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

    /// The **pipelined fp8 kernel** (`fp8_gemm_pipe`, dispatched by `gemm_nt_fp8` for %128/%128/%64
    /// shapes) must match the E4M3-rounded f64 reference — same `m16n8k32` layout as the proven single/mt
    /// fp8 kernels, only now staged through a padded conflict-free SMEM ring with rasterization. Shapes
    /// exercise the %128/%64 divisibility, a single K-tile (BK=64 ⇒ prologue guard), the ring wrap
    /// (K > stages·BK), and a rectangular multi-CTA case.
    #[test]
    fn fp8_pipe_matches_reference_within_tol() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_pipe", |g| {
            let mut rng = crate::diff::Rng::new(0xF8B1);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            for (m, k, n) in [(128usize, 64usize, 128usize), (128, 128, 128), (256, 256, 256), (128, 192, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = gemm_nt_fp8_pipe(g, &a, &b, m, k, n).unwrap();
                let r = ref_nt_rounded(&a, &b, m, k, n, round);
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                let st = crate::diff::assert_close(&format!("fp8_pipe {m}x{k}x{n}"), &c, &r, 1e-2, rel);
                eprintln!("fp8_pipe {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", st.max_abs, st.max_rel);
            }
        });
    }

    /// Gate for the **regime-aware fp8 pipe dispatch** (M2 lever): `gemm_nt_fp8_pipe` now routes M≤2048
    /// (and any M where 128∤M but 64∣M) to the 64×128 `fp8_gemm_pipe_m64` entry, and larger M to the
    /// 128×128 entry — the higher-occupancy small-tile won the cuBLASLt-fp8 sweep at M≤2048. Both entries
    /// must match the same E4M3-rounded f64 reference; the shapes straddle the M=2048 threshold and
    /// include a 128∤M case, so each entry (and the dispatch boundary) is exercised. Same codegen, only
    /// BM differs ⇒ bit-identical accumulation ⇒ the same fp8 `c·√K·ε` tolerance.
    #[test]
    fn fp8_pipe_regime_matches_reference() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_pipe_regime", |g| {
            let mut rng = crate::diff::Rng::new(0xF8D3);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            // (m, k, n): three M≤2048 (incl. 128∤M=192) → m64 entry; two M>2048 → 128×128 entry.
            for (m, k, n) in [
                (64usize, 128usize, 128usize),
                (192, 128, 256),
                (1024, 128, 256),
                (2304, 128, 256),
                (2560, 64, 128),
            ] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = gemm_nt_fp8_pipe(g, &a, &b, m, k, n).unwrap();
                let r = ref_nt_rounded(&a, &b, m, k, n, round);
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                let bm = if m % 128 != 0 || m <= 2048 { 64 } else { 128 };
                let st = crate::diff::assert_close(
                    &format!("fp8_pipe_regime {m}x{k}x{n} bm={bm}"),
                    &c,
                    &r,
                    1e-2,
                    rel,
                );
                eprintln!("fp8_pipe_regime {m}x{k}x{n} (bm={bm}): max_abs={:.2e}", st.max_abs);
            }
        });
    }

    /// The fused `C = act(A·Bᵀ + bias)` epilogues on the **fp8 mma workhorse**
    /// (`gemm_nt_fp8_mma_bias{,_relu,_silu,_gelu}`) — the fastest fused inference path (Ada runs fp8
    /// `mma.sync` at 2× the fp16 TC rate). The register-level bias epilogue acts on the f32 accumulator
    /// (the m16n8k32 D-fragment column map matches m16n8k16), so each output must equal
    /// `act(e4m3-rounded(A·Bᵀ) + bias)`; fp8's coarse E4M3 quantization sets the tolerance (same as the
    /// plain fp8 pipe gate). Workhorse shape constraints: M%128, N%128, K%64.
    #[test]
    fn fp8_mma_bias_match_reference_within_tol() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_mma_bias", |g| {
            let mut rng = crate::diff::Rng::new(0xF8B2);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 192, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                // Reference: (e4m3-rounded GEMM) + bias[col], then the activation — per the kernel order.
                let base = ref_nt_rounded(&a, &b, m, k, n, round);
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
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                for (name, got, refv) in [
                    (
                        "bias",
                        gemm_nt_fp8_mma_bias(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&id),
                    ),
                    (
                        "bias_relu",
                        gemm_nt_fp8_mma_bias_relu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&|x| x.max(0.0)),
                    ),
                    (
                        "bias_silu",
                        gemm_nt_fp8_mma_bias_silu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&silu),
                    ),
                    (
                        "bias_gelu",
                        gemm_nt_fp8_mma_bias_gelu(g, &a, &b, &bias, m, k, n).unwrap(),
                        with_bias(&gelu),
                    ),
                ] {
                    let s = crate::diff::assert_close(
                        &format!("fp8_mma_{name} {m}x{k}x{n}"),
                        &got,
                        &refv,
                        1e-2,
                        rel,
                    );
                    eprintln!(
                        "fp8_mma_{name} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                        s.max_abs, s.max_rel
                    );
                }
            }
        });
    }

    /// The fused `out = A·Bᵀ + bias + residual` epilogue on the fp8 workhorse
    /// (`gemm_nt_fp8_mma_bias_residual`) — the fastest down-proj / output-proj (Ada 2× fp8 rate, residual
    /// kept f32). Each output must equal `(e4m3-rounded(A·Bᵀ) + bias[col]) + residual[i]`. Gates the
    /// per-column bias map AND the per-element residual addressing under the fp8 m16n8k32 D-fragment
    /// layout. Workhorse shape constraints: M%128, N%128, K%64.
    #[test]
    fn fp8_mma_bias_residual_match_reference_within_tol() {
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("fp8_mma_bias_residual", |g| {
            let mut rng = crate::diff::Rng::new(0xF8B3);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256), (128, 192, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                let resid = rng.vec(m * n, -1.0, 1.0);
                let mut refv = ref_nt_rounded(&a, &b, m, k, n, round);
                for i in 0..m {
                    for j in 0..n {
                        refv[i * n + j] += bias[j] + resid[i * n + j];
                    }
                }
                let got = gemm_nt_fp8_mma_bias_residual(g, &a, &b, &bias, &resid, m, k, n).unwrap();
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                let s = crate::diff::assert_close(
                    &format!("fp8_mma_bias_residual {m}x{k}x{n}"),
                    &got,
                    &refv,
                    1e-2,
                    rel,
                );
                eprintln!(
                    "fp8_mma_bias_residual {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    s.max_abs, s.max_rel
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

    /// **M2: the pipelined fp8 (E4M3) GEMM** — the cliff fix carried to fp8 — measured same-run against
    /// (a) the old un-staged `fp8_gemm_nt_mt` (the speedup the cp.async pipeline + padded conflict-free
    /// SMEM + raster buys) and (b) the dispatched **fp16** mma kernel (the **Ada 2× fp8-rate** check — fp8
    /// tensor cores run ~2× the fp16 rate, so fp8 GFLOP/s should be ~2× fp16's at the same shape). Same
    /// buffers' worth of work, `best_of` peak-clock sampling, with a checksum cross-check (fp8-pipe must
    /// equal fp8-mt). No cuBLASLt fp8 peer (cudarc's safe `Matmul` is f32/f16/bf16 only; a raw-sys E4M3
    /// peer with scale descriptors is the follow-up for the literal %-of-cuBLASLt number).
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn fp8_pipe_vs_peers() {
        use crate::ptx_fp8::{f32_to_e4m3, FP8_PIPE_BM, FP8_PIPE_BN, FP8_PIPE_THREADS, FP8_TM, FP8_TN};
        use half::f16;
        with_gpu("fp8_pipe_vs_peers", |g| {
            let mut rng = crate::diff::Rng::new(0xF8FE);
            // Clock warmup (peak-vs-peak; absolutes swing ~7× with boost on this part).
            let wa = rng.vec(2048 * 2048, -1.0, 1.0);
            let wb = rng.vec(2048 * 2048, -1.0, 1.0);
            for _ in 0..15 {
                let _ = gemm_nt_fp8_pipe(g, &wa, &wb, 2048, 2048, 2048).unwrap();
            }
            let csum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
            for sz in [2048usize, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = 2.0 * m as f64 * k as f64 * n as f64;
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
                let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
                let a8_d = g.stream.memcpy_stod(&a8).unwrap();
                let b8_d = g.stream.memcpy_stod(&b8).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();

                // fp8 pipelined (new) — 1-D rasterized grid.
                let f_pipe = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), "fp8_gemm_pipe").unwrap();
                let cfg_pipe = LaunchConfig {
                    grid_dim: (((m / FP8_PIPE_BM) * (n / FP8_PIPE_BN)) as u32, 1, 1),
                    block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_pipe = best_of(4, || time_wmma(g, &f_pipe, cfg_pipe, dims, &a8_d, &b8_d, &mut c_d, 50));
                let cs_pipe = csum(&g.stream.memcpy_dtov(&c_d).unwrap());

                // fp8 fragment-reuse multi-tile (old, un-staged global loads).
                let f_mt = g.function("fp8_gemm_mt", crate::ptx_fp8::fp8_gemm_mt_ptx(), "fp8_gemm_nt_mt").unwrap();
                let cfg_mt = LaunchConfig {
                    grid_dim: ((n / (8 * FP8_TN)) as u32, (m / (16 * FP8_TM)) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_mt = best_of(4, || time_wmma(g, &f_mt, cfg_mt, dims, &a8_d, &b8_d, &mut c_d, 50));
                let cs_mt = csum(&g.stream.memcpy_dtov(&c_d).unwrap());
                assert!(
                    (cs_pipe - cs_mt).abs() / cs_mt.max(1.0) < 3e-2,
                    "{sz}³ fp8 pipe/mt checksum mismatch: {cs_pipe:.3e} vs {cs_mt:.3e}"
                );

                // fp16 dispatched mma kernel (Ada 2× rate reference) — same matrix.
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a16_d = g.stream.memcpy_stod(&a16).unwrap();
                let b16_d = g.stream.memcpy_stod(&b16).unwrap();
                let v = crate::ptx_wmma::pipe_variant("mma_nt_f16_128_bk32_s2_r16");
                let f16f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), v.name).unwrap();
                let s_f16 = best_of(4, || time_wmma(g, &f16f, pipe_cfg(v, m, n), dims, &a16_d, &b16_d, &mut c_d, 50));

                eprintln!(
                    "{sz}³ fp8 (same-run): pipe {:>7.0} GFLOP/s | {:>4.2}× vs fp8-mt ({:.0}) | {:>4.2}× vs fp16 (Ada 2× rate: fp16 {:.0})",
                    flop / s_pipe / 1e9,
                    s_mt / s_pipe,
                    flop / s_mt / 1e9,
                    s_f16 / s_pipe,
                    flop / s_f16 / 1e9,
                );
            }
        });
    }

    /// **Correctness gate for the cuBLASLt fp8 peer** (M2) — gate first, measure second. Before any
    /// speed number, the peer must agree with the *same* E4M3-rounded f64 reference Mercury's own fp8
    /// kernels are gated against ([`fp8_pipe_matches_reference_within_tol`]). This confirms the
    /// column-major transpose mapping (`Cᵀ = B̌ᵀ·Ǎ`, the fp8 "TN" form) and the E4M3/f32 dtype wiring
    /// are right; tolerance is the fp8 `c·√K·ε` accumulation bound. Skips (never fails) when the redist
    /// DLLs aren't on PATH, or when cuBLASLt reports no fp8 algo for the shape on this device.
    #[test]
    fn cublaslt_fp8_matches_reference_within_tol() {
        use crate::baselines::{
            cublaslt_available, cublaslt_gemm_nt_fp8_e4m3, peer_env_hint, peers_available,
        };
        use crate::ptx_fp8::{e4m3_to_f32, f32_to_e4m3};
        with_gpu("cublaslt_fp8_gate", |g| {
            if !peers_available(g) || !cublaslt_available() {
                eprintln!("[skip] cublaslt_fp8_gate: cuBLASLt not loadable.\n{}", peer_env_hint());
                return;
            }
            let mut rng = crate::diff::Rng::new(0xF8C7);
            let round = |x: f32| e4m3_to_f32(f32_to_e4m3(x));
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 256, 256), (128, 512, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let c = match cublaslt_gemm_nt_fp8_e4m3(g, &a, &b, m, k, n) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("[skip] cuBLASLt fp8 unsupported for {m}x{k}x{n} on this device: {e}");
                        return;
                    }
                };
                let r = ref_nt_rounded(&a, &b, m, k, n, round);
                let rel = ((8.0 * (k as f64).sqrt()) * f32::EPSILON as f64).max(2e-3);
                let st = crate::diff::assert_close(
                    &format!("cublaslt_fp8 {m}x{k}x{n}"),
                    &c,
                    &r,
                    1e-2,
                    rel,
                );
                eprintln!(
                    "cublaslt_fp8 {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// Mercury's fp8 (E4M3) GEMM as a **% of cuBLASLt fp8** — the real M2 peer (cuBLASLt is the *only*
    /// cuBLAS surface with an fp8 matmul). Same-run, same E4M3 bytes fed to both, so the % is
    /// clock-invariant (absolutes swing ~7× with Ada's boost). Reports both Mercury kernels: the
    /// pipelined `fp8_gemm_pipe` and the fragment-reuse `_mt`. Needs the redist DLLs on PATH.
    #[test]
    #[ignore = "throughput bench; needs cublasLt64_12.dll on PATH; run explicitly"]
    fn fp8_vs_cublaslt_pct() {
        use crate::baselines::{
            cublaslt_available, peer_env_hint, peers_available, time_cublaslt_gemm_nt_fp8_e4m3,
        };
        use crate::ptx_fp8::{
            f32_to_e4m3, FP8_PIPE_BM, FP8_PIPE_BN, FP8_PIPE_M64_BM, FP8_PIPE_THREADS, FP8_TM, FP8_TN,
        };
        with_gpu("fp8_vs_cublaslt", |g| {
            if !peers_available(g) || !cublaslt_available() {
                eprintln!("[skip] fp8_vs_cublaslt: cuBLASLt not loadable.\n{}", peer_env_hint());
                return;
            }
            let mut rng = crate::diff::Rng::new(0xF8C8);
            // Clock warm-up (peak-vs-peak; absolutes swing ~7× with boost).
            let wa = rng.vec(2048 * 2048, -1.0, 1.0);
            let wb = rng.vec(2048 * 2048, -1.0, 1.0);
            for _ in 0..15 {
                let _ = gemm_nt_fp8_pipe(g, &wa, &wb, 2048, 2048, 2048).unwrap();
            }
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = 2.0 * m as f64 * k as f64 * n as f64;
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
                let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
                let a8_d = g.stream.memcpy_stod(&a8).unwrap();
                let b8_d = g.stream.memcpy_stod(&b8).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();

                // cuBLASLt fp8 peer (plan built once inside, then 50 resident matmuls).
                let s_lt = match time_cublaslt_gemm_nt_fp8_e4m3(g, &a8_d, &b8_d, &mut c_d, m, k, n, 50) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("{sz}³ [skip] cuBLASLt fp8 unsupported: {e}");
                        continue;
                    }
                };

                // Mercury fp8 pipelined — the *dispatched* tile (`gemm_nt_fp8_pipe`'s regime rule:
                // 64×128 for M≤2048, else 128×128), so the reported % reflects what ships.
                let (pipe_entry, pipe_bm) = if m <= 2048 {
                    ("fp8_gemm_pipe_m64", FP8_PIPE_M64_BM)
                } else {
                    ("fp8_gemm_pipe", FP8_PIPE_BM)
                };
                let f_pipe = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), pipe_entry).unwrap();
                let cfg_pipe = LaunchConfig {
                    grid_dim: (((m / pipe_bm) * (n / FP8_PIPE_BN)) as u32, 1, 1),
                    block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_pipe = best_of(4, || time_wmma(g, &f_pipe, cfg_pipe, dims, &a8_d, &b8_d, &mut c_d, 50));

                // Mercury fp8 fragment-reuse (_mt).
                let f_mt = g.function("fp8_gemm_mt", crate::ptx_fp8::fp8_gemm_mt_ptx(), "fp8_gemm_nt_mt").unwrap();
                let cfg_mt = LaunchConfig {
                    grid_dim: ((n / (8 * FP8_TN)) as u32, (m / (16 * FP8_TM)) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_mt = best_of(4, || time_wmma(g, &f_mt, cfg_mt, dims, &a8_d, &b8_d, &mut c_d, 50));

                eprintln!(
                    "{sz}³ fp8 vs cuBLASLt (same-run): cuBLASLt {:>7.0} GFLOP/s | pipe {:>7.0} = {:>5.1}% of LT | mt {:>7.0} = {:>5.1}% of LT",
                    flop / s_lt / 1e9,
                    flop / s_pipe / 1e9,
                    100.0 * s_lt / s_pipe,
                    flop / s_mt / 1e9,
                    100.0 * s_lt / s_mt,
                );
            }
        });
    }

    /// **fp8 GEMM config sweep vs cuBLASLt** — the M2 "pull levers toward ≥90%" search. Times several
    /// tile/pipeline configs of the same `fp8_gemm_pipe` kernel (the shipped default plus a
    /// deeper-pipeline, a smaller-tile/higher-occupancy, a tighter-raster, and a wider-warp variant)
    /// against the cuBLASLt fp8 peer, same-run, and prints each as a % of cuBLASLt. The per-size winner
    /// says whether a regime-aware dispatch can close the gap, or whether (as for the fp16 cliff) the
    /// default is already at the driver-JIT PTX ceiling. Needs the redist DLLs on PATH.
    #[test]
    #[ignore = "throughput sweep; needs cublasLt64_12.dll on PATH; run explicitly"]
    fn fp8_pipe_config_sweep_vs_cublaslt() {
        use crate::baselines::{
            cublaslt_available, peer_env_hint, peers_available, time_cublaslt_gemm_nt_fp8_e4m3,
        };
        use crate::ptx_fp8::{f32_to_e4m3, fp8_pipe_cfg_ptx};
        with_gpu("fp8_pipe_sweep", |g| {
            if !peers_available(g) || !cublaslt_available() {
                eprintln!("[skip] fp8_pipe_config_sweep: cuBLASLt not loadable.\n{}", peer_env_hint());
                return;
            }
            // (key, bm, bn, bk, warps_m, warps_n, stages, raster) — each ≤48 KiB SMEM.
            let configs: [(&'static str, usize, usize, usize, usize, usize, usize, usize); 5] = [
                ("fp8sw_def", 128, 128, 64, 2, 4, 2, 16),   // shipped default (40 KiB)
                ("fp8sw_s3b32", 128, 128, 32, 2, 4, 3, 16), // deeper pipeline, shorter k-step (36 KiB)
                ("fp8sw_m64", 64, 128, 64, 2, 4, 2, 16),    // smaller tile → more CTAs (30 KiB)
                ("fp8sw_r8", 128, 128, 64, 2, 4, 2, 8),     // tighter rasterization
                ("fp8sw_w44", 128, 128, 64, 4, 4, 2, 16),   // 16 warps/CTA → more ILP
            ];
            let mut rng = crate::diff::Rng::new(0xF8C9);
            // Clock warm-up (peak-vs-peak).
            let wa = rng.vec(2048 * 2048, -1.0, 1.0);
            let wb = rng.vec(2048 * 2048, -1.0, 1.0);
            for _ in 0..15 {
                let _ = gemm_nt_fp8_pipe(g, &wa, &wb, 2048, 2048, 2048).unwrap();
            }
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = 2.0 * m as f64 * k as f64 * n as f64;
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
                let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
                let a8_d = g.stream.memcpy_stod(&a8).unwrap();
                let b8_d = g.stream.memcpy_stod(&b8).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let s_lt = match time_cublaslt_gemm_nt_fp8_e4m3(g, &a8_d, &b8_d, &mut c_d, m, k, n, 50) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("{sz}³ [skip] cuBLASLt fp8: {e}");
                        continue;
                    }
                };
                eprint!("{sz}³ cuBLASLt {:>6.0} GFLOP/s |", flop / s_lt / 1e9);
                for (key, bm, bn, bk, wm, wn, stg, ras) in configs {
                    if m % bm != 0 || n % bn != 0 || k % bk != 0 {
                        eprint!(" {key}:n/a");
                        continue;
                    }
                    let ptx = fp8_pipe_cfg_ptx(bm, bn, bk, wm, wn, stg, ras);
                    let f = g.function(key, &ptx, "fp8_gemm_pipe").unwrap();
                    let cfg = LaunchConfig {
                        grid_dim: (((m / bm) * (n / bn)) as u32, 1, 1),
                        block_dim: ((wm * wn * 32) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let s = best_of(4, || time_wmma(g, &f, cfg, dims, &a8_d, &b8_d, &mut c_d, 50));
                    eprint!(" {key} {:>5.1}%", 100.0 * s_lt / s);
                }
                eprintln!();
            }
        });
    }

    /// **Internal same-family A/B: fp8 64×128 (`_m64`) vs the 128×128 default** — the contention-robust
    /// metric (no cuBLASLt). The %-of-cuBLASLt swings with the shared clock (the baseline is sampled at a
    /// different instant than Mercury), but timing the *two Mercury kernels back-to-back, interleaved
    /// round-by-round*, cancels the clock entirely — the same A/B discipline the int8 swz comparison
    /// uses. Confirms the dispatch lever: `_m64` should be ≥1× the default at M≤2048 (its higher
    /// occupancy) and ~1× at 4096³ (where the default ties). Speedup >1 ⇒ m64 faster.
    #[test]
    #[ignore = "throughput A/B; run explicitly (GPU; no DLLs needed)"]
    fn fp8_pipe_m64_vs_default_ab() {
        use crate::ptx_fp8::{f32_to_e4m3, FP8_PIPE_BM, FP8_PIPE_BN, FP8_PIPE_M64_BM, FP8_PIPE_THREADS};
        with_gpu("fp8_m64_ab", |g| {
            let mut rng = crate::diff::Rng::new(0xF8DA);
            let wa = rng.vec(2048 * 2048, -1.0, 1.0);
            let wb = rng.vec(2048 * 2048, -1.0, 1.0);
            for _ in 0..15 {
                let _ = gemm_nt_fp8_pipe(g, &wa, &wb, 2048, 2048, 2048).unwrap();
            }
            for sz in [512usize, 1024, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = 2.0 * m as f64 * k as f64 * n as f64;
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a8: Vec<u8> = a.iter().map(|&x| f32_to_e4m3(x)).collect();
                let b8: Vec<u8> = b.iter().map(|&x| f32_to_e4m3(x)).collect();
                let a8_d = g.stream.memcpy_stod(&a8).unwrap();
                let b8_d = g.stream.memcpy_stod(&b8).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let f_def = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), "fp8_gemm_pipe").unwrap();
                let f_m64 = g.function("fp8_pipe", crate::ptx_fp8::fp8_pipe_ptx(), "fp8_gemm_pipe_m64").unwrap();
                let cfg_def = LaunchConfig {
                    grid_dim: (((m / FP8_PIPE_BM) * (n / FP8_PIPE_BN)) as u32, 1, 1),
                    block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let cfg_m64 = LaunchConfig {
                    grid_dim: (((m / FP8_PIPE_M64_BM) * (n / FP8_PIPE_BN)) as u32, 1, 1),
                    block_dim: (FP8_PIPE_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // Interleaved best-of so default and m64 sample the same clock state.
                let (mut bd, mut bm) = (f64::MAX, f64::MAX);
                for _ in 0..6 {
                    bd = bd.min(time_wmma(g, &f_def, cfg_def, dims, &a8_d, &b8_d, &mut c_d, 50));
                    bm = bm.min(time_wmma(g, &f_m64, cfg_m64, dims, &a8_d, &b8_d, &mut c_d, 50));
                }
                eprintln!(
                    "{sz}³ fp8 m64-vs-default (same-family A/B): default {:>6.0} GFLOP/s | m64 {:>6.0} = {:.3}× speedup",
                    flop / bd / 1e9,
                    flop / bm / 1e9,
                    bd / bm,
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

    /// **Gate for the multi-head layout shims** (`ptx::HEAD_TRANSPOSE_PTX`): `cast_transpose_qkv`
    /// (f32 `[S,H·dh]` → f16 `[H,S,dh]`, folding the narrowing in) and `transpose_attn_out` (f32
    /// `[H,S,dh]` → f32 `[S,H·dh]`) must match a CPU reference **bit-for-bit** — the forward with
    /// `f16::from_f32` rounding, the inverse exactly. Covers `H=1` (degenerates to cast/copy) and the
    /// GPT-2 `H=12,dh=64` shape. These shims are what let `ResidentLayerF16` run true multi-head attention
    /// on the head-major flash kernel without touching the kernel; a layout bug would scatter the heads.
    #[test]
    fn head_transpose_round_trips() {
        use half::f16;
        with_gpu("head_transpose_round_trips", |g| {
            let mut rng = crate::diff::Rng::new(0x7AB1E5);
            for &(s, heads, dh) in &[(16usize, 1usize, 64usize), (64, 12, 64), (32, 4, 32), (16, 2, 128)] {
                let d = heads * dh;
                let (n, dd, dhh, sdh) = ((s * d) as u32, d as u32, dh as u32, (s * dh) as u32);

                // --- forward: token-major [S,H·dh] f32 -> head-major [H,S,dh] f16 (with f16 rounding) ---
                let src = rng.vec(s * d, -1.0, 1.0);
                let mut want_fwd = vec![f16::from_f32(0.0); s * d];
                for row in 0..s {
                    for head in 0..heads {
                        for i in 0..dh {
                            want_fwd[head * s * dh + row * dh + i] = f16::from_f32(src[row * d + head * dh + i]);
                        }
                    }
                }
                let f_fwd = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "cast_transpose_qkv").unwrap();
                let src_d = g.stream.memcpy_stod(&src).unwrap();
                let mut dst_d = g.stream.alloc_zeros::<f16>(s * d).unwrap();
                let mut b = g.stream.launch_builder(&f_fwd);
                b.arg(&n).arg(&dd).arg(&dhh).arg(&sdh).arg(&src_d).arg(&mut dst_d);
                unsafe { b.launch(LaunchConfig::for_num_elems(n)).unwrap() };
                g.stream.synchronize().unwrap();
                let got_fwd = g.stream.memcpy_dtov(&dst_d).unwrap();
                for (idx, (a, e)) in got_fwd.iter().zip(want_fwd.iter()).enumerate() {
                    assert_eq!(a.to_bits(), e.to_bits(), "fwd mismatch s={s} h={heads} dh={dh} idx={idx}");
                }

                // --- inverse: head-major [H,S,dh] f32 -> token-major [S,H·dh] f32 (exact) ---
                let hsd = rng.vec(s * d, -1.0, 1.0);
                let f_inv = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "transpose_attn_out").unwrap();
                let hsd_d = g.stream.memcpy_stod(&hsd).unwrap();
                let mut out_d = g.stream.alloc_zeros::<f32>(s * d).unwrap();
                let mut b2 = g.stream.launch_builder(&f_inv);
                b2.arg(&n).arg(&dd).arg(&dhh).arg(&sdh).arg(&hsd_d).arg(&mut out_d);
                unsafe { b2.launch(LaunchConfig::for_num_elems(n)).unwrap() };
                g.stream.synchronize().unwrap();
                let got_inv = g.stream.memcpy_dtov(&out_d).unwrap();
                for head in 0..heads {
                    for row in 0..s {
                        for i in 0..dh {
                            assert_eq!(
                                got_inv[row * d + head * dh + i],
                                hsd[head * s * dh + row * dh + i],
                                "inv mismatch s={s} h={heads} dh={dh}"
                            );
                        }
                    }
                }
            }
        });
    }

    /// **Capability probe (not a perf or correctness gate): can NVRTC compile `nvcuda::wmma` here?**
    /// Decides whether a *genuinely FA2-class* (tensor-core, fused) attention peer can be written in
    /// CUDA-C and compiled by the redist NVRTC — vs. falling back to a cuBLAS unfused attention chain.
    /// This toolkit-free box has no `mma.h` on disk, so this asks empirically whether NVRTC bundles it
    /// (it is known to bundle `cuda_fp16.h`). Informational: prints PASS/FAIL + the compile error;
    /// never fails the suite (skips without NVRTC). The result picks which Tier-B M5 peer to build.
    /// Run: `cargo test -p mercury_codegen_gpu --features gpu -- --ignored --nocapture nvrtc_wmma_probe`.
    #[test]
    #[ignore = "capability probe; needs CUDA redist DLLs on PATH; run explicitly"]
    fn nvrtc_wmma_probe() {
        use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
        with_gpu("nvrtc_wmma_probe", |g| {
            if !crate::baselines::peers_available(g) {
                eprintln!("[skip] nvrtc_wmma_probe: NVRTC not loadable.");
                return;
            }
            let opts = || CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            };
            // (1) Does NVRTC bundle cuda_fp16.h? (expected yes — sanity for the header mechanism.)
            let fp16_src = "#include <cuda_fp16.h>\nextern \"C\" __global__ void p(__half* x){ x[0] = __float2half(1.0f); }";
            match compile_ptx_with_opts(fp16_src, opts()) {
                Ok(_) => eprintln!("[probe] cuda_fp16.h: COMPILES OK"),
                Err(e) => eprintln!("[probe] cuda_fp16.h: FAILS -- {e}"),
            }
            // (2) The decisive one: nvcuda::wmma via mma.h. If this compiles, a real fused tensor-core
            //     flash peer (the FA2-class M5 denominator) is feasible with zero toolkit install.
            let wmma_src = r#"
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void wmma_probe(const __half* a, const __half* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(c, acc, 16, wmma::mem_row_major);
}
"#;
            match compile_ptx_with_opts(wmma_src, opts()) {
                Ok(_) => eprintln!(
                    "[probe] nvcuda::wmma (mma.h): COMPILES OK -- FA2-class fused CUDA-C peer is FEASIBLE."
                ),
                Err(e) => eprintln!(
                    "[probe] nvcuda::wmma (mma.h): FAILS -- fall back to the cuBLAS unfused chain peer.\n  {e}"
                ),
            }
        });
    }

    /// **Multi-stage `cp.async` pipeline sweep** — the large-GEMM-cliff lever, measured same-run against
    /// cuBLAS at 1024³/2048³/4096³. Every [`crate::ptx_wmma::PIPE_VARIANTS`] entry (depth × staged-BK ×
    /// macro-tile) whose dims divide the size is timed back-to-back with cuBLAS using the identical clock
    /// warmup + `best_of` peak-clock sampling as [`gemm_vs_peers`], so each %-of-cuBLAS is clock-invariant.
    /// This is the diagnostic that picks the per-size winner the `gemm_nt_f16` dispatch then routes to.
    ///
    /// Correctness gates speed (first law): each variant is cross-checked against the f64 oracle at a small
    /// shape, and at every timing size all variants' checksums must agree with cuBLAS. Needs the cuBLAS
    /// redist DLL on PATH; skips (never fails) if absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture gemm_pipe_sweep`.
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn gemm_pipe_sweep() {
        use crate::baselines::{
            cublas_gemm_nt_f16, gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_f16,
        };
        use crate::ptx_wmma::PIPE_VARIANTS;
        use half::f16;
        with_gpu("gemm_pipe_sweep", |g| {
            if !peers_available(g) {
                eprintln!("[skip] gemm_pipe_sweep: cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());

            // --- Correctness first: every variant matches the f64 oracle at a small valid shape. ---
            let mut rng = crate::diff::Rng::new(0x5177E);
            for v in PIPE_VARIANTS {
                let (m, k, n) = (v.bm * 2, v.bk * (v.stages + 2), v.bn * 2);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt(&a, &b, m, k, n);
                let c = gemm_nt_f16_pipe(g, &a, &b, m, k, n, v).unwrap();
                crate::diff::assert_close(&format!("{} gate", v.name), &c, &r, 5e-2, 2e-2);
            }
            eprintln!("[gate] all {} pipe variants match the f64 oracle ✓", PIPE_VARIANTS.len());

            // --- Clock warmup (same-run peak-vs-peak; cf. gemm_vs_peers). ---
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            // Each variant's cuBLAS baseline is sampled *adjacent* to it via `best_pair` (not once for the
            // whole sweep), so a parallel session sharing the GPU can't make a stale baseline inflate or
            // deflate the ratio (the run-to-run swing this sweep exposed). More rounds → more chances to
            // catch a low-contention sample for both.
            const ROUNDS: usize = 6;

            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a16_d = g.stream.memcpy_stod(&a16).unwrap();
                let b16_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let cs_c = cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap().iter().map(|x| x.abs() as f64).sum::<f64>();

                eprintln!("\n{sz}³ fp16 GEMM pipe sweep (same-run, cuBLAS-adjacent best_pair):");
                let mut best: Option<(&str, f64)> = None;
                for v in PIPE_VARIANTS {
                    if m % v.bm != 0 || n % v.bn != 0 || k % v.bk != 0 {
                        continue;
                    }
                    let f = g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), v.name).unwrap();
                    let cfg = pipe_cfg(v, m, n);
                    // Interleave variant and cuBLAS round-by-round (inlined, not a 2-closure helper: the
                    // variant borrows `g` shared via time_wmma, cuBLAS borrows it mutably — they can't be
                    // captured by two live closures, but the sequential calls' borrows end at each return).
                    let (mut s_v, mut s_cub) = (f64::INFINITY, f64::INFINITY);
                    for _ in 0..ROUNDS {
                        s_v = s_v.min(time_wmma(g, &f, cfg, dims, &a16_d, &b16_d, &mut c_d, 50));
                        s_cub = s_cub.min(time_cublas_gemm_nt_f16(g, m, k, n, 50).unwrap());
                    }
                    // Checksum cross-check at this size: the timed kernel computes cuBLAS's matrix.
                    let cs = gemm_nt_f16_pipe(g, &a, &b, m, k, n, v).unwrap().iter().map(|x| x.abs() as f64).sum::<f64>();
                    assert!(
                        (cs - cs_c).abs() / cs_c.max(1.0) < 3e-2,
                        "{sz}³ {} checksum {cs:.3e} vs cuBLAS {cs_c:.3e}",
                        v.name
                    );
                    let (gf, g_cub) = (flop / s_v, flop / s_cub);
                    let pct = 100.0 * gf / g_cub;
                    eprintln!(
                        "  {:<30}: {:>7.0} GFLOP/s | {:>5.1}% of cuBLAS ({:>6.0})  (smem={}KiB, {} warps)",
                        v.name,
                        gf / 1e9,
                        pct,
                        g_cub / 1e9,
                        v.smem_bytes() / 1024,
                        v.threads() / 32,
                    );
                    if best.map_or(true, |(_, p)| pct > p) {
                        best = Some((v.name, pct));
                    }
                }
                if let Some((name, pct)) = best {
                    eprintln!("  → best @{sz}³: {name} at {pct:.1}% of cuBLAS");
                }
            }
        });
    }

    /// **`ldmatrix` + XOR-swizzle + no-pad vs the hand-placed/padded workhorse** — the bet that the *proper*
    /// CUTLASS pairing (`mma_nt_f16_128_bk32_s2_r16_swz`: conflict-free gathers AT 3 CTAs/SM, vs the padded
    /// 2) reclaims the HBM-bound 4096³ the naive padded-`ldmatrix` lost. **Identical** tile / pipeline /
    /// raster / `mma.sync`; only the SMEM layout + fragment load differ. Same-run interleaved, cuBLAS-adjacent
    /// (the only honest metric under the shared-GPU clock swing); %-of-cuBLAS for both + the swz/hand ratio.
    #[test]
    #[ignore]
    fn mma_swizzle_vs_handplaced() {
        use crate::baselines::{gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_f16};
        use crate::ptx_wmma::{pipe_variant, PipeCfg};
        use half::f16;
        with_gpu("mma_swizzle_bench", |g| {
            if !peers_available(g) {
                eprintln!("[skip] mma_swizzle_vs_handplaced: cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let wh = *pipe_variant("mma_nt_f16_128_bk32_s2_r16");
            let swz = PipeCfg { name: "mma_nt_f16_128_bk32_s2_r16_swz", pad: 0, ..wh };
            eprintln!(
                "SMEM/CTA: hand-placed (padded) {} KiB → {} CTAs/SM | swz (no-pad) {} KiB → {} CTAs/SM",
                wh.smem_bytes() / 1024,
                100 * 1024 / wh.smem_bytes().max(1),
                swz.smem_bytes() / 1024,
                100 * 1024 / swz.smem_bytes().max(1),
            );
            let ptx = crate::ptx_wmma::wmma_f16_ptx();
            let mut rng = crate::diff::Rng::new(0x5712_0B57);
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            const ROUNDS: usize = 6;
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let cs_hp = gemm_nt_f16_pipe(g, &a, &b, m, k, n, &wh).unwrap().iter().map(|x| x.abs() as f64).sum::<f64>();
                let cs_sz = gemm_nt_f16_pipe(g, &a, &b, m, k, n, &swz).unwrap().iter().map(|x| x.abs() as f64).sum::<f64>();
                assert!((cs_hp - cs_sz).abs() / cs_hp.max(1.0) < 1e-3, "{sz}³ swz checksum {cs_sz:.3e} vs hand-placed {cs_hp:.3e}");
                let f_hp = g.function("wmma_f16", ptx, wh.name).unwrap();
                let f_sz = g.function("wmma_f16", ptx, swz.name).unwrap();
                let cfg = pipe_cfg(&wh, m, n);
                let (mut s_hp, mut s_sz, mut s_cub) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
                for _ in 0..ROUNDS {
                    s_hp = s_hp.min(time_wmma(g, &f_hp, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                    s_sz = s_sz.min(time_wmma(g, &f_sz, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                    s_cub = s_cub.min(time_cublas_gemm_nt_f16(g, m, k, n, 50).unwrap());
                }
                let g_cub = flop / s_cub;
                eprintln!(
                    "  {sz}³: hand-placed {:>6.0} GFLOP/s ({:>5.1}% cuBLAS) | swizzle {:>6.0} GFLOP/s ({:>5.1}% cuBLAS) | swz/hand {:>4.2}×",
                    flop / s_hp / 1e9,
                    100.0 * (flop / s_hp) / g_cub,
                    flop / s_sz / 1e9,
                    100.0 * (flop / s_sz) / g_cub,
                    s_hp / s_sz,
                );
            }
        });
    }

    /// Launch config for an experimental cliff variant: r16-rasterized ⇒ a 1-D grid of `(M/bm)·(N/bn)`
    /// blocks of `wm·wn·32` threads, all SMEM static (`shared_mem_bytes = 0`).
    #[cfg(feature = "gpu")]
    fn cliff_cfg_for(v: &crate::ptx_wmma::CliffCfg, m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (((m / v.bm) * (n / v.bn)) as u32, 1, 1),
            block_dim: (v.threads() as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// **Correctness gate for the GEMM-cliff candidates** (binding law #1: every kernel matches an
    /// independent f64 reference over the full output before any speed number counts). Each
    /// [`CLIFF_VARIANTS`] entry — padded/swizzle × pipeline depth × launch-bounds — computes `C = A·Bᵀ`
    /// and must match the f16-rounded f64 oracle within the fp16 GEMM tolerance (abs 1e-2, rel 2e-3).
    /// Runs under plain `cargo test --features gpu` (skips without a GPU); no cuBLAS/redist needed.
    #[test]
    fn gemm_cliff_matches_reference() {
        use crate::ptx_wmma::{gemm_cliff_ptx, CLIFF_VARIANTS};
        use half::f16;
        with_gpu("gemm_cliff_gate", |g| {
            let mut rng = crate::diff::Rng::new(0xC11F_6A7E);
            for &(m, k, n) in &[(256usize, 256usize, 256usize), (256, 160, 512), (512, 128, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let r = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let (mm, nn, kk) = (m as u32, n as u32, k as u32);
                for v in CLIFF_VARIANTS {
                    if m % v.bm != 0 || n % v.bn != 0 || k % v.bk != 0 {
                        continue; // this shape doesn't tile this variant's macro-tile
                    }
                    let f = g.function("gemm_cliff", gemm_cliff_ptx(), v.name).unwrap();
                    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                    let mut bld = g.stream.launch_builder(&f);
                    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
                    unsafe { bld.launch(cliff_cfg_for(v, m, n)).unwrap() };
                    let c = g.stream.memcpy_dtov(&c_d).unwrap();
                    let s = crate::diff::assert_close(&format!("{} {m}x{k}x{n}", v.name), &c, &r, 1e-2, 2e-3);
                    eprintln!("{:<20} {m}x{k}x{n}: max_abs={:.2e} max_rel={:.2e}", v.name, s.max_abs, s.max_rel);
                }
            }
        });
    }

    /// **Correctness gate for the dispatched w22 swizzle workhorses** (`mma_nt_{f16,bf16}_128_bk32_s2_r16_
    /// w22swz`, the GEMM-cliff win `gemm_nt_{f16,bf16}` route ≥48 MB to). Small shapes never reach the ≥48 MB
    /// arm, so this loads the production kernels by name and checks `C = A·Bᵀ` vs the {f16,bf16}-rounded f64
    /// oracle (128-thread w22 launch). Runs under plain `cargo test --features gpu`; no cuBLAS needed.
    #[test]
    fn gemm_cliff_w22swz_matches_reference() {
        use crate::ptx_wmma::{wmma_bf16_ptx, wmma_f16_ptx};
        use half::{bf16, f16};
        with_gpu("gemm_cliff_w22swz_gate", |g| {
            let mut rng = crate::diff::Rng::new(0x7722_5217);
            for &(m, k, n) in &[(256usize, 256usize, 256usize), (384, 160, 256)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let cfg = LaunchConfig {
                    grid_dim: (((m / 128) * (n / 128)) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (mm, nn, kk) = (m as u32, n as u32, k as u32);
                // f16
                let rf = ref_nt_rounded(&a, &b, m, k, n, |x| f16::from_f32(x).to_f32());
                let af: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let bf: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let ad = g.stream.memcpy_stod(&af).unwrap();
                let bd = g.stream.memcpy_stod(&bf).unwrap();
                let mut cd = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let f = g.function("wmma_f16", wmma_f16_ptx(), "mma_nt_f16_128_bk32_s2_r16_w22swz").unwrap();
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&mm).arg(&nn).arg(&kk).arg(&ad).arg(&bd).arg(&mut cd);
                unsafe { bld.launch(cfg).unwrap() };
                let cf = g.stream.memcpy_dtov(&cd).unwrap();
                let s = crate::diff::assert_close(&format!("f16 w22swz {m}x{k}x{n}"), &cf, &rf, 1e-2, 2e-3);
                eprintln!("f16  w22swz {m}x{k}x{n}: max_abs={:.2e}", s.max_abs);
                // bf16
                let rb = ref_nt_rounded(&a, &b, m, k, n, |x| bf16::from_f32(x).to_f32());
                let ab: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
                let bb: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
                let ad = g.stream.memcpy_stod(&ab).unwrap();
                let bd = g.stream.memcpy_stod(&bb).unwrap();
                let mut cd = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let f = g.function("wmma_bf16", wmma_bf16_ptx(), "mma_nt_bf16_128_bk32_s2_r16_w22swz").unwrap();
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&mm).arg(&nn).arg(&kk).arg(&ad).arg(&bd).arg(&mut cd);
                unsafe { bld.launch(cfg).unwrap() };
                let cb = g.stream.memcpy_dtov(&cd).unwrap();
                let s = crate::diff::assert_close(&format!("bf16 w22swz {m}x{k}x{n}"), &cb, &rb, 5e-2, 2e-2);
                eprintln!("bf16 w22swz {m}x{k}x{n}: max_abs={:.2e}", s.max_abs);
            }
        });
    }

    /// **GEMM-cliff A/B instrument** — the iteration loop for closing the large fp16 GEMM gap to cuBLAS.
    /// Clock-locking is denied on this mobile part, so absolute GFLOP/s is meaningless; the trustworthy
    /// signal is **ratio-of-best round-robin**: every kernel (each [`CLIFF_VARIANTS`] candidate + cuBLAS)
    /// is timed once per round so all sample the same clock evolution, and `best_of` converges each to its
    /// peak-clock time — ratio-of-best is *unbiased* for identical kernels (→1.0), unlike min-of-ratio
    /// which picks anti-correlated-noise extremes. A two-slot cuBLAS self-noise sentinel (~1.00 = trust)
    /// flags clock drift. Reports candidate %-of-cuBLAS, ×-vs-`cliff_swz_s2` (the production swizzle base),
    /// and **achieved CTAs/SM** (`occupancy_max_active_blocks_per_multiprocessor`). Checksum-cross-checked
    /// vs cuBLAS at each size; the f64-tolerance gate is `gemm_cliff_matches_reference`. 2048³ + 4096³, fp16.
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture gemm_cliff_ab`.
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn gemm_cliff_ab() {
        use crate::baselines::{
            cublas_gemm_nt_f16, gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_f16,
        };
        use crate::ptx_wmma::{gemm_cliff_ptx, CLIFF_VARIANTS};
        use half::f16;
        with_gpu("gemm_cliff_ab", |g| {
            if !peers_available(g) {
                eprintln!("[skip] gemm_cliff_ab: cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            // Clock warmup — pin the boost clock high before measuring (cf. gemm_pipe_sweep).
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            let mut rng = crate::diff::Rng::new(0xC11FF_AB);
            for sz in [2048usize, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let iters = if sz >= 4096 { 20 } else { 40 };
                let rounds = 10usize;
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let cs_ref = cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap().iter().map(|x| x.abs() as f64).sum::<f64>();
                // Preload every variant function, gate its checksum vs cuBLAS, record achieved occupancy.
                let mut variants: Vec<(&str, cudarc::driver::CudaFunction, u32, usize, LaunchConfig)> = Vec::new();
                for v in CLIFF_VARIANTS {
                    if m % v.bm != 0 || n % v.bn != 0 || k % v.bk != 0 {
                        continue;
                    }
                    let vcfg = cliff_cfg_for(v, m, n);
                    let f = g.function("gemm_cliff", gemm_cliff_ptx(), v.name).unwrap();
                    let occ = f
                        .occupancy_max_active_blocks_per_multiprocessor(v.threads() as u32, 0, None)
                        .unwrap_or(0);
                    let (mm, nn, kk) = dims;
                    {
                        let mut bld = g.stream.launch_builder(&f);
                        bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
                        unsafe { bld.launch(vcfg).unwrap() };
                    }
                    let cs = g.stream.memcpy_dtov(&c_d).unwrap().iter().map(|x| x.abs() as f64).sum::<f64>();
                    assert!(
                        (cs - cs_ref).abs() / cs_ref.max(1.0) < 3e-2,
                        "{sz}³ {} checksum {cs:.3e} vs cuBLAS {cs_ref:.3e}",
                        v.name
                    );
                    variants.push((v.name, f, occ, v.smem_bytes(), vcfg));
                }
                // Round-robin best-of-N. **Ratio-of-best is unbiased** for identical kernels (→ 1.0),
                // unlike min-of-ratio which picks anti-correlated-noise extremes. Every kernel is timed
                // once per round so all sample the same clock evolution; the min over rounds converges to
                // each kernel's peak-clock time. cuBLAS is timed in two slots as a noise sentinel — its
                // self-ratio should be ~1.00; a larger value means the clock was still drifting (distrust).
                let mut best: Vec<f64> = vec![f64::INFINITY; variants.len()];
                let (mut best_cub, mut best_cub2) = (f64::INFINITY, f64::INFINITY);
                for _ in 0..rounds {
                    for (i, (_, f, _, _, vcfg)) in variants.iter().enumerate() {
                        best[i] = best[i].min(time_wmma(g, f, *vcfg, dims, &a_d, &b_d, &mut c_d, iters));
                    }
                    best_cub = best_cub.min(time_cublas_gemm_nt_f16(g, m, k, n, iters as u32).unwrap());
                    best_cub2 = best_cub2.min(time_cublas_gemm_nt_f16(g, m, k, n, iters as u32).unwrap());
                }
                let base_i = variants.iter().position(|v| v.0 == "cliff_swz_s2").unwrap();
                let base_t = best[base_i];
                eprintln!(
                    "\n{sz}³ fp16 GEMM-cliff A/B (best-of-{rounds} round-robin; base=cliff_swz_s2; cuBLAS self-noise {:.3}×):",
                    best_cub2 / best_cub
                );
                for (i, (name, _, occ, smem, _)) in variants.iter().enumerate() {
                    eprintln!(
                        "  {:<24}: {:>7.0} GFLOP/s | {:>5.1}% cuBLAS | {:>6.3}× base | {} CTAs/SM, {}KiB",
                        name,
                        flop / best[i] / 1e9,
                        100.0 * best_cub / best[i],
                        base_t / best[i],
                        occ,
                        smem / 1024,
                    );
                }
            }
        });
    }

    /// **M5/M6: register-resident flash vs two peers — Tier-A naive CUDA-C *and* the Tier-B cuBLAS
    /// unfused attention chain**, same-run, single head, D=64. Mercury's `flash_d64_mp` (tensor-core
    /// `mma.sync`, O/m/l in registers, `cp.async`-staged double-buffered K/V, online softmax) vs:
    /// (A) `naive_attn` (one thread per query row, two-pass softmax, no SMEM/tensor cores) — the M6
    /// "beat the hand-written C flash" bar; and (B) `cublas_attn_chain` (QKᵀ and P·V on tensor-core
    /// cuBLAS, the S×S scores **materialized to HBM** with a softmax in between) — the M5 gold-standard
    /// *library* bar. Mercury's fused flash never spills the S×S scores, so the cuBLAS-chain gap **is**
    /// the value of fusion and grows with S. Reports GFLOP/s (`4·S²·D`) and Mercury × vs each at
    /// S∈{512..4096}. (A genuinely *fused* FA2-class CUDA-C peer is not buildable on this toolkit-free
    /// box — NVRTC has no header path, so `nvcuda::wmma` won't compile; see `nvrtc_wmma_probe`. The
    /// cuBLAS chain is the strongest library peer obtainable here.)
    ///
    /// Correctness gates speed (the first law): `naive_attn`, `cublas_attn_chain`, and `flash_d64_mp` are
    /// all first cross-checked against the f64 `ref_attn` oracle, and at every S their output checksums
    /// must agree. Same-run only (the ~7× laptop clock swing makes cross-run flash numbers meaningless):
    /// a clock warmup + `best_of`. Needs the NVRTC + cuBLAS redist DLLs on PATH; skips (never fails) if
    /// absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture flash_vs_peers`.
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn flash_vs_peers() {
        use crate::baselines::{
            attn_flop, cublas_attn_chain, nvrtc_naive_attn, peer_env_hint, peers_available,
            time_cublas_attn_chain, time_nvrtc_naive_attn,
        };
        use half::f16;
        with_gpu("flash_vs_peers", |g| {
            if !peers_available(g) {
                eprintln!("[skip] flash_vs_peers: NVRTC not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let d = 64usize;
            let mut rng = crate::diff::Rng::new(0xF1A57E5);

            // --- Correctness first: naive (f32 in) and Mercury flash (f16 in) both match the f64 oracle. ---
            for s in [64usize, 256] {
                let qf = rng.vec(s * d, -1.0, 1.0);
                let kf = rng.vec(s * d, -1.0, 1.0);
                let vf = rng.vec(s * d, -1.0, 1.0);
                let scale = 1.0f32 / (d as f32).sqrt();
                let oracle = ref_attn(&qf, &kf, &vf, s, d, scale);
                let naive = nvrtc_naive_attn(g, &qf, &kf, &vf, 1, s, d, scale).unwrap();
                let rel_f32 = ((8.0 * (d as f64).sqrt()) * f32::EPSILON as f64).max(1e-4);
                crate::diff::assert_close(&format!("naive attn s={s}"), &naive, &oracle, 1e-3, rel_f32);
                // cuBLAS chain takes f16 Q/K/V (the tensor-core dtype) ⇒ ~f16 tol; loose enough that only
                // a transpose/config slip (O(0.1+) scatter) trips it, tight enough to catch one.
                let chain = cublas_attn_chain(g, &qf, &kf, &vf, s, d, scale).unwrap();
                crate::diff::assert_close(&format!("cublas attn chain s={s}"), &chain, &oracle, 3e-2, 5e-2);
            }
            eprintln!("[gate] naive CUDA-C + cuBLAS-chain attention both match the f64 oracle ✓");

            // --- Clock warmup (peak-vs-peak, same-run). Hammer a GEMM until the mobile clock settles. ---
            let wa = rng.vec(1024 * 1024, -1.0, 1.0);
            let wb = rng.vec(1024 * 1024, -1.0, 1.0);
            for _ in 0..40 {
                gemm_nt_f16_sm_db(g, &wa, &wb, 1024, 1024, 1024).unwrap();
            }
            const ROUNDS: usize = 5;

            let f_m = g
                .function("flash", crate::ptx_flash::flash_ptx(), "flash_d64_mp")
                .unwrap();
            let to16 =
                |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
            for s in [512usize, 1024, 2048, 4096] {
                let qf = rng.vec(s * d, -1.0, 1.0);
                let kf = rng.vec(s * d, -1.0, 1.0);
                let vf = rng.vec(s * d, -1.0, 1.0);
                let scale = 1.0f32 / (d as f32).sqrt();
                let q16 = g.stream.memcpy_stod(&to16(&qf)).unwrap();
                let k16 = g.stream.memcpy_stod(&to16(&kf)).unwrap();
                let v16 = g.stream.memcpy_stod(&to16(&vf)).unwrap();
                let mut o_d = g.stream.alloc_zeros::<f32>(s * d).unwrap();
                let ss = s as u32;
                let cfg = wmma_flash_cfg(s);

                // Checksum cross-check: Mercury flash vs naive (f16-vs-f32 input ⇒ ~f16 tol on the sum).
                {
                    let mut bld = g.stream.launch_builder(&f_m);
                    bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o_d);
                    unsafe { bld.launch(cfg).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_m = g.stream.memcpy_dtov(&o_d).unwrap();
                let naive = nvrtc_naive_attn(g, &qf, &kf, &vf, 1, s, d, scale).unwrap();
                let csum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
                let (cs_m, cs_n) = (csum(&out_m), csum(&naive));
                assert!(
                    (cs_m - cs_n).abs() / cs_n.max(1.0) < 3e-2,
                    "S={s}: flash vs naive checksum disagree: mma={cs_m:.3e} naive={cs_n:.3e}"
                );

                // Speed (same-run, peak clock). Naive is O(S²D)/thread → very slow at long S; fewer iters.
                let t_m = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let mut bld = g.stream.launch_builder(&f_m);
                        bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o_d);
                        unsafe { bld.launch(cfg).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 100.0
                });
                let naive_iters = if s >= 2048 { 3 } else { 10 };
                let t_n = time_nvrtc_naive_attn(g, 1, s, d, scale, naive_iters).unwrap();

                // Tier-B cuBLAS unfused chain: cross-check vs naive (same f16-vs-f32 sum tol), then time
                // same-run. Heavier than the fused flash (materializes S×S) but far lighter than naive.
                let chain_out = cublas_attn_chain(g, &qf, &kf, &vf, s, d, scale).unwrap();
                let cs_c = csum(&chain_out);
                assert!(
                    (cs_c - cs_n).abs() / cs_n.max(1.0) < 3e-2,
                    "S={s}: cuBLAS chain vs naive checksum disagree: chain={cs_c:.3e} naive={cs_n:.3e}"
                );
                let t_c = time_cublas_attn_chain(g, s, d, scale, if s >= 2048 { 10 } else { 30 }).unwrap();

                let flop = attn_flop(1, s, d);
                let (g_m, g_n, g_c) = (flop / t_m, flop / t_n, flop / t_c);
                eprintln!(
                    "S={s:>4} D={d} (1 head, same-run): Mercury flash {:.4} ms ({:>6.0} GF/s) | cuBLAS chain {:.4} ms ({:>6.0} GF/s) | naive {:.4} ms ({:>5.0} GF/s) || Mercury {:>4.1}× vs cuBLAS-chain, {:>5.1}× vs naive",
                    t_m * 1e3,
                    g_m / 1e9,
                    t_c * 1e3,
                    g_c / 1e9,
                    t_n * 1e3,
                    g_n / 1e9,
                    g_m / g_c,
                    g_m / g_n,
                );
            }

            // --- Multi-head (GPT-2 shape: H=12, dh=64): grid.y=H fills the GPU the single-head case
            //     starves at small S. [H,S,D] layout; Mercury launches grid (S/16, H, 1). ---
            let heads = 12usize;
            for s in [512usize, 1024, 2048] {
                let n = heads * s * d;
                let qf = rng.vec(n, -1.0, 1.0);
                let kf = rng.vec(n, -1.0, 1.0);
                let vf = rng.vec(n, -1.0, 1.0);
                let scale = 1.0f32 / (d as f32).sqrt();
                let q16 = g.stream.memcpy_stod(&to16(&qf)).unwrap();
                let k16 = g.stream.memcpy_stod(&to16(&kf)).unwrap();
                let v16 = g.stream.memcpy_stod(&to16(&vf)).unwrap();
                let mut o_d = g.stream.alloc_zeros::<f32>(n).unwrap();
                let ss = s as u32;
                let cfg = LaunchConfig {
                    grid_dim: ((s / 16) as u32, heads as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // checksum cross-check vs naive multi-head
                {
                    let mut bld = g.stream.launch_builder(&f_m);
                    bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o_d);
                    unsafe { bld.launch(cfg).unwrap() };
                }
                g.stream.synchronize().unwrap();
                let out_m = g.stream.memcpy_dtov(&o_d).unwrap();
                let naive = nvrtc_naive_attn(g, &qf, &kf, &vf, heads, s, d, scale).unwrap();
                let csum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
                let (cs_m, cs_n) = (csum(&out_m), csum(&naive));
                assert!(
                    (cs_m - cs_n).abs() / cs_n.max(1.0) < 3e-2,
                    "H={heads} S={s}: flash vs naive checksum disagree: mma={cs_m:.3e} naive={cs_n:.3e}"
                );
                let t_m = best_of(ROUNDS, || {
                    let t0 = Instant::now();
                    for _ in 0..50 {
                        let mut bld = g.stream.launch_builder(&f_m);
                        bld.arg(&ss).arg(&scale).arg(&q16).arg(&k16).arg(&v16).arg(&mut o_d);
                        unsafe { bld.launch(cfg).unwrap() };
                    }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 50.0
                });
                let t_n = time_nvrtc_naive_attn(g, heads, s, d, scale, if s >= 2048 { 2 } else { 5 }).unwrap();
                let flop = attn_flop(heads, s, d);
                let (g_m, g_n) = (flop / t_m, flop / t_n);
                eprintln!(
                    "H={heads} S={s:>4} D={d} (same-run): Mercury flash {:.4} ms ({:>6.0} GFLOP/s) | naive CUDA-C {:.4} ms ({:>5.0} GFLOP/s) || Mercury {:>5.1}× vs naive",
                    t_m * 1e3,
                    g_m / 1e9,
                    t_n * 1e3,
                    g_n / 1e9,
                    g_m / g_n,
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

    /// **Beat-cuBLAS via fusion, on the FAST `mma.sync` workhorse**: `C = act(A·Bᵀ + bias)` — the
    /// canonical nn.Linear / transformer-FFN epilogue — as ONE kernel vs the two-kernel chain plain cuBLAS
    /// forces (cuBLAS sgemm writes C to HBM; a second epilogue kernel reads it back, adds `bias[col]`,
    /// applies the activation, writes again). Where [`fused_gemm_activation_vs_chain`] fuses onto the
    /// slower `_sm_db` WMMA base, this fuses onto the **r16-raster mma workhorse** — the *fastest*
    /// large-GEMM path (~90–97% of cuBLAS ≤2048³) — so the comparison is the honest one: a near-cuBLAS
    /// GEMM **plus a free epilogue** vs cuBLAS GEMM **plus a mandatory HBM round-trip**. (cuBLASLt *can*
    /// fuse a bias+gelu epilogue, but cudarc exposes only plain cublas Matmul — f32/f16/bf16 — so the
    /// honest peer with this toolchain is the two-kernel chain; the epilogue's cost is its 2·M·N f32
    /// round-trip, proxied by a `time_vmath` pass.) Correctness gates speed: the fused output must equal
    /// `act(cuBLAS GEMM + bias)` at every shape. Redist DLLs on PATH (see `gemm_vs_peers`).
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn fused_gemm_bias_act_vs_chain() {
        use crate::baselines::{
            cublas_gemm_nt_f16, gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_f16,
        };
        use half::f16;
        with_gpu("fused_bias_act", |g| {
            if !peers_available(g) {
                eprintln!("[skip] fused_gemm_bias_act_vs_chain: cuBLAS/NVRTC not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gelu = |x: f32| {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (c0 * (x + 0.044715 * x * x * x)).tanh())
            };
            let wh = mma_workhorse();
            let ptx = crate::ptx_wmma::wmma_f16_ptx();
            let mut rng = crate::diff::Rng::new(0xB1A5_F0ED);
            // Warm the clock (cf. fused_gemm_activation_vs_chain / gemm_vs_peers): the cuBLAS baseline is
            // measured first (cold) and the fused kernel later (warm), so without a warmup the laptop GPU's
            // load-ramp boost would inflate the ratio — a cold-vs-warm artifact, not an honest peak-vs-peak.
            for _ in 0..30 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            for sz in [512usize, 1024, 2048] {
                let (m, k, n) = (sz, sz, sz);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let bias_d = g.stream.memcpy_stod(&bias).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let dims = (m as u32, n as u32, k as u32);
                let cfg = pipe_cfg(wh, m, n);
                let flop = gemm_flop(m, n, k);

                // cuBLAS GEMM output — the chain's first kernel and the correctness-reference base.
                let cub_out = cublas_gemm_nt_f16(g, &a, &b, m, k, n).unwrap();
                eprintln!("\n{sz}³ act(A·Bᵀ + bias) on the mma workhorse (same-run, on-device):");

                for act in ["bias", "bias_relu", "bias_silu", "bias_gelu"] {
                    let entry: &'static str = match act {
                        "bias" => "mma_nt_f16_128_bk32_s2_r16_bias",
                        "bias_relu" => "mma_nt_f16_128_bk32_s2_r16_bias_relu",
                        "bias_silu" => "mma_nt_f16_128_bk32_s2_r16_bias_silu",
                        _ => "mma_nt_f16_128_bk32_s2_r16_bias_gelu",
                    };
                    // Correctness: the fused output must equal act(cuBLAS GEMM + bias[col]).
                    let fused = match act {
                        "bias" => gemm_nt_f16_mma_bias(g, &a, &b, &bias, m, k, n),
                        "bias_relu" => gemm_nt_f16_mma_bias_relu(g, &a, &b, &bias, m, k, n),
                        "bias_silu" => gemm_nt_f16_mma_bias_silu(g, &a, &b, &bias, m, k, n),
                        _ => gemm_nt_f16_mma_bias_gelu(g, &a, &b, &bias, m, k, n),
                    }
                    .unwrap();
                    let act_fn = |x: f32| match act {
                        "bias" => x,
                        "bias_relu" => x.max(0.0),
                        "bias_silu" => silu(x),
                        _ => gelu(x),
                    };
                    let refout: Vec<f32> =
                        cub_out.iter().enumerate().map(|(i, &x)| act_fn(x + bias[i % n])).collect();
                    crate::diff::assert_close(
                        &format!("fused {act} vs cuBLAS chain {sz}³"),
                        &fused,
                        &refout,
                        5e-2,
                        2e-2,
                    );

                    // Timing: the fused one-kernel vs cuBLAS GEMM + a separate bias+activation epilogue.
                    // The epilogue's cost is its 2·M·N f32 HBM round-trip, proxied by a vmath pass (the
                    // plain-bias variant has no activation, so its epilogue is just the round-trip — relu's
                    // vmath time stands in: same memory traffic, a slight over-charge for the no-op math).
                    let f_fused = g.function("wmma_f16", ptx, entry).unwrap();
                    let proxy_act = if act == "bias" { "relu" } else { &act[5..] };
                    let f_act = g.function("vmath", crate::ptx::vmath_ptx(), proxy_act).unwrap();
                    // The **pipe_64_s6** fused twin — the ≤1024³ champion base. At 1024³ the mma-workhorse
                    // base loses (its GEMM is ~80% of cuBLAS there vs pipe_64_s6's ~90%), so the workhorse
                    // fused kernel's GEMM deficit outweighs the saved epilogue round-trip; the pipe_64_s6
                    // base should flip that to a win. Timed in the SAME interleaved window for an honest A/B.
                    let do_p64 = sz <= 1024;
                    let p64_entry: &'static str = match act {
                        "bias" => "wmma_nt_f16_pipe_64_s6_bias",
                        "bias_relu" => "wmma_nt_f16_pipe_64_s6_bias_relu",
                        "bias_silu" => "wmma_nt_f16_pipe_64_s6_bias_silu",
                        _ => "wmma_nt_f16_pipe_64_s6_bias_gelu",
                    };
                    let f_p64 = if do_p64 { Some(g.function("wmma_f16", ptx, p64_entry).unwrap()) } else { None };
                    let cfg64 = pipe_cfg(pipe64(), m, n);
                    // Contention-robust same-run timing: interleave the cuBLAS GEMM, the epilogue round-trip,
                    // and the fused kernel(s) round-by-round, taking each kernel's min across rounds, so all
                    // see the same least-contended clock window. The parallel flash session bursts the GPU; a
                    // once-per-size cuBLAS baseline would go stale against a later fused sample (the cliff
                    // sweep hit exactly this). The min-of-rounds is the peak-clock, least-throttled read.
                    let (mut bc, mut be, mut bf, mut bp) =
                        (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
                    for _ in 0..6 {
                        bc = bc.min(time_cublas_gemm_nt_f16(g, m, k, n, 20).unwrap());
                        be = be.min(time_vmath(g, &f_act, m * n, 20));
                        bf = bf.min(time_wmma_bias(g, &f_fused, cfg, dims, &a_d, &b_d, &mut c_d, &bias_d, 20));
                        if let Some(ref f) = f_p64 {
                            bp = bp.min(time_wmma_bias(g, f, cfg64, dims, &a_d, &b_d, &mut c_d, &bias_d, 20));
                        }
                    }
                    let (t_cub, t_epi, t_fused) = (bc, be, bf);
                    let cub_chain = t_cub + t_epi;
                    eprintln!(
                        "  {act:>9}: mma    {:>7.3} ms ({:>6.0} GFLOP/s) | cuBLAS GEMM {:>6.3} + epilogue {:>5.3} = {:>7.3} ms (mma {:>4.2}× faster)",
                        t_fused * 1e3,
                        flop / t_fused / 1e9,
                        t_cub * 1e3,
                        t_epi * 1e3,
                        cub_chain * 1e3,
                        cub_chain / t_fused,
                    );
                    if do_p64 {
                        eprintln!(
                            "  {:>9}: pipe64 {:>7.3} ms ({:>6.0} GFLOP/s) | cuBLAS chain {:>30.3} ms (pipe64 {:>4.2}× faster)",
                            "", bp * 1e3, flop / bp / 1e9, cub_chain * 1e3, cub_chain / bp,
                        );
                    }
                }

                // Residual arm: out = A·Bᵀ + bias + residual (the down-proj / attention output-proj). The
                // plain-cuBLAS chain runs the GEMM then a residual-add kernel that reads C, adds bias[col]
                // and the residual, writes out — a 3·M·N round-trip (read C + read residual + write),
                // proxied by `time_vadd`. The fused mma kernel folds both adds into the GEMM store. Same
                // interleaved same-run methodology as the activation arm above.
                {
                    let resid = rng.vec(m * n, -1.0, 1.0);
                    let resid_d = g.stream.memcpy_stod(&resid).unwrap();
                    let fused = gemm_nt_f16_mma_bias_residual(g, &a, &b, &bias, &resid, m, k, n).unwrap();
                    let refout: Vec<f32> =
                        cub_out.iter().enumerate().map(|(i, &x)| x + bias[i % n] + resid[i]).collect();
                    crate::diff::assert_close(
                        &format!("fused bias_residual vs cuBLAS chain {sz}³"),
                        &fused,
                        &refout,
                        5e-2,
                        2e-2,
                    );
                    let f_fused = g
                        .function("wmma_f16", ptx, "mma_nt_f16_128_bk32_s2_r16_bias_residual")
                        .unwrap();
                    let f_vadd = g.function("vadd", crate::ptx::VADD, "vadd").unwrap();
                    // The pipe_64_s6 down-proj twin — interleaved for the ≤1024³ A/B (wins where mma loses).
                    let do_p64 = sz <= 1024;
                    let f_p64 = if do_p64 {
                        Some(g.function("wmma_f16", ptx, "wmma_nt_f16_pipe_64_s6_bias_residual").unwrap())
                    } else {
                        None
                    };
                    let cfg64 = pipe_cfg(pipe64(), m, n);
                    let (mut bc, mut be, mut bf, mut bp) =
                        (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
                    for _ in 0..6 {
                        bc = bc.min(time_cublas_gemm_nt_f16(g, m, k, n, 20).unwrap());
                        be = be.min(time_vadd(g, &f_vadd, m * n, 20));
                        bf = bf.min(time_wmma_bias_residual(
                            g, &f_fused, cfg, dims, &a_d, &b_d, &mut c_d, &bias_d, &resid_d, 20,
                        ));
                        if let Some(ref f) = f_p64 {
                            bp = bp.min(time_wmma_bias_residual(
                                g, f, cfg64, dims, &a_d, &b_d, &mut c_d, &bias_d, &resid_d, 20,
                            ));
                        }
                    }
                    let (t_cub, t_epi, t_fused) = (bc, be, bf);
                    let cub_chain = t_cub + t_epi;
                    eprintln!(
                        "  bias_residual: mma    {:>7.3} ms ({:>6.0} GFLOP/s) | cuBLAS GEMM {:>6.3} + residual-add {:>5.3} = {:>7.3} ms (mma {:>4.2}× faster)",
                        t_fused * 1e3,
                        flop / t_fused / 1e9,
                        t_cub * 1e3,
                        t_epi * 1e3,
                        cub_chain * 1e3,
                        cub_chain / t_fused,
                    );
                    if do_p64 {
                        eprintln!(
                            "  bias_residual: pipe64 {:>7.3} ms ({:>6.0} GFLOP/s) | cuBLAS chain {:>30.3} ms (pipe64 {:>4.2}× faster)",
                            bp * 1e3, flop / bp / 1e9, cub_chain * 1e3, cub_chain / bp,
                        );
                    }
                }
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

    /// Per-iter device time of a fused-bias mma GEMM (`mma_nt_f16_128_bk32_s2_r16_bias{,_relu,_silu,
    /// _gelu}`), which takes `bias[N]` (f32) as a 7th param after C. Same warmup+loop shape as
    /// [`time_wmma`]; the launch is identical to the residual timer's but the trailing buffer is length-N.
    fn time_wmma_bias<T: cudarc::driver::DeviceRepr>(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<T>,
        b_d: &cudarc::driver::CudaSlice<T>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        bias_d: &cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(b_d).arg(c_d).arg(bias_d);
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

    /// Per-iter device time of a fused bias+residual mma GEMM (`..._bias_residual`), which takes both
    /// `bias[N]` and `residual[M,N]` (f32) as the 7th and 8th params after C. Same warmup+loop as
    /// [`time_wmma`].
    fn time_wmma_bias_residual<T: cudarc::driver::DeviceRepr>(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<T>,
        b_d: &cudarc::driver::CudaSlice<T>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        bias_d: &cudarc::driver::CudaSlice<f32>,
        resid_d: &cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(b_d).arg(c_d).arg(bias_d).arg(resid_d);
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

    /// Per-iter device time of the fused gated-FFN kernel (`mma_nt_{f16,bf16}_128x64_gate_*`, no-bias
    /// variants): the dual-B gate takes x, Wg, Wu (half-precision) and writes the f32 gated output C —
    /// seven args (M,N,K,x,Wg,Wu,C). Same warmup+loop shape as [`time_wmma`].
    fn time_gate<T: cudarc::driver::DeviceRepr>(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        x_d: &cudarc::driver::CudaSlice<T>,
        wg_d: &cudarc::driver::CudaSlice<T>,
        wu_d: &cudarc::driver::CudaSlice<T>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(x_d).arg(wg_d).arg(wu_d).arg(c_d);
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

    /// **Beat-cuBLAS via gated-FFN fusion**: the SwiGLU gate `out = silu(x·Wgᵀ) ⊙ (x·Wuᵀ)` as ONE dual-B
    /// kernel vs the **three-kernel** chain a GEMM library must run — GEMM `x·Wgᵀ`, GEMM `x·Wuᵀ`, then an
    /// elementwise `silu(gate)⊙up` kernel (a 3·M·N HBM round-trip: read gate, read up, write out — proxied
    /// by `time_vadd`, identical traffic). The fused kernel reads `x` ONCE (shared A fragments feed both
    /// GEMMs) and never materializes the two `[M,N]` intermediates, so it folds away both the redundant
    /// `x` read and the 4·M·N intermediate round-trip cuBLAS cannot avoid (it has no fused-gate path).
    /// Same contention-robust **interleaved best-of-6** same-run methodology as
    /// [`fused_gemm_bias_act_vs_chain`] — a once-per-size baseline goes stale under the parallel flash
    /// session's bursts. GFLOP/s counts both GEMMs (the gate's real work). The in-bench check is a loose
    /// gross-error guard (the two f16-accumulation orders' silu-product compounds at large K); the binding
    /// correctness proof is [`swiglu_gate_match_reference_within_tol`] vs an exact f64 reference.
    #[test]
    #[ignore]
    fn fused_swiglu_gate_vs_chain() {
        use crate::baselines::{
            cublas_gemm_nt_f16, gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_f16,
        };
        use half::f16;
        with_gpu("swiglu_gate_bench", |g| {
            if !peers_available(g) {
                eprintln!("[skip] fused_swiglu_gate_vs_chain: cuBLAS/NVRTC not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let ptx = crate::ptx_wmma::wmma_f16_ptx();
            let mut rng = crate::diff::Rng::new(0x5_71_6C_BE);
            // Warm the clock: cuBLAS is measured before the fused kernel (cf. fused_gemm_bias_act_vs_chain).
            for _ in 0..30 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            for sz in [512usize, 1024, 2048] {
                let (m, k, n) = (sz, sz, sz);
                let x = rng.vec(m * k, -1.0, 1.0);
                let wg = rng.vec(n * k, -1.0, 1.0);
                let wu = rng.vec(n * k, -1.0, 1.0);
                let x16: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
                let wg16: Vec<f16> = wg.iter().map(|&v| f16::from_f32(v)).collect();
                let wu16: Vec<f16> = wu.iter().map(|&v| f16::from_f32(v)).collect();
                let x_d = g.stream.memcpy_stod(&x16).unwrap();
                let wg_d = g.stream.memcpy_stod(&wg16).unwrap();
                let wu_d = g.stream.memcpy_stod(&wu16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let dims = (m as u32, n as u32, k as u32);
                let cfg = gate_cfg(m, n);
                let flop = 2.0 * gemm_flop(m, n, k); // the gate is two GEMMs' worth of FLOPs

                // In-bench gross-error guard: fused ≈ silu(cuBLAS x·Wgᵀ) ⊙ (cuBLAS x·Wuᵀ). Loose tol — the
                // products reach |·|~225 at K=2048 and two f16-accumulation orders compound there.
                let gate_cub = cublas_gemm_nt_f16(g, &x, &wg, m, k, n).unwrap();
                let up_cub = cublas_gemm_nt_f16(g, &x, &wu, m, k, n).unwrap();
                let fused = gemm_nt_f16_swiglu(g, &x, &wg, &wu, m, k, n).unwrap();
                let refout: Vec<f32> =
                    gate_cub.iter().zip(&up_cub).map(|(&gv, &uv)| silu(gv) * uv).collect();
                crate::diff::assert_close(&format!("fused swiglu vs cuBLAS chain {sz}³"), &fused, &refout, 2e-1, 5e-2);

                // Timing: fused one-kernel vs 2 cuBLAS GEMMs + 1 elementwise silu⊙ (vadd proxy, 3·M·N).
                let f_fused = g.function("wmma_f16", ptx, "mma_nt_f16_128x64_gate_silu").unwrap();
                let f_vadd = g.function("vadd", crate::ptx::VADD, "vadd").unwrap();
                let (mut bc, mut be, mut bf) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
                for _ in 0..6 {
                    bc = bc.min(time_cublas_gemm_nt_f16(g, m, k, n, 20).unwrap());
                    be = be.min(time_vadd(g, &f_vadd, m * n, 20));
                    bf = bf.min(time_gate(g, &f_fused, cfg, dims, &x_d, &wg_d, &wu_d, &mut c_d, 20));
                }
                let chain = 2.0 * bc + be; // two GEMMs + the elementwise combine
                eprintln!(
                    "  {sz}³ swiglu: fused {:>7.3} ms ({:>6.0} GFLOP/s) | cuBLAS 2×GEMM {:>6.3} + silu⊙ {:>5.3} = {:>7.3} ms (fused {:>4.2}× faster)",
                    bf * 1e3,
                    flop / bf / 1e9,
                    2.0 * bc * 1e3,
                    be * 1e3,
                    chain * 1e3,
                    chain / bf,
                );
            }
        });
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

    /// Per-iter device time of the W4A16 decode kernel (`gemm_nt_w4a16`, symmetric): args
    /// `(M,N,K, A_f16, Bq_u32, Scales_f16, C_f32)` over resident buffers. Same warmup+loop shape as
    /// [`time_wmma`], so the ratio vs the fp16 path and the naive peer is same-run apples-to-apples.
    #[allow(clippy::too_many_arguments)]
    fn time_w4a16(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<half::f16>,
        bq_d: &cudarc::driver::CudaSlice<u32>,
        scl_d: &cudarc::driver::CudaSlice<half::f16>,
        c_d: &mut cudarc::driver::CudaSlice<f32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<f32>| {
            let mut bld = g.stream.launch_builder(f);
            bld.arg(&mm).arg(&nn).arg(&kk).arg(a_d).arg(bq_d).arg(scl_d).arg(c_d);
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

    /// **The honest W4A16 (int4 weight-only decode) scoreboard — M4 / M6.** Mercury's int4-decode GEMM
    /// measured *same-run, same buffers* against (Tier A) a **naive CUDA-C W4A16** kernel compiled by
    /// NVRTC — the literal "beat the hand-written int4 decode kernel" — and against Mercury's **own
    /// fp16 GEMM on the identical 64×64 tile**, which isolates the weight-bandwidth win: the *only*
    /// difference is the B-load (packed int4 vs full fp16), so the ratio is the value of moving 4× fewer
    /// weight bytes. **Tier B is honestly empty:** there is no robust general W4A16-decode GEMM bindable
    /// through `cudarc` (cuBLASLt offers none), so the strongest *measurable* int4 peer is the naive
    /// kernel and M4 stands as a documented lead — stated in the output, not papered over.
    ///
    /// Correctness gates speed (first law): Mercury and the naive peer are first cross-checked against
    /// the f64 dequant reference, and at each timing shape their checksums must agree. Needs the redist
    /// DLLs on PATH; skips (never fails) if absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int4_gemm_vs_peers`
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn int4_gemm_vs_peers() {
        use crate::baselines::{
            gemm_flop, nvrtc_naive_w4a16, peer_env_hint, peers_available, time_cublas_gemm_nt_f16,
            time_nvrtc_naive_w4a16,
        };
        use crate::ptx_int4::{
            quantize_weight_symmetric, reference_w4a16, GROUP_SIZE, W4_BM, W4_BN, W4_THREADS,
        };
        use half::f16;
        with_gpu("int4_gemm_vs_peers", |g| {
            if !peers_available(g) {
                eprintln!("[skip] int4_gemm_vs_peers: NVRTC/cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let group = GROUP_SIZE;

            // --- Correctness first: Mercury W4A16 + the naive peer both match the f64 dequant oracle. ---
            let mut rng = crate::diff::Rng::new(0x4B17);
            for (m, n, k) in [(64usize, 128usize, 256usize), (128, 128, 384)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let w = rng.vec(n * k, -0.8, 0.8);
                let qw = quantize_weight_symmetric(&w, n, k, group);
                let r = reference_w4a16(&a, &qw, m);
                let merc = gemm_nt_w4a16(g, &a, &qw, m, k, n).unwrap();
                crate::diff::assert_close(&format!("Mercury W4A16 {m}x{k}x{n}"), &merc, &r, 1e-2, 2e-3);
                let naive = nvrtc_naive_w4a16(g, &a, &qw, m, k, n).unwrap();
                crate::diff::assert_close(&format!("naive W4A16 {m}x{k}x{n}"), &naive, &r, 5e-2, 2e-2);
            }
            eprintln!("[gate] Mercury W4A16 + naive CUDA-C W4A16 both match the f64 dequant oracle ✓");
            eprintln!(
                "[peer] No robust library int4-decode GEMM is bindable here (cuBLASLt has no general \
                 W4A16 decode), so naive CUDA-C is the honest Tier-A peer and M4 is a *documented lead*."
            );

            // --- Clock warmup (same-run peak-vs-peak; the ~7× boost ramp corrupts a cold first shape). ---
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_f16(g, 2048, 2048, 2048, 20);
            }
            const ROUNDS: usize = 4;

            // Shapes: decode-like (small M, big N=K) where weight-BW dominates → the int4 win shows;
            // plus a squarer, more compute-leaning shape. All M%64==0, N%64==0, K%128==0.
            for (m, k, n) in [
                (64usize, 4096usize, 4096usize),
                (64, 2048, 2048),
                (256, 2048, 2048),
                (512, 512, 512),
            ] {
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let w = rng.vec(n * k, -0.8, 0.8);
                let qw = quantize_weight_symmetric(&w, n, k, group);

                // Mercury W4A16 — resident packed int4 weights + fp16 activations.
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let bq_d = g.stream.memcpy_stod(&qw.packed).unwrap();
                let scl_d = g.stream.memcpy_stod(&qw.scales).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let f_w4 = g.function("w4a16", crate::ptx_int4::w4a16_ptx(), "gemm_nt_w4a16").unwrap();
                let cfg_w4 = LaunchConfig {
                    grid_dim: ((n / W4_BN) as u32, (m / W4_BM) as u32, 1),
                    block_dim: (W4_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_w4 = best_of(ROUNDS, || {
                    time_w4a16(g, &f_w4, cfg_w4, dims, &a_d, &bq_d, &scl_d, &mut c_d, 50)
                });

                // Static-shape-specialized kernel (M/N/K baked) — Mercury's no-library lever. ptxas
                // strength-reduces the baked strides (`×K`/`×N` → shifts for the power-of-2 dims here);
                // the dynamic kernel keeps them as register multiplies. To isolate the *baked-constants*
                // effect, time the static kernel against a DYNAMIC kernel loaded the SAME raw-JIT way
                // (the headline s_w4 above uses the cubin cache, a different ptxas opt level — comparing
                // to it would confound the loading path with the specialization).
                let ptx_s = crate::ptx_int4::w4a16_static_ptx(m, n, k, false);
                let mod_s = g.ctx.load_module(ptx_s.as_str().into()).unwrap();
                let f_s = mod_s.load_function(crate::ptx_int4::w4a16_static_entry(false)).unwrap();
                let s_static = best_of(ROUNDS, || {
                    time_w4a16(g, &f_s, cfg_w4, dims, &a_d, &bq_d, &scl_d, &mut c_d, 50)
                });
                let mod_dyn_raw = g.ctx.load_module(crate::ptx_int4::w4a16_ptx().into()).unwrap();
                let f_dyn_raw = mod_dyn_raw.load_function("gemm_nt_w4a16").unwrap();
                let s_dyn_raw = best_of(ROUNDS, || {
                    time_w4a16(g, &f_dyn_raw, cfg_w4, dims, &a_d, &bq_d, &scl_d, &mut c_d, 50)
                });

                // Mercury fp16 on the SAME 64×64 tile (`wmma_nt_f16_sm`) — full fp16 weights. The only
                // difference vs W4A16 is the B-load (fp16 vs packed int4), so s_f16/s_w4 IS the
                // weight-bandwidth win in the decode regime.
                let b16: Vec<f16> = w.iter().map(|&x| f16::from_f32(x)).collect();
                let bf16_d = g.stream.memcpy_stod(&b16).unwrap();
                let f_f16 =
                    g.function("wmma_f16", crate::ptx_wmma::wmma_f16_ptx(), "wmma_nt_f16_sm").unwrap();
                let s_f16 = best_of(ROUNDS, || {
                    time_wmma(g, &f_f16, wmma_sm_cfg(m, n), dims, &a_d, &bf16_d, &mut c_d, 50)
                });

                // Naive CUDA-C W4A16 (Tier A). Slow (one thread/output, full K-loop) → fewer iters.
                let naive_iters = if (m * n * k) as u64 >= 500_000_000 { 3 } else { 10 };
                let s_naive = time_nvrtc_naive_w4a16(g, m, k, n, group, naive_iters).unwrap();

                // Checksum cross-check: Mercury and naive compute the same matrix (within the dequant gap).
                let csum = |v: &[f32]| v.iter().map(|x| x.abs() as f64).sum::<f64>();
                let cs_w4 = csum(&gemm_nt_w4a16(g, &a, &qw, m, k, n).unwrap());
                let cs_n = csum(&nvrtc_naive_w4a16(g, &a, &qw, m, k, n).unwrap());
                assert!(
                    (cs_w4 - cs_n).abs() / cs_n.max(1.0) < 3e-2,
                    "{m}x{k}x{n} checksum disagreement: w4={cs_w4:.3e} naive={cs_n:.3e}"
                );

                // Weight bytes moved from HBM per pass: int4 = N·K/2, fp16 = N·K·2 (the structural 4×).
                let (wb_int4, wb_fp16) = ((n * k / 2) as f64, (n * k * 2) as f64);
                let (g_w4, g_static, g_dyn_raw, g_f16, g_naive) = (
                    flop / s_w4,
                    flop / s_static,
                    flop / s_dyn_raw,
                    flop / s_f16,
                    flop / s_naive,
                );
                eprintln!(
                    "\n{m}x{k}x{n} W4A16 (same-run):\n  \
                     Mercury W4A16     : {:>8.0} GFLOP/s | {:>6.1}× vs naive CUDA-C | {:>5.2}× vs Mercury fp16 (same tile)\n  \
                     Mercury W4A16 stat: {:>8.0} GFLOP/s | {:>5.2}× vs dynamic (same raw-JIT load; M/N/K baked)\n  \
                     Mercury fp16      : {:>8.0} GFLOP/s | full fp16 weights — the HBM traffic int4 avoids\n  \
                     naive CUDA-C      : {:>8.0} GFLOP/s | Tier-A int4 baseline (no robust library peer exists)\n  \
                     weight HBM/pass: int4 {:.1} MB vs fp16 {:.1} MB ({:.1}× less weight traffic)",
                    g_w4 / 1e9,
                    g_w4 / g_naive,
                    g_w4 / g_f16,
                    g_static / 1e9,
                    g_static / g_dyn_raw,
                    g_f16 / 1e9,
                    g_naive / 1e9,
                    wb_int4 / 1e6,
                    wb_fp16 / 1e6,
                    wb_fp16 / wb_int4,
                );
            }
        });
    }

    // ============================================================================================
    // M7 runtime (Phase 7): device memory pool + CUDA graphs. Gates assert the pooled / graphed /
    // multi-stream path is **bit-identical** to the per-op-alloc + individual-launch baseline (the
    // runtime changes when/where memory lives and how launches issue, not *what* is computed), and
    // benches report the latency delta **same-run** at decode/small-batch shapes.
    // ============================================================================================

    /// Build a `ResidentLayerF16` and random `[S,D]` input for a `(s,d,dff,heads)` case — the shared
    /// setup for the pool/graph gates and benches.
    fn pool_layer_fixture(
        g: &mut Gpu,
        s: usize,
        d: usize,
        dff: usize,
        heads: usize,
        seed: u64,
    ) -> (ResidentLayerF16, Vec<f32>) {
        let mut rng = crate::diff::Rng::new(seed);
        let wq = rng.vec(d * d, -0.08, 0.08);
        let wk = rng.vec(d * d, -0.08, 0.08);
        let wv = rng.vec(d * d, -0.08, 0.08);
        let wo = rng.vec(d * d, -0.08, 0.08);
        let w1 = rng.vec(dff * d, -0.05, 0.05);
        let w2 = rng.vec(d * dff, -0.05, 0.05);
        let w = TransformerWeights { wq: &wq, wk: &wk, wv: &wv, wo: &wo, w1: &w1, w2: &w2 };
        let layer = ResidentLayerF16::new_mha(g, &w, s, d, dff, heads).unwrap();
        let x = rng.vec(s * d, -1.0, 1.0);
        (layer, x)
    }

    /// **Identical-numerics gate (the first law).** The pooled forward must produce output
    /// **bit-for-bit** equal to the per-op-`alloc_zeros` eager forward — same kernels, same launch
    /// order, only the scratch provenance differs. The slab is **poisoned with 0xFF (NaN-ish)** before
    /// the pooled run, so any intermediate read before it is fully written would leak a NaN and fail
    /// the equality — proving the uninitialized `alloc` fast path is only used on full-overwrite
    /// outputs. Covers the single-head (f32 flash) and multi-head (tensor-core flash + transposes)
    /// attention paths, plus the real GPT-2 layer shape.
    #[test]
    fn resident_layer_pooled_matches_eager() {
        with_gpu("resident_layer_pooled_matches_eager", |g| {
            // (S, D, Dff, heads): single-head small (f32 flash); multi-head (tensor-core flash);
            // GPT-2 layer (D=768/H=12).
            let cases = [(64usize, 64usize, 256usize, 1usize), (512, 128, 512, 2), (512, 768, 3072, 12)];
            for (ci, &(s, d, dff, heads)) in cases.iter().enumerate() {
                let (layer, x) = pool_layer_fixture(g, s, d, dff, heads, 0x9001 + ci as u64);
                let stream = g.stream.clone();
                let x_d = stream.memcpy_stod(&x).unwrap();

                // eager reference (per-op alloc_zeros + individual launches).
                let ref_out = layer.forward_device(&x_d).unwrap();
                let ref_host = stream.memcpy_dtov(&ref_out).unwrap();

                // pooled: hostile (poisoned) slab + persistent out buffer outside the arena.
                let cap = 256 * 1024 * 1024; // generous; high-water reports the real footprint.
                let mut pool = crate::pool::DevicePool::new(stream.clone(), cap).unwrap();
                pool.poison(0xFF).unwrap();
                let mut out_d = stream.alloc_zeros::<f32>(s * d).unwrap();
                layer.forward_device_pooled(&mut pool, &x_d, &mut out_d).unwrap();
                let pooled_host = stream.memcpy_dtov(&out_d).unwrap();

                assert_eq!(pooled_host.len(), ref_host.len());
                for i in 0..ref_host.len() {
                    assert_eq!(
                        pooled_host[i].to_bits(),
                        ref_host[i].to_bits(),
                        "pooled != eager at {i} (S={s} D={d} Dff={dff} heads={heads})"
                    );
                }
                assert!(
                    pool.high_water_bytes() <= cap,
                    "pool overflowed: high_water {} > cap {}",
                    pool.high_water_bytes(),
                    cap
                );
                eprintln!(
                    "pooled==eager bit-identical: S={s} D={d} Dff={dff} heads={heads}; \
                     pool high-water {} KiB across {} sub-allocs (cap {} MiB)",
                    pool.high_water_bytes() / 1024,
                    pool.served(),
                    cap / (1024 * 1024)
                );
            }
        });
    }

    /// Run `f` with the shared context's event tracking **disabled**, re-enabling on the way out
    /// (panic-safe). cudarc records per-buffer read/write events (on by default) and, once a second
    /// stream exists, inserts a cross-stream `cuStreamWaitEvent` on every buffer use — which
    /// `cuStreamBeginCapture` rejects as a dependency on uncaptured work. A buffer created while
    /// tracking is off carries no events, so a multi-stream capture inserts no such waits. The whole
    /// GPU suite is serialized under one process-wide mutex, so toggling the shared context here races
    /// with nothing; the layer/buffers built inside `f` must be created here (so they are event-free),
    /// and the caller is responsible for explicit stream synchronization (done in the helpers below).
    fn with_event_tracking_disabled(g: &mut Gpu, f: impl FnOnce(&mut Gpu)) {
        struct Reenable(Arc<CudaContext>);
        impl Drop for Reenable {
            fn drop(&mut self) {
                unsafe { self.0.enable_event_tracking() };
            }
        }
        let _guard = Reenable(g.ctx.clone());
        unsafe { g.ctx.disable_event_tracking() };
        f(g);
    }

    /// Capture `layer.forward_device_pooled_on` into a replayable CUDA graph on a fresh non-blocking
    /// stream, with the scratch pool and persistent in/out buffers all event-free on that stream.
    /// **Must run inside [`with_event_tracking_disabled`]** so the capture inserts no cross-stream
    /// waits. Returns the capture stream, the pool, the persistent input/output device buffers (whose
    /// pointers are baked into the graph — keep them alive and on this stream for replay), and the
    /// graph. `x` seeds the input. A warmup forward on the capture stream precedes capture (primes
    /// state; makes the NULL-stream-uploaded weights visible to the capture stream).
    #[allow(clippy::type_complexity)]
    fn capture_resident_layer(
        g: &Gpu,
        layer: &ResidentLayerF16,
        s: usize,
        d: usize,
        x: &[f32],
        cap_bytes: usize,
    ) -> (
        Arc<CudaStream>,
        crate::pool::DevicePool,
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<f32>,
        crate::graph::Graph,
    ) {
        let cap = g.ctx.new_stream().unwrap();
        let mut pool = crate::pool::DevicePool::new(cap.clone(), cap_bytes).unwrap();
        let x_d = cap.memcpy_stod(x).unwrap();
        let mut out_d = cap.alloc_zeros::<f32>(s * d).unwrap();
        // Weights were uploaded on the NULL stream; make them visible to the capture stream.
        g.stream.synchronize().unwrap();
        // Warmup once on the capture stream, then capture into a graph.
        pool.reset();
        layer.forward_device_pooled_on(&cap, &mut pool, &x_d, &mut out_d).unwrap();
        cap.synchronize().unwrap();
        pool.reset();
        let graph = crate::graph::Graph::capture(cap.clone(), || {
            layer.forward_device_pooled_on(&cap, &mut pool, &x_d, &mut out_d)
        })
        .unwrap();
        (cap, pool, x_d, out_d, graph)
    }

    /// **Identical-numerics gate for graph replay (the first law).** Capturing the pooled forward into
    /// a CUDA graph and replaying it with a single `cuGraphLaunch` must produce output **bit-for-bit**
    /// equal to the eager per-op forward — the graph changes *how* the same launches are issued, not
    /// *what* they compute. Capture runs on a dedicated non-blocking stream (the layer's default stream
    /// is the un-capturable NULL stream); the slab is poisoned 0xFF before replay (hostile dirty), and
    /// a second replay must match the first **bit-for-bit** (M12 determinism — replay is deterministic
    /// against stable baked-in pointers). Covers single-head, multi-head, and the GPT-2 layer.
    #[test]
    fn resident_layer_graphed_matches_eager() {
        with_gpu("resident_layer_graphed_matches_eager", |g| {
            with_event_tracking_disabled(g, |g| {
                let cases = [(64usize, 64usize, 256usize, 1usize), (512, 128, 512, 2), (512, 768, 3072, 12)];
                for (ci, &(s, d, dff, heads)) in cases.iter().enumerate() {
                    let (layer, x) = pool_layer_fixture(g, s, d, dff, heads, 0xA001 + ci as u64);

                    // eager reference on the default stream (same kernels, same weights, same input).
                    let x_d_ref = g.stream.memcpy_stod(&x).unwrap();
                    let ref_host =
                        g.stream.memcpy_dtov(&layer.forward_device(&x_d_ref).unwrap()).unwrap();

                    // capture the pooled forward into a graph on a dedicated stream.
                    let (cap, mut pool, _x_d, out_d, graph) =
                        capture_resident_layer(g, &layer, s, d, &x, 256 * 1024 * 1024);

                    // Poison the slab (ordered on `cap` before replay), then replay twice.
                    pool.poison(0xFF).unwrap();
                    graph.launch().unwrap();
                    cap.synchronize().unwrap();
                    let g1 = cap.memcpy_dtov(&out_d).unwrap();
                    graph.launch().unwrap();
                    cap.synchronize().unwrap();
                    let g2 = cap.memcpy_dtov(&out_d).unwrap();

                    assert_eq!(g1.len(), ref_host.len());
                    for i in 0..ref_host.len() {
                        assert_eq!(
                            g1[i].to_bits(),
                            ref_host[i].to_bits(),
                            "graphed != eager at {i} (S={s} D={d} Dff={dff} heads={heads})"
                        );
                        assert_eq!(
                            g1[i].to_bits(),
                            g2[i].to_bits(),
                            "graph replay non-deterministic at {i} (S={s} D={d} Dff={dff} heads={heads})"
                        );
                    }
                    eprintln!(
                        "graphed==eager bit-identical & replay deterministic: S={s} D={d} Dff={dff} \
                         heads={heads}; one cuGraphLaunch replays the whole resident layer (high-water {} KiB)",
                        pool.high_water_bytes() / 1024
                    );
                }
            });
        });
    }

    /// Best (lowest) per-iteration wall time of `run`, in seconds, synchronizing `sync_stream` to
    /// retire the work. Warms up to coax the boost clock, then takes the fastest of several timed
    /// rounds — the least-throttled measurement, mirroring `best_bw`'s reasoning for the ~7× laptop
    /// clock swing. Only **ratios** of two such numbers measured back-to-back in one process are
    /// reported (the honesty law).
    fn min_latency(sync_stream: &Arc<CudaStream>, mut run: impl FnMut()) -> f64 {
        const WARMUP: usize = 30;
        const ROUNDS: usize = 12;
        const ITERS: usize = 40;
        for _ in 0..WARMUP {
            run();
        }
        sync_stream.synchronize().unwrap();
        let mut best = f64::MAX;
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                run();
            }
            sync_stream.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() / ITERS as f64);
        }
        best
    }

    /// **M7 same-run latency — eager vs pooled vs graphed.** All three run the *identical* kernel
    /// sequence; they differ only in runtime overhead:
    /// - **eager**: a `cuMemAllocAsync`+`cuMemsetD8Async` per intermediate, a `cuMemFreeAsync` on drop,
    ///   and one `cuLaunchKernel` per kernel.
    /// - **pooled**: a host cursor bump per intermediate (no driver call, no zeroing) + per-kernel
    ///   launches — removes the alloc/free/zero traffic.
    /// - **graphed**: the whole captured layer replayed by a **single `cuGraphLaunch`** — also removes
    ///   the per-kernel launch overhead.
    ///
    /// Reported back-to-back at decode/small-batch + GPT-2 shapes, where this allocate/free/zero/launch
    /// overhead is the largest fraction of a tiny layer's wall time. eager runs on the default (NULL)
    /// stream; pooled/graphed on a dedicated capture stream (event tracking off so capture inserts no
    /// cross-stream waits — see `with_event_tracking_disabled`).
    #[test]
    #[ignore = "perf bench; needs a GPU. Run with --ignored --nocapture"]
    fn pool_graph_vs_unpooled() {
        with_gpu("pool_graph_vs_unpooled", |g| {
            eprintln!("device: {}", g.device_name());
            with_event_tracking_disabled(g, |g| {
                let cases = [(64usize, 64usize, 256usize, 1usize), (512, 128, 512, 2), (512, 768, 3072, 12)];
                let cap_bytes = 256 * 1024 * 1024;
                for &(s, d, dff, heads) in &cases {
                    let (layer, x) = pool_layer_fixture(g, s, d, dff, heads, 0x7001);

                    // eager on the default stream: per-op alloc_zeros + free-on-drop + N launches.
                    let x_d_g = g.stream.memcpy_stod(&x).unwrap();
                    let eager = min_latency(&g.stream, || {
                        let _ = layer.forward_device(&x_d_g).unwrap();
                    });

                    // Capture the layer into a graph on a dedicated stream (keeps its pool + buffers).
                    let (cap, pool, x_d_cap, _out_cap, graph) =
                        capture_resident_layer(g, &layer, s, d, &x, cap_bytes);

                    // pooled on the capture stream: bump arena, N individual launches (own pool/out).
                    let mut pool2 = crate::pool::DevicePool::new(cap.clone(), cap_bytes).unwrap();
                    let mut out2 = cap.alloc_zeros::<f32>(s * d).unwrap();
                    pool2.reset();
                    layer.forward_device_pooled_on(&cap, &mut pool2, &x_d_cap, &mut out2).unwrap();
                    let allocs_per_fwd = pool2.served();
                    let pooled = min_latency(&cap, || {
                        pool2.reset();
                        layer.forward_device_pooled_on(&cap, &mut pool2, &x_d_cap, &mut out2).unwrap();
                    });

                    // graphed on the capture stream: one cuGraphLaunch replays the whole layer.
                    let graphed = min_latency(&cap, || {
                        graph.launch().unwrap();
                    });

                    eprintln!(
                        "S={s:4} D={d:4} Dff={dff:5} h{heads:<2}: eager {:7.1} | pooled {:7.1} | graphed {:7.1} us \
                         → graphed {:.2}x vs eager, {:.2}x vs pooled  ({} pooled sub-allocs/fwd; high-water {} KiB)",
                        eager * 1e6,
                        pooled * 1e6,
                        graphed * 1e6,
                        eager / graphed,
                        pooled / graphed,
                        allocs_per_fwd,
                        pool.high_water_bytes() / 1024,
                    );
                }
            });
        });
    }

    // ============================================================================================
    // M7 runtime (Phase 7): multi-stream copy/compute overlap + pinned host memory.
    // ============================================================================================

    /// A double-buffered, two-stream pipeline that processes a batch of independent `[S,D]` inputs
    /// through the resident layer, overlapping **H2D(next) ‖ compute(cur) ‖ D2H(prev)**. Inputs/outputs
    /// stage through **pinned** host memory (so the copies are truly async on the copy stream while the
    /// compute stream runs), and ordering across the two streams is enforced by explicit CUDA events.
    /// Must be built/run inside [`with_event_tracking_disabled`] (so cudarc inserts no implicit
    /// cross-stream waits — this pipeline owns all the ordering).
    struct OverlapPipeline<'a> {
        layer: &'a ResidentLayerF16,
        cs: Arc<CudaStream>, // compute stream
        cp: Arc<CudaStream>, // copy stream (H2D + D2H)
        nb: usize,           // double-buffer slots
        x_d: Vec<cudarc::driver::CudaSlice<f32>>,
        out_d: Vec<cudarc::driver::CudaSlice<f32>>,
        pool: Vec<crate::pool::DevicePool>,
        h2d_done: Vec<cudarc::driver::CudaEvent>,
        comp_done: Vec<cudarc::driver::CudaEvent>,
        d2h_done: Vec<cudarc::driver::CudaEvent>,
        pin_in: Vec<crate::graph::PinnedBuf<f32>>,
        pin_out: Vec<crate::graph::PinnedBuf<f32>>,
        b: usize,
    }

    impl<'a> OverlapPipeline<'a> {
        fn new(
            g: &Gpu,
            layer: &'a ResidentLayerF16,
            inputs: &[Vec<f32>],
            s: usize,
            d: usize,
            cap_bytes: usize,
        ) -> Self {
            let nb = 2;
            let cs = g.ctx.new_stream().unwrap();
            let cp = g.ctx.new_stream().unwrap();
            let x_d = (0..nb).map(|_| cs.alloc_zeros::<f32>(s * d).unwrap()).collect();
            let out_d = (0..nb).map(|_| cs.alloc_zeros::<f32>(s * d).unwrap()).collect();
            let pool = (0..nb)
                .map(|_| crate::pool::DevicePool::new(cs.clone(), cap_bytes).unwrap())
                .collect();
            let ev = || g.ctx.new_event(None).unwrap();
            let h2d_done = (0..nb).map(|_| ev()).collect();
            let comp_done = (0..nb).map(|_| ev()).collect();
            let d2h_done = (0..nb).map(|_| ev()).collect();
            let mut pin_in = Vec::with_capacity(inputs.len());
            let mut pin_out = Vec::with_capacity(inputs.len());
            for inp in inputs {
                let mut pi = crate::graph::PinnedBuf::<f32>::alloc(&g.ctx, s * d).unwrap();
                pi.copy_from_slice(inp).unwrap();
                pin_in.push(pi);
                pin_out.push(crate::graph::PinnedBuf::<f32>::alloc(&g.ctx, s * d).unwrap());
            }
            // Weights uploaded on the NULL stream must be visible to the compute stream.
            g.stream.synchronize().unwrap();
            let _ = (s, d); // shapes are captured by the device buffers; not stored.
            Self {
                layer, cs, cp, nb, x_d, out_d, pool, h2d_done, comp_done, d2h_done, pin_in, pin_out,
                b: inputs.len(),
            }
        }

        /// **Serial baseline:** one stream, one buffer set — H2D, compute, D2H fully ordered with no
        /// overlap. Even with pinned memory, a single stream cannot overlap its own copies and compute.
        fn run_serial(&mut self) {
            for i in 0..self.b {
                self.cs.memcpy_htod(&*self.pin_in[i], &mut self.x_d[0]).unwrap();
                self.pool[0].reset();
                self.layer
                    .forward_device_pooled_on(&self.cs, &mut self.pool[0], &self.x_d[0], &mut self.out_d[0])
                    .unwrap();
                self.cs.memcpy_dtoh(&self.out_d[0], &mut *self.pin_out[i]).unwrap();
            }
            self.cs.synchronize().unwrap();
        }

        /// **Overlapped:** H2D(next) on the copy stream runs while the compute stream runs the current
        /// forward and D2H(prev) drains the previous result — double-buffered, with events guarding
        /// every read-after-write and write-after-read hazard across the two streams.
        fn run_overlapped(&mut self) {
            let nb = self.nb;
            for i in 0..self.b {
                let b = i % nb;
                // WAR on x_d[b]: don't overwrite until compute(i-nb), which read it, has finished.
                if i >= nb {
                    self.cp.wait(&self.comp_done[b]).unwrap();
                }
                self.cp.memcpy_htod(&*self.pin_in[i], &mut self.x_d[b]).unwrap();
                self.h2d_done[b].record(&self.cp).unwrap();
                // Compute waits for its input (RAW on x_d[b]); WAR on out_d[b]: don't overwrite until
                // D2H(i-nb), which read it, has finished.
                self.cs.wait(&self.h2d_done[b]).unwrap();
                if i >= nb {
                    self.cs.wait(&self.d2h_done[b]).unwrap();
                }
                self.pool[b].reset();
                self.layer
                    .forward_device_pooled_on(&self.cs, &mut self.pool[b], &self.x_d[b], &mut self.out_d[b])
                    .unwrap();
                self.comp_done[b].record(&self.cs).unwrap();
                // D2H waits for compute (RAW on out_d[b]).
                self.cp.wait(&self.comp_done[b]).unwrap();
                self.cp.memcpy_dtoh(&self.out_d[b], &mut *self.pin_out[i]).unwrap();
                self.d2h_done[b].record(&self.cp).unwrap();
            }
            self.cs.synchronize().unwrap();
            self.cp.synchronize().unwrap();
        }

        /// Collect the `B` pinned output buffers into host `Vec`s.
        fn collect(&self) -> Vec<Vec<f32>> {
            self.pin_out.iter().map(|p| p.to_vec().unwrap()).collect()
        }
    }

    /// Best (lowest) per-batch wall times of two self-synchronizing batch runs `a` and `b`, measured in
    /// **interleaved** rounds so both see the same drifting clock/contention state (this laptop GPU
    /// swings ~7× with memory-clock throttling, and the parallel sessions add contention — timing the
    /// two paths in separate phases makes the ratio meaningless). Returns `(best_a, best_b)` in seconds.
    fn best_batch_pair(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
        const WARMUP: usize = 4;
        const ROUNDS: usize = 16;
        for _ in 0..WARMUP {
            a();
            b();
        }
        let (mut best_a, mut best_b) = (f64::MAX, f64::MAX);
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            a();
            best_a = best_a.min(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            b();
            best_b = best_b.min(t1.elapsed().as_secs_f64());
        }
        (best_a, best_b)
    }

    /// **Identical-numerics gate for multi-stream overlap (the first law).** The overlapped pipeline
    /// must produce, for every input in the batch, output **bit-for-bit** equal to the serial
    /// single-stream pipeline — the overlap changes *when* copies/compute run and *on which stream*,
    /// never the math (M12 determinism holds). Also checks the serial path ties the eager
    /// `forward_device` reference. Covers single-head, multi-head, and the GPT-2 layer.
    #[test]
    fn multistream_overlap_matches_serial() {
        with_gpu("multistream_overlap_matches_serial", |g| {
            with_event_tracking_disabled(g, |g| {
                let cases = [(64usize, 64usize, 256usize, 1usize), (512, 128, 512, 2), (512, 768, 3072, 12)];
                for (ci, &(s, d, dff, heads)) in cases.iter().enumerate() {
                    let (layer, _x) = pool_layer_fixture(g, s, d, dff, heads, 0xB001 + ci as u64);
                    let mut rng = crate::diff::Rng::new(0x00C0_FFEE + ci as u64);
                    let bsz = 12usize;
                    let inputs: Vec<Vec<f32>> = (0..bsz).map(|_| rng.vec(s * d, -1.0, 1.0)).collect();
                    let cap_bytes = 64 * 1024 * 1024;

                    let mut serial = OverlapPipeline::new(g, &layer, &inputs, s, d, cap_bytes);
                    serial.run_serial();
                    let serial_out = serial.collect();

                    let mut over = OverlapPipeline::new(g, &layer, &inputs, s, d, cap_bytes);
                    over.run_overlapped();
                    let over_out = over.collect();

                    for i in 0..bsz {
                        for j in 0..s * d {
                            assert_eq!(
                                over_out[i][j].to_bits(),
                                serial_out[i][j].to_bits(),
                                "overlap != serial at input {i} elem {j} (S={s} D={d} Dff={dff} heads={heads})"
                            );
                        }
                    }
                    // Sanity: the serial path ties the established eager reference for input 0.
                    let x_d = g.stream.memcpy_stod(&inputs[0]).unwrap();
                    let eager = g.stream.memcpy_dtov(&layer.forward_device(&x_d).unwrap()).unwrap();
                    for j in 0..s * d {
                        assert_eq!(
                            serial_out[0][j].to_bits(),
                            eager[j].to_bits(),
                            "serial != eager at elem {j} (S={s} D={d})"
                        );
                    }
                    eprintln!(
                        "overlap==serial==eager bit-identical: S={s} D={d} Dff={dff} heads={heads}, B={bsz} \
                         ({} KiB I/O per item, pinned, 2-stream)",
                        s * d * 4 / 1024
                    );
                }
            });
        });
    }

    /// **M7 same-run throughput — multi-stream copy/compute overlap vs serial.** Both process `B`
    /// independent inputs; the overlapped path runs H2D/compute/D2H on two streams with pinned staging
    /// so copies can hide under compute. Measured with interleaved timing (shared clock state). Honest
    /// finding on this box: **~1.0× at both shapes** — the resident layer is strongly compute-bound, so
    /// even the GPT-2 shape's 1.5 MiB/item transfer is a small, largely overhead-bound fraction of the
    /// ~1.2 ms compute and there is little to hide. The mechanism is correct (gated bit-identical); the
    /// throughput lever for the *underutilized decode* regime is concurrent forwards, not copy overlap
    /// (see `concurrent_forwards_throughput`). Pinned + overlap matter most when transfers genuinely
    /// rival compute (large prompts / a slower link) — not realized here.
    #[test]
    #[ignore = "perf bench; needs a GPU. Run with --ignored --nocapture"]
    fn overlap_throughput() {
        with_gpu("overlap_throughput", |g| {
            eprintln!("device: {}", g.device_name());
            with_event_tracking_disabled(g, |g| {
                let cases = [(64usize, 64usize, 256usize, 1usize), (512, 768, 3072, 12)];
                let bsz = 16usize;
                for &(s, d, dff, heads) in &cases {
                    let (layer, _x) = pool_layer_fixture(g, s, d, dff, heads, 0x7777);
                    let mut rng = crate::diff::Rng::new(0x5EED);
                    let inputs: Vec<Vec<f32>> = (0..bsz).map(|_| rng.vec(s * d, -1.0, 1.0)).collect();
                    // Two pipelines so serial and overlapped can be timed interleaved (shared clock).
                    let mut pipe_s = OverlapPipeline::new(g, &layer, &inputs, s, d, 64 * 1024 * 1024);
                    let mut pipe_o = OverlapPipeline::new(g, &layer, &inputs, s, d, 64 * 1024 * 1024);

                    let (serial, over) =
                        best_batch_pair(|| pipe_s.run_serial(), || pipe_o.run_overlapped());
                    let bf = bsz as f64;
                    eprintln!(
                        "S={s:4} D={d:4} Dff={dff:5} h{heads:<2}: serial {:7.1} us/item | overlapped {:7.1} us/item \
                         → overlap {:.2}x  (B={bsz}, {} KiB I/O per item, pinned)",
                        serial / bf * 1e6,
                        over / bf * 1e6,
                        serial / over,
                        s * d * 4 / 1024
                    );
                }
            });
        });
    }

    /// Best (lowest) per-iteration wall time of `run` while synchronizing **all** of `streams` to
    /// retire the work (the multi-stream analogue of `min_latency`).
    fn min_latency_multi(streams: &[Arc<CudaStream>], mut run: impl FnMut()) -> f64 {
        const WARMUP: usize = 20;
        const ROUNDS: usize = 12;
        const ITERS: usize = 30;
        for _ in 0..WARMUP {
            run();
        }
        for st in streams {
            st.synchronize().unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                run();
            }
            for st in streams {
                st.synchronize().unwrap();
            }
            best = best.min(t0.elapsed().as_secs_f64() / ITERS as f64);
        }
        best
    }

    /// **M7 same-run throughput — concurrent forwards on K streams vs serial.** A single small (decode)
    /// layer launches far too few CTAs to fill the GPU, so the SMs sit idle. Running `K` independent
    /// requests on `K` streams lets the scheduler co-resident them and reclaim that idle capacity — the
    /// real multi-stream lever for small-batch *serving*. Compared back-to-back: `K` forwards serialized
    /// on one stream vs one forward on each of `K` streams (each its own event-free pool + buffers, so
    /// nothing cross-stream serializes them). The win shrinks as the layer grows to fill the GPU on its
    /// own (the GPT-2 shape already saturates).
    #[test]
    #[ignore = "perf bench; needs a GPU. Run with --ignored --nocapture"]
    fn concurrent_forwards_throughput() {
        with_gpu("concurrent_forwards_throughput", |g| {
            eprintln!("device: {}", g.device_name());
            with_event_tracking_disabled(g, |g| {
                let cases = [(64usize, 64usize, 256usize, 1usize), (512, 128, 512, 2), (512, 768, 3072, 12)];
                let kk = 4usize;
                for &(s, d, dff, heads) in &cases {
                    let (layer, x) = pool_layer_fixture(g, s, d, dff, heads, 0x9999);
                    let streams: Vec<Arc<CudaStream>> = (0..kk).map(|_| g.ctx.new_stream().unwrap()).collect();
                    let mut pools: Vec<crate::pool::DevicePool> = streams
                        .iter()
                        .map(|st| crate::pool::DevicePool::new(st.clone(), 64 * 1024 * 1024).unwrap())
                        .collect();
                    let x_ds: Vec<cudarc::driver::CudaSlice<f32>> =
                        streams.iter().map(|st| st.memcpy_stod(&x).unwrap()).collect();
                    let mut out_ds: Vec<cudarc::driver::CudaSlice<f32>> =
                        streams.iter().map(|st| st.alloc_zeros::<f32>(s * d).unwrap()).collect();
                    g.stream.synchronize().unwrap();

                    // serial: K forwards back-to-back on ONE stream.
                    let serial = min_latency(&streams[0], || {
                        for _ in 0..kk {
                            pools[0].reset();
                            layer
                                .forward_device_pooled_on(&streams[0], &mut pools[0], &x_ds[0], &mut out_ds[0])
                                .unwrap();
                        }
                    });
                    // concurrent: one forward on each of K streams, all retired together.
                    let sync_streams: Vec<Arc<CudaStream>> = streams.clone();
                    let concurrent = min_latency_multi(&sync_streams, || {
                        for k in 0..kk {
                            pools[k].reset();
                            layer
                                .forward_device_pooled_on(&streams[k], &mut pools[k], &x_ds[k], &mut out_ds[k])
                                .unwrap();
                        }
                    });

                    eprintln!(
                        "S={s:4} D={d:4} Dff={dff:5} h{heads:<2}: {kk} fwds serial {:7.1} us | concurrent {:7.1} us \
                         → concurrent {:.2}x throughput (K={kk} streams)",
                        serial * 1e6,
                        concurrent * 1e6,
                        serial / concurrent,
                    );
                }
            });
        });
    }

    // ============================================================================================
    // M7 / M13: the whole resident STACK captured into ONE graph — a whole-model forward replayed by
    // a single cuGraphLaunch. The pool is reset between layers (so its footprint is one layer, not N),
    // inter-layer activations ping-pong through two persistent buffers, and the N×(~13) launches fold
    // into one driver call — the decode/small-batch latency lever where launch overhead dominates.
    // ============================================================================================

    /// Run an `N`-layer resident stack pooled on `stream`: layer 0 reads `x_d`, each later layer reads
    /// the previous layer's output, every layer's intermediates come from `pool` (**reset between
    /// layers**, so the slab holds one layer's scratch, not N), and inter-layer activations ping-pong
    /// through `bufs[0]`/`bufs[1]` (persistent, outside the pool, so a reset never clobbers them). All
    /// launches land on `stream`. Returns the index in `bufs` holding the final output.
    fn forward_stack_pooled_on(
        layers: &[ResidentLayerF16],
        stream: &Arc<CudaStream>,
        pool: &mut crate::pool::DevicePool,
        x_d: &cudarc::driver::CudaSlice<f32>,
        bufs: &mut [cudarc::driver::CudaSlice<f32>; 2],
    ) -> Result<usize, DriverError> {
        let n = layers.len();
        pool.reset();
        layers[0].forward_device_pooled_on(stream, pool, x_d, &mut bufs[0])?;
        for i in 1..n {
            pool.reset();
            let (lo, hi) = bufs.split_at_mut(1);
            if (i - 1) % 2 == 0 {
                layers[i].forward_device_pooled_on(stream, pool, &lo[0], &mut hi[0])?;
            } else {
                layers[i].forward_device_pooled_on(stream, pool, &hi[0], &mut lo[0])?;
            }
        }
        Ok((n - 1) % 2)
    }

    /// Build `n` independent resident layers (distinct random weights each) + a random `[S,D]` input.
    fn multi_layer_fixture(
        g: &mut Gpu,
        n: usize,
        s: usize,
        d: usize,
        dff: usize,
        heads: usize,
        seed: u64,
    ) -> (Vec<ResidentLayerF16>, Vec<f32>) {
        let mut rng = crate::diff::Rng::new(seed);
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let wq = rng.vec(d * d, -0.08, 0.08);
            let wk = rng.vec(d * d, -0.08, 0.08);
            let wv = rng.vec(d * d, -0.08, 0.08);
            let wo = rng.vec(d * d, -0.08, 0.08);
            let w1 = rng.vec(dff * d, -0.05, 0.05);
            let w2 = rng.vec(d * dff, -0.05, 0.05);
            let w = TransformerWeights { wq: &wq, wk: &wk, wv: &wv, wo: &wo, w1: &w1, w2: &w2 };
            layers.push(ResidentLayerF16::new_mha(g, &w, s, d, dff, heads).unwrap());
        }
        let x = rng.vec(s * d, -1.0, 1.0);
        (layers, x)
    }

    /// Capture the whole pooled `N`-layer stack into one graph on a dedicated stream. Returns the
    /// stream, pool, persistent input + ping-pong buffers (all baked into the graph — keep alive), the
    /// index of the result buffer, and the graph. **Must run inside [`with_event_tracking_disabled`].**
    #[allow(clippy::type_complexity)]
    fn capture_resident_stack(
        g: &Gpu,
        layers: &[ResidentLayerF16],
        s: usize,
        d: usize,
        x: &[f32],
        cap_bytes: usize,
    ) -> (
        Arc<CudaStream>,
        crate::pool::DevicePool,
        cudarc::driver::CudaSlice<f32>,
        [cudarc::driver::CudaSlice<f32>; 2],
        usize,
        crate::graph::Graph,
    ) {
        let cap = g.ctx.new_stream().unwrap();
        let mut pool = crate::pool::DevicePool::new(cap.clone(), cap_bytes).unwrap();
        let x_d = cap.memcpy_stod(x).unwrap();
        let mut bufs = [cap.alloc_zeros::<f32>(s * d).unwrap(), cap.alloc_zeros::<f32>(s * d).unwrap()];
        g.stream.synchronize().unwrap();
        // Warmup, then capture.
        let _ = forward_stack_pooled_on(layers, &cap, &mut pool, &x_d, &mut bufs).unwrap();
        cap.synchronize().unwrap();
        let mut result_idx = 0usize;
        let graph = crate::graph::Graph::capture(cap.clone(), || {
            result_idx = forward_stack_pooled_on(layers, &cap, &mut pool, &x_d, &mut bufs)?;
            Ok(())
        })
        .unwrap();
        (cap, pool, x_d, bufs, result_idx, graph)
    }

    /// **Identical-numerics gate for the whole-model graph (the first law).** An `N`-layer resident
    /// stack captured into one graph and replayed by a single `cuGraphLaunch` must produce output
    /// **bit-for-bit** equal to the eager layer-by-layer `forward_device` chain, and be deterministic
    /// across two replays (M12). The slab is poisoned 0xFF first; the pool is reset between layers, so
    /// this also proves the inter-layer ping-pong + per-layer scratch reuse is correct under capture.
    #[test]
    fn resident_stack_graphed_matches_eager() {
        with_gpu("resident_stack_graphed_matches_eager", |g| {
            with_event_tracking_disabled(g, |g| {
                let (n, s, d, dff, heads) = (12usize, 64usize, 64usize, 256usize, 1usize);
                let (layers, x) = multi_layer_fixture(g, n, s, d, dff, heads, 0xD00D);

                // eager: layer-by-layer forward_device chain.
                let x_d_ref = g.stream.memcpy_stod(&x).unwrap();
                let mut cur = layers[0].forward_device(&x_d_ref).unwrap();
                for l in &layers[1..] {
                    cur = l.forward_device(&cur).unwrap();
                }
                let ref_host = g.stream.memcpy_dtov(&cur).unwrap();

                // graphed whole stack.
                let (cap, mut pool, _x_d, bufs, ridx, graph) =
                    capture_resident_stack(g, &layers, s, d, &x, 64 * 1024 * 1024);
                pool.poison(0xFF).unwrap();
                graph.launch().unwrap();
                cap.synchronize().unwrap();
                let g1 = cap.memcpy_dtov(&bufs[ridx]).unwrap();
                graph.launch().unwrap();
                cap.synchronize().unwrap();
                let g2 = cap.memcpy_dtov(&bufs[ridx]).unwrap();

                assert_eq!(g1.len(), ref_host.len());
                for i in 0..ref_host.len() {
                    assert_eq!(g1[i].to_bits(), ref_host[i].to_bits(), "stack graphed != eager at {i}");
                    assert_eq!(g1[i].to_bits(), g2[i].to_bits(), "stack replay non-deterministic at {i}");
                }
                eprintln!(
                    "{n}-layer stack graphed==eager bit-identical & deterministic (S={s} D={d} Dff={dff}); \
                     whole model = one cuGraphLaunch (high-water {} KiB, one layer's scratch)",
                    pool.high_water_bytes() / 1024
                );
            });
        });
    }

    /// **M7 decode latency — whole-model: eager `N`-layer chain vs one graph replay.** A token's forward
    /// through an `N`-layer resident stack is `N×(~13)` individual launches eagerly; captured, it is a
    /// **single `cuGraphLaunch`**. At the decode shape the launch overhead dominates, so folding the
    /// whole stack collapses per-token latency. Same-run ratio across depths.
    #[test]
    #[ignore = "perf bench; needs a GPU. Run with --ignored --nocapture"]
    fn decode_stack_latency() {
        with_gpu("decode_stack_latency", |g| {
            eprintln!("device: {}", g.device_name());
            with_event_tracking_disabled(g, |g| {
                let (s, d, dff, heads) = (64usize, 64usize, 256usize, 1usize);
                for &n in &[1usize, 6, 12] {
                    let (layers, x) = multi_layer_fixture(g, n, s, d, dff, heads, 0xBEEF);

                    // eager: per-op alloc + N*(~13) launches, layer by layer.
                    let x_d = g.stream.memcpy_stod(&x).unwrap();
                    let eager = min_latency(&g.stream, || {
                        let mut cur = layers[0].forward_device(&x_d).unwrap();
                        for l in &layers[1..] {
                            cur = l.forward_device(&cur).unwrap();
                        }
                    });

                    // graphed: the whole stack as one cuGraphLaunch.
                    let (cap, _pool, _x_d, _bufs, _ridx, graph) =
                        capture_resident_stack(g, &layers, s, d, &x, 64 * 1024 * 1024);
                    let graphed = min_latency(&cap, || {
                        graph.launch().unwrap();
                    });

                    eprintln!(
                        "depth N={n:2} (S={s} D={d} Dff={dff}, decode): eager {:8.1} us | graphed {:7.1} us \
                         → graphed {:.2}x  (~{} launches → 1 cuGraphLaunch)",
                        eager * 1e6,
                        graphed * 1e6,
                        eager / graphed,
                        n * 13
                    );
                }
            });
        });
    }

    // ===============================================================================================
    // int8 (W8A8) tensor-core GEMM (M3) — bit-exact gate + peer scoreboard.
    // ===============================================================================================

    /// Exact `i32` reference for `C = A·Bᵀ`: `A` is `[M,K]` **u8**, `B` is `[N,K]` **i8**, accumulation
    /// is **wrapping** `i32` (matching the tensor core's mod-2³² accumulate exactly — no rounding, no
    /// reassociation). This is the bit-exact oracle: the GPU must equal it lane-for-lane.
    fn ref_nt_int8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                for kk in 0..k {
                    let av = a[i * k + kk] as i32; // u8 → i32 (0..255)
                    let bv = b[j * k + kk] as i32; // i8 → i32 (-128..127)
                    acc = acc.wrapping_add(av.wrapping_mul(bv));
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// **M3 bit-exact gate.** int8 W8A8 GEMM (`u8`×`i8`→`i32`) must equal the wrapping-`i32` CPU
    /// reference **exactly** (not within tolerance) over the full output, at shapes hitting both the
    /// single-tile kernel (16×8 / non-`_mt`-divisible) and the fragment-reuse `_mt` kernel. Asymmetric
    /// data (a u8 ramp × an i8 ±ramp incl. negatives) so a lane/sign/transpose slip can't hide.
    #[test]
    fn int8_gemm_matches_reference() {
        with_gpu("int8_gemm", |g| {
            let mut rng = crate::diff::Rng::new(0x1278);
            // (m,k,n): the first two are `_mt`-divisible (m%32==0,n%32==0); the 16×8×32 and (48,32,40)
            // shapes fall to the single-tile kernel (n%32!=0 or m%32!=0) — both paths gated.
            for (m, k, n) in [
                (16usize, 32usize, 8usize),
                (48, 32, 40),
                (64, 64, 64),
                (128, 256, 96),
                (128, 128, 128), // multi-CTA, multi-K-step — exercises the smdb pipeline (swz, K%64==0)
                (128, 96, 128),  // K%32==0 but K%64!=0 — exercises the smdb dispatch's hand-placed fallback
            ] {
                // u8 activations in [0,255], i8 weights in [-128,127] — full range, deterministic.
                let a: Vec<u8> = (0..m * k)
                    .map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8)
                    .collect();
                let b: Vec<i8> = (0..n * k)
                    .map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8)
                    .collect();
                let got = gemm_nt_int8(g, &a, &b, m, k, n).unwrap();
                let want = ref_nt_int8(&a, &b, m, k, n);
                assert_eq!(
                    got, want,
                    "int8_gemm {m}x{k}x{n}: GPU output must equal the i32 reference bit-for-bit"
                );
                // SMEM-staged + cp.async path (when CTA-tile-divisible) — same bit-exact contract.
                use crate::ptx_int8::{INT8_BK, INT8_BM, INT8_BN};
                if m % INT8_BM == 0 && n % INT8_BN == 0 && k % INT8_BK == 0 {
                    let got_smdb = gemm_nt_int8_smdb(g, &a, &b, m, k, n).unwrap();
                    assert_eq!(
                        got_smdb, want,
                        "int8_gemm_smdb {m}x{k}x{n}: SMEM-staged output must equal the i32 reference"
                    );
                }
                let checksum = got.iter().map(|&x| x as i64).sum::<i64>();
                eprintln!("int8_gemm {m}x{k}x{n}: bit-exact ✓ (checksum {checksum})");
            }
        });
    }

    /// **Fused per-channel dequant epilogue gate.** `out[i,j] = f32(Σ u8·i8)·scale[j]` (the
    /// cuBLAS-can't-fuse path) must equal the CPU reference — the exact `i32` accumulate converted to
    /// f32 and multiplied by the per-column scale, both sides rounding identically (i32→f32 cvt.rn +
    /// one f32 mul). Tight tolerance (essentially exact: the only rounding is the shared final mul).
    #[test]
    fn int8_dequant_matches_reference() {
        with_gpu("int8_dequant", |g| {
            let mut rng = crate::diff::Rng::new(0x0DE9);
            // K=64/128/256 take the swz_deq path (K%64==0); K=96 is K%32==0 only → hand-placed deq fallback.
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 128, 128), (64, 256, 192), (64, 96, 128)] {
                let a: Vec<u8> = (0..m * k)
                    .map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8)
                    .collect();
                let b: Vec<i8> = (0..n * k)
                    .map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8)
                    .collect();
                // per-channel scales spanning a realistic quant range (~1/127 .. small).
                let scale: Vec<f32> = (0..n).map(|_| rng.f32_range(1e-3, 5e-2)).collect();
                let acc = ref_nt_int8(&a, &b, m, k, n);
                let want: Vec<f32> = (0..m * n).map(|t| acc[t] as f32 * scale[t % n]).collect();
                let got = gemm_nt_int8_smdb_dequant(g, &a, &b, &scale, m, k, n).unwrap();
                let st = crate::diff::assert_close(
                    &format!("int8_dequant {m}x{k}x{n}"),
                    &got,
                    &want,
                    1e-3,
                    1e-6,
                );
                eprintln!(
                    "int8_dequant {m}x{k}x{n}: fused i32→f32·scale[j] ✓ max_abs={:.2e} max_rel={:.2e}",
                    st.max_abs, st.max_rel
                );
            }
        });
    }

    /// **Fused-dequant cost bench.** Times the SMEM-staged int8 GEMM with the per-channel dequant
    /// epilogue (`int8_gemm_nt_smdb_deq`, f32 out) against the plain i32-output kernel
    /// (`int8_gemm_nt_smdb`), same GEMM. The dequant is computed **in registers at the C store**, so the
    /// two times are ~equal → Mercury gets the `i32→f32·scale[j]` dequant at ≈0 marginal cost. A cuBLAS
    /// int8 pipeline (raw `i32` out) must instead launch a *separate* dequant kernel that re-reads the
    /// whole `M×N` `i32` matrix from HBM and writes `M×N` f32 — a round-trip + launch this fusion
    /// removes. Same-run; clock-warmed + best_of. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int8_dequant_fusion`
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn int8_dequant_fusion() {
        with_gpu("int8_dequant_fusion", |g| {
            let mut rng = crate::diff::Rng::new(0x0DEF);
            // warm the clock
            for _ in 0..20 {
                let (a, b) = (vec![1u8; 2048 * 2048], vec![1i8; 2048 * 2048]);
                let _ = gemm_nt_int8_smdb(g, &a, &b, 2048, 2048, 2048);
            }
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = 2.0 * m as f64 * n as f64 * k as f64;
                let dims = (m as u32, n as u32, k as u32);
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                let scale: Vec<f32> = (0..n).map(|_| rng.f32_range(1e-3, 5e-2)).collect();
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let scale_d = g.stream.memcpy_stod(&scale).unwrap();
                use crate::ptx_int8::{INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N};
                let cfg = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, INT8_WARPS_M * INT8_WARPS_N);

                // plain i32-output kernel
                let f_i32 = g.function("int8_gemm_smdb", crate::ptx_int8::int8_gemm_smdb_ptx(), "int8_gemm_nt_smdb").unwrap();
                let mut ci_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                let s_i32 = best_of(4, || time_gemm_int8(g, &f_i32, cfg, dims, &a_d, &b_d, &mut ci_d, 50));

                // fused dequant kernel (f32 out + scale) — time its resident launches.
                let f_deq = g.function("int8_gemm_smdb_deq", crate::ptx_int8::int8_gemm_smdb_deq_ptx(), "int8_gemm_nt_smdb_deq").unwrap();
                let mut cf_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let time_deq = || {
                    let launch = |c: &mut cudarc::driver::CudaSlice<f32>| {
                        let (mm, nn, kk) = dims;
                        let mut bld = g.stream.launch_builder(&f_deq);
                        bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(c).arg(&scale_d);
                        unsafe { bld.launch(cfg).unwrap() };
                    };
                    launch(&mut cf_d);
                    g.stream.synchronize().unwrap();
                    let t0 = Instant::now();
                    for _ in 0..50 { launch(&mut cf_d); }
                    g.stream.synchronize().unwrap();
                    t0.elapsed().as_secs_f64() / 50.0
                };
                let s_deq = best_of(4, time_deq);

                eprintln!(
                    "{sz}³ int8: plain-i32 {:.0} GFLOP/s | fused-dequant(f32) {:.0} GFLOP/s | dequant overhead {:+.1}% (fused at ~0 cost; cuBLAS pays a separate i32→f32 kernel + HBM round-trip)",
                    flop / s_i32 / 1e9, flop / s_deq / 1e9, 100.0 * (s_deq - s_i32) / s_i32,
                );
            }
        });
    }

    /// Time `iters` resident launches of an int8 GEMM kernel `(M,N,K, A:u8, B:i8, C:i32)`; sec/iter.
    fn time_gemm_int8(
        g: &Gpu,
        f: &cudarc::driver::CudaFunction,
        cfg: LaunchConfig,
        dims: (u32, u32, u32),
        a_d: &cudarc::driver::CudaSlice<u8>,
        b_d: &cudarc::driver::CudaSlice<i8>,
        c_d: &mut cudarc::driver::CudaSlice<i32>,
        iters: usize,
    ) -> f64 {
        let (mm, nn, kk) = dims;
        let launch = |c_d: &mut cudarc::driver::CudaSlice<i32>| {
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

    /// Deterministic int8 peer-bench buffers with **activations in `[0,127]`** (the range where `u8` and
    /// `s8` reinterpretations coincide, so Mercury's `u8×s8`, the NVRTC peers, and cuBLAS's `s8×s8` all
    /// compute the *identical* matrix and cross-check bit-for-bit — see the cuBLAS signedness caveat in
    /// `baselines.rs`). Returns A both as `u8` (Mercury/naive/dp4a) and as `i8` (cuBLAS), plus i8 B.
    fn int8_inputs_a127(
        rng: &mut crate::diff::Rng,
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<u8>, Vec<i8>, Vec<i8>) {
        let a_u8: Vec<u8> = (0..m * k)
            .map(|_| (rng.f32_range(0.0, 128.0) as u32 & 0x7f) as u8)
            .collect();
        let a_i8: Vec<i8> = a_u8.iter().map(|&x| x as i8).collect(); // x<128 → same bits/value
        let b: Vec<i8> = (0..n * k)
            .map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8)
            .collect();
        (a_u8, a_i8, b)
    }

    /// **M3/M6: int8 (W8A8) tensor-core GEMM vs the Tier-A int8 CUDA-C peers**, same-run. Mercury's
    /// fragment-reuse `int8_gemm_nt_mt` (`mma.sync.m16n8k32.s32.u8.s8.s32`, A-fragment reused across the
    /// N tiles) vs **naive** int8 CUDA-C (one thread/output, scalar `(int)A·(int)B`) and **dp4a** int8
    /// CUDA-C (one thread/output, the 4-way `dp4a.u32.s32` byte dot-product — the strong hand-written
    /// SIMD-int8 baseline). The literal "beat the hand-written C int8 on the GPU" (M6) plus the strongest
    /// non-library int8 peer this box can compile (a cuBLASLt IMMA Tier-B peer is the follow-up; this
    /// nails the Tier-A wins and the dp4a bar first). Reports int8-MAC GFLOP/s (`2·M·N·K`) and Mercury
    /// × vs each peer.
    ///
    /// Correctness gates speed (the first law, here **bit-exact**): both peers are first cross-checked
    /// to **equal** the wrapping-`i32` CPU reference, and at every size all three outputs' checksums must
    /// agree exactly. Same-run only (the ~7× laptop clock swing): a clock warmup + `best_of`. Needs the
    /// NVRTC redist DLL on PATH (see `gemm_vs_peers`); skips (never fails) if absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int8_gemm_vs_peers`
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn int8_gemm_vs_peers() {
        use crate::baselines::{
            cublas_gemm_nt_int8, gemm_flop, nvrtc_dp4a_gemm_nt_int8, nvrtc_naive_gemm_nt_int8,
            peer_env_hint, peers_available, time_cublas_gemm_nt_int8, time_nvrtc_dp4a_gemm_nt_int8,
            time_nvrtc_naive_gemm_nt_int8,
        };
        use crate::ptx_int8::{INT8_TM, INT8_TN};
        with_gpu("int8_gemm_vs_peers", |g| {
            if !peers_available(g) {
                eprintln!("[skip] int8_gemm_vs_peers: NVRTC/cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());

            // --- Correctness first: all four paths must EQUAL the i32 oracle (bit-exact) at a small
            // shape. Activations in [0,127] (see int8_inputs_a127) so cuBLAS's s8×s8 == Mercury's u8×s8.
            let mut rng = crate::diff::Rng::new(0x1287);
            for (m, k, n) in [(256usize, 256usize, 256usize), (128, 320, 96)] {
                let (a_u8, a_i8, b) = int8_inputs_a127(&mut rng, m, k, n);
                let r = ref_nt_int8(&a_u8, &b, m, k, n);
                assert_eq!(gemm_nt_int8(g, &a_u8, &b, m, k, n).unwrap(), r, "Mercury int8 {m}x{k}x{n}");
                assert_eq!(nvrtc_naive_gemm_nt_int8(g, &a_u8, &b, m, k, n).unwrap(), r, "naive {m}x{k}x{n}");
                assert_eq!(nvrtc_dp4a_gemm_nt_int8(g, &a_u8, &b, m, k, n).unwrap(), r, "dp4a {m}x{k}x{n}");
                assert_eq!(cublas_gemm_nt_int8(g, &a_i8, &b, m, k, n).unwrap(), r, "cuBLAS {m}x{k}x{n}");
            }
            eprintln!("[gate] Mercury + naive + dp4a + cuBLAS int8 all equal the i32 oracle bit-for-bit ✓");

            // --- Clock warmup (cf. gemm_vs_peers): boost the clock before sampling so each size's ratio
            // is peak-vs-peak. Hammer cuBLAS int8 (the heaviest) until the clock settles. ---
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_int8(g, 2048, 2048, 2048, 20);
            }
            const ROUNDS: usize = 4;

            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k); // 2·M·N·K int8 MACs
                let dims = (m as u32, n as u32, k as u32);
                let (a_u8, a_i8, b) = int8_inputs_a127(&mut rng, m, k, n);

                // Mercury fragment-reuse _mt path (the fast int8 kernel).
                let a_d = g.stream.memcpy_stod(&a_u8).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                let f_mt = g
                    .function("int8_gemm_mt", crate::ptx_int8::int8_gemm_mt_ptx(), "int8_gemm_nt_mt")
                    .unwrap();
                let cfg_mt = LaunchConfig {
                    grid_dim: ((n / (8 * INT8_TN)) as u32, (m / (16 * INT8_TM)) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_mt = best_of(ROUNDS, || time_gemm_int8(g, &f_mt, cfg_mt, dims, &a_d, &b_d, &mut c_d, 50));

                // Mercury single-tile path (one 16×8 tile/warp — the pre-fragment-reuse baseline).
                let f_st = g
                    .function("int8_gemm", crate::ptx_int8::int8_gemm_ptx(), "int8_gemm_nt")
                    .unwrap();
                let cfg_st = LaunchConfig {
                    grid_dim: ((n / 8) as u32, (m / 16) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let s_st = best_of(ROUNDS, || time_gemm_int8(g, &f_st, cfg_st, dims, &a_d, &b_d, &mut c_d, 50));

                // Mercury SMEM-staged + cp.async double-buffered paths (the latency-hiding lever): 64×64
                // and the bigger-reuse 128×128 tile. Report the better as `_smdb`.
                use crate::ptx_int8::{
                    INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_WARPS_M, INT8_WARPS_M128,
                    INT8_WARPS_N, INT8_WARPS_N128,
                };
                let f_smdb = g
                    .function("int8_gemm_smdb", crate::ptx_int8::int8_gemm_smdb_ptx(), "int8_gemm_nt_smdb")
                    .unwrap();
                let cfg_smdb = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, INT8_WARPS_M * INT8_WARPS_N);
                let s_smdb64 = best_of(ROUNDS, || time_gemm_int8(g, &f_smdb, cfg_smdb, dims, &a_d, &b_d, &mut c_d, 50));
                let s_smdb128 = if m % INT8_BM128 == 0 && n % INT8_BN128 == 0 {
                    let f = g
                        .function("int8_gemm_smdb128", crate::ptx_int8::int8_gemm_smdb128_ptx(), "int8_gemm_nt_smdb128")
                        .unwrap();
                    let cfg = int8_smdb_cfg(m, n, INT8_BM128, INT8_BN128, INT8_WARPS_M128 * INT8_WARPS_N128);
                    best_of(ROUNDS, || time_gemm_int8(g, &f, cfg, dims, &a_d, &b_d, &mut c_d, 50))
                } else {
                    f64::INFINITY
                };
                let s_smdb = s_smdb64.min(s_smdb128);

                // Peers. Naive is slow → fewer iters; dp4a is the strong hand-written baseline; cuBLAS
                // int8 IMMA is the Tier-B gold standard (Mercury reported as % of it).
                let naive_iters = if sz >= 4096 { 3 } else { 10 };
                let s_naive = time_nvrtc_naive_gemm_nt_int8(g, m, k, n, naive_iters).unwrap();
                let s_dp4a = best_of(ROUNDS, || time_nvrtc_dp4a_gemm_nt_int8(g, m, k, n, 20).unwrap());
                let s_cub = best_of(ROUNDS, || time_cublas_gemm_nt_int8(g, m, k, n, 50).unwrap());

                // Checksum cross-check at this shape: all paths compute the same matrix.
                let csum = |v: &[i32]| v.iter().map(|&x| x as i64).sum::<i64>();
                let cs_mt = csum(&gemm_nt_int8(g, &a_u8, &b, m, k, n).unwrap());
                let cs_smdb = csum(&gemm_nt_int8_smdb(g, &a_u8, &b, m, k, n).unwrap());
                let cs_n = csum(&nvrtc_naive_gemm_nt_int8(g, &a_u8, &b, m, k, n).unwrap());
                let cs_d = csum(&nvrtc_dp4a_gemm_nt_int8(g, &a_u8, &b, m, k, n).unwrap());
                let cs_c = csum(&cublas_gemm_nt_int8(g, &a_i8, &b, m, k, n).unwrap());
                assert!(
                    cs_mt == cs_n && cs_mt == cs_d && cs_mt == cs_c && cs_mt == cs_smdb,
                    "{sz}³ int8 checksum disagreement: mt={cs_mt} smdb={cs_smdb} naive={cs_n} dp4a={cs_d} cublas={cs_c}"
                );

                let (g_mt, g_st, g_smdb64, g_smdb128, g_naive, g_dp4a, g_cub) = (
                    flop / s_mt, flop / s_st, flop / s_smdb64, flop / s_smdb128,
                    flop / s_naive, flop / s_dp4a, flop / s_cub,
                );
                let g_smdb = flop / s_smdb;
                eprintln!(
                    "\n{sz}³ int8 W8A8 GEMM (same-run, 2·M·N·K MAC-FLOP):\n  \
                     Mercury _smdb128 : {:>8.0} GFLOP/s  | {:>6.1}% of cuBLAS\n  \
                     Mercury _smdb64  : {:>8.0} GFLOP/s  | {:>6.1}% of cuBLAS\n  \
                     Mercury _smdb*   : {:>8.0} GFLOP/s  | {:>6.1}% of cuBLAS | {:>6.1}× vs naive | {:>5.2}× vs dp4a | {:>5.2}× vs _mt\n  \
                     Mercury _mt      : {:>8.0} GFLOP/s  | {:>6.1}% of cuBLAS | {:>6.1}× vs naive | {:>5.2}× vs dp4a\n  \
                     Mercury single   : {:>8.0} GFLOP/s  | {:>6.1}% of cuBLAS | {:>6.1}× vs naive | {:>5.2}× vs dp4a\n  \
                     cuBLAS int8 IMMA : {:>8.0} GFLOP/s  | Tier-B gold standard\n  \
                     dp4a CUDA-C      : {:>8.0} GFLOP/s  | strong hand-written int8 peer\n  \
                     naive CUDA-C     : {:>8.0} GFLOP/s  | Tier-A floor",
                    g_smdb128 / 1e9, 100.0 * g_smdb128 / g_cub,
                    g_smdb64 / 1e9, 100.0 * g_smdb64 / g_cub,
                    g_smdb / 1e9, 100.0 * g_smdb / g_cub, g_smdb / g_naive, g_smdb / g_dp4a, g_smdb / g_mt,
                    g_mt / 1e9, 100.0 * g_mt / g_cub, g_mt / g_naive, g_mt / g_dp4a,
                    g_st / 1e9, 100.0 * g_st / g_cub, g_st / g_naive, g_st / g_dp4a,
                    g_cub / 1e9,
                    g_dp4a / 1e9,
                    g_naive / 1e9,
                );
            }
        });
    }

    /// Launch one multi-stage int8 SMEM kernel and copy back the i32 result (gate/bench helper).
    fn launch_int8_smdb(
        g: &mut Gpu,
        ptx: &'static str,
        entry: &'static str,
        bm: usize,
        bn: usize,
        warps: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &[u8],
        b: &[i8],
    ) -> Vec<i32> {
        let f = g.function(entry, ptx, entry).unwrap();
        let cfg = int8_smdb_cfg(m, n, bm, bn, warps);
        let a_d = g.stream.memcpy_stod(a).unwrap();
        let b_d = g.stream.memcpy_stod(b).unwrap();
        let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
        let (mm, nn, kk) = (m as u32, n as u32, k as u32);
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
        unsafe { bld.launch(cfg).unwrap() };
        g.stream.memcpy_dtov(&c_d).unwrap()
    }

    /// **Multi-stage int8 gate (first law, bit-exact).** The deeper-pipeline `cp.async` variants
    /// (`int8_gemm_nt_smdb_s{3,4}`, 64×64 and 128×128) must reproduce the wrapping-`i32` CPU reference
    /// EXACTLY — identical `u8`×`i8`→`i32` mod-2³² contract as the 2-buffer kernel, only the pipeline
    /// depth differs. Full-range activations (`u8` 0..255) and weights (`i8`), several shapes including a
    /// non-128-divisible one (exercises the 64×64 tail) and the K-not-multiple-of-(stages·BK) cases.
    #[test]
    fn int8_smdb_ms_matches_reference() {
        use crate::ptx_int8::{
            int8_gemm_smdb128_s3_ptx, int8_gemm_smdb128_s4_ptx, int8_gemm_smdb_s3_ptx,
            int8_gemm_smdb_s4_ptx, INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_WARPS_M,
            INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
        };
        with_gpu("int8_smdb_ms", |g| {
            let mut rng = crate::diff::Rng::new(0x5A8D);
            let gen = |rng: &mut crate::diff::Rng, m: usize, k: usize, n: usize| {
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                (a, b)
            };
            // 64×64 variants (M%64==0, N%64==0, K%32==0); 160/96 exercise non-128 + short K.
            let w64 = INT8_WARPS_M * INT8_WARPS_N;
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 96, 192), (192, 160, 128)] {
                let (a, b) = gen(&mut rng, m, k, n);
                let r = ref_nt_int8(&a, &b, m, k, n);
                for (ptx, entry) in [
                    (int8_gemm_smdb_s3_ptx(), "int8_gemm_nt_smdb_s3"),
                    (int8_gemm_smdb_s4_ptx(), "int8_gemm_nt_smdb_s4"),
                ] {
                    assert_eq!(
                        launch_int8_smdb(g, ptx, entry, INT8_BM, INT8_BN, w64, m, k, n, &a, &b),
                        r,
                        "{entry} {m}x{k}x{n}"
                    );
                }
            }
            // 128×128 variants (M%128==0, N%128==0).
            let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
            for (m, k, n) in [(128usize, 128usize, 128usize), (256, 96, 128)] {
                let (a, b) = gen(&mut rng, m, k, n);
                let r = ref_nt_int8(&a, &b, m, k, n);
                for (ptx, entry) in [
                    (int8_gemm_smdb128_s3_ptx(), "int8_gemm_nt_smdb128_s3"),
                    (int8_gemm_smdb128_s4_ptx(), "int8_gemm_nt_smdb128_s4"),
                ] {
                    assert_eq!(
                        launch_int8_smdb(g, ptx, entry, INT8_BM128, INT8_BN128, w128, m, k, n, &a, &b),
                        r,
                        "{entry} {m}x{k}x{n}"
                    );
                }
            }
            eprintln!("[gate] int8 smdb multi-stage s3/s4 (64 & 128) bit-exact vs i32 oracle ✓");
        });
    }

    /// **`ldmatrix`+XOR-swizzle int8 gate (first law, bit-exact).** The conflict-free-SMEM variants
    /// (`int8_gemm_nt_smdb_swz` 64×64 and `int8_gemm_nt_smdb128_swz` 128×128, BK=64) must reproduce the
    /// wrapping-`i32` CPU reference EXACTLY: the XOR swizzle only reorders SMEM bytes and `ldmatrix`
    /// only changes *how* the A/B fragments are gathered — the `u8`×`i8`→`i32` mod-2³² arithmetic is
    /// identical to every other int8 kernel. Full-range `u8`/`i8`; K multiples of 64 (the swz slab is a
    /// full 64-wide K-step). A wrong fragment/swizzle layout fails this `assert_eq!` unambiguously — the
    /// exact oracle that lets the tricky ldmatrix derivation be validated independent of GPU contention.
    #[test]
    fn int8_smdb_swz_matches_reference() {
        use crate::ptx_int8::{
            int8_gemm_smdb128_swz_ptx, int8_gemm_smdb_swz_ptx, INT8_BM, INT8_BM128, INT8_BN,
            INT8_BN128, INT8_WARPS_M, INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
        };
        with_gpu("int8_smdb_swz", |g| {
            let mut rng = crate::diff::Rng::new(0x5111);
            let gen = |rng: &mut crate::diff::Rng, m: usize, k: usize, n: usize| {
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                (a, b)
            };
            // 64×64 swz (M%64==0, N%64==0, K%64==0); 192/128 exercise the multi-CTA tail.
            let w64 = INT8_WARPS_M * INT8_WARPS_N;
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 64, 192), (192, 128, 128)] {
                let (a, b) = gen(&mut rng, m, k, n);
                let r = ref_nt_int8(&a, &b, m, k, n);
                assert_eq!(
                    launch_int8_smdb(g, int8_gemm_smdb_swz_ptx(), "int8_gemm_nt_smdb_swz", INT8_BM, INT8_BN, w64, m, k, n, &a, &b),
                    r,
                    "smdb_swz 64x64 {m}x{k}x{n}"
                );
            }
            // 128×128 swz (M%128==0, N%128==0, K%64==0).
            let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
            for (m, k, n) in [(128usize, 128usize, 128usize), (256, 128, 256)] {
                let (a, b) = gen(&mut rng, m, k, n);
                let r = ref_nt_int8(&a, &b, m, k, n);
                assert_eq!(
                    launch_int8_smdb(g, int8_gemm_smdb128_swz_ptx(), "int8_gemm_nt_smdb128_swz", INT8_BM128, INT8_BN128, w128, m, k, n, &a, &b),
                    r,
                    "smdb128_swz {m}x{k}x{n}"
                );
            }
            eprintln!("[gate] int8 ldmatrix+swizzle smdb_swz (64 & 128) bit-exact vs i32 oracle ✓");
        });
    }

    /// **int8 split-K gate (first law: bit-exact AND deterministic).** The split-K swz kernel
    /// (`int8_gemm_nt_smdb_swz_sk`, launched with `gridDim.z = sk`) must reproduce the i32 reference
    /// EXACTLY for every split count: each of the `sk` CTAs computes a partial product over its K-range
    /// and folds it into the pre-zeroed C by `red.global.add.u32`. Integer add commutes, so the result is
    /// **order-independent** — bit-exact and deterministic (M12) no matter how the CTAs interleave, the
    /// property a float split-K reduction can't offer. K % (sk·64) == 0 (each split is whole BK=64 slabs).
    #[test]
    fn int8_smdb_swz_splitk_matches_reference() {
        use crate::ptx_int8::{int8_gemm_smdb_swz_splitk_ptx, INT8_BM, INT8_BN, INT8_WARPS_M, INT8_WARPS_N};
        with_gpu("int8_smdb_swz_sk", |g| {
            let mut rng = crate::diff::Rng::new(0x5C0DE);
            let entry = "int8_gemm_nt_smdb_swz_sk";
            let f = g.function(entry, int8_gemm_smdb_swz_splitk_ptx(), entry).unwrap();
            let warps = INT8_WARPS_M * INT8_WARPS_N;
            // (m,k,n,sk): K % (sk·64) == 0; sk≥2 K-splits each fold a partial via red.global.add.u32.
            for (m, k, n, sk) in [
                (64usize, 128usize, 64usize, 2usize),
                (64, 256, 128, 4),
                (128, 512, 64, 8),
                (64, 256, 192, 2),
                (192, 128, 128, 2),
            ] {
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                let want = ref_nt_int8(&a, &b, m, k, n);
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap(); // split-K requires C pre-zeroed
                let mut cfg = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, warps);
                cfg.grid_dim.2 = sk as u32; // gridDim.z = number of K-splits
                let (mm, nn, kk) = (m as u32, n as u32, k as u32);
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
                unsafe { bld.launch(cfg).unwrap() };
                let got = g.stream.memcpy_dtov(&c_d).unwrap();
                assert_eq!(got, want, "smdb_swz_sk {m}x{k}x{n} sk={sk}");
            }
            // gate the public launcher wrapper (gemm_nt_int8_splitk) end-to-end too.
            {
                let (m, k, n, sk) = (128usize, 256usize, 128usize, 4usize);
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                let want = ref_nt_int8(&a, &b, m, k, n);
                assert_eq!(gemm_nt_int8_splitk(g, &a, &b, m, k, n, sk).unwrap(), want, "gemm_nt_int8_splitk {m}x{k}x{n} sk={sk}");
            }
            eprintln!("[gate] int8 split-K swz (red.global.add.u32, sk=2/4/8) bit-exact vs i32 oracle ✓");
        });
    }

    /// **int8 multi-stage `cp.async` depth sweep (M3 lever)** — same-run vs cuBLAS IMMA at 1024/2048/4096.
    /// Times the 2-buffer double-buffer (`_smdb`, stages=2) against the 3- and 4-stage rings
    /// (`_smdb_s{3,4}`) for both the 64×64 and 128×128 tiles, picks the per-size winner, reports % of
    /// cuBLAS. Bit-exact checksum cross-check first (the first law). Needs the cuBLAS redist DLL on PATH;
    /// skips (never fails) if absent. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int8_smdb_sweep`
    #[test]
    #[ignore = "throughput bench; needs CUDA redist DLLs on PATH; run explicitly"]
    fn int8_smdb_sweep() {
        use crate::baselines::{gemm_flop, peer_env_hint, peers_available, time_cublas_gemm_nt_int8};
        use crate::ptx_int8::{
            int8_gemm_smdb128_ptx, int8_gemm_smdb128_s3_ptx, int8_gemm_smdb128_s4_ptx,
            int8_gemm_smdb128_swz_ptx, int8_gemm_smdb_ptx, int8_gemm_smdb_s3_ptx,
            int8_gemm_smdb_s4_ptx, int8_gemm_smdb_swz_ptx, INT8_BM, INT8_BM128, INT8_BN, INT8_BN128,
            INT8_WARPS_M, INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
        };
        with_gpu("int8_smdb_sweep", |g| {
            if !peers_available(g) {
                eprintln!("[skip] int8_smdb_sweep: cuBLAS not loadable.\n{}", peer_env_hint());
                return;
            }
            eprintln!("device: {}", g.device_name());
            let w64 = INT8_WARPS_M * INT8_WARPS_N;
            let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
            // (label, ptx, entry, bm, bn, warps, needs_128_divisible). The `_swz` rows are the
            // ldmatrix+XOR-swizzle conflict-free-SMEM candidates (BK=64); same-run vs the s2/s3/s4
            // hand-placed depths and cuBLAS picks the per-size winner.
            let variants: [(&str, &'static str, &'static str, usize, usize, usize, bool); 8] = [
                ("smdb64_s2", int8_gemm_smdb_ptx(), "int8_gemm_nt_smdb", INT8_BM, INT8_BN, w64, false),
                ("smdb64_s3", int8_gemm_smdb_s3_ptx(), "int8_gemm_nt_smdb_s3", INT8_BM, INT8_BN, w64, false),
                ("smdb64_s4", int8_gemm_smdb_s4_ptx(), "int8_gemm_nt_smdb_s4", INT8_BM, INT8_BN, w64, false),
                ("smdb64_swz", int8_gemm_smdb_swz_ptx(), "int8_gemm_nt_smdb_swz", INT8_BM, INT8_BN, w64, false),
                ("smdb128_s2", int8_gemm_smdb128_ptx(), "int8_gemm_nt_smdb128", INT8_BM128, INT8_BN128, w128, true),
                ("smdb128_s3", int8_gemm_smdb128_s3_ptx(), "int8_gemm_nt_smdb128_s3", INT8_BM128, INT8_BN128, w128, true),
                ("smdb128_s4", int8_gemm_smdb128_s4_ptx(), "int8_gemm_nt_smdb128_s4", INT8_BM128, INT8_BN128, w128, true),
                ("smdb128_swz", int8_gemm_smdb128_swz_ptx(), "int8_gemm_nt_smdb128_swz", INT8_BM128, INT8_BN128, w128, true),
            ];
            let mut rng = crate::diff::Rng::new(0x5A8D5);
            for _ in 0..40 {
                let _ = time_cublas_gemm_nt_int8(g, 2048, 2048, 2048, 20);
            }
            const ROUNDS: usize = 6;
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let (a_u8, _a_i8, b) = int8_inputs_a127(&mut rng, m, k, n);
                let cs_ref: i64 = ref_nt_int8(&a_u8, &b, m, k, n).iter().map(|&x| x as i64).sum();
                let a_d = g.stream.memcpy_stod(&a_u8).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                eprintln!("\n{sz}³ int8 smdb depth sweep (same-run, cuBLAS-adjacent best-pair):");
                let mut best: Option<(&str, f64)> = None;
                for (label, ptx, entry, bm, bn, warps, needs128) in variants {
                    if needs128 && (m % 128 != 0 || n % 128 != 0) {
                        continue;
                    }
                    let f = g.function(entry, ptx, entry).unwrap();
                    let cfg = int8_smdb_cfg(m, n, bm, bn, warps);
                    // bit-exact checksum cross-check at this size (first law).
                    let cs: i64 = {
                        let mut cc = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                        let mut bld = g.stream.launch_builder(&f);
                        bld.arg(&dims.0).arg(&dims.1).arg(&dims.2).arg(&a_d).arg(&b_d).arg(&mut cc);
                        unsafe { bld.launch(cfg).unwrap() };
                        g.stream.memcpy_dtov(&cc).unwrap().iter().map(|&x| x as i64).sum()
                    };
                    assert_eq!(cs, cs_ref, "{label} {sz}³ checksum");
                    // Interleave the variant and cuBLAS round-by-round so each %-of-cuBLAS uses a
                    // *contemporaneous* cuBLAS baseline (the clock/contention swing on this shared mobile
                    // GPU makes a once-per-size baseline unreliable — cf. `gemm_pipe_sweep`).
                    let (mut s_v, mut s_cub) = (f64::INFINITY, f64::INFINITY);
                    for _ in 0..ROUNDS {
                        s_v = s_v.min(time_gemm_int8(g, &f, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                        s_cub = s_cub.min(time_cublas_gemm_nt_int8(g, m, k, n, 50).unwrap());
                    }
                    let (gf, g_cub) = (flop / s_v, flop / s_cub);
                    let pct = 100.0 * gf / g_cub;
                    eprintln!("  {label:<11}: {:>8.0} GFLOP/s | {:>5.1}% of cuBLAS ({:>6.0})", gf / 1e9, pct, g_cub / 1e9);
                    if best.map_or(true, |(_, p)| pct > p) {
                        best = Some((label, pct));
                    }
                }
                if let Some((name, pct)) = best {
                    eprintln!("  → best @{sz}³: {name} at {pct:.1}% of cuBLAS");
                }
            }
        });
    }

    /// **int8 `ldmatrix`+swizzle vs hand-placed, same-run internal A/B (M3 lever, contention-robust).**
    /// Times the conflict-free-SMEM `_swz` kernels against the hand-placed `_smdb` baseline back-to-back
    /// under one clock state and reports the **Mercury-internal** swz/handplaced ratio — which stays
    /// honest even when the cuBLAS baseline is contention-corrupted (the documented measurement caveat is
    /// specifically about the *cuBLAS* swing; an A/B of two adjacent same-family kernels cancels the
    /// shared clock). Answers the real M3 question: does the `ldmatrix.x4`/`.x2` gather from swizzled
    /// SMEM actually beat the 4+2 bank-conflicted `ld.shared.b32`? Bit-exact checksum cross-check first
    /// (first law). **No cuBLAS dependency** — runs on any CUDA device. Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int8_swz_vs_handplaced`
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn int8_swz_vs_handplaced() {
        use crate::baselines::gemm_flop;
        use crate::ptx_int8::{
            int8_gemm_smdb128_ptx, int8_gemm_smdb128_swz_ptx, int8_gemm_smdb_ptx,
            int8_gemm_smdb_swz_ptx, INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_WARPS_M,
            INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
        };
        with_gpu("int8_swz_vs_handplaced", |g| {
            eprintln!("device: {}", g.device_name());
            let w64 = INT8_WARPS_M * INT8_WARPS_N;
            let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
            let mut rng = crate::diff::Rng::new(0x5217);
            const ROUNDS: usize = 8;
            // (label, ptx, entry, bm, bn, warps); rows 0/1 are the 64×64 pair, 2/3 the 128×128 pair.
            let pairs: [(&str, &'static str, &'static str, usize, usize, usize); 4] = [
                ("smdb64    ", int8_gemm_smdb_ptx(), "int8_gemm_nt_smdb", INT8_BM, INT8_BN, w64),
                ("smdb64_swz", int8_gemm_smdb_swz_ptx(), "int8_gemm_nt_smdb_swz", INT8_BM, INT8_BN, w64),
                ("smdb128   ", int8_gemm_smdb128_ptx(), "int8_gemm_nt_smdb128", INT8_BM128, INT8_BN128, w128),
                ("smdb128swz", int8_gemm_smdb128_swz_ptx(), "int8_gemm_nt_smdb128_swz", INT8_BM128, INT8_BN128, w128),
            ];
            for sz in [1024usize, 2048, 4096] {
                let (m, k, n) = (sz, sz, sz);
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let (a_u8, _a_i8, b) = int8_inputs_a127(&mut rng, m, k, n);
                let cs_ref: i64 = ref_nt_int8(&a_u8, &b, m, k, n).iter().map(|&x| x as i64).sum();
                let a_d = g.stream.memcpy_stod(&a_u8).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                eprintln!("\n{sz}³ int8 swz-vs-handplaced (same-run, internal ratio):");
                let mut gf = [f64::NAN; 4];
                for (i, (label, ptx, entry, bm, bn, warps)) in pairs.iter().enumerate() {
                    if m % bm != 0 || n % bn != 0 {
                        continue;
                    }
                    let f = g.function(entry, ptx, entry).unwrap();
                    let cfg = int8_smdb_cfg(m, n, *bm, *bn, *warps);
                    // first law: bit-exact checksum cross-check at this size before timing.
                    let cs: i64 = {
                        let mut cc = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                        let mut bld = g.stream.launch_builder(&f);
                        bld.arg(&dims.0).arg(&dims.1).arg(&dims.2).arg(&a_d).arg(&b_d).arg(&mut cc);
                        unsafe { bld.launch(cfg).unwrap() };
                        g.stream.memcpy_dtov(&cc).unwrap().iter().map(|&x| x as i64).sum()
                    };
                    assert_eq!(cs, cs_ref, "{label} {sz}³ checksum");
                    let mut s = f64::INFINITY;
                    for _ in 0..ROUNDS {
                        s = s.min(time_gemm_int8(g, &f, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                    }
                    gf[i] = flop / s;
                    eprintln!("  {label}: {:>8.0} GFLOP/s", gf[i] / 1e9);
                }
                if gf[0].is_finite() && gf[1].is_finite() {
                    eprintln!("  → 64×64   swz/handplaced = {:.3}×", gf[1] / gf[0]);
                }
                if gf[2].is_finite() && gf[3].is_finite() {
                    eprintln!("  → 128×128 swz/handplaced = {:.3}×", gf[3] / gf[2]);
                }
            }
        });
    }

    /// **Static-shape int8 gate (M1, first law).** `gemm_nt_int8_static` bakes M/N/K into the swz kernel;
    /// only the dim *constants* change vs the dynamic kernel — the u8×i8→i32 mod-2³² arithmetic is
    /// identical — so it must match BOTH the i32 oracle and the dynamic `gemm_nt_int8_smdb` EXACTLY. The
    /// 64×64 entry is gated through the public launcher; the 128×128 entry (the dispatch only picks it at
    /// M,N≥4096, too big for a quick gate) is gated directly at a small shape. K multiples of 64.
    #[test]
    fn int8_static_matches_reference() {
        use crate::ptx_int8::{
            int8_gemm_smdb_swz_static_entry, int8_gemm_smdb_swz_static_ptx, INT8_BM128, INT8_BN128,
            INT8_WARPS_M128, INT8_WARPS_N128,
        };
        with_gpu("int8_static", |g| {
            let mut rng = crate::diff::Rng::new(0x5777);
            let gen = |rng: &mut crate::diff::Rng, m: usize, k: usize, n: usize| {
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                (a, b)
            };
            // 64×64 static via the public launcher: == the i32 oracle AND the dynamic kernel, exactly.
            for (m, k, n) in [(64usize, 64usize, 64usize), (128, 128, 192), (192, 64, 128), (256, 192, 256)] {
                let (a, b) = gen(&mut rng, m, k, n);
                let r = ref_nt_int8(&a, &b, m, k, n);
                let cs = gemm_nt_int8_static(g, &a, &b, m, k, n).unwrap();
                assert_eq!(cs, r, "int8_static vs oracle {m}x{k}x{n}");
                assert_eq!(cs, gemm_nt_int8_smdb(g, &a, &b, m, k, n).unwrap(), "int8_static vs dynamic {m}x{k}x{n}");
            }
            // 128×128 static entry directly (small shape; the dispatch only selects it at ≥4096²).
            let w128 = INT8_WARPS_M128 * INT8_WARPS_N128;
            for (m, k, n) in [(128usize, 64usize, 128usize), (256, 128, 256)] {
                let (a, b) = gen(&mut rng, m, k, n);
                let r = ref_nt_int8(&a, &b, m, k, n);
                let ptx = int8_gemm_smdb_swz_static_ptx(m, n, k, true);
                let module = g.ctx.load_module(ptx.as_str().into()).unwrap();
                let f = module.load_function(int8_gemm_smdb_swz_static_entry(true)).unwrap();
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                let (mm, nn, kk) = (m as u32, n as u32, k as u32);
                let cfg = int8_smdb_cfg(m, n, INT8_BM128, INT8_BN128, w128);
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
                unsafe { bld.launch(cfg).unwrap() };
                assert_eq!(g.stream.memcpy_dtov(&c_d).unwrap(), r, "int8_static128 {m}x{k}x{n}");
            }
            eprintln!("[gate] int8 static-shape (64 & 128) bit-exact vs i32 oracle AND the dynamic swz kernel ✓");
        });
    }

    /// **Static-shape int8 A/B (M1 lever, contention-robust internal ratio).** Times the static-shape
    /// kernel (M/N/K baked → ptxas constant-folds strides, knows the K trip count) against the dynamic
    /// swz kernel, same-run. CRITICAL (the int4 confound): load BOTH via the *same* raw `load_module`
    /// path — comparing a raw-loaded kernel to a `g.function`-cached one confounds the ptxas opt level
    /// with the specialization. Bit-exact checksum cross-check before timing. Expect ≥1× (int4's static
    /// lever measured 1.05–1.30×); the win is the strength-reduced strides on the power-of-two dims.
    #[test]
    #[ignore = "throughput A/B; run explicitly (GPU; no DLLs needed)"]
    fn int8_static_vs_dynamic_ab() {
        use crate::baselines::gemm_flop;
        use crate::ptx_int8::{
            int8_gemm_smdb128_swz_ptx, int8_gemm_smdb_swz_ptx, int8_gemm_smdb_swz_static_entry,
            int8_gemm_smdb_swz_static_ptx, INT8_BM, INT8_BM128, INT8_BN, INT8_BN128, INT8_WARPS_M,
            INT8_WARPS_M128, INT8_WARPS_N, INT8_WARPS_N128,
        };
        with_gpu("int8_static_ab", |g| {
            eprintln!("device: {}", g.device_name());
            let mut rng = crate::diff::Rng::new(0x5817);
            const ROUNDS: usize = 8;
            // Square compute shapes + a decode-ish thin-M shape. K%64; the 4096² shape exercises the 128 tile.
            for (m, k, n) in [(1024usize, 1024usize, 1024usize), (2048, 2048, 2048), (4096, 4096, 4096), (256, 4096, 4096)] {
                let use_128 = m >= 4096 && n >= 4096;
                let (bm, bn, warps, dyn_ptx, dyn_entry) = if use_128 {
                    (INT8_BM128, INT8_BN128, INT8_WARPS_M128 * INT8_WARPS_N128, int8_gemm_smdb128_swz_ptx(), "int8_gemm_nt_smdb128_swz")
                } else {
                    (INT8_BM, INT8_BN, INT8_WARPS_M * INT8_WARPS_N, int8_gemm_smdb_swz_ptx(), "int8_gemm_nt_smdb_swz")
                };
                if m % bm != 0 || n % bn != 0 {
                    continue;
                }
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let (a_u8, _a_i8, b) = int8_inputs_a127(&mut rng, m, k, n);
                let a_d = g.stream.memcpy_stod(&a_u8).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                let cfg = int8_smdb_cfg(m, n, bm, bn, warps);
                // bit-exact cross-check before timing (first law).
                let csum = |v: &[i32]| v.iter().map(|&x| x as i64).sum::<i64>();
                assert_eq!(
                    csum(&gemm_nt_int8_static(g, &a_u8, &b, m, k, n).unwrap()),
                    csum(&gemm_nt_int8_smdb(g, &a_u8, &b, m, k, n).unwrap()),
                    "{m}x{k}x{n} static/dynamic checksum"
                );
                // Both raw-loaded (same JIT path) → the ratio isolates the baked-constants effect.
                let ps = int8_gemm_smdb_swz_static_ptx(m, n, k, use_128);
                let mod_s = g.ctx.load_module(ps.as_str().into()).unwrap();
                let f_s = mod_s.load_function(int8_gemm_smdb_swz_static_entry(use_128)).unwrap();
                let mod_d = g.ctx.load_module(dyn_ptx.into()).unwrap();
                let f_d = mod_d.load_function(dyn_entry).unwrap();
                let s_static = best_of(ROUNDS, || time_gemm_int8(g, &f_s, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                let s_dyn = best_of(ROUNDS, || time_gemm_int8(g, &f_d, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                eprintln!(
                    "{m}x{k}x{n} int8 static-vs-dynamic (same raw-JIT load, M/N/K baked): dynamic {:>7.0} GFLOP/s | static {:>7.0} = {:.3}× speedup",
                    flop / s_dyn / 1e9,
                    flop / s_static / 1e9,
                    s_dyn / s_static,
                );
            }
        });
    }

    /// **Static-shape fp16 gate (M1, first law).** `gemm_nt_f16_static` bakes M/N/K into the SMEM-staged
    /// `wmma_nt_f16_sm` kernel; only the dim *constants* change vs the dynamic kernel and the f32
    /// accumulation order is unchanged, so it must match the dynamic `gemm_nt_f16_sm`/`_sm128` EXACTLY
    /// (bit-for-bit). 64×64 via the public launcher; 128×128 entry directly (the dispatch only picks it
    /// at M,N≥4096). K multiples of 16.
    #[test]
    fn f16_static_matches_reference() {
        use crate::ptx_wmma::{
            wmma_f16_sm_static_entry, wmma_f16_sm_static_ptx, SM128_BM, SM128_BN, SM128_THREADS,
        };
        use half::f16;
        with_gpu("f16_static", |g| {
            let mut rng = crate::diff::Rng::new(0x6111);
            // 64×64 static via the public launcher == the dynamic _sm kernel, bit-exact (same f16 codegen).
            for (m, k, n) in [(64usize, 16usize, 64usize), (128, 64, 192), (192, 32, 128), (256, 80, 256)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let cs = gemm_nt_f16_static(g, &a, &b, m, k, n).unwrap();
                let cd = gemm_nt_f16_sm(g, &a, &b, m, k, n).unwrap();
                assert_eq!(cs, cd, "f16_static vs dynamic _sm {m}x{k}x{n}");
            }
            // 128×128 static entry directly (small shape; the dispatch only selects it at ≥4096²).
            for (m, k, n) in [(128usize, 16usize, 128usize), (256, 64, 256)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let cd = gemm_nt_f16_sm128(g, &a, &b, m, k, n).unwrap();
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let ptx = wmma_f16_sm_static_ptx(m, n, k, true);
                let module = g.ctx.load_module(ptx.as_str().into()).unwrap();
                let f = module.load_function(wmma_f16_sm_static_entry(true)).unwrap();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let (mm, nn, kk) = (m as u32, n as u32, k as u32);
                let cfg = LaunchConfig {
                    grid_dim: ((n / SM128_BN) as u32, (m / SM128_BM) as u32, 1),
                    block_dim: (SM128_THREADS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut bld = g.stream.launch_builder(&f);
                bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
                unsafe { bld.launch(cfg).unwrap() };
                assert_eq!(g.stream.memcpy_dtov(&c_d).unwrap(), cd, "f16_static128 vs dynamic _sm128 {m}x{k}x{n}");
            }
            eprintln!("[gate] fp16 static-shape (64 & 128) bit-exact vs the dynamic _sm kernel ✓");
        });
    }

    /// **Static-shape fp16 A/B (M1 lever, contention-robust internal ratio).** Times the static-shape
    /// `_sm` kernel (M/N/K baked) against the dynamic one, same-run, BOTH raw-loaded the same way (the
    /// int4 confound). Expect ≥1×, biggest in the latency-bound thin-M regime (the int8 twin measured
    /// up to 1.37× there). f16 GEMM is deterministic so a checksum cross-check guards correctness first.
    #[test]
    #[ignore = "throughput A/B; run explicitly (GPU; no DLLs needed)"]
    fn f16_static_vs_dynamic_ab() {
        use crate::baselines::gemm_flop;
        use crate::ptx_wmma::{
            wmma_f16_ptx, wmma_f16_sm_static_entry, wmma_f16_sm_static_ptx, SM128_BM, SM128_BN,
            SM128_THREADS, SM_BM, SM_BN, SM_THREADS,
        };
        use half::f16;
        with_gpu("f16_static_ab", |g| {
            eprintln!("device: {}", g.device_name());
            let mut rng = crate::diff::Rng::new(0x6817);
            const ROUNDS: usize = 8;
            for (m, k, n) in [(1024usize, 1024usize, 1024usize), (2048, 2048, 2048), (4096, 4096, 4096), (256, 4096, 4096)] {
                let use_128 = m >= 4096 && n >= 4096;
                let (bm, bn, threads, dyn_entry) = if use_128 {
                    (SM128_BM, SM128_BN, SM128_THREADS, "wmma_nt_f16_sm128")
                } else {
                    (SM_BM, SM_BN, SM_THREADS, "wmma_nt_f16_sm")
                };
                if m % bm != 0 || n % bn != 0 {
                    continue;
                }
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
                let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
                let a_d = g.stream.memcpy_stod(&a16).unwrap();
                let b_d = g.stream.memcpy_stod(&b16).unwrap();
                let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n]).unwrap();
                let cfg = LaunchConfig {
                    grid_dim: ((n / bn) as u32, (m / bm) as u32, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // bit-exact checksum cross-check (static == dynamic) before timing.
                let csum = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
                assert!(
                    (csum(&gemm_nt_f16_static(g, &a, &b, m, k, n).unwrap())
                        - csum(&gemm_nt_f16_sm(g, &a, &b, m, k, n).unwrap()))
                    .abs()
                        < 1.0,
                    "{m}x{k}x{n} f16 static/dynamic checksum"
                );
                // Both raw-loaded (same JIT path) → the ratio isolates the baked-constants effect.
                let ps = wmma_f16_sm_static_ptx(m, n, k, use_128);
                let mod_s = g.ctx.load_module(ps.as_str().into()).unwrap();
                let f_s = mod_s.load_function(wmma_f16_sm_static_entry(use_128)).unwrap();
                let mod_d = g.ctx.load_module(wmma_f16_ptx().into()).unwrap();
                let f_d = mod_d.load_function(dyn_entry).unwrap();
                let s_static = best_of(ROUNDS, || time_wmma(g, &f_s, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                let s_dyn = best_of(ROUNDS, || time_wmma(g, &f_d, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                eprintln!(
                    "{m}x{k}x{n} f16 static-vs-dynamic (same raw-JIT load, M/N/K baked): dynamic {:>7.0} GFLOP/s | static {:>7.0} = {:.3}× speedup",
                    flop / s_dyn / 1e9,
                    flop / s_static / 1e9,
                    s_dyn / s_static,
                );
            }
        });
    }

    /// **int8 split-K occupancy bench (thin-M / small-N lever, contention-robust internal A/B).** When the
    /// M,N grid alone leaves SMs idle (thin M, small N), split-K adds `sk×` more CTAs along K to fill the
    /// GPU. Times the non-split swz kernel (sk=1) against the split-K kernel at sk∈{2,4,8} for small-N /
    /// large-K shapes, same-run; the ratio is Mercury-internal (no cuBLAS) so it stays honest under
    /// contention. Reports GFLOP/s alongside the launched CTA count so the saturation effect is visible.
    /// Bit-exact checksum cross-check at each sk first (first law). Run:
    /// `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture int8_splitk_occupancy`
    #[test]
    #[ignore = "throughput bench; run explicitly"]
    fn int8_splitk_occupancy() {
        use crate::baselines::gemm_flop;
        use crate::ptx_int8::{
            int8_gemm_smdb_swz_ptx, int8_gemm_smdb_swz_splitk_ptx, INT8_BM, INT8_BN, INT8_WARPS_M,
            INT8_WARPS_N,
        };
        with_gpu("int8_splitk_occupancy", |g| {
            eprintln!("device: {}", g.device_name());
            let warps = INT8_WARPS_M * INT8_WARPS_N;
            let mut rng = crate::diff::Rng::new(0x5C0FF);
            const ROUNDS: usize = 8;
            let f_base = g
                .function("int8_gemm_nt_smdb_swz", int8_gemm_smdb_swz_ptx(), "int8_gemm_nt_smdb_swz")
                .unwrap();
            let f_sk = g
                .function("int8_gemm_nt_smdb_swz_sk", int8_gemm_smdb_swz_splitk_ptx(), "int8_gemm_nt_smdb_swz_sk")
                .unwrap();
            for (m, n, k) in [(64usize, 256usize, 4096usize), (64, 512, 4096), (128, 256, 8192), (64, 128, 8192)] {
                let flop = gemm_flop(m, n, k);
                let dims = (m as u32, n as u32, k as u32);
                let a: Vec<u8> = (0..m * k).map(|_| (rng.f32_range(0.0, 256.0) as u32 & 0xff) as u8).collect();
                let b: Vec<i8> = (0..n * k).map(|_| ((rng.f32_range(0.0, 256.0) as i32) - 128) as i8).collect();
                let cs_ref: i64 = ref_nt_int8(&a, &b, m, k, n).iter().map(|&x| x as i64).sum();
                let a_d = g.stream.memcpy_stod(&a).unwrap();
                let b_d = g.stream.memcpy_stod(&b).unwrap();
                let base_ctas = (n / INT8_BN) * (m / INT8_BM);
                eprintln!("\nM{m} N{n} K{k}  (base grid = {base_ctas} CTAs):");
                // base checksum + timing.
                let mut cc = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                {
                    let mut bld = g.stream.launch_builder(&f_base);
                    bld.arg(&dims.0).arg(&dims.1).arg(&dims.2).arg(&a_d).arg(&b_d).arg(&mut cc);
                    unsafe { bld.launch(int8_smdb_cfg(m, n, INT8_BM, INT8_BN, warps)).unwrap() };
                }
                assert_eq!(g.stream.memcpy_dtov(&cc).unwrap().iter().map(|&x| x as i64).sum::<i64>(), cs_ref, "base {m}x{k}x{n}");
                let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                let base = {
                    let cfg = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, warps);
                    let mut s = f64::INFINITY;
                    for _ in 0..ROUNDS {
                        s = s.min(time_gemm_int8(g, &f_base, cfg, dims, &a_d, &b_d, &mut c_d, 50));
                    }
                    flop / s
                };
                eprintln!("  sk=1 (base): {:>7.0} GFLOP/s", base / 1e9);
                for sk in [2usize, 4, 8] {
                    if k % (sk * 64) != 0 {
                        continue;
                    }
                    let mut cfg = int8_smdb_cfg(m, n, INT8_BM, INT8_BN, warps);
                    cfg.grid_dim.2 = sk as u32;
                    // split-K needs a zeroed C; verify the deterministic red.add sum equals the reference.
                    let mut cz = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                    {
                        let mut bld = g.stream.launch_builder(&f_sk);
                        bld.arg(&dims.0).arg(&dims.1).arg(&dims.2).arg(&a_d).arg(&b_d).arg(&mut cz);
                        unsafe { bld.launch(cfg).unwrap() };
                    }
                    assert_eq!(g.stream.memcpy_dtov(&cz).unwrap().iter().map(|&x| x as i64).sum::<i64>(), cs_ref, "sk={sk} {m}x{k}x{n}");
                    let mut c2 = g.stream.memcpy_stod(&vec![0i32; m * n]).unwrap();
                    let mut s = f64::INFINITY;
                    for _ in 0..ROUNDS {
                        s = s.min(time_gemm_int8(g, &f_sk, cfg, dims, &a_d, &b_d, &mut c2, 50));
                    }
                    let gf = flop / s;
                    eprintln!("  sk={sk} ({:>4} CTAs): {:>7.0} GFLOP/s  → {:.3}× base", base_ctas * sk, gf / 1e9, gf / base);
                }
            }
        });
    }
}
