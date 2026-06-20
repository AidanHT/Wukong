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
    use half::f16;
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert!(
        m % 16 == 0 && n % 16 == 0 && k % 16 == 0,
        "WMMA requires 16-multiple dims"
    );
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

                eprintln!(
                    "{m}³: f32-rb {:.0} GFLOP/s | f16-TC {:.0} GFLOP/s ({:.1}× rb) | bf16-TC {:.0} GFLOP/s ({:.1}× rb)",
                    flop / s_rb / 1e9,
                    flop / s_f16 / 1e9,
                    s_rb / s_f16,
                    flop / s_bf16 / 1e9,
                    s_rb / s_bf16,
                );
            }
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
}
