//! Honest GPU peer baselines — the scoreboard's Tier-A and Tier-B competitors.
//!
//! Mercury's GPU kernels are worthless as a *claim* until they are measured against the kernels a
//! real engineer would reach for **on the same GPU**. Comparing a Mercury GPU kernel against
//! C-on-the-CPU would show a trivial 50–100× and prove nothing but "a GPU beats a CPU" — that
//! comparison is forbidden as a headline (see the GPU plan §1). The fair, harder bars are:
//!
//! * **Tier A — naive CUDA-C** (`nvrtc_naive_gemm_nt`): the idiomatic kernel a programmer writes by
//!   hand, compiled at runtime by **NVRTC** and launched through the same driver as Mercury's PTX.
//!   This is the literal "beat C/C++/Rust on the GPU" — the GPU twin of beating scalar C on the CPU.
//! * **Tier B — cuBLAS** (`cublas_gemm_nt_f16`): NVIDIA's hand-tuned closed-source GEMM, the gold
//!   standard. Mercury is reported as a **% of cuBLAS**, same-run, same buffers.
//!
//! Both peers `dlopen` their redistributable DLLs (`nvrtc64_120_0.dll`, `cublas64_12.dll`) exactly
//! the way `cudarc` already `dlopen`s the driver (`nvcuda.dll`). So *building* this crate still needs
//! no CUDA toolkit; only *running these benches* needs the DLLs reachable on the loader path. On this
//! box they live in a git-ignored `tools/cuda-redist/` (see `peer_env_hint`), put on `PATH` by the
//! bench invocation. If they are absent the loader would `panic!`, so [`peers_available`] probes for
//! them under `catch_unwind` and the peer benches **skip** (never fail) when they are missing — the
//! same "green without the hardware" discipline the GPU tests already follow.
//!
//! Correctness first (the plan's first law): every peer is cross-checked against the **same f64 CPU
//! reference** as Mercury's own kernels before any speed number counts, so a fast-but-wrong peer
//! can't flatter Mercury and a fast-but-wrong Mercury can't beat a correct peer.

use std::sync::Arc;

use cudarc::cublas::result as cublas_result;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DriverError,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;

use crate::gpu::{Gpu, TransformerWeights};

/// Any peer-setup failure unifies to this so a bench can `?`/`unwrap` across the three cudarc error
/// families (driver / NVRTC compile / cuBLAS) with one type.
pub type PeerError = Box<dyn std::error::Error + Send + Sync>;

/// One-line hint, printed by the benches, for reproducing the redistributable DLLs and the `PATH`
/// they must sit on. Kept here so the "how do I run the peer benches" answer lives next to the code.
pub fn peer_env_hint() -> &'static str {
    // cudarc 0.16's `cuda-12060` bindings actually reference 12.8-era NVRTC PCH symbols and 12.9-era
    // cuBLAS emulation symbols, so the redist DLLs must be the 12.9 superset; NVRTC 12.9 also hard-
    // imports nvJitLink, hence the third wheel. Install all three in ONE command — pip's `--target`
    // clobbers the shared `nvidia/` namespace if they go in separately.
    "GPU peer baselines need the CUDA 12.9 redist DLLs on PATH. From the repo root:\n  \
     pip install --target tools/cuda-redist --no-deps nvidia-cuda-nvrtc-cu12==12.9.86 \
     nvidia-cublas-cu12==12.9.2.10 nvidia-nvjitlink-cu12==12.9.86\n  \
     then prepend these to PATH: tools/cuda-redist/nvidia/{cuda_nvrtc,cublas,nvjitlink}/bin"
}

/// Probe whether the NVRTC + cuBLAS DLLs are loadable in this process, **without** letting a missing
/// DLL abort the run: `cudarc`'s lazy loader `panic!`s if no candidate library is found, so we drive
/// the first real call to each library under `catch_unwind` and treat a panic (or any error) as "not
/// available → skip". The probe is cheap and its result is cached for the whole process.
pub fn peers_available(g: &mut Gpu) -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        // NVRTC: compile a trivial program. cuBLAS: create a handle on the stream. Either touching a
        // missing DLL panics inside cudarc's loader; catch it so we report false instead of crashing.
        let stream = g.stream.clone();
        let nvrtc_ok = std::panic::catch_unwind(|| {
            compile_ptx_with_opts("extern \"C\" __global__ void p(){}", CompileOptions::default())
                .is_ok()
        })
        .unwrap_or(false);
        let cublas_ok =
            std::panic::catch_unwind(|| CudaBlas::new(stream).is_ok()).unwrap_or(false);
        nvrtc_ok && cublas_ok
    })
}

// ---------------------------------------------------------------------------------------------------
// Tier A — naive CUDA-C GEMM, compiled by NVRTC (the "beat the hand-written C kernel" baseline).
// ---------------------------------------------------------------------------------------------------

/// The idiomatic one-thread-per-output-element GEMM a programmer writes first: no tiling, no shared
/// memory, no tensor cores. `C = A·Bᵀ` (the `nn.Linear` contract: `A` is `M×K` row-major, `B` is
/// `N×K` row-major), one fused-multiply-add chain per output. NVRTC compiles this CUDA-C to PTX at
/// runtime; the driver JITs it to SASS just like Mercury's own PTX, so the comparison is pure
/// kernel-quality on one device.
const NAIVE_GEMM_NT_CUDA: &str = r#"
extern "C" __global__ void naive_gemm_nt(int M, int N, int K,
                                         const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m index
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[col * K + k];
        C[row * N + col] = acc;
    }
}
"#;

/// Compile the naive CUDA-C GEMM with NVRTC (targeting this box's `sm_89`) and load it. Returned as a
/// loaded module so callers can launch it many times for timing without recompiling.
fn nvrtc_naive_gemm_module(g: &Gpu) -> Result<Arc<CudaModule>, PeerError> {
    let opts = CompileOptions {
        arch: Some("compute_89"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(NAIVE_GEMM_NT_CUDA, opts)?;
    Ok(g.ctx.load_module(ptx)?)
}

fn naive_cfg(m: usize, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    }
}

/// Run the naive CUDA-C GEMM once and copy the result back — the correctness-gate entry point.
pub fn nvrtc_naive_gemm_nt(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let module = nvrtc_naive_gemm_module(g)?;
    let f = module.load_function("naive_gemm_nt")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as i32, n as i32, k as i32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(naive_cfg(m, n))? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// Time the naive CUDA-C GEMM: `iters` kernel-resident launches (buffers stay on-device) bracketed by
/// one sync, after a warm-up launch — the identical timing shape Mercury's own GEMM benches use, so
/// the ratio is apples-to-apples. Returns seconds per launch.
pub fn time_nvrtc_naive_gemm_nt(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let module = nvrtc_naive_gemm_module(g)?;
    let f = module.load_function("naive_gemm_nt")?;
    let a_d = g.stream.memcpy_stod(&vec![0.01f32; m * k])?;
    let b_d = g.stream.memcpy_stod(&vec![0.01f32; n * k])?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk) = (m as i32, n as i32, k as i32);
    let cfg = naive_cfg(m, n);
    let mut launch = |g: &Gpu| -> Result<(), DriverError> {
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
        unsafe { bld.launch(cfg) }.map(|_| ())
    };
    launch(g)?; // warm up
    g.stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch(g)?;
    }
    g.stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// ---------------------------------------------------------------------------------------------------
// Tier B — cuBLAS fp16 GEMM (the gold-standard peer; Mercury reports as a % of this).
// ---------------------------------------------------------------------------------------------------

/// Build the column-major cuBLAS config that makes cuBLAS compute Mercury's **row-major** `C = A·Bᵀ`.
///
/// cuBLAS is column-major; Mercury is row-major. A row-major `C (M×N)` is a column-major `Cᵀ (N×M)`,
/// and `Cᵀ = (A·Bᵀ)ᵀ = B·Aᵀ`. Writing Mercury's row-major buffers as their column-major transposes
/// (`Ǎ = Aᵀ` is `K×M` ld=K, `B̌ = Bᵀ` is `K×N` ld=K) gives `Cᵀ = B̌ᵀ · Ǎ`: pass **B as the first
/// operand transposed** and **A as the second untransposed**, with `m,n` swapped. (The checksum gate
/// vs the f64 reference verifies this mapping — a transpose slip shows up as a gross mismatch.)
fn cublas_nt_cfg(m: usize, k: usize, n: usize) -> GemmConfig<f16> {
    GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, // rows of Cᵀ
        n: m as i32, // cols of Cᵀ
        k: k as i32,
        alpha: f16::from_f32(1.0),
        lda: k as i32, // B̌ (Mercury B), K×N col-major
        ldb: k as i32, // Ǎ (Mercury A), K×M col-major
        beta: f16::from_f32(0.0),
        ldc: n as i32, // Cᵀ, N×M col-major
    }
}

/// Run cuBLAS fp16 (`cublasGemmEx`, `CUDA_R_16F` data, `CUBLAS_COMPUTE_32F` accumulate — tensor cores
/// via the default algorithm) once and return the result widened to f32. The correctness-gate entry.
/// Inputs are the same f32 buffers Mercury gets; we round to f16 on the host (the price cuBLAS's fp16
/// path also pays), so all three tiers consume identical numerics.
pub fn cublas_gemm_nt_f16(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let blas = CudaBlas::new(g.stream.clone())?;
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.0); m * n])?;
    let cfg = cublas_nt_cfg(m, k, n);
    // Per the mapping: first operand is Mercury's B (transposed), second is Mercury's A.
    unsafe { blas.gemm(cfg, &b_d, &a_d, &mut c_d)? };
    let c16: Vec<f16> = g.stream.memcpy_dtov(&c_d)?;
    Ok(c16.iter().map(|x| x.to_f32()).collect())
}

/// Time cuBLAS fp16 GEMM: `iters` resident calls on the stream, one warm-up, one trailing sync —
/// matching the Mercury/NVRTC timing shape. Returns seconds per call.
pub fn time_cublas_gemm_nt_f16(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let blas = CudaBlas::new(g.stream.clone())?;
    let a_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.01); m * k])?;
    let b_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.01); n * k])?;
    let mut c_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.0); m * n])?;
    let cfg = cublas_nt_cfg(m, k, n);
    unsafe { blas.gemm(cfg, &b_d, &a_d, &mut c_d)? }; // warm up
    g.stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { blas.gemm(cfg, &b_d, &a_d, &mut c_d)? };
    }
    g.stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

/// `2·M·N·K` — the FLOP count of an `M×N×K` GEMM, for turning seconds/call into FLOP/s.
pub fn gemm_flop(m: usize, n: usize, k: usize) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64
}

// ---------------------------------------------------------------------------------------------------
// Tier B — cuBLAS **call-chain transformer layer** (the literal "library call-chain" the M13
// milestone must beat). cuBLAS only does GEMM, so a transformer built on it is the canonical chain of
// `cublasGemmEx` calls glued by hand-written norm/attention/activation/residual kernels — and crucially
// **cannot fuse** the activation or the skip-connection add into the GEMM (plain cuBLAS has no epilogue
// for SiLU or residual-from-a-third-buffer). That non-fusion is exactly the cost Mercury's fused WMMA
// stack ([`ResidentLayerF16`]) folds away. To make the comparison about *fusion and GEMM quality only*,
// this peer reuses Mercury's identical norm/flash/cast/SiLU/vadd kernels (cuBLAS provides none of them);
// the sole substitution is cuBLAS for the six projections.
// ---------------------------------------------------------------------------------------------------

/// `cublasGemmEx`, **f16 in / f32 out** (`CUDA_R_16F` data, `CUDA_R_32F` C, `CUBLAS_COMPUTE_32F`): the
/// fragments multiply in fp16, accumulate in f32, and the result is **stored straight to f32** — the
/// identical dtype boundary as Mercury's WMMA GEMM, which also accumulates f32 and stores f32. This is
/// the fairness keystone: if cuBLAS stored f16 we would owe an f16→f32 cast kernel before the next
/// (f32) norm/flash stage that Mercury never pays, silently biasing the chain in Mercury's favour. With
/// f32 out, **both** stacks pay exactly one f32→f16 narrowing *before* each GEMM and nothing after.
///
/// Computes Mercury's row-major `C[M×N] = A[M×K]·B[N×K]ᵀ` (the `nn.Linear` contract) using the same
/// column-major transpose identity as [`cublas_nt_cfg`]: `Cᵀ = B̌ᵀ·Ǎ`, so B is passed first transposed
/// and A second untransposed with `m,n` swapped. Buffers must already be resident; nothing is synced.
///
/// # Safety
/// `a_d`, `b_d`, `c_d` must be valid device buffers of length `m*k`, `n*k`, `m*n`; the cuBLAS handle in
/// `blas` and `stream` must be live. The device-pointer guards are held across the `gemm_ex` call.
unsafe fn gemm_ex_nt_f16_f32out(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    a_d: &CudaSlice<f16>,
    b_d: &CudaSlice<f16>,
    c_d: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), PeerError> {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let (ap, _ra) = a_d.device_ptr(stream);
    let (bp, _rb) = b_d.device_ptr(stream);
    let (cp, _rc) = c_d.device_ptr_mut(stream);
    cublas_result::gemm_ex(
        *blas.handle(),
        cublasOperation_t::CUBLAS_OP_T, // B̌ transposed (Mercury's B, first operand)
        cublasOperation_t::CUBLAS_OP_N, // Ǎ untransposed (Mercury's A, second operand)
        n as i32,                       // rows of Cᵀ
        m as i32,                       // cols of Cᵀ
        k as i32,
        (&alpha) as *const f32 as *const _,
        bp as *const _,
        cudaDataType_t::CUDA_R_16F,
        k as i32, // lda: B̌ is K×N col-major
        ap as *const _,
        cudaDataType_t::CUDA_R_16F,
        k as i32, // ldb: Ǎ is K×M col-major
        (&beta) as *const f32 as *const _,
        cp as *mut _,
        cudaDataType_t::CUDA_R_32F,
        n as i32, // ldc: Cᵀ is N×M col-major
        cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    )?;
    Ok(())
}

/// One-shot correctness-gate entry for [`gemm_ex_nt_f16_f32out`]: host f32 `A`,`B` in (rounded to f16,
/// the price the fp16 path pays), f32 `C` out. The f32-out sibling of [`cublas_gemm_nt_f16`] — cross-
/// checked against the same f64 reference so a transpose slip or a wrong dtype shows as a gross miss.
pub fn cublas_gemm_nt_f16_f32out(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let blas = CudaBlas::new(g.stream.clone())?;
    let a16: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let a_d = g.stream.memcpy_stod(&a16)?;
    let b_d = g.stream.memcpy_stod(&b16)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let stream = g.stream.clone();
    unsafe { gemm_ex_nt_f16_f32out(&blas, &stream, &a_d, &b_d, &mut c_d, m, k, n)? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// The **cuBLAS-GEMM call-chain transformer layer** — Tier-B peer to [`ResidentLayerF16`]. Same resident
/// f16 weights, same `[S,D]` f32 activation contract, same pre-norm encoder math; the six projections
/// (Q/K/V/O, W1, W2) run on cuBLAS ([`gemm_ex_nt_f16_f32out`]) and the skip-connection adds + SiLU run as
/// **separate** kernels (cuBLAS cannot fuse them). Structurally this is exactly Mercury's *unfused* path
/// [`ResidentLayerF16::forward_device_unfused`] with cuBLAS substituted for the WMMA GEMM — so the
/// measured gap against Mercury's *fused* [`forward_device`](ResidentLayerF16::forward_device) is purely
/// (cuBLAS-vs-WMMA GEMM quality) + (the epilogue fusion cuBLAS structurally can't do).
pub struct CublasChainLayer {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    f_norm: CudaFunction,
    f_cast: CudaFunction,
    f_flash: CudaFunction,
    f_silu: CudaFunction,
    f_vadd: CudaFunction,
    wq: CudaSlice<f16>,
    wk: CudaSlice<f16>,
    wv: CudaSlice<f16>,
    wo: CudaSlice<f16>,
    w1: CudaSlice<f16>,
    w2: CudaSlice<f16>,
    s: usize,
    d: usize,
    dff: usize,
    eps: f32,
}

impl CublasChainLayer {
    /// Upload the weights (narrowed to f16) and preload the glue kernels — mirrors
    /// [`ResidentLayerF16::new`]'s shape constraints (S,D,Dff multiples of 64; `d` a supported flash head
    /// dim) and reuses the *same* kernel keys (`norm`/`cast`/`flash`/`vmath`/`vadd`) so those kernels are
    /// literally identical between the two stacks — only the GEMM differs.
    pub fn new(
        g: &mut Gpu,
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
    ) -> Result<Self, PeerError> {
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
            "CublasChainLayer needs S,D,Dff multiples of 64 (to match the WMMA peer's tiles)"
        );
        assert!(
            crate::ptx_flash::SUPPORTED_D.contains(&d),
            "CublasChainLayer: head dim {d} unsupported by flash (need {:?})",
            crate::ptx_flash::SUPPORTED_D
        );
        let blas = CudaBlas::new(g.stream.clone())?;
        let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
        let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
        let f_flash =
            g.function("flash", crate::ptx_flash::flash_ptx(), &format!("flash_d{d}"))?;
        let f_silu = g.function("vmath", crate::ptx::vmath_ptx(), "silu")?;
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
            blas,
            f_norm,
            f_cast,
            f_flash,
            f_silu,
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

    /// RMSNorm a `[rows, d]` f32 buffer (one warp per row) — Mercury's exact `rmsnorm` kernel.
    fn norm(&self, src: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.memcpy_stod(&vec![0f32; rows * self.d])?;
        let (r, c) = (rows as u32, self.d as u32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut bld = self.stream.launch_builder(&self.f_norm);
        bld.arg(&r).arg(&c).arg(&self.eps).arg(src).arg(&mut out);
        unsafe { bld.launch(cfg)? };
        Ok(out)
    }

    /// device f32 → device f16 narrowing (the stage boundary before each cuBLAS GEMM) — Mercury's
    /// exact `cast_f32_f16` kernel, so the f16 inputs both stacks feed their GEMMs are bit-identical.
    fn cast(&self, src: &CudaSlice<f32>, n: usize) -> Result<CudaSlice<f16>, DriverError> {
        let mut dst = self.stream.memcpy_stod(&vec![f16::from_f32(0.0); n])?;
        let nn = n as u32;
        let mut b = self.stream.launch_builder(&self.f_cast);
        b.arg(&nn).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// `C[M×N] = A·Bᵀ` on cuBLAS (f16-in/f32-out) — the one substitution vs Mercury's WMMA path.
    fn gemm(
        &self,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, PeerError> {
        let mut c = self.stream.memcpy_stod(&vec![0f32; m * n])?;
        unsafe { gemm_ex_nt_f16_f32out(&self.blas, &self.stream, a, b, &mut c, m, k, n)? };
        Ok(c)
    }

    /// Fused flash-attention over `[S,D]` Q/K/V (Mercury's exact `flash_d{D}` kernel) → `[S,D]` f32.
    fn flash(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let mut attn = self.stream.memcpy_stod(&vec![0f32; self.s * self.d])?;
        let scale = 1.0f32 / (self.d as f32).sqrt();
        let ss = self.s as u32;
        let cfg = LaunchConfig { grid_dim: (self.s as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut bld = self.stream.launch_builder(&self.f_flash);
        bld.arg(&ss).arg(&scale).arg(q).arg(k).arg(v).arg(&mut attn);
        unsafe { bld.launch(cfg)? };
        Ok(attn)
    }

    /// Separate residual add `out = a + b` — the kernel cuBLAS forces (Mercury folds it via wmma.load.c).
    fn vadd(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.memcpy_stod(&vec![0f32; n])?;
        let nn = n as u32;
        let mut bld = self.stream.launch_builder(&self.f_vadd);
        bld.arg(&nn).arg(a).arg(b).arg(&mut out);
        unsafe { bld.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(out)
    }

    /// Separate SiLU activation — the kernel cuBLAS forces (Mercury folds it into the up-proj store).
    fn silu(&self, src: &CudaSlice<f32>, n: usize) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.memcpy_stod(&vec![0f32; n])?;
        let nn = n as u32;
        let mut bld = self.stream.launch_builder(&self.f_silu);
        bld.arg(&nn).arg(src).arg(&mut out);
        unsafe { bld.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(out)
    }

    /// Run the layer on a **resident** `[S,D]` f32 buffer → fresh resident `[S,D]` f32 — the cuBLAS
    /// call-chain forward, no host transfer. The kernel sequence is identical to
    /// [`ResidentLayerF16::forward_device_unfused`] (norm → cast → 3 proj → flash → cast → O-proj →
    /// **separate** residual → norm → cast → up-proj → **separate** SiLU → cast → down-proj →
    /// **separate** residual), with every projection on cuBLAS instead of WMMA.
    pub fn forward_device(
        &self,
        x_d: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, PeerError> {
        let (s, d, dff) = (self.s, self.d, self.dff);
        // attention
        let h1 = self.norm(x_d, s)?;
        let h1_16 = self.cast(&h1, s * d)?;
        let q = self.gemm(&h1_16, &self.wq, s, d, d)?;
        let k = self.gemm(&h1_16, &self.wk, s, d, d)?;
        let v = self.gemm(&h1_16, &self.wv, s, d, d)?;
        let attn = self.flash(&q, &k, &v)?;
        let attn_16 = self.cast(&attn, s * d)?;
        let o = self.gemm(&attn_16, &self.wo, s, d, d)?;
        let x1 = self.vadd(x_d, &o, s * d)?; // residual 1 (separate add)
        // FFN
        let h2 = self.norm(&x1, s)?;
        let h2_16 = self.cast(&h2, s * d)?;
        let f1 = self.gemm(&h2_16, &self.w1, s, d, dff)?;
        let f1act = self.silu(&f1, s * dff)?; // SiLU (separate)
        let f1act_16 = self.cast(&f1act, s * dff)?;
        let f2 = self.gemm(&f1act_16, &self.w2, s, dff, d)?;
        let out = self.vadd(&x1, &f2, s * d)?; // residual 2 (separate add)
        Ok(out)
    }

    /// One-shot host call: upload `x` (`[S,D]` f32), run [`forward_device`](Self::forward_device), copy
    /// the `[S,D]` result back — the transfer-inclusive convenience path, mirroring
    /// [`ResidentLayerF16::forward`].
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, PeerError> {
        assert_eq!(x.len(), self.s * self.d, "x must be S×D");
        let x_d = self.stream.memcpy_stod(x)?;
        let out = self.forward_device(&x_d)?;
        Ok(self.stream.memcpy_dtov(&out)?)
    }
}

/// A stack of N [`CublasChainLayer`]s — the cuBLAS-call-chain counterpart to Mercury's whole-model
/// `ResidentModelF16`, for the M13 "**beat the library-call-chain stack end-to-end**" claim. Each layer
/// keeps its own f16 weights and cuBLAS handle; [`forward_device`](Self::forward_device) chains them on
/// device buffers exactly like the resident model (one layer's output is the next's input). The contrast
/// the depth bench draws out: Mercury's resident model keeps every activation on-device across all N
/// layers with fused epilogues, whereas this chain pays cuBLAS's per-call dispatch *plus* the separate
/// residual-add/SiLU launches at **every** layer — so the per-layer gap is paid N times.
pub struct CublasChainModel {
    stream: Arc<CudaStream>,
    layers: Vec<CublasChainLayer>,
    s: usize,
    d: usize,
}

impl CublasChainModel {
    /// Build the N cuBLAS-chain layers (one [`CublasChainLayer::new`] per weight set) — all share the one
    /// device stream, so the whole stack enqueues in order on a single timeline (as the resident model does).
    pub fn new(
        g: &mut Gpu,
        weights: &[TransformerWeights],
        s: usize,
        d: usize,
        dff: usize,
    ) -> Result<Self, PeerError> {
        assert!(!weights.is_empty(), "model needs at least one layer");
        let mut layers = Vec::with_capacity(weights.len());
        for w in weights {
            layers.push(CublasChainLayer::new(g, w, s, d, dff)?);
        }
        let stream = g.stream.clone();
        Ok(Self { stream, layers, s, d })
    }

    /// Number of layers in the stack.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Run the whole cuBLAS call-chain stack on a **resident** `[S,D]` buffer → resident final output, the
    /// pure on-device N-layer chain (no host transfer between layers) — the fair counterpart to
    /// [`ResidentModelF16::forward_device`], differing only in cuBLAS-GEMM + unfused epilogues per layer.
    pub fn forward_device(
        &self,
        x_d: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, PeerError> {
        let mut cur = self.layers[0].forward_device(x_d)?;
        for layer in &self.layers[1..] {
            cur = layer.forward_device(&cur)?;
        }
        Ok(cur)
    }

    /// One-shot host call: upload `x` once, run the whole chain stack, copy the final `[S,D]` back once.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, PeerError> {
        assert_eq!(x.len(), self.s * self.d, "x must be S×D");
        let x_d = self.stream.memcpy_stod(x)?;
        let out = self.forward_device(&x_d)?;
        Ok(self.stream.memcpy_dtov(&out)?)
    }
}
