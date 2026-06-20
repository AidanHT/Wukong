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

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaModule, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;

use crate::gpu::Gpu;

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
