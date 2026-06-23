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
// Tier A — naive CUDA-C attention, compiled by NVRTC (the "beat the hand-written C flash" baseline).
// ---------------------------------------------------------------------------------------------------

/// The idiomatic attention a programmer writes first: **one thread per (head, query row)**, a two-pass
/// online-stable softmax streamed over the keys, no shared memory, no tensor cores, the `S×S` scores
/// never materialized (recomputed in pass 2 rather than stored — the natural way to keep it O(D) state
/// per thread). `O = softmax(scale·Q·Kᵀ)·V`, all of Q/K/V/O laid out `[H, S, D]` (head-major). This is
/// the GPU twin of the scalar CPU baseline: NVRTC compiles it to PTX, the driver JITs it like Mercury's
/// own PTX, so the gap is pure kernel quality (tensor cores + register-resident accumulation vs none).
const NAIVE_ATTN_CUDA: &str = r#"
extern "C" __global__ void naive_attn(int H, int S, int D, float scale,
                                      const float* Q, const float* K, const float* V, float* O) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // query row
    int h = blockIdx.y;                            // head
    if (i >= S) return;
    const float* q  = Q + ((long)h * S + i) * D;
    const float* Kh = K + (long)h * S * D;
    const float* Vh = V + (long)h * S * D;
    float* o = O + ((long)h * S + i) * D;
    float m = -1e30f;
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < D; ++t) s += q[t] * Kh[(long)j * D + t];
        s *= scale;
        if (s > m) m = s;
    }
    for (int t = 0; t < D; ++t) o[t] = 0.0f;
    float l = 0.0f;
    for (int j = 0; j < S; ++j) {
        float s = 0.0f;
        for (int t = 0; t < D; ++t) s += q[t] * Kh[(long)j * D + t];
        float p = __expf(s * scale - m);
        l += p;
        for (int t = 0; t < D; ++t) o[t] += p * Vh[(long)j * D + t];
    }
    float inv = 1.0f / l;
    for (int t = 0; t < D; ++t) o[t] *= inv;
}
"#;

fn nvrtc_naive_attn_module(g: &Gpu) -> Result<Arc<CudaModule>, PeerError> {
    let opts = CompileOptions {
        arch: Some("compute_89"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(NAIVE_ATTN_CUDA, opts)?;
    Ok(g.ctx.load_module(ptx)?)
}

fn naive_attn_cfg(h: usize, s: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((s as u32).div_ceil(128), h as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Run the naive CUDA-C attention once and copy `O` back — the correctness-gate entry. Q/K/V are the
/// same f32 buffers Mercury's flash is cross-checked against (`[H,S,D]`, head-major).
pub fn nvrtc_naive_attn(
    g: &mut Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    h: usize,
    s: usize,
    d: usize,
    scale: f32,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(q.len(), h * s * d);
    assert_eq!(k.len(), h * s * d);
    assert_eq!(v.len(), h * s * d);
    let module = nvrtc_naive_attn_module(g)?;
    let f = module.load_function("naive_attn")?;
    let q_d = g.stream.memcpy_stod(q)?;
    let k_d = g.stream.memcpy_stod(k)?;
    let v_d = g.stream.memcpy_stod(v)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; h * s * d])?;
    let (hh, ss, dd) = (h as i32, s as i32, d as i32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&hh).arg(&ss).arg(&dd).arg(&scale).arg(&q_d).arg(&k_d).arg(&v_d).arg(&mut o_d);
    unsafe { bld.launch(naive_attn_cfg(h, s))? };
    Ok(g.stream.memcpy_dtov(&o_d)?)
}

/// Time the naive CUDA-C attention: `iters` resident launches bracketed by one sync, after a warm-up —
/// the identical timing shape Mercury's flash benches use, so the ratio is apples-to-apples. Seconds/launch.
pub fn time_nvrtc_naive_attn(
    g: &mut Gpu,
    h: usize,
    s: usize,
    d: usize,
    scale: f32,
    iters: u32,
) -> Result<f64, PeerError> {
    let module = nvrtc_naive_attn_module(g)?;
    let f = module.load_function("naive_attn")?;
    let q_d = g.stream.memcpy_stod(&vec![0.01f32; h * s * d])?;
    let k_d = g.stream.memcpy_stod(&vec![0.01f32; h * s * d])?;
    let v_d = g.stream.memcpy_stod(&vec![0.01f32; h * s * d])?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; h * s * d])?;
    let (hh, ss, dd) = (h as i32, s as i32, d as i32);
    let cfg = naive_attn_cfg(h, s);
    let mut launch = |g: &Gpu| -> Result<(), DriverError> {
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&hh).arg(&ss).arg(&dd).arg(&scale).arg(&q_d).arg(&k_d).arg(&v_d).arg(&mut o_d);
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

/// `4·H·S²·D` — attention FLOPs (QKᵀ `2·H·S²·D` + P·V `2·H·S²·D`), for turning seconds into FLOP/s.
pub fn attn_flop(h: usize, s: usize, d: usize) -> f64 {
    4.0 * h as f64 * s as f64 * s as f64 * d as f64
}

// ---------------------------------------------------------------------------------------------------
// Tier B — cuBLAS **unfused attention chain** (the gold-standard library bar for M5). This is the
// canonical *pre-FlashAttention* attention: run the two matmuls on tensor-core cuBLAS and **materialize
// the S×S score matrix to HBM** in between (write S1, read it for softmax, write P, read it for P·V).
// That HBM round-trip of the S×S scores is exactly what Mercury's *fused* flash never pays — so the gap
// the bench measures is the value of fusion (and grows with S, since S² dwarfs the S·D I/O). cuBLAS
// provides no softmax/cast, so — exactly as `CublasChainLayer` does for the layer — those two glue
// kernels are Mercury's own (identical in both stacks); the only thing being compared is fused-vs-not.
// A genuinely *fused* FA2-class CUDA-C peer is not buildable here: NVRTC on this toolkit-free box has no
// header search path at all (even `#include <cuda_fp16.h>` fails), so `nvcuda::wmma` can't be compiled
// (see the `nvrtc_wmma_probe` capability test). The cuBLAS chain is the strongest library peer available.
// ---------------------------------------------------------------------------------------------------

/// cuBLAS `O[M×N] = A[M×K]·B[K×N]` — the **NN** (no-transpose) product `P·V` needs — f16 in / f32 out
/// (`CUDA_R_16F` data, `CUDA_R_32F` C, `CUBLAS_COMPUTE_32F`), tensor cores via the default algorithm:
/// the same f32-accumulate / f32-store boundary Mercury's flash uses. Row-major→col-major identity for a
/// plain product: row-major `O[M,N]` is col-major `Oᵀ[N,M] = Bᵀ·Aᵀ`, and the stored row-major buffers
/// *are* `Bᵀ`/`Aᵀ` when read column-major — so pass **B first** (its `[K,N]` buffer, ld=N) and **A
/// second** (its `[M,K]` buffer, ld=K), both `OP_N`, with `m,n` swapped. (Symmetric to [`cublas_nt_cfg`]'s
/// trick but both-N; the f64 gate on the full chain catches any slip.)
///
/// # Safety
/// `a_d`,`b_d`,`c_d` must be valid device buffers of length `m*k`, `k*n`, `m*n`; the handle/stream live.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_ex_nn_f16_f32out(
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
        cublasOperation_t::CUBLAS_OP_N, // B as [N,K] col-major view (= Bᵀ)
        cublasOperation_t::CUBLAS_OP_N, // A as [K,M] col-major view (= Aᵀ)
        n as i32,                       // rows of Oᵀ
        m as i32,                       // cols of Oᵀ
        k as i32,
        (&alpha) as *const f32 as *const _,
        bp as *const _,
        cudaDataType_t::CUDA_R_16F,
        n as i32, // lda: B is K×N row-major ⇒ N-wide
        ap as *const _,
        cudaDataType_t::CUDA_R_16F,
        k as i32, // ldb: A is M×K row-major ⇒ K-wide
        (&beta) as *const f32 as *const _,
        cp as *mut _,
        cudaDataType_t::CUDA_R_32F,
        n as i32, // ldc: Oᵀ is N×M col-major ⇒ N-wide
        cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    )?;
    Ok(())
}

/// One run of the unfused attention chain over **resident** single-head buffers, no allocation/sync —
/// the timing-loop body: `S1 = (scale·Q)·Kᵀ` (cuBLAS NT) → `P = softmax_row(S1)` (Mercury's warp-per-row
/// softmax glue) → `P16 = f16(P)` (cast glue) → `O = P·V` (cuBLAS NN). `scale` is pre-folded into `q_d`
/// by the caller (cuBLAS `gemm_ex` runs α=1), so the softmax sees `scale·Q·Kᵀ`.
///
/// # Safety
/// All slices must be valid resident buffers of the documented `[S,D]`/`[S,S]` sizes; handle/stream live.
#[allow(clippy::too_many_arguments)]
unsafe fn cublas_attn_chain_once(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    f_softmax: &CudaFunction,
    f_cast: &CudaFunction,
    q_d: &CudaSlice<f16>,
    k_d: &CudaSlice<f16>,
    v_d: &CudaSlice<f16>,
    o_d: &mut CudaSlice<f32>,
    s1_d: &mut CudaSlice<f32>,
    p_d: &mut CudaSlice<f32>,
    p16_d: &mut CudaSlice<f16>,
    s: usize,
    d: usize,
) -> Result<(), PeerError> {
    // (1) scores S1[S,S] = (scale·Q)·Kᵀ — tensor-core cuBLAS, f32 out (reuses the NT helper).
    unsafe { gemm_ex_nt_f16_f32out(blas, stream, q_d, k_d, s1_d, s, d, s)? };
    // (2) P[S,S] = softmax over each of the S rows — Mercury's exact warp-per-row softmax (cuBLAS has
    //     none); `eps` is ignored by the softmax arm of the shared norm signature.
    {
        let (r, c) = (s as u32, s as u32);
        let eps = 0.0f32;
        let cfg = LaunchConfig { grid_dim: (r, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut bld = stream.launch_builder(f_softmax);
        bld.arg(&r).arg(&c).arg(&eps).arg(&*s1_d).arg(&mut *p_d);
        unsafe { bld.launch(cfg)? };
    }
    // (3) narrow P→f16 for the tensor-core second GEMM — the inter-stage cast cuBLAS forces.
    {
        let n = (s * s) as u32;
        let mut bld = stream.launch_builder(f_cast);
        bld.arg(&n).arg(&*p_d).arg(&mut *p16_d);
        unsafe { bld.launch(LaunchConfig::for_num_elems(n))? };
    }
    // (4) O[S,D] = P·V — tensor-core cuBLAS NN, f32 out.
    unsafe { gemm_ex_nn_f16_f32out(blas, stream, &*p16_d, v_d, o_d, s, s, d)? };
    Ok(())
}

/// Run the cuBLAS **unfused attention chain** (single head) once and copy `O` back — the correctness
/// gate entry. `O = softmax(scale·Q·Kᵀ)·V`, Q/K/V `[S,D]` row-major, the two matmuls on tensor-core
/// cuBLAS with Mercury's softmax+cast as the glue cuBLAS can't provide. Cross-checked against the same
/// f64 oracle Mercury's flash is, so a transpose slip or wrong config shows as a gross mismatch.
pub fn cublas_attn_chain(
    g: &mut Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    d: usize,
    scale: f32,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(q.len(), s * d);
    assert_eq!(k.len(), s * d);
    assert_eq!(v.len(), s * d);
    let f_softmax = g.function("norm", crate::ptx_norm::norm_ptx(), "softmax")?;
    let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
    let blas = CudaBlas::new(g.stream.clone())?;
    let stream = g.stream.clone();
    // Fold scale into Q (cuBLAS gemm_ex uses α=1, so the softmax must see scale·Q·Kᵀ).
    let q16: Vec<f16> = q.iter().map(|&x| f16::from_f32(x * scale)).collect();
    let k16: Vec<f16> = k.iter().map(|&x| f16::from_f32(x)).collect();
    let v16: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
    let q_d = stream.memcpy_stod(&q16)?;
    let k_d = stream.memcpy_stod(&k16)?;
    let v_d = stream.memcpy_stod(&v16)?;
    let mut o_d = stream.alloc_zeros::<f32>(s * d)?;
    let mut s1_d = stream.alloc_zeros::<f32>(s * s)?;
    let mut p_d = stream.alloc_zeros::<f32>(s * s)?;
    let mut p16_d = stream.alloc_zeros::<f16>(s * s)?;
    unsafe {
        cublas_attn_chain_once(
            &blas, &stream, &f_softmax, &f_cast, &q_d, &k_d, &v_d, &mut o_d, &mut s1_d, &mut p_d,
            &mut p16_d, s, d,
        )?;
    }
    stream.synchronize()?;
    Ok(stream.memcpy_dtov(&o_d)?)
}

/// Time the cuBLAS unfused attention chain (single head): `iters` resident chain runs bracketed by one
/// sync after a warm-up — the same timing shape Mercury's flash benches use, so the ratio is same-run
/// apples-to-apples. Scratch (S1, P, P16) is allocated once and reused. Returns seconds per chain.
pub fn time_cublas_attn_chain(
    g: &mut Gpu,
    s: usize,
    d: usize,
    _scale: f32, // dummy data ⇒ scale doesn't affect timing
    iters: u32,
) -> Result<f64, PeerError> {
    let f_softmax = g.function("norm", crate::ptx_norm::norm_ptx(), "softmax")?;
    let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
    let blas = CudaBlas::new(g.stream.clone())?;
    let stream = g.stream.clone();
    let q_d = stream.memcpy_stod(&vec![f16::from_f32(0.01); s * d])?;
    let k_d = stream.memcpy_stod(&vec![f16::from_f32(0.01); s * d])?;
    let v_d = stream.memcpy_stod(&vec![f16::from_f32(0.01); s * d])?;
    let mut o_d = stream.alloc_zeros::<f32>(s * d)?;
    let mut s1_d = stream.alloc_zeros::<f32>(s * s)?;
    let mut p_d = stream.alloc_zeros::<f32>(s * s)?;
    let mut p16_d = stream.alloc_zeros::<f16>(s * s)?;
    let mut once = || -> Result<(), PeerError> {
        unsafe {
            cublas_attn_chain_once(
                &blas, &stream, &f_softmax, &f_cast, &q_d, &k_d, &v_d, &mut o_d, &mut s1_d, &mut p_d,
                &mut p16_d, s, d,
            )
        }
    };
    once()?; // warm up
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        once()?;
    }
    stream.synchronize()?;
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
    /// Launch config matched to `f_flash` (untiled vs SMEM-tiled), resolved once by `gpu::flash_plan`.
    flash_cfg: LaunchConfig,
    /// Tensor-core flash (`flash_d64_w`) + cfg — `Some` when `gpu::wmma_flash_applies`. Kept identical
    /// to `ResidentLayerF16` so the flash stays *common* to both stacks and the measured gap is purely
    /// cuBLAS-vs-WMMA GEMM + epilogue fusion, not a difference in the attention kernel.
    f_flash_w: Option<(CudaFunction, LaunchConfig)>,
    f_silu: CudaFunction,
    f_vadd: CudaFunction,
    /// Multi-head layout shims — identical to `ResidentLayerF16`'s, so attention stays *common* to both
    /// stacks at `heads > 1` too (forward f32 `[S,H·dh]`→f16 `[H,S,dh]`, inverse f32 back).
    f_qkv_trans: CudaFunction,
    f_attn_trans: CudaFunction,
    wq: CudaSlice<f16>,
    wk: CudaSlice<f16>,
    wv: CudaSlice<f16>,
    wo: CudaSlice<f16>,
    w1: CudaSlice<f16>,
    w2: CudaSlice<f16>,
    s: usize,
    d: usize,
    heads: usize,
    dh: usize,
    dff: usize,
    eps: f32,
}

impl CublasChainLayer {
    /// Upload the weights (narrowed to f16) and preload the glue kernels — mirrors
    /// [`ResidentLayerF16::new`]'s shape constraints (S,D,Dff multiples of 64; `d` a supported flash head
    /// dim) and reuses the *same* kernel keys (`norm`/`cast`/`flash`/`vmath`/`vadd`) so those kernels are
    /// literally identical between the two stacks — only the GEMM differs.
    /// Single-head — the original API, unchanged; delegates to [`new_mha`](Self::new_mha) with `heads=1`.
    pub fn new(
        g: &mut Gpu,
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
    ) -> Result<Self, PeerError> {
        Self::new_mha(g, w, s, d, dff, 1)
    }

    /// **Multi-head** cuBLAS-chain layer — the Tier-B peer to [`ResidentLayerF16::new_mha`]. `heads`
    /// attention heads of `dh = d/heads`; the six projections still run on cuBLAS, and attention reuses
    /// **the same** multi-head flash path as the Mercury layer (cast-transpose Q/K/V to head-major f16,
    /// `grid.y=heads` tensor-core flash, transpose back), so attention stays *common* to both stacks and
    /// the measured gap remains purely cuBLAS-vs-WMMA GEMM + epilogue fusion. `heads>1` requires the
    /// tensor-core flash (`dh=64`, `S≥512`); `heads==1` is the original single-head chain.
    pub fn new_mha(
        g: &mut Gpu,
        w: &TransformerWeights,
        s: usize,
        d: usize,
        dff: usize,
        heads: usize,
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
        assert!(heads >= 1 && d % heads == 0, "d={d} must be divisible by heads={heads}");
        let dh = d / heads;
        assert!(
            crate::ptx_flash::SUPPORTED_D.contains(&dh),
            "CublasChainLayer: head dim dh={dh} (d={d}/heads={heads}) unsupported by flash (need {:?})",
            crate::ptx_flash::SUPPORTED_D
        );
        assert!(
            heads == 1 || crate::gpu::wmma_flash_applies(dh, s),
            "multi-head chain needs the tensor-core flash: dh=64, S>=512, S%16==0 (got dh={dh}, S={s})"
        );
        let blas = CudaBlas::new(g.stream.clone())?;
        let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
        let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
        let f_qkv_trans = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "cast_transpose_qkv")?;
        let f_attn_trans = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "transpose_attn_out")?;
        let (flash_name, flash_cfg) = crate::gpu::flash_plan(dh, s);
        let f_flash = g.function("flash", crate::ptx_flash::flash_ptx(), &flash_name)?;
        let f_flash_w = if crate::gpu::wmma_flash_applies(dh, s) {
            let f = g.function("flash", crate::ptx_flash::flash_ptx(), crate::gpu::wmma_flash_entry(s))?;
            Some((f, crate::gpu::wmma_flash_cfg(s)))
        } else {
            None
        };
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
            flash_cfg,
            f_flash_w,
            f_silu,
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

    /// RMSNorm a `[rows, d]` f32 buffer (one warp per row) — Mercury's exact `rmsnorm` kernel.
    fn norm(&self, src: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.alloc_zeros::<f32>(rows * self.d)?;
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
        let mut dst = self.stream.alloc_zeros::<f16>(n)?;
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
        let mut c = self.stream.alloc_zeros::<f32>(m * n)?;
        unsafe { gemm_ex_nt_f16_f32out(&self.blas, &self.stream, a, b, &mut c, m, k, n)? };
        Ok(c)
    }

    /// Fused flash-attention over `[S,D]` Q/K/V → `[S,D]` f32, **identical to `ResidentLayerF16::run_attn`**
    /// so the attention is common to both stacks. Single-head (`heads==1`): Mercury's flash directly
    /// (tensor-core when `wmma_flash_applies`, else f32). Multi-head: cast-transpose Q/K/V to head-major
    /// f16, `grid.y=heads` tensor-core flash, transpose back. `scale = 1/√dh`.
    fn flash(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let scale = 1.0f32 / (self.dh as f32).sqrt();
        let ss = self.s as u32;
        if self.heads == 1 {
            let mut attn = self.stream.alloc_zeros::<f32>(self.s * self.d)?;
            if let Some((f_w, cfg_w)) = &self.f_flash_w {
                let q16 = self.cast16(q)?;
                let k16 = self.cast16(k)?;
                let v16 = self.cast16(v)?;
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
            let (f_w, _) = self.f_flash_w.as_ref().expect("multi-head requires the tensor-core flash");
            let q_hsd = self.cast_transpose(q)?;
            let k_hsd = self.cast_transpose(k)?;
            let v_hsd = self.cast_transpose(v)?;
            let mut attn_hsd = self.stream.alloc_zeros::<f32>(self.s * self.d)?;
            let cfg = LaunchConfig {
                grid_dim: ((self.s / 16) as u32, self.heads as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut bld = self.stream.launch_builder(f_w);
            bld.arg(&ss).arg(&scale).arg(&q_hsd).arg(&k_hsd).arg(&v_hsd).arg(&mut attn_hsd);
            unsafe { bld.launch(cfg)? };
            self.transpose_back(&attn_hsd)
        }
    }

    /// f32 `[S,H·dh]` → f16 `[H,S,dh]` (cast folded in) — the multi-head flash input layout shim
    /// (`ptx::HEAD_TRANSPOSE_PTX`), identical to `ResidentLayerF16`'s.
    fn cast_transpose(&self, src: &CudaSlice<f32>) -> Result<CudaSlice<f16>, DriverError> {
        let n = self.s * self.d;
        let mut dst = self.stream.alloc_zeros::<f16>(n)?;
        let (nn, dd, dhh, sdh) =
            (n as u32, self.d as u32, self.dh as u32, (self.s * self.dh) as u32);
        let mut b = self.stream.launch_builder(&self.f_qkv_trans);
        b.arg(&nn).arg(&dd).arg(&dhh).arg(&sdh).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// f32 `[H,S,dh]` → f32 `[S,H·dh]` — the multi-head flash output layout shim (`transpose_attn_out`).
    fn transpose_back(&self, src: &CudaSlice<f32>) -> Result<CudaSlice<f32>, DriverError> {
        let n = self.s * self.d;
        let mut dst = self.stream.alloc_zeros::<f32>(n)?;
        let (nn, dd, dhh, sdh) =
            (n as u32, self.d as u32, self.dh as u32, (self.s * self.dh) as u32);
        let mut b = self.stream.launch_builder(&self.f_attn_trans);
        b.arg(&nn).arg(&dd).arg(&dhh).arg(&sdh).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// Narrow a device `[S·D]` f32 buffer to f16 (the tensor-core flash input dtype) via `f_cast`.
    fn cast16(&self, src: &CudaSlice<f32>) -> Result<CudaSlice<f16>, DriverError> {
        let n = self.s * self.d;
        let mut dst = self.stream.alloc_zeros::<f16>(n)?;
        let nn = n as u32;
        let mut b = self.stream.launch_builder(&self.f_cast);
        b.arg(&nn).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// Separate residual add `out = a + b` — the kernel cuBLAS forces (Mercury folds it via wmma.load.c).
    fn vadd(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        let nn = n as u32;
        let mut bld = self.stream.launch_builder(&self.f_vadd);
        bld.arg(&nn).arg(a).arg(b).arg(&mut out);
        unsafe { bld.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(out)
    }

    /// Separate SiLU activation — the kernel cuBLAS forces (Mercury folds it into the up-proj store).
    fn silu(&self, src: &CudaSlice<f32>, n: usize) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
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

// ---------------------------------------------------------------------------------------------------
// Tier A — naive CUDA-C **W4A16** (int4 weight-only decode), compiled by NVRTC. The idiomatic kernel a
// programmer writes for 4-bit weight decode: one thread per output element, unpack the int4 weight to
// float on the fly (shift + mask + sign-extend, × the per-group scale), `acc += a·w`. No tiling, no
// shared memory, no tensor cores, no fused unpack. NVRTC on this toolkit-free box has no fp16 headers
// (even `#include <cuda_fp16.h>` fails — see the cuBLAS-chain note), so it dequantizes and accumulates
// in **float**; the result is checksum-cross-checked against Mercury within the fp16-vs-fp32 dequant
// gap. This is the M6 wide-win floor and — crucially — the honest M4 headline: there is **no robust
// library int4-decode GEMM** bindable through `cudarc` (cuBLASLt offers no general W4A16 decode), so
// the strongest *measurable* int4 peer on this box is this naive kernel, and M4 is a documented lead.
// ---------------------------------------------------------------------------------------------------

/// Naive **symmetric** W4A16: `C[M×N] = A·dequant(W)ᵀ`, `A` `[M,K]` f32, `Bq` packed int4 `[N,K/8]`
/// (8 nibbles/word, Marlin-**interleaved** `nibble_pos(j)=(j/2)*4+(j%2)*16`, offset-binary `u=q+8`), `S`
/// per-group f32 scales `[N,K/group]`. One thread per output, full K-loop with an on-the-fly unpack
/// (`w = (u-8)*scale`) — the "beat the hand-written int4 decode kernel" baseline. Reads the *same*
/// packed layout Mercury's kernel consumes, so the comparison is pure kernel quality on identical bytes.
const NAIVE_W4A16_CUDA: &str = r#"
extern "C" __global__ void naive_w4a16(int M, int N, int K, int group,
        const float* A, const unsigned* Bq, const float* S, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n (weight row)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m (activation row)
    if (row < M && col < N) {
        int KW = K >> 3;        // u32 words per row (8 nibbles/word)
        int KG = K / group;     // groups per row
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            unsigned word = Bq[col * KW + (k >> 3)];
            int j = k & 7;
            int pos = (j >> 1) * 4 + (j & 1) * 16;          // interleaved nibble position
            int u = (word >> pos) & 0xF;                    // unsigned nibble (offset-binary)
            float w = (float)(u - 8) * S[col * KG + k / group]; // dequant: (u-8)*scale
            acc += A[row * K + k] * w;
        }
        C[row * N + col] = acc;
    }
}
"#;

fn nvrtc_naive_w4a16_module(g: &Gpu) -> Result<Arc<CudaModule>, PeerError> {
    let opts = CompileOptions { arch: Some("compute_89"), ..Default::default() };
    let ptx = compile_ptx_with_opts(NAIVE_W4A16_CUDA, opts)?;
    Ok(g.ctx.load_module(ptx)?)
}

/// Run the naive CUDA-C W4A16 once and copy the result back — the peer correctness-gate entry. `qw`
/// must be the **symmetric** quant (this peer's nibbles are signed); its fp16 scales are widened to f32
/// for the NVRTC-no-fp16 kernel, so the peer's output differs from Mercury's f16-dequant only by the
/// ~2⁻¹¹ dequant precision gap (checksum-cross-checked, and gated against the same f64 reference).
pub fn nvrtc_naive_w4a16(
    g: &mut Gpu,
    a: &[f32],
    qw: &crate::ptx_int4::QuantWeight,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(qw.n, n, "weight N mismatch");
    assert_eq!(qw.k, k, "weight K mismatch");
    assert!(qw.signed, "naive_w4a16 peer expects the symmetric (signed) quant");
    let module = nvrtc_naive_w4a16_module(g)?;
    let f = module.load_function("naive_w4a16")?;
    let s_f32: Vec<f32> = qw.scales.iter().map(|x| x.to_f32()).collect();
    let a_d = g.stream.memcpy_stod(a)?;
    let bq_d = g.stream.memcpy_stod(&qw.packed)?;
    let s_d = g.stream.memcpy_stod(&s_f32)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk, gg) = (m as i32, n as i32, k as i32, qw.group as i32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&gg).arg(&a_d).arg(&bq_d).arg(&s_d).arg(&mut c_d);
    unsafe { bld.launch(naive_cfg(m, n))? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// Time the naive CUDA-C W4A16: `iters` resident launches bracketed by one sync after a warm-up — the
/// identical timing shape Mercury's W4A16 bench uses, so the ratio is same-run apples-to-apples. Dummy
/// buffers of the correct `[M,K]` / `[N,K/8]` / `[N,K/group]` sizes (values don't affect timing).
/// Returns seconds per launch.
pub fn time_nvrtc_naive_w4a16(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let module = nvrtc_naive_w4a16_module(g)?;
    let f = module.load_function("naive_w4a16")?;
    let a_d = g.stream.memcpy_stod(&vec![0.01f32; m * k])?;
    let bq_d = g.stream.memcpy_stod(&vec![0u32; n * (k / 8)])?;
    let s_d = g.stream.memcpy_stod(&vec![0.01f32; n * (k / group)])?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let (mm, nn, kk, gg) = (m as i32, n as i32, k as i32, group as i32);
    let cfg = naive_cfg(m, n);
    let mut launch = |g: &Gpu| -> Result<(), DriverError> {
        let mut bld = g.stream.launch_builder(&f);
        bld.arg(&mm).arg(&nn).arg(&kk).arg(&gg).arg(&a_d).arg(&bq_d).arg(&s_d).arg(&mut c_d);
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

// Tier A — int8 (W8A8) CUDA-C peers, compiled by NVRTC (the "beat the hand-written C int8" baselines).
//
// Two peers of escalating quality, both `C = A·Bᵀ` with `u8` activations × `i8` weights → `i32`
// (Mercury's quantized-nn.Linear contract, exact mod 2³²), so all three implementations compute the
// *identical* integer matrix and cross-check bit-for-bit:
//   * `naive_gemm_nt_int8` — one thread per output, scalar `(int)A·(int)B` chain. The idiomatic kernel
//     a programmer writes first; the M6 wide-win floor.
//   * `dp4a_gemm_nt_int8`  — one thread per output, but the K-loop uses the **`dp4a.u32.s32`** 4-way
//     byte dot-product (the SIMD int8 instruction a programmer reaches for next; mixed u8×s8 via inline
//     PTX, which the `__dp4a` C intrinsic doesn't expose). A much stronger hand-written baseline than
//     naive — the honest "beat the optimized C int8" bar short of a tensor-core library.
// Both NVRTC-compile to PTX and the driver JITs them to SASS exactly like Mercury's PTX, so the gap is
// pure kernel quality (tensor cores + fragment reuse vs none).
// ---------------------------------------------------------------------------------------------------

const NAIVE_GEMM_NT_INT8_CUDA: &str = r#"
extern "C" __global__ void naive_gemm_nt_int8(int M, int N, int K,
        const unsigned char* A, const signed char* B, int* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m index
    if (row < M && col < N) {
        int acc = 0;
        for (int k = 0; k < K; ++k)
            acc += (int)A[row * K + k] * (int)B[col * K + k];
        C[row * N + col] = acc;
    }
}
"#;

/// `dp4a.u32.s32` 4-way dot product: each step consumes 4 `u8` of A and 4 `s8` of B (packed as one
/// `int` each) and accumulates the four products into the `s32` accumulator in a single instruction —
/// the SIMD int8 primitive on Ada short of the tensor core. Mixed `u8×s8` is expressed via inline PTX
/// because the `__dp4a` C intrinsic only exposes the same-signedness forms. K must be a multiple of 4.
const DP4A_GEMM_NT_INT8_CUDA: &str = r#"
extern "C" __global__ void dp4a_gemm_nt_int8(int M, int N, int K,
        const int* A, const int* B, int* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m index
    if (row < M && col < N) {
        int acc = 0;
        int K4 = K >> 2;
        const int* a = A + row * K4;   // A reinterpreted as packed 4×u8 per int
        const int* b = B + col * K4;   // B reinterpreted as packed 4×s8 per int
        for (int k = 0; k < K4; ++k) {
            int av = a[k], bv = b[k];
            asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(acc) : "r"(av), "r"(bv));
        }
        C[row * N + col] = acc;
    }
}
"#;

fn nvrtc_int8_module(g: &Gpu, src: &str) -> Result<Arc<CudaModule>, PeerError> {
    let opts = CompileOptions {
        arch: Some("compute_89"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, opts)?;
    Ok(g.ctx.load_module(ptx)?)
}

/// Run the naive int8 CUDA-C GEMM once and copy the `i32` result back — the correctness-gate entry.
pub fn nvrtc_naive_gemm_nt_int8(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let module = nvrtc_int8_module(g, NAIVE_GEMM_NT_INT8_CUDA)?;
    let f = module.load_function("naive_gemm_nt_int8")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
    let (mm, nn, kk) = (m as i32, n as i32, k as i32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(naive_cfg(m, n))? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// Run the `dp4a` int8 CUDA-C GEMM once and copy the `i32` result back — the stronger-peer gate entry.
pub fn nvrtc_dp4a_gemm_nt_int8(
    g: &mut Gpu,
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(k % 4, 0, "dp4a peer needs K%4==0");
    let module = nvrtc_int8_module(g, DP4A_GEMM_NT_INT8_CUDA)?;
    let f = module.load_function("dp4a_gemm_nt_int8")?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
    let (mm, nn, kk) = (m as i32, n as i32, k as i32);
    let mut bld = g.stream.launch_builder(&f);
    bld.arg(&mm).arg(&nn).arg(&kk).arg(&a_d).arg(&b_d).arg(&mut c_d);
    unsafe { bld.launch(naive_cfg(m, n))? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// Time the naive int8 CUDA-C GEMM: `iters` resident launches bracketed by one sync, after a warm-up —
/// the identical timing shape Mercury's GEMM benches use, so the ratio is apples-to-apples. Sec/launch.
pub fn time_nvrtc_naive_gemm_nt_int8(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    time_int8_peer(g, NAIVE_GEMM_NT_INT8_CUDA, "naive_gemm_nt_int8", m, k, n, iters)
}

/// Time the `dp4a` int8 CUDA-C GEMM (same timing shape as the naive peer). Seconds per launch.
pub fn time_nvrtc_dp4a_gemm_nt_int8(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    time_int8_peer(g, DP4A_GEMM_NT_INT8_CUDA, "dp4a_gemm_nt_int8", m, k, n, iters)
}

/// Shared timing harness for the int8 CUDA-C peers: upload once (dummy bytes), `iters` resident
/// launches after a warm-up, one trailing sync. Both peers take the same `(M,N,K,A,B,C)` signature.
fn time_int8_peer(
    g: &mut Gpu,
    src: &str,
    entry: &str,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let module = nvrtc_int8_module(g, src)?;
    let f = module.load_function(entry)?;
    let a_d = g.stream.memcpy_stod(&vec![1u8; m * k])?;
    let b_d = g.stream.memcpy_stod(&vec![1i8; n * k])?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
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
// Tier B — cuBLAS **int8 IMMA** GEMM via `cublasGemmEx` (the gold-standard int8 peer; Mercury reports
// as a % of this). `CUDA_R_8I` data, `CUDA_R_32I` output, `CUBLAS_COMPUTE_32I` (the Ada int8 tensor
// cores). cuBLASLt's *safe* cudarc wrapper only impls `Matmul` for f32/f16/bf16 (no int8), and the raw
// IMMA path needs fiddly COL32/COL4 memory ordering — so `cublasGemmEx` is the robust binding here.
//
// **Signedness caveat (honesty law):** classic `cublasGemmEx` int8 is **s8×s8→s32**; there is no mixed
// `u8×s8` form (that lives only in cuBLASLt's specially-ordered IMMA). Mercury's contract is `u8×s8`.
// To make all three implementations compute the *identical* matrix for the bit-exact cross-check, the
// int8 peer comparison restricts **activations to `[0,127]`** (where the `u8` and `s8` reinterpretations
// coincide); weights keep the full `[-128,127]`. The tensor-core *work* is identical regardless of
// signedness, so the **timing** is a faithful int8-IMMA measurement; only the test data is range-bound.
// ---------------------------------------------------------------------------------------------------

/// Column-major transpose mapping for Mercury's row-major `C[M×N] = A[M×K]·B[N×K]ᵀ` on cuBLAS — the
/// int8 twin of [`cublas_nt_cfg`]/[`gemm_ex_nt_f16_f32out`]: `Cᵀ = B̌ᵀ·Ǎ`, so B is the first operand
/// transposed and A the second untransposed, with `m,n` swapped, `lda=ldb=K`, `ldc=N`. K (=lda=ldb) is
/// a multiple of 32 and N (=ldc) a multiple of 8 — both satisfy IMMA's multiple-of-4 leading-dim rule.
///
/// # Safety
/// `a_d`/`b_d` (i8) and `c_d` (i32) must be valid device buffers of length `m*k`, `n*k`, `m*n`; the
/// cuBLAS handle and stream must be live. The device-pointer guards are held across the call.
unsafe fn gemm_ex_nt_int8(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    a_d: &CudaSlice<i8>,
    b_d: &CudaSlice<i8>,
    c_d: &mut CudaSlice<i32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), PeerError> {
    let alpha: i32 = 1;
    let beta: i32 = 0;
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
        (&alpha) as *const i32 as *const _,
        bp as *const _,
        cudaDataType_t::CUDA_R_8I,
        k as i32, // lda: B̌ is K×N col-major
        ap as *const _,
        cudaDataType_t::CUDA_R_8I,
        k as i32, // ldb: Ǎ is K×M col-major
        (&beta) as *const i32 as *const _,
        cp as *mut _,
        cudaDataType_t::CUDA_R_32I,
        n as i32, // ldc: Cᵀ is N×M col-major
        cublasComputeType_t::CUBLAS_COMPUTE_32I,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    )?;
    Ok(())
}

/// Run cuBLAS int8 (`cublasGemmEx`, IMMA tensor cores) once and return the `i32` result — the
/// correctness-gate entry. `a` activations must be in `[0,127]` (see the signedness caveat above) so
/// the `s8×s8` cuBLAS computes the same matrix as Mercury's `u8×s8`; passed here as `i8`.
pub fn cublas_gemm_nt_int8(
    g: &mut Gpu,
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<i32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let blas = CudaBlas::new(g.stream.clone())?;
    let a_d = g.stream.memcpy_stod(a)?;
    let b_d = g.stream.memcpy_stod(b)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
    let stream = g.stream.clone();
    unsafe { gemm_ex_nt_int8(&blas, &stream, &a_d, &b_d, &mut c_d, m, k, n)? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// Time cuBLAS int8 GEMM: `iters` resident `cublasGemmEx` calls, one warm-up, one trailing sync —
/// matching the Mercury/NVRTC timing shape. Returns seconds per call.
pub fn time_cublas_gemm_nt_int8(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let blas = CudaBlas::new(g.stream.clone())?;
    let a_d = g.stream.memcpy_stod(&vec![1i8; m * k])?;
    let b_d = g.stream.memcpy_stod(&vec![1i8; n * k])?;
    let mut c_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
    let stream = g.stream.clone();
    unsafe { gemm_ex_nt_int8(&blas, &stream, &a_d, &b_d, &mut c_d, m, k, n)? }; // warm up
    g.stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { gemm_ex_nt_int8(&blas, &stream, &a_d, &b_d, &mut c_d, m, k, n)? };
    }
    g.stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}
