//! Honest GPU peer baselines — the scoreboard's Tier-A and Tier-B competitors.
//!
//! Wukong's GPU kernels are worthless as a *claim* until they are measured against the kernels a
//! real engineer would reach for **on the same GPU**. Comparing a Wukong GPU kernel against
//! C-on-the-CPU would show a trivial 50–100× and prove nothing but "a GPU beats a CPU" — that
//! comparison is forbidden as a headline (see the GPU plan §1). The fair, harder bars are:
//!
//! * **Tier A — naive CUDA-C** (`nvrtc_naive_gemm_nt`): the idiomatic kernel a programmer writes by
//!   hand, compiled at runtime by **NVRTC** and launched through the same driver as Wukong's PTX.
//!   This is the literal "beat C/C++/Rust on the GPU" — the GPU twin of beating scalar C on the CPU.
//! * **Tier B — cuBLAS** (`cublas_gemm_nt_f16`): NVIDIA's hand-tuned closed-source GEMM, the gold
//!   standard. Wukong is reported as a **% of cuBLAS**, same-run, same buffers.
//!
//! Both peers `dlopen` their redistributable DLLs (`nvrtc64_120_0.dll`, `cublas64_12.dll`) exactly
//! the way `cudarc` already `dlopen`s the driver (`nvcuda.dll`). So *building* this crate still needs
//! no CUDA toolkit; only *running these benches* needs the DLLs reachable on the loader path. On this
//! box they live in a git-ignored `tools/cuda-redist/` (see `peer_env_hint`), put on `PATH` by the
//! bench invocation. If they are absent the loader would `panic!`, so [`peers_available`] probes for
//! them under `catch_unwind` and the peer benches skip when they are missing — the same "green
//! without the hardware" discipline the GPU tests already follow. **A skip is not silent**: a run
//! that is *supposed* to have the peers sets `WUKONG_PEER_REQUIRED=1` and the skip becomes a
//! failure, and [`peer_probe`] names which library was unreachable (§3A P3).
//!
//! Correctness first (the plan's first law): every peer is cross-checked against the **same f64 CPU
//! reference** as Wukong's own kernels before any speed number counts, so a fast-but-wrong peer
//! can't flatter Wukong and a fast-but-wrong Wukong can't beat a correct peer.

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
    peer_probe(g).is_none()
}

/// Why the peer probe failed, or `None` when both peers are loadable. **§3A P3 — a sweep that cannot
/// find its peer must fail LOUDLY.** A caller that takes a `&mut Gpu` already *has* a live device, so
/// a `Some` here is never "this box has no GPU" (a legitimate skip): it is always "the device is fine
/// but the peer toolchain is unreachable", which is the case that must not report green having
/// measured nothing. The string names *which* library failed so the operator does not have to guess
/// between a missing NVRTC and a missing cuBLAS (they live in different redist wheels).
pub fn peer_probe(g: &mut Gpu) -> Option<&'static str> {
    use std::sync::OnceLock;
    static WHY: OnceLock<Option<String>> = OnceLock::new();
    WHY.get_or_init(|| {
        // NVRTC: compile a trivial program. cuBLAS: create a handle on the stream. Either touching a
        // missing DLL panics inside cudarc's loader; catch it so we report a reason instead of crashing.
        let stream = g.stream.clone();
        let nvrtc = std::panic::catch_unwind(|| {
            compile_ptx_with_opts("extern \"C\" __global__ void p(){}", CompileOptions::default())
                .map(|_| ())
                .map_err(|e| format!("{e:?}"))
        })
        .unwrap_or_else(|_| Err("loader panicked (DLL not found)".to_string()));
        let cublas = std::panic::catch_unwind(|| CudaBlas::new(stream).map(|_| ()).map_err(|e| format!("{e:?}")))
            .unwrap_or_else(|_| Err("loader panicked (DLL not found)".to_string()));
        match (nvrtc, cublas) {
            (Ok(()), Ok(())) => None,
            (Err(n), Err(c)) => Some(format!("NVRTC unavailable ({n}) and cuBLAS unavailable ({c})")),
            (Err(n), Ok(())) => Some(format!("NVRTC unavailable ({n})")),
            (Ok(()), Err(c)) => Some(format!("cuBLAS unavailable ({c})")),
        }
    })
    .as_deref()
}

// ---------------------------------------------------------------------------------------------------
// Tier A — naive CUDA-C GEMM, compiled by NVRTC (the "beat the hand-written C kernel" baseline).
// ---------------------------------------------------------------------------------------------------

/// The idiomatic one-thread-per-output-element GEMM a programmer writes first: no tiling, no shared
/// memory, no tensor cores. `C = A·Bᵀ` (the `nn.Linear` contract: `A` is `M×K` row-major, `B` is
/// `N×K` row-major), one fused-multiply-add chain per output. NVRTC compiles this CUDA-C to PTX at
/// runtime; the driver JITs it to SASS just like Wukong's own PTX, so the comparison is pure
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
/// one sync, after a warm-up launch — the identical timing shape Wukong's own GEMM benches use, so
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
/// the GPU twin of the scalar CPU baseline: NVRTC compiles it to PTX, the driver JITs it like Wukong's
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
/// same f32 buffers Wukong's flash is cross-checked against (`[H,S,D]`, head-major).
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
/// the identical timing shape Wukong's flash benches use, so the ratio is apples-to-apples. Seconds/launch.
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
// That HBM round-trip of the S×S scores is exactly what Wukong's *fused* flash never pays — so the gap
// the bench measures is the value of fusion (and grows with S, since S² dwarfs the S·D I/O). cuBLAS
// provides no softmax/cast, so — exactly as `CublasChainLayer` does for the layer — those two glue
// kernels are Wukong's own (identical in both stacks); the only thing being compared is fused-vs-not.
// A genuinely *fused* FA2-class CUDA-C peer is not buildable here: NVRTC on this toolkit-free box has no
// header search path at all (even `#include <cuda_fp16.h>` fails), so `nvcuda::wmma` can't be compiled
// (see the `nvrtc_wmma_probe` capability test). The cuBLAS chain is the strongest library peer available.
// ---------------------------------------------------------------------------------------------------

/// cuBLAS `O[M×N] = A[M×K]·B[K×N]` — the **NN** (no-transpose) product `P·V` needs — f16 in / f32 out
/// (`CUDA_R_16F` data, `CUDA_R_32F` C, `CUBLAS_COMPUTE_32F`), tensor cores via the default algorithm:
/// the same f32-accumulate / f32-store boundary Wukong's flash uses. Row-major→col-major identity for a
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
/// the timing-loop body: `S1 = (scale·Q)·Kᵀ` (cuBLAS NT) → `P = softmax_row(S1)` (Wukong's warp-per-row
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
    // (2) P[S,S] = softmax over each of the S rows — Wukong's exact warp-per-row softmax (cuBLAS has
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
/// cuBLAS with Wukong's softmax+cast as the glue cuBLAS can't provide. Cross-checked against the same
/// f64 oracle Wukong's flash is, so a transpose slip or wrong config shows as a gross mismatch.
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
/// sync after a warm-up — the same timing shape Wukong's flash benches use, so the ratio is same-run
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
// Tier B — cuBLAS fp16 GEMM (the gold-standard peer; Wukong reports as a % of this).
// ---------------------------------------------------------------------------------------------------

/// Build the column-major cuBLAS config that makes cuBLAS compute Wukong's **row-major** `C = A·Bᵀ`.
///
/// cuBLAS is column-major; Wukong is row-major. A row-major `C (M×N)` is a column-major `Cᵀ (N×M)`,
/// and `Cᵀ = (A·Bᵀ)ᵀ = B·Aᵀ`. Writing Wukong's row-major buffers as their column-major transposes
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
        lda: k as i32, // B̌ (Wukong B), K×N col-major
        ldb: k as i32, // Ǎ (Wukong A), K×M col-major
        beta: f16::from_f32(0.0),
        ldc: n as i32, // Cᵀ, N×M col-major
    }
}

/// Run cuBLAS fp16 (`cublasGemmEx`, `CUDA_R_16F` data, `CUBLAS_COMPUTE_32F` accumulate — tensor cores
/// via the default algorithm) once and return the result widened to f32. The correctness-gate entry.
/// Inputs are the same f32 buffers Wukong gets; we round to f16 on the host (the price cuBLAS's fp16
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
    // Per the mapping: first operand is Wukong's B (transposed), second is Wukong's A.
    unsafe { blas.gemm(cfg, &b_d, &a_d, &mut c_d)? };
    let c16: Vec<f16> = g.stream.memcpy_dtov(&c_d)?;
    Ok(c16.iter().map(|x| x.to_f32()).collect())
}

/// Time cuBLAS fp16 GEMM: `iters` resident calls on the stream, one warm-up, one trailing sync —
/// matching the Wukong/NVRTC timing shape. Returns seconds per call.
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
// for SiLU or residual-from-a-third-buffer). That non-fusion is exactly the cost Wukong's fused WMMA
// stack ([`ResidentLayerF16`]) folds away. To make the comparison about *fusion and GEMM quality only*,
// this peer reuses Wukong's identical norm/flash/cast/SiLU/vadd kernels (cuBLAS provides none of them);
// the sole substitution is cuBLAS for the six projections.
// ---------------------------------------------------------------------------------------------------

/// `cublasGemmEx`, **f16 in / f32 out** (`CUDA_R_16F` data, `CUDA_R_32F` C, `CUBLAS_COMPUTE_32F`): the
/// fragments multiply in fp16, accumulate in f32, and the result is **stored straight to f32** — the
/// identical dtype boundary as Wukong's WMMA GEMM, which also accumulates f32 and stores f32. This is
/// the fairness keystone: if cuBLAS stored f16 we would owe an f16→f32 cast kernel before the next
/// (f32) norm/flash stage that Wukong never pays, silently biasing the chain in Wukong's favour. With
/// f32 out, **both** stacks pay exactly one f32→f16 narrowing *before* each GEMM and nothing after.
///
/// Computes Wukong's row-major `C[M×N] = A[M×K]·B[N×K]ᵀ` (the `nn.Linear` contract) using the same
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
        cublasOperation_t::CUBLAS_OP_T, // B̌ transposed (Wukong's B, first operand)
        cublasOperation_t::CUBLAS_OP_N, // Ǎ untransposed (Wukong's A, second operand)
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

/// Time cuBLAS fp16 GEMM with an **f32 output** ([`gemm_ex_nt_f16_f32out`]): `iters` resident calls, one
/// warm-up, one trailing sync — the same timing shape as [`time_cublas_gemm_nt_f16`], but storing C as
/// f32. This is the **apples-to-apples** peer for Wukong's WMMA kernel, which also accumulates and stores
/// C as f32. The f16-out sibling writes C at half the width (33.5 MB vs 67 MB at 4096³), so on this
/// ~192 GB/s bus it moves *half* the epilogue write traffic Wukong pays — timing both isolates how much
/// of the Wukong-vs-cuBLAS gap is that C-dtype asymmetry rather than GEMM quality. Returns seconds/call.
pub fn time_cublas_gemm_nt_f16_f32out(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let blas = CudaBlas::new(g.stream.clone())?;
    let a_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.01); m * k])?;
    let b_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.01); n * k])?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let stream = g.stream.clone();
    unsafe { gemm_ex_nt_f16_f32out(&blas, &stream, &a_d, &b_d, &mut c_d, m, k, n)? }; // warm up
    g.stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { gemm_ex_nt_f16_f32out(&blas, &stream, &a_d, &b_d, &mut c_d, m, k, n)? };
    }
    g.stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

/// The **cuBLAS-GEMM call-chain transformer layer** — Tier-B peer to [`ResidentLayerF16`]. Same resident
/// f16 weights, same `[S,D]` f32 activation contract, same pre-norm encoder math; the six projections
/// (Q/K/V/O, W1, W2) run on cuBLAS ([`gemm_ex_nt_f16_f32out`]) and the skip-connection adds + SiLU run as
/// **separate** kernels (cuBLAS cannot fuse them). Structurally this is exactly Wukong's *unfused* path
/// [`ResidentLayerF16::forward_device_unfused`] with cuBLAS substituted for the WMMA GEMM — so the
/// measured gap against Wukong's *fused* [`forward_device`](ResidentLayerF16::forward_device) is purely
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
    /// cuBLAS-vs-WMMA GEMM + epilogue fusion, not a difference in the attention kernel. Resolved
    /// through [`Self::flash_plan_w`] — the *same* `gpu::wmma_flash_plan` seam the Wukong layer uses.
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
    /// **The tensor-core flash the chain runs — resolved through the *same* seam as the Wukong layer.**
    ///
    /// The whole point of this peer is that attention is *common* to both stacks, so the measured gap is
    /// purely cuBLAS-vs-WMMA GEMM + epilogue fusion. That only holds if the chain launches the kernel
    /// `ResidentLayerF16` launches, and `ResidentLayerF16::new_mha` resolves it with
    /// [`gpu::wmma_flash_plan`](crate::gpu) — which at `S >= 4096` routes to the warp-specialized
    /// `flash_d64_ws` / `flash_d128_ws3_lm` (2 warps/CTA, grid `ceil((S/16)/2)`), not to
    /// `wmma_flash_entry`'s `flash_d*_mp` (1 warp, grid `S/16`). Pairing `wmma_flash_entry` with
    /// `wmma_flash_cfg` here — as this did — gave the *peer* the mp kernel while Wukong ran ws, so the
    /// S=4096 GPT-2-layer headline carried an attention-kernel delta (`flash_ws_vs_mp` measured ws/mp
    /// 0.944× at S=4096) inside a number documented as GEMM-only. Entry and config also *must* travel
    /// together: the ws family is a 64-thread named-barrier kernel and would hang on a 32-thread launch.
    fn flash_plan_w(dh: usize, s: usize, heads: usize) -> (&'static str, LaunchConfig) {
        crate::gpu::wmma_flash_plan(dh, s, heads)
    }

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
    /// **the same** multi-head flash path as the Wukong layer (cast-transpose Q/K/V to head-major f16,
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
            "multi-head chain needs the tensor-core flash: dh=64 or 128, S>=512, S%16==0 (got dh={dh}, S={s})"
        );
        let blas = CudaBlas::new(g.stream.clone())?;
        let f_norm = g.function("norm", crate::ptx_norm::norm_ptx(), "rmsnorm")?;
        let f_cast = g.function("cast", crate::ptx::CAST_F32_F16, "cast_f32_f16")?;
        let f_qkv_trans = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "cast_transpose_qkv")?;
        let f_attn_trans = g.function("htrans", crate::ptx::HEAD_TRANSPOSE_PTX, "transpose_attn_out")?;
        let (flash_name, flash_cfg) = crate::gpu::flash_plan(dh, s);
        let f_flash = g.function("flash", crate::ptx_flash::flash_ptx(), &flash_name)?;
        let f_flash_w = if crate::gpu::wmma_flash_applies(dh, s) {
            let (entry, cfg) = Self::flash_plan_w(dh, s, heads);
            let f = g.function("flash", crate::ptx_flash::flash_ptx(), entry)?;
            Some((f, cfg))
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

    /// RMSNorm a `[rows, d]` f32 buffer (one warp per row) — Wukong's exact `rmsnorm` kernel.
    fn norm(&self, src: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>, DriverError> {
        let mut out = self.stream.alloc_zeros::<f32>(rows * self.d)?;
        let (r, c) = (rows as u32, self.d as u32);
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut bld = self.stream.launch_builder(&self.f_norm);
        bld.arg(&r).arg(&c).arg(&self.eps).arg(src).arg(&mut out);
        unsafe { bld.launch(cfg)? };
        Ok(out)
    }

    /// device f32 → device f16 narrowing (the stage boundary before each cuBLAS GEMM) — Wukong's
    /// exact `cast_f32_f16` kernel, so the f16 inputs both stacks feed their GEMMs are bit-identical.
    fn cast(&self, src: &CudaSlice<f32>, n: usize) -> Result<CudaSlice<f16>, DriverError> {
        let mut dst = self.stream.alloc_zeros::<f16>(n)?;
        let nn = n as u32;
        let mut b = self.stream.launch_builder(&self.f_cast);
        b.arg(&nn).arg(src).arg(&mut dst);
        unsafe { b.launch(LaunchConfig::for_num_elems(nn))? };
        Ok(dst)
    }

    /// `C[M×N] = A·Bᵀ` on cuBLAS (f16-in/f32-out) — the one substitution vs Wukong's WMMA path.
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
    /// so the attention is common to both stacks. Single-head (`heads==1`): Wukong's flash directly
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
            // Entry AND config both come from `flash_plan_w` (grid.y = heads is already folded in by
            // `wmma_flash_plan`): a hand-built 1-warp/`S/16` config here would silently pair the
            // warp-specialized S>=4096 entry with a 32-thread launch — a different kernel from the one
            // `ResidentLayerF16` runs, and a hang for the 64-thread named-barrier ws family.
            let (f_w, cfg) = self
                .f_flash_w
                .as_ref()
                .expect("multi-head requires the tensor-core flash");
            let q_hsd = self.cast_transpose(q)?;
            let k_hsd = self.cast_transpose(k)?;
            let v_hsd = self.cast_transpose(v)?;
            let mut attn_hsd = self.stream.alloc_zeros::<f32>(self.s * self.d)?;
            let cfg = *cfg;
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

    /// Separate residual add `out = a + b` — the kernel cuBLAS forces (Wukong folds it via wmma.load.c).
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

    /// Separate SiLU activation — the kernel cuBLAS forces (Wukong folds it into the up-proj store).
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

/// A stack of N [`CublasChainLayer`]s — the cuBLAS-call-chain counterpart to Wukong's whole-model
/// `ResidentModelF16`, for the M13 "**beat the library-call-chain stack end-to-end**" claim. Each layer
/// keeps its own f16 weights and cuBLAS handle; [`forward_device`](Self::forward_device) chains them on
/// device buffers exactly like the resident model (one layer's output is the next's input). The contrast
/// the depth bench draws out: Wukong's resident model keeps every activation on-device across all N
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
// in **float**; the result is checksum-cross-checked against Wukong within the fp16-vs-fp32 dequant
// gap. This is the M6 wide-win floor and — crucially — the honest M4 headline: there is **no robust
// library int4-decode GEMM** bindable through `cudarc` (cuBLASLt offers no general W4A16 decode), so
// the strongest *measurable* int4 peer on this box is this naive kernel, and M4 is a documented lead.
// ---------------------------------------------------------------------------------------------------

/// Naive **symmetric** W4A16: `C[M×N] = A·dequant(W)ᵀ`, `A` `[M,K]` f32, `Bq` packed int4 `[N,K/8]`
/// (8 nibbles/word, Marlin-**interleaved** `nibble_pos(j)=(j/2)*4+(j%2)*16`, offset-binary `u=q+8`), `S`
/// per-group f32 scales `[N,K/group]`. One thread per output, full K-loop with an on-the-fly unpack
/// (`w = (u-8)*scale`) — the "beat the hand-written int4 decode kernel" baseline. Reads the *same*
/// packed layout Wukong's kernel consumes, so the comparison is pure kernel quality on identical bytes.
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
/// for the NVRTC-no-fp16 kernel, so the peer's output differs from Wukong's f16-dequant only by the
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
/// identical timing shape Wukong's W4A16 bench uses, so the ratio is same-run apples-to-apples. Dummy
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

// Tier A — naive CUDA-C conv2d, compiled by NVRTC (the "beat the hand-written C conv" baseline).
// ---------------------------------------------------------------------------------------------------

/// The idiomatic conv a programmer writes first: **one thread per output element** `(k,p,q)`, looping
/// the whole `c,r,s` window with a fused-multiply-add chain, no shared memory, no tiling, no tensor
/// cores — the GPU twin of Wukong's *old* naive PTX conv. Single batch, stride 1, no padding (valid
/// cross-correlation): input `X[C,H,W]`, weights `W[K,C,R,S]`, output `O[K,P,Q]` with `P=H-R+1`,
/// `Q=W-S+1`. NVRTC compiles this CUDA-C to PTX at runtime; the driver JITs it to SASS just like
/// Wukong's own PTX, so the comparison is pure kernel quality (tiling/SMEM/tensor-cores vs none).
const NAIVE_CONV_CUDA: &str = r#"
extern "C" __global__ void naive_conv(int C, int H, int W, int K, int R, int S, int P, int Q,
                                      const float* X, const float* Wt, float* O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // linear (k,p,q)
    int total = K * P * Q;
    if (idx >= total) return;
    int q = idx % Q;
    int t = idx / Q;
    int p = t % P;
    int k = t / P;
    float acc = 0.0f;
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s) {
                int ih = p + r, iw = q + s;
                acc += X[(c * H + ih) * W + iw] * Wt[((k * C + c) * R + r) * S + s];
            }
    O[idx] = acc;
}
"#;

fn nvrtc_naive_conv_module(g: &Gpu) -> Result<Arc<CudaModule>, PeerError> {
    let opts = CompileOptions {
        arch: Some("compute_89"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(NAIVE_CONV_CUDA, opts)?;
    Ok(g.ctx.load_module(ptx)?)
}

fn naive_conv_cfg(k: usize, p: usize, q: usize) -> LaunchConfig {
    let total = (k * p * q) as u32;
    LaunchConfig {
        grid_dim: (total.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Run the naive CUDA-C conv once and copy `O` back — the correctness-gate entry. `X`/`W` are the same
/// f32 buffers Wukong's conv is cross-checked against (`[C,H,W]` / `[K,C,R,S]`).
#[allow(clippy::too_many_arguments)]
pub fn nvrtc_naive_conv(
    g: &mut Gpu,
    x: &[f32],
    w: &[f32],
    c: usize,
    h: usize,
    width: usize,
    k: usize,
    r: usize,
    s: usize,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(x.len(), c * h * width);
    assert_eq!(w.len(), k * c * r * s);
    let (p, q) = (h - r + 1, width - s + 1);
    let module = nvrtc_naive_conv_module(g)?;
    let f = module.load_function("naive_conv")?;
    let x_d = g.stream.memcpy_stod(x)?;
    let w_d = g.stream.memcpy_stod(w)?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; k * p * q])?;
    let dims = [c, h, width, k, r, s, p, q].map(|v| v as i32);
    let mut bld = g.stream.launch_builder(&f);
    for d in &dims {
        bld.arg(d);
    }
    bld.arg(&x_d).arg(&w_d).arg(&mut o_d);
    unsafe { bld.launch(naive_conv_cfg(k, p, q))? };
    Ok(g.stream.memcpy_dtov(&o_d)?)
}

/// Time the naive CUDA-C conv: `iters` resident launches bracketed by one sync, after a warm-up — the
/// identical timing shape Wukong's own conv bench uses, so the ratio is apples-to-apples. Seconds/launch.
#[allow(clippy::too_many_arguments)]
pub fn time_nvrtc_naive_conv(
    g: &mut Gpu,
    c: usize,
    h: usize,
    width: usize,
    k: usize,
    r: usize,
    s: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let (p, q) = (h - r + 1, width - s + 1);
    let module = nvrtc_naive_conv_module(g)?;
    let f = module.load_function("naive_conv")?;
    let x_d = g.stream.memcpy_stod(&vec![0.01f32; c * h * width])?;
    let w_d = g.stream.memcpy_stod(&vec![0.01f32; k * c * r * s])?;
    let mut o_d = g.stream.memcpy_stod(&vec![0f32; k * p * q])?;
    let dims = [c, h, width, k, r, s, p, q].map(|v| v as i32);
    let cfg = naive_conv_cfg(k, p, q);
    let mut launch = |g: &Gpu| -> Result<(), DriverError> {
        let mut bld = g.stream.launch_builder(&f);
        for d in &dims {
            bld.arg(d);
        }
        bld.arg(&x_d).arg(&w_d).arg(&mut o_d);
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

/// `2·K·P·Q·C·R·S` — conv2d FLOPs (each output is a `C·R·S` MAC reduction), for turning
/// seconds/launch into FLOP/s.
#[allow(clippy::too_many_arguments)]
pub fn conv_flop(c: usize, h: usize, width: usize, k: usize, r: usize, s: usize) -> f64 {
    let (p, q) = (h - r + 1, width - s + 1);
    2.0 * k as f64 * p as f64 * q as f64 * c as f64 * r as f64 * s as f64
}

// Tier A — int8 (W8A8) CUDA-C peers, compiled by NVRTC (the "beat the hand-written C int8" baselines).
//
// Two peers of escalating quality, both `C = A·Bᵀ` with `u8` activations × `i8` weights → `i32`
// (Wukong's quantized-nn.Linear contract, exact mod 2³²), so all three implementations compute the
// *identical* integer matrix and cross-check bit-for-bit:
//   * `naive_gemm_nt_int8` — one thread per output, scalar `(int)A·(int)B` chain. The idiomatic kernel
//     a programmer writes first; the M6 wide-win floor.
//   * `dp4a_gemm_nt_int8`  — one thread per output, but the K-loop uses the **`dp4a.u32.s32`** 4-way
//     byte dot-product (the SIMD int8 instruction a programmer reaches for next; mixed u8×s8 via inline
//     PTX, which the `__dp4a` C intrinsic doesn't expose). A much stronger hand-written baseline than
//     naive — the honest "beat the optimized C int8" bar short of a tensor-core library.
// Both NVRTC-compile to PTX and the driver JITs them to SASS exactly like Wukong's PTX, so the gap is
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
/// the identical timing shape Wukong's GEMM benches use, so the ratio is apples-to-apples. Sec/launch.
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
// Tier B — cuBLAS **int8 IMMA** GEMM via `cublasGemmEx` (the gold-standard int8 peer; Wukong reports
// as a % of this). `CUDA_R_8I` data, `CUDA_R_32I` output, `CUBLAS_COMPUTE_32I` (the Ada int8 tensor
// cores). cuBLASLt's *safe* cudarc wrapper only impls `Matmul` for f32/f16/bf16 (no int8), and the raw
// IMMA path needs fiddly COL32/COL4 memory ordering — so `cublasGemmEx` is the robust binding here.
//
// **Signedness caveat (honesty law):** classic `cublasGemmEx` int8 is **s8×s8→s32**; there is no mixed
// `u8×s8` form (that lives only in cuBLASLt's specially-ordered IMMA). Wukong's contract is `u8×s8`.
// To make all three implementations compute the *identical* matrix for the bit-exact cross-check, the
// int8 peer comparison restricts **activations to `[0,127]`** (where the `u8` and `s8` reinterpretations
// coincide); weights keep the full `[-128,127]`. The tensor-core *work* is identical regardless of
// signedness, so the **timing** is a faithful int8-IMMA measurement; only the test data is range-bound.
// ---------------------------------------------------------------------------------------------------

/// Column-major transpose mapping for Wukong's row-major `C[M×N] = A[M×K]·B[N×K]ᵀ` on cuBLAS — the
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
        cublasOperation_t::CUBLAS_OP_T, // B̌ transposed (Wukong's B, first operand)
        cublasOperation_t::CUBLAS_OP_N, // Ǎ untransposed (Wukong's A, second operand)
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
/// the `s8×s8` cuBLAS computes the same matrix as Wukong's `u8×s8`; passed here as `i8`.
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
/// matching the Wukong/NVRTC timing shape. Returns seconds per call.
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

// ---------------------------------------------------------------------------------------------------
// Tier B — cuBLASLt **fp8 (E4M3) GEMM** — the gold-standard fp8 peer (M2). cuBLASLt is the *only*
// cuBLAS surface with an fp8 matmul; classic `cublasGemmEx` has none, and cudarc's *safe* `Matmul<T>`
// wrapper is f32/f16/bf16-only — so this peer drives the raw `cublaslt::{sys,result}` layer directly
// (the "raw-sys cuBLASLt E4M3" binding the GPU plan calls for). It dlopens `cublasLt64_12.dll` exactly
// like the cublas/nvrtc peers, so building still needs no toolkit and the benches skip cleanly when the
// redist DLLs are absent.
//
// **Layout mapping — identical to the f16 [`gemm_ex_nt_f16_f32out`].** Wukong computes row-major
// `C[M×N] = A[M×K]·B[N×K]ᵀ`. cuBLAS(Lt) is column-major, so we compute `Cᵀ[N×M] = B̌ᵀ·Ǎ`: the first
// operand is Wukong's **B** transposed (`OP_T`), the second is Wukong's **A** untransposed (`OP_N`),
// output dims swapped. Crucially `transa=T, transb=N` ("TN") is **also the only transpose combo
// cuBLASLt's fp8 kernels accept** — the fair NT mapping and the fp8 hardware constraint coincide. A and
// B are E4M3 (the byte-identical operands Wukong's kernel consumes, via the same `f32_to_e4m3`); C/D
// are f32 — Wukong also accumulates in f32 and stores f32, the identical dtype boundary (the fairness
// keystone of [`gemm_ex_nt_f16_f32out`]). A/B scale factors are device `1.0` (Wukong's fp8 is
// unscaled); `FAST_ACCUM` is left default (the higher-precision split accumulation, matching Wukong's
// full f32 accumulate — so the peer lands inside the same `c·√K·ε` tolerance gate).
// ---------------------------------------------------------------------------------------------------

use core::ffi::c_void;
use cudarc::cublaslt::result as cublaslt_result;
use cudarc::cublaslt::sys as cublaslt_sys;

/// cuBLASLt fp8 workspace — 32 MiB sits comfortably above any Ada fp8 algo's requirement.
const FP8_LT_WORKSPACE: usize = 32 * 1024 * 1024;

/// A built, reusable cuBLASLt fp8 (E4M3·E4M3 → f32) matmul plan: handle + descriptor + the three
/// matrix layouts + the heuristic-chosen algorithm + workspace + the (1.0) A/B scale buffers. Built
/// once — the heuristic search is host-only work we keep *out* of any timing loop — and re-run per call
/// on fresh device buffers. `Drop` tears down the sys objects so an early `?` cannot leak them.
struct Fp8LtPlan {
    handle: cublaslt_sys::cublasLtHandle_t,
    desc: cublaslt_sys::cublasLtMatmulDesc_t,
    a_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    b_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    cd_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    pref: cublaslt_sys::cublasLtMatmulPreference_t,
    algo: cublaslt_sys::cublasLtMatmulAlgo_t,
    workspace: CudaSlice<u8>,
    // The 1.0 scales are kept alive for the plan's life: their device addresses are baked into `desc`.
    _scale_a: CudaSlice<f32>,
    _scale_b: CudaSlice<f32>,
}

impl Drop for Fp8LtPlan {
    fn drop(&mut self) {
        unsafe {
            let _ = cublaslt_result::destroy_matmul_pref(self.pref);
            let _ = cublaslt_result::destroy_matrix_layout(self.cd_layout);
            let _ = cublaslt_result::destroy_matrix_layout(self.b_layout);
            let _ = cublaslt_result::destroy_matrix_layout(self.a_layout);
            let _ = cublaslt_result::destroy_matmul_desc(self.desc);
            let _ = cublaslt_result::destroy_handle(self.handle);
        }
    }
}

impl Fp8LtPlan {
    /// Build the plan for a row-major `C[M×N] = A[M×K]·B[N×K]ᵀ` E4M3 GEMM. Errors propagate as
    /// `PeerError` — e.g. the heuristic finding no fp8 algo for this shape/device returns
    /// `CUBLAS_STATUS_NOT_SUPPORTED`, and the bench then *honestly* reports "no cuBLASLt fp8 peer"
    /// rather than a fabricated ratio.
    fn new(g: &mut Gpu, m: usize, k: usize, n: usize) -> Result<Self, PeerError> {
        // E4M3 leading dims are 1 byte; cuBLASLt wants 16-byte-aligned lda/ldb ⇒ K%16, and the f32 C
        // ld=N must be 4-element (16-byte) aligned ⇒ N%4. Wukong's m16n8k32 tiles satisfy both.
        assert!(k % 16 == 0, "cuBLASLt fp8 needs K%16==0 (got K={k})");
        assert!(n % 4 == 0, "cuBLASLt fp8 needs N%4==0 (got N={n})");
        let stream = g.stream.clone();

        let handle = cublaslt_result::create_handle()?;
        let scale_a = stream.memcpy_stod(&[1.0f32])?;
        let scale_b = stream.memcpy_stod(&[1.0f32])?;
        let workspace = stream.alloc_zeros::<u8>(FP8_LT_WORKSPACE)?;

        // Descriptor: f32 compute, f32 scale.
        let desc = cublaslt_result::create_matmul_desc(
            cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublaslt_sys::cudaDataType_t::CUDA_R_32F,
        )?;
        unsafe {
            // transa = T (Wukong's B, first operand), transb = N (Wukong's A). 1==T, 0==N as i32.
            let op_t: i32 = 1;
            let op_n: i32 = 0;
            cublaslt_result::set_matmul_desc_attribute(
                desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&op_t) as *const i32 as *const c_void,
                core::mem::size_of::<i32>(),
            )?;
            cublaslt_result::set_matmul_desc_attribute(
                desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                (&op_n) as *const i32 as *const c_void,
                core::mem::size_of::<i32>(),
            )?;
            // A/B scale device pointers (both = 1.0). The address is stable across the move into Self.
            let (sa, _ga) = scale_a.device_ptr(&stream);
            let (sb, _gb) = scale_b.device_ptr(&stream);
            cublaslt_result::set_matmul_desc_attribute(
                desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                (&sa) as *const _ as *const c_void,
                core::mem::size_of_val(&sa),
            )?;
            cublaslt_result::set_matmul_desc_attribute(
                desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                (&sb) as *const _ as *const c_void,
                core::mem::size_of_val(&sb),
            )?;
        }

        // Layouts (column-major, matrices as stored). cuBLAS A = Wukong B stored [K,N] ld=K E4M3;
        // cuBLAS B = Wukong A stored [K,M] ld=K E4M3; C/D = Wukong C stored [N,M] ld=N f32.
        let a_layout = cublaslt_result::create_matrix_layout(
            cublaslt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            n as u64,
            k as i64,
        )?;
        let b_layout = cublaslt_result::create_matrix_layout(
            cublaslt_sys::cudaDataType_t::CUDA_R_8F_E4M3,
            k as u64,
            m as u64,
            k as i64,
        )?;
        let cd_layout = cublaslt_result::create_matrix_layout(
            cublaslt_sys::cudaDataType_t::CUDA_R_32F,
            n as u64,
            m as u64,
            n as i64,
        )?;

        let pref = cublaslt_result::create_matmul_pref()?;
        unsafe {
            let ws = FP8_LT_WORKSPACE;
            cublaslt_result::set_matmul_pref_attribute(
                pref,
                cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws) as *const usize as *const c_void,
                core::mem::size_of::<usize>(),
            )?;
        }

        // Heuristic: the single fastest algo for this fp8 config (A,B,C,D layouts).
        let heuristic = unsafe {
            cublaslt_result::get_matmul_algo_heuristic(
                handle, desc, a_layout, b_layout, cd_layout, cd_layout, pref,
            )?
        };

        Ok(Self {
            handle,
            desc,
            a_layout,
            b_layout,
            cd_layout,
            pref,
            algo: heuristic.algo,
            workspace,
            _scale_a: scale_a,
            _scale_b: scale_b,
        })
    }

    /// One `cublasLtMatmul` on resident device buffers. `b_d` is Wukong's **B** (cuBLAS operand A,
    /// `OP_T`), `a_d` is Wukong's **A** (cuBLAS operand B, `OP_N`), `c_d` is Wukong's f32 C. Nothing
    /// is synced here — the caller owns the warm-up/sync discipline.
    ///
    /// # Safety
    /// Buffers must be the documented E4M3/E4M3/f32 sizes; the plan's sys objects must be live.
    unsafe fn run(
        &self,
        stream: &Arc<CudaStream>,
        b_d: &CudaSlice<u8>,
        a_d: &CudaSlice<u8>,
        c_d: &mut CudaSlice<f32>,
    ) -> Result<(), PeerError> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let (bp, _rb) = b_d.device_ptr(stream); // cuBLAS operand A
        let (ap, _ra) = a_d.device_ptr(stream); // cuBLAS operand B
        let (cp, _rc) = c_d.device_ptr_mut(stream);
        let (wp, _rw) = self.workspace.device_ptr(stream);
        cublaslt_result::matmul(
            self.handle,
            self.desc,
            (&alpha) as *const f32 as *const c_void,
            (&beta) as *const f32 as *const c_void,
            bp as *const c_void,
            self.a_layout,
            ap as *const c_void,
            self.b_layout,
            cp as *const c_void,
            self.cd_layout,
            cp as *mut c_void,
            self.cd_layout,
            (&self.algo) as *const _,
            wp as *mut c_void,
            FP8_LT_WORKSPACE,
            stream.cu_stream() as *mut _,
        )?;
        Ok(())
    }
}

/// Run cuBLASLt fp8 (E4M3·E4M3 → f32, tensor cores) once and return the f32 result — the
/// correctness-gate entry. Host f32 in, **rounded to E4M3 on the host with Wukong's own
/// `f32_to_e4m3`** so the peer multiplies the byte-identical operands Wukong's kernel does; cross-
/// checked against the same E4M3-rounded f64 reference, so a transpose slip shows as a gross miss.
pub fn cublaslt_gemm_nt_fp8_e4m3(
    g: &mut Gpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, PeerError> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let a8: Vec<u8> = a.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
    let b8: Vec<u8> = b.iter().map(|&x| crate::ptx_fp8::f32_to_e4m3(x)).collect();
    let a8_d = g.stream.memcpy_stod(&a8)?;
    let b8_d = g.stream.memcpy_stod(&b8)?;
    let mut c_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let plan = Fp8LtPlan::new(g, m, k, n)?;
    let stream = g.stream.clone();
    unsafe { plan.run(&stream, &b8_d, &a8_d, &mut c_d)? };
    Ok(g.stream.memcpy_dtov(&c_d)?)
}

/// Time cuBLASLt fp8 GEMM: build the plan once (the heuristic search is host-only), then `iters`
/// resident `cublasLtMatmul` calls, one warm-up, one trailing sync — the same timing shape as
/// [`time_cublas_gemm_nt_f16`]. Returns **seconds per call**. The caller passes its already-resident
/// E4M3 A/B (so the bench feeds the *identical* bytes it gave Wukong) and an f32 C of length `m*n`.
pub fn time_cublaslt_gemm_nt_fp8_e4m3(
    g: &mut Gpu,
    a8_d: &CudaSlice<u8>,
    b8_d: &CudaSlice<u8>,
    c_d: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let plan = Fp8LtPlan::new(g, m, k, n)?;
    let stream = g.stream.clone();
    unsafe { plan.run(&stream, b8_d, a8_d, c_d)? }; // warm up
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { plan.run(&stream, b8_d, a8_d, c_d)? };
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

/// Probe whether `cublasLt64_12.dll` is loadable (it lives beside `cublas64_12.dll` in the redist, so
/// in practice it tracks [`peers_available`], but [`peers_available`] only checks cublas/nvrtc). The fp8
/// peer benches gate on this to **skip, not fail**, when the DLL is absent — the same "green without the
/// hardware" discipline as the other peers. Loading `cublasLt` panics if the DLL is missing, so the
/// probe is wrapped in `catch_unwind`.
pub fn cublaslt_available() -> bool {
    std::panic::catch_unwind(|| match cublaslt_result::create_handle() {
        Ok(h) => {
            unsafe {
                let _ = cublaslt_result::destroy_handle(h);
            }
            true
        }
        Err(_) => false,
    })
    .unwrap_or(false)
}

// ===================================================================================================
// Tier B — cuDNN conv2d (the gold-standard convolution peer). fp16 NHWC tensor-core fast path.
// ===================================================================================================
//
// cuDNN is the industry-standard convolution library: implicit-precomp-GEMM / Winograd / FFT engines
// with per-shape autotuning, the bar Wukong's conv must close on. We drive the **legacy** forward API
// through cudarc's safe `cudnn` module (`ConvForward` over `cudnnConvolutionForward`), letting
// `cudnnGetConvolutionForwardAlgorithm_v7` (the autotuner heuristic) pick the algorithm — exactly the
// "cuDNN chooses its best engine" comparison. To engage the tensor cores on Ada (sm_89) we feed cuDNN
// its fast path: **NHWC fp16** inputs, f32 accumulate, `CUDNN_TENSOR_OP_MATH`. Wukong stores NCHW, so
// X/W are transposed to NHWC/KRSC **once** at setup (outside the timed loop) and cuDNN's NHWC output is
// transposed back to `[K,P,Q]` for the same f64 cross-check Wukong's own conv faces. The chosen algo
// is disclosed in the bench so the % is honest about which engine cuDNN ran.

/// `[C,H,W]` (NCHW, N=1) f32 → `[H,W,C]` (NHWC) f16 — the layout cuDNN's fp16 tensor-core path wants.
fn nchw_to_nhwc_f16(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f16> {
    let mut o = vec![f16::from_f32(0.0); c * h * w];
    for cc in 0..c {
        for hh in 0..h {
            for ww in 0..w {
                o[(hh * w + ww) * c + cc] = f16::from_f32(x[(cc * h + hh) * w + ww]);
            }
        }
    }
    o
}

/// `[K,C,R,S]` (KCRS) f32 → `[K,R,S,C]` (KRSC, the NHWC filter layout) f16.
fn kcrs_to_krsc_f16(wt: &[f32], k: usize, c: usize, r: usize, s: usize) -> Vec<f16> {
    let mut o = vec![f16::from_f32(0.0); k * c * r * s];
    for kk in 0..k {
        for cc in 0..c {
            for rr in 0..r {
                for ss in 0..s {
                    o[((kk * r + rr) * s + ss) * c + cc] =
                        f16::from_f32(wt[((kk * c + cc) * r + rr) * s + ss]);
                }
            }
        }
    }
    o
}

/// `[P,Q,K]` (NHWC output, N=1) f32 → `[K,P,Q]` (NCHW) f32 — back to Wukong's layout for the cross-check.
fn nhwc_out_to_kpq(y: &[f32], k: usize, p: usize, q: usize) -> Vec<f32> {
    let mut o = vec![0f32; k * p * q];
    for pp in 0..p {
        for qq in 0..q {
            for kk in 0..k {
                o[(kk * p + pp) * q + qq] = y[(pp * q + qq) * k + kk];
            }
        }
    }
    o
}

/// Short name of a cuDNN forward-conv algorithm, for honest disclosure of which engine the v7
/// heuristic chose (e.g. `IMPLICIT_PRECOMP_GEMM`, `WINOGRAD_NONFUSED`).
pub fn cudnn_fwd_algo_name(algo: cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t) -> &'static str {
    use cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t as A;
    match algo {
        A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => "IMPLICIT_GEMM",
        A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => "IMPLICIT_PRECOMP_GEMM",
        A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => "GEMM",
        A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => "DIRECT",
        A::CUDNN_CONVOLUTION_FWD_ALGO_FFT => "FFT",
        A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => "FFT_TILING",
        A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => "WINOGRAD",
        A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => "WINOGRAD_NONFUSED",
        _ => "UNKNOWN",
    }
}

/// Probe whether cuDNN (the Tier-B conv peer) is loadable in this process **without** aborting the run:
/// `Cudnn::new` dlopens `cudnn64_9.dll` + the cuDNN-9 sublibraries; a missing DLL panics inside cudarc's
/// loader, so drive it under `catch_unwind` and report `false` (→ the conv bench skips the cuDNN column)
/// on any failure. Cheap; cached for the whole process.
pub fn cudnn_available(g: &mut Gpu) -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let stream = g.stream.clone();
        std::panic::catch_unwind(|| cudarc::cudnn::Cudnn::new(stream).is_ok()).unwrap_or(false)
    })
}

/// Build the cuDNN handle + descriptors for a single-batch `[C,H,W] ⊛ [K,C,R,S]` conv (NHWC fp16,
/// f32 accumulate, `CROSS_CORRELATION` == Wukong's valid conv, dilation 1) and let the v7 heuristic
/// pick the algorithm. Returns the handle, the four descriptors, the chosen algo, and `(P,Q)`. Kept
/// private so the run/time entries share one setup path (the descriptors borrow the handle, so the
/// caller owns the whole tuple for the launch's lifetime).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn cudnn_conv_setup(
    g: &Gpu,
    c: usize,
    h: usize,
    width: usize,
    k: usize,
    r: usize,
    s: usize,
    pad: usize,
    stride: usize,
) -> Result<
    (
        std::sync::Arc<cudarc::cudnn::Cudnn>,
        cudarc::cudnn::ConvDescriptor<f32>,
        cudarc::cudnn::TensorDescriptor<f16>,
        cudarc::cudnn::FilterDescriptor<f16>,
        cudarc::cudnn::TensorDescriptor<f16>,
        cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t,
        (usize, usize),
    ),
    PeerError,
> {
    use cudarc::cudnn::sys::{cudnnConvolutionMode_t, cudnnMathType_t, cudnnTensorFormat_t};
    use cudarc::cudnn::{ConvForward, Cudnn};
    let p = (h + 2 * pad - r) / stride + 1;
    let q = (width + 2 * pad - s) / stride + 1;
    let cudnn = Cudnn::new(g.stream.clone())?;
    let nhwc = cudnnTensorFormat_t::CUDNN_TENSOR_NHWC;
    let x_desc = cudnn.create_4d_tensor::<f16>(nhwc, [1, c as i32, h as i32, width as i32])?;
    let w_desc = cudnn.create_4d_filter::<f16>(nhwc, [k as i32, c as i32, r as i32, s as i32])?;
    let mut conv_desc = cudnn.create_conv2d::<f32>(
        [pad as i32, pad as i32],
        [stride as i32, stride as i32],
        [1, 1],
        cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
    )?;
    // Allow tensor cores (mixed fp16-in / f32-accumulate) — without this the v7 heuristic only offers
    // the slow CUDNN_DEFAULT_MATH engines.
    conv_desc.set_math_type(cudnnMathType_t::CUDNN_TENSOR_OP_MATH)?;
    // All-f16 I/O with f32 accumulate is cuDNN's standard tensor-core conv config (the precision-fair
    // match to Wukong's fp16 path); f16-in/f32-out is CUDNN_STATUS_NOT_SUPPORTED for the TC algos.
    let y_desc = cudnn.create_4d_tensor::<f16>(nhwc, [1, k as i32, p as i32, q as i32])?;
    let algo = {
        let fwd = ConvForward { conv: &conv_desc, x: &x_desc, w: &w_desc, y: &y_desc };
        fwd.pick_algorithm()?
    };
    Ok((cudnn, conv_desc, x_desc, w_desc, y_desc, algo, (p, q)))
}

/// Run cuDNN's convolution forward once and return `(O[K,P,Q] NCHW f32, chosen algo)` — the
/// correctness-gate + checksum entry. fp16 NHWC tensor-core fast path; the f32 NHWC output is
/// transposed back to Wukong's `[K,P,Q]` so it faces the identical f64 reference.
#[allow(clippy::too_many_arguments)]
pub fn cudnn_conv2d_run(
    g: &mut Gpu,
    x: &[f32],
    w: &[f32],
    c: usize,
    h: usize,
    width: usize,
    k: usize,
    r: usize,
    s: usize,
    pad: usize,
    stride: usize,
) -> Result<(Vec<f32>, cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t), PeerError> {
    use cudarc::cudnn::ConvForward;
    assert_eq!(x.len(), c * h * width);
    assert_eq!(w.len(), k * c * r * s);
    let (cudnn, conv_desc, x_desc, w_desc, y_desc, algo, (p, q)) =
        cudnn_conv_setup(g, c, h, width, k, r, s, pad, stride)?;
    let _ = &cudnn; // keep the handle alive for the launch
    let fwd = ConvForward { conv: &conv_desc, x: &x_desc, w: &w_desc, y: &y_desc };
    let ws_size = fwd.get_workspace_size(algo)?;
    let x_d = g.stream.memcpy_stod(&nchw_to_nhwc_f16(x, c, h, width))?;
    let w_d = g.stream.memcpy_stod(&kcrs_to_krsc_f16(w, k, c, r, s))?;
    let mut y_d = g.stream.alloc_zeros::<f16>(k * p * q)?;
    let mut ws: Option<CudaSlice<u8>> =
        if ws_size > 0 { Some(g.stream.alloc_zeros::<u8>(ws_size)?) } else { None };
    let (one, zero) = (f16::from_f32(1.0), f16::from_f32(0.0));
    unsafe {
        fwd.launch(algo, ws.as_mut(), (one, zero), &x_d, &w_d, &mut y_d)?;
    }
    g.stream.synchronize()?;
    let y_nhwc: Vec<f32> = g.stream.memcpy_dtov(&y_d)?.iter().map(|v| v.to_f32()).collect();
    Ok((nhwc_out_to_kpq(&y_nhwc, k, p, q), algo))
}

/// Time cuDNN's convolution forward: `iters` resident launches bracketed by one sync, after a warm-up —
/// the identical timing shape Wukong's conv bench uses, so the ratio is apples-to-apples. Returns
/// `(seconds/launch, chosen algo)`.
#[allow(clippy::too_many_arguments)]
pub fn time_cudnn_conv2d(
    g: &mut Gpu,
    c: usize,
    h: usize,
    width: usize,
    k: usize,
    r: usize,
    s: usize,
    pad: usize,
    stride: usize,
    iters: u32,
) -> Result<(f64, cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t), PeerError> {
    use cudarc::cudnn::ConvForward;
    let (cudnn, conv_desc, x_desc, w_desc, y_desc, algo, (p, q)) =
        cudnn_conv_setup(g, c, h, width, k, r, s, pad, stride)?;
    let _ = &cudnn;
    let fwd = ConvForward { conv: &conv_desc, x: &x_desc, w: &w_desc, y: &y_desc };
    let ws_size = fwd.get_workspace_size(algo)?;
    let x_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.01); c * h * width])?;
    let w_d = g.stream.memcpy_stod(&vec![f16::from_f32(0.01); k * c * r * s])?;
    let mut y_d = g.stream.alloc_zeros::<f16>(k * p * q)?;
    let mut ws: Option<CudaSlice<u8>> =
        if ws_size > 0 { Some(g.stream.alloc_zeros::<u8>(ws_size)?) } else { None };
    let (one, zero) = (f16::from_f32(1.0), f16::from_f32(0.0));
    let do_launch = |ws: &mut Option<CudaSlice<u8>>, y: &mut CudaSlice<f16>| -> Result<(), PeerError> {
        unsafe { fwd.launch(algo, ws.as_mut(), (one, zero), &x_d, &w_d, y)? };
        Ok(())
    };
    do_launch(&mut ws, &mut y_d)?; // warm up
    g.stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        do_launch(&mut ws, &mut y_d)?;
    }
    g.stream.synchronize()?;
    Ok((t0.elapsed().as_secs_f64() / iters as f64, algo))
}

// ---------------------------------------------------------------------------------------------------
// int8 GEMM+dequant **chain** peer — the round-trip a cuBLAS int8 inference output stage must pay
// because cuBLAS emits raw i32. Wukong folds `out = f32(Σ u8·i8)·scale[j]` into the GEMM store for ~0
// cost (`int8_gemm_nt_*_deq`), a fusion a closed-source library kernel structurally can't do — so the
// honest end-to-end comparison for the quantized output stage is Wukong's **single fused kernel** vs
// the cuBLAS **GEMM + separate dequant kernel** chain. This is the int8 "beat cuBLAS outright" lever.
// ---------------------------------------------------------------------------------------------------

/// Per-channel int8→f32 dequant `out[r,c] = f32(in[r,c])·scale[c]` — one element/thread on a 2-D grid
/// (`gridDim.x = ⌈N/256⌉`, `gridDim.y = M`; block 256×1), so the output column `c = ctaid.x·256 + tid.x`
/// needs no per-element integer remainder and `scale[c]` / the i32 row stream coalesce. A fair, fast
/// dequant — the second kernel a cuBLAS int8 pipeline launches, re-reading the whole M×N i32 from HBM and
/// writing M×N f32 (the HBM round-trip + launch Wukong's fused epilogue removes).
const INT8_DEQUANT_CHAIN_PTX: &str = r#".version 8.4
.target sm_89
.address_size 64
.visible .entry int8_dequant_chain(
    .param .u64 pIn,
    .param .u64 pScale,
    .param .u64 pOut,
    .param .u32 pM,
    .param .u32 pN
)
{
    .reg .pred %pc,%pr;
    .reg .b32 %col,%row,%m,%n,%idx,%vi,%bx,%bdx;
    .reg .f32 %f,%sc;
    .reg .b64 %In,%Scale,%Out,%off,%pp;
    ld.param.u64 %In,[pIn];
    ld.param.u64 %Scale,[pScale];
    ld.param.u64 %Out,[pOut];
    ld.param.u32 %m,[pM];
    ld.param.u32 %n,[pN];
    cvta.to.global.u64 %In,%In;
    cvta.to.global.u64 %Scale,%Scale;
    cvta.to.global.u64 %Out,%Out;
    mov.u32 %bx,%ctaid.x;
    mov.u32 %bdx,%ntid.x;
    mov.u32 %col,%tid.x;
    mad.lo.s32 %col,%bx,%bdx,%col;
    mov.u32 %row,%ctaid.y;
    setp.ge.u32 %pc,%col,%n;
    @%pc bra END;
    setp.ge.u32 %pr,%row,%m;
    @%pr bra END;
    mad.lo.s32 %idx,%row,%n,%col;
    mul.wide.u32 %off,%idx,4;
    add.s64 %pp,%In,%off;
    ld.global.b32 %vi,[%pp];
    cvt.rn.f32.s32 %f,%vi;
    mul.wide.u32 %off,%col,4;
    add.s64 %pp,%Scale,%off;
    ld.global.f32 %sc,[%pp];
    mul.f32 %f,%f,%sc;
    mul.wide.u32 %off,%idx,4;
    add.s64 %pp,%Out,%off;
    st.global.f32 [%pp],%f;
END:
    ret;
}
"#;

/// Time the **cuBLAS int8 GEMM + dequant chain**: `iters` resident pairs of (`cublasGemmEx` i32 →
/// `int8_dequant_chain` i32→f32), one warm-up, one trailing sync. Returns **seconds per pair** — the
/// honest peer for Wukong's single fused `int8_gemm_nt_*_deq`. A/B/scale are dummy (timing is
/// data-independent); the dequant kernel is JITed once. Needs cuBLAS (skips via `peers_available`).
pub fn time_cublas_int8_gemm_dequant_chain(
    g: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    iters: u32,
) -> Result<f64, PeerError> {
    let blas = CudaBlas::new(g.stream.clone())?;
    let a_d = g.stream.memcpy_stod(&vec![1i8; m * k])?;
    let b_d = g.stream.memcpy_stod(&vec![1i8; n * k])?;
    let mut ci_d = g.stream.memcpy_stod(&vec![0i32; m * n])?;
    let scale_d = g.stream.memcpy_stod(&vec![1.0f32 / 127.0; n])?;
    let mut cf_d = g.stream.memcpy_stod(&vec![0f32; m * n])?;
    let module = g.ctx.load_module(INT8_DEQUANT_CHAIN_PTX.into())?;
    let deq = module.load_function("int8_dequant_chain")?;
    let stream = g.stream.clone();
    let (mm, nn) = (m as u32, n as u32);
    let dcfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), m as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // warm up: one GEMM (i32) then one dequant (i32→f32).
    unsafe { gemm_ex_nt_int8(&blas, &stream, &a_d, &b_d, &mut ci_d, m, k, n)? };
    {
        let mut bld = stream.launch_builder(&deq);
        bld.arg(&ci_d).arg(&scale_d).arg(&mut cf_d).arg(&mm).arg(&nn);
        unsafe { bld.launch(dcfg)? };
    }
    stream.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        unsafe { gemm_ex_nt_int8(&blas, &stream, &a_d, &b_d, &mut ci_d, m, k, n)? };
        let mut bld = stream.launch_builder(&deq);
        bld.arg(&ci_d).arg(&scale_d).arg(&mut cf_d).arg(&mm).arg(&nn);
        unsafe { bld.launch(dcfg)? };
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// ---------------------------------------------------------------------------------------------------
// Fused FA2-class peer — PyTorch SDPA's *fused* backends (cuDNN fused attention + cutlass mem-efficient
// fMHA), driven as a subprocess. THE bar the M5 milestone actually requires: a genuinely *fused*
// FlashAttention-class kernel, NOT the pre-FlashAttention unfused cuBLAS chain ([`cublas_attn_chain`]).
//
// Why a subprocess: a real fused FA-class kernel exists for Ada sm_89 but is reachable here only from
// Python (NVRTC has no headers, so `nvcuda::wmma`/CUTLASS/FlashAttention won't compile as an in-process
// peer). The peer script `tools/fa2_sdpa_peer.py` forces each fused SDPA backend in turn over the SAME
// f16 Q/K/V bytes Wukong's flash runs, CUDA-event-times it (so Python's per-call dispatch overhead is
// excluded — fair to the peer), writes the chosen backend's O (f32) for the same `ref_attn` f64-oracle
// cross-check Wukong's flash gets, and a flat `key=value` report this driver parses (no serde). On the
// Windows PyTorch wheel FLASH_ATTENTION is not built, but **CUDNN_ATTENTION (cuDNN's fused flash) and
// EFFICIENT_ATTENTION (cutlass mem-efficient fMHA) are** — both genuinely fused; the faster is chosen.
//
// Honesty: cross-process means the GPU clock can differ between Wukong's timing and the peer's, so the
// caller must (a) warm the GPU, (b) run the peer and Wukong back-to-back in one window, (c) repeat >=3x
// and report best-of. The peer's CUDA-event time excludes its launch overhead while Wukong's wall-clock
// includes its (negligible, Rust) launch overhead, so any Wukong win is the *conservative* direction.
// The MATH backend (also reported) is the unfused in-process analogue of [`cublas_attn_chain`] — a
// cross-anchor that should track Wukong's existing chain ratio.
// ---------------------------------------------------------------------------------------------------

/// Outcome of one fused-peer run. `chosen`/`chosen_sec`/`chosen_gflops` are the *fastest fused* backend
/// (cuDNN or cutlass-efficient); `math_sec` is the unfused softmax-materialize anchor; `o` is the chosen
/// backend's `[b*h*s*d]` f32 output, for the same `ref_attn` f64 cross-check Wukong's flash gets.
pub struct Fa2PeerReport {
    pub chosen: String,
    pub chosen_sec: f64,
    pub chosen_gflops: f64,
    /// The chosen backend's *sdpa-only* time. Without `--rope` this equals `chosen_sec`. With `--rope`,
    /// `chosen_sec` is the rope+sdpa pipeline (what a model pays because the fused-attention library
    /// can't absorb RoPE) and `chosen_sdpa_sec` is the no-rope reference — the gap is the RoPE kernel
    /// overhead Wukong fuses into its one flash launch for free.
    pub chosen_sdpa_sec: Option<f64>,
    /// Which RoPE variant the peer's chosen backend used (`eager`/`complex`/`compiled`), or `None`/`"none"`
    /// without `--rope`. Names the rope the peer pipeline was timed against, so the writeup can't be
    /// accused of strawmanning a slow rope — the harness times all variants and reports the fastest.
    pub chosen_rope_variant: Option<String>,
    pub cudnn_sec: Option<f64>,
    pub efficient_sec: Option<f64>,
    pub math_sec: Option<f64>,
    pub o: Vec<f32>,
    pub device: String,
    pub torch: String,
}

/// Resolve `(python, peer_script)`: env `WUKONG_FA2_PYTHON` / `WUKONG_FA2_PEER` override the defaults
/// (the workspace `tools/torch-cuda-venv` python + `tools/fa2_sdpa_peer.py`, relative to this crate).
/// In a git worktree the venv usually lives in the main checkout, so set `WUKONG_FA2_PYTHON` there.
fn fa2_peer_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_default(); // crates/wukong_codegen_gpu -> workspace root
    let python = std::env::var_os("WUKONG_FA2_PYTHON")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("tools/torch-cuda-venv/Scripts/python.exe"));
    let peer = std::env::var_os("WUKONG_FA2_PEER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.join("tools/fa2_sdpa_peer.py"));
    (python, peer)
}

/// True iff the fused-peer Python with CUDA torch is runnable, cached for the process. Lets the bench
/// **skip, not fail**, when the optional torch-CUDA venv isn't installed — the same "green without the
/// hardware" discipline as [`peers_available`]. Probes `import torch; torch.cuda.is_available()`.
pub fn fa2_peer_available() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let (python, peer) = fa2_peer_paths();
        if !peer.exists() {
            return false;
        }
        std::process::Command::new(&python)
            .args([
                "-c",
                "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)",
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Drive the fused FA2-class peer over the SAME f16 Q/K/V Wukong's flash runs (`[b,h,s,d]` head-major,
/// `b=1` for the single-batch bench). Returns the chosen fused backend's O (f32) and its same-shape
/// timing. `scale` is folded the way SDPA expects: `softmax(scale * Q*Kᵀ) * V`. The Python harness times
/// each backend best-of-`runs` over `iters` CUDA-event-bracketed launches after `warmup` launches.
pub fn fa2_sdpa_peer(
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    q16: &[f16],
    k16: &[f16],
    v16: &[f16],
    scale: f32,
    causal: bool,
    warmup: u32,
    iters: u32,
    runs: u32,
) -> Result<Fa2PeerReport, PeerError> {
    use std::io::Write;
    let n = b * h * s * d;
    assert_eq!(q16.len(), n, "Q must be b*h*s*d");
    assert_eq!(k16.len(), n, "K must be b*h*s*d");
    assert_eq!(v16.len(), n, "V must be b*h*s*d");
    let (python, peer) = fa2_peer_paths();

    // Per-call temp workspace (the Gpu mutex serializes callers in practice; the counter is belt-and-
    // suspenders against any concurrent use).
    static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uid = CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("wukong_fa2_{}_{}", std::process::id(), uid));
    std::fs::create_dir_all(&dir)?;
    let dump = |name: &str, v: &[f16]| -> Result<std::path::PathBuf, PeerError> {
        let p = dir.join(name);
        let mut bytes = Vec::with_capacity(v.len() * 2);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::File::create(&p)?.write_all(&bytes)?;
        Ok(p)
    };
    let qp = dump("q.bin", q16)?;
    let kp = dump("k.bin", k16)?;
    let vp = dump("v.bin", v16)?;
    let op = dir.join("o.bin");
    let rp = dir.join("rep.txt");

    let out = std::process::Command::new(&python)
        .arg(&peer)
        .arg("--q").arg(&qp).arg("--k").arg(&kp).arg("--v").arg(&vp)
        .arg("--o").arg(&op).arg("--report").arg(&rp)
        .arg("--B").arg(b.to_string()).arg("--H").arg(h.to_string())
        .arg("--S").arg(s.to_string()).arg("--D").arg(d.to_string())
        .arg("--scale").arg(format!("{scale:.9}"))
        .arg("--causal").arg(if causal { "1" } else { "0" })
        .arg("--warmup").arg(warmup.to_string())
        .arg("--iters").arg(iters.to_string())
        .arg("--runs").arg(runs.to_string())
        .output()?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "fa2 peer subprocess failed ({}):\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }

    let rep = std::fs::read_to_string(&rp)?;
    let mut map = std::collections::HashMap::new();
    for line in rep.lines() {
        if let Some((k, val)) = line.split_once('=') {
            map.insert(k.to_string(), val.to_string());
        }
    }
    let getf = |k: &str| -> Option<f64> { map.get(k).and_then(|v| v.parse::<f64>().ok()) };
    let chosen = map.get("chosen").cloned().unwrap_or_else(|| "none".into());
    if chosen == "none" || chosen == "None" {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!("fa2 peer: no fused backend available; report:\n{rep}").into());
    }
    let chosen_sec = getf(&format!("{chosen}_sec"))
        .ok_or_else(|| -> PeerError { format!("fa2 peer: missing {chosen}_sec in report").into() })?;
    let chosen_gflops = getf(&format!("{chosen}_gflops")).unwrap_or(0.0);

    let obytes = std::fs::read(&op)?;
    if obytes.len() != n * 4 {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!("fa2 peer: O is {} bytes != expected {}", obytes.len(), n * 4).into());
    }
    let o: Vec<f32> = obytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let _ = std::fs::remove_dir_all(&dir); // best-effort cleanup

    Ok(Fa2PeerReport {
        chosen: chosen.clone(),
        chosen_sec,
        chosen_gflops,
        chosen_sdpa_sec: getf(&format!("{chosen}_sdpa_sec")),
        chosen_rope_variant: map.get(&format!("{chosen}_rope_variant")).cloned(),
        cudnn_sec: getf("cudnn_sec"),
        efficient_sec: getf("efficient_sec"),
        math_sec: getf("math_sec"),
        o,
        device: map.get("device").cloned().unwrap_or_default(),
        torch: map.get("torch").cloned().unwrap_or_default(),
    })
}

/// Fused-RoPE variant of [`fa2_sdpa_peer`]: the honest peer for Wukong's `flash_d64_mprope`. A model
/// using a fused attention library (cuDNN/cutlass) **cannot fold RoPE into the attention kernel** — it
/// must run an interleaved-RoPE elementwise pass over Q and K *first*, then the fused SDPA. This drives
/// exactly that pipeline (the optimized eager — and `torch.compile` if available — RoPE Python can
/// produce, so the peer isn't strawmanned) and times it as one unit. `chosen_sec` is rope+sdpa (what the
/// model pays); `chosen_sdpa_sec` is sdpa-only (the no-rope reference). Wukong pays neither separately —
/// its flash rotates Q/K in-register at load, so the rope+sdpa→sdpa gap is the lever the library can't
/// touch. `cos`/`sin` are the `[s, d/2]` interleaved-RoPE tables (θ_t = base^(−2t/d)).
#[allow(clippy::too_many_arguments)]
pub fn fa2_sdpa_peer_rope(
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    q16: &[f16],
    k16: &[f16],
    v16: &[f16],
    cos: &[f32],
    sin: &[f32],
    scale: f32,
    causal: bool,
    warmup: u32,
    iters: u32,
    runs: u32,
) -> Result<Fa2PeerReport, PeerError> {
    use std::io::Write;
    let n = b * h * s * d;
    assert_eq!(q16.len(), n, "Q must be b*h*s*d");
    assert_eq!(k16.len(), n, "K must be b*h*s*d");
    assert_eq!(v16.len(), n, "V must be b*h*s*d");
    assert_eq!(cos.len(), s * (d / 2), "cos must be s*(d/2)");
    assert_eq!(sin.len(), s * (d / 2), "sin must be s*(d/2)");
    let (python, peer) = fa2_peer_paths();

    static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uid = CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("wukong_fa2r_{}_{}", std::process::id(), uid));
    std::fs::create_dir_all(&dir)?;
    let dump16 = |name: &str, v: &[f16]| -> Result<std::path::PathBuf, PeerError> {
        let p = dir.join(name);
        let mut bytes = Vec::with_capacity(v.len() * 2);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::File::create(&p)?.write_all(&bytes)?;
        Ok(p)
    };
    let dump32 = |name: &str, v: &[f32]| -> Result<std::path::PathBuf, PeerError> {
        let p = dir.join(name);
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::File::create(&p)?.write_all(&bytes)?;
        Ok(p)
    };
    let qp = dump16("q.bin", q16)?;
    let kp = dump16("k.bin", k16)?;
    let vp = dump16("v.bin", v16)?;
    let cp = dump32("cos.bin", cos)?;
    let sp = dump32("sin.bin", sin)?;
    let op = dir.join("o.bin");
    let rp = dir.join("rep.txt");

    let out = std::process::Command::new(&python)
        .arg(&peer)
        .arg("--q").arg(&qp).arg("--k").arg(&kp).arg("--v").arg(&vp)
        .arg("--o").arg(&op).arg("--report").arg(&rp)
        .arg("--B").arg(b.to_string()).arg("--H").arg(h.to_string())
        .arg("--S").arg(s.to_string()).arg("--D").arg(d.to_string())
        .arg("--scale").arg(format!("{scale:.9}"))
        .arg("--causal").arg(if causal { "1" } else { "0" })
        .arg("--warmup").arg(warmup.to_string())
        .arg("--iters").arg(iters.to_string())
        .arg("--runs").arg(runs.to_string())
        .arg("--rope").arg("1").arg("--cos").arg(&cp).arg("--sin").arg(&sp)
        .output()?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "fa2 rope peer subprocess failed ({}):\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }

    let rep = std::fs::read_to_string(&rp)?;
    let mut map = std::collections::HashMap::new();
    for line in rep.lines() {
        if let Some((k, val)) = line.split_once('=') {
            map.insert(k.to_string(), val.to_string());
        }
    }
    let getf = |k: &str| -> Option<f64> { map.get(k).and_then(|v| v.parse::<f64>().ok()) };
    let chosen = map.get("chosen").cloned().unwrap_or_else(|| "none".into());
    if chosen == "none" || chosen == "None" {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!("fa2 rope peer: no fused backend available; report:\n{rep}").into());
    }
    let chosen_sec = getf(&format!("{chosen}_sec"))
        .ok_or_else(|| -> PeerError { format!("fa2 rope peer: missing {chosen}_sec").into() })?;
    let chosen_gflops = getf(&format!("{chosen}_gflops")).unwrap_or(0.0);

    let obytes = std::fs::read(&op)?;
    if obytes.len() != n * 4 {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!("fa2 rope peer: O is {} bytes != {}", obytes.len(), n * 4).into());
    }
    let o: Vec<f32> = obytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let _ = std::fs::remove_dir_all(&dir);

    Ok(Fa2PeerReport {
        chosen: chosen.clone(),
        chosen_sec,
        chosen_gflops,
        chosen_sdpa_sec: getf(&format!("{chosen}_sdpa_sec")),
        chosen_rope_variant: map.get(&format!("{chosen}_rope_variant")).cloned(),
        cudnn_sec: getf("cudnn_sec"),
        efficient_sec: getf("efficient_sec"),
        math_sec: getf("math_sec"),
        o,
        device: map.get("device").cloned().unwrap_or_default(),
        torch: map.get("torch").cloned().unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    /// **The peer chain's attention must be the *same kernel* the Wukong layer runs** (device-free).
    ///
    /// `CublasChainLayer`'s whole claim is that attention is common to both stacks, so the printed
    /// GPT-2-layer ratio is cuBLAS-vs-WMMA GEMM + epilogue fusion and nothing else. It used to pair
    /// `gpu::wmma_flash_entry` with `gpu::wmma_flash_cfg` while `ResidentLayerF16` resolved the pair
    /// through `gpu::wmma_flash_plan` — identical up to S=2048 and *different* at S>=4096, where the
    /// plan routes to the warp-specialized kernels. This pins the seam at every shape the MHA bench
    /// runs, single- and multi-head, and pins the two families' launch shapes so an entry can never be
    /// paired with the other family's config (2-warp ws on a 32-thread launch is a hang, not a number).
    #[test]
    fn chain_flash_is_the_wukong_plan() {
        use super::CublasChainLayer;
        // (d, heads) pairs whose head dim `d/heads` the tensor-core flash covers: 64 and 128.
        for &(d, heads) in &[(768usize, 12usize), (64, 1), (128, 1), (256, 2)] {
            let dh = d / heads;
            for &s in &[512usize, 1024, 2048, 4096] {
                let (entry, cfg) = CublasChainLayer::flash_plan_w(dh, s, heads);
                let (want_entry, want_cfg) = crate::gpu::wmma_flash_plan(dh, s, heads);
                assert_eq!(entry, want_entry, "chain flash entry dh={dh} S={s} heads={heads}");
                assert_eq!(cfg.grid_dim, want_cfg.grid_dim, "chain flash grid dh={dh} S={s}");
                assert_eq!(cfg.block_dim, want_cfg.block_dim, "chain flash block dh={dh} S={s}");
                assert_eq!(cfg.grid_dim.1, heads as u32, "grid.y must carry the head count");
                // Entry and config must belong to the same family: the `_ws*` kernels are 2 warps per
                // CTA with a halved grid; every other tensor-core flash entry is 1 warp, grid S/16.
                if entry.contains("_ws") {
                    assert_eq!(cfg.block_dim.0, 64, "{entry} is warp-specialized: 2 warps/CTA");
                    assert_eq!(cfg.grid_dim.0, ((s / 16) as u32).div_ceil(2), "{entry} grid");
                } else {
                    assert_eq!(cfg.block_dim.0, 32, "{entry} is 1 warp/CTA");
                    assert_eq!(cfg.grid_dim.0, (s / 16) as u32, "{entry} grid");
                }
            }
        }
        // The regression this closes, stated as a fact about the shape that carried it: with the ws
        // route live (the default), at S=4096 the plan and the bare `wmma_flash_entry` name DIFFERENT
        // kernels — so a peer built on the latter measures a different attention than Wukong runs.
        // Skipped only under the `WUKONG_FLASH_WS=0` kill-switch, where the two legitimately coincide.
        if crate::gpu::ws_flash_route(64, 4096).is_some() {
            assert_ne!(
                crate::gpu::wmma_flash_plan(64, 4096, 12).0,
                crate::gpu::wmma_flash_entry(64, 4096),
                "S=4096 D=64: plan and bare entry must differ (else this gate proves nothing)"
            );
        }
    }
}
