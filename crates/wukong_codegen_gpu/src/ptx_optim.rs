//! Fused optimizer-step GPU kernels (PTX): **AdamW** and **SGD**.
//!
//! The backward pass produces one gradient per parameter; the optimizer turns gradients into a
//! weight update. A tensor library issues *one launch per parameter tensor*; Wukong lays every
//! parameter's `(w, g, m, v)` buffers out contiguously and updates them **all in one grid-stride
//! launch** — the fusion a library splits into many launches (the M8 lever).
//!
//! The AdamW kernel mirrors [`wukong_autodiff::optim::build_adamw_step`] **op-for-op** (same
//! operation order; `fma.rn` / `div.rn` / `sqrt.rn`, every result IEEE-754 single-rounded f32), so
//! the device update matches the CPU autodiff reference to a **tight tolerance** — most lanes
//! bit-exact (~98%), the residual a ≤1-ulp difference from the *interpreter's* f64-intermediate
//! double-rounding (the GPU's single-rounded f32 is, if anything, the more accurate of the two).
//! Gated below against the real `build_adamw_step` MIR run on the interpreter oracle.
//!
//! PTX is pure ASCII (a non-ASCII byte is a `ptxas fatal`); both modules are tagged at the
//! **`sm_80` floor** ([`crate::ptx_target::HDR_SM80`]) — the instruction mix is plain f32, legal on
//! Ampere and every later part, and PTX is forward-compatible only, so tagging the development
//! device's arch would only cost us the A100. The grid-stride loop makes correctness independent of
//! the launch grid, so one fixed kernel handles any element count.

use crate::gpu::Gpu;
use cudarc::driver::{CudaSlice, DriverError, LaunchConfig, PushKernelArg};

/// Hyperparameter-buffer layout (an f32 vector), mirroring [`wukong_autodiff::optim::hp`]. The
/// caller fills these each step — the bias corrections `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t`
/// advance with the step count `t`.
pub mod hp {
    pub const LR: usize = 0;
    pub const BETA1: usize = 1;
    pub const BETA2: usize = 2;
    pub const EPS: usize = 3;
    pub const WD: usize = 4;
    /// `1 - beta1^t`
    pub const BC1: usize = 5;
    /// `1 - beta2^t`
    pub const BC2: usize = 6;
    pub const LEN: usize = 7;
}

/// Fused **AdamW** update over `n` contiguous f32 elements (grid-stride). One launch updates every
/// parameter when `(w, g, m, v)` are laid out contiguously across all params. Per element:
///
/// ```text
/// m = beta1*m + (1-beta1)*g
/// v = beta2*v + (1-beta2)*g^2
/// w -= lr * ( (m/bc1) / (sqrt(v/bc2) + eps) + wd*w )      // decoupled weight decay
/// ```
///
/// `fma.rn` for the two moment updates, `div.rn`/`sqrt.rn` elsewhere — every result single-rounded
/// f32, matching the same MIR run on the interpreter to a tight tolerance (mostly bit-exact).
///
/// (Header floor: this literal carries exactly [`crate::ptx_target::HDR_SM80`] — a `const` cannot
/// interpolate the constant, so `optimizer_ptx_is_ascii` asserts the two agree byte-for-byte.)
pub const ADAMW_STEP_PTX: &str = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry adamw_step(
    .param .u64 adamw_w,
    .param .u64 adamw_g,
    .param .u64 adamw_m,
    .param .u64 adamw_v,
    .param .u64 adamw_hp,
    .param .u32 adamw_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<11>;
    .reg .f32   %f<30>;

    ld.param.u64    %rd1, [adamw_w];
    ld.param.u64    %rd2, [adamw_g];
    ld.param.u64    %rd3, [adamw_m];
    ld.param.u64    %rd4, [adamw_v];
    ld.param.u64    %rd5, [adamw_hp];
    ld.param.u32    %r1,  [adamw_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;
    cvta.to.global.u64  %rd4, %rd4;
    cvta.to.global.u64  %rd5, %rd5;

    // Hyperparameters, loaded once (hp[0..6]).
    ld.global.f32   %f1, [%rd5];        // lr
    ld.global.f32   %f2, [%rd5+4];      // beta1
    ld.global.f32   %f3, [%rd5+8];      // beta2
    ld.global.f32   %f4, [%rd5+12];     // eps
    ld.global.f32   %f5, [%rd5+16];     // wd
    ld.global.f32   %f6, [%rd5+20];     // bc1
    ld.global.f32   %f7, [%rd5+24];     // bc2
    mov.f32         %f8, 0f3F800000;    // 1.0
    sub.f32         %f9,  %f8, %f2;     // om1 = 1 - beta1
    sub.f32         %f10, %f8, %f3;     // om2 = 1 - beta2

    // idx = ctaid.x*ntid.x + tid.x ; stride = ntid.x*nctaid.x
    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r5, %r3, %r2, %r4;
    mov.u32         %r6, %nctaid.x;
    mul.lo.s32      %r7, %r2, %r6;

ADAMW_LOOP:
    setp.ge.s32     %p1, %r5, %r1;
    @%p1 bra        ADAMW_END;

    mul.wide.s32    %rd6, %r5, 4;
    add.s64         %rd7,  %rd1, %rd6;  // &w[idx]
    add.s64         %rd8,  %rd2, %rd6;  // &g[idx]
    add.s64         %rd9,  %rd3, %rd6;  // &m[idx]
    add.s64         %rd10, %rd4, %rd6;  // &v[idx]

    ld.global.f32   %f11, [%rd8];       // g
    ld.global.f32   %f12, [%rd9];       // m_old
    ld.global.f32   %f13, [%rd10];      // v_old
    ld.global.f32   %f14, [%rd7];       // w_old

    // m = fma(beta1, m_old, om1*g)
    mul.f32         %f15, %f9,  %f11;
    fma.rn.f32      %f16, %f2,  %f12, %f15;
    // v = fma(beta2, v_old, om2*(g*g))
    mul.f32         %f17, %f11, %f11;
    mul.f32         %f18, %f10, %f17;
    fma.rn.f32      %f19, %f3,  %f13, %f18;
    st.global.f32   [%rd9],  %f16;
    st.global.f32   [%rd10], %f19;

    // w -= lr * ( (m/bc1) / (sqrt(v/bc2)+eps) + wd*w )
    div.rn.f32      %f20, %f16, %f6;    // mhat
    div.rn.f32      %f21, %f19, %f7;    // vhat
    sqrt.rn.f32     %f22, %f21;
    add.f32         %f23, %f22, %f4;    // + eps
    div.rn.f32      %f24, %f20, %f23;   // step
    mul.f32         %f25, %f5,  %f14;   // wd*w
    add.f32         %f26, %f24, %f25;   // upd
    mul.f32         %f27, %f1,  %f26;   // lr*upd
    sub.f32         %f28, %f14, %f27;   // w_new
    st.global.f32   [%rd7], %f28;

    add.s32         %r5, %r5, %r7;
    bra             ADAMW_LOOP;
ADAMW_END:
    ret;
}
"#;

/// Fused **SGD** update with decoupled weight decay over `n` contiguous f32 elements (grid-stride):
/// `w -= lr*(g + wd*w)`. Reads `hp[LR]` and `hp[WD]` from the same [`hp`] layout (the moment slots
/// are unused). All ops correctly-rounded f32, so it is bit-exact to the same formula on the CPU.
///
/// (Header floor: this literal carries exactly [`crate::ptx_target::HDR_SM80`] — see
/// [`ADAMW_STEP_PTX`].)
pub const SGD_STEP_PTX: &str = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry sgd_step(
    .param .u64 sgd_w,
    .param .u64 sgd_g,
    .param .u64 sgd_hp,
    .param .u32 sgd_n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<12>;

    ld.param.u64    %rd1, [sgd_w];
    ld.param.u64    %rd2, [sgd_g];
    ld.param.u64    %rd3, [sgd_hp];
    ld.param.u32    %r1,  [sgd_n];
    cvta.to.global.u64  %rd1, %rd1;
    cvta.to.global.u64  %rd2, %rd2;
    cvta.to.global.u64  %rd3, %rd3;

    ld.global.f32   %f1, [%rd3];        // lr
    ld.global.f32   %f2, [%rd3+16];     // wd

    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r5, %r3, %r2, %r4;
    mov.u32         %r6, %nctaid.x;
    mul.lo.s32      %r7, %r2, %r6;

SGD_LOOP:
    setp.ge.s32     %p1, %r5, %r1;
    @%p1 bra        SGD_END;

    mul.wide.s32    %rd4, %r5, 4;
    add.s64         %rd5, %rd1, %rd4;   // &w[idx]
    add.s64         %rd6, %rd2, %rd4;   // &g[idx]

    ld.global.f32   %f3, [%rd6];        // g
    ld.global.f32   %f4, [%rd5];        // w
    mul.f32         %f5, %f2, %f4;      // wd*w
    add.f32         %f6, %f3, %f5;      // g + wd*w
    mul.f32         %f7, %f1, %f6;      // lr*(...)
    sub.f32         %f8, %f4, %f7;      // w_new
    st.global.f32   [%rd5], %f8;

    add.s32         %r5, %r5, %r7;
    bra             SGD_LOOP;
SGD_END:
    ret;
}
"#;

/// A grid that saturates the device for a memory-bound grid-stride kernel: 256-thread blocks, capped
/// at `32·SM` blocks (enough resident waves to hide HBM latency on Ada), but never more blocks than
/// elements. The grid-stride loop covers every element regardless, so this only sizes occupancy.
/// Shared with the backward kernels (`ptx_autodiff_bwd`), which are the same memory-bound shape.
pub(crate) fn grid_stride_cfg(g: &Gpu, n: u32) -> LaunchConfig {
    let block = 256u32;
    let max_blocks = (g.sm_count() as u32).saturating_mul(32).max(1);
    let grid = n.div_ceil(block).clamp(1, max_blocks);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch the fused **AdamW** step on the GPU: `w`, `m`, `v` are updated in place from the gradient
/// `grad` and the [`hp`] hyperparameter vector. Host-buffer convenience wrapper (uploads, launches,
/// downloads) — the resident training step uses device buffers directly. All four data buffers must
/// have the same length `n`; `hp` must have at least [`hp::LEN`] entries.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    g: &mut Gpu,
    w: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    hp: &[f32],
) -> Result<(), DriverError> {
    let n = w.len();
    assert_eq!(grad.len(), n, "adamw: grad length");
    assert_eq!(m.len(), n, "adamw: m length");
    assert_eq!(v.len(), n, "adamw: v length");
    assert!(
        hp.len() >= hp::LEN,
        "adamw: hp must have >= {} entries",
        hp::LEN
    );

    let f = g.function("adamw_step", ADAMW_STEP_PTX, "adamw_step")?;
    let mut w_d = g.stream.memcpy_stod(w)?;
    let g_d = g.stream.memcpy_stod(grad)?;
    let mut m_d = g.stream.memcpy_stod(m)?;
    let mut v_d = g.stream.memcpy_stod(v)?;
    let hp_d = g.stream.memcpy_stod(hp)?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&mut w_d)
        .arg(&g_d)
        .arg(&mut m_d)
        .arg(&mut v_d)
        .arg(&hp_d)
        .arg(&n_u);
    // SAFETY: the `adamw_step` kernel reads `grad[i]` and `hp[0..hp::LEN)` and writes `w[i]`, `m[i]`,
    // `v[i]` for `i` in `[0, n)` only. Every device buffer here was just uploaded from a host slice
    // the asserts above proved is `n` long (`hp` at least `hp::LEN`), so no access leaves its
    // allocation.
    unsafe { b.launch(cfg)? };
    w.copy_from_slice(&g.stream.memcpy_dtov(&w_d)?);
    m.copy_from_slice(&g.stream.memcpy_dtov(&m_d)?);
    v.copy_from_slice(&g.stream.memcpy_dtov(&v_d)?);
    Ok(())
}

/// Device-buffer **AdamW** step (no host round-trip) — the resident-training entry point. Updates
/// `w`, `m`, `v` in place on the device from the device gradient `grad` and hyperparameters `hp`.
/// One launch over a contiguous `(w,g,m,v)` updates every parameter it spans.
///
/// The element count `n` is a *separate* parameter from the buffers, and every caller computes it at
/// the call site from the layer dims (`self.h * self.i`) rather than reading `w.len()` — so nothing
/// in the type system ties the two together. The asserts below do: an `n` longer than any of the five
/// buffers would otherwise launch a grid-stride kernel that reads and **writes** past the end of `w`,
/// `m` and `v` on the device, silently clobbering whatever cudarc allocated next (three
/// `st.global.f32` per element; no fault, just wrong weights on some later step). They are real
/// asserts, not `debug_assert`s, because the GPU path is exercised in release builds — four integer
/// comparisons are free next to a kernel launch.
pub fn adamw_step_device(
    g: &mut Gpu,
    w: &mut CudaSlice<f32>,
    grad: &CudaSlice<f32>,
    m: &mut CudaSlice<f32>,
    v: &mut CudaSlice<f32>,
    hp: &CudaSlice<f32>,
    n: usize,
) -> Result<(), DriverError> {
    assert!(
        n <= w.len() && n <= grad.len() && n <= m.len() && n <= v.len(),
        "adamw_step_device: n={n} exceeds a parameter buffer (w={}, grad={}, m={}, v={})",
        w.len(),
        grad.len(),
        m.len(),
        v.len()
    );
    assert!(
        hp.len() >= hp::LEN,
        "adamw_step_device: hp has {} entries, needs >= {}",
        hp.len(),
        hp::LEN
    );
    let f = g.function("adamw_step", ADAMW_STEP_PTX, "adamw_step")?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(w).arg(grad).arg(m).arg(v).arg(hp).arg(&n_u);
    // SAFETY: `adamw_step` is a grid-stride loop over `[0, n)` that reads `g[i]`/`hp[0..hp::LEN)` and
    // writes `w[i]`, `m[i]`, `v[i]` — and nothing else. The asserts above establish that each of the
    // five device buffers is at least that long, so no access leaves its allocation. The launch
    // config only sizes occupancy: the grid-stride loop covers `[0, n)` for any grid.
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Launch the fused **SGD** step (`w -= lr*(g + wd*w)`) on the GPU; `w` is updated in place. Uses
/// `hp[LR]` and `hp[WD]` from the [`hp`] layout.
pub fn sgd_step(g: &mut Gpu, w: &mut [f32], grad: &[f32], hp: &[f32]) -> Result<(), DriverError> {
    let n = w.len();
    assert_eq!(grad.len(), n, "sgd: grad length");
    assert!(
        hp.len() >= hp::LEN,
        "sgd: hp must have >= {} entries",
        hp::LEN
    );

    let f = g.function("sgd_step", SGD_STEP_PTX, "sgd_step")?;
    let mut w_d = g.stream.memcpy_stod(w)?;
    let g_d = g.stream.memcpy_stod(grad)?;
    let hp_d = g.stream.memcpy_stod(hp)?;
    let n_u = n as u32;
    let cfg = grid_stride_cfg(g, n_u);
    let mut b = g.stream.launch_builder(&f);
    b.arg(&mut w_d).arg(&g_d).arg(&hp_d).arg(&n_u);
    // SAFETY: `sgd_step` reads `g[i]`/`hp[LR]`/`hp[WD]` and writes `w[i]` for `i` in `[0, n)` only.
    // `n` is `w.len()` and the asserts above proved `grad` is the same length and `hp` is at least
    // `hp::LEN`, so every access stays inside the buffer it was uploaded from.
    unsafe { b.launch(cfg)? };
    w.copy_from_slice(&g.stream.memcpy_dtov(&w_d)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{assert_close, Rng};
    use crate::gpu::gpu;

    /// Run `body` with the process-wide GPU, or skip (no device) — mirrors the `gpu.rs` test guard.
    ///
    /// A skip is a *passing* libtest test whose `eprintln!` libtest captures, so without the
    /// `WUKONG_GPU_REQUIRED` escalation this whole module reports `test result: ok` — having run no
    /// kernel at all — whenever the device is unreachable for any reason (§3A P3).
    fn with_gpu(name: &str, body: impl FnOnce(&mut Gpu)) {
        let mut guard = gpu();
        match guard.as_mut() {
            Some(g) => body(g),
            None => {
                let why = crate::gpu::init_error().unwrap_or("no CUDA device reachable");
                assert!(
                    !crate::gpu::gpu_required(),
                    "{name}: WUKONG_GPU_REQUIRED is set but the GPU is unusable: {why}"
                );
                eprintln!("[skip] {name}: GPU unavailable: {why}");
            }
        }
    }

    /// **§3A P1 gate, device-free.** Both optimizer modules are literals sitting beside doc comments
    /// full of non-ASCII (`—`, `≤`, `·`, `⊙`); a single non-ASCII byte in the PTX is a `ptxas fatal`
    /// that surfaces on the device only as an opaque `cuModuleLoadData` `DriverError`. Checked here so
    /// it fails on any machine, with or without a GPU.
    #[test]
    fn optimizer_ptx_is_ascii() {
        for (what, ptx) in [
            ("ADAMW_STEP_PTX", ADAMW_STEP_PTX),
            ("SGD_STEP_PTX", SGD_STEP_PTX),
        ] {
            if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
                panic!(
                    "{what}: PTX line {} is not ASCII (ptxas fatal): {line:?}",
                    i + 1
                );
            }
            // Header floor. Both kernels are plain single-rounded f32 (`fma.rn`/`div.rn`/`sqrt.rn`)
            // over a grid-stride loop — nothing above `sm_80` — and PTX is forward-compatible only,
            // so the old `sm_89` tag bought nothing on Ada and made the module unloadable on every
            // A100. A `const` cannot interpolate `HDR_SM80`, so this is the tie to the single source.
            assert!(
                ptx.contains(crate::ptx_target::HDR_SM80),
                "{what}: must carry ptx_target::HDR_SM80 verbatim"
            );
            assert!(
                !ptx.contains(crate::ptx_target::TARGET_SM89),
                "{what}: an Ampere-legal module must not claim the Ada floor"
            );
        }
        assert!(ADAMW_STEP_PTX.contains(".visible .entry adamw_step("));
        assert!(SGD_STEP_PTX.contains(".visible .entry sgd_step("));
    }

    /// Fill a hyperparameter vector for step `t`.
    fn make_hp(lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, t: i32) -> Vec<f32> {
        let mut h = vec![0.0f32; hp::LEN];
        h[hp::LR] = lr;
        h[hp::BETA1] = beta1;
        h[hp::BETA2] = beta2;
        h[hp::EPS] = eps;
        h[hp::WD] = wd;
        h[hp::BC1] = 1.0 - (beta1 as f64).powi(t) as f32;
        h[hp::BC2] = 1.0 - (beta2 as f64).powi(t) as f32;
        h
    }

    /// **The gate:** the GPU AdamW kernel vs the *real* `wukong_autodiff::optim::build_adamw_step`
    /// MIR run on the interpreter oracle (`run_kernel_f32`). Both update `w`, `m`, `v` in place over
    /// several steps with a fixed gradient (exercising moment accumulation + bias correction); the
    /// device result must match the CPU autodiff reference at every element of all three buffers,
    /// within a tight tolerance — ~98% of lanes bit-exact, the rest a ≤1-ulp difference from the
    /// interpreter's f64-intermediate double-rounding (reported via the bit-exact lane count).
    #[test]
    fn adamw_step_matches_autodiff_reference() {
        use wukong_autodiff::optim::build_adamw_step;
        use wukong_interp::run_kernel_f32;
        use wukong_mir::{MirLevel, Program};
        use wukong_span::Interner;

        with_gpu("adamw_step_matches_autodiff_reference", |g| {
            let n = 1024usize;
            let mut it = Interner::default();
            let kernel = build_adamw_step(&mut it, n);
            let kname = kernel.name;
            let prog = Program {
                funcs: vec![kernel],
                statics: Vec::new(),
                level: MirLevel::Low,
            };

            let mut rng = Rng::new(0x4D11);
            // Start the CPU and GPU runs from identical state; step both in lockstep.
            let grad = rng.vec(n, -0.5, 0.5);
            let mut cw = rng.vec(n, -1.0, 1.0);
            let mut cm = vec![0.0f32; n];
            let mut cv = vec![0.0f32; n];
            let (mut gw, mut gm, mut gv) = (cw.clone(), cm.clone(), cv.clone());

            let (lr, b1, b2, eps, wd) = (0.01f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32);
            let mut bit_exact = 0usize;
            let total = 3 * n;
            for t in 1..=6i32 {
                let hpv = make_hp(lr, b1, b2, eps, wd, t);

                // CPU reference: run the emitted AdamW MIR on the interpreter.
                let mut bufs = [
                    cw.clone(),
                    grad.clone(),
                    cm.clone(),
                    cv.clone(),
                    hpv.clone(),
                ];
                let mut views: Vec<&mut [f32]> =
                    bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
                run_kernel_f32(&prog, kname, &mut views, &it).expect("cpu adamw run");
                cw = bufs[0].clone();
                cm = bufs[2].clone();
                cv = bufs[3].clone();

                // GPU kernel on the same inputs.
                adamw_step(g, &mut gw, &grad, &mut gm, &mut gv, &hpv).expect("gpu adamw");

                // Tolerance gate (robust); also count bit-exact lanes to substantiate the claim.
                assert_close(&format!("adamw w step {t}"), &gw, &cw, 1e-6, 1e-5);
                assert_close(&format!("adamw m step {t}"), &gm, &cm, 1e-6, 1e-5);
                assert_close(&format!("adamw v step {t}"), &gv, &cv, 1e-6, 1e-5);
                for (a, b) in gw
                    .iter()
                    .zip(&cw)
                    .chain(gm.iter().zip(&cm))
                    .chain(gv.iter().zip(&cv))
                {
                    if a.to_bits() == b.to_bits() {
                        bit_exact += 1;
                    }
                }
            }
            eprintln!(
                "adamw GPU vs autodiff MIR: {bit_exact}/{} buffer lanes bit-exact across 6 steps",
                total * 6
            );
        });
    }

    /// The GPU SGD kernel vs a plain-Rust f32 reference of the same `w -= lr*(g + wd*w)` formula.
    #[test]
    fn sgd_step_matches_reference() {
        with_gpu("sgd_step_matches_reference", |g| {
            let n = 777usize; // not a block multiple — exercise the grid-stride tail
            let mut rng = Rng::new(0x5D11);
            let grad = rng.vec(n, -1.0, 1.0);
            let w0 = rng.vec(n, -2.0, 2.0);
            let (lr, wd) = (0.05f32, 0.02f32);
            let hpv = make_hp(lr, 0.9, 0.999, 1e-8, wd, 1);

            let mut gw = w0.clone();
            sgd_step(g, &mut gw, &grad, &hpv).expect("gpu sgd");

            let cref: Vec<f32> = w0
                .iter()
                .zip(&grad)
                .map(|(&w, &gr)| {
                    let wdw = wd * w;
                    let t = gr + wdw;
                    let lt = lr * t;
                    w - lt
                })
                .collect();
            assert_close("sgd", &gw, &cref, 1e-7, 1e-6);
        });
    }

    /// **The `adamw_step_device` precondition gate.** `n` is passed independently of the five device
    /// buffers and every caller in `train_resident` computes it from the layer dims (`self.h *
    /// self.i`) rather than from `w.len()`, so nothing but this check stands between a padded or
    /// re-shaped weight allocation and a grid-stride kernel that *writes* `w`/`m`/`v` past their
    /// allocations. Asserts both that the exactly-sized call still launches and that an over-long `n`
    /// (or a short `hp`) is refused *before* the launch.
    #[test]
    fn adamw_step_device_refuses_n_past_the_buffers() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        }

        with_gpu("adamw_step_device_refuses_n_past_the_buffers", |g| {
            let n = 256usize;
            let zeros = vec![0.0f32; n];
            let hpv = make_hp(0.01, 0.9, 0.999, 1e-8, 0.0, 1);
            let mut w = g.stream.memcpy_stod(&zeros).unwrap();
            let grad = g.stream.memcpy_stod(&zeros).unwrap();
            let mut m = g.stream.memcpy_stod(&zeros).unwrap();
            let mut v = g.stream.memcpy_stod(&zeros).unwrap();
            let hp_d = g.stream.memcpy_stod(&hpv).unwrap();

            // Exactly-sized: launches normally.
            adamw_step_device(g, &mut w, &grad, &mut m, &mut v, &hp_d, n)
                .expect("exact n launches");

            // One element past the end: refused before the launch.
            let e = catch_unwind(AssertUnwindSafe(|| {
                let _ = adamw_step_device(g, &mut w, &grad, &mut m, &mut v, &hp_d, n + 1);
            }))
            .expect_err("n past the buffer length must be refused, not launched");
            let msg = panic_msg(e);
            assert!(
                msg.contains("exceeds a parameter buffer"),
                "unexpected panic: {msg}"
            );

            // A short hyperparameter buffer is refused too (the kernel reads hp[0..hp::LEN)).
            let short = g.stream.memcpy_stod(&hpv[..hp::LEN - 1]).unwrap();
            let e = catch_unwind(AssertUnwindSafe(|| {
                let _ = adamw_step_device(g, &mut w, &grad, &mut m, &mut v, &short, n);
            }))
            .expect_err("a short hp buffer must be refused");
            let msg = panic_msg(e);
            assert!(msg.contains("hp has"), "unexpected panic: {msg}");
            eprintln!("[gate] adamw_step_device bounds n by its five device buffers");
        });
    }

    /// A tiny convex problem trained purely by the GPU AdamW kernel: minimize `L = sum (w - t)^2`
    /// (gradient `2(w - t)`), driven for many steps. The loss must fall **strictly monotonically**
    /// and converge — the GPU optimizer reduces a real objective. (Full MLP fwd+bwd+optim residency
    /// is the increment-3 gate; this isolates the optimizer.)
    #[test]
    fn gpu_adamw_decreases_loss() {
        with_gpu("gpu_adamw_decreases_loss", |g| {
            let n = 4096usize;
            let mut rng = Rng::new(0x10AD);
            let target = rng.vec(n, -1.0, 1.0);
            let mut w = rng.vec(n, -3.0, 3.0); // start far from the optimum
            let mut m = vec![0.0f32; n];
            let mut v = vec![0.0f32; n];
            let (lr, b1, b2, eps, wd) = (0.05f32, 0.9f32, 0.999f32, 1e-8f32, 0.0f32);

            let loss = |w: &[f32]| -> f64 {
                w.iter()
                    .zip(&target)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                    .sum()
            };
            let mut prev = loss(&w);
            let l0 = prev;
            for t in 1..=200i32 {
                // grad of sum (w-t)^2 = 2(w-t)
                let grad: Vec<f32> = w
                    .iter()
                    .zip(&target)
                    .map(|(&a, &b)| 2.0 * (a - b))
                    .collect();
                let hpv = make_hp(lr, b1, b2, eps, wd, t);
                adamw_step(g, &mut w, &grad, &mut m, &mut v, &hpv).expect("gpu adamw");
                let cur = loss(&w);
                assert!(
                    cur < prev,
                    "loss not strictly decreasing at step {t}: {cur} >= {prev}"
                );
                prev = cur;
            }
            eprintln!("gpu adamw: loss {l0:.4e} -> {prev:.4e} over 200 steps");
            assert!(
                prev < l0 * 1e-3,
                "expected strong convergence; {prev} vs {l0}"
            );
        });
    }
}
