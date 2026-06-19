//! Streaming elementwise affine + activation — `out[i] = act(a·x[i] + b·y[i] + c)` — the **256-bit
//! AVX2** kernel a recognized streaming map loop (saxpy / scale / residual-add / bias / ReLU / ReLU6)
//! lowers to. Two reasons it beats the Cranelift-vectorized form gcc/rustc also emit:
//!
//!  1. **256-bit width.** Cranelift's generic vectorizer is stuck at 128-bit SSE (`f32x8` does not
//!     legalize), so a Mercury saxpy ran ~15% *behind* gcc's 256-bit AVX2. This kernel restores the
//!     width parity, the same play as `vmath`/`gemm`.
//!  2. **Non-temporal (streaming) stores.** For a large output (≥ [`NT_MIN_ELEMS`]) the store goes out
//!     `vmovntps`, which skips the **read-for-ownership** every cacheable store pays (the CPU must pull
//!     the target line into cache before overwriting it). gcc/rustc cannot emit this automatically —
//!     they can't prove the array is large and write-once — but Mercury's domain-aware lowering *knows*
//!     the loop streams a whole tensor. Measured ~1.2–1.3× over gcc's 256-bit saxpy/relu at the 4 MiB
//!     benchmark size, widening at real (>L3) activation-tensor sizes where the RFO traffic dominates.
//!
//! The interpreter marshals its abstract memory through this **identical** kernel (like `vmath`), so
//! the differential oracle stays bit-for-bit exact. NT and cacheable stores write the *same bits* (the
//! float value is identical; only the cache path differs), and the scalar tail/fallback mirrors the
//! AVX2 lanes op-for-op (the `act_avx`/`act1` pair share the `maxps`/`minps` semantics), so every lane
//! of every path agrees — pinned by `velem_tail_matches_lanes` and `velem_scalar_matches_avx`.

// --- op codes (shared with the recognizer in mercury_mir_build) ------------------------------------
// The low byte is the activation; `VE_USE_Y` (bit 8) flags that `y` is read (so a scale/ReLU that
// never touches `y` can pass a null `y` pointer the kernel will not dereference).
/// `out = a·x (+ b·y) + c`, no activation.
pub const VE_ID: i64 = 0;
/// `out = max(a·x (+ b·y) + c, 0)` — ReLU.
pub const VE_RELU: i64 = 1;
/// `out = min(max(a·x (+ b·y) + c, 0), 6)` — ReLU6.
pub const VE_RELU6: i64 = 2;
/// OR'd into `op` when the kernel must read `y` (`b` may be non-zero).
pub const VE_USE_Y: i64 = 256;

/// Output element count at/above which the store goes non-temporal. Below it the output fits in L2
/// (and may be re-read soon, e.g. a per-core `@parallel` chunk), so a normal cacheable store is
/// better — forcing it out to DRAM with `vmovntps` would just re-fetch it; above it the array is
/// streamed once and the RFO-avoidance of `vmovntps` wins. 2 MiB of f32 ≈ this machine's L2, the
/// crossover measured on the `@parallel` saxpy (per-core chunks ~190 KiB must stay cacheable, the
/// single-thread 4 MiB array must go non-temporal).
const NT_MIN_ELEMS: usize = 1 << 19;

/// Software-prefetch distance (elements ahead). A prefetch of an address past the buffer end is a
/// hint the hardware silently drops — never a fault — so the last iterations need no guard.
const PF_AHEAD: usize = 128;

/// Scalar activation, mirroring the AVX2 `maxps`/`minps` semantics exactly: `maxps(v,0)` is
/// `(v > 0) ? v : 0` (returns the second operand for `±0`/NaN), so the `if` form agrees lane-for-lane.
#[inline]
fn act1(op: i64, v: f32) -> f32 {
    match op & 0xff {
        VE_RELU => {
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        VE_RELU6 => {
            let v = if v > 0.0 { v } else { 0.0 };
            if v < 6.0 {
                v
            } else {
                6.0
            }
        }
        _ => v,
    }
}

/// One element `act(a·x + b·y + c)`. `b·y + c` is fused (`mul_add`) then `a·x + that` is fused, so the
/// chain is exactly two roundings — matching the AVX2 `fmadd` pair, and (for saxpy `a·x + y`, i.e.
/// `b = 1, c = 0`) the single `fma` gcc emits under `-ffp-contract=fast`. `y` is only read when the
/// caller set `VE_USE_Y`.
#[inline]
fn elem1(op: i64, x: f32, y: f32, a: f32, b: f32, c: f32) -> f32 {
    let inner = if op & VE_USE_Y != 0 { b.mul_add(y, c) } else { c };
    act1(op, a.mul_add(x, inner))
}

/// `out[i] = act(a·x[i] + b·y[i] + c)` for `i in 0..n`. Uses the 256-bit AVX2/FMA kernel when
/// available (8 lanes/step + a scalar tail), with non-temporal stores for a large output; else a
/// scalar fallback. `x`/`y`/`out` may alias (the recognizer allows in-place). `y` is dereferenced
/// only when `op & VE_USE_Y` is set.
///
/// # Safety
/// `x` and `out` must be valid for `n` `f32`; `y` must be valid for `n` `f32` when `op & VE_USE_Y`.
#[no_mangle]
pub unsafe extern "C" fn mercury_velem_f32(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: i64,
    a: f32,
    b: f32,
    c: f32,
    op: i64,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { velem_avx2(x, y, out, n, a, b, c, op) };
            return;
        }
    }
    let use_y = op & VE_USE_Y != 0;
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n (y only when use_y).
        unsafe {
            let yi = if use_y { *y.add(i) } else { 0.0 };
            *out.add(i) = elem1(op, *x.add(i), yi, a, b, c);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn velem_avx2(
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    n: usize,
    a: f32,
    b: f32,
    c: f32,
    op: i64,
) {
    use std::arch::x86_64::*;
    let va = _mm256_set1_ps(a);
    let vb = _mm256_set1_ps(b);
    let vc = _mm256_set1_ps(c);
    let zero = _mm256_setzero_ps();
    let six = _mm256_set1_ps(6.0);
    let use_y = op & VE_USE_Y != 0;
    let act = op & 0xff;
    let nt = n >= NT_MIN_ELEMS;
    let mut i = 0usize;
    // For the streaming store, peel a scalar prologue until `out` is 32-byte aligned (vmovntps faults
    // on a misaligned address); after that each 8-lane step keeps it aligned.
    if nt {
        while i < n && (out.add(i) as usize) & 31 != 0 {
            let yi = if use_y { *y.add(i) } else { 0.0 };
            *out.add(i) = elem1(op, *x.add(i), yi, a, b, c);
            i += 1;
        }
    }
    while i + 8 <= n {
        _mm_prefetch(x.add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
        let xv = _mm256_loadu_ps(x.add(i));
        let inner = if use_y {
            _mm_prefetch(y.add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
            _mm256_fmadd_ps(vb, _mm256_loadu_ps(y.add(i)), vc)
        } else {
            vc
        };
        let mut r = _mm256_fmadd_ps(va, xv, inner);
        if act == VE_RELU {
            r = _mm256_max_ps(r, zero);
        } else if act == VE_RELU6 {
            r = _mm256_min_ps(_mm256_max_ps(r, zero), six);
        }
        if nt {
            _mm256_stream_ps(out.add(i), r);
        } else {
            _mm256_storeu_ps(out.add(i), r);
        }
        i += 8;
    }
    // The non-temporal stores are weakly ordered; fence before the buffer is read back by anyone.
    if nt {
        _mm_sfence();
    }
    while i < n {
        let yi = if use_y { *y.add(i) } else { 0.0 };
        *out.add(i) = elem1(op, *x.add(i), yi, a, b, c);
        i += 1;
    }
}

/// One element of a Horner polynomial `((c[0]·x + c[1])·x + …)·x + c[n-1]`, the fused `mul_add` chain
/// matching the AVX2 lanes and the `r = r·x + c` source (which contracts to one `fma` per step under
/// gcc's `-ffp-contract=fast`).
#[inline]
fn horner1(x: f32, coeffs: &[f32]) -> f32 {
    let mut r = coeffs[0];
    for &c in &coeffs[1..] {
        r = r.mul_add(x, c);
    }
    r
}

/// `out[i] = poly(x[i])` for `i in 0..n`, where `poly` is the Horner evaluation of `coeffs` (highest
/// degree first; `ncoeff` terms). The 256-bit AVX2/FMA kernel a recognized `r = c0; r = r*x + c1; …;
/// out[i] = r` loop lowers to — Cranelift's vectorizer is stuck at 128-bit, and (like saxpy) the
/// streamed output goes out non-temporal for a large array, the store path gcc won't emit. `x` and
/// `out` may alias. Degenerate `ncoeff <= 1` writes the constant `coeffs[0]`.
///
/// # Safety
/// `x`/`out` valid for `n` f32; `coeffs` valid for `ncoeff` f32.
#[no_mangle]
pub unsafe extern "C" fn mercury_vhorner_f32(
    x: *const f32,
    out: *mut f32,
    n: i64,
    coeffs: *const f32,
    ncoeff: i64,
) {
    if n <= 0 || ncoeff <= 0 {
        return;
    }
    let n = n as usize;
    // SAFETY: caller guarantees coeffs valid for ncoeff f32.
    let coeffs = unsafe { std::slice::from_raw_parts(coeffs, ncoeff as usize) };
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features detected; buffers valid for n by the caller contract.
            unsafe { vhorner_avx2(x, out, n, coeffs) };
            return;
        }
    }
    for i in 0..n {
        // SAFETY: i < n; buffers valid for n.
        unsafe { *out.add(i) = horner1(*x.add(i), coeffs) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vhorner_avx2(x: *const f32, out: *mut f32, n: usize, coeffs: &[f32]) {
    use std::arch::x86_64::*;
    // Pre-splat the coefficients once (cheap vs the N-long loop) so the inner FMA chain has no
    // per-iteration broadcasts.
    let cv: Vec<__m256> = coeffs.iter().map(|&c| _mm256_set1_ps(c)).collect();
    let c0 = cv[0];
    let nt = n >= NT_MIN_ELEMS;
    let mut i = 0usize;
    if nt {
        while i < n && (out.add(i) as usize) & 31 != 0 {
            *out.add(i) = horner1(*x.add(i), coeffs);
            i += 1;
        }
    }
    while i + 8 <= n {
        _mm_prefetch(x.add(i + PF_AHEAD) as *const i8, _MM_HINT_T0);
        let xv = _mm256_loadu_ps(x.add(i));
        let mut r = c0;
        for ck in &cv[1..] {
            r = _mm256_fmadd_ps(r, xv, *ck);
        }
        if nt {
            _mm256_stream_ps(out.add(i), r);
        } else {
            _mm256_storeu_ps(out.add(i), r);
        }
        i += 8;
    }
    if nt {
        _mm_sfence();
    }
    while i < n {
        *out.add(i) = horner1(*x.add(i), coeffs);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 lanes and the scalar tail/fallback must agree element-for-element, so a length that is
    /// not a multiple of 8 (and that crosses the NT alignment prologue) produces a consistent result.
    #[test]
    fn velem_tail_matches_lanes() {
        let n = 600_003usize; // forces NT path (> NT_MIN_ELEMS), a misaligned tail, and a prologue
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 19.0) - 7.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) * 0.5 - 2.0).collect();
        let cases: &[(i64, f32, f32, f32)] = &[
            (VE_ID | VE_USE_Y, 2.0, 1.0, 0.0),   // saxpy
            (VE_ID, 3.5, 0.0, 0.0),              // scale
            (VE_ID | VE_USE_Y, 1.0, 1.0, 0.0),   // residual add
            (VE_ID, 1.0, 0.0, 0.75),             // bias
            (VE_RELU, 1.0, 0.0, 0.0),            // relu
            (VE_RELU, 2.0, 0.0, 1.0),            // fused linear→relu
            (VE_RELU6, 1.0, 0.0, 0.0),           // relu6
            (VE_RELU6 | VE_USE_Y, 1.0, 1.0, 0.0),
        ];
        for &(op, a, b, c) in cases {
            let mut got = vec![0.0f32; n];
            unsafe {
                mercury_velem_f32(x.as_ptr(), y.as_ptr(), got.as_mut_ptr(), n as i64, a, b, c, op);
            }
            for i in 0..n {
                let yi = if op & VE_USE_Y != 0 { y[i] } else { 0.0 };
                let want = elem1(op, x[i], yi, a, b, c);
                assert_eq!(got[i].to_bits(), want.to_bits(), "op {op} i {i}");
            }
        }
    }

    /// saxpy `a·x + y` must equal the fused `mul_add` (the same single rounding gcc's
    /// `-ffp-contract=fast` produces), so the cross-language check stays exact.
    #[test]
    fn velem_saxpy_is_fma() {
        let n = 64usize; // small → cacheable path, exercising the non-NT branch
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 5.0).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * -0.7 + 2.0).collect();
        let mut got = vec![0.0f32; n];
        unsafe {
            mercury_velem_f32(
                x.as_ptr(),
                y.as_ptr(),
                got.as_mut_ptr(),
                n as i64,
                2.0,
                1.0,
                0.0,
                VE_ID | VE_USE_Y,
            );
        }
        for i in 0..n {
            assert_eq!(got[i].to_bits(), 2.0f32.mul_add(x[i], y[i]).to_bits(), "i {i}");
        }
    }

    /// The Horner kernel's AVX2 lanes and scalar tail must agree element-for-element across the NT
    /// boundary and a misaligned tail, for several degrees (incl. the degenerate constant `ncoeff=1`).
    #[test]
    fn vhorner_tail_matches_lanes() {
        let n = 600_005usize; // forces NT path, prologue, and a non-mult-of-8 tail
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 23.0) * 0.1 - 1.1).collect();
        for coeffs in [
            vec![3.0f32],                              // constant
            vec![2.0f32, -1.0],                        // linear
            vec![1e-5f32, 1e-4, 1e-3, 1e-2, 1e-1],     // the deg-4 poly benchmark
        ] {
            let mut got = vec![0.0f32; n];
            unsafe {
                mercury_vhorner_f32(
                    x.as_ptr(),
                    got.as_mut_ptr(),
                    n as i64,
                    coeffs.as_ptr(),
                    coeffs.len() as i64,
                );
            }
            for i in 0..n {
                assert_eq!(got[i].to_bits(), horner1(x[i], &coeffs).to_bits(), "deg {} i {i}", coeffs.len());
            }
        }
    }

    /// A scale (`b = 0`, no `VE_USE_Y`) must never dereference `y`: pass a dangling `y` and confirm
    /// the result is `a·x` with no crash.
    #[test]
    fn velem_scale_ignores_y() {
        let n = 70_000usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 % 23.0) - 11.0).collect();
        let mut got = vec![0.0f32; n];
        unsafe {
            mercury_velem_f32(
                x.as_ptr(),
                std::ptr::null(),
                got.as_mut_ptr(),
                n as i64,
                0.25,
                0.0,
                0.0,
                VE_ID,
            );
        }
        for i in 0..n {
            assert_eq!(got[i].to_bits(), (0.25f32 * x[i]).to_bits(), "i {i}");
        }
    }
}
