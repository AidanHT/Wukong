//! Low-precision (bf16 / f16) CPU reduction kernels with an **f32 accumulator** — the standard
//! mixed-precision contract. On this AVX2+F16C box there is no native bf16/f16 MAC, so low precision
//! is a **bandwidth / footprint** win, not a FLOP/s one: a reduction over half-width inputs streams
//! half the bytes, so a memory-bound reduction runs ~2× faster than its f32 twin. We widen on load
//! (F16C `vcvtph2ps` for f16; a `<<16` bit-extend for bf16 — both *lossless*) and accumulate in f32.
//!
//! Determinism / twin contract: the SIMD path keeps 8 lane accumulators (lane `l` sums elements
//! `8k+l` in ascending `k`) and combines them with a fixed tree; the scalar twin keeps the identical
//! 8 logical lanes, so SIMD == scalar **bit-for-bit** (a test pins this across partial tails). The
//! widen is exact, so the only float rounding is the f32 accumulation — identical in both paths.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::bf16_bits_to_f32;

/// IEEE f16 (stored bits) → f32. Lossless, so it equals the F16C `vcvtph2ps` result exactly.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}

/// Fixed 8-lane horizontal combine tree (the same order the scalar twin and the SIMD path use).
#[inline]
fn hcombine8(l: &[f32; 8]) -> f32 {
    ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
}

/// Element kind for the generic scalar twin.
#[derive(Clone, Copy)]
enum Half {
    F16,
    Bf16,
}

impl Half {
    #[inline]
    fn widen(self, bits: u16) -> f32 {
        match self {
            Half::F16 => f16_to_f32(bits),
            Half::Bf16 => bf16_bits_to_f32(bits),
        }
    }
}

/// Scalar twin of the SIMD sum: 8 logical lane accumulators, fixed combine, scalar tail. This is the
/// reference the SIMD path must match bit-for-bit, and the no-AVX2 fallback.
fn sum_scalar(kind: Half, x: &[u16]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        for (l, a) in acc.iter_mut().enumerate() {
            *a += kind.widen(x[c * 8 + l]);
        }
    }
    let mut s = hcombine8(&acc);
    for &v in &x[chunks * 8..] {
        s += kind.widen(v);
    }
    s
}

/// Scalar twin of the SIMD dot (f32-accumulated, fused like the SIMD `fmadd`).
fn dot_scalar(kind: Half, x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        for (l, a) in acc.iter_mut().enumerate() {
            *a = kind
                .widen(x[c * 8 + l])
                .mul_add(kind.widen(y[c * 8 + l]), *a);
        }
    }
    let mut s = hcombine8(&acc);
    for i in chunks * 8..n {
        s = kind.widen(x[i]).mul_add(kind.widen(y[i]), s);
    }
    s
}

// ---- SIMD widen helpers (8 lanes) ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
unsafe fn widen_f16(p: *const u16) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
}

/// `pub(crate)` so `vmath.rs`'s bf16-input activation kernel (`mercury_vmath_bf16`) widens with the
/// *identical* lossless `<<16` bit-extend this module's reductions use — one source of truth for the
/// bf16→f32 widen keeps every bf16 dispatch path bit-for-bit consistent.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn widen_bf16(p: *const u16) -> __m256 {
    // zero-extend 8×u16 → 8×u32, shift the bf16 bits into the f32 high half, reinterpret.
    let lo = _mm_loadu_si128(p as *const __m128i);
    let w = _mm256_cvtepu16_epi32(lo);
    _mm256_castsi256_ps(_mm256_slli_epi32(w, 16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn sum_f16_avx(x: &[u16]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        acc = _mm256_add_ps(acc, widen_f16(x.as_ptr().add(c * 8)));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(&lanes);
    for &v in &x[chunks * 8..] {
        s += f16_to_f32(v);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_bf16_avx(x: &[u16]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        acc = _mm256_add_ps(acc, widen_bf16(x.as_ptr().add(c * 8)));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(&lanes);
    for &v in &x[chunks * 8..] {
        s += bf16_bits_to_f32(v);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c,fma")]
unsafe fn dot_f16_avx(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        let a = widen_f16(x.as_ptr().add(c * 8));
        let b = widen_f16(y.as_ptr().add(c * 8));
        acc = _mm256_fmadd_ps(a, b, acc);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(&lanes);
    for i in chunks * 8..n {
        s = f16_to_f32(x[i]).mul_add(f16_to_f32(y[i]), s);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_avx(x: &[u16], y: &[u16]) -> f32 {
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        let a = widen_bf16(x.as_ptr().add(c * 8));
        let b = widen_bf16(y.as_ptr().add(c * 8));
        acc = _mm256_fmadd_ps(a, b, acc);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(&lanes);
    for i in chunks * 8..n {
        s = bf16_bits_to_f32(x[i]).mul_add(bf16_bits_to_f32(y[i]), s);
    }
    s
}

// ---- bf16 max / min / absmax (the max-family folds; widen is lossless and max/min round nothing,
//      so the result is the *exact* reduction of the widened values — no f32-accumulator caveat) ----

/// One widened element folded into a max/min/maxabs accumulator, matching `reduce.rs::contrib` for
/// these ops so the bf16 max-family fold is consistent with the f32 reduction's semantics (and the
/// scalar twin matches the AVX2 lanes: `f32::abs` clears the sign bit ≡ `andnot(-0.0, v)`).
#[inline(always)]
fn rfold(a: f32, xi: f32, op: i64) -> f32 {
    use crate::reduce::{fold2, RED_MAX, RED_MAXABS};
    if op == RED_MAXABS {
        fold2(a, xi.abs(), RED_MAX)
    } else {
        fold2(a, xi, op)
    }
}

/// Scalar twin of the bf16 max/min/maxabs reduction: 8 logical lane accumulators (lane `l` folds
/// elements `8k+l`), the fixed [`crate::reduce::hcombine8`] combine, then a scalar tail — bit-identical
/// to the AVX2 path (a test pins it) and the no-AVX2 fallback.
fn reduce_minmax_scalar(op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = [ident(op); 8];
    for c in 0..chunks {
        for (l, a) in acc.iter_mut().enumerate() {
            *a = rfold(*a, bf16_bits_to_f32(x[c * 8 + l]), op);
        }
    }
    let mut s = hcombine8(acc, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, bf16_bits_to_f32(v), op);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_minmax_avx(op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident, RED_MAXABS, RED_MIN};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_set1_ps(ident(op));
    let absmask = _mm256_set1_ps(-0.0); // andnot(-0.0, v) clears the sign bit = |v|
    for c in 0..chunks {
        let mut v = widen_bf16(x.as_ptr().add(c * 8));
        if op == RED_MAXABS {
            v = _mm256_andnot_ps(absmask, v);
        }
        acc = if op == RED_MIN {
            _mm256_min_ps(acc, v)
        } else {
            _mm256_max_ps(acc, v) // RED_MAX and the (already-abs'd) RED_MAXABS both fold by max
        };
    }
    let mut lanes = [ident(op); 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(lanes, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, bf16_bits_to_f32(v), op);
    }
    s
}

/// `reduce_i widen(x[i])` over `n` bf16 values for the **max-family** ops — `RED_MAX` (per-tensor
/// max, e.g. the softmax-stability shift), `RED_MIN`, and `RED_MAXABS` (the symmetric-quantization
/// absmax scale a `[bf16]` weight tensor's int8 export needs). f32 result; half the bytes of the f32
/// reduction. The widen is lossless and the folds round nothing, so this is the *exact* reduction of
/// the widened values — the interpreter marshals through this very kernel, so interp == native.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn mercury_reduce_bf16(x: *const u16, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return reduce_minmax_avx(op, x);
    }
    reduce_minmax_scalar(op, x)
}

// ---- Public C-ABI entry points (f32-accumulated) ----

/// `sum(widen(x[i]))` over `n` IEEE-f16 values (stored as `u16` bits), accumulated in f32.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn mercury_sum_f16(x: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
        return sum_f16_avx(x);
    }
    sum_scalar(Half::F16, x)
}

/// `sum(widen(x[i]))` over `n` bf16 values (stored as `u16` bits), accumulated in f32.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn mercury_sum_bf16(x: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return sum_bf16_avx(x);
    }
    sum_scalar(Half::Bf16, x)
}

/// `sum(widen(x[i])·widen(y[i]))` over `n` IEEE-f16 values, accumulated in f32.
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn mercury_dot_f16(x: *const u16, y: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("fma") {
        return dot_f16_avx(x, y);
    }
    dot_scalar(Half::F16, x, y)
}

/// `sum(widen(x[i])·widen(y[i]))` over `n` bf16 values, accumulated in f32.
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn mercury_dot_bf16(x: *const u16, y: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return dot_bf16_avx(x, y);
    }
    dot_scalar(Half::Bf16, x, y)
}

/// Scalar twin of the bf16→f32 axpby: `out[i] = a·widen(x[i]) + b·widen(y[i])`. The op order
/// (`t = a·xw` then `fma(b, yw, t)`) is chosen to match the AVX2 path's `mul`+`fmadd` lane-for-lane,
/// so SIMD == scalar bit-for-bit (the twin test pins this); it's also the no-AVX2 fallback.
fn axpby_bf16_scalar(kind: Half, x: &[u16], y: &[u16], out: &mut [f32], a: f32, b: f32) {
    for i in 0..out.len() {
        let xw = kind.widen(x[i]);
        let yw = kind.widen(y[i]);
        out[i] = b.mul_add(yw, a * xw); // b·yw + a·xw, the same op order as the fmadd path
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpby_bf16_avx(x: &[u16], y: &[u16], out: &mut [f32], a: f32, b: f32) {
    let n = out.len();
    let chunks = n / 8;
    let av = _mm256_set1_ps(a);
    let bv = _mm256_set1_ps(b);
    for c in 0..chunks {
        let xw = widen_bf16(x.as_ptr().add(c * 8));
        let yw = widen_bf16(y.as_ptr().add(c * 8));
        let t = _mm256_mul_ps(av, xw);
        let r = _mm256_fmadd_ps(bv, yw, t); // b·yw + a·xw
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), r);
    }
    for i in chunks * 8..n {
        let xw = bf16_bits_to_f32(x[i]);
        let yw = bf16_bits_to_f32(y[i]);
        out[i] = b.mul_add(yw, a * xw);
    }
}

/// `out[i] = a·widen(x[i]) + b·widen(y[i])` over `n` **bf16** inputs (stored as `u16` bits) with an
/// **f32 output** — the mixed-precision streaming elementwise (saxpy/axpby) twin of the f32
/// `mercury_velem_f32`. bf16 in + f32 out streams 8 bytes/elem vs the f32 kernel's 12, so on a
/// memory-bound elementwise it runs ~1.5× faster. Compute is f32 (widen is lossless), so the only
/// rounding is the f32 math — identical in the AVX2 and scalar paths (twin-tested).
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `f32`.
#[no_mangle]
pub unsafe extern "C" fn mercury_axpby_bf16(
    x: *const u16,
    y: *const u16,
    out: *mut f32,
    n: i64,
    a: f32,
    b: f32,
) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    let x = std::slice::from_raw_parts(x, n);
    let y = std::slice::from_raw_parts(y, n);
    let out = std::slice::from_raw_parts_mut(out, n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return axpby_bf16_avx(x, y, out, a, b);
    }
    axpby_bf16_scalar(Half::Bf16, x, y, out, a, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bits(x: f32) -> u16 {
        crate::f32_to_bf16_bits(x)
    }
    fn f16_bits(x: f32) -> u16 {
        half::f16::from_f32(x).to_bits()
    }

    #[test]
    fn simd_equals_scalar_twin_bit_for_bit() {
        // include a non-multiple-of-8 tail
        for n in [0usize, 1, 7, 8, 9, 100, 1000, 4099] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013 - 3.1).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.019 + 1.7).cos()).collect();
            let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
            let yf16: Vec<u16> = ys.iter().map(|&v| f16_bits(v)).collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();

            unsafe {
                assert_eq!(
                    mercury_sum_f16(xf16.as_ptr(), n as i64).to_bits(),
                    sum_scalar(Half::F16, &xf16).to_bits(),
                    "sum_f16 n={n}"
                );
                assert_eq!(
                    mercury_sum_bf16(xbf.as_ptr(), n as i64).to_bits(),
                    sum_scalar(Half::Bf16, &xbf).to_bits(),
                    "sum_bf16 n={n}"
                );
                assert_eq!(
                    mercury_dot_f16(xf16.as_ptr(), yf16.as_ptr(), n as i64).to_bits(),
                    dot_scalar(Half::F16, &xf16, &yf16).to_bits(),
                    "dot_f16 n={n}"
                );
                assert_eq!(
                    mercury_dot_bf16(xbf.as_ptr(), ybf.as_ptr(), n as i64).to_bits(),
                    dot_scalar(Half::Bf16, &xbf, &ybf).to_bits(),
                    "dot_bf16 n={n}"
                );
            }
        }
    }

    #[test]
    fn reduce_bf16_simd_equals_scalar_and_reference() {
        use crate::reduce::{ident, RED_MAX, RED_MAXABS, RED_MIN};
        // fold matching `reduce.rs::fold2` exactly, so the reference picks the same element on ties.
        let fold = |a: f32, v: f32, op: i64| -> f32 {
            match op {
                RED_MIN => {
                    if a < v {
                        a
                    } else {
                        v
                    }
                }
                _ => {
                    if a > v {
                        a
                    } else {
                        v
                    }
                }
            }
        };
        for n in [0usize, 1, 7, 8, 9, 100, 1000, 4099] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013 - 7.0).sin() * 3.0).collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            for op in [RED_MAX, RED_MIN, RED_MAXABS] {
                let got = unsafe { mercury_reduce_bf16(xbf.as_ptr(), n as i64, op) };
                // AVX2 == scalar twin, bit-for-bit.
                let twin = reduce_minmax_scalar(op, &xbf);
                assert_eq!(got.to_bits(), twin.to_bits(), "twin op {op} n {n}");
                // == the exact reduction of the widened values (max/min round nothing).
                let mut want = ident(op);
                for &b in &xbf {
                    let v = bf16_bits_to_f32(b);
                    let v = if op == RED_MAXABS { v.abs() } else { v };
                    want = fold(want, v, if op == RED_MAXABS { RED_MAX } else { op });
                }
                assert_eq!(got.to_bits(), want.to_bits(), "ref op {op} n {n}");
            }
        }
    }

    #[test]
    fn axpby_bf16_simd_equals_scalar_twin() {
        for n in [0usize, 1, 7, 8, 9, 100, 1000, 4099] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 2.3).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.9).cos()).collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
            let (a, b) = (1.5f32, -0.75f32);
            let mut got = vec![0f32; n];
            unsafe {
                mercury_axpby_bf16(xbf.as_ptr(), ybf.as_ptr(), got.as_mut_ptr(), n as i64, a, b);
            }
            let mut want = vec![0f32; n];
            axpby_bf16_scalar(Half::Bf16, &xbf, &ybf, &mut want, a, b);
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "axpby_bf16 n={n} i={i}"
                );
            }
        }
    }

    #[test]
    fn axpby_bf16_within_ulp_of_f64_reference() {
        let n = 1 << 14;
        let xs: Vec<f32> = (0..n).map(|i| ((i % 89) as f32) * 0.03 - 1.0).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 61) as f32) * 0.02 - 0.5).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let (a, b) = (0.6f32, 1.3f32);
        let mut got = vec![0f32; n];
        unsafe {
            mercury_axpby_bf16(xbf.as_ptr(), ybf.as_ptr(), got.as_mut_ptr(), n as i64, a, b);
        }
        // f64 reference over the SAME widened bf16 inputs (so this isolates the f32 math error).
        let mut max_rel = 0.0f64;
        for i in 0..n {
            let r = a as f64 * bf16_bits_to_f32(xbf[i]) as f64
                + b as f64 * bf16_bits_to_f32(ybf[i]) as f64;
            let rel = (got[i] as f64 - r).abs() / r.abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-6, "axpby_bf16 rel {max_rel:.2e}");
    }

    /// Bandwidth win: a reduction over half-width inputs streams half the bytes, so for a working
    /// set well past L3 it runs ~2× faster than the f32 reduction. Compares Mercury's SIMD bf16/f16
    /// sum to Mercury's tuned f32 sum AND to a plain (rustc-autovectorized) Rust f32/bf16 sum.
    /// Run: `cargo test -p mercury_runtime --release lowp_bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore = "bandwidth bench; run explicitly in --release"]
    fn lowp_bandwidth() {
        use std::time::Instant;
        let n = 32 << 20; // 32M elements — 128 MB f32, 64 MB half, both ≫ L3
        let xs: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) * 0.001).collect();
        let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();

        let best = |iters: usize, mut f: Box<dyn FnMut() -> f32>| -> f64 {
            let _ = f(); // warmup
            let mut t = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                std::hint::black_box(f());
                t = t.min(t0.elapsed().as_secs_f64());
            }
            t
        };

        let xs2 = xs.clone();
        let t_rust_f32 = best(5, Box::new(move || xs2.iter().copied().sum()));
        let xp = xs.as_ptr() as usize;
        let t_merc_f32 = best(
            5,
            Box::new(move || unsafe {
                crate::mercury_sreduce_f32(
                    xp as *const f32,
                    xp as *const f32,
                    n as i64,
                    crate::RED_SUM,
                )
            }),
        );
        let xbf2 = xbf.clone();
        let t_rust_bf16 = best(
            5,
            Box::new(move || xbf2.iter().map(|&b| bf16_bits_to_f32(b)).sum()),
        );
        let bp = xbf.as_ptr() as usize;
        let t_merc_bf16 = best(
            5,
            Box::new(move || unsafe { mercury_sum_bf16(bp as *const u16, n as i64) }),
        );
        let fp = xf16.as_ptr() as usize;
        let t_merc_f16 = best(
            5,
            Box::new(move || unsafe { mercury_sum_f16(fp as *const u16, n as i64) }),
        );

        let gbps = |bytes: f64, t: f64| bytes / t / 1e9;
        let nf = n as f64;
        eprintln!("sum over {n} elements (best of 5):");
        eprintln!(
            "  rust f32 (autovec): {:.2} ms  {:.0} GB/s",
            t_rust_f32 * 1e3,
            gbps(4.0 * nf, t_rust_f32)
        );
        eprintln!(
            "  merc f32 (AVX2):    {:.2} ms  {:.0} GB/s",
            t_merc_f32 * 1e3,
            gbps(4.0 * nf, t_merc_f32)
        );
        eprintln!(
            "  rust bf16 (autovec):{:.2} ms  {:.0} GB/s",
            t_rust_bf16 * 1e3,
            gbps(2.0 * nf, t_rust_bf16)
        );
        eprintln!(
            "  merc bf16 (AVX2):   {:.2} ms  {:.0} GB/s  → {:.2}× vs merc-f32",
            t_merc_bf16 * 1e3,
            gbps(2.0 * nf, t_merc_bf16),
            t_merc_f32 / t_merc_bf16
        );
        eprintln!(
            "  merc f16  (F16C):   {:.2} ms  {:.0} GB/s  → {:.2}× vs merc-f32",
            t_merc_f16 * 1e3,
            gbps(2.0 * nf, t_merc_f16),
            t_merc_f32 / t_merc_f16
        );
    }

    /// Bandwidth win for the streaming **axpby** (`out = a·x + b·y`): bf16 inputs + f32 output move
    /// 8 bytes/elem vs an all-f32 axpby's 12, so on a working set ≫ L3 the bf16-in kernel runs ~1.5×
    /// faster. Compares `mercury_axpby_bf16` to a plain (rustc-autovectorized) f32 axpby.
    /// Run: `cargo test -p mercury_runtime --release axpby_bf16_bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore = "bandwidth bench; run explicitly in --release"]
    fn axpby_bf16_bandwidth() {
        use std::time::Instant;
        let n = 32 << 20; // 32M elems — bf16 in 128 MB, f32 in 256 MB, both ≫ L3
        let xs: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) * 0.001).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 199) as f32) * 0.002).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let (a, b) = (1.5f32, 0.75f32);
        let best = |iters: usize, mut f: Box<dyn FnMut()>| -> f64 {
            f();
            let mut t = f64::INFINITY;
            for _ in 0..iters {
                let t0 = Instant::now();
                f();
                t = t.min(t0.elapsed().as_secs_f64());
            }
            t
        };

        let (xp, yp) = (xbf.as_ptr() as usize, ybf.as_ptr() as usize);
        let mut out_bf = vec![0f32; n];
        let obp = out_bf.as_mut_ptr() as usize;
        let t_bf = best(
            5,
            Box::new(move || unsafe {
                mercury_axpby_bf16(
                    xp as *const u16,
                    yp as *const u16,
                    obp as *mut f32,
                    n as i64,
                    a,
                    b,
                );
                std::hint::black_box(obp);
            }),
        );
        let (xfp, yfp) = (xs.as_ptr() as usize, ys.as_ptr() as usize);
        let mut out_f = vec![0f32; n];
        let ofp = out_f.as_mut_ptr() as usize;
        let t_f = best(
            5,
            Box::new(move || unsafe {
                let x = std::slice::from_raw_parts(xfp as *const f32, n);
                let y = std::slice::from_raw_parts(yfp as *const f32, n);
                let o = std::slice::from_raw_parts_mut(ofp as *mut f32, n);
                for i in 0..n {
                    o[i] = a * x[i] + b * y[i];
                }
                std::hint::black_box(ofp);
            }),
        );
        let gbps = |bytes: f64, t: f64| bytes / t / 1e9;
        let nf = n as f64;
        eprintln!("axpby out=a·x+b·y over {n} elements (best of 5):");
        eprintln!(
            "  rust f32  (autovec, 12 B/elem): {:.2} ms  {:.0} GB/s",
            t_f * 1e3,
            gbps(12.0 * nf, t_f)
        );
        eprintln!(
            "  merc bf16-in/f32-out (8 B/elem): {:.2} ms  {:.0} GB/s  → {:.2}× faster",
            t_bf * 1e3,
            gbps(8.0 * nf, t_bf),
            t_f / t_bf
        );
    }

    #[test]
    fn within_ulp_tolerance_of_f64_reference() {
        let n = 1 << 16;
        let xs: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.01).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 53) as f32) * 0.02).collect();
        // f16 sum
        let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
        let ref_sum: f64 = xf16.iter().map(|&b| f16_to_f32(b) as f64).sum();
        let got = unsafe { mercury_sum_f16(xf16.as_ptr(), n as i64) };
        let rel = (got as f64 - ref_sum).abs() / ref_sum.abs().max(1.0);
        assert!(rel < 1e-3, "f16 sum rel {rel:.2e}");
        // bf16 dot
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let ref_dot: f64 = xbf
            .iter()
            .zip(&ybf)
            .map(|(&a, &b)| bf16_bits_to_f32(a) as f64 * bf16_bits_to_f32(b) as f64)
            .sum();
        let got = unsafe { mercury_dot_bf16(xbf.as_ptr(), ybf.as_ptr(), n as i64) };
        let rel = (got as f64 - ref_dot).abs() / ref_dot.abs().max(1.0);
        assert!(rel < 1e-2, "bf16 dot rel {rel:.2e}");
    }
}
