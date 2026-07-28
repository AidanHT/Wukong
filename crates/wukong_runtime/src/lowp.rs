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

use rayon::prelude::*;

use crate::bf16_bits_to_f32;

/// IEEE f16 (stored bits) → f32. Lossless, so it equals the F16C `vcvtph2ps` result exactly. Thin
/// alias for the crate-root [`crate::f16_bits_to_f32`] — one shared definition of the f16 widen.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    crate::f16_bits_to_f32(h)
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

    /// Round an `f32` to this half type's stored bits — the shared `wukong_f32_to_{bf16,f16}_bits`
    /// shim the interpreter and the native narrowing store both use, so a half-**output** kernel is
    /// bit-for-bit consistent with a scalar `<f32> as bf16/f16` store (round at the value, not the
    /// store — the sweep-6 lesson).
    #[inline]
    fn narrow(self, v: f32) -> u16 {
        match self {
            Half::F16 => crate::f32_to_f16_bits(v),
            Half::Bf16 => crate::f32_to_bf16_bits(v),
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

/// `pub(crate)` so `vmath.rs`'s f16-input activation kernel widens with the *identical* F16C
/// `vcvtph2ps` this module's f16 reductions use — one source of truth for the f16 widen.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
pub(crate) unsafe fn widen_f16(p: *const u16) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
}

/// `pub(crate)` so `vmath.rs`'s bf16-input activation kernel (`wukong_vmath_bf16`) widens with the
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

// ---- SIMD narrowing store helpers (8 f32 lanes → 8 contiguous half bits) — the reverse of the widen
//      helpers, used by the half-**output** streaming kernels. Each is bit-identical to the scalar
//      `f32_to_{bf16,f16}_bits` shim the interpreter narrows through (twin-tested), so a half-output
//      kernel stays bit-for-bit consistent with the differential oracle. `pub(crate)` so `velem.rs`'s
//      and `bias.rs`'s narrowing stores share the one definition. ----

/// Round 8 f32 lanes to bf16 stored bits, returned as 8× `u32` (each bf16 value in its low 16 bits) —
/// the store-free core shared by [`narrow_bf16`] (8-lane store) and [`narrow_bf16_pack`] (16-lane NT
/// pack). Round-to-nearest-even `(bits + 0x7fff + ((bits>>16)&1)) >> 16`, with a blend to the
/// quieted-NaN form `(bits>>16)|0x40` on unordered lanes — bit-identical to [`crate::f32_to_bf16_bits`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bf16_round8(v: __m256) -> __m256i {
    let bits = _mm256_castps_si256(v);
    let lsb = _mm256_and_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(1));
    let bias = _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff));
    let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(bits, bias));
    let nan_res = _mm256_or_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(0x40));
    let is_nan = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
    _mm256_blendv_epi8(rounded, nan_res, is_nan) // per-32-bit-lane mask (all-ones on NaN)
}

/// Round 8 f32 lanes to bf16 stored bits and store them as 8 contiguous `u16` at `out` — bit-identical
/// to [`crate::f32_to_bf16_bits`] applied lane-by-lane. Used for the ≤8-lane tail of the streaming
/// kernels (the bulk goes through [`narrow_bf16_pack`] + a 256-bit store).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn narrow_bf16(v: __m256, out: *mut u16) {
    let res = bf16_round8(v);
    // Pack the 8× u32 (each a u16 in its low half) to 8× u16, then gather the two low quadwords.
    let packed = _mm256_packus_epi32(res, res); // [r0..3 r0..3 | r4..7 r4..7] as u16
    let lo = _mm256_permute4x64_epi64::<0b0000_1000>(packed); // qwords [q0, q2, _, _] = r0..7
    _mm_storeu_si128(out as *mut __m128i, _mm256_castsi256_si128(lo));
}

/// Round **16** f32 lanes (two `__m256`) to 16 contiguous bf16 stored bits packed in one `__m256i` — the
/// 256-bit core of the streaming half-output kernel, so a whole cache-line-friendly 32-byte block is
/// produced per store (enables a single non-temporal `vmovntdq`). Each 8-lane half is
/// [`bf16_round8`], so it is bit-identical to [`narrow_bf16`] (and thus the scalar shim) per lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn narrow_bf16_pack(v0: __m256, v1: __m256) -> __m256i {
    let r0 = bf16_round8(v0);
    let r1 = bf16_round8(v1);
    // packus interleaves per 128-bit lane: [r0.0-3 r1.0-3 | r0.4-7 r1.4-7]; permute qwords (0,2,1,3)
    // → [r0.0-7 r1.0-7], 16 contiguous bf16.
    let packed = _mm256_packus_epi32(r0, r1);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

/// Round 8 f32 lanes to IEEE-f16 stored bits and store them as 8 contiguous `u16` — F16C `vcvtps2ph`
/// (round to nearest even), which the `simd_equals_scalar_twin` test pins == [`crate::f32_to_f16_bits`]
/// (the `half` crate). So the f16 narrowing store matches the shim the interpreter uses. Used for the
/// ≤8-lane tail (the bulk goes through [`narrow_f16_pack`] + a 256-bit store).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
pub(crate) unsafe fn narrow_f16(v: __m256, out: *mut u16) {
    // `vcvtps2ph` rounding imm is 3 bits; round-to-nearest-even (`_MM_FROUND_TO_NEAREST_INT` = 0) is
    // what `half::f16::from_f32` uses (the narrow twin test pins F16C == the shim bit-for-bit).
    let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
    _mm_storeu_si128(out as *mut __m128i, h);
}

/// Round **16** f32 lanes (two `__m256`) to 16 contiguous f16 stored bits packed in one `__m256i` — two
/// `vcvtps2ph` (each 8 lanes → a `__m128i`) joined low/high, so a 32-byte block is produced per store
/// (single non-temporal `vmovntdq`). Bit-identical to [`narrow_f16`] per lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
#[inline]
unsafe fn narrow_f16_pack(v0: __m256, v1: __m256) -> __m256i {
    let h0 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v0); // 8 halves (low block)
    let h1 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v1); // 8 halves (high block)
    _mm256_set_m128i(h1, h0) // [h0 (low 128) | h1 (high 128)] = 16 contiguous f16
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

/// Scalar twin of the bf16/f16 max/min/maxabs reduction: 8 logical lane accumulators (lane `l` folds
/// elements `8k+l`), the fixed [`crate::reduce::hcombine8`] combine, then a scalar tail — bit-identical
/// to the AVX2 path (a test pins it) and the no-AVX2 fallback. Generic over the half kind.
fn reduce_minmax_scalar(kind: Half, op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = [ident(op); 8];
    for c in 0..chunks {
        for (l, a) in acc.iter_mut().enumerate() {
            *a = rfold(*a, kind.widen(x[c * 8 + l]), op);
        }
    }
    let mut s = hcombine8(acc, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, kind.widen(v), op);
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_minmax_bf16_avx(op: i64, x: &[u16]) -> f32 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn reduce_minmax_f16_avx(op: i64, x: &[u16]) -> f32 {
    use crate::reduce::{hcombine8, ident, RED_MAXABS, RED_MIN};
    let n = x.len();
    let chunks = n / 8;
    let mut acc = _mm256_set1_ps(ident(op));
    let absmask = _mm256_set1_ps(-0.0);
    for c in 0..chunks {
        let mut v = widen_f16(x.as_ptr().add(c * 8)); // F16C vcvtph2ps, lossless
        if op == RED_MAXABS {
            v = _mm256_andnot_ps(absmask, v);
        }
        acc = if op == RED_MIN {
            _mm256_min_ps(acc, v)
        } else {
            _mm256_max_ps(acc, v)
        };
    }
    let mut lanes = [ident(op); 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = hcombine8(lanes, op);
    for &v in &x[chunks * 8..] {
        s = rfold(s, f16_to_f32(v), op);
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
pub unsafe extern "C" fn wukong_reduce_bf16(x: *const u16, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return reduce_minmax_bf16_avx(op, x);
    }
    reduce_minmax_scalar(Half::Bf16, op, x)
}

/// `reduce_i widen(x[i])` over `n` IEEE-f16 values for the max-family ops (RED_MAX/RED_MIN/RED_MAXABS),
/// f32 result. The F16C twin of [`wukong_reduce_bf16`] — widens with `vcvtph2ps` (lossless); the
/// folds round nothing, so it is the exact reduction. Interp marshals through this very kernel.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_f16(x: *const u16, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx") {
        return reduce_minmax_f16_avx(op, x);
    }
    reduce_minmax_scalar(Half::F16, op, x)
}

// ---- Public C-ABI entry points (f32-accumulated) ----

/// `sum(widen(x[i]))` over `n` IEEE-f16 values (stored as `u16` bits), accumulated in f32.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_f16(x: *const u16, n: i64) -> f32 {
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
pub unsafe extern "C" fn wukong_sum_bf16(x: *const u16, n: i64) -> f32 {
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
pub unsafe extern "C" fn wukong_dot_f16(x: *const u16, y: *const u16, n: i64) -> f32 {
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
pub unsafe extern "C" fn wukong_dot_bf16(x: *const u16, y: *const u16, n: i64) -> f32 {
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

// ---- Parallel reductions (`@parallel`): deterministic, bit-identical on any thread count ----

/// Cut `[0, n)` into fixed `crate::reduce::RCHUNK` chunks, reduce each with `per_chunk(lo, hi)` (one of
/// the serial kernels over the sub-range — a pure function of its chunk), then fold the partials in
/// **ascending chunk order** from `ident`. rayon's indexed `collect()` preserves chunk order regardless
/// of which thread computed each, so the result is identical on every call no matter the core count:
/// the native `@parallel` path and the interpreter (which marshals this very function) agree
/// bit-for-bit. NOTE this chunked fold reassociates vs the *flat* whole-array serial kernel
/// (`wukong_sum_bf16` et al.), so a `@parallel` reduction is **not** bit-equal to its non-`@parallel`
/// twin — a reassociation exception, exactly like the f32 `wukong_sreduce_f32_parallel`.
#[inline]
fn par_chunk_reduce(
    n: usize,
    ident: f32,
    per_chunk: impl Fn(usize, usize) -> f32 + Sync,
    combine: impl Fn(f32, f32) -> f32,
) -> f32 {
    let nchunks = n.div_ceil(crate::reduce::RCHUNK);
    // Fork on the unified kernel pool, never on rayon's implicit global registry: this is a path that
    // can be a process's FIRST rayon touch (a bf16 model whose first parallel op is a reduction), and
    // `run_on_wuk_pool` is what installs the runtime's 16 MiB worker stacks before anything can build
    // rayon's 2 MiB default (`crate::ensure_global_pool`, lib.rs:324-329). Scheduling-only: the fixed
    // `RCHUNK` decomposition and the ordered fold below are what fix the bits, so which pool runs the
    // map cannot change them. `per_chunk` is borrowed, not moved — `&P` is `Send` because `P: Sync`.
    let per_chunk = &per_chunk;
    let partials: Vec<f32> = crate::run_on_wuk_pool(move || {
        (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let lo = c * crate::reduce::RCHUNK;
                let hi = ((c + 1) * crate::reduce::RCHUNK).min(n);
                per_chunk(lo, hi)
            })
            .collect()
    });
    let mut acc = ident;
    for p in partials {
        acc = combine(acc, p);
    }
    acc
}

/// Multicore bf16 sum — the `@parallel` twin of [`wukong_sum_bf16`]. Each fixed chunk is summed by the
/// serial kernel and the partials added ascending, so the result is deterministic (interp == native).
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_bf16_parallel(x: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let n = n as usize;
    if n.div_ceil(crate::reduce::RCHUNK) < 2 {
        return wukong_sum_bf16(x, n as i64);
    }
    let xa = x as usize;
    par_chunk_reduce(
        n,
        0.0,
        // SAFETY: disjoint read-only chunk; `x` valid for `n` u16 by contract.
        |lo, hi| unsafe { wukong_sum_bf16((xa as *const u16).add(lo), (hi - lo) as i64) },
        |a, b| a + b,
    )
}

/// Multicore IEEE-f16 sum — the `@parallel` twin of [`wukong_sum_f16`].
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_sum_f16_parallel(x: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let n = n as usize;
    if n.div_ceil(crate::reduce::RCHUNK) < 2 {
        return wukong_sum_f16(x, n as i64);
    }
    let xa = x as usize;
    par_chunk_reduce(
        n,
        0.0,
        // SAFETY: disjoint read-only chunk; `x` valid for `n` u16 by contract.
        |lo, hi| unsafe { wukong_sum_f16((xa as *const u16).add(lo), (hi - lo) as i64) },
        |a, b| a + b,
    )
}

/// Multicore bf16 dot — the `@parallel` twin of [`wukong_dot_bf16`].
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_dot_bf16_parallel(x: *const u16, y: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let n = n as usize;
    if n.div_ceil(crate::reduce::RCHUNK) < 2 {
        return wukong_dot_bf16(x, y, n as i64);
    }
    let (xa, ya) = (x as usize, y as usize);
    par_chunk_reduce(
        n,
        0.0,
        // SAFETY: disjoint read-only chunks; `x`/`y` valid for `n` u16 by contract.
        |lo, hi| unsafe {
            wukong_dot_bf16(
                (xa as *const u16).add(lo),
                (ya as *const u16).add(lo),
                (hi - lo) as i64,
            )
        },
        |a, b| a + b,
    )
}

/// Multicore IEEE-f16 dot — the `@parallel` twin of [`wukong_dot_f16`].
///
/// # Safety
/// `x` and `y` must each point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_dot_f16_parallel(x: *const u16, y: *const u16, n: i64) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    let n = n as usize;
    if n.div_ceil(crate::reduce::RCHUNK) < 2 {
        return wukong_dot_f16(x, y, n as i64);
    }
    let (xa, ya) = (x as usize, y as usize);
    par_chunk_reduce(
        n,
        0.0,
        // SAFETY: disjoint read-only chunks; `x`/`y` valid for `n` u16 by contract.
        |lo, hi| unsafe {
            wukong_dot_f16(
                (xa as *const u16).add(lo),
                (ya as *const u16).add(lo),
                (hi - lo) as i64,
            )
        },
        |a, b| a + b,
    )
}

/// Multicore bf16 max-family reduction (`RED_MAX`/`RED_MIN`/`RED_MAXABS`) — the `@parallel` twin of
/// [`wukong_reduce_bf16`]. Partials fold with the op's `fold2`/`ident` (the same `Cmp+Select` the
/// recognizer emits), so max/min/absmax stay deterministic over the fixed chunk decomposition.
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_bf16_parallel(x: *const u16, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let n = n as usize;
    if n.div_ceil(crate::reduce::RCHUNK) < 2 {
        return wukong_reduce_bf16(x, n as i64, op);
    }
    let xa = x as usize;
    par_chunk_reduce(
        n,
        crate::reduce::ident(op),
        // SAFETY: disjoint read-only chunk; `x` valid for `n` u16 by contract.
        |lo, hi| unsafe { wukong_reduce_bf16((xa as *const u16).add(lo), (hi - lo) as i64, op) },
        |a, b| crate::reduce::fold2(a, b, op),
    )
}

/// Multicore IEEE-f16 max-family reduction — the `@parallel` twin of [`wukong_reduce_f16`].
///
/// # Safety
/// `x` must point to `n` readable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_reduce_f16_parallel(x: *const u16, n: i64, op: i64) -> f32 {
    if n <= 0 {
        return crate::reduce::ident(op);
    }
    let n = n as usize;
    if n.div_ceil(crate::reduce::RCHUNK) < 2 {
        return wukong_reduce_f16(x, n as i64, op);
    }
    let xa = x as usize;
    par_chunk_reduce(
        n,
        crate::reduce::ident(op),
        // SAFETY: disjoint read-only chunk; `x` valid for `n` u16 by contract.
        |lo, hi| unsafe { wukong_reduce_f16((xa as *const u16).add(lo), (hi - lo) as i64, op) },
        |a, b| crate::reduce::fold2(a, b, op),
    )
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
/// `wukong_velem_f32`. bf16 in + f32 out streams 8 bytes/elem vs the f32 kernel's 12, so on a
/// memory-bound elementwise it runs ~1.5× faster. Compute is f32 (widen is lossless), so the only
/// rounding is the f32 math — identical in the AVX2 and scalar paths (twin-tested).
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_bf16(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpby_f16_avx(x: &[u16], y: &[u16], out: &mut [f32], a: f32, b: f32) {
    let n = out.len();
    let chunks = n / 8;
    let av = _mm256_set1_ps(a);
    let bv = _mm256_set1_ps(b);
    for c in 0..chunks {
        let xw = widen_f16(x.as_ptr().add(c * 8)); // F16C vcvtph2ps
        let yw = widen_f16(y.as_ptr().add(c * 8));
        let t = _mm256_mul_ps(av, xw);
        let r = _mm256_fmadd_ps(bv, yw, t); // b·yw + a·xw, same op order as the scalar twin
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), r);
    }
    for i in chunks * 8..n {
        let xw = f16_to_f32(x[i]);
        let yw = f16_to_f32(y[i]);
        out[i] = b.mul_add(yw, a * xw);
    }
}

/// `out[i] = a·widen(x[i]) + b·widen(y[i])` over `n` **IEEE-f16** inputs with an **f32 output** — the
/// F16C twin of [`wukong_axpby_bf16`] (widens with `vcvtph2ps`). 8 bytes/elem vs an all-f32 axpby's
/// 12; the only rounding is the f32 math, identical in the AVX2 and scalar paths (twin-tested).
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `f32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_f16(
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
    if is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        return axpby_f16_avx(x, y, out, a, b);
    }
    axpby_bf16_scalar(Half::F16, x, y, out, a, b);
}

// ---- Half-precision **output** axpby: `out[i] = a·widen(x[i]) + b·widen(y[i])` narrowed to bf16/f16.
//      The all-half twin of the bf16-in/f32-out axpby above — bf16/f16 in AND out streams 6 bytes/elem
//      vs an all-f32 axpby's 12, so on a memory-bound elementwise it runs ~2× faster (the narrowing
//      store halves the write traffic that dominates). Compute is f32 (widen is lossless), and the only
//      new rounding is the narrowing store, which goes through the shared `Half::narrow` shim (== the
//      interpreter's), so interp == native bit-for-bit and `-O0` == `-O2`. ----

/// Scalar twin / no-AVX2 fallback: `out[i] = narrow(b·yw + a·xw)`, the same op order (`t = a·xw` then
/// `fma(b, yw, t)`) and narrowing shim as the AVX2 path, so SIMD == scalar bit-for-bit.
fn axpby_narrow_scalar(kind: Half, x: &[u16], y: &[u16], out: &mut [u16], a: f32, b: f32) {
    for i in 0..out.len() {
        let xw = kind.widen(x[i]);
        let yw = kind.widen(y[i]);
        out[i] = kind.narrow(b.mul_add(yw, a * xw));
    }
}

/// Total half traffic of the all-half axpby is `6·n` bytes (read x,y bf16 = 4·n + write out = 2·n).
/// At/above this the narrowing store goes **non-temporal** (`vmovntdq`): once the working set spills
/// L3, streaming the 2·n-byte output skips the write-allocate read-for-ownership that would otherwise
/// double the write traffic and erase the whole point of the narrowing store (~10 MiB ≈ this L3).
const HALFOUT_NT_MIN_BYTES: usize = 10 * 1024 * 1024;

#[inline]
fn use_nt_halfout(n: usize) -> bool {
    n.saturating_mul(6) >= HALFOUT_NT_MIN_BYTES
}

/// Elements ahead to software-prefetch the `x`/`y` reads in the streaming path. The output is
/// non-temporal (not prefetched — we never read it back here); the two half-width input streams are the
/// DRAM-read-bound side, so pulling them in a few lines early hides the miss latency. The last few
/// iterations of every bulk loop deliberately address past the end of `x`/`y` — a prefetch past the
/// buffer end is a hint the hardware drops, so no end guard is needed. That is a statement about the
/// *hardware*; the Rust-level obligation is separate, so the prefetch sites form the address with
/// `wrapping_add`, NOT `add`: `<*const T>::add` requires its result to stay inside (or one past) the
/// same allocated object, which this by construction does not. `wrapping_add` carries no such
/// precondition and lowers to the identical address arithmetic; the pointer is only ever handed to
/// `_mm_prefetch`, which never dereferences it.
const HALFOUT_PF_AHEAD: usize = 256;

/// The streaming-narrow skeleton shared by the half-output kernels (axpby, activations): write
/// `out[i] = narrow(⟨op8 i⟩)` for `i in 0..n`. The bulk runs **32 lanes/iter** — 4 independent 8-lane
/// chains packed into 2 stores — and when `6·n` spills L3 the stores are **non-temporal** (`vmovntdq`,
/// with a scalar alignment prologue + `sfence`), else cacheable; an 8-lane + scalar tail finishes.
/// `$op8(i)` yields the 8 f32 lanes at element `i` (reads its own inputs); `$sc(i)` the scalar `u16`
/// (prologue/tail); `$pf(i)` software-prefetches the inputs ahead; `$pack`/`$narrow8` are the
/// precision's 16-/8-lane narrowing stores. Expanded *inside* the caller's `#[target_feature]` fn so
/// the intrinsics and closures share its feature context (the same idiom as `bias.rs`'s `run!`).
macro_rules! stream_narrow {
    ($outp:expr, $n:expr, $op8:expr, $sc:expr, $pf:expr, $pack:expr, $narrow8:expr) => {{
        let outp = $outp;
        let n = $n;
        let op8 = $op8;
        let sc = $sc;
        let pf = $pf;
        let mut i = 0usize;
        if use_nt_halfout(n) {
            // Scalar prologue until `out` is 32-byte aligned (`vmovntdq` faults otherwise).
            while i < n && (outp.add(i) as usize) & 31 != 0 {
                *outp.add(i) = sc(i);
                i += 1;
            }
            while i + 32 <= n {
                pf(i);
                let p0 = $pack(op8(i), op8(i + 8));
                let p1 = $pack(op8(i + 16), op8(i + 24));
                _mm256_stream_si256(outp.add(i) as *mut __m256i, p0);
                _mm256_stream_si256(outp.add(i + 16) as *mut __m256i, p1);
                i += 32;
            }
            while i + 16 <= n {
                _mm256_stream_si256(outp.add(i) as *mut __m256i, $pack(op8(i), op8(i + 8)));
                i += 16;
            }
            _mm_sfence(); // NT stores are weakly ordered; fence before the buffer is read back.
        } else {
            // L3-resident: cacheable 256-bit stores (no NT — that would evict useful lines).
            while i + 16 <= n {
                _mm256_storeu_si256(outp.add(i) as *mut __m256i, $pack(op8(i), op8(i + 8)));
                i += 16;
            }
        }
        while i + 8 <= n {
            $narrow8(op8(i), outp.add(i));
            i += 8;
        }
        for j in i..n {
            *outp.add(j) = sc(j);
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpby_narrow_bf16_avx(x: &[u16], y: &[u16], out: &mut [u16], a: f32, b: f32) {
    let (av, bv) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    // 8-lane axpby `b·widen(y) + a·widen(x)` (same op order / FMA as the scalar tail).
    stream_narrow!(
        out.as_mut_ptr(),
        out.len(),
        |i: usize| _mm256_fmadd_ps(bv, widen_bf16(yp.add(i)), _mm256_mul_ps(av, widen_bf16(xp.add(i)))),
        |i: usize| crate::f32_to_bf16_bits(b.mul_add(bf16_bits_to_f32(y[i]), a * bf16_bits_to_f32(x[i]))),
        |i: usize| {
            _mm_prefetch(xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8, _MM_HINT_T0);
            _mm_prefetch(yp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8, _MM_HINT_T0);
        },
        narrow_bf16_pack,
        narrow_bf16
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn axpby_narrow_f16_avx(x: &[u16], y: &[u16], out: &mut [u16], a: f32, b: f32) {
    let (av, bv) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    stream_narrow!(
        out.as_mut_ptr(),
        out.len(),
        |i: usize| _mm256_fmadd_ps(bv, widen_f16(yp.add(i)), _mm256_mul_ps(av, widen_f16(xp.add(i)))),
        |i: usize| crate::f32_to_f16_bits(b.mul_add(f16_to_f32(y[i]), a * f16_to_f32(x[i]))),
        |i: usize| {
            _mm_prefetch(xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8, _MM_HINT_T0);
            _mm_prefetch(yp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8, _MM_HINT_T0);
        },
        narrow_f16_pack,
        narrow_f16
    );
}

/// Scalar twin / no-AVX2 fallback for the half-output activation: `out[i] = narrow(act(widen(x[i])))`,
/// the same shared `apply1` activation and narrowing shim as the AVX2 lanes, so SIMD == scalar.
fn vmath_narrow_scalar(kind: Half, x: &[u16], out: &mut [u16], op: i64) {
    for i in 0..out.len() {
        out[i] = kind.narrow(crate::vmath::apply1(op, kind.widen(x[i])));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vmath_narrow_bf16_avx(x: &[u16], out: &mut [u16], op: i64) {
    // The 8-lane activation twin of `apply1` (or scalar-only op → the scalar path for every element).
    let Some(f) = crate::vmath::vmath8_for(op) else {
        return vmath_narrow_scalar(Half::Bf16, x, out, op);
    };
    let xp = x.as_ptr();
    stream_narrow!(
        out.as_mut_ptr(),
        out.len(),
        |i: usize| f(widen_bf16(xp.add(i))),
        |i: usize| crate::f32_to_bf16_bits(crate::vmath::apply1(op, bf16_bits_to_f32(x[i]))),
        |i: usize| _mm_prefetch(xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8, _MM_HINT_T0),
        narrow_bf16_pack,
        narrow_bf16
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn vmath_narrow_f16_avx(x: &[u16], out: &mut [u16], op: i64) {
    let Some(f) = crate::vmath::vmath8_for(op) else {
        return vmath_narrow_scalar(Half::F16, x, out, op);
    };
    let xp = x.as_ptr();
    stream_narrow!(
        out.as_mut_ptr(),
        out.len(),
        |i: usize| f(widen_f16(xp.add(i))),
        |i: usize| crate::f32_to_f16_bits(crate::vmath::apply1(op, f16_to_f32(x[i]))),
        |i: usize| _mm_prefetch(xp.wrapping_add(i + HALFOUT_PF_AHEAD) as *const i8, _MM_HINT_T0),
        narrow_f16_pack,
        narrow_f16
    );
}

/// `out[i] = round_bf16(a·widen(x[i]) + b·widen(y[i]))` over `n` **bf16** inputs with a **bf16 output**
/// — the all-half streaming axpby. 6 bytes/elem vs an all-f32 axpby's 12; the compute is f32 (lossless
/// widen) and the store rounds through the shared `f32_to_bf16_bits` shim, so interp == native.
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_bf16_out(
    x: *const u16,
    y: *const u16,
    out: *mut u16,
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
        return axpby_narrow_bf16_avx(x, y, out, a, b);
    }
    axpby_narrow_scalar(Half::Bf16, x, y, out, a, b);
}

/// `out[i] = round_f16(a·widen(x[i]) + b·widen(y[i]))` over `n` **f16** inputs with an **f16 output** —
/// the F16C twin of [`wukong_axpby_bf16_out`] (widen `vcvtph2ps`, narrow `vcvtps2ph`). 6 bytes/elem.
///
/// # Safety
/// `x`/`y` must point to `n` readable `u16`; `out` to `n` writable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_axpby_f16_out(
    x: *const u16,
    y: *const u16,
    out: *mut u16,
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
    if is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        return axpby_narrow_f16_avx(x, y, out, a, b);
    }
    axpby_narrow_scalar(Half::F16, x, y, out, a, b);
}

/// `out[i] = round_bf16(act(widen(x[i])))` over `n` **bf16** inputs with a **bf16 output** (`op` a
/// `crate::vmath::VM_*` activation code) — a **half-output activation**: the transformer
/// activation-store / KV-cache-write path computes in f32 but stores bf16. 4 bytes/elem (2 in + 2 out)
/// vs the half-in/f32-out activation's 6 and the all-f32's 8, so on a memory-bound activation the
/// narrowing store is the bandwidth win (the >L3 non-temporal path via [`stream_narrow`]). `act` is the
/// shared `vmath` kernel and the store the shared `f32_to_bf16_bits` shim, so it equals a scalar
/// `apply1` + narrow lane-for-lane — interp == native, `-O0` == `-O2`.
///
/// # Safety
/// `x` must point to `n` readable `u16` (bf16 bits); `out` to `n` writable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_vmath_bf16_out(x: *const u16, out: *mut u16, n: i64, op: i64) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    let x = std::slice::from_raw_parts(x, n);
    let out = std::slice::from_raw_parts_mut(out, n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return vmath_narrow_bf16_avx(x, out, op);
    }
    vmath_narrow_scalar(Half::Bf16, x, out, op);
}

/// `out[i] = round_f16(act(widen(x[i])))` over `n` **f16** inputs with an **f16 output** — the F16C twin
/// of [`wukong_vmath_bf16_out`] (widen `vcvtph2ps`, narrow `vcvtps2ph`). 4 bytes/elem.
///
/// # Safety
/// `x` must point to `n` readable `u16` (f16 bits); `out` to `n` writable `u16`.
#[no_mangle]
pub unsafe extern "C" fn wukong_vmath_f16_out(x: *const u16, out: *mut u16, n: i64, op: i64) {
    if n <= 0 {
        return;
    }
    let n = n as usize;
    let x = std::slice::from_raw_parts(x, n);
    let out = std::slice::from_raw_parts_mut(out, n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("f16c")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        return vmath_narrow_f16_avx(x, out, op);
    }
    vmath_narrow_scalar(Half::F16, x, out, op);
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
                    wukong_sum_f16(xf16.as_ptr(), n as i64).to_bits(),
                    sum_scalar(Half::F16, &xf16).to_bits(),
                    "sum_f16 n={n}"
                );
                assert_eq!(
                    wukong_sum_bf16(xbf.as_ptr(), n as i64).to_bits(),
                    sum_scalar(Half::Bf16, &xbf).to_bits(),
                    "sum_bf16 n={n}"
                );
                assert_eq!(
                    wukong_dot_f16(xf16.as_ptr(), yf16.as_ptr(), n as i64).to_bits(),
                    dot_scalar(Half::F16, &xf16, &yf16).to_bits(),
                    "dot_f16 n={n}"
                );
                assert_eq!(
                    wukong_dot_bf16(xbf.as_ptr(), ybf.as_ptr(), n as i64).to_bits(),
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
            let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
            for op in [RED_MAX, RED_MIN, RED_MAXABS] {
                for (kind, xb, got) in [
                    (Half::Bf16, &xbf, unsafe {
                        wukong_reduce_bf16(xbf.as_ptr(), n as i64, op)
                    }),
                    (Half::F16, &xf16, unsafe {
                        wukong_reduce_f16(xf16.as_ptr(), n as i64, op)
                    }),
                ] {
                    // AVX2/F16C == scalar twin, bit-for-bit.
                    let twin = reduce_minmax_scalar(kind, op, xb);
                    assert_eq!(got.to_bits(), twin.to_bits(), "twin op {op} n {n}");
                    // == the exact reduction of the widened values (max/min round nothing).
                    let mut want = ident(op);
                    for &b in xb {
                        let v = kind.widen(b);
                        let v = if op == RED_MAXABS { v.abs() } else { v };
                        want = fold(want, v, if op == RED_MAXABS { RED_MAX } else { op });
                    }
                    assert_eq!(got.to_bits(), want.to_bits(), "ref op {op} n {n}");
                }
            }
        }
    }

    #[test]
    fn axpby_lowp_simd_equals_scalar_twin() {
        for n in [0usize, 1, 7, 8, 9, 100, 1000, 4099] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 2.3).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.9).cos()).collect();
            let (a, b) = (1.5f32, -0.75f32);
            for (kind, conv) in [
                (Half::Bf16, bf16_bits as fn(f32) -> u16),
                (Half::F16, f16_bits as fn(f32) -> u16),
            ] {
                let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                let yb: Vec<u16> = ys.iter().map(|&v| conv(v)).collect();
                let mut got = vec![0f32; n];
                unsafe {
                    match kind {
                        Half::Bf16 => wukong_axpby_bf16(
                            xb.as_ptr(),
                            yb.as_ptr(),
                            got.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        ),
                        Half::F16 => wukong_axpby_f16(
                            xb.as_ptr(),
                            yb.as_ptr(),
                            got.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        ),
                    }
                }
                let mut want = vec![0f32; n];
                axpby_bf16_scalar(kind, &xb, &yb, &mut want, a, b);
                for i in 0..n {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "axpby n={n} i={i}");
                }
            }
        }
    }

    /// The SIMD narrowing store (`narrow_bf16`/`narrow_f16`, 8 lanes at once) must be **bit-identical**
    /// to the scalar `f32_to_{bf16,f16}_bits` shim the interpreter narrows through — across ordinary
    /// values, negatives, subnormals, ±inf, and NaN (the round's every branch). This is the gate that
    /// keeps a half-**output** kernel bit-for-bit consistent with the differential oracle.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn narrow_simd_equals_scalar_shim() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("f16c") {
            return;
        }
        // A spread of tricky f32s: 0/±0, small/large, subnormal, ties, ±inf, NaN, negatives.
        let mut vals: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828,
            1e-40, -1e-40, // subnormal after narrowing
            65504.0, -65504.0, // f16 max
            1e30, -1e30, // overflows f16 → inf
            f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN,
            1.0000001, 1.9999999, 0.999999, // near ties
        ];
        // Add a deterministic sweep that hits many mantissa bits, then pad to a multiple of 8.
        for i in 0..128 {
            vals.push((i as f32 * 0.013 - 0.4).sin() * 123.456);
        }
        while vals.len() % 8 != 0 {
            vals.push(0.0);
        }
        let chunks = vals.len() / 8;
        for c in 0..chunks {
            let mut gb = [0u16; 8];
            let mut gf = [0u16; 8];
            unsafe {
                let v = _mm256_loadu_ps(vals.as_ptr().add(c * 8));
                narrow_bf16(v, gb.as_mut_ptr());
                narrow_f16(v, gf.as_mut_ptr());
            }
            for l in 0..8 {
                let x = vals[c * 8 + l];
                assert_eq!(gb[l], crate::f32_to_bf16_bits(x), "bf16 narrow x={x}");
                assert_eq!(gf[l], crate::f32_to_f16_bits(x), "f16 narrow x={x}");
            }
        }
        // The 16-lane pack helpers (the 256-bit NT-store core) must produce the identical bits as two
        // 8-lane narrows concatenated — same round, just a wider pack.
        while vals.len() % 16 != 0 {
            vals.push(0.0);
        }
        for c in 0..vals.len() / 16 {
            let mut gb = [0u16; 16];
            let mut gf = [0u16; 16];
            unsafe {
                let v0 = _mm256_loadu_ps(vals.as_ptr().add(c * 16));
                let v1 = _mm256_loadu_ps(vals.as_ptr().add(c * 16 + 8));
                _mm256_storeu_si256(gb.as_mut_ptr() as *mut __m256i, narrow_bf16_pack(v0, v1));
                _mm256_storeu_si256(gf.as_mut_ptr() as *mut __m256i, narrow_f16_pack(v0, v1));
            }
            for l in 0..16 {
                let x = vals[c * 16 + l];
                assert_eq!(gb[l], crate::f32_to_bf16_bits(x), "bf16 pack x={x}");
                assert_eq!(gf[l], crate::f32_to_f16_bits(x), "f16 pack x={x}");
            }
        }
    }

    /// The half-**output** axpby (`wukong_axpby_{bf16,f16}_out`) SIMD path must equal its scalar twin
    /// bit-for-bit across tail sizes — the narrowing store rounds through the same shim, so the two
    /// agree lane-for-lane (and it is the no-AVX2 fallback).
    #[test]
    fn axpby_narrow_out_simd_equals_scalar_twin() {
        // The large sizes (> HALFOUT_NT_MIN_BYTES/6 ≈ 1.75M elems) drive the non-temporal 256-bit
        // streaming path — its scalar alignment prologue, 16-lane packs, and tail — not just the small
        // cacheable path. 1_749_000 straddles the NT threshold; the +3 keeps a non-mult-of-16 tail.
        for n in [0usize, 1, 7, 8, 9, 16, 17, 100, 1000, 4099, 1_749_000, 2_000_003] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 2.3).sin()).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 + 0.9).cos()).collect();
            let (a, b) = (1.5f32, -0.75f32);
            for (kind, conv) in [
                (Half::Bf16, bf16_bits as fn(f32) -> u16),
                (Half::F16, f16_bits as fn(f32) -> u16),
            ] {
                let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                let yb: Vec<u16> = ys.iter().map(|&v| conv(v)).collect();
                let mut got = vec![0u16; n];
                unsafe {
                    match kind {
                        Half::Bf16 => wukong_axpby_bf16_out(
                            xb.as_ptr(), yb.as_ptr(), got.as_mut_ptr(), n as i64, a, b,
                        ),
                        Half::F16 => wukong_axpby_f16_out(
                            xb.as_ptr(), yb.as_ptr(), got.as_mut_ptr(), n as i64, a, b,
                        ),
                    }
                }
                let mut want = vec![0u16; n];
                axpby_narrow_scalar(kind, &xb, &yb, &mut want, a, b);
                for i in 0..n {
                    assert_eq!(got[i], want[i], "narrow axpby n={n} i={i}");
                }
            }
        }
    }

    /// The half-**output** activation (`wukong_vmath_{bf16,f16}_out`) SIMD path must equal its scalar
    /// twin bit-for-bit across activations and tail sizes — the vector activation is the twin of
    /// `apply1` and the narrowing store the shared shim, so it agrees lane-for-lane (this is the gate
    /// that keeps the half-output activation consistent with the differential oracle). The large sizes
    /// drive the non-temporal streaming path (alignment prologue + 16-lane packs + tail).
    #[test]
    fn vmath_narrow_out_simd_equals_scalar_twin() {
        use crate::vmath::{VM_ERF, VM_EXP, VM_GELU, VM_RELU, VM_SIGMOID, VM_SILU, VM_TANH};
        for op in [VM_RELU, VM_SILU, VM_GELU, VM_EXP, VM_SIGMOID, VM_TANH, VM_ERF] {
            for n in [0usize, 1, 7, 8, 9, 16, 17, 100, 1000, 4099, 1_749_000] {
                // A spread over sign/magnitude (activations bend near 0 and saturate for large |x|).
                let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011 - 3.7).sin() * 4.0).collect();
                for (kind, conv) in [
                    (Half::Bf16, bf16_bits as fn(f32) -> u16),
                    (Half::F16, f16_bits as fn(f32) -> u16),
                ] {
                    let xb: Vec<u16> = xs.iter().map(|&v| conv(v)).collect();
                    let mut got = vec![0u16; n];
                    unsafe {
                        match kind {
                            Half::Bf16 => {
                                wukong_vmath_bf16_out(xb.as_ptr(), got.as_mut_ptr(), n as i64, op)
                            }
                            Half::F16 => {
                                wukong_vmath_f16_out(xb.as_ptr(), got.as_mut_ptr(), n as i64, op)
                            }
                        }
                    }
                    let mut want = vec![0u16; n];
                    vmath_narrow_scalar(kind, &xb, &mut want, op);
                    for i in 0..n {
                        assert_eq!(got[i], want[i], "narrow vmath op={op} n={n} i={i}");
                    }
                }
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
            wukong_axpby_bf16(xbf.as_ptr(), ybf.as_ptr(), got.as_mut_ptr(), n as i64, a, b);
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
    /// set well past L3 it runs ~2× faster than the f32 reduction. Compares Wukong's SIMD bf16/f16
    /// sum to Wukong's tuned f32 sum AND to a plain (rustc-autovectorized) Rust f32/bf16 sum.
    /// Run: `cargo test -p wukong_runtime --release lowp_bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore = "bandwidth bench; run explicitly in --release"]
    fn lowp_bandwidth() {
        use std::time::Instant;
        assert!(
            !cfg!(debug_assertions),
            "this is a throughput measurement, not a test — rebuild with --release \
             (cargo test -p wukong_runtime --release lowp_bandwidth -- --ignored --nocapture)"
        );
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
                crate::wukong_sreduce_f32(
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
            Box::new(move || unsafe { wukong_sum_bf16(bp as *const u16, n as i64) }),
        );
        let fp = xf16.as_ptr() as usize;
        let t_merc_f16 = best(
            5,
            Box::new(move || unsafe { wukong_sum_f16(fp as *const u16, n as i64) }),
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
    /// faster. Compares `wukong_axpby_bf16` to a plain (rustc-autovectorized) f32 axpby.
    /// Run: `cargo test -p wukong_runtime --release axpby_bf16_bandwidth -- --ignored --nocapture`.
    #[test]
    #[ignore = "bandwidth bench; run explicitly in --release"]
    fn axpby_bf16_bandwidth() {
        use std::time::Instant;
        assert!(
            !cfg!(debug_assertions),
            "this is a throughput measurement, not a test — rebuild with --release \
             (cargo test -p wukong_runtime --release axpby_bf16_bandwidth -- --ignored --nocapture)"
        );
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
                wukong_axpby_bf16(
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
        let got = unsafe { wukong_sum_f16(xf16.as_ptr(), n as i64) };
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
        let got = unsafe { wukong_dot_bf16(xbf.as_ptr(), ybf.as_ptr(), n as i64) };
        let rel = (got as f64 - ref_dot).abs() / ref_dot.abs().max(1.0);
        assert!(rel < 1e-2, "bf16 dot rel {rel:.2e}");
    }

    /// The IDENTICAL chunk fold the `_parallel` kernels perform, computed sequentially (no rayon): cut
    /// `[0, n)` on the fixed `RCHUNK` boundary, reduce each chunk with the serial kernel, fold the
    /// partials ascending from `ident(op)`. The parallel kernel must equal this bit-for-bit — that is
    /// its thread-count-independence contract — and it reassociates vs the flat whole-array serial.
    fn seq_chunked(x: &[u16], y: &[u16], op: i64, is_f16: bool) -> f32 {
        use crate::reduce::{fold2, ident, RCHUNK, RED_DOT, RED_SUM};
        let n = x.len();
        let nchunks = n.div_ceil(RCHUNK).max(1);
        let mut acc = ident(op);
        for c in 0..nchunks {
            let lo = c * RCHUNK;
            let hi = ((c + 1) * RCHUNK).min(n);
            let len = (hi - lo) as i64;
            // SAFETY: [lo, hi) ⊆ [0, n); the slices are valid for `len` u16.
            let p = unsafe {
                match (op, is_f16) {
                    (RED_DOT, false) => wukong_dot_bf16(x[lo..].as_ptr(), y[lo..].as_ptr(), len),
                    (RED_DOT, true) => wukong_dot_f16(x[lo..].as_ptr(), y[lo..].as_ptr(), len),
                    (RED_SUM, false) => wukong_sum_bf16(x[lo..].as_ptr(), len),
                    (RED_SUM, true) => wukong_sum_f16(x[lo..].as_ptr(), len),
                    (_, false) => wukong_reduce_bf16(x[lo..].as_ptr(), len, op),
                    (_, true) => wukong_reduce_f16(x[lo..].as_ptr(), len, op),
                }
            };
            acc = fold2(acc, p, op);
        }
        acc
    }

    #[test]
    fn parallel_reductions_match_sequential_chunked() {
        use crate::reduce::{RED_DOT, RED_MAX, RED_MAXABS, RED_MIN, RED_SUM};
        // Sizes straddling the fixed RCHUNK (8192) boundary: the < 2-chunk guard path, an exact chunk,
        // a one-over partial, and several full chunks + a non-mult-of-8 tail.
        for &n in &[1usize, 8, 8191, 8192, 8193, 3 * 8192 + 13] {
            let xs: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.013 - 0.5).collect();
            let ys: Vec<f32> = (0..n).map(|i| (i % 53) as f32 * 0.017 - 0.3).collect();
            let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
            let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
            let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
            let yf16: Vec<u16> = ys.iter().map(|&v| f16_bits(v)).collect();
            let nn = n as i64;
            unsafe {
                assert_eq!(
                    wukong_sum_bf16_parallel(xbf.as_ptr(), nn).to_bits(),
                    seq_chunked(&xbf, &xbf, RED_SUM, false).to_bits(),
                    "sum_bf16 n={n}"
                );
                assert_eq!(
                    wukong_sum_f16_parallel(xf16.as_ptr(), nn).to_bits(),
                    seq_chunked(&xf16, &xf16, RED_SUM, true).to_bits(),
                    "sum_f16 n={n}"
                );
                assert_eq!(
                    wukong_dot_bf16_parallel(xbf.as_ptr(), ybf.as_ptr(), nn).to_bits(),
                    seq_chunked(&xbf, &ybf, RED_DOT, false).to_bits(),
                    "dot_bf16 n={n}"
                );
                assert_eq!(
                    wukong_dot_f16_parallel(xf16.as_ptr(), yf16.as_ptr(), nn).to_bits(),
                    seq_chunked(&xf16, &yf16, RED_DOT, true).to_bits(),
                    "dot_f16 n={n}"
                );
                for &op in &[RED_MAX, RED_MIN, RED_MAXABS] {
                    let gb = wukong_reduce_bf16_parallel(xbf.as_ptr(), nn, op);
                    assert_eq!(
                        gb.to_bits(),
                        seq_chunked(&xbf, &xbf, op, false).to_bits(),
                        "reduce_bf16 op={op} n={n}"
                    );
                    // Determinism: a second call returns the identical bits.
                    assert_eq!(
                        gb.to_bits(),
                        wukong_reduce_bf16_parallel(xbf.as_ptr(), nn, op).to_bits(),
                        "reduce_bf16 nondeterministic op={op} n={n}"
                    );
                    let gf = wukong_reduce_f16_parallel(xf16.as_ptr(), nn, op);
                    assert_eq!(
                        gf.to_bits(),
                        seq_chunked(&xf16, &xf16, op, true).to_bits(),
                        "reduce_f16 op={op} n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_reductions_fork_on_the_unified_pool() {
        // `par_chunk_reduce` can be a process's FIRST rayon touch (a bf16 model whose first parallel
        // op is a reduction). Forking `into_par_iter()` straight from there builds rayon's DEFAULT
        // global registry — 2 MiB worker stacks — and the runtime's later `ensure_global_pool()` then
        // loses the race silently (`build_global` → GlobalPoolAlreadyInitialized, discarded at
        // lib.rs), so every subsequent outlined `@parallel` region body runs on a stack an eighth of
        // the required 16 MiB. That first-touch ordering is process-global, so a shared-process test
        // harness cannot pin it directly; what it CAN pin, order-independently, is the cause: the
        // chunks must execute on the unified kernel pool, i.e. this fork must go through
        // `run_on_wuk_pool`, which is what configures the global pool before anything else can.
        //
        // On a part with no HyperThreads to shed, `gemm_pool()` is None and the two pools coincide,
        // so the comparison is vacuous there (and under `WUKONG_POOL_UNIFY=0`, and at
        // `RAYON_NUM_THREADS=1`) — it still cannot regress. On this 6P+8E+2LPE machine the widths are
        // 16 vs 22 and it is a real discriminator.
        let want = crate::wuk_pool_width();
        let n = 4 * crate::reduce::RCHUNK + 7;
        let seen = std::sync::Mutex::new(std::collections::BTreeSet::new());
        let got = par_chunk_reduce(
            n,
            0.0,
            |lo, hi| {
                seen.lock().unwrap().insert(rayon::current_num_threads());
                (hi - lo) as f32
            },
            |a, b| a + b,
        );
        // Every chunk ran exactly once (so the widths below are the whole story, not a sample).
        assert_eq!(got, n as f32, "chunk coverage");
        assert_eq!(
            seen.into_inner().unwrap(),
            std::collections::BTreeSet::from([want]),
            "par_chunk_reduce forked off the unified kernel pool (width {want})"
        );
    }

    #[test]
    fn parallel_reductions_within_f64_tolerance() {
        use crate::reduce::{RED_MAX, RED_MAXABS, RED_MIN};
        // A size that exercises several real chunks (RCHUNK = 8192), so rayon actually runs.
        let n = 3 * 8192 + 251;
        let xs: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.01).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i % 53) as f32) * 0.02).collect();
        let xbf: Vec<u16> = xs.iter().map(|&v| bf16_bits(v)).collect();
        let ybf: Vec<u16> = ys.iter().map(|&v| bf16_bits(v)).collect();
        let xf16: Vec<u16> = xs.iter().map(|&v| f16_bits(v)).collect();
        unsafe {
            // bf16 / f16 sum: f32-accumulation error within ULP tolerance of the f64 reference.
            let want: f64 = xbf.iter().map(|&b| bf16_bits_to_f32(b) as f64).sum();
            let got = wukong_sum_bf16_parallel(xbf.as_ptr(), n as i64) as f64;
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 1e-2,
                "bf16 sum got {got} want {want}"
            );
            let want: f64 = xf16.iter().map(|&b| f16_to_f32(b) as f64).sum();
            let got = wukong_sum_f16_parallel(xf16.as_ptr(), n as i64) as f64;
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 1e-3,
                "f16 sum got {got} want {want}"
            );
            // bf16 dot.
            let want: f64 = xbf
                .iter()
                .zip(&ybf)
                .map(|(&a, &b)| bf16_bits_to_f32(a) as f64 * bf16_bits_to_f32(b) as f64)
                .sum();
            let got = wukong_dot_bf16_parallel(xbf.as_ptr(), ybf.as_ptr(), n as i64) as f64;
            assert!(
                (got - want).abs() / want.abs().max(1.0) < 1e-2,
                "bf16 dot got {got} want {want}"
            );
            // max / min / absmax round nothing and the widen is lossless, so the parallel result is
            // the *exact* reduction of the widened values — assert bit-for-bit, no tolerance.
            let widen = |b: &[u16]| -> Vec<f32> { b.iter().map(|&v| bf16_bits_to_f32(v)).collect() };
            let xw = widen(&xbf);
            let want_max = xw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let want_min = xw.iter().copied().fold(f32::INFINITY, f32::min);
            let want_amx = xw
                .iter()
                .map(|v| v.abs())
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                wukong_reduce_bf16_parallel(xbf.as_ptr(), n as i64, RED_MAX).to_bits(),
                want_max.to_bits(),
                "bf16 max"
            );
            assert_eq!(
                wukong_reduce_bf16_parallel(xbf.as_ptr(), n as i64, RED_MIN).to_bits(),
                want_min.to_bits(),
                "bf16 min"
            );
            assert_eq!(
                wukong_reduce_bf16_parallel(xbf.as_ptr(), n as i64, RED_MAXABS).to_bits(),
                want_amx.to_bits(),
                "bf16 absmax"
            );
        }
    }
}
