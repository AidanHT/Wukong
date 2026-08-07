//! int8 quantized GEMM — **`C = A·Bᵀ`**, the quantized `nn.Linear`: `u8` activations × `i8` weights
//! → `i32` accumulator. This is the layout real quantized inference uses (QNNPACK / XNNPACK / oneDNN):
//! unsigned activations, signed weights, a wide integer accumulator. Because A's row `i` and B's row
//! `j` are both contiguous over the contraction `K`, each `C[i,j]` is a plain dot product over `K` —
//! the int8 analog of the f32 reduction — so no VNNI repacking is needed; the inner loop widens `u8`/
//! `i8` to `i16` and uses `vpmaddwd` (`_mm256_madd_epi16`) to fold 16 lanes at a time into `i32`.
//! On CPUs with **AVX-VNNI** the inner op is a single `vpdpbusd` (`_mm256_dpbusd_avx_epi32`), which
//! folds 32 `u8×i8` products into the 8 `i32` lanes *and* accumulates in one instruction — the path
//! gcc's `-march=native` takes, and what it takes to match/beat it. Both paths are **register-blocked
//! four B-rows at a time** (`dot4_i8_{vnni,avx2}`): the A-row chunk is loaded once per step and reused
//! across the four dots, with four independent accumulator chains for ILP and an in-register
//! horizontal sum (no per-`(i,j)` stack round-trip). The single-threaded VNNI path tiles **two A-rows
//! at a time as well** (`dot2x4_i8_vnni`, a 2×4 register tile): each B-row chunk loaded once feeds
//! both rows, halving B-matrix traffic. CPU-feature detection happens **once per call**,
//! not per element (VNNI → widen+`vpmaddwd` AVX2 → scalar), so the hot loop is branch-free
//! `target_feature` code. `vpdpbusd` is the non-saturating form, so it sums the products into `i32`
//! exactly — bit-identical to the scalar fold, same as the `vpmaddwd` path.
//!
//! **Why int8 here is the cleanest correctness story in the codebase.** Integer `i32` addition is
//! associative and commutative *modulo 2³²* — wrapping arithmetic, the Rust/​C release default — so a
//! sum of products is bit-identical regardless of the order it is accumulated. The AVX2 path folds
//! into lane accumulators and combines them; the scalar twin folds left-to-right; both land on the
//! same `i32`. There is therefore **no "reassociated form is the oracle" exception** as there is for
//! float reductions: the fused kernel equals the naive `s += a[k]*b[k]` loop bit-for-bit, full stop.
//! Each product `u8·i8 ∈ [-32640, 32385]` fits `i16`, so `vpmaddwd`'s `i16×i16→i32` is exact; the
//! `i32` accumulation wraps (it can overflow for very large `K`, exactly as the source loop would).
//!
//! The compiler recognizes a `u8×i8→i32` `C = A·Bᵀ` nest and lowers it to one [`wukong_i8gemm_nt`]
//! call (the int8 sibling of `wukong_sgemm_nt`); the interpreter marshals its abstract memory through
//! the **identical** kernel, so the differential oracle stays bit-for-bit exact. Rows of `C` are
//! independent, so the `_parallel` entry maps the per-row routine across cores: `serial == parallel`
//! with no cross-row combine, independent of thread count.

use rayon::prelude::*;

/// Scalar reference: `sum_t (a[t] as i32) * (b[t] as i32)` over `k` elements, wrapping `i32`. Backs
/// the AVX2 tail and the no-AVX2 fallback, and is the differential reference for the unit tests.
///
/// # Safety
/// `a` and `b` must be valid for `k` elements (`u8` and `i8` respectively).
#[inline]
unsafe fn dot_i8_scalar(a: *const u8, b: *const i8, k: usize) -> i32 {
    let mut s: i32 = 0;
    for t in 0..k {
        s = s.wrapping_add((*a.add(t) as i32).wrapping_mul(*b.add(t) as i32));
    }
    s
}

/// AVX2 `u8×i8→i32` dot: widen 16 `u8`/`i8` at a time to `i16` (`vpmovzxbw`/`vpmovsxbw`) and fold with
/// `vpmaddwd` into `i32` lane accumulators, 32 elements per step across two chains for ILP. Bit-equal
/// to [`dot_i8_scalar`] for *every* input — `i16×i16→i32` is exact (products fit `i16`) and the lane
/// combine is order-immaterial under wrapping `i32` (associative mod 2³²), so no fixed-order combine
/// is required (unlike the float kernels).
///
/// # Safety
/// `a`/`b` valid for `k` elements; requires the `avx2` feature (callers gate on it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: *const u8, b: *const i8, k: usize) -> i32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0usize;
    // 32 int8 per step: two independent widen+madd chains keep both vector ALUs busy.
    while i + 32 <= k {
        let a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.add(i) as *const __m128i));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
        let a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.add(i + 16) as *const __m128i));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    // a single 16-wide chunk if it remains
    while i + 16 <= k {
        let av = _mm256_cvtepu8_epi16(_mm_loadu_si128(a.add(i) as *const __m128i));
        let bv = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.add(i) as *const __m128i));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(av, bv));
        i += 16;
    }
    // Horizontal sum of the 8 i32 lanes (any order — wrapping add is associative mod 2³²).
    let acc = _mm256_add_epi32(acc0, acc1);
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut s: i32 = 0;
    for v in lanes {
        s = s.wrapping_add(v);
    }
    // scalar tail (< 16 elements)
    s.wrapping_add(dot_i8_scalar(a.add(i), b.add(i), k - i))
}

/// Horizontal wrapping sum of the 8 `i32` lanes of `v`, staying in registers (extract + shuffle +
/// add, no memory round-trip). Any summation order is exact under wrapping `i32` (associative mod
/// 2³²), so this equals the scalar left-fold bit-for-bit.
///
/// # Safety
/// Requires the `avx2` feature (callers gate on it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b1110_1110>(s)); // + upper 2 lanes
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b0000_0001>(s)); // + lane 1
    _mm_cvtsi128_si32(s)
}

/// Four `u8×i8→i32` dots at once: one A-row against four B-rows, AVX2. The A-row chunk is widened
/// **once per step** and reused across all four B-rows (4× less A-side load+widen), and the four
/// independent accumulator chains expose the ILP that hides `vpmaddwd` latency. Bit-equal to four
/// [`dot_i8_scalar`] calls (exact `i16` products folded by `vpmaddwd`; wrapping `i32` combine is
/// order-immaterial).
///
/// # Safety
/// `arow`/`b0..b3` valid for `k` elements; requires `avx2` (callers gate on it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot4_i8_avx2(
    arow: *const u8,
    b0: *const i8,
    b1: *const i8,
    b2: *const i8,
    b3: *const i8,
    k: usize,
) -> (i32, i32, i32, i32) {
    use std::arch::x86_64::*;
    let (mut c0, mut c1, mut c2, mut c3) = (
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    );
    let mut i = 0usize;
    while i + 16 <= k {
        let av = _mm256_cvtepu8_epi16(_mm_loadu_si128(arow.add(i) as *const __m128i));
        let w = |p: *const i8| _mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(i) as *const __m128i));
        c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(av, w(b0)));
        c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(av, w(b1)));
        c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(av, w(b2)));
        c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(av, w(b3)));
        i += 16;
    }
    let (mut s0, mut s1, mut s2, mut s3) = (
        hsum_i32_avx2(c0),
        hsum_i32_avx2(c1),
        hsum_i32_avx2(c2),
        hsum_i32_avx2(c3),
    );
    if i < k {
        let t = k - i;
        s0 = s0.wrapping_add(dot_i8_scalar(arow.add(i), b0.add(i), t));
        s1 = s1.wrapping_add(dot_i8_scalar(arow.add(i), b1.add(i), t));
        s2 = s2.wrapping_add(dot_i8_scalar(arow.add(i), b2.add(i), t));
        s3 = s3.wrapping_add(dot_i8_scalar(arow.add(i), b3.add(i), t));
    }
    (s0, s1, s2, s3)
}

/// One row of `C = A·Bᵀ` on the AVX2 path: `arow · B[j]ᵀ` for every `j`, blocked four columns at a
/// time via [`dot4_i8_avx2`] with a `dot_i8_avx2` tail. Factored out so the serial and `_parallel`
/// kernels share the exact same per-row code (hence serial == parallel bit-for-bit).
///
/// # Safety
/// `arow` valid for `k` `u8`; `b` for `n*k` `i8`; `crow` for `n` `i32`; requires `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gemm_row_nt_avx2(arow: *const u8, b: *const i8, crow: *mut i32, k: usize, n: usize) {
    let mut j = 0usize;
    while j + 4 <= n {
        let (s0, s1, s2, s3) = dot4_i8_avx2(
            arow,
            b.add(j * k),
            b.add((j + 1) * k),
            b.add((j + 2) * k),
            b.add((j + 3) * k),
            k,
        );
        *crow.add(j) = s0;
        *crow.add(j + 1) = s1;
        *crow.add(j + 2) = s2;
        *crow.add(j + 3) = s3;
        j += 4;
    }
    while j < n {
        *crow.add(j) = dot_i8_avx2(arow, b.add(j * k), k);
        j += 1;
    }
}

/// Scalar `C = A·Bᵀ` nest — the no-AVX2 fallback (and the non-x86 path). Identical math to the AVX2
/// kernel.
///
/// # Safety
/// `a` valid for `m*k` `u8`, `b` for `n*k` `i8`, `c` for `m*n` `i32`.
unsafe fn gemm_nt_scalar(a: *const u8, b: *const i8, c: *mut i32, m: usize, k: usize, n: usize) {
    for i in 0..m {
        let arow = a.add(i * k);
        for j in 0..n {
            *c.add(i * n + j) = dot_i8_scalar(arow, b.add(j * k), k);
        }
    }
}

/// Single `u8×i8→i32` dot via **AVX-VNNI** `vpdpbusd` (`_mm256_dpbusd_avx_epi32`): one instruction
/// folds 32 `u8×i8` products into the 8 `i32` lanes *and* accumulates (no separate `vpaddd`), so it
/// is ~3–4× denser than the widen+`vpmaddwd`+add path. `vpdpbusd` (the non-saturating form) sums the
/// products into `i32` exactly, so this is the same sum mod 2³² as [`dot_i8_scalar`] — bit-for-bit.
///
/// # Safety
/// `a`/`b` valid for `k` elements; requires `avx2`+`avxvnni` (callers gate on it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dot_i8_vnni(a: *const u8, b: *const i8, k: usize) -> i32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= k {
        let av = _mm256_loadu_si256(a.add(i) as *const __m256i);
        let bv = _mm256_loadu_si256(b.add(i) as *const __m256i);
        acc = _mm256_dpbusd_avx_epi32(acc, av, bv);
        i += 32;
    }
    let mut s = hsum_i32_avx2(acc);
    if i < k {
        s = s.wrapping_add(dot_i8_scalar(a.add(i), b.add(i), k - i));
    }
    s
}

/// Four `u8×i8→i32` dots at once via AVX-VNNI — the `vpdpbusd` twin of [`dot4_i8_avx2`]: the 32-`u8`
/// A-row chunk is loaded once per step and reused across the four B-rows (4× less A traffic), with
/// four independent `vpdpbusd` accumulator chains for ILP. Bit-equal to four [`dot_i8_scalar`] calls.
///
/// # Safety
/// `arow`/`b0..b3` valid for `k` elements; requires `avx2`+`avxvnni`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn dot4_i8_vnni(
    arow: *const u8,
    b0: *const i8,
    b1: *const i8,
    b2: *const i8,
    b3: *const i8,
    k: usize,
) -> (i32, i32, i32, i32) {
    use std::arch::x86_64::*;
    let (mut c0, mut c1, mut c2, mut c3) = (
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    );
    let mut i = 0usize;
    while i + 32 <= k {
        let av = _mm256_loadu_si256(arow.add(i) as *const __m256i);
        let ld = |p: *const i8| _mm256_loadu_si256(p.add(i) as *const __m256i);
        c0 = _mm256_dpbusd_avx_epi32(c0, av, ld(b0));
        c1 = _mm256_dpbusd_avx_epi32(c1, av, ld(b1));
        c2 = _mm256_dpbusd_avx_epi32(c2, av, ld(b2));
        c3 = _mm256_dpbusd_avx_epi32(c3, av, ld(b3));
        i += 32;
    }
    let (mut s0, mut s1, mut s2, mut s3) = (
        hsum_i32_avx2(c0),
        hsum_i32_avx2(c1),
        hsum_i32_avx2(c2),
        hsum_i32_avx2(c3),
    );
    if i < k {
        let t = k - i;
        s0 = s0.wrapping_add(dot_i8_scalar(arow.add(i), b0.add(i), t));
        s1 = s1.wrapping_add(dot_i8_scalar(arow.add(i), b1.add(i), t));
        s2 = s2.wrapping_add(dot_i8_scalar(arow.add(i), b2.add(i), t));
        s3 = s3.wrapping_add(dot_i8_scalar(arow.add(i), b3.add(i), t));
    }
    (s0, s1, s2, s3)
}

/// One row of `C = A·Bᵀ` on the AVX-VNNI path — the `vpdpbusd` twin of [`gemm_row_nt_avx2`].
///
/// # Safety
/// `arow` valid for `k` `u8`; `b` for `n*k` `i8`; `crow` for `n` `i32`; requires `avx2`+`avxvnni`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn gemm_row_nt_vnni(arow: *const u8, b: *const i8, crow: *mut i32, k: usize, n: usize) {
    let mut j = 0usize;
    while j + 4 <= n {
        let (s0, s1, s2, s3) = dot4_i8_vnni(
            arow,
            b.add(j * k),
            b.add((j + 1) * k),
            b.add((j + 2) * k),
            b.add((j + 3) * k),
            k,
        );
        *crow.add(j) = s0;
        *crow.add(j + 1) = s1;
        *crow.add(j + 2) = s2;
        *crow.add(j + 3) = s3;
        j += 4;
    }
    while j < n {
        *crow.add(j) = dot_i8_vnni(arow, b.add(j * k), k);
        j += 1;
    }
}

/// Eight `u8×i8→i32` dots — a **2×4 register tile** (two A-rows × four B-rows) via AVX-VNNI. Each
/// 32-`i8` B-row chunk is loaded once and reused across *both* A-rows (halving B traffic vs the 1×4
/// `dot4_i8_vnni`), and eight independent `vpdpbusd` accumulator chains saturate the unit and fully
/// hide its latency. Returns `[row0 cols0..4, row1 cols0..4]`. Bit-equal to eight [`dot_i8_scalar`]
/// calls (same products, wrapping-`i32` order-immaterial).
///
/// # Safety
/// `a0`/`a1` valid for `k` `u8`; `b0..b3` for `k` `i8`; requires `avx2`+`avxvnni`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[allow(clippy::too_many_arguments)]
unsafe fn dot2x4_i8_vnni(
    a0: *const u8,
    a1: *const u8,
    b0: *const i8,
    b1: *const i8,
    b2: *const i8,
    b3: *const i8,
    k: usize,
) -> [i32; 8] {
    use std::arch::x86_64::*;
    let z = _mm256_setzero_si256();
    let (mut c00, mut c01, mut c02, mut c03) = (z, z, z, z);
    let (mut c10, mut c11, mut c12, mut c13) = (z, z, z, z);
    let mut i = 0usize;
    while i + 32 <= k {
        let av0 = _mm256_loadu_si256(a0.add(i) as *const __m256i);
        let av1 = _mm256_loadu_si256(a1.add(i) as *const __m256i);
        // Load each B-row chunk once, feed it to both A-rows before moving on (1 live B reg).
        let bv = _mm256_loadu_si256(b0.add(i) as *const __m256i);
        c00 = _mm256_dpbusd_avx_epi32(c00, av0, bv);
        c10 = _mm256_dpbusd_avx_epi32(c10, av1, bv);
        let bv = _mm256_loadu_si256(b1.add(i) as *const __m256i);
        c01 = _mm256_dpbusd_avx_epi32(c01, av0, bv);
        c11 = _mm256_dpbusd_avx_epi32(c11, av1, bv);
        let bv = _mm256_loadu_si256(b2.add(i) as *const __m256i);
        c02 = _mm256_dpbusd_avx_epi32(c02, av0, bv);
        c12 = _mm256_dpbusd_avx_epi32(c12, av1, bv);
        let bv = _mm256_loadu_si256(b3.add(i) as *const __m256i);
        c03 = _mm256_dpbusd_avx_epi32(c03, av0, bv);
        c13 = _mm256_dpbusd_avx_epi32(c13, av1, bv);
        i += 32;
    }
    let mut s = [
        hsum_i32_avx2(c00),
        hsum_i32_avx2(c01),
        hsum_i32_avx2(c02),
        hsum_i32_avx2(c03),
        hsum_i32_avx2(c10),
        hsum_i32_avx2(c11),
        hsum_i32_avx2(c12),
        hsum_i32_avx2(c13),
    ];
    if i < k {
        let t = k - i;
        let bb = [b0, b1, b2, b3];
        for col in 0..4 {
            s[col] = s[col].wrapping_add(dot_i8_scalar(a0.add(i), bb[col].add(i), t));
            s[col + 4] = s[col + 4].wrapping_add(dot_i8_scalar(a1.add(i), bb[col].add(i), t));
        }
    }
    s
}

/// Two rows of `C = A·Bᵀ` on the AVX-VNNI path, 2×4-tiled via [`dot2x4_i8_vnni`] with a per-row
/// `dot_i8_vnni` column tail. Halves B-matrix traffic vs running two `gemm_row_nt_vnni` passes.
///
/// # Safety
/// `a0r`/`a1r` valid for `k` `u8`; `b` for `n*k` `i8`; `c0r`/`c1r` for `n` `i32`; requires
/// `avx2`+`avxvnni`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn gemm_2rows_nt_vnni(
    a0r: *const u8,
    a1r: *const u8,
    b: *const i8,
    c0r: *mut i32,
    c1r: *mut i32,
    k: usize,
    n: usize,
) {
    let mut j = 0usize;
    while j + 4 <= n {
        let s = dot2x4_i8_vnni(
            a0r,
            a1r,
            b.add(j * k),
            b.add((j + 1) * k),
            b.add((j + 2) * k),
            b.add((j + 3) * k),
            k,
        );
        *c0r.add(j) = s[0];
        *c0r.add(j + 1) = s[1];
        *c0r.add(j + 2) = s[2];
        *c0r.add(j + 3) = s[3];
        *c1r.add(j) = s[4];
        *c1r.add(j + 1) = s[5];
        *c1r.add(j + 2) = s[6];
        *c1r.add(j + 3) = s[7];
        j += 4;
    }
    while j < n {
        let bj = b.add(j * k);
        *c0r.add(j) = dot_i8_vnni(a0r, bj, k);
        *c1r.add(j) = dot_i8_vnni(a1r, bj, k);
        j += 1;
    }
}

/// Quantized `nn.Linear` `C = A·Bᵀ` (serial): `A` is `[m,k]` `u8` row-major, `B` is `[n,k]` `i8`
/// row-major (so `B`'s rows are the weight vectors), `C` is `[m,n]` `i32` row-major. Detects the SIMD
/// tier **once** per call (not per element) — AVX-VNNI (2×4 register tile) → widen+`vpmaddwd` AVX2
/// (1×4) → scalar nest — and all three are bit-identical (wrapping `i32` is order-immaterial).
///
/// # Safety
/// `a` valid for `m*k` `u8`, `b` for `n*k` `i8`, `c` for `m*n` `i32`.
#[no_mangle]
pub unsafe extern "C" fn wukong_i8gemm_nt(
    a: *const u8,
    b: *const i8,
    c: *mut i32,
    m: i64,
    k: i64,
    n: i64,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    #[cfg(target_arch = "x86_64")]
    {
        // VNNI (vpdpbusd) is the densest path; widen+madd AVX2 is the fallback; scalar otherwise.
        // VNNI runs a 2×4 register tile (two C-rows at a time) so each B-row load feeds both rows.
        if is_x86_feature_detected!("avxvnni") {
            let mut i = 0usize;
            while i + 2 <= m {
                // SAFETY: rows i, i+1 of A/C valid; b valid for n*k.
                gemm_2rows_nt_vnni(
                    a.add(i * k),
                    a.add((i + 1) * k),
                    b,
                    c.add(i * n),
                    c.add((i + 1) * n),
                    k,
                    n,
                );
                i += 2;
            }
            if i < m {
                gemm_row_nt_vnni(a.add(i * k), b, c.add(i * n), k, n);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") {
            for i in 0..m {
                gemm_row_nt_avx2(a.add(i * k), b, c.add(i * n), k, n);
            }
            return;
        }
    }
    gemm_nt_scalar(a, b, c, m, k, n);
}

/// Multicore `C = A·Bᵀ` — **bit-identical** to [`wukong_i8gemm_nt`]. Each `C[i,j]` is an independent
/// dot, so rows are mapped across cores with no cross-row combine; the result is independent of thread
/// count and the interpreter (serial) agrees with the native `@parallel` path exactly.
///
/// # Safety
/// Same as [`wukong_i8gemm_nt`].
#[no_mangle]
pub unsafe extern "C" fn wukong_i8gemm_nt_parallel(
    a: *const u8,
    b: *const i8,
    c: *mut i32,
    m: i64,
    k: i64,
    n: i64,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel GEMM/reduce).
    let (au, bu, cu) = (a as usize, b as usize, c as usize);
    // Detect the SIMD tier once, not per row — each row then runs the identical per-row kernel the
    // serial path uses, so serial == parallel bit-for-bit regardless of thread count.
    #[cfg(target_arch = "x86_64")]
    let vnni = is_x86_feature_detected!("avxvnni");
    #[cfg(target_arch = "x86_64")]
    let avx2 = is_x86_feature_detected!("avx2");
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
    (0..m).into_par_iter().for_each(|i| {
        let (a, b, c) = (au as *const u8, bu as *const i8, cu as *mut i32);
        // SAFETY: disjoint output row i; pointers valid for the declared extents by contract.
        unsafe {
            let arow = a.add(i * k);
            let crow = c.add(i * n);
            #[cfg(target_arch = "x86_64")]
            {
                if vnni {
                    gemm_row_nt_vnni(arow, b, crow, k, n);
                    return;
                }
                if avx2 {
                    gemm_row_nt_avx2(arow, b, crow, k, n);
                    return;
                }
            }
            for j in 0..n {
                *crow.add(j) = dot_i8_scalar(arow, b.add(j * k), k);
            }
        }
    });
}

/// Dequantize one i32 C row in place to `dst`: `dst[j] = act((src[j] as f32)·scale_a·scale_b[j]
/// (+ bias[j]))`. The multiply order (`·scale_a·scale_b[j]`, left-associated) and the activation match
/// the recognizer's canonical source form exactly, so the interpreter (which marshals this kernel) and
/// the native backend agree bit-for-bit. `act`: 0 id / 1 ReLU / 2 GELU / 3 SiLU (GELU/SiLU reuse the
/// `vmath` scalar twins, so fused == the unfused activation).
#[inline]
unsafe fn dequant_row(
    src: *const i32,
    dst: *mut f32,
    n: usize,
    scale_a: f32,
    scale_b: *const f32,
    bias: *const f32,
    act: i64,
) {
    for j in 0..n {
        let mut v = (*src.add(j) as f32) * scale_a * *scale_b.add(j);
        if !bias.is_null() {
            v += *bias.add(j);
        }
        v = match act {
            1 => {
                if v > 0.0 {
                    v
                } else {
                    0.0
                }
            }
            2 => crate::vmath::gelu1(v),
            3 => crate::vmath::silu1(v),
            _ => v,
        };
        *dst.add(j) = v;
    }
}

/// Quantized `nn.Linear` with a **fused per-channel dequant epilogue**:
/// `out_f32 = act(((A·Bᵀ) as f32)·scale_a·scale_b[j] (+ bias[j]))`. The i32 accumulator never round-
/// trips through main memory: each C row is computed into a small L1-resident i32 scratch by the *same*
/// proven `vpdpbusd` per-row helpers `wukong_i8gemm_nt` uses (so the integer GEMM stays bit-exact),
/// then immediately dequantized to f32. gcc/cuBLAS **cannot** fuse this — their int8 GEMM emits i32 and
/// the dequant is a separate kernel — so it is the decode-regime win, where the dequant pass is a
/// non-trivial fraction of a thin-M GEMM. The 2-row VNNI tile is preserved (B-row loads feed both rows).
///
/// `scale_a` = per-tensor activation scale; `scale_b` = per-output-channel weight scale (length `n`);
/// `bias` = per-output-channel (length `n`) or null; `act` = 0/1/2/3 (id/ReLU/GELU/SiLU).
///
/// # Safety
/// `a` valid for `m*k` u8, `b` for `n*k` i8, `out` for `m*n` f32, `scale_b` for `n` f32, `bias` null or
/// `n` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_i8gemm_nt_deq(
    a: *const u8,
    b: *const i8,
    out: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    scale_a: f32,
    scale_b: *const f32,
    bias: *const f32,
    act: i64,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let mut rows = vec![0i32; 2 * n]; // L1-resident scratch for up to 2 C rows
    let r0 = rows.as_mut_ptr();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avxvnni") {
            let mut i = 0usize;
            while i + 2 <= m {
                gemm_2rows_nt_vnni(a.add(i * k), a.add((i + 1) * k), b, r0, r0.add(n), k, n);
                dequant_row(r0, out.add(i * n), n, scale_a, scale_b, bias, act);
                dequant_row(
                    r0.add(n),
                    out.add((i + 1) * n),
                    n,
                    scale_a,
                    scale_b,
                    bias,
                    act,
                );
                i += 2;
            }
            if i < m {
                gemm_row_nt_vnni(a.add(i * k), b, r0, k, n);
                dequant_row(r0, out.add(i * n), n, scale_a, scale_b, bias, act);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") {
            for i in 0..m {
                gemm_row_nt_avx2(a.add(i * k), b, r0, k, n);
                dequant_row(r0, out.add(i * n), n, scale_a, scale_b, bias, act);
            }
            return;
        }
    }
    for i in 0..m {
        let arow = a.add(i * k);
        for j in 0..n {
            *r0.add(j) = dot_i8_scalar(arow, b.add(j * k), k);
        }
        dequant_row(r0, out.add(i * n), n, scale_a, scale_b, bias, act);
    }
}

/// Multicore fused-dequant `nn.Linear` — **bit-identical** to [`wukong_i8gemm_nt_deq`] (rows are
/// independent; each maps to a core with its own i32 scratch row, dequantized by the identical
/// `dequant_row`). The interpreter calls the serial form, so serial == parallel == interp.
///
/// # Safety
/// Same as [`wukong_i8gemm_nt_deq`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn wukong_i8gemm_nt_deq_parallel(
    a: *const u8,
    b: *const i8,
    out: *mut f32,
    m: i64,
    k: i64,
    n: i64,
    scale_a: f32,
    scale_b: *const f32,
    bias: *const f32,
    act: i64,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let (au, bu, ou, sbu, biu) = (
        a as usize,
        b as usize,
        out as usize,
        scale_b as usize,
        bias as usize,
    );
    #[cfg(target_arch = "x86_64")]
    let vnni = is_x86_feature_detected!("avxvnni");
    #[cfg(target_arch = "x86_64")]
    let avx2 = is_x86_feature_detected!("avx2");
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
    (0..m).into_par_iter().for_each(|i| {
        // SAFETY: disjoint output row i; per-row i32 scratch; pointers valid by contract.
        unsafe {
            let (a, b, out) = (au as *const u8, bu as *const i8, ou as *mut f32);
            let (scale_b, bias) = (sbu as *const f32, biu as *const f32);
            let mut row = vec![0i32; n];
            let r = row.as_mut_ptr();
            let arow = a.add(i * k);
            #[cfg(target_arch = "x86_64")]
            {
                if vnni {
                    gemm_row_nt_vnni(arow, b, r, k, n);
                } else if avx2 {
                    gemm_row_nt_avx2(arow, b, r, k, n);
                } else {
                    for j in 0..n {
                        *r.add(j) = dot_i8_scalar(arow, b.add(j * k), k);
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            for j in 0..n {
                *r.add(j) = dot_i8_scalar(arow, b.add(j * k), k);
            }
            dequant_row(r, out.add(i * n), n, scale_a, scale_b, bias, act);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, varied fills (no RNG — reproducible). Activations span the full u8 range; weights
    // span the full i8 range, so products exercise both signs and the magnitude extremes.
    fn fill_u8(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + 11) % 256) as u8).collect()
    }
    fn fill_i8(n: usize) -> Vec<i8> {
        (0..n)
            .map(|i| (((i * 53 + 7) % 256) as i32 - 128) as i8)
            .collect()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // Sizes straddling the 32/16-wide chunks and the scalar tail.
        for &k in &[1usize, 7, 15, 16, 17, 31, 32, 33, 48, 64, 100, 257, 1024] {
            let a = fill_u8(k);
            let b = fill_i8(k);
            let s = unsafe { dot_i8_scalar(a.as_ptr(), b.as_ptr(), k) };
            let v = unsafe { dot_i8_avx2(a.as_ptr(), b.as_ptr(), k) };
            assert_eq!(s, v, "scalar != avx2 at k={k}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot4_matches_scalar_bit_for_bit() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // The 4-wide blocked path must equal four independent scalar dots, at every K (incl. the
        // 16-wide-chunk boundaries and the scalar tail).
        for &k in &[1usize, 7, 15, 16, 17, 31, 32, 48, 63, 64, 100, 257, 1024] {
            let a = fill_u8(k);
            let b: Vec<i8> = fill_i8(4 * k);
            let (b0, b1, b2, b3) = (
                b.as_ptr(),
                b.as_ptr().wrapping_add(k),
                b.as_ptr().wrapping_add(2 * k),
                b.as_ptr().wrapping_add(3 * k),
            );
            let got = unsafe { dot4_i8_avx2(a.as_ptr(), b0, b1, b2, b3, k) };
            let want = unsafe {
                (
                    dot_i8_scalar(a.as_ptr(), b0, k),
                    dot_i8_scalar(a.as_ptr(), b1, k),
                    dot_i8_scalar(a.as_ptr(), b2, k),
                    dot_i8_scalar(a.as_ptr(), b3, k),
                )
            };
            assert_eq!(got, want, "dot4 != 4×scalar at k={k}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot4_vnni_matches_scalar_bit_for_bit() {
        if !is_x86_feature_detected!("avxvnni") {
            return;
        }
        // The AVX-VNNI (vpdpbusd) 4-wide path must equal four scalar dots at every K, incl. the
        // 32-wide-chunk boundaries and the scalar tail. vpdpbusd (non-saturating) sums the products
        // into i32 exactly, so the bar is bit-exact (no tolerance).
        for &k in &[1usize, 7, 16, 31, 32, 33, 47, 64, 65, 96, 100, 257, 1024] {
            let a = fill_u8(k);
            let b: Vec<i8> = fill_i8(4 * k);
            let (b0, b1, b2, b3) = (
                b.as_ptr(),
                b.as_ptr().wrapping_add(k),
                b.as_ptr().wrapping_add(2 * k),
                b.as_ptr().wrapping_add(3 * k),
            );
            let got = unsafe { dot4_i8_vnni(a.as_ptr(), b0, b1, b2, b3, k) };
            let want = unsafe {
                (
                    dot_i8_scalar(a.as_ptr(), b0, k),
                    dot_i8_scalar(a.as_ptr(), b1, k),
                    dot_i8_scalar(a.as_ptr(), b2, k),
                    dot_i8_scalar(a.as_ptr(), b3, k),
                )
            };
            assert_eq!(got, want, "dot4_vnni != 4×scalar at k={k}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot2x4_vnni_matches_scalar_bit_for_bit() {
        if !is_x86_feature_detected!("avxvnni") {
            return;
        }
        // The 2×4 tile (two A-rows × four B-rows) must equal eight scalar dots at every K.
        for &k in &[1usize, 7, 16, 31, 32, 33, 47, 64, 65, 96, 100, 257, 1024] {
            let a = fill_u8(2 * k);
            let b = fill_i8(4 * k);
            let (a0, a1) = (a.as_ptr(), a.as_ptr().wrapping_add(k));
            let bp = |c: usize| b.as_ptr().wrapping_add(c * k);
            let got = unsafe { dot2x4_i8_vnni(a0, a1, bp(0), bp(1), bp(2), bp(3), k) };
            let mut want = [0i32; 8];
            for col in 0..4 {
                want[col] = unsafe { dot_i8_scalar(a0, bp(col), k) };
                want[col + 4] = unsafe { dot_i8_scalar(a1, bp(col), k) };
            }
            assert_eq!(got, want, "dot2x4_vnni != 8×scalar at k={k}");
        }
    }

    #[test]
    fn i8gemm_deq_matches_unfused_and_parallel() {
        // The fused dequant kernel must equal: the i32 GEMM, then a separate scalar dequant pass —
        // bit-for-bit, for every activation and bias setting, serial == parallel. This is the contract
        // the recognizer relies on (the interp marshals this kernel; native JITs the same one).
        for &(m, k, n) in &[
            (1usize, 7usize, 5usize),
            (2, 16, 8),
            (3, 33, 4),
            (5, 64, 9),
            (8, 100, 16),
        ] {
            let a = fill_u8(m * k);
            let b = fill_i8(n * k);
            let scale_a = 0.018f32;
            let scale_b: Vec<f32> = (0..n).map(|j| 0.003 + (j % 5) as f32 * 0.001).collect();
            let bias: Vec<f32> = (0..n).map(|j| (j as f32) * 0.01 - 0.05).collect();
            // i32 GEMM reference (the unfused path the dequant fuses).
            let mut ci = vec![0i32; m * n];
            unsafe {
                wukong_i8gemm_nt(
                    a.as_ptr(),
                    b.as_ptr(),
                    ci.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                );
            }
            for &act in &[0i64, 1, 2, 3] {
                for use_bias in [false, true] {
                    let bias_ptr = if use_bias {
                        bias.as_ptr()
                    } else {
                        std::ptr::null()
                    };
                    let mut want = vec![0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            let mut v = (ci[i * n + j] as f32) * scale_a * scale_b[j];
                            if use_bias {
                                v += bias[j];
                            }
                            v = match act {
                                1 => {
                                    if v > 0.0 {
                                        v
                                    } else {
                                        0.0
                                    }
                                }
                                2 => crate::vmath::gelu1(v),
                                3 => crate::vmath::silu1(v),
                                _ => v,
                            };
                            want[i * n + j] = v;
                        }
                    }
                    let mut got = vec![0f32; m * n];
                    let mut got_par = vec![0f32; m * n];
                    unsafe {
                        wukong_i8gemm_nt_deq(
                            a.as_ptr(),
                            b.as_ptr(),
                            got.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            scale_a,
                            scale_b.as_ptr(),
                            bias_ptr,
                            act,
                        );
                        wukong_i8gemm_nt_deq_parallel(
                            a.as_ptr(),
                            b.as_ptr(),
                            got_par.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            scale_a,
                            scale_b.as_ptr(),
                            bias_ptr,
                            act,
                        );
                    }
                    assert_eq!(
                        got, want,
                        "fused deq != unfused (m={m} k={k} n={n} act={act} bias={use_bias})"
                    );
                    assert_eq!(
                        got, got_par,
                        "deq serial != parallel (m={m} k={k} n={n} act={act} bias={use_bias})"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_i64_reference() {
        // The kernel result equals the true integer dot reduced mod 2³² (i.e. cast to i32) — the
        // exact same value the naive `s += a[k]*b[k]` loop produces. Verified against an i64 sum.
        let (m, k, n) = (5usize, 100usize, 7usize);
        let a = fill_u8(m * k);
        let b = fill_i8(n * k);
        let mut c = vec![0i32; m * n];
        unsafe {
            wukong_i8gemm_nt(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
            );
        }
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for t in 0..k {
                    acc += a[i * k + t] as i64 * b[j * k + t] as i64;
                }
                assert_eq!(c[i * n + j], acc as i32, "C[{i},{j}] mismatch");
            }
        }
    }

    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        let (m, k, n) = (33usize, 257usize, 9usize);
        let a = fill_u8(m * k);
        let b = fill_i8(n * k);
        let mut cs = vec![0i32; m * n];
        let mut cp = vec![0i32; m * n];
        unsafe {
            wukong_i8gemm_nt(
                a.as_ptr(),
                b.as_ptr(),
                cs.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
            );
            wukong_i8gemm_nt_parallel(
                a.as_ptr(),
                b.as_ptr(),
                cp.as_mut_ptr(),
                m as i64,
                k as i64,
                n as i64,
            );
        }
        assert_eq!(cs, cp, "serial != parallel");
    }

    #[test]
    fn overflow_wraps_consistently() {
        // Max-magnitude inputs over a large K overflow i32; scalar and AVX2 must wrap identically
        // (both are sums mod 2³²), proving the gate stays exact even past i32 range.
        let k = 100_000usize;
        let a = vec![255u8; k];
        let b = vec![127i8; k];
        let s = unsafe { dot_i8_scalar(a.as_ptr(), b.as_ptr(), k) };
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let v = unsafe { dot_i8_avx2(a.as_ptr(), b.as_ptr(), k) };
            assert_eq!(s, v, "scalar != avx2 under overflow");
        }
        // The true sum is 255*127*k = 4_064_250_000, which wraps in i32.
        let want = (255i64 * 127 * k as i64) as i32;
        assert_eq!(s, want, "overflow wrap mismatch");
    }
}
