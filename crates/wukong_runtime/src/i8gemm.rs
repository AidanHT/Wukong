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
//! the **identical** kernel, so the differential oracle stays bit-for-bit exact.
//!
//! **Cache blocking (P9.5, 2026-10).** Until then the whole kernel was `for each pair of A rows:
//! stream ALL of B`, with no blocking at all, so B — the weight matrix — was re-read from memory
//! once per two rows of A: at a quantized FFN shape (m=512, n=3072, k=768) that is 256 passes over
//! 2.4 MB of weights, ~600 MB of traffic for a 1.2 GMAC problem, capping `vpdpbusd` (the one
//! instruction on this part with 4x the f32 FMA MAC rate) at memory bandwidth exactly where int8
//! should win. Both operands are already contiguous over the contraction, so — unlike f32 — no
//! repacking is needed (the dot kernels consume 32-byte runs of each row directly); what was missing
//! is a column block of B kept L2-resident while every A row streams past it. [`i8_col_block`]
//! sizes that block to [`I8_B_BLOCK_BYTES`] of B, the serial entries walk column blocks outermost,
//! and the `_parallel` entries split C into a 2D grid of (row block × column block) tasks
//! ([`i8_par_grid`]) instead of one task per row — so a decode-shaped call (m = 1..8), which used to
//! run on at most m cores, now spreads its columns across the pool, and every task reuses its B
//! block across up to 32 rows. Every C element is still one full-`k` dot from the same per-row
//! helpers, so blocking changes no bit (and wrapping `i32` addition is order-immaterial anyway):
//! `serial == parallel` with no cross-task combine, independent of thread count and grid.

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

/// Which int8 dot-product body a call runs. Detected **once per call** ([`i8_tier`]), never per
/// element, so the hot loops are branch-free `target_feature` code; all three tiers produce the same
/// `i32` for every input (wrapping `i32` is associative and commutative mod 2³²).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum I8Tier {
    /// AVX-VNNI `vpdpbusd`, 2×4 register tile.
    #[cfg(target_arch = "x86_64")]
    Vnni,
    /// AVX2 widen + `vpmaddwd`, 1×4 register tile.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// The portable scalar nest.
    Scalar,
}

/// The densest tier this host supports: VNNI → AVX2 → scalar.
fn i8_tier() -> I8Tier {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::isa::avxvnni() {
            return I8Tier::Vnni;
        }
        if crate::isa::avx2() {
            return I8Tier::Avx2;
        }
    }
    I8Tier::Scalar
}

/// Bytes of B (`ncb` rows of `k` int8 weights) one column block keeps resident while every A row
/// streams past it: 1 MiB, half the 2 MiB P-core L2 of the development part (the other half holds
/// the streamed A rows, the C rows and whatever else is live), and the full per-core share of an
/// E-core cluster's 4 MiB L2. Blocking-only: no result bit depends on it.
const I8_B_BLOCK_BYTES: usize = 1 << 20;

/// Column-block width (B rows per block) for a contraction length `k`: as many rows as fit
/// [`I8_B_BLOCK_BYTES`], rounded down to the 4-column register tile and never below one tile, so
/// the per-row helpers' 1-column tails only ever run at the true matrix edge.
fn i8_col_block(k: usize) -> usize {
    (I8_B_BLOCK_BYTES / k.max(1)).max(4) & !3
}

/// The `_parallel` task grid: `rb` rows by `cb` columns per task. `rb` is even (so the VNNI tier
/// keeps its 2-row tile) and at most 32 — every task re-reads its B block from L2 once per row pair,
/// so taller tasks reuse it longer; `cb` is a multiple of 4 in `[4, i8_col_block(k)]`, rounded DOWN
/// from `n / (column blocks wanted)` so the grid reaches `3 × nthreads` tasks whenever the 4-column
/// granularity allows — a short (decode) GEMM spreads its columns across the pool instead of
/// leaving it idle (pinned by `i8_par_grid_policy`). Throughput-only: every C element is the same
/// full-`k` dot whatever the grid.
fn i8_par_grid(m: usize, n: usize, k: usize, nthreads: usize) -> (usize, usize) {
    const RB_MAX: usize = 32;
    let target = 3 * nthreads.max(1);
    let rb = RB_MAX.min(m.div_ceil(2) * 2);
    let col_blocks_wanted = target.div_ceil(m.div_ceil(rb));
    let cb = ((n / col_blocks_wanted) & !3).clamp(4, i8_col_block(k));
    (rb, cb)
}

/// Rows `[i0, i0 + mr)` × columns `[j0, j0 + nc)` of `C = A·Bᵀ`, written to `out` with row stride
/// `ldo` (`out` addresses the block's top-left `C` element, or a scratch tile). Every element is the
/// full-`k` dot of the per-row helpers the unblocked kernel used, so any tiling of rows and columns
/// produces the same `i32`s. The VNNI tier walks the rows in pairs (the 2×4 tile: each B-row chunk
/// loaded once feeds both rows), the others one row at a time.
///
/// # Safety
/// `a` valid for rows `[i0, i0 + mr)` of `k` `u8`; `b` for rows `[j0, j0 + nc)` of `k` `i8`; `out`
/// for `mr` rows of `nc` `i32` at stride `ldo`; `tier` from [`i8_tier`] on this host.
#[allow(clippy::too_many_arguments)]
unsafe fn i8_tile(
    tier: I8Tier,
    a: *const u8,
    b: *const i8,
    k: usize,
    i0: usize,
    mr: usize,
    j0: usize,
    nc: usize,
    out: *mut i32,
    ldo: usize,
) {
    let bj = b.add(j0 * k);
    match tier {
        #[cfg(target_arch = "x86_64")]
        I8Tier::Vnni => {
            let mut r = 0;
            while r + 2 <= mr {
                gemm_2rows_nt_vnni(
                    a.add((i0 + r) * k),
                    a.add((i0 + r + 1) * k),
                    bj,
                    out.add(r * ldo),
                    out.add((r + 1) * ldo),
                    k,
                    nc,
                );
                r += 2;
            }
            if r < mr {
                gemm_row_nt_vnni(a.add((i0 + r) * k), bj, out.add(r * ldo), k, nc);
            }
        }
        #[cfg(target_arch = "x86_64")]
        I8Tier::Avx2 => {
            for r in 0..mr {
                gemm_row_nt_avx2(a.add((i0 + r) * k), bj, out.add(r * ldo), k, nc);
            }
        }
        I8Tier::Scalar => {
            for r in 0..mr {
                let (arow, orow) = (a.add((i0 + r) * k), out.add(r * ldo));
                for j in 0..nc {
                    *orow.add(j) = dot_i8_scalar(arow, bj.add(j * k), k);
                }
            }
        }
    }
}

/// Quantized `nn.Linear` `C = A·Bᵀ` (serial): `A` is `[m,k]` `u8` row-major, `B` is `[n,k]` `i8`
/// row-major (so `B`'s rows are the weight vectors), `C` is `[m,n]` `i32` row-major. Detects the SIMD
/// tier **once** per call (not per element) — AVX-VNNI (2×4 register tile) → widen+`vpmaddwd` AVX2
/// (1×4) → scalar nest — and all three are bit-identical (wrapping `i32` is order-immaterial).
/// Column blocks of [`i8_col_block`] B rows are walked outermost, so each block is read from memory
/// once and from L2 by every row of A (see the module header).
///
/// # Safety
/// `a` valid for `m*k` `u8`, `b` for `n*k` `i8`, `c` for `m*n` `i32`.
/// **Aliasing: all operands disjoint** — the `i32` accumulator `c` must overlap neither int8
/// operand (it is written in tiled order while both are still read).
#[no_mangle]
pub unsafe extern "C" fn wukong_i8gemm_nt(
    a: *const u8,
    b: *const i8,
    c: *mut i32,
    m: i64,
    k: i64,
    n: i64,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let tier = i8_tier();
    let ncb = i8_col_block(k);
    let mut j0 = 0;
    while j0 < n {
        let nc = (n - j0).min(ncb);
        // SAFETY: rows [0, m) of A, B rows [j0, j0+nc), C's column block at row stride n.
        i8_tile(tier, a, b, k, 0, m, j0, nc, c.add(j0), n);
        j0 += nc;
    }
}

/// Multicore `C = A·Bᵀ` — **bit-identical** to [`wukong_i8gemm_nt`]. C is split into the
/// [`i8_par_grid`] 2D grid of disjoint (row block × column block) tasks, each computing its
/// elements with the same full-`k` per-row helpers, so the result is independent of thread count
/// and grid and the interpreter (serial) agrees with the native `@parallel` path exactly.
///
/// # Safety
/// Same as [`wukong_i8gemm_nt`].
/// **Aliasing:** that of [`wukong_i8gemm_nt`] — all operands disjoint: `c` overlaps neither `a` nor
/// `b`.
#[no_mangle]
pub unsafe extern "C" fn wukong_i8gemm_nt_parallel(
    a: *const u8,
    b: *const i8,
    c: *mut i32,
    m: i64,
    k: i64,
    n: i64,
) {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(c, m, n);
        crate::safety::debug_disjoint(c, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(c, m.saturating_mul(n), b, k.saturating_mul(n));
    }
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    // Work gate + width-1 fast path (`par::fork_units`, m*k*n int8 MACs): a small int8 GEMM no
    // longer forks at all. The independent units are the grid's (row x 4-column) cells, not the rows:
    // the 2D grid below splits a single-token (m = 1) decode GEMM across its columns, so gating on
    // `m` alone would keep exactly that case serial. Integer math, cells independent: bit-identical.
    if !crate::par::fork_units(i8_units(mu, nu), i8_work(mu, ku, nu)) {
        return wukong_i8gemm_nt(a, b, c, m, k, n);
    }
    let (m, k, n) = (mu, ku, nu);
    // Raw pointers cross the rayon boundary as integers (same pattern as the parallel GEMM/reduce).
    let (au, bu, cu) = (a as usize, b as usize, c as usize);
    let tier = i8_tier();
    // The (row block x column block) grid of `i8_par_grid`, sized for the pool that actually runs it
    // (`wuk_pool_width`, which also provisions the 16 MiB-stack global pool before any fork), with
    // the tasks claimed from the shared queue on the unified kernel pool (`par::claim_chunks`) so
    // this kernel runs beside its neighbours instead of on the bare global registry. Every C element
    // is the same full-`k` dot whatever the grid or the claim order: bit-identical.
    let (rb, cb) = i8_par_grid(m, n, k, crate::wuk_pool_width());
    let (nrb, ncbs) = (m.div_ceil(rb), n.div_ceil(cb));
    crate::par::claim_chunks(nrb * ncbs, 1, true, |lo, hi| {
        for t in lo..hi {
            let (i0, j0) = ((t / ncbs) * rb, (t % ncbs) * cb);
            let (mr, nc) = ((m - i0).min(rb), (n - j0).min(cb));
            let (a, b, c) = (au as *const u8, bu as *const i8, cu as *mut i32);
            // SAFETY: each task owns the disjoint C block [i0, i0+mr) x [j0, j0+nc); A and B are
            // read-only and valid for the declared extents by contract; `tier` was detected above.
            unsafe { i8_tile(tier, a, b, k, i0, mr, j0, nc, c.add(i0 * n + j0), n) };
        }
    });
}

/// The int8 GEMM's weighted work for the shared fork gate: `m·k·n` MACs scaled down by
/// [`crate::par::I8_MACS_PER_WORK`] (saturating). Scheduling-only.
fn i8_work(m: usize, k: usize, n: usize) -> usize {
    m.saturating_mul(k).saturating_mul(n) / crate::par::I8_MACS_PER_WORK
}

/// The int8 GEMM's independent units for the shared fork gate: the (row x 4-column) cells that
/// [`i8_par_grid`]'s finest grid can hand to distinct workers. Scheduling-only.
fn i8_units(m: usize, n: usize) -> usize {
    m.saturating_mul(n.div_ceil(4))
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
/// trips through main memory: each pair of C rows of one column block is computed into a small
/// L1-resident i32 tile by the *same* proven per-row helpers `wukong_i8gemm_nt` uses (so the integer
/// GEMM stays bit-exact), then immediately dequantized to f32. gcc/cuBLAS **cannot** fuse this — their
/// int8 GEMM emits i32 and the dequant is a separate kernel — so it is the decode-regime win, where
/// the dequant pass is a non-trivial fraction of a thin-M GEMM. The 2-row VNNI tile is preserved
/// (B-row loads feed both rows) and the column blocks are those of [`wukong_i8gemm_nt`]: the
/// dequant is per element, so applying it one column block at a time (with `scale_b`/`bias` offset
/// to the block) computes exactly the same f32 for every element.
///
/// `scale_a` = per-tensor activation scale; `scale_b` = per-output-channel weight scale (length `n`);
/// `bias` = per-output-channel (length `n`) or null; `act` = 0/1/2/3 (id/ReLU/GELU/SiLU).
///
/// # Safety
/// `a` valid for `m*k` u8, `b` for `n*k` i8, `out` for `m*n` f32, `scale_b` for `n` f32, `bias` null or
/// `n` f32.
/// When non-null, the `n`-element bias range must be disjoint from the `m*n`-element
/// output range; rows may read bias while other output elements are written.
/// **Aliasing: all operands disjoint** — `out` overlaps none of `a`, `b`, `scale_b`, `bias`.
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(out, m, n);
        crate::safety::debug_extent(scale_b, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(out, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(out, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(out, m.saturating_mul(n), scale_b, n);
        crate::safety::debug_disjoint(out, m.saturating_mul(n), bias, n);
    }
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let tier = i8_tier();
    let ncb = i8_col_block(k);
    deq_block(
        tier,
        a,
        b,
        out,
        DeqArgs {
            k,
            n,
            scale_a,
            scale_b,
            bias,
            act,
        },
        0,
        m,
        0,
        n,
        ncb,
    );
}

/// The per-call constants of a fused-dequant int8 GEMM, bundled so a block body can be handed
/// across a rayon boundary as one `Send` value (see [`wukong_i8gemm_nt_deq_parallel`]).
#[derive(Clone, Copy)]
struct DeqArgs {
    k: usize,
    n: usize,
    scale_a: f32,
    scale_b: *const f32,
    bias: *const f32,
    act: i64,
}
// SAFETY: the pointers are read-only views the caller guarantees for the whole call; every task
// joins before the extern entry returns, exactly the lifetime discipline of the `usize`-smuggled
// operand pointers beside them.
// SAFETY: `DeqArgs` bundles raw operand addresses for the parallel dequant; each worker writes a DISJOINT output band, so sharing it across threads is race-free.
unsafe impl Send for DeqArgs {}
// SAFETY: see above -- disjoint output bands make `DeqArgs` safe to share across threads.
unsafe impl Sync for DeqArgs {}

/// Rows `[i0, i0 + mr)` × columns `[j0, j0 + nc)` of the fused-dequant GEMM, walked in column blocks
/// of at most `cb`: each pair of rows of a column block is computed into an L1-resident i32 tile by
/// [`i8_tile`] and dequantized straight into `out` by [`dequant_row`], with `scale_b`/`bias` offset to
/// the block (a null bias stays null).
///
/// # Safety
/// The [`wukong_i8gemm_nt_deq`] operand contract for the whole matrix; the block lies inside it.
#[allow(clippy::too_many_arguments)]
unsafe fn deq_block(
    tier: I8Tier,
    a: *const u8,
    b: *const i8,
    out: *mut f32,
    d: DeqArgs,
    i0: usize,
    mr: usize,
    j0: usize,
    nc: usize,
    cb: usize,
) {
    let width = nc.min(cb);
    let mut tile = vec![0i32; 2 * width];
    let t = tile.as_mut_ptr();
    let mut jb = j0;
    while jb < j0 + nc {
        let w = (j0 + nc - jb).min(cb);
        let scale_bj = d.scale_b.add(jb);
        let bias_j = if d.bias.is_null() {
            d.bias
        } else {
            d.bias.add(jb)
        };
        let mut i = i0;
        while i < i0 + mr {
            let rows = (i0 + mr - i).min(2);
            i8_tile(tier, a, b, d.k, i, rows, jb, w, t, w);
            for r in 0..rows {
                dequant_row(
                    t.add(r * w),
                    out.add((i + r) * d.n + jb),
                    w,
                    d.scale_a,
                    scale_bj,
                    bias_j,
                    d.act,
                );
            }
            i += rows;
        }
        jb += w;
    }
}

/// Multicore fused-dequant `nn.Linear` — **bit-identical** to [`wukong_i8gemm_nt_deq`]: the same
/// [`i8_par_grid`] tasks as [`wukong_i8gemm_nt_parallel`], each running [`deq_block`] on its disjoint
/// block with its own i32 tile (one allocation per task, where the per-row form allocated one per
/// row), dequantized by the identical `dequant_row`. The interpreter calls the serial form, so
/// serial == parallel == interp.
///
/// # Safety
/// Same as [`wukong_i8gemm_nt_deq`].
/// **Aliasing:** that of [`wukong_i8gemm_nt_deq`] — all operands disjoint: `out` overlaps none of
/// `a`, `b`, `scale_b`, `bias`.
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
    // P14.5: debug-build contract checks (the `# Safety` section above).
    if cfg!(debug_assertions) && m > 0 && n > 0 {
        crate::safety::debug_extent_2d(a, m, k);
        crate::safety::debug_extent_2d(b, n, k);
        crate::safety::debug_extent_2d(out, m, n);
        crate::safety::debug_extent(scale_b, n);
        crate::safety::debug_extent_opt(bias, n);
        crate::safety::debug_disjoint(out, m.saturating_mul(n), a, m.saturating_mul(k));
        crate::safety::debug_disjoint(out, m.saturating_mul(n), b, k.saturating_mul(n));
        crate::safety::debug_disjoint(out, m.saturating_mul(n), scale_b, n);
        crate::safety::debug_disjoint(out, m.saturating_mul(n), bias, n);
    }
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }
    let (mu, ku, nu) = (m as usize, k as usize, n as usize);
    // Work gate + width-1 fast path, as in `wukong_i8gemm_nt_parallel`.
    if !crate::par::fork_units(i8_units(mu, nu), i8_work(mu, ku, nu)) {
        return wukong_i8gemm_nt_deq(a, b, out, m, k, n, scale_a, scale_b, bias, act);
    }
    let (m, k, n) = (mu, ku, nu);
    let (au, bu, ou) = (a as usize, b as usize, out as usize);
    let d = DeqArgs {
        k,
        n,
        scale_a,
        scale_b,
        bias,
        act,
    };
    let tier = i8_tier();
    // Same grid, pool and claim queue as `wukong_i8gemm_nt_parallel`; each task runs `deq_block` on
    // its disjoint block with its own i32 tile, dequantized by the identical `dequant_row`.
    let (rb, cb) = i8_par_grid(m, n, k, crate::wuk_pool_width());
    let (nrb, ncbs) = (m.div_ceil(rb), n.div_ceil(cb));
    crate::par::claim_chunks(nrb * ncbs, 1, true, |lo, hi| {
        for t in lo..hi {
            let (i0, j0) = ((t / ncbs) * rb, (t % ncbs) * cb);
            let (mr, nc) = ((m - i0).min(rb), (n - j0).min(cb));
            // SAFETY: each task owns the disjoint output block [i0, i0+mr) x [j0, j0+nc) and its
            // own i32 tile; operands are read-only and valid by contract; `tier` was detected above.
            unsafe {
                deq_block(
                    tier,
                    au as *const u8,
                    bu as *const i8,
                    ou as *mut f32,
                    d,
                    i0,
                    mr,
                    j0,
                    nc,
                    cb,
                )
            };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_extents_are_safe() {
        // P14.6: C = A * B^T, A [m,k] u8 ones, B [n,k] i8 ones -> C[i,j] = k.
        let (m, k, n) = (2usize, 2usize, 2usize);
        let a = vec![1u8; m * k];
        let b = vec![1i8; n * k];
        let mut c = vec![-7i32; m * n];
        // SAFETY: `a`/`b`/`c` sized m*k/n*k/m*n; a nonpositive m is a no-op.
        unsafe {
            wukong_i8gemm_nt(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                0,
                k as i64,
                n as i64,
            );
        }
        assert!(c.iter().all(|&v| v == -7), "nonpositive m must not write c");
        // SAFETY: `a`/`b`/`c` sized m*k/n*k/m*n; the positive call writes c.
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
        assert_eq!(c, vec![2, 2, 2, 2]);
    }

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
        if !crate::isa::avx2() {
            return;
        }
        // Sizes straddling the 32/16-wide chunks and the scalar tail.
        for &k in &[1usize, 7, 15, 16, 17, 31, 32, 33, 48, 64, 100, 257, 1024] {
            let a = fill_u8(k);
            let b = fill_i8(k);
            // SAFETY: `a`/`b` hold k quantized values; dot_i8_scalar reads k.
            let s = unsafe { dot_i8_scalar(a.as_ptr(), b.as_ptr(), k) };
            // SAFETY: AVX2 confirmed by the enclosing feature guard; `a`/`b` hold k values.
            let v = unsafe { dot_i8_avx2(a.as_ptr(), b.as_ptr(), k) };
            assert_eq!(s, v, "scalar != avx2 at k={k}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot4_matches_scalar_bit_for_bit() {
        if !crate::isa::avx2() {
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
            // SAFETY: AVX2 confirmed by the enclosing feature guard; `a` holds k u8 and b0..b3 each k i8.
            let got = unsafe { dot4_i8_avx2(a.as_ptr(), b0, b1, b2, b3, k) };
            // SAFETY: the four reference dots read k each from `a` and the b pointers (scalar).
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
        if !crate::isa::avxvnni() {
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
            // SAFETY: AVX-VNNI confirmed by the enclosing feature guard; `a` holds k u8 and b0..b3 each k i8.
            let got = unsafe { dot4_i8_vnni(a.as_ptr(), b0, b1, b2, b3, k) };
            // SAFETY: the four reference dots read k each (scalar).
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
        if !crate::isa::avxvnni() {
            return;
        }
        // The 2×4 tile (two A-rows × four B-rows) must equal eight scalar dots at every K.
        for &k in &[1usize, 7, 16, 31, 32, 33, 47, 64, 65, 96, 100, 257, 1024] {
            let a = fill_u8(2 * k);
            let b = fill_i8(4 * k);
            let (a0, a1) = (a.as_ptr(), a.as_ptr().wrapping_add(k));
            let bp = |c: usize| b.as_ptr().wrapping_add(c * k);
            // SAFETY: AVX-VNNI confirmed by the enclosing feature guard; a0/a1 each k u8 and bp(0..3) each k i8.
            let got = unsafe { dot2x4_i8_vnni(a0, a1, bp(0), bp(1), bp(2), bp(3), k) };
            let mut want = [0i32; 8];
            for col in 0..4 {
                // SAFETY: `a0` and `bp(col)` each hold k values (scalar reference).
                want[col] = unsafe { dot_i8_scalar(a0, bp(col), k) };
                // SAFETY: `a1` and `bp(col)` each hold k values (scalar reference).
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
            // SAFETY: `a`/`b` sized m*k/n*k; `ci` sized m*n (the i32 reference).
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
                    // SAFETY: `a`/`b` sized m*k/n*k; `got`/`got_par` sized m*n (dequant).
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
        // SAFETY: `a`/`b` sized m*k/n*k; `c` sized m*n for the dispatched i8 GEMM.
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
        // SAFETY: `cs`/`cp` sized m*n (serial vs parallel).
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

    /// The cache-blocked kernel against the exact integer reference at a shape DERIVED from the
    /// blocking constant, so it spans three column blocks (two full, one ragged) whatever
    /// [`I8_B_BLOCK_BYTES`] is tuned to, with a `k` that is not a multiple of the 32-byte VNNI
    /// step (so every dot has a scalar tail) and an odd `m` (so the VNNI tier's single-row tail
    /// runs). Checked: the serial entry, the parallel entry (bit-for-bit), the portable scalar tier
    /// driven directly on the same blocks, and both fused-dequant entries against the unfused
    /// formula — every value exact, no tolerance.
    #[test]
    fn blocked_kernel_matches_i64_reference_across_column_blocks() {
        let k = 4000usize;
        let ncb = i8_col_block(k);
        let (m, n) = (7usize, 2 * ncb + 6);
        assert_eq!(n.div_ceil(ncb), 3, "shape must span three column blocks");
        let a = fill_u8(m * k);
        let b = fill_i8(n * k);
        let mut want = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for t in 0..k {
                    acc += a[i * k + t] as i64 * b[j * k + t] as i64;
                }
                want[i * n + j] = acc as i32;
            }
        }
        let (mi, ki, ni) = (m as i64, k as i64, n as i64);
        let mut serial = vec![0i32; m * n];
        let mut par = vec![0i32; m * n];
        let mut scalar = vec![0i32; m * n];
        // SAFETY: `par`/`scalar` sized m*n.
        unsafe {
            wukong_i8gemm_nt(a.as_ptr(), b.as_ptr(), serial.as_mut_ptr(), mi, ki, ni);
            wukong_i8gemm_nt_parallel(a.as_ptr(), b.as_ptr(), par.as_mut_ptr(), mi, ki, ni);
            let mut j0 = 0;
            while j0 < n {
                let nc = (n - j0).min(ncb);
                i8_tile(
                    I8Tier::Scalar,
                    a.as_ptr(),
                    b.as_ptr(),
                    k,
                    0,
                    m,
                    j0,
                    nc,
                    scalar.as_mut_ptr().add(j0),
                    n,
                );
                j0 += nc;
            }
        }
        assert_eq!(serial, want, "blocked serial != i64 reference");
        assert_eq!(par, want, "blocked parallel != i64 reference");
        assert_eq!(scalar, want, "scalar tier != i64 reference");
        // Fused dequant, serial and parallel, against the unfused formula on the exact i32s.
        let scale_a = 0.0125f32;
        let scale_b: Vec<f32> = (0..n).map(|j| 0.002 + (j % 7) as f32 * 0.0005).collect();
        let bias: Vec<f32> = (0..n).map(|j| (j % 11) as f32 * 0.25 - 1.0).collect();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        for &act in &[0i64, 1, 3] {
            let mut expect = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut v = (want[i * n + j] as f32) * scale_a * scale_b[j];
                    v += bias[j];
                    expect[i * n + j] = match act {
                        1 => {
                            if v > 0.0 {
                                v
                            } else {
                                0.0
                            }
                        }
                        3 => crate::vmath::silu1(v),
                        _ => v,
                    };
                }
            }
            let mut got = vec![0f32; m * n];
            let mut got_par = vec![0f32; m * n];
            // SAFETY: `got`/`got_par` sized m*n (dequant).
            unsafe {
                wukong_i8gemm_nt_deq(
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    scale_a,
                    scale_b.as_ptr(),
                    bias.as_ptr(),
                    act,
                );
                wukong_i8gemm_nt_deq_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    got_par.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    scale_a,
                    scale_b.as_ptr(),
                    bias.as_ptr(),
                    act,
                );
            }
            assert_eq!(
                bits(&got),
                bits(&expect),
                "blocked deq != unfused (act {act})"
            );
            assert_eq!(
                bits(&got_par),
                bits(&expect),
                "blocked deq parallel != unfused (act {act})"
            );
        }
    }

    /// Decode- and prefill-shaped calls: `m` from 1 (a single token, which the per-row parallel
    /// split used to run on ONE core) to a tall odd `m`, with `n` below, at and above one 4-column
    /// tile. Serial == parallel bit-for-bit for the plain and fused-dequant entries.
    #[test]
    fn parallel_grid_matches_serial_on_decode_and_tall_shapes() {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        for &(m, k, n) in &[
            (1usize, 96usize, 3usize),
            (1, 768, 300),
            (2, 33, 64),
            (3, 257, 129),
            (37, 64, 41),
        ] {
            let a = fill_u8(m * k);
            let b = fill_i8(n * k);
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            let (mut s, mut p) = (vec![0i32; m * n], vec![0i32; m * n]);
            let scale_b: Vec<f32> = (0..n).map(|j| 0.01 + j as f32 * 1e-4).collect();
            let (mut ds, mut dp) = (vec![0f32; m * n], vec![0f32; m * n]);
            // SAFETY: `ds`/`dp` sized m*n; `scale_b` sized n.
            unsafe {
                wukong_i8gemm_nt(a.as_ptr(), b.as_ptr(), s.as_mut_ptr(), mi, ki, ni);
                wukong_i8gemm_nt_parallel(a.as_ptr(), b.as_ptr(), p.as_mut_ptr(), mi, ki, ni);
                wukong_i8gemm_nt_deq(
                    a.as_ptr(),
                    b.as_ptr(),
                    ds.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0.5,
                    scale_b.as_ptr(),
                    std::ptr::null(),
                    1,
                );
                wukong_i8gemm_nt_deq_parallel(
                    a.as_ptr(),
                    b.as_ptr(),
                    dp.as_mut_ptr(),
                    mi,
                    ki,
                    ni,
                    0.5,
                    scale_b.as_ptr(),
                    std::ptr::null(),
                    1,
                );
            }
            assert_eq!(s, p, "serial != parallel ({m}x{k}x{n})");
            assert_eq!(bits(&ds), bits(&dp), "deq serial != parallel ({m}x{k}x{n})");
        }
    }

    /// [`i8_par_grid`] as a pure policy: an even row block in `[2, 32]`, a column block that is a
    /// multiple of the 4-column tile inside `[4, i8_col_block(k)]`, and at least `3 × nthreads`
    /// tasks whenever the 4-column granularity allows that many — in particular a single-token
    /// decode GEMM must fill a 16-thread pool rather than run on one core.
    #[test]
    fn i8_par_grid_policy() {
        for &(m, n, k, threads) in &[
            (1usize, 3072usize, 768usize, 16usize),
            (1, 7, 768, 16),
            (8, 4096, 4096, 16),
            (512, 3072, 768, 16),
            (37, 41, 64, 4),
            (2048, 50257, 768, 22),
        ] {
            let (rb, cb) = i8_par_grid(m, n, k, threads);
            assert!(
                (2..=32).contains(&rb) && rb.is_multiple_of(2),
                "rb {rb} (m{m} n{n} k{k} t{threads})"
            );
            assert!(
                (4..=i8_col_block(k)).contains(&cb) && cb.is_multiple_of(4),
                "cb {cb} (m{m} n{n} k{k} t{threads})"
            );
            let nrb = m.div_ceil(rb);
            let tasks = nrb * n.div_ceil(cb);
            let possible = nrb * n.div_ceil(4);
            assert!(
                tasks >= (3 * threads).min(possible),
                "only {tasks} tasks (m{m} n{n} k{k} t{threads}, {possible} possible)"
            );
        }
        // The single-token decode GEMM now spreads over the pool.
        let (rb, cb) = i8_par_grid(1, 3072, 768, 16);
        assert_eq!(rb, 2);
        assert!(
            3072usize.div_ceil(cb) >= 48,
            "decode grid has too few tasks"
        );
    }

    #[test]
    fn overflow_wraps_consistently() {
        // Max-magnitude inputs over a large K overflow i32; scalar and AVX2 must wrap identically
        // (both are sums mod 2³²), proving the gate stays exact even past i32 range.
        let k = 100_000usize;
        let a = vec![255u8; k];
        let b = vec![127i8; k];
        // SAFETY: `a`/`b` hold k values; dot_i8_scalar reads k.
        let s = unsafe { dot_i8_scalar(a.as_ptr(), b.as_ptr(), k) };
        #[cfg(target_arch = "x86_64")]
        if crate::isa::avx2() {
            // SAFETY: AVX2 confirmed by the enclosing feature guard; `a`/`b` hold k values.
            let v = unsafe { dot_i8_avx2(a.as_ptr(), b.as_ptr(), k) };
            assert_eq!(s, v, "scalar != avx2 under overflow");
        }
        // The true sum is 255*127*k = 4_064_250_000, which wraps in i32.
        let want = (255i64 * 127 * k as i64) as i32;
        assert_eq!(s, want, "overflow wrap mismatch");
    }
}
