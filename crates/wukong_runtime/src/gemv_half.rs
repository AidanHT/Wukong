//! Half-precision-weight GEMV and the decode-shaped half GEMM: `C[M, N] = A[M, K] · Bᵀ` with `B`
//! stored as `bf16`/`f16` bits `[N, K]` (the `nn.Linear` weight layout), `A` and `C` in `f32`, for
//! `M <= SMALL_M`.
//!
//! # Why this exists
//!
//! The half GEMM entries (`wukong_sgemm_{bf16,f16}_nt`) widen *both* operands into f32 scratch and
//! run the f32 GEMM — the right trade for a prefill, where every weight is reused `M` times. A
//! decode step is `M = 1`: every weight is read exactly once, the product is memory-bound, and the
//! prepass reads 2 bytes per weight, writes 4 and reads the 4 back, turning the half storage into a
//! net loss. Here the weights stay in their 2-byte storage: eight weight rows are loaded 8 elements
//! at a time, transposed in registers (`punpck{l,h}{wd,dq,qdq}`), widened (`vpmovzxwd`+`vpslld` for
//! bf16, F16C `vcvtph2ps` for f16) and fed to one FMA per row of `A` — llama.cpp's
//! `ggml_vec_dot_f16` idea, with the output columns rather than K in the lanes.
//!
//! # Numeric contract
//!
//! Every output is computed by **the f32 GEMM's own per-element contract** (`gemm.rs` module docs):
//! K is split by `select_kc` into blocks, each block accumulates ascending `p` from `+0.0` with one
//! fused multiply-add per step, and is folded into `C` with one further rounding (the caller's
//! `beta` on the first block, accumulate after), the epilogue applied once on the last block. Lane
//! `l` of a column group is output `j0 + l` and runs exactly that chain, so the AVX2 group, the
//! scalar twin [`elem_scalar`] and the f32 GEMM on the losslessly widened operands agree **bit for
//! bit** — which is what lets `gemm_lowp_nt` route a small `M` here without moving a single bit of
//! any program's output, and what makes `wukong_sgemv_bf16(w, x, y)` exactly the `M = 1` row of
//! `wukong_sgemm_bf16_nt` with an f32 `x`. Columns are independent, so the `_parallel` entries, which
//! split column groups across cores, are bit-identical to the serial ones.
//!
//! # The checked entries
//!
//! `wukong_sgemv_{bf16,f16}[_parallel](w, w_len, x, x_len, y, y_len) -> i64` are the `gemv_bf16` /
//! `gemv_f16` builtins' runtime: `N = y_len`, `K = x_len`, and `w_len` must be exactly `N·K`
//! ([`gemv_half_plan`], shared with the interpreter so both backends return the same status). An
//! empty contraction (`K = 0`) writes zeros. An output overlapping `w` or `x` is computed into
//! scratch and copied out, so the call has value semantics — the answer the interpreter's
//! copy-in/copy-out gives.

#![allow(clippy::needless_range_loop)]

use crate::gemm::{select_kc, Epilogue, HalfKind};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The largest `M` the half NT GEMM routes here instead of widening `B` (one to four tokens).
pub(crate) const SMALL_M: usize = 4;

/// Output columns per vector group.
const COLS: usize = 8;

/// Checked-entry status: success.
pub const GEMV_OK: i64 = 0;
/// Checked-entry status: an operand's length disagrees with the shape the others imply (or is
/// negative). Nothing was read or written.
pub const GEMV_BAD_EXTENT: i64 = -2;

/// An epilogue that may cross into the pool's workers: its bias pointer is only read, and the
/// caller keeps it valid for the whole call.
#[derive(Clone, Copy)]
struct SendEpi(Option<Epilogue>);

// SAFETY: see the type's docs — the wrapped pointer is read-only and outlives the fork-join.
unsafe impl Send for SendEpi {}
// SAFETY: as above.
unsafe impl Sync for SendEpi {}

impl SendEpi {
    fn get(&self) -> Option<Epilogue> {
        self.0
    }
}

/// `c[i, j]` for one output by the f32 GEMM's per-element contract: the portable twin of the AVX2
/// column group, its `N % 8` tail, and the whole kernel where AVX2 is absent. `c_old` is read only
/// when `beta != 0`.
///
/// # Safety
/// `arow` valid for `k` f32 reads, `brow` for `k` u16 reads, `k >= 1`; a non-null epilogue bias is
/// valid at index `j`.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn elem_scalar(
    arow: *const f32,
    brow: *const u16,
    k: usize,
    kind: HalfKind,
    c_old: f32,
    beta: f32,
    epi: Option<Epilogue>,
    j: usize,
) -> f32 {
    let kcap = select_kc(k);
    let mut cval = c_old;
    let mut pc = 0;
    while pc < k {
        let pend = pc + (k - pc).min(kcap);
        let mut acc = 0.0f32;
        for p in pc..pend {
            acc = (*arow.add(p)).mul_add(kind.widen1(*brow.add(p)), acc);
        }
        let beta_eff = if pc == 0 { beta } else { 1.0 };
        let val = if beta_eff == 0.0 { acc } else { cval + acc };
        cval = match epi {
            Some(e) if pend == k => e.apply(val, j),
            _ => val,
        };
        pc = pend;
    }
    cval
}

/// Transpose an 8×8 block of 16-bit elements: `r[row]` holds 8 consecutive K elements of one weight
/// row; lane `l` of result `t` is row `l`'s element `t`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transpose8x8_u16(r: [__m128i; 8]) -> [__m128i; 8] {
    let t0 = _mm_unpacklo_epi16(r[0], r[1]);
    let t1 = _mm_unpackhi_epi16(r[0], r[1]);
    let t2 = _mm_unpacklo_epi16(r[2], r[3]);
    let t3 = _mm_unpackhi_epi16(r[2], r[3]);
    let t4 = _mm_unpacklo_epi16(r[4], r[5]);
    let t5 = _mm_unpackhi_epi16(r[4], r[5]);
    let t6 = _mm_unpacklo_epi16(r[6], r[7]);
    let t7 = _mm_unpackhi_epi16(r[6], r[7]);
    let u0 = _mm_unpacklo_epi32(t0, t2);
    let u1 = _mm_unpackhi_epi32(t0, t2);
    let u2 = _mm_unpacklo_epi32(t1, t3);
    let u3 = _mm_unpackhi_epi32(t1, t3);
    let u4 = _mm_unpacklo_epi32(t4, t6);
    let u5 = _mm_unpackhi_epi32(t4, t6);
    let u6 = _mm_unpacklo_epi32(t5, t7);
    let u7 = _mm_unpackhi_epi32(t5, t7);
    [
        _mm_unpacklo_epi64(u0, u4),
        _mm_unpackhi_epi64(u0, u4),
        _mm_unpacklo_epi64(u1, u5),
        _mm_unpackhi_epi64(u1, u5),
        _mm_unpacklo_epi64(u2, u6),
        _mm_unpackhi_epi64(u2, u6),
        _mm_unpacklo_epi64(u3, u7),
        _mm_unpackhi_epi64(u3, u7),
    ]
}

/// Widen eight half values held in a register — the in-register form of `lowp::widen_bf16` /
/// `lowp::widen_f16`, the same instructions, so the same bits as [`HalfKind::widen1`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
unsafe fn widen8(v: __m128i, kind: HalfKind) -> __m256 {
    match kind {
        HalfKind::Bf16 => _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16)),
        HalfKind::F16 => _mm256_cvtph_ps(v),
    }
}

/// Eight output columns `j0..j0+8` of `R` rows (`R <= SMALL_M`): one FMA chain per lane, the weight
/// rows transposed and widened in registers.
///
/// # Safety
/// `a` valid for `R` rows of `k` f32 (row stride `k`); `b` for 8 rows of `k` u16 (row stride `k`)
/// starting at weight row `j0`; `c` for `R` rows of 8 f32 at row stride `ldc`; `k >= 1`; AVX2, FMA
/// and F16C available; a non-null epilogue bias valid for columns `j0..j0+8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn group8<const R: usize>(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    k: usize,
    ldc: usize,
    kind: HalfKind,
    beta: f32,
    epi: Option<Epilogue>,
    j0: usize,
) {
    let kcap = select_kc(k);
    let mut cval = [[0.0f32; COLS]; R];
    if beta != 0.0 {
        for r in 0..R {
            for l in 0..COLS {
                cval[r][l] = *c.add(r * ldc + l);
            }
        }
    }
    let mut pc = 0;
    while pc < k {
        let pend = pc + (k - pc).min(kcap);
        let mut acc = [_mm256_setzero_ps(); R];
        let mut p = pc;
        while p + 8 <= pend {
            let rows = [
                _mm_loadu_si128(b.add(p) as *const __m128i),
                _mm_loadu_si128(b.add(k + p) as *const __m128i),
                _mm_loadu_si128(b.add(2 * k + p) as *const __m128i),
                _mm_loadu_si128(b.add(3 * k + p) as *const __m128i),
                _mm_loadu_si128(b.add(4 * k + p) as *const __m128i),
                _mm_loadu_si128(b.add(5 * k + p) as *const __m128i),
                _mm_loadu_si128(b.add(6 * k + p) as *const __m128i),
                _mm_loadu_si128(b.add(7 * k + p) as *const __m128i),
            ];
            let cols = transpose8x8_u16(rows);
            for t in 0..8 {
                let w = widen8(cols[t], kind);
                for r in 0..R {
                    let av = _mm256_set1_ps(*a.add(r * k + p + t));
                    acc[r] = _mm256_fmadd_ps(av, w, acc[r]);
                }
            }
            p += 8;
        }
        while p < pend {
            let mut lane = [0.0f32; COLS];
            for l in 0..COLS {
                lane[l] = kind.widen1(*b.add(l * k + p));
            }
            let w = _mm256_loadu_ps(lane.as_ptr());
            for r in 0..R {
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(*a.add(r * k + p)), w, acc[r]);
            }
            p += 1;
        }
        let beta_eff = if pc == 0 { beta } else { 1.0 };
        for r in 0..R {
            let mut sums = [0.0f32; COLS];
            _mm256_storeu_ps(sums.as_mut_ptr(), acc[r]);
            for l in 0..COLS {
                let val = if beta_eff == 0.0 {
                    sums[l]
                } else {
                    cval[r][l] + sums[l]
                };
                cval[r][l] = match epi {
                    Some(e) if pend == k => e.apply(val, j0 + l),
                    _ => val,
                };
            }
        }
        pc = pend;
    }
    for r in 0..R {
        for l in 0..COLS {
            *c.add(r * ldc + l) = cval[r][l];
        }
    }
}

/// The vector column groups of `rows` rows over `[j_lo, j_hi)`, returning the first column it did
/// not compute (the start of the scalar tail). On a host without AVX2/FMA/F16C it computes nothing.
///
/// # Safety
/// As [`group8`] for every group in the span: `a` holds `rows` rows of `k` f32, `b` the weight rows
/// `j_lo..j_hi` (row stride `k`), `c` `rows` output rows at stride `ldc`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn vector_groups(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    rows: usize,
    k: usize,
    ldc: usize,
    span: (usize, usize),
    kind: HalfKind,
    beta: f32,
    epi: Option<Epilogue>,
) -> usize {
    let (mut j, j_hi) = span;
    if !(crate::isa::avx2() && crate::isa::fma() && crate::isa::f16c()) {
        return j;
    }
    while j + COLS <= j_hi {
        let bj = b.add(j * k);
        let cj = c.add(j);
        match rows {
            1 => group8::<1>(a, bj, cj, k, ldc, kind, beta, epi, j),
            2 => group8::<2>(a, bj, cj, k, ldc, kind, beta, epi, j),
            3 => group8::<3>(a, bj, cj, k, ldc, kind, beta, epi, j),
            _ => group8::<4>(a, bj, cj, k, ldc, kind, beta, epi, j),
        }
        j += COLS;
    }
    j
}

/// The portable build: no vector groups.
///
/// # Safety
/// None required; nothing is read or written.
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn vector_groups(
    _a: *const f32,
    _b: *const u16,
    _c: *mut f32,
    _rows: usize,
    _k: usize,
    _ldc: usize,
    span: (usize, usize),
    _kind: HalfKind,
    _beta: f32,
    _epi: Option<Epilogue>,
) -> usize {
    span.0
}

/// Output columns `[j_lo, j_hi)` of every row: rows four at a time, vector column groups, then the
/// scalar twin for the `N % 8` tail.
///
/// # Safety
/// `a` valid for `m*k` f32, `b` for `n*k` u16, `c` for `m*n` f32 (row-major); `k >= 1`;
/// `j_hi <= n`; a non-null epilogue bias valid for `n` f32.
#[allow(clippy::too_many_arguments)]
unsafe fn small_m_span(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    span: (usize, usize),
    kind: HalfKind,
    beta: f32,
    epi: Option<Epilogue>,
) {
    let mut i0 = 0;
    while i0 < m {
        let rows = (m - i0).min(SMALL_M);
        let ai = a.add(i0 * k);
        let ci = c.add(i0 * n);
        let tail = vector_groups(ai, b, ci, rows, k, n, span, kind, beta, epi);
        for j in tail..span.1 {
            for r in 0..rows {
                let cij = ci.add(r * n + j);
                let c_old = if beta != 0.0 { *cij } else { 0.0 };
                *cij = elem_scalar(ai.add(r * k), b.add(j * k), k, kind, c_old, beta, epi, j);
            }
        }
        i0 += rows;
    }
}

/// `C[M, N] (beta-accumulate) = A[M, K] · Bᵀ` with half `B[N, K]`, every element by the f32 GEMM's
/// order (see the module docs). Rows are processed four at a time, so `B` streams from memory once
/// per four rows — once in total at the decode shapes this is routed for. `par` splits the column
/// groups across the kernel pool (bit-identical: columns are independent).
///
/// # Safety
/// `a` valid for `m*k` f32, `b` for `n*k` u16, `c` for `m*n` f32, row-major; `c` overlaps neither
/// `a` nor `b`; a non-null epilogue bias valid for `n` f32 and disjoint from `c`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn half_nt_small_m(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
    kind: HalfKind,
    epi: Option<Epilogue>,
    par: bool,
) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    let groups = n.div_ceil(COLS);
    let work = m.saturating_mul(n).saturating_mul(k);
    if par && crate::par::fork_units(groups, work) {
        let (a_addr, b_addr, c_addr) = (a as usize, b as usize, c as usize);
        let epi = SendEpi(epi);
        crate::par::claim_chunks(groups, 1, true, move |g0, g1| {
            let span = (g0 * COLS, (g1 * COLS).min(n));
            // SAFETY: every span owns the disjoint output columns `span` of every row; the operand
            // contract is this function's own.
            unsafe {
                small_m_span(
                    a_addr as *const f32,
                    b_addr as *const u16,
                    c_addr as *mut f32,
                    m,
                    k,
                    n,
                    span,
                    kind,
                    beta,
                    epi.get(),
                );
            }
        });
        return;
    }
    small_m_span(a, b, c, m, k, n, (0, n), kind, beta, epi);
}

/// Validate a checked half-GEMV call: `N = y_len`, `K = x_len`, and the weight must hold exactly
/// `N·K` elements. Returns `(n, k)`, or [`GEMV_BAD_EXTENT`]. Shared with the interpreter.
pub fn gemv_half_plan(w_len: i64, x_len: i64, y_len: i64) -> Result<(usize, usize), i64> {
    if w_len < 0 || x_len < 0 || y_len < 0 {
        return Err(GEMV_BAD_EXTENT);
    }
    let (n, k) = (y_len as u64, x_len as u64);
    match n.checked_mul(k) {
        Some(need) if need == w_len as u64 => Ok((n as usize, k as usize)),
        _ => Err(GEMV_BAD_EXTENT),
    }
}

/// Whether the byte ranges `[a, a + a_len)` and `[b, b + b_len)` share a byte (empty ranges overlap
/// nothing).
pub(crate) fn bytes_overlap(a: usize, a_len: usize, b: usize, b_len: usize) -> bool {
    a_len != 0 && b_len != 0 && a < b.saturating_add(b_len) && b < a.saturating_add(a_len)
}

/// The shared body of the four checked half-GEMV entries.
///
/// # Safety
/// Each pointer addresses its length of elements (`w`: `w_len` u16, `x`: `x_len` f32, `y`: `y_len`
/// f32); a pointer whose length is 0 is never dereferenced (it may be null).
#[allow(clippy::too_many_arguments)]
unsafe fn sgemv_half_entry(
    w: *const u16,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
    kind: HalfKind,
    par: bool,
) -> i64 {
    let (n, k) = match gemv_half_plan(w_len, x_len, y_len) {
        Ok(nk) => nk,
        Err(code) => return code,
    };
    if n == 0 {
        return GEMV_OK;
    }
    if k == 0 {
        // The empty contraction: every output is the empty sum.
        std::slice::from_raw_parts_mut(y, n).fill(0.0);
        return GEMV_OK;
    }
    let y_bytes = n.saturating_mul(4);
    let hazard = bytes_overlap(y as usize, y_bytes, w as usize, (n * k).saturating_mul(2))
        || bytes_overlap(y as usize, y_bytes, x as usize, k.saturating_mul(4));
    if hazard {
        let mut tmp = vec![0.0f32; n];
        half_nt_small_m(x, w, tmp.as_mut_ptr(), 1, k, n, 0.0, kind, None, par);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), y, n);
    } else {
        half_nt_small_m(x, w, y, 1, k, n, 0.0, kind, None, par);
    }
    GEMV_OK
}

/// `y[j] = Σ_p x[p]·w[j, p]` with `bf16` weights `w[N, K]` (stored bits), `f32` `x[K]` and `y[N]` —
/// the `gemv_bf16(w, x, y)` builtin. Returns [`GEMV_OK`] or [`GEMV_BAD_EXTENT`] (nothing touched).
///
/// # Safety
/// `w` addresses `w_len` u16, `x` `x_len` f32 and `y` `y_len` writable f32 (a zero-length operand may
/// be null). Any overlap of `y` with an input is handled (value semantics).
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_bf16(
    w: *const u16,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an overlapping `y` is handled (value semantics).
    crate::safety::debug_extent(w, w_len);
    crate::safety::debug_extent(x, x_len);
    crate::safety::debug_extent(y, y_len);
    sgemv_half_entry(w, w_len, x, x_len, y, y_len, HalfKind::Bf16, false)
}

/// Multi-threaded [`wukong_sgemv_bf16`]: column groups across cores, bit-identical to serial.
///
/// # Safety
/// The contract of [`wukong_sgemv_bf16`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_bf16_parallel(
    w: *const u16,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an overlapping `y` is handled (value semantics).
    crate::safety::debug_extent(w, w_len);
    crate::safety::debug_extent(x, x_len);
    crate::safety::debug_extent(y, y_len);
    sgemv_half_entry(w, w_len, x, x_len, y, y_len, HalfKind::Bf16, true)
}

/// `y[j] = Σ_p x[p]·w[j, p]` with IEEE `f16` weights — the `gemv_f16(w, x, y)` builtin. See
/// [`wukong_sgemv_bf16`].
///
/// # Safety
/// The contract of [`wukong_sgemv_bf16`], with `f16` weight bits.
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_f16(
    w: *const u16,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an overlapping `y` is handled (value semantics).
    crate::safety::debug_extent(w, w_len);
    crate::safety::debug_extent(x, x_len);
    crate::safety::debug_extent(y, y_len);
    sgemv_half_entry(w, w_len, x, x_len, y, y_len, HalfKind::F16, false)
}

/// Multi-threaded [`wukong_sgemv_f16`]: column groups across cores, bit-identical to serial.
///
/// # Safety
/// The contract of [`wukong_sgemv_f16`].
#[no_mangle]
pub unsafe extern "C" fn wukong_sgemv_f16_parallel(
    w: *const u16,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); operand lengths are
    // explicit; an overlapping `y` is handled (value semantics).
    crate::safety::debug_extent(w, w_len);
    crate::safety::debug_extent(x, x_len);
    crate::safety::debug_extent(y, y_len);
    sgemv_half_entry(w, w_len, x, x_len, y, y_len, HalfKind::F16, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    fn to_half(xs: &[f32], kind: HalfKind) -> Vec<u16> {
        xs.iter()
            .map(|&x| match kind {
                HalfKind::Bf16 => crate::f32_to_bf16_bits(x),
                HalfKind::F16 => crate::f32_to_f16_bits(x),
            })
            .collect()
    }

    fn widen_all(bits: &[u16], kind: HalfKind) -> Vec<f32> {
        bits.iter().map(|&b| kind.widen1(b)).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Shapes straddling the 8-column groups, the 8-wide K step, the `select_kc` K-block boundary
    /// (`k = 1000` splits into three blocks) and the four-row grouping.
    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 7, 3),
        (1, 8, 8),
        (1, 13, 11),
        (2, 33, 17),
        (3, 64, 24),
        (4, 100, 9),
        (1, 1000, 19),
        (3, 777, 40),
    ];

    /// The routed small-M path equals the f32 NT GEMM on the widened operands **bit for bit**, for
    /// both half formats, `beta` 0 and 1, and serial and parallel — the property that lets
    /// `gemm_lowp_nt` route here without changing any output.
    #[test]
    fn small_m_equals_the_f32_gemm_on_widened_operands() {
        for &(m, k, n) in SHAPES {
            for kind in [HalfKind::Bf16, HalfKind::F16] {
                let a = fill(1 + m as u64, m * k);
                let bh = to_half(&fill(2 + k as u64, n * k), kind);
                let bw = widen_all(&bh, kind);
                let c0 = fill(3, m * n);
                for beta in [0i64, 1] {
                    let mut want = c0.clone();
                    // SAFETY: every buffer is exactly the size its role needs.
                    unsafe {
                        crate::wukong_sgemm_nt(
                            a.as_ptr(),
                            bw.as_ptr(),
                            want.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            beta,
                        );
                    }
                    for par in [false, true] {
                        let mut got = c0.clone();
                        // SAFETY: as above; `got` overlaps neither input.
                        unsafe {
                            half_nt_small_m(
                                a.as_ptr(),
                                bh.as_ptr(),
                                got.as_mut_ptr(),
                                m,
                                k,
                                n,
                                beta as f32,
                                kind,
                                None,
                                par,
                            );
                        }
                        assert_eq!(
                            bits(&got),
                            bits(&want),
                            "{kind:?} m={m} k={k} n={n} beta={beta} par={par}"
                        );
                    }
                }
            }
        }
    }

    /// The scalar twin alone (the whole kernel on a host without AVX2) equals the vector groups bit
    /// for bit: drive [`elem_scalar`] over every output and compare with the dispatched span.
    #[test]
    fn scalar_twin_matches_the_vector_groups_bit_for_bit() {
        for &(m, k, n) in SHAPES {
            for kind in [HalfKind::Bf16, HalfKind::F16] {
                let a = fill(7, m * k);
                let bh = to_half(&fill(8, n * k), kind);
                let mut got = vec![0.0f32; m * n];
                // SAFETY: buffers sized to the shape.
                unsafe {
                    half_nt_small_m(
                        a.as_ptr(),
                        bh.as_ptr(),
                        got.as_mut_ptr(),
                        m,
                        k,
                        n,
                        0.0,
                        kind,
                        None,
                        false,
                    );
                }
                for i in 0..m {
                    for j in 0..n {
                        // SAFETY: row `i` of `a` and row `j` of `bh` are `k` long.
                        let want = unsafe {
                            elem_scalar(
                                a.as_ptr().add(i * k),
                                bh.as_ptr().add(j * k),
                                k,
                                kind,
                                0.0,
                                0.0,
                                None,
                                j,
                            )
                        };
                        assert_eq!(
                            got[i * n + j].to_bits(),
                            want.to_bits(),
                            "{kind:?} ({m}x{k}x{n}) [{i}][{j}]"
                        );
                    }
                }
            }
        }
    }

    /// The routed path through the public half GEMM entries (`m <= 4`) still equals the f32 GEMM on
    /// widened operands, with the fused epilogue too — the `gemm_lowp_nt` dispatch itself.
    #[test]
    fn the_lowp_entries_route_small_m_without_moving_a_bit() {
        for &(m, k, n) in SHAPES {
            let a = fill(11, m * k);
            let b = fill(12, n * k);
            let bias = fill(13, n);
            let ah = to_half(&a, HalfKind::Bf16);
            let bh = to_half(&b, HalfKind::Bf16);
            let aw = widen_all(&ah, HalfKind::Bf16);
            let bw = widen_all(&bh, HalfKind::Bf16);
            let mut want = vec![0.0f32; m * n];
            let mut got = vec![0.0f32; m * n];
            let mut want_epi = vec![0.0f32; m * n];
            let mut got_epi = vec![0.0f32; m * n];
            // SAFETY: buffers sized to the shape; outputs disjoint from inputs.
            unsafe {
                crate::wukong_sgemm_nt(
                    aw.as_ptr(),
                    bw.as_ptr(),
                    want.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                crate::wukong_sgemm_bf16_nt(
                    ah.as_ptr(),
                    bh.as_ptr(),
                    got.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                );
                // act = 3 (SiLU) with a bias: the epilogue applies on the last K-block only.
                crate::wukong_sgemm_nt_epi(
                    aw.as_ptr(),
                    bw.as_ptr(),
                    want_epi.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    bias.as_ptr(),
                    3,
                );
                crate::wukong_sgemm_bf16_nt_epi_parallel(
                    ah.as_ptr(),
                    bh.as_ptr(),
                    got_epi.as_mut_ptr(),
                    m as i64,
                    k as i64,
                    n as i64,
                    0,
                    bias.as_ptr(),
                    3,
                );
            }
            assert_eq!(bits(&got), bits(&want), "bf16 nt ({m}x{k}x{n})");
            assert_eq!(bits(&got_epi), bits(&want_epi), "bf16 nt_epi ({m}x{k}x{n})");
        }
    }

    /// The ABI of the four checked half-GEMV entries.
    type Entry = unsafe extern "C" fn(*const u16, i64, *const f32, i64, *mut f32, i64) -> i64;

    /// The checked GEMV entries: the `M = 1` row of the f32 GEMM on the widened weight, serial ==
    /// parallel, a mismatched weight length refused with nothing written, `K = 0` writes zeros, and
    /// an output aliasing the activation still sees the activation as it was before the call.
    #[test]
    fn checked_gemv_entries() {
        let entries: [(HalfKind, Entry, Entry); 2] = [
            (
                HalfKind::Bf16,
                wukong_sgemv_bf16,
                wukong_sgemv_bf16_parallel,
            ),
            (HalfKind::F16, wukong_sgemv_f16, wukong_sgemv_f16_parallel),
        ];
        for &(_, k, n) in SHAPES {
            for (kind, serial, parallel) in entries {
                let x = fill(21, k);
                let wh = to_half(&fill(22, n * k), kind);
                let ww = widen_all(&wh, kind);
                let mut want = vec![0.0f32; n];
                let mut got = vec![0.0f32; n];
                let mut got_par = vec![0.0f32; n];
                // SAFETY: buffers sized to the shape.
                unsafe {
                    crate::wukong_sgemm_nt(
                        x.as_ptr(),
                        ww.as_ptr(),
                        want.as_mut_ptr(),
                        1,
                        k as i64,
                        n as i64,
                        0,
                    );
                    let (wl, xl, yl) = ((n * k) as i64, k as i64, n as i64);
                    assert_eq!(
                        serial(wh.as_ptr(), wl, x.as_ptr(), xl, got.as_mut_ptr(), yl),
                        GEMV_OK
                    );
                    assert_eq!(
                        parallel(wh.as_ptr(), wl, x.as_ptr(), xl, got_par.as_mut_ptr(), yl),
                        GEMV_OK
                    );
                }
                assert_eq!(bits(&got), bits(&want), "{kind:?} gemv k={k} n={n}");
                assert_eq!(bits(&got_par), bits(&want), "{kind:?} gemv par k={k} n={n}");

                // A weight one element short is refused before anything is written.
                let mut untouched = vec![7.0f32; n];
                // SAFETY: the entry reads nothing on a refused extent.
                let code = unsafe {
                    serial(
                        wh.as_ptr(),
                        (n * k) as i64 - 1,
                        x.as_ptr(),
                        k as i64,
                        untouched.as_mut_ptr(),
                        n as i64,
                    )
                };
                assert_eq!(code, GEMV_BAD_EXTENT);
                assert!(untouched.iter().all(|&v| v == 7.0));
            }
        }
        // K = 0: the empty sum, even through a null weight and activation.
        let mut y = [5.0f32; 3];
        // SAFETY: zero-length operands are never dereferenced.
        let code = unsafe {
            wukong_sgemv_bf16(std::ptr::null(), 0, std::ptr::null(), 0, y.as_mut_ptr(), 3)
        };
        assert_eq!(code, GEMV_OK);
        assert_eq!(y, [0.0; 3]);
        // y aliasing x (n == k): the result is computed from x as it was before the call.
        let k = 12;
        let x = fill(31, k);
        let wh = to_half(&fill(32, k * k), HalfKind::Bf16);
        let mut want = vec![0.0f32; k];
        let mut buf = x.clone();
        // SAFETY: sized buffers; the aliased call relies on the entry's overlap handling.
        unsafe {
            wukong_sgemv_bf16(
                wh.as_ptr(),
                (k * k) as i64,
                x.as_ptr(),
                k as i64,
                want.as_mut_ptr(),
                k as i64,
            );
            let p = buf.as_mut_ptr();
            wukong_sgemv_bf16(wh.as_ptr(), (k * k) as i64, p, k as i64, p, k as i64);
        }
        assert_eq!(bits(&buf), bits(&want));
    }

    #[test]
    fn plan_rejects_every_malformed_extent() {
        assert_eq!(gemv_half_plan(12, 4, 3), Ok((3, 4)));
        assert_eq!(gemv_half_plan(0, 0, 3), Ok((3, 0)));
        assert_eq!(gemv_half_plan(0, 5, 0), Ok((0, 5)));
        assert_eq!(gemv_half_plan(11, 4, 3), Err(GEMV_BAD_EXTENT));
        assert_eq!(gemv_half_plan(12, -4, -3), Err(GEMV_BAD_EXTENT));
        assert_eq!(gemv_half_plan(-1, 0, 0), Err(GEMV_BAD_EXTENT));
        assert_eq!(gemv_half_plan(0, i64::MAX, i64::MAX), Err(GEMV_BAD_EXTENT));
        assert!(
            bytes_overlap(0, 8, 4, 8) && !bytes_overlap(0, 8, 8, 8) && !bytes_overlap(4, 0, 0, 8)
        );
    }
}
