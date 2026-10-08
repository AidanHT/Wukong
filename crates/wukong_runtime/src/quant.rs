//! Block-quantized weights: the GGML `Q8_0` / `Q4_0` formats, their packers, and the weight-only
//! quantized GEMV of a decode step — the runtime of the `quantize_q8_0` / `quantize_q4_0` and
//! `gemv_q8_0` / `gemv_q4_0` builtins.
//!
//! # Formats (byte-compatible with llama.cpp's `block_q8_0` / `block_q4_0`)
//!
//! A row of `K` values is cut into `ceil(K / 32)` blocks of [`QK`] = 32; a short last block is
//! padded with zeros. Each block is a little-endian IEEE `f16` scale `d` followed by its quants:
//!
//! * `Q8_0` ([`Q8_0_BLOCK_BYTES`] = 34): 32 signed bytes `q`; the value is `d · q`. The packer takes
//!   `d = amax / 127` and `q = round(x / d)` (half away from zero), llama.cpp's
//!   `quantize_row_q8_0_ref`.
//! * `Q4_0` ([`Q4_0_BLOCK_BYTES`] = 18): 16 bytes; byte `j` holds element `j` in its low nibble and
//!   element `j + 16` in its high nibble, and the value is `d · (nibble − 8)`. The packer takes the
//!   element of largest magnitude `max` (the first, on a tie), `d = max / −8` and
//!   `nibble = min(15, trunc(x / d + 8.5))`, llama.cpp's `quantize_row_q4_0_ref`.
//!
//! The packers ignore a NaN when they pick the scale (`amax` folds `|x| > amax`), and a NaN element
//! quantizes to `0` (Q8_0) / nibble `8` (Q4_0). A matrix `[N, K]` is `N` rows of blocks, row-major.
//!
//! # The GEMV
//!
//! `y[j] = Σ_k W[j, k] · x[k]` for a quantized `W[N, K]` and an f32 activation `x[K]`, computed the
//! way llama.cpp's `ggml_vec_dot_q4_0_q8_0` / `ggml_vec_dot_q8_0_q8_0` do: `x` is quantized per call
//! to Q8_0 blocks (the same rule as the packer, except that its scale `dx` stays an exact `f32`
//! rather than being rounded to `f16`), each block's integer dot `isum = Σ w_q · x_q` is exact in
//! `i32`, and the row accumulates **in block order**:
//!
//! ```text
//! acc = +0.0;  for b in blocks: acc = acc + (dw_b · dx_b) · f32(isum_b)
//! ```
//!
//! with every product and sum rounded once (no fused multiply-add). The AVX2 path computes the
//! integer dots with `vpsignb` + `vpmaddubsw` + `vpmaddwd` (exact: the activation quants are in
//! `[-127, 127]`, so no pair sum reaches the `i16` saturation bound, whatever the weight bytes) and
//! runs the float recurrence eight rows to a vector, one row per lane — so it agrees with the scalar
//! twin **bit for bit**, and the `_parallel` entries (rows split across cores) with the serial ones.
//!
//! # The checked entries
//!
//! Every entry takes each buffer with its length and validates the whole shape first
//! ([`quantize_plan`], [`gemv_q_plan`] — shared with the interpreter, so both backends return the
//! same status): `N = len(y)`, `K = len(x)`, and the weight must be exactly `N · ceil(K/32)` blocks.
//! An empty contraction (`K = 0`) writes zeros. An output overlapping an input is computed so that
//! the inputs are read as they were before the call (value semantics).

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::gemv_half::bytes_overlap;

/// Elements per quantization block.
pub const QK: usize = 32;
/// Bytes per `Q8_0` block: an `f16` scale and 32 signed quants.
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Bytes per `Q4_0` block: an `f16` scale and 16 bytes of nibble pairs.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Status: success.
pub const QUANT_OK: i64 = 0;
/// Status: a negative length, a non-positive row length, or an operand whose length disagrees with
/// the shape the others imply. Nothing was read or written.
pub const QUANT_BAD_EXTENT: i64 = -2;

/// A block-quantized format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QFormat {
    /// GGML `Q8_0`.
    Q8,
    /// GGML `Q4_0`.
    Q4,
}

impl QFormat {
    /// Bytes per block of this format.
    pub fn block_bytes(self) -> usize {
        match self {
            QFormat::Q8 => Q8_0_BLOCK_BYTES,
            QFormat::Q4 => Q4_0_BLOCK_BYTES,
        }
    }

    /// Bytes of one quantized row of `k` values.
    pub fn row_bytes(self, k: usize) -> usize {
        k.div_ceil(QK) * self.block_bytes()
    }
}

/// Validate a packer call: `x` holds whole rows of `k` values, and `q` exactly their blocks.
/// Returns `(rows, k)`, or [`QUANT_BAD_EXTENT`].
pub fn quantize_plan(x_len: i64, q_len: i64, k: i64, fmt: QFormat) -> Result<(usize, usize), i64> {
    if x_len < 0 || q_len < 0 || k <= 0 {
        return Err(QUANT_BAD_EXTENT);
    }
    let (xl, ku) = (x_len as u64, k as u64);
    if !xl.is_multiple_of(ku) {
        return Err(QUANT_BAD_EXTENT);
    }
    let rows = xl / ku;
    let need = rows
        .checked_mul(ku.div_ceil(QK as u64))
        .and_then(|b| b.checked_mul(fmt.block_bytes() as u64));
    if need != Some(q_len as u64) {
        return Err(QUANT_BAD_EXTENT);
    }
    Ok((rows as usize, ku as usize))
}

/// Validate a quantized GEMV call: `N = y_len`, `K = x_len`, and the weight exactly `N` rows of
/// `ceil(K/32)` blocks. Returns `(n, k)`, or [`QUANT_BAD_EXTENT`].
pub fn gemv_q_plan(
    wq_len: i64,
    x_len: i64,
    y_len: i64,
    fmt: QFormat,
) -> Result<(usize, usize), i64> {
    if wq_len < 0 || x_len < 0 || y_len < 0 {
        return Err(QUANT_BAD_EXTENT);
    }
    let (n, k) = (y_len as u64, x_len as u64);
    let need = n
        .checked_mul(k.div_ceil(QK as u64))
        .and_then(|b| b.checked_mul(fmt.block_bytes() as u64));
    if need != Some(wq_len as u64) {
        return Err(QUANT_BAD_EXTENT);
    }
    Ok((n as usize, k as usize))
}

/// The largest magnitude of a block, ignoring NaN (`|x| > amax` is false for it).
fn block_amax(blk: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    for &v in blk {
        if v.abs() > amax {
            amax = v.abs();
        }
    }
    amax
}

/// `round(v)` (half away from zero) as a Q8 quant in `[-127, 127]`; NaN is `0`.
fn q8_quant(v: f32) -> i8 {
    v.round().clamp(-127.0, 127.0) as i8
}

/// The f16 scale bits of a block.
fn block_scale(blk: &[u8]) -> f32 {
    crate::f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]))
}

/// Quantize one row (`x.len()` values) into `out` (its `ceil(len/32)` Q8_0 blocks).
pub fn quantize_row_q8_0(x: &[f32], out: &mut [u8]) {
    for (b, o) in out.chunks_exact_mut(Q8_0_BLOCK_BYTES).enumerate() {
        let lo = (b * QK).min(x.len());
        let blk = &x[lo..(lo + QK).min(x.len())];
        let d = block_amax(blk) / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o[..2].copy_from_slice(&crate::f32_to_f16_bits(d).to_le_bytes());
        for e in 0..QK {
            let q = blk.get(e).map_or(0, |&v| q8_quant(v * id));
            o[2 + e] = q as u8;
        }
    }
}

/// Quantize one row into its Q4_0 blocks.
pub fn quantize_row_q4_0(x: &[f32], out: &mut [u8]) {
    for (b, o) in out.chunks_exact_mut(Q4_0_BLOCK_BYTES).enumerate() {
        let lo = (b * QK).min(x.len());
        let blk = &x[lo..(lo + QK).min(x.len())];
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &v in blk {
            if amax < v.abs() {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o[..2].copy_from_slice(&crate::f32_to_f16_bits(d).to_le_bytes());
        // A padded element is 0, which quantizes to the zero nibble 8. So does a NaN (and an
        // infinite element, whose `id` is zero): `(NaN + 8.5) as i8` is 0, the most negative
        // nibble, which would silently decode a NaN weight as `-8·d` instead of the documented 0.
        let nib = |e: usize| -> u8 {
            let v = blk.get(e).copied().unwrap_or(0.0) * id;
            if v.is_nan() {
                return 8;
            }
            ((v + 8.5) as i8).min(15) as u8
        };
        for j in 0..QK / 2 {
            o[2 + j] = nib(j) | (nib(j + QK / 2) << 4);
        }
    }
}

/// Quantize an activation vector to Q8_0 quants (`xq`, zero-padded to whole blocks) and exact f32
/// block scales (`dx`, one per block).
fn quantize_activation(x: &[f32], xq: &mut [i8], dx: &mut [f32]) {
    for (b, (q, d_out)) in xq.chunks_exact_mut(QK).zip(dx.iter_mut()).enumerate() {
        let lo = (b * QK).min(x.len());
        let blk = &x[lo..(lo + QK).min(x.len())];
        let d = block_amax(blk) / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        *d_out = d;
        for e in 0..QK {
            q[e] = blk.get(e).map_or(0, |&v| q8_quant(v * id));
        }
    }
}

/// The weight quant `e` (0..32) of a block's quant bytes `qs`.
fn weight_quant(fmt: QFormat, qs: &[u8], e: usize) -> i32 {
    match fmt {
        QFormat::Q8 => i32::from(qs[e] as i8),
        QFormat::Q4 => {
            let byte = qs[e % (QK / 2)];
            let nib = if e < QK / 2 { byte & 0x0F } else { byte >> 4 };
            i32::from(nib) - 8
        }
    }
}

/// One output row by the module's contract — the portable twin of [`rows8_avx2`].
fn row_dot(fmt: QFormat, wrow: &[u8], xq: &[i8], dx: &[f32]) -> f32 {
    let bb = fmt.block_bytes();
    let mut acc = 0.0f32;
    for (b, &dxb) in dx.iter().enumerate() {
        let blk = &wrow[b * bb..(b + 1) * bb];
        let mut isum = 0i32;
        for e in 0..QK {
            isum += weight_quant(fmt, &blk[2..], e) * i32::from(xq[b * QK + e]);
        }
        acc += (block_scale(blk) * dxb) * (isum as f32);
    }
    acc
}

/// The 32 weight quants of a Q4_0 block as signed bytes in element order (low nibbles, then high).
///
/// # Safety
/// `qs` valid for 16 byte reads; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_q4(qs: *const u8) -> __m256i {
    let raw = _mm_loadu_si128(qs as *const __m128i);
    let m = _mm_set1_epi8(0x0F);
    let lo = _mm_and_si128(raw, m);
    let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m);
    _mm256_sub_epi8(_mm256_set_m128i(hi, lo), _mm256_set1_epi8(8))
}

/// Lane `r` = the sum of the eight `i32` lanes of `v[r]` (exact).
///
/// # Safety
/// AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum8x8_epi32(v: [__m256i; 8]) -> __m256i {
    // hadd(a, b) = [a0+a1, a2+a3, b0+b1, b2+b3 | a4+a5, a6+a7, b4+b5, b6+b7] per 128-bit half: two
    // levels leave rows 0-3 / 4-7 split across the halves, and the cross-half add closes them.
    let t0 = _mm256_hadd_epi32(v[0], v[1]);
    let t1 = _mm256_hadd_epi32(v[2], v[3]);
    let t2 = _mm256_hadd_epi32(v[4], v[5]);
    let t3 = _mm256_hadd_epi32(v[6], v[7]);
    let u0 = _mm256_hadd_epi32(t0, t1);
    let u1 = _mm256_hadd_epi32(t2, t3);
    _mm256_add_epi32(
        _mm256_permute2x128_si256(u0, u1, 0x20),
        _mm256_permute2x128_si256(u0, u1, 0x31),
    )
}

/// Eight output rows, one per lane: the exact integer block dots, then the float recurrence of
/// [`row_dot`] lane-wise.
///
/// # Safety
/// `w` valid for 8 rows of `row_bytes` (row stride `row_bytes`), `row_bytes == nb · block_bytes`;
/// `xq` for `nb·32` and `dx` for `nb` reads; `y` for 8 writes; AVX2 and F16C available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows8_avx2(
    w: *const u8,
    row_bytes: usize,
    fmt: QFormat,
    xq: *const i8,
    dx: *const f32,
    nb: usize,
    y: *mut f32,
) {
    let ones = _mm256_set1_epi16(1);
    let bb = fmt.block_bytes();
    let mut acc = _mm256_setzero_ps();
    for b in 0..nb {
        let xv = _mm256_loadu_si256(xq.add(b * QK) as *const __m256i);
        let mut dots = [_mm256_setzero_si256(); 8];
        let mut scales = [0u16; 8];
        for r in 0..8 {
            let blk = w.add(r * row_bytes + b * bb);
            scales[r] = u16::from_le_bytes([*blk, *blk.add(1)]);
            let wv = match fmt {
                QFormat::Q8 => _mm256_loadu_si256(blk.add(2) as *const __m256i),
                QFormat::Q4 => unpack_q4(blk.add(2)),
            };
            // |w| as unsigned bytes and x carrying w's sign: their pairwise products are w·x, and
            // with |x| <= 127 no pair sum reaches the i16 saturation bound (2·128·127 < 2^15).
            let ax = _mm256_sign_epi8(wv, wv);
            let sx = _mm256_sign_epi8(xv, wv);
            dots[r] = _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sx), ones);
        }
        let isum = _mm256_cvtepi32_ps(hsum8x8_epi32(dots));
        let dw = _mm256_cvtph_ps(_mm_loadu_si128(scales.as_ptr() as *const __m128i));
        let p = _mm256_mul_ps(dw, _mm256_set1_ps(*dx.add(b)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(p, isum));
    }
    _mm256_storeu_ps(y, acc);
}

/// Output rows `[lo, hi)`: eight at a time on AVX2, the scalar twin for the rest.
///
/// # Safety
/// `w` valid for rows `lo..hi` of `row_bytes` bytes (row stride `row_bytes`); `y` for those rows'
/// writes; `xq.len() == 32 · dx.len()`, `row_bytes == dx.len() · block_bytes`.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_rows(
    w: *const u8,
    row_bytes: usize,
    fmt: QFormat,
    xq: &[i8],
    dx: &[f32],
    y: *mut f32,
    span: (usize, usize),
) {
    let (mut r, hi) = span;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            while r + 8 <= hi {
                rows8_avx2(
                    w.add(r * row_bytes),
                    row_bytes,
                    fmt,
                    xq.as_ptr(),
                    dx.as_ptr(),
                    dx.len(),
                    y.add(r),
                );
                r += 8;
            }
        }
    }
    while r < hi {
        let wrow = std::slice::from_raw_parts(w.add(r * row_bytes), row_bytes);
        *y.add(r) = row_dot(fmt, wrow, xq, dx);
        r += 1;
    }
}

/// Every output row, `par` splitting groups of eight rows across the kernel pool.
///
/// # Safety
/// As [`gemv_rows`] for all `n` rows.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_all(
    w: *const u8,
    row_bytes: usize,
    fmt: QFormat,
    xq: &[i8],
    dx: &[f32],
    y: *mut f32,
    n: usize,
    par: bool,
) {
    let groups = n.div_ceil(8);
    if par && crate::par::fork_units(groups, n.saturating_mul(row_bytes)) {
        let (w_addr, y_addr) = (w as usize, y as usize);
        crate::par::claim_chunks(groups, 1, true, move |g0, g1| {
            // SAFETY: each span owns the disjoint output rows `8·g0 .. 8·g1`; the operand contract
            // is this function's own.
            unsafe {
                gemv_rows(
                    w_addr as *const u8,
                    row_bytes,
                    fmt,
                    xq,
                    dx,
                    y_addr as *mut f32,
                    (g0 * 8, (g1 * 8).min(n)),
                );
            }
        });
        return;
    }
    gemv_rows(w, row_bytes, fmt, xq, dx, y, (0, n));
}

/// The shared body of the quantized GEMV entries.
///
/// # Safety
/// `w` addresses `w_len` bytes, `x` `x_len` f32 and `y` `y_len` writable f32; a zero-length operand
/// is never dereferenced.
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_q_entry(
    w: *const u8,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
    fmt: QFormat,
    par: bool,
) -> i64 {
    let (n, k) = match gemv_q_plan(w_len, x_len, y_len, fmt) {
        Ok(nk) => nk,
        Err(code) => return code,
    };
    if n == 0 {
        return QUANT_OK;
    }
    if k == 0 {
        std::slice::from_raw_parts_mut(y, n).fill(0.0);
        return QUANT_OK;
    }
    let nb = k.div_ceil(QK);
    let mut xq = vec![0i8; nb * QK];
    let mut dx = vec![0.0f32; nb];
    // The activation is read completely here, before any output is written.
    quantize_activation(std::slice::from_raw_parts(x, k), &mut xq, &mut dx);
    let row_bytes = fmt.row_bytes(k);
    if bytes_overlap(y as usize, n * 4, w as usize, n * row_bytes) {
        let mut tmp = vec![0.0f32; n];
        gemv_all(w, row_bytes, fmt, &xq, &dx, tmp.as_mut_ptr(), n, par);
        std::ptr::copy_nonoverlapping(tmp.as_ptr(), y, n);
    } else {
        gemv_all(w, row_bytes, fmt, &xq, &dx, y, n, par);
    }
    QUANT_OK
}

/// The shared body of the two packers.
///
/// # Safety
/// `x` addresses `x_len` f32 and `q` `q_len` writable bytes; a zero-length operand is never
/// dereferenced.
unsafe fn quantize_entry(
    x: *const f32,
    x_len: i64,
    q: *mut u8,
    q_len: i64,
    k: i64,
    fmt: QFormat,
) -> i64 {
    let (rows, k) = match quantize_plan(x_len, q_len, k, fmt) {
        Ok(rk) => rk,
        Err(code) => return code,
    };
    if rows == 0 {
        return QUANT_OK;
    }
    let row_bytes = fmt.row_bytes(k);
    let src: std::borrow::Cow<[f32]> =
        if bytes_overlap(q as usize, rows * row_bytes, x as usize, rows * k * 4) {
            std::borrow::Cow::Owned(std::slice::from_raw_parts(x, rows * k).to_vec())
        } else {
            std::borrow::Cow::Borrowed(std::slice::from_raw_parts(x, rows * k))
        };
    let out = std::slice::from_raw_parts_mut(q, rows * row_bytes);
    for (xr, qr) in src.chunks_exact(k).zip(out.chunks_exact_mut(row_bytes)) {
        match fmt {
            QFormat::Q8 => quantize_row_q8_0(xr, qr),
            QFormat::Q4 => quantize_row_q4_0(xr, qr),
        }
    }
    QUANT_OK
}

/// Pack `x` (whole rows of `k` values) into Q8_0 blocks in `q` — the `quantize_q8_0(x, q, k)`
/// builtin. Returns [`QUANT_OK`] or [`QUANT_BAD_EXTENT`] (nothing written).
///
/// # Safety
/// `x` addresses `x_len` f32 and `q` `q_len` writable bytes (a zero-length operand may be null).
#[no_mangle]
pub unsafe extern "C" fn wukong_quantize_q8_0_f32(
    x: *const f32,
    x_len: i64,
    q: *mut u8,
    q_len: i64,
    k: i64,
) -> i64 {
    quantize_entry(x, x_len, q, q_len, k, QFormat::Q8)
}

/// Pack `x` into Q4_0 blocks — the `quantize_q4_0(x, q, k)` builtin. See
/// [`wukong_quantize_q8_0_f32`].
///
/// # Safety
/// The contract of [`wukong_quantize_q8_0_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_quantize_q4_0_f32(
    x: *const f32,
    x_len: i64,
    q: *mut u8,
    q_len: i64,
    k: i64,
) -> i64 {
    quantize_entry(x, x_len, q, q_len, k, QFormat::Q4)
}

/// `y = W·x` for Q8_0 weights `W[N, K]` — the `gemv_q8_0(w, x, y)` builtin. Returns [`QUANT_OK`] or
/// [`QUANT_BAD_EXTENT`] (nothing written).
///
/// # Safety
/// `w` addresses `w_len` bytes, `x` `x_len` f32 and `y` `y_len` writable f32 (a zero-length operand
/// may be null). An output overlapping an input is handled (value semantics).
#[no_mangle]
pub unsafe extern "C" fn wukong_gemv_q8_0_f32(
    w: *const u8,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    gemv_q_entry(w, w_len, x, x_len, y, y_len, QFormat::Q8, false)
}

/// Multi-threaded [`wukong_gemv_q8_0_f32`]: row groups across cores, bit-identical to serial.
///
/// # Safety
/// The contract of [`wukong_gemv_q8_0_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_gemv_q8_0_f32_parallel(
    w: *const u8,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    gemv_q_entry(w, w_len, x, x_len, y, y_len, QFormat::Q8, true)
}

/// `y = W·x` for Q4_0 weights `W[N, K]` — the `gemv_q4_0(w, x, y)` builtin. See
/// [`wukong_gemv_q8_0_f32`].
///
/// # Safety
/// The contract of [`wukong_gemv_q8_0_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_gemv_q4_0_f32(
    w: *const u8,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    gemv_q_entry(w, w_len, x, x_len, y, y_len, QFormat::Q4, false)
}

/// Multi-threaded [`wukong_gemv_q4_0_f32`]: row groups across cores, bit-identical to serial.
///
/// # Safety
/// The contract of [`wukong_gemv_q8_0_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_gemv_q4_0_f32_parallel(
    w: *const u8,
    w_len: i64,
    x: *const f32,
    x_len: i64,
    y: *mut f32,
    y_len: i64,
) -> i64 {
    gemv_q_entry(w, w_len, x, x_len, y, y_len, QFormat::Q4, true)
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

    type Quantize = unsafe extern "C" fn(*const f32, i64, *mut u8, i64, i64) -> i64;

    fn pack(x: &[f32], k: usize, fmt: QFormat) -> Vec<u8> {
        let rows = x.len() / k;
        let mut q = vec![0u8; rows * fmt.row_bytes(k)];
        let entry: Quantize = match fmt {
            QFormat::Q8 => wukong_quantize_q8_0_f32,
            QFormat::Q4 => wukong_quantize_q4_0_f32,
        };
        // SAFETY: sized buffers.
        let code = unsafe {
            entry(
                x.as_ptr(),
                x.len() as i64,
                q.as_mut_ptr(),
                q.len() as i64,
                k as i64,
            )
        };
        assert_eq!(code, QUANT_OK);
        q
    }

    /// Dequantize element `e` of a packed row.
    fn dequant(fmt: QFormat, row: &[u8], e: usize) -> f32 {
        let bb = fmt.block_bytes();
        let blk = &row[(e / QK) * bb..(e / QK + 1) * bb];
        block_scale(blk) * weight_quant(fmt, &blk[2..], e % QK) as f32
    }

    type Gemv = unsafe extern "C" fn(*const u8, i64, *const f32, i64, *mut f32, i64) -> i64;

    fn entries(fmt: QFormat) -> (Gemv, Gemv) {
        match fmt {
            QFormat::Q8 => (wukong_gemv_q8_0_f32, wukong_gemv_q8_0_f32_parallel),
            QFormat::Q4 => (wukong_gemv_q4_0_f32, wukong_gemv_q4_0_f32_parallel),
        }
    }

    fn gemv(fmt: QFormat, w: &[u8], x: &[f32], n: usize, par: bool) -> Vec<f32> {
        let mut y = vec![0.0f32; n];
        let (serial, parallel) = entries(fmt);
        let f = if par { parallel } else { serial };
        // SAFETY: sized buffers.
        let code = unsafe {
            f(
                w.as_ptr(),
                w.len() as i64,
                x.as_ptr(),
                x.len() as i64,
                y.as_mut_ptr(),
                n as i64,
            )
        };
        assert_eq!(code, QUANT_OK);
        y
    }

    /// The packed layout, byte for byte, on values the formats represent exactly: a block holding
    /// 127 gets scale 1.0 (`0x3C00`) and `q = x`; a Q4 block whose largest magnitude is `-8` gets
    /// scale `1.0` and nibbles `x + 8`, element `j` low and `j + 16` high. `K = 40` pads the second
    /// block with zeros.
    #[test]
    fn packers_write_the_ggml_block_layout() {
        let k = 40;
        let x: Vec<f32> = (0..k)
            .map(|e| match e {
                0 | 32 => 127.0,
                _ => (e as f32) - 20.0,
            })
            .collect();
        let q8 = pack(&x, k, QFormat::Q8);
        assert_eq!(q8.len(), 2 * Q8_0_BLOCK_BYTES);
        assert_eq!(&q8[0..2], &[0x00, 0x3C]);
        assert_eq!(&q8[34..36], &[0x00, 0x3C]);
        for e in 0..k {
            let b = e / QK;
            assert_eq!(q8[b * 34 + 2 + e % QK] as i8 as f32, x[e], "q8 element {e}");
        }
        for e in k..64 {
            assert_eq!(q8[34 + 2 + e % QK], 0, "q8 padding {e}");
        }

        let x4: Vec<f32> = (0..k)
            .map(|e| match e {
                3 | 35 => -8.0,
                _ => ((e % 15) as f32) - 7.0,
            })
            .collect();
        let q4 = pack(&x4, k, QFormat::Q4);
        assert_eq!(q4.len(), 2 * Q4_0_BLOCK_BYTES);
        assert_eq!(&q4[0..2], &[0x00, 0x3C]);
        for e in 0..64 {
            let want = if e < k { x4[e] } else { 0.0 };
            assert_eq!(dequant(QFormat::Q4, &q4, e), want, "q4 element {e}");
        }
        assert_eq!(q4[2], ((x4[0] + 8.0) as u8) | (((x4[16] + 8.0) as u8) << 4));
    }

    /// A NaN element is ignored by the scale and quantizes to zero in both formats (Q8_0 quant 0,
    /// Q4_0 nibble 8), as the module docs promise — not to Q4_0's most negative nibble.
    #[test]
    fn a_nan_element_quantizes_to_zero() {
        let mut x: Vec<f32> = (0..QK).map(|e| (e % 7) as f32 - 3.0).collect();
        x[1] = -8.0;
        x[5] = f32::NAN;
        let q8 = pack(&x, QK, QFormat::Q8);
        assert_eq!(q8[2 + 5], 0);
        let q4 = pack(&x, QK, QFormat::Q4);
        assert_eq!(&q4[0..2], &[0x00, 0x3C]);
        assert_eq!(q4[2 + 5] & 0x0F, 8);
        assert_eq!(dequant(QFormat::Q4, &q4, 5), 0.0);
    }

    /// With exactly representable weights and activations (every block scale 1) the GEMV is the
    /// exact integer matrix-vector product, whatever the accumulation order.
    #[test]
    fn exact_values_give_the_exact_product() {
        let (n, k) = (11usize, 40usize);
        for fmt in [QFormat::Q8, QFormat::Q4] {
            let top = if fmt == QFormat::Q8 { 127.0 } else { -8.0 };
            let w: Vec<f32> = (0..n * k)
                .map(|t| {
                    let e = t % k;
                    if e == 1 || e == 33 {
                        top
                    } else {
                        (((t * 7) % 13) as f32) - 6.0
                    }
                })
                .collect();
            let x: Vec<f32> = (0..k)
                .map(|e| {
                    if e == 0 || e == 32 {
                        127.0
                    } else {
                        ((e % 9) as f32) - 4.0
                    }
                })
                .collect();
            let wq = pack(&w, k, fmt);
            for par in [false, true] {
                let y = gemv(fmt, &wq, &x, n, par);
                for j in 0..n {
                    let want: f32 = (0..k).map(|e| w[j * k + e] * x[e]).sum();
                    assert_eq!(y[j], want, "{fmt:?} par={par} row {j}");
                }
            }
        }
    }

    /// The AVX2 eight-row path, the scalar twin and the parallel entry agree bit for bit on random
    /// data, at row counts around the 8-row group and K around the block size.
    #[test]
    fn simd_scalar_and_parallel_agree_bit_for_bit() {
        for fmt in [QFormat::Q8, QFormat::Q4] {
            for &(n, k) in &[
                (1usize, 1usize),
                (7, 31),
                (8, 32),
                (9, 33),
                (17, 100),
                (40, 257),
            ] {
                let w = fill(3 + n as u64, n * k);
                let x = fill(4 + k as u64, k);
                let wq = pack(&w, k, fmt);
                let got = gemv(fmt, &wq, &x, n, false);
                let got_par = gemv(fmt, &wq, &x, n, true);
                let nb = k.div_ceil(QK);
                let mut xq = vec![0i8; nb * QK];
                let mut dx = vec![0.0f32; nb];
                quantize_activation(&x, &mut xq, &mut dx);
                let rb = fmt.row_bytes(k);
                for j in 0..n {
                    let want = row_dot(fmt, &wq[j * rb..(j + 1) * rb], &xq, &dx);
                    assert_eq!(got[j].to_bits(), want.to_bits(), "{fmt:?} {n}x{k} row {j}");
                    assert_eq!(got_par[j].to_bits(), want.to_bits(), "{fmt:?} par row {j}");
                }
            }
        }
    }

    /// Against the f64 product of the *dequantized* weights and the original activation, the only
    /// error left is the activation's own quantization (half a step per element) plus f32 rounding.
    #[test]
    fn matches_the_dequantized_reference_within_the_activation_step() {
        for fmt in [QFormat::Q8, QFormat::Q4] {
            let (n, k) = (13usize, 96usize);
            let w = fill(9, n * k);
            let x = fill(10, k);
            let wq = pack(&w, k, fmt);
            let y = gemv(fmt, &wq, &x, n, false);
            let rb = fmt.row_bytes(k);
            for j in 0..n {
                let row = &wq[j * rb..(j + 1) * rb];
                let (mut exact, mut bound) = (0.0f64, 0.0f64);
                for b in 0..k / QK {
                    let blk = &x[b * QK..(b + 1) * QK];
                    let step = block_amax(blk) as f64 / 127.0;
                    for e in 0..QK {
                        let wd = dequant(fmt, row, b * QK + e) as f64;
                        exact += wd * x[b * QK + e] as f64;
                        bound += wd.abs() * (0.5 * step + 1e-6);
                    }
                }
                assert!(
                    (y[j] as f64 - exact).abs() <= bound + 1e-5,
                    "{fmt:?} row {j}: got {} exact {exact} bound {bound}",
                    y[j]
                );
            }
        }
    }

    /// Weight bytes outside what the packer emits (-128) still dot exactly: the sign trick keeps
    /// every pair sum below the i16 saturation bound.
    #[test]
    fn extreme_weight_bytes_do_not_saturate() {
        let k = 64;
        let mut wq = vec![0u8; 2 * Q8_0_BLOCK_BYTES * 8];
        for r in 0..8 {
            for b in 0..2 {
                let blk = &mut wq[(r * 2 + b) * 34..(r * 2 + b + 1) * 34];
                blk[..2].copy_from_slice(&crate::f32_to_f16_bits(1.0).to_le_bytes());
                for e in 0..QK {
                    blk[2 + e] = if (e + r) % 2 == 0 { 0x80 } else { 0x7F };
                }
            }
        }
        let x: Vec<f32> = (0..k)
            .map(|e| if e % 2 == 0 { -127.0 } else { 127.0 })
            .collect();
        let y = gemv(QFormat::Q8, &wq, &x, 8, false);
        for r in 0..8 {
            let mut want = 0i64;
            for e in 0..k {
                let w = if (e % QK + r) % 2 == 0 { -128i64 } else { 127 };
                want += w * x[e] as i64;
            }
            assert_eq!(y[r], want as f32, "row {r}");
        }
    }

    #[test]
    fn malformed_shapes_are_refused_and_empty_contractions_are_zero() {
        assert_eq!(quantize_plan(64, 68, 32, QFormat::Q8), Ok((2, 32)));
        assert_eq!(quantize_plan(80, 72, 40, QFormat::Q4), Ok((2, 40)));
        assert_eq!(
            quantize_plan(64, 67, 32, QFormat::Q8),
            Err(QUANT_BAD_EXTENT)
        );
        assert_eq!(
            quantize_plan(65, 68, 32, QFormat::Q8),
            Err(QUANT_BAD_EXTENT)
        );
        assert_eq!(quantize_plan(64, 68, 0, QFormat::Q8), Err(QUANT_BAD_EXTENT));
        assert_eq!(gemv_q_plan(36, 40, 1, QFormat::Q4), Ok((1, 40)));
        assert_eq!(gemv_q_plan(35, 40, 1, QFormat::Q4), Err(QUANT_BAD_EXTENT));
        assert_eq!(gemv_q_plan(0, 0, 5, QFormat::Q8), Ok((5, 0)));
        assert_eq!(
            gemv_q_plan(0, i64::MAX, i64::MAX, QFormat::Q8),
            Err(QUANT_BAD_EXTENT)
        );
        let mut y = [3.0f32; 5];
        // SAFETY: K = 0, so the weight and activation are never read.
        let code = unsafe {
            wukong_gemv_q4_0_f32(std::ptr::null(), 0, std::ptr::null(), 0, y.as_mut_ptr(), 5)
        };
        assert_eq!(code, QUANT_OK);
        assert_eq!(y, [0.0f32; 5]);
        let w = [0u8; 36];
        let x = [1.0f32; 40];
        // SAFETY: refused before anything is read.
        let code =
            unsafe { wukong_gemv_q4_0_f32(w.as_ptr(), 35, x.as_ptr(), 40, y.as_mut_ptr(), 1) };
        assert_eq!(code, QUANT_BAD_EXTENT);
    }
}
