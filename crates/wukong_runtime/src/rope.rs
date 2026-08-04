//! RoPE (rotary position embedding) with **inline sin/cos** — the per-attention-layer positional
//! rotation in LLaMA/Mistral/Qwen/GPT-NeoX. Applied over a row-major `[rows, dim]` f32 tensor with
//! `dim = 2*half`, the **half-split** (GPT-NeoX/LLaMA) convention: the first `half` channels pair with
//! the last `half`, and row `r`'s absolute position is the row index `r`. The C entries take
//! **`half`**, not `dim` — the extent guard is `rows <= 0 || half <= 0`, and `dim` is derived as
//! `2*half` inside.
//!
//! Per row `r` and each pair `j ∈ 0..half`:
//!
//! ```text
//! theta = (r as f32) * inv_freq[j]        // inv_freq is a precomputed [half] table, shared across rows
//! c = cos(theta); s = sin(theta)
//! a = x[r*dim + j]; b = x[r*dim + j + half]
//! out[r*dim + j]        = a*c - b*s
//! out[r*dim + j + half] = b*c + a*s
//! ```
//!
//! This is a 2×2 rotation of the pair `(a, b)` by `theta`, so it **preserves the pair norm**
//! `a² + b²` (a unit-test sanity check), and `out` may alias `x` (each pair reads both its inputs
//! before writing either output).
//!
//! **Why a kernel wins.** The angles' `sin`/`cos` are computed *inline* per element (not read from a
//! precomputed `[rows, half]` table), so a C/Rust loop calls `sinf`/`cosf` **scalar** — there is no
//! library vectorized `sincosf`. A 256-bit fused kernel that evaluates 8 angles' sin/cos at once wins
//! like the `vmath` sin/cos dispatch (~6–8×). The trig is the **shared Cephes minimax** from
//! [`crate::vmath`] — scalar [`sincos1`](crate::vmath::sincos1) for the tail/fallback, AVX2
//! [`sin8`](crate::vmath::sin8)/[`cos8`](crate::vmath::cos8) for the body — so this rotation is
//! **bit-identical with the `vmath` sin/cos dispatch** (no new polynomial, no new reduction).
//!
//! **Determinism.** The whole op is pure elementwise (no reduction), so the AVX2 body and the scalar
//! twin agree **bit-for-bit**: the body vectorizes over `j` (8 pairs/step), broadcasting `pos = r as
//! f32`, computing `theta = pos * inv_freq[j..]`, then `sin8`/`cos8` (lane-for-lane identical to
//! `sincos1`), and the rotation combines use one FMA each (`_mm256_fnmadd_ps`/`_mm256_fmadd_ps` ==
//! scalar `mul_add`), with a scalar tail for `half % 8`. Rows are independent, so the `_parallel`
//! entry just maps the identical per-row routine across rows — `serial == parallel` bit-for-bit with
//! no cross-row combine, so the interpreter's serial call agrees with the `@parallel` path exactly.
//! (The trig's lane-reassociation-free poly is the documented `vmath` sin/cos contract: every backend
//! runs this same Cephes sequence, so they agree.)

use rayon::prelude::*;

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// One row of RoPE, scalar reference / AVX2 tail / no-AVX2 fallback.
///
/// `pos` is the row's absolute position (`r as f32`); `inv_freq` is the `[half]` frequency table;
/// `x`/`out` point at this row's `[dim]` slice (`dim = 2*half`). Each pair `(a, b) =
/// (x[j], x[j+half])` is rotated by `theta = pos * inv_freq[j]` into `(out[j], out[j+half]) = (a·c −
/// b·s, b·c + a·s)`. `theta` is a plain multiply (matching the AVX2 `mulps`); `sincos1` is the shared
/// Cephes poly; the two combines fold via `mul_add` so they match the AVX2 `fnmadd`/`fmadd`.
///
/// # Safety
/// `inv_freq` valid for `half` f32; `x`, `out` valid for `2*half` f32 (`out` may alias `x` — both of a
/// pair's inputs are read before either output is written).
#[inline]
unsafe fn rope_row_scalar(pos: f32, inv_freq: *const f32, x: *const f32, out: *mut f32, half: usize) {
    for j in 0..half {
        let theta = pos * *inv_freq.add(j);
        let c = crate::vmath::sincos1(theta, true);
        let s = crate::vmath::sincos1(theta, false);
        let a = *x.add(j);
        let b = *x.add(j + half);
        // out[j] = a*c - b*s = fma(b, -s, a*c);  out[j+half] = b*c + a*s = fma(a, s, b*c).
        *out.add(j) = b.mul_add(-s, a * c);
        *out.add(j + half) = a.mul_add(s, b * c);
    }
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) -------------------------

/// One row of RoPE, AVX2/FMA. Vectorizes over `j` (8 pairs/step): broadcast `pos`, load `inv_freq[j..
/// j+8]`, `theta = pos * inv_freq`, then `c = cos8(theta)` / `s = sin8(theta)` (the shared Cephes,
/// lane-for-lane identical to `sincos1`); load `a = x[j..]` and `b = x[j+half..]` and combine with one
/// FMA each (`fnmadd`/`fmadd` == the scalar `mul_add`). A scalar tail folds the `half % 8` remainder
/// through the identical ops.
///
/// # Safety
/// `inv_freq` valid for `half` f32; `x`, `out` valid for `2*half` f32 (`out` may alias `x`); AVX2+FMA
/// available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rope_row_avx2(pos: f32, inv_freq: *const f32, x: *const f32, out: *mut f32, half: usize) {
    use std::arch::x86_64::*;
    let posv = _mm256_set1_ps(pos);
    let mut j = 0;
    while j + 8 <= half {
        let invf = _mm256_loadu_ps(inv_freq.add(j));
        let theta = _mm256_mul_ps(posv, invf);
        let c = crate::vmath::cos8(theta);
        let s = crate::vmath::sin8(theta);
        let a = _mm256_loadu_ps(x.add(j));
        let b = _mm256_loadu_ps(x.add(j + half));
        // out_lo = a*c - b*s = fnmadd(b, s, a*c);  out_hi = b*c + a*s = fmadd(a, s, b*c).
        let out_lo = _mm256_fnmadd_ps(b, s, _mm256_mul_ps(a, c));
        let out_hi = _mm256_fmadd_ps(a, s, _mm256_mul_ps(b, c));
        _mm256_storeu_ps(out.add(j), out_lo);
        _mm256_storeu_ps(out.add(j + half), out_hi);
        j += 8;
    }
    // Tail: identical ops on the `half % 8` remainder (note `out` may alias `x`, but the head wrote
    // only `[0,j)` ∪ `[half, half+j)`, disjoint from the pairs read here).
    while j < half {
        let theta = pos * *inv_freq.add(j);
        let c = crate::vmath::sincos1(theta, true);
        let s = crate::vmath::sincos1(theta, false);
        let a = *x.add(j);
        let b = *x.add(j + half);
        *out.add(j) = b.mul_add(-s, a * c);
        *out.add(j + half) = a.mul_add(s, b * c);
        j += 1;
    }
}

/// One row through RoPE, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `inv_freq` valid for `half` f32; `x`, `out` valid for `2*half` f32 (`out` may alias `x`).
#[inline]
unsafe fn rope_row(pos: f32, inv_freq: *const f32, x: *const f32, out: *mut f32, half: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return rope_row_avx2(pos, inv_freq, x, out, half);
        }
    }
    rope_row_scalar(pos, inv_freq, x, out, half);
}

/// RoPE over a `[rows, dim]` row-major batch (`dim = 2*half`), single-threaded. Row `r`'s absolute
/// position is the row index `r`; `inv_freq` (length `half`, shared across rows) is the precomputed
/// frequency table. `out` may alias `x` (each pair reads both inputs before writing either output).
///
/// # Safety
/// `x`, `out` valid for `rows*dim = rows*2*half` f32; `inv_freq` valid for `half` f32. `out` may alias
/// `x` but must not otherwise overlap it.
#[no_mangle]
pub unsafe extern "C" fn wukong_rope_f32(
    x: *const f32,
    inv_freq: *const f32,
    out: *mut f32,
    rows: i64,
    half: i64,
) {
    if rows <= 0 || half <= 0 {
        return;
    }
    let (r, h) = (rows as usize, half as usize);
    let dim = 2 * h;
    for row in 0..r {
        let off = row * dim;
        // SAFETY: row `row` occupies [off, off+dim) ⊆ [0, rows*dim); inv_freq indexed within [0, half).
        rope_row(row as f32, inv_freq, x.add(off), out.add(off), h);
    }
}

/// Row count below which the parallel RoPE just runs serially (the rayon fan-out isn't worth it).
const ROPE_PAR_MIN: usize = 8;

/// Multi-threaded RoPE: rows are mapped across cores, each computed by the identical per-row routine —
/// so the result is **bit-identical to [`wukong_rope_f32`]** (rows are independent, no cross-row
/// combine, so thread count is irrelevant and the interpreter's serial call agrees with this
/// `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_rope_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_rope_f32_parallel(
    x: *const f32,
    inv_freq: *const f32,
    out: *mut f32,
    rows: i64,
    half: i64,
) {
    if rows <= 0 || half <= 0 {
        return;
    }
    let (r, h) = (rows as usize, half as usize);
    if r < ROPE_PAR_MIN {
        wukong_rope_f32(x, inv_freq, out, rows, half);
        return;
    }
    let dim = 2 * h;
    // Raw pointers cross the rayon boundary as integers; each row is a disjoint sub-slice, inv_freq is
    // shared read-only (copy the rmsnorm_bwd.rs re-derive-from-usize pattern).
    let (x_addr, f_addr, out_addr) = (x as usize, inv_freq as usize, out as usize);
    // This fork can be the process's FIRST rayon touch, so it must provision the global pool first —
    // [`crate::ensure_global_pool`]'s stated precondition. Forking bare builds rayon's default
    // 2 MiB-stack registry, so the runtime's later 16 MiB `build_global` silently loses the race and
    // outlined `@parallel` region bodies are left on undersized stacks. Idempotent (`Once`) and
    // provisioning-only: the work split below is unchanged, so serial == parallel stays bit-exact.
    crate::ensure_global_pool();
    (0..r).into_par_iter().for_each(|row| {
        let off = row * dim;
        // SAFETY: disjoint row slices; pointers re-derived from the captured addresses; inv_freq valid
        // for `h` by contract.
        unsafe {
            rope_row(
                row as f32,
                f_addr as *const f32,
                (x_addr as *const f32).add(off),
                (out_addr as *mut f32).add(off),
                h,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, mildly varied streams (no RNG — reproducible). Distinct phases/scales so x and
    // inv_freq are not accidentally equal. inv_freq is kept small-positive (the real RoPE
    // `base^(-2j/dim)` ∈ (0,1]) so angles stay in the ≈1-ULP poly-reduction range.
    fn fill_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.019 + 0.3).sin() * 1.7 - 0.25)
            .collect()
    }
    fn fill_inv_freq(half: usize) -> Vec<f32> {
        // Geometric-ish decay in (0, 1], like 10000^(-2j/dim), without pulling in a base/pow.
        (0..half)
            .map(|j| (-(j as f32) * (3.0 / half.max(1) as f32)).exp())
            .collect()
    }

    /// 1) scalar path == AVX2 path bit-for-bit across `half` straddling the 8-lane edge (incl.
    /// non-multiples of 8 and `half < 8`) and several positions — the body/tail agreement that
    /// underwrites the differential gate.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_matches_avx2_bit_for_bit() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for &half in &[1usize, 4, 7, 8, 9, 16, 17, 31, 64, 100, 257] {
            let dim = 2 * half;
            let inv_freq = fill_inv_freq(half);
            // A few representative absolute positions (incl. 0, where theta == 0 ⇒ c=1, s=0).
            for &pos in &[0.0f32, 1.0, 7.0, 42.0, 511.0] {
                let x = fill_x(dim);
                let mut a = vec![0.0f32; dim];
                let mut b = vec![0.0f32; dim];
                unsafe {
                    rope_row_scalar(pos, inv_freq.as_ptr(), x.as_ptr(), a.as_mut_ptr(), half);
                    rope_row_avx2(pos, inv_freq.as_ptr(), x.as_ptr(), b.as_mut_ptr(), half);
                }
                for i in 0..dim {
                    assert_eq!(
                        a[i].to_bits(),
                        b[i].to_bits(),
                        "scalar != avx2 half={half} pos={pos} i={i}: {} vs {}",
                        a[i],
                        b[i]
                    );
                }
            }
        }
    }

    /// 2) serial == parallel bit-for-bit across shapes straddling the parallel threshold and the
    /// 8-lane edge (rows ≥ ROPE_PAR_MIN actually fan out).
    #[test]
    fn serial_matches_parallel_bit_for_bit() {
        for (rows, half) in [
            (1usize, 1usize),
            (3, 7),
            (4, 8),
            (8, 9),
            (16, 17),
            (64, 100),
            (40, 257),
        ] {
            let dim = 2 * half;
            let x = fill_x(rows * dim);
            let inv_freq = fill_inv_freq(half);
            let mut got = vec![0.0f32; rows * dim];
            let mut got_par = vec![0.0f32; rows * dim];
            unsafe {
                wukong_rope_f32(
                    x.as_ptr(),
                    inv_freq.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    half as i64,
                );
                wukong_rope_f32_parallel(
                    x.as_ptr(),
                    inv_freq.as_ptr(),
                    got_par.as_mut_ptr(),
                    rows as i64,
                    half as i64,
                );
            }
            for i in 0..rows * dim {
                assert_eq!(
                    got[i].to_bits(),
                    got_par[i].to_bits(),
                    "serial != parallel {rows}x{half} at {i}: {} vs {}",
                    got[i],
                    got_par[i]
                );
            }
        }
    }

    /// 3) the kernel ≈ an independent f64 reference (the rotation recomputed with f64 `sin`/`cos`),
    /// within a tolerance reflecting the ~1-ULP f32 Cephes poly — guards the *formula* (not just
    /// scalar==avx2). Positions are the row indices, as in the real op.
    #[test]
    fn rope_matches_f64_reference() {
        let (rows, half) = (12usize, 96usize);
        let dim = 2 * half;
        let x = fill_x(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut got = vec![0.0f32; rows * dim];
        unsafe {
            wukong_rope_f32(
                x.as_ptr(),
                inv_freq.as_ptr(),
                got.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
        }
        let mut max_err = 0.0f64;
        for row in 0..rows {
            let off = row * dim;
            let pos = row as f64;
            for j in 0..half {
                let theta = pos * inv_freq[j] as f64;
                let (s, c) = theta.sin_cos();
                let a = x[off + j] as f64;
                let b = x[off + j + half] as f64;
                let want_lo = a * c - b * s;
                let want_hi = b * c + a * s;
                let e_lo = (got[off + j] as f64 - want_lo).abs();
                let e_hi = (got[off + j + half] as f64 - want_hi).abs();
                max_err = max_err.max(e_lo).max(e_hi);
            }
        }
        // Unit-scale data, ~1-ULP f32 trig ⇒ comfortably under 1e-4.
        assert!(
            max_err < 1e-4,
            "rope vs f64 reference: max abs err {max_err} too large"
        );
    }

    /// 4) norm-preservation: RoPE is a 2×2 rotation, so each pair's `a² + b²` is preserved. Check
    /// `out[j]² + out[j+half]² ≈ x[j]² + x[j+half]²` per pair (f32 tolerance — the rotation and the
    /// squares both round).
    #[test]
    fn rope_preserves_pair_norm() {
        let (rows, half) = (9usize, 80usize);
        let dim = 2 * half;
        let x = fill_x(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut out = vec![0.0f32; rows * dim];
        unsafe {
            wukong_rope_f32(
                x.as_ptr(),
                inv_freq.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
        }
        for row in 0..rows {
            let off = row * dim;
            for j in 0..half {
                let a = x[off + j];
                let b = x[off + j + half];
                let oa = out[off + j];
                let ob = out[off + j + half];
                let before = (a as f64) * (a as f64) + (b as f64) * (b as f64);
                let after = (oa as f64) * (oa as f64) + (ob as f64) * (ob as f64);
                let denom = before.max(1.0);
                assert!(
                    ((after - before).abs() / denom) < 1e-4,
                    "pair norm not preserved row={row} j={j}: {before} -> {after}"
                );
            }
        }
    }

    /// 5) in-place (`out == x`) equals the out-of-place result — RoPE reads both of a pair's inputs
    /// before writing either output, so aliasing is sound.
    #[test]
    fn rope_in_place_matches_out_of_place() {
        let (rows, half) = (6usize, 70usize);
        let dim = 2 * half;
        let x = fill_x(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut out = vec![0.0f32; rows * dim];
        let mut inplace = x.clone();
        unsafe {
            wukong_rope_f32(
                x.as_ptr(),
                inv_freq.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
            // out aliases x.
            wukong_rope_f32(
                inplace.as_ptr(),
                inv_freq.as_ptr(),
                inplace.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
        }
        for i in 0..rows * dim {
            assert_eq!(
                out[i].to_bits(),
                inplace[i].to_bits(),
                "in-place != out-of-place at {i}: {} vs {}",
                inplace[i],
                out[i]
            );
        }
    }

    /// Edge: zero/negative rows or half is a no-op (don't write, don't panic).
    #[test]
    fn degenerate_shapes_are_noops() {
        let inv_freq = fill_inv_freq(4);
        let x = fill_x(8);
        let mut out = vec![42.0f32; 8];
        unsafe {
            wukong_rope_f32(x.as_ptr(), inv_freq.as_ptr(), out.as_mut_ptr(), 0, 4);
            wukong_rope_f32(x.as_ptr(), inv_freq.as_ptr(), out.as_mut_ptr(), 2, 0);
            wukong_rope_f32_parallel(x.as_ptr(), inv_freq.as_ptr(), out.as_mut_ptr(), -1, 4);
        }
        assert!(out.iter().all(|&v| v == 42.0), "no-op must not write out");
    }
}
