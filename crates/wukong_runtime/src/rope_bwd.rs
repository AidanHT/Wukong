//! RoPE backward — the gradient through the rotary position embedding (the LLaMA/Mistral/Qwen/
//! GPT-NeoX per-attention-layer positional rotation). Pairs with the forward [`crate::rope`] over a
//! row-major `[rows, dim]` f32 tensor with `dim = 2*half`, the **half-split** convention: the first
//! `half` channels pair with the last `half`, and row `r`'s absolute position is the row index `r`.
//!
//! RoPE forward rotates each pair `(a, b)` by `+theta` (the matrix `[[c, -s], [s, c]]`). The backward
//! pass propagates the upstream gradient `g` through that rotation, which — since a rotation's Jacobian
//! is the rotation itself — is the **transpose (= inverse) rotation** by `-theta` (the matrix
//! `[[c, s], [-s, c]]`). Per row `r` and each pair `j ∈ 0..half`:
//!
//! ```text
//! theta = (r as f32) * inv_freq[j]        // inv_freq is a precomputed [half] table, shared across rows
//! c = cos(theta); s = sin(theta)
//! a = g[r*dim + j]; b = g[r*dim + j + half]
//! dx[r*dim + j]        = a*c + b*s         // transpose of [[c,-s],[s,c]] is [[c,s],[-s,c]]
//! dx[r*dim + j + half] = b*c - a*s
//! ```
//!
//! Because the inverse rotation undoes the forward rotation, `rope_bwd(rope_fwd(x)) ≈ x` (a round-trip
//! unit test), and `dx` may alias `g` (each pair reads both its inputs before writing either output).
//! This is also a 2×2 rotation, so it preserves the pair norm `a² + b²`.
//!
//! **Why a kernel wins.** As in the forward, the angles' `sin`/`cos` are computed *inline* per element
//! (not read from a precomputed `[rows, half]` table), so a C/Rust loop calls `sinf`/`cosf` **scalar** —
//! there is no library vectorized `sincosf`. A 256-bit fused kernel that evaluates 8 angles' sin/cos at
//! once wins like the `vmath` sin/cos dispatch (~6–8×). The trig is the **shared Cephes minimax** from
//! [`crate::vmath`] — scalar [`sincos1`](crate::vmath::sincos1) for the tail/fallback, AVX2
//! [`sin8`](crate::vmath::sin8)/[`cos8`](crate::vmath::cos8) for the body — so this rotation is
//! **bit-identical with the `vmath` sin/cos dispatch and the RoPE forward** (no new polynomial, no new
//! reduction). Only the two combine signs differ from the forward.
//!
//! **Determinism.** The whole op is pure elementwise (no reduction), so the AVX2 body and the scalar
//! twin agree **bit-for-bit**: the body vectorizes over `j` (8 pairs/step), broadcasting `pos = r as
//! f32`, computing `theta = pos * inv_freq[j..]`, then `sin8`/`cos8` (lane-for-lane identical to
//! `sincos1`), and the rotation combines use one FMA each (`_mm256_fmadd_ps`/`_mm256_fnmadd_ps` ==
//! scalar `mul_add`), with a scalar tail for `half % 8`. Rows are independent, so the `_parallel`
//! entry just maps the identical per-row routine across rows — `serial == parallel` bit-for-bit with
//! no cross-row combine, so the interpreter's serial call agrees with the `@parallel` path exactly.
//! (The trig's lane-reassociation-free poly is the documented `vmath` sin/cos contract: every backend
//! runs this same Cephes sequence, so they agree.)

use rayon::prelude::*;

// --- scalar twin (the AVX2 tail + the no-AVX2 fallback) -------------------------------------------

/// One row of RoPE backward, scalar reference / AVX2 tail / no-AVX2 fallback.
///
/// `pos` is the row's absolute position (`r as f32`); `inv_freq` is the `[half]` frequency table;
/// `g`/`dx` point at this row's `[dim]` slice (`dim = 2*half`). Each pair `(a, b) = (g[j], g[j+half])`
/// is rotated by the **inverse** of `theta = pos * inv_freq[j]` into `(dx[j], dx[j+half]) = (a·c + b·s,
/// b·c − a·s)`. `theta` is a plain multiply (matching the AVX2 `mulps`); `sincos1` is the shared
/// Cephes poly; the two combines fold via `mul_add` so they match the AVX2 `fmadd`/`fnmadd`.
///
/// # Safety
/// `inv_freq` valid for `half` f32; `g`, `dx` valid for `2*half` f32 (`dx` may alias `g` — both of a
/// pair's inputs are read before either output is written).
#[inline]
unsafe fn rope_bwd_row_scalar(pos: f32, inv_freq: *const f32, g: *const f32, dx: *mut f32, half: usize) {
    for j in 0..half {
        let theta = pos * *inv_freq.add(j);
        let c = crate::vmath::sincos1(theta, true);
        let s = crate::vmath::sincos1(theta, false);
        let a = *g.add(j);
        let b = *g.add(j + half);
        // dx[j] = a*c + b*s = fma(b, s, a*c);  dx[j+half] = b*c - a*s = fma(a, -s, b*c).
        *dx.add(j) = b.mul_add(s, a * c);
        *dx.add(j + half) = a.mul_add(-s, b * c);
    }
}

// --- AVX2 kernel (mirrors the scalar twin lane-for-lane on finite inputs) -------------------------

/// One row of RoPE backward, AVX2/FMA. Vectorizes over `j` (8 pairs/step): broadcast `pos`, load
/// `inv_freq[j..j+8]`, `theta = pos * inv_freq`, then `c = cos8(theta)` / `s = sin8(theta)` (the shared
/// Cephes, lane-for-lane identical to `sincos1`); load `a = g[j..]` and `b = g[j+half..]` and combine
/// with one FMA each (`fmadd`/`fnmadd` == the scalar `mul_add`). A scalar tail folds the `half % 8`
/// remainder through the identical ops.
///
/// # Safety
/// `inv_freq` valid for `half` f32; `g`, `dx` valid for `2*half` f32 (`dx` may alias `g`); AVX2+FMA
/// available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rope_bwd_row_avx2(pos: f32, inv_freq: *const f32, g: *const f32, dx: *mut f32, half: usize) {
    use std::arch::x86_64::*;
    let posv = _mm256_set1_ps(pos);
    let mut j = 0;
    while j + 8 <= half {
        let invf = _mm256_loadu_ps(inv_freq.add(j));
        let theta = _mm256_mul_ps(posv, invf);
        let c = crate::vmath::cos8(theta);
        let s = crate::vmath::sin8(theta);
        let a = _mm256_loadu_ps(g.add(j));
        let b = _mm256_loadu_ps(g.add(j + half));
        // dx_lo = a*c + b*s = fmadd(b, s, a*c);  dx_hi = b*c - a*s = fnmadd(a, s, b*c).
        let dx_lo = _mm256_fmadd_ps(b, s, _mm256_mul_ps(a, c));
        let dx_hi = _mm256_fnmadd_ps(a, s, _mm256_mul_ps(b, c));
        _mm256_storeu_ps(dx.add(j), dx_lo);
        _mm256_storeu_ps(dx.add(j + half), dx_hi);
        j += 8;
    }
    // Tail: identical ops on the `half % 8` remainder (note `dx` may alias `g`, but the head wrote
    // only `[0,j)` ∪ `[half, half+j)`, disjoint from the pairs read here).
    while j < half {
        let theta = pos * *inv_freq.add(j);
        let c = crate::vmath::sincos1(theta, true);
        let s = crate::vmath::sincos1(theta, false);
        let a = *g.add(j);
        let b = *g.add(j + half);
        *dx.add(j) = b.mul_add(s, a * c);
        *dx.add(j + half) = a.mul_add(-s, b * c);
        j += 1;
    }
}

/// One row through RoPE backward, AVX2 when available, else the scalar twin.
///
/// # Safety
/// `inv_freq` valid for `half` f32; `g`, `dx` valid for `2*half` f32 (`dx` may alias `g`).
#[inline]
unsafe fn rope_bwd_row(pos: f32, inv_freq: *const f32, g: *const f32, dx: *mut f32, half: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return rope_bwd_row_avx2(pos, inv_freq, g, dx, half);
        }
    }
    rope_bwd_row_scalar(pos, inv_freq, g, dx, half);
}

/// RoPE backward (input-gradient) over a `[rows, dim]` row-major batch (`dim = 2*half`),
/// single-threaded. Row `r`'s absolute position is the row index `r`; `inv_freq` (length `half`,
/// shared across rows) is the precomputed frequency table. Applies the transpose (inverse) rotation to
/// the upstream gradient `g`. `dx` may alias `g` (each pair reads both inputs before writing either
/// output).
///
/// # Safety
/// `g`, `dx` valid for `rows*dim = rows*2*half` f32; `inv_freq` valid for `half` f32. `dx` may alias
/// `g` but must not otherwise overlap it.
#[no_mangle]
pub unsafe extern "C" fn wukong_rope_bwd_f32(
    g: *const f32,
    inv_freq: *const f32,
    dx: *mut f32,
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
        rope_bwd_row(row as f32, inv_freq, g.add(off), dx.add(off), h);
    }
}

/// Row count below which the parallel RoPE backward just runs serially (the rayon fan-out isn't worth
/// it).
const ROPE_BWD_PAR_MIN: usize = 8;

/// Multi-threaded RoPE backward: rows are mapped across cores, each computed by the identical per-row
/// routine — so the result is **bit-identical to [`wukong_rope_bwd_f32`]** (rows are independent, no
/// cross-row combine, so thread count is irrelevant and the interpreter's serial call agrees with this
/// `@parallel` path exactly).
///
/// # Safety
/// Operand-size contract of [`wukong_rope_bwd_f32`].
#[no_mangle]
pub unsafe extern "C" fn wukong_rope_bwd_f32_parallel(
    g: *const f32,
    inv_freq: *const f32,
    dx: *mut f32,
    rows: i64,
    half: i64,
) {
    if rows <= 0 || half <= 0 {
        return;
    }
    let (r, h) = (rows as usize, half as usize);
    if r < ROPE_BWD_PAR_MIN {
        wukong_rope_bwd_f32(g, inv_freq, dx, rows, half);
        return;
    }
    let dim = 2 * h;
    // Raw pointers cross the rayon boundary as integers; each row is a disjoint sub-slice, inv_freq is
    // shared read-only (copy the rope.rs / rmsnorm_bwd.rs re-derive-from-usize pattern).
    let (g_addr, f_addr, dx_addr) = (g as usize, inv_freq as usize, dx as usize);
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
            rope_bwd_row(
                row as f32,
                f_addr as *const f32,
                (g_addr as *const f32).add(off),
                (dx_addr as *mut f32).add(off),
                h,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic, mildly varied streams (no RNG — reproducible). Distinct phases/scales so g and
    // inv_freq are not accidentally equal. inv_freq is kept small-positive (the real RoPE
    // `base^(-2j/dim)` ∈ (0,1]) so angles stay in the ≈1-ULP poly-reduction range.
    fn fill_g(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.023 + 0.5).sin() * 1.4 - 0.3)
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
                let g = fill_g(dim);
                let mut a = vec![0.0f32; dim];
                let mut b = vec![0.0f32; dim];
                unsafe {
                    rope_bwd_row_scalar(pos, inv_freq.as_ptr(), g.as_ptr(), a.as_mut_ptr(), half);
                    rope_bwd_row_avx2(pos, inv_freq.as_ptr(), g.as_ptr(), b.as_mut_ptr(), half);
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
    /// 8-lane edge (rows ≥ ROPE_BWD_PAR_MIN actually fan out).
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
            let g = fill_g(rows * dim);
            let inv_freq = fill_inv_freq(half);
            let mut got = vec![0.0f32; rows * dim];
            let mut got_par = vec![0.0f32; rows * dim];
            unsafe {
                wukong_rope_bwd_f32(
                    g.as_ptr(),
                    inv_freq.as_ptr(),
                    got.as_mut_ptr(),
                    rows as i64,
                    half as i64,
                );
                wukong_rope_bwd_f32_parallel(
                    g.as_ptr(),
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

    /// 3) the kernel ≈ an independent f64 reference (the inverse rotation recomputed with f64
    /// `sin`/`cos`), within a tolerance reflecting the ~1-ULP f32 Cephes poly — guards the *formula*
    /// (not just scalar==avx2). Positions are the row indices, as in the real op.
    #[test]
    fn rope_bwd_matches_f64_reference() {
        let (rows, half) = (12usize, 96usize);
        let dim = 2 * half;
        let g = fill_g(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut got = vec![0.0f32; rows * dim];
        unsafe {
            wukong_rope_bwd_f32(
                g.as_ptr(),
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
                let a = g[off + j] as f64;
                let b = g[off + j + half] as f64;
                // Inverse (transpose) rotation: dx_lo = a*c + b*s, dx_hi = b*c - a*s.
                let want_lo = a * c + b * s;
                let want_hi = b * c - a * s;
                let e_lo = (got[off + j] as f64 - want_lo).abs();
                let e_hi = (got[off + j + half] as f64 - want_hi).abs();
                max_err = max_err.max(e_lo).max(e_hi);
            }
        }
        // Unit-scale data, ~1-ULP f32 trig ⇒ comfortably under 1e-4.
        assert!(
            max_err < 1e-4,
            "rope_bwd vs f64 reference: max abs err {max_err} too large"
        );
    }

    /// 4) forward∘backward round-trip: applying the RoPE *forward* (the `+theta` rotation) and then
    /// this backward (the `-theta` inverse rotation) with the SAME `inv_freq`/positions returns the
    /// original (within f32 tolerance) — the defining property of the transpose-rotation gradient,
    /// `rope_bwd(rope_fwd(x)) ≈ x`.
    #[test]
    fn forward_then_backward_round_trips() {
        let (rows, half) = (10usize, 88usize);
        let dim = 2 * half;
        let x = fill_g(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut fwd = vec![0.0f32; rows * dim];
        let mut back = vec![0.0f32; rows * dim];
        unsafe {
            // forward: y = R(+theta) x
            crate::wukong_rope_f32(
                x.as_ptr(),
                inv_freq.as_ptr(),
                fwd.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
            // backward: dx = Rᵀ(theta) y = R(-theta) y ≈ x
            wukong_rope_bwd_f32(
                fwd.as_ptr(),
                inv_freq.as_ptr(),
                back.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
        }
        let mut max_err = 0.0f64;
        for i in 0..rows * dim {
            max_err = max_err.max((back[i] as f64 - x[i] as f64).abs());
        }
        // Two rotations + four roundings per element on unit-scale data ⇒ well under 1e-4.
        assert!(
            max_err < 1e-4,
            "rope_bwd(rope_fwd(x)) != x: max abs err {max_err} too large"
        );
    }

    /// 5) norm-preservation: the backward is a 2×2 (inverse) rotation, so each pair's `a² + b²` is
    /// preserved. Check `dx[j]² + dx[j+half]² ≈ g[j]² + g[j+half]²` per pair (f32 tolerance).
    #[test]
    fn rope_bwd_preserves_pair_norm() {
        let (rows, half) = (9usize, 80usize);
        let dim = 2 * half;
        let g = fill_g(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut dx = vec![0.0f32; rows * dim];
        unsafe {
            wukong_rope_bwd_f32(
                g.as_ptr(),
                inv_freq.as_ptr(),
                dx.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
        }
        for row in 0..rows {
            let off = row * dim;
            for j in 0..half {
                let a = g[off + j];
                let b = g[off + j + half];
                let oa = dx[off + j];
                let ob = dx[off + j + half];
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

    /// 6) in-place (`dx == g`) equals the out-of-place result — the backward reads both of a pair's
    /// inputs before writing either output, so aliasing is sound.
    #[test]
    fn rope_bwd_in_place_matches_out_of_place() {
        let (rows, half) = (6usize, 70usize);
        let dim = 2 * half;
        let g = fill_g(rows * dim);
        let inv_freq = fill_inv_freq(half);
        let mut out = vec![0.0f32; rows * dim];
        let mut inplace = g.clone();
        unsafe {
            wukong_rope_bwd_f32(
                g.as_ptr(),
                inv_freq.as_ptr(),
                out.as_mut_ptr(),
                rows as i64,
                half as i64,
            );
            // dx aliases g.
            wukong_rope_bwd_f32(
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
        let g = fill_g(8);
        let mut dx = vec![42.0f32; 8];
        unsafe {
            wukong_rope_bwd_f32(g.as_ptr(), inv_freq.as_ptr(), dx.as_mut_ptr(), 0, 4);
            wukong_rope_bwd_f32(g.as_ptr(), inv_freq.as_ptr(), dx.as_mut_ptr(), 2, 0);
            wukong_rope_bwd_f32_parallel(g.as_ptr(), inv_freq.as_ptr(), dx.as_mut_ptr(), -1, 4);
        }
        assert!(dx.iter().all(|&v| v == 42.0), "no-op must not write dx");
    }
}
