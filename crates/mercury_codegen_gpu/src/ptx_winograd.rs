//! **Winograd** convolution F(2×2,3×3) and F(4×4,3×3) — a *verified f64 CPU reference*.
//!
//! This module is the correctness oracle the GPU Winograd kernel will be built on top of. It is
//! pure `f64` Rust with **no GPU symbols**, so a plain `cargo test` (no `--features gpu`) compiles
//! and checks it. The whole point is the `#[cfg(test)]` differential against the naive direct conv:
//! in f64, Winograd is exact up to floating rounding, so `winograd ≈ direct` to < 1e-9.
//!
//! Same op the rest of the crate's conv2d calls: single batch, stride 1, no padding, **valid
//! cross-correlation**. Input `X[C,H,W]` (row-major), weights `W[K,C,R,S]` with `R=S=3`, output
//! `O[K,P,Q]` with `P=H-2`, `Q=W-2`.
//!
//! ## Winograd math (wincnn convention — the B/G/A sets below are *self-consistent*)
//!
//! For one output tile and one output channel `k`:
//! ```text
//!   M = Σ_c (G · g_kc · Gᵀ) ⊙ (Bᵀ · d_c · B)          // α×α accumulation over channels c
//!   Y = Aᵀ · M · A                                     // single inverse transform → m×m
//! ```
//! where `g_kc` is the 3×3 filter for `(k,c)`, `d_c` the α×α input tile for channel `c`, ⊙ is
//! elementwise. The channel reduction happens in the **transform domain** (sum of `U⊙V` over `c`),
//! then *one* inverse transform per `(k, tile)` — that amortization is the Winograd win.
//!
//! * F(2×2,3×3): `α=4`, `m=2`, input tiles 4×4 → output tiles 2×2.
//! * F(4×4,3×3): `α=6`, `m=4`, input tiles 6×6 → output tiles 4×4.

// ---------------------------------------------------------------------------------------------
// Linear-algebra helpers (row-major, private).
// ---------------------------------------------------------------------------------------------

/// Row-major matrix product `c[m×n] = a[m×k] · b[k×n]`.
fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            let brow = &b[p * n..p * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += aip * brow[j];
            }
        }
    }
    c
}

/// Transpose a row-major `a[rows×cols]` into `out[cols×rows]`.
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), rows * cols);
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Winograd constant matrices (wincnn convention).
// ---------------------------------------------------------------------------------------------

/// F(2×2,3×3): `Bᵀ` (4×4), row-major.
#[rustfmt::skip]
const F23_BT: [f64; 16] = [
    1.0,  0.0, -1.0,  0.0,
    0.0,  1.0,  1.0,  0.0,
    0.0, -1.0,  1.0,  0.0,
    0.0, -1.0,  0.0,  1.0,
];

/// F(2×2,3×3): `G` (4×3), row-major.
#[rustfmt::skip]
const F23_G: [f64; 12] = [
    1.0,   0.0,   0.0,
    0.5,   0.5,   0.5,
    0.5,  -0.5,   0.5,
    0.0,   0.0,   1.0,
];

/// F(2×2,3×3): `Aᵀ` (2×4), row-major.
#[rustfmt::skip]
const F23_AT: [f64; 8] = [
    1.0,  1.0,  1.0,  0.0,
    0.0,  1.0, -1.0,  1.0,
];

/// F(4×4,3×3): `Bᵀ` (6×6), row-major.
#[rustfmt::skip]
const F43_BT: [f64; 36] = [
    4.0,  0.0, -5.0,  0.0,  1.0,  0.0,
    0.0, -4.0, -4.0,  1.0,  1.0,  0.0,
    0.0,  4.0, -4.0, -1.0,  1.0,  0.0,
    0.0, -2.0, -1.0,  2.0,  1.0,  0.0,
    0.0,  2.0, -1.0, -2.0,  1.0,  0.0,
    0.0,  4.0,  0.0, -5.0,  0.0,  1.0,
];

/// F(4×4,3×3): `G` (6×3), row-major.
#[rustfmt::skip]
const F43_G: [f64; 18] = [
     1.0 / 4.0,    0.0,          0.0,
    -1.0 / 6.0,   -1.0 / 6.0,   -1.0 / 6.0,
    -1.0 / 6.0,    1.0 / 6.0,   -1.0 / 6.0,
     1.0 / 24.0,   1.0 / 12.0,   1.0 / 6.0,
     1.0 / 24.0,  -1.0 / 12.0,   1.0 / 6.0,
     0.0,          0.0,          1.0,
];

/// F(4×4,3×3): `Aᵀ` (4×6), row-major.
#[rustfmt::skip]
const F43_AT: [f64; 24] = [
    1.0,  1.0,  1.0,  1.0,  1.0,  0.0,
    0.0,  1.0, -1.0,  2.0, -2.0,  0.0,
    0.0,  1.0,  1.0,  4.0,  4.0,  0.0,
    0.0,  1.0, -1.0,  8.0, -8.0,  1.0,
];

// ---------------------------------------------------------------------------------------------
// Direct (naive) reference convolution — R=S=3 baked in.
// ---------------------------------------------------------------------------------------------

/// Naive triple-loop valid cross-correlation, `R=S=3` baked in.
///
/// `x` = `X[C,H,W]`, `w` = `W[K,C,3,3]`, returns `O[K,P,Q]` with `P=H-2`, `Q=W-2` (row-major).
pub fn direct_conv(x: &[f64], w: &[f64], c: usize, h: usize, wd: usize, k: usize) -> Vec<f64> {
    let p = h - 2;
    let q = wd - 2;
    debug_assert_eq!(x.len(), c * h * wd);
    debug_assert_eq!(w.len(), k * c * 3 * 3);
    let mut o = vec![0.0f64; k * p * q];
    for kk in 0..k {
        for oi in 0..p {
            for oj in 0..q {
                let mut acc = 0.0f64;
                for cc in 0..c {
                    let xbase = cc * h * wd;
                    let wbase = ((kk * c) + cc) * 9;
                    for r in 0..3 {
                        for s in 0..3 {
                            acc += x[xbase + (oi + r) * wd + (oj + s)] * w[wbase + r * 3 + s];
                        }
                    }
                }
                o[(kk * p + oi) * q + oj] = acc;
            }
        }
    }
    o
}

// ---------------------------------------------------------------------------------------------
// Winograd driver + the two public entry points.
// ---------------------------------------------------------------------------------------------

/// Generic Winograd F(m×m,3×3) driver.
///
/// * `bt` — `Bᵀ`, the α×α input-transform matrix (row-major), where `α = m + 2`.
/// * `g`  — `G`, the α×3 filter-transform matrix (row-major).
/// * `at` — `Aᵀ`, the m×α output (inverse) transform matrix (row-major).
///
/// Per output tile: (1) compute the input transforms `V_c = Bᵀ d_c B` (α×α) for all channels;
/// (2) for each output channel `k`, reduce `M = Σ_c U_kc ⊙ V_c` in the transform domain (with
/// the precomputed filter transforms `U_kc = G g_kc Gᵀ`); (3) apply the single inverse
/// `Y = Aᵀ M A` (m×m) and scatter into `O`. Out-of-range input reads are `0.0`; out-of-range
/// output writes are skipped (right/bottom edge tiles).
fn winograd_fm3(
    x: &[f64],
    w: &[f64],
    c: usize,
    h: usize,
    wd: usize,
    k: usize,
    m: usize,
    bt: &[f64],
    g: &[f64],
    at: &[f64],
) -> Vec<f64> {
    let alpha = m + 2;
    debug_assert_eq!(bt.len(), alpha * alpha);
    debug_assert_eq!(g.len(), alpha * 3);
    debug_assert_eq!(at.len(), m * alpha);
    debug_assert_eq!(x.len(), c * h * wd);
    debug_assert_eq!(w.len(), k * c * 3 * 3);

    let p = h - 2;
    let q = wd - 2;
    let aa = alpha * alpha;
    let mut o = vec![0.0f64; k * p * q];

    // Transposes used by the products: B = (Bᵀ)ᵀ (α×α), Gᵀ (3×α), A = (Aᵀ)ᵀ (α×m).
    let b = transpose(bt, alpha, alpha);
    let gt = transpose(g, alpha, 3);
    let a = transpose(at, m, alpha);

    // Precompute the filter transforms `U_kc = G · g_kc · Gᵀ` (α×α) for every (k,c).
    // Layout: u[(kk*c + cc) * aa ..].
    let mut u = vec![0.0f64; k * c * aa];
    for kk in 0..k {
        for cc in 0..c {
            let wbase = ((kk * c) + cc) * 9;
            let gk = &w[wbase..wbase + 9]; // 3×3 filter
            let gg = matmul(g, alpha, 3, gk, 3); // G·g  → α×3
            let ggt = matmul(&gg, alpha, 3, &gt, alpha); // (G·g)·Gᵀ → α×α
            let dst = ((kk * c) + cc) * aa;
            u[dst..dst + aa].copy_from_slice(&ggt);
        }
    }

    // Number of output tiles along each axis (ceil-div: edge tiles allowed, clipped on write).
    let n_ti = p.div_ceil(m);
    let n_tj = q.div_ceil(m);

    // Scratch reused across tiles.
    let mut tile = vec![0.0f64; aa]; // raw input tile d_c
    let mut vbuf = vec![0.0f64; c * aa]; // V_c for every channel of the current tile
    let mut acc = vec![0.0f64; aa]; // M = Σ_c U_kc ⊙ V_c

    for ti in 0..n_ti {
        for tj in 0..n_tj {
            let in_i0 = ti * m; // top-left input row of this tile
            let in_j0 = tj * m; // top-left input col of this tile

            // (1) Input transforms V_c = Bᵀ d_c B for every channel (k-independent).
            for cc in 0..c {
                let xbase = cc * h * wd;
                // Gather the α×α input tile, zero-padding out-of-range reads.
                for a_i in 0..alpha {
                    let gi = in_i0 + a_i;
                    for a_j in 0..alpha {
                        let gj = in_j0 + a_j;
                        tile[a_i * alpha + a_j] = if gi < h && gj < wd {
                            x[xbase + gi * wd + gj]
                        } else {
                            0.0
                        };
                    }
                }
                let btd = matmul(bt, alpha, alpha, &tile, alpha); // Bᵀ·d
                let v = matmul(&btd, alpha, alpha, &b, alpha); // (Bᵀ·d)·B → α×α
                vbuf[cc * aa..cc * aa + aa].copy_from_slice(&v);
            }

            // (2)+(3) Per output channel: transform-domain reduction over c, then inverse.
            for kk in 0..k {
                for val in acc.iter_mut() {
                    *val = 0.0;
                }
                for cc in 0..c {
                    let ubase = ((kk * c) + cc) * aa;
                    let vbase = cc * aa;
                    for e in 0..aa {
                        acc[e] += u[ubase + e] * vbuf[vbase + e];
                    }
                }
                // Y = Aᵀ · M · A  → m×m
                let atm = matmul(at, m, alpha, &acc, alpha); // Aᵀ·M → m×α
                let y = matmul(&atm, m, alpha, &a, m); // (Aᵀ·M)·A → m×m

                // Scatter, clipping out-of-range output writes (edge tiles).
                let out_i0 = ti * m;
                let out_j0 = tj * m;
                for yi in 0..m {
                    let oi = out_i0 + yi;
                    if oi >= p {
                        continue;
                    }
                    for yj in 0..m {
                        let oj = out_j0 + yj;
                        if oj >= q {
                            continue;
                        }
                        o[(kk * p + oi) * q + oj] = y[yi * m + yj];
                    }
                }
            }
        }
    }

    o
}

/// F(2×2,3×3) Winograd convolution (α=4, output tiles 2×2). See [`winograd_fm3`].
pub fn winograd_f23(x: &[f64], w: &[f64], c: usize, h: usize, wd: usize, k: usize) -> Vec<f64> {
    winograd_fm3(x, w, c, h, wd, k, 2, &F23_BT, &F23_G, &F23_AT)
}

/// F(4×4,3×3) Winograd convolution (α=6, output tiles 4×4). See [`winograd_fm3`].
pub fn winograd_f43(x: &[f64], w: &[f64], c: usize, h: usize, wd: usize, k: usize) -> Vec<f64> {
    winograd_fm3(x, w, c, h, wd, k, 4, &F43_BT, &F43_G, &F43_AT)
}

// ---------------------------------------------------------------------------------------------
// Differential test: Winograd must match the naive direct conv to < 1e-9 in f64.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic counter-based pattern in roughly [-0.5, 0.5) (no `rand` crate).
    fn fill(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i + seed;
                ((t * 7 + 3) % 13) as f64 / 13.0 - 0.5
            })
            .collect()
    }

    /// Max absolute elementwise error between two equal-length slices.
    fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "shape mismatch between references");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn winograd_matches_direct() {
        // (C, H, W, K). Includes a non-tile-divisible case (9×9 → P=Q=7, multiple of neither 2 nor 4).
        let shapes = [
            (1usize, 6usize, 6usize, 1usize),
            (3, 8, 8, 4),
            (2, 9, 9, 3),
            (4, 16, 16, 8),
            (1, 5, 5, 1),
        ];

        let tol = 1e-9;
        let mut worst_f23 = 0.0f64;
        let mut worst_f43 = 0.0f64;

        for (idx, &(c, h, wd, k)) in shapes.iter().enumerate() {
            assert!(h >= 3 && wd >= 3, "need a valid 3×3 conv");
            let x = fill(c * h * wd, idx * 101 + 1);
            let w = fill(k * c * 3 * 3, idx * 211 + 7);

            let direct = direct_conv(&x, &w, c, h, wd, k);
            let wino23 = winograd_f23(&x, &w, c, h, wd, k);
            let wino43 = winograd_f43(&x, &w, c, h, wd, k);

            let e23 = max_abs_err(&direct, &wino23);
            let e43 = max_abs_err(&direct, &wino43);
            worst_f23 = worst_f23.max(e23);
            worst_f43 = worst_f43.max(e43);

            println!(
                "shape (C,H,W,K)=({c},{h},{wd},{k})  P={} Q={}  f23_err={e23:.3e}  f43_err={e43:.3e}",
                h - 2,
                wd - 2
            );

            assert!(
                e23 < tol,
                "F(2×2,3×3) mismatch on (C,H,W,K)=({c},{h},{wd},{k}): max_abs_err={e23:.3e} >= {tol:.0e}"
            );
            assert!(
                e43 < tol,
                "F(4×4,3×3) mismatch on (C,H,W,K)=({c},{h},{wd},{k}): max_abs_err={e43:.3e} >= {tol:.0e}"
            );
        }

        println!("WORST f23_err={worst_f23:.3e}  WORST f43_err={worst_f43:.3e}  (tol={tol:.0e})");
    }

    /// Spot-check the helpers in isolation.
    #[test]
    fn matmul_and_transpose_sane() {
        // [2×3]·[3×2]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul(&a, 2, 3, &b, 2);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);

        let t = transpose(&a, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
