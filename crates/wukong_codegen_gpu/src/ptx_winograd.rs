//! **Winograd** convolution F(2×2,3×3) and F(4×4,3×3) — a *verified f64 CPU reference* plus the GPU
//! transform / batched-GEMM PTX generators built on it.
//!
//! [`direct_conv`] / [`winograd_f23`] / [`winograd_f43`] are the correctness oracle: pure `f64` Rust
//! with **no device calls** (nothing here touches `cudarc`), so every gate in this file runs without a
//! GPU — though the module itself is still `#[cfg(feature = "gpu")]` in `lib.rs`, so a plain
//! `cargo test` does not compile it at all; use `cargo test -p wukong_codegen_gpu --features gpu`.
//! The core check is the `#[cfg(test)]` differential against the naive direct conv: in f64, Winograd
//! is exact up to floating rounding, so `winograd ≈ direct` to < 1e-9. The four
//! `wino_*_ptx` generators below emit the device pipeline (filter/input transform → batched GEMM in
//! the transform domain → output transform) against that same math.
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

//! ## Shared memory
//!
//! Three of the four generators use **none**: the filter / input / output transforms are one thread per
//! `(k,c)` or `(c,tile)` with the whole `α×α` working set in registers. All of this family's shared
//! memory is [`wino_bgemm_ptx`]'s GEMM tile, whose closed form is [`wino_bgemm_smem_bytes`] — the same
//! `BM·16·2 + 16·BN·2 + BM·BN·4` as the implicit-GEMM conv it borrows [`crate::ptx_conv::WMMA_BM`] /
//! [`crate::ptx_conv::WMMA_BN`] from, i.e. **20480 B, constant for every shape**, 42 % of the PTX ISA's
//! static `.shared` cap. [`wino_bgemm_ptx_budget`] takes the budget the dispatch layer probed and
//! returns the [`crate::gpu::SmemMode`] its launch must honour.
//!
//! **A bigger budget is not the lever this family is missing, and the arithmetic says so.** The bgemm's
//! reduction length is `GK = C`, the channel count itself, against a 16-wide WMMA K-tile: at `C=3` a
//! whole staged tile is 13/16 zero padding, so Winograd's 2.25–4× multiply reduction is spent on
//! padding before it reaches the tensor cores. That is the mechanism behind "F(4×4,3×3) loses at low
//! channel count", and no amount of shared memory changes it — the fixes are a `C`-packed bgemm (batch
//! several transform planes into one 16-deep K-tile) or a narrower MMA shape.

// The module header for all four generators, from the single source (`crate::ptx_target`). The
// Winograd pipeline emits only f32/f16 arithmetic, shared memory and `wmma` fragments — every one of
// them legal at the `sm_80` floor. PTX is forward-compatible only, so the module is tagged with that
// floor, never with the device's own arch (an `sm_89` tag loads on ZERO A100s).
use crate::gpu::{smem_mode_for, SmemMode, STATIC_SMEM_CAP};
use crate::ptx_target::HDR_SM80;

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

// ===================================================================================================
// GPU Winograd — the **non-fused, tensor-core** path (cuDNN's `WINOGRAD_NONFUSED` strategy).
// ===================================================================================================
//
// Four phases over global buffers (fp16 storage, f32 transform arithmetic), tolerance-gated against the
// f64 `winograd_f43`/`winograd_f23` reference above:
//
//   1. **filter transform**  U[ξν,k,c] = (G g_kc Gᵀ)[ξν]      — one thread per (k,c); one-time (weights).
//   2. **input transform**   V[ξν,c,t] = (Bᵀ d_{c,t} B)[ξν]   — one thread per (c, output-tile t).
//   3. **batched GEMM**      M[ξν,k,t] = Σ_c U[ξν,k,c]·V[ξν,c,t] — α² independent [K×C]·[C×T] GEMMs,
//      all in ONE launch by [`wino_bgemm_ptx`] (`z = ctaid.z` selects the plane): a plain row-major NN
//      fp16 `m16n16k16` WMMA GEMM sharing `conv2d_wmma`'s tensor-core core and tile constants, but with
//      no im2col gather and the plane loop folded into `gridDim.z` rather than run host-side. This is
//      where the FLOPs are, and where Winograd's 2.25–4× multiply reduction pays — the transforms are
//      cheap elementwise maps.
//   4. **output transform**  O[k,p,q] = (Aᵀ M_{k,t} A)[..]    — one thread per (k, tile), scatter.
//
// Each transform `out = Mat · in` is a constant linear map (the B/G/A matrices are sparse integers /
// dyadic fractions), so the kernels are a load → unrolled FMA chain (nonzero coefficients only) → store,
// computed in f32 to keep fp16's ~1e-3 quantization the only error. The buffer layouts are chosen so
// phase 3 is a plain row-major `W[K,C]·X[C,T]`: U is [ξν][K][C], V is [ξν][C][T], M is [ξν][K][T]
// (M is f32 — it is conv2d_wmma's f32 accumulator output, read directly by the output transform).

/// f32 hex-bits literal for a PTX immediate (`0fXXXXXXXX`).
fn f32_hex(v: f64) -> String {
    format!("0f{:08X}", (v as f32).to_bits())
}

/// The (Bᵀ, G, Aᵀ, α) constant set for output-tile size `m` (2 → F(2,3), 4 → F(4,3)).
fn wino_consts(m: usize) -> (&'static [f64], &'static [f64], &'static [f64], usize) {
    match m {
        2 => (&F23_BT, &F23_G, &F23_AT, 4),
        4 => (&F43_BT, &F43_G, &F43_AT, 6),
        _ => panic!("Winograd output tile m={m} unsupported (use 2 or 4)"),
    }
}

/// Filter-transform map `U_flat[α²] = Mf · g_flat[9]`, where `Mf[(ξ,ν)][(i,j)] = G[ξ,i]·G[ν,j]`
/// (so `U[ξ,ν] = Σ_ij G[ξ,i] g[i,j] G[ν,j] = (G g Gᵀ)[ξ,ν]`). Row-major `α²×9`.
fn filter_map(g: &[f64], alpha: usize) -> Vec<f64> {
    let mut mf = vec![0.0f64; alpha * alpha * 9];
    for xi in 0..alpha {
        for nu in 0..alpha {
            for i in 0..3 {
                for j in 0..3 {
                    mf[(xi * alpha + nu) * 9 + (i * 3 + j)] = g[xi * 3 + i] * g[nu * 3 + j];
                }
            }
        }
    }
    mf
}

/// Input-transform map `V_flat[α²] = Mi · d_flat[α²]`, `Mi[(ξ,ν)][(a,b)] = Bᵀ[ξ,a]·Bᵀ[ν,b]`
/// (so `V[ξ,ν] = Σ_ab Bᵀ[ξ,a] d[a,b] Bᵀ[ν,b] = (Bᵀ d B)[ξ,ν]`). Row-major `α²×α²`.
fn input_map(bt: &[f64], alpha: usize) -> Vec<f64> {
    let aa = alpha * alpha;
    let mut mi = vec![0.0f64; aa * aa];
    for xi in 0..alpha {
        for nu in 0..alpha {
            for a in 0..alpha {
                for b in 0..alpha {
                    mi[(xi * alpha + nu) * aa + (a * alpha + b)] =
                        bt[xi * alpha + a] * bt[nu * alpha + b];
                }
            }
        }
    }
    mi
}

/// Output-transform map `Y_flat[m²] = Mo · M_flat[α²]`, `Mo[(yi,yj)][(ξ,ν)] = Aᵀ[yi,ξ]·Aᵀ[yj,ν]`
/// (so `Y[yi,yj] = Σ_ξν Aᵀ[yi,ξ] M[ξ,ν] Aᵀ[yj,ν] = (Aᵀ M A)[yi,yj]`). Row-major `m²×α²`.
fn output_map(at: &[f64], m: usize, alpha: usize) -> Vec<f64> {
    let aa = alpha * alpha;
    let mut mo = vec![0.0f64; m * m * aa];
    for yi in 0..m {
        for yj in 0..m {
            for xi in 0..alpha {
                for nu in 0..alpha {
                    mo[(yi * m + yj) * aa + (xi * alpha + nu)] =
                        at[yi * alpha + xi] * at[yj * alpha + nu];
                }
            }
        }
    }
    mo
}

/// Emit the FMA chain `%{acc} = Σ_j row[j]·%{pre}{j}` over the nonzero coefficients of `row` (a `mul`
/// seeds the first term so no separate zero-init; an all-zero row yields a `mov 0`).
fn emit_lincomb(s: &mut String, acc: &str, row: &[f64], pre: &str) {
    use std::fmt::Write as _;
    let mut first = true;
    for (j, &co) in row.iter().enumerate() {
        if co.abs() < 1e-12 {
            continue;
        }
        if first {
            let _ = writeln!(s, "    mul.f32 {acc},%{pre}{j},{};", f32_hex(co));
            first = false;
        } else {
            let _ = writeln!(s, "    fma.rn.f32 {acc},%{pre}{j},{},{acc};", f32_hex(co));
        }
    }
    if first {
        let _ = writeln!(s, "    mov.f32 {acc},0f00000000;");
    }
}

/// Number of Winograd output tiles for `[H,W]`, output-tile size `m`: `ceil((H-2)/m)·ceil((W-2)/m)`.
pub fn wino_ntiles(h: usize, w: usize, m: usize) -> (usize, usize, usize) {
    let (p, q) = (h - 2, w - 2);
    let (nti, ntj) = (p.div_ceil(m), q.div_ceil(m));
    (nti, ntj, nti * ntj)
}

/// **Phase 1 — filter transform.** `wino_filter_xform(pW, pU)`: `W[K,C,3,3]` (f16) → `U[α²,K,C]` (f16),
/// `U[ξν,k,c] = (G g_kc Gᵀ)[ξν]`. One thread per `(k,c)` (1-D grid over `K·C`); loads the 9 filter taps,
/// widens to f32, runs the `α²×9` constant map, stores `α²` outputs `U[ξν·KC + tid]`.
pub fn wino_filter_xform_ptx(c: usize, k: usize, m: usize) -> String {
    use std::fmt::Write as _;
    let (_, g, _, alpha) = wino_consts(m);
    let aa = alpha * alpha;
    let mf = filter_map(g, alpha);
    let kc = k * c;
    let mut s = String::new();
    let _ = writeln!(s, "{HDR_SM80}");
    let _ = writeln!(
        s,
        "// Winograd F({m},3) filter transform: W[K{k},C{c},3,3] -> U[{aa},K,C]"
    );
    let _ = writeln!(
        s,
        ".visible .entry wino_filter_xform(\n    .param .u64 pW,\n    .param .u64 pU\n)\n{{"
    );
    let _ = writeln!(s, "    .reg .pred %p0;");
    let _ = writeln!(s, "    .reg .b16 %h;");
    let _ = writeln!(s, "    .reg .b32 %gid,%t,%n,%i;");
    let _ = writeln!(s, "    .reg .f32 {};", {
        let mut d = String::from("%acc");
        for j in 0..9 {
            d += &format!(",%in{j}");
        }
        d
    });
    let _ = writeln!(s, "    .reg .b64 %W,%U,%off,%base,%ptr;");
    let _ = writeln!(s, "    ld.param.u64 %W,[pW];\n    ld.param.u64 %U,[pU];");
    let _ = writeln!(
        s,
        "    cvta.to.global.u64 %W,%W;\n    cvta.to.global.u64 %U,%U;"
    );
    let _ = writeln!(
        s,
        "    mov.u32 %t,%ctaid.x;\n    mov.u32 %n,%ntid.x;\n    mov.u32 %i,%tid.x;"
    );
    let _ = writeln!(s, "    mad.lo.s32 %gid,%t,%n,%i;");
    let _ = writeln!(s, "    setp.ge.u32 %p0,%gid,{kc};\n    @%p0 bra RET;");
    // load 9 taps: W[tid*9 + j]
    let _ = writeln!(s, "    mul.lo.s32 %t,%gid,9;");
    let _ = writeln!(s, "    mul.wide.u32 %off,%t,2;\n    add.s64 %base,%W,%off;");
    for j in 0..9 {
        let _ = writeln!(
            s,
            "    ld.global.u16 %h,[%base+{}];\n    cvt.f32.f16 %in{j},%h;",
            j * 2
        );
    }
    // U base for this thread: pU + tid*2 ; per-xi store adds xi*KC*2 (constant)
    let _ = writeln!(
        s,
        "    mul.wide.u32 %off,%gid,2;\n    add.s64 %base,%U,%off;"
    );
    for e in 0..aa {
        emit_lincomb(&mut s, "%acc", &mf[e * 9..e * 9 + 9], "in");
        let _ = writeln!(
            s,
            "    cvt.rn.f16.f32 %h,%acc;\n    st.global.u16 [%base+{}],%h;",
            e * kc * 2
        );
    }
    let _ = writeln!(s, "RET:\n    ret;\n}}");
    s
}

/// **Phase 2 — input transform.** `wino_input_xform(pX, pV)`: `X[C,H,W]` (f16) → `V[α²,C,T]` (f16),
/// `V[ξν,c,t] = (Bᵀ d_{c,t} B)[ξν]` with `T` = number of output tiles. One thread per `(c, tile)`
/// (1-D grid over `C·T`); gathers the `α×α` input tile (zero-padding out-of-range edge reads), widens to
/// f32, runs the `α²×α²` constant map, stores `V[ξν·CT + tid]`.
pub fn wino_input_xform_ptx(c: usize, h: usize, w: usize, m: usize) -> String {
    use std::fmt::Write as _;
    let (bt, _, _, alpha) = wino_consts(m);
    let aa = alpha * alpha;
    let mi = input_map(bt, alpha);
    let (_nti, ntj, nt) = wino_ntiles(h, w, m);
    let ct = c * nt;
    let hw = h * w;
    let mut s = String::new();
    let _ = writeln!(s, "{HDR_SM80}");
    let _ = writeln!(
        s,
        "// Winograd F({m},3) input transform: X[C{c},H{h},W{w}] -> V[{aa},C,T{nt}]"
    );
    let _ = writeln!(
        s,
        ".visible .entry wino_input_xform(\n    .param .u64 pX,\n    .param .u64 pV\n)\n{{"
    );
    let _ = writeln!(s, "    .reg .pred %p0,%pi,%pj;");
    let _ = writeln!(s, "    .reg .b16 %h;");
    let _ = writeln!(
        s,
        "    .reg .b32 %gid,%t,%n,%i,%cc,%tile,%ti,%tj,%i0,%j0,%gi,%gj,%xb,%idx;"
    );
    let _ = writeln!(s, "    .reg .f32 {};", {
        let mut d = String::from("%acc");
        for e in 0..aa {
            d += &format!(",%in{e}");
        }
        d
    });
    let _ = writeln!(s, "    .reg .b64 %X,%V,%off,%base,%ptr;");
    let _ = writeln!(s, "    ld.param.u64 %X,[pX];\n    ld.param.u64 %V,[pV];");
    let _ = writeln!(
        s,
        "    cvta.to.global.u64 %X,%X;\n    cvta.to.global.u64 %V,%V;"
    );
    let _ = writeln!(
        s,
        "    mov.u32 %t,%ctaid.x;\n    mov.u32 %n,%ntid.x;\n    mov.u32 %i,%tid.x;"
    );
    let _ = writeln!(s, "    mad.lo.s32 %gid,%t,%n,%i;");
    let _ = writeln!(s, "    setp.ge.u32 %p0,%gid,{ct};\n    @%p0 bra RET;");
    // c = tid/nt ; tile = tid%nt ; ti = tile/ntj ; tj = tile%ntj
    let _ = writeln!(
        s,
        "    div.u32 %cc,%gid,{nt};\n    rem.u32 %tile,%gid,{nt};"
    );
    let _ = writeln!(
        s,
        "    div.u32 %ti,%tile,{ntj};\n    rem.u32 %tj,%tile,{ntj};"
    );
    let _ = writeln!(
        s,
        "    mul.lo.s32 %i0,%ti,{m};\n    mul.lo.s32 %j0,%tj,{m};"
    );
    let _ = writeln!(s, "    mul.lo.s32 %xb,%cc,{hw};        // c*H*W");
    // gather d[a][b] -> %in{a*alpha+b}, zero-pad OOB
    for a in 0..alpha {
        let _ = writeln!(s, "    add.s32 %gi,%i0,{a};\n    setp.lt.u32 %pi,%gi,{h};");
        for b in 0..alpha {
            let e = a * alpha + b;
            let _ = writeln!(
                s,
                "    add.s32 %gj,%j0,{b};\n    setp.lt.u32 %pj,%gj,{w};\n    and.pred %pj,%pj,%pi;"
            );
            let _ = writeln!(
                s,
                "    mad.lo.s32 %idx,%gi,{w},%xb;\n    add.s32 %idx,%idx,%gj;"
            );
            let _ = writeln!(
                s,
                "    mul.wide.u32 %off,%idx,2;\n    add.s64 %ptr,%X,%off;"
            );
            let _ = writeln!(
                s,
                "    mov.u16 %h,0;\n    @%pj ld.global.u16 %h,[%ptr];\n    cvt.f32.f16 %in{e},%h;"
            );
        }
    }
    // V base: pV + tid*2 ; per-xi store adds xi*CT*2
    let _ = writeln!(
        s,
        "    mul.wide.u32 %off,%gid,2;\n    add.s64 %base,%V,%off;"
    );
    for e in 0..aa {
        emit_lincomb(&mut s, "%acc", &mi[e * aa..e * aa + aa], "in");
        let _ = writeln!(
            s,
            "    cvt.rn.f16.f32 %h,%acc;\n    st.global.u16 [%base+{}],%h;",
            e * ct * 2
        );
    }
    let _ = writeln!(s, "RET:\n    ret;\n}}");
    s
}

/// **Phase 4 — output transform.** `wino_output_xform(pM, pO)`: `M[α²,K,T]` (f32) → `O[K,P,Q]` (f32),
/// `O[k,·] = (Aᵀ M_{k,t} A)`. One thread per `(k, tile)` (1-D grid over `K·T`); gathers the `α²`
/// transform-domain values `M[ξν·KT + tid]`, widens to f32, runs the `m²×α²` constant map, scatters the
/// `m×m` outputs into `O` (clipping the right/bottom edge tiles).
pub fn wino_output_xform_ptx(k: usize, h: usize, w: usize, m: usize) -> String {
    use std::fmt::Write as _;
    let (_, _, at, alpha) = wino_consts(m);
    let aa = alpha * alpha;
    let mo = output_map(at, m, alpha);
    let (p, q) = (h - 2, w - 2);
    let (_nti, ntj, nt) = wino_ntiles(h, w, m);
    let kt = k * nt;
    let mut s = String::new();
    let _ = writeln!(s, "{HDR_SM80}");
    let _ = writeln!(
        s,
        "// Winograd F({m},3) output transform: M[{aa},K{k},T{nt}] -> O[K,P{p},Q{q}]"
    );
    let _ = writeln!(
        s,
        ".visible .entry wino_output_xform(\n    .param .u64 pM,\n    .param .u64 pO\n)\n{{"
    );
    let _ = writeln!(s, "    .reg .pred %p0,%pi,%pj;");
    let _ = writeln!(s, "    .reg .b16 %h;");
    let _ = writeln!(
        s,
        "    .reg .b32 %gid,%t,%n,%i,%kk,%tile,%ti,%tj,%i0,%j0,%oi,%oj,%idx;"
    );
    let _ = writeln!(s, "    .reg .f32 {};", {
        let mut d = String::from("%acc");
        for e in 0..aa {
            d += &format!(",%in{e}");
        }
        d
    });
    let _ = writeln!(s, "    .reg .b64 %M,%O,%off,%base,%ptr;");
    let _ = writeln!(s, "    ld.param.u64 %M,[pM];\n    ld.param.u64 %O,[pO];");
    let _ = writeln!(
        s,
        "    cvta.to.global.u64 %M,%M;\n    cvta.to.global.u64 %O,%O;"
    );
    let _ = writeln!(
        s,
        "    mov.u32 %t,%ctaid.x;\n    mov.u32 %n,%ntid.x;\n    mov.u32 %i,%tid.x;"
    );
    let _ = writeln!(s, "    mad.lo.s32 %gid,%t,%n,%i;");
    let _ = writeln!(s, "    setp.ge.u32 %p0,%gid,{kt};\n    @%p0 bra RET;");
    let _ = writeln!(
        s,
        "    div.u32 %kk,%gid,{nt};\n    rem.u32 %tile,%gid,{nt};"
    );
    let _ = writeln!(
        s,
        "    div.u32 %ti,%tile,{ntj};\n    rem.u32 %tj,%tile,{ntj};"
    );
    let _ = writeln!(
        s,
        "    mul.lo.s32 %i0,%ti,{m};\n    mul.lo.s32 %j0,%tj,{m};"
    );
    // gather M_flat[xi] = pM[xi*KT + tid]. M is **f32** (it is conv2d_wmma's f32 output), so a 4-byte
    // stride and a direct f32 load — no f16 widening.
    let _ = writeln!(
        s,
        "    mul.wide.u32 %off,%gid,4;\n    add.s64 %base,%M,%off;"
    );
    for e in 0..aa {
        let _ = writeln!(s, "    ld.global.f32 %in{e},[%base+{}];", e * kt * 4);
    }
    // Y = Mo · M_flat ; scatter Y[yi*m+yj] -> O[(kk*P + oi)*Q + oj], oi=i0+yi, oj=j0+yj, clip edges
    for yi in 0..m {
        let _ = writeln!(s, "    add.s32 %oi,%i0,{yi};\n    setp.lt.u32 %pi,%oi,{p};");
        for yj in 0..m {
            let e = yi * m + yj;
            emit_lincomb(&mut s, "%acc", &mo[e * aa..e * aa + aa], "in");
            let _ = writeln!(s, "    add.s32 %oj,%j0,{yj};\n    setp.lt.u32 %pj,%oj,{q};\n    and.pred %pj,%pj,%pi;");
            let _ = writeln!(s, "    mad.lo.s32 %idx,%kk,{p},%oi;\n    mul.lo.s32 %idx,%idx,{q};\n    add.s32 %idx,%idx,%oj;");
            let _ = writeln!(
                s,
                "    mul.wide.u32 %off,%idx,4;\n    add.s64 %ptr,%O,%off;"
            );
            let _ = writeln!(s, "    @%pj st.global.f32 [%ptr],%acc;");
        }
    }
    let _ = writeln!(s, "RET:\n    ret;\n}}");
    s
}

/// **Phase 3 (batched) — the α² channel-reduction GEMMs in ONE launch.** `wino_bgemm(pV, pU, pM)`
/// computes every transform-position plane `M[z] = U[z]·V[z]` for `z = ctaid.z ∈ [0,α²)`:
/// `M[z][K,T] = U[z][K,C] · V[z][C,T]` (row-major **NN**, no im2col — the Winograd GEMM is plain). A
/// fp16 `m16n16k16` WMMA GEMM (the same tensor-core core as `conv2d_wmma`), but the per-plane loop is
/// folded into `gridDim.z`: launching the α² planes separately fills only `ceil(T/BN)·ceil(K/BM)` CTAs
/// each (≈4 on a feature map) and runs them serially — ~idle on a 20-SM GPU — whereas batching puts
/// `α²·that` CTAs in flight at once. Plane strides (U:`K·C`, V:`C·T`, M:`K·T`) are baked; `M` is **f32**
/// (read directly by the output transform). Launch: block `(WMMA_THREADS,1,1)`, grid
/// `(ceil(T/WMMA_BN), ceil(K/WMMA_BM), α²)`, `shared_mem_bytes: 0` (the tile is a static `.shared`).
pub fn wino_bgemm_ptx(c: usize, nt: usize, k: usize) -> String {
    let (ptx, mode) = wino_bgemm_ptx_budget(c, nt, k, STATIC_SMEM_CAP);
    debug_assert_eq!(
        mode,
        SmemMode::Static,
        "the static cap cannot yield a window"
    );
    ptx
}

/// **The Winograd batched GEMM's shared-memory footprint in bytes — the single source.**
///
/// `BM·16·2 + 16·BN·2 + BM·BN·4` over [`crate::ptx_conv::WMMA_BM`]/[`crate::ptx_conv::WMMA_BN`]: the
/// staged fp16 `BM×16` `U` tile and `16×BN` `V` tile plus the f32 `BM×BN` epilogue scratch. Identical to
/// the single-buffered [`crate::ptx_conv::conv_wmma_smem_bytes`] — this kernel *is* that tensor-core
/// core with the im2col gather replaced by a plain row-major load — so it is 20480 B for every shape.
pub const fn wino_bgemm_smem_bytes() -> usize {
    crate::ptx_conv::conv_wmma_smem_bytes(crate::ptx_conv::WMMA_BM, crate::ptx_conv::WMMA_BN, 1)
}

/// [`wino_bgemm_ptx`] against an explicit shared-memory budget, returning the module **and the
/// [`SmemMode`] its launch must honour** (`Gpu::function_smem` consumes exactly this pair).
///
/// `smem_budget` is the ceiling this entry may spend — [`STATIC_SMEM_CAP`] for the historical static
/// form, or `Gpu::smem_budget()` (the probed opt-in window) from the dispatch layer. The generator
/// stays a pure text function: the budget is passed *in*, never probed here, so an A100/H100 budget is
/// checkable on a laptop with no device.
///
/// The CTA tile is a pair of compile-time constants, so [`wino_bgemm_smem_bytes`] is 20480 B for every
/// shape and the mode is always [`SmemMode::Static`]; the budget is a ceiling assert, positioned so
/// that a future widening is declined at generation (naming the family, the tile and the byte count)
/// instead of as an opaque `ptxas error: uses too much shared data` from `cuModuleLoadData`.
pub fn wino_bgemm_ptx_budget(
    c: usize,
    nt: usize,
    k: usize,
    smem_budget: usize,
) -> (String, SmemMode) {
    use crate::ptx_conv::{WMMA_BM, WMMA_BN, WMMA_THREADS, WMMA_WM, WMMA_WN};
    use std::fmt::Write as _;
    let m = k; // GEMM M
    let n = nt; // GEMM N
    let gk = c; // GEMM K (contraction)
    let kc = k * c; // U plane stride (f16 elems)
    let cnt = c * nt; // V plane stride
    let knt = k * nt; // M plane stride (f32 elems)
    let (bm, bn) = (WMMA_BM, WMMA_BN);
    let (warps_m, warps_n) = (WMMA_WM, WMMA_WN);
    let threads = WMMA_THREADS;
    let wm = bm / warps_m;
    let wn = bn / warps_n;
    let tm = wm / 16;
    let tn = wn / 16;
    let wn_shift = warps_n.trailing_zeros();
    let bn_shift = bn.trailing_zeros();
    let a_per = bm * 16 / threads;
    let b_per = 16 * bn / threads;
    let c_per = bm * bn / threads;
    let smem_a = bm * 16 * 2;
    let smem_b = 16 * bn * 2;
    let smem_c = bm * bn * 4;
    // The closed form, from the one place callers read it. `STATIC_SMEM_CAP` decides the emission FORM
    // (static `.shared` vs the `.extern` window); `smem_budget` is the ceiling — two boundaries.
    let smem_total = wino_bgemm_smem_bytes();
    debug_assert_eq!(smem_total, smem_a + smem_b + smem_c);
    assert!(
        smem_total <= smem_budget,
        "wino_bgemm: SMEM {smem_total} B ({bm}x{bn} tile) exceeds the budget {smem_budget} B"
    );
    let mode = smem_mode_for(smem_total);
    assert!(
        !mode.is_dynamic(),
        "wino_bgemm: {smem_total} B needs the dynamic window, which this generator does not emit \
         (widening the CTA tile must land with the `.extern` window AND gpu.rs's launch config)"
    );

    let veclist = |pre: &str| -> String {
        let regs: Vec<String> = (0..8).map(|i| format!("%{pre}{i}")).collect();
        format!("{{{}}}", regs.join(","))
    };

    let mut b = String::new();
    let _ = writeln!(b, "{HDR_SM80}");
    let _ = writeln!(
        b,
        "// Winograd batched NN GEMM: M[z][K{k},T{nt}] = U[z][K,C{c}] * V[z][C,T], z=gridDim.z"
    );
    let _ = writeln!(b, ".visible .entry wino_bgemm(\n    .param .u64 pV,\n    .param .u64 pU,\n    .param .u64 pM\n)\n{{");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemA[{smem_a}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemB[{smem_b}];");
    let _ = writeln!(b, "    .shared .align 16 .b8 smemC[{smem_c}];");
    let _ = writeln!(b, "    .reg .pred %p0,%pv;");
    let _ = writeln!(b, "    .reg .b16 %hv;");
    let _ = writeln!(
        b,
        "    .reg .b32 %tix,%m0,%n0,%kt,%e,%mm,%gkk,%ncol,%gkv,%nn,%xidx,%widx,%tmp,%tmp2,%saddr,%warpId,%wrb,%wcb,%zA,%zB,%zM,%z;"
    );
    let mut decl = String::new();
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                decl += &format!("%c{ti}_{tj}_{rr},");
            }
        }
    }
    for ti in 0..tm {
        for rr in 0..8 {
            decl += &format!("%a{ti}_{rr},");
        }
    }
    for tj in 0..tn {
        for rr in 0..8 {
            decl += &format!("%b{tj}_{rr},");
        }
    }
    let _ = writeln!(b, "    .reg .f32 %cf;");
    let _ = writeln!(b, "    .reg .b32 {};", decl.trim_end_matches(','));
    let _ = writeln!(b, "    .reg .b64 %V,%U,%M,%off,%gp,%ptr;");
    let _ = writeln!(
        b,
        "    ld.param.u64 %V,[pV];\n    ld.param.u64 %U,[pU];\n    ld.param.u64 %M,[pM];"
    );
    let _ = writeln!(b, "    cvta.to.global.u64 %V,%V;\n    cvta.to.global.u64 %U,%U;\n    cvta.to.global.u64 %M,%M;");
    let _ = writeln!(b, "    mov.u32 %tix,%tid.x;");
    let _ = writeln!(
        b,
        "    mov.u32 %tmp,%ctaid.y;\n    mul.lo.s32 %m0,%tmp,{bm};"
    );
    let _ = writeln!(
        b,
        "    mov.u32 %tmp,%ctaid.x;\n    mul.lo.s32 %n0,%tmp,{bn};"
    );
    // plane (z) base offsets, in ELEMENTS (folded into each index below)
    let _ = writeln!(b, "    mov.u32 %z,%ctaid.z;");
    let _ = writeln!(b, "    mul.lo.s32 %zA,%z,{kc};      // U plane = z*K*C");
    let _ = writeln!(b, "    mul.lo.s32 %zB,%z,{cnt};     // V plane = z*C*T");
    let _ = writeln!(b, "    mul.lo.s32 %zM,%z,{knt};     // M plane = z*K*T");
    let _ = writeln!(b, "    shr.u32 %warpId,%tix,5;");
    let _ = writeln!(b, "    shr.u32 %tmp,%warpId,{wn_shift};");
    let _ = writeln!(b, "    mul.lo.s32 %wrb,%tmp,{wm};");
    let _ = writeln!(b, "    and.b32 %tmp,%warpId,{};", warps_n - 1);
    let _ = writeln!(b, "    mul.lo.s32 %wcb,%tmp,{wn};");
    for ti in 0..tm {
        for tj in 0..tn {
            for rr in 0..8 {
                let _ = writeln!(b, "    mov.f32 %c{ti}_{tj}_{rr},0f00000000;");
            }
        }
    }
    let _ = writeln!(b, "    mov.u32 %kt,0;");
    let _ = writeln!(b, "KLOOP:");
    let _ = writeln!(b, "    setp.ge.u32 %p0,%kt,{gk};\n    @%p0 bra KEND;");
    // stage A (U[K,C]): A[m][gc] = U[zA + gm*C + gc]
    for li in 0..a_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(b, "    shr.u32 %mm,%e,4;\n    and.b32 %gkk,%e,15;");
        let _ = writeln!(b, "    add.u32 %tmp,%m0,%mm;\n    add.u32 %tmp2,%kt,%gkk;");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%tmp,{m};\n    setp.lt.u32 %p0,%tmp2,{gk};\n    and.pred %pv,%pv,%p0;");
        let _ = writeln!(
            b,
            "    mad.lo.s32 %widx,%tmp,{gk},%tmp2;\n    add.u32 %widx,%widx,%zA;"
        );
        let _ = writeln!(
            b,
            "    mul.wide.u32 %off,%widx,2;\n    add.s64 %ptr,%U,%off;"
        );
        let _ = writeln!(b, "    mov.u16 %hv,0;\n    @%pv ld.global.u16 %hv,[%ptr];");
        let _ = writeln!(b, "    mov.u32 %saddr,smemA;\n    shl.b32 %tmp,%e,1;\n    add.u32 %saddr,%saddr,%tmp;\n    st.shared.u16 [%saddr],%hv;");
    }
    // stage B (V[C,T]): B[gk][n] = V[zB + gkv*T + nn]  (plain row-major, no im2col)
    for li in 0..b_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(
            b,
            "    shr.u32 %gkk,%e,{bn_shift};\n    and.b32 %ncol,%e,{};",
            bn - 1
        );
        let _ = writeln!(b, "    add.u32 %gkv,%kt,%gkk;\n    add.u32 %nn,%n0,%ncol;");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%gkv,{gk};\n    setp.lt.u32 %p0,%nn,{n};\n    and.pred %pv,%pv,%p0;");
        let _ = writeln!(
            b,
            "    mad.lo.s32 %xidx,%gkv,{nt},%nn;\n    add.u32 %xidx,%xidx,%zB;"
        );
        let _ = writeln!(
            b,
            "    mul.wide.u32 %off,%xidx,2;\n    add.s64 %ptr,%V,%off;"
        );
        let _ = writeln!(b, "    mov.u16 %hv,0;\n    @%pv ld.global.u16 %hv,[%ptr];");
        let _ = writeln!(b, "    mov.u32 %saddr,smemB;\n    shl.b32 %tmp,%e,1;\n    add.u32 %saddr,%saddr,%tmp;\n    st.shared.u16 [%saddr],%hv;");
    }
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    mov.u32 %tmp,16;");
    for ti in 0..tm {
        let _ = writeln!(
            b,
            "    add.u32 %tmp2,%wrb,{};\n    mul.lo.s32 %tmp2,%tmp2,32;",
            ti * 16
        );
        let _ = writeln!(
            b,
            "    mov.u32 %saddr,smemA;\n    add.u32 %tmp2,%tmp2,%saddr;"
        );
        let _ = writeln!(
            b,
            "    cvt.u64.u32 %gp,%tmp2;\n    cvta.shared.u64 %gp,%gp;"
        );
        let ra = veclist(&format!("a{ti}_"));
        let _ = writeln!(
            b,
            "    wmma.load.a.sync.aligned.m16n16k16.row.f16 {ra}, [%gp], %tmp;"
        );
    }
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for tj in 0..tn {
        let _ = writeln!(
            b,
            "    add.u32 %tmp2,%wcb,{};\n    shl.b32 %tmp2,%tmp2,1;",
            tj * 16
        );
        let _ = writeln!(
            b,
            "    mov.u32 %saddr,smemB;\n    add.u32 %tmp2,%tmp2,%saddr;"
        );
        let _ = writeln!(
            b,
            "    cvt.u64.u32 %gp,%tmp2;\n    cvta.shared.u64 %gp,%gp;"
        );
        let rb = veclist(&format!("b{tj}_"));
        let _ = writeln!(
            b,
            "    wmma.load.b.sync.aligned.m16n16k16.row.f16 {rb}, [%gp], %tmp;"
        );
    }
    for ti in 0..tm {
        let ra = veclist(&format!("a{ti}_"));
        for tj in 0..tn {
            let rb = veclist(&format!("b{tj}_"));
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(
                b,
                "    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {cc}, {ra}, {rb}, {cc};"
            );
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    let _ = writeln!(b, "    add.u32 %kt,%kt,16;\n    bra KLOOP;");
    let _ = writeln!(b, "KEND:");
    let _ = writeln!(b, "    mov.u32 %tmp,{bn};");
    for ti in 0..tm {
        for tj in 0..tn {
            let _ = writeln!(
                b,
                "    add.u32 %tmp2,%wrb,{};\n    mul.lo.s32 %tmp2,%tmp2,{bn};",
                ti * 16
            );
            let _ = writeln!(
                b,
                "    add.u32 %tmp2,%tmp2,%wcb;\n    add.u32 %tmp2,%tmp2,{};",
                tj * 16
            );
            let _ = writeln!(b, "    shl.b32 %tmp2,%tmp2,2;\n    mov.u32 %saddr,smemC;\n    add.u32 %tmp2,%tmp2,%saddr;");
            let _ = writeln!(
                b,
                "    cvt.u64.u32 %gp,%tmp2;\n    cvta.shared.u64 %gp,%gp;"
            );
            let cc = veclist(&format!("c{ti}_{tj}_"));
            let _ = writeln!(
                b,
                "    wmma.store.d.sync.aligned.m16n16k16.row.f32 [%gp], {cc}, %tmp;"
            );
        }
    }
    let _ = writeln!(b, "    bar.sync 0;");
    for li in 0..c_per {
        let off = li * threads;
        let _ = writeln!(b, "    add.u32 %e,%tix,{off};");
        let _ = writeln!(
            b,
            "    shr.u32 %mm,%e,{bn_shift};\n    and.b32 %ncol,%e,{};",
            bn - 1
        );
        let _ = writeln!(b, "    add.u32 %tmp,%m0,%mm;\n    add.u32 %nn,%n0,%ncol;");
        let _ = writeln!(b, "    setp.lt.u32 %pv,%tmp,{m};\n    setp.lt.u32 %p0,%nn,{n};\n    and.pred %pv,%pv,%p0;");
        let _ = writeln!(b, "    mov.u32 %saddr,smemC;\n    shl.b32 %tmp2,%e,2;\n    add.u32 %saddr,%saddr,%tmp2;\n    ld.shared.f32 %cf,[%saddr];");
        let _ = writeln!(
            b,
            "    mad.lo.s32 %xidx,%tmp,{n},%nn;\n    add.u32 %xidx,%xidx,%zM;"
        );
        let _ = writeln!(
            b,
            "    mul.wide.u32 %off,%xidx,4;\n    add.s64 %ptr,%M,%off;"
        );
        let _ = writeln!(b, "    @%pv st.global.f32 [%ptr],%cf;");
    }
    let _ = writeln!(b, "    ret;\n}}");
    (b, mode)
}

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

    /// **§3A P1 gate, device-free.** The four Winograd PTX generators bake f32 hex constants and
    /// `//` header comments into their modules, and the Rust doc comments right beside those
    /// `writeln!`s are full of non-ASCII (`α`, `ξν`, `Bᵀ`, `→`) — one copy-paste away from a `ptxas
    /// fatal` that surfaces on the device only as an opaque `cuModuleLoadData` `DriverError`. Both
    /// output-tile sizes (`m = 2` and `m = 4`, i.e. `α = 4` and `6`) are swept for every shape.
    #[test]
    fn every_winograd_generator_emits_ascii_ptx() {
        for m in [2usize, 4] {
            for (c, h, w, k) in [
                (3usize, 32usize, 32usize, 16usize),
                (64, 14, 14, 64),
                (8, 9, 9, 32),
            ] {
                let (_, _, nt) = wino_ntiles(h, w, m);
                for (what, ptx) in [
                    ("wino_filter_xform_ptx", wino_filter_xform_ptx(c, k, m)),
                    ("wino_input_xform_ptx", wino_input_xform_ptx(c, h, w, m)),
                    ("wino_output_xform_ptx", wino_output_xform_ptx(k, h, w, m)),
                    ("wino_bgemm_ptx", wino_bgemm_ptx(c, nt, k)),
                    // The budget-carrying entry point is its own text path (it decides the emission
                    // form), so it gets its own pass — no variant goes unchecked.
                    (
                        "wino_bgemm_ptx_budget",
                        wino_bgemm_ptx_budget(c, nt, k, STATIC_SMEM_CAP).0,
                    ),
                ] {
                    if let Some((i, line)) = ptx.lines().enumerate().find(|(_, l)| !l.is_ascii()) {
                        panic!(
                            "{what} (C{c} {h}x{w} K{k} m{m}): PTX line {} is not ASCII (ptxas \
                             fatal): {line:?}",
                            i + 1
                        );
                    }
                    assert!(!ptx.is_empty(), "{what}: generated an empty module");
                    // Header floor: every Winograd module opens with the shared `sm_80` header, never
                    // the device's arch — PTX is forward-compatible only, so an `sm_89` tag would be
                    // unloadable on every A100 while buying nothing on Ada.
                    assert!(
                        ptx.starts_with(crate::ptx_target::HDR_SM80),
                        "{what}: must open with ptx_target::HDR_SM80, got {:?}",
                        &ptx[..ptx.len().min(64)]
                    );
                    assert!(
                        !ptx.contains(crate::ptx_target::TARGET_SM89),
                        "{what}: an Ampere-legal module must not claim the Ada floor"
                    );
                }
            }
        }
    }

    /// FNV-1a 64 over the raw bytes — a dependency-free, deterministic digest of a PTX module.
    pub(super) fn ptx_digest(s: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Every Winograd module this crate can dispatch, at the shapes the dispatch path uses.
    pub(super) fn shipped_wino_modules() -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for m in [2usize, 4] {
            for (c, h, w, k) in [
                (3usize, 32usize, 32usize, 16usize),
                (64, 14, 14, 64),
                (128, 28, 28, 128),
                (8, 9, 9, 32),
            ] {
                let (_, _, nt) = wino_ntiles(h, w, m);
                let tag = format!("C{c}_{h}x{w}_K{k}_m{m}");
                out.push((
                    format!("wino_filter_xform_ptx/{tag}"),
                    wino_filter_xform_ptx(c, k, m),
                ));
                out.push((
                    format!("wino_input_xform_ptx/{tag}"),
                    wino_input_xform_ptx(c, h, w, m),
                ));
                out.push((
                    format!("wino_output_xform_ptx/{tag}"),
                    wino_output_xform_ptx(k, h, w, m),
                ));
                out.push((format!("wino_bgemm_ptx/{tag}"), wino_bgemm_ptx(c, nt, k)));
            }
        }
        out
    }

    /// `(label, byte length, FNV-1a 64)` of every module in [`shipped_wino_modules`], **recorded from
    /// the binary built at this commit's parent** — the "before" half of
    /// [`shipped_wino_ptx_is_byte_identical`].
    #[rustfmt::skip]
    const SHIPPED_WINO_PTX: &[(&str, usize, u64)] = &[
        ("wino_filter_xform_ptx/C3_32x32_K16_m2", 4842, 0xba73b657911a2f30),
        ("wino_input_xform_ptx/C3_32x32_K16_m2", 9094, 0x6ddc4a45d488fb05),
        ("wino_output_xform_ptx/C3_32x32_K16_m2", 4183, 0x74ebabe8289621e6),
        ("wino_bgemm_ptx/C3_32x32_K16_m2", 30053, 0xda28caf1cf2dfb89),
        ("wino_filter_xform_ptx/C64_14x14_K64_m2", 4873, 0xc95ed6fb178727c4),
        ("wino_input_xform_ptx/C64_14x14_K64_m2", 9095, 0x0700054488208d6e),
        ("wino_output_xform_ptx/C64_14x14_K64_m2", 4173, 0xf874de507b393bf9),
        ("wino_bgemm_ptx/C64_14x14_K64_m2", 30001, 0xafd21cf8d89eba66),
        ("wino_filter_xform_ptx/C128_28x28_K128_m2", 4886, 0x3d833fe0d99b7965),
        ("wino_input_xform_ptx/C128_28x28_K128_m2", 9117, 0x64a7648f875db826),
        ("wino_output_xform_ptx/C128_28x28_K128_m2", 4194, 0xc3a25506cc203dc6),
        ("wino_bgemm_ptx/C128_28x28_K128_m2", 30152, 0x0abdda02f5c18117),
        ("wino_filter_xform_ptx/C8_9x9_K32_m2", 4853, 0x02771c2b49aa5ad9),
        ("wino_input_xform_ptx/C8_9x9_K32_m2", 9038, 0x18a354f05ec9b52f),
        ("wino_output_xform_ptx/C8_9x9_K32_m2", 4148, 0x76ee5ed265096f48),
        ("wino_bgemm_ptx/C8_9x9_K32_m2", 29972, 0x008ad0b39c8cec56),
        ("wino_filter_xform_ptx/C3_32x32_K16_m4", 11486, 0x61b13dd6c63f5a87),
        ("wino_input_xform_ptx/C3_32x32_K16_m4", 33812, 0xf5a80b1c29bb3b60),
        ("wino_output_xform_ptx/C3_32x32_K16_m4", 20506, 0x63d5f5e471f06a2b),
        ("wino_bgemm_ptx/C3_32x32_K16_m4", 29972, 0xb89728874dd18bc8),
        ("wino_filter_xform_ptx/C64_14x14_K64_m4", 11557, 0x17e3246ea3bde475),
        ("wino_input_xform_ptx/C64_14x14_K64_m4", 33829, 0x4cb37deacfbf0f6c),
        ("wino_output_xform_ptx/C64_14x14_K64_m4", 20489, 0x311ef83afff5182d),
        ("wino_bgemm_ptx/C64_14x14_K64_m4", 29918, 0xf4811ce80bbf004e),
        ("wino_filter_xform_ptx/C128_28x28_K128_m4", 11575, 0x8a9bd8819be5079f),
        ("wino_input_xform_ptx/C128_28x28_K128_m4", 33870, 0xd67a7e5a4b16ef52),
        ("wino_output_xform_ptx/C128_28x28_K128_m4", 20530, 0xd5f9bc5dccee45b0),
        ("wino_bgemm_ptx/C128_28x28_K128_m4", 30069, 0x21a607a0f4b6a0aa),
        ("wino_filter_xform_ptx/C8_9x9_K32_m4", 11513, 0x5b8545bbcba1fde7),
        ("wino_input_xform_ptx/C8_9x9_K32_m4", 33703, 0x2c256f2f6730c87f),
        ("wino_output_xform_ptx/C8_9x9_K32_m4", 20419, 0xe4fd35045c8f6889),
        ("wino_bgemm_ptx/C8_9x9_K32_m4", 29890, 0x700d1335583fdbe2),
    ];

    /// **The before/after gate for the budget seam, device-free.** Threading `smem_budget` through
    /// [`wino_bgemm_ptx`] must leave every shipped Winograd module byte-identical: at or below
    /// [`STATIC_SMEM_CAP`] the historical static `.shared` spelling is emitted verbatim, and every
    /// caller passes exactly that. The pins were recorded from the binary at this commit's **parent**,
    /// so this is a real across-the-change comparison, not a self-consistency tautology.
    #[test]
    fn shipped_wino_ptx_is_byte_identical() {
        let got = shipped_wino_modules();
        assert_eq!(got.len(), SHIPPED_WINO_PTX.len());
        let mut bad = Vec::new();
        for ((label, ptx), (plabel, plen, pdig)) in got.iter().zip(SHIPPED_WINO_PTX) {
            assert_eq!(label, plabel, "pin order must match the generated order");
            let (len, dig) = (ptx.len(), ptx_digest(ptx));
            if len != *plen || dig != *pdig {
                bad.push(format!(
                    "  {label}: pinned ({plen} B, 0x{pdig:016x}) != got ({len} B, 0x{dig:016x})"
                ));
            }
        }
        if !bad.is_empty() {
            for (label, ptx) in &got {
                println!("(\"{label}\", {}, 0x{:016x}),", ptx.len(), ptx_digest(ptx));
            }
            panic!(
                "{} shipped Winograd module(s) changed byte-for-byte:\n{}\nIf intended, replace \
                 SHIPPED_WINO_PTX with the table printed above.",
                bad.len(),
                bad.join("\n")
            );
        }
    }

    /// **The Winograd half of the shared-memory census, device-free.**
    ///
    /// Three of the four kernels use no shared memory at all; the whole family's footprint is the
    /// bgemm's 20480 B, constant for every shape and 42 % of the PTX ISA's static cap. So a budget lift
    /// frees exactly nothing here either — and the arithmetic says why the family's real weakness is
    /// elsewhere: `GK = C`, so a `C=3` first layer feeds a 16-wide WMMA K-tile 3 real channels and 13
    /// of padding. That is the mechanism behind "F(4×4,3×3) loses at low channel count", and it is a
    /// K-tile problem, not an SMEM one.
    #[test]
    fn wino_smem_census_and_the_low_channel_k_tile() {
        assert_eq!(wino_bgemm_smem_bytes(), 20480);
        assert_eq!(
            wino_bgemm_smem_bytes(),
            crate::ptx_conv::conv_wmma_smem_bytes(
                crate::ptx_conv::WMMA_BM,
                crate::ptx_conv::WMMA_BN,
                1
            ),
            "the bgemm IS the implicit-GEMM tensor-core core with a plain load"
        );
        // Constant across every shape the dispatcher can hand it, and always the static form.
        for (c, nt, k) in [(3usize, 64usize, 16usize), (128, 49, 128), (512, 9, 512)] {
            let (_, mode) = wino_bgemm_ptx_budget(c, nt, k, STATIC_SMEM_CAP);
            assert_eq!(mode, SmemMode::Static);
            assert_eq!(mode.launch_bytes(), 0, "a static tile launches with 0");
        }
        // The transforms declare no shared memory whatsoever.
        for ptx in [
            wino_filter_xform_ptx(64, 64, 4),
            wino_input_xform_ptx(64, 14, 14, 4),
            wino_output_xform_ptx(64, 14, 14, 4),
        ] {
            assert!(!ptx.contains(".shared"), "the transforms are register-only");
        }
        // The low-channel cliff, as arithmetic: fraction of each staged 16-deep K-tile that is real.
        for (c, live_frac_pct) in [(3usize, 18usize), (16, 100), (64, 100)] {
            let padded = c.div_ceil(16) * 16;
            assert_eq!(c * 100 / padded, live_frac_pct, "C={c} K-tile utilisation");
        }
    }

    /// Over-budget generation must fail loudly at generation, naming the family, the tile and both byte
    /// counts — not as an opaque `ptxas error: uses too much shared data` out of `cuModuleLoadData`.
    #[test]
    #[should_panic(expected = "exceeds the budget")]
    fn wino_bgemm_over_budget_panics_at_generation() {
        let _ = wino_bgemm_ptx_budget(64, 49, 64, 4096);
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
