//! Norm-backward **pass decomposition + before/after A/B** probe (dev instrument, not a shipped
//! bench). SOTA campaign 2026-09, workstream W3.
//!
//! The 2026-09 CPU evaluation put `layernorm_bwd` at 0.780x of `C(fast)` at 1024x1024 and 0.914x at
//! 4096x512, and `softmax_bwd` at 0.914x at 1024x1024. This probe answers two questions in ONE
//! process, so no cross-run drift, power state or thermal history can enter the comparison:
//!
//!   1. **Where does the time go?** Each row pass of the *old* LayerNorm-backward kernel is timed on
//!      its own over the real shapes (mean, variance, s1+s2, apply), against a one-pass streaming
//!      floor over the same three buffers. A pass far above the floor is latency-bound, not
//!      bandwidth-bound; a pass at the floor has nothing left to give.
//!   2. **What did the change buy?** [`old_layernorm_bwd`] is a verbatim copy of the kernel as it
//!      stood before the multi-accumulator + pass-fusion rewrite; it alternates ABBA with the shipped
//!      [`wukong_layernorm_bwd_f32`], best-of-N, and their outputs are diffed afterwards.
//!
//! Every timed arm writes to the SAME output buffer, exactly as `wukong_xbench` does: private 4-8 MB
//! destinations per arm would push the working set past this host's 24 MB L3 and turn every arm
//! DRAM-bound, compressing the very difference being measured. (That mistake was made once while
//! building this probe and it halved the apparent effect.)
//!
//! Run: `cargo run -p wukong_runtime --release --example normbwd_var [-- <rounds>]`
//! AGENT-LOCAL, DIRECTIONAL ONLY: same-process ratios only; never quote the absolute ns.

use std::time::Instant;
use wukong_runtime::{wukong_layernorm_bwd_f32, wukong_softmax_bwd_f32};

#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
}

// ---------------------------------------------------------------------------------------------
// The kernel as it stood BEFORE this workstream: four passes over the row, each reduction on one
// `__m256` accumulator. Kept verbatim so the A/B below stays meaningful as the shipped side moves.
// ---------------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn old_mean_pass(x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        sv = _mm256_add_ps(sv, _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    let mut s = [0.0f32; 8];
    _mm256_storeu_ps(s.as_mut_ptr(), sv);
    for (j, sj) in s.iter_mut().enumerate().take(n - i) {
        *sj += *x.add(i + j);
    }
    hsum8(s)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn old_var_pass(x: *const f32, n: usize, mean: f32) -> f32 {
    use std::arch::x86_64::*;
    let mb = _mm256_set1_ps(mean);
    let mut vv = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb);
        vv = _mm256_fmadd_ps(d, d, vv);
        i += 8;
    }
    let mut v = [0.0f32; 8];
    _mm256_storeu_ps(v.as_mut_ptr(), vv);
    for (j, vj) in v.iter_mut().enumerate().take(n - i) {
        let d = *x.add(i + j) - mean;
        *vj = d.mul_add(d, *vj);
    }
    hsum8(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn old_s1s2_pass(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    n: usize,
    mean: f32,
    rstd: f32,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mb = _mm256_set1_ps(mean);
    let rb = _mm256_set1_ps(rstd);
    let mut s1v = _mm256_setzero_ps();
    let mut s2v = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let g = _mm256_mul_ps(_mm256_loadu_ps(dy.add(i)), _mm256_loadu_ps(gamma.add(i)));
        let xh = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb), rb);
        s1v = _mm256_add_ps(s1v, g);
        s2v = _mm256_fmadd_ps(g, xh, s2v);
        i += 8;
    }
    let (mut s1, mut s2) = ([0.0f32; 8], [0.0f32; 8]);
    _mm256_storeu_ps(s1.as_mut_ptr(), s1v);
    _mm256_storeu_ps(s2.as_mut_ptr(), s2v);
    for j in 0..(n - i) {
        let g = *dy.add(i + j) * *gamma.add(i + j);
        let xh = (*x.add(i + j) - mean) * rstd;
        s1[j] += g;
        s2[j] = g.mul_add(xh, s2[j]);
    }
    (hsum8(s1), hsum8(s2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn old_apply_pass(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    n: usize,
    mean: f32,
    rstd: f32,
    s1c: f32,
    s2c: f32,
) {
    use std::arch::x86_64::*;
    let mb = _mm256_set1_ps(mean);
    let rb = _mm256_set1_ps(rstd);
    let s1b = _mm256_set1_ps(s1c);
    let s2b = _mm256_set1_ps(s2c);
    let mut i = 0;
    while i + 8 <= n {
        let gv = _mm256_mul_ps(_mm256_loadu_ps(dy.add(i)), _mm256_loadu_ps(gamma.add(i)));
        let xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x.add(i)), mb), rb);
        let inner = _mm256_sub_ps(_mm256_sub_ps(gv, s1b), _mm256_mul_ps(xhat, s2b));
        _mm256_storeu_ps(dx.add(i), _mm256_mul_ps(rb, inner));
        i += 8;
    }
    while i < n {
        let g = *dy.add(i) * *gamma.add(i);
        let xhat = (*x.add(i) - mean) * rstd;
        *dx.add(i) = rstd * (g - s1c - xhat * s2c);
        i += 1;
    }
}

/// The pre-change kernel: mean pass, variance pass, s1+s2 pass, apply pass.
///
/// # Safety
/// `x`, `dy`, `dx` valid for `rows*cols` f32; `gamma` valid for `cols` f32; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
unsafe fn old_layernorm_bwd(
    x: *const f32,
    dy: *const f32,
    gamma: *const f32,
    dx: *mut f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let invn = 1.0 / (cols as f32);
    for r in 0..rows {
        let o = r * cols;
        let (xp, dyp, dxp) = (x.add(o), dy.add(o), dx.add(o));
        let mean = old_mean_pass(xp, cols) * invn;
        let rstd = 1.0 / (old_var_pass(xp, cols, mean) * invn + eps).sqrt();
        let (s1, s2) = old_s1s2_pass(xp, dyp, gamma, cols, mean, rstd);
        old_apply_pass(xp, dyp, gamma, dxp, cols, mean, rstd, s1 * invn, s2 * invn);
    }
}

/// One-pass streaming floor over the same three buffers: read x, read dy, write dx. Nothing that
/// touches all three can beat this, so it bounds what any restructuring of the kernel can reach.
///
/// # Safety
/// `x`, `dy`, `dx` valid for `n` f32; AVX2 available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn stream_floor(x: *const f32, dy: *const f32, dx: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(
            dx.add(i),
            _mm256_add_ps(_mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(dy.add(i))),
        );
        i += 8;
    }
}

/// `dx = y*(dy - SUM y*dy)` per row with an INLINED four-chain dot. The shipped kernel delegates the
/// dot to the out-of-line `wukong_sreduce_f32`, whose AVX2 body runs one accumulator; this arm
/// measures what both of those cost together. It is a probe, not a proposal: it changes the
/// reduction order that `softmax_bwd`'s module contract pins to the shared `sreduce`.
///
/// # Safety
/// `y`, `dy`, `dx` valid for `rows*cols` f32; AVX2+FMA available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sm_bwd_inlined_dot(
    y: *const f32,
    dy: *const f32,
    dx: *mut f32,
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    for r in 0..rows {
        let o0 = r * cols;
        let (yp, dyp, dxp) = (y.add(o0), dy.add(o0), dx.add(o0));
        let mut a = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= cols {
            for (t, at) in a.iter_mut().enumerate() {
                let o = i + 8 * t;
                *at = _mm256_fmadd_ps(_mm256_loadu_ps(yp.add(o)), _mm256_loadu_ps(dyp.add(o)), *at);
            }
            i += 32;
        }
        while i + 8 <= cols {
            a[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(yp.add(i)),
                _mm256_loadu_ps(dyp.add(i)),
                a[0],
            );
            i += 8;
        }
        a[0] = _mm256_add_ps(a[0], a[2]);
        a[1] = _mm256_add_ps(a[1], a[3]);
        a[0] = _mm256_add_ps(a[0], a[1]);
        let mut s = [0.0f32; 8];
        _mm256_storeu_ps(s.as_mut_ptr(), a[0]);
        for (j, sj) in s.iter_mut().enumerate().take(cols - i) {
            *sj = (*yp.add(i + j)).mul_add(*dyp.add(i + j), *sj);
        }
        let sc = hsum8(s);
        let sv = _mm256_set1_ps(sc);
        let mut k = 0;
        while k + 8 <= cols {
            _mm256_storeu_ps(
                dxp.add(k),
                _mm256_mul_ps(
                    _mm256_loadu_ps(yp.add(k)),
                    _mm256_sub_ps(_mm256_loadu_ps(dyp.add(k)), sv),
                ),
            );
            k += 8;
        }
        while k < cols {
            *dxp.add(k) = *yp.add(k) * (*dyp.add(k) - sc);
            k += 1;
        }
    }
}

fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for r in 0..=reps {
        let t = Instant::now();
        f();
        let ns = t.elapsed().as_nanos() as f64;
        if r > 0 && ns < best {
            best = ns;
        }
    }
    best
}

fn maxdev(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut m = 0.0f64;
    let mut d = 0.0f64;
    for (av, bv) in a.iter().zip(b.iter()) {
        d = d.max(av.abs() as f64);
        m = m.max((*av as f64 - *bv as f64).abs());
    }
    (m, m / d.max(1e-30))
}

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("normbwd_var: x86_64 only");
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("normbwd_var: needs AVX2+FMA");
            return;
        }
        let reps: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);
        println!("normbwd_var: AGENT-LOCAL, DIRECTIONAL ONLY (same-process ABBA, best-of-{reps})");

        for &(rows, cols) in &[(1024usize, 1024usize), (4096usize, 512usize)] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i % 19) as f32 - 9.0) * 0.1 + 0.3)
                .collect();
            let dy: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
            let gamma: Vec<f32> = (0..cols).map(|i| (i % 11) as f32 * 0.05 + 0.7).collect();
            let mut out = vec![0.0f32; n];
            let op = out.as_mut_ptr();
            let (xp, dyp, gp) = (x.as_ptr(), dy.as_ptr(), gamma.as_ptr());
            let eps = 1e-5f32;
            let ebits = eps.to_bits() as i64;
            println!(
                "\n=== layernorm_bwd {rows}x{cols} ({:.1} MB/buffer) ===",
                (n * 4) as f64 / 1048576.0
            );

            let mut sink = 0.0f32;
            let t_mean = best_of(reps, || {
                let mut s = 0.0f32;
                for r in 0..rows {
                    s += unsafe { old_mean_pass(xp.add(r * cols), cols) };
                }
                sink += s;
            });
            let t_var = best_of(reps, || {
                let mut s = 0.0f32;
                for r in 0..rows {
                    s += unsafe { old_var_pass(xp.add(r * cols), cols, 0.3) };
                }
                sink += s;
            });
            let t_s1s2 = best_of(reps, || {
                let mut s = 0.0f32;
                for r in 0..rows {
                    let (a, b) = unsafe {
                        old_s1s2_pass(xp.add(r * cols), dyp.add(r * cols), gp, cols, 0.3, 1.7)
                    };
                    s += a + b;
                }
                sink += s;
            });
            let t_apply = best_of(reps, || {
                for r in 0..rows {
                    unsafe {
                        old_apply_pass(
                            xp.add(r * cols),
                            dyp.add(r * cols),
                            gp,
                            op.add(r * cols),
                            cols,
                            0.3,
                            1.7,
                            0.1,
                            0.2,
                        )
                    };
                }
            });
            let t_floor = best_of(reps, || unsafe { stream_floor(xp, dyp, op, n) });
            println!(
                "  old passes (ns): mean {t_mean:.0} | var {t_var:.0} | s1s2 {t_s1s2:.0} | \
                 apply {t_apply:.0} || 1-pass stream floor {t_floor:.0}"
            );

            let mut b_old = f64::INFINITY;
            let mut b_new = f64::INFINITY;
            for round in 0..reps {
                let run_old = || {
                    let t = Instant::now();
                    unsafe { old_layernorm_bwd(xp, dyp, gp, op, rows, cols, eps) };
                    t.elapsed().as_nanos() as f64
                };
                let run_new = || {
                    let t = Instant::now();
                    unsafe {
                        wukong_layernorm_bwd_f32(xp, dyp, gp, op, rows as i64, cols as i64, ebits)
                    };
                    t.elapsed().as_nanos() as f64
                };
                let (o, w) = if round % 2 == 0 {
                    let o = run_old();
                    let w = run_new();
                    (o, w)
                } else {
                    let w = run_new();
                    let o = run_old();
                    (o, w)
                };
                if round > 0 {
                    b_old = b_old.min(o);
                    b_new = b_new.min(w);
                }
            }
            // Deviation check, UNTIMED and after every timed arm, so these buffers never enlarge the
            // timed working set.
            let mut c_old = vec![0.0f32; n];
            let mut c_new = vec![0.0f32; n];
            unsafe {
                old_layernorm_bwd(xp, dyp, gp, c_old.as_mut_ptr(), rows, cols, eps);
                wukong_layernorm_bwd_f32(
                    xp,
                    dyp,
                    gp,
                    c_new.as_mut_ptr(),
                    rows as i64,
                    cols as i64,
                    ebits,
                );
            }
            let (m, rl) = maxdev(&c_old, &c_new);
            println!(
                "  old {b_old:.0} ns | shipped {b_new:.0} ns | speedup {:.3}x | \
                 max|delta| {m:.2e} rel {rl:.2e} | sink {sink:.3e}",
                b_old / b_new
            );
        }

        for &(rows, cols) in &[(1024usize, 1024usize)] {
            let n = rows * cols;
            let y: Vec<f32> = (0..n).map(|i| ((i % 29) as f32 + 1.0) / 435.0).collect();
            let dy: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
            let mut out = vec![0.0f32; n];
            let op = out.as_mut_ptr();
            let (yp, dyp) = (y.as_ptr(), dy.as_ptr());
            println!(
                "\n=== softmax_bwd {rows}x{cols} ({:.1} MB/buffer) ===",
                (n * 4) as f64 / 1048576.0
            );
            let t_floor = best_of(reps, || unsafe { stream_floor(yp, dyp, op, n) });
            let mut b0 = f64::INFINITY;
            let mut b1 = f64::INFINITY;
            for round in 0..reps {
                let r0 = || {
                    let t = Instant::now();
                    unsafe { wukong_softmax_bwd_f32(yp, dyp, op, rows as i64, cols as i64) };
                    t.elapsed().as_nanos() as f64
                };
                let r1 = || {
                    let t = Instant::now();
                    unsafe { sm_bwd_inlined_dot(yp, dyp, op, rows, cols) };
                    t.elapsed().as_nanos() as f64
                };
                let (v0, v1) = if round % 2 == 0 {
                    let a = r0();
                    let b = r1();
                    (a, b)
                } else {
                    let b = r1();
                    let a = r0();
                    (a, b)
                };
                if round > 0 {
                    b0 = b0.min(v0);
                    b1 = b1.min(v1);
                }
            }
            let mut c0 = vec![0.0f32; n];
            let mut c1 = vec![0.0f32; n];
            unsafe {
                wukong_softmax_bwd_f32(yp, dyp, c0.as_mut_ptr(), rows as i64, cols as i64);
                sm_bwd_inlined_dot(yp, dyp, c1.as_mut_ptr(), rows, cols);
            }
            let (m, rl) = maxdev(&c0, &c1);
            println!(
                "  shipped {b0:.0} ns | inlined 4-chain dot {b1:.0} ns | speedup {:.3}x | \
                 max|delta| {m:.2e} rel {rl:.2e} || 1-pass stream floor {t_floor:.0}",
                b0 / b1
            );
        }
    }
}
