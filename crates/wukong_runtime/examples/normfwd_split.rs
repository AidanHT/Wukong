//! Batched-norm forward **cost split** probe (dev instrument, not a shipped bench).
//! SOTA campaign 2026-09, workstream W3 — hand-over evidence, not a proposed change.
//!
//! `wukong_xbench`'s "batched norms" section benchmarks a two-statement program:
//!
//! ```text
//! for c in 0..N { out[c] = x[c]; }            ->  wukong_velem_f32(x, x, out, N, 1, 0, 0, COPY)
//! for r in 0..R { <norm over out[r*C + i]> }  ->  wukong_norm_f32(out, out, R, C, eps, op)
//! ```
//!
//! (verified with `--emit=mir -O2` on the same spelling). The C and Rust peers copy too, so the copy
//! is not an unfair extra — but it *is* work that no change to the norm kernel can touch, and the
//! 2026-09 evaluation's two losing rows (`layernorm` 512x768 at 0.718x of C(fast), 4096x4096 at
//! 0.842x) are for the whole two-call program. This probe measures, in one process, what fraction of
//! that program each call is, so the reachable ceiling for a norm-kernel change is a number rather
//! than a guess.
//!
//! Run: `cargo run -p wukong_runtime --release --example normfwd_split [-- <rounds>]`
//! AGENT-LOCAL, DIRECTIONAL ONLY: shares and same-process ratios only, never the absolute ns.

use std::time::Instant;
use wukong_runtime::{wukong_norm_f32, wukong_velem_f32};

/// `wukong_velem_f32` op `VE_ID` with `a = 1, b = 0, c = 0` and a null-unused `y` is exactly the
/// copy the recognizer emits for `out[c] = x[c]`; `NORM_LAYERNORM` / `NORM_RMSNORM` are `crate::norm`.
const VE_ID: i64 = 0;
const NORM_LAYERNORM: i64 = 1;
const NORM_RMSNORM: i64 = 2;

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

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    println!("normfwd_split: AGENT-LOCAL, DIRECTIONAL ONLY (same-process, best-of-{reps})");
    let eps = 1e-5f32.to_bits() as i64;

    for &(rows, cols) in &[(512usize, 768usize), (4096usize, 4096usize)] {
        let n = rows * cols;
        let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.5 + 1.0).collect();
        let mut out = vec![0.0f32; n];
        let xp = x.as_ptr();
        let op = out.as_mut_ptr();
        println!(
            "\n=== batched norms {rows}x{cols} ({:.1} MB/buffer) ===",
            (n * 4) as f64 / 1048576.0
        );
        for (name, code) in [("layernorm", NORM_LAYERNORM), ("rmsnorm", NORM_RMSNORM)] {
            // The copy alone.
            let t_copy = best_of(reps, || unsafe {
                wukong_velem_f32(xp, xp, op, n as i64, 1.0, 0.0, 0.0, VE_ID);
            });
            // The norm alone, in place on an already-copied buffer (what the second call does).
            let t_norm = best_of(reps, || unsafe {
                wukong_norm_f32(op, op, rows as i64, cols as i64, eps, code);
            });
            // Both, back to back — the whole benchmarked program.
            let t_both = best_of(reps, || unsafe {
                wukong_velem_f32(xp, xp, op, n as i64, 1.0, 0.0, 0.0, VE_ID);
                wukong_norm_f32(op, op, rows as i64, cols as i64, eps, code);
            });
            println!(
                "  {name:<10} copy {t_copy:.0} ns ({:.0}% of both) | norm {t_norm:.0} ns \
                 ({:.0}%) | both {t_both:.0} ns",
                100.0 * t_copy / t_both,
                100.0 * t_norm / t_both
            );
        }
    }
}
