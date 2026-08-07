//! Fast, clean serial-vs-parallel GEMM scaling probe (dev instrument, not a shipped bench).
//!
//! Times `wukong_sgemm_nt` (serial) vs `wukong_sgemm_nt_parallel` back-to-back, best-of-many, at
//! the transformer's real GEMM shapes (M = sequence length, the small-M regime that scales worst) and
//! a few squares for reference. Prints GFLOP/s and the parallel speedup so I can iterate on the
//! multicore kernel in seconds instead of a 40 s xbench run. Same-run adjacent A/B is the only
//! reliable instrument on this throttling hybrid laptop, so serial and parallel are timed adjacently.
//!
//! Run: `cargo run -p wukong_runtime --release --example gemm_scaling`

use std::time::Instant;

fn one<F: FnMut()>(f: &mut F, reps: u32) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_nanos() as f64 / reps as f64
}

/// Measure serial and parallel **adjacently** each round and return (best_ser_ns, best_par_ns,
/// median_scaling). Interleaving cancels the laptop's slow thermal drift: each round's ser and par
/// see nearly the same clock, so the per-round ratio is stable even as the absolute clock swings.
fn time_pair<S: FnMut(), P: FnMut()>(
    mut ser: S,
    mut par: P,
    rounds: u32,
    reps: u32,
) -> (f64, f64, f64) {
    for _ in 0..3 {
        ser();
        par();
    }
    let mut best_s = f64::INFINITY;
    let mut best_p = f64::INFINITY;
    let mut ratios = Vec::new();
    for _ in 0..rounds {
        let s = one(&mut ser, reps);
        let p = one(&mut par, reps);
        best_s = best_s.min(s);
        best_p = best_p.min(p);
        ratios.push(s / p);
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (best_s, best_p, ratios[ratios.len() / 2])
}

fn main() {
    println!(
        "threads: rayon={} physical={}",
        rayon::current_num_threads(),
        num_cpus::get_physical()
    );
    println!(
        "{:<22} {:>10} {:>10} {:>8}",
        "shape (MxKxN)", "ser GF/s", "par GF/s", "scaling"
    );
    // (M, K, N): the transformer GEMMs at S=128 and S=512, plus reference squares.
    let shapes: &[(usize, usize, usize)] = &[
        (128, 768, 768),  // S=128 qkv / attn-out proj
        (128, 768, 3072), // S=128 ffn up
        (128, 3072, 768), // S=128 ffn down
        (512, 768, 768),  // S=512 qkv / attn-out proj
        (512, 768, 3072), // S=512 ffn up
        (512, 3072, 768), // S=512 ffn down
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];
    for &(m, k, n) in shapes {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; n * k]; // B is [n, k] for the C = A·Bᵀ (nn.Linear) form.
        let mut c = vec![0.0f32; m * n];
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        // Bigger GEMMs get fewer reps so each measurement is ~a few ms.
        let reps = (200_000_000 / (m * k * n).max(1)).clamp(2, 200) as u32;
        // Capture as usize so the serial and parallel closures can coexist (both touch `c`).
        let (ap, bp, cp) = (
            a.as_ptr() as usize,
            b.as_ptr() as usize,
            c.as_mut_ptr() as usize,
        );
        let (mi, ki, ni) = (m as i64, k as i64, n as i64);
        let (ser, par, scaling) = time_pair(
            || unsafe {
                wukong_runtime::wukong_sgemm_nt(
                    ap as *const f32,
                    bp as *const f32,
                    cp as *mut f32,
                    mi,
                    ki,
                    ni,
                    0,
                )
            },
            || unsafe {
                wukong_runtime::wukong_sgemm_nt_parallel(
                    ap as *const f32,
                    bp as *const f32,
                    cp as *mut f32,
                    mi,
                    ki,
                    ni,
                    0,
                )
            },
            25,
            reps,
        );
        println!(
            "{:<22} {:>10.1} {:>10.1} {:>7.2}x",
            format!("{}x{}x{}", m, k, n),
            flop / ser,
            flop / par,
            scaling,
        );
    }
}
