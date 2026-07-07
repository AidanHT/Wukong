//! Fast, clean serial-vs-parallel GEMM scaling probe (dev instrument, not a shipped bench).
//!
//! Times `mercury_sgemm_nt` (serial) vs `mercury_sgemm_nt_parallel` back-to-back, best-of-many, at
//! the transformer's real GEMM shapes (M = sequence length, the small-M regime that scales worst) and
//! a few squares for reference. Prints GFLOP/s and the parallel speedup so I can iterate on the
//! multicore kernel in seconds instead of a 40 s xbench run. Same-run adjacent A/B is the only
//! reliable instrument on this throttling hybrid laptop, so serial and parallel are timed adjacently.
//!
//! Run: `cargo run -p mercury_runtime --release --example gemm_scaling`

use std::time::Instant;

fn time_best<F: FnMut()>(mut f: F, iters: u32, reps: u32) -> f64 {
    // Warm up.
    for _ in 0..3 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t.elapsed().as_nanos() as f64 / reps as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

fn main() {
    println!("threads: rayon={} physical={}", rayon::current_num_threads(), num_cpus::get_physical());
    println!(
        "{:<22} {:>10} {:>10} {:>8}",
        "shape (MxKxN)", "ser GF/s", "par GF/s", "scaling"
    );
    // (M, K, N): the transformer GEMMs at S=128 and S=512, plus reference squares.
    let shapes: &[(usize, usize, usize)] = &[
        (128, 768, 768),   // S=128 qkv / attn-out proj
        (128, 768, 3072),  // S=128 ffn up
        (128, 3072, 768),  // S=128 ffn down
        (512, 768, 768),   // S=512 qkv / attn-out proj
        (512, 768, 3072),  // S=512 ffn up
        (512, 3072, 768),  // S=512 ffn down
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
        let ser = time_best(
            || unsafe {
                mercury_runtime::mercury_sgemm_nt(
                    a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i64, k as i64, n as i64, 0,
                )
            },
            12,
            reps,
        );
        let par = time_best(
            || unsafe {
                mercury_runtime::mercury_sgemm_nt_parallel(
                    a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i64, k as i64, n as i64, 0,
                )
            },
            12,
            reps,
        );
        println!(
            "{:<22} {:>10.1} {:>10.1} {:>7.2}x",
            format!("{}x{}x{}", m, k, n),
            flop / ser,
            flop / par,
            ser / par,
        );
    }
}
