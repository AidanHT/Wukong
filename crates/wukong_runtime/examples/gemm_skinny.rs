//! Skinny-GEMM (small-M) diagnostic probe — dev instrument, not a shipped bench.
//!
//! The 2026-09 CPU evaluation put single-core `wukong_sgemm_nt` at 0.716x-0.739x of
//! oneMKL(1 thread) on `M=128` NT shapes with a 3072 dimension, while the same kernel reads
//! 1.09x of MKL at 2048³. This probe separates the two candidate causes:
//!
//!   * **pack amortization** — an `M`-sweep at fixed `(K, N)`. The B panel is packed once per
//!     `(jc, pc)` block regardless of `M`, so its cost per flop falls as `1/M`. If packing is
//!     the cause, throughput must rise monotonically with `M` and flatten once `M >> MC`.
//!   * **peer gap** — the same six shapes the xbench `matmul_skinny` section runs, timed
//!     adjacent to oneMKL single-thread in one process.
//!
//! Same-process, same-run, best-of-N minima with the first run discarded. AGENT-LOCAL,
//! DIRECTIONAL ONLY: never quote the absolute GFLOP/s printed here as a campaign number.
//!
//! Run: `cargo run -p wukong_runtime --release --example gemm_skinny`
//! Env: `WUKONG_MKL_DLL` to point at `mkl_rt.dll`; `GEMM_SKINNY_ONLY=sweep|peer`.

// Measurement probe: every unsafe block calls a runtime kernel entry on buffers this file sized
// for that entry's documented extents. The crate-level SAFETY-comment lint targets the library.
#![allow(clippy::undocumented_unsafe_blocks)]

#[cfg(windows)]
use std::path::PathBuf;
use std::time::Instant;

type CblasSgemmFn = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i64,
    i64,
    i64,
    f32,
    *const f32,
    i64,
    *const f32,
    i64,
    f32,
    *mut f32,
    i64,
);
#[cfg(windows)]
type MklSetNumThreadsFn = unsafe extern "C" fn(i32);

#[allow(dead_code)]
struct Mkl {
    sgemm: CblasSgemmFn,
}

#[cfg(windows)]
fn mkl_dll_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WUKONG_MKL_DLL") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        bases.push(PathBuf::from(prefix).join("Library").join("bin"));
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        for name in [
            "Anaconda3",
            "anaconda3",
            "miniconda3",
            "Miniconda3",
            "miniforge3",
        ] {
            bases.push(PathBuf::from(&home).join(name).join("Library").join("bin"));
        }
    }
    bases.push(PathBuf::from(r"C:\ProgramData\Anaconda3\Library\bin"));
    for base in bases {
        for fname in ["mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll"] {
            let p = base.join(fname);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(not(windows))]
fn load_mkl() -> Option<Mkl> {
    None
}

#[cfg(windows)]
fn load_mkl() -> Option<Mkl> {
    let path = mkl_dll_path()?;
    use libloading::os::windows::{Library as WinLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};
    let lib = unsafe { WinLibrary::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH) }.ok()?;
    let lib: libloading::Library = lib.into();
    let api = unsafe {
        let sgemm = *lib.get::<CblasSgemmFn>(b"cblas_sgemm_64\0").ok()?;
        let set_threads = *lib
            .get::<MklSetNumThreadsFn>(b"MKL_Set_Num_Threads\0")
            .ok()?;
        set_threads(1); // the cap-immune single-core comparison
        println!("oneMKL: {} (pinned to 1 thread)", path.display());
        Mkl { sgemm }
    };
    std::mem::forget(lib);
    Some(api)
}

/// Pin THIS (the timing) thread to one logical CPU. **Instrument only** — this is the dev probe's
/// own thread, never a worker in the shipped kernel (pinning the GEMM's worker pool was measured
/// 25-35% SLOWER on this part; see `gemm_pool`'s `WUKONG_GEMM_AFFINITY` note).
///
/// Necessary here: an unpinned single-thread timing loop migrates across this hybrid 6P+8E+2LP-E
/// part mid-run, and a Crestmont E-core has half the FMA width of a Redwood Cove P-core, so the
/// SAME shape read 40.1 GF/s in one phase and 70.2 GF/s in the next of one unpinned process.
/// Logical 0..11 are the six P-cores' SMT pairs on this part (the topology `gemm_pool`'s affinity
/// probe already encodes), so CPU 0 is a P-core.
#[cfg(windows)]
fn pin_timing_thread(cpu: usize) {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(h: isize, mask: usize) -> usize;
    }
    unsafe {
        SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu);
    }
}
#[cfg(not(windows))]
fn pin_timing_thread(_cpu: usize) {}

/// Best-of-`n` (minimum) seconds per call, first call discarded as warm-up.
fn best_of(n: usize, f: &mut dyn FnMut()) -> f64 {
    f();
    let mut best = f64::INFINITY;
    for _ in 0..n {
        let t = Instant::now();
        f();
        let s = t.elapsed().as_secs_f64();
        if s < best {
            best = s;
        }
    }
    best
}

fn operands(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Same fill pattern as the xbench matmul_skinny section: identical operand values.
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
    let c = vec![0.0f32; m * n];
    (a, b, c)
}

fn wuk_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        wukong_runtime::wukong_sgemm_nt(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m as i64,
            k as i64,
            n as i64,
            0,
        )
    }
}

fn wuk_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        wukong_runtime::wukong_sgemm(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m as i64,
            k as i64,
            n as i64,
            0,
        )
    }
}

/// The six xbench `matmul_skinny` shapes.
const SHAPES: [(usize, usize, usize); 6] = [
    (128, 768, 768),
    (128, 768, 3072),
    (128, 3072, 768),
    (512, 768, 768),
    (512, 768, 3072),
    (512, 3072, 768),
];

fn main() {
    let only = std::env::var("GEMM_SKINNY_ONLY").unwrap_or_default();
    let reps = 5usize;
    let cpu: usize = std::env::var("GEMM_SKINNY_CPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    pin_timing_thread(cpu);
    println!(
        "AGENT-LOCAL, DIRECTIONAL ONLY. best-of-{reps}, warm-up discarded, one process, \
         timing thread pinned to logical CPU {cpu} (a P-core).\n"
    );

    let mkl = load_mkl();

    if only.is_empty() || only == "sweep" {
        // --- M-amortization sweep, against MKL's own M-curve ----------------------------
        // Fixed (K, N); only M varies. Packed-B cost is O(K*N) and independent of M, so its
        // share of the runtime falls as 1/M. A flat Wukong curve would rule packing out; MKL
        // in the same column says whether small M is intrinsically hard or only hard for us.
        for (k, n) in [(3072usize, 768usize), (768, 3072), (768, 768)] {
            println!("=== NT M-sweep at K={k}, N={n} (GFLOP/s) ===");
            println!("  {:<7} {:>9} {:>9} {:>8}", "M", "Wuk", "MKL", "ratio");
            for m in [128usize, 192, 256, 384, 512, 768, 1536] {
                let (a, b, mut c) = operands(m, k, n);
                let mut cm = vec![0.0f32; m * n];
                let (mi, ki, ni) = (m as i64, k as i64, n as i64);
                let (apr, bpr, cmp) = (a.as_ptr(), b.as_ptr(), cm.as_mut_ptr());
                let flop = 2.0 * m as f64 * k as f64 * n as f64;
                let sw = best_of(reps, &mut || wuk_nt(&a, &b, &mut c, m, k, n));
                let sm = mkl.as_ref().map(|x| {
                    best_of(reps, &mut || unsafe {
                        (x.sgemm)(
                            101, 111, 112, mi, ni, ki, 1.0, apr, ki, bpr, ki, 0.0, cmp, ni,
                        )
                    })
                });
                let gw = flop / sw / 1e9;
                match sm {
                    // ratio = Wukong GF/s over MKL GF/s = MKL seconds over Wukong seconds.
                    Some(s) => {
                        println!("  M={m:<5} {gw:9.1} {:9.1} {:8.3}", flop / s / 1e9, s / sw)
                    }
                    None => println!("  M={m:<5} {gw:9.1} {:>9} {:>8}", "n/a", "n/a"),
                }
            }
            println!();
        }
    }

    if only.is_empty() || only == "packcost" {
        // --- What the SCALAR transposing B-packer costs ---------------------------------
        // NN and NT run the identical five-loop, identical 6x16 microkernel and identical
        // A-packer, and pack exactly the same COUNT of B elements. The only difference is
        // which B-packer runs: NN's `pack_b_panel` is a contiguous 16-float row copy that
        // LLVM vectorizes; NT's `pack_b_trans_panel` is an element-at-a-time transpose with
        // a 64-byte store stride. So (NT time - NN time) / NT time is a direct, in-process
        // upper bound on what SIMD-ising the transposing packer can win.
        println!("=== transposing-B-pack penalty: same shape, NT vs NN (GFLOP/s) ===");
        println!(
            "  {:<20} {:>9} {:>9} {:>8}",
            "M x K x N", "NT", "NN", "NT/NN"
        );
        for (m, k, n) in SHAPES {
            let (a, b, mut c) = operands(m, k, n);
            let flop = 2.0 * m as f64 * k as f64 * n as f64;
            // ABBA over the two layouts; `b` is reinterpreted, not re-laid-out, because only
            // timing is compared here (values differ between the two calls, cost does not).
            let t1 = best_of(reps, &mut || wuk_nt(&a, &b, &mut c, m, k, n));
            let u1 = best_of(reps, &mut || wuk_nn(&a, &b, &mut c, m, k, n));
            let u2 = best_of(reps, &mut || wuk_nn(&a, &b, &mut c, m, k, n));
            let t2 = best_of(reps, &mut || wuk_nt(&a, &b, &mut c, m, k, n));
            let (t, u) = (t1.min(t2), u1.min(u2));
            println!(
                "  {:<20} {:9.1} {:9.1} {:8.3}",
                format!("{m}x{k}x{n}"),
                flop / t / 1e9,
                flop / u / 1e9,
                u / t
            );
        }
        println!();
    }

    if only.is_empty() || only == "peer" {
        // --- The six xbench matmul_skinny shapes, adjacent to MKL(1c) --------------------
        let Some(mkl) = mkl.as_ref() else {
            eprintln!("oneMKL not found — set WUKONG_MKL_DLL; skipping the peer table.");
            return;
        };
        println!("=== matmul_skinny shapes vs oneMKL(1 thread) ===");
        println!(
            "  {:<22} {:>9} {:>9} {:>8}",
            "M x K . (N x K)t", "Wuk", "MKL", "ratio"
        );
        for (m, k, n) in SHAPES {
            let (a, b, mut c) = operands(m, k, n);
            let mut cm = vec![0.0f32; m * n];
            let (mi, ki, ni) = (m as i64, k as i64, n as i64);
            let (ap, bp) = (a.as_ptr(), b.as_ptr());
            let cmp = cm.as_mut_ptr();
            // CblasRowMajor=101, CblasNoTrans=111, CblasTrans=112: C[m,n] = A[m,k] * B[n,k]^T.
            let mut mklc = || unsafe {
                (mkl.sgemm)(101, 111, 112, mi, ni, ki, 1.0, ap, ki, bp, ki, 0.0, cmp, ni)
            };
            wuk_nt(&a, &b, &mut c, m, k, n);
            mklc();
            let max_rel = c
                .iter()
                .zip(&cm)
                .map(|(x, y)| (x - y).abs() as f64 / (y.abs() as f64).max(1e-6))
                .fold(0.0f64, f64::max);
            assert!(
                max_rel < 1e-4,
                "Wukong vs MKL disagree (rel {max_rel:.1e}) at {m}x{k}x{n}"
            );
            let flop = 2.0 * m as f64 * k as f64 * n as f64;
            // ABBA: W M M W, take each side's min, so neither side owns a slot.
            let w1 = best_of(reps, &mut || wuk_nt(&a, &b, &mut c, m, k, n));
            let m1 = best_of(reps, &mut mklc);
            let m2 = best_of(reps, &mut mklc);
            let w2 = best_of(reps, &mut || wuk_nt(&a, &b, &mut c, m, k, n));
            let (wg, mg) = (flop / w1.min(w2) / 1e9, flop / m1.min(m2) / 1e9);
            println!(
                "  {:<22} {wg:9.1} {mg:9.1} {:8.3}",
                format!("{m}x{k} . ({n}x{k})t"),
                wg / mg
            );
        }
    }
}
