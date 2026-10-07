//! **Measurement-position** probe for the all-core GEMM comparison (dev instrument, not a gate).
//!
//! `wukong_xbench`'s `matmul`/`linear` sections publish `Wuk(par)` against `MKL(all)` from a
//! FIXED sequence: the single-core group first, then `MKL(all)`, then the all-core `C(omp)` nest,
//! and `Wuk(par)` LAST. The harness documents that ordering as deliberate ("we throttle ourselves,
//! never the competitor"), and it also documents — in `abba_min`'s own comment — that it measured
//! a **+5.0% to +17.3%** second-slot penalty between two *single-core* peers, large enough to
//! swamp the language differences it prints. Nobody had measured the same penalty for the
//! ALL-CORE slot, where a preceding multi-second all-core burst throttles this 28 W laptop part.
//!
//! This probe measures it with an **identical-arm control**, the discipline the repo already uses
//! for the C(twin) clock experiment: it replays the xbench sequence and then times, at the tail,
//! `Wuk(par)` and `MKL(all)` **A B B A** with each column keeping its own minimum. Three ratios
//! come out:
//!
//!   * `W(late)/M(early)` — what xbench publishes (Wukong last, MKL early);
//!   * `W(late)/M(late)`  — the same two kernels compared in the SAME slot (position removed);
//!   * `M(late)/M(early)` — the **twin control**: one identical kernel, two slots. Whatever this
//!     is below 1.00 is pure measurement position, not a property of any kernel.
//!
//! The Wukong arm here is `wukong_sgemm[_nt]_parallel` called directly, which is exactly what
//! xbench's `Wuk(par)` runs: its `@parallel` `.wk` program lowers to a single
//! `call wukong_sgemm_parallel(a, b, c, n, n, n, 0)` (matmul) or
//! `call wukong_sgemm_nt_parallel(...)` (linear) and nothing else — verified with
//! `wukongc --emit=mir -O3` on the exact sources `wk_matmul`/`wk_linear` generate — so no compiler
//! machinery is in the loop on either instrument.
//!
//! `GEMM_ORDER_NT=1` switches both arms to the `C = A·Bᵀ` (`nn.Linear`) layout, which is the
//! xbench `linear` section's shape. That layout has no coverage in `gemm_var` at all, so it is
//! the only controlled peer number available for those rows.
//!
//! Run: `cargo run -p wukong_runtime --release --example gemm_order`
//! Env: `WUKONG_MKL_DLL` to point at `mkl_rt.dll`; `GEMM_ORDER_SIZES=512,1024` to pick sizes
//! (default `512,1024`); `GEMM_ORDER_NT=1` for the `A·Bᵀ` layout; `GEMM_ORDER_NO_PRELUDE=1` to
//! skip the single-core group.

#[cfg(windows)]
use std::path::PathBuf;
use std::time::{Duration, Instant};

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
#[cfg(windows)]
type MklGetMaxThreadsFn = unsafe extern "C" fn() -> i32;

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

/// The MKL peer resolves `mkl_rt.dll` through the Windows loader; elsewhere the column is absent.
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
        let max_threads = (*lib
            .get::<MklGetMaxThreadsFn>(b"MKL_Get_Max_Threads\0")
            .ok()?)();
        set_threads(max_threads);
        println!(
            "oneMKL: {} (MKL_Get_Max_Threads = {max_threads}, the xbench MKL(all) setting)",
            path.display()
        );
        Mkl { sgemm }
    };
    std::mem::forget(lib);
    Some(api)
}

/// `wukong_xbench::time_ns`, replicated in behaviour: warm up, grow the batch to ~50 ms, then take
/// the fastest of 14 batches (or best-of-6 single calls when one call already exceeds the target).
/// The two instruments must not differ in their timing rule.
fn time_ns(mut run: impl FnMut()) -> f64 {
    run();
    run();
    let probe = {
        let t = Instant::now();
        run();
        t.elapsed()
    };
    let target = Duration::from_millis(50);
    if probe >= target {
        let mut best = probe;
        for _ in 0..6 {
            let t = Instant::now();
            run();
            let e = t.elapsed();
            if e < best {
                best = e;
            }
        }
        return best.as_secs_f64() * 1e9;
    }
    run();
    run();
    let mut reps = 1u64;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            run();
        }
        let e = t.elapsed();
        if e >= target || reps >= (1 << 26) {
            let mut best = e;
            for _ in 0..14 {
                let t = Instant::now();
                for _ in 0..reps {
                    run();
                }
                let e2 = t.elapsed();
                if e2 < best {
                    best = e2;
                }
            }
            return best.as_secs_f64() * 1e9 / reps as f64;
        }
        reps = reps.saturating_mul(2);
    }
}

/// The naive `ikj` nest the C/C++/Rust peers run, in Rust — the single-core group's workload.
/// Layout-agnostic on purpose: its only job is to hold ONE core at load for a few seconds, and
/// the position question does not depend on which nest does that.
///
/// # Safety
/// `a`/`b` hold `ns²` floats; `c` is writable for `ns²` floats and aliases neither.
unsafe fn naive_ikj(a: *const f32, b: *const f32, c: *mut f32, ns: usize) {
    unsafe {
        for i in 0..ns {
            for j in 0..ns {
                *c.add(i * ns + j) = 0.0;
            }
            for k in 0..ns {
                let aik = *a.add(i * ns + k);
                for j in 0..ns {
                    *c.add(i * ns + j) += aik * *b.add(k * ns + j);
                }
            }
        }
    }
}

/// The `C(omp)` peer's shape: the same naive nest with the `i` loop split across all cores. Its
/// role in the sequence is to hold every core at full AVX2/FMA load for ~1-2 s immediately before
/// the slot Wukong is measured in.
fn naive_ikj_par(a: &[f32], b: &[f32], c: &mut [f32], ns: usize) {
    use rayon::prelude::*;
    c.par_chunks_mut(ns).enumerate().for_each(|(i, row)| {
        row.iter_mut().for_each(|v| *v = 0.0);
        for k in 0..ns {
            let aik = a[i * ns + k];
            let brow = &b[k * ns..k * ns + ns];
            for (j, rj) in row.iter_mut().enumerate() {
                *rj += aik * brow[j];
            }
        }
    });
}

fn main() {
    let Some(mkl) = load_mkl() else {
        eprintln!("oneMKL not found - set WUKONG_MKL_DLL. The position question needs the peer.");
        std::process::exit(1);
    };
    let sizes: Vec<usize> = std::env::var("GEMM_ORDER_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![512, 1024]);
    let prelude = std::env::var("GEMM_ORDER_NO_PRELUDE").is_err();
    // `nt` selects the `nn.Linear` layout (`C = A·Bᵀ`, B stored [N,K]) — the xbench `linear`
    // section's shape, and the one `gemm_var` never measures.
    let nt = std::env::var("GEMM_ORDER_NT").is_ok();
    println!(
        "rayon(global)={} physical={}  sizes={sizes:?}  layout={}  single-core prelude={prelude}",
        rayon::current_num_threads(),
        num_cpus::get_physical(),
        if nt {
            "C=A.Bt (linear)"
        } else {
            "C=A.B (matmul)"
        }
    );
    println!(
        "sequence per size: [serial group] -> MKL(all) EARLY -> C(omp)-shaped all-core burn -> \
         tail A B B A of Wuk(par) / MKL(all)"
    );

    for ns in sizes {
        let n2 = ns * ns;
        // The xbench matmul fill, element for element.
        let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut cw = vec![0.0f32; n2];
        let mut cm = vec![0.0f32; n2];
        let mut cn = vec![0.0f32; n2];
        let flop = 2.0 * (ns as f64).powi(3);
        let (ni, ap, bp) = (ns as i64, a.as_ptr(), b.as_ptr());
        let cwp = cw.as_mut_ptr();
        let cmp = cm.as_mut_ptr();

        // CBLAS: layout=101 (row major), transa=111 (no trans), transb=111/112. `ldb` follows the
        // storage: B is [N,K] under NT (so ldb = K = ns) and [K,N] otherwise — square here, so the
        // leading dimension is `ns` either way, exactly as `bench_mm_mkl_rect` computes it.
        let transb = if nt { 112 } else { 111 };
        let mut wuk = || unsafe {
            if nt {
                wukong_runtime::wukong_sgemm_nt_parallel(ap, bp, cwp, ni, ni, ni, 0)
            } else {
                wukong_runtime::wukong_sgemm_parallel(ap, bp, cwp, ni, ni, ni, 0)
            }
        };
        let mut mklc = || unsafe {
            (mkl.sgemm)(
                101, 111, transb, ni, ni, ni, 1.0, ap, ni, bp, ni, 0.0, cmp, ni,
            )
        };

        // ABI + correctness check before any timing: a garbage MKL layout would otherwise read as
        // a fast wrong answer (the gemm_var discipline).
        wuk();
        mklc();
        let max_rel = cw
            .iter()
            .zip(&cm)
            .map(|(x, y)| (x - y).abs() as f64 / (y.abs() as f64).max(1e-6))
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 1e-4,
            "Wukong vs MKL disagree (rel {max_rel:.1e}) at {ns}^3"
        );

        println!("\n=== {ns}^3 {} ===", if nt { "A.Bt" } else { "A.B" });
        // The single-core group: Wukong serial, then three naive nests (xbench runs six
        // single-core timed regions here; the point is only that the package sits at one-core
        // load for several seconds before the all-core group starts).
        if prelude {
            let cnp = cn.as_mut_ptr();
            let t = Instant::now();
            let ser = time_ns(|| unsafe {
                if nt {
                    wukong_runtime::wukong_sgemm_nt(ap, bp, cnp, ni, ni, ni, 0);
                } else {
                    wukong_runtime::wukong_sgemm(ap, bp, cnp, ni, ni, ni, 0);
                }
            });
            for _ in 0..3 {
                time_ns(|| unsafe { naive_ikj(ap, bp, cnp, ns) });
            }
            println!(
                "  serial group done in {:.1} s (Wuk(1c) {:.1} GF/s)",
                t.elapsed().as_secs_f64(),
                flop / ser
            );
        }

        // All-core group, xbench order: MKL(all) EARLY.
        let m_early = flop / time_ns(&mut mklc);

        // The C(omp) slot: all-core naive nest, the burn Wukong inherits.
        let t = Instant::now();
        let comp = flop / time_ns(|| naive_ikj_par(&a, &b, &mut cn, ns));
        let burn = t.elapsed().as_secs_f64();

        // Tail: A B B A over the two all-core arms, each column keeping its own minimum (xbench's
        // own `abba_min` rule, applied to the all-core pair it does not apply it to).
        let w1 = time_ns(&mut wuk);
        let m1 = time_ns(&mut mklc);
        let m2 = time_ns(&mut mklc);
        let w2 = time_ns(&mut wuk);
        let w_late = flop / w1.min(w2);
        let m_late = flop / m1.min(m2);
        // The FIRST tail sample of Wukong is the closest analogue of the published number: xbench
        // times Wuk(par) exactly once, immediately after the C(omp) burn.
        let w_first = flop / w1;

        println!(
            "  MKL(all) early {m_early:7.1} | C(omp)-shaped burn {comp:7.1} GF/s over {burn:.1} s"
        );
        println!(
            "  tail ABBA: Wuk {:7.1}/{:7.1}  MKL {:7.1}/{:7.1} GF/s (slot order W,M,M,W)",
            flop / w1,
            flop / w2,
            flop / m1,
            flop / m2
        );
        println!(
            "  -> as xbench publishes it   W(first,late)/M(early) = {:.3}x",
            w_first / m_early
        );
        println!(
            "  -> position removed         W(late)/M(late)        = {:.3}x",
            w_late / m_late
        );
        println!(
            "  -> TWIN CONTROL             M(late)/M(early)       = {:.3}x  <- pure slot penalty",
            m_late / m_early
        );
    }
}
