//! Mid-size GEMM **run-variance** probe (dev instrument, not a shipped bench).
//!
//! The perf-perfect ledger's open A(a) question: 512³/1024³ read 76-97% of same-run MKL(all)
//! across healthy-state runs — whose variance is that? This probe alternates Wukong-parallel and
//! MKL(all) batches **per round, adjacently, with the order swapped every round** (the ABBA
//! discipline), printing every round's GFLOP/s rather than a single best-of. Three outcomes it
//! can distinguish:
//!   * both sides swing together        → machine state (clock/thermal/background), not Wukong;
//!   * only Wukong swings               → our scheduling/packing has a nondeterministic stall;
//!   * per-round ratio tight in-process but different across invocations → process-level state
//!     (pool placement, page placement).
//!
//! Run: `cargo run -p wukong_runtime --release --example gemm_var [-- <rounds>]`
//! Env: `WUKONG_MKL_DLL` to point at mkl_rt.dll explicitly; `GEMM_VAR_HUGE=1` adds 2048³.
//!
//! **Skinny and `nn.Linear` shapes (P6.7).** Finding 01 withdrew twelve `Wuk(par)`-vs-`MKL(all)`
//! rows on the strength of THIS instrument, but it only ever ran square `C = A·B`: six of the twelve
//! are skinny NT shapes and two are square NT (`linear`), and none of the eight had better-instrument
//! evidence of its own. `GEMM_VAR_NT=1` appends exactly those eight - the six `matmul_skinny` shapes
//! and the two `linear` sizes xbench measures - timed through `wukong_sgemm_nt_parallel` against
//! `cblas_sgemm` with `transb = CblasTrans`, under the same order-swapped rounds. `GEMM_VAR_SHAPES=
//! MxKxN[t],...` replaces the whole list with explicit shapes (`t` = NT, `B` stored `[N, K]`).

// Measurement probe: every unsafe block calls a runtime kernel entry on buffers this file sized
// for that entry's documented extents. The crate-level SAFETY-comment lint targets the library.
#![allow(clippy::undocumented_unsafe_blocks)]

#[cfg(windows)]
use std::path::PathBuf;
use std::time::Instant;

// cblas_sgemm_64 (ILP64 by-value CBLAS), MKL_Set_Num_Threads (by value — the lowercase twin is
// the Fortran by-reference binding and segfaults), MKL_Get_Max_Threads. Same resolution rules as
// the xbench peer loader.
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
    max_threads: i32,
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

/// The MKL peer resolves `mkl_rt.dll` through the Windows loader; elsewhere the column is absent
/// (the un-gated `libloading::os::windows` import broke `--all-targets` on Linux, E0432).
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
            "oneMKL: {} (pinned to {max_threads} threads)",
            path.display()
        );
        Mkl { sgemm, max_threads }
    };
    std::mem::forget(lib);
    Some(api)
}

/// One GEMM shape: `C[M,N] = A[M,K] · B`, with `B` stored `[K,N]` (NN) or `[N,K]` (`nt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    m: usize,
    k: usize,
    n: usize,
    nt: bool,
}

impl Shape {
    fn square(n: usize, nt: bool) -> Shape {
        Shape { m: n, k: n, n, nt }
    }

    /// Square NN keeps its historical `256³` heading (Finding 01 quotes it); everything else
    /// names its shape the way xbench's `matmul_skinny` / `linear` headings do.
    fn label(&self) -> String {
        match (self.nt, self.m == self.k && self.k == self.n) {
            (false, true) => format!("{}³", self.n),
            (false, false) => format!("{}x{} · {}x{} (NN)", self.m, self.k, self.k, self.n),
            (true, _) => format!("{}x{} · ({}x{})ᵀ (NT)", self.m, self.k, self.n, self.k),
        }
    }
}

/// The eight NT shapes Finding 01 withdrew without an instrument of their own: xbench's six
/// `matmul_skinny` shapes `(M, K, N)` and its two `linear` sizes.
fn nt_presets() -> Vec<Shape> {
    let skinny = [
        (128, 768, 768),
        (128, 768, 3072),
        (128, 3072, 768),
        (512, 768, 768),
        (512, 768, 3072),
        (512, 3072, 768),
    ];
    skinny
        .into_iter()
        .map(|(m, k, n)| Shape { m, k, n, nt: true })
        .chain([512, 1024].into_iter().map(|n| Shape::square(n, true)))
        .collect()
}

/// `GEMM_VAR_SHAPES=128x768x3072t,1024x1024x1024` - `None` on anything malformed, so a typo is an
/// error rather than a silently shorter sweep.
fn parse_shapes(s: &str) -> Option<Vec<Shape>> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            let (dims, nt) = match t.strip_suffix('t') {
                Some(d) => (d, true),
                None => (t, false),
            };
            let v: Vec<usize> = dims
                .split('x')
                .map(|d| d.parse().ok().filter(|&x: &usize| x > 0))
                .collect::<Option<Vec<_>>>()?;
            match v[..] {
                [m, k, n] => Some(Shape { m, k, n, nt }),
                _ => None,
            }
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Coefficient of variation, percent.
fn cov(v: &[f64]) -> f64 {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
    var.sqrt() / mean * 100.0
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let Some(mkl) = load_mkl() else {
        eprintln!("oneMKL not found — set WUKONG_MKL_DLL. The variance question needs the peer.");
        std::process::exit(1);
    };
    println!(
        "rayon(global)={} physical={}  rounds={rounds}  (order swaps every round: even=W->M, odd=M->W)",
        rayon::current_num_threads(),
        num_cpus::get_physical()
    );

    let mut sizes = vec![256usize, 512, 1024];
    if std::env::var("GEMM_VAR_HUGE").is_ok() {
        sizes.push(2048);
    }
    // GEMM_VAR_SIZES=512[,1024,…]: restrict to the named sizes for fast episode-hunting loops.
    if let Ok(s) = std::env::var("GEMM_VAR_SIZES") {
        let keep: Vec<usize> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if !keep.is_empty() {
            sizes.retain(|n| keep.contains(n));
        }
    }
    let mut shapes: Vec<Shape> = sizes.into_iter().map(|n| Shape::square(n, false)).collect();
    if std::env::var("GEMM_VAR_NT").is_ok() {
        shapes.extend(nt_presets());
    }
    if let Ok(s) = std::env::var("GEMM_VAR_SHAPES") {
        match parse_shapes(&s) {
            Some(list) if !list.is_empty() => shapes = list,
            _ => {
                eprintln!("GEMM_VAR_SHAPES={s:?} is not a list of MxKxN[t] - refusing to guess");
                std::process::exit(2);
            }
        }
    }
    for shape in shapes {
        let Shape { m, k, n, nt } = shape;
        // Same fill pattern as the xbench matmul sections — identical operand values. `B` is
        // `[K, N]` for NN and `[N, K]` for NT (the `nn.Linear` weight layout), `k * n` either way.
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut cw = vec![0.0f32; m * n];
        let mut cm = vec![0.0f32; m * n];
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let (mi, ki, ni, ap, bp) = (m as i64, k as i64, n as i64, a.as_ptr(), b.as_ptr());
        let cwp = cw.as_mut_ptr();
        let cmp = cm.as_mut_ptr();
        // Row-major leading dimensions follow the storage: lda = K, ldc = N, and ldb = N for a
        // `[K, N]` B or K for an `[N, K]` B taken with transb = CblasTrans (112).
        let (transb, ldb) = if nt { (112, ki) } else { (111, ni) };

        let mut wuk = || unsafe {
            if nt {
                wukong_runtime::wukong_sgemm_nt_parallel(ap, bp, cwp, mi, ki, ni, 0)
            } else {
                wukong_runtime::wukong_sgemm_parallel(ap, bp, cwp, mi, ki, ni, 0)
            }
        };
        let mut mklc = || unsafe {
            (mkl.sgemm)(
                101, 111, transb, mi, ni, ki, 1.0, ap, ki, bp, ldb, 0.0, cmp, ni,
            )
        };
        let label = shape.label();

        // Warm both sides (pool spawn, packing scratch, MKL's own thread wake) and ABI-check once:
        // a garbage MKL layout would otherwise read as a fast wrong answer.
        for _ in 0..3 {
            wuk();
            mklc();
        }
        let max_rel = cw
            .iter()
            .zip(&cm)
            .map(|(x, y)| {
                let d = (x - y).abs() as f64;
                d / (y.abs() as f64).max(1e-6)
            })
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 1e-4,
            "Wukong vs MKL disagree (rel {max_rel:.1e}) at {label}"
        );

        // Batch reps sized off a calibration call so one batch is ~30 ms — long enough to blur
        // scheduler noise, short enough that 12 rounds×2 sides stay in one thermal window.
        let t = Instant::now();
        wuk();
        let one_call = t.elapsed().as_secs_f64();
        let reps = ((0.030 / one_call).round() as usize).clamp(2, 400);

        println!("\n=== {label}  (reps/batch={reps}, one timed batch per side per round) ===");
        let batch = |f: &mut dyn FnMut()| -> f64 {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            flop / (t.elapsed().as_nanos() as f64 / reps as f64)
        };
        // GEMM_VAR_SOLO=1: time ONLY Wukong batches back-to-back (MKL silent after the warmup/
        // ABI check) — splits "intrinsic Wukong variance" from "cross-pool interaction with the
        // adjacent MKL batches". Ratio lines are meaningless in this mode and are suppressed.
        if std::env::var("GEMM_VAR_SOLO").is_ok() {
            let mut ws = Vec::new();
            for r in 0..rounds {
                // Per-call timing inside the batch: a slow round with a normal p50 but huge max is
                // discrete stall events; a slow round with every percentile shifted is a state
                // (clock, worker placement) that held for the whole batch.
                let mut calls: Vec<f64> = (0..reps)
                    .map(|_| {
                        let t = Instant::now();
                        wuk();
                        t.elapsed().as_nanos() as f64
                    })
                    .collect();
                let w = flop / (calls.iter().sum::<f64>() / reps as f64);
                calls.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let ms = |x: f64| x / 1e6;
                println!(
                    "  r{r:02} solo  wuk {w:7.1}  call ms min/p50/p90/max {:.2}/{:.2}/{:.2}/{:.2}",
                    ms(calls[0]),
                    ms(calls[reps / 2]),
                    ms(calls[(reps * 9 / 10).min(reps - 1)]),
                    ms(calls[reps - 1])
                );
                ws.push(w);
            }
            let (wmin, wmax) = (
                ws.iter().cloned().fold(f64::MAX, f64::min),
                ws.iter().cloned().fold(0.0, f64::max),
            );
            println!(
                "  wuk  min/med/max {wmin:.1}/{:.1}/{wmax:.1} GF/s  CoV {:.1}%",
                median(&mut ws.clone()),
                cov(&ws)
            );
            continue;
        }
        let (mut ws, mut ms, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..rounds {
            let (w, m) = if r % 2 == 0 {
                let w = batch(&mut wuk);
                let m = batch(&mut mklc);
                (w, m)
            } else {
                let m = batch(&mut mklc);
                let w = batch(&mut wuk);
                (w, m)
            };
            println!(
                "  r{r:02} {}  wuk {w:7.1}  mkl {m:7.1}  ratio {:5.1}%",
                if r % 2 == 0 { "W->M" } else { "M->W" },
                w / m * 100.0
            );
            ws.push(w);
            ms.push(m);
            ratios.push(w / m * 100.0);
        }
        let (wmin, wmax) = (
            ws.iter().cloned().fold(f64::MAX, f64::min),
            ws.iter().cloned().fold(0.0, f64::max),
        );
        let (mmin, mmax) = (
            ms.iter().cloned().fold(f64::MAX, f64::min),
            ms.iter().cloned().fold(0.0, f64::max),
        );
        let (rmin, rmax) = (
            ratios.iter().cloned().fold(f64::MAX, f64::min),
            ratios.iter().cloned().fold(0.0, f64::max),
        );
        println!(
            "  wuk  min/med/max {wmin:.1}/{:.1}/{wmax:.1} GF/s  CoV {:.1}%",
            median(&mut ws.clone()),
            cov(&ws)
        );
        println!(
            "  mkl  min/med/max {mmin:.1}/{:.1}/{mmax:.1} GF/s  CoV {:.1}%",
            median(&mut ms.clone()),
            cov(&ms)
        );
        println!(
            "  ratio min/med/max {rmin:.1}/{:.1}/{rmax:.1} %  (MKL pinned {} threads)",
            median(&mut ratios.clone()),
            mkl.max_threads
        );
    }
}
