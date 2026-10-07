//! `wukong-xbench` — an honest cross-language benchmark.
//!
//! For each kernel it builds the *same* computation several ways — Wukong (compiled to native code
//! by the Cranelift backend), C (gcc `-O3 -march=native`), C++ (g++ at the same flags over the same
//! numeric body, rendered from the C source by [`cpp_from_c`]), and Rust (rustc `-C opt-level=3 -C
//! target-cpu=native`) — and times them all through one identical Rust loop over the same
//! buffers. The peers are compiled to shared libraries at **runtime** and called via their C ABI
//! (`CC` / `CXX` override the C / C++ compiler, default `gcc` / `g++`; the Rust peer is always
//! `rustc` — there is no clang/llc on this box); Wukong is JIT-compiled in process, and its `.wk`
//! sources are likewise generated and compiled at runtime. It also reports each toolchain's compile
//! time. Further peer columns appear only where they apply: `C(fast)` (the same C source at
//! `-ffast-math`, on the reduction-bearing rows), `C(omp)` (the same kernel under `#pragma omp
//! parallel for`, on the `@parallel` rows), oneMKL (cblas + VML), the pure-Rust `matrixmultiply`
//! sgemm, and — in the `model` section — PyTorch. A missing toolchain prints a note and shows
//! `n/a`; it never fails the run. The first CLI argument, if given, is a substring filter over
//! kernel/section names.
//!
//! Fairness notes:
//!  * ALIASING: every generated C/C++ peer declares its kernel parameters `__restrict__` (the GCC
//!    spelling accepted in both gcc's C mode and g++'s C++ mode, so the C++ column rendered by
//!    [`cpp_from_c`] inherits it). Without it gcc must assume the output buffer may overlap the
//!    inputs and cannot vectorize or reorder a nested kernel — a handicap no competent C programmer
//!    would accept for a kernel with distinct in/out buffers. Wukong's own tensor parameters carry
//!    that non-overlap in the type system, so withholding `restrict` from the peer compared a
//!    no-alias compiler against a may-alias one. Every call site in this file passes three (or four)
//!    genuinely distinct allocations, which is what makes the qualifier true and not just fast.
//!    MEASURING THIS CORRECTLY MATTERS: a standalone probe that puts the kernel and its caller in
//!    ONE translation unit over `static` arrays lets gcc's interprocedural alias analysis prove
//!    non-overlap by itself, and `restrict` then looks like a no-op. It is not — the peers here are
//!    compiled to a **shared library**, where gcc sees only pointer parameters. Re-measured in that
//!    model (kernels in their own TU, gcc 14.2 `-O3 -march=native`, best-of-N, one process):
//!    `matmul_tn` 512³ **176.6 -> 20.2 ms (8.7x)**, `colsum` 4096x1024 **21.3 -> 5.0 ms (4.2x)`,
//!    the direct convolution **1.99 -> 0.40 ms (5.0x)**, `saxpy` 1.33x; `relu`/`biasadd` genuinely
//!    unaffected (0.96x / 1.01x — gcc already versions those loops with a runtime alias check).
//!  * FMA: Wukong now contracts `x + y*z` to a fused multiply-add, so gcc is given its *default*
//!    `-ffp-contract=fast` (the old `-ffp-contract=off` was actually suppressing C's natural FMA).
//!    Both Wukong and gcc-compiled C therefore fuse. Idiomatic Rust does *not* contract unless the
//!    author writes `f32::mul_add`, so the Rust column reflects rustc's default (two rounded ops) —
//!    a real toolchain-defaults difference, not a handicap.
//!  * Reductions (dot) are strict left-to-right f32, so none of the three auto-vectorize them
//!    (though both Wukong and C may use a *scalar* FMA for the `s + x*y` step).
//!  * The comparison basis is *idiomatic, single-threaded* code at the given flags. Where Wukong
//!    later auto-parallelizes/vectorizes, that is called out explicitly.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use wukong_span::{Interner, SourceId};

mod compare;
mod general;
mod model;
mod sota;

/// The shared C ABI of every kernel: `(x, y, out)` over `N` `f32` elements (`N` baked in).
type KernelFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

/// The int8 quantized-GEMM ABI: `(a: u8, b: i8, c: i32)` — `u8` activations, `i8` weights, `i32`
/// accumulator (the QNNPACK/oneDNN layout). Dims are baked into the kernel source.
type I8KernelFn = unsafe extern "C" fn(*const u8, *const i8, *mut i32);

const N: usize = 1 << 20; // 1,048,576 elements (4 MiB per f32 array)

/// One row of the elementwise table: the same computation in three source languages behind the
/// shared `(x, y, out)` ABI (the C++ column is rendered from `c` by [`cpp_from_c`], so it carries no
/// field of its own). `x` and `out` are read-only input / write-only output for every row; the
/// MIDDLE buffer `y` is an input for most rows but is used as a WRITABLE scratch array by
/// `fused_linear_relu` (whose peers stream the fused intermediate through it), so `main` derives
/// `yp` with `as_mut_ptr` rather than `as_ptr`.
struct Kernel {
    name: &'static str,
    /// bytes of memory traffic per call (for a GB/s figure)
    bytes_per_call: usize,
    /// 2*flops per element estimate (for context only)
    note: &'static str,
    wuk: String,
    c: String,
    rust: String,
}

#[derive(Clone)]
struct Measure {
    compile: Duration,
    ns_per_call: f64,
    /// A snapshot of the kernel's full output buffer, so the cross-language check can diff *every*
    /// element (not a 3-point sample, which a kernel wrong everywhere else would slip past).
    out: Vec<f32>,
}

/// The int8 twin of [`Measure`]: full `i32` output snapshot for an exact (integer) cross-check.
#[derive(Clone)]
struct MeasureI8 {
    compile: Duration,
    ns_per_call: f64,
    out: Vec<i32>,
}

/// The bf16-reduction ABI: `(x, y: *const u16, o: *mut f32)` — bf16 stored bits in, the f32 scalar
/// accumulator written to `o[0]`. `y` is unused (aliases `x`) for the unary sum.
type Bf16KernelFn = unsafe extern "C" fn(*const u16, *const u16, *mut f32);

/// The bf16 twin of [`Measure`]: just the scalar reduction result (checked across languages with a
/// tight relative tolerance — the three reassociate the f32 sum differently).
#[derive(Clone)]
struct MeasureBf16 {
    compile: Duration,
    ns_per_call: f64,
    out: f32,
}

/// The **all-half** streaming ABI: `(x, y, out)` all `*const/*mut u16` (bf16 stored bits). Unlike
/// [`Bf16KernelFn`] (bf16 in, f32 scalar out) the output is a full half buffer — the narrowing store.
type HalfOutKernelFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16);

/// The all-half twin of [`Measure`]: a full snapshot of the half output buffer widened to f32, so the
/// cross-language check diffs *every* element (bf16 rounds to ~8 mantissa bits, so a bf16-scale
/// relative tolerance is the honest bar — Wukong's FMA'd sum vs C's `a·x+b·y` can round to a
/// neighbouring bf16 in the last bit).
#[derive(Clone)]
struct MeasureHalfOut {
    compile: Duration,
    ns_per_call: f64,
    out: Vec<f32>,
}

/// The maximum relative element-wise error between two output buffers, and the index where it
/// occurs. NaN-vs-NaN and same-sign-Inf agree; a small absolute floor keeps near-zero elements from
/// blowing up the ratio. This is the honest full-buffer cross-language equality check: the three
/// languages compute *slightly* differently (Wukong's ≈1-ULP poly vs libm `expf`, FMA vs not), so
/// it is a tight tolerance rather than bit-exactness — but it sees all N elements.
fn max_rel_err(a: &[f32], b: &[f32]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        if x == y {
            continue; // exact (covers ±0 and equal Inf)
        }
        // NaN vs finite is a DIVERGENCE, and the loudest one there is. Falling through would make
        // `rel` NaN, and `NaN > worst` is false — so the pair would be silently skipped and the
        // function would return its "bit-exact" sentinel 0.0 for buffers that do not match. That
        // is exactly how a broken kernel (an unwritten row seeding a fused softmax's max with
        // +Inf, giving `0 * Inf == NaN`) would still print BIT-EXACT. +Inf vs -Inf lands here too,
        // via the NaN `rel`.
        if x.is_nan() != y.is_nan() {
            return (f64::INFINITY, i);
        }
        let (x, y) = (x as f64, y as f64);
        let rel = (x - y).abs() / x.abs().max(y.abs()).max(1e-6);
        if rel.is_nan() {
            return (f64::INFINITY, i);
        }
        if rel > worst {
            worst = rel;
            at = i;
        }
    }
    (worst, at)
}

/// Flags for the relaxed-FP **C(fast)** peer: the *same* C kernel source compiled a second time with
/// `-ffast-math`, letting gcc reassociate (and thus vectorize) float reductions the way Wukong's
/// recognized kernels do. The plain C column keeps the honest default flags (see the fairness notes
/// in BENCHMARKS.md — withholding `-ffast-math` inflates the reduction-bearing rows); this column is
/// the reassociation-normalized comparison, printed alongside, never replacing, the plain-C ratio.
///
/// NOTE (checked 2026-08-06, because the inconsistency looks like a defect): 23 of the honest-flags
/// C call sites in this file spell `["-O3", "-march=native", "-shared"]` and 16 spell
/// `["-O3", "-march=native", "-ffp-contract=fast", "-shared"]`, and **the two are equivalent**. gcc
/// invoked without a `-std=` flag is in GNU mode, whose `-ffp-contract` default already is `fast`.
/// Verified on `out[i] = x[i] + y[i]*2.5f` in its own TU: `-O3 -march=native` emits `vfmadd`, adding
/// `-ffp-contract=fast` emits `vfmadd`, `-ffp-contract=off` emits none. No family is silently denied
/// the FMA that Wukong fuses.
const C_FAST_FLAGS: &[&str] = &["-O3", "-march=native", "-ffast-math", "-shared"];

/// LOOSE cross-check for the relaxed-FP peers (C(fast) / C(omp)). `-ffast-math` and OpenMP-partitioned
/// reductions legitimately reassociate, so their results differ from Wukong's beyond the tight 1e-3
/// bar of the honest-flags columns — the bar here is a magnitude-normalized 1e-2. Returns `false`
/// (and prints the mismatch) when the peer's output is unusable; the caller then drops the column
/// rather than failing the bench.
fn relaxed_peer_ok(label: &str, peer: &str, wuk: &Measure, p: &Measure) -> bool {
    let maxabs = wuk
        .out
        .iter()
        .fold(0.0f32, |a, &v| a.max(v.abs()))
        .max(1e-6);
    let maxerr = wuk
        .out
        .iter()
        .zip(&p.out)
        .fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
    let rel = (maxerr / maxabs) as f64;
    if rel > 1e-2 {
        println!(
            "  ! {label}: {peer} disagrees with Wukong (max|Δ|/max = {rel:.2e} > 1e-2) — {peer} column skipped"
        );
        return false;
    }
    true
}

/// The C++ compiler (`$CXX`, default `g++`), resolved once for the whole process.
///
/// Held in a `OnceLock` rather than threaded through every `bench_*` signature deliberately: the C++
/// column has to exist on **every** family (see [`build_c_cpp`]), and a 40-signature change is
/// exactly the kind of churn that ends with the column quietly missing from the two benches whose
/// call sites nobody updated.
fn cxx() -> &'static str {
    static CXX: OnceLock<String> = OnceLock::new();
    CXX.get_or_init(|| std::env::var("CXX").unwrap_or_else(|_| "g++".to_string()))
}

/// Validate a C++ column against the C column it was rendered from, and drop it if they disagree.
///
/// [`cpp_from_c`] hands g++ the *same* numeric body gcc compiled at the *same* flags, so the honest
/// bar is **bit equality**; anything else is a real toolchain difference and is printed rather than
/// tolerated silently. A difference within the suite's tight like-for-like float bar (`1e-3`
/// magnitude-normalized) keeps the column with a note — the timing is still a valid measurement of
/// the C++ toolchain on that kernel. Beyond that the column is dropped, because a peer computing
/// something else must never be reported beside a ratio (the rule [`relaxed_peer_ok`] and the VML
/// column already follow). Comparison is by RAW BITS, so a NaN-producing kernel cannot compare
/// "equal-by-accident" through `PartialEq`, and `±0` differences are seen.
fn cpp_checked(label: &str, c: &Option<Measure>, cpp: Option<Measure>) -> Option<Measure> {
    let p = cpp?;
    let Some(cm) = c else { return Some(p) };
    let bitwise_same = cm.out.len() == p.out.len()
        && cm
            .out
            .iter()
            .zip(&p.out)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if bitwise_same {
        return Some(p);
    }
    let (rel, at) = max_rel_err(&cm.out, &p.out);
    if rel > 1e-3 || cm.out.len() != p.out.len() {
        println!(
            "  ! {label}: C++ (g++) disagrees with C (gcc) at [{at}] (rel {rel:.2e}) — C++ column dropped"
        );
        return None;
    }
    println!("  . {label}: C++ (g++) is not bit-identical to C (gcc) at [{at}] (rel {rel:.2e})");
    Some(p)
}

/// Compile both C and C++ from the identical numeric body and flags without timing either.
/// Keeping the pair together prevents a family from publishing a C-only comparison by accident.
/// Both columns join the family's shared rotation; the caller applies cpp_checked (or its
/// typed equivalent) after timing, using the same numerical gate as before the split.
fn build_c_cpp(
    label: &str,
    c_src: &str,
    dir: &Path,
    cc: &str,
    args: &[&str],
) -> (Option<PeerLib>, Option<PeerLib>) {
    let cl = build_peer("c", c_src, dir, label, cc, args);
    let pl = build_peer("cpp", &cpp_from_c(c_src), dir, label, cxx(), args);
    (cl, pl)
}

/// Integer outputs must agree exactly before a C++ column is reported.
fn cpp_checked_i8(label: &str, c: &Option<MeasureI8>, cpp: Option<MeasureI8>) -> Option<MeasureI8> {
    cpp.and_then(|p| match c {
        Some(cm) if cm.out != p.out => {
            println!("  ! {label}: C++ (g++) disagrees with C (gcc) on the i32 output — C++ column dropped");
            None
        }
        _ => Some(p),
    })
}

fn cpp_checked_bf16(
    label: &str,
    c: &Option<MeasureBf16>,
    cpp: Option<MeasureBf16>,
) -> Option<MeasureBf16> {
    cpp.and_then(|p| match c {
        Some(cm) => {
            let rel = ((cm.out - p.out).abs() / cm.out.abs().max(1e-6)) as f64;
            if rel > 1e-3 {
                println!("  ! {label}: C++ (g++) disagrees with C (gcc) (rel {rel:.2e}) — C++ column dropped");
                None
            } else {
                Some(p)
            }
        }
        None => Some(p),
    })
}

/// Preserve the existing 1e-3 C/C++ check over widened half-output snapshots.
fn cpp_checked_halfout(
    label: &str,
    c: &Option<MeasureHalfOut>,
    cpp: Option<MeasureHalfOut>,
) -> Option<MeasureHalfOut> {
    cpp.and_then(|p| match c {
        Some(cm) => {
            let (rel, at) = max_rel_err(&cm.out, &p.out);
            if rel > 1e-3 {
                println!("  ! {label}: C++ (g++) disagrees with C (gcc) at [{at}] (rel {rel:.2e}) — C++ column dropped");
                None
            } else {
                Some(p)
            }
        }
        None => Some(p),
    })
}

/// Preserve the existing value-equality gate over indices carried in f32 storage.
fn fast_indices_checked(label: &str, m: &Measure, cf: Measure) -> Option<Measure> {
    if m.out != cf.out {
        println!(
            "  ! {label}: C(fast) picks different indices under -ffast-math — C(fast) column skipped"
        );
        return None;
    }
    Some(cf)
}

fn bf16_fast_checked(label: &str, m: &MeasureBf16, cf: MeasureBf16) -> Option<MeasureBf16> {
    let rel = ((m.out - cf.out).abs() / cf.out.abs().max(1e-6)) as f64;
    if rel > 1e-2 {
        println!(
            "  ! {label}: C(fast) disagrees with Wukong (rel {rel:.2e} > 1e-2) — C(fast) column skipped"
        );
        return None;
    }
    Some(cf)
}

/// [`report_relaxed_ratio`] for the scalar-output bf16 [`MeasureBf16`] harness.
fn bf16_relaxed_ratio(
    peer: &str,
    wuk: &Option<MeasureBf16>,
    wk_par: &Option<MeasureBf16>,
    p: &Option<MeasureBf16>,
) {
    let Some(pm) = p else { return };
    if let Some(m) = wuk {
        let r = pm.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong single-core is {:.2}x {} than {peer}",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let Some(mp) = wk_par {
        let r = pm.ns_per_call / mp.ns_per_call;
        println!(
            "  -> Wukong @parallel is {:.2}x {} than {peer}",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
}

/// Print a `Wukong @parallel` standing **direction-aware**. `r` is `peer_ns / wukong_ns`, so `r < 1`
/// means Wukong is SLOWER. Twenty of these call sites printed the bare ratio — `"@parallel is 0.73x
/// idiomatic single-threaded C"` — which reads as a win at a glance and is in fact a 27% loss. Every
/// other summary in this file ([`report_relaxed_ratio`], [`report_ratio`], the geomeans) already
/// picks the word; these per-bench lines were the exception, and the exception only became visible
/// once the corrected peers started beating Wukong on some rows.
fn par_standing(peer: &str, r: f64) {
    println!(
        "  -> Wukong @parallel is {:.2}x {} than {peer}",
        if r >= 1.0 { r } else { 1.0 / r },
        if r >= 1.0 { "faster" } else { "slower" }
    );
}

/// Print Wukong's standing vs a relaxed-FP peer (C(fast) / C(omp)) — reported ALONGSIDE the
/// honest-flags C ratio above it, never replacing it.
fn report_relaxed_ratio(
    peer: &str,
    wuk: &Option<Measure>,
    wk_par: &Option<Measure>,
    p: &Option<Measure>,
) {
    let Some(pm) = p else { return };
    if let Some(m) = wuk {
        let r = pm.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong single-core is {:.2}x {} than {peer}",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let Some(mp) = wk_par {
        let r = pm.ns_per_call / mp.ns_per_call;
        println!(
            "  -> Wukong @parallel is {:.2}x {} than {peer}",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
}

/// Print Wukong's standing vs the **C++ (g++)** column produced by [`build_c_cpp`]. Same shape as
/// [`report_relaxed_ratio`] — which is a general peer-standing printer despite its name — but with
/// its own entry point so the C++ label is spelled one way across ~40 benches, and so a reader can
/// tell an honest-flags column apart from a relaxed-FP one at a glance.
fn report_cpp_ratio(wuk: &Option<Measure>, wk_par: &Option<Measure>, p: &Option<Measure>) {
    report_relaxed_ratio("idiomatic C++ (g++)", wuk, wk_par, p);
}

/// The elementwise-table rows whose C baseline is an IEEE-serial float reduction — exactly the rows
/// the "`-ffast-math` is withheld" disclosure applies to. These get the additional C(fast) column
/// (and, on the `@parallel` rows, [`C_OMP_FAST_FLAGS`] instead of [`C_OMP_FLAGS`]).
///
/// `argmax` and `argmax@parallel` share byte-identical C source, so both are listed: gcc can
/// vectorize a float max-with-index reduction only under relaxed FP (the comparison must be
/// reassociable), so withholding `-ffast-math` from the serial row alone would publish its
/// un-normalized ratio while its parallel twin published a normalized one. Every name here must
/// exist in [`kernels()`] — pinned by the `peer_selector_names_exist_in_the_catalogue` test (and the
/// byte-identical-source rule by `identical_c_sources_get_identical_peer_columns`).
fn is_reduction_kernel(name: &str) -> bool {
    matches!(
        name,
        "dot"
            | "ssd"
            | "argmax"
            | "dot@parallel"
            | "ssd@parallel"
            | "max@parallel"
            | "absmax@parallel"
            | "argmax@parallel"
    )
}

/// Flags for the multithreaded **C(omp)** peer: the same C kernel under `#pragma omp parallel for`,
/// compiled `-fopenmp` — the honest multicore C baseline for the `@parallel` rows (which otherwise
/// compare Wukong-multicore against single-threaded C, a disclosed but one-sided basis). Gated by
/// the [`omp_threads`] runtime probe.
const C_OMP_FLAGS: &[&str] = &["-O3", "-march=native", "-fopenmp", "-shared"];
/// C(omp) flags for the *reduction* rows: an OpenMP `reduction` clause already reassociates across
/// threads, and `-ffast-math` additionally vectorizes each thread's partial — the strongest honest
/// C peer for a parallel reduction. The label stays C(omp).
const C_OMP_FAST_FLAGS: &[&str] = &["-O3", "-march=native", "-fopenmp", "-ffast-math", "-shared"];

/// One-time empirical probe: does this gcc accept `-fopenmp`, does the resulting shared library
/// *load* (its `libgomp-1.dll` dependency must resolve from PATH — a known Windows failure mode),
/// and does a parallel region actually run multi-threaded? Returns the thread count seen inside a
/// parallel region (`None` ⇒ the C(omp) columns are skipped, with a one-time note). Verified rather
/// than assumed, per the fairness-audit requirement.
fn omp_threads(cc: &str, dir: &Path) -> Option<i32> {
    static PROBE: OnceLock<Option<i32>> = OnceLock::new();
    *PROBE.get_or_init(|| {
        let src = "#include <omp.h>\n\
                   __declspec(dllexport) int kprobe(void) {\n\
                   \x20 int n = 0;\n\
                   #pragma omp parallel\n\
                   \x20 {\n\
                   #pragma omp master\n\
                   \x20   n = omp_get_num_threads();\n\
                   \x20 }\n\
                   \x20 return n;\n}\n";
        let run = || -> Option<i32> {
            let src_path = dir.join("omp_probe.c");
            let dll = dir.join("omp_probe.dll");
            std::fs::write(&src_path, src).ok()?;
            let status = Command::new(cc)
                .args(["-O2", "-fopenmp", "-shared"])
                .arg("-o")
                .arg(&dll)
                .arg(&src_path)
                .status()
                .ok()?;
            if !status.success() {
                return None;
            }
            unsafe {
                let lib = libloading::Library::new(&dll).ok()?;
                let threads = {
                    let f: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                        lib.get(b"kprobe\0").ok()?;
                    f()
                };
                // LEAK THIS HANDLE DELIBERATELY. Dropping it runs `FreeLibrary` microseconds
                // after `GOMP_parallel` returns, which unmaps `libgomp-1.dll` and
                // `libwinpthread-1.dll` while the region's ~21 workers are still walking their
                // parking path (`gomp_finish_task`, the dock barrier mutex, then `sem_wait`).
                // A worker that has not yet reached the kernel wait fetches an instruction from
                // an unmapped page: an unhandled exception on a non-main thread, so the process
                // dies with `0xC0000005` and an EMPTY stderr, at whatever point the benchmark had
                // reached. That is the fault that killed the whole-suite baseline before its first
                // parallel table and `baseline-recovery-12-transpose` before its first table,
                // twice in ~40 processes and never on demand.
                //
                // Reproduced deterministically outside this tree: LoadLibrary -> kprobe() ->
                // FreeLibrary in a loop faults with 0xC0000005 inside 200 iterations, 3 runs of 3;
                // holding 50 ms before the free survives 300 iterations, and never freeing
                // survives 2000. The probe is memoized behind `PROBE: OnceLock`, so it loads
                // exactly once per process and the leak can never collide with a rebuild.
                //
                // `bench_mm_mkl`'s loader already leaks its handle for the same reason. The
                // `-fopenmp` PEER libraries in `time_peer` take the same risk and have survived 74
                // unloads, because they copy a 4-16 MiB output buffer between the last region and
                // the drop, which is a head start rather than a guarantee; see
                // docs/sota-campaign-2026-09/FINDING-02-omp-probe-unload-race.md.
                std::mem::forget(lib);
                Some(threads)
            }
        };
        match run() {
            Some(n) if n >= 2 => {
                println!("[C(omp) peer enabled: {cc} -fopenmp runs {n} threads in a parallel region]");
                Some(n)
            }
            Some(n) => {
                println!("[C(omp) peer disabled: OpenMP parallel region ran {n} thread(s) — columns skipped]");
                None
            }
            None => {
                println!("[C(omp) peer disabled: -fopenmp compile/load probe failed (libgomp?) — columns skipped]");
                None
            }
        }
    })
}

/// The OpenMP twin of the idiomatic C body for an elementwise-table `@parallel` row: the SAME
/// kernel with `#pragma omp parallel for` (+ a `reduction` clause where the loop is a reduction;
/// argmax uses per-thread partials + an order-independent lowest-index combine, matching the serial
/// first-max semantics). Built by name so the 3-way `Kernel` table stays untouched.
fn c_omp_source(name: &str) -> Option<String> {
    let body = match name {
        "saxpy@parallel" => {
            "float a=2.0f;\n#pragma omp parallel for\n  for(long i=0;i<N;i++) out[i]=a*x[i]+y[i];"
        }
        "poly@parallel" => {
            "#pragma omp parallel for\n  for(long i=0;i<N;i++){ float v=x[i]; float r=0.00001f; \
             r=r*v+0.0001f; r=r*v+0.001f; r=r*v+0.01f; r=r*v+0.1f; out[i]=r; }"
        }
        "relu6@parallel" => {
            "#pragma omp parallel for\n  for(long i=0;i<N;i++){ float v=x[i]; v = v<6.0f?v:6.0f; out[i] = v>0.0f?v:0.0f; }"
        }
        "gelu@parallel" => {
            "#pragma omp parallel for\n  for(long i=0;i<N;i++){ float v=x[i]; \
             float u=0.7978845608f*(v+0.044715f*v*v*v); \
             float t=1.0f-2.0f/(expf(2.0f*u)+1.0f); \
             out[i]=0.5f*v*(1.0f+t); }"
        }
        "dot@parallel" => {
            "float s=0.0f;\n#pragma omp parallel for reduction(+:s)\n  for(long i=0;i<N;i++) s+=x[i]*y[i];\n  out[0]=s;"
        }
        "ssd@parallel" => {
            "float s=0.0f;\n#pragma omp parallel for reduction(+:s)\n  for(long i=0;i<N;i++){ float d=x[i]-y[i]; s+=d*d; }\n  out[0]=s;"
        }
        "max@parallel" => {
            "float m=x[0];\n#pragma omp parallel for reduction(max:m)\n  for(long i=0;i<N;i++){ float v=x[i]; m = v>m? v:m; }\n  out[0]=m;"
        }
        "absmax@parallel" => {
            "float m=0.0f;\n#pragma omp parallel for reduction(max:m)\n  for(long i=0;i<N;i++){ float a=fabsf(x[i]); m = a>m? a:m; }\n  out[0]=m;"
        }
        "argmax@parallel" => {
            "float bv=x[0]; long bi=0;\n\
             #pragma omp parallel\n\
             \x20 {\n\
             \x20   float lbv=x[0]; long lbi=0;\n\
             #pragma omp for nowait\n\
             \x20   for(long k=0;k<N;k++){ if(x[k]>lbv){ lbv=x[k]; lbi=k; } }\n\
             #pragma omp critical\n\
             \x20   { if(lbv>bv || (lbv==bv && lbi<bi)){ bv=lbv; bi=lbi; } }\n\
             \x20 }\n\
             \x20 out[0]=(float)bi;"
        }
        _ => return None,
    };
    Some(c_kernel(body))
}

/// First line of `<tool> --version` for each peer compiler, on one line.
///
/// Best effort: a compiler that is absent or does not answer prints `?`, exactly as its column
/// later prints `n/a`. This never fails the run.
fn toolchain_banner(cc: &str) -> String {
    let first_line = |tool: &str| -> String {
        match Command::new(tool).arg("--version").output() {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("?")
                .trim()
                .to_string(),
            _ => "?".to_string(),
        }
    };
    format!(
        "peer toolchains: CC={} [{}] | CXX={} [{}] | rustc [{}]",
        cc,
        first_line(cc),
        cxx(),
        first_line(cxx()),
        first_line("rustc")
    )
}

fn main() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    // `$CXX` (default g++) is resolved once by `cxx()`, which every C++ peer column goes through.
    let cxx = cxx();
    let dir = std::env::temp_dir().join("wukong_xbench");
    let _ = std::fs::create_dir_all(&dir);
    // Publish it so `announce_dispatch` can write each generated `.wk` beside the peer sources.
    let _ = XBENCH_DIR.set(dir.clone());

    println!(
        "Cross-language kernel benchmark — Wukong (native) vs C (gcc -O3) vs Rust (rustc -Copt-level=3)"
    );
    println!(
        "N = {N} f32 elements, single-threaded except the `@parallel` rows (labelled), \
         -march=native. Lower ns is better."
    );
    // Power state up front: sustained-load timings on battery are not comparable to AC runs.
    println!("{}", model::power_status_line());
    // Toolchain identity, so a stdout log archived months later can name the compilers it timed.
    // The measured-run manifest records the two executables' SHA256 and (since 2026-09-13) the git
    // revision, but nothing recorded WHICH gcc/g++/rustc produced the peer columns - and Finding 01
    // turned entirely on re-reading a log header long after the run.
    println!("{}", toolchain_banner(&cc));
    println!("scratch: {}", dir.display());
    println!();

    // Pin the timing thread for the WHOLE run, not only the general suite. Until 2026-08-22 the
    // ~45-family battery below ran unpinned on this hybrid P/E machine — the exact configuration
    // the CPU PINNING block (see `pin_for_section`) measured publishing a byte-identical binary as
    // 1.31x faster than itself. Uniform across every column; `XBENCH_PIN=off` is the kill switch;
    // `@parallel`/C(omp) workers are unaffected (affinity is not inherited by spawned threads).
    let _pin = pin_for_section();
    println!();

    // Optional substring filter (first CLI arg): run only the kernels/sections whose name contains it,
    // for fast single-kernel iteration. No arg → the full suite, byte-for-byte as before.
    let filter = std::env::args().nth(1);
    let want = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    // `compare` and `sota-check` are MODES, not substring filters, dispatched before the battery
    // so their runs contain nothing else. `compare` is the timed `tests/compare/` corpus
    // (compare.rs); `sota-check` is part 11 of the verification gate (sota.rs) and is the one
    // subcommand here whose EXIT CODE is the product.
    if filter.as_deref() == Some("compare") {
        let rows = compare::bench_compare(&cc, &dir);
        print_sota_scorecard(&rows);
        return;
    }
    if filter.as_deref() == Some("sota-check") {
        std::process::exit(sota::sota_check(&cc, cxx, &dir));
    }

    let kernels = kernels();
    // Two separate accumulators, because the two groups of rows are not the same comparison. The
    // plain rows are single-threaded Wukong vs single-threaded C; the `@parallel` rows are
    // all-core Wukong vs the SAME single-threaded C. Pooling them into one geomean would let nine
    // multicore-vs-1-thread ratios inflate a headline printed under a "single-threaded" header —
    // the asymmetry is disclosed per row, so it must not be erased at the point the summary number
    // is printed. (The apples-to-apples multicore number is the C(omp) column on those rows.)
    let mut runtime_ratios_c = Vec::new();
    let mut compile_ratios_c = Vec::new();
    let mut runtime_ratios_cpp = Vec::new();
    let mut par_ratios_c = Vec::new();
    let mut par_ratios_cpp = Vec::new();
    // The recognition-off accumulators (master plan §7.2): the same split, so the wk(gen) headline
    // can never pool a multicore row into a single-threaded geomean either.
    let mut gen_ratios_c = Vec::new();
    let mut gen_par_ratios_c = Vec::new();

    for k in &kernels {
        if !want(k.name) {
            continue;
        }
        // Shared buffers, filled once; kernels read x,y and write out. `y` doubles as the scratch
        // array for the `fused_linear_relu` row (see [`Kernel`]), whose C/Rust peers cast the const
        // away and write it — so `yp` is derived with `as_mut_ptr`, giving it write provenance. A
        // pointer from `as_ptr()` is read-only: writing through it is UB, and the `Vec` would be
        // entitled to be treated as unmodified across the FFI call. Rebuilt per kernel, so no row
        // can contaminate another.
        let x: Vec<f32> = (0..N).map(|i| (i as f32 % 17.0) * 0.5 + 1.0).collect();
        let mut y: Vec<f32> = (0..N).map(|i| (i as f32 % 13.0) * 0.25 - 0.5).collect();
        let mut out: Vec<f32> = vec![0.0; N];
        let (xp, yp): (*const f32, *const f32) = (x.as_ptr(), y.as_mut_ptr());

        // The recognizer-on and recognition-off columns (master plan §7.2) are the SAME source
        // lowered twice: what the pattern matcher dispatches, and what the compiler compiles. Both
        // are built before anything is timed and both sit in the one position-balanced rotation with
        // every peer - see [`bench_table_kernel`].
        let TableRow {
            wukong,
            wukgen,
            wkgen_census,
            wk_floor,
            c,
            cpp,
            rust,
            cfast,
            comp,
            vml,
        } = bench_table_kernel(k, &cc, &dir, &mut out, xp, yp);

        println!("=== {} ({}) ===", k.name, k.note);
        report(k, &wukong, &c, &cpp, &rust, &cfast, &comp);
        // oneMKL VML peer for the transcendentals MKL ships a vector op for: the elementwise analogue
        // of the GEMM-vs-cblas comparison. Wukong's hand-AVX2 vmath vs Intel's hand-tuned VML, both
        // single-thread, same buffer. Cross-checked against Wukong's output (a large rel error would
        // expose a VML ABI mismatch — then the column is dishonest, so it is dropped with a note).
        if let (Some(v), Some(m)) = (vml, &wukong) {
            let (rel, at) = max_rel_err(&m.out, &v.out);
            if rel > 1e-3 {
                println!(
                    "  -> oneMKL VML disagrees with Wukong at [{at}] (rel {rel:.1e}) — likely an ABI \
                     mismatch on this MKL build; VML column dropped"
                );
            } else {
                let r = v.ns_per_call / m.ns_per_call; // >1 ⇒ Wukong faster
                println!(
                    "  -> Wukong vmath is {:.2}x {} than oneMKL VML (hand-tuned vector math), 1 thread",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" },
                );
                // NOT ACCURACY-MATCHED, and the row must say so. `vs_exp`/`vs_ln`/`vs_tanh` run in
                // VML's DEFAULT mode, which is VML_HA at ~0.5 ULP, while Wukong ships ≤2 ULP exp,
                // ≤12 ULP log and ≤1 ULP tanh. A faster-but-less-accurate kernel is not
                // straightforwardly faster. `wukong_runtime`'s own `vmath_exp_log_vs_vml` probe
                // re-runs the peer through `vmlSetMode(VML_LA)` — measured at 2 ULP exp / 4 ULP log,
                // where Wukong measures 2 and 4 — and at that matched accuracy exp and log come out
                // a TIE (2026-08-06). This row has NOT been re-measured that way, and it also times
                // each arm in its own tight rep-block, so it is a different cache regime from the
                // in-crate matched-accuracy probe. Treat it as an upper bound, not a standing.
                println!(
                    "     (VML here runs in its DEFAULT ~0.5-ULP HA mode — NOT accuracy-matched \
                     against Wukong's ≤2 ULP exp / ≤12 ULP log / ≤1 ULP tanh. At matched accuracy \
                     (VML_LA) the in-crate probe reads exp and log as a TIE. Upper bound, not a \
                     standing.)"
                );
            }
        }
        let is_par = k.name.contains("@parallel");
        // The recognition-off row: ratio vs C (the §7.2 column) and vs the recognizer-on lowering
        // of the same source (the dispatch win, measured rather than asserted), plus the census
        // proving no `wukong_*` kernel survived. A divergence between the two lowerings of ONE
        // source is a compiler defect, not a timing: print it loudly and withhold the ratio.
        if let (Some(g), Some(census)) = (&wukgen, &wkgen_census) {
            let checked = match &wukong {
                Some(m) => {
                    let (rel, at) = max_rel_err(&m.out, &g.out);
                    // Reduction rows get the relaxed bar: the recognized kernel legitimately
                    // reassociates the f32 sum the as-written serial loop performs.
                    let bar = if is_reduction_kernel(k.name) {
                        1e-2
                    } else {
                        1e-3
                    };
                    if rel > bar {
                        println!(
                            "  !!! wk(gen) DIVERGES from the recognizer-on lowering at [{at}] \
                             (rel {rel:.2e}) — recognition-off column WITHHELD; a divergence \
                             between two lowerings of one source is a compiler defect"
                        );
                        false
                    } else {
                        true
                    }
                }
                None => true,
            };
            if checked {
                let dir = |r: f64| -> (f64, &'static str) {
                    if r >= 1.0 {
                        (r, "faster")
                    } else {
                        (1.0 / r, "slower")
                    }
                };
                let mut line = format!("  wk(gen) recognition-off: {:.0} ns", g.ns_per_call);
                // Every ratio on this line is checked against the floor the row measured for
                // itself: the ABBA timed each build TWICE, so the spread of one build's two samples
                // is a twin control for this column, in this run. An effect inside it is not an
                // effect (§4.1), and saying so here is the difference between a number and a claim.
                let inside = |v: f64| wk_floor.is_some_and(|f| v < f);
                if let Some(p) = &c {
                    let r = p.ns_per_call / g.ns_per_call;
                    let (v, w) = dir(r);
                    line.push_str(&format!(" — {v:.2}x {w} than C"));
                    if inside(v) {
                        line.push_str(" [INSIDE FLOOR]");
                    }
                    if is_par {
                        gen_par_ratios_c.push(r);
                    } else {
                        gen_ratios_c.push(r);
                    }
                }
                if let Some(m) = &wukong {
                    let (v, w) = dir(m.ns_per_call / g.ns_per_call);
                    line.push_str(&format!("; {v:.2}x {w} than recognizer-on Wukong"));
                    if inside(v) {
                        line.push_str(" [INSIDE FLOOR]");
                    }
                }
                if let Some(f) = wk_floor {
                    line.push_str(&format!("; row floor {f:.2}x"));
                }
                println!("{line}");
                println!("     wk(gen) dispatch: {census}");
            }
        }
        if let (Some(m), Some(c)) = (&wukong, &c) {
            let r = c.ns_per_call / m.ns_per_call; // >1 => Wukong faster
            if is_par {
                par_ratios_c.push(r);
            } else {
                runtime_ratios_c.push(r);
                // Compile time is threading-independent, so it pools over every row.
            }
            compile_ratios_c.push(c.compile.as_secs_f64() / m.compile.as_secs_f64());
        }
        if let (Some(m), Some(cpp)) = (&wukong, &cpp) {
            let r = cpp.ns_per_call / m.ns_per_call; // >1 => Wukong faster
            if is_par {
                par_ratios_cpp.push(r);
            } else {
                runtime_ratios_cpp.push(r);
            }
        }
        println!();
    }

    // Direction-aware ratio + word: a geomean below 1.0 is a LOSS and must print as one.
    let standing = |g: f64| {
        (
            if g >= 1.0 { g } else { 1.0 / g },
            if g >= 1.0 { "faster" } else { "slower" },
        )
    };
    if !runtime_ratios_c.is_empty() || !compile_ratios_c.is_empty() {
        println!("Summary vs C (geomean over kernels):");
        if !runtime_ratios_c.is_empty() {
            let (g, w) = standing(geomean(&runtime_ratios_c));
            println!(
                "  runtime:  Wukong is {g:.2}x {w} than C  [single-threaded rows: {}]",
                runtime_ratios_c.len()
            );
        }
        if !par_ratios_c.is_empty() {
            let (g, w) = standing(geomean(&par_ratios_c));
            println!(
                "  runtime:  Wukong is {g:.2}x {w} than C  [@parallel rows: {} — all-core Wukong \
                 vs 1-thread C; see each row's C(omp) column for multicore-vs-multicore]",
                par_ratios_c.len()
            );
        }
        if !compile_ratios_c.is_empty() {
            let (g, w) = standing(geomean(&compile_ratios_c));
            println!("  compile:  Wukong is {g:.1}x {w} to compile than C");
        }
    }
    // The master plan §7.2 headline: the same rows with every recognizer OFF — general code vs gcc,
    // not the recognizer dialect vs gcc. Printed as its own summary, never pooled with the above.
    if !gen_ratios_c.is_empty() || !gen_par_ratios_c.is_empty() {
        println!("Summary vs C, RECOGNITION-OFF (geomean over kernels) — the language without the dispatcher:");
        if !gen_ratios_c.is_empty() {
            let (g, w) = standing(geomean(&gen_ratios_c));
            println!(
                "  runtime:  Wukong(gen) is {g:.2}x {w} than C  [single-threaded rows: {}]",
                gen_ratios_c.len()
            );
        }
        if !gen_par_ratios_c.is_empty() {
            let (g, w) = standing(geomean(&gen_par_ratios_c));
            println!(
                "  runtime:  Wukong(gen) is {g:.2}x {w} than C  [@parallel rows: {} — all-core \
                 Wukong vs 1-thread C; see each row's C(omp) column]",
                gen_par_ratios_c.len()
            );
        }
    }
    if !runtime_ratios_cpp.is_empty() || !par_ratios_cpp.is_empty() {
        println!("Summary vs C++ (g++, geomean over kernels):");
        if !runtime_ratios_cpp.is_empty() {
            let (g, w) = standing(geomean(&runtime_ratios_cpp));
            println!(
                "  runtime:  Wukong is {g:.2}x {w} than C++  [single-threaded rows: {}]",
                runtime_ratios_cpp.len()
            );
        }
        if !par_ratios_cpp.is_empty() {
            let (g, w) = standing(geomean(&par_ratios_cpp));
            println!(
                "  runtime:  Wukong is {g:.2}x {w} than C++  [@parallel rows: {} — all-core \
                 Wukong vs 1-thread C++]",
                par_ratios_cpp.len()
            );
        }
    }

    println!();
    let roof = measure_fma_roofline();
    if roof > 0.0 {
        println!(
            "AVX2-FMA roofline (this run, single core): {roof:.0} GFLOP/s — GEMM is reported as % \
             of THIS, which is clock-invariant (absolute GFLOP/s swings with the laptop's power state)."
        );
        println!();
    }
    match mkl() {
        Some(api) => println!(
            "oneMKL peer: {} (max {} threads) — Intel's hand-tuned CPU GEMM, the Tier-B library bar.\n",
            api.path.display(),
            api.max_threads
        ),
        None => println!(
            "oneMKL peer: not found — set WUKONG_MKL_DLL=<path to mkl_rt.dll> to enable the MKL column.\n"
        ),
    }
    if want("matmul") {
        bench_matmul(&cc, &dir, roof);
    }
    if want("matmul_skinny") {
        bench_matmul_skinny(&cc, &dir, roof);
    }
    if want("linear") {
        bench_linear(&cc, &dir, roof);
    }
    if want("ffn") {
        bench_ffn(&cc, &dir, roof);
    }
    if want("linear_bf16") {
        bench_linear_bf16(&cc, &dir);
    }
    if want("matmul_tn") {
        bench_matmul_tn(&cc, &dir, roof);
    }
    if want("gemv") {
        bench_gemv(&cc, &dir);
    }
    if want("scaled_gemm") {
        bench_scaled_gemm(&cc, &dir);
    }
    if want("conv") {
        bench_conv(&cc, &dir);
    }
    if want("norm") {
        bench_norm(&cc, &dir);
        bench_norm_batched(&cc, &dir);
    }
    if want("i8gemm") {
        bench_i8gemm(&cc, &dir);
    }
    if want("dequant") {
        bench_dequant(&cc, &dir);
    }
    if want("bf16") {
        bench_bf16(&cc, &dir);
    }
    if want("streaming") {
        bench_streaming_large(&cc, &dir);
    }
    if want("transpose") {
        bench_transpose(&cc, &dir);
    }
    if want("colsum") {
        bench_colsum(&cc, &dir);
    }
    if want("biasadd") {
        bench_biasadd(&cc, &dir);
    }
    if want("axpby_half") {
        bench_axpby_half_out(&cc, &dir);
    }
    if want("colmax") {
        bench_colmax(&cc, &dir);
    }
    if want("colstat") {
        bench_colstat(&cc, &dir);
    }
    if want("softmax_bwd") {
        bench_softmax_bwd(&cc, &dir);
    }
    if want("rmsnorm_bwd") {
        bench_rmsnorm_bwd(&cc, &dir);
    }
    if want("layernorm_bwd") {
        bench_layernorm_bwd(&cc, &dir);
    }
    if want("xent") {
        bench_xent(&cc, &dir);
    }
    if want("rope") {
        bench_rope(&cc, &dir);
    }
    if want("xent_bwd") {
        bench_xent_bwd(&cc, &dir);
    }
    if want("rope_bwd") {
        bench_rope_bwd(&cc, &dir);
    }
    if want("gate") {
        bench_gate(&cc, &dir);
    }
    if want("row_losses") {
        bench_row_losses(&cc, &dir);
    }
    if want("rowarg") {
        bench_rowarg(&cc, &dir);
    }
    if want("colarg") {
        bench_colarg(&cc, &dir);
    }
    if want("cumsum") {
        bench_cumsum(&cc, &dir);
    }
    if want("cumprod") {
        bench_cumprod(&cc, &dir);
    }
    if want("lrscan") {
        bench_lrscan(&cc, &dir);
    }
    if want("cumminmax") {
        bench_cumminmax(&cc, &dir);
    }
    if want("act_backward") {
        bench_act_backward(&cc, &dir);
    }
    if want("model") {
        model::bench_model(&cc, &dir);
    }
    // The general-code suite: Wukong written OUTSIDE the recognizer dialect. Opt-in (`general`),
    // because every other row above deliberately writes the pattern.
    if filter.as_deref() == Some("general") {
        let rows = general::bench_general(&cc, cxx, &dir);
        println!();
        print_sota_scorecard(&rows);
    }
}

/// One row of the main kernel table: every column's best time, plus the recognition-off column's
/// dispatch line and the row's own measured noise floor.
struct TableRow {
    wukong: Option<Measure>,
    /// The same `.wk` source lowered with every recognizer OFF (master plan §7.2) - `wk(gen)`.
    wukgen: Option<Measure>,
    /// `wk(gen)`'s dispatch line from [`announce_dispatch`]: the T2 witness that no recognized
    /// kernel survived in the column the language ratio is read from.
    wkgen_census: Option<String>,
    /// The widest `max/min` spread either Wukong column showed between its own two samples. The
    /// rotation times every column twice, so that spread is a free twin control for THIS row: a
    /// `wk(gen)` effect inside it is not an effect (§4.1).
    wk_floor: Option<f64>,
    c: Option<Measure>,
    cpp: Option<Measure>,
    rust: Option<Measure>,
    cfast: Option<Measure>,
    comp: Option<Measure>,
    vml: Option<Measure>,
}

/// Build all table peers, including optional VML, before rotating their fixed slots.
/// The shared scratch pointer has the same provenance as the original table runner.
///
/// Both Wukong lowerings of the row's source - recognizer-on (the dispatcher) and recognition-off
/// (`wk(gen)`, the language) - are built here with the peers and timed in the SAME
/// position-balanced rotation, so neither inherits a slot advantage over the other or over C.
fn bench_table_kernel(
    k: &Kernel,
    cc: &str,
    dir: &Path,
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
) -> TableRow {
    let wk = build_wukong(&k.wuk, 3);
    let wk_gen = build_wukong_gen(&k.wuk, 3);
    let wkgen_census = wk_gen.as_ref().map(|j| j.dispatch.clone());
    let (cl, pl) = build_c_cpp(
        k.name,
        &k.c,
        dir,
        cc,
        &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
    );
    let rl = build_peer(
        "rs",
        &k.rust,
        dir,
        k.name,
        "rustc",
        &[
            "-Copt-level=3",
            "-Ctarget-cpu=native",
            "--crate-type=cdylib",
        ],
    );
    let fl = if is_reduction_kernel(k.name) {
        build_peer(
            "c",
            &k.c,
            dir,
            &format!("{}_fast", k.name),
            cc,
            C_FAST_FLAGS,
        )
    } else {
        None
    };
    let ol = c_omp_source(k.name).and_then(|src| {
        omp_threads(cc, dir)?;
        let flags = if is_reduction_kernel(k.name) {
            C_OMP_FAST_FLAGS
        } else {
            C_OMP_FLAGS
        };
        build_peer("c", &src, dir, &format!("{}_omp", k.name), cc, flags)
    });
    let vml_fn = match k.name {
        "exp" => mkl().and_then(|a| a.vs_exp),
        "log" => mkl().and_then(|a| a.vs_ln),
        "tanh" => mkl().and_then(|a| a.vs_tanh),
        _ => None,
    };
    let columns = [
        Built::wk(wk),
        Built::wk(wk_gen),
        Built::peer(cl),
        Built::peer(pl),
        Built::peer(rl),
        Built::peer(fl),
        Built::peer(ol),
        match vml_fn {
            Some(f) => Built::Direct(Box::new(move |o| Some(bench_vml(f, xp, o)))),
            None => Built::Missing,
        },
    ];
    // Each column's first sample, then the spread its second sample makes with it.
    let mut first: Vec<Option<f64>> = vec![None; columns.len()];
    let mut spread: Vec<Option<f64>> = vec![None; columns.len()];
    let mut timed = abba_min_columns(columns.len(), |i| {
        let sample = time_built(&columns[i], out, xp, yp);
        if let Some(ns) = sample
            .as_ref()
            .map(|m| m.ns_per_call)
            .filter(|&ns| ns > 0.0)
        {
            match first[i] {
                Some(a) => spread[i] = Some((a / ns).max(ns / a)),
                None => first[i] = Some(ns),
            }
        }
        sample
    })
    .into_iter();
    let wk_floor = match (spread[0], spread[1]) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    let wukong = timed.next().flatten();
    let wukgen = timed.next().flatten();
    let c = timed.next().flatten();
    let cpp = cpp_checked(k.name, &c, timed.next().flatten());
    let rust = timed.next().flatten();
    let cfast = timed.next().flatten().filter(|p| {
        wukong
            .as_ref()
            .is_some_and(|m| relaxed_peer_ok(k.name, "C(fast)", m, p))
    });
    let comp = timed.next().flatten().filter(|p| {
        wukong
            .as_ref()
            .is_some_and(|m| relaxed_peer_ok(k.name, "C(omp)", m, p))
    });
    let vml = timed.next().flatten();
    TableRow {
        wukong,
        wukgen,
        wkgen_census,
        wk_floor,
        c,
        cpp,
        rust,
        cfast,
        comp,
        vml,
    }
}

/// Matmul is the canonical ML kernel and is compute-bound, so both SIMD and multicore pay off — the
/// regime where a tensor compiler should genuinely beat idiomatic scalar-source code. We benchmark
/// an `ikj`-ordered C = A·B (the cache-friendly idiom that auto-vectorizes well) written the same
/// way in each language, and additionally Wukong's `@parallel` form. Reported as GFLOP/s. The win
/// is shown across a size sweep so it is clearly structural, not a single-size artifact.
fn bench_matmul(cc: &str, dir: &Path, roof: f64) {
    // 2048³ already spills L3 (3×16 MB operands + packing vs ~24 MB L3) — the regime where multi-level
    // blocking and the library gap show. 4096³ is the same regime, larger; it is multi-second per call
    // so it is gated behind XBENCH_HUGE to keep the default run quick.
    let mut sizes = vec![256usize, 512, 1024, 2048];
    if std::env::var("XBENCH_HUGE").is_ok() {
        sizes.push(4096);
    }
    // Size filter for fast lever iteration (`XBENCH_MATMUL_SIZES=512,1024`): an ABBA A/B on one
    // band shouldn't pay the whole sweep (2048³ dominates a full run's wall time and heat).
    // Measurement-only — the selected sizes run the identical harness.
    if let Ok(f) = std::env::var("XBENCH_MATMUL_SIZES") {
        let want: Vec<usize> = f.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        if !want.is_empty() {
            let offered = sizes.clone();
            sizes.retain(|ns| want.contains(ns));
            // A value matching nothing (a typo, or 4096 without XBENCH_HUGE) used to leave `sizes`
            // empty and run zero benchmarks silently — indistinguishable from a filter miss. Say
            // so and fall back to the full sweep, like the sibling XBENCH_MODEL_S knob.
            if sizes.is_empty() {
                println!(
                    "  ! XBENCH_MATMUL_SIZES={f} selected none of {offered:?}; running the full sweep"
                );
                sizes = offered;
            }
        }
    }
    for ns in sizes {
        bench_matmul_size(cc, dir, ns, roof);
        println!();
    }
}

/// The skinny transformer GEMMs: tall-thin activation matrices against the GPT-2 block's weight
/// shapes in the `nn.Linear` NT layout (`C = A·Bᵀ`) — attention/output projections (K=N=768) and
/// the FFN up/down projections (768→3072, 3072→768), at S=128 and S=512 rows. The square sweeps
/// never enter this low-M regime, where the parallel grid's grain (not peak FLOPs) decides the
/// gap to MKL(all).
///
/// **The C / C++ / Rust reference columns (added 2026-08-05).** This section used to print library
/// peers ONLY — "the naive-C columns of the square section add nothing here" — which left the one
/// table that carries a library bar with no plain-language baseline at all. A reader could see how
/// far Wukong was from oneMKL and had no way to see how far *either* was from the loop a programmer
/// writes, at the shapes a real transformer actually runs. That is the same blank-cell problem as a
/// missing C++ column, one table over. They are naive `ijk` NT nests ([`c_linear_rect`]), the exact
/// spelling the square section uses, so the two tables' C columns mean the same thing.
///
/// Build every column first, then rotate all peers through forward/reverse positions.
/// Missing optional peers retain their slots.
fn bench_matmul_skinny(cc: &str, dir: &Path, roof: f64) {
    for (m, k, n) in [
        (128usize, 768usize, 768usize),
        (128, 768, 3072),
        (128, 3072, 768),
        (512, 768, 768),
        (512, 768, 3072),
        (512, 3072, 768),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut c = vec![0.0f32; m * n];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = |mm: &Option<Measure>| {
            mm.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== matmul_skinny {m}x{k} · ({n}x{k})ᵀ (nn.Linear NT; GFLOP/s, higher is better) ==="
        );
        // Build every column before the shared position-balanced rotation.
        let wk1 = build_wukong(&wk_linear_rect(m, k, n, false), 3);
        let wkp = build_wukong(&wk_linear_rect(m, k, n, true), 3);
        let (cl, pl) = build_c_cpp(
            "linear_rect",
            &c_linear_rect(m, k, n),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let fl = build_peer(
            "c",
            &c_linear_rect(m, k, n),
            dir,
            "linear_rect_fast",
            cc,
            C_FAST_FLAGS,
        );
        let rl = build_peer(
            "rs",
            &rust_linear_rect(m, k, n),
            dir,
            "linear_rect",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let (ar, br) = (&a[..], &b[..]);
        let mkl_all_threads = mkl().map(|api| api.max_threads);
        let columns = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(fl),
            Built::peer(rl),
            Built::Direct(Box::new(move |o| {
                bench_mm_mkl_rect(m, k, n, true, 1, ar, br, o)
            })),
            match mkl_all_threads {
                Some(t) => Built::Direct(Box::new(move |o| {
                    bench_mm_mkl_rect(m, k, n, true, t, ar, br, o)
                })),
                None => Built::Missing,
            },
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut c, ap, bp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("linear_rect", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("linear_rect", "C(fast)", m, p))
        });
        let rm = timed.next().flatten();
        let mkl_1c = timed.next().flatten();
        let mkl_all = timed.next().flatten();
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "",
            "Wuk(1c)",
            "Wuk(par)",
            "MKL(1c)",
            "MKL(all)",
            "C(gcc)",
            "C++(g++)",
            "C(fast)",
            "Rust"
        );
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&mkl_1c),
            gflops(&mkl_all),
            gflops(&cm),
            gflops(&cm_cpp),
            gflops(&cfast),
            gflops(&rm),
        );
        if roof > 0.0 {
            if let Some(w) = &wuk {
                println!(
                    "  -> Wukong single-core = {:.0}% of measured roofline",
                    (flops / w.ns_per_call) / roof * 100.0
                );
            }
        }
        // The bit-exactness law exercised at the exact shape: serial and @parallel must agree
        // bit-for-bit (same per-(i,j) K order regardless of the parallel grid).
        if let (Some(s), Some(p)) = (&wuk, &wk_par) {
            match s
                .out
                .iter()
                .zip(&p.out)
                .position(|(x, y)| x.to_bits() != y.to_bits())
            {
                Some(at) => println!("  ! serial vs @parallel NOT bit-exact at [{at}]"),
                None => println!("  cross-check Wukong serial vs @parallel: BIT-EXACT"),
            }
        }
        // Cross-language correctness against the naive C reference: the tiled kernel reassociates
        // the K-dot, so the bar is the suite's tight magnitude-normalized 1e-3, as in bench_matmul.
        if let (Some(mm), Some(c2)) = (&wuk, &cm) {
            let (rel, at) = max_rel_err(&mm.out, &c2.out);
            if rel > 1e-3 {
                println!(
                    "  ! full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {rel:.2e})",
                    mm.out[at], c2.out[at]
                );
            }
        }
        report_gemm_vs_mkl(&wuk, &wk_par, &mkl_1c, &mkl_all, flops);
        if let (Some(mm), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / mm.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

fn bench_matmul_size(cc: &str, dir: &Path, ns: usize, roof: f64) {
    let n2 = ns * ns;
    let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
    let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
    let mut c = vec![0.0f32; n2];
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let flops = 2.0 * (ns as f64).powi(3);

    println!("=== matmul {ns}x{ns} (C=A·B, ikj order; GFLOP/s, higher is better) ===");
    let gflops = |m: &Option<Measure>| {
        m.as_ref()
            .map(|x| format!("{:.1}", flops / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };

    // Slot balance does not establish thermal equivalence: an earlier interleaved GEMM
    // timer suffered rival Rayon/OpenMP pool scheduling and residual all-core heat. Retain
    // the campaign's idle, AC-powered measurement gate and MKL(all) withdrawal until a
    // controlled rerun establishes comparability; reversal alone cannot remove carry-over.
    // Build every column before the shared position-balanced rotation.
    let wk1 = build_wukong(&wk_matmul(ns, false), 3);
    let wkp = build_wukong(&wk_matmul(ns, true), 3);
    let run_naive = ns < 2048 || std::env::var("XBENCH_NAIVE_HUGE").is_ok();
    let (cl, pl, fl, rl) = if run_naive {
        let (cl, pl) = build_c_cpp(
            "matmul",
            &c_matmul(ns),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let fl = build_peer("c", &c_matmul(ns), dir, "matmul_fast", cc, C_FAST_FLAGS);
        let rl = build_peer(
            "rs",
            &rust_matmul(ns),
            dir,
            "matmul",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        (cl, pl, fl, rl)
    } else {
        (None, None, None, None)
    };
    let ol = if run_naive {
        omp_threads(cc, dir)
            .and_then(|_| build_peer("c", &c_matmul_omp(ns), dir, "matmul_omp", cc, C_OMP_FLAGS))
    } else {
        None
    };
    let (ar, br) = (&a[..], &b[..]);
    let mkl_all_threads = mkl().map(|api| api.max_threads);
    let columns = [
        Built::wk(wk1),
        Built::wk(wkp),
        Built::peer(cl),
        Built::peer(pl),
        Built::peer(fl),
        Built::peer(rl),
        Built::peer(ol),
        Built::Direct(Box::new(move |o| bench_mm_mkl(ns, false, 1, ar, br, o))),
        match mkl_all_threads {
            Some(t) => Built::Direct(Box::new(move |o| bench_mm_mkl(ns, false, t, ar, br, o))),
            None => Built::Missing,
        },
        Built::Direct(Box::new(move |o| {
            Some(bench_mm_tuned(ns, false, ar, br, o))
        })),
    ];
    let mut timed =
        abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut c, ap, bp)).into_iter();
    let wuk = timed.next().flatten();
    let wk_par = timed.next().flatten();
    let cm = timed.next().flatten();
    let cm_cpp = cpp_checked("matmul", &cm, timed.next().flatten());
    let cfast = timed.next().flatten().filter(|p| {
        wuk.as_ref()
            .is_some_and(|m| relaxed_peer_ok("matmul", "C(fast)", m, p))
    });
    let rm = timed.next().flatten();
    let comp = timed.next().flatten().filter(|p| {
        wuk.as_ref()
            .is_some_and(|m| relaxed_peer_ok("matmul", "C(omp)", m, p))
    });
    let mkl_1c = timed.next().flatten();
    let mkl_all = timed.next().flatten();
    let tuned = timed
        .next()
        .flatten()
        .expect("the in-process tuned GEMM column always returns a measure");
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "",
        "Wuk(1c)",
        "Wuk(par)",
        "MKL(1c)",
        "MKL(all)",
        "tuned(mm)",
        "C(gcc)",
        "C++(g++)",
        "C(fast)",
        "C(omp)",
        "Rust"
    );
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "GFLOP/s",
        gflops(&wuk),
        gflops(&wk_par),
        gflops(&mkl_1c),
        gflops(&mkl_all),
        gflops(&Some(tuned.clone())),
        gflops(&cm),
        gflops(&cm_cpp),
        gflops(&cfast),
        gflops(&comp),
        gflops(&rm)
    );
    report_gemm_standing(&wuk, &tuned, roof, flops);
    report_gemm_vs_mkl(&wuk, &wk_par, &mkl_1c, &mkl_all, flops);
    if !run_naive {
        println!(
            "  -> naive C/Rust omitted at {ns}³ (multi-second per call; their loss widens monotonically\n     below — Wuk 1c is 2.4–3.8× naive C at 256–1024³ and naive C keeps falling on cache misses).\n     Correctness here is cross-checked vs MKL (above), the stronger oracle."
        );
    }
    // Cross-language correctness: Wukong's C must match the C/gcc reference element-by-element. Uses
    // the single-core `wuk`; every timed column also retains its own output snapshot.
    if let (Some(m), Some(c)) = (&wuk, &cm) {
        let (rel, at) = max_rel_err(&m.out, &c.out);
        if rel > 1e-3 {
            println!(
                "  ! full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {:.2e})",
                m.out[at], c.out[at], rel
            );
        }
    }
    if let (Some(mp), Some(c)) = (&wk_par, &cm) {
        let r = flops / mp.ns_per_call / (flops / c.ns_per_call);
        println!(
            "  -> Wukong @parallel is {:.2}x {} than idiomatic single-threaded C",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let (Some(ms), Some(c)) = (&wuk, &cm) {
        let r = flops / ms.ns_per_call / (flops / c.ns_per_call);
        println!(
            "  -> Wukong single-core (SIMD) is {:.2}x {} than C single-threaded",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
    report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
    report_relaxed_ratio("C(omp) [-fopenmp, all cores]", &wuk, &wk_par, &comp);
}

/// `ikj`-ordered matmul, optionally `@parallel` (parallelizes the outer `i` loop across cores).
fn wk_matmul(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j0 in 0..{ns} {{ c[i * {ns} + j0] = 0.0; }}\n\
         \x20       for k in 0..{ns} {{\n\
         \x20           let aik: f32 = a[i * {ns} + k];\n\
         \x20           for j in 0..{ns} {{\n\
         \x20               c[i * {ns} + j] = c[i * {ns} + j] + aik * b[k * {ns} + j];\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_matmul(ns: usize) -> String {
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         \x20 for (long i=0;i<NS;i++){{\n\
         \x20   for (long j=0;j<NS;j++) c[i*NS+j]=0.0f;\n\
         \x20   for (long k=0;k<NS;k++){{\n\
         \x20     float aik=a[i*NS+k];\n\
         \x20     for (long j=0;j<NS;j++) c[i*NS+j]+=aik*b[k*NS+j];\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

/// The OpenMP twin of [`c_matmul`]: the same naive `ikj` nest with the outer `i` loop split across
/// cores (`#pragma omp parallel for`) — each row's arithmetic is unchanged, so it matches the
/// serial nest exactly; the parallelism only partitions rows.
fn c_matmul_omp(ns: usize) -> String {
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         #pragma omp parallel for\n\
         \x20 for (long i=0;i<NS;i++){{\n\
         \x20   for (long j=0;j<NS;j++) c[i*NS+j]=0.0f;\n\
         \x20   for (long k=0;k<NS;k++){{\n\
         \x20     float aik=a[i*NS+k];\n\
         \x20     for (long j=0;j<NS;j++) c[i*NS+j]+=aik*b[k*NS+j];\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

fn rust_matmul(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{ *c.add(i*NS+j)=0.0; }}\n\
         \x20   for k in 0..NS {{\n\
         \x20     let aik=*a.add(i*NS+k);\n\
         \x20     for j in 0..NS {{ *c.add(i*NS+j)+=aik* *b.add(k*NS+j); }}\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

/// `C = Aᵀ·B` — the **weight-gradient** GEMM of a training backward pass (`dW = dYᵀ·X`). A is stored
/// `[K, M]` (the contraction/batch axis is the OUTER index of A's storage), so its logical operand is
/// the transpose of its layout. Wukong recognizes `a[k*M+i]*b[k*N+j]` and dispatches to
/// `wukong_sgemm_tn` (transpose A once, then the tuned NN kernel); the C/Rust peers run the natural
/// `kij` nest, which hoists `a[k*M+i]` and streams B and C contiguously (see [`c_matmul_tn`] for the
/// spelling correction and its measured cost). Square M=K=N for the shared-buffer ABI.
fn bench_matmul_tn(cc: &str, dir: &Path, roof: f64) {
    for ns in [256usize, 512, 1024] {
        let n2 = ns * ns;
        let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut c = vec![0.0f32; n2];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        println!(
            "=== matmul_tn {ns}x{ns} (C=Aᵀ·B, the dW weight-gradient; GFLOP/s, higher is better) ==="
        );
        let gflops = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        // POSITION-BALANCED TIMING (2026-09-13) - see `abba_min_columns` and the note in
        // `bench_colsum`. Build every column first, then time them all in one rotation.
        let wk1 = build_wukong(&wk_matmul_tn(ns, false), 3);
        let wkp = build_wukong(&wk_matmul_tn(ns, true), 3);
        let (cl, pl) = build_c_cpp(
            "matmul_tn",
            &c_matmul_tn(ns),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        // Reassociation-normalized peer: -ffast-math may vectorize the column-strided reduction.
        let fl = build_peer(
            "c",
            &c_matmul_tn(ns),
            dir,
            "matmul_tn_fast",
            cc,
            C_FAST_FLAGS,
        );
        let rl = build_peer(
            "rs",
            &rust_matmul_tn(ns),
            dir,
            "matmul_tn",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // LIBRARY-GRADE PEER (2026-09-13). The evaluation listed the TN weight-gradient under
        // "where Wukong genuinely leads" on a 3.2-6.6x margin over idiomatic C, and this was the one
        // GEMM family with NO library bar at all - while `cblas_sgemm_64` was already resolved in
        // this process and needed only `transa = CblasTrans`.
        let (ar, br) = (&a[..], &b[..]);
        let mkl_all_threads = mkl().map(|api| api.max_threads);
        let cols_all = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(fl),
            Built::Direct(Box::new(move |o: &mut [f32]| {
                bench_mm_mkl_tn(ns, ns, ns, 1, ar, br, o)
            })),
            match mkl_all_threads {
                Some(t) => Built::Direct(Box::new(move |o: &mut [f32]| {
                    bench_mm_mkl_tn(ns, ns, ns, t, ar, br, o)
                })),
                None => Built::Missing,
            },
            Built::peer(rl),
        ];
        let mut timed =
            abba_min_columns(cols_all.len(), |i| time_built(&cols_all[i], &mut c, ap, bp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("matmul_tn", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("matmul_tn", "C(fast)", m, p))
        });
        // MKL reassociates its accumulation, so the bar is the harness's magnitude-normalized 1e-2
        // relaxed bar, the same one C(fast) and C(omp) are held to - not the 1e-3 like-for-like bar.
        let mkl_1c = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("matmul_tn", "MKL(1c)", m, p))
        });
        let mkl_all = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("matmul_tn", "MKL(all)", m, p))
        });
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "",
            "Wuk(1core)",
            "Wuk(par)",
            "C (gcc)",
            "C++ (g++)",
            "C(fast)",
            "MKL(1c)",
            "MKL(all)",
            "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
            gflops(&cm_cpp),
            gflops(&cfast),
            gflops(&mkl_1c),
            gflops(&mkl_all),
            gflops(&rm)
        );
        // Absolute GFLOP/s on this laptop swings ~3x with the power state, so the number above is
        // only comparable within a run. The clock-invariant denominator is the roofline measured by
        // the SAME process (`measure_fma_roofline`) — print the percentage next to the table, as
        // `bench_matmul_skinny`/`bench_linear` already do.
        if roof > 0.0 {
            if let Some(w) = &wuk {
                println!(
                    "  -> Wukong single-core = {:.0}% of measured roofline",
                    (flops / w.ns_per_call) / roof * 100.0
                );
            }
        }
        // Cross-language correctness: the transpose-once-then-NN result must match the `kij` nest,
        // in BOTH peers (each was re-spelled in the 2026-08-04 peer audit).
        for (lang, peer) in [("C", &cm), ("Rust", &rm)] {
            if let (Some(m), Some(c)) = (&wuk, peer) {
                let (rel, at) = max_rel_err(&m.out, &c.out);
                if rel > 1e-3 {
                    println!(
                        "  ! full-buffer mismatch vs {lang} at [{at}]: Wukong={} {lang}={} (rel {:.2e})",
                        m.out[at], c.out[at], rel
                    );
                }
            }
        }
        if let (Some(ms), Some(c)) = (&wuk, &cm) {
            let r = c.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core (SIMD) is {:.2}x {} than C single-threaded",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c)) = (&wk_par, &cm) {
            let r = c.ns_per_call / mp.ns_per_call;
            println!(
                "  -> Wukong @parallel is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// `ijk` dot-product `C = Aᵀ·B`: A indexed `a[k*NS+i]` (transposed — k is A's outer index), B
/// `b[k*NS+j]` (normal). Wukong folds this to `wukong_sgemm_tn[_parallel]`. Optionally `@parallel`.
fn wk_matmul_tn(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{ns} {{\n\
         \x20               s = s + a[k * {ns} + i] * b[k * {ns} + j];\n\
         \x20           }}\n\
         \x20           c[i * {ns} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

/// The C weight-gradient peer, in the natural **`kij`** order: hoist `a[k*M+i]` out of the inner
/// loop and stream `b[k*N+·]` and `c[i*N+·]` contiguously, exactly as [`c_matmul`] does for the
/// untransposed nest. The previous spelling was `ijk` with **both** operands read column-strided
/// (`s += a[k*NS+i]*b[k*NS+j]`) — a k-loop that touches two fresh cache lines per iteration and that
/// gcc cannot vectorize, which is not how a competent C programmer writes `C = Aᵀ·B`. The `kij` order
/// accumulates each `c[i][j]` over k in the identical ascending order, so the result is unchanged.
/// Measured standalone at `-O3 -march=native`, 512³, kernels in their own TU: **176.6 ms `ijk` →
/// 20.25 ms `ijk` + `restrict` → 9.70 ms `kij` + `restrict`, an 18.2× total handicap** — of which
/// 8.7× is `restrict` alone and 2.1× the loop order.
fn c_matmul_tn(ns: usize) -> String {
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         \x20 for (long t=0;t<(long)NS*NS;t++) c[t]=0.0f;\n\
         \x20 for (long k=0;k<NS;k++){{\n\
         \x20   for (long i=0;i<NS;i++){{\n\
         \x20     float aki=a[k*NS+i];\n\
         \x20     for (long j=0;j<NS;j++) c[i*NS+j] += aki*b[k*NS+j];\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

/// The Rust weight-gradient peer. Same `kij` nest as [`c_matmul_tn`], and the slices are taken
/// **as function parameters** (see the LANDMINE on [`rust_kernel`]) so this column gets the same
/// `noalias` its C twin gets from `__restrict__` — the defect whose isolated cost on this very
/// family was measured at 8.7× for the C peer. Building the slices inside `kbench` (the previous
/// spelling) grants no aliasing information at all.
fn rust_matmul_tn(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n\
         #[inline(always)]\n\
         fn kbody(a: &[f32], b: &[f32], c: &mut [f32]) {{\n\
         \x20 for v in c.iter_mut() {{ *v = 0.0; }}\n\
         \x20 for k in 0..NS {{\n\
         \x20   let brow = &b[k*NS..k*NS+NS];\n\
         \x20   for i in 0..NS {{\n\
         \x20     let aki = a[k*NS+i];\n\
         \x20     let crow = &mut c[i*NS..i*NS+NS];\n\
         \x20     for (cv, &bv) in crow.iter_mut().zip(brow) {{ *cv += aki*bv; }}\n\
         \x20   }}\n\
         \x20 }}\n}}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 kbody(core::slice::from_raw_parts(a, NS*NS), core::slice::from_raw_parts(b, NS*NS), \
         core::slice::from_raw_parts_mut(c, NS*NS))\n}}\n"
    )
}

// ---- GEMV (matrix-times-vector, the batch-1 attention/projection shape) -------------------------

/// GEMV `y[M] = A[M,N]·x[N]` — memory-bound (each A element read once, no reuse), so the figure of
/// merit is A-streaming GB/s. gcc/rustc keep the row dot a latency-bound in-order scalar/`vfmadd…ss`
/// chain (they will not reassociate an f32 reduction without `-ffast-math`); Wukong dispatches
/// `wukong_sgemv[_parallel]`, folding each row 8-wide across four accumulators, and the `@parallel`
/// form spreads the rows across cores to saturate aggregate bandwidth.
fn bench_gemv(cc: &str, dir: &Path) {
    for (m, n) in [(4096usize, 4096usize), (8192, 2048), (16384, 1024)] {
        let a: Vec<f32> = (0..m * n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.2 - 0.5).collect();
        let mut y = vec![0.0f32; m];
        let (ap, xp) = (a.as_ptr(), x.as_ptr());
        let bytes = (m * n) as f64 * 4.0; // A streamed once — the DRAM roofline traffic
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|mm| format!("{:.1}", bytes / mm.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== gemv (y[i] = Sum_j A[i,j]*x[j]) {m}x{n} (A-stream GB/s, higher is better) ==="
        );
        // POSITION-BALANCED TIMING (2026-09-13) - see `abba_min_columns` and the note in
        // `bench_colsum`. Build every column first, then time them all in one rotation.
        let wk1 = build_wukong(&wk_gemv(m, n, false), 3);
        let wkp = build_wukong(&wk_gemv(m, n, true), 3);
        let (cl, pl) = build_c_cpp(
            "gemv",
            &c_gemv(m, n),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let rl = build_peer(
            "rs",
            &rust_gemv(m, n),
            dir,
            "gemv",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer. Wukong's `wukong_sgemv` folds each row across four 8-wide
        // accumulators — it REASSOCIATES the dot — so a plain-flags C column that must keep the sum
        // strictly left-to-right is not the like-for-like comparison. Printed alongside, never
        // instead of, the honest-default C ratio.
        let fl = build_peer("c", &c_gemv(m, n), dir, "gemv_fast", cc, C_FAST_FLAGS);
        // LIBRARY-GRADE PEER (2026-09-13). `cblas_sgemv_64` is the exact primitive and it resolves
        // off the handle this process already holds; this family had no library bar at all.
        let (ar, xr) = (&a[..], &x[..]);
        let mkl_all_threads = mkl().map(|api| api.max_threads);
        let cols_all = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(fl),
            Built::Direct(Box::new(move |o: &mut [f32]| {
                bench_gemv_mkl(m, n, 1, ar, xr, o)
            })),
            match mkl_all_threads {
                Some(t) => Built::Direct(Box::new(move |o: &mut [f32]| {
                    bench_gemv_mkl(m, n, t, ar, xr, o)
                })),
                None => Built::Missing,
            },
            Built::peer(rl),
        ];
        let mut timed =
            abba_min_columns(cols_all.len(), |i| time_built(&cols_all[i], &mut y, ap, xp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("gemv", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("gemv", "C(fast)", m, p))
        });
        // MKL reassociates the row dot exactly as Wukong does, so the bar is the harness's
        // magnitude-normalized 1e-2 relaxed bar rather than the 1e-3 like-for-like one.
        let mkl_1c = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("gemv", "MKL(1c)", m, p))
        });
        let mkl_all = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("gemv", "MKL(all)", m, p))
        });
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "",
            "Wuk(1core)",
            "Wuk(par)",
            "C (gcc)",
            "C++ (g++)",
            "C(fast)",
            "MKL(1c)",
            "MKL(all)",
            "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&cfast),
            gbps(&mkl_1c),
            gbps(&mkl_all),
            gbps(&rm)
        );
        // The 8-wide row dot reassociates → magnitude-normalized tolerance (max|Δ| / max|C|), not a
        // pointwise ratio (mean-zero inputs put outputs near 0). Same basis as the reduction cross-checks.
        if let (Some(a2), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2
                .out
                .iter()
                .fold(0.0f32, |mx, &v| mx.max(v.abs()))
                .max(1e-6);
            let maxerr = a2
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |mx, (&u, &v)| mx.max((u - v).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! gemv mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

fn wk_gemv(m: usize, n: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mn = m * n;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {mn}], x: [f32; {n}], mut y: [f32; {m}]) {{\n\
         \x20   for i in 0..{m} {{\n\
         \x20       let mut s: f32 = 0.0;\n\
         \x20       for j in 0..{n} {{ s = s + a[i * {n} + j] * x[j]; }}\n\
         \x20       y[i] = s;\n\
         \x20   }}\n}}\n"
    )
}

fn c_gemv(m: usize, n: usize) -> String {
    format!(
        "#define M {m}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ x, float* __restrict__ y){{\n\
         \x20 for (long i=0;i<M;i++){{ float s=0.0f; for (long j=0;j<N;j++) s+=a[i*N+j]*x[j]; y[i]=s; }} }}\n"
    )
}

fn rust_gemv(m: usize, n: usize) -> String {
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(a:*const f32, x:*const f32, y:*mut f32) {{\n\
         \x20 for i in 0..M {{ let mut s=0.0f32; for j in 0..N {{ s += *a.add(i*N+j) * *x.add(j); }} *y.add(i)=s; }} }}\n"
    )
}

// ---- α-scaled GEMM (attention scores QKᵀ·(1/√d)) ------------------------------------------------

/// α-scaled attention scores `scores[S,S] = (Q[S,D]·K[S,D]ᵀ)·scale` (`scale = 1/√d`, a literal here).
/// The `* scale` on the store previously blocked the matmul matcher and dropped the whole GEMM to a
/// scalar nest; Wukong now dispatches `wukong_sgemm_nt_alpha[_parallel]`, folding the scale into the
/// tuned GEMM C-tile writeback. Compute-bound → GFLOP/s. We also report Wuk(α) vs Wuk(unscaled) to
/// show the α is *free* (one FMA folded into a writeback the GEMM already does), not a second pass.
fn bench_scaled_gemm(cc: &str, dir: &Path) {
    for (s, d) in [(256usize, 64usize), (512, 64), (512, 128)] {
        let q: Vec<f32> = (0..s * d).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let k: Vec<f32> = (0..s * d).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let mut out = vec![0.0f32; s * s];
        let (qp, kp) = (q.as_ptr(), k.as_ptr());
        let flops = 2.0 * (s as f64) * (s as f64) * (d as f64); // the GEMM; the S² scale mults are noise
        let gflops = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== scaled_gemm (scores = (Q·Kᵀ)·scale) S={s} D={d} (GFLOP/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_scaled_scores(s, d, false), 3);
        let built_wk_par = build_wukong(&wk_scaled_scores(s, d, true), 3);
        // Unscaled QKᵀ (plain nt) — same tuned kernel without the α; the ratio isolates the α cost.
        let built_wk_noscale = build_wukong(&wk_scores_noscale(s, d, false), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "scaled_gemm",
            &c_scaled_scores(s, d),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_scaled_scores(s, d),
            dir,
            "scaled_gemm",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer: the D-long score dot is a float reduction Wukong's blocked
        // GEMM accumulates out of order, so C is given the same freedom in this column.
        let built_cfast = build_peer(
            "c",
            &c_scaled_scores(s, d),
            dir,
            &format!("{}_fast", "scaled_gemm"),
            cc,
            C_FAST_FLAGS,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::wk(built_wk_noscale),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
            Built::peer(built_cfast),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, qp, kp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let wk_noscale = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("scaled_gemm", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("scaled_gemm", "C(fast)", m, p))
        });

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
            gflops(&cm_cpp),
            gflops(&cfast),
            gflops(&rm)
        );
        // The D-long score dot reassociates (FMA + blocked accumulation vs C's naive scalar order), and
        // the small mixed-sign Q/K put some scores near 0 → a pointwise relative check divides by ~0 and
        // blows up on a purely-reassociation Δ. Use the magnitude-normalized tolerance (max|Δ| / max|C|),
        // the same basis as gemv and the reduction cross-checks. The bit-exact gate is the runtime f64
        // reference + interp==native unit test; this is only a sanity ceiling.
        if let (Some(a2), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2
                .out
                .iter()
                .fold(0.0f32, |mx, &v| mx.max(v.abs()))
                .max(1e-6);
            let maxerr = a2
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |mx, (&u, &v)| mx.max((u - v).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! scaled_gemm mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        if let (Some(ms), Some(mn)) = (&wuk, &wk_noscale) {
            // >1 ⇒ the α costs time; ~1.0 ⇒ the scale is free (folded into the writeback).
            let r = ms.ns_per_call / mn.ns_per_call;
            println!(
                "  -> α overhead vs unscaled QKᵀ (same kernel): {r:.3}x (≈1.0 ⇒ the scale is free)"
            );
        }
        println!();
    }
}

fn wk_scaled_scores(s: usize, d: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let (sd, ss) = (s * d, s * s);
    format!(
        "module bench\n{attr}fn kbench(q: [f32; {sd}], k: [f32; {sd}], mut out: [f32; {ss}]) {{\n\
         \x20   for i in 0..{s} {{\n\
         \x20       for j in 0..{s} {{\n\
         \x20           let mut acc: f32 = 0.0;\n\
         \x20           for p in 0..{d} {{ acc = acc + q[i * {d} + p] * k[j * {d} + p]; }}\n\
         \x20           out[i * {s} + j] = 0.125 * acc;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

/// The unscaled twin (plain `Q·Kᵀ`, dispatches `wukong_sgemm_nt`) — the α-overhead baseline.
fn wk_scores_noscale(s: usize, d: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let (sd, ss) = (s * d, s * s);
    format!(
        "module bench\n{attr}fn kbench(q: [f32; {sd}], k: [f32; {sd}], mut out: [f32; {ss}]) {{\n\
         \x20   for i in 0..{s} {{\n\
         \x20       for j in 0..{s} {{\n\
         \x20           let mut acc: f32 = 0.0;\n\
         \x20           for p in 0..{d} {{ acc = acc + q[i * {d} + p] * k[j * {d} + p]; }}\n\
         \x20           out[i * {s} + j] = acc;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_scaled_scores(s: usize, d: usize) -> String {
    format!(
        "#define S {s}\n#define D {d}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ q, const float* __restrict__ k, float* __restrict__ out){{\n\
         \x20 for (long i=0;i<S;i++){{ for (long j=0;j<S;j++){{ float s=0.0f; for (long p=0;p<D;p++) s+=q[i*D+p]*k[j*D+p]; out[i*S+j]=0.125f*s; }} }} }}\n"
    )
}

fn rust_scaled_scores(s: usize, d: usize) -> String {
    format!(
        "const S: usize = {s};\nconst D: usize = {d};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(q:*const f32, k:*const f32, out:*mut f32) {{\n\
         \x20 for i in 0..S {{ for j in 0..S {{ let mut s=0.0f32; for p in 0..D {{ s += *q.add(i*D+p) * *k.add(j*D+p); }} *out.add(i*S+j)=0.125f32*s; }} }} }}\n"
    )
}

/// `nn.Linear`: `C = A·Bᵀ` (A is `[M,K]`, B is `[N,K]`), the matmul every Dense layer runs. Wukong
/// recognizes the transposed-B nest and dispatches to its tuned GEMM; idiomatic C/Rust write the
/// naive nest. Square M=K=N for the harness's shared-buffer ABI.
fn bench_linear(cc: &str, dir: &Path, roof: f64) {
    for ns in [512usize, 1024] {
        let n2 = ns * ns;
        let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut c = vec![0.0f32; n2];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        let gflops = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== linear (nn.Linear C=A·Bᵀ) {ns}x{ns} (GFLOP/s, higher is better) ===");
        // Build every column before the shared position-balanced rotation.
        let wk1 = build_wukong(&wk_linear(ns, false), 3);
        let wkp = build_wukong(&wk_linear(ns, true), 3);
        let (cl, pl) = build_c_cpp(
            "linear",
            &c_linear(ns),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let fl = build_peer("c", &c_linear(ns), dir, "linear_fast", cc, C_FAST_FLAGS);
        let rl = build_peer(
            "rs",
            &rust_linear(ns),
            dir,
            "linear",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let ol = omp_threads(cc, dir).and_then(|_| {
            build_peer(
                "c",
                &c_linear_omp(ns),
                dir,
                "linear_omp",
                cc,
                C_OMP_FAST_FLAGS,
            )
        });
        let (ar, br) = (&a[..], &b[..]);
        let mkl_all_threads = mkl().map(|api| api.max_threads);
        let columns = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(fl),
            Built::peer(rl),
            Built::peer(ol),
            Built::Direct(Box::new(move |o| bench_mm_mkl(ns, true, 1, ar, br, o))),
            match mkl_all_threads {
                Some(t) => Built::Direct(Box::new(move |o| bench_mm_mkl(ns, true, t, ar, br, o))),
                None => Built::Missing,
            },
            Built::Direct(Box::new(move |o| Some(bench_mm_tuned(ns, true, ar, br, o)))),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut c, ap, bp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("linear", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("linear", "C(fast)", m, p))
        });
        let rm = timed.next().flatten();
        let comp = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("linear", "C(omp)", m, p))
        });
        let mkl_1c = timed.next().flatten();
        let mkl_all = timed.next().flatten();
        let tuned = timed
            .next()
            .flatten()
            .expect("the in-process tuned GEMM column always returns a measure");
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "",
            "Wuk(1c)",
            "Wuk(par)",
            "MKL(1c)",
            "MKL(all)",
            "tuned(mm)",
            "C(gcc)",
            "C++(g++)",
            "C(fast)",
            "C(omp)",
            "Rust"
        );
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&mkl_1c),
            gflops(&mkl_all),
            gflops(&Some(tuned.clone())),
            gflops(&cm),
            gflops(&cm_cpp),
            gflops(&cfast),
            gflops(&comp),
            gflops(&rm)
        );
        report_gemm_standing(&wuk, &tuned, roof, flops);
        report_gemm_vs_mkl(&wuk, &wk_par, &mkl_1c, &mkl_all, flops);
        if let (Some(mp), Some(c)) = (&wk_par, &cm) {
            let r = (flops / mp.ns_per_call) / (flops / c.ns_per_call);
            println!(
                "  -> Wukong @parallel is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        report_relaxed_ratio(
            "C(omp) [-fopenmp -ffast-math, all cores]",
            &wuk,
            &wk_par,
            &comp,
        );
        println!();
    }
}

/// `nn.Linear` source, written the *idiomatic* way for `C = A·Bᵀ`: the `ijk` dot-product form, where
/// each `C[i,j]` is the dot product of A's row i and B's row j — both read contiguously (cache
/// friendly for all three languages). Wukong recognizes this and dispatches to its tiled GEMM;
/// gcc/rustc vectorize the inner reduction but never tile/pack, so Wukong wins on cache behavior.
fn wk_linear(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{ns} {{ s = s + a[i * {ns} + k] * b[j * {ns} + k]; }}\n\
         \x20           c[i * {ns} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

/// Rectangular `nn.Linear` NT nest (`C[M,N] = A[M,K] · B[N,K]ᵀ`) — the same dot-product spelling
/// as [`wk_linear`] (so the recognizer dispatches the identical `wukong_sgemm_nt[_parallel]`),
/// with independent M/K/N for the skinny transformer shapes.
fn wk_linear_rect(m: usize, k: usize, n: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let (mk, nk, mn) = (m * k, n * k, m * n);
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {mk}], b: [f32; {nk}], mut c: [f32; {mn}]) {{\n\
         \x20   for i in 0..{m} {{\n\
         \x20       for j in 0..{n} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{k} {{ s = s + a[i * {k} + k] * b[j * {k} + k]; }}\n\
         \x20           c[i * {n} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_linear(ns: usize) -> String {
    c_linear_rect(ns, ns, ns)
}

/// The RECTANGULAR `nn.Linear` C peer `C[M,N] = A[M,K]·B[N,K]ᵀ` — the same `ijk` dot-product nest as
/// [`c_linear`], with the three dimensions independent. Needed by the skinny-GEMM sweep, whose shapes
/// (M=128/512 rows against the GPT-2 block's 768/3072 weight dims) are never square. Both operands are
/// walked with unit stride over `k`, which is the natural spelling for the NT layout — the row-major
/// competence rule the 2026-08-04 audit added.
fn c_linear_rect(m: usize, k: usize, n: usize) -> String {
    format!(
        "#define M {m}\n#define K {k}\n#define N {n}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         \x20 for (long i=0;i<M;i++)\n\
         \x20   for (long j=0;j<N;j++){{ float s=0.0f;\n\
         \x20     for (long p=0;p<K;p++) s+=a[i*K+p]*b[j*K+p];\n\
         \x20     c[i*N+j]=s; }}\n}}\n"
    )
}

/// The OpenMP twin of [`c_linear`]: the `ijk` dot-product nest with rows across cores. Compiled with
/// [`C_OMP_FAST_FLAGS`] (the inner loop is a reduction, so -ffast-math lets each thread's dot
/// vectorize — the strongest honest multithreaded C for `nn.Linear`).
fn c_linear_omp(ns: usize) -> String {
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         #pragma omp parallel for\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++){{ float s=0.0f;\n\
         \x20     for (long k=0;k<NS;k++) s+=a[i*NS+k]*b[j*NS+k];\n\
         \x20     c[i*NS+j]=s; }}\n}}\n"
    )
}

fn rust_linear(ns: usize) -> String {
    rust_linear_rect(ns, ns, ns)
}

/// The rectangular Rust twin of [`c_linear_rect`].
fn rust_linear_rect(m: usize, k: usize, n: usize) -> String {
    format!(
        "const M: usize = {m};\nconst K: usize = {k};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 for i in 0..M {{\n\
         \x20   for j in 0..N {{ let mut s=0.0f32;\n\
         \x20     for p in 0..K {{ s+=*a.add(i*K+p)* *b.add(j*K+p); }}\n\
         \x20     *c.add(i*N+j)=s; }} }}\n}}\n"
    )
}

/// Fused FFN projection `C = silu(A·Bᵀ)` — the real transformer Dense / SwiGLU layer (a `nn.Linear`
/// immediately followed by an activation). Wukong's epilogue-fusion look-ahead folds the matmul and
/// the silu pass into ONE `wukong_sgemm_nt_epi` call (SILU act, null bias): the activation is computed
/// in-register on the GEMM's C-tile writeback, so C is written exactly once. The idiomatic C/Rust do
/// the GEMM, then a SECOND full pass that reads C back and applies silu with scalar libm `expf` (which
/// they cannot vectorize) — paying both an extra n² memory round-trip AND a scalar transcendental pass
/// on top of the GEMM. So Wukong wins the GEMM tiling, the fused (vectorized) silu, and the saved
/// re-stream all at once.
///
/// HONEST DISCLOSURE (same basis as `bench_linear`/`dot`): the GEMM is the idiomatic dot-product
/// `C=A·Bᵀ` form, whose inner float reduction gcc/rustc keep SERIAL without `-ffast-math` (~5 GFLOP/s).
/// So the BULK of the headline ratio is the tiled-vs-serial-reduction GEMM gap (≈ the ~22× `bench_linear`
/// single-core shows on the very same form), and the fused silu is the *incremental* win over C's
/// separate scalar-`expf` pass (it widens the lead a little and proves the activation does not erode it).
/// Square M=K=N for the shared-buffer ABI; GFLOP/s on the 2·n³ GEMM-flop basis (the n² silu is fused in).
/// silu(GEMM): the GEMM reassociates and silu is poly-vs-libm ~1 ULP, so the full-buffer check is a tight
/// tolerance (`max_rel_err`'s near-zero floor covers silu's zero crossing), exactly like `bench_matmul`.
fn bench_ffn(cc: &str, dir: &Path, roof: f64) {
    for ns in [512usize, 1024] {
        let n2 = ns * ns;
        let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut c = vec![0.0f32; n2];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        let gflops = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== ffn (fused C=silu(A·Bᵀ): matmul+act folded into one C-write) {ns}x{ns} (GFLOP/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_ffn(ns, false), 3);
        let built_wk_par = build_wukong(&wk_ffn(ns, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "ffn",
            &c_ffn(ns),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_ffn(ns),
            dir,
            "ffn",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer for the GEMM half of the fused op (-ffast-math may
        // reassociate the dot-product; the scalar-expf silu pass is unaffected either way).
        let built_cfast = build_peer(
            "c",
            &c_ffn(ns),
            dir,
            &format!("{}_fast", "ffn"),
            cc,
            C_FAST_FLAGS,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
            Built::peer(built_cfast),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut c, ap, bp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("ffn", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("ffn", "C(fast)", m, p))
        });

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
            gflops(&cm_cpp),
            gflops(&cfast),
            gflops(&rm)
        );
        // Absolute GFLOP/s on this laptop swings ~3x with the power state, so the number above is
        // only comparable within a run. The clock-invariant denominator is the roofline measured by
        // the SAME process (`measure_fma_roofline`) — print the percentage next to the table, as
        // `bench_matmul_skinny`/`bench_linear` already do.
        if roof > 0.0 {
            if let Some(w) = &wuk {
                println!(
                    "  -> Wukong single-core = {:.0}% of measured roofline",
                    (flops / w.ns_per_call) / roof * 100.0
                );
            }
        }
        // Cross-language correctness over the whole buffer (silu(GEMM) — tight tolerance, see doc).
        if let (Some(m), Some(c)) = (&wk_par, &cm) {
            let (rel, at) = max_rel_err(&m.out, &c.out);
            if rel > 1e-3 {
                println!(
                    "  ! full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {:.2e})",
                    m.out[at], c.out[at], rel
                );
            }
        }
        if let (Some(ms), Some(c)) = (&wuk, &cm) {
            let r = flops / ms.ns_per_call / (flops / c.ns_per_call);
            println!(
                "  -> Wukong single-core (fused GEMM+silu) is {:.2}x {} than C (un-tiled GEMM + scalar-expf silu pass)",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c)) = (&wk_par, &cm) {
            let r = flops / mp.ns_per_call / (flops / c.ns_per_call);
            println!(
                "  -> Wukong @parallel is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// `C = silu(A·Bᵀ)` written as the matmul nest + a silu epilogue nest (two consecutive statements).
/// Wukong's `lower_block` look-ahead fuses them into one `wukong_sgemm_nt_epi` (SILU, null bias).
fn wk_ffn(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{ns} {{ s = s + a[i * {ns} + k] * b[j * {ns} + k]; }}\n\
         \x20           c[i * {ns} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{ c[i * {ns} + j] = silu(c[i * {ns} + j]); }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_ffn(ns: usize) -> String {
    format!(
        "#include <math.h>\n#define NS {ns}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++){{ float s=0.0f;\n\
         \x20     for (long k=0;k<NS;k++) s+=a[i*NS+k]*b[j*NS+k];\n\
         \x20     c[i*NS+j]=s; }}\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++){{ float v=c[i*NS+j]; c[i*NS+j]=v/(1.0f+expf(-v)); }}\n}}\n"
    )
}

fn rust_ffn(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{ let mut s=0.0f32;\n\
         \x20     for k in 0..NS {{ s+=*a.add(i*NS+k)* *b.add(j*NS+k); }}\n\
         \x20     *c.add(i*NS+j)=s; }} }}\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{ let v=*c.add(i*NS+j); *c.add(i*NS+j)=v/(1.0f32+(-v).exp()); }} }}\n}}\n"
    )
}

/// bf16 mixed-precision `nn.Linear` (`C = A·Bᵀ`, bf16 inputs, f32 accumulate) — the standard
/// transformer matmul. Wukong recognizes the half-precision dot-product nest and folds it to one
/// `wukong_sgemm_bf16_nt[_parallel]` call (a lossless widen prepass + the tuned AVX2 f32 GEMM); the
/// idiomatic C/Rust store bf16 as `uint16_t` and widen each element inline inside the triple loop —
/// which they can't vectorize, and the sequential float reduction stays scalar (no `-ffast-math`),
/// the same basis as the f32 `linear`/`dot` kernels. Reuses the `(u16, u16, f32)` bench ABI. The
/// `out[0]` (= `c[0]`) cross-check is a sanity guard; the bit-exact correctness rests on the runtime
/// twin test + the interp/native differential gate. The f16 twin (`wukong_sgemm_f16_nt`) is the same
/// dispatch with an F16C-exact widen.
fn bench_linear_bf16(cc: &str, dir: &Path) {
    for ns in [512usize, 1024] {
        let n2 = ns * ns;
        // bf16 stored bits of small well-conditioned values (the cross-check is a tolerance anyway).
        let a: Vec<u16> = (0..n2)
            .map(|i| to_bf16_bits((i % 7) as f32 * 0.5 + 0.1))
            .collect();
        let b: Vec<u16> = (0..n2)
            .map(|i| to_bf16_bits((i % 5) as f32 * 0.25 - 0.3))
            .collect();
        let mut c = vec![0.0f32; n2];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        let gflops = |m: &Option<MeasureBf16>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== linear_bf16 (bf16 nn.Linear C=A·Bᵀ, f32 accumulate) {ns}x{ns} (GFLOP/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_linear_bf16(ns, false), 3);
        let built_wk_par = build_wukong(&wk_linear_bf16(ns, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "linear_bf16",
            &c_linear_bf16(ns),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_linear_bf16(ns),
            dir,
            "linear_bf16",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer: the K-long f32 accumulation is a float reduction, and
        // Wukong's widen-prepass + tuned GEMM accumulates it blocked. Validity-checked at the loose
        // 1e-2 bar on c[0], the same sanity guard the plain column uses.
        let built_cfast = build_peer(
            "c",
            &c_linear_bf16(ns),
            dir,
            &format!("{}_fast", "linear_bf16"),
            cc,
            C_FAST_FLAGS,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            BuiltBf16::Wukong(built_wuk.map(Box::new)),
            BuiltBf16::Wukong(built_wk_par.map(Box::new)),
            BuiltBf16::Peer(built_cm),
            BuiltBf16::Peer(built_cm_cpp),
            BuiltBf16::Peer(built_rm),
            BuiltBf16::Peer(built_cfast),
        ];
        let mut timed = abba_min_columns(columns.len(), |i| {
            time_built_bf16(&columns[i], &mut c, ap, bp)
        })
        .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked_bf16("linear_bf16", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        let cfast = timed
            .next()
            .flatten()
            .and_then(|p| bf16_fast_checked("linear_bf16", wuk.as_ref()?, p));

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
            gflops(&cm_cpp),
            gflops(&cfast),
            gflops(&rm)
        );
        // Sanity cross-check on c[0] (the bf16 widen is lossless, so all three compute the same GEMM
        // up to the documented float-reassociation tolerance).
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let rel = ((m.out - c2.out).abs() / c2.out.abs().max(1e-6)) as f64;
            if rel > 1e-2 {
                println!(
                    "  ! c[0] mismatch vs C: Wuk={} C={} (rel {:.2e})",
                    m.out, c2.out, rel
                );
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic bf16 C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            println!(
                "  -> Wukong @parallel is {:.2}x {} than idiomatic single-threaded bf16 C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        bf16_relaxed_ratio("idiomatic C++ (g++)", &wuk, &wk_par, &cm_cpp);
        bf16_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// Wukong bf16 `nn.Linear`, the idiomatic `ijk` dot-product `C = A·Bᵀ` with `[bf16]` operands widened
/// `as f32` and an f32 accumulator — what the `mir_build` recognizer folds to `wukong_sgemm_bf16_nt`.
fn wk_linear_bf16(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [bf16; {n2}], b: [bf16; {n2}], mut c: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{ns} {{ s = s + (a[i * {ns} + k] as f32) * (b[j * {ns} + k] as f32); }}\n\
         \x20           c[i * {ns} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_linear_bf16(ns: usize) -> String {
    format!(
        "#include <stdint.h>\n#include <string.h>\n#define NS {ns}\n\
         static inline float bf(uint16_t b){{ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }}\n\
         __declspec(dllexport) void kbench(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, float* __restrict__ c){{\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++){{ float s=0.0f;\n\
         \x20     for (long k=0;k<NS;k++) s += bf(a[i*NS+k]) * bf(b[j*NS+k]);\n\
         \x20     c[i*NS+j]=s; }}\n}}\n"
    )
}

fn rust_linear_bf16(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[inline] fn bf(b:u16)->f32 {{ f32::from_bits((b as u32)<<16) }}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const u16, b:*const u16, c:*mut f32) {{\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{ let mut s=0.0f32;\n\
         \x20     for k in 0..NS {{ s += bf(*a.add(i*NS+k)) * bf(*b.add(j*NS+k)); }}\n\
         \x20     *c.add(i*NS+j)=s; }} }}\n}}\n"
    )
}

/// The shapes [`bench_transpose`] sweeps, as `(rows, cols, aliasing_stride)`.
///
/// **Both regimes are measured on purpose, and they do NOT give the same answer.** Until 2026-08-05
/// this bench ran `ns in [1024, 2048]` and nothing else — two power-of-two squares — and the ~5–7×
/// multiple it printed was published as *the* transpose result. It is not. A transpose reads one
/// matrix row-major and writes the other column-major, so the peer's write stream steps by the
/// destination row stride; when that stride in BYTES is a large power of two, consecutive writes map
/// onto the same handful of L1 sets and the blocked C peer thrashes on set conflicts. Wukong's
/// `wukong_transpose_f32` moves 32×32 tiles through registers and is much less exposed, so the
/// power-of-two shapes flatter it for a reason that is a property of the SIZE, not of the compiler.
/// Measured on this box the same kernel pair goes from ~5–7× at 1024²/2048² down to ~1.4× at 1000²
/// and 1031² — so the honest general number is the small one, and the old sweep could not see it.
///
/// `false` marks the general regime: 1000² (stride 4000 B), 1031² (prime) and the rectangular
/// 1100×950 (different read and write strides, neither a power of two). The power-of-two shapes are
/// kept — the aliasing regime is real and worth reporting — but they are labelled, geomeaned
/// separately, and the summary says in words which number generalizes.
const TRANSPOSE_SHAPES: &[(usize, usize, bool)] = &[
    (1024, 1024, true),
    (2048, 2048, true),
    (1000, 1000, false),
    (1031, 1031, false),
    (1100, 950, false),
];

/// Matrix transpose `dst = srcᵀ` — the memory-bound layout op (attention score transposes, weight
/// layout conversions). Wukong folds the `dst[j*R+i] = src[i*C+j]` nest to the cache-blocked
/// `wukong_transpose_f32`; the C/Rust peers are **also** 32×32 cache-blocked (see [`c_transpose`]),
/// so what is measured is Wukong's tile-mover against gcc's/rustc's codegen for the same algorithm —
/// not blocked-vs-unblocked. gcc still does not loop-tile a transpose *by itself*, which is why the
/// blocking has to be written out; the point is that a competent programmer writes it.
/// The kernels carry an unused middle pointer so they share the `(src, _, dst)` `KernelFn` ABI and the
/// f32 harness. Shapes large enough to spill L2 (where the cache pattern dominates), reported as
/// GB/s (`2·R·C·4` bytes moved per call: read `src` + write `dst`). Transpose is a permutation, so the
/// cross-language check is **bit-exact** (no float reassociation — a stronger bar than the GEMM gate).
///
/// The sweep covers BOTH stride regimes — see [`TRANSPOSE_SHAPES`] — and closes with a summary that
/// geomeans them apart and names the general one, because reporting only the power-of-two rows is
/// reporting the size at which the peer is worst.
/// Keep a oneMKL transpose column only if it is BIT-identical to Wukong's.
///
/// A transposition is a permutation, so there is no reassociation to excuse a difference and no
/// tolerance to invent: `MKL_Somatcopy(alpha = 1)` must move exactly the same bits. A column that
/// disagrees is computing something else and is dropped rather than printed beside a ratio - the
/// rule [`relaxed_peer_ok`] and [`cpp_checked`] already follow.
///
/// BITS, NOT VALUES. `Vec<f32>` equality compares by VALUE, so `-0.0 == 0.0` and the check would
/// have passed a column that moved a sign bit — while this doc, and [`cpp_checked`] which it cites
/// as its model, both say BIT-identical. For a pure permutation the two bars can only differ when
/// something is wrong, which is exactly when the check has to hold, so it compares `to_bits` the
/// way `cpp_checked` does. (A NaN payload is the other case the value comparison silently accepts:
/// `NaN != NaN`, so a value comparison would REJECT two identical NaNs and accept two different
/// ones. Bits get both right.)
fn mkl_transpose_exact(wuk: &Option<Measure>, p: Option<Measure>, label: &str) -> Option<Measure> {
    let p = p?;
    let bitwise_differs = |a: &[f32], b: &[f32]| {
        a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
    };
    match wuk {
        Some(m) if bitwise_differs(&m.out, &p.out) => {
            println!(
                "  ! transpose: {label} (MKL_Somatcopy) disagrees with Wukong — column dropped"
            );
            None
        }
        _ => Some(p),
    }
}

fn bench_transpose(cc: &str, dir: &Path) {
    // (single-core ratio vs blocked C, @parallel ratio vs blocked C), split by regime.
    let (mut alias_1c, mut gen_1c): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let (mut alias_par, mut gen_par): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for &(rows, cols, aliasing) in TRANSPOSE_SHAPES {
        let n2 = rows * cols;
        let src: Vec<f32> = (0..n2).map(|i| (i % 1000) as f32 * 0.5 - 250.0).collect();
        let dummy = vec![0.0f32; n2];
        let mut dst = vec![0.0f32; n2];
        let (sp, yp) = (src.as_ptr(), dummy.as_ptr());
        let bytes = 2.0 * n2 as f64 * 4.0; // read src + write dst
        let gbps = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", bytes / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        let regime = if aliasing {
            "power-of-two stride - FAVOURABLE regime, does NOT generalize"
        } else {
            "general stride"
        };
        println!(
            "=== transpose (dst = srcᵀ) {rows}x{cols} [{regime}] (GB/s, higher is better) ==="
        );
        // POSITION-BALANCED TIMING (2026-09-13) - see `abba_min_columns` and the note in
        // `bench_colsum`. Build every column first, then time them all in one rotation.
        let wk1 = build_wukong(&wk_transpose(rows, cols, false), 3);
        let wkp = build_wukong(&wk_transpose(rows, cols, true), 3);
        let (cl, pl) = build_c_cpp(
            "transpose",
            &c_transpose(rows, cols),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        // Multithreaded C peer: rows across cores (a permutation — no reduction, plain -fopenmp).
        let ol = omp_threads(cc, dir).and_then(|_| {
            build_peer(
                "c",
                &c_transpose_omp(rows, cols),
                dir,
                "transpose_omp",
                cc,
                C_OMP_FLAGS,
            )
        });
        let rl = build_peer(
            "rs",
            &rust_transpose(rows, cols),
            dir,
            "transpose",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // LIBRARY-GRADE PEER (2026-09-13). `mkl_somatcopy` is the exact primitive this section
        // benchmarks and it was sitting unloaded in the same process: until now W8 was being tuned
        // against gcc's blocked nest and nothing else. Beating a hand-written loop is not beating
        // the state of the art, so the bar here is now Intel's.
        let mkl_all_threads = mkl().map(|api| api.max_threads);
        // A shared `&[f32]` (Copy) rather than a borrow of the `Vec`, so both `move` closures can
        // hold it without either of them taking ownership of `src`.
        let srcr: &[f32] = &src;
        let cols_all = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::Direct(Box::new(move |d: &mut [f32]| {
                bench_transpose_mkl(rows, cols, 1, srcr, d)
            })),
            match mkl_all_threads {
                Some(t) => Built::Direct(Box::new(move |d: &mut [f32]| {
                    bench_transpose_mkl(rows, cols, t, srcr, d)
                })),
                None => Built::Missing,
            },
            Built::peer(ol),
            Built::peer(rl),
        ];
        let mut timed = abba_min_columns(cols_all.len(), |i| {
            time_built(&cols_all[i], &mut dst, sp, yp)
        })
        .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("transpose", &cm, timed.next().flatten());
        // A transposition is a permutation: the MKL columns are held to BIT equality with Wukong,
        // the same bar the C and Rust columns already carry, and dropped on any disagreement.
        let mkl_1c = mkl_transpose_exact(&wuk, timed.next().flatten(), "MKL(1c)");
        let mkl_all = mkl_transpose_exact(&wuk, timed.next().flatten(), "MKL(all)");
        let comp = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("transpose", "C(omp)", m, p))
        });
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "",
            "Wuk(1core)",
            "Wuk(par)",
            "C (gcc)",
            "C++ (g++)",
            "MKL(1c)",
            "MKL(all)",
            "C(omp)",
            "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&mkl_1c),
            gbps(&mkl_all),
            gbps(&comp),
            gbps(&rm)
        );
        // Transpose is a permutation — exact, so the full-buffer cross-check is bit equality.
        // BOTH peers are checked: a peer whose spelling changed (blocked, sliced) could in principle
        // get "faster" by not doing the work, and an unchecked column would never say so. This is
        // also what makes the non-square shape safe: a peer that muddled `rows`/`cols` would produce
        // a different buffer, not a faster one.
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            if m.out != c2.out {
                println!("  ! transpose output mismatch vs C");
            }
        }
        if let (Some(m), Some(r2)) = (&wuk, &rm) {
            if m.out != r2.out {
                println!("  ! transpose output mismatch vs Rust");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
            if aliasing {
                alias_1c.push(r)
            } else {
                gen_1c.push(r)
            }
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
            if aliasing {
                alias_par.push(r)
            } else {
                gen_par.push(r)
            }
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(omp) [-fopenmp, all cores]", &wuk, &wk_par, &comp);
        println!();
    }
    transpose_regime_summary(&alias_1c, &gen_1c, &alias_par, &gen_par);
}

/// Close the transpose section by geomeaning the two stride regimes SEPARATELY and saying, in the
/// output itself, which one is the general result.
///
/// Printing one pooled number over `[1024², 2048², 1000², 1031², 1100×950]` would hide the split
/// just as effectively as sweeping only the power-of-two shapes did — a reader who takes the
/// headline away needs the caveat attached to it, not filed in a doc comment.
fn transpose_regime_summary(alias_1c: &[f64], gen_1c: &[f64], alias_par: &[f64], gen_par: &[f64]) {
    if alias_1c.is_empty() && gen_1c.is_empty() {
        return;
    }
    println!("=== transpose REGIME SUMMARY (vs the 32x32-blocked C peer, geomean of ratios) ===");
    let row = |label: &str, xs: &[f64]| {
        if xs.is_empty() {
            return;
        }
        let g = geomean(xs);
        println!(
            "  {:<44} {:>6.2}x {} ({} shape{})",
            label,
            if g >= 1.0 { g } else { 1.0 / g },
            if g >= 1.0 { "faster" } else { "slower" },
            xs.len(),
            if xs.len() == 1 { "" } else { "s" }
        );
    };
    row("single-core, power-of-two stride", alias_1c);
    row("single-core, general stride  <- THE RESULT", gen_1c);
    row("@parallel,   power-of-two stride", alias_par);
    row("@parallel,   general stride  <- THE RESULT", gen_par);
    println!(
        "  ! The power-of-two rows are NOT the general case. A row stride that is a large power of\n\
         \x20   two maps the peer's column-major write stream onto a few L1 sets, which penalizes the\n\
         \x20   PEER for the size rather than rewarding Wukong for the codegen. Quote the general-stride\n\
         \x20   line; the power-of-two line is the best case, and is labelled as such above."
    );
    println!();
}

/// Wukong transpose kernel: the idiomatic `dst[j*R+i] = src[i*C+j]` nest the `mir_build` recognizer
/// folds to one `wukong_transpose_f32[_parallel]` call. `y` is an unused param so the signature
/// matches the `(src, _, dst)` 3-pointer harness ABI. `rows`/`cols` are the SOURCE shape, so the
/// destination is `[cols, rows]` and the two strides differ on a non-square shape.
fn wk_transpose(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(src: [f32; {n2}], y: [f32; {n2}], mut dst: [f32; {n2}]) {{\n\
         \x20   for i in 0..{rows} {{\n\
         \x20       for j in 0..{cols} {{ dst[j * {rows} + i] = src[i * {cols} + j]; }}\n\
         \x20   }}\n}}\n"
    )
}

/// The C transpose peer, **cache-blocked at 32×32** — the textbook optimization for a transpose and
/// the same algorithm Wukong's `wukong_transpose_f32` uses, so the comparison is kernel-vs-kernel
/// rather than blocked-vs-unblocked. The old peer was the naive `for i { for j { dst[j*NS+i] =
/// src[i*NS+j] } }`, which writes `dst` with stride `NS` (a fresh cache line per element); the
/// bench's own commentary named that as the reason Wukong won, which makes it a peer defect, not a
/// compiler win. Measured standalone at `-O3 -march=native`, 2048², kernels in their own TU:
/// **29.51 ms naive → 24.31 ms naive + `restrict` → 14.09 ms 32×32-blocked + `restrict`, a 2.09×
/// total handicap**.
///
/// Since 2026-08-05 the sweep also covers shapes that are NOT multiples of the tile
/// ([`TRANSPOSE_SHAPES`]), so the nest carries edge handling. It is written as a **full-tile fast
/// path plus an edge case**, not as a clamped bound on every tile, precisely so the interior nest
/// gcc compiles is still the constant-trip-count `for (i=ii;i<ii+TB;i++)` it saw before this change:
/// a clamped bound would have made the tile trip count opaque and could have quietly slowed the peer
/// at 1024²/2048², which would have made Wukong look better at exactly the sizes this change exists
/// to put in context.
fn c_transpose(rows: usize, cols: usize) -> String {
    format!(
        "#define NR {rows}\n#define NC {cols}\n#define TB 32\n\
         __declspec(dllexport) void kbench(const float* __restrict__ src, const float* __restrict__ y, float* __restrict__ dst){{\n\
         \x20 (void)y;\n\
         \x20 for (long ii=0;ii<NR;ii+=TB) for (long jj=0;jj<NC;jj+=TB) {{\n\
         \x20   if (ii+TB<=NR && jj+TB<=NC) {{\n\
         \x20     for (long i=ii;i<ii+TB;i++)\n\
         \x20       for (long j=jj;j<jj+TB;j++) dst[j*NR+i] = src[i*NC+j];\n\
         \x20   }} else {{\n\
         \x20     long i1 = ii+TB<NR?ii+TB:NR, j1 = jj+TB<NC?jj+TB:NC;\n\
         \x20     for (long i=ii;i<i1;i++)\n\
         \x20       for (long j=jj;j<j1;j++) dst[j*NR+i] = src[i*NC+j];\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

/// The OpenMP twin of [`c_transpose`]: blocked tile rows across cores. A permutation, so it stays exact.
fn c_transpose_omp(rows: usize, cols: usize) -> String {
    format!(
        "#define NR {rows}\n#define NC {cols}\n#define TB 32\n\
         __declspec(dllexport) void kbench(const float* __restrict__ src, const float* __restrict__ y, float* __restrict__ dst){{\n\
         \x20 (void)y;\n\
         #pragma omp parallel for\n\
         \x20 for (long ii=0;ii<NR;ii+=TB) for (long jj=0;jj<NC;jj+=TB) {{\n\
         \x20   if (ii+TB<=NR && jj+TB<=NC) {{\n\
         \x20     for (long i=ii;i<ii+TB;i++)\n\
         \x20       for (long j=jj;j<jj+TB;j++) dst[j*NR+i] = src[i*NC+j];\n\
         \x20   }} else {{\n\
         \x20     long i1 = ii+TB<NR?ii+TB:NR, j1 = jj+TB<NC?jj+TB:NC;\n\
         \x20     for (long i=ii;i<i1;i++)\n\
         \x20       for (long j=jj;j<j1;j++) dst[j*NR+i] = src[i*NC+j];\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

/// The Rust transpose peer — the same blocked algorithm and the same full-tile/edge split as
/// [`c_transpose`], so the two peer languages stay comparable to each other as well as to Wukong.
/// The slices are function parameters, not locals, so this column carries `noalias` like its C
/// twin's `__restrict__` (see the LANDMINE on [`rust_kernel`]).
fn rust_transpose(rows: usize, cols: usize) -> String {
    format!(
        "const NR: usize = {rows};\nconst NC: usize = {cols};\nconst TB: usize = 32;\n\
         #[inline(always)]\nfn kbody(s: &[f32], d: &mut [f32]) {{\n\
         \x20 for ii in (0..NR).step_by(TB) {{ for jj in (0..NC).step_by(TB) {{\n\
         \x20   if ii+TB<=NR && jj+TB<=NC {{\n\
         \x20     for i in ii..ii+TB {{ for j in jj..jj+TB {{ d[j*NR+i] = s[i*NC+j]; }} }}\n\
         \x20   }} else {{\n\
         \x20     let i1 = if ii+TB<NR {{ ii+TB }} else {{ NR }};\n\
         \x20     let j1 = if jj+TB<NC {{ jj+TB }} else {{ NC }};\n\
         \x20     for i in ii..i1 {{ for j in jj..j1 {{ d[j*NR+i] = s[i*NC+j]; }} }}\n\
         \x20   }} }} }}\n}}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(src:*const f32, _y:*const f32, dst:*mut f32) {{\n\
         \x20 kbody(core::slice::from_raw_parts(src, NR*NC), core::slice::from_raw_parts_mut(dst, NR*NC))\n}}\n"
    )
}

/// Column reduction `out[j] = Σ_i x[i, j]` — the sum over the outer (batch/row) axis (the bias gradient
/// `db = Σ_batch dY`, batch sum, reduce-along-axis-0). Wukong folds the nest to `wukong_colsum_f32`,
/// which streams `x` row-major + 8 columns at a time. The kernels carry an unused middle pointer so
/// they share the `(x, _, out)` 3-pointer harness. Reported as GB/s (`M·N·4` bytes — the matrix read
/// once). Every language sums each column i-ascending, so the cross-check is **bit-exact**.
///
/// PEER SPELLING (corrected 2026-08-04). The C/Rust peers used to be written **column-outer**,
/// `for j { float s=0; for i { s += x[i*N+j]; } out[j]=s; }` — the single worst loop order for a
/// row-major column reduction (stride-`N` reads, one cache line touched per element, and gcc/rustc
/// leave it fully scalar). That was the *only* order measured, and the ~29–50× multiple this bench
/// published was mostly the peer's loop order, not Wukong's kernel. The peers are now **row-outer**,
/// `for i { for j { out[j] += x[i*N+j]; } }` — the natural, cache-friendly spelling, which folds each
/// column in the identical i-ascending order (so the bit-exact cross-check still holds) and which
/// gcc auto-vectorizes. Measured standalone at `-O3 -march=native`, 4096×1024, with the kernels in
/// their own translation unit so gcc sees only pointer parameters (the shared-library model this
/// harness actually uses): **21.33 ms column-outer → 5.03 ms column-outer + `restrict` → 0.58 ms
/// row-outer + `restrict`, a 36.7× total handicap**, with byte-identical output (sum |Δ| = 0).
fn bench_colsum(cc: &str, dir: &Path) {
    for (m, n) in [(1024usize, 1024usize), (4096, 1024)] {
        let mn = m * n;
        let x: Vec<f32> = (0..mn).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
        let dummy = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
        let bytes = mn as f64 * 4.0; // the matrix is read once
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== colsum (out[j] = Σ_i x[i,j]) {m}x{n} (GB/s, higher is better) ===");
        // POSITION-BALANCED TIMING (2026-09-13). Build every column FIRST, then time them all in
        // one rotation - see `abba_min_columns`. Before this, Wukong was compiled-and-timed in one
        // call, so it always held slot 1 (the coldest, quietest slot, with no compiler spawn in
        // front of it) while every peer's timed region sat behind two all-core gcc invocations.
        // The harness's own byte-identical control measured that slot worth +5.0% and +17.3% on the
        // two `colsum` rows, and the Wukong-over-peer ratio is the one ratio the campaign states
        // its targets in.
        let wk1 = build_wukong(&wk_colsum(m, n, false), 3);
        let wkp = build_wukong(&wk_colsum(m, n, true), 3);
        let (cl, pl) = build_c_cpp(
            "colsum",
            &c_colsum(m, n),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        // Reassociation-normalized peer: -ffast-math lets gcc reassociate the strided column fold.
        let fl = build_peer("c", &c_colsum(m, n), dir, "colsum_fast", cc, C_FAST_FLAGS);
        // Multithreaded C peer: columns across cores (+ -ffast-math, the reduction-row rule).
        let ol = omp_threads(cc, dir).and_then(|_| {
            build_peer(
                "c",
                &c_colsum_omp(m, n),
                dir,
                "colsum_omp",
                cc,
                C_OMP_FAST_FLAGS,
            )
        });
        let rl = build_peer(
            "rs",
            &rust_colsum(m, n),
            dir,
            "colsum",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let cols = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(fl),
            Built::peer(ol),
            Built::peer(rl),
        ];
        let mut timed =
            abba_min_columns(cols.len(), |i| time_built(&cols[i], &mut out, xp, yp)).into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("colsum", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|mm| relaxed_peer_ok("colsum", "C(fast)", mm, p))
        });
        let comp = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|mm| relaxed_peer_ok("colsum", "C(omp)", mm, p))
        });
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "C(omp)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&cfast),
            gbps(&comp),
            gbps(&rm)
        );
        // Both sum each column in i-ascending order, so the full-buffer cross-check is bit equality.
        if let (Some(a), Some(c2)) = (&wuk, &cm) {
            if a.out != c2.out {
                println!("  ! colsum output mismatch vs C");
            }
        }
        if let (Some(a), Some(r2)) = (&wuk, &rm) {
            if a.out != r2.out {
                println!("  ! colsum output mismatch vs Rust");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        report_relaxed_ratio(
            "C(omp) [-fopenmp -ffast-math, all cores]",
            &wuk,
            &wk_par,
            &comp,
        );
        println!();
    }
}

/// Wukong column-sum kernel: the idiomatic `for j { let s=0; for i { s += x[i*N+j] }; out[j]=s }` the
/// `mir_build` recognizer folds to one `wukong_colsum_f32[_parallel]` call. `y` is unused (the `(x, _,
/// out)` 3-pointer harness ABI). `out` is the `[N]` result; `x` is the `[M, N]` matrix.
fn wk_colsum(m: usize, n: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mn = m * n;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {mn}], y: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for j in 0..{n} {{\n\
         \x20       let mut s: f32 = 0.0;\n\
         \x20       for i in 0..{m} {{ s = s + x[i * {n} + j]; }}\n\
         \x20       out[j] = s;\n\
         \x20   }}\n}}\n"
    )
}

fn c_colsum(m: usize, n: usize) -> String {
    format!(
        "#define M {m}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long j=0;j<N;j++) out[j]=0.0f;\n\
         \x20 for (long i=0;i<M;i++) for (long j=0;j<N;j++) out[j] += x[i*N+j];\n}}\n"
    )
}

/// The OpenMP twin of [`c_colsum`]: **column blocks** across cores, each block folded row-outer
/// inside. Every `out[j]` is owned by exactly one thread, so there is no cross-thread combine and
/// the per-column fold order stays i-ascending — identical to the serial peer and to Wukong.
/// Compiled with [`C_OMP_FAST_FLAGS`].
fn c_colsum_omp(m: usize, n: usize) -> String {
    format!(
        "#define M {m}\n#define N {n}\n#define JB 64\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         #pragma omp parallel for\n\
         \x20 for (long jb=0;jb<N;jb+=JB){{ long je = jb+JB<N ? jb+JB : N;\n\
         \x20   for (long j=jb;j<je;j++) out[j]=0.0f;\n\
         \x20   for (long i=0;i<M;i++) for (long j=jb;j<je;j++) out[j] += x[i*N+j]; }}\n}}\n"
    )
}

fn rust_colsum(m: usize, n: usize) -> String {
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n\
         #[inline(always)]\nfn kbody(xs: &[f32], os: &mut [f32]) {{\n\
         \x20 for o in os.iter_mut() {{ *o = 0.0; }}\n\
         \x20 for i in 0..M {{ let row = &xs[i*N..i*N+N]; for (o, &v) in os.iter_mut().zip(row) {{ *o += v; }} }}\n}}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 kbody(core::slice::from_raw_parts(x, M*N), core::slice::from_raw_parts_mut(out, N))\n}}\n"
    )
}

/// Broadcast bias-add `out[r,c] = x[r,c] + bias[c]` over a `[R, C]` matrix — the post-projection bias
/// add / the affine after a non-fused norm / FiLM conditioning, one of the most ubiquitous transformer
/// elementwise ops. The bias is broadcast down the rows (one `[C]` vector reused for every row). It is
/// memory-bound (read x + write out, bias stays cached). Reported as GB/s (`2·R·C·4`: x read once,
/// out written once). The full-buffer cross-check is bit-exact (a plain add, no reassociation).
fn bench_biasadd(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 1024)] {
        let rc = r * c;
        let x: Vec<f32> = (0..rc).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
        let bias: Vec<f32> = (0..c).map(|i| (i % 13) as f32 * 0.5 - 1.0).collect();
        let mut out = vec![0.0f32; rc];
        let (xp, bp) = (x.as_ptr(), bias.as_ptr());
        let bytes = 2.0 * rc as f64 * 4.0; // read x once, write out once
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== biasadd (out[r,c] = x[r,c] + bias[c]) {r}x{c} (GB/s, higher is better) ===");
        // POSITION-BALANCED TIMING (2026-09-13) - see `abba_min_columns` and the note in
        // `bench_colsum`. Build every column first, then time them all in one rotation.
        let wk1 = build_wukong(&wk_biasadd(r, c, false), 3);
        let wkp = build_wukong(&wk_biasadd(r, c, true), 3);
        let (cl, pl) = build_c_cpp(
            "biasadd",
            &c_biasadd(r, c),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let rl = build_peer(
            "rs",
            &rust_biasadd(r, c),
            dir,
            "biasadd",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let cols = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(rl),
        ];
        let mut timed =
            abba_min_columns(cols.len(), |i| time_built(&cols[i], &mut out, xp, bp)).into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("biasadd", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&rm)
        );
        if let (Some(a), Some(c2)) = (&wuk, &cm) {
            if a.out != c2.out {
                println!("  ! biasadd output mismatch vs C");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let ratio = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than C",
                if ratio >= 1.0 { ratio } else { 1.0 / ratio },
                if ratio >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let ratio = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", ratio);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        println!();
    }
}

fn wk_biasadd(r: usize, c: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let rc = r * c;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {rc}], bias: [f32; {c}], mut out: [f32; {rc}]) {{\n\
         \x20   for r in 0..{r} {{\n\
         \x20       for c in 0..{c} {{ out[r * {c} + c] = x[r * {c} + c] + bias[c]; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_biasadd(r: usize, c: usize) -> String {
    format!(
        "#define R {r}\n#define C {c}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out){{\n\
         \x20 for (long r=0;r<R;r++){{ for (long c=0;c<C;c++){{ out[r*C+c] = x[r*C+c] + bias[c]; }} }}\n}}\n"
    )
}

fn rust_biasadd(r: usize, c: usize) -> String {
    format!(
        "const R: usize = {r};\nconst C: usize = {c};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, bias:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ for c in 0..C {{ *out.add(r*C+c) = *x.add(r*C+c) + *bias.add(c); }} }}\n}}\n"
    )
}

/// Int-input **dequant** `out[j] = (q[j] as f32)·scale` over an `[i8]`/`[i32]` array, plus the
/// **per-channel** form `out[i*C+j] = (q[i*C+j] as f32)·scale[j]`. The int buffer is reinterpreted
/// through the shared 3-pointer `(x, y, out)` harness (the pointer element type is irrelevant to the
/// ABI). gcc/rustc vectorize the `i32 -> f32` widen (`vcvtdq2ps`) but leave the **`i8 -> i32 -> f32`**
/// narrowing-load widen scalar, and neither emits non-temporal stores for the streamed `f32` output —
/// the two levers Wukong's `wukong_dequant_f32` pulls (folded 256-bit widen+scale + `vmovntps` past
/// L3). Reported as GB/s (`(in_bytes + 4)·N`, read once + written once); the dequant is exact so the
/// cross-check is bit-exact. Sizes past L3 so the streaming-store advantage is exercised.
fn bench_dequant(cc: &str, dir: &Path) {
    let ext_flags_c = ["-O3", "-march=native", "-ffp-contract=fast", "-shared"];
    let ext_flags_rs = [
        "-Copt-level=3",
        "-Ctarget-cpu=native",
        "--crate-type=cdylib",
    ];

    // --- 1-D dequant: out[j] = (q[j] as f32)·scale, for i32 and i8 inputs -------------------------
    for (ty, in_bytes, is_i8) in [("i32", 4usize, false), ("i8", 1usize, true)] {
        let n = 1usize << 23; // 8M elements: f32 output = 32 MiB ≫ L3, so NT stores engage.
                              // A varied signed int fill (both signs, magnitudes past the i8 boundary for i32).
        let qi32: Vec<i32> = (0..n)
            .map(|i| (((i as i64 * 1103515245 + 12345) >> 9) as i32) % 4096 - 2048)
            .collect();
        let qi8: Vec<i8> = (0..n).map(|i| ((i * 37 + 5) % 251) as i64 as i8).collect();
        let mut out = vec![0.0f32; n];
        let qp = if is_i8 {
            qi8.as_ptr() as *const f32
        } else {
            qi32.as_ptr() as *const f32
        };
        // A REAL, distinct filler for the unused middle pointer. It used to be `out.as_ptr()`, which
        // aliased the output buffer the kernel writes — harmless while the C peer never dereferenced
        // it, but a latent `restrict` violation the moment the peer's parameters carry `__restrict__`
        // (an unused-but-aliasing `restrict` pointer to a modified object). No peer — Wukong, C, C++
        // or Rust — ever indexes this parameter, so one element is enough; the point is only that it
        // is not the output buffer.
        let dummy_buf = [0.0f32; 1];
        let dummy = dummy_buf.as_ptr();
        let bytes = (in_bytes + 4) as f64 * n as f64;
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== dequant-1d ({ty}: out[j] = (q[j] as f32)*scale) N={n} (GB/s, higher is better) ===");
        let built_wuk = build_wukong(&wk_dequant_1d(n, is_i8, false), 3);
        let built_wk_par = build_wukong(&wk_dequant_1d(n, is_i8, true), 3);
        let (built_cm, built_cpp) =
            build_c_cpp("dequant1d", &c_dequant_1d(n, is_i8), dir, cc, &ext_flags_c);
        let built_rm = build_peer(
            "rs",
            &rust_dequant_1d(n, is_i8),
            dir,
            "dequant1d",
            "rustc",
            &ext_flags_rs,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cpp),
            Built::peer(built_rm),
        ];
        let mut timed = abba_min_columns(columns.len(), |i| {
            time_built(&columns[i], &mut out, qp, dummy)
        })
        .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cpp = cpp_checked("dequant1d", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cpp),
            gbps(&rm)
        );
        dequant_check(&wuk, &cm, "C");
        dequant_ratio(&wuk, &wk_par, &cm, &cpp, &rm);
        println!();
    }

    // --- per-channel dequant: out[i*C+j] = (q[i*C+j] as f32)·scale[j] -----------------------------
    for (ty, in_bytes, is_i8) in [("i32", 4usize, false), ("i8", 1usize, true)] {
        let (r, c) = (8192usize, 1024usize); // 8M elements, 32 MiB f32 out ≫ L3
        let rc = r * c;
        let qi32: Vec<i32> = (0..rc)
            .map(|i| (((i as i64 * 22695477 + 1) >> 7) as i32) % 4096 - 2048)
            .collect();
        let qi8: Vec<i8> = (0..rc).map(|i| ((i * 29 + 7) % 251) as i64 as i8).collect();
        let scale: Vec<f32> = (0..c).map(|j| (j % 7) as f32 * 0.003 + 0.002).collect();
        let mut out = vec![0.0f32; rc];
        let qp = if is_i8 {
            qi8.as_ptr() as *const f32
        } else {
            qi32.as_ptr() as *const f32
        };
        let sp = scale.as_ptr();
        let bytes = (in_bytes + 4) as f64 * rc as f64; // scale[C] is negligible
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== dequant-perchan ({ty}: out[i*C+j] = (q as f32)*scale[j]) {r}x{c} (GB/s) ===");
        let built_wuk = build_wukong(&wk_dequant_perchan(r, c, is_i8, false), 3);
        let built_wk_par = build_wukong(&wk_dequant_perchan(r, c, is_i8, true), 3);
        let (built_cm, built_cpp) = build_c_cpp(
            "dequantpc",
            &c_dequant_perchan(r, c, is_i8),
            dir,
            cc,
            &ext_flags_c,
        );
        let built_rm = build_peer(
            "rs",
            &rust_dequant_perchan(r, c, is_i8),
            dir,
            "dequantpc",
            "rustc",
            &ext_flags_rs,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, qp, sp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cpp = cpp_checked("dequantpc", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cpp),
            gbps(&rm)
        );
        dequant_check(&wuk, &cm, "C");
        dequant_ratio(&wuk, &wk_par, &cm, &cpp, &rm);
        println!();
    }
}

/// Exact cross-check: the dequant is integer→f32 exact, so every backend must produce the identical
/// f32 buffer. A mismatch is a real miscompile, not tolerance.
fn dequant_check(wuk: &Option<Measure>, peer: &Option<Measure>, lang: &str) {
    if let (Some(m), Some(p)) = (wuk, peer) {
        if m.out != p.out {
            let at = m
                .out
                .iter()
                .zip(&p.out)
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            println!(
                "  ! dequant output mismatch vs {lang} at [{at}]: {} vs {}",
                m.out[at], p.out[at]
            );
        }
    }
}

/// Report the single-core and `@parallel` C-ratios (clock-invariant; ~3× swing on this laptop).
fn dequant_ratio(
    wuk: &Option<Measure>,
    wk_par: &Option<Measure>,
    cm: &Option<Measure>,
    cpp: &Option<Measure>,
    rm: &Option<Measure>,
) {
    if let (Some(m), Some(c)) = (wuk, cm) {
        let r = c.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong single-core is {:.2}x {} than C (gcc -O3 -march=native)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let (Some(m), Some(cpp)) = (wuk, cpp) {
        let r = cpp.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong single-core is {:.2}x {} than C++ (g++)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let (Some(m), Some(rm)) = (wuk, rm) {
        let r = rm.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong single-core is {:.2}x {} than Rust (rustc -Copt-level=3)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let (Some(mp), Some(c)) = (wk_par, cm) {
        let r = c.ns_per_call / mp.ns_per_call;
        par_standing("single-threaded C", r);
    }
}

fn wk_dequant_1d(n: usize, is_i8: bool, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let ty = if is_i8 { "i8" } else { "i32" };
    format!(
        "module bench\n{attr}fn kbench(q: [{ty}; {n}], u: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for j in 0..{n} {{ out[j] = (q[j] as f32) * 0.0125; }}\n}}\n"
    )
}

fn c_dequant_1d(n: usize, is_i8: bool) -> String {
    let ty = if is_i8 { "signed char" } else { "int" };
    format!(
        "#define N {n}\n\
         __declspec(dllexport) void kbench(const {ty}* __restrict__ q, const float* __restrict__ u, float* __restrict__ out){{\n\
         \x20 (void)u; for (long j=0;j<N;j++){{ out[j] = (float)q[j] * 0.0125f; }}\n}}\n"
    )
}

fn rust_dequant_1d(n: usize, is_i8: bool) -> String {
    let ty = if is_i8 { "i8" } else { "i32" };
    format!(
        "const N: usize = {n};\n#[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(q:*const {ty}, u:*const f32, out:*mut f32) {{\n\
         \x20 for j in 0..N {{ *out.add(j) = (*q.add(j) as f32) * 0.0125; }}\n}}\n"
    )
}

fn wk_dequant_perchan(r: usize, c: usize, is_i8: bool, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let ty = if is_i8 { "i8" } else { "i32" };
    let rc = r * c;
    format!(
        "module bench\n{attr}fn kbench(q: [{ty}; {rc}], scale: [f32; {c}], mut out: [f32; {rc}]) {{\n\
         \x20   for i in 0..{r} {{\n\
         \x20       for j in 0..{c} {{ out[i * {c} + j] = (q[i * {c} + j] as f32) * scale[j]; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_dequant_perchan(r: usize, c: usize, is_i8: bool) -> String {
    let ty = if is_i8 { "signed char" } else { "int" };
    format!(
        "#define R {r}\n#define C {c}\n\
         __declspec(dllexport) void kbench(const {ty}* __restrict__ q, const float* __restrict__ scale, float* __restrict__ out){{\n\
         \x20 for (long i=0;i<R;i++){{ for (long j=0;j<C;j++){{ out[i*C+j] = (float)q[i*C+j] * scale[j]; }} }}\n}}\n"
    )
}

fn rust_dequant_perchan(r: usize, c: usize, is_i8: bool) -> String {
    let ty = if is_i8 { "i8" } else { "i32" };
    format!(
        "const R: usize = {r};\nconst C: usize = {c};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(q:*const {ty}, scale:*const f32, out:*mut f32) {{\n\
         \x20 for i in 0..R {{ for j in 0..C {{ *out.add(i*C+j) = (*q.add(i*C+j) as f32) * *scale.add(j); }} }}\n}}\n"
    )
}

/// Column max / min / **abs-max** `out[j] = max/min_i x[i,j]` (and `max_i |x[i,j]|`, the per-channel
/// symmetric-quant scale) — per-channel statistics / axis-0 pooling, the siblings of `colsum`. Wukong
/// folds them to `wukong_col{max,min,maxabs}_f32`, streaming `x` row-major + 8 columns at a time.
/// Reported as GB/s (`M·N·4`, the matrix read once); the kernels carry an unused middle pointer to
/// share the `(x, _, out)` 3-pointer harness. Every language folds each column i-ascending (`s ⊕ v`
/// mirrors `_mm256_{max,min}_ps`, abs via sign-mask == `fabsf`), so the cross-check is **bit-exact**
/// on finite data.
///
/// PEER SPELLING (corrected 2026-08-04) — same defect as [`bench_colsum`]: the C/Rust peers were
/// written column-outer, `for j { s=x[j]; for i { s = s>v?s:v } out[j]=s; }`, striding `x` by `N`.
/// They are now row-outer (`out[]` seeded from row 0, then `for i { for j { … } }`), which keeps the
/// identical i-ascending fold *and* the identical `s{cmp}v?s:v` expression (so NaN behaviour and
/// bit-exactness are preserved) while streaming `x` sequentially. Measured standalone at `-O3
/// -march=native`, 4096×1024 max, kernels in their own TU: **34.82 ms column-outer vs 0.857 ms
/// row-outer + `restrict`, a 40.6× handicap**, output identical (sum |Δ| = 0).
fn bench_colmax(cc: &str, dir: &Path) {
    // 0 = max, 1 = min, 2 = abs-max (the per-channel symmetric-quant scale).
    for (opc, label, sym) in [
        (0u8, "colmax", "max"),
        (1, "colmin", "min"),
        (2, "colmaxabs", "amax"),
    ] {
        for (m, n) in [(1024usize, 1024usize), (4096, 1024)] {
            let mn = m * n;
            let x: Vec<f32> = (0..mn).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
            let dummy = vec![0.0f32; n];
            let mut out = vec![0.0f32; n];
            let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
            let bytes = mn as f64 * 4.0; // the matrix is read once
            let gbps = |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let desc = if opc == 2 { "|x[i,j]|" } else { "x[i,j]" };
            println!("=== {label} (out[j] = {sym}_i {desc}) {m}x{n} (GB/s, higher is better) ===");
            let built_wuk = build_wukong(&wk_colmax(m, n, false, opc), 3);
            let built_wk_par = build_wukong(&wk_colmax(m, n, true, opc), 3);
            let (built_cm, built_cm_cpp) = build_c_cpp(
                label,
                &c_colmax(m, n, opc),
                dir,
                cc,
                &["-O3", "-march=native", "-shared"],
            );
            let built_rm = build_peer(
                "rs",
                &rust_colmax(m, n, opc),
                dir,
                label,
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            // Reassociation-normalized peer. `fmax`/`fmin` are not associative, so at honest flags
            // gcc keeps a max/min reduction in source order; `-ffast-math` lets it vectorize the
            // fold, which is what Wukong's `_mm256_max_ps` lane fold does.
            let built_cfast = build_peer(
                "c",
                &c_colmax(m, n, opc),
                dir,
                &format!("{}_fast", label),
                cc,
                C_FAST_FLAGS,
            );
            // Every compiled column keeps its slot, including unavailable peers.
            let columns = [
                Built::wk(built_wuk),
                Built::wk(built_wk_par),
                Built::peer(built_cm),
                Built::peer(built_cm_cpp),
                Built::peer(built_rm),
                Built::peer(built_cfast),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wk_par = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked(label, &cm, timed.next().flatten());
            let rm = timed.next().flatten();
            let cfast = timed.next().flatten().filter(|p| {
                wuk.as_ref()
                    .is_some_and(|m| relaxed_peer_ok(label, "C(fast)", m, p))
            });

            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
                gbps(&cm_cpp),
                gbps(&cfast),
                gbps(&rm)
            );
            // Both fold each column i-ascending, so the full-buffer cross-check is bit equality.
            if let (Some(a), Some(c2)) = (&wuk, &cm) {
                if a.out != c2.out {
                    println!("  ! {label} output mismatch vs C");
                }
            }
            if let (Some(a), Some(r2)) = (&wuk, &rm) {
                if a.out != r2.out {
                    println!("  ! {label} output mismatch vs Rust");
                }
            }
            if let (Some(ms), Some(c2)) = (&wuk, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                par_standing("idiomatic single-threaded C", r);
            }
            report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
            report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
            println!();
        }
    }
}

/// Wukong column max/min/abs-max kernel: `for j { let s=⟨x[j]⟩; for i in 1..M { s = fmax/fmin(s, ⟨x[i*N+j]⟩) }; out[j]=s }`
/// the recognizer folds to one `wukong_col{max,min,maxabs}_f32[_parallel]` call (`⟨·⟩` = `abs(·)` for
/// op 2). `y` is unused (the 3-pointer harness). op: 0=max, 1=min, 2=abs-max.
fn wk_colmax(m: usize, n: usize, parallel: bool, op: u8) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let f = if op == 1 { "fmin" } else { "fmax" };
    let (lo, hi) = if op == 2 { ("abs(", ")") } else { ("", "") };
    let mn = m * n;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {mn}], y: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for j in 0..{n} {{\n\
         \x20       let mut s: f32 = {lo}x[j]{hi};\n\
         \x20       for i in 1..{m} {{ s = {f}(s, {lo}x[i * {n} + j]{hi}); }}\n\
         \x20       out[j] = s;\n\
         \x20   }}\n}}\n"
    )
}

fn c_colmax(m: usize, n: usize, op: u8) -> String {
    let cmp = if op == 1 { "<" } else { ">" };
    let (lo, hi) = if op == 2 { ("fabsf(", ")") } else { ("", "") };
    format!(
        "#include <math.h>\n#define M {m}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long j=0;j<N;j++) out[j]={lo}x[j]{hi};\n\
         \x20 for (long i=1;i<M;i++) for (long j=0;j<N;j++){{ float s=out[j], v={lo}x[i*N+j]{hi}; out[j] = s{cmp}v?s:v; }}\n}}\n"
    )
}

fn rust_colmax(m: usize, n: usize, op: u8) -> String {
    let cmp = if op == 1 { "<" } else { ">" };
    let (lo, hi) = if op == 2 { ("(", ").abs()") } else { ("", "") };
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n\
         #[inline(always)]\nfn kbody(xs: &[f32], os: &mut [f32]) {{\n\
         \x20 for (o, &v) in os.iter_mut().zip(&xs[0..N]) {{ *o = {lo}v{hi}; }}\n\
         \x20 for i in 1..M {{ let row = &xs[i*N..i*N+N];\n\
         \x20   for (o, &r) in os.iter_mut().zip(row) {{ let s=*o; let v={lo}r{hi}; *o = if s{cmp}v {{s}} else {{v}}; }} }}\n}}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 kbody(core::slice::from_raw_parts(x, M*N), core::slice::from_raw_parts_mut(out, N))\n}}\n"
    )
}

/// Wukong per-row argmax/argmin returning an **i32 index**: `for r { let bv=x[r*C]; let bi=0; for j in
/// 1..C { if x[r*C+j] >|< bv { bv=x[r*C+j]; bi=j } }; out[r]=bi }` — the recognizer folds it to one
/// `wukong_rowarg{max,min}_i32[_parallel]` call. `out` is an i32 buffer (the harness passes its f32
/// pointer; a pointer is a pointer). `y` is the unused middle of the 3-pointer harness. `is_max` picks
/// argmax (`>`) vs argmin (`<`).
fn wk_rowarg(rows: usize, cols: usize, is_max: bool, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [i32; {rows}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut bv: f32 = x[r * {cols}];\n\
         \x20       let mut bi: i32 = 0;\n\
         \x20       for j in 1..{cols} {{ if x[r * {cols} + j] {cmp} bv {{ bv = x[r * {cols} + j]; bi = j; }} }}\n\
         \x20       out[r] = bi;\n\
         \x20   }}\n}}\n"
    )
}

fn c_rowarg(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, int* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long r=0;r<R;r++){{ float bv=x[r*C]; int bi=0;\n\
         \x20   for (long j=1;j<C;j++){{ float v=x[r*C+j]; if (v {cmp} bv){{ bv=v; bi=j; }} }}\n\
         \x20   out[r]=bi; }} }}\n"
    )
}

fn rust_rowarg(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut i32) {{\n\
         \x20 for r in 0..R {{ let mut bv=*x.add(r*C); let mut bi=0i32;\n\
         \x20   for j in 1..C {{ let v=*x.add(r*C+j); if v {cmp} bv {{ bv=v; bi=j as i32; }} }}\n\
         \x20   *out.add(r)=bi; }} }}\n"
    )
}

/// Per-row argmax/argmin (classification-head / greedy-decode top-1) returning the index. The
/// (value,index) bookkeeping defeats gcc/rustc auto-vectorization (verified: scalar inner loop), while
/// Wukong's AVX2 kernel tracks 8 (value,index) lanes via blend. Output is an i32 index buffer, so the
/// cross-check reinterprets the f32 harness slots as i32 and compares **exactly** (a selection, not a
/// float reduction — no tolerance). GB/s = `R·C·4` (the matrix read once); latency-bound, so the ratio
/// vs naive C is the reported figure.
fn bench_rowarg(cc: &str, dir: &Path) {
    for (is_max, label) in [(true, "rowargmax"), (false, "rowargmin")] {
        for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 31 + 7) % 101) as f32 * 0.5 - 25.0)
                .collect();
            let dummy = vec![0.0f32; n];
            let mut out = vec![0.0f32; rows];
            let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
            let bytes = n as f64 * 4.0; // the matrix is read once
            let gbps = |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let sym = if is_max { "argmax" } else { "argmin" };
            println!(
                "=== {label} (out[r] = {sym}_j x[r,j]) {rows}x{cols} (GB/s, higher is better) ==="
            );
            let built_wuk = build_wukong(&wk_rowarg(rows, cols, is_max, false), 3);
            let built_wk_par = build_wukong(&wk_rowarg(rows, cols, is_max, true), 3);
            let (built_cm, built_cm_cpp) = build_c_cpp(
                label,
                &c_rowarg(rows, cols, is_max),
                dir,
                cc,
                &["-O3", "-march=native", "-shared"],
            );
            let built_rm = build_peer(
                "rs",
                &rust_rowarg(rows, cols, is_max),
                dir,
                label,
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            // Reassociation-normalized peer (exact-index-checked, see `fast_indices_checked`).
            let built_cfast = build_peer(
                "c",
                &c_rowarg(rows, cols, is_max),
                dir,
                &format!("{}_fast", label),
                cc,
                C_FAST_FLAGS,
            );
            // Every compiled column keeps its slot, including unavailable peers.
            let columns = [
                Built::wk(built_wuk),
                Built::wk(built_wk_par),
                Built::peer(built_cm),
                Built::peer(built_cm_cpp),
                Built::peer(built_rm),
                Built::peer(built_cfast),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wk_par = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked(label, &cm, timed.next().flatten());
            let rm = timed.next().flatten();
            let cfast = timed
                .next()
                .flatten()
                .and_then(|p| fast_indices_checked(label, wuk.as_ref()?, p));

            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
                gbps(&cm_cpp),
                gbps(&cfast),
                gbps(&rm)
            );
            // Output is an i32 index buffer — reinterpret the f32 harness slots as i32 and compare exactly
            // (no tolerance: a per-row arg-selection is deterministic, lowest index winning on a tie).
            let as_i32 = |v: &[f32]| v.iter().map(|x| x.to_bits() as i32).collect::<Vec<i32>>();
            if let (Some(a), Some(c2)) = (&wuk, &cm) {
                if as_i32(&a.out) != as_i32(&c2.out) {
                    println!("  ! {label} index mismatch vs C");
                }
            }
            if let (Some(a), Some(r2)) = (&wuk, &rm) {
                if as_i32(&a.out) != as_i32(&r2.out) {
                    println!("  ! {label} index mismatch vs Rust");
                }
            }
            if let (Some(ms), Some(c2)) = (&wuk, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                par_standing("idiomatic single-threaded C", r);
            }
            report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
            report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
            println!();
        }
    }
}

/// Wukong per-COLUMN argmax/argmin returning an i32 ROW index: `for j { let bv=x[j]; let bi=0; for i in
/// 1..R { if x[i*C+j] >|< bv { bv=x[i*C+j]; bi=i } }; out[j]=bi }` — folds to one
/// `wukong_colarg{max,min}_i32[_parallel]`. The STRIDED column-outer (value,index) scan is what gcc
/// leaves scalar. `out` is a `cols`-long i32 buffer; `y` is the unused middle of the 3-pointer harness.
fn wk_colarg(rows: usize, cols: usize, is_max: bool, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [i32; {cols}]) {{\n\
         \x20   for j in 0..{cols} {{\n\
         \x20       let mut bv: f32 = x[j];\n\
         \x20       let mut bi: i32 = 0;\n\
         \x20       for i in 1..{rows} {{ if x[i * {cols} + j] {cmp} bv {{ bv = x[i * {cols} + j]; bi = i; }} }}\n\
         \x20       out[j] = bi;\n\
         \x20   }}\n}}\n"
    )
}

/// The C column-argmax peer, written **row-outer** with a `C`-long running-best scratch vector — the
/// spelling any performance-aware C programmer uses for an axis-0 arg-reduction (and what NumPy's
/// `argmax(axis=0)` does internally), rather than the column-outer scan that reads `x` with stride
/// `C`. The comparison is `v {cmp} bv[j]` exactly as before, so the first-extremum tie-break and the
/// resulting index buffer are unchanged; only the traversal order of `x` moves. Measured standalone
/// at `-O3 -march=native`, 4096×1024 argmax, kernels in their own TU: **9.87 ms column-outer vs
/// 1.569 ms row-outer + `restrict`, a 6.3× handicap**, with identical output on every column.
fn c_colarg(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, int* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 float bv[C];\n\
         \x20 for (long j=0;j<C;j++){{ bv[j]=x[j]; out[j]=0; }}\n\
         \x20 for (long i=1;i<R;i++) for (long j=0;j<C;j++){{ float v=x[i*C+j]; if (v {cmp} bv[j]){{ bv[j]=v; out[j]=(int)i; }} }}\n}}\n"
    )
}

/// The Rust column-arg peer. Two fixes over the first row-outer rewrite, both of which cost the peer
/// time its C twin never paid:
///   * the running-best scratch was a **heap `vec![0.0f32; C]` allocated inside the timed kernel**,
///     and zero-filled before being overwritten by the seed loop. `c_colarg`'s twin is `float bv[C]`,
///     a stack array with no allocator call and no pre-fill. It is a fixed-size stack array here now.
///   * the slices were built inside `kbench`, which grants no `noalias` (see [`rust_kernel`]).
fn rust_colarg(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n\
         #[inline(always)]\nfn kbody(xs: &[f32], os: &mut [i32]) {{\n\
         \x20 let mut bv = [0.0f32; C];\n\
         \x20 for j in 0..C {{ bv[j]=xs[j]; os[j]=0; }}\n\
         \x20 for i in 1..R {{ let row = &xs[i*C..i*C+C];\n\
         \x20   for j in 0..C {{ let v=row[j]; if v {cmp} bv[j] {{ bv[j]=v; os[j]=i as i32; }} }} }}\n}}\n\
         #[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut i32) {{\n\
         \x20 kbody(core::slice::from_raw_parts(x, R*C), core::slice::from_raw_parts_mut(out, C))\n}}\n"
    )
}

/// Per-column argmax/argmin (axis-0 top-1) returning the ROW index. Wukong streams row-major
/// tracking 8 column lanes via blend; the C/Rust peers do the same traversal over a `C`-long
/// running-best vector (see [`c_colarg`] — they used to scan column-outer, which is what the ratio
/// was really measuring). Output is a `cols`-long i32 buffer; the cross-check reinterprets the f32
/// harness slots as i32 and compares EXACTLY. GB/s = `R·C·4` (matrix read once).
fn bench_colarg(cc: &str, dir: &Path) {
    for (is_max, label) in [(true, "colargmax"), (false, "colargmin")] {
        for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 + 11) % 103) as f32 * 0.5 - 25.0)
                .collect();
            let dummy = vec![0.0f32; n];
            let mut out = vec![0.0f32; cols];
            let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
            let bytes = n as f64 * 4.0; // the matrix is read once
            let gbps = |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let sym = if is_max { "argmax" } else { "argmin" };
            println!(
                "=== {label} (out[j] = {sym}_i x[i,j]) {rows}x{cols} (GB/s, higher is better) ==="
            );
            let built_wuk = build_wukong(&wk_colarg(rows, cols, is_max, false), 3);
            let built_wk_par = build_wukong(&wk_colarg(rows, cols, is_max, true), 3);
            let (built_cm, built_cm_cpp) = build_c_cpp(
                label,
                &c_colarg(rows, cols, is_max),
                dir,
                cc,
                &["-O3", "-march=native", "-shared"],
            );
            let built_rm = build_peer(
                "rs",
                &rust_colarg(rows, cols, is_max),
                dir,
                label,
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            // Reassociation-normalized peer (exact-index-checked, see `fast_indices_checked`).
            let built_cfast = build_peer(
                "c",
                &c_colarg(rows, cols, is_max),
                dir,
                &format!("{}_fast", label),
                cc,
                C_FAST_FLAGS,
            );
            // Every compiled column keeps its slot, including unavailable peers.
            let columns = [
                Built::wk(built_wuk),
                Built::wk(built_wk_par),
                Built::peer(built_cm),
                Built::peer(built_cm_cpp),
                Built::peer(built_rm),
                Built::peer(built_cfast),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wk_par = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked(label, &cm, timed.next().flatten());
            let rm = timed.next().flatten();
            let cfast = timed
                .next()
                .flatten()
                .and_then(|p| fast_indices_checked(label, wuk.as_ref()?, p));

            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
                gbps(&cm_cpp),
                gbps(&cfast),
                gbps(&rm)
            );
            let as_i32 = |v: &[f32]| v.iter().map(|x| x.to_bits() as i32).collect::<Vec<i32>>();
            if let (Some(a), Some(c2)) = (&wuk, &cm) {
                if as_i32(&a.out) != as_i32(&c2.out) {
                    println!("  ! {label} index mismatch vs C");
                }
            }
            if let (Some(a), Some(r2)) = (&wuk, &rm) {
                if as_i32(&a.out) != as_i32(&r2.out) {
                    println!("  ! {label} index mismatch vs Rust");
                }
            }
            if let (Some(ms), Some(c2)) = (&wuk, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                par_standing("idiomatic single-threaded C", r);
            }
            report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
            report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
            println!();
        }
    }
}

/// Wukong per-row inclusive prefix sum (cumsum): `for r { let acc=0; for i in 0..C { acc=acc+x[r*C+i];
/// out[r*C+i]=acc } }` — folds to one `wukong_cumsum_f32[_parallel]`. The loop-carried `acc` recurrence
/// is what gcc/rustc keep scalar; the SIMD Hillis-Steele scan vectorizes it. `y` is the unused middle.
fn wk_cumsum(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut acc: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ acc = acc + x[r * {cols} + i]; out[r * {cols} + i] = acc; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_cumsum(rows: usize, cols: usize) -> String {
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long r=0;r<R;r++){{ float acc=0.0f; for (long i=0;i<C;i++){{ acc+=x[r*C+i]; out[r*C+i]=acc; }} }} }}\n"
    )
}

fn rust_cumsum(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ let mut acc=0.0f32; for i in 0..C {{ acc += *x.add(r*C+i); *out.add(r*C+i)=acc; }} }} }}\n"
    )
}

fn wk_lrscan(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n}], b: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut h: f32 = 0.0;\n\
         \x20       for t in 0..{cols} {{ h = a[r * {cols} + t] * h + b[r * {cols} + t]; out[r * {cols} + t] = h; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_lrscan(rows: usize, cols: usize) -> String {
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){{\n\
         \x20 for (long r=0;r<R;r++){{ float h=0.0f; for (long t=0;t<C;t++){{ h = a[r*C+t]*h + b[r*C+t]; out[r*C+t]=h; }} }} }}\n"
    )
}

fn rust_lrscan(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ let mut h=0.0f32; for t in 0..C {{ h = *a.add(r*C+t)*h + *b.add(r*C+t); *out.add(r*C+t)=h; }} }} }}\n"
    )
}

/// First-order linear-recurrence / selective scan `out[r,t] = a[r,t]·h_{t-1} + b[r,t]` (the SSM/Mamba/S4
/// step, also EMA). The carried `h` is a true loop-carried dependency, so gcc/rustc cannot auto-vectorize
/// the inner time loop (like cumsum) and emit **one serial mul+add chain** per row — latency-bound, a few
/// GB/s. Wukong's kernel **interleaves 4 independent rows**, keeping four chains in flight to fill the
/// idle ports: a genuine single-core WIN (~1.6–1.9×).
///
/// PEER-FAIRNESS NOTE (corrected 2026-08-06). This used to justify that win by saying "gcc/rustc may
/// not legally re-order an f32 recurrence across rows". That is **false**: the rows are independent,
/// so unroll-and-jam across `r` is perfectly legal and changes no value — gcc and rustc simply do not
/// do it here. The honest statement is that the peer is the *idiomatic* spelling and neither
/// toolchain finds the row interleave on its own, not that it is forbidden from finding it. What IS
/// illegal without `-ffast-math` is re-ordering *within* a row, which is why the peer's inner time
/// loop stays one serial chain. `@parallel` maps independent row chunks across cores on top (~5–8×). GB/s =
/// `3·R·C·4` (read `a` + read `b` + write `out`). The recurrence is sequential within a row (no
/// reassociation) and the kernel does plain mul+add (two roundings) where gcc may fuse to one `fma`, so
/// the cross-check is a ~1-ULP magnitude-normalized tolerance like cumsum.
fn bench_lrscan(cc: &str, dir: &Path) {
    for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
        let n = rows * cols;
        // `a` (the gate) bounded in (−1, 1) so the recurrence is a contraction and `h` stays finite;
        // `b` (the input drive) small. Deterministic, no RNG.
        let a: Vec<f32> = (0..n)
            .map(|i| ((i * 13 + 5) % 19) as f32 * 0.1 - 0.9)
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.2 - 1.0)
            .collect();
        let mut out = vec![0.0f32; n];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let bytes = n as f64 * 4.0 * 3.0; // read a + read b + write out
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== lrscan (out[r,t] = a[r,t]·h + b[r,t], SSM/Mamba selective scan) {rows}x{cols} (GB/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_lrscan(rows, cols, false), 3);
        let built_wk_par = build_wukong(&wk_lrscan(rows, cols, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "lrscan",
            &c_lrscan(rows, cols),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_lrscan(rows, cols),
            dir,
            "lrscan",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, ap, bp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("lrscan", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&rm)
        );
        if let (Some(a2), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let maxerr = a2
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! lrscan mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C (4-row-interleaved ILP vs C's single serial chain)",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        println!();
    }
}

fn wk_cumprod(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut p: f32 = 1.0;\n\
         \x20       for i in 0..{cols} {{ p = p * x[r * {cols} + i]; out[r * {cols} + i] = p; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_cumprod(rows: usize, cols: usize) -> String {
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long r=0;r<R;r++){{ float p=1.0f; for (long i=0;i<C;i++){{ p*=x[r*C+i]; out[r*C+i]=p; }} }} }}\n"
    )
}

fn rust_cumprod(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ let mut p=1.0f32; for i in 0..C {{ p *= *x.add(r*C+i); *out.add(r*C+i)=p; }} }} }}\n"
    )
}

/// Per-row inclusive prefix **product** (cumprod / scan). Loop-carried `out[i]=out[i-1]*x[i]`, so gcc
/// `-O3 -march=native` / rustc keep it SCALAR (one serial `mulss` chain per row, latency-bound). Wukong's
/// `wukong_cumprod_f32` interleaves **4 independent rows** for ILP (the same lever as `lrscan`). A bare
/// product is *not* fused, so each row folds strictly left-to-right — **bit-exact** vs the scalar C/Rust
/// nest (no reassociation, unlike the prefix sum), though the cross-check stays magnitude-normalized since
/// a running product spans a wide dynamic range. GB/s = `R·C·4·2` (read x + write out); `@parallel` maps
/// independent rows across cores, bit-equal to serial. Inputs oscillate around 1.0 so the product stays
/// finite and `O(1)`.
fn bench_cumprod(cc: &str, dir: &Path) {
    for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
        let n = rows * cols;
        // Oscillate around 1.0 ({0.96,0.98,1.0,1.02,1.04}) so the running product is a bounded random walk.
        let x: Vec<f32> = (0..n)
            .map(|i| 0.96 + 0.02 * ((i * 7 + 3) % 5) as f32)
            .collect();
        let dummy = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
        let bytes = n as f64 * 4.0 * 2.0; // read x + write out
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== cumprod (out[r,i] = Prod_k<=i x[r,k]) {rows}x{cols} (GB/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_cumprod(rows, cols, false), 3);
        let built_wk_par = build_wukong(&wk_cumprod(rows, cols, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "cumprod",
            &c_cumprod(rows, cols),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_cumprod(rows, cols),
            dir,
            "cumprod",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("cumprod", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&rm)
        );
        if let (Some(a), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let maxerr = a
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! cumprod mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C (4-row-interleaved ILP vs C's serial chain)",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        println!();
    }
}

/// Per-row inclusive prefix sum (cumsum / scan). gcc/rustc keep the loop-carried `out[i]=out[i-1]+x[i]`
/// recurrence SCALAR (a prefix sum does not auto-vectorize); Wukong folds it to the SIMD Hillis-Steele
/// scan kernel. GB/s = `R·C·4·2` (read x + write out). The in-lane tree reassociates the float sum, so
/// the cross-check is a tight magnitude-normalized tolerance (like the reductions), not exact.
fn bench_cumsum(cc: &str, dir: &Path) {
    for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
        let n = rows * cols;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 31 + 7) % 101) as f32 * 0.01 - 0.5)
            .collect();
        let dummy = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
        let bytes = n as f64 * 4.0 * 2.0; // read x + write out
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== cumsum (out[r,i] = Sum_k<=i x[r,k]) {rows}x{cols} (GB/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_cumsum(rows, cols, false), 3);
        let built_wk_par = build_wukong(&wk_cumsum(rows, cols, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "cumsum",
            &c_cumsum(rows, cols),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_cumsum(rows, cols),
            dir,
            "cumsum",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer. Wukong's Hillis-Steele in-lane tree scan REASSOCIATES the
        // prefix sum (that is exactly why this bench's cross-check is a tolerance and not bit
        // equality), so the plain-flags C column — which must keep `out[i]=out[i-1]+x[i]` in source
        // order — is not the like-for-like peer. This is the rule the file applies to `dot`/`ssd`;
        // `cumsum` is the one scan that also needs it (cumprod / cummax / cummin / lrscan stay
        // bit-exact in Wukong, so their plain-flags column already IS like-for-like).
        let built_cfast = build_peer(
            "c",
            &c_cumsum(rows, cols),
            dir,
            &format!("{}_fast", "cumsum"),
            cc,
            C_FAST_FLAGS,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
            Built::peer(built_cfast),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("cumsum", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("cumsum", "C(fast)", m, p))
        });

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&cfast),
            gbps(&rm)
        );
        // The in-lane tree scan reassociates → a **magnitude-normalized** tolerance (max|Δ| over the
        // largest prefix sum), not a pointwise relative one: a mean-zero input makes the prefix sum a
        // random walk that crosses zero, where a pointwise ratio divides by ~0. Same basis as the norm
        // / reduction cross-checks.
        if let (Some(a), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let maxerr = a
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! cumsum mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// Wukong per-row cumulative max/min: `for r { let m=x[r*C]; for i { m=fmax(m,x[r*C+i]); out[r*C+i]=m } }`
/// — folds to one `wukong_cum{max,min}_f32[_parallel]`. The loop-carried fmax/fmin recurrence is what
/// gcc/rustc keep scalar; the SIMD in-lane max/min scan vectorizes it. `y` unused.
fn wk_cumminmax(rows: usize, cols: usize, is_max: bool, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    let f = if is_max { "fmax" } else { "fmin" };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut m: f32 = x[r * {cols}];\n\
         \x20       for i in 0..{cols} {{ m = {f}(m, x[r * {cols} + i]); out[r * {cols} + i] = m; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_cumminmax(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long r=0;r<R;r++){{ float m=x[r*C]; for (long i=0;i<C;i++){{ float v=x[r*C+i]; if (v {cmp} m) m=v; out[r*C+i]=m; }} }} }}\n"
    )
}

fn rust_cumminmax(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ let mut m=*x.add(r*C); for i in 0..C {{ let v=*x.add(r*C+i); if v {cmp} m {{ m=v; }} *out.add(r*C+i)=m; }} }} }}\n"
    )
}

/// Per-row cumulative max / min (running-extreme scan). gcc/rustc keep the loop-carried fmax/fmin
/// recurrence SCALAR; Wukong folds it to the SIMD in-lane max/min scan kernel. GB/s = `R·C·4·2`. max/min
/// select an input value (no float arithmetic, no reassociation), so the cross-check is **bit-exact**.
fn bench_cumminmax(cc: &str, dir: &Path) {
    for (is_max, label) in [(true, "cummax"), (false, "cummin")] {
        for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
            let n = rows * cols;
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 47 + 13) % 101) as f32 * 0.5 - 25.0)
                .collect();
            let dummy = vec![0.0f32; n];
            let mut out = vec![0.0f32; n];
            let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
            let bytes = n as f64 * 4.0 * 2.0; // read x + write out
            let gbps = |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let sym = if is_max { "max" } else { "min" };
            println!("=== {label} (out[r,i] = {sym}_k<=i x[r,k]) {rows}x{cols} (GB/s, higher is better) ===");
            let built_wuk = build_wukong(&wk_cumminmax(rows, cols, is_max, false), 3);
            let built_wk_par = build_wukong(&wk_cumminmax(rows, cols, is_max, true), 3);
            let (built_cm, built_cm_cpp) = build_c_cpp(
                label,
                &c_cumminmax(rows, cols, is_max),
                dir,
                cc,
                &["-O3", "-march=native", "-shared"],
            );
            let built_rm = build_peer(
                "rs",
                &rust_cumminmax(rows, cols, is_max),
                dir,
                label,
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            // Every compiled column keeps its slot, including unavailable peers.
            let columns = [
                Built::wk(built_wuk),
                Built::wk(built_wk_par),
                Built::peer(built_cm),
                Built::peer(built_cm_cpp),
                Built::peer(built_rm),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wk_par = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked(label, &cm, timed.next().flatten());
            let rm = timed.next().flatten();

            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
                gbps(&cm_cpp),
                gbps(&rm)
            );
            // max/min select a value → bit-exact cross-check (no reassociation tolerance).
            if let (Some(a), Some(c2)) = (&wuk, &cm) {
                if a.out != c2.out {
                    println!("  ! {label} output mismatch vs C");
                }
            }
            if let (Some(a), Some(r2)) = (&wuk, &rm) {
                if a.out != r2.out {
                    println!("  ! {label} output mismatch vs Rust");
                }
            }
            if let (Some(ms), Some(c2)) = (&wuk, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                par_standing("idiomatic single-threaded C", r);
            }
            report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
            println!();
        }
    }
}

/// Column **statistics** `out[j] = mean/sumsq/L2/RMS_i x[i,j]` — the per-channel BatchNorm mean, 2nd
/// moment / energy, column L2 norm, and per-feature RMS. Wukong folds each nest to
/// `wukong_col{mean,sumsq,l2,rms}_f32[_parallel]` (row-major streaming + a per-column finalize).
/// Reported as GB/s (`M·N·4`, the matrix read once), reusing the `(x, _, out)` 3-pointer harness via
/// the unused middle. Each language folds its column i-ascending and `/M`/`sqrt` are correctly
/// rounded, so the full-buffer cross-check is **bit-exact**.
///
/// PEER SPELLING (corrected 2026-08-04) — the third instance of the [`bench_colsum`] defect. The
/// C/Rust peers were column-outer; they are now row-outer (accumulate into `out[]` row by row, then
/// one finalize pass), same fold order, same result. Measured standalone at `-O3 -march=native`,
/// 4096×1024 column mean, kernels in their own TU: **33.03 ms column-outer vs 1.238 ms row-outer +
/// `restrict`, a 26.7× handicap**.
fn bench_colstat(cc: &str, dir: &Path) {
    // 0 = mean, 1 = sumsq (energy), 2 = L2, 3 = RMS.
    for (opc, label, sym) in [
        (0u8, "colmean", "mean"),
        (1, "colsumsq", "Σ"),
        (2, "coll2", "L2"),
        (3, "colrms", "rms"),
    ] {
        for (m, n) in [(1024usize, 1024usize), (4096, 1024)] {
            let mn = m * n;
            let x: Vec<f32> = (0..mn).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
            let dummy = vec![0.0f32; n];
            let mut out = vec![0.0f32; n];
            let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
            let bytes = mn as f64 * 4.0; // the matrix is read once
            let gbps = |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let desc = if opc == 0 { "x[i,j]" } else { "x[i,j]²" };
            println!("=== {label} (out[j] = {sym}_i {desc}) {m}x{n} (GB/s, higher is better) ===");
            // POSITION-BALANCED TIMING (2026-09-13) - see `abba_min_columns` and the note in
            // `bench_colsum`. Build every column first, then time them all in one rotation.
            let wk1 = build_wukong(&wk_colstat(m, n, false, opc), 3);
            let wkp = build_wukong(&wk_colstat(m, n, true, opc), 3);
            let (cl, pl) = build_c_cpp(
                label,
                &c_colstat(m, n, opc),
                dir,
                cc,
                &["-O3", "-march=native", "-shared"],
            );
            // Reassociation-normalized peer: -ffast-math lets gcc reassociate the strided fold.
            let fl = build_peer(
                "c",
                &c_colstat(m, n, opc),
                dir,
                &format!("{label}_fast"),
                cc,
                C_FAST_FLAGS,
            );
            let rl = build_peer(
                "rs",
                &rust_colstat(m, n, opc),
                dir,
                label,
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            let cols = [
                Built::wk(wk1),
                Built::wk(wkp),
                Built::peer(cl),
                Built::peer(pl),
                Built::peer(fl),
                Built::peer(rl),
            ];
            let mut timed =
                abba_min_columns(cols.len(), |i| time_built(&cols[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wk_par = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked(label, &cm, timed.next().flatten());
            let cfast = timed.next().flatten().filter(|p| {
                wuk.as_ref()
                    .is_some_and(|mm| relaxed_peer_ok(label, "C(fast)", mm, p))
            });
            let rm = timed.next().flatten();
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
                gbps(&cm_cpp),
                gbps(&cfast),
                gbps(&rm)
            );
            // Both fold each column i-ascending and finalize with correctly-rounded /M and sqrt, so the
            // full-buffer cross-check is bit equality.
            if let (Some(a), Some(c2)) = (&wuk, &cm) {
                if a.out != c2.out {
                    println!("  ! {label} output mismatch vs C");
                }
            }
            if let (Some(a), Some(r2)) = (&wuk, &rm) {
                if a.out != r2.out {
                    println!("  ! {label} output mismatch vs Rust");
                }
            }
            if let (Some(ms), Some(c2)) = (&wuk, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                par_standing("idiomatic single-threaded C", r);
            }
            report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
            report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
            println!();
        }
    }
}

/// Wukong column-statistics kernel: `for j { let s=0; for i in 0..M { s = s + ⟨x[i*N+j]⟩ }; out[j]=fin(s) }`
/// the recognizer folds to one `wukong_col{mean,sumsq,l2,rms}_f32[_parallel]` call. op: 0=mean
/// (`s/M`), 1=sumsq (`s` over `x²`), 2=L2 (`sqrt(s)` over `x²`), 3=RMS (`sqrt(s/M)` over `x²`).
fn wk_colstat(m: usize, n: usize, parallel: bool, op: u8) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mn = m * n;
    let prod = format!("x[i * {n} + j]");
    let (fold, fin) = match op {
        0 => (format!("s = s + {prod};"), format!("s / {m}.0")),
        1 => (format!("s = s + {prod} * {prod};"), "s".to_string()),
        2 => (format!("s = s + {prod} * {prod};"), "sqrt(s)".to_string()),
        _ => (
            format!("s = s + {prod} * {prod};"),
            format!("sqrt(s / {m}.0)"),
        ),
    };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {mn}], y: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for j in 0..{n} {{\n\
         \x20       let mut s: f32 = 0.0;\n\
         \x20       for i in 0..{m} {{ {fold} }}\n\
         \x20       out[j] = {fin};\n\
         \x20   }}\n}}\n"
    )
}

fn c_colstat(m: usize, n: usize, op: u8) -> String {
    // Row-outer: accumulate the column folds into `out[]` sweeping `x` sequentially, then finalize.
    let (fold, fin) = match op {
        0 => ("out[j] += x[i*N+j];", "s / (float)M"),
        1 => ("{ float v=x[i*N+j]; out[j] += v*v; }", "s"),
        2 => ("{ float v=x[i*N+j]; out[j] += v*v; }", "sqrtf(s)"),
        _ => (
            "{ float v=x[i*N+j]; out[j] += v*v; }",
            "sqrtf(s / (float)M)",
        ),
    };
    format!(
        "#include <math.h>\n#define M {m}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out){{\n\
         \x20 (void)y;\n\
         \x20 for (long j=0;j<N;j++) out[j]=0.0f;\n\
         \x20 for (long i=0;i<M;i++) for (long j=0;j<N;j++) {fold}\n\
         \x20 for (long j=0;j<N;j++){{ float s=out[j]; out[j]={fin}; }}\n}}\n"
    )
}

fn rust_colstat(m: usize, n: usize, op: u8) -> String {
    let (fold, fin) = match op {
        0 => ("*o += r;", "s / M as f32"),
        1 => ("*o += r*r;", "s"),
        2 => ("*o += r*r;", "s.sqrt()"),
        _ => ("*o += r*r;", "(s / M as f32).sqrt()"),
    };
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n\
         #[inline(always)]\nfn kbody(xs: &[f32], os: &mut [f32]) {{\n\
         \x20 for o in os.iter_mut() {{ *o = 0.0; }}\n\
         \x20 for i in 0..M {{ let row = &xs[i*N..i*N+N]; for (o, &r) in os.iter_mut().zip(row) {{ {fold} }} }}\n\
         \x20 for o in os.iter_mut() {{ let s = *o; *o = {fin}; }}\n}}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 kbody(core::slice::from_raw_parts(x, M*N), core::slice::from_raw_parts_mut(out, N))\n}}\n"
    )
}

/// Softmax backward `dx[r,i] = y[r,i]·(dy[r,i] − Σ_j y[r,j]·dy[r,j])` — the attention/classification
/// training gradient. Each row's dot `Σ y·dy` is a reduction gcc/rustc keep **scalar** (verified: they
/// load/multiply `y·dy` wide but the accumulation is a serial `vaddss` chain — no `ymm` accumulator,
/// the float sum is not reassociated), then a (vectorized) elementwise `y·(dy − s)`. Wukong folds the
/// `[R,C]` nest to one `wukong_softmax_bwd_f32[_parallel]` call: the dot uses 8 independent lane
/// accumulators (no dependency chain), then the apply runs 8-wide. Uses **all three** harness pointers
/// (`y, dy, dx`). Reported as GB/s (`3·R·C·4`, the two reads + one write). The dot reassociates, so the
/// cross-check is a **tight relative tolerance** (`< 1e-3`), like the norm benches.
fn bench_softmax_bwd(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 512)] {
        let n = r * c;
        // A plausible softmax output (positive, rows ~normalized) and a small grad — well-conditioned so
        // the reassociated dot stays close to the sequential one.
        let y: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 + 1.0) / 200.0).collect();
        let dy: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let mut dx = vec![0.0f32; n];
        let (yp, dyp) = (y.as_ptr(), dy.as_ptr());
        let bytes = 3.0 * n as f64 * 4.0; // y read + dy read + dx write
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== softmax_bwd (dx = y·(dy − Σ y·dy)) {r}x{c} (GB/s, higher is better) ===");
        // Build every column before the shared position-balanced rotation.
        let wk1 = build_wukong(&wk_softmax_bwd(r, c, false), 3);
        let wkp = build_wukong(&wk_softmax_bwd(r, c, true), 3);
        let (cl, pl) = build_c_cpp(
            "softmax_bwd",
            &c_softmax_bwd(r, c),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let fl = build_peer(
            "c",
            &c_softmax_bwd(r, c),
            dir,
            "softmax_bwd_fast",
            cc,
            C_FAST_FLAGS,
        );
        let rl = build_peer(
            "rs",
            &rust_softmax_bwd(r, c),
            dir,
            "softmax_bwd",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let columns = [
            Built::wk(wk1),
            Built::wk(wkp),
            Built::peer(cl),
            Built::peer(pl),
            Built::peer(fl),
            Built::peer(rl),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut dx, yp, dyp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("softmax_bwd", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("softmax_bwd", "C(fast)", m, p))
        });
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&cfast),
            gbps(&rm)
        );
        // The dot reassociates (lane accumulators vs C's serial chain), so check a tolerance. Softmax
        // backward sums to ~0 per row, so individual dx are catastrophic-cancellation zeros where a
        // *per-element* relative error is meaningless — normalize the max abs error by the max output
        // magnitude instead (the honest "is the whole vector close" metric).
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! softmax_bwd mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r2);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// Wukong softmax-backward: the batched nest the recognizer folds to one `wukong_softmax_bwd_f32
/// [_parallel]` call. The harness's `(x, y, out)` pointers carry `(y, dy, dx)`.
fn wk_softmax_bwd(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(y: [f32; {n}], dy: [f32; {n}], mut dx: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut s: f32 = 0.0;\n\
         \x20       for j in 0..{cols} {{ s = s + y[r * {cols} + j] * dy[r * {cols} + j]; }}\n\
         \x20       for i in 0..{cols} {{ dx[r * {cols} + i] = y[r * {cols} + i] * (dy[r * {cols} + i] - s); }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_softmax_bwd(rows: usize, cols: usize) -> String {
    format!(
        "#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ y, const float* __restrict__ dy, float* __restrict__ dx){{\n\
         \x20 for (long r=0;r<R;r++){{ float s=0.0f; for (long j=0;j<C;j++) s += y[r*C+j]*dy[r*C+j];\n\
         \x20   for (long i=0;i<C;i++) dx[r*C+i] = y[r*C+i]*(dy[r*C+i]-s); }}\n}}\n"
    )
}

fn rust_softmax_bwd(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(y:*const f32, dy:*const f32, dx:*mut f32) {{\n\
         \x20 for r in 0..R {{ let mut s=0.0f32; for j in 0..C {{ s += *y.add(r*C+j) * *dy.add(r*C+j); }}\n\
         \x20   for i in 0..C {{ *dx.add(r*C+i) = *y.add(r*C+i) * (*dy.add(r*C+i) - s); }} }}\n}}\n"
    )
}

/// Batched **RMSNorm backward** (input gradient) `dx = r·(g − x·r²·(Σ g·x)/C)`, `g = dy·γ`,
/// `r = 1/√(Σx²/C + eps)` over a `[rows, cols]` matrix — the gradient through every RMSNorm in a
/// transformer's backward pass (Llama/Mistral/Qwen). Each row needs **two** reductions (`Σx²`, `Σ g·x`)
/// that gcc/rustc keep scalar at `-O3` (they won't reassociate the float sums), then a fused
/// elementwise apply. Wukong folds the nest to one `wukong_rmsnorm_bwd_f32[_parallel]` call (both
/// reductions 8-wide, then the apply). Four buffers (`x, dy, gamma, dx`), so it uses the 4-pointer
/// harness. The reductions reassociate, so the cross-check is a magnitude-normalized tolerance like
/// the softmax-backward / norm benches.
fn bench_rmsnorm_bwd(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 512)] {
        let n = r * c;
        // Well-conditioned: x ~ O(1) (non-zero mean-square), a small grad, gamma ~ 1.
        let x: Vec<f32> = (0..n)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.1 + 0.3)
            .collect();
        let dy: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let gamma: Vec<f32> = (0..c).map(|i| (i % 11) as f32 * 0.05 + 0.7).collect();
        let mut dx = vec![0.0f32; n];
        let (xp, dyp, gp) = (x.as_ptr(), dy.as_ptr(), gamma.as_ptr());
        let bytes = (3.0 * n as f64 + c as f64) * 4.0; // x + dy read, gamma read, dx write
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== rmsnorm_bwd (dx = r·(g − x·r²·Σg·x/C)) {r}x{c} (GB/s, higher is better) ===");
        // Prepare every four-pointer column before entering the shared schedule.
        let wk = build_wukong(&wk_rmsnorm_bwd(r, c, false), 4);
        let wkp = build_wukong(&wk_rmsnorm_bwd(r, c, true), 4);
        let (cl, pl) = build_c_cpp(
            "rmsnorm_bwd",
            &c_rmsnorm_bwd(r, c),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let rl = build_peer(
            "rs",
            &rust_rmsnorm_bwd(r, c),
            dir,
            "rmsnorm_bwd",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let fl = build_peer(
            "c",
            &c_rmsnorm_bwd(r, c),
            dir,
            "rmsnorm_bwd_fast",
            cc,
            C_FAST_FLAGS,
        );
        let columns = [
            built_wukong4(wk, xp, dyp, gp),
            built_wukong4(wkp, xp, dyp, gp),
            built_peer4(cl, xp, dyp, gp),
            built_peer4(pl, xp, dyp, gp),
            built_peer4(fl, xp, dyp, gp),
            built_peer4(rl, xp, dyp, gp),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut dx, xp, dyp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("rmsnorm_bwd", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|cf| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("rmsnorm_bwd", "C(fast)", m, cf))
        });
        let rm = timed.next().flatten();
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&cfast),
            gbps(&rm)
        );
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! rmsnorm_bwd mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r2);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// Wukong RMSNorm-backward: the batched nest the recognizer folds to one
/// `wukong_rmsnorm_bwd_f32[_parallel]` call. The 4-pointer harness carries `(x, dy, gamma, dx)`.
fn wk_rmsnorm_bwd(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], dy: [f32; {n}], gamma: [f32; {cols}], mut dx: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut ms: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ ms = ms + x[r * {cols} + i] * x[r * {cols} + i]; }}\n\
         \x20       let rinv: f32 = 1.0 / sqrt(ms / {cols}.0 + 0.00001);\n\
         \x20       let mut sg: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ sg = sg + dy[r * {cols} + i] * gamma[i] * x[r * {cols} + i]; }}\n\
         \x20       let coef: f32 = rinv * rinv * sg / {cols}.0;\n\
         \x20       for i in 0..{cols} {{ dx[r * {cols} + i] = rinv * (dy[r * {cols} + i] * gamma[i] - x[r * {cols} + i] * coef); }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_rmsnorm_bwd(rows: usize, cols: usize) -> String {
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ dy, const float* __restrict__ gamma, float* __restrict__ dx){{\n\
         \x20 for (long r=0;r<R;r++){{\n\
         \x20   float ms=0.0f; for(long i=0;i<C;i++) ms += x[r*C+i]*x[r*C+i];\n\
         \x20   float rinv = 1.0f/sqrtf(ms/(float)C + 0.00001f);\n\
         \x20   float sg=0.0f; for(long i=0;i<C;i++) sg += dy[r*C+i]*gamma[i]*x[r*C+i];\n\
         \x20   float coef = rinv*rinv*sg/(float)C;\n\
         \x20   for(long i=0;i<C;i++) dx[r*C+i] = rinv*(dy[r*C+i]*gamma[i] - x[r*C+i]*coef); }}\n}}\n"
    )
}

fn rust_rmsnorm_bwd(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, dy:*const f32, gamma:*const f32, dx:*mut f32) {{\n\
         \x20 for r in 0..R {{\n\
         \x20   let mut ms=0.0f32; for i in 0..C {{ ms += *x.add(r*C+i) * *x.add(r*C+i); }}\n\
         \x20   let rinv = 1.0f32/(ms/(C as f32) + 0.00001f32).sqrt();\n\
         \x20   let mut sg=0.0f32; for i in 0..C {{ sg += *dy.add(r*C+i) * *gamma.add(i) * *x.add(r*C+i); }}\n\
         \x20   let coef = rinv*rinv*sg/(C as f32);\n\
         \x20   for i in 0..C {{ *dx.add(r*C+i) = rinv*(*dy.add(r*C+i) * *gamma.add(i) - *x.add(r*C+i) * coef); }} }}\n}}\n"
    )
}

/// Batched **LayerNorm backward** (input gradient) `dx = rstd·(g − mean(g) − xhat·mean(g·xhat))`,
/// `g = dy·γ`, `xhat = (x−mean)·rstd` over `[rows, cols]` — the gradient through every LayerNorm
/// (GPT-2/BERT/ViT). Four per-row reductions (Σx, Σ(x−mean)², Σg, Σg·xhat) gcc/rustc keep **scalar**;
/// Wukong folds them 8-wide → `wukong_layernorm_bwd_f32[_parallel]`. Four buffers → the 4-pointer
/// harness; the reassociated reductions → magnitude-normalized cross-check (like rmsnorm_bwd).
fn bench_layernorm_bwd(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 512)] {
        let n = r * c;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.1 + 0.3)
            .collect();
        let dy: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let gamma: Vec<f32> = (0..c).map(|i| (i % 11) as f32 * 0.05 + 0.7).collect();
        let mut dx = vec![0.0f32; n];
        let (xp, dyp, gp) = (x.as_ptr(), dy.as_ptr(), gamma.as_ptr());
        let bytes = (3.0 * n as f64 + c as f64) * 4.0;
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== layernorm_bwd (dx = rstd·(g − mean(g) − xhat·mean(g·xhat))) {r}x{c} ===");
        // Prepare every four-pointer column before entering the shared schedule.
        let wk = build_wukong(&wk_layernorm_bwd(r, c, false), 4);
        let wkp = build_wukong(&wk_layernorm_bwd(r, c, true), 4);
        let (cl, pl) = build_c_cpp(
            "layernorm_bwd",
            &c_layernorm_bwd(r, c),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let rl = build_peer(
            "rs",
            &rust_layernorm_bwd(r, c),
            dir,
            "layernorm_bwd",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        let fl = build_peer(
            "c",
            &c_layernorm_bwd(r, c),
            dir,
            "layernorm_bwd_fast",
            cc,
            C_FAST_FLAGS,
        );
        let columns = [
            built_wukong4(wk, xp, dyp, gp),
            built_wukong4(wkp, xp, dyp, gp),
            built_peer4(cl, xp, dyp, gp),
            built_peer4(pl, xp, dyp, gp),
            built_peer4(fl, xp, dyp, gp),
            built_peer4(rl, xp, dyp, gp),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut dx, xp, dyp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("layernorm_bwd", &cm, timed.next().flatten());
        let cfast = timed.next().flatten().filter(|cf| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("layernorm_bwd", "C(fast)", m, cf))
        });
        let rm = timed.next().flatten();
        report_ratio(
            "layernorm_bwd",
            &wuk,
            &wk_par,
            &cm,
            &cm_cpp,
            &rm,
            &cfast,
            &gbps,
        );
    }
}

fn wk_layernorm_bwd(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], dy: [f32; {n}], gamma: [f32; {cols}], mut dx: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut sm: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ sm = sm + x[r * {cols} + i]; }}\n\
         \x20       let mean: f32 = sm / {cols}.0;\n\
         \x20       let mut vv: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ vv = vv + (x[r * {cols} + i] - mean) * (x[r * {cols} + i] - mean); }}\n\
         \x20       let rstd: f32 = 1.0 / sqrt(vv / {cols}.0 + 0.00001);\n\
         \x20       let mut s1: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ s1 = s1 + dy[r * {cols} + i] * gamma[i]; }}\n\
         \x20       let mut s2: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ s2 = s2 + dy[r * {cols} + i] * gamma[i] * ((x[r * {cols} + i] - mean) * rstd); }}\n\
         \x20       let m1: f32 = s1 / {cols}.0;\n\
         \x20       let m2: f32 = s2 / {cols}.0;\n\
         \x20       for i in 0..{cols} {{ dx[r * {cols} + i] = rstd * (dy[r * {cols} + i] * gamma[i] - m1 - ((x[r * {cols} + i] - mean) * rstd) * m2); }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_layernorm_bwd(rows: usize, cols: usize) -> String {
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ dy, const float* __restrict__ gamma, float* __restrict__ dx){{\n\
         \x20 for (long r=0;r<R;r++){{\n\
         \x20   float sm=0.0f; for(long i=0;i<C;i++) sm+=x[r*C+i]; float mean=sm/(float)C;\n\
         \x20   float vv=0.0f; for(long i=0;i<C;i++){{ float d=x[r*C+i]-mean; vv+=d*d; }} float rstd=1.0f/sqrtf(vv/(float)C+0.00001f);\n\
         \x20   float s1=0.0f,s2=0.0f; for(long i=0;i<C;i++){{ float g=dy[r*C+i]*gamma[i]; float xh=(x[r*C+i]-mean)*rstd; s1+=g; s2+=g*xh; }}\n\
         \x20   float m1=s1/(float)C, m2=s2/(float)C;\n\
         \x20   for(long i=0;i<C;i++){{ float g=dy[r*C+i]*gamma[i]; float xh=(x[r*C+i]-mean)*rstd; dx[r*C+i]=rstd*(g-m1-xh*m2); }} }}\n}}\n"
    )
}

fn rust_layernorm_bwd(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, dy:*const f32, gamma:*const f32, dx:*mut f32) {{\n\
         \x20 for r in 0..R {{\n\
         \x20   let mut sm=0.0f32; for i in 0..C {{ sm+=*x.add(r*C+i); }} let mean=sm/(C as f32);\n\
         \x20   let mut vv=0.0f32; for i in 0..C {{ let d=*x.add(r*C+i)-mean; vv+=d*d; }} let rstd=1.0f32/(vv/(C as f32)+0.00001f32).sqrt();\n\
         \x20   let mut s1=0.0f32; let mut s2=0.0f32; for i in 0..C {{ let g=*dy.add(r*C+i)* *gamma.add(i); let xh=(*x.add(r*C+i)-mean)*rstd; s1+=g; s2+=g*xh; }}\n\
         \x20   let m1=s1/(C as f32); let m2=s2/(C as f32);\n\
         \x20   for i in 0..C {{ let g=*dy.add(r*C+i)* *gamma.add(i); let xh=(*x.add(r*C+i)-mean)*rstd; *dx.add(r*C+i)=rstd*(g-m1-xh*m2); }} }} }}\n"
    )
}

/// Batched **softmax cross-entropy forward loss** `loss[r] = m + log(Σexp(x[r,·]−m)) − x[r,target[r]]`
/// over `[rows, classes]` logits — the training loss of every classifier / language model. The
/// stabilizing row-max + Σexp(x−m) reduction gcc/rustc keep **scalar** (`expf`/`logf` won't vectorize),
/// so Wukong's fused 256-bit `wukong_xent_fwd_f32[_parallel]` wins. The `target` labels are `i32`, so
/// the harness's middle `*const f32` pointer carries the `i32` buffer's address (a ptr is a ptr; the
/// kernel / C baseline cast it back to `int`). GB/s counts the logits read once + the small label/loss
/// vectors. The lse reductions reassociate → magnitude-normalized cross-check, like softmax_bwd.
fn bench_xent(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 512)] {
        let n = r * c;
        let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
        let target: Vec<i32> = (0..r).map(|i| ((i * 7 + 3) % c) as i32).collect();
        let mut loss = vec![0.0f32; r];
        let xp = x.as_ptr();
        let tp = target.as_ptr() as *const f32; // i32 labels carried through the f32 ptr slot
        let bytes = (n as f64 + 2.0 * r as f64) * 4.0; // logits read once + labels + loss
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== xent (loss = lse(x) − x[target]) {r}x{c} (GB/s, higher is better) ===");
        let built_wuk = build_wukong(&wk_xent(r, c, false), 3);
        let built_wk_par = build_wukong(&wk_xent(r, c, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "xent",
            &c_xent(r, c),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_xent(r, c),
            dir,
            "xent",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer: -ffast-math on the row-max + Σexp reductions.
        let built_cfast = build_peer(
            "c",
            &c_xent(r, c),
            dir,
            &format!("{}_fast", "xent"),
            cc,
            C_FAST_FLAGS,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
            Built::peer(built_cfast),
        ];
        let mut timed = abba_min_columns(columns.len(), |i| {
            time_built(&columns[i], &mut loss, xp, tp)
        })
        .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("xent", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("xent", "C(fast)", m, p))
        });

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&cfast),
            gbps(&rm)
        );
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! xent mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r2);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        println!();
    }
}

/// Wukong cross-entropy loss: the batched nest the recognizer folds to one
/// `wukong_xent_fwd_f32[_parallel]` call. The 3-pointer harness carries `(x, target-as-f32, loss)`.
fn wk_xent(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], target: [i32; {rows}], mut loss: [f32; {rows}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut m: f32 = x[r * {cols}];\n\
         \x20       for i in 0..{cols} {{ m = fmax(m, x[r * {cols} + i]); }}\n\
         \x20       let mut s: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ s = s + exp(x[r * {cols} + i] - m); }}\n\
         \x20       loss[r] = m + log(s) - x[r * {cols} + target[r]];\n\
         \x20   }}\n}}\n"
    )
}

fn c_xent(rows: usize, cols: usize) -> String {
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ target_f, float* __restrict__ loss){{\n\
         \x20 const int* target = (const int*)target_f;\n\
         \x20 for (long r=0;r<R;r++){{\n\
         \x20   float m=x[r*C]; for(long i=0;i<C;i++) if(x[r*C+i]>m) m=x[r*C+i];\n\
         \x20   float s=0.0f; for(long i=0;i<C;i++) s+=expf(x[r*C+i]-m);\n\
         \x20   loss[r] = m + logf(s) - x[r*C + target[r]]; }}\n}}\n"
    )
}

fn rust_xent(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, target_f:*const f32, loss:*mut f32) {{\n\
         \x20 let target = target_f as *const i32;\n\
         \x20 for r in 0..R {{\n\
         \x20   let mut m=*x.add(r*C); for i in 0..C {{ let v=*x.add(r*C+i); if v>m {{ m=v; }} }}\n\
         \x20   let mut s=0.0f32; for i in 0..C {{ s += (*x.add(r*C+i) - m).exp(); }}\n\
         \x20   *loss.add(r) = m + s.ln() - *x.add(r*C + *target.add(r) as usize); }} }}\n"
    )
}

/// **RoPE** (rotary position embedding) `out = rotate(x[r,·]) by theta = r·inv_freq[j]` over `[rows, D]`
/// (`D = 2·half`) — the per-attention-layer positional rotation of every modern LLM (Llama/Qwen/…). The
/// angles' cos/sin are computed **inline** per element, so C/Rust keep `sinf`/`cosf` scalar; Wukong
/// folds the nest to one `wukong_rope_f32[_parallel]` (the 8-wide sin8/cos8). All three buffers are
/// f32, so the plain 3-pointer harness `(x, inv_freq, out)` serves. The kernel is a bit-exact rotation;
/// only the poly-vs-libm ~1-ULP gap remains, so the cross-check is a tight magnitude-normalized tol.
fn bench_rope(cc: &str, dir: &Path) {
    for (rows, half) in [(8192usize, 64usize), (16384, 32)] {
        let d = 2 * half;
        let n = rows * d;
        let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        // The standard RoPE schedule inv_freq[k] = base^(−k/half), base = 10000.
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| 10000f32.powf(-(k as f32) / half as f32))
            .collect();
        let mut out = vec![0.0f32; n];
        let (xp, fp) = (x.as_ptr(), inv_freq.as_ptr());
        let bytes = (2.0 * n as f64 + half as f64) * 4.0; // x read + out write + inv_freq
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== rope (rotate by r·inv_freq[j]) {rows}x{d} (GB/s, higher is better) ===");
        let built_wuk = build_wukong(&wk_rope(rows, half, false), 3);
        let built_wk_par = build_wukong(&wk_rope(rows, half, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "rope",
            &c_rope(rows, half),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_rope(rows, half),
            dir,
            "rope",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, fp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("rope", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&rm)
        );
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! rope mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core is {:.2}x {} than idiomatic C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r2);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        println!();
    }
}

/// Wukong RoPE: the nest the recognizer folds to one `wukong_rope_f32[_parallel]` call. `D = 2·half`.
fn wk_rope(rows: usize, half: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let d = 2 * half;
    let n = rows * d;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], inv_freq: [f32; {half}], mut out: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       for j in 0..{half} {{\n\
         \x20           let theta: f32 = (r as f32) * inv_freq[j];\n\
         \x20           let c: f32 = cos(theta);\n\
         \x20           let s: f32 = sin(theta);\n\
         \x20           let a: f32 = x[r * {d} + j];\n\
         \x20           let b: f32 = x[r * {d} + j + {half}];\n\
         \x20           out[r * {d} + j] = a * c - b * s;\n\
         \x20           out[r * {d} + j + {half}] = b * c + a * s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_rope(rows: usize, half: usize) -> String {
    let d = 2 * half;
    format!(
        "#include <math.h>\n#define R {rows}\n#define H {half}\n#define D {d}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ inv_freq, float* __restrict__ out){{\n\
         \x20 for (long r=0;r<R;r++) for (long j=0;j<H;j++){{\n\
         \x20   float theta=(float)r*inv_freq[j]; float c=cosf(theta), s=sinf(theta);\n\
         \x20   float a=x[r*D+j], b=x[r*D+j+H];\n\
         \x20   out[r*D+j]=a*c-b*s; out[r*D+j+H]=b*c+a*s; }}\n}}\n"
    )
}

fn rust_rope(rows: usize, half: usize) -> String {
    let d = 2 * half;
    format!(
        "const R: usize = {rows};\nconst H: usize = {half};\nconst D: usize = {d};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, inv_freq:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ for j in 0..H {{\n\
         \x20   let theta=(r as f32)* *inv_freq.add(j); let c=theta.cos(); let s=theta.sin();\n\
         \x20   let a=*x.add(r*D+j); let b=*x.add(r*D+j+H);\n\
         \x20   *out.add(r*D+j)=a*c-b*s; *out.add(r*D+j+H)=b*c+a*s; }} }} }}\n"
    )
}

/// Softmax cross-entropy **backward** `dx[r,i] = softmax(x[r])[i] − onehot(target[r])[i]` over
/// `[rows, classes]` — the input gradient of every classifier/LM training step. The softmax max+Σexp+
/// recompute that gcc/rustc keep scalar (expf reduction + per-element expf) → Wukong's fused 256-bit
/// `wukong_xent_bwd_f32[_parallel]`. i32 labels ride the harness's middle f32 pointer (cast back).
fn bench_xent_bwd(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 512)] {
        let n = r * c;
        let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
        let target: Vec<i32> = (0..r).map(|i| ((i * 7 + 3) % c) as i32).collect();
        let mut dx = vec![0.0f32; n];
        let xp = x.as_ptr();
        let tp = target.as_ptr() as *const f32;
        let bytes = (2.0 * n as f64 + r as f64) * 4.0; // x read + dx write + labels
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== xent_bwd (dx = softmax(x) − onehot) {r}x{c} (GB/s, higher is better) ===");
        let built_wuk = build_wukong(&wk_xent_bwd(r, c, false), 3);
        let built_wk_par = build_wukong(&wk_xent_bwd(r, c, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "xent_bwd",
            &c_xent_bwd(r, c),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_xent_bwd(r, c),
            dir,
            "xent_bwd",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Reassociation-normalized peer: the per-row `max` and `Σexp` are float reductions Wukong
        // folds 8 lanes wide, so the plain-flags C column is the strictly-in-order one. (`rope`,
        // `rope_bwd`, `gate` and `act_backward` deliberately pass `&None` here — they are pure
        // elementwise maps with no reduction, so `-ffast-math` normalizes nothing for them.)
        let built_cfast = build_peer(
            "c",
            &c_xent_bwd(r, c),
            dir,
            &format!("{}_fast", "xent_bwd"),
            cc,
            C_FAST_FLAGS,
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
            Built::peer(built_cfast),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut dx, xp, tp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("xent_bwd", &cm, timed.next().flatten());
        let rm = timed.next().flatten();
        let cfast = timed.next().flatten().filter(|p| {
            wuk.as_ref()
                .is_some_and(|m| relaxed_peer_ok("xent_bwd", "C(fast)", m, p))
        });

        report_ratio("xent_bwd", &wuk, &wk_par, &cm, &cm_cpp, &rm, &cfast, &gbps);
    }
}

fn wk_xent_bwd(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], target: [i32; {rows}], mut dx: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       let mut m: f32 = x[r * {cols}];\n\
         \x20       for i in 0..{cols} {{ m = fmax(m, x[r * {cols} + i]); }}\n\
         \x20       let mut z: f32 = 0.0;\n\
         \x20       for i in 0..{cols} {{ z = z + exp(x[r * {cols} + i] - m); }}\n\
         \x20       let invz: f32 = 1.0 / z;\n\
         \x20       for i in 0..{cols} {{ dx[r * {cols} + i] = exp(x[r * {cols} + i] - m) * invz; }}\n\
         \x20       dx[r * {cols} + target[r]] = dx[r * {cols} + target[r]] - 1.0;\n\
         \x20   }}\n}}\n"
    )
}

fn c_xent_bwd(rows: usize, cols: usize) -> String {
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ target_f, float* __restrict__ dx){{\n\
         \x20 const int* target = (const int*)target_f;\n\
         \x20 for (long r=0;r<R;r++){{\n\
         \x20   float m=x[r*C]; for(long i=0;i<C;i++) if(x[r*C+i]>m) m=x[r*C+i];\n\
         \x20   float z=0.0f; for(long i=0;i<C;i++) z+=expf(x[r*C+i]-m);\n\
         \x20   float invz=1.0f/z; for(long i=0;i<C;i++) dx[r*C+i]=expf(x[r*C+i]-m)*invz;\n\
         \x20   dx[r*C+target[r]] -= 1.0f; }}\n}}\n"
    )
}

fn rust_xent_bwd(rows: usize, cols: usize) -> String {
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, target_f:*const f32, dx:*mut f32) {{\n\
         \x20 let target = target_f as *const i32;\n\
         \x20 for r in 0..R {{\n\
         \x20   let mut m=*x.add(r*C); for i in 0..C {{ let v=*x.add(r*C+i); if v>m {{ m=v; }} }}\n\
         \x20   let mut z=0.0f32; for i in 0..C {{ z += (*x.add(r*C+i)-m).exp(); }}\n\
         \x20   let invz=1.0f32/z; for i in 0..C {{ *dx.add(r*C+i)=(*x.add(r*C+i)-m).exp()*invz; }}\n\
         \x20   *dx.add(r*C + *target.add(r) as usize) -= 1.0f32; }} }}\n"
    )
}

/// RoPE **backward** (transpose rotation) over `[rows, D]` — pairs with the forward; the inline
/// sin/cos gcc/rustc keep scalar → `wukong_rope_bwd_f32[_parallel]`. All-f32 3-pointer harness.
fn bench_rope_bwd(cc: &str, dir: &Path) {
    for (rows, half) in [(8192usize, 64usize), (16384, 32)] {
        let d = 2 * half;
        let n = rows * d;
        let g: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| 10000f32.powf(-(k as f32) / half as f32))
            .collect();
        let mut dx = vec![0.0f32; n];
        let (gp, fp) = (g.as_ptr(), inv_freq.as_ptr());
        let bytes = (2.0 * n as f64 + half as f64) * 4.0;
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== rope_bwd (transpose rotate) {rows}x{d} (GB/s, higher is better) ===");
        let built_wuk = build_wukong(&wk_rope_bwd(rows, half, false), 3);
        let built_wk_par = build_wukong(&wk_rope_bwd(rows, half, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "rope_bwd",
            &c_rope_bwd(rows, half),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_rope_bwd(rows, half),
            dir,
            "rope_bwd",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut dx, gp, fp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("rope_bwd", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        report_ratio("rope_bwd", &wuk, &wk_par, &cm, &cm_cpp, &rm, &None, &gbps);
    }
}

fn wk_rope_bwd(rows: usize, half: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let d = 2 * half;
    let n = rows * d;
    format!(
        "module bench\n{attr}fn kbench(g: [f32; {n}], inv_freq: [f32; {half}], mut dx: [f32; {n}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20       for j in 0..{half} {{\n\
         \x20           let theta: f32 = (r as f32) * inv_freq[j];\n\
         \x20           let c: f32 = cos(theta);\n\
         \x20           let s: f32 = sin(theta);\n\
         \x20           let a: f32 = g[r * {d} + j];\n\
         \x20           let b: f32 = g[r * {d} + j + {half}];\n\
         \x20           dx[r * {d} + j] = a * c + b * s;\n\
         \x20           dx[r * {d} + j + {half}] = b * c - a * s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_rope_bwd(rows: usize, half: usize) -> String {
    let d = 2 * half;
    format!(
        "#include <math.h>\n#define R {rows}\n#define H {half}\n#define D {d}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ g, const float* __restrict__ inv_freq, float* __restrict__ dx){{\n\
         \x20 for (long r=0;r<R;r++) for (long j=0;j<H;j++){{\n\
         \x20   float theta=(float)r*inv_freq[j]; float c=cosf(theta), s=sinf(theta);\n\
         \x20   float a=g[r*D+j], b=g[r*D+j+H];\n\
         \x20   dx[r*D+j]=a*c+b*s; dx[r*D+j+H]=b*c-a*s; }}\n}}\n"
    )
}

fn rust_rope_bwd(rows: usize, half: usize) -> String {
    let d = 2 * half;
    format!(
        "const R: usize = {rows};\nconst H: usize = {half};\nconst D: usize = {d};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(g:*const f32, inv_freq:*const f32, dx:*mut f32) {{\n\
         \x20 for r in 0..R {{ for j in 0..H {{\n\
         \x20   let theta=(r as f32)* *inv_freq.add(j); let c=theta.cos(); let s=theta.sin();\n\
         \x20   let a=*g.add(r*D+j); let b=*g.add(r*D+j+H);\n\
         \x20   *dx.add(r*D+j)=a*c+b*s; *dx.add(r*D+j+H)=b*c-a*s; }} }} }}\n"
    )
}

/// Gated-FFN activation `out[i] = act(a[i]) * b[i]` (SwiGLU silu / GeGLU gelu). The activation's exp
/// gcc/rustc keep scalar → Wukong's 256-bit `wukong_vmath2_f32` gate op. The op is selected
/// 0=silu/1=gelu. All-f32 3-pointer harness `(a, b, out)`.
///
/// **Two sizes (added 2026-08-05).** `BENCHMARKS.md` carried this family as "~5–13×" with the note
/// "per-size table not reproduced here", and the reason there was no per-size table is that the bench
/// measured exactly one size. `N=2²⁰` (3 × 4 MiB, ~L3-resident) is the compute-bound regime where the
/// vectorized `exp` is the whole story; `N=2²⁴` (3 × 64 MiB, ≫L3) is the streaming regime where DRAM
/// bandwidth caps both languages and the ratio must compress. A one-size row cannot distinguish them.
fn bench_gate(cc: &str, dir: &Path) {
    for n in [1usize << 20, 1usize << 24] {
        bench_gate_n(cc, dir, n);
    }
}

fn bench_gate_n(cc: &str, dir: &Path, n: usize) {
    let a: Vec<f32> = (0..n)
        .map(|i| (i as f32 - (n / 2) as f32) * (12.0 / n as f32))
        .collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();
    let mut out = vec![0.0f32; n];
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let bytes = 3.0 * n as f64 * 4.0; // a + b read, out write
    let gbps = |v: &Option<Measure>| {
        v.as_ref()
            .map(|m| format!("{:.1}", bytes / m.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    for (name, act) in [("silu (SwiGLU)", "silu"), ("gelu (GeGLU)", "gelu")] {
        println!(
            "=== gate {name}: out = {act}(a)*b, N=2^{} (GB/s, higher is better) ===",
            n.trailing_zeros()
        );
        let built_wuk = build_wukong(&wk_gate(n, act, false), 3);
        let built_wk_par = build_wukong(&wk_gate(n, act, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "gate",
            &c_gate(n, act),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_gate(n, act),
            dir,
            "gate",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, ap, bp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("gate", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        report_ratio("gate", &wuk, &wk_par, &cm, &cm_cpp, &rm, &None, &gbps);
    }
}

fn wk_gate(n: usize, act: &str, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n}], b: [f32; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for i in 0..{n} {{ out[i] = {act}(a[i]) * b[i]; }}\n}}\n"
    )
}

fn c_gate(n: usize, act: &str) -> String {
    // silu(x)=x/(1+e^-x); gelu(x)=0.5x(1+tanh(√(2/π)(x+0.044715x³))) — match Wukong's tanh-approx gelu.
    let actexpr = if act == "silu" {
        "x/(1.0f+expf(-x))"
    } else {
        "0.5f*x*(1.0f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)))"
    };
    format!(
        "#include <math.h>\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){{\n\
         \x20 for (long i=0;i<N;i++){{ float x=a[i]; out[i]=({actexpr})*b[i]; }}\n}}\n"
    )
}

fn rust_gate(n: usize, act: &str) -> String {
    let actexpr = if act == "silu" {
        "x/(1.0f32+(-x).exp())"
    } else {
        "0.5f32*x*(1.0f32+(0.7978845608f32*(x+0.044715f32*x*x*x)).tanh())"
    };
    format!(
        "const N: usize = {n};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, out:*mut f32) {{\n\
         \x20 for i in 0..N {{ let x=*a.add(i); *out.add(i)=({actexpr})* *b.add(i); }} }}\n"
    )
}

/// Per-row **log/exp-bound loss reductions** — KL divergence `out[r]=Σ p·(log p−log q)`, Shannon
/// entropy `out[r]=−Σ p·log p`, and soft-label cross-entropy `out[r]=Σ q·(lse(x)−x)` — over `[rows,
/// cols]`. C/Rust keep the logf/expf reductions scalar; Wukong folds them 8-wide. All write a
/// `rows`-length scalar output; the 3-pointer harness carries the inputs (entropy aliases its unused
/// middle pointer). Magnitude-normalized cross-check (the reductions reassociate / differ ~1 ULP).
fn bench_row_losses(cc: &str, dir: &Path) {
    for (r, c) in [(1024usize, 1024usize), (4096, 512)] {
        let n = r * c;
        // Strictly-positive "distributions" (un-normalized is fine for timing; log stays finite).
        let p: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 + 1.0) * 0.05).collect();
        let q: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 + 1.0) * 0.07).collect();
        // logits for kd_loss
        let x: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.3).collect();
        let mut out = vec![0.0f32; r];
        let (pp, qp, xp) = (p.as_ptr(), q.as_ptr(), x.as_ptr());
        let basis = move |reads: f64| (reads * n as f64 + r as f64) * 4.0;
        for (label, src_m, src_c, src_r, p0, p1, reads) in [
            (
                "kldiv",
                wk_row_loss(r, c, "kldiv", false),
                c_row_loss(r, c, "kldiv"),
                rust_row_loss(r, c, "kldiv"),
                pp,
                qp,
                2.0,
            ),
            (
                "entropy",
                wk_row_loss(r, c, "entropy", false),
                c_row_loss(r, c, "entropy"),
                rust_row_loss(r, c, "entropy"),
                pp,
                pp,
                1.0,
            ),
            (
                "kd_loss",
                wk_row_loss(r, c, "kd_loss", false),
                c_row_loss(r, c, "kd_loss"),
                rust_row_loss(r, c, "kd_loss"),
                xp,
                qp,
                2.0,
            ),
        ] {
            let bytes = basis(reads);
            let gbps = move |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let src_mp = wk_row_loss(r, c, label, true);
            println!("=== {label} (per-row log/exp loss) {r}x{c} ===");
            let built_wuk = build_wukong(&src_m, 3);
            let built_wk_par = build_wukong(&src_mp, 3);
            let (built_cm, built_cm_cpp) =
                build_c_cpp(label, &src_c, dir, cc, &["-O3", "-march=native", "-shared"]);
            let built_rm = build_peer(
                "rs",
                &src_r,
                dir,
                label,
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            // Reassociation-normalized peer: the per-row logf/expf reductions under -ffast-math.
            let built_cfast = build_peer(
                "c",
                &src_c,
                dir,
                &format!("{}_fast", label),
                cc,
                C_FAST_FLAGS,
            );
            // Every compiled column keeps its slot, including unavailable peers.
            let columns = [
                Built::wk(built_wuk),
                Built::wk(built_wk_par),
                Built::peer(built_cm),
                Built::peer(built_cm_cpp),
                Built::peer(built_rm),
                Built::peer(built_cfast),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, p0, p1))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wk_par = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked(label, &cm, timed.next().flatten());
            let rm = timed.next().flatten();
            let cfast = timed.next().flatten().filter(|p| {
                wuk.as_ref()
                    .is_some_and(|m| relaxed_peer_ok(label, "C(fast)", m, p))
            });

            report_ratio(label, &wuk, &wk_par, &cm, &cm_cpp, &rm, &cfast, &gbps);
        }
    }
}

fn wk_row_loss(rows: usize, cols: usize, kind: &str, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    let body = match kind {
        "kldiv" => format!(
            "       let mut s: f32 = 0.0;\n\
             \x20       for i in 0..{cols} {{ s = s + a[r * {cols} + i] * (log(a[r * {cols} + i]) - log(b[r * {cols} + i])); }}\n\
             \x20       out[r] = s;"
        ),
        "entropy" => format!(
            "       let mut s: f32 = 0.0;\n\
             \x20       for i in 0..{cols} {{ s = s + a[r * {cols} + i] * log(a[r * {cols} + i]); }}\n\
             \x20       out[r] = -s;"
        ),
        _ => format!(
            "       let mut m: f32 = a[r * {cols}];\n\
             \x20       for i in 0..{cols} {{ m = fmax(m, a[r * {cols} + i]); }}\n\
             \x20       let mut z: f32 = 0.0;\n\
             \x20       for i in 0..{cols} {{ z = z + exp(a[r * {cols} + i] - m); }}\n\
             \x20       let lse: f32 = m + log(z);\n\
             \x20       let mut s: f32 = 0.0;\n\
             \x20       for i in 0..{cols} {{ s = s + b[r * {cols} + i] * (lse - a[r * {cols} + i]); }}\n\
             \x20       out[r] = s;"
        ),
    };
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n}], b: [f32; {n}], mut out: [f32; {rows}]) {{\n\
         \x20   for r in 0..{rows} {{\n\
         \x20{body}\n\
         \x20   }}\n}}\n"
    )
}

fn c_row_loss(rows: usize, cols: usize, kind: &str) -> String {
    let body = match kind {
        "kldiv" => "float s=0.0f; for(long i=0;i<C;i++) s+=a[r*C+i]*(logf(a[r*C+i])-logf(b[r*C+i])); out[r]=s;",
        "entropy" => "float s=0.0f; for(long i=0;i<C;i++) s+=a[r*C+i]*logf(a[r*C+i]); out[r]=-s;",
        _ => "float m=a[r*C]; for(long i=0;i<C;i++) if(a[r*C+i]>m) m=a[r*C+i]; float z=0.0f; for(long i=0;i<C;i++) z+=expf(a[r*C+i]-m); float lse=m+logf(z); float s=0.0f; for(long i=0;i<C;i++) s+=b[r*C+i]*(lse-a[r*C+i]); out[r]=s;",
    };
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){{\n\
         \x20 for (long r=0;r<R;r++){{ {body} }}\n}}\n"
    )
}

fn rust_row_loss(rows: usize, cols: usize, kind: &str) -> String {
    let body = match kind {
        "kldiv" => "let mut s=0.0f32; for i in 0..C { s+=*a.add(r*C+i)*((*a.add(r*C+i)).ln()-(*b.add(r*C+i)).ln()); } *out.add(r)=s;",
        "entropy" => "let mut s=0.0f32; for i in 0..C { s+=*a.add(r*C+i)*(*a.add(r*C+i)).ln(); } *out.add(r)=-s;",
        _ => "let mut m=*a.add(r*C); for i in 0..C { let v=*a.add(r*C+i); if v>m { m=v; } } let mut z=0.0f32; for i in 0..C { z+=(*a.add(r*C+i)-m).exp(); } let lse=m+z.ln(); let mut s=0.0f32; for i in 0..C { s+=*b.add(r*C+i)*(lse-*a.add(r*C+i)); } *out.add(r)=s;",
    };
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         #[allow(unused_variables)]\n\
         pub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, out:*mut f32) {{\n\
         \x20 for r in 0..R {{ {body} }} }}\n"
    )
}

/// Shared ratio report for the backward/gate benches (Wuk 1-core / Wuk par / C / C++ / Rust + the
/// single-core and @parallel ratios vs idiomatic C and C++). The cross-check (when both present) uses a
/// magnitude-normalized tolerance, since the transcendental reductions reassociate / differ from libm by
/// ~1 ULP. `cfast` is the optional relaxed-FP C(fast) peer column (already loose-cross-checked by the
/// caller); benches whose C baseline is not an IEEE-serial reduction pass `&None`. `cppm` comes from
/// [`build_c_cpp`], which is the only way to obtain a C column — so this table cannot print "vs C"
/// without the C++ number beside it.
#[allow(clippy::too_many_arguments)]
fn report_ratio(
    label: &str,
    wuk: &Option<Measure>,
    wk_par: &Option<Measure>,
    cm: &Option<Measure>,
    cppm: &Option<Measure>,
    rm: &Option<Measure>,
    cfast: &Option<Measure>,
    gbps: &dyn Fn(&Option<Measure>) -> String,
) {
    let has_fast = cfast.is_some();
    let mut hdr = format!(
        "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
    );
    let mut vals = format!(
        "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "GB/s",
        gbps(wuk),
        gbps(wk_par),
        gbps(cm),
        gbps(cppm),
        gbps(rm)
    );
    if has_fast {
        hdr.push_str(&format!(" {:>11}", "C(fast)"));
        vals.push_str(&format!(" {:>11}", gbps(cfast)));
    }
    println!("{hdr}");
    println!("{vals}");
    if let (Some(m), Some(c2)) = (wuk, cm) {
        let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
        let maxerr = m
            .out
            .iter()
            .zip(&c2.out)
            .fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
        let rel = (maxerr / maxabs) as f64;
        if rel > 1e-3 {
            println!("  ! {label} mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
        }
    }
    if let (Some(ms), Some(c2)) = (wuk, cm) {
        let r2 = c2.ns_per_call / ms.ns_per_call;
        println!(
            "  -> Wukong single-core is {:.2}x {} than idiomatic C",
            if r2 >= 1.0 { r2 } else { 1.0 / r2 },
            if r2 >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let (Some(mp), Some(c2)) = (wk_par, cm) {
        let r2 = c2.ns_per_call / mp.ns_per_call;
        par_standing("idiomatic single-threaded C", r2);
    }
    report_cpp_ratio(wuk, wk_par, cppm);
    report_relaxed_ratio("C(fast) [-ffast-math]", wuk, wk_par, cfast);
    println!();
}

/// Activation **backward** `dx[i] = dy[i]·act'(x[i])` (silu/gelu — the SiLU/GELU training gradient).
/// Compute-bound: the derivative folds a sigmoid/tanh (an `expf`) that C/Rust call as scalar libm
/// inside the loop, so they cannot vectorize it. Wukong folds the elementwise
/// `dx[i]=act_backward(x[i],dy[i])` loop to one **256-bit** `wukong_vmath2_f32` call; the `@parallel`
/// form spreads (128-bit) SIMD across cores. Same `(x, dy, dx)` 3-pointer harness as softmax_bwd (all
/// three used). The poly derivative differs from C's libm by ~1 ULP, so the cross-check is a
/// magnitude-normalized tolerance (`max|Δ|/max|C| < 1e-3`), like the norm/softmax benches.
///
/// **Two sizes (added 2026-08-05).** This family published a single N and therefore no per-size
/// table, so a reader could not tell whether the ratio was a compute win or a cache artefact — the
/// same blank-cell problem as a missing language column. `N=2²⁰` keeps all three buffers (12 MiB)
/// near L3 so the transcendental cost dominates; `N=2²⁴` (192 MiB) spills it, so the row becomes
/// bandwidth-bound and the ratio has to fall towards a tie. Both regimes are real; publishing only
/// the first is picking the flattering one.
fn bench_act_backward(cc: &str, dir: &Path) {
    for n in [1usize << 20, 1usize << 24] {
        bench_act_backward_n(cc, dir, n);
    }
}

fn bench_act_backward_n(cc: &str, dir: &Path, n: usize) {
    // A realistic pre-activation range [-8, 8) and a small varying upstream gradient.
    let x: Vec<f32> = (0..n)
        .map(|i| (i as f32 - (n / 2) as f32) * (16.0 / n as f32))
        .collect();
    let dy: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let mut dx = vec![0.0f32; n];
    let (xp, dyp) = (x.as_ptr(), dy.as_ptr());
    let bytes = 3.0 * n as f64 * 4.0; // x read + dy read + dx write
    let gbps = |v: &Option<Measure>| {
        v.as_ref()
            .map(|m| format!("{:.1}", bytes / m.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    for op in ["silu", "gelu", "sigmoid", "tanh", "elu", "softplus"] {
        println!(
            "=== {op}_backward (dx = dy·{op}'(x)) N=2^{} (GB/s, higher is better) ===",
            n.trailing_zeros()
        );
        let built_wuk = build_wukong(&wk_act_backward(n, op, false), 3);
        let built_wk_par = build_wukong(&wk_act_backward(n, op, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "act_backward",
            &c_act_backward(n, op),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_act_backward(n, op),
            dir,
            "act_backward",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::wk(built_wk_par),
            Built::peer(built_cm),
            Built::peer(built_cm_cpp),
            Built::peer(built_rm),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut dx, xp, dyp))
                .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked("act_backward", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&cm_cpp),
            gbps(&rm)
        );
        // Same magnitude-normalized check as softmax_bwd: the poly sigmoid/tanh differs from libm by
        // ~1 ULP, and act'(x) has zeros where a per-element relative error is meaningless.
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m
                .out
                .iter()
                .zip(&c2.out)
                .fold(0.0f32, |a, (&p, &q)| a.max((p - q).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > 1e-3 {
                println!("  ! {op}_backward mismatch vs C: max|Δ|/max|C| = {rel:.2e}");
            }
        }
        if let (Some(ms), Some(c2)) = (&wuk, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Wukong single-core (256-bit) is {:.2}x {} than scalar C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            par_standing("scalar single-threaded C", r2);
        }
        report_cpp_ratio(&wuk, &wk_par, &cm_cpp);
        println!();
    }
}

/// Wukong activation backward: the elementwise loop the recognizer folds to one
/// `wukong_vmath2_f32(x, dy, dx, n, VM2_*_BWD)` call. The harness's `(x, y, out)` carry `(x, dy, dx)`.
fn wk_act_backward(n: usize, op: &str, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], dy: [f32; {n}], mut dx: [f32; {n}]) {{\n\
         \x20   for i in 0..{n} {{ dx[i] = {op}_backward(x[i], dy[i]); }}\n}}\n"
    )
}

/// C reference: the same derivative math written with scalar libm `expf`/`tanhf` — gcc keeps the loop
/// scalar (it cannot vectorize a libm call), which is exactly the wall the 256-bit kernel clears.
fn c_act_backward(n: usize, op: &str) -> String {
    let body = match op {
        "silu" => "float s=1.0f/(1.0f+expf(-v)); float g=s+v*s*(1.0f-s);",
        "sigmoid" => "float s=1.0f/(1.0f+expf(-v)); float g=s*(1.0f-s);",
        "tanh" => "float t=tanhf(v); float g=1.0f-t*t;",
        "elu" => "float g = v>0.0f ? 1.0f : expf(v);",
        "softplus" => "float g=1.0f/(1.0f+expf(-v));",
        _ => {
            "float c0=0.7978845608f,c1=0.044715f; float u=tanhf(c0*(v+c1*v*v*v)); \
              float g=0.5f*(1.0f+u)+0.5f*v*(1.0f-u*u)*c0*(1.0f+3.0f*c1*v*v);"
        }
    };
    format!(
        "#include <math.h>\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ dy, float* __restrict__ dx){{\n\
         \x20 for (long i=0;i<N;i++){{ float v=x[i]; {body} dx[i]=dy[i]*g; }}\n}}\n"
    )
}

fn rust_act_backward(n: usize, op: &str) -> String {
    let body = match op {
        "silu" => "let s=1.0f32/(1.0+(-v).exp()); let g=s+v*s*(1.0-s);",
        "sigmoid" => "let s=1.0f32/(1.0+(-v).exp()); let g=s*(1.0-s);",
        "tanh" => "let t=v.tanh(); let g=1.0f32-t*t;",
        "elu" => "let g=if v>0.0 {1.0f32} else {v.exp()};",
        "softplus" => "let g=1.0f32/(1.0+(-v).exp());",
        _ => {
            "let (c0,c1)=(0.7978845608f32,0.044715f32); let u=(c0*(v+c1*v*v*v)).tanh(); \
              let g=0.5*(1.0+u)+0.5*v*(1.0-u*u)*c0*(1.0+3.0*c1*v*v);"
        }
    };
    format!(
        "const N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, dy:*const f32, dx:*mut f32) {{\n\
         \x20 for i in 0..N {{ let v=*x.add(i); {body} *dx.add(i)=*dy.add(i)*g; }}\n}}\n"
    )
}

/// int8 quantized `nn.Linear` (`C = A·Bᵀ`, `u8` activations × `i8` weights → `i32` accumulator) — the
/// quantized-inference GEMM that QNNPACK/oneDNN exist for. Wukong recognizes the nest and dispatches
/// it to the AVX2 widen+`vpmaddwd` int8 microkernel; C/Rust run the *idiomatic* naive int8 GEMM at
/// `-O3 -march=native` / `-O -Ctarget-cpu=native` — whatever their auto-vectorizers produce is the
/// honest baseline (no hand intrinsics, same as every other kernel here). **Integer arithmetic, so
/// the cross-language check is bit-exact** — a stronger bar than the f32 kernels' tolerance. Reported
/// as int8 GOP/s (2 ops per multiply-accumulate). The inputs stay within `i32` (no overflow at these
/// sizes: max ≈ ns·250·125 ≪ 2³¹), so all three languages must agree exactly — widening the input
/// ranges without re-checking that bound would invalidate the exact-equality gate below.
fn bench_i8gemm(cc: &str, dir: &Path) {
    for ns in [512usize, 1024] {
        let n2 = ns * ns;
        // u8 activations 0..250, i8 weights -125..125 — full-range-ish, and small enough that the
        // K-sum stays well inside i32 (max ≈ ns·250·125 ≪ 2³¹), so the result is overflow-free and
        // identical across languages.
        let a: Vec<u8> = (0..n2).map(|i| (i % 251) as u8).collect();
        let b: Vec<i8> = (0..n2).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let mut c = vec![0i32; n2];
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let ops = 2.0 * (ns as f64).powi(3); // 2 ops per MAC
        let gops = |m: &Option<MeasureI8>| {
            m.as_ref()
                .map(|x| format!("{:.1}", ops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== i8gemm (int8 nn.Linear C=A·Bᵀ, u8×i8→i32) {ns}x{ns} (GOP/s, higher is better) ==="
        );
        let built_wuk = build_wukong(&wk_i8gemm(ns, false), 3);
        let built_wk_par = build_wukong(&wk_i8gemm(ns, true), 3);
        let (built_cm, built_cm_cpp) = build_c_cpp(
            "i8gemm",
            &c_i8gemm(ns),
            dir,
            cc,
            &["-O3", "-march=native", "-shared"],
        );
        let built_rm = build_peer(
            "rs",
            &rust_i8gemm(ns),
            dir,
            "i8gemm",
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            BuiltI8::Wukong(built_wuk.map(Box::new)),
            BuiltI8::Wukong(built_wk_par.map(Box::new)),
            BuiltI8::Peer(built_cm),
            BuiltI8::Peer(built_cm_cpp),
            BuiltI8::Peer(built_rm),
        ];
        let mut timed = abba_min_columns(columns.len(), |i| {
            time_built_i8(&columns[i], &mut c, ap, bp)
        })
        .into_iter();
        let wuk = timed.next().flatten();
        let wk_par = timed.next().flatten();
        let cm = timed.next().flatten();
        let cm_cpp = cpp_checked_i8("i8gemm", &cm, timed.next().flatten());
        let rm = timed.next().flatten();

        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GOP/s",
            gops(&wuk),
            gops(&wk_par),
            gops(&cm),
            gops(&cm_cpp),
            gops(&rm)
        );
        // Compile time (Wukong front-end + JIT vs gcc/rustc to a shared lib) — Wukong is far cheaper.
        let cms = |m: &Option<MeasureI8>| {
            m.as_ref()
                .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "compile ms",
            cms(&wuk),
            cms(&wk_par),
            cms(&cm),
            cms(&cm_cpp),
            cms(&rm)
        );
        // Integer arithmetic → exact equality is the correct cross-language bar. Every available
        // result must match the single-core Wukong reference exactly; a mismatch is a real bug.
        if let Some(m) = &wuk {
            for (lang, other) in [
                ("Wuk(par)", &wk_par),
                ("C", &cm),
                ("C++", &cm_cpp),
                ("Rust", &rm),
            ] {
                if let Some(o) = other {
                    if o.out != m.out {
                        let at = m
                            .out
                            .iter()
                            .zip(&o.out)
                            .position(|(x, y)| x != y)
                            .unwrap_or(0);
                        println!(
                            "  ! full-buffer mismatch {lang} vs Wuk(1core) at [{at}]: {} vs {}",
                            o.out[at], m.out[at]
                        );
                    }
                }
            }
        }
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let r = (ops / m.ns_per_call) / (ops / c2.ns_per_call);
            println!(
                "  -> Wukong (1 core) is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = (ops / mp.ns_per_call) / (ops / c2.ns_per_call);
            println!(
                "  -> Wukong @parallel is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        // The C++ (g++) column, printed alongside — the same int8 nest through the other front end.
        if let (Some(m), Some(c2)) = (&wuk, &cm_cpp) {
            let r = c2.ns_per_call / m.ns_per_call;
            println!(
                "  -> Wukong (1 core) is {:.2}x {} than idiomatic C++ (g++)",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm_cpp) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C++ (g++)", r);
        }
        println!();
    }
}

/// Wukong int8 `nn.Linear`, the idiomatic `ijk` dot-product `C = A·Bᵀ` with `u8`/`i8` operands cast
/// to `i32` before the multiply — exactly what the `mir_build` recognizer folds to one
/// `wukong_i8gemm_nt[_parallel]` call.
fn wk_i8gemm(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [u8; {n2}], b: [i8; {n2}], mut c: [i32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{\n\
         \x20           let mut s: i32 = 0;\n\
         \x20           for k in 0..{ns} {{ s = s + (a[i * {ns} + k] as i32) * (b[j * {ns} + k] as i32); }}\n\
         \x20           c[i * {ns} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_i8gemm(ns: usize) -> String {
    format!(
        "#include <stdint.h>\n#define NS {ns}\n\
         __declspec(dllexport) void kbench(const uint8_t* __restrict__ a, const int8_t* __restrict__ b, int32_t* __restrict__ c) {{\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++){{ int32_t s=0;\n\
         \x20     for (long k=0;k<NS;k++) s += (int32_t)a[i*NS+k] * (int32_t)b[j*NS+k];\n\
         \x20     c[i*NS+j]=s; }}\n}}\n"
    )
}

fn rust_i8gemm(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const u8, b:*const i8, c:*mut i32) {{\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{ let mut s: i32 = 0;\n\
         \x20     for k in 0..NS {{ s += *a.add(i*NS+k) as i32 * *b.add(j*NS+k) as i32; }}\n\
         \x20     *c.add(i*NS+j)=s; }} }}\n}}\n"
    )
}

/// Round an `f32` to bf16 (round-to-nearest-even), returning the 16 stored bits — the same arithmetic
/// `wukong_runtime::f32_to_bf16_bits` uses, so the benchmark data matches what the compiler stores.
/// (Benchmark inputs are finite, so the NaN case the runtime handles is irrelevant here.)
fn to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let bias = 0x0000_7fff + ((b >> 16) & 1);
    ((b + bias) >> 16) as u16
}

/// Widen bf16 stored bits back to f32 (the 16 bits are the high half of the f32; low half is zero).
fn widen_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// bf16 **mixed-precision reductions** (bf16 storage, f32 accumulate — the standard ML contract):
/// dot `Σ x·y` and unary sum `Σ x` over `[bf16; N]` arrays. Wukong folds the loop to one
/// `wukong_dot_bf16` / `wukong_sum_bf16` SIMD kernel (F16C-class widen + 8-lane f32 accumulate);
/// C/Rust run the idiomatic `<<16` widen + accumulate at honest default flags (no `-ffast-math`, so
/// their f32 reductions stay sequential — the same basis as the `dot` kernel). The payoff is
/// **bandwidth**: bf16 moves half the bytes of f32, so the win grows once the working set spills L3.
/// Two sizes bracket that: 1<<20 (L2/L3-resident) and 1<<24 (32 MB of bf16 ≫ L3). All-positive inputs
/// keep the reduction well-conditioned, so the cross-language scalar check is a tight tolerance.
fn bench_bf16(cc: &str, dir: &Path) {
    for nbits in [20u32, 24] {
        let n = 1usize << nbits;
        // bf16 stored bits of small positive values (well-conditioned f32 accumulation).
        let x: Vec<u16> = (0..n)
            .map(|i| to_bf16_bits((i as f32 % 17.0) * 0.05 + 0.5))
            .collect();
        let y: Vec<u16> = (0..n)
            .map(|i| to_bf16_bits((i as f32 % 13.0) * 0.03 + 0.25))
            .collect();
        let mut o = vec![0.0f32; 1];
        let (xp, yp) = (x.as_ptr(), y.as_ptr());
        println!("=== bf16 reductions (bf16 in, f32 accumulate) N=2^{nbits} (GB/s, higher is better) ===");
        for (kind, is_dot) in [("dot Σx·y", true), ("sum Σx", false)] {
            let streams = if is_dot { 2 } else { 1 };
            let bytes = (streams * n * 2) as f64; // bf16 input traffic
            let gbps = |m: &Option<MeasureBf16>| {
                m.as_ref()
                    .map(|x| format!("{:.1}", bytes / x.ns_per_call)) // bytes/ns == GB/s
                    .unwrap_or_else(|| "n/a".into())
            };
            // Compile all five columns before visiting any timing slot.
            let wk = build_wukong(&wk_bf16(n, is_dot), 3);
            let (cl, pl) = build_c_cpp(
                "bf16",
                &c_bf16(n, is_dot),
                dir,
                cc,
                &["-O3", "-march=native", "-shared"],
            );
            let fl = build_peer("c", &c_bf16(n, is_dot), dir, "bf16_fast", cc, C_FAST_FLAGS);
            let rl = build_peer(
                "rs",
                &rust_bf16(n, is_dot),
                dir,
                "bf16",
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            let columns = [
                BuiltBf16::Wukong(wk.map(Box::new)),
                BuiltBf16::Peer(cl),
                BuiltBf16::Peer(pl),
                BuiltBf16::Peer(fl),
                BuiltBf16::Peer(rl),
            ];
            let mut timed = abba_min_columns(columns.len(), |i| {
                time_built_bf16(&columns[i], &mut o, xp, yp)
            })
            .into_iter();
            let wuk = timed.next().flatten();
            let cm = timed.next().flatten();
            let cm_cpp = cpp_checked_bf16("bf16", &cm, timed.next().flatten());
            let cfast = timed
                .next()
                .flatten()
                .and_then(|cf| bf16_fast_checked("bf16", wuk.as_ref()?, cf));
            let rm = timed.next().flatten();
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                kind, "Wukong", "C (gcc)", "C++ (g++)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&cm),
                gbps(&cm_cpp),
                gbps(&cfast),
                gbps(&rm)
            );
            // All-positive, well-conditioned reduction → a tight relative tolerance is the right
            // cross-language bar (the three reassociate the f32 sum differently: Wukong 8-lane SIMD
            // vs sequential C/Rust). Report the scalars so any drift is visible.
            //
            // C(fast) IS IN THIS LOOP AS OF 2026-09-13, AT ITS OWN BAR. Every bf16 loss in this
            // section is against C(fast) and nothing else, and it was the one peer whose scalar was
            // never printed anywhere: `bf16_fast_checked` keeps or drops the column at rel > 1e-2, a
            // bar ten times looser than the one the three sequential peers are held to, so "C(fast)
            // printed a column" bounded its answer only to within 1e-2 of Wukong - an error of the
            // same order the sequential peers' own drift reaches. The section's accuracy ruling was
            // derived from the three peers Wukong BEATS and said nothing about the one it loses to.
            // The bar is NOT tightened here (that would delete the peer that wins, which is the
            // opposite of a fix); the scalar is put on the record so the ruling can be re-derived
            // against the f64 reference with C(fast) in it.
            if let Some(m) = &wuk {
                for (lang, other, bar) in [
                    ("C", &cm, 1e-3),
                    ("C++", &cm_cpp, 1e-3),
                    ("Rust", &rm, 1e-3),
                    ("C(fast)", &cfast, 1e-2),
                ] {
                    if let Some(o2) = other {
                        let (a, b) = (m.out, o2.out);
                        let rel = (a - b).abs() as f64 / (a.abs().max(b.abs()).max(1e-6) as f64);
                        println!(
                            "  . {lang} scalar {b} vs Wukong {a} (rel {rel:.2e}, bar {bar:.0e})"
                        );
                        if rel > bar {
                            println!("  ! {lang} scalar drift: {b} vs Wukong {a} (rel {rel:.2e})");
                        }
                    }
                }
            }
            if let (Some(m), Some(c2)) = (&wuk, &cm) {
                let r = c2.ns_per_call / m.ns_per_call;
                println!(
                    "  -> Wukong is {:.2}x {} than idiomatic single-threaded C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            bf16_relaxed_ratio("idiomatic C++ (g++)", &wuk, &None, &cm_cpp);
            bf16_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &None, &cfast);
            // Compile time (Wukong front-end + JIT vs gcc/g++/rustc to a shared lib).
            let cms = |m: &Option<MeasureBf16>| {
                m.as_ref()
                    .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
                    .unwrap_or_else(|| "n/a".into())
            };
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "compile ms",
                cms(&wuk),
                cms(&cm),
                cms(&cm_cpp),
                cms(&cfast),
                cms(&rm)
            );
        }
        println!();
    }
}

/// Wukong bf16 reduction: `s += (x[k] as f32) [* (y[k] as f32)]` — exactly what the `mir_build`
/// recognizer folds to one `wukong_dot_bf16` / `wukong_sum_bf16` call (bf16 storage, f32 accumulate).
fn wk_bf16(n: usize, is_dot: bool) -> String {
    let term = if is_dot {
        "(x[k] as f32) * (y[k] as f32)"
    } else {
        "(x[k] as f32)"
    };
    format!(
        "module bench\nfn kbench(x: [bf16; {n}], y: [bf16; {n}], mut o: [f32; 1]) {{\n\
         \x20   let mut s: f32 = 0.0;\n\
         \x20   for k in 0..{n} {{ s = s + {term}; }}\n\
         \x20   o[0] = s;\n}}\n"
    )
}

/// Idiomatic C bf16 reduction: widen the 16 stored bits to f32 (`<<16` into the high half) and
/// accumulate. gcc `-O3 -march=native` is free to vectorize the widen; the f32 reduction stays
/// sequential at default flags (no `-ffast-math`), the honest baseline.
fn c_bf16(n: usize, is_dot: bool) -> String {
    let term = if is_dot {
        "bf(x[k]) * bf(y[k])"
    } else {
        "bf(x[k])"
    };
    format!(
        "#include <stdint.h>\n#include <string.h>\n#define N {n}\n\
         static inline float bf(uint16_t b){{ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }}\n\
         __declspec(dllexport) void kbench(const uint16_t* __restrict__ x, const uint16_t* __restrict__ y, float* __restrict__ o){{\n\
         \x20   float s=0.0f;\n\
         \x20   for (long k=0;k<N;k++) s += {term};\n\
         \x20   o[0]=s;\n}}\n"
    )
}

/// Idiomatic Rust bf16 reduction (same `<<16` widen; rustc `-Copt-level=3 -Ctarget-cpu=native`, no contraction).
fn rust_bf16(n: usize, is_dot: bool) -> String {
    let term = if is_dot {
        "bf(*x.add(k)) * bf(*y.add(k))"
    } else {
        "bf(*x.add(k))"
    };
    let yname = if is_dot { "y" } else { "_y" }; // y is unused for the unary sum
    format!(
        "#[inline(always)]\nfn bf(b: u16) -> f32 {{ f32::from_bits((b as u32) << 16) }}\n\
         #[no_mangle]\npub extern \"C\" fn kbench(x: *const u16, {yname}: *const u16, o: *mut f32) {{ unsafe {{\n\
         \x20   let mut s = 0.0f32;\n\
         \x20   for k in 0..{n} {{ s += {term}; }}\n\
         \x20   *o = s;\n}} }}\n"
    )
}

/// **All-half streaming axpby** `out[k] = (a·(x[k] as f32) + b·(y[k] as f32)) as bf16` — bf16 in AND a
/// bf16 (narrowing) store. Two same-run levers (the only honest instrument on this throttling laptop):
///   1. **Write-traffic halving**: vs the bf16-in / **f32-out** axpby (same kernel family, wider store)
///      the half output moves 6 bytes/elem instead of 8 (read x,y bf16 = 4 + write 2 vs + write 4), so
///      on a >L3, store-bound stream it runs faster per call. Reported as the ns ratio (each on its own
///      traffic for GB/s).
///   2. **vs a C all-half peer** that does the identical `<<16` widen + round-to-nearest-even bf16
///      narrow: gcc/rustc leave the bf16 round scalar, so Wukong's 256-bit `narrow_bf16` (pack +
///      permute) wins. Cross-checked full-buffer at a bf16-scale tolerance (Wukong FMAs the sum, C
///      does not, so the last bf16 bit can differ).
///
/// N = 1<<24 (32 MB per bf16 array ≫ L3) so the narrowing store dominates.
fn bench_axpby_half_out(cc: &str, dir: &Path) {
    let n = 1usize << 24;
    let (a, b) = (1.5f32, 2.0f32);
    // bf16-stored small positive inputs (well-conditioned).
    let x: Vec<u16> = (0..n)
        .map(|i| to_bf16_bits((i as f32 % 17.0) * 0.05 + 0.5))
        .collect();
    let y: Vec<u16> = (0..n)
        .map(|i| to_bf16_bits((i as f32 % 13.0) * 0.03 + 0.25))
        .collect();
    let mut oh = vec![0u16; n]; // bf16 output buffer
    let mut of = vec![0.0f32; n]; // f32 output buffer (for the write-halving A/B)
    let (xp, yp) = (x.as_ptr(), y.as_ptr());

    println!(
        "=== all-half axpby out=(a·x+b·y) as bf16, N=2^24 (bf16 in AND out; GB/s, higher is better) ==="
    );
    // Lever 1: the half-out kernel vs the f32-out kernel (same math, wider store) — Wukong vs Wukong.
    let built_half = build_wukong(&wk_axpby_half_out(n, a, b), 3);
    let built_f32 = build_wukong(&wk_axpby_f32_out(n, a, b), 3);
    let (cl, pl) = build_c_cpp(
        "axpbyhalf",
        &c_axpby_half_out(n, a, b),
        dir,
        cc,
        &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
    );
    let rl = build_peer(
        "rs",
        &rust_axpby_half_out(n, a, b),
        dir,
        "axpbyhalf",
        "rustc",
        &[
            "-Copt-level=3",
            "-Ctarget-cpu=native",
            "--crate-type=cdylib",
        ],
    );
    // Both output representations participate in one schedule. Each timer derives its
    // output pointer from the corresponding mutable slice and retains its original snapshot.
    let columns = [
        BuiltHalfExperiment::HalfWukong(built_half.map(Box::new)),
        BuiltHalfExperiment::FloatWukong(built_f32.map(Box::new)),
        BuiltHalfExperiment::HalfPeer(cl),
        BuiltHalfExperiment::HalfPeer(pl),
        BuiltHalfExperiment::HalfPeer(rl),
    ];
    let mut timed = abba_min_columns(columns.len(), |i| {
        time_built_half_experiment(&columns[i], &mut oh, &mut of, xp, yp)
    })
    .into_iter();
    let wk_half = timed.next().flatten().map(HalfExperimentMeasure::half);
    let wk_f32 = timed.next().flatten().map(HalfExperimentMeasure::float);
    let cm = timed.next().flatten().map(HalfExperimentMeasure::half);
    let cm_cpp = cpp_checked_halfout(
        "axpbyhalf",
        &cm,
        timed.next().flatten().map(HalfExperimentMeasure::half),
    );
    let rm = timed.next().flatten().map(HalfExperimentMeasure::half);
    // GB/s: half-out moves 6·n bytes (read x,y bf16 + write bf16); f32-out moves 8·n (write f32).
    let half_bytes = 6.0 * n as f64;
    let f32_bytes = 8.0 * n as f64;
    let gbps = |m: &Option<MeasureHalfOut>, bytes: f64| {
        m.as_ref()
            .map(|x| format!("{:.1}", bytes / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    let gbps_bf = |m: &Option<MeasureBf16>, bytes: f64| {
        m.as_ref()
            .map(|x| format!("{:.1}", bytes / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    println!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "", "Wuk half-out", "Wuk f32-out", "C half-out", "C++ half-out", "Rust half-out"
    );
    println!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "GB/s",
        gbps(&wk_half, half_bytes),
        gbps_bf(&wk_f32, f32_bytes),
        gbps(&cm, half_bytes),
        gbps(&cm_cpp, half_bytes),
        gbps(&rm, half_bytes)
    );
    // Cross-check the C/C++/Rust half output against Wukong's (bf16-scale tolerance: 8-bit mantissa ≈ 4e-3).
    if let Some(m) = &wk_half {
        for (lang, other) in [("C", &cm), ("C++", &cm_cpp), ("Rust", &rm)] {
            if let Some(o2) = other {
                let (rel, at) = max_rel_err(&m.out, &o2.out);
                if rel > 8e-3 {
                    println!(
                        "  ! {lang} half-out drift: rel {rel:.2e} at [{at}] ({} vs Wukong {})",
                        o2.out[at], m.out[at]
                    );
                }
            }
        }
    }
    // Lever 1 payoff: the halved write should make the half-out kernel faster per call than f32-out.
    if let (Some(h), Some(f)) = (&wk_half, &wk_f32) {
        let r = f.ns_per_call / h.ns_per_call;
        println!(
            "  -> half-out is {r:.2}x the f32-out axpby (narrowing store halves the write traffic)"
        );
    }
    // Lever 2 payoff: vs the C and C++ all-half peers (same body, gcc vs g++).
    for (lang, peer) in [("C", &cm), ("C++ (g++)", &cm_cpp)] {
        if let (Some(h), Some(c2)) = (&wk_half, peer) {
            let r = c2.ns_per_call / h.ns_per_call;
            println!(
                "  -> Wukong half-out is {:.2}x {} than {lang} (vectorized widen+narrow vs scalar bf16 round)",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
    }
    // Compile time (Wukong front-end + JIT vs gcc/rustc to a shared lib).
    let cms = |m: &Option<MeasureHalfOut>| {
        m.as_ref()
            .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "n/a".into())
    };
    println!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12}",
        "compile ms",
        cms(&wk_half),
        "-",
        cms(&cm),
        cms(&rm)
    );
    println!();
}

/// Wukong all-half axpby: bf16 in AND out — the recognizer folds it to `wukong_axpby_bf16_out`.
fn wk_axpby_half_out(n: usize, a: f32, b: f32) -> String {
    format!(
        "module bench\nfn kbench(x: [bf16; {n}], y: [bf16; {n}], mut out: [bf16; {n}]) {{\n\
         \x20   for k in 0..{n} {{ out[k] = ({a:?} * (x[k] as f32) + {b:?} * (y[k] as f32)) as bf16; }}\n}}\n"
    )
}

/// Wukong bf16-in / f32-out axpby (the wider-store sibling) — folds to `wukong_axpby_bf16`.
fn wk_axpby_f32_out(n: usize, a: f32, b: f32) -> String {
    format!(
        "module bench\nfn kbench(x: [bf16; {n}], y: [bf16; {n}], mut out: [f32; {n}]) {{\n\
         \x20   for k in 0..{n} {{ out[k] = {a:?} * (x[k] as f32) + {b:?} * (y[k] as f32); }}\n}}\n"
    )
}

/// Idiomatic C all-half axpby: widen bf16 (`<<16`), compute `a·x+b·y`, round to nearest-even bf16, store.
fn c_axpby_half_out(n: usize, a: f32, b: f32) -> String {
    format!(
        "#include <stdint.h>\n#include <string.h>\n#define N {n}\n\
         static inline float bf(uint16_t b){{ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }}\n\
         static inline uint16_t nb(float f){{ uint32_t u; memcpy(&u,&f,4); uint32_t bias=0x7fffu+((u>>16)&1u); return (uint16_t)((u+bias)>>16); }}\n\
         __declspec(dllexport) void kbench(const uint16_t* __restrict__ x, const uint16_t* __restrict__ y, uint16_t* __restrict__ out){{\n\
         \x20   for (long k=0;k<N;k++) out[k] = nb({a:?}f*bf(x[k]) + {b:?}f*bf(y[k]));\n}}\n"
    )
}

/// Idiomatic Rust all-half axpby (same widen + round-to-nearest-even bf16 narrow).
fn rust_axpby_half_out(n: usize, a: f32, b: f32) -> String {
    format!(
        "#[inline(always)]\nfn bf(b: u16) -> f32 {{ f32::from_bits((b as u32) << 16) }}\n\
         #[inline(always)]\nfn nb(f: f32) -> u16 {{ let u = f.to_bits(); let bias = 0x7fffu32 + ((u>>16)&1); ((u+bias)>>16) as u16 }}\n\
         #[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const u16, y:*const u16, out:*mut u16) {{\n\
         \x20   for k in 0..{n} {{ *out.add(k) = nb({a:?}f32*bf(*x.add(k)) + {b:?}f32*bf(*y.add(k))); }}\n}}\n"
    )
}

/// 2D convolution — the other heavy ML kernel. Wukong lowers it the XLA/cuDNN way: an im2col gather
/// builds a `[Cin·K·K, OH·OW]` column matrix, then the conv is a matmul `Y = W · col` that the
/// recognizer dispatches to the tuned GEMM. C/Rust run the *idiomatic direct* convolution (the
/// six-deep loop nest everyone writes). Same math, cross-checked by checksum; reported as GFLOP/s.
/// Cin=16, 20×20 input, 64 filters of 3×3, stride 1, no pad → 18×18 output.
fn bench_conv(cc: &str, dir: &Path) {
    let (cin, h, cout, k) = (16usize, 20usize, 64usize, 3usize);
    let oh = h - k + 1; // 18
    let (hw, ohw, ckk) = (h * h, oh * oh, cin * k * k);
    let input: Vec<f32> = (0..cin * hw).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
    let weight: Vec<f32> = (0..cout * ckk)
        .map(|i| (i % 5) as f32 * 0.05 - 0.1)
        .collect();
    let mut output = vec![0.0f32; cout * ohw];
    let (ip, wp) = (input.as_ptr(), weight.as_ptr());
    let flops = 2.0 * (cout * ckk * ohw) as f64;

    println!(
        "=== conv2d Cin{cin} {h}x{h} -> {cout}@{oh}x{oh} (3x3); Wukong im2col+GEMM vs direct conv; GFLOP/s ==="
    );
    let gflops = |m: &Option<Measure>| {
        m.as_ref()
            .map(|x| format!("{:.1}", flops / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    let built_wuk = build_wukong(&wk_conv(cin, h, cout, k), 3);
    let (built_cm, built_cm_cpp) = build_c_cpp(
        "conv",
        &c_conv(cin, h, cout, k),
        dir,
        cc,
        &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
    );
    let built_rm = build_peer(
        "rs",
        &rust_conv(cin, h, cout, k),
        dir,
        "conv",
        "rustc",
        &[
            "-Copt-level=3",
            "-Ctarget-cpu=native",
            "--crate-type=cdylib",
        ],
    );
    // Reassociation-normalized peer: the direct convolution's `Cin·K·K`-long accumulation is a float
    // reduction, and Wukong's im2col+GEMM path accumulates it in a blocked (reassociated) order.
    let built_cfast = build_peer(
        "c",
        &c_conv(cin, h, cout, k),
        dir,
        &format!("{}_fast", "conv"),
        cc,
        C_FAST_FLAGS,
    );
    // Every compiled column keeps its slot, including unavailable peers.
    let columns = [
        Built::wk(built_wuk),
        Built::peer(built_cm),
        Built::peer(built_cm_cpp),
        Built::peer(built_rm),
        Built::peer(built_cfast),
    ];
    let mut timed = abba_min_columns(columns.len(), |i| {
        time_built(&columns[i], &mut output, ip, wp)
    })
    .into_iter();
    let wuk = timed.next().flatten();
    let cm = timed.next().flatten();
    let cm_cpp = cpp_checked("conv", &cm, timed.next().flatten());
    let rm = timed.next().flatten();
    let cfast = timed.next().flatten().filter(|p| {
        wuk.as_ref()
            .is_some_and(|m| relaxed_peer_ok("conv", "C(fast)", m, p))
    });

    println!(
        "  {:<18} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "", "Wuk im2col+GEMM", "C (direct)", "C++ (direct)", "C(fast) direct", "Rust (direct)"
    );
    println!(
        "  {:<18} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "GFLOP/s",
        gflops(&wuk),
        gflops(&cm),
        gflops(&cm_cpp),
        gflops(&cfast),
        gflops(&rm)
    );
    if let (Some(m), Some(c)) = (&wuk, &cm) {
        let (rel, at) = max_rel_err(&m.out, &c.out);
        if rel > 1e-3 {
            println!(
                "  ! full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {:.2e})",
                m.out[at], c.out[at], rel
            );
        }
        let r = (flops / m.ns_per_call) / (flops / c.ns_per_call);
        println!(
            "  -> Wukong (im2col+GEMM) is {:.2}x {} than idiomatic direct-convolution C",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    report_cpp_ratio(&wuk, &None, &cm_cpp);
    report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &None, &cfast);
    println!();
}

/// Wukong conv: im2col into a `[Cin·K·K, OH·OW]` scratch, then `Y = W · col` (the matmul recognizer
/// dispatches the second nest to the GEMM kernel). `col` is a function-local array.
fn wk_conv(cin: usize, h: usize, cout: usize, k: usize) -> String {
    let oh = h - k + 1;
    let (hw, ohw, ckk, kk) = (h * h, oh * oh, cin * k * k, k * k);
    format!(
        "module bench\nfn kbench(input: [f32; {inlen}], weight: [f32; {wlen}], mut output: [f32; {olen}]) {{\n\
         \x20   let mut col: [f32; {collen}] = [0.0; {collen}];\n\
         \x20   for oy in 0..{oh} {{ for ox in 0..{oh} {{\n\
         \x20     for ic in 0..{cin} {{ for ky in 0..{k} {{ for kx in 0..{k} {{\n\
         \x20       col[(ic * {kk} + ky * {k} + kx) * {ohw} + (oy * {oh} + ox)] = input[ic * {hw} + (oy + ky) * {h} + (ox + kx)];\n\
         \x20     }} }} }}\n\
         \x20   }} }}\n\
         \x20   for i in 0..{cout} {{ for j in 0..{ohw} {{\n\
         \x20     let mut s: f32 = 0.0;\n\
         \x20     for p in 0..{ckk} {{ s = s + weight[i * {ckk} + p] * col[p * {ohw} + j]; }}\n\
         \x20     output[i * {ohw} + j] = s;\n\
         \x20   }} }}\n}}\n",
        inlen = cin * hw,
        wlen = cout * ckk,
        olen = cout * ohw,
        collen = ckk * ohw,
    )
}

fn c_conv(cin: usize, h: usize, cout: usize, k: usize) -> String {
    let oh = h - k + 1;
    let (hw, ohw, ckk, kk) = (h * h, oh * oh, cin * k * k, k * k);
    format!(
        "__declspec(dllexport) void kbench(const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output) {{\n\
         \x20 for (long oc=0; oc<{cout}; oc++)\n\
         \x20  for (long oy=0; oy<{oh}; oy++)\n\
         \x20   for (long ox=0; ox<{oh}; ox++) {{\n\
         \x20     float s=0.0f;\n\
         \x20     for (long ic=0; ic<{cin}; ic++)\n\
         \x20      for (long ky=0; ky<{k}; ky++)\n\
         \x20       for (long kx=0; kx<{k}; kx++)\n\
         \x20        s += input[ic*{hw} + (oy+ky)*{h} + (ox+kx)] * weight[oc*{ckk} + ic*{kk} + ky*{k} + kx];\n\
         \x20     output[oc*{ohw} + oy*{oh} + ox] = s;\n\
         \x20   }}\n}}\n"
    )
}

fn rust_conv(cin: usize, h: usize, cout: usize, k: usize) -> String {
    let oh = h - k + 1;
    let (hw, ohw, ckk, kk) = (h * h, oh * oh, cin * k * k, k * k);
    format!(
        "#[no_mangle]\npub unsafe extern \"C\" fn kbench(input:*const f32, weight:*const f32, output:*mut f32) {{\n\
         \x20 for oc in 0..{cout} {{ for oy in 0..{oh} {{ for ox in 0..{oh} {{\n\
         \x20   let mut s=0.0f32;\n\
         \x20   for ic in 0..{cin} {{ for ky in 0..{k} {{ for kx in 0..{k} {{\n\
         \x20     s += *input.add(ic*{hw} + (oy+ky)*{h} + (ox+kx)) * *weight.add(oc*{ckk} + ic*{kk} + ky*{k} + kx);\n\
         \x20   }} }} }}\n\
         \x20   *output.add(oc*{ohw} + oy*{oh} + ox) = s;\n\
         \x20 }} }} }}\n}}\n"
    )
}

/// Fused row normalizations — **softmax / LayerNorm / RMSNorm** over one feature row of `cols` f32,
/// the per-token normalization every transformer layer runs. Wukong folds the canonical multi-pass
/// form into one `wukong_norm_f32` call: 256-bit AVX2, a hand-vectorized `exp` for softmax, and
/// *reassociated* lane-accumulator reductions for the mean / variance / sum-of-squares. gcc/rustc at
/// their honest defaults (no `-ffast-math`) auto-vectorize the elementwise passes but keep the float
/// reductions strictly sequential — and call scalar libm `expf` for softmax — so this is the same
/// reduction-vectorization + transcendental story as the `dot`/`exp` kernels, now fused. All three
/// languages are WRITTEN the same way — copy `x`→`out`, then normalize `out` in place — so the
/// cross-language full-buffer check sees the same result. One row = one token's hidden vector; a
/// real `[tokens, hidden]` batch maps the identical kernel per row.
///
/// THEY NO LONGER DO THE SAME WORK, and this comment used to claim they did ("identical work").
/// `wukong_mir_build::try_fuse_copy_norm` folds the copy into the kernel's source operand, so the
/// Wukong program that gets timed here is ONE call over ONE pass while the C/C++/Rust peers still
/// make two passes over the buffer. That is a legitimate compiler win — gcc declined a fusion
/// Wukong performed on the same spelling — and it is exactly why the ratio is what it is; but it
/// is a *compiler* result, not an equal-work kernel comparison, and this is the campaign's largest
/// predicted movement. Publishing that swing under a stale "identical work" claim is the class of
/// error the evaluation's own instrument-failure table exists to catalogue. The dispatch census
/// printed above each table names the kernels the timed program actually called.
/// (`bench_norm_batched` is unaffected: `match_inplace_norm_window` accepts only a FLAT window, so
/// the batched program still copies, and its comment is still true.)
fn bench_norm(cc: &str, dir: &Path) {
    for &cols in &[768usize, 4096] {
        let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 * 0.5 + 1.0).collect();
        // gamma (and, for the affine LayerNorm, beta — reused) for the *_affine variants: a distinct
        // per-column array, the learned scale/shift every real transformer norm carries.
        let gamma: Vec<f32> = (0..cols).map(|i| (i % 11) as f32 * 0.1 + 0.5).collect();
        let mut out = vec![0.0f32; cols];
        let (xp, yp) = (x.as_ptr(), gamma.as_ptr());
        println!(
            "=== fused norms, 1x{cols} feature row (ns/call, lower is better; Wukong → wukong_norm_f32[_affine]) ==="
        );
        println!(
            "  {:<16} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16} {:>16}",
            "",
            "Wukong",
            "C (gcc)",
            "C++ (g++)",
            "C(fast)",
            "Rust",
            "Wuk vs C",
            "vs C++",
            "vs C(fast)"
        );
        for op in [
            "softmax",
            "logsoftmax",
            "layernorm",
            "rmsnorm",
            "l2norm",
            "layernorm_affine",
            "rmsnorm_affine",
        ] {
            // Build all columns first, then time a shared forward/reverse schedule.
            // Missing peers retain fixed slots through Built::Missing.
            let wk = build_wukong(&wk_norm(cols, op), 3);
            let (cl, pl) = build_c_cpp(
                &format!("norm_{op}"),
                &c_norm(cols, op),
                dir,
                cc,
                &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            );
            let rl = build_peer(
                "rs",
                &rust_norm(cols, op),
                dir,
                &format!("norm_{op}"),
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            let fl = build_peer(
                "c",
                &c_norm(cols, op),
                dir,
                &format!("norm_{op}_fast"),
                cc,
                C_FAST_FLAGS,
            );
            let columns = [
                Built::wk(wk),
                Built::peer(cl),
                Built::peer(pl),
                Built::peer(fl),
                Built::peer(rl),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let c = timed.next().flatten();
            let c_cpp = cpp_checked(&format!("norm_{op}"), &c, timed.next().flatten());
            // Preserve the fast peer's numerical gate against the serial Wukong row.
            let cfast = timed.next().flatten().filter(|cf| {
                wuk.as_ref()
                    .is_some_and(|m| relaxed_peer_ok(&format!("norm_{op}"), "C(fast)", m, cf))
            });
            let rust = timed.next().flatten();
            let ns = |m: &Option<Measure>| {
                m.as_ref()
                    .map(|x| format!("{:.0}", x.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let standing_vs = |peer: &Option<Measure>| {
                if let (Some(m), Some(p)) = (&wuk, peer) {
                    let r = p.ns_per_call / m.ns_per_call;
                    format!(
                        "{:.2}x {}",
                        if r >= 1.0 { r } else { 1.0 / r },
                        if r >= 1.0 { "faster" } else { "slower" }
                    )
                } else {
                    "n/a".into()
                }
            };
            println!(
                "  {:<16} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16} {:>16}",
                op,
                ns(&wuk),
                ns(&c),
                ns(&c_cpp),
                ns(&cfast),
                ns(&rust),
                standing_vs(&c),
                standing_vs(&c_cpp),
                standing_vs(&cfast)
            );
            // Cross-language correctness: same normalized row, element by element (f32 tolerance —
            // Wukong reassociates the reductions, C does not, so it is a tight rel-err, not bits).
            if let (Some(m), Some(c)) = (&wuk, &c) {
                let (rel, at) = max_rel_err(&m.out, &c.out);
                if rel > 1e-3 {
                    println!(
                        "  ! {op} full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {:.2e})",
                        m.out[at], c.out[at], rel
                    );
                }
            }
            if let (Some(m), Some(rust)) = (&wuk, &rust) {
                let (rel, at) = max_rel_err(&m.out, &rust.out);
                if rel > 1e-3 {
                    println!(
                        "  ! {op} full-buffer mismatch vs Rust at [{at}]: Wukong={} Rust={} (rel {:.2e})",
                        m.out[at], rust.out[at], rel
                    );
                }
            }
        }
        println!();
    }
}

/// Wukong norm source: copy `x`→`out`, then the canonical in-place multi-pass form the recognizer
/// folds into one `wukong_norm_f32(out, out, 1, cols, eps, op)` call — or, for the `*_affine`
/// variants whose normalize step also applies the per-column `y` (gamma, and for LayerNorm beta too),
/// one `wukong_norm_affine_f32(out, out, y, y|null, 1, cols, eps, op)` call. The `/ {cols}.0` divisor
/// (and the `out[0]` softmax seed) are exactly the spellings the matchers accept, so the kernel fires.
fn wk_norm(cols: usize, op: &str) -> String {
    let body = match op {
        "softmax" => format!(
            "let mut m: f32 = out[0]; \
             for i in 0..{cols} {{ m = fmax(m, out[i]); }} \
             for i in 0..{cols} {{ out[i] = exp(out[i] - m); }} \
             let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i]; }} \
             let inv: f32 = 1.0 / s; \
             for i in 0..{cols} {{ out[i] = out[i] * inv; }}"
        ),
        "logsoftmax" => format!(
            "let mut m: f32 = out[0]; \
             for i in 0..{cols} {{ m = fmax(m, out[i]); }} \
             let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + exp(out[i] - m); }} \
             let ls: f32 = log(s); \
             for i in 0..{cols} {{ out[i] = (out[i] - m) - ls; }}"
        ),
        "layernorm" => format!(
            "let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i]; }} \
             let mean: f32 = s / {cols}.0; \
             let mut v: f32 = 0.0; \
             for i in 0..{cols} {{ v = v + (out[i] - mean) * (out[i] - mean); }} \
             let inv: f32 = rsqrt(v / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[i] = (out[i] - mean) * inv; }}"
        ),
        "layernorm_affine" => format!(
            "let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i]; }} \
             let mean: f32 = s / {cols}.0; \
             let mut v: f32 = 0.0; \
             for i in 0..{cols} {{ v = v + (out[i] - mean) * (out[i] - mean); }} \
             let inv: f32 = rsqrt(v / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[i] = (out[i] - mean) * inv * y[i] + y[i]; }}"
        ),
        "rmsnorm_affine" => format!(
            "let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i] * out[i]; }} \
             let inv: f32 = rsqrt(s / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[i] = out[i] * inv * y[i]; }}"
        ),
        // L2 / unit normalize: RMSNorm's `rsqrt(s / cols + eps)` without the `/ cols` mean divisor.
        "l2norm" => format!(
            "let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i] * out[i]; }} \
             let inv: f32 = rsqrt(s + 0.00001); \
             for i in 0..{cols} {{ out[i] = out[i] * inv; }}"
        ),
        _ => format!(
            "let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i] * out[i]; }} \
             let inv: f32 = rsqrt(s / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[i] = out[i] * inv; }}"
        ),
    };
    format!(
        "module bench\nfn kbench(x: [f32; {cols}], y: [f32; {cols}], mut out: [f32; {cols}]) {{ \
         for c in 0..{cols} {{ out[c] = x[c]; }} {body} }}\n"
    )
}

fn c_norm(cols: usize, op: &str) -> String {
    let body = match op {
        "softmax" => {
            "float m=out[0]; for(long i=0;i<C;i++) if(out[i]>m) m=out[i]; \
             float s=0.0f; for(long i=0;i<C;i++){ out[i]=expf(out[i]-m); s+=out[i]; } \
             float inv=1.0f/s; for(long i=0;i<C;i++) out[i]*=inv;"
        }
        "logsoftmax" => {
            "float m=out[0]; for(long i=0;i<C;i++) if(out[i]>m) m=out[i]; \
             float s=0.0f; for(long i=0;i<C;i++) s+=expf(out[i]-m); \
             float ls=logf(s); for(long i=0;i<C;i++) out[i]=(out[i]-m)-ls;"
        }
        "layernorm" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]; \
             float mean=s/(float)C; float v=0.0f; \
             for(long i=0;i<C;i++){ float d=out[i]-mean; v+=d*d; } \
             float inv=1.0f/sqrtf(v/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]=(out[i]-mean)*inv;"
        }
        "layernorm_affine" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]; \
             float mean=s/(float)C; float v=0.0f; \
             for(long i=0;i<C;i++){ float d=out[i]-mean; v+=d*d; } \
             float inv=1.0f/sqrtf(v/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]=(out[i]-mean)*inv*y[i]+y[i];"
        }
        "rmsnorm_affine" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]*out[i]; \
             float inv=1.0f/sqrtf(s/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]=out[i]*inv*y[i];"
        }
        "l2norm" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]*out[i]; \
             float inv=1.0f/sqrtf(s+1e-5f); \
             for(long i=0;i<C;i++) out[i]*=inv;"
        }
        _ => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]*out[i]; \
             float inv=1.0f/sqrtf(s/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]*=inv;"
        }
    };
    format!(
        "#include <math.h>\n#define C {cols}\n__declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out) {{ \
         for(long i=0;i<C;i++) out[i]=x[i]; {body} }}\n"
    )
}

fn rust_norm(cols: usize, op: &str) -> String {
    let body = match op {
        "softmax" => {
            "let mut m=*out.add(0); for i in 0..C { let v=*out.add(i); if v>m { m=v; } } \
             let mut s=0.0f32; for i in 0..C { let e=(*out.add(i)-m).exp(); *out.add(i)=e; s+=e; } \
             let inv=1.0f32/s; for i in 0..C { *out.add(i)*=inv; }"
        }
        "logsoftmax" => {
            "let mut m=*out.add(0); for i in 0..C { let v=*out.add(i); if v>m { m=v; } } \
             let mut s=0.0f32; for i in 0..C { s+=(*out.add(i)-m).exp(); } \
             let ls=s.ln(); for i in 0..C { *out.add(i)=(*out.add(i)-m)-ls; }"
        }
        "layernorm" => {
            "let mut s=0.0f32; for i in 0..C { s+=*out.add(i); } \
             let mean=s/(C as f32); let mut v=0.0f32; \
             for i in 0..C { let d=*out.add(i)-mean; v+=d*d; } \
             let inv=1.0f32/(v/(C as f32)+1e-5f32).sqrt(); \
             for i in 0..C { *out.add(i)=(*out.add(i)-mean)*inv; }"
        }
        "layernorm_affine" => {
            "let mut s=0.0f32; for i in 0..C { s+=*out.add(i); } \
             let mean=s/(C as f32); let mut v=0.0f32; \
             for i in 0..C { let d=*out.add(i)-mean; v+=d*d; } \
             let inv=1.0f32/(v/(C as f32)+1e-5f32).sqrt(); \
             for i in 0..C { *out.add(i)=(*out.add(i)-mean)*inv * *y.add(i) + *y.add(i); }"
        }
        "rmsnorm_affine" => {
            "let mut s=0.0f32; for i in 0..C { let v=*out.add(i); s+=v*v; } \
             let inv=1.0f32/(s/(C as f32)+1e-5f32).sqrt(); \
             for i in 0..C { *out.add(i)=*out.add(i)*inv * *y.add(i); }"
        }
        "l2norm" => {
            "let mut s=0.0f32; for i in 0..C { let v=*out.add(i); s+=v*v; } \
             let inv=1.0f32/(s+1e-5f32).sqrt(); \
             for i in 0..C { *out.add(i)*=inv; }"
        }
        _ => {
            "let mut s=0.0f32; for i in 0..C { let v=*out.add(i); s+=v*v; } \
             let inv=1.0f32/(s/(C as f32)+1e-5f32).sqrt(); \
             for i in 0..C { *out.add(i)*=inv; }"
        }
    };
    format!(
        "const C: usize = {cols};\n#[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{ \
         for i in 0..C {{ *out.add(i)=*x.add(i); }} {body} }}\n"
    )
}

/// Batched norms (softmax / LayerNorm / RMSNorm) over a `[rows, cols]` activation — the *real*
/// transformer shape (norm over `[batch*seq, hidden]`, one row per token), where `bench_norm`'s single
/// feature row was just one token. Wukong folds the `for r { <norm over out[r*C+i]> }` loop into one
/// `wukong_norm_f32` call (each row a fused single pass); under `@parallel` the independent rows map across cores via
/// `wukong_norm_f32_parallel`. C/Rust are the idiomatic per-row nested loops at honest defaults
/// (sequential float reductions, no `-ffast-math`). All three copy `x`→`out` then normalize in place,
/// so the full-buffer cross-check is valid. The serial row is apples-to-apples (both single-threaded);
/// the `@parallel` row pits Wukong's automatic SIMD+multicore against idiomatic single-threaded C/Rust.
fn bench_norm_batched(cc: &str, dir: &Path) {
    // Two regimes: an L3-resident batch (the fused single-pass *serial* win, both single-threaded), and
    // a batch whose working set far exceeds L3 — memory-bandwidth-bound, where a single core cannot
    // saturate DRAM and `@parallel` (rows across cores) pays off. RMSNorm is memory-bound (read twice,
    // write once), so `@parallel` only helps once the data spills L3 — the same working-set rule as the
    // streaming-elementwise non-temporal dispatch.
    for &(rows, cols) in &[(512usize, 768usize), (4096usize, 4096usize)] {
        let n = rows * cols;
        let mb = (n * 4) as f64 / (1 << 20) as f64;
        let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.5 + 1.0).collect();
        let mut out = vec![0.0f32; n];
        // No affine params here, so the `y` arg is unused; alias it to `x` rather than allocate a buffer.
        let (xp, yp) = (x.as_ptr(), x.as_ptr());
        println!(
            "=== batched norms (softmax / LayerNorm / RMSNorm), {rows}x{cols} = [tokens, hidden], {mb:.1} MB/buffer (ns/call, lower is better; Wukong → wukong_norm_f32[_parallel]) ==="
        );
        println!(
            "  {:<18} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16} {:>16}",
            "",
            "Wukong",
            "C (gcc)",
            "C++ (g++)",
            "C(fast)",
            "Rust",
            "Wuk vs C",
            "vs C++",
            "vs C(fast)"
        );
        for op in ["rmsnorm", "layernorm", "softmax"] {
            // Build all columns first, then time a shared forward/reverse schedule.
            // Missing peers retain fixed slots through Built::Missing.
            let wk = build_wukong(&wk_norm_batched(rows, cols, op, false), 3);
            let wkp = build_wukong(&wk_norm_batched(rows, cols, op, true), 3);
            let (cl, pl) = build_c_cpp(
                "bnorm",
                &c_norm_batched(rows, cols, op),
                dir,
                cc,
                &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            );
            let rl = build_peer(
                "rs",
                &rust_norm_batched(rows, cols, op),
                dir,
                "bnorm",
                "rustc",
                &[
                    "-Copt-level=3",
                    "-Ctarget-cpu=native",
                    "--crate-type=cdylib",
                ],
            );
            let fl = build_peer(
                "c",
                &c_norm_batched(rows, cols, op),
                dir,
                "bnorm_fast",
                cc,
                C_FAST_FLAGS,
            );
            let columns = [
                Built::wk(wk),
                Built::wk(wkp),
                Built::peer(cl),
                Built::peer(pl),
                Built::peer(fl),
                Built::peer(rl),
            ];
            let mut timed =
                abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                    .into_iter();
            let wuk = timed.next().flatten();
            let wuk_par = timed.next().flatten();
            let c = timed.next().flatten();
            let c_cpp = cpp_checked("bnorm", &c, timed.next().flatten());
            // Preserve the fast peer's numerical gate against the serial Wukong row.
            let cfast = timed.next().flatten().filter(|cf| {
                wuk.as_ref()
                    .is_some_and(|m| relaxed_peer_ok(op, "C(fast)", m, cf))
            });
            let rust = timed.next().flatten();
            for (suffix, wuk) in [("", wuk), ("@parallel", wuk_par)] {
                let label = format!("{op}{suffix}");
                let ns = |m: &Option<Measure>| {
                    m.as_ref()
                        .map(|x| format!("{:.0}", x.ns_per_call))
                        .unwrap_or_else(|| "n/a".into())
                };
                let standing_vs = |peer: &Option<Measure>| {
                    if let (Some(m), Some(p)) = (&wuk, peer) {
                        let r = p.ns_per_call / m.ns_per_call;
                        format!(
                            "{:.2}x {}",
                            if r >= 1.0 { r } else { 1.0 / r },
                            if r >= 1.0 { "faster" } else { "slower" }
                        )
                    } else {
                        "n/a".into()
                    }
                };
                println!(
                    "  {:<18} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16} {:>16}",
                    label,
                    ns(&wuk),
                    ns(&c),
                    ns(&c_cpp),
                    ns(&cfast),
                    ns(&rust),
                    standing_vs(&c),
                    standing_vs(&c_cpp),
                    standing_vs(&cfast)
                );
                // Full-buffer cross-check (f32 tolerance: Wukong reassociates the per-row reductions).
                if let (Some(m), Some(c)) = (&wuk, &c) {
                    let (rel, at) = max_rel_err(&m.out, &c.out);
                    if rel > 1e-3 {
                        println!(
                            "  ! {label} full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {:.2e})",
                            m.out[at], c.out[at], rel
                        );
                    }
                }
                if let (Some(m), Some(rust)) = (&wuk, &rust) {
                    let (rel, at) = max_rel_err(&m.out, &rust.out);
                    if rel > 1e-3 {
                        println!(
                            "  ! {label} full-buffer mismatch vs Rust at [{at}]: Wukong={} Rust={} (rel {:.2e})",
                            m.out[at], rust.out[at], rel
                        );
                    }
                }
            }
        }
        println!();
    }
}

/// Wukong batched-norm source for `op` ∈ {rmsnorm, layernorm, softmax}: copy `x`→`out`, then the
/// `for r { <op over out[r*C+i]> }` form the `mir_build` recognizer folds to one `wukong_norm_f32(out,
/// out, R, C, eps, op)` call — or, under `@parallel`, `wukong_norm_f32_parallel` (rows across cores).
/// The `r*{cols}+i` offset, `/{cols}.0` divisor, and softmax's `out[r*C]` row-local max-seed are exactly
/// the spellings `match_batched_norm` accepts (all three norms share the kernel + dispatch path).
fn wk_norm_batched(rows: usize, cols: usize, op: &str, parallel: bool) -> String {
    let n = rows * cols;
    let attr = if parallel { "@parallel\n" } else { "" };
    let body = match op {
        "layernorm" => format!(
            "let mut s: f32 = 0.0; for i in 0..{cols} {{ s = s + out[r*{cols}+i]; }} \
             let mean: f32 = s / {cols}.0; let mut v: f32 = 0.0; \
             for i in 0..{cols} {{ v = v + (out[r*{cols}+i] - mean) * (out[r*{cols}+i] - mean); }} \
             let inv: f32 = rsqrt(v / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[r*{cols}+i] = (out[r*{cols}+i] - mean) * inv; }}"
        ),
        "softmax" => format!(
            "let mut m: f32 = out[r*{cols}]; for i in 0..{cols} {{ m = fmax(m, out[r*{cols}+i]); }} \
             for i in 0..{cols} {{ out[r*{cols}+i] = exp(out[r*{cols}+i] - m); }} \
             let mut s: f32 = 0.0; for i in 0..{cols} {{ s = s + out[r*{cols}+i]; }} \
             let inv: f32 = 1.0 / s; for i in 0..{cols} {{ out[r*{cols}+i] = out[r*{cols}+i] * inv; }}"
        ),
        _ => format!(
            "let mut s: f32 = 0.0; for i in 0..{cols} {{ s = s + out[r*{cols}+i] * out[r*{cols}+i]; }} \
             let inv: f32 = rsqrt(s / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[r*{cols}+i] = out[r*{cols}+i] * inv; }}"
        ),
    };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [f32; {n}]) {{ \
         for c in 0..{n} {{ out[c] = x[c]; }} \
         for r in 0..{rows} {{ {body} }} }}\n"
    )
}

fn c_norm_batched(rows: usize, cols: usize, op: &str) -> String {
    let n = rows * cols;
    // `o = out + r*C` is the row base; the softmax seed `o[0]` is the row's first element.
    let body = match op {
        "layernorm" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=o[i]; float mean=s/(float)C; float v=0.0f; \
             for(long i=0;i<C;i++){ float d=o[i]-mean; v+=d*d; } \
             float inv=1.0f/sqrtf(v/(float)C+1e-5f); for(long i=0;i<C;i++) o[i]=(o[i]-mean)*inv;"
        }
        "softmax" => {
            "float m=o[0]; for(long i=0;i<C;i++) if(o[i]>m) m=o[i]; \
             float s=0.0f; for(long i=0;i<C;i++){ o[i]=expf(o[i]-m); s+=o[i]; } \
             float inv=1.0f/s; for(long i=0;i<C;i++) o[i]*=inv;"
        }
        _ => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=o[i]*o[i]; \
             float inv=1.0f/sqrtf(s/(float)C+1e-5f); for(long i=0;i<C;i++) o[i]*=inv;"
        }
    };
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out) {{ \
         for(long i=0;i<N;i++) out[i]=x[i]; \
         for(long r=0;r<R;r++){{ float* o=out+(long)r*C; {body} }} }}\n"
    )
}

fn rust_norm_batched(rows: usize, cols: usize, op: &str) -> String {
    let n = rows * cols;
    let body = match op {
        "layernorm" => {
            "let mut s=0.0f32; for i in 0..C { s+=*o.add(i); } let mean=s/(C as f32); let mut v=0.0f32; \
             for i in 0..C { let d=*o.add(i)-mean; v+=d*d; } \
             let inv=1.0f32/(v/(C as f32)+1e-5f32).sqrt(); for i in 0..C { *o.add(i)=(*o.add(i)-mean)*inv; }"
        }
        "softmax" => {
            "let mut m=*o.add(0); for i in 0..C { let val=*o.add(i); if val>m { m=val; } } \
             let mut s=0.0f32; for i in 0..C { let e=(*o.add(i)-m).exp(); *o.add(i)=e; s+=e; } \
             let inv=1.0f32/s; for i in 0..C { *o.add(i)*=inv; }"
        }
        _ => {
            "let mut s=0.0f32; for i in 0..C { let val=*o.add(i); s+=val*val; } \
             let inv=1.0f32/(s/(C as f32)+1e-5f32).sqrt(); for i in 0..C { *o.add(i)*=inv; }"
        }
    };
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\nconst N: usize = {n};\n\
         #[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{ \
         for i in 0..N {{ *out.add(i)=*x.add(i); }} \
         for r in 0..R {{ let o=out.add(r*C); {body} }} }}\n"
    )
}

fn report(
    k: &Kernel,
    m: &Option<Measure>,
    c: &Option<Measure>,
    cpp: &Option<Measure>,
    r: &Option<Measure>,
    cfast: &Option<Measure>,
    comp: &Option<Measure>,
) {
    // C(fast) / C(omp) are *additional* peer columns present only where they apply (C(fast) on the
    // reduction-bearing rows: the same C source at -O3 -march=native -ffast-math; C(omp) on the
    // @parallel rows: the same kernel under #pragma omp parallel for). Both were already
    // loose-cross-checked at the caller.
    let has_fast = cfast.is_some();
    let has_omp = comp.is_some();
    let row = |label: &str, f: &dyn Fn(&Measure) -> String| {
        let cell = |x: &Option<Measure>| x.as_ref().map(f).unwrap_or_else(|| "n/a".into());
        let mut line = format!(
            "  {:<14} {:>13} {:>13} {:>13} {:>13}",
            label,
            cell(m),
            cell(c),
            cell(cpp),
            cell(r),
        );
        if has_fast {
            line.push_str(&format!(" {:>13}", cell(cfast)));
        }
        if has_omp {
            line.push_str(&format!(" {:>13}", cell(comp)));
        }
        println!("{line}");
    };
    let mut hdr = format!(
        "  {:<14} {:>13} {:>13} {:>13} {:>13}",
        "", "Wukong", "C (gcc)", "C++ (g++)", "Rust"
    );
    if has_fast {
        hdr.push_str(&format!(" {:>13}", "C(fast)"));
    }
    if has_omp {
        hdr.push_str(&format!(" {:>13}", "C(omp)"));
    }
    println!("{hdr}");
    row("compile (ms)", &|x| {
        format!("{:.1}", x.compile.as_secs_f64() * 1e3)
    });
    row("runtime (ns)", &|x| format!("{:.0}", x.ns_per_call));
    row("GB/s", &|x| {
        format!("{:.1}", k.bytes_per_call as f64 / x.ns_per_call)
    });
    // Cross-check the two gcc-family peers against Wukong element by element (within f32 tol). The
    // Rust column is timed but NOT diffed here, and C(fast)/C(omp) were already loose-checked at the
    // caller (`relaxed_peer_ok`), so this loop covers C and C++ only.
    for (lang, peer) in [("C", c), ("C++", cpp)] {
        if let (Some(m), Some(p)) = (m, peer) {
            let (rel, at) = max_rel_err(&m.out, &p.out);
            if rel > 1e-3 {
                println!(
                    "  ! full-buffer mismatch vs {lang} at [{at}]: Wukong={} {lang}={} (rel {:.2e})",
                    m.out[at], p.out[at], rel
                );
            }
        }
    }
    if let (Some(m), Some(c)) = (m, c) {
        let ratio = c.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong runtime is {:.2}x {} than C; compiles {:.1}x faster",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            if ratio >= 1.0 { "faster" } else { "slower" },
            c.compile.as_secs_f64() / m.compile.as_secs_f64(),
        );
    }
    // C++ (g++) peer: same idiomatic numeric body through the C++ toolchain. g++ shares gcc's
    // middle/back-end, so for a numeric kernel it produces ≈the same code as the C column — the point
    // is to *measure* the "beat C++" claim rather than assume it. The structural wins (vectorized
    // transcendentals, tiled GEMM, reassociated reductions) hold against g++ exactly as against gcc.
    if let (Some(m), Some(cpp)) = (m, cpp) {
        let ratio = cpp.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong runtime is {:.2}x {} than C++ (g++ -O3 -march=native)",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            if ratio >= 1.0 { "faster" } else { "slower" },
        );
    }
    // The reassociation-normalized standing: the same C source, -ffast-math. Reported alongside
    // (never replacing) the honest-flags C ratio above.
    if let (Some(m), Some(cf)) = (m, cfast) {
        let ratio = cf.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong runtime is {:.2}x {} than C(fast) (gcc -O3 -march=native -ffast-math)",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            if ratio >= 1.0 { "faster" } else { "slower" },
        );
    }
    // The multicore-vs-multicore standing: Wukong @parallel vs the OpenMP'd C twin — the
    // apples-to-apples multithreaded comparison for the @parallel rows.
    if let (Some(m), Some(co)) = (m, comp) {
        let ratio = co.ns_per_call / m.ns_per_call;
        println!(
            "  -> Wukong @parallel is {:.2}x {} than C(omp) (gcc -O3 -march=native -fopenmp, all cores)",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            if ratio >= 1.0 { "faster" } else { "slower" },
        );
    }
}

/// Check that the just-lowered `kbench` really has the ABI the caller is about to `transmute` it
/// to, at the exact point where that precondition is otherwise only *hoped* for.
///
/// The harnesses transmute a raw code pointer to a fixed `extern "C"` signature; the guarantee that
/// the JIT'd function matches is a hand-maintained convention spread over ~60 independent source
/// generators in this file. Nothing in the compiler enforces it. Give a generated kernel a fourth
/// buffer while its caller still uses the 3-pointer harness and the code parses, type-checks,
/// lowers and JITs cleanly — then the callee reads its missing pointer argument out of an
/// uninitialized register inside the timed region and dereferences garbage. `Program` carries the
/// signature, so it is checkable right here. Every generated kernel in this file lowers to N
/// pointer parameters and no return value (verified across all five harness families).
fn kbench_abi_ok(
    program: &wukong_mir::Program,
    sym: wukong_span::Symbol,
    ptr_params: usize,
) -> bool {
    let Some(f) = program.function(sym) else {
        eprintln!("wukong ABI error: the lowered module has no `kbench`");
        return false;
    };
    let all_ptr = f
        .params
        .iter()
        .all(|&p| matches!(f.value_type(p), wukong_mir::MirType::Ptr));
    if f.params.len() != ptr_params || !all_ptr || !matches!(f.ret, wukong_mir::MirType::Void) {
        eprintln!(
            "wukong ABI error: kbench lowered to ({}) -> {}, harness expects {ptr_params} \
             pointer(s) -> void",
            f.params
                .iter()
                .map(|&p| f.value_type(p).display())
                .collect::<Vec<_>>()
                .join(", "),
            f.ret.display(),
        );
        return false;
    }
    true
}

/// A Wukong kernel COMPILED and JIT-loaded but **not yet timed** - the Wukong-side twin of
/// [`build_peer`], and the thing that makes [`abba_min_columns`] possible.
///
/// Until 2026-09-13 the Wukong column of every family was compiled and timed in one call, which
/// forced it to be the FIRST timed region of its section (nothing else was built yet) and left
/// every peer's timed region preceded by all-core compiler spawns that Wukong's had never paid.
/// The harness's own control experiment (see [`abba_min_columns`]) measured that slot to be worth up to
/// +17.3% on `colsum`. Splitting build from time is what lets a section build every column first
/// and then time them in a position-balanced rotation, exactly as [`build_c_cpp`] already did for
/// the C-vs-C++ pair and never did for the Wukong-vs-peer pair the checklist states its targets in.
struct WukongJit {
    /// Wall time the whole front end + optimizer + Cranelift JIT took.
    compile: Duration,
    /// `kbench`'s entry point. Valid only while `_handle` is alive, which is why the two travel
    /// together in one value.
    ptr: *const u8,
    /// The dispatch census line. Re-printed beside the main table's `wk(gen)` row; other families
    /// read it only through the `. dispatch` line [`announce_dispatch`] already printed.
    dispatch: String,
    _handle: wukong_codegen_cranelift::JitModuleHandle,
}

/// Compile `src` through the whole Wukong pipeline at `-O3` and JIT it, checking that the lowered
/// `kbench` really takes `ptr_params` pointers and returns void before anything transmutes it.
///
/// Also records the dispatch census and persists the generated source - see [`announce_dispatch`].
fn build_wukong(src: &str, ptr_params: usize) -> Option<WukongJit> {
    build_wukong_lowered(src, ptr_params, true)
}

/// [`build_wukong`] with every kernel recognizer OFF: the `WUKONG_NO_RECOGNIZE` / `@generic`
/// lowering, reached through `wukong_mir_build::LowerOptions { recognize: false }` - the same knob
/// the driver sets from the environment. The loop nest is lowered as written (scalar or
/// autovectorized, never a recognized `wukong_*` kernel call), which makes this the only Wukong
/// column a *language* ratio may be read from (master plan §7.2); the recognizer-on column
/// measures the dispatcher. Its dispatch census is announced with a `(gen)` tag.
fn build_wukong_gen(src: &str, ptr_params: usize) -> Option<WukongJit> {
    build_wukong_lowered(src, ptr_params, false)
}

fn build_wukong_lowered(src: &str, ptr_params: usize, recognize: bool) -> Option<WukongJit> {
    let t = Instant::now();
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("wukong parse error");
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("wukong sema error: {sd:?}");
        return None;
    }
    let opts = wukong_mir_build::LowerOptions {
        recognize,
        ..Default::default()
    };
    let (mut program, ld) =
        wukong_mir_build::lower_program_with(&module, &sema, &mut interner, opts);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let dispatch = announce_dispatch(
        src,
        &program,
        &interner,
        if recognize { "" } else { "(gen)" },
    );
    let handle = match wukong_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("wukong codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    if !kbench_abi_ok(&program, sym, ptr_params) {
        return None;
    }
    let ptr = handle.func_ptr(sym)?;
    Some(WukongJit {
        compile: t.elapsed(),
        ptr,
        dispatch,
        _handle: handle,
    })
}

/// Time an already-built Wukong kernel over the 3-pointer ABI. Derives the kernel's output pointer
/// from `out` - see the one-live-pointer law on [`time_wukong`].
/// ONE-LIVE-POINTER LAW (every harness in this file obeys it). The kernel's output pointer is
/// derived from `out` *here*, not handed in by the caller. A caller that hoisted `op =
/// buf.as_mut_ptr()` and then passed `&mut buf` alongside it would be lying to the optimizer:
/// `&mut [f32]` lowers to a `noalias` parameter, which asserts that nothing else reaching this
/// function touches that memory — yet the JIT'd kernel writes through exactly such a pointer. LLVM
/// would then be entitled to forward the zeroing stores across the opaque call and lower the
/// `to_vec()` read-back to a memset, so every cross-language check in the suite would compare
/// all-zeros against all-zeros and pass vacuously. Deriving `op` from `out` keeps the kernel's
/// writes *based on* the parameter, which is exactly what `noalias` permits.
fn time_wukong(
    jit: &WukongJit,
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
) -> Option<Measure> {
    // SAFETY: `build_wukong` checked the lowered `kbench` is (ptr, ptr, ptr) -> void, which is
    // exactly `KernelFn`; `jit` owns the JIT module and outlives this call.
    let f: KernelFn = unsafe { std::mem::transmute(jit.ptr) };
    out.iter_mut().for_each(|v| *v = 0.0);
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    Some(Measure {
        compile: jit.compile,
        ns_per_call: ns,
        out: out.to_vec(),
    })
}

/// The `wukong_*` runtime-kernel CALL TARGETS in a printed MIR module, as a sorted multiset.
///
/// Same shape as `general::kernel_census` / `model::kernel_calls`, which have printed this for
/// their own programs since those suites landed. `main.rs` printed nothing, which is why no
/// campaign-measured row could say whether a movement came from a recognizer firing or from the
/// JIT: recognized kernels are gate-blind (interp and native both call the same symbol at every
/// `-O` level), so the dispatch census of the program that was timed is the only evidence there is.
///
/// CALL SITES ONLY. This scanned for the bare substring `wukong_` anywhere in the text, which
/// counts anything the printer might put a kernel name in that is not a dispatch — a `func_addr`
/// operand (`wukong_mir::print` emits one for every taken function address), a function whose own
/// name begins `wukong_`, a type or a comment. A census that over-counts is worse than none: the
/// dispatch line is the only attribution a measured row carries, so it has to name what was
/// CALLED. `wukong_mir::print::fmt_inst` renders a call as `call <name>(<args>)` and nothing else
/// in the printer produces that shape, so anchoring on `call ` + name + `(` is exact rather than
/// heuristic. `kernel_census_counts_real_printed_mir` pins it against real printer output rather
/// than against a string the test wrote itself.
fn kernel_census(mir: &str) -> Vec<(String, usize)> {
    const CALL: &str = "call ";
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (i, _) in mir.match_indices(CALL) {
        let rest = &mir[i + CALL.len()..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let (name, tail) = rest.split_at(end);
        // `call name(` — the argument list is what separates a call from any other use of the name.
        if !tail.starts_with('(') || !name.starts_with("wukong_") {
            continue;
        }
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

/// A content-addressed file name for a generated `.wk` program, so the same source always lands in
/// the same file and a log line names a file a reader can actually open.
fn wk_source_name(src: &str) -> String {
    // FNV-1a 64. Not cryptographic and does not need to be: it only has to be stable across runs
    // and collision-free over the ~100 generated programs one run produces.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("wk_{h:016x}.wk")
}

/// Print the dispatch census for a just-optimized program, write its source next to the peer
/// sources, and return the line.
///
/// TWO defects this closes, both from the 2026-09-13 instrument sweep. (1) No campaign row recorded
/// which recognizer fired, so a measured movement could not be attributed to a dispatch change
/// rather than a codegen change - and 10 of the 21 measured families are recognizer-dependent.
/// (2) The `wk_*` programs were `format!` strings parsed in-process and never persisted, so a
/// measured row could not be re-fed to `wukongc --emit=mir -O2` to check it. The peer sources have
/// been written to this same directory since [`build_peer`] existed; now the Wukong source is too.
///
/// The line is prefixed `  . ` - the harness's existing informational-note marker, which
/// `docs/cpu-evaluation-probes/extract-kernels.py` skips rather than mistaking for a table row.
///
/// `tag` follows the word `dispatch` (`(gen)` for a recognition-off build), so the two lowerings
/// of one generated source can be told apart in the log.
fn announce_dispatch(
    src: &str,
    program: &wukong_mir::Program,
    interner: &Interner,
    tag: &str,
) -> String {
    let census = kernel_census(&wukong_mir::print::print_program(program, interner));
    let name = wk_source_name(src);
    if let Some(dir) = XBENCH_DIR.get() {
        let _ = std::fs::write(dir.join(&name), src);
    }
    let par = if src.contains("@parallel") {
        " [@parallel]"
    } else {
        ""
    };
    let body = if census.is_empty() {
        "NONE - every loop stays generic Cranelift code".to_string()
    } else {
        census
            .iter()
            .map(|(k, n)| {
                if *n == 1 {
                    k.clone()
                } else {
                    format!("{k}x{n}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let line = format!("{name}{par} -> {body}");
    println!("  . dispatch{tag} {line}");
    line
}

/// The scratch directory the peer sources are written to, published once by `main` so
/// [`announce_dispatch`] can drop the generated `.wk` beside them without threading a `&Path`
/// through 40 `bench_*` signatures - the same reasoning [`cxx`] records for `$CXX`.
static XBENCH_DIR: OnceLock<PathBuf> = OnceLock::new();

/// A built-and-loaded peer library, plus the wall time its compiler process took.
///
/// The `Library` must outlive every call through the symbol taken out of it, which is why this is
/// carried around as a value rather than a path: dropping it unloads the DLL.
type PeerLib = (Duration, libloading::Library);

/// Compile `src` to a shared library with `compiler args…` (timed) and LOAD it — everything
/// `bench_external*` does EXCEPT running the kernel.
///
/// Split out on 2026-08-05 so the `bench_c_cpp*` pair helpers can build both peers first and then
/// interleave only the *timed* regions (see [`abba_min_columns`]). Building both up front also means no
/// compiler process spawn — the noisiest thing this harness does, it loads every core — lands
/// between two timed regions that are about to be compared.
fn build_peer(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
) -> Option<PeerLib> {
    // Sanitize the kernel name for use as a filename: rustc derives the crate name from the source
    // file stem and rejects characters like `@` (e.g. `saxpy@parallel`).
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let src_path = dir.join(format!("{safe}.{ext}"));
    let dll: PathBuf = dir.join(format!("{safe}_{ext}.dll"));
    if std::fs::write(&src_path, src).is_err() {
        return None;
    }
    let t = Instant::now();
    let status = Command::new(compiler)
        .args(args)
        .arg("-o")
        .arg(&dll)
        .arg(&src_path)
        .status();
    let compile = t.elapsed();
    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            eprintln!("{compiler} failed to compile {name}.{ext}");
            return None;
        }
        Err(_) => {
            eprintln!("could not run `{compiler}` (skipping)");
            return None;
        }
    }
    // SAFETY: loading a library this process just produced from its own generated source. The
    // caller keeps the returned `Library` alive for as long as it calls through it.
    match unsafe { libloading::Library::new(&dll) } {
        Ok(l) => Some((compile, l)),
        Err(e) => {
            eprintln!("load {}: {e}", dll.display());
            None
        }
    }
}

/// Fetch the `kbench` entry point out of a loaded peer library.
///
/// # Safety
/// `F` must be the fn-pointer type matching the generated kernel's real ABI. The five generators
/// each pass their own (`KernelFn`, `KernelFn4`, `I8KernelFn`, `Bf16KernelFn`, `HalfOutKernelFn`),
/// exactly as the pre-split `bench_external*` bodies did.
unsafe fn peer_kbench<F: Copy>(lib: &libloading::Library) -> Option<F> {
    match lib.get::<F>(b"kbench\0") {
        Ok(s) => Some(*s),
        Err(e) => {
            eprintln!("symbol kbench in a peer library: {e}");
            None
        }
    }
}

/// The one thing [`abba_min_columns`] needs from a measurement: how long a call took.
trait Timed {
    fn ns(&self) -> f64;
}
impl Timed for Measure {
    fn ns(&self) -> f64 {
        self.ns_per_call
    }
}
impl Timed for MeasureI8 {
    fn ns(&self) -> f64 {
        self.ns_per_call
    }
}
impl Timed for MeasureBf16 {
    fn ns(&self) -> f64 {
        self.ns_per_call
    }
}
impl Timed for MeasureHalfOut {
    fn ns(&self) -> f64 {
        self.ns_per_call
    }
}

/// Keep the faster of two samples of the SAME column (the harness's minimum-is-the-estimate rule,
/// already used inside [`time_ns`] across its 14 blocks).
fn faster<M: Timed>(x: Option<M>, y: Option<M>) -> Option<M> {
    match (x, y) {
        (Some(p), Some(q)) => Some(if q.ns() < p.ns() { q } else { p }),
        (p, q) => p.or(q),
    }
}

/// One column of a section, BUILT but not yet timed.
///
/// A column that could not be built stays in the list as [`Built::Missing`] rather than being
/// dropped, so the *positions* the surviving columns occupy do not shift when a toolchain is
/// absent. It times to `None` in every slot.
/// A library peer already resident in the process: given the output buffer, it performs exactly
/// the timed call and returns the measurement.
type DirectColumn<'a> = Box<dyn Fn(&mut [f32]) -> Option<Measure> + 'a>;

enum Built<'a> {
    /// A JIT-compiled Wukong kernel. Boxed because `JitModuleHandle` is large and would otherwise
    /// set the size of every variant of this enum, including `Missing`.
    Wukong(Box<WukongJit>),
    /// A compiled-and-loaded C / C++ / Rust peer DLL.
    Peer(PeerLib),
    /// A library peer already resident in this process (oneMKL). Nothing to build.
    Direct(DirectColumn<'a>),
    /// Toolchain missing, compile failed, or the column does not apply to this section.
    Missing,
}

impl Built<'_> {
    fn wk(j: Option<WukongJit>) -> Self {
        j.map_or(Built::Missing, |j| Built::Wukong(Box::new(j)))
    }
    fn peer(p: Option<PeerLib>) -> Self {
        p.map_or(Built::Missing, Built::Peer)
    }
}

/// Time one already-built column over the 3-pointer ABI.
fn time_built(b: &Built, out: &mut [f32], xp: *const f32, yp: *const f32) -> Option<Measure> {
    match b {
        Built::Wukong(j) => time_wukong(j, out, xp, yp),
        Built::Peer((c, l)) => time_peer(l, *c, out, xp, yp),
        Built::Direct(f) => f(out),
        Built::Missing => None,
    }
}

/// The historical pair rotation generalized to N columns: visit them `0 1 .. n-1` and then
/// `n-1 .. 1 0`, keeping each column's own fastest sample.
///
/// **This fixes the ratio the earlier pair timer never balanced.** Its two-column predecessor
/// removed C-vs-C++ position bias in 2026-08-05 after a byte-identical-source control measured that
/// the second slot cost **+5.0% and +17.3%** on the two `colsum` rows. It was wired into
/// `bench_c_cpp*` only. The Wukong column stayed the FIRST timed region of every section - the
/// coldest, quietest slot - and the Wukong-over-peer ratio is precisely the ratio the campaign
/// checklist states every target in. A 17.3% positional advantage is larger than the entire deficit
/// on more than twenty of the sixty-seven recorded losses, so those starting values were not
/// established and the true gaps are LARGER than they were reported to be, not smaller. The bias
/// pointed our way, which is the only reason it survived a year of reading.
///
/// The schedule is [`round_order`]`(n, 0)` then [`round_order`]`(n, 1)` - a forward pass and its
/// exact reverse. Column `i` therefore sits at position `i` in one pass and `n-1-i` in the other, so
/// every column's positions sum to exactly `n-1`: the balance is arithmetic, not statistical, and is
/// pinned by `round_order_balances_positions_exactly_over_each_pair`. This is the counterbalanced
/// (reversal / Williams-design) ordering that experimental design uses to cancel a monotone
/// carry-over effect, and it is the same remedy Mytkowicz et al. prescribe for measurement bias in
/// ASPLOS 2009 "Producing Wrong Data Without Doing Anything Obviously Wrong!" - randomize or
/// counterbalance the dimension you cannot hold fixed. Here that dimension is the slot.
///
/// EVERY column gets exactly two slots, deliberately. Giving Wukong two and a peer one would trade
/// one bias for a worse one: a minimum over two samples is systematically below a minimum over one.
///
/// The caller must BUILD every column before calling this. That is the second half of the same
/// defect: `bench_wukong` used to compile-and-time in one call, so Wukong's timed region never had a
/// gcc process in front of it while every peer's did - the exact hazard [`build_peer`]'s doc says
/// the build/time split exists to prevent.
fn abba_min_columns<M: Timed>(
    n: usize,
    mut time: impl FnMut(usize) -> Option<M>,
) -> Vec<Option<M>> {
    let mut best: Vec<Option<M>> = Vec::with_capacity(n);
    best.resize_with(n, || None);
    for round in 0..2 {
        for i in round_order(n, round) {
            let sample = time(i);
            let previous = best[i].take();
            best[i] = faster(previous, sample);
        }
    }
    best
}

/// Time an already-loaded 3-pointer peer. Derives the kernel's output pointer from `out` — see the
/// one-live-pointer law on [`time_wukong`].
fn time_peer(
    lib: &libloading::Library,
    compile: Duration,
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
) -> Option<Measure> {
    // SAFETY: the generated C/Rust `kbench` is `(const float*, const float*, float*) -> void`.
    let f: KernelFn = unsafe { peer_kbench(lib)? };
    out.iter_mut().for_each(|v| *v = 0.0);
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot = out.to_vec();
    Some(Measure {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// A 4-pointer kernel ABI `(p0, p1, p2: *const, op: *mut)` — for kernels that read three input buffers
/// and write one output (e.g. the norm backwards `(x, dy, gamma) -> dx`), which don't fit the 3-pointer
/// `KernelFn`. Same timing/snapshot protocol as [`time_wukong`]/[`time_peer`].
type KernelFn4 = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);

/// Time an already-built four-pointer JIT; its owning handle stays alive throughout.
fn time_wukong4(
    jit: &WukongJit,
    out: &mut [f32],
    p0: *const f32,
    p1: *const f32,
    p2: *const f32,
) -> Option<Measure> {
    // SAFETY: callers build this JIT with ptr_params=4, matching KernelFn4.
    let f: KernelFn4 = unsafe { std::mem::transmute(jit.ptr) };
    out.iter_mut().for_each(|v| *v = 0.0);
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(p0, p1, p2, op) });
    Some(Measure {
        compile: jit.compile,
        ns_per_call: ns,
        out: out.to_vec(),
    })
}

/// Own a four-pointer norm JIT and its input pointers until the schedule finishes.
/// The caller must have used build_wukong(src, 4) and keep all inputs alive.
fn built_wukong4(
    jit: Option<WukongJit>,
    p0: *const f32,
    p1: *const f32,
    p2: *const f32,
) -> Built<'static> {
    jit.map_or(Built::Missing, |jit| {
        Built::Direct(Box::new(move |out| time_wukong4(&jit, out, p0, p1, p2)))
    })
}

/// Own a four-pointer norm peer library; absent peers retain their schedule slot.
fn built_peer4(
    peer: Option<PeerLib>,
    p0: *const f32,
    p1: *const f32,
    p2: *const f32,
) -> Built<'static> {
    peer.map_or(Built::Missing, |(compile, lib)| {
        Built::Direct(Box::new(move |out| {
            time_peer4(&lib, compile, out, p0, p1, p2)
        }))
    })
}

/// Time an already-loaded four-pointer peer using the existing timing/snapshot protocol.
fn time_peer4(
    lib: &libloading::Library,
    compile: Duration,
    out: &mut [f32],
    p0: *const f32,
    p1: *const f32,
    p2: *const f32,
) -> Option<Measure> {
    // SAFETY: the generated `kbench` is `(const float*, const float*, const float*, float*)`.
    let f: KernelFn4 = unsafe { peer_kbench(lib)? };
    out.iter_mut().for_each(|v| *v = 0.0);
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(p0, p1, p2, op) });
    let snapshot = out.to_vec();
    Some(Measure {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

fn time_wukong_i8(
    jit: &WukongJit,
    out: &mut [i32],
    ap: *const u8,
    bp: *const i8,
) -> Option<MeasureI8> {
    // SAFETY: build_wukong checked three pointer parameters and void return; generated
    // sources establish pointee types, and jit owns the code through the final snapshot.
    let f: I8KernelFn = unsafe { std::mem::transmute(jit.ptr) };
    out.iter_mut().for_each(|v| *v = 0);
    let cp = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(ap, bp, cp) });
    let snapshot = out.to_vec();
    Some(MeasureI8 {
        compile: jit.compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// Time an already-loaded int8 peer (the [`abba_min_columns`] half of [`time_peer_i8`]).
fn time_peer_i8(
    lib: &libloading::Library,
    compile: Duration,
    out: &mut [i32],
    ap: *const u8,
    bp: *const i8,
) -> Option<MeasureI8> {
    // SAFETY: the generated `kbench` is `(const unsigned char*, const signed char*, int*)`.
    let f: I8KernelFn = unsafe { peer_kbench(lib)? };
    out.iter_mut().for_each(|v| *v = 0);
    let cp = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(ap, bp, cp) });
    let snapshot = out.to_vec();
    Some(MeasureI8 {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// Compiled integer-output columns retain missing peers without casting output storage.
enum BuiltI8 {
    Wukong(Option<Box<WukongJit>>),
    Peer(Option<PeerLib>),
}

fn time_built_i8(
    column: &BuiltI8,
    out: &mut [i32],
    ap: *const u8,
    bp: *const i8,
) -> Option<MeasureI8> {
    match column {
        BuiltI8::Wukong(jit) => time_wukong_i8(jit.as_ref()?, out, ap, bp),
        BuiltI8::Peer(peer) => {
            let (compile, lib) = peer.as_ref()?;
            time_peer_i8(lib, *compile, out, ap, bp)
        }
    }
}

/// Time a built bf16-input/f32-output JIT using its checked three-pointer ABI.
fn time_wukong_bf16(
    jit: &WukongJit,
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureBf16> {
    // SAFETY: build_wukong checked three pointer parameters and void return; the generated
    // bf16 source establishes the pointee types, and jit owns the code throughout timing.
    let f: Bf16KernelFn = unsafe { std::mem::transmute(jit.ptr) };
    out[0] = 0.0;
    let op = out.as_mut_ptr();
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    Some(MeasureBf16 {
        compile: jit.compile,
        ns_per_call: ns,
        out: out[0],
    })
}

/// Typed scalar-output columns keep unavailable bf16 peers in their original slots.
enum BuiltBf16 {
    Wukong(Option<Box<WukongJit>>),
    Peer(Option<PeerLib>),
}

fn time_built_bf16(
    column: &BuiltBf16,
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureBf16> {
    match column {
        BuiltBf16::Wukong(jit) => time_wukong_bf16(jit.as_ref()?, out, xp, yp),
        BuiltBf16::Peer(peer) => {
            let (compile, lib) = peer.as_ref()?;
            time_peer_bf16(lib, *compile, out, xp, yp)
        }
    }
}

/// Time an already-loaded bf16-reduction peer (the [`abba_min_columns`] half of [`time_peer_bf16`]).
fn time_peer_bf16(
    lib: &libloading::Library,
    compile: Duration,
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureBf16> {
    // SAFETY: the generated `kbench` is `(const unsigned short*, const unsigned short*, float*)`.
    let f: Bf16KernelFn = unsafe { peer_kbench(lib)? };
    out[0] = 0.0;
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot = out[0];
    Some(MeasureBf16 {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

fn time_wukong_halfout(
    jit: &WukongJit,
    out: &mut [u16],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureHalfOut> {
    // SAFETY: build_wukong checked three pointer parameters and void return; generated
    // sources establish pointee types, and jit owns the code through the final snapshot.
    let f: HalfOutKernelFn = unsafe { std::mem::transmute(jit.ptr) };
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot: Vec<f32> = out.iter().map(|&b| widen_bf16(b)).collect();
    Some(MeasureHalfOut {
        compile: jit.compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// The half-output experiment compares two output ABIs without casting either buffer.
enum BuiltHalfExperiment {
    HalfWukong(Option<Box<WukongJit>>),
    FloatWukong(Option<Box<WukongJit>>),
    HalfPeer(Option<PeerLib>),
}

enum HalfExperimentMeasure {
    Half(MeasureHalfOut),
    Float(MeasureBf16),
}

impl Timed for HalfExperimentMeasure {
    fn ns(&self) -> f64 {
        match self {
            Self::Half(m) => m.ns(),
            Self::Float(m) => m.ns(),
        }
    }
}

impl HalfExperimentMeasure {
    fn half(self) -> MeasureHalfOut {
        match self {
            Self::Half(m) => m,
            Self::Float(_) => unreachable!("half-output slot produced a float-output measure"),
        }
    }
    fn float(self) -> MeasureBf16 {
        match self {
            Self::Float(m) => m,
            Self::Half(_) => unreachable!("float-output slot produced a half-output measure"),
        }
    }
}

fn time_built_half_experiment(
    column: &BuiltHalfExperiment,
    half: &mut [u16],
    float: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<HalfExperimentMeasure> {
    match column {
        BuiltHalfExperiment::HalfWukong(jit) => {
            time_wukong_halfout(jit.as_ref()?, half, xp, yp).map(HalfExperimentMeasure::Half)
        }
        BuiltHalfExperiment::FloatWukong(jit) => {
            time_wukong_bf16(jit.as_ref()?, float, xp, yp).map(HalfExperimentMeasure::Float)
        }
        BuiltHalfExperiment::HalfPeer(peer) => {
            let (compile, lib) = peer.as_ref()?;
            time_peer_halfout(lib, *compile, half, xp, yp).map(HalfExperimentMeasure::Half)
        }
    }
}

/// Time an already-loaded all-half peer (the [`abba_min_columns`] half of [`time_peer_halfout`]).
fn time_peer_halfout(
    lib: &libloading::Library,
    compile: Duration,
    out: &mut [u16],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureHalfOut> {
    // SAFETY: the generated `kbench` takes and returns `unsigned short*` (bf16 stored bits).
    let f: HalfOutKernelFn = unsafe { peer_kbench(lib)? };
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot: Vec<f32> = out.iter().map(|&b| widen_bf16(b)).collect();
    Some(MeasureHalfOut {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// Best-of-many-batches timing: warm up, grow the batch until ~50 ms, then take the fastest of many
/// batches. On a busy multicore box the *minimum* batch is the least-interfered estimate (the run
/// that suffered the least scheduler/thermal noise), so more samples tighten the result.
fn time_ns(mut run: impl FnMut()) -> f64 {
    // Warm up twice, then probe one call to size the sampling. A kernel whose *single* call already
    // exceeds the 50 ms target (the large GEMMs, 2048³/4096³) is sampled best-of-6 single calls
    // instead of best-of-14 rep-blocks, so a multi-second GEMM benches in seconds, not minutes —
    // identical "best observed" methodology, just fewer samples. Sub-target kernels are unchanged:
    // 5 warmups total, reps scaled until a block ≥ 50 ms, then best-of-14.
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

/// Print where Wukong's single-core GEMM stands: vs the tuned library (the SOTA-parity check) and
/// as a % of the measured AVX2-FMA roofline (clock-invariant — the honest figure on this box, whose
/// absolute GFLOP/s swings with the power/thermal state).
fn report_gemm_standing(wuk: &Option<Measure>, tuned: &Measure, roof: f64, flops: f64) {
    if let Some(m) = wuk {
        let wk_g = flops / m.ns_per_call;
        let tuned_g = flops / tuned.ns_per_call;
        let r = wk_g / tuned_g;
        println!(
            "  -> Wukong single-core is {:.2}x {} than tuned matrixmultiply ({wk_g:.0} vs {tuned_g:.0} GFLOP/s)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" },
        );
        if roof > 0.0 {
            println!(
                "  -> Wukong single-core = {:.0}% of measured roofline; tuned matrixmultiply = {:.0}%",
                wk_g / roof * 100.0,
                tuned_g / roof * 100.0
            );
        }
    }
}

// ----------------------------------------------------------------------------------------------
// oneMKL library peer — the gold-standard CPU GEMM (Intel MKL), resolved at runtime via dlopen.
// MKL ships in Anaconda/Miniconda as `mkl_rt.dll`; we load it with no build-time linkage so the
// bench runs with or without it (the column shows "n/a" when absent). On Meteor Lake MKL dispatches
// its AVX2 kernels (this part has no AVX-512), so MKL(1 thread) is an apples-to-apples 256-bit
// single-core peer, and MKL(all) uses MKL's own threading — the true Tier-B library bar, a tier
// above naive C/Rust and the pure-Rust `matrixmultiply` crate.
// ----------------------------------------------------------------------------------------------

// CBLAS enums (Intel MKL / Netlib CBLAS).
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

// Unconditional ILP64 GEMM `cblas_sgemm_64(layout, transa, transb, m, n, k, alpha, A, lda, B, ldb,
// beta, C, ldc)`: the CBLAS enums stay 32-bit `int`; the dimensions and leading dims are 64-bit
// `MKL_INT`. Using the `_64` symbol sidesteps the interface-layer default — Anaconda's `mkl_rt`
// defaults to ILP64 here, so the plain `cblas_sgemm` with 32-bit dims is misread and segfaults.
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
type MklSetNumThreadsFn = unsafe extern "C" fn(i32);
#[cfg(windows)] // resolved only inside the Windows-only `mkl()` loader
type MklGetMaxThreadsFn = unsafe extern "C" fn() -> i32;
// oneMKL VML (Vector Math Library) single-precision unary op: `vsExp(n, a, y)` ⇒ `y[i] = exp(a[i])`.
// The `n` width follows the same interface layer as CBLAS — ILP64 here (so `i64`, matching the
// `cblas_sgemm_64` we resolve). The default HA (high-accuracy, ~0.5 ULP) mode is left in force; the
// per-kernel cross-check vs Wukong's ~1-ULP poly tolerates the ≤2-ULP difference and would flag an
// ABI mismatch (garbage output) by reporting a large rel error, in which case the column is dropped.
type VmlUnaryFn = unsafe extern "C" fn(i64, *const f32, *mut f32);

// ILP64 GEMV `cblas_sgemv_64(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy)`. Same
// interface-layer reasoning as `cblas_sgemm_64` above: the enums stay 32-bit `int`, the dimensions
// and strides are 64-bit `MKL_INT`, and resolving the `_64` spelling sidesteps mkl_rt's default
// interface layer (which is ILP64 on this Anaconda build - the plain `cblas_sgemv` would be misread).
type CblasSgemvFn = unsafe extern "C" fn(
    i32,
    i32,
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

// Out-of-place matrix transposition `MKL_Somatcopy(ordering, trans, rows, cols, alpha, a, lda, b,
// ldb)`. Two spelling traps here, both measured against the installed mkl_rt.2.dll rather than
// guessed, and the first one is a hard crash:
//
//  * CAMELCASE, BY VALUE. The lowercase `mkl_somatcopy` is the Fortran BY-REFERENCE binding, so
//    calling it by value makes it dereference `alpha` as a pointer: it faults reading
//    0x000000003F800000, which is the bit pattern of 1.0f. That is exactly the trap already
//    recorded above for `mkl_set_num_threads` vs `MKL_Set_Num_Threads`. It cost one narrow test
//    (STATUS_ACCESS_VIOLATION 0xC0000005) instead of a measured round.
//  * NO `_64` SPELLING EXISTS, and none is needed. oneMKL declares the ?omatcopy family with
//    `size_t` extents, so the widths are 64-bit on x64 whichever interface layer mkl_rt selects -
//    unlike CBLAS, where the `_64` symbol is mandatory on this build.
//
// `ordering`/`trans` are C `char`s: 'R' row-major, 'T' transpose.
type MklSomatcopyFn =
    unsafe extern "C" fn(i8, i8, usize, usize, f32, *const f32, usize, *mut f32, usize);

/// Resolved oneMKL entry points. Function pointers are `Copy + Send + Sync`; the backing library is
/// leaked (`mem::forget`) so the pointers stay valid for the whole process. The VML ops are optional
/// (`None` if the symbol is absent) — they peer Wukong's vectorized transcendentals against Intel's
/// hand-tuned vector math, the elementwise analogue of the GEMM-vs-cblas comparison.
struct MklApi {
    sgemm: CblasSgemmFn,
    set_threads: MklSetNumThreadsFn,
    max_threads: i32,
    vs_exp: Option<VmlUnaryFn>,
    vs_ln: Option<VmlUnaryFn>,
    vs_tanh: Option<VmlUnaryFn>,
    /// `cblas_sgemv_64` - the library bar for the `gemv` family, which had none.
    sgemv: Option<CblasSgemvFn>,
    /// `mkl_somatcopy` - the library bar for the `transpose` family, which had none.
    somatcopy: Option<MklSomatcopyFn>,
    path: PathBuf,
}

/// Best-effort discovery of `mkl_rt.dll`: an explicit `WUKONG_MKL_DLL` override first, then the
/// standard conda layouts (`$CONDA_PREFIX`, `%USERPROFILE%\{Anaconda3,miniconda3,…}\Library\bin`,
/// and the system-wide ProgramData install).
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

/// The oneMKL peer is Windows-only in this harness (it resolves `mkl_rt.dll` through the Windows
/// loader); elsewhere the MKL columns are simply absent, same as any other missing toolchain.
/// This stub is what let the whole binary compile on Linux at all — the un-gated
/// `libloading::os::windows` import below broke `cargo check --all-targets` on ubuntu (E0432).
#[cfg(not(windows))]
fn mkl() -> Option<&'static MklApi> {
    None
}

/// Load oneMKL once (cached). `LOAD_WITH_ALTERED_SEARCH_PATH` makes the Windows loader resolve MKL's
/// own dependencies (`libiomp5md.dll`, `mkl_core`, `mkl_intel_thread`) from the DLL's directory.
#[cfg(windows)]
fn mkl() -> Option<&'static MklApi> {
    static API: OnceLock<Option<MklApi>> = OnceLock::new();
    API.get_or_init(|| {
        let path = mkl_dll_path()?;
        use libloading::os::windows::{Library as WinLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};
        let lib =
            unsafe { WinLibrary::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH) }.ok()?;
        let lib: libloading::Library = lib.into();
        let api = unsafe {
            // CamelCase = MKL's C by-value interface. The lowercase `mkl_set_num_threads` is the
            // Fortran *by-reference* binding (`const int*`); calling it by value dereferences the
            // thread count as a pointer and segfaults — use `MKL_Set_Num_Threads` (by value).
            let sgemm = *lib.get::<CblasSgemmFn>(b"cblas_sgemm_64\0").ok()?;
            let set_threads = *lib
                .get::<MklSetNumThreadsFn>(b"MKL_Set_Num_Threads\0")
                .ok()?;
            let get_max = *lib
                .get::<MklGetMaxThreadsFn>(b"MKL_Get_Max_Threads\0")
                .ok()?;
            // VML transcendentals are optional (older MKL builds, or a stripped redist, may omit them).
            let vml = |sym: &[u8]| lib.get::<VmlUnaryFn>(sym).ok().map(|s| *s);
            let vs_exp = vml(b"vsExp\0");
            let vs_ln = vml(b"vsLn\0");
            let vs_tanh = vml(b"vsTanh\0");
            // Optional, exactly like the VML ops: an older MKL or a stripped redist may omit them,
            // and the column then reads `n/a` the same way a missing toolchain does. Added
            // 2026-09-13 because 143 of the 155 measured sections had NO library-grade peer at all
            // while the DLL that ships the exact primitive was already resident in this process -
            // a win over a hand-written gcc loop is not a win over the state of the art.
            let sgemv = lib
                .get::<CblasSgemvFn>(b"cblas_sgemv_64\0")
                .ok()
                .map(|s| *s);
            let somatcopy = lib
                .get::<MklSomatcopyFn>(b"MKL_Somatcopy\0")
                .ok()
                .map(|s| *s);
            MklApi {
                sgemm,
                set_threads,
                max_threads: get_max(),
                vs_exp,
                vs_ln,
                vs_tanh,
                sgemv,
                somatcopy,
                path,
            }
        };
        std::mem::forget(lib);
        Some(api)
    })
    .as_ref()
}

/// Time oneMKL's `cblas_sgemm` for `C = A·B` (or `A·Bᵀ` when `transpose_b`), row-major square `ns`,
/// pinned to `threads` MKL threads. `None` if MKL is unavailable.
fn bench_mm_mkl(
    ns: usize,
    transpose_b: bool,
    threads: i32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Option<Measure> {
    bench_mm_mkl_rect(ns, ns, ns, transpose_b, threads, a, b, c)
}

/// Rectangular twin of [`bench_mm_mkl`]: `C[M,N] = A[M,K] · B` with `B` stored `[N,K]` when
/// `transpose_b` (the `nn.Linear` layout) or `[K,N]` otherwise. Row-major leading dimensions
/// follow the storage: `lda = K`, `ldb = K` (transposed) / `N` (not), `ldc = N`.
#[allow(clippy::too_many_arguments)]
fn bench_mm_mkl_rect(
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
    threads: i32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Option<Measure> {
    let api = mkl()?;
    let (m, n, k) = (m as i64, n as i64, k as i64);
    let lda = k;
    let (transb, ldb) = if transpose_b {
        (CBLAS_TRANS, k)
    } else {
        (CBLAS_NO_TRANS, n)
    };
    let ldc = n;
    unsafe {
        (api.set_threads)(threads.max(1));
    }
    c.iter_mut().for_each(|v| *v = 0.0);
    let ns_per_call = time_ns(|| unsafe {
        (api.sgemm)(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            transb,
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,
            c.as_mut_ptr(),
            ldc,
        );
    });
    Some(Measure {
        compile: Duration::ZERO,
        ns_per_call,
        out: c.to_vec(),
    })
}

/// Time oneMKL's `cblas_sgemv` for `y[M] = A[M,N].x[N]`, row-major, pinned to `threads` MKL threads.
/// `None` when MKL is absent or the symbol did not resolve - the column then prints `n/a`.
fn bench_gemv_mkl(
    m: usize,
    n: usize,
    threads: i32,
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
) -> Option<Measure> {
    let api = mkl()?;
    let sgemv = api.sgemv?;
    let (mi, ni) = (m as i64, n as i64);
    unsafe {
        (api.set_threads)(threads.max(1));
    }
    y.iter_mut().for_each(|v| *v = 0.0);
    let yp = y.as_mut_ptr();
    let ns_per_call = time_ns(|| unsafe {
        sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            mi,
            ni,
            1.0,
            a.as_ptr(),
            ni,
            x.as_ptr(),
            1,
            0.0,
            yp,
            1,
        );
    });
    Some(Measure {
        compile: Duration::ZERO,
        ns_per_call,
        out: y.to_vec(),
    })
}

/// Time oneMKL's `MKL_Somatcopy` for `dst = src^T`, row-major, pinned to `threads` MKL threads.
///
/// `src` is `[rows, cols]` with `lda = cols`; `dst` is `[cols, rows]` with `ldb = rows`, which is
/// exactly the layout every peer in `bench_transpose` writes (`d[j*rows + i] = s[i*cols + j]`).
/// A transposition is a permutation, so the caller checks this column for BIT equality like the
/// others and drops it on any disagreement.
fn bench_transpose_mkl(
    rows: usize,
    cols: usize,
    threads: i32,
    src: &[f32],
    dst: &mut [f32],
) -> Option<Measure> {
    let api = mkl()?;
    let somatcopy = api.somatcopy?;
    unsafe {
        (api.set_threads)(threads.max(1));
    }
    dst.iter_mut().for_each(|v| *v = 0.0);
    let dp = dst.as_mut_ptr();
    let ns_per_call = time_ns(|| unsafe {
        somatcopy(
            b'R' as i8,
            b'T' as i8,
            rows,
            cols,
            1.0,
            src.as_ptr(),
            cols,
            dp,
            rows,
        );
    });
    Some(Measure {
        compile: Duration::ZERO,
        ns_per_call,
        out: dst.to_vec(),
    })
}

/// Time oneMKL's `cblas_sgemm` for the weight-gradient shape `C[M,N] = A^T . B`, with A stored
/// `[K, M]` and B stored `[K, N]` row-major - the layout `bench_matmul_tn` uses.
///
/// [`bench_mm_mkl_rect`] hardcodes `CBLAS_NO_TRANS` for A and only parameterises `transb`, so the
/// TN shape needs its own wrapper rather than an enum swap. Same resolved `cblas_sgemm_64`, so
/// there is no new symbol and no new ABI risk.
fn bench_mm_mkl_tn(
    m: usize,
    k: usize,
    n: usize,
    threads: i32,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Option<Measure> {
    let api = mkl()?;
    let (mi, ni, ki) = (m as i64, n as i64, k as i64);
    unsafe {
        (api.set_threads)(threads.max(1));
    }
    c.iter_mut().for_each(|v| *v = 0.0);
    let cp = c.as_mut_ptr();
    let ns_per_call = time_ns(|| unsafe {
        (api.sgemm)(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            mi,
            ni,
            ki,
            1.0,
            a.as_ptr(),
            mi, // A is [K, M] row-major: its leading dimension is M
            b.as_ptr(),
            ni, // B is [K, N] row-major
            0.0,
            cp,
            ni,
        );
    });
    Some(Measure {
        compile: Duration::ZERO,
        ns_per_call,
        out: c.to_vec(),
    })
}

/// Time a oneMKL VML unary op (`vsExp`/`vsLn`/`vsTanh`) over the buffer: `out = f(x)`, single-threaded
/// (VML respects `MKL_Set_Num_Threads`; we leave it at 1 to peer Wukong's single-core vmath kernel).
/// Snapshots the output for the correctness cross-check against Wukong's vectorized transcendental.
fn bench_vml(f: VmlUnaryFn, x: *const f32, out: &mut [f32]) -> Measure {
    if let Some(api) = mkl() {
        unsafe { (api.set_threads)(1) };
    }
    let nn = out.len() as i64;
    let op = out.as_mut_ptr();
    let ns_per_call = time_ns(|| unsafe { f(nn, x, op) });
    Measure {
        compile: Duration::ZERO,
        ns_per_call,
        out: out.to_vec(),
    }
}

/// Print Wukong's GEMM standing against the oneMKL peer: single-core vs MKL(1 thread) and
/// @parallel vs MKL(all threads), as a clock-invariant ratio + Wukong's %-of-MKL. MKL is Intel's
/// hand-tuned, JIT'd GEMM (AVX2 dispatch on this part) — the honest library bar. Also cross-checks
/// that Wukong's blocked kernel agrees with MKL element-by-element (independent correctness).
fn report_gemm_vs_mkl(
    wuk: &Option<Measure>,
    wk_par: &Option<Measure>,
    mkl_1c: &Option<Measure>,
    mkl_all: &Option<Measure>,
    flops: f64,
) {
    if let (Some(m), Some(k)) = (wuk, mkl_1c) {
        let (mg, kg) = (flops / m.ns_per_call, flops / k.ns_per_call);
        let r = mg / kg;
        println!(
            "  -> Wukong 1-core is {:.2}x {} than oneMKL(1 thread) — {:.0}% of MKL ({mg:.0} vs {kg:.0} GFLOP/s)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" },
            mg / kg * 100.0,
        );
        let (rel, at) = max_rel_err(&m.out, &k.out);
        if rel > 1e-3 {
            println!("  ! Wukong vs MKL full-buffer mismatch at [{at}] (rel {rel:.2e})");
        }
    }
    // Wukong's own parallel scaling: @parallel ÷ its own single core. Both are measured in the same
    // run with the same (rayon) thread pool, so this is robust to the laptop's power state and free of
    // the cross-pool noise that wrecks the MKL all-core comparison — the honest "how well does Wukong
    // scale" figure (16 physical cores is the ceiling; HT siblings add no FMA throughput).
    if let (Some(m1), Some(mp)) = (wuk, wk_par) {
        let eff = m1.ns_per_call / mp.ns_per_call;
        println!("  -> Wukong @parallel scales {eff:.1}× over its own single core");
    }
    if let (Some(m), Some(k1), Some(ka)) = (wk_par, mkl_1c, mkl_all) {
        let (mg, kag, k1g) = (
            flops / m.ns_per_call,
            flops / ka.ns_per_call,
            flops / k1.ns_per_call,
        );
        // Degeneracy guard. A real all-core GEMM is ≥4× its single-core self; if MKL(all) failed to
        // clear even 1.2× MKL(1c), its 16 OpenMP workers did not actually scale this call — a known
        // pathology on this thermally-constrained hybrid when the box is loaded (the pool stalls, and
        // we have measured MKL(all) read *below* MKL(1c)). Printing "1200% of MKL" off such a run would
        // be a measurement artifact dressed as a win — omit the ratio and say why (honesty law).
        if kag <= k1g * 1.2 {
            println!(
                "  -> oneMKL(all) {kag:.0} ≤ MKL(1c) {k1g:.0} GFLOP/s: degenerate all-core run (OMP did \
                 not scale on this loaded hybrid) — ratio omitted; see the @parallel scaling above"
            );
        } else {
            let r = mg / kag;
            println!(
                "  -> Wukong @parallel is {:.2}x {} than oneMKL(all threads) — {:.0}% of MKL ({mg:.0} vs {kag:.0} GFLOP/s)",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" },
                mg / kag * 100.0,
            );
        }
    }
}

/// Benchmark a single-threaded tuned-library GEMM (the pure-Rust `matrixmultiply` crate, the engine
/// behind `ndarray`) at the same size and layout, so Wukong's GEMM is measured against a genuine
/// optimized peer — not only the naive C/Rust nests. `transpose_b` selects `C = A·Bᵀ` (nn.Linear).
fn bench_mm_tuned(ns: usize, transpose_b: bool, a: &[f32], b: &[f32], c: &mut [f32]) -> Measure {
    let (m, k, n) = (ns, ns, ns);
    // matrixmultiply takes explicit row/col strides; B stored [n,k] row-major is read as Bᵀ (k×n)
    // by swapping its strides (rsb=1 walks the contraction, csb=k walks the output column).
    let (rsb, csb) = if transpose_b {
        (1isize, ns as isize)
    } else {
        (ns as isize, 1isize)
    };
    c.iter_mut().for_each(|v| *v = 0.0);
    let ns_per_call = time_ns(|| unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            ns as isize,
            1,
            b.as_ptr(),
            rsb,
            csb,
            0.0,
            c.as_mut_ptr(),
            ns as isize,
            1,
        );
    });
    Measure {
        compile: Duration::ZERO,
        ns_per_call,
        out: c.to_vec(),
    }
}

/// This machine's *currently achievable* single-core AVX2-FMA peak (GFLOP/s), via a tight,
/// memory-free FMA loop carrying 12 independent accumulators (matching the GEMM microkernel's 12
/// live `ymm`) so the two FMA units stay saturated past the ~4-cycle FMA latency. Absolute
/// throughput on this laptop swings with the power/thermal clock state, so the honest figure for the
/// GEMM is "% of THIS roofline" (clock-invariant), not a fixed GFLOP/s. Returns 0.0 without AVX2/FMA.
fn measure_fma_roofline() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { fma_roofline_avx2() };
        }
    }
    0.0
}

/// The sustained roofline: warm ≥400 ms, then best of 8 × 40M-iteration blocks. The arguments are
/// spelled out here rather than baked into [`fma_probe_avx2`] because [`clock_probe_gflops`] runs the
/// IDENTICAL loop at a much shorter setting, and the two must not drift apart.
///
/// # Safety
/// Requires AVX2 + FMA; [`measure_fma_roofline`] is the checked entry point.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_roofline_avx2() -> f64 {
    fma_probe_avx2(400, 40_000_000, 8)
}

/// The FMA probe loop, parameterized by warm-up wall time, iterations per timed block and block
/// count. GFLOP/s = best block.
///
/// # Safety
/// Requires AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_probe_avx2(warm_ms: u64, iters: u64, samples: usize) -> f64 {
    use std::arch::x86_64::*;
    let a = _mm256_set1_ps(1.000_000_1);
    let b = _mm256_set1_ps(0.999_999_9);
    let run = |iters: u64| -> f32 {
        let mut acc = [_mm256_set1_ps(1e-6f32); 12];
        for _ in 0..iters {
            // 12 independent FMA chains: each `acc[i]` depends only on its own prior value, so the
            // scheduler keeps both FMA units busy rather than stalling on a single 4-cycle chain.
            for a_i in acc.iter_mut() {
                *a_i = _mm256_fmadd_ps(a, b, *a_i);
            }
        }
        let mut s = _mm256_setzero_ps();
        for x in acc {
            s = _mm256_add_ps(s, x);
        }
        let mut tmp = [0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), s);
        tmp.iter().sum()
    };
    // Ramp the clock to its sustained turbo state before timing. A fixed-iteration warmup (~5 ms) is
    // far too short: the chip boosts over ~100-400 ms from a cold idle start, so the roofline would be
    // measured throttled and then read *below* the warm GEMM that runs minutes later — making
    // "% of roofline" exceed 100% (an obvious honesty bug). Warm for ≥400 ms of wall time instead.
    // (The per-round clock trace passes a much shorter `warm_ms` on purpose — it is sampled between
    // rounds of an already-hot section and only has to read the clock, not ramp it.)
    {
        let t = Instant::now();
        let mut warm = 0f32;
        while t.elapsed() < Duration::from_millis(warm_ms) {
            warm += run(2_000_000);
        }
        std::hint::black_box(warm);
    }
    let mut best = f64::INFINITY;
    let mut sink = 0f32;
    for _ in 0..samples {
        let t = Instant::now();
        sink += run(iters);
        let e = t.elapsed().as_secs_f64();
        if e < best {
            best = e;
        }
    }
    std::hint::black_box(sink);
    // 12 FMAs/iter × 8 lanes × 2 flops/FMA.
    (iters * 12 * 8 * 2) as f64 / best / 1e9
}

// ---------------------------------------------------------------------------------------------
// CPU PINNING — the largest single confound this harness has, measured (on the general suite,
// which is where this machinery lived until 2026-08-22; the whole binary now pins, because the
// ~45-family battery was publishing cross-language ratios with the confound below still in them)
// ---------------------------------------------------------------------------------------------
//
// This laptop is an Intel Core Ultra 7 155H: 6 P-cores (SMT, logical 0-11), 8 E-cores and 2 LP
// E-cores, 22 logical processors in three performance classes. Every program in this section is
// single-threaded and each peer call is 50-200 ms, so a visit is ~1 s of uninterrupted scalar/AVX2
// work — exactly the profile Windows' Thread Director demotes off a P-core.
//
// MEASURED, 2026-08-06, AC+CHARGING, same binary, same 4-round protocol, back to back, kill switch
// only. The block section's raw per-round ms for the peer columns:
//
//   UNPINNED   C     81.16  105.32  127.66  174.30      (2.15x within ONE run, monotone)
//              C++  108.88  109.65  112.24  205.66
//              Rust 135.97  136.06  120.28  123.91
//              -> C(twin) control 1.057x [0.695-1.472], spread 2.118x: THE CONTROL FAILED
//
//   PINNED     C     75.21   74.77   74.09   70.55      (1.07x)
//   (CPU 0-1)  C++   73.46   74.15   73.21   73.75      (1.01x)
//              Rust  88.70   83.70   85.12   87.13      (1.06x)
//              -> C(twin) control spread ~1.05x: THE CONTROL PASSES
//
// The C column does not merely jitter unpinned — it DEGRADES MONOTONICALLY, 81 -> 174 ms, inside one
// run, while the 4-5 ms Wukong columns in the same rounds do not move at all. That is the signature
// of a long-running thread being walked down the performance classes, not of noise.
//
// Replicated on BATTERY the same day, three interleaved runs per arm (2026-08-06, discharging
// 83%->76%, so the LEVELS are battery levels and not comparable to the AC pair above; the RATIO
// between the arms is what is being read):
//
//   block C best-of-rounds, unpinned  276.6 / 280.9 / 289.2 ms
//   block C best-of-rounds, pinned    154.5 / 130.0 / 154.8 ms      -> 1.79x / 2.16x / 1.87x
//   run-to-run spread of the published `vs C` ratio, 19 rows:
//       unpinned  median 1.085x  worst 1.823x
//       pinned    median 1.035x  worst 1.076x
//   and the same lever helps the OLD single-visit protocol just as much (main's binary, same runs):
//       unpinned  median 1.170x  worst 1.271x
//       pinned    median 1.055x  worst 1.103x
//
// So the "protocol changed the measured LEVEL of an unchanged peer by ~1.78x" finding is neither
// thermal accumulation nor the round protocol's arithmetic. It is CORE PLACEMENT, it reproduces to
// the decimal as a same-binary kill-switch A/B, and it is the single largest confound this section
// has ever had — larger than the protocol change that exposed it.
//
// AND THE HARNESS HAD ALREADY PRINTED THE SMOKING GUN. The per-round clock probe of an unpinned run
// (ec08db2's binary, AC+CHARGING, block section) traced
//
//     145 -> 85 -> 71 -> 51 -> 106 -> 101 GF/s   spread 2.85x
//
// across six rounds of ONE section, on a machine whose power state never changed. A 2.85x collapse
// and partial recovery in a fixed AVX2-FMA loop is a thread moving between core classes, and it sat
// in the output for anyone to read. The lesson is in `clock_probe_gflops`: a trace is only useful if
// someone acts on it, and this one was reported as "the drift the ratios had to survive" rather than
// as a defect to remove.
//
// Pinning is uniform across every column and carries no language content whatever — and it does not
// have to be taken on trust, because `C(twin)` is exactly the instrument that checks it: a control
// whose true ratio is 1.00 went from 2.12x spread to ~1.05x. That is the justification for turning it
// on by default, and it is the only kind of justification worth having here.
//
// LIMITS, stated rather than glossed. One core is not the machine: an L3 shared with 15 idle
// siblings behaves differently from a loaded one, and a single-core measurement says nothing about
// throughput under load. The single-threaded columns — every general-suite column and every plain
// battery row — compare pinned against pinned, which is the right trade. The battery's `@parallel`
// and C(omp) rows are different: only the CALLING thread is pinned (Windows thread affinity is not
// inherited, so pool/team workers spawned by it still get the whole machine), identically for every
// language, so those rows stay same-machine comparisons — but a pinned figure is still not a
// multicore claim. And the pin does NOT make the `ms` column comparable across protocols: see
// [`general`]'s `time_round`.

/// Which logical CPUs this harness pins its timing thread to, from `XBENCH_PIN`.
///
/// * unset — CPUs 0 and 1, the two SMT threads of the first P-core. Logical processors are
///   enumerated performance-class-first on every Intel hybrid client part, so CPU 0 is a P-core;
///   allowing both siblings of one physical core costs nothing (only one of our threads runs) and
///   leaves the OS somewhere to put an interrupt without evicting us.
/// * `off` / `none` / `0` — no pinning. This is the kill switch the A/B above was measured with.
/// * a comma-separated list of logical CPU indices — pin to exactly those.
///
/// An unparseable value falls back to the default rather than to "off": a typo must not silently
/// remove the control that makes this section measurable.
fn pin_mask() -> Option<usize> {
    parse_pin(std::env::var("XBENCH_PIN").ok().as_deref())
}

/// The pure half of [`pin_mask`], so the contract is testable without mutating a process-global env
/// var underneath every other test in the binary.
fn parse_pin(v: Option<&str>) -> Option<usize> {
    let Some(v) = v.map(str::trim) else {
        return Some(0b11);
    };
    if v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("none") || v == "0" {
        return None;
    }
    let mut mask = 0usize;
    for tok in v.split(',') {
        match tok.trim().parse::<u32>() {
            Ok(n) if (n as usize) < usize::BITS as usize => mask |= 1usize << n,
            _ => return Some(0b11),
        }
    }
    if mask == 0 {
        Some(0b11)
    } else {
        Some(mask)
    }
}

/// Pin the CALLING THREAD (not the process) to `mask`, returning the previous mask so it can be put
/// back. Thread affinity is not inherited by child processes, so the `gcc`/`g++`/`rustc` peer
/// compiles this section spawns still get the whole machine.
#[cfg(windows)]
fn set_thread_affinity(mask: usize) -> Option<usize> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
    }
    // SAFETY: `GetCurrentThread` returns a pseudo-handle that needs no closing, and
    // `SetThreadAffinityMask` only reads it. A zero return is the documented failure signal (an
    // empty mask, or one naming a processor this process is not allowed to run on).
    let prev = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
    (prev != 0).then_some(prev)
}

#[cfg(not(windows))]
fn set_thread_affinity(_mask: usize) -> Option<usize> {
    None
}

/// Restores the thread's original affinity when the section ends, however it ends. A section that
/// left the benchmark process pinned to one core would silently cap every LATER section's multicore
/// rows — a benchmark bug that would look like a performance finding.
struct PinGuard(Option<usize>);

impl Drop for PinGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0 {
            let _ = set_thread_affinity(prev);
        }
    }
}

/// Apply [`pin_mask`], print what happened, and hand back the guard that undoes it.
fn pin_for_section() -> PinGuard {
    let Some(mask) = pin_mask() else {
        println!(
            "  CPU PINNING: OFF (XBENCH_PIN). On this hybrid P/E laptop an unpinned single-threaded\n\
             \x20   section migrates across performance classes mid-run — measured, the C(twin) control\n\
             \x20   went 1.05x spread pinned to 2.12x unpinned, and the block C column drifted 81 ->\n\
             \x20   174 ms inside ONE run. Expect this section to resolve nothing."
        );
        return PinGuard(None);
    };
    match set_thread_affinity(mask) {
        Some(prev) => {
            println!(
                "  CPU PINNING: the timing thread is pinned to logical CPU mask {mask:#x}\n\
                 \x20   (XBENCH_PIN=off disables; XBENCH_PIN=4,5 chooses). Every column is pinned the\n\
                 \x20   same way, so this carries no language content — and the C(twin) control is what\n\
                 \x20   checks it rather than asserting it. Peer compiles are separate processes and\n\
                 \x20   still get the whole machine. A pinned figure is single-core only."
            );
            PinGuard(Some(prev))
        }
        None => {
            println!(
                "  CPU PINNING: requested mask {mask:#x} was REFUSED by the OS — running unpinned."
            );
            PinGuard(None)
        }
    }
}

fn geomean(xs: &[f64]) -> f64 {
    let s: f64 = xs.iter().map(|x| x.ln()).sum();
    (s / xs.len() as f64).exp()
}

// ----------------------------------------------------------------------------------------------
// ROUND-INTERLEAVED MEASUREMENT
//
// WHAT THIS IS AND IS NOT. Two things here are established, one thing that was claimed for it is
// NOT, and the difference matters more than either.
//
// ESTABLISHED 1 — A BIAS FIX. `bench_structure_tax` used to time all five Wukong spellings to
// completion and only then build and time the peers, one language after another, each exactly once,
// in a FIXED order. C therefore always held the first and coolest peer slot and Rust always held the
// last, so any drift across the section was charged to Rust and the Wukong columns always got the
// earliest slots — in Wukong's own benchmark. That is a systematic, reproducible error in the
// method, not noise: the harness published "C++ 1.89x slower than C" on byte-identical numeric
// bodies from position alone. (The loss and scan sections had already been given a forward/reverse
// two-pass at 17ab8d0; the structure tax had not.) Timing every column once per round in a rotating,
// direction-alternating order fixes it by construction, and forming each ratio INSIDE a round means
// no ratio is ever a quotient of two different instants of the machine's clock.
//
// ESTABLISHED 2 — A MEASURED NOISE FLOOR. `C(twin)` is the identical C source through the identical
// compiler at the identical flags in a second DLL. Its ratio against `C` has expected value exactly
// 1.00 and contains no language content whatsoever, so what it reads away from 1.00 is the
// instrument. It is the most valuable thing in this module. In one validation run it read C 75.20 ms
// against C(twin) 57.60 ms — the pre-2026-08-06 estimator would have published A BYTE-IDENTICAL
// BINARY as "1.31x faster than C". Every cross-language cell is now classified against that floor:
// a range that overlaps what the control did reads BELOW FLOOR and carries no number.
//
// NOT ESTABLISHED — THE VARIANCE HEADLINE, WHICH IS RETRACTED. The commit that introduced this
// protocol (ec08db2, branch bench/noise-resistant-ratios) claimed the run-to-run spread of the
// published `vs C` ratio fell from median 1.451x / worst 3.764x to median 1.142x / worst 1.404x. IT
// DOES NOT REPLICATE. An adversarial re-run of the general suite EIGHT times with both binaries
// interleaved, on AC+CHARGING, measured OLD median 1.195x / worst 2.876x against NEW median 1.270x /
// worst 1.869x — the MEDIAN GOT WORSE. Do not restate the retracted claim.
//
// ESTABLISHED 3 — AND THE REASON THE MEDIAN MOVED AT ALL WAS NEITHER PROTOCOL. It was the Windows
// scheduler walking a long single-threaded section down a hybrid P/E core topology, and once the
// timing thread is pinned it goes away — see `general`'s CPU PINNING section for the A/B. That is
// the largest confound this suite has ever had and it was invisible to both protocols. Measured,
// same binary, kill switch only, three interleaved runs per arm (battery):
//
//     ROUND PROTOCOL, unpinned    median 1.085x   worst 1.823x
//     ROUND PROTOCOL, pinned      median 1.035x   worst 1.076x
//     OLD PROTOCOL,   unpinned    median 1.170x   worst 1.271x
//     OLD PROTOCOL,   pinned      median 1.055x   worst 1.103x
//
// Read that honestly: PINNING is the big lever and it helps BOTH protocols. What the round protocol
// adds on top of a pinned machine is real but small (1.055 -> 1.035 median, 1.103 -> 1.076 worst),
// and a separate interleaved pair measured 1.068x/1.176x for ec08db2's exact protocol against
// 1.028x/1.127x for this one. The round protocol's case rests on the bias fix and the control, not on
// a variance headline.
//
// The per-visit budget ec08db2 cut (2 warm-ups + best-of-7 -> 1 + best-of-3) is restored, but for a
// measured reason that is NOT the one it was blamed for: on a pinned machine it does not move the
// published ratio's spread at all (1.049x/1.094x restored against 1.041x/1.092x cut). What it moves
// is the run's own NOISE FLOOR — the control's per-round spread, 7 of 9 sections inside the 1.10
// limit against 2 of 9 — and the floor is the gate that decides what may be printed. See
// `general::time_round`. The default round count is cut 6 -> 4 to pay for it.
//
// AND THE PROTOCOL STILL SHIFTS THE LEVEL, NOT ONLY THE ORDER — the residual, after pinning, is in
// the block section only, and it is the difference between timing a column right after a gcc pause
// and timing continuously. See `general::time_round`. Read `ms` columns only against other `ms`
// columns from the same run.
//
// WHAT THE WHOLE THING BUYS, END TO END, MEASURED IN THE STATE THE VERIFIER USED. Six general runs,
// 2026-08-06, AC+CHARGING 68%->72% (charging caps ALL-CORE only; every column in this suite is
// single-threaded, and it is pinned, so the cap does not bind — but the LEVELS are not comparable to
// an AC+full run and no level here is reportable). ec08db2's binary and this one INTERLEAVED, three
// runs each, order ship / pred / pred / ship / ship / pred:
//
//                                     ec08db2      SHIPPED
//     cells resolved to a SIZE          0 of 57     19 of 57
//     sections whose CONTROL failed     9 of 9       3 of 9
//     control per-round spread       1.702x med   1.049x med   (worst 2.169x -> 1.503x)
//     run-to-run spread of `vs C`    1.276x med   1.056x med   (worst 1.633x -> 1.285x)
//
// The 1.276x replicates the verifier's 1.270x for this protocol almost exactly, which is the
// strongest evidence available that the comparison itself is sound. It is a THREE-change A/B (pin,
// per-visit budget, round count), so read it with the decomposition above rather than instead of it:
// the battery A/Bs attribute the published-ratio spread almost entirely to the PIN and the control's
// spread — the floor — largely to the restored BUDGET. Nothing here says anything about any language.
// ----------------------------------------------------------------------------------------------

/// The order to visit `n` columns in on round `r`.
///
/// Rounds come in PAIRS: round `2k` visits the columns rotated left by `k`, round `2k+1` visits that
/// exact list reversed. Every column therefore sits at position `p` in the forward round and at
/// `n-1-p` in the reverse one, so after each complete pair every column's positions sum to exactly
/// `n-1` — the position balance is exact, not statistical. At `n = 2` this reproduces the A B / B A
/// of [`abba_min_columns`] literally; this is its N-column form. The rotation across pairs additionally moves
/// which column *leads* a round, since the leader is the one that pays for whatever ran between
/// rounds (the clock probe, the round's power sample).
///
/// Use an EVEN number of rounds — see [`gen_rounds`], which enforces it — or the last unpaired round
/// reintroduces exactly the positional bias this removes.
fn round_order(n: usize, round: usize) -> Vec<usize> {
    let shift = round / 2;
    let mut ord: Vec<usize> = (0..n).map(|p| (shift + p) % n.max(1)).collect();
    if round % 2 == 1 {
        ord.reverse();
    }
    ord
}

/// Number of rounds for the round-interleaved sections, from `XBENCH_ROUNDS` (default 4), rounded UP
/// to an even number so [`round_order`]'s position balance is exact, and floored at 2 (a single
/// round has no dispersion to report and is exactly the protocol this replaces).
///
/// The default was 6 when the protocol landed and is 4 now. Rounds and per-visit samples are two
/// different axes and the budget has to be split between them: rounds buy protection from drift
/// BETWEEN visits, per-visit samples buy a good estimate of the speed AT a visit. ec08db2 spent
/// almost everything on rounds (6 rounds x [1 warm + best-of-3]) and the resulting per-visit estimate
/// was measurably worse than the protocol it replaced on the block section's 4-5 ms columns. Total
/// calls per column is `rounds * (warmups + samples)`; at 4 x (2 + 7) that is 36, against 6 x (1 + 3)
/// = 24 and the old single-visit protocol's 9. Reduce the ROUND COUNT, never the per-visit budget, if
/// a section becomes too slow.
fn gen_rounds() -> usize {
    parse_rounds(std::env::var("XBENCH_ROUNDS").ok().as_deref())
}

/// The pure half of [`gen_rounds`], so the even/floored contract is testable without a test having
/// to mutate a process-global env var underneath every other test in the binary.
fn parse_rounds(v: Option<&str>) -> usize {
    let r = v.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(4);
    let r = r.max(2);
    r + (r % 2)
}

/// Untimed warm-up calls at the start of every visit to a column (`XBENCH_WARMUPS`, default 2).
///
/// Two, not one, and this is the number ec08db2 cut. A column last ran a whole round ago and every
/// other column has walked the same buffers since, so the first call after the gap pays a cache
/// refill and the second pays branch-predictor and (for the JIT columns) page-fault warm-up. Charging
/// either to the timed minimum is charging the schedule to the column.
fn gen_warmups() -> usize {
    parse_budget(std::env::var("XBENCH_WARMUPS").ok().as_deref(), 2, 0)
}

/// Timed calls per visit; the visit's sample is their MINIMUM (`XBENCH_SAMPLES`, default 7).
///
/// Seven is not arbitrary — it is what the pre-round protocol used, and the measured reason to
/// restore it is in the module header: at best-of-3 the block section's Wukong columns roughly
/// doubled their run-to-run spread. The minimum of a sample is a biased-low estimator whose bias
/// shrinks with the sample size, so cutting 7 to 3 does not merely widen the estimate, it moves it,
/// and it moves it by different amounts for a 4 ms column and a 70 ms one.
fn gen_samples() -> usize {
    parse_budget(std::env::var("XBENCH_SAMPLES").ok().as_deref(), 7, 1)
}

/// The pure half of [`gen_warmups`] / [`gen_samples`]. A garbage or out-of-range override falls back
/// to the default rather than to `lo`: `XBENCH_SAMPLES=0` would leave a visit with no timed call and
/// a `NaN` in the round table, and a `NaN` propagates into a dropped `RatioStat`, which reads exactly
/// like "this row was not measurable" rather than like "you typed a zero".
fn parse_budget(v: Option<&str>, dflt: usize, lo: usize) -> usize {
    v.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= lo && n <= 1000)
        .unwrap_or(dflt)
}

/// Median of a sample. Even-length samples take the mean of the two middle order statistics.
/// The median, not the mean, is the point estimate for a per-round ratio: one round that collided
/// with a background task is a large outlier in one direction only, and a mean would carry it.
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The one-screen summary of a sweep's [`SotaRow`]s: every field a floor-file row records, in
/// floor-file order, so a passing round can be transcribed into `bench/sota/FLOOR.md` by reading
/// this block and nothing else.
fn print_sota_scorecard(rows: &[SotaRow]) {
    println!("SOTA scorecard (ratio = peer_time / wukong_time; > 1.00 means Wukong faster):");
    for r in rows {
        let ratio = match &r.ratio {
            Some(s) => format!("{:.2}x [{:.2}-{:.2}]", s.med, s.lo, s.hi),
            None => "n/a".into(),
        };
        let twin = match &r.twin {
            Some(t) => format!("{:.2}x", floor_factor(*t)),
            None => "none".into(),
        };
        let mut flags = String::new();
        if !r.reportable {
            flags.push_str("  POWER-CHANGED");
        }
        if r.peer_missing {
            flags.push_str("  PEER-MISSING");
        }
        if !r.oracle_ok {
            flags.push_str("  ORACLE-FAILED");
        }
        println!(
            "  {:<22} vs {:<3} ratio {:<22} twin-floor {:<6} power {}{}",
            r.program, r.peer, ratio, twin, r.power, flags
        );
        println!("      census: {}", r.census);
    }
}

/// One in-scope program's outcome from a sweep — what `sota-check` (part 11 of the language
/// close-out campaign's verification gate) evaluates against `bench/sota/FLOOR.md`, and what the
/// producing section prints its scorecard from. Produced by `compare::bench_compare` (and the
/// general suite once its programs join the floor file).
pub(crate) struct SotaRow {
    /// Floor-file row name, e.g. `compare/saxpy`.
    pub(crate) program: String,
    /// Which peer the ratio is against (the strongest honest one this row carries).
    pub(crate) peer: String,
    /// `peer_time / wukong_time`, per round — `> 1` means Wukong faster. `None` when the row could
    /// not honestly produce one (a column missing, the oracle failed, the census inadmissible).
    pub(crate) ratio: Option<RatioStat>,
    /// The run's own `C(twin)` control for this section.
    pub(crate) twin: Option<RatioStat>,
    /// Did the power state hold across every round?
    pub(crate) reportable: bool,
    /// The power-state key the rounds ran under (`AC+full` / `AC+CHARGING` / `BATTERY` / …).
    pub(crate) power: String,
    /// The dispatch-census line for the Wukong column — the T2 recognition-off witness.
    pub(crate) census: String,
    /// The bar peer failed to compile or load — `sota-check` reads this as a FAILURE, never a skip
    /// (a cross-language sweep drops a column silently otherwise).
    pub(crate) peer_missing: bool,
    /// Every column agreed with the f64 scalar reference within the section's bar.
    pub(crate) oracle_ok: bool,
}

/// A ratio between two columns, measured ROUND BY ROUND: the median across rounds and the range the
/// rounds spanned.
///
/// There is deliberately no constructor that yields a point estimate without its dispersion, and no
/// accessor that hands out `med` without `spread`. On this machine a lone ratio is not a measurement,
/// and the type is shaped so a caller cannot accidentally print one as if it were.
#[derive(Clone, Copy, Debug)]
struct RatioStat {
    /// Median of the per-round ratios — the point estimate.
    med: f64,
    /// Smallest per-round ratio observed.
    lo: f64,
    /// Largest per-round ratio observed.
    hi: f64,
    /// How many rounds went into it.
    rounds: usize,
}

impl RatioStat {
    /// Per-round `num[r] / den[r]`, i.e. "how many times faster than `den` is `num`'s column" when
    /// both are times and `num` is the reference. Both slices must be the same length (one entry per
    /// round) and every entry finite and positive; anything else returns `None` rather than a
    /// plausible-looking number, because a dropped or failed round must not silently shrink the
    /// spread this type exists to report.
    fn over_rounds(num: &[f64], den: &[f64]) -> Option<RatioStat> {
        if num.len() != den.len() || num.is_empty() {
            return None;
        }
        let mut rs = Vec::with_capacity(num.len());
        for (&a, &b) in num.iter().zip(den) {
            if !(a.is_finite() && b.is_finite()) || a <= 0.0 || b <= 0.0 {
                return None;
            }
            rs.push(a / b);
        }
        let lo = rs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = rs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(RatioStat {
            med: median(&rs),
            lo,
            hi,
            rounds: rs.len(),
        })
    }

    /// How far the rounds disagreed, as a factor ≥ 1.
    fn spread(&self) -> f64 {
        self.hi / self.lo
    }

    /// Do the rounds agree closely enough for this ratio to be quoted as a number?
    fn conclusive(&self) -> bool {
        self.rounds >= 2 && self.spread() <= spread_limit()
    }
}

/// The spread above which a round-interleaved ratio is printed as INCONCLUSIVE instead of as a
/// number. `XBENCH_SPREAD_LIMIT` overrides it.
///
/// 1.10 is a floor on the *claims* this harness makes. It is deliberately NOT a description of the
/// machine, and how far the machine is from meeting it depends entirely on the confounds that have
/// been removed. Measured on 2026-08-06, `C(twin)` CONTROL — identical C source, identical compiler,
/// identical flags, a ratio with no language content in it whatever, true value 1.00 — over 9
/// section-runs per configuration, each pair of configurations INTERLEAVED with each other rather
/// than run in blocks:
///
/// | power | configuration | control per-round spread | inside 1.10 | control median off 1.00 |
/// |---|---|---|---|---|
/// | battery | as ec08db2 shipped it (unpinned, 6 x [1+3]) | median 1.290x, worst 1.509x | 0 of 9 | 3.3% typical, 7.5% worst |
/// | battery | pinned, 4 x [2 + best-of-7] (SHIPPED) | median 1.075x, worst 1.543x | 6 of 9 | 0.5% typical, 2.2% worst |
/// | AC+charging | as ec08db2 shipped it | median 1.702x, worst 2.169x | 0 of 9 | 9.9% typical, 27.3% worst |
/// | AC+charging | SHIPPED | median 1.049x, worst 1.503x | 6 of 9 | 1.9% typical, 4.3% worst |
///
/// The ec08db2 rows are why the harness went nearly silent: a control that fails in 9 sections of 9
/// leaves nothing to say, and on AC it was reading a byte-identical pair of DLLs 27% apart. The
/// SHIPPED rows are why it can speak again — in the same AC+charging set, 19 of 57 cross-language
/// cells resolved to a size against 0 of 57. The limit did not move; the instrument did. Loosening
/// the limit to make rows pass would invert its purpose.
///
/// Deliberately NOT adaptive. Widening the limit when a run is noisy would let exactly the worst runs
/// publish the most.
///
/// NOTE THE DIVISION OF LABOUR, because this limit alone is NOT enough and was never meant to be.
/// It is a gate on DISPERSION: it asks whether the rounds agreed. It cannot ask whether the effect is
/// bigger than the instrument, so a small-but-steady 1.03x sails through it — precisely the reading
/// `C(twin)` exists to catch. [`floor_factor`] is the gate on SIZE, measured per run from the
/// control, and [`verdict`] applies them in order: below the floor, nothing is printed at all,
/// whatever the spread.
fn spread_limit() -> f64 {
    parse_spread_limit(std::env::var("XBENCH_SPREAD_LIMIT").ok().as_deref())
}

/// The pure half of [`spread_limit`] — see [`parse_rounds`] for why the split exists. An override
/// that is not a factor strictly greater than 1 falls back to the default rather than disabling the
/// check: `XBENCH_SPREAD_LIMIT=99` is a visible, deliberate act, whereas a typo silently accepting
/// every ratio would be exactly the failure this module is here to prevent.
fn parse_spread_limit(v: Option<&str>) -> f64 {
    v.and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|&x| x > 1.0)
        .unwrap_or(1.10)
}

/// This run's own measured resolution, as a factor ≥ 1, from the `C(twin)` control.
///
/// The control's true value is exactly 1.00, so the largest departure from 1.00 it produced in
/// EITHER direction — `max(hi, 1/lo)` — is the smallest effect this run could not have manufactured
/// out of nothing. Taken symmetrically on purpose: the two DLLs are separate images and can land on
/// different alignments, so a control interval is not guaranteed to straddle 1.00, and a one-sided
/// band would then classify "no difference at all" as a resolved difference on the other side.
///
/// This is a FLOOR, never a certificate. It bounds the confounds the control SHARES with the real
/// columns (timing, scheduling, thermal drift between two slots of one round, image placement). It
/// cannot bound a confound unique to one language's column.
fn floor_factor(twin: RatioStat) -> f64 {
    twin.hi.max(1.0 / twin.lo).max(1.0)
}

/// What a run was able to say about one ratio, decided by the two gates in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    /// The whole per-round range lies outside the noise floor AND the rounds agree within
    /// [`spread_limit`]: direction and size both hold.
    Sized,
    /// The whole range lies outside the floor but the rounds disagree by more than the limit: the
    /// DIRECTION holds and the range is a bound, but the run did not pin the size.
    DirectionOnly,
    /// The range overlaps what a byte-identical control column did. Nothing at all: this machine in
    /// this state cannot resolve a difference this small.
    BelowFloor,
}

/// Classify a ratio against the run's own floor. With no control column there is no floor, and the
/// spread gate alone decides — which is strictly weaker, because a tight spread around a tiny effect
/// passes it. That is exactly the failure `C(twin)` was added to catch, so a run without a control
/// says so out loud rather than quietly reverting to the old behaviour.
fn verdict(s: RatioStat, floor: Option<f64>) -> Verdict {
    let resolved = match floor {
        Some(f) => s.lo > f || s.hi < 1.0 / f,
        None => true,
    };
    if !resolved {
        Verdict::BelowFloor
    } else if s.conclusive() {
        Verdict::Sized
    } else {
        Verdict::DirectionOnly
    }
}

/// Render a [`RatioStat`] as a table cell: its verdict against this run's floor, and always the range
/// the rounds spanned.
///
/// `reportable` is the power-state verdict for the section. When the state changed mid-section the
/// cell reads `NON-REPORTABLE` and carries NO number: a reader who scrolls past a warning line still
/// cannot come away with a figure, which is the whole difference between warning and invalidating.
///
/// `floor` is [`floor_factor`] of the `C(twin)` control, or `None` when the run has no control. The
/// point of printing `BELOW FLOOR` rather than a blank is that a reader should come away knowing the
/// INSTRUMENT'S RESOLUTION, not just that a cell was empty: a run where everything reads BELOW FLOOR
/// has measured its own resolution and reported it, which is a result.
fn fmt_ratio_stat(s: Option<RatioStat>, floor: Option<f64>, reportable: bool) -> String {
    let Some(s) = s else {
        return "-".to_string();
    };
    if !reportable {
        return "NON-REPORTABLE".to_string();
    }
    // Direction words are chosen from the range, not the median, in the two cases where the range is
    // all that holds: `lo > f` means every round read faster, `hi < 1/f` every round read slower.
    match verdict(s, floor) {
        Verdict::BelowFloor => format!("BELOW FLOOR [{:.2}-{:.2}]", s.lo, s.hi),
        Verdict::DirectionOnly if s.lo > 1.0 => {
            format!("FASTER, size ? [{:.2}-{:.2}]", s.lo, s.hi)
        }
        Verdict::DirectionOnly if s.hi < 1.0 => {
            format!("SLOWER, size ? [{:.2}-{:.2}]", 1.0 / s.hi, 1.0 / s.lo)
        }
        // No floor to clear and a range straddling 1.00: the run established neither size nor
        // direction. (Unreachable when a control exists — a straddling range always overlaps the
        // floor — so this arm is the no-control fallback.)
        Verdict::DirectionOnly => format!("INCONCLUSIVE [{:.2}-{:.2}]", s.lo, s.hi),
        Verdict::Sized if s.med >= 1.0 => format!("{:.2}x faster [{:.2}-{:.2}]", s.med, s.lo, s.hi),
        // Inverting the ratio inverts AND swaps the ends of the interval.
        Verdict::Sized => format!(
            "{:.2}x slower [{:.2}-{:.2}]",
            1.0 / s.med,
            1.0 / s.hi,
            1.0 / s.lo
        ),
    }
}

/// The legend that must sit under every round-interleaved table, so the cell vocabulary above is not
/// something a reader has to guess at. `floor` is [`floor_factor`] of this run's control.
fn ratio_legend(floor: Option<f64>) -> String {
    let head = match floor {
        Some(f) => format!(
            "  NOISE FLOOR, MEASURED THIS RUN: {f:.2}x. A byte-identical control column (C(twin):\n\
             \x20   same source, same compiler, same flags, second DLL, true ratio exactly 1.00) moved\n\
             \x20   by that much. Nothing smaller than it is resolvable on this machine in this state,\n\
             \x20   whatever it looks like."
        ),
        None => "  NOISE FLOOR: no control column in this run — the floor is UNKNOWN and every cell\n\
                 \x20   below is only as good as the spread gate, which a tiny-but-steady effect passes."
            .to_string(),
    };
    format!(
        "{head}\n  \
         vs C legend — \"1.90x slower [1.87-1.94]\": median of the per-round ratios and the range\n\
         \x20   the rounds spanned; printed only when the whole range clears the floor AND the rounds\n\
         \x20   agree within {:.2}x. \"FASTER/SLOWER, size ?\": every round cleared the floor in that\n\
         \x20   direction, so the direction holds and the range is a bound, but this run did not pin\n\
         \x20   the size. \"BELOW FLOOR\": the range overlaps what the control did — not resolvable.\n\
         \x20   \"NON-REPORTABLE\": the power state changed inside the section.",
        spread_limit()
    )
}

/// A SHORT same-protocol AVX2-FMA probe, sampled once per round as a relative clock trace.
///
/// This is NOT [`measure_fma_roofline`] and must never be quoted as a roofline: it warms for ~20 ms
/// and times ~8 ms blocks, so it reads the short-burst turbo clock rather than the sustained one.
/// Its only job is to answer "did this machine's clock move between round 1 and round R", in the same
/// units every round, so a reader can see the drift the ratios had to survive. Returns 0.0 without
/// AVX2/FMA.
///
/// It does NOT normalize anything. That was evaluated rather than assumed, on the raw per-round data
/// of three general runs (2026-08-06, battery) — the evaluation is `bench/noise-resistant-ratios`'
/// (ec08db2), carried over here rather than repeated, and it is a NEGATIVE result about a rejected
/// alternative, so nothing below depends on it:
///
/// * On the published RATIO it is a mathematical no-op. Scaling both columns of a per-round ratio by
///   that round's probe cancels: checked over all 57 (row, section, run) ratios, the normalized and
///   un-normalized per-round ratios agreed to 1e-12 in 57 of 57 cases.
/// * On the ABSOLUTE ms column — where a normaliser could in principle have helped — it makes things
///   strictly WORSE. Round-to-round spread of `ms` across 66 (column, section, run) triples: raw
///   median 1.266x / worst 3.447x, against 2.173x / 6.264x after scaling by the same round's probe.
///   It reduced the spread in 9 of 66 cases. A 20 ms-warmed probe reads short-burst turbo and is
///   simply noisier than the 15-second column it would be correcting, so it injects more variance
///   than it removes.
///
/// So the probe survives ONLY as a printed drift trace — evidence of what the ratios had to survive.
/// Do not promote it to a normaliser without redoing that measurement; on this data it fails.
///
/// AND IT IS OFF BY DEFAULT, because a probe that perturbs what it observes is worse than no probe.
/// It is a saturating AVX2-FMA burn loop that runs immediately before the first timed column of every
/// round: it heats the core, pulls the package into its AVX power license, and hands the round's
/// leading column a chip in a different state from the one the trailing column gets.
///
/// Measured, same binary, kill switch only, two interleaved runs per arm (2026-08-06, battery,
/// pinned) — DIRECTIONAL ONLY at n=2 per arm, but both directions point the same way and neither
/// favours keeping it on:
///
/// * it lowers the level it is supposed to be observing — block C best-of-rounds 120.3 / 114.6 ms
///   with the probe on against 144.7 / 148.8 ms with it off, i.e. the burn loop pre-boosts the core
///   the next column is then timed on;
/// * and it WIDENS the `C(twin)` control, whose true value is 1.00: per-round spread median 1.112x /
///   worst 1.416x with the probe on against 1.095x / 1.152x with it off.
///
/// `XBENCH_CLOCK_PROBE=1` turns it back on. Keep it opt-in: the honest per-round drift trace is the
/// raw ns table the sections already print, which costs nothing and perturbs nothing.
fn clock_probe_enabled() -> bool {
    matches!(
        std::env::var("XBENCH_CLOCK_PROBE").ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

fn clock_probe_gflops() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if clock_probe_enabled()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { fma_probe_avx2(20, 4_000_000, 3) };
        }
    }
    0.0
}

fn kernels() -> Vec<Kernel> {
    let nlit = N.to_string();
    vec![
        Kernel {
            name: "saxpy",
            bytes_per_call: 3 * N * 4,
            note: "y = a*x + y, memory-bound",
            wuk: wk_kernel(&format!(
                "let a: f32 = 2.0; for i in 0..{nlit} {{ out[i] = a * x[i] + y[i]; }}"
            )),
            c: c_kernel("float a=2.0f; for(long i=0;i<N;i++) out[i]=a*x[i]+y[i];"),
            rust: rust_kernel("let a=2.0f32; for i in 0..N { *out.add(i)=a* *x.add(i)+ *y.add(i); }"),
        },
        Kernel {
            name: "dot",
            bytes_per_call: 2 * N * 4,
            note: "sum(x*y) reduction — Wukong vectorizes it; gcc/rustc keep it serial",
            wuk: wk_kernel(&format!(
                "let mut s: f32 = 0.0; for i in 0..{nlit} {{ s = s + x[i] * y[i]; }} out[0] = s;"
            )),
            c: c_kernel("float s=0.0f; for(long i=0;i<N;i++) s+=x[i]*y[i]; out[0]=s;"),
            rust: rust_kernel(
                "let mut s=0.0f32; for i in 0..N { s+= *x.add(i)* *y.add(i); } *out.add(0)=s;",
            ),
        },
        Kernel {
            name: "ssd",
            bytes_per_call: 2 * N * 4,
            note: "sum((x-y)^2), an L2-loss reduction (Wukong vectorizes it)",
            wuk: wk_kernel(&format!(
                "let mut s: f32 = 0.0; for i in 0..{nlit} {{ s += (x[i] - y[i]) * (x[i] - y[i]); }} out[0] = s;"
            )),
            c: c_kernel("float s=0.0f; for(long i=0;i<N;i++){ float d=x[i]-y[i]; s+=d*d; } out[0]=s;"),
            rust: rust_kernel(
                "let mut s=0.0f32; for i in 0..N { let d= *x.add(i)- *y.add(i); s+=d*d; } *out.add(0)=s;",
            ),
        },
        Kernel {
            name: "relu",
            bytes_per_call: 2 * N * 4,
            note: "out = max(x,0), memory-bound + branch",
            wuk: wk_kernel(&format!(
                "for i in 0..{nlit} {{ out[i] = if x[i] > 0.0 {{ x[i] }} else {{ 0.0 }}; }}"
            )),
            c: c_kernel("for(long i=0;i<N;i++){ float v=x[i]; out[i] = v>0.0f? v:0.0f; }"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)= if v>0.0 {v} else {0.0}; }",
            ),
        },
        Kernel {
            name: "poly",
            bytes_per_call: 2 * N * 4,
            note: "degree-4 Horner per element, compute-bound (vectorizable)",
            wuk: wk_kernel(&format!(
                "for i in 0..{nlit} {{ let v: f32 = x[i]; let mut r: f32 = 0.00001; \
                 r = r * v + 0.0001; r = r * v + 0.001; r = r * v + 0.01; r = r * v + 0.1; out[i] = r; }}"
            )),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; float r=0.00001f; \
                 r=r*v+0.0001f; r=r*v+0.001f; r=r*v+0.01f; r=r*v+0.1f; out[i]=r; }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let mut r=0.00001f32; \
                 r=r*v+0.0001; r=r*v+0.001; r=r*v+0.01; r=r*v+0.1; *out.add(i)=r; }",
            ),
        },
        // Hadamard product `out = x*y` — gating / attention-mask / residual-scaling. Wukong dispatches
        // it to the 256-bit `wukong_velem_f32` (VE_HADAMARD) with non-temporal stores; gcc/rustc also
        // auto-vectorize a bare product, so at this L3-resident size it is a bandwidth tie (the NT-store
        // edge shows >L3, like saxpy). Recognized where the affine matcher declines (both factors vary).
        Kernel {
            name: "hadamard",
            bytes_per_call: 3 * N * 4,
            note: "out = x*y (Hadamard): 256-bit velem (+NT store); gcc/rustc autovectorize too",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = x[i] * y[i]; }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=x[i]*y[i];"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= *x.add(i)* *y.add(i); }"),
        },
        // SiLU written the textbook way, as a product `x*sigmoid(x)` (the form before the `silu()`
        // intrinsic, and the value==gate SwiGLU case). Wukong folds the product to the 256-bit
        // VMATH_SILU kernel; gcc/rustc must call scalar `expf` (no vectorized libm), so it stays serial
        // — a compute-bound win like the bare transcendentals, not a bandwidth tie.
        Kernel {
            name: "gated_silu",
            bytes_per_call: 2 * N * 4,
            note: "out = x*sigmoid(x): product folded to 256-bit VMATH_SILU; C/Rust scalar expf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = x[i] * sigmoid(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++){ float v=x[i]; out[i]= v/(1.0f+expf(-v)); }"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)= v/(1.0+(-v).exp()); }",
            ),
        },
        // argmax — the greedy-decode hot path (`next_token = argmax(logits)`). Wukong recognizes the
        // `if x[k] > bv { bv = x[k]; bi = k }` loop and dispatches to the deterministic argreduce kernel
        // (branchless 8-lane fold); gcc/rustc run a branchy scalar loop. Like `dot`, it writes only
        // out[0] (the index), so the full-buffer cross-check sees the same single value in all three.
        Kernel {
            name: "argmax",
            bytes_per_call: N * 4,
            note: "out[0] = argmax(x): branchless argreduce kernel vs a branchy scalar loop",
            wuk: wk_kernel(&format!(
                "let mut bv: f32 = x[0]; let mut bi: i64 = 0; \
                 for k in 0..{nlit} {{ if x[k] > bv {{ bv = x[k]; bi = k as i64; }} }} \
                 out[0] = bi as f32;"
            )),
            c: c_kernel(
                "float bv=x[0]; long bi=0; \
                 for(long k=0;k<N;k++){ if(x[k]>bv){ bv=x[k]; bi=k; } } out[0]=(float)bi;",
            ),
            rust: rust_kernel(
                "let mut bv= *x.add(0); let mut bi=0i64; \
                 for k in 0..N { let v= *x.add(k); if v>bv { bv=v; bi=k as i64; } } *out.add(0)=bi as f32;",
            ),
        },
        // Transcendentals: the regime a tensor compiler should dominate idiomatic scalar source.
        // Wukong lowers `exp` to a ~1-ULP f32 polynomial and auto-vectorizes it (128-bit x 4);
        // gcc/rustc call scalar libm `expf` per element and cannot vectorize a loop with a call
        // (no libmvec on this mingw toolchain), so the loop stays serial.
        Kernel {
            name: "exp",
            bytes_per_call: 2 * N * 4,
            note: "out = exp(x): Wukong vectorizes a ~1-ULP poly; C/Rust call scalar libm expf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = exp(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=expf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).exp(); }"),
        },
        // Natural log, the partner of exp (log-softmax / cross-entropy). The x buffer is all
        // positive (≥1), so this is a clean domain. Wukong vectorizes a ~1-ULP Cephes poly; C/Rust
        // call scalar libm logf.
        Kernel {
            name: "log",
            bytes_per_call: 2 * N * 4,
            note: "out = log(x): Wukong vectorizes a ~1-ULP poly; C/Rust call scalar libm logf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = log(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=logf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).ln(); }"),
        },
        // GELU (tanh approximation) written with the *identical* algorithm in all three languages —
        // tanh(z) built from exp as `1 - 2/(exp(2z)+1)`. The only difference is that Wukong
        // auto-vectorizes the exp; C/Rust call scalar expf. The fairest transcendental comparison.
        Kernel {
            name: "gelu",
            bytes_per_call: 2 * N * 4,
            note: "GELU (tanh approx): Wukong dispatches gelu() to a fused 256-bit AVX2 kernel; C/Rust scalar",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = gelu(x[i]); }}")),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; \
                 float u=0.7978845608f*(v+0.044715f*v*v*v); \
                 float t=1.0f-2.0f/(expf(2.0f*u)+1.0f); \
                 out[i]=0.5f*v*(1.0f+t); }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); \
                 let u=0.7978845608f32*(v+0.044715*v*v*v); \
                 let t=1.0f32-2.0/((2.0*u).exp()+1.0); \
                 *out.add(i)=0.5*v*(1.0+t); }",
            ),
        },
        // Batched GELU `out[r*C+j] = gelu(x[r*C+j])` over a flat [R, C] = 1024x1024 matrix (= N) — the
        // real [tokens, hidden] FFN/attention activation shape (the bare `gelu` above is one flat row).
        // Wukong's batched recognizer folds the nest to ONE flat 256-bit `wukong_vmath_f32` over the
        // whole [0, R*C) buffer; before that generalization the offset-indexed inner loop fell to the
        // 128-bit generic vectorizer (half the kernel width). C/Rust write the inner loop with scalar
        // libm `expf` (the tanh-approx gelu won't vectorize), so the structural win that the flat `gelu`
        // gets is now *also* captured for the nested transformer shape — at the full 256-bit width.
        Kernel {
            name: "gelu_batched",
            bytes_per_call: 2 * N * 4,
            note: "out[r*C+j]=gelu(x[r*C+j]) [R,C]=1024x1024: batched 256-bit dispatch vs scalar C/Rust",
            wuk: wk_kernel(
                "for r in 0..1024 { for j in 0..1024 { out[r * 1024 + j] = gelu(x[r * 1024 + j]); } }",
            ),
            c: c_kernel(
                "for(long r=0;r<1024;r++) for(long j=0;j<1024;j++){ long ix=r*1024+j; float v=x[ix]; \
                 float u=0.7978845608f*(v+0.044715f*v*v*v); \
                 float t=1.0f-2.0f/(expf(2.0f*u)+1.0f); \
                 out[ix]=0.5f*v*(1.0f+t); }",
            ),
            rust: rust_kernel(
                "for r in 0..1024 { for j in 0..1024 { let ix=r*1024+j; let v= *x.add(ix); \
                 let u=0.7978845608f32*(v+0.044715*v*v*v); \
                 let t=1.0f32-2.0/((2.0*u).exp()+1.0); \
                 *out.add(ix)=0.5*v*(1.0+t); } }",
            ),
        },
        // SiLU / swish: x * sigmoid(x), the activation in Llama/modern transformers. sigmoid is one
        // intrinsic that vectorizes; C/Rust spell it 1/(1+exp(-x)) with a scalar libm expf.
        Kernel {
            name: "silu",
            bytes_per_call: 2 * N * 4,
            note: "silu/swish x*sigmoid(x): Wukong dispatches silu() to a fused 256-bit AVX2 kernel; C/Rust scalar",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = silu(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++){ float v=x[i]; out[i]=v/(1.0f+expf(-v)); }"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)=v/(1.0f32+(-v).exp()); }",
            ),
        },
        // softplus ln(1+e^x): Wukong dispatches to the vectorized exp+log; C/Rust call scalar libm.
        // x ∈ [1,9] here, so the idiomatic naive log(1+exp(x)) is overflow-safe and equals Wukong's
        // stable max(x,0)+log(1+exp(-|x|)) to f32 tolerance.
        Kernel {
            name: "softplus",
            bytes_per_call: 2 * N * 4,
            note: "softplus ln(1+e^x): Wukong dispatches to a 256-bit AVX2 exp+log kernel; C/Rust scalar",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = softplus(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++){ float v=x[i]; out[i]=logf(1.0f+expf(v)); }"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)=(1.0f32+v.exp()).ln(); }",
            ),
        },
        // mish x*tanh(softplus(x)): the heaviest activation (three transcendentals), so the widest
        // gap vs scalar libm. C/Rust spell tanh and softplus with scalar expf/logf.
        Kernel {
            name: "mish",
            bytes_per_call: 2 * N * 4,
            note: "mish x*tanh(softplus(x)): Wukong dispatches to a 256-bit AVX2 kernel; C/Rust scalar",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = mish(x[i]); }}")),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; float sp=logf(1.0f+expf(v)); \
                 out[i]=v*(1.0f-2.0f/(expf(2.0f*sp)+1.0f)); }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let sp=(1.0f32+v.exp()).ln(); \
                 *out.add(i)=v*(1.0f32-2.0/((2.0*sp).exp()+1.0)); }",
            ),
        },
        // tanh activation, the *identical* exp-based algorithm in all three (Wukong vectorizes it).
        Kernel {
            name: "tanh",
            bytes_per_call: 2 * N * 4,
            note: "tanh via exp, same algorithm everywhere; Wukong vectorizes the exp",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = tanh(x[i]); }}")),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; out[i]=1.0f-2.0f/(expf(2.0f*v)+1.0f); }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)=1.0f32-2.0/((2.0*v).exp()+1.0); }",
            ),
        },
        // sin/cos: the rotary-position-embedding (RoPE) transcendentals every modern LLM precomputes.
        // Wukong dispatches a pure `out[i]=sin(x[i])` loop to the 256-bit AVX2 `wukong_vmath_f32`
        // (VM_SIN/VM_COS); gcc/rustc call scalar libm `sinf`/`cosf` and cannot vectorize a loop with the
        // call — the same compute-bound regime as `exp`.
        Kernel {
            name: "sin",
            bytes_per_call: 2 * N * 4,
            note: "out = sin(x): Wukong dispatches to a 256-bit AVX2 poly; C/Rust call scalar libm sinf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = sin(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=sinf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).sin(); }"),
        },
        Kernel {
            name: "cos",
            bytes_per_call: 2 * N * 4,
            note: "out = cos(x): Wukong dispatches to a 256-bit AVX2 poly; C/Rust call scalar libm cosf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = cos(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=cosf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).cos(); }"),
        },
        // erf: the exact (erf-based) GELU's core transcendental — BERT/GPT-2's original activation.
        // Wukong vectorizes the Abramowitz–Stegun poly at 256-bit; C calls scalar libm `erff`, and Rust
        // (no std erf) runs the idiomatic scalar A&S a programmer writes without a libm dependency.
        Kernel {
            name: "erf",
            bytes_per_call: 2 * N * 4,
            note: "out = erf(x): Wukong dispatches to a 256-bit AVX2 poly; C calls scalar libm erff",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = erf(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=erff(x[i]);"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let s=if v<0.0 {-1.0f32} else {1.0}; \
                 let ax=v.abs(); let t=1.0f32/(1.0+0.3275911*ax); \
                 let y=1.0f32-(((((1.061405429f32*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t)*(-ax*ax).exp(); \
                 *out.add(i)=s*y; }",
            ),
        },
        // Comprehensive vectorized elementwise math: base-2 exp/log (FlashAttention-2 base-2 softmax,
        // quantization bit-width, entropy in bits) and hyperbolic sinh/cosh. Wukong dispatches each to
        // the 256-bit `wukong_vmath_f32`; C/Rust call scalar libm `exp2f`/`log2f`/`sinhf`/`coshf` and
        // cannot vectorize the call — the same compute-bound win as `exp`.
        Kernel {
            name: "exp2",
            bytes_per_call: 2 * N * 4,
            note: "out = exp2(x) = 2^x: 256-bit AVX2 (exp(x·ln2)) vs scalar libm exp2f",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = exp2(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=exp2f(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).exp2(); }"),
        },
        Kernel {
            name: "log2",
            bytes_per_call: 2 * N * 4,
            note: "out = log2(x): 256-bit AVX2 (log(x)·log2e) vs scalar libm log2f",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = log2(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=log2f(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).log2(); }"),
        },
        Kernel {
            name: "sinh",
            bytes_per_call: 2 * N * 4,
            note: "out = sinh(x) = (e^x - e^-x)/2: 256-bit AVX2 vs scalar libm sinhf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = sinh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=sinhf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).sinh(); }"),
        },
        Kernel {
            name: "cosh",
            bytes_per_call: 2 * N * 4,
            note: "out = cosh(x) = (e^x + e^-x)/2: 256-bit AVX2 vs scalar libm coshf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = cosh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=coshf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).cosh(); }"),
        },
        // Inverse hyperbolics — asinh = log(|x|+√(x²+1)) (sign-stable), acosh = log(x+√(x²−1)) for x≥1
        // (the shared input is [1,9], in-domain for both). Compose the shared 256-bit `log`; C/Rust call
        // scalar libm `asinhf`/`acoshf` and can't vectorize the call. `atanh` shares the identical
        // kernel shape (log + a divide) but needs |x|<1, outside this buffer's range, so it isn't
        // separately timed here — its throughput tracks these.
        Kernel {
            name: "asinh",
            bytes_per_call: 2 * N * 4,
            note: "out = asinh(x): 256-bit AVX2 (log+sqrt) vs scalar libm asinhf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = asinh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=asinhf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).asinh(); }"),
        },
        Kernel {
            name: "acosh",
            bytes_per_call: 2 * N * 4,
            note: "out = acosh(x), x>=1: 256-bit AVX2 (log+sqrt) vs scalar libm acoshf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = acosh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=acoshf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).acosh(); }"),
        },
        // atan (all-real) — Cephes 3-region reduction + degree-3 poly at 256-bit; C/Rust call scalar
        // libm atanf and can't vectorize the call. Angle/geometry ops, atan2-style positional schemes.
        Kernel {
            name: "atan",
            bytes_per_call: 2 * N * 4,
            note: "out = atan(x): 256-bit AVX2 (Cephes) vs scalar libm atanf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = atan(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=atanf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).atan(); }"),
        },
        // expm1/log1p — the numerically-stable eˣ−1 / ln(1+x) (Kahan, reusing the 256-bit exp/log);
        // C/Rust call scalar libm expm1f/log1pf and can't vectorize the call. ELU exact tail, stable losses.
        Kernel {
            name: "expm1",
            bytes_per_call: 2 * N * 4,
            note: "out = expm1(x) = e^x-1: 256-bit AVX2 (Kahan) vs scalar libm expm1f",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = expm1(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=expm1f(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).exp_m1(); }"),
        },
        Kernel {
            name: "log1p",
            bytes_per_call: 2 * N * 4,
            note: "out = log1p(x) = ln(1+x): 256-bit AVX2 (Kahan) vs scalar libm log1pf",
            wuk: wk_kernel(&format!("for i in 0..{nlit} {{ out[i] = log1p(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=log1pf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).ln_1p(); }"),
        },
        // Operator fusion: a linear map then ReLU, written as TWO loops in every language. Wukong's
        // compiler fuses them into one pass (intermediate stays in registers, not streamed to the
        // scratch array `y`); idiomatic C/Rust as-written make two passes over `y`.
        Kernel {
            name: "fused_linear_relu",
            bytes_per_call: 2 * N * 4,
            note: "linear→relu: Wukong auto-fuses 2 loops; C/Rust as-written stream the intermediate",
            wuk: format!(
                "module bench\nfn kbench(x: [f32; {N}], mut y: [f32; {N}], mut out: [f32; {N}]) {{\n    \
                 for i in 0..{N} {{ y[i] = 2.0 * x[i] + 1.0; }} \
                 for i in 0..{N} {{ out[i] = if y[i] > 0.0 {{ y[i] }} else {{ 0.0 }}; }}\n}}\n"
            ),
            c: c_kernel_rw(
                "for(long i=0;i<N;i++) y[i]=2.0f*x[i]+1.0f; \
                 for(long i=0;i<N;i++){ float v=y[i]; out[i]= v>0.0f? v:0.0f; }",
            ),
            rust: rust_kernel(
                "let t = y as *mut f32; for i in 0..N { *t.add(i)=2.0* *x.add(i)+1.0; } \
                 for i in 0..N { let v= *t.add(i); *out.add(i)= if v>0.0 {v} else {0.0}; }",
            ),
        },
        // --- Wukong @parallel (multicore) vs idiomatic single-threaded C/Rust ---
        Kernel {
            name: "saxpy@parallel",
            bytes_per_call: 3 * N * 4,
            note: "Wukong auto-parallel across cores vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "for i in 0..{N} {{ out[i] = 2.0 * x[i] + y[i]; }}"
            )),
            c: c_kernel("float a=2.0f; for(long i=0;i<N;i++) out[i]=a*x[i]+y[i];"),
            rust: rust_kernel("let a=2.0f32; for i in 0..N { *out.add(i)=a* *x.add(i)+ *y.add(i); }"),
        },
        Kernel {
            name: "poly@parallel",
            bytes_per_call: 2 * N * 4,
            note: "Wukong auto-parallel across cores vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "for i in 0..{N} {{ let v: f32 = x[i]; let mut r: f32 = 0.00001; \
                 r = r * v + 0.0001; r = r * v + 0.001; r = r * v + 0.01; r = r * v + 0.1; out[i] = r; }}"
            )),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; float r=0.00001f; \
                 r=r*v+0.0001f; r=r*v+0.001f; r=r*v+0.01f; r=r*v+0.1f; out[i]=r; }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let mut r=0.00001f32; \
                 r=r*v+0.0001; r=r*v+0.001; r=r*v+0.01; r=r*v+0.1; *out.add(i)=r; }",
            ),
        },
        Kernel {
            name: "relu6@parallel",
            bytes_per_call: 2 * N * 4,
            note: "clamp(x,0,6) — nested branch vectorization (if-conversion) × cores",
            wuk: wk_par_kernel(&format!(
                "for i in 0..{N} {{ out[i] = if x[i] < 6.0 {{ if x[i] > 0.0 {{ x[i] }} else {{ 0.0 }} }} else {{ 6.0 }}; }}"
            )),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; v = v<6.0f?v:6.0f; out[i] = v>0.0f?v:0.0f; }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let v= if v<6.0 {v} else {6.0}; *out.add(i)= if v>0.0 {v} else {0.0}; }",
            ),
        },
        // GELU across cores: each thread's chunk dispatches to the fused 256-bit AVX2 gelu kernel, so
        // this is multicore × 256-bit vs single-threaded scalar C — the compute-bound activation
        // sweep over a large tensor (e.g. a transformer FFN's hidden state) where it pays the most.
        Kernel {
            name: "gelu@parallel",
            bytes_per_call: 2 * N * 4,
            note: "GELU across cores (multicore × 256-bit AVX2) vs single-threaded scalar C/Rust",
            wuk: wk_par_kernel(&format!("for i in 0..{N} {{ out[i] = gelu(x[i]); }}")),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; \
                 float u=0.7978845608f*(v+0.044715f*v*v*v); \
                 float t=1.0f-2.0f/(expf(2.0f*u)+1.0f); \
                 out[i]=0.5f*v*(1.0f+t); }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); \
                 let u=0.7978845608f32*(v+0.044715*v*v*v); \
                 let t=1.0f32-2.0/((2.0*u).exp()+1.0); \
                 *out.add(i)=0.5*v*(1.0+t); }",
            ),
        },
        // Reductions across cores: a `@parallel` reduction loop dispatches to the multicore
        // deterministic reduction kernel (wukong_sreduce_f32_parallel), so this is one core's
        // memory bandwidth × all cores vs the single-threaded sequential C/Rust reduction (which
        // gcc/rustc keep latency-bound on the dependent add chain). The transformer-relevant case:
        // attention scores, L2 losses, LayerNorm sums over a large activation.
        Kernel {
            name: "dot@parallel",
            bytes_per_call: 2 * N * 4,
            note: "sum(x*y) across cores (multicore reduction kernel) vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "let mut s: f32 = 0.0; for k in 0..{N} {{ s = s + x[k] * y[k]; }} out[0] = s;"
            )),
            c: c_kernel("float s=0.0f; for(long i=0;i<N;i++) s+=x[i]*y[i]; out[0]=s;"),
            rust: rust_kernel(
                "let mut s=0.0f32; for i in 0..N { s+= *x.add(i)* *y.add(i); } *out.add(0)=s;",
            ),
        },
        Kernel {
            name: "ssd@parallel",
            bytes_per_call: 2 * N * 4,
            note: "sum((x-y)^2) across cores (multicore reduction kernel) vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "let mut s: f32 = 0.0; for k in 0..{N} {{ s += (x[k] - y[k]) * (x[k] - y[k]); }} out[0] = s;"
            )),
            c: c_kernel("float s=0.0f; for(long i=0;i<N;i++){ float d=x[i]-y[i]; s+=d*d; } out[0]=s;"),
            rust: rust_kernel(
                "let mut s=0.0f32; for i in 0..N { let d= *x.add(i)- *y.add(i); s+=d*d; } *out.add(0)=s;",
            ),
        },
        // Max across cores — the per-tensor max softmax stability and dynamic int8 quantization
        // (absmax → scale = absmax/127) need. A `@parallel` fmax reduction dispatches to the multicore
        // kernel (RED_MAX); the C/Rust baselines are the idiomatic compare-select max the compiler
        // can auto-vectorize (the honest single-thread bar), so the win is one core's bandwidth ×
        // all cores. Single input stream, so bytes = N·4 (unary).
        Kernel {
            name: "max@parallel",
            bytes_per_call: N * 4,
            note: "max(x) across cores (multicore reduction kernel) vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "let mut m: f32 = x[0]; for k in 0..{N} {{ m = fmax(m, x[k]); }} out[0] = m;"
            )),
            c: c_kernel("float m=x[0]; for(long i=0;i<N;i++){ float v=x[i]; m = v>m? v:m; } out[0]=m;"),
            rust: rust_kernel(
                "let mut m= *x.add(0); for i in 0..N { let v= *x.add(i); if v>m { m=v; } } *out.add(0)=m;",
            ),
        },
        // Absmax across cores — the per-tensor max|x| symmetric dynamic int8 quantization needs for
        // the scale (scale = absmax/127). A `@parallel` fmax(m, abs(x[k])) loop → the multicore
        // RED_MAXABS kernel; C/Rust use the branchless `fabsf`/`.abs()` (a recognized bitwise builtin,
        // not a libm call), then a compare-select max the compiler still keeps serial without
        // -ffast-math. Single input stream, bytes = N*4 (unary).
        Kernel {
            name: "absmax@parallel",
            bytes_per_call: N * 4,
            note: "max(|x|) across cores (multicore reduction kernel) vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "let mut m: f32 = 0.0; for k in 0..{N} {{ m = fmax(m, abs(x[k])); }} out[0] = m;"
            )),
            c: c_kernel(
                "float m=0.0f; for(long i=0;i<N;i++){ float a=fabsf(x[i]); m = a>m? a:m; } out[0]=m;",
            ),
            rust: rust_kernel(
                "let mut m=0.0f32; for i in 0..N { let a=(*x.add(i)).abs(); if a>m { m=a; } } *out.add(0)=m;",
            ),
        },
        // Argmax across cores — the greedy-decode top-1 (`next_token = argmax(logits)`) parallelized.
        // A `@parallel` argmax loop dispatches to the multicore `wukong_argreduce_f32_parallel`: the
        // FIXED RCHUNK decomposition folded in ascending order, so the returned index is bit-identical
        // to the serial kernel and to the interpreter, independent of thread count. C/Rust are the
        // idiomatic single-threaded branchy scalar argmax (the honest bar); the win is one core's
        // branchless 8-lane (value,index) fold × all cores. Writes only out[0] (the index, exact in f32
        // since N < 2^24), like `argmax`/`dot`, so the full-buffer cross-check agrees in all three.
        Kernel {
            name: "argmax@parallel",
            bytes_per_call: N * 4,
            note: "argmax(x) across cores (multicore argreduce kernel) vs single-threaded C/Rust",
            wuk: wk_par_kernel(&format!(
                "let mut bv: f32 = x[0]; let mut bi: i64 = 0; \
                 for k in 0..{N} {{ if x[k] > bv {{ bv = x[k]; bi = k as i64; }} }} \
                 out[0] = bi as f32;"
            )),
            c: c_kernel(
                "float bv=x[0]; long bi=0; \
                 for(long k=0;k<N;k++){ if(x[k]>bv){ bv=x[k]; bi=k; } } out[0]=(float)bi;",
            ),
            rust: rust_kernel(
                "let mut bv= *x.add(0); let mut bi=0i64; \
                 for k in 0..N { let v= *x.add(k); if v>bv { bv=v; bi=k as i64; } } *out.add(0)=bi as f32;",
            ),
        },
    ]
}

/// Streaming elementwise at a **>L3** tensor size — the regime that matters for real activation
/// tensors (a `[batch, seq, hidden]` block is hundreds of MB, far larger than the 4 MiB kernels
/// above). At N=2²⁴ (64 MiB/array) the working set cannot stay in cache, so Wukong's recognized
/// streaming maps dispatch to `wukong_velem_f32` with **non-temporal** stores, skipping the
/// read-for-ownership traffic gcc/rustc pay on every cacheable store (a store path they will not emit
/// automatically). The lead is larger here than at 2²⁰, where the RFO is partly absorbed by L3.
fn bench_streaming_large(cc: &str, dir: &Path) {
    const NL: usize = 1 << 24; // 16,777,216 elements, 64 MiB per f32 array
    let x: Vec<f32> = (0..NL).map(|i| (i % 17) as f32 * 0.5 - 3.0).collect();
    let y: Vec<f32> = (0..NL).map(|i| (i % 13) as f32 * 0.25 - 0.5).collect();
    let mut out = vec![0.0f32; NL];
    let (xp, yp) = (x.as_ptr(), y.as_ptr());
    println!("=== streaming elementwise at N=2^24 (64 MiB/array, >L3; GB/s, higher is better) ===");
    println!(
        "  {:<10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>14}",
        "", "Wukong", "C (gcc)", "C++ (g++)", "Rust", "Wuk vs C", "Wuk vs C++"
    );
    // (name, body-in-each-language(identical math), traffic bytes/call)
    let cases: [(&str, String, String, String, usize); 4] = [
        (
            "saxpy",
            format!("for i in 0..{NL} {{ out[i] = 2.0 * x[i] + y[i]; }}"),
            "for(long i=0;i<N;i++) out[i]=2.0f*x[i]+y[i];".into(),
            "for i in 0..N { *out.add(i)=2.0* *x.add(i)+ *y.add(i); }".into(),
            3 * NL * 4,
        ),
        (
            "residual",
            format!("for i in 0..{NL} {{ out[i] = x[i] + y[i]; }}"),
            "for(long i=0;i<N;i++) out[i]=x[i]+y[i];".into(),
            "for i in 0..N { *out.add(i)= *x.add(i)+ *y.add(i); }".into(),
            3 * NL * 4,
        ),
        (
            "scale",
            format!("for i in 0..{NL} {{ out[i] = 0.5 * x[i]; }}"),
            "for(long i=0;i<N;i++) out[i]=0.5f*x[i];".into(),
            "for i in 0..N { *out.add(i)=0.5* *x.add(i); }".into(),
            2 * NL * 4,
        ),
        (
            "relu",
            format!("for i in 0..{NL} {{ out[i] = if x[i] > 0.0 {{ x[i] }} else {{ 0.0 }}; }}"),
            "for(long i=0;i<N;i++){ float v=x[i]; out[i]= v>0.0f? v:0.0f; }".into(),
            "for i in 0..N { let v= *x.add(i); *out.add(i)= if v>0.0 {v} else {0.0}; }".into(),
            2 * NL * 4,
        ),
    ];
    for (name, mb, cb, rb, bytes) in &cases {
        let built_wuk = build_wukong(&wk_kernel_n(NL, mb), 3);
        let (built_c, built_c_cpp) = build_c_cpp(
            &format!("stream_{name}"),
            &c_kernel_n(NL, cb),
            dir,
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        );
        let built_rust = build_peer(
            "rs",
            &rust_kernel_n(NL, rb),
            dir,
            &format!("stream_{name}"),
            "rustc",
            &[
                "-Copt-level=3",
                "-Ctarget-cpu=native",
                "--crate-type=cdylib",
            ],
        );
        // Every compiled column keeps its slot, including unavailable peers.
        let columns = [
            Built::wk(built_wuk),
            Built::peer(built_c),
            Built::peer(built_c_cpp),
            Built::peer(built_rust),
        ];
        let mut timed =
            abba_min_columns(columns.len(), |i| time_built(&columns[i], &mut out, xp, yp))
                .into_iter();
        let wuk = timed.next().flatten();
        let c = timed.next().flatten();
        let c_cpp = cpp_checked(&format!("stream_{name}"), &c, timed.next().flatten());
        let rust = timed.next().flatten();

        let gbs = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", *bytes as f64 / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        let standing_vs = |peer: &Option<Measure>| {
            if let (Some(m), Some(p)) = (&wuk, peer) {
                let r = p.ns_per_call / m.ns_per_call;
                format!(
                    "{:.2}x {}",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                )
            } else {
                "n/a".into()
            }
        };
        println!(
            "  {:<10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>14}",
            name,
            gbs(&wuk),
            gbs(&c),
            gbs(&c_cpp),
            gbs(&rust),
            standing_vs(&c),
            standing_vs(&c_cpp)
        );
        if let (Some(m), Some(c)) = (&wuk, &c) {
            let (rel, at) = max_rel_err(&m.out, &c.out);
            if rel > 1e-4 {
                println!(
                    "  ! {name} full-buffer mismatch vs C at [{at}]: Wukong={} C={} (rel {:.2e})",
                    m.out[at], c.out[at], rel
                );
            }
        }
    }
    println!();
}

fn wk_kernel(body: &str) -> String {
    format!(
        "module bench\nfn kbench(x: [f32; {N}], y: [f32; {N}], mut out: [f32; {N}]) {{\n    {body}\n}}\n"
    )
}

/// A `@parallel` Wukong kernel (whole body is one `for` loop, so it parallelizes).
fn wk_par_kernel(loop_body: &str) -> String {
    format!(
        "module bench\n@parallel\nfn kbench(x: [f32; {N}], y: [f32; {N}], mut out: [f32; {N}]) {{\n    {loop_body}\n}}\n"
    )
}

fn c_kernel(body: &str) -> String {
    format!("#include <math.h>\n#define N {N}\n__declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out) {{\n  {body}\n}}\n")
}

/// [`c_kernel`] with a **writable** middle buffer. `fused_linear_relu` streams its intermediate
/// through `y` (the harness derives `yp` with `as_mut_ptr`, so the write has provenance), and the
/// old spelling reached it by casting the `const` away: `float* t = (float*)y;`. That is fatal once
/// the parameters carry `__restrict__` — C11 6.7.3.1p4 makes it undefined behaviour to modify an
/// object designated by a `restrict` pointer to a **const-qualified** type, so gcc would be entitled
/// to assume `y` never changes and hoist the second loop's loads above the first loop's stores.
/// Declaring the middle `float* __restrict__ y` states the truth instead of casting it away.
fn c_kernel_rw(body: &str) -> String {
    format!("#include <math.h>\n#define N {N}\n__declspec(dllexport) void kbench(const float* __restrict__ x, float* __restrict__ y, float* __restrict__ out) {{\n  {body}\n}}\n")
}

/// The Rust twin of [`c_kernel`] — **including its aliasing information**.
///
/// The C peers have declared every pointer parameter `__restrict__` since the 2026-08-04 audit. The
/// Rust peers were still taking bare `*const f32` / `*mut f32`, which carries **no** aliasing
/// information at all: rustc emits `noalias` on *reference* parameters, never on raw pointers, so
/// every Rust peer in this file was compiled under may-alias assumptions while its C twin was not.
/// Verified directly on the `saxpy` body at `-Copt-level=3 -Ctarget-cpu=native`:
///
/// | spelling | asm |
/// |---|---|
/// | raw pointers | 108 lines, **4 blocks** — a runtime `\|out-x\|<128 \|\| \|out-y\|<128` overlap test and a fully scalar `vaddss` fallback loop beside the vector one |
/// | slices threaded through a `fn` boundary | 56 lines, **1 block**, zero scalar FP |
///
/// LANDMINE: `let x = core::slice::from_raw_parts(x, N);` **at the top of `kbench` does not do it.**
/// That spelling was measured to emit byte-identical code to the raw-pointer one (108 lines, 4
/// blocks) and its LLVM-IR carries no `llvm.experimental.noalias.scope.decl` at all, because rustc
/// attaches `noalias` to function *parameters*. The slices must cross a **function boundary**; the
/// `#[inline(always)]` shim below is what turns them into `!alias.scope`/`!noalias` metadata that
/// survives inlining. Several peers in this file (`rust_colsum`, `rust_transpose`, `rust_model`, …)
/// built slices at the top of `kbench` and were documented as getting `noalias` from it; they did
/// not, and are routed through shims now.
///
/// The kernel `{body}` text is **unchanged** by the shim — it still indexes through `x`/`y`/`out`
/// raw pointers, so no bounds check appears and no accumulation order moves. Only the aliasing facts
/// change, which is exactly what `__restrict__` does for the C column.
///
/// `y` is `&mut` because the harness derives `yp` with `as_mut_ptr` and the `fused_linear_relu` row
/// streams its intermediate through it (its C twin is [`c_kernel_rw`], `float* __restrict__ y`).
/// `#[allow(unused_variables)]`: some kernels (relu, poly) don't read `y`; the fixed `(x,y,out)` ABI
/// keeps the param, so silence the warning rather than clutter the benchmark output.
fn rust_kernel(body: &str) -> String {
    format!(
        "#[allow(dead_code)]\nconst N: usize = {N};\n\
         #[inline(always)]\n#[allow(unused_variables)]\n\
         unsafe fn kbody(xs: &[f32], ys: &mut [f32], os: &mut [f32]) {{\n  \
         let (x, y, out) = (xs.as_ptr(), ys.as_mut_ptr(), os.as_mut_ptr());\n  {body}\n}}\n\
         #[no_mangle]\n#[allow(unused_variables)]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  \
         kbody(core::slice::from_raw_parts(x, N), \
         core::slice::from_raw_parts_mut(y as *mut f32, N), \
         core::slice::from_raw_parts_mut(out, N))\n}}\n"
    )
}

/// Render a C kernel source as **C++** for the g++ peer column. C is a subset of C++, so the numeric
/// body compiles unchanged; we only prepend `extern "C"` to the export so the C++ compiler keeps the
/// `kbench` symbol unmangled (libloading looks it up by that exact name — exactly how real C++ projects
/// expose a C-ABI kernel). `<math.h>` stays (valid C++, keeps `expf`/`logf`/… in global scope), so no
/// body rewrite is needed. Idiomatic C++ for a numeric kernel *is* this loop (a `std::transform` lowers
/// to the same code), and g++ shares gcc's middle/back-end — so this measures whether the C++ toolchain
/// beats Wukong (it does not), proving the "beat C++" claim instead of assuming it.
fn cpp_from_c(c_src: &str) -> String {
    c_src.replace(
        "__declspec(dllexport)",
        "extern \"C\" __declspec(dllexport)",
    )
}

// Parameterized kernel builders (an explicit element count `n`) — used by the large-tensor streaming
// benchmark, which runs at N=2²⁴ rather than the module-global N=2²⁰.
fn wk_kernel_n(n: usize, body: &str) -> String {
    format!("module bench\nfn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [f32; {n}]) {{\n    {body}\n}}\n")
}
fn c_kernel_n(n: usize, body: &str) -> String {
    format!("#include <math.h>\n#define N {n}\n__declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out) {{\n  {body}\n}}\n")
}
/// [`rust_kernel`]'s parameterized twin, with the same `noalias` shim. `y` stays `&[f32]` here (not
/// `&mut` as in [`rust_kernel`]) because the streaming bench derives `yp` with `as_ptr()` — building
/// a `&mut` from a shared-provenance pointer would be UB even though no streaming body writes `y`.
fn rust_kernel_n(n: usize, body: &str) -> String {
    format!(
        "const N: usize = {n};\n\
         #[inline(always)]\n#[allow(unused_variables)]\n\
         unsafe fn kbody(xs: &[f32], ys: &[f32], os: &mut [f32]) {{\n  \
         let (x, y, out) = (xs.as_ptr(), ys.as_ptr(), os.as_mut_ptr());\n  {body}\n}}\n\
         #[no_mangle]\n#[allow(unused_variables)]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  \
         kbody(core::slice::from_raw_parts(x, N), core::slice::from_raw_parts(y, N), \
         core::slice::from_raw_parts_mut(out, N))\n}}\n"
    )
}

/// Regression guards for the pure decision functions this file's peer columns and correctness
/// checks hang off. None of them were covered before: `crates/wukong_xbench` had no test over
/// `main.rs` at all, so a rename that silently DROPPED a peer column — weakening the comparison
/// rather than breaking the build — left `cargo test` green.
#[cfg(test)]
mod tests {
    use super::*;

    /// `XBENCH_PIN` must default to pinning and must never fall THROUGH to "off" on a typo.
    ///
    /// The direction of the fallback is the whole point. Unpinned, this section's control column
    /// (`C(twin)`, true ratio 1.00) measured 2.12x spread and the C column drifted 81 -> 174 ms
    /// inside one run; pinned it measured ~1.05x and stayed flat. A mistyped `XBENCH_PIN=cpu0`
    /// silently turning the pin off would hand a reader a page of numbers with no resolution at all
    /// and no indication that anything had changed.
    #[test]
    fn pin_defaults_on_and_a_typo_does_not_disable_it() {
        assert_eq!(parse_pin(None), Some(0b11), "unset must pin");
        for off in ["off", "OFF", "none", "0", " off "] {
            assert_eq!(parse_pin(Some(off)), None, "XBENCH_PIN={off:?}");
        }
        assert_eq!(parse_pin(Some("4")), Some(1 << 4));
        assert_eq!(parse_pin(Some(" 4 , 5 ")), Some((1 << 4) | (1 << 5)));
        for junk in ["cpu0", "", "-1", "4,cpu5", "999"] {
            assert_eq!(
                parse_pin(Some(junk)),
                Some(0b11),
                "XBENCH_PIN={junk:?} must fall back to the default, not to off"
            );
        }
        // And the live reader must agree with the pure one under the ambient environment.
        assert_eq!(
            pin_mask(),
            parse_pin(std::env::var("XBENCH_PIN").ok().as_deref())
        );
    }

    /// A pin must be UNDONE. `bench_general` runs inside one process with the rest of the suite, and
    /// a section that leaked a one-core affinity would cap every later multicore row — which would
    /// read as a performance finding, not as a benchmark bug.
    #[test]
    fn the_pin_guard_restores_the_previous_affinity() {
        let Some(before) = set_thread_affinity(!0usize) else {
            return; // not Windows, or the OS refused: nothing to restore, nothing to prove.
        };
        {
            let _g = PinGuard(set_thread_affinity(0b1));
            // Re-reading the mask requires setting it, so prove the restore instead: the guard was
            // handed the mask that was live before it narrowed things.
        }
        // After the guard, setting a fresh mask must report the FULL mask as the previous one, i.e.
        // the guard put it back rather than leaving us on CPU 0.
        let after = set_thread_affinity(!0usize).expect("affinity readable");
        assert!(
            after.count_ones() > 1,
            "still pinned to one CPU after the guard: {after:#x}"
        );
        let _ = set_thread_affinity(before);
    }

    /// Every `@parallel` row must have an OpenMP twin in [`c_omp_source`]. Rename a row in
    /// [`kernels()`] without renaming its arm and `c_omp_source` falls through to `None`: the
    /// C(omp) column vanishes with no diagnostic, and the multicore-vs-multicore comparison is
    /// silently replaced by multicore-Wukong-vs-1-thread-C.
    #[test]
    fn every_parallel_row_has_an_omp_peer() {
        for k in kernels() {
            if k.name.contains("@parallel") {
                assert!(
                    c_omp_source(k.name).is_some(),
                    "`{}` is a @parallel row with no c_omp_source arm — its C(omp) column would \
                     silently disappear",
                    k.name
                );
            }
        }
    }

    /// Rows built from byte-identical C source must get identical peer-column treatment. This is
    /// the invariant `argmax` / `argmax@parallel` broke: the same C string, but only the parallel
    /// row was listed in [`is_reduction_kernel`], so only it got the `-ffast-math` C(fast)
    /// normalization while the serial row published an un-normalized ratio.
    #[test]
    fn identical_c_sources_get_identical_peer_columns() {
        let ks = kernels();
        for i in 0..ks.len() {
            for j in (i + 1)..ks.len() {
                if ks[i].c == ks[j].c {
                    assert_eq!(
                        is_reduction_kernel(ks[i].name),
                        is_reduction_kernel(ks[j].name),
                        "`{}` and `{}` have byte-identical C source but disagree on the C(fast) \
                         column",
                        ks[i].name,
                        ks[j].name
                    );
                }
            }
        }
    }

    /// Both peer selectors are keyed by hard-coded kernel names; a name that no longer exists in
    /// [`kernels()`] is a dead arm and a sign the two lists have drifted.
    #[test]
    fn peer_selector_names_exist_in_the_catalogue() {
        let names: Vec<&str> = kernels().iter().map(|k| k.name).collect();
        for n in [
            "dot",
            "ssd",
            "argmax",
            "dot@parallel",
            "ssd@parallel",
            "max@parallel",
            "absmax@parallel",
            "argmax@parallel",
        ] {
            assert!(names.contains(&n), "`{n}` is no longer a kernel row");
            assert!(
                is_reduction_kernel(n),
                "`{n}` dropped out of is_reduction_kernel"
            );
        }
        for n in [
            "saxpy@parallel",
            "poly@parallel",
            "relu6@parallel",
            "gelu@parallel",
            "dot@parallel",
            "ssd@parallel",
            "max@parallel",
            "absmax@parallel",
            "argmax@parallel",
        ] {
            assert!(names.contains(&n), "`{n}` is no longer a kernel row");
            assert!(
                c_omp_source(n).is_some(),
                "`{n}` dropped out of c_omp_source"
            );
        }
    }

    /// `max_rel_err` is the suite's cross-language equality metric; 0.0 is its "bit-exact"
    /// sentinel, so every way of reaching 0.0 on unequal buffers is a vacuous pass.
    #[test]
    fn max_rel_err_reports_divergence() {
        assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), (0.0, 0));
        // ±0 and equal Inf are exact; NaN agrees with NaN.
        assert_eq!(max_rel_err(&[0.0, -0.0], &[-0.0, 0.0]).0, 0.0);
        assert_eq!(
            max_rel_err(&[f32::INFINITY, f32::NAN], &[f32::INFINITY, f32::NAN]).0,
            0.0
        );
        // NaN vs finite is a divergence, not a skip (this used to return 0.0 = "bit-exact").
        let (rel, at) = max_rel_err(&[1.0, f32::NAN, 3.0], &[1.0, 0.42, 3.0]);
        assert!(
            rel.is_infinite(),
            "NaN vs finite must not read as bit-exact"
        );
        assert_eq!(at, 1);
        assert!(max_rel_err(&[0.42], &[f32::NAN]).0.is_infinite());
        // +Inf vs -Inf makes `rel` NaN, which must not be silently skipped either.
        assert!(max_rel_err(&[f32::INFINITY], &[f32::NEG_INFINITY])
            .0
            .is_infinite());
        // A real relative difference, magnitude-normalized.
        let (rel, at) = max_rel_err(&[100.0, 1.0], &[100.0, 1.5]);
        assert_eq!(at, 1);
        assert!((rel - 1.0 / 3.0).abs() < 1e-9, "rel = {rel}");
    }

    /// bf16 conversion is round-to-nearest-EVEN on the discarded low half — a truncating version
    /// would bias every bf16 row's inputs low.
    #[test]
    fn to_bf16_bits_rounds_to_nearest_even() {
        // Exactly representable: the low 16 bits are already zero.
        assert_eq!(to_bf16_bits(1.0), (1.0f32.to_bits() >> 16) as u16);
        assert_eq!(widen_bf16(to_bf16_bits(1.0)), 1.0);
        // Halfway cases tie to even: 0x3f80_8000 sits between 0x3f80 and 0x3f81.
        assert_eq!(to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
        // Above halfway rounds up, below rounds down.
        assert_eq!(to_bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(to_bf16_bits(f32::from_bits(0x3f80_7fff)), 0x3f80);
        // Round-trip through widen is idempotent.
        for v in [0.5f32, -1.25, 3.0, -0.0, 256.0] {
            assert_eq!(to_bf16_bits(widen_bf16(to_bf16_bits(v))), to_bf16_bits(v));
        }
    }

    // ---- PEER-STRENGTH GUARDS ------------------------------------------------------------------
    //
    // A benchmark peer can be weakened without breaking anything: the suite still runs, the
    // cross-language check still passes (a slow kernel is not a wrong kernel), and the only visible
    // effect is that Wukong's published multiple goes UP. That is the failure mode this file
    // actually suffered — `restrict` absent from all 43 C kernels, column reductions written
    // column-outer, `matmul_tn` written with both operands column-strided, the transpose unblocked —
    // and it survived for as long as it did precisely because nothing tested for it.
    //
    // These tests are the guard. THE CHECKLIST for adding or editing a C/C++/Rust peer:
    //   1. Distinct in/out buffers  -> `__restrict__` on every pointer parameter (guarded below).
    //      If a parameter genuinely may alias, leave it off AND say why at the call site.
    //   2. Loop order  -> the innermost loop must walk memory with unit stride wherever the
    //      algorithm allows it (guarded below for the families that got this wrong).
    //   3. Same algorithmic opportunity as Wukong's kernel: if Wukong dispatches a cache-blocked
    //      kernel, the peer is blocked too (transpose); if Wukong reassociates a float reduction,
    //      a `C(fast)` [-ffast-math] column exists AND is printed.
    //   4. Ask the question the whole exercise turns on: *is this how a competent C programmer
    //      would write it?* If the answer needs a caveat, the caveat belongs in BENCHMARKS.md.
    //   5. The RUST peer gets the same aliasing information: its buffers reach the loops as slice
    //      PARAMETERS of a `kbody` fn (guarded below). `__restrict__` on the C side and raw pointers
    //      on the Rust side is not a language difference, it is a peer defect.

    /// **The Rust half of rule 1.** `__restrict__` on every C peer buys nothing for fairness if the
    /// Rust peer computing the same thing is compiled under may-alias assumptions — and it was.
    /// rustc emits LLVM `noalias` on *reference parameters only*: never on a raw pointer, and
    /// (measured, not assumed) never on a slice built as a LOCAL inside the `extern "C"` entry.
    ///
    /// Evidence, `saxpy` body, rustc 1.94.1 `-Copt-level=3 -Ctarget-cpu=native`, `--emit=asm`:
    ///
    /// | spelling | asm lines | basic blocks | scalar-FP ops |
    /// |---|---|---|---|
    /// | `*out.add(i) = …` raw pointers | 108 | 4 | 32 |
    /// | `from_raw_parts` **inside** `kbench`, loop inside `kbench` | 108 | 4 | 32 |
    /// | slices passed to a `fn` taking `&[f32]`/`&mut [f32]` | **56** | **1** | **0** |
    ///
    /// The first two are byte-identical (a runtime `|out-x| < 128` overlap test plus a scalar
    /// fallback loop); only the third carries `llvm.experimental.noalias.scope.decl` in the IR.
    ///
    /// So the shim is the fix: `kbench` converts the pointers and calls `kbody`, whose parameters
    /// are slices; `kbody` may immediately take `.as_ptr()` back, which keeps every kernel body's
    /// loop text — and therefore its accumulation order — byte-for-byte what it was.
    ///
    /// This test pins the generators that carry the shim today. It is deliberately a **named list**
    /// rather than a textual scan of the whole file: the remaining ~30 Rust peers are still raw
    /// pointers, so a blanket scan could only be satisfied by converting all of them at once, and
    /// the measured effect on their shapes was inside this machine's noise. Add a generator here
    /// when you convert it.
    #[test]
    fn converted_rust_peers_take_their_buffers_as_slice_parameters() {
        let cases: Vec<(&str, String)> = vec![
            (
                "rust_kernel",
                rust_kernel("for i in 0..N { *out.add(i)= *x.add(i); }"),
            ),
            (
                "rust_kernel_n",
                rust_kernel_n(64, "for i in 0..N { *out.add(i)= *x.add(i); }"),
            ),
            ("rust_matmul_tn", rust_matmul_tn(32)),
            ("rust_transpose", rust_transpose(64, 32)),
            ("rust_colsum", rust_colsum(64, 32)),
            ("rust_colmax", rust_colmax(64, 32, 0)),
            ("rust_colarg", rust_colarg(64, 32, true)),
            ("rust_colstat", rust_colstat(64, 32, 3)),
        ];
        for (name, src) in &cases {
            assert!(
                src.contains("fn kbody("),
                "{name}: no `kbody` shim — its buffers never cross a fn boundary as slices, so this \
                 peer is compiled may-alias while its C twin has __restrict__"
            );
            let entry = src
                .find("pub unsafe extern \"C\" fn kbench(")
                .unwrap_or_else(|| panic!("{name}: no `kbench` entry point"));
            let body = &src[entry..];
            let open = body.find(") {").expect("kbench must open a body") + 3;
            assert!(
                body[open..].contains("kbody("),
                "{name}: `kbench` must delegate to `kbody`"
            );
            assert!(
                !body[open..].contains("for "),
                "{name}: `kbench` still runs a loop directly. Slices built as locals grant NO \
                 aliasing information (verified: byte-identical asm to raw pointers) — move the \
                 loop behind `kbody`'s slice parameters."
            );
        }
        // The scratch vector `rust_colarg` needs is a stack array, not a heap allocation inside the
        // timed kernel: `c_colarg`'s twin is `float bv[C]` and pays neither the allocator call nor
        // the zero-fill that `vec![0.0f32; C]` does.
        let arg = rust_colarg(64, 32, true);
        assert!(
            !arg.contains("vec!"),
            "rust_colarg allocates inside the timed kernel; its C twin does not"
        );
    }

    /// EVERY generated C kernel must declare its pointer parameters `__restrict__`. Scanned out of
    /// the crate's own source text rather than from a hand-kept list, so a NEW peer generator added
    /// later is covered automatically — a list would have to be remembered, and the thing being
    /// guarded against is exactly a peer nobody remembered to check.
    ///
    /// **`model.rs` is scanned too (added 2026-08-05).** This test used to read `main.rs` alone, and
    /// the hole was not hypothetical: `c_model` — the peer for the flagship 12-layer end-to-end row —
    /// carried the exact defect the 2026-08-04 audit was about, 24 unqualified pointer parameters plus
    /// two unqualified static helpers, for as long as the guard existed. A guard that covers one file
    /// certifies one file. It also now accepts a parameter list spanning several LINES (`c_model`'s
    /// does) and only requires the qualifier on parameters that are actually pointers, so a `long n`
    /// dimension argument does not have to lie.
    #[test]
    fn every_generated_c_kernel_declares_restrict() {
        // Split so these needles do not occur contiguously in the files they scan — otherwise the
        // test matches its own source line and reports its own Rust code as an unqualified peer.
        let exported = concat!("__declspec(dllexport)", " void ");
        let helper = concat!("static", " void ");
        let mut seen = 0usize;
        let mut model_seen = 0usize;
        for (file, src) in [
            ("main.rs", include_str!("main.rs")),
            ("model.rs", include_str!("model.rs")),
        ] {
            for open in [exported, helper] {
                let mut rest = src;
                while let Some(i) = rest.find(open) {
                    let after = &rest[i + open.len()..];
                    // `<name>(` … `)` — the generated prototypes have no nested parentheses.
                    let Some(lp) = after.find('(') else { break };
                    let end = after[lp..]
                        .find(')')
                        .expect("a generated C prototype must close its parameter list");
                    let name = after[..lp].trim();
                    let params = &after[lp + 1..lp + end];
                    for p in params.split(',') {
                        // Dimension scalars (`long m`) are not pointers and need no qualifier.
                        if !p.contains('*') {
                            continue;
                        }
                        assert!(
                            p.contains("__restrict__"),
                            "{file}: C peer `{name}` parameter `{}` is not __restrict__ (full list: \
                             `{params}`). Without it gcc must assume the output aliases the inputs \
                             and cannot vectorize the kernel — see the peer-strength checklist above.",
                            p.trim()
                        );
                    }
                    seen += 1;
                    if file == "model.rs" {
                        model_seen += 1;
                    }
                    rest = &after[lp + end..];
                }
            }
        }
        // The model peer contributes `kbench`, `kfinal`, `linear_nt` and `layernorm_affine`; a floor
        // here is what makes the model.rs half of this test impossible to satisfy vacuously.
        assert!(
            model_seen >= 4,
            "only {model_seen} C prototypes found in model.rs — did c_model move or lose its exports?"
        );
        // A floor, so that deleting or renaming the peer generators cannot make this pass vacuously
        // with zero matches. 43 signatures at the time of writing.
        assert!(
            seen >= 40,
            "only {seen} C kbench signatures found — did the peer generators move?"
        );
    }

    /// Split an `f(…)` argument list into its TOP-LEVEL arguments.
    ///
    /// `src[open]` must be the `(` that opens the list. Nesting of `()`/`[]`/`{}` is tracked and
    /// string / char literals are skipped, so neither the comma inside `&["-O3", "-shared"]` nor the
    /// one inside `&format!("{}_omp", k.name)` is mistaken for an argument separator, and a `)`
    /// inside a nested call (`&c_rmsnorm_bwd(r, c)`) does not end the list. Returns `None` if the
    /// list never closes. Byte-indexed, which is safe because every delimiter it splits on is ASCII
    /// and the multi-byte text in this file lives inside string literals it skips wholesale.
    fn split_call_args(src: &str, open: usize) -> Option<Vec<&str>> {
        let b = src.as_bytes();
        assert_eq!(b[open], b'(');
        let (mut depth, mut start, mut i) = (0i32, open + 1, open);
        let mut args: Vec<&str> = Vec::new();
        while i < b.len() {
            match b[i] {
                b'"' => {
                    i += 1;
                    while i < b.len() && b[i] != b'"' {
                        i += if b[i] == b'\\' { 2 } else { 1 };
                    }
                }
                // `'x'` / `'\n'` is a char literal; a lone `'` is a lifetime, so just step over it.
                b'\'' => {
                    if i + 2 < b.len() && b[i + 2] == b'\'' {
                        i += 2;
                    } else if i + 3 < b.len() && b[i + 1] == b'\\' && b[i + 3] == b'\'' {
                        i += 3;
                    }
                }
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        args.push(&src[start..i]);
                        return Some(
                            args.into_iter()
                                .map(str::trim)
                                .filter(|a| !a.is_empty())
                                .collect(),
                        );
                    }
                }
                b',' if depth == 1 => {
                    args.push(&src[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// **Every honest-flags C column must come from a `build_c_cpp` pair**, so the C++ column
    /// cannot go missing from a family. This is the guard behind the claim `BENCHMARKS.md` used to
    /// make by assertion — "g++ and gcc share a backend, so the C ratios stand for C++" — which no
    /// benchmark run could falsify because ~37 of the ~40 families measured no C++ at all.
    ///
    /// The rule, scanned out of this file's own text: a `bench_external*` or `build_peer` call whose
    /// first argument is `"c"` is allowed only (a) inside the pair helpers and the relaxed-FP helpers
    /// themselves, or (b) when the peer's name marks it as one of the deliberately C-only *extra*
    /// columns — `…_fast` (the `-ffast-math` reassociation column) and `…_omp` (the OpenMP column).
    /// Those two are normalizations of the C baseline, not a second language, so they have no C++
    /// twin by design. Anything else is a family that has quietly reverted to a C-only comparison.
    ///
    /// `build_peer` is scanned as well as `bench_external*` because the ABBA change split the
    /// compile-and-load step out of the timing step: a family that wanted a bare C column would now
    /// most naturally reach for `build_peer("c", …)`, and a guard that only knew the old entry point
    /// would go quietly blind to exactly the thing it exists to catch. Both take the ext first and
    /// the peer name fourth, so one scan covers them.
    ///
    /// **This guard was VACUOUS until 2026-08-05 and is only here on the second attempt.** The old
    /// scan needle was the CONTIGUOUS string `bench_external` + `("c",`, which requires the `"c"`
    /// literal to sit on the same LINE as the opening paren. That spelling occurs exactly ONCE in
    /// this ~8,700-line file — inside `bench_c_cpp` itself, which is on the exemption list — while
    /// all fifteen other C-ext call sites use the rustfmt-produced multi-line spelling with `"c",`
    /// on the next line. It could not match `bench_external4(` / `_i8(` / `_bf16(` / `_halfout(` at
    /// all, since `bench_external` + `(` is not a substring of `bench_external4(`. The `unpaired`
    /// vector was therefore unconditionally empty and the `_fast`/`_omp` branch had never once
    /// executed. Falsified directly: reverting `bench_matmul_tn`'s `bench_c_cpp` call to a
    /// multi-line C-only `bench_external` plus `let cm_cpp: Option<Measure> = None;` compiled and
    /// the old test still reported `ok`. The scan below is therefore **whitespace-insensitive**, is
    /// **suffix-aware**, and reads the peer name out of the parsed argument list rather than out of
    /// a fixed-width text window (the window could not reach `"rmsnorm_bwd_fast"` either, because
    /// the nested `&c_rmsnorm_bwd(r, c)` closes the window's first `)` before the name).
    ///
    /// Falsify it again after touching it. A guard nobody has falsified is not a guard.
    #[test]
    fn every_c_column_is_paired_with_a_cpp_column() {
        const SRC: &str = include_str!("main.rs");
        // Scan only the NON-TEST half of the file. Every real call site lives there, and the cut
        // makes the needles below unable to match this test's own source text — which is what the
        // old version was contorting its needle to avoid, at the cost of matching almost nothing.
        let marker = concat!("#[cfg", "(test)]");
        let end = SRC
            .find(marker)
            .expect("main.rs must contain a #[cfg(test)] module");
        assert!(
            end > SRC.len() / 2,
            "the {marker} marker was found at byte {end} of {} — the scan region is not the body \
             of the file, and a stray early occurrence would hide every call site after it",
            SRC.len()
        );
        let scan = &SRC[..end];
        // Helpers that legitimately hold a bare C-ext `bench_external*` / `build_peer` call: the
        // PAIR helpers, which add the C++ column themselves, and the relaxed-FP column helpers,
        // which are a C-only extra column by construction.
        //
        // `build_c_cpp` is the build-only half of `bench_c_cpp`, added when six families were
        // converted to build-every-column-then-rotate; it builds exactly the same two columns and
        // belongs in the same list. IT WAS MISSING, and this test has been RED on the branch since
        // that commit — the wave-1 report did not run it and the read-only audit did not either.
        //
        // AN EXEMPTION LIST NOBODY CHECKS IS A HOLE. Adding a name here is how this guard would be
        // silently disabled, so the list is itself asserted below: a pair helper must really build
        // a `"cpp"` column and a relaxed helper must really be a `_fast` one. Neither exemption can
        // be granted to a helper that does not earn it.
        const PAIR_HELPERS: &[&str] = &["build_c_cpp"];
        const FAST_HELPERS: &[&str] = &[];
        let body_of = |name: &str| -> &str {
            let head = format!("\nfn {name}(");
            let at = scan
                .find(&head)
                .unwrap_or_else(|| panic!("{name} is exempted here but does not exist"));
            let rest = &scan[at + 1..];
            let end = rest
                .find("\n}\n")
                .unwrap_or_else(|| panic!("{name} has no closing brace at column 0"));
            &rest[..end]
        };
        for name in PAIR_HELPERS {
            let body = body_of(name).replace([' ', '\n'], "");
            assert!(
                body.contains("(\"cpp\","),
                "{name} is exempted from the C/C++ pairing rule as a PAIR helper, but its body \
                 builds no \"cpp\" column — the exemption is unearned and every caller of it is \
                 printing a C column with no C++ twin"
            );
        }
        for name in FAST_HELPERS {
            assert!(
                name.contains("_fast"),
                "{name} is exempted as a relaxed-FP C-only column but is not a _fast helper"
            );
        }
        let helpers: Vec<&str> = PAIR_HELPERS.iter().chain(FAST_HELPERS).copied().collect();
        // (byte offset, name) of every top-level `fn …` so an occurrence can be attributed.
        let mut fns: Vec<(usize, &str)> = Vec::new();
        let mut at = 0usize;
        for line in scan.split_inclusive('\n') {
            if let Some(r) = line.strip_prefix("fn ") {
                let n = r.split(['(', '<', ' ']).next().unwrap_or("");
                fns.push((at, n));
            }
            at += line.len();
        }
        let owner = |pos: usize| -> &str {
            fns.iter()
                .rev()
                .find(|(o, _)| *o <= pos)
                .map(|(_, n)| *n)
                .unwrap_or("<top level>")
        };
        let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        // Split so the base names are not present contiguously in the text being scanned. (Belt and
        // braces — the `#[cfg(test)]` cut above already excludes this module.)
        let bases = [concat!("bench_", "external"), concat!("build_", "peer")];
        let (mut defs, mut calls, mut c_ext, mut exempt) = (Vec::new(), 0usize, 0usize, Vec::new());
        let mut unpaired: Vec<String> = Vec::new();
        for base in bases {
            let mut pos = 0usize;
            while let Some(i) = scan[pos..].find(base) {
                let abs = pos + i;
                pos = abs + base.len();
                // Not the tail of a longer identifier.
                if abs > 0 && ident(scan.as_bytes()[abs - 1]) {
                    continue;
                }
                // Consume the ABI suffix (`4` / `_i8` / `_bf16` / `_halfout` / anything added
                // later), then any whitespace, and require the `(` of a call.
                let mut j = pos;
                while j < scan.len() && ident(scan.as_bytes()[j]) {
                    j += 1;
                }
                let name = &scan[abs..j];
                let lp = j + scan[j..].len() - scan[j..].trim_start().len();
                if scan.as_bytes().get(lp) != Some(&b'(') {
                    continue; // a doc-comment mention or a bare fn-item reference, not a call
                }
                if scan[..abs].ends_with("fn ") {
                    defs.push(name);
                    continue;
                }
                calls += 1;
                let line = scan[..abs].matches('\n').count() + 1;
                let args = split_call_args(scan, lp)
                    .unwrap_or_else(|| panic!("{name} at line {line}: argument list never closes"));
                if args.first().copied() != Some("\"c\"") {
                    continue; // the C++ / Rust / MKL columns
                }
                c_ext += 1;
                let f = owner(abs);
                if helpers.contains(&f) {
                    continue;
                }
                // The peer name is the 4th argument of BOTH families; `_fast` / `_omp` mark the
                // C-only extra columns.
                let peer = *args.get(3).unwrap_or_else(|| {
                    panic!("{name} at line {line}: fewer than 4 arguments — has the ABI changed?")
                });
                if peer.contains("_fast") || peer.contains("_omp") {
                    exempt.push(format!("  line {line}: {f} -> {peer}"));
                    continue;
                }
                unpaired.push(format!("  line {line}: {f} -> {name}(\"c\", …, {peer}, …)"));
            }
        }
        assert!(
            unpaired.is_empty(),
            "these benches take a C column WITHOUT a C++ twin — use build_c_cpp(…) instead:\n{}",
            unpaired.join("\n")
        );
        // Pin the live definition names, not just their count: removing the obsolete
        // four-pointer compile-and-time wrapper must not hide any other ABI family.
        defs.sort_unstable();
        assert_eq!(
            defs,
            ["build_peer"],
            "peer builder inventory changed; inspect every ABI family"
        );
        // Floors prevent a scan that matches nothing from passing vacuously.
        assert!(
            calls >= 58,
            "only {calls} peer call sites — did the scan stop matching?"
        );
        assert!(
            c_ext >= 14,
            "only {c_ext} C-ext call sites — is the first argument still the ext?"
        );
        assert!(
            exempt.len() >= 6,
            "only {} C-only extra columns recognized, so the `_fast`/`_omp` exemption is close to \
             dead code and the guard is close to vacuous again — found:\n{}",
            exempt.len(),
            exempt.join("\n")
        );
        // …and a floor on the pairs themselves, so deleting every pair call is loud too.
        let pairs = SRC.matches(concat!("bench_c_cpp", "(")).count()
            + SRC.matches(concat!("build_c_cpp", "(")).count()
            + SRC.matches(concat!("bench_c_cpp", "_i8(")).count()
            + SRC.matches(concat!("bench_c_cpp", "_bf16(")).count()
            + SRC.matches(concat!("bench_c_cpp", "_halfout(")).count();
        assert!(
            pairs >= 35,
            "only {pairs} C/C++ pair call sites — did a family lose its peers?"
        );
    }

    /// The C++ render must keep the numeric body byte-for-byte and only unmangle the export, or the
    /// "same code, other front end" claim is false. `__restrict__` in particular must survive: it is
    /// the GCC spelling accepted in both C and C++ mode, and dropping it in the C++ column alone
    /// would make the C++ peer systematically slower for a reason that has nothing to do with g++.
    #[test]
    fn cpp_render_preserves_the_c_body_and_unmangles_the_export() {
        let c = c_colsum(64, 32);
        let cpp = cpp_from_c(&c);
        assert!(cpp.contains(concat!("extern \"C\" ", "__declspec(dllexport)")));
        assert_eq!(
            cpp.matches("__restrict__").count(),
            c.matches("__restrict__").count(),
            "cpp_from_c dropped a __restrict__ qualifier"
        );
        // Everything except the export prefix is untouched.
        assert_eq!(cpp.replace("extern \"C\" ", ""), c);
        // The symbol the harness looks up must still be spelled `kbench`.
        assert!(cpp.contains("void kbench("));
    }

    /// The column-reduction / column-argmax / weight-gradient peers must keep the loop order that
    /// walks `x` sequentially. Each of these was published as a 4-50x Wukong win purely because the
    /// peer traversed a row-major matrix down its columns.
    #[test]
    fn column_family_peers_stay_row_outer() {
        // colsum / colstat: `for (i) for (j) out[j] += ...` — i OUTSIDE j.
        let s = c_colsum(64, 32);
        assert!(
            s.contains("for (long i=0;i<M;i++) for (long j=0;j<N;j++)"),
            "c_colsum regressed to a column-outer fold:\n{s}"
        );
        for op in 0u8..4 {
            let s = c_colstat(64, 32, op);
            assert!(
                s.contains("for (long i=0;i<M;i++) for (long j=0;j<N;j++)"),
                "c_colstat(op={op}) regressed to a column-outer fold:\n{s}"
            );
        }
        // colmax/min/absmax: row 0 seeds `out[]`, then i OUTSIDE j from row 1.
        for op in 0u8..3 {
            let s = c_colmax(64, 32, op);
            assert!(
                s.contains("for (long i=1;i<M;i++) for (long j=0;j<N;j++)"),
                "c_colmax(op={op}) regressed to a column-outer fold:\n{s}"
            );
        }
        // colarg: same, over a C-long running-best vector.
        for is_max in [true, false] {
            let s = c_colarg(64, 32, is_max);
            assert!(
                s.contains("for (long i=1;i<R;i++) for (long j=0;j<C;j++)"),
                "c_colarg(is_max={is_max}) regressed to a column-outer scan:\n{s}"
            );
            assert!(
                s.contains("float bv[C];"),
                "c_colarg lost its running-best vector:\n{s}"
            );
        }
    }

    /// `C = Aᵀ·B` must be the `kij` nest (hoisted `a[k*M+i]`, contiguous B and C), not the `ijk`
    /// nest that reads BOTH operands column-strided.
    #[test]
    fn matmul_tn_peer_stays_kij() {
        let s = c_matmul_tn(64);
        let k = s.find("for (long k=").expect("no k loop in c_matmul_tn");
        let i = s.find("for (long i=").expect("no i loop in c_matmul_tn");
        assert!(
            k < i,
            "c_matmul_tn regressed to an i-outer (ijk) nest:\n{s}"
        );
        assert!(
            s.contains("float aki=a[k*NS+i];"),
            "c_matmul_tn no longer hoists the A element out of the inner loop:\n{s}"
        );
        assert!(
            !s.contains("a[k*NS+i]*b[k*NS+j]"),
            "c_matmul_tn is back to the both-column-strided dot product:\n{s}"
        );
    }

    /// Wukong dispatches a cache-blocked transpose; the peer must be blocked too, or the bench
    /// measures loop tiling rather than codegen. The interior nest must also keep its
    /// **constant-trip-count** tile body (the full-tile fast path), because an always-clamped bound
    /// would slow the peer at the tile-aligned sizes and inflate Wukong there.
    #[test]
    fn transpose_peer_stays_cache_blocked() {
        for &(r, c, _) in TRANSPOSE_SHAPES {
            for (which, s) in [
                ("c_transpose", c_transpose(r, c)),
                ("omp", c_transpose_omp(r, c)),
            ] {
                assert!(
                    s.contains("#define TB 32"),
                    "{which} lost its blocking:\n{s}"
                );
                assert!(
                    s.contains("for (long ii=0;ii<NR;ii+=TB) for (long jj=0;jj<NC;jj+=TB)"),
                    "{which} is back to the naive un-tiled nest:\n{s}"
                );
                assert!(
                    s.contains("if (ii+TB<=NR && jj+TB<=NC)")
                        && s.contains("for (long i=ii;i<ii+TB;i++)"),
                    "{which} lost its constant-trip-count full-tile fast path — an always-clamped \
                     tile bound handicaps the peer at the tile-aligned sizes:\n{s}"
                );
            }
            assert!(c_transpose_omp(r, c).contains("#pragma omp parallel for"));
            // The Rust peer must run the same algorithm or the third column is not comparable.
            let rs = rust_transpose(r, c);
            assert!(
                rs.contains("const TB: usize = 32;") && rs.contains("if ii+TB<=NR && jj+TB<=NC")
            );
        }
    }

    /// **The transpose sweep must measure both stride regimes.** Until 2026-08-05 it ran
    /// `[1024, 2048]` and nothing else — two power-of-two squares, the shapes at which a
    /// column-major write stream aliases in L1 and the blocked C peer is worst. The ~5–7× that
    /// produced was published as the transpose result; at 1000²/1031² the same kernel pair measures
    /// ~1.4×. Sweeping only the favourable sizes is not a benchmark of the compiler, so this pins
    /// that the general regime is present, is the majority of the sweep, includes a non-square
    /// shape, and that the favourable regime is still there to be reported beside it.
    #[test]
    fn transpose_sweep_covers_both_stride_regimes() {
        let alias: Vec<_> = TRANSPOSE_SHAPES.iter().filter(|s| s.2).collect();
        let general: Vec<_> = TRANSPOSE_SHAPES.iter().filter(|s| !s.2).collect();
        assert!(
            !alias.is_empty(),
            "the power-of-two regime must still be reported"
        );
        assert!(
            general.len() >= 3 && general.len() > alias.len(),
            "the general (non-power-of-two) regime must dominate the sweep, got {} general vs {} \
             aliasing — reporting mostly power-of-two shapes measures the size, not the compiler",
            general.len(),
            alias.len()
        );
        // The `aliasing` flag has to mean what it says, or the split is decorative.
        for &&(r, c, aliasing) in &alias {
            assert!(
                r.is_power_of_two() && c.is_power_of_two(),
                "{r}x{c} is flagged aliasing={aliasing} but its strides are not powers of two"
            );
        }
        for &&(r, c, _) in &general {
            assert!(
                !r.is_power_of_two() && !c.is_power_of_two(),
                "{r}x{c} is flagged as the general regime but a stride is a power of two"
            );
        }
        assert!(
            general.iter().any(|s| s.0 != s.1),
            "the general regime needs a non-square shape: a square transpose reads and writes at \
             the SAME stride, so it cannot show a rectangular layout conversion"
        );
        // Every shape must still spill L2 (~2 MiB/core here), or the bench measures cache residency.
        for &&(r, c, _) in TRANSPOSE_SHAPES.iter().collect::<Vec<_>>().iter() {
            assert!(
                2 * r * c * 4 >= 4 << 20,
                "{r}x{c} moves under 4 MiB — too small to be L2-bound"
            );
        }
    }

    /// The headline summary numbers are geometric means of ratios; an arithmetic mean would not be
    /// symmetric under inverting the comparison.
    #[test]
    fn geomean_is_the_geometric_mean() {
        assert!((geomean(&[2.0, 8.0]) - 4.0).abs() < 1e-12);
        assert!((geomean(&[3.0]) - 3.0).abs() < 1e-12);
        // Symmetry: geomean(1/x) == 1/geomean(x), the property that makes it the honest summary.
        let xs = [0.5, 2.0, 4.0, 0.25];
        let inv: Vec<f64> = xs.iter().map(|x| 1.0 / x).collect();
        assert!((geomean(&xs) * geomean(&inv) - 1.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------------------------------
    // The round-interleaved protocol. These are the properties the general suite's defensibility
    // actually rests on, and every one of them is a pure function of its input — so unlike the
    // measurements themselves they can be proved by `cargo test` on any machine in any power state.
    // ------------------------------------------------------------------------------------------

    /// The load-bearing property of [`round_order`]: over each PAIR of rounds every column's two
    /// positions sum to exactly `n-1`. That is what makes the schedule unbiased by construction
    /// rather than on average — no column can be systematically early (cool, uncontended) or
    /// systematically late.
    #[test]
    fn round_order_balances_positions_exactly_over_each_pair() {
        for n in 1..=12usize {
            for pair in 0..5usize {
                let fwd = round_order(n, 2 * pair);
                let rev = round_order(n, 2 * pair + 1);
                let mut pos = vec![0usize; n];
                for (p, &col) in fwd.iter().enumerate() {
                    pos[col] += p;
                }
                for (p, &col) in rev.iter().enumerate() {
                    pos[col] += p;
                }
                assert!(
                    pos.iter().all(|&s| s == n - 1),
                    "n={n} pair={pair}: position sums {pos:?} are not all {}",
                    n - 1
                );
            }
        }
    }

    /// Every round must visit every column exactly once. A schedule that dropped or repeated a
    /// column would leave a NaN in `Rounds::ns` (or double-count one), and the dispersion this
    /// whole design exists to report would be computed over the wrong sample.
    #[test]
    fn round_order_is_a_permutation() {
        for n in 1..=12usize {
            for r in 0..24usize {
                let mut ord = round_order(n, r);
                assert_eq!(ord.len(), n, "n={n} r={r}: wrong length");
                ord.sort_unstable();
                assert!(
                    ord.iter().copied().eq(0..n),
                    "n={n} r={r}: not a permutation ({ord:?})"
                );
            }
        }
    }

    /// At two columns the schedule must degenerate to literally A B / B A — the same A-B-B-A
    /// pairing `abba_min` uses for the peer pairs elsewhere in this file. If these two ever
    /// disagreed, the file would contain two different definitions of "unbiased order".
    #[test]
    fn round_order_at_two_columns_is_abba() {
        assert_eq!(round_order(2, 0), vec![0, 1]);
        assert_eq!(round_order(2, 1), vec![1, 0]);
    }

    /// [`gen_rounds`] must always hand back an EVEN count of at least 2: `round_order`'s balance
    /// holds per pair, so an odd number of rounds reintroduces exactly the positional bias the
    /// rotation removes.
    #[test]
    fn gen_rounds_is_even_and_at_least_two() {
        for (set, want) in [
            (None, 4usize),
            (Some("1"), 2),
            (Some("2"), 2),
            (Some("3"), 4),
            (Some(" 7 "), 8),
            (Some("garbage"), 4),
            (Some(""), 4),
        ] {
            let r = parse_rounds(set);
            assert_eq!(r, want, "XBENCH_ROUNDS={set:?}");
            assert!(
                r >= 2 && r.is_multiple_of(2),
                "XBENCH_ROUNDS={set:?} gave {r}"
            );
        }
        // And the live reader must agree with the pure one under the ambient environment.
        assert_eq!(
            gen_rounds(),
            parse_rounds(std::env::var("XBENCH_ROUNDS").ok().as_deref())
        );
    }

    /// The per-visit budget is the thing ec08db2 cut and this branch restored, so its default is
    /// pinned: 2 warm-ups + best-of-7, the pre-round protocol's budget, NOT 1 + 3.
    ///
    /// It is pinned as a REGRESSION guard, not as a preference. Rounds and per-visit samples trade
    /// off against each other and a future measurement may well move the split — but it must move it
    /// deliberately, with this test's expectations edited in the same commit, rather than by someone
    /// shaving the visit to make a suite finish sooner.
    #[test]
    fn per_visit_budget_defaults_to_two_warmups_and_best_of_seven() {
        assert_eq!(parse_budget(None, 2, 0), 2, "warm-up default");
        assert_eq!(parse_budget(None, 7, 1), 7, "sample default");
        // A zero sample count would leave a visit with no timed call at all, so it must fall back to
        // the default rather than be honoured.
        assert_eq!(parse_budget(Some("0"), 7, 1), 7);
        // ...while zero WARM-UPS is a legitimate (if unwise) experiment and is honoured.
        assert_eq!(parse_budget(Some("0"), 2, 0), 0);
        for bad in ["garbage", "", "-1", "1e9", "100000"] {
            assert_eq!(parse_budget(Some(bad), 7, 1), 7, "XBENCH_SAMPLES={bad:?}");
        }
        assert_eq!(parse_budget(Some(" 9 "), 7, 1), 9);
        // The live readers must agree with the pure one under the ambient environment.
        assert_eq!(
            gen_samples(),
            parse_budget(std::env::var("XBENCH_SAMPLES").ok().as_deref(), 7, 1)
        );
        assert_eq!(
            gen_warmups(),
            parse_budget(std::env::var("XBENCH_WARMUPS").ok().as_deref(), 2, 0)
        );
    }

    /// The clock probe must stay OPT-IN. It is a saturating AVX2-FMA loop that runs immediately
    /// before the first timed column of every round, so leaving it on by default would have every
    /// round's measurement preceded by a burst of self-inflicted heat.
    #[test]
    fn the_clock_probe_is_off_unless_asked_for() {
        // Whatever the ambient environment says, the reader and the probe must agree — a probe that
        // ran while `clock_probe_enabled()` was false would be invisible perturbation.
        let on = clock_probe_enabled();
        assert_eq!(
            on,
            matches!(
                std::env::var("XBENCH_CLOCK_PROBE").ok().as_deref(),
                Some("1") | Some("on") | Some("true")
            )
        );
        if !on {
            assert_eq!(clock_probe_gflops(), 0.0, "the probe ran while disabled");
        }
    }

    #[test]
    fn median_is_the_middle_order_statistic() {
        assert!(median(&[3.0, 1.0, 2.0]) == 2.0);
        assert!(median(&[4.0, 1.0, 3.0, 2.0]) == 2.5);
        assert!(median(&[7.0]) == 7.0);
        assert!(median(&[]).is_nan());
        // The reason it is the median and not the mean: one round that collided with a background
        // task must not move the point estimate.
        assert!(median(&[1.0, 1.0, 1.0, 1.0, 100.0]) == 1.0);
    }

    /// A ratio must be formed round by round, not from the two columns' aggregates. This is the
    /// whole point: under a clock that drifts, per-round ratios stay put while a ratio of
    /// separately-aggregated columns does not.
    #[test]
    fn ratio_is_formed_within_each_round() {
        // Two columns whose TRUE relation is a constant 2x, measured while the machine slows down
        // by 3x across the rounds.
        let drift = [1.0, 1.6, 2.2, 3.0];
        let a: Vec<f64> = drift.iter().map(|d| 100.0 * d).collect();
        let b: Vec<f64> = drift.iter().map(|d| 50.0 * d).collect();
        let s = RatioStat::over_rounds(&a, &b).unwrap();
        assert!((s.med - 2.0).abs() < 1e-12, "median {}", s.med);
        assert!((s.spread() - 1.0).abs() < 1e-12, "spread {}", s.spread());
        assert!(s.conclusive());
    }

    /// A dropped or non-finite round must NOT silently shrink the spread — that would turn a
    /// failed measurement into a confident one, the exact failure mode this type exists to prevent.
    #[test]
    fn ratio_refuses_bad_samples() {
        assert!(RatioStat::over_rounds(&[1.0, 2.0], &[1.0]).is_none());
        assert!(RatioStat::over_rounds(&[], &[]).is_none());
        assert!(RatioStat::over_rounds(&[1.0, f64::NAN], &[1.0, 1.0]).is_none());
        assert!(RatioStat::over_rounds(&[1.0, 0.0], &[1.0, 1.0]).is_none());
        assert!(RatioStat::over_rounds(&[1.0, -1.0], &[1.0, 1.0]).is_none());
        assert!(RatioStat::over_rounds(&[1.0, 1.0], &[1.0, f64::INFINITY]).is_none());
    }

    /// Rounds that disagree by more than [`spread_limit`] must NOT print a number — a reader must
    /// not be able to quote a figure the run did not establish. Both fixtures are built FROM the
    /// active limit so the test states the contract rather than a hard-coded pair of numbers that an
    /// `XBENCH_SPREAD_LIMIT` override would falsify.
    #[test]
    fn wide_spread_never_prints_a_size() {
        let limit = spread_limit();
        // Per-round ratios wandering by strictly more than the limit, well clear of a 1.05x floor so
        // the DIRECTION survives and only the size is withheld.
        let wide = RatioStat::over_rounds(&[100.0, 100.0 * limit * 1.5], &[50.0, 50.0]).unwrap();
        assert!(!wide.conclusive(), "spread {} was accepted", wide.spread());
        let text = fmt_ratio_stat(Some(wide), Some(1.05), true);
        assert!(text.starts_with("FASTER, size ?"), "{text}");
        assert!(!text.contains('x'), "a size survived a wide spread: {text}");
        // ...and one wandering by strictly less must still be quoted, or the gate is vacuous.
        let inside = 1.0 + (limit - 1.0) * 0.5;
        let tight = RatioStat::over_rounds(&[100.0, 100.0 * inside], &[50.0, 50.0]).unwrap();
        assert!(tight.conclusive(), "spread {} was rejected", tight.spread());
        assert!(fmt_ratio_stat(Some(tight), Some(1.05), true).contains("x faster"));
        // The SLOWER direction-only cell inverts its interval the same way the sized one does: the
        // printed pair must read low-to-high in "times slower", not the raw ratios.
        let slow = RatioStat::over_rounds(&[50.0, 50.0 / (limit * 1.5)], &[100.0, 100.0]).unwrap();
        let text = fmt_ratio_stat(Some(slow), Some(1.05), true);
        assert!(text.starts_with("SLOWER, size ?"), "{text}");
        let lo: f64 = text[text.find('[').unwrap() + 1..text.rfind('-').unwrap()]
            .parse()
            .unwrap();
        let hi: f64 = text[text.rfind('-').unwrap() + 1..text.find(']').unwrap()]
            .parse()
            .unwrap();
        assert!(lo <= hi, "{text}: interval is inverted");
        assert!(
            (lo - 2.0).abs() < 0.01,
            "{text}: expected 2.00x slower at the near end"
        );
    }

    /// THE `C(twin)` CONTRACT. A ratio smaller than what a byte-identical control column managed
    /// must print NO number and NO direction, however tightly its rounds agreed.
    ///
    /// This is the failure that motivated the whole control: a spread gate alone accepts a steady
    /// 1.03x, and a steady 1.03x on this machine is placement luck. The fixture is the real one — in
    /// a validation run the control read C 75.20 ms against C(twin) 57.60 ms, so the old estimator
    /// would have published a byte-identical binary as "1.31x faster than C".
    #[test]
    fn an_effect_below_the_control_prints_nothing_however_tight() {
        // Four rounds of a rock-steady 1.03x — spread 1.00, so the dispersion gate passes it.
        let steady = RatioStat::over_rounds(&[103.0; 4], &[100.0; 4]).unwrap();
        assert!(
            steady.conclusive(),
            "the fixture must clear the spread gate"
        );
        let text = fmt_ratio_stat(Some(steady), Some(1.31), true);
        assert!(text.starts_with("BELOW FLOOR"), "{text}");
        assert!(
            !text.contains("faster") && !text.contains("slower"),
            "{text}"
        );
        assert_eq!(verdict(steady, Some(1.31)), Verdict::BelowFloor);
        // The same reading against a run whose control behaved is a real result.
        assert_eq!(verdict(steady, Some(1.01)), Verdict::Sized);
        // And the floor is symmetric: a 1.03x SLOWER reading is refused by the same floor, which a
        // one-sided band taken straight off a control interval that does not straddle 1.00 would not
        // have done.
        let slow = RatioStat::over_rounds(&[100.0; 4], &[103.0; 4]).unwrap();
        assert_eq!(verdict(slow, Some(1.31)), Verdict::BelowFloor);
        // `floor_factor` takes the larger departure in either direction, and never reports < 1.
        let one_sided = RatioStat::over_rounds(&[105.0, 120.0], &[100.0, 100.0]).unwrap();
        assert!((floor_factor(one_sided) - 1.20).abs() < 1e-12);
        let below = RatioStat::over_rounds(&[80.0, 90.0], &[100.0, 100.0]).unwrap();
        assert!((floor_factor(below) - 1.25).abs() < 1e-12);
        assert!(floor_factor(RatioStat::over_rounds(&[1.0], &[1.0]).unwrap()) >= 1.0);
    }

    /// A run with NO control has no floor, and must say so rather than quietly reverting to the
    /// weaker spread-only behaviour without telling anyone.
    #[test]
    fn a_run_without_a_control_declares_its_floor_unknown() {
        let legend = ratio_legend(None);
        assert!(legend.contains("UNKNOWN"), "{legend}");
        assert!(ratio_legend(Some(1.42)).contains("1.42x"));
        // ...and without a floor the small steady effect is no longer refused, which is precisely
        // why the legend has to shout.
        let steady = RatioStat::over_rounds(&[103.0; 4], &[100.0; 4]).unwrap();
        assert_eq!(verdict(steady, None), Verdict::Sized);
    }

    /// A power-state change must REPLACE every number, not annotate it. A warning line can be
    /// scrolled past; `NON-REPORTABLE` in the cell cannot.
    #[test]
    fn power_change_removes_the_number_entirely() {
        let s = RatioStat::over_rounds(&[200.0, 200.0], &[100.0, 100.0]).unwrap();
        assert!(s.conclusive());
        assert_eq!(
            fmt_ratio_stat(Some(s), Some(1.05), true),
            "2.00x faster [2.00-2.00]"
        );
        let bad = fmt_ratio_stat(Some(s), Some(1.05), false);
        assert_eq!(bad, "NON-REPORTABLE");
        assert!(!bad.contains('2'), "a digit survived invalidation: {bad}");
    }

    /// Inverting a ratio must invert AND swap the ends of its interval, or a "slower" row would
    /// advertise a range that does not contain its own point estimate.
    #[test]
    fn slower_direction_inverts_the_interval_correctly() {
        // Per-round ratios both well below 1 ("slower"), disagreeing by strictly less than the limit
        // so the cell renders as a number rather than as a bare direction.
        let f = 1.0 + (spread_limit() - 1.0) * 0.5;
        let s = RatioStat::over_rounds(&[50.0, 50.0 / f], &[100.0, 100.0]).unwrap();
        assert!(s.med < 1.0);
        assert!(s.conclusive());
        let text = fmt_ratio_stat(Some(s), Some(1.05), true);
        assert!(text.contains("slower"), "{text}");
        let lo: f64 = text[text.find('[').unwrap() + 1..text.find('-').unwrap()]
            .parse()
            .unwrap();
        let hi: f64 = text[text.find('-').unwrap() + 1..text.find(']').unwrap()]
            .parse()
            .unwrap();
        assert!(lo <= hi, "{text}: interval is inverted");
        let point: f64 = text[..text.find('x').unwrap()].parse().unwrap();
        assert!(
            lo <= point && point <= hi,
            "{text}: point outside its own range"
        );
    }

    /// `spread_limit` must stay a strict factor > 1: a limit of 1.0 or below would make every
    /// ratio inconclusive, and a typo must fall back to the default rather than silently disable
    /// the check.
    #[test]
    fn spread_limit_rejects_degenerate_overrides() {
        for (set, want) in [
            (None, 1.10f64),
            (Some("1.25"), 1.25),
            (Some("1.0"), 1.10),
            (Some("0.5"), 1.10),
            (Some("-3"), 1.10),
            (Some("nonsense"), 1.10),
        ] {
            let got = parse_spread_limit(set);
            assert!(
                (got - want).abs() < 1e-12,
                "XBENCH_SPREAD_LIMIT={set:?} gave {got}"
            );
            assert!(got > 1.0, "XBENCH_SPREAD_LIMIT={set:?} gave a limit <= 1");
        }
        assert!(
            (spread_limit()
                - parse_spread_limit(std::env::var("XBENCH_SPREAD_LIMIT").ok().as_deref()))
            .abs()
                < 1e-12
        );
    }
    // ------------------------------------------------------------------------------------------
    // P6 instrument-honesty tests (2026-09-13). Every one of these pins a property the harness
    // silently violated until this commit; none of them time anything.
    // ------------------------------------------------------------------------------------------

    /// A `Timed` stand-in so the schedule can be tested without running a kernel.
    #[derive(Clone, Copy, Debug)]
    struct FakeTimed(f64);
    impl Timed for FakeTimed {
        fn ns(&self) -> f64 {
            self.0
        }
    }

    /// `abba_min_columns` must give EVERY column the same two slots, and those slots must sum to
    /// `n-1` for every column — the arithmetic balance `round_order` proves, now enforced at the
    /// level a `bench_*` section actually uses.
    ///
    /// This is the property whose absence made every published Wukong-over-peer ratio optimistic:
    /// Wukong was compiled-and-timed in one call, so it always occupied slot 0, and the harness's
    /// own byte-identical control measured slot 0 worth up to 17.3% on `colsum`.
    #[test]
    fn abba_min_columns_gives_every_column_the_same_two_slots() {
        for n in 1..=12usize {
            let mut visits: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut slot = 0usize;
            let _ = abba_min_columns(n, |i| {
                visits[i].push(slot % n);
                slot += 1;
                Some(FakeTimed(1.0))
            });
            for (col, seen) in visits.iter().enumerate() {
                assert_eq!(
                    seen.len(),
                    2,
                    "n={n}: column {col} was timed {} time(s), not twice — an unequal slot count \
                     biases the minimum toward whichever column got more samples",
                    seen.len()
                );
                assert_eq!(
                    seen[0] + seen[1],
                    n - 1,
                    "n={n}: column {col} held slots {seen:?}, which sum to {} rather than {}",
                    seen[0] + seen[1],
                    n - 1
                );
            }
        }
    }

    /// Each column keeps its OWN fastest sample, never another column's, and a column that fails to
    /// produce a sample stays `None` rather than inheriting a neighbour's.
    #[test]
    fn abba_min_columns_keeps_each_column_its_own_minimum() {
        // Column i reads `10*i + 5` on the forward pass and `10*i + 2` on the reverse one, except
        // column 2, which never produces a sample at all.
        let mut pass = 0usize;
        let mut seen = 0usize;
        let out = abba_min_columns(4, |i| {
            if seen == 4 {
                pass = 1;
            }
            seen += 1;
            if i == 2 {
                return None;
            }
            Some(FakeTimed(
                10.0 * i as f64 + if pass == 0 { 5.0 } else { 2.0 },
            ))
        });
        assert_eq!(out.len(), 4);
        for (i, m) in out.iter().enumerate() {
            if i == 2 {
                assert!(m.is_none(), "an absent column must stay absent, got {m:?}");
                continue;
            }
            let got = m.expect("present column").0;
            assert!(
                (got - (10.0 * i as f64 + 2.0)).abs() < 1e-12,
                "column {i} kept {got}, not its own minimum {}",
                10.0 * i as f64 + 2.0
            );
        }
    }

    /// A `Missing` column must consume its position rather than collapsing the rotation, so an
    /// absent toolchain cannot silently re-bias the columns that DID build.
    #[test]
    fn a_missing_column_still_consumes_its_position() {
        let cols = [
            Built::Missing,
            Built::Missing,
            Built::Missing,
            Built::Missing,
        ];
        let mut order = Vec::new();
        // `impl Trait` in argument position forbids a turbofish, so the element type is pinned by
        // the binding instead.
        let out: Vec<Option<Measure>> = abba_min_columns(cols.len(), |i| {
            order.push(i);
            match &cols[i] {
                Built::Missing => None,
                _ => unreachable!(),
            }
        });
        assert_eq!(order, vec![0, 1, 2, 3, 3, 2, 1, 0]);
        assert!(out.iter().all(|m| m.is_none()));
    }

    #[test]
    fn norm_four_pointer_missing_columns_do_not_touch_buffers() {
        let null = std::ptr::null();
        let columns = [
            built_wukong4(None, null, null, null),
            built_peer4(None, null, null, null),
        ];
        let mut output = [17.0];
        let measured = abba_min_columns(columns.len(), |i| {
            time_built(&columns[i], &mut output, null, null)
        });
        assert!(measured.iter().all(Option::is_none));
        assert_eq!(output, [17.0]);
    }

    /// Cover norm, bf16, table, skinny GEMM, and square GEMM column counts.
    /// Exercise every absent-peer combination without a compiler, clock, or kernel.
    #[test]
    fn position_balanced_column_slots_preserve_results_with_missing_peers() {
        for n in [5usize, 6, 7, 8, 10] {
            for missing in 0..(1usize << n) {
                let columns: Vec<Built> = (0..n)
                    .map(|i| {
                        if missing & (1 << i) != 0 {
                            Built::Missing
                        } else {
                            Built::Direct(Box::new(move |out| {
                                out[0] = i as f32;
                                Some(Measure {
                                    compile: Duration::ZERO,
                                    ns_per_call: (i + 1) as f64,
                                    out: out.to_vec(),
                                })
                            }))
                        }
                    })
                    .collect();
                let mut visits = Vec::new();
                let mut output = [0.0];
                let measured = abba_min_columns(n, |i| {
                    visits.push(i);
                    time_built(&columns[i], &mut output, std::ptr::null(), std::ptr::null())
                });
                assert_eq!(visits, (0..n).chain((0..n).rev()).collect::<Vec<_>>());
                for (i, value) in measured.iter().enumerate() {
                    if missing & (1 << i) != 0 {
                        assert!(value.is_none());
                    } else {
                        let value = value.as_ref().unwrap();
                        assert_eq!(value.out, [i as f32]);
                        assert_eq!(value.ns_per_call, (i + 1) as f64);
                    }
                }
            }
        }
    }

    /// The position-balanced families must not contain a compile-and-time call any more, and
    /// EVERY timed region in them must be inside a rotation.
    ///
    /// SOURCE-TEXT SCAN, in the style of `every_generated_c_kernel_declares_restrict`, because this
    /// is a drift guard and not an algorithm: `bench_wukong` / `bench_c_cpp` / `bench_external` /
    /// `bench_c_fast` all build AND time in one call, so a single one of them reappearing inside a
    /// converted body silently puts one column back in a fixed slot with a compiler spawn in front
    /// of the others — the exact defect the conversion removed, and one no differential or
    /// invariance gate can see.
    ///
    /// EXISTENCE IS NOT ENOUGH. This used to require only that the string `abba_min_columns(`
    /// appear SOMEWHERE in the body, so a family with two shape loops of which one was converted
    /// passed while half its rows stayed biased — and a converted family that later grew a second,
    /// unrotated loop would pass forever. So the scan also pins that `time_built` (the only way a
    /// built column reaches the clock) appears exactly once per rotation and that every one of
    /// those calls is lexically inside the `abba_min_columns(...)` statement, and it forbids the
    /// direct timing entry points that would bypass the rotation altogether.
    #[test]
    fn position_balanced_families_build_before_they_time() {
        let src = include_str!("main.rs");
        let compile_and_time = [
            concat!("bench_", "wukong("),
            concat!("bench_", "c_cpp("),
            concat!("bench_", "external("),
            concat!("bench_", "c_fast("),
            concat!("bench_", "wukong4("),
            concat!("bench_", "c_cpp4("),
            concat!("bench_", "external4("),
            concat!("bench_", "wukong_bf16("),
            concat!("bench_", "c_cpp_bf16("),
            concat!("bench_", "external_bf16("),
            concat!("bench_", "bf16_fast("),
            concat!("bench_", "c_fast_idx("),
            concat!("bench_", "wukong_i8("),
            concat!("bench_", "c_cpp_i8("),
            concat!("bench_", "external_i8("),
            concat!("bench_", "wukong_halfout("),
            concat!("bench_", "c_cpp_halfout("),
            concat!("bench_", "external_halfout("),
        ];
        // Ways to reach the clock that are NOT a rotation. `abba_min(` is the old two-column
        // A-B-B-A helper: it balances C against C++ and leaves every other column in a fixed slot,
        // which is precisely the protocol these families were converted out of.
        let bypasses_the_rotation = [
            concat!("time_", "wukong("),
            concat!("time_", "peer("),
            concat!("time_", "peer4("),
            concat!("time_", "wukong_bf16("),
            concat!("time_", "peer_bf16("),
            concat!("time_", "wukong4("),
            concat!("time_", "wukong_i8("),
            concat!("time_", "peer_i8("),
            concat!("time_", "wukong_halfout("),
            concat!("time_", "peer_halfout("),
            concat!("time_", "ns("),
            concat!("abba_", "min("),
        ];
        let families = [
            "fn bench_matmul_tn(",
            "fn bench_matmul_size(",
            "fn bench_matmul_skinny(",
            "fn bench_linear(",
            "fn bench_softmax_bwd(",
            "fn bench_table_kernel(",
            "fn bench_bf16(",
            "fn bench_gemv(",
            "fn bench_transpose(",
            "fn bench_colsum(",
            "fn bench_biasadd(",
            "fn bench_colstat(",
            "fn bench_norm(",
            "fn bench_norm_batched(",
            "fn bench_rmsnorm_bwd(",
            "fn bench_layernorm_bwd(",
            "fn bench_scaled_gemm(",
            "fn bench_ffn(",
            "fn bench_linear_bf16(",
            "fn bench_dequant(",
            "fn bench_colmax(",
            "fn bench_rowarg(",
            "fn bench_colarg(",
            "fn bench_lrscan(",
            "fn bench_cumprod(",
            "fn bench_cumsum(",
            "fn bench_cumminmax(",
            "fn bench_xent(",
            "fn bench_rope(",
            "fn bench_xent_bwd(",
            "fn bench_rope_bwd(",
            "fn bench_gate_n(",
            "fn bench_row_losses(",
            "fn bench_act_backward_n(",
            "fn bench_i8gemm(",
            "fn bench_axpby_half_out(",
            "fn bench_conv(",
            "fn bench_streaming_large(",
        ];
        // No benchmark family can silently evade this list. Dispatch-only wrappers and
        // direct library timers are the only exemptions, pinned by name.
        let direct_or_dispatch = [
            "bench_matmul",
            "bench_gate",
            "bench_act_backward",
            "bench_mm_mkl",
            "bench_mm_mkl_rect",
            "bench_gemv_mkl",
            "bench_transpose_mkl",
            "bench_mm_mkl_tn",
            "bench_vml",
            "bench_mm_tuned",
        ];
        let production = src.split("#[cfg(test)]\nmod tests").next().unwrap();
        let mut discovered = Vec::new();
        for line in production
            .lines()
            .filter(|line| line.starts_with("fn bench_"))
        {
            let head = &line[..line.find('(').unwrap() + 1];
            let name = &head[3..head.len() - 1];
            if !direct_or_dispatch.contains(&name) {
                discovered.push(head);
            }
        }
        let mut expected = families.to_vec();
        discovered.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            discovered, expected,
            "main.rs benchmark inventory changed; audit the new family's timing path"
        );
        for family in families {
            let start = src
                .find(family)
                .unwrap_or_else(|| panic!("{family} not found in main.rs"));
            // A top-level `fn` body ends at the next line that is exactly `}`.
            let rest = &src[start..];
            let end = rest
                .find("\n}\n")
                .unwrap_or_else(|| panic!("{family} has no closing brace at column 0"));
            let body = &rest[..end];
            let rotations = body.matches("abba_min_columns(").count();
            assert!(
                rotations >= 1,
                "{family} no longer times its columns through abba_min_columns: its Wukong column \
                 is back in a fixed slot"
            );
            for needle in compile_and_time {
                assert!(
                    !body.contains(needle),
                    "{family} calls `{needle}` — that helper compiles AND times in one call, which \
                     forces its column into a fixed slot and puts a compiler spawn between two \
                     timed regions. Use build_wukong/build_c_cpp/build_peer + abba_min_columns."
                );
            }
            for needle in bypasses_the_rotation {
                assert!(
                    !body.contains(needle),
                    "{family} calls `{needle}` — that times a column outside the rotation, so its \
                     slot is fixed. Put the column in the `Built` array and time it through \
                     abba_min_columns."
                );
            }
            // Dequant has two independent shape loops; each must finish its own
            // preparation before entering its rotation, just like every other family.
            let stages: Vec<&str> = if family == "fn bench_dequant(" {
                body.split("// --- per-channel dequant:").collect()
            } else {
                vec![body]
            };
            assert_eq!(
                stages.len(),
                if family == "fn bench_dequant(" { 2 } else { 1 }
            );
            for stage in stages {
                let timing_start = stage
                    .find("abba_min_columns(")
                    .expect("shape loop has no rotation");
                for build in ["build_wukong(", "build_c_cpp(", "build_peer("] {
                    assert!(
                        !stage[timing_start..].contains(build),
                        "{family}: build after timing began in a shape loop"
                    );
                }
            }
            let timer = match family {
                "fn bench_bf16(" | "fn bench_linear_bf16(" => "time_built_bf16(",
                "fn bench_i8gemm(" => "time_built_i8(",
                "fn bench_axpby_half_out(" => "time_built_half_experiment(",
                _ => "time_built(",
            };
            assert_eq!(rotations, if family == "fn bench_dequant(" { 2 } else { 1 });
            let timed = body.matches(timer).count();
            assert_eq!(
                timed, rotations,
                "{family} has {timed} time_built call(s) against {rotations} abba_min_columns \
                 rotation(s): one shape loop was converted and another was not, so half this \
                 family's rows still carry the slot advantage"
            );
            // ...and each of them is inside its rotation's own statement, not merely nearby.
            let mut cursor = 0usize;
            while let Some(rel) = body[cursor..].find(timer) {
                let at = cursor + rel;
                let before = &body[..at];
                let rotation = before.rfind("abba_min_columns(");
                assert!(
                    rotation.is_some_and(|r| !before[r..].contains(';')),
                    "{family}: the `time_built(` at byte {at} of the body is not inside an \
                     abba_min_columns(...) call, so that column is timed in a fixed slot"
                );
                cursor = at + 1;
            }
        }
    }

    #[test]
    fn position_balanced_bf16_missing_slots_and_scalar_gates() {
        let columns = [
            BuiltBf16::Wukong(None),
            BuiltBf16::Peer(None),
            BuiltBf16::Peer(None),
            BuiltBf16::Peer(None),
            BuiltBf16::Peer(None),
        ];
        let mut visits = Vec::new();
        let mut out = [17.0];
        let measured = abba_min_columns(columns.len(), |i| {
            visits.push(i);
            time_built_bf16(&columns[i], &mut out, std::ptr::null(), std::ptr::null())
        });
        assert_eq!(visits, [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]);
        assert!(measured.iter().all(Option::is_none));
        assert_eq!(out, [17.0]);
        let scalar = |value| MeasureBf16 {
            compile: Duration::ZERO,
            ns_per_call: 1.0,
            out: value,
        };
        assert!(cpp_checked_bf16("fixture", &Some(scalar(1.0)), Some(scalar(1.0005))).is_some());
        assert!(cpp_checked_bf16("fixture", &Some(scalar(1.0)), Some(scalar(1.002))).is_none());
        assert!(bf16_fast_checked("fixture", &scalar(1.005), scalar(1.0)).is_some());
        assert!(bf16_fast_checked("fixture", &scalar(1.02), scalar(1.0)).is_none());
    }

    #[test]
    fn position_balanced_typed_missing_columns_preserve_both_outputs() {
        let null = std::ptr::null();
        let mut ints = [37i32; 4];
        let int_columns = [
            BuiltI8::Wukong(None),
            BuiltI8::Wukong(None),
            BuiltI8::Peer(None),
            BuiltI8::Peer(None),
            BuiltI8::Peer(None),
        ];
        let samples = abba_min_columns(int_columns.len(), |i| {
            time_built_i8(&int_columns[i], &mut ints, null, std::ptr::null())
        });
        assert!(samples.iter().all(Option::is_none));
        assert_eq!(ints, [37; 4]);
        let columns = [
            BuiltHalfExperiment::HalfWukong(None),
            BuiltHalfExperiment::FloatWukong(None),
            BuiltHalfExperiment::HalfPeer(None),
            BuiltHalfExperiment::HalfPeer(None),
            BuiltHalfExperiment::HalfPeer(None),
        ];
        let mut half = [0x3f80u16; 4];
        let mut float = [19.0f32; 4];
        let mut visits = Vec::new();
        let samples = abba_min_columns(columns.len(), |i| {
            visits.push(i);
            time_built_half_experiment(
                &columns[i],
                &mut half,
                &mut float,
                std::ptr::null(),
                std::ptr::null(),
            )
        });
        assert_eq!(visits, [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]);
        assert!(samples.iter().all(Option::is_none));
        assert_eq!(half, [0x3f80; 4]);
        assert_eq!(float, [19.0; 4]);
    }

    #[test]
    fn position_balanced_mixed_measures_keep_variant_and_winning_snapshot() {
        let mut calls = [0usize; 5];
        let mut samples = abba_min_columns(5, |i| {
            calls[i] += 1;
            let ns = if calls[i] == 1 { 9.0 } else { 3.0 };
            let compile = Duration::from_millis(i as u64);
            if i == 1 {
                Some(HalfExperimentMeasure::Float(MeasureBf16 {
                    compile,
                    ns_per_call: ns,
                    out: calls[i] as f32,
                }))
            } else {
                Some(HalfExperimentMeasure::Half(MeasureHalfOut {
                    compile,
                    ns_per_call: ns,
                    out: vec![i as f32, calls[i] as f32],
                }))
            }
        })
        .into_iter();
        for i in 0..5 {
            let measure = samples.next().flatten().unwrap();
            assert_eq!(measure.ns(), 3.0);
            if i == 1 {
                let m = measure.float();
                assert_eq!(m.out, 2.0);
                assert_eq!(m.compile, Duration::from_millis(1));
            } else {
                let m = measure.half();
                assert_eq!(m.out, vec![i as f32, 2.0]);
                assert_eq!(m.compile, Duration::from_millis(i as u64));
            }
        }
        assert_eq!(calls, [2; 5]);
    }

    #[test]
    fn position_balanced_extracted_index_integer_and_half_gates() {
        let index = |v| Measure {
            compile: Duration::ZERO,
            ns_per_call: 1.0,
            out: vec![f32::from_bits(v)],
        };
        assert!(fast_indices_checked("fixture", &index(3), index(3)).is_some());
        assert!(fast_indices_checked("fixture", &index(3), index(4)).is_none());
        let ints = |v| MeasureI8 {
            compile: Duration::ZERO,
            ns_per_call: 1.0,
            out: vec![v],
        };
        assert!(cpp_checked_i8("fixture", &Some(ints(7)), Some(ints(7))).is_some());
        assert!(cpp_checked_i8("fixture", &Some(ints(7)), Some(ints(8))).is_none());
        let half = |v| MeasureHalfOut {
            compile: Duration::ZERO,
            ns_per_call: 1.0,
            out: vec![v],
        };
        assert!(cpp_checked_halfout("fixture", &Some(half(1.0)), Some(half(1.0005))).is_some());
        assert!(cpp_checked_halfout("fixture", &Some(half(1.0)), Some(half(1.002))).is_none());
    }

    #[test]
    fn position_balanced_table_caller_uses_rotated_vml() {
        let src = include_str!("main.rs");
        let start = src.find("fn main(").unwrap();
        let body = &src[start..start + src[start..].find("\n}\n").unwrap()];
        assert!(body.contains("bench_table_kernel(k,"));
        assert!(!body.contains(concat!("bench_", "vml(")));
        assert!(!body.contains(concat!("bench_", "wukong(")));
        assert!(body.contains("(vml, &wukong)"));
    }

    /// An in-process library column (oneMKL) must go through the SAME rotation as a JIT'd or
    /// DLL-loaded one. It is the column most easily left out: it needs no build step, so the
    /// obvious place to put it is at the end of the section, in the worst slot - which biases the
    /// row in Wukong's favour, the exact direction this workstream exists to remove.
    #[test]
    fn a_direct_library_column_is_timed_in_the_rotation_like_any_other() {
        // Returns a falling time on each visit, so "kept its own minimum" is checkable.
        let visits = std::cell::Cell::new(0u32);
        let cols = [
            Built::Direct(Box::new(|out: &mut [f32]| {
                visits.set(visits.get() + 1);
                out[0] = visits.get() as f32;
                Some(Measure {
                    compile: Duration::ZERO,
                    ns_per_call: 100.0 - visits.get() as f64,
                    out: out.to_vec(),
                })
            })),
            Built::Missing,
        ];
        let mut buf = vec![0.0f32; 1];
        let timed = abba_min_columns(cols.len(), |i| {
            // SAFETY-adjacent: a Direct column never reads the two input pointers, and this test
            // exists to pin that. `time_built` hands them straight to the closure's ABI-free path.
            time_built(&cols[i], &mut buf, std::ptr::null(), std::ptr::null())
        });
        assert_eq!(
            visits.get(),
            2,
            "the library column must get exactly two slots"
        );
        let m = timed[0]
            .as_ref()
            .expect("a Direct column produces a measurement");
        assert!(
            (m.ns_per_call - 98.0).abs() < 1e-12,
            "kept {} rather than its own minimum 98.0",
            m.ns_per_call
        );
        assert!(timed[1].is_none());
    }

    /// The dispatch census must actually find the recognized kernels in printed MIR, and must not
    /// invent one where there is none. Recognized kernels are gate-blind, so this line is the only
    /// evidence a measured row carries about WHAT it timed.
    ///
    /// A NAME THAT IS NOT A CALL TARGET MUST NOT BE COUNTED. `wukong_mir::print` emits
    /// `func_addr <name>` for a taken function address and `fn <name>(...)` for every definition;
    /// the census used to scan for the bare substring `wukong_`, so any of those would have been
    /// counted as a dispatch and the attribution line on a measured row would have overstated what
    /// the program called.
    #[test]
    fn kernel_census_counts_recognized_calls() {
        let mir = "v3 = call wukong_colsum_f32(v0, v1)\n\
                   v4 = call wukong_colsum_f32(v0, v2)\n\
                   v5 = call wukong_parallel_for(v4)\n";
        assert_eq!(
            kernel_census(mir),
            vec![
                ("wukong_colsum_f32".to_string(), 2),
                ("wukong_parallel_for".to_string(), 1)
            ]
        );
        assert!(kernel_census("v0 = fadd v1, v2\n").is_empty());
        // Every non-call shape the MIR printer can put a `wukong_` name in.
        assert!(
            kernel_census(
                "fn wukong_colsum_f32(v0 ptr) -> void {\n    \
                 v1 = func_addr wukong_colsum_f32\n    \
                 v2 = global_addr wukong_colsum_f32\n}\n"
            )
            .is_empty(),
            "a definition, a func_addr and a global_addr are not dispatches"
        );
    }

    /// Compile a `.wk` source the way [`build_wukong`] does, up to the printed MIR, with no JIT.
    fn printed_mir_for(src: &str) -> String {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema: {sd:?}");
        let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!ld.iter().any(|d| d.is_error()), "lower: {ld:?}");
        wukong_opt::optimize(&mut program, 3);
        wukong_mir::print::print_program(&program, &interner)
    }

    /// The census against REAL printer output, not a string the test wrote itself.
    ///
    /// The unit case above validates the scanner in isolation, which is exactly the blindness the
    /// audit named: a census that disagrees with `wukong_mir::print::print_program` — by counting a
    /// name the printer emits somewhere other than a call — still passes it. This drives the same
    /// pipeline `build_wukong` drives, on a generated source the harness actually times, and pins
    /// the two invariants that matter: every name reported is a call target in that text, and the
    /// census total is exactly the number of `call wukong_` sites, never more.
    #[test]
    fn kernel_census_agrees_with_real_printed_mir() {
        let mir = printed_mir_for(&wk_colsum(256, 256, false));
        let census = kernel_census(&mir);
        assert!(
            !census.is_empty(),
            "colsum lowered to no runtime-kernel call at all, so the dispatch line a measured \
             colsum row carries would be empty. MIR:\n{mir}"
        );
        let mut total = 0;
        for (name, n) in &census {
            assert_eq!(
                mir.matches(&format!("call {name}(")).count(),
                *n,
                "census counted {name} {n} time(s), which is not its call-site count"
            );
            total += *n;
        }
        assert_eq!(
            total,
            mir.matches("call wukong_").count(),
            "the census total must be exactly the `call wukong_` sites in the printed MIR"
        );
    }

    /// A generated `.wk` source must always land in the same file, and two different programs must
    /// not collide onto one — otherwise the persisted source does not identify the timed row.
    #[test]
    fn wk_source_names_are_stable_and_distinct() {
        let a = wk_colsum(1024, 1024, false);
        let b = wk_colsum(1024, 1024, true);
        assert_eq!(wk_source_name(&a), wk_source_name(&a));
        assert_ne!(wk_source_name(&a), wk_source_name(&b));
        assert!(wk_source_name(&a).starts_with("wk_") && wk_source_name(&a).ends_with(".wk"));
        // STABLE ACROSS RUNS *AND* ACROSS REVISIONS, which determinism and distinctness alone do
        // not pin: any hash function whatever satisfies those two, so the audit was right that
        // there was no realistic mutation of FNV-1a that could fail this test. The name is the
        // only link between a printed dispatch line and the `.wk` file it names, so a re-spelled
        // hash silently orphans every source persisted by an earlier run. `af63dc4c8601ec8c` for
        // "a" is FNV's OWN published 64-bit test vector, so this checks the implementation
        // against the reference rather than against itself.
        assert_eq!(wk_source_name("a"), "wk_af63dc4c8601ec8c.wk");
        assert_eq!(wk_source_name(""), "wk_cbf29ce484222325.wk");
        assert_eq!(wk_source_name("module bench\n"), "wk_f23e80a4f7082feb.wk");
    }

    /// The gate a oneMKL transpose column has to survive before it is printed beside a ratio.
    ///
    /// The ABI test below proves the binding computes the right values; this proves the SECTION
    /// path's post-check accepts them and rejects anything else. Its doc claims BIT equality and
    /// cites `cpp_checked` as its model, and it compared `Vec<f32>` by VALUE - so `-0.0` for `+0.0`
    /// passed as identical and two identical NaN payloads were REJECTED. A transpose is a
    /// permutation: neither can happen unless something is wrong, which is exactly when the check
    /// has to hold. Needs no MKL, so it runs everywhere.
    #[test]
    fn a_library_transpose_column_is_held_to_bit_equality() {
        let m = |v: Vec<f32>| Measure {
            compile: Duration::ZERO,
            ns_per_call: 1.0,
            out: v,
        };
        let wuk = Some(m(vec![0.0, 1.0]));
        assert!(
            mkl_transpose_exact(&wuk, Some(m(vec![0.0, 1.0])), "t").is_some(),
            "identical bits must keep the column"
        );
        assert!(
            mkl_transpose_exact(&wuk, Some(m(vec![-0.0, 1.0])), "t").is_none(),
            "-0.0 for +0.0 is equal by VALUE and different by BITS; a permutation cannot flip a \
             sign bit, so the column must be dropped"
        );
        let nan = Some(m(vec![f32::from_bits(0x7fc0_0001), 1.0]));
        assert!(
            mkl_transpose_exact(&nan, Some(m(vec![f32::from_bits(0x7fc0_0001), 1.0])), "t")
                .is_some(),
            "two identical NaN payloads are the SAME bits; a value comparison would have dropped \
             this column"
        );
        assert!(
            mkl_transpose_exact(&nan, Some(m(vec![f32::from_bits(0x7fc0_0002), 1.0])), "t")
                .is_none(),
            "two different NaN payloads are a disagreement; a value comparison could not see it"
        );
        assert!(
            mkl_transpose_exact(&wuk, Some(m(vec![0.0])), "t").is_none(),
            "a length mismatch is a disagreement, not a panic"
        );
        assert!(
            mkl_transpose_exact(&None, Some(m(vec![9.0])), "t").is_some(),
            "with no Wukong column there is nothing to disagree with"
        );
        assert!(mkl_transpose_exact(&wuk, None, "t").is_none());
    }

    /// The oneMKL bindings added for the previously unpeered families must compute the right thing.
    ///
    /// This is an ABI test, not a timing test: mkl_rt on this host defaults to ILP64, so a
    /// mis-sized dimension is a segfault or garbage rather than a slow number. `mkl_somatcopy` is
    /// additionally NOT interface-layer dependent (oneMKL declares the ?omatcopy family with
    /// `size_t`), which is why it is bound without a `_64` suffix — the installed mkl_rt exports no
    /// `mkl_somatcopy_64` at all. Skips cleanly when MKL is absent, exactly as the columns do.
    #[test]
    fn mkl_library_peers_compute_the_right_thing() {
        let Some(api) = mkl() else {
            eprintln!("oneMKL not present on this host — skipping the ABI check");
            return;
        };

        if api.somatcopy.is_some() {
            // 2x3 -> 3x2, then a non-square 3x4 so a rows/cols mix-up cannot pass. Every value is
            // exact in f32, so the bar is bit equality.
            let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let mut dst = vec![0.0f32; 6];
            let m = bench_transpose_mkl(2, 3, 1, &src, &mut dst).expect("MKL_Somatcopy resolved");
            assert_eq!(m.out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            let src: Vec<f32> = (0..12).map(|i| i as f32).collect();
            let mut dst = vec![0.0f32; 12];
            let m = bench_transpose_mkl(3, 4, 1, &src, &mut dst).expect("MKL_Somatcopy resolved");
            assert_eq!(
                m.out,
                vec![0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
            );
        } else {
            eprintln!("MKL_Somatcopy absent — transpose MKL column will read n/a");
        }

        if api.sgemv.is_some() {
            // y = A x, A = [[1,2],[3,4]], x = [1,1]  =>  y = [3, 7]
            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let x: Vec<f32> = vec![1.0, 1.0];
            let mut y = vec![0.0f32; 2];
            let m = bench_gemv_mkl(2, 2, 1, &a, &x, &mut y).expect("sgemv resolved");
            assert_eq!(m.out, vec![3.0, 7.0]);
        } else {
            eprintln!("cblas_sgemv_64 absent — gemv MKL column will read n/a");
        }

        // C = A^T . B with A stored [K, M] and B stored [K, N], K = M = N = 2.
        // A = [[1,2],[3,4]] as [K][M]  => A^T = [[1,3],[2,4]]
        // B = [[5,6],[7,8]] as [K][N]
        // A^T . B = [[1*5+3*7, 1*6+3*8],[2*5+4*7, 2*6+4*8]] = [[26,30],[38,44]]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        let m = bench_mm_mkl_tn(2, 2, 2, 1, &a, &b, &mut c).expect("cblas_sgemm_64 resolved");
        assert_eq!(m.out, vec![26.0, 30.0, 38.0, 44.0]);
    }
}
