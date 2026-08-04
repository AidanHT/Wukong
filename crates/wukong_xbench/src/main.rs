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

mod general;
mod model;

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
const C_FAST_FLAGS: &[&str] = &["-O3", "-march=native", "-ffast-math", "-shared"];

/// LOOSE cross-check for the relaxed-FP peers (C(fast) / C(omp)). `-ffast-math` and OpenMP-partitioned
/// reductions legitimately reassociate, so their results differ from Wukong's beyond the tight 1e-3
/// bar of the honest-flags columns — the bar here is a magnitude-normalized 1e-2. Returns `false`
/// (and prints the mismatch) when the peer's output is unusable; the caller then drops the column
/// rather than failing the bench.
fn relaxed_peer_ok(label: &str, peer: &str, wuk: &Measure, p: &Measure) -> bool {
    let maxabs = wuk.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
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

/// Compile + run the C(fast) peer for a 3-pointer bench (the same C source, [`C_FAST_FLAGS`]),
/// loose-check it against Wukong, and return it only when usable. Compiled lazily — only the
/// reduction-bearing benches call this, so the extra gcc invocation is paid only where the column
/// exists.
#[allow(clippy::too_many_arguments)]
fn bench_c_fast(
    label: &str,
    c_src: &str,
    dir: &Path,
    cc: &str,
    wuk: &Option<Measure>,
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
) -> Option<Measure> {
    let m = wuk.as_ref()?;
    let cf = bench_external(
        "c",
        c_src,
        dir,
        &format!("{label}_fast"),
        cc,
        C_FAST_FLAGS,
        out,
        xp,
        yp,
    )?;
    relaxed_peer_ok(label, "C(fast)", m, &cf).then_some(cf)
}

/// The C(fast) peer for an **index-output** bench (`rowarg` / `colarg`). Same C source at
/// [`C_FAST_FLAGS`]. gcc will only vectorize a float max-with-index reduction under relaxed FP (the
/// comparison has to be reassociable), so withholding `-ffast-math` here compares Wukong's
/// branchless 8-lane (value,index) fold against a compiler that is *forbidden* from doing the same
/// thing — exactly the asymmetry `is_reduction_kernel` already normalizes for the elementwise
/// `argmax` row. Unlike [`bench_c_fast`] the validity check is EXACT index equality: the output is
/// an i32 buffer riding the f32 slots, so a magnitude-normalized tolerance would be meaningless.
#[allow(clippy::too_many_arguments)]
fn bench_c_fast_idx(
    label: &str,
    c_src: &str,
    dir: &Path,
    cc: &str,
    wuk: &Option<Measure>,
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
) -> Option<Measure> {
    let m = wuk.as_ref()?;
    let cf = bench_external(
        "c",
        c_src,
        dir,
        &format!("{label}_fast"),
        cc,
        C_FAST_FLAGS,
        out,
        xp,
        yp,
    )?;
    if m.out != cf.out {
        println!(
            "  ! {label}: C(fast) picks different indices under -ffast-math — C(fast) column skipped"
        );
        return None;
    }
    Some(cf)
}

/// The C(fast) peer for the **bf16** benches (`linear_bf16`, the bf16 dot/sum), whose ABI writes an
/// f32 scalar rather than a buffer so [`bench_c_fast`] does not fit. Same C source at
/// [`C_FAST_FLAGS`]; the validity bar is the loose 1e-2 relative check the plain column already uses
/// on that scalar. Without this column the bf16 GEMM row compared Wukong's blocked, reassociated
/// f32 accumulation against a C peer required to keep the K-long sum strictly in order.
#[allow(clippy::too_many_arguments)]
fn bench_bf16_fast(
    label: &str,
    c_src: &str,
    dir: &Path,
    cc: &str,
    wuk: &Option<MeasureBf16>,
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureBf16> {
    let m = wuk.as_ref()?;
    let cf = bench_external_bf16(
        "c",
        c_src,
        dir,
        &format!("{label}_fast"),
        cc,
        C_FAST_FLAGS,
        out,
        xp,
        yp,
    )?;
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
        "dot" | "ssd"
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
                let f: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                    lib.get(b"kprobe\0").ok()?;
                Some(f())
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

fn main() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "g++".to_string());
    let dir = std::env::temp_dir().join("wukong_xbench");
    let _ = std::fs::create_dir_all(&dir);

    println!(
        "Cross-language kernel benchmark — Wukong (native) vs C (gcc -O3) vs Rust (rustc -Copt-level=3)"
    );
    println!(
        "N = {N} f32 elements, single-threaded except the `@parallel` rows (labelled), \
         -march=native. Lower ns is better."
    );
    // Power state up front: sustained-load timings on battery are not comparable to AC runs.
    println!("{}\n", model::power_status_line());

    // Optional substring filter (first CLI arg): run only the kernels/sections whose name contains it,
    // for fast single-kernel iteration. No arg → the full suite, byte-for-byte as before.
    let filter = std::env::args().nth(1);
    let want = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

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

        let wukong = bench_wukong(&k.wuk, &mut out, xp, yp);
        let c = bench_external(
            "c",
            &k.c,
            &dir,
            k.name,
            &cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut out,
            xp,
            yp,
        );
        let cpp = bench_external(
            "cpp",
            &cpp_from_c(&k.c),
            &dir,
            k.name,
            &cxx,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut out,
            xp,
            yp,
        );
        let rust = bench_external(
            "rs",
            &k.rust,
            &dir,
            k.name,
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
        );
        // The relaxed-FP C(fast) peer (same C source, -ffast-math) runs only for the
        // reduction-bearing rows — the ones whose honest-flags C column is IEEE-serial and therefore
        // inflated (see the fairness notes). Cross-checked at the loose 1e-2 bar inside.
        let cfast = if is_reduction_kernel(k.name) {
            bench_c_fast(k.name, &k.c, &dir, &cc, &wukong, &mut out, xp, yp)
        } else {
            None
        };
        // The multithreaded C(omp) peer for the @parallel rows: the same kernel under
        // `#pragma omp parallel for` (probed once; skipped with a note when this gcc/loader can't
        // run OpenMP DLLs). Reduction rows also get -ffast-math — an OpenMP reduction clause
        // already reassociates, so the label stays C(omp). Loose-checked like C(fast).
        let comp = c_omp_source(k.name).and_then(|src| {
            omp_threads(&cc, &dir)?;
            let flags = if is_reduction_kernel(k.name) {
                C_OMP_FAST_FLAGS
            } else {
                C_OMP_FLAGS
            };
            let cm = bench_external(
                "c",
                &src,
                &dir,
                &format!("{}_omp", k.name),
                &cc,
                flags,
                &mut out,
                xp,
                yp,
            )?;
            wukong
                .as_ref()
                .and_then(|m| relaxed_peer_ok(k.name, "C(omp)", m, &cm).then_some(cm))
        });

        println!("=== {} ({}) ===", k.name, k.note);
        report(k, &wukong, &c, &cpp, &rust, &cfast, &comp);
        // oneMKL VML peer for the transcendentals MKL ships a vector op for: the elementwise analogue
        // of the GEMM-vs-cblas comparison. Wukong's hand-AVX2 vmath vs Intel's hand-tuned VML, both
        // single-thread, same buffer. Cross-checked against Wukong's output (a large rel error would
        // expose a VML ABI mismatch — then the column is dishonest, so it is dropped with a note).
        let vml_fn = match k.name {
            "exp" => mkl().and_then(|a| a.vs_exp),
            "log" => mkl().and_then(|a| a.vs_ln),
            "tanh" => mkl().and_then(|a| a.vs_tanh),
            _ => None,
        };
        if let (Some(f), Some(m)) = (vml_fn, &wukong) {
            let v = bench_vml(f, xp, &mut out);
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
            }
        }
        let is_par = k.name.contains("@parallel");
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
        bench_matmul_skinny(roof);
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
        bench_dequant(&cc, &cxx, &dir);
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
        general::bench_general(&cc, &cxx, &dir);
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
/// gap to MKL(all). Library peers only (oneMKL 1c/all — the bar at these shapes; the naive-C
/// columns of the square section add nothing here). Thermal order follows the standing law:
/// single-core group first (Wuk(1c), MKL(1c) adjacent), then MKL(all), Wuk(par) LAST — residual
/// heat lands on Wukong, never the peer.
fn bench_matmul_skinny(roof: f64) {
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
        let wuk = bench_wukong(&wk_linear_rect(m, k, n, false), &mut c, ap, bp);
        let mkl_1c = bench_mm_mkl_rect(m, k, n, true, 1, &a, &b, &mut c);
        let mkl_all = mkl()
            .map(|api| api.max_threads)
            .and_then(|t| bench_mm_mkl_rect(m, k, n, true, t, &a, &b, &mut c));
        let wk_par = bench_wukong(&wk_linear_rect(m, k, n, true), &mut c, ap, bp);
        println!(
            "  GFLOP/s     Wuk(1c) {:>7}   Wuk(par) {:>7}   MKL(1c) {:>7}   MKL(all) {:>7}",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&mkl_1c),
            gflops(&mkl_all),
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
        report_gemm_vs_mkl(&wuk, &wk_par, &mkl_1c, &mkl_all, flops);
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

    // Measurement ordering is thermal hygiene (the honesty law). On this hybrid laptop a multi-second
    // all-core run heat-throttles the chip for ~seconds after it, so WHO RUNS BEFORE WHOM decides
    // whether a peer ratio is fair. Two groups, coolest-first:
    //  (1) the single-core group — Wuk(1c), MKL(1c), tuned, then the naive C/Rust/C(fast) nests —
    //      each internally warmed and single-threaded (one core barely heats the package even at
    //      2048³), so every single-core peer is measured in the same near-cold state. Wuk(1c) and
    //      MKL(1c) stay ADJACENT at the head (the documented MKL(1c) protection: the old order once
    //      ran MKL(1c) *last*, after the all-core burst + naive nests, throttling it to a bogus
    //      32 GFLOP/s at 4096³). The naive nests were previously measured dead-LAST — *after* the
    //      all-core Wuk(par)/MKL(all) bursts — so they could run heat-throttled, which understated
    //      them and overstated Wukong's ratio (the fairness-audit finding). They now close out the
    //      single-core group: measured in the same thermal group as Wuk(1c), before any all-core
    //      burst, and after the library peers so their long scalar runs pollute no library number.
    //      They are still skipped at ≥2048³ where a single call is tens of seconds (their loss is
    //      already overwhelming and widening at ≤1024³; set XBENCH_NAIVE_HUGE to force them) — there
    //      correctness is cross-checked against MKL instead of C.
    //  (2) the all-core group — MKL(all), then C(omp), then Wuk(par) LAST. Wuk(1c)/tuned/naive use
    //      serial kernels that never touch rayon, so rayon's global pool is still DORMANT when
    //      MKL(all) runs: otherwise-idle cores, no rival thread pool, the coolest available all-core
    //      state (its documented protection — the only heat before it is single-core heat). Wuk(par)
    //      runs after every peer, inheriting whatever residual heat exists, so the Wuk/MKL and
    //      Wuk/C(omp) all-core ratios are CONSERVATIVE lower bounds on Wukong — we throttle
    //      ourselves, never the competitor, the honest direction when two all-core runs cannot both
    //      be cool. (An earlier interleaved A/B timer was reproducibility-fragile: alternating two
    //      live thread pools thrashes the OS scheduler and MKL's OpenMP workers park between blocks,
    //      reading a bogus sub-1-thread number.)
    let wuk = bench_wukong(&wk_matmul(ns, false), &mut c, ap, bp);
    let mkl_1c = bench_mm_mkl(ns, false, 1, &a, &b, &mut c);
    let tuned = bench_mm_tuned(ns, false, &a, &b, &mut c);
    let run_naive = ns < 2048 || std::env::var("XBENCH_NAIVE_HUGE").is_ok();
    // Naive single-core peers — the tail of the single-core group (see the ordering comment).
    let (cm, rm, cfast) = if run_naive {
        let cm = bench_external(
            "c",
            &c_matmul(ns),
            dir,
            "matmul",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut c,
            ap,
            bp,
        );
        let rm = bench_external(
            "rs",
            &rust_matmul(ns),
            dir,
            "matmul",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
        );
        // The reassociation-normalized C peer: the same naive source at -ffast-math (gcc may
        // reassociate/vectorize more aggressively). Loose-checked vs Wukong inside; the column is
        // skipped (None) when the fast output can't meet the 1e-2 bar.
        let cfast = bench_c_fast("matmul", &c_matmul(ns), dir, cc, &wuk, &mut c, ap, bp);
        (cm, rm, cfast)
    } else {
        (None, None, None)
    };
    // The all-core group: MKL(all) first (coolest), then C(omp), then Wuk(par) last.
    let mkl_all = mkl()
        .map(|api| api.max_threads)
        .and_then(|t| bench_mm_mkl(ns, false, t, &a, &b, &mut c));
    // All-core C(omp) peer (the naive ikj source + `#pragma omp parallel for` on the row loop):
    // measured inside the all-core group and BEFORE Wuk(par), so any residual heat lands on
    // Wukong, never the peer. Gated like the other naive-source peers (multi-second at ≥2048³).
    let comp = if run_naive {
        omp_threads(cc, dir)
            .and_then(|_| {
                bench_external(
                    "c",
                    &c_matmul_omp(ns),
                    dir,
                    "matmul_omp",
                    cc,
                    C_OMP_FLAGS,
                    &mut c,
                    ap,
                    bp,
                )
            })
            .filter(|p| wuk.as_ref().is_some_and(|m| relaxed_peer_ok("matmul", "C(omp)", m, p)))
    } else {
        None
    };
    let wk_par = bench_wukong(&wk_matmul(ns, true), &mut c, ap, bp);

    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "Wuk(1c)", "Wuk(par)", "MKL(1c)", "MKL(all)", "tuned(mm)", "C(gcc)", "C(fast)", "C(omp)", "Rust"
    );
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "GFLOP/s",
        gflops(&wuk),
        gflops(&wk_par),
        gflops(&mkl_1c),
        gflops(&mkl_all),
        gflops(&Some(tuned.clone())),
        gflops(&cm),
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
    // the single-core `wuk` (which carries an output snapshot) — the interleaved all-core pair skips
    // the snapshot (its kernel is proven bit-identical to serial by the runtime unit test, and serial
    // is the cross-checked `wuk` here), so we validate the serial output and trust the equivalence.
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
        let wuk = bench_wukong(&wk_matmul_tn(ns, false), &mut c, ap, bp);
        let wk_par = bench_wukong(&wk_matmul_tn(ns, true), &mut c, ap, bp);
        let cm = bench_external(
            "c",
            &c_matmul_tn(ns),
            dir,
            "matmul_tn",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut c,
            ap,
            bp,
        );
        let rm = bench_external(
            "rs",
            &rust_matmul_tn(ns),
            dir,
            "matmul_tn",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
        );
        // Reassociation-normalized peer: -ffast-math may vectorize the column-strided reduction.
        let cfast = bench_c_fast("matmul_tn", &c_matmul_tn(ns), dir, cc, &wuk, &mut c, ap, bp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
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

fn rust_matmul_tn(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 let a = core::slice::from_raw_parts(a, NS*NS);\n\
         \x20 let b = core::slice::from_raw_parts(b, NS*NS);\n\
         \x20 let c = core::slice::from_raw_parts_mut(c, NS*NS);\n\
         \x20 for v in c.iter_mut() {{ *v = 0.0; }}\n\
         \x20 for k in 0..NS {{\n\
         \x20   let brow = &b[k*NS..k*NS+NS];\n\
         \x20   for i in 0..NS {{\n\
         \x20     let aki = a[k*NS+i];\n\
         \x20     let crow = &mut c[i*NS..i*NS+NS];\n\
         \x20     for (cv, &bv) in crow.iter_mut().zip(brow) {{ *cv += aki*bv; }}\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
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
        println!("=== gemv (y[i] = Sum_j A[i,j]*x[j]) {m}x{n} (A-stream GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_gemv(m, n, false), &mut y, ap, xp);
        let wk_par = bench_wukong(&wk_gemv(m, n, true), &mut y, ap, xp);
        let cm = bench_external(
            "c", &c_gemv(m, n), dir, "gemv", cc,
            &["-O3", "-march=native", "-shared"], &mut y, ap, xp,
        );
        let rm = bench_external(
            "rs", &rust_gemv(m, n), dir, "gemv", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut y, ap, xp,
        );
        // Reassociation-normalized peer. Wukong's `wukong_sgemv` folds each row across four 8-wide
        // accumulators — it REASSOCIATES the dot — so a plain-flags C column that must keep the sum
        // strictly left-to-right is not the like-for-like comparison. Printed alongside, never
        // instead of, the honest-default C ratio.
        let cfast = bench_c_fast("gemv", &c_gemv(m, n), dir, cc, &wuk, &mut y, ap, xp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s", gbps(&wuk), gbps(&wk_par), gbps(&cm), gbps(&cfast), gbps(&rm)
        );
        // The 8-wide row dot reassociates → magnitude-normalized tolerance (max|Δ| / max|C|), not a
        // pointwise ratio (mean-zero inputs put outputs near 0). Same basis as the reduction cross-checks.
        if let (Some(a2), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |mx, &v| mx.max(v.abs())).max(1e-6);
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
        println!("=== scaled_gemm (scores = (Q·Kᵀ)·scale) S={s} D={d} (GFLOP/s, higher is better) ===");
        let wuk = bench_wukong(&wk_scaled_scores(s, d, false), &mut out, qp, kp);
        let wk_par = bench_wukong(&wk_scaled_scores(s, d, true), &mut out, qp, kp);
        // Unscaled QKᵀ (plain nt) — same tuned kernel without the α; the ratio isolates the α cost.
        let wk_noscale = bench_wukong(&wk_scores_noscale(s, d, false), &mut out, qp, kp);
        let cm = bench_external(
            "c", &c_scaled_scores(s, d), dir, "scaled_gemm", cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"], &mut out, qp, kp,
        );
        let rm = bench_external(
            "rs", &rust_scaled_scores(s, d), dir, "scaled_gemm", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut out, qp, kp,
        );
        // Reassociation-normalized peer: the D-long score dot is a float reduction Wukong's blocked
        // GEMM accumulates out of order, so C is given the same freedom in this column.
        let cfast = bench_c_fast("scaled_gemm", &c_scaled_scores(s, d), dir, cc, &wuk, &mut out, qp, kp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s", gflops(&wuk), gflops(&wk_par), gflops(&cm), gflops(&cfast), gflops(&rm)
        );
        // The D-long score dot reassociates (FMA + blocked accumulation vs C's naive scalar order), and
        // the small mixed-sign Q/K put some scores near 0 → a pointwise relative check divides by ~0 and
        // blows up on a purely-reassociation Δ. Use the magnitude-normalized tolerance (max|Δ| / max|C|),
        // the same basis as gemv and the reduction cross-checks. The bit-exact gate is the runtime f64
        // reference + interp==native unit test; this is only a sanity ceiling.
        if let (Some(a2), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |mx, &v| mx.max(v.abs())).max(1e-6);
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
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        if let (Some(ms), Some(mn)) = (&wuk, &wk_noscale) {
            // >1 ⇒ the α costs time; ~1.0 ⇒ the scale is free (folded into the writeback).
            let r = ms.ns_per_call / mn.ns_per_call;
            println!("  -> α overhead vs unscaled QKᵀ (same kernel): {r:.3}x (≈1.0 ⇒ the scale is free)");
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
        // Single-core group, coolest-first (thermal hygiene, mirrors bench_matmul_size): Wuk(1c) and
        // MKL(1c) adjacent at the head, then tuned, then the naive C/Rust/C(fast) nests. The all-core
        // group (MKL(all), C(omp), Wuk(par)) runs last so any residual heat lands on Wukong, never a
        // peer. nn.Linear (C=A·Bᵀ) is the shape that dominates a transformer, so it is now measured
        // against real oneMKL — the true SOTA bar — not only the naive C nests and the tuned(mm) crate.
        let wuk = bench_wukong(&wk_linear(ns, false), &mut c, ap, bp);
        let mkl_1c = bench_mm_mkl(ns, true, 1, &a, &b, &mut c);
        let tuned = bench_mm_tuned(ns, true, &a, &b, &mut c);
        let cm = bench_external(
            "c",
            &c_linear(ns),
            dir,
            "linear",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut c,
            ap,
            bp,
        );
        let rm = bench_external(
            "rs",
            &rust_linear(ns),
            dir,
            "linear",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
        );
        // Reassociation-normalized peer: with -ffast-math gcc may reassociate + vectorize the
        // ijk dot-product reduction that the honest-flags column keeps serial.
        let cfast = bench_c_fast("linear", &c_linear(ns), dir, cc, &wuk, &mut c, ap, bp);
        // All-core group: MKL(all) first (coolest — rayon still dormant), then C(omp), then Wuk(par) last.
        let mkl_all = mkl()
            .map(|api| api.max_threads)
            .and_then(|t| bench_mm_mkl(ns, true, t, &a, &b, &mut c));
        // Multithreaded C peer: rows across cores + -ffast-math (the reduction row rule).
        let comp = omp_threads(cc, dir)
            .and_then(|_| {
                bench_external(
                    "c",
                    &c_linear_omp(ns),
                    dir,
                    "linear_omp",
                    cc,
                    C_OMP_FAST_FLAGS,
                    &mut c,
                    ap,
                    bp,
                )
            })
            .filter(|p| wuk.as_ref().is_some_and(|m| relaxed_peer_ok("linear", "C(omp)", m, p)));
        let wk_par = bench_wukong(&wk_linear(ns, true), &mut c, ap, bp);
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "", "Wuk(1c)", "Wuk(par)", "MKL(1c)", "MKL(all)", "tuned(mm)", "C(gcc)", "C(fast)", "C(omp)", "Rust"
        );
        println!(
            "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&mkl_1c),
            gflops(&mkl_all),
            gflops(&Some(tuned.clone())),
            gflops(&cm),
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
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        report_relaxed_ratio("C(omp) [-fopenmp -ffast-math, all cores]", &wuk, &wk_par, &comp);
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
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {{\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++){{ float s=0.0f;\n\
         \x20     for (long k=0;k<NS;k++) s+=a[i*NS+k]*b[j*NS+k];\n\
         \x20     c[i*NS+j]=s; }}\n}}\n"
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
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{ let mut s=0.0f32;\n\
         \x20     for k in 0..NS {{ s+=*a.add(i*NS+k)* *b.add(j*NS+k); }}\n\
         \x20     *c.add(i*NS+j)=s; }} }}\n}}\n"
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
        let wuk = bench_wukong(&wk_ffn(ns, false), &mut c, ap, bp);
        let wk_par = bench_wukong(&wk_ffn(ns, true), &mut c, ap, bp);
        let cm = bench_external(
            "c",
            &c_ffn(ns),
            dir,
            "ffn",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut c,
            ap,
            bp,
        );
        let rm = bench_external(
            "rs",
            &rust_ffn(ns),
            dir,
            "ffn",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
        );
        // Reassociation-normalized peer for the GEMM half of the fused op (-ffast-math may
        // reassociate the dot-product; the scalar-expf silu pass is unaffected either way).
        let cfast = bench_c_fast("ffn", &c_ffn(ns), dir, cc, &wuk, &mut c, ap, bp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
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
        let wuk = bench_wukong_bf16(&wk_linear_bf16(ns, false), &mut c, ap, bp);
        let wk_par = bench_wukong_bf16(&wk_linear_bf16(ns, true), &mut c, ap, bp);
        let cm = bench_external_bf16(
            "c",
            &c_linear_bf16(ns),
            dir,
            "linear_bf16",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut c,
            ap,
            bp,
        );
        let rm = bench_external_bf16(
            "rs",
            &rust_linear_bf16(ns),
            dir,
            "linear_bf16",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
        );
        // Reassociation-normalized peer: the K-long f32 accumulation is a float reduction, and
        // Wukong's widen-prepass + tuned GEMM accumulates it blocked. Validity-checked at the loose
        // 1e-2 bar on c[0], the same sanity guard the plain column uses.
        let cfast = bench_bf16_fast("linear_bf16", &c_linear_bf16(ns), dir, cc, &wuk, &mut c, ap, bp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&wuk),
            gflops(&wk_par),
            gflops(&cm),
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

/// Matrix transpose `dst = srcᵀ` — the memory-bound layout op (attention score transposes, weight
/// layout conversions). Wukong folds the `dst[j*R+i] = src[i*C+j]` nest to the cache-blocked
/// `wukong_transpose_f32`; the C/Rust peers are **also** 32×32 cache-blocked (see [`c_transpose`]),
/// so what is measured is Wukong's tile-mover against gcc's/rustc's codegen for the same algorithm —
/// not blocked-vs-unblocked. gcc still does not loop-tile a transpose *by itself*, which is why the
/// blocking has to be written out; the point is that a competent programmer writes it.
/// The kernels carry an unused middle pointer so they share the `(src, _, dst)` `KernelFn` ABI and the
/// f32 harness. Square shapes large enough to spill L2 (where the cache pattern dominates), reported as
/// GB/s (`2·N²·4` bytes moved per call: read `src` + write `dst`). Transpose is a permutation, so the
/// cross-language check is **bit-exact** (no float reassociation — a stronger bar than the GEMM gate).
fn bench_transpose(cc: &str, dir: &Path) {
    for ns in [1024usize, 2048] {
        let n2 = ns * ns;
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
        println!("=== transpose (dst = srcᵀ) {ns}x{ns} (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_transpose(ns, false), &mut dst, sp, yp);
        let wk_par = bench_wukong(&wk_transpose(ns, true), &mut dst, sp, yp);
        let cm = bench_external(
            "c",
            &c_transpose(ns),
            dir,
            "transpose",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut dst,
            sp,
            yp,
        );
        let rm = bench_external(
            "rs",
            &rust_transpose(ns),
            dir,
            "transpose",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut dst,
            sp,
            yp,
        );
        // Multithreaded C peer: rows across cores (a permutation — no reduction, plain -fopenmp).
        let comp = omp_threads(cc, dir)
            .and_then(|_| {
                bench_external(
                    "c",
                    &c_transpose_omp(ns),
                    dir,
                    "transpose_omp",
                    cc,
                    C_OMP_FLAGS,
                    &mut dst,
                    sp,
                    yp,
                )
            })
            .filter(|p| wuk.as_ref().is_some_and(|m| relaxed_peer_ok("transpose", "C(omp)", m, p)));
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(omp)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
            gbps(&comp),
            gbps(&rm)
        );
        // Transpose is a permutation — exact, so the full-buffer cross-check is bit equality.
        // BOTH peers are checked: a peer whose spelling changed (blocked, sliced) could in principle
        // get "faster" by not doing the work, and an unchecked column would never say so.
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
        }
        if let (Some(mp), Some(c2)) = (&wk_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            par_standing("idiomatic single-threaded C", r);
        }
        report_relaxed_ratio("C(omp) [-fopenmp, all cores]", &wuk, &wk_par, &comp);
        println!();
    }
}

/// Wukong transpose kernel: the idiomatic `dst[j*R+i] = src[i*C+j]` nest the `mir_build` recognizer
/// folds to one `wukong_transpose_f32[_parallel]` call. `y` is an unused param so the signature
/// matches the `(src, _, dst)` 3-pointer harness ABI.
fn wk_transpose(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(src: [f32; {n2}], y: [f32; {n2}], mut dst: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{ dst[j * {ns} + i] = src[i * {ns} + j]; }}\n\
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
/// total handicap**. `NS` is 1024/2048 here, both multiples of 32, so the blocked nest needs no
/// remainder handling.
fn c_transpose(ns: usize) -> String {
    format!(
        "#define NS {ns}\n#define TB 32\n\
         __declspec(dllexport) void kbench(const float* __restrict__ src, const float* __restrict__ y, float* __restrict__ dst){{\n\
         \x20 (void)y;\n\
         \x20 for (long ii=0;ii<NS;ii+=TB) for (long jj=0;jj<NS;jj+=TB)\n\
         \x20   for (long i=ii;i<ii+TB;i++)\n\
         \x20     for (long j=jj;j<jj+TB;j++) dst[j*NS+i] = src[i*NS+j];\n}}\n"
    )
}

/// The OpenMP twin of [`c_transpose`]: blocked tile rows across cores. A permutation, so it stays exact.
fn c_transpose_omp(ns: usize) -> String {
    format!(
        "#define NS {ns}\n#define TB 32\n\
         __declspec(dllexport) void kbench(const float* __restrict__ src, const float* __restrict__ y, float* __restrict__ dst){{\n\
         \x20 (void)y;\n\
         #pragma omp parallel for\n\
         \x20 for (long ii=0;ii<NS;ii+=TB) for (long jj=0;jj<NS;jj+=TB)\n\
         \x20   for (long i=ii;i<ii+TB;i++)\n\
         \x20     for (long j=jj;j<jj+TB;j++) dst[j*NS+i] = src[i*NS+j];\n}}\n"
    )
}

fn rust_transpose(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\nconst TB: usize = 32;\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(src:*const f32, _y:*const f32, dst:*mut f32) {{\n\
         \x20 let s = core::slice::from_raw_parts(src, NS*NS);\n\
         \x20 let d = core::slice::from_raw_parts_mut(dst, NS*NS);\n\
         \x20 for ii in (0..NS).step_by(TB) {{ for jj in (0..NS).step_by(TB) {{\n\
         \x20   for i in ii..ii+TB {{ for j in jj..jj+TB {{ d[j*NS+i] = s[i*NS+j]; }} }} }} }}\n}}\n"
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
        let wuk = bench_wukong(&wk_colsum(m, n, false), &mut out, xp, yp);
        let wk_par = bench_wukong(&wk_colsum(m, n, true), &mut out, xp, yp);
        let cm = bench_external(
            "c",
            &c_colsum(m, n),
            dir,
            "colsum",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut out,
            xp,
            yp,
        );
        let rm = bench_external(
            "rs",
            &rust_colsum(m, n),
            dir,
            "colsum",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
        );
        // Reassociation-normalized peer: -ffast-math lets gcc reassociate the strided column fold.
        let cfast = bench_c_fast("colsum", &c_colsum(m, n), dir, cc, &wuk, &mut out, xp, yp);
        // Multithreaded C peer: columns across cores (+ -ffast-math, the reduction-row rule).
        let comp = omp_threads(cc, dir)
            .and_then(|_| {
                bench_external(
                    "c",
                    &c_colsum_omp(m, n),
                    dir,
                    "colsum_omp",
                    cc,
                    C_OMP_FAST_FLAGS,
                    &mut out,
                    xp,
                    yp,
                )
            })
            .filter(|p| wuk.as_ref().is_some_and(|mm| relaxed_peer_ok("colsum", "C(omp)", mm, p)));
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "C(omp)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
        report_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &wk_par, &cfast);
        report_relaxed_ratio("C(omp) [-fopenmp -ffast-math, all cores]", &wuk, &wk_par, &comp);
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
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 let xs = core::slice::from_raw_parts(x, M*N);\n\
         \x20 let os = core::slice::from_raw_parts_mut(out, N);\n\
         \x20 for o in os.iter_mut() {{ *o = 0.0; }}\n\
         \x20 for i in 0..M {{ let row = &xs[i*N..i*N+N]; for (o, &v) in os.iter_mut().zip(row) {{ *o += v; }} }}\n}}\n"
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
        let wuk = bench_wukong(&wk_biasadd(r, c, false), &mut out, xp, bp);
        let wk_par = bench_wukong(&wk_biasadd(r, c, true), &mut out, xp, bp);
        let cm = bench_external(
            "c",
            &c_biasadd(r, c),
            dir,
            "biasadd",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut out,
            xp,
            bp,
        );
        let rm = bench_external(
            "rs",
            &rust_biasadd(r, c),
            dir,
            "biasadd",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            bp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
fn bench_dequant(cc: &str, cxx: &str, dir: &Path) {
    let ext_flags_c = ["-O3", "-march=native", "-ffp-contract=fast", "-shared"];
    let ext_flags_rs = ["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"];

    // --- 1-D dequant: out[j] = (q[j] as f32)·scale, for i32 and i8 inputs -------------------------
    for (ty, in_bytes, is_i8) in [("i32", 4usize, false), ("i8", 1usize, true)] {
        let n = 1usize << 23; // 8M elements: f32 output = 32 MiB ≫ L3, so NT stores engage.
        // A varied signed int fill (both signs, magnitudes past the i8 boundary for i32).
        let qi32: Vec<i32> = (0..n).map(|i| (((i as i64 * 1103515245 + 12345) >> 9) as i32) % 4096 - 2048).collect();
        let qi8: Vec<i8> = (0..n).map(|i| ((i * 37 + 5) % 251) as i64 as i8).collect();
        let mut out = vec![0.0f32; n];
        let qp = if is_i8 { qi8.as_ptr() as *const f32 } else { qi32.as_ptr() as *const f32 };
        // A REAL, distinct filler for the unused middle pointer. It used to be `out.as_ptr()`, which
        // aliased the output buffer the kernel writes — harmless while the C peer never dereferenced
        // it, but a latent `restrict` violation the moment the peer's parameters carry `__restrict__`
        // (an unused-but-aliasing `restrict` pointer to a modified object). No peer — Wukong, C, C++
        // or Rust — ever indexes this parameter, so one element is enough; the point is only that it
        // is not the output buffer.
        let dummy_buf = vec![0.0f32; 1];
        let dummy = dummy_buf.as_ptr();
        let bytes = (in_bytes + 4) as f64 * n as f64;
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== dequant-1d ({ty}: out[j] = (q[j] as f32)*scale) N={n} (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_dequant_1d(n, is_i8, false), &mut out, qp, dummy);
        let wk_par = bench_wukong(&wk_dequant_1d(n, is_i8, true), &mut out, qp, dummy);
        let cm = bench_external("c", &c_dequant_1d(n, is_i8), dir, "dequant1d", cc, &ext_flags_c, &mut out, qp, dummy);
        let cpp = bench_external("cpp", &cpp_from_c(&c_dequant_1d(n, is_i8)), dir, "dequant1d", cxx, &ext_flags_c, &mut out, qp, dummy);
        let rm = bench_external("rs", &rust_dequant_1d(n, is_i8), dir, "dequant1d", "rustc", &ext_flags_rs, &mut out, qp, dummy);
        println!("  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}", "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust");
        println!("  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}", "GB/s", gbps(&wuk), gbps(&wk_par), gbps(&cm), gbps(&cpp), gbps(&rm));
        dequant_check(&wuk, &cm, "C");
        dequant_ratio(&wuk, &wk_par, &cm, &cpp, &rm);
        println!();
    }

    // --- per-channel dequant: out[i*C+j] = (q[i*C+j] as f32)·scale[j] -----------------------------
    for (ty, in_bytes, is_i8) in [("i32", 4usize, false), ("i8", 1usize, true)] {
        let (r, c) = (8192usize, 1024usize); // 8M elements, 32 MiB f32 out ≫ L3
        let rc = r * c;
        let qi32: Vec<i32> = (0..rc).map(|i| (((i as i64 * 22695477 + 1) >> 7) as i32) % 4096 - 2048).collect();
        let qi8: Vec<i8> = (0..rc).map(|i| ((i * 29 + 7) % 251) as i64 as i8).collect();
        let scale: Vec<f32> = (0..c).map(|j| (j % 7) as f32 * 0.003 + 0.002).collect();
        let mut out = vec![0.0f32; rc];
        let qp = if is_i8 { qi8.as_ptr() as *const f32 } else { qi32.as_ptr() as *const f32 };
        let sp = scale.as_ptr();
        let bytes = (in_bytes + 4) as f64 * rc as f64; // scale[C] is negligible
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== dequant-perchan ({ty}: out[i*C+j] = (q as f32)*scale[j]) {r}x{c} (GB/s) ===");
        let wuk = bench_wukong(&wk_dequant_perchan(r, c, is_i8, false), &mut out, qp, sp);
        let wk_par = bench_wukong(&wk_dequant_perchan(r, c, is_i8, true), &mut out, qp, sp);
        let cm = bench_external("c", &c_dequant_perchan(r, c, is_i8), dir, "dequantpc", cc, &ext_flags_c, &mut out, qp, sp);
        let cpp = bench_external("cpp", &cpp_from_c(&c_dequant_perchan(r, c, is_i8)), dir, "dequantpc", cxx, &ext_flags_c, &mut out, qp, sp);
        let rm = bench_external("rs", &rust_dequant_perchan(r, c, is_i8), dir, "dequantpc", "rustc", &ext_flags_rs, &mut out, qp, sp);
        println!("  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}", "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C++ (g++)", "Rust");
        println!("  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}", "GB/s", gbps(&wuk), gbps(&wk_par), gbps(&cm), gbps(&cpp), gbps(&rm));
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
            let at = m.out.iter().zip(&p.out).position(|(x, y)| x != y).unwrap_or(0);
            println!("  ! dequant output mismatch vs {lang} at [{at}]: {} vs {}", m.out[at], p.out[at]);
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
        println!("  -> Wukong single-core is {:.2}x {} than C (gcc -O3 -march=native)", if r >= 1.0 { r } else { 1.0 / r }, if r >= 1.0 { "faster" } else { "slower" });
    }
    if let (Some(m), Some(cpp)) = (wuk, cpp) {
        let r = cpp.ns_per_call / m.ns_per_call;
        println!("  -> Wukong single-core is {:.2}x {} than C++ (g++)", if r >= 1.0 { r } else { 1.0 / r }, if r >= 1.0 { "faster" } else { "slower" });
    }
    if let (Some(m), Some(rm)) = (wuk, rm) {
        let r = rm.ns_per_call / m.ns_per_call;
        println!("  -> Wukong single-core is {:.2}x {} than Rust (rustc -Copt-level=3)", if r >= 1.0 { r } else { 1.0 / r }, if r >= 1.0 { "faster" } else { "slower" });
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
            let wuk = bench_wukong(&wk_colmax(m, n, false, opc), &mut out, xp, yp);
            let wk_par = bench_wukong(&wk_colmax(m, n, true, opc), &mut out, xp, yp);
            let cm = bench_external(
                "c",
                &c_colmax(m, n, opc),
                dir,
                label,
                cc,
                &["-O3", "-march=native", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rm = bench_external(
                "rs",
                &rust_colmax(m, n, opc),
                dir,
                label,
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            // Reassociation-normalized peer. `fmax`/`fmin` are not associative, so at honest flags
            // gcc keeps a max/min reduction in source order; `-ffast-math` lets it vectorize the
            // fold, which is what Wukong's `_mm256_max_ps` lane fold does.
            let cfast = bench_c_fast(label, &c_colmax(m, n, opc), dir, cc, &wuk, &mut out, xp, yp);
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
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
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 let xs = core::slice::from_raw_parts(x, M*N);\n\
         \x20 let os = core::slice::from_raw_parts_mut(out, N);\n\
         \x20 for (o, &v) in os.iter_mut().zip(&xs[0..N]) {{ *o = {lo}v{hi}; }}\n\
         \x20 for i in 1..M {{ let row = &xs[i*N..i*N+N];\n\
         \x20   for (o, &r) in os.iter_mut().zip(row) {{ let s=*o; let v={lo}r{hi}; *o = if s{cmp}v {{s}} else {{v}}; }} }}\n}}\n"
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
            let x: Vec<f32> = (0..n).map(|i| ((i * 31 + 7) % 101) as f32 * 0.5 - 25.0).collect();
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
            println!("=== {label} (out[r] = {sym}_j x[r,j]) {rows}x{cols} (GB/s, higher is better) ===");
            let wuk = bench_wukong(&wk_rowarg(rows, cols, is_max, false), &mut out, xp, yp);
            let wk_par = bench_wukong(&wk_rowarg(rows, cols, is_max, true), &mut out, xp, yp);
            let cm = bench_external(
                "c",
                &c_rowarg(rows, cols, is_max),
                dir,
                label,
                cc,
                &["-O3", "-march=native", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rm = bench_external(
                "rs",
                &rust_rowarg(rows, cols, is_max),
                dir,
                label,
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            // Reassociation-normalized peer (exact-index-checked, see `bench_c_fast_idx`).
            let cfast =
                bench_c_fast_idx(label, &c_rowarg(rows, cols, is_max), dir, cc, &wuk, &mut out, xp, yp);
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
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

fn rust_colarg(rows: usize, cols: usize, is_max: bool) -> String {
    let cmp = if is_max { ">" } else { "<" };
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\n#[no_mangle]\n\
         pub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut i32) {{\n\
         \x20 let xs = core::slice::from_raw_parts(x, R*C);\n\
         \x20 let os = core::slice::from_raw_parts_mut(out, C);\n\
         \x20 let mut bv = vec![0.0f32; C];\n\
         \x20 for j in 0..C {{ bv[j]=xs[j]; os[j]=0; }}\n\
         \x20 for i in 1..R {{ let row = &xs[i*C..i*C+C];\n\
         \x20   for j in 0..C {{ let v=row[j]; if v {cmp} bv[j] {{ bv[j]=v; os[j]=i as i32; }} }} }}\n}}\n"
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
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 + 11) % 103) as f32 * 0.5 - 25.0).collect();
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
            println!("=== {label} (out[j] = {sym}_i x[i,j]) {rows}x{cols} (GB/s, higher is better) ===");
            let wuk = bench_wukong(&wk_colarg(rows, cols, is_max, false), &mut out, xp, yp);
            let wk_par = bench_wukong(&wk_colarg(rows, cols, is_max, true), &mut out, xp, yp);
            let cm = bench_external(
                "c",
                &c_colarg(rows, cols, is_max),
                dir,
                label,
                cc,
                &["-O3", "-march=native", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rm = bench_external(
                "rs",
                &rust_colarg(rows, cols, is_max),
                dir,
                label,
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            // Reassociation-normalized peer (exact-index-checked, see `bench_c_fast_idx`).
            let cfast =
                bench_c_fast_idx(label, &c_colarg(rows, cols, is_max), dir, cc, &wuk, &mut out, xp, yp);
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
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
/// idle ports: a genuine single-core WIN (~1.6–1.9×), since gcc/rustc may not legally re-order an f32
/// recurrence across rows. `@parallel` maps independent row chunks across cores on top (~5–8×). GB/s =
/// `3·R·C·4` (read `a` + read `b` + write `out`). The recurrence is sequential within a row (no
/// reassociation) and the kernel does plain mul+add (two roundings) where gcc may fuse to one `fma`, so
/// the cross-check is a ~1-ULP magnitude-normalized tolerance like cumsum.
fn bench_lrscan(cc: &str, dir: &Path) {
    for (rows, cols) in [(1024usize, 1024usize), (4096, 1024)] {
        let n = rows * cols;
        // `a` (the gate) bounded in (−1, 1) so the recurrence is a contraction and `h` stays finite;
        // `b` (the input drive) small. Deterministic, no RNG.
        let a: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 19) as f32 * 0.1 - 0.9).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 11) as f32 * 0.2 - 1.0).collect();
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
        let wuk = bench_wukong(&wk_lrscan(rows, cols, false), &mut out, ap, bp);
        let wk_par = bench_wukong(&wk_lrscan(rows, cols, true), &mut out, ap, bp);
        let cm = bench_external(
            "c",
            &c_lrscan(rows, cols),
            dir,
            "lrscan",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut out,
            ap,
            bp,
        );
        let rm = bench_external(
            "rs",
            &rust_lrscan(rows, cols),
            dir,
            "lrscan",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            ap,
            bp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
        println!("=== cumprod (out[r,i] = Prod_k<=i x[r,k]) {rows}x{cols} (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_cumprod(rows, cols, false), &mut out, xp, yp);
        let wk_par = bench_wukong(&wk_cumprod(rows, cols, true), &mut out, xp, yp);
        let cm = bench_external(
            "c",
            &c_cumprod(rows, cols),
            dir,
            "cumprod",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut out,
            xp,
            yp,
        );
        let rm = bench_external(
            "rs",
            &rust_cumprod(rows, cols),
            dir,
            "cumprod",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
        let x: Vec<f32> = (0..n).map(|i| ((i * 31 + 7) % 101) as f32 * 0.01 - 0.5).collect();
        let dummy = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let (xp, yp) = (x.as_ptr(), dummy.as_ptr());
        let bytes = n as f64 * 4.0 * 2.0; // read x + write out
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== cumsum (out[r,i] = Sum_k<=i x[r,k]) {rows}x{cols} (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_cumsum(rows, cols, false), &mut out, xp, yp);
        let wk_par = bench_wukong(&wk_cumsum(rows, cols, true), &mut out, xp, yp);
        let cm = bench_external(
            "c",
            &c_cumsum(rows, cols),
            dir,
            "cumsum",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut out,
            xp,
            yp,
        );
        let rm = bench_external(
            "rs",
            &rust_cumsum(rows, cols),
            dir,
            "cumsum",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
        );
        // Reassociation-normalized peer. Wukong's Hillis-Steele in-lane tree scan REASSOCIATES the
        // prefix sum (that is exactly why this bench's cross-check is a tolerance and not bit
        // equality), so the plain-flags C column — which must keep `out[i]=out[i-1]+x[i]` in source
        // order — is not the like-for-like peer. This is the rule the file applies to `dot`/`ssd`;
        // `cumsum` is the one scan that also needs it (cumprod / cummax / cummin / lrscan stay
        // bit-exact in Wukong, so their plain-flags column already IS like-for-like).
        let cfast = bench_c_fast("cumsum", &c_cumsum(rows, cols), dir, cc, &wuk, &mut out, xp, yp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
            let x: Vec<f32> = (0..n).map(|i| ((i * 47 + 13) % 101) as f32 * 0.5 - 25.0).collect();
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
            let wuk = bench_wukong(&wk_cumminmax(rows, cols, is_max, false), &mut out, xp, yp);
            let wk_par = bench_wukong(&wk_cumminmax(rows, cols, is_max, true), &mut out, xp, yp);
            let cm = bench_external(
                "c",
                &c_cumminmax(rows, cols, is_max),
                dir,
                label,
                cc,
                &["-O3", "-march=native", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rm = bench_external(
                "rs",
                &rust_cumminmax(rows, cols, is_max),
                dir,
                label,
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
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
            let desc = if opc == 0 {
                "x[i,j]"
            } else {
                "x[i,j]²"
            };
            println!("=== {label} (out[j] = {sym}_i {desc}) {m}x{n} (GB/s, higher is better) ===");
            let wuk = bench_wukong(&wk_colstat(m, n, false, opc), &mut out, xp, yp);
            let wk_par = bench_wukong(&wk_colstat(m, n, true, opc), &mut out, xp, yp);
            let cm = bench_external(
                "c",
                &c_colstat(m, n, opc),
                dir,
                label,
                cc,
                &["-O3", "-march=native", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rm = bench_external(
                "rs",
                &rust_colstat(m, n, opc),
                dir,
                label,
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            // Reassociation-normalized peer: -ffast-math lets gcc reassociate the strided fold.
            let cfast = bench_c_fast(label, &c_colstat(m, n, opc), dir, cc, &wuk, &mut out, xp, yp);
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&wk_par),
                gbps(&cm),
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
        _ => (format!("s = s + {prod} * {prod};"), format!("sqrt(s / {m}.0)")),
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
        _ => ("{ float v=x[i*N+j]; out[j] += v*v; }", "sqrtf(s / (float)M)"),
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
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 let xs = core::slice::from_raw_parts(x, M*N);\n\
         \x20 let os = core::slice::from_raw_parts_mut(out, N);\n\
         \x20 for o in os.iter_mut() {{ *o = 0.0; }}\n\
         \x20 for i in 0..M {{ let row = &xs[i*N..i*N+N]; for (o, &r) in os.iter_mut().zip(row) {{ {fold} }} }}\n\
         \x20 for o in os.iter_mut() {{ let s = *o; *o = {fin}; }}\n}}\n"
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
        let wuk = bench_wukong(&wk_softmax_bwd(r, c, false), &mut dx, yp, dyp);
        let wk_par = bench_wukong(&wk_softmax_bwd(r, c, true), &mut dx, yp, dyp);
        let cm = bench_external(
            "c",
            &c_softmax_bwd(r, c),
            dir,
            "softmax_bwd",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut dx,
            yp,
            dyp,
        );
        let rm = bench_external(
            "rs",
            &rust_softmax_bwd(r, c),
            dir,
            "softmax_bwd",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut dx,
            yp,
            dyp,
        );
        // Reassociation-normalized peer: -ffast-math lets gcc reassociate the per-row dot.
        let cfast = bench_c_fast("softmax_bwd", &c_softmax_bwd(r, c), dir, cc, &wuk, &mut dx, yp, dyp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
        let x: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.1 + 0.3).collect();
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
        let wuk = bench_wukong4(&wk_rmsnorm_bwd(r, c, false), &mut dx, xp, dyp, gp);
        let wk_par = bench_wukong4(&wk_rmsnorm_bwd(r, c, true), &mut dx, xp, dyp, gp);
        let cm = bench_external4(
            "c", &c_rmsnorm_bwd(r, c), dir, "rmsnorm_bwd", cc,
            &["-O3", "-march=native", "-shared"], &mut dx, xp, dyp, gp,
        );
        let rm = bench_external4(
            "rs", &rust_rmsnorm_bwd(r, c), dir, "rmsnorm_bwd", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut dx, xp, dyp, gp,
        );
        // Reassociation-normalized peer: -ffast-math lets gcc reassociate the two per-row reductions.
        let cfast = bench_external4(
            "c", &c_rmsnorm_bwd(r, c), dir, "rmsnorm_bwd_fast", cc,
            C_FAST_FLAGS, &mut dx, xp, dyp, gp,
        )
        .filter(|cf| wuk.as_ref().is_some_and(|m| relaxed_peer_ok("rmsnorm_bwd", "C(fast)", m, cf)));
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s", gbps(&wuk), gbps(&wk_par), gbps(&cm), gbps(&cfast), gbps(&rm)
        );
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m.out.iter().zip(&c2.out).fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
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
        let x: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.1 + 0.3).collect();
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
        let wuk = bench_wukong4(&wk_layernorm_bwd(r, c, false), &mut dx, xp, dyp, gp);
        let wk_par = bench_wukong4(&wk_layernorm_bwd(r, c, true), &mut dx, xp, dyp, gp);
        let cm = bench_external4(
            "c", &c_layernorm_bwd(r, c), dir, "layernorm_bwd", cc,
            &["-O3", "-march=native", "-shared"], &mut dx, xp, dyp, gp,
        );
        let rm = bench_external4(
            "rs", &rust_layernorm_bwd(r, c), dir, "layernorm_bwd", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut dx, xp, dyp, gp,
        );
        // Reassociation-normalized peer: -ffast-math lets gcc reassociate the four per-row reductions.
        let cfast = bench_external4(
            "c", &c_layernorm_bwd(r, c), dir, "layernorm_bwd_fast", cc,
            C_FAST_FLAGS, &mut dx, xp, dyp, gp,
        )
        .filter(|cf| wuk.as_ref().is_some_and(|m| relaxed_peer_ok("layernorm_bwd", "C(fast)", m, cf)));
        report_ratio("layernorm_bwd", &wuk, &wk_par, &cm, &rm, &cfast, &gbps);
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
        let wuk = bench_wukong(&wk_xent(r, c, false), &mut loss, xp, tp);
        let wk_par = bench_wukong(&wk_xent(r, c, true), &mut loss, xp, tp);
        let cm = bench_external(
            "c", &c_xent(r, c), dir, "xent", cc,
            &["-O3", "-march=native", "-shared"], &mut loss, xp, tp,
        );
        let rm = bench_external(
            "rs", &rust_xent(r, c), dir, "xent", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut loss, xp, tp,
        );
        // Reassociation-normalized peer: -ffast-math on the row-max + Σexp reductions.
        let cfast = bench_c_fast("xent", &c_xent(r, c), dir, cc, &wuk, &mut loss, xp, tp);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "C(fast)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GB/s", gbps(&wuk), gbps(&wk_par), gbps(&cm), gbps(&cfast), gbps(&rm)
        );
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m.out.iter().zip(&c2.out).fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
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
        let wuk = bench_wukong(&wk_rope(rows, half, false), &mut out, xp, fp);
        let wk_par = bench_wukong(&wk_rope(rows, half, true), &mut out, xp, fp);
        let cm = bench_external(
            "c", &c_rope(rows, half), dir, "rope", cc,
            &["-O3", "-march=native", "-shared"], &mut out, xp, fp,
        );
        let rm = bench_external(
            "rs", &rust_rope(rows, half), dir, "rope", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut out, xp, fp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s", gbps(&wuk), gbps(&wk_par), gbps(&cm), gbps(&rm)
        );
        if let (Some(m), Some(c2)) = (&wuk, &cm) {
            let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let maxerr = m.out.iter().zip(&c2.out).fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
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
        let wuk = bench_wukong(&wk_xent_bwd(r, c, false), &mut dx, xp, tp);
        let wk_par = bench_wukong(&wk_xent_bwd(r, c, true), &mut dx, xp, tp);
        let cm = bench_external(
            "c", &c_xent_bwd(r, c), dir, "xent_bwd", cc,
            &["-O3", "-march=native", "-shared"], &mut dx, xp, tp,
        );
        let rm = bench_external(
            "rs", &rust_xent_bwd(r, c), dir, "xent_bwd", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut dx, xp, tp,
        );
        // Reassociation-normalized peer: the per-row `max` and `Σexp` are float reductions Wukong
        // folds 8 lanes wide, so the plain-flags C column is the strictly-in-order one. (`rope`,
        // `rope_bwd`, `gate` and `act_backward` deliberately pass `&None` here — they are pure
        // elementwise maps with no reduction, so `-ffast-math` normalizes nothing for them.)
        let cfast = bench_c_fast("xent_bwd", &c_xent_bwd(r, c), dir, cc, &wuk, &mut dx, xp, tp);
        report_ratio("xent_bwd", &wuk, &wk_par, &cm, &rm, &cfast, &gbps);
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
        let inv_freq: Vec<f32> = (0..half).map(|k| 10000f32.powf(-(k as f32) / half as f32)).collect();
        let mut dx = vec![0.0f32; n];
        let (gp, fp) = (g.as_ptr(), inv_freq.as_ptr());
        let bytes = (2.0 * n as f64 + half as f64) * 4.0;
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== rope_bwd (transpose rotate) {rows}x{d} (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_rope_bwd(rows, half, false), &mut dx, gp, fp);
        let wk_par = bench_wukong(&wk_rope_bwd(rows, half, true), &mut dx, gp, fp);
        let cm = bench_external(
            "c", &c_rope_bwd(rows, half), dir, "rope_bwd", cc,
            &["-O3", "-march=native", "-shared"], &mut dx, gp, fp,
        );
        let rm = bench_external(
            "rs", &rust_rope_bwd(rows, half), dir, "rope_bwd", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut dx, gp, fp,
        );
        report_ratio("rope_bwd", &wuk, &wk_par, &cm, &rm, &None, &gbps);
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

/// Gated-FFN activation `out[i] = act(a[i]) * b[i]` (SwiGLU silu / GeGLU gelu) at N=2²⁰. The
/// activation's exp gcc/rustc keep scalar → Wukong's 256-bit `wukong_vmath2_f32` gate op. The op is
/// selected 0=silu/1=gelu. All-f32 3-pointer harness `(a, b, out)`.
fn bench_gate(cc: &str, dir: &Path) {
    let n = 1usize << 20;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 - (n / 2) as f32) * (12.0 / n as f32)).collect();
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
        println!("=== gate {name}: out = {act}(a)*b, N=2^20 (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_gate(n, act, false), &mut out, ap, bp);
        let wk_par = bench_wukong(&wk_gate(n, act, true), &mut out, ap, bp);
        let cm = bench_external(
            "c", &c_gate(n, act), dir, "gate", cc,
            &["-O3", "-march=native", "-shared"], &mut out, ap, bp,
        );
        let rm = bench_external(
            "rs", &rust_gate(n, act), dir, "gate", "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut out, ap, bp,
        );
        report_ratio("gate", &wuk, &wk_par, &cm, &rm, &None, &gbps);
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
            let wuk = bench_wukong(&src_m, &mut out, p0, p1);
            let wk_par = bench_wukong(&src_mp, &mut out, p0, p1);
            let cm = bench_external(
                "c", &src_c, dir, label, cc,
                &["-O3", "-march=native", "-shared"], &mut out, p0, p1,
            );
            let rm = bench_external(
                "rs", &src_r, dir, label, "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"], &mut out, p0, p1,
            );
            // Reassociation-normalized peer: the per-row logf/expf reductions under -ffast-math.
            let cfast = bench_c_fast(label, &src_c, dir, cc, &wuk, &mut out, p0, p1);
            report_ratio(label, &wuk, &wk_par, &cm, &rm, &cfast, &gbps);
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

/// Shared 4-column ratio report for the backward/gate benches (Wuk 1-core / Wuk par / C / Rust + the
/// single-core and @parallel ratios vs idiomatic C). The cross-check (when both present) uses a magnitude-
/// normalized tolerance, since the transcendental reductions reassociate / differ from libm by ~1 ULP.
/// `cfast` is the optional relaxed-FP C(fast) peer column (already loose-cross-checked by the caller);
/// benches whose C baseline is not an IEEE-serial reduction pass `&None`.
fn report_ratio(
    label: &str,
    wuk: &Option<Measure>,
    wk_par: &Option<Measure>,
    cm: &Option<Measure>,
    rm: &Option<Measure>,
    cfast: &Option<Measure>,
    gbps: &dyn Fn(&Option<Measure>) -> String,
) {
    let has_fast = cfast.is_some();
    let mut hdr = format!(
        "  {:<10} {:>11} {:>11} {:>11} {:>11}",
        "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
    );
    let mut vals = format!(
        "  {:<10} {:>11} {:>11} {:>11} {:>11}",
        "GB/s", gbps(wuk), gbps(wk_par), gbps(cm), gbps(rm)
    );
    if has_fast {
        hdr.push_str(&format!(" {:>11}", "C(fast)"));
        vals.push_str(&format!(" {:>11}", gbps(cfast)));
    }
    println!("{hdr}");
    println!("{vals}");
    if let (Some(m), Some(c2)) = (wuk, cm) {
        let maxabs = c2.out.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
        let maxerr = m.out.iter().zip(&c2.out).fold(0.0f32, |a, (&x, &y)| a.max((x - y).abs()));
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
fn bench_act_backward(cc: &str, dir: &Path) {
    let n = 1usize << 20;
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
        println!("=== {op}_backward (dx = dy·{op}'(x)) N={n} (GB/s, higher is better) ===");
        let wuk = bench_wukong(&wk_act_backward(n, op, false), &mut dx, xp, dyp);
        let wk_par = bench_wukong(&wk_act_backward(n, op, true), &mut dx, xp, dyp);
        let cm = bench_external(
            "c",
            &c_act_backward(n, op),
            dir,
            "act_backward",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut dx,
            xp,
            dyp,
        );
        let rm = bench_external(
            "rs",
            &rust_act_backward(n, op),
            dir,
            "act_backward",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut dx,
            xp,
            dyp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&wuk),
            gbps(&wk_par),
            gbps(&cm),
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
        _ => "float c0=0.7978845608f,c1=0.044715f; float u=tanhf(c0*(v+c1*v*v*v)); \
              float g=0.5f*(1.0f+u)+0.5f*v*(1.0f-u*u)*c0*(1.0f+3.0f*c1*v*v);",
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
        _ => "let (c0,c1)=(0.7978845608f32,0.044715f32); let u=(c0*(v+c1*v*v*v)).tanh(); \
              let g=0.5*(1.0+u)+0.5*v*(1.0-u*u)*c0*(1.0+3.0*c1*v*v);",
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
        let wuk = bench_wukong_i8(&wk_i8gemm(ns, false), &mut c, ap, bp);
        let wk_par = bench_wukong_i8(&wk_i8gemm(ns, true), &mut c, ap, bp);
        let cm = bench_external_i8(
            "c",
            &c_i8gemm(ns),
            dir,
            "i8gemm",
            cc,
            &["-O3", "-march=native", "-shared"],
            &mut c,
            ap,
            bp,
        );
        let rm = bench_external_i8(
            "rs",
            &rust_i8gemm(ns),
            dir,
            "i8gemm",
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Wuk(1core)", "Wuk(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GOP/s",
            gops(&wuk),
            gops(&wk_par),
            gops(&cm),
            gops(&rm)
        );
        // Compile time (Wukong front-end + JIT vs gcc/rustc to a shared lib) — Wukong is far cheaper.
        let cms = |m: &Option<MeasureI8>| {
            m.as_ref()
                .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "compile ms",
            cms(&wuk),
            cms(&wk_par),
            cms(&cm),
            cms(&rm)
        );
        // Integer arithmetic → exact equality is the correct cross-language bar. Every available
        // result must match the single-core Wukong reference exactly; a mismatch is a real bug.
        if let Some(m) = &wuk {
            for (lang, other) in [("Wuk(par)", &wk_par), ("C", &cm), ("Rust", &rm)] {
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
            let wuk = bench_wukong_bf16(&wk_bf16(n, is_dot), &mut o, xp, yp);
            let cm = bench_external_bf16(
                "c",
                &c_bf16(n, is_dot),
                dir,
                "bf16",
                cc,
                &["-O3", "-march=native", "-shared"],
                &mut o,
                xp,
                yp,
            );
            let rm = bench_external_bf16(
                "rs",
                &rust_bf16(n, is_dot),
                dir,
                "bf16",
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut o,
                xp,
                yp,
            );
            // Reassociation-normalized peer: this row IS a float reduction (bf16 in, f32
            // accumulate) and Wukong folds it 8 lanes wide, so the honest-flags C column is the
            // in-order sum and C(fast) is the like-for-like one.
            let cfast = bench_bf16_fast("bf16", &c_bf16(n, is_dot), dir, cc, &wuk, &mut o, xp, yp);
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                kind, "Wukong", "C (gcc)", "C(fast)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&wuk),
                gbps(&cm),
                gbps(&cfast),
                gbps(&rm)
            );
            // All-positive, well-conditioned reduction → a tight relative tolerance is the right
            // cross-language bar (the three reassociate the f32 sum differently: Wukong 8-lane SIMD
            // vs sequential C/Rust). Report the scalars so any drift is visible.
            if let Some(m) = &wuk {
                for (lang, other) in [("C", &cm), ("Rust", &rm)] {
                    if let Some(o2) = other {
                        let (a, b) = (m.out, o2.out);
                        let rel = (a - b).abs() as f64 / (a.abs().max(b.abs()).max(1e-6) as f64);
                        if rel > 1e-3 {
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
            bf16_relaxed_ratio("C(fast) [-ffast-math]", &wuk, &None, &cfast);
            // Compile time (Wukong front-end + JIT vs gcc/rustc to a shared lib).
            let cms = |m: &Option<MeasureBf16>| {
                m.as_ref()
                    .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
                    .unwrap_or_else(|| "n/a".into())
            };
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "compile ms",
                cms(&wuk),
                cms(&cm),
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
    let wk_half = bench_wukong_halfout(&wk_axpby_half_out(n, a, b), &mut oh, xp, yp);
    let wk_f32 = bench_wukong_bf16(&wk_axpby_f32_out(n, a, b), &mut of, xp, yp);
    // Lever 2: a C all-half peer (same round).
    let cm = bench_external_halfout(
        "c",
        &c_axpby_half_out(n, a, b),
        dir,
        "axpbyhalf",
        cc,
        &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        &mut oh,
        xp,
        yp,
    );
    let rm = bench_external_halfout(
        "rs",
        &rust_axpby_half_out(n, a, b),
        dir,
        "axpbyhalf",
        "rustc",
        &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        &mut oh,
        xp,
        yp,
    );
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
        "  {:<12} {:>12} {:>12} {:>12} {:>12}",
        "", "Wuk half-out", "Wuk f32-out", "C half-out", "Rust half-out"
    );
    println!(
        "  {:<12} {:>12} {:>12} {:>12} {:>12}",
        "GB/s",
        gbps(&wk_half, half_bytes),
        gbps_bf(&wk_f32, f32_bytes),
        gbps(&cm, half_bytes),
        gbps(&rm, half_bytes)
    );
    // Cross-check the C/Rust half output against Wukong's (bf16-scale tolerance: 8-bit mantissa ≈ 4e-3).
    if let Some(m) = &wk_half {
        for (lang, other) in [("C", &cm), ("Rust", &rm)] {
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
    // Lever 2 payoff: vs the C all-half peer.
    if let (Some(h), Some(c2)) = (&wk_half, &cm) {
        let r = c2.ns_per_call / h.ns_per_call;
        println!(
            "  -> Wukong half-out is {:.2}x {} than C (vectorized widen+narrow vs scalar bf16 round)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
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
    let wuk = bench_wukong(&wk_conv(cin, h, cout, k), &mut output, ip, wp);
    let cm = bench_external(
        "c",
        &c_conv(cin, h, cout, k),
        dir,
        "conv",
        cc,
        &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        &mut output,
        ip,
        wp,
    );
    let rm = bench_external(
        "rs",
        &rust_conv(cin, h, cout, k),
        dir,
        "conv",
        "rustc",
        &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        &mut output,
        ip,
        wp,
    );
    // Reassociation-normalized peer: the direct convolution's `Cin·K·K`-long accumulation is a float
    // reduction, and Wukong's im2col+GEMM path accumulates it in a blocked (reassociated) order.
    let cfast = bench_c_fast("conv", &c_conv(cin, h, cout, k), dir, cc, &wuk, &mut output, ip, wp);
    println!(
        "  {:<18} {:>14} {:>14} {:>14} {:>14}",
        "", "Wuk im2col+GEMM", "C (direct)", "C(fast) direct", "Rust (direct)"
    );
    println!(
        "  {:<18} {:>14} {:>14} {:>14} {:>14}",
        "GFLOP/s",
        gflops(&wuk),
        gflops(&cm),
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
/// languages copy `x`→`out` then normalize `out` in place (identical work), so the cross-language
/// full-buffer check sees the same result. One row = one token's hidden vector; a real `[tokens,
/// hidden]` batch maps the identical kernel per row.
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
            "  {:<16} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16}",
            "", "Wukong", "C (gcc)", "C(fast)", "Rust", "Wuk vs C", "vs C(fast)"
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
            let wuk = bench_wukong(&wk_norm(cols, op), &mut out, xp, yp);
            let c = bench_external(
                "c",
                &c_norm(cols, op),
                dir,
                &format!("norm_{op}"),
                cc,
                &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rust = bench_external(
                "rs",
                &rust_norm(cols, op),
                dir,
                &format!("norm_{op}"),
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            // Reassociation-normalized peer: with -ffast-math gcc may reassociate the norm's
            // reductions (and vectorize expf via its own fast paths). Loose-checked inside.
            let cfast =
                bench_c_fast(&format!("norm_{op}"), &c_norm(cols, op), dir, cc, &wuk, &mut out, xp, yp);
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
                "  {:<16} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16}",
                op,
                ns(&wuk),
                ns(&c),
                ns(&cfast),
                ns(&rust),
                standing_vs(&c),
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
            "  {:<18} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16}",
            "", "Wukong", "C (gcc)", "C(fast)", "Rust", "Wuk vs C", "vs C(fast)"
        );
        for op in ["rmsnorm", "layernorm", "softmax"] {
            // C/Rust baselines are single-threaded per-row norms — the same for both Wukong rows
            // (serial + @parallel), so time once per op.
            let c = bench_external(
                "c",
                &c_norm_batched(rows, cols, op),
                dir,
                "bnorm",
                cc,
                &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
                &mut out,
                xp,
                yp,
            );
            let rust = bench_external(
                "rs",
                &rust_norm_batched(rows, cols, op),
                dir,
                "bnorm",
                "rustc",
                &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
            );
            // Reassociation-normalized peer, also once per op (single-threaded like the C column);
            // loose-checked against the serial Wukong row below before the column is shown.
            let mut cfast_raw = bench_external(
                "c",
                &c_norm_batched(rows, cols, op),
                dir,
                "bnorm_fast",
                cc,
                C_FAST_FLAGS,
                &mut out,
                xp,
                yp,
            );
            let mut cfast: Option<Measure> = None;
            for (suffix, par) in [("", false), ("@parallel", true)] {
                let label = format!("{op}{suffix}");
                let wuk = bench_wukong(
                    &wk_norm_batched(rows, cols, op, par),
                    &mut out,
                    xp,
                    yp,
                );
                if !par {
                    cfast = cfast_raw.take().filter(|cf| {
                        wuk.as_ref()
                            .is_some_and(|m| relaxed_peer_ok(&label, "C(fast)", m, cf))
                    });
                }
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
                    "  {:<18} {:>12} {:>12} {:>12} {:>12} {:>16} {:>16}",
                    label,
                    ns(&wuk),
                    ns(&c),
                    ns(&cfast),
                    ns(&rust),
                    standing_vs(&c),
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
    // caller (`bench_c_fast` / `relaxed_peer_ok`), so this loop covers C and C++ only.
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

/// Compile a Wukong kernel to native code (timed) and benchmark it.
///
/// ONE-LIVE-POINTER LAW (every harness in this file obeys it). The kernel's output pointer is
/// derived from `out` *here*, not handed in by the caller. A caller that hoisted `op =
/// buf.as_mut_ptr()` and then passed `&mut buf` alongside it would be lying to the optimizer:
/// `&mut [f32]` lowers to a `noalias` parameter, which asserts that nothing else reaching this
/// function touches that memory — yet the JIT'd kernel writes through exactly such a pointer. LLVM
/// would then be entitled to forward the zeroing stores across the opaque call and lower the
/// `to_vec()` read-back to a memset, so every cross-language check in the suite would compare
/// all-zeros against all-zeros and pass vacuously. Deriving `op` from `out` keeps the kernel's
/// writes *based on* the parameter, which is exactly what `noalias` permits.
fn bench_wukong(src: &str, out: &mut [f32], xp: *const f32, yp: *const f32) -> Option<Measure> {
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
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let handle = match wukong_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("wukong codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    if !kbench_abi_ok(&program, sym, 3) {
        return None;
    }
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    // SAFETY: `kbench_abi_ok` just checked the lowered `kbench` is (ptr, ptr, ptr) -> void,
    // which is exactly `KernelFn`; `handle` is kept alive across the call below.
    let f: KernelFn = unsafe { std::mem::transmute(ptr) };

    out.iter_mut().for_each(|v| *v = 0.0);
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot = out.to_vec();
    drop(handle); // keep alive through timing
    Some(Measure {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// Compile a C/Rust source to a shared library with `compiler args…` (timed), load it, and bench.
/// Derives the kernel's output pointer from `out` — see the one-live-pointer law on [`bench_wukong`].
#[allow(clippy::too_many_arguments)]
fn bench_external(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
) -> Option<Measure> {
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

    unsafe {
        let lib = match libloading::Library::new(&dll) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("load {}: {e}", dll.display());
                return None;
            }
        };
        let sym: libloading::Symbol<KernelFn> = match lib.get(b"kbench\0") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("symbol kbench in {}: {e}", dll.display());
                return None;
            }
        };
        let f: KernelFn = *sym;
        out.iter_mut().for_each(|v| *v = 0.0);
        let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
        let ns = time_ns(|| f(xp, yp, op));
        let snapshot = out.to_vec();
        Some(Measure {
            compile,
            ns_per_call: ns,
            out: snapshot,
        })
    }
}

/// A 4-pointer kernel ABI `(p0, p1, p2: *const, op: *mut)` — for kernels that read three input buffers
/// and write one output (e.g. the norm backwards `(x, dy, gamma) -> dx`), which don't fit the 3-pointer
/// `KernelFn`. Same timing/snapshot protocol as [`bench_wukong`]/[`bench_external`].
type KernelFn4 = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);

/// 4-pointer twin of [`bench_wukong`] (same one-live-pointer law).
fn bench_wukong4(
    src: &str,
    out: &mut [f32],
    p0: *const f32,
    p1: *const f32,
    p2: *const f32,
) -> Option<Measure> {
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
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let handle = match wukong_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("wukong codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    if !kbench_abi_ok(&program, sym, 4) {
        return None;
    }
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    // SAFETY: `kbench_abi_ok` just checked the lowered `kbench` is (ptr, ptr, ptr, ptr) -> void,
    // which is exactly `KernelFn4`; `handle` is kept alive across the call below.
    let f: KernelFn4 = unsafe { std::mem::transmute(ptr) };
    out.iter_mut().for_each(|v| *v = 0.0);
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(p0, p1, p2, op) });
    let snapshot = out.to_vec();
    drop(handle);
    Some(Measure {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// 4-pointer twin of [`bench_external`].
#[allow(clippy::too_many_arguments)]
fn bench_external4(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
    out: &mut [f32],
    p0: *const f32,
    p1: *const f32,
    p2: *const f32,
) -> Option<Measure> {
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
    unsafe {
        let lib = match libloading::Library::new(&dll) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("load {}: {e}", dll.display());
                return None;
            }
        };
        let sym: libloading::Symbol<KernelFn4> = match lib.get(b"kbench\0") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("symbol kbench in {}: {e}", dll.display());
                return None;
            }
        };
        let f: KernelFn4 = *sym;
        out.iter_mut().for_each(|v| *v = 0.0);
        let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
        let ns = time_ns(|| f(p0, p1, p2, op));
        let snapshot = out.to_vec();
        Some(Measure {
            compile,
            ns_per_call: ns,
            out: snapshot,
        })
    }
}

/// The int8 twin of [`bench_wukong`]: JIT the int8 GEMM kernel and time it through the `(u8, i8,
/// i32)` ABI.
fn bench_wukong_i8(src: &str, out: &mut [i32], ap: *const u8, bp: *const i8) -> Option<MeasureI8> {
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
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let handle = match wukong_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("wukong codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    if !kbench_abi_ok(&program, sym, 3) {
        return None;
    }
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    // SAFETY: `kbench_abi_ok` just checked the lowered `kbench` is (ptr, ptr, ptr) -> void,
    // which is exactly `I8KernelFn`; `handle` is kept alive across the call below.
    let f: I8KernelFn = unsafe { std::mem::transmute(ptr) };

    out.iter_mut().for_each(|v| *v = 0);
    let cp = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(ap, bp, cp) });
    let snapshot = out.to_vec();
    drop(handle); // keep alive through timing
    Some(MeasureI8 {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// The int8 twin of [`bench_external`]: compile a C/Rust int8 GEMM to a shared lib and time it.
#[allow(clippy::too_many_arguments)]
fn bench_external_i8(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
    out: &mut [i32],
    ap: *const u8,
    bp: *const i8,
) -> Option<MeasureI8> {
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

    unsafe {
        let lib = match libloading::Library::new(&dll) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("load {}: {e}", dll.display());
                return None;
            }
        };
        let sym: libloading::Symbol<I8KernelFn> = match lib.get(b"kbench\0") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("symbol kbench in {}: {e}", dll.display());
                return None;
            }
        };
        let f: I8KernelFn = *sym;
        out.iter_mut().for_each(|v| *v = 0);
        let cp = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
        let ns = time_ns(|| f(ap, bp, cp));
        let snapshot = out.to_vec();
        Some(MeasureI8 {
            compile,
            ns_per_call: ns,
            out: snapshot,
        })
    }
}

/// The bf16 twin of [`bench_wukong_i8`]: JIT-compile a Wukong bf16 reduction and time it. `out` is a
/// 1-element f32 buffer holding the scalar result.
fn bench_wukong_bf16(
    src: &str,
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureBf16> {
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
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let handle = match wukong_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("wukong codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    if !kbench_abi_ok(&program, sym, 3) {
        return None;
    }
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    // SAFETY: `kbench_abi_ok` just checked the lowered `kbench` is (ptr, ptr, ptr) -> void,
    // which is exactly `Bf16KernelFn`; `handle` is kept alive across the call below.
    let f: Bf16KernelFn = unsafe { std::mem::transmute(ptr) };

    out[0] = 0.0;
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot = out[0];
    drop(handle); // keep alive through timing
    Some(MeasureBf16 {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// The bf16 twin of [`bench_external_i8`]: compile a C/Rust bf16 reduction to a shared lib and time it.
#[allow(clippy::too_many_arguments)]
fn bench_external_bf16(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureBf16> {
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

    unsafe {
        let lib = match libloading::Library::new(&dll) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("load {}: {e}", dll.display());
                return None;
            }
        };
        let sym: libloading::Symbol<Bf16KernelFn> = match lib.get(b"kbench\0") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("symbol kbench in {}: {e}", dll.display());
                return None;
            }
        };
        let f: Bf16KernelFn = *sym;
        out[0] = 0.0;
        let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
        let ns = time_ns(|| f(xp, yp, op));
        let snapshot = out[0];
        Some(MeasureBf16 {
            compile,
            ns_per_call: ns,
            out: snapshot,
        })
    }
}

/// The all-half twin of [`bench_wukong_bf16`]: JIT a Wukong kernel with the `(x, y, out)` all-`u16`
/// (bf16) ABI, time it, and snapshot the half output buffer widened to f32 for the cross-check.
fn bench_wukong_halfout(
    src: &str,
    out: &mut [u16],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureHalfOut> {
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
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let handle = match wukong_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("wukong codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    if !kbench_abi_ok(&program, sym, 3) {
        return None;
    }
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    // SAFETY: `kbench_abi_ok` just checked the lowered `kbench` is (ptr, ptr, ptr) -> void,
    // which is exactly `HalfOutKernelFn`; `handle` is kept alive across the call below.
    let f: HalfOutKernelFn = unsafe { std::mem::transmute(ptr) };
    let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
    let ns = time_ns(|| unsafe { f(xp, yp, op) });
    let snapshot: Vec<f32> = out.iter().map(|&b| widen_bf16(b)).collect();
    drop(handle); // keep alive through timing
    Some(MeasureHalfOut {
        compile,
        ns_per_call: ns,
        out: snapshot,
    })
}

/// The all-half twin of [`bench_external_bf16`]: compile a C/Rust all-half kernel to a shared lib,
/// time it, and snapshot the widened half output.
#[allow(clippy::too_many_arguments)]
fn bench_external_halfout(
    ext: &str,
    src: &str,
    dir: &Path,
    name: &str,
    compiler: &str,
    args: &[&str],
    out: &mut [u16],
    xp: *const u16,
    yp: *const u16,
) -> Option<MeasureHalfOut> {
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

    unsafe {
        let lib = match libloading::Library::new(&dll) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("load {}: {e}", dll.display());
                return None;
            }
        };
        let sym: libloading::Symbol<HalfOutKernelFn> = match lib.get(b"kbench\0") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("symbol kbench in {}: {e}", dll.display());
                return None;
            }
        };
        let f: HalfOutKernelFn = *sym;
        let op = out.as_mut_ptr(); // the ONE live pointer to `out` while the kernel runs
        let ns = time_ns(|| f(xp, yp, op));
        let snapshot: Vec<f32> = out.iter().map(|&b| widen_bf16(b)).collect();
        Some(MeasureHalfOut {
            compile,
            ns_per_call: ns,
            out: snapshot,
        })
    }
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
type MklGetMaxThreadsFn = unsafe extern "C" fn() -> i32;
// oneMKL VML (Vector Math Library) single-precision unary op: `vsExp(n, a, y)` ⇒ `y[i] = exp(a[i])`.
// The `n` width follows the same interface layer as CBLAS — ILP64 here (so `i64`, matching the
// `cblas_sgemm_64` we resolve). The default HA (high-accuracy, ~0.5 ULP) mode is left in force; the
// per-kernel cross-check vs Wukong's ~1-ULP poly tolerates the ≤2-ULP difference and would flag an
// ABI mismatch (garbage output) by reporting a large rel error, in which case the column is dropped.
type VmlUnaryFn = unsafe extern "C" fn(i64, *const f32, *mut f32);

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
    path: PathBuf,
}

/// Best-effort discovery of `mkl_rt.dll`: an explicit `WUKONG_MKL_DLL` override first, then the
/// standard conda layouts (`$CONDA_PREFIX`, `%USERPROFILE%\{Anaconda3,miniconda3,…}\Library\bin`,
/// and the system-wide ProgramData install).
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
        for name in ["Anaconda3", "anaconda3", "miniconda3", "Miniconda3", "miniforge3"] {
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

/// Load oneMKL once (cached). `LOAD_WITH_ALTERED_SEARCH_PATH` makes the Windows loader resolve MKL's
/// own dependencies (`libiomp5md.dll`, `mkl_core`, `mkl_intel_thread`) from the DLL's directory.
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
            let set_threads = *lib.get::<MklSetNumThreadsFn>(b"MKL_Set_Num_Threads\0").ok()?;
            let get_max = *lib.get::<MklGetMaxThreadsFn>(b"MKL_Get_Max_Threads\0").ok()?;
            // VML transcendentals are optional (older MKL builds, or a stripped redist, may omit them).
            let vml = |sym: &[u8]| lib.get::<VmlUnaryFn>(sym).ok().map(|s| *s);
            let vs_exp = vml(b"vsExp\0");
            let vs_ln = vml(b"vsLn\0");
            let vs_tanh = vml(b"vsTanh\0");
            MklApi {
                sgemm,
                set_threads,
                max_threads: get_max(),
                vs_exp,
                vs_ln,
                vs_tanh,
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_roofline_avx2() -> f64 {
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
    {
        let t = Instant::now();
        let mut warm = 0f32;
        while t.elapsed() < Duration::from_millis(400) {
            warm += run(2_000_000);
        }
        std::hint::black_box(warm);
    }
    let iters = 40_000_000u64;
    let mut best = f64::INFINITY;
    let mut sink = 0f32;
    for _ in 0..8 {
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

fn geomean(xs: &[f64]) -> f64 {
    let s: f64 = xs.iter().map(|x| x.ln()).sum();
    (s / xs.len() as f64).exp()
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
        "  {:<10} {:>10} {:>10} {:>10} {:>14}",
        "", "Wukong", "C (gcc)", "Rust", "Wuk vs C"
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
        let wuk = bench_wukong(&wk_kernel_n(NL, mb), &mut out, xp, yp);
        let c = bench_external(
            "c",
            &c_kernel_n(NL, cb),
            dir,
            &format!("stream_{name}"),
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut out,
            xp,
            yp,
        );
        let rust = bench_external(
            "rs",
            &rust_kernel_n(NL, rb),
            dir,
            &format!("stream_{name}"),
            "rustc",
            &["-Copt-level=3", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
        );
        let gbs = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", *bytes as f64 / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        let standing = if let (Some(m), Some(c)) = (&wuk, &c) {
            let r = c.ns_per_call / m.ns_per_call;
            format!(
                "{:.2}x {}",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            )
        } else {
            "n/a".into()
        };
        println!(
            "  {:<10} {:>10} {:>10} {:>10} {:>14}",
            name,
            gbs(&wuk),
            gbs(&c),
            gbs(&rust),
            standing
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

fn rust_kernel(body: &str) -> String {
    // `#[allow(unused_variables)]`: some kernels (relu, poly) don't read `y`; the fixed `(x,y,out)`
    // ABI keeps the param, so silence the warning rather than clutter the benchmark output.
    format!("#[allow(dead_code)]\nconst N: usize = {N};\n#[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  {body}\n}}\n")
}

/// Render a C kernel source as **C++** for the g++ peer column. C is a subset of C++, so the numeric
/// body compiles unchanged; we only prepend `extern "C"` to the export so the C++ compiler keeps the
/// `kbench` symbol unmangled (libloading looks it up by that exact name — exactly how real C++ projects
/// expose a C-ABI kernel). `<math.h>` stays (valid C++, keeps `expf`/`logf`/… in global scope), so no
/// body rewrite is needed. Idiomatic C++ for a numeric kernel *is* this loop (a `std::transform` lowers
/// to the same code), and g++ shares gcc's middle/back-end — so this measures whether the C++ toolchain
/// beats Wukong (it does not), proving the "beat C++" claim instead of assuming it.
fn cpp_from_c(c_src: &str) -> String {
    c_src.replace("__declspec(dllexport)", "extern \"C\" __declspec(dllexport)")
}

// Parameterized kernel builders (an explicit element count `n`) — used by the large-tensor streaming
// benchmark, which runs at N=2²⁴ rather than the module-global N=2²⁰.
fn wk_kernel_n(n: usize, body: &str) -> String {
    format!("module bench\nfn kbench(x: [f32; {n}], y: [f32; {n}], mut out: [f32; {n}]) {{\n    {body}\n}}\n")
}
fn c_kernel_n(n: usize, body: &str) -> String {
    format!("#include <math.h>\n#define N {n}\n__declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out) {{\n  {body}\n}}\n")
}
fn rust_kernel_n(n: usize, body: &str) -> String {
    format!("const N: usize = {n};\n#[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  {body}\n}}\n")
}

/// Regression guards for the pure decision functions this file's peer columns and correctness
/// checks hang off. None of them were covered before: `crates/wukong_xbench` had no test over
/// `main.rs` at all, so a rename that silently DROPPED a peer column — weakening the comparison
/// rather than breaking the build — left `cargo test` green.
#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(is_reduction_kernel(n), "`{n}` dropped out of is_reduction_kernel");
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
            assert!(c_omp_source(n).is_some(), "`{n}` dropped out of c_omp_source");
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
        assert!(rel.is_infinite(), "NaN vs finite must not read as bit-exact");
        assert_eq!(at, 1);
        assert!(max_rel_err(&[0.42], &[f32::NAN]).0.is_infinite());
        // +Inf vs -Inf makes `rel` NaN, which must not be silently skipped either.
        assert!(
            max_rel_err(&[f32::INFINITY], &[f32::NEG_INFINITY])
                .0
                .is_infinite()
        );
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

    /// EVERY generated C kernel must declare its pointer parameters `__restrict__`. Scanned out of
    /// this file's own source text rather than from a hand-kept list, so a NEW peer generator added
    /// later is covered automatically — a list would have to be remembered, and the thing being
    /// guarded against is exactly a peer nobody remembered to check.
    #[test]
    fn every_generated_c_kernel_declares_restrict() {
        const SRC: &str = include_str!("main.rs");
        // Split so this needle does not occur contiguously in the file it scans — otherwise the
        // test matches its own source line and reports its own Rust code as an unqualified peer.
        const OPEN: &str = concat!("__declspec(dllexport)", " void kbench(");
        let mut seen = 0usize;
        let mut rest = SRC;
        while let Some(i) = rest.find(OPEN) {
            let after = &rest[i + OPEN.len()..];
            let end = after.find(')').expect("kbench parameter list must close on one line");
            let params = &after[..end];
            for p in params.split(',') {
                assert!(
                    p.contains("__restrict__"),
                    "C peer parameter `{}` is not __restrict__ (full list: `{params}`). Without it \
                     gcc must assume the output aliases the inputs and cannot vectorize the kernel \
                     — see the peer-strength checklist above.",
                    p.trim()
                );
            }
            seen += 1;
            rest = &after[end..];
        }
        // A floor, so that deleting or renaming the peer generators cannot make this pass vacuously
        // with zero matches. 43 signatures at the time of writing.
        assert!(seen >= 40, "only {seen} C kbench signatures found — did the peer generators move?");
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
            assert!(s.contains("float bv[C];"), "c_colarg lost its running-best vector:\n{s}");
        }
    }

    /// `C = Aᵀ·B` must be the `kij` nest (hoisted `a[k*M+i]`, contiguous B and C), not the `ijk`
    /// nest that reads BOTH operands column-strided.
    #[test]
    fn matmul_tn_peer_stays_kij() {
        let s = c_matmul_tn(64);
        let k = s.find("for (long k=").expect("no k loop in c_matmul_tn");
        let i = s.find("for (long i=").expect("no i loop in c_matmul_tn");
        assert!(k < i, "c_matmul_tn regressed to an i-outer (ijk) nest:\n{s}");
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
    /// measures loop tiling rather than codegen.
    #[test]
    fn transpose_peer_stays_cache_blocked() {
        for ns in [1024usize, 2048] {
            let s = c_transpose(ns);
            assert!(s.contains("#define TB 32"), "c_transpose lost its blocking:\n{s}");
            assert!(
                s.contains("for (long ii=0;ii<NS;ii+=TB)"),
                "c_transpose is back to the naive un-tiled nest:\n{s}"
            );
            assert_eq!(ns % 32, 0, "the blocked transpose peer needs NS % 32 == 0");
            // The OpenMP twin must stay the same algorithm, or the C(omp) column is not comparable.
            assert!(c_transpose_omp(ns).contains("for (long ii=0;ii<NS;ii+=TB)"));
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
}
