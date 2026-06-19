//! `mercury-xbench` — an honest cross-language benchmark.
//!
//! For each kernel it builds the *same* computation three ways — Mercury (compiled to native code
//! by the Cranelift backend), C (gcc `-O3 -march=native`), and Rust (rustc `-O -C
//! target-cpu=native`) — and times all three through one identical Rust loop over the same
//! buffers. C/Rust are compiled to shared libraries and called via their C ABI; Mercury is
//! JIT-compiled in process. It also reports each toolchain's compile time.
//!
//! Fairness notes:
//!  * FMA: Mercury now contracts `x + y*z` to a fused multiply-add, so gcc is given its *default*
//!    `-ffp-contract=fast` (the old `-ffp-contract=off` was actually suppressing C's natural FMA).
//!    Both Mercury and gcc-compiled C therefore fuse. Idiomatic Rust does *not* contract unless the
//!    author writes `f32::mul_add`, so the Rust column reflects rustc's default (two rounded ops) —
//!    a real toolchain-defaults difference, not a handicap.
//!  * Reductions (dot) are strict left-to-right f32, so none of the three auto-vectorize them
//!    (though both Mercury and C may use a *scalar* FMA for the `s + x*y` step).
//!  * The comparison basis is *idiomatic, single-threaded* code at the given flags. Where Mercury
//!    later auto-parallelizes/vectorizes, that is called out explicitly.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use mercury_span::{Interner, SourceId};

/// The shared C ABI of every kernel: `(x, y, out)` over `N` `f32` elements (`N` baked in).
type KernelFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

/// The int8 quantized-GEMM ABI: `(a: u8, b: i8, c: i32)` — `u8` activations, `i8` weights, `i32`
/// accumulator (the QNNPACK/oneDNN layout). Dims are baked into the kernel source.
type I8KernelFn = unsafe extern "C" fn(*const u8, *const i8, *mut i32);

const N: usize = 1 << 20; // 1,048,576 elements (4 MiB per f32 array)

struct Kernel {
    name: &'static str,
    /// bytes of memory traffic per call (for a GB/s figure)
    bytes_per_call: usize,
    /// 2*flops per element estimate (for context only)
    note: &'static str,
    mer: String,
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

/// The maximum relative element-wise error between two output buffers, and the index where it
/// occurs. NaN-vs-NaN and same-sign-Inf agree; a small absolute floor keeps near-zero elements from
/// blowing up the ratio. This is the honest full-buffer cross-language equality check: the three
/// languages compute *slightly* differently (Mercury's ≈1-ULP poly vs libm `expf`, FMA vs not), so
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
        let (x, y) = (x as f64, y as f64);
        let rel = (x - y).abs() / x.abs().max(y.abs()).max(1e-6);
        if rel > worst {
            worst = rel;
            at = i;
        }
    }
    (worst, at)
}

fn main() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let dir = std::env::temp_dir().join("mercury_xbench");
    let _ = std::fs::create_dir_all(&dir);

    println!(
        "Cross-language kernel benchmark — Mercury (native) vs C (gcc -O3) vs Rust (rustc -O)"
    );
    println!("N = {N} f32 elements, single-threaded, -march=native. Lower ns is better.\n");

    let kernels = kernels();
    let mut runtime_ratios_c = Vec::new();
    let mut compile_ratios_c = Vec::new();

    for k in &kernels {
        // Shared buffers, filled once; kernels read x,y and write out.
        let x: Vec<f32> = (0..N).map(|i| (i as f32 % 17.0) * 0.5 + 1.0).collect();
        let y: Vec<f32> = (0..N).map(|i| (i as f32 % 13.0) * 0.25 - 0.5).collect();
        let mut out: Vec<f32> = vec![0.0; N];
        let (xp, yp, op) = (x.as_ptr(), y.as_ptr(), out.as_mut_ptr());

        let mercury = bench_mercury(&k.mer, &mut out, xp, yp, op);
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
            op,
        );
        let rust = bench_external(
            "rs",
            &k.rust,
            &dir,
            k.name,
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
            op,
        );

        println!("=== {} ({}) ===", k.name, k.note);
        report(k, &mercury, &c, &rust);
        if let (Some(m), Some(c)) = (&mercury, &c) {
            runtime_ratios_c.push(c.ns_per_call / m.ns_per_call); // >1 => Mercury faster
            compile_ratios_c.push(c.compile.as_secs_f64() / m.compile.as_secs_f64());
        }
        println!();
    }

    if !runtime_ratios_c.is_empty() {
        let g_rt = geomean(&runtime_ratios_c);
        let g_ct = geomean(&compile_ratios_c);
        println!("Summary vs C (geomean over kernels):");
        println!(
            "  runtime:  Mercury is {:.2}x {} than C",
            if g_rt >= 1.0 { g_rt } else { 1.0 / g_rt },
            if g_rt >= 1.0 { "faster" } else { "slower" }
        );
        println!("  compile:  Mercury is {g_ct:.1}x faster to compile than C");
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
    bench_matmul(&cc, &dir, roof);
    bench_linear(&cc, &dir, roof);
    bench_conv(&cc, &dir);
    bench_norm(&cc, &dir);
    bench_norm_batched(&cc, &dir);
    bench_i8gemm(&cc, &dir);
    bench_streaming_large(&cc, &dir);
}

/// Matmul is the canonical ML kernel and is compute-bound, so both SIMD and multicore pay off — the
/// regime where a tensor compiler should genuinely beat idiomatic scalar-source code. We benchmark
/// an `ikj`-ordered C = A·B (the cache-friendly idiom that auto-vectorizes well) written the same
/// way in each language, and additionally Mercury's `@parallel` form. Reported as GFLOP/s. The win
/// is shown across a size sweep so it is clearly structural, not a single-size artifact.
fn bench_matmul(cc: &str, dir: &Path, roof: f64) {
    for ns in [256usize, 512, 1024] {
        bench_matmul_size(cc, dir, ns, roof);
        println!();
    }
}

fn bench_matmul_size(cc: &str, dir: &Path, ns: usize, roof: f64) {
    let n2 = ns * ns;
    let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
    let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
    let mut c = vec![0.0f32; n2];
    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    let flops = 2.0 * (ns as f64).powi(3);

    println!("=== matmul {ns}x{ns} (C=A·B, ikj order; GFLOP/s, higher is better) ===");
    let gflops = |m: &Option<Measure>| {
        m.as_ref()
            .map(|x| format!("{:.1}", flops / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };

    let mer = bench_mercury(&mer_matmul(ns, false), &mut c, ap, bp, cp);
    let mer_par = bench_mercury(&mer_matmul(ns, true), &mut c, ap, bp, cp);
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
        cp,
    );
    let rm = bench_external(
        "rs",
        &rust_matmul(ns),
        dir,
        "matmul",
        "rustc",
        &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        &mut c,
        ap,
        bp,
        cp,
    );
    let tuned = bench_mm_tuned(ns, false, &a, &b, &mut c);

    println!(
        "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "", "Mer(1core)", "Mer(par)", "tuned(mm)", "C (gcc)", "Rust"
    );
    println!(
        "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "GFLOP/s",
        gflops(&mer),
        gflops(&mer_par),
        gflops(&Some(tuned.clone())),
        gflops(&cm),
        gflops(&rm)
    );
    report_gemm_standing(&mer, &tuned, roof, flops);
    // Cross-language correctness: every backend must compute the same C, element by element.
    if let (Some(m), Some(c)) = (&mer_par, &cm) {
        let (rel, at) = max_rel_err(&m.out, &c.out);
        if rel > 1e-3 {
            println!(
                "  ! full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                m.out[at], c.out[at], rel
            );
        }
    }
    if let (Some(mp), Some(c)) = (&mer_par, &cm) {
        let r = flops / mp.ns_per_call / (flops / c.ns_per_call);
        println!(
            "  -> Mercury @parallel is {:.2}x {} than idiomatic single-threaded C",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    if let (Some(ms), Some(c)) = (&mer, &cm) {
        let r = flops / ms.ns_per_call / (flops / c.ns_per_call);
        println!(
            "  -> Mercury single-core (SIMD) is {:.2}x {} than C single-threaded",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
}

/// `ikj`-ordered matmul, optionally `@parallel` (parallelizes the outer `i` loop across cores).
fn mer_matmul(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], c: [f32; {n2}]) {{\n\
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
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* a, const float* b, float* c) {{\n\
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

/// `nn.Linear`: `C = A·Bᵀ` (A is `[M,K]`, B is `[N,K]`), the matmul every Dense layer runs. Mercury
/// recognizes the transposed-B nest and dispatches to its tuned GEMM; idiomatic C/Rust write the
/// naive nest. Square M=K=N for the harness's shared-buffer ABI.
fn bench_linear(cc: &str, dir: &Path, roof: f64) {
    for ns in [512usize, 1024] {
        let n2 = ns * ns;
        let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut c = vec![0.0f32; n2];
        let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        let gflops = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== linear (nn.Linear C=A·Bᵀ) {ns}x{ns} (GFLOP/s, higher is better) ===");
        let mer = bench_mercury(&mer_linear(ns, false), &mut c, ap, bp, cp);
        let mer_par = bench_mercury(&mer_linear(ns, true), &mut c, ap, bp, cp);
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
            cp,
        );
        let rm = bench_external(
            "rs",
            &rust_linear(ns),
            dir,
            "linear",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
            cp,
        );
        let tuned = bench_mm_tuned(ns, true, &a, &b, &mut c);
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "", "Mer(1core)", "Mer(par)", "tuned(mm)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "GFLOP/s",
            gflops(&mer),
            gflops(&mer_par),
            gflops(&Some(tuned.clone())),
            gflops(&cm),
            gflops(&rm)
        );
        if let (Some(mp), Some(c)) = (&mer_par, &cm) {
            let r = (flops / mp.ns_per_call) / (flops / c.ns_per_call);
            println!(
                "  -> Mercury @parallel is {:.2}x faster than idiomatic single-threaded C",
                r
            );
        }
        report_gemm_standing(&mer, &tuned, roof, flops);
        println!();
    }
}

/// `nn.Linear` source, written the *idiomatic* way for `C = A·Bᵀ`: the `ijk` dot-product form, where
/// each `C[i,j]` is the dot product of A's row i and B's row j — both read contiguously (cache
/// friendly for all three languages). Mercury recognizes this and dispatches to its tiled GEMM;
/// gcc/rustc vectorize the inner reduction but never tile/pack, so Mercury wins on cache behavior.
fn mer_linear(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], c: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{\n\
         \x20           let mut s: f32 = 0.0;\n\
         \x20           for k in 0..{ns} {{ s = s + a[i * {ns} + k] * b[j * {ns} + k]; }}\n\
         \x20           c[i * {ns} + j] = s;\n\
         \x20       }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_linear(ns: usize) -> String {
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* a, const float* b, float* c) {{\n\
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

/// int8 quantized `nn.Linear` (`C = A·Bᵀ`, `u8` activations × `i8` weights → `i32` accumulator) — the
/// quantized-inference GEMM that QNNPACK/oneDNN exist for. Mercury recognizes the nest and dispatches
/// it to the AVX2 widen+`vpmaddwd` int8 microkernel; C/Rust run the *idiomatic* naive int8 GEMM at
/// `-O3 -march=native` / `-O -Ctarget-cpu=native` — whatever their auto-vectorizers produce is the
/// honest baseline (no hand intrinsics, same as every other kernel here). **Integer arithmetic, so
/// the cross-language check is bit-exact** — a stronger bar than the f32 kernels' tolerance. Reported
/// as int8 GOP/s (2 ops per multiply-accumulate). The inputs stay within `i32` (no overflow at these
/// sizes), so all three languages must agree exactly.
fn bench_i8gemm(cc: &str, dir: &Path) {
    for ns in [512usize, 1024] {
        let n2 = ns * ns;
        // u8 activations 0..250, i8 weights -125..125 — full-range-ish, and small enough that the
        // K-sum stays well inside i32 (max ≈ ns·250·125 ≪ 2³¹), so the result is overflow-free and
        // identical across languages.
        let a: Vec<u8> = (0..n2).map(|i| (i % 251) as u8).collect();
        let b: Vec<i8> = (0..n2).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let mut c = vec![0i32; n2];
        let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
        let ops = 2.0 * (ns as f64).powi(3); // 2 ops per MAC
        let gops = |m: &Option<MeasureI8>| {
            m.as_ref()
                .map(|x| format!("{:.1}", ops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== i8gemm (int8 nn.Linear C=A·Bᵀ, u8×i8→i32) {ns}x{ns} (GOP/s, higher is better) ==="
        );
        let mer = bench_mercury_i8(&mer_i8gemm(ns, false), &mut c, ap, bp, cp);
        let mer_par = bench_mercury_i8(&mer_i8gemm(ns, true), &mut c, ap, bp, cp);
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
            cp,
        );
        let rm = bench_external_i8(
            "rs",
            &rust_i8gemm(ns),
            dir,
            "i8gemm",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut c,
            ap,
            bp,
            cp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GOP/s",
            gops(&mer),
            gops(&mer_par),
            gops(&cm),
            gops(&rm)
        );
        // Compile time (Mercury front-end + JIT vs gcc/rustc to a shared lib) — Mercury is far cheaper.
        let cms = |m: &Option<MeasureI8>| {
            m.as_ref()
                .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "compile ms",
            cms(&mer),
            cms(&mer_par),
            cms(&cm),
            cms(&rm)
        );
        // Integer arithmetic → exact equality is the correct cross-language bar. Every available
        // result must match the single-core Mercury reference exactly; a mismatch is a real bug.
        if let Some(m) = &mer {
            for (lang, other) in [("Mer(par)", &mer_par), ("C", &cm), ("Rust", &rm)] {
                if let Some(o) = other {
                    if o.out != m.out {
                        let at = m
                            .out
                            .iter()
                            .zip(&o.out)
                            .position(|(x, y)| x != y)
                            .unwrap_or(0);
                        println!(
                            "  ! full-buffer mismatch {lang} vs Mer(1core) at [{at}]: {} vs {}",
                            o.out[at], m.out[at]
                        );
                    }
                }
            }
        }
        if let (Some(m), Some(c2)) = (&mer, &cm) {
            let r = (ops / m.ns_per_call) / (ops / c2.ns_per_call);
            println!(
                "  -> Mercury (1 core) is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
            let r = (ops / mp.ns_per_call) / (ops / c2.ns_per_call);
            println!("  -> Mercury @parallel is {r:.2}x faster than idiomatic single-threaded C");
        }
        println!();
    }
}

/// Mercury int8 `nn.Linear`, the idiomatic `ijk` dot-product `C = A·Bᵀ` with `u8`/`i8` operands cast
/// to `i32` before the multiply — exactly what the `mir_build` recognizer folds to one
/// `mercury_i8gemm_nt[_parallel]` call.
fn mer_i8gemm(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [u8; {n2}], b: [i8; {n2}], c: [i32; {n2}]) {{\n\
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
         __declspec(dllexport) void kbench(const uint8_t* a, const int8_t* b, int32_t* c) {{\n\
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

/// 2D convolution — the other heavy ML kernel. Mercury lowers it the XLA/cuDNN way: an im2col gather
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
    let (ip, wp, opp) = (input.as_ptr(), weight.as_ptr(), output.as_mut_ptr());
    let flops = 2.0 * (cout * ckk * ohw) as f64;

    println!(
        "=== conv2d Cin{cin} {h}x{h} -> {cout}@{oh}x{oh} (3x3); Mercury im2col+GEMM vs direct conv; GFLOP/s ==="
    );
    let gflops = |m: &Option<Measure>| {
        m.as_ref()
            .map(|x| format!("{:.1}", flops / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    let mer = bench_mercury(&mer_conv(cin, h, cout, k), &mut output, ip, wp, opp);
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
        opp,
    );
    let rm = bench_external(
        "rs",
        &rust_conv(cin, h, cout, k),
        dir,
        "conv",
        "rustc",
        &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        &mut output,
        ip,
        wp,
        opp,
    );
    println!(
        "  {:<18} {:>14} {:>14} {:>14}",
        "", "Mer im2col+GEMM", "C (direct)", "Rust (direct)"
    );
    println!(
        "  {:<18} {:>14} {:>14} {:>14}",
        "GFLOP/s",
        gflops(&mer),
        gflops(&cm),
        gflops(&rm)
    );
    if let (Some(m), Some(c)) = (&mer, &cm) {
        let (rel, at) = max_rel_err(&m.out, &c.out);
        if rel > 1e-3 {
            println!(
                "  ! full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                m.out[at], c.out[at], rel
            );
        }
        let r = (flops / m.ns_per_call) / (flops / c.ns_per_call);
        println!(
            "  -> Mercury (im2col+GEMM) is {:.2}x {} than idiomatic direct-convolution C",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" }
        );
    }
    println!();
}

/// Mercury conv: im2col into a `[Cin·K·K, OH·OW]` scratch, then `Y = W · col` (the matmul recognizer
/// dispatches the second nest to the GEMM kernel). `col` is a function-local array.
fn mer_conv(cin: usize, h: usize, cout: usize, k: usize) -> String {
    let oh = h - k + 1;
    let (hw, ohw, ckk, kk) = (h * h, oh * oh, cin * k * k, k * k);
    format!(
        "module bench\nfn kbench(input: [f32; {inlen}], weight: [f32; {wlen}], output: [f32; {olen}]) {{\n\
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
        "__declspec(dllexport) void kbench(const float* input, const float* weight, float* output) {{\n\
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
/// the per-token normalization every transformer layer runs. Mercury folds the canonical multi-pass
/// form into one `mercury_norm_f32` call: 256-bit AVX2, a hand-vectorized `exp` for softmax, and
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
        let (xp, yp, op_) = (x.as_ptr(), gamma.as_ptr(), out.as_mut_ptr());
        println!(
            "=== fused norms, 1x{cols} feature row (ns/call, lower is better; Mercury → mercury_norm_f32[_affine]) ==="
        );
        println!(
            "  {:<16} {:>12} {:>12} {:>12} {:>16}",
            "", "Mercury", "C (gcc)", "Rust", "Mer vs C"
        );
        for op in [
            "softmax",
            "layernorm",
            "rmsnorm",
            "layernorm_affine",
            "rmsnorm_affine",
        ] {
            let mer = bench_mercury(&mer_norm(cols, op), &mut out, xp, yp, op_);
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
                op_,
            );
            let rust = bench_external(
                "rs",
                &rust_norm(cols, op),
                dir,
                &format!("norm_{op}"),
                "rustc",
                &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
                op_,
            );
            let ns = |m: &Option<Measure>| {
                m.as_ref()
                    .map(|x| format!("{:.0}", x.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let standing = if let (Some(m), Some(c)) = (&mer, &c) {
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
                "  {:<16} {:>12} {:>12} {:>12} {:>16}",
                op,
                ns(&mer),
                ns(&c),
                ns(&rust),
                standing
            );
            // Cross-language correctness: same normalized row, element by element (f32 tolerance —
            // Mercury reassociates the reductions, C does not, so it is a tight rel-err, not bits).
            if let (Some(m), Some(c)) = (&mer, &c) {
                let (rel, at) = max_rel_err(&m.out, &c.out);
                if rel > 1e-3 {
                    println!(
                        "  ! {op} full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                        m.out[at], c.out[at], rel
                    );
                }
            }
        }
        println!();
    }
}

/// Mercury norm source: copy `x`→`out`, then the canonical in-place multi-pass form the recognizer
/// folds into one `mercury_norm_f32(out, out, 1, cols, eps, op)` call — or, for the `*_affine`
/// variants whose normalize step also applies the per-column `y` (gamma, and for LayerNorm beta too),
/// one `mercury_norm_affine_f32(out, out, y, y|null, 1, cols, eps, op)` call. The `/ {cols}.0` divisor
/// (and the `out[0]` softmax seed) are exactly the spellings the matchers accept, so the kernel fires.
fn mer_norm(cols: usize, op: &str) -> String {
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
        _ => format!(
            "let mut s: f32 = 0.0; \
             for i in 0..{cols} {{ s = s + out[i] * out[i]; }} \
             let inv: f32 = rsqrt(s / {cols}.0 + 0.00001); \
             for i in 0..{cols} {{ out[i] = out[i] * inv; }}"
        ),
    };
    format!(
        "module bench\nfn kbench(x: [f32; {cols}], y: [f32; {cols}], out: [f32; {cols}]) {{ \
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
        _ => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]*out[i]; \
             float inv=1.0f/sqrtf(s/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]*=inv;"
        }
    };
    format!(
        "#include <math.h>\n#define C {cols}\n__declspec(dllexport) void kbench(const float* x, const float* y, float* out) {{ \
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

/// Batched RMSNorm over a `[rows, cols]` activation — the *real* transformer shape (norm over
/// `[batch*seq, hidden]`, one row per token), where `bench_norm`'s single feature row was just one
/// token. Mercury folds the `for r { <RMSNorm over out[r*C+i]> }` loop into one `mercury_norm_f32`
/// call (each row a fused single pass); under `@parallel` the independent rows map across cores via
/// `mercury_norm_f32_parallel`. C/Rust are the idiomatic per-row nested loops at honest defaults
/// (sequential float reductions, no `-ffast-math`). All three copy `x`→`out` then normalize in place,
/// so the full-buffer cross-check is valid. The serial row is apples-to-apples (both single-threaded);
/// the `@parallel` row pits Mercury's automatic SIMD+multicore against idiomatic single-threaded C/Rust.
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
        let (xp, yp, op_) = (x.as_ptr(), x.as_ptr(), out.as_mut_ptr());
        println!(
            "=== batched RMSNorm, {rows}x{cols} = [tokens, hidden], {mb:.1} MB/buffer (ns/call, lower is better; Mercury → mercury_norm_f32[_parallel]) ==="
        );
        println!(
            "  {:<16} {:>12} {:>12} {:>12} {:>16}",
            "", "Mercury", "C (gcc)", "Rust", "Mer vs C"
        );
        // C/Rust baselines are single-threaded per-row norms — the same for both Mercury rows, so time once.
        let c = bench_external(
            "c",
            &c_norm_batched(rows, cols),
            dir,
            "bnorm",
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
            &mut out,
            xp,
            yp,
            op_,
        );
        let rust = bench_external(
            "rs",
            &rust_norm_batched(rows, cols),
            dir,
            "bnorm",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
            op_,
        );
        for (label, par) in [("rmsnorm", false), ("rmsnorm@parallel", true)] {
            let mer = bench_mercury(&mer_norm_batched(rows, cols, par), &mut out, xp, yp, op_);
            let ns = |m: &Option<Measure>| {
                m.as_ref()
                    .map(|x| format!("{:.0}", x.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let standing = if let (Some(m), Some(c)) = (&mer, &c) {
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
                "  {:<16} {:>12} {:>12} {:>12} {:>16}",
                label,
                ns(&mer),
                ns(&c),
                ns(&rust),
                standing
            );
            // Full-buffer cross-check (f32 tolerance: Mercury reassociates the per-row reductions, C does not).
            if let (Some(m), Some(c)) = (&mer, &c) {
                let (rel, at) = max_rel_err(&m.out, &c.out);
                if rel > 1e-3 {
                    println!(
                        "  ! {label} full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                        m.out[at], c.out[at], rel
                    );
                }
            }
        }
        println!();
    }
}

/// Mercury batched-RMSNorm source: copy `x`→`out`, then the `for r { <RMSNorm over out[r*C+i]> }` form
/// the `mir_build` recognizer folds to one `mercury_norm_f32(out, out, R, C, eps, op)` call — or, under
/// `@parallel`, `mercury_norm_f32_parallel` (rows across cores). The `r*{cols}+i` offset and `/{cols}.0`
/// divisor are exactly the spellings `match_batched_norm` accepts.
fn mer_norm_batched(rows: usize, cols: usize, parallel: bool) -> String {
    let n = rows * cols;
    let attr = if parallel { "@parallel\n" } else { "" };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], out: [f32; {n}]) {{ \
         for c in 0..{n} {{ out[c] = x[c]; }} \
         for r in 0..{rows} {{ \
         let mut s: f32 = 0.0; \
         for i in 0..{cols} {{ s = s + out[r*{cols}+i] * out[r*{cols}+i]; }} \
         let inv: f32 = rsqrt(s / {cols}.0 + 0.00001); \
         for i in 0..{cols} {{ out[r*{cols}+i] = out[r*{cols}+i] * inv; }} }} }}\n"
    )
}

fn c_norm_batched(rows: usize, cols: usize) -> String {
    let n = rows * cols;
    format!(
        "#include <math.h>\n#define R {rows}\n#define C {cols}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* x, const float* y, float* out) {{ \
         for(long i=0;i<N;i++) out[i]=x[i]; \
         for(long r=0;r<R;r++){{ float* o=out+(long)r*C; \
         float s=0.0f; for(long i=0;i<C;i++) s+=o[i]*o[i]; \
         float inv=1.0f/sqrtf(s/(float)C+1e-5f); \
         for(long i=0;i<C;i++) o[i]*=inv; }} }}\n"
    )
}

fn rust_norm_batched(rows: usize, cols: usize) -> String {
    let n = rows * cols;
    format!(
        "const R: usize = {rows};\nconst C: usize = {cols};\nconst N: usize = {n};\n\
         #[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{ \
         for i in 0..N {{ *out.add(i)=*x.add(i); }} \
         for r in 0..R {{ let o=out.add(r*C); \
         let mut s=0.0f32; for i in 0..C {{ let v=*o.add(i); s+=v*v; }} \
         let inv=1.0f32/(s/(C as f32)+1e-5f32).sqrt(); \
         for i in 0..C {{ *o.add(i)*=inv; }} }} }}\n"
    )
}

fn report(k: &Kernel, m: &Option<Measure>, c: &Option<Measure>, r: &Option<Measure>) {
    let row = |label: &str, f: &dyn Fn(&Measure) -> String| {
        println!(
            "  {:<14} {:>14} {:>14} {:>14}",
            label,
            m.as_ref().map(f).unwrap_or_else(|| "n/a".into()),
            c.as_ref().map(f).unwrap_or_else(|| "n/a".into()),
            r.as_ref().map(f).unwrap_or_else(|| "n/a".into()),
        );
    };
    println!(
        "  {:<14} {:>14} {:>14} {:>14}",
        "", "Mercury", "C (gcc)", "Rust"
    );
    row("compile (ms)", &|x| {
        format!("{:.1}", x.compile.as_secs_f64() * 1e3)
    });
    row("runtime (ns)", &|x| format!("{:.0}", x.ns_per_call));
    row("GB/s", &|x| {
        format!("{:.1}", k.bytes_per_call as f64 / x.ns_per_call)
    });
    // Cross-check that all three computed the same thing, element by element (within f32 tol).
    if let (Some(m), Some(c)) = (m, c) {
        let (rel, at) = max_rel_err(&m.out, &c.out);
        if rel > 1e-3 {
            println!(
                "  ! full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                m.out[at], c.out[at], rel
            );
        }
    }
    if let (Some(m), Some(c)) = (m, c) {
        let ratio = c.ns_per_call / m.ns_per_call;
        println!(
            "  -> Mercury runtime is {:.2}x {} than C; compiles {:.1}x faster",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            if ratio >= 1.0 { "faster" } else { "slower" },
            c.compile.as_secs_f64() / m.compile.as_secs_f64(),
        );
    }
}

/// Compile a Mercury kernel to native code (timed) and benchmark it.
fn bench_mercury(
    src: &str,
    out: &mut [f32],
    xp: *const f32,
    yp: *const f32,
    op: *mut f32,
) -> Option<Measure> {
    let t = Instant::now();
    let mut interner = Interner::new();
    let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("mercury parse error");
        return None;
    }
    let (sema, sd) = mercury_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("mercury sema error: {sd:?}");
        return None;
    }
    let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("mercury lower error: {ld:?}");
        return None;
    }
    mercury_opt::optimize(&mut program, 3);
    let handle = match mercury_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("mercury codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    let f: KernelFn = unsafe { std::mem::transmute(ptr) };

    out.iter_mut().for_each(|v| *v = 0.0);
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
    op: *mut f32,
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
        let ns = time_ns(|| f(xp, yp, op));
        let snapshot = out.to_vec();
        Some(Measure {
            compile,
            ns_per_call: ns,
            out: snapshot,
        })
    }
}

/// The int8 twin of [`bench_mercury`]: JIT the int8 GEMM kernel and time it through the `(u8, i8,
/// i32)` ABI.
fn bench_mercury_i8(
    src: &str,
    out: &mut [i32],
    ap: *const u8,
    bp: *const i8,
    cp: *mut i32,
) -> Option<MeasureI8> {
    let t = Instant::now();
    let mut interner = Interner::new();
    let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("mercury parse error");
        return None;
    }
    let (sema, sd) = mercury_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("mercury sema error: {sd:?}");
        return None;
    }
    let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("mercury lower error: {ld:?}");
        return None;
    }
    mercury_opt::optimize(&mut program, 3);
    let handle = match mercury_codegen_cranelift::jit_module(&program, &interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("mercury codegen error: {e}");
            return None;
        }
    };
    let sym = interner.intern("kbench");
    let ptr = handle.func_ptr(sym)?;
    let compile = t.elapsed();
    let f: I8KernelFn = unsafe { std::mem::transmute(ptr) };

    out.iter_mut().for_each(|v| *v = 0);
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
    cp: *mut i32,
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
        let ns = time_ns(|| f(ap, bp, cp));
        let snapshot = out.to_vec();
        Some(MeasureI8 {
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
    for _ in 0..5 {
        run();
    }
    let target = Duration::from_millis(50);
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

/// Print where Mercury's single-core GEMM stands: vs the tuned library (the SOTA-parity check) and
/// as a % of the measured AVX2-FMA roofline (clock-invariant — the honest figure on this box, whose
/// absolute GFLOP/s swings with the power/thermal state).
fn report_gemm_standing(mer: &Option<Measure>, tuned: &Measure, roof: f64, flops: f64) {
    if let Some(m) = mer {
        let mer_g = flops / m.ns_per_call;
        let tuned_g = flops / tuned.ns_per_call;
        let r = mer_g / tuned_g;
        println!(
            "  -> Mercury single-core is {:.2}x {} than tuned matrixmultiply ({mer_g:.0} vs {tuned_g:.0} GFLOP/s)",
            if r >= 1.0 { r } else { 1.0 / r },
            if r >= 1.0 { "faster" } else { "slower" },
        );
        if roof > 0.0 {
            println!(
                "  -> Mercury single-core = {:.0}% of measured roofline; tuned matrixmultiply = {:.0}%",
                mer_g / roof * 100.0,
                tuned_g / roof * 100.0
            );
        }
    }
}

/// Benchmark a single-threaded tuned-library GEMM (the pure-Rust `matrixmultiply` crate, the engine
/// behind `ndarray`) at the same size and layout, so Mercury's GEMM is measured against a genuine
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
    let _ = run(2_000_000); // warm up the clock
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
            mer: mer_kernel(&format!(
                "let a: f32 = 2.0; for i in 0..{nlit} {{ out[i] = a * x[i] + y[i]; }}"
            )),
            c: c_kernel("float a=2.0f; for(long i=0;i<N;i++) out[i]=a*x[i]+y[i];"),
            rust: rust_kernel("let a=2.0f32; for i in 0..N { *out.add(i)=a* *x.add(i)+ *y.add(i); }"),
        },
        Kernel {
            name: "dot",
            bytes_per_call: 2 * N * 4,
            note: "sum(x*y) reduction — Mercury vectorizes it; gcc/rustc keep it serial",
            mer: mer_kernel(&format!(
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
            note: "sum((x-y)^2), an L2-loss reduction (Mercury vectorizes it)",
            mer: mer_kernel(&format!(
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
            mer: mer_kernel(&format!(
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
            mer: mer_kernel(&format!(
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
        // Transcendentals: the regime a tensor compiler should dominate idiomatic scalar source.
        // Mercury lowers `exp` to a ~1-ULP f32 polynomial and auto-vectorizes it (128-bit x 4);
        // gcc/rustc call scalar libm `expf` per element and cannot vectorize a loop with a call
        // (no libmvec on this mingw toolchain), so the loop stays serial.
        Kernel {
            name: "exp",
            bytes_per_call: 2 * N * 4,
            note: "out = exp(x): Mercury vectorizes a ~1-ULP poly; C/Rust call scalar libm expf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = exp(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=expf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).exp(); }"),
        },
        // Natural log, the partner of exp (log-softmax / cross-entropy). The x buffer is all
        // positive (≥1), so this is a clean domain. Mercury vectorizes a ~1-ULP Cephes poly; C/Rust
        // call scalar libm logf.
        Kernel {
            name: "log",
            bytes_per_call: 2 * N * 4,
            note: "out = log(x): Mercury vectorizes a ~1-ULP poly; C/Rust call scalar libm logf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = log(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=logf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).ln(); }"),
        },
        // GELU (tanh approximation) written with the *identical* algorithm in all three languages —
        // tanh(z) built from exp as `1 - 2/(exp(2z)+1)`. The only difference is that Mercury
        // auto-vectorizes the exp; C/Rust call scalar expf. The fairest transcendental comparison.
        Kernel {
            name: "gelu",
            bytes_per_call: 2 * N * 4,
            note: "GELU (tanh approx): Mercury dispatches gelu() to a fused 256-bit AVX2 kernel; C/Rust scalar",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = gelu(x[i]); }}")),
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
        // SiLU / swish: x * sigmoid(x), the activation in Llama/modern transformers. sigmoid is one
        // intrinsic that vectorizes; C/Rust spell it 1/(1+exp(-x)) with a scalar libm expf.
        Kernel {
            name: "silu",
            bytes_per_call: 2 * N * 4,
            note: "silu/swish x*sigmoid(x): Mercury dispatches silu() to a fused 256-bit AVX2 kernel; C/Rust scalar",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = silu(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++){ float v=x[i]; out[i]=v/(1.0f+expf(-v)); }"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)=v/(1.0f32+(-v).exp()); }",
            ),
        },
        // softplus ln(1+e^x): Mercury dispatches to the vectorized exp+log; C/Rust call scalar libm.
        // x ∈ [1,9] here, so the idiomatic naive log(1+exp(x)) is overflow-safe and equals Mercury's
        // stable max(x,0)+log(1+exp(-|x|)) to f32 tolerance.
        Kernel {
            name: "softplus",
            bytes_per_call: 2 * N * 4,
            note: "softplus ln(1+e^x): Mercury dispatches to a 256-bit AVX2 exp+log kernel; C/Rust scalar",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = softplus(x[i]); }}")),
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
            note: "mish x*tanh(softplus(x)): Mercury dispatches to a 256-bit AVX2 kernel; C/Rust scalar",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = mish(x[i]); }}")),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; float sp=logf(1.0f+expf(v)); \
                 out[i]=v*(1.0f-2.0f/(expf(2.0f*sp)+1.0f)); }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let sp=(1.0f32+v.exp()).ln(); \
                 *out.add(i)=v*(1.0f32-2.0/((2.0*sp).exp()+1.0)); }",
            ),
        },
        // tanh activation, the *identical* exp-based algorithm in all three (Mercury vectorizes it).
        Kernel {
            name: "tanh",
            bytes_per_call: 2 * N * 4,
            note: "tanh via exp, same algorithm everywhere; Mercury vectorizes the exp",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = tanh(x[i]); }}")),
            c: c_kernel(
                "for(long i=0;i<N;i++){ float v=x[i]; out[i]=1.0f-2.0f/(expf(2.0f*v)+1.0f); }",
            ),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)=1.0f32-2.0/((2.0*v).exp()+1.0); }",
            ),
        },
        // Operator fusion: a linear map then ReLU, written as TWO loops in every language. Mercury's
        // compiler fuses them into one pass (intermediate stays in registers, not streamed to the
        // scratch array `y`); idiomatic C/Rust as-written make two passes over `y`.
        Kernel {
            name: "fused_linear_relu",
            bytes_per_call: 2 * N * 4,
            note: "linear→relu: Mercury auto-fuses 2 loops; C/Rust as-written stream the intermediate",
            mer: mer_kernel(&format!(
                "for i in 0..{N} {{ y[i] = 2.0 * x[i] + 1.0; }} \
                 for i in 0..{N} {{ out[i] = if y[i] > 0.0 {{ y[i] }} else {{ 0.0 }}; }}"
            )),
            c: c_kernel(
                "float* t=(float*)y; for(long i=0;i<N;i++) t[i]=2.0f*x[i]+1.0f; \
                 for(long i=0;i<N;i++){ float v=t[i]; out[i]= v>0.0f? v:0.0f; }",
            ),
            rust: rust_kernel(
                "let t = y as *mut f32; for i in 0..N { *t.add(i)=2.0* *x.add(i)+1.0; } \
                 for i in 0..N { let v= *t.add(i); *out.add(i)= if v>0.0 {v} else {0.0}; }",
            ),
        },
        // --- Mercury @parallel (multicore) vs idiomatic single-threaded C/Rust ---
        Kernel {
            name: "saxpy@parallel",
            bytes_per_call: 3 * N * 4,
            note: "Mercury auto-parallel across cores vs single-threaded C/Rust",
            mer: mer_par_kernel(&format!(
                "for i in 0..{N} {{ out[i] = 2.0 * x[i] + y[i]; }}"
            )),
            c: c_kernel("float a=2.0f; for(long i=0;i<N;i++) out[i]=a*x[i]+y[i];"),
            rust: rust_kernel("let a=2.0f32; for i in 0..N { *out.add(i)=a* *x.add(i)+ *y.add(i); }"),
        },
        Kernel {
            name: "poly@parallel",
            bytes_per_call: 2 * N * 4,
            note: "Mercury auto-parallel across cores vs single-threaded C/Rust",
            mer: mer_par_kernel(&format!(
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
            mer: mer_par_kernel(&format!(
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
            mer: mer_par_kernel(&format!("for i in 0..{N} {{ out[i] = gelu(x[i]); }}")),
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
        // deterministic reduction kernel (mercury_sreduce_f32_parallel), so this is one core's
        // memory bandwidth × all cores vs the single-threaded sequential C/Rust reduction (which
        // gcc/rustc keep latency-bound on the dependent add chain). The transformer-relevant case:
        // attention scores, L2 losses, LayerNorm sums over a large activation.
        Kernel {
            name: "dot@parallel",
            bytes_per_call: 2 * N * 4,
            note: "sum(x*y) across cores (multicore reduction kernel) vs single-threaded C/Rust",
            mer: mer_par_kernel(&format!(
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
            mer: mer_par_kernel(&format!(
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
            mer: mer_par_kernel(&format!(
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
            mer: mer_par_kernel(&format!(
                "let mut m: f32 = 0.0; for k in 0..{N} {{ m = fmax(m, abs(x[k])); }} out[0] = m;"
            )),
            c: c_kernel(
                "float m=0.0f; for(long i=0;i<N;i++){ float a=fabsf(x[i]); m = a>m? a:m; } out[0]=m;",
            ),
            rust: rust_kernel(
                "let mut m=0.0f32; for i in 0..N { let a=(*x.add(i)).abs(); if a>m { m=a; } } *out.add(0)=m;",
            ),
        },
    ]
}

/// Streaming elementwise at a **>L3** tensor size — the regime that matters for real activation
/// tensors (a `[batch, seq, hidden]` block is hundreds of MB, far larger than the 4 MiB kernels
/// above). At N=2²⁴ (64 MiB/array) the working set cannot stay in cache, so Mercury's recognized
/// streaming maps dispatch to `mercury_velem_f32` with **non-temporal** stores, skipping the
/// read-for-ownership traffic gcc/rustc pay on every cacheable store (a store path they will not emit
/// automatically). The lead is larger here than at 2²⁰, where the RFO is partly absorbed by L3.
fn bench_streaming_large(cc: &str, dir: &Path) {
    const NL: usize = 1 << 24; // 16,777,216 elements, 64 MiB per f32 array
    let x: Vec<f32> = (0..NL).map(|i| (i % 17) as f32 * 0.5 - 3.0).collect();
    let y: Vec<f32> = (0..NL).map(|i| (i % 13) as f32 * 0.25 - 0.5).collect();
    let mut out = vec![0.0f32; NL];
    let (xp, yp, op) = (x.as_ptr(), y.as_ptr(), out.as_mut_ptr());
    println!("=== streaming elementwise at N=2^24 (64 MiB/array, >L3; GB/s, higher is better) ===");
    println!(
        "  {:<10} {:>10} {:>10} {:>10} {:>14}",
        "", "Mercury", "C (gcc)", "Rust", "Mer vs C"
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
        let mer = bench_mercury(&mer_kernel_n(NL, mb), &mut out, xp, yp, op);
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
            op,
        );
        let rust = bench_external(
            "rs",
            &rust_kernel_n(NL, rb),
            dir,
            &format!("stream_{name}"),
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
            op,
        );
        let gbs = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", *bytes as f64 / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        let standing = if let (Some(m), Some(c)) = (&mer, &c) {
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
            gbs(&mer),
            gbs(&c),
            gbs(&rust),
            standing
        );
        if let (Some(m), Some(c)) = (&mer, &c) {
            let (rel, at) = max_rel_err(&m.out, &c.out);
            if rel > 1e-4 {
                println!(
                    "  ! {name} full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                    m.out[at], c.out[at], rel
                );
            }
        }
    }
    println!();
}

fn mer_kernel(body: &str) -> String {
    format!(
        "module bench\nfn kbench(x: [f32; {N}], y: [f32; {N}], out: [f32; {N}]) {{\n    {body}\n}}\n"
    )
}

/// A `@parallel` Mercury kernel (whole body is one `for` loop, so it parallelizes).
fn mer_par_kernel(loop_body: &str) -> String {
    format!(
        "module bench\n@parallel\nfn kbench(x: [f32; {N}], y: [f32; {N}], out: [f32; {N}]) {{\n    {loop_body}\n}}\n"
    )
}

fn c_kernel(body: &str) -> String {
    format!("#include <math.h>\n#define N {N}\n__declspec(dllexport) void kbench(const float* x, const float* y, float* out) {{\n  {body}\n}}\n")
}

fn rust_kernel(body: &str) -> String {
    // `#[allow(unused_variables)]`: some kernels (relu, poly) don't read `y`; the fixed `(x,y,out)`
    // ABI keeps the param, so silence the warning rather than clutter the benchmark output.
    format!("const N: usize = {N};\n#[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  {body}\n}}\n")
}

// Parameterized kernel builders (an explicit element count `n`) — used by the large-tensor streaming
// benchmark, which runs at N=2²⁴ rather than the module-global N=2²⁰.
fn mer_kernel_n(n: usize, body: &str) -> String {
    format!("module bench\nfn kbench(x: [f32; {n}], y: [f32; {n}], out: [f32; {n}]) {{\n    {body}\n}}\n")
}
fn c_kernel_n(n: usize, body: &str) -> String {
    format!("#include <math.h>\n#define N {n}\n__declspec(dllexport) void kbench(const float* x, const float* y, float* out) {{\n  {body}\n}}\n")
}
fn rust_kernel_n(n: usize, body: &str) -> String {
    format!("const N: usize = {n};\n#[no_mangle]\n#[allow(unused_variables)]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  {body}\n}}\n")
}
