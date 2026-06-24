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

    // Optional substring filter (first CLI arg): run only the kernels/sections whose name contains it,
    // for fast single-kernel iteration. No arg → the full suite, byte-for-byte as before.
    let filter = std::env::args().nth(1);
    let want = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    let kernels = kernels();
    let mut runtime_ratios_c = Vec::new();
    let mut compile_ratios_c = Vec::new();

    for k in &kernels {
        if !want(k.name) {
            continue;
        }
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
    if want("matmul") {
        bench_matmul(&cc, &dir, roof);
    }
    if want("linear") {
        bench_linear(&cc, &dir, roof);
    }
    if want("linear_bf16") {
        bench_linear_bf16(&cc, &dir);
    }
    if want("matmul_tn") {
        bench_matmul_tn(&cc, &dir, roof);
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
    if want("colmax") {
        bench_colmax(&cc, &dir);
    }
    if want("colstat") {
        bench_colstat(&cc, &dir);
    }
    if want("softmax_bwd") {
        bench_softmax_bwd(&cc, &dir);
    }
    if want("act_backward") {
        bench_act_backward(&cc, &dir);
    }
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

/// `C = Aᵀ·B` — the **weight-gradient** GEMM of a training backward pass (`dW = dYᵀ·X`). A is stored
/// `[K, M]` (the contraction/batch axis is the OUTER index of A's storage), so its logical operand is
/// the transpose of its layout. Mercury recognizes `a[k*M+i]*b[k*N+j]` and dispatches to
/// `mercury_sgemm_tn` (transpose A once, then the tuned NN kernel); idiomatic C/Rust compile the
/// column-strided A reads (one cache line per element) as a near-scalar k-loop gcc cannot vectorize —
/// the regime the domain lowering should dominate hardest. Square M=K=N for the shared-buffer ABI.
fn bench_matmul_tn(cc: &str, dir: &Path, roof: f64) {
    let _ = roof;
    for ns in [256usize, 512, 1024] {
        let n2 = ns * ns;
        let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
        let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
        let mut c = vec![0.0f32; n2];
        let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        println!(
            "=== matmul_tn {ns}x{ns} (C=Aᵀ·B, the dW weight-gradient; GFLOP/s, higher is better) ==="
        );
        let gflops = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        let mer = bench_mercury(&mer_matmul_tn(ns, false), &mut c, ap, bp, cp);
        let mer_par = bench_mercury(&mer_matmul_tn(ns, true), &mut c, ap, bp, cp);
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
            cp,
        );
        let rm = bench_external(
            "rs",
            &rust_matmul_tn(ns),
            dir,
            "matmul_tn",
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
            "GFLOP/s",
            gflops(&mer),
            gflops(&mer_par),
            gflops(&cm),
            gflops(&rm)
        );
        // Cross-language correctness: the transpose-once-then-NN result must match the naive nest.
        if let (Some(m), Some(c)) = (&mer, &cm) {
            let (rel, at) = max_rel_err(&m.out, &c.out);
            if rel > 1e-3 {
                println!(
                    "  ! full-buffer mismatch vs C at [{at}]: Mercury={} C={} (rel {:.2e})",
                    m.out[at], c.out[at], rel
                );
            }
        }
        if let (Some(ms), Some(c)) = (&mer, &cm) {
            let r = c.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Mercury single-core (SIMD) is {:.2}x {} than C single-threaded",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c)) = (&mer_par, &cm) {
            let r = c.ns_per_call / mp.ns_per_call;
            println!(
                "  -> Mercury @parallel is {:.2}x {} than idiomatic single-threaded C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        println!();
    }
}

/// `ijk` dot-product `C = Aᵀ·B`: A indexed `a[k*NS+i]` (transposed — k is A's outer index), B
/// `b[k*NS+j]` (normal). Mercury folds this to `mercury_sgemm_tn[_parallel]`. Optionally `@parallel`.
fn mer_matmul_tn(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [f32; {n2}], b: [f32; {n2}], c: [f32; {n2}]) {{\n\
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

fn c_matmul_tn(ns: usize) -> String {
    format!(
        "#define NS {ns}\n__declspec(dllexport) void kbench(const float* a, const float* b, float* c) {{\n\
         \x20 for (long i=0;i<NS;i++){{\n\
         \x20   for (long j=0;j<NS;j++){{\n\
         \x20     float s=0.0f;\n\
         \x20     for (long k=0;k<NS;k++) s += a[k*NS+i]*b[k*NS+j];\n\
         \x20     c[i*NS+j]=s;\n\
         \x20   }}\n\
         \x20 }}\n}}\n"
    )
}

fn rust_matmul_tn(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(a:*const f32, b:*const f32, c:*mut f32) {{\n\
         \x20 for i in 0..NS {{\n\
         \x20   for j in 0..NS {{\n\
         \x20     let mut s=0.0f32;\n\
         \x20     for k in 0..NS {{ s += *a.add(k*NS+i) * *b.add(k*NS+j); }}\n\
         \x20     *c.add(i*NS+j)=s;\n\
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
/// bf16 mixed-precision `nn.Linear` (`C = A·Bᵀ`, bf16 inputs, f32 accumulate) — the standard
/// transformer matmul. Mercury recognizes the half-precision dot-product nest and folds it to one
/// `mercury_sgemm_bf16_nt[_parallel]` call (a lossless widen prepass + the tuned AVX2 f32 GEMM); the
/// idiomatic C/Rust store bf16 as `uint16_t` and widen each element inline inside the triple loop —
/// which they can't vectorize, and the sequential float reduction stays scalar (no `-ffast-math`),
/// the same basis as the f32 `linear`/`dot` kernels. Reuses the `(u16, u16, f32)` bench ABI. The
/// `out[0]` (= `c[0]`) cross-check is a sanity guard; the bit-exact correctness rests on the runtime
/// twin test + the interp/native differential gate. The f16 twin (`mercury_sgemm_f16_nt`) is the same
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
        let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
        let flops = 2.0 * (ns as f64).powi(3);
        let gflops = |m: &Option<MeasureBf16>| {
            m.as_ref()
                .map(|x| format!("{:.1}", flops / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "=== linear_bf16 (bf16 nn.Linear C=A·Bᵀ, f32 accumulate) {ns}x{ns} (GFLOP/s, higher is better) ==="
        );
        let mer = bench_mercury_bf16(&mer_linear_bf16(ns, false), &mut c, ap, bp, cp);
        let mer_par = bench_mercury_bf16(&mer_linear_bf16(ns, true), &mut c, ap, bp, cp);
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
            cp,
        );
        let rm = bench_external_bf16(
            "rs",
            &rust_linear_bf16(ns),
            dir,
            "linear_bf16",
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
            "GFLOP/s",
            gflops(&mer),
            gflops(&mer_par),
            gflops(&cm),
            gflops(&rm)
        );
        // Sanity cross-check on c[0] (the bf16 widen is lossless, so all three compute the same GEMM
        // up to the documented float-reassociation tolerance).
        if let (Some(m), Some(c2)) = (&mer, &cm) {
            let rel = ((m.out - c2.out).abs() / c2.out.abs().max(1e-6)) as f64;
            if rel > 1e-2 {
                println!(
                    "  ! c[0] mismatch vs C: Mer={} C={} (rel {:.2e})",
                    m.out, c2.out, rel
                );
            }
        }
        if let (Some(ms), Some(c2)) = (&mer, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Mercury single-core is {:.2}x {} than idiomatic bf16 C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            println!("  -> Mercury @parallel is {r:.2}x faster than idiomatic single-threaded bf16 C");
        }
        println!();
    }
}

/// Mercury bf16 `nn.Linear`, the idiomatic `ijk` dot-product `C = A·Bᵀ` with `[bf16]` operands widened
/// `as f32` and an f32 accumulator — what the `mir_build` recognizer folds to `mercury_sgemm_bf16_nt`.
fn mer_linear_bf16(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(a: [bf16; {n2}], b: [bf16; {n2}], c: [f32; {n2}]) {{\n\
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
         __declspec(dllexport) void kbench(const uint16_t* a, const uint16_t* b, float* c){{\n\
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
/// layout conversions). Mercury folds the `dst[j*R+i] = src[i*C+j]` nest to the cache-blocked
/// `mercury_transpose_f32`; C/Rust are the idiomatic naive transpose at `-O3 -march=native`. The naive
/// transpose writes `dst` with stride `R` — a fresh cache line per element once `R` is large — while
/// the blocked kernel keeps a `B×B` tile L1-resident; gcc/rustc do not loop-tile a transpose at `-O3`.
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
        let (sp, yp, dp) = (src.as_ptr(), dummy.as_ptr(), dst.as_mut_ptr());
        let bytes = 2.0 * n2 as f64 * 4.0; // read src + write dst
        let gbps = |m: &Option<Measure>| {
            m.as_ref()
                .map(|x| format!("{:.1}", bytes / x.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== transpose (dst = srcᵀ) {ns}x{ns} (GB/s, higher is better) ===");
        let mer = bench_mercury(&mer_transpose(ns, false), &mut dst, sp, yp, dp);
        let mer_par = bench_mercury(&mer_transpose(ns, true), &mut dst, sp, yp, dp);
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
            dp,
        );
        let rm = bench_external(
            "rs",
            &rust_transpose(ns),
            dir,
            "transpose",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut dst,
            sp,
            yp,
            dp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&mer),
            gbps(&mer_par),
            gbps(&cm),
            gbps(&rm)
        );
        // Transpose is a permutation — exact, so the full-buffer cross-check is bit equality.
        if let (Some(m), Some(c2)) = (&mer, &cm) {
            if m.out != c2.out {
                println!("  ! transpose output mismatch vs C");
            }
        }
        if let (Some(ms), Some(c2)) = (&mer, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Mercury single-core is {:.2}x {} than naive C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            println!("  -> Mercury @parallel is {r:.2}x naive single-threaded C");
        }
        println!();
    }
}

/// Mercury transpose kernel: the idiomatic `dst[j*R+i] = src[i*C+j]` nest the `mir_build` recognizer
/// folds to one `mercury_transpose_f32[_parallel]` call. `y` is an unused param so the signature
/// matches the `(src, _, dst)` 3-pointer harness ABI.
fn mer_transpose(ns: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n2 = ns * ns;
    format!(
        "module bench\n{attr}fn kbench(src: [f32; {n2}], y: [f32; {n2}], dst: [f32; {n2}]) {{\n\
         \x20   for i in 0..{ns} {{\n\
         \x20       for j in 0..{ns} {{ dst[j * {ns} + i] = src[i * {ns} + j]; }}\n\
         \x20   }}\n}}\n"
    )
}

fn c_transpose(ns: usize) -> String {
    format!(
        "#define NS {ns}\n\
         __declspec(dllexport) void kbench(const float* src, const float* y, float* dst){{\n\
         \x20 (void)y;\n\
         \x20 for (long i=0;i<NS;i++)\n\
         \x20   for (long j=0;j<NS;j++) dst[j*NS+i] = src[i*NS+j];\n}}\n"
    )
}

fn rust_transpose(ns: usize) -> String {
    format!(
        "const NS: usize = {ns};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(src:*const f32, _y:*const f32, dst:*mut f32) {{\n\
         \x20 for i in 0..NS {{ for j in 0..NS {{ *dst.add(j*NS+i) = *src.add(i*NS+j); }} }}\n}}\n"
    )
}

/// Column reduction `out[j] = Σ_i x[i, j]` — the sum over the outer (batch/row) axis (the bias gradient
/// `db = Σ_batch dY`, batch sum, reduce-along-axis-0). The naive `for j { for i { s += x[i*N+j] } }`
/// reads `x` with stride `N` — a strided reduction gcc/rustc leave **scalar** (verified: no packed
/// `vaddps` at `-O3 -march=native`). Mercury folds the nest to `mercury_colsum_f32`, which streams `x`
/// row-major + 8 columns at a time. The kernels carry an unused middle pointer so they share the
/// `(x, _, out)` 3-pointer harness. Reported as GB/s (`M·N·4` bytes — the matrix read once). Both
/// languages sum each column i-ascending, so the cross-check is **bit-exact** (no reassociation).
fn bench_colsum(cc: &str, dir: &Path) {
    for (m, n) in [(1024usize, 1024usize), (4096, 1024)] {
        let mn = m * n;
        let x: Vec<f32> = (0..mn).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
        let dummy = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let (xp, yp, op) = (x.as_ptr(), dummy.as_ptr(), out.as_mut_ptr());
        let bytes = mn as f64 * 4.0; // the matrix is read once
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== colsum (out[j] = Σ_i x[i,j]) {m}x{n} (GB/s, higher is better) ===");
        let mer = bench_mercury(&mer_colsum(m, n, false), &mut out, xp, yp, op);
        let mer_par = bench_mercury(&mer_colsum(m, n, true), &mut out, xp, yp, op);
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
            op,
        );
        let rm = bench_external(
            "rs",
            &rust_colsum(m, n),
            dir,
            "colsum",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut out,
            xp,
            yp,
            op,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&mer),
            gbps(&mer_par),
            gbps(&cm),
            gbps(&rm)
        );
        // Both sum each column in i-ascending order, so the full-buffer cross-check is bit equality.
        if let (Some(a), Some(c2)) = (&mer, &cm) {
            if a.out != c2.out {
                println!("  ! colsum output mismatch vs C");
            }
        }
        if let (Some(ms), Some(c2)) = (&mer, &cm) {
            let r = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Mercury single-core is {:.2}x {} than naive C",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
            let r = c2.ns_per_call / mp.ns_per_call;
            println!("  -> Mercury @parallel is {r:.2}x naive single-threaded C");
        }
        println!();
    }
}

/// Mercury column-sum kernel: the idiomatic `for j { let s=0; for i { s += x[i*N+j] }; out[j]=s }` the
/// `mir_build` recognizer folds to one `mercury_colsum_f32[_parallel]` call. `y` is unused (the `(x, _,
/// out)` 3-pointer harness ABI). `out` is the `[N]` result; `x` is the `[M, N]` matrix.
fn mer_colsum(m: usize, n: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mn = m * n;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {mn}], y: [f32; {n}], out: [f32; {n}]) {{\n\
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
         __declspec(dllexport) void kbench(const float* x, const float* y, float* out){{\n\
         \x20 (void)y;\n\
         \x20 for (long j=0;j<N;j++){{ float s=0.0f; for (long i=0;i<M;i++) s += x[i*N+j]; out[j]=s; }}\n}}\n"
    )
}

fn rust_colsum(m: usize, n: usize) -> String {
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 for j in 0..N {{ let mut s=0.0f32; for i in 0..M {{ s += *x.add(i*N+j); }} *out.add(j)=s; }}\n}}\n"
    )
}

/// Column max / min / **abs-max** `out[j] = max/min_i x[i,j]` (and `max_i |x[i,j]|`, the per-channel
/// symmetric-quant scale) — per-channel statistics / axis-0 pooling, the siblings of `colsum`. The naive
/// `for j { let s=x[j]; for i { s = max(s, x[i*N+j]) } }` strides `x` by `N` and — verified — gcc/rustc
/// leave all three **scalar** (no packed `vmaxps`/`vminps` at `-O3 -march=native`: `fmax`/`fmin` are
/// non-associative so they will not reassociate the strided fold). Mercury folds them to
/// `mercury_col{max,min,maxabs}_f32`, streaming `x` row-major + 8 columns at a time. Reported as GB/s
/// (`M·N·4`, the matrix read once); the kernels carry an unused middle pointer to share the `(x, _, out)`
/// 3-pointer harness. Both fold each column i-ascending (`s ⊕ v` mirrors `_mm256_{max,min}_ps`, abs via
/// sign-mask == `fabsf`), so the cross-check is **bit-exact** on finite data.
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
            let (xp, yp, op) = (x.as_ptr(), dummy.as_ptr(), out.as_mut_ptr());
            let bytes = mn as f64 * 4.0; // the matrix is read once
            let gbps = |v: &Option<Measure>| {
                v.as_ref()
                    .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                    .unwrap_or_else(|| "n/a".into())
            };
            let desc = if opc == 2 { "|x[i,j]|" } else { "x[i,j]" };
            println!("=== {label} (out[j] = {sym}_i {desc}) {m}x{n} (GB/s, higher is better) ===");
            let mer = bench_mercury(&mer_colmax(m, n, false, opc), &mut out, xp, yp, op);
            let mer_par = bench_mercury(&mer_colmax(m, n, true, opc), &mut out, xp, yp, op);
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
                op,
            );
            let rm = bench_external(
                "rs",
                &rust_colmax(m, n, opc),
                dir,
                label,
                "rustc",
                &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
                op,
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&mer),
                gbps(&mer_par),
                gbps(&cm),
                gbps(&rm)
            );
            // Both fold each column i-ascending, so the full-buffer cross-check is bit equality.
            if let (Some(a), Some(c2)) = (&mer, &cm) {
                if a.out != c2.out {
                    println!("  ! {label} output mismatch vs C");
                }
            }
            if let (Some(ms), Some(c2)) = (&mer, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Mercury single-core is {:.2}x {} than naive C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                println!("  -> Mercury @parallel is {r:.2}x naive single-threaded C");
            }
            println!();
        }
    }
}

/// Mercury column max/min/abs-max kernel: `for j { let s=⟨x[j]⟩; for i in 1..M { s = fmax/fmin(s, ⟨x[i*N+j]⟩) }; out[j]=s }`
/// the recognizer folds to one `mercury_col{max,min,maxabs}_f32[_parallel]` call (`⟨·⟩` = `abs(·)` for
/// op 2). `y` is unused (the 3-pointer harness). op: 0=max, 1=min, 2=abs-max.
fn mer_colmax(m: usize, n: usize, parallel: bool, op: u8) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let f = if op == 1 { "fmin" } else { "fmax" };
    let (lo, hi) = if op == 2 { ("abs(", ")") } else { ("", "") };
    let mn = m * n;
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {mn}], y: [f32; {n}], out: [f32; {n}]) {{\n\
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
         __declspec(dllexport) void kbench(const float* x, const float* y, float* out){{\n\
         \x20 (void)y;\n\
         \x20 for (long j=0;j<N;j++){{ float s={lo}x[j]{hi}; for (long i=1;i<M;i++){{ float v={lo}x[i*N+j]{hi}; s = s{cmp}v?s:v; }} out[j]=s; }}\n}}\n"
    )
}

fn rust_colmax(m: usize, n: usize, op: u8) -> String {
    let cmp = if op == 1 { "<" } else { ">" };
    let (lo, hi) = if op == 2 { ("(", ").abs()") } else { ("", "") };
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 for j in 0..N {{ let mut s={lo}*x.add(j){hi}; for i in 1..M {{ let v={lo}*x.add(i*N+j){hi}; s = if s{cmp}v {{s}} else {{v}}; }} *out.add(j)=s; }}\n}}\n"
    )
}

/// Column **statistics** `out[j] = mean/sumsq/L2/RMS_i x[i,j]` — the per-channel BatchNorm mean, 2nd
/// moment / energy, column L2 norm, and per-feature RMS. Same strided `Σ`/`Σx²` column-outer fold as
/// `colsum` that gcc/rustc leave **scalar** (verified: no packed `vaddps` for the stride-N reduction);
/// Mercury folds each nest to `mercury_col{mean,sumsq,l2,rms}_f32[_parallel]` (row-major streaming + a
/// per-column finalize). Reported as GB/s (`M·N·4`, the matrix read once), reusing the `(x, _, out)`
/// 3-pointer harness via the unused middle. Each folds its column i-ascending — exactly the naive C
/// order — and `/M`/`sqrt` are correctly-rounded, so the full-buffer cross-check is **bit-exact**.
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
            let (xp, yp, op) = (x.as_ptr(), dummy.as_ptr(), out.as_mut_ptr());
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
            let mer = bench_mercury(&mer_colstat(m, n, false, opc), &mut out, xp, yp, op);
            let mer_par = bench_mercury(&mer_colstat(m, n, true, opc), &mut out, xp, yp, op);
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
                op,
            );
            let rm = bench_external(
                "rs",
                &rust_colstat(m, n, opc),
                dir,
                label,
                "rustc",
                &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
                op,
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&mer),
                gbps(&mer_par),
                gbps(&cm),
                gbps(&rm)
            );
            // Both fold each column i-ascending and finalize with correctly-rounded /M and sqrt, so the
            // full-buffer cross-check is bit equality.
            if let (Some(a), Some(c2)) = (&mer, &cm) {
                if a.out != c2.out {
                    println!("  ! {label} output mismatch vs C");
                }
            }
            if let (Some(ms), Some(c2)) = (&mer, &cm) {
                let r = c2.ns_per_call / ms.ns_per_call;
                println!(
                    "  -> Mercury single-core is {:.2}x {} than naive C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
                let r = c2.ns_per_call / mp.ns_per_call;
                println!("  -> Mercury @parallel is {r:.2}x naive single-threaded C");
            }
            println!();
        }
    }
}

/// Mercury column-statistics kernel: `for j { let s=0; for i in 0..M { s = s + ⟨x[i*N+j]⟩ }; out[j]=fin(s) }`
/// the recognizer folds to one `mercury_col{mean,sumsq,l2,rms}_f32[_parallel]` call. op: 0=mean
/// (`s/M`), 1=sumsq (`s` over `x²`), 2=L2 (`sqrt(s)` over `x²`), 3=RMS (`sqrt(s/M)` over `x²`).
fn mer_colstat(m: usize, n: usize, parallel: bool, op: u8) -> String {
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
        "module bench\n{attr}fn kbench(x: [f32; {mn}], y: [f32; {n}], out: [f32; {n}]) {{\n\
         \x20   for j in 0..{n} {{\n\
         \x20       let mut s: f32 = 0.0;\n\
         \x20       for i in 0..{m} {{ {fold} }}\n\
         \x20       out[j] = {fin};\n\
         \x20   }}\n}}\n"
    )
}

fn c_colstat(m: usize, n: usize, op: u8) -> String {
    let (fold, fin) = match op {
        0 => ("s += x[i*N+j];", "s / (float)M"),
        1 => ("s += x[i*N+j]*x[i*N+j];", "s"),
        2 => ("s += x[i*N+j]*x[i*N+j];", "sqrtf(s)"),
        _ => ("s += x[i*N+j]*x[i*N+j];", "sqrtf(s / (float)M)"),
    };
    format!(
        "#include <math.h>\n#define M {m}\n#define N {n}\n\
         __declspec(dllexport) void kbench(const float* x, const float* y, float* out){{\n\
         \x20 (void)y;\n\
         \x20 for (long j=0;j<N;j++){{ float s=0.0f; for (long i=0;i<M;i++){{ {fold} }} out[j]={fin}; }}\n}}\n"
    )
}

fn rust_colstat(m: usize, n: usize, op: u8) -> String {
    let (fold, fin) = match op {
        0 => ("s += *x.add(i*N+j);", "s / M as f32"),
        1 => ("s += *x.add(i*N+j) * *x.add(i*N+j);", "s"),
        2 => ("s += *x.add(i*N+j) * *x.add(i*N+j);", "s.sqrt()"),
        _ => ("s += *x.add(i*N+j) * *x.add(i*N+j);", "(s / M as f32).sqrt()"),
    };
    format!(
        "const M: usize = {m};\nconst N: usize = {n};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, _y:*const f32, out:*mut f32) {{\n\
         \x20 for j in 0..N {{ let mut s=0.0f32; for i in 0..M {{ {fold} }} *out.add(j)={fin}; }}\n}}\n"
    )
}

/// Softmax backward `dx[r,i] = y[r,i]·(dy[r,i] − Σ_j y[r,j]·dy[r,j])` — the attention/classification
/// training gradient. Each row's dot `Σ y·dy` is a reduction gcc/rustc keep **scalar** (verified: they
/// load/multiply `y·dy` wide but the accumulation is a serial `vaddss` chain — no `ymm` accumulator,
/// the float sum is not reassociated), then a (vectorized) elementwise `y·(dy − s)`. Mercury folds the
/// `[R,C]` nest to one `mercury_softmax_bwd_f32[_parallel]` call: the dot uses 8 independent lane
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
        let (yp, dyp, dxp) = (y.as_ptr(), dy.as_ptr(), dx.as_mut_ptr());
        let bytes = 3.0 * n as f64 * 4.0; // y read + dy read + dx write
        let gbps = |v: &Option<Measure>| {
            v.as_ref()
                .map(|m| format!("{:.1}", bytes / m.ns_per_call))
                .unwrap_or_else(|| "n/a".into())
        };
        println!("=== softmax_bwd (dx = y·(dy − Σ y·dy)) {r}x{c} (GB/s, higher is better) ===");
        let mer = bench_mercury(&mer_softmax_bwd(r, c, false), &mut dx, yp, dyp, dxp);
        let mer_par = bench_mercury(&mer_softmax_bwd(r, c, true), &mut dx, yp, dyp, dxp);
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
            dxp,
        );
        let rm = bench_external(
            "rs",
            &rust_softmax_bwd(r, c),
            dir,
            "softmax_bwd",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut dx,
            yp,
            dyp,
            dxp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&mer),
            gbps(&mer_par),
            gbps(&cm),
            gbps(&rm)
        );
        // The dot reassociates (lane accumulators vs C's serial chain), so check a tolerance. Softmax
        // backward sums to ~0 per row, so individual dx are catastrophic-cancellation zeros where a
        // *per-element* relative error is meaningless — normalize the max abs error by the max output
        // magnitude instead (the honest "is the whole vector close" metric).
        if let (Some(m), Some(c2)) = (&mer, &cm) {
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
        if let (Some(ms), Some(c2)) = (&mer, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Mercury single-core is {:.2}x {} than naive C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            println!("  -> Mercury @parallel is {r2:.2}x naive single-threaded C");
        }
        println!();
    }
}

/// Mercury softmax-backward: the batched nest the recognizer folds to one `mercury_softmax_bwd_f32
/// [_parallel]` call. The harness's `(x, y, out)` pointers carry `(y, dy, dx)`.
fn mer_softmax_bwd(rows: usize, cols: usize, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let n = rows * cols;
    format!(
        "module bench\n{attr}fn kbench(y: [f32; {n}], dy: [f32; {n}], dx: [f32; {n}]) {{\n\
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
         __declspec(dllexport) void kbench(const float* y, const float* dy, float* dx){{\n\
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

/// Activation **backward** `dx[i] = dy[i]·act'(x[i])` (silu/gelu — the SiLU/GELU training gradient).
/// Compute-bound: the derivative folds a sigmoid/tanh (an `expf`) that C/Rust call as scalar libm
/// inside the loop, so they cannot vectorize it. Mercury folds the elementwise
/// `dx[i]=act_backward(x[i],dy[i])` loop to one **256-bit** `mercury_vmath2_f32` call; the `@parallel`
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
    let (xp, dyp, dxp) = (x.as_ptr(), dy.as_ptr(), dx.as_mut_ptr());
    let bytes = 3.0 * n as f64 * 4.0; // x read + dy read + dx write
    let gbps = |v: &Option<Measure>| {
        v.as_ref()
            .map(|m| format!("{:.1}", bytes / m.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };
    for op in ["silu", "gelu", "sigmoid", "tanh", "elu", "softplus"] {
        println!("=== {op}_backward (dx = dy·{op}'(x)) N={n} (GB/s, higher is better) ===");
        let mer = bench_mercury(&mer_act_backward(n, op, false), &mut dx, xp, dyp, dxp);
        let mer_par = bench_mercury(&mer_act_backward(n, op, true), &mut dx, xp, dyp, dxp);
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
            dxp,
        );
        let rm = bench_external(
            "rs",
            &rust_act_backward(n, op),
            dir,
            "act_backward",
            "rustc",
            &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
            &mut dx,
            xp,
            dyp,
            dxp,
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "", "Mer(1core)", "Mer(par)", "C (gcc)", "Rust"
        );
        println!(
            "  {:<10} {:>11} {:>11} {:>11} {:>11}",
            "GB/s",
            gbps(&mer),
            gbps(&mer_par),
            gbps(&cm),
            gbps(&rm)
        );
        // Same magnitude-normalized check as softmax_bwd: the poly sigmoid/tanh differs from libm by
        // ~1 ULP, and act'(x) has zeros where a per-element relative error is meaningless.
        if let (Some(m), Some(c2)) = (&mer, &cm) {
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
        if let (Some(ms), Some(c2)) = (&mer, &cm) {
            let r2 = c2.ns_per_call / ms.ns_per_call;
            println!(
                "  -> Mercury single-core (256-bit) is {:.2}x {} than scalar C",
                if r2 >= 1.0 { r2 } else { 1.0 / r2 },
                if r2 >= 1.0 { "faster" } else { "slower" }
            );
        }
        if let (Some(mp), Some(c2)) = (&mer_par, &cm) {
            let r2 = c2.ns_per_call / mp.ns_per_call;
            println!("  -> Mercury @parallel is {r2:.2}x scalar single-threaded C");
        }
        println!();
    }
}

/// Mercury activation backward: the elementwise loop the recognizer folds to one
/// `mercury_vmath2_f32(x, dy, dx, n, VM2_*_BWD)` call. The harness's `(x, y, out)` carry `(x, dy, dx)`.
fn mer_act_backward(n: usize, op: &str, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    format!(
        "module bench\n{attr}fn kbench(x: [f32; {n}], dy: [f32; {n}], dx: [f32; {n}]) {{\n\
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
         __declspec(dllexport) void kbench(const float* x, const float* dy, float* dx){{\n\
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

/// Round an `f32` to bf16 (round-to-nearest-even), returning the 16 stored bits — the same arithmetic
/// `mercury_runtime::f32_to_bf16_bits` uses, so the benchmark data matches what the compiler stores.
/// (Benchmark inputs are finite, so the NaN case the runtime handles is irrelevant here.)
fn to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let bias = 0x0000_7fff + ((b >> 16) & 1);
    ((b + bias) >> 16) as u16
}

/// bf16 **mixed-precision reductions** (bf16 storage, f32 accumulate — the standard ML contract):
/// dot `Σ x·y` and unary sum `Σ x` over `[bf16; N]` arrays. Mercury folds the loop to one
/// `mercury_dot_bf16` / `mercury_sum_bf16` SIMD kernel (F16C-class widen + 8-lane f32 accumulate);
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
        let (xp, yp, op) = (x.as_ptr(), y.as_ptr(), o.as_mut_ptr());
        println!("=== bf16 reductions (bf16 in, f32 accumulate) N=2^{nbits} (GB/s, higher is better) ===");
        for (kind, is_dot) in [("dot Σx·y", true), ("sum Σx", false)] {
            let streams = if is_dot { 2 } else { 1 };
            let bytes = (streams * n * 2) as f64; // bf16 input traffic
            let gbps = |m: &Option<MeasureBf16>| {
                m.as_ref()
                    .map(|x| format!("{:.1}", bytes / x.ns_per_call)) // bytes/ns == GB/s
                    .unwrap_or_else(|| "n/a".into())
            };
            let mer = bench_mercury_bf16(&mer_bf16(n, is_dot), &mut o, xp, yp, op);
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
                op,
            );
            let rm = bench_external_bf16(
                "rs",
                &rust_bf16(n, is_dot),
                dir,
                "bf16",
                "rustc",
                &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut o,
                xp,
                yp,
                op,
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11}",
                kind, "Mercury", "C (gcc)", "Rust"
            );
            println!(
                "  {:<10} {:>11} {:>11} {:>11}",
                "GB/s",
                gbps(&mer),
                gbps(&cm),
                gbps(&rm)
            );
            // All-positive, well-conditioned reduction → a tight relative tolerance is the right
            // cross-language bar (the three reassociate the f32 sum differently: Mercury 8-lane SIMD
            // vs sequential C/Rust). Report the scalars so any drift is visible.
            if let Some(m) = &mer {
                for (lang, other) in [("C", &cm), ("Rust", &rm)] {
                    if let Some(o2) = other {
                        let (a, b) = (m.out, o2.out);
                        let rel = (a - b).abs() as f64 / (a.abs().max(b.abs()).max(1e-6) as f64);
                        if rel > 1e-3 {
                            println!("  ! {lang} scalar drift: {b} vs Mercury {a} (rel {rel:.2e})");
                        }
                    }
                }
            }
            if let (Some(m), Some(c2)) = (&mer, &cm) {
                let r = c2.ns_per_call / m.ns_per_call;
                println!(
                    "  -> Mercury is {:.2}x {} than idiomatic single-threaded C",
                    if r >= 1.0 { r } else { 1.0 / r },
                    if r >= 1.0 { "faster" } else { "slower" }
                );
            }
            // Compile time (Mercury front-end + JIT vs gcc/rustc to a shared lib).
            let cms = |m: &Option<MeasureBf16>| {
                m.as_ref()
                    .map(|x| format!("{:.0}", x.compile.as_secs_f64() * 1e3))
                    .unwrap_or_else(|| "n/a".into())
            };
            println!(
                "  {:<10} {:>11} {:>11} {:>11}",
                "compile ms",
                cms(&mer),
                cms(&cm),
                cms(&rm)
            );
        }
        println!();
    }
}

/// Mercury bf16 reduction: `s += (x[k] as f32) [* (y[k] as f32)]` — exactly what the `mir_build`
/// recognizer folds to one `mercury_dot_bf16` / `mercury_sum_bf16` call (bf16 storage, f32 accumulate).
fn mer_bf16(n: usize, is_dot: bool) -> String {
    let term = if is_dot {
        "(x[k] as f32) * (y[k] as f32)"
    } else {
        "(x[k] as f32)"
    };
    format!(
        "module bench\nfn kbench(x: [bf16; {n}], y: [bf16; {n}], o: [f32; 1]) {{\n\
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
         __declspec(dllexport) void kbench(const uint16_t* x, const uint16_t* y, float* o){{\n\
         \x20   float s=0.0f;\n\
         \x20   for (long k=0;k<N;k++) s += {term};\n\
         \x20   o[0]=s;\n}}\n"
    )
}

/// Idiomatic Rust bf16 reduction (same `<<16` widen; rustc `-O -Ctarget-cpu=native`, no contraction).
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
            "logsoftmax",
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
/// feature row was just one token. Mercury folds the `for r { <norm over out[r*C+i]> }` loop into one
/// `mercury_norm_f32` call (each row a fused single pass); under `@parallel` the independent rows map across cores via
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
            "=== batched norms (softmax / LayerNorm / RMSNorm), {rows}x{cols} = [tokens, hidden], {mb:.1} MB/buffer (ns/call, lower is better; Mercury → mercury_norm_f32[_parallel]) ==="
        );
        println!(
            "  {:<18} {:>12} {:>12} {:>12} {:>16}",
            "", "Mercury", "C (gcc)", "Rust", "Mer vs C"
        );
        for op in ["rmsnorm", "layernorm", "softmax"] {
            // C/Rust baselines are single-threaded per-row norms — the same for both Mercury rows
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
                op_,
            );
            let rust = bench_external(
                "rs",
                &rust_norm_batched(rows, cols, op),
                dir,
                "bnorm",
                "rustc",
                &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
                &mut out,
                xp,
                yp,
                op_,
            );
            for (suffix, par) in [("", false), ("@parallel", true)] {
                let label = format!("{op}{suffix}");
                let mer = bench_mercury(
                    &mer_norm_batched(rows, cols, op, par),
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
                    "  {:<18} {:>12} {:>12} {:>12} {:>16}",
                    label,
                    ns(&mer),
                    ns(&c),
                    ns(&rust),
                    standing
                );
                // Full-buffer cross-check (f32 tolerance: Mercury reassociates the per-row reductions).
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
        }
        println!();
    }
}

/// Mercury batched-norm source for `op` ∈ {rmsnorm, layernorm, softmax}: copy `x`→`out`, then the
/// `for r { <op over out[r*C+i]> }` form the `mir_build` recognizer folds to one `mercury_norm_f32(out,
/// out, R, C, eps, op)` call — or, under `@parallel`, `mercury_norm_f32_parallel` (rows across cores).
/// The `r*{cols}+i` offset, `/{cols}.0` divisor, and softmax's `out[r*C]` row-local max-seed are exactly
/// the spellings `match_batched_norm` accepts (all three norms share the kernel + dispatch path).
fn mer_norm_batched(rows: usize, cols: usize, op: &str, parallel: bool) -> String {
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
        "module bench\n{attr}fn kbench(x: [f32; {n}], y: [f32; {n}], out: [f32; {n}]) {{ \
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
         __declspec(dllexport) void kbench(const float* x, const float* y, float* out) {{ \
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

/// The bf16 twin of [`bench_mercury_i8`]: JIT-compile a Mercury bf16 reduction and time it. `out` is a
/// 1-element f32 buffer holding the scalar result.
fn bench_mercury_bf16(
    src: &str,
    out: &mut [f32],
    xp: *const u16,
    yp: *const u16,
    op: *mut f32,
) -> Option<MeasureBf16> {
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
    let f: Bf16KernelFn = unsafe { std::mem::transmute(ptr) };

    out[0] = 0.0;
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
    op: *mut f32,
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
        let ns = time_ns(|| f(xp, yp, op));
        let snapshot = out[0];
        Some(MeasureBf16 {
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
        // Hadamard product `out = x*y` — gating / attention-mask / residual-scaling. Mercury dispatches
        // it to the 256-bit `mercury_velem_f32` (VE_HADAMARD) with non-temporal stores; gcc/rustc also
        // auto-vectorize a bare product, so at this L3-resident size it is a bandwidth tie (the NT-store
        // edge shows >L3, like saxpy). Recognized where the affine matcher declines (both factors vary).
        Kernel {
            name: "hadamard",
            bytes_per_call: 3 * N * 4,
            note: "out = x*y (Hadamard): 256-bit velem (+NT store); gcc/rustc autovectorize too",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = x[i] * y[i]; }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=x[i]*y[i];"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= *x.add(i)* *y.add(i); }"),
        },
        // SiLU written the textbook way, as a product `x*sigmoid(x)` (the form before the `silu()`
        // intrinsic, and the value==gate SwiGLU case). Mercury folds the product to the 256-bit
        // VMATH_SILU kernel; gcc/rustc must call scalar `expf` (no vectorized libm), so it stays serial
        // — a compute-bound win like the bare transcendentals, not a bandwidth tie.
        Kernel {
            name: "gated_silu",
            bytes_per_call: 2 * N * 4,
            note: "out = x*sigmoid(x): product folded to 256-bit VMATH_SILU; C/Rust scalar expf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = x[i] * sigmoid(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++){ float v=x[i]; out[i]= v/(1.0f+expf(-v)); }"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); *out.add(i)= v/(1.0+(-v).exp()); }",
            ),
        },
        // argmax — the greedy-decode hot path (`next_token = argmax(logits)`). Mercury recognizes the
        // `if x[k] > bv { bv = x[k]; bi = k }` loop and dispatches to the deterministic argreduce kernel
        // (branchless 8-lane fold); gcc/rustc run a branchy scalar loop. Like `dot`, it writes only
        // out[0] (the index), so the full-buffer cross-check sees the same single value in all three.
        Kernel {
            name: "argmax",
            bytes_per_call: N * 4,
            note: "out[0] = argmax(x): branchless argreduce kernel vs a branchy scalar loop",
            mer: mer_kernel(&format!(
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
        // sin/cos: the rotary-position-embedding (RoPE) transcendentals every modern LLM precomputes.
        // Mercury dispatches a pure `out[i]=sin(x[i])` loop to the 256-bit AVX2 `mercury_vmath_f32`
        // (VM_SIN/VM_COS); gcc/rustc call scalar libm `sinf`/`cosf` and cannot vectorize a loop with the
        // call — the same compute-bound regime as `exp`.
        Kernel {
            name: "sin",
            bytes_per_call: 2 * N * 4,
            note: "out = sin(x): Mercury dispatches to a 256-bit AVX2 poly; C/Rust call scalar libm sinf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = sin(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=sinf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).sin(); }"),
        },
        Kernel {
            name: "cos",
            bytes_per_call: 2 * N * 4,
            note: "out = cos(x): Mercury dispatches to a 256-bit AVX2 poly; C/Rust call scalar libm cosf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = cos(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=cosf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).cos(); }"),
        },
        // erf: the exact (erf-based) GELU's core transcendental — BERT/GPT-2's original activation.
        // Mercury vectorizes the Abramowitz–Stegun poly at 256-bit; C calls scalar libm `erff`, and Rust
        // (no std erf) runs the idiomatic scalar A&S a programmer writes without a libm dependency.
        Kernel {
            name: "erf",
            bytes_per_call: 2 * N * 4,
            note: "out = erf(x): Mercury dispatches to a 256-bit AVX2 poly; C calls scalar libm erff",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = erf(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=erff(x[i]);"),
            rust: rust_kernel(
                "for i in 0..N { let v= *x.add(i); let s=if v<0.0 {-1.0f32} else {1.0}; \
                 let ax=v.abs(); let t=1.0f32/(1.0+0.3275911*ax); \
                 let y=1.0f32-(((((1.061405429f32*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t)*(-ax*ax).exp(); \
                 *out.add(i)=s*y; }",
            ),
        },
        // Comprehensive vectorized elementwise math: base-2 exp/log (FlashAttention-2 base-2 softmax,
        // quantization bit-width, entropy in bits) and hyperbolic sinh/cosh. Mercury dispatches each to
        // the 256-bit `mercury_vmath_f32`; C/Rust call scalar libm `exp2f`/`log2f`/`sinhf`/`coshf` and
        // cannot vectorize the call — the same compute-bound win as `exp`.
        Kernel {
            name: "exp2",
            bytes_per_call: 2 * N * 4,
            note: "out = exp2(x) = 2^x: 256-bit AVX2 (exp(x·ln2)) vs scalar libm exp2f",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = exp2(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=exp2f(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).exp2(); }"),
        },
        Kernel {
            name: "log2",
            bytes_per_call: 2 * N * 4,
            note: "out = log2(x): 256-bit AVX2 (log(x)·log2e) vs scalar libm log2f",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = log2(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=log2f(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).log2(); }"),
        },
        Kernel {
            name: "sinh",
            bytes_per_call: 2 * N * 4,
            note: "out = sinh(x) = (e^x - e^-x)/2: 256-bit AVX2 vs scalar libm sinhf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = sinh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=sinhf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).sinh(); }"),
        },
        Kernel {
            name: "cosh",
            bytes_per_call: 2 * N * 4,
            note: "out = cosh(x) = (e^x + e^-x)/2: 256-bit AVX2 vs scalar libm coshf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = cosh(x[i]); }}")),
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
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = asinh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=asinhf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).asinh(); }"),
        },
        Kernel {
            name: "acosh",
            bytes_per_call: 2 * N * 4,
            note: "out = acosh(x), x>=1: 256-bit AVX2 (log+sqrt) vs scalar libm acoshf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = acosh(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=acoshf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).acosh(); }"),
        },
        // atan (all-real) — Cephes 3-region reduction + degree-3 poly at 256-bit; C/Rust call scalar
        // libm atanf and can't vectorize the call. Angle/geometry ops, atan2-style positional schemes.
        Kernel {
            name: "atan",
            bytes_per_call: 2 * N * 4,
            note: "out = atan(x): 256-bit AVX2 (Cephes) vs scalar libm atanf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = atan(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=atanf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).atan(); }"),
        },
        // expm1/log1p — the numerically-stable eˣ−1 / ln(1+x) (Kahan, reusing the 256-bit exp/log);
        // C/Rust call scalar libm expm1f/log1pf and can't vectorize the call. ELU exact tail, stable losses.
        Kernel {
            name: "expm1",
            bytes_per_call: 2 * N * 4,
            note: "out = expm1(x) = e^x-1: 256-bit AVX2 (Kahan) vs scalar libm expm1f",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = expm1(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=expm1f(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).exp_m1(); }"),
        },
        Kernel {
            name: "log1p",
            bytes_per_call: 2 * N * 4,
            note: "out = log1p(x) = ln(1+x): 256-bit AVX2 (Kahan) vs scalar libm log1pf",
            mer: mer_kernel(&format!("for i in 0..{nlit} {{ out[i] = log1p(x[i]); }}")),
            c: c_kernel("for(long i=0;i<N;i++) out[i]=log1pf(x[i]);"),
            rust: rust_kernel("for i in 0..N { *out.add(i)= (*x.add(i)).ln_1p(); }"),
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
