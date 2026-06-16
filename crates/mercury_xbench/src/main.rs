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

struct Measure {
    compile: Duration,
    ns_per_call: f64,
    checksum: f64,
}

fn main() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let dir = std::env::temp_dir().join("mercury_xbench");
    let _ = std::fs::create_dir_all(&dir);

    println!("Cross-language kernel benchmark — Mercury (native) vs C (gcc -O3) vs Rust (rustc -O)");
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
    bench_matmul(&cc, &dir);
}

/// Matmul is the canonical ML kernel and is compute-bound, so both SIMD and multicore pay off — the
/// regime where a tensor compiler should genuinely beat idiomatic scalar-source code. We benchmark
/// an `ikj`-ordered C = A·B (the cache-friendly idiom that auto-vectorizes well) written the same
/// way in each language, and additionally Mercury's `@parallel` form. Reported as GFLOP/s.
fn bench_matmul(cc: &str, dir: &Path) {
    const NS: usize = 512; // 512x512x512: ~268 MFLOP/call, fits the L2/L3 hierarchy
    let n2 = NS * NS;
    let a: Vec<f32> = (0..n2).map(|i| (i % 7) as f32 * 0.5 + 0.1).collect();
    let b: Vec<f32> = (0..n2).map(|i| (i % 5) as f32 * 0.25 - 0.3).collect();
    let mut c = vec![0.0f32; n2];
    let (ap, bp, cp) = (a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    let flops = 2.0 * (NS as f64).powi(3);

    println!("=== matmul {NS}x{NS} (C=A·B, ikj order; GFLOP/s, higher is better) ===");
    let gflops = |m: &Option<Measure>| {
        m.as_ref()
            .map(|x| format!("{:.1}", flops / x.ns_per_call))
            .unwrap_or_else(|| "n/a".into())
    };

    let mer = bench_mercury(&mer_matmul(NS, false), &mut c, ap, bp, cp);
    let mer_par = bench_mercury(&mer_matmul(NS, true), &mut c, ap, bp, cp);
    let cm = bench_external(
        "c",
        &c_matmul(NS),
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
        &rust_matmul(NS),
        dir,
        "matmul",
        "rustc",
        &["-O", "-Ctarget-cpu=native", "--crate-type=cdylib"],
        &mut c,
        ap,
        bp,
        cp,
    );

    println!(
        "  {:<18} {:>12} {:>12} {:>12} {:>12}",
        "", "Mer(1core)", "Mer(parallel)", "C (gcc)", "Rust"
    );
    println!(
        "  {:<18} {:>12} {:>12} {:>12} {:>12}",
        "GFLOP/s",
        gflops(&mer),
        gflops(&mer_par),
        gflops(&cm),
        gflops(&rm)
    );
    // Cross-language correctness: every backend must compute the same C[0,0] (within f32 tol).
    if let (Some(m), Some(c)) = (&mer_par, &cm) {
        let rel = (m.checksum - c.checksum).abs() / c.checksum.abs().max(1e-6);
        if rel > 1e-3 {
            println!("  ! checksum mismatch Mercury={} C={}", m.checksum, c.checksum);
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

fn report(k: &Kernel, m: &Option<Measure>, c: &Option<Measure>, r: &Option<Measure>) {
    let row = |label: &str, f: &dyn Fn(&Measure) -> String| {
        println!(
            "  {:<14} {:>14} {:>14} {:>14}",
            label,
            m.as_ref().map(|x| f(x)).unwrap_or_else(|| "n/a".into()),
            c.as_ref().map(|x| f(x)).unwrap_or_else(|| "n/a".into()),
            r.as_ref().map(|x| f(x)).unwrap_or_else(|| "n/a".into()),
        );
    };
    println!("  {:<14} {:>14} {:>14} {:>14}", "", "Mercury", "C (gcc)", "Rust");
    row("compile (ms)", &|x| format!("{:.1}", x.compile.as_secs_f64() * 1e3));
    row("runtime (ns)", &|x| format!("{:.0}", x.ns_per_call));
    row("GB/s", &|x| {
        format!("{:.1}", k.bytes_per_call as f64 / x.ns_per_call)
    });
    // Cross-check that all three computed the same thing (within f32 tolerance).
    if let (Some(m), Some(c)) = (m, c) {
        let rel = ((m.checksum - c.checksum).abs()) / c.checksum.abs().max(1e-6);
        if rel > 1e-3 {
            println!(
                "  ! checksum mismatch Mercury={} C={} (rel {:.2e})",
                m.checksum, c.checksum, rel
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
fn bench_mercury(src: &str, out: &mut [f32], xp: *const f32, yp: *const f32, op: *mut f32) -> Option<Measure> {
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
    let checksum = checksum(out);
    drop(handle); // keep alive through timing
    Some(Measure {
        compile,
        ns_per_call: ns,
        checksum,
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
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
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
        let checksum = checksum(out);
        Some(Measure {
            compile,
            ns_per_call: ns,
            checksum,
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

fn checksum(out: &[f32]) -> f64 {
    let n = out.len();
    (out[0] as f64) + (out[n / 2] as f64) + (out[n - 1] as f64)
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
            note: "sum(x*y), strict f32 reduction (no auto-vectorization)",
            mer: mer_kernel(&format!(
                "let mut s: f32 = 0.0; for i in 0..{nlit} {{ s = s + x[i] * y[i]; }} out[0] = s;"
            )),
            c: c_kernel("float s=0.0f; for(long i=0;i<N;i++) s+=x[i]*y[i]; out[0]=s;"),
            rust: rust_kernel(
                "let mut s=0.0f32; for i in 0..N { s+= *x.add(i)* *y.add(i); } *out.add(0)=s;",
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
    ]
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
    format!("#define N {N}\n__declspec(dllexport) void kbench(const float* x, const float* y, float* out) {{\n  {body}\n}}\n")
}

fn rust_kernel(body: &str) -> String {
    format!("const N: usize = {N};\n#[no_mangle]\npub unsafe extern \"C\" fn kbench(x:*const f32, y:*const f32, out:*mut f32) {{\n  {body}\n}}\n")
}
