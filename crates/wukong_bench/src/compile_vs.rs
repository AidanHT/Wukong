//! Same-run compile-time comparison: wukongc vs gcc / g++ / rustc.
//!
//! Compile time is one of the three metrics an ML/DL compiler is judged on, and the axis where
//! Wukong's toolchain-free, LLVM-free design should dominate. This measures the honest, user-visible
//! quantity: the wall-clock time to compile an *equivalent* tensor kernel to a native object file,
//! measured **same-run** and back-to-back so the laptop's clock state cancels out of the ratio.
//!
//!   wukongc:  <wukongc> --emit=obj -O2 k.wk -o out.o
//!   gcc:       gcc -O2 -c k.c -o out.o
//!   g++:       g++ -O2 -c k.cpp -o out.o
//!   rustc:     rustc -O --crate-type=lib --emit=obj -A warnings k.rs -o out.o
//!
//! Each is compile-only (no link), producing an object from the same computation, so the numbers
//! reflect the compilers' own work — front-end + optimizer + native codegen + process startup. That
//! startup + backend weight is exactly what Wukong's lean Cranelift path avoids, so it is fair to
//! include: it is what a user waits for. Best-of-N minimum per compiler (least noise), reported as a
//! clock-invariant ratio, never an absolute headline. Missing toolchains are skipped, not failed.
//!
//! Fairness: the C/C++ kernels are **bare translation units** — a `void`-returning exported
//! function whose result escapes through an out-parameter, with **no `#include` and no `main`** —
//! matching the .rs kernels (a bare `#[no_mangle]` fn), so the three peers compile a comparable pure
//! kernel to an object. The `.wk` arm is *not* bare — it is a full program with `main` and a `print`
//! — which is the one asymmetry left, and it is printed under the table with the ratios rather than
//! only recorded here. (Earlier versions gave C/C++ a `#include <stdio.h>` + `main`/`printf`
//! harness, charging them a header-parse cost the .rs kernels never paid — flagged and fixed by the
//! benchmark-fairness audit.) rustc keeps `-O` (= opt-level 2) because gcc/g++ compile at `-O2`:
//! level 2 across the board is the symmetric choice for a *compile-time* measurement.
//!
//! ```text
//! cargo run -p wukong_bench --release -- compile-vs [path/to/wukongc]
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// One kernel expressed identically across the four languages.
struct Kernel {
    name: &'static str,
    wuk: &'static str,
    c: &'static str,
    cpp: &'static str,
    rs: &'static str,
}

const BUDGET: Duration = Duration::from_millis(600);
const MAX_REPS: u32 = 40;

pub fn report(wukongc: Option<PathBuf>) {
    let mc = wukongc.unwrap_or_else(default_wukongc);
    if !mc.exists() {
        eprintln!(
            "wukongc not found at {} — build it (cargo build --release -p wukongc) or pass its path",
            mc.display()
        );
        std::process::exit(2);
    }
    // `compile-vs [path]` takes an arbitrary wukongc, and the whole table is a ratio against it, so
    // a debug wukongc silently turns every published speedup into fiction. gcc/g++/rustc are always
    // the shipped release binaries, so the comparison would not even be like-for-like.
    if mc.components().any(|c| c.as_os_str() == "debug") {
        eprintln!(
            "*** wukongc at {} looks like a DEBUG build — the ratios below are against an\n\
             *** unoptimized compiler while gcc/g++/rustc are release binaries. Not reportable.",
            mc.display()
        );
    }
    let have_gcc = tool_exists("gcc");
    let have_gpp = tool_exists("g++");
    let have_rustc = tool_exists("rustc");

    let dir = std::env::temp_dir().join("wukong_compile_vs");
    let _ = std::fs::create_dir_all(&dir);

    println!("same-run compile-to-object wall time (best-of-N minimum), and speedup vs wukongc\n");
    println!(
        "{:<10} {:>11} {:>11} {:>11} {:>11}   {:>7} {:>7} {:>7}",
        "kernel", "wukongc", "gcc", "g++", "rustc", "gcc/mc", "g++/mc", "rc/mc"
    );
    println!("{}", "-".repeat(90));

    for k in KERNELS {
        let wk_path = dir.join(format!("{}.wk", k.name));
        let c_path = dir.join(format!("{}.c", k.name));
        let cpp_path = dir.join(format!("{}.cpp", k.name));
        let rs_path = dir.join(format!("{}.rs", k.name));
        std::fs::write(&wk_path, k.wuk).unwrap();
        std::fs::write(&c_path, k.c).unwrap();
        std::fs::write(&cpp_path, k.cpp).unwrap();
        std::fs::write(&rs_path, k.rs).unwrap();
        let out = dir.join("out.o");
        let o = |p: &std::path::Path| p.to_string_lossy().into_owned();

        let t_mc = best_of(|| {
            run(Command::new(&mc).args(["--emit=obj", "-O2", &o(&wk_path), "-o", &o(&out)]))
        });
        let t_gcc = have_gcc.then(|| {
            best_of(|| run(Command::new("gcc").args(["-O2", "-c", &o(&c_path), "-o", &o(&out)])))
        });
        let t_gpp = have_gpp.then(|| {
            best_of(|| run(Command::new("g++").args(["-O2", "-c", &o(&cpp_path), "-o", &o(&out)])))
        });
        let t_rc = have_rustc.then(|| {
            best_of(|| {
                run(Command::new("rustc").args([
                    "-O",
                    "--crate-type=lib",
                    "--emit=obj",
                    "-A",
                    "warnings",
                    &o(&rs_path),
                    "-o",
                    &o(&out),
                ]))
            })
        });

        println!(
            "{:<10} {:>11} {:>11} {:>11} {:>11}   {:>7} {:>7} {:>7}",
            k.name,
            fmt(Some(t_mc)),
            fmt(t_gcc),
            fmt(t_gpp),
            fmt(t_rc),
            ratio(t_gcc, t_mc),
            ratio(t_gpp, t_mc),
            ratio(t_rc, t_mc),
        );
    }
    println!("{}", "-".repeat(90));
    println!(
        "ratios > 1.0 mean wukongc compiled the equivalent kernel that many times faster \
         (same run).\ncompile-only (no link); wukongc uses Cranelift (no LLVM), gcc/g++/rustc \
         their own -O2 backend."
    );
    // The fairness decision the ratios rest on, printed with the ratios. This table is what gets
    // pasted into BENCHMARKS.md and docs/metrics.md; from the numbers alone a reader cannot tell
    // whether gcc was charged a libc header parse, so a regression that re-added an `#include` to
    // the C kernel would restore previously-fixed rigging while the published output looked
    // unchanged.
    println!(
        "peer sources are header-free bare translation units — no #include, no main — matching the \
         bare\n#[no_mangle] .rs kernels; the .wk arm is a full program with main + print. rustc uses \
         -O\n(= opt-level 2) to match gcc/g++ -O2."
    );
}

fn default_wukongc() -> PathBuf {
    let exe = if cfg!(windows) {
        "wukongc.exe"
    } else {
        "wukongc"
    };
    PathBuf::from("target/release").join(exe)
}

fn tool_exists(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run a compile command once, returning its wall time. Panics if the compile fails (a broken
/// equivalent kernel is a bug in this harness, not a measurement).
fn run(cmd: &mut Command) -> Duration {
    let t0 = Instant::now();
    let status = cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("failed to spawn compiler");
    let dt = t0.elapsed();
    assert!(status.success(), "compile failed: {cmd:?}");
    dt
}

/// Best-of-N minimum wall time — the run least perturbed by scheduler/thermal noise.
fn best_of<F: FnMut() -> Duration>(mut run: F) -> Duration {
    run(); // warm up (fill the OS file cache, page in the compiler)
    let mut best = Duration::MAX;
    let mut reps = 0u32;
    let start = Instant::now();
    loop {
        best = best.min(run());
        reps += 1;
        if start.elapsed() >= BUDGET || reps >= MAX_REPS {
            break;
        }
    }
    best
}

fn ratio(other: Option<Duration>, mc: Duration) -> String {
    match other {
        Some(d) => format!("{:.2}x", d.as_secs_f64() / mc.as_secs_f64().max(1e-9)),
        None => "-".into(),
    }
}

fn fmt(d: Option<Duration>) -> String {
    match d {
        Some(d) => {
            let ms = d.as_secs_f64() * 1e3;
            format!("{ms:.2}ms")
        }
        None => "n/a".into(),
    }
}

// ---- the equivalent kernels ---------------------------------------------------------------------

const KERNELS: &[Kernel] = &[GEMM, SAXPY, DOT];

const GEMM: Kernel = Kernel {
    name: "gemm4x4",
    wuk: r#"module bench.gemm
fn main() -> i32 {
    let a: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let b: [f32; 16] = [16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut c: [f32; 16] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut i: i32 = 0;
    while i < 4 {
        let mut j: i32 = 0;
        while j < 4 {
            let mut s: f32 = 0.0;
            let mut k: i32 = 0;
            while k < 4 {
                s = s + a[i * 4 + k] * b[k * 4 + j];
                k = k + 1;
            }
            c[i * 4 + j] = s;
            j = j + 1;
        }
        i = i + 1;
    }
    print(c[0]);
    return 0;
}
"#,
    c: r#"void gemm(float* out) {
    float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float b[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    float c[16] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a[i*4+k] * b[k*4+j];
            c[i*4+j] = s;
        }
    out[0] = c[0];
}
"#,
    cpp: r#"extern "C" void gemm(float* out) {
    float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float b[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    float c[16] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a[i*4+k] * b[k*4+j];
            c[i*4+j] = s;
        }
    out[0] = c[0];
}
"#,
    rs: r#"#[no_mangle]
pub extern "C" fn gemm() -> f32 {
    let a: [f32; 16] = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0];
    let b: [f32; 16] = [16.0,15.0,14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0];
    let mut c: [f32; 16] = [0.0; 16];
    for i in 0..4usize {
        for j in 0..4usize {
            let mut s = 0.0f32;
            for k in 0..4usize { s += a[i*4+k] * b[k*4+j]; }
            c[i*4+j] = s;
        }
    }
    c[0]
}
"#,
};

const SAXPY: Kernel = Kernel {
    name: "saxpy",
    wuk: r#"module bench.saxpy
fn main() -> i32 {
    let a: f32 = 2.0;
    let x: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: [f32; 8] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let mut out: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut i: i32 = 0;
    while i < 8 {
        out[i] = a * x[i] + y[i];
        i = i + 1;
    }
    print(out[0]);
    return 0;
}
"#,
    c: r#"void saxpy(float* res) {
    float a = 2.0f;
    float x[8] = {1,2,3,4,5,6,7,8};
    float y[8] = {10,20,30,40,50,60,70,80};
    float out[8] = {0};
    for (int i = 0; i < 8; i++) out[i] = a*x[i] + y[i];
    res[0] = out[0];
}
"#,
    cpp: r#"extern "C" void saxpy(float* res) {
    float a = 2.0f;
    float x[8] = {1,2,3,4,5,6,7,8};
    float y[8] = {10,20,30,40,50,60,70,80};
    float out[8] = {0};
    for (int i = 0; i < 8; i++) out[i] = a*x[i] + y[i];
    res[0] = out[0];
}
"#,
    rs: r#"#[no_mangle]
pub extern "C" fn saxpy() -> f32 {
    let a = 2.0f32;
    let x: [f32; 8] = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0];
    let y: [f32; 8] = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0];
    let mut out: [f32; 8] = [0.0; 8];
    for i in 0..8usize { out[i] = a*x[i] + y[i]; }
    out[0]
}
"#,
};

const DOT: Kernel = Kernel {
    name: "dot",
    wuk: r#"module bench.dot
fn main() -> i32 {
    let x: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let y: [i32; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
    let mut acc: i32 = 0;
    let mut i: i32 = 0;
    while i < 8 {
        acc = acc + x[i] * y[i];
        i = i + 1;
    }
    print(acc);
    return 0;
}
"#,
    c: r#"void dot(int* out) {
    int x[8] = {1,2,3,4,5,6,7,8};
    int y[8] = {8,7,6,5,4,3,2,1};
    int acc = 0;
    for (int i = 0; i < 8; i++) acc += x[i] * y[i];
    out[0] = acc;
}
"#,
    cpp: r#"extern "C" void dot(int* out) {
    int x[8] = {1,2,3,4,5,6,7,8};
    int y[8] = {8,7,6,5,4,3,2,1};
    int acc = 0;
    for (int i = 0; i < 8; i++) acc += x[i] * y[i];
    out[0] = acc;
}
"#,
    rs: r#"#[no_mangle]
pub extern "C" fn dot() -> i32 {
    let x: [i32; 8] = [1,2,3,4,5,6,7,8];
    let y: [i32; 8] = [8,7,6,5,4,3,2,1];
    let mut acc = 0i32;
    for i in 0..8usize { acc += x[i] * y[i]; }
    acc
}
"#,
};
