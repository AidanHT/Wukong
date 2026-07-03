//! Same-run compile-time comparison: mercuryc vs gcc / g++ / rustc.
//!
//! Compile time is one of the three metrics an ML/DL compiler is judged on, and the axis where
//! Mercury's toolchain-free, LLVM-free design should dominate. This measures the honest, user-visible
//! quantity: the wall-clock time to compile an *equivalent* tensor kernel to a native object file,
//! measured **same-run** and back-to-back so the laptop's clock state cancels out of the ratio.
//!
//!   mercuryc:  <mercuryc> --emit=obj -O2 k.mer
//!   gcc:       gcc -O2 -c k.c
//!   g++:       g++ -O2 -c k.cpp
//!   rustc:     rustc -O --crate-type=lib --emit=obj k.rs
//!
//! Each is compile-only (no link), producing an object from the same computation, so the numbers
//! reflect the compilers' own work — front-end + optimizer + native codegen + process startup. That
//! startup + backend weight is exactly what Mercury's lean Cranelift path avoids, so it is fair to
//! include: it is what a user waits for. Best-of-N minimum per compiler (least noise), reported as a
//! clock-invariant ratio, never an absolute headline. Missing toolchains are skipped, not failed.
//!
//! Fairness: the C/C++ kernels are **bare translation units** — a `void`-returning exported
//! function whose result escapes through an out-parameter, with **no `#include` and no `main`** —
//! matching the .rs kernels (a bare `#[no_mangle]` fn), so all four languages compile a comparable
//! pure kernel to an object. (Earlier versions gave C/C++ a `#include <stdio.h>` + `main`/`printf`
//! harness, charging them a header-parse cost the .rs kernels never paid — flagged and fixed by the
//! benchmark-fairness audit.) rustc keeps `-O` (= opt-level 2) because gcc/g++ compile at `-O2`:
//! level 2 across the board is the symmetric choice for a *compile-time* measurement.
//!
//! ```text
//! cargo run -p mercury_bench --release -- compile-vs [path/to/mercuryc]
//! ```

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// One kernel expressed identically across the four languages.
struct Kernel {
    name: &'static str,
    mer: &'static str,
    c: &'static str,
    cpp: &'static str,
    rs: &'static str,
}

const BUDGET: Duration = Duration::from_millis(600);
const MAX_REPS: u32 = 40;

pub fn report(mercuryc: Option<PathBuf>) {
    let mc = mercuryc.unwrap_or_else(default_mercuryc);
    if !mc.exists() {
        eprintln!(
            "mercuryc not found at {} — build it (cargo build --release -p mercuryc) or pass its path",
            mc.display()
        );
        std::process::exit(2);
    }
    let have_gcc = tool_exists("gcc");
    let have_gpp = tool_exists("g++");
    let have_rustc = tool_exists("rustc");

    let dir = std::env::temp_dir().join("mercury_compile_vs");
    let _ = std::fs::create_dir_all(&dir);

    println!(
        "same-run compile-to-object wall time (best-of-N minimum), and speedup vs mercuryc\n"
    );
    println!(
        "{:<10} {:>11} {:>11} {:>11} {:>11}   {:>7} {:>7} {:>7}",
        "kernel", "mercuryc", "gcc", "g++", "rustc", "gcc/mc", "g++/mc", "rc/mc"
    );
    println!("{}", "-".repeat(90));

    for k in KERNELS {
        let mer_path = dir.join(format!("{}.mer", k.name));
        let c_path = dir.join(format!("{}.c", k.name));
        let cpp_path = dir.join(format!("{}.cpp", k.name));
        let rs_path = dir.join(format!("{}.rs", k.name));
        std::fs::write(&mer_path, k.mer).unwrap();
        std::fs::write(&c_path, k.c).unwrap();
        std::fs::write(&cpp_path, k.cpp).unwrap();
        std::fs::write(&rs_path, k.rs).unwrap();
        let out = dir.join("out.o");
        let o = |p: &std::path::Path| p.to_string_lossy().into_owned();

        let t_mc = best_of(|| {
            run(Command::new(&mc).args([
                "--emit=obj",
                "-O2",
                &o(&mer_path),
                "-o",
                &o(&out),
            ]))
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
        "ratios > 1.0 mean mercuryc compiled the equivalent kernel that many times faster \
         (same run).\ncompile-only (no link); mercuryc uses Cranelift (no LLVM), gcc/g++/rustc \
         their own -O2 backend."
    );
}

fn default_mercuryc() -> PathBuf {
    let exe = if cfg!(windows) { "mercuryc.exe" } else { "mercuryc" };
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
    mer: r#"module bench.gemm
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
    mer: r#"module bench.saxpy
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
    mer: r#"module bench.dot
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
