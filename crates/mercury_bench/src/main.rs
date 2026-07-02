//! `mercury-bench` — the optimizer-effectiveness and interpreter-performance harness.
//!
//! For each `.mer` program it can lower, it reports, comparing `-O0` against `-O3`:
//!   * total MIR op count (instructions plus one terminator per block) and the reduction;
//!   * interpreter wall-clock time per run (adaptively batched) and the resulting speedup.
//!
//! Every program is also a correctness check: the `-O0` and `-O3` builds must verify and must
//! produce the same result, otherwise it is reported and skipped. This quantifies the optimizer
//! and guards against regressions without needing an LLVM toolchain. Run with:
//!
//! ```text
//! cargo run -p mercury_bench --release -- tests/run examples bench/kernels
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mercury_span::{Interner, SourceId};

mod compile_time;
mod compile_vs;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    // Mode selector: `compile-time` switches to the in-process optimizer-timing report; anything
    // else keeps the original optimizer-effectiveness + execution-timing report. Appended, not
    // overlaid, so the default invocation is unchanged.
    // `compile-vs [mercuryc-path]` runs the cross-compiler compile-time comparison and exits.
    if matches!(args.first().map(String::as_str), Some("compile-vs")) {
        let mc = args.get(1).map(std::path::PathBuf::from);
        compile_vs::report(mc);
        return;
    }

    let ctime = matches!(
        args.first().map(String::as_str),
        Some("compile-time" | "--compile-time" | "ctime")
    );
    if ctime {
        args.remove(0);
    }

    let dirs: Vec<PathBuf> = if args.is_empty() {
        vec![PathBuf::from("tests/run")]
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let mut files: Vec<PathBuf> = Vec::new();
    for d in &dirs {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map(|x| x == "mer").unwrap_or(false) {
                    files.push(p);
                }
            }
        } else {
            eprintln!("warning: cannot read directory {}", d.display());
        }
    }
    files.sort();

    if ctime {
        compile_time::report(&files);
        return;
    }

    println!(
        "{:<20} {:>7} {:>7} {:>9}  {:>11} {:>11} {:>9}",
        "program", "O0 ops", "O3 ops", "reduction", "interp O3", "native O3", "nat:intp"
    );
    println!("{}", "-".repeat(82));

    let (mut tot0, mut tot3) = (0usize, 0usize);
    let mut log_speedup_sum = 0.0f64;
    let mut speedup_n = 0u32;
    let mut failures: Vec<String> = Vec::new();
    for f in &files {
        match bench_one(f) {
            Outcome::Ran(r) => {
                tot0 += r.ops0;
                tot3 += r.ops3;
                let pct = reduction(r.ops0, r.ops3);
                let speedup = r.t_interp.as_secs_f64() / r.t_native.as_secs_f64().max(1e-12);
                log_speedup_sum += speedup.ln();
                speedup_n += 1;
                println!(
                    "{:<20} {:>7} {:>7} {:>8.1}%  {:>11} {:>11} {:>8.1}x",
                    short_name(f),
                    r.ops0,
                    r.ops3,
                    pct,
                    fmt_dur(r.t_interp),
                    fmt_dur(r.t_native),
                    speedup,
                );
            }
            Outcome::Skipped(why) => {
                println!("{:<20} (skipped: {why})", short_name(f))
            }
            Outcome::Failed(why) => {
                println!("{:<20} *** FAILED: {why} ***", short_name(f));
                failures.push(format!("{}: {why}", short_name(f)));
            }
        }
    }

    println!("{}", "-".repeat(82));
    // Geometric mean of the per-program native-vs-interpreter speedups (the right average for
    // ratios).
    let geo = if speedup_n > 0 {
        (log_speedup_sum / speedup_n as f64).exp()
    } else {
        0.0
    };
    println!(
        "{:<20} {:>7} {:>7} {:>8.1}%  {:>11} {:>11} {:>8.1}x",
        "TOTAL / geomean",
        tot0,
        tot3,
        reduction(tot0, tot3),
        "",
        "",
        geo,
    );

    // The harness doubles as a correctness gate: any program that lowers but misbehaves under
    // optimization fails the process so CI catches the regression.
    if !failures.is_empty() {
        eprintln!(
            "\n{} program(s) failed optimization equivalence:",
            failures.len()
        );
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}

/// The result of benchmarking one program.
enum Outcome {
    /// Lowered, optimized, verified, and produced identical results at -O0 and -O3.
    Ran(Res),
    /// Did not lower to runnable MIR (e.g. uses tensor/SIMD constructs codegen doesn't support),
    /// or fails identically at both levels (e.g. a deliberate runtime assertion). Not a regression.
    Skipped(String),
    /// Lowered but the optimizer changed behavior or produced invalid MIR — a real bug.
    Failed(String),
}

struct Res {
    ops0: usize,
    ops3: usize,
    /// Per-run interpreter time at -O3.
    t_interp: Duration,
    /// Per-run native (Cranelift JIT) time at -O3.
    t_native: Duration,
}

fn reduction(a: usize, b: usize) -> f64 {
    if a == 0 {
        0.0
    } else {
        100.0 * (a - b) as f64 / a as f64
    }
}

fn bench_one(path: &Path) -> Outcome {
    let Ok(src) = std::fs::read_to_string(path) else {
        return Outcome::Skipped("cannot read file".into());
    };
    let mut interner = Interner::new();
    let (module, pd) = mercury_parser::parse_module(&src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return Outcome::Skipped("parse error".into());
    }
    let (sema, sd) = mercury_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Outcome::Skipped("type/shape error".into());
    }
    let (mut p0, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return Outcome::Skipped("uses constructs codegen does not support yet".into());
    }
    let mut p3 = p0.clone();
    mercury_opt::optimize(&mut p0, 0);
    mercury_opt::optimize(&mut p3, 3);

    // From here a discrepancy is a real bug: the program lowered cleanly, so optimization must
    // preserve both well-formedness and behavior.
    for f in p0.funcs.iter().chain(p3.funcs.iter()) {
        if !mercury_mir::verify::verify_function(f).is_empty() {
            return Outcome::Failed("MIR fails verification after optimization".into());
        }
    }
    let main = interner.intern("main");
    let r0 = mercury_interp::run(&p0, main, &interner);
    let r3 = mercury_interp::run_with_output(&p3, main, &interner);
    let interp3 = match (r0, r3) {
        // Both fail identically (e.g. a deliberate runtime assertion) — consistent, not a bug.
        (Err(_), Err(_)) => return Outcome::Skipped("runtime error at both -O0 and -O3".into()),
        (Ok(a), Ok(out3)) if a == out3.0 => out3,
        (Ok(a), Ok(out3)) => {
            return Outcome::Failed(format!("result differs: -O0 = {a}, -O3 = {}", out3.0))
        }
        _ => return Outcome::Failed("runs at one optimization level but not the other".into()),
    };

    // Second soundness gate: the native (Cranelift) backend at -O3 must match the interpreter at
    // -O3 on both exit code and stdout. This makes the native path a checked peer of the oracle.
    match mercury_codegen_cranelift::jit_run(&p3, main, &interner) {
        Ok(native3) if native3 == interp3 => {}
        Ok(native3) => {
            return Outcome::Failed(format!(
                "native != interpreter at -O3 (interp exit={}, native exit={})",
                interp3.0, native3.0
            ))
        }
        Err(e) => return Outcome::Failed(format!("native backend failed at -O3: {e}")),
    }

    Outcome::Ran(Res {
        ops0: count_ops(&p0),
        ops3: count_ops(&p3),
        t_interp: time_run(&p3, main, &interner),
        t_native: time_native(&p3, main, &interner),
    })
}

/// Time one native execution at -O3: compile once, warm up, then batch raw calls until at least
/// 50 ms has elapsed. Returns the per-run duration (or zero if it does not compile).
fn time_native(
    p: &mercury_mir::Program,
    entry: mercury_span::Symbol,
    interner: &Interner,
) -> Duration {
    let prog = match mercury_codegen_cranelift::jit_compile(p, entry, interner) {
        Ok(prog) => prog,
        Err(_) => return Duration::ZERO,
    };
    for _ in 0..2 {
        let _ = prog.call();
    }
    let mut reps: u32 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            let _ = prog.call();
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(50) || reps >= 1 << 22 {
            return elapsed / reps;
        }
        reps *= 2;
    }
}

/// Time one full execution, adaptively batching until at least 50 ms has elapsed so even fast
/// programs are measured stably. Returns the per-run duration.
fn time_run(
    p: &mercury_mir::Program,
    entry: mercury_span::Symbol,
    interner: &Interner,
) -> Duration {
    for _ in 0..2 {
        let _ = mercury_interp::run(p, entry, interner); // warm up
    }
    let mut reps: u32 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            let _ = mercury_interp::run(p, entry, interner);
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(50) || reps >= 1 << 22 {
            return elapsed / reps;
        }
        reps *= 2;
    }
}

/// Total MIR operations: instructions plus one terminator per block.
fn count_ops(p: &mercury_mir::Program) -> usize {
    p.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len() + 1)
        .sum()
}

fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2}µs", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

fn short_name(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}
