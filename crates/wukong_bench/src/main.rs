//! `wukong-bench` — the optimizer-effectiveness and interpreter-performance harness.
//!
//! For each `.wk` program it can lower, it reports, comparing `-O0` against `-O3`:
//!   * total MIR op count (instructions plus one terminator per block) and the reduction;
//!   * interpreter wall-clock time per run (adaptively batched) and the resulting speedup. Each such
//!     run includes one 512 MiB worker-thread spawn+join, because that is what `wukong_interp::run`
//!     costs a caller; the report measures that floor and prints both the raw and the floor-net
//!     speedup so the two are never confused.
//!
//! Every program is also a correctness check: the `-O0` and `-O3` builds must both verify, must
//! agree on **exit code and stdout**, and the Cranelift backend at `-O3` must agree with the
//! interpreter at `-O3`; a program that lowered cleanly and then breaks any of those is reported as
//! `*** FAILED` and the process exits 1. (Programs that never lower, or that error identically at
//! both levels, are merely skipped — that is not a regression.) This quantifies the optimizer and
//! guards against regressions without needing an LLVM toolchain. Run with:
//!
//! ```text
//! cargo run -p wukong_bench --release -- tests/run examples bench/kernels
//! ```
//!
//! The default report above is one of five modes; the first CLI argument selects among them
//! (`compile-time`, `compile-profile` and `spawn-overhead` — see the `compile_time` and `profile`
//! modules — plus `compile-vs`, see `compile_vs`). Anything that is not a mode word is a corpus
//! directory, and with no arguments at all the corpus defaults to `tests/run`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use wukong_span::{Interner, SourceId};

mod compile_time;
mod compile_vs;
mod profile;

/// Every mode of this harness prints a wall-clock number, and every one of those numbers is
/// meaningless from a build with optimizations off — an unoptimized `wukong_opt`/`wukong_interp`
/// does not have the same shape as the shipped one, and the project has already been burned by
/// exactly this (a debug build makes runtime kernels read ~7x slower than JIT loops, inverting A/B
/// conclusions). The harness still runs, because the -O0-vs-O3 equivalence check it also performs is
/// a correctness gate that is valid in any build; but it says so first, unmissably, so no number
/// taken from a debug build can be mistaken for a measurement.
fn warn_if_debug_build() {
    if cfg!(debug_assertions) {
        eprintln!(
            "*** DEBUG BUILD — every timing below is NOT a measurement and must not be reported. \
             Re-run\n*** with `cargo run -p wukong_bench --release`. (Correctness results are still \
             valid.)"
        );
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    warn_if_debug_build();

    // Mode selector: `compile-time` switches to the in-process optimizer-timing report; anything
    // else keeps the original optimizer-effectiveness + execution-timing report. Appended, not
    // overlaid, so the default invocation is unchanged.
    // `compile-vs [wukongc-path]` runs the cross-compiler compile-time comparison and exits.
    if matches!(args.first().map(String::as_str), Some("compile-vs")) {
        let mc = args.get(1).map(std::path::PathBuf::from);
        compile_vs::report(mc);
        return;
    }

    let ctime = matches!(
        args.first().map(String::as_str),
        Some("compile-time" | "--compile-time" | "ctime")
    );
    // `compile-profile` — full-pipeline per-stage breakdown; `spawn-overhead` — in-process vs
    // process-spawn characterization. Both consume the same corpus dirs as `compile-time`, so they
    // are detected here (after `compile-vs`'s early return) and dispatched once `files` is built.
    let profile = matches!(
        args.first().map(String::as_str),
        Some("compile-profile" | "profile")
    );
    let spawn = matches!(
        args.first().map(String::as_str),
        Some("spawn-overhead" | "spawn")
    );
    if ctime || profile || spawn {
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
                if p.extension().map(|x| x == "wk").unwrap_or(false) {
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
    if profile {
        profile::report(&files);
        return;
    }
    if spawn {
        profile::spawn_report(&files);
        return;
    }

    // Measure the interpreter's per-call floor BEFORE the table, so the disclosure sits above the
    // numbers it qualifies (see `interp_call_floor`).
    let floor = interp_call_floor();
    match floor {
        Some(fl) => println!(
            "interp-call floor {}: `wukong_interp::run` wraps EVERY call in a 512 MiB-stack worker\n\
             thread (`with_big_stack`), so one thread spawn+join is inside every `interp O3` cell\n\
             below and inside every `nat:intp` ratio. Measured here by timing the smallest possible\n\
             Wukong program with this same instrument. A cell at or near the floor is reporting\n\
             Windows thread creation, not interpretation.\n",
            fmt_dur(fl)
        ),
        None => println!(
            "interp-call floor: NOT MEASURED — the probe program failed to lower. Every `interp O3`\n\
             cell below still contains one 512 MiB worker-thread spawn+join, of unknown size here.\n"
        ),
    }

    println!(
        "{:<20} {:>7} {:>7} {:>9}  {:>11} {:>11} {:>9}",
        "program", "O0 ops", "O3 ops", "reduction", "interp O3", "native O3", "nat:intp"
    );
    println!("{}", "-".repeat(82));

    let (mut tot0, mut tot3) = (0usize, 0usize);
    let mut log_speedup_sum = 0.0f64;
    let mut speedup_n = 0u32;
    // The same geomean with the interp-call floor taken out of the numerator, plus a count of the
    // programs that interpret faster than the floor (for which no interpreter time can be separated
    // from thread creation at all).
    let mut log_net_sum = 0.0f64;
    let mut net_n = 0u32;
    let mut below_floor = 0u32;
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
                match floor {
                    Some(fl) if r.t_interp > fl => {
                        let net = (r.t_interp - fl).as_secs_f64() / r.t_native.as_secs_f64().max(1e-12);
                        log_net_sum += net.ln();
                        net_n += 1;
                    }
                    Some(_) => below_floor += 1,
                    None => {}
                }
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

    // The geomean above is RAW: its numerator still contains one interp-call floor per program.
    // Print the floor-net figure next to it so nobody can quote the raw one as "native is Nx faster
    // than interpreting" without seeing what that claim shrinks to.
    if let Some(fl) = floor {
        if net_n > 0 {
            let net_geo = (log_net_sum / net_n as f64).exp();
            println!(
                "nat:intp above is RAW (one {} interp-call floor is inside every numerator).\n\
                 Net of that floor: geomean {net_geo:.1}x over {net_n} program(s).",
                fmt_dur(fl)
            );
        }
        if below_floor > 0 {
            println!(
                "{below_floor} program(s) interpret faster than the {} floor, so no interpreter time \
                 can be\nseparated from thread creation for them; they are excluded from the net \
                 geomean but not\nfrom the raw one.",
                fmt_dur(fl)
            );
        }
    }

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
    /// Lowered, optimized, verified, produced identical (exit code, stdout) at -O0 and -O3, and the
    /// Cranelift backend at -O3 agreed with the interpreter at -O3.
    Ran(Res),
    /// Did not lower to runnable MIR (e.g. uses tensor/SIMD constructs codegen doesn't support),
    /// or fails identically at both levels (e.g. a deliberate runtime assertion). Not a regression.
    Skipped(String),
    /// Lowered but the optimizer changed behavior, produced invalid MIR, or the native backend
    /// disagreed with the interpreter at -O3 — a real bug. Any `Failed` exits the process 1.
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

/// Percent change in op count from `a` (-O0) to `b` (-O3), **signed**.
///
/// The subtraction is done in `f64`, not `usize`, because nothing guarantees `a >= b`: inlining at
/// -O2/-O3 can make the optimized program larger than the unoptimized one. A 150-call-site corpus
/// program measured 962 ops at -O0 and 1416 at -O3, and the old `(a - b) as f64` wrapped that into a
/// printed `1917540964003072000.0%` — corrupting the row *and* the TOTAL — while a debug build (where
/// overflow checks are on) would instead panic and abort the whole optimizer-equivalence gate.
/// A negative value, which the `{:>8.1}%` format already renders as `-47.2%`, is the honest report
/// for a program the optimizer grows.
fn reduction(a: usize, b: usize) -> f64 {
    if a == 0 {
        0.0
    } else {
        100.0 * (a as f64 - b as f64) / a as f64
    }
}

fn bench_one(path: &Path) -> Outcome {
    let Ok(src) = std::fs::read_to_string(path) else {
        return Outcome::Skipped("cannot read file".into());
    };
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return Outcome::Skipped("parse error".into());
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Outcome::Skipped("type/shape error".into());
    }
    let (mut p0, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return Outcome::Skipped("uses constructs codegen does not support yet".into());
    }
    let mut p3 = p0.clone();
    wukong_opt::optimize(&mut p0, 0);
    wukong_opt::optimize(&mut p3, 3);

    // From here a discrepancy is a real bug: the program lowered cleanly, so optimization must
    // preserve both well-formedness and behavior.
    for f in p0.funcs.iter().chain(p3.funcs.iter()) {
        if !wukong_mir::verify::verify_function(f).is_empty() {
            return Outcome::Failed("MIR fails verification after optimization".into());
        }
    }
    let main = interner.intern("main");
    // Compare -O0 vs -O3 on BOTH exit code and stdout (the same equality the native gate below
    // uses): an optimizer bug that changes what a program PRINTS while preserving its exit code
    // slipped through the old exit-only comparison — and the bench dirs (bench/kernels,
    // examples) have no other opt-invariance gate, unlike tests/run.
    let r0 = wukong_interp::run_with_output(&p0, main, &interner);
    let r3 = wukong_interp::run_with_output(&p3, main, &interner);
    let interp3 = match (r0, r3) {
        // Both fail identically (e.g. a deliberate runtime assertion) — consistent, not a bug.
        (Err(_), Err(_)) => return Outcome::Skipped("runtime error at both -O0 and -O3".into()),
        (Ok(out0), Ok(out3)) if out0 == out3 => out3,
        (Ok(out0), Ok(out3)) => {
            return Outcome::Failed(format!(
                "result differs: -O0 exit={} vs -O3 exit={}{}",
                out0.0,
                out3.0,
                if out0.1 != out3.1 {
                    " (stdout differs)"
                } else {
                    ""
                }
            ))
        }
        _ => return Outcome::Failed("runs at one optimization level but not the other".into()),
    };

    // Second soundness gate: the native (Cranelift) backend at -O3 must match the interpreter at
    // -O3 on both exit code and stdout. This makes the native path a checked peer of the oracle.
    match wukong_codegen_cranelift::jit_run(&p3, main, &interner) {
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
    p: &wukong_mir::Program,
    entry: wukong_span::Symbol,
    interner: &Interner,
) -> Duration {
    let prog = match wukong_codegen_cranelift::jit_compile(p, entry, interner) {
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

/// The fixed cost `wukong_interp::run` pays per call, before it interprets anything.
///
/// `run` → `run_with_output` → `with_big_stack` (wukong_interp/src/lib.rs) spawns a 512 MiB-stack
/// worker thread and joins it on **every** call, so the recursive tree-walker cannot overflow the
/// host stack. [`time_run`] calls `run` inside its rep loop, so that spawn+join is charged to every
/// rep — and for any program that interprets faster than one thread creation, the `interp O3` column
/// and the `nat:intp` geomean are measuring Windows, not the interpreter.
///
/// This measures the floor instead of assuming it: lower the smallest Wukong program there is and
/// time it with the *same* [`time_run`] instrument. Its interpretation is a handful of MIR ops, so
/// what comes back is the per-call floor. It deliberately goes through the real `wukong_interp::run`
/// rather than replicating `with_big_stack` here, so the number stays true if the interpreter ever
/// changes how it obtains its stack. `None` if the probe program does not lower (never expected).
fn interp_call_floor() -> Option<Duration> {
    const SRC: &str = "module bench.floor\nfn main() -> i32 { return 0; }\n";
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(SRC, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (mut p, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    wukong_opt::optimize(&mut p, 3);
    let main = interner.intern("main");
    wukong_interp::run(&p, main, &interner).ok()?;
    Some(time_run(&p, main, &interner))
}

/// Time one full execution, adaptively batching until at least 50 ms has elapsed so even fast
/// programs are measured stably. Returns the per-run duration.
///
/// NOTE: every rep pays one 512 MiB worker-thread spawn+join inside `wukong_interp::run` — see
/// [`interp_call_floor`], which measures it and which the report prints alongside this column.
fn time_run(
    p: &wukong_mir::Program,
    entry: wukong_span::Symbol,
    interner: &Interner,
) -> Duration {
    for _ in 0..2 {
        let _ = wukong_interp::run(p, entry, interner); // warm up
    }
    let mut reps: u32 = 1;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            let _ = wukong_interp::run(p, entry, interner);
        }
        let elapsed = start.elapsed();
        if elapsed >= Duration::from_millis(50) || reps >= 1 << 22 {
            return elapsed / reps;
        }
        reps *= 2;
    }
}

/// Total MIR operations: instructions plus one terminator per block.
fn count_ops(p: &wukong_mir::Program) -> usize {
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
