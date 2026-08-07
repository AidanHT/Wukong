//! In-process compile-time measurement.
//!
//! `--emit`-level (process-spawn) timing of the compiler is noise-dominated — process startup, I/O,
//! and OS scheduling swamp the microseconds the optimizer actually takes. So this measures the cost
//! *inside* one process, on the same in-memory data: it lowers each program once, then times the
//! front-end (parse + sema + mir_build) and the optimizer (`optimize`) each with a best-of-N loop
//! (the minimum single-run time is the least perturbed by scheduler noise), and separately runs one
//! instrumented [`wukong_opt::optimize_timed`] pass to attribute time per optimizer pass.
//!
//! That instrumented pass is also gated: `measure_one` renders both the `optimize` and the
//! `optimize_timed` program with `wukong_mir::print::print_program` and requires the two texts to be
//! byte-identical, so the timing path can never quietly become a second optimizer. A divergence is
//! reported as `*** FAILED` and this mode exits 1.
//!
//! Because ~80–85% of front→O2 compile time is the optimizer, the per-pass table is the map of where
//! the time goes — the thing this harness exists to expose so it can be cut. Run with:
//!
//! ```text
//! cargo run -p wukong_bench --release -- compile-time tests/run examples bench/kernels
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use wukong_span::{Interner, SourceId};

/// Optimization level the compile-time report measures. `-O3` runs the full `-O2` pipeline, which
/// is the heaviest and the one worth optimizing.
const LEVEL: u8 = 3;

/// Per-measurement wall-clock budget: keep sampling the min until this much time has elapsed (or
/// the rep cap is hit). Big enough for a stable minimum, small enough to keep the whole run brisk.
const BUDGET: Duration = Duration::from_millis(40);
const MAX_REPS: u32 = 1500;

pub fn report(files: &[PathBuf]) {
    println!(
        "{:<26} {:>9} {:>9} {:>6} {:>6} {:>13}",
        "program", "front", "opt-O3", "opt%", "iters", "ops O0->O3"
    );
    println!("{}", "-".repeat(74));

    let mut tot_front = Duration::ZERO;
    let mut tot_opt = Duration::ZERO;
    // Per-pass aggregate over the whole corpus, in pipeline order (index-stable across files).
    let mut agg: Vec<wukong_opt::PassStat> = Vec::new();
    let mut tot_inline = Duration::ZERO;
    let mut tot_unroll = Duration::ZERO;
    let mut measured = 0u32;
    let mut skipped = 0u32;
    let mut failures: Vec<String> = Vec::new();

    for path in files {
        match measure_one(path) {
            Meas::Ran(m) => {
                tot_front += m.front;
                tot_opt += m.opt;
                tot_inline += m.timed.inline;
                tot_unroll += m.timed.unroll;
                merge_passes(&mut agg, &m.timed.per_pass);
                measured += 1;
                println!(
                    "{:<26} {:>9} {:>9} {:>5.1}% {:>6} {:>6}->{:<6}",
                    short(path),
                    fmt(m.front),
                    fmt(m.opt),
                    100.0 * m.opt.as_secs_f64() / (m.front + m.opt).as_secs_f64().max(1e-12),
                    m.timed.max_iterations,
                    m.ops0,
                    m.ops3,
                );
            }
            Meas::Skipped => skipped += 1,
            Meas::Failed(why) => {
                println!("{:<26} *** FAILED: {why} ***", short(path));
                failures.push(format!("{}: {why}", short(path)));
            }
        }
    }

    println!("{}", "-".repeat(74));
    let tot = tot_front + tot_opt;
    println!(
        "{:<26} {:>9} {:>9} {:>5.1}%   (measured {measured}, skipped {skipped})",
        "TOTAL",
        fmt(tot_front),
        fmt(tot_opt),
        100.0 * tot_opt.as_secs_f64() / tot.as_secs_f64().max(1e-12),
    );

    // Per-pass breakdown: where the optimizer spends its time across the whole corpus. Sorted by
    // cost so the hottest pass — the lever — is on top. `inline` is a whole-program prepass and
    // `unroll` a whole-program postpass; both are listed alongside (leave neither out of
    // `timed_total`, or every other pass's share reads high). Times here come from the instrumented run (a small `Instant`-per-call overhead), so
    // they are for *attribution*; the headline opt time above is the clean best-of-N.
    let timed_total: Duration =
        agg.iter().map(|p| p.time).sum::<Duration>() + tot_inline + tot_unroll;
    println!("\nper-pass share of the optimizer (summed over corpus):");
    println!(
        "  {:<14} {:>10} {:>7} {:>10}",
        "pass", "time", "%opt", "calls"
    );
    let mut rows: Vec<(&'static str, Duration, u64)> =
        agg.iter().map(|p| (p.name, p.time, p.calls)).collect();
    rows.push((
        "inline",
        tot_inline,
        if tot_inline > Duration::ZERO { 1 } else { 0 },
    ));
    rows.push((
        "unroll",
        tot_unroll,
        if tot_unroll > Duration::ZERO { 1 } else { 0 },
    ));
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, time, calls) in rows {
        println!(
            "  {:<14} {:>10} {:>6.1}% {:>10}",
            name,
            fmt(time),
            100.0 * time.as_secs_f64() / timed_total.as_secs_f64().max(1e-12),
            calls,
        );
    }

    if !failures.is_empty() {
        eprintln!(
            "\n{} program(s) failed the byte-identical self-check:",
            failures.len()
        );
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}

enum Meas {
    Ran(Sample),
    /// Does not lower to runnable MIR (uses unsupported constructs) — not a regression.
    Skipped,
    /// `optimize_timed` produced different MIR than `optimize` — would mean the timing path is not
    /// measurement-only. Must never happen; guards against future divergence.
    Failed(String),
}

struct Sample {
    front: Duration,
    opt: Duration,
    ops0: usize,
    ops3: usize,
    timed: wukong_opt::Timings,
}

fn measure_one(path: &Path) -> Meas {
    let Ok(src) = std::fs::read_to_string(path) else {
        return Meas::Skipped;
    };

    // One clean lowering to decide runnability and to be the fixed input for optimizer timing.
    let Some(base) = lower(&src) else {
        return Meas::Skipped;
    };
    let (p0, interner) = base;

    // Byte-identical self-check: the instrumented optimizer must match the production one exactly.
    let mut a = p0.clone();
    wukong_opt::optimize(&mut a, LEVEL);
    let mut b = p0.clone();
    let _ = wukong_opt::optimize_timed(&mut b, LEVEL);
    if wukong_mir::print::print_program(&a, &interner)
        != wukong_mir::print::print_program(&b, &interner)
    {
        return Meas::Failed("optimize_timed diverged from optimize".into());
    }

    // ops counts (context for how much MIR each pass chews on).
    let ops0 = count_ops(&p0);
    let ops3 = count_ops(&a);

    // Front-end: parse + sema + mir_build, re-run from source with a fresh interner each rep.
    // Bind the result and stop the clock before dropping it: `let _ = lower(&src);` drops the
    // returned Program+Interner *before* `elapsed()`, charging the front-end for tearing down its
    // own output, while the `opt` measurement below never pays that (its scratch clone outlives the
    // closure). That asymmetry inflates `front`, and the headline `opt%` is a ratio of the two.
    let front = best_of(|| {
        let t0 = Instant::now();
        let out = lower(&src);
        let e = t0.elapsed();
        drop(out);
        e
    });

    // Optimizer: clone the lowered program (untimed), then time `optimize` alone.
    let opt = best_of(|| {
        let mut c = p0.clone();
        let t0 = Instant::now();
        wukong_opt::optimize(&mut c, LEVEL);
        t0.elapsed()
    });

    // One instrumented run for the per-pass attribution.
    let mut c = p0.clone();
    let timed = wukong_opt::optimize_timed(&mut c, LEVEL);

    Meas::Ran(Sample {
        front,
        opt,
        ops0,
        ops3,
        timed,
    })
}

/// Parse + sema + mir_build. `None` if the program does not lower to runnable MIR.
fn lower(src: &str) -> Option<(wukong_mir::Program, Interner)> {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    Some((program, interner))
}

/// Best-of-N: warm up, then keep the minimum single-run time until the wall budget or rep cap. The
/// minimum is the run least perturbed by scheduler/thermal noise — the honest lower bound on cost.
fn best_of<F: FnMut() -> Duration>(mut run: F) -> Duration {
    run();
    run();
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

/// Accumulate one run's per-pass timings into the corpus aggregate, matched by pipeline index.
fn merge_passes(agg: &mut Vec<wukong_opt::PassStat>, run: &[wukong_opt::PassStat]) {
    for (i, s) in run.iter().enumerate() {
        if agg.len() <= i {
            agg.push(wukong_opt::PassStat {
                name: s.name,
                time: Duration::ZERO,
                calls: 0,
            });
        }
        agg[i].name = s.name;
        agg[i].time += s.time;
        agg[i].calls += s.calls;
    }
}

fn count_ops(p: &wukong_mir::Program) -> usize {
    p.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len() + 1)
        .sum()
}

fn fmt(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2}us", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

fn short(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}
