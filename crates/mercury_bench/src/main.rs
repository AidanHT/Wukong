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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
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

    println!(
        "{:<20} {:>7} {:>7} {:>9}  {:>11} {:>11} {:>8}",
        "program", "O0 ops", "O3 ops", "reduction", "O0 time", "O3 time", "speedup"
    );
    println!("{}", "-".repeat(80));

    let (mut tot0, mut tot3) = (0usize, 0usize);
    let mut log_speedup_sum = 0.0f64;
    let mut speedup_n = 0u32;
    for f in &files {
        match bench_one(f) {
            Some(r) => {
                tot0 += r.ops0;
                tot3 += r.ops3;
                let pct = reduction(r.ops0, r.ops3);
                let speedup = r.t0.as_secs_f64() / r.t3.as_secs_f64().max(1e-12);
                log_speedup_sum += speedup.ln();
                speedup_n += 1;
                println!(
                    "{:<20} {:>7} {:>7} {:>8.1}%  {:>11} {:>11} {:>7.2}x",
                    short_name(f),
                    r.ops0,
                    r.ops3,
                    pct,
                    fmt_dur(r.t0),
                    fmt_dur(r.t3),
                    speedup,
                );
            }
            None => println!(
                "{:<20} {:>7}",
                short_name(f),
                "(skipped: does not lower / run / agree)"
            ),
        }
    }

    println!("{}", "-".repeat(80));
    // Geometric mean of the per-program speedups (the right average for ratios).
    let geo = if speedup_n > 0 {
        (log_speedup_sum / speedup_n as f64).exp()
    } else {
        0.0
    };
    println!(
        "{:<20} {:>7} {:>7} {:>8.1}%  {:>11} {:>11} {:>7.2}x",
        "TOTAL / geomean",
        tot0,
        tot3,
        reduction(tot0, tot3),
        "",
        "",
        geo,
    );
}

struct Res {
    ops0: usize,
    ops3: usize,
    t0: Duration,
    t3: Duration,
}

fn reduction(a: usize, b: usize) -> f64 {
    if a == 0 {
        0.0
    } else {
        100.0 * (a - b) as f64 / a as f64
    }
}

fn bench_one(path: &Path) -> Option<Res> {
    let src = std::fs::read_to_string(path).ok()?;
    let mut interner = Interner::new();
    let (module, pd) = mercury_parser::parse_module(&src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = mercury_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }

    let (mut p0, ld) = mercury_mir_build::lower_program(&module, &sema, &interner);
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    let mut p3 = p0.clone();
    mercury_opt::optimize(&mut p0, 0);
    mercury_opt::optimize(&mut p3, 3);

    // The optimizer must preserve well-formedness and behavior.
    for f in p0.funcs.iter().chain(p3.funcs.iter()) {
        if !mercury_mir::verify::verify_function(f).is_empty() {
            eprintln!("warning: {} fails MIR verification", short_name(path));
            return None;
        }
    }
    let main = interner.intern("main");
    let r0 = mercury_interp::run(&p0, main, &interner).ok()?;
    let r3 = mercury_interp::run(&p3, main, &interner).ok()?;
    if r0 != r3 {
        eprintln!("warning: {} differs O0 vs O3", short_name(path));
        return None;
    }

    Some(Res {
        ops0: count_ops(&p0),
        ops3: count_ops(&p3),
        t0: time_run(&p0, main, &interner),
        t3: time_run(&p3, main, &interner),
    })
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
