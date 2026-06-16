//! `mercury-bench` — a tiny benchmark harness.
//!
//! For each `.mer` program it can lower, it reports:
//!   * MIR instruction count at `-O0` vs `-O3` and the reduction the optimizer achieved;
//!   * the time to run the program through the interpreter (median of a few iterations).
//!
//! This quantifies the optimizer's effect and gives a regression signal for codegen/interpreter
//! performance, all without an LLVM toolchain. Run with:
//!
//! ```text
//! cargo run -p mercury_bench --release -- tests/run examples
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

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
        "{:<22} {:>8} {:>8} {:>9}  {:>12}",
        "program", "O0 inst", "O3 inst", "reduction", "run (median)"
    );
    println!("{}", "-".repeat(66));

    let (mut tot0, mut tot3) = (0usize, 0usize);
    for f in &files {
        match bench_one(f) {
            Some(r) => {
                tot0 += r.insts_o0;
                tot3 += r.insts_o3;
                let pct = if r.insts_o0 == 0 {
                    0.0
                } else {
                    100.0 * (r.insts_o0 - r.insts_o3) as f64 / r.insts_o0 as f64
                };
                println!(
                    "{:<22} {:>8} {:>8} {:>8.1}%  {:>10.1?}",
                    short_name(f),
                    r.insts_o0,
                    r.insts_o3,
                    pct,
                    r.run_median
                );
            }
            None => println!(
                "{:<22} {:>8}",
                short_name(f),
                "(skipped: does not lower or run cleanly)"
            ),
        }
    }

    println!("{}", "-".repeat(66));
    let tot_pct = if tot0 == 0 {
        0.0
    } else {
        100.0 * (tot0 - tot3) as f64 / tot0 as f64
    };
    println!("{:<22} {:>8} {:>8} {:>8.1}%", "TOTAL", tot0, tot3, tot_pct);
}

struct Result {
    insts_o0: usize,
    insts_o3: usize,
    run_median: std::time::Duration,
}

fn bench_one(path: &Path) -> Option<Result> {
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

    let main = interner.intern("main");
    // Confirm it actually runs (and that O0/O3 agree) before timing.
    let r0 = mercury_interp::run(&p3, main, &interner).ok()?;
    let r_check = mercury_interp::run(&p0, main, &interner).ok()?;
    if r0 != r_check {
        eprintln!("warning: {} differs O0 vs O3", short_name(path));
        return None;
    }

    let mut times = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        let _ = mercury_interp::run(&p3, main, &interner);
        times.push(t.elapsed());
    }
    times.sort();

    Some(Result {
        insts_o0: count_insts(&p0),
        insts_o3: count_insts(&p3),
        run_median: times[times.len() / 2],
    })
}

fn count_insts(p: &mercury_mir::Program) -> usize {
    p.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len())
        .sum()
}

fn short_name(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}
