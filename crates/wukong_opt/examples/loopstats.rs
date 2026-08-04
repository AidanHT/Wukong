//! `loopstats` — corpus-wide census of the loop shapes `wukong_opt::loop_info` sees.
//!
//! ```text
//! cargo run -q -p wukong_opt --example loopstats -- [-O<n>] <file-or-dir>...
//! ```
//!
//! A development tool, not part of the compiler. It compiles every `.wk` file given (directories are
//! scanned non-recursively, matching the test harnesses) at one optimization level and counts, over
//! every natural loop of every function, the structural properties a canonicalizer is supposed to
//! guarantee: a preheader, one latch, one exit edge, no abnormal exit, a primary induction variable,
//! a constant step, and a known trip count. Files that do not parse or type-check on their own — the
//! corpus has fixtures that `import` support modules, which this tool does not resolve — are counted
//! as skipped rather than reported as failures.
//!
//! It exists so "canonicalization made loops converge" is a number, not an impression.

use std::path::{Path, PathBuf};

use wukong_opt::loop_info;
use wukong_span::{Interner, SourceId};

#[derive(Default)]
struct Stats {
    files: usize,
    skipped: usize,
    funcs: usize,
    loops: usize,
    preheader: usize,
    one_latch: usize,
    one_exit: usize,
    no_abnormal: usize,
    /// The single exit edge leaves from the header (a top-tested `while`), rather than from a latch
    /// (a rotated do-while).
    exit_in_header: usize,
    exit_in_latch: usize,
    /// Every conditional branch that leaves the loop takes its *false* arm out, so the test always
    /// reads "keep going".
    exit_on_true: usize,
    primary_iv: usize,
    const_step: usize,
    unit_step: usize,
    zero_start: usize,
    trip_const: usize,
    trip_affine: usize,
    /// Every structural property at once — what a vectorizer wants handed to it.
    canonical: usize,
}

impl Stats {
    fn row(&self, label: &str, n: usize) -> String {
        let pct = if self.loops == 0 {
            0.0
        } else {
            100.0 * n as f64 / self.loops as f64
        };
        format!("  {label:<22} {n:5} / {:<5} ({pct:5.1}%)", self.loops)
    }

    fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "files {} (skipped {}), functions {}, loops {}\n",
            self.files, self.skipped, self.funcs, self.loops
        ));
        for (label, n) in [
            ("has preheader", self.preheader),
            ("single latch", self.one_latch),
            ("single exit edge", self.one_exit),
            ("no abnormal exit", self.no_abnormal),
            ("exit tested in header", self.exit_in_header),
            ("exit tested in latch", self.exit_in_latch),
            ("continues on true", self.exit_on_true),
            ("primary IV", self.primary_iv),
            ("constant step", self.const_step),
            ("step == 1", self.unit_step),
            ("start == 0", self.zero_start),
            ("trip: constant", self.trip_const),
            ("trip: affine", self.trip_affine),
            ("FULLY CANONICAL", self.canonical),
        ] {
            s.push_str(&self.row(label, n));
            s.push('\n');
        }
        s
    }
}

fn main() {
    let mut level: u8 = 2;
    let mut inputs: Vec<PathBuf> = Vec::new();
    for arg in std::env::args().skip(1) {
        if let Some(n) = arg.strip_prefix("-O") {
            match n.parse::<u8>() {
                Ok(v) => level = v,
                Err(_) => {
                    eprintln!("loopstats: bad optimization level `{arg}`");
                    std::process::exit(2);
                }
            }
        } else {
            inputs.push(PathBuf::from(arg));
        }
    }
    if inputs.is_empty() {
        eprintln!("usage: loopstats [-O<n>] <file-or-dir>...");
        std::process::exit(2);
    }

    let mut files: Vec<PathBuf> = Vec::new();
    for p in &inputs {
        if p.is_dir() {
            // Non-recursive, like every harness in `crates/wukongc/tests`.
            let mut here: Vec<PathBuf> = std::fs::read_dir(p)
                .unwrap_or_else(|e| panic!("read_dir {}: {e}", p.display()))
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "wk"))
                .collect();
            here.sort();
            files.extend(here);
        } else {
            files.push(p.clone());
        }
    }

    let mut stats = Stats::default();
    for path in &files {
        stats.files += 1;
        match census(path, level, &mut stats) {
            Ok(()) => {}
            Err(_) => stats.skipped += 1,
        }
    }
    print!("-O{level}\n{}", stats.report());
}

fn census(path: &Path, level: u8, stats: &mut Stats) -> Result<(), String> {
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return Err("parse".into());
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Err("sema".into());
    }
    let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    wukong_opt::optimize(&mut program, level);

    for f in &program.funcs {
        stats.funcs += 1;
        let forest = loop_info::analyze_function(f);
        for l in &forest.loops {
            stats.loops += 1;
            let preheader = l.preheader.is_some();
            let one_latch = l.latches.len() == 1;
            let one_exit = l.exits.len() == 1;
            let no_abnormal = l.abnormal_exits.is_empty();
            stats.preheader += preheader as usize;
            stats.one_latch += one_latch as usize;
            stats.one_exit += one_exit as usize;
            stats.no_abnormal += no_abnormal as usize;
            if one_exit {
                let from = l.exits[0].0;
                if from == l.header {
                    stats.exit_in_header += 1;
                } else if l.latches.contains(&from) {
                    stats.exit_in_latch += 1;
                }
            }
            // Polarity, over *every* exit edge (a `break` in the body is an exit too).
            let inside: std::collections::HashSet<u32> =
                l.blocks.iter().map(|b| b.0).collect();
            let all_on_true = !l.exits.is_empty()
                && l.exits.iter().all(|&(from, _)| {
                    match &f.blocks[from.0 as usize].term {
                        wukong_mir::Terminator::CondBr {
                            then_blk, else_blk, ..
                        } => inside.contains(&then_blk.0) && !inside.contains(&else_blk.0),
                        _ => false,
                    }
                });
            stats.exit_on_true += all_on_true as usize;
            let iv = l.primary();
            stats.primary_iv += iv.is_some() as usize;
            let const_step = iv.and_then(|iv| iv.step.as_const());
            stats.const_step += const_step.is_some() as usize;
            stats.unit_step += (const_step == Some(1)) as usize;
            let zero_start = iv
                .and_then(|iv| iv.start)
                .map(|s| is_zero_const(f, s))
                .unwrap_or(false);
            stats.zero_start += zero_start as usize;
            match &l.trip {
                loop_info::TripCount::Const(_) => stats.trip_const += 1,
                loop_info::TripCount::Affine { .. } => stats.trip_affine += 1,
                loop_info::TripCount::Unknown => {}
            }
            if preheader
                && one_latch
                && one_exit
                && no_abnormal
                && const_step == Some(1)
                && zero_start
                && !matches!(l.trip, loop_info::TripCount::Unknown)
            {
                stats.canonical += 1;
            }
        }
    }
    Ok(())
}

/// Is `v` the integer literal 0, wherever it is defined?
fn is_zero_const(f: &wukong_mir::Function, v: wukong_mir::ValueId) -> bool {
    f.blocks.iter().flat_map(|b| &b.insts).any(|i| {
        i.result == Some(v) && matches!(&i.op, wukong_mir::Op::ConstInt(0, ty) if ty.is_int())
    })
}
