//! The MIR source-location table (`wukong_mir::SourceLocs`) through the optimizer.
//!
//! Lowering locates every value of a source function. Two passes create values that are copies of
//! located source — inlining (the callee's body, spliced into its caller) and loop unrolling (each
//! copy of the body) — and both carry the original's location onto the copy. Every other pass's new
//! values stay unknown, which is sound (an entry may be lost, never invented). The corpus census
//! below measures what survives `-O2` across `tests/run`, and pins it from below.

use std::path::{Path, PathBuf};

use wukong_mir::{BinOp, Function, Op, Program, ValueId};
use wukong_span::{Interner, SourceId, SourceMap};

/// Parse, check and lower `src` with the default options; `None` when any stage reports an error.
fn lower(src: &str) -> Option<(Program, Interner)> {
    let mut interner = Interner::new();
    let (module, parsed) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if parsed.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, checked) = wukong_sema::check(&module, &interner);
    if checked.iter().any(|d| d.is_error()) {
        return None;
    }
    let (program, lowered) = wukong_mir_build::lower_program_with(
        &module,
        &sema,
        &mut interner,
        wukong_mir_build::LowerOptions::default(),
    );
    if lowered.iter().any(|d| d.is_error()) {
        return None;
    }
    Some((program, interner))
}

fn optimized(src: &str, level: u8) -> (Program, Interner, SourceMap) {
    let (mut program, interner) = lower(src).expect("the test program compiles");
    wukong_opt::optimize(&mut program, level);
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    let mut sm = SourceMap::new();
    sm.add("t.wk", src);
    (program, interner, sm)
}

fn func<'p>(p: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    p.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

fn results(f: &Function) -> impl Iterator<Item = (ValueId, &Op)> {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| i.result.map(|r| (r, &i.op)))
}

fn line(sm: &SourceMap, f: &Function, v: ValueId) -> Option<u32> {
    f.value_loc(v).map(|s| sm.span_location(s).line)
}

/// (a) A callee inlined into its caller keeps its own source lines: the multiply in `first` is
/// `sq`'s `x * x`. The argument is a buffer parameter, so nothing folds it away.
#[test]
fn inlined_values_keep_the_callee_lines() {
    let src = "\
fn sq(x: f32) -> f32 {
    return x * x;
}

fn first(a: [f32; 4]) -> f32 {
    return sq(a[0]) + a[1];
}
";
    let (p, interner, sm) = optimized(src, 2);
    let first = func(&p, &interner, "first");
    assert!(
        !results(first).any(|(_, op)| matches!(op, Op::Call { .. })),
        "`sq` was inlined"
    );
    let muls: Vec<ValueId> = results(first)
        .filter(|(_, op)| matches!(op, Op::Bin(BinOp::FMul, ..)))
        .map(|(v, _)| v)
        .collect();
    assert!(!muls.is_empty(), "the inlined multiply is in `first`");
    for v in muls {
        assert_eq!(line(&sm, first, v), Some(2), "{v:?} is `sq`'s `x * x`");
    }
    let add = results(first)
        .find(|(_, op)| matches!(op, Op::Bin(BinOp::FAdd, ..)))
        .map(|(v, _)| v)
        .expect("the caller's own add");
    assert_eq!(line(&sm, first, add), Some(6));
}

/// (b) Every unrolled copy of a loop body carries the body's line. A loop-carried recurrence
/// under `@generic` keeps the vectorizer and the recognizers away, so the unroller takes it.
#[test]
fn unrolled_copies_keep_the_loop_body_line() {
    let src = "\
@generic
fn horner(a: [f32; 64], k: f32) -> f32 {
    let mut s: f32 = 0.0;
    for i in 0..64 {
        s = s * k + a[i];
    }
    return s;
}
";
    let (p, interner, sm) = optimized(src, 2);
    let f = func(&p, &interner, "horner");
    // `s * k + a[i]` contracts to one `fma` (or stays a multiply and an add): either way the
    // recurrence's operation appears once per copy.
    let copies = [
        results(f)
            .filter(|(_, op)| matches!(op, Op::Fma(..)))
            .count(),
        results(f)
            .filter(|(_, op)| matches!(op, Op::Bin(BinOp::FMul, ..)))
            .count(),
    ];
    assert!(
        copies.iter().any(|&n| n > 1),
        "the loop body was not unrolled: (fma, fmul) counts {copies:?}"
    );
    let body: Vec<ValueId> = results(f)
        .filter(|(_, op)| {
            matches!(
                op,
                Op::Fma(..) | Op::Bin(BinOp::FMul | BinOp::FAdd, ..) | Op::Load(..)
            )
        })
        .map(|(v, _)| v)
        .collect();
    for v in body {
        assert_eq!(line(&sm, f, v), Some(5), "{v:?} is a copy of the loop body");
    }
}

fn run_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

/// (located, total) result-bearing instructions over the functions lowered from a source
/// declaration.
fn coverage(p: &Program) -> (usize, usize) {
    let mut located = 0;
    let mut total = 0;
    for f in p.funcs.iter().filter(|f| f.decl_span().is_some()) {
        for (v, _) in results(f) {
            total += 1;
            located += usize::from(f.value_loc(v).is_some());
        }
    }
    (located, total)
}

/// Programs in `tests/run` that do not lower standalone here, without the driver — measured 1:
/// `type_alias.wk`, whose `type` aliases the driver's loader expands before sema. The allowance
/// keeps an unrelated fixture of that kind from failing this census.
const MAX_SKIPPED: usize = 3;
/// Located share of result-bearing instructions after `-O2`, in whole percent: measured 74.78%
/// (78886 of 105488 over 554 programs, with the SLP vectorizer in the pipeline), rounded down.
/// The rest are the optimizer's own new values (vector widening, SLP packs, scalar replacement,
/// loop canonicalization), which carry no location by design. A pass that clones located code
/// should copy its locations (`Function::copy_value_loc`), as inlining and unrolling do.
const O2_FLOOR_PERCENT: usize = 74;

/// (c) Over the whole run corpus: lowering locates every result-bearing instruction of every
/// source function, and `-O2` keeps at least the measured share of them located.
#[test]
fn corpus_census() {
    let mut programs: Vec<PathBuf> = std::fs::read_dir(run_dir())
        .expect("tests/run")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wk"))
        .collect();
    programs.sort();
    let started = std::time::Instant::now();
    let (mut checked, mut skipped) = (0usize, Vec::new());
    let (mut o0, mut o2) = ((0usize, 0usize), (0usize, 0usize));
    for path in &programs {
        let src = std::fs::read_to_string(path).unwrap();
        // Multi-file programs need the driver's loader.
        if src.lines().any(|l| l.trim_start().starts_with("import ")) {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Some((mut p, _)) = lower(&src) else {
            skipped.push(name);
            continue;
        };
        checked += 1;
        let (located, total) = coverage(&p);
        assert_eq!(
            located,
            total,
            "{name}: {} result(s) of source functions have no location straight out of lowering",
            total - located
        );
        o0.0 += located;
        o0.1 += total;
        wukong_opt::optimize(&mut p, 2);
        let (located, total) = coverage(&p);
        o2.0 += located;
        o2.1 += total;
    }
    let pct = |(l, t): (usize, usize)| {
        if t == 0 {
            100.0
        } else {
            100.0 * l as f64 / t as f64
        }
    };
    eprintln!(
        "source-location census: {checked} programs (skipped: {skipped:?}) in {:.1?}; \
         -O0 {}/{} = {:.2}%, -O2 {}/{} = {:.2}%",
        started.elapsed(),
        o0.0,
        o0.1,
        pct(o0),
        o2.0,
        o2.1,
        pct(o2)
    );
    assert!(checked > 0);
    assert!(
        skipped.len() <= MAX_SKIPPED,
        "{} programs no longer lower standalone: {skipped:?}",
        skipped.len()
    );
    assert!(
        pct(o2) >= O2_FLOOR_PERCENT as f64,
        "-O2 location coverage {:.2}% fell below the {O2_FLOOR_PERCENT}% floor",
        pct(o2)
    );
}
