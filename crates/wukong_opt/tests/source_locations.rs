//! The MIR source-location table (`wukong_mir::SourceLocs`) through the optimizer.
//!
//! Lowering locates every value of a source function. The passes that create values as copies of
//! located source carry the original's location onto the copy: inlining (the callee's body, spliced
//! into its caller), loop unrolling and unroll-and-jam (each copy of the body), bounds-check
//! versioning and unswitching (each copy of the loop) and jump threading (the private copy of a
//! block). The vectorizer gives each widened step its scalar step's location (each lane's copy of a
//! lane-serial step and each instruction of an outer-vectorized nest included), a lane test its
//! bounds check's, and the guard, trip and exit it adds the loop's. Every other new value stays
//! unknown, which is sound (an entry may be lost, never invented). The corpus census below measures
//! what survives `-O2` across `tests/run`, and pins it from below.

use std::path::{Path, PathBuf};

use wukong_mir::{BinOp, Function, MirType, Op, Program, ValueId};
use wukong_span::{Interner, SourceId, SourceMap};

/// Parse, check and lower `src` with the default options; `None` when any stage reports an error.
fn lower(src: &str) -> Option<(Program, Interner)> {
    lower_with(src, wukong_mir_build::LowerOptions::default())
}

/// [`lower`] with the given lowering options.
fn lower_with(src: &str, options: wukong_mir_build::LowerOptions) -> Option<(Program, Interner)> {
    let mut interner = Interner::new();
    let (module, parsed) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if parsed.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, checked) = wukong_sema::check(&module, &interner);
    if checked.iter().any(|d| d.is_error()) {
        return None;
    }
    let (program, lowered) =
        wukong_mir_build::lower_program_with(&module, &sema, &mut interner, options);
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

/// (c) A loop the optimizer's vectorizer widens keeps the scalar body's line: every vector value
/// of the widened loop is a lane-wise copy of a scalar step on line 4 — the loads, the compare,
/// the multiply, the select that merges the two arms' stores and the splatted `a`. Lowering leaves
/// a branchy body scalar; the MIR vectorizer if-converts and widens it (with the 256-bit kernels
/// off, in lowering and in the optimizer alike, so the 128-bit form is what it emits).
#[test]
fn vectorized_values_keep_the_scalar_lines() {
    let src = "fn clip(x: [f32; 64], mut y: [f32; 64], a: f32) {
    let mut i: i64 = 0;
    while i < 64 {
        if x[i] > a { y[i] = x[i] * a; } else { y[i] = x[i]; }
        i = i + 1;
    }
}
";
    let no_kernels = wukong_mir_build::LowerOptions {
        vec_kernels: false,
        ..Default::default()
    };
    let (mut p, interner) = lower_with(src, no_kernels).expect("the test program compiles");
    let lowered = func(&p, &interner, "clip");
    assert!(
        !results(lowered).any(|(v, _)| matches!(lowered.value_type(v), MirType::Vec(..))),
        "lowering left the loop scalar"
    );
    wukong_opt::optimize_preserving_with(
        &mut p,
        2,
        &[],
        wukong_opt::OptimizeOptions::FULL.with_vec_kernels(false),
    );
    assert!(wukong_mir::verify::verify_program(&p).is_empty());
    let mut sm = SourceMap::new();
    sm.add("t.wk", src);
    let f = func(&p, &interner, "clip");
    let vectors: Vec<ValueId> = results(f)
        .filter(|(v, _)| matches!(f.value_type(*v), MirType::Vec(..)))
        .map(|(v, _)| v)
        .collect();
    assert!(!vectors.is_empty(), "the loop was widened");
    for v in vectors {
        assert_eq!(line(&sm, f, v), Some(4), "{v:?} widens the body's line");
    }
}

/// (e) A nest widened across its rows (`vectorize::outer`): lane `k` runs row `r + k`'s inner
/// loop, so each vector value is a lane-wise copy of a scalar step of the nest and keeps that
/// step's line — the row sum's multiply-add is line 8's.
#[test]
fn outer_vectorized_values_keep_the_nest_lines() {
    let src = "\
@unchecked
fn mv(a: []f32, x: []f32, @noalias mut y: []f32, rows: i64, cols: i64) {
    let mut r: i64 = 0;
    while r < rows {
        let mut s: f32 = 0.0;
        let mut c: i64 = 0;
        while c < cols {
            s = s + a[r * cols + c] * x[c];
            c = c + 1;
        }
        y[r] = s;
        r = r + 1;
    }
}
";
    let (p, interner, sm) = optimized(src, 2);
    let f = func(&p, &interner, "mv");
    assert!(
        f.blocks
            .iter()
            .flat_map(|b| &b.params)
            .any(|v| f.value_type(*v).is_vector()),
        "the nest was not widened across its rows (no vector accumulator)"
    );
    let vectors: Vec<(ValueId, &Op)> = results(f)
        .filter(|(v, _)| f.value_type(*v).is_vector())
        .collect();
    let sum = |op: &Op| matches!(op, Op::Fma(..) | Op::Bin(BinOp::FAdd | BinOp::FMul, ..));
    assert!(vectors.iter().any(|(_, op)| sum(op)), "no vector row sum");
    for (v, op) in vectors {
        assert!(f.value_loc(v).is_some(), "{v:?} = {op:?} has no location");
        if sum(op) {
            assert_eq!(
                line(&sm, f, v),
                Some(8),
                "{v:?} = {op:?} is the row sum's step"
            );
        }
    }
}

/// (f) A lane-serial region runs its steps lane by lane, in iteration order, in the vector body:
/// each lane's copy of a step keeps that step's line. The region runs from the gather `h[j]` on
/// line 8 to the scatter `h[k]` on line 9 (two indices, so the two accesses keep two addresses);
/// line 9, the update's add and store, reads no memory, so no load of the function may sit on it.
#[test]
fn lane_serial_copies_keep_their_own_lines() {
    let src = "\
@unchecked
fn move_add(@noalias mut h: []f32, q: []i32, p: []i32, x: []f32, y: []f32, n: i64) {
    let mut i: i64 = 0;
    while i < n {
        let v: f32 = x[i] * 0.5 + y[i] * 0.25 - 1.0;
        let j: i64 = q[i] as i64;
        let k: i64 = p[i] as i64;
        let g: f32 = h[j];
        h[k] = g + v * v;
        i = i + 1;
    }
}
";
    let (p, interner, sm) = optimized(src, 2);
    let f = func(&p, &interner, "move_add");
    assert!(
        results(f).any(|(v, _)| f.value_type(v).is_vector()),
        "the loop was not widened around its lane-serial region"
    );
    let mut gathers = 0;
    for (v, op) in results(f) {
        if !matches!(op, Op::Load(..)) {
            continue;
        }
        assert_ne!(
            line(&sm, f, v),
            Some(9),
            "{v:?} = {op:?}: line 9 loads nothing"
        );
        if line(&sm, f, v) == Some(8) {
            gathers += 1;
        }
    }
    assert!(
        gathers >= 5,
        "{gathers} loads on line 8: four lanes of the vector body and the scalar loop's"
    );
}

/// (g) A checked gather's lane test — the group's indices tested at once and the mask folded to
/// one branch — sits on its bounds check's line, 4, like every other vector value of the body; the
/// loop's own line, 3, is the search exit's.
#[test]
fn a_lane_test_keeps_its_bounds_check_line() {
    let src = "\
fn g(t: []f32, q: []i32, x: []f32, @noalias mut o: []f32, n: i64) {
    let mut i: i64 = 0;
    while i < n {
        o[i] = t[q[i] as i64] * x[i] + 1.0;
        i = i + 1;
    }
}
";
    let (p, interner, sm) = optimized(src, 2);
    let f = func(&p, &interner, "g");
    let vectors: Vec<(ValueId, &Op)> = results(f)
        .filter(|(v, _)| f.value_type(*v).is_vector())
        .collect();
    assert!(
        vectors
            .iter()
            .any(|(_, op)| matches!(op, Op::Shuffle { .. })),
        "no lane test folded its mask: {vectors:?}"
    );
    for (v, op) in vectors {
        assert_eq!(
            line(&sm, f, v),
            Some(4),
            "{v:?} = {op:?} is the body's line"
        );
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
/// Located share of result-bearing instructions after `-O2`, in whole percent: measured 96.48%
/// (120499 of 124890 over 608 programs, with the SLP vectorizer in the pipeline), rounded down.
/// The rest are values the optimizer synthesizes with no source of their own: the
/// unroller's wrap-safe limits and exit tests, the bounds-check and unswitch guards, LICM's
/// selects, SLP packs. A pass that clones located code should copy its locations
/// (`Function::copy_value_loc`), as inlining, unrolling and the loop-copying passes do.
const O2_FLOOR_PERCENT: usize = 96;

/// (d) Over the whole run corpus: lowering locates every result-bearing instruction of every
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
