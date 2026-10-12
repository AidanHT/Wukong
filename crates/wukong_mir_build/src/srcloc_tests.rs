//! The MIR source-location table (`wukong_mir::SourceLocs`) as lowering fills it.
//!
//! Every value `mir_build` creates is attributed to the innermost expression, place or statement
//! being lowered when it was created, and every source function starts at its declaration — so
//! straight out of lowering, every value of a source function has a location. These tests pin
//! that, plus the attribution of the constructs a diagnostic is most likely to point at: a `let`
//! slot, an arithmetic operation (to the column), a call, a parameter, a `@parallel` region and a
//! generic instance.

use wukong_mir::{BinOp, Function, Op, Program, ValueId};
use wukong_span::{Interner, Location, SourceId, SourceMap, Span};

use crate::{lower_program_with, LowerOptions};

/// `src` lowered with `opts` (sema and lowering must be clean), with a source map holding the same
/// text under the `SourceId` the parser was given.
fn lowered_with(src: &str, opts: LowerOptions) -> (Program, Interner, SourceMap) {
    let mut sm = SourceMap::new();
    let id = sm.add("t.wk", src);
    assert_eq!(id, SourceId(0));
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, id, &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, ldiags) = lower_program_with(&module, &sema, &mut interner, opts);
    assert!(ldiags.iter().all(|d| !d.is_error()), "lower: {ldiags:?}");
    (prog, interner, sm)
}

fn lowered(src: &str) -> (Program, Interner, SourceMap) {
    lowered_with(src, LowerOptions::default())
}

fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    prog.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

/// Every value an instruction produces, with the instruction's op.
fn results(f: &Function) -> impl Iterator<Item = (ValueId, &Op)> {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| i.result.map(|r| (r, &i.op)))
}

fn loc(sm: &SourceMap, f: &Function, v: ValueId) -> Location {
    let span = f
        .value_loc(v)
        .unwrap_or_else(|| panic!("{v:?} has no source location"));
    sm.span_location(span)
}

/// (a) Every result-bearing instruction of every function lowered from source has a location.
fn assert_fully_located(prog: &Program, interner: &Interner) {
    for f in &prog.funcs {
        assert!(
            f.decl_span().is_some(),
            "`{}` has no declaration span",
            interner.resolve(f.name)
        );
        for (v, op) in results(f) {
            assert!(
                f.value_loc(v).is_some(),
                "`{}`: {v:?} = {op:?} has no source location",
                interner.resolve(f.name)
            );
        }
        for &p in &f.params {
            assert!(f.value_loc(p).is_some(), "parameter {p:?} has no location");
        }
    }
}

const ADD: &str = "\
fn add(a: f32, b: f32) -> f32 {
    let s = a + b;
    return s;
}

fn main() -> i32 {
    let x: f32 = 1.5;
    let y = add(x, 2.0);
    print(y);
    return 0;
}
";

#[test]
fn every_lowered_value_has_a_location() {
    let (prog, interner, _) = lowered(ADD);
    assert_fully_located(&prog, &interner);
}

/// (b) A `let` slot maps to its statement's line, `a + b` to its own line and column, a call
/// result to the call's line, and the parameters to the declaration, which is the function's name.
#[test]
fn lets_operations_calls_and_params_map_to_their_source() {
    let (prog, interner, sm) = lowered(ADD);
    let add = func(&prog, &interner, "add");

    let decl = add.decl_span().expect("a declaration span");
    assert_eq!(sm.span_text(decl), "add");
    for &p in &add.params {
        assert_eq!(add.value_loc(p), Some(decl), "parameter {p:?}");
    }

    let (sum, _) = results(add)
        .find(|(_, op)| matches!(op, Op::Bin(BinOp::FAdd, ..)))
        .expect("an `fadd` for `a + b`");
    assert_eq!(loc(&sm, add, sum), Location { line: 2, col: 13 });
    assert_eq!(sm.span_text(add.value_loc(sum).unwrap()), "a + b");

    // `s`'s slot is the one the sum is stored into.
    let slot = add
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find_map(|i| match i.op {
            Op::Store { ptr, value } if value == sum => Some(ptr),
            _ => None,
        })
        .expect("the sum is stored into `s`");
    assert!(
        matches!(
            results(add).find(|(v, _)| *v == slot),
            Some((_, Op::Alloca { .. }))
        ),
        "`s` lives in an alloca"
    );
    assert_eq!(loc(&sm, add, slot).line, 2);

    let main = func(&prog, &interner, "main");
    let (call, _) = results(main)
        .find(|(_, op)| matches!(op, Op::Call { func, .. } if interner.resolve(*func) == "add"))
        .expect("the call to `add`");
    assert_eq!(loc(&sm, main, call).line, 8);
    assert_eq!(sm.span_text(main.decl_span().unwrap()), "main");
}

/// (c) A `@parallel` loop that is not the whole body is outlined into a region function whose
/// declaration is the loop, and whose values all come from inside it.
#[test]
fn a_parallel_region_is_declared_by_its_loop() {
    let src = "\
@parallel
fn scaled(x: [i32; 4096], mut q: [f32; 4096]) {
    let k: f32 = 3.0;
    for i in 0..4096 {
        q[i] = (x[i] as f32) * k;
    }
}

fn main() -> i32 {
    let x: [i32; 4096] = [2; 4096];
    let mut q: [f32; 4096] = [0.0; 4096];
    scaled(x, q);
    print(q[7]);
    return 0;
}
";
    // Recognition off, so the loop is outlined rather than replaced by a kernel call. The two
    // parameters have different element types, so the region's independence proof need not treat
    // them as one buffer.
    let opts = LowerOptions {
        recognize: false,
        ..LowerOptions::default()
    };
    let (prog, interner, sm) = lowered_with(src, opts);
    assert_fully_located(&prog, &interner);
    let region = prog
        .funcs
        .iter()
        .find(|f| interner.resolve(f.name).starts_with("wukong$par$"))
        .expect("the loop is outlined into a region function");
    let decl = region.decl_span().expect("the region has a declaration");
    assert_eq!(sm.span_location(decl).line, 4, "declared by the loop");
    assert!(sm.span_text(decl).starts_with("for i in 0..4096"));
    let (lo, hi) = (4, 6);
    for (v, op) in results(region) {
        let line = loc(&sm, region, v).line;
        assert!(
            (lo..=hi).contains(&line),
            "{v:?} = {op:?} is attributed to line {line}, outside the loop"
        );
    }
    // The multiply is the body's own.
    let (mul, _) = results(region)
        .find(|(_, op)| matches!(op, Op::Bin(BinOp::FMul, ..)))
        .expect("the body's `fmul`");
    assert_eq!(loc(&sm, region, mul), Location { line: 5, col: 16 });
}

/// (d) Every instance of a type-generic function carries the template's lines.
#[test]
fn generic_instances_carry_the_template_lines() {
    let src = "\
fn twice<T>(x: T) -> T {
    return x + x;
}

fn main() -> i32 {
    let a = twice(3);
    let b = twice(1.5);
    print(a);
    print(b);
    return 0;
}
";
    let (prog, interner, sm) = lowered(src);
    assert_fully_located(&prog, &interner);
    let instances: Vec<&Function> = prog
        .funcs
        .iter()
        .filter(|f| interner.resolve(f.name).starts_with("twice$"))
        .collect();
    assert_eq!(instances.len(), 2, "one instance per type");
    for f in instances {
        let decl = f.decl_span().unwrap();
        assert_eq!(sm.span_text(decl), "twice");
        assert_eq!(sm.span_location(decl).line, 1);
        let (sum, _) = results(f)
            .find(|(_, op)| matches!(op, Op::Bin(BinOp::Add | BinOp::FAdd, ..)))
            .expect("the instance's add");
        assert_eq!(loc(&sm, f, sum), Location { line: 2, col: 12 });
    }
}

/// A function built with no source declaration (a dummy decl) records no locations at all: an
/// unknown entry, never an invented one.
#[test]
fn a_function_without_a_source_has_no_locations() {
    let mut i = Interner::new();
    let mut b = wukong_mir::Builder::new(i.intern("synth"), wukong_mir::MirType::I32);
    b.set_decl(Span::dummy());
    let z = b.build(
        wukong_mir::MirType::I32,
        Op::ConstInt(0, wukong_mir::MirType::I32),
    );
    b.ret(Some(z));
    let f = b.finish();
    assert_eq!(f.decl_span(), None);
    assert_eq!(f.value_loc(z), None);
}
