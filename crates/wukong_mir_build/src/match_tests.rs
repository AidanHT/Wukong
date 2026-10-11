//! The untested last arm of a proven-exhaustive `match` (N29).
//!
//! Sema publishes the `match` expressions pattern usefulness proved exhaustive over a type whose
//! zero value it models (`SemaResult::exhaustive_matches`); `lower_match` takes the last unguarded
//! arm of each without a test. These tests count what reaches MIR straight out of lowering (no
//! optimization): the arm compares and the `unreachable` trap a no-arm-matched fall-through ends
//! in. A proven match has one compare fewer than it has tested arms and no trap; an unproven one
//! keeps both.

use wukong_mir::{CmpOp, Function, Op, Program, Terminator};
use wukong_span::{Interner, SourceId};

use crate::lower_program;

/// Parse, check (which must be clean) and lower `src`; lowering must be clean too.
fn lowered(src: &str) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, ldiags) = lower_program(&module, &sema, &mut interner);
    assert!(ldiags.iter().all(|d| !d.is_error()), "lower: {ldiags:?}");
    (prog, interner)
}

/// The functions lowered from source function `name`: the function itself, or each of its
/// monomorphized instances (`name$…`).
fn funcs<'p>(prog: &'p Program, interner: &Interner, name: &str) -> Vec<&'p Function> {
    let inst = format!("{name}$");
    let found: Vec<&Function> = prog
        .funcs
        .iter()
        .filter(|f| {
            let n = interner.resolve(f.name);
            n == name || n.starts_with(&inst)
        })
        .collect();
    assert!(!found.is_empty(), "no function `{name}`");
    found
}

/// `(cmp.eq count, unreachable-terminator count)` of `f`.
fn tests_and_traps(f: &Function) -> (usize, usize) {
    let eqs = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i.op, Op::Cmp(CmpOp::Eq, ..)))
        .count();
    let traps = f
        .blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::Unreachable))
        .count();
    (eqs, traps)
}

/// The counts of the single function `name` lowers to.
fn counts(src: &str, name: &str) -> (usize, usize) {
    let (prog, interner) = lowered(src);
    let fs = funcs(&prog, &interner, name);
    assert_eq!(fs.len(), 1, "`{name}` lowers to one function");
    tests_and_traps(fs[0])
}

const MAIN: &str = "fn main() -> i32 { return 0; }";

#[test]
fn a_five_variant_enum_match_tests_four_arms() {
    let src = format!(
        "enum Op {{ A, B, C, D, E }}
         fn f(o: Op) -> i32 {{
             return match o {{ Op::A => 1, Op::B => 2, Op::C => 3, Op::D => 4, Op::E => 5 }};
         }}
         {MAIN}"
    );
    assert_eq!(counts(&src, "f"), (4, 0), "4 compares, no trap");
}

#[test]
fn an_exhaustive_bool_match_is_one_branch() {
    let src = format!(
        "fn f(b: bool) -> i32 {{ return match b {{ true => 7, false => 9 }}; }}
         {MAIN}"
    );
    assert_eq!(counts(&src, "f"), (1, 0));
}

#[test]
fn an_enum_with_no_zero_discriminant_keeps_its_trap() {
    // `let e: E;` is zero-filled, and no variant is 0: the analysis does not model that value, so
    // the match is not proven and every arm keeps its test.
    let src = format!(
        "enum E {{ A = 1, B = 2 }}
         fn f(e: E) -> i32 {{ return match e {{ E::A => 10, E::B => 20 }}; }}
         {MAIN}"
    );
    assert_eq!(counts(&src, "f"), (2, 1));
}

#[test]
fn a_struct_holding_such_an_enum_keeps_its_trap() {
    let src = format!(
        "enum E {{ A = 1, B = 2 }}
         struct S {{ e: E, k: i32 }}
         fn f(s: S) -> i32 {{
             return match s {{ S {{ e: E::A, .. }} => 10, S {{ e: E::B, .. }} => 20 }};
         }}
         {MAIN}"
    );
    assert_eq!(counts(&src, "f"), (2, 1));
}

#[test]
fn a_generic_function_is_proven_only_on_a_concrete_scrutinee() {
    // A `bool` scrutinee in a value-type-generic function is the same type in every instance.
    let on_bool = "fn f<T>(x: T, b: bool) -> i32 { return match b { true => 1, false => 0 }; }
         fn main() -> i32 { print(f(2.5, true)); print(f(3, false)); return 0; }";
    let (prog, interner) = lowered(on_bool);
    let fs = funcs(&prog, &interner, "f");
    assert_eq!(fs.len(), 2, "two instances");
    for f in fs {
        assert_eq!(tests_and_traps(f), (1, 0), "{}", interner.resolve(f.name));
    }
    // A `(T, bool)` scrutinee names the generic: no verdict, both tests and the trap stay.
    let on_pair = "fn f<T>(x: T, b: bool) -> i32 {
             return match (x, b) { (_, true) => 1, (_, false) => 0 };
         }
         fn main() -> i32 { print(f(2.5, true)); print(f(3, false)); return 0; }";
    let (prog, interner) = lowered(on_pair);
    for f in funcs(&prog, &interner, "f") {
        assert_eq!(tests_and_traps(f), (2, 1), "{}", interner.resolve(f.name));
    }
}

#[test]
fn the_untested_arm_still_binds_its_payload() {
    // The last arm's pattern is never compared, but its bindings are made exactly as a tested
    // arm's would be: the payload of `Shape::Rect` reaches the body.
    let src = format!(
        "enum Shape {{ Dot, Circle(i32), Rect(i32, i32) }}
         fn area(s: Shape) -> i32 {{
             return match s {{ Shape::Dot => 0, Shape::Circle(r) => 3 * r * r, Shape::Rect(w, h) => w * h }};
         }}
         {MAIN}"
    );
    let (prog, interner) = lowered(&src);
    let f = funcs(&prog, &interner, "area")[0];
    assert_eq!(tests_and_traps(f), (2, 0));
    let loads = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i.op, Op::Load(..)))
        .count();
    // `r`, and both `w` and `h`, are loaded out of the payload.
    assert!(loads >= 3, "the payload fields are loaded: {loads}");
}

#[test]
fn a_literal_inference_re_pass_keeps_the_final_verdict() {
    // `k`'s type is decided by a later use, so the body is checked twice; the verdict the second
    // pass reaches is the one published.
    let src = format!(
        "fn f(x: u8) -> i64 {{
             let k = 7;
             let y: i64 = k;
             return match x {{ 0..=99 => y, 100..=199 => 2, 200..=255 => 3 }};
         }}
         {MAIN}"
    );
    let (prog, interner) = lowered(&src);
    let f = funcs(&prog, &interner, "f")[0];
    assert_eq!(tests_and_traps(f).1, 0, "no trap: the match is proven");
}
