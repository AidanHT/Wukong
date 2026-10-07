//! Structural tests for the recognizer spelling-coverage modules (`canon::inline_value_lets`,
//! `elementwise`, `ucount`, `colacc`, `softmax_oop`, `detile`, `dispatch`).
//!
//! Recognized kernels are GATE-BLIND: they fire identically on both backends at every `-O` level, so
//! no differential gate can see whether a spelling dispatched. These tests assert the emitted symbol
//! (and, where it matters, the operands) on the MIR itself; the value half lives in the
//! `tests/run/*_spelling*.wk` fixtures, which print every lane.

use wukong_diag::{Diagnostic, Severity};
use wukong_mir::verify::verify_function;
use wukong_mir::{Function, Op, Program};
use wukong_span::{Interner, SourceId};

use crate::{lower_program, lower_program_reporting_dispatch, LowerOptions};

fn lower_src(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, diags) = lower_program(&module, &sema, &mut interner);
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    for f in &prog.funcs {
        assert!(
            verify_function(f).is_empty(),
            "MIR verify failed in `{}`",
            interner.resolve(f.name)
        );
    }
    (prog, diags, interner)
}

fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    prog.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("function `{name}` was not lowered"))
}

/// Every recognized-kernel callee in `f`, in emission order (`wukong_parallel_for` excluded).
fn kernels(f: &Function, interner: &Interner) -> Vec<String> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for i in &b.insts {
            if let Op::Call { func, .. } = &i.op {
                let name = interner.resolve(*func);
                if name.starts_with("wukong_") && name != "wukong_parallel_for" {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

// ---- P12.14: per-loop `@generic` and the dispatch report ---------------------------------------

#[test]
fn generic_on_a_loop_opts_out_that_loop_only() {
    let src = "module m
fn two(x: [f32; 64], mut o: [f32; 64], mut p: [f32; 64]) {
    for i in 0..64 { o[i] = exp(x[i]); }
    @generic
    for i in 0..64 { p[i] = exp(x[i]); }
}
fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    @generic
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
";
    let (prog, _, interner) = lower_src(src);
    assert_eq!(
        kernels(func(&prog, &interner, "two"), &interner),
        vec!["wukong_vmath_f32"],
        "exactly the un-annotated loop dispatches"
    );
    assert!(
        kernels(func(&prog, &interner, "mm"), &interner).is_empty(),
        "a `@generic` loop is never claimed by a whole-function interception"
    );
}

/// A `@generic` loop nested inside a recognizable nest keeps the WHOLE nest from being replaced (the
/// GEMM recognizer probing `for i` would otherwise swallow the `@generic` k-loop), while a loop that
/// merely encloses one hands recognition back to its body, so the `@generic` loop's sibling still
/// dispatches.
#[test]
fn a_nested_generic_loop_keeps_its_enclosing_nest_and_not_its_siblings() {
    let src = "module m
fn nest(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        @generic
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
fn siblings(x: [f32; 64], mut o: [f32; 64], mut p: [f32; 64], n: i64) {
    for r in 0..n {
        for c in 0..64 { o[c] = exp(x[c]); }
        @generic
        for c in 0..64 { p[c] = exp(x[c]); }
    }
}
";
    let (prog, _, interner) = lower_src(src);
    let ks = kernels(func(&prog, &interner, "nest"), &interner);
    assert!(
        !ks.iter().any(|k| k.starts_with("wukong_sgemm")),
        "a nest holding a `@generic` loop must not become a GEMM, got {ks:?}"
    );
    assert_eq!(
        kernels(func(&prog, &interner, "siblings"), &interner),
        vec!["wukong_vmath_f32"],
        "the un-annotated sibling inside the enclosing loop still dispatches"
    );
}

/// [`lower_src`] through `lower_program_reporting_dispatch` (`--remarks=dispatch`).
fn lower_src_reporting(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, diags) =
        lower_program_reporting_dispatch(&module, &sema, &mut interner, LowerOptions::default());
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    (prog, diags, interner)
}

fn dispatch_notes(diags: &[Diagnostic]) -> Vec<&Diagnostic> {
    diags
        .iter()
        .filter(|d| {
            d.severity == Severity::Note && d.message.starts_with("recognized kernel dispatch")
        })
        .collect()
}

#[test]
fn the_dispatch_report_names_each_site_and_changes_no_mir() {
    let src = "module m
fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
fn nested(x: [f32; 64], mut o: [f32; 64], mut t: [f32; 4]) {
    for r in 0..4 {
        t[r] = 1.0;
        for c in 0..64 { o[c] = exp(x[c]); }
    }
}
";
    let (quiet, quiet_diags, qi) = lower_src(src);
    assert!(
        dispatch_notes(&quiet_diags).is_empty(),
        "reporting is opt-in"
    );
    let (loud, diags, li) = lower_src_reporting(src);
    assert_eq!(
        wukong_mir::print::print_program(&quiet, &qi),
        wukong_mir::print::print_program(&loud, &li),
        "the report must not change the MIR"
    );
    let notes = dispatch_notes(&diags);
    assert_eq!(notes.len(), 2, "{notes:?}");
    // The whole-function GEMM interception is reported at the function, by symbol.
    assert!(notes.iter().any(|d| d.message.contains("`wukong_sgemm`")));
    // The activation is reported at the INNER loop, and only there (not again at `for r`).
    let act: Vec<_> = notes
        .iter()
        .filter(|d| d.message.contains("`wukong_vmath_f32`"))
        .collect();
    assert_eq!(act.len(), 1);
    let lo = act[0].primary_span().expect("a located note").lo as usize;
    assert!(
        lo >= src.find("for c in").unwrap(),
        "attributed to the inner loop"
    );
}
