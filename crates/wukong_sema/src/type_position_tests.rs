//! What may stand where a type is written. An integer and a type given arguments it does not take
//! used to type as `Ty::Error` with no diagnostic, so the program checked clean and reached MIR
//! lowering with a type it cannot size (N2): `let x: 512 = 3;` ran, its slot an `i32`. And an
//! undeclared type name's `E0324` was lost whenever literal inference re-checked the body.

use wukong_span::{Interner, SourceId};

/// The diagnostic codes `src` checks to.
fn codes(src: &str) -> Vec<&'static str> {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pd.is_empty(), "parse errors: {pd:?}");
    crate::check(&module, &interner)
        .1
        .into_iter()
        .filter_map(|d| d.code)
        .collect()
}

#[test]
fn an_integer_is_not_a_type() {
    assert_eq!(
        codes("fn main() -> i32 { let x: 512 = 3; return x; }"),
        vec!["E0324"]
    );
    assert_eq!(
        codes("fn main() -> i32 { let x: [512; 2] = [3, 4]; return x[0]; }"),
        vec!["E0324"]
    );
    assert_eq!(codes("fn f(a: 4) -> i32 { return 0; }"), vec!["E0324"]);
    // As a dimension (a turbofish argument) it is what it always was.
    assert_eq!(
        codes(
            "fn first<N>(x: Tensor[f32, N]) -> f32 { return x[0]; } \
             fn main() -> i32 { let a: [f32; 4] = [1.0, 2.0, 3.0, 4.0]; \
             return first::<4>(a) as i32; }"
        ),
        Vec::<&str>::new()
    );
}

#[test]
fn a_type_that_is_not_an_alias_takes_no_arguments() {
    // A declared struct given arguments: the alias expander reports this only in a module that
    // declares an alias, and sema took it as `Ty::Error` without a word.
    assert_eq!(
        codes(
            "struct P { x: f32 } \
             fn main() -> i32 { let p: P<2> = P { x: 1.5 }; return 0; }"
        ),
        vec!["E0327"]
    );
    // An enum, and a type parameter, the same.
    assert_eq!(
        codes("enum C { A } fn main() -> i32 { let c: C<f32>; return 0; }"),
        vec!["E0327"]
    );
    assert_eq!(
        codes("fn f<T>(x: T) -> i32 { let y: T<2>; return 0; }"),
        vec!["E0327"]
    );
    // A name that is no type at all is still `E0324`, and only that.
    assert_eq!(
        codes("fn main() -> i32 { let q: Nope<2>; return 0; }"),
        vec!["E0324"]
    );
}

#[test]
fn an_unknown_type_survives_the_literal_inference_recheck() {
    // `let y: i64 = k` decides `k`, so the body is checked again — and the re-check used to drop
    // the first pass's `E0324` while `resolve.rs` still remembered reporting it: the program
    // checked clean and lowering gave `x` an `i32` slot (`x = 2.5` printed 2).
    let src = "fn main() -> i32 { let k = 7; let y: i64 = k; let mut x: Foo; x = 2.5; \
               println(y); println(x); return 0; }";
    assert_eq!(codes(src), vec!["E0324"]);
    // It is reported where a single pass reports it: before the mismatch the unknown type
    // causes, and once.
    let src = "fn main() -> i32 { let k = 7; let y: i64 = k; let x: Foo = 3; return 0; }";
    assert_eq!(codes(src), vec!["E0324", "E0401"]);
}
