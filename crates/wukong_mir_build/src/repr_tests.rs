//! Storage classification is total (N2): every `wukong_types::Ty` form `mir_ty` / `mir_ty_of`
//! meet, and every `ast::TypeKind` form `mir_ty_of_ann` meets, is answered by an arm of its own.
//! There is no `_ => I32` left: a type with a real machine representation gets it (a C-style
//! enum is the one `Named` type whose representation is `i32`), and one without is refused as
//! `C0001` naming what could not be sized, never handed a silently sized slot.
//!
//! The user-reachable refusals are driven from source. The arms sema guarantees never reach
//! lowering are driven by editing a checked program's AST or type table before lowering it; their
//! `debug_assert` is disabled under `cfg(test)` precisely so these tests can see the release
//! behaviour, the refusal.

use wukong_ast::{ItemKind, Module, Path, StmtKind, TypeExpr, TypeKind};
use wukong_diag::{Diagnostic, NoteKind};
use wukong_mir::{MirType, Op, Program};
use wukong_sema::SemaResult;
use wukong_span::{Interner, SourceId, Span};
use wukong_types::{Scalar, Ty};

use crate::{
    lower_program, mir_ty, ret_mir_ty, HELP_ANNOTATE, HELP_BIND_TYPE_PARAM, HELP_FN_VALUE,
    HELP_UNBOUND_NAME, HELP_UNREACHABLE_TYPE,
};

/// Parse and check `src`, which must check clean.
fn checked(src: &str) -> (Module, SemaResult, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    (module, sema, interner)
}

/// The slot type of every `alloca` in function `name`'s entry block, in order.
fn allocas(prog: &Program, interner: &Interner, name: &str) -> Vec<MirType> {
    let f = prog
        .funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function `{name}`"));
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| match &i.op {
            Op::Alloca { ty, .. } => Some(ty.clone()),
            _ => None,
        })
        .collect()
}

/// The errors among `diags`, each as `(code, message)`.
fn errors(diags: &[Diagnostic]) -> Vec<(Option<&'static str>, String)> {
    diags
        .iter()
        .filter(|d| d.is_error())
        .map(|d| (d.code, d.message.clone()))
        .collect()
}

/// The single refusal `diags` must hold: a `C0001` whose message names `what`, pointing at
/// `span` when one is given.
fn assert_one_refusal(diags: &[Diagnostic], what: &str, span: Option<Span>) {
    let errs = errors(diags);
    assert_eq!(errs.len(), 1, "expected exactly one error: {errs:?}");
    let (code, message) = &errs[0];
    assert_eq!(*code, Some("C0001"), "{errs:?}");
    assert!(message.contains(what), "`{message}` should name `{what}`");
    if let Some(span) = span {
        let d = diags.iter().find(|d| d.is_error()).unwrap();
        assert_eq!(d.primary_span(), Some(span), "{d:?}");
    }
}

/// Every `let` of function `name`, as `&mut` so a test can rewrite what sema checked.
fn lets_mut<'m>(
    module: &'m mut Module,
    interner: &Interner,
    name: &str,
) -> Vec<(&'m mut Option<TypeExpr>, &'m mut Option<wukong_ast::Expr>)> {
    let body = module
        .items
        .iter_mut()
        .find_map(|it| match &mut it.kind {
            ItemKind::Fn(f) if interner.resolve(f.name.sym) == name => f.body.as_mut(),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no function `{name}`"));
    body.stmts
        .iter_mut()
        .filter_map(|s| match &mut s.kind {
            StmtKind::Let { ty, init, .. } => Some((ty, init)),
            _ => None,
        })
        .collect()
}

#[test]
fn the_free_mir_ty_answers_only_from_structure() {
    let mut interner = Interner::new();
    let p = interner.intern("P");
    let f32_ = Ty::Scalar(Scalar::F32);
    let ptr = |pointee: Ty| Ty::Ptr {
        mutable: false,
        pointee: Box::new(pointee),
    };
    // Structural forms have their representation.
    assert_eq!(mir_ty(&f32_), Some(MirType::F32));
    assert_eq!(mir_ty(&Ty::Scalar(Scalar::I64)), Some(MirType::I64));
    assert_eq!(mir_ty(&ptr(f32_.clone())), Some(MirType::Ptr));
    assert_eq!(
        mir_ty(&Ty::Ref {
            mutable: true,
            pointee: Box::new(f32_.clone()),
        }),
        Some(MirType::Ptr)
    );
    assert_eq!(
        mir_ty(&Ty::Slice(Box::new(f32_.clone()))),
        Some(MirType::Ptr)
    );
    assert_eq!(
        mir_ty(&Ty::Tensor {
            elem: Scalar::F32,
            shape: wukong_types::Shape(vec![wukong_types::Dim::Const(4)]),
            layout: wukong_types::Layout::Contiguous,
            elem_var: None,
        }),
        Some(MirType::Ptr)
    );
    assert_eq!(
        mir_ty(&Ty::Vector {
            elem: Scalar::F32,
            lanes: 8,
        }),
        Some(MirType::Vec(Box::new(MirType::F32), 8))
    );
    assert_eq!(mir_ty(&Ty::Unit), Some(MirType::Void));
    let arr = |elem: Ty, len: u64| Ty::Array {
        elem: Box::new(elem),
        len,
        shared: false,
    };
    assert_eq!(
        mir_ty(&arr(Ty::Scalar(Scalar::I16), 3)),
        Some(MirType::Array(Box::new(MirType::I16), 3))
    );
    // (i8, f64): the f64 aligns to 8, so 16 bytes.
    assert_eq!(
        mir_ty(&Ty::Tuple(vec![
            Ty::Scalar(Scalar::I8),
            Ty::Scalar(Scalar::F64)
        ])),
        Some(MirType::Array(Box::new(MirType::I8), 16))
    );
    // A `Named` needs the registry: a struct, an enum and a type parameter all answer `None`
    // here, and so does anything built from one. It used to answer `i32`.
    assert_eq!(mir_ty(&Ty::Named(p)), None);
    assert_eq!(mir_ty(&arr(Ty::Named(p), 4)), None);
    assert_eq!(mir_ty(&Ty::Tuple(vec![f32_.clone(), Ty::Named(p)])), None);
    // An extent the MIR's `u32` cannot hold is no answer, not a truncated one.
    assert_eq!(mir_ty(&arr(f32_.clone(), u64::from(u32::MAX) + 2)), None);
    // No machine representation at all.
    let fn_ty = Ty::Fn {
        params: vec![f32_.clone()],
        ret: Box::new(f32_),
    };
    assert_eq!(mir_ty(&fn_ty), None);
    assert_eq!(mir_ty(&Ty::Unknown), None);
    assert_eq!(mir_ty(&Ty::Error), None);
}

#[test]
fn every_annotation_form_sizes_its_slot_from_its_own_arm() {
    let src = "struct P { x: f32, y: f64 }\n\
               enum C { A, B }\n\
               enum D { V(i64), W }\n\
               fn locals() -> i32 {\n\
                   let a: f32;\n\
                   let b: P;\n\
                   let c: C;\n\
                   let d: D;\n\
                   let e: [i16; 3];\n\
                   let g: (i8, f64);\n\
                   let h: []f32;\n\
                   let i: *f32;\n\
                   let k: Tensor[f32, 4];\n\
                   let l: f32x8;\n\
                   return 0;\n\
               }\n\
               fn main() -> i32 { return locals(); }\n";
    let (module, sema, mut interner) = checked(src);
    let (prog, diags) = lower_program(&module, &sema, &mut interner);
    assert!(errors(&diags).is_empty(), "{diags:?}");
    let bytes = |n| MirType::Array(Box::new(MirType::I8), n);
    assert_eq!(
        allocas(&prog, &interner, "locals"),
        vec![
            MirType::F32,                              // a scalar path
            bytes(16),                                 // a struct: f32 @ 0, f64 @ 8
            MirType::I32,                              // a C-style enum: its discriminant
            bytes(16),                                 // a data-carrying enum: tag + i64 payload
            MirType::Array(Box::new(MirType::I16), 3), // an array
            bytes(16),                                 // a tuple: i8 @ 0, f64 @ 8
            bytes(16),                                 // a slice: its fat pointer
            MirType::Ptr,                              // a pointer
            MirType::Ptr,                              // a tensor view
            MirType::Vec(Box::new(MirType::F32), 8),   // a SIMD vector
        ]
    );
}

#[test]
fn a_c_style_enum_is_an_i32_parameter_return_and_local() {
    let src = "enum C { A, B }\n\
               fn pick(c: C) -> C { let d: C = c; return d; }\n\
               fn main() -> i32 {\n\
                   let x = pick(C::B);\n\
                   return match x { C::A => 1, C::B => 2 };\n\
               }\n";
    let (module, sema, mut interner) = checked(src);
    let (prog, diags) = lower_program(&module, &sema, &mut interner);
    assert!(errors(&diags).is_empty(), "{diags:?}");
    let pick = prog
        .funcs
        .iter()
        .find(|f| interner.resolve(f.name) == "pick")
        .unwrap();
    assert_eq!(pick.ret, MirType::I32);
    assert_eq!(pick.value_type(pick.params[0]), &MirType::I32);
    assert!(allocas(&prog, &interner, "pick").contains(&MirType::I32));
}

#[test]
fn a_let_with_no_type_and_no_initializer_is_refused() {
    // `let mut x; x = 2.75;` used to get an `i32` slot and print `2`-based results.
    let src = "fn main() -> i32 {\n\
                   let mut x;\n\
                   x = 2.75;\n\
                   println(x * 2.0);\n\
                   return 0;\n\
               }\n";
    let (module, sema, mut interner) = checked(src);
    let (_, diags) = lower_program(&module, &sema, &mut interner);
    assert_one_refusal(
        &diags,
        "a `let` with neither a type annotation nor an initializer",
        None,
    );
}

#[test]
fn a_type_parameter_no_call_binds_is_refused_at_its_annotation() {
    // `U` appears in no parameter or return type, so the monomorphizer never instantiates it and
    // the body lowers with it opaque. It used to be an `i32` slot: `y = 2.5; println(y)` printed 2.
    let src = "fn g<U>(n: i32) -> i32 {\n\
                   let mut y: U;\n\
                   y = 2.5;\n\
                   println(y);\n\
                   return n;\n\
               }\n\
               fn main() -> i32 { return g(0); }\n";
    let (mut module, sema, mut interner) = checked(src);
    let span = lets_mut(&mut module, &interner, "g")[0]
        .0
        .as_ref()
        .unwrap()
        .span;
    let (_, diags) = lower_program(&module, &sema, &mut interner);
    assert_one_refusal(&diags, "unbound type parameter `U`", Some(span));
}

/// The annotation forms sema never hands lowering (each is an error there) are refused, not
/// sized: rewrite `let a: f32;` into each, after sema checked it, and lower.
#[test]
fn an_annotation_sema_rejects_is_refused_if_it_reaches_lowering() {
    let src = "struct P { x: f32 }\n\
               fn locals() -> i32 { let a: f32; return 0; }\n\
               fn main() -> i32 { return locals(); }\n";
    let cases: [(&str, fn(&mut Interner, Span) -> TypeKind); 3] = [
        ("dimension `512` written as a type", |i, _| {
            TypeKind::Int(i.intern("512"))
        }),
        ("a value of function type", |_, _| TypeKind::Fn {
            params: vec![],
            ret: None,
        }),
        ("unexpanded generic type `P<..>`", |i, span| {
            TypeKind::Generic {
                path: Box::new(Path {
                    segments: vec![wukong_ast::Ident {
                        sym: i.intern("P"),
                        span,
                    }],
                    span,
                }),
                args: vec![],
            }
        }),
    ];
    for (what, kind) in cases {
        let (mut module, sema, mut interner) = checked(src);
        let span = {
            let mut lets = lets_mut(&mut module, &interner, "locals");
            let ann = lets[0].0.as_mut().unwrap();
            ann.kind = kind(&mut interner, ann.span);
            ann.span
        };
        let (_, diags) = lower_program(&module, &sema, &mut interner);
        assert_one_refusal(&diags, what, Some(span));
    }
}

/// The help of the single error in `diags`.
fn help_of(diags: &[Diagnostic]) -> &str {
    let d = diags.iter().find(|d| d.is_error()).expect("an error");
    d.notes
        .iter()
        .find_map(|(kind, text)| matches!(kind, NoteKind::Help).then_some(text.as_str()))
        .unwrap_or_else(|| panic!("no help on {d:?}"))
}

/// The value types with no machine representation are refused where a value of one needs
/// storage: retype `let v = 1.5;`'s initializer, after sema checked it, and lower. Each refusal a
/// program that checks clean can reach says how to fix the program; only `Ty::Error`, which never
/// survives sema, is the compiler-bug refusal (a debug-build panic outside these tests).
#[test]
fn a_value_type_with_no_representation_is_refused() {
    let src = "fn other() -> i32 { return 1; }\n\
               fn locals() -> i32 { let v = 1.5; return 0; }\n\
               fn main() -> i32 { return locals() + other(); }\n";
    let cases: [(&str, &str, fn(&mut Interner) -> Ty); 4] = [
        // Reachable: sema leaves a value it does not model `Unknown`.
        (
            "a value whose type was never determined (`_`)",
            HELP_ANNOTATE,
            |_| Ty::Unknown,
        ),
        // Reachable: a function name in an array or tuple literal, or alone as a statement, is
        // typed as a function value (`[square, square]` is `[fn(f32) -> f32; 2]`).
        (
            "a function value of type `fn(f32) -> f32`",
            HELP_FN_VALUE,
            |_| Ty::Fn {
                params: vec![Ty::Scalar(Scalar::F32)],
                ret: Box::new(Ty::Scalar(Scalar::F32)),
            },
        ),
        // Reachable: a name declared as something other than a type, shadowed by a type
        // parameter that is not the enclosing function's (a generic struct's own `T`).
        (
            "a value of the type `other`, which is neither a struct nor an enum",
            HELP_UNBOUND_NAME,
            |i| Ty::Named(i.intern("other")),
        ),
        // Unreachable: the driver stops on sema's error first.
        ("a value of type `<error>`", HELP_UNREACHABLE_TYPE, |_| {
            Ty::Error
        }),
    ];
    for (what, help, ty) in cases {
        let (mut module, mut sema, mut interner) = checked(src);
        let (id, span) = {
            let lets = lets_mut(&mut module, &interner, "locals");
            let init = lets[0].1.as_ref().unwrap();
            (init.id, init.span)
        };
        sema.types.insert(id, ty(&mut interner));
        let (_, diags) = lower_program(&module, &sema, &mut interner);
        assert_one_refusal(&diags, what, Some(span));
        assert_eq!(help_of(&diags), help, "{what}");
    }
}

/// A type parameter no call binds is opaque even when a `const`, `fn`, enum or struct of the same
/// name exists: the enclosing function's generics decide it, not the item registry. Sized from the
/// registry, `fn g<E>` over a C-style `enum E` gave `y` an `i32` slot (`y = 2.5; println(y)`
/// printed `2` on both backends), `fn g<P>` over a `struct P` gave it the struct's buffer (a MIR
/// verifier ICE), and `fn g<N>` over a `const N` took the debug-build panic for a non-type name.
#[test]
fn a_type_parameter_that_shadows_an_item_is_refused_not_sized_as_the_item() {
    let items = "const N: usize = 4;\n\
                 fn h(x: i32) -> i32 { return x; }\n\
                 enum E { A, B }\n\
                 enum D { V(i64), W }\n\
                 struct P { a: i32, b: i32 }\n";
    for shadowed in ["N", "h", "E", "D", "P"] {
        let src = format!(
            "{items}\
             fn g<{shadowed}>(n: i32) -> i32 {{\n\
                 let _y: {shadowed};\n\
                 return n;\n\
             }}\n\
             fn main() -> i32 {{ return g(0) + h(0); }}\n"
        );
        let (mut module, sema, mut interner) = checked(&src);
        let span = lets_mut(&mut module, &interner, "g")[0]
            .0
            .as_ref()
            .unwrap()
            .span;
        let (_, diags) = lower_program(&module, &sema, &mut interner);
        assert_one_refusal(
            &diags,
            &format!("a value of the unbound type parameter `{shadowed}`"),
            Some(span),
        );
        assert_eq!(help_of(&diags), HELP_BIND_TYPE_PARAM, "{shadowed}");
    }
}

/// A function named as a value outside a call is refused as `C0001` where it is lowered, with no
/// detour through the compiler-bug refusal: in an array literal (sema types it
/// `[fn(f32) -> f32; 2]`) and alone as a statement. Both were the debug-build panic.
#[test]
fn a_function_value_outside_a_call_is_a_codegen_limit() {
    for body in ["let _fs = [square, square];", "square;"] {
        let src = format!(
            "fn square(x: f32) -> f32 {{ return x * x; }}\n\
             fn main() -> i32 {{ {body} return 0; }}\n"
        );
        let (module, sema, mut interner) = checked(&src);
        let (_, diags) = lower_program(&module, &sema, &mut interner);
        let errs = errors(&diags);
        assert!(!errs.is_empty(), "{body}: lowered clean");
        for (code, message) in &errs {
            assert_eq!(*code, Some("C0001"), "{body}: {errs:?}");
            assert!(!message.contains("<error>"), "{body}: {message}");
        }
        assert!(
            diags.iter().all(|d| d
                .notes
                .iter()
                .all(|(_, text)| text != HELP_UNREACHABLE_TYPE)),
            "{body}: the compiler-bug refusal: {diags:?}"
        );
    }
}

#[test]
fn a_return_type_with_no_representation_is_refused() {
    let src = "enum C { A, B }\n\
               fn f() -> f32 { return 1.0; }\n\
               fn main() -> i32 { return 0; }\n";
    let (module, sema, interner) = checked(src);
    let f = module
        .items
        .iter()
        .find_map(|it| match &it.kind {
            ItemKind::Fn(f) if interner.resolve(f.name.sym) == "f" => Some(f),
            _ => None,
        })
        .unwrap();
    let mut diags = Vec::new();
    let c = interner.get("C").unwrap();
    // A C-style enum returns its discriminant; a scalar returns itself.
    assert_eq!(
        ret_mir_ty(f, &Ty::Named(c), &sema, &interner, &mut diags),
        MirType::I32
    );
    assert_eq!(
        ret_mir_ty(f, &Ty::Scalar(Scalar::F64), &sema, &interner, &mut diags),
        MirType::F64
    );
    assert!(diags.is_empty(), "{diags:?}");
    for ty in [Ty::Unknown, Ty::Error] {
        let mut diags = Vec::new();
        ret_mir_ty(f, &ty, &sema, &interner, &mut diags);
        let span = f.ret.as_ref().unwrap().span;
        assert_one_refusal(&diags, "a return value of type", Some(span));
    }
}

#[test]
fn a_function_reports_one_refusal_at_its_first_unsized_value() {
    // Two values of the unbound `U` in one function: one report, at the first.
    let src = "fn g<U>(n: i32) -> i32 {\n\
                   let a: U;\n\
                   let b: U;\n\
                   return n;\n\
               }\n\
               fn main() -> i32 { return g(0); }\n";
    let (mut module, sema, mut interner) = checked(src);
    let first = lets_mut(&mut module, &interner, "g")[0]
        .0
        .as_ref()
        .unwrap()
        .span;
    let (_, diags) = lower_program(&module, &sema, &mut interner);
    assert_one_refusal(&diags, "unbound type parameter `U`", Some(first));
}
