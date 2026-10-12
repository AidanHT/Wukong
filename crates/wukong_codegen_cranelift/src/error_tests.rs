//! The native backend's failure classes (`classify_error`), pinned against the messages the backend
//! actually produces: the driver renders `LayoutTooLarge` as `C0002`, `Unsupported` as `C0001` and
//! `Internal` as an internal-compiler-error diagnostic, so a reworded message that silently changed
//! class would change the user-visible code.

use super::*;
use wukong_mir::Builder;

fn program_with(f: Function) -> Program {
    let mut p = Program::new();
    p.funcs.push(f);
    p
}

#[test]
fn oversized_local_is_a_layout_error_naming_type_and_function() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    // 600_000_000 f64 = 4.8e9 bytes > 2^32: representable in MIR (a u32 extent), not in the
    // backend's 32-bit stack-slot sizes.
    b.alloca(MirType::Array(Box::new(MirType::F64), 600_000_000));
    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let p = program_with(b.finish());
    let err = emit_object(&p, &i).expect_err("a >4 GiB slot cannot be laid out");
    assert_eq!(
        classify_error(&err),
        NativeErrorKind::LayoutTooLarge,
        "{err}"
    );
    assert!(err.contains("[600000000 x f64]"), "{err}");
    assert!(err.contains("function `main`"), "{err}");
    let main = i.intern("main");
    let jit = jit_compile(&p, main, &i)
        .err()
        .expect("the JIT shares the layout rule");
    assert_eq!(
        classify_error(&jit),
        NativeErrorKind::LayoutTooLarge,
        "{jit}"
    );
}

#[test]
fn unbound_external_call_is_unsupported() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    b.build_void(Op::Call {
        func: i.intern("mystery_symbol"),
        args: vec![],
    });
    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let err = emit_object(&program_with(b.finish()), &i).expect_err("no ABI for the symbol");
    assert_eq!(classify_error(&err), NativeErrorKind::Unsupported, "{err}");
}

#[test]
fn parameterized_entry_is_unsupported() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let mut b = Builder::new(main, MirType::I32);
    let x = b.add_param(MirType::I32);
    b.ret(Some(x));
    let err = jit_compile(&program_with(b.finish()), main, &i)
        .err()
        .expect("an entry point with parameters cannot be invoked");
    assert_eq!(classify_error(&err), NativeErrorKind::Unsupported, "{err}");
}

#[test]
fn runtime_kernel_arity_mismatch_is_internal() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    let p = b.add_param(MirType::Ptr);
    // `wukong_sgemm` takes seven arguments; one is a recognizer bug, never a user error.
    b.build_void(Op::Call {
        func: i.intern("wukong_sgemm"),
        args: vec![p],
    });
    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let err = emit_object(&program_with(b.finish()), &i).expect_err("no ABI matches one argument");
    assert_eq!(classify_error(&err), NativeErrorKind::Internal, "{err}");
}

#[test]
fn unclassified_backend_text_is_internal() {
    assert_eq!(classify_error("Verifier errors"), NativeErrorKind::Internal);
    assert_eq!(
        classify_error("avx2 assemble vec kernel 0.0: encode"),
        NativeErrorKind::Internal
    );
}

// --- Where a failure points (`NativeFailure::site`) -------------------------------------------

/// A span in a source the tests never render: only identity matters here.
fn span(lo: u32) -> wukong_span::Span {
    wukong_span::Span::new(wukong_span::SourceId(0), lo, lo + 4)
}

/// `main` with a >4 GiB slot. `alloca_at` locates the alloca; `decl` declares the function after
/// every value is built, so it locates nothing but the function itself. The `return 0` after the
/// slot is located either way: it is not what failed, so an unlocated slot must not borrow its span.
fn oversized(i: &mut Interner, alloca_at: Option<u32>, decl: Option<u32>) -> Program {
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    if let Some(lo) = alloca_at {
        b.set_loc(span(lo));
    }
    b.alloca(MirType::Array(Box::new(MirType::F64), 600_000_000));
    b.set_loc(span(500));
    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let mut f = b.finish();
    if let Some(lo) = decl {
        f.locs.decl = span(lo);
    }
    program_with(f)
}

/// The site `emit_object_kind_for_with` reports for `f`, declared at `span(1)` after it was built
/// (so the declaration locates no value).
fn site_of(i: &Interner, mut f: Function) -> Option<wukong_span::Span> {
    f.locs.decl = span(1);
    emit_object_kind_for_with(
        &program_with(f),
        i,
        TargetCpu::Native,
        BackendOpt::default(),
        ObjectKind::Library,
    )
    .expect_err("the program cannot be compiled")
    .site
}

/// An oversized slot that is not located, after a located value of the same block: the failure is
/// the slot's, so it falls back to the declaration instead of naming the value before it.
#[test]
fn an_unlocated_failure_does_not_borrow_the_value_before_it() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    b.set_loc(span(20));
    let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
    b.restore_loc(wukong_span::Span::dummy());
    b.alloca(MirType::Array(Box::new(MirType::F64), 600_000_000));
    b.ret(Some(one));
    assert_eq!(site_of(&i, b.finish()), Some(span(1)));
}

/// A block with no located value of its own does not inherit the location of the block lowered
/// before it, nor take one from a value after the failure.
#[test]
fn an_unlocated_failure_does_not_borrow_across_blocks_or_from_later_values() {
    for later in [false, true] {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("main"), MirType::I32);
        b.set_loc(span(20));
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let next = b.new_block();
        b.br(next, vec![]);
        b.switch_to(next);
        b.build_void(Op::Call {
            func: i.intern("mystery_symbol"),
            args: vec![],
        });
        let ret = if later {
            b.set_loc(span(60));
            b.build(MirType::I32, Op::ConstInt(0, MirType::I32))
        } else {
            one
        };
        b.ret(Some(ret));
        assert_eq!(
            site_of(&i, b.finish()),
            Some(span(1)),
            "later value: {later}"
        );
    }
}

/// A result-less call that does not read the value lowered just before it (`bar();` after
/// `let a = 1;`) is not that value's statement: it falls back to the declaration.
#[test]
fn a_call_that_reads_nothing_does_not_borrow_the_value_before_it() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    b.set_loc(span(20));
    let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
    b.build_void(Op::Call {
        func: i.intern("mystery_symbol"),
        args: vec![],
    });
    b.ret(Some(one));
    assert_eq!(site_of(&i, b.finish()), Some(span(1)));
}

/// (a) A layout failure points at the slot that could not be laid out — on the object path and
/// the JIT path, which share `build_function_clif`.
#[test]
fn a_layout_failure_points_at_the_oversized_slot() {
    let mut i = Interner::new();
    let p = oversized(&mut i, Some(40), Some(7));
    let main = i.intern("main");
    let obj = emit_object_kind_for_with(
        &p,
        &i,
        TargetCpu::Native,
        BackendOpt::default(),
        ObjectKind::Library,
    )
    .expect_err("a >4 GiB slot cannot be laid out");
    assert_eq!(
        classify_error(&obj.message),
        NativeErrorKind::LayoutTooLarge
    );
    assert_eq!(obj.site, Some(span(40)), "{obj:?}");
    let jit = jit_compile_for_with(&p, main, &i, TargetCpu::Native, BackendOpt::default())
        .err()
        .expect("the JIT shares the layout rule");
    assert_eq!(jit, obj);
    // The String-returning entry points report the same message without the site.
    assert_eq!(emit_object(&p, &i).unwrap_err(), obj.message);
}

/// (b) No locations, no site; a declaration alone is the fallback.
#[test]
fn an_unlocated_failure_falls_back_to_the_declaration_or_nothing() {
    let mut i = Interner::new();
    let bare = emit_object(&oversized(&mut i, None, None), &i).unwrap_err();
    let site = |p: &Program, i: &Interner| {
        emit_object_kind_for_with(
            p,
            i,
            TargetCpu::Native,
            BackendOpt::default(),
            ObjectKind::Library,
        )
        .unwrap_err()
    };
    let none = site(&oversized(&mut i, None, None), &i);
    assert_eq!(none.site, None);
    assert_eq!(none.message, bare);
    let decl = site(&oversized(&mut i, None, Some(7)), &i);
    assert_eq!(decl.site, Some(span(7)));
}

/// (c) An unsupported call has no result of its own, so it is reported at the located value it
/// reads that was lowered just before it (its argument) — never at one after it.
#[test]
fn an_unsupported_call_points_at_the_value_it_reads() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::I32);
    b.set_decl(span(1));
    b.set_loc(span(20));
    let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
    b.build_void(Op::Call {
        func: i.intern("mystery_symbol"),
        args: vec![one],
    });
    b.set_loc(span(60));
    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let err = emit_object_kind_for_with(
        &program_with(b.finish()),
        &i,
        TargetCpu::Native,
        BackendOpt::default(),
        ObjectKind::Library,
    )
    .unwrap_err();
    assert_eq!(classify_error(&err.message), NativeErrorKind::Unsupported);
    assert_eq!(err.site, Some(span(20)), "{err:?}");
}

/// (d) An entry point with parameters is reported at its declaration.
#[test]
fn a_parameterized_entry_points_at_its_declaration() {
    let mut i = Interner::new();
    let main = i.intern("main");
    let mut b = Builder::new(main, MirType::I32);
    b.set_decl(span(3));
    let x = b.add_param(MirType::I32);
    b.ret(Some(x));
    let err = jit_compile_for_with(
        &program_with(b.finish()),
        main,
        &i,
        TargetCpu::Native,
        BackendOpt::default(),
    )
    .err()
    .expect("an entry point with parameters cannot be invoked");
    assert_eq!(classify_error(&err.message), NativeErrorKind::Unsupported);
    assert_eq!(err.site, Some(span(3)));
}

/// A failure that is not raised while lowering one instruction but still belongs to one function —
/// a declined compact boolean-vector legalization (`C0001`) — is reported at that function's
/// declaration, not span-less.
#[test]
fn a_declined_legalization_points_at_the_declaration() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("unsupported_abi"), MirType::Void);
    let source = MirType::Vec(Box::new(MirType::F32), 4);
    let target = MirType::Vec(Box::new(MirType::I1), 4);
    let input = b.add_param(source);
    b.build(
        target.clone(),
        Op::Cast(wukong_mir::CastKind::FpToSi, input, target),
    );
    b.ret(None);
    let mut f = b.finish();
    f.locs.decl = span(1);
    let err = emit_object_kind_for_with(
        &program_with(f),
        &i,
        TargetCpu::Native,
        BackendOpt::default(),
        ObjectKind::Library,
    )
    .unwrap_err();
    assert!(err.message.contains("require a vector ABI"), "{err:?}");
    assert_eq!(classify_error(&err.message), NativeErrorKind::Unsupported);
    assert_eq!(err.site, Some(span(1)), "{err:?}");
}

/// A function Cranelift itself rejects (its verifier: an `i32` function that returns nothing) is an
/// internal compiler error at that function's declaration — on the serial definer and the parallel
/// one, which compiles each function on its own worker.
#[test]
fn a_rejected_function_points_at_its_declaration() {
    let mut i = Interner::new();
    let mut ok = Builder::new(i.intern("fine"), MirType::Void);
    ok.ret(None);
    let mut ok = ok.finish();
    ok.locs.decl = span(30);
    let mut bad = Builder::new(i.intern("main"), MirType::I32);
    bad.ret(None);
    let mut bad = bad.finish();
    bad.locs.decl = span(1);
    let mut p = Program::new();
    p.funcs.push(ok);
    p.funcs.push(bad);
    for parallel in [false, true] {
        let err = emit_object_full(
            &p,
            &i,
            EmitOptions {
                verify: true,
                parallel,
            },
            TargetCpu::Native,
            BackendOpt::default(),
            ObjectKind::Library,
        )
        .err()
        .expect("the verifier rejects a value-less return");
        assert_eq!(
            classify_error(&err.message),
            NativeErrorKind::Internal,
            "{err:?}"
        );
        assert_eq!(err.site, Some(span(1)), "parallel {parallel}: {err:?}");
    }
}
