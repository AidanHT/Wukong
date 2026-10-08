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
