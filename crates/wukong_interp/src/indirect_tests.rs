//! `Op::CallIndirect` in the interpreter: a function address is the tag `FUNC_TAG + index`, so a
//! call through one enters that function's frame, and a call through anything else is an error
//! (never a jump into memory, or into whatever function an unrelated integer happens to index).

use super::*;
use wukong_mir::Builder;

fn i64c(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

/// `inc(x) = x + 1` and a `main` that calls through `callee(main)` with `41`.
fn program(it: &mut Interner, callee: impl FnOnce(&mut Builder, Symbol) -> ValueId) -> Program {
    let inc_name = it.intern("inc");
    let mut f = Builder::new(inc_name, MirType::I64);
    let x = f.add_param(MirType::I64);
    let one = i64c(&mut f, 1);
    let r = f.build(MirType::I64, Op::Bin(BinOp::Add, x, one));
    f.ret(Some(r));
    let mut m = Builder::new(it.intern("main"), MirType::I64);
    let fp = callee(&mut m, inc_name);
    let arg = i64c(&mut m, 41);
    let r = m.build(
        MirType::I64,
        Op::CallIndirect {
            callee: fp,
            args: vec![arg],
            ret: MirType::I64,
        },
    );
    m.ret(Some(r));
    let mut p = Program::new();
    p.funcs = vec![f.finish(), m.finish()];
    p
}

#[test]
fn a_call_through_a_function_address_enters_the_function() {
    let mut it = Interner::new();
    let p = program(&mut it, |m, inc| m.build(MirType::Ptr, Op::FuncAddr(inc)));
    assert_eq!(run(&p, it.intern("main"), &it), Ok(42));
}

#[test]
fn a_call_through_a_data_pointer_is_an_error() {
    let mut it = Interner::new();
    let p = program(&mut it, |m, _| m.alloca(MirType::I64));
    let err = run(&p, it.intern("main"), &it).unwrap_err();
    assert!(err.contains("not a function address"), "{err}");
}
