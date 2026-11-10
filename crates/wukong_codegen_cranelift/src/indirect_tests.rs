//! `Op::CallIndirect` on the native backend, against the interpreter (the oracle): a call through
//! a function pointer held in memory, with a result, discarded, and void; and a callee that fails
//! an assertion, which the caller must stop on exactly as after a direct call of an asserting
//! function (`asserting_functions` seeds every caller of an indirect call).

use super::*;
use wukong_mir::Builder;

fn i64c(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

/// `twice_plus(x) = 2x + 1`, `shout(x) = print(x)`, `fails() = { print(7); assert(false);
/// print(98); 2 }`, and a `main` built by `body`.
fn program(it: &mut Interner, body: impl FnOnce(&mut Builder, &mut Interner)) -> Program {
    let mut funcs = Vec::new();
    let mut t = Builder::new(it.intern("twice_plus"), MirType::I64);
    let x = t.add_param(MirType::I64);
    let two = i64c(&mut t, 2);
    let d = t.build(MirType::I64, Op::Bin(BinOp::Mul, x, two));
    let one = i64c(&mut t, 1);
    let r = t.build(MirType::I64, Op::Bin(BinOp::Add, d, one));
    t.ret(Some(r));
    funcs.push(t.finish());

    let mut s = Builder::new(it.intern("shout"), MirType::Void);
    let x = s.add_param(MirType::I64);
    s.build_void(Op::Call {
        func: it.intern("print"),
        args: vec![x],
    });
    s.ret(None);
    funcs.push(s.finish());

    let mut f = Builder::new(it.intern("fails"), MirType::I64);
    let seven = i64c(&mut f, 7);
    f.build_void(Op::Call {
        func: it.intern("print"),
        args: vec![seven],
    });
    let no = f.build(MirType::I1, Op::ConstInt(0, MirType::I1));
    f.build_void(Op::Call {
        func: it.intern("assert"),
        args: vec![no],
    });
    let late = i64c(&mut f, 98);
    f.build_void(Op::Call {
        func: it.intern("print"),
        args: vec![late],
    });
    let two = i64c(&mut f, 2);
    f.ret(Some(two));
    funcs.push(f.finish());

    let mut m = Builder::new(it.intern("main"), MirType::I64);
    body(&mut m, it);
    funcs.push(m.finish());
    let mut p = Program::new();
    p.funcs = funcs;
    assert_eq!(wukong_mir::verify::verify_program(&p), Vec::<String>::new());
    p
}

/// The function pointer of `name`, stored to a slot and loaded back (so nothing can see it is a
/// `func_addr` at the call).
fn pointer_to(m: &mut Builder, it: &mut Interner, name: &str) -> ValueId {
    let slot = m.alloca(MirType::Ptr);
    let addr = m.build(MirType::Ptr, Op::FuncAddr(it.intern(name)));
    m.build_void(Op::Store {
        ptr: slot,
        value: addr,
    });
    m.build(MirType::Ptr, Op::Load(slot, MirType::Ptr))
}

#[test]
fn indirect_calls_match_the_interpreter() {
    let mut it = Interner::new();
    let p = program(&mut it, |m, it| {
        let tp = pointer_to(m, it, "twice_plus");
        let sh = pointer_to(m, it, "shout");
        let five = i64c(m, 5);
        let a = m.build(
            MirType::I64,
            Op::CallIndirect {
                callee: tp,
                args: vec![five],
                ret: MirType::I64,
            },
        );
        // A value-returning signature whose result is discarded, and a void one.
        m.build_void(Op::CallIndirect {
            callee: tp,
            args: vec![a],
            ret: MirType::I64,
        });
        m.build_void(Op::CallIndirect {
            callee: sh,
            args: vec![a],
            ret: MirType::Void,
        });
        let b = m.build(
            MirType::I64,
            Op::CallIndirect {
                callee: tp,
                args: vec![a],
                ret: MirType::I64,
            },
        );
        m.ret(Some(b));
    });
    let main = it.intern("main");
    let want = wukong_interp::run_with_output(&p, main, &it).unwrap();
    assert_eq!(want, (23, b"11\n".to_vec()));
    assert_eq!(jit_run(&p, main, &it).unwrap(), want);
    // The object path builds the same `call_indirect`.
    assert!(!emit_object(&p, &it).unwrap().is_empty());
}

#[test]
fn a_failing_indirect_callee_stops_its_caller() {
    let mut it = Interner::new();
    let p = program(&mut it, |m, it| {
        let one = i64c(m, 1);
        m.build_void(Op::Call {
            func: it.intern("print"),
            args: vec![one],
        });
        let f = pointer_to(m, it, "fails");
        let r = m.build(
            MirType::I64,
            Op::CallIndirect {
                callee: f,
                args: vec![],
                ret: MirType::I64,
            },
        );
        m.build_void(Op::Call {
            func: it.intern("print"),
            args: vec![r],
        });
        m.ret(Some(r));
    });
    let main = it.intern("main");
    let failed: Result<i64, String> = Err("assertion failed".to_owned());
    let printed = b"1\n7\n".to_vec();
    assert_eq!(
        wukong_interp::run_capturing(&p, main, &it),
        (failed.clone(), printed.clone())
    );
    assert_eq!(jit_run_capturing(&p, main, &it), (failed, printed));
}
