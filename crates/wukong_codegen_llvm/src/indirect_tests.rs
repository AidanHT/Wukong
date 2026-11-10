//! `Op::CallIndirect` in the textual-LLVM emitter: a `call` through the `ptr` operand, typed by the
//! MIR signature (argument types and return type) — with a result, discarded, and `void`.

use super::*;
use wukong_mir::{BinOp, Builder, Function, MirType, Op, Program};
use wukong_span::Interner;

#[test]
fn an_indirect_call_calls_through_the_pointer() {
    let mut i = Interner::new();
    let g_name = i.intern("g");
    let mut g = Builder::new(g_name, MirType::I64);
    let x = g.add_param(MirType::I64);
    let y = g.build(MirType::I64, Op::Bin(BinOp::Add, x, x));
    g.ret(Some(y));
    let sink_name = i.intern("sink");
    let mut s = Builder::new(sink_name, MirType::Void);
    s.add_param(MirType::I64);
    s.ret(None);

    let mut m = Builder::new(i.intern("main"), MirType::I64);
    let fp = m.add_param(MirType::Ptr);
    let a = m.add_param(MirType::I64);
    let sp = m.build(MirType::Ptr, Op::FuncAddr(sink_name));
    let r = m.build(
        MirType::I64,
        Op::CallIndirect {
            callee: fp,
            args: vec![a],
            ret: MirType::I64,
        },
    );
    m.build_void(Op::CallIndirect {
        callee: fp,
        args: vec![r],
        ret: MirType::I64,
    });
    m.build_void(Op::CallIndirect {
        callee: sp,
        args: vec![r],
        ret: MirType::Void,
    });
    m.ret(Some(r));
    let funcs: Vec<Function> = vec![g.finish(), s.finish(), m.finish()];
    let mut p = Program::new();
    p.funcs = funcs;
    let ir = try_emit_llvm_ir(&p, &i).unwrap();
    assert!(ir.contains("= call i64 %v0(i64 %v1)"), "{ir}");
    assert!(ir.contains("\n  call i64 %v0(i64 %v3)"), "{ir}");
    assert!(ir.contains("\n  call void %v2(i64 %v3)"), "{ir}");
}
