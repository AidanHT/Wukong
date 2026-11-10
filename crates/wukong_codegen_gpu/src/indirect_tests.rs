//! `Op::CallIndirect` on the GPU paths: PTX has no portable function pointers, so the MIR -> PTX
//! lowering declines a call through a function value with an `UNSUPPORTED:` error (the coverage
//! gate's skip, never wrong PTX), and the megakernel's eligibility analysis declines it too, so
//! eligibility and lowering agree. Pure host code: no device is touched.

use wukong_mir::{Builder, MirType, Op, Program};
use wukong_span::Interner;

#[test]
fn an_indirect_call_is_declined_not_lowered() {
    let mut i = Interner::new();
    let g_name = i.intern("g");
    let mut g = Builder::new(g_name, MirType::I64);
    let x = g.add_param(MirType::I64);
    g.ret(Some(x));
    let main = i.intern("main");
    let mut m = Builder::new(main, MirType::I64);
    let fp = m.build(MirType::Ptr, Op::FuncAddr(g_name));
    let one = m.build(MirType::I64, Op::ConstInt(1, MirType::I64));
    let r = m.build(
        MirType::I64,
        Op::CallIndirect {
            callee: fp,
            args: vec![one],
            ret: MirType::I64,
        },
    );
    m.ret(Some(r));
    let mut p = Program::new();
    p.funcs = vec![g.finish(), m.finish()];
    let err = crate::lower::emit_ptx(&p, main, &i).unwrap_err();
    assert!(
        err.starts_with(crate::lower::UNSUPPORTED) && err.contains("CallIndirect"),
        "{err}"
    );
}
