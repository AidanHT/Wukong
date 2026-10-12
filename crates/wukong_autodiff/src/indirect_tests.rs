//! A loss that calls through a function value is refused by name: the callee is not known at
//! compile time, so neither its result nor anything it writes has a VJP rule, and a silently
//! missing contribution would be a wrong gradient.

use wukong_mir::{BinOp, Builder, MirType, Op};
use wukong_span::Interner;

#[test]
fn an_indirect_call_is_refused_by_name() {
    let mut it = Interner::new();
    let mut b = Builder::new(it.intern("loss"), MirType::F32);
    let w = b.add_param(MirType::Ptr);
    let fp = b.add_param(MirType::Ptr);
    let x = b.build(MirType::F32, Op::Load(w, MirType::F32));
    let y = b.build(
        MirType::F32,
        Op::CallIndirect {
            callee: fp,
            args: vec![x],
            ret: MirType::F32,
        },
    );
    let l = b.build(MirType::F32, Op::Bin(BinOp::FMul, y, x));
    b.ret(Some(l));
    let err = crate::grad(&b.finish(), &[0], &mut it).unwrap_err();
    assert!(err.contains("indirect call"), "{err}");
}
