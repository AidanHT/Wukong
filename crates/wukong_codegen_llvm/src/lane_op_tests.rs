//! The textual form of the lane-moving SIMD ops and the vector shift by a scalar amount: each one is
//! spelled with the LLVM vector instruction of the same meaning, and the shift amount is masked to
//! the lane width before it is splatted (an LLVM shift by the lane width or more is poison, while
//! MIR's masks).

use super::*;
use wukong_mir::{Builder, Half, NarrowKind};

fn vt(lane: MirType, n: u32) -> MirType {
    MirType::Vec(Box::new(lane), n)
}

#[test]
fn lane_ops_emit_their_vector_instructions() {
    let mut it = Interner::new();
    let mut b = Builder::new(it.intern("lanes"), MirType::I32);
    let bytes = b.add_param(vt(MirType::I8, 16));
    let words = b.add_param(vt(MirType::I32, 4));
    let doubles = b.add_param(vt(MirType::F64, 2));
    let s = b.add_param(MirType::I32);
    let hi = b.build(
        vt(MirType::I16, 8),
        Op::Widen {
            v: bytes,
            half: Half::Hi,
            kind: CastKind::SExt,
        },
    );
    b.build(
        vt(MirType::I8, 16),
        Op::Narrow {
            lo: hi,
            hi,
            kind: NarrowKind::SatUnsigned,
        },
    );
    b.build(
        vt(MirType::F32, 4),
        Op::Narrow {
            lo: doubles,
            hi: doubles,
            kind: NarrowKind::FpTrunc,
        },
    );
    b.build(
        vt(MirType::I32, 4),
        Op::Shuffle {
            a: words,
            b: words,
            mask: vec![0, 2, 4, 6],
        },
    );
    b.build(vt(MirType::I32, 4), Op::InsertLane(words, s, 3));
    b.build(vt(MirType::I32, 4), Op::Bin(BinOp::LShr, words, s));
    b.ret(Some(s));
    let f = b.finish();
    assert!(wukong_mir::verify::verify_function(&f).is_empty());
    let mut program = Program::new();
    program.funcs.push(f);
    let text = try_emit_llvm_ir(&program, &it).unwrap();
    for needle in [
        // Widen: the high half, then the scalar cast's vector form.
        "<8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>",
        "sext <8 x i8>",
        // Narrow: concatenate, clamp to [0, 255], truncate.
        "shufflevector <8 x i16>",
        "icmp slt <16 x i16>",
        "icmp sgt <16 x i16>",
        "trunc <16 x i16>",
        "fptrunc <4 x double>",
        // Shuffle and InsertLane.
        "<4 x i32> <i32 0, i32 2, i32 4, i32 6>",
        "insertelement <4 x i32>",
        // The shift amount: masked as a scalar, splatted, then shifted.
        "and i32",
        "zeroinitializer",
        "lshr <4 x i32>",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in:\n{text}");
    }
}
