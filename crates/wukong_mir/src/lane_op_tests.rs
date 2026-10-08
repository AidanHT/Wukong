//! The lane-moving SIMD ops (`Widen`, `Narrow`, `Shuffle`, `InsertLane`) and the vector-by-scalar
//! shift: their result-type tables, the verifier's acceptance of well-formed uses and its
//! diagnosis of malformed ones, and their printed form.

use crate::print::print_function;
use crate::verify::verify_function;
use crate::{narrow_result, widen_result, BinOp, Builder, CastKind, Half, MirType, NarrowKind, Op};
use wukong_span::Interner;

fn v(lane: MirType, n: u32) -> MirType {
    MirType::Vec(Box::new(lane), n)
}

#[test]
fn widen_and_narrow_result_tables() {
    use MirType::*;
    assert_eq!(widen_result(&v(I8, 16), CastKind::SExt), Some(v(I16, 8)));
    assert_eq!(widen_result(&v(I16, 8), CastKind::ZExt), Some(v(I32, 4)));
    assert_eq!(widen_result(&v(I32, 4), CastKind::SExt), Some(v(I64, 2)));
    assert_eq!(widen_result(&v(F32, 4), CastKind::FpExt), Some(v(F64, 2)));
    // No 128-bit integer lane, no float sign-extension, no odd lane count, no scalar.
    assert_eq!(widen_result(&v(I64, 2), CastKind::SExt), None);
    assert_eq!(widen_result(&v(F32, 4), CastKind::SExt), None);
    assert_eq!(widen_result(&v(I32, 3), CastKind::SExt), None);
    assert_eq!(widen_result(&I32, CastKind::SExt), None);

    assert_eq!(
        narrow_result(&v(I16, 8), NarrowKind::Trunc),
        Some(v(I8, 16))
    );
    assert_eq!(
        narrow_result(&v(I32, 4), NarrowKind::SatSigned),
        Some(v(I16, 8))
    );
    assert_eq!(
        narrow_result(&v(I16, 8), NarrowKind::SatUnsigned),
        Some(v(I8, 16))
    );
    assert_eq!(
        narrow_result(&v(I64, 2), NarrowKind::Trunc),
        Some(v(I32, 4))
    );
    assert_eq!(
        narrow_result(&v(F64, 2), NarrowKind::FpTrunc),
        Some(v(F32, 4))
    );
    // There is no packed 64-to-32 saturation to lower these to, and no 8-bit target below i8.
    assert_eq!(narrow_result(&v(I64, 2), NarrowKind::SatSigned), None);
    assert_eq!(narrow_result(&v(I8, 16), NarrowKind::Trunc), None);
    assert_eq!(narrow_result(&v(F32, 4), NarrowKind::FpTrunc), None);
}

#[test]
fn well_formed_lane_ops_verify_clean_and_print() {
    let mut it = Interner::new();
    let mut b = Builder::new(it.intern("lanes"), MirType::I32);
    let bytes = b.add_param(v(MirType::I8, 16));
    let words = b.add_param(v(MirType::I32, 4));
    let doubles = b.add_param(v(MirType::F64, 2));
    let s = b.add_param(MirType::I32);
    let lo = b.build(
        v(MirType::I16, 8),
        Op::Widen {
            v: bytes,
            half: Half::Lo,
            kind: CastKind::SExt,
        },
    );
    let hi = b.build(
        v(MirType::I16, 8),
        Op::Widen {
            v: bytes,
            half: Half::Hi,
            kind: CastKind::ZExt,
        },
    );
    b.build(
        v(MirType::I8, 16),
        Op::Narrow {
            lo,
            hi,
            kind: NarrowKind::Trunc,
        },
    );
    b.build(
        v(MirType::F32, 4),
        Op::Narrow {
            lo: doubles,
            hi: doubles,
            kind: NarrowKind::FpTrunc,
        },
    );
    b.build(
        v(MirType::I32, 4),
        Op::Shuffle {
            a: words,
            b: words,
            mask: vec![3, 6, 1, 4],
        },
    );
    b.build(v(MirType::I32, 4), Op::InsertLane(words, s, 2));
    b.build(v(MirType::I32, 4), Op::Bin(BinOp::AShr, words, s));
    b.ret(Some(s));
    let f = b.finish();
    let errors = verify_function(&f);
    assert!(errors.is_empty(), "{errors:?}");

    let text = print_function(&f, &it);
    for needle in [
        "widen.sext.lo",
        "widen.zext.hi",
        "narrow.trunc",
        "narrow.fptrunc",
        ", [3, 6, 1, 4]",
        "insertlane",
        "ashr",
    ] {
        assert!(text.contains(needle), "missing `{needle}` in:\n{text}");
    }
}

/// Build `ret 0` around one op whose result is typed `ty`, and verify it.
fn errors_of(
    params: &[MirType],
    ty: MirType,
    op: impl FnOnce(&[crate::ValueId]) -> Op,
) -> Vec<String> {
    let mut it = Interner::new();
    let mut b = Builder::new(it.intern("bad"), MirType::I32);
    let ps: Vec<crate::ValueId> = params.iter().map(|t| b.add_param(t.clone())).collect();
    b.build(ty, op(&ps));
    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    verify_function(&b.finish())
}

fn assert_reports(errors: &[String], needle: &str) {
    assert!(
        errors.iter().any(|e| e.contains(needle)),
        "expected an error containing `{needle}`, got {errors:?}"
    );
}

#[test]
fn malformed_lane_ops_are_reported() {
    let i8x16 = v(MirType::I8, 16);
    let i32x4 = v(MirType::I32, 4);
    let i64x2 = v(MirType::I64, 2);

    // A widen whose result does not halve the lanes.
    let e = errors_of(std::slice::from_ref(&i8x16), v(MirType::I16, 16), |p| {
        Op::Widen {
            v: p[0],
            half: Half::Lo,
            kind: CastKind::SExt,
        }
    });
    assert_reports(&e, "must yield <8 x i16>");

    // Sign-extending a float lane is not a widen.
    let e = errors_of(&[v(MirType::F32, 4)], v(MirType::F64, 2), |p| Op::Widen {
        v: p[0],
        half: Half::Hi,
        kind: CastKind::SExt,
    });
    assert_reports(&e, "does not apply");

    // A saturating 64-to-32 narrow has no lowering, so it is not MIR.
    let e = errors_of(&[i64x2.clone(), i64x2.clone()], i32x4.clone(), |p| {
        Op::Narrow {
            lo: p[0],
            hi: p[1],
            kind: NarrowKind::SatSigned,
        }
    });
    assert_reports(&e, "does not apply");

    // Narrow operands must agree.
    let e = errors_of(&[i32x4.clone(), i64x2.clone()], v(MirType::I16, 8), |p| {
        Op::Narrow {
            lo: p[0],
            hi: p[1],
            kind: NarrowKind::Trunc,
        }
    });
    assert_reports(&e, "narrow operands differ");

    // One mask entry per lane, each naming a lane of `a ++ b`.
    let e = errors_of(&[i32x4.clone(), i32x4.clone()], i32x4.clone(), |p| {
        Op::Shuffle {
            a: p[0],
            b: p[1],
            mask: vec![0, 1, 2],
        }
    });
    assert_reports(&e, "needs 4 mask entries");
    let e = errors_of(&[i32x4.clone(), i32x4.clone()], i32x4.clone(), |p| {
        Op::Shuffle {
            a: p[0],
            b: p[1],
            mask: vec![0, 1, 2, 8],
        }
    });
    assert_reports(&e, "out of range");

    // InsertLane: index in range, scalar of the lane type.
    let e = errors_of(&[i32x4.clone(), MirType::I32], i32x4.clone(), |p| {
        Op::InsertLane(p[0], p[1], 4)
    });
    assert_reports(&e, "insertlane index 4");
    let e = errors_of(&[i32x4.clone(), MirType::I64], i32x4.clone(), |p| {
        Op::InsertLane(p[0], p[1], 0)
    });
    assert_reports(&e, "takes a i32");

    // A vector shift's scalar amount must be the LANE type, and only shifts take one.
    let e = errors_of(&[i32x4.clone(), MirType::I64], i32x4.clone(), |p| {
        Op::Bin(BinOp::Shl, p[0], p[1])
    });
    assert_reports(&e, "expected <4 x i32>");
    let e = errors_of(&[i32x4.clone(), MirType::I32], i32x4, |p| {
        Op::Bin(BinOp::Add, p[0], p[1])
    });
    assert_reports(&e, "expected <4 x i32>");
}
