//! The lane-moving SIMD ops (`Widen`, `Narrow`, `Shuffle`, `InsertLane`) and the vector shift by a
//! scalar amount, hand-built in MIR and run on both execution models.
//!
//! Every lane of every result is printed — as its integer value, or as the raw bits of a float lane
//! — and compared against expectations derived from the ops' **scalar** semantics (sign/zero
//! extension, two's-complement truncation, signed and unsigned saturation, IEEE round-to-nearest-
//! even, shift counts masked to the lane width), never recorded from compiler output. The inputs sit
//! on the edges where a lowering would go wrong: the sign bit of every width, values one past each
//! saturation bound, a NaN, a demotion tie, an overflow to infinity, and shift counts past the lane
//! width.

use wukong_mir::{BinOp, Builder, CastKind, Half, MirType, NarrowKind, Op, Program, ValueId};
use wukong_span::Interner;

fn vt(lane: &MirType, n: u32) -> MirType {
    MirType::Vec(Box::new(lane.clone()), n)
}

/// A vector of integer constants, through memory: one store per lane, one vector load.
fn int_vec(b: &mut Builder, lane: &MirType, vals: &[i128]) -> ValueId {
    let n = vals.len() as u32;
    let slot = b.alloca(MirType::Array(Box::new(lane.clone()), n));
    for (k, &x) in vals.iter().enumerate() {
        let index = b.build(MirType::I64, Op::ConstInt(k as i128, MirType::I64));
        let ptr = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index,
                elem: lane.clone(),
            },
        );
        let value = b.build(lane.clone(), Op::ConstInt(x, lane.clone()));
        b.build_void(Op::Store { ptr, value });
    }
    b.build(vt(lane, n), Op::Load(slot, vt(lane, n)))
}

/// The float twin of [`int_vec`].
fn float_vec(b: &mut Builder, lane: &MirType, vals: &[f64]) -> ValueId {
    let n = vals.len() as u32;
    let slot = b.alloca(MirType::Array(Box::new(lane.clone()), n));
    for (k, &x) in vals.iter().enumerate() {
        let index = b.build(MirType::I64, Op::ConstInt(k as i128, MirType::I64));
        let ptr = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index,
                elem: lane.clone(),
            },
        );
        let value = b.build(lane.clone(), Op::ConstFloat(x, lane.clone()));
        b.build_void(Op::Store { ptr, value });
    }
    b.build(vt(lane, n), Op::Load(slot, vt(lane, n)))
}

/// Print every lane of `v` as an `i64`: integers sign-extended, floats as their raw bits.
fn print_lanes(b: &mut Builder, print: wukong_span::Symbol, v: ValueId) {
    let MirType::Vec(lane, n) = b.value_type(v).clone() else {
        panic!("print_lanes of a scalar");
    };
    for k in 0..n {
        let x = b.build((*lane).clone(), Op::ExtractLane(v, k));
        let wide = match *lane {
            MirType::I64 => x,
            MirType::F64 => b.build(MirType::I64, Op::Cast(CastKind::Bitcast, x, MirType::I64)),
            MirType::F32 => {
                let bits = b.build(MirType::I32, Op::Cast(CastKind::Bitcast, x, MirType::I32));
                b.build(MirType::I64, Op::Cast(CastKind::SExt, bits, MirType::I64))
            }
            _ => b.build(MirType::I64, Op::Cast(CastKind::SExt, x, MirType::I64)),
        };
        b.build_void(Op::Call {
            func: print,
            args: vec![wide],
        });
    }
}

/// Verify, then run at `-O0` and `-O2` on the interpreter and natively; all four must print
/// `expected` and exit 0, and the object emitter must accept the program.
fn assert_all_backends(
    program: Program,
    main: wukong_span::Symbol,
    it: &Interner,
    expected: &[i128],
) {
    let errors = wukong_mir::verify::verify_program(&program);
    assert!(errors.is_empty(), "{errors:?}");
    let mut text = String::new();
    for x in expected {
        text.push_str(&format!("{x}\n"));
    }
    let want = (0i64, text.into_bytes());
    for opt in [0u8, 2] {
        let mut optimized = program.clone();
        wukong_opt::optimize(&mut optimized, opt);
        let errors = wukong_mir::verify::verify_program(&optimized);
        assert!(errors.is_empty(), "-O{opt}: {errors:?}");
        let interpreted =
            wukong_interp::run_with_output(&optimized, main, it).expect("interpreter run");
        assert_eq!(
            String::from_utf8_lossy(&interpreted.1),
            String::from_utf8_lossy(&want.1),
            "interpreter, -O{opt}"
        );
        assert_eq!(interpreted.0, 0);
        let native = crate::jit_run(&optimized, main, it).expect("native run");
        assert_eq!(
            String::from_utf8_lossy(&native.1),
            String::from_utf8_lossy(&want.1),
            "native, -O{opt}"
        );
        assert_eq!(native.0, 0);
        assert!(!crate::emit_object(&optimized, it)
            .expect("object emission")
            .is_empty());
    }
}

#[test]
fn widen_and_narrow_lanes_match_their_scalar_casts() {
    let mut it = Interner::new();
    let main = it.intern("main");
    let print = it.intern("print");
    let mut b = Builder::new(main, MirType::I32);
    let (i8t, i16t, i32t, i64t) = (MirType::I8, MirType::I16, MirType::I32, MirType::I64);
    let mut expected: Vec<i128> = Vec::new();

    // i8 -> i16: sign-extend the low half, zero-extend the high half.
    let bytes = int_vec(
        &mut b,
        &i8t,
        &[
            -128, -1, 0, 1, 127, 100, -100, 55, -56, 7, -7, 64, -65, 3, 126, -127,
        ],
    );
    let lo = b.build(
        vt(&i16t, 8),
        Op::Widen {
            v: bytes,
            half: Half::Lo,
            kind: CastKind::SExt,
        },
    );
    print_lanes(&mut b, print, lo);
    expected.extend([-128, -1, 0, 1, 127, 100, -100, 55]);
    let hi = b.build(
        vt(&i16t, 8),
        Op::Widen {
            v: bytes,
            half: Half::Hi,
            kind: CastKind::ZExt,
        },
    );
    print_lanes(&mut b, print, hi);
    expected.extend([200, 7, 249, 64, 191, 3, 126, 129]);

    // i16 -> i8, all three integer kinds, over values one past every bound.
    let a16 = int_vec(&mut b, &i16t, &[300, -300, 255, -129, 128, -128, 127, 0]);
    let c16 = int_vec(&mut b, &i16t, &[32767, -32768, 1, -1, 256, 511, -256, 65]);
    for (kind, lanes) in [
        (
            NarrowKind::Trunc,
            [
                44, -44, -1, 127, -128, -128, 127, 0, -1, 0, 1, -1, 0, -1, 0, 65,
            ],
        ),
        (
            NarrowKind::SatSigned,
            [
                127, -128, 127, -128, 127, -128, 127, 0, 127, -128, 1, -1, 127, 127, -128, 65,
            ],
        ),
        (
            NarrowKind::SatUnsigned,
            [-1, 0, -1, 0, -128, 0, 127, 0, -1, 0, 1, 0, -1, -1, 0, 65],
        ),
    ] {
        let r = b.build(
            vt(&i8t, 16),
            Op::Narrow {
                lo: a16,
                hi: c16,
                kind,
            },
        );
        print_lanes(&mut b, print, r);
        expected.extend(lanes);
    }

    // i32 -> i16.
    let a32 = int_vec(&mut b, &i32t, &[70000, -70000, 32768, -32769]);
    let c32 = int_vec(&mut b, &i32t, &[65535, -1, 40000, 12345]);
    for (kind, lanes) in [
        (
            NarrowKind::Trunc,
            [4464, -4464, -32768, 32767, -1, -1, -25536, 12345],
        ),
        (
            NarrowKind::SatSigned,
            [32767, -32768, 32767, -32768, 32767, -1, 32767, 12345],
        ),
        (
            NarrowKind::SatUnsigned,
            [-1, 0, -32768, 0, -1, 0, -25536, 12345],
        ),
    ] {
        let r = b.build(
            vt(&i16t, 8),
            Op::Narrow {
                lo: a32,
                hi: c32,
                kind,
            },
        );
        print_lanes(&mut b, print, r);
        expected.extend(lanes);
    }

    // i64 -> i32 truncation (the one 64-bit narrow).
    let a64 = int_vec(&mut b, &i64t, &[0x1_0000_0005, -1]);
    let c64 = int_vec(&mut b, &i64t, &[0x7FFF_FFFF_FFFF_FFFF, -(4294967296 + 7)]);
    let r = b.build(
        vt(&i32t, 4),
        Op::Narrow {
            lo: a64,
            hi: c64,
            kind: NarrowKind::Trunc,
        },
    );
    print_lanes(&mut b, print, r);
    expected.extend([5, -1, -1, -7]);

    // i32 -> i64 both ways.
    let w32 = int_vec(&mut b, &i32t, &[10, -20, 2147483647, -2147483648]);
    let hi = b.build(
        vt(&i64t, 2),
        Op::Widen {
            v: w32,
            half: Half::Hi,
            kind: CastKind::SExt,
        },
    );
    print_lanes(&mut b, print, hi);
    expected.extend([2147483647, -2147483648]);
    let lo = b.build(
        vt(&i64t, 2),
        Op::Widen {
            v: w32,
            half: Half::Lo,
            kind: CastKind::ZExt,
        },
    );
    print_lanes(&mut b, print, lo);
    expected.extend([10, 4294967276]);

    // f32 -> f64 (a NaN and the largest finite f32 ride the high half), printed as f64 bits.
    let f4 = float_vec(
        &mut b,
        &MirType::F32,
        &[1.5, -0.0, f64::NAN, 3.4028234663852886e38],
    );
    for (half, lanes) in [
        (Half::Lo, [4609434218613702656i128, -9223372036854775808]),
        (Half::Hi, [9221120237041090560, 5183643170566569984]),
    ] {
        let r = b.build(
            vt(&MirType::F64, 2),
            Op::Widen {
                v: f4,
                half,
                kind: CastKind::FpExt,
            },
        );
        print_lanes(&mut b, print, r);
        expected.extend(lanes);
    }

    // f64 -> f32: a tie that rounds to even (down), an overflow to +inf, and a tie that rounds to
    // even (up). Printed as f32 bits.
    let d_lo = float_vec(&mut b, &MirType::F64, &[1.0 + 2f64.powi(-24), -2.5]);
    let d_hi = float_vec(&mut b, &MirType::F64, &[1e40, 1.0 + 3.0 * 2f64.powi(-24)]);
    let r = b.build(
        vt(&MirType::F32, 4),
        Op::Narrow {
            lo: d_lo,
            hi: d_hi,
            kind: NarrowKind::FpTrunc,
        },
    );
    print_lanes(&mut b, print, r);
    expected.extend([1065353216, -1071644672, 2139095040, 1065353218]);

    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    assert_all_backends(program, main, &it, &expected);
}

#[test]
fn shuffle_and_insertlane_move_lanes_exactly() {
    let mut it = Interner::new();
    let main = it.intern("main");
    let print = it.intern("print");
    let mut b = Builder::new(main, MirType::I32);
    let mut expected: Vec<i128> = Vec::new();

    // Two-source dword permutation.
    let a = int_vec(&mut b, &MirType::I32, &[10, 11, 12, 13]);
    let c = int_vec(&mut b, &MirType::I32, &[20, 21, 22, 23]);
    let r = b.build(
        vt(&MirType::I32, 4),
        Op::Shuffle {
            a,
            b: c,
            mask: vec![7, 0, 5, 2],
        },
    );
    print_lanes(&mut b, print, r);
    expected.extend([23, 10, 21, 12]);

    // Byte interleave: even lanes of `a`, odd lanes of `b`.
    let a8: Vec<i128> = (0..16).collect();
    let c8: Vec<i128> = (100..116).collect();
    let a8 = int_vec(&mut b, &MirType::I8, &a8);
    let c8 = int_vec(&mut b, &MirType::I8, &c8);
    let mask: Vec<u8> = (0..16u8)
        .map(|k| if k % 2 == 0 { k } else { 16 + k })
        .collect();
    let r = b.build(vt(&MirType::I8, 16), Op::Shuffle { a: a8, b: c8, mask });
    print_lanes(&mut b, print, r);
    expected.extend([
        0, 101, 2, 103, 4, 105, 6, 107, 8, 109, 10, 111, 12, 113, 14, 115,
    ]);

    // f64 lanes across the two sources, printed as bits.
    let a64 = float_vec(&mut b, &MirType::F64, &[1.5, -2.25]);
    let c64 = float_vec(&mut b, &MirType::F64, &[4.0, 8.0]);
    let r = b.build(
        vt(&MirType::F64, 2),
        Op::Shuffle {
            a: a64,
            b: c64,
            mask: vec![1, 2],
        },
    );
    print_lanes(&mut b, print, r);
    expected.extend([-4611123068473966592, 4616189618054758400]);

    // InsertLane, integer and float.
    let h: Vec<i128> = (1..9).collect();
    let h = int_vec(&mut b, &MirType::I16, &h);
    let s = b.build(MirType::I16, Op::ConstInt(777, MirType::I16));
    let r = b.build(vt(&MirType::I16, 8), Op::InsertLane(h, s, 5));
    print_lanes(&mut b, print, r);
    expected.extend([1, 2, 3, 4, 5, 777, 7, 8]);
    let f = float_vec(&mut b, &MirType::F32, &[1.0, 2.0, 3.0, 4.0]);
    let s = b.build(MirType::F32, Op::ConstFloat(9.5, MirType::F32));
    let r = b.build(vt(&MirType::F32, 4), Op::InsertLane(f, s, 0));
    print_lanes(&mut b, print, r);
    expected.extend([1092091904, 1073741824, 1077936128, 1082130432]);

    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    assert_all_backends(program, main, &it, &expected);
}

/// `Bin(shift, Vec, scalar)` at every integer lane width, by an amount inside the lane and by one
/// past it (which both backends mask to the lane width, exactly as a scalar shift is masked).
#[test]
fn vector_shift_by_a_scalar_amount_masks_like_the_scalar_shift() {
    let mut it = Interner::new();
    let main = it.intern("main");
    let print = it.intern("print");
    let mut b = Builder::new(main, MirType::I32);
    let mut expected: Vec<i128> = Vec::new();

    let shifts = |b: &mut Builder,
                  expected: &mut Vec<i128>,
                  lane: MirType,
                  vals: &[i128],
                  amount: i128,
                  want: [&[i128]; 3]| {
        let x = int_vec(b, &lane, vals);
        let amt = b.build(lane.clone(), Op::ConstInt(amount, lane.clone()));
        for (op, lanes) in [BinOp::Shl, BinOp::LShr, BinOp::AShr].into_iter().zip(want) {
            let r = b.build(vt(&lane, vals.len() as u32), Op::Bin(op, x, amt));
            print_lanes(b, print, r);
            expected.extend_from_slice(lanes);
        }
    };

    let b8 = [
        -128, -1, 0, 1, 127, 100, -100, 55, -56, 7, -7, 64, -65, 3, 126, -127,
    ];
    let shl8: &[i128] = &[
        0, -8, 0, 8, -8, 32, -32, -72, 64, 56, -56, 0, -8, 24, -16, 8,
    ];
    let lshr8: &[i128] = &[16, 31, 0, 0, 15, 12, 19, 6, 25, 0, 31, 8, 23, 0, 15, 16];
    let ashr8: &[i128] = &[-16, -1, 0, 0, 15, 12, -13, 6, -7, 0, -1, 8, -9, 0, 15, -16];
    for amount in [3, 11] {
        shifts(
            &mut b,
            &mut expected,
            MirType::I8,
            &b8,
            amount,
            [shl8, lshr8, ashr8],
        );
    }
    shifts(
        &mut b,
        &mut expected,
        MirType::I16,
        &[-32768, -1, 0, 1, 32767, 1000, -1000, 12345],
        9,
        [
            &[0, -512, 0, 512, -512, -12288, 12288, 29184],
            &[64, 127, 0, 0, 63, 1, 126, 24],
            &[-64, -1, 0, 0, 63, 1, -2, 24],
        ],
    );
    let w32 = [-2147483648, -7, 7, 2147483647];
    shifts(
        &mut b,
        &mut expected,
        MirType::I32,
        &w32,
        31,
        [
            &[0, -2147483648, -2147483648, -2147483648],
            &[1, 1, 0, 0],
            &[-1, -1, 0, 0],
        ],
    );
    shifts(
        &mut b,
        &mut expected,
        MirType::I32,
        &w32,
        33,
        [
            &[0, -14, 14, -2],
            &[1073741824, 2147483644, 3, 1073741823],
            &[-1073741824, -4, 3, 1073741823],
        ],
    );
    let d64 = [-9223372036854775808, 5];
    shifts(
        &mut b,
        &mut expected,
        MirType::I64,
        &d64,
        63,
        [&[0, -9223372036854775808], &[1, 0], &[-1, 0]],
    );
    shifts(
        &mut b,
        &mut expected,
        MirType::I64,
        &d64,
        65,
        [
            &[0, 10],
            &[4611686018427387904, 2],
            &[-4611686018427387904, 2],
        ],
    );

    let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    b.ret(Some(zero));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    assert_all_backends(program, main, &it, &expected);
}
