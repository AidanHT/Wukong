//! Scalar numeric-contract tests for the interpreter — the crate that **is** the language spec.
//!
//! Every rule asserted here is one this crate defines and is the sole authority for: integer
//! division / remainder by zero, the `INT_MIN / -1` wrap, the shift-count mask, unsigned-width
//! windows for `ZExt`/`UDiv`/`URem`/`UiToFp`, single-rounded int->float, saturating fp->int, and
//! the NaN behaviour of the ordered float predicates.
//!
//! Before this file, every one of those contracts was held in place *only* by
//! `native_matches_interpreter` — a differential that compares the oracle against the very backend
//! the oracle defines. If both sides were edited to agree on a wrong rule (exactly what happened to
//! `Fone` on the LLVM path), no test in the repo failed. A spec whose only witness is the thing it
//! specifies is not pinned at all. These assertions are the language definition written down against
//! the free functions directly (`apply_bin`/`apply_cmp`/`apply_cast`/`mask`/`uval`/`int_to_float`),
//! independent of any backend, so a change to a rule has to change a stated truth here too.
//!
//! The reference values are hand-derived two's-complement / IEEE-754 truths, never readings taken
//! from a backend.

use super::*;

/// Extract the integer payload of a scalar result, mirroring what `exec` observes. Panics on a
/// non-integer so a contract that silently changed a result's *kind* is caught, not papered over.
fn ival(v: Value) -> i128 {
    match v {
        Value::Int(i) => i,
        Value::Ptr(p) => p as i128,
        other => panic!("expected an integer result, got {other:?}"),
    }
}

/// Extract the float payload of a scalar result.
fn fval(v: Value) -> f64 {
    match v {
        Value::Float(f) => f,
        other => panic!("expected a float result, got {other:?}"),
    }
}

/// An integer binary op, normalized to its result width exactly as `exec` does (it masks a
/// `Value::Int` whose result type `is_int()`). The width wrap is `mask`'s job; the shift-count mask
/// and the div/rem-by-zero rule live inside `apply_bin`, so this composition is the observable
/// scalar result for an `i`-typed op.
fn bin_i(op: BinOp, a: i128, b: i128, ty: &MirType) -> i128 {
    mask(
        ival(apply_bin(op, Value::Int(a), Value::Int(b), Some(ty))),
        ty,
    )
}

/// Shift counts are masked to the result width (x86 / Cranelift semantics): `1i32 << 32 == 1`, not
/// `0`, and a logical right shift reads the operand's own unsigned width window.
#[test]
fn shift_counts_are_masked_to_the_result_width() {
    use BinOp::*;
    // Left shift: the count wraps mod the width, so a count == width is a no-op, not a zero.
    assert_eq!(bin_i(Shl, 1, 8, &MirType::I8), 1);
    assert_eq!(bin_i(Shl, 1, 16, &MirType::I16), 1);
    assert_eq!(bin_i(Shl, 1, 32, &MirType::I32), 1);
    assert_eq!(bin_i(Shl, 1, 33, &MirType::I32), 2);
    assert_eq!(bin_i(Shl, 1, 64, &MirType::I64), 1);
    // 1 << 31 sets the i32 sign bit.
    assert_eq!(bin_i(Shl, 1, 31, &MirType::I32), i32::MIN as i128);

    // Logical right shift reads the unsigned width window: -1 as u32 is 0xFFFF_FFFF.
    assert_eq!(bin_i(LShr, -1, 1, &MirType::I32), 0x7FFF_FFFF);
    // Count 32 masks to 0: the whole u32 (0xFFFF_FFFF) survives, which re-signs to -1 in i32.
    assert_eq!(bin_i(LShr, -1, 32, &MirType::I32), -1);
    assert_eq!(bin_i(LShr, -1, 1, &MirType::I64), 0x7FFF_FFFF_FFFF_FFFF);

    // Arithmetic right shift keeps the sign.
    assert_eq!(bin_i(AShr, -1, 1, &MirType::I32), -1);
    assert_eq!(
        bin_i(AShr, i32::MIN as i128, 1, &MirType::I32),
        -1_073_741_824
    );
}

/// Integer division and remainder by zero yield `0` on both backends (the documented determinism
/// contract — real hardware would trap, so the oracle fixes a value rather than leaving it UB).
#[test]
fn integer_div_rem_by_zero_is_zero() {
    use BinOp::*;
    for ty in [MirType::I8, MirType::I16, MirType::I32, MirType::I64] {
        assert_eq!(bin_i(SDiv, 7, 0, &ty), 0, "SDiv/0 {ty:?}");
        assert_eq!(bin_i(UDiv, 7, 0, &ty), 0, "UDiv/0 {ty:?}");
        assert_eq!(bin_i(SRem, 7, 0, &ty), 0, "SRem/0 {ty:?}");
        assert_eq!(bin_i(URem, 7, 0, &ty), 0, "URem/0 {ty:?}");
    }
}

/// `INT_MIN / -1` overflows the signed range and must wrap to `INT_MIN` at the result width (the
/// i128 arithmetic produces `+2^(w-1)`, which the width mask re-signs); the remainder is `0`.
#[test]
fn signed_div_rem_int_min_over_minus_one_wraps() {
    use BinOp::*;
    assert_eq!(
        bin_i(SDiv, i8::MIN as i128, -1, &MirType::I8),
        i8::MIN as i128
    );
    assert_eq!(
        bin_i(SDiv, i32::MIN as i128, -1, &MirType::I32),
        i32::MIN as i128
    );
    assert_eq!(
        bin_i(SDiv, i64::MIN as i128, -1, &MirType::I64),
        i64::MIN as i128
    );
    assert_eq!(bin_i(SRem, i32::MIN as i128, -1, &MirType::I32), 0);
}

/// `UDiv`/`URem` read each operand as unsigned in its own width — a high-bit-set value is a large
/// positive number, not a negative one.
#[test]
fn unsigned_div_rem_use_the_width_window() {
    use BinOp::*;
    // -1 as u32 == 4294967295; /2 == 2147483647 (signed division would give 0).
    assert_eq!(bin_i(UDiv, -1, 2, &MirType::I32), 2_147_483_647);
    assert_eq!(bin_i(URem, -1, 10, &MirType::I32), 5);
    assert_eq!(bin_i(UDiv, -1, 2, &MirType::I64), 9_223_372_036_854_775_807);
}

/// `ZExt` widens the source's own unsigned width window; `SExt`/`Trunc` sign-extend / wrap.
#[test]
fn integer_cast_width_windows() {
    use CastKind::*;
    // ZExt of a high-bit-set value reads it as unsigned (stored sign-extended as -1).
    assert_eq!(
        ival(apply_cast(
            ZExt,
            Value::Int(-1),
            &MirType::I8,
            &MirType::I32
        )),
        255
    );
    assert_eq!(
        ival(apply_cast(
            ZExt,
            Value::Int(-1),
            &MirType::I16,
            &MirType::I32
        )),
        65_535
    );
    assert_eq!(
        ival(apply_cast(
            ZExt,
            Value::Int(i32::MIN as i128),
            &MirType::I32,
            &MirType::I64
        )),
        2_147_483_648
    );
    // SExt keeps the sign.
    assert_eq!(
        ival(apply_cast(
            SExt,
            Value::Int(-1),
            &MirType::I8,
            &MirType::I32
        )),
        -1
    );
    // Trunc wraps: 255 -> i8 == -1, 256 -> i8 == 0.
    assert_eq!(
        ival(apply_cast(
            Trunc,
            Value::Int(255),
            &MirType::I32,
            &MirType::I8
        )),
        -1
    );
    assert_eq!(
        ival(apply_cast(
            Trunc,
            Value::Int(256),
            &MirType::I32,
            &MirType::I8
        )),
        0
    );
}

/// Int->float must round in a **single** step to the target's precision: a narrow (`f32`/`bf16`/`f16`)
/// target rounds the integer straight to `f32`, an `f64` target to `f64`. Routing a wide integer
/// through `f64` first and re-rounding to `f32` double-rounds and disagrees with native above 2^53.
#[test]
fn int_to_float_rounds_once_to_the_target() {
    // Structural: the `F32`/narrow targets take the `narrow` operand, `F64` the `wide` one. These
    // artificial inputs fail loudly if someone reroutes the narrow path through the f64 value.
    assert_eq!(int_to_float(2.0, 3.0, &MirType::F32), 3.0);
    assert_eq!(int_to_float(2.0, 3.0, &MirType::F16), 3.0);
    assert_eq!(int_to_float(2.0, 3.0, &MirType::BF16), 3.0);
    assert_eq!(int_to_float(2.0, 3.0, &MirType::F64), 2.0);

    // Realistic: 2^53 + 1 to f32 must equal the single `as f32` rounding, not the f64 round-trip.
    use CastKind::SiToFp;
    let n: i64 = (1i64 << 53) + 1;
    assert_eq!(
        fval(apply_cast(
            SiToFp,
            Value::Int(n as i128),
            &MirType::I64,
            &MirType::F32
        )),
        n as f32 as f64
    );
    assert_eq!(
        fval(apply_cast(
            SiToFp,
            Value::Int(n as i128),
            &MirType::I64,
            &MirType::F64
        )),
        n as f64
    );
}

/// `UiToFp` reads the source as unsigned in its own width before converting.
#[test]
fn unsigned_to_float_uses_the_width_window() {
    use CastKind::UiToFp;
    assert_eq!(
        fval(apply_cast(
            UiToFp,
            Value::Int(-1),
            &MirType::I8,
            &MirType::F32
        )),
        255.0
    );
    assert_eq!(
        fval(apply_cast(
            UiToFp,
            Value::Int(i32::MIN as i128),
            &MirType::I32,
            &MirType::F32
        )),
        2_147_483_648.0
    );
}

/// Float->int is saturating (matching the native `fcvt_to_{sint,uint}_sat`): NaN -> 0, out-of-range
/// clamps to the target's min/max, and a negative value -> 0 for an unsigned target.
#[test]
fn float_to_int_saturates() {
    use CastKind::{FpToSi, FpToUi};
    let si = |f: f64, to: &MirType| ival(apply_cast(FpToSi, Value::Float(f), &MirType::F32, to));
    let ui = |f: f64, to: &MirType| ival(apply_cast(FpToUi, Value::Float(f), &MirType::F32, to));

    // Signed.
    assert_eq!(si(f64::NAN, &MirType::I32), 0);
    assert_eq!(si(f64::INFINITY, &MirType::I32), i32::MAX as i128);
    assert_eq!(si(f64::NEG_INFINITY, &MirType::I32), i32::MIN as i128);
    assert_eq!(si(1e30, &MirType::I8), 127);
    assert_eq!(si(-1e30, &MirType::I8), -128);
    assert_eq!(si(-5.0, &MirType::I8), -5);
    assert_eq!(si(f64::NAN, &MirType::I64), 0);

    // Unsigned: negatives and NaN saturate to 0, large magnitudes to the max.
    assert_eq!(ui(-1.0, &MirType::I32), 0);
    assert_eq!(ui(f64::NAN, &MirType::I32), 0);
    assert_eq!(ui(1e30, &MirType::I8), 255);
    assert_eq!(ui(300.0, &MirType::I8), 255);
    assert_eq!(ui(-1.0, &MirType::I64), 0);
}

/// The ordered float predicates are false whenever an operand is NaN; `Fone` (`!=`) is the lone
/// true one, including NaN != NaN. This is the exact contract a backend regression once broke.
#[test]
fn ordered_float_compares_against_nan() {
    use CmpOp::*;
    let nan = Value::Float(f64::NAN);
    let one = Value::Float(1.0);
    assert!(!apply_cmp(Foeq, nan, one));
    assert!(apply_cmp(Fone, nan, one));
    assert!(!apply_cmp(Folt, nan, one));
    assert!(!apply_cmp(Fole, nan, one));
    assert!(!apply_cmp(Fogt, nan, one));
    assert!(!apply_cmp(Foge, nan, one));
    // NaN compared with itself: equal is false, not-equal is true.
    assert!(!apply_cmp(Foeq, nan, nan));
    assert!(apply_cmp(Fone, nan, nan));
    // Sanity: ordinary ordered compares still behave.
    assert!(apply_cmp(Foeq, one, one));
    assert!(apply_cmp(Folt, one, Value::Float(2.0)));
}

/// `mask` sign-extends from the type's width; `i1` is a boolean (low bit, unsigned).
#[test]
fn mask_sign_extends_from_the_type_width() {
    assert_eq!(mask(0xFF, &MirType::I8), -1);
    assert_eq!(mask(0x7F, &MirType::I8), 127);
    assert_eq!(mask(0x80, &MirType::I8), -128);
    assert_eq!(mask(256, &MirType::I8), 0);
    assert_eq!(mask(0xFFFF, &MirType::I16), -1);
    assert_eq!(mask(70_000, &MirType::I16), 4464);
    assert_eq!(mask(0x8000_0000, &MirType::I32), i32::MIN as i128);
    // i1 keeps the low bit, unsigned: true == 1, not a sign-extended -1.
    assert_eq!(mask(-1, &MirType::I1), 1);
    assert_eq!(mask(3, &MirType::I1), 1);
    assert_eq!(mask(2, &MirType::I1), 0);
}

/// `uval` reinterprets the low `bits` of a sign-extended value as unsigned.
#[test]
fn uval_reads_the_unsigned_window() {
    assert_eq!(uval(-1, 8), 255);
    assert_eq!(uval(-1, 16), 65_535);
    assert_eq!(uval(-1, 32), 4_294_967_295);
    assert_eq!(uval(-1, 64), u64::MAX as u128);
    assert_eq!(uval(5, 8), 5);
    // A width at or past 128 is the identity reinterpretation.
    assert_eq!(uval(5, 128), 5);
}
