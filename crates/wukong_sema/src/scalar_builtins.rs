//! Typing for the **scalar builtins**: the operations every systems language gives its numeric
//! types and Wukong lacked — type-preserving `clamp`, an explicit single-rounding `fma`, integer bit
//! operations (`popcount`, `clz`, `ctz`, `bswap`, `rotl`, `rotr`), saturating, widening and
//! overflow-reporting arithmetic, float bit access and classification (`bits`, `f32_from_bits`, …,
//! `isnan`, `isinf`, `isfinite`, `signbit`, `copysign`, `ldexp`, `frexp`), the special functions
//! (`erfc`, `erfinv`, `lgamma`, `logaddexp`), and the FP8 conversions on `u8` storage.
//!
//! [`scalar_builtin`] is the one name table; `wukong_mir_build::scalar_builtins` lowers exactly the
//! names it accepts, from the types recorded here. A user function (or an `extern` one) of the same
//! name shadows a builtin: `type_call` resolves defined functions first.
//!
//! **Operand rules**, each a specific code so the fix is obvious:
//! * an *integer* operand of a floating-point-only builtin (`fmax`/`fmin` — they used to convert it
//!   to `f32` silently, rounding integers above 2²⁴ — and `fma`, `bits`, `isnan`, `copysign`,
//!   `erfc`, `f32_to_f8e4m3`, …) is **`E0445`**; an unsuffixed integer *literal* is still accepted
//!   and converted, as for `sqrt(2)`;
//! * a non-integer operand of an integer builtin, a bit pattern of the wrong width
//!   (`f32_from_bits(x_u16)`, `f8e4m3_to_f32(x_i32)`), a non-integer `sat_cast` target, or a 64-bit
//!   `widening_mul` (there is no 128-bit result type) is **`E0446`**;
//! * two operands of different integer types where the type *is* the operation's meaning
//!   (`sat_add`, `sat_sub`, `mulhi`, `widening_mul`, `overflowing_*`) is **`E0447`** — saturating
//!   at `i8` and at `i32` give different answers, so neither may be chosen silently. An unsuffixed
//!   literal adapts to the other operand's type, so `sat_add(x, 1)` needs no suffix.
//!
//! Anything else that is not a scalar is the usual `E0401`, a wrong argument count `E0503`.

use wukong_ast::{Expr, TypeExpr};
use wukong_span::Span;
use wukong_types::{Scalar, Ty};

use crate::Sema;

/// An 8-bit floating-point storage format (OCP FP8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp8 {
    /// E4M3FN: 4 exponent and 3 mantissa bits, no infinity, saturating conversion.
    E4M3,
    /// E5M2: 5 exponent and 2 mantissa bits, IEEE-style infinities.
    E5M2,
}

/// One scalar builtin (see the module docs and `docs/language-guide.md` for the semantics).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarBuiltin {
    /// `clamp(x, lo, hi)` — `x < lo ? lo : (x > hi ? hi : x)` in the operands' common type.
    Clamp,
    /// `fma(a, b, c)` — `a·b + c` rounded once, whatever the contraction mode.
    Fma,
    /// `popcount(x)` — the number of one bits.
    Popcount,
    /// `clz(x)` — leading zero bits (the width for zero).
    Clz,
    /// `ctz(x)` — trailing zero bits (the width for zero).
    Ctz,
    /// `bswap(x)` — the bytes reversed.
    Bswap,
    /// `rotl(x, n)` — rotate left by `n` modulo the width.
    Rotl,
    /// `rotr(x, n)` — rotate right by `n` modulo the width.
    Rotr,
    /// `sat_add(a, b)` — `a + b` clamped to the type's range.
    SatAdd,
    /// `sat_sub(a, b)` — `a − b` clamped to the type's range.
    SatSub,
    /// `sat_cast::<T>(x)` — `x` converted to the integer type `T`, clamped to its range.
    SatCast,
    /// `mulhi(a, b)` — the high half of the double-width product.
    MulHi,
    /// `widening_mul(a, b)` — the exact product in the double-width type.
    WideningMul,
    /// `overflowing_add(a, b) -> (T, bool)` — the wrapped sum and whether it overflowed.
    OverflowingAdd,
    /// `overflowing_sub(a, b) -> (T, bool)`.
    OverflowingSub,
    /// `overflowing_mul(a, b) -> (T, bool)`.
    OverflowingMul,
    /// `bits(x)` — a float's bit pattern as the same-width unsigned integer.
    Bits,
    /// `f32_from_bits(u)` / `f64_from_bits` / `f16_from_bits` / `bf16_from_bits` — the inverse.
    FromBits(Scalar),
    /// `isnan(x)`.
    IsNan,
    /// `isinf(x)`.
    IsInf,
    /// `isfinite(x)`.
    IsFinite,
    /// `signbit(x)` — the sign bit, so `signbit(-0.0)` is true.
    SignBit,
    /// `copysign(x, y)` — the magnitude of `x` with the sign of `y`.
    CopySign,
    /// `ldexp(x, n)` — `x·2ⁿ`, rounded once.
    Ldexp,
    /// `frexp(x) -> (m, e)` — `x = m·2ᵉ` with `0.5 ≤ |m| < 1`.
    Frexp,
    /// `erfc(x)` — `1 − erf(x)` without cancellation.
    Erfc,
    /// `erfinv(x)` — the inverse error function.
    Erfinv,
    /// `lgamma(x)` — `ln|Γ(x)|`.
    Lgamma,
    /// `logaddexp(a, b)` — `ln(eᵃ + eᵇ)` without overflow.
    LogAddExp,
    /// `f8e4m3_to_f32(b)` / `f8e5m2_to_f32(b)` — decode an FP8 code held in a `u8`.
    Fp8ToF32(Fp8),
    /// `f32_to_f8e4m3(x)` / `f32_to_f8e5m2(x)` — encode to an FP8 code (round to nearest even).
    F32ToFp8(Fp8),
}

/// The scalar builtin `name` denotes, or `None`.
pub fn scalar_builtin(name: &str) -> Option<ScalarBuiltin> {
    use ScalarBuiltin::*;
    Some(match name {
        "clamp" => Clamp,
        "fma" => Fma,
        "popcount" => Popcount,
        "clz" => Clz,
        "ctz" => Ctz,
        "bswap" => Bswap,
        "rotl" => Rotl,
        "rotr" => Rotr,
        "sat_add" => SatAdd,
        "sat_sub" => SatSub,
        "sat_cast" => SatCast,
        "mulhi" => MulHi,
        "widening_mul" => WideningMul,
        "overflowing_add" => OverflowingAdd,
        "overflowing_sub" => OverflowingSub,
        "overflowing_mul" => OverflowingMul,
        "bits" => Bits,
        "f32_from_bits" => FromBits(Scalar::F32),
        "f64_from_bits" => FromBits(Scalar::F64),
        "f16_from_bits" => FromBits(Scalar::F16),
        "bf16_from_bits" => FromBits(Scalar::Bf16),
        "isnan" => IsNan,
        "isinf" => IsInf,
        "isfinite" => IsFinite,
        "signbit" => SignBit,
        "copysign" => CopySign,
        "ldexp" => Ldexp,
        "frexp" => Frexp,
        "erfc" => Erfc,
        "erfinv" => Erfinv,
        "lgamma" => Lgamma,
        "logaddexp" => LogAddExp,
        "f8e4m3_to_f32" => Fp8ToF32(Fp8::E4M3),
        "f8e5m2_to_f32" => Fp8ToF32(Fp8::E5M2),
        "f32_to_f8e4m3" => F32ToFp8(Fp8::E4M3),
        "f32_to_f8e5m2" => F32ToFp8(Fp8::E5M2),
        _ => return None,
    })
}

/// The number of value arguments `b` takes.
pub fn scalar_builtin_arity(b: ScalarBuiltin) -> usize {
    use ScalarBuiltin::*;
    match b {
        Clamp | Fma => 3,
        Rotl | Rotr | SatAdd | SatSub | MulHi | WideningMul | OverflowingAdd | OverflowingSub
        | OverflowingMul | CopySign | Ldexp | LogAddExp => 2,
        Popcount | Clz | Ctz | Bswap | SatCast | Bits | FromBits(_) | IsNan | IsInf | IsFinite
        | SignBit | Frexp | Erfc | Erfinv | Lgamma | Fp8ToF32(_) | F32ToFp8(_) => 1,
    }
}

/// The unsigned integer type holding `f`'s bits (`f32` → `u32`, the 16-bit floats → `u16`).
pub fn float_bits_type(f: Scalar) -> Scalar {
    match f {
        Scalar::F64 => Scalar::U64,
        Scalar::F16 | Scalar::Bf16 => Scalar::U16,
        _ => Scalar::U32,
    }
}

/// The type a `widening_mul` of two `s` operands produces, or `None` for a 64-bit operand.
pub fn widened_int(s: Scalar) -> Option<Scalar> {
    Some(match s {
        Scalar::I8 => Scalar::I16,
        Scalar::U8 => Scalar::U16,
        Scalar::I16 => Scalar::I32,
        Scalar::U16 => Scalar::U32,
        Scalar::I32 => Scalar::I64,
        Scalar::U32 => Scalar::U64,
        _ => return None,
    })
}

/// An integer scalar the bit builtins accept: every integer type except `char`.
fn is_bit_int(s: Scalar) -> bool {
    s.is_int() && s != Scalar::Char
}

/// How one operand was classified.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Opnd {
    /// A value of this scalar type (for a float operand, possibly an integer literal standing for
    /// the float of the same value).
    Known(Scalar),
    /// An unsuffixed numeric literal, which adapts to the type around it.
    Literal(Scalar),
    /// `Unknown`/`Error`: already reported or not modelled; stay quiet.
    Lenient,
    /// Rejected (a diagnostic was emitted).
    Bad,
}

impl Opnd {
    fn scalar(self) -> Option<Scalar> {
        match self {
            Opnd::Known(s) | Opnd::Literal(s) => Some(s),
            _ => None,
        }
    }
}

/// The ordinal word for argument `i` in a diagnostic.
fn ordinal(i: usize) -> &'static str {
    ["first", "second", "third"]
        .get(i)
        .copied()
        .unwrap_or("next")
}

impl Sema<'_> {
    /// `fmax`/`fmin` compare floats. An integer operand used to be converted to `f32` without a
    /// word — `fmax(a, b)` over two `i64` above 2²⁴ returned a rounded value — so it is now `E0445`
    /// pointing at the type-preserving `max`/`min`. An unsuffixed integer literal (`fmax(x, 0)`)
    /// still converts, as before. Returns whether it reported.
    pub(crate) fn reject_integer_float_minmax(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_tys: &[Ty],
        span: Span,
    ) -> bool {
        let alt = match name {
            "fmax" => "max",
            "fmin" => "min",
            _ => return false,
        };
        for (i, (arg, ty)) in args.iter().zip(arg_tys).enumerate() {
            if let Ty::Scalar(s) = ty {
                if s.is_int() && !self.is_adaptable_num_literal(arg) {
                    self.error(
                        span,
                        "E0445",
                        format!(
                            "`{name}` compares floating-point values, but its {} operand has the \
                             integer type `{}`; use `{alt}`, which keeps the integer type (`{name}` \
                             would convert both operands to `f32`, rounding integers above 2^24)",
                            ordinal(i),
                            s.name()
                        ),
                    );
                    return true;
                }
            }
        }
        false
    }

    /// Type a call to a scalar builtin, or return `None` when `name` is not one. Diagnoses its own
    /// misuse; a rejected call types as `Unknown` so one error does not cascade.
    pub(crate) fn type_scalar_builtin(
        &mut self,
        name: &str,
        generic_args: &[TypeExpr],
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Option<Ty> {
        use ScalarBuiltin::*;
        let b = scalar_builtin(name)?;
        let want = scalar_builtin_arity(b);
        if args.len() != want {
            let plural = if want == 1 { "" } else { "s" };
            self.error(
                span,
                "E0503",
                format!(
                    "`{name}` takes exactly {want} argument{plural}, but {} were supplied",
                    args.len()
                ),
            );
            return Some(Ty::Unknown);
        }
        if b != SatCast && !generic_args.is_empty() {
            self.error(
                span,
                "E0503",
                format!("`{name}` takes no generic arguments"),
            );
            return Some(Ty::Unknown);
        }
        let ty = match b {
            Clamp => self.sb_clamp(name, args, arg_tys),
            Fma | CopySign | LogAddExp => {
                let mut ops = Vec::with_capacity(args.len());
                for i in 0..args.len() {
                    ops.push(self.sb_float(name, args, arg_tys, i));
                }
                self.sb_float_result(args, arg_tys, &ops)
            }
            Popcount | Clz | Ctz | Bswap => match self.sb_int(name, args, arg_tys, 0) {
                Opnd::Known(s) | Opnd::Literal(s) => Ty::Scalar(s),
                _ => Ty::Unknown,
            },
            Rotl | Rotr => {
                let x = self.sb_int(name, args, arg_tys, 0);
                let n = self.sb_int(name, args, arg_tys, 1);
                match (x.scalar(), n) {
                    (Some(s), Opnd::Known(_) | Opnd::Literal(_) | Opnd::Lenient) => Ty::Scalar(s),
                    _ => Ty::Unknown,
                }
            }
            SatAdd | SatSub | MulHi | WideningMul | OverflowingAdd | OverflowingSub
            | OverflowingMul => match self.sb_same_int(name, args, arg_tys) {
                Some(s) => match b {
                    WideningMul => match widened_int(s) {
                        Some(w) => Ty::Scalar(w),
                        None => {
                            self.error(
                                span,
                                "E0446",
                                format!(
                                    "`widening_mul` of two `{}` operands would need a 128-bit \
                                     result, which Wukong does not have; use `mulhi` for the high \
                                     half and `*` for the low half",
                                    s.name()
                                ),
                            );
                            Ty::Unknown
                        }
                    },
                    OverflowingAdd | OverflowingSub | OverflowingMul => {
                        Ty::Tuple(vec![Ty::Scalar(s), Ty::Scalar(Scalar::Bool)])
                    }
                    _ => Ty::Scalar(s),
                },
                None => Ty::Unknown,
            },
            SatCast => self.sb_sat_cast(name, generic_args, args, arg_tys, span),
            Bits => match self.sb_float(name, args, arg_tys, 0) {
                Opnd::Known(f) | Opnd::Literal(f) => Ty::Scalar(float_bits_type(f)),
                _ => Ty::Unknown,
            },
            FromBits(f) => {
                if self.sb_bit_pattern(name, args, arg_tys, float_bits_type(f)) {
                    Ty::Scalar(f)
                } else {
                    Ty::Unknown
                }
            }
            IsNan | IsInf | IsFinite | SignBit => match self.sb_float(name, args, arg_tys, 0) {
                Opnd::Bad => Ty::Unknown,
                _ => Ty::Scalar(Scalar::Bool),
            },
            Ldexp => {
                let x = self.sb_float(name, args, arg_tys, 0);
                let n = self.sb_int(name, args, arg_tys, 1);
                match (x, n) {
                    (Opnd::Bad, _) | (_, Opnd::Bad) => Ty::Unknown,
                    (x, _) => self.sb_float_result(&args[..1], &mut arg_tys[..1], &[x]),
                }
            }
            Frexp => match self.sb_float(name, args, arg_tys, 0) {
                Opnd::Bad | Opnd::Lenient => Ty::Unknown,
                x => {
                    let f = self.sb_float_result(args, arg_tys, &[x]);
                    Ty::Tuple(vec![f, Ty::Scalar(Scalar::I32)])
                }
            },
            Erfc | Erfinv | Lgamma => {
                let x = self.sb_float(name, args, arg_tys, 0);
                self.sb_float_result(args, arg_tys, &[x])
            }
            Fp8ToF32(_) => {
                if self.sb_bit_pattern(name, args, arg_tys, Scalar::U8) {
                    Ty::Scalar(Scalar::F32)
                } else {
                    Ty::Unknown
                }
            }
            F32ToFp8(_) => match self.sb_float(name, args, arg_tys, 0) {
                Opnd::Known(Scalar::F64) => {
                    self.error(
                        args[0].span,
                        "E0401",
                        format!(
                            "`{name}` encodes an `f32`, found an `f64`; convert it first with \
                             `as f32` (one rounding there, a second to FP8)"
                        ),
                    );
                    Ty::Unknown
                }
                Opnd::Bad => Ty::Unknown,
                _ => Ty::Scalar(Scalar::U8),
            },
        };
        Some(ty)
    }

    /// Classify float operand `i` of a floating-point builtin: a float, or an unsuffixed integer
    /// literal standing for the float of its value. A non-literal integer is `E0445`; any other
    /// value `E0401`.
    fn sb_float(&mut self, name: &str, args: &[Expr], arg_tys: &[Ty], i: usize) -> Opnd {
        let arg = &args[i];
        match &arg_tys[i] {
            Ty::Scalar(s) if s.is_float() => {
                if self.is_adaptable_num_literal(arg) {
                    Opnd::Literal(*s)
                } else {
                    Opnd::Known(*s)
                }
            }
            Ty::Scalar(s) if s.is_int() && self.is_adaptable_num_literal(arg) => {
                Opnd::Literal(Scalar::F32)
            }
            Ty::Scalar(s) if s.is_int() => {
                self.error(
                    arg.span,
                    "E0445",
                    format!(
                        "`{name}` operates on floating-point values, but its {} operand has the \
                         integer type `{}`; convert it with `as f32` or `as f64`",
                        ordinal(i),
                        s.name()
                    ),
                );
                Opnd::Bad
            }
            Ty::Unknown | Ty::Error => Opnd::Lenient,
            other => {
                self.error(
                    arg.span,
                    "E0401",
                    format!(
                        "`{name}` expects a floating-point scalar as its {} operand, found a value \
                         of type `{}`",
                        ordinal(i),
                        other.display(self.interner)
                    ),
                );
                Opnd::Bad
            }
        }
    }

    /// Classify integer operand `i` of an integer builtin: any integer type but `char`. A float,
    /// `bool` or `char` is `E0446`; a non-scalar `E0401`.
    fn sb_int(&mut self, name: &str, args: &[Expr], arg_tys: &[Ty], i: usize) -> Opnd {
        let arg = &args[i];
        match &arg_tys[i] {
            Ty::Scalar(s) if is_bit_int(*s) => {
                if self.is_adaptable_num_literal(arg) {
                    Opnd::Literal(*s)
                } else {
                    Opnd::Known(*s)
                }
            }
            Ty::Scalar(s) => {
                self.error(
                    arg.span,
                    "E0446",
                    format!(
                        "`{name}` operates on integers, but its {} operand has type `{}`",
                        ordinal(i),
                        s.name()
                    ),
                );
                Opnd::Bad
            }
            Ty::Unknown | Ty::Error => Opnd::Lenient,
            other => {
                self.error(
                    arg.span,
                    "E0401",
                    format!(
                        "`{name}` expects an integer scalar as its {} operand, found a value of \
                         type `{}`",
                        ordinal(i),
                        other.display(self.interner)
                    ),
                );
                Opnd::Bad
            }
        }
    }

    /// The float type of a floating-point builtin's result -- the widest non-literal float
    /// operand, else the widest literal (`f32` by default) -- with each unsuffixed *float* literal
    /// operand re-stamped at that type, so `fma(y, y, 0.1)` over `f64` uses the `f64` nearest 0.1
    /// rather than a widened `f32`. (An integer literal stays an integer and converts exactly.)
    /// `Unknown` when an operand was rejected. `ops[i]` classifies `args[i]`.
    fn sb_float_result(&mut self, args: &[Expr], arg_tys: &mut [Ty], ops: &[Opnd]) -> Ty {
        if ops.contains(&Opnd::Bad) {
            return Ty::Unknown;
        }
        let pick = |want_known: bool| -> Option<Ty> {
            let mut out: Option<Ty> = None;
            for op in ops {
                let s = match (op, want_known) {
                    (Opnd::Known(s), true) | (Opnd::Literal(s), false) => *s,
                    _ => continue,
                };
                let t = Ty::Scalar(s);
                out = Some(match out {
                    Some(prev) => crate::join(prev, t),
                    None => t,
                });
            }
            out
        };
        let out = pick(true)
            .or_else(|| pick(false))
            .unwrap_or(Ty::Scalar(Scalar::F32));
        if let Ty::Scalar(f) = out {
            for (arg, ty) in args.iter().zip(arg_tys.iter_mut()) {
                if matches!(ty, Ty::Scalar(s) if s.is_float()) {
                    self.sb_adapt(arg, ty, f);
                }
            }
        }
        out
    }

    /// Re-stamp an unsuffixed literal operand with `ty` (and range-check it).
    fn sb_adapt(&mut self, arg: &Expr, arg_ty: &mut Ty, ty: Scalar) {
        let t = Ty::Scalar(ty);
        if self.literal_adapts(&t, arg) {
            self.retype_adapted_literal(arg, &t);
            self.range_check_int_literal(arg, &t);
            *arg_ty = t;
        }
    }

    /// The common integer type of a two-operand builtin whose type is its meaning. A literal adapts
    /// to the other operand; two different non-literal types are `E0447`. `None` when rejected or
    /// unknown.
    fn sb_same_int(&mut self, name: &str, args: &[Expr], arg_tys: &mut [Ty]) -> Option<Scalar> {
        let a = self.sb_int(name, args, arg_tys, 0);
        let b = self.sb_int(name, args, arg_tys, 1);
        match (a, b) {
            (Opnd::Known(x), Opnd::Known(y)) => {
                if x == y {
                    Some(x)
                } else {
                    self.error(
                        args[1].span,
                        "E0447",
                        format!(
                            "`{name}` needs both operands to have one integer type — its result \
                             saturates, overflows or widens at that type — but they are `{}` and \
                             `{}`; convert one with `as`",
                            x.name(),
                            y.name()
                        ),
                    );
                    None
                }
            }
            (Opnd::Known(x), Opnd::Literal(_)) => {
                self.sb_adapt(&args[1], &mut arg_tys[1], x);
                Some(x)
            }
            (Opnd::Literal(_), Opnd::Known(y)) => {
                self.sb_adapt(&args[0], &mut arg_tys[0], y);
                Some(y)
            }
            (Opnd::Literal(x), Opnd::Literal(y)) => {
                // Two literals: the wider default (a literal past `i32` is `i64`) wins.
                let t = match crate::join(Ty::Scalar(x), Ty::Scalar(y)) {
                    Ty::Scalar(t) => t,
                    _ => x,
                };
                self.sb_adapt(&args[0], &mut arg_tys[0], t);
                self.sb_adapt(&args[1], &mut arg_tys[1], t);
                Some(t)
            }
            _ => None,
        }
    }

    /// An integer holding the `width` bits of a float or FP8 code (`width` the unsigned type of
    /// that width): any integer of exactly that width, or an unsuffixed literal, which adapts to
    /// the unsigned type. Another width is `E0446`. Returns whether the operand is acceptable.
    fn sb_bit_pattern(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_tys: &mut [Ty],
        width: Scalar,
    ) -> bool {
        match self.sb_int(name, args, arg_tys, 0) {
            Opnd::Literal(_) => {
                self.sb_adapt(&args[0], &mut arg_tys[0], width);
                true
            }
            Opnd::Known(s) => {
                if s.size() == width.size() {
                    true
                } else {
                    self.error(
                        args[0].span,
                        "E0446",
                        format!(
                            "`{name}` takes a {}-bit pattern (a `{}`), found a `{}`",
                            width.size() * 8,
                            width.name(),
                            s.name()
                        ),
                    );
                    false
                }
            }
            Opnd::Lenient => true,
            Opnd::Bad => false,
        }
    }

    /// `clamp(x, lo, hi)`: three numbers; unsuffixed literals adapt to the non-literal operands'
    /// common type, and the result is the common type of all three (as `min`/`max` compute it).
    fn sb_clamp(&mut self, name: &str, args: &[Expr], arg_tys: &mut [Ty]) -> Ty {
        let mut bad = false;
        let mut lenient = false;
        let mut anchor: Option<Ty> = None;
        for (i, (arg, ty)) in args.iter().zip(arg_tys.iter()).enumerate() {
            match ty {
                Ty::Scalar(s) if s.is_int() || s.is_float() => {
                    if !self.is_adaptable_num_literal(arg) {
                        let t = Ty::Scalar(*s);
                        anchor = Some(match anchor {
                            Some(prev) => crate::join(prev, t),
                            None => t,
                        });
                    }
                }
                Ty::Unknown | Ty::Error => lenient = true,
                other => {
                    self.error(
                        arg.span,
                        "E0401",
                        format!(
                            "`{name}` takes numbers, but its {} operand has type `{}`",
                            ordinal(i),
                            other.display(self.interner)
                        ),
                    );
                    bad = true;
                }
            }
        }
        if bad || lenient {
            return Ty::Unknown;
        }
        if let Some(Ty::Scalar(a)) = anchor {
            for (arg, ty) in args.iter().zip(arg_tys.iter_mut()) {
                self.sb_adapt(arg, ty, a);
            }
        }
        let mut out = arg_tys[0].clone();
        for t in &arg_tys[1..3] {
            out = crate::join(out, t.clone());
        }
        out
    }

    /// `sat_cast::<T>(x)`: `T` an integer type, `x` an integer or a float.
    fn sb_sat_cast(
        &mut self,
        name: &str,
        generic_args: &[TypeExpr],
        args: &[Expr],
        arg_tys: &[Ty],
        span: Span,
    ) -> Ty {
        if generic_args.len() != 1 {
            self.error(
                span,
                "E0503",
                format!(
                    "`{name}` takes one type argument, the integer type to convert to \
                     (`sat_cast::<i8>(x)`), but {} were supplied",
                    generic_args.len()
                ),
            );
            return Ty::Unknown;
        }
        let target = self.lower_type(&generic_args[0]);
        let t = match target {
            Ty::Scalar(t) if is_bit_int(t) => t,
            Ty::Unknown | Ty::Error => return Ty::Unknown,
            other => {
                self.error(
                    generic_args[0].span,
                    "E0446",
                    format!(
                        "`{name}` converts to an integer type, but its target is `{}`",
                        other.display(self.interner)
                    ),
                );
                return Ty::Unknown;
            }
        };
        match &arg_tys[0] {
            Ty::Scalar(s) if is_bit_int(*s) || s.is_float() => Ty::Scalar(t),
            Ty::Unknown | Ty::Error => Ty::Scalar(t),
            Ty::Scalar(s) => {
                self.error(
                    args[0].span,
                    "E0446",
                    format!(
                        "`{name}` converts an integer or a float, found a `{}`",
                        s.name()
                    ),
                );
                Ty::Unknown
            }
            other => {
                self.error(
                    args[0].span,
                    "E0401",
                    format!(
                        "`{name}` expects a numeric scalar, found a value of type `{}`",
                        other.display(self.interner)
                    ),
                );
                Ty::Unknown
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn errors(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (_res, diags) = crate::check(&module, &interner);
        diags
            .into_iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect()
    }

    fn assert_clean(src: &str) {
        assert!(errors(src).is_empty(), "{src}: {:?}", errors(src));
    }

    #[test]
    fn name_table_and_arity() {
        assert_eq!(scalar_builtin("popcount"), Some(ScalarBuiltin::Popcount));
        assert_eq!(
            scalar_builtin("bf16_from_bits"),
            Some(ScalarBuiltin::FromBits(Scalar::Bf16))
        );
        assert_eq!(
            scalar_builtin("f32_to_f8e5m2"),
            Some(ScalarBuiltin::F32ToFp8(Fp8::E5M2))
        );
        assert_eq!(scalar_builtin("popcnt"), None);
        assert_eq!(scalar_builtin_arity(ScalarBuiltin::Clamp), 3);
        assert_eq!(scalar_builtin_arity(ScalarBuiltin::Ldexp), 2);
        assert_eq!(scalar_builtin_arity(ScalarBuiltin::Frexp), 1);
        assert_eq!(widened_int(Scalar::U32), Some(Scalar::U64));
        assert_eq!(widened_int(Scalar::I64), None);
        assert_eq!(float_bits_type(Scalar::Bf16), Scalar::U16);
    }

    #[test]
    fn results_are_typed() {
        // Each binding's annotation is the builtin's result type: a wrong one would be E0401.
        assert_clean(
            "fn f(a: i64, b: i64, x: f32, y: f64, h: f16, q: u8, n: u32) { \
             let c: i64 = clamp(a, 0, 10); let cf: f64 = clamp(y, 0.0, 1.0); \
             let p: i64 = popcount(a); let r: u8 = rotl(q, 3); let s: u32 = bswap(n); \
             let sa: u8 = sat_add(q, 200); let ss: i64 = sat_sub(a, b); \
             let sc: i8 = sat_cast::<i8>(a); let sf: u8 = sat_cast::<u8>(x); \
             let mh: u32 = mulhi(n, n); let wm: u64 = widening_mul(n, 7); \
             let o: (i64, bool) = overflowing_mul(a, b); \
             let bx: u32 = bits(x); let by: u64 = bits(y); let bh: u16 = bits(h); \
             let fx: f32 = f32_from_bits(0x7f800000); let fy: f64 = f64_from_bits(by); \
             let fh: f16 = f16_from_bits(bh); \
             let k: bool = isnan(x) || isinf(y) || isfinite(h) || signbit(x); \
             let cs: f64 = copysign(x, y); let ld: f32 = ldexp(x, n); \
             let fr: (f64, i32) = frexp(y); \
             let sp: f32 = erfc(x) + erfinv(x) + lgamma(x) + logaddexp(x, 1.0); \
             let fm: f64 = fma(y, y, 1.0); \
             let d: f32 = f8e4m3_to_f32(q) + f8e5m2_to_f32(0x3c); \
             let en: u8 = f32_to_f8e4m3(x); let en2: u8 = f32_to_f8e5m2(h); }",
        );
    }

    #[test]
    fn fmax_on_integers_is_e0445() {
        assert_eq!(
            errors("fn f(a: i64, b: i64) -> f32 { return fmax(a, b); }"),
            ["E0445"]
        );
        assert_eq!(
            errors("fn f(a: f32, b: i32) -> f32 { return fmin(a, b); }"),
            ["E0445"]
        );
        // Literals still convert, and floats are untouched.
        assert_clean("fn f(x: f32) -> f32 { return fmax(x, 0) + fmin(3, 5); }");
        assert_eq!(
            errors("fn f(n: i32) -> f32 { return bits(n) as f32; }"),
            ["E0445"]
        );
        assert_eq!(
            errors("fn f(n: i64) -> bool { return isnan(n); }"),
            ["E0445"]
        );
    }

    #[test]
    fn integer_builtins_reject_other_operands_with_e0446() {
        assert_eq!(
            errors("fn f(x: f32) -> i32 { return popcount(x); }"),
            ["E0446"]
        );
        assert_eq!(errors("fn f(x: bool) -> i32 { return clz(x); }"), ["E0446"]);
        assert_eq!(
            errors("fn f(u: u16) -> f32 { return f32_from_bits(u); }"),
            ["E0446"]
        );
        assert_eq!(
            errors("fn f(u: i32) -> f32 { return f8e4m3_to_f32(u); }"),
            ["E0446"]
        );
        assert_eq!(
            errors("fn f(a: u64) -> u64 { return widening_mul(a, a); }"),
            ["E0446"]
        );
        assert_eq!(
            errors("fn f(a: i32) -> f32 { return sat_cast::<f32>(a); }"),
            ["E0446"]
        );
        assert_eq!(
            errors("fn f(x: f32) -> f32 { return ldexp(x, 1.5); }"),
            ["E0446"]
        );
        // Same-width integers of either signedness hold the bits.
        assert_clean("fn f(u: i32) -> f32 { return f32_from_bits(u); }");
    }

    #[test]
    fn mixed_operand_types_are_e0447() {
        assert_eq!(
            errors("fn f(a: i8, b: i32) -> i32 { return sat_add(a, b) as i32; }"),
            ["E0447"]
        );
        assert_eq!(
            errors("fn f(a: u32, b: u64) -> u32 { return mulhi(a, b); }"),
            ["E0447"]
        );
        // A literal adapts, and is range-checked at the adapted type.
        assert_clean("fn f(a: i8) -> i8 { return sat_add(a, 100); }");
        assert_eq!(
            errors("fn f(a: i8) -> i8 { return sat_add(a, 300); }"),
            ["E0431"]
        );
    }

    #[test]
    fn arity_and_turbofish() {
        assert_eq!(
            errors("fn f(x: f32) -> f32 { return fma(x, x); }"),
            ["E0503"]
        );
        assert_eq!(
            errors("fn f(a: i32) -> i8 { return sat_cast(a); }"),
            ["E0503"]
        );
        assert_eq!(
            errors("fn f(a: i32) -> i32 { return popcount::<i32>(a); }"),
            ["E0503"]
        );
        // A user function of the same name shadows the builtin.
        assert_clean("fn clamp(x: i32) -> i32 { return x; } fn f() -> i32 { return clamp(4); }");
    }
}
