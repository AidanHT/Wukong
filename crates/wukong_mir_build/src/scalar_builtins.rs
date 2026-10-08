//! Lowering of the scalar builtins `wukong_sema::scalar_builtins` types: `clamp`, `fma`, the
//! integer bit operations, saturating / widening / overflow-reporting arithmetic, float bit access
//! and classification, the special functions, and the FP8 conversions.
//!
//! **Plain MIR, by design.** Everything but four bit counts and three special functions is built
//! from compares, selects, shifts, masks, `bitcast`s and adds — so it needs no backend support and
//! the interpreter and every backend agree bit-for-bit by construction (the MIR loop vectorizer
//! treats it as ordinary arithmetic). `popcount`/`clz`/`ctz`/`bswap` are one [`Op::IntUnary`] each
//! (one machine instruction; scalar only, so the vectorizer declines a loop containing one), and
//! `erfc`/`erfinv`/`lgamma` are one call of the double-precision library (`wukong_rt_math_f64`
//! with an `M64_*` code — every precision computes in `f64` and rounds once).
//!
//! **Signaling NaNs.** The interpreter holds an `f32` as an `f64`, which cannot carry a signaling
//! `f32` NaN (the widening quiets it), while native code can load one from memory. So the bit
//! builtins define the boundary: `bits(x)` never reports a signaling NaN and `*_from_bits` never
//! makes one — the quiet bit is set, every other payload bit is kept — and the two backends cannot
//! disagree about a payload.
//!
//! Every lowering returns `None` for a shape sema rejected (it reported the error), so
//! `lower_call` falls through to its usual unsupported-call path rather than miscompiling.

use wukong_ast::Expr;
use wukong_mir::{BinOp, CastKind, CmpOp, IntUnOp, MirType, Op, RoundMode, ValueId};
use wukong_sema::{scalar_builtin, DefKind, Fp8, ScalarBuiltin};
use wukong_span::Symbol;
use wukong_types::{Scalar, Ty};

use crate::FnLowerer;

/// The `wukong_rt_math_f64` codes of the library-only special functions — the compiler's copy of
/// `wukong_runtime::special::M64_*` (this crate does not depend on the runtime; the
/// `tests/run/scalar_special_functions.wk` fixture pins every value end to end).
const M64_ERFC: u32 = 100;
const M64_ERFINV: u32 = 101;
const M64_LGAMMA: u32 = 102;

/// The bit width of an integer MIR type.
fn width(ty: &MirType) -> u32 {
    match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        _ => 64,
    }
}

/// `v` in the canonical form an integer constant of `ty` takes: its low bits, sign-extended.
fn canon(v: i128, ty: &MirType) -> i128 {
    let shift = 128 - width(ty);
    (v << shift) >> shift
}

impl FnLowerer<'_> {
    /// Whether the call `name(…)` is a scalar builtin (not shadowed by a user function). All of
    /// them are pure functions of their arguments, so a `@parallel` region may evaluate one per
    /// iteration in any order.
    pub(crate) fn is_pure_scalar_builtin(&self, name: Symbol) -> bool {
        scalar_builtin(self.interner.resolve(name)).is_some()
            && !matches!(
                self.sema.defs.lookup(name).map(|d| &d.kind),
                Some(DefKind::Fn(_))
            )
    }

    /// Lower a call to a scalar builtin, or `None` for any other callee (or a call sema rejected).
    pub(crate) fn lower_scalar_builtin(
        &mut self,
        name: Symbol,
        args: &[Expr],
        e: &Expr,
    ) -> Option<ValueId> {
        use ScalarBuiltin::*;
        let b = scalar_builtin(self.interner.resolve(name))?;
        if args.len() != wukong_sema::scalar_builtin_arity(b) {
            return None;
        }
        match b {
            Clamp => self.sb_clamp(args, e),
            Fma => {
                let rty = self.sb_float_ty(e)?;
                let a = self.sb_operand(&args[0], &rty);
                let x = self.sb_operand(&args[1], &rty);
                let c = self.sb_operand(&args[2], &rty);
                let v = self.builder.build(rty.clone(), Op::Fma(a, x, c));
                Some(self.round_half_result(v, &rty))
            }
            Popcount | Clz | Ctz | Bswap => {
                let rty = self.sb_int_ty(e)?;
                let x = self.sb_operand(&args[0], &rty);
                let op = match b {
                    Popcount => IntUnOp::Popcnt,
                    Clz => IntUnOp::Clz,
                    Ctz => IntUnOp::Ctz,
                    _ => IntUnOp::Bswap,
                };
                if op == IntUnOp::Bswap && rty == MirType::I8 {
                    // One byte reversed is itself (and Cranelift has no 8-bit `bswap`).
                    return Some(x);
                }
                Some(self.builder.build(rty, Op::IntUnary(op, x)))
            }
            Rotl | Rotr => {
                let rty = self.sb_int_ty(e)?;
                let x = self.sb_operand(&args[0], &rty);
                let n = self.sb_operand(&args[1], &rty);
                Some(self.sb_rotate(x, n, &rty, b == Rotl))
            }
            SatAdd | SatSub => {
                let rty = self.sb_int_ty(e)?;
                let signed = self.signed(e);
                let a = self.sb_operand(&args[0], &rty);
                let x = self.sb_operand(&args[1], &rty);
                Some(self.sb_saturating(a, x, &rty, signed, b == SatAdd))
            }
            SatCast => self.sb_sat_cast(&args[0], e),
            MulHi => {
                let rty = self.sb_int_ty(e)?;
                let signed = self.signed(e);
                let a = self.sb_operand(&args[0], &rty);
                let x = self.sb_operand(&args[1], &rty);
                Some(self.sb_mulhi(a, x, &rty, signed))
            }
            WideningMul => {
                // The operands convert to the double-width result type by their own signedness
                // (exactly), and one multiply there cannot overflow.
                let rty = self.sb_int_ty(e)?;
                let a = self.sb_operand(&args[0], &rty);
                let x = self.sb_operand(&args[1], &rty);
                Some(self.builder.build(rty.clone(), Op::Bin(BinOp::Mul, a, x)))
            }
            OverflowingAdd | OverflowingSub | OverflowingMul => {
                let Ty::Tuple(fields) = self.expr_ty(e) else {
                    return None;
                };
                let Some(Ty::Scalar(s)) = fields.first() else {
                    return None;
                };
                let signed = s.is_signed();
                let rty = MirType::from_scalar(*s);
                if !rty.is_int() {
                    return None;
                }
                let a = self.sb_operand(&args[0], &rty);
                let x = self.sb_operand(&args[1], &rty);
                let (r, ov) = match b {
                    OverflowingAdd => self.sb_overflowing_add_sub(a, x, &rty, signed, true),
                    OverflowingSub => self.sb_overflowing_add_sub(a, x, &rty, signed, false),
                    _ => self.sb_overflowing_mul(a, x, &rty, signed),
                };
                self.sb_tuple(e, &[r, ov])
            }
            Bits => {
                let ft = self.sb_float_operand_ty(&args[0]);
                let x = self.sb_operand(&args[0], &ft);
                Some(self.sb_bits(x, &ft))
            }
            FromBits(f) => {
                let ity = MirType::from_scalar(wukong_sema::float_bits_type(f));
                let u = self.sb_operand(&args[0], &ity);
                Some(self.sb_from_bits(u, f))
            }
            IsNan | IsInf | IsFinite | SignBit => {
                let ft = self.sb_float_operand_ty(&args[0]);
                let x = self.sb_operand(&args[0], &ft);
                Some(match b {
                    IsNan => self.sb_cmp(CmpOp::Fone, x, x),
                    IsInf => {
                        let pinf = self.splat_const_f(f64::INFINITY, &ft);
                        let ninf = self.splat_const_f(f64::NEG_INFINITY, &ft);
                        let p = self.sb_cmp(CmpOp::Foeq, x, pinf);
                        let n = self.sb_cmp(CmpOp::Foeq, x, ninf);
                        self.sb_bin(BinOp::Or, p, n, &MirType::I1)
                    }
                    IsFinite => {
                        let pinf = self.splat_const_f(f64::INFINITY, &ft);
                        let ninf = self.splat_const_f(f64::NEG_INFINITY, &ft);
                        let lt = self.sb_cmp(CmpOp::Folt, x, pinf);
                        let gt = self.sb_cmp(CmpOp::Fogt, x, ninf);
                        self.sb_bin(BinOp::And, lt, gt, &MirType::I1)
                    }
                    _ => self.sb_signbit(x, &ft),
                })
            }
            CopySign => {
                let rty = self.sb_float_ty(e)?;
                let x = self.sb_operand(&args[0], &rty);
                let y = self.sb_operand(&args[1], &rty);
                Some(self.emit_copysign(x, y, &rty))
            }
            Ldexp => {
                let rty = self.sb_float_ty(e)?;
                let x = self.sb_operand(&args[0], &rty);
                let n = self.lower_expr(&args[1]);
                let nty = self.operand_mir(n, &args[1]);
                if !nty.is_int() {
                    return None;
                }
                let n_signed = self.signed(&args[1]);
                Some(self.sb_ldexp(x, &rty, n, &nty, n_signed))
            }
            Frexp => {
                let Ty::Tuple(fields) = self.expr_ty(e) else {
                    return None;
                };
                let Some(Ty::Scalar(f)) = fields.first() else {
                    return None;
                };
                let rty = MirType::from_scalar(*f);
                if !rty.is_float() {
                    return None;
                }
                let x = self.sb_operand(&args[0], &rty);
                let (m, ex) = self.sb_frexp(x, &rty);
                self.sb_tuple(e, &[m, ex])
            }
            Erfc | Erfinv | Lgamma => {
                let rty = self.sb_float_ty(e)?;
                let x = self.sb_operand(&args[0], &rty);
                let code = match b {
                    Erfc => M64_ERFC,
                    Erfinv => M64_ERFINV,
                    _ => M64_LGAMMA,
                };
                Some(self.sb_special(x, &rty, code))
            }
            LogAddExp => {
                let rty = self.sb_float_ty(e)?;
                let a = self.sb_operand(&args[0], &rty);
                let x = self.sb_operand(&args[1], &rty);
                Some(self.sb_logaddexp(a, x, &rty))
            }
            Fp8ToF32(fmt) => {
                let byte = self.sb_operand(&args[0], &MirType::I8);
                Some(self.sb_fp8_decode(byte, fmt))
            }
            F32ToFp8(fmt) => {
                let x = self.sb_operand(&args[0], &MirType::F32);
                Some(self.sb_fp8_encode(x, fmt))
            }
        }
    }

    // --- small builders ---------------------------------------------------------------------

    /// Lower `e` and convert it to `to` — an integer extends by its own signedness, an integer
    /// literal standing for a float converts exactly.
    fn sb_operand(&mut self, e: &Expr, to: &MirType) -> ValueId {
        let v = self.lower_expr(e);
        let from = self.operand_mir(v, e);
        let signed = self.signed(e);
        self.coerce_to(v, &from, to, signed)
    }

    /// The integer MIR result type of the call `e`.
    fn sb_int_ty(&self, e: &Expr) -> Option<MirType> {
        match self.expr_ty(e) {
            Ty::Scalar(s) if s.is_int() => Some(MirType::from_scalar(s)),
            _ => None,
        }
    }

    /// The float MIR result type of the call `e`.
    fn sb_float_ty(&self, e: &Expr) -> Option<MirType> {
        match self.expr_ty(e) {
            Ty::Scalar(s) if s.is_float() => Some(MirType::from_scalar(s)),
            _ => None,
        }
    }

    /// The float type a floating-point builtin reads its operand at: the operand's own float type,
    /// or `f32` for an integer literal (which sema accepted as the float of its value).
    fn sb_float_operand_ty(&self, arg: &Expr) -> MirType {
        match self.expr_ty(arg) {
            Ty::Scalar(s) if s.is_float() => MirType::from_scalar(s),
            _ => MirType::F32,
        }
    }

    /// An integer constant of `ty`, canonicalized.
    fn sb_int(&mut self, v: i128, ty: &MirType) -> ValueId {
        self.builder
            .build(ty.clone(), Op::ConstInt(canon(v, ty), ty.clone()))
    }

    fn sb_bin(&mut self, op: BinOp, a: ValueId, b: ValueId, ty: &MirType) -> ValueId {
        self.builder.build(ty.clone(), Op::Bin(op, a, b))
    }

    /// `a <op> k` with `k` an integer constant of `ty`.
    fn sb_bin_k(&mut self, op: BinOp, a: ValueId, k: i128, ty: &MirType) -> ValueId {
        let k = self.sb_int(k, ty);
        self.sb_bin(op, a, k, ty)
    }

    fn sb_cmp(&mut self, op: CmpOp, a: ValueId, b: ValueId) -> ValueId {
        self.builder.build(MirType::I1, Op::Cmp(op, a, b))
    }

    /// `a <op> k` with `k` an integer constant of `ty`.
    fn sb_cmp_k(&mut self, op: CmpOp, a: ValueId, k: i128, ty: &MirType) -> ValueId {
        let k = self.sb_int(k, ty);
        self.sb_cmp(op, a, k)
    }

    fn sb_select(&mut self, c: ValueId, a: ValueId, b: ValueId, ty: &MirType) -> ValueId {
        self.builder.build(ty.clone(), Op::Select(c, a, b))
    }

    fn sb_cast(&mut self, kind: CastKind, v: ValueId, to: &MirType) -> ValueId {
        self.builder
            .build(to.clone(), Op::Cast(kind, v, to.clone()))
    }

    /// A fresh `(T, U)` tuple buffer holding `values` at the call's tuple layout — the by-pointer
    /// aggregate value a `let (a, b) = …` or `t.0` reads, as a tuple literal builds it.
    fn sb_tuple(&mut self, e: &Expr, values: &[ValueId]) -> Option<ValueId> {
        let tty = self.expr_ty(e);
        let Ty::Tuple(fields) = &tty else {
            return None;
        };
        let (offsets, size, _) = self.aggregate_layout(fields.iter())?;
        let len = self.byte_buffer_len(Some(size));
        let buf = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::I8), len));
        for (&value, &off) in values.iter().zip(&offsets) {
            let ptr = self.field_ptr(buf, off);
            self.builder.build_void(Op::Store { ptr, value });
        }
        Some(buf)
    }

    // --- type-preserving min/max family ------------------------------------------------------

    /// `clamp(x, lo, hi)` = `x < lo ? lo : (x > hi ? hi : x)` in the result type, compared in its
    /// signedness (floats ordered: a NaN `x` passes through, as in Rust).
    fn sb_clamp(&mut self, args: &[Expr], e: &Expr) -> Option<ValueId> {
        let Ty::Scalar(s) = self.expr_ty(e) else {
            return None;
        };
        if !(s.is_int() || s.is_float()) {
            return None;
        }
        let rty = MirType::from_scalar(s);
        let x = self.sb_operand(&args[0], &rty);
        let lo = self.sb_operand(&args[1], &rty);
        let hi = self.sb_operand(&args[2], &rty);
        let (lt, gt) = if s.is_float() {
            (CmpOp::Folt, CmpOp::Fogt)
        } else if s.is_signed() {
            (CmpOp::Slt, CmpOp::Sgt)
        } else {
            (CmpOp::Ult, CmpOp::Ugt)
        };
        let above = self.sb_cmp(gt, x, hi);
        let capped = self.sb_select(above, hi, x, &rty);
        let below = self.sb_cmp(lt, x, lo);
        Some(self.sb_select(below, lo, capped, &rty))
    }

    // --- integer bit and arithmetic operations -----------------------------------------------

    /// Rotate `x` by `n` (both of type `ty`) modulo the width: `(x << k) | (x >> ((0 − k) & m))`
    /// with `k = n & m`, `m = width − 1` — every shift amount is in range, and `k = 0` is `x | x`.
    fn sb_rotate(&mut self, x: ValueId, n: ValueId, ty: &MirType, left: bool) -> ValueId {
        let m = i128::from(width(ty) - 1);
        let k = self.sb_bin_k(BinOp::And, n, m, ty);
        let zero = self.sb_int(0, ty);
        let neg = self.sb_bin(BinOp::Sub, zero, k, ty);
        let kc = self.sb_bin_k(BinOp::And, neg, m, ty);
        let (first, second) = if left {
            (BinOp::Shl, BinOp::LShr)
        } else {
            (BinOp::LShr, BinOp::Shl)
        };
        let a = self.sb_bin(first, x, k, ty);
        let c = self.sb_bin(second, x, kc, ty);
        self.sb_bin(BinOp::Or, a, c, ty)
    }

    /// `ty`'s range as `(min, max)` in canonical constants.
    fn sb_range(&mut self, ty: &MirType, signed: bool) -> (ValueId, ValueId) {
        let w = width(ty);
        let (lo, hi) = if signed {
            (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1)
        } else {
            (0, (1i128 << w) - 1)
        };
        (self.sb_int(lo, ty), self.sb_int(hi, ty))
    }

    /// `sat_add` / `sat_sub`: the wrapped result, replaced by the bound it crossed. Unsigned: an
    /// add overflowed iff the sum is below an operand, a subtract iff `a < b`. Signed: the
    /// classic sign tests `((a ^ r) & (b ^ r)) < 0` (add) and `((a ^ b) & (a ^ r)) < 0`
    /// (subtract), saturating toward the sign of `a`.
    fn sb_saturating(
        &mut self,
        a: ValueId,
        b: ValueId,
        ty: &MirType,
        signed: bool,
        add: bool,
    ) -> ValueId {
        let (r, overflow) = self.sb_overflowing_add_sub(a, b, ty, signed, add);
        let (min, max) = self.sb_range(ty, signed);
        let bound = if signed {
            let neg = self.sb_cmp_k(CmpOp::Slt, a, 0, ty);
            self.sb_select(neg, min, max, ty)
        } else if add {
            max
        } else {
            min
        };
        self.sb_select(overflow, bound, r, ty)
    }

    /// The wrapped `a ± b` and whether it overflowed `ty` (see [`Self::sb_saturating`]).
    fn sb_overflowing_add_sub(
        &mut self,
        a: ValueId,
        b: ValueId,
        ty: &MirType,
        signed: bool,
        add: bool,
    ) -> (ValueId, ValueId) {
        let op = if add { BinOp::Add } else { BinOp::Sub };
        let r = self.sb_bin(op, a, b, ty);
        let overflow = if signed {
            let t1 = if add {
                self.sb_bin(BinOp::Xor, a, r, ty)
            } else {
                self.sb_bin(BinOp::Xor, a, b, ty)
            };
            let t2 = if add {
                self.sb_bin(BinOp::Xor, b, r, ty)
            } else {
                self.sb_bin(BinOp::Xor, a, r, ty)
            };
            let both = self.sb_bin(BinOp::And, t1, t2, ty);
            self.sb_cmp_k(CmpOp::Slt, both, 0, ty)
        } else if add {
            self.sb_cmp(CmpOp::Ult, r, a)
        } else {
            self.sb_cmp(CmpOp::Ult, a, b)
        };
        (r, overflow)
    }

    /// The wrapped `a · b` and whether it overflowed `ty`. Below 64 bits the exact product is
    /// formed in `i64` and must survive the round trip through `ty`; at 64 bits the high half of
    /// the 128-bit product must be the sign extension of the low half (all zero, unsigned).
    fn sb_overflowing_mul(
        &mut self,
        a: ValueId,
        b: ValueId,
        ty: &MirType,
        signed: bool,
    ) -> (ValueId, ValueId) {
        let i64t = MirType::I64;
        if width(ty) < 64 {
            let a64 = self.coerce_to(a, ty, &i64t, signed);
            let b64 = self.coerce_to(b, ty, &i64t, signed);
            let wide = self.sb_bin(BinOp::Mul, a64, b64, &i64t);
            let r = self.sb_cast(CastKind::Trunc, wide, ty);
            let back = self.coerce_to(r, ty, &i64t, signed);
            let overflow = self.sb_cmp(CmpOp::Ne, back, wide);
            return (r, overflow);
        }
        let r = self.sb_bin(BinOp::Mul, a, b, ty);
        let hi = self.sb_mulhi(a, b, ty, signed);
        let expect = if signed {
            self.sb_bin_k(BinOp::AShr, r, 63, ty)
        } else {
            self.sb_int(0, ty)
        };
        let overflow = self.sb_cmp(CmpOp::Ne, hi, expect);
        (r, overflow)
    }

    /// The high half of the double-width product of `a` and `b` (type `ty`). Below 64 bits: one
    /// exact `i64` multiply and a shift. At 64 bits the unsigned high half comes from four
    /// 32×32→64 partial products (Hacker's Delight `mulhu`), and the signed one subtracts the
    /// operands' sign corrections: `mulhs = mulhu − (a < 0 ? b : 0) − (b < 0 ? a : 0)`.
    fn sb_mulhi(&mut self, a: ValueId, b: ValueId, ty: &MirType, signed: bool) -> ValueId {
        let i64t = MirType::I64;
        let w = width(ty);
        if w < 64 {
            let a64 = self.coerce_to(a, ty, &i64t, signed);
            let b64 = self.coerce_to(b, ty, &i64t, signed);
            let p = self.sb_bin(BinOp::Mul, a64, b64, &i64t);
            let shr = if signed { BinOp::AShr } else { BinOp::LShr };
            let hi = self.sb_bin_k(shr, p, i128::from(w), &i64t);
            return self.sb_cast(CastKind::Trunc, hi, ty);
        }
        let mask = 0xffff_ffff;
        let a0 = self.sb_bin_k(BinOp::And, a, mask, ty);
        let a1 = self.sb_bin_k(BinOp::LShr, a, 32, ty);
        let b0 = self.sb_bin_k(BinOp::And, b, mask, ty);
        let b1 = self.sb_bin_k(BinOp::LShr, b, 32, ty);
        let w0 = self.sb_bin(BinOp::Mul, a0, b0, ty);
        let w0h = self.sb_bin_k(BinOp::LShr, w0, 32, ty);
        let a1b0 = self.sb_bin(BinOp::Mul, a1, b0, ty);
        let t = self.sb_bin(BinOp::Add, a1b0, w0h, ty);
        let w1 = self.sb_bin_k(BinOp::And, t, mask, ty);
        let w2 = self.sb_bin_k(BinOp::LShr, t, 32, ty);
        let a0b1 = self.sb_bin(BinOp::Mul, a0, b1, ty);
        let w1b = self.sb_bin(BinOp::Add, a0b1, w1, ty);
        let a1b1 = self.sb_bin(BinOp::Mul, a1, b1, ty);
        let s = self.sb_bin(BinOp::Add, a1b1, w2, ty);
        let w1h = self.sb_bin_k(BinOp::LShr, w1b, 32, ty);
        let hu = self.sb_bin(BinOp::Add, s, w1h, ty);
        if !signed {
            return hu;
        }
        let zero = self.sb_int(0, ty);
        let a_neg = self.sb_cmp(CmpOp::Slt, a, zero);
        let fix_a = self.sb_select(a_neg, b, zero, ty);
        let b_neg = self.sb_cmp(CmpOp::Slt, b, zero);
        let fix_b = self.sb_select(b_neg, a, zero, ty);
        let h1 = self.sb_bin(BinOp::Sub, hu, fix_a, ty);
        self.sb_bin(BinOp::Sub, h1, fix_b, ty)
    }

    /// `sat_cast::<T>(x)`: a float converts with `as` (already saturating, NaN to 0); an integer is
    /// clamped to `T`'s range in its own type — only the bounds `T` actually has inside the
    /// source's range are tested — and then extended or truncated (losslessly, by now).
    fn sb_sat_cast(&mut self, arg: &Expr, e: &Expr) -> Option<ValueId> {
        let Ty::Scalar(t) = self.expr_ty(e) else {
            return None;
        };
        let tty = MirType::from_scalar(t);
        if !tty.is_int() {
            return None;
        }
        let v = self.lower_expr(arg);
        let from = self.operand_mir(v, arg);
        if from.is_float() {
            return Some(self.coerce_to(v, &from, &tty, t.is_signed()));
        }
        if !from.is_int() {
            return None;
        }
        let ss = self.signed(arg);
        let range = |w: u32, signed: bool| -> (i128, i128) {
            if signed {
                (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1)
            } else {
                (0, (1i128 << w) - 1)
            }
        };
        let (lo_s, hi_s) = range(width(&from), ss);
        let (lo_t, hi_t) = range(width(&tty), t.is_signed());
        let mut y = v;
        if hi_t < hi_s {
            let gt = if ss { CmpOp::Sgt } else { CmpOp::Ugt };
            let k = self.sb_int(hi_t, &from);
            let c = self.sb_cmp(gt, y, k);
            y = self.sb_select(c, k, y, &from);
        }
        if lo_t > lo_s {
            let lt = if ss { CmpOp::Slt } else { CmpOp::Ult };
            let k = self.sb_int(lo_t, &from);
            let c = self.sb_cmp(lt, y, k);
            y = self.sb_select(c, k, y, &from);
        }
        Some(self.coerce_to(y, &from, &tty, ss))
    }

    // --- float bits and classification ---------------------------------------------------------

    /// Set the quiet bit of a signaling NaN held as raw bits (`bits` an `i32` or `i64` — `f32` or
    /// `f64` layout); every other pattern passes through.
    fn sb_quiet(&mut self, bits: ValueId, ty: &MirType) -> ValueId {
        let (exp_quiet, exp, payload, quiet): (i128, i128, i128, i128) = if width(ty) == 64 {
            (
                0x7ff8_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0x0007_ffff_ffff_ffff,
                0x0008_0000_0000_0000,
            )
        } else {
            (0x7fc0_0000, 0x7f80_0000, 0x003f_ffff, 0x0040_0000)
        };
        // A signaling NaN: exponent all ones, quiet bit clear, some payload bit set.
        let head = self.sb_bin_k(BinOp::And, bits, exp_quiet, ty);
        let is_exp = self.sb_cmp_k(CmpOp::Eq, head, exp, ty);
        let tail = self.sb_bin_k(BinOp::And, bits, payload, ty);
        let has_payload = self.sb_cmp_k(CmpOp::Ne, tail, 0, ty);
        let snan = self.sb_bin(BinOp::And, is_exp, has_payload, &MirType::I1);
        let quieted = self.sb_bin_k(BinOp::Or, bits, quiet, ty);
        self.sb_select(snan, quieted, bits, ty)
    }

    /// `bits(x)`: the IEEE bit pattern of `x` (type `ft`) as the same-width integer.
    fn sb_bits(&mut self, x: ValueId, ft: &MirType) -> ValueId {
        let i32t = MirType::I32;
        match ft {
            MirType::F64 => {
                let b = self.sb_cast(CastKind::Bitcast, x, &MirType::I64);
                self.sb_quiet(b, &MirType::I64)
            }
            MirType::F32 => {
                let b = self.sb_cast(CastKind::Bitcast, x, &i32t);
                self.sb_quiet(b, &i32t)
            }
            // A half lives as the f32 of its value (widening is exact, and its NaNs are already
            // quiet): bf16 is the top half of those bits, f16 is re-encoded from them.
            MirType::BF16 => {
                let w = self.sb_cast(CastKind::FpExt, x, &MirType::F32);
                let b = self.sb_cast(CastKind::Bitcast, w, &i32t);
                let h = self.sb_bin_k(BinOp::LShr, b, 16, &i32t);
                self.sb_cast(CastKind::Trunc, h, &MirType::I16)
            }
            _ => {
                let w = self.sb_cast(CastKind::FpExt, x, &MirType::F32);
                let b = self.sb_cast(CastKind::Bitcast, w, &i32t);
                let h = self.sb_f32_bits_to_f16_bits(b);
                self.sb_cast(CastKind::Trunc, h, &MirType::I16)
            }
        }
    }

    /// The `f16` encoding (in the low 16 bits of an `i32`) of the `f32` with bits `b`, for a value
    /// an `f16` represents exactly: rebias a normal exponent, shift a subnormal's significand into
    /// place (`(1.m) >> (126 − e)` for f32 exponents 103..=112), keep an infinity or NaN payload.
    fn sb_f32_bits_to_f16_bits(&mut self, b: ValueId) -> ValueId {
        let t = MirType::I32;
        let hi = self.sb_bin_k(BinOp::LShr, b, 16, &t);
        let sign = self.sb_bin_k(BinOp::And, hi, 0x8000, &t);
        let shifted = self.sb_bin_k(BinOp::LShr, b, 23, &t);
        let e = self.sb_bin_k(BinOp::And, shifted, 0xff, &t);
        let m = self.sb_bin_k(BinOp::And, b, 0x007f_ffff, &t);
        let m13 = self.sb_bin_k(BinOp::LShr, m, 13, &t);
        // Normal: exponent e − 112, the top ten mantissa bits.
        let e16 = self.sb_bin_k(BinOp::Sub, e, 112, &t);
        let e16s = self.sb_bin_k(BinOp::Shl, e16, 10, &t);
        let normal = self.sb_bin(BinOp::Or, e16s, m13, &t);
        // Subnormal: the significand with its implicit one, shifted down by 126 − e.
        let sig = self.sb_bin_k(BinOp::Or, m, 0x0080_0000, &t);
        let k126 = self.sb_int(126, &t);
        let amount = self.sb_bin(BinOp::Sub, k126, e, &t);
        let sub = self.sb_bin(BinOp::LShr, sig, amount, &t);
        // Infinity / NaN: the payload's top ten bits (an f16 NaN's quiet bit is among them).
        let special = self.sb_bin_k(BinOp::Or, m13, 0x7c00, &t);
        let zero = self.sb_int(0, &t);
        let is_zero = self.sb_cmp_k(CmpOp::Eq, e, 0, &t);
        let low = self.sb_select(is_zero, zero, sub, &t);
        let is_normal = self.sb_cmp_k(CmpOp::Uge, e, 113, &t);
        let finite = self.sb_select(is_normal, normal, low, &t);
        let is_special = self.sb_cmp_k(CmpOp::Eq, e, 0xff, &t);
        let mag = self.sb_select(is_special, special, finite, &t);
        self.sb_bin(BinOp::Or, sign, mag, &t)
    }

    /// The `f32` bits (an `i32`) of the `f16` whose encoding is the low 16 bits of `hz` (an `i32`,
    /// zero-extended). Exact; a NaN keeps its payload with the quiet bit set.
    fn sb_f16_bits_to_f32_bits(&mut self, hz: ValueId) -> ValueId {
        let t = MirType::I32;
        let s = self.sb_bin_k(BinOp::And, hz, 0x8000, &t);
        let sign = self.sb_bin_k(BinOp::Shl, s, 16, &t);
        let shifted = self.sb_bin_k(BinOp::LShr, hz, 10, &t);
        let e = self.sb_bin_k(BinOp::And, shifted, 0x1f, &t);
        let m = self.sb_bin_k(BinOp::And, hz, 0x3ff, &t);
        // Normal: exponent e + 112, mantissa moved up 13 bits.
        let e32 = self.sb_bin_k(BinOp::Add, e, 112, &t);
        let e32s = self.sb_bin_k(BinOp::Shl, e32, 23, &t);
        let m13 = self.sb_bin_k(BinOp::Shl, m, 13, &t);
        let normal = self.sb_bin(BinOp::Or, e32s, m13, &t);
        // Subnormal: m · 2⁻²⁴, formed in f32 (both steps exact) and read back as bits.
        let mf = self.sb_cast(CastKind::SiToFp, m, &MirType::F32);
        let scale = self.splat_const_f(f64::from_bits(0x3e70_0000_0000_0000), &MirType::F32);
        let subf = self
            .builder
            .build(MirType::F32, Op::Bin(BinOp::FMul, mf, scale));
        let sub = self.sb_cast(CastKind::Bitcast, subf, &t);
        // Infinity / NaN, a NaN quieted.
        let has_payload = self.sb_cmp_k(CmpOp::Ne, m, 0, &t);
        let q = self.sb_int(0x200, &t);
        let zero = self.sb_int(0, &t);
        let qbit = self.sb_select(has_payload, q, zero, &t);
        let mq = self.sb_bin(BinOp::Or, m, qbit, &t);
        let mq13 = self.sb_bin_k(BinOp::Shl, mq, 13, &t);
        let special = self.sb_bin_k(BinOp::Or, mq13, 0x7f80_0000, &t);
        let is_zero_exp = self.sb_cmp_k(CmpOp::Eq, e, 0, &t);
        let finite = self.sb_select(is_zero_exp, sub, normal, &t);
        let is_special = self.sb_cmp_k(CmpOp::Eq, e, 0x1f, &t);
        let mag = self.sb_select(is_special, special, finite, &t);
        self.sb_bin(BinOp::Or, sign, mag, &t)
    }

    /// `f32_from_bits` / `f64_from_bits` / `bf16_from_bits` / `f16_from_bits` of the integer `u`
    /// (already the float's width).
    fn sb_from_bits(&mut self, u: ValueId, f: Scalar) -> ValueId {
        let i32t = MirType::I32;
        match f {
            Scalar::F64 => {
                let q = self.sb_quiet(u, &MirType::I64);
                self.sb_cast(CastKind::Bitcast, q, &MirType::F64)
            }
            Scalar::Bf16 => {
                let z = self.sb_cast(CastKind::ZExt, u, &i32t);
                let b = self.sb_bin_k(BinOp::Shl, z, 16, &i32t);
                let q = self.sb_quiet(b, &i32t);
                let w = self.sb_cast(CastKind::Bitcast, q, &MirType::F32);
                self.sb_cast(CastKind::FpTrunc, w, &MirType::BF16)
            }
            Scalar::F16 => {
                let z = self.sb_cast(CastKind::ZExt, u, &i32t);
                let b = self.sb_f16_bits_to_f32_bits(z);
                let w = self.sb_cast(CastKind::Bitcast, b, &MirType::F32);
                self.sb_cast(CastKind::FpTrunc, w, &MirType::F16)
            }
            _ => {
                let q = self.sb_quiet(u, &i32t);
                self.sb_cast(CastKind::Bitcast, q, &MirType::F32)
            }
        }
    }

    /// `signbit(x)`: the sign bit of the bit pattern (so `-0.0` and a negative NaN are negative).
    fn sb_signbit(&mut self, x: ValueId, ft: &MirType) -> ValueId {
        if *ft == MirType::F64 {
            let b = self.sb_cast(CastKind::Bitcast, x, &MirType::I64);
            return self.sb_cmp_k(CmpOp::Slt, b, 0, &MirType::I64);
        }
        let w = if *ft == MirType::F32 {
            x
        } else {
            self.sb_cast(CastKind::FpExt, x, &MirType::F32)
        };
        let b = self.sb_cast(CastKind::Bitcast, w, &MirType::I32);
        self.sb_cmp_k(CmpOp::Slt, b, 0, &MirType::I32)
    }

    /// `ldexp(x, n)` = `x·2ⁿ` rounded once (to nearest even, subnormals included). `f32` and the
    /// halves scale exactly in `f64` — `n` clamped to ±300, beyond which every result has already
    /// overflowed or vanished — and round once on the way back. `f64` is musl's `scalbn` with
    /// selects for its branches: at most two `2¹⁰²³` or `2⁻⁹⁶⁹` prescales and one final multiply.
    fn sb_ldexp(
        &mut self,
        x: ValueId,
        ty: &MirType,
        n: ValueId,
        nty: &MirType,
        n_signed: bool,
    ) -> ValueId {
        let i64t = MirType::I64;
        let f64t = MirType::F64;
        let n64 = self.coerce_to(n, nty, &i64t, n_signed);
        // An unsigned 64-bit count above i64::MAX is a huge positive scale, not a negative one.
        let unsigned64 = !n_signed && width(nty) == 64;
        let limit = if *ty == f64t { 2200 } else { 300 };
        let upper = if unsigned64 { CmpOp::Ugt } else { CmpOp::Sgt };
        let hi_k = self.sb_int(limit, &i64t);
        let over = self.sb_cmp(upper, n64, hi_k);
        let mut c = self.sb_select(over, hi_k, n64, &i64t);
        if !unsigned64 {
            let lo_k = self.sb_int(-limit, &i64t);
            let under = self.sb_cmp(CmpOp::Slt, c, lo_k);
            c = self.sb_select(under, lo_k, c, &i64t);
        }
        if *ty != f64t {
            let x32 = if *ty == MirType::F32 {
                x
            } else {
                self.sb_cast(CastKind::FpExt, x, &MirType::F32)
            };
            let x64 = self.sb_cast(CastKind::FpExt, x32, &f64t);
            let scale = self.sb_pow2_f64(c);
            let p = self
                .builder
                .build(f64t.clone(), Op::Bin(BinOp::FMul, x64, scale));
            let r = self.sb_cast(CastKind::FpTrunc, p, &MirType::F32);
            return if *ty == MirType::F32 {
                r
            } else {
                self.sb_cast(CastKind::FpTrunc, r, ty)
            };
        }
        let up = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
        let down = f64::from_bits(0x0360_0000_0000_0000); // 2^-969 = 2^-1022 · 2^53
        let mut y = x;
        for _ in 0..2 {
            let k = self.sb_int(1023, &i64t);
            let big = self.sb_cmp(CmpOp::Sgt, c, k);
            let f = self.splat_const_f(up, &f64t);
            let scaled = self.builder.build(f64t.clone(), Op::Bin(BinOp::FMul, y, f));
            y = self.sb_select(big, scaled, y, &f64t);
            let less = self.sb_bin(BinOp::Sub, c, k, &i64t);
            c = self.sb_select(big, less, c, &i64t);
        }
        let k = self.sb_int(1023, &i64t);
        let big = self.sb_cmp(CmpOp::Sgt, c, k);
        c = self.sb_select(big, k, c, &i64t);
        for _ in 0..2 {
            let k = self.sb_int(-1022, &i64t);
            let small = self.sb_cmp(CmpOp::Slt, c, k);
            let f = self.splat_const_f(down, &f64t);
            let scaled = self.builder.build(f64t.clone(), Op::Bin(BinOp::FMul, y, f));
            y = self.sb_select(small, scaled, y, &f64t);
            let more = self.sb_bin_k(BinOp::Add, c, 969, &i64t);
            c = self.sb_select(small, more, c, &i64t);
        }
        let k = self.sb_int(-1022, &i64t);
        let small = self.sb_cmp(CmpOp::Slt, c, k);
        c = self.sb_select(small, k, c, &i64t);
        let scale = self.sb_pow2_f64(c);
        self.builder.build(f64t, Op::Bin(BinOp::FMul, y, scale))
    }

    /// `2ᶜ` as an `f64`, for an `i64` `c` in the normal exponent range: the bits `(c + 1023) << 52`.
    fn sb_pow2_f64(&mut self, c: ValueId) -> ValueId {
        let i64t = MirType::I64;
        let biased = self.sb_bin_k(BinOp::Add, c, 1023, &i64t);
        let bits = self.sb_bin_k(BinOp::Shl, biased, 52, &i64t);
        self.sb_cast(CastKind::Bitcast, bits, &MirType::F64)
    }

    /// `frexp(x)`: `(m, e)` with `x = m·2ᵉ`, `0.5 ≤ |m| < 1` and `m` carrying `x`'s sign; a
    /// subnormal is first scaled up exactly (by 2³² or 2⁶⁴). `±0`, `±∞` and NaN give `(x, 0)`.
    fn sb_frexp(&mut self, x: ValueId, ty: &MirType) -> (ValueId, ValueId) {
        let i32t = MirType::I32;
        if *ty == MirType::F64 {
            let i64t = MirType::I64;
            let (m, e) = self.sb_frexp_bits(x, &MirType::F64, &i64t);
            let e32 = self.sb_cast(CastKind::Trunc, e, &i32t);
            return (m, e32);
        }
        let x32 = if *ty == MirType::F32 {
            x
        } else {
            self.sb_cast(CastKind::FpExt, x, &MirType::F32)
        };
        let (m, e) = self.sb_frexp_bits(x32, &MirType::F32, &i32t);
        let m = if *ty == MirType::F32 {
            m
        } else {
            // |m| in [0.5, 1) with at most 11 significant bits: exact in either half type.
            self.sb_cast(CastKind::FpTrunc, m, ty)
        };
        (m, e)
    }

    /// [`Self::sb_frexp`] on an `f32` (with `it` = `i32`) or an `f64` (`it` = `i64`); the exponent
    /// comes back in `it`.
    fn sb_frexp_bits(&mut self, x: ValueId, ft: &MirType, it: &MirType) -> (ValueId, ValueId) {
        let (mbits, emask, bias, scale_exp, keep, half_exp): (i128, i128, i128, i128, i128, i128) =
            if *ft == MirType::F64 {
                (
                    52,
                    0x7ff,
                    1022,
                    64,
                    0x800f_ffff_ffff_ffff,
                    0x3fe0_0000_0000_0000,
                )
            } else {
                (23, 0xff, 126, 32, 0x807f_ffff, 0x3f00_0000)
            };
        let b = self.sb_cast(CastKind::Bitcast, x, it);
        let sh = self.sb_bin_k(BinOp::LShr, b, mbits, it);
        let e = self.sb_bin_k(BinOp::And, sh, emask, it);
        let is_sub = self.sb_cmp_k(CmpOp::Eq, e, 0, it);
        // 2^32 (f32) or 2^64 (f64) — exact powers of two in either type.
        let factor = self.splat_const_f(2f64.powi(scale_exp as i32), ft);
        let up = self
            .builder
            .build(ft.clone(), Op::Bin(BinOp::FMul, x, factor));
        let xs = self.sb_select(is_sub, up, x, ft);
        let bs = self.sb_cast(CastKind::Bitcast, xs, it);
        let shs = self.sb_bin_k(BinOp::LShr, bs, mbits, it);
        let es = self.sb_bin_k(BinOp::And, shs, emask, it);
        let adj_k = self.sb_int(scale_exp, it);
        let zero = self.sb_int(0, it);
        let adj = self.sb_select(is_sub, adj_k, zero, it);
        let unbiased = self.sb_bin_k(BinOp::Sub, es, bias, it);
        let ex = self.sb_bin(BinOp::Sub, unbiased, adj, it);
        let kept = self.sb_bin_k(BinOp::And, bs, keep, it);
        let mb = self.sb_bin_k(BinOp::Or, kept, half_exp, it);
        let mant = self.sb_cast(CastKind::Bitcast, mb, ft);
        let fzero = self.splat_const_f(0.0, ft);
        let is_zero = self.sb_cmp(CmpOp::Foeq, x, fzero);
        let is_special = self.sb_cmp_k(CmpOp::Eq, e, emask, it);
        let pass = self.sb_bin(BinOp::Or, is_zero, is_special, &MirType::I1);
        let m = self.sb_select(pass, x, mant, ft);
        let e_out = self.sb_select(pass, zero, ex, it);
        (m, e_out)
    }

    // --- special functions --------------------------------------------------------------------

    /// One call of the double-precision library: an `f64` directly, any other float widened
    /// (exactly), evaluated in `f64`, and rounded back once.
    fn sb_special(&mut self, x: ValueId, ty: &MirType, code: u32) -> ValueId {
        let f64t = MirType::F64;
        if *ty == f64t {
            return self.emit_math_f64(x, code, &f64t);
        }
        let x64 = self.sb_cast(CastKind::FpExt, x, &f64t);
        let r = self.emit_math_f64(x64, code, &f64t);
        self.sb_cast(CastKind::FpTrunc, r, ty)
    }

    /// `logaddexp(a, b) = max(a, b) + ln(1 + e^−|a−b|)`, without overflow; `a == b` (both
    /// infinities included) is `a + ln 2`, and a NaN propagates. Built on the shared `exp`/`log1p`
    /// (the f32 polynomials or the f64 library); a half computes in `f32` and rounds once.
    fn sb_logaddexp(&mut self, a: ValueId, b: ValueId, ty: &MirType) -> ValueId {
        let half = matches!(ty, MirType::F16 | MirType::BF16);
        let ct = if half { MirType::F32 } else { ty.clone() };
        let (a, b) = if half {
            (
                self.sb_cast(CastKind::FpExt, a, &ct),
                self.sb_cast(CastKind::FpExt, b, &ct),
            )
        } else {
            (a, b)
        };
        let gt = self.sb_cmp(CmpOp::Fogt, a, b);
        let m = self.sb_select(gt, a, b, &ct);
        let d = self.builder.build(ct.clone(), Op::Bin(BinOp::FSub, a, b));
        let ad = self.emit_abs(d, &ct);
        let t = self.builder.build(ct.clone(), Op::Neg(ad));
        let ex = self.emit_exp(t, &ct);
        let l = self.emit_log1p(ex, &ct);
        let general = self.builder.build(ct.clone(), Op::Bin(BinOp::FAdd, m, l));
        let ln2 = self.splat_const_f(std::f64::consts::LN_2, &ct);
        let equal = self.builder.build(ct.clone(), Op::Bin(BinOp::FAdd, a, ln2));
        let eq = self.sb_cmp(CmpOp::Foeq, a, b);
        let r = self.sb_select(eq, equal, general, &ct);
        if half {
            self.sb_cast(CastKind::FpTrunc, r, ty)
        } else {
            r
        }
    }

    // --- FP8 ----------------------------------------------------------------------------------

    /// Decode the FP8 code in the `i8` value `byte` to `f32` (exact). E5M2 is the top byte of an
    /// `f16`; E4M3FN is decoded directly: bias 7, a subnormal `m·2⁻⁹`, its one NaN `S.1111.111`
    /// decoding to the quiet NaN of that sign.
    fn sb_fp8_decode(&mut self, byte: ValueId, fmt: Fp8) -> ValueId {
        let t = MirType::I32;
        let bz = self.sb_cast(CastKind::ZExt, byte, &t);
        let bits = match fmt {
            Fp8::E5M2 => {
                let h = self.sb_bin_k(BinOp::Shl, bz, 8, &t);
                self.sb_f16_bits_to_f32_bits(h)
            }
            Fp8::E4M3 => {
                let s = self.sb_bin_k(BinOp::And, bz, 0x80, &t);
                let sign = self.sb_bin_k(BinOp::Shl, s, 24, &t);
                let shifted = self.sb_bin_k(BinOp::LShr, bz, 3, &t);
                let e = self.sb_bin_k(BinOp::And, shifted, 0xf, &t);
                let m = self.sb_bin_k(BinOp::And, bz, 7, &t);
                let e32 = self.sb_bin_k(BinOp::Add, e, 120, &t);
                let e32s = self.sb_bin_k(BinOp::Shl, e32, 23, &t);
                let m20 = self.sb_bin_k(BinOp::Shl, m, 20, &t);
                let normal = self.sb_bin(BinOp::Or, e32s, m20, &t);
                let mf = self.sb_cast(CastKind::SiToFp, m, &MirType::F32);
                let scale = self.splat_const_f(1.0 / 512.0, &MirType::F32);
                let subf = self
                    .builder
                    .build(MirType::F32, Op::Bin(BinOp::FMul, mf, scale));
                let sub = self.sb_cast(CastKind::Bitcast, subf, &t);
                let is_sub = self.sb_cmp_k(CmpOp::Eq, e, 0, &t);
                let finite = self.sb_select(is_sub, sub, normal, &t);
                let low7 = self.sb_bin_k(BinOp::And, bz, 0x7f, &t);
                let is_nan = self.sb_cmp_k(CmpOp::Eq, low7, 0x7f, &t);
                let qnan = self.sb_int(0x7fc0_0000, &t);
                let mag = self.sb_select(is_nan, qnan, finite, &t);
                self.sb_bin(BinOp::Or, sign, mag, &t)
            }
        };
        self.sb_cast(CastKind::Bitcast, bits, &MirType::F32)
    }

    /// Encode the `f32` `x` as an FP8 code (an `i8`), rounding to nearest even. A normal result
    /// rounds on the bit pattern — add `half − 1` plus the kept lsb below the format's last
    /// mantissa bit, so a carry bumps the exponent — then rebiases; a result below the least normal
    /// rounds `|x|·2^k` (exact) to an integer. E4M3FN saturates at 448 (`0x7E`) and its NaN is
    /// `0x7F`; E5M2 overflows to infinity (`0x7C`) and its NaN is `0x7E`. The sign is copied.
    fn sb_fp8_encode(&mut self, x: ValueId, fmt: Fp8) -> ValueId {
        let t = MirType::I32;
        // (mantissa shift, rounding half − 1, exponent rebias in code units, largest code,
        //  least normal as f32 bits, subnormal scale 2^k, NaN code)
        let (shift, half_m1, rebias, top, min_normal, sub_scale, nan): (
            i128,
            i128,
            i128,
            i128,
            i128,
            f64,
            i128,
        ) = match fmt {
            Fp8::E4M3 => (20, 0x7_ffff, 960, 0x7e, 0x3c80_0000, 512.0, 0x7f),
            Fp8::E5M2 => (21, 0xf_ffff, 448, 0x7c, 0x3880_0000, 65536.0, 0x7e),
        };
        let b = self.sb_cast(CastKind::Bitcast, x, &t);
        let hi = self.sb_bin_k(BinOp::LShr, b, 24, &t);
        let sign = self.sb_bin_k(BinOp::And, hi, 0x80, &t);
        let a = self.sb_bin_k(BinOp::And, b, 0x7fff_ffff, &t);
        let is_nan = self.sb_cmp_k(CmpOp::Ugt, a, 0x7f80_0000, &t);
        // Normal range: round on the bits, rebias, and clamp to the largest code.
        let lsb_sh = self.sb_bin_k(BinOp::LShr, a, shift, &t);
        let lsb = self.sb_bin_k(BinOp::And, lsb_sh, 1, &t);
        let biased = self.sb_bin_k(BinOp::Add, a, half_m1, &t);
        let rounded = self.sb_bin(BinOp::Add, biased, lsb, &t);
        let code_sh = self.sb_bin_k(BinOp::LShr, rounded, shift, &t);
        let code_n = self.sb_bin_k(BinOp::Sub, code_sh, rebias, &t);
        let top_k = self.sb_int(top, &t);
        let over = self.sb_cmp(CmpOp::Sgt, code_n, top_k);
        let normal = self.sb_select(over, top_k, code_n, &t);
        // Below the least normal: |x|·2^k is exact; round it to the nearest (even) integer.
        let ax = self.sb_cast(CastKind::Bitcast, a, &MirType::F32);
        let scale = self.splat_const_f(sub_scale, &MirType::F32);
        let scaled = self
            .builder
            .build(MirType::F32, Op::Bin(BinOp::FMul, ax, scale));
        let r = self
            .builder
            .build(MirType::F32, Op::Round(RoundMode::Nearest, scaled));
        let sub = self.sb_cast(CastKind::FpToSi, r, &t);
        let is_sub = self.sb_cmp_k(CmpOp::Ult, a, min_normal, &t);
        let finite = self.sb_select(is_sub, sub, normal, &t);
        let nan_k = self.sb_int(nan, &t);
        let code = self.sb_select(is_nan, nan_k, finite, &t);
        let signed = self.sb_bin(BinOp::Or, code, sign, &t);
        self.sb_cast(CastKind::Trunc, signed, &MirType::I8)
    }
}
