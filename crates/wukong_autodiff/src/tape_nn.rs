//! Nonlinear and transformer VJPs for the buffer tape: all 36 elementwise activations, the two-input
//! `wukong_vmath2_f32` family (pow / atan2 / hypot and the SwiGLU / GeGLU / GLU gates), the
//! row-wise losses (softmax cross-entropy, log-sum-exp, entropy, KL divergence) and the transformer
//! pieces (rotary position embedding, embedding gather).
//!
//! Where the runtime ships a fused backward kernel the rule calls it (`wukong_vmath2_f32`'s
//! `*_BWD` codes, `wukong_xent_bwd_f32`, `wukong_rope_bwd_f32`, `wukong_scatter_add_f32`); where a
//! derivative is a different transcendental it is composed from the forward kernels
//! (`cos` for `sin`, `exp(-x²)` for `erf`, ...); only closed forms in `x` or `y` synthesize a loop.

use crate::tape::{
    pick, Loop, F32, I64T, INV6, LEAKY_ALPHA, NORM_SOFTMAX, QUICK_GELU_SLOPE, SELU_ALPHA,
    SELU_LAMBDA, VE_ID, VE_USE_Y, VM2_ATAN2, VM2_ELU_BWD, VM2_GELU_BWD, VM2_GELU_GATE, VM2_HYPOT,
    VM2_POW, VM2_SIGMOID_BWD, VM2_SIGMOID_GATE, VM2_SILU_BWD, VM2_SILU_GATE, VM2_SOFTPLUS_BWD,
    VM2_TANH_BWD, VM_ACOS, VM_ACOSH, VM_ASIN, VM_ASINH, VM_ATAN, VM_ATANH, VM_CBRT, VM_COS,
    VM_COSH, VM_ELU, VM_ELU_EXPM1, VM_ERF, VM_EXP, VM_EXP10, VM_EXP2, VM_EXPM1, VM_GELU,
    VM_GELU_ERF_CDF, VM_GELU_ERF_DIV, VM_GELU_ERF_MUL, VM_GELU_TANH, VM_GELU_TANH_CDF,
    VM_GELU_TANH_FAST, VM_GELU_TANH_HF, VM_GELU_TANH_NOFMA, VM_HARDSIGMOID, VM_HARDSIGMOID_CLAMP,
    VM_HARDSIGMOID_MINMAX, VM_HARDSWISH, VM_HARDSWISH_CLAMP, VM_HARDSWISH_MINMAX, VM_LEAKYRELU,
    VM_LEAKY_GE, VM_LEAKY_LT, VM_LEAKY_MAX, VM_LOG, VM_LOG10, VM_LOG1P, VM_LOG2, VM_LOGSIGMOID,
    VM_MISH, VM_MISH_LOG1P, VM_MISH_NAIVE, VM_MISH_SOFTPLUS, VM_QUICK_GELU, VM_RELU, VM_SELU,
    VM_SIGMOID, VM_SIGMOID_RATIO, VM_SILU, VM_SILU_DIV, VM_SILU_SX, VM_SIN, VM_SINH, VM_SOFTPLUS,
    VM_SOFTPLUS_LOG1P, VM_SOFTPLUS_NAIVE, VM_SOFTSIGN, VM_TAN, VM_TANH, VM_TANHSHRINK,
};
use crate::Vjp;
use wukong_mir::{BinOp, CastKind, CmpOp, MirType, Op, ValueId};

/// The fused `wukong_vmath2_f32` backward code (`dx = dy·act'(x)` in one 256-bit pass) for the
/// activations the runtime ships one for. Each kernel reuses the forward's own polynomial
/// (`sigmoid1`, `tanh_exp1`, `exp1`), so the derivative is of the function the forward evaluated.
fn vm2_bwd_code(op: i64) -> Option<i64> {
    Some(match op {
        VM_SILU => VM2_SILU_BWD,
        VM_GELU => VM2_GELU_BWD,
        VM_ELU => VM2_ELU_BWD,
        VM_SOFTPLUS => VM2_SOFTPLUS_BWD,
        VM_SIGMOID => VM2_SIGMOID_BWD,
        VM_TANH => VM2_TANH_BWD,
        _ => return None,
    })
}

/// The activation a smooth natural spelling (`wukong_runtime::vspell`, P12.10) IS, written another
/// way: its derivative is that activation's, so it takes that activation's backward rule. The
/// forward kernels differ in the last bits (that is why they are separate codes); the derivative is
/// of the function, and every rule this routes to reads only `x` and `dout`, never the forward
/// output (whose bits are the spelling's). QuickGELU and the erf GELU are functions no other code
/// computes, so [`Vjp::act_backward`] composes theirs.
///
/// The ELU spelling's derivative has no kink (`e^0 = 1` from both sides), so it shares `VM_ELU`'s.
/// The select-spelled spellings WITH a kink -- the leaky selects and the `clamp` / `min`-`max` hard
/// activations -- are deliberately absent: at the kink the derivative is a choice, and theirs is
/// the one their own selects make (the language's `Select` VJP hands the adjoint to the operand
/// selected), which the intrinsics' rules do not all match. `x < 0 ? 0.01x : x` takes the identity
/// branch at `x = 0` where `VM_LEAKYRELU`'s `x > 0` takes the slope (and `max(x, 0.01x)` the slope at
/// `x = +inf`, where `x > 0` takes the identity); `clamp(v, 0, 6)` passes `v` on
/// the closed `[0, 6]` where `VM_HARDSIGMOID` uses the open `(-3, 3)` in `x`; and even
/// `min(max(v, 0), 6)` sees `v = x + 3` round to 6 at `x = 3 - 2^-22`, inside that open interval.
/// Their rules are in [`Vjp::algebraic_derivative`], each its spelling's selects over the rounded
/// `v`.
fn spelling_base(op: i64) -> Option<i64> {
    Some(match op {
        VM_GELU_TANH | VM_GELU_TANH_HF | VM_GELU_TANH_NOFMA | VM_GELU_TANH_FAST
        | VM_GELU_TANH_CDF => VM_GELU,
        VM_SILU_DIV | VM_SILU_SX => VM_SILU,
        VM_SIGMOID_RATIO => VM_SIGMOID,
        VM_SOFTPLUS_NAIVE | VM_SOFTPLUS_LOG1P => VM_SOFTPLUS,
        VM_MISH_NAIVE | VM_MISH_SOFTPLUS | VM_MISH_LOG1P => VM_MISH,
        VM_ELU_EXPM1 => VM_ELU,
        _ => return None,
    })
}

/// The activations whose derivative is a closed form in the input `x` or the output `y` — emitted
/// as one synthesized loop by [`Vjp::act_backward`].
fn is_algebraic(op: i64) -> bool {
    matches!(
        op,
        VM_RELU
            | VM_EXP
            | VM_LOG
            | VM_LEAKYRELU
            | VM_SELU
            | VM_TANHSHRINK
            | VM_SOFTSIGN
            | VM_HARDSIGMOID
            | VM_HARDSWISH
            | VM_EXP2
            | VM_EXP10
            | VM_LOG2
            | VM_LOG10
            | VM_EXPM1
            | VM_LOG1P
            | VM_ATAN
            | VM_ATANH
            | VM_TAN
            | VM_ASINH
            | VM_ACOSH
            | VM_ASIN
            | VM_ACOS
            | VM_SINH
            | VM_CBRT
            | VM_HARDSIGMOID_MINMAX
            | VM_HARDSIGMOID_CLAMP
            | VM_HARDSWISH_MINMAX
            | VM_HARDSWISH_CLAMP
            | VM_LEAKY_LT
            | VM_LEAKY_GE
            | VM_LEAKY_MAX
    )
}

impl<'a> Vjp<'a> {
    /// `dst = dout (.) act'(x)` over `n` elements, for the forward activation `y = act(x)` (`x` the
    /// pre-activation buffer, `y` the output buffer, both in the new value space). `dst` must not
    /// alias `dout`, `x` or `y`; it is fully overwritten. All 36 `wukong_runtime::vmath` codes, and
    /// the natural-spelling codes after them (`wukong_runtime::vspell`; see [`spelling_base`]):
    ///
    /// - **fused backward kernel** (silu, gelu, elu, softplus, sigmoid, tanh) — one
    ///   `wukong_vmath2_f32(x, dout, dst, n, *_BWD)`;
    /// - **kernel composition** (the derivative is a different transcendental): `sin' = cos`,
    ///   `cos' = -sin`, `cosh' = sinh`, `erf' = 2/√π·e^{-x²}`, `logsigmoid' = σ(-x)`, and mish's
    ///   `tanh(sp) + x·σ·(1 - tanh²(sp))` — each built from forward `vmath`/`velem` calls in `dst`;
    /// - **algebraic** (a closed form in `x` or `y`) — one synthesized loop.
    pub(crate) fn act_backward(
        &mut self,
        op: i64,
        dout: ValueId,
        x: ValueId,
        y: ValueId,
        dst: ValueId,
        n: ValueId,
    ) -> Result<(), String> {
        if let Some(base) = spelling_base(op) {
            return self.act_backward(base, dout, x, y, dst, n);
        }
        if let Some(bwd) = vm2_bwd_code(op) {
            self.vmath2_call(x, dout, dst, n, bwd);
            return Ok(());
        }
        match op {
            VM_QUICK_GELU => {
                // x*s(kx), k = 1.702: s(kx) + k*x*s(kx)*(1 - s(kx)).
                self.velem_axpc(dst, x, QUICK_GELU_SLOPE, 0.0, n);
                self.vmath_call(dst, dst, n, VM_SIGMOID);
                let lp = self.open_loop(n);
                let i = lp.idx;
                let s = self.load_at(dst, i);
                let xi = self.load_at(x, i);
                let d = self.load_at(dout, i);
                let one = self.cf32(1.0);
                let k = self.cf32(QUICK_GELU_SLOPE);
                let oms = self.b.build(F32, Op::Bin(BinOp::FSub, one, s));
                let kx = self.fmul(k, xi, &F32);
                let kxs = self.fmul(kx, s, &F32);
                let term = self.fmul(kxs, oms, &F32);
                let fp = self.fadd(s, term, &F32);
                let v = self.fmul(d, fp, &F32);
                self.store_at(dst, i, v);
                self.close_loop(lp);
                return Ok(());
            }
            VM_GELU_ERF_DIV | VM_GELU_ERF_MUL | VM_GELU_ERF_CDF => {
                // x*Phi(x): Phi(x) + x*phi(x) = 0.5*(1 + erf(x/sqrt 2)) + x*exp(-x^2/2)/sqrt(2 pi).
                self.velem_axpc(dst, x, std::f64::consts::FRAC_1_SQRT_2, 0.0, n);
                self.vmath_call(dst, dst, n, VM_ERF);
                let g = self.scratch(n);
                self.velem_hadamard(g, x, x, n);
                self.velem_axpc(g, g, -0.5, 0.0, n);
                self.vmath_call(g, g, n, VM_EXP);
                let lp = self.open_loop(n);
                let i = lp.idx;
                let e = self.load_at(dst, i);
                let gi = self.load_at(g, i);
                let xi = self.load_at(x, i);
                let d = self.load_at(dout, i);
                let half = self.cf32(0.5);
                // 1/sqrt(2 pi) = (2/sqrt(pi)) * (1/sqrt(2)) / 2.
                let inv_sqrt_2pi = self
                    .cf32(0.5 * std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2);
                let he = self.fmul(half, e, &F32);
                let cdf = self.fadd(half, he, &F32);
                let xg = self.fmul(xi, gi, &F32);
                let pdf = self.fmul(xg, inv_sqrt_2pi, &F32);
                let fp = self.fadd(cdf, pdf, &F32);
                let v = self.fmul(d, fp, &F32);
                self.store_at(dst, i, v);
                self.close_loop(lp);
                return Ok(());
            }
            VM_SIN => {
                self.vmath_call(x, dst, n, VM_COS);
                self.velem_hadamard(dst, dst, dout, n);
                return Ok(());
            }
            VM_COS => {
                self.vmath_call(x, dst, n, VM_SIN);
                self.velem_hadamard(dst, dst, dout, n);
                self.velem_axpc(dst, dst, -1.0, 0.0, n);
                return Ok(());
            }
            VM_COSH => {
                self.vmath_call(x, dst, n, VM_SINH);
                self.velem_hadamard(dst, dst, dout, n);
                return Ok(());
            }
            VM_ERF => {
                // erf'(x) = 2/sqrt(pi) · exp(-x^2)
                self.velem_hadamard(dst, x, x, n);
                self.velem_axpc(dst, dst, -1.0, 0.0, n);
                self.vmath_call(dst, dst, n, VM_EXP);
                self.velem_hadamard(dst, dst, dout, n);
                self.velem_axpc(dst, dst, std::f64::consts::FRAC_2_SQRT_PI, 0.0, n);
                return Ok(());
            }
            VM_LOGSIGMOID => {
                // logsigmoid(x) = -softplus(-x), so logsigmoid'(x) = sigmoid(-x) — computed as a
                // sigmoid (no `1 - exp(y)` cancellation for large x).
                self.velem_axpc(dst, x, -1.0, 0.0, n);
                self.vmath_call(dst, dst, n, VM_SIGMOID);
                self.velem_hadamard(dst, dst, dout, n);
                return Ok(());
            }
            VM_MISH => {
                // mish(x) = x·tanh(sp(x)), sp = softplus:
                // mish'(x) = tanh(sp) + x·sigmoid(x)·(1 - tanh²(sp)).
                self.vmath_call(x, dst, n, VM_SOFTPLUS);
                self.vmath_call(dst, dst, n, VM_TANH);
                let s = self.scratch(n);
                self.vmath_call(x, s, n, VM_SIGMOID);
                let lp = self.open_loop(n);
                let i = lp.idx;
                let t = self.load_at(dst, i);
                let sg = self.load_at(s, i);
                let xi = self.load_at(x, i);
                let d = self.load_at(dout, i);
                let one = self.cf32(1.0);
                let t2 = self.fmul(t, t, &F32);
                let omt = self.b.build(F32, Op::Bin(BinOp::FSub, one, t2));
                let xs = self.fmul(xi, sg, &F32);
                let term = self.fmul(xs, omt, &F32);
                let fp = self.fadd(t, term, &F32);
                let v = self.fmul(d, fp, &F32);
                self.store_at(dst, i, v);
                self.close_loop(lp);
                return Ok(());
            }
            _ => {}
        }
        if !is_algebraic(op) {
            return Err(format!(
                "autodiff: no VJP for vmath op {op} (not a wukong_runtime::vmath activation code)"
            ));
        }
        let lp = self.open_loop(n);
        let i = lp.idx;
        let doi = self.load_at(dout, i);
        let fp = self.algebraic_derivative(op, x, y, i);
        // dst[i] = dout[i] * f'(x[i])
        let v = self.b.build(F32, Op::Bin(BinOp::FMul, doi, fp));
        self.store_at(dst, i, v);
        self.close_loop(lp);
        Ok(())
    }

    /// `act'(x[i])` for an [`is_algebraic`] op, emitted inside the backward loop at index `i`.
    fn algebraic_derivative(&mut self, op: i64, x: ValueId, y: ValueId, i: ValueId) -> ValueId {
        match op {
            VM_RELU => {
                // (x[i] > 0) ? 1 : 0
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                self.b.build(F32, Op::Select(pos, one, zero))
            }
            VM_EXP => self.load_at(y, i), // exp' = y
            VM_LOG => {
                // log' = 1/x
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                self.fdiv(one, xi, &F32)
            }
            VM_LEAKYRELU => {
                // leakyrelu' = x > 0 ? 1 : alpha   (the kernel's `x > 0.0` test, mirrored)
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let al = self.cf32(LEAKY_ALPHA);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                self.b.build(F32, Op::Select(pos, one, al))
            }
            VM_SELU => {
                // selu(x) = lambda·x for x > 0, else lambda·alpha·(e^x - 1). For x <= 0 the
                // derivative lambda·alpha·e^x is exactly `y + lambda·alpha`, so no exp is needed.
                let xi = self.load_at(x, i);
                let yi = self.load_at(y, i);
                let zero = self.cf32(0.0);
                let lam = self.cf32(SELU_LAMBDA);
                let la = self.cf32(SELU_LAMBDA * SELU_ALPHA);
                let negb = self.fadd(yi, la, &F32);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                self.b.build(F32, Op::Select(pos, lam, negb))
            }
            VM_TANHSHRINK => {
                // y = x - tanh(x), so tanh(x) = x - y and tanhshrink' = 1 - (1-tanh^2) = tanh^2.
                let xi = self.load_at(x, i);
                let yi = self.load_at(y, i);
                let t = self.b.build(F32, Op::Bin(BinOp::FSub, xi, yi));
                self.fmul(t, t, &F32)
            }
            VM_SOFTSIGN => {
                // softsign' = 1/(1 + |x|)^2
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let nx = self.neg(xi, &F32);
                let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, zero));
                let ax = self.b.build(F32, Op::Select(pos, xi, nx));
                let d = self.fadd(one, ax, &F32);
                let d2 = self.fmul(d, d, &F32);
                self.fdiv(one, d2, &F32)
            }
            VM_HARDSIGMOID => {
                // clamp(x+3, 0, 6)/6  ->  1/6 on (-3, 3), 0 outside.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let inv6 = self.cf32(INV6);
                let three = self.cf32(3.0);
                let nthree = self.cf32(-3.0);
                let below = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, xi, three));
                let inb = self.b.build(F32, Op::Select(below, inv6, zero));
                let above = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, nthree));
                self.b.build(F32, Op::Select(above, inb, zero))
            }
            // The kinked spellings: each derivative is the spelling's own selects, so the adjoint
            // goes where the select sent the value (see [`spelling_base`]).
            VM_LEAKY_LT => {
                // `x < 0 ? 0.01x : x`, i.e. `0 > x ? .. : ..`: 0.01 strictly below zero, 1 at +-0.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let al = self.cf32(LEAKY_ALPHA);
                let neg = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, zero, xi));
                self.b.build(F32, Op::Select(neg, al, one))
            }
            VM_LEAKY_GE => {
                // `x >= 0 ? x : 0.01x`: 1 at +-0 and above, 0.01 below.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let al = self.cf32(LEAKY_ALPHA);
                let nonneg = self.b.build(MirType::I1, Op::Cmp(CmpOp::Foge, xi, zero));
                self.b.build(F32, Op::Select(nonneg, one, al))
            }
            VM_LEAKY_MAX => {
                // `max(x, 0.01x)` is `x > 0.01x ? x : 0.01x`, over the forward's rounded product:
                // 1 where the identity wins, 0.01 at +-0, +inf and below zero.
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let al = self.cf32(LEAKY_ALPHA);
                let scaled = self.fmul(al, xi, &F32);
                let wins = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, scaled));
                self.b.build(F32, Op::Select(wins, one, al))
            }
            VM_HARDSIGMOID_MINMAX
            | VM_HARDSIGMOID_CLAMP
            | VM_HARDSWISH_MINMAX
            | VM_HARDSWISH_CLAMP => {
                // `relu6(v)` with `v = x + 3` (the forward's rounded add). `h` is the value the
                // spelling's selects produce, and the adjoint of `v` is passed only where both of
                // them handed `v` itself through:
                //   min(max(v, 0), 6): m = v > 0 ? v : 0, h = 6 > m ? m : 6 -- passed iff
                //                      v > 0 and 6 > m;
                //   clamp(v, 0, 6):    h = 0 > v ? 0 : (v > 6 ? 6 : v) -- passed iff neither
                //                      0 > v nor v > 6.
                // hardsigmoid = h/6, so its derivative is passed/6; hardswish = (x*h)/6, so its
                // derivative is (h + passed*x)/6, the product rule through the same selects.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let three = self.cf32(3.0);
                let six = self.cf32(6.0);
                let v = self.fadd(three, xi, &F32);
                let minmax = matches!(op, VM_HARDSIGMOID_MINMAX | VM_HARDSWISH_MINMAX);
                let hardswish = matches!(op, VM_HARDSWISH_MINMAX | VM_HARDSWISH_CLAMP);
                // What a passed adjoint carries: `x` for hardswish's product, else 1.
                let carried = if hardswish { xi } else { self.cf32(1.0) };
                let (h, passed) = if minmax {
                    let pos = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, v, zero));
                    let m = self.b.build(F32, Op::Select(pos, v, zero));
                    let under = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, six, m));
                    let h = self.b.build(F32, Op::Select(under, m, six));
                    let inner = self.b.build(F32, Op::Select(under, carried, zero));
                    (h, self.b.build(F32, Op::Select(pos, inner, zero)))
                } else {
                    // The select arms are swapped rather than the compares negated: a negated
                    // ordered compare answers a NaN `v` the other way.
                    let below = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, zero, v));
                    let above = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, v, six));
                    let capped = self.b.build(F32, Op::Select(above, six, v));
                    let h = self.b.build(F32, Op::Select(below, zero, capped));
                    let inner = self.b.build(F32, Op::Select(above, zero, carried));
                    (h, self.b.build(F32, Op::Select(below, zero, inner)))
                };
                let num = if hardswish {
                    self.fadd(h, passed, &F32)
                } else {
                    passed
                };
                let inv6 = self.cf32(INV6);
                self.fmul(num, inv6, &F32)
            }
            VM_HARDSWISH => {
                // x·hardsigmoid(x)  ->  0 below -3, 1 above 3, (2x + 3)/6 in between.
                let xi = self.load_at(x, i);
                let zero = self.cf32(0.0);
                let one = self.cf32(1.0);
                let two = self.cf32(2.0);
                let three = self.cf32(3.0);
                let nthree = self.cf32(-3.0);
                let inv6 = self.cf32(INV6);
                let t2x = self.fmul(two, xi, &F32);
                let t = self.fadd(t2x, three, &F32);
                let ramp = self.fmul(t, inv6, &F32);
                let below = self.b.build(MirType::I1, Op::Cmp(CmpOp::Folt, xi, three));
                let mid = self.b.build(F32, Op::Select(below, ramp, one));
                let above = self.b.build(MirType::I1, Op::Cmp(CmpOp::Fogt, xi, nthree));
                self.b.build(F32, Op::Select(above, mid, zero))
            }
            VM_EXP2 => {
                // (2^x)' = 2^x·ln2 = y·ln2
                let yi = self.load_at(y, i);
                let c = self.cf32(std::f64::consts::LN_2);
                self.fmul(yi, c, &F32)
            }
            VM_EXP10 => {
                let yi = self.load_at(y, i);
                let c = self.cf32(std::f64::consts::LN_10);
                self.fmul(yi, c, &F32)
            }
            VM_LOG2 => {
                // (log2 x)' = 1/(x·ln2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let c = self.cf32(std::f64::consts::LN_2);
                let d = self.fmul(xi, c, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_LOG10 => {
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let c = self.cf32(std::f64::consts::LN_10);
                let d = self.fmul(xi, c, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_EXPM1 => {
                // (e^x - 1)' = e^x = y + 1
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                self.fadd(yi, one, &F32)
            }
            VM_LOG1P => {
                // (ln(1+x))' = 1/(1+x)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let d = self.fadd(one, xi, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_ATAN => {
                // atan' = 1/(1 + x^2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.fadd(one, x2, &F32);
                self.fdiv(one, d, &F32)
            }
            VM_ATANH => {
                // atanh' = 1/(1 - x^2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.b.build(F32, Op::Bin(BinOp::FSub, one, x2));
                self.fdiv(one, d, &F32)
            }
            VM_TAN => {
                // tan' = 1 + tan^2 = 1 + y^2
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let y2 = self.fmul(yi, yi, &F32);
                self.fadd(one, y2, &F32)
            }
            VM_ASINH => {
                // asinh' = 1/sqrt(1 + x^2)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.fadd(one, x2, &F32);
                let s = self.b.build(F32, Op::Sqrt(d));
                self.fdiv(one, s, &F32)
            }
            VM_ACOSH => {
                // acosh' = 1/sqrt(x^2 - 1)   (domain x > 1)
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.b.build(F32, Op::Bin(BinOp::FSub, x2, one));
                let s = self.b.build(F32, Op::Sqrt(d));
                self.fdiv(one, s, &F32)
            }
            VM_ASIN | VM_ACOS => {
                // asin' = 1/sqrt(1 - x^2); acos' is its negation.
                let xi = self.load_at(x, i);
                let one = self.cf32(1.0);
                let x2 = self.fmul(xi, xi, &F32);
                let d = self.b.build(F32, Op::Bin(BinOp::FSub, one, x2));
                let s = self.b.build(F32, Op::Sqrt(d));
                let r = self.fdiv(one, s, &F32);
                if op == VM_ACOS {
                    self.neg(r, &F32)
                } else {
                    r
                }
            }
            VM_SINH => {
                // sinh' = cosh = sqrt(1 + sinh^2) = sqrt(1 + y^2)  (cosh is always positive)
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let y2 = self.fmul(yi, yi, &F32);
                let d = self.fadd(one, y2, &F32);
                self.b.build(F32, Op::Sqrt(d))
            }
            // VM_CBRT — the last algebraic code ([`is_algebraic`] gates the call):
            // (x^(1/3))' = 1/(3·x^(2/3)) = 1/(3 y^2)
            _ => {
                let yi = self.load_at(y, i);
                let one = self.cf32(1.0);
                let three = self.cf32(3.0);
                let y2 = self.fmul(yi, yi, &F32);
                let d = self.fmul(three, y2, &F32);
                self.fdiv(one, d, &F32)
            }
        }
    }

    /// Open a loop over `n` elements and load `bufs[k][i]` for every buffer: the common head of the
    /// two-input derivative loops. The caller stores into its destination and closes the loop.
    fn open_elementwise(&mut self, n: ValueId, bufs: &[ValueId]) -> (Loop, Vec<ValueId>) {
        let lp = self.open_loop(n);
        let i = lp.idx;
        let vals = bufs.iter().map(|&b| self.load_at(b, i)).collect();
        (lp, vals)
    }

    // --- two-input kernel family (vmath2) ------------------------------------------------------

    /// `vmath2(x, y, out, n, op)`: `out = f(x, y)` elementwise.
    /// - `pow`:   `dx = dout·y·out/x`, `dy = dout·out·ln x` (the `x > 0` domain the kernel defines);
    /// - `atan2(x, y)` (`x` the ordinate): `dx = dout·y/(x²+y²)`, `dy = -dout·x/(x²+y²)`;
    /// - `hypot`: `dx = dout·x/out`, `dy = dout·y/out`;
    /// - the gates `out = act(a)·b` (SwiGLU / GeGLU / GLU): `da = act'(a)·(dout·b)` through the
    ///   fused `*_BWD` kernel, `db = dout·act(a)`.
    ///
    /// The `*_BWD` codes are themselves derivatives; differentiating one is second order and refuses.
    pub(crate) fn diff_vmath2(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, y, out) = (args[0], args[1], args[2]);
        let dout = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        let op = self.const_i64(args[4])?;
        if out == x || out == y {
            return Err(format!(
                "autodiff: vmath2 VJP cannot differentiate an in-place two-input op {op} \
                 (output buffer v{} aliases an input)",
                out.0
            ));
        }
        let n = self.dim(args[3]);
        let (xn, yn, outn) = (self.remap_v(x), self.remap_v(y), self.remap_v(out));
        match op {
            VM2_POW => {
                if let Some(c) = self.begin(x, n)? {
                    let (lp, v) = self.open_elementwise(n, &[dout, yn, outn, xn]);
                    let dy_ = self.fmul(v[0], v[1], &F32);
                    let t = self.fmul(dy_, v[2], &F32);
                    let r = self.fdiv(t, v[3], &F32);
                    self.store_at(c.write, lp.idx, r);
                    self.close_loop(lp);
                    self.end(c);
                }
                if let Some(c) = self.begin(y, n)? {
                    self.vmath_call(xn, c.write, n, VM_LOG);
                    let (lp, v) = self.open_elementwise(n, &[dout, outn, c.write]);
                    let t = self.fmul(v[0], v[1], &F32);
                    let r = self.fmul(t, v[2], &F32);
                    self.store_at(c.write, lp.idx, r);
                    self.close_loop(lp);
                    self.end(c);
                }
            }
            VM2_ATAN2 => {
                for (target, num_is_y, negate) in [(x, true, false), (y, false, true)] {
                    if let Some(c) = self.begin(target, n)? {
                        let (lp, v) = self.open_elementwise(n, &[dout, xn, yn]);
                        let x2 = self.fmul(v[1], v[1], &F32);
                        let y2 = self.fmul(v[2], v[2], &F32);
                        let den = self.fadd(x2, y2, &F32);
                        let num = if num_is_y { v[2] } else { v[1] };
                        let t = self.fmul(v[0], num, &F32);
                        let mut r = self.fdiv(t, den, &F32);
                        if negate {
                            r = self.neg(r, &F32);
                        }
                        self.store_at(c.write, lp.idx, r);
                        self.close_loop(lp);
                        self.end(c);
                    }
                }
            }
            VM2_HYPOT => {
                for (target, src) in [(x, xn), (y, yn)] {
                    if let Some(c) = self.begin(target, n)? {
                        let (lp, v) = self.open_elementwise(n, &[dout, src, outn]);
                        let t = self.fmul(v[0], v[1], &F32);
                        let r = self.fdiv(t, v[2], &F32);
                        self.store_at(c.write, lp.idx, r);
                        self.close_loop(lp);
                        self.end(c);
                    }
                }
            }
            VM2_SILU_GATE | VM2_GELU_GATE | VM2_SIGMOID_GATE => {
                let (act, bwd) = match op {
                    VM2_SILU_GATE => (VM_SILU, VM2_SILU_BWD),
                    VM2_GELU_GATE => (VM_GELU, VM2_GELU_BWD),
                    _ => (VM_SIGMOID, VM2_SIGMOID_BWD),
                };
                if let Some(c) = self.begin(x, n)? {
                    let t = self.scratch(n);
                    self.velem_hadamard(t, dout, yn, n);
                    self.vmath2_call(xn, t, c.write, n, bwd);
                    self.end(c);
                }
                if let Some(c) = self.begin(y, n)? {
                    self.vmath_call(xn, c.write, n, act);
                    self.velem_hadamard(c.write, c.write, dout, n);
                    self.end(c);
                }
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP for vmath2 op {other} (the activation-backward codes are \
                     themselves derivatives; differentiating one is second order)"
                ))
            }
        }
        Ok(())
    }

    // --- row-wise losses -----------------------------------------------------------------------

    /// `xent_fwd(x, target, loss, rows, cols)`: `loss[r] = lse(x[r,:]) - x[r, target[r]]` — the
    /// softmax cross-entropy of every classifier and language-model head. `dx[r,:] =
    /// dloss[r]·(softmax(x[r,:]) - onehot(target[r]))`: the runtime's fused `wukong_xent_bwd_f32`
    /// (unscaled) plus a per-row scale. The integer labels carry no gradient.
    pub(crate) fn diff_xent(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (x, target, loss) = (args[0], args[1], args[2]);
        let dl = match self.buf_adj.get(&loss).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if loss == x {
            return Err(format!(
                "autodiff: cross-entropy loss vector v{} aliases its logits",
                loss.0
            ));
        }
        let rows = self.dim(args[3]);
        let cols = self.dim(args[4]);
        let total = self.mul64(rows, cols);
        let (xn, tn) = (self.remap_v(x), self.remap_v(target));
        if let Some(c) = self.begin(x, total)? {
            let f = pick(self.syms.xent_bwd, self.syms.xent_bwd_par, par);
            self.emit(f, vec![xn, tn, c.write, rows, cols]);
            self.row_scale(c.write, dl, rows, cols);
            self.end(c);
        }
        Ok(())
    }

    /// `logsumexp(x, out, rows, cols)`: `out[r] = log Σ_j exp(x[r,j])`, so
    /// `dx[r,:] = dout[r]·softmax(x[r,:])` — the forward softmax kernel plus a per-row scale.
    pub(crate) fn diff_logsumexp(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (x, out) = (args[0], args[1]);
        let d = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if x == out {
            return Err(format!(
                "autodiff: log-sum-exp output v{} aliases its input",
                out.0
            ));
        }
        let rows = self.dim(args[2]);
        let cols = self.dim(args[3]);
        let total = self.mul64(rows, cols);
        let xn = self.remap_v(x);
        if let Some(c) = self.begin(x, total)? {
            let eps = self.cint(0);
            let sm = self.cint(NORM_SOFTMAX);
            let f = self.syms.norm;
            self.emit(f, vec![xn, c.write, rows, cols, eps, sm]);
            self.row_scale(c.write, d, rows, cols);
            self.end(c);
        }
        Ok(())
    }

    /// `entropy(p, out, rows, cols)`: `out[r] = -Σ_j p·log p`, so `dp = -dout[r]·(log p + 1)`.
    pub(crate) fn diff_entropy(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (p, out) = (args[0], args[1]);
        let d = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if p == out {
            return Err(format!(
                "autodiff: entropy output v{} aliases its input",
                out.0
            ));
        }
        let rows = self.dim(args[2]);
        let cols = self.dim(args[3]);
        let total = self.mul64(rows, cols);
        let pn = self.remap_v(p);
        if let Some(c) = self.begin(p, total)? {
            self.vmath_call(pn, c.write, total, VM_LOG);
            self.velem_axpc(c.write, c.write, -1.0, -1.0, total); // -(log p) - 1
            self.row_scale(c.write, d, rows, cols);
            self.end(c);
        }
        Ok(())
    }

    /// `kldiv(p, q, out, rows, cols)`: `out[r] = Σ_j p·(log p - log q)` — the distillation / VAE
    /// loss. `dp = dout[r]·(log p - log q + 1)`, `dq = -dout[r]·p/q`.
    pub(crate) fn diff_kldiv(&mut self, args: &[ValueId]) -> Result<(), String> {
        let (p, q, out) = (args[0], args[1], args[2]);
        let d = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if out == p || out == q {
            return Err(format!(
                "autodiff: KL-divergence output v{} aliases an input",
                out.0
            ));
        }
        let rows = self.dim(args[3]);
        let cols = self.dim(args[4]);
        let total = self.mul64(rows, cols);
        let (pn, qn) = (self.remap_v(p), self.remap_v(q));
        if let Some(c) = self.begin(p, total)? {
            self.vmath_call(pn, c.write, total, VM_LOG);
            let lq = self.scratch(total);
            self.vmath_call(qn, lq, total, VM_LOG);
            // write = 1·log p + (-1·log q + 1)
            let one = self.cf32(1.0);
            let m1 = self.cf32(-1.0);
            let opv = self.cint(VE_ID | VE_USE_Y);
            let f = self.syms.velem;
            self.emit(f, vec![c.write, lq, c.write, total, one, m1, one, opv]);
            self.row_scale(c.write, d, rows, cols);
            self.end(c);
        }
        if let Some(c) = self.begin(q, total)? {
            self.velem_div(c.write, pn, qn, total);
            self.velem_axpc(c.write, c.write, -1.0, 0.0, total);
            self.row_scale(c.write, d, rows, cols);
            self.end(c);
        }
        Ok(())
    }

    // --- transformer pieces --------------------------------------------------------------------

    /// `rope(x, inv_freq, out, rows, half)`: each pair `(a, b) = (x[j], x[j+half])` of row `r` is
    /// rotated by `θ = r·inv_freq[j]`. A rotation's Jacobian is the rotation, so
    /// `dx = R(θ)ᵀ·dout` — the runtime's `wukong_rope_bwd_f32`, in place when the forward was.
    ///
    /// The frequency table is usually fixed, but it is a buffer like any other and a gradient
    /// requested for it is computed rather than silently zeroed: `∂out_a/∂θ = -out_b`,
    /// `∂out_b/∂θ = out_a`, so `dinv_freq[j] = Σ_r r·(dout_b·out_a - dout_a·out_b)` (one row×pair
    /// loop over the forward output). It is taken first, while `dout` is still the output adjoint.
    pub(crate) fn diff_rope(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (x, freq, out) = (args[0], args[1], args[2]);
        let d = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if out == freq {
            return Err(format!(
                "autodiff: RoPE output v{} aliases its frequency table",
                out.0
            ));
        }
        let rows = self.dim(args[3]);
        let half = self.dim(args[4]);
        let two = self.cint(2);
        let rh = self.mul64(rows, half);
        let total = self.mul64(rh, two);
        let (freqn, outn) = (self.remap_v(freq), self.remap_v(out));
        if let Some(c) = self.begin(freq, half)? {
            self.rope_freq_grad(d, outn, c.write, rows, half);
            self.end(c);
        }
        let f = pick(self.syms.rope_bwd, self.syms.rope_bwd_par, par);
        if out == x {
            self.emit(f, vec![d, freqn, d, rows, half]);
        } else if let Some(c) = self.begin(x, total)? {
            self.emit(f, vec![d, freqn, c.write, rows, half]);
            self.end(c);
        }
        Ok(())
    }

    /// `dst[j] += Σ_r r·(dout[r, j+half]·out[r, j] - dout[r, j]·out[r, j+half])` — the RoPE frequency
    /// gradient. `dst` must start zeroed (a first contribution's gradient buffer, or a fresh scratch).
    fn rope_freq_grad(
        &mut self,
        dout: ValueId,
        out: ValueId,
        dst: ValueId,
        rows: ValueId,
        half: ValueId,
    ) {
        let two = self.cint(2);
        let dim = self.mul64(half, two);
        let outer = self.open_loop(rows);
        let r = outer.idx;
        let rf = self.b.build(F32, Op::Cast(CastKind::SiToFp, r, F32));
        let base = self.mul64(r, dim);
        let inner = self.open_loop(half);
        let j = inner.idx;
        let ia = self.b.build(I64T, Op::Bin(BinOp::Add, base, j));
        let ib = self.b.build(I64T, Op::Bin(BinOp::Add, ia, half));
        let oa = self.load_at(out, ia);
        let ob = self.load_at(out, ib);
        let da = self.load_at(dout, ia);
        let db = self.load_at(dout, ib);
        let p = self.fmul(db, oa, &F32);
        let q = self.fmul(da, ob, &F32);
        let t = self.b.build(F32, Op::Bin(BinOp::FSub, p, q));
        let term = self.fmul(rf, t, &F32);
        let old = self.load_at(dst, j);
        let acc = self.fadd(old, term, &F32);
        self.store_at(dst, j, acc);
        self.close_loop(inner);
        self.close_loop(outer);
    }

    /// `embedding(out, weight, ids, t, h, v)`: `out[t, :] = weight[ids[t], :]` — the first layer of
    /// every language model. Its VJP is the scatter-add `dweight[ids[t], :] += dout[t, :]` the
    /// runtime ships as `wukong_scatter_add_f32` (colliding ids sum in a fixed ascending order, so
    /// serial == parallel bit for bit). The kernel ACCUMULATES, which is exactly right for both a
    /// first contribution (an untouched gradient buffer is zero: caller-zeroed, or fresh scratch)
    /// and a later one. The integer ids carry no gradient.
    pub(crate) fn diff_embedding(&mut self, args: &[ValueId], par: bool) -> Result<(), String> {
        let (out, weight, ids) = (args[0], args[1], args[2]);
        let d = match self.buf_adj.get(&out).copied() {
            Some(d) => d,
            None => return Ok(()),
        };
        if out == weight {
            return Err(format!(
                "autodiff: embedding output v{} aliases its weight table",
                out.0
            ));
        }
        let t = self.dim(args[3]);
        let h = self.dim(args[4]);
        let v = self.dim(args[5]);
        let idsn = self.remap_v(ids);
        if let Some(g) = self.grad_target(weight)? {
            let f = pick(self.syms.scatter_add, self.syms.scatter_add_par, par);
            self.emit(f, vec![g, d, idsn, t, h, v]);
            self.contributed.insert(weight);
        }
        Ok(())
    }
}
