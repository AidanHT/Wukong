//! **Natural spellings** of the elementwise activations (P12.10): an elementwise loop whose body
//! writes an activation out by hand -- HuggingFace's `0.5*x*(1 + tanh(0.7978845608*(x +
//! 0.044715*x*x*x)))`, PyTorch's `x*0.5*(1 + erf(x*0.7071067811865476))`, the textbook
//! `x / (1 + exp(-x))`, `min(max(x + 3, 0), 6) / 6`, `if x > 0 { x } else { expm1(x) }` -- dispatches
//! to a 256-bit `wukong_vmath_f32` kernel, but only to one that computes **that spelling's own
//! bits**.
//!
//! The spellings are different functions from the intrinsics in the last bits (the language's
//! `tanh` is the <=1-ULP piecewise kernel while `gelu()` keeps an exp-composed `tanh`; the HF cubic
//! `((C1*x)*x)*x` rounds differently from `C1*((x*x)*x)`; a division is not a multiplication by a
//! reciprocal; `min(max(x + 3, 0), 6)` sends a NaN to `0` where `hardsigmoid()` keeps it), so no
//! spelling is ever substituted by an intrinsic's kernel unless it IS that kernel's expression.
//! Instead each spelling has its own kernel code (`wukong_runtime::vspell`), whose scalar twin is
//! the spelling's expression operation for operation.
//!
//! The matcher works on **values, not syntax**. [`FnLowerer::spell_value`] turns the loop body's
//! store value -- through any leading `let` bindings -- into the tree of operations the lowering
//! will actually emit ([`V`]): `a + b*c` becomes the `Fma` that `try_contract_fma` builds (and a
//! product bound by a `let` stays a rounded product, because the contraction is decided on the
//! syntax of the `+`), `max`/`min`/`clamp`/`if` become the compare-and-select they lower to, a
//! literal becomes the `f32` it rounds to, and an operation on constants folds to the `f32` the
//! interpreter would compute. That tree is canonicalized ([`canon`]: a constant operand of a `*`,
//! `+` or a fused product is moved first -- exact, because with one non-NaN operand IEEE
//! multiplication and addition are commutative bit for bit, NaN payload included) and compared for
//! equality with the canonical tree of each kernel ([`table`]). Two spellings that lower to the same
//! operations (PyTorch's `x_cube` binding and its inline `(x*x*x)`) share a kernel for free; two
//! that differ in a single rounding never do. A row names an EXISTING kernel only for the tree that
//! IS that kernel's operations, and every row has a `tests/run/spelling_act_*.wk` fixture comparing
//! each lane with the loop the recognizer never sees.
//!
//! A code also names the spelling to the **autodiff tape**, which differentiates the call by it
//! (`wukong_autodiff::tape_nn`). So two trees share a code only when they also share a derivative
//! at every input, kinks included: the backward of a select-spelled activation hands the adjoint
//! to the branch its own compare takes. `x >= 0 ? x : 0.01*x` and `max(x, 0.01*x)` compute the
//! leaky kernel's bits on every input, but the first takes the identity branch at `x = 0` and the
//! second the slope at `x = +inf`, where the leaky kernel's `x > 0` does the opposite -- the same
//! forward, different tie-breaks, hence codes of their own (`VMATH_LEAKY_GE`, `VMATH_LEAKY_MAX`).

use std::sync::OnceLock;

use wukong_ast::{self as ast, Block, Expr, ExprKind, PatKind, Pattern, Stmt, StmtKind};
use wukong_mir::MirType;
use wukong_sema::DefKind;
use wukong_span::Symbol;

use crate::{
    parse_float, FnLowerer, MathIntrinsic, VMATH_ELU, VMATH_LEAKYRELU, VMATH_SIGMOID, VMATH_SILU,
    VMATH_SOFTPLUS,
};

// The spelling op codes (`wukong_runtime::vspell::VM_*`), mirrored like every recognizer constant.
pub(crate) const VMATH_GELU_TANH: u32 = 36;
pub(crate) const VMATH_GELU_TANH_HF: u32 = 37;
pub(crate) const VMATH_GELU_TANH_NOFMA: u32 = 38;
pub(crate) const VMATH_GELU_TANH_FAST: u32 = 39;
pub(crate) const VMATH_GELU_TANH_CDF: u32 = 40;
pub(crate) const VMATH_QUICK_GELU: u32 = 41;
pub(crate) const VMATH_GELU_ERF_DIV: u32 = 42;
pub(crate) const VMATH_GELU_ERF_MUL: u32 = 43;
pub(crate) const VMATH_GELU_ERF_CDF: u32 = 44;
pub(crate) const VMATH_SILU_DIV: u32 = 45;
pub(crate) const VMATH_SILU_SX: u32 = 46;
pub(crate) const VMATH_SIGMOID_RATIO: u32 = 47;
pub(crate) const VMATH_SOFTPLUS_NAIVE: u32 = 48;
pub(crate) const VMATH_SOFTPLUS_LOG1P: u32 = 49;
pub(crate) const VMATH_MISH_NAIVE: u32 = 50;
pub(crate) const VMATH_MISH_SOFTPLUS: u32 = 51;
pub(crate) const VMATH_MISH_LOG1P: u32 = 52;
pub(crate) const VMATH_HARDSIGMOID_MINMAX: u32 = 53;
pub(crate) const VMATH_HARDSIGMOID_CLAMP: u32 = 54;
pub(crate) const VMATH_HARDSWISH_MINMAX: u32 = 55;
pub(crate) const VMATH_HARDSWISH_CLAMP: u32 = 56;
pub(crate) const VMATH_ELU_EXPM1: u32 = 57;
pub(crate) const VMATH_LEAKY_LT: u32 = 58;
pub(crate) const VMATH_LEAKY_GE: u32 = 59;
pub(crate) const VMATH_LEAKY_MAX: u32 = 60;

/// A unary function a spelling applies -- each the language's own inlined form, which the runtime
/// kernel of the same name mirrors op for op (`emit_tanh` / `tanh1`, `emit_erf` / `erf1`, ...).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fun {
    Exp,
    Log,
    Tanh,
    Erf,
    Sigmoid,
    Softplus,
    Log1p,
    Expm1,
    /// The sign-bit clear (`emit_abs`).
    Abs,
    /// Only ever folded into a constant: no spelling takes the square root of the input.
    Sqrt,
}

/// One value of a spelling: the operation the lowering emits for it, over the loaded element `X`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum V {
    /// The element `x[j]` (one array throughout a spelling).
    X,
    /// An `f32` constant, by its bits.
    K(u32),
    Add(Box<V>, Box<V>),
    Sub(Box<V>, Box<V>),
    Mul(Box<V>, Box<V>),
    Div(Box<V>, Box<V>),
    /// `a*b + c`, rounded once (a contracted source `+`).
    Fma(Box<V>, Box<V>, Box<V>),
    /// The sign flip of a unary `-`.
    Neg(Box<V>),
    F(Fun, Box<V>),
    /// `a > b ? t : e` (ordered: a NaN operand selects `e`).
    Gt(Box<V>, Box<V>, Box<V>, Box<V>),
    /// `a >= b ? t : e` (ordered).
    Ge(Box<V>, Box<V>, Box<V>, Box<V>),
}

impl V {
    fn konst(&self) -> Option<f32> {
        match self {
            V::K(b) => Some(f32::from_bits(*b)),
            _ => None,
        }
    }
}

/// Canonicalize a tree: fold operations on constants to the `f32` the interpreter computes (one
/// IEEE `f32` operation each, which is also what the optimizer's fold produces), and move a
/// constant operand of a `+`, a `*` or a fused product first. Both rewrites keep every bit: the
/// fold is the very operation, and with one operand a non-NaN constant the two operand orders of
/// an IEEE `+` or `*` return the same bits (a NaN result is the other operand's NaN, quieted, in
/// either order).
pub(crate) fn canon(v: V) -> V {
    fn b(v: V) -> Box<V> {
        Box::new(canon(v))
    }
    match v {
        V::X | V::K(_) => v,
        V::Neg(a) => match canon(*a) {
            V::K(bits) => V::K(bits ^ 0x8000_0000),
            a => V::Neg(Box::new(a)),
        },
        // `sigmoid` and `softplus` ARE the operations they lower to (`emit_sigmoid`,
        // `emit_softplus`; the runtime's `sigmoid1`/`softplus1` likewise), so a spelling that writes
        // them out (`x * (1.0 / (1.0 + exp(-x)))`) is the same tree as one that calls them.
        V::F(Fun::Sigmoid, a) => canon(div(k(1.0), add(k(1.0), f(Fun::Exp, neg(*a))))),
        V::F(Fun::Softplus, a) => canon(add(
            max((*a).clone(), k(0.0)),
            f(Fun::Log, add(k(1.0), f(Fun::Exp, neg(f(Fun::Abs, *a))))),
        )),
        V::F(fun, a) => {
            let a = canon(*a);
            match (fun, a.konst()) {
                (Fun::Sqrt, Some(c)) => k(c.sqrt()),
                _ => V::F(fun, Box::new(a)),
            }
        }
        V::Add(l, r) => commute(canon(*l), canon(*r), V::Add, |x, y| x + y),
        V::Mul(l, r) => commute(canon(*l), canon(*r), V::Mul, |x, y| x * y),
        V::Sub(l, r) => fold2(canon(*l), canon(*r), V::Sub, |x, y| x - y),
        V::Div(l, r) => fold2(canon(*l), canon(*r), V::Div, |x, y| x / y),
        V::Fma(a, bb, c) => {
            let (a, bb, c) = (canon(*a), canon(*bb), canon(*c));
            match (a.konst(), bb.konst(), c.konst()) {
                (Some(p), Some(q), Some(r)) => k(p.mul_add(q, r)),
                (None, Some(_), _) => V::Fma(Box::new(bb), Box::new(a), Box::new(c)),
                _ => V::Fma(Box::new(a), Box::new(bb), Box::new(c)),
            }
        }
        V::Gt(a, bb, t, e) => V::Gt(b(*a), b(*bb), b(*t), b(*e)),
        V::Ge(a, bb, t, e) => V::Ge(b(*a), b(*bb), b(*t), b(*e)),
    }
}

/// A commutative operation: fold two constants, else put a lone constant first.
fn commute(l: V, r: V, mk: fn(Box<V>, Box<V>) -> V, f: fn(f32, f32) -> f32) -> V {
    match (l.konst(), r.konst()) {
        (Some(x), Some(y)) => V::K(f(x, y).to_bits()),
        (None, Some(_)) => mk(Box::new(r), Box::new(l)),
        _ => mk(Box::new(l), Box::new(r)),
    }
}

/// A non-commutative operation: fold two constants, else keep the order.
fn fold2(l: V, r: V, mk: fn(Box<V>, Box<V>) -> V, f: fn(f32, f32) -> f32) -> V {
    match (l.konst(), r.konst()) {
        (Some(x), Some(y)) => V::K(f(x, y).to_bits()),
        _ => mk(Box::new(l), Box::new(r)),
    }
}

// --- the kernel table ------------------------------------------------------------------------------

fn x() -> V {
    V::X
}
fn k(f: f32) -> V {
    V::K(f.to_bits())
}
fn kb(bits: u32) -> V {
    V::K(bits)
}
fn add(a: V, b: V) -> V {
    V::Add(Box::new(a), Box::new(b))
}
fn sub(a: V, b: V) -> V {
    V::Sub(Box::new(a), Box::new(b))
}
fn mul(a: V, b: V) -> V {
    V::Mul(Box::new(a), Box::new(b))
}
fn div(a: V, b: V) -> V {
    V::Div(Box::new(a), Box::new(b))
}
fn fma(a: V, b: V, c: V) -> V {
    V::Fma(Box::new(a), Box::new(b), Box::new(c))
}
fn neg(a: V) -> V {
    V::Neg(Box::new(a))
}
fn f(fun: Fun, a: V) -> V {
    V::F(fun, Box::new(a))
}
fn gt(a: V, b: V, t: V, e: V) -> V {
    V::Gt(Box::new(a), Box::new(b), Box::new(t), Box::new(e))
}
fn ge(a: V, b: V, t: V, e: V) -> V {
    V::Ge(Box::new(a), Box::new(b), Box::new(t), Box::new(e))
}
/// `max(a, b)` as `minmax.rs` lowers it: `a > b ? a : b`.
fn max(a: V, b: V) -> V {
    gt(a.clone(), b.clone(), a, b)
}
/// `min(a, b)`: `a < b ? a : b`, i.e. `b > a ? a : b`.
fn min(a: V, b: V) -> V {
    gt(b.clone(), a.clone(), a, b)
}
/// `clamp(v, lo, hi)` as `sb_clamp` lowers it: `v < lo ? lo : (v > hi ? hi : v)`.
fn clamp(v: V, lo: V, hi: V) -> V {
    gt(lo.clone(), v.clone(), lo, gt(v.clone(), hi.clone(), hi, v))
}

/// `0.7978845608` (sqrt(2/pi)) as `f32` -- also `0.7978845608028654` and `0.79788456`.
const C0: u32 = 0x3f4c_422a;
/// `0.044715` as `f32`.
const C1: u32 = 0x3d37_2713;
/// QuickGELU's `1.702`.
const QUICK: u32 = 0x3fd9_db23;
/// `sqrt(2.0)` / `1.4142135623730951` as `f32`.
const SQRT2: u32 = 0x3fb5_04f3;
/// `0.7071067811865476` / `0.70710678` / `1.0 / sqrt(2.0)` as `f32`.
const RSQRT2: u32 = 0x3f35_04f3;

/// `0.5*x*(1 + tanh(inner))` with `inner` the GELU argument -- the outer shape every `0.5*x`-first
/// tanh GELU shares.
fn half_x_times(inner_tanh_arg: V) -> V {
    mul(mul(k(0.5), x()), add(k(1.0), f(Fun::Tanh, inner_tanh_arg)))
}

/// `fma(C1, (x*x)*x, x)` -- `x + C1*(x*x*x)` contracted.
fn cubic_grouped() -> V {
    fma(kb(C1), mul(mul(x(), x()), x()), x())
}

/// The kernel table: `(canonical tree, op code)`. Several trees may name one code; each row's
/// comment says what source it is and, for a row naming an existing kernel, why the kernel IS that
/// tree on every input.
fn table() -> &'static [(V, u32)] {
    static TABLE: OnceLock<Vec<(V, u32)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let hardsig_minmax = min(max(add(k(3.0), x()), k(0.0)), k(6.0));
        let hardsig_clamp = clamp(add(k(3.0), x()), k(0.0), k(6.0));
        let rows = vec![
            // ---- GELU, tanh approximation ----
            // PyTorch's C++ kernel (`x_cube = x*x*x`), `C1*(x*x*x)` inline or bound by a `let`.
            (half_x_times(mul(kb(C0), cubic_grouped())), VMATH_GELU_TANH),
            // HuggingFace `NewGELUActivation` / GPT-2: `x + C1*x*x*x` contracts as
            // `fma(round(round(C1*x)*x), x, x)`.
            (
                half_x_times(mul(kb(C0), fma(mul(mul(kb(C1), x()), x()), x(), x()))),
                VMATH_GELU_TANH_HF,
            ),
            // llm.c: `let cube = C1*x*x*x; ... x + cube` -- a bound product does not contract.
            (
                half_x_times(mul(kb(C0), add(x(), mul(mul(mul(kb(C1), x()), x()), x())))),
                VMATH_GELU_TANH_NOFMA,
            ),
            // HuggingFace `FastGELUActivation`: `x*C0*(1 + C1*x*x)`.
            (
                half_x_times(mul(mul(kb(C0), x()), fma(mul(kb(C1), x()), x(), k(1.0)))),
                VMATH_GELU_TANH_FAST,
            ),
            // JAX `gelu(approximate=True)`: `x * (0.5*(1 + tanh(C0*(x + C1*(x*x*x)))))`.
            (
                mul(
                    x(),
                    mul(
                        k(0.5),
                        add(k(1.0), f(Fun::Tanh, mul(kb(C0), cubic_grouped()))),
                    ),
                ),
                VMATH_GELU_TANH_CDF,
            ),
            // HuggingFace `QuickGELUActivation`: `x * sigmoid(1.702*x)`.
            (
                mul(x(), f(Fun::Sigmoid, mul(kb(QUICK), x()))),
                VMATH_QUICK_GELU,
            ),
            // ---- GELU, exact ----
            // BERT `x*0.5*(1 + erf(x / sqrt(2)))`.
            (
                mul(
                    mul(k(0.5), x()),
                    add(k(1.0), f(Fun::Erf, div(x(), kb(SQRT2)))),
                ),
                VMATH_GELU_ERF_DIV,
            ),
            // PyTorch's C++ kernel `x*0.5*(1 + erf(x*M_SQRT1_2))`.
            (
                mul(
                    mul(k(0.5), x()),
                    add(k(1.0), f(Fun::Erf, mul(kb(RSQRT2), x()))),
                ),
                VMATH_GELU_ERF_MUL,
            ),
            // JAX `gelu(approximate=False)`: `x*(erf(x/sqrt(2)) + 1)/2`.
            (
                div(
                    mul(x(), add(k(1.0), f(Fun::Erf, div(x(), kb(SQRT2))))),
                    k(2.0),
                ),
                VMATH_GELU_ERF_CDF,
            ),
            // ---- SiLU / swish ----
            // `x * sigmoid(x)` IS `vmath::silu1` (`x * sigmoid1(x)`, its NaN the input's quieted,
            // which the product's first operand also gives).
            (mul(x(), f(Fun::Sigmoid, x())), VMATH_SILU),
            // `sigmoid(x) * x`: the NaN comes from the sigmoid's negated exponential -- its own code.
            (mul(f(Fun::Sigmoid, x()), x()), VMATH_SILU_SX),
            // The textbook `x / (1 + exp(-x))`.
            (div(x(), add(k(1.0), f(Fun::Exp, neg(x())))), VMATH_SILU_DIV),
            // ---- sigmoid ----
            // `1 / (1 + exp(-x))` IS `vmath::sigmoid1` and the language's `sigmoid` (`emit_sigmoid`).
            (
                div(k(1.0), add(k(1.0), f(Fun::Exp, neg(x())))),
                VMATH_SIGMOID,
            ),
            // `exp(x) / (1 + exp(x))`.
            (
                div(f(Fun::Exp, x()), add(k(1.0), f(Fun::Exp, x()))),
                VMATH_SIGMOID_RATIO,
            ),
            // ---- softplus ----
            // `max(x, 0) + log(1 + exp(-abs(x)))` IS `vmath::softplus1` and `emit_softplus`.
            (
                add(
                    max(x(), k(0.0)),
                    f(Fun::Log, add(k(1.0), f(Fun::Exp, neg(f(Fun::Abs, x()))))),
                ),
                VMATH_SOFTPLUS,
            ),
            (
                f(Fun::Log, add(k(1.0), f(Fun::Exp, x()))),
                VMATH_SOFTPLUS_NAIVE,
            ),
            (f(Fun::Log1p, f(Fun::Exp, x())), VMATH_SOFTPLUS_LOG1P),
            // ---- mish, over the language's `tanh` ----
            (
                mul(
                    x(),
                    f(Fun::Tanh, f(Fun::Log, add(k(1.0), f(Fun::Exp, x())))),
                ),
                VMATH_MISH_NAIVE,
            ),
            (
                mul(x(), f(Fun::Tanh, f(Fun::Softplus, x()))),
                VMATH_MISH_SOFTPLUS,
            ),
            (
                mul(x(), f(Fun::Tanh, f(Fun::Log1p, f(Fun::Exp, x())))),
                VMATH_MISH_LOG1P,
            ),
            // ---- hard sigmoid / hard swish: `relu6(x + 3) / 6` ----
            (
                div(hardsig_minmax.clone(), k(6.0)),
                VMATH_HARDSIGMOID_MINMAX,
            ),
            (div(hardsig_clamp.clone(), k(6.0)), VMATH_HARDSIGMOID_CLAMP),
            (
                div(mul(x(), hardsig_minmax), k(6.0)),
                VMATH_HARDSWISH_MINMAX,
            ),
            (div(mul(x(), hardsig_clamp), k(6.0)), VMATH_HARDSWISH_CLAMP),
            // ---- leaky ReLU, slope 0.01 ----
            // `x > 0 ? x : 0.01*x` IS `vmath::leakyrelu1` (`LEAKY_ALPHA * x`; the product commutes).
            (gt(x(), k(0.0), x(), mul(k(0.01), x())), VMATH_LEAKYRELU),
            // `x >= 0 ? x : 0.01*x` differs from the `>` form only at x = +-0, where it returns x
            // and the kernel returns 0.01*x -- the same zero, sign included; a NaN fails both
            // compares. So its kernel computes the leaky kernel's bits, but under its own code: its
            // derivative at +-0 is the identity branch's 1, not the leaky kernel's 0.01.
            (ge(x(), k(0.0), x(), mul(k(0.01), x())), VMATH_LEAKY_GE),
            // `max(x, 0.01*x)` = `x > 0.01*x ? x : 0.01*x`. For finite x > 0, 0.01*x < x (the
            // product rounds below x), so it returns x; for x < 0 (and -inf), 0.01*x >= x, so it
            // returns 0.01*x; at +-0 and +inf the compare is false and it returns 0.01*x, which is
            // x's zero and +inf -- what the leaky kernel returns at +-0 and +inf; a NaN fails the
            // compare and both return 0.01*x. So its kernel computes the leaky kernel's bits, under
            // its own code: at +inf its derivative is the product's 0.01, not the leaky kernel's 1.
            (max(x(), mul(k(0.01), x())), VMATH_LEAKY_MAX),
            // `x < 0 ? 0.01*x : x` returns a NaN untouched (no arithmetic on it) -- its own code.
            (gt(k(0.0), x(), mul(k(0.01), x()), x()), VMATH_LEAKY_LT),
            // ---- ELU, alpha 1 ----
            // `x > 0 ? x : exp(x) - 1` IS `vmath::elu1` and `emit_elu`.
            (
                gt(x(), k(0.0), x(), sub(f(Fun::Exp, x()), k(1.0))),
                VMATH_ELU,
            ),
            (gt(x(), k(0.0), x(), f(Fun::Expm1, x())), VMATH_ELU_EXPM1),
        ];
        rows.into_iter().map(|(v, code)| (canon(v), code)).collect()
    })
}

/// The kernel code whose canonical tree `v` is, if any.
pub(crate) fn kernel_for(v: V) -> Option<u32> {
    let v = canon(v);
    table().iter().find(|(t, _)| *t == v).map(|&(_, code)| code)
}

// --- from the AST to the value tree ----------------------------------------------------------------

/// The `let` bindings in scope while a store value is read: each name's value tree.
type Env = Vec<(Symbol, V)>;

impl FnLowerer<'_> {
    /// The value tree `e` lowers to (see the module docs), reading the element through `x[j]`
    /// (`batch` as in [`Self::index_off`]) and recording the array in `xs`. `None` -- decline --
    /// for anything that is not an `f32` value built from that element, `f32` literals, the
    /// arithmetic operators and the builtins the spellings use.
    fn spell_value(
        &self,
        e: &Expr,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
        env: &Env,
        xs: &mut Option<Symbol>,
    ) -> Option<V> {
        if self.expr_mir(e) != MirType::F32 {
            return None;
        }
        let sub = |e: &Expr, xs: &mut Option<Symbol>| self.spell_value(e, j, batch, env, xs);
        Some(match &e.kind {
            ExprKind::Index { .. } => {
                let s = self.index_off(e, j, batch)?;
                if xs.replace(s).is_some_and(|prev| prev != s) {
                    return None;
                }
                V::X
            }
            ExprKind::Path(p) if p.is_single() => {
                let name = p.first().sym;
                env.iter().rev().find(|(n, _)| *n == name)?.1.clone()
            }
            ExprKind::Float(t) => V::K((parse_float(self.interner.resolve(*t)) as f32).to_bits()),
            ExprKind::Unary {
                op: ast::UnOp::Neg,
                expr,
            } => V::Neg(Box::new(sub(expr, xs)?)),
            ExprKind::Binary { op, lhs, rhs } => {
                match op {
                    // `try_contract_fma`: the LEFT product of a `+` fuses, else the right one.
                    ast::BinOp::Add if !self.gemm.strict_fp => {
                        if let Some((y, z)) = fmul(lhs) {
                            V::Fma(
                                Box::new(sub(y, xs)?),
                                Box::new(sub(z, xs)?),
                                Box::new(sub(rhs, xs)?),
                            )
                        } else if let Some((y, z)) = fmul(rhs) {
                            V::Fma(
                                Box::new(sub(y, xs)?),
                                Box::new(sub(z, xs)?),
                                Box::new(sub(lhs, xs)?),
                            )
                        } else {
                            V::Add(Box::new(sub(lhs, xs)?), Box::new(sub(rhs, xs)?))
                        }
                    }
                    ast::BinOp::Add => V::Add(Box::new(sub(lhs, xs)?), Box::new(sub(rhs, xs)?)),
                    ast::BinOp::Sub => V::Sub(Box::new(sub(lhs, xs)?), Box::new(sub(rhs, xs)?)),
                    ast::BinOp::Mul => V::Mul(Box::new(sub(lhs, xs)?), Box::new(sub(rhs, xs)?)),
                    ast::BinOp::Div => V::Div(Box::new(sub(lhs, xs)?), Box::new(sub(rhs, xs)?)),
                    _ => return None,
                }
            }
            ExprKind::Call { callee, args, .. } => {
                if let Some(op) = self.vectorizable_intrinsic(callee) {
                    let fun = match op {
                        MathIntrinsic::Exp => Fun::Exp,
                        MathIntrinsic::Log => Fun::Log,
                        MathIntrinsic::Tanh => Fun::Tanh,
                        MathIntrinsic::Erf => Fun::Erf,
                        MathIntrinsic::Sigmoid => Fun::Sigmoid,
                        MathIntrinsic::Softplus => Fun::Softplus,
                        MathIntrinsic::Log1p => Fun::Log1p,
                        MathIntrinsic::Expm1 => Fun::Expm1,
                        MathIntrinsic::Abs => Fun::Abs,
                        MathIntrinsic::Sqrt => Fun::Sqrt,
                        MathIntrinsic::Fmax | MathIntrinsic::Fmin if args.len() == 2 => {
                            let a = sub(&args[0], xs)?;
                            let b = sub(&args[1], xs)?;
                            return Some(if matches!(op, MathIntrinsic::Fmax) {
                                max(a, b)
                            } else {
                                min(a, b)
                            });
                        }
                        _ => return None,
                    };
                    if args.len() != 1 {
                        return None;
                    }
                    V::F(fun, Box::new(sub(&args[0], xs)?))
                } else {
                    // `min`/`max` (`minmax.rs`), `clamp` and `fma` (`scalar_builtins.rs`), unless a
                    // user function of the name shadows the builtin.
                    let name = single_name(callee)?;
                    if matches!(
                        self.sema.defs.lookup(name).map(|d| &d.kind),
                        Some(DefKind::Fn(_))
                    ) {
                        return None;
                    }
                    let mut vals = Vec::with_capacity(args.len());
                    for a in args {
                        vals.push(sub(a, xs)?);
                    }
                    match (self.interner.resolve(name), vals.as_slice()) {
                        ("max", [a, b]) => max(a.clone(), b.clone()),
                        ("min", [a, b]) => min(a.clone(), b.clone()),
                        ("clamp", [v, lo, hi]) => clamp(v.clone(), lo.clone(), hi.clone()),
                        ("fma", [a, b, c]) => fma(a.clone(), b.clone(), c.clone()),
                        _ => return None,
                    }
                }
            }
            // `if a <cmp> b { t } else { e }` over `f32` operands: the ordered compare's select.
            ExprKind::If {
                cond,
                then_branch,
                else_branch: Some(else_branch),
            } => {
                let t = tail_only(then_branch)?;
                let ExprKind::Block(eb) = &else_branch.kind else {
                    return None;
                };
                let el = tail_only(eb)?;
                let ExprKind::Binary { op, lhs, rhs } = &cond.kind else {
                    return None;
                };
                let a = sub(lhs, xs)?;
                let b = sub(rhs, xs)?;
                let t = sub(t, xs)?;
                let el = sub(el, xs)?;
                match op {
                    ast::BinOp::Gt => gt(a, b, t, el),
                    ast::BinOp::Lt => gt(b, a, t, el),
                    ast::BinOp::Ge => ge(a, b, t, el),
                    ast::BinOp::Le => ge(b, a, t, el),
                    _ => return None,
                }
            }
            _ => return None,
        })
    }

    /// One store `out[j] = <spelling>` with the `let` bindings `env` in scope, whose values read
    /// the array `xs` (if any): `(out, x, code)`.
    fn match_spelling_store(
        &self,
        stmt: &Stmt,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
        env: &Env,
        mut xs: Option<Symbol>,
    ) -> Option<(Symbol, Symbol, u32)> {
        let StmtKind::Assign {
            target,
            op: ast::AssignOp::Assign,
            value,
        } = &stmt.kind
        else {
            return None;
        };
        if !stmt.attrs.is_empty() || self.expr_mir(target) != MirType::F32 {
            return None;
        }
        let out = self.index_off(target, j, batch)?;
        let v = self.spell_value(value, j, batch, env, &mut xs)?;
        // A spelling of constants alone has no input array.
        let x = xs?;
        Some((out, x, kernel_for(v)?))
    }

    /// The let-free statement form, one `out[j] = <spelling>` among the statements of a
    /// `match_vmath_body` body.
    pub(super) fn match_spelling_stmt(
        &self,
        stmt: &Stmt,
        j: Symbol,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Symbol, Symbol, u32)> {
        self.match_spelling_store(stmt, j, batch, &Vec::new(), None)
    }

    /// A body of `let` bindings followed by ONE store, `let v = x[j]; let x3 = v*v*v; ...;
    /// out[j] = <spelling>`: `(out, x, code)`. Each binding is read as the value it holds -- its
    /// tree, which keeps a bound product a rounded product -- so nothing about the bindings is
    /// assumed beyond what the tree records. Every binding must be an immutable, unannotated-or-`f32`
    /// name bound to an `f32` spelling value of the same element (pure, so evaluating it once at
    /// the `let` or wherever the kernel needs it is the same value).
    pub(super) fn match_spelling_with_lets(
        &self,
        j: Symbol,
        body: &Block,
        batch: Option<(Symbol, &Expr)>,
    ) -> Option<(Symbol, Symbol, u32)> {
        if body.tail.is_some() {
            return None;
        }
        let (store, lets) = body.stmts.split_last()?;
        if lets.is_empty() {
            return None;
        }
        let mut env: Env = Vec::with_capacity(lets.len());
        let mut xs = None;
        for s in lets {
            let StmtKind::Let {
                pat:
                    Pattern {
                        kind: PatKind::Ident(name),
                        ..
                    },
                mutable: false,
                init: Some(init),
                ..
            } = &s.kind
            else {
                return None;
            };
            if !s.attrs.is_empty() || *name == j {
                return None;
            }
            let v = self.spell_value(init, j, batch, &env, &mut xs)?;
            env.push((*name, v));
        }
        // The store continues the bindings' element: one array throughout.
        self.match_spelling_store(store, j, batch, &env, xs)
    }
}

/// The two factors of a syntactic `*` -- the operand shape `try_contract_fma` fuses.
fn fmul(e: &Expr) -> Option<(&Expr, &Expr)> {
    match &e.kind {
        ExprKind::Binary {
            op: ast::BinOp::Mul,
            lhs,
            rhs,
        } => Some((lhs.as_ref(), rhs.as_ref())),
        _ => None,
    }
}

/// The single name of a callee path.
fn single_name(callee: &Expr) -> Option<Symbol> {
    match &callee.kind {
        ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
        _ => None,
    }
}

/// The tail expression of a block with no statements.
fn tail_only(b: &Block) -> Option<&Expr> {
    if b.stmts.is_empty() {
        b.tail.as_deref()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table row is canonical and names one code; no two rows hold the same tree (a
    /// duplicate would make the code a row dispatches depend on table order).
    #[test]
    fn the_spelling_table_has_no_duplicate_trees() {
        let t = table();
        for (i, (a, _)) in t.iter().enumerate() {
            assert_eq!(&canon(a.clone()), a, "row {i} is not canonical");
            for (b, _) in &t[i + 1..] {
                assert_ne!(a, b, "row {i} appears twice");
            }
        }
    }

    /// A constant operand moves first, and only a constant: swapping two non-constant operands
    /// changes which NaN a NaN input yields, so it is not an identity.
    #[test]
    fn canonical_form_commutes_constants_only() {
        assert_eq!(canon(mul(x(), k(0.5))), canon(mul(k(0.5), x())));
        assert_eq!(
            canon(add(f(Fun::Exp, x()), k(1.0))),
            add(k(1.0), f(Fun::Exp, x()))
        );
        assert_ne!(
            canon(mul(f(Fun::Sigmoid, x()), x())),
            canon(mul(x(), f(Fun::Sigmoid, x())))
        );
        assert_eq!(canon(div(k(1.0), f(Fun::Sqrt, k(2.0)))), kb(RSQRT2));
        assert_eq!(canon(f(Fun::Sqrt, k(2.0))), kb(SQRT2));
        assert_eq!(canon(neg(k(1.0))), k(-1.0));
        assert_eq!(
            canon(fma(x(), kb(C1), x())),
            fma(kb(C1), x(), x()),
            "a fused product commutes like a product"
        );
    }
}
