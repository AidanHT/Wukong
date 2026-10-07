//! `VecKernel` — a backend-agnostic recipe for a vectorizable straight-line loop body, produced by
//! the general vectorizer (`wukong_mir_build`) and consumed by two backends that must agree
//! bit-for-bit:
//!
//! * the **interpreter** marshals it lane-by-lane through its abstract slot memory (the oracle), and
//! * the **Cranelift** backend assembles it into true 256-bit AVX2 machine code — the width Cranelift's
//!   128-bit CLIF vector ISA cannot legalize (see `wukong_codegen_cranelift::avx2`).
//!
//! Both derive their semantics from the reference evaluators below — [`VecKernel::eval_lane`] for an
//! elementwise recipe, [`VecKernel::eval_reduction`] for a reduction one (both funnel through the same
//! private `eval_all`) — so the recipe is the single source of truth, tails included: the caller runs
//! the non-multiple-of-lanes remainder scalar-side and the differential gate polices the whole loop.
//!
//! Every recipe carries one [`VecLane`] element type — `f32` (8 lanes per YMM register), `f64` (4),
//! `i32` (8) or `i64` (4) — and the reference evaluators are generic over the matching Rust type
//! ([`VecValue`]), so each lane type has exactly the arithmetic its AVX2 instruction family
//! performs: IEEE single/double rounding with a single-rounded `mul_add` for the two float lanes,
//! two's-complement wrapping for the two integer lanes.
//!
//! A kernel has the fixed ABI `fn(ptrs: *const *mut u8, scalars: *const T, n: u64)`: `ptrs[k]` is the
//! (already `start`-offset) base of unit-stride stream `k`, `scalars[k]` the k-th loop-invariant value
//! of the lane type `T` (a param, an outer local, or a stride-0 array read the caller pre-loaded), and
//! `n` the multiple-of-[`VecLane::lanes`] element count assigned to the vector part. The caller runs
//! the scalar remainder. An elementwise kernel returns nothing and writes through its output streams;
//! a reduction kernel (`reduce.is_some()`) writes no memory and returns its horizontal fold as a `T`
//! (in XMM0 for a float lane, RAX/EAX for an integer one). Both shapes are declared to Cranelift from
//! this one signature, so `reduce` and `lane` are what pick between them.
//!
//! # The canonical reduction tree
//!
//! A reduction's float result depends on how its addends are grouped, so the grouping is a property
//! of the *program*, fixed here once and reproduced by every lowering of the same loop — the 256-bit
//! kernel, the portable 128-bit CLIF strips (`wukong_mir_build` represents each logical accumulator
//! as a pair of 128-bit vectors), the interpreter, `-O0` and `-O2`, every host, every trip count:
//!
//! 1. `U` = [`VecKernel::reduction_unroll`] logical accumulators of `L` = [`VecLane::lanes`] lanes
//!    each, initialised to the fold identity. `U` is the recipe's requested unroll — **never** a
//!    function of register pressure: a body too large for `U` accumulators plus one register bank
//!    simply does not take the 256-bit kernel ([`VecKernel::reduction_fits`]), and the portable path
//!    computes the identical tree.
//! 2. Main loop: element `i + u·L + l` of each `U·L` block folds into accumulator `u`, lane `l`.
//! 3. Cleanup: every remaining full `L`-element group folds into accumulator 0.
//! 4. Accumulators `1..U` fold into accumulator 0 lane-wise, in order `1, 2, …, U-1`.
//! 5. Lanes `0..L` of accumulator 0 fold left to right into one value `r`.
//! 6. When at least one full group ran, the scalar seed `s` becomes `fold(s, r)`; with no full group
//!    the seed is left untouched (so a `-0.0` or NaN seed is never disturbed by the identity).
//! 7. The `< L` remaining elements fold into `s` one at a time, in index order.
//!
//! `fold` is `+` for a sum (a fused `fma(x, y, acc)` when the addend is a product) and the x86
//! operand-order `max`/`min` for a running maximum/minimum ([`VecRedOp::fold`]). Integer sums wrap
//! and are exact under any grouping; the tree is the same anyway.

use crate::MirType;
use wukong_span::Symbol;

/// The element type of a recipe's lanes. One YMM register holds [`VecLane::lanes`] of them.
///
/// The integer lanes are *signless* like every MIR integer: arithmetic wraps, and a compare is the
/// signed one (`vpcmpgtd`/`vpcmpgtq` are the only ordered integer compares AVX2 has), so the
/// vectorizer must not hand an unsigned comparison to an integer recipe.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum VecLane {
    #[default]
    F32,
    F64,
    I32,
    I64,
}

impl VecLane {
    /// Bytes per lane.
    pub fn bytes(self) -> u32 {
        match self {
            VecLane::F32 | VecLane::I32 => 4,
            VecLane::F64 | VecLane::I64 => 8,
        }
    }

    /// Lanes per 256-bit register: 8 for the 32-bit lanes, 4 for the 64-bit ones.
    pub fn lanes(self) -> u32 {
        VEC_BYTES / self.bytes()
    }

    pub fn is_float(self) -> bool {
        matches!(self, VecLane::F32 | VecLane::F64)
    }

    /// The MIR type of one lane (and of a reduction kernel's result).
    pub fn mir_type(self) -> MirType {
        match self {
            VecLane::F32 => MirType::F32,
            VecLane::F64 => MirType::F64,
            VecLane::I32 => MirType::I32,
            VecLane::I64 => MirType::I64,
        }
    }

    /// The lane type for a MIR element type, or `None` for one the AVX2 recipe does not cover.
    pub fn of(t: &MirType) -> Option<VecLane> {
        Some(match t {
            MirType::F32 => VecLane::F32,
            MirType::F64 => VecLane::F64,
            MirType::I32 => VecLane::I32,
            MirType::I64 => VecLane::I64,
            _ => return None,
        })
    }
}

/// One lane's value in the reference evaluators, implemented for the four Rust types that model the
/// four [`VecLane`]s. Each method is the exact per-lane semantics of the AVX2 instruction the
/// assembler emits for that lane type, so the interpreter (which runs these) and the machine code
/// agree bit-for-bit. Integer methods for ops [`VecKernel::validate`] rejects on an integer lane
/// (`Div`, `Fma`, `Sqrt`, `Fmax`/`Fmin` folds) are still total — never a panic — but unreachable on a
/// validated recipe.
pub trait VecValue: Copy {
    /// The lane type this Rust type models.
    const LANE: VecLane;
    /// A `Const` immediate (an `f32` bit pattern; only well-formed on an `F32` lane).
    fn from_const_bits(bits: u32) -> Self;
    /// Lane-wise `x ⊕ y`.
    fn bin(op: VecBin, x: Self, y: Self) -> Self;
    /// `a * b + c` — single-rounded for a float lane (`vfmadd`, Rust `mul_add`).
    fn fused(a: Self, b: Self, c: Self) -> Self;
    /// Lane-wise square root.
    fn root(self) -> Self;
    /// Negation: a sign-bit flip for a float lane (bit-exact on ±0 and NaN), `0 - x` for an integer.
    fn negate(self) -> Self;
    /// The compare `pred` evaluates, as a Rust boolean (see [`VecCmp`] for the NaN rules).
    fn compare(pred: VecCmp, x: Self, y: Self) -> bool;
    /// A compare's lane value in the reference: `1` when true, `0` when false.
    fn mask(t: bool) -> Self;
    /// Does a mask lane select the `Select`'s first operand (non-zero)?
    fn is_set(self) -> bool;
    /// The fold identity of a reduction.
    fn identity(op: VecRedOp) -> Self;
    /// `fold(acc, x)` with the assembler's operand order (see [`VecRedOp::fold`]).
    fn fold(op: VecRedOp, acc: Self, x: Self) -> Self;
    /// The placeholder a `Store` leaves in the value list (stores carry no value; never read).
    fn unused() -> Self;
}

impl VecValue for f32 {
    const LANE: VecLane = VecLane::F32;
    fn from_const_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }
    fn bin(op: VecBin, x: Self, y: Self) -> Self {
        match op {
            VecBin::Add => x + y,
            VecBin::Sub => x - y,
            VecBin::Mul => x * y,
            VecBin::Div => x / y,
        }
    }
    fn fused(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
    fn root(self) -> Self {
        f32::sqrt(self)
    }
    fn negate(self) -> Self {
        f32::from_bits(self.to_bits() ^ 0x8000_0000)
    }
    fn compare(pred: VecCmp, x: Self, y: Self) -> bool {
        float_compare(pred, x, y)
    }
    fn mask(t: bool) -> Self {
        if t {
            1.0
        } else {
            0.0
        }
    }
    fn is_set(self) -> bool {
        self != 0.0
    }
    fn identity(op: VecRedOp) -> Self {
        op.identity()
    }
    fn fold(op: VecRedOp, acc: Self, x: Self) -> Self {
        op.fold(acc, x)
    }
    fn unused() -> Self {
        f32::NAN
    }
}

impl VecValue for f64 {
    const LANE: VecLane = VecLane::F64;
    fn from_const_bits(bits: u32) -> Self {
        f32::from_bits(bits) as f64
    }
    fn bin(op: VecBin, x: Self, y: Self) -> Self {
        match op {
            VecBin::Add => x + y,
            VecBin::Sub => x - y,
            VecBin::Mul => x * y,
            VecBin::Div => x / y,
        }
    }
    fn fused(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
    fn root(self) -> Self {
        f64::sqrt(self)
    }
    fn negate(self) -> Self {
        f64::from_bits(self.to_bits() ^ 0x8000_0000_0000_0000)
    }
    fn compare(pred: VecCmp, x: Self, y: Self) -> bool {
        float_compare(pred, x, y)
    }
    fn mask(t: bool) -> Self {
        if t {
            1.0
        } else {
            0.0
        }
    }
    fn is_set(self) -> bool {
        self != 0.0
    }
    fn identity(op: VecRedOp) -> Self {
        match op {
            VecRedOp::Add => 0.0,
            VecRedOp::Fmax => f64::NEG_INFINITY,
            VecRedOp::Fmin => f64::INFINITY,
        }
    }
    fn fold(op: VecRedOp, acc: Self, x: Self) -> Self {
        match op {
            VecRedOp::Add => acc + x,
            VecRedOp::Fmax => {
                if acc > x {
                    acc
                } else {
                    x
                }
            }
            VecRedOp::Fmin => {
                if acc < x {
                    acc
                } else {
                    x
                }
            }
        }
    }
    fn unused() -> Self {
        f64::NAN
    }
}

/// The four Rust operators a float [`VecCmp`] evaluates (`!=` is unordered, every other one ordered).
fn float_compare<T: PartialOrd>(pred: VecCmp, x: T, y: T) -> bool {
    match pred {
        VecCmp::Lt => x < y,
        VecCmp::Le => x <= y,
        VecCmp::Gt => x > y,
        VecCmp::Ge => x >= y,
        VecCmp::Eq => x == y,
        VecCmp::Ne => x != y,
    }
}

/// `VecValue` for the two integer lanes: wrapping arithmetic, signed compares.
macro_rules! int_vec_value {
    ($t:ty, $lane:expr, $from_bits:path) => {
        impl VecValue for $t {
            const LANE: VecLane = $lane;
            fn from_const_bits(bits: u32) -> Self {
                $from_bits(bits)
            }
            fn bin(op: VecBin, x: Self, y: Self) -> Self {
                match op {
                    VecBin::Add => x.wrapping_add(y),
                    VecBin::Sub => x.wrapping_sub(y),
                    VecBin::Mul => x.wrapping_mul(y),
                    // Rejected by `validate` on an integer lane; total so a stray one cannot panic.
                    VecBin::Div => {
                        if y == 0 {
                            0
                        } else {
                            x.wrapping_div(y)
                        }
                    }
                }
            }
            fn fused(a: Self, b: Self, c: Self) -> Self {
                a.wrapping_mul(b).wrapping_add(c)
            }
            fn root(self) -> Self {
                self
            }
            fn negate(self) -> Self {
                self.wrapping_neg()
            }
            fn compare(pred: VecCmp, x: Self, y: Self) -> bool {
                match pred {
                    VecCmp::Lt => x < y,
                    VecCmp::Le => x <= y,
                    VecCmp::Gt => x > y,
                    VecCmp::Ge => x >= y,
                    VecCmp::Eq => x == y,
                    VecCmp::Ne => x != y,
                }
            }
            fn mask(t: bool) -> Self {
                t as $t
            }
            fn is_set(self) -> bool {
                self != 0
            }
            fn identity(op: VecRedOp) -> Self {
                match op {
                    VecRedOp::Add => 0,
                    VecRedOp::Fmax => <$t>::MIN,
                    VecRedOp::Fmin => <$t>::MAX,
                }
            }
            fn fold(op: VecRedOp, acc: Self, x: Self) -> Self {
                match op {
                    VecRedOp::Add => acc.wrapping_add(x),
                    VecRedOp::Fmax => {
                        if acc > x {
                            acc
                        } else {
                            x
                        }
                    }
                    VecRedOp::Fmin => {
                        if acc < x {
                            acc
                        } else {
                            x
                        }
                    }
                }
            }
            fn unused() -> Self {
                0
            }
        }
    };
}
/// An immediate's bit pattern read as a two's-complement `i32` (sign-extended for `i64`).
fn const_bits_i32(bits: u32) -> i32 {
    bits as i32
}
fn const_bits_i64(bits: u32) -> i64 {
    i64::from(bits as i32)
}
int_vec_value!(i32, VecLane::I32, const_bits_i32);
int_vec_value!(i64, VecLane::I64, const_bits_i64);

/// A lane-wise binary arithmetic op. Division is included; the assembler emits `vdivps`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecBin {
    Add,
    Sub,
    Mul,
    Div,
}

/// A lane-wise float comparison producing a boolean mask (1.0 / 0.0 in the reference evaluator;
/// all-ones / all-zero lanes under `vcmpps`). Each predicate is picked to match the corresponding Rust
/// operator on *every* input including NaN, which is what the interpreter oracle evaluates:
/// `Lt`/`Le`/`Gt`/`Ge`/`Eq` are ordered (false on NaN, `vcmpps` `*_OS`/`EQ_OQ`) and `Ne` is
/// **unordered** (true on NaN, `NEQ_UQ`) — exactly as `!=` behaves in Rust.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecCmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// One straight-line SSA operation. Operands are indices into the body prefix (op `i` may only
/// reference ops `< i`); the result of op `i` is "value `i`". `Store` produces no value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VecOp {
    /// Unit-stride vector load of stream `s` at the current lane group.
    Load { stream: u32 },
    /// Broadcast the k-th loop-invariant scalar.
    Splat { scalar: u32 },
    /// Broadcast an immediate f32 constant (its raw bits).
    Const { bits: u32 },
    /// Lane-wise `a ⊕ b`.
    Bin { op: VecBin, a: u32, b: u32 },
    /// Fused `a * b + c`, single-rounded (`vfmadd` / Rust `mul_add`).
    Fma { a: u32, b: u32, c: u32 },
    /// Lane-wise square root.
    Sqrt { a: u32 },
    /// Lane-wise negation (sign-bit flip; bit-exact for ±0 and NaN).
    Neg { a: u32 },
    /// Lane-wise ordered compare → mask.
    Cmp { pred: VecCmp, a: u32, b: u32 },
    /// Blend: `mask ? a : b` (`vblendvps`; the reference picks `a` when the mask lane is non-zero).
    /// PRECONDITION: `mask` must be a `Cmp` result. `vblendvps` keys on the lane's *sign bit* while the
    /// reference keys on non-zero, and only the all-ones / all-zero masks a compare produces make the
    /// two agree (e.g. a `-0.0` mask would diverge).
    Select { mask: u32, a: u32, b: u32 },
    /// Unit-stride vector store of value `val` to stream `s`.
    Store { stream: u32, val: u32 },
}

/// The fold of a reduction kernel. `Fmax`/`Fmin` use the x86 `vmaxps`/`vminps` operand order
/// (`src1` returned only when strictly greater/less), so the reference below and the machine code
/// agree on ties and NaN.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecRedOp {
    Add,
    Fmax,
    Fmin,
}

impl VecRedOp {
    /// The fold identity (initial accumulator): 0 for a sum, ∓∞ for max/min so empty lanes lose.
    pub fn identity(self) -> f32 {
        match self {
            VecRedOp::Add => 0.0,
            VecRedOp::Fmax => f32::NEG_INFINITY,
            VecRedOp::Fmin => f32::INFINITY,
        }
    }
    /// `fold(acc, x)` with the assembler's operand order (`acc` = src1, `x` = src2). `vmaxps`/`vminps`
    /// return src2 unless src1 is strictly greater/less, so `if acc > x { acc } else { x }` matches
    /// them exactly on equal values (incl. ±0) and NaN.
    pub fn fold(self, acc: f32, x: f32) -> f32 {
        match self {
            VecRedOp::Add => acc + x,
            VecRedOp::Fmax => {
                if acc > x {
                    acc
                } else {
                    x
                }
            }
            VecRedOp::Fmin => {
                if acc < x {
                    acc
                } else {
                    x
                }
            }
        }
    }
}

/// The reduction a reduction-kernel performs: fold every lane's value of op `value` (the addend) into
/// an accumulator and return the horizontal fold. Present ⇒ the kernel has no `Store` and returns an
/// f32 (in xmm0) instead of writing memory.
///
/// `fma` fuses an `Add` reduction of a product: when the addend is `X*Y`, `fma = Some((X, Y))` and the
/// per-lane fold is `acc = fma(X, Y, acc)` in *one* rounding (`vfmadd231ps` / `f32::mul_add`) — the
/// product op never enters the recipe, so the fold matches the 128-bit path's fused accumulate, costs
/// one fewer op, and frees a register for more accumulators. `None` ⇒ the plain fold of `value`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VecReduce {
    pub op: VecRedOp,
    pub value: u32,
    pub fma: Option<(u32, u32)>,
}

/// A synthesized 256-bit vector kernel: a symbol the caller `Call`s, the lane type, the stream/scalar
/// counts, the straight-line body, and the unroll. An elementwise kernel's unroll is a request (the
/// assembler may interleave fewer copies to fit 16 YMM registers); a reduction's is the exact
/// accumulator count of the canonical tree (see the module docs). `reduce` distinguishes an
/// *elementwise* kernel (`None` — stores its results) from a *reduction* kernel (`Some` — folds one
/// value across all lanes and returns the horizontal result).
#[derive(Clone, Debug)]
pub struct VecKernel {
    pub name: Symbol,
    pub lane: VecLane,
    pub streams: u32,
    pub scalars: u32,
    pub ops: Vec<VecOp>,
    pub unroll: u32,
    pub reduce: Option<VecReduce>,
}

/// Reusable storage for reference lane evaluation. Each evaluation resets its values and
/// store-forwarding state, so one scratch can serve successive lanes or different kernels.
/// Capacity is retained until this value is dropped; it contains no references to stream memory.
pub struct VecKernelScratch<T = f32> {
    vals: Vec<T>,
    stored: Vec<Option<T>>,
}

impl<T> Default for VecKernelScratch<T> {
    fn default() -> Self {
        VecKernelScratch {
            vals: Vec::new(),
            stored: Vec::new(),
        }
    }
}

impl VecKernel {
    /// Check the recipe's SSA indices and ABI-independent invariants before evaluation or lowering.
    /// Unused recipes are checked too: backends may assemble every registered recipe.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let value_exists = |v: u32, end: usize| {
            (v as usize) < end
                && self
                    .ops
                    .get(v as usize)
                    .is_some_and(|o| !matches!(o, VecOp::Store { .. }))
        };
        if self.unroll == 0 {
            errors.push("unroll must be positive".into());
        }
        for (i, op) in self.ops.iter().enumerate() {
            match *op {
                VecOp::Load { stream } | VecOp::Store { stream, .. } if stream >= self.streams => {
                    errors.push(format!("op {i} stream {stream} is out of range"));
                }
                VecOp::Splat { scalar } if scalar >= self.scalars => {
                    errors.push(format!("op {i} scalar {scalar} is out of range"));
                }
                _ => {}
            }
            let operands = match *op {
                VecOp::Load { .. } | VecOp::Splat { .. } | VecOp::Const { .. } => [None; 3],
                VecOp::Bin { a, b, .. } | VecOp::Cmp { a, b, .. } => [Some(a), Some(b), None],
                VecOp::Fma { a, b, c } => [Some(a), Some(b), Some(c)],
                VecOp::Select { mask, a, b } => [Some(mask), Some(a), Some(b)],
                VecOp::Sqrt { a } | VecOp::Neg { a } => [Some(a), None, None],
                VecOp::Store { val, .. } => [Some(val), None, None],
            };
            for operand in operands.into_iter().flatten() {
                if !value_exists(operand, i) {
                    errors.push(format!("op {i} operand {operand} is not an earlier value"));
                }
            }
            if let VecOp::Select { mask, .. } = *op {
                if !matches!(self.ops.get(mask as usize), Some(VecOp::Cmp { .. })) {
                    errors.push(format!("op {i} select mask must be a compare result"));
                }
            }
            if self.reduce.is_some() && matches!(op, VecOp::Store { .. }) {
                errors.push(format!("reduction contains store at op {i}"));
            }
            // An immediate is an f32 bit pattern; on any other lane it has no single meaning.
            if self.lane != VecLane::F32 && matches!(op, VecOp::Const { .. }) {
                errors.push(format!("op {i} is a Const on a non-f32 lane"));
            }
            // Integer lanes are wrapping two's-complement arithmetic: no division (x86 has no SIMD
            // integer divide, and the scalar semantics of a zero divisor are not lane-wise), no
            // square root and no fused multiply-add.
            if !self.lane.is_float()
                && matches!(
                    op,
                    VecOp::Bin {
                        op: VecBin::Div,
                        ..
                    } | VecOp::Fma { .. }
                        | VecOp::Sqrt { .. }
                )
            {
                errors.push(format!("op {i} needs a float lane"));
            }
        }
        if let Some(red) = self.reduce {
            let operands = match red.fma {
                Some((a, b)) => {
                    if red.op != VecRedOp::Add {
                        errors.push("fused reduction must use Add".into());
                    }
                    if !self.lane.is_float() {
                        errors.push("fused reduction needs a float lane".into());
                    }
                    [Some(a), Some(b)]
                }
                None => [Some(red.value), None],
            };
            if !self.lane.is_float() && red.op != VecRedOp::Add {
                errors.push("an integer reduction must be a sum".into());
            }
            for operand in operands.into_iter().flatten() {
                if !value_exists(operand, self.ops.len()) {
                    errors.push(format!("reduction operand {operand} is not a value"));
                }
            }
        }
        errors
    }

    /// Evaluate the body for one lane and write its stores. `load(stream)` yields that stream's
    /// current element, `scalar(k)` the k-th invariant, `store(stream, v)` writes an output. This is
    /// the reference semantics the AVX2 assembler reproduces per lane and the interpreter marshals
    /// element-by-element — the two must agree bit-for-bit (all arithmetic is in the lane type `T`,
    /// which must model [`Self::lane`]; a float `Fma` is single-rounded via `mul_add`; a float `Neg`
    /// flips the sign bit; each compare follows the matching Rust operator — see [`VecCmp`]).
    pub fn eval_lane<T: VecValue>(
        &self,
        load: impl FnMut(u32) -> T,
        scalar: impl Fn(u32) -> T,
        store: impl FnMut(u32, T),
    ) {
        self.eval_lane_with_scratch(&mut VecKernelScratch::default(), load, scalar, store);
    }

    /// Evaluate a lane with reusable storage. Callback order and numerical semantics match
    /// [`Self::eval_lane`]; retained capacity avoids allocating fresh scratch for each lane.
    pub fn eval_lane_with_scratch<T: VecValue>(
        &self,
        scratch: &mut VecKernelScratch<T>,
        load: impl FnMut(u32) -> T,
        scalar: impl Fn(u32) -> T,
        store: impl FnMut(u32, T),
    ) {
        self.eval_all(scratch, load, scalar, store);
    }

    /// Evaluate every op for one lane, retaining all its values in scratch (the reference used by both
    /// [`Self::eval_lane`] for elementwise kernels and [`Self::eval_reduction`] for reductions).
    fn eval_all<T: VecValue>(
        &self,
        scratch: &mut VecKernelScratch<T>,
        mut load: impl FnMut(u32) -> T,
        scalar: impl Fn(u32) -> T,
        mut store: impl FnMut(u32, T),
    ) {
        debug_assert_eq!(T::LANE, self.lane, "evaluated in the wrong lane type");
        let VecKernelScratch { vals, stored } = scratch;
        vals.clear();
        vals.reserve(self.ops.len());
        // Store-to-load forwarding within the lane: the assembler stores each stream to memory and a
        // later `Load` of that stream re-reads the same address, so a body that writes then reads a
        // stream — e.g. a fused `t[k] = …; o[k] = f(t[k])` — observes the just-written value. Mirror
        // that here so the interpreter oracle matches the machine code bit-for-bit. `self.streams`
        // bounds every stream index.
        // Reset even when the stream count is unchanged: forwarding never crosses a lane boundary.
        stored.clear();
        stored.resize(self.streams as usize, None);
        for op in &self.ops {
            let v = match *op {
                VecOp::Load { stream } => stored[stream as usize].unwrap_or_else(|| load(stream)),
                VecOp::Splat { scalar: k } => scalar(k),
                VecOp::Const { bits } => T::from_const_bits(bits),
                VecOp::Bin { op, a, b } => T::bin(op, vals[a as usize], vals[b as usize]),
                VecOp::Fma { a, b, c } => {
                    T::fused(vals[a as usize], vals[b as usize], vals[c as usize])
                }
                VecOp::Sqrt { a } => vals[a as usize].root(),
                VecOp::Neg { a } => vals[a as usize].negate(),
                VecOp::Cmp { pred, a, b } => {
                    T::mask(T::compare(pred, vals[a as usize], vals[b as usize]))
                }
                VecOp::Select { mask, a, b } => {
                    if vals[mask as usize].is_set() {
                        vals[a as usize]
                    } else {
                        vals[b as usize]
                    }
                }
                VecOp::Store { stream, val } => {
                    let x = vals[val as usize];
                    stored[stream as usize] = Some(x); // forward to later loads of this stream
                    store(stream, x);
                    T::unused() // stores carry no value; never referenced
                }
            };
            vals.push(v);
        }
    }

    /// The number of accumulators `U` of a reduction's canonical tree (module docs): exactly the
    /// recipe's requested unroll. It is deliberately **not** a function of register pressure — the
    /// grouping of a float sum is part of what the program computes, so it may depend only on the
    /// program. A body whose registers cannot hold `U` accumulators beside one bank of its own
    /// values is refused by [`Self::reduction_fits`] instead, and the portable lowering runs the
    /// same tree. When the body fits beside the accumulators but not `U` times over, the assembler
    /// shares register banks between unrolled copies ([`Self::reduction_banks`]); that changes the
    /// instruction schedule, never the grouping.
    pub fn reduction_unroll(&self) -> u32 {
        self.unroll.max(1)
    }

    /// Whether this reduction body fits the AVX2 emitter: one bank of the addend's live registers,
    /// the `U` accumulators and the hoisted broadcasts fit the register file (`P + U + H ≤ 16`). The
    /// vectorizer checks this before committing to the 256-bit reduction kernel.
    pub fn reduction_fits(&self) -> bool {
        // Saturating: `unroll` is recipe data, and an absurd one must read as "does not fit".
        let u = self.reduction_unroll();
        self.pressure()
            .is_some_and(|p| p.per_group.saturating_add(u).saturating_add(p.hoist()) <= VEC_NREG)
    }

    /// How many independent register banks the assembler gives a reduction's `U` unrolled copies:
    /// as many as fit beside the accumulators and broadcasts, at most `U`, at least 1. Copy `u` uses
    /// bank `u mod banks`. Out-of-order execution renames the shared architectural registers, so the
    /// copies' dependence chains stay independent — only the accumulators need distinct registers.
    pub fn reduction_banks(&self) -> u32 {
        let Some(p) = self.pressure() else { return 1 };
        let u = self.reduction_unroll();
        let avail = VEC_NREG.saturating_sub(p.hoist().saturating_add(u));
        (avail / p.per_group.max(1)).clamp(1, u)
    }

    /// Evaluate a reduction kernel over `n` elements (a multiple of [`VecLane::lanes`]) exactly as
    /// the AVX2 assembler does, returning the horizontal fold — steps 1-5 of the canonical tree in
    /// the module docs. `load(stream, element)` reads a stream's element; the caller combines the
    /// seed and folds the `[n, end)` scalar tail afterward (steps 6-7). The reassociated form is the
    /// oracle (both backends run it).
    pub fn eval_reduction<T: VecValue>(
        &self,
        n: usize,
        mut load: impl FnMut(u32, usize) -> T,
        scalar: impl Fn(u32) -> T,
    ) -> T {
        let red = self
            .reduce
            .expect("eval_reduction on a non-reduction kernel");
        let lanes = self.lane.lanes() as usize;
        let u = self.reduction_unroll() as usize;
        // Fold one lane's addend into its accumulator: a fused `acc = fma(X, Y, acc)` in a single
        // rounding (bit-identical to the assembler's `vfmadd231ps`/`pd`) when `fma` is set, else the
        // plain fold of the addend value.
        let addend = |acc: T, vals: &[T]| -> T {
            match red.fma {
                Some((x, y)) => T::fused(vals[x as usize], vals[y as usize], acc),
                None => T::fold(red.op, acc, vals[red.value as usize]),
            }
        };
        // acc[copy * lanes + lane]. The addend of element `e` is op `red.value` evaluated with the
        // load closure reading element `e`.
        let mut acc = vec![T::identity(red.op); u * lanes];
        let mut scratch = VecKernelScratch::<T>::default();
        let mut i = 0usize;
        // Main loop: U independent accumulators, computed and folded one copy at a time.
        let ustep = u * lanes;
        while i + ustep <= n {
            for cu in 0..u {
                for lane in 0..lanes {
                    let e = i + cu * lanes + lane;
                    self.eval_all(&mut scratch, |s| load(s, e), &scalar, |_, _| {});
                    let idx = cu * lanes + lane;
                    acc[idx] = addend(acc[idx], &scratch.vals);
                }
            }
            i += ustep;
        }
        // Cleanup: remaining full lane groups fold into accumulator 0.
        while i + lanes <= n {
            for (lane, a) in acc.iter_mut().take(lanes).enumerate() {
                let e = i + lane;
                self.eval_all(&mut scratch, |s| load(s, e), &scalar, |_, _| {});
                *a = addend(*a, &scratch.vals);
            }
            i += lanes;
        }
        // Combine the U accumulators into copy 0, lane-wise.
        for cu in 1..u {
            for lane in 0..lanes {
                acc[lane] = T::fold(red.op, acc[lane], acc[cu * lanes + lane]);
            }
        }
        // Horizontal fold of the lanes, in order 0..lanes.
        let mut r = acc[0];
        for &a in &acc[1..lanes] {
            r = T::fold(red.op, r, a);
        }
        r
    }

    /// The register pressure of this body, or `None` if it is outside the AVX2 emitter's coverage
    /// (0 or more than [`VEC_MAX_STREAMS`] streams, a `Const` the vectorizer should have folded into
    /// `scalars`, a 64-bit integer multiply — AVX2 has no `vpmullq` — or a compare whose result is
    /// used as anything but a `Select` mask). Single-sourced here so the Cranelift assembler's
    /// `plan_registers` and the vectorizer's feasibility gate compute the *same* P/H and can never
    /// disagree about whether a body fits — a mismatch would turn a fallback-eligible body into a
    /// hard `assemble_kernel` error. Pure over the op list.
    pub fn pressure(&self) -> Option<VecPressure> {
        if !self.validate().is_empty() {
            return None;
        }
        if self.streams == 0 || self.streams > VEC_MAX_STREAMS {
            return None;
        }
        if self.ops.iter().any(|o| matches!(o, VecOp::Const { .. })) {
            return None;
        }
        if self.lane == VecLane::I64
            && self.ops.iter().any(|o| {
                matches!(
                    o,
                    VecOp::Bin {
                        op: VecBin::Mul,
                        ..
                    }
                )
            })
        {
            return None;
        }
        // A compare's machine value is an all-ones/all-zero lane mask (and, on an integer lane, the
        // emitter may compute the *complement* and swap the blend's operands), while the reference
        // value is 1/0: the two agree only through a `Select` mask. The vectorizer never builds any
        // other use; refuse one rather than assemble a divergence.
        let is_cmp = |v: u32| matches!(self.ops.get(v as usize), Some(VecOp::Cmp { .. }));
        for o in &self.ops {
            let misuse = match *o {
                VecOp::Select { a, b, .. } => is_cmp(a) || is_cmp(b),
                _ => op_operands(o).into_iter().any(is_cmp),
            };
            if misuse {
                return None;
            }
        }
        if let Some(red) = self.reduce {
            let folded = match red.fma {
                Some((x, y)) => is_cmp(x) || is_cmp(y),
                None => is_cmp(red.value),
            };
            if folded {
                return None;
            }
        }
        // Distinct invariant scalars referenced by `Splat`, first-seen order (each hoisted to a reg).
        let mut hoist_scalars: Vec<u32> = Vec::new();
        for o in &self.ops {
            if let VecOp::Splat { scalar } = *o {
                if !hoist_scalars.contains(&scalar) {
                    hoist_scalars.push(scalar);
                }
            }
        }
        // A float negation is a sign-bit `xor` against a hoisted mask; an integer one is `0 - x`
        // built in its own result register, which needs no fixed register.
        let needs_signmask =
            self.lane.is_float() && self.ops.iter().any(|o| matches!(o, VecOp::Neg { .. }));

        // last_use[i]: last op index that references value i.
        let mut last_use = vec![0usize; self.ops.len()];
        for (i, o) in self.ops.iter().enumerate() {
            for v in op_operands(o) {
                last_use[v as usize] = i;
            }
        }
        // A reduction's fold reads its addend operand(s) *after* the last op (the fold is not itself an
        // op), so those registers must survive the whole body. Mark their last use one past the end so
        // neither the peak-liveness scan below nor the emitter's `free_after` reclaims them early —
        // essential for the fused `fma` operands, which have no consuming op at all.
        if let Some(red) = self.reduce {
            let end = self.ops.len();
            match red.fma {
                Some((x, y)) => {
                    last_use[x as usize] = end;
                    last_use[y as usize] = end;
                }
                None => last_use[red.value as usize] = end,
            }
        }
        // P = peak simultaneously-live body registers: allocate a result before freeing this op's
        // dead operands (mirrors the emitter's ordering, so P is an exact upper bound).
        let mut live: Vec<usize> = Vec::new();
        let mut per_group: usize = 0;
        for (i, o) in self.ops.iter().enumerate() {
            if op_produces_value(o) {
                live.push(i);
                per_group = per_group.max(live.len());
            }
            for v in op_operands(o) {
                // Hoisted (Splat/Const) values never occupy a body register.
                let hoisted = matches!(
                    self.ops[v as usize],
                    VecOp::Splat { .. } | VecOp::Const { .. }
                );
                if !hoisted && last_use[v as usize] == i {
                    live.retain(|&x| x != v as usize);
                }
            }
        }
        Some(VecPressure {
            hoist_scalars,
            needs_signmask,
            last_use,
            per_group: per_group.max(1) as u32,
        })
    }
}

/// Whether **this host** can execute an assembled [`VecKernel`]: the AVX2 + FMA3 instructions the
/// Cranelift assembler encodes, delivered under the Win64 argument-register mapping (rcx/rdx/r8)
/// that same emitter hardcodes with no in-kernel dispatch.
///
/// It lives here — below both users — because two crates have to answer it *identically*:
///
/// * `wukong_codegen_cranelift::avx2::host_supports_kernels` (a thin wrapper over this), which makes
///   `assemble_kernel` refuse rather than emit bytes that would `#UD` or dereference whatever rcx
///   happened to hold; and
/// * the general vectorizer in `wukong_mir_build`, which must therefore not *produce* a recipe the
///   backend is going to refuse.
///
/// Only the first half existed until this predicate was shared, so on every non-Windows host
/// `mir_build` attached a recipe that `populate_module` then hard-failed on
/// (`avx2 assemble vec kernel 0.0: …`), taking the whole compile with it. Declining the recipe up
/// front is safe and result-identical: the 128-bit CLIF strips are the documented fallback — the
/// very same one `WUKONG_P4_NO_256=1` forces, pinned by `p4_kill_switch_is_result_identical`.
///
/// This is a *host* test, not a target test, and both consumers want it that way: the kernels are
/// assembled into the JIT module and into `--emit=obj` alike, and that object is host-ISA- and
/// host-ABI-specific either way.
///
/// LANDMINE: `std::is_x86_feature_detected!` only exists on x86/x86-64, so the other arm must be a
/// plain `false` — do not "simplify" the two `cfg` blocks into one expression. The macro caches its
/// CPUID probe, so calling this once per vectorizable loop is cheap.
pub fn host_supports_vec_kernels() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "windows")))]
    {
        false
    }
}

/// Architectural YMM registers a kernel may use.
pub const VEC_NREG: u32 = 16;
/// Bytes per YMM register — the vector width every lane type divides.
pub const VEC_BYTES: u32 = 32;
/// f32 lanes per 256-bit YMM register (`VecLane::F32.lanes()`).
pub const VEC_LANES: u32 = 8;
/// The maximum number of distinct unit-stride streams the AVX2 emitter keeps in base-pointer GPRs:
/// the volatile r9/r10/r11/rdx plus the callee-saved rsi/rdi/rbx/r12-r15 it saves and restores
/// (rbp, rsp, and the rcx/r8/rax it uses as counter, bound and scratch are never stream bases).
pub const VEC_MAX_STREAMS: u32 = 11;

/// A backend-agnostic register-pressure summary of a [`VecKernel`] body, consumed by the Cranelift
/// assembler (to build a concrete register plan) and by the vectorizer (to check the body fits before
/// committing to the 256-bit path).
pub struct VecPressure {
    /// Distinct invariant scalars referenced by `Splat`, first-seen order (each hoisted to a reg).
    pub hoist_scalars: Vec<u32>,
    /// The body negates, so a sign-mask constant occupies one more fixed register.
    pub needs_signmask: bool,
    /// `last_use[i]` = the last op index that references value `i`.
    pub last_use: Vec<usize>,
    /// P: peak simultaneously-live body registers (results that are neither hoisted nor stores).
    pub per_group: u32,
}

impl VecPressure {
    /// H: fixed registers taken by hoisted broadcasts (invariant scalars + an optional sign-mask).
    pub fn hoist(&self) -> u32 {
        self.hoist_scalars.len() as u32 + self.needs_signmask as u32
    }
    /// Whether one group plus the hoisted broadcasts fit the register file (unroll ≥ 1 feasible).
    pub fn fits(&self) -> bool {
        self.per_group + self.hoist() <= VEC_NREG
    }
}

/// The operand value-indices an op reads (for liveness). `Load`/`Splat`/`Const` read nothing.
pub fn op_operands(op: &VecOp) -> Vec<u32> {
    match *op {
        VecOp::Load { .. } | VecOp::Splat { .. } | VecOp::Const { .. } => vec![],
        VecOp::Bin { a, b, .. } | VecOp::Cmp { a, b, .. } => vec![a, b],
        VecOp::Fma { a, b, c } => vec![a, b, c],
        VecOp::Select { mask, a, b } => vec![mask, a, b],
        VecOp::Sqrt { a } | VecOp::Neg { a } => vec![a],
        VecOp::Store { val, .. } => vec![val],
    }
}

/// Whether an op produces a body value that needs a (non-hoisted) register.
pub fn op_produces_value(op: &VecOp) -> bool {
    !matches!(
        op,
        VecOp::Store { .. } | VecOp::Splat { .. } | VecOp::Const { .. }
    )
}
