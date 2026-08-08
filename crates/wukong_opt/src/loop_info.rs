//! Loop analysis: natural loops, induction variables, affine memory accesses, and a conservative
//! loop-carried dependence test.
//!
//! This module is **pure analysis** — it never mutates a [`Function`]. It exists so that later
//! passes (canonicalization, vectorization, unrolling) can ask structural questions about a loop
//! instead of pattern-matching the AST shape that produced it, which is what the `mir_build`
//! kernel recognizers do today. The whole point is that a `while`, a `for i in 0..n`, a hoisted
//! temporary and a user-written index expression all look the same here.
//!
//! # What it computes
//!
//! * [`LoopForest`] — every natural loop of a function (back edges off the dominator tree), each
//!   with its header, latches, block set, entering edges, preheader, exit edges, and its place in
//!   the nesting forest. [`LoopForest::innermost_first`] iterates children before parents.
//! * [`InductionVar`] — *basic* IVs: a header block parameter whose latch argument is
//!   `param + step`. Wukong's MIR is block-parameter SSA, so an IV is a block param, never a phi
//!   node. [`DerivedIv`] records values that are affine functions of the primary basic IV.
//! * [`TripCount`] — where provable, the iteration count, from the header's exit test.
//! * [`MemAccess`] — for every load/store in the loop, whether its address is
//!   `base + stride·iv + offset` with a loop-invariant base and offset ([`AddrForm`]).
//! * [`DepSummary`] — a conservative answer to "may one iteration write a location another
//!   iteration touches?", plus the [`Carried`] classification of every header parameter
//!   (induction variable / reduction / invariant / true recurrence).
//!
//! # Soundness posture
//!
//! Every "unknown" answer is reported as the pessimistic one: an unrecognized address is
//! [`AddrForm::Unknown`], an unrecognized loop-carried value is [`Carried::Recurrence`], and any
//! call in the loop forces [`MemDep::Carried`]. The two places where the analysis takes a stated
//! assumption rather than proving something are documented on [`AffineExpr`] (integer arithmetic
//! does not wrap) and [`MemDep::IndependentIfBasesStable`] (a reloaded base pointer is stable).
//! Nothing else is assumed.

use std::fmt::Write as _;

use wukong_mir::{
    BinOp, BlockId, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId,
};
use wukong_span::Interner;

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::{cfg, dom, each_op_use};

/// Sentinel for "no immediate dominator" (an unreachable block).
const NO_IDOM: u32 = u32::MAX;

/// An index into [`LoopForest::loops`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LoopId(pub u32);

impl std::fmt::Debug for LoopId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}", self.0)
    }
}

/// One natural loop: a header `h` plus every block that can reach a back edge `n -> h` without
/// leaving through `h`. Loops that share a header (several back edges) are merged into one.
#[derive(Clone, Debug)]
pub struct NaturalLoop {
    pub id: LoopId,
    /// The single entry point. Dominates every block in [`Self::blocks`].
    pub header: BlockId,
    /// Blocks inside the loop with a back edge to the header, ascending. Non-empty.
    pub latches: Vec<BlockId>,
    /// Every block of the loop, header included, ascending.
    pub blocks: Vec<BlockId>,
    /// Predecessors of the header from *outside* the loop, ascending.
    pub entering: Vec<BlockId>,
    /// The unique preheader if the loop has one: a single entering block whose terminator is an
    /// unconditional branch to the header. (It then necessarily dominates the header.) `None`
    /// means the loop is entered from several places, or through a conditional branch.
    pub preheader: Option<BlockId>,
    /// Edges that leave the loop, as `(block inside, block outside)`, ascending.
    pub exits: Vec<(BlockId, BlockId)>,
    /// Blocks inside the loop that leave it without an edge — a `return` or an `unreachable` in the
    /// body. They are invisible to [`Self::exits`] (those terminators have no successors), so any
    /// consumer that reasons about "how many times the body runs" has to check this list too: a
    /// loop with an early `return` runs its exit test the same number of times but its body fewer.
    /// [`TripCount`] is only ever computed when this is empty.
    pub abnormal_exits: Vec<BlockId>,
    /// The innermost loop that strictly contains this one.
    pub parent: Option<LoopId>,
    /// Loops immediately nested in this one, ascending by header.
    pub children: Vec<LoopId>,
    /// Nesting depth; an outermost loop is 0.
    pub depth: u32,

    // ---- value analyses (filled in by the IV / access / dependence stages) ----
    /// Basic induction variables carried by the header's block parameters.
    pub ivs: Vec<InductionVar>,
    /// Index into [`Self::ivs`] of the IV the exit test compares — the one every affine
    /// expression in this loop is expressed against.
    pub primary_iv: Option<usize>,
    /// How many times the body runs, where provable.
    pub trip: TripCount,
    /// Values defined in the loop that are affine functions of the primary IV.
    pub derived: Vec<DerivedIv>,
    /// Reductions found among the header parameters.
    pub reductions: Vec<Reduction>,
    /// One entry per header block parameter, in parameter order.
    pub carried: Vec<CarriedValue>,
    /// Every load/store in the loop, in block-then-instruction order.
    pub accesses: Vec<MemAccess>,
    /// The loop-carried memory dependence verdict.
    pub dep: DepSummary,
}

impl NaturalLoop {
    /// The primary induction variable, if the exit test identified one.
    pub fn primary(&self) -> Option<&InductionVar> {
        self.primary_iv.map(|i| &self.ivs[i])
    }

    /// Nothing in this loop forbids evaluating several iterations at once: there is no
    /// loop-carried memory dependence, and every loop-carried *register* value is an induction
    /// variable, a loop-invariant copy, or a reduction.
    ///
    /// This is **not** a licence to reassociate: a float `Add`/`Mul` reduction still has to keep
    /// its accumulate serial. Check [`Reduction::reassociable`] per reduction.
    ///
    /// Nor is it, on its own, a licence to *run*: [`MemDep::IndependentIfBasesDisjoint`] carries a
    /// proof obligation the consumer has to discharge with a runtime pointer-range check. Read that
    /// variant before acting on a `true` here.
    pub fn is_vectorizable_shape(&self) -> bool {
        matches!(
            self.dep.memory,
            MemDep::Independent
                | MemDep::IndependentIfBasesStable
                | MemDep::IndependentIfBasesDisjoint
        ) && self
            .carried
            .iter()
            .all(|c| !matches!(c.kind, Carried::Recurrence))
    }
}

/// Every natural loop of one function, plus a block -> innermost-loop map.
#[derive(Clone, Debug, Default)]
pub struct LoopForest {
    /// All loops, ascending by header block id.
    pub loops: Vec<NaturalLoop>,
    /// For each block id, the innermost loop containing it.
    innermost: Vec<Option<LoopId>>,
}

impl LoopForest {
    pub fn is_empty(&self) -> bool {
        self.loops.is_empty()
    }

    pub fn get(&self, id: LoopId) -> &NaturalLoop {
        &self.loops[id.0 as usize]
    }

    /// The innermost loop containing `b`, if any.
    pub fn innermost_of(&self, b: BlockId) -> Option<LoopId> {
        *self.innermost.get(b.0 as usize)?
    }

    /// The outermost loops (those with no parent), ascending by header.
    pub fn roots(&self) -> impl Iterator<Item = &NaturalLoop> {
        self.loops.iter().filter(|l| l.parent.is_none())
    }

    /// Every loop, children strictly before their parents. Deterministic: deeper loops first,
    /// ties broken by header id. This is the order a transform wants — rewriting an inner loop
    /// cannot invalidate an outer loop's block set, but not the other way round.
    pub fn innermost_first(&self) -> Vec<LoopId> {
        let mut ids: Vec<LoopId> = self.loops.iter().map(|l| l.id).collect();
        ids.sort_by_key(|id| {
            let l = self.get(*id);
            (std::cmp::Reverse(l.depth), l.header.0)
        });
        ids
    }
}

// ---------------------------------------------------------------------------------------------
// Induction variables and trip counts
// ---------------------------------------------------------------------------------------------

/// The per-iteration increment of a basic induction variable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IvStep {
    /// `i_next = i + k` with a literal `k` (negative for a countdown).
    Const(i64),
    /// `i_next = i + s` with a non-constant `s` that is defined **outside** the loop, so a
    /// consumer can read it in the preheader.
    Invariant(ValueId),
}

impl IvStep {
    pub fn as_const(self) -> Option<i64> {
        match self {
            IvStep::Const(k) => Some(k),
            IvStep::Invariant(_) => None,
        }
    }
}

/// A *basic* induction variable: a header block parameter `i` whose latch argument is `i + step`.
///
/// Wukong's MIR merges values with block parameters, not phi nodes, so an IV is a parameter of the
/// loop header and its "phi operands" are the branch arguments each predecessor passes.
#[derive(Clone, Debug)]
pub struct InductionVar {
    /// The header block parameter carrying the IV.
    pub value: ValueId,
    /// Its position in the header's parameter list.
    pub param_index: usize,
    /// The value passed in on the entering edge(s). `None` when the loop has several entering
    /// edges that disagree.
    pub start: Option<ValueId>,
    /// The value the latch passes back — the `i + step` instruction's result.
    pub next: ValueId,
    pub step: IvStep,
    pub ty: MirType,
}

/// A value defined inside the loop that is an affine function of the primary IV.
#[derive(Clone, Debug)]
pub struct DerivedIv {
    pub value: ValueId,
    pub expr: AffineExpr,
}

/// How many times a loop's body runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TripCount {
    /// Exactly `n` iterations, known at compile time.
    Const(u64),
    /// `start`, `end` and `step` are known but the count is not: the loop runs while
    /// `iv < end` (or `iv <= end` when `inclusive`), starting at `start` and stepping by `step`.
    /// `signed` reflects the comparison predicate. `end` is reported as written — an inclusive
    /// bound is *not* normalized to `end + 1`, because that addition can overflow.
    Affine {
        start: ValueId,
        end: ValueId,
        step: i64,
        inclusive: bool,
        signed: bool,
    },
    /// Not provable.
    Unknown,
}

impl TripCount {
    pub fn as_const(&self) -> Option<u64> {
        match self {
            TripCount::Const(n) => Some(*n),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Affine index expressions
// ---------------------------------------------------------------------------------------------

/// The multiplier on the loop's primary induction variable in an [`AffineExpr`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Coeff {
    /// The expression does not mention the IV: it is loop-invariant.
    #[default]
    Zero,
    /// A compile-time constant stride.
    Const(i64),
    /// `value * mult` — a loop-invariant, non-constant stride, e.g. the row pitch `n` in
    /// `a[i*n + k]` viewed from the `i` loop.
    Sym(ValueId, i64),
}

impl Coeff {
    pub fn is_zero(self) -> bool {
        matches!(self, Coeff::Zero) || matches!(self, Coeff::Const(0))
    }

    /// The stride as a compile-time constant, if it is one. `Coeff::Zero` is stride 0.
    pub fn as_const(self) -> Option<i64> {
        match self {
            Coeff::Zero => Some(0),
            Coeff::Const(k) => Some(k),
            Coeff::Sym(..) => None,
        }
    }

    fn scale(self, k: i64) -> Option<Coeff> {
        Some(match self {
            Coeff::Zero => Coeff::Zero,
            Coeff::Const(c) => Coeff::Const(c.checked_mul(k)?),
            Coeff::Sym(v, m) => Coeff::Sym(v, m.checked_mul(k)?),
        })
    }

    fn add(self, other: Coeff) -> Option<Coeff> {
        Some(match (self, other) {
            (Coeff::Zero, x) | (x, Coeff::Zero) => x,
            (Coeff::Const(a), Coeff::Const(b)) => Coeff::Const(a.checked_add(b)?),
            (Coeff::Sym(v, a), Coeff::Sym(w, b)) if v == w => Coeff::Sym(v, a.checked_add(b)?),
            // `n·i + 3·i` with `n` symbolic is not representable in this form.
            _ => return None,
        })
    }
}

/// An integer expression in the canonical form `coeff · iv + Σ (value · mult) + konst`, where
/// `iv` is the loop's primary basic induction variable and every `value` is loop-invariant.
///
/// **Stated assumption**: the arithmetic is evaluated in `i64` and is assumed not to wrap in the
/// source expression's own width. `sext` is therefore treated as transparent (`sext x == x` when
/// `x` did not overflow), while `zext`, `trunc` and `bitcast` are not — a `zext` of a negative
/// narrow value is a large positive one, which would silently invalidate the distance reasoning in
/// [`DepSummary`]. Any expression whose i64 arithmetic *does* overflow is rejected outright
/// (`checked_*` throughout), so this assumption only ever covers wraparound in the narrower source
/// type, exactly the `nsw` assumption every vectorizer makes about an index computation.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct AffineExpr {
    pub coeff: Coeff,
    /// Loop-invariant addends `(value, multiplier)`, ascending by value id, deduped, never with a
    /// zero multiplier.
    pub terms: Vec<(ValueId, i64)>,
    pub konst: i64,
}

impl AffineExpr {
    fn constant(k: i64) -> AffineExpr {
        AffineExpr {
            coeff: Coeff::Zero,
            terms: Vec::new(),
            konst: k,
        }
    }

    /// The opaque loop-invariant value `v`, as the one-term expression `v · 1`.
    fn symbol(v: ValueId) -> AffineExpr {
        AffineExpr {
            coeff: Coeff::Zero,
            terms: vec![(v, 1)],
            konst: 0,
        }
    }

    /// `1 · iv`.
    fn iv() -> AffineExpr {
        AffineExpr {
            coeff: Coeff::Const(1),
            terms: Vec::new(),
            konst: 0,
        }
    }

    /// Does this expression not mention the induction variable?
    pub fn is_invariant(&self) -> bool {
        self.coeff.is_zero()
    }

    /// The whole expression as a compile-time constant, if it is one.
    pub fn as_const(&self) -> Option<i64> {
        (self.coeff.is_zero() && self.terms.is_empty()).then_some(self.konst)
    }

    /// Are these two expressions equal apart from their constant term? If so, the difference
    /// `self - other` is exactly `self.konst - other.konst`.
    #[allow(dead_code)] // used by the dependence stage
    fn same_shape(&self, other: &AffineExpr) -> bool {
        self.coeff == other.coeff && self.terms == other.terms
    }

    fn normalize(mut self) -> AffineExpr {
        self.terms.sort_unstable_by_key(|(v, _)| v.0);
        let mut merged: Vec<(ValueId, i64)> = Vec::with_capacity(self.terms.len());
        for (v, m) in self.terms {
            match merged.last_mut() {
                Some((pv, pm)) if *pv == v => *pm += m,
                _ => merged.push((v, m)),
            }
        }
        merged.retain(|(_, m)| *m != 0);
        self.terms = merged;
        self
    }

    fn add(&self, other: &AffineExpr) -> Option<AffineExpr> {
        let mut terms = self.terms.clone();
        terms.extend_from_slice(&other.terms);
        Some(
            AffineExpr {
                coeff: self.coeff.add(other.coeff)?,
                terms,
                konst: self.konst.checked_add(other.konst)?,
            }
            .normalize(),
        )
    }

    fn scale(&self, k: i64) -> Option<AffineExpr> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for (v, m) in &self.terms {
            terms.push((*v, m.checked_mul(k)?));
        }
        Some(
            AffineExpr {
                coeff: self.coeff.scale(k)?,
                terms,
                konst: self.konst.checked_mul(k)?,
            }
            .normalize(),
        )
    }

    fn neg(&self) -> Option<AffineExpr> {
        self.scale(-1)
    }

    fn sub(&self, other: &AffineExpr) -> Option<AffineExpr> {
        self.add(&other.neg()?)
    }

    /// The bare loop-invariant symbol this expression is, if it is exactly `v · 1`.
    fn as_symbol(&self) -> Option<ValueId> {
        match (self.coeff, self.terms.as_slice(), self.konst) {
            (Coeff::Zero, [(v, 1)], 0) => Some(*v),
            _ => None,
        }
    }

    /// Multiply two affine expressions. Only the cases that stay linear in the IV are accepted:
    /// scaling by a constant, and multiplying a bare loop-invariant symbol by `c·iv + k` (which is
    /// how a row pitch `a[i*n + j]` shows up).
    fn mul(&self, other: &AffineExpr) -> Option<AffineExpr> {
        if let Some(k) = other.as_const() {
            return self.scale(k);
        }
        if let Some(k) = self.as_const() {
            return other.scale(k);
        }
        for (sym_side, lin) in [(self, other), (other, self)] {
            let Some(v) = sym_side.as_symbol() else {
                continue;
            };
            if !lin.terms.is_empty() {
                continue; // v · (w + …) — a product of two invariants, not representable
            }
            let coeff = match lin.coeff {
                Coeff::Zero => Coeff::Zero,
                Coeff::Const(k) => Coeff::Sym(v, k),
                Coeff::Sym(..) => continue, // v · (w·iv) is quadratic in invariants
            };
            let terms = if lin.konst == 0 {
                Vec::new()
            } else {
                vec![(v, lin.konst)]
            };
            return Some(
                AffineExpr {
                    coeff,
                    terms,
                    konst: 0,
                }
                .normalize(),
            );
        }
        None
    }

    #[allow(dead_code)] // used by the debug dump once accesses are analyzed
    fn display(&self) -> String {
        let mut s = String::new();
        match self.coeff {
            Coeff::Zero => {}
            Coeff::Const(k) => {
                let _ = write!(s, "{k}*iv");
            }
            Coeff::Sym(v, 1) => {
                let _ = write!(s, "v{}*iv", v.0);
            }
            Coeff::Sym(v, m) => {
                let _ = write!(s, "{m}*v{}*iv", v.0);
            }
        }
        for (v, m) in &self.terms {
            if !s.is_empty() {
                s.push_str(" + ");
            }
            if *m == 1 {
                let _ = write!(s, "v{}", v.0);
            } else {
                let _ = write!(s, "{m}*v{}", v.0);
            }
        }
        if self.konst != 0 || s.is_empty() {
            if !s.is_empty() {
                s.push_str(" + ");
            }
            let _ = write!(s, "{}", self.konst);
        }
        s
    }
}

// ---------------------------------------------------------------------------------------------
// Memory accesses
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccessKind {
    Load,
    Store,
}

/// The object a loop access is relative to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemBase {
    /// A pointer value defined outside the loop — an `alloca`, a function parameter, or anything
    /// else the loop does not recompute.
    Outer(ValueId),
    /// `value = load ptr addr` executed *inside* the loop, with `addr` loop-invariant. This is how
    /// a `[]T` slice parameter reads its data pointer today: the base is re-loaded from the fat
    /// pointer on every access. Two of these denote the same pointer when their `addr`s are equal
    /// — see [`MemDep::IndependentIfBasesStable`] for the assumption that makes explicit.
    ReloadedOuter { addr: ValueId, value: ValueId },
}

/// The address of one load/store, decomposed relative to the loop's primary IV.
#[derive(Clone, Debug)]
pub enum AddrForm {
    /// `base` indexed by `index` elements of type `elem` (the `Gep`'s element type, which is the
    /// unit `index` counts in — not necessarily the type transferred).
    Affine {
        base: MemBase,
        index: AffineExpr,
        elem: MirType,
    },
    /// The pointer itself is loop-invariant: the same address every iteration.
    Invariant(MemBase),
    /// Not understood. Treated as "may touch anything".
    Unknown,
}

impl MemBase {
    /// Do these two bases denote the *same* object? Two `ReloadedOuter`s are the same object when
    /// they re-load the same invariant address — their `value`s differ (they are two separate load
    /// instructions) but the pointer they yield is the same, which is exactly the assumption
    /// [`MemDep::IndependentIfBasesStable`] names.
    pub fn same_object(self, other: MemBase) -> bool {
        match (self, other) {
            (MemBase::Outer(a), MemBase::Outer(b)) => a == b,
            (MemBase::ReloadedOuter { addr: a, .. }, MemBase::ReloadedOuter { addr: b, .. }) => {
                a == b
            }
            _ => false,
        }
    }

    /// The value a consumer can read in the preheader to get this base's pointer: the pointer
    /// itself for an [`MemBase::Outer`], or the *address it is re-loaded from* for a
    /// [`MemBase::ReloadedOuter`] (which has to be re-loaded there, since the load instruction
    /// itself lives inside the loop and is not available outside it).
    pub fn outer_handle(self) -> ValueId {
        match self {
            MemBase::Outer(v) => v,
            MemBase::ReloadedOuter { addr, .. } => addr,
        }
    }
}

impl AddrForm {
    /// The element stride per iteration, when it is a compile-time constant. `Some(1)` is the
    /// unit-stride case a vectorizer wants; `Some(0)` means the address never moves.
    pub fn const_stride(&self) -> Option<i64> {
        match self {
            AddrForm::Affine { index, .. } => index.coeff.as_const(),
            AddrForm::Invariant(_) => Some(0),
            AddrForm::Unknown => None,
        }
    }

    pub fn base(&self) -> Option<MemBase> {
        match self {
            AddrForm::Affine { base, .. } => Some(*base),
            AddrForm::Invariant(b) => Some(*b),
            AddrForm::Unknown => None,
        }
    }
}

/// One load or store inside a loop.
#[derive(Clone, Debug)]
pub struct MemAccess {
    pub block: BlockId,
    /// Index of the instruction within its block's `insts`.
    pub inst: usize,
    pub kind: AccessKind,
    /// The pointer operand.
    pub ptr: ValueId,
    /// The type transferred (the `Load`'s result type / the stored value's type).
    pub value_ty: MirType,
    pub addr: AddrForm,
}

// ---------------------------------------------------------------------------------------------
// Loop-carried values
// ---------------------------------------------------------------------------------------------

/// The operator a reduction accumulates with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RedKind {
    Add,
    Mul,
    And,
    Or,
    Xor,
    FAdd,
    FMul,
    /// `acc = acc - x`, integer. The accumulator is always the LEFT operand — `x - acc` is not a
    /// reduction, it is a recurrence that negates its own history.
    Sub,
    /// `acc = acc - x`, float.
    FSub,
    SMax,
    SMin,
    UMax,
    UMin,
    FMax,
    FMin,
}

impl RedKind {
    /// May the partial results be reassociated — i.e. may a vectorizer keep N independent
    /// accumulators and fold them at the end?
    ///
    /// Integer `add`/`mul`/bitwise and integer min/max are associative, so yes.
    ///
    /// Float `add`/`mul` are **not**: `(a+b)+c != a+(b+c)` in IEEE-754, and Wukong's interpreter is
    /// the language spec, so reassociating one changes the program's answer and breaks the
    /// differential gate. A vectorizer may still widen the *body* that feeds such a reduction and
    /// keep the accumulate serial — exactly what `gcc -O3` does without `-ffast-math`.
    ///
    /// Float min/max are **also not** associative here, which is easy to get wrong. Wukong spells
    /// them `select(a > b, a, b)`, i.e. "`b` unless `a > b`", and every ordered compare against a
    /// NaN is false. With `a = 1, b = NaN, c = 0`: folding left gives
    /// `max(max(1, NaN), 0) = max(NaN, 0) = 0`, folding right gives
    /// `max(1, max(NaN, 0)) = max(1, 0) = 1`. Different answers, so lane-wise partial maxima may
    /// not be recombined in a different order than the source loop takes them.
    /// Subtraction is **never** reassociable, at any element type. `((a - x) - y)` is
    /// `a - (x + y)`, not `a - (x - y)`, so lane-parallel partials would need a *different*
    /// operator to combine them and a different identity to start from. Its accumulate stays
    /// serial, which for the float case is what it would have been anyway.
    pub fn reassociable(self) -> bool {
        !matches!(
            self,
            RedKind::FAdd
                | RedKind::FMul
                | RedKind::FMax
                | RedKind::FMin
                | RedKind::Sub
                | RedKind::FSub
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            RedKind::Add => "add",
            RedKind::Mul => "mul",
            RedKind::And => "and",
            RedKind::Or => "or",
            RedKind::Xor => "xor",
            RedKind::FAdd => "fadd",
            RedKind::FMul => "fmul",
            RedKind::Sub => "sub",
            RedKind::FSub => "fsub",
            RedKind::SMax => "smax",
            RedKind::SMin => "smin",
            RedKind::UMax => "umax",
            RedKind::UMin => "umin",
            RedKind::FMax => "fmax",
            RedKind::FMin => "fmin",
        }
    }
}

/// An accumulator carried across iterations: `acc = acc ⊕ x`, with `acc` used nowhere else inside
/// the loop.
#[derive(Clone, Debug)]
pub struct Reduction {
    /// The header block parameter holding the accumulator.
    pub param: ValueId,
    /// Its position in the header's parameter list.
    pub param_index: usize,
    /// The initial value, from the entering edge.
    pub start: Option<ValueId>,
    pub kind: RedKind,
    /// The result of the combining instruction — the value the latch passes back.
    pub combine: ValueId,
    /// The non-accumulator operand of the combine. `None` for an `Fma` accumulate, where the
    /// addend is the (never-materialized) product in [`Self::fma_factors`].
    pub addend: Option<ValueId>,
    /// For `acc = fma(x, y, acc)`: the two multiplicands, fused into the accumulate with a single
    /// rounding.
    pub fma_factors: Option<(ValueId, ValueId)>,
    /// Copy of [`RedKind::reassociable`], so a consumer cannot forget to ask.
    pub reassociable: bool,
}

/// What a header block parameter carries across iterations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Carried {
    /// A basic induction variable; the index is into [`NaturalLoop::ivs`].
    Iv(usize),
    /// An accumulator; the index is into [`NaturalLoop::reductions`].
    Reduction(usize),
    /// The same value on every iteration (the latch passes the parameter straight back, or a copy
    /// cycle through inner-loop parameters resolves to it).
    Invariant,
    /// A true recurrence (`h = a[i]*h + b[i]`), a conditionally-updated value, or anything else
    /// this analysis does not recognize. Iterations cannot be reordered.
    Recurrence,
}

/// One header block parameter and what it carries.
#[derive(Clone, Copy, Debug)]
pub struct CarriedValue {
    pub param: ValueId,
    pub kind: Carried,
}

// ---------------------------------------------------------------------------------------------
// Dependence
// ---------------------------------------------------------------------------------------------

/// The loop-carried *memory* dependence verdict. Register-carried dependences are reported
/// separately, per header parameter, by [`Carried`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemDep {
    /// Proved: no iteration writes a location any other iteration reads or writes. (A load and a
    /// store at the *same* address in the *same* iteration is not a loop-carried dependence.)
    Independent,
    /// The same conclusion, but resting on one stated assumption: that a `load ptr <invariant>`
    /// executed inside the loop — the reloaded base pointer of a `[]T` slice, see
    /// [`MemBase::ReloadedOuter`] — yields the same pointer on every iteration. That holds unless
    /// the loop stores a pointer or calls something, both of which are already excluded before
    /// this verdict is reached; what remains is a store through an address the analysis
    /// understands that nevertheless lands on the slice header. Promote this to
    /// [`Self::Independent`] with a real alias analysis, or guard it at runtime.
    IndependentIfBasesStable,
    /// Every pair of accesses the analysis *could* relate is at distance 0 (same element, same
    /// iteration), and every pair it could not relate sits on two different [`MemBase`]s whose
    /// relationship is unknown — two `[]T` slice parameters, say, which the caller is free to have
    /// aliased. There is no loop-carried dependence **provided** those bases denote either the very
    /// same pointer (distance 0 again) or disjoint objects.
    ///
    /// That is not an assumption: it is a runtime-checkable fact, and the consumer owes the check.
    /// `vectorize` emits it in the loop's preheader as a pointer range test per unrelated pair and
    /// runs zero vector iterations when it fails, so the original scalar loop still does all the
    /// work. Anything that reads this verdict without emitting such a check is unsound.
    IndependentIfBasesDisjoint,
    /// A loop-carried dependence exists, or could not be ruled out.
    Carried,
}

/// Why the dependence test reached its verdict.
#[derive(Clone, Debug)]
pub struct DepSummary {
    pub memory: MemDep,
    /// A short human-readable justification, for the dump.
    pub reason: String,
}

impl Default for DepSummary {
    fn default() -> DepSummary {
        DepSummary {
            memory: MemDep::Carried,
            reason: "not analyzed".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

/// Analyze every natural loop of `f`. Blocks unreachable from the entry are ignored (dominance is
/// only defined on reachable blocks), so the result is the same whether or not
/// `cfg::prune_unreachable` has run.
pub fn analyze_function(f: &Function) -> LoopForest {
    let rpo = cfg::reverse_postorder(f);
    let preds = cfg::predecessors(f);
    let idom = dom::idoms_from(f, &rpo, &preds);
    analyze_with(f, &idom, &preds)
}

/// [`analyze_function`] against precomputed dominators/predecessors (what a pass holding a
/// [`crate::CfgAnalyses`] already has).
pub(crate) fn analyze_with(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> LoopForest {
    let mut forest = find_loops(f, idom, preds);
    for i in 0..forest.loops.len() {
        analyze_loop(f, &mut forest, i, Stages::All);
    }
    forest
}

/// Run the value analyses for **one** loop of an already-structured forest, out to `stages`.
///
/// This is what a transform wants instead of [`analyze_with`]: the full analysis walks every
/// instruction of every loop and builds an affine expression for each integer result, and a
/// transform that widens (or declines) one loop at a time pays that for every loop it never looks
/// at. Stopping at [`Stages::Ivs`] answers "unit-stride counted loop?" — the single largest reason
/// the corpus declines — for a fraction of the cost, and [`Stages::Memory`] skips the derived-IV
/// list, which no transform reads.
///
/// Calling it twice on the same loop with a wider `stages` recomputes the earlier stages; that is
/// deliberate, because the alternative is caching a half-analysis whose defaults no longer mean
/// "not analyzed".
pub(crate) fn analyze_one(f: &Function, forest: &mut LoopForest, idx: usize, stages: Stages) {
    analyze_loop(f, forest, idx, stages);
}

/// Loop **structure only** — headers, latches, block sets, preheaders, exits and the nesting forest
/// — skipping the induction-variable, trip-count, affine and dependence stages.
///
/// For a consumer that only rewires edges (`loop_canon`), the value analyses are pure cost: they
/// walk every instruction of every loop and build an affine expression for each integer result, and
/// they would be thrown away and recomputed after each rewrite. Every field they fill is left at its
/// default here, so `ivs` is empty, `primary_iv` is `None` and `trip` is `TripCount::Unknown` —
/// **do not read them off this result**; call [`analyze_with`] if you need them.
pub(crate) fn structure_with(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> LoopForest {
    find_loops(f, idom, preds)
}

// ---------------------------------------------------------------------------------------------
// Stage 1: loop structure
// ---------------------------------------------------------------------------------------------

/// Does `a` dominate `b`? Walks the immediate-dominator chain up from `b`.
fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    if idom[b as usize] == NO_IDOM {
        return false; // b is unreachable: dominance is undefined
    }
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        let id = idom[x as usize];
        if id == x || id == NO_IDOM {
            return false; // reached the entry
        }
        x = id;
    }
}

/// Find every natural loop and build the nesting forest. The value analyses are left at their
/// defaults; [`analyze_loop`] fills them in.
fn find_loops(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> LoopForest {
    // header -> (block set, latches)
    let mut by_header: FxHashMap<u32, (FxHashSet<u32>, Vec<u32>)> = FxHashMap::default();
    for b in &f.blocks {
        let n = b.id.0;
        if idom[n as usize] == NO_IDOM {
            continue; // unreachable
        }
        for s in cfg::successors(&b.term) {
            let h = s.0;
            if !dominates(h, n, idom) {
                continue;
            }
            // Back edge n -> h. The loop body is h plus every block that reaches n without
            // passing through h.
            let entry = by_header.entry(h).or_insert_with(|| {
                let mut set = FxHashSet::default();
                set.insert(h);
                (set, Vec::new())
            });
            entry.1.push(n);
            if n != h {
                let mut stack = vec![n];
                entry.0.insert(n);
                while let Some(x) = stack.pop() {
                    for &p in &preds[x as usize] {
                        if entry.0.insert(p) {
                            stack.push(p);
                        }
                    }
                }
            }
        }
    }

    // Deterministic order: ascending header id.
    let mut headers: Vec<u32> = by_header.keys().copied().collect();
    headers.sort_unstable();

    let mut loops: Vec<NaturalLoop> = Vec::with_capacity(headers.len());
    for (i, h) in headers.iter().enumerate() {
        let (set, mut latches) = by_header.remove(h).unwrap();
        latches.sort_unstable();
        latches.dedup();
        let mut blocks: Vec<u32> = set.iter().copied().collect();
        blocks.sort_unstable();

        let entering: Vec<u32> = preds[*h as usize]
            .iter()
            .copied()
            .filter(|p| !set.contains(p))
            .collect();
        // A single entering block that branches unconditionally to the header is a preheader. It
        // then necessarily dominates the header: every other path into the header is a back edge
        // from inside the loop, and the header dominates those.
        let preheader = match entering.as_slice() {
            [p] if matches!(&f.blocks[*p as usize].term,
                            Terminator::Br { target, .. } if target.0 == *h) =>
            {
                Some(BlockId(*p))
            }
            _ => None,
        };

        let mut exits: Vec<(BlockId, BlockId)> = Vec::new();
        for &b in &blocks {
            for s in cfg::successors(&f.blocks[b as usize].term) {
                if !set.contains(&s.0) {
                    exits.push((BlockId(b), s));
                }
            }
        }
        exits.sort_unstable_by_key(|(a, b)| (a.0, b.0));
        exits.dedup();

        // A `ret`/`unreachable` inside the loop leaves it with no edge at all, so it never appears
        // in `exits`. Record it separately rather than let a consumer mistake the loop for
        // single-exit.
        let abnormal_exits: Vec<BlockId> = blocks
            .iter()
            .filter(|&&b| {
                matches!(
                    f.blocks[b as usize].term,
                    Terminator::Ret(_) | Terminator::Unreachable
                )
            })
            .map(|&b| BlockId(b))
            .collect();

        loops.push(NaturalLoop {
            id: LoopId(i as u32),
            header: BlockId(*h),
            latches: latches.into_iter().map(BlockId).collect(),
            blocks: blocks.into_iter().map(BlockId).collect(),
            entering: entering.into_iter().map(BlockId).collect(),
            preheader,
            exits,
            abnormal_exits,
            parent: None,
            children: Vec::new(),
            depth: 0,
            ivs: Vec::new(),
            primary_iv: None,
            trip: TripCount::Unknown,
            derived: Vec::new(),
            reductions: Vec::new(),
            carried: Vec::new(),
            accesses: Vec::new(),
            dep: DepSummary::default(),
        });
    }

    // Nesting: a loop's parent is the smallest *other* loop whose block set contains its header.
    // Distinct natural loops are either disjoint or properly nested (they cannot partially
    // overlap), so "smallest containing" is unambiguous.
    let sizes: Vec<usize> = loops.iter().map(|l| l.blocks.len()).collect();
    let block_sets: Vec<FxHashSet<u32>> = loops
        .iter()
        .map(|l| l.blocks.iter().map(|b| b.0).collect())
        .collect();
    for i in 0..loops.len() {
        let h = loops[i].header.0;
        let mut best: Option<usize> = None;
        for j in 0..loops.len() {
            if i == j || !block_sets[j].contains(&h) {
                continue;
            }
            if best.is_none_or(|b| sizes[j] < sizes[b]) {
                best = Some(j);
            }
        }
        loops[i].parent = best.map(|j| LoopId(j as u32));
    }
    for i in 0..loops.len() {
        if let Some(p) = loops[i].parent {
            let child = loops[i].id;
            loops[p.0 as usize].children.push(child);
        }
    }
    for l in &mut loops {
        l.children.sort_unstable();
    }
    // Depth: walk up the parent chain (the forest is acyclic and shallow).
    for i in 0..loops.len() {
        let mut d = 0;
        let mut cur = loops[i].parent;
        while let Some(p) = cur {
            d += 1;
            cur = loops[p.0 as usize].parent;
        }
        loops[i].depth = d;
    }

    // Innermost-loop map: the deepest loop containing each block.
    let mut innermost: Vec<Option<LoopId>> = vec![None; f.blocks.len()];
    for l in &loops {
        for b in &l.blocks {
            let slot = &mut innermost[b.0 as usize];
            let deeper = match slot {
                Some(prev) => l.depth > loops[prev.0 as usize].depth,
                None => true,
            };
            if deeper {
                *slot = Some(l.id);
            }
        }
    }

    LoopForest { loops, innermost }
}

// ---------------------------------------------------------------------------------------------
// Stage 2: induction variables, trip counts, affine expressions
// ---------------------------------------------------------------------------------------------

/// Everything the value analyses need about one loop, computed once.
struct LoopCtx<'a> {
    f: &'a Function,
    /// Value id -> `(block, inst)` of its defining instruction. `None` for block parameters and
    /// for ids no instruction defines.
    def: Vec<Option<(u32, u32)>>,
    /// Blocks in the loop.
    blocks: FxHashSet<u32>,
    /// Values defined inside the loop: every loop block's parameters and instruction results.
    defined_in_loop: FxHashSet<u32>,
    /// The induction variable every [`AffineExpr`] in this loop is written against.
    primary: Option<ValueId>,
    /// Memo for [`LoopCtx::is_value_invariant`].
    inv_memo: std::cell::RefCell<FxHashMap<u32, bool>>,
}

/// How deep `affine_of` will chase an operand chain. Cycles are impossible (every cycle in SSA
/// runs through a block parameter, which is a base case), so this only guards against a
/// pathologically long straight-line chain.
const AFFINE_DEPTH_LIMIT: u32 = 64;

impl<'a> LoopCtx<'a> {
    fn new(f: &'a Function, l: &NaturalLoop) -> LoopCtx<'a> {
        let mut def: Vec<Option<(u32, u32)>> = vec![None; f.value_types.len()];
        for b in &f.blocks {
            for (i, inst) in b.insts.iter().enumerate() {
                if let Some(r) = inst.result {
                    if (r.0 as usize) < def.len() {
                        def[r.0 as usize] = Some((b.id.0, i as u32));
                    }
                }
            }
        }
        let blocks: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
        let mut defined_in_loop = FxHashSet::default();
        for &b in &blocks {
            let blk = &f.blocks[b as usize];
            for p in &blk.params {
                defined_in_loop.insert(p.0);
            }
            for inst in &blk.insts {
                if let Some(r) = inst.result {
                    defined_in_loop.insert(r.0);
                }
            }
        }
        LoopCtx {
            f,
            def,
            blocks,
            defined_in_loop,
            primary: None,
            inv_memo: Default::default(),
        }
    }

    /// Memoized [`LoopCtx::value_invariant`].
    fn is_value_invariant(&self, v: ValueId) -> bool {
        let mut seen = self.inv_memo.take();
        let r = self.value_invariant(v, &mut seen, 0);
        self.inv_memo.replace(seen);
        r
    }

    fn op_of(&self, v: ValueId) -> Option<&'a Op> {
        let (b, i) = (*self.def.get(v.0 as usize)?)?;
        Some(&self.f.blocks[b as usize].insts[i as usize].op)
    }

    /// Is `v` unchanged by the loop — i.e. defined outside it?
    fn is_invariant(&self, v: ValueId) -> bool {
        !self.defined_in_loop.contains(&v.0)
    }

    /// Does `v` hold the same value on every iteration, even if the instruction computing it sits
    /// *inside* the loop? True when every operand is itself loop-invariant and the operation is
    /// pure integer/boolean arithmetic — a `Load` is excluded, because an invariant address can
    /// still yield a different value after a store.
    ///
    /// This is what keeps the analysis independent of whether LICM has run. Before `-O2`, the row
    /// base `i*n` of an `a[i*n + k]` is still computed inside the `k` loop; without this the `k`
    /// loop's index would look non-affine at `-O1` and affine at `-O2`.
    fn value_invariant(&self, v: ValueId, seen: &mut FxHashMap<u32, bool>, depth: u32) -> bool {
        if !self.defined_in_loop.contains(&v.0) {
            return true;
        }
        if let Some(&cached) = seen.get(&v.0) {
            return cached;
        }
        if depth > AFFINE_DEPTH_LIMIT {
            return false;
        }
        // A block parameter defined in the loop is where a cycle closes: assume it varies.
        let Some(op) = self.op_of(v) else {
            return false;
        };
        let pure = matches!(
            op,
            Op::ConstInt(..)
                | Op::ConstFloat(..)
                | Op::Bin(..)
                | Op::Cmp(..)
                | Op::Neg(..)
                | Op::Not(..)
                | Op::Cast(..)
                | Op::Select(..)
        );
        if !pure {
            seen.insert(v.0, false);
            return false;
        }
        // Insert `false` before recursing so a (impossible in SSA, but cheap to guard) cycle
        // resolves to "varies" rather than looping forever.
        seen.insert(v.0, false);
        let mut uses = Vec::new();
        each_op_use(op, &mut |u| uses.push(u));
        let ok = uses
            .into_iter()
            .all(|u| self.value_invariant(u, seen, depth + 1));
        seen.insert(v.0, ok);
        ok
    }

    /// `v` as a compile-time integer constant, wherever it is defined.
    fn const_int(&self, v: ValueId) -> Option<i64> {
        match self.op_of(v)? {
            Op::ConstInt(c, ty) if ty.is_int() => i64::try_from(*c).ok(),
            _ => None,
        }
    }

    /// Express `v` as `coeff·iv + invariants + konst`, or `None` if it is not affine in the
    /// loop's primary induction variable. See [`AffineExpr`] for the no-wraparound assumption.
    fn affine_of(
        &self,
        v: ValueId,
        memo: &mut FxHashMap<u32, Option<AffineExpr>>,
        depth: u32,
    ) -> Option<AffineExpr> {
        if let Some(cached) = memo.get(&v.0) {
            return cached.clone();
        }
        if depth > AFFINE_DEPTH_LIMIT {
            return None;
        }
        let r = self.affine_uncached(v, memo, depth);
        memo.insert(v.0, r.clone());
        r
    }

    fn affine_uncached(
        &self,
        v: ValueId,
        memo: &mut FxHashMap<u32, Option<AffineExpr>>,
        depth: u32,
    ) -> Option<AffineExpr> {
        if Some(v) == self.primary {
            return Some(AffineExpr::iv());
        }
        if self.is_invariant(v) {
            // A loop-invariant value is either a literal we can fold into `konst`, or an opaque
            // symbol we carry along. Either way it does not move with the IV.
            return Some(match self.const_int(v) {
                Some(c) => AffineExpr::constant(c),
                None => AffineExpr::symbol(v),
            });
        }
        // Defined in the loop. A block parameter (other than the primary IV) is where every SSA
        // cycle closes, so it is also where the recursion stops: a second induction variable, a
        // reduction, or a merge we do not model.
        let op = self.op_of(v)?;
        match op {
            Op::ConstInt(c, ty) if ty.is_int() => {
                Some(AffineExpr::constant(i64::try_from(*c).ok()?))
            }
            Op::Bin(BinOp::Add, a, b) => {
                let (x, y) = (
                    self.affine_of(*a, memo, depth + 1)?,
                    self.affine_of(*b, memo, depth + 1)?,
                );
                x.add(&y)
            }
            Op::Bin(BinOp::Sub, a, b) => {
                let (x, y) = (
                    self.affine_of(*a, memo, depth + 1)?,
                    self.affine_of(*b, memo, depth + 1)?,
                );
                x.sub(&y)
            }
            Op::Bin(BinOp::Mul, a, b) => {
                let (x, y) = (
                    self.affine_of(*a, memo, depth + 1)?,
                    self.affine_of(*b, memo, depth + 1)?,
                );
                x.mul(&y)
            }
            // `x << k` with a literal k is `x * 2^k`. Signed shifts by >= 63 are not modelled.
            Op::Bin(BinOp::Shl, a, b) => {
                let k = self.const_int(*b)?;
                if !(0..63).contains(&k) {
                    return None;
                }
                self.affine_of(*a, memo, depth + 1)?.scale(1i64 << k)
            }
            Op::Neg(a) => self.affine_of(*a, memo, depth + 1)?.neg(),
            // `sext x == x` whenever `x` did not overflow its own width — the assumption the whole
            // affine model already rests on. `zext` is NOT transparent (a negative narrow value
            // becomes a large positive one), nor are `trunc`/`bitcast`.
            Op::Cast(CastKind::SExt, a, _) => self.affine_of(*a, memo, depth + 1),
            _ => None,
        }
        // A computation this decomposition cannot take apart — most often a product of two
        // loop-invariant values, like the row base `i*n` of an inner loop — is still a usable
        // opaque symbol as long as its value does not change across iterations.
        .or_else(|| self.is_value_invariant(v).then(|| AffineExpr::symbol(v)))
    }
}

/// The argument every latch passes to the header at parameter position `k`, when they all agree.
fn latch_arg(f: &Function, l: &NaturalLoop, k: usize) -> Option<ValueId> {
    let mut seen: Option<ValueId> = None;
    for latch in &l.latches {
        let args = args_to(&f.blocks[latch.0 as usize].term, l.header)?;
        let v = *args.get(k)?;
        match seen {
            Some(prev) if prev != v => return None,
            _ => seen = Some(v),
        }
    }
    seen
}

/// The argument every entering edge passes to the header at parameter position `k`, when they all
/// agree.
fn entry_arg(f: &Function, l: &NaturalLoop, k: usize) -> Option<ValueId> {
    let mut seen: Option<ValueId> = None;
    for pred in &l.entering {
        let args = args_to(&f.blocks[pred.0 as usize].term, l.header)?;
        let v = *args.get(k)?;
        match seen {
            Some(prev) if prev != v => return None,
            _ => seen = Some(v),
        }
    }
    seen
}

/// Basic induction variables: header parameters whose latch argument is `param + step` with a
/// loop-invariant step.
fn find_basic_ivs(ctx: &LoopCtx<'_>, l: &NaturalLoop) -> Vec<InductionVar> {
    let f = ctx.f;
    let header = &f.blocks[l.header.0 as usize];
    let mut ivs = Vec::new();
    for (k, &p) in header.params.iter().enumerate() {
        let Some(next) = latch_arg(f, l, k) else {
            continue;
        };
        if next == p {
            continue; // carried unchanged — an invariant, not an IV
        }
        let Some(op) = ctx.op_of(next) else { continue };
        let step = match op {
            Op::Bin(BinOp::Add, a, b) => {
                // `p + s` or `s + p`, with `s` loop-invariant.
                let s = if *a == p && *b != p {
                    *b
                } else if *b == p && *a != p {
                    *a
                } else {
                    continue;
                };
                match ctx.const_int(s) {
                    Some(c) => IvStep::Const(c),
                    None if ctx.is_invariant(s) => IvStep::Invariant(s),
                    None => continue,
                }
            }
            // `p - c` is a countdown. Only a literal is accepted: there is no MIR value for `-s`
            // to point `IvStep::Invariant` at, and inventing one would be a transform.
            Op::Bin(BinOp::Sub, a, b) if *a == p => {
                match ctx.const_int(*b).and_then(i64::checked_neg) {
                    Some(c) => IvStep::Const(c),
                    None => continue,
                }
            }
            _ => continue,
        };
        ivs.push(InductionVar {
            value: p,
            param_index: k,
            start: entry_arg(f, l, k),
            next,
            step,
            ty: f.value_type(p).clone(),
        });
    }
    ivs
}

/// The exit test, if the loop has exactly one exit edge and it is a comparison of a basic IV
/// against a loop-invariant bound. Returns `(iv index, continue-predicate, bound, exiting block)`,
/// where the predicate is oriented as `iv <pred> bound` **while the loop keeps going**.
fn exit_test(
    ctx: &LoopCtx<'_>,
    l: &NaturalLoop,
    ivs: &[InductionVar],
) -> Option<(usize, CmpOp, ValueId, BlockId)> {
    if l.exits.len() != 1 {
        return None;
    }
    let (from, to) = l.exits[0];
    let Terminator::CondBr {
        cond,
        then_blk,
        else_blk,
        ..
    } = &ctx.f.blocks[from.0 as usize].term
    else {
        return None;
    };
    // Which arm stays in the loop? (`to` is the arm that leaves.)
    let continue_on_true = if *else_blk == to && *then_blk != to {
        true
    } else if *then_blk == to && *else_blk != to {
        false
    } else {
        return None;
    };
    let Some(Op::Cmp(pred, a, b)) = ctx.op_of(*cond) else {
        return None;
    };
    // Orient as `iv <pred> bound`.
    let (idx, pred, bound) = if let Some(i) = ivs.iter().position(|iv| iv.value == *a) {
        (i, *pred, *b)
    } else if let Some(i) = ivs.iter().position(|iv| iv.value == *b) {
        (i, swap_cmp(*pred)?, *a)
    } else {
        return None;
    };
    if !ctx.is_value_invariant(bound) {
        return None;
    }
    let pred = if continue_on_true {
        pred
    } else {
        negate_cmp(pred)?
    };
    Some((idx, pred, bound, from))
}

/// The predicate that means the same thing with the operands exchanged (`a < b` == `b > a`).
fn swap_cmp(p: CmpOp) -> Option<CmpOp> {
    use CmpOp::*;
    Some(match p {
        Eq => Eq,
        Ne => Ne,
        Slt => Sgt,
        Sle => Sge,
        Sgt => Slt,
        Sge => Sle,
        Ult => Ugt,
        Ule => Uge,
        Ugt => Ult,
        Uge => Ule,
        // Float compares never bound an integer induction variable.
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
}

/// The negation of an integer predicate.
fn negate_cmp(p: CmpOp) -> Option<CmpOp> {
    use CmpOp::*;
    Some(match p {
        Eq => Ne,
        Ne => Eq,
        Slt => Sge,
        Sle => Sgt,
        Sgt => Sle,
        Sge => Slt,
        Ult => Uge,
        Ule => Ugt,
        Ugt => Ule,
        Uge => Ult,
        // `!(a < b)` is NOT `a >= b` for floats (both are false on NaN).
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
}

/// The number of iterations, from the exit test. Requires the single exit to be tested in the
/// **header** — a bottom-tested loop runs its body once before the test, which is a different
/// count — and a constant step in the direction the predicate walks.
fn trip_count(
    ctx: &LoopCtx<'_>,
    l: &NaturalLoop,
    iv: &InductionVar,
    pred: CmpOp,
    bound: ValueId,
    from: BlockId,
) -> TripCount {
    use CmpOp::*;
    if from != l.header {
        return TripCount::Unknown;
    }
    // An early `return` in the body cuts the count short with no exit edge to see it.
    if !l.abnormal_exits.is_empty() {
        return TripCount::Unknown;
    }
    let Some(step) = iv.step.as_const() else {
        return TripCount::Unknown;
    };
    let (inclusive, signed, ascending) = match pred {
        Slt => (false, true, true),
        Sle => (true, true, true),
        Ult => (false, false, true),
        Ule => (true, false, true),
        Sgt => (false, true, false),
        Sge => (true, true, false),
        Ugt => (false, false, false),
        Uge => (true, false, false),
        _ => return TripCount::Unknown,
    };
    // The IV must walk toward the bound, or the loop is infinite / zero-trip in a way this does
    // not model.
    if (ascending && step <= 0) || (!ascending && step >= 0) {
        return TripCount::Unknown;
    }
    let Some(start_v) = iv.start else {
        return TripCount::Unknown;
    };

    if let (Some(s), Some(e)) = (ctx.const_int(start_v), ctx.const_int(bound)) {
        // An unsigned predicate on a negative literal would compare as a huge positive; decline
        // rather than guess.
        if !signed && (s < 0 || e < 0) {
            return TripCount::Unknown;
        }
        let span = if ascending {
            e.checked_sub(s)
        } else {
            s.checked_sub(e)
        };
        let Some(span) = span else {
            return TripCount::Unknown;
        };
        let mag = step.unsigned_abs() as i64;
        let n = if inclusive {
            if span < 0 {
                0
            } else {
                span / mag + 1
            }
        } else if span <= 0 {
            0
        } else {
            // ceil(span / mag), both positive.
            (span + mag - 1) / mag
        };
        return TripCount::Const(n as u64);
    }

    // A symbolic count is only useful if both endpoints are evaluable *before* the loop, so a
    // consumer can compute the count in the preheader.
    if !ctx.is_invariant(start_v) || !ctx.is_invariant(bound) {
        return TripCount::Unknown;
    }
    TripCount::Affine {
        start: start_v,
        end: bound,
        step,
        inclusive,
        signed,
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 3: memory accesses
// ---------------------------------------------------------------------------------------------

/// How many nested `gep`s `addr_form` will fold into one index expression.
const GEP_CHAIN_LIMIT: u32 = 4;

impl<'a> LoopCtx<'a> {
    /// The object `p` points at, when the loop does not recompute it.
    ///
    /// Two shapes qualify. A pointer *defined outside* the loop is an [`MemBase::Outer`]. A
    /// `load ptr <loop-invariant address>` executed *inside* the loop is a
    /// [`MemBase::ReloadedOuter`]: that is how every `[]T` slice reads its data pointer today, and
    /// refusing it would leave the analysis blind to every slice loop in the corpus.
    fn mem_base(&self, p: ValueId) -> Option<MemBase> {
        if self.is_invariant(p) {
            return Some(MemBase::Outer(p));
        }
        match self.op_of(p)? {
            Op::Load(addr, MirType::Ptr) if self.is_invariant(*addr) => {
                Some(MemBase::ReloadedOuter {
                    addr: *addr,
                    value: p,
                })
            }
            _ => None,
        }
    }

    /// Decompose the address `p` into `base + index · elem`.
    ///
    /// A chain of `gep`s over the *same* element type folds into one index (`gep(gep(b, j), k)` is
    /// `b + (j + k)·elem`); a chain that changes element type is not foldable without a size ratio
    /// this pass does not track, so it declines. Anything else is [`AddrForm::Unknown`], which every
    /// consumer must read as "may touch any address".
    fn addr_form(&self, p: ValueId, memo: &mut FxHashMap<u32, Option<AffineExpr>>) -> AddrForm {
        // The pointer value itself may already be the base — a whole-object load/store, or a
        // pointer the loop never recomputes.
        if let Some(b) = self.mem_base(p) {
            return AddrForm::Invariant(b);
        }
        let mut cur = p;
        let mut acc: Option<AffineExpr> = None;
        let mut elem_ty: Option<MirType> = None;
        for _ in 0..GEP_CHAIN_LIMIT {
            let Some(Op::Gep { ptr, index, elem }) = self.op_of(cur) else {
                return AddrForm::Unknown;
            };
            if let Some(prev) = &elem_ty {
                if prev != elem {
                    return AddrForm::Unknown; // a change of stride unit mid-chain
                }
            } else {
                elem_ty = Some(elem.clone());
            }
            let Some(e) = self.affine_of(*index, memo, 0) else {
                return AddrForm::Unknown;
            };
            acc = match acc {
                None => Some(e),
                Some(a) => match a.add(&e) {
                    Some(s) => Some(s),
                    None => return AddrForm::Unknown,
                },
            };
            if let Some(base) = self.mem_base(*ptr) {
                let (Some(index), Some(elem)) = (acc, elem_ty) else {
                    return AddrForm::Unknown;
                };
                return AddrForm::Affine { base, index, elem };
            }
            cur = *ptr;
        }
        AddrForm::Unknown
    }
}

/// Every load and store in the loop, in block-then-instruction order, with its address decomposed.
fn collect_accesses(
    ctx: &LoopCtx<'_>,
    l: &NaturalLoop,
    memo: &mut FxHashMap<u32, Option<AffineExpr>>,
) -> Vec<MemAccess> {
    let mut out = Vec::new();
    for &b in &l.blocks {
        for (i, inst) in ctx.f.blocks[b.0 as usize].insts.iter().enumerate() {
            let (kind, ptr, value_ty) = match &inst.op {
                Op::Load(p, ty) => (AccessKind::Load, *p, ty.clone()),
                Op::Store { ptr, value } => {
                    (AccessKind::Store, *ptr, ctx.f.value_type(*value).clone())
                }
                _ => continue,
            };
            let addr = ctx.addr_form(ptr, memo);
            out.push(MemAccess {
                block: b,
                inst: i,
                kind,
                ptr,
                value_ty,
                addr,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Stage 4: loop-carried values and dependence
// ---------------------------------------------------------------------------------------------

/// The reduction operator a combining instruction implements, given that one operand is the
/// accumulator `p`. Returns `(kind, the other operand, fma factors)`.
///
/// A combine is `acc ⊕ x` where `⊕` is one of the operators [`RedKind`] names and `acc` is used
/// nowhere else in the loop. Whether the partials may then be reassociated is a separate question
/// that [`RedKind::reassociable`] answers; a float `add` is a reduction here and is still not
/// reassociable.
///
/// **Subtraction is included, and only with the accumulator on the LEFT.** `acc = acc - x` is an
/// accumulator like any other: it has a fixed operand order, its partials may not be reassociated,
/// and a consumer folds it serially in index order — the same treatment a float `add` already gets.
/// `x - acc` is a different thing entirely (it negates the whole history every iteration) and stays
/// a `Recurrence`. This matters because a negated accumulate is how a loss is spelled:
/// `lr = lr - alpha*q*(1-p)^2*log p` in a focal loss, `e = e - t[i]*log(y[i])` in a cross-entropy.
/// Refusing it made the entire enclosing loop unvectorizable, transcendental and all.
// The tuple return is three independent optional facts about one combining instruction; a named
// struct for a single internal caller would add ceremony, not clarity.
#[allow(clippy::type_complexity)]
fn combine_kind(
    ctx: &LoopCtx<'_>,
    p: ValueId,
    next: ValueId,
) -> Option<(RedKind, Option<ValueId>, Option<(ValueId, ValueId)>)> {
    match ctx.op_of(next)? {
        Op::Bin(b, a, c) => {
            // `acc - x` only; `x - acc` is not an accumulate.
            if matches!(b, BinOp::Sub | BinOp::FSub) {
                if *a != p || *c == p {
                    return None;
                }
                let kind = if matches!(b, BinOp::Sub) {
                    RedKind::Sub
                } else {
                    RedKind::FSub
                };
                return Some((kind, Some(*c), None));
            }
            let other = if *a == p && *c != p {
                *c
            } else if *c == p && *a != p {
                *a
            } else {
                return None; // `acc ⊕ acc` is not a reduction over the loop's data
            };
            let kind = match b {
                BinOp::Add => RedKind::Add,
                BinOp::Mul => RedKind::Mul,
                BinOp::And => RedKind::And,
                BinOp::Or => RedKind::Or,
                BinOp::Xor => RedKind::Xor,
                BinOp::FAdd => RedKind::FAdd,
                BinOp::FMul => RedKind::FMul,
                _ => return None,
            };
            Some((kind, Some(other), None))
        }
        // `acc = fma(x, y, acc)` — the front end contracts `acc + x*y` into this, so it is the
        // shape a dot-product / weighted-sum reduction actually has after lowering. The addend is
        // the never-materialized product `x*y`; a consumer that widens the body has to produce
        // both factors and re-fuse them per lane, or it changes the rounding.
        Op::Fma(x, y, c) if *c == p && *x != p && *y != p => {
            Some((RedKind::FAdd, None, Some((*x, *y))))
        }
        _ => None,
    }
}

/// How many times `v` is read inside the loop (instruction operands and terminator arguments).
fn uses_in_loop(ctx: &LoopCtx<'_>, l: &NaturalLoop, v: ValueId) -> usize {
    let mut n = 0;
    for &b in &l.blocks {
        let blk = &ctx.f.blocks[b.0 as usize];
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |u| {
                if u == v {
                    n += 1;
                }
            });
        }
        crate::each_term_use(&blk.term, &mut |u| {
            if u == v {
                n += 1;
            }
        });
    }
    n
}

/// Classify every header block parameter, and collect the reductions among them.
///
/// A parameter is a *reduction* when the latch passes back `param ⊕ x` and the parameter's **only**
/// use in the whole loop is that combine (plus the latch's branch argument, which reads the
/// combine, not the parameter). The second condition is what rules out `s = s + x; use(s); s = s +
/// y` shapes where an intermediate partial is observed: reordering the accumulate would change what
/// that observer sees.
///
/// SSA does the hardest part for free. The combine's value is an argument of the latch's
/// terminator, so the combine's block *dominates* the latch; and with a single latch and the only
/// exit in the header, every iteration that enters the body reaches the latch. A conditionally
/// updated accumulator (`if c { s = s + x }`) therefore cannot be mistaken for a reduction: its
/// latch argument is a merge *parameter*, not a combining instruction, so it falls through to
/// [`Carried::Recurrence`].
fn classify_carried(
    ctx: &LoopCtx<'_>,
    l: &NaturalLoop,
    ivs: &[InductionVar],
) -> (Vec<CarriedValue>, Vec<Reduction>) {
    let f = ctx.f;
    let header = &f.blocks[l.header.0 as usize];
    let mut carried = Vec::with_capacity(header.params.len());
    let mut reductions = Vec::new();
    for (k, &p) in header.params.iter().enumerate() {
        if let Some(i) = ivs.iter().position(|iv| iv.param_index == k) {
            carried.push(CarriedValue {
                param: p,
                kind: Carried::Iv(i),
            });
            continue;
        }
        let next = latch_arg(f, l, k);
        if next == Some(p) {
            carried.push(CarriedValue {
                param: p,
                kind: Carried::Invariant,
            });
            continue;
        }
        let kind = next
            .and_then(|n| {
                let (kind, addend, fma_factors) = combine_kind(ctx, p, n)?;
                if uses_in_loop(ctx, l, p) != 1 {
                    return None; // an intermediate partial is observed somewhere else
                }
                reductions.push(Reduction {
                    param: p,
                    param_index: k,
                    start: entry_arg(f, l, k),
                    kind,
                    combine: n,
                    addend,
                    fma_factors,
                    reassociable: kind.reassociable(),
                });
                Some(Carried::Reduction(reductions.len() - 1))
            })
            .unwrap_or(Carried::Recurrence);
        carried.push(CarriedValue { param: p, kind });
    }
    (carried, reductions)
}

/// The loop-carried memory-dependence verdict.
///
/// The test is deliberately blunt, because a blunt test that is *right* beats a clever one that is
/// nearly right when the consumer is a vectorizer. Two accesses conflict unless one of these holds:
///
/// * they are both loads (read/read is never a dependence);
/// * their bases are provably different objects (two distinct `alloca`s);
/// * their bases are the same object *and* their index expressions are identical, which puts them
///   at dependence distance 0 — the same element in the same iteration, which is a
///   loop-*independent* dependence that widening preserves.
///
/// A same-object pair at a **non-zero** distance is reported [`MemDep::Carried`] even when the
/// distance is large enough to be harmless at a given width: the width is not known here, and
/// leaving the refinement to the consumer would put the unsafe default on the wrong side.
///
/// Pairs whose bases could not be related at all downgrade the verdict to
/// [`MemDep::IndependentIfBasesDisjoint`] instead of failing it — see that variant for the
/// obligation that creates.
fn dependence(ctx: &LoopCtx<'_>, l: &NaturalLoop, accesses: &[MemAccess]) -> DepSummary {
    let carried = |why: &str| DepSummary {
        memory: MemDep::Carried,
        reason: why.to_string(),
    };
    // A call may read or write anything, and `VecKernelCall` writes through its output streams.
    for &b in &l.blocks {
        for inst in &ctx.f.blocks[b.0 as usize].insts {
            if matches!(inst.op, Op::Call { .. } | Op::VecKernelCall { .. }) {
                return carried("a call in the loop may touch any memory");
            }
        }
    }
    if accesses.iter().any(|a| matches!(a.addr, AddrForm::Unknown)) {
        return carried("an address the analysis could not decompose");
    }
    let mut needs_runtime_check = false;
    let mut reloaded_base = false;
    for (i, a) in accesses.iter().enumerate() {
        for b in &accesses[i + 1..] {
            if a.kind == AccessKind::Load && b.kind == AccessKind::Load {
                continue;
            }
            let (Some(ba), Some(bb)) = (a.addr.base(), b.addr.base()) else {
                return carried("an access with no identifiable base");
            };
            if matches!(ba, MemBase::ReloadedOuter { .. })
                || matches!(bb, MemBase::ReloadedOuter { .. })
            {
                reloaded_base = true;
            }
            if ba.same_object(bb) {
                // Same object: the distance is the difference of the two index expressions, and
                // only an exactly-zero distance is safe at every width.
                let same_index = match (&a.addr, &b.addr) {
                    (
                        AddrForm::Affine {
                            index: ia,
                            elem: ea,
                            ..
                        },
                        AddrForm::Affine {
                            index: ib,
                            elem: eb,
                            ..
                        },
                    ) => ea == eb && ia == ib,
                    (AddrForm::Invariant(_), AddrForm::Invariant(_)) => true,
                    _ => false,
                };
                if !same_index {
                    return carried("two accesses to one object at a non-zero distance");
                }
                continue;
            }
            if distinct_allocas(ctx, ba, bb) {
                continue;
            }
            needs_runtime_check = true;
        }
    }
    let memory = if needs_runtime_check {
        MemDep::IndependentIfBasesDisjoint
    } else if reloaded_base {
        MemDep::IndependentIfBasesStable
    } else {
        MemDep::Independent
    };
    DepSummary {
        memory,
        reason: match memory {
            MemDep::Independent => "every conflicting pair is provably independent".into(),
            MemDep::IndependentIfBasesStable => {
                "independent, assuming a re-loaded base pointer does not change".into()
            }
            MemDep::IndependentIfBasesDisjoint => {
                "independent if unrelated bases are equal or disjoint (runtime check owed)".into()
            }
            MemDep::Carried => unreachable!(),
        },
    }
}

/// Are these two bases two *different* `alloca`s? Distinct stack slots are distinct objects, which
/// is the one disjointness fact available without an alias analysis.
fn distinct_allocas(ctx: &LoopCtx<'_>, a: MemBase, b: MemBase) -> bool {
    let is_alloca = |m: MemBase| match m {
        MemBase::Outer(v) => matches!(ctx.op_of(v), Some(Op::Alloca(..))),
        MemBase::ReloadedOuter { .. } => false,
    };
    is_alloca(a) && is_alloca(b) && !a.same_object(b)
}

/// How far [`analyze_loop`] runs. The stages are strictly nested, and each one is materially more
/// expensive than the one before it, so a consumer that can decide from an early stage should stop
/// there rather than pay for the rest.
///
/// A loop analyzed to less than [`Stages::All`] leaves the later stages' fields at their defaults —
/// and those defaults are the *conservative* answers, not empty ones: `primary_iv` is `None`,
/// `trip` is [`TripCount::Unknown`] and `dep` is [`MemDep::Carried`] ("not analyzed"). A consumer
/// that reads a field it did not ask for therefore declines to transform rather than transforming
/// on a fact nobody established.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Stages {
    /// Basic induction variables, the primary one, and the trip count. Enough to answer "is this a
    /// unit-stride counted loop?", which is what rejects the largest share of the corpus.
    Ivs,
    /// … plus memory accesses, carried-value classification and the dependence verdict. Everything
    /// a transform reads, *except* the derived-IV list — no transform consumes that.
    Memory,
    /// … plus the derived-IV list: an affine expression for every integer instruction in the loop.
    /// Only [`dump_function_loops`] and this module's own tests read it.
    All,
}

fn analyze_loop(f: &Function, forest: &mut LoopForest, idx: usize, stages: Stages) {
    let l = &forest.loops[idx];
    let mut ctx = LoopCtx::new(f, l);

    let ivs = find_basic_ivs(&ctx, l);
    let test = exit_test(&ctx, l, &ivs);
    // The IV every affine expression is written against: the one the exit test walks, or — when
    // there is no usable exit test — the single constant-step IV, if there is exactly one.
    let primary = match test {
        Some((i, ..)) => Some(i),
        None => {
            let mut cands = ivs
                .iter()
                .enumerate()
                .filter(|(_, iv)| iv.step.as_const().is_some());
            match (cands.next(), cands.next()) {
                (Some((i, _)), None) => Some(i),
                _ => None,
            }
        }
    };
    ctx.primary = primary.map(|i| ivs[i].value);

    let trip = match (primary, test) {
        (Some(i), Some((_, pred, bound, from))) => trip_count(&ctx, l, &ivs[i], pred, bound, from),
        _ => TripCount::Unknown,
    };

    if stages == Stages::Ivs {
        let l = &mut forest.loops[idx];
        l.ivs = ivs;
        l.primary_iv = primary;
        l.trip = trip;
        return;
    }

    // Derived IVs: everything in the loop that moves with the primary IV.
    let mut memo: FxHashMap<u32, Option<AffineExpr>> = FxHashMap::default();
    let mut derived = Vec::new();
    if stages == Stages::All && ctx.primary.is_some() {
        let mut blocks: Vec<u32> = ctx.blocks.iter().copied().collect();
        blocks.sort_unstable();
        for b in blocks {
            for inst in &f.blocks[b as usize].insts {
                let Some(r) = inst.result else { continue };
                if !f.value_type(r).is_int() || Some(r) == ctx.primary {
                    continue;
                }
                if let Some(e) = ctx.affine_of(r, &mut memo, 0) {
                    if !e.is_invariant() {
                        derived.push(DerivedIv { value: r, expr: e });
                    }
                }
            }
        }
    }

    // Stage 3/4. The access decomposition is written against the primary IV, so it is only
    // meaningful once `ctx.primary` is set; without one every index would be `Unknown` and the
    // dependence test would answer `Carried` for reasons that say nothing about the loop.
    let (accesses, carried, reductions, dep) = if ctx.primary.is_some() {
        let accesses = collect_accesses(&ctx, l, &mut memo);
        let (carried, reductions) = classify_carried(&ctx, l, &ivs);
        let dep = dependence(&ctx, l, &accesses);
        (accesses, carried, reductions, dep)
    } else {
        (Vec::new(), Vec::new(), Vec::new(), DepSummary::default())
    };

    let l = &mut forest.loops[idx];
    l.ivs = ivs;
    l.primary_iv = primary;
    l.trip = trip;
    l.derived = derived;
    l.accesses = accesses;
    l.carried = carried;
    l.reductions = reductions;
    l.dep = dep;
}

// ---------------------------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------------------------

/// The branch arguments a terminator passes to `target`. `None` when the answer is ambiguous — a
/// `cond_br` whose two arms both go to `target` with different argument lists.
#[allow(dead_code)] // used by the induction-variable stage
fn args_to(t: &Terminator, target: BlockId) -> Option<&[ValueId]> {
    match t {
        Terminator::Br { target: tb, args } if *tb == target => Some(args),
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => match (*then_blk == target, *else_blk == target) {
            (true, true) => (then_args == else_args).then_some(then_args.as_slice()),
            (true, false) => Some(then_args),
            (false, true) => Some(else_args),
            (false, false) => None,
        },
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Debug dump
// ---------------------------------------------------------------------------------------------

/// Render the loop analysis of one function. `name` labels the output.
pub fn dump_function_loops(f: &Function, name: &str) -> String {
    let forest = analyze_function(f);
    let mut out = String::new();
    if forest.is_empty() {
        let _ = writeln!(out, "== loops in fn {name}: none ==");
        return out;
    }
    let _ = writeln!(out, "== loops in fn {name}: {} ==", forest.loops.len());
    for id in forest.innermost_first() {
        let l = forest.get(id);
        let _ = writeln!(
            out,
            "  {:?} header bb{} depth {} blocks [{}]",
            l.id,
            l.header.0,
            l.depth,
            l.blocks
                .iter()
                .map(|b| format!("bb{}", b.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "    latches [{}]  preheader {}  entering [{}]  parent {}",
            l.latches
                .iter()
                .map(|b| format!("bb{}", b.0))
                .collect::<Vec<_>>()
                .join(", "),
            l.preheader
                .map(|b| format!("bb{}", b.0))
                .unwrap_or_else(|| "none".into()),
            l.entering
                .iter()
                .map(|b| format!("bb{}", b.0))
                .collect::<Vec<_>>()
                .join(", "),
            l.parent
                .map(|p| format!("{p:?}"))
                .unwrap_or_else(|| "none".into()),
        );
        let _ = writeln!(
            out,
            "    exits [{}]",
            l.exits
                .iter()
                .map(|(a, b)| format!("bb{}->bb{}", a.0, b.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (i, iv) in l.ivs.iter().enumerate() {
            let step = match iv.step {
                IvStep::Const(k) => format!("{k:+}"),
                IvStep::Invariant(s) => format!("+v{}", s.0),
            };
            let start = iv
                .start
                .map(|s| format!("v{}", s.0))
                .unwrap_or_else(|| "?".into());
            let mark = if l.primary_iv == Some(i) {
                " [primary]"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "    iv v{} {} start {start} step {step} next v{}{mark}",
                iv.value.0,
                iv.ty.display(),
                iv.next.0
            );
        }
        let _ = writeln!(out, "    trip: {}", fmt_trip(&l.trip));
        for d in &l.derived {
            let _ = writeln!(out, "    derived v{} = {}", d.value.0, d.expr.display());
        }
        for c in &l.carried {
            let kind = match c.kind {
                Carried::Iv(i) => format!("iv#{i}"),
                Carried::Reduction(i) => {
                    format!("reduction#{i} ({})", l.reductions[i].kind.name())
                }
                Carried::Invariant => "invariant".to_string(),
                Carried::Recurrence => "RECURRENCE".to_string(),
            };
            let _ = writeln!(out, "    carried v{} {kind}", c.param.0);
        }
        for a in &l.accesses {
            let k = match a.kind {
                AccessKind::Load => "load ",
                AccessKind::Store => "store",
            };
            let _ = writeln!(
                out,
                "    {k} bb{}#{} {} @ {}",
                a.block.0,
                a.inst,
                a.value_ty.display(),
                fmt_addr(&a.addr)
            );
        }
        let _ = writeln!(
            out,
            "    dep: {:?} — {}  vectorizable-shape {}",
            l.dep.memory,
            l.dep.reason,
            l.is_vectorizable_shape()
        );
    }
    out
}

fn fmt_base(b: MemBase) -> String {
    match b {
        MemBase::Outer(v) => format!("v{}", v.0),
        MemBase::ReloadedOuter { addr, .. } => format!("*v{}", addr.0),
    }
}

fn fmt_addr(a: &AddrForm) -> String {
    match a {
        AddrForm::Affine { base, index, elem } => format!(
            "{}[{}] : {}",
            fmt_base(*base),
            index.display(),
            elem.display()
        ),
        AddrForm::Invariant(b) => format!("{} (invariant)", fmt_base(*b)),
        AddrForm::Unknown => "unknown".to_string(),
    }
}

fn fmt_trip(t: &TripCount) -> String {
    match t {
        TripCount::Const(n) => format!("{n}"),
        TripCount::Affine {
            start,
            end,
            step,
            inclusive,
            signed,
        } => format!(
            "v{} .. {}v{} step {step} ({})",
            start.0,
            if *inclusive { "=" } else { "" },
            end.0,
            if *signed { "signed" } else { "unsigned" }
        ),
        TripCount::Unknown => "unknown".into(),
    }
}

/// Render the loop analysis of a whole program. `interner` resolves function names when the caller
/// has one; without it functions are labelled by their raw symbol.
pub fn dump_program_loops(p: &Program, interner: Option<&Interner>) -> String {
    let mut out = String::new();
    for f in &p.funcs {
        let name = match interner {
            Some(i) => i.resolve(f.name).to_string(),
            None => format!("{:?}", f.name),
        };
        out.push_str(&dump_function_loops(f, &name));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    /// Compile a source string to (unoptimized) MIR and hand back the named function.
    #[allow(dead_code)] // used by the induction-variable stage's -O0 tests
    fn mir_of(src: &str, func: &str) -> Function {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let sym = interner.intern(func);
        program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"))
            .clone()
    }

    /// Compile and optimize, then hand back the named function (post-mem2reg SSA, which is the
    /// shape the vectorizer will actually see). Most tests use `-O1`: it runs mem2reg, so loops
    /// are already in block-parameter form, but not the `-O2` inliner, which would delete the small
    /// helper functions the tests name.
    fn opt_mir_of(src: &str, func: &str, level: u8) -> Function {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        crate::optimize(&mut program, level);
        let sym = interner.intern(func);
        program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"))
            .clone()
    }

    #[test]
    fn finds_a_single_loop() {
        let f = opt_mir_of(
            "fn s(n: i32) -> i32 { let mut a: i32 = 0; let mut i: i32 = 0; \
             while i < n { a = a + i; i = i + 1; } return a; } \
             fn main() -> i32 { return s(4); }",
            "s",
            1,
        );
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1, "{}", dump_function_loops(&f, "s"));
        let l = &forest.loops[0];
        assert_eq!(l.depth, 0);
        assert!(l.parent.is_none());
        assert!(!l.latches.is_empty());
        assert!(l.blocks.contains(&l.header));
        assert!(l.preheader.is_some(), "loop should have a preheader");
        assert_eq!(l.exits.len(), 1);
        assert_eq!(l.exits[0].0, l.header, "the exit test is in the header");
        // Every latch is inside the loop and branches back to the header.
        for lat in &l.latches {
            assert!(l.blocks.contains(lat));
        }
    }

    #[test]
    fn finds_nested_loops_and_their_nesting() {
        let f = opt_mir_of(
            "fn mm(a: [f32; 64], b: [f32; 64], mut c: [f32; 64], n: i32) { \
               let mut i: i32 = 0; \
               while i < n { let mut j: i32 = 0; \
                 while j < n { let mut s: f32 = 0.0; let mut k: i32 = 0; \
                   while k < n { s = s + a[i*n+k] * b[k*n+j]; k = k + 1; } \
                   c[i*n+j] = s; j = j + 1; } \
                 i = i + 1; } } \
             fn main() -> i32 { return 0; }",
            "mm",
            1,
        );
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 3, "{}", dump_function_loops(&f, "mm"));

        let depths: Vec<u32> = forest.loops.iter().map(|l| l.depth).collect();
        assert_eq!(depths, vec![0, 1, 2], "loops are ordered by header id");

        // Ascending header order == outer to inner here, so nesting is i > j > k.
        assert_eq!(forest.loops[0].parent, None);
        assert_eq!(forest.loops[1].parent, Some(LoopId(0)));
        assert_eq!(forest.loops[2].parent, Some(LoopId(1)));
        assert_eq!(forest.loops[0].children, vec![LoopId(1)]);
        assert_eq!(forest.loops[1].children, vec![LoopId(2)]);
        assert!(forest.loops[2].children.is_empty());

        // Block sets nest.
        let outer: FxHashSet<u32> = forest.loops[0].blocks.iter().map(|b| b.0).collect();
        for l in &forest.loops[1..] {
            for b in &l.blocks {
                assert!(
                    outer.contains(&b.0),
                    "inner block bb{} escapes the outer loop",
                    b.0
                );
            }
        }

        // innermost_first hands back children before parents.
        let order = forest.innermost_first();
        assert_eq!(order, vec![LoopId(2), LoopId(1), LoopId(0)]);

        // The innermost map points every inner block at the k loop.
        for b in &forest.loops[2].blocks {
            assert_eq!(forest.innermost_of(*b), Some(LoopId(2)));
        }
        assert_eq!(forest.innermost_of(forest.loops[0].header), Some(LoopId(0)));
    }

    #[test]
    fn sibling_loops_are_not_nested() {
        let f = opt_mir_of(
            "fn two(n: i32) -> i32 { let mut a: i32 = 0; \
             let mut i: i32 = 0; while i < n { a = a + i; i = i + 1; } \
             let mut j: i32 = 0; while j < n { a = a + j; j = j + 1; } return a; } \
             fn main() -> i32 { return two(3); }",
            "two",
            1,
        );
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 2, "{}", dump_function_loops(&f, "two"));
        assert!(forest
            .loops
            .iter()
            .all(|l| l.parent.is_none() && l.depth == 0));
        assert!(forest.loops.iter().all(|l| l.children.is_empty()));
        // Disjoint block sets.
        let a: FxHashSet<u32> = forest.loops[0].blocks.iter().map(|b| b.0).collect();
        assert!(forest.loops[1].blocks.iter().all(|b| !a.contains(&b.0)));
    }

    #[test]
    fn loop_with_a_conditional_body_keeps_one_header() {
        // `continue`-free, but the body branches: the loop still has one header and one back edge
        // reaching it from the merge block.
        let f = opt_mir_of(
            "fn c(n: i32) -> i32 { let mut a: i32 = 0; let mut i: i32 = 0; \
             while i < n { if i % 2 == 0 { a = a + i; } else { a = a - i; } i = i + 1; } \
             return a; } fn main() -> i32 { return c(5); }",
            "c",
            1,
        );
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1, "{}", dump_function_loops(&f, "c"));
        let l = &forest.loops[0];
        assert!(l.blocks.len() >= 3, "the if-arms are inside the loop");
        assert_eq!(l.exits.len(), 1);
    }

    #[test]
    fn no_loops_in_straight_line_code() {
        let f = opt_mir_of(
            "fn f(a: i32) -> i32 { if a > 0 { return a; } return -a; } \
             fn main() -> i32 { return f(1); }",
            "f",
            1,
        );
        assert!(analyze_function(&f).is_empty());
    }

    // ---- hand-built CFGs, for shapes the front-end never emits ----

    use wukong_mir::{BasicBlock, Inst, Op};

    /// A bare function skeleton: `blocks` are `(params, insts, terminator)`, entry is block 0.
    fn build(
        value_types: Vec<MirType>,
        blocks: Vec<(Vec<u32>, Vec<Inst>, Terminator)>,
    ) -> Function {
        let mut interner = Interner::new();
        Function {
            name: interner.intern("t"),
            params: Vec::new(),
            ret: MirType::Void,
            blocks: blocks
                .into_iter()
                .enumerate()
                .map(|(i, (params, insts, term))| BasicBlock {
                    id: BlockId(i as u32),
                    params: params.into_iter().map(ValueId).collect(),
                    insts,
                    term,
                })
                .collect(),
            value_types,
            entry: BlockId(0),
            vec_kernels: Vec::new(),
        }
    }

    fn br(target: u32) -> Terminator {
        Terminator::Br {
            target: BlockId(target),
            args: Vec::new(),
        }
    }

    fn cond_br(cond: u32, t: u32, e: u32) -> Terminator {
        Terminator::CondBr {
            cond: ValueId(cond),
            then_blk: BlockId(t),
            then_args: Vec::new(),
            else_blk: BlockId(e),
            else_args: Vec::new(),
        }
    }

    fn konst(v: u32, ty: MirType) -> Inst {
        Inst {
            result: Some(ValueId(v)),
            op: Op::ConstInt(1, ty),
        }
    }

    #[test]
    fn irreducible_cycle_is_reported_as_no_loop() {
        //   bb0 -> bb1, bb2 ;  bb1 <-> bb2  (two entries into the cycle: irreducible)
        // Neither bb1 nor bb2 dominates the other, so neither edge is a back edge and there is no
        // *natural* loop. Reporting nothing is the correct, conservative answer — a later pass
        // must not try to rotate or vectorize a cycle it cannot describe.
        let f = build(
            vec![MirType::I1],
            vec![
                (vec![], vec![konst(0, MirType::I1)], cond_br(0, 1, 2)),
                (vec![], vec![], cond_br(0, 2, 3)),
                (vec![], vec![], cond_br(0, 1, 3)),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert!(analyze_function(&f).is_empty());
    }

    #[test]
    fn self_loop_is_its_own_latch() {
        // bb1 branches to itself: header == latch, one block in the loop.
        let f = build(
            vec![MirType::I1],
            vec![
                (vec![], vec![konst(0, MirType::I1)], br(1)),
                (vec![], vec![], cond_br(0, 1, 2)),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        let l = &forest.loops[0];
        assert_eq!(l.header, BlockId(1));
        assert_eq!(l.latches, vec![BlockId(1)]);
        assert_eq!(l.blocks, vec![BlockId(1)]);
        assert_eq!(l.preheader, Some(BlockId(0)));
        assert_eq!(l.exits, vec![(BlockId(1), BlockId(2))]);
    }

    #[test]
    fn two_back_edges_to_one_header_merge_into_one_loop() {
        //   bb0 -> bb1 ; bb1 -> bb2 | bb3 ; bb2 -> bb1 ; bb3 -> bb1 | bb4
        // Both bb2 and bb3 are latches of the same loop.
        let f = build(
            vec![MirType::I1],
            vec![
                (vec![], vec![konst(0, MirType::I1)], br(1)),
                (vec![], vec![], cond_br(0, 2, 3)),
                (vec![], vec![], br(1)),
                (vec![], vec![], cond_br(0, 1, 4)),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        let l = &forest.loops[0];
        assert_eq!(l.header, BlockId(1));
        assert_eq!(l.latches, vec![BlockId(2), BlockId(3)]);
        assert_eq!(l.blocks, vec![BlockId(1), BlockId(2), BlockId(3)]);
        assert_eq!(l.exits, vec![(BlockId(3), BlockId(4))]);
    }

    #[test]
    fn conditional_entry_means_no_preheader() {
        // bb0 branches conditionally straight into the header, so there is nowhere to hoist to.
        let f = build(
            vec![MirType::I1],
            vec![
                (vec![], vec![konst(0, MirType::I1)], cond_br(0, 1, 2)),
                (vec![], vec![], cond_br(0, 1, 2)),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        assert_eq!(forest.loops[0].preheader, None);
        assert_eq!(forest.loops[0].entering, vec![BlockId(0)]);
    }

    #[test]
    fn unreachable_blocks_are_ignored() {
        // bb2 -> bb3 -> bb2 is a cycle, but bb2 is unreachable from the entry. Dominance is
        // undefined there, so the analysis must not report a loop (and must not panic).
        let f = build(
            vec![MirType::I1],
            vec![
                (vec![], vec![konst(0, MirType::I1)], br(1)),
                (vec![], vec![], Terminator::Ret(None)),
                (vec![], vec![], br(3)),
                (vec![], vec![], br(2)),
            ],
        );
        assert!(analyze_function(&f).is_empty());
    }

    #[test]
    fn multiple_exits_are_all_reported() {
        //   bb0 -> bb1 ; bb1 -> bb2 | bb4 ; bb2 -> bb1 | bb4 ; bb4 = exit
        let f = build(
            vec![MirType::I1],
            vec![
                (vec![], vec![konst(0, MirType::I1)], br(1)),
                (vec![], vec![], cond_br(0, 2, 4)),
                (vec![], vec![], cond_br(0, 1, 4)),
                (vec![], vec![], Terminator::Unreachable),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        assert_eq!(
            forest.loops[0].exits,
            vec![(BlockId(1), BlockId(4)), (BlockId(2), BlockId(4))]
        );
    }

    #[test]
    fn affine_expr_algebra() {
        let a = ValueId(10);
        let b = ValueId(11);
        // (2*iv + a + 3) + (iv + b - 1) == 3*iv + a + b + 2
        let x = AffineExpr::iv()
            .scale(2)
            .unwrap()
            .add(&AffineExpr::symbol(a))
            .unwrap()
            .add(&AffineExpr::constant(3))
            .unwrap();
        let y = AffineExpr::iv()
            .add(&AffineExpr::symbol(b))
            .unwrap()
            .add(&AffineExpr::constant(-1))
            .unwrap();
        let s = x.add(&y).unwrap();
        assert_eq!(s.coeff, Coeff::Const(3));
        assert_eq!(s.terms, vec![(a, 1), (b, 1)]);
        assert_eq!(s.konst, 2);

        // Terms that cancel are dropped.
        let z = AffineExpr::symbol(a).sub(&AffineExpr::symbol(a)).unwrap();
        assert!(z.terms.is_empty());
        assert_eq!(z.as_const(), Some(0));

        // a * (iv + 1) == a*iv + a   (the row-pitch shape)
        let p = AffineExpr::symbol(a)
            .mul(&AffineExpr::iv().add(&AffineExpr::constant(1)).unwrap())
            .unwrap();
        assert_eq!(p.coeff, Coeff::Sym(a, 1));
        assert_eq!(p.terms, vec![(a, 1)]);
        assert_eq!(p.konst, 0);
        assert_eq!(
            p.coeff.as_const(),
            None,
            "a symbolic stride is not a constant"
        );

        // a * b (two invariants) is not representable.
        assert!(AffineExpr::symbol(a).mul(&AffineExpr::symbol(b)).is_none());
        // a*iv * b*iv is quadratic.
        assert!(p.mul(&p).is_none());
        // Overflow is rejected, never wrapped.
        assert!(AffineExpr::constant(i64::MAX).scale(2).is_none());
    }

    // ---- induction variables and trip counts ----

    /// The only loop of a function with exactly one.
    fn only_loop(forest: &LoopForest) -> &NaturalLoop {
        assert_eq!(forest.loops.len(), 1, "expected exactly one loop");
        &forest.loops[0]
    }

    #[test]
    fn basic_iv_step_and_symbolic_trip_count() {
        let f = opt_mir_of(
            "fn s(n: i32) -> i32 { let mut a: i32 = 0; let mut i: i32 = 0; \
             while i < n { a = a + i; i = i + 1; } return a; } \
             fn main() -> i32 { return s(4); }",
            "s",
            1,
        );
        let forest = analyze_function(&f);
        let l = only_loop(&forest);
        let iv = l.primary().expect("the exit test names an IV");
        assert_eq!(iv.step, IvStep::Const(1));
        assert!(iv.ty.is_int());
        // `i < n` with a runtime `n`: the pieces are known, the count is not.
        match &l.trip {
            TripCount::Affine {
                step,
                inclusive,
                signed,
                ..
            } => {
                assert_eq!(*step, 1);
                assert!(!*inclusive);
                assert!(*signed);
            }
            other => panic!(
                "expected an affine trip count, got {other:?}: {}",
                dump_function_loops(&f, "s")
            ),
        }
        // `i + 1` is derived from `i`.
        assert!(l
            .derived
            .iter()
            .any(|d| d.value == iv.next && d.expr.coeff == Coeff::Const(1)));
    }

    #[test]
    fn constant_trip_counts_for_every_predicate_and_step() {
        // (source, expected iterations). Written as `while` loops so the front-end cannot fold
        // them into a recognized kernel.
        let cases: [(&str, u64, i64); 6] = [
            (
                "let mut i: i32 = 0; while i < 100 { a = a + i; i = i + 1; }",
                100,
                1,
            ),
            (
                "let mut i: i32 = 0; while i <= 9 { a = a + i; i = i + 1; }",
                10,
                1,
            ),
            (
                "let mut i: i32 = 0; while i < 10 { a = a + i; i = i + 3; }",
                4,
                3,
            ),
            (
                "let mut i: i32 = 0; while i < 0 { a = a + i; i = i + 1; }",
                0,
                1,
            ),
            (
                "let mut i: i32 = 10; while i > 0 { a = a + i; i = i - 1; }",
                10,
                -1,
            ),
            (
                "let mut i: i32 = 10; while i >= 0 { a = a + i; i = i - 2; }",
                6,
                -2,
            ),
        ];
        for (body, want, step) in cases {
            let src = format!("fn t() -> i32 {{ let mut a: i32 = 0; {body} return a; }} fn main() -> i32 {{ return t(); }}");
            let f = opt_mir_of(&src, "t", 1);
            let forest = analyze_function(&f);
            let l = only_loop(&forest);
            assert_eq!(
                l.primary().map(|iv| iv.step),
                Some(IvStep::Const(step)),
                "step for `{body}`\n{}",
                dump_function_loops(&f, "t")
            );
            assert_eq!(
                l.trip,
                TripCount::Const(want),
                "trip for `{body}`\n{}",
                dump_function_loops(&f, "t")
            );
        }
    }

    #[test]
    fn for_range_over_a_constant_is_a_constant_trip_count() {
        let f = opt_mir_of(
            "fn t() -> i32 { let mut a: i32 = 0; for i in 0..64 { a = a + (i as i32); } return a; } \
             fn main() -> i32 { return t(); }",
            "t",
            1,
        );
        let forest = analyze_function(&f);
        let l = only_loop(&forest);
        assert_eq!(
            l.trip,
            TripCount::Const(64),
            "{}",
            dump_function_loops(&f, "t")
        );
    }

    #[test]
    fn runtime_step_is_an_iv_with_no_trip_count() {
        let f = opt_mir_of(
            "fn t(n: i32, s: i32) -> i32 { let mut a: i32 = 0; let mut i: i32 = 0; \
             while i < n { a = a + i; i = i + s; } return a; } \
             fn main() -> i32 { return t(4, 2); }",
            "t",
            1,
        );
        let forest = analyze_function(&f);
        let l = only_loop(&forest);
        let iv = l.primary().expect("still a basic IV");
        assert!(
            matches!(iv.step, IvStep::Invariant(_)),
            "a loop-invariant step is still an IV: {}",
            dump_function_loops(&f, "t")
        );
        assert_eq!(
            l.trip,
            TripCount::Unknown,
            "an unknown step means an unknown count"
        );
    }

    #[test]
    fn a_second_counter_is_an_iv_but_not_the_primary() {
        let f = opt_mir_of(
            "fn t(n: i32) -> i32 { let mut a: i32 = 0; let mut i: i32 = 0; let mut j: i32 = 5; \
             while i < n { a = a + j; i = i + 1; j = j + 3; } return a; } \
             fn main() -> i32 { return t(4); }",
            "t",
            1,
        );
        let forest = analyze_function(&f);
        let l = only_loop(&forest);
        assert_eq!(
            l.ivs.len(),
            2,
            "both counters are basic IVs: {}",
            dump_function_loops(&f, "t")
        );
        let primary = l.primary().expect("the exit test picks one");
        assert_eq!(primary.step, IvStep::Const(1), "the exit test walks `i`");
        assert!(l.ivs.iter().any(|iv| iv.step == IvStep::Const(3)));
        // The secondary IV is a block parameter, so it is deliberately NOT expressed as an affine
        // function of the primary — see `affine_uncached`.
        let secondary = l.ivs.iter().find(|iv| iv.step == IvStep::Const(3)).unwrap();
        assert!(!l.derived.iter().any(|d| d.value == secondary.value));
    }

    #[test]
    fn derived_ivs_record_constant_and_symbolic_strides() {
        let f = opt_mir_of(
            "fn mm(a: [f32; 64], b: [f32; 64], mut c: [f32; 64], n: i32) { \
               let mut i: i32 = 0; \
               while i < n { let mut j: i32 = 0; \
                 while j < n { let mut s: f32 = 0.0; let mut k: i32 = 0; \
                   while k < n { s = s + a[i*n+k] * b[k*n+j]; k = k + 1; } \
                   c[i*n+j] = s; j = j + 1; } \
                 i = i + 1; } } \
             fn main() -> i32 { return 0; }",
            "mm",
            1,
        );
        let forest = analyze_function(&f);
        let inner = forest
            .loops
            .iter()
            .max_by_key(|l| l.depth)
            .expect("the k loop");
        assert_eq!(inner.depth, 2);
        let dump = dump_function_loops(&f, "mm");
        // `a[i*n + k]`: unit stride in k, with the row base `i*n` carried as an invariant term.
        assert!(
            inner
                .derived
                .iter()
                .any(|d| d.expr.coeff == Coeff::Const(1) && d.expr.terms.len() == 1),
            "expected a unit-stride derived IV\n{dump}"
        );
        // `b[k*n + j]`: the stride is the runtime pitch `n`, so it is symbolic, not constant.
        assert!(
            inner.derived.iter().any(
                |d| matches!(d.expr.coeff, Coeff::Sym(..)) && d.expr.coeff.as_const().is_none()
            ),
            "expected a symbolic-stride derived IV\n{dump}"
        );
    }

    #[test]
    fn a_scaled_index_has_the_scaled_stride() {
        let src = "fn t(x: [f32; 64], mut o: [f32; 64], n: i32) { let mut i: i32 = 0; \
                   while i <= n - 1 { o[2*i] = x[i] + 1.0; i = i + 2; } } \
                   fn main() -> i32 { return 0; }";
        for level in [1, 2] {
            let f = opt_mir_of(src, "t", level);
            let forest = analyze_function(&f);
            let l = only_loop(&forest);
            let dump = dump_function_loops(&f, "t");
            assert_eq!(
                l.primary().map(|iv| iv.step),
                Some(IvStep::Const(2)),
                "-O{level}\n{dump}"
            );
            // `2*i` walks 2 elements per iteration.
            assert!(
                l.derived.iter().any(|d| d.expr.coeff == Coeff::Const(2)),
                "expected a stride-2 derived IV at -O{level}\n{dump}"
            );
        }

        // The trip count needs endpoints a consumer can evaluate in the *preheader*. At `-O2` LICM
        // has hoisted `n - 1` out, so the count is reported; at `-O1` the bound is still computed
        // inside the loop and the analysis declines rather than hand back a value that is not
        // available where the count would be computed.
        let f1 = opt_mir_of(src, "t", 1);
        assert_eq!(analyze_function(&f1).loops[0].trip, TripCount::Unknown);

        let f2 = opt_mir_of(src, "t", 2);
        let forest = analyze_function(&f2);
        let dump = dump_function_loops(&f2, "t");
        // `i <= n - 1` is reported inclusive against `n - 1`, never normalized to `n`: that
        // addition can overflow.
        match &forest.loops[0].trip {
            TripCount::Affine {
                inclusive, step, ..
            } => {
                assert!(*inclusive, "{dump}");
                assert_eq!(*step, 2);
            }
            other => panic!("expected an affine trip count, got {other:?}\n{dump}"),
        }
    }

    #[test]
    fn a_loop_with_no_recognizable_exit_test_still_reports_its_iv() {
        // Two exits (an early `return` inside the body), so no trip count — but the counter is
        // still a basic IV and still the primary, because it is the only constant-step one.
        let f = opt_mir_of(
            "fn t(x: [i32; 64], n: i32) -> i32 { let mut i: i32 = 0; \
             while i < n { if x[i] < 0 { return i; } i = i + 1; } return -1; } \
             fn main() -> i32 { return 0; }",
            "t",
            1,
        );
        let forest = analyze_function(&f);
        let l = only_loop(&forest);
        let dump = dump_function_loops(&f, "t");
        assert!(
            l.exits.len() >= 2,
            "the early return is a second exit\n{dump}"
        );
        assert_eq!(l.trip, TripCount::Unknown, "{dump}");
        assert_eq!(
            l.primary().map(|iv| iv.step),
            Some(IvStep::Const(1)),
            "{dump}"
        );
    }

    #[test]
    fn zext_is_not_treated_as_transparent() {
        // Hand-built: `q = zext(i)` then `p = gep(base, q)`. `sext` would be folded through;
        // `zext` must not be, because a negative narrow value becomes a huge positive one.
        let ctx_expr = |cast: CastKind| {
            //  v0: i32 base ptr slot (unused), v1: i1 cond, v2: i32 IV param, v3: i32 one,
            //  v4: i32 next, v5: i64 widened iv

            build(
                vec![
                    MirType::I1,
                    MirType::I32,
                    MirType::I32,
                    MirType::I32,
                    MirType::I64,
                ],
                vec![
                    (
                        vec![],
                        vec![
                            Inst {
                                result: Some(ValueId(0)),
                                op: Op::ConstInt(1, MirType::I1),
                            },
                            Inst {
                                result: Some(ValueId(1)),
                                op: Op::ConstInt(0, MirType::I32),
                            },
                            Inst {
                                result: Some(ValueId(3)),
                                op: Op::ConstInt(1, MirType::I32),
                            },
                        ],
                        Terminator::Br {
                            target: BlockId(1),
                            args: vec![ValueId(1)],
                        },
                    ),
                    (
                        vec![2],
                        vec![],
                        Terminator::CondBr {
                            cond: ValueId(0),
                            then_blk: BlockId(2),
                            then_args: vec![],
                            else_blk: BlockId(3),
                            else_args: vec![],
                        },
                    ),
                    (
                        vec![],
                        vec![Inst {
                            result: Some(ValueId(4)),
                            op: Op::Cast(cast, ValueId(2), MirType::I64),
                        }],
                        Terminator::Br {
                            target: BlockId(1),
                            args: vec![ValueId(2)],
                        },
                    ),
                    (vec![], vec![], Terminator::Ret(None)),
                ],
            )
        };
        // The IV never advances here (the latch passes the param straight back), so instead build
        // the real shape below; this test only checks the cast rule, via `affine_of` directly.
        let f = ctx_expr(CastKind::SExt);
        let forest = analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        let mut ctx = LoopCtx::new(&f, &forest.loops[0]);
        ctx.primary = Some(ValueId(2));
        let mut memo = FxHashMap::default();
        assert_eq!(
            ctx.affine_of(ValueId(4), &mut memo, 0).map(|e| e.coeff),
            Some(Coeff::Const(1)),
            "sext of the IV is the IV"
        );

        let f = ctx_expr(CastKind::ZExt);
        let forest = analyze_function(&f);
        let mut ctx = LoopCtx::new(&f, &forest.loops[0]);
        ctx.primary = Some(ValueId(2));
        let mut memo = FxHashMap::default();
        assert!(
            ctx.affine_of(ValueId(4), &mut memo, 0).is_none(),
            "zext of the IV is not affine in it"
        );
    }

    // ---- stage 3: memory accesses ------------------------------------------------------------

    /// The *kernel* loop of `func` at `-O1`: the one with the most decomposed memory accesses.
    ///
    /// A source line like `let mut a: [f32; 64] = [1.0; 64]` lowers to its own initializer loop, so
    /// "the function's only loop" is usually wrong; and the loop under test is always the one that
    /// touches the most memory. Ties break toward the later header, which is the body of a
    /// `let`-then-loop function.
    fn kernel_loop(src: &str, func: &str) -> (Function, NaturalLoop) {
        let f = opt_mir_of(src, func, 1);
        let forest = analyze_function(&f);
        let best = forest
            .loops
            .iter()
            .max_by_key(|l| {
                (
                    l.accesses
                        .iter()
                        .filter(|a| matches!(a.addr, AddrForm::Affine { .. }))
                        .count(),
                    l.header.0,
                )
            })
            .unwrap_or_else(|| panic!("no loop in {func}"))
            .clone();
        (f, best)
    }

    #[test]
    fn a_slice_access_decomposes_to_base_plus_iv() {
        // The `while` spelling on purpose: `for i in 0..n { o[i] = x[i] * 2.0; }` is matched by a
        // `mir_build` recognizer and reaches the optimizer as a `wukong_velem_f32` call, with no
        // loop left to analyze.
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 2.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        // Two data accesses; the `[]f32` base re-loads are `Invariant`, not `Affine`.
        let affine: Vec<&MemAccess> = l
            .accesses
            .iter()
            .filter(|a| matches!(a.addr, AddrForm::Affine { .. }))
            .collect();
        assert_eq!(affine.len(), 2, "one load and one store: {:?}", l.accesses);
        for a in &affine {
            assert_eq!(a.addr.const_stride(), Some(1), "unit stride: {:?}", a.addr);
            assert_eq!(a.value_ty, MirType::F32);
        }
        assert_eq!(affine[0].kind, AccessKind::Load);
        assert_eq!(affine[1].kind, AccessKind::Store);
    }

    #[test]
    fn a_constant_offset_shows_up_in_the_index_constant() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { \
                   for i in 0..n { o[i] = x[i + 3]; } } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        let konsts: Vec<i64> = l
            .accesses
            .iter()
            .filter_map(|a| match &a.addr {
                AddrForm::Affine { index, .. } => Some(index.konst),
                _ => None,
            })
            .collect();
        assert_eq!(konsts, vec![3, 0], "load at iv+3, store at iv+0");
    }

    #[test]
    fn a_data_dependent_index_is_unknown() {
        let src = "fn k(idx: []i32, x: []f32, mut o: []f32, n: i64) { \
                   for i in 0..n { o[i] = x[idx[i] as i64]; } } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        assert!(
            l.accesses
                .iter()
                .any(|a| matches!(a.addr, AddrForm::Unknown)),
            "a gathered address is not affine: {:?}",
            l.accesses
        );
        assert_eq!(l.dep.memory, MemDep::Carried);
        assert!(!l.is_vectorizable_shape());
    }

    // ---- stage 4: carried values -------------------------------------------------------------

    #[test]
    fn a_float_sum_is_a_non_reassociable_reduction() {
        let src = "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; \
                   for i in 0..n { s = s + x[i]; } return s; } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        assert_eq!(l.reductions.len(), 1, "{:?}", l.carried);
        assert_eq!(l.reductions[0].kind, RedKind::FAdd);
        assert!(
            !l.reductions[0].reassociable,
            "IEEE addition is not associative"
        );
        assert!(l.is_vectorizable_shape());
    }

    #[test]
    fn an_integer_sum_is_a_reassociable_reduction() {
        let src = "fn k(x: []i32, n: i64) -> i32 { let mut s: i32 = 0; \
                   for i in 0..n { s = s + x[i]; } return s; } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        assert_eq!(l.reductions.len(), 1, "{:?}", l.carried);
        assert_eq!(l.reductions[0].kind, RedKind::Add);
        assert!(l.reductions[0].reassociable, "integer add is associative");
    }

    /// `s = s + w[i]*d*d` reaches MIR as `s = fma(w[i]*d, d, s)` — the front end contracts the
    /// multiply into the accumulate. A consumer that missed this shape would see a `Recurrence` and
    /// decline every dot product in the corpus.
    #[test]
    fn a_contracted_fma_accumulate_is_a_reduction() {
        let src = "fn k(w: []f32, a: []f32, b: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; \
                   for i in 0..n { let d: f32 = a[i] - b[i]; s = s + w[i] * d * d; } return s; } \
                   fn main() -> i32 { return 0; }";
        let (f, l) = kernel_loop(src, "k");
        let l = &l;
        assert_eq!(l.reductions.len(), 1, "{:?}", l.carried);
        let r = &l.reductions[0];
        assert_eq!(r.kind, RedKind::FAdd);
        assert!(r.addend.is_none(), "the addend is the fused product");
        let (x, y) = r.fma_factors.expect("fma factors");
        assert!(matches!(f.value_type(x), MirType::F32));
        assert!(matches!(f.value_type(y), MirType::F32));
    }

    #[test]
    fn a_true_recurrence_is_not_a_reduction() {
        let src = "fn k(a: []f32, b: []f32, mut o: []f32, n: i64) -> f32 { let mut h: f32 = 0.0; \
                   for i in 0..n { h = a[i] * h + b[i]; o[i] = h; } return h; } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        assert!(
            l.carried.iter().any(|c| c.kind == Carried::Recurrence),
            "h = a*h + b is a recurrence: {:?}",
            l.carried
        );
        assert!(!l.is_vectorizable_shape(), "must decline the SSM shape");
    }

    /// An accumulator whose partial value is *read* mid-loop cannot be reordered, even though the
    /// combine itself looks like a textbook reduction.
    #[test]
    fn an_observed_partial_sum_is_not_a_reduction() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; \
                   for i in 0..n { o[i] = s; s = s + x[i]; } return s; } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        assert!(l.reductions.is_empty(), "{:?}", l.carried);
        assert!(l.carried.iter().any(|c| c.kind == Carried::Recurrence));
    }

    /// A conditionally-updated accumulator reaches the latch as a *merge parameter*, not as a
    /// combining instruction, so SSA alone keeps it out of the reduction set.
    #[test]
    fn a_conditional_accumulate_is_not_a_reduction() {
        let src = "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; \
                   for i in 0..n { if x[i] > 0.0 { s = s + x[i]; } } return s; } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        let l = &l;
        assert!(l.reductions.is_empty(), "{:?}", l.carried);
        assert!(!l.is_vectorizable_shape());
    }

    // ---- stage 4: dependence -----------------------------------------------------------------

    #[test]
    fn a_call_in_the_loop_forces_a_carried_verdict() {
        let src = "fn g(x: f32) -> f32 { return x * 2.0; } \
                   fn k(x: []f32, mut o: []f32, n: i64) { \
                   for i in 0..n { o[i] = g(x[i]); } } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        assert_eq!(l.dep.memory, MemDep::Carried);
    }

    /// Two `[]f32` parameters could be the same buffer; only a runtime check can rule it out, and
    /// the verdict says so rather than quietly claiming independence.
    #[test]
    fn two_slice_parameters_need_a_runtime_check() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 2.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (_, l) = kernel_loop(src, "k");
        assert_eq!(l.dep.memory, MemDep::IndependentIfBasesDisjoint);
    }

    /// Two distinct stack arrays are distinct objects: no check needed, no assumption taken.
    #[test]
    fn two_distinct_allocas_are_provably_independent() {
        let src = "fn main() -> i32 { let mut a: [f32; 64] = [0.0; 64]; \
                   let mut b: [f32; 64] = [0.0; 64]; \
                   for i in 0..64 { b[i] = a[i] * 2.0; } \
                   return b[3] as i32; }";
        let (_, l) = kernel_loop(src, "main");
        assert_eq!(l.dep.memory, MemDep::Independent);
        assert!(l.is_vectorizable_shape());
    }

    /// `a[i] = a[i-1] + 1` writes what the next iteration reads: distance 1, a real carried
    /// dependence. This is the case a vectorizer must never widen.
    #[test]
    fn a_nonzero_distance_on_one_object_is_carried() {
        let src = "fn main() -> i32 { let mut a: [f32; 64] = [1.0; 64]; \
                   for i in 1..64 { a[i] = a[i - 1] + 1.0; } \
                   return a[63] as i32; }";
        let (_, l) = kernel_loop(src, "main");
        assert_eq!(l.dep.memory, MemDep::Carried);
        assert!(!l.is_vectorizable_shape());
    }

    /// A load and a store at the *same* address in the same iteration is a loop-*independent*
    /// dependence, which widening preserves — `a[i] = a[i] + 1` must stay vectorizable.
    #[test]
    fn a_read_modify_write_of_one_element_is_independent() {
        let src = "fn main() -> i32 { let mut a: [f32; 64] = [1.0; 64]; \
                   for i in 0..64 { a[i] = a[i] + 1.0; } \
                   return a[63] as i32; }";
        let (_, l) = kernel_loop(src, "main");
        assert_eq!(l.dep.memory, MemDep::Independent);
        assert!(l.is_vectorizable_shape());
    }
}
