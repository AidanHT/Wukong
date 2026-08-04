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

use wukong_mir::{BlockId, Function, MirType, Program, Terminator, ValueId};
use wukong_span::Interner;

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::{cfg, dom};

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
    pub fn is_vectorizable_shape(&self) -> bool {
        matches!(
            self.dep.memory,
            MemDep::Independent | MemDep::IndependentIfBasesStable
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
    /// `i_next = i + s` with a loop-invariant but non-constant `s`.
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
            let Some(v) = sym_side.as_symbol() else { continue };
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
    SMax,
    SMin,
    UMax,
    UMin,
    FMax,
    FMin,
}

impl RedKind {
    /// May the partial sums be reassociated — i.e. may a vectorizer keep N independent
    /// accumulators and fold them at the end?
    ///
    /// Integer `add`/`mul`/bitwise and min/max are associative, so yes. Float `add`/`mul` are
    /// **not**: `(a+b)+c != a+(b+c)` in IEEE-754, and Wukong's interpreter is the language spec,
    /// so reassociating one changes the program's answer and breaks the differential gate. A
    /// vectorizer may still widen the *body* that feeds such a reduction and keep the accumulate
    /// serial — which is exactly what `gcc -O3` does without `-ffast-math`.
    pub fn reassociable(self) -> bool {
        !matches!(self, RedKind::FAdd | RedKind::FMul)
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
        analyze_loop(f, &mut forest, i);
    }
    forest
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

        loops.push(NaturalLoop {
            id: LoopId(i as u32),
            header: BlockId(*h),
            latches: latches.into_iter().map(BlockId).collect(),
            blocks: blocks.into_iter().map(BlockId).collect(),
            entering: entering.into_iter().map(BlockId).collect(),
            preheader,
            exits,
            parent: None,
            children: Vec::new(),
            depth: 0,
            ivs: Vec::new(),
            primary_iv: None,
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
// Stage 2+: per-loop value analyses (filled in by later stages of this module)
// ---------------------------------------------------------------------------------------------

fn analyze_loop(_f: &Function, _forest: &mut LoopForest, _idx: usize) {
    // Filled in by the induction-variable, memory-access and dependence stages.
}

// ---------------------------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------------------------

/// The branch arguments a terminator passes to `target`. `None` when the answer is ambiguous — a
/// `cond_br` whose two arms both go to `target` with different argument lists.
#[allow(dead_code)] // used by the induction-variable stage
fn args_to<'a>(t: &'a Terminator, target: BlockId) -> Option<&'a [ValueId]> {
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
    let _ = writeln!(
        out,
        "== loops in fn {name}: {} ==",
        forest.loops.len()
    );
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
    }
    out
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
                assert!(outer.contains(&b.0), "inner block bb{} escapes the outer loop", b.0);
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
        assert!(forest.loops.iter().all(|l| l.parent.is_none() && l.depth == 0));
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
        assert_eq!(p.coeff.as_const(), None, "a symbolic stride is not a constant");

        // a * b (two invariants) is not representable.
        assert!(AffineExpr::symbol(a).mul(&AffineExpr::symbol(b)).is_none());
        // a*iv * b*iv is quadratic.
        assert!(p.mul(&p).is_none());
        // Overflow is rejected, never wrapped.
        assert!(AffineExpr::constant(i64::MAX).scale(2).is_none());
    }
}
