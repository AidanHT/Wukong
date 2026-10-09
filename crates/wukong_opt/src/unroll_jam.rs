//! Unroll-and-jam: run four outer iterations of a two-deep loop nest side by side, so an inner loop
//! that carries a serial chain gets four independent chains to overlap.
//!
//! # Why
//!
//! An inner loop that carries a recurrence (`h = a*h + x[r,t]`, an IIR filter, a hash, a running
//! state) or a reduction this compiler may not reassociate (a strict-order float sum down a column)
//! is one dependency chain: every iteration waits for the previous one's multiply-add, however
//! wide the machine is. The vectorizer cannot help — lanes of one chain are not independent, and
//! a float reduction is never reassociated (`vectorize.rs`, "Float semantics"). The only bit-exact
//! parallelism is *across* the outer iterations: rows `r .. r+3` are four separate chains. GCC's
//! `-floop-unroll-and-jam` and LLVM's `LoopUnrollAndJam` exploit exactly that, and the repo already
//! measured it for one recognized spelling — `wukong_lrscan_f32` interleaves four rows for
//! ~1.6-1.9x (`wukong_xbench` notes). This pass is the general form, for every spelling the
//! recognizers miss.
//!
//! # Shape
//!
//! The canonical two-deep nest `LoopCanon` produces, exactly five blocks:
//!
//! ```text
//!   OP:            ...                                   br OH(r0, ..)
//!   OH(r, p..):    c = cmp.slt r, R                      cond_br c, OB, OE(..)
//!   OB:            <row prologue>                        br IH(t0, ..)
//!   IH(t, q..):    ci = cmp.slt t, T                     cond_br ci, IB, IX(..)
//!   IB:            <inner body>                          br IH(t + st, q'..)
//!   IX(..):        <row epilogue>                        br OH(r + s, p'..)
//! ```
//!
//! `OH` and `IH` compute their exit test and nothing else; `r` and `t` are constant positive-step
//! induction variables; `T` and `t0` do not depend on the row, so every row runs the inner loop the
//! same number of times; and nothing in the nest calls, allocates, synchronizes, or divides by a
//! value that could trap. A carried outer value other than `r` (`total = total*10 + h`) may only be
//! read in `IX`, where the jammed rows still update it in row order.
//!
//! # Transform
//!
//! ```text
//!   OP:            lim = R - 3s, wrap- and alias-guarded (below)
//!   OH(r, p..):    u = cmp.slt r, lim                    cond_br u, J0, RH(r, p..)
//!   J0:            OB for r, r+s, r+2s, r+3s             br JH(t0, q0.., q1.., q2.., q3..)
//!   JH(t, q0..q3): ci = cmp.slt t, T                     cond_br ci, JB, JX(..)
//!   JB:            IB for each of the four rows          br JH(t + st, ..)
//!   JX(..):        IX for each of the four rows, in order  br OH(r + 4s, p'..)
//!   RH(r, p..):    the original OH test → OB/IH/IB/IX, the original nest, now the remainder
//! ```
//!
//! Each row keeps its own copy of every inner-loop value; only the inner counter `t` is shared, and
//! it is identical across rows by the trip-count condition. Every row therefore computes exactly
//! the values, in exactly the order, it computed before — no operation is reassociated, so the
//! result is bit-identical on every backend, NaN payloads and signed zeros included. What changes
//! is the *interleaving* of different rows' memory operations, and that is what the legality
//! proof below is about. The last `trip mod 4` rows, and every row when a guard fails, run in the
//! untouched original nest (`RH` onward).
//!
//! The limit `R - 3s` is computed with the same wrap guard `unroll.rs` uses: if the subtraction
//! wraps, the limit becomes a sentinel no counter is below and every row goes to the remainder.
//! Given `r < lim`, rows `r .. r+3s` are all rows the original loop runs, and `r + 4s` is the
//! increment the original computes after row `r + 3s`.
//!
//! # Legality: rows must not touch each other's memory
//!
//! Jamming runs row `r+1`'s prologue before row `r`'s inner loop, and row `r+1`'s iteration `t`
//! before row `r`'s iteration `t+1`. That is a legal reordering exactly when no location one row
//! writes is read or written by another row. Every access's address is decomposed as
//! `base + α·r + β·t + Σ invariants + k` ([`Idx2`]), and every object the nest stores to must
//! follow one of two layouts in which distinct rows provably touch distinct elements:
//!
//! * **Row slabs** — `α` is a pitch `P` and every access's offset within the row, `β·t + k`, lies
//!   in `[0, P)`: row `r` owns `[S + P·r, S + P·r + P)`. `P` is a positive constant (with constant
//!   inner bounds, checked exactly), or the inner loop's own runtime bound `T` (`x[r*T + t]` for
//!   `t in t0..T`, `t0 + k >= 0`, `k <= 0`) — the shape of every row-major tensor
//!   `Tensor[f32, R, T]`, symbolic dimensions included. `o[r]` is a row slab of pitch 1.
//! * **Column slabs** — `α = 1` and `β` is a pitch `B` with every row in `[0, B)`: the time-major
//!   layout `x[t*D + d]` with the outer loop over `d in 0..D`. Two rows differ by less than the
//!   pitch, so they never meet. Every access is in the inner loop, at one common offset.
//!
//! Loads of an object the nest never stores to need no layout. Two *different* objects may still be
//! one buffer at run time (two tensor parameters the caller aliased), so every stored object and
//! every other object the nest touches that is not provably distinct (two stack slots, or a
//! `@noalias` promise) is checked once, in the preheader: the address ranges the whole nest touches
//! in each must be ordered and disjoint, or every row runs in the original nest. At most
//! [`MAX_GUARD_PAIRS`] such checks are paid, LoopVectorize's runtime-check threshold. A `[]T`
//! slice's data pointer, re-read inside the nest from its fat pointer, is stable because nothing in
//! the nest can store a pointer, a byte, or anything not through a typed element `gep` (the same
//! no-type-punning argument `bce.rs` makes).
//!
//! # Profitability, scheduling and the switch
//!
//! Only nests whose inner loop carries a recurrence or a reduction are jammed — a loop without a
//! chain gains nothing the vectorizer and the unroller do not already give it — with an inner body
//! of at most [`MAX_INNER_INSTS`] instructions. It runs once per function, from
//! [`crate::unroll_program`], ahead of partial unrolling and inside the same cleanup sweep, after
//! the main pipeline (vectorizer included) has reached its fixpoint, so it only ever sees loops the
//! vectorizer left scalar. `WUKONG_NO_UNROLL_JAM=1` turns it off; `WUKONG_VEC_TRACE=1` prints one
//! `unroll-jam: JAM` line per nest. No speed-up is claimed here: the gain has to be measured with a
//! same-run A/B against `WUKONG_NO_UNROLL_JAM`.

use std::sync::OnceLock;

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId,
};

use crate::alias::PromisedBases;
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{self, Carried, Coeff, IvStep, LoopForest, MemBase, NaturalLoop, Stages};
use crate::{cfg, dom, each_op_use, each_term_use, map_op_uses, map_term_uses};

/// Outer iterations per pass of the jammed loop.
const FACTOR: i64 = 4;
/// The largest inner body jammed: four copies of it make the jammed inner loop.
const MAX_INNER_INSTS: usize = 32;
/// The largest row prologue plus epilogue jammed.
const MAX_OUTER_INSTS: usize = 48;
/// Runtime alias checks one nest may pay on entry (LoopVectorize's default threshold).
const MAX_GUARD_PAIRS: usize = 8;
/// Nests per function, so a pathological function cannot blow up compile time.
const MAX_NESTS_PER_FN: usize = 8;
/// A function may not grow past this many instructions in total.
const MAX_FN_INSTS: usize = 6000;
/// How deep an index expression is followed.
const DEPTH_LIMIT: u32 = 32;
/// How many nested `gep`s one address may fold.
const GEP_CHAIN_LIMIT: u32 = 4;

/// `WUKONG_NO_UNROLL_JAM=1` disables the pass entirely (read once).
fn disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("WUKONG_NO_UNROLL_JAM").is_ok_and(|v| v != "0"))
}

/// Unroll-and-jam every eligible nest of `f`. Returns whether anything changed.
pub(crate) fn jam_function(f: &mut Function) -> bool {
    if disabled() || !may_hold_a_nest(f) {
        return false;
    }
    let trace = std::env::var_os("WUKONG_VEC_TRACE").is_some();
    // The jammed loop keeps the original outer header, and the remainder is a copy of the original
    // nest: both would match again, so both headers are retired once used.
    let mut banned: FxHashSet<u32> = FxHashSet::default();
    let mut changed = false;
    for _ in 0..MAX_NESTS_PER_FN {
        if f.blocks.iter().map(|b| b.insts.len()).sum::<usize>() >= MAX_FN_INSTS {
            break;
        }
        let Some(nest) = find_nest(f, &banned) else {
            break;
        };
        if trace {
            eprintln!(
                "unroll-jam: JAM bb{} x{} guards={}",
                nest.oh.0,
                FACTOR,
                nest.pairs.len()
            );
        }
        let rh = apply(f, &nest);
        banned.insert(nest.oh.0);
        banned.insert(rh.0);
        changed = true;
    }
    changed
}

/// A cheap pre-filter: a nest needs at least six blocks, two conditional branches, and an inner
/// loop of exactly the two-block shape [`plan`] accepts — a header that conditionally enters a
/// block whose only exit branches straight back. One linear scan with no CFG analysis, so a
/// function without such a loop never pays for the dominator tree [`find_nest`] builds.
fn may_hold_a_nest(f: &Function) -> bool {
    let two_block_loop = f.blocks.iter().any(|hb| {
        let Terminator::CondBr { then_blk, .. } = &hb.term else {
            return false;
        };
        then_blk.0 != hb.id.0
            && matches!(
                &f.blocks[then_blk.0 as usize].term,
                Terminator::Br { target, .. } if target.0 == hb.id.0
            )
    });
    f.blocks.len() >= 6
        && two_block_loop
        && f.blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
            .count()
            >= 2
}

// ---------------------------------------------------------------------------------------------
// Index expressions over two induction variables
// ---------------------------------------------------------------------------------------------

/// An integer index `r·α + t·β + Σ (value · mult) + k`, where `r` is the outer induction variable,
/// `t` the inner one and every `value` is defined outside the nest. The same no-wraparound
/// assumption as [`crate::loop_info::AffineExpr`]: `sext` is transparent, and the arithmetic is
/// assumed not to overflow in the source width.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Idx2 {
    r: Coeff,
    t: Coeff,
    /// Ascending by value id, merged, never a zero multiplier.
    terms: Vec<(ValueId, i64)>,
    k: i64,
}

/// One canonical spelling per coefficient, so two equal coefficients compare equal.
fn canon(c: Coeff) -> Coeff {
    match c {
        Coeff::Const(0) | Coeff::Sym(_, 0) => Coeff::Zero,
        other => other,
    }
}

impl Idx2 {
    fn konst(k: i64) -> Idx2 {
        Idx2 {
            k,
            ..Idx2::default()
        }
    }

    fn sym(v: ValueId) -> Idx2 {
        Idx2 {
            terms: vec![(v, 1)],
            ..Idx2::default()
        }
    }

    /// Canonical coefficients and merged, sorted terms; `None` when a merged multiplier overflows.
    fn normalize(mut self) -> Option<Idx2> {
        self.r = canon(self.r);
        self.t = canon(self.t);
        self.terms.sort_unstable_by_key(|(v, _)| v.0);
        let mut merged: Vec<(ValueId, i64)> = Vec::with_capacity(self.terms.len());
        for (v, m) in self.terms {
            match merged.last_mut() {
                Some((pv, pm)) if *pv == v => *pm = pm.checked_add(m)?,
                _ => merged.push((v, m)),
            }
        }
        merged.retain(|(_, m)| *m != 0);
        self.terms = merged;
        Some(self)
    }

    fn add(&self, o: &Idx2) -> Option<Idx2> {
        let mut terms = self.terms.clone();
        terms.extend_from_slice(&o.terms);
        Idx2 {
            r: self.r.add(o.r)?,
            t: self.t.add(o.t)?,
            terms,
            k: self.k.checked_add(o.k)?,
        }
        .normalize()
    }

    fn scale(&self, c: i64) -> Option<Idx2> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for (v, m) in &self.terms {
            terms.push((*v, m.checked_mul(c)?));
        }
        Idx2 {
            r: self.r.scale(c)?,
            t: self.t.scale(c)?,
            terms,
            k: self.k.checked_mul(c)?,
        }
        .normalize()
    }

    fn as_const(&self) -> Option<i64> {
        (self.r.is_zero() && self.t.is_zero() && self.terms.is_empty()).then_some(self.k)
    }

    fn as_symbol(&self) -> Option<ValueId> {
        match (
            self.r.is_zero(),
            self.t.is_zero(),
            self.terms.as_slice(),
            self.k,
        ) {
            (true, true, [(v, 1)], 0) => Some(*v),
            _ => None,
        }
    }

    /// A product that stays linear: by a constant, or a bare invariant symbol times an expression
    /// with no symbolic part (the row pitch `r * T`).
    fn mul(&self, o: &Idx2) -> Option<Idx2> {
        if let Some(c) = o.as_const() {
            return self.scale(c);
        }
        if let Some(c) = self.as_const() {
            return o.scale(c);
        }
        for (s, lin) in [(self, o), (o, self)] {
            let Some(v) = s.as_symbol() else {
                continue;
            };
            if !lin.terms.is_empty() {
                continue;
            }
            let by = |c: Coeff| match c {
                Coeff::Zero => Some(Coeff::Zero),
                Coeff::Const(k) => Some(Coeff::Sym(v, k)),
                Coeff::Sym(..) => None,
            };
            let (Some(r), Some(t)) = (by(lin.r), by(lin.t)) else {
                continue;
            };
            let terms = if lin.k == 0 {
                Vec::new()
            } else {
                vec![(v, lin.k)]
            };
            return Idx2 { r, t, terms, k: 0 }.normalize();
        }
        None
    }
}

fn int_ty(t: &MirType) -> bool {
    matches!(t, MirType::I8 | MirType::I16 | MirType::I32 | MirType::I64)
}

/// A value the jammed preheader can read: a constant, or a value defined outside the nest.
#[derive(Clone, Copy, Debug)]
enum Val {
    C(i64),
    V(ValueId),
}

/// The function, seen from one nest.
struct Ctx<'a> {
    f: &'a Function,
    /// Value id -> `(block, inst)` of its defining instruction, over the whole function.
    def: Vec<Option<(u32, u32)>>,
    /// Every value defined in the nest: its blocks' parameters and instruction results.
    in_nest: FxHashSet<u32>,
    r: ValueId,
    t: ValueId,
}

impl<'a> Ctx<'a> {
    fn new(f: &'a Function, nest: &[BlockId], r: ValueId, t: ValueId) -> Ctx<'a> {
        let mut def = vec![None; f.value_types.len()];
        for b in &f.blocks {
            for (i, inst) in b.insts.iter().enumerate() {
                if let Some(v) = inst.result {
                    if let Some(slot) = def.get_mut(v.0 as usize) {
                        *slot = Some((b.id.0, i as u32));
                    }
                }
            }
        }
        let mut in_nest = FxHashSet::default();
        for &b in nest {
            let blk = &f.blocks[b.0 as usize];
            in_nest.extend(blk.params.iter().map(|p| p.0));
            in_nest.extend(blk.insts.iter().filter_map(|i| i.result.map(|v| v.0)));
        }
        Ctx {
            f,
            def,
            in_nest,
            r,
            t,
        }
    }

    fn op(&self, v: ValueId) -> Option<&'a Op> {
        let (b, i) = (*self.def.get(v.0 as usize)?)?;
        Some(&self.f.blocks[b as usize].insts[i as usize].op)
    }

    fn in_nest(&self, v: ValueId) -> bool {
        self.in_nest.contains(&v.0)
    }

    fn konst(&self, v: ValueId) -> Option<i64> {
        match self.op(v)? {
            Op::ConstInt(c, ty) if int_ty(ty) => i64::try_from(*c).ok(),
            _ => None,
        }
    }

    fn val(&self, v: ValueId) -> Option<Val> {
        match self.konst(v) {
            Some(c) => Some(Val::C(c)),
            None if !self.in_nest(v) => Some(Val::V(v)),
            None => None,
        }
    }

    fn idx(&self, v: ValueId, depth: u32) -> Option<Idx2> {
        if depth > DEPTH_LIMIT {
            return None;
        }
        if v == self.r {
            return Some(Idx2 {
                r: Coeff::Const(1),
                ..Idx2::default()
            });
        }
        if v == self.t {
            return Some(Idx2 {
                t: Coeff::Const(1),
                ..Idx2::default()
            });
        }
        if let Some(c) = self.konst(v) {
            return Some(Idx2::konst(c));
        }
        if !self.in_nest(v) {
            return int_ty(self.f.value_type(v)).then(|| Idx2::sym(v));
        }
        let d = depth + 1;
        match self.op(v)? {
            Op::Bin(BinOp::Add, a, b) => self.idx(*a, d)?.add(&self.idx(*b, d)?),
            Op::Bin(BinOp::Sub, a, b) => self.idx(*a, d)?.add(&self.idx(*b, d)?.scale(-1)?),
            Op::Bin(BinOp::Mul, a, b) => self.idx(*a, d)?.mul(&self.idx(*b, d)?),
            Op::Bin(BinOp::Shl, a, b) => {
                let k = self.konst(*b)?;
                if !(0..62).contains(&k) {
                    return None;
                }
                self.idx(*a, d)?.scale(1i64 << k)
            }
            Op::Neg(a) => self.idx(*a, d)?.scale(-1),
            Op::Cast(CastKind::SExt, a, _) => self.idx(*a, d),
            _ => None,
        }
    }

    /// The object `p` points at when the nest does not recompute it: a pointer defined outside
    /// the nest, or a `[]T` slice's data pointer re-read from a fat pointer defined outside it.
    fn base_of(&self, p: ValueId) -> Option<MemBase> {
        if !self.in_nest(p) {
            return Some(MemBase::Outer(p));
        }
        match self.op(p)? {
            Op::Load(a, MirType::Ptr) if !self.in_nest(*a) => {
                Some(MemBase::ReloadedOuter { addr: *a, value: p })
            }
            _ => None,
        }
    }

    /// `p` as `base + index · elem`, and whether it went through a `gep` at all.
    fn addr(&self, p: ValueId, value_ty: &MirType) -> Option<(MemBase, Idx2, MirType, bool)> {
        if let Some(b) = self.base_of(p) {
            return Some((b, Idx2::default(), value_ty.clone(), false));
        }
        let mut cur = p;
        let mut acc = Idx2::default();
        let mut elem: Option<MirType> = None;
        for _ in 0..GEP_CHAIN_LIMIT {
            let Some(Op::Gep {
                ptr,
                index,
                elem: e,
            }) = self.op(cur)
            else {
                return None;
            };
            if let Some(prev) = &elem {
                if prev != e {
                    return None;
                }
            } else {
                elem = Some(e.clone());
            }
            acc = acc.add(&self.idx(*index, 0)?)?;
            if let Some(b) = self.base_of(*ptr) {
                return Some((b, acc, elem?, true));
            }
            cur = *ptr;
        }
        None
    }

    /// Two different stack slots of this function.
    fn distinct_allocas(&self, a: MemBase, b: MemBase) -> bool {
        match (a, b) {
            (MemBase::Outer(x), MemBase::Outer(y)) => {
                x != y
                    && matches!(self.op(x), Some(Op::Alloca { .. }))
                    && matches!(self.op(y), Some(Op::Alloca { .. }))
            }
            _ => false,
        }
    }

    /// May the preheader read this base's pointer before the nest would have? Free for a pointer
    /// it already holds. A fat pointer is read only when it is a parameter (live for the whole
    /// call, the vectorizer's alias guard reads it the same way) or a stack slot of this function.
    fn readable_base(&self, b: MemBase) -> bool {
        match b {
            MemBase::Outer(_) => true,
            MemBase::ReloadedOuter { addr, .. } => {
                self.f.params.contains(&addr) || matches!(self.op(addr), Some(Op::Alloca { .. }))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------------------------

/// A point of the iteration space the preheader evaluates, as an `i64`.
#[derive(Clone, Copy, Debug)]
enum Pt {
    R0 = 0,
    RLast = 1,
    REnd = 2,
    T0 = 3,
    TLast = 4,
    TEnd = 5,
}

/// An element index the preheader evaluates: `Σ coeff · point + Σ value · mult + k`.
#[derive(Clone, Debug)]
struct Ex {
    parts: Vec<(Coeff, Pt)>,
    terms: Vec<(ValueId, i64)>,
    k: i64,
}

/// The element range `[lo, hi)` the whole nest may touch in one object.
#[derive(Clone, Debug)]
struct Region {
    base: MemBase,
    elem: MirType,
    lo: Ex,
    hi: Ex,
}

/// The iteration-space values the preheader reads.
#[derive(Clone, Copy, Debug)]
struct Points {
    r0: Val,
    rend: Val,
    r_signed: bool,
    t0: Val,
    tend: Val,
    t_signed: bool,
}

/// Everything [`apply`] needs, decided before any mutation.
struct Nest {
    op: BlockId,
    oh: BlockId,
    ob: BlockId,
    ih: BlockId,
    ib: BlockId,
    ix: BlockId,
    r_index: usize,
    r_ty: MirType,
    step: i64,
    /// The outer exit test, `r <pred> bound` while the loop runs: `Slt` or `Ult`.
    pred: CmpOp,
    bound: Val,
    t_index: usize,
    /// Outer header parameters other than `r` that change from row to row; read only in `IX`.
    carried: Vec<usize>,
    pts: Points,
    /// Pairs of objects whose ranges the preheader must find ordered and disjoint.
    pairs: Vec<(Region, Region)>,
}

/// One load or store of the nest.
struct Acc {
    base: MemBase,
    idx: Idx2,
    elem: MirType,
    store: bool,
    inner: bool,
    via_gep: bool,
}

/// The accesses of the nest to one object.
struct Group {
    base: MemBase,
    elem: MirType,
    members: Vec<usize>,
    stored: bool,
}

/// How a stored object is split between rows.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// `α` is the pitch; each row owns `[S + α·r, S + α·r + α)`.
    Row(Coeff),
    /// `α = 1`, `β` is the pitch, every row in `[0, β)`.
    Col(Coeff),
}

/// What the slab checks know about the two loops.
struct Ranges {
    r0: Option<i64>,
    rend: ValueId,
    r_signed: bool,
    s: i64,
    outer_trip: Option<u64>,
    t0: Option<i64>,
    tend: ValueId,
    t_signed: bool,
    st: i64,
    inner_trip: Option<u64>,
}

fn find_nest(f: &Function, banned: &FxHashSet<u32>) -> Option<Nest> {
    let rpo = cfg::reverse_postorder(f);
    let preds = cfg::predecessors(f);
    let idom = dom::idoms_from(f, &rpo, &preds);
    let mut forest = loop_info::structure_with(f, &idom, &preds);
    for oi in 0..forest.loops.len() {
        if banned.contains(&forest.loops[oi].header.0) {
            continue;
        }
        let Some(ii) = nest_shape(&forest, oi) else {
            continue;
        };
        loop_info::analyze_one(f, &mut forest, oi, Stages::Ivs);
        loop_info::analyze_one(f, &mut forest, ii, Stages::Memory);
        if let Some(n) = plan(f, &forest.loops[oi], &forest.loops[ii]) {
            return Some(n);
        }
    }
    None
}

/// The inner loop's index, when loop `oi` is a five-block nest around one two-block loop.
fn nest_shape(forest: &LoopForest, oi: usize) -> Option<usize> {
    let lo = &forest.loops[oi];
    let [child] = lo.children.as_slice() else {
        return None;
    };
    let ii = child.0 as usize;
    let li = &forest.loops[ii];
    let simple = |l: &NaturalLoop| {
        l.latches.len() == 1
            && l.exits.len() == 1
            && l.abnormal_exits.is_empty()
            && l.preheader.is_some()
            && l.exits[0].0 == l.header
    };
    (li.children.is_empty()
        && lo.blocks.len() == 5
        && li.blocks.len() == 2
        && simple(lo)
        && simple(li))
    .then_some(ii)
}

/// A header that computes its exit test and nothing else: constants and the one compare `cond`.
fn test_only(b: &BasicBlock, cond: ValueId) -> bool {
    b.insts.iter().all(|i| {
        matches!(i.op, Op::ConstInt(..) | Op::ConstFloat(..))
            || (i.result == Some(cond) && matches!(i.op, Op::Cmp(..)))
    })
}

/// The exit test `cond` as `iv <pred> bound` while the loop runs, for `pred` `Slt` or `Ult`.
fn oriented(ctx: &Ctx<'_>, cond: ValueId, iv: ValueId) -> Option<(CmpOp, ValueId)> {
    let Op::Cmp(p, a, b) = ctx.op(cond)? else {
        return None;
    };
    let (p, bound) = if *a == iv && *b != iv {
        (*p, *b)
    } else if *b == iv && *a != iv {
        let swapped = match p {
            CmpOp::Sgt => CmpOp::Slt,
            CmpOp::Ugt => CmpOp::Ult,
            _ => return None,
        };
        (swapped, *a)
    } else {
        return None;
    };
    matches!(p, CmpOp::Slt | CmpOp::Ult).then_some((p, bound))
}

/// How many times the nest's blocks read `v`, and whether any read is somewhere other than `ok`
/// allows (`ok(block, in_terminator)`).
fn uses(
    f: &Function,
    nest: &[BlockId],
    v: ValueId,
    ok: impl Fn(BlockId, bool) -> bool,
) -> (usize, bool) {
    let mut n = 0;
    let mut stray = false;
    for &b in nest {
        let blk = &f.blocks[b.0 as usize];
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |u| {
                if u == v {
                    n += 1;
                    stray |= !ok(b, false);
                }
            });
        }
        each_term_use(&blk.term, &mut |u| {
            if u == v {
                n += 1;
                stray |= !ok(b, true);
            }
        });
    }
    (n, stray)
}

/// The largest value the type holds read as signed.
fn signed_max(ty: &MirType) -> i128 {
    match ty {
        MirType::I8 => i8::MAX as i128,
        MirType::I16 => i16::MAX as i128,
        MirType::I32 => i32::MAX as i128,
        MirType::I64 => i64::MAX as i128,
        _ => 0,
    }
}

/// A limit no counter compares below under `pred`: every row then goes to the remainder.
fn sentinel(pred: CmpOp, ty: &MirType) -> i128 {
    match pred {
        CmpOp::Slt => match ty {
            MirType::I8 => i8::MIN as i128,
            MirType::I16 => i16::MIN as i128,
            MirType::I32 => i32::MIN as i128,
            _ => i64::MIN as i128,
        },
        _ => 0,
    }
}

fn plan(f: &Function, lo: &NaturalLoop, li: &NaturalLoop) -> Option<Nest> {
    // ---- shape -----------------------------------------------------------------------------
    let oh = lo.header;
    let op = lo.preheader?;
    let ih = li.header;
    let ob = li.preheader?;
    let ib = li.latches[0];
    let ix = li.exits[0].1;
    let oe = lo.exits[0].1;
    if ix != lo.latches[0] {
        return None;
    }
    let mut ids = [oh.0, ob.0, ih.0, ib.0, ix.0];
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) || ids.iter().any(|b| !lo.blocks.contains(&BlockId(*b)))
    {
        return None;
    }
    let blk = |b: BlockId| &f.blocks[b.0 as usize];
    let Terminator::CondBr {
        cond: oc,
        then_blk,
        then_args,
        else_blk,
        ..
    } = &blk(oh).term
    else {
        return None;
    };
    if *then_blk != ob || !then_args.is_empty() || *else_blk != oe {
        return None;
    }
    let Terminator::Br {
        target,
        args: ob_args,
    } = &blk(ob).term
    else {
        return None;
    };
    if *target != ih || !blk(ob).params.is_empty() {
        return None;
    }
    let Terminator::CondBr {
        cond: ic,
        then_blk: ithen,
        then_args: ithen_args,
        else_blk: ielse,
        ..
    } = &blk(ih).term
    else {
        return None;
    };
    if *ithen != ib || !ithen_args.is_empty() || *ielse != ix || !blk(ib).params.is_empty() {
        return None;
    }
    if !matches!(&blk(ib).term, Terminator::Br { target, .. } if *target == ih) {
        return None;
    }
    let Terminator::Br {
        target: ix_to,
        args: ix_args,
    } = &blk(ix).term
    else {
        return None;
    };
    let Terminator::Br {
        target: op_to,
        args: op_args,
    } = &blk(op).term
    else {
        return None;
    };
    if *ix_to != oh || *op_to != oh {
        return None;
    }
    if !test_only(blk(oh), *oc) || !test_only(blk(ih), *ic) {
        return None;
    }
    if blk(ib).insts.len() > MAX_INNER_INSTS
        || blk(ob).insts.len() + blk(ix).insts.len() > MAX_OUTER_INSTS
    {
        return None;
    }

    // ---- induction variables -----------------------------------------------------------------
    let oiv = lo.primary()?;
    let iiv = li.primary()?;
    let (IvStep::Const(step), IvStep::Const(st)) = (oiv.step, iiv.step) else {
        return None;
    };
    if step <= 0 || st <= 0 || !int_ty(&oiv.ty) || !int_ty(&iiv.ty) {
        return None;
    }
    if i128::from(FACTOR) * i128::from(step) > signed_max(&oiv.ty) {
        return None;
    }
    let (r_index, r, r_ty) = (oiv.param_index, oiv.value, oiv.ty.clone());
    let (t_index, t) = (iiv.param_index, iiv.value);
    let nest = [oh, ob, ih, ib, ix];
    let ctx = Ctx::new(f, &nest, r, t);

    let (pred, bound) = oriented(&ctx, *oc, r)?;
    let (ipred, tend) = oriented(&ctx, *ic, t)?;
    // An unsigned counter narrower than `i64` would need its constants zero-extended; not worth it.
    if (pred == CmpOp::Ult && r_ty != MirType::I64)
        || (ipred == CmpOp::Ult && iiv.ty != MirType::I64)
    {
        return None;
    }
    let bound_v = ctx.val(bound)?;
    let tend_v = ctx.val(tend)?;
    let t0 = *ob_args.get(t_index)?;
    let t0_v = ctx.val(t0)?;
    let r0 = *op_args.get(r_index)?;
    let r0_v = ctx.val(r0)?;
    if lo.trip.as_const().is_some_and(|n| n < FACTOR as u64) {
        return None;
    }
    // The exit tests are read by their own branches and nothing else in the nest.
    let only_term = |home: BlockId| move |b: BlockId, term: bool| b == home && term;
    if uses(f, &nest, *oc, only_term(oh)) != (1, false)
        || uses(f, &nest, *ic, only_term(ih)) != (1, false)
    {
        return None;
    }

    // ---- outer carried values ----------------------------------------------------------------
    let oh_params = &blk(oh).params;
    let mut carried = Vec::new();
    for (j, &p) in oh_params.iter().enumerate() {
        if j == r_index || *ix_args.get(j)? == p {
            continue; // the counter, or a value every row sees unchanged
        }
        // Rows run their epilogues in order, so a running value threads through them — but a
        // prologue or the inner loop reading it would see the wrong row's value.
        let (_, stray) = uses(f, &nest, p, |b, term| b == ix || (b == oh && term));
        if stray {
            return None;
        }
        carried.push(j);
    }

    // ---- what the nest may contain -----------------------------------------------------------
    for &b in &nest {
        let bb = blk(b);
        if bb.params.iter().any(|p| f.value_type(*p).is_vector()) {
            return None;
        }
        for inst in &bb.insts {
            if inst.result.is_some_and(|v| f.value_type(v).is_vector()) {
                return None;
            }
            match &inst.op {
                Op::Call { .. } | Op::VecKernelCall { .. } | Op::Barrier | Op::Alloca { .. } => {
                    return None
                }
                // A division runs in a different order across rows; one that can trap must not.
                Op::Bin(BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem, _, d) => {
                    if !matches!(ctx.konst(*d), Some(c) if c != 0 && c != -1) {
                        return None;
                    }
                }
                Op::Store { value, .. } if f.value_type(*value).is_vector() => return None,
                _ => {}
            }
        }
    }

    // ---- profitability: the inner loop carries a chain -----------------------------------------
    let chain = li.carried.iter().enumerate().any(|(k, c)| {
        k != t_index && matches!(c.kind, Carried::Recurrence | Carried::Reduction(_))
    });
    if !chain {
        return None;
    }

    // ---- memory ------------------------------------------------------------------------------
    let mut accs: Vec<Acc> = Vec::new();
    let mut reloads = false;
    for b in [ob, ib, ix] {
        for inst in &blk(b).insts {
            let (ptr, vty, store) = match &inst.op {
                Op::Load(p, ty) => (*p, ty.clone(), false),
                Op::Store { ptr, value } => (*ptr, f.value_type(*value).clone(), true),
                _ => continue,
            };
            // A pointer re-read from an address the nest does not compute — a slice's data pointer
            // from its fat pointer — is stable once no store below can write a pointer or a byte.
            if !store && vty == MirType::Ptr && !ctx.in_nest(ptr) {
                reloads = true;
                continue;
            }
            let (base, idx, elem, via_gep) = ctx.addr(ptr, &vty)?;
            if elem != vty || (b != ib && !idx.t.is_zero()) {
                return None;
            }
            reloads |= matches!(base, MemBase::ReloadedOuter { .. });
            accs.push(Acc {
                base,
                idx,
                elem,
                store,
                inner: b == ib,
                via_gep,
            });
        }
    }
    if accs.iter().any(|a| {
        a.store && (a.elem == MirType::Ptr || (reloads && (a.elem == MirType::I8 || !a.via_gep)))
    }) {
        return None;
    }
    let mut groups: Vec<Group> = Vec::new();
    for (i, a) in accs.iter().enumerate() {
        match groups.iter_mut().find(|g| g.base.same_object(a.base)) {
            Some(g) => {
                if g.elem != a.elem {
                    return None;
                }
                g.members.push(i);
                g.stored |= a.store;
            }
            None => groups.push(Group {
                base: a.base,
                elem: a.elem.clone(),
                members: vec![i],
                stored: a.store,
            }),
        }
    }

    let rg = Ranges {
        r0: ctx.konst(r0),
        rend: bound,
        r_signed: pred == CmpOp::Slt,
        s: step,
        outer_trip: lo.trip.as_const(),
        t0: ctx.konst(t0),
        tend,
        t_signed: ipred == CmpOp::Slt,
        st,
        inner_trip: li.trip.as_const(),
    };
    let mut regions: Vec<Option<Region>> = vec![None; groups.len()];
    for (gi, g) in groups.iter().enumerate() {
        if !g.stored {
            continue;
        }
        let members: Vec<&Acc> = g.members.iter().map(|&i| &accs[i]).collect();
        let layout = row_slab(&members, &rg).or_else(|| col_slab(&members, &rg))?;
        regions[gi] = Some(slab_region(g, &members[0].idx, layout));
    }

    // ---- objects that may be one buffer at run time ---------------------------------------------
    let mut promised = PromisedBases::new(f);
    let mut pair_ids: Vec<(usize, usize)> = Vec::new();
    for i in 0..groups.len() {
        for j in i + 1..groups.len() {
            if !groups[i].stored && !groups[j].stored {
                continue;
            }
            if ctx.distinct_allocas(groups[i].base, groups[j].base)
                || promised.disjoint(groups[i].base, groups[j].base)
            {
                continue;
            }
            for g in [i, j] {
                if !ctx.readable_base(groups[g].base) {
                    return None;
                }
                if regions[g].is_none() {
                    regions[g] = Some(load_region(&groups[g], &accs)?);
                }
            }
            pair_ids.push((i, j));
        }
    }
    if pair_ids.len() > MAX_GUARD_PAIRS {
        return None;
    }
    let mut pairs = Vec::with_capacity(pair_ids.len());
    for (i, j) in pair_ids {
        pairs.push((regions[i].clone()?, regions[j].clone()?));
    }

    Some(Nest {
        op,
        oh,
        ob,
        ih,
        ib,
        ix,
        r_index,
        r_ty,
        step,
        pred,
        bound: bound_v,
        t_index,
        carried,
        pts: Points {
            r0: r0_v,
            rend: bound_v,
            r_signed: pred == CmpOp::Slt,
            t0: t0_v,
            tend: tend_v,
            t_signed: ipred == CmpOp::Slt,
        },
        pairs,
    })
}

/// Row slabs: every access is `P·r + S + w` with one pitch `P` and one invariant part `S`, and the
/// in-row offset `w = β·t + k` provably in `[0, P)`.
fn row_slab(accs: &[&Acc], rg: &Ranges) -> Option<Layout> {
    let first = &accs[0].idx;
    if accs
        .iter()
        .any(|a| a.idx.r != first.r || a.idx.terms != first.terms)
    {
        return None;
    }
    match first.r {
        Coeff::Const(p) if p > 0 => {
            for a in accs {
                let k = i128::from(a.idx.k);
                let (lo, hi) = match a.idx.t {
                    Coeff::Zero => (k, k),
                    Coeff::Const(b) if b > 0 => {
                        let n = rg.inner_trip?;
                        if n == 0 {
                            continue; // the inner loop never runs, so neither does the access
                        }
                        let t0 = i128::from(rg.t0?);
                        let last = t0 + (i128::from(n) - 1) * i128::from(rg.st);
                        (i128::from(b) * t0 + k, i128::from(b) * last + k)
                    }
                    _ => return None,
                };
                if lo < 0 || hi >= i128::from(p) {
                    return None;
                }
            }
            Some(Layout::Row(first.r))
        }
        // The pitch is the inner loop's own runtime bound: `x[r*T + t]` for `t in t0..T`. The
        // offset `t + k` stays in `[0, T)` when `k <= 0` and `t0 + k >= 0`, because the loop only
        // runs while `t < T` (signed). A row-level access (`x[r*T]`) would need `T > 0`, which
        // nothing proves, so the whole object must be inner.
        Coeff::Sym(v, 1) => {
            if rg.tend != v || !rg.t_signed {
                return None;
            }
            let t0 = rg.t0?;
            for a in accs {
                if !a.inner || a.idx.t != Coeff::Const(1) {
                    return None;
                }
                if a.idx.k > 0 || t0.checked_add(a.idx.k)? < 0 {
                    return None;
                }
            }
            Some(Layout::Row(first.r))
        }
        _ => None,
    }
}

/// Column slabs: every access is the inner `r + B·t + S + k` with one pitch `B`, one `S` and one
/// `k`, and every row `r` in `[0, B)` — so two rows differ by less than the pitch and never meet.
fn col_slab(accs: &[&Acc], rg: &Ranges) -> Option<Layout> {
    let first = &accs[0].idx;
    if first.r != Coeff::Const(1) {
        return None;
    }
    if accs.iter().any(|a| {
        !a.inner
            || a.idx.r != first.r
            || a.idx.t != first.t
            || a.idx.terms != first.terms
            || a.idx.k != first.k
    }) {
        return None;
    }
    if rg.r0? < 0 {
        return None;
    }
    match first.t {
        Coeff::Const(b) if b > 0 => {
            let n = rg.outer_trip?;
            if n > 0 {
                let last = i128::from(rg.r0?) + (i128::from(n) - 1) * i128::from(rg.s);
                if last >= i128::from(b) {
                    return None;
                }
            }
        }
        Coeff::Sym(v, 1) => {
            if rg.rend != v || !rg.r_signed {
                return None;
            }
        }
        _ => return None,
    }
    Some(Layout::Col(first.t))
}

/// The element range a slab-laid-out object occupies over every row.
fn slab_region(g: &Group, idx: &Idx2, layout: Layout) -> Region {
    let (lo, hi) = match layout {
        // Row `r` owns `[S + P·r, S + P·r + P)`; the rows run over `[r0, R)`.
        Layout::Row(p) => (
            Ex {
                parts: vec![(p, Pt::R0)],
                terms: idx.terms.clone(),
                k: 0,
            },
            Ex {
                parts: vec![(p, Pt::REnd)],
                terms: idx.terms.clone(),
                k: 0,
            },
        ),
        // `r` in `[0, B)` and `t` in `[t0, T)`: `S + k + B·t0 <= index < S + k + B·T`.
        Layout::Col(b) => (
            Ex {
                parts: vec![(b, Pt::T0)],
                terms: idx.terms.clone(),
                k: idx.k,
            },
            Ex {
                parts: vec![(b, Pt::TEnd)],
                terms: idx.terms.clone(),
                k: idx.k,
            },
        ),
    };
    Region {
        base: g.base,
        elem: g.elem.clone(),
        lo,
        hi,
    }
}

/// The element range an object only read occupies, when its accesses share one `α`, `β` and `S`
/// with non-negative multipliers: evaluated at the first and the last row and inner iteration.
/// A symbolic multiplier may still be negative at run time, which would leave accesses outside the
/// range: the guard checks every one is non-negative ([`Em::nonnegative_multipliers`]).
fn load_region(g: &Group, accs: &[Acc]) -> Option<Region> {
    let first = &accs[g.members[0]].idx;
    let (mut kmin, mut kmax) = (first.k, first.k);
    for &i in &g.members {
        let a = &accs[i].idx;
        if a.r != first.r || a.t != first.t || a.terms != first.terms {
            return None;
        }
        kmin = kmin.min(a.k);
        kmax = kmax.max(a.k);
    }
    let non_negative = |c: Coeff| match c {
        Coeff::Zero => true,
        Coeff::Const(k) => k > 0,
        Coeff::Sym(_, m) => m > 0,
    };
    if !non_negative(first.r) || !non_negative(first.t) {
        return None;
    }
    Some(Region {
        base: g.base,
        elem: g.elem.clone(),
        lo: Ex {
            parts: vec![(first.r, Pt::R0), (first.t, Pt::T0)],
            terms: first.terms.clone(),
            k: kmin,
        },
        hi: Ex {
            parts: vec![(first.r, Pt::RLast), (first.t, Pt::TLast)],
            terms: first.terms.clone(),
            k: kmax.checked_add(1)?,
        },
    })
}

// ---------------------------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------------------------

/// Allocate a fresh SSA value of the given type.
fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

fn look(map: &FxHashMap<u32, ValueId>, v: ValueId) -> ValueId {
    map.get(&v.0).copied().unwrap_or(v)
}

/// Clone `src` into `out`, giving every result a fresh value and recording it in `map`.
fn clone_insts(
    f: &mut Function,
    src: &[Inst],
    map: &mut FxHashMap<u32, ValueId>,
    out: &mut Vec<Inst>,
) {
    for inst in src {
        let mut op = inst.op.clone();
        map_op_uses(&mut op, |v| look(map, v));
        let result = inst.result.map(|r| {
            let ty = f.value_type(r).clone();
            let nv = fresh(f, ty);
            map.insert(r.0, nv);
            nv
        });
        out.push(Inst { result, op });
    }
}

/// Instructions appended to one block, in order.
struct Em<'a> {
    f: &'a mut Function,
    out: Vec<Inst>,
}

impl Em<'_> {
    fn push(&mut self, ty: MirType, op: Op) -> ValueId {
        let v = fresh(self.f, ty);
        self.out.push(Inst {
            result: Some(v),
            op,
        });
        v
    }

    fn int(&mut self, ty: &MirType, k: i64) -> ValueId {
        self.push(ty.clone(), Op::ConstInt(i128::from(k), ty.clone()))
    }

    /// `v` as an `i64`.
    fn wide(&mut self, v: ValueId, signed: bool) -> ValueId {
        if *self.f.value_type(v) == MirType::I64 {
            return v;
        }
        let kind = if signed {
            CastKind::SExt
        } else {
            CastKind::ZExt
        };
        self.push(MirType::I64, Op::Cast(kind, v, MirType::I64))
    }

    fn val(&mut self, x: Val, signed: bool) -> ValueId {
        match x {
            Val::C(c) => self.int(&MirType::I64, c),
            Val::V(v) => self.wide(v, signed),
        }
    }

    fn bin(&mut self, op: BinOp, a: ValueId, b: ValueId) -> ValueId {
        self.push(MirType::I64, Op::Bin(op, a, b))
    }

    fn scaled(&mut self, v: ValueId, m: i64) -> ValueId {
        if m == 1 {
            return v;
        }
        let c = self.int(&MirType::I64, m);
        self.bin(BinOp::Mul, v, c)
    }

    /// An [`Ex`] as an `i64` element index.
    fn eval(&mut self, ex: &Ex, pts: &[ValueId; 6]) -> ValueId {
        let mut acc = self.int(&MirType::I64, ex.k);
        for &(v, m) in &ex.terms {
            let w = self.wide(v, true);
            let s = self.scaled(w, m);
            acc = self.bin(BinOp::Add, acc, s);
        }
        for &(c, pt) in &ex.parts {
            let p = pts[pt as usize];
            let term = match c {
                Coeff::Zero => continue,
                Coeff::Const(k) => self.scaled(p, k),
                Coeff::Sym(v, m) => {
                    let w = self.wide(v, true);
                    let s = self.scaled(w, m);
                    self.bin(BinOp::Mul, s, p)
                }
            };
            acc = self.bin(BinOp::Add, acc, term);
        }
        acc
    }

    /// A region's two ends as integer addresses. Both execution models agree on the order of two
    /// addresses built by `gep`s of one element type (the vectorizer's alias guard rests on the same
    /// fact); this is an ordering question about live objects, never a null test.
    fn region(
        &mut self,
        g: &Region,
        pts: &[ValueId; 6],
        bases: &mut FxHashMap<u32, ValueId>,
    ) -> (ValueId, ValueId) {
        let base = match g.base {
            MemBase::Outer(v) => v,
            MemBase::ReloadedOuter { addr, .. } => match bases.get(&addr.0) {
                Some(&p) => p,
                None => {
                    let p = self.push(MirType::Ptr, Op::Load(addr, MirType::Ptr));
                    bases.insert(addr.0, p);
                    p
                }
            },
        };
        let lo = self.eval(&g.lo, pts);
        let hi = self.eval(&g.hi, pts);
        (
            self.address(base, lo, &g.elem),
            self.address(base, hi, &g.elem),
        )
    }

    /// `base + index · elem` as an integer.
    fn address(&mut self, base: ValueId, index: ValueId, elem: &MirType) -> ValueId {
        let p = self.push(
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index,
                elem: elem.clone(),
            },
        );
        self.push(MirType::I64, Op::Cast(CastKind::PtrToInt, p, MirType::I64))
    }

    /// The conjunction of every pair check: each range ordered (`lo <= hi`), and the two apart.
    fn guard(&mut self, n: &Nest) -> ValueId {
        let p = n.pts;
        let r0 = self.val(p.r0, p.r_signed);
        let rend = self.val(p.rend, p.r_signed);
        let t0 = self.val(p.t0, p.t_signed);
        let tend = self.val(p.tend, p.t_signed);
        let one = self.int(&MirType::I64, 1);
        let rlast = self.bin(BinOp::Sub, rend, one);
        let tlast = self.bin(BinOp::Sub, tend, one);
        let pts = [r0, rlast, rend, t0, tlast, tend];
        let mut bases: FxHashMap<u32, ValueId> = FxHashMap::default();
        let mut acc: Option<ValueId> = None;
        for (g, h) in &n.pairs {
            let (gl, gh) = self.region(g, &pts, &mut bases);
            let (hl, hh) = self.region(h, &pts, &mut bases);
            let g_ok = self.push(MirType::I1, Op::Cmp(CmpOp::Ule, gl, gh));
            let h_ok = self.push(MirType::I1, Op::Cmp(CmpOp::Ule, hl, hh));
            let g_first = self.push(MirType::I1, Op::Cmp(CmpOp::Ule, gh, hl));
            let h_first = self.push(MirType::I1, Op::Cmp(CmpOp::Ule, hh, gl));
            let apart = self.push(MirType::I1, Op::Bin(BinOp::Or, g_first, h_first));
            let ordered = self.push(MirType::I1, Op::Bin(BinOp::And, g_ok, h_ok));
            let mut pair = self.push(MirType::I1, Op::Bin(BinOp::And, ordered, apart));
            for region in [g, h] {
                pair = self.nonnegative_multipliers(region, pair);
            }
            acc = Some(match acc {
                None => pair,
                Some(prev) => self.push(MirType::I1, Op::Bin(BinOp::And, prev, pair)),
            });
        }
        acc.expect("a guard has at least one pair")
    }

    /// `acc`, and every symbolic multiplier in `g`'s ends non-negative at run time. A range is
    /// evaluated at the first and last row and inner iteration, which bounds every access only when
    /// each multiplier is non-negative: a negative one mixed with a positive one (`x[r*s + t]`
    /// with `s < 0`) can still leave `lo <= hi` while missing accesses on both sides, so the
    /// `ordered` test alone does not catch it.
    fn nonnegative_multipliers(&mut self, g: &Region, mut acc: ValueId) -> ValueId {
        for &(c, _) in g.lo.parts.iter().chain(&g.hi.parts) {
            let Coeff::Sym(v, m) = c else {
                continue;
            };
            let w = self.wide(v, true);
            let s = self.scaled(w, m);
            let zero = self.int(&MirType::I64, 0);
            let ok = self.push(MirType::I1, Op::Cmp(CmpOp::Sge, s, zero));
            acc = self.push(MirType::I1, Op::Bin(BinOp::And, acc, ok));
        }
        acc
    }
}

/// `lim = R - (FACTOR-1)·s`, or the sentinel when that wraps or a runtime alias check fails —
/// appended to the outer preheader.
fn emit_limit(f: &mut Function, n: &Nest) -> ValueId {
    let ty = n.r_ty.clone();
    let mut em = Em {
        f: &mut *f,
        out: Vec::new(),
    };
    let bound = match n.bound {
        Val::C(c) => em.int(&ty, c),
        Val::V(v) => v,
    };
    let k = em.int(&ty, (FACTOR - 1) * n.step);
    let lim0 = em.push(ty.clone(), Op::Bin(BinOp::Sub, bound, k));
    let ok = em.push(MirType::I1, Op::Cmp(n.pred, lim0, bound));
    let sent = em.push(ty.clone(), Op::ConstInt(sentinel(n.pred, &ty), ty.clone()));
    let mut lim = em.push(ty.clone(), Op::Select(ok, lim0, sent));
    if !n.pairs.is_empty() {
        let apart = em.guard(n);
        lim = em.push(ty.clone(), Op::Select(apart, lim, sent));
    }
    let out = em.out;
    f.blocks[n.op.0 as usize].insts.extend(out);
    lim
}

/// Rewrite the nest; returns the remainder's header.
fn apply(f: &mut Function, n: &Nest) -> BlockId {
    let oh = f.blocks[n.oh.0 as usize].clone();
    let ob = f.blocks[n.ob.0 as usize].clone();
    let ih = f.blocks[n.ih.0 as usize].clone();
    let ib = f.blocks[n.ib.0 as usize].clone();
    let ix = f.blocks[n.ix.0 as usize].clone();
    let (
        Terminator::Br { args: ob_args, .. },
        Terminator::Br { args: ib_args, .. },
        Terminator::Br { args: ix_args, .. },
    ) = (&ob.term, &ib.term, &ix.term)
    else {
        unreachable!("plan checked the nest's unconditional branches")
    };
    let Terminator::CondBr {
        cond: ih_cond,
        else_args: ih_else,
        ..
    } = &ih.term
    else {
        unreachable!("plan checked the inner exit test")
    };
    let r = oh.params[n.r_index];
    let first = f.blocks.len() as u32;
    let (rh, j0, jh, jb, jx) = (
        BlockId(first),
        BlockId(first + 1),
        BlockId(first + 2),
        BlockId(first + 3),
        BlockId(first + 4),
    );

    let lim = emit_limit(f, n);

    // ---- RH: the original outer test, heading the remainder (the original nest) ---------------
    let mut rmap: FxHashMap<u32, ValueId> = FxHashMap::default();
    let mut rh_params = Vec::with_capacity(oh.params.len());
    for &p in &oh.params {
        let ty = f.value_type(p).clone();
        let nv = fresh(f, ty);
        rmap.insert(p.0, nv);
        rh_params.push(nv);
    }
    let mut rh_insts = Vec::with_capacity(oh.insts.len());
    clone_insts(f, &oh.insts, &mut rmap, &mut rh_insts);
    let mut rh_term = oh.term.clone();
    map_term_uses(&mut rh_term, |v| look(&rmap, v));

    // ---- J0: the four rows' prologues --------------------------------------------------------
    let u = FACTOR as usize;
    let mut maps: Vec<FxHashMap<u32, ValueId>> = vec![FxHashMap::default(); u];
    let mut j0_insts = Vec::new();
    for (k, map) in maps.iter_mut().enumerate().skip(1) {
        let off = fresh(f, n.r_ty.clone());
        j0_insts.push(Inst {
            result: Some(off),
            op: Op::ConstInt(i128::from(k as i64 * n.step), n.r_ty.clone()),
        });
        let rk = fresh(f, n.r_ty.clone());
        j0_insts.push(Inst {
            result: Some(rk),
            op: Op::Bin(BinOp::Add, r, off),
        });
        map.insert(r.0, rk);
    }
    for map in maps.iter_mut() {
        clone_insts(f, &ob.insts, map, &mut j0_insts);
    }

    // ---- JH: one shared inner counter, every other inner value once per row ------------------
    let mut jh_params = Vec::new();
    for (i, &p) in ih.params.iter().enumerate() {
        let ty = f.value_type(p).clone();
        if i == n.t_index {
            let nv = fresh(f, ty);
            for map in maps.iter_mut() {
                map.insert(p.0, nv);
            }
            jh_params.push(nv);
        } else {
            for map in maps.iter_mut() {
                let nv = fresh(f, ty.clone());
                map.insert(p.0, nv);
                jh_params.push(nv);
            }
        }
    }
    let per_row_args = |maps: &[FxHashMap<u32, ValueId>], args: &[ValueId]| -> Vec<ValueId> {
        let mut out = Vec::new();
        for (i, &a) in args.iter().enumerate() {
            if i == n.t_index {
                out.push(look(&maps[0], a));
            } else {
                out.extend(maps.iter().map(|m| look(m, a)));
            }
        }
        out
    };
    let j0_args = per_row_args(maps.as_slice(), ob_args.as_slice());
    // The exit test reads only the shared counter and invariants: one copy serves every row.
    let mut jh_insts = Vec::new();
    clone_insts(f, &ih.insts, &mut maps[0], &mut jh_insts);
    for inst in &ih.insts {
        if let Some(res) = inst.result {
            let nv = look(&maps[0], res);
            for map in maps.iter_mut().skip(1) {
                map.insert(res.0, nv);
            }
        }
    }
    let jh_cond = look(&maps[0], *ih_cond);
    let jh_else: Vec<ValueId> = maps
        .iter()
        .flat_map(|m| ih_else.iter().map(move |&a| look(m, a)))
        .collect();

    // ---- JB: the four rows' inner bodies, row by row -------------------------------------------
    let mut jb_insts = Vec::new();
    for map in maps.iter_mut() {
        clone_insts(f, &ib.insts, map, &mut jb_insts);
    }
    let jb_args = per_row_args(maps.as_slice(), ib_args.as_slice());

    // ---- JX: the four rows' epilogues, in row order ------------------------------------------
    let mut jx_params = Vec::new();
    for map in maps.iter_mut() {
        for &p in &ix.params {
            let ty = f.value_type(p).clone();
            let nv = fresh(f, ty);
            map.insert(p.0, nv);
            jx_params.push(nv);
        }
    }
    let mut jx_insts = Vec::new();
    // A running outer value enters each row as the row before it left it.
    let mut handed_on: Vec<ValueId> = Vec::new();
    for (k, map) in maps.iter_mut().enumerate() {
        if k > 0 {
            for (&j, &v) in n.carried.iter().zip(&handed_on) {
                map.insert(oh.params[j].0, v);
            }
        }
        clone_insts(f, &ix.insts, map, &mut jx_insts);
        handed_on = n.carried.iter().map(|&j| look(map, ix_args[j])).collect();
    }
    let step = fresh(f, n.r_ty.clone());
    jx_insts.push(Inst {
        result: Some(step),
        op: Op::ConstInt(i128::from(FACTOR * n.step), n.r_ty.clone()),
    });
    let r_next = fresh(f, n.r_ty.clone());
    jx_insts.push(Inst {
        result: Some(r_next),
        op: Op::Bin(BinOp::Add, r, step),
    });
    let jx_args: Vec<ValueId> = (0..oh.params.len())
        .map(|j| {
            if j == n.r_index {
                r_next
            } else {
                look(&maps[u - 1], ix_args[j])
            }
        })
        .collect();

    // ---- OH: enter the jammed rows while four remain -------------------------------------------
    let go = fresh(f, MirType::I1);
    let ohb = &mut f.blocks[n.oh.0 as usize];
    ohb.insts.push(Inst {
        result: Some(go),
        op: Op::Cmp(n.pred, r, lim),
    });
    ohb.term = Terminator::CondBr {
        cond: go,
        then_blk: j0,
        then_args: Vec::new(),
        else_blk: rh,
        else_args: oh.params.clone(),
    };
    // The original nest now closes on RH.
    if let Terminator::Br { target, .. } = &mut f.blocks[n.ix.0 as usize].term {
        *target = rh;
    }
    // Everything the original nest and the code after it read from OH now comes from RH, the only
    // way into either: OB/IH/IB/IX are reached only through RH, and so is the outer exit.
    for b in f.blocks.iter_mut() {
        if b.id == n.oh {
            continue;
        }
        for inst in &mut b.insts {
            map_op_uses(&mut inst.op, |v| look(&rmap, v));
        }
        map_term_uses(&mut b.term, |v| look(&rmap, v));
    }

    f.blocks.push(BasicBlock {
        id: rh,
        params: rh_params,
        insts: rh_insts,
        term: rh_term,
    });
    f.blocks.push(BasicBlock {
        id: j0,
        params: Vec::new(),
        insts: j0_insts,
        term: Terminator::Br {
            target: jh,
            args: j0_args,
        },
    });
    f.blocks.push(BasicBlock {
        id: jh,
        params: jh_params,
        insts: jh_insts,
        term: Terminator::CondBr {
            cond: jh_cond,
            then_blk: jb,
            then_args: Vec::new(),
            else_blk: jx,
            else_args: jh_else,
        },
    });
    f.blocks.push(BasicBlock {
        id: jb,
        params: Vec::new(),
        insts: jb_insts,
        term: Terminator::Br {
            target: jh,
            args: jb_args,
        },
    });
    f.blocks.push(BasicBlock {
        id: jx,
        params: jx_params,
        insts: jx_insts,
        term: Terminator::Br {
            target: n.oh,
            args: jx_args,
        },
    });
    rh
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Program;
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    fn output(p: &Program, interner: &mut Interner) -> (i64, String) {
        let main = interner.intern("main");
        let (code, bytes) =
            wukong_interp::run_with_output(p, main, interner).unwrap_or_else(|e| panic!("{e}"));
        (code, String::from_utf8_lossy(&bytes).to_string())
    }

    /// Run `src` at `-O0`; then through the standard `-O2` pipeline *without* inlining (so each
    /// kernel keeps its own nest and its parameters stay distinct values), jam every function,
    /// verify, clean up and run again; then through the real `-O2`. All three must print the same
    /// bytes. Returns whether anything was jammed.
    fn jams_and_agrees(src: &str) -> bool {
        let (p0, mut i0) = lower(src);
        let want = output(&p0, &mut i0);

        let (mut p, mut interner) = lower(src);
        crate::PassManager::standard(2).run(&mut p);
        let mut jammed = false;
        for f in &mut p.funcs {
            jammed |= jam_function(f);
            let errs = wukong_mir::verify::verify_function(f);
            assert!(
                errs.is_empty(),
                "MIR invalid after unroll-and-jam: {errs:?}"
            );
        }
        crate::PassManager::unroll_cleanup().run(&mut p);
        assert_eq!(
            output(&p, &mut interner),
            want,
            "-O0 and the jammed nest disagree"
        );

        let (mut p2, mut i2) = lower(src);
        crate::optimize(&mut p2, 2);
        assert_eq!(output(&p2, &mut i2), want, "-O0 and -O2 disagree");
        jammed
    }

    /// A row-major float recurrence over two array parameters. The first call runs the jammed rows
    /// (the runtime alias check passes); the second passes one buffer twice, and row `r` reads the
    /// first element of row `r + 1` before that row overwrites it — the check must fail and the
    /// original row order run, or the second half of the output changes.
    #[test]
    fn a_row_recurrence_is_jammed_and_an_aliased_call_keeps_row_order() {
        let src = "fn k(x: [f32; 29], mut o: [f32; 29], rows: i64) { \
                     let mut r: i64 = 0; \
                     while r < rows { \
                       let mut h: f32 = 1.0; let mut t: i64 = 0; \
                       while t < 4 { h = max(h, x[r * 4 + t + 1]) * 0.5 + x[r * 4 + t]; \
                                     o[r * 4 + t] = h; t = t + 1; } \
                       r = r + 1; } } \
                   fn main() -> i32 { \
                     let mut a: [f32; 29] = [0.0; 29]; let mut b: [f32; 29] = [0.0; 29]; \
                     let mut i: i64 = 0; \
                     while i < 29 { a[i] = (((i * 7) % 11) - 5) as f32 * 0.25; i = i + 1; } \
                     k(a, b, 7); k(a, a, 7); \
                     let mut j: i64 = 0; \
                     while j < 28 { print((b[j] * 64.0) as i32); print((a[j] * 64.0) as i32); \
                                    j = j + 1; } \
                     return 0; }";
        assert!(jams_and_agrees(src), "the row recurrence was not jammed");
    }

    /// A time-major recurrence (`x[t*5 + d]`, the outer loop over the columns) whose stores are
    /// column slabs, with a per-column result and a running total the jammed epilogues must update
    /// in column order.
    #[test]
    fn a_time_major_recurrence_with_a_running_total_is_jammed() {
        let src = "fn k(x: [i64; 15], mut y: [i64; 15], mut last: [i64; 5]) -> i64 { \
                     let mut total: i64 = 0; let mut d: i64 = 0; \
                     while d < 5 { \
                       let mut h: i64 = d + 1; let mut t: i64 = 0; \
                       while t < 3 { h = h * 3 - x[t * 5 + d]; y[t * 5 + d] = h; t = t + 1; } \
                       last[d] = h; total = total * 10 + h; d = d + 1; } \
                     return total; } \
                   fn main() -> i32 { \
                     let mut x: [i64; 15] = [0; 15]; let mut y: [i64; 15] = [0; 15]; \
                     let mut last: [i64; 5] = [0; 5]; \
                     let mut i: i64 = 0; \
                     while i < 15 { x[i] = ((i * 5) % 7) - 3; i = i + 1; } \
                     print(k(x, y, last)); \
                     let mut j: i64 = 0; while j < 15 { print(y[j]); j = j + 1; } \
                     j = 0; while j < 5 { print(last[j]); j = j + 1; } \
                     return 0; }";
        assert!(
            jams_and_agrees(src),
            "the time-major recurrence was not jammed"
        );
    }

    /// A symbolic-shape tensor kernel: the row pitch is the hidden dimension `T`, the inner loop's
    /// own bound, so each row is a slab of the runtime width.
    #[test]
    fn a_symbolic_pitch_tensor_nest_is_jammed() {
        let src = "fn k<R, T>(x: Tensor[i64, R, T], mut o: Tensor[i64, R, T]) { \
                     for r in 0..R { let mut h: i64 = 7; \
                       for t in 0..T { h = (h * h + x[r, t]) % 1009; o[r, t] = h; } } } \
                   fn main() -> i32 { \
                     let mut x: [i64; 15] = [0; 15]; let mut o: [i64; 15] = [0; 15]; \
                     let mut i: i64 = 0; \
                     while i < 15 { x[i] = (i * 13) % 17; i = i + 1; } \
                     let mut rows = 0; let mut n = 0; \
                     while n < 5 { rows = rows + 1; n = n + 1; } \
                     let cols = 3; \
                     k::<rows, cols>(x, o); \
                     let mut j: i64 = 0; while j < 15 { print(o[j]); j = j + 1; } \
                     return 0; }";
        assert!(
            jams_and_agrees(src),
            "the symbolic-pitch nest was not jammed"
        );
    }

    /// `o[r + t]`: neighbouring rows write overlapping windows, so the rows are not independent and
    /// the nest must stay as written.
    #[test]
    fn overlapping_row_windows_are_not_jammed() {
        let src = "fn k(x: [f32; 32], mut o: [f32; 32], rows: i64) { \
                     let mut r: i64 = 0; \
                     while r < rows { \
                       let mut h: f32 = 0.0; let mut t: i64 = 0; \
                       while t < 4 { h = max(h, x[r + t]) * 0.5 + 1.0; o[r + t] = h; t = t + 1; } \
                       r = r + 1; } } \
                   fn main() -> i32 { \
                     let mut a: [f32; 32] = [0.0; 32]; let mut b: [f32; 32] = [0.0; 32]; \
                     let mut i: i64 = 0; \
                     while i < 32 { a[i] = ((i * 3) % 8) as f32; i = i + 1; } \
                     k(a, b, 9); \
                     let mut j: i64 = 0; while j < 12 { print((b[j] * 16.0) as i32); j = j + 1; } \
                     return 0; }";
        assert!(
            !jams_and_agrees(src),
            "overlapping row windows must not be jammed"
        );
    }

    #[test]
    fn two_variable_index_algebra() {
        let r = ValueId(1);
        let n = ValueId(7);
        let row = Idx2 {
            r: Coeff::Const(1),
            ..Idx2::default()
        };
        let t = Idx2 {
            t: Coeff::Const(1),
            ..Idx2::default()
        };
        // `r * n + t + 1`: the symbolic pitch becomes the row coefficient.
        let e = row
            .mul(&Idx2::sym(n))
            .and_then(|x| x.add(&t))
            .and_then(|x| x.add(&Idx2::konst(1)))
            .expect("affine");
        assert_eq!(e.r, Coeff::Sym(n, 1));
        assert_eq!(e.t, Coeff::Const(1));
        assert!(e.terms.is_empty());
        assert_eq!(e.k, 1);
        // A product of the two counters is not affine.
        assert!(row.add(&Idx2::sym(r)).and_then(|x| x.mul(&t)).is_none());
        // `e - e` cancels to the canonical zero.
        let z = e.add(&e.scale(-1).expect("scale")).expect("add");
        assert_eq!(z, Idx2::default());
    }
}
