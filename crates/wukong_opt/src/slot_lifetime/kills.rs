//! Full overwrites: where a slot's previous contents can never be read again, because something
//! writes every byte of it before anything reads it (the parent module's "Full overwrites").
//!
//! Two kinds of kill come out of here:
//!
//! * **edge kills** `(from, header, slot)` — on every entering edge of a loop that starts a
//!   *covering region*: a sequence of counted loops, each run to completion in turn, that only ever
//!   write the slot (plain stores, a vector kernel's write-only stream, its address stored into a
//!   stream table) and between them write every byte of it;
//! * **point kills** `(block, instruction, slot)` — a runtime kernel call or a synthesized vector
//!   kernel that writes the whole slot before reading any of it.
//!
//! # Counted loops
//!
//! A loop is *counted* when its one way out is the header's exit test, that test compares an
//! induction variable against a constant, the variable starts at a constant on every entering edge
//! and moves by a constant step on every iteration, and the resulting trip count `n` is exact. The
//! induction variable is either a header block parameter whose latch argument evaluates to itself
//! plus the step — through a chain of adds too, which is how an unrolled body advances it — or, as
//! every `for` loop is spelled at `-O0`, a **counter slot**: an integer alloca that is not escaped,
//! is stored inside the loop only by one increment in the loop's single latch (its last access
//! there before the back edge) and otherwise only loaded directly, and is set to a constant as the
//! last access of every entering block. Every load of the counter inside the loop then reads the
//! current iteration's value. The exit test may be `<`, `<=`, `>`, `>=` (signed or unsigned) or
//! `!=` — the vectorizer's main loop steps by the vector width and tests `!=`, which is exact only
//! when the distance is a multiple of the step, so that is required. The start value of a loop that
//! follows another counted loop may be the other loop's induction variable read after it exited:
//! its exit value, `start + step * n`.
//!
//! # The integer evaluator and wraparound
//!
//! Store addresses, bounds, counts and increments are evaluated as exact affine functions of the
//! induction variables in scope (`Lin`), through `add`, `sub`, `mul`/`shl` by a constant, `neg` and
//! the integer casts; bit operations, comparisons and `select`s fold when their operands are
//! constants (the vectorizer computes a vector part's length that way), and a direct load of a
//! non-escaped local slot reads the value the last store into it earlier in the same block left
//! (the `-O0` spelling of a `let` the front end keeps in memory). Every intermediate value's range
//! over the induction variables' ranges must fit its MIR type (a `zext` operand must be
//! non-negative), so the exact value is the value the machine computes — no "indices do not wrap"
//! assumption is taken.
//!
//! # Coverage
//!
//! One iteration of a loop writes a set of byte runs: its own stores and vector-kernel streams in
//! blocks that dominate every latch (so they run on every iteration), each at
//! `offset(ivs) .. + bytes written`, and the runs of its child loops whose headers dominate every
//! latch (so they run, to completion, on every iteration). Runs with the same dependence on the
//! induction variables merge when they overlap or touch, and runs of one length at equal spacing (an
//! unrolled body storing at a stride wider than an element) become one run repeated along that
//! spacing. Sweeping the loop's own variable over its `n` iterations repeats the run `n` times,
//! `g` bytes apart; every repetition whose stride `|g|` is no wider than the run closes into it —
//! a run of `len + |g|·(n - 1)` bytes — which is the mixed-radix argument applied stride by
//! stride, smallest first, whatever order the loops nest in (a column-major nest closes at its
//! outer level). A store, a kernel or a child loop that is not understood contributes nothing (it
//! only writes, so it is harmless); anything in the region that may *read* the slot disqualifies the
//! region.

use std::cell::RefCell;

use wukong_mir::{
    BinOp, BlockId, CastKind, CmpOp, Function, MirType, Op, Terminator, ValueId, VecOp,
};

use super::{dominates, CaptureSummary, Flow, KernelOverwrite};
use crate::alias::type_bytes;
use crate::fxhash::FxHashMap;
use crate::loop_info::{self, LoopForest};
use crate::{cfg, dom};

/// Every full overwrite of one function (see the module note).
#[derive(Default)]
pub(super) struct Kills {
    /// `(from, to, slot)`: the edges along which a full overwrite kills `slot`.
    pub(super) edges: Vec<(u32, u32, u32)>,
    /// `(block, position, slot)`: the instructions that overwrite `slot` completely before reading
    /// it.
    pub(super) points: Vec<(u32, u32, u32)>,
}

/// Find every full overwrite of `f`'s allocas. `flow` is `f`'s settled address flow.
pub(super) fn find(f: &Function, flow: &Flow, calls: &CaptureSummary) -> Kills {
    let mut kills = Kills::default();
    if flow.allocas.is_empty() {
        return kills;
    }
    let proof = Proof::new(f, flow, calls);
    proof.call_kills(&mut kills.points);
    proof.region_kills(&mut kills.edges);
    kills
}

/// An integer as an exact affine function of the induction variables in scope:
/// `konst + Σ coeff · var`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Lin {
    konst: i128,
    /// `(variable, coefficient)`, ascending by variable, never a zero coefficient.
    terms: Vec<(usize, i128)>,
}

impl Lin {
    fn konst(k: i128) -> Lin {
        Lin {
            konst: k,
            terms: Vec::new(),
        }
    }

    fn var(v: usize) -> Lin {
        Lin {
            konst: 0,
            terms: vec![(v, 1)],
        }
    }

    fn as_const(&self) -> Option<i128> {
        self.terms.is_empty().then_some(self.konst)
    }

    fn add(&self, other: &Lin) -> Option<Lin> {
        let mut terms = self.terms.clone();
        for &(v, c) in &other.terms {
            match terms.binary_search_by_key(&v, |t| t.0) {
                Ok(i) => {
                    terms[i].1 = terms[i].1.checked_add(c)?;
                    if terms[i].1 == 0 {
                        terms.remove(i);
                    }
                }
                Err(i) => terms.insert(i, (v, c)),
            }
        }
        Some(Lin {
            konst: self.konst.checked_add(other.konst)?,
            terms,
        })
    }

    fn scale(&self, k: i128) -> Option<Lin> {
        if k == 0 {
            return Some(Lin::konst(0));
        }
        let mut terms = Vec::with_capacity(self.terms.len());
        for &(v, c) in &self.terms {
            terms.push((v, c.checked_mul(k)?));
        }
        Some(Lin {
            konst: self.konst.checked_mul(k)?,
            terms,
        })
    }

    fn coeff(&self, v: usize) -> i128 {
        self.terms.iter().find(|t| t.0 == v).map_or(0, |t| t.1)
    }

    /// This expression with variable `v` replaced by the constant `value`.
    fn substitute(&self, v: usize, value: i128) -> Option<Lin> {
        let c = self.coeff(v);
        let mut out = self.clone();
        out.terms.retain(|t| t.0 != v);
        out.konst = out.konst.checked_add(c.checked_mul(value)?)?;
        Some(out)
    }

    /// The least and greatest value over the variables' inclusive ranges.
    fn range(&self, ranges: &[(i128, i128)]) -> Option<(i128, i128)> {
        let (mut lo, mut hi) = (self.konst, self.konst);
        for &(v, c) in &self.terms {
            let (a, b) = *ranges.get(v)?;
            let (x, y) = (c.checked_mul(a)?, c.checked_mul(b)?);
            lo = lo.checked_add(x.min(y))?;
            hi = hi.checked_add(x.max(y))?;
        }
        Some((lo, hi))
    }
}

/// The signed range of an integer MIR type, or `None` for a type the evaluator does not handle.
fn int_range(ty: &MirType) -> Option<(i128, i128)> {
    Some(match ty {
        // A comparison's result, as 0 or 1 (only ever a constant here).
        MirType::I1 => (0, 1),
        MirType::I8 => (i8::MIN.into(), i8::MAX.into()),
        MirType::I16 => (i16::MIN.into(), i16::MAX.into()),
        MirType::I32 => (i32::MIN.into(), i32::MAX.into()),
        MirType::I64 => (i64::MIN.into(), i64::MAX.into()),
        MirType::I128 => (i128::MIN, i128::MAX),
        _ => return None,
    })
}

/// What carries a counted loop's induction variable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IvKey {
    /// A parameter of the loop header.
    Param(ValueId),
    /// A counter slot (the alloca's value), read by direct loads.
    Mem(ValueId),
}

/// A counted loop (see the module note).
#[derive(Clone, Debug)]
struct Counted {
    key: IvKey,
    /// The induction variable's integer type.
    ty: MirType,
    start: i128,
    step: i128,
    /// How many times the body runs (the header runs once more).
    trips: i128,
}

impl Counted {
    /// The value the induction variable holds once the loop has exited.
    fn exit_value(&self) -> Option<i128> {
        self.start.checked_add(self.step.checked_mul(self.trips)?)
    }

    /// Every value the induction variable takes, the exit value included.
    fn range(&self) -> Option<(i128, i128)> {
        let end = self.exit_value()?;
        Some((self.start.min(end), self.start.max(end)))
    }
}

/// The induction variables in scope while an expression is evaluated: variable `v` is `ivs[v]`.
#[derive(Clone, Default)]
struct Scope {
    /// `(key, loop index, inclusive value range)`.
    ivs: Vec<(IvKey, usize, (i128, i128))>,
    /// Skip the per-value range checks: only while an induction variable's range is still being
    /// derived (the result is re-checked once it is known).
    unchecked: bool,
}

impl Scope {
    fn ranges(&self) -> Vec<(i128, i128)> {
        self.ivs.iter().map(|iv| iv.2).collect()
    }
}

/// Where a value is read: an instruction's own block, or the edge `block -> to` a branch argument
/// travels.
#[derive(Clone, Copy)]
struct At {
    block: u32,
    to: Option<u32>,
}

impl At {
    fn block(block: u32) -> At {
        At { block, to: None }
    }
}

/// How deep the evaluator follows an operand chain.
const DEPTH_LIMIT: u32 = 64;
/// How many straight-line blocks a covering region may pass between two of its loops.
const CHAIN_LIMIT: usize = 32;

/// The bytes `⋃ [lo + Σ stride·m, lo + Σ stride·m + len)` over `m` in `0..count` for each
/// `(stride, count)` of `dims` — a run of `len` bytes repeated along the swept loops whose stride is
/// too wide to close the gaps by itself. `lo` may depend on the induction variables in scope.
#[derive(Clone, Debug)]
struct Piece {
    lo: Lin,
    len: i128,
    /// `(stride in bytes, count >= 2)`, never with a stride the run already spans.
    dims: Vec<(i128, i128)>,
}

impl Piece {
    fn run(lo: Lin, len: i128) -> Piece {
        Piece {
            lo,
            len,
            dims: Vec::new(),
        }
    }

    /// Fold every dimension whose stride is no wider than the run into the run — exact, because
    /// consecutive copies then overlap or touch — until none is left that can be: the mixed-radix
    /// argument, smallest stride first by construction.
    fn close(mut self) -> Option<Piece> {
        while let Some(i) = self.dims.iter().position(|&(g, _)| g.abs() <= self.len) {
            let (g, n) = self.dims.remove(i);
            let span = g.abs().checked_mul(n - 1)?;
            if g < 0 {
                self.lo = self.lo.add(&Lin::konst(-span))?;
            }
            self.len = self.len.checked_add(span)?;
        }
        Some(self)
    }

    /// An upper bound on the bytes the piece names, to pick between pieces that do not merge.
    fn extent(&self) -> i128 {
        self.dims
            .iter()
            .fold(self.len, |acc, &(_, n)| acc.saturating_mul(n))
    }
}

/// What a counted-loop query has found so far for one loop.
#[derive(Clone)]
enum Memo {
    Unknown,
    /// Being computed: a query that reaches it again (a cycle through exit values) gives up.
    Busy,
    Done(Option<Counted>),
}

/// The facts the overwrite proofs share for one function.
struct Proof<'a> {
    f: &'a Function,
    flow: &'a Flow,
    /// The program's call summary, with the runtime's overwrite promises.
    calls: &'a CaptureSummary,
    forest: LoopForest,
    idom: Vec<u32>,
    /// Per value: the `(block, instruction)` defining it (`None` for a block parameter).
    def: Vec<Option<(u32, u32)>>,
    /// Header parameter -> the loop it heads.
    header_param: FxHashMap<ValueId, usize>,
    /// Header block -> its loop.
    header_loop: FxHashMap<u32, usize>,
    /// Alloca value -> its root index.
    alloca_index: FxHashMap<ValueId, u32>,
    counted: RefCell<Vec<Memo>>,
}

impl<'a> Proof<'a> {
    fn new(f: &'a Function, flow: &'a Flow, calls: &'a CaptureSummary) -> Proof<'a> {
        let rpo = cfg::reverse_postorder(f);
        let preds = cfg::predecessors(f);
        let idom = dom::idoms_from(f, &rpo, &preds);
        let forest = loop_info::structure_with(f, &idom, &preds);
        let mut def = vec![None; f.value_types.len()];
        for b in &f.blocks {
            for (i, inst) in b.insts.iter().enumerate() {
                if let Some(slot) = inst.result.and_then(|r| def.get_mut(r.0 as usize)) {
                    *slot = Some((b.id.0, i as u32));
                }
            }
        }
        let mut header_param = FxHashMap::default();
        let mut header_loop = FxHashMap::default();
        for (li, l) in forest.loops.iter().enumerate() {
            header_loop.insert(l.header.0, li);
            for &p in &f.block(l.header).params {
                header_param.insert(p, li);
            }
        }
        let alloca_index = flow
            .allocas
            .iter()
            .enumerate()
            .map(|(i, &a)| (a, i as u32))
            .collect();
        let n = forest.loops.len();
        Proof {
            f,
            flow,
            calls,
            forest,
            idom,
            def,
            header_param,
            header_loop,
            alloca_index,
            counted: RefCell::new(vec![Memo::Unknown; n]),
        }
    }

    fn in_loop(&self, li: usize, b: u32) -> bool {
        self.forest.loops[li]
            .blocks
            .binary_search_by_key(&b, |x| x.0)
            .is_ok()
    }

    fn op_of(&self, v: ValueId) -> Option<(u32, &'a Op)> {
        let (b, i) = (*self.def.get(v.0 as usize)?)?;
        Some((b, &self.f.blocks[b as usize].insts[i as usize].op))
    }

    // -----------------------------------------------------------------------------------------
    // The integer evaluator
    // -----------------------------------------------------------------------------------------

    /// `v` as an exact affine function of the induction variables in `scope`, read at `at`.
    fn lin(&self, v: ValueId, at: At, scope: &Scope) -> Option<Lin> {
        let ranges = scope.ranges();
        let mut memo = FxHashMap::default();
        self.lin_in(v, at, scope, &ranges, &mut memo, 0)
    }

    fn lin_in(
        &self,
        v: ValueId,
        at: At,
        scope: &Scope,
        ranges: &[(i128, i128)],
        memo: &mut FxHashMap<u32, Option<Lin>>,
        depth: u32,
    ) -> Option<Lin> {
        if depth > DEPTH_LIMIT {
            return None;
        }
        let bounds = int_range(self.f.value_types.get(v.0 as usize)?)?;
        let r = match self.op_of(v) {
            Some((b, op)) => {
                if let Some(cached) = memo.get(&v.0) {
                    return cached.clone();
                }
                let r = self.lin_op(op, b, v, scope, ranges, memo, depth);
                memo.insert(v.0, r.clone());
                r?
            }
            None => self.lin_param(v, at, scope)?,
        };
        if !scope.unchecked {
            let (lo, hi) = r.range(ranges)?;
            if lo < bounds.0 || hi > bounds.1 {
                return None;
            }
        }
        Some(r)
    }

    #[allow(clippy::too_many_arguments)]
    fn lin_op(
        &self,
        op: &Op,
        b: u32,
        result: ValueId,
        scope: &Scope,
        ranges: &[(i128, i128)],
        memo: &mut FxHashMap<u32, Option<Lin>>,
        depth: u32,
    ) -> Option<Lin> {
        let here = At::block(b);
        let mut sub = |x: ValueId| self.lin_in(x, here, scope, ranges, memo, depth + 1);
        match op {
            Op::ConstInt(k, ty) if int_range(ty).is_some() => Some(Lin::konst(*k)),
            Op::Bin(BinOp::Add, a, c) => {
                let x = sub(*a)?;
                x.add(&sub(*c)?)
            }
            Op::Bin(BinOp::Sub, a, c) => {
                let x = sub(*a)?;
                x.add(&sub(*c)?.scale(-1)?)
            }
            Op::Bin(BinOp::Mul, a, c) => {
                let (x, y) = (sub(*a)?, sub(*c)?);
                match (x.as_const(), y.as_const()) {
                    (_, Some(k)) => x.scale(k),
                    (Some(k), None) => y.scale(k),
                    (None, None) => None,
                }
            }
            Op::Bin(BinOp::Shl, a, c) => {
                let k = sub(*c)?.as_const()?;
                if !(0..63).contains(&k) {
                    return None;
                }
                sub(*a)?.scale(1i128 << k)
            }
            Op::Neg(a) => sub(*a)?.scale(-1),
            // Bit operations, comparisons and selects fold when their operands are constants: the
            // vectorizer computes a vector part's length that way (`(n - lo) & -8`, a `select` on
            // whether any group fits).
            Op::Bin(op @ (BinOp::And | BinOp::Or | BinOp::Xor), a, c) => {
                let (x, y) = (sub(*a)?.as_const()?, sub(*c)?.as_const()?);
                Some(Lin::konst(match op {
                    BinOp::And => x & y,
                    BinOp::Or => x | y,
                    _ => x ^ y,
                }))
            }
            Op::Cmp(pred, a, c) => {
                let (x, y) = (sub(*a)?.as_const()?, sub(*c)?.as_const()?);
                let unsigned = is_unsigned(*pred);
                if unsigned && (x < 0 || y < 0) {
                    return None;
                }
                let holds = match pred {
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                    CmpOp::Slt | CmpOp::Ult => x < y,
                    CmpOp::Sle | CmpOp::Ule => x <= y,
                    CmpOp::Sgt | CmpOp::Ugt => x > y,
                    CmpOp::Sge | CmpOp::Uge => x >= y,
                    _ => return None,
                };
                Some(Lin::konst(i128::from(holds)))
            }
            Op::Select(cond, a, c) => match sub(*cond)?.as_const()? {
                0 => sub(*c),
                _ => sub(*a),
            },
            // A sign extension is exact: the operand's own range was checked against its type.
            // `trunc` is exact when the result's range fits the narrower type (checked by the
            // caller), and `zext` when the operand is never negative.
            Op::Cast(CastKind::SExt | CastKind::Trunc, a, _) => sub(*a),
            Op::Cast(CastKind::ZExt, a, _) => {
                let x = sub(*a)?;
                if !scope.unchecked && x.range(ranges)?.0 < 0 {
                    return None;
                }
                Some(x)
            }
            // A direct load of a counter slot inside its loop is the loop's induction variable; a
            // direct load of another local slot is the value the last store into it earlier in the
            // same block left there.
            Op::Load(p, ty) => {
                let counter = scope.ivs.iter().enumerate().find_map(|(v, iv)| {
                    (iv.0 == IvKey::Mem(*p)
                        && self.in_loop(iv.1, b)
                        && *ty == self.counter_ty(iv.1)?)
                    .then(|| Lin::var(v))
                });
                match counter {
                    Some(l) => Some(l),
                    None => {
                        let stored = self.stored_before(b, result, *p)?;
                        if self.f.value_type(stored) != ty {
                            return None;
                        }
                        sub(stored)
                    }
                }
            }
            _ => None,
        }
    }

    /// The value the load `load` (of the non-escaped alloca `p`, in block `b`) reads, when the last
    /// instruction before it in `b` that touches `p` other than by a direct load is a direct store
    /// into `p`.
    fn stored_before(&self, b: u32, load: ValueId, p: ValueId) -> Option<ValueId> {
        let (_, at) = (*self.def.get(load.0 as usize)?)?;
        let &slot = self.alloca_index.get(&p)?;
        if self.flow.escaped[slot as usize] {
            return None;
        }
        let insts = &self.f.blocks[b as usize].insts;
        let (pos, _) = self.flow.touches[b as usize]
            .iter()
            .rev()
            .find(|(pos, s)| {
                *pos < at
                    && s.contains(&slot)
                    && !matches!(&insts[*pos as usize].op, Op::Load(q, _) if *q == p)
            })?;
        match &insts[*pos as usize].op {
            Op::Store { ptr, value } if *ptr == p => Some(*value),
            _ => None,
        }
    }

    /// The type of the counter slot loop `li`'s induction variable lives in, if it has one.
    fn counter_ty(&self, li: usize) -> Option<MirType> {
        match &self.counted.borrow()[li] {
            Memo::Done(Some(c)) => Some(c.ty.clone()),
            // While loop `li` itself is being analyzed, its candidate counter is in scope under the
            // type its header load reads; `step_of`/`start_of` pass that type explicitly.
            _ => None,
        }
    }

    /// A block parameter: an induction variable in scope, or a counted loop's induction variable
    /// read after that loop has exited (its exit value).
    fn lin_param(&self, v: ValueId, at: At, scope: &Scope) -> Option<Lin> {
        if let Some(i) = scope.ivs.iter().position(|iv| iv.0 == IvKey::Param(v)) {
            return Some(Lin::var(i));
        }
        let &li = self.header_param.get(&v)?;
        let l = &self.forest.loops[li];
        let outside = !self.in_loop(li, at.block)
            || (at.block == l.header.0 && at.to.is_some_and(|t| !self.in_loop(li, t)));
        if !outside {
            return None;
        }
        let c = self.counted(li)?;
        if c.key != IvKey::Param(v) {
            return None;
        }
        Some(Lin::konst(c.exit_value()?))
    }

    /// `p` as `(slot, byte offset)`: a chain of `gep`s and pointer casts off one alloca.
    fn addr_of(&self, p: ValueId, scope: &Scope, depth: u32) -> Option<(u32, Lin)> {
        if let Some(&i) = self.alloca_index.get(&p) {
            return Some((i, Lin::konst(0)));
        }
        if depth > DEPTH_LIMIT {
            return None;
        }
        let (b, op) = self.op_of(p)?;
        match op {
            Op::Gep { ptr, index, elem } => {
                let (slot, base) = self.addr_of(*ptr, scope, depth + 1)?;
                let idx = self.lin(*index, At::block(b), scope)?;
                let off = base.add(&idx.scale(i128::from(type_bytes(elem)?))?)?;
                let (lo, hi) = off.range(&scope.ranges())?;
                let i64r = (i128::from(i64::MIN), i128::from(i64::MAX));
                (lo >= i64r.0 && hi <= i64r.1).then_some((slot, off))
            }
            Op::Cast(CastKind::Bitcast, q, MirType::Ptr) => self.addr_of(*q, scope, depth + 1),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------------------------
    // Counted loops
    // -----------------------------------------------------------------------------------------

    fn counted(&self, li: usize) -> Option<Counted> {
        match &self.counted.borrow()[li] {
            Memo::Done(c) => return c.clone(),
            Memo::Busy => return None,
            Memo::Unknown => {}
        }
        self.counted.borrow_mut()[li] = Memo::Busy;
        let c = self.counted_uncached(li);
        self.counted.borrow_mut()[li] = Memo::Done(c.clone());
        c
    }

    fn counted_uncached(&self, li: usize) -> Option<Counted> {
        let l = &self.forest.loops[li];
        if l.exits.len() != 1 || l.exits[0].0 != l.header || !l.abnormal_exits.is_empty() {
            return None;
        }
        let h = l.header.0;
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &self.f.blocks[h as usize].term
        else {
            return None;
        };
        let on_true = match (self.in_loop(li, then_blk.0), self.in_loop(li, else_blk.0)) {
            (true, false) => true,
            (false, true) => false,
            _ => return None,
        };
        let (cb, op) = self.op_of(*cond)?;
        let Op::Cmp(pred, a, b) = op else {
            return None;
        };
        if cb != h {
            return None;
        }
        let pred = if on_true { *pred } else { negate_cmp(*pred)? };
        for (iv, bound, pred) in [(*a, *b, Some(pred)), (*b, *a, swap_cmp(pred))] {
            if let Some(c) = pred.and_then(|p| self.try_iv(li, iv, bound, p)) {
                return Some(c);
            }
        }
        None
    }

    /// Loop `li` counted by the induction variable `iv` (a header parameter or a direct load of a
    /// counter slot in the header), continuing while `iv <pred> bound`.
    fn try_iv(&self, li: usize, iv: ValueId, bound: ValueId, pred: CmpOp) -> Option<Counted> {
        let l = &self.forest.loops[li];
        let h = l.header.0;
        let ty = self.f.value_type(iv).clone();
        int_range(&ty)?;
        let key = if self.f.block(l.header).params.contains(&iv) {
            IvKey::Param(iv)
        } else {
            match self.op_of(iv)? {
                (b, Op::Load(p, lty)) if b == h && *lty == ty && self.is_counter(*p, &ty) => {
                    IvKey::Mem(*p)
                }
                _ => return None,
            }
        };
        let bound = self
            .lin(bound, At::block(h), &Scope::default())?
            .as_const()?;
        let step = self.step_of(li, key, &ty, None)?;
        let start = self.start_of(li, key)?;
        let trips = trip_count(pred, start, bound, step)?;
        let c = Counted {
            key,
            ty: ty.clone(),
            start,
            step,
            trips,
        };
        let (lo, hi) = c.range()?;
        let (tlo, thi) = int_range(&ty)?;
        if lo < tlo || hi > thi || (is_unsigned(pred) && (lo < 0 || bound < 0)) {
            return None;
        }
        // The increment again, now with every intermediate value range-checked.
        (self.step_of(li, key, &ty, Some((lo, hi)))? == step).then_some(c)
    }

    /// Is `p` an integer alloca of type `ty` whose address never escapes?
    fn is_counter(&self, p: ValueId, ty: &MirType) -> bool {
        let Some(&i) = self.alloca_index.get(&p) else {
            return false;
        };
        !self.flow.escaped[i as usize]
            && matches!(self.op_of(p), Some((_, Op::Alloca { ty: at, .. })) if at == ty)
    }

    /// The constant the induction variable moves by per iteration. `range` is its value range when
    /// known; without one the increment is evaluated without range checks.
    fn step_of(
        &self,
        li: usize,
        key: IvKey,
        ty: &MirType,
        range: Option<(i128, i128)>,
    ) -> Option<i128> {
        let l = &self.forest.loops[li];
        let scope = Scope {
            ivs: vec![(key, li, range.unwrap_or((0, 0)))],
            unchecked: range.is_none(),
        };
        let ranges = scope.ranges();
        let as_step = |next: Lin| -> Option<i128> {
            (next.terms == [(0, 1)] && next.konst != 0).then_some(next.konst)
        };
        match key {
            IvKey::Param(p) => {
                let k = self.f.block(l.header).params.iter().position(|&x| x == p)?;
                let mut step = None;
                for latch in &l.latches {
                    let next =
                        *edge_args(&self.f.blocks[latch.0 as usize].term, l.header)?.get(k)?;
                    let at = At {
                        block: latch.0,
                        to: Some(l.header.0),
                    };
                    let s = as_step(self.lin(next, at, &scope)?)?;
                    if step.is_some_and(|prev| prev != s) {
                        return None;
                    }
                    step = Some(s);
                }
                step
            }
            IvKey::Mem(c) => {
                // The counter: loaded directly anywhere in the loop, stored only by one increment
                // that is the last access of it in the loop's single latch, before the back edge.
                let [latch] = l.latches[..] else {
                    return None;
                };
                if !matches!(&self.f.blocks[latch.0 as usize].term,
                    Terminator::Br { target, .. } if *target == l.header)
                {
                    return None;
                }
                let ci = *self.alloca_index.get(&c)?;
                let mut store: Option<(u32, u32)> = None;
                let mut last_in_latch: Option<u32> = None;
                for &b in &l.blocks {
                    let insts = &self.f.blocks[b.0 as usize].insts;
                    for (at, slots) in &self.flow.touches[b.0 as usize] {
                        if !slots.contains(&ci) {
                            continue;
                        }
                        if b == latch {
                            last_in_latch = Some(*at);
                        }
                        match insts.get(*at as usize).map(|i| &i.op) {
                            Some(Op::Load(p, lty)) if *p == c && lty == ty => {}
                            Some(Op::Store { ptr, value })
                                if *ptr == c
                                    && self.f.value_type(*value) == ty
                                    && store.is_none() =>
                            {
                                store = Some((b.0, *at));
                            }
                            _ => return None,
                        }
                    }
                }
                let (sb, sat) = store?;
                if sb != latch.0 || last_in_latch != Some(sat) {
                    return None;
                }
                let Op::Store { value, .. } = &self.f.blocks[sb as usize].insts[sat as usize].op
                else {
                    return None;
                };
                let mut memo = FxHashMap::default();
                as_step(self.lin_counter(*value, &scope, &ranges, ty, &mut memo, 0)?)
            }
        }
    }

    /// [`Proof::lin_in`] for a loop whose counter-slot type is not memoized yet (the loop is being
    /// analyzed): a direct load of the counter of type `ty` inside the loop is variable 0.
    fn lin_counter(
        &self,
        v: ValueId,
        scope: &Scope,
        ranges: &[(i128, i128)],
        ty: &MirType,
        memo: &mut FxHashMap<u32, Option<Lin>>,
        depth: u32,
    ) -> Option<Lin> {
        if depth > DEPTH_LIMIT {
            return None;
        }
        let (IvKey::Mem(c), li, _) = scope.ivs[0] else {
            return None;
        };
        let (vb, op) = self.op_of(v)?;
        let r = match op {
            Op::Load(p, lty) if *p == c && lty == ty && self.in_loop(li, vb) => Lin::var(0),
            Op::Bin(BinOp::Add, x, y) => {
                let a = self.lin_counter(*x, scope, ranges, ty, memo, depth + 1)?;
                a.add(&self.lin_counter(*y, scope, ranges, ty, memo, depth + 1)?)?
            }
            Op::Bin(BinOp::Sub, x, y) => {
                let a = self.lin_counter(*x, scope, ranges, ty, memo, depth + 1)?;
                a.add(
                    &self
                        .lin_counter(*y, scope, ranges, ty, memo, depth + 1)?
                        .scale(-1)?,
                )?
            }
            // Anything else is evaluated without the counter: it must not depend on it.
            _ => self.lin_in(v, At::block(vb), &Scope::default(), &[], memo, depth + 1)?,
        };
        if !scope.unchecked {
            let (lo, hi) = r.range(ranges)?;
            let (tlo, thi) = int_range(self.f.value_types.get(v.0 as usize)?)?;
            if lo < tlo || hi > thi {
                return None;
            }
        }
        Some(r)
    }

    /// The constant every entering edge starts the induction variable at.
    fn start_of(&self, li: usize, key: IvKey) -> Option<i128> {
        let l = &self.forest.loops[li];
        let mut start = None;
        for &e in &l.entering {
            let v = match key {
                IvKey::Param(p) => {
                    let k = self.f.block(l.header).params.iter().position(|&x| x == p)?;
                    let arg = *edge_args(&self.f.blocks[e.0 as usize].term, l.header)?.get(k)?;
                    let at = At {
                        block: e.0,
                        to: Some(l.header.0),
                    };
                    self.lin(arg, at, &Scope::default())?.as_const()?
                }
                IvKey::Mem(c) => {
                    // The counter's last access in the entering block stores a constant into it.
                    let ci = *self.alloca_index.get(&c)?;
                    let block = &self.f.blocks[e.0 as usize];
                    let (at, _) = self.flow.touches[e.0 as usize]
                        .iter()
                        .rev()
                        .find(|(_, s)| s.contains(&ci))?;
                    let Some(Op::Store { ptr, value }) =
                        block.insts.get(*at as usize).map(|i| &i.op)
                    else {
                        return None;
                    };
                    if *ptr != c {
                        return None;
                    }
                    self.lin(*value, At::block(e.0), &Scope::default())?
                        .as_const()?
                }
            };
            if start.is_some_and(|s| s != v) {
                return None;
            }
            start = Some(v);
        }
        start
    }

    // -----------------------------------------------------------------------------------------
    // Covering regions (edge kills)
    // -----------------------------------------------------------------------------------------

    /// The slot a store writes as part of a full overwrite: through a pointer derived from that
    /// slot alone, storing a value that holds no slot's address.
    fn store_of(&self, b: usize, at: u32, slots: &[u32]) -> Option<u32> {
        let [x] = slots else { return None };
        let inst = self.f.blocks[b].insts.get(at as usize)?;
        let Op::Store { ptr, value } = &inst.op else {
            return None;
        };
        let alone = self.flow.roots_of(*ptr) == [*x] && !self.flow.other_of(*ptr);
        let plain = !self
            .flow
            .roots_of(*value)
            .iter()
            .any(|&r| (r as usize) < self.flow.allocas.len());
        (alone && plain).then_some(*x)
    }

    /// What the touch at position `at` of block `b` (touching `slots`) does to slot `x`.
    fn touch_kind(&self, b: u32, at: u32, slots: &[u32], x: u32) -> Touch {
        if self.store_of(b as usize, at, slots) == Some(x) {
            return Touch::Write;
        }
        match self.f.blocks[b as usize]
            .insts
            .get(at as usize)
            .map(|i| &i.op)
        {
            // Its address stored into another tracked slot (a vector kernel's stream table): the
            // slot itself is not accessed. (Into untracked memory it would have escaped.)
            Some(Op::Store { ptr, value })
                if !self.flow.roots_of(*ptr).contains(&x)
                    && self.flow.roots_of(*value).contains(&x) =>
            {
                Touch::Address
            }
            Some(Op::VecKernelCall { .. }) => {
                match self.vec_call(b, at as usize, &Scope::default()) {
                    Some(call) if self.vec_writes_only(&call, x) => Touch::Write,
                    _ => Touch::Other,
                }
            }
            Some(Op::Call { func, args }) => match self.calls.overwrite(*func) {
                Some(ow)
                    if self
                        .kernel_write(b, args, ow, x, &Scope::default())
                        .is_some() =>
                {
                    Touch::Write
                }
                _ => Touch::Other,
            },
            _ => Touch::Other,
        }
    }

    /// Does every touch of slot `x` inside loop `li` write it without reading it, or not access it
    /// at all?
    fn only_writes(&self, li: usize, x: u32) -> bool {
        self.forest.loops[li].blocks.iter().all(|&b| {
            self.flow.touches[b.0 as usize]
                .iter()
                .filter(|(_, s)| s.contains(&x))
                .all(|(at, s)| self.touch_kind(b.0, *at, s, x) != Touch::Other)
        })
    }

    /// Every covering region: for each loop and each slot it writes and never reads, the bytes the
    /// loop and the loops forced to run after it write, and an edge kill on the loop's entering
    /// edges when they cover the slot.
    fn region_kills(&self, kills: &mut Vec<(u32, u32, u32)>) {
        for li in 0..self.forest.loops.len() {
            let l = &self.forest.loops[li];
            let mut cands: Vec<u32> = Vec::new();
            for &b in &l.blocks {
                for (at, slots) in &self.flow.touches[b.0 as usize] {
                    for &x in slots {
                        if !cands.contains(&x)
                            && !self.flow.escaped[x as usize]
                            && self.touch_kind(b.0, *at, slots, x) == Touch::Write
                        {
                            cands.push(x);
                        }
                    }
                }
            }
            cands.retain(|&x| self.only_writes(li, x));
            if cands.is_empty() || self.counted(li).is_none() {
                continue;
            }
            cands.sort_unstable();
            for x in cands {
                let Some(size) = self.flow.alloca_bytes[x as usize] else {
                    continue;
                };
                let size = i128::from(size);
                let mut runs: Vec<(i128, i128)> = Vec::new();
                if let Some(run) = self
                    .cover(li, x, &mut Scope::default())
                    .as_ref()
                    .and_then(constant_run)
                {
                    runs.push(run);
                }
                if !covers(&mut runs, size) {
                    self.follow(li, x, size, &mut runs);
                }
                if covers(&mut runs, size) {
                    for e in &l.entering {
                        kills.push((e.0, l.header.0, x));
                    }
                }
            }
        }
    }

    /// Extend `runs` with the loops forced to run after loop `li0` exits — through straight-line
    /// blocks that do not touch `x`, into the header of a counted loop that only ever writes `x`,
    /// and on from its exit — until they cover `size` bytes.
    fn follow(&self, li0: usize, x: u32, size: i128, runs: &mut Vec<(i128, i128)>) {
        let l0 = &self.forest.loops[li0];
        let mut visited = vec![li0];
        let mut cur = l0.exits[0].1 .0;
        for _ in 0..CHAIN_LIMIT {
            if covers(runs, size) {
                return;
            }
            if let Some(&lj) = self.header_loop.get(&cur) {
                if visited.contains(&lj)
                    || self.in_loop(lj, l0.header.0)
                    || !self.only_writes(lj, x)
                    || self.counted(lj).is_none()
                {
                    return;
                }
                if let Some(run) = self
                    .cover(lj, x, &mut Scope::default())
                    .as_ref()
                    .and_then(constant_run)
                {
                    runs.push(run);
                }
                visited.push(lj);
                cur = self.forest.loops[lj].exits[0].1 .0;
            } else {
                let touches_x = self.flow.touches[cur as usize]
                    .iter()
                    .any(|(_, s)| s.contains(&x));
                let Terminator::Br { target, .. } = &self.f.blocks[cur as usize].term else {
                    return;
                };
                if touches_x {
                    return;
                }
                cur = target.0;
            }
        }
    }

    /// The bytes of slot `x` one complete run of loop `li` is sure to write, as a single run
    /// relative to the induction variables already in `scope`, or `None` when it is sure of none.
    fn cover(&self, li: usize, x: u32, scope: &mut Scope) -> Option<Piece> {
        let c = self.counted(li)?;
        if c.trips <= 0 {
            return None;
        }
        let l = &self.forest.loops[li];
        scope.ivs.push((c.key, li, c.range()?));
        let var = scope.ivs.len() - 1;
        let every_iteration = |b: u32| l.latches.iter().all(|lt| dominates(b, lt.0, &self.idom));
        let mut pieces: Vec<Piece> = Vec::new();
        for &b in &l.blocks {
            if self.forest.innermost_of(b) != Some(l.id) || !every_iteration(b.0) {
                continue;
            }
            for (at, slots) in &self.flow.touches[b.0 as usize] {
                if !slots.contains(&x) {
                    continue;
                }
                let op = self.f.blocks[b.0 as usize]
                    .insts
                    .get(*at as usize)
                    .map(|i| &i.op);
                if let Some(Op::VecKernelCall { .. }) = op {
                    pieces.extend(self.vec_pieces(b.0, *at as usize, x, scope));
                    continue;
                }
                if let Some(Op::Call { func, args }) = op {
                    if let Some(ow) = self.calls.overwrite(*func) {
                        pieces.extend(self.kernel_write(b.0, args, ow, x, scope).flatten());
                    }
                    continue;
                }
                if self.store_of(b.0 as usize, *at, slots) != Some(x) {
                    continue;
                }
                let Some(Op::Store { ptr, value }) = op else {
                    continue;
                };
                let (Some((slot, lo)), Some(len)) = (
                    self.addr_of(*ptr, scope, 0),
                    type_bytes(self.f.value_type(*value)),
                ) else {
                    continue;
                };
                if slot == x && len > 0 {
                    pieces.push(Piece::run(lo, i128::from(len)));
                }
            }
        }
        for child in &l.children {
            let ci = child.0 as usize;
            if every_iteration(self.forest.loops[ci].header.0) {
                if let Some(p) = self.cover(ci, x, scope) {
                    pieces.push(p);
                }
            }
        }
        scope.ivs.pop();
        let p = merge(pieces)?;
        sweep(p, var, &c)
    }

    /// The runs of slot `x` a vector-kernel call at instruction `at` of block `b` overwrites, at
    /// offsets relative to the induction variables in `scope`.
    fn vec_pieces(&self, b: u32, at: usize, x: u32, scope: &Scope) -> Vec<Piece> {
        let Some(call) = self.vec_call(b, at, scope) else {
            return Vec::new();
        };
        let Some(bytes) = call.bytes.filter(|&n| n > 0) else {
            return Vec::new();
        };
        if !self.vec_writes_only(&call, x) {
            return Vec::new();
        }
        call.entries
            .iter()
            .filter_map(|&v| match self.addr_of(v, scope, 0) {
                Some((slot, lo)) if slot == x => Some(Piece::run(lo, bytes)),
                _ => None,
            })
            .collect()
    }

    // -----------------------------------------------------------------------------------------
    // Kernel calls (point kills)
    // -----------------------------------------------------------------------------------------

    /// The non-escaped slot `p` is exactly the start of, with its byte size.
    fn slot_base(&self, p: ValueId) -> Option<(u32, i128)> {
        let (x, off) = self.addr_of(p, &Scope::default(), 0)?;
        if off.as_const()? != 0 || self.flow.escaped[x as usize] {
            return None;
        }
        Some((x, i128::from(self.flow.alloca_bytes[x as usize]?)))
    }

    /// Does slot `x` reach a call through any of `others` (directly, or held in a slot they reach)?
    fn reached_through(&self, x: u32, others: impl IntoIterator<Item = ValueId>) -> bool {
        others.into_iter().any(|a| {
            let roots = self.flow.roots_of(a);
            roots.contains(&x) || self.flow.contents_closure(roots).contains(&x)
        })
    }

    /// Runtime kernel calls and vector-kernel calls that overwrite a slot completely.
    fn call_kills(&self, kills: &mut Vec<(u32, u32, u32)>) {
        for (bi, block) in self.f.blocks.iter().enumerate() {
            for (ii, inst) in block.insts.iter().enumerate() {
                let x = match &inst.op {
                    Op::Call { func, args } => match self.calls.overwrite(*func) {
                        Some(ow) => self.kernel_kill(bi as u32, args, ow),
                        None => None,
                    },
                    Op::VecKernelCall { .. } => {
                        for x in self.vec_kernel_kills(bi as u32, ii) {
                            kills.push((bi as u32, ii as u32, x));
                        }
                        None
                    }
                    _ => None,
                };
                if let Some(x) = x {
                    kills.push((bi as u32, ii as u32, x));
                }
            }
        }
    }

    /// The slot a runtime call with overwrite promise `ow` writes completely, if any.
    fn kernel_kill(&self, b: u32, args: &[ValueId], ow: &KernelOverwrite) -> Option<u32> {
        let (x, size) = self.slot_base(*args.get(ow.out)?)?;
        let piece = self.kernel_write(b, args, ow, x, &Scope::default())??;
        (piece.lo.as_const()? == 0 && piece.len >= size).then_some(x)
    }

    /// What a runtime call with overwrite promise `ow` (operands `args`, in block `b`) does to slot
    /// `x`: `None` when it may read it (the promise's conditions do not hold as constants, or `x`
    /// reaches the call through another operand); otherwise `Some` of the bytes it overwrites, at an
    /// offset relative to the induction variables in `scope` (`Some(None)` when that offset is not
    /// understood — the call still only writes `x`).
    fn kernel_write(
        &self,
        b: u32,
        args: &[ValueId],
        ow: &KernelOverwrite,
        x: u32,
        scope: &Scope,
    ) -> Option<Option<Piece>> {
        if args.len() != ow.arity {
            return None;
        }
        let konst = |k: usize| self.lin(*args.get(k)?, At::block(b), scope)?.as_const();
        let mut elems: i128 = 1;
        for &k in ow.extent {
            let v = konst(k)?;
            if v < 1 {
                return None;
            }
            elems = elems.checked_mul(v)?;
        }
        for &k in ow.positive {
            if konst(k)? < 1 {
                return None;
            }
        }
        for &(k, allowed) in ow.fixed {
            let v = konst(k)?;
            if !allowed.iter().any(|&a| i128::from(a) == v) {
                return None;
            }
        }
        let out = *args.get(ow.out)?;
        let others = args
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != ow.out)
            .map(|(_, &a)| a);
        if !self.flow.roots_of(out).contains(&x)
            || self.reached_through(x, others)
            || self
                .flow
                .contents_closure(self.flow.roots_of(out))
                .contains(&x)
        {
            return None;
        }
        let bytes = elems.checked_mul(i128::from(ow.elem_bytes))?;
        Some(match self.addr_of(out, scope, 0) {
            Some((slot, lo)) if slot == x => Some(Piece::run(lo, bytes)),
            _ => None,
        })
    }

    /// The slots a vector-kernel call at instruction `at` of block `b` writes completely: each one a
    /// write-only stream names exactly (offset 0, nothing else reaching it) over a vector part at
    /// least as long as the slot.
    fn vec_kernel_kills(&self, b: u32, at: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let Some(call) = self.vec_call(b, at, &Scope::default()) else {
            return out;
        };
        let Some(bytes) = call.bytes else {
            return out;
        };
        for (s, &v) in call.entries.iter().enumerate() {
            if !call.write_only[s] {
                continue;
            }
            let Some((x, size)) = self.slot_base(v) else {
                continue;
            };
            if bytes >= size && self.vec_writes_only(&call, x) && !out.contains(&x) {
                out.push(x);
            }
        }
        out
    }

    /// The vector-kernel call at instruction `at` of block `b`, decoded: its stream table entries
    /// (read off the stores into the table earlier in the block), which streams the recipe only
    /// stores to, and the bytes of each stream its vector part covers (whole lane groups of a count
    /// that is constant in `scope`). `None` for a reduction recipe, or when the table cannot be read.
    fn vec_call(&self, b: u32, at: usize, scope: &Scope) -> Option<VecCall> {
        let Op::VecKernelCall {
            kernel,
            ptrs,
            scalars,
            n,
        } = &self.f.blocks[b as usize].insts.get(at)?.op
        else {
            return None;
        };
        let k = self.f.vec_kernels.get(*kernel as usize)?;
        if k.reduce.is_some() {
            return None;
        }
        let entries = self.table_entries(b, at, *ptrs, k.streams as usize)?;
        let write_only = (0..entries.len())
            .map(|s| {
                let stores = k
                    .ops
                    .iter()
                    .any(|op| matches!(op, VecOp::Store { stream, .. } if *stream as usize == s));
                let loads = k
                    .ops
                    .iter()
                    .any(|op| matches!(op, VecOp::Load { stream } if *stream as usize == s));
                stores && !loads
            })
            .collect();
        let lanes = i128::from(k.lane.lanes());
        let bytes = self
            .lin(*n, At::block(b), scope)
            .and_then(|n| n.as_const())
            .filter(|&n| n >= 0)
            .map(|n| (n - n % lanes) * i128::from(k.lane.bytes()));
        Some(VecCall {
            entries,
            write_only,
            bytes,
            scalars: *scalars,
        })
    }

    /// Does `call` reach slot `x` only through streams it stores to without loading — never through
    /// a loaded stream, the scalars, or an address held in a slot it reaches?
    fn vec_writes_only(&self, call: &VecCall, x: u32) -> bool {
        let mut named = false;
        for (s, &v) in call.entries.iter().enumerate() {
            let roots = self.flow.roots_of(v);
            if roots.contains(&x) {
                if !call.write_only[s] {
                    return false;
                }
                named = true;
            }
            if self.flow.contents_closure(roots).contains(&x) {
                return false;
            }
        }
        named && !self.reached_through(x, [call.scalars])
    }

    /// The pointer each of a vector kernel's `streams` table entries holds at instruction `at` of
    /// block `b`: the last store into that entry earlier in the block, when nothing else there
    /// touches the table and the table itself is a non-escaped slot.
    fn table_entries(
        &self,
        b: u32,
        at: usize,
        ptrs: ValueId,
        streams: usize,
    ) -> Option<Vec<ValueId>> {
        let &table = self.alloca_index.get(&ptrs)?;
        if self.flow.escaped[table as usize] {
            return None;
        }
        let ptr_bytes = i128::from(type_bytes(&MirType::Ptr)?);
        let insts = &self.f.blocks[b as usize].insts;
        let mut found: Vec<Option<ValueId>> = vec![None; streams];
        for (pos, slots) in self.flow.touches[b as usize].iter().rev() {
            if *pos as usize >= at || !slots.contains(&table) {
                continue;
            }
            if found.iter().all(Option::is_some) {
                break;
            }
            let Op::Store { ptr, value } = &insts[*pos as usize].op else {
                return None;
            };
            let (slot, off) = self.addr_of(*ptr, &Scope::default(), 0)?;
            let off = off.as_const()?;
            if slot != table || off % ptr_bytes != 0 || self.f.value_type(*value) != &MirType::Ptr {
                return None;
            }
            let entry = usize::try_from(off / ptr_bytes).ok()?;
            if let Some(e) = found.get_mut(entry) {
                e.get_or_insert(*value);
            }
        }
        found.into_iter().collect()
    }
}

/// What one touch of a slot does to it, as far as a covering region is concerned.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Touch {
    /// Writes it without reading it: a plain store, or a vector kernel's write-only stream.
    Write,
    /// Does not access it at all: its address is stored into another tracked slot.
    Address,
    /// Anything that may read it.
    Other,
}

/// A vector-kernel call's operands as [`Proof::vec_call`] decodes them.
struct VecCall {
    /// Per stream: the pointer its table entry holds.
    entries: Vec<ValueId>,
    /// Per stream: does the recipe store to it and never load it?
    write_only: Vec<bool>,
    /// The bytes of each stream the vector part covers, when the count is a constant.
    bytes: Option<i128>,
    scalars: ValueId,
}

/// Merge pieces written on the same iteration: plain runs with the same dependence on the induction
/// variables join when they overlap or touch, and runs of one length at equal spacing (the copies an
/// unrolled body stores at a stride wider than one element) become one piece with that spacing as a
/// dimension. The piece naming the most bytes is kept (any one is sound — every piece is written).
fn merge(pieces: Vec<Piece>) -> Option<Piece> {
    // Per dependence on the induction variables (a `Lin`'s terms): the constant runs.
    type Group = (Vec<(usize, i128)>, Vec<(i128, i128)>);
    let mut groups: Vec<Group> = Vec::new();
    let mut best: Option<Piece> = None;
    let consider = |p: Piece, best: &mut Option<Piece>| {
        if best.as_ref().is_none_or(|b| p.extent() > b.extent()) {
            *best = Some(p);
        }
    };
    for p in pieces {
        if !p.dims.is_empty() {
            consider(p, &mut best);
            continue;
        }
        let run = (p.lo.konst, p.lo.konst.checked_add(p.len)?);
        match groups.iter_mut().find(|g| g.0 == p.lo.terms) {
            Some(g) => g.1.push(run),
            None => groups.push((p.lo.terms, vec![run])),
        }
    }
    for (terms, mut runs) in groups {
        runs.sort_unstable();
        let mut merged: Vec<(i128, i128)> = Vec::new();
        for (lo, hi) in runs {
            match merged.last_mut() {
                Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        let len = merged[0].1 - merged[0].0;
        let gap = merged.get(1).map(|r| r.0 - merged[0].0);
        let progression = merged.len() >= 2
            && merged
                .windows(2)
                .all(|w| w[1].1 - w[1].0 == len && Some(w[1].0 - w[0].0) == gap);
        if let (true, Some(gap)) = (progression, gap) {
            let lin = Lin {
                konst: merged[0].0,
                terms: terms.clone(),
            };
            let count = i128::try_from(merged.len()).ok()?;
            let p = Piece {
                lo: lin,
                len,
                dims: vec![(gap, count)],
            };
            consider(p.close()?, &mut best);
            continue;
        }
        for (lo, hi) in merged {
            let lin = Lin {
                konst: lo,
                terms: terms.clone(),
            };
            consider(Piece::run(lin, hi - lo), &mut best);
        }
    }
    best
}

/// One iteration's piece `p`, swept over the `trips` values of variable `var` (loop `c`): the
/// variable's coefficient times the step is a new dimension, closed into the run when the copies
/// meet.
fn sweep(p: Piece, var: usize, c: &Counted) -> Option<Piece> {
    let a = p.lo.coeff(var);
    let lo = p.lo.substitute(var, c.start)?;
    let mut dims = p.dims;
    let g = a.checked_mul(c.step)?;
    if c.trips > 1 && g != 0 {
        dims.push((g, c.trips));
    }
    Piece {
        lo,
        len: p.len,
        dims,
    }
    .close()
}

/// The constant byte run a piece over no induction variable is sure to cover: the whole piece when
/// it is one run, else its first copy.
fn constant_run(p: &Piece) -> Option<(i128, i128)> {
    let lo = p.lo.as_const()?;
    Some((lo, lo.checked_add(p.len)?))
}

/// Do `runs` (constant byte runs, merged in place) cover `[0, size)`?
fn covers(runs: &mut Vec<(i128, i128)>, size: i128) -> bool {
    runs.sort_unstable();
    let mut merged: Vec<(i128, i128)> = Vec::with_capacity(runs.len());
    for &(lo, hi) in runs.iter() {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    *runs = merged;
    runs.iter().any(|&(lo, hi)| lo <= 0 && hi >= size)
}

/// How many times the body runs for an induction variable starting at `s`, moving by `t` and
/// continuing while `iv <pred> bound`; `None` when that is not exact (a `!=` test the variable can
/// step over, a step away from the bound).
fn trip_count(pred: CmpOp, s: i128, bound: i128, t: i128) -> Option<i128> {
    use CmpOp::*;
    let ceil = |num: i128, den: i128| (num + den - 1) / den;
    Some(match pred {
        Slt | Ult if t > 0 => {
            if bound > s {
                ceil(bound - s, t)
            } else {
                0
            }
        }
        Sle | Ule if t > 0 => {
            if bound >= s {
                (bound - s) / t + 1
            } else {
                0
            }
        }
        Sgt | Ugt if t < 0 => {
            if s > bound {
                ceil(s - bound, -t)
            } else {
                0
            }
        }
        Sge | Uge if t < 0 => {
            if s >= bound {
                (s - bound) / -t + 1
            } else {
                0
            }
        }
        Ne => {
            let d = bound.checked_sub(s)?;
            if d % t != 0 || d / t < 0 {
                return None;
            }
            d / t
        }
        _ => return None,
    })
}

fn is_unsigned(p: CmpOp) -> bool {
    matches!(p, CmpOp::Ult | CmpOp::Ule | CmpOp::Ugt | CmpOp::Uge)
}

/// The predicate that means the same with the operands exchanged (`a < b` is `b > a`).
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
        _ => return None,
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
        _ => return None,
    })
}

/// The arguments `t` passes to `target`, when they are unambiguous.
fn edge_args(t: &Terminator, target: BlockId) -> Option<&[ValueId]> {
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
        Terminator::Switch {
            cases,
            default,
            default_args,
            ..
        } => {
            let mut into = cases
                .iter()
                .filter(|c| c.target == target)
                .map(|c| c.args.as_slice())
                .chain((*default == target).then_some(default_args.as_slice()));
            let first = into.next()?;
            into.all(|a| a == first).then_some(first)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
