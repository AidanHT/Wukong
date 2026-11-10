//! Jump threading: an edge into a block whose branch that edge already decides goes straight to the
//! decided successor, through a private copy of the block's instructions.
//!
//! ```text
//!   P: …; cond_br (x < n), A, …        P: …; cond_br (x < n), A, …
//!   A: …; br M(1)                      A: …; br M'
//!   Q: …; br M(c)                 =>   Q: …; br M(c)
//!   M(p): t = …; cond_br p, T, E       M(p): t = …; cond_br p, T, E
//!                                      M':   t' = …; br T          (from A only)
//! ```
//!
//! An edge `P -> M` *decides* `M`'s branch when the condition evaluates to a constant on it, which
//! this pass establishes two ways (LLVM's JumpThreading, restricted to what it can prove locally):
//!
//! * **a constant argument** — `M`'s condition is computed from `M`'s own parameters, and the edge
//!   passes constants for them: `cond_br p` with `p = 1` on the edge, or `p < 10` with `p = 3`;
//! * **a dominating compare** — the edge lies under a branch on a compare of the *same values*
//!   that implies `M`'s: below the true arm of `x < n`, a re-test `x < n`, `x <= n` or `x != n` is
//!   true and `x >= n`, `x > n` or `x == n` false. "Below" means the edge leaves `P` on that arm, or
//!   `P` is dominated by that arm of a branch whose target has no other predecessor; `and`/`or`
//!   conditions contribute both operands where the outcome determines them.
//!
//! [`crate::branch_fold`]'s threading (run inside `simplify-cfg`) already covers the first case for
//! an *empty* `M`; this pass copies up to [`MAX_THREAD_INSTS`] instructions, which is what a merge
//! block that recomputes its test, or a loop-exit check after a guarded update, needs.
//!
//! # Legality
//!
//! The copy `M'` has exactly one predecessor, `P`, so every value it reads is available there: a
//! parameter of `M` is replaced by the argument `P` passed for it, and anything else `M` reads is
//! defined outside `M`, hence dominates `M` and every one of its predecessors. Each path through the
//! program still executes `M`'s instructions exactly once (in `M` or in `M'`), in the same order, on
//! the same operands, so loads and stores are copied as freely as arithmetic; calls, `alloca`s and
//! barriers are not copied at all.
//!
//! A value of `M` read *past* `M` — `if x < n { a = .. } else { a = .. }; if x < n { b = a + 1 }`
//! reads the join's parameter `a` in the second diamond's arms — has two definitions once the
//! threaded path runs `M'` instead: `M`'s and `M'`'s. Every block that read it was dominated by `M`,
//! so every path into it now passes through exactly one of the two, and the reaching definition is
//! found by walking predecessors back to them, with a new block parameter wherever several paths
//! meet (Braun et al.'s on-demand SSA construction, the job LLVM's `SSAUpdater` does for its jump
//! threading). Parameters that turn out trivial are `simplify-phis`' to remove. At most
//! [`MAX_REPAIRED_VALUES`] values are repaired per thread.
//!
//! # Loops
//!
//! Threading never copies a loop header (that would enter the loop at a second block, making it
//! irreducible) and never targets one (the new edge would be a second entry or a second latch,
//! undoing `loop-canon`). The first rule also keeps the dominating facts sound: only a header has
//! an incoming edge from a block it dominates (a back edge), and there a fact about the header's
//! own compare is the previous iteration's, not the one the header is about to recompute. Copies
//! are made from blocks that are not headers, so a chain of threads along one path cannot go
//! around a cycle, and the copy count per run and the function's size are capped besides. The pass
//! runs at `-O2`, inside the fixpoint and just before `simplify-cfg`, which merges each copy into
//! its predecessor and deletes a block threading left without predecessors; `loop-canon` re-runs
//! in the same sweep.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId,
};

use crate::fxhash::FxHashMap;
use crate::{cfg, each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// The most instructions one thread copies. Above it the copy costs more than the branch it saves.
pub(crate) const MAX_THREAD_INSTS: usize = 8;
/// Threads one invocation may perform before yielding to the rest of the fixpoint.
const MAX_THREADS_PER_CALL: usize = 16;
/// No thread may grow a function past this many instructions.
const MAX_FUNCTION_INSTS: usize = 4096;
/// How far up the dominator tree the facts of an edge are collected.
const MAX_FACT_DEPTH: usize = 32;
/// How deep the condition evaluator follows operands.
const MAX_EVAL_DEPTH: u32 = 8;
/// The most values of the threaded block that may be read past it (each needs an SSA repair).
const MAX_REPAIRED_VALUES: usize = 8;

/// Jump threading (see the module docs).
pub struct JumpThread;

impl Pass for JumpThread {
    fn name(&self) -> &'static str {
        "jump-thread"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let mut changed = false;
        for _ in 0..MAX_THREADS_PER_CALL {
            if f.blocks.len() < 3
                || f.blocks
                    .iter()
                    .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
                    .count()
                    < 2
            {
                break;
            }
            if !cache.all_reachable(f) {
                cfg::prune_unreachable(f);
                cache.invalidate();
            }
            let plan = {
                let (idom, preds) = cache.idoms_and_preds(f);
                find_thread(f, idom, preds)
            };
            let Some(plan) = plan else {
                break;
            };
            apply(f, plan);
            cache.invalidate();
            changed = true;
        }
        if changed && !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        changed
    }
}

/// Which edge of the predecessor's terminator is retargeted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeSel {
    Br,
    Then,
    Else,
}

/// One thread: the edge `pred -> m` (selected by `sel`) goes to a copy of `m` that branches to
/// `m`'s then-successor (`taken`) or else-successor.
struct Thread {
    pred: usize,
    sel: EdgeSel,
    m: usize,
    taken: bool,
    /// Values of `m` read by other blocks, which need a reaching definition there afterwards.
    repair: Vec<ValueId>,
}

/// Does `a` dominate `b`? `false` when `b` is unreachable.
fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    if idom.get(b as usize).copied().unwrap_or(u32::MAX) == u32::MAX {
        return false;
    }
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        let id = idom[x as usize];
        if id == x || id == u32::MAX {
            return false;
        }
        x = id;
    }
}

fn func_insts(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// `ValueId -> (block, instruction index)` of its defining instruction, and `ValueId -> block` for
/// a block parameter.
struct Defs {
    inst: Vec<Option<(u32, u32)>>,
    param: Vec<Option<u32>>,
}

impl Defs {
    fn of(f: &Function) -> Defs {
        let n = f.value_types.len();
        let mut inst = vec![None; n];
        let mut param = vec![None; n];
        for (bi, b) in f.blocks.iter().enumerate() {
            for p in &b.params {
                if let Some(s) = param.get_mut(p.0 as usize) {
                    *s = Some(bi as u32);
                }
            }
            for (ii, i) in b.insts.iter().enumerate() {
                if let Some(r) = i.result {
                    if let Some(s) = inst.get_mut(r.0 as usize) {
                        *s = Some((bi as u32, ii as u32));
                    }
                }
            }
        }
        Defs { inst, param }
    }

    fn op<'a>(&self, f: &'a Function, v: ValueId) -> Option<&'a Op> {
        let (b, i) = (*self.inst.get(v.0 as usize)?)?;
        Some(&f.blocks[b as usize].insts[i as usize].op)
    }

    /// The block that defines `v`, by instruction or as a parameter.
    fn block(&self, v: ValueId) -> Option<u32> {
        self.inst
            .get(v.0 as usize)
            .copied()
            .flatten()
            .map(|(b, _)| b)
            .or_else(|| self.param.get(v.0 as usize).copied().flatten())
    }
}

/// For every block, the values it defines (parameters and results) that another block reads.
fn escaping_values(f: &Function, defs: &Defs) -> Vec<Vec<ValueId>> {
    let mut esc: Vec<Vec<ValueId>> = vec![Vec::new(); f.blocks.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        let mut note = |v: ValueId| {
            if let Some(d) = defs.block(v) {
                let list = &mut esc[d as usize];
                if d as usize != bi && !list.contains(&v) {
                    list.push(v);
                }
            }
        };
        for inst in &b.insts {
            each_op_use(&inst.op, &mut note);
        }
        each_term_use(&b.term, &mut note);
    }
    esc
}

/// What one instruction may not be copied for.
fn copyable(op: &Op) -> bool {
    // An allocation, or any effect but a plain store (`Op::effects`): a call of any kind, a
    // barrier, an atomic.
    !matches!(op, Op::Alloca { .. }) && (matches!(op, Op::Store { .. }) || !op.has_side_effects())
}

fn find_thread(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> Option<Thread> {
    let n = f.blocks.len();
    let entry = f.entry.0 as usize;
    let defs = Defs::of(f);
    let escaping = escaping_values(f, &defs);
    let budget = MAX_FUNCTION_INSTS.saturating_sub(func_insts(f));
    // A loop header dominates one of its predecessors (the back edge's source).
    let is_header = |b: usize| -> bool { preds[b].iter().any(|&p| dominates(b as u32, p, idom)) };
    for m in 0..n {
        let mb = &f.blocks[m];
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &mb.term
        else {
            continue;
        };
        if m == entry
            || then_blk == else_blk
            || escaping[m].len() > MAX_REPAIRED_VALUES
            || mb.insts.len() > MAX_THREAD_INSTS
            || mb.insts.len() > budget
            || !mb.insts.iter().all(|i| copyable(&i.op))
            || idom.get(m).copied().unwrap_or(u32::MAX) == u32::MAX
            || is_header(m)
            || is_header(then_blk.0 as usize)
            || is_header(else_blk.0 as usize)
        {
            continue;
        }
        for &p in &preds[m] {
            let p = p as usize;
            if p == m {
                continue;
            }
            let sel = match &f.blocks[p].term {
                Terminator::Br { target, .. } if target.0 as usize == m => EdgeSel::Br,
                Terminator::CondBr {
                    then_blk: t,
                    else_blk: e,
                    ..
                } if t != e => {
                    if t.0 as usize == m {
                        EdgeSel::Then
                    } else if e.0 as usize == m {
                        EdgeSel::Else
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            let ev = EdgeEval::new(f, &defs, idom, preds, p, sel, m);
            if let Some(taken) = ev.bool_of(*cond, 0) {
                return Some(Thread {
                    pred: p,
                    sel,
                    m,
                    taken,
                    repair: escaping[m].clone(),
                });
            }
        }
    }
    None
}

/// The arguments the edge `pred -> m` passes.
fn edge_args(f: &Function, pred: usize, sel: EdgeSel) -> &[ValueId] {
    match (&f.blocks[pred].term, sel) {
        (Terminator::Br { args, .. }, EdgeSel::Br) => args,
        (Terminator::CondBr { then_args, .. }, EdgeSel::Then) => then_args,
        (Terminator::CondBr { else_args, .. }, EdgeSel::Else) => else_args,
        _ => &[],
    }
}

/// Evaluates `m`'s condition as seen on one incoming edge.
struct EdgeEval<'a> {
    f: &'a Function,
    defs: &'a Defs,
    m: usize,
    /// `m`'s parameters, and the edge's arguments for them.
    subst: FxHashMap<u32, ValueId>,
    /// `i1` values known on the edge: branch conditions on the dominating path.
    facts: Vec<(ValueId, bool)>,
}

impl<'a> EdgeEval<'a> {
    fn new(
        f: &'a Function,
        defs: &'a Defs,
        idom: &[u32],
        preds: &[Vec<u32>],
        pred: usize,
        sel: EdgeSel,
        m: usize,
    ) -> EdgeEval<'a> {
        let args = edge_args(f, pred, sel);
        let params = &f.blocks[m].params;
        let mut subst = FxHashMap::default();
        if args.len() == params.len() {
            for (p, a) in params.iter().zip(args) {
                subst.insert(p.0, *a);
            }
        }
        let mut ev = EdgeEval {
            f,
            defs,
            m,
            subst,
            facts: Vec::new(),
        };
        // The predecessor's own branch, on the arm that leads here.
        if let Terminator::CondBr { cond, .. } = &f.blocks[pred].term {
            match sel {
                EdgeSel::Then => ev.learn(*cond, true, 0),
                EdgeSel::Else => ev.learn(*cond, false, 0),
                EdgeSel::Br => {}
            }
        }
        // Up the dominator tree: a branch whose arm is the only way into the next block down.
        let mut x = pred as u32;
        for _ in 0..MAX_FACT_DEPTH {
            let d = idom[x as usize];
            if d == x || d == u32::MAX {
                break;
            }
            if let Terminator::CondBr {
                cond,
                then_blk,
                else_blk,
                ..
            } = &f.blocks[d as usize].term
            {
                if then_blk != else_blk && preds[x as usize] == [d] {
                    if then_blk.0 == x {
                        ev.learn(*cond, true, 0);
                    } else if else_blk.0 == x {
                        ev.learn(*cond, false, 0);
                    }
                }
            }
            x = d;
        }
        ev
    }

    /// Record that `c` is `val`, and what that says about the operands of an `and`/`or`/`not`.
    fn learn(&mut self, c: ValueId, val: bool, depth: u32) {
        if depth > MAX_EVAL_DEPTH || self.facts.len() > 64 {
            return;
        }
        self.facts.push((c, val));
        match self.defs.op(self.f, c) {
            Some(Op::Bin(BinOp::And, a, b)) if val && *self.f.value_type(c) == MirType::I1 => {
                let (a, b) = (*a, *b);
                self.learn(a, true, depth + 1);
                self.learn(b, true, depth + 1);
            }
            Some(Op::Bin(BinOp::Or, a, b)) if !val && *self.f.value_type(c) == MirType::I1 => {
                let (a, b) = (*a, *b);
                self.learn(a, false, depth + 1);
                self.learn(b, false, depth + 1);
            }
            Some(Op::Not(a)) if *self.f.value_type(c) == MirType::I1 => {
                let a = *a;
                self.learn(a, !val, depth + 1);
            }
            _ => {}
        }
    }

    /// `v` as the edge sees it: a parameter of `m` is the argument the edge passes.
    fn on_edge(&self, v: ValueId) -> ValueId {
        self.subst.get(&v.0).copied().unwrap_or(v)
    }

    /// The integer constant `v` is on the edge, with its type's width.
    fn const_of(&self, v: ValueId) -> Option<(i128, u32)> {
        let v = self.on_edge(v);
        match self.defs.op(self.f, v)? {
            Op::ConstInt(c, ty) => Some((*c, int_bits(ty)?)),
            _ => None,
        }
    }

    /// Are `a` and `b` the same value on the edge: one SSA value, or two equal integer constants?
    fn same(&self, a: ValueId, b: ValueId) -> bool {
        let (a, b) = (self.on_edge(a), self.on_edge(b));
        if a == b {
            return true;
        }
        match (self.defs.op(self.f, a), self.defs.op(self.f, b)) {
            (Some(Op::ConstInt(x, tx)), Some(Op::ConstInt(y, ty))) => x == y && tx == ty,
            _ => false,
        }
    }

    /// The value of the `i1` `v` on this edge, when it is decided.
    fn bool_of(&self, v: ValueId, depth: u32) -> Option<bool> {
        if depth > MAX_EVAL_DEPTH {
            return None;
        }
        if let Some((c, _)) = self.const_of(v) {
            return Some(c & 1 != 0);
        }
        let v = self.on_edge(v);
        if let Some(&(_, val)) = self.facts.iter().find(|(c, _)| *c == v) {
            return Some(val);
        }
        // Only `m`'s own instructions and the dominating facts' compares are evaluated: a value
        // defined elsewhere is what it is on every edge, and only a fact can say which.
        let op = self.defs.op(self.f, v)?;
        let d = depth + 1;
        match op {
            Op::Cmp(p, a, b) => self.cmp_of(*p, *a, *b),
            Op::Not(a) if self.local(v) => self.bool_of(*a, d).map(|x| !x),
            Op::Bin(BinOp::And, a, b) if self.local(v) => {
                match (self.bool_of(*a, d), self.bool_of(*b, d)) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }
            Op::Bin(BinOp::Or, a, b) if self.local(v) => {
                match (self.bool_of(*a, d), self.bool_of(*b, d)) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                }
            }
            Op::Bin(BinOp::Xor, a, b) if self.local(v) => {
                Some(self.bool_of(*a, d)? != self.bool_of(*b, d)?)
            }
            _ => None,
        }
    }

    /// Is `v` an instruction of `m` (whose operands the edge may substitute)?
    fn local(&self, v: ValueId) -> bool {
        self.defs
            .inst
            .get(v.0 as usize)
            .copied()
            .flatten()
            .is_some_and(|(b, _)| b as usize == self.m)
    }

    /// `a ⋈ b` on the edge: folded when both are constants, else implied by a known compare of the
    /// same two values.
    fn cmp_of(&self, p: CmpOp, a: ValueId, b: ValueId) -> Option<bool> {
        if let (Some((x, bx)), Some((y, by))) = (self.const_of(a), self.const_of(b)) {
            if bx == by {
                return fold_cmp(p, x, y, bx);
            }
        }
        let rel_q = relation(p)?;
        for &(c, val) in &self.facts {
            let Some(Op::Cmp(fp, fa, fb)) = self.defs.op(self.f, c) else {
                continue;
            };
            // Orient the fact as `a ⋈ b`.
            let fp = if self.same(*fa, a) && self.same(*fb, b) {
                *fp
            } else if self.same(*fa, b) && self.same(*fb, a) {
                let Some(s) = swap_cmp(*fp) else { continue };
                s
            } else {
                continue;
            };
            let Some(mut rel_p) = relation(fp) else {
                continue;
            };
            if !val {
                rel_p.set = !rel_p.set & 0b111;
            }
            if let Some(answer) = implies(rel_p, rel_q) {
                return Some(answer);
            }
        }
        None
    }
}

/// An integer comparison as the set of orderings of `(a, b)` it accepts: bit 0 `a < b`, bit 1
/// `a == b`, bit 2 `a > b`, in the signed or unsigned order (`None` for `==`/`!=`, which mean the
/// same in both).
#[derive(Clone, Copy)]
struct Relation {
    signed: Option<bool>,
    set: u8,
}

const LT: u8 = 1;
const EQ: u8 = 2;
const GT: u8 = 4;

fn relation(p: CmpOp) -> Option<Relation> {
    use CmpOp::*;
    let (signed, set) = match p {
        Eq => (None, EQ),
        Ne => (None, LT | GT),
        Slt => (Some(true), LT),
        Sle => (Some(true), LT | EQ),
        Sgt => (Some(true), GT),
        Sge => (Some(true), GT | EQ),
        Ult => (Some(false), LT),
        Ule => (Some(false), LT | EQ),
        Ugt => (Some(false), GT),
        Uge => (Some(false), GT | EQ),
        // An ordered float compare is false on a NaN, so its negation is not the opposite order.
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    };
    Some(Relation { signed, set })
}

/// Given that `p` holds between two values, does `q`? `None` when it can go either way.
///
/// Two orders of different signedness relate only through equality, which means the same in both:
/// `<` and `>` of one order say nothing about the other's. Between `==`/`!=` and any order the sets
/// compare directly, since `{<, >}` of either order is exactly "not equal".
fn implies(p: Relation, q: Relation) -> Option<bool> {
    if let (Some(sp), Some(sq)) = (p.signed, q.signed) {
        if sp != sq {
            return None;
        }
    }
    if p.set & !q.set == 0 {
        Some(true)
    } else if p.set & q.set == 0 {
        Some(false)
    } else {
        None
    }
}

/// The predicate with the operands exchanged.
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
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
}

fn int_bits(ty: &MirType) -> Option<u32> {
    Some(match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return None,
    })
}

/// An integer compare of two constants of `bits` width, read as the predicate reads them.
fn fold_cmp(p: CmpOp, x: i128, y: i128, bits: u32) -> Option<bool> {
    let mask = (1i128 << bits) - 1;
    let (ux, uy) = (x & mask, y & mask);
    let sign = |u: i128| {
        if bits > 1 && u >= 1i128 << (bits - 1) {
            u - (1i128 << bits)
        } else {
            u
        }
    };
    let (sx, sy) = (sign(ux), sign(uy));
    use CmpOp::*;
    Some(match p {
        Eq => ux == uy,
        Ne => ux != uy,
        Slt => sx < sy,
        Sle => sx <= sy,
        Sgt => sx > sy,
        Sge => sx >= sy,
        Ult => ux < uy,
        Ule => ux <= uy,
        Ugt => ux > uy,
        Uge => ux >= uy,
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

fn apply(f: &mut Function, t: Thread) {
    let args: Vec<ValueId> = edge_args(f, t.pred, t.sel).to_vec();
    let (params, insts, term) = {
        let mb = &f.blocks[t.m];
        (mb.params.clone(), mb.insts.clone(), mb.term.clone())
    };
    let mut map: FxHashMap<u32, ValueId> = FxHashMap::default();
    for (p, a) in params.iter().zip(&args) {
        map.insert(p.0, *a);
    }
    let mut copied = Vec::with_capacity(insts.len());
    for inst in insts {
        let result = inst.result.map(|r| {
            let ty = f.value_type(r).clone();
            let nr = fresh(f, ty);
            // The private copy of the block is the same source as the block.
            f.copy_value_loc(r, nr);
            map.insert(r.0, nr);
            nr
        });
        let mut op = inst.op;
        map_op_uses(&mut op, |v| map.get(&v.0).copied().unwrap_or(v));
        copied.push(Inst { result, op });
    }
    let Terminator::CondBr {
        then_blk,
        then_args,
        else_blk,
        else_args,
        ..
    } = term
    else {
        unreachable!("a threaded block ends in a conditional branch");
    };
    let (target, targs) = if t.taken {
        (then_blk, then_args)
    } else {
        (else_blk, else_args)
    };
    let mut new_term = Terminator::Br {
        target,
        args: targs,
    };
    map_term_uses(&mut new_term, |v| map.get(&v.0).copied().unwrap_or(v));
    let id = BlockId(f.blocks.len() as u32);
    f.blocks.push(BasicBlock {
        id,
        params: Vec::new(),
        insts: copied,
        term: new_term,
    });
    retarget(f, t.pred, t.sel, id);
    // Repair SSA for the values of `m` read past it: they now have a second definition in the copy.
    if t.repair.is_empty() {
        return;
    }
    let preds = cfg::predecessors(f);
    let copy = id.0 as usize;
    for v in t.repair {
        let Some(&v2) = map.get(&v.0) else {
            continue;
        };
        let mut up = Updater {
            preds: &preds,
            m: t.m,
            copy,
            v,
            v2,
            ty: f.value_type(v).clone(),
            at_entry: FxHashMap::default(),
        };
        // The blocks that read `v`, other than the two that define it.
        let users: Vec<usize> = (0..f.blocks.len())
            .filter(|&b| b != t.m && b != copy)
            .filter(|&b| {
                let blk = &f.blocks[b];
                let mut reads = false;
                for inst in &blk.insts {
                    each_op_use(&inst.op, &mut |u| reads |= u == v);
                }
                each_term_use(&blk.term, &mut |u| reads |= u == v);
                reads
            })
            .collect();
        for b in users {
            let r = up.at_entry(f, b);
            if r == v {
                continue;
            }
            let blk = &mut f.blocks[b];
            for inst in &mut blk.insts {
                map_op_uses(&mut inst.op, |u| if u == v { r } else { u });
            }
            map_term_uses(&mut blk.term, |u| if u == v { r } else { u });
        }
    }
}

/// On-demand SSA reconstruction for one value with two definitions: `v` at the end of `m` and `v2`
/// at the end of its copy (see the module docs).
struct Updater<'a> {
    preds: &'a [Vec<u32>],
    m: usize,
    copy: usize,
    v: ValueId,
    v2: ValueId,
    ty: MirType,
    /// The value reaching the entry of each block visited, a new parameter at a join.
    at_entry: FxHashMap<u32, ValueId>,
}

impl Updater<'_> {
    fn at_exit(&mut self, f: &mut Function, b: usize) -> ValueId {
        if b == self.m {
            self.v
        } else if b == self.copy {
            self.v2
        } else {
            self.at_entry(f, b)
        }
    }

    fn at_entry(&mut self, f: &mut Function, b: usize) -> ValueId {
        if let Some(&x) = self.at_entry.get(&(b as u32)) {
            return x;
        }
        let mut ps: Vec<u32> = self.preds[b].clone();
        ps.dedup();
        match ps.as_slice() {
            // Dead code (the pass prunes it): any value of the right type will do.
            [] => self.v,
            [p] => {
                let x = self.at_exit(f, *p as usize);
                self.at_entry.insert(b as u32, x);
                x
            }
            _ => {
                // Cached before the predecessors are asked, so a cycle back here finds it.
                let param = fresh(f, self.ty.clone());
                f.blocks[b].params.push(param);
                self.at_entry.insert(b as u32, param);
                for p in ps {
                    let x = self.at_exit(f, p as usize);
                    f.blocks[p as usize].term.for_each_edge_mut(|target, args| {
                        if target.0 as usize == b {
                            args.push(x);
                        }
                    });
                }
                param
            }
        }
    }
}

/// Point the selected edge of `pred`'s terminator at the argument-less block `to`.
fn retarget(f: &mut Function, pred: usize, sel: EdgeSel, to: BlockId) {
    let id = to;
    match (&mut f.blocks[pred].term, sel) {
        (Terminator::Br { target, args }, EdgeSel::Br) => {
            *target = id;
            args.clear();
        }
        (
            Terminator::CondBr {
                then_blk,
                then_args,
                ..
            },
            EdgeSel::Then,
        ) => {
            *then_blk = id;
            then_args.clear();
        }
        (
            Terminator::CondBr {
                else_blk,
                else_args,
                ..
            },
            EdgeSel::Else,
        ) => {
            *else_blk = id;
            else_args.clear();
        }
        _ => unreachable!("a thread is planned against this very terminator"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dce, Simplify, SimplifyCfg};
    use wukong_mir::{Builder, CastKind, Program};
    use wukong_span::{Interner, SourceId};

    /// Call `f(x, y)` from `main` for each input pair, printing every result.
    fn run_xy(f: &Function, it: &mut Interner, inputs: &[(i128, i128)]) -> Vec<u8> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &(x, y) in inputs {
            let a = m.build(MirType::I32, Op::ConstInt(x, MirType::I32));
            let b = m.build(MirType::I32, Op::ConstInt(y, MirType::I32));
            let r = m.build(
                MirType::I32,
                Op::Call {
                    func: f.name,
                    args: vec![a, b],
                },
            );
            m.build_void(Op::Call {
                func: it.intern("print"),
                args: vec![r],
            });
        }
        let z = m.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        m.ret(Some(z));
        let mut p = Program::new();
        p.funcs = vec![f.clone(), m.finish()];
        let errs = wukong_mir::verify::verify_program(&p);
        assert!(errs.is_empty(), "{errs:?}");
        let main = it.intern("main");
        let (code, out) = wukong_interp::run_with_output(&p, main, it).unwrap();
        assert_eq!(code, 0);
        out
    }

    const GRID: [(i128, i128); 9] = [
        (0, 0),
        (1, 2),
        (2, 1),
        (-3, 4),
        (4, -3),
        (-1, -1),
        (7, 7),
        (2147483647, -2147483648),
        (-2147483648, 2147483647),
    ];

    fn cond_brs(f: &Function) -> usize {
        f.blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
            .count()
    }

    fn cleanup(f: &mut Function) {
        let mut cache = CfgAnalyses::default();
        for _ in 0..4 {
            let mut c = Simplify.run_function(f, &mut cache);
            c |= SimplifyCfg.run_function(f, &mut cache);
            c |= Dce.run_function(f, &mut cache);
            if !c {
                break;
            }
        }
    }

    /// How the merge block of [`retest`] spells its second test of `x ⋈ y`.
    #[derive(Clone, Copy, Debug)]
    enum Retest {
        /// The dominating compare's own value.
        Same,
        /// A recomputed compare with this predicate, on `(x, y)` or, swapped, on `(y, x)`.
        Recomputed(CmpOp, bool),
    }

    /// `f(x, y)`: two diamonds on one condition, the second test in a merge block that also does
    /// some arithmetic of its own on the first diamond's result:
    ///
    /// ```text
    ///   entry: c = x <first y; cond_br c, A, B
    ///   A:     br M(x * 3)
    ///   B:     br M(y - 5)
    ///   M(p):  q = p + 100; t = <retest>; cond_br t, T(q), E(q)
    ///   T(r):  ret r * 2
    ///   E(r):  ret r * 3 + 1
    /// ```
    fn retest(it: &mut Interner, first: CmpOp, second: Retest) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let c = b.build(MirType::I1, Op::Cmp(first, x, y));
        let a_blk = b.new_block();
        let b_blk = b.new_block();
        let m = b.new_block();
        let p = b.block_param(m, MirType::I32);
        let t_blk = b.new_block();
        let tr = b.block_param(t_blk, MirType::I32);
        let e_blk = b.new_block();
        let er = b.block_param(e_blk, MirType::I32);
        b.cond_br(c, a_blk, vec![], b_blk, vec![]);
        b.switch_to(a_blk);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let x3 = b.build(MirType::I32, Op::Bin(BinOp::Mul, x, three));
        b.br(m, vec![x3]);
        b.switch_to(b_blk);
        let five = b.build(MirType::I32, Op::ConstInt(5, MirType::I32));
        let y5 = b.build(MirType::I32, Op::Bin(BinOp::Sub, y, five));
        b.br(m, vec![y5]);
        b.switch_to(m);
        let hundred = b.build(MirType::I32, Op::ConstInt(100, MirType::I32));
        let q = b.build(MirType::I32, Op::Bin(BinOp::Add, p, hundred));
        let t = match second {
            Retest::Same => c,
            Retest::Recomputed(pred, false) => b.build(MirType::I1, Op::Cmp(pred, x, y)),
            Retest::Recomputed(pred, true) => b.build(MirType::I1, Op::Cmp(pred, y, x)),
        };
        b.cond_br(t, t_blk, vec![q], e_blk, vec![q]);
        b.switch_to(t_blk);
        let two = b.build(MirType::I32, Op::ConstInt(2, MirType::I32));
        let r2 = b.build(MirType::I32, Op::Bin(BinOp::Mul, tr, two));
        b.ret(Some(r2));
        b.switch_to(e_blk);
        let three2 = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let r3 = b.build(MirType::I32, Op::Bin(BinOp::Mul, er, three2));
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let r4 = b.build(MirType::I32, Op::Bin(BinOp::Add, r3, one));
        b.ret(Some(r4));
        b.finish()
    }

    /// A merge block re-testing what the branch above it already decided -- the same compare, or a
    /// compare of the same values its outcome implies, either operand order -- is threaded on every
    /// incoming edge that decides it, and every answer is the same. Below `x < y`, a re-test `x < y`,
    /// `x >= y`, `y > x` or `y <= x` is decided on both edges, so one conditional branch is left;
    /// `x <= y`, `x != y` and `x == y` only on the true edge (`x >= y` leaves `x == y` open), so the
    /// false edge still runs the re-test.
    #[test]
    fn a_retest_implied_by_the_dominating_compare_is_threaded() {
        use CmpOp::*;
        let seconds = [
            (Retest::Same, 1),
            (Retest::Recomputed(Slt, false), 1),
            (Retest::Recomputed(Sge, false), 1),
            (Retest::Recomputed(Sgt, true), 1),
            (Retest::Recomputed(Sle, true), 1),
            (Retest::Recomputed(Sle, false), 2),
            (Retest::Recomputed(Ne, false), 2),
            (Retest::Recomputed(Eq, false), 2),
        ];
        for (second, left) in seconds {
            let mut it = Interner::new();
            let mut f = retest(&mut it, Slt, second);
            let before = run_xy(&f, &mut it, &GRID);
            assert_eq!(cond_brs(&f), 2);
            assert!(
                JumpThread.run_function(&mut f, &mut CfgAnalyses::default()),
                "{second:?}"
            );
            assert!(
                wukong_mir::verify::verify_function(&f).is_empty(),
                "{second:?}"
            );
            assert_eq!(run_xy(&f, &mut it, &GRID), before, "{second:?}");
            cleanup(&mut f);
            assert_eq!(
                cond_brs(&f),
                left,
                "{second:?}\n{}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(run_xy(&f, &mut it, &GRID), before, "{second:?}");
        }
    }

    /// Facts decide only what they prove. A signed order says nothing about the unsigned one (or
    /// the reverse) on either arm. Floats: `a < b` false does not make `a >= b` true (both are false
    /// on a NaN).
    #[test]
    fn a_retest_the_dominating_compare_does_not_decide_is_left_alone() {
        use CmpOp::*;
        for (first, second) in [
            (Slt, Retest::Recomputed(Ult, false)),
            (Ult, Retest::Recomputed(Sgt, false)),
            (Sle, Retest::Recomputed(Ugt, false)),
            (Uge, Retest::Recomputed(Sle, true)),
        ] {
            let mut it = Interner::new();
            let mut f = retest(&mut it, first, second);
            let before = run_xy(&f, &mut it, &GRID);
            assert!(
                !JumpThread.run_function(&mut f, &mut CfgAnalyses::default()),
                "{first:?} then {second:?} is not decided on either edge\n{}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(run_xy(&f, &mut it, &GRID), before);
        }
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let fx = b.build(MirType::F32, Op::Cast(CastKind::SiToFp, x, MirType::F32));
        let fy = b.build(MirType::F32, Op::Cast(CastKind::SiToFp, y, MirType::F32));
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Folt, fx, fy));
        let a_blk = b.new_block();
        let m = b.new_block();
        let t_blk = b.new_block();
        let e_blk = b.new_block();
        b.cond_br(c, a_blk, vec![], m, vec![]);
        b.switch_to(a_blk);
        b.br(m, vec![]);
        b.switch_to(m);
        let t = b.build(MirType::I1, Op::Cmp(CmpOp::Foge, fx, fy));
        b.cond_br(t, t_blk, vec![], e_blk, vec![]);
        b.switch_to(t_blk);
        b.ret(Some(x));
        b.switch_to(e_blk);
        b.ret(Some(y));
        let mut f = b.finish();
        assert!(!JumpThread.run_function(&mut f, &mut CfgAnalyses::default()));
    }

    /// `f(x, y)`: three edges into `M(p): t = p < 10; w = p * 10; cond_br t, T(w), E(p)` passing
    /// 3, 12 and the runtime `x`. With `escape`, `T` and `E` read `w` and `p` directly instead.
    fn constant_edges(it: &mut Interner, escape: bool) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let c1 = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, y, zero));
        let c2 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, y, zero));
        let a_blk = b.new_block();
        let rest = b.new_block();
        let b_blk = b.new_block();
        let c_blk = b.new_block();
        let m = b.new_block();
        let p = b.block_param(m, MirType::I32);
        let t_blk = b.new_block();
        let e_blk = b.new_block();
        let (tw, ep) = if escape {
            (None, None)
        } else {
            (
                Some(b.block_param(t_blk, MirType::I32)),
                Some(b.block_param(e_blk, MirType::I32)),
            )
        };
        b.cond_br(c1, a_blk, vec![], rest, vec![]);
        b.switch_to(rest);
        b.cond_br(c2, b_blk, vec![], c_blk, vec![]);
        b.switch_to(a_blk);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        b.br(m, vec![three]);
        b.switch_to(b_blk);
        let twelve = b.build(MirType::I32, Op::ConstInt(12, MirType::I32));
        b.br(m, vec![twelve]);
        b.switch_to(c_blk);
        b.br(m, vec![x]);
        b.switch_to(m);
        let ten = b.build(MirType::I32, Op::ConstInt(10, MirType::I32));
        let t = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, p, ten));
        let w = b.build(MirType::I32, Op::Bin(BinOp::Mul, p, ten));
        if escape {
            b.cond_br(t, t_blk, vec![], e_blk, vec![]);
        } else {
            b.cond_br(t, t_blk, vec![w], e_blk, vec![p]);
        }
        b.switch_to(t_blk);
        b.ret(Some(tw.unwrap_or(w)));
        b.switch_to(e_blk);
        b.ret(Some(ep.unwrap_or(p)));
        b.finish()
    }

    /// A constant argument decides a compare computed in the block: the edges passing 3 (to the
    /// true successor) and 12 (to the false one) are threaded, the edge passing the runtime `x` is
    /// not. Whether the block's values reach its successors as branch arguments or are read there
    /// directly (repaired with new parameters where the copy and the original meet), every answer
    /// is the same.
    #[test]
    fn a_constant_argument_decides_a_compare_in_the_block() {
        let inputs = [(5, 1), (5, -1), (5, 0), (20, 0), (-7, 0), (10, 0), (9, 0)];
        // y > 0: p = 3 -> 30; y < 0: p = 12 -> 12; y == 0: p = x -> x < 10 ? 10x : x.
        let expected = b"30\n12\n50\n20\n-70\n10\n90\n".to_vec();
        for escape in [false, true] {
            let mut it = Interner::new();
            let mut f = constant_edges(&mut it, escape);
            assert_eq!(run_xy(&f, &mut it, &inputs), expected);
            assert!(JumpThread.run_function(&mut f, &mut CfgAnalyses::default()));
            let errs = wukong_mir::verify::verify_function(&f);
            assert!(
                errs.is_empty(),
                "{errs:?}\n{}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(run_xy(&f, &mut it, &inputs), expected, "escape={escape}");
            cleanup(&mut f);
            // `y > 0`, `y < 0`, and the runtime edge's own test of `x < 10`.
            assert_eq!(
                cond_brs(&f),
                3,
                "{}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(run_xy(&f, &mut it, &inputs), expected, "escape={escape}");
        }
    }

    /// Does the edge `pred -> m` decide `cond`, and to what? The question `find_thread` asks of every
    /// edge, asked of one: a loop test below must be decided on its edge, or "not threaded" would
    /// prove nothing about the loop rule.
    fn decided_on_edge(
        f: &Function,
        pred: BlockId,
        sel: EdgeSel,
        m: BlockId,
        cond: ValueId,
    ) -> Option<bool> {
        let mut cache = CfgAnalyses::default();
        let (idom, preds) = cache.idoms_and_preds(f);
        let defs = Defs::of(f);
        EdgeEval::new(f, &defs, idom, preds, pred.0 as usize, sel, m.0 as usize).bool_of(cond, 0)
    }

    fn plan_of(f: &Function) -> Option<(usize, usize, bool)> {
        let mut cache = CfgAnalyses::default();
        let (idom, preds) = cache.idoms_and_preds(f);
        find_thread(f, idom, preds).map(|t| (t.pred, t.m, t.taken))
    }

    /// A loop header is never copied, even when the edge entering the loop decides its test:
    ///
    /// ```text
    ///   entry: big = x > 100; cond_br big, pre, out
    ///   pre:   br H(0)
    ///   H(i):  i1 = i + 1; c = i < 5; cond_br c, body, exit     -- `0 < 5` on the edge from pre
    ///   body:  br H(i1)
    ///   exit:  ret i + x
    ///   out:   ret x
    /// ```
    ///
    /// Threading `pre -> H` would enter the loop at `body` through a copy of `H`, a second entry
    /// that makes the loop irreducible. (The back edge `body -> H` would look decided too: below
    /// `H`'s true arm `c` is known true, but that is the previous iteration's `c`.) The entry's
    /// branch is there so the function has the two conditional branches `run_function` requires
    /// before it looks for a thread at all.
    #[test]
    fn a_loop_header_is_never_copied() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let _y = b.add_param(MirType::I32);
        let hundred = b.build(MirType::I32, Op::ConstInt(100, MirType::I32));
        let big = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, hundred));
        let pre = b.new_block();
        let header = b.new_block();
        let i = b.block_param(header, MirType::I32);
        let body = b.new_block();
        let exit = b.new_block();
        let out = b.new_block();
        b.cond_br(big, pre, vec![], out, vec![]);
        b.switch_to(pre);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.br(header, vec![zero]);
        b.switch_to(header);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let i1 = b.build(MirType::I32, Op::Bin(BinOp::Add, i, one));
        let five = b.build(MirType::I32, Op::ConstInt(5, MirType::I32));
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, five));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        b.br(header, vec![i1]);
        b.switch_to(exit);
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, i, x));
        b.ret(Some(s));
        b.switch_to(out);
        b.ret(Some(x));
        let mut f = b.finish();
        let inputs = [(200, 0), (101, 0), (100, 0), (4, 0), (-2, 0)];
        // x > 100: the loop leaves i = 5, so x + 5; otherwise x.
        let expected = b"205\n106\n100\n4\n-2\n".to_vec();
        assert_eq!(run_xy(&f, &mut it, &inputs), expected);
        assert_eq!(cond_brs(&f), 2);
        assert_eq!(decided_on_edge(&f, pre, EdgeSel::Br, header, c), Some(true));
        assert_eq!(plan_of(&f), None);
        assert!(!JumpThread.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(run_xy(&f, &mut it, &inputs), expected);
    }

    /// Nor is a loop header ever a thread's target, even when the edge into the block that enters
    /// the loop decides that block's branch:
    ///
    /// ```text
    ///   entry: c0 = y > 0; cond_br c0, A, B
    ///   A:     br M(true)
    ///   B:     br M(y < -5)
    ///   M(p):  cond_br p, H(0), out                              -- `true` on the edge from A
    ///   H(i):  c = i < x; cond_br c, body, exit
    ///   body:  br H(i + 1)
    ///   exit:  ret i
    ///   out:   ret -1
    /// ```
    ///
    /// Threading `A -> M` would give `H` a second entering edge (from the copy of `M`) beside `M`'s,
    /// undoing `loop-canon`'s single preheader.
    #[test]
    fn a_loop_header_is_never_a_thread_target() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let c0 = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, y, zero));
        let a_blk = b.new_block();
        let b_blk = b.new_block();
        let m = b.new_block();
        let p = b.block_param(m, MirType::I1);
        let header = b.new_block();
        let i = b.block_param(header, MirType::I32);
        let body = b.new_block();
        let exit = b.new_block();
        let out = b.new_block();
        b.cond_br(c0, a_blk, vec![], b_blk, vec![]);
        b.switch_to(a_blk);
        let yes = b.build(MirType::I1, Op::ConstInt(1, MirType::I1));
        b.br(m, vec![yes]);
        b.switch_to(b_blk);
        let minus5 = b.build(MirType::I32, Op::ConstInt(-5, MirType::I32));
        let c1 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, y, minus5));
        b.br(m, vec![c1]);
        b.switch_to(m);
        b.cond_br(p, header, vec![zero], out, vec![]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, x));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let i1 = b.build(MirType::I32, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![i1]);
        b.switch_to(exit);
        b.ret(Some(i));
        b.switch_to(out);
        let minus1 = b.build(MirType::I32, Op::ConstInt(-1, MirType::I32));
        b.ret(Some(minus1));
        let mut f = b.finish();
        let inputs = [(3, 1), (-2, 1), (3, -6), (3, -5), (3, 0), (0, 7)];
        // y > 0 or y < -5: the loop counts to max(x, 0); otherwise -1.
        let expected = b"3\n0\n3\n-1\n-1\n0\n".to_vec();
        assert_eq!(run_xy(&f, &mut it, &inputs), expected);
        assert_eq!(cond_brs(&f), 3);
        assert_eq!(decided_on_edge(&f, a_blk, EdgeSel::Br, m, p), Some(true));
        assert_eq!(plan_of(&f), None);
        assert!(!JumpThread.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(run_xy(&f, &mut it, &inputs), expected);
    }

    /// From source: a value guarded twice by the same condition runs one test at `-O2`, and every
    /// level prints what the scalar code says.
    #[test]
    fn source_double_tests_become_one() {
        let src = concat!(
            "fn twice(x: i32, n: i32) -> i32 { let mut a = 0; let mut b = 0; ",
            "if x < n { a = x * 2; } else { a = n - x; } ",
            "if x < n { b = a + 1; } else { b = a * 3; } return b; }\n",
            "fn main() -> i32 { for x in 0 - 3..4 { for n in 0 - 2..3 { print(twice(x, n)); } } ",
            "return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let mut expected = String::new();
        for x in -3i32..4 {
            for n in -2i32..3 {
                let a = if x < n { x * 2 } else { n - x };
                let b = if x < n { a + 1 } else { a * 3 };
                expected.push_str(&format!("{b}\n"));
            }
        }
        let main = interner.intern("main");
        for level in [0u8, 1, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            assert_eq!(String::from_utf8(stdout).unwrap(), expected, "O{level}");
        }
        let mut optimized = program.clone();
        crate::PassManager::standard(2).run(&mut optimized);
        let twice = optimized.function(interner.intern("twice")).unwrap();
        assert_eq!(
            cond_brs(twice),
            1,
            "{}",
            wukong_mir::print::print_function(twice, &interner)
        );
    }
}
