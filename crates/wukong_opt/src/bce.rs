//! Bounds-check elimination: version a counted loop on one guard so its slice checks disappear.
//!
//! `mir_build` checks every slice access (`wukong_mir_build/src/bounds.rs`):
//!
//! ```text
//!     len = load i64 (hdr + 8)
//!     ok  = cmp ult idx, len
//!     cond_br ok, cont, fail        ; fail: call wukong_rt_bounds_fail(..); unreachable
//! ```
//!
//! In a hot loop that is a load, a compare and a branch per access, and — worse — every check is a
//! second loop exit, so `loop_info::exit_test` finds no single exit, no trip count, and the
//! vectorizer declines the loop outright. This pass removes those checks without changing what the
//! program does, by **loop versioning**:
//!
//! ```text
//!   P:      runs = start < end ? → G0 : slow.ph
//!   G0:     every "free" guard (below) → G1 / fast.ph : slow.ph
//!   G1..Gk: one "ordered" guard each, in dominance order → next : slow.ph
//!   fast.ph → fast loop   (a clone of the loop with the proved checks replaced by plain branches)
//!   slow.ph → the original loop, unchanged, checks and all
//! ```
//!
//! The fast loop runs only when the guards prove every eliminated check true **for every iteration
//! the loop can execute**, so the checks it drops could never have failed. Otherwise the original
//! loop runs and stops at exactly the iteration — with exactly the message — it always did. Both
//! copies exit to the same block, whose new parameters merge the values the loop leaves behind.
//!
//! # The guard, and why it is sound under wrapping arithmetic
//!
//! The loop must be top-tested with `iv < end` (signed, or unsigned for an unsigned counter) on a
//! header parameter stepping by exactly +1 from a `start` the preheader passes in. It then runs
//! `iv = start, start+1, …, end-1` (when `start < end`), and `iv + 1` never wraps because
//! `iv < end`. An eliminated check's index must be `±iv + Σ invariant` in `i64` (a widening cast is
//! allowed directly on a narrower counter, `sext` for a signed one and `zext` for an unsigned one),
//! so evaluated at `iv = start` and `iv = end - 1` it gives the two ends `a`, `b` of a run of
//! consecutive values modulo 2^64. With `lo, hi` = `(a, b)` for `+iv` and `(b, a)` for `-iv`:
//!
//! ```text
//!     guard = lo <=u hi  &&  hi <u len
//! ```
//!
//! If the run does not wrap, every value lies in `[lo, hi] ⊂ [0, len)`. If it wraps (once — it has
//! at most 2^64 - 1 elements) then `hi < lo` unsigned and the guard fails. `len < 2^63` always, so a
//! negative index fails exactly as the check would. An index that does not move (`iv` absent) is the
//! single compare `idx <u len`.
//!
//! # Where `len` comes from, and when it may be read early
//!
//! The guard runs before the loop, so it reads each slice length **before** the original program
//! would have. That needs two facts:
//!
//! * **Stability** — nothing in the loop may write the length. The loop may contain no call, and
//!   every store is checked against the length's address with [`AliasInfo`], plus one rule `alias`
//!   cannot know and the language guarantees: Wukong has no type punning (sema accepts a
//!   pointer-to-pointer cast only between identical pointees and rejects pointer-to-integer), and a
//!   fat pointer's length is only ever written through a byte-addressed field `gep` or a byte copy,
//!   so a floating-point store, or a store of a non-byte scalar through a `gep` whose element type is
//!   that scalar (an element of a typed array), can never land on a slice header.
//! * **Speculation** — reading the length early must not fault where the program would not have.
//!   A length already in an SSA value, or in a stack slot of this function
//!   ([`AliasInfo::is_dereferenceable`]), is free to read: it goes in `G0`. A length behind a
//!   pointer the caller handed in (a `[]T` parameter's fat pointer) is read only where the original
//!   iteration 0 provably reads it too: its check must execute on every iteration (it dominates every
//!   latch), the guard chain reaches it only after `start < end` and every earlier guard passed, and
//!   no check that stays in the loop can stop iteration 0 before it (it dominates every such check).
//!
//! # What it declines
//!
//! Loops with a call or a vector-kernel call (a call could rewrite a header, and cloning one would
//! duplicate a dispatched kernel), with more than one normal exit, with an exit block shared with
//! other code, larger than [`MAX_LOOP_INSTS`] instructions, or that are not innermost. An index that
//! is not of the form above keeps its check — in both copies. Nothing here is a heuristic about
//! *whether* a check can fail; every elimination is a proof plus a runtime guard.
//!
//! # Termination
//!
//! The original loop's eliminated checks are rewritten from `cmp ult idx, len` to the equivalent
//! `cmp ugt len, idx`, and a loop holding such a check is never versioned again. The fast copy has
//! strictly fewer candidate checks than the loop it came from. So each loop is versioned at most
//! once per check it started with, and [`MAX_VERSIONS_PER_CALL`] bounds a single invocation.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId,
};

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{self, IvStep, NaturalLoop, Stages};
use crate::{cfg, each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

pub struct Bce;

/// The largest loop (in instructions) worth duplicating. Hot inner loops are far smaller; past this
/// the checks stay, which costs speed, never correctness.
const MAX_LOOP_INSTS: usize = 512;
/// The largest loop, in blocks.
const MAX_LOOP_BLOCKS: usize = 64;
/// Versionings one invocation may perform before yielding to the rest of the fixpoint.
const MAX_VERSIONS_PER_CALL: usize = 32;
/// How deep an index expression is followed.
const MAX_EXPR_DEPTH: u32 = 16;

impl Pass for Bce {
    fn name(&self) -> &'static str {
        "bce"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // The steady state of almost every function: no check left to look at.
        if f.blocks.len() < 4 || !has_candidate_check(f) {
            return false;
        }
        let mut changed = false;
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate();
            changed = true;
        }
        for _ in 0..MAX_VERSIONS_PER_CALL {
            let plan = {
                let (idom, preds) = cache.idoms_and_preds(f);
                find_plan(f, idom, preds)
            };
            let Some(plan) = plan else {
                break;
            };
            apply(f, &plan);
            cache.invalidate();
            changed = true;
        }
        changed
    }
}

// ---------------------------------------------------------------------------------------------
// Recognition
// ---------------------------------------------------------------------------------------------

/// A block that only stops the program: no parameters and a `ret`/`unreachable` terminator — the
/// shape of a bounds check's failure block.
fn is_dead_end(f: &Function, b: BlockId) -> bool {
    f.blocks.get(b.0 as usize).is_some_and(|blk| {
        blk.params.is_empty() && matches!(blk.term, Terminator::Unreachable | Terminator::Ret(_))
    })
}

/// Is there any branch on a `cmp ult` whose false arm is a dead end? A cheap whole-function scan,
/// so a function with no slice check pays nothing more.
fn has_candidate_check(f: &Function) -> bool {
    let mut ult: FxHashSet<u32> = FxHashSet::default();
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::Cmp(CmpOp::Ult, ..)) = (inst.result, &inst.op) {
                ult.insert(r.0);
            }
        }
    }
    !ult.is_empty()
        && f.blocks.iter().any(|b| {
            matches!(&b.term, Terminator::CondBr { cond, else_blk, .. }
                if ult.contains(&cond.0) && is_dead_end(f, *else_blk))
        })
}

/// Where each value is defined.
struct Defs {
    /// Defining block per value (`u32::MAX`: no definition).
    block: Vec<u32>,
    /// `(block, instruction index)` per instruction result.
    inst: Vec<Option<(u32, u32)>>,
}

impl Defs {
    fn of(f: &Function) -> Defs {
        let n = f.value_types.len();
        let mut block = vec![u32::MAX; n];
        let mut inst = vec![None; n];
        for b in &f.blocks {
            for p in &b.params {
                if let Some(slot) = block.get_mut(p.0 as usize) {
                    *slot = b.id.0;
                }
            }
            for (i, ins) in b.insts.iter().enumerate() {
                if let Some(r) = ins.result {
                    if (r.0 as usize) < n {
                        block[r.0 as usize] = b.id.0;
                        inst[r.0 as usize] = Some((b.id.0, i as u32));
                    }
                }
            }
        }
        Defs { block, inst }
    }

    fn block_of(&self, v: ValueId) -> Option<u32> {
        self.block
            .get(v.0 as usize)
            .copied()
            .filter(|b| *b != u32::MAX)
    }

    fn at(&self, v: ValueId) -> Option<(u32, u32)> {
        self.inst.get(v.0 as usize).copied().flatten()
    }

    fn op<'f>(&self, f: &'f Function, v: ValueId) -> Option<&'f Op> {
        let (b, i) = self.at(v)?;
        Some(&f.blocks[b as usize].insts[i as usize].op)
    }
}

/// Does `a` dominate `b`? `idom[entry] == entry`.
fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        match idom.get(x as usize) {
            Some(&d) if d != x && d != u32::MAX => x = d,
            _ => return false,
        }
    }
}

fn dom_depth(b: u32, idom: &[u32]) -> u32 {
    let mut depth = 0;
    let mut x = b;
    while let Some(&d) = idom.get(x as usize) {
        if d == x || d == u32::MAX {
            break;
        }
        depth += 1;
        x = d;
    }
    depth
}

/// An eliminated check's index, as a tree over the induction variable and loop-invariant values,
/// re-emitted in the guard with the induction variable replaced by an endpoint.
#[derive(Clone, Debug)]
enum IdxExpr {
    Iv,
    /// A value defined outside the loop (it dominates the preheader).
    Outside(ValueId),
    Const(i128, MirType),
    Bin(BinOp, Box<IdxExpr>, Box<IdxExpr>, MirType),
    Cast(CastKind, Box<IdxExpr>, MirType),
}

#[derive(Clone, Copy, Debug)]
enum LenSrc {
    /// The length is already an SSA value defined outside the loop.
    Value(ValueId),
    /// The length is loaded from this loop-invariant address, which nothing in the loop writes.
    Load(ValueId),
}

#[derive(Clone, Debug)]
struct CheckPlan {
    /// The block whose `cond_br` is the check.
    block: BlockId,
    /// Where the check's `cmp ult` is, to mark it in the original loop.
    cmp_at: (u32, u32),
    idx: IdxExpr,
    /// The index's coefficient on the induction variable: -1, 0 or +1.
    coeff: i64,
    len: LenSrc,
}

struct Plan {
    header: BlockId,
    preheader: BlockId,
    exit: BlockId,
    blocks: Vec<BlockId>,
    dead_ends: Vec<BlockId>,
    start: ValueId,
    bound: ValueId,
    iv_ty: MirType,
    unsigned: bool,
    /// Checks whose guard may be evaluated in `G0`.
    free: Vec<CheckPlan>,
    /// Checks whose length read must follow the guards before it, in dominance order.
    ordered: Vec<CheckPlan>,
}

fn find_plan(f: &Function, idom: &[u32], preds: &[Vec<u32>]) -> Option<Plan> {
    let mut forest = loop_info::structure_with(f, idom, preds);
    if forest.is_empty() {
        return None;
    }
    let defs = Defs::of(f);
    let mut alias: Option<AliasInfo> = None;
    for id in forest.innermost_first() {
        let idx = id.0 as usize;
        if !cheap_shape_ok(f, forest.get(id)) {
            continue;
        }
        loop_info::analyze_one(f, &mut forest, idx, Stages::Ivs);
        let ai = alias.get_or_insert_with(|| AliasInfo::analyze(f));
        if let Some(plan) = plan_loop(f, forest.get(id), idom, preds, &defs, ai) {
            return Some(plan);
        }
    }
    None
}

/// The structural filters that need no value analysis.
fn cheap_shape_ok(f: &Function, l: &NaturalLoop) -> bool {
    if !l.children.is_empty()
        || l.preheader.is_none()
        || !l.abnormal_exits.is_empty()
        || l.blocks.len() > MAX_LOOP_BLOCKS
        || l.exits.len() < 2
    {
        return false;
    }
    let insts: usize = l.blocks.iter().map(|b| f.block(*b).insts.len()).sum();
    insts <= MAX_LOOP_INSTS
}

/// Everything the per-check analysis needs about one loop.
struct Ctx<'a> {
    f: &'a Function,
    defs: &'a Defs,
    in_loop: &'a FxHashSet<u32>,
    blocks: &'a [BlockId],
    iv: ValueId,
    unsigned: bool,
}

impl Ctx<'_> {
    /// Defined outside the loop (so invariant, and available in the preheader).
    fn outside(&self, v: ValueId) -> bool {
        self.defs
            .block_of(v)
            .is_some_and(|b| !self.in_loop.contains(&b))
    }

    /// `v` as `coeff · iv + invariant` with `coeff ∈ {-1, 0, 1}`, or `None`.
    fn affine(&self, v: ValueId, depth: u32) -> Option<(IdxExpr, i64)> {
        if v == self.iv {
            return Some((IdxExpr::Iv, 1));
        }
        if self.outside(v) {
            return Some((IdxExpr::Outside(v), 0));
        }
        if depth >= MAX_EXPR_DEPTH {
            return None;
        }
        let ty = self.f.value_type(v).clone();
        match self.defs.op(self.f, v)? {
            Op::ConstInt(k, cty) => Some((IdxExpr::Const(*k, cty.clone()), 0)),
            Op::Bin(op @ (BinOp::Add | BinOp::Sub), a, b) => {
                let (ea, ca) = self.affine(*a, depth + 1)?;
                let (eb, cb) = self.affine(*b, depth + 1)?;
                let c = if *op == BinOp::Add {
                    ca.checked_add(cb)?
                } else {
                    ca.checked_sub(cb)?
                };
                if c.abs() > 1 {
                    return None;
                }
                Some((IdxExpr::Bin(*op, Box::new(ea), Box::new(eb), ty), c))
            }
            // A widening cast keeps consecutive counter values consecutive only in the counter's own
            // signedness: `sext` of a signed counter, `zext` of an unsigned one, and only directly on
            // the counter (a narrow `iv + k` may wrap before it is widened).
            Op::Cast(kind @ (CastKind::SExt | CastKind::ZExt), x, to) => {
                let (ex, cx) = self.affine(*x, depth + 1)?;
                let widening_ok = if self.unsigned {
                    *kind == CastKind::ZExt
                } else {
                    *kind == CastKind::SExt
                };
                let directly_on_iv = matches!(ex, IdxExpr::Iv);
                match cx {
                    0 => Some((IdxExpr::Cast(*kind, Box::new(ex), to.clone()), 0)),
                    1 if directly_on_iv && widening_ok => {
                        Some((IdxExpr::Cast(*kind, Box::new(ex), to.clone()), 1))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Could this store write a slice header's length field? See the module docs for the
    /// no-punning rule that answers `false` without consulting alias analysis.
    fn store_cannot_hit_header(&self, ptr: ValueId, ty: &MirType) -> bool {
        if ty.lane_type().is_float() {
            return true;
        }
        if matches!(
            ty,
            MirType::I1 | MirType::I16 | MirType::I32 | MirType::I64 | MirType::Ptr
        ) {
            if let Some(Op::Gep { elem, .. }) = self.defs.op(self.f, ptr) {
                return elem == ty;
            }
        }
        false
    }

    /// Does nothing in the loop write the 8 bytes at `addr`?
    fn stable(&self, addr: ValueId, alias: &AliasInfo) -> bool {
        for b in self.blocks {
            for inst in &self.f.block(*b).insts {
                match &inst.op {
                    Op::Store { ptr, value } => {
                        let ty = self.f.value_type(*value);
                        if self.store_cannot_hit_header(*ptr, ty) {
                            continue;
                        }
                        let bytes = type_bytes(ty).unwrap_or(u32::MAX);
                        if alias.may_alias_sized(addr, 8, *ptr, bytes) {
                            return false;
                        }
                    }
                    other => {
                        if alias.may_clobber(addr, 8, other) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Where the guard can get this check's length, and whether reading it early needs ordering.
    fn len_source(&self, len: ValueId, alias: &AliasInfo) -> Option<(LenSrc, bool)> {
        if self.outside(len) {
            return Some((LenSrc::Value(len), false));
        }
        let Op::Load(addr, MirType::I64) = self.defs.op(self.f, len)? else {
            return None;
        };
        if !self.outside(*addr) || !self.stable(*addr, alias) {
            return None;
        }
        Some((LenSrc::Load(*addr), !alias.is_dereferenceable(*addr, 8)))
    }
}

/// The predicate meaning the same thing with the operands exchanged.
fn swap_cmp(p: CmpOp) -> Option<CmpOp> {
    Some(match p {
        CmpOp::Slt => CmpOp::Sgt,
        CmpOp::Sgt => CmpOp::Slt,
        CmpOp::Ult => CmpOp::Ugt,
        CmpOp::Ugt => CmpOp::Ult,
        _ => return None,
    })
}

fn plan_loop(
    f: &Function,
    l: &NaturalLoop,
    idom: &[u32],
    preds: &[Vec<u32>],
    defs: &Defs,
    alias: &AliasInfo,
) -> Option<Plan> {
    let in_loop: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
    let header = l.header;
    let preheader = l.preheader?;
    if !matches!(&f.block(preheader).term, Terminator::Br { target, .. } if *target == header) {
        return None;
    }

    // The header's exit test: stay on the true arm, leave on the false arm to a block nothing else
    // reaches (so the two copies can share it through fresh parameters).
    let Terminator::CondBr {
        cond: exit_cond,
        then_blk,
        else_blk: exit,
        ..
    } = &f.block(header).term
    else {
        return None;
    };
    let (exit_cond, exit) = (*exit_cond, *exit);
    if !in_loop.contains(&then_blk.0) || in_loop.contains(&exit.0) {
        return None;
    }
    if preds.get(exit.0 as usize).map(Vec::as_slice) != Some(&[header.0][..]) {
        return None;
    }

    // Every other exit leaves to a dead end owned by the exiting block.
    let mut dead_ends: Vec<BlockId> = Vec::new();
    for &(from, to) in &l.exits {
        if from == header {
            if to != exit {
                return None;
            }
            continue;
        }
        if to == exit
            || !is_dead_end(f, to)
            || preds.get(to.0 as usize).map(Vec::as_slice) != Some(&[from.0][..])
        {
            return None;
        }
        dead_ends.push(to);
    }
    if dead_ends.is_empty() {
        return None;
    }

    // A call could rewrite a slice header behind the guard, and a cloned kernel call would run on
    // both paths' census. Neither is worth reasoning about inside a hot loop.
    for b in &l.blocks {
        for inst in &f.block(*b).insts {
            if matches!(
                inst.op,
                Op::Call { .. } | Op::VecKernelCall { .. } | Op::Barrier
            ) {
                return None;
            }
        }
    }

    // The counter: the header parameter the exit test compares, stepping by +1 from a start the
    // preheader passes, against an invariant bound.
    let Some(Op::Cmp(pred, a, b)) = defs.op(f, exit_cond) else {
        return None;
    };
    let iv = l.ivs.iter().find(|iv| iv.value == *a || iv.value == *b)?;
    let (pred, bound) = if iv.value == *a {
        (*pred, *b)
    } else {
        (swap_cmp(*pred)?, *a)
    };
    let unsigned = match pred {
        CmpOp::Slt => false,
        CmpOp::Ult => true,
        _ => return None,
    };
    if iv.step != IvStep::Const(1) || !iv.ty.is_int() {
        return None;
    }
    let start = iv.start?;
    let ctx = Ctx {
        f,
        defs,
        in_loop: &in_loop,
        blocks: &l.blocks,
        iv: iv.value,
        unsigned,
    };
    if !ctx.outside(start) || !ctx.outside(bound) {
        return None;
    }

    // Every block that can leave to a dead end, and the checks among them this guard can prove.
    let mut exiting: Vec<u32> = Vec::new();
    let mut free: Vec<CheckPlan> = Vec::new();
    let mut ordered: Vec<CheckPlan> = Vec::new();
    for blk_id in &l.blocks {
        if *blk_id == header {
            continue;
        }
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &f.block(*blk_id).term
        else {
            continue;
        };
        if !dead_ends.contains(then_blk) && !dead_ends.contains(else_blk) {
            continue;
        }
        exiting.push(blk_id.0);
        if !in_loop.contains(&then_blk.0) || !dead_ends.contains(else_blk) {
            continue;
        }
        let Some(op) = defs.op(f, *cond) else {
            continue;
        };
        // The original of an already-versioned loop: never version it again (see Termination).
        if matches!(op, Op::Cmp(CmpOp::Ugt, ..)) {
            return None;
        }
        let Op::Cmp(CmpOp::Ult, idx, len) = op else {
            continue;
        };
        if f.value_type(*idx) != &MirType::I64 || f.value_type(*len) != &MirType::I64 {
            continue;
        }
        let Some((idx, coeff)) = ctx.affine(*idx, 0) else {
            continue;
        };
        let Some((len, needs_order)) = ctx.len_source(*len, alias) else {
            continue;
        };
        let Some(cmp_at) = defs.at(*cond) else {
            continue;
        };
        let check = CheckPlan {
            block: *blk_id,
            cmp_at,
            idx,
            coeff,
            len,
        };
        if needs_order {
            ordered.push(check);
        } else {
            free.push(check);
        }
    }

    // An ordered check reads a caller's slice header early, which is sound only where iteration 0
    // of the original loop provably reads it too (module docs): it must run on every iteration, and
    // every check that stays behind must be unable to stop iteration 0 before it.
    ordered.retain(|c| l.latches.iter().all(|lt| dominates(c.block.0, lt.0, idom)));
    loop {
        let eliminated: FxHashSet<u32> = free.iter().chain(&ordered).map(|c| c.block.0).collect();
        let kept: Vec<u32> = exiting
            .iter()
            .copied()
            .filter(|b| !eliminated.contains(b))
            .collect();
        let before = ordered.len();
        ordered.retain(|c| kept.iter().all(|&k| dominates(c.block.0, k, idom)));
        if ordered.len() == before {
            break;
        }
    }
    ordered.sort_by_key(|c| (dom_depth(c.block.0, idom), c.block.0));
    if free.is_empty() && ordered.is_empty() {
        return None;
    }

    Some(Plan {
        header,
        preheader,
        exit,
        blocks: l.blocks.clone(),
        dead_ends,
        start,
        bound,
        iv_ty: iv.ty.clone(),
        unsigned,
        free,
        ordered,
    })
}

// ---------------------------------------------------------------------------------------------
// The rewrite
// ---------------------------------------------------------------------------------------------

fn push_inst(out: &mut Vec<Inst>, types: &mut Vec<MirType>, ty: MirType, op: Op) -> ValueId {
    let r = ValueId(types.len() as u32);
    types.push(ty);
    out.push(Inst {
        result: Some(r),
        op,
    });
    r
}

fn fresh_like(types: &mut Vec<MirType>, v: ValueId) -> ValueId {
    let ty = types[v.0 as usize].clone();
    let r = ValueId(types.len() as u32);
    types.push(ty);
    r
}

/// Re-emit an index expression with the induction variable replaced by `iv`.
fn emit_idx(e: &IdxExpr, iv: ValueId, out: &mut Vec<Inst>, types: &mut Vec<MirType>) -> ValueId {
    match e {
        IdxExpr::Iv => iv,
        IdxExpr::Outside(v) => *v,
        IdxExpr::Const(k, ty) => push_inst(out, types, ty.clone(), Op::ConstInt(*k, ty.clone())),
        IdxExpr::Bin(op, a, b, ty) => {
            let a = emit_idx(a, iv, out, types);
            let b = emit_idx(b, iv, out, types);
            push_inst(out, types, ty.clone(), Op::Bin(*op, a, b))
        }
        IdxExpr::Cast(kind, a, ty) => {
            let a = emit_idx(a, iv, out, types);
            push_inst(out, types, ty.clone(), Op::Cast(*kind, a, ty.clone()))
        }
    }
}

/// Emit one check's guard (module docs) and return its `i1`.
fn emit_guard(
    c: &CheckPlan,
    start: ValueId,
    last: ValueId,
    out: &mut Vec<Inst>,
    types: &mut Vec<MirType>,
) -> ValueId {
    let len = match c.len {
        LenSrc::Value(v) => v,
        LenSrc::Load(addr) => push_inst(out, types, MirType::I64, Op::Load(addr, MirType::I64)),
    };
    if c.coeff == 0 {
        let v = emit_idx(&c.idx, start, out, types);
        return push_inst(out, types, MirType::I1, Op::Cmp(CmpOp::Ult, v, len));
    }
    let first = emit_idx(&c.idx, start, out, types);
    let final_ = emit_idx(&c.idx, last, out, types);
    let (lo, hi) = if c.coeff > 0 {
        (first, final_)
    } else {
        (final_, first)
    };
    let no_wrap = push_inst(out, types, MirType::I1, Op::Cmp(CmpOp::Ule, lo, hi));
    let in_range = push_inst(out, types, MirType::I1, Op::Cmp(CmpOp::Ult, hi, len));
    push_inst(
        out,
        types,
        MirType::I1,
        Op::Bin(BinOp::And, no_wrap, in_range),
    )
}

/// Point every successor edge of `t` through `map`.
fn retarget(t: &mut Terminator, map: impl Fn(BlockId) -> BlockId) {
    match t {
        Terminator::Br { target, .. } => *target = map(*target),
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            *then_blk = map(*then_blk);
            *else_blk = map(*else_blk);
        }
        // Switches only form after every loop pass (`switch_form`), but the retarget is total.
        Terminator::Switch { cases, default, .. } => {
            for c in cases.iter_mut() {
                c.target = map(c.target);
            }
            *default = map(*default);
        }
        Terminator::Ret(_) | Terminator::Unreachable => {}
    }
}

/// Append `vals` to the arguments `t` passes to `target`.
fn append_edge_args(t: &mut Terminator, target: BlockId, vals: &[ValueId]) {
    match t {
        Terminator::Br {
            target: tb, args, ..
        } if *tb == target => args.extend_from_slice(vals),
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => {
            if *else_blk == target {
                else_args.extend_from_slice(vals);
            } else if *then_blk == target {
                then_args.extend_from_slice(vals);
            }
        }
        _ => {}
    }
}

fn apply(f: &mut Function, p: &Plan) {
    let in_loop: FxHashSet<u32> = p.blocks.iter().map(|b| b.0).collect();
    let dead: FxHashSet<u32> = p.dead_ends.iter().map(|b| b.0).collect();
    let elim: FxHashSet<u32> = p.free.iter().chain(&p.ordered).map(|c| c.block.0).collect();

    // Values the loop defines, and those of them used after it: they leave through the exit edge.
    let mut defined: FxHashSet<u32> = FxHashSet::default();
    for b in &p.blocks {
        let blk = f.block(*b);
        defined.extend(blk.params.iter().map(|v| v.0));
        defined.extend(blk.insts.iter().filter_map(|i| i.result).map(|r| r.0));
    }
    let mut live_out: Vec<ValueId> = Vec::new();
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for blk in &f.blocks {
        if in_loop.contains(&blk.id.0) || dead.contains(&blk.id.0) {
            continue;
        }
        let mut note = |v: ValueId| {
            if defined.contains(&v.0) && seen.insert(v.0) {
                live_out.push(v);
            }
        };
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut note);
        }
        each_term_use(&blk.term, &mut note);
    }
    live_out.sort_by_key(|v| v.0);

    // New block ids: the fast copy, the dead ends it still needs, then the two preheaders and the
    // guard chain.
    let base = f.blocks.len() as u32;
    let mut next = base;
    let mut bmap: FxHashMap<u32, BlockId> = FxHashMap::default();
    for b in &p.blocks {
        bmap.insert(b.0, BlockId(next));
        next += 1;
    }
    let mut dmap: Vec<(BlockId, BlockId)> = Vec::new();
    for b in &p.blocks {
        if elim.contains(&b.0) {
            continue;
        }
        for s in cfg::successors(&f.block(*b).term) {
            if dead.contains(&s.0) && !dmap.iter().any(|(d, _)| *d == s) {
                dmap.push((s, BlockId(next)));
                next += 1;
            }
        }
    }
    let ph_fast = BlockId(next);
    let ph_slow = BlockId(next + 1);
    let g0 = BlockId(next + 2);
    next += 3;
    let g_ordered: Vec<BlockId> = (0..p.ordered.len() as u32)
        .map(|k| BlockId(next + k))
        .collect();

    // Fresh values for everything the copy defines.
    let mut vmap: FxHashMap<u32, ValueId> = FxHashMap::default();
    let cloned_srcs: Vec<BlockId> = p
        .blocks
        .iter()
        .copied()
        .chain(dmap.iter().map(|(d, _)| *d))
        .collect();
    for b in &cloned_srcs {
        let blk = f.blocks[b.0 as usize].clone();
        for v in blk
            .params
            .iter()
            .copied()
            .chain(blk.insts.iter().filter_map(|i| i.result))
        {
            let nv = fresh_like(&mut f.value_types, v);
            vmap.insert(v.0, nv);
        }
    }
    let m = |v: ValueId| vmap.get(&v.0).copied().unwrap_or(v);
    let blk_map = |b: BlockId| {
        bmap.get(&b.0)
            .copied()
            .or_else(|| dmap.iter().find(|(d, _)| *d == b).map(|(_, n)| *n))
            .unwrap_or(b)
    };

    // The fast copy: the loop with every eliminated check turned into a plain branch to its
    // in-bounds continuation, and the dead ends its remaining checks still need.
    let mut new_blocks: Vec<BasicBlock> = Vec::new();
    for src in &cloned_srcs {
        let blk = &f.blocks[src.0 as usize];
        let params = blk.params.iter().map(|v| m(*v)).collect();
        let insts = blk
            .insts
            .iter()
            .map(|inst| {
                let mut op = inst.op.clone();
                map_op_uses(&mut op, m);
                Inst {
                    result: inst.result.map(m),
                    op,
                }
            })
            .collect();
        let term = match &blk.term {
            Terminator::CondBr {
                then_blk,
                then_args,
                ..
            } if elim.contains(&src.0) => Terminator::Br {
                target: blk_map(*then_blk),
                args: then_args.iter().map(|v| m(*v)).collect(),
            },
            other => {
                let mut t = other.clone();
                retarget(&mut t, blk_map);
                map_term_uses(&mut t, m);
                t
            }
        };
        new_blocks.push(BasicBlock {
            id: blk_map(*src),
            params,
            insts,
            term,
        });
    }

    // Both copies leave through the one exit block; what the loop leaves behind arrives there as
    // fresh parameters, and every use after the loop reads those instead.
    let merged: Vec<ValueId> = live_out
        .iter()
        .map(|v| fresh_like(&mut f.value_types, *v))
        .collect();
    let fast_out: Vec<ValueId> = live_out.iter().map(|v| m(*v)).collect();
    let fast_header = blk_map(p.header);
    if let Some(h) = new_blocks.iter_mut().find(|b| b.id == fast_header) {
        append_edge_args(&mut h.term, p.exit, &fast_out);
    }
    append_edge_args(&mut f.blocks[p.header.0 as usize].term, p.exit, &live_out);
    f.blocks[p.exit.0 as usize]
        .params
        .extend(merged.iter().copied());
    let replace: FxHashMap<u32, ValueId> = live_out
        .iter()
        .zip(&merged)
        .map(|(v, n)| (v.0, *n))
        .collect();
    if !replace.is_empty() {
        let r = |v: ValueId| replace.get(&v.0).copied().unwrap_or(v);
        for blk in f.blocks.iter_mut() {
            if in_loop.contains(&blk.id.0) || dead.contains(&blk.id.0) {
                continue;
            }
            for inst in &mut blk.insts {
                map_op_uses(&mut inst.op, r);
            }
            map_term_uses(&mut blk.term, r);
        }
    }

    // The original loop keeps its checks; mark the ones the guard proved so this pass never versions
    // the original again (`ugt len, idx` is the same predicate).
    for c in p.free.iter().chain(&p.ordered) {
        let (b, i) = c.cmp_at;
        let inst = &mut f.blocks[b as usize].insts[i as usize];
        if let Op::Cmp(CmpOp::Ult, idx, len) = inst.op {
            inst.op = Op::Cmp(CmpOp::Ugt, len, idx);
        }
    }

    // The guard chain.
    let pre = p.preheader.0 as usize;
    let pre_args = match &f.blocks[pre].term {
        Terminator::Br { args, .. } => args.clone(),
        _ => Vec::new(),
    };
    let runs_pred = if p.unsigned { CmpOp::Ult } else { CmpOp::Slt };
    let mut pre_insts = Vec::new();
    let runs = push_inst(
        &mut pre_insts,
        &mut f.value_types,
        MirType::I1,
        Op::Cmp(runs_pred, p.start, p.bound),
    );
    f.blocks[pre].insts.extend(pre_insts);
    f.blocks[pre].term = Terminator::CondBr {
        cond: runs,
        then_blk: g0,
        then_args: Vec::new(),
        else_blk: ph_slow,
        else_args: Vec::new(),
    };

    let mut g0_insts = Vec::new();
    let one = push_inst(
        &mut g0_insts,
        &mut f.value_types,
        p.iv_ty.clone(),
        Op::ConstInt(1, p.iv_ty.clone()),
    );
    let last = push_inst(
        &mut g0_insts,
        &mut f.value_types,
        p.iv_ty.clone(),
        Op::Bin(BinOp::Sub, p.bound, one),
    );
    let mut all = None;
    for c in &p.free {
        let ok = emit_guard(c, p.start, last, &mut g0_insts, &mut f.value_types);
        all = Some(match all {
            None => ok,
            Some(acc) => push_inst(
                &mut g0_insts,
                &mut f.value_types,
                MirType::I1,
                Op::Bin(BinOp::And, acc, ok),
            ),
        });
    }
    let all = match all {
        Some(v) => v,
        None => push_inst(
            &mut g0_insts,
            &mut f.value_types,
            MirType::I1,
            Op::ConstInt(1, MirType::I1),
        ),
    };
    let after_g0 = g_ordered.first().copied().unwrap_or(ph_fast);
    new_blocks.push(BasicBlock {
        id: g0,
        params: Vec::new(),
        insts: g0_insts,
        term: Terminator::CondBr {
            cond: all,
            then_blk: after_g0,
            then_args: Vec::new(),
            else_blk: ph_slow,
            else_args: Vec::new(),
        },
    });
    for (k, c) in p.ordered.iter().enumerate() {
        let mut insts = Vec::new();
        let ok = emit_guard(c, p.start, last, &mut insts, &mut f.value_types);
        let next_blk = g_ordered.get(k + 1).copied().unwrap_or(ph_fast);
        new_blocks.push(BasicBlock {
            id: g_ordered[k],
            params: Vec::new(),
            insts,
            term: Terminator::CondBr {
                cond: ok,
                then_blk: next_blk,
                then_args: Vec::new(),
                else_blk: ph_slow,
                else_args: Vec::new(),
            },
        });
    }
    new_blocks.push(BasicBlock {
        id: ph_fast,
        params: Vec::new(),
        insts: Vec::new(),
        term: Terminator::Br {
            target: fast_header,
            args: pre_args.clone(),
        },
    });
    new_blocks.push(BasicBlock {
        id: ph_slow,
        params: Vec::new(),
        insts: Vec::new(),
        term: Terminator::Br {
            target: p.header,
            args: pre_args,
        },
    });

    // Block ids must equal their index: append in id order.
    new_blocks.sort_by_key(|b| b.id.0);
    debug_assert!(new_blocks
        .iter()
        .enumerate()
        .all(|(k, b)| b.id.0 == base + k as u32));
    f.blocks.extend(new_blocks);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize;
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str) -> (wukong_mir::Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
        (program, interner)
    }

    fn run(src: &str, level: u8) -> (wukong_mir::Program, Interner, Result<i64, String>) {
        let (mut program, mut interner) = lower(src);
        optimize(&mut program, level);
        for f in &program.funcs {
            let errors = wukong_mir::verify::verify_function(f);
            assert!(errors.is_empty(), "-O{level}: {errors:?}");
        }
        let main = interner.intern("main");
        let result = wukong_interp::run(&program, main, &interner);
        (program, interner, result)
    }

    /// Program-wide `(loops, loops still carrying a check, original-loop checks marked as proved)`.
    /// Counted over every function because `-O2` may inline the function the loop was written in.
    fn census(p: &wukong_mir::Program) -> (usize, usize, usize) {
        let (mut loops, mut checked, mut marked) = (0, 0, 0);
        for f in &p.funcs {
            let forest = loop_info::analyze_function(f);
            loops += forest.loops.len();
            checked += forest
                .loops
                .iter()
                .filter(|l| l.exits.iter().any(|&(_, to)| is_dead_end(f, to)))
                .count();
            let defs = Defs::of(f);
            marked += f
                .blocks
                .iter()
                .filter(|b| {
                    matches!(&b.term, Terminator::CondBr { cond, else_blk, .. }
                        if is_dead_end(f, *else_blk)
                            && matches!(defs.op(f, *cond), Some(Op::Cmp(CmpOp::Ugt, ..))))
                })
                .count();
        }
        (loops, checked, marked)
    }

    /// How many loops this pass versioned (or whose check it removed outright after the guard
    /// folded and the original copy died).
    fn versioned(p: &wukong_mir::Program) -> usize {
        let (loops, checked, marked) = census(p);
        (loops - checked).max(marked)
    }

    /// The same program at -O0 and -O2: identical result (value or diagnostic), and at least
    /// `loops` loops versioned at -O2.
    fn assert_versioned(src: &str, loops: usize) {
        let (_, _, o0) = run(src, 0);
        let (p2, _, o2) = run(src, 2);
        assert_eq!(o0, o2, "-O0 vs -O2: {src}");
        let v = versioned(&p2);
        assert!(
            v >= loops,
            "expected {loops} versioned loop(s) at -O2, found {v}: {:?}",
            census(&p2)
        );
    }

    #[test]
    fn a_counted_slice_loop_gets_a_check_free_copy() {
        // In bounds: 1, 2, 3, 4 folded as t*3 + s[i] → ((1*3+2)*3+3)*3+4 = 58.
        let body = "fn total(s: []i32, n: i64) -> i32 { let mut t: i32 = 0; \
                    for i in 0..n { t = t * 3 + s[i]; } return t; }";
        let ok = format!(
            "{body} fn main() -> i32 {{ let a: [i32; 4] = [1, 2, 3, 4]; return total(a, 4); }}"
        );
        assert_eq!(run(&ok, 2).2, Ok(58));
        assert_versioned(&ok, 1);
        // One past the end: the guard fails, the original loop runs and stops at index 4 with the
        // same diagnostic as -O0.
        let oob = format!(
            "{body} fn main() -> i32 {{ let a: [i32; 4] = [1, 2, 3, 4]; return total(a, 5); }}"
        );
        let (_, _, o0) = run(&oob, 0);
        assert_eq!(
            o0,
            Err("index out of bounds: the len is 4 but the index is 4".to_owned())
        );
        assert_eq!(run(&oob, 2).2, o0);
    }

    #[test]
    fn reversed_offset_and_loop_invariant_indices_are_proved() {
        // s[n - 1 - i] (coefficient -1), s[k + i] (an invariant offset) and s[k] (no iv at all).
        // [5, 6, 7, 8]: reverse folds 8, 7, 6, 5 → ((8*10+7)*10+6)*10+5 = 8765; window k = 1, w = 3
        // folds 6, 7, 8 → 678; s[1] three times → 6 * 3 = 18.
        let src = "fn rev(s: []i32) -> i32 { let n: i64 = s.len(); let mut t: i32 = 0; \
                   for i in 0..n { t = t * 10 + s[n - 1 - i]; } return t; } \
                   fn win(s: []i32, k: i64, w: i64) -> i32 { let mut t: i32 = 0; \
                   for i in 0..w { t = t * 10 + s[k + i]; } return t; } \
                   fn inv(s: []i32, k: i64, w: i64) -> i32 { let mut t: i32 = 0; \
                   for i in 0..w { t = t + s[k]; } return t; } \
                   fn main() -> i32 { let a: [i32; 4] = [5, 6, 7, 8]; \
                   return rev(a) - 8765 + win(a, 1, 3) - 678 + inv(a, 1, 3); }";
        assert_eq!(run(src, 0).2, Ok(18));
        assert_versioned(src, 3);
        // Out of range through each form: same stopping index at -O0 and -O2.
        for (call, idx) in [("win(a, 2, 3)", 4), ("inv(a, 4, 2)", 4)] {
            let oob = src.replace("return rev(a)", &format!("return {call} + rev(a)"));
            let (_, _, o0) = run(&oob, 0);
            assert_eq!(
                o0,
                Err(format!(
                    "index out of bounds: the len is 4 but the index is {idx}"
                ))
            );
            assert_eq!(run(&oob, 2).2, o0, "{call}");
        }
    }

    #[test]
    fn a_loop_that_rebinds_its_slice_keeps_every_check() {
        // `s = t` inside the loop writes the header the checks read, so no guard can prove them:
        // nothing may be versioned, and the answer is unchanged. s/t = [1,2,3]/[4,5,6]: i = 0 reads
        // s[0] = 1 then rebinds s to t, so i = 1, 2 read 5 and 6: 1 + 5 + 6 = 12.
        let src = "fn f(a: []i32, b: []i32) -> i32 { let mut s: []i32 = a; let mut t: i32 = 0; \
                   for i in 0..3 { t = t + s[i]; s = b; } return t; } \
                   fn main() -> i32 { let x: [i32; 3] = [1, 2, 3]; let y: [i32; 3] = [4, 5, 6]; \
                   return f(x, y); }";
        let (_, _, o0) = run(src, 0);
        assert_eq!(o0, Ok(12));
        let (p2, _, o2) = run(src, 2);
        assert_eq!(o2, o0);
        assert_eq!(
            versioned(&p2),
            0,
            "a header written in the loop must not be speculated: {:?}",
            census(&p2)
        );
    }

    #[test]
    fn a_parameter_header_is_read_early_only_behind_the_checks_before_it() {
        // `u`'s check is conditional, so its (parameter) header may not be read before the loop;
        // `s`'s check runs every iteration and is proved. The fast copy keeps `u`'s check. With u
        // shorter than the trip count the original diagnostics must be reproduced exactly.
        // In range: s = [1,2,3], u = [10,20,30]: s[i] > 1 for i = 1, 2 → 20 + 30 = 50.
        let body = "fn f(s: []i32, u: []i32, n: i64) -> i32 { let mut t: i32 = 0; \
                    for i in 0..n { if s[i] > 1 { t = t + u[i]; } } return t; }";
        let ok = format!(
            "{body} fn main() -> i32 {{ let a: [i32; 3] = [1, 2, 3]; \
             let b: [i32; 3] = [10, 20, 30]; return f(a, b, 3); }}"
        );
        assert_eq!(run(&ok, 0).2, Ok(50));
        assert_versioned(&ok, 1);
        let short = format!(
            "{body} fn main() -> i32 {{ let a: [i32; 3] = [1, 2, 3]; \
             let b: [i32; 2] = [10, 20]; return f(a, b, 3); }}"
        );
        let (_, _, o0) = run(&short, 0);
        assert_eq!(
            o0,
            Err("index out of bounds: the len is 2 but the index is 2".to_owned())
        );
        assert_eq!(run(&short, 2).2, o0);
    }
}
