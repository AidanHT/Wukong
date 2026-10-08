//! Loop fusion on MIR: two adjacent counted loops over the same iteration space become one.
//!
//! `mir_build` already fuses adjacent same-range `for` loops in one function body
//! (`try_fuse_run`), but only before inlining and only when both loops bind the same variable name.
//! A model written as helper calls — `affine(a, t); combine(t, b, o)` — has its loops in different
//! functions until the inliner splices them, so they reach the optimizer as two loops with an
//! intermediate buffer between them. This pass fuses them structurally, after inlining and loop
//! canonicalization (LLVM's LoopFuse, restricted to what can be proved here):
//!
//! ```text
//!   PH1: br H1(s, ..)              PH1: <PH2's instructions>; br H1(s, .., c2_0..)
//!   H1(i, ..):  i < e ? B1 : PH2   H1(i, .., c2..): i < e ? B1 : X2(..)
//!   B1: ..; br H1(i + 1, ..)    => B1: ..; <B2's body with j := i, c2 := H1's c2>;
//!   PH2: ..; br H2(s, c2_0..)          br H1(i + 1, .., c2_next..)
//!   H2(j, c2..): j < e ? B2 : X2
//!   B2: ..; br H2(j + 1, ..)
//! ```
//!
//! # When it is legal
//!
//! * Both loops are innermost two-block counted loops (`iv < end` tested in the header, unit step,
//!   one latch, one exit, no early return) with no call, `alloca`, vector-kernel call, barrier or
//!   vector value, and they run the **same** iterations: equal start and end (the same value, or
//!   equal constants), the same predicate and induction-variable type.
//! * The first loop exits straight into the second's preheader, which nothing else reaches, which
//!   carries no state, and whose instructions are pure and cannot trap or read memory — they move
//!   ahead of the first loop.
//! * The second loop reads nothing the first loop defines: a value of the first loop is only right
//!   once that loop has finished.
//! * **Memory.** Fusion runs iteration `i` of the second loop before iterations `i + 1..` of the
//!   first, so every pair of accesses where one writes must touch a location at the same iteration
//!   or never. Two accesses to one object must use the identical affine index with a nonzero
//!   constant stride (distance 0 — within a fused iteration the first loop's access still comes
//!   first, as it did); accesses to different objects must be proved disjoint by [`AliasInfo`]. An
//!   unanalyzable address, a loop-invariant address that is written, or two possibly-aliased bases
//!   decline. A store of a pointer declines (it could move a reloaded slice base).
//! * **Profit.** A loop the vectorizer can widen is never fused with one it cannot, and two
//!   widenable loops fuse only when every access has one width — so fusion never costs a loop its
//!   vectorization.
//!
//! The fused loop evaluates every operation of both loops on the same values, in an order whose
//! only change is the interleaving the memory rule just proved unobservable, so results are
//! bit-identical. The intermediate's store and reload then sit in one block, where load forwarding
//! removes the reload; dead-store elimination of the buffer itself is a separate pass.

use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, Function, MirType, Op, Terminator, ValueId};

use crate::alias::{type_bytes, AliasInfo};
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{
    self, AccessKind, AddrForm, Coeff, IvStep, LoopForest, MemAccess, MemBase, NaturalLoop, Stages,
};
use crate::{cfg, each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// The largest loop, in instructions, considered for fusion.
const MAX_LOOP_INSTS: usize = 64;
/// Fusions one invocation may perform before yielding to the rest of the fixpoint.
const MAX_FUSIONS_PER_CALL: usize = 8;

/// Loop fusion (see the module docs).
pub struct LoopFuse;

impl Pass for LoopFuse {
    fn name(&self) -> &'static str {
        "fuse"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let mut changed = false;
        for _ in 0..MAX_FUSIONS_PER_CALL {
            if f.blocks.len() < 6 || !cache.all_reachable(f) {
                break;
            }
            let mut forest = {
                let (idom, preds) = cache.idoms_and_preds(f);
                loop_info::structure_with(f, idom, preds)
            };
            if forest.loops.len() < 2 {
                break;
            }
            let preds = cache.predecessors(f).to_vec();
            let Some(plan) = find_pair(f, &mut forest, &preds) else {
                break;
            };
            apply(f, plan);
            // The second loop's preheader and header are now unreachable.
            cfg::prune_unreachable(f);
            cache.invalidate();
            changed = true;
        }
        changed
    }
}

/// The blocks of one fusion.
struct Plan {
    ph1: usize,
    h1: usize,
    latch1: usize,
    ph2: usize,
    h2: usize,
    latch2: usize,
    /// The first loop's induction variable, which replaces the second's.
    iv1: ValueId,
    /// The second loop's induction variable's position among its header's parameters.
    iv2_index: usize,
}

fn def_sites(f: &Function) -> Vec<Option<(usize, usize)>> {
    let mut d = vec![None; f.value_types.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Some(r) = inst.result {
                if let Some(slot) = d.get_mut(r.0 as usize) {
                    *slot = Some((bi, ii));
                }
            }
        }
    }
    d
}

fn op_of<'a>(f: &'a Function, defs: &[Option<(usize, usize)>], v: ValueId) -> Option<&'a Op> {
    let (b, i) = (*defs.get(v.0 as usize)?)?;
    Some(&f.blocks[b].insts[i].op)
}

/// May this instruction move ahead of the first loop: pure, unable to trap, no memory?
fn movable(op: &Op) -> bool {
    match op {
        Op::ConstInt(..) | Op::ConstFloat(..) | Op::Cmp(..) | Op::Not(_) | Op::Neg(_) => true,
        Op::Select(..) | Op::Gep { .. } | Op::FuncAddr(_) | Op::GlobalAddr(_) => true,
        Op::Bin(b, ..) => !matches!(b, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem),
        Op::Cast(k, ..) => !matches!(k, CastKind::FpToSi | CastKind::FpToUi),
        _ => false,
    }
}

/// An innermost two-block counted loop with nothing fusion cannot reorder or copy:
/// `H: cond_br c, L, X` with `L: …; br H(..)`.
fn shape_ok(f: &Function, l: &NaturalLoop) -> bool {
    if !l.children.is_empty()
        || l.preheader.is_none()
        || l.blocks.len() != 2
        || l.latches.len() != 1
        || l.exits.len() != 1
        || l.exits[0].0 != l.header
        || !l.abnormal_exits.is_empty()
    {
        return false;
    }
    let latch = l.latches[0];
    match &f.block(l.header).term {
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            ..
        } if *then_blk == latch && then_args.is_empty() && *else_blk == l.exits[0].1 => {}
        _ => return false,
    }
    if !matches!(&f.block(latch).term, Terminator::Br { target, .. } if *target == l.header)
        || !f.block(latch).params.is_empty()
    {
        return false;
    }
    let mut n = 0;
    for b in &l.blocks {
        let blk = f.block(*b);
        if blk.params.iter().any(|p| f.value_type(*p).is_vector()) {
            return false;
        }
        for inst in &blk.insts {
            n += 1;
            let bad = match &inst.op {
                Op::Alloca { .. } | Op::Call { .. } | Op::VecKernelCall { .. } | Op::Barrier => {
                    true
                }
                // A pointer stored into memory could move a slice base another access reloads.
                Op::Store { value, .. } => *f.value_type(*value) == MirType::Ptr,
                _ => false,
            };
            if bad || inst.result.is_some_and(|r| f.value_type(r).is_vector()) {
                return false;
            }
        }
    }
    n <= MAX_LOOP_INSTS
}

/// The counted-loop facts fusion compares: the induction variable, its position, its type, the
/// exit predicate, the bound and the exit test's value.
struct Counted {
    iv: ValueId,
    index: usize,
    ty: MirType,
    pred: CmpOp,
    end: ValueId,
    cond: ValueId,
}

fn counted(f: &Function, defs: &[Option<(usize, usize)>], l: &NaturalLoop) -> Option<Counted> {
    let iv = l.primary()?;
    if iv.step != IvStep::Const(1) || !iv.ty.is_int() || iv.ty == MirType::I1 {
        return None;
    }
    let Terminator::CondBr { cond, .. } = &f.block(l.header).term else {
        return None;
    };
    let Some(Op::Cmp(pred, a, end)) = op_of(f, defs, *cond) else {
        return None;
    };
    if *a != iv.value || !matches!(pred, CmpOp::Slt | CmpOp::Ult) {
        return None;
    }
    Some(Counted {
        iv: iv.value,
        index: iv.param_index,
        ty: iv.ty.clone(),
        pred: *pred,
        end: *end,
        cond: *cond,
    })
}

/// The argument a preheader's branch passes at parameter position `k`.
fn preheader_arg(f: &Function, ph: BlockId, k: usize) -> Option<ValueId> {
    match &f.block(ph).term {
        Terminator::Br { args, .. } => args.get(k).copied(),
        _ => None,
    }
}

/// Are `a` and `b` the same value: one SSA value, or two equal integer constants of one type?
fn same_value(f: &Function, defs: &[Option<(usize, usize)>], a: ValueId, b: ValueId) -> bool {
    if a == b {
        return true;
    }
    match (op_of(f, defs, a), op_of(f, defs, b)) {
        (Some(Op::ConstInt(x, tx)), Some(Op::ConstInt(y, ty))) => x == y && tx == ty,
        _ => false,
    }
}

fn values_defined_in(f: &Function, blocks: &[BlockId]) -> FxHashSet<u32> {
    let mut s = FxHashSet::default();
    for b in blocks {
        let blk = f.block(*b);
        s.extend(blk.params.iter().map(|p| p.0));
        s.extend(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
    }
    s
}

/// Can the accesses `a` (first loop) and `b` (second loop), at least one a store, be reordered as
/// fusion reorders them (see the module docs)?
fn pair_ok(alias: &AliasInfo, a: &MemAccess, b: &MemAccess) -> bool {
    match (&a.addr, &b.addr) {
        (AddrForm::Unknown, _) | (_, AddrForm::Unknown) => false,
        (
            AddrForm::Affine {
                base: ba,
                index: ia,
                elem: ea,
            },
            AddrForm::Affine {
                base: bb,
                index: ib,
                elem: eb,
            },
        ) if ba.same_object(*bb) => {
            ia == ib
                && ea == eb
                && a.value_ty == b.value_ty
                && matches!(ia.coeff, Coeff::Const(c) if c != 0)
        }
        _ => {
            let (Some(ba), Some(bb)) = (a.addr.base(), b.addr.base()) else {
                return false;
            };
            if ba.same_object(bb) {
                return false; // one object, but not the identical moving index
            }
            match (ba, bb) {
                (MemBase::Outer(p), MemBase::Outer(q)) => !alias.may_alias(p, q),
                _ => false,
            }
        }
    }
}

fn find_pair(f: &Function, forest: &mut LoopForest, preds: &[Vec<u32>]) -> Option<Plan> {
    let defs = def_sites(f);
    for i1 in 0..forest.loops.len() {
        let (h1, ph2) = {
            let l1 = &forest.loops[i1];
            if !shape_ok(f, l1) {
                continue;
            }
            (l1.header, l1.exits[0].1)
        };
        let Some(i2) = forest.loops.iter().position(|l| l.preheader == Some(ph2)) else {
            continue;
        };
        if i2 == i1 || !shape_ok(f, &forest.loops[i2]) {
            continue;
        }
        // The second preheader is reached only by the first loop's exit, carries no state, and
        // computes nothing that could not run before the first loop.
        let p2 = f.block(ph2);
        if preds[ph2.0 as usize] != [h1.0]
            || !p2.params.is_empty()
            || !p2.insts.iter().all(|i| movable(&i.op))
        {
            continue;
        }
        loop_info::analyze_one(f, forest, i1, Stages::Memory);
        loop_info::analyze_one(f, forest, i2, Stages::Memory);
        if let Some(plan) = plan_pair(f, &defs, &forest.loops[i1], &forest.loops[i2]) {
            return Some(plan);
        }
    }
    None
}

fn plan_pair(
    f: &Function,
    defs: &[Option<(usize, usize)>],
    l1: &NaturalLoop,
    l2: &NaturalLoop,
) -> Option<Plan> {
    let c1 = counted(f, defs, l1)?;
    let c2 = counted(f, defs, l2)?;
    if c1.pred != c2.pred || c1.ty != c2.ty {
        return None;
    }
    let (ph1, ph2) = (l1.preheader?, l2.preheader?);
    let start1 = preheader_arg(f, ph1, c1.index)?;
    let start2 = preheader_arg(f, ph2, c2.index)?;
    let defined1 = values_defined_in(f, &l1.blocks);
    let defined2 = values_defined_in(f, &l2.blocks);
    // The bounds are read before each loop runs, and they are the same bounds.
    if defined1.contains(&c1.end.0)
        || defined2.contains(&c2.end.0)
        || !same_value(f, defs, start1, start2)
        || !same_value(f, defs, c1.end, c2.end)
    {
        return None;
    }
    // The second header computes only its exit test, and nothing else reads that test.
    let h2 = f.block(l2.header);
    if h2.insts.len() != 1 || h2.insts[0].result != Some(c2.cond) {
        return None;
    }
    let mut test_read_elsewhere = false;
    for (bi, b) in f.blocks.iter().enumerate() {
        if bi == l2.header.0 as usize {
            continue;
        }
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| test_read_elsewhere |= v == c2.cond);
        }
        each_term_use(&b.term, &mut |v| test_read_elsewhere |= v == c2.cond);
    }
    if test_read_elsewhere {
        return None;
    }
    // Nothing the first loop defines is read before it finishes in the original: not by the
    // second preheader, not by the second loop.
    let mut reads_first = false;
    for b in std::iter::once(&ph2).chain(&l2.blocks) {
        let blk = f.block(*b);
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |v| reads_first |= defined1.contains(&v.0));
        }
        each_term_use(&blk.term, &mut |v| reads_first |= defined1.contains(&v.0));
    }
    if reads_first {
        return None;
    }
    // Memory.
    let alias = AliasInfo::analyze(f);
    for a in &l1.accesses {
        for b in &l2.accesses {
            if a.kind == AccessKind::Load && b.kind == AccessKind::Load {
                continue;
            }
            if !pair_ok(&alias, a, b) {
                return None;
            }
        }
    }
    // Profit: never trade a widenable loop for a fused one the vectorizer would refuse.
    let (w1, w2) = (l1.is_vectorizable_shape(), l2.is_vectorizable_shape());
    if w1 != w2 {
        return None;
    }
    if w1 {
        let mut widths = l1
            .accesses
            .iter()
            .chain(&l2.accesses)
            .map(|a| type_bytes(&a.value_ty));
        let first = widths.next().flatten();
        if first.is_none() || !widths.all(|w| w == first) {
            return None;
        }
    }
    Some(Plan {
        ph1: ph1.0 as usize,
        h1: l1.header.0 as usize,
        latch1: l1.latches[0].0 as usize,
        ph2: ph2.0 as usize,
        h2: l2.header.0 as usize,
        latch2: l2.latches[0].0 as usize,
        iv1: c1.iv,
        iv2_index: c2.index,
    })
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

fn apply(f: &mut Function, p: Plan) {
    // The second loop's carried values other than its induction variable become new parameters of
    // the first header; its induction variable becomes the first's.
    let h2_params = f.blocks[p.h2].params.clone();
    let mut map: FxHashMap<u32, ValueId> = FxHashMap::default();
    let mut extra = Vec::new();
    for (k, q) in h2_params.iter().enumerate() {
        if k == p.iv2_index {
            map.insert(q.0, p.iv1);
        } else {
            let ty = f.value_type(*q).clone();
            let nq = fresh(f, ty);
            map.insert(q.0, nq);
            extra.push(nq);
        }
    }
    f.blocks[p.h1].params.extend(extra);
    let skip_iv = |args: &[ValueId]| -> Vec<ValueId> {
        args.iter()
            .enumerate()
            .filter(|(k, _)| *k != p.iv2_index)
            .map(|(_, v)| *v)
            .collect()
    };

    // The second preheader's work moves ahead of the first loop, and its initial values enter the
    // fused header (the induction variable's start is the first loop's, proved equal).
    let Terminator::Br { args: init2, .. } = f.blocks[p.ph2].term.clone() else {
        unreachable!("a preheader ends in an unconditional branch");
    };
    let hoisted = std::mem::take(&mut f.blocks[p.ph2].insts);
    f.blocks[p.ph1].insts.extend(hoisted);
    match &mut f.blocks[p.ph1].term {
        Terminator::Br { args, .. } => args.extend(skip_iv(init2.as_slice())),
        _ => unreachable!("a preheader ends in an unconditional branch"),
    }

    // The fused loop leaves where the second one did, with the second one's exit arguments.
    let Terminator::CondBr {
        else_blk: x2,
        else_args: x2_args,
        ..
    } = f.blocks[p.h2].term.clone()
    else {
        unreachable!("a counted loop's header ends in its exit test");
    };
    match &mut f.blocks[p.h1].term {
        Terminator::CondBr {
            else_blk,
            else_args,
            ..
        } => {
            *else_blk = x2;
            *else_args = x2_args;
        }
        _ => unreachable!("a counted loop's header ends in its exit test"),
    }

    // The second body runs straight after the first, in the first's block, and the fused latch
    // passes both loops' next values back: the fused loop stays a two-block counted loop, so it can
    // fuse again with the loop after it.
    let mut back = match &f.blocks[p.latch1].term {
        Terminator::Br { args, .. } => args.clone(),
        _ => unreachable!("a latch ends in an unconditional branch"),
    };
    let next2 = match &f.blocks[p.latch2].term {
        Terminator::Br { args, .. } => skip_iv(args.as_slice()),
        _ => unreachable!("a latch ends in an unconditional branch"),
    };
    back.extend(next2);
    let body2 = std::mem::take(&mut f.blocks[p.latch2].insts);
    f.blocks[p.latch2].term = Terminator::Unreachable;
    f.blocks[p.latch1].insts.extend(body2);
    f.blocks[p.latch1].term = Terminator::Br {
        target: BlockId(p.h1 as u32),
        args: back,
    };

    // Every read of the second header's parameters now reads the fused header's.
    let rename = |v: ValueId| map.get(&v.0).copied().unwrap_or(v);
    for b in &mut f.blocks {
        for inst in &mut b.insts {
            map_op_uses(&mut inst.op, rename);
        }
        map_term_uses(&mut b.term, rename);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dce, SimplifyCfg};
    use wukong_mir::{Builder, Program};
    use wukong_span::{Interner, SourceId};

    /// `f(n)`: three loops over `0..n` on two 8-element stack buffers — `t[i] = 3i`, then
    /// `o[j] = t[j] + 1`, then `acc += o[k]` — returning `acc`. All three fuse into one.
    fn three_loops(it: &mut Interner, shift: i128) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let t = b.alloca(MirType::Array(Box::new(MirType::I32), 9));
        let o = b.alloca(MirType::Array(Box::new(MirType::I32), 9));
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        // Make sure every slot has a defined value even where a loop does not write.
        for buf in [t, o] {
            for k in 0..9 {
                let idx = b.build(MirType::I32, Op::ConstInt(k, MirType::I32));
                let p = b.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: buf,
                        index: idx,
                        elem: MirType::I32,
                    },
                );
                let v = b.build(MirType::I32, Op::ConstInt(100 + k, MirType::I32));
                b.build_void(Op::Store { ptr: p, value: v });
            }
        }
        let h1 = b.new_block();
        let i = b.block_param(h1, MirType::I32);
        let l1 = b.new_block();
        let p2 = b.new_block();
        let h2 = b.new_block();
        let j = b.block_param(h2, MirType::I32);
        let l2 = b.new_block();
        let p3 = b.new_block();
        let h3 = b.new_block();
        let k = b.block_param(h3, MirType::I32);
        let acc = b.block_param(h3, MirType::I32);
        let l3 = b.new_block();
        let x = b.new_block();
        let r = b.block_param(x, MirType::I32);
        b.br(h1, vec![zero]);

        b.switch_to(h1);
        let c1 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c1, l1, vec![], p2, vec![]);
        b.switch_to(l1);
        let pt = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: t,
                index: i,
                elem: MirType::I32,
            },
        );
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let v = b.build(MirType::I32, Op::Bin(BinOp::Mul, i, three));
        b.build_void(Op::Store { ptr: pt, value: v });
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let i2 = b.build(MirType::I32, Op::Bin(BinOp::Add, i, one));
        b.br(h1, vec![i2]);

        b.switch_to(p2);
        let zero2 = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.br(h2, vec![zero2]);
        b.switch_to(h2);
        let c2 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
        b.cond_br(c2, l2, vec![], p3, vec![]);
        b.switch_to(l2);
        // Reads `t` at `j + shift`: 0 fuses, 1 must not (it reads what the first loop writes on a
        // later iteration).
        let sh = b.build(MirType::I32, Op::ConstInt(shift, MirType::I32));
        let js = b.build(MirType::I32, Op::Bin(BinOp::Add, j, sh));
        let qt = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: t,
                index: js,
                elem: MirType::I32,
            },
        );
        let w = b.build(MirType::I32, Op::Load(qt, MirType::I32));
        let one2 = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let w1 = b.build(MirType::I32, Op::Bin(BinOp::Add, w, one2));
        let qo = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: o,
                index: j,
                elem: MirType::I32,
            },
        );
        b.build_void(Op::Store { ptr: qo, value: w1 });
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one2));
        b.br(h2, vec![j2]);

        b.switch_to(p3);
        let zero3 = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.br(h3, vec![zero3, zero3]);
        b.switch_to(h3);
        let c3 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, k, n));
        b.cond_br(c3, l3, vec![], x, vec![acc]);
        b.switch_to(l3);
        let ro = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: o,
                index: k,
                elem: MirType::I32,
            },
        );
        let u = b.build(MirType::I32, Op::Load(ro, MirType::I32));
        let acc2 = b.build(MirType::I32, Op::Bin(BinOp::Add, acc, u));
        let one3 = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let k2 = b.build(MirType::I32, Op::Bin(BinOp::Add, k, one3));
        b.br(h3, vec![k2, acc2]);
        b.switch_to(x);
        b.ret(Some(r));
        b.finish()
    }

    fn run_n(f: &Function, it: &mut Interner, ns: &[i128]) -> Vec<u8> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &n in ns {
            let a = m.build(MirType::I32, Op::ConstInt(n, MirType::I32));
            let r = m.build(
                MirType::I32,
                Op::Call {
                    func: f.name,
                    args: vec![a],
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

    fn loop_count(f: &Function) -> usize {
        crate::loop_info::analyze_function(f).loops.len()
    }

    #[test]
    fn three_adjacent_loops_fuse_into_one() {
        let mut it = Interner::new();
        let mut f = three_loops(&mut it, 0);
        // acc = Σ (3i + 1) over 0..n: n = 0 -> 0, 1 -> 1, 4 -> 22, 8 -> 92.
        let ns = [0, 1, 4, 8];
        let before = run_n(&f, &mut it, &ns);
        assert_eq!(before, b"0\n1\n22\n92\n".to_vec());
        assert_eq!(loop_count(&f), 3);
        assert!(LoopFuse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(
            loop_count(&f),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run_n(&f, &mut it, &ns), before);
        let mut cache = CfgAnalyses::default();
        SimplifyCfg.run_function(&mut f, &mut cache);
        Dce.run_function(&mut f, &mut cache);
        assert_eq!(run_n(&f, &mut it, &ns), before);
    }

    /// The second loop reads `t[j + 1]`, which the first loop writes one iteration *later*: fusing
    /// would read it too early, so the first two loops stay apart (the last two still fuse).
    #[test]
    fn a_read_ahead_of_the_writer_blocks_fusion() {
        let mut it = Interner::new();
        let mut f = three_loops(&mut it, 1);
        // o[j] = t[j + 1] + 1 = 3(j + 1) + 1 for j + 1 < n, and t[n] = 100 + n + ... untouched
        // slot: o[n - 1] = (100 + n) + 1. acc = Σ_{j < n-1} (3j + 4) + 101 + n.
        let ns = [0, 1, 4, 8];
        let before = run_n(&f, &mut it, &ns);
        // n = 0: 0. n = 1: 102. n = 4: (4 + 7 + 10) + 105 = 126. n = 8: (4+7+..+22) + 109 = 200.
        assert_eq!(before, b"0\n102\n126\n200\n".to_vec());
        LoopFuse.run_function(&mut f, &mut CfgAnalyses::default());
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(
            loop_count(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run_n(&f, &mut it, &ns), before);
    }

    /// From source, after inlining: a helper chain over local buffers runs as one loop at `-O2`
    /// and every level prints the same lanes.
    #[test]
    fn a_helper_chain_fuses_after_inlining() {
        let src = concat!(
            "fn affine(a: [i32; 16], mut t: [i32; 16]) { for i in 0..16 { t[i] = a[i] * 3 + 1; } }\n",
            "fn combine(t: [i32; 16], b: [i32; 16], mut o: [i32; 16]) { ",
            "for i in 0..16 { o[i] = t[i] + b[i] * 2; } }\n",
            "fn main() -> i32 { let mut a: [i32; 16] = [0; 16]; let mut b: [i32; 16] = [0; 16]; ",
            "let mut t: [i32; 16] = [0; 16]; let mut o: [i32; 16] = [0; 16]; ",
            "for i in 0..16 { a[i] = i; b[i] = 16 - i; } ",
            "affine(a, t); combine(t, b, o); ",
            "for i in 0..16 { print(o[i]); } return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        // o[i] = (3i + 1) + 2(16 - i) = i + 33.
        let expected: String = (0..16).map(|i| format!("{}\n", i + 33)).collect();
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
    }

    /// `movable` admits exactly what may run ahead of the first loop.
    #[test]
    fn only_pure_non_trapping_work_moves_ahead() {
        let v = ValueId(0);
        let pure = [
            Op::ConstInt(1, MirType::I32),
            Op::Bin(BinOp::Add, v, v),
            Op::Gep {
                ptr: v,
                index: v,
                elem: MirType::I32,
            },
        ];
        assert!(pure.iter().all(movable));
        let effectful = [
            Op::Bin(BinOp::SDiv, v, v),
            Op::Load(v, MirType::I32),
            Op::Store { ptr: v, value: v },
            Op::Cast(CastKind::FpToSi, v, MirType::I32),
        ];
        assert!(!effectful.iter().any(movable));
    }
}
