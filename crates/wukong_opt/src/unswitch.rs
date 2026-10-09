//! Loop unswitching and induction-variable range splitting: take a per-iteration branch out of a
//! loop by running two copies of the loop, each with the branch decided.
//!
//! Both transforms clone an innermost loop and then *assume* a condition inside each copy —
//! replace every use of it in that copy by the constant it has there (folding the `and`/`or`/`not`
//! that exposes), after which `simplify-cfg` deletes the branch. A loop body that was a diamond with
//! one guarded store becomes one copy with the store unconditional and one without it: straight-line
//! bodies the vectorizer accepts, where it refuses the diamond (`vectorize`'s "the two arms do not
//! store the same number of times").
//!
//! # Induction-variable range splitting (GCC `-fsplit-loops`)
//!
//! A counted loop `for iv in start..end` (unit step, `iv < end` tested in the header, signed or
//! unsigned) whose body branches on `iv ⋈ inv` — `<`, `<=`, `>` or `>=` against a loop-invariant
//! `inv`, with the exit test's signedness — has a guard that is monotone along the iterations: it
//! holds on a prefix of them and fails on the rest, or the reverse. The preheader computes the first
//! iteration `s` where it changes, clamped into `[start, end]`:
//!
//! ```text
//!   iv <  inv, iv >= inv:  s = max(start, min(inv, end))
//!   iv <= inv, iv >  inv:  s = inv < end ? max(start, inv + 1) : end     // inv + 1 cannot wrap
//! ```
//!
//! and the loop runs as two loops back to back: a copy over `[start, s)` with the guard assumed to
//! be its prefix value, exiting into the original loop's header with its whole state, and the
//! original over `[s, end)` with the guard assumed to be the other value. Both halves visit the
//! iterations in the original increasing order and compute every value exactly as before, so the
//! result is bit-identical; the split point only decides which copy runs a given iteration, and
//! every iteration a copy runs is one where its assumption is true. The causal mask
//! `if j > i { s[i*n + j] = -inf }` becomes a loop that never stores and a loop that always does.
//!
//! The loop may also have side exits besides the header's test -- a bounds check's trap on a
//! runtime-sized slice, a `break`, an early `return` -- since each copy reaches such an exit at the
//! iteration the unsplit loop does, with the same state. The one hazard is a block after the loop
//! that reads a loop value directly: leaving from the copy bypasses the original's definition. A
//! read inside an exit block entered only from the loop (the trap block reading the failing index)
//! is rerouted through a new block parameter; any other read blocks the split.
//!
//! # Unswitching (LLVM SimpleLoopUnswitch, non-trivial)
//!
//! A branch inside the loop on an `i1` defined outside it — after LICM, any condition that does not
//! change across iterations — is decided once: the preheader branches on it into the original loop
//! (assuming `true`) or into a copy (assuming `false`). Both copies leave through the original exit
//! edges, so this is restricted to loops whose values reach the code after them only through exit
//! edge arguments.
//!
//! # Limits
//!
//! Innermost loops with a preheader and no `alloca`, call, vector-kernel call or barrier (a call
//! would be duplicated, and with it any dispatched kernel), at most [`MAX_LOOP_INSTS`]
//! instructions, in a function under [`MAX_FUNCTION_INSTS`]; a loop fed by an already-vectorized
//! loop (its scalar epilogue) is left alone. Each rewrite removes its condition from both copies, so
//! every loop is rewritten at most once per guard it holds, and the function cap bounds the total.
//! At `-O2`, after LICM (which hoists invariant conditions and bounds into the preheader) and
//! before the vectorizer.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId,
};

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{self, IvStep, LoopForest, NaturalLoop, Stages};
use crate::{each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// The largest loop, in instructions, worth duplicating.
pub(crate) const MAX_LOOP_INSTS: usize = 128;
/// No rewrite may grow a function past this many instructions.
pub(crate) const MAX_FUNCTION_INSTS: usize = 4096;
/// Rewrites one invocation may perform before yielding to the rest of the fixpoint.
const MAX_REWRITES_PER_CALL: usize = 16;

/// Loop unswitching and induction-variable range splitting (see the module docs).
pub struct LoopUnswitch;

impl Pass for LoopUnswitch {
    fn name(&self) -> &'static str {
        "unswitch"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let mut changed = false;
        for _ in 0..MAX_REWRITES_PER_CALL {
            if f.blocks.len() < 3
                || f.blocks
                    .iter()
                    .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
                    .count()
                    < 2
                || !cache.all_reachable(f)
            {
                break;
            }
            let forest = {
                let (idom, preds) = cache.idoms_and_preds(f);
                loop_info::structure_with(f, idom, preds)
            };
            let Some(plan) = pick(f, forest) else {
                break;
            };
            match plan {
                Plan::Split(p) => apply_split(f, p),
                Plan::Unswitch(p) => apply_unswitch(f, p),
            }
            cache.invalidate();
            changed = true;
        }
        changed
    }
}

/// A value the preheader can read: one defined outside the loop, or an integer constant that the
/// loop happens to materialize itself (rematerialized in the preheader).
#[derive(Clone, Copy)]
enum Avail {
    Outside(ValueId),
    Const(i128),
}

/// The guard's predicate, with the induction variable on the left.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GuardKind {
    Lt,
    Le,
    Gt,
    Ge,
}

struct SplitPlan {
    blocks: Vec<BlockId>,
    header: BlockId,
    preheader: BlockId,
    /// The induction variable (a header parameter) and its parameter position.
    iv: ValueId,
    iv_index: usize,
    signed: bool,
    end: Avail,
    guard: ValueId,
    kind: GuardKind,
    inv: Avail,
    /// Loop values the side-exit blocks read directly, to be passed as edge arguments instead
    /// (see [`exit_routes`]).
    routes: Vec<(u32, Vec<ValueId>)>,
}

struct UnswitchPlan {
    blocks: Vec<BlockId>,
    header: BlockId,
    preheader: BlockId,
    cond: ValueId,
}

enum Plan {
    Split(SplitPlan),
    Unswitch(UnswitchPlan),
}

fn func_insts(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

/// `ValueId -> (block, instruction index)` of its defining instruction.
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

/// Does this loop qualify for duplication at all (see the module's limits)?
fn clonable(f: &Function, forest: &LoopForest, l: &NaturalLoop) -> bool {
    if l.preheader.is_none() || !l.children.is_empty() {
        return false;
    }
    let mut n = 0;
    for b in &l.blocks {
        let blk = f.block(*b);
        if blk.params.iter().any(|p| f.value_type(*p).is_vector())
            || matches!(blk.term, Terminator::Switch { .. })
        {
            return false;
        }
        for inst in &blk.insts {
            n += 1;
            if matches!(
                inst.op,
                Op::Alloca { .. } | Op::Call { .. } | Op::VecKernelCall { .. } | Op::Barrier
            ) || inst.result.is_some_and(|r| f.value_type(r).is_vector())
            {
                return false;
            }
        }
    }
    if n > MAX_LOOP_INSTS || func_insts(f) + n > MAX_FUNCTION_INSTS {
        return false;
    }
    // The scalar epilogue of a loop the vectorizer already widened is entered from that loop's
    // exit; it is the remainder, not a loop worth duplicating.
    let ph = l.preheader.expect("checked above");
    !forest.loops.iter().any(|v| {
        v.id != l.id
            && v.exits.iter().any(|&(_, to)| to == ph)
            && v.blocks.iter().any(|b| {
                let blk = f.block(*b);
                blk.params.iter().any(|p| f.value_type(*p).is_vector())
                    || blk
                        .insts
                        .iter()
                        .any(|i| i.result.is_some_and(|r| f.value_type(r).is_vector()))
            })
    })
}

/// The conditions a branch on `cond` decides: `cond` itself and, when it is an `i1` `and`/`or`,
/// its two operands (one level — the shape branch folding produces).
fn branch_conditions(f: &Function, defs: &[Option<(usize, usize)>], cond: ValueId) -> Vec<ValueId> {
    let mut out = vec![cond];
    if let Some(Op::Bin(BinOp::And | BinOp::Or, a, b)) = op_of(f, defs, cond) {
        if *f.value_type(cond) == MirType::I1 {
            out.push(*a);
            out.push(*b);
        }
    }
    out
}

/// The first rewrite any innermost loop admits, innermost first.
fn pick(f: &Function, mut forest: LoopForest) -> Option<Plan> {
    if forest.is_empty() {
        return None;
    }
    let defs = def_sites(f);
    for id in forest.innermost_first() {
        let idx = id.0 as usize;
        {
            let l = forest.get(id);
            if !clonable(f, &forest, l) {
                continue;
            }
            // Something to decide: a two-way branch in the body.
            let any_guard = l.blocks.iter().any(|b| {
                *b != l.header
                    && matches!(&f.block(*b).term,
                        Terminator::CondBr { then_blk, else_blk, .. } if then_blk != else_blk)
            });
            if !any_guard {
                continue;
            }
        }
        loop_info::analyze_one(f, &mut forest, idx, Stages::Ivs);
        let l = forest.get(id);
        let in_loop: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
        let mut defined: FxHashSet<u32> = FxHashSet::default();
        for b in &l.blocks {
            let blk = f.block(*b);
            defined.extend(blk.params.iter().map(|p| p.0));
            defined.extend(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
        }
        if let Some(p) = plan_split(f, l, &defs, &in_loop, &defined) {
            return Some(Plan::Split(p));
        }
        if let Some(p) = plan_unswitch(f, l, &defs, &in_loop, &defined) {
            return Some(Plan::Unswitch(p));
        }
    }
    None
}

/// `v` as something the preheader can read (see [`Avail`]).
fn avail(
    f: &Function,
    defs: &[Option<(usize, usize)>],
    defined: &FxHashSet<u32>,
    v: ValueId,
) -> Option<Avail> {
    if !defined.contains(&v.0) {
        return Some(Avail::Outside(v));
    }
    match op_of(f, defs, v)? {
        Op::ConstInt(n, _) => Some(Avail::Const(*n)),
        _ => None,
    }
}

/// `a ⋈ b` with the operands exchanged.
fn swapped(p: CmpOp) -> Option<CmpOp> {
    use CmpOp::*;
    Some(match p {
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

fn plan_split(
    f: &Function,
    l: &NaturalLoop,
    defs: &[Option<(usize, usize)>],
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> Option<SplitPlan> {
    let iv = l.primary()?;
    if iv.step != IvStep::Const(1) || !iv.ty.is_int() || iv.ty == MirType::I1 {
        return None;
    }
    let preheader = l.preheader?;
    // The header's test `iv < end` exits on its false arm (loop-canon's polarity). Any other exit
    // is a side exit -- a bounds check's trap, a `break`, an early `return` -- taken by the copy
    // or the original at exactly the iteration the unsplit loop takes it, with the same state.
    let mut header_exits = l.exits.iter().filter(|e| e.0 == l.header);
    let (Some(&(_, header_exit)), None) = (header_exits.next(), header_exits.next()) else {
        return None;
    };
    let Terminator::CondBr {
        cond,
        then_blk,
        else_blk,
        ..
    } = &f.block(l.header).term
    else {
        return None;
    };
    if *else_blk != header_exit || !in_loop.contains(&then_blk.0) {
        return None;
    }
    // A side exit from the copy bypasses the original loop's definitions, so code after the loop
    // may then read a loop value only through exit-edge arguments (the original header always
    // runs last before the header exit, so that exit alone needs no such care). A bounds check's
    // trap block reads the failing index directly; that read becomes an edge argument.
    let routes = if l.exits.len() > 1 {
        exit_routes(f, in_loop, defined)?
    } else {
        Vec::new()
    };
    let Some(Op::Cmp(exit_pred, lhs, end)) = op_of(f, defs, *cond) else {
        return None;
    };
    if *lhs != iv.value {
        return None;
    }
    let signed = match exit_pred {
        CmpOp::Slt => true,
        CmpOp::Ult => false,
        _ => return None,
    };
    let end = avail(f, defs, defined, *end)?;
    for b in &l.blocks {
        if *b == l.header {
            continue;
        }
        let Terminator::CondBr {
            cond: g,
            then_blk,
            else_blk,
            ..
        } = &f.block(*b).term
        else {
            continue;
        };
        if then_blk == else_blk {
            continue;
        }
        // The guard is assumed only in body blocks (see `apply_split`), so an `and`/`or` the header
        // computes from it would never fold, and the branch would be split again.
        let g_in_header = defs
            .get(g.0 as usize)
            .copied()
            .flatten()
            .is_some_and(|(bi, _)| bi == l.header.0 as usize);
        for v in branch_conditions(f, defs, *g) {
            // The guard varies with the induction variable, so the loop computes it.
            if !defined.contains(&v.0) || (v != *g && g_in_header) {
                continue;
            }
            let Some(Op::Cmp(p, x, y)) = op_of(f, defs, v) else {
                continue;
            };
            let (p, other) = if *x == iv.value && *y != iv.value {
                (*p, *y)
            } else if *y == iv.value && *x != iv.value {
                let Some(sp) = swapped(*p) else {
                    continue;
                };
                (sp, *x)
            } else {
                continue;
            };
            let kind = match (signed, p) {
                (true, CmpOp::Slt) | (false, CmpOp::Ult) => GuardKind::Lt,
                (true, CmpOp::Sle) | (false, CmpOp::Ule) => GuardKind::Le,
                (true, CmpOp::Sgt) | (false, CmpOp::Ugt) => GuardKind::Gt,
                (true, CmpOp::Sge) | (false, CmpOp::Uge) => GuardKind::Ge,
                _ => continue,
            };
            let Some(inv) = avail(f, defs, defined, other) else {
                continue;
            };
            return Some(SplitPlan {
                blocks: l.blocks.clone(),
                header: l.header,
                preheader,
                iv: iv.value,
                iv_index: iv.param_index,
                signed,
                end,
                guard: v,
                kind,
                inv,
                routes,
            });
        }
    }
    None
}

fn plan_unswitch(
    f: &Function,
    l: &NaturalLoop,
    defs: &[Option<(usize, usize)>],
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> Option<UnswitchPlan> {
    let preheader = l.preheader?;
    let mut cond = None;
    'search: for b in &l.blocks {
        if *b == l.header {
            continue;
        }
        let Terminator::CondBr {
            cond: c,
            then_blk,
            else_blk,
            ..
        } = &f.block(*b).term
        else {
            continue;
        };
        if then_blk == else_blk {
            continue;
        }
        for v in branch_conditions(f, defs, *c) {
            if defined.contains(&v.0)
                || *f.value_type(v) != MirType::I1
                || matches!(op_of(f, defs, v), Some(Op::ConstInt(..)))
            {
                continue;
            }
            cond = Some(v);
            break 'search;
        }
    }
    let cond = cond?;
    // Both copies leave through the original exit edges, so nothing after the loop may read a
    // loop value except through those edges' arguments.
    if reads_loop_values_after(f, in_loop, defined) {
        return None;
    }
    Some(UnswitchPlan {
        blocks: l.blocks.clone(),
        header: l.header,
        preheader,
        cond,
    })
}

/// Where code outside the loop reads loop values directly, other than through exit-edge arguments:
/// `None` unless every such read sits in an exit block whose single incoming edge leaves the loop
/// (a bounds check's trap block reading the failing index, say), in which case the values each
/// such block reads, in first-use order. Such a read can be rerouted through a new parameter of
/// that block ([`route_exit_values`]): the value dominates the block, so it dominates the end of its
/// only predecessor.
fn exit_routes(
    f: &Function,
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> Option<Vec<(u32, Vec<ValueId>)>> {
    let n = f.blocks.len();
    let mut edges_in = vec![0usize; n];
    let mut from_loop = vec![false; n];
    for (bi, b) in f.blocks.iter().enumerate() {
        b.term.for_each_edge(|t, _| {
            if let Some(c) = edges_in.get_mut(t.0 as usize) {
                *c += 1;
            }
            if in_loop.contains(&(bi as u32)) {
                if let Some(fl) = from_loop.get_mut(t.0 as usize) {
                    *fl = true;
                }
            }
        });
    }
    let mut routes = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        if in_loop.contains(&(bi as u32)) {
            continue;
        }
        let mut used: Vec<ValueId> = Vec::new();
        let mut note = |v: ValueId| {
            if defined.contains(&v.0) && !used.contains(&v) {
                used.push(v);
            }
        };
        for inst in &b.insts {
            each_op_use(&inst.op, &mut note);
        }
        each_term_use(&b.term, &mut note);
        if used.is_empty() {
            continue;
        }
        if bi == 0 || edges_in[bi] != 1 || !from_loop[bi] {
            return None;
        }
        routes.push((bi as u32, used));
    }
    Some(routes)
}

/// Give each exit block of `routes` one new parameter per loop value it reads, passed by its only
/// incoming edge, and read the parameter instead.
fn route_exit_values(f: &mut Function, routes: &[(u32, Vec<ValueId>)]) {
    for (blk, vals) in routes {
        let bi = *blk as usize;
        let mut subst: FxHashMap<u32, ValueId> = FxHashMap::default();
        for v in vals {
            let ty = f.value_type(*v).clone();
            let p = fresh(f, ty);
            f.blocks[bi].params.push(p);
            subst.insert(v.0, p);
        }
        let sub = |u: ValueId| subst.get(&u.0).copied().unwrap_or(u);
        for inst in &mut f.blocks[bi].insts {
            map_op_uses(&mut inst.op, sub);
        }
        map_term_uses(&mut f.blocks[bi].term, sub);
        for b in &mut f.blocks {
            b.term.for_each_edge_mut(|t, args| {
                if t.0 == *blk {
                    args.extend(vals.iter().copied());
                }
            });
        }
    }
}

/// Does any block outside the loop read a value the loop defines, other than through the
/// arguments of an exit edge?
fn reads_loop_values_after(
    f: &Function,
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> bool {
    f.blocks.iter().enumerate().any(|(bi, b)| {
        if in_loop.contains(&(bi as u32)) {
            return false;
        }
        let mut leaks = false;
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| leaks |= defined.contains(&v.0));
        }
        each_term_use(&b.term, &mut |v| leaks |= defined.contains(&v.0));
        leaks
    })
}

/// Append a copy of `blocks` to `f`: fresh block ids and fresh values for everything the blocks
/// define; edges between them stay inside the copy, edges out of them keep their targets. Returns
/// the copy's block indices (in `blocks` order), the block map and the value map.
fn clone_loop(
    f: &mut Function,
    blocks: &[BlockId],
) -> (Vec<usize>, FxHashMap<u32, usize>, FxHashMap<u32, ValueId>) {
    let base = f.blocks.len();
    let bmap: FxHashMap<u32, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.0, base + i))
        .collect();
    let mut vmap: FxHashMap<u32, ValueId> = FxHashMap::default();
    for b in blocks {
        let defs: Vec<ValueId> = {
            let blk = f.block(*b);
            blk.params
                .iter()
                .copied()
                .chain(blk.insts.iter().filter_map(|i| i.result))
                .collect()
        };
        for v in defs {
            let ty = f.value_type(v).clone();
            let nv = fresh(f, ty);
            vmap.insert(v.0, nv);
        }
    }
    let remap = |v: ValueId| vmap.get(&v.0).copied().unwrap_or(v);
    let mut copies = Vec::with_capacity(blocks.len());
    for (i, b) in blocks.iter().enumerate() {
        let src = f.block(*b);
        let params = src.params.iter().map(|p| remap(*p)).collect();
        let insts = src
            .insts
            .iter()
            .map(|inst| {
                let mut op = inst.op.clone();
                map_op_uses(&mut op, remap);
                Inst {
                    result: inst.result.map(remap),
                    op,
                }
            })
            .collect();
        let mut term = src.term.clone();
        map_term_uses(&mut term, remap);
        term.for_each_edge_mut(|t, _| {
            if let Some(&nb) = bmap.get(&t.0) {
                *t = BlockId(nb as u32);
            }
        });
        copies.push(BasicBlock {
            id: BlockId((base + i) as u32),
            params,
            insts,
            term,
        });
    }
    f.blocks.extend(copies);
    ((base..base + blocks.len()).collect(), bmap, vmap)
}

/// Within `blocks`, replace every use of `v` by the constant `k` (`t`/`fv` are the `i1` constants
/// `true`/`false`, defined where they dominate `blocks`) and fold the `and`/`or`/`not` that exposes.
/// The definitions of the folded values stay, for DCE.
fn assume(f: &mut Function, blocks: &[usize], v: ValueId, k: bool, t: ValueId, fv: ValueId) {
    let mut subst: FxHashMap<u32, ValueId> = FxHashMap::default();
    let mut known: FxHashMap<u32, bool> = FxHashMap::default();
    subst.insert(v.0, if k { t } else { fv });
    known.insert(t.0, true);
    known.insert(fv.0, false);
    let Function {
        blocks: fblocks,
        value_types,
        ..
    } = f;
    // A fold in one block can feed an instruction of another block visited earlier, so the sweep
    // repeats until nothing new is learned (one or two rounds in practice).
    for _ in 0..4 {
        let mut grew = false;
        for &b in blocks {
            for inst in &mut fblocks[b].insts {
                map_op_uses(&mut inst.op, |u| subst.get(&u.0).copied().unwrap_or(u));
                let Some(r) = inst.result else {
                    continue;
                };
                if subst.contains_key(&r.0) || value_types[r.0 as usize] != MirType::I1 {
                    continue;
                }
                let kn = |x: &ValueId| known.get(&x.0).copied();
                let repl = match &inst.op {
                    Op::Bin(BinOp::And, a, c) => match (kn(a), kn(c)) {
                        (Some(false), _) | (_, Some(false)) => Some(fv),
                        (Some(true), _) => Some(*c),
                        (_, Some(true)) => Some(*a),
                        _ => None,
                    },
                    Op::Bin(BinOp::Or, a, c) => match (kn(a), kn(c)) {
                        (Some(true), _) | (_, Some(true)) => Some(t),
                        (Some(false), _) => Some(*c),
                        (_, Some(false)) => Some(*a),
                        _ => None,
                    },
                    Op::Not(a) => kn(a).map(|x| if x { fv } else { t }),
                    _ => None,
                };
                if let Some(rv) = repl {
                    if let Some(kv) = known.get(&rv.0).copied() {
                        known.insert(r.0, kv);
                    }
                    subst.insert(r.0, rv);
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    for &b in blocks {
        map_term_uses(&mut fblocks[b].term, |u| {
            subst.get(&u.0).copied().unwrap_or(u)
        });
    }
}

/// Emit `op` (of type `ty`) into `out`, returning its value.
fn emit(f: &mut Function, out: &mut Vec<Inst>, ty: MirType, op: Op) -> ValueId {
    let v = fresh(f, ty);
    out.push(Inst {
        result: Some(v),
        op,
    });
    v
}

fn materialize(f: &mut Function, out: &mut Vec<Inst>, a: Avail, ty: &MirType) -> ValueId {
    match a {
        Avail::Outside(v) => v,
        Avail::Const(n) => emit(f, out, ty.clone(), Op::ConstInt(n, ty.clone())),
    }
}

fn apply_split(f: &mut Function, p: SplitPlan) {
    route_exit_values(f, &p.routes);
    let ph = p.preheader.0 as usize;
    let ity = f.value_type(p.iv).clone();
    let Terminator::Br { args, .. } = &f.blocks[ph].term else {
        unreachable!("a preheader ends in an unconditional branch to the header");
    };
    let start = args[p.iv_index];
    let mut out = Vec::new();
    let end = materialize(f, &mut out, p.end, &ity);
    let inv = materialize(f, &mut out, p.inv, &ity);
    let lt = if p.signed { CmpOp::Slt } else { CmpOp::Ult };
    // The first iteration where the guard changes, clamped into [start, end].
    let split = match p.kind {
        GuardKind::Lt | GuardKind::Ge => {
            let inv_lt_end = emit(f, &mut out, MirType::I1, Op::Cmp(lt, inv, end));
            let lo = emit(f, &mut out, ity.clone(), Op::Select(inv_lt_end, inv, end));
            let lo_lt_start = emit(f, &mut out, MirType::I1, Op::Cmp(lt, lo, start));
            emit(f, &mut out, ity.clone(), Op::Select(lo_lt_start, start, lo))
        }
        GuardKind::Le | GuardKind::Gt => {
            let one = emit(f, &mut out, ity.clone(), Op::ConstInt(1, ity.clone()));
            let next = emit(f, &mut out, ity.clone(), Op::Bin(BinOp::Add, inv, one));
            let inv_lt_end = emit(f, &mut out, MirType::I1, Op::Cmp(lt, inv, end));
            let next_lt_start = emit(f, &mut out, MirType::I1, Op::Cmp(lt, next, start));
            let hi = emit(
                f,
                &mut out,
                ity.clone(),
                Op::Select(next_lt_start, start, next),
            );
            emit(f, &mut out, ity.clone(), Op::Select(inv_lt_end, hi, end))
        }
    };
    let t = emit(f, &mut out, MirType::I1, Op::ConstInt(1, MirType::I1));
    let fv = emit(f, &mut out, MirType::I1, Op::ConstInt(0, MirType::I1));
    f.blocks[ph].insts.extend(out);

    let originals: Vec<usize> = p.blocks.iter().map(|b| b.0 as usize).collect();
    let (copy, bmap, vmap) = clone_loop(f, &p.blocks);
    let ha = bmap[&p.header.0];
    // The copy runs [start, split) and then hands its whole state to the original header.
    let iv_a = vmap[&p.iv.0];
    let ca = fresh(f, MirType::I1);
    f.blocks[ha].insts.push(Inst {
        result: Some(ca),
        op: Op::Cmp(lt, iv_a, split),
    });
    let state = f.blocks[ha].params.clone();
    match &mut f.blocks[ha].term {
        Terminator::CondBr {
            cond,
            else_blk,
            else_args,
            ..
        } => {
            *cond = ca;
            *else_blk = p.header;
            *else_args = state;
        }
        _ => unreachable!("the copied header ends in the exit test"),
    }
    match &mut f.blocks[ph].term {
        Terminator::Br { target, .. } => *target = BlockId(ha as u32),
        _ => unreachable!("a preheader ends in an unconditional branch to the header"),
    }
    let prefix = matches!(p.kind, GuardKind::Lt | GuardKind::Le);
    // The guard holds its assumed value on every iteration a copy's *body* runs, but each header
    // also runs once more, for the exit test, with `iv` already past that copy's range (`split`
    // in the copy, `end` in the original). A header-computed guard there can take the other value
    // and reach the exit arguments or the code after the loop, so headers are never rewritten.
    let header = p.header.0 as usize;
    let copy_body: Vec<usize> = copy.iter().copied().filter(|&b| b != ha).collect();
    let orig_body: Vec<usize> = originals.iter().copied().filter(|&b| b != header).collect();
    assume(f, &copy_body, vmap[&p.guard.0], prefix, t, fv);
    assume(f, &orig_body, p.guard, !prefix, t, fv);
}

fn apply_unswitch(f: &mut Function, p: UnswitchPlan) {
    let ph = p.preheader.0 as usize;
    let mut out = Vec::new();
    let t = emit(f, &mut out, MirType::I1, Op::ConstInt(1, MirType::I1));
    let fv = emit(f, &mut out, MirType::I1, Op::ConstInt(0, MirType::I1));
    f.blocks[ph].insts.extend(out);
    let originals: Vec<usize> = p.blocks.iter().map(|b| b.0 as usize).collect();
    let (copy, bmap, _) = clone_loop(f, &p.blocks);
    let Terminator::Br { target, args } = &f.blocks[ph].term else {
        unreachable!("a preheader ends in an unconditional branch to the header");
    };
    debug_assert_eq!(*target, p.header);
    let args = args.clone();
    f.blocks[ph].term = Terminator::CondBr {
        cond: p.cond,
        then_blk: p.header,
        then_args: args.clone(),
        else_blk: BlockId(bmap[&p.header.0] as u32),
        else_args: args,
    };
    assume(f, &originals, p.cond, true, t, fv);
    assume(f, &copy, p.cond, false, t, fv);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dce, Simplify, SimplifyCfg};
    use wukong_mir::{Builder, Program};
    use wukong_span::{Interner, SourceId};

    /// `f(start, end, inv) -> i32`, a canonical counted loop whose body guards an
    /// order-dependent update with `guard(j, inv)`:
    ///
    /// ```text
    ///   entry: br H(start, 7)
    ///   H(j, acc):  cond_br (j <exit end), B, X(acc)
    ///   B:          cond_br guard, T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   X(r):       ret r
    /// ```
    ///
    /// `swap` writes the guard as `inv ⋈' j`.
    fn guarded_loop(it: &mut Interner, exit: CmpOp, pred: CmpOp, swap: bool) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let start = b.add_param(MirType::I32);
        let end = b.add_param(MirType::I32);
        let inv = b.add_param(MirType::I32);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        b.br(h, vec![start, seven]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(exit, j, end));
        b.cond_br(c, body, vec![], exit_blk, vec![acc]);
        b.switch_to(body);
        let g = if swap {
            b.build(MirType::I1, Op::Cmp(pred, inv, j))
        } else {
            b.build(MirType::I1, Op::Cmp(pred, j, inv))
        };
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(exit_blk);
        b.ret(Some(r));
        b.finish()
    }

    /// Run `f` on every triple from `main`, printing each result.
    fn run3(f: &Function, it: &mut Interner, inputs: &[(i128, i128, i128)]) -> Vec<u8> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &(a, b, c) in inputs {
            let args = [a, b, c].map(|v| m.build(MirType::I32, Op::ConstInt(v, MirType::I32)));
            let r = m.build(
                MirType::I32,
                Op::Call {
                    func: f.name,
                    args: args.to_vec(),
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

    /// Every (start, end, inv) combination: empty and reversed ranges, an invariant before, inside
    /// and after the range, and the type's extremes (where `inv + 1` would wrap).
    fn grid(unsigned: bool) -> Vec<(i128, i128, i128)> {
        let starts: &[i128] = if unsigned { &[0, 2, 6] } else { &[-3, 0, 5] };
        let ends: &[i128] = if unsigned { &[0, 4, 9] } else { &[-1, 4, 9] };
        let invs: &[i128] = &[-2147483648, -5, -1, 0, 1, 2, 4, 5, 8, 9, 20, 2147483647];
        let mut g = Vec::new();
        for &s in starts {
            for &e in ends {
                for &i in invs {
                    g.push((s, e, i));
                }
            }
        }
        g
    }

    #[test]
    fn iv_range_split_keeps_every_answer() {
        use CmpOp::*;
        for (exit, preds) in [(Slt, [Slt, Sle, Sgt, Sge]), (Ult, [Ult, Ule, Ugt, Uge])] {
            let inputs = grid(exit == Ult);
            for pred in preds {
                for swap in [false, true] {
                    let mut it = Interner::new();
                    let mut f = guarded_loop(&mut it, exit, pred, swap);
                    let before = run3(&f, &mut it, &inputs);
                    assert!(
                        LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                        "{exit:?}/{pred:?}/swap={swap}: the guard is splittable"
                    );
                    assert!(wukong_mir::verify::verify_function(&f).is_empty());
                    assert_eq!(
                        run3(&f, &mut it, &inputs),
                        before,
                        "{exit:?}/{pred:?}/{swap}"
                    );
                    cleanup(&mut f);
                    // Both copies decided the guard, so no branch is left in either body: only
                    // the two exit tests remain.
                    let branches = f
                        .blocks
                        .iter()
                        .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
                        .count();
                    assert_eq!(
                        branches,
                        2,
                        "{}",
                        wukong_mir::print::print_function(&f, &it)
                    );
                    assert_eq!(
                        run3(&f, &mut it, &inputs),
                        before,
                        "{exit:?}/{pred:?}/{swap}"
                    );
                }
            }
        }
    }

    /// A guard the *header* computes also runs at the exit test, where `j` has left the body's
    /// range, and here it reaches the exit arguments: the split assumes it in the bodies only, so the
    /// exit still sees the guard's real final value (`end < inv`).
    ///
    /// ```text
    ///   H(j, acc):  g = j < inv; c = j < end; cond_br c, B, X(acc, g)
    ///   B:          cond_br g, T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   X(r, q):    ret r + 1000 * zext(q)
    /// ```
    #[test]
    fn a_header_guard_keeps_its_value_at_the_exit_test() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let start = b.add_param(MirType::I32);
        let end = b.add_param(MirType::I32);
        let inv = b.add_param(MirType::I32);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        let q = b.block_param(exit_blk, MirType::I1);
        b.br(h, vec![start, seven]);
        b.switch_to(h);
        let g = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, inv));
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, end));
        b.cond_br(c, body, vec![], exit_blk, vec![acc, g]);
        b.switch_to(body);
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(exit_blk);
        let w = b.build(
            MirType::I32,
            Op::Cast(wukong_mir::CastKind::ZExt, q, MirType::I32),
        );
        let k = b.build(MirType::I32, Op::ConstInt(1000, MirType::I32));
        let wk = b.build(MirType::I32, Op::Bin(BinOp::Mul, w, k));
        let out = b.build(MirType::I32, Op::Bin(BinOp::Add, r, wk));
        b.ret(Some(out));
        let mut f = b.finish();

        let inputs = grid(false);
        let before = run3(&f, &mut it, &inputs);
        assert!(LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(run3(&f, &mut it, &inputs), before);
        cleanup(&mut f);
        assert_eq!(run3(&f, &mut it, &inputs), before);
    }

    /// How the side exit of [`side_exit_loop`] reads the induction variable.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Leak {
        /// Only through the exit edge's arguments.
        None,
        /// Directly, in the exit block itself (as a bounds check's trap reads the index).
        InExit,
        /// Directly, in a block after the exit block.
        Downstream,
    }

    /// `guarded_loop` with a side exit ahead of the guard -- the shape a bounds check's trap or a
    /// `break` gives -- leaving with the accumulator once it passes 1000:
    ///
    /// ```text
    ///   H(j, acc):  cond_br (j < end), B, X(acc)
    ///   B:          cond_br (acc > 1000), E(acc), G
    ///   G:          cond_br guard, T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   E(q):       ret q + 5             (InExit: ret q + 5 + j; Downstream: br D(q + 5))
    ///   D(z):       ret z + j             (Downstream only)
    ///   X(r):       ret r
    /// ```
    fn side_exit_loop(it: &mut Interner, pred: CmpOp, leak: Leak) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let start = b.add_param(MirType::I32);
        let end = b.add_param(MirType::I32);
        let inv = b.add_param(MirType::I32);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let lim = b.build(MirType::I32, Op::ConstInt(1000, MirType::I32));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let guard_blk = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let side = b.new_block();
        let q = b.block_param(side, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        b.br(h, vec![start, seven]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, end));
        b.cond_br(c, body, vec![], exit_blk, vec![acc]);
        b.switch_to(body);
        let big = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, acc, lim));
        b.cond_br(big, side, vec![acc], guard_blk, vec![]);
        b.switch_to(guard_blk);
        let g = b.build(MirType::I1, Op::Cmp(pred, j, inv));
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(side);
        let five = b.build(MirType::I32, Op::ConstInt(5, MirType::I32));
        let mut out = b.build(MirType::I32, Op::Bin(BinOp::Add, q, five));
        match leak {
            Leak::None => b.ret(Some(out)),
            Leak::InExit => {
                out = b.build(MirType::I32, Op::Bin(BinOp::Add, out, j));
                b.ret(Some(out));
            }
            Leak::Downstream => {
                let d = b.new_block();
                let z = b.block_param(d, MirType::I32);
                b.br(d, vec![out]);
                b.switch_to(d);
                let w = b.build(MirType::I32, Op::Bin(BinOp::Add, z, j));
                b.ret(Some(w));
            }
        }
        b.switch_to(exit_blk);
        b.ret(Some(r));
        b.finish()
    }

    /// A loop with a side exit still splits on its induction-variable guard: whichever copy runs
    /// the iteration that leaves does so with the same state the unsplit loop had there. A side
    /// exit block that reads the induction variable itself gets it as an edge argument instead.
    #[test]
    fn iv_range_split_with_a_side_exit_keeps_every_answer() {
        use CmpOp::*;
        for leak in [Leak::None, Leak::InExit] {
            let mut inputs = grid(false);
            // Long enough ranges that the accumulator passes 1000 inside either copy.
            for &(s, e, i) in &[
                (0, 12, 3),
                (0, 12, 9),
                (-3, 20, 0),
                (-3, 20, 15),
                (2, 30, 30),
            ] {
                inputs.push((s, e, i));
            }
            for pred in [Slt, Sle, Sgt, Sge] {
                let mut it = Interner::new();
                let mut f = side_exit_loop(&mut it, pred, leak);
                let before = run3(&f, &mut it, &inputs);
                // The side exit is reached in some of these runs, or the test would prove nothing.
                let reached = before
                    .split(|&c| c == b'\n')
                    .filter_map(|l| std::str::from_utf8(l).ok()?.parse::<i64>().ok())
                    .any(|v| v > 1000);
                assert!(reached, "{pred:?}: no run took the side exit");
                assert!(
                    LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                    "{pred:?}: the guard is splittable past a side exit"
                );
                assert!(wukong_mir::verify::verify_function(&f).is_empty());
                assert_eq!(run3(&f, &mut it, &inputs), before, "{pred:?}");
                cleanup(&mut f);
                assert!(wukong_mir::verify::verify_function(&f).is_empty());
                // No compare against `inv` survives in either copy (the preheader's split-point
                // arithmetic is all selects).
                let inv = f.blocks[0].params[2];
                let guards = f
                    .blocks
                    .iter()
                    .skip(1)
                    .flat_map(|b| &b.insts)
                    .filter(
                        |i| matches!(i.op, Op::Cmp(p, x, y) if p == pred && (x == inv || y == inv)),
                    )
                    .count();
                assert_eq!(guards, 0, "{}", wukong_mir::print::print_function(&f, &it));
                assert_eq!(run3(&f, &mut it, &inputs), before, "{leak:?}/{pred:?}");
            }
        }
    }

    /// Code after a side exit that reads a loop value directly, beyond the exit block where an edge
    /// argument could carry it, would read the original loop's definition after leaving from the
    /// copy: no split.
    #[test]
    fn a_side_exit_reading_a_loop_value_downstream_blocks_the_split() {
        let mut it = Interner::new();
        let mut f = side_exit_loop(&mut it, CmpOp::Sgt, Leak::Downstream);
        let inputs = [(0, 12, 3), (0, 12, 9), (-3, 4, 0)];
        let before = run3(&f, &mut it, &inputs);
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(run3(&f, &mut it, &inputs), before);
    }

    /// A body branch on a loop-invariant condition (also as one operand of an `and`) is decided once
    /// in the preheader.
    #[test]
    fn an_invariant_branch_is_unswitched() {
        for with_and in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::I32);
            let n = b.add_param(MirType::I32);
            let x = b.add_param(MirType::I32);
            let y = b.add_param(MirType::I32);
            let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
            let flag = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, zero));
            let h = b.new_block();
            let j = b.block_param(h, MirType::I32);
            let acc = b.block_param(h, MirType::I32);
            let body = b.new_block();
            let then = b.new_block();
            let latch = b.new_block();
            let p = b.block_param(latch, MirType::I32);
            let exit_blk = b.new_block();
            let r = b.block_param(exit_blk, MirType::I32);
            b.br(h, vec![zero, y]);
            b.switch_to(h);
            let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
            b.cond_br(c, body, vec![], exit_blk, vec![acc]);
            b.switch_to(body);
            let g = if with_and {
                // `flag && acc is odd`: the second operand varies, so this is not a split.
                let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
                let bit = b.build(MirType::I32, Op::Bin(BinOp::And, acc, one));
                let odd = b.build(MirType::I1, Op::Cmp(CmpOp::Ne, bit, zero));
                b.build(MirType::I1, Op::Bin(BinOp::And, flag, odd))
            } else {
                flag
            };
            b.cond_br(g, then, vec![], latch, vec![acc]);
            b.switch_to(then);
            let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
            let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
            let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
            b.br(latch, vec![s]);
            b.switch_to(latch);
            let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
            let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
            b.br(h, vec![j2, p]);
            b.switch_to(exit_blk);
            b.ret(Some(r));
            let mut f = b.finish();

            let inputs: Vec<(i128, i128, i128)> = [0, 3, 7]
                .into_iter()
                .flat_map(|n| [-2, 0, 5].map(move |x| (n, x, 5)))
                .collect();
            let before = run3(&f, &mut it, &inputs);
            assert!(LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
            assert!(wukong_mir::verify::verify_function(&f).is_empty());
            assert!(
                matches!(f.blocks[0].term, Terminator::CondBr { cond, .. } if cond == flag),
                "the preheader decides the flag: {}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(run3(&f, &mut it, &inputs), before);
            cleanup(&mut f);
            assert_eq!(run3(&f, &mut it, &inputs), before);
        }
    }

    /// A loop whose value is read after it other than through an exit edge's arguments is not
    /// unswitched: the copy's exit would bypass that value's definition.
    #[test]
    fn a_loop_value_read_after_the_loop_blocks_unswitching() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let x = b.add_param(MirType::I32);
        let _y = b.add_param(MirType::I32);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let flag = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, zero));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let exit_blk = b.new_block();
        b.br(h, vec![zero]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
        b.cond_br(c, body, vec![], exit_blk, vec![]);
        b.switch_to(body);
        b.cond_br(flag, then, vec![], latch, vec![]);
        b.switch_to(then);
        b.br(latch, vec![]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2]);
        b.switch_to(exit_blk);
        // `j` is read after the loop directly: a leak.
        b.ret(Some(j));
        let mut f = b.finish();
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(
            run3(&f, &mut it, &[(4, 1, 0), (4, -1, 0)]),
            b"4\n4\n".to_vec()
        );
    }

    /// From source, at every level: a causal (strictly upper) mask, its lower complement and a
    /// `>=` against a parameter, all splittable, compute what the scalar loops say.
    #[test]
    fn source_masks_split_and_agree() {
        let src = concat!(
            "fn upper(mut s: [i32; 16]) { for i in 0..4 { for j in 0..4 { ",
            "if j > i { s[i * 4 + j] = 0 - 1; } } } }\n",
            "fn lower(mut s: [i32; 16]) { for i in 0..4 { for j in 0..4 { ",
            "if j < i { s[i * 4 + j] = 0 - 2; } } } }\n",
            "fn from(mut s: [i32; 16], k: i32) { for j in 0..16 { if j >= k { s[j] = 9; } } }\n",
            "fn main() -> i32 { let mut a: [i32; 16] = [0; 16]; let mut b: [i32; 16] = [0; 16]; ",
            "let mut c: [i32; 16] = [0; 16]; ",
            "for t in 0..16 { a[t] = t as i32; b[t] = t as i32; c[t] = t as i32; } ",
            "upper(a); lower(b); from(c, 11); ",
            "for t in 0..16 { print(a[t] * 100 + b[t] * 10 + c[t]); } return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        // Lane t = 4i + j: a = -1 when j > i else t; b = -2 when j < i else t; c = 9 when t >= 11
        // else t. Printed as a*100 + b*10 + c.
        let mut expected = String::new();
        for t in 0..16i64 {
            let (i, j) = (t / 4, t % 4);
            let a = if j > i { -1 } else { t };
            let b = if j < i { -2 } else { t };
            let c = if t >= 11 { 9 } else { t };
            expected.push_str(&format!("{}\n", a * 100 + b * 10 + c));
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
    }
}
