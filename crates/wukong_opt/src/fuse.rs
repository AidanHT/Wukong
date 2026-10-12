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
//! A body need not be one block. When either loop's body still branches -- an `if`/`else` of
//! stores, a clamp written as a branch -- the two bodies are spliced as regions instead: the first
//! body's latch runs on into the second body (absorbing its entry block, which only the second
//! header entered), and the second body's latch carries both loops' next values back to the fused
//! header (at most [`MAX_FUSE_BLOCKS`] blocks per loop):
//!
//! ```text
//!   H1(i, ..):  i < e ? E1 : PH2       H1(i, .., c2..): i < e ? E1 : X2(..)
//!   E1 .. L1(..): ..; br H1(back1)  => E1 .. L1(..): ..; <E2's instructions and branch>
//!   H2(j, c2..): j < e ? E2 : X2       .. L2(..): ..; br H1(back1, c2_next..)
//!   E2 .. L2(..): ..; br H2(j + 1, ..)
//! ```
//!
//! `back1` still dominates `L2`: every path into the second body now passes through `L1`. When
//! both bodies are single blocks the first form is used unchanged, so a pair that fused before
//! produces the same MIR.
//!
//! # When it is legal
//!
//! * Both loops are innermost counted loops (`iv < end` tested in the header, unit step, one
//!   latch, one exit -- the header's -- and no early return) whose body is one block, or a region
//!   of jumps and two-way branches entered from the header by one parameterless block, with no
//!   call, `alloca`, vector-kernel call, barrier, atomic or vector value, and they run the **same**
//!   iterations: equal start and end (the same value, or equal constants), the same predicate and
//!   induction-variable type.
//! * The first loop exits straight into the second's preheader, which nothing else reaches and
//!   whose instructions are pure and cannot trap or read memory — they move ahead of the first
//!   loop. Parameters of that preheader are the first loop's **live-outs**, passed by its exit edge;
//!   after fusion the code that read them reads what the edge passed (values of the fused header,
//!   which hold the first loop's final state when the loop exits). A live-out read straight from
//!   the first header needs nothing at all.
//! * The second loop (and its preheader) reads nothing the first loop defines, live-outs included:
//!   a value of the first loop is only right once that loop has finished.
//! * **Memory.** Fusion runs iteration `i` of the second loop before iterations `i + 1..` of the
//!   first, so every pair of accesses where one writes must keep its order: the first loop's
//!   access first. Two accesses to one object — the same base, or two bases the alias analysis
//!   places at a whole number of elements apart (`t` and `gep t, 2`) — need the same nonzero
//!   constant stride `c` and invariant part; they touch one element when `j - i = (ka - kb) / c`,
//!   and the fused loop keeps their order exactly when that **distance** is not negative (or `c`
//!   does not divide `ka - kb`: no common element). Reading what the first loop stored `d`
//!   iterations back (`t[j - d]` after `t[i]`) fuses; reading ahead (`t[j + d]`) does not. A nonzero
//!   distance needs element-sized accesses so that two never partially overlap. Accesses to
//!   different objects must be proved disjoint by [`AliasInfo`] — distinct stack slots, a
//!   `@noalias` parameter against any other, a `@noalias` slice's elements (reloaded from a header
//!   nothing writes) — and a base reloaded inside the loop must be stable: no store of either loop
//!   may write the slot it is reloaded from. An unanalyzable address, a loop-invariant address that
//!   is written, or two possibly-aliased bases decline. A store of a pointer declines (it could
//!   move a reloaded slice base). The rule covers every access in every block of both bodies, and
//!   a branching body needs nothing more: an access under a condition runs on a subset of the
//!   iterations, so it only removes executions from a set of pairs already proved reorderable.
//! * **Profit.** A loop the vectorizer can widen is never fused with one it cannot, and two
//!   widenable loops fuse only when every access has one width -- and, when either body branches,
//!   only when the vectorizer still widens the fused loop (tried on a copy of the function: an
//!   if-converted region sinks its stores to one address, which two bodies' stores may not share).
//!   Two loops it cannot widen fuse unless either body branches on a condition `unswitch` may
//!   decide later in the sweep, once LICM has run -- an invariant, or the counter (or
//!   `ext(iv ± k)`) against an invariant ([`may_decide_a_branch`]): split or unswitched alone, that
//!   loop may become copies the vectorizer widens, while fused, every copy would carry the other
//!   loop's unwidenable work. That test is conservative -- such a pair stays apart even when no
//!   copy would widen -- and every verdict is taken on the loops as fusion sees them, so fusion
//!   never costs a loop the vector body that the vectorizer, directly or after a split or unswitch,
//!   would give it there.
//!
//! The fused loop evaluates every operation of both loops on the same values, in an order whose
//! only change is the interleaving the memory rule just proved unobservable, so results are
//! bit-identical. The intermediate's store and reload then sit in one block, where load forwarding
//! removes the reload; dead-store elimination of the buffer itself is a separate pass.
//!
//! [`may_decide_a_branch`]: crate::unswitch::may_decide_a_branch

use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, Function, MirType, Op, Terminator, ValueId};

use crate::alias::{type_bytes, AliasInfo, Prov};
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{
    self, AccessKind, AddrForm, AffineExpr, Coeff, IvStep, LoopForest, MemAccess, MemBase,
    NaturalLoop, Stages,
};
use crate::{cfg, each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// The largest loop, in instructions, considered for fusion.
const MAX_LOOP_INSTS: usize = 64;
/// The most blocks, header included, a loop with a branching body may have to be fused.
const MAX_FUSE_BLOCKS: usize = 8;
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
            apply(f, &plan);
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
    /// The second body's first block: its header's staying target (`latch2` for a two-block loop).
    entry2: usize,
    /// Both loops are [`Shape::TwoBlock`]: the second body moves into the first's block.
    /// Otherwise the two bodies are spliced as regions (see the module docs).
    two_block: bool,
    /// The first loop's induction variable, which replaces the second's.
    iv1: ValueId,
    /// The second loop's induction variable's position among its header's parameters.
    iv2_index: usize,
}

/// The two loop shapes fusion accepts (see [`shape_ok`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    /// `H: cond_br c, L, X` with `L: …; br H(..)`: one body block, which is also the latch.
    TwoBlock,
    /// `H: cond_br c, E, X` with `E` .. `L` a branching body: jumps and two-way branches from the
    /// entry `E` to the latch `L: …; br H(..)`, which may take the join's parameters.
    Region,
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

/// An innermost counted loop with nothing fusion cannot reorder or copy, and which [`Shape`] it
/// has: two blocks, `H: cond_br c, L, X` with `L: …; br H(..)`, or a branching body of at most
/// [`MAX_FUSE_BLOCKS`] blocks entered by one parameterless block, whose blocks end only in jumps
/// and two-way branches and whose one latch jumps back to the header (it may take parameters: the
/// join of an `if`/`else` is the latch). Either way the header's false arm is the only exit.
fn shape_ok(f: &Function, l: &NaturalLoop) -> Option<Shape> {
    if !l.children.is_empty()
        || l.preheader.is_none()
        || l.blocks.len() > MAX_FUSE_BLOCKS
        || l.latches.len() != 1
        || l.exits.len() != 1
        || l.exits[0].0 != l.header
        || !l.abnormal_exits.is_empty()
    {
        return None;
    }
    let latch = l.latches[0];
    let Terminator::CondBr {
        then_blk: entry,
        then_args,
        else_blk,
        ..
    } = &f.block(l.header).term
    else {
        return None;
    };
    if !then_args.is_empty() || *else_blk != l.exits[0].1 {
        return None;
    }
    if !matches!(&f.block(latch).term, Terminator::Br { target, .. } if *target == l.header) {
        return None;
    }
    let shape = if l.blocks.len() == 2 {
        if *entry != latch || !f.block(latch).params.is_empty() {
            return None;
        }
        Shape::TwoBlock
    } else {
        if *entry == l.header
            || *entry == latch
            || !l.blocks.contains(entry)
            || !f.block(*entry).params.is_empty()
        {
            return None;
        }
        let jumps_only = l.blocks.iter().all(|b| {
            *b == l.header
                || matches!(
                    f.block(*b).term,
                    Terminator::Br { .. } | Terminator::CondBr { .. }
                )
        });
        if !jumps_only {
            return None;
        }
        Shape::Region
    };
    let mut n = 0;
    for b in &l.blocks {
        let blk = f.block(*b);
        if blk.params.iter().any(|p| f.value_type(*p).is_vector()) {
            return None;
        }
        for inst in &blk.insts {
            n += 1;
            let bad = match &inst.op {
                Op::Alloca { .. }
                | Op::Call { .. }
                | Op::VecKernelCall { .. }
                | Op::Barrier
                | Op::AtomicAdd { .. } => true,
                // A pointer stored into memory could move a slice base another access reloads.
                Op::Store { value, .. } => *f.value_type(*value) == MirType::Ptr,
                _ => false,
            };
            if bad || inst.result.is_some_and(|r| f.value_type(r).is_vector()) {
                return None;
            }
        }
    }
    (n <= MAX_LOOP_INSTS).then_some(shape)
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

/// The pointer an access base denotes: the outer pointer itself, or the value of the in-loop
/// reload (whose provenance the alias analysis knows: a `@noalias` slice's elements).
fn base_ptr(base: MemBase) -> ValueId {
    match base {
        MemBase::Outer(v) => v,
        MemBase::ReloadedOuter { value, .. } => value,
    }
}

/// When `ba` and `bb` are two different values pointing into one object -- `t` and `gep t, 2`,
/// both known to the alias analysis with constant byte offsets -- how many `elem`s `bb` lies past
/// `ba`. `None` when that is not known or not a whole number of elements.
fn element_shift(alias: &AliasInfo, ba: MemBase, bb: MemBase, elem: &MirType) -> Option<i64> {
    let (pa, oa) = alias.base_and_offset(base_ptr(ba))?;
    let (pb, ob) = alias.base_and_offset(base_ptr(bb))?;
    if pa != pb || pa == Prov::Unknown {
        return None;
    }
    let size = i64::from(type_bytes(elem)?);
    let diff = ob.checked_sub(oa)?;
    (size > 0 && diff % size == 0).then(|| diff / size)
}

/// Are two accesses' objects provably disjoint? A `false` from [`AliasInfo::may_alias`] on two
/// base pointers is a statement about the objects, at every offset. A reloaded base must also be
/// *stable*: nothing either loop stores may write the slot it is reloaded from, or fusion could
/// interleave such a store with reloads that used to run after it.
fn disjoint(alias: &AliasInfo, ba: MemBase, bb: MemBase, stores: &[(ValueId, u32)]) -> bool {
    if alias.may_alias(base_ptr(ba), base_ptr(bb)) {
        return false;
    }
    [ba, bb].iter().all(|b| match b {
        MemBase::Outer(_) => true,
        MemBase::ReloadedOuter { addr, .. } => stores
            .iter()
            .all(|&(p, bytes)| !alias.may_alias_sized(p, bytes, *addr, 8)),
    })
}

/// Same-object affine accesses `a` (first loop, iteration `i`) and `b` (second loop, iteration
/// `j`), `b`'s index shifted by `shift` elements into `a`'s frame: may fusion reorder them?
///
/// With the identical stride `c` and invariant part, they touch one element exactly when
/// `c·i + ka = c·j + kb`, i.e. `j - i = (ka - kb) / c`. The fused loop runs `a`'s iteration `i`
/// before `b`'s iteration `j` exactly when `i <= j` (within one iteration the first body still runs
/// first), so the pair keeps its order -- `a` before `b`, as in the unfused program -- when that
/// distance is `>= 0`, and touches no common element when `c` does not divide `ka - kb`. Reading
/// the store `d` iterations back (`t[j - d]` after `t[i]`) is distance `d`; reading ahead
/// (`t[j + d]`) is `-d` and must stay unfused. A nonzero distance needs element-sized accesses (so
/// two of them never partially overlap); distance 0 is the same element at the same iteration.
fn ordered(
    ia: &AffineExpr,
    ib: &AffineExpr,
    shift: i64,
    a: &MemAccess,
    b: &MemAccess,
    elem: &MirType,
) -> bool {
    if ia.coeff != ib.coeff || ia.terms != ib.terms || a.value_ty != b.value_ty {
        return false;
    }
    let Coeff::Const(c) = ia.coeff else {
        return false;
    };
    if c == 0 {
        return false;
    }
    let Some(delta) = ib
        .konst
        .checked_add(shift)
        .and_then(|kb| ia.konst.checked_sub(kb))
    else {
        return false;
    };
    if delta == 0 {
        return true;
    }
    if type_bytes(&a.value_ty) != type_bytes(elem) {
        return false;
    }
    delta % c != 0 || delta / c > 0
}

/// Can the accesses `a` (first loop) and `b` (second loop), at least one a store, be reordered as
/// fusion reorders them (see the module docs)? `stores` lists every store of both loops (pointer,
/// width), for the stability of reloaded bases.
fn pair_ok(alias: &AliasInfo, a: &MemAccess, b: &MemAccess, stores: &[(ValueId, u32)]) -> bool {
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
        ) => {
            // One object -- the same base, or two bases at a known element distance -- compares
            // indices; two objects must be proved apart.
            let shift = if ba.same_object(*bb) {
                Some(0)
            } else if ea == eb {
                element_shift(alias, *ba, *bb, ea)
            } else {
                None
            };
            match shift {
                Some(sh) if ea == eb => ordered(ia, ib, sh, a, b, ea),
                _ => disjoint(alias, *ba, *bb, stores),
            }
        }
        _ => {
            let (Some(ba), Some(bb)) = (a.addr.base(), b.addr.base()) else {
                return false;
            };
            if ba.same_object(bb) {
                return false; // one object, but not a moving index on both sides
            }
            disjoint(alias, ba, bb, stores)
        }
    }
}

fn find_pair(f: &Function, forest: &mut LoopForest, preds: &[Vec<u32>]) -> Option<Plan> {
    let defs = def_sites(f);
    for i1 in 0..forest.loops.len() {
        let (h1, ph2, s1) = {
            let l1 = &forest.loops[i1];
            let Some(s1) = shape_ok(f, l1) else {
                continue;
            };
            (l1.header, l1.exits[0].1, s1)
        };
        let Some(i2) = forest.loops.iter().position(|l| l.preheader == Some(ph2)) else {
            continue;
        };
        if i2 == i1 {
            continue;
        }
        let Some(s2) = shape_ok(f, &forest.loops[i2]) else {
            continue;
        };
        // The second preheader is reached only by the first loop's exit and computes nothing
        // that could not run before the first loop. Its parameters, if any, are the first loop's
        // live-outs (`plan_pair` checks that only the code after the second loop reads them).
        let p2 = f.block(ph2);
        if preds[ph2.0 as usize] != [h1.0] || !p2.insts.iter().all(|i| movable(&i.op)) {
            continue;
        }
        loop_info::analyze_one(f, forest, i1, Stages::Memory);
        loop_info::analyze_one(f, forest, i2, Stages::Memory);
        let two_block = s1 == Shape::TwoBlock && s2 == Shape::TwoBlock;
        if let Some(plan) = plan_pair(f, &defs, &forest.loops[i1], &forest.loops[i2], two_block) {
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
    two_block: bool,
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
    // The second preheader's parameters are the first loop's live-outs, passed by its exit edge:
    // equally unfinished until the first loop is done.
    let live_outs = &f.block(ph2).params;
    let first = |v: ValueId| defined1.contains(&v.0) || live_outs.contains(&v);
    let mut reads_first = false;
    for b in std::iter::once(&ph2).chain(&l2.blocks) {
        let blk = f.block(*b);
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |v| reads_first |= first(v));
        }
        each_term_use(&blk.term, &mut |v| reads_first |= first(v));
    }
    if reads_first {
        return None;
    }
    // Memory.
    let alias = AliasInfo::analyze(f);
    let stores: Vec<(ValueId, u32)> = l1
        .accesses
        .iter()
        .chain(&l2.accesses)
        .filter(|x| x.kind == AccessKind::Store)
        .map(|x| (x.ptr, type_bytes(&x.value_ty).unwrap_or(u32::MAX)))
        .collect();
    for a in &l1.accesses {
        for b in &l2.accesses {
            if a.kind == AccessKind::Load && b.kind == AccessKind::Load {
                continue;
            }
            if !pair_ok(&alias, a, b, &stores) {
                return None;
            }
        }
    }
    // Profit: never trade a widenable loop for a fused one the vectorizer would refuse. The
    // verdict is the vectorizer's own, not the dependence shape alone: a loop that reads a 64-bit
    // counter as an `f32` value has a vectorizable *shape* and is still declined, and fusing it
    // with a widenable reduction loop would cost that loop its vector body.
    let (w1, w2) = (
        crate::vectorize::would_widen(f, l1),
        crate::vectorize::would_widen(f, l2),
    );
    if w1 != w2 {
        return None;
    }
    // Two loops it cannot widen as they stand: a branching one that `unswitch` may still split or
    // unswitch, later in this sweep, could become copies the vectorizer widens -- unless fusion has
    // put the other loop's unwidenable work into every copy. Keep such a loop alone.
    if !two_block
        && !w1
        && [(l1, c1.iv), (l2, c2.iv)]
            .into_iter()
            .any(|(l, iv)| crate::unswitch::may_decide_a_branch(f, l, iv))
    {
        return None;
    }
    if w1 {
        // A loop-invariant access -- a slice's data pointer reloaded from its header -- is hoisted
        // or splatted, never widened, so it does not take part in the width.
        let mut widths = l1
            .accesses
            .iter()
            .chain(&l2.accesses)
            .filter(|a| !matches!(a.addr, AddrForm::Invariant(_)))
            .map(|a| type_bytes(&a.value_ty));
        let first = widths.next().flatten();
        if first.is_none() || !widths.all(|w| w == first) {
            return None;
        }
    }
    let Terminator::CondBr { then_blk, .. } = &h2.term else {
        return None;
    };
    let plan = Plan {
        ph1: ph1.0 as usize,
        h1: l1.header.0 as usize,
        latch1: l1.latches[0].0 as usize,
        ph2: ph2.0 as usize,
        h2: l2.header.0 as usize,
        latch2: l2.latches[0].0 as usize,
        entry2: then_blk.0 as usize,
        two_block,
        iv1: c1.iv,
        iv2_index: c2.index,
    };
    // Two widenable loops whose fused body branches: the if-converters flatten one region with one
    // sunk store address, and the fused region may hold two. Fuse only when the vectorizer still
    // widens the result, asked on a fused copy of the function.
    if !two_block && w1 && !fused_loop_widens(f, &plan, c1.cond) {
        return None;
    }
    Some(plan)
}

/// Would the vectorizer widen the loop `plan` produces? Fuses a copy of `f` and finds the fused loop
/// by its header's exit test `cond` (values keep their ids when unreachable blocks are pruned).
fn fused_loop_widens(f: &Function, plan: &Plan, cond: ValueId) -> bool {
    let mut g = f.clone();
    apply(&mut g, plan);
    cfg::prune_unreachable(&mut g);
    let mut cache = CfgAnalyses::default();
    let mut forest = {
        let (idom, preds) = cache.idoms_and_preds(&g);
        loop_info::structure_with(&g, idom, preds)
    };
    let fused = forest.loops.iter().position(
        |l| matches!(&g.block(l.header).term, Terminator::CondBr { cond: c, .. } if *c == cond),
    );
    let Some(idx) = fused else {
        return false;
    };
    loop_info::analyze_one(&g, &mut forest, idx, Stages::Memory);
    crate::vectorize::would_widen(&g, &forest.loops[idx])
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

fn apply(f: &mut Function, p: &Plan) {
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

    // The first loop's live-outs, which its exit edge passed to the second preheader's parameters,
    // are read after the fused loop straight from what that edge passed: values of the fused
    // header (or from before it), which hold the first loop's final state when the loop exits.
    if let Terminator::CondBr { else_args, .. } = &f.blocks[p.h1].term {
        for (q, a) in f.blocks[p.ph2].params.iter().zip(else_args) {
            map.insert(q.0, *a);
        }
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

    // The fused latch passes both loops' next values back.
    let mut back = match &f.blocks[p.latch1].term {
        Terminator::Br { args, .. } => args.clone(),
        _ => unreachable!("a latch ends in an unconditional branch"),
    };
    let next2 = match &f.blocks[p.latch2].term {
        Terminator::Br { args, .. } => skip_iv(args.as_slice()),
        _ => unreachable!("a latch ends in an unconditional branch"),
    };
    back.extend(next2);
    if p.two_block {
        // The second body runs straight after the first, in the first's block: the fused loop
        // stays a two-block counted loop, so it can fuse again with the loop after it.
        let body2 = std::mem::take(&mut f.blocks[p.latch2].insts);
        f.blocks[p.latch2].term = Terminator::Unreachable;
        f.blocks[p.latch1].insts.extend(body2);
        f.blocks[p.latch1].term = Terminator::Br {
            target: BlockId(p.h1 as u32),
            args: back,
        };
    } else {
        // The bodies are spliced as regions: the first latch runs on into the second body and the
        // second latch closes the fused loop. The second body's entry was entered only from its
        // header (another in-loop predecessor would make it an inner loop's header), so the first
        // latch is now its only predecessor and simply absorbs it: the entry's instructions follow
        // the first latch's and its terminator becomes the first latch's -- which keeps the fused
        // loop as small as simplify-cfg would make it, so it can fuse again. Everything the second
        // body reads from the first latch's back edge, it dominates.
        f.blocks[p.latch2].term = Terminator::Br {
            target: BlockId(p.h1 as u32),
            args: back,
        };
        let entry_insts = std::mem::take(&mut f.blocks[p.entry2].insts);
        let entry_term = std::mem::replace(&mut f.blocks[p.entry2].term, Terminator::Unreachable);
        f.blocks[p.latch1].insts.extend(entry_insts);
        f.blocks[p.latch1].term = entry_term;
        f.blocks[p.h2].term = Terminator::Unreachable;
    }

    // Every read of the second header's parameters now reads the fused header's, and every read of
    // a live-out the second preheader received reads what the first loop's exit passed for it.
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

    /// `f(n)`: three loops over `0..n` on two 8-element stack buffers — `t[i] = i`, then
    /// `o[j] = t[j] + 1`, then `acc += o[k]` — returning `acc`. All three fuse into one.
    ///
    /// Every one of the three is a loop the vectorizer widens on its own (the first stores its
    /// counter, which a lane ramp answers), so fusion's profit rule is not what decides here. A
    /// first loop storing `3 * i` would not do: the vectorizer declines a scaled counter read as a
    /// value, and fusion then rightly refuses to give the second loop's vector body away for it.
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
        b.build_void(Op::Store { ptr: pt, value: i });
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
        // acc = Σ (i + 1) over 0..n: n = 0 -> 0, 1 -> 1, 4 -> 10, 8 -> 36.
        let ns = [0, 1, 4, 8];
        let before = run_n(&f, &mut it, &ns);
        assert_eq!(before, b"0\n1\n10\n36\n".to_vec());
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
        // o[j] = t[j + 1] + 1 = (j + 1) + 1 for j + 1 < n, and t[n] is the untouched slot
        // 100 + n: o[n - 1] = (100 + n) + 1. acc = Σ_{j < n-1} (j + 2) + 101 + n.
        let ns = [0, 1, 4, 8];
        let before = run_n(&f, &mut it, &ns);
        // n = 0: 0. n = 1: 102. n = 4: (2 + 3 + 4) + 105 = 114. n = 8: (2+3+..+8) + 109 = 144.
        assert_eq!(before, b"0\n102\n114\n144\n".to_vec());
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

    // ---- distances, live-outs and parameter bases ----

    /// How a [`two_loops`] function reaches its two buffers.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Bufs {
        /// Two local stack buffers.
        Local,
        /// Two pointer parameters, the first `@noalias` or not.
        Params { noalias: bool },
        /// One local buffer, read through a second base `gep t, 2` (a different SSA value for the
        /// same object, two elements in).
        Offset,
    }

    /// The shape of a [`two_loops`] function.
    #[derive(Clone, Copy, Debug)]
    struct Two {
        start: i128,
        /// The first loop stores `t[stride * i + store_off]`.
        stride: i128,
        store_off: i128,
        /// The second loop loads `t[stride * j + load_off]` (through the `Offset` base when asked).
        load_off: i128,
        /// The first loop also counts `acc += i`, read after the second loop.
        live_out: LiveOut,
        bufs: Bufs,
    }

    /// How the first loop's count `acc` reaches the code after the second loop.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum LiveOut {
        /// There is no count.
        No,
        /// Read straight from the first header's parameter.
        Direct,
        /// Passed by the first loop's exit edge to a parameter of the second preheader.
        ThroughPreheader,
    }

    const BUF: u32 = 48;

    /// `f(n[, t, o])`: two loops over `start..n`.
    ///
    /// ```text
    ///   L1: for i: t[stride*i + store_off] = 3*i + 1;    (acc += i when live_out)
    ///   L2: for j: h = h * 31 + (t[stride*j + load_off] + j);  o[j] = h
    ///   ret h + 1000 * acc
    /// ```
    ///
    /// `t` starts as `100 + k` in every slot, so a load that runs before the store it should see
    /// returns a different value, and the order-dependent hash `h` exposes any such change.
    fn two_loops(it: &mut Interner, s: Two) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let (t, o) = match s.bufs {
            Bufs::Params { noalias } => {
                let t = b.add_param(MirType::Ptr);
                let o = b.add_param(MirType::Ptr);
                b.set_param_attrs(
                    t,
                    wukong_mir::ParamAttrs {
                        noalias,
                        slice: false,
                        extent: None,
                    },
                );
                (t, o)
            }
            Bufs::Local | Bufs::Offset => (
                b.alloca(MirType::Array(Box::new(MirType::I32), BUF)),
                b.alloca(MirType::Array(Box::new(MirType::I32), BUF)),
            ),
        };
        let konst = |b: &mut Builder, c: i128| b.build(MirType::I32, Op::ConstInt(c, MirType::I32));
        if !matches!(s.bufs, Bufs::Params { .. }) {
            for k in 0..BUF as i128 {
                let idx = konst(&mut b, k);
                let p = b.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: t,
                        index: idx,
                        elem: MirType::I32,
                    },
                );
                let v = konst(&mut b, 100 + k);
                b.build_void(Op::Store { ptr: p, value: v });
            }
        }
        // The second loop's base: `t` itself, or `t + 2` elements for `Offset`.
        let t2 = if s.bufs == Bufs::Offset {
            let two = konst(&mut b, 2);
            b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: t,
                    index: two,
                    elem: MirType::I32,
                },
            )
        } else {
            t
        };
        let start = konst(&mut b, s.start);
        let zero = konst(&mut b, 0);
        let h1 = b.new_block();
        let i = b.block_param(h1, MirType::I32);
        let acc = b.block_param(h1, MirType::I32);
        let l1 = b.new_block();
        let p2 = b.new_block();
        let carried = b.block_param(p2, MirType::I32);
        let h2 = b.new_block();
        let j = b.block_param(h2, MirType::I32);
        let h = b.block_param(h2, MirType::I32);
        let l2 = b.new_block();
        let x = b.new_block();
        let r = b.block_param(x, MirType::I32);
        b.br(h1, vec![start, zero]);

        b.switch_to(h1);
        let c1 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c1, l1, vec![], p2, vec![acc]);
        b.switch_to(l1);
        let stride = konst(&mut b, s.stride);
        let si = b.build(MirType::I32, Op::Bin(BinOp::Mul, i, stride));
        let soff = konst(&mut b, s.store_off);
        let sidx = b.build(MirType::I32, Op::Bin(BinOp::Add, si, soff));
        let pt = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: t,
                index: sidx,
                elem: MirType::I32,
            },
        );
        let three = konst(&mut b, 3);
        let i3 = b.build(MirType::I32, Op::Bin(BinOp::Mul, i, three));
        let one = konst(&mut b, 1);
        let val = b.build(MirType::I32, Op::Bin(BinOp::Add, i3, one));
        b.build_void(Op::Store {
            ptr: pt,
            value: val,
        });
        let acc2 = if s.live_out != LiveOut::No {
            b.build(MirType::I32, Op::Bin(BinOp::Add, acc, i))
        } else {
            acc
        };
        let i2 = b.build(MirType::I32, Op::Bin(BinOp::Add, i, one));
        b.br(h1, vec![i2, acc2]);

        b.switch_to(p2);
        let start2 = konst(&mut b, s.start);
        let zero2 = konst(&mut b, 0);
        b.br(h2, vec![start2, zero2]);
        b.switch_to(h2);
        let c2 = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
        b.cond_br(c2, l2, vec![], x, vec![h]);
        b.switch_to(l2);
        let stride2 = konst(&mut b, s.stride);
        let sj = b.build(MirType::I32, Op::Bin(BinOp::Mul, j, stride2));
        let loff = konst(&mut b, s.load_off);
        let lidx = b.build(MirType::I32, Op::Bin(BinOp::Add, sj, loff));
        let qt = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: t2,
                index: lidx,
                elem: MirType::I32,
            },
        );
        let w = b.build(MirType::I32, Op::Load(qt, MirType::I32));
        let wj = b.build(MirType::I32, Op::Bin(BinOp::Add, w, j));
        let k31 = konst(&mut b, 31);
        let h31 = b.build(MirType::I32, Op::Bin(BinOp::Mul, h, k31));
        let hn = b.build(MirType::I32, Op::Bin(BinOp::Add, h31, wj));
        let qo = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: o,
                index: j,
                elem: MirType::I32,
            },
        );
        b.build_void(Op::Store { ptr: qo, value: hn });
        let one2 = konst(&mut b, 1);
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one2));
        b.br(h2, vec![j2, hn]);
        b.switch_to(x);
        let thousand = konst(&mut b, 1000);
        let count = if s.live_out == LiveOut::ThroughPreheader {
            carried
        } else {
            acc
        };
        let a1000 = b.build(MirType::I32, Op::Bin(BinOp::Mul, count, thousand));
        let out = b.build(MirType::I32, Op::Bin(BinOp::Add, r, a1000));
        b.ret(Some(out));
        b.finish()
    }

    /// Run `f(n[, t, o])` for each `n`, with fresh 48-element buffers `t[k] = 100 + k` per call
    /// when `f` takes them; `alias` passes the same buffer twice.
    fn run_two(f: &Function, it: &mut Interner, ns: &[i128], params: bool, alias: bool) -> Vec<u8> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &n in ns {
            let nv = m.build(MirType::I32, Op::ConstInt(n, MirType::I32));
            let mut args = vec![nv];
            if params {
                let t = m.alloca(MirType::Array(Box::new(MirType::I32), BUF));
                let o = m.alloca(MirType::Array(Box::new(MirType::I32), BUF));
                for k in 0..BUF as i128 {
                    let idx = m.build(MirType::I32, Op::ConstInt(k, MirType::I32));
                    for (buf, base) in [(t, 100), (o, 500)] {
                        let p = m.build(
                            MirType::Ptr,
                            Op::Gep {
                                ptr: buf,
                                index: idx,
                                elem: MirType::I32,
                            },
                        );
                        let v = m.build(MirType::I32, Op::ConstInt(base + k, MirType::I32));
                        m.build_void(Op::Store { ptr: p, value: v });
                    }
                }
                args.push(t);
                args.push(if alias { t } else { o });
            }
            let r = m.build(MirType::I32, Op::Call { func: f.name, args });
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

    /// Fuse `f` if the pass will, check the MIR, and compare every answer with the unfused run.
    fn fuse_and_compare(s: Two, ns: &[i128]) -> bool {
        let mut it = Interner::new();
        let mut f = two_loops(&mut it, s);
        let params = matches!(s.bufs, Bufs::Params { .. });
        let before = run_two(&f, &mut it, ns, params, false);
        assert_eq!(loop_count(&f), 2);
        let fused = LoopFuse.run_function(&mut f, &mut CfgAnalyses::default());
        assert!(wukong_mir::verify::verify_function(&f).is_empty(), "{s:?}");
        assert_eq!(
            loop_count(&f),
            if fused { 1 } else { 2 },
            "{s:?}\n{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run_two(&f, &mut it, ns, params, false), before, "{s:?}");
        fused
    }

    const NS: [i128; 6] = [0, 1, 2, 5, 9, 16];

    fn local(start: i128, stride: i128, store_off: i128, load_off: i128) -> Two {
        Two {
            start,
            stride,
            store_off,
            load_off,
            live_out: LiveOut::No,
            bufs: Bufs::Local,
        }
    }

    /// The second loop reads what the first wrote `d` iterations *earlier* (`t[j - d]` after
    /// `t[i]`): in the fused loop that store has already run, so they fuse for every `d > 0`, at
    /// unit and non-unit strides. Reading `d` iterations *later* (`t[j + d]`) must not fuse.
    #[test]
    fn a_read_behind_the_writer_fuses_at_any_distance() {
        for d in [1, 2, 5] {
            assert!(fuse_and_compare(local(5, 1, 0, -d), &NS), "t[j - {d}]");
            assert!(
                fuse_and_compare(local(5, 1, 0, 0), &NS),
                "distance 0 still fuses"
            );
            assert!(!fuse_and_compare(local(0, 1, 0, d), &NS), "t[j + {d}]");
        }
        // Stride 2: `t[2j - 2]` is the element written one iteration earlier; `t[2j + 1]` is never
        // an element the first loop writes; `t[2j + 2]` is written one iteration later.
        assert!(fuse_and_compare(local(1, 2, 0, -2), &NS));
        assert!(fuse_and_compare(local(0, 2, 0, 1), &NS));
        assert!(!fuse_and_compare(local(0, 2, 0, 2), &NS));
    }

    /// A value the first loop carries out, read after the second loop -- straight from the first
    /// header, or through a parameter of the second preheader that the first loop's exit edge
    /// fills -- reads the first loop's final value from the fused header.
    #[test]
    fn a_live_out_of_the_first_loop_read_after_the_second_fuses() {
        for live_out in [LiveOut::Direct, LiveOut::ThroughPreheader] {
            let s = Two {
                live_out,
                ..local(0, 1, 0, 0)
            };
            assert!(fuse_and_compare(s, &NS), "{live_out:?}");
        }
    }

    /// Two different bases of one object two elements apart (`t` and `t + 2`): reading
    /// `(t + 2)[j - 3]` is `t[j - 1]`, one iteration behind the writer, and fuses; `(t + 2)[j]` is
    /// `t[j + 2]`, ahead of it, and must not.
    #[test]
    fn two_bases_of_one_object_are_compared_at_their_offset() {
        let base = |load_off| Two {
            bufs: Bufs::Offset,
            ..local(3, 1, 0, load_off)
        };
        assert!(fuse_and_compare(base(-3), &NS));
        assert!(!fuse_and_compare(base(0), &NS));
    }

    /// Pointer parameters fuse when one is `@noalias` (the alias analysis proves the buffers
    /// disjoint) and not otherwise: the caller may pass the same buffer twice. With the promise
    /// broken by such a call, the unfused function is what defines the answer.
    #[test]
    fn parameter_bases_fuse_only_when_proved_disjoint() {
        let s = |noalias| Two {
            bufs: Bufs::Params { noalias },
            ..local(0, 1, 0, 0)
        };
        assert!(fuse_and_compare(s(true), &NS));
        assert!(!fuse_and_compare(s(false), &NS));
        // Without the promise an aliasing call is legal, and the unfused loops agree with it.
        let mut it = Interner::new();
        let f = two_loops(&mut it, s(false));
        let before = run_two(&f, &mut it, &NS, true, true);
        let mut g = f.clone();
        assert!(!LoopFuse.run_function(&mut g, &mut CfgAnalyses::default()));
        assert_eq!(run_two(&g, &mut it, &NS, true, true), before);
    }

    /// From source, the shapes of the `opt4_fuse_*` fixtures reach the pass as two adjacent
    /// canonical loops, and it fuses exactly the legal ones: a read one iteration behind the writer
    /// (unit stride and stride 2), a first-loop live-out read after the second loop, and `@noalias`
    /// slices (whose data pointers are reloaded in the loop); not a read ahead of the writer, and
    /// not slices without the promise, which the caller may alias. (The array parameters written
    /// here are `@noalias` too: unpromised, the caller could pass one array twice. The fixtures
    /// call them on distinct local arrays, which inlining turns into distinct stack slots.)
    #[test]
    fn source_shapes_fuse_exactly_when_legal() {
        let src = concat!(
            "fn dist1(a: [i32; 32], @noalias mut t: [i32; 64], @noalias mut o: [i32; 32], n: i32) { ",
            "for i in 1..n { t[i] = a[i] / 3 + 7; } ",
            "for j in 1..n { o[j] = t[j - 1] / 2 - t[j]; } }\n",
            "fn stride2(a: [i32; 32], @noalias mut t: [i32; 64], @noalias mut o: [i32; 32], n: i32) { ",
            "for i in 1..n { t[i * 2] = a[i] / 3; } ",
            "for j in 1..n { o[j] = t[j * 2 - 2] / 2 + j; } }\n",
            "fn ahead(a: [i32; 32], @noalias mut t: [i32; 64], @noalias mut o: [i32; 32], n: i32) { ",
            "for i in 0..n { t[i] = a[i] / 5; } ",
            "for j in 0..n { o[j] = t[j + 1] / 2; } }\n",
            "fn sum_scale(a: [i32; 16], @noalias mut b: [i32; 16], n: i32) -> i32 { let mut s = 0; ",
            "for i in 0..n { s = s + a[i] * a[i] - i; } ",
            "for j in 0..n { b[j] = a[j] / 3 + j; } return s; }\n",
            "@unchecked\nfn slices(@noalias mut tmp: []i32, @noalias mut out: []i32, src: []i32, ",
            "n: i64) { for i in 0..n { tmp[i] = src[i] / 2 - (i as i32); } ",
            "for j in 0..n { out[j] = tmp[j] / 3 + src[j]; } }\n",
            "@unchecked\nfn aliased(mut tmp: []i32, mut out: []i32, src: []i32, n: i64) { ",
            "for i in 0..n { tmp[i] = src[i] / 2 - (i as i32); } ",
            "for j in 0..n { out[j] = tmp[j] / 3 + src[j]; } }\n",
            "fn main() -> i32 { return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (mut program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        // What the -O2 fixpoint has done by the time it reaches `fuse`.
        let mut pm = crate::PassManager::new();
        pm.add(Box::new(crate::Mem2Reg));
        pm.add(Box::new(crate::Simplify));
        pm.add(Box::new(SimplifyCfg));
        pm.add(Box::new(crate::SimplifyPhis));
        pm.add(Box::new(Dce));
        pm.add(Box::new(crate::Cse));
        pm.add(Box::new(crate::LoopCanon));
        pm.run(&mut program);
        for (name, legal) in [
            ("dist1", true),
            ("stride2", true),
            ("ahead", false),
            ("sum_scale", true),
            ("slices", true),
            ("aliased", false),
        ] {
            let sym = interner.intern(name);
            let f = program
                .funcs
                .iter_mut()
                .find(|f| f.name == sym)
                .expect("defined above");
            assert_eq!(loop_count(f), 2, "{name}");
            let fused = LoopFuse.run_function(f, &mut CfgAnalyses::default());
            assert_eq!(
                fused,
                legal,
                "{name}\n{}",
                wukong_mir::print::print_function(f, &interner)
            );
            assert!(wukong_mir::verify::verify_function(f).is_empty(), "{name}");
            assert_eq!(loop_count(f), if legal { 1 } else { 2 }, "{name}");
        }
    }

    // ---- branching (multi-block) bodies ----

    /// `src` lowered and run through what the `-O2` fixpoint has done by the time it reaches
    /// `fuse` (the pre-pipeline of [`source_shapes_fuse_exactly_when_legal`]).
    fn before_fuse(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (mut program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let mut pm = crate::PassManager::new();
        pm.add(Box::new(crate::Mem2Reg));
        pm.add(Box::new(crate::Simplify));
        pm.add(Box::new(SimplifyCfg));
        pm.add(Box::new(crate::SimplifyPhis));
        pm.add(Box::new(Dce));
        pm.add(Box::new(crate::Cse));
        pm.add(Box::new(crate::LoopCanon));
        pm.run(&mut program);
        (program, interner)
    }

    /// The plan that fuses the first two adjacent loops of `f` as regions, with every legality and
    /// profit check skipped -- what [`apply`] would do if the pass said yes -- and the first
    /// loop's exit test, which names the fused loop.
    fn forced_plan(f: &Function) -> (Plan, ValueId) {
        let defs = def_sites(f);
        let forest = crate::loop_info::analyze_function(f);
        let (l1, l2) = forest
            .loops
            .iter()
            .find_map(|l1| {
                let l2 = forest
                    .loops
                    .iter()
                    .find(|l2| l2.preheader == Some(l1.exits[0].1))?;
                Some((l1, l2))
            })
            .expect("two adjacent loops");
        let (c1, c2) = (
            counted(f, &defs, l1).unwrap(),
            counted(f, &defs, l2).unwrap(),
        );
        let Terminator::CondBr { then_blk, .. } = &f.block(l2.header).term else {
            panic!("a counted loop's header ends in its exit test");
        };
        let plan = Plan {
            ph1: l1.preheader.unwrap().0 as usize,
            h1: l1.header.0 as usize,
            latch1: l1.latches[0].0 as usize,
            ph2: l2.preheader.unwrap().0 as usize,
            h2: l2.header.0 as usize,
            latch2: l2.latches[0].0 as usize,
            entry2: then_blk.0 as usize,
            two_block: false,
            iv1: c1.iv,
            iv2_index: c2.index,
        };
        (plan, c1.cond)
    }

    /// The kernels of the branching-body tests. Every array parameter written is `@noalias`
    /// (unpromised, a caller could pass one array twice). The two-block partners of a diamond carry
    /// a running recurrence `h`, which no vectorizer widens, and the diamonds divide in their arms,
    /// which keeps both the MIR vectorizer and the front end's elementwise vectorizer out: these
    /// pairs test legality, not profit. (`o[j] = t[j] / 2 + 1` alone would widen, by replicating the
    /// division, and fusion rightly refuses to trade that loop away for a diamond the vectorizer
    /// cannot flatten.)
    const BRANCHING_KERNELS: &str = concat!(
        // A diamond in the first loop.
        "fn first(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut o: [i32; 32], n: i32) { ",
        "for i in 0..n { if a[i] > 0 { t[i] = a[i] / 3; } else { t[i] = 0 - a[i] / 5; } } ",
        "let mut h: i32 = 0; ",
        "for j in 0..n { o[j] = t[j] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
        // A diamond in the second loop.
        "fn second(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut o: [i32; 32], n: i32) { ",
        "let mut h: i32 = 0; ",
        "for i in 0..n { t[i] = a[i] / 3 + h; h = (h * 5 + 1) % 7; } ",
        "for j in 0..n { if t[j] > 0 { o[j] = t[j] / 2; } else { o[j] = 0 - t[j] / 7; } } }\n",
        // Diamonds in both, then a third, two-block loop reading both intermediates.
        "fn both(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut u: [i32; 32], ",
        "@noalias mut o: [i32; 32], n: i32) { ",
        "for i in 0..n { if a[i] > 0 { t[i] = a[i] / 3; } else { t[i] = 0 - a[i] / 5; } } ",
        "for j in 0..n { if t[j] > 2 { u[j] = t[j] / 2; } else { u[j] = t[j] * 3 / 4; } } ",
        "let mut h: i32 = 0; ",
        "for k in 0..n { o[k] = u[k] / 3 + t[k] + h; h = (h * 5 + 1) % 7; } }\n",
        // Reads one element ahead of the writer: must stay unfused.
        "fn ahead(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut o: [i32; 32], n: i32) { ",
        "for i in 0..n { if a[i] > 0 { t[i] = a[i] / 3; } else { t[i] = 0 - a[i] / 5; } } ",
        "let mut h: i32 = 0; ",
        "for j in 0..n { o[j] = t[j + 1] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
    );

    /// From source, after the pre-fusion pipeline: pairs whose bodies branch fuse exactly when the
    /// two-block rules say they may -- a diamond in either loop, diamonds in both (and the fused
    /// region fuses again with a third loop), `@noalias` slices -- and not a read ahead of the
    /// writer, a `break` (a second exit), or slices without the promise.
    #[test]
    fn source_multi_block_bodies_fuse_exactly_when_legal() {
        let src = format!(
            "{BRANCHING_KERNELS}{}",
            concat!(
                "fn brk(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut o: [i32; 32], ",
                "n: i32) { ",
                "for i in 0..n { if a[i] > 0 { t[i] = a[i] / 3; } else { break; } } ",
                "let mut h: i32 = 0; ",
                "for j in 0..n { o[j] = t[j] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
                "@unchecked\nfn slices(@noalias mut tmp: []i32, @noalias mut out: []i32, ",
                "src: []i32, n: i64) { ",
                "for i in 0..n { if src[i] > 0 { tmp[i] = src[i] / 2; } ",
                "else { tmp[i] = 0 - src[i] / 3; } } ",
                "let mut h: i32 = 0; ",
                "for j in 0..n { out[j] = tmp[j] / 3 + src[j] + h; h = (h * 5 + 1) % 7; } }\n",
                "@unchecked\nfn aliased(mut tmp: []i32, mut out: []i32, src: []i32, n: i64) { ",
                "for i in 0..n { if src[i] > 0 { tmp[i] = src[i] / 2; } ",
                "else { tmp[i] = 0 - src[i] / 3; } } ",
                "let mut h: i32 = 0; ",
                "for j in 0..n { out[j] = tmp[j] / 3 + src[j] + h; h = (h * 5 + 1) % 7; } }\n",
                "fn main() -> i32 { return 0; }\n",
            )
        );
        let (mut program, mut interner) = before_fuse(&src);
        // (name, loops before, loops after)
        for (name, before, after) in [
            ("first", 2, 1),
            ("second", 2, 1),
            ("both", 3, 1),
            ("ahead", 2, 2),
            ("brk", 2, 2),
            ("slices", 2, 1),
            ("aliased", 2, 2),
        ] {
            let sym = interner.intern(name);
            let f = program
                .funcs
                .iter_mut()
                .find(|f| f.name == sym)
                .expect("defined above");
            let forest = crate::loop_info::analyze_function(f);
            assert_eq!(forest.loops.len(), before, "{name}");
            // Each case has a body that branches: the two-block form alone would decline it.
            assert!(
                forest.loops.iter().any(|l| l.blocks.len() > 2),
                "{name}: no branching body\n{}",
                wukong_mir::print::print_function(f, &interner)
            );
            // Neither loop of a pair is one the vectorizer widens: legality alone decides.
            assert!(
                forest
                    .loops
                    .iter()
                    .all(|l| !crate::vectorize::would_widen(f, l)),
                "{name}"
            );
            let fused = LoopFuse.run_function(f, &mut CfgAnalyses::default());
            assert_eq!(
                fused,
                after < before,
                "{name}\n{}",
                wukong_mir::print::print_function(f, &interner)
            );
            assert!(wukong_mir::verify::verify_function(f).is_empty(), "{name}");
            assert_eq!(
                loop_count(f),
                after,
                "{name}\n{}",
                wukong_mir::print::print_function(f, &interner)
            );
        }
    }

    /// Two division-free `if`/`else` store loops each widen alone; fused, their region would sink
    /// stores to two addresses, which the if-converters refuse -- so they must stay apart. A
    /// widenable straight loop beside a diamond fuses into a body the vectorizer still flattens,
    /// so those do fuse. In every case the pass's answer is the fused copy's own verdict, and a
    /// fused loop always widens.
    #[test]
    fn multi_block_fusion_never_costs_a_widened_loop() {
        let src = concat!(
            "fn two_diamonds(a: [f32; 64], @noalias mut t: [f32; 64], @noalias mut o: [f32; 64], ",
            "n: i32) { ",
            "for i in 0..n { if a[i] > 0.0 { t[i] = a[i] * 2.0; } else { t[i] = a[i] * 0.5; } } ",
            "for j in 0..n { if a[j] > 1.0 { o[j] = a[j] + 1.0; } else { o[j] = a[j] - 1.0; } } }\n",
            "fn diamond_then_div(a: [i32; 64], @noalias mut t: [i32; 64], ",
            "@noalias mut o: [i32; 64], n: i32) { ",
            "for i in 0..n { if a[i] > 0 { t[i] = a[i] * 2; } else { t[i] = 0 - a[i]; } } ",
            "for j in 0..n { o[j] = t[j] / 3 + a[j]; } }\n",
            "fn div_then_diamond(a: [i32; 64], @noalias mut t: [i32; 64], ",
            "@noalias mut o: [i32; 64], n: i32) { ",
            "for i in 0..n { t[i] = a[i] / 3 + 1; } ",
            "for j in 0..n { if t[j] > 5 { o[j] = t[j] + 1; } else { o[j] = a[j] - 1; } } }\n",
            "fn main() -> i32 { return 0; }\n",
        );
        let (mut program, mut interner) = before_fuse(src);
        for (name, fuses) in [
            ("two_diamonds", false),
            ("diamond_then_div", true),
            ("div_then_diamond", true),
        ] {
            let sym = interner.intern(name);
            let f = program
                .funcs
                .iter_mut()
                .find(|f| f.name == sym)
                .expect("defined above");
            let forest = crate::loop_info::analyze_function(f);
            assert_eq!(forest.loops.len(), 2, "{name}");
            assert!(
                forest
                    .loops
                    .iter()
                    .all(|l| crate::vectorize::would_widen(f, l)),
                "{name}: each loop widens alone"
            );
            // What fusing regardless would leave behind.
            let (plan, cond) = forced_plan(f);
            assert_eq!(fused_loop_widens(f, &plan, cond), fuses, "{name}");
            let fused = LoopFuse.run_function(f, &mut CfgAnalyses::default());
            assert_eq!(fused, fuses, "{name}");
            assert!(wukong_mir::verify::verify_function(f).is_empty(), "{name}");
            let after = crate::loop_info::analyze_function(f);
            if fused {
                assert_eq!(after.loops.len(), 1, "{name}");
                assert!(
                    crate::vectorize::would_widen(f, &after.loops[0]),
                    "{name}: the fused loop must still widen\n{}",
                    wukong_mir::print::print_function(f, &interner)
                );
            } else {
                assert_eq!(after.loops.len(), 2, "{name}");
            }
        }
    }

    /// Two widenable shapes whose fused region the vectorizer still flattens (see
    /// [`multi_block_fusion_never_costs_a_widened_loop`]): a diamond, then a straight loop that
    /// divides; a straight loop that divides, then a diamond.
    const WIDENED_KERNELS: &str = concat!(
        "fn dthen(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut o: [i32; 32], n: i32) { ",
        "for i in 0..n { if a[i] > 0 { t[i] = a[i] * 2; } else { t[i] = 0 - a[i]; } } ",
        "for j in 0..n { o[j] = t[j] / 3 + a[j]; } }\n",
        "fn thend(a: [i32; 32], @noalias mut t: [i32; 33], @noalias mut o: [i32; 32], n: i32) { ",
        "for i in 0..n { t[i] = a[i] / 3 + 1; } ",
        "for j in 0..n { if t[j] > 1 { o[j] = t[j] + 1; } else { o[j] = a[j] - 1; } } }\n",
    );

    /// `-O0` against `-O2` through the whole optimizer and the interpreter, over the fused shapes
    /// and the declining one, for several trip counts (0 and 1 included). Each kernel stays out of
    /// line (it and `again` call each other, so the inliner never splices it into `main`, where its
    /// parameters would lose their promise), and the `-O2` kernels are checked to be fused -- one
    /// loop, or for the widened shapes one vector loop and its scalar tail -- so the comparison is
    /// not vacuous.
    #[test]
    fn fused_branching_bodies_agree_with_the_unfused_program() {
        let kernels = format!("{BRANCHING_KERNELS}{WIDENED_KERNELS}").replace(
            ", n: i32) { ",
            ", n: i32, d: i64) { if d > 0 { again(d - 1); return; } ",
        );
        let src = format!(
            "{kernels}{}",
            concat!(
                "fn again(d: i64) { let a: [i32; 32] = [0; 32]; let mut t: [i32; 33] = [0; 33]; ",
                "let mut u: [i32; 32] = [0; 32]; let mut o: [i32; 32] = [0; 32]; ",
                "first(a, t, o, 0, d); second(a, t, o, 0, d); both(a, t, u, o, 0, d); ",
                "ahead(a, t, o, 0, d); dthen(a, t, o, 0, d); thend(a, t, o, 0, d); }\n",
                "fn show(t: [i32; 33], u: [i32; 32], o: [i32; 32]) { ",
                "for k in 0..33 { print(t[k]); } for k in 0..32 { print(u[k]); print(o[k]); } }\n",
                "fn reset(mut t: [i32; 33], mut u: [i32; 32], mut o: [i32; 32]) { ",
                "for k in 0..33 { t[k] = 1000 + k; } ",
                "for k in 0..32 { u[k] = 0 - 1; o[k] = 0 - 2; } }\n",
                "fn main() -> i32 { let mut a: [i32; 32] = [0; 32]; ",
                "for k in 0..32 { a[k] = (k * 7) % 23 - 11; } ",
                "let mut t: [i32; 33] = [0; 33]; let mut u: [i32; 32] = [0; 32]; ",
                "let mut o: [i32; 32] = [0; 32]; ",
                "let ns: [i32; 6] = [0, 1, 2, 5, 7, 32]; ",
                "for q in 0..6 { let n = ns[q]; ",
                "reset(t, u, o); first(a, t, o, n, 0); show(t, u, o); ",
                "reset(t, u, o); second(a, t, o, n, 0); show(t, u, o); ",
                "reset(t, u, o); both(a, t, u, o, n, 0); show(t, u, o); ",
                "reset(t, u, o); ahead(a, t, o, n, 0); show(t, u, o); ",
                "reset(t, u, o); dthen(a, t, o, n, 0); show(t, u, o); ",
                "reset(t, u, o); thend(a, t, o, n, 0); show(t, u, o); } ",
                "return 0; }\n",
            )
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let main = interner.intern("main");
        let mut outputs = Vec::new();
        for level in [0u8, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            if level == 2 {
                // How many loops each kernel keeps: one fused loop; for a widened fused loop its
                // vector body and its scalar tail; the declining `ahead` keeps both loops (and its
                // two-block second loop is partially unrolled, with a remainder loop).
                for (name, ok, vector) in [
                    ("first", 1..=1, false),
                    ("second", 1..=1, false),
                    ("both", 1..=1, false),
                    ("ahead", 2..=usize::MAX, false),
                    ("dthen", 2..=2, true),
                    ("thend", 2..=2, true),
                ] {
                    let f = optimized
                        .function(interner.intern(name))
                        .expect("kept out of line");
                    let loops = loop_count(f);
                    let widened = f.value_types.iter().any(MirType::is_vector);
                    assert!(
                        ok.contains(&loops) && widened == vector,
                        "{name} at -O2: {loops} loops, widened {widened}\n{}",
                        wukong_mir::print::print_function(f, &interner)
                    );
                }
            }
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            outputs.push(stdout);
        }
        assert!(!outputs[0].is_empty());
        assert_eq!(outputs[0], outputs[1]);
    }

    /// A branching first loop beside a recurrence loop no vectorizer widens. Each arm stores to an
    /// array of its own, so the branching loop does not widen as it stands either ("both arms
    /// store, to different places") -- but `split`, `flag` and `derived` branch on a condition
    /// `unswitch` decides later in the sweep: the counter against a parameter, a parameter flag
    /// (compared inside the loop until LICM hoists it), the counter's successor against a
    /// parameter. Split or unswitched alone, the loop becomes the copies `t[i] = a[i] * 3` and
    /// `u[i] = a[i] * 5 - 1`, which widen. (Arms that divide would not show it: `t[i] = a[i] / 3`
    /// alone is declined, "most of the body would run lane by lane".) `data` branches on a loaded
    /// value instead, which nothing decides.
    const DECIDABLE_KERNELS: &str = concat!(
        "fn split(a: [i32; 32], @noalias mut t: [i32; 32], @noalias mut u: [i32; 32], ",
        "@noalias mut o: [i32; 32], n: i32, m: i32) { ",
        "for i in 0..n { if i < m { t[i] = a[i] * 3; } else { u[i] = a[i] * 5 - 1; } } ",
        "let mut h: i32 = 0; ",
        "for j in 0..n { o[j] = t[j] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
        "fn flag(a: [i32; 32], @noalias mut t: [i32; 32], @noalias mut u: [i32; 32], ",
        "@noalias mut o: [i32; 32], n: i32, m: i32) { ",
        "for i in 0..n { if m > 0 { t[i] = a[i] * 3; } else { u[i] = a[i] * 5 - 1; } } ",
        "let mut h: i32 = 0; ",
        "for j in 0..n { o[j] = t[j] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
        "fn derived(a: [i32; 32], @noalias mut t: [i32; 32], @noalias mut u: [i32; 32], ",
        "@noalias mut o: [i32; 32], n: i32, m: i32) { ",
        "for i in 0..n { if i + 1 < m { t[i] = a[i] * 3; } else { u[i] = a[i] * 5 - 1; } } ",
        "let mut h: i32 = 0; ",
        "for j in 0..n { o[j] = t[j] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
        "fn data(a: [i32; 32], @noalias mut t: [i32; 32], @noalias mut u: [i32; 32], ",
        "@noalias mut o: [i32; 32], n: i32, m: i32) { ",
        "for i in 0..n { if a[i] < m { t[i] = a[i] * 3; } else { u[i] = a[i] * 5 - 1; } } ",
        "let mut h: i32 = 0; ",
        "for j in 0..n { o[j] = t[j] / 2 + h; h = (h * 5 + 1) % 7; } }\n",
    );

    /// Two loops neither of which widens may still cost a loop its vector body: a branching one
    /// that `unswitch` splits or unswitches later in the sweep becomes copies the vectorizer widens,
    /// unless fusion has already put the other loop's unwidenable work into every copy. So such a
    /// pair stays apart, while the same pair branching on a loaded value -- which nothing decides --
    /// still fuses (the accesses are identical across the four kernels, so legality is not what
    /// decides).
    #[test]
    fn a_branch_unswitch_would_decide_keeps_its_loop_out_of_an_unwidenable_fusion() {
        let src = format!("{DECIDABLE_KERNELS}fn main() -> i32 {{ return 0; }}\n");
        let (mut program, mut interner) = before_fuse(&src);
        for (name, fuses) in [
            ("split", false),
            ("flag", false),
            ("derived", false),
            ("data", true),
        ] {
            let sym = interner.intern(name);
            let f = program
                .funcs
                .iter_mut()
                .find(|f| f.name == sym)
                .expect("defined above");
            let forest = crate::loop_info::analyze_function(f);
            assert_eq!(forest.loops.len(), 2, "{name}");
            assert!(
                forest.loops.iter().any(|l| l.blocks.len() > 2),
                "{name}: no branching body"
            );
            assert!(
                forest
                    .loops
                    .iter()
                    .all(|l| !crate::vectorize::would_widen(f, l)),
                "{name}: neither loop widens as fusion sees it"
            );
            let fused = LoopFuse.run_function(f, &mut CfgAnalyses::default());
            assert_eq!(
                fused,
                fuses,
                "{name}\n{}",
                wukong_mir::print::print_function(f, &interner)
            );
            assert!(wukong_mir::verify::verify_function(f).is_empty(), "{name}");
            assert_eq!(loop_count(f), if fuses { 1 } else { 2 }, "{name}");
        }
    }

    /// The same kernels through the whole `-O2` pipeline, kept out of line: the decidable ones
    /// end up with a vector body (their split or unswitched copies widen), and every one agrees
    /// with `-O0` on every lane of `t`, `u` and `o`, over trip counts and split points on both
    /// sides of every edge (0 and 1 iterations, `m` below, inside and past the range).
    #[test]
    fn a_loop_unswitch_would_decide_keeps_its_vector_body() {
        let kernels = DECIDABLE_KERNELS.replace(
            ", n: i32, m: i32) { ",
            ", n: i32, m: i32, d: i64) { if d > 0 { again(d - 1); return; } ",
        );
        let src = format!(
            "{kernels}{}",
            concat!(
                "fn again(d: i64) { let a: [i32; 32] = [0; 32]; let mut t: [i32; 32] = [0; 32]; ",
                "let mut u: [i32; 32] = [0; 32]; let mut o: [i32; 32] = [0; 32]; ",
                "split(a, t, u, o, 0, 0, d); flag(a, t, u, o, 0, 0, d); ",
                "derived(a, t, u, o, 0, 0, d); data(a, t, u, o, 0, 0, d); }\n",
                "fn show(t: [i32; 32], u: [i32; 32], o: [i32; 32]) { ",
                "for k in 0..32 { print(t[k]); print(u[k]); print(o[k]); } }\n",
                "fn reset(mut t: [i32; 32], mut u: [i32; 32], mut o: [i32; 32]) { ",
                "for k in 0..32 { t[k] = 1000 + k; u[k] = 0 - 1; o[k] = 0 - 2; } }\n",
                "fn main() -> i32 { let mut a: [i32; 32] = [0; 32]; ",
                "for k in 0..32 { a[k] = (k * 7) % 23 - 11; } ",
                "let mut t: [i32; 32] = [0; 32]; let mut u: [i32; 32] = [0; 32]; ",
                "let mut o: [i32; 32] = [0; 32]; ",
                "let ns: [i32; 4] = [0, 1, 7, 32]; let ms: [i32; 5] = [0 - 3, 0, 1, 5, 40]; ",
                "for q in 0..4 { for r in 0..5 { let n = ns[q]; let m = ms[r]; ",
                "reset(t, u, o); split(a, t, u, o, n, m, 0); show(t, u, o); ",
                "reset(t, u, o); flag(a, t, u, o, n, m, 0); show(t, u, o); ",
                "reset(t, u, o); derived(a, t, u, o, n, m, 0); show(t, u, o); ",
                "reset(t, u, o); data(a, t, u, o, n, m, 0); show(t, u, o); } } ",
                "return 0; }\n",
            )
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let main = interner.intern("main");
        let mut outputs = Vec::new();
        for level in [0u8, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            if level == 2 {
                for (name, vector) in [
                    ("split", true),
                    ("flag", true),
                    ("derived", true),
                    ("data", false),
                ] {
                    let f = optimized
                        .function(interner.intern(name))
                        .expect("kept out of line");
                    let widened = f.value_types.iter().any(MirType::is_vector);
                    assert_eq!(
                        widened,
                        vector,
                        "{name} at -O2\n{}",
                        wukong_mir::print::print_function(f, &interner)
                    );
                }
            }
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            outputs.push(stdout);
        }
        assert!(!outputs[0].is_empty());
        assert_eq!(outputs[0], outputs[1]);
    }
}
