//! `mem2reg` — promote scalar stack slots to SSA registers.
//!
//! The front-end gives every local an `alloca` and accesses it with `load`/`store`. That keeps
//! lowering trivial but blinds every value-based optimization: two reads of the same variable are
//! distinct `load` results, a constant assigned to a variable is hidden behind memory, and a loop
//! counter never becomes a recognizable induction value. Promoting those slots to SSA is the
//! single highest-leverage transform in the pipeline — it is what lets constant propagation, CSE,
//! and DCE actually fire, and it removes a load/store per access at run time.
//!
//! Algorithm: classic SSA construction (Cytron et al.). A slot is *promotable* if every use of its
//! pointer is the address operand of a `load` or `store` (never escaping via `gep`, a call, a
//! returned/stored pointer, etc.). For each promotable slot we place block parameters ("phis") at
//! the iterated dominance frontier of its defining blocks, then rename by a dominator-tree walk
//! that keeps a per-slot stack of the reaching definition. Reads become the reaching value, writes
//! update it, and each CFG edge is given the arguments its destination's phis expect.
//!
//! Integer, float and pointer slots are promoted. A slot read before any write on some path becomes
//! a zero constant of its type, materialized once per type at the top of the entry block, which is
//! what the interpreter's zero-initialized memory would have yielded. A pointer slot's zero is the
//! null pointer, `inttoptr 0`: the interpreter reserves slot 0 as a never-allocated null
//! (`wukong_interp::NULL_SLOT`), so its zero-initialized pointer memory reads as exactly the null
//! native's `inttoptr 0` is. (Before that reservation slot 0 was the first live allocation of a run,
//! and only the definitely-initialized subset of pointer slots could be promoted; a pointer local
//! whose initializing store sat behind a branch kept its slot and was re-loaded at every use.) Array
//! and vector slots stay in memory.

use crate::fxhash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};

use wukong_mir::{CastKind, Function, Inst, MirType, Op, Terminator, ValueId};

use crate::{cfg, each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

pub struct Mem2Reg;

impl Pass for Mem2Reg {
    fn name(&self) -> &'static str {
        "mem2reg"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // Dominance is only defined on reachable blocks. Pruning renumbers blocks, so any cached
        // analysis is stale afterwards. When the CFG is already fully reachable (the steady state)
        // pruning is a no-op, so skip its reachability DFS — `all_reachable` is free off the cached
        // rpo that `idoms` needs anyway.
        let pruned = !cache.all_reachable(f);
        if pruned {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        let promotable = find_promotable(f);
        if promotable.is_empty() {
            return pruned;
        }
        promote(f, &promotable, cache);
        true
    }
}

/// A slot type is promotable to a register if it is a scalar integer, float or pointer. Arrays and
/// vectors are left in memory.
fn is_promotable_ty(ty: &MirType) -> bool {
    ty.is_int() || ty.is_float() || matches!(ty, MirType::Ptr)
}

/// Find allocas whose pointer is used *only* as the address of `load`/`store`. Returns a map from
/// the alloca's result value id to the slot's element type.
fn find_promotable(f: &Function) -> BTreeMap<u32, MirType> {
    let mut cand: BTreeMap<u32, MirType> = BTreeMap::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::Alloca { ty, .. }) = (inst.result, &inst.op) {
                if is_promotable_ty(ty) {
                    cand.insert(r.0, ty.clone());
                }
            }
        }
    }
    if cand.is_empty() {
        return cand;
    }

    // Any appearance other than `load <slot>` / `store _, <slot>` means the address escapes.
    // A load or store whose *type* disagrees with the slot's own is a type-punned access: the value
    // that flows through the promoted register would then have the wrong MIR type, which the
    // verifier rejects (and, unverified, is a silent reinterpret). Leave those in memory.
    let mut bad: FxHashSet<u32> = FxHashSet::default();
    for b in &f.blocks {
        for inst in &b.insts {
            match &inst.op {
                // The pointer operand of a load is fine; a load has no other operands.
                Op::Load(p, ty) => {
                    if cand.get(&p.0).is_some_and(|slot| slot != ty) {
                        bad.insert(p.0);
                    }
                }
                // The pointer operand of a store is fine, but the stored *value* being the slot
                // pointer means the address escapes.
                Op::Store { ptr, value } => {
                    if cand.contains_key(&value.0) {
                        bad.insert(value.0);
                    }
                    if let Some(slot) = cand.get(&ptr.0) {
                        if f.value_types.get(value.0 as usize) != Some(slot) {
                            bad.insert(ptr.0);
                        }
                    }
                }
                other => each_op_use(other, &mut |v| {
                    if cand.contains_key(&v.0) {
                        bad.insert(v.0);
                    }
                }),
            }
        }
        each_term_use(&b.term, &mut |v| {
            if cand.contains_key(&v.0) {
                bad.insert(v.0);
            }
        });
    }
    for x in bad {
        cand.remove(&x);
    }
    cand
}

fn promote(f: &mut Function, promotable: &BTreeMap<u32, MirType>, cache: &mut CfgAnalyses) {
    // The iterated dominance frontier drives phi placement; the dominator-tree children drive the
    // rename walk. Both come from the shared cache (borrow `cache`, not `f`, so the phi/rename edits
    // to `f` below are unaffected).
    let (df, children) = cache.df_and_children(f);

    // Blocks that contain a store to each promotable slot.
    let mut def_blocks: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Op::Store { ptr, .. } = &inst.op {
                if promotable.contains_key(&ptr.0) {
                    def_blocks.entry(ptr.0).or_default().insert(b.id.0);
                }
            }
        }
    }

    // --- Phase 1: place phis (block parameters) at the iterated dominance frontier. ---
    // block id -> ordered list of (phi value id, slot id); the order matches the params we append.
    let mut block_phis: BTreeMap<u32, Vec<(ValueId, u32)>> = BTreeMap::new();
    for (&var, defs) in &def_blocks {
        let ty = promotable[&var].clone();
        let mut worklist: Vec<u32> = defs.iter().copied().collect();
        let mut on_list: FxHashSet<u32> = defs.iter().copied().collect();
        let mut has_phi: FxHashSet<u32> = FxHashSet::default();
        while let Some(b) = worklist.pop() {
            for &d in &df[b as usize] {
                if has_phi.insert(d) {
                    let pv = ValueId(f.value_types.len() as u32);
                    f.value_types.push(ty.clone());
                    f.blocks[d as usize].params.push(pv);
                    block_phis.entry(d).or_default().push((pv, var));
                    if on_list.insert(d) {
                        worklist.push(d);
                    }
                }
            }
        }
    }

    // --- Phase 2: rename (collect edits without mutating `f`). ---
    let mut r = Rename {
        promotable,
        block_phis: &block_phis,
        children,
        stack: FxHashMap::default(),
        replace: FxHashMap::default(),
        delete: FxHashSet::default(),
        append_then: FxHashMap::default(),
        append_else: FxHashMap::default(),
        append_br: FxHashMap::default(),
        append_switch: FxHashMap::default(),
        zero_for: FxHashMap::default(),
        new_consts: Vec::new(),
        next_value: f.value_types.len() as u32,
    };
    r.visit(f, f.entry.0);
    let append_switch = std::mem::take(&mut r.append_switch);
    let Rename {
        replace,
        delete,
        append_then,
        append_else,
        append_br,
        zero_for,
        new_consts,
        ..
    } = r;

    // --- Phase 3: apply edits to `f`. ---
    // (a) Append phi arguments to each edge. Independent of instruction indices, so do it first.
    for b in &mut f.blocks {
        match &mut b.term {
            Terminator::Br { args, .. } => {
                if let Some(extra) = append_br.get(&b.id.0) {
                    args.extend(extra.iter().copied());
                }
            }
            Terminator::CondBr {
                then_args,
                else_args,
                ..
            } => {
                if let Some(extra) = append_then.get(&b.id.0) {
                    then_args.extend(extra.iter().copied());
                }
                if let Some(extra) = append_else.get(&b.id.0) {
                    else_args.extend(extra.iter().copied());
                }
            }
            Terminator::Switch {
                cases,
                default_args,
                ..
            } => {
                // One list per case, then one for the default (see `Rename::append_switch`).
                if let Some(extra) = append_switch.get(&b.id.0) {
                    for (c, more) in cases.iter_mut().zip(extra) {
                        c.args.extend(more.iter().copied());
                    }
                    if let Some(more) = extra.get(cases.len()) {
                        default_args.extend(more.iter().copied());
                    }
                }
            }
            _ => {}
        }
    }

    // (b) Delete promoted loads/stores (by original index) and the promoted allocas.
    for (bi, b) in f.blocks.iter_mut().enumerate() {
        let mut idx = 0u32;
        b.insts.retain(|inst| {
            let i = idx;
            idx += 1;
            if delete.contains(&(bi as u32, i)) {
                return false;
            }
            if let (Some(res), Op::Alloca { .. }) = (inst.result, &inst.op) {
                if promotable.contains_key(&res.0) {
                    return false;
                }
            }
            true
        });
    }

    // (c) Materialize any zero constants for read-before-write slots, at the top of the entry block.
    if !new_consts.is_empty() {
        let mut const_insts: Vec<Inst> = Vec::with_capacity(new_consts.len());
        for (id, ty) in &new_consts {
            debug_assert_eq!(
                f.value_types.len() as u32,
                *id,
                "zero-const ids are sequential"
            );
            f.value_types.push(ty.clone());
            let op = if ty.is_float() {
                Op::ConstFloat(0.0, ty.clone())
            } else if *ty == MirType::Ptr {
                // `zero_of` minted the `i64` zero before this null, so it is already emitted above.
                Op::Cast(CastKind::IntToPtr, zero_for[&MirType::I64], MirType::Ptr)
            } else {
                Op::ConstInt(0, ty.clone())
            };
            const_insts.push(Inst {
                result: Some(ValueId(*id)),
                op,
            });
        }
        let entry = f.entry.0 as usize;
        const_insts.append(&mut f.blocks[entry].insts);
        f.blocks[entry].insts = const_insts;
    }

    // (d) Rewrite every remaining use of a deleted load through its reaching definition.
    if !replace.is_empty() {
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                map_op_uses(&mut inst.op, |v| resolve(&replace, v));
            }
            map_term_uses(&mut b.term, |v| resolve(&replace, v));
        }
    }
}

/// Per-walk renamer state. Collects edits; `f` is only read during the walk.
struct Rename<'a> {
    promotable: &'a BTreeMap<u32, MirType>,
    block_phis: &'a BTreeMap<u32, Vec<(ValueId, u32)>>,
    children: &'a [Vec<u32>],
    /// slot id -> stack of reaching definitions (innermost last).
    stack: FxHashMap<u32, Vec<ValueId>>,
    /// deleted load result -> reaching definition.
    replace: FxHashMap<u32, ValueId>,
    /// (block, original inst index) of loads/stores to delete.
    delete: FxHashSet<(u32, u32)>,
    /// values to append to each edge's argument list, keyed by source block.
    append_then: FxHashMap<u32, Vec<ValueId>>,
    append_else: FxHashMap<u32, Vec<ValueId>>,
    append_br: FxHashMap<u32, Vec<ValueId>>,
    /// a `switch`'s edges, keyed by source block: one list per case in order, then the default's.
    append_switch: FxHashMap<u32, Vec<Vec<ValueId>>>,
    /// cached zero constant per type (for read-before-write).
    zero_for: FxHashMap<MirType, ValueId>,
    /// freshly allocated (value id, type) zero constants to materialize later.
    new_consts: Vec<(u32, MirType)>,
    next_value: u32,
}

impl Rename<'_> {
    /// The current reaching definition of a slot, or a freshly minted zero constant if the slot is
    /// read before any write on this path (matching the interpreter's zero-initialized memory).
    fn current_def(&mut self, var: u32) -> ValueId {
        if let Some(top) = self.stack.get(&var).and_then(|s| s.last()) {
            return *top;
        }
        let ty = self.promotable[&var].clone();
        self.zero_of(ty)
    }

    /// The cached zero of `ty`, minted on first use. A pointer's zero is the null pointer,
    /// `inttoptr (i64 0)`, which is what the interpreter's zero-initialized pointer memory holds
    /// now that it reserves slot 0 as null. Its `i64` operand is minted *first*, so it precedes the
    /// cast in `new_consts` — the order the entry-block materialization emits them in.
    fn zero_of(&mut self, ty: MirType) -> ValueId {
        if let Some(z) = self.zero_for.get(&ty) {
            return *z;
        }
        if ty == MirType::Ptr {
            self.zero_of(MirType::I64);
        }
        let id = ValueId(self.next_value);
        self.next_value += 1;
        self.new_consts.push((id.0, ty.clone()));
        self.zero_for.insert(ty, id);
        id
    }

    fn phi_args_for(&mut self, target: u32) -> Vec<ValueId> {
        let Some(phis) = self.block_phis.get(&target) else {
            return Vec::new();
        };
        let vars: Vec<u32> = phis.iter().map(|&(_, var)| var).collect();
        vars.into_iter().map(|var| self.current_def(var)).collect()
    }

    fn visit(&mut self, f: &Function, entry: u32) {
        // Keep one undo log for definitions introduced along the active dominator path. An
        // explicit DFS stack avoids using the compiler's call stack for arbitrary CFG depth.
        let mut pushed: Vec<u32> = Vec::new();
        self.enter(f, entry, &mut pushed);
        let mut scopes = vec![(entry, 0usize, 0usize)];
        while let Some((block, next_child, pushed_start)) = scopes.last_mut() {
            if let Some(&child) = self.children[*block as usize].get(*next_child) {
                *next_child += 1;
                let start = pushed.len();
                self.enter(f, child, &mut pushed);
                scopes.push((child, 0, start));
            } else {
                // Reverse insertion order preserves the reaching-definition stack for slots
                // written more than once in a block, then exposes the parent's definitions.
                for var in pushed.drain(*pushed_start..).rev() {
                    if let Some(s) = self.stack.get_mut(&var) {
                        s.pop();
                    }
                }
                scopes.pop();
            }
        }
    }

    fn enter(&mut self, f: &Function, block: u32, pushed: &mut Vec<u32>) {
        // (a) Phi definitions provided by this block.
        if let Some(phis) = self.block_phis.get(&block) {
            for &(pv, var) in phis {
                self.stack.entry(var).or_default().push(pv);
                pushed.push(var);
            }
        }

        // (b) Instructions: forward loads to the reaching def, record stores as new defs.
        let bb = &f.blocks[block as usize];
        for (ii, inst) in bb.insts.iter().enumerate() {
            match &inst.op {
                Op::Load(ptr, _) if self.promotable.contains_key(&ptr.0) => {
                    let cur = self.current_def(ptr.0);
                    if let Some(res) = inst.result {
                        self.replace.insert(res.0, cur);
                    }
                    self.delete.insert((block, ii as u32));
                }
                Op::Store { ptr, value } if self.promotable.contains_key(&ptr.0) => {
                    self.stack.entry(ptr.0).or_default().push(*value);
                    pushed.push(ptr.0);
                    self.delete.insert((block, ii as u32));
                }
                _ => {}
            }
        }

        // (c) Terminator: hand each successor the arguments its phis expect.
        match &bb.term {
            Terminator::Br { target, .. } => {
                let v = self.phi_args_for(target.0);
                if !v.is_empty() {
                    self.append_br.insert(block, v);
                }
            }
            Terminator::CondBr {
                then_blk, else_blk, ..
            } => {
                let vt = self.phi_args_for(then_blk.0);
                if !vt.is_empty() {
                    self.append_then.insert(block, vt);
                }
                let ve = self.phi_args_for(else_blk.0);
                if !ve.is_empty() {
                    self.append_else.insert(block, ve);
                }
            }
            Terminator::Switch { cases, default, .. } => {
                let mut lists: Vec<Vec<ValueId>> = Vec::with_capacity(cases.len() + 1);
                for c in cases {
                    lists.push(self.phi_args_for(c.target.0));
                }
                lists.push(self.phi_args_for(default.0));
                if lists.iter().any(|l| !l.is_empty()) {
                    self.append_switch.insert(block, lists);
                }
            }
            _ => {}
        }
    }
}

/// Follow `load -> reaching def` chains to a fixed point.
fn resolve(map: &FxHashMap<u32, ValueId>, v: ValueId) -> ValueId {
    let mut cur = v;
    let mut guard = 0;
    while let Some(&next) = map.get(&cur.0) {
        if next == cur || guard > 100_000 {
            break;
        }
        cur = next;
        guard += 1;
    }
    cur
}
