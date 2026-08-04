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
//! Integer, float and pointer slots are promoted. An integer/float slot read before any write on
//! some path becomes a zero constant of its type, materialized once per type at the top of the entry
//! block, which is what the interpreter's zero-initialized memory would have yielded. A pointer slot
//! has no such natural zero, so only the definitely-initialized subset is promoted
//! ([`initialized_ptr_slots`]) — which is exactly the case that matters, since `mir_build` gives every
//! pointer-typed parameter (`*T`, `&T`, `Tensor[…]`) an `alloca ptr` + `store` in the entry block and
//! then re-`load`s the base pointer at *every* element access. Array and vector slots stay in memory.

use std::collections::{BTreeMap, BTreeSet};
use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{Function, Inst, MirType, Op, Terminator, ValueId};

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
        let mut promotable = find_promotable(f);
        // A pointer slot additionally has to be provably written before it is read — see
        // `initialized_ptr_slots`. That test needs dominance frontiers, so only pay for them when
        // some candidate is actually a pointer (`promote` would compute them anyway if we proceed,
        // and the cache hands back the same result).
        if promotable.values().any(|ty| matches!(ty, MirType::Ptr)) {
            let init = {
                let (df, _) = cache.df_and_children(f);
                initialized_ptr_slots(f, &promotable, df)
            };
            promotable.retain(|slot, ty| !matches!(ty, MirType::Ptr) || init.contains(slot));
        }
        if promotable.is_empty() {
            return pruned;
        }
        promote(f, &promotable, cache);
        true
    }
}

/// A slot type is promotable to a register if it is a scalar integer or float. Arrays and vectors
/// are left in memory. Pointers are handled too, but only for the definitely-initialized subset —
/// see [`initialized_ptr_slots`] for why they need the extra condition.
fn is_promotable_ty(ty: &MirType) -> bool {
    ty.is_int() || ty.is_float() || matches!(ty, MirType::Ptr)
}

/// Find allocas whose pointer is used *only* as the address of `load`/`store`. Returns a map from
/// the alloca's result value id to the slot's element type.
fn find_promotable(f: &Function) -> BTreeMap<u32, MirType> {
    let mut cand: BTreeMap<u32, MirType> = BTreeMap::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::Alloca(ty)) = (inst.result, &inst.op) {
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

/// The pointer slots that are provably **written before they are read**.
///
/// A read-before-write slot is promoted to a zero constant of the slot's type — what the
/// interpreter's zero-initialized memory would have yielded. There is no such constant for a
/// pointer: `inttoptr 0` is a genuine null on the native backend but a *valid, addressable* slot in
/// the interpreter (interp address 0 is a real address — the standing cross-cutting landmine), so
/// synthesizing one would trade a stack slot for an interp-vs-native divergence. Rather than reason
/// about undef we simply refuse any pointer slot that could be read first.
///
/// A slot qualifies when the **first access to it inside its own alloca's block `B` is a store**,
/// and `B` is either the entry block or has an empty dominance frontier. Why that is sufficient:
///
/// * SSA dominance makes `B` dominate every block that mentions the slot at all (a use must be
///   dominated by the `alloca` that defines it), and a block is straight-line, so every access
///   outside `B` happens after all of `B` — hence after the store.
/// * `DF(B) = ∅` means the set `S` of blocks `B` dominates is closed under *both* successors (a
///   successor of a dominated block is dominated, else it would be in `DF(B)`) and predecessors (a
///   predecessor outside `S` would give a path to a dominated block that bypasses `B`). So the
///   iterated dominance frontier of the store blocks, *and every predecessor of every block that
///   receives a phi*, lie inside `S` — which is exactly the set of program points at which the
///   renamer asks for a reaching definition. The stack is therefore never empty.
/// * The entry block is admitted directly: it dominates everything, so the same argument holds even
///   when a back edge puts the entry itself in its own frontier.
///
/// This admits the case the pass exists for — `mir_build` gives every pointer-typed parameter
/// (`*T`, `&T`, and every `Tensor[…]`) an `alloca ptr` immediately followed by `store <param>` in
/// the entry block — and, after `-O2` inlining has spliced a callee's entry block into the middle of
/// a caller block, that callee's pointer parameters too. It excludes a pointer whose `alloca` and
/// initializing store are separated by a branch (`mir_build` hoists a local's `alloca` to the entry
/// block but emits its store at the `let`, so `let p = &mut a;` after a loop keeps its slot).
///
/// One pass per phase classifies every candidate at once — a per-slot scan would be
/// `O(slots × block length)`, and the entry block holds one `alloca` per local of the whole function.
fn initialized_ptr_slots(
    f: &Function,
    cand: &BTreeMap<u32, MirType>,
    df: &[Vec<u32>],
) -> FxHashSet<u32> {
    // Phase 1: each pointer candidate's home block (where its `alloca` is).
    let mut home: FxHashMap<u32, u32> = FxHashMap::default();
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::Alloca(_)) = (inst.result, &inst.op) {
                if matches!(cand.get(&r.0), Some(MirType::Ptr)) {
                    home.insert(r.0, b.id.0);
                }
            }
        }
    }
    // Phase 2: was the first access inside the home block a store? (Accesses in any other block are
    // dominated by the home block, so they cannot come first.)
    let mut first_is_store: FxHashMap<u32, bool> = FxHashMap::default();
    for b in &f.blocks {
        for inst in &b.insts {
            match &inst.op {
                Op::Store { ptr, .. } if home.get(&ptr.0) == Some(&b.id.0) => {
                    first_is_store.entry(ptr.0).or_insert(true);
                }
                Op::Load(p, _) if home.get(&p.0) == Some(&b.id.0) => {
                    first_is_store.entry(p.0).or_insert(false);
                }
                _ => {}
            }
        }
    }
    first_is_store
        .into_iter()
        .filter(|&(slot, stored)| {
            let b = home[&slot];
            stored && (b == f.entry.0 || df[b as usize].is_empty())
        })
        .map(|(slot, _)| slot)
        .collect()
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
        children: &children,
        stack: FxHashMap::default(),
        replace: FxHashMap::default(),
        delete: FxHashSet::default(),
        append_then: FxHashMap::default(),
        append_else: FxHashMap::default(),
        append_br: FxHashMap::default(),
        zero_for: FxHashMap::default(),
        new_consts: Vec::new(),
        next_value: f.value_types.len() as u32,
    };
    r.visit(f, f.entry.0);
    let Rename {
        replace,
        delete,
        append_then,
        append_else,
        append_br,
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
            if let (Some(res), Op::Alloca(_)) = (inst.result, &inst.op) {
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
        // `initialized_ptr_slots` only admits a pointer slot whose first access in its own alloca's
        // block is a store, so its reaching definition is dominated by that store and this
        // read-before-write path is unreachable for it. There is no `const.ptr 0` to fall back on.
        debug_assert!(
            !matches!(ty, MirType::Ptr),
            "pointer slot v{var} reached the read-before-write path; initialized_ptr_slots is wrong"
        );
        if let Some(z) = self.zero_for.get(&ty) {
            return *z;
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

    fn visit(&mut self, f: &Function, block: u32) {
        let mut pushed: Vec<u32> = Vec::new();

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
            _ => {}
        }

        // (d) Recurse into dominator-tree children.
        for &c in &self.children[block as usize] {
            self.visit(f, c);
        }

        // (e) Pop the definitions introduced in this block.
        for var in pushed.into_iter().rev() {
            if let Some(s) = self.stack.get_mut(&var) {
                s.pop();
            }
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
