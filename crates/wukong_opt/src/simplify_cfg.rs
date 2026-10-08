//! Control-flow-graph cleanup.
//!
//! Three cooperating transforms, run to a fixpoint with the rest of the pipeline (with jump
//! threading and branch folding from `branch_fold` run around the merge, see that module):
//!  1. **Constant branch folding** — a `cond_br` whose condition is a known constant (after
//!     const-folding upstream) becomes an unconditional `br` to the taken side, carrying that
//!     side's block arguments. A `cond_br` whose two targets *and* arguments are identical also
//!     collapses to a `br`, and so does a constant or degenerate `switch` (`switch_form`).
//!  2. **Straight-line merging** — when a block ends in an unconditional `br` to a block whose
//!     only predecessor it is, the two are fused into one. This collapses the forwarding chains
//!     that branch folding and mem2reg leave behind and, by enlarging basic blocks, gives the
//!     intra-block CSE and DSE passes much more to work with.
//!  3. **Unreachable-block elimination** — blocks no longer reachable from the entry are removed
//!     and the remaining blocks are renumbered, with every terminator target and the function
//!     entry rewritten to the new ids.
//!
//! Folding and merging can make blocks unreachable, so pruning always runs last.

use wukong_mir::{Function, Op, Terminator, ValueId};

use crate::{cfg, map_op_uses, map_term_uses, CfgAnalyses, Pass};

pub struct SimplifyCfg;

impl Pass for SimplifyCfg {
    fn name(&self) -> &'static str {
        "simplify-cfg"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // This is the one pass that restructures the CFG (folded branches, merged/removed blocks),
        // so every rewrite below invalidates the cached analyses *as it happens* — which is also
        // what lets the steady-state call, where nothing folds, merges or dies, read predecessors
        // and reachability from the cache instead of recomputing both from scratch.
        let mut changed = fold_constant_branches(f);
        // A constant or degenerate `switch` (`switch_form`).
        changed |= crate::switch_form::fold_switches(f);
        if changed {
            cache.invalidate(); // successor edges changed
        }
        // Jump threading over constant edge arguments (`branch_fold`).
        changed |= crate::branch_fold::thread_edges(f, cache);
        changed |= merge_straight_line(f, cache);
        // One branch on `and`/`or` for a successor that re-tests to the same place (`branch_fold`).
        changed |= crate::branch_fold::fold_branches(f, cache);
        // Pruning is a no-op when every block is reachable, and the cached reverse postorder
        // answers that without the reachability DFS `prune_unreachable` would run on every call.
        if !cache.all_reachable(f) {
            changed |= cfg::prune_unreachable(f);
            cache.invalidate(); // blocks renumbered
        }
        changed
    }
}

/// `ValueId -> integer constant`, dense over the value arena (the same layout `dce` uses): a
/// branch condition is looked up once per `cond_br`, and a hash map over a dense `u32` keyspace
/// only added hashing and growth to that.
fn const_ints(f: &Function) -> Vec<Option<i128>> {
    let mut m = vec![None; f.value_types.len()];
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(n, _)) = (inst.result, &inst.op) {
                if let Some(slot) = m.get_mut(r.0 as usize) {
                    *slot = Some(*n);
                }
            }
        }
    }
    m
}

fn fold_constant_branches(f: &mut Function) -> bool {
    // Only a `cond_br` can fold, and after the first sweep most functions have none left: build
    // the constant map only when there is a branch it could apply to.
    if !f
        .blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::CondBr { .. }))
    {
        return false;
    }
    let consts = const_ints(f);
    let mut changed = false;
    for b in &mut f.blocks {
        if let Terminator::CondBr {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } = &b.term
        {
            // Same destination and arguments on both arms: unconditional.
            let identical = then_blk == else_blk && then_args == else_args;
            let folded = match consts.get(cond.0 as usize).copied().flatten() {
                Some(n) => Some(if n != 0 {
                    (*then_blk, then_args.clone())
                } else {
                    (*else_blk, else_args.clone())
                }),
                None if identical => Some((*then_blk, then_args.clone())),
                None => None,
            };
            if let Some((target, args)) = folded {
                b.term = Terminator::Br { target, args };
                changed = true;
            }
        }
    }
    changed
}

/// The block `A` (index `ai`) can absorb, if any: the target `B` of its unconditional branch, when
/// `B` is a different block whose only predecessor is `A`.
fn merge_target(f: &Function, preds: &[Vec<u32>], ai: usize) -> Option<usize> {
    let Terminator::Br { target, .. } = &f.blocks[ai].term else {
        return None;
    };
    let bi = target.0 as usize;
    (bi != ai && preds[bi].len() == 1 && preds[bi][0] == ai as u32).then_some(bi)
}

/// Fuse a block `A` that ends in `br B` into `B` whenever `A` is `B`'s only predecessor. `B`'s
/// parameters take the values `A` passed, so they are substituted away; `B`'s instructions and
/// terminator move into `A`, and the now-orphaned `B` is left for the pruner (`run_function`'s).
///
/// One linear worklist sweep. Predecessors are read once and then maintained incrementally — a
/// merge changes only `A`'s successor set (it becomes `B`'s) and orphans `B`, both of which are
/// local updates — and each head absorbs its whole straight-line chain before the scan moves on.
/// Every maximal chain collapses into its head whatever order the merges happen in, and the
/// parameter substitutions compose, so the result is the one the old restart-from-block-zero loop
/// produced; that loop rebuilt every predecessor list (one heap `Vec` per block) after each merge,
/// O(merges x blocks) on exactly the long chains mem2reg and branch folding leave behind. The
/// steady state, where nothing merges, reads the cached predecessors and allocates nothing.
fn merge_straight_line(f: &mut Function, cache: &mut CfgAnalyses) -> bool {
    let n = f.blocks.len();
    let mut preds: Vec<Vec<u32>> = {
        let cached = cache.predecessors(f);
        if !(0..n).any(|ai| merge_target(f, cached, ai).is_some()) {
            return false;
        }
        cached.to_vec()
    };
    // `B`-parameter -> the argument `A` passed for it, resolved through chains at the end. Sized
    // lazily: most merged blocks carry no parameters.
    let mut subst: Vec<Option<ValueId>> = Vec::new();
    let mut changed = false;
    for ai in 0..n {
        while let Some(bi) = merge_target(f, &preds, ai) {
            let args = match &f.blocks[ai].term {
                Terminator::Br { args, .. } => args.clone(),
                _ => unreachable!("merge_target only selects a block ending in br"),
            };
            let bparams = std::mem::take(&mut f.blocks[bi].params);
            if !bparams.is_empty() && subst.is_empty() {
                subst = vec![None; f.value_types.len()];
            }
            for (p, a) in bparams.iter().zip(&args) {
                if let Some(slot) = subst.get_mut(p.0 as usize) {
                    *slot = Some(*a);
                }
            }
            let b_insts = std::mem::take(&mut f.blocks[bi].insts);
            let b_term = std::mem::replace(&mut f.blocks[bi].term, Terminator::Unreachable);
            // `B`'s successors now see `A` in its place. `A` reached nothing but `B`, so this can
            // never list `A` twice.
            for s in cfg::successors(&b_term) {
                for p in &mut preds[s.0 as usize] {
                    if *p == bi as u32 {
                        *p = ai as u32;
                    }
                }
            }
            preds[bi].clear();
            f.blocks[ai].insts.extend(b_insts);
            f.blocks[ai].term = b_term;
            changed = true;
        }
    }
    if !changed {
        return false;
    }
    if !subst.is_empty() {
        // In reachable code the chains are acyclic (each argument dominates the block it is passed
        // into); the step bound only keeps a cycle among unreachable blocks — which the pruner
        // deletes next — from spinning.
        let limit = subst.len();
        let resolve = |v: ValueId| {
            let mut cur = v;
            for _ in 0..limit {
                match subst.get(cur.0 as usize).copied().flatten() {
                    Some(next) if next != cur => cur = next,
                    _ => break,
                }
            }
            cur
        };
        for blk in &mut f.blocks {
            for inst in &mut blk.insts {
                map_op_uses(&mut inst.op, resolve);
            }
            map_term_uses(&mut blk.term, resolve);
        }
    }
    cache.invalidate(); // merged blocks' successors moved and the absorbed blocks are orphaned
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, BlockId, Builder, MirType, Program};
    use wukong_span::Interner;

    /// `chain(x)`: the entry branches to the *last-numbered* block and every link branches to the
    /// block numbered one below it, so each link's only predecessor sits *after* it in layout order.
    /// Even links add one; odd links pass their parameter straight through, so a substitution can
    /// name a parameter that is itself substituted by a later merge. One run must collapse the whole
    /// chain into the entry and compose those substitutions back into the original arithmetic.
    fn reverse_chain(it: &mut Interner, links: usize) -> Function {
        let mut b = Builder::new(it.intern("chain"), MirType::I64);
        let x = b.add_param(MirType::I64);
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let blocks: Vec<BlockId> = (0..links).map(|_| b.new_block()).collect();
        let params: Vec<ValueId> = blocks
            .iter()
            .map(|&blk| b.block_param(blk, MirType::I64))
            .collect();
        b.br(blocks[links - 1], vec![x]);
        for (k, (&blk, &p)) in blocks.iter().zip(&params).enumerate().rev() {
            b.switch_to(blk);
            let next = if k % 2 == 0 {
                b.build(MirType::I64, Op::Bin(BinOp::Add, p, one))
            } else {
                p
            };
            if k == 0 {
                b.ret(Some(next));
            } else {
                b.br(blocks[k - 1], vec![next]);
            }
        }
        b.finish()
    }

    #[test]
    fn a_reverse_laid_out_parameter_chain_collapses_in_one_run() {
        let mut it = Interner::new();
        let links = 64;
        let mut f = reverse_chain(&mut it, links);
        let mut cache = CfgAnalyses::default();
        assert!(SimplifyCfg.run_function(&mut f, &mut cache));
        assert_eq!(
            f.blocks.len(),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert!(f.blocks[0].params == f.params && f.blocks[0].params.len() == 1);
        // Nothing left to merge: the steady state reports no change.
        assert!(!SimplifyCfg.run_function(&mut f, &mut cache));

        // chain(5) = 5 + one add per even link.
        let chain = f.name;
        let mut main = Builder::new(it.intern("main"), MirType::I64);
        let five = main.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        let r = main.build(
            MirType::I64,
            Op::Call {
                func: chain,
                args: vec![five],
            },
        );
        main.ret(Some(r));
        let mut p = Program::new();
        p.funcs = vec![f, main.finish()];
        assert!(wukong_mir::verify::verify_program(&p).is_empty());
        let main = it.intern("main");
        assert_eq!(
            wukong_interp::run(&p, main, &it).unwrap(),
            5 + links as i64 / 2
        );
    }

    /// A diamond has nothing to merge, and the merge scan must say so without touching the CFG.
    #[test]
    fn a_diamond_has_nothing_to_merge() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("d"), MirType::I64);
        let c = b.add_param(MirType::I1);
        let t = b.new_block();
        let e = b.new_block();
        let j = b.new_block();
        let v = b.block_param(j, MirType::I64);
        b.cond_br(c, t, vec![], e, vec![]);
        b.switch_to(t);
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        b.br(j, vec![one]);
        b.switch_to(e);
        let two = b.build(MirType::I64, Op::ConstInt(2, MirType::I64));
        b.br(j, vec![two]);
        b.switch_to(j);
        b.ret(Some(v));
        let mut f = b.finish();
        assert!(!SimplifyCfg.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(f.blocks.len(), 4);
    }
}
