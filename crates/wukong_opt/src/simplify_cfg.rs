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

use crate::fxhash::FxHashMap;

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

/// Map every value id that names an integer constant to its value.
fn const_ints(f: &Function) -> FxHashMap<u32, i128> {
    let mut m = FxHashMap::default();
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(n, _)) = (inst.result, &inst.op) {
                m.insert(r.0, *n);
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
            let folded = match consts.get(&cond.0) {
                Some(&n) => Some(if n != 0 {
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

/// Fuse a block `A` that ends in `br B` into `B` whenever `A` is `B`'s only predecessor. `B`'s
/// parameters take the values `A` passed, so they are substituted away; `B`'s instructions and
/// terminator move into `A`, and the now-orphaned `B` is left for the pruner (`run_function`'s).
///
/// Predecessors come from the cache: valid on entry (the caller invalidated after folding), and
/// invalidated here after every merge, so they are rebuilt exactly once per merge and — the steady
/// state — not at all when there is nothing to merge.
fn merge_straight_line(f: &mut Function, cache: &mut CfgAnalyses) -> bool {
    let mut changed = false;
    loop {
        let preds = cache.predecessors(f);
        let mut pair = None;
        for (ai, b) in f.blocks.iter().enumerate() {
            if let Terminator::Br { target, .. } = &b.term {
                let bi = target.0 as usize;
                // `B` must be a different block whose sole predecessor is `A`.
                if bi != ai && preds[bi].len() == 1 && preds[bi][0] == ai as u32 {
                    pair = Some((ai, bi));
                    break;
                }
            }
        }
        let Some((ai, bi)) = pair else { break };

        let args = match &f.blocks[ai].term {
            Terminator::Br { args, .. } => args.clone(),
            _ => unreachable!("selected block ends in br"),
        };
        let bparams = std::mem::take(&mut f.blocks[bi].params);
        // B's parameters are exactly the values A handed it.
        let subst: FxHashMap<u32, ValueId> =
            bparams.iter().zip(&args).map(|(p, a)| (p.0, *a)).collect();
        if !subst.is_empty() {
            for blk in &mut f.blocks {
                for inst in &mut blk.insts {
                    map_op_uses(&mut inst.op, |v| *subst.get(&v.0).unwrap_or(&v));
                }
                map_term_uses(&mut blk.term, |v| *subst.get(&v.0).unwrap_or(&v));
            }
        }

        let b_insts = std::mem::take(&mut f.blocks[bi].insts);
        let b_term = std::mem::replace(&mut f.blocks[bi].term, Terminator::Unreachable);
        f.blocks[ai].insts.extend(b_insts);
        f.blocks[ai].term = b_term;
        cache.invalidate(); // `A`'s successors are now `B`'s, and `B` is orphaned
        changed = true;
    }
    changed
}
