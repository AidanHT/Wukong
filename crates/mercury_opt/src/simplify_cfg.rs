//! Control-flow-graph cleanup.
//!
//! Three cooperating transforms, run to a fixpoint with the rest of the pipeline:
//!  1. **Constant branch folding** — a `cond_br` whose condition is a known constant (after
//!     const-folding upstream) becomes an unconditional `br` to the taken side, carrying that
//!     side's block arguments. A `cond_br` whose two targets *and* arguments are identical also
//!     collapses to a `br`.
//!  2. **Straight-line merging** — when a block ends in an unconditional `br` to a block whose
//!     only predecessor it is, the two are fused into one. This collapses the forwarding chains
//!     that branch folding and mem2reg leave behind and, by enlarging basic blocks, gives the
//!     intra-block CSE and DSE passes much more to work with.
//!  3. **Unreachable-block elimination** — blocks no longer reachable from the entry are removed
//!     and the remaining blocks are renumbered, with every terminator target and the function
//!     entry rewritten to the new ids.
//!
//! Folding and merging can make blocks unreachable, so pruning always runs last.

use std::collections::HashMap;

use mercury_mir::{Function, Op, Terminator, ValueId};

use crate::{cfg, map_op_uses, map_term_uses, Pass};

pub struct SimplifyCfg;

impl Pass for SimplifyCfg {
    fn name(&self) -> &'static str {
        "simplify-cfg"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        let mut changed = fold_constant_branches(f);
        changed |= merge_straight_line(f);
        changed |= cfg::prune_unreachable(f);
        changed
    }
}

/// Map every value id that names an integer constant to its value.
fn const_ints(f: &Function) -> HashMap<u32, i128> {
    let mut m = HashMap::new();
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
/// terminator move into `A`, and the now-orphaned `B` is left for the pruner.
fn merge_straight_line(f: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let preds = cfg::predecessors(f);
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
        let subst: HashMap<u32, ValueId> =
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
        changed = true;
    }
    if changed {
        cfg::prune_unreachable(f);
    }
    changed
}
