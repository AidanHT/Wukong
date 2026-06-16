//! Control-flow-graph cleanup.
//!
//! Two cooperating transforms, run to a fixpoint with the rest of the pipeline:
//!  1. **Constant branch folding** — a `cond_br` whose condition is a known constant (after
//!     const-folding upstream) becomes an unconditional `br` to the taken side, carrying that
//!     side's block arguments. A `cond_br` whose two targets *and* arguments are identical also
//!     collapses to a `br`.
//!  2. **Unreachable-block elimination** — blocks no longer reachable from the entry are removed
//!     and the remaining blocks are renumbered, with every terminator target and the function
//!     entry rewritten to the new ids.
//!
//! Folding can make blocks unreachable, so we always fold first, then prune.

use std::collections::HashMap;

use mercury_mir::{Function, Op, Terminator};

use crate::{cfg, Pass};

pub struct SimplifyCfg;

impl Pass for SimplifyCfg {
    fn name(&self) -> &'static str {
        "simplify-cfg"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        let mut changed = fold_constant_branches(f);
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
