//! Block-parameter ("phi") cleanup.
//!
//! `mem2reg` places SSA merges as block parameters. Some are redundant: a parameter that is never
//! read, or one that receives the *same* value from every predecessor edge. Ordinary DCE cannot
//! touch block parameters (it only removes instructions) and the arguments feeding a dead parameter
//! keep otherwise-dead values alive, so this pass is what unblocks the rest of the pipeline.
//!
//! Two rewrites, applied one at a time to a fixed point:
//!  * **dead parameter** — a block parameter with no uses is removed, along with the matching
//!    argument on every incoming edge.
//!  * **trivial phi** — a parameter whose incoming arguments are all one value `v` (ignoring
//!    self-references) is replaced everywhere by `v` and removed.
//!
//! Removing arguments can make further parameters dead or trivial, hence the fixpoint.

use crate::fxhash::FxHashSet;

use mercury_mir::{Function, Terminator, ValueId};

use crate::{each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

pub struct SimplifyPhis;

impl Pass for SimplifyPhis {
    fn name(&self) -> &'static str {
        "simplify-phis"
    }

    fn run_function(&self, f: &mut Function, _cache: &mut CfgAnalyses) -> bool {
        let mut changed = false;
        while let Some((blk, k, repl)) = find_removable(f) {
            if let Some(v) = repl {
                let pv = f.blocks[blk as usize].params[k];
                substitute(f, pv, v);
            }
            remove_param(f, blk, k);
            changed = true;
        }
        changed
    }
}

/// Find one removable block parameter: `(block, param index, Some(replacement) | None)`. `None`
/// means the parameter is dead; `Some(v)` means it is a trivial phi to be replaced by `v`.
fn find_removable(f: &Function) -> Option<(u32, usize, Option<ValueId>)> {
    let entry = f.entry.0;

    let mut used: FxHashSet<u32> = FxHashSet::default();
    for b in &f.blocks {
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| {
                used.insert(v.0);
            });
        }
        each_term_use(&b.term, &mut |v| {
            used.insert(v.0);
        });
    }

    for b in &f.blocks {
        if b.id.0 == entry {
            continue; // entry parameters are the function's parameters
        }
        for (k, &p) in b.params.iter().enumerate() {
            if !used.contains(&p.0) {
                return Some((b.id.0, k, None));
            }
            // Gather the incoming arguments at this parameter position.
            let mut distinct: Vec<ValueId> = Vec::new();
            for src in &f.blocks {
                for arg in edge_args_to(&src.term, b.id.0, k) {
                    if arg != p && !distinct.contains(&arg) {
                        distinct.push(arg);
                    }
                }
            }
            if distinct.len() == 1 {
                return Some((b.id.0, k, Some(distinct[0])));
            }
        }
    }
    None
}

/// The argument(s) at position `k` on every edge from `term` to `target`.
fn edge_args_to(term: &Terminator, target: u32, k: usize) -> Vec<ValueId> {
    let mut out = Vec::new();
    match term {
        Terminator::Br { target: t, args } if t.0 == target => {
            if let Some(a) = args.get(k) {
                out.push(*a);
            }
        }
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => {
            if then_blk.0 == target {
                if let Some(a) = then_args.get(k) {
                    out.push(*a);
                }
            }
            if else_blk.0 == target {
                if let Some(a) = else_args.get(k) {
                    out.push(*a);
                }
            }
        }
        _ => {}
    }
    out
}

/// Replace every use of `from` with `to` across the function.
fn substitute(f: &mut Function, from: ValueId, to: ValueId) {
    let rw = |v: ValueId| if v.0 == from.0 { to } else { v };
    for b in &mut f.blocks {
        for inst in &mut b.insts {
            map_op_uses(&mut inst.op, rw);
        }
        map_term_uses(&mut b.term, rw);
    }
}

/// Remove parameter `k` from `blk` and the matching argument from every edge that targets it.
fn remove_param(f: &mut Function, blk: u32, k: usize) {
    f.blocks[blk as usize].params.remove(k);
    for b in &mut f.blocks {
        match &mut b.term {
            Terminator::Br { target, args } if target.0 == blk => {
                if k < args.len() {
                    args.remove(k);
                }
            }
            Terminator::CondBr {
                then_blk,
                then_args,
                else_blk,
                else_args,
                ..
            } => {
                if then_blk.0 == blk && k < then_args.len() {
                    then_args.remove(k);
                }
                if else_blk.0 == blk && k < else_args.len() {
                    else_args.remove(k);
                }
            }
            _ => {}
        }
    }
}
