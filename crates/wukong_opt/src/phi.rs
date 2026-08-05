//! Block-parameter ("phi") cleanup.
//!
//! `mem2reg` places SSA merges as block parameters. Some are redundant: a parameter that is never
//! read, or one that receives the *same* value from every predecessor edge. Ordinary DCE cannot
//! touch block parameters (it only removes instructions) and the arguments feeding a dead parameter
//! keep otherwise-dead values alive, so this pass is what unblocks the rest of the pipeline.
//!
//! Two rewrites, applied one at a time to a fixed point:
//!  * **dead parameter** — a block parameter no *live* code reads is removed, along with the
//!    matching argument on every incoming edge. "Live" is a fixpoint, not a use count: see
//!    [`live_params`].
//!  * **trivial phi** — a parameter whose incoming arguments are all one value `v` (ignoring
//!    self-references) is replaced everywhere by `v` and removed.
//!
//! Removing arguments can make further parameters dead or trivial, hence the fixpoint.

use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{Function, Terminator, ValueId};

use crate::{each_op_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

pub struct SimplifyPhis;

impl Pass for SimplifyPhis {
    fn name(&self) -> &'static str {
        "simplify-phis"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // This pass rewires block parameters and edge arguments but never the CFG's successor edges,
        // so the predecessor map stays valid across every removal; compute it once (cached) and hand
        // it to each scan. Removals mutate `f`, not the (successor-derived) cache.
        let preds: Vec<Vec<u32>> = cache.predecessors(f).to_vec();
        let mut changed = false;
        while let Some((blk, k, repl)) = find_removable(f, &preds) {
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

/// Which block parameters are *effectively* live — the least fixed point of "something that is not
/// itself a dead parameter reads it".
///
/// A plain use count is not enough, and the case it misses is not exotic: it is what the front end
/// emits for **every `let` inside a loop body**. `mir_build` gives each local one `alloca` in the
/// entry block, so a local declared inside a loop is one slot written every iteration; `mem2reg`
/// then threads that slot through the loop as a block parameter whose latch argument is the value
/// the body computed. When nothing after the loop reads the local — the usual case for a temporary
/// — the parameter's only consumer is the branch argument that feeds the *next* header parameter in
/// the cycle, and every parameter in that cycle therefore has a non-zero use count. The old scan saw
/// each of them as used and removed none.
///
/// That is not merely untidy. `loop_info` classifies such a parameter [`crate::loop_info::Carried`]
/// `::Recurrence` (its latch argument is neither the parameter itself nor a recognized combine), and
/// `NaturalLoop::is_vectorizable_shape` refuses any loop that carries one. So a single dead `let` in
/// the body — `let e: f32 = exp(z[i] - m);` — was enough to make the whole loop unvectorizable.
///
/// The fixpoint: seed with every parameter read by an *instruction* or by a terminator operand that
/// is not an edge argument (`ret`'s value, `cond_br`'s condition), then propagate backwards — if
/// parameter `k` of block `b` is live, every argument at position `k` on every edge into `b` is a
/// live read. Anything never reached is dead: removing it deletes the only edges that referenced it.
///
/// Entry-block parameters are the function's own parameters and are never candidates, so they are
/// left out of the map entirely rather than seeded.
fn live_params(f: &Function, preds: &[Vec<u32>]) -> FxHashSet<u32> {
    let entry = f.entry.0;
    // value -> (block, parameter index), for non-entry block parameters only.
    let mut owner: FxHashMap<u32, (u32, usize)> = FxHashMap::default();
    for b in &f.blocks {
        if b.id.0 == entry {
            continue;
        }
        for (k, &p) in b.params.iter().enumerate() {
            owner.insert(p.0, (b.id.0, k));
        }
    }

    let mut live: FxHashSet<u32> = FxHashSet::default();
    let mut work: Vec<(u32, usize)> = Vec::new();
    let mut seeds: Vec<ValueId> = Vec::new();
    for b in &f.blocks {
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| seeds.push(v));
        }
        match &b.term {
            // The only terminator operands that are not edge arguments.
            Terminator::Ret(Some(v)) => seeds.push(*v),
            Terminator::CondBr { cond, .. } => seeds.push(*cond),
            _ => {}
        }
    }
    for v in seeds {
        if let Some(&at) = owner.get(&v.0) {
            if live.insert(v.0) {
                work.push(at);
            }
        }
    }
    while let Some((b, k)) = work.pop() {
        for &src in &preds[b as usize] {
            for arg in edge_args_to(&f.blocks[src as usize].term, b, k) {
                if let Some(&at) = owner.get(&arg.0) {
                    if live.insert(arg.0) {
                        work.push(at);
                    }
                }
            }
        }
    }
    live
}

/// Find one removable block parameter: `(block, param index, Some(replacement) | None)`. `None`
/// means the parameter is dead; `Some(v)` means it is a trivial phi to be replaced by `v`.
fn find_removable(f: &Function, preds: &[Vec<u32>]) -> Option<(u32, usize, Option<ValueId>)> {
    let entry = f.entry.0;
    let live = live_params(f, preds);

    for b in &f.blocks {
        if b.id.0 == entry {
            continue; // entry parameters are the function's parameters
        }
        for (k, &p) in b.params.iter().enumerate() {
            if !live.contains(&p.0) {
                return Some((b.id.0, k, None));
            }
            // Gather the incoming arguments at this parameter position — only predecessors of `b`
            // can supply them, so scan `preds[b]` rather than every block. (Byte-identical: a
            // non-predecessor contributes no edge, and predecessors are in ascending block order.)
            let mut distinct: Vec<ValueId> = Vec::new();
            for &src in &preds[b.id.0 as usize] {
                for arg in edge_args_to(&f.blocks[src as usize].term, b.id.0, k) {
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
