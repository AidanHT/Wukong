//! Block-parameter ("phi") cleanup.
//!
//! `mem2reg` places SSA merges as block parameters. Some are redundant: a parameter that is never
//! read, or one that receives the *same* value from every predecessor edge. Ordinary DCE cannot
//! touch block parameters (it only removes instructions) and the arguments feeding a dead parameter
//! keep otherwise-dead values alive, so this pass is what unblocks the rest of the pipeline.
//!
//! Two rewrites, applied one at a time to a fixed point:
//!  * **dead parameter** — a block parameter no *live* value depends on is removed, along with the
//!    matching argument on every incoming edge. "Live" is a fixpoint, not a use count: see
//!    [`live_values`].
//!  * **trivial phi** — a parameter whose incoming arguments are all one value `v` (ignoring
//!    self-references) is replaced everywhere by `v` and removed.
//!
//! Removing arguments can make further parameters dead or trivial, hence the fixpoint.
//!
//! # Why "dead" is a fixpoint and not a use count
//!
//! Counting uses answers the wrong question. A block argument is a use, so a *cycle* of parameters
//! that only feed each other keeps itself alive forever, and this front end builds those by the
//! dozen: a `let` inside an inner loop is an entry-block alloca, so after `mem2reg` its value is
//! threaded as a parameter through every enclosing loop header, even though nothing ever reads the
//! value the loop was entered with.
//!
//! That is not cosmetic. `loop_info` classifies such a parameter [`crate::loop_info::Carried`]
//! `::Recurrence` — a carried value it cannot account for — and `vectorize` declines any loop that
//! has one. So a dead parameter cycle around an inner loop is enough, on its own, to make the loop
//! unvectorizable. `for n in 0..N { let d = f(..); h[n] = d * h[n] + g; acc = acc + c[n] * d; }`
//! is exactly that shape.
//!
//! [`live_values`] therefore computes a least fixpoint: a value is live if an *instruction*, a
//! `ret`, or a branch condition reads it, or if it is passed on an edge into a parameter that is
//! itself live. Anything outside that closure is dead however many arguments mention it.

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
        let entry = f.entry.0;
        let mut changed = false;
        loop {
            // Dead parameters go **in one batch**, because [`live_values`] is already a transitive
            // fixpoint: a parameter whose only consumers are other dead parameters is not in the
            // live set either, so removing one can never reveal a second. Doing them one at a time
            // would recompute the whole fixpoint per parameter, and one loop nest can contribute
            // dozens.
            let live = live_values(f);
            let mut dead: Vec<(u32, usize)> = Vec::new();
            for b in &f.blocks {
                if b.id.0 == entry {
                    continue; // entry parameters are the function's parameters
                }
                for (k, p) in b.params.iter().enumerate() {
                    if !live.contains(&p.0) {
                        dead.push((b.id.0, k));
                    }
                }
            }
            // Descending order, so each removal leaves every position still to be removed valid.
            for &(blk, k) in dead.iter().rev() {
                remove_param(f, blk, k);
            }
            let mut progress = !dead.is_empty();
            // Trivial phis stay one at a time: `substitute` rewrites uses, which can make another
            // parameter trivial *or* dead, so the batch above has to be recomputed after each one.
            if let Some((blk, k, v)) = find_trivial(f, &preds, entry) {
                let pv = f.blocks[blk as usize].params[k];
                substitute(f, pv, v);
                remove_param(f, blk, k);
                progress = true;
            }
            if !progress {
                break;
            }
            changed = true;
        }
        changed
    }
}

/// Find one *trivial* block parameter — one whose incoming arguments are all a single value `v`
/// (ignoring self-references) — as `(block, param index, v)`.
fn find_trivial(f: &Function, preds: &[Vec<u32>], entry: u32) -> Option<(u32, usize, ValueId)> {
    for b in &f.blocks {
        if b.id.0 == entry {
            continue; // entry parameters are the function's parameters
        }
        for (k, &p) in b.params.iter().enumerate() {
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
                return Some((b.id.0, k, distinct[0]));
            }
        }
    }
    None
}

/// The least set of values that some *real* consumer depends on.
///
/// Seeded with every value an instruction operand, a `ret` or a branch condition reads — the uses
/// that actually do something — and closed under "an argument feeding a live parameter is live".
/// A block argument on its own is not a use: that is the whole point (see the module docs).
///
/// The case a use *count* misses is not exotic: it is what the front end emits for **every `let`
/// inside a loop body**. `mir_build` gives each local one `alloca` in the entry block, so a local
/// declared inside a loop is one slot written every iteration; `mem2reg` then threads that slot
/// through the loop as a block parameter whose latch argument is the value the body computed. When
/// nothing after the loop reads the local — the usual case for a temporary — that parameter's only
/// consumer is the branch argument feeding the *next* header parameter in the cycle, so every
/// parameter in the cycle has a non-zero use count and the counting scan removed none of them.
/// `loop_info` then classifies each one [`crate::loop_info::Carried`]`::Recurrence` and
/// `NaturalLoop::is_vectorizable_shape` refuses the loop, so a single dead `let` in the body —
/// `let e: f32 = exp(z[i] - m);` — was enough to make the whole loop unvectorizable.
///
/// Unreachable blocks are scanned like any other. Their terminators can only add values to the
/// live set, which is the conservative direction, and it keeps the answer independent of whether
/// `prune_unreachable` has run.
fn live_values(f: &Function) -> FxHashSet<u32> {
    // Where each block parameter sits, so a live parameter can find the arguments feeding it.
    let mut param_pos: FxHashMap<u32, (u32, usize)> = FxHashMap::default();
    for b in &f.blocks {
        for (k, p) in b.params.iter().enumerate() {
            param_pos.insert(p.0, (b.id.0, k));
        }
    }
    // Reverse edge index: `incoming[block][k]` is every argument passed at parameter position `k`.
    let mut incoming: Vec<Vec<Vec<ValueId>>> = f
        .blocks
        .iter()
        .map(|b| vec![Vec::new(); b.params.len()])
        .collect();
    let add_edge = |target: u32, args: &[ValueId], incoming: &mut Vec<Vec<Vec<ValueId>>>| {
        let slots = &mut incoming[target as usize];
        for (k, a) in args.iter().enumerate() {
            if let Some(s) = slots.get_mut(k) {
                s.push(*a);
            }
        }
    };
    let mut live: FxHashSet<u32> = FxHashSet::default();
    let mut work: Vec<u32> = Vec::new();
    let mark = |v: ValueId, live: &mut FxHashSet<u32>, work: &mut Vec<u32>| {
        if live.insert(v.0) {
            work.push(v.0);
        }
    };
    for b in &f.blocks {
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| mark(v, &mut live, &mut work));
        }
        match &b.term {
            Terminator::Ret(Some(v)) => mark(*v, &mut live, &mut work),
            Terminator::Br { target, args } => add_edge(target.0, args, &mut incoming),
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                mark(*cond, &mut live, &mut work);
                add_edge(then_blk.0, then_args, &mut incoming);
                add_edge(else_blk.0, else_args, &mut incoming);
            }
            Terminator::Ret(None) | Terminator::Unreachable => {}
        }
    }
    while let Some(v) = work.pop() {
        let Some(&(b, k)) = param_pos.get(&v) else {
            continue; // not a block parameter: nothing feeds it on an edge
        };
        for i in 0..incoming[b as usize][k].len() {
            let a = incoming[b as usize][k][i];
            mark(a, &mut live, &mut work);
        }
    }
    live
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

#[cfg(test)]
mod tests {
    use wukong_mir::{Function, MirType};
    use wukong_span::{Interner, SourceId};

    fn optimized(src: &str, func: &str) -> Function {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        crate::optimize(&mut program, 2);
        for f in &program.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "MIR invalid after -O2: {errs:?}");
        }
        let sym = interner.intern(func);
        program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"))
            .clone()
    }

    /// `d` and `h` are declared inside the innermost loop and read only inside the iteration that
    /// wrote them, but the front end allocates them in the entry block, so after `mem2reg` their
    /// values are threaded as parameters through all three loop headers. Nothing reads the value a
    /// loop was *entered* with, and the parameters form a closed cycle: each one's only use is the
    /// branch argument feeding the next.
    ///
    /// A dead-parameter test that counts uses cannot break that cycle. This is the regression: with
    /// the counting version, the inner loop's header carried two `f32` parameters `loop_info` could
    /// only classify as `Carried::Recurrence`, and `vectorize` declines any loop that has one — so
    /// a genuinely parallel inner loop stayed scalar because of two values that do not exist.
    #[test]
    fn a_dead_parameter_cycle_around_a_loop_nest_is_removed() {
        let src = "fn k(a: []f32, b: []f32, c: []f32, mut h: []f32, mut y: []f32) { \
                   for t in 0..4 { \
                     for d in 0..3 { \
                       let db: i32 = d * 2; \
                       let mut acc: f32 = 0.0; \
                       for n in 0..2 { \
                         let dec: f32 = a[db + n] * 0.5; \
                         let hn: f32 = dec * h[db + n] + b[db + n]; \
                         h[db + n] = hn; \
                         acc = acc + c[db + n] * hn; \
                       } \
                       y[t * 3 + d] = acc; \
                     } \
                   } } \
                   fn main() -> i32 { return 0; }";
        let f = optimized(src, "k");
        // Every remaining f32 loop-header parameter must be one a consumer actually reads. `acc`
        // is the only one: `dec` and `hn` are dead round-trips.
        let f32_params: usize = f
            .blocks
            .iter()
            .filter(|b| b.id != f.entry)
            .flat_map(|b| b.params.iter())
            .filter(|p| *f.value_type(**p) == MirType::F32)
            .count();
        assert_eq!(
            f32_params, 1,
            "only the accumulator should survive as a carried f32; got {f32_params}"
        );
    }

    /// The counterpart: a parameter whose value really is read on the next iteration must stay.
    /// `h = (a[i] + h) * b[i]` is a recurrence, and dropping its carried value would be a
    /// miscompile rather than a missed optimization. (Spelled with the add on the *outside* so it
    /// misses `wukong_lrscan_f32`, which would otherwise replace the whole loop with a kernel call
    /// and leave nothing to check.)
    #[test]
    fn a_live_recurrence_parameter_is_kept() {
        let src = "fn k(a: []f32, b: []f32, mut o: []f32, n: i64) { \
                   let mut h: f32 = 0.0; \
                   for i in 0..n { h = (a[i] + h) * b[i]; o[i] = h; } } \
                   fn main() -> i32 { return 0; }";
        let f = optimized(src, "k");
        let f32_params: usize = f
            .blocks
            .iter()
            .filter(|b| b.id != f.entry)
            .flat_map(|b| b.params.iter())
            .filter(|p| *f.value_type(**p) == MirType::F32)
            .count();
        assert!(f32_params >= 1, "the recurrence's carried value was dropped");
    }
}
