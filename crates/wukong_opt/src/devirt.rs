//! Devirtualization: a call through a function pointer whose callee is a visible `func_addr g`
//! becomes the direct call `call g(..)`.
//!
//! A first-class function value (`let f: fn(i32) -> i32 = add1; f(3)`) lowers to
//! [`Op::CallIndirect`] through whatever pointer the value is held in. Once mem2reg, simplify and
//! cse have run, the common case is visible: the callee operand is the `func_addr` the value was
//! made from, in the same function. Rewriting it to a direct call is the whole win — every pass
//! that models calls precisely does so for a *named* callee (the call-effect summaries let cse,
//! dce and licm treat a side-effect-free callee like an instruction, the inliner splices it, tail
//! recursion and the recognizers see it), while an indirect call stays the opaque op
//! [`wukong_mir::Effects`] says it is.
//!
//! The rewrite requires the target's MIR signature to be exactly the call's — the argument types
//! and the return type. The front end only builds a `call_indirect` from the function type of the
//! value, so that always holds for source programs; a mismatch (hand-written MIR) is left alone,
//! for the verifier to report, rather than turned into a direct call of the wrong ABI.
//!
//! # Why it runs between pipeline runs, not inside the fixpoint
//!
//! The per-function pipeline consults the call-effect table about every symbol an `Op::Call`
//! names, and the table it sees depends on the schedule: it holds exactly the call-graph SCCs
//! optimized *before* the function, which the parallel schedule and the sequential one agree on
//! only for the function's own (direct) callees. A call created inside the fixpoint could name a
//! function the two schedules have summarized differently, and the optimized MIR would then depend
//! on the thread count. So [`devirtualize_program`] runs as a program-level step after the
//! pipeline; when it rewrites anything, the optimizer re-runs inlining (at `-O2`) and the pipeline
//! over the program, whose call graph now has the new direct edges.

use wukong_mir::{Function, MirType, Op, Program};
use wukong_span::Symbol;

use crate::fxhash::{FxHashMap, FxHashSet};

/// Each defined function's MIR signature, by name. A name carried by more than one function is
/// left out: a call names its callee by symbol, so which of them a direct call would reach is not
/// this pass's to decide.
fn signatures(program: &Program) -> FxHashMap<Symbol, (Vec<MirType>, MirType)> {
    let mut sigs: FxHashMap<Symbol, (Vec<MirType>, MirType)> = FxHashMap::default();
    let mut duplicated: FxHashSet<Symbol> = FxHashSet::default();
    for f in &program.funcs {
        let params = f.params.iter().map(|p| f.value_type(*p).clone()).collect();
        if sigs.insert(f.name, (params, f.ret.clone())).is_some() {
            duplicated.insert(f.name);
        }
    }
    for d in duplicated {
        sigs.remove(&d);
    }
    sigs
}

/// Rewrite every devirtualizable indirect call of the program. Returns the indices of the
/// functions that changed, in order. [`crate::optimize`] runs it after the per-function pipeline
/// and, when it changed anything, inlines (at `-O2`) and runs the pipeline again.
pub fn devirtualize_program(program: &mut Program) -> Vec<usize> {
    if !program
        .funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i.op, Op::CallIndirect { .. }))
    {
        return Vec::new();
    }
    let sigs = signatures(program);
    program
        .funcs
        .iter_mut()
        .enumerate()
        .filter_map(|(i, f)| devirtualize_function(f, &sigs).then_some(i))
        .collect()
}

/// Rewrite `f`'s indirect calls whose callee is a `func_addr g` of `f` itself, when `g`'s
/// signature in `sigs` is exactly the call's. Returns whether anything changed.
pub(crate) fn devirtualize_function(
    f: &mut Function,
    sigs: &FxHashMap<Symbol, (Vec<MirType>, MirType)>,
) -> bool {
    let addrs: FxHashMap<u32, Symbol> = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|inst| match (&inst.op, inst.result) {
            (Op::FuncAddr(g), Some(r)) => Some((r.0, *g)),
            _ => None,
        })
        .collect();
    if addrs.is_empty() {
        return false;
    }
    let mut changed = false;
    for bi in 0..f.blocks.len() {
        for ii in 0..f.blocks[bi].insts.len() {
            let Op::CallIndirect { callee, args, ret } = &f.blocks[bi].insts[ii].op else {
                continue;
            };
            let Some(&g) = addrs.get(&callee.0) else {
                continue;
            };
            let Some((params, want_ret)) = sigs.get(&g) else {
                continue;
            };
            let same = want_ret == ret
                && params.len() == args.len()
                && args.iter().zip(params).all(|(a, p)| f.value_type(*a) == p);
            if !same {
                continue;
            }
            let args = args.clone();
            f.blocks[bi].insts[ii].op = Op::Call { func: g, args };
            changed = true;
        }
    }
    changed
}
