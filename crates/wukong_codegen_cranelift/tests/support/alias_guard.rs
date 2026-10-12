//! Shared by the alias-semantics integration tests: does a recognized kernel call sit behind
//! `wukong_mir_build`'s run-time alias guard (`kernel_guard.rs`)?
//!
//! Since run-time alias versioning (`f4082bbf`), a `MayAlias` operand pair no longer makes the
//! recognizer decline at compile time. It opens a guard instead:
//!
//! ```text
//!   ok = AND over checked pairs of (ptrtoint(gep(a, n)) <= ptrtoint(b) || ... )
//!   cond_br ok, kernel, fallback      ; fallback = the ORIGINAL source loop
//! ```
//!
//! so an aliased call site may contain the kernel call, but only in the `then` arm of that
//! overlap test. The invariant the old "an alias must not dispatch" assertions encoded is
//! restated here exactly as `wukong_mir_build`'s own `fn_calls_unguarded` test helper states it:
//! a guarded call sits in a block whose every predecessor is a `cond_br` on an alias-guard
//! condition with that block as its `then` arm.
//!
//! The `@parallel` region outliner versions a threaded region on the same test
//! (`wukong_mir_build::par_region::emit_region_guard`): `wukong_parallel_for` in the `then` arm, a
//! direct call of the outlined body over every iteration on the calling thread in the other, so
//! [`is_guarded_block`] answers for a region call as well.

use std::collections::HashMap;
use wukong_mir::{BinOp, BlockId, CastKind, CmpOp, Function, Op, Program, Terminator, ValueId};
use wukong_span::Interner;

/// Is `cond` (transitively, through `and`/`or`) a `ule`/`eq` comparison of a `ptrtoint`?
fn is_alias_guard_cond(defs: &HashMap<ValueId, &Op>, cond: ValueId, depth: u32) -> bool {
    if depth > 64 {
        return false;
    }
    match defs.get(&cond) {
        Some(Op::Bin(BinOp::And | BinOp::Or, a, b)) => {
            is_alias_guard_cond(defs, *a, depth + 1) || is_alias_guard_cond(defs, *b, depth + 1)
        }
        Some(Op::Cmp(CmpOp::Ule | CmpOp::Eq, a, b)) => [a, b]
            .iter()
            .any(|v| matches!(defs.get(*v), Some(Op::Cast(CastKind::PtrToInt, _, _)))),
        _ => false,
    }
}

fn successors(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => vec![*then_blk, *else_blk],
        Terminator::Switch { cases, default, .. } => {
            cases.iter().map(|c| c.target).chain([*default]).collect()
        }
        Terminator::Ret(_) | Terminator::Unreachable => vec![],
    }
}

/// Is block `id` of `f` behind a run-time alias guard: does every predecessor reach it as the
/// `then` arm (and only that arm) of a `cond_br` on an alias-guard condition?
pub fn is_guarded_block(f: &Function, id: BlockId) -> bool {
    let mut defs: HashMap<ValueId, &Op> = HashMap::new();
    for block in &f.blocks {
        for inst in &block.insts {
            if let Some(result) = inst.result {
                defs.insert(result, &inst.op);
            }
        }
    }
    let preds: Vec<&Terminator> = f
        .blocks
        .iter()
        .map(|p| &p.term)
        .filter(|term| successors(term).contains(&id))
        .collect();
    !preds.is_empty()
        && preds.iter().all(|term| {
            matches!(
                term,
                Terminator::CondBr { cond, then_blk, else_blk, .. }
                    if *then_blk == id
                        && *else_blk != id
                        && is_alias_guard_cond(&defs, *cond, 0)
            )
        })
}

/// The callees, among those `is_kernel` selects, that `f` calls from a block NOT behind a run-time
/// alias guard.
fn unguarded_in(
    f: &Function,
    interner: &Interner,
    is_kernel: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let mut unguarded = Vec::new();
    for block in &f.blocks {
        if is_guarded_block(f, block.id) {
            continue;
        }
        for inst in &block.insts {
            if let Op::Call { func, .. } = &inst.op {
                let name = interner.resolve(*func);
                if is_kernel(name) {
                    unguarded.push(name.to_string());
                }
            }
        }
    }
    unguarded
}

/// Every kernel call (as `is_kernel` selects them) in `program` that is not behind a run-time
/// alias guard. An aliased call site must make this empty: the kernel may exist in the program,
/// but only on the arm where the operands were proven disjoint (or exactly in place, where the
/// runtime documents that) at run time.
pub fn unguarded_dispatches(
    program: &Program,
    interner: &Interner,
    is_kernel: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    program
        .funcs
        .iter()
        .flat_map(|f| unguarded_in(f, interner, is_kernel))
        .collect()
}
