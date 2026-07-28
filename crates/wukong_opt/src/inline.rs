//! Function inlining (a program-level transform).
//!
//! Inlining replaces a call with a copy of the callee's body, removing the call/return overhead and
//! exposing the callee's code to the caller's optimizer (constant args become constants, results
//! feed local CSE, etc.). It runs once before the function-level pipeline so that mem2reg and the
//! rest then optimize the spliced-in code in context.
//!
//! We only inline **leaf** functions — ones that call no other user function — and only when they
//! are small. Leaf-only is a deliberately strong restriction: it makes runaway and recursive
//! inlining impossible (a leaf cannot call back into anything), so no cycle detection or depth
//! bound is needed beyond a per-caller growth cap.
//!
//! The splice is standard SSA inlining over block-parameter MIR: the callee's values and blocks are
//! copied with fresh ids; the call's block is split so the call's successors live in a new
//! continuation block that takes the result as a parameter; the caller branches into the callee's
//! (copied) entry passing the arguments; and each `ret` in the copy becomes a branch to the
//! continuation carrying the returned value.

use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{BasicBlock, BlockId, Function, Inst, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

use crate::map_op_uses;

/// Largest callee (in instructions) we will inline.
const SIZE_LIMIT: usize = 40;
/// Don't grow any one caller past this many instructions via inlining.
const GROWTH_LIMIT: usize = 5000;

/// Inline small leaf functions throughout the program. Returns whether anything changed.
pub fn inline_program(program: &mut Program) -> bool {
    let names: FxHashSet<Symbol> = program.funcs.iter().map(|f| f.name).collect();

    // Snapshot the bodies eligible to be inlined: small, and leaf (no user-function calls).
    let inlinable: FxHashMap<Symbol, Function> = program
        .funcs
        .iter()
        .filter(|f| is_leaf(f, &names) && func_insts(f) <= SIZE_LIMIT)
        .map(|f| (f.name, f.clone()))
        .collect();
    if inlinable.is_empty() {
        return false;
    }

    let mut changed = false;
    let mut inlined: FxHashSet<Symbol> = FxHashSet::default();
    for ci in 0..program.funcs.len() {
        loop {
            if func_insts(&program.funcs[ci]) > GROWTH_LIMIT {
                break;
            }
            let Some((bi, ii, callee_name)) = find_call_site(&program.funcs[ci], &inlinable) else {
                break;
            };
            // A leaf never calls itself, but guard against inlining a function into itself anyway.
            if callee_name == program.funcs[ci].name {
                break;
            }
            let callee = &inlinable[&callee_name];
            inline_call_site(&mut program.funcs[ci], callee, bi, ii);
            inlined.insert(callee_name);
            changed = true;
        }
    }

    // Drop callees that we inlined and that are now called nowhere. The entry (`main`) is never a
    // call target, so it is never in `inlined` and is always kept — making this safe without
    // knowing which function is the entry.
    if changed {
        let mut still_called: FxHashSet<Symbol> = FxHashSet::default();
        for f in &program.funcs {
            for b in &f.blocks {
                for inst in &b.insts {
                    if let Op::Call { func, .. } = &inst.op {
                        still_called.insert(*func);
                    }
                }
            }
        }
        program
            .funcs
            .retain(|f| !inlined.contains(&f.name) || still_called.contains(&f.name));
    }
    changed
}

/// A function is a leaf if it calls no other user function (intrinsic calls are fine).
fn is_leaf(f: &Function, names: &FxHashSet<Symbol>) -> bool {
    for b in &f.blocks {
        for inst in &b.insts {
            if let Op::Call { func, .. } = &inst.op {
                if names.contains(func) {
                    return false;
                }
            }
        }
    }
    true
}

fn func_insts(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// The first call to an inlinable callee in `f`, as `(block index, inst index, callee name)`.
fn find_call_site(
    f: &Function,
    inlinable: &FxHashMap<Symbol, Function>,
) -> Option<(usize, usize, Symbol)> {
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Op::Call { func, .. } = &inst.op {
                if inlinable.contains_key(func) {
                    return Some((bi, ii, *func));
                }
            }
        }
    }
    None
}

fn inline_call_site(caller: &mut Function, callee: &Function, bi: usize, ii: usize) {
    let (result, args) = match &caller.blocks[bi].insts[ii] {
        Inst {
            result,
            op: Op::Call { args, .. },
        } => (*result, args.clone()),
        _ => unreachable!("call site must be a Call"),
    };

    // Fresh caller value id for every callee value (params included — they receive the call's
    // arguments through the entry block's parameters).
    let mut vmap: FxHashMap<u32, ValueId> = FxHashMap::default();
    for (vid, ty) in callee.value_types.iter().enumerate() {
        let nv = ValueId(caller.value_types.len() as u32);
        caller.value_types.push(ty.clone());
        vmap.insert(vid as u32, nv);
    }
    let mapv = |v: ValueId| -> ValueId { vmap[&v.0] };

    // Callee block b -> caller block (base + b); the continuation block follows them.
    let base = caller.blocks.len() as u32;
    let bmap = |b: BlockId| -> BlockId { BlockId(base + b.0) };
    let cont_id = BlockId(base + callee.blocks.len() as u32);

    // Split the call's block: instructions after the call (plus its terminator) become the
    // continuation, which takes the call's result as a parameter.
    let tail = caller.blocks[bi].insts.split_off(ii + 1);
    caller.blocks[bi].insts.truncate(ii); // drop the call instruction itself
    let orig_term = std::mem::replace(
        &mut caller.blocks[bi].term,
        Terminator::Br {
            target: bmap(callee.entry),
            args, // the call's arguments flow into the callee entry's parameters
        },
    );
    let cont = BasicBlock {
        id: cont_id,
        params: match result {
            Some(r) => vec![r],
            None => vec![],
        },
        insts: tail,
        term: orig_term,
    };

    // `Op::VecKernelCall.kernel` is an index into the *owning* function's `vec_kernels` table, not a
    // `ValueId`, so `map_op_uses` does not (and must not) touch it. Re-parenting the callee's body
    // therefore has to carry its recipes into the caller's table and shift every copied index by
    // where they landed — otherwise the spliced call names one of the caller's own recipes (silent
    // wrong kernel) or an index past the end (dangling). Empty for every non-vectorized callee, so
    // this is a no-op on the common path.
    let kbase = caller.vec_kernels.len() as u32;
    caller
        .vec_kernels
        .extend(callee.vec_kernels.iter().cloned());

    // Copy the callee's blocks with everything remapped; `ret` becomes a branch to the continuation.
    for cb in &callee.blocks {
        let params = cb.params.iter().map(|p| mapv(*p)).collect();
        let insts = cb
            .insts
            .iter()
            .map(|inst| {
                let mut op = inst.op.clone();
                map_op_uses(&mut op, mapv);
                if let Op::VecKernelCall { kernel, .. } = &mut op {
                    *kernel += kbase;
                }
                Inst {
                    result: inst.result.map(mapv),
                    op,
                }
            })
            .collect();
        let term = remap_term(&cb.term, &bmap, &mapv, cont_id);
        caller.blocks.push(BasicBlock {
            id: bmap(cb.id),
            params,
            insts,
            term,
        });
    }
    caller.blocks.push(cont);
}

fn remap_term(
    t: &Terminator,
    bmap: &impl Fn(BlockId) -> BlockId,
    mapv: &impl Fn(ValueId) -> ValueId,
    cont: BlockId,
) -> Terminator {
    let margs = |a: &[ValueId]| a.iter().map(|v| mapv(*v)).collect();
    match t {
        Terminator::Ret(Some(v)) => Terminator::Br {
            target: cont,
            args: vec![mapv(*v)],
        },
        Terminator::Ret(None) => Terminator::Br {
            target: cont,
            args: vec![],
        },
        Terminator::Br { target, args } => Terminator::Br {
            target: bmap(*target),
            args: margs(args),
        },
        Terminator::CondBr {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => Terminator::CondBr {
            cond: mapv(*cond),
            then_blk: bmap(*then_blk),
            then_args: margs(then_args),
            else_blk: bmap(*else_blk),
            else_args: margs(else_args),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}
