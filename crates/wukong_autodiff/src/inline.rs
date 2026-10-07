//! Call flattening for differentiation: splice straight-line user functions into the loss before
//! the reverse pass, so a loss written as composed helpers differentiates at every `-O` level.
//!
//! The reverse pass is a single-block transform, so a call is differentiable exactly when its
//! callee is a single block too — and then the cheapest correct VJP of the call is the VJP of the
//! callee's body in the caller's context (JAX traces through Python functions for the same
//! reason). `-O2`'s inliner already splices small callees; this does it for any straight-line
//! callee, at `-O0`/`-O1` too, and independent of the inliner's size budget.
//!
//! The splice is the single-block special case of SSA inlining: the callee's instructions are
//! copied in place of the call with fresh value ids, its parameters map to the call's arguments, and
//! every later use of the call's result is rewritten to the callee's returned value. No blocks are
//! created, so the result stays single-block. Callees are flattened bottom-up (a helper's own
//! helpers first); a recursive cycle, a multi-block callee or one carrying vectorized recipes (whose
//! `VecKernelCall` indexes a function-local table) is left as a call.

use crate::map_operands;
use std::collections::{HashMap, HashSet};
use wukong_mir::{BasicBlock, Function, Inst, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

/// The deepest helper chain spliced. A bound, not a policy: real models nest a handful deep.
const MAX_DEPTH: usize = 32;

/// `f` with every splicable call flattened, or `None` when nothing was spliced.
pub(crate) fn flatten_calls(program: &Program, f: &Function) -> Option<Function> {
    let mut stack = vec![f.name];
    flatten(program, f, &mut stack)
}

fn flatten(program: &Program, f: &Function, stack: &mut Vec<Symbol>) -> Option<Function> {
    if f.blocks.len() != 1 {
        return None;
    }
    let block = &f.blocks[0];
    let has_target = {
        let view: &[Symbol] = stack;
        block.insts.iter().any(|inst| match &inst.op {
            Op::Call { func, args } => callee(program, *func, args.len(), view).is_some(),
            _ => false,
        })
    };
    if !has_target {
        return None;
    }
    let mut value_types = f.value_types.clone();
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let mut insts: Vec<Inst> = Vec::with_capacity(block.insts.len());
    let mut changed = false;
    for inst in &block.insts {
        let op = map_operands(&inst.op, |v| *subst.get(&v).unwrap_or(&v));
        if let Op::Call { func, args } = &op {
            if let Some(cf) = callee(program, *func, args.len(), stack) {
                stack.push(*func);
                let flat = flatten(program, cf, stack);
                stack.pop();
                let body = flat.as_ref().unwrap_or(cf);
                if splicable(body, inst.result.is_some()) {
                    let ret = splice(body, args, &mut value_types, &mut insts);
                    if let (Some(r), Some(rv)) = (inst.result, ret) {
                        subst.insert(r, rv);
                    }
                    changed = true;
                    continue;
                }
            }
        }
        insts.push(Inst {
            result: inst.result,
            op,
        });
    }
    if !changed {
        return None;
    }
    let mapv = |v: ValueId| *subst.get(&v).unwrap_or(&v);
    let term = match &block.term {
        Terminator::Ret(v) => Terminator::Ret(v.map(mapv)),
        Terminator::Br { target, args } => Terminator::Br {
            target: *target,
            args: args.iter().map(|&a| mapv(a)).collect(),
        },
        Terminator::CondBr {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => Terminator::CondBr {
            cond: mapv(*cond),
            then_blk: *then_blk,
            then_args: then_args.iter().map(|&a| mapv(a)).collect(),
            else_blk: *else_blk,
            else_args: else_args.iter().map(|&a| mapv(a)).collect(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    };
    Some(Function {
        name: f.name,
        params: f.params.clone(),
        ret: f.ret.clone(),
        blocks: vec![BasicBlock {
            id: block.id,
            params: block.params.clone(),
            insts,
            term,
        }],
        value_types,
        entry: f.entry,
        vec_kernels: f.vec_kernels.clone(),
    })
}

/// The program function a call names, if it may be spliced at this depth: defined in `program`,
/// not already on the splice stack (recursion), and called with its own arity.
fn callee<'p>(
    program: &'p Program,
    func: Symbol,
    nargs: usize,
    stack: &[Symbol],
) -> Option<&'p Function> {
    if stack.len() >= MAX_DEPTH || stack.contains(&func) {
        return None;
    }
    let cf = program.function(func)?;
    (cf.params.len() == nargs).then_some(cf)
}

/// Whether `body` can replace a call in place: one block, no function-local vector recipes, a
/// `ret` whose shape matches whether the call's result is used, and every operand defined in the
/// body (verified MIR guarantees it; checked so a malformed callee is left as a call rather than
/// spliced with a dangling value).
fn splicable(body: &Function, wants_result: bool) -> bool {
    if body.blocks.len() != 1 || !body.vec_kernels.is_empty() {
        return false;
    }
    let block = &body.blocks[0];
    let ret = match block.term {
        Terminator::Ret(v) => v,
        _ => return false,
    };
    if wants_result && ret.is_none() {
        return false;
    }
    let mut defined: HashSet<ValueId> = body.params.iter().copied().collect();
    let mut ok = true;
    for inst in &block.insts {
        let _ = map_operands(&inst.op, |v| {
            if !defined.contains(&v) {
                ok = false;
            }
            v
        });
        if let Some(r) = inst.result {
            defined.insert(r);
        }
    }
    ok && ret.is_none_or(|v| defined.contains(&v))
}

/// Copy `body`'s instructions into `insts` with fresh value ids (its parameters bound to `args`),
/// returning the caller-side value of its `ret` operand.
fn splice(
    body: &Function,
    args: &[ValueId],
    value_types: &mut Vec<wukong_mir::MirType>,
    insts: &mut Vec<Inst>,
) -> Option<ValueId> {
    let block = &body.blocks[0];
    let mut vmap: HashMap<ValueId, ValueId> = HashMap::new();
    for (&p, &a) in body.params.iter().zip(args) {
        vmap.insert(p, a);
    }
    for inst in &block.insts {
        if let Some(r) = inst.result {
            let nv = ValueId(value_types.len() as u32);
            value_types.push(body.value_type(r).clone());
            vmap.insert(r, nv);
        }
    }
    for inst in &block.insts {
        let op = map_operands(&inst.op, |v| vmap[&v]);
        insts.push(Inst {
            result: inst.result.map(|r| vmap[&r]),
            op,
        });
    }
    match block.term {
        Terminator::Ret(Some(v)) => Some(vmap[&v]),
        _ => None,
    }
}
