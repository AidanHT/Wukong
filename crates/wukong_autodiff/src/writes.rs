//! Which buffer parameters a function may write — the fact the `--train` loop needs to run the
//! gradient on the live buffers instead of on a fresh copy of every buffer at every step.
//!
//! The analysis is the tape's own knowledge of the kernels: a recognized tensor kernel writes
//! exactly its output operand (`tape::Kernel::out_index`) and only reads the rest, so a weight
//! passed to `wukong_sgemm_nt` is provably left unchanged while the buffer it writes is not.
//! Everything the tape cannot see through is conservatively a write.

use crate::coverage::is_effect_only;
use crate::map_operands;
use crate::tape::kernel_by_name;
use std::collections::HashMap;
use wukong_mir::{Function, Op, Terminator, ValueId};
use wukong_span::Interner;

/// One flag per parameter of `func`, in parameter order: `true` when calling `func` may change the
/// memory that parameter points into.
///
/// A `false` is a proof. Every use of the parameter — and of every `gep` derived from it, in any
/// block — is the address of a `load`, the base of a further `gep`, an input operand of a
/// recognized tensor kernel (`tape::FORWARD_KERNELS`; its output operand is a write), or an
/// argument of an effect-only call that reads its buffers (`print`, `assert`, a file write).
/// Anything else is conservatively a write: a `store` through the pointer, a call the tape does not
/// recognize (or one whose argument count is not the kernel's ABI), and every use that lets the
/// pointer escape — stored as a value, cast to an integer, selected, compared, returned or passed
/// along a branch.
pub fn written_params(func: &Function, interner: &Interner) -> Vec<bool> {
    // The parameter each pointer derives from through `gep` chains. Blocks need not be in dominance
    // order, so propagate to a fixpoint (each pass only adds entries; a chain of length L settles
    // in at most L passes).
    let mut root: HashMap<ValueId, usize> = func
        .params
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();
    loop {
        let mut grew = false;
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            if let (Some(r), Op::Gep { ptr, .. }) = (inst.result, &inst.op) {
                if let Some(&i) = root.get(ptr) {
                    grew |= root.insert(r, i).is_none();
                }
            }
        }
        if !grew {
            break;
        }
    }
    let mut written = vec![false; func.params.len()];
    let mut mark = |v: ValueId| {
        if let Some(&i) = root.get(&v) {
            written[i] = true;
        }
    };
    for block in &func.blocks {
        for inst in &block.insts {
            match &inst.op {
                // A read through the address, or a further derivation (whose own uses are
                // classified where they occur). A `gep` index is an integer, never a buffer.
                Op::Load(..) | Op::Gep { .. } => {}
                Op::Store { ptr, value } => {
                    mark(*ptr);
                    mark(*value);
                }
                Op::Call { func: callee, args } => {
                    let name = interner.resolve(*callee);
                    // `print`, `assert`, a file write: they read their buffers. Releasing one is
                    // not a read, so `wukong_rt_free` stays a (conservative) write.
                    let reads_only = is_effect_only(name) && name != "wukong_rt_free";
                    match kernel_by_name(name) {
                        Some(kind) if kind.arity() == args.len() => {
                            if let Some(oi) = kind.out_index() {
                                mark(args[oi]);
                            }
                        }
                        _ if reads_only => {}
                        // A kernel symbol with a foreign signature, or any other call.
                        _ => args.iter().for_each(|&a| mark(a)),
                    }
                }
                other => {
                    let _ = map_operands(other, |v| {
                        mark(v);
                        v
                    });
                }
            }
        }
        match &block.term {
            Terminator::Ret(v) => v.iter().for_each(|&v| mark(v)),
            Terminator::Br { args, .. } => args.iter().for_each(|&a| mark(a)),
            Terminator::CondBr {
                then_args,
                else_args,
                ..
            } => then_args.iter().chain(else_args).for_each(|&a| mark(a)),
            Terminator::Switch {
                cases,
                default_args,
                ..
            } => cases
                .iter()
                .flat_map(|c| &c.args)
                .chain(default_args)
                .for_each(|&a| mark(a)),
            Terminator::Unreachable => {}
        }
    }
    written
}
