//! What executing an operation may do besides computing its result: the one effect classification
//! every optimizer pass asks.
//!
//! A pass must not delete, duplicate, reorder, value-number or speculate an instruction whose effect
//! it would change. Before this module each pass kept its own list of the opcodes it considered
//! effectful — `Call | VecKernelCall | AtomicAdd` in one, `Call | VecKernelCall | Barrier` in
//! another — next to a permissive `_` arm, so a new effectful `Op` that one list forgot looked pure
//! *there* and was moved or deleted with no gate able to see it (both execution backends agree on
//! the miscompiled MIR).
//!
//! [`Op::effects`] is now the single classification: one match with **no wildcard arm**, so a new
//! variant does not compile until it is classified here, and every pass that asks
//! [`Op::may_write_memory`] or [`Op::has_side_effects`] sees it without being edited. Passes keep
//! their own opcode arms only for what they model precisely (a `store`'s one destination, a `call`
//! whose callee has an effect summary); everything else they ask of this classification, which
//! errs on the side of "has the effect".

use crate::{BinOp, Op};

/// The effects of one operation. Every field is a *may*: `false` is a proof that the operation
/// never does it, `true` only that it might.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Effects {
    /// Reads memory through its own pointer operand (`load`, `atomic.add`).
    pub reads: bool,
    /// Writes memory through its own pointer operand (`store`, `atomic.add`).
    pub writes: bool,
    /// Runs code outside this instruction — a direct or indirect call, or a synthesized vector
    /// kernel. The effect is opaque: it may read and write any memory reachable from an escaped
    /// address or from a pointer it was handed, perform I/O, trap, or never return. A pass that
    /// knows more about one particular callee (`wukong_opt`'s call summaries) may refine it.
    pub calls: bool,
    /// May stop the program (a fault) for some operand values: a memory access, an integer
    /// division or remainder by zero.
    pub traps: bool,
    /// Orders memory between threads (`barrier`): another thread may write shared memory across
    /// it.
    pub syncs: bool,
    /// Allocates storage: every execution yields a distinct address (`alloca`), so two executions
    /// are never interchangeable.
    pub allocates: bool,
}

impl Effects {
    /// No effect at all: the result is a function of the operand values alone.
    pub const NONE: Effects = Effects {
        reads: false,
        writes: false,
        calls: false,
        traps: false,
        syncs: false,
        allocates: false,
    };
}

impl Op {
    /// The effect classification of this operation. Deliberately exhaustive — never add a `_`
    /// arm: an unclassified variant must be a compile error here, not a silently pure op in every
    /// pass that asks.
    pub fn effects(&self) -> Effects {
        let none = Effects::NONE;
        match self {
            Op::ConstInt(..)
            | Op::ConstFloat(..)
            | Op::Cmp(..)
            | Op::Neg(..)
            | Op::Not(..)
            | Op::Cast(..)
            | Op::Select(..)
            | Op::Gep { .. }
            | Op::FuncAddr(..)
            | Op::GlobalAddr(..)
            | Op::Splat(..)
            | Op::ExtractLane(..)
            | Op::Iota(..)
            | Op::Widen { .. }
            | Op::Narrow { .. }
            | Op::Shuffle { .. }
            | Op::InsertLane(..)
            | Op::Fma(..)
            | Op::Sqrt(..)
            | Op::Round(..)
            | Op::IntUnary(..) => none,
            // Integer division and remainder trap on a zero divisor (and `MIN / -1` on x86).
            Op::Bin(b, ..) => Effects {
                traps: matches!(b, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem),
                ..none
            },
            Op::Alloca { .. } => Effects {
                allocates: true,
                ..none
            },
            Op::Load(..) => Effects {
                reads: true,
                traps: true,
                ..none
            },
            Op::Store { .. } => Effects {
                writes: true,
                traps: true,
                ..none
            },
            Op::AtomicAdd { .. } => Effects {
                reads: true,
                writes: true,
                traps: true,
                ..none
            },
            Op::Barrier => Effects {
                syncs: true,
                ..none
            },
            Op::Call { .. } | Op::VecKernelCall { .. } => Effects {
                calls: true,
                ..none
            },
        }
    }

    /// Runs code outside this instruction ([`Effects::calls`]): a call of any kind, whose memory
    /// effect is opaque.
    pub fn is_call(&self) -> bool {
        self.effects().calls
    }

    /// May read memory: through its own pointer operand, or anything a call can reach.
    pub fn may_read_memory(&self) -> bool {
        let e = self.effects();
        e.reads || e.calls
    }

    /// May write memory: through its own pointer operand, or anything a call can reach.
    pub fn may_write_memory(&self) -> bool {
        let e = self.effects();
        e.writes || e.calls
    }

    /// May trap for some operand values (a call may, too).
    pub fn may_trap(&self) -> bool {
        let e = self.effects();
        e.traps || e.calls
    }

    /// Must stay even when nothing reads its result: it writes memory, calls out, or orders memory
    /// between threads. (A load or a division whose result is unused may go: the optimizer does not
    /// preserve a trap whose only purpose was to be observed.)
    pub fn has_side_effects(&self) -> bool {
        let e = self.effects();
        e.writes || e.calls || e.syncs
    }

    /// No effect at all ([`Effects::NONE`]): the result is a function of the operands alone, so
    /// the operation may be value-numbered, hoisted, duplicated or deleted freely.
    pub fn is_pure(&self) -> bool {
        self.effects() == Effects::NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddrSpace, MirType, ValueId};
    use wukong_span::Symbol;

    fn v(n: u32) -> ValueId {
        ValueId(n)
    }

    /// The sets the optimizer's passes used to spell out opcode by opcode, rebuilt from the
    /// predicates: the refactor that routed them here must not change one member.
    #[test]
    fn the_predicates_reproduce_the_sets_the_passes_listed() {
        let call = Op::Call {
            func: Symbol(0),
            args: vec![v(1)],
        };
        let kernel = Op::VecKernelCall {
            kernel: 0,
            ptrs: v(1),
            scalars: v(2),
            n: v(3),
        };
        let store = Op::Store {
            ptr: v(1),
            value: v(2),
        };
        let atomic = Op::AtomicAdd {
            ptr: v(1),
            value: v(2),
        };
        let load = Op::Load(v(1), MirType::I32);
        let alloca = Op::Alloca {
            ty: MirType::I32,
            space: AddrSpace::Local,
            align: 0,
        };
        let div = Op::Bin(BinOp::SDiv, v(1), v(2));
        let add = Op::Bin(BinOp::Add, v(1), v(2));
        let all = [
            &call,
            &kernel,
            &store,
            &atomic,
            &load,
            &alloca,
            &div,
            &add,
            &Op::Barrier,
        ];
        let set = |p: &dyn Fn(&Op) -> bool| -> Vec<usize> {
            all.iter()
                .enumerate()
                .filter(|(_, op)| p(op))
                .map(|(i, _)| i)
                .collect()
        };
        // has_side_effects: Store | Call | VecKernelCall | Barrier | AtomicAdd.
        assert_eq!(set(&|op| op.has_side_effects()), vec![0, 1, 2, 3, 8]);
        // A call of any kind.
        assert_eq!(set(&|op| op.is_call()), vec![0, 1]);
        // Writes memory: Store | AtomicAdd | the calls.
        assert_eq!(set(&|op| op.may_write_memory()), vec![0, 1, 2, 3]);
        // Reads memory: Load | AtomicAdd | the calls.
        assert_eq!(set(&|op| op.may_read_memory()), vec![0, 1, 3, 4]);
        // Traps: every access, the calls and an integer division.
        assert_eq!(set(&|op| op.may_trap()), vec![0, 1, 2, 3, 4, 6]);
        // Pure: only the add.
        assert_eq!(set(&|op| op.is_pure()), vec![7]);
    }
}
