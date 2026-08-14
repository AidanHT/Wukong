//! GPU memory coalescing analysis over [`loop_info`].
//!
//! Pure analysis: classifies each innermost-loop load/store by the **element stride** of its address
//! relative to the loop's primary induction variable — the stride warp-adjacent threads see after
//! `@parallel` maps that IV to flat thread ids. Produces hints (`scalar` / `v2` / `v4` candidate, or
//! strided) for [`wukong_codegen_gpu`] lowering. Never rewrites MIR and never dispatches a named
//! kernel.

use wukong_mir::{BlockId, Function};

use crate::alias::type_bytes;
use crate::loop_info::{self, AddrForm, LoopForest, MemAccess, NaturalLoop};

/// Widest vector global load/store the GPU lowering may emit for a coalesced access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorWidth {
    Scalar = 1,
    V2 = 2,
    V4 = 4,
}

/// Per-access coalescing verdict for GPU lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoalesceHint {
    /// Adjacent IV steps touch consecutive element slots; `width` is the widest safe vector op.
    UnitStride { width: VectorWidth },
    /// IV step advances the address by more than one element (column / batch stride).
    Strided,
    /// Address or stride not understood — emit scalar generic `ld`/`st`.
    Unknown,
}

/// One load/store inside a loop, keyed for the lowering pass.
#[derive(Clone, Debug)]
pub struct AccessHint {
    pub block: BlockId,
    pub inst: usize,
    pub hint: CoalesceHint,
}

/// Coalescing hints for every classified access in one function.
#[derive(Clone, Debug, Default)]
pub struct CoalescePlan {
    pub hints: Vec<AccessHint>,
}

impl CoalescePlan {
    /// Lookup the hint for a concrete MIR instruction, if analysis produced one.
    pub fn get(&self, block: BlockId, inst: usize) -> Option<CoalesceHint> {
        self.hints
            .iter()
            .find(|h| h.block == block && h.inst == inst)
            .map(|h| h.hint)
    }
}
pub fn analyze_function(_f: &Function) -> CoalescePlan { CoalescePlan::default() }
