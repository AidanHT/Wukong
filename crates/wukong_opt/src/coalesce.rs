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

/// Analyze every natural loop of `f` and record coalescing hints for its memory accesses.
pub fn analyze_function(f: &Function) -> CoalescePlan {
    let forest = loop_info::analyze_function(f);
    analyze_forest(&forest)
}

fn analyze_forest(forest: &LoopForest) -> CoalescePlan {
    let mut hints = Vec::new();
    for l in &forest.loops {
        for access in &l.accesses {
            let hint = hint_for_access(l, access);
            if !matches!(hint, CoalesceHint::Unknown) {
                hints.push(AccessHint {
                    block: access.block,
                    inst: access.inst,
                    hint,
                });
            }
        }
    }
    CoalescePlan { hints }
}

fn hint_for_access(l: &NaturalLoop, access: &MemAccess) -> CoalesceHint {
  // Only classify accesses whose affine index is expressed against this loop's primary IV.
  if l.primary_iv.is_none() {
    return CoalesceHint::Unknown;
  }
  match &access.addr {
    AddrForm::Affine { index, elem, .. } => {
      let elem_bytes = type_bytes(elem)
        .or_else(|| type_bytes(&access.value_ty))
        .unwrap_or(0) as u64;
      if elem_bytes == 0 {
        return CoalesceHint::Unknown;
      }
      let stride_elems = iv_stride_elems(index);
      match stride_elems {
        None => CoalesceHint::Unknown,
        Some(None) => CoalesceHint::Strided,
        Some(Some(0)) => CoalesceHint::Unknown,
        Some(Some(1)) => CoalesceHint::UnitStride {
          width: vector_width_for(elem_bytes),
        },
        Some(Some(_)) => CoalesceHint::Strided,
      }
    }
    AddrForm::Invariant(_) | AddrForm::Unknown => CoalesceHint::Unknown,
  }
}

/// Element stride per primary-IV step. `Coeff::Sym` is a runtime stride (e.g. symbolic `N` in
/// `i*N`) and is treated as strided — only a compile-time unit stride is a vector candidate.
fn iv_stride_elems(index: &loop_info::AffineExpr) -> Option<Option<i64>> {
  match index.coeff {
    loop_info::Coeff::Zero => Some(Some(0)),
    loop_info::Coeff::Const(k) => Some(Some(k)),
    loop_info::Coeff::Sym(..) => Some(None), // runtime stride → strided, not unit
  }
}

/// Pick the widest 128-bit vector op that fits `elem_bytes` lanes.
fn vector_width_for(elem_bytes: u64) -> VectorWidth {
  match elem_bytes {
    1 | 2 | 4 => VectorWidth::V4,
    8 => VectorWidth::V2,
    _ => VectorWidth::Scalar,
  }
}
