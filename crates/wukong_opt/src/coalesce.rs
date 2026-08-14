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

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn opt_mir_of(src: &str, func: &str, level: u8) -> wukong_mir::Function {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        crate::optimize(&mut program, level);
        let sym = interner.intern(func);
        program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"))
            .clone()
    }

    fn hints_for(src: &str, func: &str) -> CoalescePlan {
        analyze_function(&opt_mir_of(src, func, 1))
    }

    fn unit_stride_f32_accesses(plan: &CoalescePlan) -> Vec<(BlockId, usize)> {
        plan.hints
            .iter()
            .filter(|h| {
                matches!(
                    h.hint,
                    CoalesceHint::UnitStride {
                        width: VectorWidth::V4
                    }
                )
            })
            .map(|h| (h.block, h.inst))
            .collect()
    }

    fn strided_accesses(plan: &CoalescePlan) -> Vec<(BlockId, usize)> {
        plan
            .hints
            .iter()
            .filter(|h| matches!(h.hint, CoalesceHint::Strided))
            .map(|h| (h.block, h.inst))
            .collect()
    }

    #[test]
    fn unit_stride_f32_is_v4_candidate() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 2.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let plan = hints_for(src, "k");
        let v4 = unit_stride_f32_accesses(&plan);
        assert!(
            v4.len() >= 2,
            "expected load+store unit-stride f32 hints, got {:?}",
            plan.hints
        );
        assert!(strided_accesses(&plan).is_empty());
    }

    #[test]
    fn column_stride_is_not_coalesced() {
        // One loop: IV `j` steps the address by `n` elements (`a[j*n]`) — the column stride a warp
        // would see if `@parallel` mapped `j` to thread ids.
        let src = "fn col(a: [f32; 64], mut o: [f32; 64], n: i32) { \
                   let mut j: i32 = 0; \
                   while j < n { \
                     o[j] = a[j * 64] + 1.0; \
                     j = j + 1; \
                   } } \
                   fn main() -> i32 { return 0; }";
        let plan = hints_for(src, "col");
        let strided = strided_accesses(&plan);
        assert!(
            !strided.is_empty(),
            "load a[j*64] must be strided (coeff 64), got {:?}",
            plan.hints
        );
        // Store o[j] is unit-stride on the same IV — both hints may coexist on one loop.
        assert!(
            !unit_stride_f32_accesses(&plan).is_empty(),
            "store o[j] should still be a unit-stride v4 candidate"
        );
    }
}
