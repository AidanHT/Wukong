//! Legalize compact boolean casts without requiring an unsupported CLIF vector width.
//!
//! Only functions containing a float-to-boolean-vector cast take this path. Their vector SSA
//! values become individual scalar lanes, including block parameters and memory accesses. This
//! preserves MIR's byte-per-boolean layout and reuses scalar i32/u32 saturation before parity.
//! Function ABIs are unchanged: vector parameters/returns and vector-valued calls are declined.

use std::borrow::Cow;
use wukong_mir::{CastKind, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId};
use wukong_span::Interner;

fn needs_legalization(f: &Function) -> bool {
    f.blocks.iter().flat_map(|b| &b.insts).any(|i| {
        matches!(&i.op, Op::Cast(CastKind::FpToSi | CastKind::FpToUi, _, MirType::Vec(lane, _))
            if **lane == MirType::I1)
    })
}

/// Legalize one function for codegen: borrowed unchanged unless it holds a float-to-`i1`-vector
/// cast, in which case only this function is copied and scalarized. Called per function from
/// `build_function_clif`, so the copy happens on that function's codegen worker and the rest of the
/// program is never cloned. Scalarization leaves the signature, every call and every function
/// address untouched, so module-level declarations and the assert analysis see the same program.
pub(super) fn legalize<'a>(
    f: &'a Function,
    interner: &Interner,
) -> Result<Cow<'a, Function>, String> {
    if !needs_legalization(f) {
        return Ok(Cow::Borrowed(f));
    }
    let mut lowered = f.clone();
    scalarize(&mut lowered).map_err(|why| {
        format!(
            "unsupported compact boolean vector legalization in `{}`: {why}",
            interner.resolve(f.name)
        )
    })?;
    Ok(Cow::Owned(lowered))
}

fn fresh(types: &mut Vec<MirType>, ty: MirType) -> ValueId {
    let id = ValueId(types.len() as u32);
    types.push(ty);
    id
}

fn emit(types: &mut Vec<MirType>, out: &mut Vec<Inst>, ty: MirType, op: Op) -> ValueId {
    let result = fresh(types, ty);
    out.push(Inst {
        result: Some(result),
        op,
    });
    result
}

fn lane_ptr(
    types: &mut Vec<MirType>,
    out: &mut Vec<Inst>,
    ptr: ValueId,
    lane: usize,
    ty: &MirType,
) -> ValueId {
    let index = emit(
        types,
        out,
        MirType::I64,
        Op::ConstInt(lane as i128, MirType::I64),
    );
    emit(
        types,
        out,
        MirType::Ptr,
        Op::Gep {
            ptr,
            index,
            elem: ty.clone(),
        },
    )
}

fn check_shape(lane: &MirType, n: u32) -> Result<(), String> {
    if n == 0 || n > 16 {
        return Err("compact boolean legalization supports 1 through 16 lanes".into());
    }
    if !lane.is_int() && !matches!(lane, MirType::F32 | MirType::F64) {
        return Err("only integer, f32 and f64 vector lanes are supported".into());
    }
    Ok(())
}

fn scalarize(f: &mut Function) -> Result<(), String> {
    if f.ret.is_vector() || f.params.iter().any(|p| f.value_type(*p).is_vector()) {
        return Err("vector function parameters or returns require a vector ABI".into());
    }
    // Allocate lane IDs before visiting blocks: dominance need not follow block layout order,
    // and loops may pass vector values through backedges.
    let original_types = f.value_types.clone();
    let mut lanes = Vec::with_capacity(original_types.len());
    for (i, ty) in original_types.iter().enumerate() {
        if let MirType::Vec(lane, n) = ty {
            check_shape(lane, *n)?;
            lanes.push(
                (0..*n)
                    .map(|_| fresh(&mut f.value_types, (**lane).clone()))
                    .collect::<Vec<_>>(),
            );
        } else {
            lanes.push(vec![ValueId(i as u32)]);
        }
    }
    let expand = |args: &[ValueId]| {
        args.iter()
            .flat_map(|v| lanes[v.0 as usize].iter().copied())
            .collect()
    };
    for block in &mut f.blocks {
        block.params = expand(&block.params);
        let mut out = Vec::new();
        for inst in std::mem::take(&mut block.insts) {
            // The MIR verifier requires a result for every value-producing operation.
            // Keep direct backend callers on an error path if they bypass that precondition.
            if inst.result.is_none()
                && !matches!(
                    inst.op,
                    Op::Store { .. } | Op::Call { .. } | Op::VecKernelCall { .. } | Op::Barrier
                )
            {
                return Err("value-producing operation has no result".into());
            }
            let result_ty = inst.result.map(|r| &original_types[r.0 as usize]);
            if let Some(MirType::Vec(ty, n)) = result_ty {
                let result = inst.result.unwrap();
                for k in 0..*n as usize {
                    let lane = |v: ValueId| -> Result<ValueId, String> {
                        if !original_types[v.0 as usize].is_vector()
                            || lanes[v.0 as usize].len() != *n as usize
                        {
                            return Err(
                                "casts that change vector lane count require repacking".into()
                            );
                        }
                        Ok(lanes[v.0 as usize][k])
                    };
                    let op = match &inst.op {
                        Op::Load(ptr, _) => {
                            let ptr = lane_ptr(&mut f.value_types, &mut out, *ptr, k, ty);
                            Op::Load(ptr, (**ty).clone())
                        }
                        Op::Splat(v) => Op::Cast(CastKind::Bitcast, *v, (**ty).clone()),
                        Op::Iota(_) => Op::ConstInt(k as i128, (**ty).clone()),
                        Op::Bin(op, a, b) => Op::Bin(*op, lane(*a)?, lane(*b)?),
                        Op::Neg(v) => Op::Neg(lane(*v)?),
                        Op::Not(v) => Op::Not(lane(*v)?),
                        Op::Cast(kind, v, _) => Op::Cast(*kind, lane(*v)?, (**ty).clone()),
                        Op::Fma(a, b, c) => Op::Fma(lane(*a)?, lane(*b)?, lane(*c)?),
                        Op::Sqrt(v) => Op::Sqrt(lane(*v)?),
                        Op::Round(mode, v) => Op::Round(*mode, lane(*v)?),
                        Op::Cmp(op, a, b) => {
                            if !ty.is_int() {
                                return Err("comparison masks must have integer lanes".into());
                            }
                            let cmp = Op::Cmp(*op, lane(*a)?, lane(*b)?);
                            if **ty == MirType::I1 {
                                cmp
                            } else {
                                // The interpreter's vector comparison yields 0/1, not a full
                                // register mask. Scalar selects below test nonzero explicitly.
                                let bit = emit(&mut f.value_types, &mut out, MirType::I1, cmp);
                                Op::Cast(CastKind::ZExt, bit, (**ty).clone())
                            }
                        }
                        Op::Select(c, a, b) => {
                            let cty = original_types[c.0 as usize].lane_type();
                            let zero_op = if cty.is_float() {
                                Op::ConstFloat(0.0, cty.clone())
                            } else {
                                Op::ConstInt(0, cty.clone())
                            };
                            let zero = emit(&mut f.value_types, &mut out, cty.clone(), zero_op);
                            let cmp = if cty.is_float() {
                                CmpOp::Fone
                            } else {
                                CmpOp::Ne
                            };
                            let bit = emit(
                                &mut f.value_types,
                                &mut out,
                                MirType::I1,
                                Op::Cmp(cmp, lane(*c)?, zero),
                            );
                            Op::Select(bit, lane(*a)?, lane(*b)?)
                        }
                        Op::Call { .. } => {
                            return Err("vector-valued calls require a vector ABI".into())
                        }
                        _ => return Err(format!("unsupported vector operation: {:?}", inst.op)),
                    };
                    out.push(Inst {
                        result: Some(lanes[result.0 as usize][k]),
                        op,
                    });
                }
            } else {
                match &inst.op {
                    Op::Store { ptr, value } if original_types[value.0 as usize].is_vector() => {
                        let ty = original_types[value.0 as usize].lane_type();
                        for (k, value) in lanes[value.0 as usize].iter().enumerate() {
                            let ptr = lane_ptr(&mut f.value_types, &mut out, *ptr, k, ty);
                            out.push(Inst {
                                result: None,
                                op: Op::Store { ptr, value: *value },
                            });
                        }
                    }
                    Op::ExtractLane(v, k) => out.push(Inst {
                        result: inst.result,
                        op: Op::Cast(
                            CastKind::Bitcast,
                            lanes[v.0 as usize][*k as usize],
                            result_ty.unwrap().clone(),
                        ),
                    }),
                    Op::Cast(_, v, _) if original_types[v.0 as usize].is_vector() => {
                        return Err(
                            "casts between scalar and vector values require repacking".into()
                        );
                    }
                    Op::Call { args, .. }
                        if args
                            .iter()
                            .any(|v| original_types[v.0 as usize].is_vector()) =>
                    {
                        return Err("vector-valued calls require a vector ABI".into());
                    }
                    _ => out.push(inst),
                }
            }
        }
        block.insts = out;
        match &mut block.term {
            Terminator::Br { args, .. } => *args = expand(args),
            Terminator::CondBr {
                then_args,
                else_args,
                ..
            } => {
                *then_args = expand(then_args);
                *else_args = expand(else_args);
            }
            Terminator::Ret(_) | Terminator::Unreachable => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Builder;

    fn ret_zero(b: &mut Builder) {
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(zero));
    }

    /// Legalization is per function: a function without a compact boolean cast is borrowed (no
    /// copy), only a function with one is copied and scalarized, and its input is left untouched.
    #[test]
    fn legalize_copies_only_the_function_that_needs_it() {
        let mut interner = Interner::new();
        let mut b = Builder::new(interner.intern("plain"), MirType::I32);
        ret_zero(&mut b);
        let plain = b.finish();
        let out = legalize(&plain, &interner).unwrap();
        assert!(matches!(out, Cow::Borrowed(g) if std::ptr::eq(g, &plain)));

        let src = MirType::Vec(Box::new(MirType::F32), 4);
        let dst = MirType::Vec(Box::new(MirType::I1), 4);
        let mut b = Builder::new(interner.intern("casts"), MirType::I32);
        let array = b.alloca(MirType::Array(Box::new(MirType::F32), 4));
        let values = b.build(src.clone(), Op::Load(array, src.clone()));
        let mask = b.build(dst.clone(), Op::Cast(CastKind::FpToSi, values, dst.clone()));
        let lane = b.build(MirType::I1, Op::ExtractLane(mask, 2));
        let wide = b.build(MirType::I32, Op::Cast(CastKind::ZExt, lane, MirType::I32));
        b.ret(Some(wide));
        let casts = b.finish();
        let out = legalize(&casts, &interner).unwrap();
        assert!(matches!(out, Cow::Owned(_)));
        assert!(!needs_legalization(&out));
        let vector_results = out
            .blocks
            .iter()
            .flat_map(|blk| &blk.insts)
            .filter(|i| i.result.iter().any(|r| out.value_type(*r).is_vector()))
            .count();
        assert_eq!(vector_results, 0, "every vector value became scalar lanes");
        assert_eq!(out.params, casts.params);
        assert_eq!(out.ret, casts.ret);
        assert!(
            needs_legalization(&casts),
            "the input function is not modified"
        );

        // A legalization failure names its function.
        let mut b = Builder::new(interner.intern("vec_abi"), MirType::I32);
        let param = b.add_param(src);
        b.build(dst.clone(), Op::Cast(CastKind::FpToUi, param, dst));
        ret_zero(&mut b);
        let err = legalize(&b.finish(), &interner).unwrap_err();
        assert!(
            err.contains("`vec_abi`") && err.contains("vector ABI"),
            "{err}"
        );
    }
}
