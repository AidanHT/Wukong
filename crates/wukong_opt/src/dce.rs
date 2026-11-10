//! Dead-code elimination: retain the dependency closure of observable instructions and terminators.
//!
//! A use-count sweep removes only the last instruction in a dead chain, requiring another complete
//! optimizer sweep for each preceding link. Marking backwards from the instructions we must keep
//! removes the entire chain in one invocation. Each value is visited at most once, irrespective of
//! block layout, so the work is linear in the value arena, instructions and operand uses.
//!
//! A call to a user function whose summary says it has no side effect and always returns (see the
//! `effects` module) is as removable as arithmetic: it is not a root, and it goes once nothing
//! reads its result — or immediately, when it has none.

use wukong_mir::{Function, Op, ValueId};

use crate::{each_op_use, each_term_use, CfgAnalyses, Pass};

pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let effects = cache.call_effects();
        let removable_call = |op: &Op| match op {
            Op::Call { func, .. } => effects.as_ref().is_some_and(|e| e.removable(*func)),
            _ => false,
        };
        // Must this instruction stay even if nothing reads its result?
        let rooted = |op: &Op, has_result: bool| {
            !removable_call(op) && (!has_result || op.has_side_effects())
        };
        let mut definitions = vec![None; f.value_types.len()];
        let mut live = vec![false; f.value_types.len()];
        let mut work = Vec::new();
        let mark = |v: ValueId, live: &mut [bool], work: &mut Vec<ValueId>| {
            if !live[v.0 as usize] {
                live[v.0 as usize] = true;
                work.push(v);
            }
        };
        for b in &f.blocks {
            for inst in &b.insts {
                if let Some(result) = inst.result {
                    definitions[result.0 as usize] = Some(&inst.op);
                }
                // Preserve the existing effect classification, including instructions without a
                // result. DCE does not introduce a new policy for potentially trapping operations.
                if rooted(&inst.op, inst.result.is_some()) {
                    each_op_use(&inst.op, &mut |v| mark(v, &mut live, &mut work));
                }
            }
            // Edge arguments remain roots: their matching parameters are owned by SimplifyPhis.
            // Keeping them also makes this pass correct when used independently of that pass.
            each_term_use(&b.term, &mut |v| mark(v, &mut live, &mut work));
        }
        while let Some(v) = work.pop() {
            if let Some(op) = definitions[v.0 as usize] {
                each_op_use(op, &mut |operand| mark(operand, &mut live, &mut work));
            }
        }

        let mut changed = false;
        for b in &mut f.blocks {
            let before = b.insts.len();
            b.insts.retain(|inst| match inst.result {
                Some(r) => live[r.0 as usize] || rooted(&inst.op, true),
                None => rooted(&inst.op, false),
            });
            if b.insts.len() != before {
                changed = true;
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, Builder, MirType, Op};
    use wukong_span::Interner;

    #[test]
    fn removes_a_dead_chain_beyond_the_pipeline_iteration_limit_in_one_pass() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("dead_chain"), MirType::I64);
        let input = b.add_param(MirType::I64);
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let mut dead = input;
        for _ in 0..256 {
            dead = b.build(MirType::I64, Op::Bin(BinOp::Add, dead, one));
        }
        b.ret(Some(input));
        let mut f = b.finish();
        let mut cache = CfgAnalyses::default();

        assert!(Dce.run_function(&mut f, &mut cache));
        assert!(f.blocks[0].insts.is_empty());
        assert!(!Dce.run_function(&mut f, &mut cache));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
    }

    #[test]
    fn preserves_effect_dependencies_and_arguments_across_block_layout() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("roots"), MirType::I64);
        let ptr = b.add_param(MirType::Ptr);
        let input = b.add_param(MirType::I64);
        let exit = b.new_block();
        let body = b.new_block();
        let output = b.block_param(exit, MirType::I64);
        b.br(body, vec![]);
        b.switch_to(body);
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let sum = b.build(MirType::I64, Op::Bin(BinOp::Add, input, one));
        b.build_void(Op::Store { ptr, value: sum });
        let call = it.intern("observable");
        b.build(
            MirType::I64,
            Op::Call {
                func: call,
                args: vec![sum],
            },
        );
        let argument = b.build(MirType::I64, Op::Bin(BinOp::Mul, sum, sum));
        b.build(MirType::I64, Op::Not(argument));
        b.br(exit, vec![argument]);
        b.switch_to(exit);
        b.ret(Some(output));
        let mut f = b.finish();

        assert!(Dce.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(f.blocks[body.0 as usize].insts.len(), 5);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
    }
}
