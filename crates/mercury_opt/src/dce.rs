//! Dead-code elimination: remove pure instructions whose results are never used.

use crate::fxhash::FxHashSet;

use mercury_mir::Function;

use crate::{each_op_use, each_term_use, has_side_effects, CfgAnalyses, Pass};

pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run_function(&self, f: &mut Function, _cache: &mut CfgAnalyses) -> bool {
        let mut used: FxHashSet<u32> = FxHashSet::default();
        for b in &f.blocks {
            for inst in &b.insts {
                each_op_use(&inst.op, &mut |v| {
                    used.insert(v.0);
                });
            }
            each_term_use(&b.term, &mut |v| {
                used.insert(v.0);
            });
        }

        let mut changed = false;
        for b in &mut f.blocks {
            let before = b.insts.len();
            b.insts.retain(|inst| match inst.result {
                Some(r) => used.contains(&r.0) || has_side_effects(&inst.op),
                None => true,
            });
            if b.insts.len() != before {
                changed = true;
            }
        }
        changed
    }
}
