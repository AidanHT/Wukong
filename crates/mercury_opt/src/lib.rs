//! `mercury_opt` — the optimizer: a pass manager plus MIR transforms.
//!
//! Passes run to a fixpoint at `-O1` and above. The transforms here are correct on the
//! alloca-based MIR the front-end emits (alloca promotion / mem2reg is left to LLVM's pipeline):
//!  * **simplify** — constant folding and algebraic identities (`x+0`, `x*1`, `x*0`, ...).
//!  * **dce** — remove pure instructions whose results are never used, and unused allocas.

mod dce;
mod simplify;

pub use dce::Dce;
pub use simplify::Simplify;

use mercury_mir::{Op, Program, Terminator, ValueId};

/// A function-level transform. Returns whether it changed anything.
pub trait Pass {
    fn name(&self) -> &'static str;
    fn run_function(&self, f: &mut mercury_mir::Function) -> bool;
}

/// An ordered list of passes, run to a per-function fixpoint.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> PassManager {
        PassManager { passes: Vec::new() }
    }

    pub fn add(&mut self, p: Box<dyn Pass>) {
        self.passes.push(p);
    }

    /// The standard pipeline for an optimization level. `-O0` does nothing.
    pub fn standard(opt_level: u8) -> PassManager {
        let mut pm = PassManager::new();
        if opt_level >= 1 {
            pm.add(Box::new(Simplify));
            pm.add(Box::new(Dce));
        }
        pm
    }

    pub fn run(&self, program: &mut Program) {
        for f in &mut program.funcs {
            self.run_function(f);
        }
    }

    fn run_function(&self, f: &mut mercury_mir::Function) {
        let mut iterations = 0;
        loop {
            let mut changed = false;
            for p in &self.passes {
                changed |= p.run_function(f);
            }
            iterations += 1;
            if !changed || iterations > 100 {
                break;
            }
        }
    }
}

impl Default for PassManager {
    fn default() -> PassManager {
        PassManager::new()
    }
}

/// Optimize a whole program at the given level.
pub fn optimize(program: &mut Program, opt_level: u8) {
    PassManager::standard(opt_level).run(program);
}

// ---- shared use-visiting helpers ----

/// Apply `f` to every value *used* (read) by an operation, in place.
pub(crate) fn map_op_uses(op: &mut Op, mut f: impl FnMut(ValueId) -> ValueId) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) => {
            *a = f(*a);
            *b = f(*b);
        }
        Op::Neg(a) | Op::Not(a) | Op::Cast(_, a, _) | Op::Load(a, _) => {
            *a = f(*a);
        }
        Op::Select(c, a, b) => {
            *c = f(*c);
            *a = f(*a);
            *b = f(*b);
        }
        Op::Store { ptr, value } => {
            *ptr = f(*ptr);
            *value = f(*value);
        }
        Op::Gep { ptr, index, .. } => {
            *ptr = f(*ptr);
            *index = f(*index);
        }
        Op::Call { args, .. } => {
            for a in args {
                *a = f(*a);
            }
        }
        Op::ConstInt(..) | Op::ConstFloat(..) | Op::Alloca(..) => {}
    }
}

/// Apply `f` to every value used by a terminator, in place.
pub(crate) fn map_term_uses(t: &mut Terminator, mut f: impl FnMut(ValueId) -> ValueId) {
    match t {
        Terminator::Ret(Some(v)) => *v = f(*v),
        Terminator::Br { args, .. } => {
            for a in args {
                *a = f(*a);
            }
        }
        Terminator::CondBr { cond, then_args, else_args, .. } => {
            *cond = f(*cond);
            for a in then_args {
                *a = f(*a);
            }
            for a in else_args {
                *a = f(*a);
            }
        }
        Terminator::Ret(None) | Terminator::Unreachable => {}
    }
}

/// Visit (read-only) every value used by an operation.
pub(crate) fn each_op_use(op: &Op, f: &mut impl FnMut(ValueId)) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) => {
            f(*a);
            f(*b);
        }
        Op::Neg(a) | Op::Not(a) | Op::Cast(_, a, _) | Op::Load(a, _) => f(*a),
        Op::Select(c, a, b) => {
            f(*c);
            f(*a);
            f(*b);
        }
        Op::Store { ptr, value } => {
            f(*ptr);
            f(*value);
        }
        Op::Gep { ptr, index, .. } => {
            f(*ptr);
            f(*index);
        }
        Op::Call { args, .. } => {
            for a in args {
                f(*a);
            }
        }
        Op::ConstInt(..) | Op::ConstFloat(..) | Op::Alloca(..) => {}
    }
}

/// Visit (read-only) every value used by a terminator.
pub(crate) fn each_term_use(t: &Terminator, f: &mut impl FnMut(ValueId)) {
    match t {
        Terminator::Ret(Some(v)) => f(*v),
        Terminator::Br { args, .. } => args.iter().for_each(|a| f(*a)),
        Terminator::CondBr { cond, then_args, else_args, .. } => {
            f(*cond);
            then_args.iter().for_each(|a| f(*a));
            else_args.iter().for_each(|a| f(*a));
        }
        Terminator::Ret(None) | Terminator::Unreachable => {}
    }
}

/// Does this op have a side effect that prevents removing it even if its result is unused?
pub(crate) fn has_side_effects(op: &Op) -> bool {
    matches!(op, Op::Store { .. } | Op::Call { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_span::{Interner, SourceId};

    fn run_main_opt(src: &str, opt: u8) -> i64 {
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &interner);
        optimize(&mut program, opt);
        // verify still well-formed
        for f in &program.funcs {
            assert!(mercury_mir::verify::verify_function(f).is_empty(), "verify after opt");
        }
        let main = interner.intern("main");
        mercury_interp::run(&program, main, &interner).unwrap()
    }

    #[test]
    fn optimization_preserves_results() {
        let fib = "fn fib(n: i32) -> i32 { if n < 2 { return n; } \
                   return fib(n-1) + fib(n-2); } fn main() -> i32 { return fib(12); }";
        assert_eq!(run_main_opt(fib, 0), run_main_opt(fib, 2));
        assert_eq!(run_main_opt(fib, 2), 144);

        let loops = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                     while i < 100 { s += i * 2 + 0; i += 1; } return s; }";
        assert_eq!(run_main_opt(loops, 0), run_main_opt(loops, 2));
    }

    #[test]
    fn folds_and_dces_constants() {
        // main computes (2*3 + 4) entirely from constants; after -O2 the body should be tiny.
        let src = "fn main() -> i32 { let x: i32 = 2 * 3 + 4; return x; }";
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = mercury_sema::check(&module, &interner);
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &interner);
        let before: usize = program.funcs.iter().map(|f| count_insts(f)).sum();
        optimize(&mut program, 2);
        let after: usize = program.funcs.iter().map(|f| count_insts(f)).sum();
        assert!(after < before, "expected fewer insts after opt ({before} -> {after})");
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&program, main, &interner).unwrap(), 10);
    }

    fn count_insts(f: &mercury_mir::Function) -> usize {
        f.blocks.iter().map(|b| b.insts.len()).sum()
    }
}
