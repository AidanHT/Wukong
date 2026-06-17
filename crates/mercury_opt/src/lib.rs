//! `mercury_opt` — the optimizer: a pass manager plus MIR transforms.
//!
//! Passes run to a fixpoint at `-O1` and above. The pipeline first promotes stack slots to SSA
//! registers, which is what makes the value-based transforms bite:
//!  * **mem2reg** — promote scalar `alloca`/`load`/`store` to block-parameter SSA.
//!  * **simplify** — constant folding and algebraic identities (`x+0`, `x*1`, `x*0`, ...).
//!  * **simplify-cfg** — fold constant branches, merge straight-line blocks, prune dead blocks.
//!  * **simplify-phis** — drop dead/trivial block parameters mem2reg introduced.
//!  * **cse** — local value numbering with load forwarding (`-O2`).
//!  * **dse** — dead-store elimination (`-O2`).
//!  * **dce** — remove pure instructions whose results are never used, and unused allocas.

mod cfg;
mod cse;
mod dce;
mod dom;
mod dse;
mod inline;
mod licm;
mod mem2reg;
mod phi;
mod simplify;
mod simplify_cfg;

pub use cse::Cse;
pub use dce::Dce;
pub use dse::Dse;
pub use inline::inline_program;
pub use licm::Licm;
pub use mem2reg::Mem2Reg;
pub use phi::SimplifyPhis;
pub use simplify::Simplify;
pub use simplify_cfg::SimplifyCfg;

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
            // Promotion comes first: every later pass is far more effective on SSA values than on
            // memory traffic. The whole list then runs to a fixpoint.
            pm.add(Box::new(Mem2Reg));
            pm.add(Box::new(Simplify));
            pm.add(Box::new(SimplifyCfg));
            pm.add(Box::new(SimplifyPhis));
            pm.add(Box::new(Dce));
        }
        if opt_level >= 2 {
            // CSE feeds Simplify/DCE more constants and dead values; the fixpoint loop reruns all.
            pm.add(Box::new(Cse));
            pm.add(Box::new(Dse));
            // LICM hoists invariant work out of loops; rerunning the pipeline then cleans up and
            // can expose further invariants (e.g. across nested loops).
            pm.add(Box::new(Licm));
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
                // verify-each: in debug builds (tests, CI) confirm every pass leaves the MIR
                // well-formed, naming the culprit immediately. Compiled out of release builds.
                #[cfg(debug_assertions)]
                {
                    let errs = mercury_mir::verify::verify_function(f);
                    assert!(
                        errs.is_empty(),
                        "pass `{}` produced invalid MIR: {errs:?}",
                        p.name()
                    );
                }
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
    // Inlining is a whole-program transform (it needs callee bodies), so it runs before the
    // function-level pipeline — which then optimizes the spliced-in code in context.
    if opt_level >= 2 {
        inline_program(program);
    }
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
        Op::Neg(a)
        | Op::Not(a)
        | Op::Cast(_, a, _)
        | Op::Load(a, _)
        | Op::Splat(a)
        | Op::Sqrt(a) => {
            *a = f(*a);
        }
        Op::Select(c, a, b) | Op::Fma(c, a, b) => {
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
        Op::ConstInt(..) | Op::ConstFloat(..) | Op::Alloca(..) | Op::FuncAddr(..) => {}
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
        Terminator::CondBr {
            cond,
            then_args,
            else_args,
            ..
        } => {
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
        Op::Neg(a)
        | Op::Not(a)
        | Op::Cast(_, a, _)
        | Op::Load(a, _)
        | Op::Splat(a)
        | Op::Sqrt(a) => f(*a),
        Op::Select(c, a, b) | Op::Fma(c, a, b) => {
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
        Op::ConstInt(..) | Op::ConstFloat(..) | Op::Alloca(..) | Op::FuncAddr(..) => {}
    }
}

/// Visit (read-only) every value used by a terminator.
pub(crate) fn each_term_use(t: &Terminator, f: &mut impl FnMut(ValueId)) {
    match t {
        Terminator::Ret(Some(v)) => f(*v),
        Terminator::Br { args, .. } => args.iter().for_each(|a| f(*a)),
        Terminator::CondBr {
            cond,
            then_args,
            else_args,
            ..
        } => {
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
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        optimize(&mut program, opt);
        // verify still well-formed
        for f in &program.funcs {
            assert!(
                mercury_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
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
    fn algebraic_identities_preserve_results() {
        // x^x == 0, x|x == x, x&x == x, x%1 == 0, and a self-comparison.
        let cases = [
            ("fn main() -> i32 { let x: i32 = 9; return x ^ x; }", 0),
            ("fn main() -> i32 { let x: i32 = 9; return x | x; }", 9),
            ("fn main() -> i32 { let x: i32 = 9; return x & x; }", 9),
            ("fn main() -> i32 { let x: i32 = 9; return x % 1; }", 0),
            (
                "fn main() -> i32 { let x: i32 = 9; if x <= x { return 1; } return 0; }",
                1,
            ),
            (
                "fn main() -> i32 { let x: i32 = 9; if x < x { return 1; } return 0; }",
                0,
            ),
        ];
        for (src, expect) in cases {
            assert_eq!(run_main_opt(src, 0), expect as i64, "O0: {src}");
            assert_eq!(run_main_opt(src, 2), expect as i64, "O2: {src}");
        }
    }

    #[test]
    fn folds_and_dces_constants() {
        // main computes (2*3 + 4) entirely from constants; after -O2 the body should be tiny.
        let src = "fn main() -> i32 { let x: i32 = 2 * 3 + 4; return x; }";
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = mercury_sema::check(&module, &interner);
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        let before: usize = program.funcs.iter().map(count_insts).sum();
        optimize(&mut program, 2);
        let after: usize = program.funcs.iter().map(count_insts).sum();
        assert!(
            after < before,
            "expected fewer insts after opt ({before} -> {after})"
        );
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&program, main, &interner).unwrap(), 10);
    }

    fn count_insts(f: &mercury_mir::Function) -> usize {
        f.blocks.iter().map(|b| b.insts.len()).sum()
    }

    #[test]
    fn cse_eliminates_redundant_expression() {
        // `a*a` is computed twice in the same block; CSE should collapse it without changing the
        // result. Use distinct params so the multiply is not itself const-folded away.
        let src =
            "fn sq2(x: i32) -> i32 { let a: i32 = x * x; let b: i32 = x * x; return a + b; } \
                   fn main() -> i32 { return sq2(7); }";
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);

        // Count `x*x` multiplies in `sq2` before and after CSE+DCE. Run the pass pipeline directly
        // (not the `optimize` wrapper) so whole-program inlining does not fold `sq2` into main.
        let muls_before = count_muls(find_fn(&program, &interner, "sq2"));
        PassManager::standard(2).run(&mut program);
        let muls_after = count_muls(find_fn(&program, &interner, "sq2"));
        assert!(
            muls_before >= 2,
            "expected two multiplies before opt, saw {muls_before}"
        );
        assert!(
            muls_after < muls_before,
            "CSE should remove a redundant multiply"
        );

        for f in &program.funcs {
            assert!(
                mercury_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&program, main, &interner).unwrap(), 98);
    }

    fn find_fn<'a>(
        p: &'a mercury_mir::Program,
        interner: &Interner,
        name: &str,
    ) -> &'a mercury_mir::Function {
        p.funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .expect("function present")
    }

    #[test]
    fn simplify_cfg_folds_constant_branch_and_prunes() {
        // The `if` condition is a compile-time constant, so one arm is dead. After -O1 the dead
        // block should be pruned and the result must be unchanged.
        let src =
            "fn main() -> i32 { let x: i32 = 0; if 1 < 2 { x = 10; } else { x = 20; } return x; }";
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        let blocks_before = find_fn(&program, &interner, "main").blocks.len();
        optimize(&mut program, 1);
        let main_fn = find_fn(&program, &interner, "main");
        assert!(
            main_fn.blocks.len() < blocks_before,
            "expected fewer blocks after CFG cleanup ({blocks_before} -> {})",
            main_fn.blocks.len()
        );
        // No CondBr should remain referencing the constant condition.
        for f in &program.funcs {
            assert!(
                mercury_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&program, main, &interner).unwrap(), 10);
    }

    #[test]
    fn dse_removes_overwritten_store() {
        // `x` is written, then immediately overwritten with no read in between: the first store is
        // dead. The result must be unchanged.
        let src = "fn main() -> i32 { let mut x: i32 = 1; x = 2; x = 3; return x; }";
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        let stores_before = count_stores(find_fn(&program, &interner, "main"));
        optimize(&mut program, 2);
        let stores_after = count_stores(find_fn(&program, &interner, "main"));
        assert!(
            stores_after < stores_before,
            "DSE should drop a dead store ({stores_before} -> {stores_after})"
        );
        for f in &program.funcs {
            assert!(
                mercury_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&program, main, &interner).unwrap(), 3);
    }

    fn count_stores(f: &mercury_mir::Function) -> usize {
        use mercury_mir::Op;
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, Op::Store { .. }))
            .count()
    }

    fn count_muls(f: &mercury_mir::Function) -> usize {
        f.blocks.iter().map(muls_in_block).sum()
    }

    fn muls_in_block(b: &mercury_mir::BasicBlock) -> usize {
        use mercury_mir::{BinOp, Op};
        b.insts
            .iter()
            .filter(|i| matches!(i.op, Op::Bin(BinOp::Mul, _, _)))
            .count()
    }

    fn count_allocas(f: &mercury_mir::Function) -> usize {
        use mercury_mir::Op;
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, Op::Alloca(_)))
            .count()
    }

    fn lower(src: &str) -> (mercury_mir::Program, Interner) {
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    #[test]
    fn mem2reg_promotes_all_scalar_slots() {
        // A loop with scalar locals only: after -O1, every alloca should be gone and the function
        // must still verify and compute the same answer.
        let src = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < 10 { s = s + i; i = i + 1; } return s; }";
        let (mut prog, mut interner) = lower(src);
        assert!(
            count_allocas(&prog.funcs[0]) > 0,
            "front-end should alloca locals"
        );
        optimize(&mut prog, 1);
        assert_eq!(
            count_allocas(&prog.funcs[0]),
            0,
            "mem2reg should promote all scalar slots"
        );
        for f in &prog.funcs {
            assert!(mercury_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&prog, main, &interner).unwrap(), 45);
    }

    #[test]
    fn mem2reg_introduces_a_loop_phi() {
        // The induction variable becomes a block parameter (phi) on the loop header.
        let src = "fn main() -> i32 { let mut i: i32 = 0; while i < 5 { i = i + 1; } return i; }";
        let (mut prog, _) = lower(src);
        optimize(&mut prog, 1);
        let has_block_param = prog.funcs[0]
            .blocks
            .iter()
            .any(|b| b.id != prog.funcs[0].entry && !b.params.is_empty());
        assert!(has_block_param, "loop should have a phi block parameter");
    }

    #[test]
    fn mem2reg_leaves_arrays_and_address_taken_in_memory() {
        // Array locals are addressed by gep, so they must NOT be promoted; the program is still
        // correct and the array alloca remains.
        let src = "fn main() -> i32 { let mut xs: [i32; 3] = [1, 2, 3]; \
                   xs[1] = xs[0] + xs[2]; return xs[1]; }";
        let (mut prog, mut interner) = lower(src);
        optimize(&mut prog, 2);
        assert!(
            count_allocas(&prog.funcs[0]) >= 1,
            "array slot must stay in memory"
        );
        for f in &prog.funcs {
            assert!(mercury_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&prog, main, &interner).unwrap(), 4);
    }

    #[test]
    fn inlining_removes_calls_and_preserves_results() {
        // `sq` is a small leaf function. At -O2 it is inlined into main and then constant-folded
        // away entirely, leaving no call. The result must be unchanged across all levels.
        let src = "fn sq(x: i32) -> i32 { return x * x; } \
                   fn main() -> i32 { return sq(6) + sq(7); }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 85, "level {lvl}");
        }
        let (mut prog, interner) = lower(src);
        optimize(&mut prog, 2);
        let main = find_fn(&prog, &interner, "main");
        let has_call = main
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i.op, mercury_mir::Op::Call { .. }));
        assert!(!has_call, "sq should be inlined into main");
    }

    #[test]
    fn inlining_handles_callee_control_flow() {
        // A leaf with branches (max) inlines correctly; results are preserved at every level.
        let src = "fn max(a: i32, b: i32) -> i32 { if a > b { return a; } return b; } \
                   fn main() -> i32 { return max(3, 9) + max(20, 5); } ";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 29, "level {lvl}");
        }
    }

    #[test]
    fn recursion_is_not_inlined() {
        // A recursive function is not a leaf, so it must survive -O2 with its self-call intact and
        // still compute correctly.
        let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } return fib(n-1) + fib(n-2); } \
                   fn main() -> i32 { return fib(12); }";
        assert_eq!(run_main_opt(src, 2), 144);
        let (mut prog, interner) = lower(src);
        optimize(&mut prog, 2);
        let fib = find_fn(&prog, &interner, "fib");
        let calls = fib
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, mercury_mir::Op::Call { .. }))
            .count();
        assert!(calls >= 2, "recursive fib should keep its self-calls");
    }

    #[test]
    fn licm_hoists_invariant_out_of_loop() {
        // `x * y` does not change across the loop, so LICM should compute it once in the preheader
        // (the entry block here) rather than every iteration.
        let src = "fn f(x: i32, y: i32, n: i32) -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < n { s = s + x * y; i = i + 1; } return s; } \
                   fn main() -> i32 { return f(3, 4, 5); }";
        let (mut prog, mut interner) = lower(src);
        // Run the pipeline directly so inlining does not fold `f` into main before LICM is observed.
        PassManager::standard(2).run(&mut prog);
        let f = find_fn(&prog, &interner, "f");
        assert_eq!(
            count_muls(f),
            1,
            "x*y must be computed once, not duplicated"
        );
        let entry_muls = muls_in_block(&f.blocks[f.entry.0 as usize]);
        assert_eq!(
            entry_muls, 1,
            "the invariant multiply should be hoisted into the preheader"
        );
        for f in &prog.funcs {
            assert!(mercury_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(mercury_interp::run(&prog, main, &interner).unwrap(), 60);
    }

    #[test]
    fn mem2reg_preserves_branchy_dataflow() {
        // A value defined on one path and merged: exercises dominance-frontier phi placement.
        let cases = [
            ("fn main() -> i32 { let mut x: i32 = 0; if 3 > 1 { x = 7; } else { x = 9; } return x; }", 7),
            ("fn f(n: i32) -> i32 { let mut r: i32 = 1; let mut k: i32 = n; \
              while k > 0 { r = r * k; k = k - 1; } return r; } \
              fn main() -> i32 { return f(5); }", 120),
            ("fn main() -> i32 { let mut a: i32 = 1; let mut b: i32 = 1; let mut n: i32 = 10; \
              while n > 0 { let t: i32 = a + b; a = b; b = t; n = n - 1; } return a; }", 89),
        ];
        for (src, expect) in cases {
            assert_eq!(run_main_opt(src, 0), expect, "O0: {src}");
            assert_eq!(run_main_opt(src, 1), expect, "O1: {src}");
            assert_eq!(run_main_opt(src, 2), expect, "O2: {src}");
            assert_eq!(run_main_opt(src, 3), expect, "O3: {src}");
        }
    }
}
