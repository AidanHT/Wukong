//! `wukong_opt` — the optimizer: a pass manager plus MIR transforms.
//!
//! Passes run to a fixpoint at `-O1` and above, in this order. The pipeline first promotes stack
//! slots to SSA registers, which is what makes the value-based transforms bite:
//!  * **mem2reg** — promote scalar `alloca`/`load`/`store` to block-parameter SSA.
//!  * **simplify** — constant folding and algebraic identities (`x+0`, `x*1`, `x*0`, ...).
//!  * **simplify-cfg** — fold constant branches, merge straight-line blocks, prune dead blocks.
//!  * **simplify-phis** — drop dead/trivial block parameters mem2reg introduced.
//!  * **dce** — remove pure instructions whose results are never used, and unused allocas.
//!  * **cse** — dominator-tree value numbering with intra-block load forwarding (`-O2`).
//!  * **dse** — dead-store elimination (`-O2`).
//!  * **loop-canon** — one shape per loop: a preheader, a single latch, one exit-test polarity
//!    (`-O2`).
//!  * **licm** — hoist loop-invariant work into an existing preheader (`-O2`).
//!  * **vectorize** — widen a canonical loop body to 128-bit SIMD (if-converting a conditional
//!    body to a lane mask), with the original loop kept as its scalar epilogue (`-O2`).
//!
//! At `-O2` and above, whole-program inlining of small leaf functions ([`inline_program`]) runs once
//! before the per-function pipeline. `-O3` adds nothing to either — see [`PassManager::standard`].

mod cache;
mod cfg;
mod cse;
mod dce;
mod dom;
mod dse;
mod fxhash;
mod inline;
mod licm;
mod loop_canon;
pub mod loop_info;
mod mem2reg;
mod phi;
mod simplify;
mod simplify_cfg;
mod vectorize;

pub use cache::CfgAnalyses;
pub use cse::Cse;
pub use dce::Dce;
pub use dse::Dse;
pub use inline::inline_program;
pub use licm::Licm;
pub use loop_canon::LoopCanon;
pub use mem2reg::Mem2Reg;
pub use phi::SimplifyPhis;
pub use simplify::Simplify;
pub use simplify_cfg::SimplifyCfg;
pub use vectorize::Vectorize;

use std::time::{Duration, Instant};

use wukong_mir::{Op, Program, Terminator, ValueId};

/// A function-level transform. Returns whether it changed anything.
///
/// `cache` holds the CFG/dominator analyses for `f`, shared across every pass in the fixpoint. A
/// pass reads whatever it needs from it (computed lazily, once) and — if it changes the block set or
/// any terminator's successor edges — must call [`CfgAnalyses::invalidate`]. Passes that only touch
/// instructions, block parameters, or edge arguments leave the cache alone.
pub trait Pass {
    fn name(&self) -> &'static str;
    fn run_function(&self, f: &mut wukong_mir::Function, cache: &mut CfgAnalyses) -> bool;
}

/// In-process optimizer timing, gathered by [`optimize_timed`]. This is **measurement only**: the
/// timed path runs the identical pipeline with the identical sequence of mutations, so it produces
/// byte-identical MIR to [`optimize`] — the only difference is an `Instant` around each pass call.
/// The production [`optimize`] path pays zero timing overhead (the timing branch is never taken).
#[derive(Clone, Default)]
pub struct Timings {
    /// Per-pass accumulated wall time and invocation count, in pipeline order.
    pub per_pass: Vec<PassStat>,
    /// Whole-program inlining time (runs at `-O2`+).
    pub inline: Duration,
    /// Total time inside `optimize` (inlining + every pass + fixpoint bookkeeping).
    pub total: Duration,
    /// The largest per-function fixpoint iteration count observed across the program.
    pub max_iterations: u32,
}

/// One pass's accumulated cost over a whole `optimize_timed` run.
#[derive(Clone)]
pub struct PassStat {
    pub name: &'static str,
    pub time: Duration,
    /// How many times this pass's `run_function` was invoked (skipped clean passes don't count).
    pub calls: u64,
}

impl Timings {
    fn record(&mut self, idx: usize, name: &'static str, dur: Duration) {
        if self.per_pass.len() <= idx {
            self.per_pass.resize(
                idx + 1,
                PassStat {
                    name,
                    time: Duration::ZERO,
                    calls: 0,
                },
            );
        }
        let s = &mut self.per_pass[idx];
        s.name = name;
        s.time += dur;
        s.calls += 1;
    }
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

    /// The standard pipeline for an optimization level. `-O0` does nothing. The only level tests are
    /// `>= 1` and `>= 2`, so `-O3` builds the identical pipeline as `-O2` — a contract the
    /// differential gates rely on; do not add an `-O3`-only pass without revisiting it.
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
            // Put every loop into one shape before LICM runs: LICM hoists only into a preheader
            // that already exists, so a loop given one here becomes hoistable in the same sweep.
            pm.add(Box::new(LoopCanon));
            // LICM hoists invariant work out of loops; rerunning the pipeline then cleans up and
            // can expose further invariants (e.g. across nested loops).
            pm.add(Box::new(Licm));
            // Widening comes last, on the cleanest MIR the pipeline produces: canonical loops with
            // one latch and one exit-test polarity, invariants already hoisted, and dead code gone.
            // Everything it emits is fed back through the fixpoint, so the vector body gets the
            // same simplification, CSE and DCE the scalar body did.
            pm.add(Box::new(Vectorize::default()));
        }
        pm
    }

    pub fn run(&self, program: &mut Program) {
        for f in &mut program.funcs {
            self.run_function(f, None);
        }
    }

    /// The fixpoint driver. `timings` is `None` on the production path (zero overhead); the bench
    /// harness passes `Some(..)` to accumulate per-pass wall time. The `timings` branch does not
    /// change which passes run or in what order, so the resulting MIR is identical either way.
    fn run_function(&self, f: &mut wukong_mir::Function, mut timings: Option<&mut Timings>) {
        // One analysis cache per function, shared across every pass and every fixpoint iteration.
        // It is invalidated by the passes that restructure the CFG (see their `run_function`), so it
        // stays valid — and reused — across the long tail of the fixpoint where only instructions
        // move. The cache never changes what a pass computes, only how often it recomputes it.
        let mut cache = CfgAnalyses::default();
        // Fixpoint with per-pass clean-tracking. A pass is a *deterministic* function of the MIR, so a
        // pass that ran and reported "no change" cannot do anything until some OTHER pass mutates the
        // function — re-running it on identical MIR would again be a no-op. We therefore skip
        // known-clean passes, and a mutation re-dirties every pass (conservatively). This skips the
        // optimizer's final all-passes no-op *confirmation* sweep (and any pass already at fixpoint
        // mid-run) WITHOUT changing the sequence of mutations, so the resulting MIR is bit-identical
        // to the naive "run everything every sweep" fixpoint — the differential gate (-O0 ≡ -O{1,2,3},
        // interp ≡ native) and the debug verify below both still hold. ~20% off optimizer time.
        let mut clean = vec![false; self.passes.len()];
        let mut iterations = 0;
        loop {
            let mut changed = false;
            for (i, p) in self.passes.iter().enumerate() {
                if clean[i] {
                    continue; // at fixpoint: nothing has mutated the MIR since this pass last ran
                }
                let did = match timings.as_deref_mut() {
                    Some(t) => {
                        let t0 = Instant::now();
                        let did = p.run_function(f, &mut cache);
                        t.record(i, p.name(), t0.elapsed());
                        did
                    }
                    None => p.run_function(f, &mut cache),
                };
                if did {
                    changed = true;
                    // A mutation may have created work for every pass again (including earlier ones
                    // already run this sweep, and `p` itself). Re-dirty all; they re-run next sweep.
                    clean.iter_mut().for_each(|c| *c = false);
                } else {
                    clean[i] = true;
                }
                // verify-each: in debug builds (tests, CI) confirm every pass *that ran* leaves the
                // MIR well-formed, naming the culprit immediately. Compiled out of release builds.
                #[cfg(debug_assertions)]
                {
                    let errs = wukong_mir::verify::verify_function(f);
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
        if let Some(t) = timings.as_deref_mut() {
            t.max_iterations = t.max_iterations.max(iterations);
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

/// Like [`optimize`], but returns an in-process [`Timings`] breakdown (whole-optimizer total,
/// inlining, and per-pass accumulated wall time + call counts). Produces MIR **byte-identical** to
/// [`optimize`] — it runs the same pipeline in the same order, only wrapped in `Instant` timing.
/// For the bench harness; the production compile path uses [`optimize`] and pays no timing cost.
pub fn optimize_timed(program: &mut Program, opt_level: u8) -> Timings {
    let mut t = Timings::default();
    let start = Instant::now();
    if opt_level >= 2 {
        let i0 = Instant::now();
        inline_program(program);
        t.inline = i0.elapsed();
    }
    let pm = PassManager::standard(opt_level);
    for f in &mut program.funcs {
        pm.run_function(f, Some(&mut t));
    }
    t.total = start.elapsed();
    t
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
        | Op::Sqrt(a)
        | Op::Round(_, a) => {
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
        Op::VecKernelCall {
            ptrs, scalars, n, ..
        } => {
            *ptrs = f(*ptrs);
            *scalars = f(*scalars);
            *n = f(*n);
        }
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Alloca(..)
        | Op::FuncAddr(..)
        | Op::GlobalAddr(..) => {}
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
        | Op::Sqrt(a)
        | Op::Round(_, a) => f(*a),
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
        Op::VecKernelCall {
            ptrs, scalars, n, ..
        } => {
            f(*ptrs);
            f(*scalars);
            f(*n);
        }
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Alloca(..)
        | Op::FuncAddr(..)
        | Op::GlobalAddr(..) => {}
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
    matches!(
        op,
        Op::Store { .. } | Op::Call { .. } | Op::VecKernelCall { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn run_main_opt(src: &str, opt: u8) -> i64 {
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        optimize(&mut program, opt);
        // verify still well-formed
        for f in &program.funcs {
            assert!(
                wukong_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        wukong_interp::run(&program, main, &interner).unwrap()
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
    fn vector_lane_self_ops_are_not_folded_to_scalar_consts() {
        // The same `x - x` / `x <cmp> x` identities as above, but written so the front-end
        // vectorizes the loop body: the Bin/Cmp results are then `<N x iW>`, which cannot hold the
        // scalar `ConstInt` the fold would materialize. Declining the fold is the only legal answer.
        let cases = [
            (
                "fn main() -> i32 { let mut n: [i32; 8] = [8; 8]; \
                 for i in 0..8 { n[i] = n[i] - n[i]; } return n[0] + n[7] + 7; }",
                7,
            ),
            (
                "fn main() -> i32 { let mut n: [i64; 16] = [8; 16]; \
                 for i in 0..16 { n[i] = n[i] - n[i]; } return (n[0] + n[15] + 7) as i32; }",
                7,
            ),
            (
                "fn main() -> i32 { let mut n: [i32; 16] = [8; 16]; let mut o: [i32; 16] = [0; 16]; \
                 for i in 0..16 { o[i] = if n[i] < n[i] { 1 } else { 2 }; } \
                 return o[0] + o[15]; }",
                4,
            ),
        ];
        for (src, expect) in cases {
            for lvl in [0, 1, 2, 3] {
                assert_eq!(run_main_opt(src, lvl), expect as i64, "O{lvl}: {src}");
            }
        }
    }

    #[test]
    fn narrow_float_constant_compares_are_not_folded() {
        // Two distinct f32 literals that share one bf16 (resp. f16) grid point compare EQUAL at
        // runtime, because both backends round the value to the narrow grid. The optimizer only
        // rounds folded constants to f32, so folding these comparisons flips the branch at -O1+.
        let cases = [
            (
                "fn main() -> i32 { let a: bf16 = 0.1; let b: bf16 = 0.1002; \
                 if a == b { return 1; } return 0; }",
                1,
            ),
            (
                "fn main() -> i32 { let a: f16 = 0.1; let b: f16 = 0.100005; \
                 if a == b { return 1; } return 0; }",
                1,
            ),
        ];
        for (src, expect) in cases {
            for lvl in [0, 1, 2, 3] {
                assert_eq!(run_main_opt(src, lvl), expect as i64, "O{lvl}: {src}");
            }
        }
    }

    #[test]
    fn folds_and_dces_constants() {
        // main computes (2*3 + 4) entirely from constants; after -O2 the body should be tiny.
        let src = "fn main() -> i32 { let x: i32 = 2 * 3 + 4; return x; }";
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = wukong_sema::check(&module, &interner);
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let before: usize = program.funcs.iter().map(count_insts).sum();
        optimize(&mut program, 2);
        let after: usize = program.funcs.iter().map(count_insts).sum();
        assert!(
            after < before,
            "expected fewer insts after opt ({before} -> {after})"
        );
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&program, main, &interner).unwrap(), 10);
    }

    fn count_insts(f: &wukong_mir::Function) -> usize {
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
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);

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
                wukong_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&program, main, &interner).unwrap(), 98);
    }

    fn find_fn<'a>(
        p: &'a wukong_mir::Program,
        interner: &Interner,
        name: &str,
    ) -> &'a wukong_mir::Function {
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
            "fn main() -> i32 { let mut x: i32 = 0; if 1 < 2 { x = 10; } else { x = 20; } return x; }";
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
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
                wukong_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&program, main, &interner).unwrap(), 10);
    }

    #[test]
    fn dse_removes_overwritten_store() {
        // `x` is written, then immediately overwritten with no read in between: the first store is
        // dead. The result must be unchanged.
        let src = "fn main() -> i32 { let mut x: i32 = 1; x = 2; x = 3; return x; }";
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let stores_before = count_stores(find_fn(&program, &interner, "main"));
        optimize(&mut program, 2);
        let stores_after = count_stores(find_fn(&program, &interner, "main"));
        assert!(
            stores_after < stores_before,
            "DSE should drop a dead store ({stores_before} -> {stores_after})"
        );
        for f in &program.funcs {
            assert!(
                wukong_mir::verify::verify_function(f).is_empty(),
                "verify after opt"
            );
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&program, main, &interner).unwrap(), 3);
    }

    fn count_stores(f: &wukong_mir::Function) -> usize {
        use wukong_mir::Op;
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, Op::Store { .. }))
            .count()
    }

    fn count_muls(f: &wukong_mir::Function) -> usize {
        f.blocks.iter().map(muls_in_block).sum()
    }

    fn muls_in_block(b: &wukong_mir::BasicBlock) -> usize {
        use wukong_mir::{BinOp, Op};
        b.insts
            .iter()
            .filter(|i| matches!(i.op, Op::Bin(BinOp::Mul, _, _)))
            .count()
    }

    fn count_allocas(f: &wukong_mir::Function) -> usize {
        use wukong_mir::Op;
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, Op::Alloca(_)))
            .count()
    }

    fn lower(src: &str) -> (wukong_mir::Program, Interner) {
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
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
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 45);
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
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 4);
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
            .any(|i| matches!(i.op, wukong_mir::Op::Call { .. }));
        assert!(!has_call, "sq should be inlined into main");
    }

    #[test]
    fn inlining_rebases_a_callee_that_owns_vec_kernels() {
        // `Op::VecKernelCall.kernel` indexes the *owning* function's `vec_kernels` table by
        // position, so splicing a vectorized leaf into a caller must carry its recipes across and
        // rebase the index. Case (a): the caller already owns a kernel, so a stale index silently
        // resolves to the WRONG recipe (`|x|` instead of `-x`). Case (b): the caller owns none, so
        // it dangles — a hard error in the interpreter and an out-of-bounds panic in codegen.
        let with_caller_kernel = "fn negv(mut d: [f32; 96]) { for i in 0..96 { d[i] = -d[i]; } } \
             fn main() -> i32 { let mut d: [f32; 96] = [0.0; 96]; \
               let mut e: [f32; 96] = [0.0; 96]; \
               for i in 0..96 { d[i] = (i as f32) - 50.0; } \
               for i in 0..96 { e[i] = (i as f32) - 50.0; } \
               for i in 0..96 { e[i] = sqrt(e[i] * e[i]); } \
               negv(d); \
               return (d[60] as i32) * 100 + (e[47] as i32); }";
        let no_caller_kernel = "fn negv(mut d: [f32; 96]) { for i in 0..96 { d[i] = -d[i]; } } \
             fn main() -> i32 { let mut d: [f32; 96] = [0.0; 96]; \
               for i in 0..96 { d[i] = (i as f32) - 50.0; } \
               negv(d); return d[47] as i32; }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(with_caller_kernel, lvl), -997, "level {lvl}");
            assert_eq!(run_main_opt(no_caller_kernel, lvl), 3, "level {lvl}");
        }
    }

    #[test]
    fn inlining_rebases_repeated_and_multiple_kernel_owning_callees() {
        // The kernel-table rebase has to accumulate: `negv` is spliced twice and `scalev` once, on
        // top of a caller that already owns a recipe, so the three splices must land at successive
        // offsets. Pinning it separately from the single-splice case because getting the base right
        // once (and then re-using it, or resetting it) would still pass that test.
        let src = "fn negv(mut d: [f32; 96]) { for i in 0..96 { d[i] = -d[i]; } } \
             fn scalev(mut d: [f32; 96]) { for i in 0..96 { d[i] = d[i] * 3.0; } } \
             fn main() -> i32 { let mut a: [f32; 96] = [0.0; 96]; \
               let mut b: [f32; 96] = [0.0; 96]; let mut c: [f32; 96] = [0.0; 96]; \
               for i in 0..96 { a[i] = (i as f32) - 50.0; } \
               for i in 0..96 { b[i] = (i as f32) - 50.0; } \
               for i in 0..96 { c[i] = (i as f32) - 50.0; } \
               for i in 0..96 { c[i] = sqrt(c[i] * c[i]); } \
               negv(a); negv(b); scalev(b); \
               return (a[60] as i32) * 10000 + (b[60] as i32) * 100 + (c[47] as i32); }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), -102997, "level {lvl}");
        }
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
            .filter(|i| matches!(i.op, wukong_mir::Op::Call { .. }))
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
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 60);
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
