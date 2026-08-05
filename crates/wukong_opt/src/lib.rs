//! `wukong_opt` — the optimizer: a pass manager plus MIR transforms.
//!
//! Passes run to a fixpoint at `-O1` and above, in this order. The pipeline first promotes stack
//! slots to SSA registers, which is what makes the value-based transforms bite:
//!  * **mem2reg** — promote scalar and pointer `alloca`/`load`/`store` to block-parameter SSA.
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
//! `cse`, `dse` and `licm` all consult [`alias`], the provenance analysis that answers *can a store
//! through `q` be seen by a load through `p`?* — the question `mir_build`'s type erasure
//! (`Ty::Ptr`/`Ref`/`Tensor`/`Slice` all become a bare `MirType::Ptr`) otherwise makes unanswerable.
//! It is a pure analysis, safe to build from anywhere, and the Cranelift backend uses it too.
//!
//! At `-O2` and above, whole-program inlining ([`inline_program`]) runs once *before* the
//! per-function pipeline, and partial loop unrolling ([`unroll_program`]) runs once *after* it — the
//! pipeline is then re-run over the functions unrolling changed, so the duplicated bodies get the
//! same cse/simplify/dce treatment as everything else. `-O3` adds nothing to any of it — see
//! [`PassManager::standard`].

pub mod alias;
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
mod unroll;
mod vectorize;

pub use alias::{type_bytes, AliasInfo, Prov};
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
pub use unroll::unroll_program;
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
    /// Loop-unrolling time (runs at `-O2`+, after the pipeline): the pass itself **plus** the
    /// reduced cleanup sweep over the functions it changed, which together are the whole cost
    /// unrolling adds. Deliberately not split across the per-pass buckets — those are keyed by pass
    /// index within [`PassManager::standard`], and the cleanup pipeline numbers its passes
    /// differently.
    pub unroll: Duration,
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

    /// The reduced pipeline re-run over the functions [`unroll_program`] changed.
    ///
    /// Unrolling duplicates a loop body, so what the result needs is value numbering over the now
    /// four-fold address arithmetic and the dead-code removal that follows it. It does not need
    /// `Mem2Reg` (the pass refuses to duplicate a header or body containing an `Alloca`, so there is
    /// no new stack slot to promote), `Dse`, or `Licm` (LICM already ran over this same body, and
    /// every instruction the copies add depends on the induction variable); `Simplify`,
    /// `SimplifyCfg` and `SimplifyPhis` have almost nothing to do to a block that was merely
    /// quadrupled. Measured in-process over the 331-program corpus, the whole unrolling stage is
    /// 23% of optimizer time with these two passes and 33-39% with all five, against 11% with none,
    /// so this is where the curve bends. Dropping passes can only leave code less optimized, never
    /// wrong, so no gate depends on this list.
    fn unroll_cleanup() -> PassManager {
        let mut pm = PassManager::new();
        pm.add(Box::new(Cse));
        pm.add(Box::new(Dce));
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
    let pm = PassManager::standard(opt_level);
    pm.run(program);
    // Unrolling runs after the pipeline, not inside its fixpoint: it needs the canonical two-block
    // counted loop that mem2reg/simplify-cfg produce, and re-running it on its own output would
    // unroll the same loop again every sweep. Only the functions it touched are re-optimized, so a
    // program with no unrollable loop pays one scan and nothing more.
    if opt_level >= 2 {
        let changed = unroll_program(program);
        if !changed.is_empty() {
            let cleanup = PassManager::unroll_cleanup();
            for i in changed {
                cleanup.run_function(&mut program.funcs[i], None);
            }
        }
    }
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
    if opt_level >= 2 {
        // The unroll bucket is the pass *and* its cleanup sweep: that sum is the cost unrolling
        // actually adds, which is the number worth reporting. Keeping the cleanup out of the
        // per-pass table also avoids charging its `Simplify` to the standard pipeline's slot 0
        // (`Timings::record` keys by pass index, and the two pipelines number their passes
        // differently).
        let u0 = Instant::now();
        let changed = unroll_program(program);
        if !changed.is_empty() {
            let cleanup = PassManager::unroll_cleanup();
            for i in changed {
                cleanup.run_function(&mut program.funcs[i], None);
            }
        }
        t.unroll = u0.elapsed();
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
        | Op::ExtractLane(a, _)
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
        | Op::ExtractLane(a, _)
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
    fn a_dead_block_parameter_cycle_is_removed() {
        // `t` is declared *inside* the inner loop and never read after it. The front end still
        // gives it one alloca in the entry block, so mem2reg places a block parameter for it at the
        // inner header *and* at the outer one — the outer header is in the inner header's iterated
        // dominance frontier. Those two feed each other and nothing else: the outer latch passes
        // the inner parameter out, the inner preheader passes the outer parameter back in. Both
        // therefore have a non-zero use count, and the old "is this parameter used?" scan removed
        // neither.
        //
        // Only `s` and the counter have to cross either back edge, so no block may carry more than
        // two parameters. Before the liveness fixpoint, both headers carried three.
        let src = "fn main() -> i32 { let mut s: i32 = 0; let mut r: i32 = 0; \
                   while r < 4 { \
                     let mut i: i32 = 0; \
                     while i < 10 { let t: i32 = i * 3 + 1; s = s + t; i = i + 1; } \
                     r = r + 1; } \
                   return s; }";
        let (mut prog, mut interner) = lower(src);
        optimize(&mut prog, 1);
        let main = find_fn(&prog, &interner, "main");
        for b in &main.blocks {
            if b.id == main.entry {
                continue;
            }
            assert!(
                b.params.len() <= 2,
                "bb{} carries {} parameters; only `s` and a counter cross a back edge",
                b.id.0,
                b.params.len()
            );
        }
        let m = interner.intern("main");
        // sum of 3i + 1 for i in 0..10 is 3*45 + 10 = 145, accumulated over 4 outer passes.
        assert_eq!(wukong_interp::run(&prog, m, &interner).unwrap(), 580);
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
    fn a_helper_chain_collapses_in_one_sweep() {
        // `main -> outer -> mid -> leaf`. Inlining bottom-up means `mid` has already absorbed
        // `leaf` by the time `outer` is considered, so the whole chain folds to a constant.
        // Leaf-only inlining spliced `leaf` into `mid` and stopped: `outer` and `main` kept their
        // calls, and the abstraction blocked every downstream transform.
        let src = "fn leaf(x: i32) -> i32 { return x * x + 1; } \
                   fn mid(x: i32) -> i32 { return leaf(x) + leaf(x + 1); } \
                   fn outer(x: i32) -> i32 { return mid(x) * 2; } \
                   fn main() -> i32 { return outer(3); }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 54, "level {lvl}");
        }
        let (mut prog, interner) = lower(src);
        optimize(&mut prog, 2);
        let main = find_fn(&prog, &interner, "main");
        let calls = main
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, wukong_mir::Op::Call { .. }))
            .count();
        assert_eq!(calls, 0, "the whole helper chain should be gone from main");
    }

    #[test]
    fn a_non_leaf_helper_is_inlined() {
        // The old rule refused any callee that called another user function, no matter how small.
        // `wrap` calls `inner`, so it was permanently un-inlinable; now it collapses.
        let src = "fn inner(x: i32) -> i32 { return x + 5; } \
                   fn wrap(x: i32, y: i32) -> i32 { return inner(x) * inner(y); } \
                   fn main() -> i32 { let a: i32 = 3; return wrap(a, a + 1); }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 72, "level {lvl}");
        }
        let (mut prog, interner) = lower(src);
        optimize(&mut prog, 2);
        let main = find_fn(&prog, &interner, "main");
        assert!(
            !main.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, wukong_mir::Op::Call { .. })),
            "a non-leaf helper must be inlinable"
        );
    }

    #[test]
    fn mutual_recursion_is_not_inlined() {
        // `is_even`/`is_odd` call each other, so they share one SCC of the call graph and neither
        // may be spliced — splicing either into the other could not terminate. The leaf-only rule
        // got this right by accident (neither is a leaf); the SCC rule gets it right on purpose.
        let src = "fn is_even(n: i32) -> i32 { if n == 0 { return 1; } return is_odd(n - 1); } \
                   fn is_odd(n: i32) -> i32 { if n == 0 { return 0; } return is_even(n - 1); } \
                   fn main() -> i32 { return is_even(10) + is_odd(7); }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 2, "level {lvl}");
        }
        let (mut prog, interner) = lower(src);
        optimize(&mut prog, 2);
        for name in ["is_even", "is_odd"] {
            let f = find_fn(&prog, &interner, name);
            let calls = f
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(i.op, wukong_mir::Op::Call { .. }))
                .count();
            assert!(calls >= 1, "{name} must keep its mutually recursive call");
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

    /// Count `alloca ptr` slots and `load`s whose result is a pointer — the two symptoms of a
    /// base pointer that lives in memory and is re-fetched at every element access.
    fn count_ptr_slots_and_reloads(f: &wukong_mir::Function) -> (usize, usize) {
        use wukong_mir::{MirType, Op};
        let mut slots = 0;
        let mut reloads = 0;
        for i in f.blocks.iter().flat_map(|b| &b.insts) {
            match &i.op {
                Op::Alloca(MirType::Ptr) => slots += 1,
                Op::Load(_, MirType::Ptr) => reloads += 1,
                _ => {}
            }
        }
        (slots, reloads)
    }

    #[test]
    fn pointer_parameters_never_reach_a_stack_slot() {
        // `mir_build` USED to give every pointer-typed parameter — `*T`, `&T` and every `Tensor[…]`
        // — an `alloca ptr` + `store <param>` in the entry block, then re-`load` that base pointer
        // at EVERY element access: three redundant `load ptr`s per iteration here (x, y and o) that
        // no other pass could remove, because `licm` will not hoist a load out of a loop and `cse`
        // cannot forward one across a block boundary. `mem2reg` pointer promotion was written to
        // clean that up after the fact.
        //
        // Binding a statically-shaped tensor parameter as its buffer removed the spill at the
        // SOURCE, so the slot is never created and there is nothing left to promote. This test
        // therefore now pins the stronger property — the front end emits no pointer-parameter slot
        // at all, at `-O0` — and it fails loudly if the spill is ever reintroduced.
        //
        // `mem2reg`'s pointer-promotion path is still live and still covered, by
        // `mem2reg_promotes_an_inlined_callees_pointer_parameter` (inlining splices a callee's
        // `alloca ptr` into the middle of a caller block), plus
        // `mem2reg_ptr_promotion_keeps_the_pointee_in_memory` and
        // `mem2reg_leaves_a_late_initialized_pointer_slot_in_memory` for its refusal cases.
        //
        // The data-dependent branch keeps every elementwise recognizer from firing, so this is
        // general code, not a dispatched kernel.
        let src = "fn work(x: Tensor[f32, 8], y: Tensor[f32, 8], mut o: Tensor[f32, 8]) { \
                     for i in 0..8 { \
                       if x[i] > y[i] { o[i] = x[i] * 3.0; } else { o[i] = y[i] * 2.0; } } } \
                   fn main() -> i32 { \
                     let a: [f32; 8] = [1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0]; \
                     let b: [f32; 8] = [4.0; 8]; \
                     let mut c: [f32; 8] = [0.0; 8]; \
                     work(a, b, c); \
                     return (c[0] as i32) * 100 + (c[1] as i32); }";
        // Counted over the WHOLE program, not one named function: `work` is small enough that the
        // caller may inline it, and this property is about the program, not about where the loop
        // happens to live.
        let ptr_traffic = |p: &wukong_mir::Program| {
            p.funcs
                .iter()
                .map(count_ptr_slots_and_reloads)
                .fold((0, 0), |(a, b), (c, d)| (a + c, b + d))
        };

        let (mut prog, mut interner) = lower(src);
        let (slots, reloads) = ptr_traffic(&prog);
        assert_eq!(
            (slots, reloads),
            (0, 0),
            "a tensor parameter must bind as its buffer, not spill to a slot; \
             got {slots} slots / {reloads} reloads at -O0"
        );
        optimize(&mut prog, 2);
        let (slots, reloads) = ptr_traffic(&prog);
        assert_eq!(
            (slots, reloads),
            (0, 0),
            "pointer parameter slots must be promoted and their reloads gone"
        );
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        // b[i] = 4 wins at i=0 (4*2 = 8), a[i] = 9 wins at i=1 (9*3 = 27).
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 827);
        assert_eq!(run_main_opt(src, 0), 827);
    }

    #[test]
    fn mem2reg_promotes_an_inlined_callees_pointer_parameter() {
        // `-O2` splices a callee's entry block into the MIDDLE of a caller block, so the pointer
        // parameter's `alloca ptr` + `store` no longer sit in the entry block — yet the reload it
        // guards is inside the callee's own loop, which is the worst place to leave one. The slot
        // still qualifies because its initializing store is in its own alloca's block and that
        // block's dominance frontier is empty, so the store dominates every point the renamer asks
        // about. Checked across every function so the assertion holds whether or not `sum` inlines.
        let src = "fn sum(p: *f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
                     while i < n { s = s + p[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { let a: [f32; 4] = [1.0, 2.0, 4.0, 8.0]; \
                     return sum(&a[0], 4) as i32; }";
        let (mut prog, mut interner) = lower(src);
        optimize(&mut prog, 2);
        let totals = prog
            .funcs
            .iter()
            .map(count_ptr_slots_and_reloads)
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        assert_eq!(
            totals,
            (0, 0),
            "an inlined pointer parameter must still be promoted"
        );
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 15);
        assert_eq!(run_main_opt(src, 0), 15);
    }


    #[test]
    fn mem2reg_ptr_promotion_keeps_the_pointee_in_memory() {
        // Promoting the *pointer* must not promote what it points at. `p` starts at `&a` and
        // becomes `&b`, so once promoted it is a loop-header block parameter that merges the two
        // addresses, and every `*p` is a load/store through that parameter. Both `a` and `b` must
        // therefore stay in memory — the escape test now sees them as `br` arguments, which is the
        // only thing keeping them there once `store a_slot, p_slot` is gone.
        //
        // Iteration 0 writes a = 1 + 10; iteration 1 writes b = 2 + 10; the answer is 1112.
        let src = "fn main() -> i32 { let mut a: i32 = 1; let mut b: i32 = 2; \
                   let mut i: i32 = 0; let mut p: *mut i32 = &mut a; \
                   while i < 2 { *p = *p + 10; p = &mut b; i = i + 1; } \
                   return a * 100 + b; }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 1112, "level {lvl}");
        }
        let (mut prog, interner) = lower(src);
        optimize(&mut prog, 2);
        let (slots, _) = count_ptr_slots_and_reloads(find_fn(&prog, &interner, "main"));
        assert_eq!(slots, 0, "the pointer itself must still be promoted here");
    }

    #[test]
    fn mem2reg_leaves_a_late_initialized_pointer_slot_in_memory() {
        // A pointer slot first written *outside* the entry block can be read before it is written
        // on some path. Promotion would have to materialize an "undefined" pointer, and there is no
        // sound one: `inttoptr 0` is null natively but a valid, addressable slot in the interpreter.
        // Such a slot must stay in memory. Here `p`'s store is inside the loop body, so the entry
        // block never initializes it.
        let src = "fn main() -> i32 { let mut a: i32 = 5; let mut t: i32 = 0; \
                   let mut i: i32 = 0; \
                   while i < 3 { let p: *mut i32 = &mut a; t = t + *p; i = i + 1; } \
                   return t; }";
        let (mut prog, mut interner) = lower(src);
        optimize(&mut prog, 2);
        let (slots, _) = count_ptr_slots_and_reloads(find_fn(&prog, &interner, "main"));
        assert_eq!(slots, 1, "a late-initialized pointer slot must stay in memory");
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 15);
        assert_eq!(run_main_opt(src, 0), 15);
    }

}
