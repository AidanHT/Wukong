//! `wukong_opt` — the optimizer: a pass manager plus MIR transforms.
//!
//! Passes run to a fixpoint at `-O1` and above, in this order. The pipeline first promotes stack
//! slots to SSA registers, which is what makes the value-based transforms bite:
//!  * **sroa** — split an aggregate slot whose every access is at a constant position into one
//!    scalar slot per accessed leaf, for mem2reg to promote in the same sweep (`-O2`).
//!  * **mem2reg** — promote scalar and pointer `alloca`/`load`/`store` to block-parameter SSA.
//!  * **simplify** — constant folding and algebraic identities (`x+0`, `x*1`, `x*0`, ...).
//!  * **jump-thread** — send an edge that already decides a block's branch (a constant argument,
//!    or a dominating compare of the same values) straight to the decided successor, through a
//!    private copy of the block (`-O2`).
//!  * **simplify-cfg** — fold constant branches, merge straight-line blocks, prune dead blocks.
//!  * **simplify-phis** — drop dead/trivial block parameters mem2reg introduced.
//!  * **dce** — remove pure instructions whose results are never used, and unused allocas.
//!  * **cse** — global value numbering: dominator-tree value numbering, load forwarding along
//!    single-predecessor edges and through joins (intersecting the incoming memory states),
//!    block-parameter numbering, and phi insertion for values every arm of a join computed (`-O2`).
//!  * **dse** — dead-store elimination: must-alias overwrites in a block and along every path,
//!    end-of-lifetime stores to private slots, and stores into slots nothing reads (`-O2`).
//!  * **loop-canon** — one shape per loop: a preheader, a single latch, one exit-test polarity
//!    (`-O2`).
//!  * **loop-delete** — remove a provably finite loop that has no observable effect, such as an
//!    initialization loop dse emptied (`-O2`).
//!  * **licm** — hoist loop-invariant work into an existing preheader, including loads that run on
//!    every iteration, behind a first-iteration guard (`-O2`).
//!  * **bce** — slice bounds-check elimination by loop versioning on a preheader guard (`-O2`).
//!  * **promote** — keep a loop-invariant memory location the loop reads and writes in a register:
//!    one load in the preheader, one store per exit (`-O2`).
//!  * **vectorize** — widen a canonical loop body to 128-bit SIMD (if-converting a conditional
//!    body to a lane mask), with the original loop kept as its scalar epilogue (`-O2`). A body
//!    with a raw-AVX2 recipe form and a trip of 2048 or more instead calls a 256-bit kernel
//!    (`Op::VecKernelCall`) ahead of the original loop, which the 128-bit path then widens as the
//!    remainder ([`OptimizeOptions::vec_kernels`]).
//!
//! `cse`, `dse` and `licm` all consult [`alias`], the provenance analysis that answers *can a store
//! through `q` be seen by a load through `p`?* — the question `mir_build`'s type erasure
//! (`Ty::Ptr`/`Ref`/`Tensor`/`Slice` all become a bare `MirType::Ptr`) otherwise makes unanswerable.
//! It is a pure analysis, safe to build from anywhere, and the Cranelift backend uses it too.
//!
//! Functions are optimized callees first, and each call-graph SCC is summarized from its optimized
//! body as soon as it is done: a callee whose calls have no side effect is then value-numbered,
//! deleted when dead, and hoisted when invariant, like any instruction (the `effects` module).
//!
//! From `-O1` the fixpoint also runs **tre** ([`Tre`]): a function's tail calls to itself (and
//! integer accumulator recursion such as `n * fact(n - 1)`) become a loop. At `-O2`, between
//! `loop-canon` and `licm`, **fuse** ([`LoopFuse`]) merges two adjacent counted loops over the same
//! range when no dependence crosses them; and between `bce` and `promote`, **unswitch**
//! ([`LoopUnswitch`]) splits a counted loop at the iteration where a guard on its induction variable
//! changes, or unswitches a branch on an invariant.
//!
//! At `-O2` and above, whole-program inlining ([`inline_program`]) runs once *before* the
//! per-function pipeline, and loop unrolling ([`unroll_program`] — complete for a small known trip
//! count, partial otherwise) runs once *after* it — a reduced pipeline is then re-run over the
//! functions unrolling changed, so the duplicated bodies get the same simplify/cse/dce treatment as
//! everything else. Then straight-line **SLP** ([`slp_program`]) packs `W` isomorphic stores to
//! consecutive elements of one object — a complete unroll's copies, a hand-written `vec4` — into
//! one 128-bit vector store, and the same reduced pipeline cleans the functions it packed (not in a
//! program autodiff will differentiate: [`OptimizeOptions::AUTODIFF_INPUT`]). Last of
//! all, [`FormSwitches`] turns each chain of equality tests on one
//! scrutinee into a single `switch` terminator (once, after every loop transform — see
//! `switch_form`). `-O3` adds nothing to any of it — see
//! [`PassManager::standard`].

pub mod alias;
mod assert_sites;
mod bce;
mod cache;
mod cfg;
pub mod coalesce;
mod cse;
mod dce;
mod dom;
mod dse;
mod effects;
mod fxhash;
mod inline;
mod jump_thread;
mod licm;
mod loop_canon;
mod loop_delete;
pub mod loop_info;
mod mem2reg;
mod parallel;
mod phi;
mod promote;
mod simplify;
mod simplify_cfg;
pub mod slot_lifetime;
mod slp;
mod sroa;
mod unroll;
mod unroll_jam;
mod vectorize;
mod verify_each;

mod branch_fold;
mod fuse;
mod switch_form;
mod tre;
mod unswitch;

pub use verify_each::{verify_program_stage, VerificationFailure};

pub use alias::{type_bytes, AliasInfo, Prov};
pub use assert_sites::strip_assert_sites;
pub use bce::Bce;
pub use cache::CfgAnalyses;
pub use coalesce::{
    analyze_function as coalesce_function, CoalesceHint, CoalescePlan, VectorWidth,
};
pub use cse::Cse;
pub use dce::Dce;
pub use dse::Dse;
pub use inline::inline_program;
pub use jump_thread::JumpThread;
pub use licm::Licm;
pub use loop_canon::LoopCanon;
pub use loop_delete::LoopDelete;
pub use mem2reg::Mem2Reg;
pub use phi::SimplifyPhis;
pub use promote::Promote;
pub use simplify::Simplify;
pub use simplify_cfg::SimplifyCfg;
pub use slot_lifetime::{CaptureSummary, KernelOverwrite, SlotLifetimes};
pub use slp::slp_program;
pub use sroa::Sroa;
pub use unroll::unroll_program;
pub use vectorize::Vectorize;

pub use fuse::LoopFuse;
pub use switch_form::{form_switches, FormSwitches};
pub use tre::Tre;
pub use unswitch::LoopUnswitch;

use std::sync::Arc;
use std::time::{Duration, Instant};

use effects::{CallEffects, CallGraph};
use parallel::Parallelism;
use wukong_mir::{Function, Op, Program, Terminator, ValueId};

/// A function-level transform. Returns whether it changed anything.
///
/// `cache` holds the CFG/dominator analyses for `f`, shared across every pass in the fixpoint. A
/// pass reads whatever it needs from it (computed lazily, once) and — if it changes the block set or
/// any terminator's successor edges — must call [`CfgAnalyses::invalidate`]. Passes that only touch
/// instructions, block parameters, or edge arguments leave the cache alone.
///
/// A pass that is also `Send + Sync` and keeps no state between calls (anything it must remember
/// about one function belongs in that function's `cache`) can be handed to
/// [`PassManager::add_shared`], and then runs on several functions at once.
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
    /// Per-pass accumulated wall time and invocation count, in pipeline order. When functions were
    /// optimized on several threads ([`Timings::threads`] above 1) each pass's time is the sum over
    /// those threads, so the column can add up to more than [`Timings::total`].
    pub per_pass: Vec<PassStat>,
    /// Whole-program inlining time (runs at `-O2`+).
    pub inline: Duration,
    /// Loop-unrolling time (runs at `-O2`+, after the pipeline): the pass itself **plus** the
    /// reduced cleanup sweep over the functions it changed, which together are the whole cost
    /// unrolling adds. Deliberately not split across the per-pass buckets — those are keyed by pass
    /// index within [`PassManager::standard`], and the cleanup pipeline numbers its passes
    /// differently.
    pub unroll: Duration,
    /// Straight-line SLP time (runs at `-O2`+, after unrolling): the pass plus the cleanup sweep
    /// over the functions it packed, bucketed together for the same reason as [`Timings::unroll`].
    pub slp: Duration,
    /// Total time inside `optimize` (inlining + every pass + fixpoint bookkeeping).
    pub total: Duration,
    /// The largest per-function fixpoint iteration count observed across the program.
    pub max_iterations: u32,
    /// The most threads one batch of the function pipeline ran on: 1 when every function was
    /// optimized on the calling thread ([`OptimizeOptions::threads`]).
    pub threads: usize,
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

    /// Fold in the timings one function's fixpoint gathered on its own (a parallel batch times each
    /// function separately). Sums and a maximum, so the order functions are folded in is immaterial.
    fn absorb(&mut self, other: Timings) {
        for (idx, stat) in other.per_pass.into_iter().enumerate() {
            if self.per_pass.len() <= idx {
                self.per_pass.resize(
                    idx + 1,
                    PassStat {
                        name: stat.name,
                        time: Duration::ZERO,
                        calls: 0,
                    },
                );
            }
            let s = &mut self.per_pass[idx];
            if stat.calls > 0 {
                s.name = stat.name;
            }
            s.time += stat.time;
            s.calls += stat.calls;
        }
        self.max_iterations = self.max_iterations.max(other.max_iterations);
    }
}

/// An ordered list of passes, run to a per-function fixpoint.
///
/// A manager whose passes were all added with [`PassManager::add_shared`] optimizes the independent
/// functions of a program on several threads at once ([`OptimizeOptions::threads`]), with output
/// byte-identical to one thread. A single pass added with [`PassManager::add`] — which may hold
/// thread-confined state, like a counting or printing wrapper — keeps the whole manager on the
/// calling thread, running functions one at a time in the original order.
pub struct PassManager {
    passes: PassList,
    /// `0`: defer to `WUKONG_OPT_THREADS`, then the host ([`OptimizeOptions::threads`]).
    threads: u32,
}

/// The passes of a [`PassManager`], in order.
enum PassList {
    /// Every pass may run on several functions at once from different threads.
    Shared(Vec<Box<dyn Pass + Send + Sync>>),
    /// At least one pass is confined to the calling thread.
    Local(Vec<Box<dyn Pass>>),
}

impl PassManager {
    pub fn new() -> PassManager {
        PassManager {
            passes: PassList::Shared(Vec::new()),
            threads: 0,
        }
    }

    /// Append a pass. It may hold state confined to the calling thread (an `Rc`, a `Cell`, a
    /// thread-local), so a manager holding one runs every function on the calling thread, in the
    /// order it always has.
    pub fn add(&mut self, p: Box<dyn Pass>) {
        if let PassList::Shared(shared) = &mut self.passes {
            let local: Vec<Box<dyn Pass>> = std::mem::take(shared)
                .into_iter()
                .map(|q| -> Box<dyn Pass> { q })
                .collect();
            self.passes = PassList::Local(local);
        }
        match &mut self.passes {
            PassList::Local(local) => local.push(p),
            PassList::Shared(_) => unreachable!("converted above"),
        }
    }

    /// Append a pass that can run on several functions at once, each on its own thread with its own
    /// [`CfgAnalyses`]. Every pass of the standard pipelines is one.
    pub fn add_shared(&mut self, p: Box<dyn Pass + Send + Sync>) {
        match &mut self.passes {
            PassList::Shared(shared) => shared.push(p),
            PassList::Local(local) => local.push(p),
        }
    }

    /// Set how many threads the per-function work may use: `0` (the default) defers to the
    /// `WUKONG_OPT_THREADS` environment variable and then to the host, `1` is the sequential
    /// schedule ([`OptimizeOptions::threads`]). Never changes the output.
    pub fn set_threads(&mut self, threads: u32) {
        self.threads = threads;
    }

    /// [`PassManager::set_threads`], by value.
    pub fn with_threads(mut self, threads: u32) -> PassManager {
        self.threads = threads;
        self
    }

    /// The standard pipeline for an optimization level. `-O0` does nothing. The only level tests are
    /// `>= 1` and `>= 2`, so `-O3` builds the identical pipeline as `-O2` — a contract the
    /// differential gates rely on; do not add an `-O3`-only pass without revisiting it.
    pub fn standard(opt_level: u8) -> PassManager {
        PassManager::standard_with(opt_level, OptimizeOptions::FULL)
    }

    /// [`PassManager::standard`] with the per-pass switches `options` carries: whether the
    /// vectorizer may emit a 256-bit kernel ([`OptimizeOptions::vec_kernels`]), and the thread count
    /// ([`OptimizeOptions::threads`]). Every pass and their order are the same for every `options`.
    pub fn standard_with(opt_level: u8, options: OptimizeOptions) -> PassManager {
        let mut pm = PassManager::new().with_threads(options.threads);
        if opt_level >= 1 {
            // Promotion comes first: every later pass is far more effective on SSA values than on
            // memory traffic. The whole list then runs to a fixpoint. At `-O2` scalar replacement
            // runs just ahead of mem2reg, so an aggregate it splits is promoted in the same sweep.
            // `-O1` keeps the plain promotion shape the autodiff transform is built against.
            if opt_level >= 2 {
                pm.add_shared(Box::new(Sroa));
            }
            pm.add_shared(Box::new(Mem2Reg));
            pm.add_shared(Box::new(Simplify));
            if opt_level >= 2 {
                // An edge that already decides a block's branch skips it, through a private copy
                // of the block (`jump_thread`); simplify-cfg then merges the copy into its
                // predecessor and removes whatever the threading left without predecessors.
                pm.add_shared(Box::new(JumpThread));
            }
            pm.add_shared(Box::new(SimplifyCfg));
            pm.add_shared(Box::new(SimplifyPhis));
            pm.add_shared(Box::new(Dce));
            // A self-call in tail position becomes a branch back to the top (`tre`).
            pm.add_shared(Box::new(Tre));
        }
        if opt_level >= 2 {
            // CSE feeds Simplify/DCE more constants and dead values; the fixpoint loop reruns all.
            pm.add_shared(Box::new(Cse));
            pm.add_shared(Box::new(Dse));
            // Put every loop into one shape before LICM runs: LICM hoists only into a preheader
            // that already exists, so a loop given one here becomes hoistable in the same sweep.
            pm.add_shared(Box::new(LoopCanon));
            // A loop dse emptied still runs every trip; with a preheader now guaranteed, delete
            // whichever loops provably terminate without doing anything observable.
            pm.add_shared(Box::new(LoopDelete));
            // Fuse adjacent counted loops over the same range once they are canonical, before LICM
            // fills the second loop's preheader with loads (`fuse`).
            pm.add_shared(Box::new(LoopFuse));
            // LICM hoists invariant work out of loops; rerunning the pipeline then cleans up and
            // can expose further invariants (e.g. across nested loops).
            pm.add_shared(Box::new(Licm));
            // Slice bounds checks are extra loop exits, which no later loop transform accepts:
            // version each counted loop on one preheader guard so its proved checks disappear
            // (`bce.rs`). After LICM, so invariant addresses and lengths are already hoisted; before
            // unswitching and the vectorizer, so the check-free, single-exit copy is what they see.
            pm.add_shared(Box::new(Bce));
            // With invariant conditions hoisted, decide a guard on the induction variable or on an
            // invariant once, by splitting or unswitching the loop, so the vectorizer below sees
            // branch-free bodies (`unswitch`).
            pm.add_shared(Box::new(LoopUnswitch));
            // With the invariant loads hoisted (and a slice loop versioned into a check-free copy
            // with a single exit), a location the loop still both reads and writes is a
            // memory-carried reduction: keep it in a register for the loop's duration.
            pm.add_shared(Box::new(Promote));
            // Widening comes last, on the cleanest MIR the pipeline produces: canonical loops with
            // one latch and one exit-test polarity, invariants already hoisted, and dead code gone.
            // Everything it emits is fed back through the fixpoint, so the vector body gets the
            // same simplification, CSE and DCE the scalar body did.
            pm.add_shared(Box::new(Vectorize::new(options.vec_kernels)));
        }
        pm
    }

    /// The reduced pipeline re-run over the functions [`unroll_program`] changed.
    ///
    /// Unrolling duplicates a loop body, so what the result needs is value numbering over the now
    /// four-fold address arithmetic and the dead-code removal that follows it. It does not need
    /// `Mem2Reg` (the pass refuses to duplicate a header or body containing an `Alloca`, so there is
    /// no new stack slot to promote), `Dse`, or `Licm` (LICM already ran over this same body, and
    /// every instruction the copies add depends on the induction variable). The synthesized
    /// wrap-safe preheader guard contains new arithmetic and a select that `Simplify` folds when
    /// the bound is constant. A completely unrolled loop leaves its header as a straight-line
    /// block with a single predecessor: `SimplifyCfg` merges it into the preheader, which turns the
    /// header parameters into the entry values, `Simplify` then folds every copy's induction
    /// variable into a constant, and `SimplifyPhis` drops what is left of the parameters. (A
    /// constant trip count needs no remainder loop at all — the unroller knows it exactly.)
    ///
    /// Public, with [`PassManager::run_subset`], so that a measurement replica of [`optimize`] can
    /// wrap exactly these passes and drive them exactly as `optimize` does.
    pub fn unroll_cleanup() -> PassManager {
        let mut pm = PassManager::new();
        pm.add_shared(Box::new(Simplify));
        pm.add_shared(Box::new(SimplifyCfg));
        pm.add_shared(Box::new(SimplifyPhis));
        pm.add_shared(Box::new(Cse));
        pm.add_shared(Box::new(Dce));
        pm
    }

    fn is_empty(&self) -> bool {
        match &self.passes {
            PassList::Shared(passes) => passes.is_empty(),
            PassList::Local(passes) => passes.is_empty(),
        }
    }

    pub fn run(&self, program: &mut Program) {
        // Preserve the legacy library API's debug assertion, while the fallible API below is
        // available to release callers and never turns a verifier error into a panic.
        self.run_program(
            program,
            None,
            cfg!(debug_assertions),
            Parallelism::resolve(self.threads),
        )
        .unwrap_or_else(|failure| panic!("{failure}"));
    }

    /// Run with fallible verification in every build profile. Check each pass that actually ran,
    /// including one reporting no mutation, and whole-program call signatures at the boundaries.
    pub fn run_verified(&self, program: &mut Program) -> Result<(), VerificationFailure> {
        verify_program_stage(program, "pipeline input")?;
        self.run_program(program, None, true, Parallelism::resolve(self.threads))?;
        verify_program_stage(program, "pipeline output")
    }

    /// Run the pipeline over just the functions at `funcs`, in that order — what [`optimize`] does
    /// to the functions loop unrolling changed. Every function's passes see the call effects of the
    /// whole program *as it stands*, summarized afresh, so a callee whose loop was just unrolled
    /// away is judged by its new body. Public so a measurement replica that wraps these passes runs
    /// the identical sequence (`wukong_bench compile-work` checks it byte for byte).
    pub fn run_subset(&self, program: &mut Program, funcs: &[usize]) {
        self.run_subset_checked(
            program,
            funcs,
            cfg!(debug_assertions),
            Parallelism::resolve(self.threads),
        )
        .unwrap_or_else(|failure| panic!("{failure}"));
    }

    /// [`PassManager::run_subset`] with fallible verification in every build profile: each pass
    /// that ran is checked, and the first failure names it and the function — exactly the check
    /// [`optimize_verified`] gives its unroll and SLP cleanup sweeps, so a replica of that can
    /// report a failure the same way.
    pub fn run_subset_verified(
        &self,
        program: &mut Program,
        funcs: &[usize],
    ) -> Result<(), VerificationFailure> {
        self.run_subset_checked(program, funcs, true, Parallelism::resolve(self.threads))
    }

    /// The subset sweep. The call-effect table is summarized once, before any of `funcs` changes,
    /// and nothing updates it while they run — so the functions are independent of one another and
    /// may be optimized at once. A repeated index is optimized twice, in order, on the calling
    /// thread.
    fn run_subset_checked(
        &self,
        program: &mut Program,
        funcs: &[usize],
        verify_each: bool,
        par: Parallelism,
    ) -> Result<(), VerificationFailure> {
        if funcs.is_empty() || self.is_empty() {
            return Ok(());
        }
        let effects = Arc::new(CallEffects::of(program));
        match &self.passes {
            PassList::Shared(passes) if !par.is_sequential() && all_distinct(funcs) => {
                let (results, _) =
                    parallel::try_map(par, parallel::select_mut(&mut program.funcs, funcs), |f| {
                        fixpoint(passes, f, None, verify_each, Some(Arc::clone(&effects)))
                    });
                results.into_iter().collect()
            }
            PassList::Shared(passes) => {
                subset_sequential(passes, program, funcs, verify_each, &effects)
            }
            PassList::Local(passes) => {
                subset_sequential(passes, program, funcs, verify_each, &effects)
            }
        }
    }

    /// Run the pipeline over every function, callees before callers, summarizing each call-graph
    /// SCC as soon as its members are optimized so that their callers — processed later — can treat
    /// side-effect-free calls like instructions. On one thread (or with a thread-confined pass) that
    /// is one SCC after another in callees-first order; otherwise one call-graph level after another,
    /// a level's functions at once ([`program_by_levels`]). The MIR is the same either way.
    fn run_program(
        &self,
        program: &mut Program,
        timings: Option<&mut Timings>,
        verify_each: bool,
        par: Parallelism,
    ) -> Result<(), VerificationFailure> {
        match &self.passes {
            PassList::Shared(passes) if !par.is_sequential() => {
                program_by_levels(passes, program, timings, verify_each, par)
            }
            PassList::Shared(passes) => program_sequential(passes, program, timings, verify_each),
            PassList::Local(passes) => program_sequential(passes, program, timings, verify_each),
        }
    }
}

impl Default for PassManager {
    fn default() -> PassManager {
        PassManager::new()
    }
}

/// Is every index of `funcs` different?
fn all_distinct(funcs: &[usize]) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    funcs.iter().all(|&i| seen.insert(i))
}

/// The sequential subset sweep: `funcs` in order, stopping at the first failure.
fn subset_sequential<P: Pass + ?Sized>(
    passes: &[Box<P>],
    program: &mut Program,
    funcs: &[usize],
    verify_each: bool,
    effects: &Arc<CallEffects>,
) -> Result<(), VerificationFailure> {
    for &i in funcs {
        fixpoint(
            passes,
            &mut program.funcs[i],
            None,
            verify_each,
            Some(Arc::clone(effects)),
        )?;
    }
    Ok(())
}

/// The sequential schedule: one SCC after another in callees-first order, each summarized as soon
/// as its members are done.
fn program_sequential<P: Pass + ?Sized>(
    passes: &[Box<P>],
    program: &mut Program,
    mut timings: Option<&mut Timings>,
    verify_each: bool,
) -> Result<(), VerificationFailure> {
    if passes.is_empty() {
        return Ok(());
    }
    let mut effects = Arc::new(CallEffects::default());
    let graph = CallGraph::of(program);
    for (scc, members) in graph.sccs.iter().enumerate() {
        for &i in members {
            fixpoint(
                passes,
                &mut program.funcs[i as usize],
                timings.as_deref_mut(),
                verify_each,
                Some(Arc::clone(&effects)),
            )?;
        }
        // Every function's cache (and its handle on the table) is gone by now, so this updates
        // the table in place.
        graph.summarize(program, scc, Arc::make_mut(&mut effects));
    }
    Ok(())
}

/// The parallel schedule: one call-graph *level* at a time ([`CallGraph::levels`]). Every function
/// of a level is optimized against the table as it stands after the levels below — which holds the
/// final summary of every callee any of them has, exactly as the sequential schedule would show it
/// (`effects` module note) — so a level's functions are independent tasks ([`parallel::try_map`]),
/// and its SCCs are then summarized in SCC order.
///
/// A failure is reported as the sequential schedule would report it: the earliest failing function
/// in SCC order, whether it failed verification or panicked. A panic comes back from its batch as a
/// value ([`parallel::Failed::Panic`]) and is recorded exactly like a verifier failure; after
/// either, only SCCs earlier than it are still optimized — any of them could fail first in
/// sequential order, and none of them can call it — and only once every level is done is the
/// earliest failure returned, or resumed if it was a panic.
fn program_by_levels(
    passes: &[Box<dyn Pass + Send + Sync>],
    program: &mut Program,
    mut timings: Option<&mut Timings>,
    verify_each: bool,
    par: Parallelism,
) -> Result<(), VerificationFailure> {
    if passes.is_empty() {
        return Ok(());
    }
    let graph = CallGraph::of(program);
    let mut effects = Arc::new(CallEffects::default());
    let mut failed: Option<(usize, parallel::Failed<VerificationFailure>)> = None;
    let timed = timings.is_some();
    for level in graph.levels() {
        let bound = failed.as_ref().map_or(usize::MAX, |(scc, _)| *scc);
        let sccs: Vec<usize> = level.into_iter().filter(|&scc| scc < bound).collect();
        if sccs.is_empty() {
            continue;
        }
        // The level's functions in sequential order: SCC order, members in SCC order.
        let tasks: Vec<(usize, usize)> = sccs
            .iter()
            .flat_map(|&scc| graph.sccs[scc].iter().map(move |&m| (scc, m as usize)))
            .collect();
        let indices: Vec<usize> = tasks.iter().map(|&(_, i)| i).collect();
        let table = &effects;
        let (results, threads) = parallel::try_map_caught(
            par,
            parallel::select_mut(&mut program.funcs, &indices),
            |f| {
                let mut own = timed.then(Timings::default);
                fixpoint(
                    passes,
                    f,
                    own.as_mut(),
                    verify_each,
                    Some(Arc::clone(table)),
                )
                .map(|()| own)
            },
        );
        // `try_map_caught` reports up to the first failure (or panic) in this order, which is the
        // earliest of the level; anything already recorded is later still, or it would have
        // filtered this level.
        for (k, result) in results.into_iter().enumerate() {
            match result {
                Ok(own) => {
                    if let (Some(acc), Some(own)) = (timings.as_deref_mut(), own) {
                        acc.absorb(own);
                    }
                }
                Err(failure) => failed = Some((tasks[k].0, failure)),
            }
        }
        if let Some(acc) = timings.as_deref_mut() {
            acc.threads = acc.threads.max(threads);
        }
        let bound = failed.as_ref().map_or(usize::MAX, |(scc, _)| *scc);
        for &scc in &sccs {
            if scc < bound {
                // Every task's handle on the table was dropped with its cache, so this updates
                // the table in place.
                graph.summarize(program, scc, Arc::make_mut(&mut effects));
            }
        }
    }
    match failed {
        Some((_, failure)) => Err(failure.into_err()),
        None => Ok(()),
    }
}

/// The fixpoint driver. `timings` is `None` on the production path (zero overhead); the bench
/// harness passes `Some(..)` to accumulate per-pass wall time. The `timings` branch does not
/// change which passes run or in what order, so the resulting MIR is identical either way.
fn fixpoint<P: Pass + ?Sized>(
    passes: &[Box<P>],
    f: &mut Function,
    mut timings: Option<&mut Timings>,
    verify_each: bool,
    effects: Option<Arc<CallEffects>>,
) -> Result<(), VerificationFailure> {
    // One analysis cache per function, shared across every pass and every fixpoint iteration.
    // It is invalidated by the passes that restructure the CFG (see their `run_function`), so it
    // stays valid — and reused — across the long tail of the fixpoint where only instructions
    // move. The cache never changes what a pass computes, only how often it recomputes it. It is
    // *managed*: every reported mutation is passed on below, which is what lets it hand one
    // function's alias facts to several memory passes in a row. It also carries the call
    // effects of the functions optimized so far.
    let mut cache = CfgAnalyses::managed(effects);
    // Fixpoint with per-pass clean-tracking. A pass is a *deterministic* function of the MIR, so a
    // pass that ran and reported "no change" cannot do anything until some OTHER pass mutates the
    // function — re-running it on identical MIR would again be a no-op. We therefore skip
    // known-clean passes, and a mutation re-dirties every pass (conservatively). This skips the
    // optimizer's final all-passes no-op *confirmation* sweep (and any pass already at fixpoint
    // mid-run) WITHOUT changing the sequence of mutations, so the resulting MIR is bit-identical
    // to the naive "run everything every sweep" fixpoint — the differential gate (-O0 ≡ -O{1,2,3},
    // interp ≡ native) and the debug verify below both still hold. ~20% off optimizer time.
    let mut clean = vec![false; passes.len()];
    let mut iterations = 0;
    loop {
        let mut changed = false;
        for (i, p) in passes.iter().enumerate() {
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
                cache.note_mutation();
                // A mutation may have created work for every pass again (including earlier ones
                // already run this sweep, and `p` itself). Re-dirty all; they re-run next sweep.
                clean.iter_mut().for_each(|c| *c = false);
            } else {
                clean[i] = true;
            }
            if verify_each {
                verify_each::verify_function_stage(f, p.name())?;
            }
        }
        iterations += 1;
        if !changed || iterations > 100 {
            break;
        }
    }
    if let Some(t) = timings {
        t.max_iterations = t.max_iterations.max(iterations);
    }
    Ok(())
}

/// The optimizer stages a caller can switch off independently of the `-O` level, and how many
/// threads the per-function work may use. Every entry point without a `_with` suffix runs
/// [`OptimizeOptions::FULL`] (also the [`Default`]); the level still decides whether a stage can
/// run at all (SLP needs `-O2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptimizeOptions {
    /// Straight-line SLP ([`slp_program`]) after unrolling, at `-O2` and above.
    pub slp: bool,
    /// May the vectorizer hand a widened loop to a 256-bit raw-AVX2 kernel (`Op::VecKernelCall`,
    /// `wukong_mir::vec_kernel`) instead of 128-bit vectors? The same switch as
    /// `wukong_mir_build::LowerOptions::vec_kernels`, and set from the same place: a kernel carries
    /// Win64 AVX2 machine code, which no PTX lowering can consume (`--backend=gpu-native` clears
    /// it) and a `--target-cpu` below x86-64-v3 must not receive. The host gate and the
    /// `WUKONG_P4_NO_256` kill-switch still apply when it is on; every setting computes the same
    /// values. The interpreter and the native backend are compiled with the same setting, so the
    /// differential compares one MIR.
    pub vec_kernels: bool,
    /// How many threads the per-function phases may use — the fixpoint pipeline over the functions
    /// of one call-graph level, unrolling, SLP, their cleanup sweeps and switch formation
    /// (`parallel` module). `0` defers to the `WUKONG_OPT_THREADS` environment variable, and an
    /// unset, `0` or unparseable variable means automatic: the host's parallelism (at most 8),
    /// used only for a batch with enough work to share. `1` is the sequential schedule; a larger
    /// count uses the pool for every batch of two or more functions. **The output never depends on
    /// it**: every setting produces byte-identical MIR.
    pub threads: u32,
}

impl OptimizeOptions {
    /// Every stage the level enables, on the automatic thread count.
    pub const FULL: OptimizeOptions = OptimizeOptions {
        slp: true,
        vec_kernels: true,
        threads: 0,
    };
    /// The pipeline for a program `wukong_autodiff` will differentiate (`--emit=grad`, `--train`,
    /// `--emit=obj --grad-of`): everything but SLP. Autodiff takes the *optimized* single-block
    /// forward and differentiates it one scalar at a time; it has no VJP rule for a vector load or
    /// store, so a packed bundle storing into a buffer whose gradient is live — in the loss, or in
    /// a helper `inline_calls` splices into it — turned a forward that differentiated into a hard
    /// error. The gradient-compile lowering (`unroll_for_autodiff`) and complete unrolling produce
    /// exactly the straight-line copies SLP packs, so this is the common case there, not an edge.
    /// The generated backward is never differentiated again, so it is optimized with
    /// [`OptimizeOptions::FULL`] under the invocation's kernel switch
    /// ([`OptimizeOptions::with_vec_kernels`]). The tape's own fill and ReLU-backward loops are
    /// plain elementwise bodies the vectorizer turns into kernels, so the backward needs the same
    /// target gate as the forward.
    ///
    /// No 256-bit kernels either: autodiff has no VJP rule for an `Op::VecKernelCall` and refuses
    /// one by name, so the forward keeps the 128-bit form it differentiated before.
    pub const AUTODIFF_INPUT: OptimizeOptions = OptimizeOptions {
        slp: false,
        vec_kernels: false,
        threads: 0,
    };

    /// These options with the vectorizer's 256-bit kernels switched on or off.
    pub const fn with_vec_kernels(self, vec_kernels: bool) -> OptimizeOptions {
        OptimizeOptions {
            vec_kernels,
            ..self
        }
    }

    /// These options with the thread count set ([`OptimizeOptions::threads`]; `1` is sequential).
    pub const fn with_threads(self, threads: u32) -> OptimizeOptions {
        OptimizeOptions { threads, ..self }
    }
}

impl Default for OptimizeOptions {
    fn default() -> OptimizeOptions {
        OptimizeOptions::FULL
    }
}

/// Optimize a whole program at the given level.
pub fn optimize(program: &mut Program, opt_level: u8) {
    optimize_preserving(program, opt_level, &[]);
}

/// Optimize normally while retaining externally selected function entry points. Calls to these
/// functions may still inline, and their bodies receive the full requested optimization pipeline.
pub fn optimize_preserving(program: &mut Program, opt_level: u8, roots: &[wukong_span::Symbol]) {
    optimize_preserving_with(program, opt_level, roots, OptimizeOptions::FULL);
}

/// Like [`optimize_preserving`], running only the stages `options` leaves on.
pub fn optimize_preserving_with(
    program: &mut Program,
    opt_level: u8,
    roots: &[wukong_span::Symbol],
    options: OptimizeOptions,
) {
    let par = Parallelism::resolve(options.threads);
    let debug = cfg!(debug_assertions);
    // Inlining is a whole-program transform (it needs callee bodies), so it runs before the
    // function-level pipeline — which then optimizes the spliced-in code in context.
    if opt_level >= 2 {
        inline::inline_program_preserving(program, roots);
    }
    PassManager::standard_with(opt_level, options)
        .run_program(program, None, debug, par)
        .unwrap_or_else(|failure| panic!("{failure}"));
    // Unrolling runs after the pipeline, not inside its fixpoint: it needs the canonical two-block
    // counted loop that mem2reg/simplify-cfg produce, and re-running it on its own output would
    // unroll the same loop again every sweep. Only the functions it touched are re-optimized, so a
    // program with no unrollable loop pays one scan and nothing more.
    if opt_level >= 2 {
        let changed = unroll::unroll_program_with(program, par);
        PassManager::unroll_cleanup()
            .run_subset_checked(program, &changed, debug, par)
            .unwrap_or_else(|failure| panic!("{failure}"));
        // Straight-line SLP, after unrolling so a complete unroll's copies are visible with their
        // counters folded; the packed functions get the same cleanup (`slp`).
        if options.slp {
            let packed = slp::slp_program_with(program, par);
            PassManager::unroll_cleanup()
                .run_subset_checked(program, &packed, debug, par)
                .unwrap_or_else(|failure| panic!("{failure}"));
        }
        // Last, and outside the fixpoint: a multi-way branch must not reach the loop passes.
        switch_form::form_switches_with(program, par);
    }
}

/// Optimize with per-pass verification enabled in both debug and release builds. Errors identify
/// the responsible pass/stage and function; no later pass runs after a verifier failure. This
/// reports invalid MIR, not arbitrary panics inside a transform. External runtime signatures are
/// outside MIR and cannot be cross-checked by the program verifier.
pub fn optimize_verified(program: &mut Program, opt_level: u8) -> Result<(), VerificationFailure> {
    optimize_verified_preserving(program, opt_level, &[])
}

/// Like [`optimize_verified`], retaining the selected entry points as in [`optimize_preserving`].
pub fn optimize_verified_preserving(
    program: &mut Program,
    opt_level: u8,
    roots: &[wukong_span::Symbol],
) -> Result<(), VerificationFailure> {
    optimize_verified_preserving_with(program, opt_level, roots, OptimizeOptions::FULL)
}

/// Like [`optimize_verified_preserving`], running only the stages `options` leaves on.
pub fn optimize_verified_preserving_with(
    program: &mut Program,
    opt_level: u8,
    roots: &[wukong_span::Symbol],
    options: OptimizeOptions,
) -> Result<(), VerificationFailure> {
    let par = Parallelism::resolve(options.threads);
    // O0 still validates its input; a malformed lowering must not reach even the first transform.
    verify_program_stage(program, "optimization input")?;
    if opt_level >= 2 {
        inline::inline_program_preserving(program, roots);
        verify_program_stage(program, "inline")?;
    }
    let pm = PassManager::standard_with(opt_level, options);
    verify_program_stage(program, "pipeline input")?;
    pm.run_program(program, None, true, par)?;
    verify_program_stage(program, "pipeline output")?;
    if opt_level >= 2 {
        let changed = unroll::unroll_program_with(program, par);
        verify_program_stage(program, "unroll")?;
        PassManager::unroll_cleanup().run_subset_checked(program, &changed, true, par)?;
        if options.slp {
            let packed = slp::slp_program_with(program, par);
            verify_program_stage(program, "slp")?;
            PassManager::unroll_cleanup().run_subset_checked(program, &packed, true, par)?;
        }
        switch_form::form_switches_with(program, par);
        verify_program_stage(program, "form-switches")?;
    }
    verify_program_stage(program, "optimization output")
}

/// Like [`optimize`], but returns an in-process [`Timings`] breakdown (whole-optimizer total,
/// inlining, and per-pass accumulated wall time + call counts). Produces MIR **byte-identical** to
/// [`optimize`] — it runs the same pipeline in the same order, only wrapped in `Instant` timing.
/// For the bench harness; the production compile path uses [`optimize`] and pays no timing cost.
pub fn optimize_timed(program: &mut Program, opt_level: u8) -> Timings {
    optimize_timed_with(program, opt_level, OptimizeOptions::FULL)
}

/// Like [`optimize_timed`], running only the stages `options` leaves on — byte-identical to
/// [`optimize_preserving_with`] with no extra roots and the same `options`, on the same threads.
pub fn optimize_timed_with(
    program: &mut Program,
    opt_level: u8,
    options: OptimizeOptions,
) -> Timings {
    let par = Parallelism::resolve(options.threads);
    let debug = cfg!(debug_assertions);
    let mut t = Timings {
        threads: 1,
        ..Timings::default()
    };
    let start = Instant::now();
    if opt_level >= 2 {
        let i0 = Instant::now();
        inline_program(program);
        t.inline = i0.elapsed();
    }
    PassManager::standard_with(opt_level, options)
        .run_program(program, Some(&mut t), debug, par)
        .unwrap_or_else(|failure| panic!("{failure}"));
    if opt_level >= 2 {
        // The unroll bucket is the pass *and* its cleanup sweep: that sum is the cost unrolling
        // actually adds, which is the number worth reporting. Keeping the cleanup out of the
        // per-pass table also avoids charging its `Simplify` to the standard pipeline's slot 0
        // (`Timings::record` keys by pass index, and the two pipelines number their passes
        // differently).
        let u0 = Instant::now();
        let changed = unroll::unroll_program_with(program, par);
        PassManager::unroll_cleanup()
            .run_subset_checked(program, &changed, debug, par)
            .unwrap_or_else(|failure| panic!("{failure}"));
        t.unroll = u0.elapsed();
        // The SLP bucket, likewise, is the stage and its cleanup.
        if options.slp {
            let s0 = Instant::now();
            let packed = slp::slp_program_with(program, par);
            PassManager::unroll_cleanup()
                .run_subset_checked(program, &packed, debug, par)
                .unwrap_or_else(|failure| panic!("{failure}"));
            t.slp = s0.elapsed();
        }
        // Counted in `total` only: one scan per function, and a cleanup only where one formed.
        switch_form::form_switches_with(program, par);
    }
    t.total = start.elapsed();
    t
}

// ---- shared use-visiting helpers ----

/// Apply `f` to every value *used* (read) by an operation, in place.
pub(crate) fn map_op_uses(op: &mut Op, mut f: impl FnMut(ValueId) -> ValueId) {
    match op {
        Op::Bin(_, a, b)
        | Op::Cmp(_, a, b)
        | Op::Narrow { lo: a, hi: b, .. }
        | Op::Shuffle { a, b, .. }
        | Op::InsertLane(a, b, _) => {
            *a = f(*a);
            *b = f(*b);
        }
        Op::Neg(a)
        | Op::Not(a)
        | Op::Cast(_, a, _)
        | Op::Load(a, _)
        | Op::Splat(a)
        | Op::ExtractLane(a, _)
        | Op::Widen { v: a, .. }
        | Op::Sqrt(a)
        | Op::Round(_, a)
        | Op::IntUnary(_, a) => {
            *a = f(*a);
        }
        Op::Select(c, a, b) | Op::Fma(c, a, b) => {
            *c = f(*c);
            *a = f(*a);
            *b = f(*b);
        }
        Op::Store { ptr, value } | Op::AtomicAdd { ptr, value } => {
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
        | Op::Alloca { .. }
        | Op::FuncAddr(..)
        | Op::GlobalAddr(..)
        | Op::Iota(..)
        | Op::Barrier => {}
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
        Terminator::Switch {
            scrutinee,
            cases,
            default_args,
            ..
        } => {
            *scrutinee = f(*scrutinee);
            for a in cases.iter_mut().flat_map(|c| c.args.iter_mut()) {
                *a = f(*a);
            }
            for a in default_args {
                *a = f(*a);
            }
        }
        Terminator::Ret(None) | Terminator::Unreachable => {}
    }
}

/// Visit (read-only) every value used by an operation.
pub(crate) fn each_op_use(op: &Op, f: &mut impl FnMut(ValueId)) {
    match op {
        Op::Bin(_, a, b)
        | Op::Cmp(_, a, b)
        | Op::Narrow { lo: a, hi: b, .. }
        | Op::Shuffle { a, b, .. }
        | Op::InsertLane(a, b, _) => {
            f(*a);
            f(*b);
        }
        Op::Neg(a)
        | Op::Not(a)
        | Op::Cast(_, a, _)
        | Op::Load(a, _)
        | Op::Splat(a)
        | Op::ExtractLane(a, _)
        | Op::Widen { v: a, .. }
        | Op::Sqrt(a)
        | Op::Round(_, a)
        | Op::IntUnary(_, a) => f(*a),
        Op::Select(c, a, b) | Op::Fma(c, a, b) => {
            f(*c);
            f(*a);
            f(*b);
        }
        Op::Store { ptr, value } | Op::AtomicAdd { ptr, value } => {
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
        | Op::Alloca { .. }
        | Op::FuncAddr(..)
        | Op::GlobalAddr(..)
        | Op::Iota(..)
        | Op::Barrier => {}
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
        Terminator::Switch {
            scrutinee,
            cases,
            default_args,
            ..
        } => {
            f(*scrutinee);
            cases.iter().flat_map(|c| &c.args).for_each(|a| f(*a));
            default_args.iter().for_each(|a| f(*a));
        }
        Terminator::Ret(None) | Terminator::Unreachable => {}
    }
}

/// Does this op have a side effect that prevents removing it even if its result is unused?
pub(crate) fn has_side_effects(op: &Op) -> bool {
    matches!(
        op,
        Op::Store { .. }
            | Op::Call { .. }
            | Op::VecKernelCall { .. }
            | Op::Barrier
            | Op::AtomicAdd { .. }
    )
}

#[cfg(test)]
mod atomic_tests;

#[cfg(test)]
mod parallel_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    #[test]
    fn aliased_recurrence_preserves_scalar_iterations_at_o0_and_o2() {
        let src = include_str!("../../../tests/run/parallel_dependence.wk");
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let recurrence = program.function(interner.intern("recurrence")).unwrap();
        assert!(
            recurrence.value_types.iter().all(|ty| !ty.is_vector()),
            "unguarded AST SIMD must not snapshot the aliased recurrence"
        );
        let expected = [4096, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            .iter()
            .map(|v| format!("{v}\n"))
            .collect::<String>();
        for level in [0, 2] {
            let mut optimized = program.clone();
            optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            assert!(!optimized.funcs.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.insts)
                .any(|inst| matches!(inst.op, Op::Call { func, .. } if interner.resolve(func) == "wukong_parallel_for")),
                "dependent iterations were threaded at O{level}");
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, interner.intern("main"), &interner)
                    .unwrap();
            assert_eq!(exit, 0);
            assert_eq!(stdout, expected.as_bytes(), "O{level}");
        }
    }

    #[test]
    fn aliased_recurrence_fix_keeps_disjoint_shifted_simd() {
        let src = "module m\n@generic\n\
            fn shifted(x: [i32; 9], mut out: [i32; 9]) {\n\
                for i in 0..8 { out[i + 1] = x[i] + 1; }\n}\n\
            fn main() -> i32 { let x: [i32; 9] = [1; 9];\n\
                let mut out: [i32; 9] = [0; 9]; shifted(x, out); return out[8]; }";
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let shifted = program.function(interner.intern("shifted")).unwrap();
        // Either vector form counts: on an AVX2+FMA3/Win64 host the eight-lane i32 body takes the
        // 256-bit recipe (a `veckernel` call, no `<N x i32>` values), elsewhere the 128-bit strips.
        assert!(
            shifted
                .value_types
                .iter()
                .any(wukong_mir::MirType::is_vector)
                || !shifted.vec_kernels.is_empty(),
            "proven-disjoint shifted arrays should retain AST SIMD"
        );
        for level in [0, 2] {
            let mut optimized = program.clone();
            optimize(&mut optimized, level);
            assert_eq!(
                wukong_interp::run(&optimized, interner.intern("main"), &interner).unwrap(),
                2
            );
        }
    }

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
        // Each trip is below one 256-bit register of its lane type (8 i32, 4 i64), so the loop keeps
        // the 128-bit `<N x iW>` strips on every host instead of becoming a raw-AVX2 `veckernel`
        // call, which carries no vector values for the fold to see.
        let cases = [
            (
                "fn main() -> i32 { let mut n: [i32; 7] = [8; 7]; \
                 for i in 0..7 { n[i] = n[i] - n[i]; } return n[0] + n[6] + 7; }",
                7,
            ),
            (
                "fn main() -> i32 { let mut n: [i64; 3] = [8; 3]; \
                 for i in 0..3 { n[i] = n[i] - n[i]; } return (n[0] + n[2] + 7) as i32; }",
                7,
            ),
            (
                "fn main() -> i32 { let mut n: [i32; 7] = [8; 7]; let mut o: [i32; 7] = [0; 7]; \
                 for i in 0..7 { o[i] = if n[i] < n[i] { 1 } else { 2 }; } \
                 return o[0] + o[6]; }",
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
    fn negated_f8_constants_round_once_from_the_literal() {
        // Every backend rounds an f8 `ConstFloat` ONCE, from its f64 value, onto the 8-bit grid. A
        // literal just past an f8 midpoint can sit within f32 half-ulp of it, so first rounding it
        // to f32 lands exactly on the tie and the tie-to-even picks the other neighbour. Folding a
        // `Neg` of such a constant must therefore not route the value through f32. Each case
        // returns the stored code (`bits`), so the exact grid point is checked, not an approximation.
        let cases = [
            // 1.12500001 is above the E5M2 midpoint 1.125 of 1.0 / 1.25 -> -1.25 = 0xBD. Through
            // f32 it is the tie 1.125 itself, which goes to the even -1.0 = 0xBC.
            (
                "fn main() -> i32 { let v: f8e5m2 = -1.12500001; return bits(v) as i32; }",
                0xBD,
            ),
            // 61439.999 is below the E5M2 tie 61440 between 57344 and infinity -> -57344 = 0xFB.
            // Through f32 it is 61440, which rounds to -inf = 0xFC.
            (
                "fn main() -> i32 { let v = -61439.999f8e5m2; return bits(v) as i32; }",
                0xFB,
            ),
            // 1.06250001 is above the E4M3FN midpoint 1.0625 of 1.0 / 1.125 -> -1.125 = 0xB9.
            // Through f32 it is the tie, which goes to the even -1.0 = 0xB8.
            (
                "fn main() -> i32 { let v: f8e4m3 = -1.06250001; return bits(v) as i32; }",
                0xB9,
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
            .filter(|i| {
                matches!(
                    i.op,
                    Op::Alloca {
                        ty: _,
                        space: wukong_mir::AddrSpace::Local,
                        ..
                    }
                )
            })
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
        // Array locals are addressed by gep, so Mem2Reg must NOT promote them. Check that pass
        // directly: the full pipeline may forward every load, then DSE/DCE legitimately remove
        // the unread array. Keeping a dead alloca is not part of Mem2Reg's safety contract.
        let src = "fn main() -> i32 { let mut xs: [i32; 3] = [1, 2, 3]; \
                   xs[1] = xs[0] + xs[2]; return xs[1]; }";
        let (mut prog, mut interner) = lower(src);
        for f in &mut prog.funcs {
            mem2reg::Mem2Reg.run_function(f, &mut CfgAnalyses::default());
        }
        assert!(
            count_allocas(&prog.funcs[0]) >= 1,
            "array slot must stay in memory"
        );
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 4);
        // Keep full-pipeline coverage too, with the same exact observable result. Here every
        // index/value is constant, so load forwarding makes the array entirely unobservable.
        optimize(&mut prog, 2);
        assert_eq!(count_allocas(&prog.funcs[0]), 0);
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 4);
        assert_eq!(run_main_opt(src, 0), 4);
    }

    #[test]
    fn sroa_keeps_a_register_blocked_accumulator_out_of_memory() {
        // The canonical hand-blocked reduction: four partial sums in a `[f32; 4]`, each updated at a
        // constant index every iteration. Every access is at a constant position, so at -O2 the slot
        // is split into four scalar slots that mem2reg promotes — no `[4 x f32]` slot survives
        // anywhere in the program, and the sums are carried in registers.
        //
        // x[k] = k, so acc[j] = sum over i in 0..4 of (4i + j) = 24 + 4j = [24, 28, 32, 36], and
        // 24 + 2*28 + 3*32 + 4*36 = 320.
        let src = "fn blocked(x: [f32; 16]) -> f32 { \
                     let mut acc: [f32; 4] = [0.0; 4]; let mut i: i64 = 0; \
                     while i < 4 { \
                       acc[0] = acc[0] + x[4 * i]; acc[1] = acc[1] + x[4 * i + 1]; \
                       acc[2] = acc[2] + x[4 * i + 2]; acc[3] = acc[3] + x[4 * i + 3]; \
                       i = i + 1; } \
                     return acc[0] + acc[1] * 2.0 + acc[2] * 3.0 + acc[3] * 4.0; } \
                   fn main() -> i32 { let mut x: [f32; 16] = [0.0; 16]; let mut k: i64 = 0; \
                     while k < 16 { x[k] = k as f32; k = k + 1; } \
                     return blocked(x) as i32; }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 320, "level {lvl}");
        }
        let blocks_of_four = |p: &wukong_mir::Program| {
            p.funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.insts)
                .filter(|i| {
                    matches!(
                        &i.op,
                        Op::Alloca { ty: wukong_mir::MirType::Array(e, 4), .. }
                            if **e == wukong_mir::MirType::F32
                    )
                })
                .count()
        };
        let (mut prog, _) = lower(src);
        assert_eq!(
            blocks_of_four(&prog),
            1,
            "the front end gives `acc` one slot"
        );
        optimize(&mut prog, 2);
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        assert_eq!(
            blocks_of_four(&prog),
            0,
            "the accumulator slot must be split and promoted at -O2"
        );
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
            !main
                .blocks
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
        // still compute correctly. Tail-recursion elimination turns the accumulating `fib(n-2)`
        // into a loop iteration (`tre`), so exactly the non-tail `fib(n-1)` call remains.
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
        assert_eq!(calls, 1, "recursive fib should keep its non-tail self-call");
    }

    #[test]
    fn side_effect_free_calls_are_numbered_deleted_and_hoisted() {
        // The bare pipeline, so no inliner hides the calls: `heavy`'s summary is what acts. Two
        // identical calls become one, a call whose result is never read disappears, and the
        // invariant call in the loop moves to its preheader.
        let src = "fn heavy(x: i64) -> i64 { return x * x + 7; } \
                   fn main() -> i32 { let a: i64 = 6; \
                     let s: i64 = heavy(a) + heavy(a); \
                     let unused: i64 = heavy(a + 1); \
                     let mut t: i64 = 0; let mut i: i64 = 0; \
                     while i < 10 { t = t + heavy(5); i = i + 1; } \
                     return (s + t) as i32; }";
        let (mut prog, mut interner) = lower(src);
        let main = interner.intern("main");
        let heavy = interner.intern("heavy");
        // heavy(6) * 2 + heavy(5) * 10 = 43 * 2 + 32 * 10.
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 406);
        PassManager::standard(2).run(&mut prog);
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main_fn = find_fn(&prog, &interner, "main");
        let in_loop = inline::loop_blocks(main_fn);
        let mut calls = 0;
        for (bi, b) in main_fn.blocks.iter().enumerate() {
            for inst in &b.insts {
                if matches!(inst.op, wukong_mir::Op::Call { func, .. } if func == heavy) {
                    calls += 1;
                    assert!(
                        !in_loop[bi],
                        "the invariant call stayed in the loop\n{}",
                        wukong_mir::print::print_function(main_fn, &interner)
                    );
                }
            }
        }
        assert_eq!(
            calls,
            2,
            "{}",
            wukong_mir::print::print_function(main_fn, &interner)
        );
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 406);
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 406, "level {lvl}");
        }
    }

    #[test]
    fn the_post_unroll_cleanup_judges_callees_by_their_unrolled_bodies() {
        // `tri` loops a constant five times, so before unrolling it might not return and the dead
        // `tri(a)` in `main` has to stay. Complete unrolling leaves `tri` straight-line; the
        // cleanup re-summarizes the program from the unrolled bodies, and `main` — changed by
        // unrolling too, so cleaned up as well — then drops the dead call. Driven by hand, without
        // the inliner, which would otherwise absorb `tri`.
        //
        // s runs x, 2x, 4x+1, 8x+4, 16x+11, 32x+26, so tri(4) = 154; x runs 1, 3, 10, 32, 99, 301.
        let src = "fn tri(x: i64) -> i64 { let mut s: i64 = x; let mut k: i64 = 0; \
                     while k < 5 { s = s * 2 + k; k = k + 1; } return s; } \
                   fn main() -> i32 { let a: i64 = 3; let unused: i64 = tri(a); \
                     let mut x: i64 = 1; let mut i: i64 = 0; \
                     while i < 5 { x = x * 3 + i; i = i + 1; } \
                     return (x + tri(a + 1)) as i32; }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 455, "level {lvl}");
        }
        let (mut prog, mut interner) = lower(src);
        let tri = interner.intern("tri");
        let main = interner.intern("main");
        let calls_to_tri = |p: &Program| {
            p.funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(i.op, Op::Call { func, .. } if func == tri))
                .count()
        };
        PassManager::standard(2).run(&mut prog);
        assert_eq!(
            calls_to_tri(&prog),
            2,
            "a callee with a loop may not return, so the dead call stays"
        );
        let changed = unroll_program(&mut prog);
        let index_of =
            |name: wukong_span::Symbol| prog.funcs.iter().position(|f| f.name == name).unwrap();
        assert!(
            changed.contains(&index_of(tri)) && changed.contains(&index_of(main)),
            "both loops must be unrolled completely: {changed:?}"
        );
        PassManager::unroll_cleanup().run_subset(&mut prog, &changed);
        assert!(wukong_mir::verify::verify_program(&prog).is_empty());
        assert_eq!(
            calls_to_tri(&prog),
            1,
            "{}",
            wukong_mir::print::print_program(&prog, &interner)
        );
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 455);
    }

    #[test]
    fn a_short_constant_loop_is_unrolled_completely_and_folds_to_its_value() {
        // `x = x * 3 + k` five times: a recurrence no vectorizer or recognizer takes. With a known
        // trip count of 5 the unroller replaces the loop by its iterations, the cleanup merges the
        // header into its preheader, and every copy folds: `main` returns a constant.
        let src = "fn main() -> i32 { let mut x: i32 = 1; let mut k: i32 = 0; \
                   while k < 5 { x = x * 3 + k; k = k + 1; } return x; }";
        for lvl in [0, 1, 2, 3] {
            assert_eq!(run_main_opt(src, lvl), 301, "level {lvl}");
        }
        let (mut prog, mut interner) = lower(src);
        optimize(&mut prog, 2);
        let main_fn = find_fn(&prog, &interner, "main");
        let shown = wukong_mir::print::print_function(main_fn, &interner);
        assert!(
            !inline::loop_blocks(main_fn).iter().any(|&b| b),
            "the loop must be gone\n{shown}"
        );
        let returns_301 = main_fn.blocks.iter().any(|b| match b.term {
            Terminator::Ret(Some(v)) => b.insts.iter().any(|i| {
                i.result == Some(v) && matches!(i.op, Op::ConstInt(301, wukong_mir::MirType::I32))
            }),
            _ => false,
        });
        assert!(returns_301, "{shown}");
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 301);
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
                Op::Alloca {
                    ty: MirType::Ptr,
                    space: wukong_mir::AddrSpace::Local,
                    ..
                } => slots += 1,
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
        // `mem2reg_promotes_a_late_initialized_pointer_slot_to_null_on_the_uninit_path`.
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
    fn mem2reg_promotes_a_late_initialized_pointer_slot_to_null_on_the_uninit_path() {
        // A pointer slot first written *outside* the entry block can be read before it is written
        // on some path, so promoting it needs a value for "not yet written". That value used to be
        // unsound: `inttoptr 0` is null natively but was the interpreter's first live allocation,
        // so this slot had to stay in memory and be re-loaded on every use. The interpreter now
        // reserves slot 0 as null (`wukong_interp::NULL_SLOT`), so the uninitialized path gets the
        // same null on both backends and the slot is promoted by mem2reg itself. Here `p`'s store
        // is inside the loop body, so the loop header merges the null with the body's `&mut a`.
        let src = "fn main() -> i32 { let mut a: i32 = 5; let mut t: i32 = 0; \
                   let mut i: i32 = 0; \
                   while i < 3 { let p: *mut i32 = &mut a; t = t + *p; i = i + 1; } \
                   return t; }";
        let (mut prog, mut interner) = lower(src);
        for f in &mut prog.funcs {
            mem2reg::Mem2Reg.run_function(f, &mut CfgAnalyses::default());
        }
        let (slots, _) = count_ptr_slots_and_reloads(find_fn(&prog, &interner, "main"));
        assert_eq!(
            slots, 0,
            "a late-initialized pointer slot must be promoted, with null on the unwritten path"
        );
        let has_null = find_fn(&prog, &interner, "main")
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| {
                matches!(
                    i.op,
                    wukong_mir::Op::Cast(
                        wukong_mir::CastKind::IntToPtr,
                        _,
                        wukong_mir::MirType::Ptr
                    )
                )
            });
        assert!(has_null, "the read-before-write path must see `inttoptr 0`");
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        let main = interner.intern("main");
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 15);
        optimize(&mut prog, 2);
        assert_eq!(
            count_ptr_slots_and_reloads(find_fn(&prog, &interner, "main")),
            (0, 0),
            "no pointer slot or reload may survive the full pipeline"
        );
        for f in &prog.funcs {
            assert!(wukong_mir::verify::verify_function(f).is_empty());
        }
        assert_eq!(wukong_interp::run(&prog, main, &interner).unwrap(), 15);
        assert_eq!(run_main_opt(src, 0), 15);
    }

    #[test]
    fn pointer_conditions_are_null_tests_at_every_level() {
        // `if p` on a pointer lowers to `ptrtoint p != 0`. The reproducer that forced the revert
        // at 0900352 is the first case: the first local's address used to be interpreter slot 0
        // and so read as null. A null from `0 as *T`, a conditionally assigned pointer (promoted
        // by mem2reg with a null on the unassigned path) and `while p` are covered too.
        // Expected values are the obvious scalar semantics: the address of a live local is
        // non-null, `0 as *T` is null.
        let cases = [
            (
                "fn main() -> i32 { let mut v: i32 = 7; let p: *mut i32 = &mut v; \
              if p { return 1; } return 0; }",
                1,
            ),
            (
                "fn main() -> i32 { let q: *mut i32 = 0 as *mut i32; \
              if q { return 1; } return 2; }",
                2,
            ),
            (
                "fn main() -> i32 { let mut v: i32 = 3; let mut n: i32 = 0; \
              let mut p: *mut i32 = 0 as *mut i32; \
              while n < 4 { if n == 2 { p = &mut v; } n = n + 1; } \
              if p { return *p * 10; } return 0; }",
                30,
            ),
            (
                "fn main() -> i32 { let mut v: i32 = 3; let mut p: *mut i32 = &mut v; \
              let mut k: i32 = 0; \
              while p { k = k + 1; if k == 5 { p = 0 as *mut i32; } } return k; }",
                5,
            ),
        ];
        for (src, expect) in cases {
            for lvl in [0, 1, 2, 3] {
                assert_eq!(run_main_opt(src, lvl), expect, "level {lvl}: {src}");
            }
            let (mut prog, _) = lower(src);
            for f in &prog.funcs {
                assert!(
                    wukong_mir::verify::verify_function(f).is_empty(),
                    "-O0 MIR must verify: {src}"
                );
            }
            optimize(&mut prog, 2);
            for f in &prog.funcs {
                assert!(wukong_mir::verify::verify_function(f).is_empty(), "{src}");
            }
        }
    }
}
