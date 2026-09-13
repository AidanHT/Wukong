//! `compile-work` — a **counting** instrument for compile-time work, clock-invariant by construction.
//!
//! Every other mode of this harness reports wall time, and on this host wall time is the wrong
//! instrument for compile time: the 2026-09 evaluation watched one revision's optimizer figure move
//! from 771.62 ms to 222.62 ms on identical inputs purely from thermal history, and then watched two
//! best-of-N harnesses disagree on the same 392-file corpus (-5.98% against 28-34%) because each had
//! sampled a different clock state. Compile time is deterministic work — the same source produces
//! the same allocations, the same pass invocations, the same fixpoint sweeps — so it can be
//! **counted**, and a count cannot be perturbed by a throttling laptop or a neighbour's build.
//!
//! What is counted, per stage, per file:
//!
//! * **heap allocations** — `alloc` calls, `realloc` calls, and bytes requested — through a
//!   counting [`GlobalAlloc`] installed for this binary. The counters are thread-local `Cell`s
//!   (no atomics), so the cost to every other mode is a handful of instructions per allocation,
//!   and no other thread's allocations can leak into a count. The optimizer is documented as
//!   HashMap-bound and the front end's 2026-09 commits are almost all titled "remove ... allocations",
//!   so allocations are the quantity those changes ask to be judged by.
//! * **optimizer pass invocations** — how many times each pass's `run_function` ran, summed over
//!   the program, which is the fixpoint's real work count (clean-tracking skips passes that cannot
//!   change anything, so a pass that converges in fewer sweeps shows up here directly).
//! * **per-pass allocation attribution** — each standard pass is wrapped in a transparent counting
//!   adapter and fed to the production [`wukong_opt::PassManager`], so the driver, the clean-tracking
//!   and the mutation sequence are the real ones. The attributed result must render byte-identical
//!   to [`wukong_opt::optimize`]'s; a divergence is `*** FAILED` and the process exits 1.
//!
//! Every stage is counted **three times** per file (`WUKONG_WORK_RUNS` raises it). The steady-state
//! figure is the *minimum* over runs 2..n: a count is a floor, a pure function of the input, and
//! anything above it is state carried between calls. Run 1's excess over that floor is first-call
//! initialisation (lazily built tables) and is printed once, corpus-wide. Where runs 2..n do **not**
//! agree, the file and stage are disclosed with every run's count and the disagreement is totalled
//! in the header — that disagreement is a measured property of the compiler (something stateful
//! across calls), reported rather than hidden, and it is the instrument's resolution limit.
//!
//! The stage decomposition (lex, parse, sema, mir_build, optimize, codegen+obj) and the artifact
//! handling (each stage counted on a fixed upstream artifact, rebuilt untimed where the stage mutates
//! its input) follow `compile-profile` exactly, so the two tables describe the same six quantities:
//! one in nanoseconds, one in allocations.
//!
//! Every file's row is printed, sorted by name, so two runs of this mode — two revisions, or one
//! revision before and after a change — can be `diff`ed to localise where the work moved.
//!
//! ```text
//! cargo run -p wukong_bench --release -- compile-work tests/run examples tests/run/lib bench/kernels
//! ```
//!
//! Counts are identical in debug and release builds *except* that a debug `wukong_opt` runs the MIR
//! verifier after every pass (`cfg(debug_assertions)`), which allocates; compare like with like.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use wukong_span::{Interner, SourceId};

/// Optimization level the counts describe. `-O2` is the heaviest pipeline (`-O3` ≡ `-O2`) and the
/// level `compile-profile` measures, so the two tables line up.
const LEVEL: u8 = 2;

// ===================================================================================================
// the counting allocator
// ===================================================================================================

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static REALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

/// A pass-through allocator that counts. Installed as this binary's `#[global_allocator]`; the
/// counters are per-thread and cost three `Cell` updates per call. `dealloc` is not counted (every
/// allocation is freed exactly once, so it carries no information the `alloc` count does not).
pub struct CountingAlloc;

#[inline]
fn bump(bytes: usize, realloc: bool) {
    // `try_with` never fails for a `const`-initialised `Cell<u64>` (no destructor to race), but an
    // allocator must not be able to panic, so the fallible form is used and the error ignored.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + u64::from(!realloc)));
    let _ = REALLOCS.try_with(|c| c.set(c.get() + u64::from(realloc)));
    let _ = BYTES.try_with(|c| c.set(c.get() + bytes as u64));
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout.size(), false);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump(layout.size(), false);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump(new_size, true);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// A snapshot of the three counters, or (by subtraction) the work between two snapshots.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Work {
    pub allocs: u64,
    pub reallocs: u64,
    pub bytes: u64,
}

impl Work {
    fn now() -> Work {
        Work {
            allocs: ALLOCS.with(Cell::get),
            reallocs: REALLOCS.with(Cell::get),
            bytes: BYTES.with(Cell::get),
        }
    }
    fn since(self, earlier: Work) -> Work {
        Work {
            allocs: self.allocs.saturating_sub(earlier.allocs),
            reallocs: self.reallocs.saturating_sub(earlier.reallocs),
            bytes: self.bytes.saturating_sub(earlier.bytes),
        }
    }
    fn add(&mut self, w: Work) {
        self.allocs += w.allocs;
        self.reallocs += w.reallocs;
        self.bytes += w.bytes;
    }
}

/// Count the allocations `f` performs. The result is returned to the caller so that its
/// destruction happens *outside* the counted window (frees are not counted anyway, but a drop can
/// allocate — e.g. a `Drop` that logs — and it is not the stage's construction cost).
fn counted<R>(f: impl FnOnce() -> R) -> (R, Work) {
    let before = Work::now();
    let r = f();
    (r, Work::now().since(before))
}

// ===================================================================================================
// per-pass attribution
// ===================================================================================================

/// One optimizer pass's accumulated counts over a whole attributed run.
#[derive(Clone, Default)]
struct PassWork {
    name: &'static str,
    calls: u64,
    work: Work,
}

thread_local! {
    /// Accumulators for the counting adapters, indexed by adapter slot. Pre-sized before an
    /// attributed run so that the adapters never allocate inside a counted window.
    static PASS_ACC: RefCell<Vec<PassWork>> = const { RefCell::new(Vec::new()) };
}

/// A transparent counting adapter around a real pass. Forwards `name` and `run_function`
/// unchanged, so the production `PassManager` sees exactly the pass it would have run; the only
/// addition is a counter snapshot on either side of the call.
struct Counted {
    inner: Box<dyn wukong_opt::Pass>,
    slot: usize,
}

impl wukong_opt::Pass for Counted {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn run_function(
        &self,
        f: &mut wukong_mir::Function,
        cache: &mut wukong_opt::CfgAnalyses,
    ) -> bool {
        let before = Work::now();
        let changed = self.inner.run_function(f, cache);
        let w = Work::now().since(before);
        PASS_ACC.with(|acc| {
            let mut acc = acc.borrow_mut();
            let s = &mut acc[self.slot];
            s.name = self.inner.name();
            s.calls += 1;
            s.work.add(w);
        });
        changed
    }
}

/// Slot layout of the attribution table: the ten standard `-O2` passes in pipeline order, then the
/// two whole-program stages, then the cleanup pipeline's two passes (kept separate because they run
/// under a different driver instance and belong to the unrolling stage's cost).
const SLOT_INLINE: usize = 10;
const SLOT_UNROLL: usize = 11;
const SLOT_CLEANUP_CSE: usize = 12;
const SLOT_CLEANUP_DCE: usize = 13;
const SLOTS: usize = 14;

fn counted_pass(slot: usize, p: Box<dyn wukong_opt::Pass>) -> Box<dyn wukong_opt::Pass> {
    Box::new(Counted { inner: p, slot })
}

/// The standard `-O2` pipeline with every pass wrapped, in `PassManager::standard(2)`'s order.
fn attributed_standard() -> wukong_opt::PassManager {
    use wukong_opt::*;
    let mut pm = PassManager::new();
    pm.add(counted_pass(0, Box::new(Mem2Reg)));
    pm.add(counted_pass(1, Box::new(Simplify)));
    pm.add(counted_pass(2, Box::new(SimplifyCfg)));
    pm.add(counted_pass(3, Box::new(SimplifyPhis)));
    pm.add(counted_pass(4, Box::new(Dce)));
    pm.add(counted_pass(5, Box::new(Cse)));
    pm.add(counted_pass(6, Box::new(Dse)));
    pm.add(counted_pass(7, Box::new(LoopCanon)));
    pm.add(counted_pass(8, Box::new(Licm)));
    pm.add(counted_pass(9, Box::new(Vectorize::default())));
    pm
}

/// The post-unroll cleanup pipeline (`cse` → `dce`), wrapped.
fn attributed_cleanup() -> wukong_opt::PassManager {
    use wukong_opt::*;
    let mut pm = PassManager::new();
    pm.add(counted_pass(SLOT_CLEANUP_CSE, Box::new(Cse)));
    pm.add(counted_pass(SLOT_CLEANUP_DCE, Box::new(Dce)));
    pm
}

/// Run the `-O2` optimizer over `p` with per-pass attribution, mirroring [`wukong_opt::optimize`]
/// stage for stage: inline → standard fixpoint → unroll → cleanup of the changed functions.
///
/// The cleanup is the one place `optimize` does something this replica cannot call directly (it
/// runs the private `run_function` on just the changed functions). Each changed function is moved
/// into a single-function `Program` shell — whose `Vec` is reserved *before* counting starts, so the
/// move itself allocates nothing — and the public `PassManager::run` is applied to that shell. The
/// per-function pipeline is function-local, so the mutation sequence is identical; the
/// byte-identical check in `measure_one` is what proves it.
fn optimize_attributed(p: &mut wukong_mir::Program) -> Vec<PassWork> {
    PASS_ACC.with(|acc| {
        let mut acc = acc.borrow_mut();
        acc.clear();
        acc.resize(SLOTS, PassWork::default());
    });
    let (_, inline) = counted(|| wukong_opt::inline_program(p));
    attributed_standard().run(p);
    let cleanup = attributed_cleanup();
    let mut shell = wukong_mir::Program {
        funcs: Vec::with_capacity(1),
        statics: Vec::new(),
        level: p.level,
    };
    let (changed, unroll) = counted(|| wukong_opt::unroll_program(p));
    for i in changed {
        let placeholder = placeholder_function(&p.funcs[i]);
        let f = std::mem::replace(&mut p.funcs[i], placeholder);
        shell.funcs.push(f);
        cleanup.run(&mut shell);
        p.funcs[i] = shell.funcs.pop().unwrap();
    }
    PASS_ACC.with(|acc| {
        let mut acc = acc.borrow_mut();
        acc[SLOT_INLINE] = PassWork {
            name: "inline",
            calls: 1,
            work: inline,
        };
        acc[SLOT_UNROLL] = PassWork {
            name: "unroll",
            calls: 1,
            work: unroll,
        };
        acc[SLOT_CLEANUP_CSE].name = "cleanup:cse";
        acc[SLOT_CLEANUP_DCE].name = "cleanup:dce";
        acc.clone()
    })
}

/// An allocation-free stand-in left in a function's slot while it is being cleaned up in the shell.
/// Shares the original's `Symbol`/`MirType` (both `Copy`/cheap) and owns no heap memory.
fn placeholder_function(f: &wukong_mir::Function) -> wukong_mir::Function {
    wukong_mir::Function {
        name: f.name,
        params: Vec::new(),
        ret: wukong_mir::MirType::Void,
        blocks: Vec::new(),
        value_types: Vec::new(),
        entry: f.entry,
        vec_kernels: Vec::new(),
    }
}

// ===================================================================================================
// the report
// ===================================================================================================

/// The six pipeline stages, in order, matching `compile-profile`.
const STAGES: [&str; 6] = [
    "lex",
    "parse",
    "sema",
    "mir_build",
    "optimize",
    "codegen+obj",
];

struct FileWork {
    name: String,
    src_bytes: usize,
    tokens: usize,
    ast_nodes: u32,
    funcs: usize,
    mir_ops0: usize,
    mir_ops2: usize,
    obj_bytes: usize,
    /// Steady-state counts (minimum over runs 2..n), indexed by `STAGES` order.
    stage: [Work; 6],
    /// Run 1 minus steady, per stage: first-call initialisation attributable to this file.
    init: [Work; 6],
    /// Max minus min over runs 2..n, per stage: zero unless the stage is not a pure function.
    jitter: [Work; 6],
    /// `(stage index, allocs on every run)` for each stage whose steady-state runs disagreed.
    disclosures: Vec<(usize, Vec<u64>)>,
    /// Per-pass attribution of the optimize stage.
    passes: Vec<PassWork>,
    /// Largest per-function fixpoint iteration count (from `optimize_timed`).
    max_iterations: u32,
}

enum Meas {
    Ran(Box<FileWork>),
    Skipped,
    Failed(String),
}

pub fn report(files: &[PathBuf]) {
    let mut rows: Vec<FileWork> = Vec::new();
    let mut skipped = 0u32;
    let mut failures: Vec<String> = Vec::new();
    for path in files {
        match measure_one(path) {
            Meas::Ran(w) => rows.push(*w),
            Meas::Skipped => skipped += 1,
            Meas::Failed(why) => {
                println!("{:<26} *** FAILED: {why} ***", short(path));
                failures.push(format!("{}: {why}", short(path)));
            }
        }
    }

    if rows.is_empty() {
        println!("compile-work: no corpus file compiled to object (measured 0, skipped {skipped})");
        if !failures.is_empty() {
            std::process::exit(1);
        }
        return;
    }

    // --- stage totals ---
    let mut stage_tot = [Work::default(); 6];
    let mut init_tot = [Work::default(); 6];
    for r in &rows {
        for (tot, w) in stage_tot.iter_mut().zip(&r.stage) {
            tot.add(*w);
        }
        for (tot, w) in init_tot.iter_mut().zip(&r.init) {
            tot.add(*w);
        }
    }
    let mut grand = Work::default();
    for s in &stage_tot {
        grand.add(*s);
    }

    let src_bytes: usize = rows.iter().map(|r| r.src_bytes).sum();
    let tokens: usize = rows.iter().map(|r| r.tokens).sum();
    let ast_nodes: u64 = rows.iter().map(|r| u64::from(r.ast_nodes)).sum();
    let funcs: usize = rows.iter().map(|r| r.funcs).sum();
    let mir_ops0: usize = rows.iter().map(|r| r.mir_ops0).sum();
    let mir_ops2: usize = rows.iter().map(|r| r.mir_ops2).sum();
    let obj_bytes: usize = rows.iter().map(|r| r.obj_bytes).sum();

    let n_runs = runs();
    let mut jitter_tot = [Work::default(); 6];
    let mut jitter_files = 0usize;
    for r in &rows {
        for (tot, w) in jitter_tot.iter_mut().zip(&r.jitter) {
            tot.add(*w);
        }
        if !r.disclosures.is_empty() {
            jitter_files += 1;
        }
    }
    let jitter_allocs: u64 = jitter_tot.iter().map(|w| w.allocs).sum();
    println!(
        "compile-work: deterministic work counts  [regime: COUNTS, clock-invariant; -O{LEVEL}; \
         each stage counted {n_runs}x per file, steady state = minimum over runs 2..{n_runs}]\n\
         corpus: {} file(s) compiled to object, {skipped} skipped; steady-state runs disagreed in \
         {jitter_files} file(s) by {jitter_allocs} alloc(s) in total (disclosed below).\n",
        rows.len()
    );
    println!(
        "{:<14} {:>12} {:>10} {:>14} {:>7}   allocs per unit of work",
        "stage", "allocs", "reallocs", "bytes", "%allocs"
    );
    println!("{}", "-".repeat(92));
    for (i, name) in STAGES.iter().enumerate() {
        let w = stage_tot[i];
        let unit = match i {
            0 => per(w.allocs, tokens as u64, "alloc/token"),
            1 => per(w.allocs, ast_nodes, "alloc/AST node"),
            2 => per(w.allocs, ast_nodes, "alloc/AST node"),
            3 => per(w.allocs, mir_ops0 as u64, "alloc/MIR op"),
            4 => per(w.allocs, mir_ops0 as u64, "alloc/MIR op in"),
            _ => per(w.allocs, mir_ops2 as u64, "alloc/MIR op out"),
        };
        println!(
            "{:<14} {:>12} {:>10} {:>14} {:>6.1}%   {unit}",
            name,
            w.allocs,
            w.reallocs,
            w.bytes,
            100.0 * w.allocs as f64 / grand.allocs.max(1) as f64,
        );
    }
    println!("{}", "-".repeat(92));
    println!(
        "{:<14} {:>12} {:>10} {:>14} {:>6.1}%",
        "TOTAL", grand.allocs, grand.reallocs, grand.bytes, 100.0
    );
    let init_allocs: u64 = init_tot.iter().map(|w| w.allocs).sum();
    let init_bytes: u64 = init_tot.iter().map(|w| w.bytes).sum();
    println!(
        "first-call initialisation (run 1 minus run 2, summed over the corpus, NOT in the table): \
         {init_allocs} allocs, {init_bytes} bytes"
    );
    if init_allocs > 0 {
        let parts: Vec<String> = STAGES
            .iter()
            .zip(&init_tot)
            .filter(|(_, w)| w.allocs > 0)
            .map(|(n, w)| format!("{n} {}", w.allocs))
            .collect();
        println!("  by stage: {}", parts.join(", "));
    }
    if jitter_files > 0 {
        println!(
            "NONDETERMINISTIC: {jitter_files} file(s) whose steady-state runs did not agree — the \
             count is not a pure function of the input there. Per stage, max-min over runs \
             2..{n_runs} summed over the corpus:"
        );
        for (name, w) in STAGES.iter().zip(&jitter_tot) {
            if w.allocs > 0 || w.bytes > 0 {
                println!("  {name}: {} allocs, {} bytes", w.allocs, w.bytes);
            }
        }
        for r in &rows {
            for (i, per_run) in &r.disclosures {
                println!("  {} {}: allocs per run {:?}", r.name, STAGES[*i], per_run);
            }
        }
    }

    // --- optimizer attribution ---
    let mut agg: Vec<PassWork> = vec![PassWork::default(); SLOTS];
    for r in &rows {
        for (a, p) in agg.iter_mut().zip(&r.passes) {
            a.name = p.name;
            a.calls += p.calls;
            a.work.add(p.work);
        }
    }
    let attributed: Work = agg.iter().fold(Work::default(), |mut acc, p| {
        acc.add(p.work);
        acc
    });
    let opt = stage_tot[4];
    let residual = opt.since(attributed);
    let max_iter = rows.iter().map(|r| r.max_iterations).max().unwrap_or(0);
    let sum_calls: u64 = agg[..10].iter().map(|p| p.calls).sum();
    println!(
        "\noptimizer attribution (summed over corpus; counting adapters around the production \
         PassManager, byte-identical to `optimize`):\n  {:<14} {:>9} {:>12} {:>14} {:>8}",
        "pass", "calls", "allocs", "bytes", "%opt"
    );
    let mut order: Vec<&PassWork> = agg.iter().collect();
    order.sort_by_key(|p| std::cmp::Reverse(p.work.allocs));
    for p in order {
        println!(
            "  {:<14} {:>9} {:>12} {:>14} {:>7.1}%",
            p.name,
            p.calls,
            p.work.allocs,
            p.work.bytes,
            100.0 * p.work.allocs as f64 / opt.allocs.max(1) as f64,
        );
    }
    println!(
        "  {:<14} {:>9} {:>12} {:>14} {:>7.1}%   (fixpoint driver: one `clean` vector per function)",
        "driver",
        "-",
        residual.allocs,
        residual.bytes,
        100.0 * residual.allocs as f64 / opt.allocs.max(1) as f64,
    );
    println!(
        "  standard-pipeline pass invocations: {sum_calls} over {funcs} functions \
         ({:.2} per function); largest per-function fixpoint sweep count: {max_iter}",
        sum_calls as f64 / funcs.max(1) as f64
    );

    println!(
        "\nwork totals: {src_bytes} src bytes, {tokens} tokens, {ast_nodes} AST nodes, {funcs} \
         functions, {mir_ops0} MIR ops in -> {mir_ops2} out, {obj_bytes} object bytes."
    );

    // --- per-file rows, sorted by name, diffable ---
    println!(
        "\nper-file steady-state allocs by stage (sorted by name; diff two runs to localise a change):"
    );
    println!(
        "{:<28} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>10} {:>7} {:>7}",
        "file", "lex", "parse", "sema", "mir", "opt", "cg+obj", "calls", "bytes", "ops0", "ops2"
    );
    println!("{}", "-".repeat(120));
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    for r in &rows {
        let bytes: u64 = r.stage.iter().map(|w| w.bytes).sum();
        let calls: u64 = r.passes[..10].iter().map(|p| p.calls).sum();
        println!(
            "{:<28} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>10} {:>7} {:>7}",
            trunc(&r.name, 28),
            r.stage[0].allocs,
            r.stage[1].allocs,
            r.stage[2].allocs,
            r.stage[3].allocs,
            r.stage[4].allocs,
            r.stage[5].allocs,
            calls,
            bytes,
            r.mir_ops0,
            r.mir_ops2,
        );
    }

    if !failures.is_empty() {
        eprintln!("\n{} program(s) failed a self-check:", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        std::process::exit(1);
    }
}

/// How many times each stage is counted per file: `WUKONG_WORK_RUNS` (minimum 2, default 3). Raise
/// it to tell a count that *drifts* with every call (a cache warming) from one that *jitters*.
fn runs() -> usize {
    std::env::var("WUKONG_WORK_RUNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(2)
}

/// One stage's counts over `runs()` repetitions on a fixed input.
struct Repeated {
    /// The steady-state figure: the **minimum** over runs 2..n. A count is a floor — a pure function
    /// of the input allocates the same amount every time, and anything above that floor on some run
    /// is state carried between calls, which is reported separately rather than mixed in.
    steady: Work,
    /// Run 1's excess over `steady`: first-call initialisation.
    init: Work,
    /// Max minus min over runs 2..n: how far the count moved between steady-state runs. Zero for a
    /// deterministic stage. Nonzero is disclosed per file and totalled over the corpus.
    jitter: Work,
    /// Every run's allocation count, for the disclosure line.
    allocs_per_run: Vec<u64>,
}

fn repeat(mut run: impl FnMut() -> Work) -> Repeated {
    let n = runs();
    let all: Vec<Work> = (0..n).map(|_| run()).collect();
    let steady = all[1..].iter().copied().reduce(min_work).unwrap();
    let peak = all[1..].iter().copied().reduce(max_work).unwrap();
    Repeated {
        steady,
        init: all[0].since(steady),
        jitter: peak.since(steady),
        allocs_per_run: all.iter().map(|w| w.allocs).collect(),
    }
}

fn min_work(a: Work, b: Work) -> Work {
    Work {
        allocs: a.allocs.min(b.allocs),
        reallocs: a.reallocs.min(b.reallocs),
        bytes: a.bytes.min(b.bytes),
    }
}

fn max_work(a: Work, b: Work) -> Work {
    Work {
        allocs: a.allocs.max(b.allocs),
        reallocs: a.reallocs.max(b.reallocs),
        bytes: a.bytes.max(b.bytes),
    }
}

fn measure_one(path: &Path) -> Meas {
    let Ok(src) = std::fs::read_to_string(path) else {
        return Meas::Skipped;
    };

    // One clean pass: validate runnability and capture the artifacts each stage is counted on.
    let (tokens, ld) = wukong_lexer::tokenize(&src, SourceId(0));
    if ld.iter().any(|d| d.is_error()) {
        return Meas::Skipped;
    }
    let mut interner = Interner::new();
    let (module, pd, ast_nodes) =
        wukong_parser::parse_module_tokens_from(&tokens, &src, &mut interner, 0);
    if pd.iter().any(|d| d.is_error()) {
        return Meas::Skipped;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Meas::Skipped;
    }
    let (p0, mld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if mld.iter().any(|d| d.is_error()) {
        return Meas::Skipped;
    }
    let mut p_opt = p0.clone();
    wukong_opt::optimize(&mut p_opt, LEVEL);
    let Ok(obj) = wukong_codegen_cranelift::emit_object(&p_opt, &interner) else {
        return Meas::Skipped;
    };
    let obj_bytes = obj.len();
    drop(obj);

    // Byte-identical self-check for the attribution replica, then its counts.
    let mut attributed = p0.clone();
    let passes = optimize_attributed(&mut attributed);
    if wukong_mir::print::print_program(&attributed, &interner)
        != wukong_mir::print::print_program(&p_opt, &interner)
    {
        return Meas::Failed("attribution replica diverged from optimize".into());
    }
    let mut timed = p0.clone();
    let max_iterations = wukong_opt::optimize_timed(&mut timed, LEVEL).max_iterations;

    let mut stage = [Work::default(); 6];
    let mut init = [Work::default(); 6];
    let mut jitter = [Work::default(); 6];
    let mut disclosures: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut set = |i: usize, r: Repeated| {
        stage[i] = r.steady;
        init[i] = r.init;
        jitter[i] = r.jitter;
        if r.jitter != Work::default() {
            disclosures.push((i, r.allocs_per_run));
        }
    };

    // Lex: input is the source text (fixed).
    set(
        0,
        repeat(|| {
            let (out, w) = counted(|| wukong_lexer::tokenize(&src, SourceId(0)));
            drop(out);
            w
        }),
    );
    // Parse: fixed token stream, a fresh (empty, unallocated) interner each run.
    set(
        1,
        repeat(|| {
            let mut it = Interner::new();
            let (out, w) =
                counted(|| wukong_parser::parse_module_tokens_from(&tokens, &src, &mut it, 0));
            drop(out);
            w
        }),
    );
    // Sema: reads the fixed module + interner, mutates neither.
    set(
        2,
        repeat(|| {
            let (out, w) = counted(|| wukong_sema::check(&module, &interner));
            drop(out);
            w
        }),
    );
    // MIR build: needs a pristine interner; rebuild module + sema untimed, count lowering only.
    set(
        3,
        repeat(|| {
            let mut it = Interner::new();
            let (m, _, _) = wukong_parser::parse_module_tokens_from(&tokens, &src, &mut it, 0);
            let (s, _) = wukong_sema::check(&m, &it);
            let (out, w) = counted(|| wukong_mir_build::lower_program(&m, &s, &mut it));
            drop(out);
            w
        }),
    );
    // Optimize: clone the unoptimized program (uncounted), count `optimize` alone.
    set(
        4,
        repeat(|| {
            let mut p = p0.clone();
            let (_, w) = counted(|| wukong_opt::optimize(&mut p, LEVEL));
            w
        }),
    );
    // Backend: Cranelift codegen + object serialization on the fixed optimized program.
    set(
        5,
        repeat(|| {
            let (out, w) = counted(|| wukong_codegen_cranelift::emit_object(&p_opt, &interner));
            drop(out);
            w
        }),
    );

    Meas::Ran(Box::new(FileWork {
        name: short(path),
        src_bytes: src.len(),
        tokens: tokens.len(),
        ast_nodes,
        funcs: p0.funcs.len(),
        mir_ops0: count_ops(&p0),
        mir_ops2: count_ops(&p_opt),
        obj_bytes,
        stage,
        init,
        jitter,
        disclosures,
        passes,
        max_iterations,
    }))
}

/// Total MIR operations: instructions plus one terminator per block (as `compile-profile`).
fn count_ops(p: &wukong_mir::Program) -> usize {
    p.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len() + 1)
        .sum()
}

fn per(count: u64, unit: u64, name: &str) -> String {
    if unit == 0 {
        return "-".into();
    }
    format!("{:.3} {name}", count as f64 / unit as f64)
}

fn short(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}

/// Truncate to at most `w` characters (char-boundary-safe).
fn trunc(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else {
        let cut: String = s.chars().take(w.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}
