//! Whole-pipeline stage profiling and in-process-vs-spawn characterization.
//!
//! Four modes live here. The first two are aimed at the "compile-time floor" question — how close is
//! code→object to the irreducible work, and what does the process-spawn path cost on top of the
//! in-process API. The last two (P6.10) time whole processes: a cold start, and a thread sweep.
//!
//! * `compile-profile` — a per-STAGE wall-time breakdown of the full pipeline (lex, parse, sema,
//!   mir_build, optimize, and the Cranelift codegen+object-emit backend) measured in-process,
//!   best-of-N per stage per file, over a corpus. `compile-time` mode already dissects the optimizer
//!   pass-by-pass; this frames the optimizer against every *other* stage so the whole shape of the
//!   compile is visible and the floor of each stage is exposed. Stage boundaries are timed at the
//!   bench level by calling the stage crates directly (`wukong_lexer` … `wukong_codegen_cranelift`),
//!   exactly as `compile-time` calls `wukong_opt` — no driver hook, so the driver stays a thin
//!   sequencer with no instrumentation woven through it.
//!
//!   The backend appears as ONE stage (`codegen+obj`) in the stage table — Cranelift
//!   instruction-selection + register allocation + machine-code emission, then object-container
//!   serialization — with its internal split (codegen vs object-write) measured via
//!   `wukong_codegen_cranelift::emit_object_timed` and printed underneath, bounding the fixed
//!   container cost `docs/compile-floor.md` reasons about. A JIT row beside the table times the
//!   `--run --backend=native` codegen of the same optimized programs (files with a `main`), as a
//!   ratio to their object-path backend stage.
//!
//! * `spawn-overhead` — the same source compiled two ways: (a) the in-process API to an object in
//!   memory, and (b) spawning the real `wukongc.exe --emit=obj`. The difference is the spawn tax
//!   (process creation + runtime init + file I/O + the driver's own front-matter). Reported both
//!   first-call and warm steady-state (best-of-N min), plus the `--emit=exe` path broken into
//!   compile vs the rustc-driven link. Only the FIRST row of the first-call table is a genuine cold
//!   start — after it the compiler image and the OS page cache are warm — and every spawn's exit
//!   status is checked, because a child that fails emits no object and its wall time is not a
//!   compile measurement.
//!
//! * `cold-start` - fresh-process compile plus first run of a fixed program (`COLD_START_SRC`, or
//!   a `.wk` path given on the command line): `--runs=N` children of `wukongc --run -O2`, each a
//!   cold start of compile + JIT + runtime initialization + the first execution, reported as
//!   min / median / p95 / max and the very first child on its own. Every child's exit status and
//!   stdout are checked.
//!
//! * `scaling` - one workload (a recognized `@parallel` 512^3 GEMM repeated `--reps` times, or a
//!   `.wk` path) in fresh processes at `WUKONG_NUM_THREADS` = 1, 2, 4, .., all (`--max-threads`),
//!   reporting per thread count the min / median wall, the speedup over one thread and the
//!   per-thread efficiency. The answer must be identical at every thread count.
//!
//! ```text
//! cargo run -p wukong_bench --release -- compile-profile tests/run examples bench/kernels
//! cargo run -p wukong_bench --release -- spawn-overhead   tests/run examples bench/kernels
//! cargo run -p wukong_bench --release -- cold-start --runs=20
//! cargo run -p wukong_bench --release -- scaling --runs=3
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use wukong_span::{Interner, SourceId};

/// Optimization level the profile measures. `-O2` is the heaviest pipeline (`-O3` ≡ `-O2` here) and
/// the level `compile-vs` compares against, so code→object numbers line up across the harness.
const LEVEL: u8 = 2;

/// Per-stage wall budget for the best-of-N minimum, and a hard rep floor/ceiling. Small enough to
/// keep a full-corpus run brisk, big enough that the minimum settles.
const BUDGET: Duration = Duration::from_millis(40);
const MIN_REPS: u32 = 5;
const MAX_REPS: u32 = 3000;

// ===================================================================================================
// compile-profile
// ===================================================================================================

/// The six pipeline stages, in order. `Backend` is Cranelift codegen + object serialization as one
/// stage; its internal split is measured separately via `emit_object_timed` and printed under the
/// table rather than as two stage rows (see the module docs).
#[derive(Clone, Copy)]
enum Stage {
    Lex,
    Parse,
    Sema,
    MirBuild,
    Optimize,
    Backend,
}

impl Stage {
    const ALL: [Stage; 6] = [
        Stage::Lex,
        Stage::Parse,
        Stage::Sema,
        Stage::MirBuild,
        Stage::Optimize,
        Stage::Backend,
    ];
    fn name(self) -> &'static str {
        match self {
            Stage::Lex => "lex",
            Stage::Parse => "parse",
            Stage::Sema => "sema",
            Stage::MirBuild => "mir_build",
            Stage::Optimize => "optimize",
            Stage::Backend => "codegen+obj",
        }
    }
}

/// One file's per-stage timings and the size counters that set each stage's floor.
struct Prof {
    name: String,
    src_bytes: usize,
    tokens: usize,
    ast_nodes: u32,
    mir_ops: usize,
    obj_bytes: usize,
    /// Indexed by `Stage::ALL` order.
    stage: [Duration; 6],
    /// Backend split (from `wukong_codegen_cranelift::emit_object_timed`): Cranelift codegen
    /// (isel/regalloc/machine-code emit) vs object-container serialization. Sub-slices of `stage[5]`.
    bk_codegen: Duration,
    bk_obj_write: Duration,
    /// In-process JIT compile of the same optimized program (`--run --backend=native`'s codegen),
    /// for files with a `main` the JIT accepts. Not a stage of the object path, so not in `stage`.
    jit: Option<Duration>,
}

impl Prof {
    fn total(&self) -> Duration {
        self.stage.iter().copied().sum()
    }
}

pub fn report(files: &[PathBuf]) {
    let mut profs: Vec<Prof> = Vec::new();
    let mut skipped = 0u32;
    for path in files {
        match measure_one(path) {
            Some(p) => profs.push(p),
            None => skipped += 1,
        }
    }

    if profs.is_empty() {
        println!(
            "compile-profile: no corpus file compiled to object (measured 0, skipped {skipped})"
        );
        return;
    }

    // --- Stage totals + shares ---
    let mut stage_tot = [Duration::ZERO; 6];
    for p in &profs {
        for (tot, st) in stage_tot.iter_mut().zip(&p.stage) {
            *tot += *st;
        }
    }
    let grand: Duration = stage_tot.iter().copied().sum();

    // Aggregate size counters (context for the throughput a stage floor implies).
    let src_bytes: usize = profs.iter().map(|p| p.src_bytes).sum();
    let tokens: usize = profs.iter().map(|p| p.tokens).sum();
    let ast_nodes: u64 = profs.iter().map(|p| p.ast_nodes as u64).sum();
    let mir_ops: usize = profs.iter().map(|p| p.mir_ops).sum();
    let obj_bytes: usize = profs.iter().map(|p| p.obj_bytes).sum();

    println!(
        "full-pipeline stage breakdown  [regime: warm steady-state, in-process, best-of-N min]\n\
         corpus: {} file(s) compiled to object, {skipped} skipped; -O{LEVEL}; best-of-N per stage.\n",
        profs.len()
    );
    println!(
        "{:<14} {:>11} {:>7}   throughput (work / stage-time)",
        "stage", "total", "%"
    );
    println!("{}", "-".repeat(72));
    for (i, s) in Stage::ALL.iter().enumerate() {
        let t = stage_tot[i];
        let share = 100.0 * t.as_secs_f64() / grand.as_secs_f64().max(1e-12);
        let thru = match s {
            Stage::Lex => rate(src_bytes as f64, t, "MB/s", 1e6),
            Stage::Parse => rate(tokens as f64, t, "Mtok/s", 1e6),
            Stage::Sema => rate(ast_nodes as f64, t, "Mnode/s", 1e6),
            Stage::MirBuild => rate(ast_nodes as f64, t, "Mnode/s", 1e6),
            Stage::Optimize => rate(mir_ops as f64, t, "Mop/s", 1e6),
            Stage::Backend => rate(obj_bytes as f64, t, "MB-obj/s", 1e6),
        };
        println!("{:<14} {:>11} {:>6.1}%   {}", s.name(), fmt(t), share, thru);
    }
    println!("{}", "-".repeat(72));
    println!("{:<14} {:>11} {:>6.1}%", "TOTAL", fmt(grand), 100.0);

    // Backend split (isel/regalloc/emit vs object-container serialization). Bounds the object-emit
    // share §4/§7.4 reasons about analytically — now measured directly via `emit_object_timed`.
    let bk_cg: Duration = profs.iter().map(|p| p.bk_codegen).sum();
    let bk_ow: Duration = profs.iter().map(|p| p.bk_obj_write).sum();
    let bk = (bk_cg + bk_ow).as_secs_f64().max(1e-12);
    println!(
        "  of which codegen+obj:  Cranelift codegen (isel/regalloc/emit) {:>6.1}%   object-write (container) {:>5.1}%",
        100.0 * bk_cg.as_secs_f64() / bk,
        100.0 * bk_ow.as_secs_f64() / bk,
    );
    // JIT row: the `--run --backend=native` codegen of the same programs, against the object path's
    // backend stage over the same files (a ratio, never an absolute rate).
    let jitted: Vec<(Duration, Duration)> = profs
        .iter()
        .filter_map(|p| p.jit.map(|j| (j, p.stage[5])))
        .collect();
    if !jitted.is_empty() {
        let jit_t: Duration = jitted.iter().map(|j| j.0).sum();
        let obj_t: Duration = jitted.iter().map(|j| j.1).sum();
        println!(
            "  JIT (--run --backend=native; not in TOTAL): {} over {} file(s) with `main` = {:.2}x \
             their codegen+obj",
            fmt(jit_t),
            jitted.len(),
            jit_t.as_secs_f64() / obj_t.as_secs_f64().max(1e-12),
        );
    }

    println!(
        "\nwork totals: {} src bytes, {tokens} tokens, {ast_nodes} AST nodes, {mir_ops} MIR ops, \
         {} object bytes.",
        src_bytes, obj_bytes
    );

    // --- Heaviest files ---
    let mut heavy: Vec<&Prof> = profs.iter().collect();
    heavy.sort_by_key(|p| std::cmp::Reverse(p.total()));
    let n = heavy.len().min(10);
    println!(
        "\nheaviest {n} file(s) by total pipeline time  [regime: warm steady-state, best-of-N min]"
    );
    println!(
        "{:<24} {:>7} {:>6} {:>6} {:>7} {:>7}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "file",
        "bytes",
        "toks",
        "ast",
        "mirops",
        "obj",
        "lex",
        "parse",
        "sema",
        "mir",
        "opt",
        "cg+obj",
        "total"
    );
    println!("{}", "-".repeat(139));
    for p in &heavy[..n] {
        println!(
            "{:<24} {:>7} {:>6} {:>6} {:>7} {:>7}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
            trunc(&p.name, 24),
            p.src_bytes,
            p.tokens,
            p.ast_nodes,
            p.mir_ops,
            p.obj_bytes,
            fmt(p.stage[0]),
            fmt(p.stage[1]),
            fmt(p.stage[2]),
            fmt(p.stage[3]),
            fmt(p.stage[4]),
            fmt(p.stage[5]),
            fmt(p.total()),
        );
    }
}

/// Run each stage in isolation on a fixed upstream artifact, timing only that stage. A stage whose
/// input the previous stage mutated (parse and mir_build both extend the interner) rebuilds its
/// input from the fixed tokens each rep, *outside* the timed region, so no `Interner` clone is
/// needed (it is not `Clone`) and the minimum reflects that stage alone. `None` if the file does not
/// compile cleanly to an object at `-O{LEVEL}` — not a regression, just outside this measurement.
fn measure_one(path: &Path) -> Option<Prof> {
    let src = std::fs::read_to_string(path).ok()?;

    // One clean pass: validate runnability and capture the size counters + reusable artifacts.
    let (tokens, ld) = wukong_lexer::tokenize(&src, SourceId(0));
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    let mut interner = Interner::new();
    let (module, pd, ast_nodes) =
        wukong_parser::parse_module_tokens_from(&tokens, &src, &mut interner, 0);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (p0, mld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if mld.iter().any(|d| d.is_error()) {
        return None;
    }
    let mir_ops = count_ops(&p0);
    let mut p_opt = p0.clone();
    wukong_opt::optimize(&mut p_opt, LEVEL);
    let obj = wukong_codegen_cranelift::emit_object(&p_opt, &interner).ok()?;
    let obj_bytes = obj.len();
    // `main` is interned before any stage closure borrows the interner.
    let main = interner.intern("main");

    // CONVENTION: every stage times CONSTRUCTION of its output only. Each rep binds the result,
    // stops the clock, and only then drops it — because `let _ = f();` drops the temporary *before*
    // `t.elapsed()` runs, charging that stage for tearing its own artifact down. `optimize` never
    // paid that (its scratch clone outlives the closure), so the table was comparing five
    // build+destroy stages against one build-only stage. Measured on the 346-file corpus, the
    // destruction inside the timed region was a median 16.7% of `parse` (p90 19.0%, max 21.4%),
    // 3.9% of `sema` and 2.3% of `mir_build`, against ~0% for `lex` and `backend`, whose outputs are
    // single flat allocations. Shares are per-stage attribution, so that bias was not cosmetic.

    // Lex: input is the source text (fixed).
    let lex = best_of(|| {
        let t = Instant::now();
        let out = wukong_lexer::tokenize(&src, SourceId(0));
        let e = t.elapsed();
        drop(out);
        e
    });
    // Parse: input is the fixed token stream; a fresh interner each rep (parse interns identifiers).
    let parse = best_of(|| {
        let mut it = Interner::new();
        let t = Instant::now();
        let out = wukong_parser::parse_module_tokens_from(&tokens, &src, &mut it, 0);
        let e = t.elapsed();
        drop(out);
        e
    });
    // Sema: reads the fixed module + interner and mutates neither, so time it directly.
    let sema_t = best_of(|| {
        let t = Instant::now();
        let out = wukong_sema::check(&module, &interner);
        let e = t.elapsed();
        drop(out);
        e
    });
    // MIR build: needs a pristine interner (it extends it with kernel symbols). Rebuild module+sema
    // from the fixed tokens each rep, untimed, then time only `lower_program`.
    let mir = best_of(|| {
        let mut it = Interner::new();
        let (m, _, _) = wukong_parser::parse_module_tokens_from(&tokens, &src, &mut it, 0);
        let (s, _) = wukong_sema::check(&m, &it);
        let t = Instant::now();
        let out = wukong_mir_build::lower_program(&m, &s, &mut it);
        let e = t.elapsed();
        drop(out);
        e
    });
    // Optimize: clone the unoptimized program each rep (untimed), then time `optimize`.
    let optimize = best_of(|| {
        let mut p = p0.clone();
        let t = Instant::now();
        wukong_opt::optimize(&mut p, LEVEL);
        t.elapsed()
    });
    // Backend: Cranelift codegen + object serialization on the fixed optimized program.
    let backend = best_of(|| {
        let t = Instant::now();
        let out = wukong_codegen_cranelift::emit_object(&p_opt, &interner);
        let e = t.elapsed();
        drop(out);
        e
    });
    // Backend split (isel/regalloc/emit vs object-container write). `emit_object_timed` returns the
    // two halves from *inside* the backend; take a best-of-N min of each independently so the split
    // reflects the same warm steady-state as the combined `backend` stage above.
    let bk_codegen = best_of(|| {
        wukong_codegen_cranelift::emit_object_timed(&p_opt, &interner)
            .map(|(_, t)| t.codegen)
            .unwrap_or(Duration::ZERO)
    });
    let bk_obj_write = best_of(|| {
        wukong_codegen_cranelift::emit_object_timed(&p_opt, &interner)
            .map(|(_, t)| t.object_write)
            .unwrap_or(Duration::ZERO)
    });
    // JIT: the same optimized program compiled into executable memory, as `--run --backend=native`
    // does, at the matching backend level. Only files whose `main` the JIT accepts are timed; the
    // handle (and its executable memory) is dropped after the clock stops.
    let jit_backend = wukong_codegen_cranelift::BackendOpt::for_level(LEVEL);
    let jit_ok = p_opt.function(main).is_some()
        && wukong_codegen_cranelift::jit_compile_with(&p_opt, main, &interner, jit_backend).is_ok();
    let jit = jit_ok.then(|| {
        best_of(|| {
            let t = Instant::now();
            let out =
                wukong_codegen_cranelift::jit_compile_with(&p_opt, main, &interner, jit_backend);
            let e = t.elapsed();
            drop(out);
            e
        })
    });

    Some(Prof {
        name: short(path),
        src_bytes: src.len(),
        tokens: tokens.len(),
        ast_nodes,
        mir_ops,
        obj_bytes,
        stage: [lex, parse, sema_t, mir, optimize, backend],
        bk_codegen,
        bk_obj_write,
        jit,
    })
}

// ===================================================================================================
// spawn-overhead
// ===================================================================================================

/// How many corpus files to characterize under spawn (each `--emit=exe` rep spawns rustc, so keep
/// the set small to stay quick).
const MAX_SPAWN_FILES: usize = 6;
/// Warm best-of budget/rep-cap for the spawn sweep (spawns are ~ms–100ms, so a bigger budget).
const SPAWN_BUDGET: Duration = Duration::from_millis(900);
const SPAWN_MAX_REPS: u32 = 25;

pub fn spawn_report(files: &[PathBuf]) {
    let mc = default_wukongc();
    if !mc.exists() {
        eprintln!(
            "spawn-overhead: wukongc not found at {} — build it (cargo build --release -p wukongc) \
             so the spawn path can be measured against the in-process API.",
            mc.display()
        );
        return;
    }
    let workdir = std::env::temp_dir().join("wukong_spawn_overhead");
    let _ = std::fs::create_dir_all(&workdir);

    // Pick the first few corpus files that compile to an object in-process (so both paths are
    // measuring the same, valid work), copying each into an isolated workdir the child runs in. The
    // source is kept in memory so the in-process timing never re-reads it from disk (the spawn path's
    // own file read stays counted in the tax, as it should be).
    let mut chosen: Vec<(String, PathBuf, String)> = Vec::new();
    for f in files {
        if chosen.len() >= MAX_SPAWN_FILES {
            break;
        }
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        if compile_to_object(&src).is_none() {
            continue;
        }
        let name = short(f);
        let wk = workdir.join(&name);
        if std::fs::write(&wk, &src).is_err() {
            continue;
        }
        chosen.push((name, wk, src));
    }
    if chosen.is_empty() {
        eprintln!(
            "spawn-overhead: no corpus file compiled to object in-process — nothing to compare."
        );
        return;
    }

    println!(
        "in-process API vs process-spawn, code -> object (and -> exe), -O{LEVEL}\n\
         wukongc: {}\n\
         (a) in-proc = lex..emit_object in memory;  (b) spawn = wukongc.exe end-to-end child wall.\n",
        mc.display()
    );

    // --- First-call table ---
    // Only the FIRST row is a true cold start: after it, wukongc.exe, its rlibs and the OS page
    // cache are warm for every later row, so those rows are warm-image first-calls. Labelled as
    // such rather than as "cold-start", which they are not.
    println!("[regime: first call for this file — the wukongc image is cold only for row 1 (*)]");
    println!(
        "{:<24} {:>12} {:>12} {:>12}",
        "file", "in-proc obj", "spawn obj", "spawn exe"
    );
    println!("{}", "-".repeat(64));
    // --- collect warm numbers as we go for the second table ---
    struct Row {
        name: String,
        inproc: Duration,
        spawn_obj: Duration,
        spawn_exe: Option<Duration>,
    }
    let mut warm_rows: Vec<Row> = Vec::new();

    let mut printed = 0usize;
    for (name, wk, src) in &chosen {
        let out_o = workdir.join("out.o");
        let out_exe = workdir.join("out.exe");

        let (ip_cold, ip_warm) = cold_warm(|| {
            let t = Instant::now();
            let _ = compile_to_object(src);
            t.elapsed()
        });
        // The obj spawn goes FIRST, so its cold sample is the genuine first `wukongc.exe` launch for
        // this file. (It used to be preceded by an untimed `--emit=exe` probe spawn, which warmed the
        // compiler image, its rlibs and the page cache before the "cold" number was taken.)
        let mut obj_ok = true;
        let (obj_cold, obj_warm) = cold_warm(|| {
            let (d, ok) = spawn_compile(&mc, wk, &workdir, "obj", &out_o);
            obj_ok &= ok;
            d
        });
        // A child that exits nonzero produced no object: what was timed is an error-and-exit, not a
        // compile. Files are admitted to `chosen` by the IN-PROCESS pipeline, which never runs the
        // driver's own front matter (source maps, import resolution, `-o` handling), so the driver
        // can legitimately reject one — e.g. a file whose `import` resolves in the repo but not in
        // the isolated workdir the child runs in. Timing that would report a fast failure as a small
        // spawn tax, and `saturating_sub` would then clamp it to `0ns` — "process spawn is free".
        if !obj_ok {
            eprintln!("{name}: spawn --emit=obj failed — excluded (nothing was compiled to time)");
            continue;
        }
        // Does `--emit=exe` link here (needs rustc + the runtime rlib next to wukongc)? Answered by
        // the status of the timed spawns themselves rather than by a separate untimed probe.
        let mut exe_ok = true;
        let (exe_cold, exe_warm) = {
            let (c, w) = cold_warm(|| {
                let (d, ok) = spawn_compile(&mc, wk, &workdir, "exe", &out_exe);
                exe_ok &= ok;
                d
            });
            if exe_ok {
                (Some(c), Some(w))
            } else {
                (None, None)
            }
        };

        println!(
            "{:<24} {:>12} {:>12} {:>12}{}",
            trunc(name, 24),
            fmt(ip_cold),
            fmt(obj_cold),
            exe_cold.map(fmt).unwrap_or_else(|| "n/a".into()),
            if printed == 0 { "  *" } else { "" },
        );
        printed += 1;
        warm_rows.push(Row {
            name: name.clone(),
            inproc: ip_warm,
            spawn_obj: obj_warm,
            spawn_exe: exe_warm,
        });
    }
    println!(
        "* row 1 only: wukongc.exe, its rlibs and the OS page cache are cold for it. Every later \
         row's\n  spawn columns are warm-image first-calls, so they are NOT cold-start numbers."
    );
    if warm_rows.is_empty() {
        eprintln!(
            "spawn-overhead: every chosen file failed to compile under the driver — no rows."
        );
        return;
    }

    // --- Warm steady-state table ---
    println!("\n[regime: warm steady-state / best-of-N min]");
    println!(
        "{:<24} {:>12} {:>12} {:>11} {:>12} {:>11}",
        "file", "in-proc obj", "spawn obj", "spawn-tax", "spawn exe", "link(exe-obj)"
    );
    println!("{}", "-".repeat(88));
    for r in &warm_rows {
        let tax = r.spawn_obj.saturating_sub(r.inproc);
        let (exe, link) = match r.spawn_exe {
            Some(e) => (fmt(e), fmt(e.saturating_sub(r.spawn_obj))),
            None => ("n/a".into(), "n/a".into()),
        };
        println!(
            "{:<24} {:>12} {:>12} {:>11} {:>12} {:>11}",
            trunc(&r.name, 24),
            fmt(r.inproc),
            fmt(r.spawn_obj),
            fmt(tax),
            exe,
            link,
        );
    }
    println!("{}", "-".repeat(88));
    println!(
        "spawn-tax = (b) spawn-obj wall - (a) in-process-obj wall: process creation + runtime init + \
         file I/O\n  + the driver's source-map/diagnostics/import front-matter that the in-process \
         path skips.\nlink(exe-obj) = the rustc-driven link the driver spawns for --emit=exe \
         (links wukong_runtime + the\n  platform linker); it is a whole second process, so it \
         dominates the exe path. 'n/a' = the child\n  exited nonzero for this file (rustc or the \
         runtime rlib not next to wukongc, or the link failed),\n  so there is no exe compile to \
         time. Any file whose --emit=obj spawn failed is dropped entirely."
    );
}

/// Full in-process pipeline, source text to object bytes. `None` on any stage error.
fn compile_to_object(src: &str) -> Option<Vec<u8>> {
    let (tokens, ld) = wukong_lexer::tokenize(src, SourceId(0));
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    let mut interner = Interner::new();
    let (module, pd, _) = wukong_parser::parse_module_tokens_from(&tokens, src, &mut interner, 0);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (mut program, mld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    if mld.iter().any(|d| d.is_error()) {
        return None;
    }
    wukong_opt::optimize(&mut program, LEVEL);
    wukong_codegen_cranelift::emit_object(&program, &interner).ok()
}

/// Spawn `wukongc --emit=<emit> -O{LEVEL} <file> -o <out>` from `workdir` and return the child wall
/// time **and whether it succeeded**. Output is discarded. The child writes its intermediates into
/// `workdir`, isolated from the repo.
///
/// The status is load-bearing, not decoration: with both pipes at `Stdio::null()` a failed compile
/// is invisible, and its wall time — an error-and-exit that emitted nothing — would otherwise be
/// reported as a spawn measurement. Callers must drop any file whose spawn ever fails.
fn spawn_compile(mc: &Path, wk: &Path, workdir: &Path, emit: &str, out: &Path) -> (Duration, bool) {
    let file = wk.file_name().unwrap().to_string_lossy().into_owned();
    let out = out.to_string_lossy().into_owned();
    let t = Instant::now();
    let st = Command::new(mc)
        .current_dir(workdir)
        .args([&format!("--emit={emit}"), "-O2", &file, "-o", &out])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let dt = t.elapsed();
    (dt, st.map(|s| s.success()).unwrap_or(false))
}

fn default_wukongc() -> PathBuf {
    let exe = if cfg!(windows) {
        "wukongc.exe"
    } else {
        "wukongc"
    };
    PathBuf::from("target/release").join(exe)
}

// ===================================================================================================
// shared helpers
// ===================================================================================================

/// Best-of-N minimum: warm up, then keep the minimum single-run time until the wall budget or rep
/// cap, with a hard floor of `MIN_REPS`. The minimum is the run least perturbed by scheduler noise.
fn best_of<F: FnMut() -> Duration>(mut run: F) -> Duration {
    run();
    run();
    let mut best = Duration::MAX;
    let mut reps = 0u32;
    let start = Instant::now();
    loop {
        best = best.min(run());
        reps += 1;
        if reps >= MAX_REPS || (reps >= MIN_REPS && start.elapsed() >= BUDGET) {
            break;
        }
    }
    best
}

/// First call (cold), then a warm best-of-N minimum — the pair the spawn characterization needs.
fn cold_warm<F: FnMut() -> Duration>(mut run: F) -> (Duration, Duration) {
    let cold = run();
    let mut best = Duration::MAX;
    let mut reps = 0u32;
    let start = Instant::now();
    loop {
        best = best.min(run());
        reps += 1;
        if reps >= SPAWN_MAX_REPS || (reps >= MIN_REPS && start.elapsed() >= SPAWN_BUDGET) {
            break;
        }
    }
    (cold, best)
}

/// Total MIR operations: instructions plus one terminator per block.
fn count_ops(p: &wukong_mir::Program) -> usize {
    p.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len() + 1)
        .sum()
}

/// `count / seconds` rendered in `unit`s of `scale` per second, or `-` when the time is zero.
fn rate(count: f64, t: Duration, unit: &str, scale: f64) -> String {
    let s = t.as_secs_f64();
    if s <= 0.0 || count <= 0.0 {
        return "-".into();
    }
    format!("{:.1} {unit}", count / s / scale)
}

fn fmt(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns >= 1_000_000 {
        format!("{:.2}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2}us", ns as f64 / 1e3)
    } else {
        format!("{ns}ns")
    }
}

fn short(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}

/// Truncate to at most `w` characters (char-boundary-safe: byte slicing would panic on a multibyte
/// filename).
fn trunc(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_string()
    } else {
        let cut: String = s.chars().take(w.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

// ===================================================================================================
// cold-start and scaling (P6.10)
// ===================================================================================================
//
// Both modes time FRESH PROCESSES of the real compiler binary, because both quantities are
// properties of a process, not of a call: a cold start is compile + JIT + runtime initialization +
// the first execution, all of which a warm in-process loop amortizes away, and the runtime reads its
// thread count (`WUKONG_NUM_THREADS`) exactly once per process. Every child's exit status and stdout
// are checked - a child that failed timed an error path, and a sweep whose answer changed with the
// thread count measured two different computations.
//
// The two report lines are a contract with `docs/sota-campaign-2026-09/analyze-sota-measure.py`
// (`COLD_START_LINE` / `SCALING_LINE`, pinned by its `--self-test`); `cold_start_line` and
// `scaling_line` are the only producers, and the unit tests below pin their spelling:
//
//   cold-start: program=<name> backend=<b> opt=O2 runs=<n> min_ms=<f> median_ms=<f> p95_ms=<f> max_ms=<f> first_ms=<f>
//   scaling: workload=<name> threads=<n> runs=<k> min_ms=<f> median_ms=<f> speedup=<f> efficiency=<f>

/// The fixed cold-start program: a recognized `@parallel` 128x128x128 GEMM (so the first run also
/// pays the runtime's thread-pool start and the kernel's first call) followed by a reduction and a
/// print. Every GEMM lane is a small exact integer and the checksum reduction has a fixed
/// partition, so every run prints the same line.
const COLD_START_SRC: &str = "module bench.cold_start

@parallel
fn mm(a: [f32; 16384], b: [f32; 16384], mut c: [f32; 16384]) {
    for i in 0..128 {
        for j in 0..128 {
            let mut s: f32 = 0.0;
            for k in 0..128 {
                s = s + a[i * 128 + k] * b[k * 128 + j];
            }
            c[i * 128 + j] = s;
        }
    }
}

fn main() -> i32 {
    let mut a: [f32; 16384] = [0.0; 16384];
    let mut b: [f32; 16384] = [0.0; 16384];
    let mut c: [f32; 16384] = [0.0; 16384];
    for t in 0..16384 {
        a[t] = (t % 7) as f32;
        b[t] = (t % 5) as f32;
    }
    mm(a, b, c);
    print(sum(c));
    return 0;
}
";

/// The scaling workload: `reps` calls of a recognized `@parallel` 512x512x512 GEMM
/// (`wukong_sgemm_parallel`, which runs on the pool `WUKONG_NUM_THREADS` sizes), then a checksum.
/// Every lane is a small exact integer and the checksum reduction's partition is fixed, so the
/// printed line cannot depend on the thread count unless something is wrong - which is checked.
fn scaling_src(reps: usize) -> String {
    format!(
        "module bench.scaling

@parallel
fn mm(a: [f32; 262144], b: [f32; 262144], mut c: [f32; 262144]) {{
    for i in 0..512 {{
        for j in 0..512 {{
            let mut s: f32 = 0.0;
            for k in 0..512 {{
                s = s + a[i * 512 + k] * b[k * 512 + j];
            }}
            c[i * 512 + j] = s;
        }}
    }}
}}

fn main() -> i32 {{
    let mut a: [f32; 262144] = [0.0; 262144];
    let mut b: [f32; 262144] = [0.0; 262144];
    let mut c: [f32; 262144] = [0.0; 262144];
    for t in 0..262144 {{
        a[t] = (t % 7) as f32;
        b[t] = (t % 5) as f32;
    }}
    for _r in 0..{reps} {{
        mm(a, b, c);
    }}
    print(sum(c));
    return 0;
}}
"
    )
}

/// Parsed `--key=value` options of the two process-level modes.
#[derive(Debug, PartialEq)]
struct ProcessOpts {
    runs: usize,
    backend: String,
    wukongc: Option<PathBuf>,
    program: Option<PathBuf>,
    max_threads: Option<usize>,
    reps: usize,
}

fn parse_process_opts(args: &[String], default_runs: usize) -> Result<ProcessOpts, String> {
    let mut o = ProcessOpts {
        runs: default_runs,
        backend: "native".into(),
        wukongc: None,
        program: None,
        max_threads: None,
        reps: 200,
    };
    let positive = |k: &str, v: &str| -> Result<usize, String> {
        v.parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("{k} needs a positive integer, got {v:?}"))
    };
    for a in args {
        if let Some(v) = a.strip_prefix("--runs=") {
            o.runs = positive("--runs", v)?;
        } else if let Some(v) = a.strip_prefix("--backend=") {
            if v != "native" && v != "interp" {
                return Err(format!("--backend must be native or interp, got {v:?}"));
            }
            o.backend = v.into();
        } else if let Some(v) = a.strip_prefix("--wukongc=") {
            o.wukongc = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--max-threads=") {
            o.max_threads = Some(positive("--max-threads", v)?);
        } else if let Some(v) = a.strip_prefix("--reps=") {
            o.reps = positive("--reps", v)?;
        } else if a.starts_with("--") {
            return Err(format!("unknown option {a:?}"));
        } else if o.program.is_none() {
            o.program = Some(PathBuf::from(a));
        } else {
            return Err(format!("more than one program given ({a:?})"));
        }
    }
    Ok(o)
}

/// Min / median / p95 / max of a non-empty sample set, in the samples' unit.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Spread {
    min: f64,
    median: f64,
    p95: f64,
    max: f64,
}

/// `p95` is the nearest-rank percentile (`ceil(0.95 n)`-th smallest), so it is always an observed
/// sample and never an interpolation between two; with fewer than 20 runs it is the maximum.
fn spread(samples: &[f64]) -> Option<Spread> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    let rank = ((0.95 * n as f64).ceil() as usize).clamp(1, n);
    Some(Spread {
        min: s[0],
        median,
        p95: s[rank - 1],
        max: s[n - 1],
    })
}

/// The thread counts a sweep visits: 1, 2, 4, ... and then `max` itself if it is not a power of
/// two, so "all" is always measured.
fn thread_sweep(max: usize) -> Vec<usize> {
    let max = max.max(1);
    let mut v = Vec::new();
    let mut t = 1usize;
    while t < max {
        v.push(t);
        t *= 2;
    }
    v.push(max);
    v
}

/// The cold-start report line (see the module comment for the contract).
fn cold_start_line(program: &str, backend: &str, runs: usize, s: Spread, first_ms: f64) -> String {
    format!(
        "cold-start: program={program} backend={backend} opt=O2 runs={runs} min_ms={:.3} \
         median_ms={:.3} p95_ms={:.3} max_ms={:.3} first_ms={first_ms:.3}",
        s.min, s.median, s.p95, s.max
    )
}

/// One scaling report line. `speedup` is the 1-thread median over this count's median, and
/// `efficiency` is that speedup per thread.
fn scaling_line(
    workload: &str,
    threads: usize,
    runs: usize,
    s: Spread,
    one_thread_median: f64,
) -> String {
    let speedup = one_thread_median / s.median;
    format!(
        "scaling: workload={workload} threads={threads} runs={runs} min_ms={:.3} median_ms={:.3} \
         speedup={speedup:.3} efficiency={:.3}",
        s.min,
        s.median,
        speedup / threads as f64
    )
}

/// One fresh `wukongc --run` child: its wall time in ms and its stdout, or why it is not a sample.
fn spawn_run(
    mc: &Path,
    workdir: &Path,
    file: &str,
    backend: &str,
    threads: Option<usize>,
) -> Result<(f64, String), String> {
    let mut cmd = Command::new(mc);
    cmd.current_dir(workdir)
        .args(["--run", &format!("--backend={backend}"), "-O2", file])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(t) = threads {
        cmd.env("WUKONG_NUM_THREADS", t.to_string());
    } else {
        cmd.env_remove("WUKONG_NUM_THREADS");
    }
    let t0 = Instant::now();
    let out = cmd
        .output()
        .map_err(|e| format!("cannot spawn {}: {e}", mc.display()))?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    if !out.status.success() {
        return Err(format!(
            "child exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok((ms, String::from_utf8_lossy(&out.stdout).into_owned()))
}

/// Resolve the compiler binary and the program to time; `fixed` is the built-in source used when
/// no `.wk` path is given. Returns (wukongc, workdir, file name in workdir, program label).
fn stage_program(
    mode: &str,
    o: &ProcessOpts,
    fixed: &str,
    fixed_name: &str,
) -> Result<(PathBuf, PathBuf, String, String), String> {
    let mc = o.wukongc.clone().unwrap_or_else(default_wukongc);
    // Absolute, because the child runs with `current_dir(workdir)` and a relative program path is
    // resolved differently across platforms once the working directory changes.
    let mc = std::path::absolute(&mc).unwrap_or(mc);
    if !mc.exists() {
        return Err(format!(
            "wukongc not found at {} - build it (cargo build --release -p wukongc) or pass \
             --wukongc=<path>",
            mc.display()
        ));
    }
    let workdir = std::env::temp_dir().join(format!("wukong_{}", mode.replace('-', "_")));
    std::fs::create_dir_all(&workdir).map_err(|e| format!("{}: {e}", workdir.display()))?;
    let (src, label) = match &o.program {
        Some(p) => (
            std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
            short(p).trim_end_matches(".wk").to_string(),
        ),
        None => (fixed.to_string(), fixed_name.to_string()),
    };
    let file = format!("{label}.wk");
    std::fs::write(workdir.join(&file), src).map_err(|e| format!("{file}: {e}"))?;
    Ok((mc, workdir, file, label))
}

/// `cold-start [--runs=N] [--backend=native|interp] [--wukongc=PATH] [program.wk]`. Returns the
/// process exit code: 0 measured, 1 a child failed or printed something different, 2 usage.
pub fn cold_start_report(args: &[String]) -> i32 {
    let o = match parse_process_opts(args, 10) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("cold-start: {e}");
            return 2;
        }
    };
    let (mc, workdir, file, label) =
        match stage_program("cold-start", &o, COLD_START_SRC, "cold_start_gemm128") {
            Ok(v) => v,
            Err(e) => {
                eprintln!("cold-start: {e}");
                return 2;
            }
        };
    println!(
        "cold-start: {} fresh `wukongc --run --backend={} -O2` processes of `{label}` \
         (wukongc: {})\n  each = process start + compile + JIT + runtime init + the first run; \
         wall time per child, ms",
        o.runs,
        o.backend,
        mc.display()
    );
    let mut samples = Vec::with_capacity(o.runs);
    let mut first_out: Option<String> = None;
    for run in 0..o.runs {
        match spawn_run(&mc, &workdir, &file, &o.backend, None) {
            Ok((ms, out)) => {
                match &first_out {
                    None => first_out = Some(out),
                    Some(f) if *f != out => {
                        eprintln!(
                            "cold-start: run {run} printed {out:?}, run 0 printed {f:?} - not the \
                             same computation, nothing to report"
                        );
                        return 1;
                    }
                    Some(_) => {}
                }
                println!("cold-start sample: run={run} ms={ms:.3}");
                samples.push(ms);
            }
            Err(e) => {
                eprintln!("cold-start: run {run}: {e} - a failed child is not a sample");
                return 1;
            }
        }
    }
    let Some(s) = spread(&samples) else {
        eprintln!("cold-start: no samples");
        return 1;
    };
    println!(
        "{}",
        cold_start_line(&label, &o.backend, samples.len(), s, samples[0])
    );
    println!(
        "  run 0 (first_ms) is the only one with a cold compiler image and page cache; min/median/\n  \
         p95 are over all runs. A lone figure is not a measurement on this host - compare arms by \
         ratio."
    );
    0
}

/// `scaling [--runs=N] [--max-threads=N] [--reps=N] [--wukongc=PATH] [workload.wk]`: the same
/// workload in fresh processes at `WUKONG_NUM_THREADS` = 1, 2, 4, .., all. Exit codes as for
/// [`cold_start_report`].
pub fn scaling_report(args: &[String]) -> i32 {
    let o = match parse_process_opts(args, 3) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("scaling: {e}");
            return 2;
        }
    };
    let fixed = scaling_src(o.reps);
    let name = format!("scaling_gemm512x{}", o.reps);
    let (mc, workdir, file, label) = match stage_program("scaling", &o, &fixed, &name) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("scaling: {e}");
            return 2;
        }
    };
    let all = o.max_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let sweep = thread_sweep(all);
    println!(
        "scaling: workload={label} backend={} sweep={} ({} fresh process(es) per count; \
         wukongc: {})\n  wall time per child includes the serial compile + start-up, so the \
         efficiency below is a whole-process (Amdahl) figure, not the parallel region's",
        o.backend,
        sweep
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        o.runs,
        mc.display()
    );
    let mut reference: Option<String> = None;
    let mut one_thread_median = None;
    for &t in &sweep {
        let mut samples = Vec::with_capacity(o.runs);
        for run in 0..o.runs {
            match spawn_run(&mc, &workdir, &file, &o.backend, Some(t)) {
                Ok((ms, out)) => {
                    match &reference {
                        None => reference = Some(out),
                        Some(r) if *r != out => {
                            eprintln!(
                                "scaling: threads={t} run {run} printed {out:?}, threads=1 printed \
                                 {r:?} - the answer depends on the thread count; nothing to report"
                            );
                            return 1;
                        }
                        Some(_) => {}
                    }
                    samples.push(ms);
                }
                Err(e) => {
                    eprintln!("scaling: threads={t} run {run}: {e}");
                    return 1;
                }
            }
        }
        let Some(s) = spread(&samples) else {
            eprintln!("scaling: threads={t}: no samples");
            return 1;
        };
        let base = *one_thread_median.get_or_insert(s.median);
        println!("{}", scaling_line(&label, t, samples.len(), s, base));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn spread_uses_nearest_rank_p95_and_the_true_median() {
        let s = spread(&[5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        assert_eq!(
            s,
            Spread {
                min: 1.0,
                median: 3.0,
                p95: 5.0,
                max: 5.0
            }
        );
        // Even count: the median is the midpoint; with 20 samples p95 is the 19th smallest.
        let twenty: Vec<f64> = (1..=20).map(f64::from).collect();
        let s = spread(&twenty).unwrap();
        assert_eq!(s.median, 10.5);
        assert_eq!(s.p95, 19.0);
        assert_eq!(s.max, 20.0);
        assert!(spread(&[]).is_none());
        assert!(spread(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn the_thread_sweep_doubles_and_always_ends_at_all() {
        assert_eq!(thread_sweep(1), vec![1]);
        assert_eq!(thread_sweep(8), vec![1, 2, 4, 8]);
        assert_eq!(thread_sweep(22), vec![1, 2, 4, 8, 16, 22]);
        assert_eq!(thread_sweep(0), vec![1]);
    }

    /// The two report lines are parsed by `analyze-sota-measure.py` (`COLD_START_LINE`,
    /// `SCALING_LINE`, and its `--self-test` embeds these exact spellings). Changing a field here
    /// without changing the analyser breaks that contract, so the spelling is pinned.
    #[test]
    fn the_report_lines_keep_the_analyser_contract() {
        let s = Spread {
            min: 10.0,
            median: 12.5,
            p95: 20.25,
            max: 21.0,
        };
        assert_eq!(
            cold_start_line("cold_start_gemm128", "native", 10, s, 21.0),
            "cold-start: program=cold_start_gemm128 backend=native opt=O2 runs=10 min_ms=10.000 \
             median_ms=12.500 p95_ms=20.250 max_ms=21.000 first_ms=21.000"
        );
        let s4 = Spread {
            min: 30.0,
            median: 31.25,
            p95: 33.0,
            max: 33.0,
        };
        assert_eq!(
            scaling_line("scaling_gemm512x200", 4, 3, s4, 100.0),
            "scaling: workload=scaling_gemm512x200 threads=4 runs=3 min_ms=30.000 median_ms=31.250 \
             speedup=3.200 efficiency=0.800"
        );
    }

    #[test]
    fn process_options_parse_and_refuse_what_they_do_not_know() {
        let o = parse_process_opts(&args(&["--runs=5", "--backend=interp", "x.wk"]), 10).unwrap();
        assert_eq!(o.runs, 5);
        assert_eq!(o.backend, "interp");
        assert_eq!(o.program, Some(PathBuf::from("x.wk")));
        let o = parse_process_opts(&args(&["--max-threads=6", "--reps=7"]), 3).unwrap();
        assert_eq!((o.runs, o.max_threads, o.reps), (3, Some(6), 7));
        assert!(parse_process_opts(&args(&["--runs=0"]), 3).is_err());
        assert!(parse_process_opts(&args(&["--backend=gpu"]), 3).is_err());
        assert!(parse_process_opts(&args(&["--bogus"]), 3).is_err());
        assert!(parse_process_opts(&args(&["a.wk", "b.wk"]), 3).is_err());
    }

    /// The two built-in programs must stay valid Wukong and keep the parallel GEMM kernel the
    /// modes exist to exercise - checked in-process, so a language change that breaks them fails
    /// here rather than in a measured round.
    #[test]
    fn the_fixed_programs_compile_and_dispatch_the_parallel_gemm() {
        for (name, src) in [
            ("cold_start", COLD_START_SRC.to_string()),
            ("scaling", scaling_src(2)),
        ] {
            let mut interner = Interner::new();
            let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            assert!(!pd.iter().any(|d| d.is_error()), "{name}: {pd:?}");
            let (sema, sd) = wukong_sema::check(&module, &interner);
            assert!(!sd.iter().any(|d| d.is_error()), "{name}: {sd:?}");
            let (mut p, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
            assert!(!ld.iter().any(|d| d.is_error()), "{name}: {ld:?}");
            wukong_opt::optimize(&mut p, LEVEL);
            let mir = wukong_mir::print::print_program(&p, &interner);
            assert!(
                mir.contains("call wukong_sgemm_parallel("),
                "{name}: the parallel GEMM kernel no longer dispatches:\n{mir}"
            );
        }
    }
}
