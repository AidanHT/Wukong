//! Whole-pipeline stage profiling and in-process-vs-spawn characterization.
//!
//! Two modes live here, both aimed at the "compile-time floor" question — how close is code→object
//! to the irreducible work, and what does the process-spawn path cost on top of the in-process API.
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
//!   container cost `docs/compile-floor.md` reasons about.
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
//! ```text
//! cargo run -p wukong_bench --release -- compile-profile tests/run examples bench/kernels
//! cargo run -p wukong_bench --release -- spawn-overhead   tests/run examples bench/kernels
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

/// The seven pipeline stages, in order. `Backend` is Cranelift codegen + object serialization
/// together (see the module docs for why they are not split here).
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
        println!("compile-profile: no corpus file compiled to object (measured 0, skipped {skipped})");
        return;
    }

    // --- Stage totals + shares ---
    let mut stage_tot = [Duration::ZERO; 6];
    for p in &profs {
        for i in 0..6 {
            stage_tot[i] += p.stage[i];
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
    println!("{:<14} {:>11} {:>7}   {}", "stage", "total", "%", "throughput (work / stage-time)");
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

    println!(
        "\nwork totals: {} src bytes, {tokens} tokens, {ast_nodes} AST nodes, {mir_ops} MIR ops, \
         {} object bytes.",
        src_bytes, obj_bytes
    );

    // --- Heaviest files ---
    let mut heavy: Vec<&Prof> = profs.iter().collect();
    heavy.sort_by(|a, b| b.total().cmp(&a.total()));
    let n = heavy.len().min(10);
    println!(
        "\nheaviest {n} file(s) by total pipeline time  [regime: warm steady-state, best-of-N min]"
    );
    println!(
        "{:<24} {:>7} {:>6} {:>6} {:>7} {:>7}  {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "file", "bytes", "toks", "ast", "mirops", "obj", "lex", "parse", "sema", "mir", "opt",
        "cg+obj", "total"
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

    // Lex: input is the source text (fixed).
    let lex = best_of(|| {
        let t = Instant::now();
        let _ = wukong_lexer::tokenize(&src, SourceId(0));
        t.elapsed()
    });
    // Parse: input is the fixed token stream; a fresh interner each rep (parse interns identifiers).
    let parse = best_of(|| {
        let mut it = Interner::new();
        let t = Instant::now();
        let _ = wukong_parser::parse_module_tokens_from(&tokens, &src, &mut it, 0);
        t.elapsed()
    });
    // Sema: reads the fixed module + interner and mutates neither, so time it directly.
    let sema_t = best_of(|| {
        let t = Instant::now();
        let _ = wukong_sema::check(&module, &interner);
        t.elapsed()
    });
    // MIR build: needs a pristine interner (it extends it with kernel symbols). Rebuild module+sema
    // from the fixed tokens each rep, untimed, then time only `lower_program`.
    let mir = best_of(|| {
        let mut it = Interner::new();
        let (m, _, _) = wukong_parser::parse_module_tokens_from(&tokens, &src, &mut it, 0);
        let (s, _) = wukong_sema::check(&m, &it);
        let t = Instant::now();
        let _ = wukong_mir_build::lower_program(&m, &s, &mut it);
        t.elapsed()
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
        let _ = wukong_codegen_cranelift::emit_object(&p_opt, &interner);
        t.elapsed()
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
        eprintln!("spawn-overhead: no corpus file compiled to object in-process — nothing to compare.");
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
        eprintln!("spawn-overhead: every chosen file failed to compile under the driver — no rows.");
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
    let exe = if cfg!(windows) { "wukongc.exe" } else { "wukongc" };
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
