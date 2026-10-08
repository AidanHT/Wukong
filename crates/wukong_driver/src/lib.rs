//! `wukong_driver` — orchestrates the compilation pipeline.
//!
//! The driver owns the [`SourceMap`], the [`Interner`] and the [`DiagnosticSink`], runs each stage in
//! order, and honors the requested `--emit` target. [`compile`] wires the whole pipeline: lex
//! (`--emit=tokens`) → parse (`--emit=ast`) → import loading (the `loader` module) → sema → MIR
//! construction (`--emit=mir-high`) → optimization → the autodiff surface (`--train`,
//! `--emit=grad`) → `--run` on the selected backend → `--emit=mir` → the verify gate → the
//! LLVM-IR / object / exe emitters.
//!
//! Flag *combinations* are not validated here: that is `wukongc`'s `parse_args`, the layer that knows
//! whether a flag was given explicitly. This crate owns the option *types* and the stage dispatch.

use std::path::{Path, PathBuf};

use wukong_diag::{Diagnostic, DiagnosticSink, Renderer};
use wukong_span::{Interner, SourceMap};

pub use wukong_codegen_cranelift::TargetCpu;
pub use wukong_diag::{all_explanations, explain, Explanation};
pub use wukong_sema::lints::{LintLevel, LintLevels, LINTS};

/// The GPU offloading bridge (`--backend=gpu`), compiled only with the `gpu` feature.
#[cfg(feature = "gpu")]
mod gpu_accel;

/// **The end-to-end measurement round**: a real `.wk` program through the real pipeline, timed
/// against cuBLAS, with compile time and execution time reported as separate quantities. Test-only —
/// it adds nothing to the shipped library — but the device-free laws in it run in the ordinary
/// `cargo test -p wukong_driver --features gpu --lib` step, which is where they belong: each one pins
/// a way the metered round could bill an H100 and prove nothing.
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_bench;

/// **The ruler** (`WUKONG_MASTER_PLAN` Task 0.4): the honest gpu-native baseline of a `@generic`
/// kernel, and the retro-measurement Phases 1 and 3 owe against it. Test-only, like
/// [`gpu_e2e_bench`]; its device round is `#[ignore]`d and its device-free laws (the fixtures carry
/// `@generic` and a `@parallel` anchor; the control fixture exists) run in the ordinary
/// `cargo test -p wukong_driver --features gpu --lib` step.
#[cfg(all(test, feature = "gpu"))]
mod lang_bench;

/// The multi-file front end: resolves `import`s to files, loads each once (depth-first,
/// cycle-safe), resolves every name against its own file's module scope (`pub`, `as`, `{x, y}`),
/// and splices the items into the root module under program-unique symbols.
mod loader;

/// Whole-program reachability pruning ahead of optimization (`--run`, `--emit=exe`).
mod prune;

/// `--time-passes` / `--time-trace` stage timing and `--depfile` dependency files.
mod report;

/// Which intermediate (or final) artifact the user asked to produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmitStage {
    Tokens,
    Ast,
    /// Run the front end only — lex, parse, import resolution, sema and the lints — and stop
    /// before MIR lowering (`--emit=check`). Reports every diagnostic sema and the lints produce and
    /// exits 0 when nothing is an error (a denied lint is one), with no lowering (`C0001`) limits in
    /// the way. The fast check loop an editor drives.
    Check,
    MirHigh,
    Mir,
    /// The **backward** MIR of a designated loss function (reverse-mode autodiff): the forward
    /// function followed by its `{name}_grad` twin (see `--grad-of` / `--grad-wrt`). Additive to
    /// the other stages — it runs after optimization (forcing at least `-O1`, which the autodiff
    /// transform requires for its single-block SSA input) and emits both functions' MIR.
    Grad,
    LlvmIr,
    Obj,
    Exe,
}

impl EmitStage {
    pub fn parse(s: &str) -> Option<EmitStage> {
        Some(match s {
            "tokens" => EmitStage::Tokens,
            "ast" => EmitStage::Ast,
            "check" => EmitStage::Check,
            "mir-high" => EmitStage::MirHigh,
            "mir" | "mir-low" => EmitStage::Mir,
            "grad" => EmitStage::Grad,
            "llvm-ir" => EmitStage::LlvmIr,
            "obj" => EmitStage::Obj,
            "exe" => EmitStage::Exe,
            _ => return None,
        })
    }
}

/// How diagnostics are presented to the user.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrorFormat {
    /// rustc-style annotated terminal output.
    Human,
    /// One JSON object per diagnostic, on its own line (JSON Lines).
    Json,
}

/// Which backend executes (`--run`) or emits native code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind {
    /// The tree-walking MIR interpreter (zero dependencies, the reference oracle).
    Interp,
    /// Cranelift native codegen — JIT for `--run`, object/exe for `--emit`.
    Native,
    /// GPU offload (`--features gpu`): the interpreter tree-walks the program but runs recognized
    /// GEMM / norm / activation / reduction calls on the local CUDA device. Tolerance-gated against
    /// the interpreter oracle (the CPU↔GPU boundary cannot be bit-exact).
    Gpu,
    /// General GPU lowering (`--features gpu`, `--backend=gpu-native`): lowers the *whole* program's
    /// MIR to PTX and runs it on the device (Phase 4) — not the recognizer offload. Additive to and
    /// independent of [`BackendKind::Gpu`]; tolerance-gated against the interpreter oracle.
    GpuLower,
}

/// Compiler invocation options, normally produced by the CLI.
#[derive(Clone, Debug)]
pub struct Options {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub emit: EmitStage,
    pub run: bool,
    pub opt_level: u8,
    /// Validate MIR after each optimizer pass, including in release builds (`--verify-each`).
    pub verify_each: bool,
    /// Lint levels from the command line (`-A/-W/-D <lint>`, `-D warnings`), applied on top of each
    /// lint's default; `@allow/@warn/@deny` attributes in the source refine them per item. See
    /// `wukong_sema::lints`.
    pub lint_levels: LintLevels,
    /// Print a per-stage compile-time breakdown — with per-optimizer-pass and codegen splits and a
    /// MIR size summary — to stderr when the compile ends (`--time-passes`).
    pub time_passes: bool,
    /// Write the same timings as a Chrome trace-event JSON file (`--time-trace=<path>`).
    pub time_trace: Option<PathBuf>,
    /// Write a Make-syntax dependency file naming every source file the compile read, root and
    /// imports (`--depfile=<path>`).
    pub depfile: Option<PathBuf>,
    pub color: bool,
    pub error_format: ErrorFormat,
    /// Cap on how many error-severity diagnostics are printed before the rest are suppressed
    /// (`--error-limit=<n>`). `None` is unlimited (the default). Only errors count toward the cap,
    /// but once it is exceeded every later diagnostic in that batch (warnings included) is
    /// suppressed; the "aborting due to N errors" summary still counts the suppressed errors.
    pub error_limit: Option<usize>,
    /// Which backend to run/emit with. `--run` defaults to the interpreter (the reference oracle);
    /// native object/exe emission always uses Cranelift.
    pub backend: BackendKind,
    /// The instruction-set level native code is generated for (`--target-cpu`): `native` (the
    /// host's full feature set, the default) or an x86-64 micro-architecture level. Governs
    /// `--emit=obj|exe` and `--run --backend=native`; a level below x86-64-v3 also builds the MIR
    /// without the raw-AVX2 vector recipes (see [`lower_options_for`]).
    pub target_cpu: TargetCpu,
    /// Autodiff / training options (see [`GradOptions`]). Only consulted by `--emit=grad` and
    /// `--train`; inert for every other stage.
    pub grad: GradOptions,
    /// Strict floating-point mode for the whole program (`--fp=strict`; `false` is the default
    /// `--fp=fast`): `@strict_fp` on every function — no FMA contraction, no reassociated float
    /// reduction, no kernel recognizer. Threaded to [`wukong_mir_build::LowerOptions::strict_fp`].
    pub strict_fp: bool,
    /// `--remarks=dispatch`: print a `note` at every statement (or function) the MIR builder
    /// replaced with a recognized runtime kernel call, naming the kernel symbols. Pure observation —
    /// the MIR, and therefore every artifact and run, is identical with it on or off.
    pub remarks_dispatch: bool,
}

/// Options for the autodiff CLI surface: which loss function to differentiate, which of its buffer
/// parameters to differentiate with respect to, and (for `--train`) the optimizer loop settings.
#[derive(Clone, Debug)]
pub struct GradOptions {
    /// The function to differentiate (`--grad-of=<name>`). Defaults to `loss` when unset.
    pub of: Option<String>,
    /// Parameter indices to differentiate w.r.t. (`--grad-wrt=0,1`). Empty means *every* pointer
    /// (buffer) parameter — the gradient of an unread/output buffer is simply zero, so this is safe —
    /// except under `--train`, where the default excludes the **last** parameter (the scalar loss
    /// output) because `train_loop` rejects it as a trainable weight (see `grad_wrt`).
    pub wrt: Vec<usize>,
    /// Run a fwd→bwd→optimizer training loop instead of just emitting the backward (`--train`).
    pub train: bool,
    /// Number of training steps (`--train-steps=N`).
    pub train_steps: usize,
    /// Learning rate for the training loop (`--train-lr=<f>`).
    pub train_lr: f32,
    /// Decoupled weight decay for AdamW (`--train-weight-decay=<f>`).
    pub train_weight_decay: f32,
    /// Which optimizer the training loop uses (`--train-opt=sgd|adamw`).
    pub train_opt: TrainOpt,
    /// Seed for the deterministic buffer initialization the training loop uses (`--train-seed=<u64>`).
    pub train_seed: u64,
}

impl Default for GradOptions {
    fn default() -> GradOptions {
        GradOptions {
            of: None,
            wrt: Vec::new(),
            train: false,
            train_steps: 100,
            train_lr: 0.01,
            train_weight_decay: 0.0,
            train_opt: TrainOpt::Sgd,
            train_seed: 0x5EED_1234,
        }
    }
}

/// The optimizer the `--train` loop applies to the differentiated parameters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrainOpt {
    /// Plain SGD (`w -= lr·g`) — needs no kernel of its own.
    Sgd,
    /// Fused decoupled-weight-decay AdamW (the `wukong_autodiff::optim` kernel).
    AdamW,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            input: PathBuf::new(),
            output: None,
            emit: EmitStage::Exe,
            run: false,
            opt_level: 0,
            verify_each: false,
            lint_levels: LintLevels::default(),
            time_passes: false,
            time_trace: None,
            depfile: None,
            color: true,
            error_format: ErrorFormat::Human,
            error_limit: None,
            backend: BackendKind::Interp,
            target_cpu: TargetCpu::Native,
            grad: GradOptions::default(),
            strict_fp: false,
            remarks_dispatch: false,
        }
    }
}

/// Process exit codes used by the driver and CLI.
pub mod exit {
    pub const OK: i32 = 0;
    pub const COMPILE_ERROR: i32 = 1;
    pub const UNIMPLEMENTED: i32 = 2;
    pub const IO_ERROR: i32 = 3;
}

/// Environment variable that builds a program with **kernel recognition off**: every `match_*`
/// recognizer and every whole-function kernel interception in `wukong_mir_build` is skipped, so the
/// loop nest lowers as written (scalar, or autovectorized) instead of becoming a `wukong_*` runtime
/// call. Set to any value to enable, in the style of `WUKONG_GPU_NO_MEGA`.
///
/// **This is an instrument, not a user feature**, and deliberately not a CLI flag: it exists so that
/// a claim about the *language* can be measured against the recognizer-free lowering of the same
/// program, since recognized kernels are otherwise gate-blind (they fire identically on interp and
/// native at every `-O` level). A user who wants one function lowered as written writes `@generic` on
/// it — that is the supported, per-function surface. See `docs/internals.md`.
pub const NO_RECOGNIZE_ENV: &str = "WUKONG_NO_RECOGNIZE";

/// Is [`NO_RECOGNIZE_ENV`] set in this process's environment?
fn no_recognize() -> bool {
    std::env::var_os(NO_RECOGNIZE_ENV).is_some()
}

/// The `mir_build` options for this invocation.
///
/// Two switches, and they are decided *here* for the same reason: the MIR is built **once** and
/// handed to whichever backend was asked for, so anything that changes what the MIR contains has to
/// be a value chosen by the pipeline rather than an environment read buried in a recognizer.
///
/// * `recognize` - the `WUKONG_NO_RECOGNIZE` measurement instrument ([`NO_RECOGNIZE_ENV`]), read in
///   this function and nowhere else in the compiler.
/// * `vec_kernels` - the 256-bit raw-AVX2 recipes, which assemble Win64-ABI machine code into
///   `Op::VecKernelCall`. No PTX backend can lower that, and `wukong_codegen_gpu` declines it by
///   design and permanently - so on an AVX2 Win64 host the *CPU* vectorizer's choice was deciding
///   whether the *GPU* backend could run the program at all. `--backend=gpu-native` on any
///   unrecognized f32 elementwise loop (`tests/run/lang_parallel_saxpy.wk`,
///   `lang_generic_saxpy.wk`, `lang_generic_copy.wk` - the three flagship language fixtures) failed
///   with `UNSUPPORTED: VecKernelCall (CPU raw-AVX2 kernel) not lowered to PTX` unless the operator
///   happened to know to set `WUKONG_P4_NO_256=1`.
///
/// [`BackendKind::Gpu`] is deliberately NOT in the second rule: that path tree-walks the program in
/// the interpreter and offloads recognized calls only, so it wants the CPU recipes exactly as much
/// as `interp` and `native` do. Only *whole-program* lowering needs MIR a PTX backend can consume.
///
/// The `WUKONG_P4_NO_256` kill-switch is untouched and still read inside `mir_build`; it remains the
/// same-run A/B knob for measuring the 256-bit win on CPU, and the escape hatch if a recipe body is
/// ever found miscompiled. This option is the *other* conjunct of the same gate.
///
/// The same conjunct carries `--target-cpu`: the recipes are AVX2+FMA machine code with no runtime
/// dispatch, so a level below x86-64-v3 must not receive them — the program keeps the 128-bit CLIF
/// strips, which are the documented result-identical form, and its object runs on that level.
fn lower_options_for(opts: &Options) -> wukong_mir_build::LowerOptions {
    wukong_mir_build::LowerOptions {
        // Only a gradient request (`--emit=grad`, `--train`, or the ahead-of-time export
        // `--emit=obj --grad-of`) has an autodiff consumer for single-block float functions; every
        // other compile gets the compact copy loops (see `LowerOptions::unroll_for_autodiff`).
        unroll_for_autodiff: opts.emit == EmitStage::Grad
            || opts.grad.train
            || exports_gradient(opts),
        recognize: !no_recognize(),
        vec_kernels: !matches!(opts.backend, BackendKind::GpuLower)
            && opts.target_cpu.allows_vec_kernels(),
        strict_fp: opts.strict_fp,
    }
}

/// Run the pipeline for `opts`, printing artifacts to stdout and diagnostics to stderr.
/// Returns a process exit code.
pub fn compile(opts: &Options) -> i32 {
    // Autodiff over a recognition-free program is refused, loudly, before anything else runs: the
    // VJP rules in `wukong_autodiff` are keyed on the recognized kernel calls (`wukong_sgemm` &c),
    // so a taped program built without them meets buffer-writing loops it has no rule for. The
    // failure mode this prevents is the repo's standing rule for autodiff holes — a named refusal,
    // never a silently zero gradient.
    if no_recognize() && (opts.grad.train || opts.emit == EmitStage::Grad || exports_gradient(opts))
    {
        let what = if opts.grad.train {
            "--train"
        } else if opts.emit == EmitStage::Grad {
            "--emit=grad"
        } else {
            "--grad-of (with --emit=obj)"
        };
        eprintln!(
            "error: {what}: {NO_RECOGNIZE_ENV} is set, and autodiff is not supported with kernel \
             recognition off (the VJP rules are kernel-shaped). Unset {NO_RECOGNIZE_ENV}, or drop \
             {what}."
        );
        return exit::COMPILE_ERROR;
    }
    // `--fp=strict` turns recognition off for every function (a recognized kernel fixes its own
    // summation tree), so it meets the same autodiff limit as the environment switch above — for
    // every autodiff consumer, the ahead-of-time `--emit=obj --grad-of` export included.
    if opts.strict_fp && (opts.grad.train || opts.emit == EmitStage::Grad || exports_gradient(opts))
    {
        let what = if opts.grad.train {
            "--train"
        } else if opts.emit == EmitStage::Grad {
            "--emit=grad"
        } else {
            "--emit=obj --grad-of"
        };
        eprintln!(
            "error: {what}: `--fp=strict` turns kernel recognition off for the whole program, and \
             autodiff is not supported with recognition off (the VJP rules are kernel-shaped). \
             Drop `--fp=strict`; `@strict_fp` on individual functions leaves recognition (and \
             autodiff) intact everywhere else."
        );
        return exit::COMPILE_ERROR;
    }

    // Declared before every other local so it reports last, on every exit path (see `report`).
    let mut timer = report::PassTimer::new(opts);
    timer.begin("read");
    let src = match std::fs::read_to_string(&opts.input) {
        Ok(s) => s,
        Err(e) => {
            emit_driver_error(
                opts,
                format!("could not read `{}`: {}", opts.input.display(), e),
            );
            return exit::IO_ERROR;
        }
    };

    let mut sm = SourceMap::new();
    let id = sm.add(opts.input.display().to_string(), src);
    let renderer = Renderer::new(opts.color);
    let mut sink = DiagnosticSink::new();

    // --- Lexing ---
    timer.begin("lex");
    let (tokens, lex_diags) = wukong_lexer::tokenize(sm.source(id), id);
    for d in lex_diags {
        sink.emit(d);
    }

    if opts.emit == EmitStage::Tokens {
        render_all(opts, &renderer, &sink, &sm);
        print_artifact(&wukong_lexer::dump(&tokens, sm.source(id)));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // --- Parsing ---
    timer.begin("parse");
    let mut interner = Interner::new();
    // The root file's parser lints (`W0030`/`W0031`) wait for the lint stage below, where every
    // lint's level is resolved together; imported files are parsed by the loader without them.
    let wukong_parser::ParseOutput {
        mut module,
        diags: parse_diags,
        mut next_node,
        lints: parse_lints,
    } = wukong_parser::parse_module_tokens_with_lints(&tokens, sm.source(id), &mut interner, 0);
    // The AST owns its data; keep root tokens out of the import/sema/backend memory peak.
    drop(tokens);
    for d in parse_diags {
        sink.emit(d);
    }

    // `--emit=ast` (like `--emit=tokens` above) deliberately operates on the ROOT file only —
    // these stages inspect the lex/parse of the exact file you point the compiler at, so imports
    // are not loaded or spliced here. The merged multi-file program is what every later stage
    // (`--emit=mir-high` onward, `--run`) sees.
    if opts.emit == EmitStage::Ast {
        // The parser's lints print after the errors and before the summary, so the first
        // diagnostic is still the first error (the `tests/fail` contract).
        let errors = emit_diags(sink.diagnostics(), opts, &renderer, &sm);
        let denied = render_lints(parse_lints, &module, &interner, opts, &renderer, &sm);
        emit_summary(errors + denied, opts, &renderer);
        print_artifact(&wukong_ast::print::print_module(&module, &interner));
        return if sink.has_errors() || denied > 0 {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    if sink.has_errors() {
        let errors = emit_diags(sink.diagnostics(), opts, &renderer, &sm);
        let denied = render_lints(parse_lints, &module, &interner, opts, &renderer, &sm);
        emit_summary(errors + denied, opts, &renderer);
        return exit::COMPILE_ERROR;
    }

    // --- Import loading (the multi-file front end; see `loader`) ---
    // Each `import`ed file is resolved relative to the root source file's directory, loaded once
    // (cycles/diamonds dedup by canonical path), its names resolved against its own module scope
    // (`pub`, `as` aliases, `{x, y}` selection — imported items are renamed to unique
    // `<module>.<name>` symbols), and its items spliced into `module`, so everything downstream sees
    // one table of unique symbols. Lex/parse diagnostics from imported
    // files — and E0305 for an unresolvable import — land in the same sink and render with their
    // own file's path/line through the shared SourceMap.
    timer.begin("imports");
    loader::load_imports(
        &mut module,
        &opts.input,
        &mut sm,
        &mut interner,
        &mut sink,
        &mut next_node,
    );
    let import_errors = emit_diags(sink.diagnostics(), opts, &renderer, &sm);
    if sink.has_errors() {
        let denied = render_lints(parse_lints, &module, &interner, opts, &renderer, &sm);
        emit_summary(import_errors + denied, opts, &renderer);
        return exit::COMPILE_ERROR;
    }
    drop(sink);
    // Every file the compile reads is in the SourceMap by now: the root and each import, once.
    if let Some(depfile) = &opts.depfile {
        let target = report::depfile_target(opts, depfile);
        if let Err(e) = std::fs::write(depfile, report::depfile_contents(&target, &sm)) {
            eprintln!("error: could not write `{}`: {e}", depfile.display());
            return exit::IO_ERROR;
        }
    }

    // --- Semantic analysis (name resolution, type checking, shape checking) ---
    timer.begin("sema");
    let (sema, sema_diags) = wukong_sema::check(&module, &interner);
    let sema_errors = emit_diags(&sema_diags, opts, &renderer, &sm);
    if sema_errors > 0 {
        // The parser's lints still print — after the errors — when sema fails: a `(` that
        // continued the line above (`W0031`) is often why a call does not type-check.
        let denied = render_lints(parse_lints, &module, &interner, opts, &renderer, &sm);
        emit_summary(sema_errors + denied, opts, &renderer);
        return exit::COMPILE_ERROR;
    }
    drop(sema_diags);

    // --- Lints (`W0xxx`; see `wukong_sema::lints`) ---
    // Only a program that checks clean is linted. Levels (`-A/-W/-D`, `@allow/@warn/@deny`) are
    // resolved now, while the AST is alive, and a denied lint stops the build here as an error.
    // Warnings are held back and printed after lowering's diagnostics, so a failed build always
    // leads with an error — the contract `tests/fail` pins on the first diagnostic.
    let mut found = parse_lints;
    found.extend(wukong_sema::lints::run(
        &module,
        &sema,
        &interner,
        &lint_options_for(opts),
    ));
    let lint_diags = wukong_sema::lints::apply_levels(found, &module, &interner, &opts.lint_levels);
    if lint_diags.iter().any(Diagnostic::is_error) {
        let (denied, warned): (Vec<&Diagnostic>, Vec<&Diagnostic>) =
            lint_diags.iter().partition(|d| d.is_error());
        let ordered: Vec<Diagnostic> = denied.into_iter().chain(warned).cloned().collect();
        let errors = emit_diags(&ordered, opts, &renderer, &sm);
        emit_summary(errors, opts, &renderer);
        return exit::COMPILE_ERROR;
    }

    // `--emit=check`: lex, parse, import resolution, sema and the lints have all run and reported
    // clean (any error would have returned above). Print the lint warnings and stop before MIR
    // lowering — the fast typo loop an editor wants, with no C0001 lowering limits in the way.
    if opts.emit == EmitStage::Check {
        emit_diags(&lint_diags, opts, &renderer, &sm);
        return exit::OK;
    }

    // For `--train`, recover the loss function's buffer element counts from the semantic types now,
    // while the `[f32; N]` / `Tensor[..]` annotations are still available (they are lost once the
    // params lower to `Ptr`).
    let train_lens = if opts.grad.train {
        match loss_param_lens(&sema, &mut interner, opts) {
            Ok(l) => Some(l),
            Err(e) => {
                emit_driver_error(opts, format!("--train: {e}"));
                return exit::COMPILE_ERROR;
            }
        }
    } else {
        None
    };

    // --- MIR construction ---
    // Both switches that change *what the MIR contains* are values chosen by the pipeline, not
    // environment reads inside a recognizer; `lower_options_for` is where that choice lives and why
    // the requested backend is part of it.
    timer.begin("lower");
    let lower_opts = lower_options_for(opts);
    // Runtime safety failures (an out-of-bounds slice index) report where they happened. The
    // location is rendered here, where the source map lives, and baked into `.rodata` by the
    // lowering, so the interpreter, the JIT and a linked executable all print the same text.
    let locate = |span: wukong_span::Span| -> String {
        let id = span.source;
        if span.is_dummy()
            || id.0 as usize >= sm.file_count()
            || span.lo as usize > sm.source(id).len()
        {
            return String::new();
        }
        let loc = sm.span_location(span);
        format!("{}:{}:{}", sm.name(id), loc.line, loc.col)
    };
    // `--remarks=dispatch` only adds `note` diagnostics; the Program is the same either way.
    let (mut program, lower_diags) = if opts.remarks_dispatch {
        wukong_mir_build::lower_program_reporting_dispatch_located(
            &module,
            &sema,
            &mut interner,
            lower_opts,
            &locate,
        )
    } else {
        wukong_mir_build::lower_program_located(&module, &sema, &mut interner, lower_opts, &locate)
    };
    // MIR and train_lens own everything needed downstream. Release completed frontend
    // artifacts before optimization and codegen allocate their working sets.
    drop(module);
    drop(sema);
    let lower_errors = emit_diags(&lower_diags, opts, &renderer, &sm);
    // The held-back lint warnings, after any lowering error (see the lint stage above) and before
    // the "aborting due to" summary below.
    emit_diags(&lint_diags, opts, &renderer, &sm);
    drop(lint_diags);
    drop(sm);
    let lowering_failed = lower_errors > 0;
    drop(lower_diags);

    // High MIR is the pre-optimization form. It is still dumped for debugging even when lowering
    // failed, since seeing the partial MIR is useful.
    if opts.emit == EmitStage::MirHigh {
        // Dump the (possibly partial) MIR, then fail the process if lowering errored or the verifier
        // reported an internal-compiler-error — a printed ICE must never report success.
        let verify_failed = emit_mir(&program, &interner);
        return if verify_failed || lowering_failed {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // A construct codegen could not lower leaves the MIR invalid, so do not optimize, run, or hand
    // it to a backend — that would crash or miscompile. Stop with a clear error instead.
    if lowering_failed {
        emit_summary(lower_errors, opts, &renderer);
        return exit::COMPILE_ERROR;
    }

    // Resolve selected autodiff entry points before optimization can remove unreferenced callees.
    let roots = match autodiff_roots(&program, &mut interner, opts) {
        Ok(roots) => roots,
        Err(message) => {
            emit_driver_error(opts, format!("autodiff: {message}"));
            return exit::COMPILE_ERROR;
        }
    };

    // --- Reachability pruning ---
    // A program with one known entry point compiles only what `main` can reach; an object file
    // exports every function and the MIR dumps show the whole program, so both keep everything,
    // as do the autodiff stages (their entry point is the loss function, not `main`).
    // `--backend=gpu-native` also keeps everything: its whole-program PTX lowering declines a
    // program on any unsupported function, and pruning would silently change which corpus programs
    // the device suite attempts.
    let single_entry = opts.run || opts.emit == EmitStage::Exe;
    if single_entry && !opts.grad.train && opts.backend != BackendKind::GpuLower {
        let main = interner.intern("main");
        let pruned = prune::retain_reachable(&mut program, &[main]);
        if pruned > 0 {
            timer.note(|| format!("pruned {pruned} function(s) unreachable from `main`"));
        }
    }

    // --- Optimization ---
    // The autodiff transform (`--emit=grad` / `--train` / an exported gradient) consumes single-block
    // SSA, so it forces at least `-O1` (mem2reg + simplify-cfg) regardless of the requested level.
    // Every other path honors the requested level exactly.
    let opt_level = if opts.emit == EmitStage::Grad || opts.grad.train || exports_gradient(opts) {
        opts.opt_level.max(1)
    } else {
        opts.opt_level
    };
    timer.begin("optimize");
    if opts.verify_each {
        if let Err(failure) =
            wukong_opt::optimize_verified_preserving(&mut program, opt_level, &roots)
        {
            emit_diag(
                &optimization_failure_diag(&failure, &interner),
                opts.error_format,
                &renderer,
                &SourceMap::new(),
            );
            return exit::COMPILE_ERROR;
        }
    } else if timer.enabled() && roots.is_empty() {
        // The timed optimizer runs the identical pipeline (byte-identical MIR) with a clock around
        // each pass; with no extra roots it is exactly `optimize_preserving(.., &[])`.
        let timings = wukong_opt::optimize_timed(&mut program, opt_level);
        timer.optimizer_passes(&timings);
    } else {
        wukong_opt::optimize_preserving(&mut program, opt_level, &roots);
    }
    timer.mir_summary(&program, &interner);

    // --- Autodiff CLI surface (additive; runs after optimization, before any backend) ---
    if opts.grad.train {
        timer.begin("train");
        let lens = train_lens.expect("train_lens computed when --train is set");
        return run_train(&program, &mut interner, opts, &lens);
    }
    if opts.emit == EmitStage::Grad {
        timer.begin("autodiff");
        return emit_grad(&program, &mut interner, opts);
    }
    if exports_gradient(opts) {
        timer.begin("autodiff");
        if let Err(e) = append_gradient(&mut program, &mut interner, opts) {
            eprintln!("error: --emit=obj --grad-of: {e}");
            return exit::COMPILE_ERROR;
        }
    }

    // --- Run via the selected backend (interpreter by default, Cranelift JIT with --backend=native) ---
    if opts.run {
        // Gate the backend on a clean MIR verify. The optimizer verifies the forms its passes
        // produce, but at -O0 it runs no passes, so an invalid-MIR lowering bug would otherwise reach
        // the backend unchecked — and the native -O0 JIT codegens it into a SIGSEGV rather than the
        // clean ICE that `--emit=mir` and the optimizer already report. Verifying here makes a
        // lowering bug a diagnosable internal-compiler-error instead of a crash or miscompile.
        timer.begin("verify");
        if verify_or_ice(&program) {
            return exit::COMPILE_ERROR;
        }
        // The interpreter (and the GPU offload path, which runs on it) calls an `extern "C"`
        // function through the same host symbol the JIT binds, but only for a scalar signature whose
        // symbol resolves. Refuse anything else up front as a diagnostic, not midway through the
        // program's output.
        if matches!(opts.backend, BackendKind::Interp | BackendKind::Gpu) {
            let refused = wukong_interp::unsupported_externs(&program, &interner);
            if !refused.is_empty() {
                for msg in refused {
                    emit_diag(
                        &Diagnostic::error(msg).with_code("C0001"),
                        opts.error_format,
                        &renderer,
                        &SourceMap::new(),
                    );
                }
                return exit::COMPILE_ERROR;
            }
        }
        let main = interner.intern("main");
        timer.begin(if opts.backend == BackendKind::Native {
            "jit-compile"
        } else {
            "run"
        });
        // The CPU paths report what the program printed *before* a runtime failure (a failed
        // `assert`, a slice index out of bounds) along with the failure, so `--run` shows the same
        // stdout a linked executable does — the output is printed, then `error: …`, exit 1.
        let split = |r: Result<(i64, Vec<u8>), String>| match r {
            Ok((code, out)) => (Ok(code), out),
            Err(e) => (Err(e), Vec::new()),
        };
        let (result, stdout) = match opts.backend {
            BackendKind::Interp => wukong_interp::run_capturing(&program, main, &interner),
            // Compile and run are separate steps so a *compile* failure is rendered as a
            // catalogued diagnostic (honouring `--error-format=json`), while a *run-time* failure
            // (an assertion, a bounds check) keeps the `error: <message>` form below, after the
            // output printed before it. The JIT honours both `--target-cpu` and the source `-O`
            // level (`-O0` = Cranelift `opt_level=none`).
            BackendKind::Native => {
                let backend = wukong_codegen_cranelift::BackendOpt::for_level(opt_level);
                match wukong_codegen_cranelift::jit_compile_for_with(
                    &program,
                    main,
                    &interner,
                    opts.target_cpu,
                    backend,
                ) {
                    Ok(compiled) => {
                        timer.begin("run");
                        compiled.run_capturing()
                    }
                    Err(message) => {
                        emit_diag(
                            &native_failure_diag(&message, &opts.input),
                            opts.error_format,
                            &renderer,
                            &SourceMap::new(),
                        );
                        return exit::COMPILE_ERROR;
                    }
                }
            }
            BackendKind::Gpu => split(run_on_gpu(&program, main, &interner)),
            BackendKind::GpuLower => split(run_on_gpu_lower(&program, main, &interner)),
        };
        {
            use std::io::Write;
            let mut out = std::io::stdout();
            let _ = out.write_all(&stdout);
            let _ = out.flush();
        }
        return match result {
            Ok(exit_code) => exit_code as i32,
            Err(e) => {
                emit_driver_error(opts, e);
                exit::COMPILE_ERROR
            }
        };
    }

    if opts.emit == EmitStage::Mir {
        timer.begin("emit-mir");
        // A verifier failure means we just printed an internal-compiler-error; never report success.
        return if emit_mir(&program, &interner) {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // The AOT exits get the same gate as `--run` above. They used to have none: the verify loop was
    // lexically inside the `if opts.run` block, and `wukong_opt`'s per-pass `verify-each` is
    // `#[cfg(debug_assertions)]`, so a release `wukongc` handed unverified MIR to `emit_llvm_ir` and
    // to Cranelift — the one path where an invalid lowering became a raw backend assertion or a
    // silently miscompiled artifact instead of a diagnosable ICE. (`--emit=mir`/`mir-high` are
    // deliberately *not* gated here: they verify inside `emit_mir` after dumping, so a broken program
    // still prints the MIR that shows why.)
    timer.begin("verify");
    if matches!(
        opts.emit,
        EmitStage::LlvmIr | EmitStage::Obj | EmitStage::Exe
    ) && verify_or_ice(&program)
    {
        return exit::COMPILE_ERROR;
    }

    // --- LLVM backend ---
    if opts.emit == EmitStage::LlvmIr {
        timer.begin("llvm-ir");
        return match wukong_codegen_llvm::try_emit_llvm_ir(&program, &interner) {
            Ok(ir) => {
                print_artifact(&ir);
                exit::OK
            }
            Err(message) => {
                emit_diag(
                    &Diagnostic::error(message).with_code("C0001"),
                    opts.error_format,
                    &renderer,
                    // Backend errors have no source labels; frontend sources were released.
                    &SourceMap::new(),
                );
                exit::UNIMPLEMENTED
            }
        };
    }
    if matches!(opts.emit, EmitStage::Obj | EmitStage::Exe) {
        return emit_native(&program, &interner, opts, &mut timer);
    }

    exit::OK
}

/// Execute `entry` on the GPU backend: the interpreter tree-walks the program while recognized kernel
/// calls run on the device (see [`gpu_accel`]). Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu(
    program: &wukong_mir::Program,
    entry: wukong_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let mut guard = wukong_codegen_gpu::gpu();
    match guard.as_mut() {
        Some(g) => {
            let mut accel = gpu_accel::GpuAccel::new(g);
            wukong_interp::run_with_output_accel(program, entry, interner, &mut accel)
        }
        // The driver library's file name is per-OS (`nvcuda.dll` on Windows, `libcuda.so.1` on
        // Linux); it comes from `wukong_codegen_gpu`'s one table so this message and the
        // `--backend=gpu-native` twin in `lower.rs` cannot drift apart.
        None => Err(format!(
            "`--backend=gpu` requires a CUDA device, but none was reachable (the driver \
             dlopens `{}`; check the NVIDIA driver is installed)",
            wukong_codegen_gpu::baselines::cuda_driver_lib_name()
        )),
    }
}

/// Without the `gpu` feature, `--backend=gpu` is a clear build-time-capability error rather than a
/// silent fallback (which would dishonestly run on the CPU when the user asked for the GPU).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu(
    _program: &wukong_mir::Program,
    _entry: wukong_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `wukongc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p wukongc --features gpu`) to use `--backend=gpu`"
        .into())
}

/// Execute `entry` on the **general MIR→PTX** GPU backend (`--backend=gpu-native`, Phase 4): the
/// whole program is lowered to PTX and run on the device — distinct from the recognizer offload
/// ([`run_on_gpu`]), which it leaves untouched. Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu_lower(
    program: &wukong_mir::Program,
    entry: wukong_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    wukong_codegen_gpu::lower::jit_run(program, entry, interner)
}

/// Without the `gpu` feature, `--backend=gpu-native` is a clear build-capability error rather than a
/// silent CPU fallback (mirrors [`run_on_gpu`]).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu_lower(
    _program: &wukong_mir::Program,
    _entry: wukong_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `wukongc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p wukongc --features gpu`) to use `--backend=gpu-native`"
        .into())
}

/// The C runtime linked into native executables by the **`cc` fallback** path: it backs the
/// `print`/`assert`/`fmod` and heap intrinsics that the Cranelift object leaves as undefined imports.
/// The JIT path binds those same symbols to Rust functions, and the preferred rustc link to
/// [`WUKONG_RT_SHIM`]; all three must print **byte-identical** output.
///
/// That includes floats. The oracle prints an `f64` with Rust's `Display`: the shortest decimal
/// digit string that round-trips (the closest one when several do), written positionally — never in
/// exponent form — with `inf`, `-inf`, `NaN` and `-0` spelled out. C has no such conversion
/// (`%g` keeps six significant digits), so `wukong_rt_print_f64` derives it: for each precision `p`
/// from 1 to 17 it takes the correctly rounded `p`-digit `%e` string and accepts it if `strtod`
/// round-trips it to `x`, and otherwise tries the two adjacent `p`-digit decimals (a double at a
/// power-of-two boundary has a lopsided rounding interval, so the nearest `p`-digit decimal can fall
/// outside it while its neighbour lands inside). The first hit is the shortest round-tripping
/// string, and among those the closest, which is exactly what Rust prints; the digits are then laid
/// out positionally. `__USE_MINGW_ANSI_STDIO` selects MinGW's correctly rounded printf family; glibc,
/// the UCRT and macOS libc are correctly rounded already.
///
/// The fallback still provides no `wukong_*` runtime kernels (a program that calls one fails to link,
/// loudly), and on a Windows host the MinGW linker cannot consume an MSVC object's data relocations.
///
/// It also carries the slice bounds-failure report (`wukong_rt_bounds_fail`, byte-identical to
/// `wukong_runtime::bounds_failure_message`) and the runtime's 64-byte heap alignment
/// (`HEAP_ALIGN`), so a linked executable fails and aligns exactly as `--run` does.
const WUKONG_RT_C: &str = r#"#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
/* Keep stdout in binary mode so the MSVCRT does not translate the runtime's `\n` into CRLF. The
   interpreter oracle (and the JIT) emit LF; without this the *linked exe* printed CRLF on Windows,
   so its stdout differed from every other backend. Runs before main via the constructor attribute. */
__attribute__((constructor)) static void wukong_rt_init(void) { _setmode(_fileno(stdout), _O_BINARY); }
#endif
void wukong_rt_print_i64(long long x) { printf("%lld\n", x); }
void wukong_rt_print_u64(unsigned long long x) { printf("%llu\n", x); }
/* Split a "d.ddde+XX" string into its digit characters and decimal exponent. */
static int wk_parse_sci(const char *s, char *digits, int *n, int *dexp) {
    int k = 0;
    for (; *s && *s != 'e' && *s != 'E'; s++) {
        if (*s >= '0' && *s <= '9') {
            if (k >= 24) return 0;
            digits[k++] = *s;
        }
    }
    if (*s != 'e' && *s != 'E') return 0;
    *dexp = atoi(s + 1);
    *n = k;
    return k > 0;
}
/* Does digits * 10^dexp (one digit before the point) parse back to exactly x — as an f32 (strtof,
   correctly rounded to float) when is_f32, else as an f64? */
static int wk_round_trips(const char *digits, int n, int dexp, double x, int is_f32) {
    char buf[64];
    int k = 0;
    buf[k++] = digits[0];
    if (n > 1) {
        buf[k++] = '.';
        memcpy(buf + k, digits + 1, (size_t)(n - 1));
        k += n - 1;
    }
    snprintf(buf + k, sizeof buf - (size_t)k, "e%d", dexp);
    if (is_f32) return strtof(buf, NULL) == (float)x;
    return strtod(buf, NULL) == x;
}
/* Step digits * 10^dexp to the adjacent n-digit decimal above (up) or below it. */
static void wk_step(char *d, int n, int *dexp, int up) {
    int i = n - 1, j, is_pow10 = 1;
    if (up) {
        while (i >= 0 && d[i] == '9') { d[i] = '0'; i--; }
        if (i < 0) { d[0] = '1'; *dexp += 1; } else { d[i]++; }
        return;
    }
    if (d[0] != '1') is_pow10 = 0;
    for (j = 1; j < n && is_pow10; j++) if (d[j] != '0') is_pow10 = 0;
    if (is_pow10) {
        /* 10^e steps down onto the finer grid of the decade below: 99..9 * 10^(e-1). */
        for (j = 0; j < n; j++) d[j] = '9';
        *dexp -= 1;
        return;
    }
    while (d[i] == '0') { d[i] = '9'; i--; }
    d[i]--;
}
/* Rust's `Display` for f64, or for f32 when is_f32 (x then holds the f32 exactly): the shortest
   digits that round-trip at that precision (see the driver's WUKONG_RT_C docs). 17 digits always
   round-trip an f64 and 9 always round-trip an f32. */
static void wk_print_float(double x, int is_f32) {
    char out[400];
    char d[24], c[24];
    int max_p = is_f32 ? 9 : 17;
    int k = 0, n = 0, e = 0, ce, p, up, found = 0, point, z;
    if (x != x) { fputs("NaN\n", stdout); return; }
    if (signbit(x)) { out[k++] = '-'; x = -x; }
    if (isinf(x)) {
        memcpy(out + k, "inf", 3);
        k += 3;
    } else if (x == 0.0) {
        out[k++] = '0';
    } else {
        for (p = 1; p <= max_p && !found; p++) {
            char s[64];
            snprintf(s, sizeof s, "%.*e", p - 1, x);
            if (!wk_parse_sci(s, d, &n, &e)) continue;
            if (wk_round_trips(d, n, e, x, is_f32)) { found = 1; break; }
            for (up = 1; up >= 0 && !found; up--) {
                memcpy(c, d, (size_t)n);
                ce = e;
                wk_step(c, n, &ce, up);
                if (wk_round_trips(c, n, ce, x, is_f32)) { memcpy(d, c, (size_t)n); e = ce; found = 1; }
            }
        }
        if (!found) {
            /* Unreachable with a correctly rounded printf: max_p digits always round-trip. */
            char s[64];
            snprintf(s, sizeof s, "%.*e", max_p - 1, x);
            if (!wk_parse_sci(s, d, &n, &e)) { d[0] = '0'; n = 1; e = 0; }
        }
        while (n > 1 && d[n - 1] == '0') n--;
        point = e + 1; /* the value is 0.d1d2...dn * 10^point */
        if (point <= 0) {
            out[k++] = '0';
            out[k++] = '.';
            for (z = 0; z < -point; z++) out[k++] = '0';
            memcpy(out + k, d, (size_t)n);
            k += n;
        } else if (point >= n) {
            memcpy(out + k, d, (size_t)n);
            k += n;
            for (z = 0; z < point - n; z++) out[k++] = '0';
        } else {
            memcpy(out + k, d, (size_t)point);
            k += point;
            out[k++] = '.';
            memcpy(out + k, d + point, (size_t)(n - point));
            k += n - point;
        }
    }
    out[k++] = '\n';
    fwrite(out, 1, (size_t)k, stdout);
}
void wukong_rt_print_f64(double x) { wk_print_float(x, 0); }
/* An f32 (or an f16/bf16 held in an f32 register) at its own precision: Rust's `Display` for f32. */
void wukong_rt_print_f32(float x) { wk_print_float((double)x, 1); }
void wukong_rt_print_bool(long long x) { fputs(x ? "true\n" : "false\n", stdout); }
/* A char's glyph as UTF-8 plus a newline; a value that is not a Unicode scalar value (a surrogate,
   or above U+10FFFF) prints U+FFFD, as `wukong_runtime::fmt::char_glyph` does. */
void wukong_rt_print_char(long long x) {
    unsigned long c = (unsigned long)(x & 0xffffffffLL);
    unsigned char b[5];
    int n = 0;
    if (c > 0x10FFFFUL || (c >= 0xD800UL && c <= 0xDFFFUL)) c = 0xFFFDUL;
    if (c < 0x80UL) {
        b[n++] = (unsigned char)c;
    } else if (c < 0x800UL) {
        b[n++] = (unsigned char)(0xC0 | (c >> 6));
        b[n++] = (unsigned char)(0x80 | (c & 0x3F));
    } else if (c < 0x10000UL) {
        b[n++] = (unsigned char)(0xE0 | (c >> 12));
        b[n++] = (unsigned char)(0x80 | ((c >> 6) & 0x3F));
        b[n++] = (unsigned char)(0x80 | (c & 0x3F));
    } else {
        b[n++] = (unsigned char)(0xF0 | (c >> 18));
        b[n++] = (unsigned char)(0x80 | ((c >> 12) & 0x3F));
        b[n++] = (unsigned char)(0x80 | ((c >> 6) & 0x3F));
        b[n++] = (unsigned char)(0x80 | (c & 0x3F));
    }
    b[n++] = '\n';
    fwrite(b, 1, (size_t)n, stdout);
}
/* A NUL-terminated string buffer plus a newline; a null pointer prints nothing (as the Rust shim). */
void wukong_rt_print_str(const char *s) {
    if (!s) return;
    fputs(s, stdout);
    fputc('\n', stdout);
}
void wukong_rt_assert(long long c) { if (!c) { fprintf(stderr, "assertion failed\n"); exit(1); } }
long long wukong_rt_assert_failed(void) { return 0; }
/* A failed slice bounds check. Mirrors `wukong_runtime::bounds_failure_message` byte for byte, and
   exits with the status `wukongc --run` reports for a runtime failure. */
void wukong_rt_bounds_fail(long long index, long long len, const char* site) {
    fflush(stdout);
    if (site && site[0]) fprintf(stderr, "error: index out of bounds: the len is %lld but the index is %lld at %s\n", len, index, site);
    else fprintf(stderr, "error: index out of bounds: the len is %lld but the index is %lld\n", len, index);
    exit(1);
}
double wukong_rt_fmod_f64(double a, double b) { return fmod(a, b); }
float wukong_rt_fmod_f32(float a, float b) { return fmodf(a, b); }
/* Heap builtins (alloc_<T>/free). calloc zero-initializes — the determinism contract. The block
   is over-allocated so the data pointer sits on a 64-byte boundary, the runtime's `HEAP_ALIGN`,
   with the block's own address stored just below it for free. An empty or overflowing request is
   null, as in the runtime. */
void* wukong_rt_alloc(long long count, long long elem_size, long long elem_is_float) {
    (void)elem_is_float;
    if (count <= 0 || elem_size <= 0) return 0;
    if ((unsigned long long)count > ((size_t)-1 - 128) / (unsigned long long)elem_size) return 0;
    char* raw = (char*)calloc(1, (size_t)count * (size_t)elem_size + 64 + sizeof(void*));
    if (!raw) return 0;
    char* data = (char*)(((uintptr_t)raw + sizeof(void*) + 63) & ~(uintptr_t)63);
    ((void**)data)[-1] = raw;
    return data;
}
void wukong_rt_free(void* p) { if (p) free(((void**)p)[-1]); }
"#;

/// The `cc` fallback's scratch allocator (`scratch_<T>` / `scratch_mark` / `scratch_release`), a C
/// mirror of `wukong_runtime::scratch`: zeroed 64-byte-aligned bump allocations in chunks that never
/// move, each at least twice the largest before it, released LIFO to a mark that counts live
/// allocations. One arena, because this fallback runtime has no threads.
const WUKONG_SCRATCH_C: &str = r#"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { char* raw; char* base; size_t cap; } wk_scratch_chunk;
typedef struct { size_t cur, off; } wk_scratch_mark;
static wk_scratch_chunk* wk_sc_chunks;
static size_t wk_sc_n, wk_sc_cur, wk_sc_off;
static wk_scratch_mark* wk_sc_marks;
static size_t wk_sc_nmarks, wk_sc_capmarks;
void* wukong_rt_scratch_alloc(long long count, long long elem_size, long long elem_is_float) {
    (void)elem_is_float;
    if (wk_sc_nmarks == wk_sc_capmarks) {
        size_t cap = wk_sc_capmarks ? 2 * wk_sc_capmarks : 64;
        wk_scratch_mark* m = (wk_scratch_mark*)realloc(wk_sc_marks, cap * sizeof(wk_scratch_mark));
        if (!m) abort();
        wk_sc_marks = m;
        wk_sc_capmarks = cap;
    }
    wk_sc_marks[wk_sc_nmarks].cur = wk_sc_cur;
    wk_sc_marks[wk_sc_nmarks].off = wk_sc_off;
    wk_sc_nmarks++;
    if (count <= 0 || elem_size <= 0) return 0;
    if ((unsigned long long)count > (unsigned long long)(PTRDIFF_MAX / 2) / (unsigned long long)elem_size) return 0;
    size_t bytes = (size_t)count * (size_t)elem_size;
    for (;;) {
        if (wk_sc_cur < wk_sc_n) {
            wk_scratch_chunk* c = &wk_sc_chunks[wk_sc_cur];
            size_t start = (wk_sc_off + 63) & ~(size_t)63;
            if (start <= c->cap && bytes <= c->cap - start) {
                wk_sc_off = start + bytes;
                memset(c->base + start, 0, bytes);
                return c->base + start;
            }
            if (wk_sc_cur + 1 < wk_sc_n && wk_sc_chunks[wk_sc_cur + 1].cap >= bytes) {
                wk_sc_cur++;
                wk_sc_off = 0;
                continue;
            }
        }
        size_t grown = 0;
        for (size_t i = 0; i < wk_sc_n; i++)
            if (wk_sc_chunks[i].cap > grown) grown = wk_sc_chunks[i].cap;
        size_t cap = bytes;
        if (cap < ((size_t)1 << 20)) cap = (size_t)1 << 20;
        if (cap < 2 * grown) cap = 2 * grown;
        cap = (cap + 63) & ~(size_t)63;
        char* raw = (char*)malloc(cap + 64);
        if (!raw) return 0;
        size_t keep = wk_sc_n ? wk_sc_cur + 1 : 0;
        for (size_t i = keep; i < wk_sc_n; i++) free(wk_sc_chunks[i].raw);
        wk_sc_n = keep;
        wk_scratch_chunk* cs = (wk_scratch_chunk*)realloc(wk_sc_chunks, (keep + 1) * sizeof(wk_scratch_chunk));
        if (!cs) { free(raw); return 0; }
        wk_sc_chunks = cs;
        wk_sc_chunks[keep].raw = raw;
        wk_sc_chunks[keep].base = (char*)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
        wk_sc_chunks[keep].cap = cap;
        wk_sc_n = keep + 1;
        wk_sc_cur = keep;
        wk_sc_off = 0;
    }
}
long long wukong_rt_scratch_mark(void) { return (long long)wk_sc_nmarks; }
void wukong_rt_scratch_release(long long mark) {
    size_t m = mark < 0 ? 0 : (size_t)mark;
    if (m >= wk_sc_nmarks) return;
    wk_sc_cur = wk_sc_marks[m].cur;
    wk_sc_off = wk_sc_marks[m].off;
    wk_sc_nmarks = m;
}
"#;

// Scalar-only C link fallback. Keep operations/constants in lockstep with runtime/vmath.rs,
// including explicit fmaf and nonfinite bits; plain sinf/cosf for every input would differ
// from the bounded polynomial used by the interpreter. cc_link_args disables contraction
// of the deliberately separate multiplies/adds. Large inputs use the platform libm in both.
const WUKONG_TRIG_C: &str = r#"
#include <stdint.h>
#include <string.h>
float wukong_rt_trig_f32(float x, long long op) {
    if (op != 15 && op != 16) return x;
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) >= UINT32_C(0x7f800000)) {
        bits = (bits & UINT32_C(0x007fffff))
            ? bits | UINT32_C(0x00400000) : UINT32_C(0x7fc00000);
        memcpy(&x, &bits, sizeof(x));
        return x;
    }
    if (fabsf(x) >= 4096.0f) return op == 16 ? cosf(x) : sinf(x);
    if (x == 0.0f) return op == 16 ? 1.0f : x;
    float tt = fmaf(x, 0.63661977236758134308f, 12582912.0f);
    float qf = tt - 12582912.0f;
    float r = fmaf(qf, -1.5703125f, x);
    r = fmaf(qf, -4.837512969970703e-4f, r);
    r = fmaf(qf, -7.54978995489188e-8f, r);
    float z = r * r;
    float s = fmaf(-1.9515295891e-4f, z, 8.3321608736e-3f);
    s = fmaf(s, z, -1.6666654611e-1f);
    float sz = s * z;
    float sin_p = fmaf(sz, r, r);
    float c = fmaf(2.443315711809948e-5f, z, -1.388731625493765e-3f);
    c = fmaf(c, z, 4.166664568298827e-2f);
    float z2 = z * z;
    float cz2 = c * z2;
    float hz = fmaf(-0.5f, z, 1.0f);
    float cos_p = hz + cz2;
    int quad = ((int32_t)qf) & 3;
    if (op == 16) {
        if (quad == 0) return cos_p;
        if (quad == 1) return sin_p * -1.0f;
        if (quad == 2) return cos_p * -1.0f;
        return sin_p;
    }
    if (quad == 0) return sin_p;
    if (quad == 1) return cos_p;
    if (quad == 2) return sin_p * -1.0f;
    return cos_p * -1.0f;
}
"#;

// The f64 transcendental ABI (`wukong_rt_math_f64(x, op)` / `wukong_rt_math2_f64(a, b, op)`) for the
// C fallback link only. The op codes are `wukong_runtime`'s `VM_*` / `VM2_*` (the compiler emits them
// as constants). This uses the host libm, so unlike the rustc-driven link — which resolves the
// symbols out of the runtime rlib and is therefore bit-identical to the oracle — a C-fallback exe
// may differ from `--run` in the last ULP; the fallback already cannot print floats at all, so this
// is not a new divergence class. Unknown op codes are the identity, matching the Rust entry.
const WUKONG_MATH64_C: &str = r#"
double wukong_rt_math_f64(double x, long long op) {
    switch (op) {
        case 0: return exp(x);
        case 1: return log(x);
        case 2: return tanh(x);
        case 15: return sin(x);
        case 16: return cos(x);
        case 17: return erf(x);
        case 18: return exp2(x);
        case 19: return log2(x);
        case 20: return sinh(x);
        case 21: return cosh(x);
        case 22: return asinh(x);
        case 23: return acosh(x);
        case 24: return atanh(x);
        case 25: return atan(x);
        case 26: return expm1(x);
        case 27: return log1p(x);
        case 28: return pow(10.0, x);
        case 29: return log10(x);
        case 32: return tan(x);
        case 33: return asin(x);
        case 34: return acos(x);
        case 35: return cbrt(x);
        default: return x;
    }
}
double wukong_rt_math2_f64(double a, double b, long long op) {
    switch (op) {
        case 0: return pow(a, b);
        case 1: return atan2(a, b);
        case 2: return hypot(a, b);
        default: return a;
    }
}
"#;

fn c_runtime_source() -> String {
    format!("{WUKONG_RT_C}\n{WUKONG_TRIG_C}\n{WUKONG_SCRATCH_C}\n{WUKONG_MATH64_C}")
}

/// A per-invocation scratch directory for the generated `--emit=exe` link inputs (the Rust link
/// shim and the C fallback runtime). They used to be written into the process's *current* directory
/// as `{stem}_shim.rs` / `{stem}_rt.c`, where the shim overwrote — and then unconditionally deleted
/// — any user file of that name, and two concurrent links on the same stem raced on it. Keyed by
/// pid so concurrent invocations cannot collide; removed when the link finishes either way.
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// `None` when the directory cannot be created — the caller then falls back to the old
    /// working-directory paths rather than failing the link outright.
    fn new(stem: &str) -> Option<ScratchDir> {
        let path = std::env::temp_dir().join(format!("wukongc-{}-{stem}", std::process::id()));
        std::fs::create_dir_all(&path).ok()?;
        Some(ScratchDir { path })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Where a generated link input goes: inside the scratch directory when one could be created, else
/// the working directory (the historical behaviour).
fn scratch_path(scratch: Option<&ScratchDir>, name: String) -> PathBuf {
    match scratch {
        Some(s) => s.path.join(name),
        None => PathBuf::from(name),
    }
}

/// Where the Cranelift object goes. **One decision, one function**, so the `--emit=obj` artifact and
/// the `--emit=exe` intermediate cannot drift apart again.
///
/// `--emit=obj`: the object *is* the artifact the user asked for. It goes to `-o` or to `<stem>.o` in
/// the invocation directory, and it stays there.
///
/// `--emit=exe`: the object is a link intermediate, exactly like the generated Rust shim and the
/// generated C runtime, so it goes into the per-pid [`ScratchDir`] and is removed with it. It used to
/// be a bare relative `<stem>.o` written into the process's *current* directory and never deleted -
/// the very defect [`ScratchDir`] was introduced for, fixed there for the shim and the C runtime and
/// missed for the object. Two consequences. The first is observed: eight stray `.o` files sit in
/// this repository's root, invisible because `.gitignore` lists `*.o` twice. The second follows from
/// the code rather than from a sighting - two concurrent `wukongc --emit=exe foo.wk` runs from one
/// directory write the same relative `foo.o` between the write and the linker's read, so the loser
/// links the winner's object, which is a wrong-object link rather than an error.
///
/// `output` is deliberately ignored when `!is_obj`: with `--emit=exe`, `-o` names the *executable*
/// (see [`emit_native`]'s `out`), never the object.
///
/// When [`ScratchDir::new`] failed there is no scratch directory to put it in and the historical
/// relative path is the fallback, exactly as in [`scratch_path`].
fn object_path(
    is_obj: bool,
    output: Option<&Path>,
    stem: &str,
    scratch: Option<&ScratchDir>,
) -> PathBuf {
    if is_obj {
        return output
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(format!("{stem}.o")));
    }
    scratch_path(scratch, format!("{stem}.o"))
}

/// Emit a native object via Cranelift (no LLVM) and, for `--emit=exe`, link it into an executable.
///
/// The link is attempted in one order: [`rustc_link`] first (rustc as the *link driver*, which pulls
/// in the `wukong_runtime` rlib plus the generated [`WUKONG_RT_SHIM`]), and only on
/// [`LinkOutcome::Unavailable`] (rustc could not be spawned, or no rlib next to `current_exe()`)
/// the `$CC` /
/// `cc` fallback with the generated C runtime shim [`WUKONG_RT_C`]. A [`LinkOutcome::Failed`] — rustc
/// ran and the link itself failed — is a real error and exits [`exit::COMPILE_ERROR`] without falling
/// back. If the `cc` fallback cannot even be spawned, the object is copied out of the scratch
/// directory to `<stem>.o` so that it survives, and the driver reports whichever path is actually
/// there - that copy is best effort - and exits [`exit::UNIMPLEMENTED`] (2).
fn emit_native(
    program: &wukong_mir::Program,
    interner: &Interner,
    opts: &Options,
    timer: &mut report::PassTimer,
) -> i32 {
    timer.begin("codegen");
    let backend = wukong_codegen_cranelift::BackendOpt::for_level(opts.opt_level);
    let emitted = if timer.enabled() {
        wukong_codegen_cranelift::emit_object_timed_for_with(
            program,
            interner,
            opts.target_cpu,
            backend,
        )
        .map(|(bytes, split)| {
            timer.object_split(&split);
            bytes
        })
    } else {
        wukong_codegen_cranelift::emit_object_for_with(program, interner, opts.target_cpu, backend)
    };
    let obj = match emitted {
        Ok(b) => b,
        Err(e) => {
            emit_diag(
                &native_failure_diag(&e, &opts.input),
                opts.error_format,
                &Renderer::new(opts.color),
                &SourceMap::new(),
            );
            return exit::COMPILE_ERROR;
        }
    };

    let stem = opts
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());

    let is_obj = opts.emit == EmitStage::Obj;
    // The scratch directory has to exist BEFORE the object is written, because under `--emit=exe` the
    // object is one of the link intermediates that lives in it (see [`object_path`]). `--emit=obj`
    // writes the user's artifact instead and needs no scratch directory at all.
    let scratch = if is_obj { None } else { ScratchDir::new(&stem) };
    let obj_path = object_path(is_obj, opts.output.as_deref(), &stem, scratch.as_ref());
    if let Err(e) = std::fs::write(&obj_path, &obj) {
        emit_driver_error(
            opts,
            format!("could not write `{}`: {e}", obj_path.display()),
        );
        return exit::IO_ERROR;
    }
    // Linkers consume the file; the serialized copy need not stay resident while they run.
    drop(obj);
    if is_obj {
        eprintln!("wrote {}", obj_path.display());
        return exit::OK;
    }

    // The default executable name is `<stem>` plus the host's executable suffix: `.exe` on Windows,
    // **nothing** on unix, where `foo.exe` would be a lie about the file's format and would not be
    // what any adjacent tooling (or the user) looks for. `std::env::consts::EXE_SUFFIX` is the
    // platform's own answer, so the Windows spelling is unchanged, byte for byte.
    let out = opts
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{stem}{}", std::env::consts::EXE_SUFFIX)));

    // Prefer the **rustc-driven link**: `rustc` invokes the object's *native* platform linker and
    // links `wukong_runtime` (the AVX2 kernels) as a real dependency, so a recognized-kernel program
    // resolves its `wukong_*` symbols AND the read-only string `.rodata` relocations link. The MinGW
    // `cc` path below can neither (it crashes on the MSVC object's data relocations and can't consume
    // the Rust runtime staticlib). Output is bit-identical to `--run`. Falls back to `cc` only when
    // rustc or the runtime rlib is unavailable (a scalar, no-data, no-kernel program still links).
    timer.begin("link");
    match rustc_link(scratch.as_ref(), &stem, &obj_path, &out) {
        LinkOutcome::Linked => {
            eprintln!("wrote {}", out.display());
            return exit::OK;
        }
        LinkOutcome::Failed(msg) => {
            emit_driver_error(opts, msg);
            return exit::COMPILE_ERROR;
        }
        LinkOutcome::Unavailable => { /* fall through to the C-runtime `cc` link */ }
    }

    cc_link(opts, scratch.as_ref(), &stem, &obj_path, &out)
}

/// A native-backend (Cranelift) failure as a diagnostic, so it reaches `--error-format=json` and
/// `--explain` like every front-end error instead of escaping as a bare `error:` line. MIR carries no
/// source spans, so the diagnostic names the program and — for a layout failure — the offending type
/// and the function that declares it (the backend puts both in the message).
fn native_failure_diag(message: &str, input: &Path) -> Diagnostic {
    use wukong_codegen_cranelift::NativeErrorKind;
    let program = format!(
        "while compiling `{}` for the native backend",
        input.display()
    );
    match wukong_codegen_cranelift::classify_error(message) {
        NativeErrorKind::LayoutTooLarge => Diagnostic::error(message)
            .with_code("C0002")
            .note(program)
            .help(
                "allocate the buffer on the heap (`alloc_f32(n)` and its siblings) or split it; \
                 `--run` (the interpreter) has no byte-layout limit",
            ),
        NativeErrorKind::Unsupported => Diagnostic::error(message).with_code("C0001").note(program),
        NativeErrorKind::Internal => {
            Diagnostic::bug(format!("native code generation failed: {message}")).note(program)
        }
    }
}

/// Called only after the preferred Rust link is unavailable: link the object against the generated
/// C runtime ([`WUKONG_RT_C`]), whose printing is byte-identical to the oracle's (floats included),
/// so the fallback produces the same output as `--run` for every program it can link. What it
/// cannot provide — the `wukong_*` runtime kernels — is a link error, never a different answer; the
/// note below says which link path is being taken so that error is legible.
fn cc_link(
    opts: &Options,
    scratch: Option<&ScratchDir>,
    stem: &str,
    obj_path: &Path,
    out: &Path,
) -> i32 {
    use std::process::Command;

    // exe (fallback): emit the C runtime into the scratch dir and link it with the system C compiler.
    let rt_path = scratch_path(scratch, format!("{stem}_rt.c"));
    if let Err(e) = std::fs::write(&rt_path, c_runtime_source()) {
        emit_driver_error(
            opts,
            format!("could not write `{}`: {e}", rt_path.display()),
        );
        return exit::IO_ERROR;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    eprint_line(&format!(
        "note: no usable `rustc` + `wukong_runtime` rlib beside wukongc; linking with `{cc}` and the \
         C runtime, which has no `wukong_*` kernels (a program that calls one will not link)"
    ));
    let status = Command::new(&cc)
        .args(cc_link_args(obj_path, &rt_path, out))
        .status();
    match status {
        Ok(s) if s.success() => {
            eprintln!("wrote {}", out.display());
            exit::OK
        }
        Ok(_) => {
            // The object path is NOT named here: it is a scratch intermediate that `scratch` deletes
            // on the way out, so printing it would send the reader to a file that no longer exists.
            // `cc` inherited this process's stderr, so its own diagnostics are directly above.
            eprintln!(
                "error: `{cc}` failed to link the native object for `{}` (its diagnostics are above)",
                opts.input.display()
            );
            exit::COMPILE_ERROR
        }
        Err(_) => {
            // This branch exists to hand the user the object when no linker can be run, so the
            // object must survive `scratch`'s `Drop`: copy it out under the historical `<stem>.o`
            // name in the invocation directory and report whichever path is actually there
            // afterwards. When `ScratchDir::new` failed, `obj_path` IS that path already, and
            // `fs::copy` of a file onto itself is not a guaranteed no-op - std documents the
            // same-file case as likely to truncate - so the equality check is load-bearing.
            let kept = PathBuf::from(format!("{stem}.o"));
            let kept = if obj_path == kept {
                Some(kept)
            } else {
                std::fs::copy(obj_path, &kept).ok().map(|_| kept)
            };
            match kept {
                Some(p) => eprintln!(
                    "error: could not run `{cc}` to link.\n\
                     note: the native object was written to `{}`.\n\
                     help: install a C compiler (gcc/clang/cc) or set CC, or use `--run` to execute.",
                    p.display()
                ),
                None => eprintln!(
                    "error: could not run `{cc}` to link.\n\
                     note: the native object could not be kept (`{}` is not writable).\n\
                     help: install a C compiler (gcc/clang/cc) or set CC, or use `--run` to \
                     execute; `--emit=obj -o <path>` writes the object wherever you choose.",
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("."))
                        .display()
                ),
            }
            exit::UNIMPLEMENTED
        }
    }
}

/// The `cc` fallback's command line, factored out of [`emit_native`] so it can be pinned by a test.
///
/// **`-lm` is load-bearing and must stay last.** [`WUKONG_RT_C`] *unconditionally* defines
/// `wukong_rt_fmod_f64`/`_f32` over C's `fmod`/`fmodf`, and on glibc those live in `libm`, which
/// `cc` does **not** link by default — so without this flag the fallback could not link *any*
/// program on Linux (`undefined reference to 'fmod'`), not merely one that uses `%` on floats. It is
/// a no-op where libm is already folded into the C library (macOS/libSystem, MinGW's stub
/// `libm.a`), so the Windows and macOS link lines are unaffected. Position matters: GNU `ld`
/// resolves left to right, so a library must follow the objects that reference it.
fn cc_link_args(obj_path: &Path, rt_path: &Path, out: &Path) -> Vec<std::ffi::OsString> {
    vec![
        obj_path.into(),
        rt_path.into(),
        "-o".into(),
        out.into(),
        "-O2".into(),
        "-ffp-contract=off".into(),
        "-lm".into(),
    ]
}

/// The outcome of the [`rustc_link`] attempt (the preferred `--emit=exe` link path).
enum LinkOutcome {
    /// The executable was produced.
    Linked,
    /// rustc or the `wukong_runtime` rlib is not available — the caller should try the `cc` fallback.
    Unavailable,
    /// rustc ran but the link failed (a genuine error — do not silently fall back).
    Failed(String),
}

/// The Rust link shim generated for the rustc-driven `--emit=exe` link. It provides the `wukong_rt_*`
/// runtime the Cranelift object calls (formatting through Rust's stdout, so output is byte-identical
/// to the interpreter/JIT oracle — raw LF on Windows, and floats via `Display` like the interp's
/// `format!("{f}")`; the C fallback reproduces that formatting in C), and `extern crate
/// wukong_runtime` pulls the AVX2 microkernels into the link so a recognized-kernel program resolves
/// its `wukong_*` symbols. `#![no_main]` leaves the Cranelift object's own `main` as the CRT entry.
const WUKONG_RT_SHIM: &str = r##"#![no_main]
//! Auto-generated by `wukongc --emit=exe`. Do not edit; it is rewritten on every link.
extern crate wukong_runtime;
use std::io::Write;
#[no_mangle]
pub extern "C" fn wukong_rt_print_i64(x: i64) {
    let o = std::io::stdout();
    let _ = write!(o.lock(), "{x}\n");
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_u64(x: u64) {
    let o = std::io::stdout();
    let _ = write!(o.lock(), "{x}\n");
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_f64(x: f64) {
    let o = std::io::stdout();
    let _ = write!(o.lock(), "{x}\n");
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_f32(x: f32) {
    let o = std::io::stdout();
    let _ = o.lock().write_all(wukong_runtime::fmt::f32_line(x).as_bytes());
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_bool(x: i64) {
    let o = std::io::stdout();
    let _ = o.lock().write_all(wukong_runtime::fmt::bool_line(x).as_bytes());
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_char(x: i64) {
    let o = std::io::stdout();
    let _ = o.lock().write_all(wukong_runtime::fmt::char_line(x).as_bytes());
}
#[no_mangle]
pub extern "C" fn wukong_rt_print_str(p: *const u8) {
    if p.is_null() {
        return;
    }
    // SAFETY: a Wukong `*u8` string buffer, always NUL-terminated by the string lowering.
    unsafe {
        let mut n = 0isize;
        while *p.offset(n) != 0 {
            n += 1;
        }
        let s = std::slice::from_raw_parts(p, n as usize);
        let o = std::io::stdout();
        let mut o = o.lock();
        let _ = o.write_all(s);
        let _ = o.write_all(b"\n");
    }
}
#[no_mangle]
pub extern "C" fn wukong_rt_assert(c: i64) {
    if c == 0 {
        eprintln!("assertion failed");
        std::process::exit(1);
    }
}
#[no_mangle]
pub extern "C" fn wukong_rt_assert_failed() -> i64 { 0 }
#[no_mangle]
pub extern "C" fn wukong_rt_fmod_f64(a: f64, b: f64) -> f64 {
    a % b
}
#[no_mangle]
pub extern "C" fn wukong_rt_fmod_f32(a: f32, b: f32) -> f32 {
    a % b
}
"##;

/// Locate the `wukong_runtime` rlib that [`rustc_link`] will link, given the directory holding the
/// compiler binary (`current_exe().parent()`, i.e. cargo's `target/<profile>/`). `None` means the
/// preferred `--emit=exe` link path is unavailable here and the `cc` fallback is all there is.
///
/// **Two locations, and the second is the one that is always there.** `target/<profile>/` holds
/// `libwukong_runtime.rlib` only when cargo *uplifted* it out of `deps/`, and cargo only uplifts a
/// library when that library is a **root unit of a `build`** — i.e. after a plain `cargo build`.
/// Under `cargo test --workspace` the package's root unit is its *test* binary and the plain rlib is
/// merely a dependency unit, so it stays in `target/<profile>/deps/` under its hash-suffixed name
/// and nothing is uplifted; `cargo run -p wukongc` does not uplift it either, because `-p` makes
/// `wukong_runtime` a dependency rather than a selected member. A tree that has the uplifted copy
/// has it as a *leftover* of some earlier `cargo build`, which is why this looked fine locally and
/// broke on every clean checkout: CI runs `cargo test --workspace`, never `cargo build`, so the
/// preferred link path degraded to the `cc` fallback with reduced capabilities: no `wukong_*`
/// kernels, so every recognized-kernel program failed to link.
/// Searching `deps/` too makes the preferred path available whenever cargo
/// built the runtime at all, however the compiler was built.
///
/// A `deps/` directory can hold several hash-suffixed rlibs (a metadata hash changes when the
/// profile or a dependency does, and cargo does not garbage-collect the old ones), and the uplifted
/// copy can itself be a *leftover* of an older `cargo build`. So **both places feed one candidate
/// list and one comparator**, rather than the uplifted copy short-circuiting the scan: the newest by
/// mtime is tried first, and the uplifted copy wins only a tie. This lookup reports a preferred
/// candidate, not proof of toolchain compatibility: [`rustc_link`] retries the remaining
/// candidates when the pinned compiler rejects an archive's metadata.
pub fn runtime_rlib_in(dir: &Path) -> Option<PathBuf> {
    runtime_rlib_candidates(dir)
        .into_iter()
        .next()
        .map(|c| c.path)
}

/// One `wukong_runtime` rlib that could be linked, with the two facts [`runtime_rlib_in`] ranks on.
struct RlibCandidate {
    path: PathBuf,
    /// `target/<profile>/libwukong_runtime.rlib` - cargo's uplifted copy - rather than a
    /// hash-suffixed entry under `target/<profile>/deps/`.
    uplifted: bool,
    /// `UNIX_EPOCH` when the metadata cannot be read, which ranks the candidate last.
    mtime: std::time::SystemTime,
}

/// **Every** rlib this compiler could link, from both places cargo puts one, as a single list.
///
/// Collapsing the two sources into one list is the point. The previous shape was a 2x2 - {uplifted,
/// `deps/`} x {fresh, stale} - in which only the `deps/` column was ranked and the uplifted copy
/// returned unconditionally from a `is_file()` test placed before the scan. That made one cell
/// unreachable by construction: a STALE uplifted copy beat a fresh `deps/` rlib. It is not
/// hypothetical. Recorded as campaign row W1b: on this repository's own long-lived checkout the
/// top-level copy was six weeks old and the release AOT link failed `E0514` until
/// `cargo build --release -p wukong_runtime` refreshed it. That particular staleness has since been
/// rebuilt away, but the mispick is still live there in miniature - the uplifted copy is 15 seconds
/// older than the newest `deps/` rlib, and the older one is what the previous rule returned. With
/// one list and one comparator that cell cannot be expressed at all.
///
/// **Freshness orders attempts; it does not establish compatibility.** The link does not need a *particular*
/// build unit of `wukong_runtime`: [`WUKONG_RT_SHIM`] names it only as `extern crate
/// wukong_runtime;`, whose entire job is to drag the `#[no_mangle] extern "C"` microkernels into the
/// link. No Rust item crosses the boundary, so a differing `-C metadata` hash is immaterial. What is
/// *not* immaterial is the rustc that produced the rlib: a mismatch there is `E0514`, a hard error.
/// A newer archive can have been built by another compiler while an older compatible archive
/// remains available. Let the selected rustc validate candidates in deterministic descending
/// `(mtime, uplifted, path)` order; never assume the first file belongs to the pinned toolchain.
///
/// The trailing `-` in the `deps/` prefix is load-bearing: it keeps a hypothetical
/// `libwukong_runtime_foo` crate out.
fn runtime_rlib_candidates(dir: &Path) -> Vec<RlibCandidate> {
    fn mtime_of(path: &Path) -> std::time::SystemTime {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    }

    let mut out: Vec<RlibCandidate> = Vec::new();
    // The uplifted name: what `cargo build` (and an installed target layout) leaves next to the
    // binary.
    let uplifted = dir.join("libwukong_runtime.rlib");
    if uplifted.is_file() {
        out.push(RlibCandidate {
            mtime: mtime_of(&uplifted),
            path: uplifted,
            uplifted: true,
        });
    }
    // `deps/libwukong_runtime-<metadata hash>.rlib` - where cargo *always* writes it.
    let Ok(entries) = std::fs::read_dir(dir.join("deps")) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("rlib") {
            continue;
        }
        let matches = path
            .file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.starts_with("libwukong_runtime-"));
        if !matches {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        out.push(RlibCandidate {
            path,
            uplifted: false,
            mtime,
        });
    }
    out.sort_by(|a, b| (b.mtime, b.uplifted, &b.path).cmp(&(a.mtime, a.uplifted, &a.path)));
    out
}

/// Only crate-loading failures justify trying an older runtime. A genuine object/linker error
/// must remain an error, rather than silently changing which runtime implementation is linked.
fn incompatible_runtime_metadata(stderr: &[u8]) -> bool {
    let diagnostics = String::from_utf8_lossy(stderr);
    [
        "error[E0514]",
        "error[E0786]",
        "error[E0460]",
        "error[E0463]",
    ]
    .iter()
    .any(|code| diagnostics.contains(code))
}

fn try_runtime_rlibs(
    candidates: Vec<RlibCandidate>,
    mut attempt: impl FnMut(&Path) -> std::io::Result<std::process::Output>,
) -> std::io::Result<Option<(PathBuf, std::process::Output)>> {
    let mut last = None;
    for candidate in candidates {
        let output = match attempt(&candidate.path) {
            Ok(output) => output,
            // Once rustc has rejected an archive, the link has a real compiler failure.
            // A later spawn/I/O error must not erase that diagnostic and enable the C fallback.
            Err(error) => {
                return match last {
                    Some(previous) => Ok(Some(previous)),
                    None => Err(error),
                }
            }
        };
        let retry = !output.status.success() && incompatible_runtime_metadata(&output.stderr);
        last = Some((candidate.path, output));
        if !retry {
            break;
        }
    }
    Ok(last)
}

/// The LLVM bitcode magic, `'B' 'C' 0xC0 0xDE`, as it appears at offset 0 of a raw `.bc` module.
const LLVM_BITCODE_MAGIC: [u8; 4] = [0x42, 0x43, 0xC0, 0xDE];
/// The LLVM *bitcode wrapper* magic, `0x0B17C0DE` little-endian — the alternative container LLVM
/// accepts wherever a raw module is accepted, so both spellings must count as "this is bitcode".
const LLVM_BITCODE_WRAPPER_MAGIC: [u8; 4] = [0xDE, 0xC0, 0x17, 0x0B];

/// Does this rlib's archive hold **LLVM bitcode members instead of native objects**?
///
/// **This is the release-profile shape of the AOT link, and it is not an anomaly — it is what cargo
/// is documented to produce.** The workspace `[profile.release]` sets `lto = "thin"`, and cargo's
/// LTO planner then compiles every *dependency* of an LTO unit as `Lto::OnlyBitcode`, i.e. with
/// `-C linker-plugin-lto`: the rlib still contains `*.rcgu.o` members, but each one is an LLVM
/// bitcode module rather than a COFF/ELF object, because the final link is expected to be the one
/// that turns them into machine code. MSVC's `link.exe` cannot read a bitcode member, so handing it
/// such an rlib is `LNK1136: invalid or corrupt file` and, in its wake, every `wukong_*` kernel
/// reads as an unresolved external. That is a *fatal* link, which is why the AOT gate scored 0/4 in
/// release while passing 4/4 in debug — cargo compiles a dev-profile dependency with
/// `-C embed-bitcode=no` and its rlib members are ordinary objects.
///
/// The supported way to consume a bitcode-only rlib is to let **rustc** do the LTO (`-C lto`), which
/// is what [`rustc_link`] switches on when this returns `true`. Note the test is deliberately
/// "*any* member is bitcode", not "no member is native": on an MSVC target the `lib.rmeta` and
/// `lib.rmeta-link` members are themselves COFF wrappers around the metadata blob and are present in
/// *both* shapes, so "contains a native object" does not discriminate. An `ObjectAndBitcode` rlib —
/// what cargo emits when a crate feeds both an LTO and a non-LTO unit — carries its bitcode inside a
/// `.llvmbc` *section* of a real object, never as a member that starts with the magic, so it is
/// correctly classified as native and linked without LTO.
///
/// Archive walking, not a full parse: the `!<arch>\n` global header, then 60-byte member headers
/// (name\[16] mtime\[12] uid\[6] gid\[6] mode\[8] size\[10] "`\n"\[2]) with the data padded to an
/// even offset. Only the first four bytes of each member are read, so the cost is a handful of
/// seeks even on the 65 MB debug rlib. Any malformed header stops the walk and answers `false` —
/// "link it the way we always did", the behaviour that is already known to work.
fn rlib_holds_bitcode_members(path: &Path) -> bool {
    use std::io::{Read, Seek, SeekFrom};

    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut global = [0u8; 8];
    if f.read_exact(&mut global).is_err() || &global != b"!<arch>\n" {
        return false;
    }
    let mut header = [0u8; 60];
    loop {
        // A short read here is the clean end of the archive: no bitcode member was found.
        if f.read_exact(&mut header).is_err() {
            return false;
        }
        if &header[58..60] != b"`\n" {
            return false; // not a header where one must be — stop rather than guess
        }
        let Ok(size_field) = std::str::from_utf8(&header[48..58]) else {
            return false;
        };
        let Ok(size) = size_field.trim().parse::<u64>() else {
            return false;
        };
        if size >= 4 {
            let mut magic = [0u8; 4];
            if f.read_exact(&mut magic).is_err() {
                return false;
            }
            if magic == LLVM_BITCODE_MAGIC || magic == LLVM_BITCODE_WRAPPER_MAGIC {
                return true;
            }
            if f.seek(SeekFrom::Current((size - 4) as i64)).is_err() {
                return false;
            }
        } else if f.seek(SeekFrom::Current(size as i64)).is_err() {
            return false;
        }
        // Member data is padded to an even offset with a single `\n`.
        if size % 2 == 1 && f.seek(SeekFrom::Current(1)).is_err() {
            return false;
        }
    }
}

/// Link the Cranelift object into an executable with **`rustc` as the link driver**. rustc invokes
/// the object's native platform linker (on the MSVC host, the installed `link.exe`, discovered
/// automatically) and links `wukong_runtime` as a real dependency, so a recognized-kernel program's
/// `wukong_*` symbols resolve and the string `.rodata` relocations link — neither of which the MinGW
/// `cc` path can do here. The runtime rlib is located by [`runtime_rlib_in`] from the directory
/// holding this compiler binary, and `deps/` next to it goes on `-L dependency=` for the runtime's
/// own dependencies. Returns [`LinkOutcome::Unavailable`] when rustc or the rlib is absent so the
/// caller can try the `cc` fallback.
fn rustc_link(
    scratch: Option<&ScratchDir>,
    stem: &str,
    obj_path: &Path,
    out: &Path,
) -> LinkOutcome {
    use std::process::Command;

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return LinkOutcome::Unavailable,
    };
    let dir = match exe.parent() {
        Some(d) => d.to_path_buf(),
        None => return LinkOutcome::Unavailable,
    };
    let deps = dir.join("deps");
    let candidates = runtime_rlib_candidates(&dir);
    if candidates.is_empty() {
        // Not a cargo target layout (e.g. an installed binary without the rlib) — use the cc path.
        return LinkOutcome::Unavailable;
    }

    let shim_path = scratch_path(scratch, format!("{stem}_shim.rs"));
    if let Err(e) = std::fs::write(&shim_path, WUKONG_RT_SHIM) {
        return LinkOutcome::Failed(format!("could not write `{}`: {e}", shim_path.display()));
    }

    // **The output path handed to rustc must be absolute.** With `-C lto=thin`, rustc names its
    // per-module LTO objects after the *output file's* stem and then passes them to the linker by a
    // path that is only valid relative to its own temp directory — so `-o hello.exe` links with
    // `LNK1181: cannot open input file 'hello.<crate>....rcgu.o.rcgu.o'` while `-o C:\..\hello.exe`
    // links fine (rustc 1.97.1). Absolutizing costs nothing on the non-LTO path and is what makes
    // the LTO path below usable from a bare `wukongc --emit=exe prog.wk`. `std::path::absolute` is
    // pure path arithmetic against the CWD — it does not touch the filesystem, and the *user-facing*
    // `wrote <path>` note in `emit_native` still prints the path the user asked for.
    let out_arg = std::path::absolute(out).unwrap_or_else(|_| out.to_path_buf());

    // Consume a bitcode-only rlib the only way it can be consumed: make rustc do the LTO. See
    // [`rlib_holds_bitcode_members`] — this is the release profile (`lto = "thin"`), where cargo
    // compiles the runtime with `-C linker-plugin-lto` and `link.exe` answers `LNK1136`. `opt-level`
    // is not cosmetic here: the LTO backend re-codegens every imported module at the *linking*
    // crate's level, and at the rustc default (`-C opt-level=0`) the AVX2 kernels come out as
    // `vfmadd213ps %ymm2,%ymm1,%ymm0` fed from stack reloads instead of the release form
    // `vfmadd231ps (%rdx,%r11,1),%ymm2,%ymm0` — a correct but deoptimized runtime in every AOT
    // binary. `lto=thin` + `opt-level=3` mirrors the workspace `[profile.release]` the rlib was
    // built under, so the kernels land in the exe exactly as that profile compiled them.
    let rustc = link_rustc();
    let run = try_runtime_rlibs(candidates, |rlib| {
        let bitcode_rlib = rlib_holds_bitcode_members(rlib);
        let mut cmd = Command::new(&rustc);
        cmd.arg(&shim_path)
            .arg("--edition")
            .arg("2021")
            // The rlib we link against is built by this workspace, whose `[profile.release]` sets
            // `panic = "abort"`. rustc defaults the shim to `unwind`, and a strategy mismatch is a hard
            // metadata error ("the crate `wukong_runtime` requires panic strategy `abort` which is
            // incompatible with this crate's strategy of `unwind`") that rejected every release-profile
            // link before it ever reached the linker. An abort-strategy crate may link unwind-built
            // dependencies, so this is equally correct against a debug-profile rlib.
            .arg("-C")
            .arg("panic=abort");
        if bitcode_rlib {
            cmd.arg("-C").arg("lto=thin").arg("-C").arg("opt-level=3");
        }
        // `.output()`, not `.status()`, for one reason: the E0514 note below has to READ rustc's
        // diagnostics to know whether to print itself. Everything rustc writes still reaches the same
        // descriptor, forwarded immediately after it exits - but two things about the SHAPE change. It
        // arrives as one block at the end rather than streaming, and rustc now sees a pipe instead of a
        // terminal, so its diagnostics come through UNCOLOURED. The link is a single short-lived child
        // at the end of compilation, so neither is something the user can act on.
        cmd.arg("--extern")
            .arg(format!("wukong_runtime={}", rlib.display()))
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("-C")
            .arg(format!("link-arg={}", obj_path.display()))
            .arg("-o")
            .arg(&out_arg)
            .output()
    });
    let _ = std::fs::remove_file(&shim_path);

    let (rlib, produced) = match run {
        Ok(Some(o)) => o,
        Ok(None) => return LinkOutcome::Unavailable,
        // rustc could not be spawned: fall back to the C-runtime `cc` link (scalar, no-data
        // programs only).
        Err(_) => return LinkOutcome::Unavailable,
    };
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(&produced.stdout);
        let _ = std::io::stderr().write_all(&produced.stderr);
    }
    if produced.status.success() {
        return LinkOutcome::Linked;
    }
    let bitcode_rlib = rlib_holds_bitcode_members(&rlib);

    // The object path is deliberately not named: it is a scratch intermediate that `emit_native`'s
    // `ScratchDir` deletes on the way out (see [`object_path`]).
    let mut msg = format!(
        "`rustc` failed to link the native object for `{stem}` (its diagnostics are above)"
    );
    if bitcode_rlib {
        msg.push_str(&format!(
            "\nnote: `{}` holds LLVM bitcode members (an LTO-profile rlib), so the link was run \
             with `-C lto=thin -C opt-level=3`",
            rlib.display()
        ));
    }
    // E0514 is "found crate `wukong_runtime` compiled by an incompatible version of rustc". It is
    // raised by the rustc invoked HERE about the rlib picked by [`runtime_rlib_in`], so naming both
    // is the whole diagnosis; without it the message reads as a Wukong bug. Matching on the bare
    // code rather than the prose keeps it stable across rustc's wording changes.
    if incompatible_runtime_metadata(&produced.stderr) {
        msg.push_str(&format!(
            "\nnote: no available runtime archive could be loaded by `{}`; the last archive tried \
             was `{}`.\n\
             help: rebuild wukong_runtime with that compiler (set RUSTC to the displayed path \
             for `cargo build -p wukong_runtime`, adding `--release` for a release compiler).",
            rustc.display(),
            rlib.display(),
        ));
    }
    LinkOutcome::Failed(msg)
}

/// The `rustc` that drives the `--emit=exe` link.
///
/// **Not a bare name looked up on `PATH`.** A spawned child inherits the caller's working directory,
/// and under rustup the `PATH` entry is a shim that resolves a toolchain from *that* directory - a
/// `rust-toolchain.toml` beside it, or a `rustup override` on it - which has nothing to do with the
/// toolchain that built the `wukong_runtime` rlib being linked. When the two differ, rustc rejects
/// the rlib with `E0514: found crate wukong_runtime compiled by an incompatible version of rustc`,
/// which reads like a Wukong bug and is a toolchain-resolution one. Running `wukongc --emit=exe`
/// from any directory outside this workspace is enough to reach it.
///
/// `build.rs` records cargo's own `RUSTC`. In a rustup setup that is an absolute path to the
/// toolchain binary itself and *not* the rustup shim (measured on this host,
/// `.../.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin/rustc.exe`), so it is immune
/// to directory-based resolution - and it is the compiler cargo used to build THIS binary. It is
/// not guaranteed to be the one that built the rlib the link picks: a `target/` tree accumulates
/// rlibs from every toolchain it has ever seen, which is why [`rustc_link`] tries candidates in
/// freshness order until one passes crate loading. The `PATH` fallback keeps a relocated or installed
/// binary working; the
/// E0514 note in [`rustc_link`] names the resolved path either way, so the failure is legible in
/// both worlds.
fn link_rustc() -> PathBuf {
    match option_env!("WUKONG_BUILD_RUSTC") {
        Some(p) if !p.is_empty() && Path::new(p).is_file() => PathBuf::from(p),
        _ => PathBuf::from("rustc"),
    }
}

/// `--emit=grad`: differentiate the designated loss function and print the forward + backward MIR.
///
/// The loss function must already be single-block SSA (the caller forces `-O1`); its buffer
/// parameters lower to `Ptr` and the scalar loss is `ret`-ed. Any recognized kernel calls in it
/// (`wukong_sgemm_nt`, `wukong_vmath_f32`, `wukong_sreduce_f32`, `wukong_norm_f32`,
/// `wukong_velem_f32` — each recognized in its serial *and* its `_parallel` spelling) get their
/// **tuned-kernel** backward — so the emitted gradient rides the same
/// kernels as the forward. A differentiable value with no VJP rule is a hard error, never a
/// silently-zero gradient (see `wukong_autodiff`).
fn emit_grad(program: &wukong_mir::Program, interner: &mut Interner, opts: &Options) -> i32 {
    let (fwd, gradfn) = match build_grad(program, interner, opts) {
        Ok(pair) => pair,
        Err(e) => {
            emit_driver_error(opts, format!("--emit=grad: {e}"));
            return exit::COMPILE_ERROR;
        }
    };
    let out = wukong_mir::Program {
        funcs: vec![fwd, gradfn],
        level: program.level,
        statics: program.statics.clone(),
        externs: program.externs.clone(),
    };
    // A verifier failure means we just printed an internal-compiler-error; never report success —
    // the same mapping the `--emit=mir` / `--emit=mir-high` callers already apply.
    if emit_mir(&out, interner) {
        exit::COMPILE_ERROR
    } else {
        exit::OK
    }
}

/// Resolve the CLI-selected entry point before whole-program optimization. Normal compilation has
/// no extra roots; selecting a gradient must not disable inlining or downgrade the requested level.
fn autodiff_roots(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
) -> Result<Vec<wukong_span::Symbol>, String> {
    if opts.emit != EmitStage::Grad && !opts.grad.train && !exports_gradient(opts) {
        return Ok(Vec::new());
    }
    let name = opts.grad.of.as_deref().unwrap_or("loss");
    let symbol = interner.intern(name);
    if program.function(symbol).is_none() {
        return Err(missing_grad_target(name));
    }
    Ok(vec![symbol])
}

/// Whether this invocation exports a gradient ahead of time: `--emit=obj` with an explicit
/// `--grad-of=<fn>`.
fn exports_gradient(opts: &Options) -> bool {
    opts.emit == EmitStage::Obj && opts.grad.of.is_some() && !opts.grad.train
}

/// `--emit=obj --grad-of=<fn>`: the ahead-of-time gradient. Appends the generated `<fn>_grad` to the
/// program so the object exports the backward next to the forward — C ABI: the loss's parameters,
/// then one caller-zeroed gradient buffer per `--grad-wrt` entry, gradients accumulated into them —
/// for a host training loop (or any C/C++/Rust caller) to link against, with no compiler at run
/// time. The forward it differentiates is the call-flattened one (`wukong_autodiff::inline_calls`),
/// so the exported backward calls no helper the object might not define.
fn append_gradient(
    program: &mut wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
) -> Result<(), String> {
    let (_, gradfn) = build_grad(program, interner, opts)?;
    if program.function(gradfn.name).is_some() {
        return Err(format!(
            "the program already defines `{}`; rename it or differentiate another function",
            interner.resolve(gradfn.name)
        ));
    }
    program.funcs.push(gradfn);
    Ok(())
}

fn missing_grad_target(name: &str) -> String {
    format!(
        "autodiff target `{name}` is not a defined function; select an existing loss function \
             with --grad-of=<fn> (the default is `loss`)"
    )
}

/// Shared core of `--emit=grad` and `--train`: resolve the target loss function and its `wrt` list
/// from `opts`, differentiate and optimize the backward, and return `(forward_clone, gradient_function)`.
fn build_grad(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
) -> Result<(wukong_mir::Function, wukong_mir::Function), String> {
    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let sym = interner.intern(&name);
    let func = program
        .function(sym)
        .ok_or_else(|| missing_grad_target(&name))?;
    let wrt = grad_wrt(func, &opts.grad.wrt, opts.grad.train)?;
    // Splice straight-line helper functions into the loss first, so a loss written as composed
    // functions differentiates at every -O level (not only where -O2's inliner happened to fire).
    // The flattened forward is the one returned and run, so `--train` never calls a helper the
    // training program does not contain.
    let func = wukong_autodiff::inline_calls(program, func);
    let gradfn = wukong_autodiff::grad(&func, &wrt, interner)?;
    // Autodiff synthesizes MIR after optimization, so its output needs a verifier boundary even
    // without --verify-each. Keep every callee definition available for signature checks before
    // this function can hand generated MIR to emission or training.
    let mut checked = program.clone();
    checked.funcs.push(gradfn.clone());
    wukong_opt::verify_program_stage(&checked, "autodiff")
        .map_err(|failure| failure.to_string())?;
    // Optimize only newly generated MIR: rerunning the pipeline over the input program could
    // change the forward or inline/delete its callees after the caller has selected them. The
    // backward still gets the requested local passes and unrolling, with the same O1 floor used
    // to prepare the forward for autodiff. Preserve its externally invoked entry point explicitly.
    let grad_name = gradfn.name;
    let mut generated = wukong_mir::Program {
        funcs: vec![gradfn],
        level: program.level,
        statics: program.statics.clone(),
        externs: program.externs.clone(),
    };
    let level = opts.opt_level.max(1);
    if opts.verify_each {
        wukong_opt::optimize_verified_preserving(&mut generated, level, &[grad_name])
            .map_err(|failure| failure.to_string())?;
    } else {
        wukong_opt::optimize_preserving(&mut generated, level, &[grad_name]);
    }
    let gradfn = generated
        .funcs
        .pop()
        .ok_or_else(|| "internal: optimization removed the gradient entry point".to_string())?;
    checked.funcs.pop();
    checked.funcs.push(gradfn.clone());
    // The isolated optimizer cannot check calls against the original callee definitions. Restore
    // that complete context at the output boundary, even when per-pass verification is disabled.
    wukong_opt::verify_program_stage(&checked, "autodiff optimization")
        .map_err(|failure| failure.to_string())?;
    Ok((func, gradfn))
}

/// The parameter indices to differentiate w.r.t.: the explicit `--grad-wrt` list (validated to be
/// distinct, in-range pointer parameters), or — when empty — *every* pointer (buffer) parameter of
/// the loss. Differentiating an unread or output buffer is harmless (its gradient is simply zero),
/// so the "all buffers" default never produces a wrong answer, only occasionally an unused zero
/// buffer.
///
/// `training` selects the `--train` flavour of the default: the trainable weights, i.e. every buffer
/// parameter *except* the last one, which by the `--train` convention is the scalar loss output.
/// Without that exclusion the documented default expanded to a list containing the loss-output index
/// and `train_loop` then rejected it, so `--train` could never run without an explicit `--grad-wrt`.
fn grad_wrt(
    func: &wukong_mir::Function,
    requested: &[usize],
    training: bool,
) -> Result<Vec<usize>, String> {
    let is_ptr = |i: usize| *func.value_type(func.params[i]) == wukong_mir::MirType::Ptr;
    if requested.is_empty() {
        let loss_out = func.params.len().wrapping_sub(1);
        let all: Vec<usize> = (0..func.params.len())
            .filter(|&i| is_ptr(i) && !(training && i == loss_out))
            .collect();
        if all.is_empty() {
            return Err(if training {
                "the loss function has no trainable buffer parameters (--train differentiates every \
                 buffer parameter except the last, which is the scalar loss output)"
                    .into()
            } else {
                "the loss function has no buffer (pointer) parameters to differentiate".to_string()
            });
        }
        return Ok(all);
    }
    let mut seen = std::collections::HashSet::new();
    for &wi in requested {
        if wi >= func.params.len() {
            return Err(format!(
                "--grad-wrt index {wi} is out of range (the function has {} parameter(s))",
                func.params.len()
            ));
        }
        if !is_ptr(wi) {
            return Err(format!(
                "--grad-wrt index {wi} is not a buffer (pointer) parameter"
            ));
        }
        // A repeated index appends a second gradient buffer to the backward's parameter list, but
        // `Vjp`'s param -> gradient-buffer map keeps only the last insert, so the earlier buffer is
        // never written and its caller reads back all zeros.
        if !seen.insert(wi) {
            return Err(format!(
                "--grad-wrt index {wi} is repeated; each parameter may be differentiated at most once"
            ));
        }
    }
    Ok(requested.to_vec())
}

/// Count concrete f32 leaves with checked arithmetic. A top-level scalar is not a buffer.
fn ty_elem_count(ty: &wukong_types::Ty) -> Option<usize> {
    use wukong_types::{Dim, Layout, Scalar, Ty};
    fn leaves(ty: &Ty) -> Option<u64> {
        match ty {
            Ty::Scalar(Scalar::F32) => Some(1),
            Ty::Array { elem, len, .. } => leaves(elem)?.checked_mul(*len),
            Ty::Tensor {
                elem: Scalar::F32,
                shape,
                layout: Layout::Contiguous | Layout::ColMajor,
            } => shape.0.iter().try_fold(1u64, |acc, d| match d {
                Dim::Const(n) => acc.checked_mul(*n),
                _ => None,
            }),
            _ => None,
        }
    }
    if !matches!(ty, Ty::Array { .. } | Ty::Tensor { .. }) {
        return None;
    }
    let count = usize::try_from(leaves(ty)?).ok()?;
    valid_train_extent(count).then_some(count)
}

// Vec allocations and native pointer offsets must fit isize, not merely usize.
fn valid_train_extent(count: usize) -> bool {
    count
        .checked_mul(std::mem::size_of::<f32>())
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

/// Recover the element count of each parameter of the `--train` / `--grad-of` loss function from the
/// semantic types (the `[f32; N]` / `Tensor[..]` annotations lost when the params lower to `Ptr`).
fn loss_param_lens(
    sema: &wukong_sema::SemaResult,
    interner: &mut Interner,
    opts: &Options,
) -> Result<Vec<usize>, String> {
    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let sym = interner.intern(&name);
    let def = sema
        .defs
        .lookup(sym)
        .ok_or_else(|| missing_grad_target(&name))?;
    let sig = match &def.kind {
        wukong_sema::DefKind::Fn(sig) => sig,
        _ => return Err(format!("`{name}` is not a function")),
    };
    sig.params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            ty_elem_count(ty).ok_or_else(|| {
                format!(
                    "parameter {i} of `{name}` must be a concrete f32 array or dense tensor \
                     with a representable byte extent; --train needs concrete buffer sizes"
                )
            })
        })
        .collect()
}

/// A tiny deterministic LCG producing `f32` in ~`[-0.4, 0.4]` — seeds the `--train` buffers so a run
/// is reproducible (honest: no hidden entropy) and a test can assert a specific loss trajectory.
fn train_rand(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((*seed >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.8
        })
        .collect()
}

/// `--train`: run a fwd→bwd→optimizer loop on the loss function and print the loss trajectory.
///
/// Convention: `--grad-wrt` names the **trainable** buffer parameters (the weights); every other
/// buffer is fixed input data; the **last** parameter is the scalar loss output `[f32; 1]`. All
/// buffers are seeded deterministically (`--train-seed`); each step runs the gradient kernel (which
/// replays the forward, so it yields both the loss and the gradients), then applies SGD or the fused
/// AdamW kernel to the weights. The printed trajectory is the honest end-to-end signal — on a
/// convex objective it strictly decreases.
fn run_train(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
    lens: &[usize],
) -> i32 {
    match train_loop(program, interner, opts, lens) {
        Ok(traj) => {
            use std::io::Write;
            let mut out = String::new();
            let opt = match opts.grad.train_opt {
                TrainOpt::Sgd => "sgd",
                TrainOpt::AdamW => "adamw",
            };
            let decay = if opts.grad.train_weight_decay == 0.0 {
                String::new()
            } else {
                format!(", weight_decay={}", opts.grad.train_weight_decay)
            };
            out.push_str(&format!(
                "train: {} step(s), lr={}, optimizer={opt}{decay}\n",
                traj.len().saturating_sub(1),
                opts.grad.train_lr
            ));
            for (i, l) in traj.iter().enumerate() {
                out.push_str(&format!("step {i:>4}: loss {l:.6}\n"));
            }
            if let (Some(&first), Some(&last)) = (traj.first(), traj.last()) {
                out.push_str(&format!(
                    "loss {first:.6} -> {last:.6}  ({:.2}x reduction)\n",
                    if last != 0.0 {
                        first / last
                    } else {
                        f32::INFINITY
                    }
                ));
            }
            let _ = std::io::stdout().write_all(out.as_bytes());
            exit::OK
        }
        Err(e) => {
            emit_driver_error(opts, format!("--train: {e}"));
            exit::COMPILE_ERROR
        }
    }
}

/// The testable core of `--train`: build the {forward, backward, optimizer} program, initialize the
/// buffers, run the loop, and return the loss trajectory (`traj[i]` = loss after `i` updates, with a
/// final entry after the last update — so `traj.len() == steps + 1`).
fn train_loop(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
    lens: &[usize],
) -> Result<Vec<f32>, String> {
    train_loop_with_buffers(program, interner, opts, lens).map(|(trajectory, _)| trajectory)
}

fn train_loop_with_buffers(
    program: &wukong_mir::Program,
    interner: &mut Interner,
    opts: &Options,
    lens: &[usize],
) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
    use wukong_autodiff::optim::{self, hp};
    use wukong_mir::Program;

    if !matches!(opts.backend, BackendKind::Interp | BackendKind::Native) {
        return Err("--train supports only --backend=interp or --backend=native".into());
    }
    if lens.iter().any(|&n| !valid_train_extent(n)) {
        return Err("training buffer byte extent is not representable".into());
    }
    let weight_decay = opts.grad.train_weight_decay;
    if !weight_decay.is_finite() || weight_decay < 0.0 {
        return Err("--train-weight-decay must be finite and non-negative".into());
    }
    if weight_decay != 0.0 && opts.grad.train_opt != TrainOpt::AdamW {
        return Err("nonzero --train-weight-decay requires --train-opt=adamw".into());
    }

    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let fsym = interner.intern(&name);
    let (nparams, wrt) = {
        let loss_fn = program
            .function(fsym)
            .ok_or_else(|| format!("no function `{name}` in the module"))?;
        if loss_fn.params.is_empty()
            || loss_fn
                .params
                .iter()
                .any(|&p| loss_fn.value_type(p) != &wukong_mir::MirType::Ptr)
        {
            return Err("--train requires every loss parameter to be a buffer (MIR Ptr)".into());
        }
        (
            loss_fn.params.len(),
            grad_wrt(loss_fn, &opts.grad.wrt, true)?,
        )
    };
    if lens.len() != nparams {
        return Err(format!(
            "internal: {} param sizes for a {nparams}-param loss function",
            lens.len()
        ));
    }
    let loss_out = nparams - 1;
    if lens[loss_out] != 1 {
        return Err(
            "the last parameter must be the scalar loss output `[f32; 1]` (the `--train` convention)"
                .into(),
        );
    }
    if wrt.contains(&loss_out) {
        return Err("--grad-wrt must not include the loss-output parameter (the last one)".into());
    }

    // Build {forward, backward} plus one fused AdamW kernel per distinct trainable-tensor size.
    let (fwd, gradfn) = build_grad(program, interner, opts)?;
    let (fwd_name, gname) = (fwd.name, gradfn.name);
    let mut funcs = vec![fwd, gradfn];
    let mut adamw: std::collections::HashMap<usize, wukong_span::Symbol> =
        std::collections::HashMap::new();
    if opts.grad.train_opt == TrainOpt::AdamW {
        let mut sizes: Vec<usize> = wrt.iter().map(|&wi| lens[wi]).collect();
        sizes.sort_unstable();
        sizes.dedup();
        for n in sizes {
            let f = optim::build_adamw_step(interner, n);
            adamw.insert(n, f.name);
            funcs.push(f);
        }
    }
    let mut train_prog = Program {
        funcs,
        level: program.level,
        statics: program.statics.clone(),
        externs: program.externs.clone(),
    };
    // AdamW adds more generated MIR after build_grad's gate. Validate the exact execution program
    // unconditionally, before allocating buffers or entering the interpreter.
    wukong_opt::verify_program_stage(&train_prog, "training kernels")
        .map_err(|failure| failure.to_string())?;

    let mut grad_lens = lens.to_vec();
    grad_lens.extend(wrt.iter().map(|&wi| lens[wi]));
    let mut entries = vec![(fwd_name, lens.to_vec()), (gname, grad_lens)];
    let mut adamw_sizes: Vec<_> = adamw.keys().copied().collect();
    adamw_sizes.sort_unstable();
    for n in adamw_sizes {
        entries.push((adamw[&n], vec![n, n, n, n, hp::LEN]));
    }
    let executor =
        training_native::TrainingExecutor::new(&mut train_prog, interner, opts.backend, entries)?;

    // Deterministic buffer init; the loss output is zeroed, moment state starts at zero.
    let mut seed = opts.grad.train_seed;
    let mut bufs: Vec<Vec<f32>> = lens.iter().map(|&n| train_rand(&mut seed, n)).collect();
    bufs[loss_out] = vec![0.0; 1];
    let mut mstate: Vec<Vec<f32>> = wrt.iter().map(|&wi| vec![0.0; lens[wi]]).collect();
    let mut vstate: Vec<Vec<f32>> = wrt.iter().map(|&wi| vec![0.0; lens[wi]]).collect();

    let lr = opts.grad.train_lr;
    let (beta1, beta2, eps) = (0.9f64, 0.999f64, 1e-8f64);
    // Grown amortized rather than pre-reserved: `--train-steps` is an unvalidated CLI integer, and
    // reserving from it let `--train-steps=999999999999` abort inside the allocator ("memory
    // allocation of 4000000000000 bytes failed", exit 127) before a single step ran — raw runtime
    // text with no diagnostic. `usize::MAX` additionally wrapped `+ 1` to a zero reservation.
    let mut traj = Vec::new();

    let run = |prog: &Program, entry, bufs: &mut [Vec<f32>], it: &Interner| {
        executor.run(prog, entry, bufs, it)
    };

    for step in 0..opts.grad.train_steps {
        // The gradient kernel replays the forward, so one call gives the current loss + the gradients.
        let mut gbufs: Vec<Vec<f32>> = bufs.clone();
        for &wi in &wrt {
            gbufs.push(vec![0.0; lens[wi]]);
        }
        run(&train_prog, gname, &mut gbufs, interner)?;
        traj.push(gbufs[loss_out][0]);

        match opts.grad.train_opt {
            TrainOpt::Sgd => {
                for (i, &wi) in wrt.iter().enumerate() {
                    let g = &gbufs[nparams + i];
                    for (wv, &gv) in bufs[wi].iter_mut().zip(g) {
                        *wv -= lr * gv;
                    }
                }
            }
            TrainOpt::AdamW => {
                let t = (step + 1) as i32;
                let mut hpbuf = vec![0.0f32; hp::LEN];
                hpbuf[hp::LR] = lr;
                hpbuf[hp::BETA1] = beta1 as f32;
                hpbuf[hp::BETA2] = beta2 as f32;
                hpbuf[hp::EPS] = eps as f32;
                hpbuf[hp::WD] = weight_decay;
                hpbuf[hp::BC1] = (1.0 - beta1.powi(t)) as f32;
                hpbuf[hp::BC2] = (1.0 - beta2.powi(t)) as f32;
                for (i, &wi) in wrt.iter().enumerate() {
                    let aname = adamw[&lens[wi]];
                    let mut ab = vec![
                        bufs[wi].clone(),
                        gbufs[nparams + i].clone(),
                        mstate[i].clone(),
                        vstate[i].clone(),
                        hpbuf.clone(),
                    ];
                    run(&train_prog, aname, &mut ab, interner)?;
                    bufs[wi] = ab[0].clone();
                    mstate[i] = ab[2].clone();
                    vstate[i] = ab[3].clone();
                }
            }
        }
    }
    // One final forward to report the loss after the last update.
    run(&train_prog, fwd_name, &mut bufs, interner)?;
    traj.push(bufs[loss_out][0]);
    Ok((traj, bufs))
}

/// Print the MIR for `program`, running the verifier first. Any verifier failure is emitted as an
/// `internal compiler error (MIR verify)` line to stderr. Returns `true` if the verifier reported
/// at least one failure, so callers can map an internal-compiler-error to a nonzero process exit
/// status instead of reporting success on invalid MIR.
fn emit_mir(program: &wukong_mir::Program, interner: &Interner) -> bool {
    let verify_failed = verify_or_ice(program);
    print_artifact(&wukong_mir::print::print_program(program, interner));
    verify_failed
}

/// Run the MIR verifier over the program, including internal call signatures, reporting each violation as an
/// `internal compiler error (MIR verify)` line. Returns `true` if anything failed. A failure is a
/// bug in a compiler pass, not in the user's program, so it is reported and the process stops —
/// never handed on to a backend.
fn verify_or_ice(program: &wukong_mir::Program) -> bool {
    let mut verify_failed = false;
    for ice in wukong_mir::verify::verify_program(program) {
        eprint_line(&format!("internal compiler error (MIR verify): {ice}"));
        verify_failed = true;
    }
    verify_failed
}

/// Write a compiler artifact to stdout.
///
/// Deliberately a *fallible* write rather than `print!`: `print!` panics when the consumer closes
/// the pipe, and with the workspace's `panic = "abort"` release profile that panic aborts the whole
/// process. `wukongc --emit=mir big.wk | head -1` therefore printed Rust's internal
/// "failed printing to stdout: The pipe has been ended. (os error 109)" and exited 127 instead of
/// reporting the compile result. A closed consumer is not a compiler error — drop the rest of the
/// artifact and let the normal exit code stand.
fn print_artifact(s: &str) {
    use std::io::Write;
    let _ = std::io::stdout().write_all(s.as_bytes());
}

/// One line of compiler output on stderr, fallible for the same reason as [`print_artifact`]: a
/// many-diagnostic file piped into `head` aborted the process partway through the diagnostic stream.
fn eprint_line(s: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{s}");
}

/// Preserve optimizer failure context in the existing structured diagnostic format.
fn optimization_failure_diag(
    failure: &wukong_opt::VerificationFailure,
    interner: &Interner,
) -> Diagnostic {
    let context = failure
        .function
        .map(|name| format!(" in function `{}`", interner.resolve(name)))
        .unwrap_or_default();
    let mut diagnostic = Diagnostic::bug(format!(
        "MIR verification failed after `{}`{context}",
        failure.pass
    ));
    for error in &failure.errors {
        diagnostic = diagnostic.note(error);
    }
    diagnostic
}

/// Emit a single diagnostic to stderr in the requested format.
fn emit_diag(d: &Diagnostic, fmt: ErrorFormat, renderer: &Renderer, sm: &SourceMap) {
    match fmt {
        ErrorFormat::Human => eprint_line(&renderer.render(d, sm)),
        ErrorFormat::Json => eprint_line(&wukong_diag::to_json(d, sm)),
    }
}

/// Emit a driver-level error (an IO, CLI, autodiff or training failure that has no source span)
/// honoring `--error-format`. These were raw `eprintln!("error: …")` lines that bypassed the JSON
/// stream entirely, so a `--error-format=json` consumer saw unparseable text interleaved with the
/// machine-readable diagnostics. Routed through the renderer, a human run is byte-identical to the
/// old `error: <msg>` line, and a JSON run now gets a well-formed object.
fn emit_driver_error(opts: &Options, message: impl Into<String>) {
    let d = Diagnostic::error(message);
    let sm = SourceMap::new();
    match opts.error_format {
        ErrorFormat::Human => eprint_line(&Renderer::new(opts.color).render(&d, &sm)),
        ErrorFormat::Json => eprint_line(&wukong_diag::to_json(&d, &sm)),
    }
}

/// Resolve the levels of `lints` (the parser's, on a build that stops before the lint stage)
/// against `module` and render them (honoring `--error-limit`), after whatever was rendered before.
/// Returns how many were denied, i.e. are now errors, for the caller's "aborting due to" summary.
fn render_lints(
    lints: Vec<Diagnostic>,
    module: &wukong_ast::Module,
    interner: &Interner,
    opts: &Options,
    renderer: &Renderer,
    sm: &SourceMap,
) -> usize {
    let lints = wukong_sema::lints::apply_levels(lints, module, interner, &opts.lint_levels);
    emit_diags(&lints, opts, renderer, sm)
}

/// What the lint pass must know about how this build's artifact is used: which functions besides
/// `main` an outside caller enters (`dead_code`'s roots). Under `--emit=grad`/`--train` the driver
/// itself calls the `--grad-of` function by name; an `--emit=obj`/`--emit=llvm-ir` artifact is
/// linked by a host that may call any of its exported functions.
fn lint_options_for(opts: &Options) -> wukong_sema::lints::LintOptions {
    let grad = opts.grad.train || opts.emit == EmitStage::Grad;
    wukong_sema::lints::LintOptions {
        roots: if grad {
            vec![opts.grad.of.clone().unwrap_or_else(|| "loss".to_string())]
        } else {
            Vec::new()
        },
        library: !opts.run && !grad && matches!(opts.emit, EmitStage::Obj | EmitStage::LlvmIr),
    }
}

/// Emit a batch of diagnostics, honoring `--error-limit` and separating consecutive diagnostics
/// with a blank line in human output (JSON stays one object per line). Returns the number of
/// error-severity diagnostics in the batch — counted even when suppressed by the limit — so the
/// caller's "aborting due to N errors" summary stays accurate.
fn emit_diags(diags: &[Diagnostic], opts: &Options, renderer: &Renderer, sm: &SourceMap) -> usize {
    let human = opts.error_format == ErrorFormat::Human;
    let mut printed = 0usize;
    let mut errors = 0usize;
    let mut suppressed = false;
    for d in diags {
        if d.is_error() {
            errors += 1;
        }
        // Once more errors than the limit have been seen, stop printing anything further.
        if let Some(n) = opts.error_limit {
            if errors > n {
                suppressed = true;
            }
        }
        if suppressed {
            continue;
        }
        if printed > 0 && human {
            eprint_line("");
        }
        emit_diag(d, opts.error_format, renderer, sm);
        printed += 1;
    }
    errors
}

/// Print rustc's "aborting due to N previous error(s)" trailer in human format. JSON consumers count
/// the stream themselves, so it is suppressed there.
fn emit_summary(errors: usize, opts: &Options, renderer: &Renderer) {
    if errors == 0 || opts.error_format != ErrorFormat::Human {
        return;
    }
    let plural = if errors == 1 { "" } else { "s" };
    let d = Diagnostic::error(format!("aborting due to {errors} previous error{plural}"));
    eprint_line(&renderer.render(&d, &SourceMap::new()));
}

fn render_all(opts: &Options, renderer: &Renderer, sink: &DiagnosticSink, sm: &SourceMap) {
    let errors = emit_diags(sink.diagnostics(), opts, renderer, sm);
    emit_summary(errors, opts, renderer);
}

/// The `--emit=grad` CLI surface, end to end from `.wk` source: compile → optimize → differentiate
/// the loss function, then **finite-difference-gate** the emitted backward. This is the real proof
/// that autodiff-from-source is correct — the recognized backward kernels are gate-blind (both
/// backends call the same symbol), so the f64/closed-form finite-difference check, not interp==native,
/// is what proves the gradient. Toolchain-free: it runs entirely on the interpreter oracle.
#[cfg(test)]
mod lower_options_tests {
    use super::*;

    fn opts_for(backend: BackendKind) -> Options {
        Options {
            backend,
            ..Options::default()
        }
    }

    /// Lower `src` at `-O2` under `lo` and count the `Op::VecKernelCall`s that survive.
    fn count_vec_kernels(src: &str, lo: wukong_mir_build::LowerOptions) -> usize {
        let mut sm = SourceMap::new();
        let id = sm.add("lower_opts.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors: {ld:?}");
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors: {sd:?}");
        let (mut program, md) =
            wukong_mir_build::lower_program_with(&module, &sema, &mut interner, lo);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors: {md:?}");
        wukong_opt::optimize(&mut program, 2);
        program
            .funcs
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, wukong_mir::Op::VecKernelCall { .. }))
            .count()
    }

    /// Whole-program GPU lowering is the one pipeline that must not be handed a CPU recipe - and it
    /// is the *only* one. In particular `--backend=gpu` (recognizer offload) still tree-walks the
    /// program in the interpreter, so it wants the recipes exactly as `interp` and `native` do; the
    /// cheapest way to break this fix would be to widen the match to "any GPU backend".
    #[test]
    fn only_whole_program_gpu_lowering_clears_the_vec_kernel_recipes() {
        for b in [BackendKind::Interp, BackendKind::Native, BackendKind::Gpu] {
            assert!(
                lower_options_for(&opts_for(b)).vec_kernels,
                "{b:?} executes on the CPU (offload tree-walks in the interpreter) and must keep \
                 the 256-bit recipes"
            );
        }
        assert!(
            !lower_options_for(&opts_for(BackendKind::GpuLower)).vec_kernels,
            "gpu-native lowers the whole program to PTX and cannot lower Win64 machine code"
        );
    }

    /// `--target-cpu` below x86-64-v3 must never receive the raw AVX2+FMA recipes: they are
    /// installed into the object verbatim, with no runtime dispatch, and would `#UD` on exactly the
    /// machines the level names.
    #[test]
    fn target_cpu_below_v3_clears_the_vec_kernel_recipes() {
        for (target, allowed) in [
            (TargetCpu::Native, true),
            (TargetCpu::X86_64, false),
            (TargetCpu::X86_64V2, false),
            (TargetCpu::X86_64V3, true),
            (TargetCpu::X86_64V4, true),
        ] {
            for backend in [BackendKind::Interp, BackendKind::Native] {
                let opts = Options {
                    backend,
                    target_cpu: target,
                    ..Options::default()
                };
                assert_eq!(
                    lower_options_for(&opts).vec_kernels,
                    allowed,
                    "{target:?} / {backend:?}"
                );
            }
        }
    }

    /// The property the option exists for, asserted on real MIR rather than on the flag: an
    /// unrecognized `@generic` f32 elementwise loop - the shape `tests/run/lang_generic_copy.wk`
    /// pins - lowers with **no** `VecKernelCall` when the target is gpu-native.
    ///
    /// This is a `cargo test --workspace` gate on purpose. The defect it pins was invisible to
    /// every CPU gate (the recipes are result-identical to the 128-bit path, so no differential or
    /// invariance test can see them) and reachable only by running the GPU backend on a host that
    /// has AVX2 - i.e. by gate part 10, on one machine.
    #[test]
    fn a_generic_f32_elementwise_loop_reaches_gpu_native_without_a_cpu_recipe() {
        const SRC: &str = "module t\n\
             \n\
             @generic\n\
             fn copy_generic(src: [f32; 8], mut dst: [f32; 8]) {\n\
                 for i in 0..8 { dst[i] = src[i]; }\n\
             }\n\
             \n\
             fn main() -> i32 {\n\
                 let src: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];\n\
                 let mut g: [f32; 8] = [0.0; 8];\n\
                 copy_generic(src, g);\n\
                 print(g[7] as i32);\n\
                 return 0;\n\
             }\n";

        assert_eq!(
            count_vec_kernels(SRC, lower_options_for(&opts_for(BackendKind::GpuLower))),
            0,
            "gpu-native MIR must carry no VecKernelCall - wukong_codegen_gpu declines it and the \
             program then fails with `UNSUPPORTED: VecKernelCall (CPU raw-AVX2 kernel) not lowered \
             to PTX`"
        );

        // The other arm is what makes the first one evidence rather than a tautology: where the
        // recipes can actually fire, the SAME program under the CPU options DOES carry one, so the
        // zero above is the backend's doing and not the program's. Skipped where the switch is
        // inert (no AVX2+FMA3, or the kill-switch set) - there is nothing to distinguish there.
        let can_fire = wukong_mir::host_supports_vec_kernels()
            && std::env::var_os("WUKONG_P4_NO_256").is_none();
        if can_fire {
            assert!(
                count_vec_kernels(SRC, lower_options_for(&opts_for(BackendKind::Native))) > 0,
                "this host assembles 256-bit recipes, so the CPU pipeline must still get one - if \
                 this fires, the elementwise recipe stopped claiming the loop and the arm above no \
                 longer proves anything"
            );
        }
    }
}

#[cfg(test)]
mod grad_cli_tests {
    use super::*;
    use wukong_mir::{MirLevel, Program};
    use wukong_span::{SourceMap, Symbol};

    #[test]
    fn verify_each_failure_preserves_pass_function_and_errors_in_both_formats() {
        let mut interner = Interner::new();
        let failure = wukong_opt::VerificationFailure {
            pass: "broken-pass",
            function: Some(interner.intern("the_function")),
            errors: vec!["undefined value v99".into(), "wrong return type".into()],
        };
        let diagnostic = optimization_failure_diag(&failure, &interner);
        assert_eq!(diagnostic.severity, wukong_diag::Severity::Bug);
        assert!(
            diagnostic.code.is_none(),
            "internal pass failures are not source error codes"
        );
        let sources = SourceMap::new();
        let human = Renderer::new(false).render(&diagnostic, &sources);
        let json = wukong_diag::to_json(&diagnostic, &sources);
        assert!(json.contains("\"severity\":\"bug\""));
        for text in [&human, &json] {
            for detail in [
                "broken-pass",
                "the_function",
                "undefined value v99",
                "wrong return type",
            ] {
                assert!(text.contains(detail), "{text}");
            }
        }
    }

    #[test]
    fn verify_each_checks_autodiff_and_training_synthesis() {
        let (program, mut interner) = compile_o1(
            "fn loss(x: [f32; 1], mut out: [f32; 1]) -> f32 { \
             let y: f32 = x[0] * x[0]; out[0] = y; return y; }",
        );
        for train_opt in [TrainOpt::Sgd, TrainOpt::AdamW] {
            let mut opts = Options {
                verify_each: true,
                ..Options::default()
            };
            opts.grad.train = true;
            opts.grad.train_steps = 1;
            opts.grad.train_opt = train_opt;
            build_grad(&program, &mut interner, &opts).unwrap();
            let checked = train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap();
            opts.verify_each = false;
            let ordinary = train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap();
            assert_eq!(checked, ordinary);
            assert_eq!(checked.len(), 2);
        }
    }

    #[test]
    fn autodiff_target_survives_every_optimization_level() {
        // Calling the loss from main used to make the O2/O3 inliner delete its definition.
        for target in ["loss", "objective"] {
            let source = format!(
                "fn {target}(x: [f32; 1], mut out: [f32; 1]) -> f32 {{ \
                 let y: f32 = x[0] * x[0]; out[0] = y; return y; }} \
                 fn main() -> i32 {{ let x: [f32; 1] = [2.0]; \
                 let mut out: [f32; 1] = [0.0]; {target}(x, out); return 0; }}"
            );
            // Establish that the fixture actually triggers deletion without an external root.
            let (mut unrooted, mut names) = compile_high(&source);
            wukong_opt::optimize(&mut unrooted, 2);
            assert!(unrooted.function(names.intern(target)).is_none());
            for level in 0..=3 {
                for verified in [false, true] {
                    for training in [false, true] {
                        let (mut program, mut interner) = compile_high(&source);
                        let mut opts = Options {
                            emit: if training {
                                EmitStage::Mir
                            } else {
                                EmitStage::Grad
                            },
                            opt_level: level,
                            verify_each: verified,
                            ..Options::default()
                        };
                        opts.grad.of = (target != "loss").then(|| target.to_string());
                        opts.grad.wrt = vec![0];
                        opts.grad.train = training;
                        opts.grad.train_steps = 1;
                        opts.grad.train_opt = TrainOpt::Sgd;
                        opts.grad.train_lr = 0.25;
                        let roots = autodiff_roots(&program, &mut interner, &opts).unwrap();
                        if verified {
                            wukong_opt::optimize_verified_preserving(
                                &mut program,
                                level.max(1),
                                &roots,
                            )
                            .unwrap();
                        } else {
                            wukong_opt::optimize_preserving(&mut program, level.max(1), &roots);
                        }
                        let fsym = interner.intern(target);
                        assert!(
                            program.function(fsym).is_some(),
                            "lost {target} at O{level}"
                        );
                        if level >= 2 {
                            let main = program.function(interner.intern("main")).unwrap();
                            assert!(!main.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                                matches!(i.op, wukong_mir::Op::Call { func, .. } if func == fsym)
                            }), "preservation must not disable inlining");
                        }
                        let (_, gradient) = build_grad(&program, &mut interner, &opts).unwrap();
                        let gradient_name = gradient.name;
                        program.funcs.push(gradient);
                        let mut buffers = vec![vec![2.0], vec![0.0], vec![0.0]];
                        run_f32(&program, gradient_name, &mut buffers, &interner);
                        assert_eq!(buffers[1], vec![4.0]);
                        assert_eq!(buffers[2], vec![4.0]);
                        program.funcs.pop();
                        if training {
                            let trajectory =
                                train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap();
                            let mut seed = opts.grad.train_seed;
                            let x = train_rand(&mut seed, 1)[0];
                            // One SGD step for x squared at lr=1/4 gives x/2 exactly.
                            assert_eq!(trajectory, vec![x * x, (x * 0.5) * (x * 0.5)]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn autodiff_missing_target_names_requested_entry_before_optimization() {
        let (program, mut interner) = compile_high("fn main() -> i32 { return 0; }");
        for training in [false, true] {
            let mut opts = Options {
                emit: EmitStage::Grad,
                ..Options::default()
            };
            opts.grad.train = training;
            opts.grad.of = Some("missing_objective".into());
            let error = autodiff_roots(&program, &mut interner, &opts).unwrap_err();
            assert!(
                error.contains("autodiff target `missing_objective`"),
                "{error}"
            );
            assert!(error.contains("--grad-of=<fn>"), "{error}");
            assert_eq!(
                build_grad(&program, &mut interner, &opts).unwrap_err(),
                error
            );
        }
        let opts = Options::default();
        assert!(autodiff_roots(&program, &mut interner, &opts)
            .unwrap()
            .is_empty());
    }

    /// Lex → parse → sema → mir_build → opt(1). Panics with the offending stage's diagnostics.
    fn compile_o1(src: &str) -> (Program, Interner) {
        let (mut program, interner) = compile_high(src);
        wukong_opt::optimize(&mut program, 1);
        (program, interner)
    }

    fn compile_high(src: &str) -> (Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("grad_test.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors: {sd:?}");
        let (program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors: {md:?}");
        (program, interner)
    }

    /// Drive the real `build_grad` path (`--grad-of`/`--grad-wrt`) and return the `{fwd, fwd_grad}`
    /// program, the forward and gradient function names, the **resolved** `wrt` list (so `[]`
    /// expands to every buffer parameter, matching the appended gradient buffers), and the interner.
    fn grad_prog(
        src: &str,
        of: &str,
        wrt: &[usize],
    ) -> (Program, Symbol, Symbol, Vec<usize>, Interner) {
        let (program, mut interner) = compile_o1(src);
        let opts = Options {
            grad: GradOptions {
                of: Some(of.to_string()),
                wrt: wrt.to_vec(),
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let fsym = interner.intern(of);
        let resolved = grad_wrt(
            program.function(fsym).expect("loss fn"),
            wrt,
            opts.grad.train,
        )
        .expect("wrt");
        let (fwd, g) = build_grad(&program, &mut interner, &opts).expect("build_grad failed");
        let (fname, gname) = (fwd.name, g.name);
        let prog = Program {
            funcs: vec![fwd, g],
            level: MirLevel::Low,
            statics: program.statics.clone(),
            externs: program.externs.clone(),
        };
        (prog, fname, gname, resolved, interner)
    }

    fn run_f32(prog: &Program, name: Symbol, bufs: &mut [Vec<f32>], it: &Interner) {
        let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
        wukong_interp::run_kernel_f32(prog, name, &mut views, it).expect("kernel run failed");
    }

    /// Run the gradient kernel over `inputs` (one zeroed gradient buffer appended per `wrt`) and
    /// return the gradient buffer for each `wrt` (as f64).
    fn analytic(
        prog: &Program,
        gname: Symbol,
        inputs: &[Vec<f32>],
        lens: &[usize],
        wrt: &[usize],
        it: &Interner,
    ) -> Vec<Vec<f64>> {
        let mut bufs = inputs.to_vec();
        for &wi in wrt {
            bufs.push(vec![0.0; lens[wi]]);
        }
        run_f32(prog, gname, &mut bufs, it);
        let np = inputs.len();
        (0..wrt.len())
            .map(|i| bufs[np + i].iter().map(|&v| v as f64).collect())
            .collect()
    }

    /// Central finite-difference gradient of `out[loss_out][0]` w.r.t. each `wrt` element.
    #[allow(clippy::too_many_arguments)]
    fn finite_diff(
        prog: &Program,
        fwd_name: Symbol,
        inputs: &[Vec<f32>],
        lens: &[usize],
        wrt: &[usize],
        loss_out: usize,
        eps: f32,
        it: &Interner,
    ) -> Vec<Vec<f64>> {
        let loss = |bufs: &mut [Vec<f32>]| -> f64 {
            run_f32(prog, fwd_name, bufs, it);
            bufs[loss_out][0] as f64
        };
        let mut out = Vec::new();
        for &wi in wrt {
            let mut g = vec![0.0; lens[wi]];
            for j in 0..lens[wi] {
                let mut bufs = inputs.to_vec();
                let orig = bufs[wi][j];
                bufs[wi][j] = orig + eps;
                let lp = loss(&mut bufs);
                bufs[wi][j] = orig - eps;
                let lm = loss(&mut bufs);
                g[j] = (lp - lm) / (2.0 * eps as f64);
            }
            out.push(g);
        }
        out
    }

    /// Full gate: analytic gradient (from the emitted backward) vs the finite difference, over every
    /// element. Returns the analytic gradients so a caller can additionally assert a closed form.
    #[allow(clippy::too_many_arguments)]
    fn gate(
        src: &str,
        of: &str,
        wrt: &[usize],
        inputs: &[Vec<f32>],
        lens: &[usize],
        loss_out: usize,
        eps: f32,
        rel: f64,
        abs: f64,
    ) -> Vec<Vec<f64>> {
        let (prog, fwd_name, gname, wrt, it) = grad_prog(src, of, wrt);
        let analytic = analytic(&prog, gname, inputs, lens, &wrt, &it);
        let fd = finite_diff(&prog, fwd_name, inputs, lens, &wrt, loss_out, eps, &it);
        for (gi, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
            for (j, (&av, &fv)) in a.iter().zip(f.iter()).enumerate() {
                let tol = abs + rel * fv.abs();
                assert!(
                    (av - fv).abs() <= tol,
                    "wrt#{gi}[{j}]: analytic {av} vs finite-diff {fv} (|d|={:.3e} > {:.3e})",
                    (av - fv).abs(),
                    tol
                );
            }
        }
        analytic
    }

    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (j, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 3e-3 + 3e-3 * w.abs(),
                "{what}[{j}]: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn cubic_scalar_grad() {
        // loss = x^3  ->  dL/dx = 3x^2. Differentiated straight from `.wk` source via --emit=grad.
        let src = "module m\nfn loss(x:[f32;1], mut out:[f32;1]) -> f32 {\n\
                   let a: f32 = x[0]; let l: f32 = a*a*a; out[0] = l; return l; }";
        let x = 1.3f32;
        let inputs = vec![vec![x], vec![0.0]];
        let g = gate(src, "loss", &[0], &inputs, &[1, 1], 1, 1e-2, 2e-2, 1e-3);
        assert_close(&g[0], &[3.0 * (x * x) as f64], "d(x^3)/dx");
    }

    #[test]
    fn raw_pointer_parameter_grad() {
        // A loss whose buffers are raw `*mut f32` parameters rather than `[f32; N]` arrays.
        //
        // Until `mem2reg` promoted pointer slots this could not be differentiated at all: the
        // parameter's base pointer reached autodiff as `load ptr <slot>`, and `Vjp::canon` refused
        // it — `--emit=grad` reported "cannot route gradient for load pointer v31 (not a parameter
        // or a one-level gep of a parameter)". With the slot promoted, the base pointer *is* the
        // parameter value, so every access is a one-level gep off a parameter and the gradient
        // routes. `--emit=grad` forces `-O1`, so the promotion is always in effect here.
        //
        // loss = sum_i (w_i * x_i)^2  ->  dL/dw_i = 2 w_i x_i^2, dL/dx_i = 2 w_i^2 x_i.
        let src = "module m\nfn loss(w: *mut f32, x: *mut f32, mut out:[f32;1]) -> f32 {\n\
                   let a: f32 = w[0]*x[0]; let b: f32 = w[1]*x[1]; let c: f32 = w[2]*x[2];\n\
                   let l: f32 = a*a + b*b + c*c; out[0] = l; return l; }";
        let w = [0.5f32, -1.25, 0.75];
        let x = [1.5f32, 0.25, -2.0];
        let inputs = vec![w.to_vec(), x.to_vec(), vec![0.0]];
        let g = gate(
            src,
            "loss",
            &[0, 1],
            &inputs,
            &[3, 3, 1],
            2,
            1e-2,
            3e-2,
            2e-3,
        );
        let dw: Vec<f64> = (0..3).map(|i| (2.0 * w[i] * x[i] * x[i]) as f64).collect();
        let dx: Vec<f64> = (0..3).map(|i| (2.0 * w[i] * w[i] * x[i]) as f64).collect();
        assert_close(&g[0], &dw, "dL/dw = 2 w x^2");
        assert_close(&g[1], &dx, "dL/dx = 2 w^2 x");
    }

    #[test]
    fn bilinear_two_input_grad() {
        // loss = a * b^2  ->  dL/da = b^2, dL/db = 2ab.
        let src = "module m\nfn loss(a:[f32;1], b:[f32;1], mut out:[f32;1]) -> f32 {\n\
                   let av: f32 = a[0]; let bv: f32 = b[0]; let l: f32 = av*bv*bv;\n\
                   out[0] = l; return l; }";
        let (a, b) = (0.7f32, 1.1f32);
        let inputs = vec![vec![a], vec![b], vec![0.0]];
        let g = gate(
            src,
            "loss",
            &[0, 1],
            &inputs,
            &[1, 1, 1],
            2,
            1e-2,
            3e-2,
            1e-3,
        );
        assert_close(&g[0], &[(b * b) as f64], "dL/da = b^2");
        assert_close(&g[1], &[(2.0 * a * b) as f64], "dL/db = 2ab");
    }

    #[test]
    fn default_wrt_is_all_buffers() {
        // With no --grad-wrt, every buffer parameter is differentiated; the output buffer (only
        // stored, never read) gets a correct zero gradient.
        let src = "module m\nfn loss(x:[f32;1], mut out:[f32;1]) -> f32 {\n\
                   let a: f32 = x[0]; let l: f32 = a*a; out[0] = l; return l; }";
        let x = 2.0f32;
        let inputs = vec![vec![x], vec![0.0]];
        // wrt defaults to [0, 1] (both buffers).
        let g = gate(src, "loss", &[], &inputs, &[1, 1], 1, 1e-2, 2e-2, 1e-3);
        assert_close(&g[0], &[2.0 * x as f64], "dL/dx = 2x");
        assert_close(&g[1], &[0.0], "dL/d(out) = 0");
    }

    // --- Recognized-kernel tape: a real linear + MSE model, differentiated from source. -----------

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*seed >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.8 // ~[-0.4, 0.4]
    }
    fn rand_vec(seed: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| lcg(seed)).collect()
    }

    /// f64 reference for `p = X·Wᵀ` (m×k · n×k → m×n), the MSE loss `Σ(p−t)²`, and its gradients
    /// `dX[m,k] = Σ_n 2(p−t)[m,n] W[n,k]`, `dW[n,k] = Σ_m 2(p−t)[m,n] X[m,k]`.
    fn linmse_ref(
        x: &[f32],
        w: &[f32],
        t: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut d = vec![0.0f64; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += x[mm * k + kk] as f64 * w[nn * k + kk] as f64;
                }
                d[mm * n + nn] = s - t[mm * n + nn] as f64;
            }
        }
        let mut dx = vec![0.0f64; m * k];
        let mut dw = vec![0.0f64; n * k];
        for mm in 0..m {
            for nn in 0..n {
                let g = 2.0 * d[mm * n + nn];
                for kk in 0..k {
                    dx[mm * k + kk] += g * w[nn * k + kk] as f64;
                    dw[nn * k + kk] += g * x[mm * k + kk] as f64;
                }
            }
        }
        (dx, dw)
    }

    /// `--emit=grad` of a **real** linear-regression model written in `.wk`: `p = x·wᵀ` (recognized
    /// as `wukong_sgemm_nt`) then the MSE reduction `Σ(p−t)²` (recognized as the `@parallel`
    /// `wukong_sreduce_f32_parallel`). The emitted backward rides the tuned kernels — `wukong_velem`
    /// for the `2(p−t)` seed, a transpose loop, and `wukong_sgemm` for `dw = dpᵀ·x` / `dx = dp·w` —
    /// and is finite-difference-gated against the forward loss and the f64 closed form.
    #[test]
    fn linear_mse_from_source() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let src = "@parallel fn loss(x:[f32;12], w:[f32;8], t:[f32;6], mut out:[f32;1]) -> f32 {\n\
                   let mut p: [f32; 6] = [0.0; 6];\n\
                   for i in 0..3 { for j in 0..2 { let mut s: f32 = 0.0;\n\
                     for kk in 0..4 { s = s + x[i*4+kk] * w[j*4+kk]; }\n\
                     p[i*2+j] = s; } }\n\
                   let mut loss: f32 = 0.0;\n\
                   for i in 0..6 { loss = loss + (p[i]-t[i])*(p[i]-t[i]); }\n\
                   out[0] = loss; return loss; }";
        let mut seed = 0xA11CEu64;
        let x = rand_vec(&mut seed, m * k);
        let w = rand_vec(&mut seed, n * k);
        let t = rand_vec(&mut seed, m * n);
        let (dx_ref, dw_ref) = linmse_ref(&x, &w, &t, m, k, n);
        let inputs = vec![x, w, t, vec![0.0]];
        let lens = [m * k, n * k, m * n, 1];

        // Weights (param 1): the trainable gradient — must ride the GEMM and match the closed form.
        let gw = gate(src, "loss", &[1], &inputs, &lens, 3, 5e-3, 5e-2, 5e-3);
        assert_close(&gw[0], &dw_ref, "dW = dpᵀ·x");

        // Input (param 0): dx = dp·w — the other GEMM adjoint.
        let gx = gate(src, "loss", &[0], &inputs, &lens, 3, 5e-3, 5e-2, 5e-3);
        assert_close(&gx[0], &dx_ref, "dX = dp·w");
    }

    // --- Activation backward: a `matmul → act → sum` model, act backward riding wukong_vmath2_f32. ---

    /// The number of GEMM adjoint calls in printed gradient MIR. The forward GEMMs here are all the
    /// `nn.Linear` spelling (`wukong_sgemm_nt*`), so every plain `wukong_sgemm*` and weight-gradient
    /// `wukong_sgemm_tn*` call — serial, or `@parallel` when the forward was — is an adjoint: `dX`
    /// rides the NN GEMM, `dW` the TN GEMM (no synthesized transpose).
    fn gemm_adjoints(mir: &str) -> usize {
        [
            "wukong_sgemm(",
            "wukong_sgemm_parallel(",
            "wukong_sgemm_tn(",
            "wukong_sgemm_tn_parallel(",
        ]
        .iter()
        .map(|name| mir.matches(name).count())
        .sum()
    }

    /// The printed MIR of `{loss, loss_grad}` for the given source, so a test can assert the emitted
    /// backward rides a particular runtime kernel (the recognized backward is otherwise gate-blind).
    fn grad_mir(src: &str, of: &str, wrt: &[usize]) -> String {
        let (prog, _fwd, _g, _wrt, it) = grad_prog(src, of, wrt);
        wukong_mir::print::print_program(&prog, &it)
    }

    /// `p = x·wᵀ; h = act(p); loss = Σ h` — the smallest model that exercises a nonlinear activation
    /// backward composed with the matmul backward.
    fn act_sum_src(act: &str) -> String {
        format!(
            "@parallel fn loss(x:[f32;12], w:[f32;8], mut out:[f32;1]) -> f32 {{\n\
             let mut p: [f32; 6] = [0.0; 6];\n\
             for i in 0..3 {{ for j in 0..2 {{ let mut s: f32 = 0.0;\n\
               for kk in 0..4 {{ s = s + x[i*4+kk] * w[j*4+kk]; }}\n\
               p[i*2+j] = s; }} }}\n\
             let mut h: [f32; 6] = [0.0; 6];\n\
             for i in 0..6 {{ h[i] = {act}(p[i]); }}\n\
             let mut loss: f32 = 0.0;\n\
             for i in 0..6 {{ loss = loss + h[i]; }}\n\
             out[0] = loss; return loss; }}"
        )
    }

    /// silu and gelu — the modern-transformer activations — differentiate from source: the backward
    /// rides the fused `wukong_vmath2_f32` (`dx = dy·act'(x)`, bit-identical with the forward) for the
    /// activation and `wukong_sgemm` for the matmul adjoint. Finite-difference-gated against the
    /// forward loss (the runtime activation's own derivative is the oracle, so no closed form to
    /// re-derive — the fused kernel and the forward share the same sigmoid/tanh polynomials).
    #[test]
    fn silu_and_gelu_backward_from_source() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let lens = [m * k, n * k, 1];
        for act in ["silu", "gelu"] {
            let src = act_sum_src(act);
            // The backward must ride the fused activation kernel + the tuned GEMM.
            let mir = grad_mir(&src, "loss", &[1]);
            assert!(
                mir.contains("wukong_vmath2_f32"),
                "{act}: backward does not ride wukong_vmath2_f32\n{mir}"
            );
            assert!(
                gemm_adjoints(&mir) > 0,
                "{act}: matmul backward does not ride the tuned GEMM\n{mir}"
            );
            // Data kept modest so the activations stay in a well-conditioned range for the difference.
            let mut seed = 0xE1F5u64 ^ (act.as_bytes()[0] as u64);
            let x = rand_vec(&mut seed, m * k);
            let w = rand_vec(&mut seed, n * k);
            let inputs = vec![x, w, vec![0.0]];
            // wrt w and wrt x — both adjoints flow through the activation backward.
            gate(&src, "loss", &[1], &inputs, &lens, 2, 1e-2, 4e-2, 1e-2);
            gate(&src, "loss", &[0], &inputs, &lens, 2, 1e-2, 4e-2, 1e-2);
        }
    }

    // --- --train: fwd → bwd → optimizer, the end-to-end loss-decrease signal. -----------------------

    /// A least-squares linear regression `loss = Σ(x·wᵀ − t)²` — convex in the weights `w`, so
    /// full-batch gradient descent with a small step descends monotonically.
    const REGRESSION_SRC: &str =
        "@parallel fn loss(x:[f32;12], w:[f32;8], t:[f32;6], mut out:[f32;1]) -> f32 {\n\
         let mut p: [f32; 6] = [0.0; 6];\n\
         for i in 0..3 { for j in 0..2 { let mut s: f32 = 0.0;\n\
           for kk in 0..4 { s = s + x[i*4+kk] * w[j*4+kk]; }\n\
           p[i*2+j] = s; } }\n\
         let mut loss: f32 = 0.0;\n\
         for i in 0..6 { loss = loss + (p[i]-t[i])*(p[i]-t[i]); }\n\
         out[0] = loss; return loss; }";

    #[allow(clippy::too_many_arguments)]
    fn train_traj(
        src: &str,
        wrt: &[usize],
        lens: &[usize],
        steps: usize,
        lr: f32,
        opt: TrainOpt,
        seed: u64,
    ) -> Vec<f32> {
        let (program, mut interner) = compile_o1(src);
        let opts = Options {
            grad: GradOptions {
                of: Some("loss".to_string()),
                wrt: wrt.to_vec(),
                train: true,
                train_steps: steps,
                train_lr: lr,
                train_weight_decay: 0.0,
                train_opt: opt,
                train_seed: seed,
            },
            ..Options::default()
        };
        train_loop(&program, &mut interner, &opts, lens).expect("train_loop")
    }

    /// The loss function's buffer sizes are recovered from the semantic types (`[f32; N]`), not the
    /// `Ptr`-erased MIR — so `--train` can allocate the right buffers with no user-supplied shapes.
    #[test]
    fn param_sizes_recovered_from_types() {
        let mut sm = SourceMap::new();
        let id = sm.add("sz.wk".to_string(), REGRESSION_SRC.to_string());
        let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()));
        let opts = Options {
            grad: GradOptions {
                of: Some("loss".to_string()),
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let lens = loss_param_lens(&sema, &mut interner, &opts).expect("param sizes");
        assert_eq!(lens, vec![12, 8, 6, 1], "x[12], w[8], t[6], out[1]");
    }

    #[test]
    fn train_sgd_loss_strictly_decreases() {
        // Full-batch GD on a convex least-squares objective: the loss never rises (f32 rounding slack).
        let traj = train_traj(
            REGRESSION_SRC,
            &[1],
            &[12, 8, 6, 1],
            120,
            0.05,
            TrainOpt::Sgd,
            0xA5A5,
        );
        for w in traj.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "SGD loss rose: {} -> {}", w[0], w[1]);
        }
        let (l0, lf) = (traj[0], *traj.last().unwrap());
        assert!(
            l0 > 0.05,
            "test setup: initial loss should be substantial, got {l0}"
        );
        assert!(
            lf < 0.3 * l0,
            "SGD did not reduce the loss enough: {l0} -> {lf}"
        );
    }

    /// The documented `--grad-wrt` default ("all buffer parameters of the loss function ... for
    /// `--train`, these are the trainable weights") must actually run. It used to expand to a list
    /// that necessarily contained the loss-output parameter, which `train_loop` then rejected, so
    /// `wukongc --train --grad-of=loss` always died with "--grad-wrt must not include the
    /// loss-output parameter (the last one)" unless an explicit list was supplied.
    #[test]
    fn train_default_wrt_excludes_the_loss_output() {
        let traj = train_traj(
            REGRESSION_SRC,
            &[],
            &[12, 8, 6, 1],
            40,
            0.02,
            TrainOpt::Sgd,
            0xA5A5,
        );
        assert_eq!(
            traj.len(),
            41,
            "traj[i] = loss after i updates, plus a final"
        );
        let (l0, lf) = (traj[0], *traj.last().unwrap());
        assert!(
            lf < l0,
            "the default --grad-wrt did not train: {l0} -> {lf}"
        );
    }

    /// A repeated `--grad-wrt` index appends a second gradient buffer to the backward's parameter
    /// list while `Vjp`'s param -> buffer map keeps only the last insert, so the first appended
    /// buffer is never written and the caller reads back zeros with exit 0. Reject it up front.
    #[test]
    fn duplicate_grad_wrt_index_is_rejected() {
        let (program, mut interner) = compile_o1(REGRESSION_SRC);
        let opts = Options {
            grad: GradOptions {
                of: Some("loss".to_string()),
                wrt: vec![1, 1],
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let err = build_grad(&program, &mut interner, &opts)
            .expect_err("a repeated --grad-wrt index must be rejected");
        assert!(err.contains("repeated"), "unexpected error: {err}");
    }

    #[test]
    fn train_adamw_reduces_loss() {
        // The fused AdamW kernel drives a much larger reduction than plain SGD in the same budget.
        let traj = train_traj(
            REGRESSION_SRC,
            &[1],
            &[12, 8, 6, 1],
            150,
            0.05,
            TrainOpt::AdamW,
            0xA5A5,
        );
        let (l0, lf) = (traj[0], *traj.last().unwrap());
        assert!(
            lf.is_finite() && lf < 0.05 * l0,
            "AdamW did not converge: {l0} -> {lf}"
        );
    }

    // --- Transformer FFN block: two matmuls + activation + reduction, differentiated from source. ---

    /// A transformer **feed-forward block** `y = silu(x·W1ᵀ)·W2ᵀ; loss = Σy`, the composition at the
    /// heart of every transformer layer. Its whole backward rides tuned kernels: `wukong_vmath2_f32`
    /// for the activation and *three* `wukong_sgemm` for the matmul adjoints (dW2, dA, dW1) — the
    /// headline capability, a real block differentiated straight from `.wk` and finite-difference-gated.
    #[test]
    fn transformer_ffn_block_from_source() {
        // x[M=2,K=4], W1[H=3,K=4] -> p[2,3]; a=silu(p); W2[N=2,H=3] -> y[2,2]; loss=Σy.
        let src =
            "@parallel fn loss(x:[f32;8], w1:[f32;12], w2:[f32;6], mut out:[f32;1]) -> f32 {\n\
             let mut p: [f32; 6] = [0.0; 6];\n\
             for i in 0..2 { for j in 0..3 { let mut s: f32 = 0.0;\n\
               for kk in 0..4 { s = s + x[i*4+kk] * w1[j*4+kk]; } p[i*3+j] = s; } }\n\
             let mut a: [f32; 6] = [0.0; 6];\n\
             for i in 0..6 { a[i] = silu(p[i]); }\n\
             let mut y: [f32; 4] = [0.0; 4];\n\
             for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0;\n\
               for kk in 0..3 { s = s + a[i*3+kk] * w2[j*3+kk]; } y[i*2+j] = s; } }\n\
             let mut loss: f32 = 0.0;\n\
             for i in 0..4 { loss = loss + y[i]; }\n\
             out[0] = loss; return loss; }";
        // The composite backward must ride the fused activation kernel + the tuned GEMM (three of them).
        let mir = grad_mir(src, "loss", &[1, 2]);
        assert!(
            mir.contains("wukong_vmath2_f32"),
            "silu backward missing\n{mir}"
        );
        assert_eq!(
            gemm_adjoints(&mir),
            3,
            "expected 3 GEMM adjoints (dW2, dA, dW1)\n{mir}"
        );

        let mut seed = 0x77A0u64;
        let x = rand_vec(&mut seed, 8);
        let w1 = rand_vec(&mut seed, 12);
        let w2 = rand_vec(&mut seed, 6);
        let inputs = vec![x, w1, w2, vec![0.0]];
        let lens = [8, 12, 6, 1];
        // Finite-difference-gate the composite gradient w.r.t. both weight matrices and the input.
        gate(src, "loss", &[1], &inputs, &lens, 3, 1e-2, 5e-2, 1e-2); // dW1
        gate(src, "loss", &[2], &inputs, &lens, 3, 1e-2, 5e-2, 1e-2); // dW2
        gate(src, "loss", &[0], &inputs, &lens, 3, 1e-2, 5e-2, 1e-2); // dX
    }
}

/// The MIR-verifier gate that stands in front of every backend entry.
#[cfg(test)]
mod verify_gate_tests {
    use super::*;
    use wukong_mir::{BasicBlock, BlockId, Function, MirType, Program, Terminator, ValueId};

    /// A function that returns a value no block parameter and no instruction ever defines — the
    /// "use of undefined value" class the verifier exists to catch.
    fn invalid_program(interner: &mut Interner) -> Program {
        let f = Function {
            name: interner.intern("bad"),
            params: Vec::new(),
            ret: MirType::I32,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: Vec::new(),
                term: Terminator::Ret(Some(ValueId(0))),
            }],
            value_types: vec![MirType::I32],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
            param_attrs: Vec::new(),
        };
        Program {
            funcs: vec![f],
            statics: Vec::new(),
            externs: Vec::new(),
            level: wukong_mir::MirLevel::Low,
        }
    }

    /// `verify_or_ice` is what `--run`, `--emit=llvm-ir`, `--emit=obj` and `--emit=exe` now all call
    /// before entering a backend; before this it was inlined in the `--run` arm only, so the three
    /// AOT exits reached their backend unverified.
    #[test]
    fn verify_or_ice_reports_an_undefined_value_use() {
        let mut interner = Interner::new();
        assert!(
            verify_or_ice(&invalid_program(&mut interner)),
            "a `ret` of an undefined value must be reported"
        );
    }

    #[test]
    fn verify_or_ice_checks_cross_function_signatures() {
        use wukong_mir::{Builder, Op};
        let mut interner = Interner::new();
        let callee_name = interner.intern("callee");
        let mut callee = Builder::new(callee_name, MirType::Void);
        callee.add_param(MirType::Ptr);
        callee.ret(None);
        let mut caller = Builder::new(interner.intern("main"), MirType::Void);
        caller.build_void(Op::Call {
            func: callee_name,
            args: vec![],
        });
        caller.ret(None);
        let program = Program {
            funcs: vec![callee.finish(), caller.finish()],
            statics: vec![],
            externs: Vec::new(),
            level: wukong_mir::MirLevel::Low,
        };
        assert!(program
            .funcs
            .iter()
            .all(|f| wukong_mir::verify::verify_function(f).is_empty()));
        assert!(verify_or_ice(&program));
    }

    /// And it must stay quiet on well-formed MIR, or every AOT compile would start failing.
    #[test]
    fn verify_or_ice_accepts_a_real_compiled_program() {
        let src = "module m\nfn main() -> i32 { let x: f32 = 0.5; print(x); return 0; }";
        let mut sm = SourceMap::new();
        let id = sm.add("v.wk".to_string(), src.to_string());
        let (tokens, _) = wukong_lexer::tokenize(sm.source(id), id);
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
        for level in [0u8, 1, 2] {
            let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
            assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
            wukong_opt::optimize(&mut program, level);
            assert!(
                !verify_or_ice(&program),
                "clean program failed verify at -O{level}"
            );
        }
    }
}

/// The `--emit=exe` link path's generated inputs (the Rust shim, the C fallback runtime).
#[cfg(test)]
mod assertion_link_tests;

#[cfg(test)]
mod generated_verify_tests;

#[cfg(test)]
mod grad_model_tests;

#[cfg(test)]
mod cc_print_tests;

#[cfg(test)]
mod native_link_tests {
    use super::*;

    #[test]
    fn runtime_archive_selection_preserves_metadata_failure_when_retry_cannot_spawn() {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;

        let candidates = || {
            ["first", "second"]
                .into_iter()
                .map(|name| RlibCandidate {
                    path: PathBuf::from(name),
                    uplifted: false,
                    mtime: secs(1),
                })
                .collect()
        };
        let mut attempts = 0;
        let diagnostic = b"error[E0514]: incompatible runtime compiler version\n";
        let (path, failure) = try_runtime_rlibs(candidates(), |_| {
            attempts += 1;
            if attempts == 1 {
                Ok(std::process::Output {
                    status: std::process::ExitStatus::from_raw(256),
                    stdout: Vec::new(),
                    stderr: diagnostic.to_vec(),
                })
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "rustc removed during retry",
                ))
            }
        })
        .expect("a previous rustc rejection must remain a compiler failure")
        .unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(path, PathBuf::from("first"));
        assert!(!failure.status.success());
        assert_eq!(failure.stderr, diagnostic);

        // Before rustc produces any diagnostic, inability to start it still permits fallback.
        let mut attempts = 0;
        let unavailable = try_runtime_rlibs(candidates(), |_| {
            attempts += 1;
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "rustc unavailable",
            ))
        })
        .expect_err("initial spawn failure must remain unavailable");
        assert_eq!(attempts, 1);
        assert_eq!(unavailable.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn runtime_archive_selection_retries_a_newer_incompatible_compiler() {
        use std::process::Command;
        let scratch = ScratchDir::new("rlib_compatibility").expect("scratch directory");
        let dir = &scratch.path;
        let deps = dir.join("deps");
        std::fs::create_dir_all(&deps).unwrap();
        let source = dir.join("runtime.rs");
        std::fs::write(&source, "pub fn answer() -> u32 { 42 }\n").unwrap();
        let good = dir.join("libwukong_runtime.rlib");
        let rustc = link_rustc();
        let built = Command::new(&rustc)
            .args([
                "--crate-name",
                "wukong_runtime",
                "--crate-type",
                "rlib",
                "--edition",
                "2021",
            ])
            .arg(&source)
            .arg("-o")
            .arg(&good)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );

        // Change only rustc's embedded version text in a real archive, retaining its structure.
        // This produces a real E0514 from the pinned compiler without installing a second one.
        let mut obsolete = std::fs::read(&good).unwrap();
        let version_offsets: Vec<usize> = obsolete
            .windows(6)
            .enumerate()
            .filter_map(|(i, bytes)| (bytes == b"rustc ").then_some(i + 6))
            .collect();
        assert!(
            !version_offsets.is_empty(),
            "archive has no compiler version metadata"
        );
        for offset in version_offsets {
            obsolete[offset] = b'0';
        }
        let bad = deps.join("libwukong_runtime-incompatible.rlib");
        std::fs::write(&bad, &obsolete).unwrap();
        for (path, when) in [(&good, secs(1_700_000_100)), (&bad, secs(1_700_000_200))] {
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(when))
                .unwrap();
        }
        assert_eq!(
            runtime_rlib_in(dir),
            Some(bad.clone()),
            "newer archive must be attempted first"
        );
        let probe = dir.join("probe.rs");
        std::fs::write(
            &probe,
            "extern crate wukong_runtime; pub fn answer() -> u32 { wukong_runtime::answer() }\n",
        )
        .unwrap();
        let mut attempted = Vec::new();
        let (selected, output) = try_runtime_rlibs(runtime_rlib_candidates(dir), |rlib| {
            attempted.push(rlib.to_path_buf());
            let output = Command::new(&rustc)
                .args([
                    "--crate-type",
                    "rlib",
                    "--emit",
                    "metadata",
                    "--edition",
                    "2021",
                ])
                .arg(&probe)
                .arg("--extern")
                .arg(format!("wukong_runtime={}", rlib.display()))
                .arg("-o")
                .arg(dir.join("probe.rmeta"))
                .output()?;
            if rlib == bad {
                assert!(
                    String::from_utf8_lossy(&output.stderr).contains("error[E0514]"),
                    "expected actual rustc version rejection: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Ok(output)
        })
        .unwrap()
        .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(selected, good);
        assert_eq!(attempted, [bad.clone(), good.clone()]);

        // Exhaustion returns a real failure; it cannot become the C-runtime fallback or success.
        std::fs::write(&good, &obsolete).unwrap();
        let mut attempts = 0;
        let (_, failed) = try_runtime_rlibs(runtime_rlib_candidates(dir), |rlib| {
            attempts += 1;
            Command::new(&rustc)
                .args(["--crate-type", "rlib", "--emit", "metadata"])
                .arg(&probe)
                .arg("--extern")
                .arg(format!("wukong_runtime={}", rlib.display()))
                .arg("-o")
                .arg(dir.join("probe.rmeta"))
                .output()
        })
        .unwrap()
        .unwrap();
        assert_eq!(attempts, 2);
        assert!(!failed.status.success());
        assert!(incompatible_runtime_metadata(&failed.stderr));
    }

    #[test]
    fn runtime_archive_selection_preserves_non_metadata_errors() {
        let scratch = ScratchDir::new("rlib_error").expect("scratch directory");
        let source = scratch.path.join("broken.rs");
        std::fs::write(&source, "pub fn unfinished( {\n").unwrap();
        let candidates = ["first", "second"]
            .into_iter()
            .map(|name| RlibCandidate {
                path: PathBuf::from(name),
                uplifted: false,
                mtime: secs(1),
            })
            .collect();
        let mut attempts = 0;
        let (selected, output) = try_runtime_rlibs(candidates, |_| {
            attempts += 1;
            std::process::Command::new(link_rustc())
                .args(["--crate-type", "rlib", "--emit", "metadata"])
                .arg(&source)
                .arg("-o")
                .arg(scratch.path.join("broken.rmeta"))
                .output()
        })
        .unwrap()
        .unwrap();
        assert!(!output.status.success());
        assert_eq!(
            attempts, 1,
            "a genuine compilation error must not select an older runtime"
        );
        assert_eq!(selected, PathBuf::from("first"));
        assert!(!incompatible_runtime_metadata(&output.stderr));
    }

    /// They must never be generated in the directory the user invoked the compiler from:
    /// `{stem}_shim.rs` was written there and then unconditionally deleted, so an `--emit=exe` run
    /// in a source directory holding a file of that name destroyed it.
    #[test]
    fn exe_link_inputs_are_generated_outside_the_working_directory() {
        let cwd = std::env::current_dir().expect("cwd");
        let scratch = ScratchDir::new("scratch_probe").expect("scratch dir");
        let shim = scratch_path(Some(&scratch), "scratch_probe_shim.rs".to_string());
        let rt = scratch_path(Some(&scratch), "scratch_probe_rt.c".to_string());
        for p in [&shim, &rt] {
            assert!(
                !p.starts_with(&cwd),
                "`{}` lands in the working directory",
                p.display()
            );
        }
        std::fs::write(&shim, "// probe").expect("write shim");
        let dir = scratch.path.clone();
        drop(scratch);
        assert!(
            !dir.exists(),
            "the scratch directory must be removed when the link finishes"
        );
    }

    /// The `--emit=exe` OBJECT is a link intermediate too, and from `ScratchDir`'s introduction
    /// until now it was the one such input `ScratchDir` did not cover: a bare relative `<stem>.o`
    /// written into the invocation directory and never deleted. Eight of those leftovers are sitting
    /// in this repository's root, unseen behind `.gitignore`'s `*.o`; and by construction, not by a
    /// recorded sighting, two concurrent `wukongc --emit=exe foo.wk` runs from one directory write
    /// the same relative path between the write and the linker's read, so the loser links the
    /// winner's object - a wrong-object link rather than an error.
    ///
    /// `--emit=obj` is the other half of the same decision and must keep writing where the user
    /// asked, because there the object IS the artifact.
    #[test]
    fn the_exe_objects_intermediate_lands_in_the_scratch_directory() {
        let scratch = ScratchDir::new("objpath_probe").expect("scratch dir");

        let exe_obj = object_path(false, None, "prog", Some(&scratch));
        assert!(
            exe_obj.starts_with(&scratch.path),
            "the `--emit=exe` object must be a scratch intermediate, got `{}`",
            exe_obj.display()
        );

        // `-o` names the EXECUTABLE under `--emit=exe`, never the object, so it must not divert it.
        let exe_obj_with_o = object_path(
            false,
            Some(Path::new("elsewhere/prog.exe")),
            "prog",
            Some(&scratch),
        );
        assert_eq!(
            exe_obj, exe_obj_with_o,
            "`-o` must not move the `--emit=exe` object: it names the executable"
        );

        // `--emit=obj`: the user's artifact, at the user's path (or `<stem>.o` beside them).
        assert_eq!(
            object_path(true, Some(Path::new("out/thing.o")), "prog", Some(&scratch)),
            PathBuf::from("out/thing.o"),
            "`--emit=obj -o <path>` must write exactly there"
        );
        assert_eq!(
            object_path(true, None, "prog", Some(&scratch)),
            PathBuf::from("prog.o"),
            "`--emit=obj` without `-o` must keep the historical `<stem>.o`"
        );

        // No scratch directory could be created: the historical relative path is the fallback, the
        // same one `scratch_path` uses, so the link still works.
        assert_eq!(
            object_path(false, None, "prog", None),
            PathBuf::from("prog.o"),
            "with no scratch directory the object falls back to `<stem>.o`"
        );
    }

    /// **A structural invariant, not a behavioural one.** The `--emit=exe` link must resolve its
    /// `rustc` in exactly one place - [`link_rustc`] - because a second spawn site is how the defect
    /// this closes comes back: a child spawned by bare name inherits the caller's working directory,
    /// and rustup's `PATH` shim resolves a toolchain from *that* directory rather than from the one
    /// that built the rlib being linked, which is `E0514`.
    ///
    /// The scan reads this file's own text, so a future editor who writes the bare spelling ANYWHERE
    /// in it - including inside a comment or a string - fails this test. That is deliberate: a
    /// tripwire that also fires on prose is cheaper than one that misses the next real call site.
    #[test]
    fn the_link_toolchain_is_resolved_in_exactly_one_place() {
        let this_file = include_str!("lib.rs");
        let bare = [
            "Command::new(\"rustc\")",
            "Command::new(\"rustc\".to_string())",
        ];
        for needle in bare {
            assert!(
                !this_file.contains(needle),
                "`{needle}` spawns whatever `PATH` resolves, which is the E0514 mechanism; the AOT link must go through `link_rustc()`. (If this fired on a COMMENT rather than on code, reword the comment - the scan is textual on purpose.)"
            );
        }
        // The needle is assembled rather than written out, because the scan reads THIS file: a
        // verbatim literal would match itself and make the count 2. It guards the scan above from
        // passing vacuously - if the resolver is deleted and its body inlined at a new spawn site
        // under a different spelling, the `Command::new` scan alone might not notice.
        assert_eq!(
            this_file.matches(concat!("fn ", "link_rustc()")).count(),
            1,
            "there must be exactly one link-toolchain resolver"
        );
    }

    /// [`link_rustc`] must name something runnable. When `build.rs` recorded cargo's `RUSTC` and
    /// that file is still there, it is used; otherwise the historical `PATH` lookup is the fallback,
    /// so a relocated or installed `wukongc` keeps linking.
    #[test]
    fn the_link_toolchain_is_a_real_file_or_the_path_fallback() {
        let rustc = link_rustc();
        assert!(
            rustc.is_file() || rustc == Path::new("rustc"),
            "`link_rustc()` must return an existing binary or the bare `PATH` fallback, got `{}`",
            rustc.display()
        );
        if let Some(recorded) = option_env!("WUKONG_BUILD_RUSTC") {
            if Path::new(recorded).is_file() {
                assert_eq!(
                    rustc,
                    Path::new(recorded),
                    "a recorded, existing `RUSTC` must be preferred over the `PATH` lookup"
                );
            }
        }
    }

    /// The `cc` fallback must link `libm`. `WUKONG_RT_C` always defines `wukong_rt_fmod_*` over
    /// `fmod`/`fmodf`, which glibc keeps in `libm` and `cc` does not link by default — so the whole
    /// fallback path failed to link *every* program on Linux with `undefined reference to 'fmod'`.
    /// A library must also follow the objects that reference it, or GNU `ld` discards it unused.
    #[test]
    fn the_cc_fallback_links_libm_after_its_objects() {
        let args = cc_link_args(
            Path::new("prog.o"),
            Path::new("prog_rt.c"),
            Path::new("prog"),
        );
        let lm = args
            .iter()
            .position(|a| a == "-lm")
            .expect("the `cc` fallback link line must pass `-lm`");
        let rt = args
            .iter()
            .position(|a| a == "prog_rt.c")
            .expect("the generated C runtime must be on the link line");
        assert!(
            lm > rt,
            "`-lm` must follow `prog_rt.c` (GNU ld resolves left to right): {args:?}"
        );
        assert!(
            args.iter().any(|a| a == "-ffp-contract=off"),
            "the scalar trig mirror requires explicit FMA boundaries: {args:?}"
        );
    }

    /// The runtime rlib must be found in `deps/` under its hash-suffixed name, not only under the
    /// uplifted `libwukong_runtime.rlib`. Cargo uplifts a library only when it is a root unit of a
    /// **`build`**, so after `cargo test --workspace` (what CI runs) or `cargo run -p wukongc` the
    /// uplifted copy does not exist and only `deps/` has it — and `--emit=exe` silently degraded to
    /// the `cc` fallback, which cannot resolve `wukong_*` at all.
    ///
    /// And when BOTH exist the newest wins, uplifted or not. This half of the assertion was the
    /// opposite until W1b: the uplifted copy was returned from an `is_file()` test placed before the
    /// `deps/` scan, so a six-week-old leftover of some earlier `cargo build` beat the rlib cargo
    /// had just written. The uplifted copy still wins a TIE, which is the case that matters in a
    /// healthy tree because cargo hard-links the uplift.
    #[test]
    fn the_runtime_rlib_is_found_in_the_deps_directory() {
        let dir = std::env::temp_dir().join(format!("wukong_rlib_probe-{}", std::process::id()));
        let deps = dir.join("deps");
        // A pid can be recycled, and a previous *failed* run leaves its files behind — start clean
        // so the "nothing matches" assertions below cannot be poisoned by an earlier process.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&deps).expect("create probe layout");

        // Nothing at all: the `cc` fallback is genuinely all there is.
        assert_eq!(runtime_rlib_in(&dir), None, "empty layout must not match");

        // Decoys: another crate's rlib, a same-prefixed crate name, and a non-rlib artifact.
        for decoy in [
            "libwukong_interp-1111111111111111.rlib",
            "libwukong_runtime_extra-2222222222222222.rlib",
            "libwukong_runtime-3333333333333333.rmeta",
        ] {
            std::fs::write(deps.join(decoy), b"x").expect("write decoy");
        }
        assert_eq!(runtime_rlib_in(&dir), None, "a decoy must not match");

        // The cargo-test / cargo-run layout: hash-suffixed, in `deps/` only.
        let hashed = deps.join("libwukong_runtime-4444444444444444.rlib");
        write_rlib_at(&hashed, secs(1_700_000_100));
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(hashed.clone()),
            "the hash-suffixed rlib in `deps/` must be found"
        );

        // A FRESH uplifted copy still wins: it is what a plain `cargo build` just produced.
        let uplifted = dir.join("libwukong_runtime.rlib");
        write_rlib_at(&uplifted, secs(1_700_000_200));
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(uplifted.clone()),
            "a newer uplifted rlib must win"
        );

        // A TIE goes to the uplifted copy, and the tie is the normal case rather than a curiosity:
        // cargo hard-links the uplift, so the two paths are usually one inode with one mtime.
        write_rlib_at(&uplifted, secs(1_700_000_100));
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(uplifted.clone()),
            "at equal mtime the uplifted rlib must win"
        );

        // THE DEFECT, W1b: a STALE uplifted copy used to short-circuit the `deps/` scan and be
        // returned unconditionally. On this repository's own checkout the top-level copy was six
        // weeks old and the release AOT link failed `E0514` until `cargo build --release -p
        // wukong_runtime` refreshed it. The newest rlib must win whichever of the two places it
        // is in.
        write_rlib_at(&uplifted, secs(1_700_000_050));
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(hashed),
            "a STALE uplifted rlib must lose to a newer one in `deps/`"
        );

        // Among several `deps/` entries - cargo garbage-collects none of them - the newest wins.
        let newer = deps.join("libwukong_runtime-5555555555555555.rlib");
        write_rlib_at(&newer, secs(1_700_000_300));
        assert_eq!(
            runtime_rlib_in(&dir),
            Some(newer),
            "the newest `deps/` rlib must win"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `SystemTime` from a fixed epoch offset, so the ordering under test is set explicitly rather
    /// than inferred from how fast the test happens to write files.
    fn secs(n: u64) -> std::time::SystemTime {
        std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(n)
    }

    /// Write a stand-in rlib with an exact modification time.
    fn write_rlib_at(path: &Path, mtime: std::time::SystemTime) {
        std::fs::write(path, b"x").expect("write rlib");
        std::fs::File::options()
            .write(true)
            .open(path)
            .expect("open rlib")
            .set_modified(mtime)
            .expect("set rlib mtime");
    }

    /// Assemble a `!<arch>` archive from `(name, data)` members, so the bitcode probe can be tested
    /// on exactly the two shapes cargo emits without needing either profile's real 4–65 MB rlib.
    fn synth_archive(members: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"!<arch>\n".to_vec();
        for (name, data) in members {
            out.extend_from_slice(format!("{name:<16}").as_bytes());
            out.extend_from_slice(b"0           "); // mtime[12]
            out.extend_from_slice(b"0     "); // uid[6]
            out.extend_from_slice(b"0     "); // gid[6]
            out.extend_from_slice(b"100644  "); // mode[8]
            out.extend_from_slice(format!("{:<10}", data.len()).as_bytes());
            out.extend_from_slice(b"`\n");
            out.extend_from_slice(data);
            if data.len() % 2 == 1 {
                out.push(b'\n'); // members are padded to an even offset
            }
        }
        out
    }

    /// The release AOT link's whole diagnosis in one probe: an rlib whose code members are LLVM
    /// bitcode (what `[profile.release] lto = "thin"` makes cargo emit for a dependency, via
    /// `-C linker-plugin-lto`) must be recognized, because handing one to MSVC `link.exe` is
    /// `LNK1136: invalid or corrupt file` and then every `wukong_*` kernel reads as an unresolved
    /// external — the state in which the AOT gate scored 0/4 in release and 4/4 in debug.
    ///
    /// The two negative cases are the ones that would silently break the *working* debug path if the
    /// probe over-fired: a dev-profile rlib (plain objects), and the `ObjectAndBitcode` shape, whose
    /// bitcode lives in a `.llvmbc` section *inside* a real object and must therefore still link
    /// without LTO. Note both shapes carry a COFF-wrapped `lib.rmeta` on an MSVC target, which is why
    /// the probe asks "is any member bitcode", never "is any member native".
    #[test]
    fn an_lto_profile_rlib_is_recognized_as_holding_bitcode_members() {
        let dir = std::env::temp_dir().join(format!("wukong_bc_probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create probe dir");

        // A COFF object header (machine 0x8664, 0 sections) — what a native member starts with, and
        // what the MSVC `lib.rmeta` wrapper starts with in BOTH shapes.
        let coff = |tail: &[u8]| {
            let mut v = vec![0x64u8, 0x86, 0x00, 0x00];
            v.extend_from_slice(tail);
            v
        };
        let bitcode = |tail: &[u8]| {
            let mut v = LLVM_BITCODE_MAGIC.to_vec();
            v.extend_from_slice(tail);
            v
        };

        /// `(what the shape is, its archive members, whether the probe must call it bitcode)`.
        type Case<'a> = (&'a str, Vec<(&'a str, Vec<u8>)>, bool);

        let cases: [Case; 6] = [
            (
                "release: rmeta wrapper + a bitcode code member",
                vec![
                    ("lib.rmeta", coff(b"metadata")),
                    ("lib.rmeta-link", coff(b"x")),
                    ("crate.rcgu.o", bitcode(b"module body")),
                ],
                true,
            ),
            (
                "debug: rmeta wrapper + native code members",
                vec![
                    ("lib.rmeta", coff(b"metadata")),
                    ("a.rcgu.o", coff(b".text")),
                    ("b.rcgu.o", coff(b".text")),
                ],
                false,
            ),
            (
                "ObjectAndBitcode: bitcode lives in a section, not as a member",
                vec![
                    ("lib.rmeta", coff(b"metadata")),
                    ("a.rcgu.o", coff(b".llvmbcBC\xc0\xde")),
                ],
                false,
            ),
            (
                "the bitcode WRAPPER container counts too",
                vec![("crate.rcgu.o", {
                    let mut v = LLVM_BITCODE_WRAPPER_MAGIC.to_vec();
                    v.extend_from_slice(b"wrapped");
                    v
                })],
                true,
            ),
            (
                // An odd-length member before the bitcode one: miss the pad byte and the next header
                // lands one byte early, the walk bails, and a release rlib is misread as native.
                "an odd-length member does not desynchronize the walk",
                vec![
                    ("odd.txt", b"abc".to_vec()),
                    ("tiny.o", b"ab".to_vec()),
                    ("crate.rcgu.o", bitcode(b"module body")),
                ],
                true,
            ),
            ("an archive with no members at all", vec![], false),
        ];

        for (what, members, expected) in cases {
            let path = dir.join("probe.rlib");
            std::fs::write(&path, synth_archive(&members)).expect("write probe rlib");
            assert_eq!(
                rlib_holds_bitcode_members(&path),
                expected,
                "{what}: misclassified"
            );
        }

        // Not an archive at all, and a missing file: both must answer "link it the ordinary way".
        let junk = dir.join("junk.rlib");
        std::fs::write(&junk, b"this is not an archive").expect("write junk");
        assert!(
            !rlib_holds_bitcode_members(&junk),
            "a non-archive is not bitcode"
        );
        assert!(
            !rlib_holds_bitcode_members(&dir.join("absent.rlib")),
            "a missing rlib is not bitcode"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// End-to-end GPU-backend gate: the `--backend=gpu` path (offloading interpreter + [`gpu_accel`])
/// must match the pure-interpreter oracle within the CPU↔GPU tolerance over the **same** lowered MIR.
/// Only built with `--features gpu`; skips when no CUDA device is present — loudly, and as a
/// *failure* under `WUKONG_GPU_REQUIRED=1` (see `skip_no_device`).
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_tests {
    use super::*;
    use wukong_codegen_gpu::diff::{assert_close, Rng};
    use wukong_span::SourceMap;

    /// Every gate below early-`return`s when no CUDA device is reachable. libtest captures stderr on
    /// a *pass*, so that return is invisible: the crate reports `test result: ok` having offloaded
    /// nothing. `WUKONG_GPU_REQUIRED=1` — what a rented-GPU run sets — must turn that into a failure,
    /// exactly as it already does for every device gate inside `wukong_codegen_gpu`
    /// (`wukong_codegen_gpu::diff::skip_or_fail`, which this delegates to so one env var still
    /// governs every skip in the workspace).
    fn skip_no_device(name: &str) {
        wukong_codegen_gpu::diff::skip_or_fail(
            name,
            wukong_codegen_gpu::gpu::init_error().unwrap_or("no CUDA device reachable"),
        );
    }

    /// Lex → parse → sema → mir_build → opt(2), asserting each stage is clean. The recognizers run in
    /// mir_build, so the resulting MIR already carries the `wukong_sgemm_nt` call the GPU offloads.
    fn build(src: &str) -> (wukong_mir::Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("gpu_e2e.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
        let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
        wukong_opt::optimize(&mut program, 2);
        (program, interner)
    }

    /// The `ijk` `C = A·Bᵀ` (nn.Linear) nest that the recognizer lowers to `wukong_sgemm_nt`.
    fn linear_src(m: usize, k: usize, n: usize) -> String {
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],mut c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a separate elementwise activation loop over `C` — which the
    /// recognizer folds into one `wukong_sgemm_nt_epi` (bias-free, the SwiGLU/FFN `act(x·Wᵀ)` shape).
    /// `actname` is `relu`/`silu`/`gelu`.
    fn linear_act_src(actname: &str, m: usize, k: usize, n: usize) -> String {
        let cij = format!("c[i*{n}+j]");
        let act = match actname {
            "relu" => format!("fmax({cij}, 0.0)"),
            "silu" => format!("silu({cij})"),
            "gelu" => format!("gelu({cij})"),
            other => panic!("unknown activation {other}"),
        };
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],mut c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a **bias-add (+ optional activation)** epilogue loop — which the
    /// recognizer folds into one `wukong_sgemm_nt_epi` with a *non-null* bias (the canonical
    /// `nn.Linear`/FFN `act(x·Wᵀ + bias)`). `actname` is `none` (affine Linear), `relu`, `silu`, `gelu`.
    /// The `bias` param sits between `b` and `c`, so the buffer list is `[a, b, bias, c]`.
    fn linear_bias_act_src(actname: &str, m: usize, k: usize, n: usize) -> String {
        let inner = format!("c[i*{n}+j] + bias[j]");
        let act = match actname {
            "none" => inner.clone(),
            "relu" => format!("fmax({inner}, 0.0)"),
            "silu" => format!("silu({inner})"),
            "gelu" => format!("gelu({inner})"),
            other => panic!("unknown activation {other}"),
        };
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],bias:[f32;{n}],mut c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// **The tolerance band a plain-GEMM offload is held to, chosen by the route that RAN.**
    ///
    /// [`gpu_accel::GemmRoute::Existing`] is `gpu::gemm_nt`, an f32 kernel: both sides are f32 and
    /// only the K-reduction order differs, so the band is `c·√K·ε` — the `sgemm_matches_naive` shape.
    /// [`gpu_accel::GemmRoute::Wgmma`] converts both operands to f16 on the host before accumulating
    /// in f32, so it carries an input rounding the CPU oracle does not and needs the fp16 band the
    /// fused-epilogue gates below already use — some 30× looser at both bounds.
    ///
    /// # The argument must be the witness, never the eligibility
    ///
    /// Callers pass [`gpu_accel::GpuAccel::gemm_route_taken`], which reads the per-route offload
    /// counters. `GpuAccel::gemm_route()` is **not** interchangeable with it: that one is
    /// `gemm_route_for(cc, env)` and never sees a shape, so on a Hopper part it answers `Wgmma` for
    /// every call — *including* the ones `gpu_accel::wgmma_declines` sends to the f32 launcher
    /// (`K % 8 != 0`, an odd `N`, `M*N > u32::MAX`, a ring past the probed shared-memory budget) and
    /// the ones `gemm_nt_wgmma` itself declines at launch. Sizing the band from it would hold an f32
    /// result to an f16 band and let a real f32 regression through, which is precisely the failure
    /// this doc comment used to name while the code keyed on the architecture anyway. The
    /// `WUKONG_GPU_NO_WGMMA` half was honest — that switch lives inside `gemm_route_for` — and the
    /// "unencodable K" half was not; the witness covers both.
    fn gemm_tolerance(route: gpu_accel::GemmRoute, k: usize) -> (f64, f64) {
        match route {
            gpu_accel::GemmRoute::Existing => (
                1e-4,
                (16.0 * (k as f64).sqrt() * f32::EPSILON as f64).max(1e-5),
            ),
            gpu_accel::GemmRoute::Wgmma => (5e-2, 2e-2),
        }
    }

    /// **The band must follow the route that ran, and this is the case that separates the two.**
    ///
    /// Needs no device: both inputs are pure functions. A Hopper part is eligible for the wgmma
    /// family at *every* shape — `gemm_route_for` takes only a capability — while `wgmma_declines`
    /// really does send a subset of shapes to `gpu::gemm_nt`. So on that device "eligible" and "ran"
    /// disagree for those shapes, and only the second one may size a band: an f32 result judged
    /// against `(5e-2, 2e-2)` instead of `(1e-4, 16√K·ε)` hides roughly two and a half decades of
    /// regression.
    #[test]
    fn the_gemm_band_follows_the_route_that_ran_not_the_device() {
        // The eligibility, on the device this campaign rents.
        assert_eq!(
            gpu_accel::gemm_route_for((9, 0), false),
            gpu_accel::GemmRoute::Wgmma
        );
        // The witness, for a call on that same device that fell back.
        let ran = gpu_accel::gemm_route_taken_from(0, 1).expect("one offload ran");
        assert_eq!(ran, gpu_accel::GemmRoute::Existing);

        let k = 4096;
        let (fallback_abs, fallback_rel) = gemm_tolerance(ran, k);
        let (wgmma_abs, wgmma_rel) = gemm_tolerance(gpu_accel::GemmRoute::Wgmma, k);
        assert!(
            fallback_abs < wgmma_abs && fallback_rel < wgmma_rel,
            "the f32 band must be strictly tighter than the f16 one, or keying on the route buys \
             nothing: ({fallback_abs:.1e}, {fallback_rel:.1e}) vs ({wgmma_abs:.1e}, {wgmma_rel:.1e})"
        );
        // And a run with even one wgmma call is judged by the wider band, because that call's f16
        // input rounding is in the buffer being compared.
        assert_eq!(
            gemm_tolerance(
                gpu_accel::gemm_route_taken_from(1, 3).expect("four offloads ran"),
                k
            ),
            (wgmma_abs, wgmma_rel)
        );
    }

    #[test]
    fn gpu_backend_linear_matches_interp_within_tol() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_linear_matches_interp_within_tol");
                return;
            }
        };
        let cc = g.target().cc();
        let mut rng = Rng::new(0x00C0FFEE);
        for &(m, k, n) in &[(8usize, 16usize, 8usize), (16, 32, 16), (32, 48, 24)] {
            let (program, mut interner) = build(&linear_src(m, k, n));
            let entry = interner.intern("lin");
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);

            // CPU oracle (no accelerator).
            let (mut ac, mut bc, mut cc_buf) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cc_buf];
                wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }

            // GPU offload over the identical MIR + inputs.
            let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            let route = accel.gemm_route();
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                wukong_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            assert!(
                accel.calls >= 1,
                "{m}x{k}x{n}: GPU offload never fired — the nest did not lower to sgemm_nt, so this \
                 would silently test CPU-vs-CPU"
            );
            // The routing law, on whatever device is actually present: `wgmma` is architecture-locked
            // to `sm_90a`, so anything that is not Hopper must be on the pre-Hopper launcher. On this
            // repo's RTX 4050 (cc 8.9) that is the assertion that keeps wave 4 from changing the
            // arithmetic of a path it was never meant to touch.
            assert!(
                cc.0 == 9 || route == gpu_accel::GemmRoute::Existing,
                "cc {cc:?} is not Hopper but took {route:?} — an sm_90a module cannot load here"
            );
            // And the same claim against the *witness* rather than the eligibility: on a non-Hopper
            // part every offload must have actually run `gpu::gemm_nt`. This is the arm that runs on
            // this repo's 4050, so it is the one that would catch the wgmma seam leaking onto a
            // device whose module cannot even load.
            if cc.0 != 9 {
                assert_eq!(
                    accel.gemm_wgmma_calls, 0,
                    "cc {cc:?}: {} offload(s) took the wgmma route on a part that is not Hopper",
                    accel.gemm_wgmma_calls
                );
            }

            // The band follows the route that RAN, not the one this device is eligible for: on a
            // Hopper part every shape is eligible and only some are taken.
            let ran = accel
                .gemm_route_taken()
                .expect("no plain GEMM was offloaded, which the assert above should have caught");
            let (abs_tol, rel_tol) = gemm_tolerance(ran, k);
            let s = assert_close(
                &format!("linear {m}x{k}x{n}"),
                &cg,
                &cc_buf,
                abs_tol,
                rel_tol,
            );
            eprintln!(
                "gpu --backend linear {m}x{k}x{n} [eligible {route:?}, ran {ran:?}]: \
                 {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                accel.calls, s.max_abs, s.max_rel
            );
        }
    }

    /// **The wgmma route's end-to-end gate: a `.wk` matmul reaching `gemm_nt_wgmma`.**
    ///
    /// A plain `#[test]`, and that is the load-bearing part: this is the only proof anywhere that
    /// the family the whole campaign publishes actually *executed* on Hopper, so it has to run under
    /// an entrypoint that can turn red. Its H100 invocation is pinned in
    /// [`gpu_accel::HOPPER_GATE_INVOCATION`], where a device-free law checks that it reaches this
    /// function *and* lands on an entrypoint that propagates a failure.
    ///
    /// It used to be `#[ignore]`d and routed through `::bench --name .. --package wukong_driver`,
    /// which is the repo's own "reads as green" class one level up from the vacuous bring-up run:
    /// `modal_app.py`'s `bench()` ran its child with `check=False` and then returned, so the
    /// function exited 0 whatever the tests did, and the witness assertion below — the entire point
    /// of the route counters — could fail on rented silicon while the detached app completed
    /// successfully. `::test` is the entrypoint that ends `if rc1 or rc2: sys.exit(1)`.
    ///
    /// The `#[ignore]` bought nothing it needed: a non-Hopper device is a **capability** skip
    /// handled in the body below, and no device at all is a `skip_no_device`. So on this repo's RTX
    /// 4050 and on a device-free CI runner it prints `[skip]` and returns, while on Hopper it is a
    /// hard gate.
    ///
    /// **Neither skip can survive the pinned invocation, and that took two different levers.** No
    /// device is `skip_no_device` -> `diff::skip_or_fail`, a failure under `WUKONG_GPU_REQUIRED=1`,
    /// which `::test` always sets. The capability skip cannot use that lever — `::test` sets it on
    /// the routine `WK_GPU=L4` rounds too, where skipping this gate is correct — so it escalates on
    /// `gpu_accel::REQUIRED_CC_ENV` instead: the capability the round *declared it rented*, exported
    /// by `::test` from `WK_GPU`. Declared Hopper + probed non-Hopper is a mis-provisioned
    /// container and fails; every other combination skips exactly as before. Without that, this
    /// gate's own pinned single-test H100 invocation printed PASS on any non-Hopper container while
    /// asserting nothing, which is the failure class the invocation constant exists to close.
    ///
    /// It asserts four separate things, and
    /// the first three are the ones a green-but-vacuous run would skip: that the device is
    /// *eligible* for [`gpu_accel::GemmRoute::Wgmma`], that the offload fired at all, that every
    /// offload it fired **took that route** rather than falling back, and only then that the numbers
    /// match the CPU oracle. The third is not implied by the first two — eligibility is `cc` plus an
    /// env switch and never sees a shape, and `accel.calls` is bumped identically by both arms — so
    /// without it a run where `gemm_nt_wgmma` declined at launch (an unencodable tensor map, a
    /// `wgmma_module` `UNSUPPORTED`) passes every assertion while measuring `gpu::gemm_nt`.
    ///
    /// Four shapes, and they live in [`gpu_accel::HOPPER_GATE_SHAPES`] because
    /// `gpu_accel::tests::the_hopper_gate_shapes_reach_every_launch_mechanism_the_rule_can_select`
    /// checks *on this laptop* what they are chosen for — **against the rule itself, never against a
    /// threshold**, which is what let the previous list rot: its clustered shape was chosen because
    /// `M * N` crossed the retired 8e6 split, and the 2026-08-12 promotion sends that shape to the
    /// un-clustered persistent arm instead.
    ///
    /// `256x1024x512` is a whole tile grid (and, under the promoted rule, the NARROW 128×64 tile,
    /// whose `m64n64k16` mainloop is its own). `130x1032x258` is ragged in M, N **and** K at once,
    /// which is what proves the driver seam did not quietly need an alignment gate (the
    /// `m%64/n%64/k%16` rule in `sgemm_nt_epi` is a *wmma* constraint — wgmma predicates its own
    /// edge). `4096x256x2048` is 2 waves at 0.970 wave efficiency with the floor dominating, so it
    /// reaches the PERSISTENT tile loop, where a `%kt` that is not reset per tile makes tile 2
    /// accumulate into tile 1 — an exact integer, and invisible to any one-tile-per-CTA gate.
    /// `2048x4096x4096` is 2 waves at `f_L2 = 0.955`, so both cluster terms pass and it is the only
    /// shape that makes the seam return a **clustered** row: the launch then carries a `1x2x1` cluster
    /// attribute, and a compiled cluster requirement the launch does not match is a *launch failure*
    /// rather than a wrong number.
    ///
    /// Every `N` here is EVEN on purpose, and it is not a raggedness choice: every arm the rule can
    /// select carries the `st.global.v2.f32` epilogue, whose 8-byte pair is aligned only when `N` is,
    /// so an odd `N` is a *sticky* `CUDA_ERROR_MISALIGNED_ADDRESS` rather than a wrong number. The
    /// driver declines it (`wgmma_declines`) and a device-free unit test pins that; putting an odd `N`
    /// in this gate would only prove the decline works by never reaching the kernel.
    #[test]
    fn gpu_backend_linear_routes_through_wgmma_on_hopper() {
        const NAME: &str = "gpu_backend_linear_routes_through_wgmma_on_hopper";
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device(NAME);
                return;
            }
        };
        let cc = g.target().cc();
        if cc.0 != 9 {
            // **A capability skip that can still be a failure — the third way this gate could go
            // green having exercised nothing.**
            //
            // It must NOT escalate under `WUKONG_GPU_REQUIRED=1`: `::test` sets that on every device
            // run, and skipping a Hopper-only gate on the L4/L40S rounds this campaign does most
            // often is the CORRECT outcome (the rule `gpu.rs`'s `with_hopper` states). But the
            // pinned single-test invocation in `gpu_accel::HOPPER_GATE_INVOCATION` books an H100 and
            // selects this one test, so on a non-Hopper container it printed `[skip]`, returned,
            // and libtest reported `ok` — the same green product as the 2026-08-11 vacuous run,
            // reached a third way, with nothing anywhere comparing the probed device against the
            // requested SKU (and `WK_GPU` defaults to `L40S`). `::test` now exports the declared
            // capability, and a declared part this route TAKES against a probed part it does not is
            // a mis-provisioned container, not a correct skip.
            assert!(
                !gpu_accel::capability_skip_is_a_failure(cc, gpu_accel::required_cc()),
                "{NAME}: this round declared {}={} — a part the wgmma route TAKES — but the \
                 container probed cc {}.{} ({}). Skipping here would report a green PASS having \
                 executed no wgmma instruction, which is exactly what the pinned H100 invocation \
                 exists to make impossible; the container is not the SKU the round paid for.",
                gpu_accel::REQUIRED_CC_ENV,
                std::env::var(gpu_accel::REQUIRED_CC_ENV).unwrap_or_default(),
                cc.0,
                cc.1,
                g.device_name()
            );
            eprintln!(
                "[skip:capability] {NAME}: wgmma is architecture-locked to sm_90a and this device \
                 is cc {}.{} ({}) — the route is correctly `Existing` here. The round this gate \
                 exists for is:\n    {}",
                cc.0,
                cc.1,
                g.device_name(),
                gpu_accel::HOPPER_GATE_INVOCATION
            );
            return;
        }
        let mut rng = Rng::new(0x090A_C0DE);
        for &(m, k, n) in &gpu_accel::HOPPER_GATE_SHAPES {
            let (program, mut interner) = build(&linear_src(m, k, n));
            let entry = interner.intern("lin");
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);

            let (mut ac, mut bc, mut cpu) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cpu];
                wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }

            let (mut ag, mut bg, mut got) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            let route = accel.gemm_route();
            assert_eq!(
                route,
                gpu_accel::GemmRoute::Wgmma,
                "this device is Hopper (cc {cc:?}) but the plain-GEMM offload routed to {route:?} — \
                 either the routing rule regressed or WUKONG_GPU_NO_WGMMA is set, and in both cases \
                 the rest of this gate would measure the kernel it exists to bypass"
            );
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut got];
                wukong_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            assert!(
                accel.calls >= 1,
                "{m}x{k}x{n}: nothing offloaded, so this would silently test CPU-vs-CPU"
            );
            // **The witness, and the assertion this gate was vacuous without.** `route` above is an
            // eligibility — `cc` and the env switch — so it says `Wgmma` on this device for every
            // shape, including the ones `wgmma_declines` sends to `gpu::gemm_nt`. `accel.calls >= 1`
            // is route-blind for the same reason: `sgemm_nt` bumps it identically on both arms. So
            // the two assertions above plus a tolerance band wide enough for f16 are all satisfied
            // by a run in which the wgmma family never executed a single instruction — which is
            // precisely the H100 round this gate exists to make impossible.
            assert_eq!(
                (accel.gemm_wgmma_calls, accel.gemm_existing_calls),
                (accel.calls, 0),
                "{m}x{k}x{n}: {} of {} plain-GEMM offloads fell back to `gpu::gemm_nt` — the wgmma \
                 family declined this shape at run time (an unencodable tensor map or a \
                 `wgmma_module` UNSUPPORTED are both silent fallbacks), so this round measured the \
                 pre-Hopper launcher",
                accel.gemm_existing_calls,
                accel.calls
            );
            let taken = accel
                .gemm_route_taken()
                .expect("no plain GEMM was offloaded, which the assert above should have caught");
            assert_eq!(taken, gpu_accel::GemmRoute::Wgmma);

            // The band comes from the witness. It equals the f16 band here only *because* the
            // assertion above proved every offload took the wgmma route; had one fallen back, this
            // shape would be judged against the f32 kernel's own band, as it should be.
            let (abs_tol, rel_tol) = gemm_tolerance(taken, k);
            let s = assert_close(
                &format!("wgmma linear {m}x{k}x{n}"),
                &got,
                &cpu,
                abs_tol,
                rel_tol,
            );
            eprintln!(
                "gpu --backend wgmma linear {m}x{k}x{n} (M%128={} N%256={} K%64={}): \
                 {} GPU call(s), {} wgmma / {} existing, max_abs={:.2e} max_rel={:.2e}",
                m % 128,
                n % 256,
                k % 64,
                accel.calls,
                accel.gemm_wgmma_calls,
                accel.gemm_existing_calls,
                s.max_abs,
                s.max_rel
            );
        }
    }

    /// **An operand past f16's range must not turn a finite GEMM into infinities.**
    ///
    /// The Hopper route converts both f32 operands with `half::f16::from_f32`, whose largest finite
    /// value is 65504 — about four decades below f32's. That is not the "input rounding" the
    /// tolerance band was widened for: an operand past it enters the GEMM as `inf` and comes back as
    /// `inf`/`NaN`, where `gpu::gemm_nt` (f32 in, f32 out) returns exactly what the CPU oracle does.
    /// A *different answer*, which no band covers — and which no other gate here can see, because
    /// every one of them seeds `rng.vec(.., -1.0, 1.0)`. `gpu_accel::wgmma_declines_operand_range`
    /// therefore declines it to the f32 launcher.
    ///
    /// A **paired control**: same shape, same `b`, one element of `a` apart, so on a Hopper part the
    /// route must differ *because of that element* and nothing else. Runs on any device — on a
    /// pre-Hopper part both arms are `Existing` and it still asserts the thing that actually matters
    /// to a user, that neither run returns a non-finite lane where the oracle is finite.
    #[test]
    fn gpu_backend_declines_operands_past_the_f16_range() {
        const NAME: &str = "gpu_backend_declines_operands_past_the_f16_range";
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device(NAME);
                return;
            }
        };
        let cc = g.target().cc();
        let mut rng = Rng::new(0xF16_0BAD);
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (program, mut interner) = build(&linear_src(m, k, n));
        let entry = interner.intern("lin");
        // `b` is kept strictly positive so the huge lane dominates its column sum outright: the
        // comparison below is then about the route, not about a cancellation.
        let b = rng.vec(n * k, 0.5, 1.0);
        let base = rng.vec(m * k, -1.0, 1.0);

        for (label, a0, hopper_route) in [
            ("control", 1.0f32, gpu_accel::GemmRoute::Wgmma),
            // 2^16: exactly representable in f32, and the first power of two past f16's ceiling.
            ("overflow", 65_536.0f32, gpu_accel::GemmRoute::Existing),
        ] {
            // On anything but Hopper the wgmma family is not eligible at all, so both arms fall to
            // the same launcher and the value assertions are the whole content of the case.
            let want = if cc.0 == 9 {
                hopper_route
            } else {
                gpu_accel::GemmRoute::Existing
            };
            let mut a = base.clone();
            a[0] = a0;

            let (mut ac, mut bc, mut cpu) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cpu];
                wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }
            assert!(
                cpu.iter().all(|v| v.is_finite()),
                "{label}: the CPU oracle itself is non-finite, so this case proves nothing"
            );

            let (mut ag, mut bg, mut got) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut got];
                wukong_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            let taken = accel
                .gemm_route_taken()
                .expect("the GEMM must have offloaded, or this tests CPU-vs-CPU");
            assert_eq!(
                taken, want,
                "{label} (cc {cc:?}, a[0] = {a0:e}): on Hopper the route must be decided by this ONE \
                 element — the shape is identical in both runs and declines in neither"
            );
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{label}: the GPU returned a non-finite lane where the f32 oracle is finite — the \
                 range decline did not fire and the f16 seam overflowed"
            );
            // The band follows the route, so a declined call is held to the f32 kernel's own bound.
            // The decline buys back the tight band as well as the finite answer.
            let (abs_tol, rel_tol) = gemm_tolerance(taken, k);
            let s = assert_close(
                &format!("range {label} {m}x{k}x{n}"),
                &got,
                &cpu,
                abs_tol,
                rel_tol,
            );
            eprintln!(
                "gpu --backend range {label} {m}x{k}x{n} a[0]={a0:e} [{taken:?}]: \
                 {} wgmma / {} existing, max_abs={:.2e} max_rel={:.2e}",
                accel.gemm_wgmma_calls, accel.gemm_existing_calls, s.max_abs, s.max_rel
            );
        }
    }

    /// **A row wholly under f16's range must not turn a finite GEMM into a row of zeros.**
    ///
    /// The underflow twin of the test above, and the harder of the two to gate, because its failure
    /// is *quiet*: an operand scaled to ~1e-8 is under f16's round-to-zero point (`2^-25` = 2.98e-8),
    /// so every element enters the wgmma seam as an exact zero and `C` comes back identically zero
    /// where `gpu::gemm_nt` returns what the CPU oracle returns. Unlike the overflow case there is
    /// no `inf` to spot — and the tolerance band cannot spot it either, in *both* of the ways a band
    /// can fail to: the true result is ~1e-13, comfortably under `gemm_tolerance(Wgmma, ..)`'s
    /// `abs_tol = 5e-2`, and `diff::assert_close` passes a lane on `abs <= abs_tol || rel <=
    /// rel_tol`. **So the value assertions in this test cannot fail on the defect; only the ROUTE
    /// assertion can.** That is the point of writing it as a paired control rather than as a
    /// tolerance check: same shape, same `b`, one scale factor apart, so on a Hopper part the route
    /// must differ *because of the magnitudes* and nothing else.
    ///
    /// # The third arm, and why the granularity is the whole point
    ///
    /// `C[i][j] = sum_k A[i][k]*B[j][k]`, so row `i` of A feeds row `i` of `C` and nothing else. The
    /// `quiet row` arm scales **one row** of A to 1e-8 and leaves the rest at O(1): the operand's
    /// single largest magnitude is then ~1.0, which is why a per-operand rule passed it and the
    /// route stayed `Wgmma` while that row of `C` came back exactly zero.
    /// `gpu_accel::wgmma_declines_operand_range` folds its maximum **per row** for exactly this
    /// case, and this arm is the device-side proof: unlike the whole-operand arm its *value*
    /// assertion is load-bearing too, because `got` would hold a zero row where the oracle is
    /// strictly positive.
    ///
    /// Runs on any device — on a pre-Hopper part both arms are `Existing` and it still asserts the
    /// user-visible property, that the GPU does not return zeros where the oracle is non-zero.
    #[test]
    fn gpu_backend_declines_operands_wholly_under_the_f16_normal_range() {
        const NAME: &str = "gpu_backend_declines_operands_wholly_under_the_f16_normal_range";
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device(NAME);
                return;
            }
        };
        let cc = g.target().cc();
        let mut rng = Rng::new(0x5AB_0BAD);
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (program, mut interner) = build(&linear_src(m, k, n));
        let entry = interner.intern("lin");
        // Both operands strictly positive so no lane of the true result can cancel to zero on its
        // own — a zero in `got` is then the seam flushing, never the arithmetic.
        let b = rng.vec(n * k, 0.5, 1.0);
        let base = rng.vec(m * k, 0.5, 1.0);

        // The per-ROW scale of A, so the third arm can be quiet in one row and loud in the rest.
        // `fn(usize) -> f32` rather than a closure so the three arms live in one array.
        for (label, scale_of_row, hopper_route) in [
            (
                "control",
                (|_row| 1.0f32) as fn(usize) -> f32,
                gpu_accel::GemmRoute::Wgmma,
            ),
            // Under 2^-25: every element of A converts to an exact f16 zero.
            (
                "underflow",
                (|_row| 1e-8f32) as fn(usize) -> f32,
                gpu_accel::GemmRoute::Existing,
            ),
            // ONE quiet row inside a loud operand: peak|A| is still ~1.0, so the per-operand rule
            // this replaced saw nothing, and row 1 of C came back exactly zero.
            (
                "quiet row",
                (|row| if row == 1 { 1e-8f32 } else { 1.0 }) as fn(usize) -> f32,
                gpu_accel::GemmRoute::Existing,
            ),
        ] {
            let want = if cc.0 == 9 {
                hopper_route
            } else {
                gpu_accel::GemmRoute::Existing
            };
            let a: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(idx, x)| x * scale_of_row(idx / k))
                .collect();
            // The number the replaced rule looked at. On the `quiet row` arm it is ~1.0, i.e. the
            // per-operand maximum is *blind* to the row this arm exists for, which is why the arm
            // is here at all.
            let peak = a.iter().fold(0.0f32, |p, x| p.max(x.abs()));

            let (mut ac, mut bc, mut cpu) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cpu];
                wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }
            assert!(
                cpu.iter().all(|v| v.is_finite() && *v > 0.0),
                "{label}: the CPU oracle must be finite and strictly positive everywhere, or a \
                 zeroed GPU result would not be evidence of anything"
            );

            let (mut ag, mut bg, mut got) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut got];
                wukong_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            let taken = accel
                .gemm_route_taken()
                .expect("the GEMM must have offloaded, or this tests CPU-vs-CPU");
            assert_eq!(
                taken, want,
                "{label} (cc {cc:?}, max|A| = {peak:e}, row scales {:e}/{:e}): on Hopper the route \
                 must be decided by the MAGNITUDES OF A ROW — the shape is identical in all three \
                 runs and declines in none. On the whole-operand arm this is the only assertion the \
                 defect can fail, because the band cannot see a 1e-13 result zeroed.",
                scale_of_row(0),
                scale_of_row(1)
            );
            assert!(
                got.iter().all(|v| *v > 0.0),
                "{label}: the GPU returned a zero lane where the f32 oracle is strictly positive — \
                 the underflow decline did not fire and the f16 seam flushed a whole row (max|A| = \
                 {peak:e}, so a rule folded over the whole operand cannot see it)"
            );
            let (abs_tol, rel_tol) = gemm_tolerance(taken, k);
            let s = assert_close(
                &format!("underflow {label} {m}x{k}x{n}"),
                &got,
                &cpu,
                abs_tol,
                rel_tol,
            );
            eprintln!(
                "gpu --backend underflow {label} {m}x{k}x{n} max|A|={peak:e} [{taken:?}]: \
                 {} wgmma / {} existing, max_abs={:.2e} max_rel={:.2e}",
                accel.gemm_wgmma_calls, accel.gemm_existing_calls, s.max_abs, s.max_rel
            );
        }
    }

    /// **`act(matmul(x,w))` from Wukong source runs the fused tensor-core kernel** (Phase-2 fusion
    /// engine). The recognizer folds the matmul + activation loop into `wukong_sgemm_nt_epi`; the GPU
    /// `Accelerator` routes that to the single fused WMMA kernel (`gemm_nt_f16_sm_db_{relu,silu,gelu}`,
    /// the one that beats the cuBLAS GEMM+activation chain). Aligned shapes (M,N %64, K %16) so the
    /// tensor-core tiling accepts them. The offload must fire (`calls >= 1`, else it would silently
    /// test CPU-vs-CPU); the fp16 path is tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_epilogue_matches_interp() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_fused_epilogue_matches_interp");
                return;
            }
        };
        let mut rng = Rng::new(0x0FADE5);
        for actname in ["relu", "silu", "gelu"] {
            for &(m, k, n) in &[(64usize, 16usize, 64usize), (64, 32, 128)] {
                let (program, mut interner) = build(&linear_act_src(actname, m, k, n));
                let entry = interner.intern("lin");
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);

                // CPU oracle (no accelerator) — the f32 fused epilogue.
                let (mut ac, mut bc, mut cc) = (a.clone(), b.clone(), vec![0f32; m * n]);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cc];
                    wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                    wukong_interp::run_kernel_f32_accel(
                        &program, entry, &mut bufs, &interner, &mut accel,
                    )
                    .unwrap();
                }
                assert!(
                    accel.calls >= 1,
                    "{actname} {m}x{k}x{n}: fused epilogue never offloaded — the act(matmul) nest did \
                     not fold to sgemm_nt_epi or the GPU declined, so this would silently test CPU-vs-CPU"
                );

                // fp16 tensor-core path → fp16 tolerance (looser than the f32 GEMM bound).
                let s = assert_close(
                    &format!("fused {actname} {m}x{k}x{n}"),
                    &cg,
                    &cc,
                    5e-2,
                    2e-2,
                );
                eprintln!(
                    "gpu --backend fused {actname} {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                    accel.calls, s.max_abs, s.max_rel
                );
            }
        }
    }

    /// **`act(matmul(x,w) + bias)` from Wukong source runs the fused tensor-core *bias* kernel** — the
    /// canonical `nn.Linear`/FFN epilogue (the affine `none` case is a plain biased Linear). The
    /// recognizer folds matmul + bias-add (+ activation) into `wukong_sgemm_nt_epi` with a non-null
    /// bias; the GPU `Accelerator` routes it to `gemm_nt_f16_sm_db_bias{,_relu,_silu,_gelu}` (which
    /// store each tile through SMEM to add the per-column bias the opaque WMMA fragment layout otherwise
    /// blocks). Aligned shapes; the offload must fire (`calls >= 1`, else it would silently test
    /// CPU-vs-CPU); fp16 path → tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_bias_epilogue_matches_interp() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_fused_bias_epilogue_matches_interp");
                return;
            }
        };
        let mut rng = Rng::new(0x0B1A5E);
        for actname in ["none", "relu", "silu", "gelu"] {
            for &(m, k, n) in &[(64usize, 16usize, 64usize), (64, 32, 128)] {
                let (program, mut interner) = build(&linear_bias_act_src(actname, m, k, n));
                let entry = interner.intern("lin");
                let a = rng.vec(m * k, -1.0, 1.0);
                let b = rng.vec(n * k, -1.0, 1.0);
                let bias = rng.vec(n, -0.5, 0.5);

                // CPU oracle (no accelerator) — the f32 fused bias epilogue. Buffers: [a, b, bias, c].
                let (mut ac, mut bc, mut biasc, mut cc) =
                    (a.clone(), b.clone(), bias.clone(), vec![0f32; m * n]);
                {
                    let mut bufs: [&mut [f32]; 4] = [&mut ac, &mut bc, &mut biasc, &mut cc];
                    wukong_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut biasg, mut cg) =
                    (a.clone(), b.clone(), bias.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 4] = [&mut ag, &mut bg, &mut biasg, &mut cg];
                    wukong_interp::run_kernel_f32_accel(
                        &program, entry, &mut bufs, &interner, &mut accel,
                    )
                    .unwrap();
                }
                assert!(
                    accel.calls >= 1,
                    "{actname} {m}x{k}x{n}: fused bias epilogue never offloaded — the act(matmul+bias) \
                     nest did not fold to sgemm_nt_epi or the GPU declined, so this would silently test \
                     CPU-vs-CPU"
                );

                let s = assert_close(
                    &format!("fused bias {actname} {m}x{k}x{n}"),
                    &cg,
                    &cc,
                    5e-2,
                    2e-2,
                );
                eprintln!(
                    "gpu --backend fused bias {actname} {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                    accel.calls, s.max_abs, s.max_rel
                );
            }
        }
    }

    /// Run `entry` over `init` (full buffer list, inputs + zeroed outputs) twice — once on the
    /// interpreter oracle, once on the GPU offload — and return both buffer sets plus how many
    /// kernels actually ran on the device. The two runs see identical inputs and the same MIR.
    fn run_both(
        g: &mut wukong_codegen_gpu::Gpu,
        program: &wukong_mir::Program,
        entry: wukong_span::Symbol,
        interner: &Interner,
        init: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, u32) {
        let mut cpu: Vec<Vec<f32>> = init.to_vec();
        {
            let mut refs: Vec<&mut [f32]> = cpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            wukong_interp::run_kernel_f32(program, entry, &mut refs, interner).unwrap();
        }
        let mut gpu: Vec<Vec<f32>> = init.to_vec();
        let mut accel = gpu_accel::GpuAccel::new(g);
        {
            let mut refs: Vec<&mut [f32]> = gpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            wukong_interp::run_kernel_f32_accel(program, entry, &mut refs, interner, &mut accel)
                .unwrap();
        }
        (cpu, gpu, accel.calls)
    }

    /// The other three GPU kernel families behind `--backend=gpu`: activation (vmath), reduction
    /// (sreduce), and a fused row norm (softmax). Each must fire on the device and match the interp
    /// oracle within tolerance (transcendental SFU approximations are looser than the GEMM bound).
    #[test]
    fn gpu_backend_activation_reduction_norm_match_interp() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_activation_reduction_norm_match_interp");
                return;
            }
        };
        let mut rng = Rng::new(0x5EED_1234);

        // 1) Activation: out[i] = silu(x[i]) -> wukong_vmath_f32 (SFU sigmoid approx on GPU).
        {
            let n = 64usize;
            let src = format!(
                "module m\nfn act(x:[f32;{n}], mut out:[f32;{n}]) {{ \
                 for i in 0..{n} {{ out[i] = silu(x[i]); }} }}"
            );
            let (program, mut interner) = build(&src);
            let entry = interner.intern("act");
            let init = vec![rng.vec(n, -4.0, 4.0), vec![0.0; n]];
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert!(calls >= 1, "activation offload never fired");
            let s = assert_close("silu activation", &gpu[1], &cpu[1], 2e-3, 5e-3);
            eprintln!(
                "gpu --backend silu[{n}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }

        // 2) Reduction: o[0] = Σ x·y -> wukong_sreduce_f32_parallel (dot, written to a [1] buffer).
        {
            let n = 1024usize;
            let src = format!(
                "@parallel fn dotp(x:[f32;{n}], y:[f32;{n}], mut o:[f32;1]) {{ \
                 let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + x[k] * y[k]; }} o[0] = s; }}"
            );
            let (program, mut interner) = build(&src);
            let entry = interner.intern("dotp");
            let init = vec![rng.vec(n, -1.0, 1.0), rng.vec(n, -1.0, 1.0), vec![0.0; 1]];
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert!(calls >= 1, "reduction offload never fired");
            let rel = (16.0 * (n as f64).sqrt() * f32::EPSILON as f64).max(1e-5);
            let s = assert_close("dot reduction", &gpu[2], &cpu[2], 1e-3, rel);
            eprintln!(
                "gpu --backend dot[{n}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }

        // 3) Fused norm: batched softmax over [R,C] -> wukong_norm_f32 (in place; GPU uses ex2.approx).
        {
            let (r, c) = (4usize, 16usize);
            let n = r * c;
            let src = format!(
                "module m\nfn sm(mut x:[f32;{n}]) {{ for row in 0..{r} {{ \
                 let mut m: f32 = x[row*{c}]; for i in 0..{c} {{ m = fmax(m, x[row*{c}+i]); }} \
                 for i in 0..{c} {{ x[row*{c}+i] = exp(x[row*{c}+i] - m); }} \
                 let mut s: f32 = 0.0; for i in 0..{c} {{ s = s + x[row*{c}+i]; }} \
                 let inv: f32 = 1.0 / s; for i in 0..{c} {{ x[row*{c}+i] = x[row*{c}+i] * inv; }} }} }}"
            );
            let (program, mut interner) = build(&src);
            let entry = interner.intern("sm");
            let init = vec![rng.vec(n, -3.0, 3.0)];
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert!(
                calls >= 1,
                "norm offload never fired (softmax did not lower to wukong_norm_f32)"
            );
            let s = assert_close("batched softmax", &gpu[0], &cpu[0], 2e-3, 5e-3);
            eprintln!(
                "gpu --backend softmax[{r}x{c}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }
    }

    /// The two `NORM_*` op codes the recognizer emits but the GPU `norm` kernel has no PTX entry for
    /// (LOGSOFTMAX=3, L2NORM=4). Before the `norm_supported` gate these forwarded to `gpu::norm` and
    /// hit its `_ => panic!("norm op {op} not implemented on GPU yet")`, aborting the process on
    /// ordinary user input. They must now decline to the CPU kernel: zero device calls, and — since
    /// nothing ran on the device — output bit-identical to the interpreter oracle.
    #[test]
    fn gpu_backend_declines_unimplemented_norm_ops_to_cpu() {
        let mut guard = wukong_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                skip_no_device("gpu_backend_declines_unimplemented_norm_ops_to_cpu");
                return;
            }
        };
        let mut rng = Rng::new(0x1EAF_0011);
        let n = 64usize;

        // NORM_LOGSOFTMAX (op 3): the stable max / log-sum-exp / subtract row window, in place.
        let logsoftmax = format!(
            "module m\nfn lsm(mut x:[f32;{n}]) {{ let mut m: f32 = x[0]; \
             for i in 0..{n} {{ m = fmax(m, x[i]); }} let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + exp(x[i] - m); }} let ls: f32 = log(s); \
             for i in 0..{n} {{ x[i] = (x[i] - m) - ls; }} }}"
        );
        // NORM_L2NORM (op 4): sum of squares / rsqrt / scale, x -> out.
        let l2norm = format!(
            "module m\nfn l2n(x:[f32;{n}], mut out:[f32;{n}]) {{ let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + x[i]*x[i]; }} let inv: f32 = rsqrt(s + 0.00001); \
             for i in 0..{n} {{ out[i] = x[i]*inv; }} }}"
        );

        for (label, src, entry_name, init) in [
            (
                "log_softmax",
                logsoftmax,
                "lsm",
                vec![rng.vec(n, -3.0, 3.0)],
            ),
            (
                "l2norm",
                l2norm,
                "l2n",
                vec![rng.vec(n, -3.0, 3.0), vec![0.0; n]],
            ),
        ] {
            let (program, mut interner) = build(&src);
            let entry = interner.intern(entry_name);
            // The window must actually lower to `wukong_norm_f32`, else this tests nothing.
            let mir = wukong_mir::print::print_program(&program, &interner);
            assert!(
                mir.contains("wukong_norm_f32"),
                "{label}: did not lower to a fused norm — the gate would never be consulted"
            );
            let (cpu, gpu, calls) = run_both(g, &program, entry, &interner, &init);
            assert_eq!(
                calls, 0,
                "{label}: an unimplemented norm op must not reach the device"
            );
            let last = cpu.len() - 1;
            assert_eq!(
                gpu[last], cpu[last],
                "{label}: the CPU fallback must be bit-identical to the oracle"
            );
        }
    }
}

#[cfg(test)]
mod native_training_tests;
mod training_native;
