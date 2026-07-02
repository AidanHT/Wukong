//! `mercury_driver` — orchestrates the compilation pipeline.
//!
//! The driver owns the [`SourceMap`] and the [`DiagnosticSink`], runs each stage in order, and
//! honors the requested `--emit` target. Stages are wired in as they come online; today the
//! lexer is connected and `--emit=tokens` works end to end.

use std::path::PathBuf;

use mercury_diag::{Diagnostic, DiagnosticSink, Renderer};
use mercury_span::{Interner, SourceMap};

pub use mercury_diag::{all_explanations, explain, Explanation};

/// The GPU offloading bridge (`--backend=gpu`), compiled only with the `gpu` feature.
#[cfg(feature = "gpu")]
mod gpu_accel;

/// Which intermediate (or final) artifact the user asked to produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmitStage {
    Tokens,
    Ast,
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
    pub color: bool,
    pub error_format: ErrorFormat,
    /// Which backend to run/emit with. `--run` defaults to the interpreter (the reference oracle);
    /// native object/exe emission always uses Cranelift.
    pub backend: BackendKind,
    /// Autodiff / training options (see [`GradOptions`]). Only consulted by `--emit=grad` and
    /// `--train`; inert for every other stage.
    pub grad: GradOptions,
}

/// Options for the autodiff CLI surface: which loss function to differentiate, which of its buffer
/// parameters to differentiate with respect to, and (for `--train`) the optimizer loop settings.
#[derive(Clone, Debug)]
pub struct GradOptions {
    /// The function to differentiate (`--grad-of=<name>`). Defaults to `loss` when unset.
    pub of: Option<String>,
    /// Parameter indices to differentiate w.r.t. (`--grad-wrt=0,1`). Empty means *every* pointer
    /// (buffer) parameter — the gradient of an unread/output buffer is simply zero, so this is safe.
    pub wrt: Vec<usize>,
    /// Run a fwd→bwd→optimizer training loop instead of just emitting the backward (`--train`).
    pub train: bool,
    /// Number of training steps (`--train-steps=N`).
    pub train_steps: usize,
    /// Learning rate for the training loop (`--train-lr=<f>`).
    pub train_lr: f32,
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
    /// Fused decoupled-weight-decay AdamW (the `mercury_autodiff::optim` kernel).
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
            color: true,
            error_format: ErrorFormat::Human,
            backend: BackendKind::Interp,
            grad: GradOptions::default(),
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

/// Run the pipeline for `opts`, printing artifacts to stdout and diagnostics to stderr.
/// Returns a process exit code.
pub fn compile(opts: &Options) -> i32 {
    let src = match std::fs::read_to_string(&opts.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read `{}`: {}", opts.input.display(), e);
            return exit::IO_ERROR;
        }
    };

    let mut sm = SourceMap::new();
    let id = sm.add(opts.input.display().to_string(), src);
    let renderer = Renderer::new(opts.color);
    let mut sink = DiagnosticSink::new();

    // --- Lexing ---
    let (tokens, lex_diags) = mercury_lexer::tokenize(sm.source(id), id);
    for d in lex_diags {
        sink.emit(d);
    }

    if opts.emit == EmitStage::Tokens {
        render_all(opts.error_format, &renderer, &sink, &sm);
        print!("{}", mercury_lexer::dump(&tokens, sm.source(id)));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // --- Parsing ---
    let mut interner = Interner::new();
    let (module, parse_diags) =
        mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
    for d in parse_diags {
        sink.emit(d);
    }
    render_all(opts.error_format, &renderer, &sink, &sm);

    if opts.emit == EmitStage::Ast {
        print!("{}", mercury_ast::print::print_module(&module, &interner));
        return if sink.has_errors() {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    if sink.has_errors() {
        return exit::COMPILE_ERROR;
    }

    // --- Semantic analysis (name resolution, type checking, shape checking) ---
    let (sema, sema_diags) = mercury_sema::check(&module, &interner);
    for d in &sema_diags {
        emit_diag(d, opts.error_format, &renderer, &sm);
    }
    if sema_diags.iter().any(|d| d.is_error()) {
        return exit::COMPILE_ERROR;
    }

    // --- MIR construction ---
    let (mut program, lower_diags) =
        mercury_mir_build::lower_program(&module, &sema, &mut interner);
    for d in &lower_diags {
        emit_diag(d, opts.error_format, &renderer, &sm);
    }

    // High MIR is the pre-optimization form. It is still dumped for debugging even when lowering
    // failed, since seeing the partial MIR is useful.
    if opts.emit == EmitStage::MirHigh {
        emit_mir(&program, &interner);
        return if lower_diags.iter().any(|d| d.is_error()) {
            exit::COMPILE_ERROR
        } else {
            exit::OK
        };
    }

    // A construct codegen could not lower leaves the MIR invalid, so do not optimize, run, or hand
    // it to a backend — that would crash or miscompile. Stop with a clear error instead.
    if lower_diags.iter().any(|d| d.is_error()) {
        return exit::COMPILE_ERROR;
    }

    // --- Optimization ---
    // The autodiff transform (`--emit=grad` / `--train`) consumes single-block SSA, so it forces at
    // least `-O1` (mem2reg + simplify-cfg) regardless of the requested level. Every other path honors
    // the requested level exactly.
    let opt_level = if opts.emit == EmitStage::Grad || opts.grad.train {
        opts.opt_level.max(1)
    } else {
        opts.opt_level
    };
    mercury_opt::optimize(&mut program, opt_level);

    // --- Autodiff CLI surface (additive; runs after optimization, before any backend) ---
    if opts.emit == EmitStage::Grad {
        return emit_grad(&program, &mut interner, opts);
    }

    // --- Run via the selected backend (interpreter by default, Cranelift JIT with --backend=native) ---
    if opts.run {
        let main = interner.intern("main");
        let result = match opts.backend {
            BackendKind::Interp => mercury_interp::run_with_output(&program, main, &interner),
            BackendKind::Native => mercury_codegen_cranelift::jit_run(&program, main, &interner),
            BackendKind::Gpu => run_on_gpu(&program, main, &interner),
            BackendKind::GpuLower => run_on_gpu_lower(&program, main, &interner),
        };
        return match result {
            Ok((exit_code, stdout)) => {
                use std::io::Write;
                let _ = std::io::stdout().write_all(&stdout);
                exit_code as i32
            }
            Err(e) => {
                eprintln!("error: {e}");
                exit::COMPILE_ERROR
            }
        };
    }

    if opts.emit == EmitStage::Mir {
        emit_mir(&program, &interner);
        return exit::OK;
    }

    // --- LLVM backend ---
    if opts.emit == EmitStage::LlvmIr {
        print!(
            "{}",
            mercury_codegen_llvm::emit_llvm_ir(&program, &interner)
        );
        return exit::OK;
    }
    if matches!(opts.emit, EmitStage::Obj | EmitStage::Exe) {
        return emit_native(&program, &interner, opts);
    }

    exit::OK
}

/// Execute `entry` on the GPU backend: the interpreter tree-walks the program while recognized kernel
/// calls run on the device (see [`gpu_accel`]). Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu(
    program: &mercury_mir::Program,
    entry: mercury_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let mut guard = mercury_codegen_gpu::gpu();
    match guard.as_mut() {
        Some(g) => {
            let mut accel = gpu_accel::GpuAccel::new(g);
            mercury_interp::run_with_output_accel(program, entry, interner, &mut accel)
        }
        None => Err(
            "`--backend=gpu` requires a CUDA device, but none was reachable (the driver \
                     dlopens `nvcuda.dll`; check the NVIDIA driver is installed)"
                .into(),
        ),
    }
}

/// Without the `gpu` feature, `--backend=gpu` is a clear build-time-capability error rather than a
/// silent fallback (which would dishonestly run on the CPU when the user asked for the GPU).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu(
    _program: &mercury_mir::Program,
    _entry: mercury_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `mercuryc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p mercuryc --features gpu`) to use `--backend=gpu`"
        .into())
}

/// Execute `entry` on the **general MIR→PTX** GPU backend (`--backend=gpu-native`, Phase 4): the
/// whole program is lowered to PTX and run on the device — distinct from the recognizer offload
/// ([`run_on_gpu`]), which it leaves untouched. Built only with `--features gpu`.
#[cfg(feature = "gpu")]
fn run_on_gpu_lower(
    program: &mercury_mir::Program,
    entry: mercury_span::Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    mercury_codegen_gpu::lower::jit_run(program, entry, interner)
}

/// Without the `gpu` feature, `--backend=gpu-native` is a clear build-capability error rather than a
/// silent CPU fallback (mirrors [`run_on_gpu`]).
#[cfg(not(feature = "gpu"))]
fn run_on_gpu_lower(
    _program: &mercury_mir::Program,
    _entry: mercury_span::Symbol,
    _interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    Err("this `mercuryc` was built without GPU support; rebuild with `--features gpu` (or, from the \
         workspace, `-p mercuryc --features gpu`) to use `--backend=gpu-native`"
        .into())
}

/// The C runtime linked into native executables: it backs the `print`/`assert` intrinsics that the
/// Cranelift object leaves as undefined imports. The JIT path binds the same symbols to Rust
/// functions instead, so the two stay in lockstep.
const MERCURY_RT_C: &str = "#include <stdio.h>\n\
#include <stdlib.h>\n\
#include <math.h>\n\
void mercury_rt_print_i64(long long x) { printf(\"%lld\\n\", x); }\n\
void mercury_rt_print_f64(double x) { printf(\"%g\\n\", x); }\n\
void mercury_rt_assert(long long c) { if (!c) { fprintf(stderr, \"assertion failed\\n\"); exit(101); } }\n\
double mercury_rt_fmod_f64(double a, double b) { return fmod(a, b); }\n\
float mercury_rt_fmod_f32(float a, float b) { return fmodf(a, b); }\n";

/// Emit a native object via Cranelift (no LLVM) and, for `--emit=exe`, link it with a small C
/// runtime using the system C compiler. `CC` overrides the compiler (default `cc`).
fn emit_native(program: &mercury_mir::Program, interner: &Interner, opts: &Options) -> i32 {
    use std::process::Command;

    let obj = match mercury_codegen_cranelift::emit_object(program, interner) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cranelift codegen failed: {e}");
            return exit::COMPILE_ERROR;
        }
    };

    let stem = opts
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());

    let is_obj = opts.emit == EmitStage::Obj;
    let obj_path = if is_obj {
        opts.output
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{stem}.o")))
    } else {
        PathBuf::from(format!("{stem}.o"))
    };
    if let Err(e) = std::fs::write(&obj_path, &obj) {
        eprintln!("error: could not write `{}`: {e}", obj_path.display());
        return exit::IO_ERROR;
    }
    if is_obj {
        eprintln!("wrote {}", obj_path.display());
        return exit::OK;
    }

    // exe: emit the C runtime next to the object and link them.
    let rt_path = PathBuf::from(format!("{stem}_rt.c"));
    if let Err(e) = std::fs::write(&rt_path, MERCURY_RT_C) {
        eprintln!("error: could not write `{}`: {e}", rt_path.display());
        return exit::IO_ERROR;
    }
    let out = opts
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{stem}.exe")));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .arg(&obj_path)
        .arg(&rt_path)
        .arg("-o")
        .arg(&out)
        .arg("-O2")
        .status();
    match status {
        Ok(s) if s.success() => {
            eprintln!("wrote {}", out.display());
            exit::OK
        }
        Ok(_) => {
            eprintln!(
                "error: `{cc}` failed to link the native object `{}`",
                obj_path.display()
            );
            exit::COMPILE_ERROR
        }
        Err(_) => {
            eprintln!(
                "error: could not run `{cc}` to link.\n\
                 note: the native object was written to `{}`.\n\
                 help: install a C compiler (gcc/clang/cc) or set CC, or use `--run` to execute.",
                obj_path.display()
            );
            exit::UNIMPLEMENTED
        }
    }
}

/// `--emit=grad`: differentiate the designated loss function and print the forward + backward MIR.
///
/// The loss function must already be single-block SSA (the caller forces `-O1`); its buffer
/// parameters lower to `Ptr` and the scalar loss is `ret`-ed. Any recognized kernel calls in it
/// (`mercury_sgemm_nt`, `mercury_vmath_f32`, `mercury_sreduce_f32[_parallel]`, `mercury_norm_f32`,
/// `mercury_velem_f32`) get their **tuned-kernel** backward — so the emitted gradient rides the same
/// kernels as the forward. A differentiable value with no VJP rule is a hard error, never a
/// silently-zero gradient (see `mercury_autodiff`).
fn emit_grad(program: &mercury_mir::Program, interner: &mut Interner, opts: &Options) -> i32 {
    let (fwd, gradfn) = match build_grad(program, interner, opts) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("error: --emit=grad: {e}");
            return exit::COMPILE_ERROR;
        }
    };
    let out = mercury_mir::Program {
        funcs: vec![fwd, gradfn],
        level: program.level,
    };
    emit_mir(&out, interner);
    exit::OK
}

/// Shared core of `--emit=grad` and `--train`: resolve the target loss function and its `wrt` list
/// from `opts`, run the autodiff transform, and return `(forward_clone, gradient_function)`.
fn build_grad(
    program: &mercury_mir::Program,
    interner: &mut Interner,
    opts: &Options,
) -> Result<(mercury_mir::Function, mercury_mir::Function), String> {
    let name = opts.grad.of.clone().unwrap_or_else(|| "loss".to_string());
    let sym = interner.intern(&name);
    let func = program.function(sym).ok_or_else(|| {
        format!("no function `{name}` in the module (choose the target with --grad-of=<fn>)")
    })?;
    let wrt = grad_wrt(func, &opts.grad.wrt)?;
    let gradfn = mercury_autodiff::grad(func, &wrt, interner)?;
    Ok((func.clone(), gradfn))
}

/// The parameter indices to differentiate w.r.t.: the explicit `--grad-wrt` list (validated to be
/// in-range pointer parameters), or — when empty — *every* pointer (buffer) parameter of the loss.
/// Differentiating an unread or output buffer is harmless (its gradient is simply zero), so the
/// "all buffers" default never produces a wrong answer, only occasionally an unused zero buffer.
fn grad_wrt(func: &mercury_mir::Function, requested: &[usize]) -> Result<Vec<usize>, String> {
    let is_ptr = |i: usize| *func.value_type(func.params[i]) == mercury_mir::MirType::Ptr;
    if requested.is_empty() {
        let all: Vec<usize> = (0..func.params.len()).filter(|&i| is_ptr(i)).collect();
        if all.is_empty() {
            return Err("the loss function has no buffer (pointer) parameters to differentiate".into());
        }
        return Ok(all);
    }
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
    }
    Ok(requested.to_vec())
}

fn emit_mir(program: &mercury_mir::Program, interner: &Interner) {
    for f in &program.funcs {
        for ice in mercury_mir::verify::verify_function(f) {
            eprintln!("internal compiler error (MIR verify): {ice}");
        }
    }
    print!("{}", mercury_mir::print::print_program(program, interner));
}

/// Emit a single diagnostic to stderr in the requested format.
fn emit_diag(d: &Diagnostic, fmt: ErrorFormat, renderer: &Renderer, sm: &SourceMap) {
    match fmt {
        ErrorFormat::Human => eprintln!("{}", renderer.render(d, sm)),
        ErrorFormat::Json => eprintln!("{}", mercury_diag::to_json(d, sm)),
    }
}

fn render_all(fmt: ErrorFormat, renderer: &Renderer, sink: &DiagnosticSink, sm: &SourceMap) {
    for d in sink.diagnostics() {
        emit_diag(d, fmt, renderer, sm);
    }
}

/// The `--emit=grad` CLI surface, end to end from `.mer` source: compile → optimize → differentiate
/// the loss function, then **finite-difference-gate** the emitted backward. This is the real proof
/// that autodiff-from-source is correct — the recognized backward kernels are gate-blind (both
/// backends call the same symbol), so the f64/closed-form finite-difference check, not interp==native,
/// is what proves the gradient. Toolchain-free: it runs entirely on the interpreter oracle.
#[cfg(test)]
mod grad_cli_tests {
    use super::*;
    use mercury_mir::{MirLevel, Program};
    use mercury_span::{SourceMap, Symbol};

    /// Lex → parse → sema → mir_build → opt(1). Panics with the offending stage's diagnostics.
    fn compile_o1(src: &str) -> (Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("grad_test.mer".to_string(), src.to_string());
        let (tokens, ld) = mercury_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors: {sd:?}");
        let (mut program, md) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors: {md:?}");
        mercury_opt::optimize(&mut program, 1);
        (program, interner)
    }

    /// Drive the real `build_grad` path (`--grad-of`/`--grad-wrt`) and return the `{fwd, fwd_grad}`
    /// program, the forward and gradient function names, the **resolved** `wrt` list (so `[]`
    /// expands to every buffer parameter, matching the appended gradient buffers), and the interner.
    fn grad_prog(src: &str, of: &str, wrt: &[usize]) -> (Program, Symbol, Symbol, Vec<usize>, Interner) {
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
        let resolved = grad_wrt(program.function(fsym).expect("loss fn"), wrt).expect("wrt");
        let (fwd, g) = build_grad(&program, &mut interner, &opts).expect("build_grad failed");
        let (fname, gname) = (fwd.name, g.name);
        let prog = Program {
            funcs: vec![fwd, g],
            level: MirLevel::Low,
        };
        (prog, fname, gname, resolved, interner)
    }

    fn run_f32(prog: &Program, name: Symbol, bufs: &mut [Vec<f32>], it: &Interner) {
        let mut views: Vec<&mut [f32]> = bufs.iter_mut().map(|v| v.as_mut_slice()).collect();
        mercury_interp::run_kernel_f32(prog, name, &mut views, it).expect("kernel run failed");
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
                g[j] = (lp - lm) as f64 / (2.0 * eps as f64);
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
        // loss = x^3  ->  dL/dx = 3x^2. Differentiated straight from `.mer` source via --emit=grad.
        let src = "module m\nfn loss(x:[f32;1], out:[f32;1]) -> f32 {\n\
                   let a: f32 = x[0]; let l: f32 = a*a*a; out[0] = l; return l; }";
        let x = 1.3f32;
        let inputs = vec![vec![x], vec![0.0]];
        let g = gate(src, "loss", &[0], &inputs, &[1, 1], 1, 1e-2, 2e-2, 1e-3);
        assert_close(&g[0], &[3.0 * (x * x) as f64], "d(x^3)/dx");
    }

    #[test]
    fn bilinear_two_input_grad() {
        // loss = a * b^2  ->  dL/da = b^2, dL/db = 2ab.
        let src = "module m\nfn loss(a:[f32;1], b:[f32;1], out:[f32;1]) -> f32 {\n\
                   let av: f32 = a[0]; let bv: f32 = b[0]; let l: f32 = av*bv*bv;\n\
                   out[0] = l; return l; }";
        let (a, b) = (0.7f32, 1.1f32);
        let inputs = vec![vec![a], vec![b], vec![0.0]];
        let g = gate(src, "loss", &[0, 1], &inputs, &[1, 1, 1], 2, 1e-2, 3e-2, 1e-3);
        assert_close(&g[0], &[(b * b) as f64], "dL/da = b^2");
        assert_close(&g[1], &[(2.0 * a * b) as f64], "dL/db = 2ab");
    }

    #[test]
    fn default_wrt_is_all_buffers() {
        // With no --grad-wrt, every buffer parameter is differentiated; the output buffer (only
        // stored, never read) gets a correct zero gradient.
        let src = "module m\nfn loss(x:[f32;1], out:[f32;1]) -> f32 {\n\
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
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

    /// `--emit=grad` of a **real** linear-regression model written in `.mer`: `p = x·wᵀ` (recognized
    /// as `mercury_sgemm_nt`) then the MSE reduction `Σ(p−t)²` (recognized as the `@parallel`
    /// `mercury_sreduce_f32_parallel`). The emitted backward rides the tuned kernels — `mercury_velem`
    /// for the `2(p−t)` seed, a transpose loop, and `mercury_sgemm` for `dw = dpᵀ·x` / `dx = dp·w` —
    /// and is finite-difference-gated against the forward loss and the f64 closed form.
    #[test]
    fn linear_mse_from_source() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let src = "@parallel fn loss(x:[f32;12], w:[f32;8], t:[f32;6], out:[f32;1]) -> f32 {\n\
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
}

/// End-to-end GPU-backend gate: the `--backend=gpu` path (offloading interpreter + [`gpu_accel`])
/// must match the pure-interpreter oracle within the CPU↔GPU tolerance over the **same** lowered MIR.
/// Only built with `--features gpu`; skips (does not fail) when no CUDA device is present.
#[cfg(all(test, feature = "gpu"))]
mod gpu_e2e_tests {
    use super::*;
    use mercury_codegen_gpu::diff::{assert_close, Rng};
    use mercury_span::SourceMap;

    /// Lex → parse → sema → mir_build → opt(2), asserting each stage is clean. The recognizers run in
    /// mir_build, so the resulting MIR already carries the `mercury_sgemm_nt` call the GPU offloads.
    fn build(src: &str) -> (mercury_mir::Program, Interner) {
        let mut sm = SourceMap::new();
        let id = sm.add("gpu_e2e.mer".to_string(), src.to_string());
        let (tokens, ld) = mercury_lexer::tokenize(sm.source(id), id);
        assert!(!ld.iter().any(|d| d.is_error()), "lex errors");
        let mut interner = Interner::new();
        let (module, pd) =
            mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        assert!(!pd.iter().any(|d| d.is_error()), "parse errors");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(!sd.iter().any(|d| d.is_error()), "sema errors");
        let (mut program, md) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(!md.iter().any(|d| d.is_error()), "mir_build errors");
        mercury_opt::optimize(&mut program, 2);
        (program, interner)
    }

    /// The `ijk` `C = A·Bᵀ` (nn.Linear) nest that the recognizer lowers to `mercury_sgemm_nt`.
    fn linear_src(m: usize, k: usize, n: usize) -> String {
        format!(
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a separate elementwise activation loop over `C` — which the
    /// recognizer folds into one `mercury_sgemm_nt_epi` (bias-free, the SwiGLU/FFN `act(x·Wᵀ)` shape).
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
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    /// The `C = A·Bᵀ` nest followed by a **bias-add (+ optional activation)** epilogue loop — which the
    /// recognizer folds into one `mercury_sgemm_nt_epi` with a *non-null* bias (the canonical
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
            "module m\nfn lin(a:[f32;{mk}],b:[f32;{nk}],bias:[f32;{n}],c:[f32;{mn}]) {{ \
             for i in 0..{m} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for kk in 0..{k} {{ s = s + a[i*{k}+kk] * b[j*{k}+kk]; }} c[i*{n}+j] = s; }} }} \
             for i in 0..{m} {{ for j in 0..{n} {{ c[i*{n}+j] = {act}; }} }} }}",
            mk = m * k,
            nk = n * k,
            mn = m * n
        )
    }

    #[test]
    fn gpu_backend_linear_matches_interp_within_tol() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!("skip gpu_backend_linear_matches_interp_within_tol: no CUDA device");
                return;
            }
        };
        let mut rng = Rng::new(0x00C0FFEE);
        for &(m, k, n) in &[(8usize, 16usize, 8usize), (16, 32, 16), (32, 48, 24)] {
            let (program, mut interner) = build(&linear_src(m, k, n));
            let entry = interner.intern("lin");
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(n * k, -1.0, 1.0);

            // CPU oracle (no accelerator).
            let (mut ac, mut bc, mut cc) = (a.clone(), b.clone(), vec![0f32; m * n]);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ac, &mut bc, &mut cc];
                mercury_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
            }

            // GPU offload over the identical MIR + inputs.
            let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
            let mut accel = gpu_accel::GpuAccel::new(&mut *g);
            {
                let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                mercury_interp::run_kernel_f32_accel(
                    &program, entry, &mut bufs, &interner, &mut accel,
                )
                .unwrap();
            }
            assert!(
                accel.calls >= 1,
                "{m}x{k}x{n}: GPU offload never fired — the nest did not lower to sgemm_nt, so this \
                 would silently test CPU-vs-CPU"
            );

            // tol = c·√K·ε (the sgemm_matches_naive shape): both sides are f32, K-reduction order differs.
            let rel_tol = (16.0 * (k as f64).sqrt() * f32::EPSILON as f64).max(1e-5);
            let s = assert_close(&format!("linear {m}x{k}x{n}"), &cg, &cc, 1e-4, rel_tol);
            eprintln!(
                "gpu --backend linear {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                accel.calls, s.max_abs, s.max_rel
            );
        }
    }

    /// **`act(matmul(x,w))` from Mercury source runs the fused tensor-core kernel** (Phase-2 fusion
    /// engine). The recognizer folds the matmul + activation loop into `mercury_sgemm_nt_epi`; the GPU
    /// `Accelerator` routes that to the single fused WMMA kernel (`gemm_nt_f16_sm_db_{relu,silu,gelu}`,
    /// the one that beats the cuBLAS GEMM+activation chain). Aligned shapes (M,N %64, K %16) so the
    /// tensor-core tiling accepts them. The offload must fire (`calls >= 1`, else it would silently
    /// test CPU-vs-CPU); the fp16 path is tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_epilogue_matches_interp() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!("skip gpu_backend_fused_epilogue_matches_interp: no CUDA device");
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
                    mercury_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut cg) = (a.clone(), b.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 3] = [&mut ag, &mut bg, &mut cg];
                    mercury_interp::run_kernel_f32_accel(
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
                let s = assert_close(&format!("fused {actname} {m}x{k}x{n}"), &cg, &cc, 5e-2, 2e-2);
                eprintln!(
                    "gpu --backend fused {actname} {m}x{k}x{n}: {} GPU call(s), max_abs={:.2e} max_rel={:.2e}",
                    accel.calls, s.max_abs, s.max_rel
                );
            }
        }
    }

    /// **`act(matmul(x,w) + bias)` from Mercury source runs the fused tensor-core *bias* kernel** — the
    /// canonical `nn.Linear`/FFN epilogue (the affine `none` case is a plain biased Linear). The
    /// recognizer folds matmul + bias-add (+ activation) into `mercury_sgemm_nt_epi` with a non-null
    /// bias; the GPU `Accelerator` routes it to `gemm_nt_f16_sm_db_bias{,_relu,_silu,_gelu}` (which
    /// store each tile through SMEM to add the per-column bias the opaque WMMA fragment layout otherwise
    /// blocks). Aligned shapes; the offload must fire (`calls >= 1`, else it would silently test
    /// CPU-vs-CPU); fp16 path → tolerance-gated against the f32 CPU oracle, not bit-exact.
    #[test]
    fn gpu_backend_fused_bias_epilogue_matches_interp() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!("skip gpu_backend_fused_bias_epilogue_matches_interp: no CUDA device");
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
                    mercury_interp::run_kernel_f32(&program, entry, &mut bufs, &interner).unwrap();
                }

                // GPU offload over the identical MIR + inputs.
                let (mut ag, mut bg, mut biasg, mut cg) =
                    (a.clone(), b.clone(), bias.clone(), vec![0f32; m * n]);
                let mut accel = gpu_accel::GpuAccel::new(&mut *g);
                {
                    let mut bufs: [&mut [f32]; 4] = [&mut ag, &mut bg, &mut biasg, &mut cg];
                    mercury_interp::run_kernel_f32_accel(
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

                let s =
                    assert_close(&format!("fused bias {actname} {m}x{k}x{n}"), &cg, &cc, 5e-2, 2e-2);
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
        g: &mut mercury_codegen_gpu::Gpu,
        program: &mercury_mir::Program,
        entry: mercury_span::Symbol,
        interner: &Interner,
        init: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, u32) {
        let mut cpu: Vec<Vec<f32>> = init.to_vec();
        {
            let mut refs: Vec<&mut [f32]> = cpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            mercury_interp::run_kernel_f32(program, entry, &mut refs, interner).unwrap();
        }
        let mut gpu: Vec<Vec<f32>> = init.to_vec();
        let mut accel = gpu_accel::GpuAccel::new(g);
        {
            let mut refs: Vec<&mut [f32]> = gpu.iter_mut().map(|v| v.as_mut_slice()).collect();
            mercury_interp::run_kernel_f32_accel(program, entry, &mut refs, interner, &mut accel)
                .unwrap();
        }
        (cpu, gpu, accel.calls)
    }

    /// The other three GPU kernel families behind `--backend=gpu`: activation (vmath), reduction
    /// (sreduce), and a fused row norm (softmax). Each must fire on the device and match the interp
    /// oracle within tolerance (transcendental SFU approximations are looser than the GEMM bound).
    #[test]
    fn gpu_backend_activation_reduction_norm_match_interp() {
        let mut guard = mercury_codegen_gpu::gpu();
        let g = match guard.as_mut() {
            Some(g) => g,
            None => {
                eprintln!(
                    "skip gpu_backend_activation_reduction_norm_match_interp: no CUDA device"
                );
                return;
            }
        };
        let mut rng = Rng::new(0x5EED_1234);

        // 1) Activation: out[i] = silu(x[i]) -> mercury_vmath_f32 (SFU sigmoid approx on GPU).
        {
            let n = 64usize;
            let src = format!(
                "module m\nfn act(x:[f32;{n}], out:[f32;{n}]) {{ \
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

        // 2) Reduction: o[0] = Σ x·y -> mercury_sreduce_f32_parallel (dot, written to a [1] buffer).
        {
            let n = 1024usize;
            let src = format!(
                "@parallel fn dotp(x:[f32;{n}], y:[f32;{n}], o:[f32;1]) {{ \
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

        // 3) Fused norm: batched softmax over [R,C] -> mercury_norm_f32 (in place; GPU uses ex2.approx).
        {
            let (r, c) = (4usize, 16usize);
            let n = r * c;
            let src = format!(
                "module m\nfn sm(x:[f32;{n}]) {{ for row in 0..{r} {{ \
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
                "norm offload never fired (softmax did not lower to mercury_norm_f32)"
            );
            let s = assert_close("batched softmax", &gpu[0], &cpu[0], 2e-3, 5e-3);
            eprintln!(
                "gpu --backend softmax[{r}x{c}]: {calls} call(s), max_abs={:.2e} max_rel={:.2e}",
                s.max_abs, s.max_rel
            );
        }
    }
}
