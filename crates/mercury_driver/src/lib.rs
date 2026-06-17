//! `mercury_driver` — orchestrates the compilation pipeline.
//!
//! The driver owns the [`SourceMap`] and the [`DiagnosticSink`], runs each stage in order, and
//! honors the requested `--emit` target. Stages are wired in as they come online; today the
//! lexer is connected and `--emit=tokens` works end to end.

use std::path::PathBuf;

use mercury_diag::{Diagnostic, DiagnosticSink, Renderer};
use mercury_span::{Interner, SourceMap};

pub use mercury_diag::{all_explanations, explain, Explanation};

/// Which intermediate (or final) artifact the user asked to produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmitStage {
    Tokens,
    Ast,
    MirHigh,
    Mir,
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
    mercury_opt::optimize(&mut program, opts.opt_level);

    // --- Run via the selected backend (interpreter by default, Cranelift JIT with --backend=native) ---
    if opts.run {
        let main = interner.intern("main");
        let result = match opts.backend {
            BackendKind::Interp => mercury_interp::run_with_output(&program, main, &interner),
            BackendKind::Native => mercury_codegen_cranelift::jit_run(&program, main, &interner),
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

/// The C runtime linked into native executables: it backs the `print`/`assert` intrinsics that the
/// Cranelift object leaves as undefined imports. The JIT path binds the same symbols to Rust
/// functions instead, so the two stay in lockstep.
const MERCURY_RT_C: &str = "#include <stdio.h>\n\
#include <stdlib.h>\n\
void mercury_rt_print_i64(long long x) { printf(\"%lld\\n\", x); }\n\
void mercury_rt_print_f64(double x) { printf(\"%g\\n\", x); }\n\
void mercury_rt_assert(long long c) { if (!c) { fprintf(stderr, \"assertion failed\\n\"); exit(101); } }\n";

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
