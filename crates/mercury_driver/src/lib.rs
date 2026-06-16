//! `mercury_driver` — orchestrates the compilation pipeline.
//!
//! The driver owns the [`SourceMap`] and the [`DiagnosticSink`], runs each stage in order, and
//! honors the requested `--emit` target. Stages are wired in as they come online; today the
//! lexer is connected and `--emit=tokens` works end to end.

use std::path::PathBuf;

use mercury_diag::{DiagnosticSink, Renderer};
use mercury_span::{Interner, SourceMap};

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

/// Compiler invocation options, normally produced by the CLI.
#[derive(Clone, Debug)]
pub struct Options {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub emit: EmitStage,
    pub run: bool,
    pub opt_level: u8,
    pub color: bool,
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
        render_all(&renderer, &sink, &sm);
        print!("{}", mercury_lexer::dump(&tokens, sm.source(id)));
        return if sink.has_errors() { exit::COMPILE_ERROR } else { exit::OK };
    }

    // --- Parsing ---
    let mut interner = Interner::new();
    let (module, parse_diags) =
        mercury_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
    for d in parse_diags {
        sink.emit(d);
    }
    render_all(&renderer, &sink, &sm);

    if opts.emit == EmitStage::Ast {
        print!("{}", mercury_ast::print::print_module(&module, &interner));
        return if sink.has_errors() { exit::COMPILE_ERROR } else { exit::OK };
    }

    if sink.has_errors() {
        return exit::COMPILE_ERROR;
    }

    // --- Semantic analysis (name resolution, type checking, shape checking) ---
    let (sema, sema_diags) = mercury_sema::check(&module, &interner);
    for d in &sema_diags {
        eprintln!("{}", renderer.render(d, &sm));
    }
    if sema_diags.iter().any(|d| d.is_error()) {
        return exit::COMPILE_ERROR;
    }

    // --- MIR construction ---
    let (mut program, lower_diags) = mercury_mir_build::lower_program(&module, &sema, &interner);
    for d in &lower_diags {
        eprintln!("{}", renderer.render(d, &sm));
    }

    // High MIR is the pre-optimization form.
    if opts.emit == EmitStage::MirHigh {
        emit_mir(&program, &interner);
        return exit::OK;
    }

    // --- Optimization ---
    mercury_opt::optimize(&mut program, opts.opt_level);

    // --- Run via the interpreter ---
    if opts.run {
        use mercury_backend::{Artifact, Backend};
        let main = interner.intern("main");
        let backend = mercury_interp::Interpreter;
        return match backend.compile(&program, main, &interner) {
            Ok(Artifact::Executed { exit_code, stdout }) => {
                use std::io::Write;
                let _ = std::io::stdout().write_all(&stdout);
                exit_code as i32
            }
            Ok(_) => exit::OK,
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

    eprintln!(
        "error: `--emit={:?}` is not implemented yet (the pipeline currently reaches MIR; \
         try `--emit=tokens`, `--emit=ast`, `--emit=mir-high`, or `--emit=mir`)",
        opts.emit
    );
    exit::UNIMPLEMENTED
}

fn emit_mir(program: &mercury_mir::Program, interner: &Interner) {
    for f in &program.funcs {
        for ice in mercury_mir::verify::verify_function(f) {
            eprintln!("internal compiler error (MIR verify): {ice}");
        }
    }
    print!("{}", mercury_mir::print::print_program(program, interner));
}

fn render_all(renderer: &Renderer, sink: &DiagnosticSink, sm: &SourceMap) {
    for d in sink.diagnostics() {
        eprintln!("{}", renderer.render(d, sm));
    }
}
