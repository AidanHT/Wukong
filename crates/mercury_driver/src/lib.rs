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

    eprintln!(
        "error: `--emit={:?}` is not implemented yet (the pipeline currently reaches the parser; \
         try `--emit=tokens` or `--emit=ast`)",
        opts.emit
    );
    exit::UNIMPLEMENTED
}

fn render_all(renderer: &Renderer, sink: &DiagnosticSink, sm: &SourceMap) {
    for d in sink.diagnostics() {
        eprintln!("{}", renderer.render(d, sm));
    }
}
