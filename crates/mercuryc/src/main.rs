//! `mercuryc` — the Mercury compiler command-line driver.

use std::path::PathBuf;
use std::process::ExitCode;

use mercury_driver::{compile, EmitStage, Options};

const USAGE: &str = "\
mercuryc — the Mercury compiler

USAGE:
    mercuryc [OPTIONS] <input.mer>

OPTIONS:
    --emit=<stage>     Emit an intermediate artifact and stop. One of:
                       tokens, ast, mir-high, mir, llvm-ir, obj, exe  (default: exe)
    --run              Compile and run via the built-in interpreter
    -o <path>          Write output to <path>
    -O0|-O1|-O2|-O3    Optimization level (default: -O0)
    --color=<when>     Colorize diagnostics: auto, always, never  (default: auto)
    --error-format=<f> Diagnostic output format: human, json  (default: human)
    --explain <CODE>   Print the extended explanation for an error code, then exit
    -h, --help         Print this help
    -V, --version      Print version

EXAMPLES:
    mercuryc --emit=tokens examples/vadd.mer
    mercuryc --run examples/matmul.mer
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(Some(opts)) => ExitCode::from(compile(&opts) as u8),
        Ok(None) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}\n");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Print the extended explanation for an error code to stdout.
fn print_explanation(code: &str) -> Result<(), String> {
    match mercury_driver::explain(code) {
        Some(e) => {
            println!("{}: {}\n", e.code, e.title);
            println!("{}", e.body);
            Ok(())
        }
        None => Err(format!(
            "unknown error code `{code}`. Run `mercuryc --explain E0502` for an example."
        )),
    }
}

/// Parse CLI arguments. `Ok(None)` means a help/version message was printed and we should exit
/// successfully without compiling.
fn parse_args(args: &[String]) -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut input: Option<PathBuf> = None;
    let mut emit_explicit = false;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("mercuryc {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--explain" => {
                i += 1;
                let code = args.get(i).ok_or("`--explain` requires an error code, e.g. E0502")?;
                print_explanation(code)?;
                return Ok(None);
            }
            "--run" => opts.run = true,
            "-O0" => opts.opt_level = 0,
            "-O1" => opts.opt_level = 1,
            "-O2" => opts.opt_level = 2,
            "-O3" => opts.opt_level = 3,
            "-o" => {
                i += 1;
                let path = args.get(i).ok_or("`-o` requires a path argument")?;
                opts.output = Some(PathBuf::from(path));
            }
            _ if arg.starts_with("--emit=") => {
                let stage = &arg["--emit=".len()..];
                opts.emit = EmitStage::parse(stage)
                    .ok_or_else(|| format!("unknown --emit target `{stage}`"))?;
                emit_explicit = true;
            }
            _ if arg.starts_with("--error-format=") => {
                let fmt = &arg["--error-format=".len()..];
                opts.error_format = match fmt {
                    "human" => mercury_driver::ErrorFormat::Human,
                    "json" => mercury_driver::ErrorFormat::Json,
                    other => return Err(format!("unknown --error-format value `{other}`")),
                };
            }
            _ if arg.starts_with("--color=") => {
                let when = &arg["--color=".len()..];
                opts.color = match when {
                    "always" => true,
                    "never" => false,
                    "auto" => std::io::IsTerminal::is_terminal(&std::io::stderr()),
                    other => return Err(format!("unknown --color value `{other}`")),
                };
            }
            _ if arg.starts_with('-') && arg != "-" => {
                return Err(format!("unknown option `{arg}`"));
            }
            _ => {
                if input.is_some() {
                    return Err("multiple input files are not supported yet".to_string());
                }
                input = Some(PathBuf::from(arg));
            }
        }
        i += 1;
    }

    // Default color decision when not explicitly set: respect TTY.
    if !args.iter().any(|a| a.starts_with("--color=")) {
        opts.color = std::io::IsTerminal::is_terminal(&std::io::stderr());
    }

    let input = input.ok_or("no input file given")?;
    opts.input = input;

    // `--run` implies we don't need a final emit target.
    let _ = emit_explicit;
    Ok(Some(opts))
}
