//! `wukongc` — the Wukong compiler command-line driver.

use std::path::PathBuf;
use std::process::ExitCode;

use wukong_driver::{compile, EmitStage, Options};

const USAGE: &str = "\
wukongc — the Wukong compiler

USAGE:
    wukongc [OPTIONS] <input.wk>

OPTIONS:
    --emit=<stage>     Emit an intermediate artifact and stop. One of:
                       tokens, ast, mir-high, mir, grad, llvm-ir, obj, exe  (default: exe)
                       (grad = reverse-mode backward MIR of a loss fn; see --grad-of/--grad-wrt)
    --grad-of=<fn>     Function to differentiate for --emit=grad / --train  (default: loss)
    --grad-wrt=<i,..>  Comma-separated parameter indices to differentiate w.r.t. (default: all
                       buffer parameters of the loss function). For --train, these are the
                       trainable weights; other buffers are fixed data; the last param is the loss.
    --train            Run a fwd->bwd->optimizer loop on the loss fn and print the loss trajectory
    --train-steps=<n>  Number of training steps  (default: 100)
    --train-lr=<f>     Learning rate  (default: 0.01)
    --train-opt=<o>    Optimizer: sgd, adamw  (default: sgd)
    --train-seed=<n>   Seed for deterministic buffer initialization
    --run              Compile and run (interpreter by default; see --backend)
    --backend=<b>      Execution backend: interp, native, gpu, gpu-native  (default: interp)
                       (gpu/gpu-native require --features gpu and a CUDA device; gpu is the
                       recognizer offload, gpu-native lowers the whole program's MIR to PTX)
    -o <path>          Write the artifact to <path>. Only --emit=obj and --emit=exe write a
                       file; every other stage prints its artifact on stdout.
    -O0|-O1|-O2|-O3    Optimization level (default: -O0)
    --color=<when>     Colorize diagnostics: auto, always, never  (default: auto)
    --error-format=<f> Diagnostic output format: human, json  (default: human)
    --explain <CODE>   Print the extended explanation for an error code, then exit
    -h, --help         Print this help
    -V, --version      Print version

EXAMPLES:
    wukongc --emit=tokens examples/vadd.wk
    wukongc --run examples/gemm.wk
    wukongc --emit=grad --grad-of=loss model.wk
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(Some(opts)) => ExitCode::from(run_compile(opts) as u8),
        Ok(None) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}\n");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Run the compile on a worker thread with a large stack.
///
/// The recursive-descent parser and the recursive AST consumers (sema, MIR lowering, and even the
/// AST's `Drop`) descend in lock-step with how deeply the source nests. Legitimately deep but
/// bounded input — up to the parser's `E0209` nesting limit — would exhaust the default ~1 MB main
/// thread stack and crash the process with no diagnostic. A roomy stack keeps the real overflow
/// threshold far above the parser's limit, so pathological input is rejected cleanly by `E0209`
/// rather than overflowing. (This mirrors how `rustc` runs its front-end on a dedicated stack.)
fn run_compile(opts: Options) -> i32 {
    const STACK_SIZE: usize = 256 * 1024 * 1024;
    std::thread::Builder::new()
        .name("wukongc-compile".to_string())
        .stack_size(STACK_SIZE)
        .spawn(move || compile(&opts))
        .expect("failed to spawn compiler thread")
        .join()
        .unwrap_or(wukong_driver::exit::COMPILE_ERROR)
}

/// Print the extended explanation for an error code to stdout.
fn print_explanation(code: &str) -> Result<(), String> {
    match wukong_driver::explain(code) {
        Some(e) => {
            println!("{}: {}\n", e.code, e.title);
            println!("{}", e.body);
            Ok(())
        }
        None => Err(format!(
            "unknown error code `{code}`. Run `wukongc --explain E0502` for an example."
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
                println!("wukongc {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--explain" => {
                i += 1;
                let code = args
                    .get(i)
                    .ok_or("`--explain` requires an error code, e.g. E0502")?;
                print_explanation(code)?;
                return Ok(None);
            }
            "--run" => opts.run = true,
            _ if arg.starts_with("--backend=") => {
                let b = &arg["--backend=".len()..];
                opts.backend = match b {
                    "interp" | "interpreter" => wukong_driver::BackendKind::Interp,
                    "native" | "cranelift" => wukong_driver::BackendKind::Native,
                    "gpu" | "cuda" => wukong_driver::BackendKind::Gpu,
                    "gpu-native" | "gpu-lower" => wukong_driver::BackendKind::GpuLower,
                    other => {
                        return Err(format!(
                            "unknown --backend value `{other}` \
                             (expected interp, native, gpu, or gpu-native)"
                        ))
                    }
                };
            }
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
            _ if arg.starts_with("--grad-of=") => {
                opts.grad.of = Some(arg["--grad-of=".len()..].to_string());
            }
            _ if arg.starts_with("--grad-wrt=") => {
                let list = &arg["--grad-wrt=".len()..];
                opts.grad.wrt = list
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        format!("--grad-wrt expects comma-separated parameter indices, got `{list}`")
                    })?;
            }
            "--train" => opts.grad.train = true,
            _ if arg.starts_with("--train-steps=") => {
                let v = &arg["--train-steps=".len()..];
                opts.grad.train_steps = v
                    .parse()
                    .map_err(|_| format!("--train-steps expects a non-negative integer, got `{v}`"))?;
            }
            _ if arg.starts_with("--train-lr=") => {
                let v = &arg["--train-lr=".len()..];
                opts.grad.train_lr = v
                    .parse()
                    .map_err(|_| format!("--train-lr expects a float, got `{v}`"))?;
            }
            _ if arg.starts_with("--train-opt=") => {
                let v = &arg["--train-opt=".len()..];
                opts.grad.train_opt = match v {
                    "sgd" => wukong_driver::TrainOpt::Sgd,
                    "adamw" => wukong_driver::TrainOpt::AdamW,
                    other => {
                        return Err(format!("--train-opt expects sgd or adamw, got `{other}`"))
                    }
                };
            }
            _ if arg.starts_with("--train-seed=") => {
                let v = &arg["--train-seed=".len()..];
                opts.grad.train_seed = v
                    .parse()
                    .map_err(|_| format!("--train-seed expects an integer, got `{v}`"))?;
            }
            _ if arg.starts_with("--error-format=") => {
                let fmt = &arg["--error-format=".len()..];
                opts.error_format = match fmt {
                    "human" => wukong_driver::ErrorFormat::Human,
                    "json" => wukong_driver::ErrorFormat::Json,
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

    // A requested `--emit=<stage>` and `--run` cannot both be honoured: the driver dispatches
    // exactly ONE of them and silently drops the other, while still exiting 0. Which one wins
    // depends on where the stage sits relative to the run block, so the discarded work is not even
    // consistent: `--run --emit=obj|exe|mir|llvm-ir` runs the program and writes/prints no artifact,
    // and `--run --emit=tokens|ast|mir-high|grad` prints the artifact and never runs. Reject the
    // combination rather than report success for work that was not done. (`emit_explicit` exists
    // precisely for this check; it used to be discarded at the end of `parse_args`.)
    if opts.run && emit_explicit {
        return Err(
            "`--run` cannot be combined with `--emit=<stage>`: the compiler honours exactly one of \
             them and would silently discard the other"
                .to_string(),
        );
    }

    // `-o` names an artifact FILE, and only the two artifact stages write one. For every textual
    // stage the driver prints to stdout and the flag was silently ignored — no file appeared and
    // the process still exited 0, so `wukongc --emit=llvm-ir -o out.ll p.wk && llc out.ll` failed
    // at the *next* command with a missing-file error naming the wrong culprit.
    if opts.output.is_some() {
        if opts.run {
            return Err(
                "`-o` has no meaning with `--run`: running the program writes no artifact file"
                    .to_string(),
            );
        }
        if !matches!(opts.emit, EmitStage::Obj | EmitStage::Exe) {
            return Err(
                "`-o` is only supported with `--emit=obj` and `--emit=exe`: every other stage \
                 prints its artifact on stdout (redirect it instead)"
                    .to_string(),
            );
        }
    }

    Ok(Some(opts))
}
