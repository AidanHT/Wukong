//! `wukongc` — the Wukong compiler command-line driver.

use std::path::PathBuf;
use std::process::ExitCode;

use wukong_driver::{compile, EmitStage, Options};

const USAGE: &str = "\
wukongc — the Wukong compiler

USAGE:
    wukongc [OPTIONS] <input.wk>

OPTIONS:
    --verify-each     Verify MIR after every optimizer pass (also in release builds)
    --emit=<stage>     Emit an intermediate artifact and stop. One of:
                       tokens, ast, check, mir-high, mir, grad, llvm-ir, obj, exe  (default: exe)
                       (grad = reverse-mode backward MIR of a loss fn; see --grad-of/--grad-wrt)
                       (check = run lex/parse/imports/sema only, report diagnostics, no lowering)
    --grad-of=<fn>     Function to differentiate for --emit=grad / --train  (default: loss)
    --grad-wrt=<i,..>  Comma-separated parameter indices to differentiate w.r.t. (default: all
                       buffer parameters of the loss function). For --train, these are the
                       trainable weights; other buffers are fixed data; the last param is the loss.
    --train            Run a fwd->bwd->optimizer loop on the loss fn and print the loss trajectory
                       (interp or native JIT; concrete f32 buffers; incompatible with --emit/-o)
    --train-steps=<n>  Number of training steps  (default: 100)
    --train-lr=<f>     Learning rate  (default: 0.01)
    --train-weight-decay=<f>  AdamW decoupled weight decay, finite and >= 0 (default: 0)
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
                       auto honors NO_COLOR, CLICOLOR_FORCE and TERM=dumb, and enables Windows VT
    --error-format=<f> Diagnostic output format: human, json  (default: human)
    --error-limit=<n>  Print at most n errors before suppressing the rest (0 = unlimited; default)
    --explain [CODE]   Print the extended explanation for an error code (no code: list all codes)
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

/// How the user asked diagnostics to be colored (`--color=<when>`), resolved to a concrete on/off
/// decision by [`resolve_color`] once the whole environment has been read.
#[derive(Clone, Copy)]
enum ColorChoice {
    Always,
    Never,
    Auto,
}

/// Resolve a [`ColorChoice`] to a concrete on/off decision, honoring the standard environment
/// overrides and (on Windows) enabling VT processing so the ANSI escapes the renderer emits are
/// actually interpreted rather than printed literally.
///
/// `auto` precedence matches the de-facto contract shared by `cargo`, `ls` and rustc:
/// `CLICOLOR_FORCE` (non-empty, not `0`) forces color on; otherwise `NO_COLOR` (non-empty) or
/// `TERM=dumb` forces it off; otherwise color follows whether stderr is a terminal. `always` and
/// `never` ignore the environment.
///
/// `always` forces color on and only *tries* to switch the console into VT mode (best effort),
/// matching rustc: the user has explicitly accepted escape codes, including down a pipe or into a
/// file where no console mode applies. For `auto`, VT setup that fails turns color back off, because
/// raw escape codes in an interactive console that cannot interpret them are worse than plain text.
fn resolve_color(when: ColorChoice) -> bool {
    match when {
        ColorChoice::Always => {
            let _ = enable_vt_processing();
            true
        }
        ColorChoice::Never => false,
        // CLICOLOR_FORCE is an explicit "force": like `always`, it must survive a pipe or a file,
        // where there is no console to switch into VT mode (`enable_vt_processing` is false there).
        ColorChoice::Auto if clicolor_force() => {
            let _ = enable_vt_processing();
            true
        }
        ColorChoice::Auto => auto_color() && enable_vt_processing(),
    }
}

/// `CLICOLOR_FORCE` set to anything other than empty or `"0"`.
fn clicolor_force() -> bool {
    std::env::var_os("CLICOLOR_FORCE").is_some_and(|v| !v.is_empty() && v != "0")
}

fn auto_color() -> bool {
    use std::env;
    let nonempty = |k: &str| env::var_os(k).is_some_and(|v| !v.is_empty());
    // CLICOLOR_FORCE wins when set to anything other than "0".
    if clicolor_force() {
        return true;
    }
    if nonempty("NO_COLOR") {
        return false;
    }
    if env::var_os("TERM").is_some_and(|v| v == "dumb") {
        return false;
    }
    std::io::IsTerminal::is_terminal(&std::io::stderr())
}

/// Enable ANSI VT processing on the stderr console (Windows). Returns whether color should be used:
/// `true` when the escapes will render (already a VT-capable console, or we switched it on), `false`
/// when stderr is not a console we can configure. On non-Windows platforms this is always `true`.
#[cfg(windows)]
fn enable_vt_processing() -> bool {
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::raw::HANDLE;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    // The two kernel32 entry points keep their Windows PascalCase names (they are the link symbols).
    #[allow(non_snake_case)]
    extern "system" {
        fn GetConsoleMode(handle: HANDLE, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: HANDLE, mode: u32) -> i32;
    }
    // `RawHandle` is `*mut c_void` i.e. `HANDLE`; no cast is needed.
    let handle: HANDLE = std::io::stderr().as_raw_handle();
    // SAFETY: `handle` is the process's live stderr handle; both calls only read/write a mode flag.
    unsafe {
        let mut mode: u32 = 0;
        if GetConsoleMode(handle, &mut mode) == 0 {
            // stderr is redirected to a file or pipe: there is no console to interpret escapes.
            return false;
        }
        if mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0 {
            return true;
        }
        SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
    }
}

#[cfg(not(windows))]
fn enable_vt_processing() -> bool {
    true
}

/// List every diagnostic code with its one-line title (`--explain` with no code).
fn list_all_codes() {
    println!(
        "Wukong diagnostic codes (run `wukongc --explain <CODE>` for the full explanation):\n"
    );
    for e in wukong_driver::all_explanations() {
        println!("  {:<6}{}", e.code, e.title);
    }
}

/// Print the extended explanation for an error code to stdout.
fn print_explanation(code: &str) -> Result<(), String> {
    match wukong_driver::explain(code) {
        Some(e) => {
            println!("{}: {}\n", e.code, e.title);
            println!("{}", e.body);
            // The worked example pair, when the entry has one. `tests/explain.rs` compiles both, so
            // what is printed here is exactly what the compiler rejects and accepts.
            if let (Some(bad), Some(good)) = (e.bad, e.good) {
                println!("\nErroneous code example:\n\n{bad}");
                println!("Corrected:\n\n{good}");
            }
            Ok(())
        }
        None => Err(format!(
            "unknown error code `{code}`. Run `wukongc --explain E0502` for an example."
        )),
    }
}

/// Parse CLI arguments. `Ok(None)` means a help / version / `--explain` message was already printed
/// and we should exit successfully without compiling. The two rejected flag *combinations* are
/// checked at the end, after the whole argv has been seen.
fn parse_args(args: &[String]) -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut input: Option<PathBuf> = None;
    let mut emit_explicit = false;
    let mut color_choice = ColorChoice::Auto;

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
                // `--explain <CODE>` explains one code; a bare `--explain` lists every code.
                match args.get(i + 1) {
                    Some(code) if !code.starts_with('-') => {
                        i += 1;
                        print_explanation(code)?;
                    }
                    _ => list_all_codes(),
                }
                return Ok(None);
            }
            "--run" => opts.run = true,
            "--verify-each" => opts.verify_each = true,
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
                        format!(
                            "--grad-wrt expects comma-separated parameter indices, got `{list}`"
                        )
                    })?;
            }
            "--train" => opts.grad.train = true,
            _ if arg.starts_with("--train-steps=") => {
                let v = &arg["--train-steps=".len()..];
                opts.grad.train_steps = v.parse().map_err(|_| {
                    format!("--train-steps expects a non-negative integer, got `{v}`")
                })?;
            }
            _ if arg.starts_with("--train-lr=") => {
                let v = &arg["--train-lr=".len()..];
                opts.grad.train_lr = v
                    .parse()
                    .map_err(|_| format!("--train-lr expects a float, got `{v}`"))?;
            }
            _ if arg.starts_with("--train-weight-decay=") => {
                let v = &arg["--train-weight-decay=".len()..];
                opts.grad.train_weight_decay = v
                    .parse()
                    .map_err(|_| format!("--train-weight-decay expects a float, got `{v}`"))?;
                if !opts.grad.train_weight_decay.is_finite() || opts.grad.train_weight_decay < 0.0 {
                    return Err("--train-weight-decay must be finite and non-negative".into());
                }
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
                color_choice = match when {
                    "always" => ColorChoice::Always,
                    "never" => ColorChoice::Never,
                    "auto" => ColorChoice::Auto,
                    other => return Err(format!("unknown --color value `{other}`")),
                };
            }
            _ if arg.starts_with("--error-limit=") => {
                let v = &arg["--error-limit=".len()..];
                let n: usize = v.parse().map_err(|_| {
                    format!("--error-limit expects a non-negative integer, got `{v}`")
                })?;
                // 0 means "no limit", matching rustc.
                opts.error_limit = if n == 0 { None } else { Some(n) };
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

    // Resolve color once the whole environment is known: `--color` (or its `auto` default) plus the
    // NO_COLOR / CLICOLOR_FORCE / TERM conventions, and VT setup on Windows.
    opts.color = resolve_color(color_choice);

    let input = input.ok_or("no input file given")?;
    opts.input = input;

    // A requested `--emit=<stage>` and `--run` cannot both be honoured: the driver dispatches
    // exactly ONE of them and silently drops the other, while still exiting 0. Which one wins
    // depends on where the stage sits relative to the run block, so the discarded work is not even
    // consistent: `--run --emit=obj|exe|mir|llvm-ir` runs the program and writes/prints no artifact,
    // and `--run --emit=tokens|ast|mir-high|grad` prints the artifact and never runs. Reject the
    // combination rather than report success for work that was not done. (`emit_explicit` exists
    // precisely for this check; it used to be discarded at the end of `parse_args`.)
    if opts.grad.train && (emit_explicit || opts.output.is_some()) {
        return Err(
            "`--train` cannot be combined with explicit `--emit=<stage>` or `-o`: \
                    training executes immediately and emits no compiled artifact"
                .into(),
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_each_flag_survives_all_optimization_levels_and_training() {
        for level in ["-O0", "-O1", "-O2", "-O3"] {
            for mode in ["--run", "--emit=mir", "--emit=grad", "--train"] {
                let args = ["--verify-each", level, mode, "input.wk"].map(str::to_owned);
                let opts = parse_args(&args).unwrap().unwrap();
                assert!(opts.verify_each);
                assert_eq!(opts.opt_level, level[2..].parse::<u8>().unwrap());
            }
        }
        assert!(
            !parse_args(&["input.wk".to_owned()])
                .unwrap()
                .unwrap()
                .verify_each
        );
        assert!(USAGE.contains("--verify-each"));
    }

    #[test]
    fn error_limit_parses_and_zero_means_unlimited() {
        let o = parse_args(&["--error-limit=5".to_owned(), "p.wk".to_owned()])
            .unwrap()
            .unwrap();
        assert_eq!(o.error_limit, Some(5));
        let z = parse_args(&["--error-limit=0".to_owned(), "p.wk".to_owned()])
            .unwrap()
            .unwrap();
        assert_eq!(z.error_limit, None);
        assert!(parse_args(&["--error-limit=x".to_owned(), "p.wk".to_owned()]).is_err());
    }

    #[test]
    fn emit_check_is_a_stage() {
        let o = parse_args(&["--emit=check".to_owned(), "p.wk".to_owned()])
            .unwrap()
            .unwrap();
        assert_eq!(o.emit, EmitStage::Check);
        assert!(USAGE.contains("check ="));
    }

    #[test]
    fn explain_with_no_code_is_accepted() {
        // A bare `--explain` lists all codes and exits successfully (Ok(None)).
        assert!(parse_args(&["--explain".to_owned()]).unwrap().is_none());
    }
}
