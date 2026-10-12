//! `wukongc` — the Wukong compiler command-line driver.

use std::path::PathBuf;
use std::process::ExitCode;

use wukong_driver::{compile, EmitStage, Options};

mod fmt_cmd;

const USAGE: &str = "\
wukongc — the Wukong compiler

USAGE:
    wukongc [OPTIONS] <input.wk> [-- <program args>...]
    wukongc [OPTIONS] -              (read the program from standard input)
    wukongc [OPTIONS] -e <source>    (the program is the <source> argument)
    wukongc --repl [OPTIONS]         (an interactive session on standard input; `:help` inside)
    wukongc --doctest [OPTIONS] <input.wk>   (run the ```wk examples in its doc comments)
    wukongc fmt [--check] [--stdout] <path|->...   (format source files; `wukongc fmt --help`)

OPTIONS:
    --verify-each     Verify MIR after every optimizer pass (also in release builds)
    --time-passes      Print per-stage compile times (with per-optimizer-pass and codegen
                       splits and a MIR size summary) to stderr when the compile ends
    --time-trace=<f>   Write the same timings to <f> as Chrome trace-event JSON
                       (chrome://tracing, Perfetto, speedscope)
    --depfile=<f>      Write a Make-syntax dependency file to <f> naming every source file
                       read (the root and each import); its targets are the files the compile
                       writes (the -o artifact of --emit=obj/exe, each <stem>.<ext> of an
                       --emit list with -o), otherwise <f> itself (not when the last stage is
                       tokens, ast or doc)
    --emit=<stage>     Emit an intermediate artifact and stop. One of:
                       tokens, ast, doc, check, mir-high, mir, mir-text, grad, llvm-ir, obj, exe
                       (default: exe)
                       (doc = the input file's documentation as Markdown: its //! docs, then
                       every item in source order with its signature as `wukongc fmt` prints
                       it and its /// docs; fields, variants and methods under their item)
                       (mir-text = the optimized MIR with every type, float bits and parameter
                       promise spelled out; `wukongc --emit=mir|mir-text [-O<n>] [--passes=..]
                       f.mir` reads it back, verifies, optimizes and prints it. A .mir input takes
                       only those two stages and no --run)
                       (grad = reverse-mode backward MIR of a loss fn; see --grad-of/--grad-wrt)
                       (check = run lex/parse/imports/sema only, report diagnostics, no lowering)
    --emit=<stage>,<stage>...
                       Emit several artifacts from ONE compile, in pipeline order whatever the
                       written order. Text artifacts go to stdout, each under a
                       `==> <stage> <==` header line; with -o <stem> every artifact goes to
                       <stem>.<ext> instead: .tokens, .ast, .md (doc), .mir-high, .mir-low
                       (mir), .mir (mir-text), .grad, .ll (llvm-ir), .o (obj), the host's
                       executable extension (exe). obj or exe in a list needs -o; check and grad
                       combine only with tokens/ast/doc; obj not with exe. The last stage
                       decides the compile: with exe, mir/mir-text/llvm-ir show the program
                       pruned to what main reaches; with obj and --grad-of, the gradient export
                       (<fn>_grad included, at least -O1)
    --grad-of=<fn>     Function to differentiate for --emit=grad / --train  (default: loss).
                       With --emit=obj, the object also exports the gradient as <fn>_grad
    --grad-wrt=<i,..>  Comma-separated parameter indices to differentiate w.r.t. (default: all
                       buffer parameters of the loss function). For --train, these are the
                       trainable weights; other buffers are fixed data; the last param is the loss.
    --train            Run a fwd->bwd->optimizer loop on the loss fn and print the loss trajectory
                       (interp or native JIT; concrete f32 buffers). With --emit=exe|obj (or -o),
                       build the loop into an executable/object instead: it prints exactly the
                       trajectory --train prints (not with --train-init/-data/-save)
    --train-steps=<n>  Number of training steps  (default: 100)
    --train-lr=<f>     Learning rate  (default: 0.01)
    --train-weight-decay=<f>  AdamW decoupled weight decay, finite and >= 0 (default: 0)
    --train-opt=<o>    Optimizer: sgd, adamw  (default: sgd)
    --train-seed=<n>   Seed for deterministic buffer initialization (uniform in [-0.4, 0.4))
    --train-init=<i>=<path>  Load buffer <i> from <path> (raw little-endian f32, exactly the
                       buffer's elements) instead of the seeded init. Repeatable
    --train-data=<i>=<path>  Stream fixed-input buffer <i> from <path>, a file of whole batches
                       (raw little-endian f32): update s trains on batch s mod batches. Repeatable
    --train-save=<i>=<path>  Write buffer <i> (e.g. a trained weight) to <path> as raw
                       little-endian f32 after training. Repeatable
    --train-clip=<f>   Clip the global gradient L2 norm to <f> before each update (finite, > 0)
    --train-warmup=<n> Ramp the learning rate linearly over the first <n> updates  (default: 0)
    --train-schedule=<s>  Learning rate after the warmup: constant, cosine  (default: constant)
    --run              Compile and run (interpreter by default; see --backend). The program's
                       stdout streams as it runs: per line on a terminal, in blocks into a pipe
                       or a file
    --repl             Read standard input a line at a time and evaluate it: items (fn, struct,
                       enum, const, import, impl, ...) are kept; a statement runs and an
                       expression prints its value; an input continues while a bracket is open.
                       Commands: :help, :quit, :reset, :items, :type <expr>. Takes --backend
                       (interp, native), -O, --fp, --target-cpu, -A/-W/-D, --color,
                       --error-format, --error-limit and --threads
    --doctest          Compile and run every ```wk example in the input's doc comments (/// and
                       //!) and report each (`doctest <file>:<line> ... ok`; exit 1 if any
                       failed). An example sees the file's items (its own main aside); one with
                       no fn main is wrapped in `fn main() -> i32 { ...; return 0; }`. It must
                       exit 0; tagged ```wk,no_run it must only compile, ```wk,compile_fail[,E0503]
                       it must fail to compile (with that code), ```wk,ignore it is skipped.
                       Takes --backend (interp, native), -O, --fp, --target-cpu, --color,
                       --error-format and --error-limit
    --backend=<b>      Execution backend: interp, native, gpu, gpu-native  (default: interp)
                       (gpu/gpu-native require --features gpu and a CUDA device; gpu is the
                       recognizer offload, gpu-native lowers the whole program's MIR to PTX)
    --target-cpu=<c>   Instruction-set level for native code (--emit=obj/exe, --backend=native):
                       native, x86-64, x86-64-v2, x86-64-v3, x86-64-v4  (default: native = the
                       host's features). Below x86-64-v3 the raw-AVX2 loop kernels are not used,
                       so the object runs on any CPU of that level; the JIT refuses a level the
                       host lacks.
    -o <path>          Write the artifact to <path>. Only --emit=obj and --emit=exe write a
                       file; every other stage prints its artifact on stdout. With several
                       --emit stages <path> is the stem every artifact is written under
    -g                 Put DWARF debug info in an --emit=obj/exe artifact: a line table mapping
                       every function's machine code to its source file:line:column, and one
                       subprogram per function. The machine code is unchanged; other stages and
                       --train ignore it. (JIT profiling: WUKONG_PERF_MAP=1 makes
                       --backend=native write perf-<pid>.map, one `<start> <size> <name>` per
                       function, to /tmp on Linux and the temp directory elsewhere)
    -O0|-O1|-O2|-O3    Optimization level (default: -O0)
    --passes=<list>    Run exactly this comma-separated optimizer pass list instead of the -O
                       level's (-O still sets the native code generator's level); `--passes=`
                       runs none. Consecutive function passes run together to a fixpoint:
                       sroa, mem2reg, simplify, jump-thread, simplify-cfg, simplify-phis, dce,
                       tre, cse, dse, loop-canon, loop-delete, fuse, licm, bce, unswitch,
                       promote, vectorize. Program steps run once where named: inline, unroll,
                       slp, form-switches. -O2 is the list `inline,sroa,...,vectorize,unroll,
                       slp,form-switches` (docs/internals.md). Not with --emit=tokens, ast,
                       check, mir-high or grad, --train or --grad-of
    --print-after=<p>  Print the MIR to stderr after each run of pass <p> on a function (a run
                       that changed nothing prints a one-line note), or after program step <p>;
                       `all` matches every pass and step. Runs the -O level's list unless
                       --passes names one. Same restrictions as --passes
    --fp=<mode>        Floating-point mode: fast, strict  (default: fast)
                       fast contracts `x + y*z` into one FMA, reassociates vectorized float
                       reductions and dispatches recognized kernels; strict evaluates every
                       function in source order with one rounding per operation (the
                       program-wide form of the @strict_fp function attribute)
    --threads=<n>      Worker threads for @parallel regions and _parallel kernels in this run
                       (1..=4096; default: every usable CPU). Sets WUKONG_NUM_THREADS for --run
                       and --train; a built executable reads WUKONG_NUM_THREADS at startup.
    --remarks=dispatch Print a note at every loop (or function) the compiler replaced with a
                       recognized runtime kernel call, naming the kernel symbol(s). The output is
                       unchanged; `@generic` on a loop or fn opts it out of recognition.
    --color=<when>     Colorize diagnostics: auto, always, never  (default: auto)
                       auto honors NO_COLOR, CLICOLOR_FORCE and TERM=dumb, and enables Windows VT
    --error-format=<f> Diagnostic output format: human, json  (default: human)
    -A|-W|-D <lint>    Allow, warn on, or deny (report as an error) a lint; `warnings` names
                       every lint, so `-D warnings` fails the build on any warning. A lint is
                       named or coded: unused_variables (W0001), unused_mut (W0002),
                       unreachable_code (W0003), unused_result (W0004), dead_code (W0005),
                       unwritten_output (W0006), unknown_lints (W0007), unused_imports (W0008),
                       unreachable_patterns (W0020), missing_semicolon (W0030),
                       ambiguous_continuation (W0031).
                       Repeatable; `-Dwarnings` also works. In source: @allow(<lint>),
                       @warn(..), @deny(..)
    --error-limit=<n>  Print at most n errors before suppressing the rest (0 = unlimited; default)
    --explain [CODE]   Print the extended explanation for an error code (no code: list all codes)
    -e <source>        Compile the program text <source> instead of a file (`-` reads it
                       from standard input instead)
    -- <args>...       Everything after `--` is passed to the program `--run` executes:
                       arg(1), arg(2), ... (arg(0) is the program); needs --run
    -h, --help         Print this help
    -V, --version      Print version

EXAMPLES:
    wukongc --emit=tokens examples/vadd.wk
    wukongc --run examples/gemm.wk
    wukongc --emit=grad --grad-of=loss model.wk
    wukongc --train --train-opt=adamw --emit=exe -o train.exe model.wk
    wukongc --run model.wk -- data/weights.bin 64
    wukongc --run -e 'fn main() -> i32 { printf(\"{}\\n\", 6 * 7); return 0; }'
    wukongc --emit=mir,llvm-ir,obj -O2 -o kernel kernel.wk
    wukongc --emit=doc examples/fib.wk > fib.md
    wukongc --doctest examples/fib.wk
    wukongc --repl --backend=native
";

fn main() -> ExitCode {
    // `wukongc fmt …` is a subcommand: it takes over before any compiler flag is read.
    if std::env::args().nth(1).as_deref() == Some("fmt") {
        return fmt_cmd::run(&std::env::args().skip(2).collect::<Vec<_>>());
    }
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--threads` configures the runtime this process links (the interpreter and the JIT call the
    // same `wukong_runtime` pools), so it is applied here, before any thread exists, rather than
    // carried through the driver's `Options`.
    match take_threads(&mut args) {
        Ok(Some(n)) => std::env::set_var("WUKONG_NUM_THREADS", n.to_string()),
        Ok(None) => {}
        Err(msg) => {
            eprintln!("error: {msg}\n");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    }
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
        // The REPL compiles every input on this same roomy stack.
        .spawn(move || {
            if opts.repl {
                wukong_driver::repl::run(&opts)
            } else if opts.doctest {
                wukong_driver::doctest::run(&opts)
            } else {
                compile(&opts)
            }
        })
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

/// Remove every `--threads=<n>` from `args` and return the last one's value: a positive decimal no
/// larger than 4096 (the runtime's own bound on `WUKONG_NUM_THREADS`), else a usage error.
fn take_threads(args: &mut Vec<String>) -> Result<Option<usize>, String> {
    // Everything after `--` belongs to the program `--run` executes, verbatim: a `--threads=8`
    // there is the program's argument, not the compiler's.
    let program_args = match args.iter().position(|a| a == "--") {
        Some(i) => args.split_off(i),
        None => Vec::new(),
    };
    let mut threads = None;
    let mut err = None;
    args.retain(|a| {
        let Some(v) = a.strip_prefix("--threads=") else {
            return true;
        };
        match v.parse::<usize>() {
            Ok(n) if (1..=4096).contains(&n) && v.bytes().all(|b| b.is_ascii_digit()) => {
                threads = Some(n)
            }
            _ => {
                err.get_or_insert_with(|| {
                    format!("--threads expects an integer from 1 to 4096, got `{v}`")
                });
            }
        }
        false
    });
    args.extend(program_args);
    match err {
        Some(e) => Err(e),
        None => Ok(threads),
    }
}

/// Record one `-A/-W/-D <lint>` flag (`flag` is the two-character switch).
fn set_lint_level(
    levels: &mut wukong_driver::LintLevels,
    flag: &str,
    lint: &str,
) -> Result<(), String> {
    let level = match flag {
        "-A" => wukong_driver::LintLevel::Allow,
        "-W" => wukong_driver::LintLevel::Warn,
        _ => wukong_driver::LintLevel::Deny,
    };
    levels
        .set(lint, level)
        .map_err(|e| format!("`{flag} {lint}`: {e}"))
}

/// `<param-index>=<path>`, the value of `--train-init`, `--train-data` and `--train-save`.
fn indexed_path(flag: &str, v: &str) -> Result<(usize, PathBuf), String> {
    let bad = || format!("{flag} expects <param-index>=<path>, got `{v}`");
    let (index, path) = v.split_once('=').ok_or_else(bad)?;
    let index = index.trim().parse::<usize>().map_err(|_| bad())?;
    if path.is_empty() {
        return Err(bad());
    }
    Ok((index, PathBuf::from(path)))
}

/// `--repl` evaluates standard input on the interpreter or the native JIT. It honours the flags
/// that shape how an input compiles and runs — `--backend=interp|native`, `-O<n>`, `--fp`,
/// `--target-cpu`, `-A/-W/-D`, `--color`, `--error-format`, `--error-limit` and `--threads` — and
/// refuses every other one by name, rather than ignoring it.
fn check_repl_flags(opts: &Options, has_input: bool, emit_explicit: bool) -> Result<(), String> {
    let defaults = Options::default();
    let refused = [
        (has_input, "an input file or `-e`"),
        (opts.run, "`--run`"),
        (emit_explicit, "`--emit`"),
        (opts.output.is_some(), "`-o`"),
        (!opts.program_args.is_empty(), "`--` program arguments"),
        (opts.grad.train, "`--train`"),
        (
            opts.grad.of.is_some() || !opts.grad.wrt.is_empty(),
            "`--grad-of` / `--grad-wrt`",
        ),
        (opts.depfile.is_some(), "`--depfile`"),
        (
            opts.time_passes || opts.time_trace.is_some(),
            "`--time-passes` / `--time-trace`",
        ),
        (
            opts.passes.is_some() || opts.print_after.is_some(),
            "`--passes` / `--print-after`",
        ),
        (opts.verify_each, "`--verify-each`"),
        (opts.remarks_dispatch, "`--remarks`"),
        (
            opts.debug_info,
            "`-g` (it puts DWARF in an object or executable, and the REPL writes neither)",
        ),
        (
            matches!(
                opts.backend,
                wukong_driver::BackendKind::Gpu | wukong_driver::BackendKind::GpuLower
            ),
            "a GPU `--backend` (the REPL runs on interp or native)",
        ),
    ];
    if let Some((_, what)) = refused.iter().find(|(given, _)| *given) {
        return Err(format!("`--repl` does not take {what}"));
    }
    // The training knobs only mean something with `--train`, which is refused above; a
    // non-default value given alongside `--repl` is refused rather than dropped.
    let g = &opts.grad;
    let d = &defaults.grad;
    if g.train_steps != d.train_steps
        || g.train_lr != d.train_lr
        || g.train_weight_decay != d.train_weight_decay
        || g.train_opt != d.train_opt
        || g.train_seed != d.train_seed
        || !g.train_init.is_empty()
        || !g.train_data.is_empty()
        || !g.train_save.is_empty()
        || g.train_clip.is_some()
        || g.train_warmup != d.train_warmup
        || g.train_schedule != d.train_schedule
    {
        return Err("`--repl` does not take the `--train-*` options".into());
    }
    Ok(())
}

/// `--doctest` compiles and runs a file's doc examples. It honours the flags that shape how an
/// example compiles and runs — `--backend=interp|native`, `-O<n>`, `--fp`, `--target-cpu`,
/// `--color`, `--error-format`, `--error-limit` and `--threads` — and refuses every other one by
/// name, rather than ignoring it. (Examples are not linted, so `-A/-W/-D` are refused too.)
fn check_doctest_flags(
    opts: &Options,
    emit_explicit: bool,
    lints_given: bool,
) -> Result<(), String> {
    let defaults = Options::default();
    let g = &opts.grad;
    let d = &defaults.grad;
    let training = g.train
        || g.of.is_some()
        || !g.wrt.is_empty()
        || g.train_steps != d.train_steps
        || g.train_lr != d.train_lr
        || g.train_weight_decay != d.train_weight_decay
        || g.train_opt != d.train_opt
        || g.train_seed != d.train_seed
        || !g.train_init.is_empty()
        || !g.train_data.is_empty()
        || !g.train_save.is_empty()
        || g.train_clip.is_some()
        || g.train_warmup != d.train_warmup
        || g.train_schedule != d.train_schedule;
    let refused = [
        (opts.repl, "`--repl`"),
        (opts.run, "`--run` (every example is run already)"),
        (emit_explicit, "`--emit`"),
        (opts.output.is_some(), "`-o`"),
        (!opts.program_args.is_empty(), "`--` program arguments"),
        (
            training,
            "`--train`, `--grad-of`, `--grad-wrt` or the `--train-*` options",
        ),
        (opts.depfile.is_some(), "`--depfile`"),
        (
            opts.time_passes || opts.time_trace.is_some(),
            "`--time-passes` / `--time-trace`",
        ),
        (
            opts.passes.is_some() || opts.print_after.is_some(),
            "`--passes` / `--print-after`",
        ),
        (opts.verify_each, "`--verify-each`"),
        (opts.remarks_dispatch, "`--remarks`"),
        (opts.debug_info, "`-g`"),
        (lints_given, "`-A`/`-W`/`-D` (examples are not linted)"),
        (
            matches!(
                opts.backend,
                wukong_driver::BackendKind::Gpu | wukong_driver::BackendKind::GpuLower
            ),
            "a GPU `--backend` (examples run on interp or native)",
        ),
    ];
    match refused.iter().find(|(given, _)| *given) {
        Some((_, what)) => Err(format!("`--doctest` does not take {what}")),
        None => Ok(()),
    }
}

/// Parse CLI arguments. `Ok(None)` means a help / version / `--explain` message was already printed
/// and we should exit successfully without compiling. The two rejected flag *combinations* are
/// checked at the end, after the whole argv has been seen.
fn parse_args(args: &[String]) -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut input: Option<PathBuf> = None;
    let mut inline_source: Option<String> = None;
    let mut emit_explicit = false;
    let mut lints_given = false;
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
                    // `--explain` is terminal: we return right after, so the `<CODE>` operand is
                    // never re-read and `i` need not be advanced past it.
                    Some(code) if !code.starts_with('-') => print_explanation(code)?,
                    _ => list_all_codes(),
                }
                return Ok(None);
            }
            "--run" => opts.run = true,
            "--repl" => opts.repl = true,
            "--doctest" => opts.doctest = true,
            // Everything after `--` belongs to the program `--run` executes, not to the compiler.
            "--" => {
                opts.program_args = args[i + 1..].to_vec();
                break;
            }
            "-e" => {
                i += 1;
                let src = args
                    .get(i)
                    .ok_or("`-e` requires the program source as its argument")?;
                if inline_source.is_some() {
                    return Err("`-e` given twice: pass the whole program in one `-e`".into());
                }
                inline_source = Some(src.clone());
            }
            "--verify-each" => opts.verify_each = true,
            _ if arg.starts_with("--remarks=") => {
                let kind = &arg["--remarks=".len()..];
                match kind {
                    "dispatch" => opts.remarks_dispatch = true,
                    other => {
                        return Err(format!(
                            "unknown --remarks value `{other}` (expected dispatch)"
                        ))
                    }
                }
            }
            "--time-passes" => opts.time_passes = true,
            _ if arg.starts_with("--time-trace=") => {
                let path = &arg["--time-trace=".len()..];
                if path.is_empty() {
                    return Err("`--time-trace=` requires a file path".into());
                }
                opts.time_trace = Some(PathBuf::from(path));
            }
            _ if arg.starts_with("--depfile=") => {
                let path = &arg["--depfile=".len()..];
                if path.is_empty() {
                    return Err("`--depfile=` requires a file path".into());
                }
                opts.depfile = Some(PathBuf::from(path));
            }
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
            _ if arg.starts_with("--target-cpu=") => {
                let cpu = &arg["--target-cpu=".len()..];
                opts.target_cpu = wukong_driver::TargetCpu::parse(cpu).ok_or_else(|| {
                    format!(
                        "unknown --target-cpu value `{cpu}` (expected {})",
                        wukong_driver::TargetCpu::NAMES.join(", ")
                    )
                })?;
            }
            _ if arg.starts_with("--passes=") => {
                let list = &arg["--passes=".len()..];
                wukong_driver::passes::parse_spec(list).map_err(|e| format!("--passes: {e}"))?;
                opts.passes = Some(list.to_string());
            }
            _ if arg.starts_with("--print-after=") => {
                let pass = &arg["--print-after=".len()..];
                wukong_driver::passes::check_print_after(pass)
                    .map_err(|e| format!("--print-after: {e}"))?;
                opts.print_after = Some(pass.to_string());
            }
            "-g" => opts.debug_info = true,
            "-O0" => opts.opt_level = 0,
            "-O1" => opts.opt_level = 1,
            "-O2" => opts.opt_level = 2,
            "-O3" => opts.opt_level = 3,
            _ if arg.starts_with("--fp=") => {
                opts.strict_fp = match &arg["--fp=".len()..] {
                    "fast" => false,
                    "strict" => true,
                    other => {
                        return Err(format!(
                            "unknown --fp value `{other}` (expected fast or strict)"
                        ))
                    }
                };
            }
            "-o" => {
                i += 1;
                let path = args.get(i).ok_or("`-o` requires a path argument")?;
                opts.output = Some(PathBuf::from(path));
            }
            _ if arg.starts_with("--emit=") => {
                // One stage, or several from one compile (`--emit=mir,llvm-ir`): the last in
                // pipeline order is the stage the compile runs to, the rest are emitted on the way.
                let list = &arg["--emit=".len()..];
                let (last, also) = EmitStage::parse_list(list)?;
                opts.emit = last;
                opts.emit_also = also;
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
            _ if arg.starts_with("--train-init=") => {
                let v = &arg["--train-init=".len()..];
                opts.grad.train_init.push(indexed_path("--train-init", v)?);
            }
            _ if arg.starts_with("--train-data=") => {
                let v = &arg["--train-data=".len()..];
                opts.grad.train_data.push(indexed_path("--train-data", v)?);
            }
            _ if arg.starts_with("--train-save=") => {
                let v = &arg["--train-save=".len()..];
                opts.grad.train_save.push(indexed_path("--train-save", v)?);
            }
            _ if arg.starts_with("--train-clip=") => {
                let v = &arg["--train-clip=".len()..];
                let clip: f32 = v
                    .parse()
                    .map_err(|_| format!("--train-clip expects a float, got `{v}`"))?;
                if !clip.is_finite() || clip <= 0.0 {
                    return Err("--train-clip must be finite and positive".into());
                }
                opts.grad.train_clip = Some(clip);
            }
            _ if arg.starts_with("--train-warmup=") => {
                let v = &arg["--train-warmup=".len()..];
                opts.grad.train_warmup = v.parse().map_err(|_| {
                    format!("--train-warmup expects a non-negative integer, got `{v}`")
                })?;
            }
            _ if arg.starts_with("--train-schedule=") => {
                let v = &arg["--train-schedule=".len()..];
                opts.grad.train_schedule = match v {
                    "constant" => wukong_driver::TrainSchedule::Constant,
                    "cosine" => wukong_driver::TrainSchedule::Cosine,
                    other => {
                        return Err(format!(
                            "--train-schedule expects constant or cosine, got `{other}`"
                        ))
                    }
                };
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
            "-A" | "-W" | "-D" => {
                lints_given = true;
                i += 1;
                let lint = args.get(i).ok_or_else(|| {
                    format!("`{arg}` requires a lint name, e.g. `{arg} warnings`")
                })?;
                set_lint_level(&mut opts.lint_levels, arg, lint)?;
            }
            _ if arg.len() > 2 && ["-A", "-W", "-D"].iter().any(|f| arg.starts_with(f)) => {
                lints_given = true;
                set_lint_level(&mut opts.lint_levels, &arg[..2], &arg[2..])?;
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

    // `--doctest` runs a file's doc examples: no stage, no artifact, no run of the file itself.
    if opts.doctest {
        check_doctest_flags(&opts, emit_explicit, lints_given)?;
    }

    // `--repl` reads its program from standard input, an input at a time: no input file, no stage
    // and no artifact. What it does honour is checked in one place.
    if opts.repl {
        check_repl_flags(
            &opts,
            input.is_some() || inline_source.is_some(),
            emit_explicit,
        )?;
        opts.input = PathBuf::from("<repl>");
        return Ok(Some(opts));
    }

    match (input, inline_source) {
        (Some(_), Some(_)) => {
            return Err(
                "give the program either as an input file (or `-` for standard input) or as \
                 `-e <source>`, not both"
                    .to_string(),
            )
        }
        (Some(path), None) => opts.input = path,
        (None, Some(src)) => {
            opts.input = PathBuf::from("<command-line>");
            opts.source = Some(src);
        }
        (None, None) => return Err("no input file given".to_string()),
    }

    // Program arguments are for the program `--run` executes; anywhere else they would be dropped
    // in silence.
    if !opts.program_args.is_empty() && !opts.run {
        return Err(
            "arguments after `--` are passed to the program by `--run`, which was not given"
                .to_string(),
        );
    }

    // A requested `--emit=<stage>` and `--run` cannot both be honoured: the driver dispatches
    // exactly ONE of them and silently drops the other, while still exiting 0. Which one wins
    // depends on where the stage sits relative to the run block, so the discarded work is not even
    // consistent: `--run --emit=obj|exe|mir|llvm-ir` runs the program and writes/prints no artifact,
    // and `--run --emit=tokens|ast|mir-high|grad` prints the artifact and never runs. Reject the
    // combination rather than report success for work that was not done. (`emit_explicit` exists
    // precisely for this check; it used to be discarded at the end of `parse_args`.)
    // `--train` either runs now or, with `--emit=exe|obj` (or `-o`, whose default stage is `exe`),
    // is built into an artifact that runs the same loop. No other stage is a training artifact.
    if opts.grad.train {
        if opts.multi_emit() {
            return Err(
                "`--train` builds one artifact (`--emit=exe` or `--emit=obj`), not an \
                 `--emit=<stage>,<stage>` list"
                    .into(),
            );
        }
        if emit_explicit && !matches!(opts.emit, EmitStage::Obj | EmitStage::Exe) {
            return Err(
                "`--train` cannot be combined with `--emit=<stage>` other than `exe` or `obj`: \
                 training executes immediately, or is built into an executable or object with \
                 `--emit=exe` / `--emit=obj`"
                    .into(),
            );
        }
        opts.grad.train_artifact = emit_explicit || opts.output.is_some();
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
    // at the *next* command with a missing-file error naming the wrong culprit. A multi-artifact
    // `--emit=a,b` is the exception: there `-o` names the stem every artifact is written under
    // (`<stem>.<ext>`), the text ones included.
    if opts.output.is_some() {
        if opts.run {
            return Err(
                "`-o` has no meaning with `--run`: running the program writes no artifact file"
                    .to_string(),
            );
        }
        if !opts.multi_emit() && !matches!(opts.emit, EmitStage::Obj | EmitStage::Exe) {
            return Err(
                "`-o` is only supported with `--emit=obj` and `--emit=exe`: every other stage \
                 prints its artifact on stdout (redirect it instead)"
                    .to_string(),
            );
        }
    }

    // Without `-o` a multi-artifact compile prints its text artifacts on stdout, one after another;
    // an object or an executable has no place in that stream.
    if opts.multi_emit() && opts.emit.is_binary() && opts.output.is_none() {
        return Err(format!(
            "`--emit={}` with `{}` writes every artifact to a file: give `-o <stem>` (each \
             artifact is written to `<stem>.<ext>`)",
            opts.emit_stages()
                .map(EmitStage::name)
                .collect::<Vec<_>>()
                .join(","),
            opts.emit.name()
        ));
    }

    // `--passes` / `--print-after` act on the optimizer, so a stage that stops before it would
    // silently ignore them. A gradient compile is refused because the pass-list replica has no
    // autodiff roots: the forward autodiff takes is always the standard pipeline's.
    if opts.passes.is_some() || opts.print_after.is_some() {
        let flag = if opts.passes.is_some() {
            "--passes"
        } else {
            "--print-after"
        };
        let early = match opts.emit {
            EmitStage::Tokens => Some("tokens"),
            EmitStage::Ast => Some("ast"),
            EmitStage::Doc => Some("doc"),
            EmitStage::Check => Some("check"),
            EmitStage::MirHigh => Some("mir-high"),
            _ => None,
        };
        if let Some(stage) = early {
            return Err(format!(
                "`{flag}` has no effect with `--emit={stage}`, which stops before the optimizer runs"
            ));
        }
        if opts.emit == EmitStage::Grad
            || opts.grad.train
            || (opts.emit == EmitStage::Obj && opts.grad.of.is_some())
        {
            return Err(format!(
                "`{flag}` cannot be combined with a gradient compile (`--emit=grad`, `--train`, \
                 `--emit=obj --grad-of`): autodiff always differentiates the standard pipeline's \
                 forward"
            ));
        }
    }

    // A `.mir` input is a typed MIR dump (`--emit=mir-text`): the driver reads it back, optimizes
    // it and prints MIR. It has no front end to lint or link against, so every other stage — and
    // running it — is out of scope rather than silently something else.
    if opts.source.is_none() && opts.input.extension().is_some_and(|e| e == "mir") {
        if opts.run {
            return Err(
                "a `.mir` input cannot be run: it is read only by `--emit=mir` and \
                 `--emit=mir-text`"
                    .to_string(),
            );
        }
        if !emit_explicit
            || !opts
                .emit_stages()
                .all(|s| matches!(s, EmitStage::Mir | EmitStage::MirText))
        {
            return Err(
                "a `.mir` input is read only by `--emit=mir` and `--emit=mir-text`".to_string(),
            );
        }
        if opts.depfile.is_some() {
            return Err(
                "`--depfile` needs an import graph, which a `.mir` input has none of".into(),
            );
        }
    }

    // The dependency set is the import graph, and these three stages stop before imports are
    // loaded: a depfile written there would silently omit every imported file.
    if opts.depfile.is_some()
        && matches!(
            opts.emit,
            EmitStage::Tokens | EmitStage::Ast | EmitStage::Doc
        )
    {
        return Err(
            "`--depfile` needs the import graph, which `--emit=tokens`, `--emit=ast` and \
             `--emit=doc` stop before loading"
                .to_string(),
        );
    }

    Ok(Some(opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `wukongc fmt` is a subcommand dispatched before flag parsing; USAGE (the only user-facing
    /// flag doc) must name it, and its own help must document every option it accepts.
    #[test]
    fn usage_documents_the_fmt_subcommand() {
        assert!(
            USAGE.contains("wukongc fmt [--check] [--stdout] <path|->..."),
            "{USAGE}"
        );
        for opt in ["--check", "--stdout", "-h, --help", "./fmt"] {
            assert!(fmt_cmd::FMT_USAGE.contains(opt), "fmt usage lacks `{opt}`");
        }
    }

    /// `--emit=doc` is a root-file stage like `ast` (it combines with `check`, takes no depfile and
    /// no pass list), and `--doctest` is a mode of its own that refuses what it cannot honour; both
    /// are in USAGE.
    #[test]
    fn doc_stage_and_doctest_flags_parse() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = parse(&["--emit=doc", "in.wk"]).unwrap().unwrap();
        assert_eq!(o.emit, EmitStage::Doc);
        let o = parse(&["--emit=check,doc", "in.wk"]).unwrap().unwrap();
        assert_eq!(
            (o.emit, o.emit_also.as_slice()),
            (EmitStage::Check, &[EmitStage::Doc][..])
        );
        assert!(parse(&["--emit=doc", "--depfile=d.d", "in.wk"]).is_err());
        assert!(parse(&["--emit=doc", "--passes=dce", "in.wk"]).is_err());
        assert!(parse(&["--emit=doc", "-o", "x.md", "in.wk"]).is_err());

        let o = parse(&["--doctest", "--backend=native", "-O2", "in.wk"])
            .unwrap()
            .unwrap();
        assert!(o.doctest && o.opt_level == 2 && !o.run);
        for bad in [
            &["--doctest", "--run", "in.wk"][..],
            &["--doctest", "--emit=mir", "in.wk"][..],
            &["--doctest", "-o", "x", "in.wk"][..],
            &["--doctest", "--backend=gpu-native", "in.wk"][..],
            &["--doctest", "-A", "dead_code", "in.wk"][..],
            &["--doctest", "--train-steps=3", "in.wk"][..],
            &["--doctest", "--repl"][..],
            &["--doctest"][..],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
        for text in ["tokens, ast, doc, check", "--doctest", ".md (doc)"] {
            assert!(USAGE.contains(text), "USAGE lacks {text:?}");
        }
    }

    /// `-A/-W/-D <lint>` (and the glued `-Dwarnings`) accept a lint name, its dashed spelling, its
    /// code or `warnings`; an unknown lint or a missing name is a usage error, not a silent no-op.
    /// USAGE lists every lint, so the help cannot fall behind the lint table.
    #[test]
    fn lint_level_flags_parse_and_reject_unknown_lints() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        for ok in [
            &["-D", "warnings", "in.wk"][..],
            &["-Dwarnings", "-A", "dead_code", "-W", "unused-mut", "in.wk"][..],
            &["-A", "W0001", "in.wk"][..],
        ] {
            assert!(parse(ok).unwrap().is_some(), "{ok:?}");
        }
        let err = parse(&["-D", "unused_varaibles", "in.wk"]).unwrap_err();
        assert!(err.contains("unknown lint"), "{err}");
        assert!(
            parse(&["in.wk", "-D"]).is_err(),
            "a level flag needs a lint"
        );
        for lint in wukong_driver::LINTS {
            assert!(USAGE.contains(lint.name), "USAGE must list {}", lint.name);
            assert!(USAGE.contains(lint.code), "USAGE must list {}", lint.code);
        }
    }

    #[test]
    fn compile_time_reporting_flags_parse() {
        let parse = |extra: &[&str]| {
            let mut args: Vec<String> = extra.iter().map(|s| s.to_string()).collect();
            args.push("input.wk".to_owned());
            parse_args(&args)
        };
        let plain = parse(&[]).unwrap().unwrap();
        assert!(!plain.time_passes && plain.time_trace.is_none() && plain.depfile.is_none());
        let opts = parse(&[
            "--run",
            "--time-passes",
            "--time-trace=t.json",
            "--depfile=p.d",
        ])
        .unwrap()
        .unwrap();
        assert!(opts.time_passes && opts.run);
        assert_eq!(opts.time_trace, Some(PathBuf::from("t.json")));
        assert_eq!(opts.depfile, Some(PathBuf::from("p.d")));
        assert!(parse(&["--time-trace="]).is_err());
        assert!(parse(&["--depfile="]).is_err());
        for early in ["--emit=tokens", "--emit=ast"] {
            assert!(parse(&[early, "--depfile=p.d"]).is_err(), "{early}");
        }
        assert!(parse(&["--emit=mir-high", "--depfile=p.d"])
            .unwrap()
            .is_some());
        for flag in ["--time-passes", "--time-trace=<f>", "--depfile=<f>"] {
            assert!(USAGE.contains(flag), "USAGE must document {flag}");
        }
    }

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

    /// The training data, checkpoint and update-policy flags: `<i>=<path>` lists accumulate in order,
    /// a malformed value is a usage error naming the flag, and every flag is in USAGE.
    #[test]
    fn training_data_and_schedule_flags_parse() {
        let parse = |extra: &[&str]| {
            let mut args: Vec<String> = ["--train"].iter().map(|s| s.to_string()).collect();
            args.extend(extra.iter().map(|s| s.to_string()));
            args.push("input.wk".to_owned());
            parse_args(&args)
        };
        let plain = parse(&[]).unwrap().unwrap();
        assert!(plain.grad.train_init.is_empty() && plain.grad.train_clip.is_none());
        assert_eq!(plain.grad.train_warmup, 0);
        assert_eq!(
            plain.grad.train_schedule,
            wukong_driver::TrainSchedule::Constant
        );
        let o = parse(&[
            "--train-init=0=w.bin",
            "--train-data=1=batches.bin",
            "--train-data=2=labels.bin",
            "--train-save=0=trained.bin",
            "--train-clip=1.5",
            "--train-warmup=10",
            "--train-schedule=cosine",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(o.grad.train_init, vec![(0, PathBuf::from("w.bin"))]);
        assert_eq!(
            o.grad.train_data,
            vec![
                (1, PathBuf::from("batches.bin")),
                (2, PathBuf::from("labels.bin"))
            ]
        );
        assert_eq!(o.grad.train_save, vec![(0, PathBuf::from("trained.bin"))]);
        assert_eq!(o.grad.train_clip, Some(1.5));
        assert_eq!(o.grad.train_warmup, 10);
        assert_eq!(o.grad.train_schedule, wukong_driver::TrainSchedule::Cosine);
        for (bad, flag) in [
            ("--train-init=w.bin", "--train-init"),
            ("--train-data=x=d.bin", "--train-data"),
            ("--train-save=0=", "--train-save"),
            ("--train-clip=0", "--train-clip"),
            ("--train-clip=nan", "--train-clip"),
            ("--train-warmup=-1", "--train-warmup"),
            ("--train-schedule=linear", "--train-schedule"),
        ] {
            let err = parse(&[bad]).unwrap_err();
            assert!(err.contains(flag), "{bad}: {err}");
        }
        for flag in [
            "--train-init=<i>=<path>",
            "--train-data=<i>=<path>",
            "--train-save=<i>=<path>",
            "--train-clip=<f>",
            "--train-warmup=<n>",
            "--train-schedule=<s>",
        ] {
            assert!(USAGE.contains(flag), "USAGE must document {flag}");
        }
    }

    /// `--train` runs at once unless an artifact is asked for: `--emit=exe|obj`, or `-o` (whose
    /// stage is the default `exe`). Every other stage is refused, as are `--run` with a stage and
    /// `-o` with `--run`.
    #[test]
    fn train_artifact_stages_parse() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let now = parse(&["--train", "m.wk"]).unwrap().unwrap();
        assert!(now.grad.train && !now.grad.train_artifact);
        for (args, stage) in [
            (&["--train", "--emit=exe", "m.wk"][..], EmitStage::Exe),
            (
                &["--train", "--emit=obj", "-o", "m.o", "m.wk"][..],
                EmitStage::Obj,
            ),
            (&["--train", "-o", "m.exe", "m.wk"][..], EmitStage::Exe),
        ] {
            let o = parse(args).unwrap().unwrap();
            assert!(o.grad.train && o.grad.train_artifact, "{args:?}");
            assert_eq!(o.emit, stage, "{args:?}");
        }
        for stage in [
            "tokens", "ast", "check", "mir-high", "mir", "grad", "llvm-ir",
        ] {
            let err = parse(&["--train", &format!("--emit={stage}"), "m.wk"]).unwrap_err();
            assert!(
                err.contains("training executes immediately"),
                "{stage}: {err}"
            );
        }
        assert!(parse(&["--train", "--run", "--emit=exe", "m.wk"]).is_err());
        assert!(parse(&["--train", "--run", "-o", "m.exe", "m.wk"]).is_err());
        assert!(USAGE.contains("--emit=exe|obj"));
    }

    #[test]
    fn fp_mode_flag_parses_and_defaults_to_fast() {
        let parse = |flag: &str| parse_args(&[flag.to_owned(), "input.wk".to_owned()]);
        assert!(parse("--fp=strict").unwrap().unwrap().strict_fp);
        assert!(!parse("--fp=fast").unwrap().unwrap().strict_fp);
        assert!(
            !parse_args(&["input.wk".to_owned()])
                .unwrap()
                .unwrap()
                .strict_fp,
            "fast is the default"
        );
        let err = parse("--fp=precise").unwrap_err();
        assert!(err.contains("--fp"), "{err}");
        assert!(USAGE.contains("--fp=<mode>"));
    }

    #[test]
    fn threads_flag_is_validated_and_removed() {
        let mut args = ["--run", "--threads=8", "input.wk"]
            .map(str::to_owned)
            .to_vec();
        assert_eq!(take_threads(&mut args), Ok(Some(8)));
        assert_eq!(args, ["--run", "input.wk"]);
        // The remaining argv parses as if the flag had never been there.
        assert!(parse_args(&args).unwrap().unwrap().run);
        let mut none = ["input.wk"].map(str::to_owned).to_vec();
        assert_eq!(take_threads(&mut none), Ok(None));
        // After `--` the argument is the program's, kept verbatim and in place.
        let mut passed = ["--run", "--threads=2", "p.wk", "--", "--threads=8", "x"]
            .map(str::to_owned)
            .to_vec();
        assert_eq!(take_threads(&mut passed), Ok(Some(2)));
        assert_eq!(passed, ["--run", "p.wk", "--", "--threads=8", "x"]);
        let o = parse_args(&passed).unwrap().unwrap();
        assert_eq!(o.program_args, ["--threads=8", "x"]);
        for bad in [
            "--threads=0",
            "--threads=",
            "--threads=-2",
            "--threads=+2",
            "--threads=4097",
        ] {
            let mut args = [bad, "input.wk"].map(str::to_owned).to_vec();
            assert!(take_threads(&mut args).is_err(), "{bad}");
        }
        assert!(USAGE.contains("--threads=<n>"));
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

    /// Everything after `--` is the program's, verbatim — flags included — and needs `--run`;
    /// `-e` replaces the input file, and `-` stays the input name the driver reads stdin for.
    #[test]
    fn program_arguments_inline_source_and_stdin_parse() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = parse(&["--run", "p.wk", "--", "a", "-O2", "--", "42"])
            .unwrap()
            .unwrap();
        assert_eq!(o.input, PathBuf::from("p.wk"));
        assert_eq!(o.program_args, ["a", "-O2", "--", "42"]);
        assert!(
            o.run && o.opt_level == 0,
            "flags after `--` are not the compiler's"
        );
        assert!(parse(&["--run", "p.wk", "--"])
            .unwrap()
            .unwrap()
            .program_args
            .is_empty());
        let err = parse(&["p.wk", "--", "a"]).unwrap_err();
        assert!(err.contains("--run"), "{err}");
        let o = parse(&["--run", "-e", "fn main() -> i32 { return 0; }"])
            .unwrap()
            .unwrap();
        assert_eq!(o.source.as_deref(), Some("fn main() -> i32 { return 0; }"));
        assert_eq!(o.input, PathBuf::from("<command-line>"));
        assert!(parse(&["-e"]).is_err());
        assert!(parse(&["p.wk", "-e", "x"]).is_err());
        assert!(parse(&["-e", "x", "-e", "y"]).is_err());
        let o = parse(&["--run", "-"]).unwrap().unwrap();
        assert_eq!((o.input, o.source), (PathBuf::from("-"), None));
        for flag in ["-e <source>", "-- <args>", "standard input"] {
            assert!(USAGE.contains(flag), "USAGE must document {flag}");
        }
    }

    #[test]
    fn explain_with_no_code_is_accepted() {
        // A bare `--explain` lists all codes and exits successfully (Ok(None)).
        assert!(parse_args(&["--explain".to_owned()]).unwrap().is_none());
    }

    /// `--remarks=dispatch` is opt-in, documented, and the only `--remarks` kind: an unknown kind
    /// is a usage error rather than a flag that silently does nothing.
    #[test]
    fn remarks_dispatch_is_opt_in_and_closed() {
        let on = parse_args(&["--remarks=dispatch", "--run", "input.wk"].map(str::to_owned))
            .unwrap()
            .unwrap();
        assert!(on.remarks_dispatch);
        assert!(
            !parse_args(&["--run".to_owned(), "input.wk".to_owned()])
                .unwrap()
                .unwrap()
                .remarks_dispatch
        );
        assert!(parse_args(&["--remarks=all", "input.wk"].map(str::to_owned)).is_err());
        assert!(USAGE.contains("--remarks=dispatch"));
    }

    /// `--passes` and `--print-after` take only names the optimizer knows, an unknown one is a
    /// usage error that lists them all, and both are refused where no optimizer runs or where
    /// autodiff needs the standard pipeline. USAGE lists every name.
    #[test]
    fn passes_and_print_after_parse_and_reject_misuse() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = parse(&["--passes=mem2reg,simplify,dce", "--emit=mir", "p.wk"])
            .unwrap()
            .unwrap();
        assert_eq!(o.passes.as_deref(), Some("mem2reg,simplify,dce"));
        assert_eq!(o.print_after, None);
        let o = parse(&["-O2", "--print-after=licm", "--run", "p.wk"])
            .unwrap()
            .unwrap();
        assert_eq!(o.print_after.as_deref(), Some("licm"));
        assert_eq!(o.passes, None);
        assert_eq!(
            parse(&["--passes=", "p.wk"])
                .unwrap()
                .unwrap()
                .passes
                .as_deref(),
            Some("")
        );
        assert!(parse(&["--print-after=all", "--passes=inline,slp", "p.wk"]).is_ok());
        let none = parse(&["p.wk"]).unwrap().unwrap();
        assert!(none.passes.is_none() && none.print_after.is_none());
        let err = parse(&["--passes=mem2reg,bogus", "p.wk"]).unwrap_err();
        assert!(err.contains("bogus") && err.contains("vectorize"), "{err}");
        let err = parse(&["--print-after=bogus", "p.wk"]).unwrap_err();
        assert!(
            err.contains("bogus") && err.contains("form-switches"),
            "{err}"
        );
        for flag in ["--passes=dce", "--print-after=dce"] {
            for stage in ["tokens", "ast", "check", "mir-high", "grad"] {
                let emit = format!("--emit={stage}");
                assert!(parse(&[flag, &emit, "p.wk"]).is_err(), "{flag} {emit}");
            }
            assert!(parse(&[flag, "--train", "p.wk"]).is_err(), "{flag} --train");
            assert!(
                parse(&[flag, "--emit=obj", "--grad-of=loss", "-o", "l.o", "p.wk"]).is_err(),
                "{flag} --grad-of"
            );
            for ok in ["--emit=mir", "--emit=llvm-ir", "--emit=obj", "--run"] {
                assert!(parse(&[flag, ok, "p.wk"]).is_ok(), "{flag} {ok}");
            }
        }
        for name in wukong_driver::passes::FUNCTION_PASSES
            .iter()
            .chain(wukong_driver::passes::PROGRAM_STEPS)
        {
            assert!(USAGE.contains(name), "USAGE must list {name}");
        }
        assert!(USAGE.contains("--passes=<list>") && USAGE.contains("--print-after=<p>"));
    }

    /// `--emit=mir-text` is a stage, and a `.mir` input admits exactly the two MIR stages: no
    /// other stage, no default `exe`, no `--run`.
    #[test]
    fn mir_text_stage_and_mir_input_parse() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = parse(&["--emit=mir-text", "-O2", "p.wk"]).unwrap().unwrap();
        assert_eq!(o.emit, EmitStage::MirText);
        for stage in ["mir", "mir-text"] {
            let emit = format!("--emit={stage}");
            let o = parse(&[&emit, "-O2", "--passes=dce", "--print-after=dce", "x.mir"])
                .unwrap()
                .unwrap();
            assert_eq!(o.input, PathBuf::from("x.mir"));
        }
        for bad in [
            &["x.mir"][..],
            &["--run", "x.mir"][..],
            &["--emit=exe", "x.mir"][..],
            &["--emit=obj", "-o", "x.o", "x.mir"][..],
            &["--emit=llvm-ir", "x.mir"][..],
            &["--emit=mir-high", "x.mir"][..],
            &["--emit=ast", "x.mir"][..],
            &["--emit=mir", "--depfile=x.d", "x.mir"][..],
        ] {
            let err = parse(bad).unwrap_err();
            assert!(err.contains(".mir"), "{bad:?}: {err}");
        }
        assert!(parse(&["--run", "--emit=mir-text", "p.wk"]).is_err());
        assert!(parse(&["--emit=mir-text", "-o", "p.txt", "p.wk"]).is_err());
        assert!(USAGE.contains("mir-text") && USAGE.contains("f.mir"));
    }

    /// `--emit=a,b`: one compile whose stages sort into pipeline order (the last is the stage the
    /// compile runs to, the others are emitted on the way), with `-o` naming the stem of every
    /// artifact. A list one compile cannot honour, and a flag a list contradicts, is a usage error.
    #[test]
    fn multi_artifact_emit_lists_parse_and_reject_contradictions() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = parse(&["--emit=llvm-ir,mir,tokens", "p.wk"])
            .unwrap()
            .unwrap();
        assert_eq!(o.emit, EmitStage::LlvmIr);
        assert_eq!(o.emit_also, [EmitStage::Tokens, EmitStage::Mir]);
        assert!(o.multi_emit() && o.output.is_none());
        let o = parse(&["--emit=exe,mir-text", "-o", "out/k", "p.wk"])
            .unwrap()
            .unwrap();
        assert_eq!(
            (o.emit, &o.emit_also[..]),
            (EmitStage::Exe, &[EmitStage::MirText][..])
        );
        assert_eq!(
            o.artifact_path(EmitStage::MirText),
            Some(PathBuf::from("out/k.mir"))
        );
        let o = parse(&["--emit=mir,llvm-ir,mir-high", "-o", "out/k.v2", "p.wk"])
            .unwrap()
            .unwrap();
        for (stage, file) in [
            (EmitStage::MirHigh, "out/k.mir-high"),
            (EmitStage::Mir, "out/k.mir-low"),
            (EmitStage::LlvmIr, "out/k.ll"),
        ] {
            assert_eq!(o.artifact_path(stage), Some(PathBuf::from(file)));
        }
        // A single stage is exactly what it was: no list, and `-o` only for obj/exe.
        let single = parse(&["--emit=mir", "p.wk"]).unwrap().unwrap();
        assert!(!single.multi_emit() && single.artifact_path(EmitStage::Mir).is_none());
        assert!(parse(&["--emit=mir", "-o", "k", "p.wk"]).is_err());
        for (bad, says) in [
            (&["--emit=mir,", "p.wk"][..], "empty stage"),
            (&["--emit=mir,bogus", "p.wk"][..], "bogus"),
            (&["--emit=mir,mir-low", "p.wk"][..], "twice"),
            (&["--emit=check,mir", "p.wk"][..], "combines only"),
            (&["--emit=grad,llvm-ir", "p.wk"][..], "combines only"),
            (
                &["--emit=obj,exe", "-o", "k", "p.wk"][..],
                "two different builds",
            ),
            (&["--emit=mir,obj", "p.wk"][..], "-o <stem>"),
            (&["--emit=mir,exe", "p.wk"][..], "-o <stem>"),
            (
                &["--train", "--emit=mir,exe", "-o", "k", "p.wk"][..],
                "one artifact",
            ),
            (&["--run", "--emit=mir,llvm-ir", "p.wk"][..], "--run"),
            (
                &["--emit=tokens,ast", "--depfile=p.d", "p.wk"][..],
                "--depfile",
            ),
            (&["--emit=mir-high,mir", "x.mir"][..], ".mir"),
            (
                &["--passes=dce", "--emit=tokens,mir-high", "p.wk"][..],
                "--passes",
            ),
            (
                &[
                    "--passes=dce",
                    "--emit=mir,obj",
                    "--grad-of=loss",
                    "-o",
                    "k",
                    "p.wk",
                ][..],
                "gradient",
            ),
        ] {
            let err = parse(bad).unwrap_err();
            assert!(err.contains(says), "{bad:?}: {err}");
        }
        for ok in [
            &["--emit=tokens,check", "p.wk"][..],
            &["--emit=ast,grad", "p.wk"][..],
            &["--emit=mir,mir-text", "x.mir"][..],
            &["--emit=mir,mir-text", "-o", "k", "x.mir"][..],
            &["--emit=tokens,mir", "--depfile=p.d", "p.wk"][..],
            &["--emit=mir-high,mir", "--passes=dce", "p.wk"][..],
            &["--emit=mir,obj", "--grad-of=loss", "-o", "k", "p.wk"][..],
        ] {
            assert!(parse(ok).is_ok(), "{ok:?}: {:?}", parse(ok).err());
        }
        for doc in [
            "--emit=<stage>,<stage>...",
            "==> <stage> <==",
            ".mir-low",
            "<stem>",
        ] {
            assert!(USAGE.contains(doc), "USAGE must document {doc}");
        }
    }

    /// `--repl` takes the flags that shape how an input compiles and runs, and refuses every other
    /// one by name; the no-argument invocation is still the "no input file" usage error.
    #[test]
    fn repl_takes_the_run_shaping_flags_and_refuses_the_rest() {
        let parse = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = parse(&["--repl"]).unwrap().unwrap();
        assert!(o.repl && !o.run && o.backend == wukong_driver::BackendKind::Interp);
        assert_eq!(o.input, PathBuf::from("<repl>"));
        let o = parse(&[
            "--backend=native",
            "-O2",
            "--repl",
            "--fp=strict",
            "--target-cpu=x86-64-v2",
            "-D",
            "warnings",
            "--color=never",
            "--error-format=json",
            "--error-limit=3",
        ])
        .unwrap()
        .unwrap();
        assert!(o.repl && o.opt_level == 2 && o.strict_fp);
        assert_eq!(o.backend, wukong_driver::BackendKind::Native);
        assert_eq!(o.error_format, wukong_driver::ErrorFormat::Json);
        assert_eq!(o.error_limit, Some(3));
        for bad in [
            &["--repl", "p.wk"][..],
            &["--repl", "-"][..],
            &["--repl", "-e", "1"][..],
            &["--repl", "--run"][..],
            &["--repl", "--emit=mir"][..],
            &["--repl", "-o", "k"][..],
            &["--repl", "--", "a"][..],
            &["--repl", "--train"][..],
            &["--repl", "--train-steps=3"][..],
            &["--repl", "--grad-of=f"][..],
            &["--repl", "--backend=gpu"][..],
            &["--repl", "--depfile=k.d"][..],
            &["--repl", "--time-passes"][..],
            &["--repl", "--passes=dce"][..],
            &["--repl", "--verify-each"][..],
            &["--repl", "--remarks=dispatch"][..],
            &["--repl", "-g"][..],
        ] {
            let err = parse(bad).unwrap_err();
            assert!(err.contains("--repl"), "{bad:?}: {err}");
        }
        assert_eq!(parse(&[]).unwrap_err(), "no input file given");
        for doc in ["--repl", ":type <expr>", ":reset", ":items"] {
            assert!(USAGE.contains(doc), "USAGE must document {doc}");
        }
    }

    #[test]
    fn target_cpu_flag_parses_every_level_and_rejects_others() {
        assert_eq!(
            parse_args(&["input.wk".to_owned()])
                .unwrap()
                .unwrap()
                .target_cpu,
            wukong_driver::TargetCpu::Native
        );
        for name in wukong_driver::TargetCpu::NAMES {
            let args = [
                format!("--target-cpu={name}"),
                "--emit=obj".into(),
                "in.wk".into(),
            ];
            let opts = parse_args(&args).unwrap().unwrap();
            assert_eq!(opts.target_cpu.name(), *name);
            assert!(USAGE.contains(name), "USAGE must list {name}");
        }
        let err =
            parse_args(&["--target-cpu=pentium4".to_owned(), "in.wk".to_owned()]).unwrap_err();
        assert!(
            err.contains("pentium4") && err.contains("x86-64-v3"),
            "{err}"
        );
    }

    /// `-g` is off by default and accepted with every stage (only obj/exe carry DWARF); USAGE
    /// documents it and the JIT perf map's environment variable.
    #[test]
    fn debug_info_flag_parses_and_is_documented() {
        let parse = |a: &[&str]| {
            parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>())
                .unwrap()
                .unwrap()
        };
        assert!(!parse(&["in.wk"]).debug_info);
        for stage in ["--emit=obj", "--emit=exe", "--emit=mir", "--run"] {
            assert!(parse(&["-g", stage, "in.wk"]).debug_info, "{stage}");
        }
        let o = parse(&["--emit=obj", "-o", "x.o", "-O2", "-g", "in.wk"]);
        assert!(o.debug_info && o.opt_level == 2);
        assert!(USAGE.contains("    -g  "), "USAGE must document -g");
        assert!(
            USAGE.contains("WUKONG_PERF_MAP=1"),
            "USAGE must name the perf map switch"
        );
    }
}
