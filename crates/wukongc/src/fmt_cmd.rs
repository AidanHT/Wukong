//! `wukongc fmt` — format Wukong source files with `wukong_fmt`.
//!
//! The formatter refuses rather than change a program's meaning: a file it cannot reprint with an
//! identical AST, the same comments and the same identifier, literal and keyword tokens is left
//! untouched and reported (exit 2). Files are rewritten only when their formatted text differs,
//! through a `<file>.fmt.tmp` written next to the file and renamed over it.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use wukong_driver::exit;
use wukong_fmt::{format_source, FmtError, FmtOptions};
use wukong_span::SourceMap;

pub(crate) const FMT_USAGE: &str = "\
wukongc fmt — format Wukong source files

USAGE:
    wukongc fmt [--check] [--stdout] <path|->...

    Each <path> is a .wk file, or a directory searched recursively for *.wk files (in sorted
    order). `-` reads one program from standard input and writes it to standard output. Files
    are rewritten in place only when their formatted text differs. (`fmt` is a subcommand only
    as the first argument: to compile a file literally named `fmt`, pass it as `./fmt`.)

OPTIONS:
    --check     Write nothing; print each file that would change, and exit 1 if any would
    --stdout    Print the formatted text instead of rewriting the file (exactly one input)
    -h, --help  Print this help

The formatter never changes what a program means: every result is re-parsed and must give the
identical AST, keep every comment, and keep every identifier, literal and keyword. A file it cannot
prove that for is left unchanged and reported.

EXIT CODES:
    0  every file is formatted (or was rewritten)
    1  a syntax error, or --check found a file that would change
    2  a usage error, or a file the formatter cannot format without changing its meaning
    3  a file or directory could not be read or written
";

/// Run `wukongc fmt <args>`. The printer and the self-check recurse with the program's nesting, so
/// this runs on a worker thread with the same large stack the compiler uses.
pub(crate) fn run(args: &[String]) -> ExitCode {
    let args = args.to_vec();
    let code = std::thread::Builder::new()
        .name("wukongc-fmt".to_string())
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_inner(&args))
        .expect("failed to spawn formatter thread")
        .join()
        .unwrap_or(exit::COMPILE_ERROR);
    ExitCode::from(code as u8)
}

fn usage_error(msg: &str) -> i32 {
    eprintln!("error: {msg}\n");
    eprint!("{FMT_USAGE}");
    exit::UNIMPLEMENTED
}

struct Mode {
    check: bool,
    stdout: bool,
    color: bool,
}

fn run_inner(args: &[String]) -> i32 {
    let mut check = false;
    let mut stdout = false;
    let mut inputs: Vec<&str> = Vec::new();
    let mut only_paths = false;
    for a in args {
        match a.as_str() {
            "--" if !only_paths => only_paths = true,
            "--check" if !only_paths => check = true,
            "--stdout" if !only_paths => stdout = true,
            "-h" | "--help" if !only_paths => {
                print!("{FMT_USAGE}");
                return exit::OK;
            }
            s if !only_paths && s.starts_with('-') && s != "-" => {
                return usage_error(&format!("unknown `wukongc fmt` option `{s}`"));
            }
            s => inputs.push(s),
        }
    }
    if inputs.is_empty() {
        return usage_error("`wukongc fmt` needs a file, a directory or `-`");
    }
    if check && stdout {
        return usage_error("`--check` and `--stdout` cannot be combined");
    }
    let stdin = inputs.contains(&"-");
    if stdin && inputs.len() != 1 {
        return usage_error("`-` (standard input) must be the only input");
    }
    if stdout && inputs.len() != 1 {
        return usage_error("`--stdout` takes exactly one input");
    }
    let mode = Mode {
        check,
        stdout: stdout || stdin,
        color: crate::resolve_color(crate::ColorChoice::Auto),
    };
    if stdin {
        let mut src = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut src) {
            eprintln!("wukongc fmt: cannot read standard input: {e}");
            return exit::IO_ERROR;
        }
        return format_one("<stdin>", None, &src, &mode);
    }

    let mut files: Vec<PathBuf> = Vec::new();
    for i in &inputs {
        let p = Path::new(i);
        match std::fs::metadata(p) {
            Ok(m) if m.is_dir() => {
                if let Err(e) = collect_wk(p, &mut files) {
                    eprintln!("wukongc fmt: cannot read directory {}: {e}", p.display());
                    return exit::IO_ERROR;
                }
            }
            Ok(_) => files.push(p.to_path_buf()),
            Err(e) => {
                eprintln!("wukongc fmt: cannot read {}: {e}", p.display());
                return exit::IO_ERROR;
            }
        }
    }
    if stdout && files.len() != 1 {
        return usage_error("`--stdout` takes exactly one file");
    }
    let mut code = exit::OK;
    for f in &files {
        let name = f.display().to_string();
        let src = match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("wukongc fmt: cannot read {name}: {e}");
                code = code.max(exit::IO_ERROR);
                continue;
            }
        };
        code = code.max(format_one(&name, Some(f), &src, &mode));
    }
    code
}

/// Every `*.wk` file under `dir`, recursively, in sorted order.
fn collect_wk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect_wk(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "wk") {
            out.push(p);
        }
    }
    Ok(())
}

/// Format one input; `path` is `None` for standard input. Returns its exit code.
fn format_one(name: &str, path: Option<&Path>, src: &str, mode: &Mode) -> i32 {
    let mut sm = SourceMap::new();
    let id = sm.add(name.to_string(), src.to_string());
    match format_source(src, id, &FmtOptions::default()) {
        Ok(out) => {
            if mode.check {
                if out != src {
                    println!("{name}");
                    return exit::COMPILE_ERROR;
                }
                return exit::OK;
            }
            if mode.stdout {
                let mut so = std::io::stdout().lock();
                if so
                    .write_all(out.as_bytes())
                    .and_then(|_| so.flush())
                    .is_err()
                {
                    return exit::IO_ERROR;
                }
                return exit::OK;
            }
            let Some(path) = path else {
                return exit::OK;
            };
            if out == src {
                return exit::OK;
            }
            match write_atomically(path, &out) {
                Ok(()) => exit::OK,
                Err(e) => {
                    eprintln!("wukongc fmt: cannot write {name}: {e}");
                    exit::IO_ERROR
                }
            }
        }
        Err(FmtError::Syntax(diags)) => {
            let r = wukong_diag::Renderer::new(mode.color);
            for d in &diags {
                eprint!("{}", r.render(d, &sm));
            }
            eprintln!("wukongc fmt: {name} has syntax errors; file left unchanged");
            exit::COMPILE_ERROR
        }
        Err(FmtError::Unsupported(msg)) => {
            eprintln!(
                "wukongc fmt: cannot format {name} without changing its meaning: {msg}; file left \
                 unchanged"
            );
            exit::UNIMPLEMENTED
        }
    }
}

/// Write `<file>.fmt.tmp` next to `path`, then rename it over `path`.
fn write_atomically(path: &Path, text: &str) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".fmt.tmp");
    let tmp = PathBuf::from(tmp);
    if let Err(e) = std::fs::write(&tmp, text) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}
