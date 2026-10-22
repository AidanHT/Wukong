//! FileCheck-style filetests: `tests/filetests/**/*.{wk,mir}`, each a program plus the commands
//! that compile it and the `CHECK:` lines its output must match.
//!
//! A filetest pins what a pass *does* to the MIR — the invariant load sits in the preheader and
//! not in the loop body, the tail call became a branch — which no differential gate can see: an
//! optimization that silently stops firing changes no program's output. Each file is
//! self-contained, LLVM `lit` style:
//!
//! ```text
//! // RUN: wukongc --emit=mir --passes=mem2reg,licm %s | filecheck %s
//! // RUN: wukongc --emit=mir -O0 %s | filecheck %s --check-prefix=O0
//! // CHECK-LABEL: fn scale(
//! // CHECK:       fmul
//! ```
//!
//! **`RUN:`** lines (a trailing `\` continues onto the next `RUN:` line) run in order and must all
//! succeed. Each is a pipeline of commands joined by `|`, run with `pipefail`: every stage must
//! succeed, not just the last. The commands:
//!
//! - `wukongc <args>` — the real binary (`CARGO_BIN_EXE_wukongc`), with the environment switches
//!   that change what it emits cleared, run in the test's scratch directory;
//! - `filecheck <checkfile> [options]` — the in-tree matcher (`filecheck/mod.rs`) over its stdin
//!   (or `--input-file=<f>`); options `--check-prefix=<P>`, `--check-prefixes=<P>,<Q>`,
//!   `--implicit-check-not=<pattern>`, `--allow-empty`, `--strict-whitespace`;
//! - `diff <a> <b>` — the two files must be byte-identical;
//! - `not <command>` — the command must fail *cleanly*: a non-zero exit that is not a crash (exit
//!   101, a signal, or a Rust panic message in its stderr). The panic message is looked for
//!   wherever the RUN line sent stderr — captured, a `2>` file, or merged into stdout by `2>&1` —
//!   because `wukongc` exits 1, not 101, after a panic on its compile thread.
//!
//! Redirections: `> <file>`, `< <file>`, `2> <file>`, and `2>&1` (stderr goes where stdout goes).
//! Substitutions: `%s` the test file, `%S` its directory, `%t` a path unique to the test in a
//! scratch directory (`%t.mir`, `%t.a`, ...), `%%` a literal `%`. Words split on blanks; `'..'`
//! and `".."` quote.
//!
//! **`REQUIRES: <feature>[, ..]`** skips the file unless every feature holds, and
//! **`UNSUPPORTED: <feature>[, ..]`** skips it if any does. The features: `vec-kernels` (this host
//! builds the vectorizer's raw-AVX2 kernels, `wukong_mir::host_supports_vec_kernels`), `windows`,
//! `unix`, `x86_64`. A skipped file is reported by name, never silently.
//!
//! Every file needs a `RUN:` line that checks something (a `filecheck` or a `diff`). A comment whose
//! first word is an all-capitals `<WORD>:` (or `<WORD>-NEXT:`, `-NOT:` ...) naming no prefix a
//! `RUN:` line checks is an error, so a renamed prefix cannot leave its directives unchecked; only
//! `RUN`, `REQUIRES`, `UNSUPPORTED`, `NOTE`, `TODO` and `FIXME` are exempt, so prose in a filetest
//! does not open a comment with `WORD:`. Files under a directory named `Inputs` are support files,
//! not tests. `WUKONG_FILETESTS=<substring>` runs only the files whose path contains it.

mod filecheck;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Below this many files the suite has lost its tree, not shrunk.
const MIN_FILES: usize = 25;

/// The variables that change what `wukongc` emits or prints, cleared from every child so a test
/// sees the compiler's defaults (`mir_text.rs` keeps the same list). `WUKONG_MAX_ISA` stays: the
/// `vec-kernels` feature is evaluated in this process under the same value.
const COMPILER_ENV: &[&str] = &[
    "WUKONG_NO_RECOGNIZE",
    "WUKONG_P4_NO_256",
    "WUKONG_NO_VECTORIZE",
    "WUKONG_NO_UNROLL",
    "WUKONG_NO_UNROLL_JAM",
    "WUKONG_NO_IV_RAMP",
    "WUKONG_NO_SLP",
    "WUKONG_NO_REVERSE_VEC",
    "WUKONG_VEC256_MIN_TRIP",
    "WUKONG_VEC_COST",
    "WUKONG_VEC_TRACE",
    "WUKONG_SLP_TRACE",
    "CLICOLOR_FORCE",
];

/// Comment words that may precede a colon without being a check prefix.
const NON_PREFIX_WORDS: &[&str] = &["RUN", "REQUIRES", "UNSUPPORTED", "NOTE", "TODO", "FIXME"];

const FEATURES: &[&str] = &["vec-kernels", "windows", "unix", "x86_64"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/wukongc sits two levels below the repository root")
        .to_path_buf()
}

fn suite_dir() -> PathBuf {
    repo_root().join("tests").join("filetests")
}

fn feature_holds(feature: &str) -> bool {
    match feature {
        "vec-kernels" => wukong_mir::host_supports_vec_kernels(),
        "windows" => cfg!(windows),
        "unix" => cfg!(unix),
        "x86_64" => cfg!(target_arch = "x86_64"),
        _ => unreachable!("features are validated when a file is read"),
    }
}

// ---------------------------------------------------------------------------------------------
// Reading a test file.
// ---------------------------------------------------------------------------------------------

/// The text after a line's comment marker (`//`, `;`), or `None` for a line that is not a comment.
fn comment_body(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let body = t.trim_start_matches(['/', ';', '#']);
    (body.len() < t.len()).then(|| body.trim_start())
}

/// One `RUN:` command, continuation lines joined.
#[derive(Debug, Clone, PartialEq)]
struct RunLine {
    line: usize,
    text: String,
}

#[derive(Debug, Default)]
struct Header {
    runs: Vec<RunLine>,
    requires: Vec<String>,
    unsupported: Vec<String>,
}

fn read_header(text: &str) -> Result<Header, String> {
    let mut h = Header::default();
    let mut pending: Option<RunLine> = None;
    for (idx, line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let body = comment_body(line);
        let run = body.and_then(|b| b.strip_prefix("RUN:"));
        if let Some(mut open) = pending.take() {
            let Some(more) = run else {
                return Err(format!(
                    "line {}: a `RUN:` line ends in `\\` but line {lineno} is not a `RUN:` line",
                    open.line
                ));
            };
            open.text.push(' ');
            open.text.push_str(more.trim());
            finish_run(open, &mut h, &mut pending);
            continue;
        }
        if let Some(cmd) = run {
            let open = RunLine {
                line: lineno,
                text: cmd.trim().to_string(),
            };
            finish_run(open, &mut h, &mut pending);
            continue;
        }
        for (key, list) in [
            ("REQUIRES:", &mut h.requires),
            ("UNSUPPORTED:", &mut h.unsupported),
        ] {
            if let Some(rest) = body.and_then(|b| b.strip_prefix(key)) {
                for f in rest.split(',').map(str::trim) {
                    if !FEATURES.contains(&f) {
                        return Err(format!(
                            "line {lineno}: unknown feature `{f}` in `{key}`; the features are {}",
                            FEATURES.join(", ")
                        ));
                    }
                    list.push(f.to_string());
                }
            }
        }
    }
    if let Some(open) = pending {
        return Err(format!(
            "line {}: the last `RUN:` line ends in `\\` with nothing after it",
            open.line
        ));
    }
    if h.runs.is_empty() {
        return Err("no `RUN:` line".to_string());
    }
    Ok(h)
}

fn finish_run(mut open: RunLine, h: &mut Header, pending: &mut Option<RunLine>) {
    if let Some(stripped) = open.text.strip_suffix('\\') {
        open.text = stripped.trim_end().to_string();
        *pending = Some(open);
    } else {
        h.runs.push(open);
    }
}

// ---------------------------------------------------------------------------------------------
// The RUN-line shell.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Word(String),
    Pipe,
    Out,
    In,
    ErrOut,
    ErrToOut,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut i = 0;
    let flush = |word: &mut String, in_word: &mut bool, toks: &mut Vec<Tok>| {
        if *in_word {
            toks.push(Tok::Word(std::mem::take(word)));
            *in_word = false;
        }
    };
    while i < chars.len() {
        let c = chars[i];
        let rest: String = chars[i..].iter().take(4).collect();
        match c {
            ' ' | '\t' => flush(&mut word, &mut in_word, &mut toks),
            '|' if rest.starts_with("||") => {
                return Err("`||` is not supported; write one command per RUN line".into())
            }
            '|' => {
                flush(&mut word, &mut in_word, &mut toks);
                toks.push(Tok::Pipe);
            }
            '&' | ';' => {
                return Err(format!(
                    "`{c}` is not supported; write one command per RUN line"
                ))
            }
            '2' if !in_word && rest.starts_with("2>&1") => {
                toks.push(Tok::ErrToOut);
                i += 4;
                continue;
            }
            '2' if !in_word && rest.starts_with("2>") => {
                toks.push(Tok::ErrOut);
                i += 2;
                continue;
            }
            '>' if rest.starts_with(">>") => {
                return Err("`>>` is not supported".into());
            }
            '>' => {
                flush(&mut word, &mut in_word, &mut toks);
                toks.push(Tok::Out);
            }
            '<' => {
                flush(&mut word, &mut in_word, &mut toks);
                toks.push(Tok::In);
            }
            '\'' => {
                in_word = true;
                let Some(len) = chars[i + 1..].iter().position(|&c| c == '\'') else {
                    return Err("unterminated `'` quote".into());
                };
                word.extend(&chars[i + 1..i + 1 + len]);
                i += len + 2;
                continue;
            }
            '"' => {
                in_word = true;
                i += 1;
                loop {
                    match chars.get(i) {
                        None => return Err("unterminated `\"` quote".into()),
                        Some('"') => break,
                        Some('\\') if matches!(chars.get(i + 1), Some('"' | '\\')) => {
                            word.push(chars[i + 1]);
                            i += 2;
                        }
                        Some(&c) => {
                            word.push(c);
                            i += 1;
                        }
                    }
                }
            }
            '\\' => {
                in_word = true;
                let Some(&next) = chars.get(i + 1) else {
                    return Err("a trailing `\\`".into());
                };
                word.push(next);
                i += 1;
            }
            c => {
                in_word = true;
                word.push(c);
            }
        }
        i += 1;
    }
    flush(&mut word, &mut in_word, &mut toks);
    Ok(toks)
}

/// Where a stage's stderr goes.
#[derive(Debug, Clone, PartialEq)]
enum ErrTo {
    /// Captured, and shown if the RUN line fails.
    Capture,
    /// Wherever stdout goes.
    Stdout,
    File(String),
}

#[derive(Debug, Clone, PartialEq)]
struct Stage {
    negate: bool,
    argv: Vec<String>,
    stdin: Option<String>,
    stdout: Option<String>,
    stderr: ErrTo,
}

/// Parse a RUN line into its stages, `subst` applied to every word.
fn parse_pipeline(
    text: &str,
    subst: &dyn Fn(&str) -> Result<String, String>,
) -> Result<Vec<Stage>, String> {
    let toks = tokenize(text)?;
    let mut stages = Vec::new();
    let mut groups: Vec<Vec<Tok>> = vec![Vec::new()];
    for t in toks {
        if t == Tok::Pipe {
            groups.push(Vec::new());
        } else {
            groups.last_mut().unwrap().push(t);
        }
    }
    let last = groups.len() - 1;
    for (gi, group) in groups.into_iter().enumerate() {
        let mut stage = Stage {
            negate: false,
            argv: Vec::new(),
            stdin: None,
            stdout: None,
            stderr: ErrTo::Capture,
        };
        let mut it = group.into_iter();
        while let Some(t) = it.next() {
            let mut target = || match it.next() {
                Some(Tok::Word(w)) => subst(&w),
                _ => Err("a redirection needs a file name after it".to_string()),
            };
            match t {
                Tok::Word(w) if w == "not" && stage.argv.is_empty() => {
                    if stage.negate {
                        return Err("`not not` is not supported".into());
                    }
                    stage.negate = true;
                }
                Tok::Word(w) => stage.argv.push(subst(&w)?),
                Tok::Out => stage.stdout = Some(target()?),
                Tok::In => stage.stdin = Some(target()?),
                Tok::ErrOut => stage.stderr = ErrTo::File(target()?),
                Tok::ErrToOut => stage.stderr = ErrTo::Stdout,
                Tok::Pipe => unreachable!("pipes split the groups"),
            }
        }
        let Some(cmd) = stage.argv.first() else {
            return Err("an empty command in the pipeline".into());
        };
        if !matches!(cmd.as_str(), "wukongc" | "filecheck" | "diff") {
            return Err(format!(
                "unknown command `{cmd}`: the commands are wukongc, filecheck, diff (and `not`)"
            ));
        }
        if stage.stdout.is_some() && gi != last {
            return Err("only the last command of a pipeline may redirect stdout".into());
        }
        if stage.stdin.is_some() && gi != 0 {
            return Err("only the first command of a pipeline may redirect stdin".into());
        }
        stages.push(stage);
    }
    Ok(stages)
}

/// `%s`, `%S`, `%t` and `%%` in `w`.
fn substitute(w: &str, file: &Path, tmp: &Path) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = w.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('s') => out.push_str(&file.display().to_string()),
            Some('S') => out.push_str(&file.parent().unwrap().display().to_string()),
            Some('t') => out.push_str(&tmp.display().to_string()),
            Some('%') => out.push('%'),
            other => {
                return Err(format!(
                    "unknown substitution `%{}`: the substitutions are %s, %S, %t and %%",
                    other.map(String::from).unwrap_or_default()
                ))
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Running a pipeline.
// ---------------------------------------------------------------------------------------------

struct StageResult {
    code: Option<i32>,
    /// What flows to the next stage (empty when stdout was redirected to a file).
    stdout: Vec<u8>,
    /// What the stage wrote to stderr, wherever the RUN line sent it: captured, read back from its
    /// `2>` file, or, under `2>&1`, the stream it shares with stdout. The crash check reads this,
    /// so no redirection can hide a panic from `not`.
    stderr: Vec<u8>,
}

/// Does `text` hold a Rust panic message (`thread '<name>' panicked at ..`, or
/// `thread '<name>' (<id>) panicked at ..` on a newer toolchain)?
fn shows_panic(text: &str) -> bool {
    text.lines().any(|l| {
        l.find("thread '")
            .is_some_and(|at| l[at..].contains(" panicked at "))
    })
}

/// Did a stage do what the RUN line asked? `None` = yes, `Some(why)` = no. `stderr` is everything
/// the stage wrote there ([`StageResult::stderr`]). `wukongc` turns a panic on its compile thread
/// into exit 1, so the panic message is the only sign of that crash.
fn verdict(negate: bool, code: Option<i32>, stderr: &str) -> Option<String> {
    let crashed = code.is_none() || code == Some(101) || shows_panic(stderr);
    match (negate, code) {
        (false, Some(0)) => None,
        (false, _) => Some(format!("exit {}", show_code(code))),
        (true, _) if crashed => Some(format!(
            "`not` needs a clean failure, but the command crashed (exit {})",
            show_code(code)
        )),
        (true, Some(0)) => Some("`not` needs a failure, but the command succeeded".into()),
        (true, _) => None,
    }
}

fn show_code(code: Option<i32>) -> String {
    code.map_or_else(|| "by signal".to_string(), |c| c.to_string())
}

/// The program a `wukongc` stage runs: the real compiler, or — in the harness's own tests — a
/// stand-in that fails the way the compiler can.
struct Compiler {
    exe: PathBuf,
    /// Placed before the stage's own arguments.
    args: Vec<String>,
    /// Added to the child's environment.
    env: Vec<(String, String)>,
}

impl Compiler {
    fn wukongc() -> Compiler {
        Compiler {
            exe: PathBuf::from(env!("CARGO_BIN_EXE_wukongc")),
            args: Vec::new(),
            env: Vec::new(),
        }
    }
}

struct Ctx<'a> {
    file: &'a Path,
    display: &'a str,
    scratch: &'a Path,
    counter: AtomicUsize,
    compiler: &'a Compiler,
}

impl Ctx<'_> {
    fn scratch_file(&self, what: &str) -> PathBuf {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        self.scratch.join(format!("{what}.{n}"))
    }

    /// Resolve a redirection target: relative names are in the scratch directory.
    fn path(&self, p: &str) -> PathBuf {
        let p = Path::new(p);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.scratch.join(p)
        }
    }

    fn run_stage(&self, stage: &Stage, input: Vec<u8>) -> Result<StageResult, String> {
        let input = match &stage.stdin {
            Some(f) => std::fs::read(self.path(f))
                .map_err(|e| format!("cannot read stdin file `{f}`: {e}"))?,
            None => input,
        };
        let mut res = match stage.argv[0].as_str() {
            "wukongc" => return self.run_wukongc(stage, &input),
            "filecheck" => self.filecheck(&stage.argv[1..], &input),
            "diff" => self.diff(&stage.argv[1..]),
            _ => unreachable!("commands are validated when parsed"),
        };
        // A builtin's streams are in memory: route them as the redirections say, keeping in
        // `stderr` what went there (under `2>&1`, the merged stream, as for `wukongc`).
        match &stage.stderr {
            ErrTo::Capture => {}
            ErrTo::Stdout => {
                res.stdout.append(&mut res.stderr);
                res.stderr = res.stdout.clone();
            }
            ErrTo::File(f) => {
                std::fs::write(self.path(f), &res.stderr)
                    .map_err(|e| format!("cannot write `{f}`: {e}"))?;
            }
        }
        if let Some(f) = &stage.stdout {
            std::fs::write(self.path(f), std::mem::take(&mut res.stdout))
                .map_err(|e| format!("cannot write `{f}`: {e}"))?;
        }
        Ok(res)
    }

    fn run_wukongc(&self, stage: &Stage, input: &[u8]) -> Result<StageResult, String> {
        let io = |e: std::io::Error| format!("scratch file: {e}");
        let mut cmd = Command::new(&self.compiler.exe);
        for v in COMPILER_ENV {
            cmd.env_remove(v);
        }
        cmd.envs(self.compiler.env.iter().map(|(k, v)| (k, v)));
        cmd.args(&self.compiler.args)
            .args(&stage.argv[1..])
            .current_dir(self.scratch);
        if input.is_empty() {
            cmd.stdin(Stdio::null());
        } else {
            let f = self.scratch_file("stdin");
            std::fs::write(&f, input).map_err(io)?;
            cmd.stdin(File::open(&f).map_err(io)?);
        }
        let out_path = match &stage.stdout {
            Some(f) => self.path(f),
            None => self.scratch_file("stdout"),
        };
        let out = File::create(&out_path).map_err(io)?;
        let err_path = match &stage.stderr {
            ErrTo::Capture => Some(self.scratch_file("stderr")),
            ErrTo::File(f) => Some(self.path(f)),
            ErrTo::Stdout => None,
        };
        match &err_path {
            Some(p) => cmd.stderr(File::create(p).map_err(io)?),
            // One handle, one file offset: the two streams interleave as they were written.
            None => cmd.stderr(out.try_clone().map_err(io)?),
        };
        cmd.stdout(out);
        let status = cmd
            .status()
            .map_err(|e| format!("cannot spawn wukongc: {e}"))?;
        let written = std::fs::read(&out_path).map_err(io)?;
        // Read stderr back wherever it went: a panic on the compile thread exits 1, and its
        // message is all that tells the crash from a clean failure.
        let stderr = match &err_path {
            Some(p) => std::fs::read(p).map_err(io)?,
            None => written.clone(),
        };
        let stdout = if stage.stdout.is_some() {
            Vec::new()
        } else {
            written
        };
        Ok(StageResult {
            code: status.code(),
            stdout,
            stderr,
        })
    }

    fn filecheck(&self, args: &[String], input: &[u8]) -> StageResult {
        let fail = |msg: String| StageResult {
            code: Some(2),
            stdout: Vec::new(),
            stderr: format!("filecheck: {msg}\n").into_bytes(),
        };
        let mut opts = filecheck::Options::default();
        let mut prefixes: Vec<String> = Vec::new();
        let mut check_file = None;
        let mut input_file = None;
        for a in args {
            if let Some(p) = a.strip_prefix("--check-prefix=") {
                prefixes.push(p.to_string());
            } else if let Some(ps) = a.strip_prefix("--check-prefixes=") {
                prefixes.extend(ps.split(',').map(String::from));
            } else if let Some(p) = a.strip_prefix("--implicit-check-not=") {
                opts.implicit_check_not.push(p.to_string());
            } else if a == "--allow-empty" {
                opts.allow_empty = true;
            } else if a == "--strict-whitespace" {
                opts.strict_whitespace = true;
            } else if let Some(f) = a.strip_prefix("--input-file=") {
                input_file = Some(f.to_string());
            } else if a.starts_with('-') {
                return fail(format!("unknown option `{a}`"));
            } else if check_file.replace(a.clone()).is_some() {
                return fail("more than one check file".into());
            }
        }
        if !prefixes.is_empty() {
            opts.prefixes = prefixes;
        }
        let Some(check_file) = check_file else {
            return fail("no check file (usually `filecheck %s`)".into());
        };
        let check_path = self.path(&check_file);
        let check_text = match std::fs::read_to_string(&check_path) {
            Ok(t) => t,
            Err(e) => return fail(format!("cannot read `{check_file}`: {e}")),
        };
        let name = if check_path == self.file {
            self.display.to_string()
        } else {
            check_file.clone()
        };
        let input = match input_file {
            Some(f) => match std::fs::read(self.path(&f)) {
                Ok(b) => b,
                Err(e) => return fail(format!("cannot read `{f}`: {e}")),
            },
            None => input.to_vec(),
        };
        let input = String::from_utf8_lossy(&input);
        match filecheck::check(&name, &check_text, &input, &opts) {
            Ok(()) => StageResult {
                code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
            },
            Err(report) => StageResult {
                code: Some(1),
                stdout: Vec::new(),
                stderr: format!("{report}\n\n{}", dump_input(&input)).into_bytes(),
            },
        }
    }

    fn diff(&self, args: &[String]) -> StageResult {
        let [a, b] = args else {
            return StageResult {
                code: Some(2),
                stdout: Vec::new(),
                stderr: b"diff: needs exactly two files\n".to_vec(),
            };
        };
        let read = |f: &str| std::fs::read(self.path(f)).map_err(|e| format!("`{f}`: {e}"));
        let (ta, tb) = match (read(a), read(b)) {
            (Ok(x), Ok(y)) => (x, y),
            (Err(e), _) | (_, Err(e)) => {
                return StageResult {
                    code: Some(2),
                    stdout: Vec::new(),
                    stderr: format!("diff: cannot read {e}\n").into_bytes(),
                }
            }
        };
        if ta == tb {
            return StageResult {
                code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
            };
        }
        let (sa, sb) = (String::from_utf8_lossy(&ta), String::from_utf8_lossy(&tb));
        let (la, lb): (Vec<&str>, Vec<&str>) = (sa.lines().collect(), sb.lines().collect());
        let at = la
            .iter()
            .zip(&lb)
            .position(|(x, y)| x != y)
            .unwrap_or(la.len().min(lb.len()));
        let mut msg = format!(
            "{a} and {b} differ ({} vs {} lines); first difference at line {}:\n",
            la.len(),
            lb.len(),
            at + 1
        );
        for k in at.saturating_sub(2)..(at + 3) {
            match (la.get(k), lb.get(k)) {
                (Some(x), Some(y)) if x == y => msg.push_str(&format!("  {:>5} | {x}\n", k + 1)),
                (x, y) => {
                    if let Some(x) = x {
                        msg.push_str(&format!("< {:>5} | {x}\n", k + 1));
                    }
                    if let Some(y) = y {
                        msg.push_str(&format!("> {:>5} | {y}\n", k + 1));
                    }
                }
            }
        }
        StageResult {
            code: Some(1),
            stdout: msg.into_bytes(),
            stderr: Vec::new(),
        }
    }

    /// Run one RUN line. `Err` is the failure report.
    fn run_line(&self, run: &RunLine) -> Result<(), String> {
        let subst = |w: &str| substitute(w, self.file, &self.scratch.join("t"));
        let stages = parse_pipeline(&run.text, &subst)
            .map_err(|e| format!("{}:{}: bad RUN line: {e}", self.display, run.line))?;
        let mut flow = Vec::new();
        let mut report = String::new();
        let mut failed = false;
        for (i, stage) in stages.iter().enumerate() {
            let res = self
                .run_stage(stage, std::mem::take(&mut flow))
                .map_err(|e| format!("{}:{}: RUN: {}\n{e}", self.display, run.line, run.text))?;
            let stderr = String::from_utf8_lossy(&res.stderr).into_owned();
            let shown = format!(
                "{}{}",
                if stage.negate { "not " } else { "" },
                stage.argv.join(" ")
            );
            if let Some(why) = verdict(stage.negate, res.code, &stderr) {
                failed = true;
                report.push_str(&format!("--- stage {} `{shown}`: {why}\n", i + 1));
                // Under `2>&1` the stderr text is the merged stream, stdout included.
                let merged = stage.stderr == ErrTo::Stdout;
                if !stderr.is_empty() {
                    let what = if merged {
                        "its stdout and stderr (2>&1)"
                    } else {
                        "its stderr"
                    };
                    report.push_str(&format!("--- {what}:\n{stderr}\n"));
                }
                // A failed compile's stdout, or what a failed diff printed.
                let out = String::from_utf8_lossy(&res.stdout);
                if !merged && !out.is_empty() && stage.argv[0] != "filecheck" {
                    report.push_str(&format!("--- its stdout:\n{out}\n"));
                }
            }
            flow = res.stdout;
        }
        if failed {
            Err(format!(
                "{}:{}: RUN: {}\n{report}",
                self.display, run.line, run.text
            ))
        } else {
            Ok(())
        }
    }
}

/// The input as filecheck saw it, numbered.
fn dump_input(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    let mut s = format!("--- the full input to filecheck ({} lines):\n", lines.len());
    for (i, l) in lines.iter().enumerate() {
        s.push_str(&format!("{:>5} | {l}\n", i + 1));
    }
    s
}

// ---------------------------------------------------------------------------------------------
// One file.
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
enum Outcome {
    Pass,
    Skip(String),
    Fail(String),
}

/// The prefixes the RUN lines check, and whether any of them asserts on output at all.
fn checked_prefixes(runs: &[RunLine], file: &Path) -> Result<(Vec<String>, bool), String> {
    let mut prefixes = Vec::new();
    let mut asserts = false;
    let tmp = Path::new("t");
    for run in runs {
        let stages = parse_pipeline(&run.text, &|w| substitute(w, file, tmp))
            .map_err(|e| format!("line {}: bad RUN line: {e}", run.line))?;
        for st in &stages {
            match st.argv[0].as_str() {
                "diff" => asserts = true,
                "filecheck" => {
                    asserts = true;
                    let mut mine = Vec::new();
                    for a in &st.argv[1..] {
                        if let Some(p) = a.strip_prefix("--check-prefix=") {
                            mine.push(p.to_string());
                        } else if let Some(ps) = a.strip_prefix("--check-prefixes=") {
                            mine.extend(ps.split(',').map(String::from));
                        }
                    }
                    if mine.is_empty() {
                        mine.push("CHECK".to_string());
                    }
                    for p in mine {
                        if !prefixes.contains(&p) {
                            prefixes.push(p);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok((prefixes, asserts))
}

/// A comment whose first word reads as a directive for a prefix nothing checks.
fn unchecked_directives(text: &str, prefixes: &[String]) -> Vec<String> {
    let mut problems = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let Some(body) = comment_body(line) else {
            continue;
        };
        let Some((word, _)) = body.split_once(':') else {
            continue;
        };
        let is_directive_word = word.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            && word
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_' || c == '-');
        if !is_directive_word {
            continue;
        }
        let mut base = word;
        for suffix in ["-NEXT", "-SAME", "-NOT", "-DAG", "-LABEL", "-EMPTY"] {
            if let Some(b) = base.strip_suffix(suffix) {
                base = b;
                break;
            }
        }
        if let Some((b, n)) = base.rsplit_once("-COUNT-") {
            if n.chars().all(|c| c.is_ascii_digit()) {
                base = b;
            }
        }
        if NON_PREFIX_WORDS.contains(&base) || prefixes.iter().any(|p| p == base) {
            continue;
        }
        problems.push(format!(
            "line {}: `{word}:` is a directive for the prefix `{base}`, which no RUN line checks \
             (checked: {})",
            idx + 1,
            prefixes.join(", ")
        ));
    }
    problems
}

fn display_name(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Run every RUN line of one test file.
fn run_file(path: &Path, scratch_root: &Path) -> Outcome {
    run_file_with(path, scratch_root, &Compiler::wukongc())
}

/// [`run_file`], its `wukongc` stages running `compiler`.
fn run_file_with(path: &Path, scratch_root: &Path, compiler: &Compiler) -> Outcome {
    let display = display_name(path);
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return Outcome::Fail(format!("{display}: cannot read: {e}")),
    };
    let header = match read_header(&text) {
        Ok(h) => h,
        Err(e) => return Outcome::Fail(format!("{display}: {e}")),
    };
    let missing: Vec<&String> = header
        .requires
        .iter()
        .filter(|f| !feature_holds(f))
        .collect();
    if !missing.is_empty() {
        return Outcome::Skip(format!("{display} (REQUIRES {})", join(&missing)));
    }
    let present: Vec<&String> = header
        .unsupported
        .iter()
        .filter(|f| feature_holds(f))
        .collect();
    if !present.is_empty() {
        return Outcome::Skip(format!("{display} (UNSUPPORTED {})", join(&present)));
    }
    let (prefixes, asserts) = match checked_prefixes(&header.runs, path) {
        Ok(x) => x,
        Err(e) => return Outcome::Fail(format!("{display}: {e}")),
    };
    if !asserts {
        return Outcome::Fail(format!(
            "{display}: no RUN line checks anything: pipe into `filecheck %s` or compare with \
             `diff`"
        ));
    }
    let problems = unchecked_directives(&text, &prefixes);
    if !problems.is_empty() {
        return Outcome::Fail(format!("{display}:\n{}", problems.join("\n")));
    }
    // Named by the path under the suite (the file name for a file elsewhere), in safe characters.
    let under_suite = path
        .strip_prefix(suite_dir())
        .unwrap_or(Path::new(path.file_name().unwrap()));
    let tag: String = under_suite
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let scratch = scratch_root.join(tag);
    let _ = std::fs::remove_dir_all(&scratch);
    if let Err(e) = std::fs::create_dir_all(&scratch) {
        return Outcome::Fail(format!(
            "{display}: cannot create {}: {e}",
            scratch.display()
        ));
    }
    let ctx = Ctx {
        file: path,
        display: &display,
        scratch: &scratch,
        counter: AtomicUsize::new(0),
        compiler,
    };
    for run in &header.runs {
        if let Err(report) = ctx.run_line(run) {
            return Outcome::Fail(format!(
                "{report}(scratch files kept in {})",
                scratch.display()
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);
    Outcome::Pass
}

fn join(v: &[&String]) -> String {
    v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
}

/// Every test file under `dir`, sorted, skipping `Inputs` directories.
fn discover(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "Inputs") {
                discover(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "wk" || x == "mir") {
            out.push(p);
        }
    }
}

fn scratch_root(tag: &str) -> PathBuf {
    std::env::temp_dir()
        .join("wukong_filetests")
        .join(format!("{}_{tag}", std::process::id()))
}

/// Run `files` on up to four workers (each on a roomy stack: the matcher recurses per character).
fn run_all(files: &[PathBuf], scratch: &Path) -> Vec<Outcome> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Outcome>>> = Mutex::new((0..files.len()).map(|_| None).collect());
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 4);
    std::thread::scope(|s| {
        for _ in 0..workers {
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn_scoped(s, || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(f) = files.get(i) else {
                        break;
                    };
                    let outcome = run_file(f, scratch);
                    results.lock().unwrap()[i] = Some(outcome);
                })
                .expect("spawn a filetest worker");
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|o| o.expect("every file ran"))
        .collect()
}

#[test]
fn filetests() {
    let mut files = Vec::new();
    discover(&suite_dir(), &mut files);
    let filter = std::env::var("WUKONG_FILETESTS")
        .ok()
        .filter(|f| !f.is_empty());
    if let Some(f) = &filter {
        files.retain(|p| display_name(p).contains(f.as_str()));
        assert!(
            !files.is_empty(),
            "WUKONG_FILETESTS={f} matches no filetest"
        );
    } else {
        assert!(
            files.len() >= MIN_FILES,
            "found {} filetests under {}; the suite holds at least {MIN_FILES}",
            files.len(),
            suite_dir().display()
        );
    }
    let scratch = scratch_root("suite");
    let outcomes = run_all(&files, &scratch);
    let _ = std::fs::remove_dir(&scratch);
    let mut failures = Vec::new();
    let mut skipped = Vec::new();
    let mut passed = 0;
    for o in outcomes {
        match o {
            Outcome::Pass => passed += 1,
            Outcome::Skip(why) => skipped.push(why),
            Outcome::Fail(report) => failures.push(report),
        }
    }
    eprintln!(
        "filetests: {} files: {passed} passed, {} failed, {} skipped",
        files.len(),
        failures.len(),
        skipped.len()
    );
    for s in &skipped {
        eprintln!("  skipped: {s}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} filetests failed:\n\n{}",
        failures.len(),
        files.len(),
        failures.join("\n\n")
    );
}

// ---------------------------------------------------------------------------------------------
// The harness's own tests.
// ---------------------------------------------------------------------------------------------

fn no_subst(w: &str) -> Result<String, String> {
    Ok(w.to_string())
}

#[test]
fn run_lines_tokenize_like_a_shell() {
    use Tok::*;
    let w = |s: &str| Word(s.to_string());
    assert_eq!(
        tokenize(r#"not wukongc -e 'fn main() {}' "a \"b\"" x\ y 2>&1 | filecheck %s > out"#)
            .unwrap(),
        vec![
            w("not"),
            w("wukongc"),
            w("-e"),
            w("fn main() {}"),
            w("a \"b\""),
            w("x y"),
            ErrToOut,
            Pipe,
            w("filecheck"),
            w("%s"),
            Out,
            w("out"),
        ]
    );
    assert_eq!(
        tokenize("a 2> e < i b2>c").unwrap(),
        vec![w("a"), ErrOut, w("e"), In, w("i"), w("b2"), Out, w("c")]
    );
    assert_eq!(tokenize("--x='a b'c").unwrap(), vec![w("--x=a bc")]);
    for bad in [
        "a && b", "a; b", "a || b", "a >> f", "'open", "\"open", "a \\",
    ] {
        assert!(tokenize(bad).is_err(), "{bad} must be refused");
    }
}

#[test]
fn run_lines_parse_into_stages() {
    let stages = parse_pipeline(
        "not wukongc --emit=mir x.mir 2>&1 | filecheck %s --check-prefix=E",
        &no_subst,
    )
    .unwrap();
    assert_eq!(stages.len(), 2);
    assert!(stages[0].negate);
    assert_eq!(stages[0].argv, ["wukongc", "--emit=mir", "x.mir"]);
    assert_eq!(stages[0].stderr, ErrTo::Stdout);
    assert_eq!(stages[1].argv, ["filecheck", "%s", "--check-prefix=E"]);
    let s = parse_pipeline("wukongc a > %t.mir", &no_subst).unwrap();
    assert_eq!(s[0].stdout.as_deref(), Some("%t.mir"));
    for (bad, want) in [
        ("cat x", "unknown command `cat`"),
        ("wukongc a > f | filecheck %s", "only the last command"),
        ("wukongc a | filecheck < f", "only the first command"),
        ("wukongc a |", "an empty command"),
        ("not not wukongc", "`not not`"),
        ("wukongc >", "needs a file name"),
    ] {
        let e = parse_pipeline(bad, &no_subst).unwrap_err();
        assert!(e.contains(want), "{bad}: {e}");
    }
}

#[test]
fn substitutions_expand() {
    let file = Path::new("/r/tests/filetests/a.mir");
    let tmp = Path::new("/tmp/x/t");
    assert_eq!(
        substitute("%s|%S|%t.mir|100%%", file, tmp).unwrap(),
        format!(
            "{}|{}|{}.mir|100%",
            file.display(),
            file.parent().unwrap().display(),
            tmp.display()
        )
    );
    assert!(substitute("%q", file, tmp).unwrap_err().contains("%q"));
    assert!(substitute("50%", file, tmp).is_err());
}

#[test]
fn run_headers_join_continuations_and_read_features() {
    let h = read_header(
        "// RUN: wukongc a \\\n//   RUN:   b | filecheck %s\n// REQUIRES: vec-kernels, windows\n; RUN: diff x y\n",
    )
    .unwrap();
    assert_eq!(
        h.runs,
        vec![
            RunLine {
                line: 1,
                text: "wukongc a b | filecheck %s".into()
            },
            RunLine {
                line: 4,
                text: "diff x y".into()
            },
        ]
    );
    assert_eq!(h.requires, ["vec-kernels", "windows"]);
    assert!(read_header("// nothing")
        .unwrap_err()
        .contains("no `RUN:` line"));
    assert!(read_header("// RUN: a \\\n// CHECK: x")
        .unwrap_err()
        .contains("is not a `RUN:` line"));
    assert!(read_header("// RUN: a\n// REQUIRES: avx9000")
        .unwrap_err()
        .contains("unknown feature `avx9000`"));
}

#[test]
fn not_needs_a_clean_failure() {
    assert_eq!(verdict(false, Some(0), ""), None);
    assert!(verdict(false, Some(1), "").unwrap().contains("exit 1"));
    assert_eq!(verdict(true, Some(1), "error: x"), None);
    assert!(verdict(true, Some(0), "").unwrap().contains("succeeded"));
    assert!(verdict(true, Some(101), "").unwrap().contains("crashed"));
    assert!(verdict(true, None, "").unwrap().contains("crashed"));
    assert!(verdict(true, Some(1), "thread 'x' panicked at src/a.rs")
        .unwrap()
        .contains("crashed"));
    // A newer toolchain puts the thread id between the name and `panicked`.
    assert!(verdict(
        true,
        Some(1),
        "fn main\nthread 'x' (7) panicked at src/a.rs:1:2:\nmsg"
    )
    .unwrap()
    .contains("crashed"));
    // Text that merely says "panicked at" is no panic message.
    assert_eq!(
        verdict(true, Some(1), "error: the pass panicked at v3"),
        None
    );
}

#[test]
fn directives_for_an_unchecked_prefix_are_errors() {
    let prefixes = vec!["CHECK".to_string(), "O2".to_string()];
    let text = "// CHECK: a\n// O2-NEXT: b\n// O3: c\n// O3-COUNT-2: d\n// NOTE: prose\n\
                // RUN: x\n// lower: case\n// Mixed: case";
    let p = unchecked_directives(text, &prefixes);
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[0].starts_with("line 3: `O3:`"), "{p:?}");
    assert!(p[1].starts_with("line 4: `O3-COUNT-2:`"), "{p:?}");
}

/// Write `body` as a filetest in a fresh directory and run it.
fn run_scratch_test(name: &str, body: &str) -> Outcome {
    run_scratch_test_with(name, body, &Compiler::wukongc())
}

/// [`run_scratch_test`], its `wukongc` stages running `compiler`.
fn run_scratch_test_with(name: &str, body: &str, compiler: &Compiler) -> Outcome {
    let dir = scratch_root("selftest").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.wk"));
    std::fs::write(&path, body).unwrap();
    let outcome = run_file_with(&path, &dir.join("scratch"), compiler);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

const TINY: &str = "fn main() -> i32 {\n    let x: i32 = 40;\n    return x + 2;\n}\n";

#[test]
fn the_harness_passes_a_correct_test() {
    let body = format!(
        "// RUN: wukongc --emit=mir %s | filecheck %s\n\
         // RUN: wukongc --emit=mir %s > %t.out\n\
         // RUN: wukongc --emit=mir %s > %t.again\n\
         // RUN: diff %t.out %t.again\n\
         // RUN: not wukongc --emit=bogus %s\n\
         // CHECK-LABEL: fn main() -> i32 {{\n\
         // CHECK: ret\n{TINY}"
    );
    match run_scratch_test("pass", &body) {
        Outcome::Pass => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_harness_reports_a_wrong_check_with_its_location_and_input() {
    let body = format!(
        "// RUN: wukongc --emit=mir %s | filecheck %s\n\
         // CHECK-LABEL: fn main() -> i32 {{\n\
         // CHECK: rett\n{TINY}"
    );
    let Outcome::Fail(report) = run_scratch_test("wrong", &body) else {
        panic!("a wrong CHECK must fail");
    };
    for want in [
        "wrong.wk:3:11: error: CHECK: expected string not found in input",
        "note: possible intended match here",
        "--- the full input to filecheck",
        "| fn main() -> i32 {",
        "RUN: wukongc --emit=mir %s | filecheck %s",
    ] {
        assert!(report.contains(want), "missing {want:?} in:\n{report}");
    }
}

#[test]
fn the_harness_fails_a_failing_compile_and_a_passing_not() {
    let body = format!("// RUN: wukongc --emit=bogus %s | filecheck %s\n// CHECK: x\n{TINY}");
    let Outcome::Fail(report) = run_scratch_test("compile", &body) else {
        panic!("a failing compile must fail the test");
    };
    assert!(report.contains("stage 1 `wukongc --emit=bogus"), "{report}");
    assert!(report.contains("--- its stderr:"), "{report}");
    let body = format!("// RUN: not wukongc --emit=mir %s | filecheck %s\n// CHECK: fn\n{TINY}");
    let Outcome::Fail(report) = run_scratch_test("notpass", &body) else {
        panic!("`not` on a success must fail the test");
    };
    assert!(report.contains("`not` needs a failure"), "{report}");
}

#[test]
fn the_harness_refuses_a_test_that_checks_nothing() {
    let body = format!("// RUN: wukongc --emit=mir %s\n{TINY}");
    let Outcome::Fail(report) = run_scratch_test("vacuous", &body) else {
        panic!("a RUN line with no assertion must fail");
    };
    assert!(report.contains("no RUN line checks anything"), "{report}");
    let body =
        format!("// RUN: wukongc --emit=mir %s | filecheck %s\n// CHECK: fn\n// O2: fn\n{TINY}");
    let Outcome::Fail(report) = run_scratch_test("unchecked", &body) else {
        panic!("an unchecked prefix must fail");
    };
    assert!(
        report.contains("`O2:` is a directive for the prefix `O2`"),
        "{report}"
    );
}

/// Set in the child [`stand_in_compiler`] runs as: `panic` or `error`.
const STAND_IN: &str = "WUKONG_FILETESTS_STAND_IN";

/// The message both stand-in modes print, as the MIR verifier words it.
const ICE: &str = "internal compiler error (MIR verify): use of undefined value v7";

/// A no-op in the suite. Run as a child of this test binary with [`STAND_IN`] set, it fails the
/// way `wukongc` does: `panic` panics on a thread named like the compile thread and exits 1, just
/// as `run_compile` turns a panicked join into a compile error, so the exit code cannot tell the
/// crash apart and only the panic message on stderr can; `error` prints the message as a
/// diagnostic and exits 1, a clean failure.
#[test]
fn stand_in_compiler() {
    let Some(mode) = std::env::var_os(STAND_IN) else {
        return;
    };
    if mode == "panic" {
        let joined = std::thread::Builder::new()
            .name("wukongc-compile".into())
            .spawn(|| panic!("{ICE}"))
            .unwrap()
            .join();
        assert!(joined.is_err());
    } else {
        eprintln!("error: {ICE}");
    }
    std::process::exit(1);
}

/// This test binary, run as [`stand_in_compiler`] in `mode`.
fn stand_in(mode: &str) -> Compiler {
    Compiler {
        exe: std::env::current_exe().expect("the test binary's path"),
        args: [
            "--exact",
            "stand_in_compiler",
            "--nocapture",
            "--test-threads=1",
        ]
        .map(String::from)
        .to_vec(),
        env: vec![(STAND_IN.to_string(), mode.to_string())],
    }
}

#[test]
fn not_rejects_a_crash_wherever_stderr_was_sent() {
    let check = format!("// CHECK: {ICE}\n");
    let cases = [
        (
            "merged",
            format!("// RUN: not wukongc 2>&1 | filecheck %s\n{check}"),
        ),
        (
            "merged_to_file",
            format!(
                "// RUN: not wukongc > %t.out 2>&1\n\
                 // RUN: filecheck %s --input-file=%t.out\n{check}"
            ),
        ),
        (
            "to_file",
            format!(
                "// RUN: not wukongc 2> %t.err\n\
                 // RUN: filecheck %s --input-file=%t.err\n{check}"
            ),
        ),
        (
            "captured",
            "// RUN: not wukongc\n// RUN: diff %s %s\n".to_string(),
        ),
    ];
    for (name, body) in cases {
        // A diagnostic and exit 1: the clean failure `not` accepts.
        match run_scratch_test_with(&format!("clean_{name}"), &body, &stand_in("error")) {
            Outcome::Pass => {}
            other => panic!("{name}: a clean failure must pass `not`: {other:?}"),
        }
        // The same exit 1 after a panic is a crash, wherever the panic message went.
        let Outcome::Fail(report) =
            run_scratch_test_with(&format!("crash_{name}"), &body, &stand_in("panic"))
        else {
            panic!("{name}: a panic must fail `not` even when it exits 1");
        };
        assert!(
            report.contains("`not` needs a clean failure, but the command crashed (exit 1)"),
            "{name}: {report}"
        );
        assert!(
            report.contains("panicked at"),
            "{name}: the report shows the panic: {report}"
        );
    }
}
