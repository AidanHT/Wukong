//! The process interface of a Wukong program: its command-line arguments, its environment, its
//! standard input, and an early exit — the `arg_count` / `arg` / `arg_i64` / `arg_f32` / `env` /
//! `read_stdin_u8` / `exit` builtins.
//!
//! **Arguments.** A linked executable sees its own `argv` (`std::env::args_os`, which the Rust
//! standard library captures however the C entry point was reached). A host that runs the program
//! in-process — `wukongc --run prog.wk -- a b` on the interpreter or the JIT — installs the
//! program's vector instead with [`set_program_args`]: `arg(0)` is the program (its source path),
//! then the arguments after `--`. An index out of range reads as the empty string, so `arg(i)` is
//! never null and is always usable as a `*u8` path.
//!
//! **Numbers.** `arg_i64(i, default)` / `arg_f32(i, default)` parse argument `i` (surrounding
//! whitespace ignored) with Rust's `str::parse`, and return `default` when it is absent or does
//! not parse — one rule, shared by every backend.
//!
//! **Environment.** `env(name)` reads a variable once and keeps its value for the rest of the run;
//! an unset (or non-UTF-8-named) variable reads as the empty string.
//!
//! **Exit.** `exit(code)` ends the program with `code` as its exit status, as returning `code`
//! from `main` would, after writing out everything printed so far. A linked executable exits the
//! process here; the in-process hosts unwind the run instead (the JIT returns through every caller
//! as for a failed `assert`, the interpreter returns an error carrying [`EXIT_SENTINEL`]) and report
//! the code as the run's result.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::Mutex;

use crate::fileio::IO_ERROR;

/// The prefix of the run error an in-process host raises for `exit(code)`; the decimal `code`
/// follows. A NUL can never start a real diagnostic, so the marker cannot be forged by one.
pub const EXIT_SENTINEL: &str = "\u{0}wukong-exit:";

/// The run's exit code when `err` is an `exit(code)` sentinel ([`EXIT_SENTINEL`]), else `None`.
pub fn exit_code_of(err: &str) -> Option<i64> {
    err.strip_prefix(EXIT_SENTINEL)?.parse().ok()
}

/// One program's argument vector, NUL-terminated so a native caller can read each in place. Leaked
/// on purpose: a pointer `arg(i)` returned stays valid for the rest of the process.
struct ProgramArgs {
    args: Vec<CString>,
}

static PROGRAM_ARGS: Mutex<Option<&'static ProgramArgs>> = Mutex::new(None);

/// A `CString` of `bytes` up to their first NUL (an argument cannot carry one through `argv`).
fn c_string(mut bytes: Vec<u8>) -> CString {
    if let Some(end) = bytes.iter().position(|&b| b == 0) {
        bytes.truncate(end);
    }
    CString::new(bytes).unwrap_or_default()
}

/// Install the program's argument vector — `args[0]` is the program — for every later `arg*`
/// query in this process. An in-process host calls this before each run.
pub fn set_program_args<I: IntoIterator<Item = String>>(args: I) {
    let leaked: &'static ProgramArgs = Box::leak(Box::new(ProgramArgs {
        args: args.into_iter().map(|a| c_string(a.into_bytes())).collect(),
    }));
    *PROGRAM_ARGS.lock().unwrap_or_else(|e| e.into_inner()) = Some(leaked);
}

/// The argument vector in effect: the installed one, else this process's own.
fn program_args() -> &'static ProgramArgs {
    let mut slot = PROGRAM_ARGS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(args) = *slot {
        return args;
    }
    let own: &'static ProgramArgs = Box::leak(Box::new(ProgramArgs {
        args: std::env::args_os()
            .map(|a| c_string(a.to_string_lossy().into_owned().into_bytes()))
            .collect(),
    }));
    *slot = Some(own);
    own
}

/// `arg_count()`: the number of arguments, the program itself included.
pub fn arg_count() -> i64 {
    i64::try_from(program_args().args.len()).unwrap_or(i64::MAX)
}

/// Argument `i` as a NUL-terminated string; the empty string when `i` is out of range.
pub fn arg_cstr(i: i64) -> &'static CStr {
    usize::try_from(i)
        .ok()
        .and_then(|i| program_args().args.get(i))
        .map_or(c"", |a| a.as_c_str())
}

/// Parse argument text as an `i64` (surrounding whitespace ignored), or `default`.
pub fn parse_i64_arg(bytes: &[u8], default: i64) -> i64 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

/// Parse argument text as an `f32` (surrounding whitespace ignored), or `default`.
pub fn parse_f32_arg(bytes: &[u8], default: f32) -> f32 {
    std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

static ENV_CACHE: Mutex<Option<HashMap<Vec<u8>, &'static CStr>>> = Mutex::new(None);

/// The value of the environment variable `name` as a NUL-terminated string, read once per name and
/// kept for the rest of the process; the empty string when it is unset or `name` is not UTF-8.
pub fn env_cstr(name: &[u8]) -> &'static CStr {
    let mut cache = ENV_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some(&v) = cache.get(name) {
        return v;
    }
    let value = std::str::from_utf8(name)
        .ok()
        .filter(|n| !n.is_empty() && !n.contains('='))
        .and_then(std::env::var_os)
        .map(|v| v.to_string_lossy().into_owned().into_bytes())
        .unwrap_or_default();
    let leaked: &'static CStr = Box::leak(c_string(value).into_boxed_c_str());
    cache.insert(name.to_vec(), leaked);
    leaked
}

/// Set once a host that owns this process's standard input itself — `wukongc --repl`, which reads
/// its inputs from it — has detached the program from it: a program then reads end of input.
static STDIN_DETACHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Detach every program this process runs from standard input, for good: `read_stdin_u8` reads
/// nothing (end of input) from now on. `wukongc --repl` calls it, so a program it evaluates cannot
/// consume the session's own input lines.
pub fn detach_stdin() {
    STDIN_DETACHED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Read standard input into `buf` until it is full or input ends; the byte count, or
/// `fileio::IO_ERROR` (`-2`) when reading fails. `0` once [`detach_stdin`] was called.
pub fn read_stdin_into(buf: &mut [u8]) -> i64 {
    use std::io::Read;
    if STDIN_DETACHED.load(std::sync::atomic::Ordering::Relaxed) {
        return 0;
    }
    let mut stdin = std::io::stdin().lock();
    let mut n = 0usize;
    while n < buf.len() {
        match stdin.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return IO_ERROR,
        }
    }
    i64::try_from(n).unwrap_or(i64::MAX)
}

// ---- The native entries (a linked executable; the JIT binds all but `wukong_rt_exit`) ----------

/// `arg_count() -> i64`.
#[no_mangle]
pub extern "C" fn wukong_rt_arg_count() -> i64 {
    arg_count()
}

/// `arg(i) -> *u8`: a NUL-terminated string valid for the rest of the process.
#[no_mangle]
pub extern "C" fn wukong_rt_arg(i: i64) -> *const u8 {
    arg_cstr(i).as_ptr() as *const u8
}

/// `arg_i64(i, default) -> i64`.
#[no_mangle]
pub extern "C" fn wukong_rt_arg_i64(i: i64, default: i64) -> i64 {
    parse_i64_arg(arg_cstr(i).to_bytes(), default)
}

/// `arg_f32(i, default) -> f32`.
#[no_mangle]
pub extern "C" fn wukong_rt_arg_f32(i: i64, default: f32) -> f32 {
    parse_f32_arg(arg_cstr(i).to_bytes(), default)
}

/// `env(name) -> *u8`: a NUL-terminated string valid for the rest of the process.
///
/// # Safety
/// `name` must be null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_env(name: *const u8) -> *const u8 {
    // SAFETY-EXEMPT: `name` is null or a NUL-terminated string -- it carries no extent to check.
    // SAFETY: forwarded from this function's contract (`str_operand` reads to the NUL).
    let name = unsafe { crate::fmt::str_operand(name, -1) };
    env_cstr(name).as_ptr() as *const u8
}

/// `read_stdin_u8(buf) -> i64`: up to `cap` bytes of standard input into `buf`.
///
/// # Safety
/// When `cap > 0`, `buf` must be valid for writes of `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_read_stdin(buf: *mut u8, cap: i64) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(buf, cap);
    let cap = usize::try_from(cap).unwrap_or(0);
    if buf.is_null() || cap == 0 {
        return 0;
    }
    // SAFETY: `buf` is valid for `cap` writable bytes (the caller's contract).
    let buf = unsafe { std::slice::from_raw_parts_mut(buf, cap) };
    read_stdin_into(buf)
}

/// `exit(code)` in a linked executable: write out what was printed, then end the process with
/// `code` — what returning `code` from `main` does.
#[no_mangle]
pub extern "C" fn wukong_rt_exit(code: i64) {
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::process::exit(code as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_parse_with_defaults() {
        assert_eq!(parse_i64_arg(b"42", 0), 42);
        assert_eq!(parse_i64_arg(b" -7 ", 0), -7);
        assert_eq!(parse_i64_arg(b"4x", 9), 9);
        assert_eq!(parse_i64_arg(b"", 9), 9);
        assert_eq!(parse_f32_arg(b"3.5", 0.0), 3.5);
        assert_eq!(parse_f32_arg(b"1e3", 0.0), 1000.0);
        assert_eq!(parse_f32_arg(b"x", 0.25), 0.25);
        assert_eq!(exit_code_of(&format!("{EXIT_SENTINEL}-3")), Some(-3));
        assert_eq!(exit_code_of("assertion failed"), None);
        // An argument cannot carry a NUL; it ends there.
        assert_eq!(c_string(b"ab\0cd".to_vec()).as_bytes(), b"ab");
    }

    #[test]
    fn installed_arguments_index_like_argv() {
        set_program_args(["prog.wk".to_string(), "alpha".to_string(), "42".to_string()]);
        assert_eq!(arg_count(), 3);
        assert_eq!(arg_cstr(1).to_bytes(), b"alpha");
        assert_eq!(arg_cstr(3).to_bytes(), b"");
        assert_eq!(arg_cstr(-1).to_bytes(), b"");
        assert_eq!(wukong_rt_arg_i64(2, 0), 42);
        assert_eq!(wukong_rt_arg_i64(1, -1), -1);
        assert_eq!(env_cstr(b"").to_bytes(), b"");
        assert_eq!(env_cstr(b"A=B").to_bytes(), b"");
    }
}
