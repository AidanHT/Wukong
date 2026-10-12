//! Every runtime trap names its source site, with the same text and exit status on every CPU
//! execution path: the interpreter (`--run`), the Cranelift JIT (`--run --backend=native`), each at
//! `-O0` and `-O2`, and a linked executable (`--emit=exe`).
//!
//! The traps are a failed `assert` (bare, with a message, a kernel builtin's run-time extent check,
//! which traps through the same intrinsic and so names the builtin call, and a value `match` no arm
//! of which matched, which does too and names the `match`), a slice index out of bounds, the two
//! range-index checks and `str_from_utf8` of bytes that are not UTF-8. Each
//! program is written to a scratch directory and compiled by its bare file name, so its site is
//! exactly `<name>.wk:<line>:<col>`; every expected line and column below is read off the program
//! text above it, never recorded from compiler output. Every case also pins the stdout printed
//! before the failure and that nothing after it runs.
//!
//! The executable half needs a link path. The driver's preferred one (rustc + the `wukong_runtime`
//! rlib) links every case; the C fallback links only the cases that call no `wukong_*` runtime
//! entry (it resolves the `wukong_rt_*` shim symbols and nothing else), so the others are skipped
//! there with a note. With no toolchain at all that half is skipped with a note — this test never
//! depends on the C fallback being present.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Case {
    /// The program's file stem; its site is `<name>.wk:line:col`.
    name: &'static str,
    source: &'static str,
    /// Stdout printed before the failure.
    stdout: &'static str,
    /// The single stderr line every path prints.
    stderr: &'static str,
    /// Calls a `wukong_*` runtime entry the C-fallback link cannot resolve.
    needs_runtime: bool,
}

const CASES: &[Case] = &[
    Case {
        // `assert` opens line 2 at column 5. check(10) passes and main prints 20; check(20) fails.
        name: "assert_site",
        source: "fn check(x: i32) -> i32 {
    assert(x < 15);
    return x * 2;
}

fn main() -> i32 {
    print(check(10));
    print(check(20));
    print(99);
    return 0;
}
",
        stdout: "20\n",
        stderr: "error: assertion failed at assert_site.wk:2:5",
        needs_runtime: false,
    },
    Case {
        // The message literal, then the site: `assert` opens line 2 at column 5. half(4) prints 2.
        name: "assert_message_site",
        source: "fn half(n: i64) -> i64 {
    assert(n % 2 == 0, \"n must be even\");
    return n / 2;
}

fn main() -> i32 {
    print(half(4));
    print(half(7));
    print(99);
    return 0;
}
",
        stdout: "2\n",
        stderr: "error: assertion failed: n must be even at assert_message_site.wk:2:5",
        needs_runtime: false,
    },
    Case {
        // `gelu(x, out)` with a 3-element `out` for a 4-element `x`: the builtin's extent check
        // traps through `assert` before the kernel runs, naming the call (line 5, column 5).
        name: "extent_site",
        source: "fn main() -> i32 {
    let x = alloc_f32(4);
    let out = alloc_f32(3);
    print(1);
    gelu(x, out);
    print(out[0]);
    return 0;
}
",
        stdout: "1\n",
        stderr: "error: assertion failed at extent_site.wk:5:5",
        needs_runtime: true,
    },
    Case {
        // `unset` is zero-filled and no variant has discriminant 0, so no arm of the `match` (line
        // 4, column 13 after `    let w = `) matches it; weight(Level::Low) prints 10 first.
        name: "match_site",
        source: "enum Level { Low = 1, High = 2 }

fn weight(l: Level) -> i32 {
    let w = match l {
        Level::Low => 10,
        Level::High => 20,
    };
    return w;
}

fn main() -> i32 {
    print(weight(Level::Low));
    let unset: Level;
    print(weight(unset));
    print(99);
    return 0;
}
",
        stdout: "10\n",
        stderr: "error: assertion failed: no `match` arm matched its value at match_site.wk:4:13",
        needs_runtime: false,
    },
    Case {
        // `s[i]` opens line 2 at column 12 (`    return ` is 11 characters). get(a, 1) prints 4.
        name: "bounds_site",
        source: "fn get(s: []i32, i: i64) -> i32 {
    return s[i];
}

fn main() -> i32 {
    let a: [i32; 2] = [3, 4];
    print(get(a, 1));
    print(get(a, 2));
    print(99);
    return 0;
}
",
        stdout: "4\n",
        stderr:
            "error: index out of bounds: the len is 2 but the index is 2 at bounds_site.wk:2:12",
        needs_runtime: false,
    },
    Case {
        // `s[2..end]` with end = 9 over a 6-byte `str`: line 5 is `    println(s[..2]); println(`
        // (29 characters) then `s[2..end]`, so the site is column 30. `s[..2]` prints "wu" first.
        name: "range_end_site",
        source: "fn main() -> i32 {
    let s: str = \"wukong\";
    let mut end: i64 = 0;
    for _ in 0..9 { end = end + 1; }
    println(s[..2]); println(s[2..end]);
    return 0;
}
",
        stdout: "wu\n",
        stderr: "error: range end index 9 out of range for slice of length 6 at \
                 range_end_site.wk:5:30",
        needs_runtime: true,
    },
    Case {
        // `s[lo..3]` with lo = 5: the start is past the end. `    println(` is 12 characters, so
        // the range index opens line 5 at column 13.
        name: "range_order_site",
        source: "fn main() -> i32 {
    let s: str = \"wukong\";
    let mut lo: i64 = 0;
    for _ in 0..5 { lo = lo + 1; }
    println(s[lo..3]);
    return 0;
}
",
        stdout: "",
        stderr: "error: slice index starts at 5 but ends at 3 at range_order_site.wk:5:13",
        needs_runtime: true,
    },
    Case {
        // Bytes "ok", 0xFF, "!": the longest valid prefix is 2 bytes. `str_from_utf8` opens line 4
        // at column 13 (`    let s = ` is 12 characters).
        name: "utf8_site",
        source: "fn main() -> i32 {
    let a: [u8; 4] = [111, 107, 0xFF, 33];
    let b: []u8 = a;
    let s = str_from_utf8(b);
    println(s);
    return 0;
}
",
        stdout: "",
        stderr: "error: str_from_utf8: invalid utf-8 sequence of 1 bytes from index 2 (the len \
                 is 4) at utf8_site.wk:4:13",
        needs_runtime: true,
    },
];

/// A scratch directory unique to this process and `tag`, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir()
            .join("wukong_runtime_sites")
            .join(format!("{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `(exit code, stdout, stderr)` of a finished command.
fn outcome(cmd: &mut Command) -> (Option<i32>, String, String) {
    let out = cmd.output().expect("spawn");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Check one path's outcome against `case`: exit 1, the stdout before the failure, and exactly the
/// one stderr line.
fn check(case: &Case, what: &str, got: (Option<i32>, String, String)) -> Option<String> {
    let (code, stdout, stderr) = got;
    let lines: Vec<&str> = stderr.lines().collect();
    if code == Some(1) && stdout.replace("\r\n", "\n") == case.stdout && lines == [case.stderr] {
        None
    } else {
        Some(format!(
            "{} [{what}]: exit {code:?}, stdout {stdout:?}, stderr {stderr:?}\n    wanted exit 1, \
             stdout {:?}, stderr {:?}",
            case.name, case.stdout, case.stderr
        ))
    }
}

/// Write each case into `dir` as `<name>.wk`.
fn write_cases(dir: &Path) {
    for case in CASES {
        std::fs::write(dir.join(format!("{}.wk", case.name)), case.source).unwrap();
    }
}

#[test]
fn every_trap_names_its_site_on_the_interpreter_and_the_jit() {
    let scratch = Scratch::new("run");
    write_cases(&scratch.0);
    let mut failures = Vec::new();
    for case in CASES {
        for backend in ["interp", "native"] {
            for opt in ["-O0", "-O2"] {
                let got = outcome(
                    Command::new(env!("CARGO_BIN_EXE_wukongc"))
                        .current_dir(&scratch.0)
                        .arg("--run")
                        .arg(format!("--backend={backend}"))
                        .arg(opt)
                        .arg(format!("{}.wk", case.name)),
                );
                failures.extend(check(case, &format!("{backend} {opt}"), got));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The link path `--emit=exe` takes here, probed as `exe.rs` does: the preferred rustc link when
/// `rustc` runs and the runtime rlib is reachable from the compiler binary, else the C fallback
/// when a C compiler runs, else none.
fn link_path() -> Option<&'static str> {
    let runs = |program: &str| {
        Command::new(program)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    };
    let rlib = Path::new(env!("CARGO_BIN_EXE_wukongc"))
        .parent()
        .and_then(wukong_driver::runtime_rlib_in)
        .is_some();
    if runs("rustc") && rlib {
        return Some("rustc");
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if runs(&cc) {
        return Some("cc");
    }
    None
}

#[test]
fn every_trap_names_its_site_in_a_linked_executable() {
    let Some(link) = link_path() else {
        eprintln!(
            "every_trap_names_its_site_in_a_linked_executable: neither rustc + the runtime rlib \
             nor a C compiler runs here, so no executable can be linked; skipped"
        );
        return;
    };
    let scratch = Scratch::new("exe");
    write_cases(&scratch.0);
    let mut failures = Vec::new();
    for case in CASES {
        if link == "cc" && case.needs_runtime {
            eprintln!(
                "{}: the C fallback cannot link a `wukong_*` runtime entry; skipped",
                case.name
            );
            continue;
        }
        let exe = scratch
            .0
            .join(format!("{}{}", case.name, std::env::consts::EXE_SUFFIX));
        let built = outcome(
            Command::new(env!("CARGO_BIN_EXE_wukongc"))
                .current_dir(&scratch.0)
                .arg("--emit=exe")
                .arg("-o")
                .arg(&exe)
                .arg(format!("{}.wk", case.name)),
        );
        if built.0 != Some(0) || !exe.exists() {
            failures.push(format!(
                "{}: --emit=exe ({link} link) failed: {}",
                case.name, built.2
            ));
            continue;
        }
        failures.extend(check(
            case,
            &format!("exe, {link} link"),
            outcome(Command::new(&exe).current_dir(&scratch.0)),
        ));
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
