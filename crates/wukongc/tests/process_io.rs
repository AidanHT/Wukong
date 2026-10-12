//! The process interface through the real binary, on the interpreter and the native JIT at `-O0`
//! and `-O2`: program arguments after `--` (`arg_count`, `arg`, `arg_i64`, `arg_f32`), standard
//! input (`read_stdin_u8`), an environment variable (`env`), `exit` from inside a nested call, a
//! program read from standard input (`wukongc -`), a program given inline (`wukongc -e`), and the
//! usage errors around them.
//!
//! `tests/run` cannot carry these: its harness runs every fixture with no program arguments and
//! no standard input (`tests/run/process_exit.wk` covers that case). Each expected line here is
//! derived from the rule the builtin documents, not recorded from a run.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// A fresh scratch file holding `src`, named `<tag>.wk` in a per-process directory.
fn program(tag: &str, src: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wukong-process-io-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let path = dir.join(format!("{tag}.wk"));
    std::fs::write(&path, src).expect("write program");
    path
}

/// Run `wukongc args…` with `stdin` piped in; (exit code, stdout, stderr).
fn wukongc(args: &[&str], stdin: &[u8]) -> (Option<i32>, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn wukongc");
    // Write, then drop the handle so the program sees end of input.
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(stdin)
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait for wukongc");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Every (backend, opt level) a run here must agree across.
const MODES: [(&str, &str); 4] = [
    ("--backend=interp", "-O0"),
    ("--backend=interp", "-O2"),
    ("--backend=native", "-O0"),
    ("--backend=native", "-O2"),
];

#[test]
fn program_arguments_reach_the_program() {
    let path = program(
        "args",
        "fn main() -> i32 {
    print(arg(0));
    print(arg_count());
    print(arg(1));
    print(arg_i64(2, 0));
    print(arg_f32(3, 0.0));
    print(arg(4));
    print(arg_i64(1, -1));
    print(arg(9));
    return 0;
}
",
    );
    let path_s = path.to_string_lossy().into_owned();
    for (backend, opt) in MODES {
        let (code, out, err) = wukongc(
            &[
                "--run", backend, opt, &path_s, "--", "alpha", "42", "3.5", "-x",
            ],
            b"",
        );
        assert_eq!(code, Some(0), "{backend} {opt}: {err}");
        let lines: Vec<&str> = out.lines().collect();
        // arg(0) is the program; then 5 = the program and four arguments; "alpha" is not an
        // integer, so arg_i64(1, -1) is its default; arg(9) is past the end, the empty string.
        assert!(
            lines.first().is_some_and(|l| l.ends_with("args.wk")),
            "{backend} {opt}: {out}"
        );
        assert_eq!(
            lines[1..],
            ["5", "alpha", "42", "3.5", "-x", "-1", ""],
            "{backend} {opt}"
        );
    }
}

#[test]
fn argument_and_environment_bytes_equal_the_same_bytes_in_a_literal() {
    // U+00E9 is C3 A9 in UTF-8, whether it arrives as program argument 1, as an environment
    // variable, or as a string literal, so each first byte is the one `u8` value 195 and the
    // byte-to-byte comparisons are all true. A `u8 == u8` compares the two 8-bit values directly
    // (no widening), so it sees how each byte is held: the interpreter held a literal's bytes
    // sign-extended (as every stored byte) but `arg`/`env` bytes zero-extended, and each
    // comparison was false there and true natively. Printed, the byte is 195 either way.
    let path = program(
        "bytes",
        "fn main() -> i32 {
    let a = arg(1);
    let e = env(\"WUKONG_PROCESS_IO_BYTES\");
    let lit = \"\u{e9}\";
    let s: str = \"\u{e9}\";
    println(*a == *lit);
    println(*e == *lit);
    println(*a == s[0]);
    println(*e == s[0]);
    println(*a == *e);
    println(*a);
    println(*e);
    return 0;
}
",
    );
    let path_s = path.to_string_lossy().into_owned();
    for (backend, opt) in MODES {
        let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .args(["--run", backend, opt, &path_s, "--", "\u{e9}"])
            .env("WUKONG_PROCESS_IO_BYTES", "\u{e9}")
            .stdin(Stdio::null())
            .output()
            .expect("run wukongc");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{backend} {opt}: {stderr}");
        assert_eq!(
            stdout, "true\ntrue\ntrue\ntrue\ntrue\n195\n195\n",
            "{backend} {opt}"
        );
    }
}

#[test]
fn standard_input_reaches_the_program() {
    let path = program(
        "stdin",
        "fn main() -> i32 {
    let mut buf: [u8; 64] = [0; 64];
    let n = read_stdin_u8(buf);
    print(n);
    write_stdout(buf, n);
    print_raw(\"|\\n\");
    let mut small: [u8; 4] = [0; 4];
    print(read_stdin_u8(small));
    return 0;
}
",
    );
    let path_s = path.to_string_lossy().into_owned();
    for (backend, opt) in MODES {
        let (code, out, err) = wukongc(&["--run", backend, opt, &path_s], b"hello, wukong");
        assert_eq!(code, Some(0), "{backend} {opt}: {err}");
        // 13 bytes fit the 64-byte buffer; the second read is at end of input.
        assert_eq!(out, "13\nhello, wukong|\n0\n", "{backend} {opt}");
    }
}

#[test]
fn exit_ends_the_run_from_a_nested_call() {
    let path = program(
        "exit",
        "fn check(v: i32) -> i32 {
    if v > 2 {
        print(\"too big\");
        exit(v * 2);
    }
    return v;
}

fn main() -> i32 {
    let mut total: i32 = 0;
    let mut i: i32 = 0;
    while i < 10 {
        total = total + check(i);
        print(total);
        i = i + 1;
    }
    return 100;
}
",
    );
    let path_s = path.to_string_lossy().into_owned();
    for (backend, opt) in MODES {
        let (code, out, err) = wukongc(&["--run", backend, opt, &path_s], b"");
        // check(0..=2) adds 0, 1, 2; check(3) prints and exits with 6 before returning.
        assert_eq!(code, Some(6), "{backend} {opt}: {err}");
        assert_eq!(out, "0\n1\n3\ntoo big\n", "{backend} {opt}");
        assert!(!err.contains("error:"), "{backend} {opt}: {err}");
    }
}

#[test]
fn a_program_reads_from_standard_input_or_the_command_line() {
    let src = b"fn main() -> i32 { printf(\"{} {}\\n\", arg_count(), 6 * 7); return 0; }";
    for (backend, opt) in MODES {
        let (code, out, err) = wukongc(&["--run", backend, opt, "-", "--", "x"], src);
        assert_eq!(
            (code, out.as_str()),
            (Some(0), "2 42\n"),
            "{backend} {opt}: {err}"
        );
        let inline = std::str::from_utf8(src).unwrap();
        let (code, out, err) = wukongc(&["--run", backend, opt, "-e", inline], b"");
        assert_eq!(
            (code, out.as_str()),
            (Some(0), "1 42\n"),
            "{backend} {opt}: {err}"
        );
    }
    // A diagnostic in a program from standard input names it `<stdin>`.
    let (code, _, err) = wukongc(&["--run", "-"], b"fn main() -> i32 { return x; }");
    assert_eq!(code, Some(1));
    assert!(err.contains("<stdin>"), "{err}");
}

#[test]
fn process_usage_errors_are_rejected() {
    let path = program("usage", "fn main() -> i32 { return 0; }\n");
    let path_s = path.to_string_lossy().into_owned();
    for args in [
        &[path_s.as_str(), "--", "a"][..],
        &[
            "--run",
            path_s.as_str(),
            "-e",
            "fn main() -> i32 { return 0; }",
        ][..],
        &["--run", "-e"][..],
    ] {
        let (code, _, err) = wukongc(args, b"");
        assert_eq!(code, Some(2), "{args:?}: {err}");
        assert!(err.contains("error:"), "{args:?}: {err}");
    }
}
