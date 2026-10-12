//! Native-backend failures carry a source span (P11.8).
//!
//! Cranelift reports a failure at the MIR value it was lowering, and that value remembers the
//! source construct it came from (`wukong_mir::SourceLocs`), so `C0002` — a slot too large for the
//! native byte layout — points at the declaration, on the object path and the JIT path alike, in
//! the human and the JSON rendering. Before, these diagnostics named only the program.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fail")
        .join("native_layout_too_large.wk")
}

/// Run wukongc on the fixture with `args`; the compile must fail. Returns stderr.
fn stderr_of(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(args)
        .arg(fixture())
        .output()
        .expect("spawn wukongc");
    assert!(!out.status.success(), "{args:?}: expected a compile error");
    assert_eq!(out.status.code(), Some(1), "{args:?}: COMPILE_ERROR");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Where `-o` points: a failed compile must not write it.
fn temp_object(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "wukongc_native_diag_spans_{tag}_{}.o",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

/// The fixture's `let mut big: [f64; 600000000];` is line 11, column 5.
const AT_THE_LET: &str = r#""line":11,"col":5,"primary":true,"label":"laid out here""#;

fn assert_c0002_at_the_let(stderr: &str, what: &str) {
    let first = stderr.lines().next().expect("a diagnostic line");
    assert!(
        first.starts_with(r#"{"severity":"error","code":"C0002","message":"#),
        "{what}: the first diagnostic is not C0002:\n{stderr}"
    );
    assert!(
        first.contains(AT_THE_LET),
        "{what}: C0002's primary span is not the `let` at 11:5:\n{stderr}"
    );
}

/// (a) `--emit=obj`, the ahead-of-time path.
#[test]
fn object_path_reports_c0002_at_the_declaration() {
    let out = temp_object("json");
    let stderr = stderr_of(&["--emit=obj", "--error-format=json", "-o", &out]);
    assert_c0002_at_the_let(&stderr, "--emit=obj");
    assert!(!Path::new(&out).exists(), "a failed compile wrote {out}");
}

/// (b) `--run --backend=native`, the JIT path — also after the optimizer has run.
#[test]
fn jit_path_reports_c0002_at_the_declaration() {
    for level in ["-O0", "-O2"] {
        let stderr = stderr_of(&["--run", "--backend=native", "--error-format=json", level]);
        assert_c0002_at_the_let(&stderr, level);
    }
}

/// (c) The human rendering shows the location, a caret under the declaration, and still the note
/// naming the program.
#[test]
fn human_rendering_points_at_the_declaration() {
    let out = temp_object("human");
    let stderr = stderr_of(&["--emit=obj", "--color=never", "-o", &out]);
    assert!(
        stderr.starts_with("error[C0002]: "),
        "not a C0002 error:\n{stderr}"
    );
    assert!(
        stderr.contains("native_layout_too_large.wk:11:5"),
        "no `file:11:5` location:\n{stderr}"
    );
    assert!(
        stderr.contains("11 |     let mut big: [f64; 600000000];"),
        "the source line is not quoted:\n{stderr}"
    );
    assert!(
        stderr
            .lines()
            .any(|l| l.trim_end().ends_with("^ laid out here") && l.contains("|     ^")),
        "no caret under `let`:\n{stderr}"
    );
    assert!(
        stderr.contains("= note: while compiling `"),
        "the `while compiling` note is gone:\n{stderr}"
    );
}
