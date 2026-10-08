//! End-to-end checks on the `--error-format=json` schema (v1).
//!
//! `fail.rs` pins the *prefix* (severity/code/message) that the compile-fail gate reads. This file
//! pins the v1 additions a tool actually needs — end positions, byte offsets, and the
//! `suggestions`/`children`/`rendered`/`version` fields — against the real binary, so a regression
//! in `wukong_diag::to_json` is caught even though no unit test drives the whole pipeline.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fail_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fail")
}

/// Compile a fixture with the JSON renderer, front end only, and return its first stderr line.
fn first_json_line(fixture: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--error-format=json")
        .arg("--emit=check")
        .arg(fail_dir().join(fixture))
        .output()
        .expect("spawn wukongc");
    assert!(!out.status.success(), "{fixture}: expected a compile error");
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .next()
        .expect("a diagnostic line")
        .to_string()
}

#[test]
fn json_v0_prefix_is_preserved() {
    // The compile-fail gate reads exactly this prefix; it must never move.
    let line = first_json_line("wrong_k.wk");
    assert!(
        line.starts_with("{\"severity\":\"error\",\"code\":\"E0502\",\"message\":\""),
        "v0 prefix changed:\n{line}"
    );
    assert!(line.ends_with('}'), "not a single JSON object:\n{line}");
}

#[test]
fn json_v1_fields_are_present() {
    let line = first_json_line("wrong_k.wk");
    // A real span → end position and byte range.
    assert!(line.contains("\"line_end\":"), "no line_end:\n{line}");
    assert!(line.contains("\"col_end\":"), "no col_end:\n{line}");
    assert!(line.contains("\"byte_start\":"), "no byte_start:\n{line}");
    assert!(line.contains("\"byte_end\":"), "no byte_end:\n{line}");
    // New top-level arrays/fields and the schema version (last field).
    assert!(
        line.contains("\"suggestions\":["),
        "no suggestions:\n{line}"
    );
    assert!(line.contains("\"children\":["), "no children:\n{line}");
    assert!(line.contains("\"rendered\":\""), "no rendered:\n{line}");
    assert!(
        line.ends_with(",\"version\":1}"),
        "no version trailer:\n{line}"
    );
}
