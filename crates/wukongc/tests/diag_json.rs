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

/// The source of a `tests/fail` fixture, for locating the bytes a suggestion must name.
fn fixture_src(fixture: &str) -> String {
    std::fs::read_to_string(fail_dir().join(fixture)).expect("read the fixture")
}

/// The first `suggestions` entry of one JSON line: `(message, replacement, applicability,
/// byte_start, byte_end)`, read through the renderer's pinned key order.
fn first_suggestion(line: &str) -> Option<(String, String, String, usize, usize)> {
    let rest = &line[line.find("\"suggestions\":[{\"message\":\"")? + 27..];
    let (message, rest) = rest.split_once("\",\"replacement\":\"")?;
    let (replacement, rest) = rest.split_once("\",\"applicability\":\"")?;
    let (applicability, rest) = rest.split_once('"')?;
    let rest = &rest[rest.find("\"byte_start\":")? + 13..];
    let (start, rest) = rest.split_once(",\"byte_end\":")?;
    let end: String = rest.chars().take_while(char::is_ascii_digit).collect();
    Some((
        message.to_string(),
        replacement.to_string(),
        applicability.to_string(),
        start.parse().ok()?,
        end.parse().ok()?,
    ))
}

/// N5: `E0304` carries its fix as data — a machine-applicable `mut ` inserted at the binding's
/// name — with a byte range that is exactly that zero-width position, and both carets labelled.
#[test]
fn e0304_carries_a_machine_applicable_mut_insertion() {
    let src = fixture_src("assign_immutable.wk");
    let line = first_json_line("assign_immutable.wk");
    assert!(
        line.starts_with("{\"severity\":\"error\",\"code\":\"E0304\""),
        "{line}"
    );
    let at = src.find("x: i32 = 5").expect("the `let`");
    assert_eq!(
        first_suggestion(&line),
        Some((
            "make the binding mutable".to_string(),
            "mut ".to_string(),
            "machine-applicable".to_string(),
            at,
            at
        )),
        "{line}"
    );
    assert!(
        line.contains("\"primary\":true,\"label\":\"cannot assign twice to an immutable binding\""),
        "{line}"
    );
    assert!(
        line.contains("\"primary\":false,\"label\":\"binding declared here without `mut`\""),
        "{line}"
    );
}

/// N5: a did-you-mean (`E0321`, a misspelled field) is a structured replacement of exactly the
/// misspelled name, marked maybe-incorrect: another field might be meant.
#[test]
fn e0321_carries_a_did_you_mean_replacement_of_the_name() {
    let src = fixture_src("unknown_field.wk");
    let line = first_json_line("unknown_field.wk");
    assert!(
        line.starts_with("{\"severity\":\"error\",\"code\":\"E0321\""),
        "{line}"
    );
    let at = src.find("return p.z").expect("the access") + "return p.".len();
    let (_, replacement, applicability, start, end) =
        first_suggestion(&line).unwrap_or_else(|| panic!("no suggestion:\n{line}"));
    assert_eq!((start, end), (at, at + 1), "{line}");
    assert_eq!(&src[start..end], "z");
    assert_eq!(replacement, "x", "{line}");
    assert_eq!(applicability, "maybe-incorrect", "{line}");
}

/// N5: an associated function of an IMPORTED type, called bare, is E0320 whose help names the
/// type as its module declares it, `Pt`. The loader gives the type the program-wide symbol
/// `lib.geom_assoc.Pt`, which no source can write, so the help must never show it and no edit
/// may be offered: whether this file may write `Pt::norm` depends on its imports, which sema
/// does not see.
#[test]
fn e0320_for_an_imported_type_never_names_the_loader_symbol() {
    let line = first_json_line("import_assoc_fn_bare.wk");
    assert!(
        line.starts_with("{\"severity\":\"error\",\"code\":\"E0320\""),
        "{line}"
    );
    assert!(!line.contains("lib.geom_assoc.Pt"), "{line}");
    assert!(line.contains("\"suggestions\":[]"), "{line}");
    assert!(
        line.contains("`Pt` (declared in module `lib.geom_assoc`) has a function named `norm`"),
        "{line}"
    );
    assert!(line.contains("`Pt::norm(..)`"), "{line}");
}
