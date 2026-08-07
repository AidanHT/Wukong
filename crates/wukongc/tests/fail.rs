//! Compile-fail tests: each `tests/fail/*.wk` program must be rejected with a specific stable
//! error code. We drive the real binary with `--error-format=json --emit=mir` (which runs sema)
//! and assert the expected `"code":"E…"` appears and the process exits non-zero.
//!
//! This is the end-to-end guard for Wukong's compile-time checks — most importantly the shape
//! checker (`E0501` rank, `E0502` dimension).
//!
//! Two properties beyond "this fixture still fails" are enforced here, because nothing else in the
//! repo enforced them: every pinned code must exist in the diagnostic catalogue (so a code the
//! compiler emits can never be one `wukongc --explain` knows nothing about), and every catalogued
//! code must be REACHED by something in this file. The second one had a real hole: twelve of the
//! twenty-nine catalogued diagnostics — the lexer/parser codes that malformed and hostile input
//! actually hits — had no fixture at all, so renumbering or deleting one of them left the whole
//! suite green.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fail_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fail")
}

fn expected_code(src: &str) -> String {
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("// EXPECT-CODE:") {
            return rest.trim().to_string();
        }
    }
    panic!("fixture is missing an `// EXPECT-CODE:` directive");
}

/// Compile `path` with the JSON diagnostic renderer; returns (succeeded, stderr).
fn compile_json(path: &Path) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--error-format=json")
        .arg("--emit=mir")
        .arg(path)
        .output()
        .expect("spawn wukongc");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn compile_fail_suite() {
    let dir = fail_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no fixtures in {}", dir.display());

    for p in &programs {
        let src = std::fs::read_to_string(p).unwrap();
        let want = expected_code(&src);
        let (ok, stderr) = compile_json(p);

        let name = p.file_name().unwrap().to_string_lossy();
        assert!(!ok, "{name}: expected a compile error but it succeeded");
        assert!(
            stderr.contains(&format!("\"code\":\"{want}\"")),
            "{name}: expected error code {want}, got:\n{stderr}"
        );
        // A pinned code that is not in the catalogue is a code `wukongc --explain` cannot describe.
        assert!(
            wukong_driver::explain(&want).is_some(),
            "{name}: EXPECT-CODE {want} is not in the diagnostic catalogue \
             (crates/wukong_diag/src/catalog.rs), so `wukongc --explain {want}` returns nothing"
        );
    }
    eprintln!("checked {} compile-fail fixture(s)", programs.len());
}

/// The lexer and parser diagnostics that had no `tests/fail/` fixture. Each source is the minimal
/// spelling that reaches the code — verified against the real binary, not assumed: several
/// plausible-looking spellings reach a *different* code (an unterminated char literal with no
/// closing quote on the line reports `E0200`, not `E0104`; a bad layout suffix only reaches `E0204`
/// through the `, .name` position). They live inline rather than as `.wk` files because two of them
/// (`E0102`, `E0103`) are unterminated constructs that would corrupt any editor's view of the
/// directory, and because keeping the code next to the source makes the mapping auditable.
const LEXER_PARSER_PROBES: &[(&str, &str)] = &[
    (
        "E0102", // unterminated string literal
        "fn main() -> i32 {\n    let s: *u8 = \"oops;\n    return 0;\n}\n",
    ),
    (
        "E0103", // unterminated block comment
        "fn main() -> i32 {\n    /* never closed\n    return 0;\n}\n",
    ),
    (
        "E0104", // empty character literal
        "fn main() -> i32 {\n    let c: char = '';\n    return 0;\n}\n",
    ),
    (
        "E0200", // expected a specific token (the `)` that closes the group)
        "fn main() -> i32 {\n    let x: i32 = (1 + 2;\n    return x;\n}\n",
    ),
    (
        "E0201", // expected an identifier (the parameter name)
        "fn f( -> i32 { return 0; }\nfn main() -> i32 { return 0; }\n",
    ),
    (
        "E0203", // expected a dimension: a 26-digit extent does not decode
        "fn main() -> i32 {\n    let t: Tensor[f32, 99999999999999999999999999] = 0;\n    return 0;\n}\n",
    ),
    (
        "E0204", // unknown tensor layout
        "fn main() -> i32 {\n    let t: Tensor[f32, 4, 4, .bogus] = 0;\n    return 0;\n}\n",
    ),
    (
        "E0205", // malformed turbofish: `::<...>` not followed by a call
        "fn main() -> i32 {\n    let x: i32 = f::<3>;\n    return 0;\n}\n",
    ),
    (
        "E0206", // expected a pattern
        "fn main() -> i32 {\n    let x: i32 = 1;\n    match x {\n        * => { return 0; }\n    }\n    return 0;\n}\n",
    ),
    (
        "E0207", // expected an attribute argument
        "@tile(*)\nfn main() -> i32 { return 0; }\n",
    ),
    (
        "E0302", // invalid element type: a tensor element must be a scalar
        "enum Col { A, B }\nfn main() -> i32 {\n    let t: Tensor[Col, 4] = 0;\n    return 0;\n}\n",
    ),
];

/// Every catalogued lexer/parser diagnostic must still be reachable from real source. Without this,
/// renumbering `"E0205"` to `"E0210"` in the parser leaves the whole suite green while the compiler
/// starts emitting an uncatalogued code for a malformed turbofish.
#[test]
fn lexer_and_parser_diagnostics_are_reachable() {
    let dir = std::env::temp_dir()
        .join("wukongc_fail_probes")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&dir).expect("create scratch dir");

    for (code, src) in LEXER_PARSER_PROBES {
        let path = dir.join(format!("probe_{code}.wk"));
        std::fs::write(&path, src).expect("write probe source");
        let (ok, stderr) = compile_json(&path);
        assert!(
            !ok,
            "{code}: the probe source compiled successfully:\n{src}"
        );
        assert!(
            stderr.contains(&format!("\"code\":\"{code}\"")),
            "{code}: the probe source no longer reaches this diagnostic. Source:\n{src}\nGot:\n{stderr}"
        );
        assert!(
            wukong_driver::explain(code).is_some(),
            "{code} is not in the diagnostic catalogue (crates/wukong_diag/src/catalog.rs)"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "checked {} lexer/parser diagnostic probe(s)",
        LEXER_PARSER_PROBES.len()
    );
}

/// The reverse direction of the catalogue check: every code the catalogue advertises must be
/// exercised by *something* here. A catalogue entry nothing reaches is documentation for a
/// diagnostic that may already have been renumbered away.
#[test]
fn every_catalogued_code_is_exercised() {
    let dir = fail_dir();
    let mut covered: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .map(|p| expected_code(&std::fs::read_to_string(&p).unwrap()))
        .collect();
    covered.extend(LEXER_PARSER_PROBES.iter().map(|(c, _)| c.to_string()));
    covered.sort();
    covered.dedup();

    let catalogue = wukong_driver::all_explanations();
    // Without this floor the test passes vacuously if the catalogue is ever emptied or the
    // re-export stops resolving to it.
    assert!(
        catalogue.len() >= 20,
        "the diagnostic catalogue has only {} entries — this gate would pass vacuously",
        catalogue.len()
    );
    let missing: Vec<&str> = catalogue
        .iter()
        .map(|e| e.code)
        .filter(|c| !covered.iter().any(|k| k == c))
        .collect();
    assert!(
        missing.is_empty(),
        "{} catalogued diagnostic(s) have no compile-fail fixture and no probe, so nothing would \
         notice if the compiler stopped emitting them: {}",
        missing.len(),
        missing.join(", ")
    );
}

/// A diagnostic raised INSIDE an imported file must carry that file's own path and position:
/// a multi-file program shares one `SourceMap` and every span keeps its own `SourceId`, so the
/// renderer must name the imported file, not the root. The fixture's root
/// (`import_type_error.wk`) is clean; its E0401 lives in `lib/badlib.wk` at line 6 (pinned by a
/// comment in that file).
#[test]
fn imported_file_diagnostics_carry_their_own_path() {
    let p = fail_dir().join("import_type_error.wk");
    let (ok, stderr) = compile_json(&p);
    assert!(!ok, "expected a compile error");
    let line = stderr
        .lines()
        .find(|l| l.contains("\"code\":\"E0401\""))
        .unwrap_or_else(|| panic!("no E0401 diagnostic in stderr:\n{stderr}"));
    assert!(
        line.contains("badlib.wk"),
        "the E0401 span must point into the imported file (lib/badlib.wk):\n{line}"
    );
    assert!(
        line.contains("\"line\":6"),
        "the E0401 span must carry the imported file's own line (6):\n{line}"
    );
}
