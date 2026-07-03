//! Compile-fail tests: each `tests/fail/*.mer` program must be rejected with a specific stable
//! error code. We drive the real binary with `--error-format=json --emit=mir` (which runs sema)
//! and assert the expected `"code":"E…"` appears and the process exits non-zero.
//!
//! This is the end-to-end guard for Mercury's compile-time checks — most importantly the shape
//! checker (`E0501` rank, `E0502` dimension).

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

#[test]
fn compile_fail_suite() {
    let dir = fail_dir();
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "mer").unwrap_or(false))
        .collect();
    programs.sort();
    assert!(!programs.is_empty(), "no fixtures in {}", dir.display());

    for p in &programs {
        let src = std::fs::read_to_string(p).unwrap();
        let want = expected_code(&src);
        let out = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
            .arg("--error-format=json")
            .arg("--emit=mir")
            .arg(p)
            .output()
            .expect("spawn mercuryc");

        let name = p.file_name().unwrap().to_string_lossy();
        assert!(
            !out.status.success(),
            "{name}: expected a compile error but it succeeded"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(&format!("\"code\":\"{want}\"")),
            "{name}: expected error code {want}, got:\n{stderr}"
        );
    }
    eprintln!("checked {} compile-fail fixture(s)", programs.len());
}

/// A diagnostic raised INSIDE an imported file must carry that file's own path and position:
/// a multi-file program shares one `SourceMap` and every span keeps its own `SourceId`, so the
/// renderer must name the imported file, not the root. The fixture's root
/// (`import_type_error.mer`) is clean; its E0401 lives in `lib/badlib.mer` at line 6 (pinned by a
/// comment in that file).
#[test]
fn imported_file_diagnostics_carry_their_own_path() {
    let p = fail_dir().join("import_type_error.mer");
    let out = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
        .arg("--error-format=json")
        .arg("--emit=mir")
        .arg(&p)
        .output()
        .expect("spawn mercuryc");
    assert!(!out.status.success(), "expected a compile error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let line = stderr
        .lines()
        .find(|l| l.contains("\"code\":\"E0401\""))
        .unwrap_or_else(|| panic!("no E0401 diagnostic in stderr:\n{stderr}"));
    assert!(
        line.contains("badlib.mer"),
        "the E0401 span must point into the imported file (lib/badlib.mer):\n{line}"
    );
    assert!(
        line.contains("\"line\":6"),
        "the E0401 span must carry the imported file's own line (6):\n{line}"
    );
}
