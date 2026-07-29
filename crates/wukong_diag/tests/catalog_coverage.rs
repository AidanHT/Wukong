//! U6: the error-code catalogue and the codes the compiler actually emits must not drift apart.
//!
//! The unit tests in `catalog.rs` only check an entry's shape (starts with `E`/`C`, non-empty
//! title, body longer than a line). Nothing looked at what the stages emit, so a stage could ship
//! a code with no explanation — the user hits `error[E0601]`, runs `wukongc --explain E0601`, and
//! is told the code is unknown — and the catalogue could keep an entry for a code no stage
//! produces, documenting a rule that does not exist. Both directions are gated here.
//!
//! The scan is textual: every `"Ennnn"` / `"Cnnnn"` string literal under `crates/`, excluding
//! `target/` build output and `wukong_diag` itself (whose code strings are the catalogue and its
//! own tests, not emissions). Codes are always written as literals at the `self.error(..)` /
//! `.with_code(..)` site, so this sees all of them. Note the reverse direction accepts a mention
//! from any source file, including a `#[cfg(test)]` module: it catches an entry with no reference
//! anywhere, not one whose only reference is a test.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Codes deliberately kept in the catalogue with no emitter. An empty list is the healthy state;
/// adding one must be a conscious, reviewable act.
const RESERVED: &[&str] = &[];

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("wukong_diag lives at <root>/crates/wukong_diag")
        .to_path_buf()
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("cannot read {}: {e}", dir.display()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name == "wukong_diag" {
                continue;
            }
            rust_sources(&path, out);
        } else if path.extension().map(|x| x == "rs").unwrap_or(false) {
            out.push(path);
        }
    }
}

/// Every `"Ennnn"` / `"Cnnnn"` string literal in `src`, paired with its 1-based line.
///
/// Comment lines are skipped. A code named in prose is *documentation*, not an emit site, and
/// counting it makes this gate fire on a file that merely explains the numbering — which is what
/// happened for `E0210`, quoted inside a doc comment in `wukongc/tests/fail.rs` that describes a
/// hypothetical renumbering. Scanning only non-comment lines keeps the gate's meaning ("a code the
/// compiler can actually emit") intact while letting docs cite codes freely.
fn codes_in(src: &str) -> Vec<(String, usize)> {
    let mut found = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }
        let b = line.as_bytes();
        for j in 0..b.len().saturating_sub(6) {
            if b[j] == b'"'
                && (b[j + 1] == b'E' || b[j + 1] == b'C')
                && b[j + 2..j + 6].iter().all(u8::is_ascii_digit)
                && b[j + 6] == b'"'
            {
                found.push((line[j + 1..j + 6].to_string(), i + 1));
            }
        }
    }
    found
}

/// (code -> first `path:line` that mentions it), over every scanned source file.
fn emitted_codes() -> BTreeMap<String, String> {
    let root = crates_dir();
    let mut files = Vec::new();
    rust_sources(&root, &mut files);
    assert!(
        files.len() > 10,
        "the source scan found only {} files under {} — the walk is broken, not the catalogue",
        files.len(),
        root.display()
    );

    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for path in &files {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let src = String::from_utf8_lossy(&bytes);
        let rel = path.strip_prefix(&root).unwrap_or(path).display().to_string();
        for (code, line) in codes_in(&src) {
            seen.entry(code).or_insert_with(|| format!("{rel}:{line}"));
        }
    }
    seen
}

#[test]
fn every_emitted_code_is_catalogued() {
    let catalogued: BTreeSet<&str> = wukong_diag::all_explanations()
        .iter()
        .map(|e| e.code)
        .collect();
    let missing: Vec<String> = emitted_codes()
        .into_iter()
        .filter(|(code, _)| !catalogued.contains(code.as_str()))
        .map(|(code, at)| format!("  {code} (first seen at {at})"))
        .collect();
    assert!(
        missing.is_empty(),
        "these codes are emitted but have no catalogue entry, so `wukongc --explain <CODE>` \
         reports them as unknown:\n{}\nAdd an `entry!(\"<CODE>\", \"<title>\", \"<body>\")` to \
         crates/wukong_diag/src/catalog.rs (U6). Keep existing codes at their current numbers.",
        missing.join("\n")
    );
}

#[test]
fn every_catalogued_code_has_an_emitter() {
    let emitted = emitted_codes();
    let dead: Vec<&str> = wukong_diag::all_explanations()
        .iter()
        .map(|e| e.code)
        .filter(|c| !emitted.contains_key(*c) && !RESERVED.contains(c))
        .collect();
    assert!(
        dead.is_empty(),
        "these codes are catalogued but no stage emits them, so `wukongc --explain <CODE>` \
         documents a rule the compiler cannot report: {dead:?}\nEither give the code an emitter, \
         drop its entry from crates/wukong_diag/src/catalog.rs (recording the number as retired so \
         it is never reused), or name it in RESERVED in this file."
    );
}
