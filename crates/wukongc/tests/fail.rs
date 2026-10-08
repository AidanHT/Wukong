//! Compile-fail tests: each `tests/fail/*.wk` program must be rejected with a specific stable
//! error code. We drive the real binary with `--error-format=json --emit=mir` (which runs sema)
//! and assert the first diagnostic is an error with the expected code and the process exits non-zero.
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

/// The stage a fixture compiles to: `mir` (sema + lowering) unless the fixture names a later one
/// with `// EMIT: <stage>`. A backend diagnostic exists only in the backend that imposes it - the
/// native layout limit `C0002` is raised by Cranelift, which `--emit=mir` never reaches.
fn emit_stage(path: &Path) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|src| {
            src.lines().find_map(|l| {
                l.trim_start()
                    .strip_prefix("// EMIT:")
                    .map(|s| s.trim().to_string())
            })
        })
        .unwrap_or_else(|| "mir".to_string())
}

/// Compile `path` with the JSON diagnostic renderer; returns (succeeded, stderr).
fn compile_json(path: &Path) -> (bool, String) {
    let stage = emit_stage(path);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    cmd.arg("--error-format=json")
        .arg(format!("--emit={stage}"))
        .arg(path);
    if stage == "obj" || stage == "exe" {
        // A fixture that unexpectedly compiles must not drop an artifact into the source tree.
        let stem = path.file_stem().unwrap().to_string_lossy();
        cmd.arg("-o").arg(
            std::env::temp_dir().join(format!("wukongc_fail_{}_{stem}.out", std::process::id())),
        );
    }
    let out = cmd.output().expect("spawn wukongc");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Read the first diagnostic's code using the renderer's pinned JSON-lines prefix.
/// This is deliberately not a search: an earlier unrelated error, a code-less error,
/// or non-diagnostic stderr must not be hidden by a matching later diagnostic.
/// `wukong_diag::to_json` fixes severity/code/message order and colon spacing.
fn first_error_code(stderr: &str) -> Option<&str> {
    let line = stderr.lines().next()?;
    let fields = line.strip_prefix("{\"severity\":\"error\",\"code\":\"")?;
    let (code, rest) = fields.split_once('"')?;
    (rest.starts_with(",\"message\":") && line.ends_with('}')).then_some(code)
}

#[test]
fn compile_fail_code_check_rejects_secondary_only_matches() {
    let wanted = r#"{"severity":"error","code":"E0502","message":"dimension mismatch","spans":[],"notes":[]}"#;
    let unrelated =
        r#"{"severity":"error","code":"E0401","message":"unrelated error","spans":[],"notes":[]}"#;
    assert_eq!(first_error_code(wanted), Some("E0502"));
    assert_eq!(
        first_error_code(&format!("{wanted}\n{unrelated}\n")),
        Some("E0502")
    );
    // The old contains check accepted this stream even though its first error changed.
    let secondary_only = format!("{unrelated}\n{wanted}\n");
    assert!(secondary_only.contains("\"code\":\"E0502\""));
    assert_eq!(first_error_code(&secondary_only), Some("E0401"));

    for prefix in [
        "",
        "compiler crashed\n",
        "\n",
        r#"{"severity":"error","message":"missing code","spans":[],"notes":[]}"#,
        r#"{"severity":"warning","code":"E0502","message":"warning","spans":[],"notes":[]}"#,
        r#"{"severity":"error","code":null,"message":"missing code","spans":[],"notes":[]}"#,
    ] {
        assert_eq!(first_error_code(prefix), None, "prefix: {prefix:?}");
        if !prefix.is_empty() {
            assert_ne!(
                first_error_code(&format!("{prefix}\n{wanted}")),
                Some("E0502"),
                "a later diagnostic hid this prefix: {prefix:?}"
            );
        }
    }
    let message_only = r#"{"severity":"error","code":"E0401","message":"text mentions \"code\":\"E0502\"","spans":[],"notes":[]}"#;
    assert_eq!(first_error_code(message_only), Some("E0401"));
    let similar =
        r#"{"severity":"error","code":"E05020","message":"different code","spans":[],"notes":[]}"#;
    assert_ne!(first_error_code(similar), Some("E0502"));
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
        assert_eq!(
            first_error_code(&stderr),
            Some(want.as_str()),
            "{name}: expected first diagnostic to be error {want}, got:\n{stderr}"
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
        assert_eq!(
            first_error_code(&stderr),
            Some(*code),
            "{code}: the probe source no longer starts with this error. Source:\n{src}\nGot:\n{stderr}"
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
/// exercised by *something* here — or, for a warning (`W` code), by a `tests/warn` fixture. A
/// catalogue entry nothing reaches is documentation for a diagnostic that may already have been
/// renumbered away.
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
    // Warnings compile, so a lint code is exercised by a `tests/warn` fixture instead, whose every
    // `EXPECT-WARN`/`EXPECT-ERROR` line `warn.rs` proves is emitted.
    let warn_dir = fail_dir().join("..").join("warn");
    for p in std::fs::read_dir(&warn_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", warn_dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
    {
        let src = std::fs::read_to_string(&p).unwrap();
        covered.extend(src.lines().filter_map(|l| {
            let l = l.trim_start();
            l.strip_prefix("// EXPECT-WARN:")
                .or_else(|| l.strip_prefix("// EXPECT-ERROR:"))
                .map(|c| c.trim().to_string())
        }));
    }
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

/// The shape diagnostics whose primary MESSAGE is pinned, not just their code: each entry is a
/// fixture, the code its headline must carry, and substrings that headline must contain. Naming the
/// disagreeing dimension is the whole contract of Phase 0.6 — a message that regresses to a bare
/// "expected 512, found 513" still carries `E0502`, so `compile_fail_suite` would stay green.
/// The channel asserted here is `--error-format=json`, the one an editor actually reads.
const NAMED_DIM_MESSAGES: &[(&str, &str, &[&str])] = &[
    (
        "wrong_k.wk",
        "E0502",
        &["dimension mismatch on `K` (axis 0): expected 512, found 513"],
    ),
    (
        "wrong_m.wk",
        "E0502",
        &["dimension mismatch on `M` (axis 0): expected 512, found 513"],
    ),
    // A rank mismatch has no single dim to name, so it names rank against rank plus both shapes.
    (
        "rank.wk",
        "E0501",
        &["tensor rank mismatch: expected rank 2 [M, N], found rank 3 [4, 4, 4]"],
    ),
    // `a[5, 5]` on a square tensor raises one diagnostic per axis; only the axis tells them apart.
    // The first diagnostic must be axis 0.
    (
        "tensor_index_oob.wk",
        "E0501",
        &["index 5 is out of bounds for axis 0 of length 2"],
    ),
];

/// The dimension a shape error is about must reach a tool through the JSON stream: the stable
/// `code` field on the first error (using the renderer's pinned field order) plus a
/// `message` field that names the dim and both extents.
#[test]
fn shape_diagnostics_name_the_dim_in_json() {
    for (fixture, code, needles) in NAMED_DIM_MESSAGES {
        let p = fail_dir().join(fixture);
        let (ok, stderr) = compile_json(&p);
        assert!(!ok, "{fixture}: expected a compile error");
        assert_eq!(
            first_error_code(&stderr),
            Some(*code),
            "{fixture}: {stderr}"
        );
        let line = stderr
            .lines()
            .next()
            .expect("the first error assertion established a diagnostic line");
        // One JSON object per line, still parseable as such.
        assert!(
            line.starts_with('{') && line.ends_with('}'),
            "{fixture}: the {code} diagnostic is not a single JSON object:\n{line}"
        );
        for needle in *needles {
            assert!(
                line.contains(&format!("\"message\":\"{needle}")),
                "{fixture}: the {code} headline must start with {needle:?}:\n{line}"
            );
        }
    }
    eprintln!(
        "checked {} pinned shape message(s)",
        NAMED_DIM_MESSAGES.len()
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
    assert_eq!(first_error_code(&stderr), Some("E0401"), "{stderr}");
    let line = stderr
        .lines()
        .next()
        .expect("the first error assertion established a diagnostic line");
    assert!(
        line.contains("badlib.wk"),
        "the E0401 span must point into the imported file (lib/badlib.wk):\n{line}"
    );
    assert!(
        line.contains("\"line\":6"),
        "the E0401 span must carry the imported file's own line (6):\n{line}"
    );
}

/// A resolution error must carry what an editor needs to fix it, in the stream an editor reads: the
/// primary label, and the structured did-you-mean replacement (`suggestions[].replacement`), not
/// only prose. Each entry is a fixture, its first code, and substrings of its first JSON line.
const RESOLUTION_JSON: &[(&str, &str, &[&str])] = &[
    (
        "unknown_fn_call.wk",
        "E0320",
        &[
            "\"message\":\"cannot find function `sqare` in this scope\"",
            "\"label\":\"not found in this scope\"",
            "\"replacement\":\"square\"",
        ],
    ),
    (
        "unknown_field.wk",
        "E0321",
        &[
            "\"message\":\"no field `z` on type `Point`\"",
            "the fields of `Point` are `x` and `y`",
        ],
    ),
    (
        "builtin_method_unknown.wk",
        "E0322",
        &["\"message\":\"no method named `dim` found for type `[]f32`\""],
    ),
    (
        "call_non_function.wk",
        "E0323",
        &[
            "\"message\":\"expected function, found `f32`\"",
            "end the previous one with `;`",
        ],
    ),
    (
        "struct_literal_unknown.wk",
        "E0324",
        &[
            "\"message\":\"cannot find struct `Pointt` in this scope\"",
            "\"replacement\":\"Point\"",
        ],
    ),
];

#[test]
fn resolution_errors_carry_labels_and_suggestions_in_json() {
    for (fixture, code, needles) in RESOLUTION_JSON {
        let p = fail_dir().join(fixture);
        let (ok, stderr) = compile_json(&p);
        assert!(!ok, "{fixture}: expected a compile error");
        assert_eq!(
            first_error_code(&stderr),
            Some(*code),
            "{fixture}: {stderr}"
        );
        let line = stderr.lines().next().unwrap_or_default();
        for needle in *needles {
            assert!(
                line.contains(needle),
                "{fixture}: the first {code} diagnostic must contain {needle:?}:\n{line}"
            );
        }
    }
}

/// A call's shape error must say where the disagreeing dimension came from, in the JSON an editor
/// reads: the primary label on the conflicting ARGUMENT, a secondary on the argument or turbofish
/// that fixed the dimension, and a secondary on the callee's parameter (`wukong_sema::provenance`).
/// The headline is pinned separately by `shape_diagnostics_name_the_dim_in_json`.
const PROVENANCE_LABELS: &[(&str, &[&str])] = &[
    (
        "wrong_k.wk",
        &[
            "\"primary\":true,\"label\":\"`K` (axis 0) is 513 in argument 2\"",
            "\"primary\":false,\"label\":\"`K` = 512 was fixed here, by the turbofish\"",
            "\"primary\":false,\"label\":\"`K` is declared in parameter `b` of `matmul`\"",
        ],
    ),
    (
        "matmul_dim.wk",
        &[
            "\"primary\":true,\"label\":\"`K` (axis 0) is 513 in argument 2\"",
            "\"primary\":false,\"label\":\"`K` = 512 was fixed here, by axis 1 of argument 1\"",
            "\"primary\":false,\"label\":\"`K` is declared in parameter `b` of `matmul`\"",
        ],
    ),
    (
        "arg_count.wk",
        &[
            "\"message\":\"function `add` takes 2 arguments but 3 were supplied\"",
            "\"primary\":false,\"label\":\"unexpected argument\"",
            "\"primary\":false,\"label\":\"`add` is defined here\"",
        ],
    ),
];

#[test]
fn call_shape_errors_carry_provenance_labels_in_json() {
    for (fixture, needles) in PROVENANCE_LABELS {
        let p = fail_dir().join(fixture);
        let (ok, stderr) = compile_json(&p);
        assert!(!ok, "{fixture}: expected a compile error");
        let line = stderr.lines().next().unwrap_or_default();
        for needle in *needles {
            assert!(
                line.contains(needle),
                "{fixture}: the first diagnostic must contain {needle:?}:\n{line}"
            );
        }
    }
}
