//! Formats every `.wk` file in the repository's corpus in-process.
//!
//! For each file: `format_source` returns `Ok` (or `Syntax`, which only `tests/fail` may), the
//! output is a fixed point (`format(out) == out`), and `tokenize_with_comments` agrees with
//! `tokenize` on the input. Any `Unsupported` fails the test, listing every offending file.

use std::path::{Path, PathBuf};
use wukong_fmt::{format_source, FmtError, FmtOptions};
use wukong_span::SourceId;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "wk") {
            out.push(p);
        }
    }
}

#[test]
fn every_corpus_file_formats_and_is_a_fixed_point() {
    // Deeply nested corpus programs recurse through the printer and the AST's `Debug`.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let root = root();
    let dirs = [
        "tests/run",
        "examples",
        "tests/imports",
        "tests/warn",
        "tests/fail",
        "tests/doc",
    ];
    let mut files = Vec::new();
    for d in dirs {
        collect(&root.join(d), &mut files);
    }
    assert!(
        files.len() > 500,
        "corpus not found under {}",
        root.display()
    );
    let opts = FmtOptions::default();
    let mut problems: Vec<String> = Vec::new();
    let (mut ok, mut syntax) = (0usize, 0usize);
    for f in &files {
        let rel = f
            .strip_prefix(&root)
            .unwrap_or(f)
            .display()
            .to_string()
            .replace('\\', "/");
        let src = std::fs::read_to_string(f).unwrap();
        let (t1, _) = wukong_lexer::tokenize(&src, SourceId(0));
        let (t2, _, _) = wukong_lexer::tokenize_with_comments(&src, SourceId(0));
        if t1 != t2 {
            problems.push(format!(
                "{rel}: tokenize_with_comments disagrees with tokenize"
            ));
        }
        match format_source(&src, SourceId(0), &opts) {
            Ok(out) => {
                ok += 1;
                match format_source(&out, SourceId(0), &opts) {
                    Ok(again) if again == out => {}
                    Ok(again) => {
                        let line = out
                            .lines()
                            .zip(again.lines())
                            .position(|(a, b)| a != b)
                            .unwrap_or(0);
                        problems.push(format!(
                            "{rel}: not idempotent at output line {}: `{}` became `{}`",
                            line + 1,
                            out.lines().nth(line).unwrap_or(""),
                            again.lines().nth(line).unwrap_or("")
                        ));
                    }
                    Err(e) => problems.push(format!("{rel}: formatted output is refused: {e}")),
                }
            }
            Err(FmtError::Syntax(d)) => {
                syntax += 1;
                if !rel.starts_with("tests/fail/") {
                    problems.push(format!(
                        "{rel}: syntax error outside tests/fail: {}",
                        d.first().map(|x| x.message.as_str()).unwrap_or("")
                    ));
                }
            }
            Err(FmtError::Unsupported(m)) => problems.push(format!("{rel}: Unsupported: {m}")),
        }
    }
    eprintln!(
        "formatted {ok} of {} corpus files ({syntax} syntax-error fixtures)",
        files.len()
    );
    assert!(
        problems.is_empty(),
        "{} corpus problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// Comments cannot change a program's AST, so a comment inserted at any token boundary must never
/// make a formattable file unformattable, and the result must still be a fixed point. For a
/// spread of corpus files, a block comment goes after every 7th token and a line comment after
/// every 23rd (at the end of that token's line), and the file is formatted at the default width
/// and at 40 columns.
#[test]
fn comments_at_token_boundaries_and_narrow_widths_never_refuse() {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_comment_stress)
        .unwrap()
        .join()
        .unwrap();
}

fn with_comments(src: &str) -> String {
    let (tokens, _) = wukong_lexer::tokenize(src, SourceId(0));
    let mut inserts: Vec<(usize, String)> = Vec::new();
    for (k, t) in tokens.iter().enumerate() {
        if t.kind == wukong_lexer::TokenKind::Eof {
            break;
        }
        let hi = t.span.hi as usize;
        if k % 7 == 3 {
            inserts.push((hi, format!(" /* c{k} */")));
        }
        if k % 23 == 11 {
            let eol = src[hi..].find('\n').map(|i| hi + i).unwrap_or(src.len());
            inserts.push((eol, format!(" // l{k}")));
        }
    }
    inserts.sort_by_key(|(at, _)| *at);
    let mut out = String::new();
    let mut last = 0;
    for (at, text) in inserts {
        out.push_str(&src[last..at]);
        out.push_str(&text);
        last = at;
    }
    out.push_str(&src[last..]);
    out
}

fn run_comment_stress() {
    let root = root();
    let mut files = Vec::new();
    collect(&root.join("examples"), &mut files);
    let mut run = Vec::new();
    collect(&root.join("tests/run"), &mut run);
    files.extend(run.into_iter().step_by(5));
    let mut problems = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
        let src = std::fs::read_to_string(f).unwrap();
        for (variant, text) in [("plain", src.clone()), ("commented", with_comments(&src))] {
            for width in [100, 40] {
                let opts = FmtOptions { max_width: width };
                match format_source(&text, SourceId(0), &opts) {
                    Ok(out) => match format_source(&out, SourceId(0), &opts) {
                        Ok(again) if again == out => {}
                        Ok(again) => {
                            let line = out.lines().zip(again.lines()).position(|(a, b)| a != b);
                            let line = line.unwrap_or(0);
                            problems.push(format!(
                                "{rel} ({variant}, width {width}): not idempotent at line {}: `{}` -> `{}`",
                                line + 1,
                                out.lines().nth(line).unwrap_or(""),
                                again.lines().nth(line).unwrap_or("")
                            ));
                        }
                        Err(e) => problems.push(format!(
                            "{rel} ({variant}, width {width}): output refused: {e}"
                        )),
                    },
                    Err(FmtError::Syntax(_)) => {
                        problems.push(format!("{rel} ({variant}): syntax error"))
                    }
                    Err(FmtError::Unsupported(m)) => problems.push(format!(
                        "{rel} ({variant}, width {width}): Unsupported: {m}"
                    )),
                }
            }
        }
    }
    eprintln!("comment stress: {} files", files.len());
    assert!(
        problems.is_empty(),
        "{} problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
}
