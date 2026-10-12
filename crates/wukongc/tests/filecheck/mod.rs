//! An in-tree FileCheck: match a program's output against `CHECK:` lines written in a check file.
//!
//! The semantics are LLVM FileCheck's, for the directives a compiler's filetests need:
//!
//! | directive | matches |
//! |---|---|
//! | `CHECK: p` | the first `p` at or after the end of the previous match (possibly on the same line) |
//! | `CHECK-NEXT: p` | a `p` on the line right after the previous match's line |
//! | `CHECK-SAME: p` | a `p` later on the previous match's line |
//! | `CHECK-EMPTY:` | the next line, which must be empty |
//! | `CHECK-NOT: p` | nothing: `p` must not occur between the matches of the directives around it (or the region's start / end) |
//! | `CHECK-DAG: p` | a run of consecutive `-DAG`s matches in any order, never two on overlapping text; a `-NOT` between two runs is checked between them |
//! | `CHECK-LABEL: p` | first, every label is found in order; they cut the input into regions and the directives between two labels must match inside theirs |
//! | `CHECK-COUNT-<n>: p` | `p`, `n` times in a row, each after the last |
//!
//! A pattern is literal text with three escapes: `{{re}}` (a regex, [`regex`]'s dialect; it ends at
//! the first `}}`, extended over a third `}` so `{{a{2}}}` works), `[[NAME:re]]` (match `re` and
//! bind its text to `NAME`) and `[[NAME]]` (the text `NAME` was bound to, literally — or, after a
//! binding in the same pattern, whatever that binding matched). `[[@LINE]]`, `[[@LINE+k]]` and
//! `[[@LINE-k]]` are the directive's own line number, offset. Bindings stay in force for the rest
//! of the check file; a later binding of the same name replaces the earlier one. A
//! `CHECK-LABEL` may neither bind nor use a variable (labels are found before anything else
//! matches), and a `CHECK-NOT` may not bind one.
//!
//! Whitespace: leading and trailing whitespace of a pattern is ignored, and a run of blanks in its
//! literal text matches any non-empty run of blanks in the input (`strict_whitespace` turns that
//! off). Whitespace inside `{{..}}` is the regex's own. A pattern never matches across a line
//! break.
//!
//! A directive is a prefix (`CHECK`, or each of `Options::prefixes`) followed by `:` or one of the
//! suffixes above, anywhere in a line, not preceded by a word character or `-`; a line holds at
//! most one. A comment whose first word starts with a prefix but is no directive (`CHECK-NXT:`,
//! `CHECK-NEXT` without its colon, `CHECK :`) is an error rather than a silently ignored line.
//!
//! A failed match reports the directive's `file:line:col` and the line it is on, where the scan
//! started in the input, the input line most like the pattern ("possible intended match"), and
//! the value of every variable the pattern used.

pub mod regex;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use regex::{Node, RegexError};

/// How a check file is read and matched.
#[derive(Clone, Debug)]
pub struct Options {
    /// The directive prefixes; `["CHECK"]` by default.
    pub prefixes: Vec<String>,
    /// Patterns that must not occur anywhere a positive directive did not match (FileCheck's
    /// `--implicit-check-not`): a `CHECK-NOT` before every directive and at the end.
    pub implicit_check_not: Vec<String>,
    /// Accept an empty input (otherwise an error, as an empty input is almost always a bug).
    pub allow_empty: bool,
    /// Match blanks in check text exactly instead of as "one or more blanks".
    pub strict_whitespace: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            prefixes: vec!["CHECK".to_string()],
            implicit_check_not: Vec::new(),
            allow_empty: false,
            strict_whitespace: false,
        }
    }
}

/// Match `input` against the directives of `check_text` (named `check_name` in messages).
/// `Err` is the full report: every check-file error, or every failed match (a failure ends its
/// label region; the other regions are still checked).
pub fn check(
    check_name: &str,
    check_text: &str,
    input: &str,
    opts: &Options,
) -> Result<(), String> {
    validate_prefixes(&opts.prefixes)?;
    let patterns = read_directives(check_name, check_text, opts)?;
    let strings = check_strings(check_name, check_text, patterns, opts)?;
    if input.is_empty() && !opts.allow_empty {
        return Err(
            "<stdin>: error: the input is empty (an empty output is a failure unless \
             --allow-empty is given)"
                .to_string(),
        );
    }
    let mut checker = Checker {
        input: Input::new(input),
        vars: HashMap::new(),
        strict_whitespace: opts.strict_whitespace,
    };
    let errors = checker.run(&strings);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn validate_prefixes(prefixes: &[String]) -> Result<(), String> {
    if prefixes.is_empty() {
        return Err("error: no check prefix given".to_string());
    }
    for (i, p) in prefixes.iter().enumerate() {
        let mut chars = p.chars();
        let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !valid {
            return Err(format!(
                "error: invalid check prefix `{p}`: a prefix starts with a letter and holds only \
                 letters, digits, `_` and `-`"
            ));
        }
        if prefixes[..i].contains(p) {
            return Err(format!("error: check prefix `{p}` is given twice"));
        }
    }
    Ok(())
}

/// What a directive asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Next,
    Same,
    Not,
    Dag,
    Label,
    Empty,
    Count(usize),
    /// The implicit end-of-input check that owns trailing `-NOT`s and `-DAG`s.
    Eof,
}

impl Kind {
    fn suffix(self) -> String {
        match self {
            Kind::Plain | Kind::Eof => String::new(),
            Kind::Next => "-NEXT".into(),
            Kind::Same => "-SAME".into(),
            Kind::Not => "-NOT".into(),
            Kind::Dag => "-DAG".into(),
            Kind::Label => "-LABEL".into(),
            Kind::Empty => "-EMPTY".into(),
            Kind::Count(n) => format!("-COUNT-{n}"),
        }
    }

    /// The suffix after a prefix, up to and including its `:`, and the byte length consumed.
    fn parse_suffix(rest: &str) -> Option<(Kind, usize)> {
        if rest.starts_with(':') {
            return Some((Kind::Plain, 1));
        }
        let rest = rest.strip_prefix('-')?;
        for (word, kind) in [
            ("NEXT:", Kind::Next),
            ("SAME:", Kind::Same),
            ("NOT:", Kind::Not),
            ("DAG:", Kind::Dag),
            ("LABEL:", Kind::Label),
            ("EMPTY:", Kind::Empty),
        ] {
            if rest.starts_with(word) {
                return Some((kind, 1 + word.len()));
            }
        }
        let count = rest.strip_prefix("COUNT-")?;
        let digits = count.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 || !count[digits..].starts_with(':') {
            return None;
        }
        let n = count[..digits].parse().ok()?;
        Some((Kind::Count(n), 1 + "COUNT-".len() + digits + 1))
    }
}

/// One piece of a pattern.
#[derive(Clone, Debug)]
enum Piece {
    Text(String),
    Regex(Node),
    Def {
        name: String,
        node: Node,
    },
    Use(String),
    /// `[[@LINE+k]]`: the directive's line plus `k`.
    Line(i64),
}

/// One directive, parsed.
#[derive(Clone, Debug)]
struct Pattern {
    kind: Kind,
    prefix: String,
    pieces: Vec<Piece>,
    /// Where the directive is: file name, 1-based line and column of the pattern, the line's text.
    file: String,
    line: usize,
    col: usize,
    source: String,
}

impl Pattern {
    fn label(&self) -> String {
        format!("{}{}", self.prefix, self.kind.suffix())
    }

    /// `file:line:col: error: PREFIX-KIND: msg`, the directive's line and a caret.
    fn error(&self, msg: &str) -> String {
        format!(
            "{}:{}:{}: error: {}: {msg}\n{}\n{}^",
            self.file,
            self.line,
            self.col,
            self.label(),
            self.source,
            pad_to(&self.source, self.col - 1)
        )
    }

    fn uses(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();
        for p in &self.pieces {
            if let Piece::Use(n) = p {
                if !names.contains(&n.as_str()) {
                    names.push(n);
                }
            }
        }
        names
    }
}

/// The whitespace of `line` up to char column `col`, with every other char a space: the padding
/// that puts a caret under column `col` whatever tabs the line holds.
fn pad_to(line: &str, col: usize) -> String {
    line.chars()
        .chain(std::iter::repeat(' '))
        .take(col)
        .map(|c| if c == '\t' { '\t' } else { ' ' })
        .collect()
}

/// The text after a line's comment marker (`//`, `;`, `#`), or `None` for a line that is not a
/// comment.
fn comment_body(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let body = t.trim_start_matches(['/', ';', '#']);
    (body.len() < t.len()).then(|| body.trim_start())
}

/// The first directive on `line`: (prefix index, kind, byte offset of the pattern text).
fn find_directive(line: &str, prefixes: &[String]) -> Option<(usize, Kind, usize)> {
    let mut by_len: Vec<usize> = (0..prefixes.len()).collect();
    by_len.sort_by_key(|&i| std::cmp::Reverse(prefixes[i].len()));
    for (pos, _) in line.char_indices() {
        let boundary = line[..pos]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'));
        if !boundary {
            continue;
        }
        for &pi in &by_len {
            let p = &prefixes[pi];
            if !line[pos..].starts_with(p.as_str()) {
                continue;
            }
            if let Some((kind, used)) = Kind::parse_suffix(&line[pos + p.len()..]) {
                return Some((pi, kind, pos + p.len() + used));
            }
        }
    }
    None
}

/// Every directive of `text` with one of `opts.prefixes`, in order, parsed.
fn read_directives(name: &str, text: &str, opts: &Options) -> Result<Vec<Pattern>, String> {
    let mut patterns = Vec::new();
    let mut errors = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let Some((pi, kind, at)) = find_directive(line, &opts.prefixes) else {
            // A comment that starts like a directive but is none is a typo, not prose.
            if let Some(body) = comment_body(line) {
                for p in &opts.prefixes {
                    let Some(rest) = body.strip_prefix(p.as_str()) else {
                        continue;
                    };
                    if rest.starts_with('-') || rest.trim_start().starts_with(':') {
                        let word: String = body
                            .chars()
                            .take_while(|c| !c.is_whitespace() && *c != ':')
                            .collect();
                        errors.push(format!(
                            "{name}:{lineno}: error: `{word}` looks like a `{p}` directive but is \
                             none: the directives are {p}:, {p}-NEXT:, {p}-SAME:, {p}-NOT:, \
                             {p}-DAG:, {p}-LABEL:, {p}-EMPTY: and {p}-COUNT-<n>:, with the colon \
                             right after the name\n{line}"
                        ));
                        break;
                    }
                }
            }
            continue;
        };
        let raw = &line[at..];
        let lead = raw.len() - raw.trim_start().len();
        let text = raw.trim();
        let col = line[..at + lead].chars().count() + 1;
        let mut pat = Pattern {
            kind,
            prefix: opts.prefixes[pi].clone(),
            pieces: Vec::new(),
            file: name.to_string(),
            line: lineno,
            col,
            source: line.to_string(),
        };
        match parse_pattern(text, lineno) {
            Ok(pieces) => pat.pieces = pieces,
            Err(msg) => {
                errors.push(pat.error(&msg));
                continue;
            }
        }
        if let Err(msg) = validate_pattern(&pat, text) {
            errors.push(pat.error(&msg));
            continue;
        }
        patterns.push(pat);
    }
    if errors.is_empty() {
        Ok(patterns)
    } else {
        Err(errors.join("\n"))
    }
}

fn validate_pattern(pat: &Pattern, text: &str) -> Result<(), String> {
    match pat.kind {
        Kind::Empty if !text.is_empty() => {
            return Err("takes no pattern: it matches an empty line".to_string())
        }
        Kind::Empty => {}
        _ if text.is_empty() => {
            return Err("found an empty pattern (use -EMPTY: to match an empty line)".to_string())
        }
        Kind::Count(0) => return Err("a count must be at least 1".to_string()),
        _ => {}
    }
    let defs: Vec<&str> = pat
        .pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Def { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    if pat.kind == Kind::Label && (!defs.is_empty() || !pat.uses().is_empty()) {
        return Err(
            "a label can neither define nor use a variable: labels are found before any other \
             directive matches"
                .to_string(),
        );
    }
    if pat.kind == Kind::Not && !defs.is_empty() {
        return Err("a -NOT pattern cannot define a variable".to_string());
    }
    for (i, d) in defs.iter().enumerate() {
        if defs[..i].contains(d) {
            return Err(format!("variable `{d}` is defined twice in one pattern"));
        }
    }
    Ok(())
}

/// Split a pattern's text into its pieces.
fn parse_pattern(text: &str, lineno: usize) -> Result<Vec<Piece>, String> {
    let mut pieces = Vec::new();
    let mut lit = String::new();
    let mut rest = text;
    let flush = |lit: &mut String, pieces: &mut Vec<Piece>| {
        if !lit.is_empty() {
            pieces.push(Piece::Text(std::mem::take(lit)));
        }
    };
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("{{") {
            flush(&mut lit, &mut pieces);
            let Some(mut end) = after.find("}}") else {
                return Err("a `{{` regex has no closing `}}`".to_string());
            };
            // `{{a{2}}}`: the regex's own `}` comes before the closing pair.
            while after[end + 2..].starts_with('}') {
                end += 1;
            }
            let src = &after[..end];
            if src.is_empty() {
                return Err("an empty `{{}}` regex".to_string());
            }
            let node = regex::parse(src).map_err(|RegexError(e)| e)?;
            pieces.push(Piece::Regex(node));
            rest = &after[end + 2..];
        } else if let Some(after) = rest.strip_prefix("[[") {
            flush(&mut lit, &mut pieces);
            let end = var_end(after)?;
            let inner = &after[..end];
            rest = &after[end + 2..];
            if let Some(pseudo) = inner.strip_prefix('@') {
                pieces.push(Piece::Line(parse_line_var(pseudo, lineno)?));
                continue;
            }
            match inner.split_once(':') {
                Some((name, src)) => {
                    check_var_name(name)?;
                    if src.is_empty() {
                        return Err(format!("variable `{name}` is defined by an empty regex"));
                    }
                    let node = regex::parse(src).map_err(|RegexError(e)| e)?;
                    pieces.push(Piece::Def {
                        name: name.to_string(),
                        node,
                    });
                }
                None => {
                    check_var_name(inner)?;
                    pieces.push(Piece::Use(inner.to_string()));
                }
            }
        } else {
            let c = rest.chars().next().unwrap();
            lit.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    flush(&mut lit, &mut pieces);
    Ok(pieces)
}

/// The offset of the `]]` closing a `[[`, skipping brackets the regex itself opens (LLVM's
/// `FindRegexVarEnd`): `[[X:[a-z]+]]` ends after the `+`.
fn var_end(s: &str) -> Result<usize, String> {
    let b = s.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while i < b.len() {
        if depth == 0 && b[i..].starts_with(b"]]") {
            return Ok(i);
        }
        match b[i] {
            b'\\' => i += 1,
            b'[' => depth += 1,
            b']' => {
                if depth == 0 {
                    return Err("a `]` closes no `[` inside `[[..]]`".to_string());
                }
                depth -= 1;
            }
            _ => {}
        }
        i += 1;
    }
    Err("a `[[` variable has no closing `]]`".to_string())
}

fn check_var_name(name: &str) -> Result<(), String> {
    let mut chars = name.chars();
    let ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid variable name `{name}`: a name is a letter or `_`, then letters, digits and `_`"
        ))
    }
}

/// `LINE`, `LINE+k` or `LINE-k` after the `@`: the line number it names.
fn parse_line_var(s: &str, lineno: usize) -> Result<i64, String> {
    let bad = || format!("unknown pseudo variable `@{s}`: only @LINE, @LINE+<n> and @LINE-<n>");
    let rest = s.strip_prefix("LINE").ok_or_else(bad)?;
    let offset: i64 = if rest.is_empty() {
        0
    } else if let Some(n) = rest.strip_prefix('+') {
        n.trim().parse().map_err(|_| bad())?
    } else if let Some(n) = rest.strip_prefix('-') {
        -n.trim().parse::<i64>().map_err(|_| bad())?
    } else {
        return Err(bad());
    };
    let line = lineno as i64 + offset;
    if line < 1 {
        return Err(format!("`@{s}` names line {line}, before the file starts"));
    }
    Ok(line)
}

/// A positive directive (or the end of input) with the `-DAG`s and `-NOT`s written before it.
#[derive(Debug)]
struct CheckString {
    pat: Pattern,
    dag_not: Vec<Pattern>,
}

fn check_strings(
    name: &str,
    text: &str,
    patterns: Vec<Pattern>,
    opts: &Options,
) -> Result<Vec<CheckString>, String> {
    if patterns.is_empty() {
        let list: Vec<String> = opts.prefixes.iter().map(|p| format!("`{p}:`")).collect();
        return Err(format!(
            "{name}: error: no check directives found with prefix {}",
            list.join(" or ")
        ));
    }
    let mut implicit = Vec::new();
    for (i, src) in opts.implicit_check_not.iter().enumerate() {
        let file = "<implicit-check-not>".to_string();
        let src = src.trim();
        let pieces = parse_pattern(src, 1).map_err(|e| format!("{file}:{}: error: {e}", i + 1))?;
        let pat = Pattern {
            kind: Kind::Not,
            prefix: "IMPLICIT-CHECK".to_string(),
            pieces,
            file,
            line: i + 1,
            col: 1,
            source: src.to_string(),
        };
        validate_pattern(&pat, src).map_err(|e| pat.error(&e))?;
        implicit.push(pat);
    }
    let mut strings: Vec<CheckString> = Vec::new();
    let mut pending: Vec<Pattern> = implicit.clone();
    for pat in patterns {
        match pat.kind {
            Kind::Dag | Kind::Not => pending.push(pat),
            Kind::Next | Kind::Same | Kind::Empty if strings.is_empty() => {
                return Err(pat.error(&format!(
                    "found `{}` with no positive directive before it to follow",
                    pat.label()
                )));
            }
            _ => {
                strings.push(CheckString {
                    pat,
                    dag_not: std::mem::replace(&mut pending, implicit.clone()),
                });
            }
        }
    }
    let lines = text.lines().count();
    strings.push(CheckString {
        pat: Pattern {
            kind: Kind::Eof,
            prefix: "end of input".to_string(),
            pieces: Vec::new(),
            file: name.to_string(),
            line: lines.max(1),
            col: 1,
            source: String::new(),
        },
        dag_not: pending,
    });
    Ok(strings)
}

/// One input line: its byte range (a `\r` before the `\n` excluded) and its chars, with each
/// char's byte offset (and the end's) so a regex over chars maps back to bytes.
struct Line {
    start: usize,
    end: usize,
    chars: Vec<char>,
    offs: Vec<usize>,
}

struct Input<'a> {
    text: &'a str,
    lines: Vec<Line>,
}

impl<'a> Input<'a> {
    fn new(text: &'a str) -> Input<'a> {
        let mut lines = Vec::new();
        let mut start = 0;
        for piece in text.split_inclusive('\n') {
            let body = piece.strip_suffix('\n').unwrap_or(piece);
            let body = body.strip_suffix('\r').unwrap_or(body);
            let mut chars = Vec::with_capacity(body.len());
            let mut offs = Vec::with_capacity(body.len() + 1);
            for (o, c) in body.char_indices() {
                chars.push(c);
                offs.push(start + o);
            }
            offs.push(start + body.len());
            lines.push(Line {
                start,
                end: start + body.len(),
                chars,
                offs,
            });
            start += piece.len();
        }
        Input { text, lines }
    }

    /// The index of the line holding byte `pos` (a position between lines belongs to the line
    /// before; the end of the text to the last line).
    fn line_of(&self, pos: usize) -> usize {
        self.lines
            .partition_point(|l| l.start <= pos)
            .saturating_sub(1)
    }

    /// `<stdin>:L:C: note: msg`, the line, and a caret (and `~`s to `end`) under `pos`.
    fn note(&self, pos: usize, end: usize, msg: &str) -> String {
        if self.lines.is_empty() {
            return format!("<stdin>:1:1: note: {msg}");
        }
        let li = self.line_of(pos);
        let l = &self.lines[li];
        let col = l.offs.partition_point(|&o| o < pos.min(l.end));
        let width = l
            .offs
            .partition_point(|&o| o < end.min(l.end))
            .saturating_sub(col);
        let text: String = l.chars.iter().collect();
        format!(
            "<stdin>:{}:{}: note: {msg}\n{text}\n{}^{}",
            li + 1,
            col + 1,
            pad_to(&text, col),
            "~".repeat(width.saturating_sub(1))
        )
    }

    /// The first match of `node` (with `slots` capture slots) wholly inside `[from, to)` of one
    /// line: its byte span and each slot's byte span.
    #[allow(clippy::type_complexity)]
    fn find(
        &self,
        node: &Node,
        slots: usize,
        from: usize,
        to: usize,
    ) -> Result<Option<(usize, usize, Vec<Option<(usize, usize)>>)>, String> {
        if self.lines.is_empty() {
            return Ok(None);
        }
        for l in &self.lines[self.line_of(from)..] {
            if l.start > to {
                break;
            }
            let lo = l.start.max(from);
            let hi = l.end.min(to);
            if lo > hi {
                continue;
            }
            let lo_c = l.offs.partition_point(|&o| o < lo);
            let hi_c = l.offs.partition_point(|&o| o < hi);
            let found = regex::find(node, slots, &l.chars[..hi_c], lo_c, hi == l.end)
                .map_err(|RegexError(e)| e)?;
            if let Some(f) = found {
                let caps = f
                    .caps
                    .iter()
                    .map(|c| c.map(|(a, b)| (l.offs[a], l.offs[b])))
                    .collect();
                return Ok(Some((l.offs[f.start], l.offs[f.end], caps)));
            }
        }
        Ok(None)
    }

    /// The input position most like `want` in `[from, to)`: per line, the best substring edit
    /// distance, plus a hundredth per line skipped (LLVM's "possible intended match" quality).
    fn fuzzy(&self, want: &str, from: usize, to: usize) -> Option<usize> {
        let want: Vec<char> = collapse_blanks(want).chars().collect();
        if want.is_empty() || self.lines.is_empty() {
            return None;
        }
        let first = self.line_of(from);
        let mut best: Option<(f64, usize)> = None;
        for (skipped, l) in self.lines[first..].iter().take(400).enumerate() {
            if l.start >= to && skipped > 0 {
                break;
            }
            let lo_c = l.offs.partition_point(|&o| o < from.max(l.start));
            let hi_c = l.offs.partition_point(|&o| o < to.min(l.end));
            if lo_c >= hi_c {
                continue;
            }
            let (dist, start) = substring_distance(&want, &l.chars[lo_c..hi_c]);
            let quality = dist as f64 + skipped as f64 / 100.0;
            if 2 * dist <= want.len() && best.is_none_or(|(q, _)| quality < q) {
                best = Some((quality, l.offs[lo_c + start]));
            }
        }
        best.map(|(_, pos)| pos)
    }
}

fn collapse_blanks(s: &str) -> String {
    let mut out = String::new();
    for word in s.split([' ', '\t']).filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// The least edit distance between `want` and any substring of `line`, and where that substring
/// starts (Sellers' approximate substring match: the line's prefix and suffix are free).
fn substring_distance(want: &[char], line: &[char]) -> (usize, usize) {
    // prev[j] / cur[j]: (distance, start) for want[..i] ending at line[..j].
    let mut prev: Vec<(usize, usize)> = (0..=line.len()).map(|j| (0, j)).collect();
    for (i, &w) in want.iter().enumerate() {
        let mut cur = vec![(i + 1, 0); line.len() + 1];
        for j in 1..=line.len() {
            let sub = (prev[j - 1].0 + usize::from(line[j - 1] != w), prev[j - 1].1);
            let del = (prev[j].0 + 1, prev[j].1);
            let ins = (cur[j - 1].0 + 1, cur[j - 1].1);
            cur[j] = sub.min(del).min(ins);
        }
        prev = cur;
    }
    prev.iter()
        .skip(1)
        .copied()
        .min()
        .unwrap_or((want.len(), 0))
}

/// The matching state: the input, and every variable bound so far.
struct Checker<'a> {
    input: Input<'a>,
    vars: HashMap<String, String>,
    strict_whitespace: bool,
}

type Span = (usize, usize);

impl Checker<'_> {
    fn run(&mut self, strings: &[CheckString]) -> Vec<String> {
        let mut errors = Vec::new();
        let mut buf_start = 0;
        let buf_end = self.input.text.len();
        let mut i = 0;
        while i < strings.len() {
            // Labels first: the next one, searched from the end of the previous one, bounds the
            // region every directive up to (and including) it must match in.
            let label = (i..strings.len()).find(|&j| strings[j].pat.kind == Kind::Label);
            let (region_end, upto) = match label {
                None => (buf_end, strings.len()),
                Some(j) => match self.positive(&strings[j].pat, buf_start, buf_end) {
                    Ok((_, end)) => (end, j + 1),
                    Err(e) => {
                        errors.push(e);
                        return errors;
                    }
                },
            };
            let mut pos = buf_start;
            for cs in &strings[i..upto] {
                match self.check_string(cs, pos, region_end) {
                    Ok((_, end)) => pos = end,
                    Err(e) => {
                        errors.push(e);
                        break;
                    }
                }
            }
            buf_start = region_end;
            i = upto;
        }
        errors
    }

    /// Build `pat`'s matcher against the current bindings: the node, its capture slot count, and
    /// which slot binds which variable.
    #[allow(clippy::type_complexity)]
    fn compile(&self, pat: &Pattern) -> Result<(Node, usize, Vec<(String, usize)>), String> {
        let mut nodes = Vec::new();
        let mut defs: Vec<(String, usize)> = Vec::new();
        for piece in &pat.pieces {
            match piece {
                Piece::Text(t) => {
                    if self.strict_whitespace {
                        nodes.push(Node::literal(t));
                        continue;
                    }
                    let mut blank_run = false;
                    for c in t.chars() {
                        if c == ' ' || c == '\t' {
                            if !blank_run {
                                nodes.push(Node::blanks());
                            }
                            blank_run = true;
                        } else {
                            nodes.push(Node::Char(c));
                            blank_run = false;
                        }
                    }
                }
                Piece::Regex(n) => nodes.push(Node::Group(None, Box::new(n.clone()))),
                Piece::Def { name, node } => {
                    let slot = defs.len();
                    defs.push((name.clone(), slot));
                    nodes.push(Node::Group(Some(slot), Box::new(node.clone())));
                }
                Piece::Use(name) => {
                    if let Some(&(_, slot)) = defs.iter().find(|(n, _)| n == name) {
                        nodes.push(Node::Backref(slot));
                    } else if let Some(v) = self.vars.get(name) {
                        nodes.push(Node::literal(v));
                    } else {
                        return Err(pat.error(&format!(
                            "uses undefined variable `{name}` (no earlier directive defined it \
                             with [[{name}:<regex>]])"
                        )));
                    }
                }
                Piece::Line(n) => nodes.push(Node::literal(&n.to_string())),
            }
        }
        let slots = defs.len();
        Ok((Node::Concat(nodes), slots, defs))
    }

    /// The pattern's text with variables substituted and regexes dropped: what the fuzzy search
    /// looks for.
    fn flat(&self, pat: &Pattern) -> String {
        let mut s = String::new();
        for piece in &pat.pieces {
            match piece {
                Piece::Text(t) => s.push_str(t),
                Piece::Use(n) => s.push_str(self.vars.get(n).map_or(n.as_str(), |v| v)),
                Piece::Line(n) => s.push_str(&n.to_string()),
                Piece::Regex(_) | Piece::Def { .. } => {}
            }
        }
        s
    }

    fn var_notes(&self, pat: &Pattern) -> String {
        let mut s = String::new();
        for name in pat.uses() {
            if let Some(v) = self.vars.get(name) {
                s.push_str(&format!("\nnote: with \"{name}\" equal to \"{v}\""));
            }
        }
        s
    }

    /// Find `pat` in `[from, to)` and bind its variables.
    fn find(&mut self, pat: &Pattern, from: usize, to: usize) -> Result<Option<Span>, String> {
        let (node, slots, defs) = self.compile(pat)?;
        let found = self
            .input
            .find(&node, slots, from, to)
            .map_err(|e| pat.error(&format!("bad pattern: {e}")))?;
        let Some((start, end, caps)) = found else {
            return Ok(None);
        };
        for (name, slot) in defs {
            let (a, b) = caps[slot].expect("a matched pattern's group matched");
            self.vars.insert(name, self.input.text[a..b].to_string());
        }
        Ok(Some((start, end)))
    }

    fn not_found(&self, pat: &Pattern, from: usize, to: usize, what: &str) -> String {
        let mut msg = pat.error(what);
        msg.push('\n');
        msg.push_str(&self.input.note(from, from, "scanning from here"));
        if let Some(at) = self.input.fuzzy(&self.flat(pat), from, to) {
            if at != from {
                msg.push('\n');
                msg.push_str(&self.input.note(at, at, "possible intended match here"));
            }
        }
        msg.push_str(&self.var_notes(pat));
        msg
    }

    /// Match a positive directive (or the end of input) in `[from, to)`.
    fn positive(&mut self, pat: &Pattern, from: usize, to: usize) -> Result<Span, String> {
        match pat.kind {
            Kind::Eof => Ok((to, to)),
            Kind::Empty => {
                let first = self.input.line_of(from);
                let hit = self.input.lines[first.min(self.input.lines.len())..]
                    .iter()
                    .take_while(|l| l.start <= to)
                    .find(|l| l.start > from && l.start == l.end);
                hit.map(|l| (l.start, l.start))
                    .ok_or_else(|| self.not_found(pat, from, to, "no empty line found in input"))
            }
            Kind::Count(n) => {
                let mut pos = from;
                let mut first = None;
                for k in 1..=n {
                    match self.find(pat, pos, to)? {
                        Some((s, e)) => {
                            first.get_or_insert(s);
                            pos = e;
                        }
                        None => {
                            return Err(self.not_found(
                                pat,
                                pos,
                                to,
                                &format!("expected string not found in input (match {k} of {n})"),
                            ))
                        }
                    }
                }
                Ok((first.expect("a count is at least 1"), pos))
            }
            _ => match self.find(pat, from, to)? {
                Some(span) => Ok(span),
                None => Err(self.not_found(pat, from, to, "expected string not found in input")),
            },
        }
    }

    fn check_string(&mut self, cs: &CheckString, start: usize, end: usize) -> Result<Span, String> {
        let mut nots: Vec<&Pattern> = Vec::new();
        let last = self.check_dag(&cs.dag_not, start, end, &mut nots)?;
        let (first, match_end) = self.positive(&cs.pat, last, end)?;
        let newlines = self.input.text[last..first].matches('\n').count();
        match cs.pat.kind {
            Kind::Next | Kind::Empty if newlines != 1 => {
                let what = if newlines == 0 {
                    "is on the same line as the previous match"
                } else {
                    "is not on the line after the previous match"
                };
                let mut msg = cs.pat.error(what);
                msg.push('\n');
                msg.push_str(&self.input.note(first, match_end, "the match is here"));
                msg.push('\n');
                msg.push_str(&self.input.note(last, last, "the previous match ended here"));
                if newlines > 1 {
                    let li = self.input.line_of(last) + 1;
                    if let Some(l) = self.input.lines.get(li) {
                        msg.push('\n');
                        msg.push_str(&self.input.note(
                            l.start,
                            l.start,
                            "the line after the previous match is here",
                        ));
                    }
                }
                return Err(msg);
            }
            Kind::Same if newlines != 0 => {
                let mut msg = cs
                    .pat
                    .error("is not on the same line as the previous match");
                msg.push('\n');
                msg.push_str(&self.input.note(first, match_end, "the match is here"));
                msg.push('\n');
                msg.push_str(&self.input.note(last, last, "the previous match ended here"));
                return Err(msg);
            }
            _ => {}
        }
        self.check_nots(&nots, last, first)?;
        Ok((first, match_end))
    }

    /// The `-NOT`s in `[from, to)`: none may match.
    fn check_nots(&mut self, nots: &[&Pattern], from: usize, to: usize) -> Result<(), String> {
        for pat in nots {
            if let Some((s, e)) = self.find(pat, from, to)? {
                let mut msg = pat.error("excluded string found in input");
                msg.push('\n');
                msg.push_str(&self.input.note(s, e, "found here"));
                msg.push_str(&self.var_notes(pat));
                return Err(msg);
            }
        }
        Ok(())
    }

    /// The `-DAG` runs (and the `-NOT`s between them) before a positive directive, from `start`.
    /// Returns where the positive directive's search starts: the end of the last run's furthest
    /// match. `-NOT`s after the last run are left in `nots` for the caller.
    fn check_dag<'p>(
        &mut self,
        dag_not: &'p [Pattern],
        start: usize,
        end: usize,
        nots: &mut Vec<&'p Pattern>,
    ) -> Result<usize, String> {
        let mut start = start;
        // The current run's matches, sorted and disjoint.
        let mut ranges: Vec<Span> = Vec::new();
        for (idx, pat) in dag_not.iter().enumerate() {
            if pat.kind == Kind::Not {
                nots.push(pat);
                continue;
            }
            let mut pos = start;
            loop {
                let Some((s, e)) = self.find(pat, pos, end)? else {
                    return Err(self.not_found(
                        pat,
                        start,
                        end,
                        "expected string not found in input",
                    ));
                };
                // The first earlier match that ends after this one starts: either this one lies
                // wholly before it (insert there) or the two overlap (search again after it).
                // `pos` strictly grows: an overlapped match ends after this one starts.
                match ranges.iter().position(|r| s < r.1) {
                    Some(at) if ranges[at].0 < e => pos = ranges[at].1,
                    Some(at) => {
                        ranges.insert(at, (s, e));
                        break;
                    }
                    None => {
                        ranges.push((s, e));
                        break;
                    }
                }
            }
            let run_ends = dag_not.get(idx + 1).is_none_or(|p| p.kind == Kind::Not);
            if run_ends {
                if !nots.is_empty() {
                    self.check_nots(nots, start, ranges[0].0)?;
                    nots.clear();
                }
                start = ranges.last().expect("a run has a match").1;
                ranges.clear();
            }
        }
        Ok(start)
    }
}
