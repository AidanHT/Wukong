//! Rustc-style terminal rendering of [`Diagnostic`]s.

use crate::{Diagnostic, NoteKind};
use wukong_span::SourceMap;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const BLUE: &str = "\x1b[1;34m"; // gutter / frame color

/// Renders diagnostics to strings. `color` toggles ANSI escapes (off for tests and pipes).
pub struct Renderer {
    pub color: bool,
}

impl Renderer {
    pub fn new(color: bool) -> Renderer {
        Renderer { color }
    }

    fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("{code}{s}{RESET}")
        } else {
            s.to_string()
        }
    }

    /// Render a single diagnostic to a multi-line string (no trailing newline).
    pub fn render(&self, d: &Diagnostic, sm: &SourceMap) -> String {
        let mut out = String::new();

        // Header: `error[E0501]: message`
        let head = match d.code {
            Some(code) => format!("{}[{}]", d.severity.header(), code),
            None => d.severity.header().to_string(),
        };
        out.push_str(&self.paint(d.severity.ansi(), &head));
        out.push_str(&self.paint(BOLD, &format!(": {}", d.message)));

        // Source frames, one per label (sorted by position).
        let mut labels: Vec<&crate::Label> =
            d.labels.iter().filter(|l| !l.span.is_dummy()).collect();
        labels.sort_by_key(|l| (l.span.source.0, l.span.lo));

        // At most one multi-line label is drawn whole, with connectors (see `render_connector`).
        let connector = connector_label(d, &labels, sm);

        // Gutter width is driven by the largest line number we will print: every label's first
        // line, plus the connector label's last line, and room for `...` when its middle is elided.
        let max_line = labels
            .iter()
            .map(|l| sm.span_location(l.span).line)
            .max()
            .unwrap_or(0);
        let mut width = max_line.max(1).to_string().len();
        if let Some(c) = connector {
            let (first, last) = (
                sm.span_location(c.span).line,
                last_char_location(sm, c.span).line,
            );
            width = width.max(last.to_string().len());
            if elides(first, last) {
                width = width.max(ELLIPSIS.len());
            }
        }
        let pad = " ".repeat(width);
        let bar = self.paint(BLUE, "|");
        // With a connector label every echoed line and underline gets a 2-column margin, where the
        // connector's `/` and `|` run; without one the frame is exactly the single-line form.
        let margin = if connector.is_some() { "  " } else { "" };

        // `--> file:line:col` for the primary span.
        if let Some(span) = d.primary_span() {
            if !span.is_dummy() {
                let loc = sm.span_location(span);
                let arrow = self.paint(BLUE, "-->");
                out.push('\n');
                out.push_str(&format!(
                    "{pad}{arrow} {}:{}:{}",
                    sm.name(span.source),
                    loc.line,
                    loc.col
                ));
            }
        }

        // A label in a *different file* than the one currently shown gets a `::: file:line:col`
        // header first, so a secondary label in an imported file is never printed under the root
        // file's `-->` arrow with only its own line number (the long-standing cross-file bug: E0300
        // duplicate-across-imports showed the second definition's line with no hint it lived in
        // another file).
        if !labels.is_empty() {
            // The `-->` arrow established the primary span's file as the current context.
            let mut cur_source = d.primary_span().map(|s| s.source);
            out.push('\n');
            out.push_str(&format!("{pad} {bar}"));
            for l in &labels {
                if Some(l.span.source) != cur_source {
                    cur_source = Some(l.span.source);
                    let loc = sm.span_location(l.span);
                    let colons = self.paint(BLUE, ":::");
                    out.push('\n');
                    out.push_str(&format!(
                        "{pad} {colons} {}:{}:{}",
                        sm.name(l.span.source),
                        loc.line,
                        loc.col
                    ));
                    out.push('\n');
                    out.push_str(&format!("{pad} {bar}"));
                }
                if connector.is_some_and(|c| std::ptr::eq(c, *l)) {
                    self.render_connector(&mut out, l, sm, width, &bar);
                } else {
                    self.render_label(&mut out, l, sm, width, &bar, margin);
                }
            }
            out.push('\n');
            out.push_str(&format!("{pad} {bar}"));
        }

        // Trailing notes / helps: `= note: ...`
        for (kind, text) in &d.notes {
            let tag = match kind {
                NoteKind::Note => "note",
                NoteKind::Help => "help",
            };
            out.push('\n');
            let eq = self.paint(BLUE, "=");
            out.push_str(&format!("{pad} {eq} {}: {text}", self.paint(BOLD, tag)));
        }

        // Structured suggestions: a `help:` line carrying the message and, when the fix is a
        // concrete one-line replacement, the replacement text itself in backticks. The precise
        // span+replacement travels in the JSON stream (see `to_json`) for an editor to apply; the
        // terminal form stays compact and never tries to splice a multi-line edit into one line.
        for s in &d.suggestions {
            out.push('\n');
            let eq = self.paint(BLUE, "=");
            let tag = self.paint(BOLD, "help");
            let single_line = !s.replacement.contains('\n');
            if single_line && !s.replacement.is_empty() {
                out.push_str(&format!(
                    "{pad} {eq} {tag}: {}: `{}`",
                    s.message, s.replacement
                ));
            } else if single_line && s.replacement.is_empty() {
                // A deletion: there is no text to show, so name the action.
                out.push_str(&format!("{pad} {eq} {tag}: {} (remove it)", s.message));
            } else {
                out.push_str(&format!("{pad} {eq} {tag}: {}", s.message));
            }
        }

        out
    }

    /// One label as a single echoed line plus its underline. A multi-line label that does not get
    /// connectors is cut here: its underline runs to one column past the end of its first line.
    /// `margin` is empty unless another label of the diagnostic is drawn with connectors, in which
    /// case it is two spaces, so this frame's text stays aligned with the connector frame's.
    fn render_label(
        &self,
        out: &mut String,
        l: &crate::Label,
        sm: &SourceMap,
        width: usize,
        bar: &str,
        margin: &str,
    ) {
        let span = l.span;
        let lo = sm.span_location(span);
        let hi = sm.location(span.source, span.hi);
        let line_text = sm.line_text(span.source, lo.line);
        let line_chars = line_text.chars().count() as u32;

        // Number of caret characters: the *rendered* width of the span on the first line, at least
        // 1. Rendered width, not the column difference, so a tab inside the span is underlined
        // across the same number of columns it occupies in the line printed above.
        let caret_len = if hi.line == lo.line {
            rendered_width(line_text, (lo.col - 1) as usize, (hi.col - 1) as usize).max(1)
        } else {
            rendered_width(line_text, (lo.col - 1) as usize, line_chars as usize + 1).max(1)
        };

        let (mark, mark_color) = if l.primary {
            ("^", span_color(true))
        } else {
            ("-", span_color(false))
        };
        let underline = mark.repeat(caret_len);
        let spaces = " ".repeat(rendered_width(line_text, 0, (lo.col - 1) as usize));

        let num = format!("{:>width$}", lo.line, width = width);
        out.push('\n');
        out.push_str(&format!(
            "{} {bar} {margin}{}",
            self.paint(BLUE, &num),
            expand_tabs(line_text)
        ));
        out.push('\n');

        let msg = label_message(l);
        let underline_painted = self.paint(mark_color, &format!("{underline}{msg}"));
        out.push_str(&format!(
            "{} {bar} {margin}{spaces}{underline_painted}",
            " ".repeat(width)
        ));
    }

    /// One multi-line label drawn whole, rustc-style, as one contiguous frame:
    ///
    /// ```text
    /// 2 |       let y = add(        2 | /     match a {
    ///   |  _____________^           3 | |         0 => 1,
    /// 3 | |         1,              4 | |     }
    /// 4 | |         2);               | |_____^ msg
    ///   | |__________^ msg
    /// ```
    ///
    /// The first line opens with `/` in the margin when only whitespace precedes the span, and
    /// otherwise with a start-marker line of `_` running to a mark under the span's first char.
    /// The lines after it carry `|` in the margin, and an end-marker line of `_` runs to a mark
    /// under the span's *last* char (see [`last_char_location`]), followed by the label's message.
    /// More than [`MAX_CONNECTOR_LINES`] lines keep the first two and the last two around one
    /// `...` line. Other labels are not interleaved into the frame; each keeps its own.
    fn render_connector(
        &self,
        out: &mut String,
        l: &crate::Label,
        sm: &SourceMap,
        width: usize,
        bar: &str,
    ) {
        let source = l.span.source;
        let lo = sm.span_location(l.span);
        let last = last_char_location(sm, l.span);
        let (first_line, last_line) = (lo.line, last.line);
        let pad = " ".repeat(width);
        let mark = if l.primary { "^" } else { "-" };
        let color = span_color(l.primary);
        let link = self.paint(color, "|");
        let num = |n: u32| self.paint(BLUE, &format!("{n:>width$}"));

        // The first line: `/` when the span opens the line's text, else a start-marker line.
        let first_text = sm.line_text(source, first_line);
        let before = (lo.col - 1) as usize;
        let opens_line = first_text.chars().take(before).all(char::is_whitespace);
        out.push('\n');
        if opens_line {
            out.push_str(&format!(
                "{} {bar} {} {}",
                num(first_line),
                self.paint(color, "/"),
                expand_tabs(first_text)
            ));
        } else {
            out.push_str(&format!(
                "{} {bar}   {}",
                num(first_line),
                expand_tabs(first_text)
            ));
            let s = rendered_width(first_text, 0, before);
            out.push('\n');
            out.push_str(&format!(
                "{pad} {bar}  {}",
                self.paint(color, &format!("{}{mark}", "_".repeat(s + 1)))
            ));
        }

        // The lines after it, the middle elided past `MAX_CONNECTOR_LINES` (`None` is the `...`).
        let rest: Vec<Option<u32>> = if elides(first_line, last_line) {
            vec![
                Some(first_line + 1),
                None,
                Some(last_line - 1),
                Some(last_line),
            ]
        } else {
            (first_line + 1..=last_line).map(Some).collect()
        };
        for line in rest {
            out.push('\n');
            match line {
                Some(n) => out.push_str(&format!(
                    "{} {bar} {link} {}",
                    num(n),
                    expand_tabs(sm.line_text(source, n))
                )),
                None => out.push_str(&format!(
                    "{} {bar} {link}",
                    self.paint(BLUE, &format!("{ELLIPSIS:<width$}"))
                )),
            }
        }

        // The end marker, under the span's last char.
        let e = rendered_width(sm.line_text(source, last_line), 0, (last.col - 1) as usize);
        let msg = label_message(l);
        out.push('\n');
        out.push_str(&format!(
            "{pad} {bar} {}",
            self.paint(color, &format!("|{}{mark}{msg}", "_".repeat(e + 1)))
        ));
    }
}

/// The most lines a connector frame prints in full; a longer label keeps its first two and last two
/// lines around one [`ELLIPSIS`] line.
const MAX_CONNECTOR_LINES: u32 = 6;

/// The gutter text of the line that stands for a connector frame's elided middle.
const ELLIPSIS: &str = "...";

/// Whether a connector frame over lines `first..=last` elides its middle.
fn elides(first: u32, last: u32) -> bool {
    last - first + 1 > MAX_CONNECTOR_LINES
}

/// `" message"`, or empty for a label without one.
fn label_message(l: &crate::Label) -> String {
    if l.message.is_empty() {
        String::new()
    } else {
        format!(" {}", l.message)
    }
}

/// The location of a span's *last* character: the char just before its exclusive end `hi`, or
/// `lo` for an empty span. A span whose `hi` is column 1 of the next line (one that includes the
/// line terminator) therefore ends on the line before. Steps back to a char boundary, so a span
/// ending in a multi-byte character is located at that character's first byte.
fn last_char_location(sm: &SourceMap, span: wukong_span::Span) -> wukong_span::Location {
    let src = sm.source(span.source);
    let mut end = span.hi.saturating_sub(1).max(span.lo);
    while end > span.lo && !src.is_char_boundary(end as usize) {
        end -= 1;
    }
    sm.location(span.source, end)
}

/// The label drawn with connectors, if any: the primary label when it spans several lines, else the
/// first multi-line label in `labels`' (source, lo) order. A label is multi-line when its last
/// character is on a later line than its first. Every other multi-line label is cut at the end of
/// its first line, as is every label of a diagnostic that has no connector label.
fn connector_label<'a>(
    d: &Diagnostic,
    labels: &[&'a crate::Label],
    sm: &SourceMap,
) -> Option<&'a crate::Label> {
    let multi_line =
        |l: &crate::Label| last_char_location(sm, l.span).line > sm.span_location(l.span).line;
    let primary = d.labels.iter().find(|l| l.primary && !l.span.is_dummy());
    if let Some(p) = primary.filter(|p| multi_line(p)) {
        return labels.iter().copied().find(|l| std::ptr::eq(*l, p));
    }
    labels.iter().copied().find(|l| multi_line(l))
}

/// How many columns a tab occupies when a source line is rendered.
///
/// The line and the caret padding beneath it must agree on this. Emitting a raw `\t` would let the
/// reader's terminal pick the width instead, which the padding cannot know — so both sides expand
/// tabs here, as rustc does.
const TAB_WIDTH: usize = 4;

/// Rendered width of the chars of `s` in the 0-based char range `[start, end)`, counting a tab as
/// [`TAB_WIDTH`] columns.
///
/// Columns past the end of `s` count one each, so a span that reaches the stripped line terminator
/// pads exactly as it did before tabs were expanded.
fn rendered_width(s: &str, start: usize, end: usize) -> usize {
    let mut width = 0usize;
    let mut i = 0usize;
    for c in s.chars() {
        if i >= end {
            break;
        }
        if i >= start {
            width += if c == '\t' {
                TAB_WIDTH
            } else {
                char_display_width(c)
            };
        }
        i += 1;
    }
    width + end.saturating_sub(i.max(start))
}

/// Terminal cell width of a single character: 2 for East-Asian wide and fullwidth forms (and the
/// common wide emoji blocks), 1 otherwise. A tab is handled separately by the caller.
///
/// This is a small dependency-free table — enough to keep carets aligned under CJK text or an emoji
/// in a comment or string, which is the drift the char-count column measure produced. It is not a
/// full Unicode width implementation; zero-width combining marks are counted as 1 (rare in source).
fn char_display_width(c: char) -> usize {
    let u = c as u32;
    let wide = matches!(u,
        0x1100..=0x115F       // Hangul Jamo
        | 0x2E80..=0x303E     // CJK radicals, Kangxi, CJK symbols/punctuation
        | 0x3041..=0x33FF     // Hiragana, Katakana, CJK compat
        | 0x3400..=0x4DBF     // CJK Ext A
        | 0x4E00..=0x9FFF     // CJK Unified Ideographs
        | 0xA000..=0xA4CF     // Yi
        | 0xAC00..=0xD7A3     // Hangul Syllables
        | 0xF900..=0xFAFF     // CJK Compatibility Ideographs
        | 0xFE30..=0xFE4F     // CJK Compatibility Forms
        | 0xFF00..=0xFF60     // Fullwidth Forms
        | 0xFFE0..=0xFFE6     // Fullwidth signs
        | 0x1F300..=0x1FAFF   // emoji & pictographs
        | 0x20000..=0x3FFFD   // CJK Ext B+
    );
    if wide {
        2
    } else {
        1
    }
}

/// `s` with each tab replaced by [`TAB_WIDTH`] spaces, so it lines up with [`rendered_width`].
fn expand_tabs(s: &str) -> String {
    if !s.contains('\t') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + TAB_WIDTH);
    for c in s.chars() {
        if c == '\t' {
            for _ in 0..TAB_WIDTH {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn span_color(primary: bool) -> &'static str {
    if primary {
        "\x1b[1;31m" // red for primary
    } else {
        "\x1b[1;34m" // blue for secondary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Diagnostic;
    use wukong_span::{SourceMap, Span};

    fn sm_with(src: &str) -> (SourceMap, wukong_span::SourceId) {
        let mut sm = SourceMap::new();
        let id = sm.add("test.wk", src);
        (sm, id)
    }

    #[test]
    fn renders_header_location_and_caret() {
        let (sm, id) = sm_with("fn main() {\n    matmul(a, b);\n}\n");
        // underline "matmul" on line 2 (cols 5..11)
        let span = Span::new(id, 16, 22);
        let d = Diagnostic::error("shape mismatch")
            .with_code("E0501")
            .primary(span, "K must equal P")
            .help("transpose b");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0501]: shape mismatch
 --> test.wk:2:5
  |
2 |     matmul(a, b);
  |     ^^^^^^ K must equal P
  |
  = help: transpose b";
        assert_eq!(r, expected);
    }

    #[test]
    fn tab_indented_line_aligns_with_its_caret() {
        // Two leading tabs. The rendered source line and the caret padding beneath it must agree
        // on how wide a tab is; printing a raw `\t` and padding with one space per column puts the
        // carets (tabstop - 1) columns left of the construct they are supposed to mark.
        let (sm, id) = sm_with("fn main() {\n\t\tmatmul(a, b);\n}\n");
        let span = Span::new(id, 14, 20); // "matmul" on line 2
        let d = Diagnostic::error("shape mismatch")
            .with_code("E0501")
            .primary(span, "K must equal P");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0501]: shape mismatch
 --> test.wk:2:3
  |
2 |         matmul(a, b);
  |         ^^^^^^ K must equal P
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn renders_without_labels() {
        let (sm, _id) = sm_with("x");
        let d = Diagnostic::error("could not find input file").note("check the path");
        let r = Renderer::new(false).render(&d, &sm);
        assert_eq!(
            r,
            "error: could not find input file\n  = note: check the path"
        );
    }

    #[test]
    fn secondary_label_in_another_file_gets_a_colon_header() {
        let mut sm = SourceMap::new();
        let a = sm.add("a.wk", "aaa\n");
        let b = sm.add("b.wk", "bbb\n");
        let d = Diagnostic::error("duplicate definition")
            .with_code("E0300")
            .primary(Span::new(a, 0, 3), "first")
            .secondary(Span::new(b, 0, 3), "second");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0300]: duplicate definition
 --> a.wk:1:1
  |
1 | aaa
  | ^^^ first
  ::: b.wk:1:1
  |
1 | bbb
  | --- second
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn caret_aligns_under_a_wide_character() {
        // 你 is an East-Asian wide character (2 terminal cells); the caret under the following `x`
        // must be padded by 2, not 1, or it drifts one column left.
        let (sm, id) = sm_with("你x\n");
        let span = Span::new(id, 3, 4); // the `x` (你 is 3 UTF-8 bytes)
        let d = Diagnostic::error("bad").primary(span, "here");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: bad
 --> test.wk:1:2
  |
1 | 你x
  |   ^ here
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn renders_a_suggestion_as_a_help_line() {
        use crate::Applicability;
        let (sm, id) = sm_with("fn main() {\n    let x = 0;\n    x = 1;\n}\n");
        let span = Span::new(id, 20, 21); // the `x` in `let x`
        let d = Diagnostic::error("cannot assign twice to immutable binding `x`")
            .with_code("E0304")
            .primary(Span::new(id, 31, 32), "cannot assign")
            .suggest_with(
                span,
                "consider making this binding mutable",
                "mut x",
                Applicability::MachineApplicable,
            );
        let r = Renderer::new(false).render(&d, &sm);
        assert!(
            r.ends_with("= help: consider making this binding mutable: `mut x`"),
            "{r}"
        );
    }

    #[test]
    fn color_adds_escapes() {
        let (sm, id) = sm_with("abc");
        let d = Diagnostic::error("e").primary(Span::new(id, 0, 1), "");
        let colored = Renderer::new(true).render(&d, &sm);
        assert!(colored.contains("\x1b["));
        let plain = Renderer::new(false).render(&d, &sm);
        assert!(!plain.contains("\x1b["));
    }

    /// The span from the first occurrence of `from` to the end of the first occurrence of `to` at
    /// or after it.
    fn span_between(src: &str, id: wukong_span::SourceId, from: &str, to: &str) -> Span {
        let lo = src.find(from).unwrap();
        let hi = lo + src[lo..].find(to).unwrap() + to.len();
        Span::new(id, lo as u32, hi as u32)
    }

    // The multi-line snapshots below are derived by hand from the connector rules on
    // `render_connector`: margin `"  "`, `"| "` or `"/ "` between `| ` and the text; a start
    // marker `{pad} |  ` + `_` x (s+1) + mark, with s the rendered width before the span; an end
    // marker `{pad} | |` + `_` x (e+1) + mark + message, with e the rendered width before the
    // span's last char. The mark sits in the column of the char it points at.

    #[test]
    fn multi_line_span_starting_mid_line_gets_a_start_marker() {
        let src = "fn main() {\n    let y = add(\n        1,\n        2);\n}\n";
        let (sm, id) = sm_with(src);
        // `add(` .. `)`: 'a' is col 13 of line 2 (s = 12), ')' col 10 of line 4 (e = 9).
        let span = span_between(src, id, "add(", ")");
        let d = Diagnostic::error("call spans three lines").primary(span, "this call");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: call spans three lines
 --> test.wk:2:13
  |
2 |       let y = add(
  |  _____________^
3 | |         1,
4 | |         2);
  | |__________^ this call
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn multi_line_span_opening_its_line_gets_a_slash() {
        let src = "fn main() {\n    match a {\n        0 => 1,\n    }\n}\n";
        let (sm, id) = sm_with(src);
        // `match` (only whitespace before it on line 2) .. the `}` at col 5 of line 4 (e = 4).
        let span = span_between(src, id, "match", "    }");
        let d = Diagnostic::error("non-exhaustive match")
            .with_code("E0405")
            .primary(span, "patterns not covered");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error[E0405]: non-exhaustive match
 --> test.wk:2:5
  |
2 | /     match a {
3 | |         0 => 1,
4 | |     }
  | |_____^ patterns not covered
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_ten_line_span_elides_its_middle_and_widens_the_gutter() {
        let src = "fn main() {\n    let v = [\n        1,\n        2,\n        3,\n        4,\n        5,\n        6,\n        7,\n        8,\n    ];\n}\n";
        let (sm, id) = sm_with(src);
        // `[` at col 13 of line 2 (s = 12) .. `]` at col 5 of line 11 (e = 4): ten lines, so lines
        // 2, 3, `...`, 10 and 11 are printed, and the gutter is 3 wide for the `...`.
        let span = span_between(src, id, "[", "]");
        let d = Diagnostic::error("long literal").primary(span, "array literal");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: long literal
   --> test.wk:2:13
    |
  2 |       let v = [
    |  _____________^
  3 | |         1,
... | |
 10 | |         8,
 11 | |     ];
    | |_____^ array literal
    |";
        assert_eq!(r, expected);
    }

    #[test]
    fn seven_lines_elide_and_six_do_not() {
        // The elision threshold: a 6-line label prints every line, a 7-line one elides lines 3..=5.
        let six = "a\nb\nc\nd\ne\nf\n";
        let (sm, id) = sm_with(six);
        let d = Diagnostic::error("six").primary(Span::new(id, 0, 11), "");
        let expected = "\
error: six
 --> test.wk:1:1
  |
1 | / a
2 | | b
3 | | c
4 | | d
5 | | e
6 | | f
  | |_^
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);

        let seven = "a\nb\nc\nd\ne\nf\ng\n";
        let (sm, id) = sm_with(seven);
        let d = Diagnostic::error("seven").primary(Span::new(id, 0, 13), "");
        let expected = "\
error: seven
   --> test.wk:1:1
    |
  1 | / a
  2 | | b
... | |
  6 | | f
  7 | | g
    | |_^
    |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);
    }

    #[test]
    fn single_line_labels_get_the_margin_and_are_not_interleaved() {
        let src = "fn main() {\n    let a = 1;\n    match a {\n        0 => 1,\n    }\n}\n";
        let (sm, id) = sm_with(src);
        let a = src.find("a = 1").unwrap() as u32; // col 9 of line 2
        let zero = src.find("0 =>").unwrap() as u32; // col 9 of line 4, inside the match
        let d = Diagnostic::error("non-exhaustive match")
            .with_code("E0405")
            .primary(span_between(src, id, "match", "    }"), "not exhaustive")
            .secondary(Span::new(id, a, a + 1), "defined here")
            .secondary(Span::new(id, zero, zero + 1), "only this value is covered");
        let r = Renderer::new(false).render(&d, &sm);
        // Sorted by position: the `a` label, the connector frame (lines 3-5), then the `0` label in a
        // frame of its own after it, not inside the connector frame.
        let expected = "\
error[E0405]: non-exhaustive match
 --> test.wk:3:5
  |
2 |       let a = 1;
  |           - defined here
3 | /     match a {
4 | |         0 => 1,
5 | |     }
  | |_____^ not exhaustive
4 |           0 => 1,
  |           - only this value is covered
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_secondary_multi_line_label_gets_connectors_with_dashes() {
        // The primary label is single-line, so the first multi-line label, a secondary, is drawn
        // with connectors, marked `-` like any secondary.
        let src = "fn main() {\n    let t = (1,\n             2);\n    t.5;\n}\n";
        let (sm, id) = sm_with(src);
        let five = src.find("5;").unwrap() as u32; // col 7 of line 4
        let d = Diagnostic::error("no field `5` on a 2-tuple")
            .primary(Span::new(id, five, five + 1), "no field 5")
            .secondary(span_between(src, id, "(1", ")"), "tuple defined here");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: no field `5` on a 2-tuple
 --> test.wk:4:7
  |
2 |       let t = (1,
  |  _____________-
3 | |              2);
  | |_______________- tuple defined here
4 |       t.5;
  |         ^ no field 5
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn tab_indented_multi_line_span_aligns_its_markers() {
        let src = "fn main() {\n\tlet y = add(\n\t\t1,\n\t);\n}\n";
        let (sm, id) = sm_with(src);
        // `add(` is col 10 of line 2, rendered at width 4 + 8 = 12; `)` is col 2 of line 4,
        // rendered at width 4. The header keeps the char column, 10.
        let span = span_between(src, id, "add(", ")");
        let d = Diagnostic::error("bad call").primary(span, "this call");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: bad call
 --> test.wk:2:10
  |
2 |       let y = add(
  |  _____________^
3 | |         1,
4 | |     );
  | |_____^ this call
  |";
        assert_eq!(r, expected);
    }

    #[test]
    fn a_span_ending_at_a_newline_ends_on_the_line_before() {
        // `hi` at column 1 of line 2: the last char is line 1's `\n`, so the label is single-line
        // and renders exactly as before connectors existed (cut, one column past the text).
        let (sm, id) = sm_with("abc\ndef\n");
        let d = Diagnostic::error("e").primary(Span::new(id, 0, 4), "here");
        let expected = "\
error: e
 --> test.wk:1:1
  |
1 | abc
  | ^^^^ here
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);

        // `hi` at column 1 of line 10: the label ends at line 9's `\n` (col 10, e = 9), so line 10
        // is neither printed nor counted in the gutter width.
        let src = "fn main() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;\n    let f = 6;\n    g(a,\n      b);\n}\n";
        let (sm, id) = sm_with(src);
        let lo = src.find("g(a").unwrap() as u32;
        let hi = src.rfind('}').unwrap() as u32;
        let d = Diagnostic::error("e").primary(Span::new(id, lo, hi), "ends at the newline");
        let expected = "\
error: e
 --> test.wk:8:5
  |
8 | /     g(a,
9 | |       b);
  | |__________^ ends at the newline
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);
    }

    #[test]
    fn a_second_multi_line_label_is_cut_and_the_primary_one_gets_connectors() {
        let src = "fn main() {\n    let a = 1;\n    let b = 2;\n    g(a,\n      b,\n      a);\n    let c = 3;\n    f(a,\n      b,\n      c);\n}\n";
        let (sm, id) = sm_with(src);
        // The secondary `g(..)` (lines 4-6) sorts first but is cut at the end of line 4 (5 dashes:
        // `g(a,` plus one column past it). The primary `f(..)` (lines 8-10) gets the connectors,
        // and its last line, 10, widens the gutter to 2 though no label starts past line 8.
        let d = Diagnostic::error("two multi-line labels")
            .secondary(span_between(src, id, "g(a", ")"), "cut to its first line")
            .primary(span_between(src, id, "f(a", ")"), "drawn with connectors");
        let r = Renderer::new(false).render(&d, &sm);
        let expected = "\
error: two multi-line labels
  --> test.wk:8:5
   |
 4 |       g(a,
   |       ----- cut to its first line
 8 | /     f(a,
 9 | |       b,
10 | |       c);
   | |________^ drawn with connectors
   |";
        assert_eq!(r, expected);
    }

    #[test]
    fn json_positions_of_a_multi_line_span_are_unchanged() {
        // Only the human form (and its `rendered` copy) draws connectors; `line`/`col` stay the
        // span's `lo` and `line_end`/`col_end` its exclusive `hi`, byte offsets untouched.
        let src = "fn main() {\n    let y = add(\n        1,\n        2);\n}\n";
        let (sm, id) = sm_with(src);
        let span = span_between(src, id, "add(", ")");
        assert_eq!((span.lo, span.hi), (24, 50));
        let d = Diagnostic::error("call spans three lines").primary(span, "this call");
        let j = crate::to_json(&d, &sm);
        assert!(
            j.contains(
                "\"line\":2,\"col\":13,\"primary\":true,\"label\":\"this call\",\
                 \"line_end\":4,\"col_end\":11,\"byte_start\":24,\"byte_end\":50}"
            ),
            "{j}"
        );
        assert!(j.contains("\\n  | |__________^ this call\\n"), "{j}");
    }

    #[test]
    fn header_column_is_the_char_column_json_reports() {
        // The `-->` column is the 1-based char column, the same number as JSON `col` — the
        // editor/LSP jump contract. Display width only places the caret: a tab (4 cells) and a
        // wide char (2 cells) put `x` at display column 7, but its header column is 3.
        let (sm, id) = sm_with("\t你x = 1;\n");
        let span = Span::new(id, 4, 5); // `x`: tab (1 byte) + 你 (3 bytes)
        let d = Diagnostic::error("bad").primary(span, "here");
        let expected = "\
error: bad
 --> test.wk:1:3
  |
1 |     你x = 1;
  |       ^ here
  |";
        assert_eq!(Renderer::new(false).render(&d, &sm), expected);
        assert!(crate::to_json(&d, &sm).contains("\"line\":1,\"col\":3,"));
    }

    #[test]
    fn connectors_take_the_label_color_and_the_gutter_stays_blue() {
        let src = "fn main() {\n    match a {\n        0 => 1,\n    }\n}\n";
        let (sm, id) = sm_with(src);
        let d = Diagnostic::error("e").primary(span_between(src, id, "match", "    }"), "m");
        let r = Renderer::new(true).render(&d, &sm);
        let (blue, red) = ("\x1b[1;34m", "\x1b[1;31m");
        assert!(
            r.contains(&format!(
                "{blue}2{RESET} {blue}|{RESET} {red}/{RESET}     match a {{"
            )),
            "{r:?}"
        );
        assert!(
            r.contains(&format!(
                "{blue}3{RESET} {blue}|{RESET} {red}|{RESET}         0 => 1,"
            )),
            "{r:?}"
        );
        assert!(
            r.contains(&format!("  {blue}|{RESET} {red}|_____^ m{RESET}")),
            "{r:?}"
        );
    }
}
