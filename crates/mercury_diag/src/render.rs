//! Rustc-style terminal rendering of [`Diagnostic`]s.

use crate::{Diagnostic, NoteKind};
use mercury_span::SourceMap;

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

        // Gutter width is driven by the largest line number we will print.
        let max_line = d
            .labels
            .iter()
            .filter(|l| !l.span.is_dummy())
            .map(|l| sm.span_location(l.span).line)
            .max()
            .unwrap_or(0);
        let width = max_line.max(1).to_string().len();
        let pad = " ".repeat(width);
        let bar = self.paint(BLUE, "|");

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

        // Source frames, one per label (sorted by position).
        let mut labels: Vec<&crate::Label> =
            d.labels.iter().filter(|l| !l.span.is_dummy()).collect();
        labels.sort_by_key(|l| (l.span.source.0, l.span.lo));

        if !labels.is_empty() {
            out.push('\n');
            out.push_str(&format!("{pad} {bar}"));
            for l in &labels {
                self.render_label(&mut out, l, sm, width, &bar);
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

        out
    }

    fn render_label(
        &self,
        out: &mut String,
        l: &crate::Label,
        sm: &SourceMap,
        width: usize,
        bar: &str,
    ) {
        let span = l.span;
        let lo = sm.span_location(span);
        let hi = sm.location(span.source, span.hi);
        let line_text = sm.line_text(span.source, lo.line);
        let line_chars = line_text.chars().count() as u32;

        // Number of caret characters: span length on the first line, at least 1.
        let caret_len = if hi.line == lo.line {
            hi.col.saturating_sub(lo.col).max(1)
        } else {
            (line_chars + 1).saturating_sub(lo.col).max(1)
        };

        let (mark, mark_color) = if l.primary {
            ("^", span_color(true))
        } else {
            ("-", span_color(false))
        };
        let underline = mark.repeat(caret_len as usize);
        let spaces = " ".repeat((lo.col - 1) as usize);

        let num = format!("{:>width$}", lo.line, width = width);
        out.push('\n');
        out.push_str(&format!("{} {bar} {}", self.paint(BLUE, &num), line_text));
        out.push('\n');

        let msg = if l.message.is_empty() {
            String::new()
        } else {
            format!(" {}", l.message)
        };
        let underline_painted = self.paint(mark_color, &format!("{underline}{msg}"));
        out.push_str(&format!(
            "{} {bar} {spaces}{underline_painted}",
            " ".repeat(width)
        ));
    }
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
    use mercury_span::{SourceMap, Span};

    fn sm_with(src: &str) -> (SourceMap, mercury_span::SourceId) {
        let mut sm = SourceMap::new();
        let id = sm.add("test.mer", src);
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
 --> test.mer:2:5
  |
2 |     matmul(a, b);
  |     ^^^^^^ K must equal P
  |
  = help: transpose b";
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
    fn color_adds_escapes() {
        let (sm, id) = sm_with("abc");
        let d = Diagnostic::error("e").primary(Span::new(id, 0, 1), "");
        let colored = Renderer::new(true).render(&d, &sm);
        assert!(colored.contains("\x1b["));
        let plain = Renderer::new(false).render(&d, &sm);
        assert!(!plain.contains("\x1b["));
    }
}
