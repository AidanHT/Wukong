//! Machine-readable diagnostics: one JSON object per line (JSON Lines), emitted under
//! `--error-format=json`. Tooling (editors, CI) can parse these without scraping the terminal
//! renderer's output. Hand-written to keep `mercury_diag` dependency-free.

use crate::{Diagnostic, NoteKind, Severity};
use mercury_span::SourceMap;

fn severity_str(s: Severity) -> &'static str {
    match s {
        Severity::Bug => "bug",
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
        Severity::Help => "help",
    }
}

/// Escape a string for inclusion in a JSON double-quoted value.
fn esc(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn field_str(out: &mut String, key: &str, val: &str, leading_comma: bool) {
    if leading_comma {
        out.push(',');
    }
    esc(key, out);
    out.push(':');
    esc(val, out);
}

/// Render one diagnostic as a single-line JSON object (no trailing newline).
pub fn to_json(d: &Diagnostic, sm: &SourceMap) -> String {
    let mut out = String::with_capacity(128);
    out.push('{');
    field_str(&mut out, "severity", severity_str(d.severity), false);
    if let Some(code) = d.code {
        field_str(&mut out, "code", code, true);
    }
    field_str(&mut out, "message", &d.message, true);

    // Spans -> array of {file, line, col, primary, message}.
    out.push_str(",\"spans\":[");
    let mut first = true;
    for l in &d.labels {
        if l.span.is_dummy() {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        let loc = sm.span_location(l.span);
        out.push('{');
        field_str(&mut out, "file", sm.name(l.span.source), false);
        out.push_str(&format!(",\"line\":{},\"col\":{}", loc.line, loc.col));
        out.push_str(&format!(",\"primary\":{}", l.primary));
        field_str(&mut out, "label", &l.message, true);
        out.push('}');
    }
    out.push(']');

    // Notes/help.
    out.push_str(",\"notes\":[");
    for (i, (kind, msg)) in d.notes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        field_str(&mut out, "kind", if matches!(kind, NoteKind::Note) { "note" } else { "help" }, false);
        field_str(&mut out, "message", msg, true);
        out.push('}');
    }
    out.push(']');

    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Diagnostic;
    use mercury_span::{SourceMap, Span};

    #[test]
    fn emits_well_formed_json_line() {
        let mut sm = SourceMap::new();
        let id = sm.add("test.mer", "fn main() {}\n");
        let span = Span::new(id, 3, 7);
        let d = Diagnostic::error("type mismatch")
            .with_code("E0401")
            .primary(span, "here")
            .help("annotate the type");
        let j = to_json(&d, &sm);
        assert!(j.starts_with('{') && j.ends_with('}'));
        assert!(!j.contains('\n'));
        assert!(j.contains("\"severity\":\"error\""));
        assert!(j.contains("\"code\":\"E0401\""));
        assert!(j.contains("\"line\":1"));
        assert!(j.contains("\"primary\":true"));
        assert!(j.contains("\"kind\":\"help\""));
    }

    #[test]
    fn escapes_special_characters() {
        let mut s = String::new();
        esc("a\"b\\c\n", &mut s);
        assert_eq!(s, "\"a\\\"b\\\\c\\n\"");
    }
}
