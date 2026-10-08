//! Machine-readable diagnostics: one JSON object per line (JSON Lines), emitted under
//! `--error-format=json`. Tooling (editors, CI) can parse these without scraping the terminal
//! renderer's output. Hand-written to keep `wukong_diag` dependency-free.
//!
//! CONTRACT: `crates/wukongc/tests/fail.rs` requires the first diagnostic to be an error and pins its code using the
//! severity/code/message field order below. That order, `field_str`'s lack of whitespace
//! around `:`, and string-valued codes are load-bearing for the compile-fail gate.

use crate::{Applicability, Diagnostic, NoteKind, Severity};
use wukong_span::SourceMap;

fn severity_str(s: Severity) -> &'static str {
    match s {
        Severity::Bug => "bug",
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
        Severity::Help => "help",
    }
}

fn applicability_str(a: Applicability) -> &'static str {
    match a {
        Applicability::MachineApplicable => "machine-applicable",
        Applicability::MaybeIncorrect => "maybe-incorrect",
        Applicability::HasPlaceholders => "has-placeholders",
        Applicability::Unspecified => "unspecified",
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

    // Spans -> array of span objects. The first five keys (file, line, col, primary, label) are the
    // original v0 schema and MUST keep their order: fail.rs and editors pin the prefix. The end
    // position (line_end/col_end) and byte range (byte_start/byte_end) are APPENDED (schema v1), so
    // an editor can select or replace the exact range. `line`/`col` are the 1-based position of the
    // span's `lo`; `line_end`/`col_end` the 1-based position of its `hi` (exclusive end). Dummy
    // spans are skipped, so a diagnostic whose only labels are dummy emits an empty `spans` array.
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
        emit_span(&mut out, sm, l.span, Some(l.primary), &l.message);
    }
    out.push(']');

    // Notes/help (v0 schema: bare {kind, message}, no spans). Kept verbatim for back-compat.
    out.push_str(",\"notes\":[");
    for (i, (kind, msg)) in d.notes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        field_str(&mut out, "kind", note_kind_str(*kind), false);
        field_str(&mut out, "message", msg, true);
        out.push('}');
    }
    out.push(']');

    // v1: structured suggestions. Each is {message, replacement, applicability, span}, where `span`
    // carries the full byte range so a tool can apply the edit without re-parsing.
    out.push_str(",\"suggestions\":[");
    for (i, s) in d.suggestions.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        field_str(&mut out, "message", &s.message, false);
        field_str(&mut out, "replacement", &s.replacement, true);
        field_str(
            &mut out,
            "applicability",
            applicability_str(s.applicability),
            true,
        );
        out.push_str(",\"span\":");
        if s.span.is_dummy() {
            out.push_str("null");
        } else {
            emit_span(&mut out, sm, s.span, None, "");
        }
        out.push('}');
    }
    out.push(']');

    // v1: `children` are rustc-style sub-diagnostics (notes/helps promoted to their own severity),
    // the shape editors expect. They mirror `notes` for now, since notes carry no spans yet.
    out.push_str(",\"children\":[");
    for (i, (kind, msg)) in d.notes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        field_str(&mut out, "severity", note_kind_str(*kind), false);
        field_str(&mut out, "message", msg, true);
        out.push_str(",\"spans\":[]");
        out.push('}');
    }
    out.push(']');

    // v1: `rendered` is the human-readable form (color off), so a consumer can show exactly what the
    // terminal would without re-implementing the renderer. Newlines are escaped by `esc`, so the
    // whole diagnostic stays one JSON line.
    let rendered = crate::Renderer::new(false).render(d, sm);
    field_str(&mut out, "rendered", &rendered, true);

    // v1: schema version. Consumers key on this; it is the last field so the prefix never moves.
    out.push_str(",\"version\":1");

    out.push('}');
    out
}

fn note_kind_str(kind: NoteKind) -> &'static str {
    match kind {
        NoteKind::Note => "note",
        NoteKind::Help => "help",
    }
}

/// Emit one span object. When `primary` is `Some`, the v0 `primary`/`label` keys are included (for
/// a diagnostic label); a suggestion span passes `None` and omits them.
fn emit_span(
    out: &mut String,
    sm: &SourceMap,
    span: wukong_span::Span,
    primary: Option<bool>,
    label: &str,
) {
    let lo = sm.span_location(span);
    let hi = sm.location(span.source, span.hi);
    out.push('{');
    field_str(out, "file", sm.name(span.source), false);
    out.push_str(&format!(",\"line\":{},\"col\":{}", lo.line, lo.col));
    if let Some(primary) = primary {
        out.push_str(&format!(",\"primary\":{primary}"));
        field_str(out, "label", label, true);
    }
    out.push_str(&format!(
        ",\"line_end\":{},\"col_end\":{},\"byte_start\":{},\"byte_end\":{}",
        hi.line, hi.col, span.lo, span.hi
    ));
    out.push('}');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Diagnostic;
    use wukong_span::{SourceMap, Span};

    #[test]
    fn emits_well_formed_json_line() {
        let mut sm = SourceMap::new();
        let id = sm.add("test.wk", "fn main() {}\n");
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
        // v1 additions.
        assert!(j.contains("\"line_end\":1"));
        assert!(j.contains("\"byte_start\":3"));
        assert!(j.contains("\"byte_end\":7"));
        assert!(j.contains("\"children\":["));
        assert!(j.contains("\"rendered\":"));
        assert!(j.ends_with(",\"version\":1}"));
    }

    #[test]
    fn v0_field_prefix_is_unchanged() {
        // fail.rs pins this exact prefix (severity, code, message, spans). It must never move.
        let mut sm = SourceMap::new();
        let id = sm.add("t.wk", "ab\n");
        let d = Diagnostic::error("dimension mismatch")
            .with_code("E0502")
            .primary(Span::new(id, 0, 2), "here");
        let j = to_json(&d, &sm);
        assert!(j.starts_with(
            "{\"severity\":\"error\",\"code\":\"E0502\",\"message\":\"dimension mismatch\",\"spans\":[{\"file\":\"t.wk\",\"line\":1,\"col\":1,\"primary\":true,\"label\":\"here\""
        ));
    }

    #[test]
    fn emits_a_suggestion_with_its_span() {
        use crate::Applicability;
        let mut sm = SourceMap::new();
        let id = sm.add("t.wk", "let x = 1;\n");
        let d = Diagnostic::error("cannot assign")
            .with_code("E0304")
            .primary(Span::new(id, 4, 5), "")
            .suggest_with(
                Span::new(id, 4, 4),
                "make it mutable",
                "mut ",
                Applicability::MachineApplicable,
            );
        let j = to_json(&d, &sm);
        assert!(j.contains("\"suggestions\":[{\"message\":\"make it mutable\",\"replacement\":\"mut \",\"applicability\":\"machine-applicable\",\"span\":{"));
        assert!(j.contains("\"byte_start\":4,\"byte_end\":4"));
    }

    #[test]
    fn escapes_special_characters() {
        let mut s = String::new();
        esc("a\"b\\c\n", &mut s);
        assert_eq!(s, "\"a\\\"b\\\\c\\n\"");
    }

    #[test]
    fn escapes_other_control_characters_as_u_escapes() {
        // A control char with no short escape must still come out as `\u00xx`; emitted raw it
        // would make the line unparseable JSON for the editors/CI that consume this format.
        let mut s = String::new();
        esc("a\u{1}b\u{1f}", &mut s);
        assert_eq!(s, "\"a\\u0001b\\u001f\"");
    }
}
