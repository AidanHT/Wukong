//! `mercury_diag` — compiler diagnostics and a rustc-style terminal renderer.
//!
//! A [`Diagnostic`] carries a severity, an optional stable error code, a headline message,
//! any number of source [`Label`]s, and trailing note/help lines. Producers (lexer, parser,
//! sema, the MIR verifier) build diagnostics with the fluent API and push them into a
//! [`DiagnosticSink`]; the [`Renderer`] turns them into the familiar `error[E0501]: ...`
//! terminal output.

mod catalog;
mod render;
pub use catalog::{all as all_explanations, explain, Explanation};
pub use render::Renderer;

use mercury_span::Span;

/// How serious a diagnostic is. `Bug` denotes an internal compiler error (a Mercury bug),
/// rendered distinctly from user-facing errors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Bug,
    Error,
    Warning,
    Note,
    Help,
}

impl Severity {
    pub fn header(self) -> &'static str {
        match self {
            Severity::Bug => "internal compiler error",
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        }
    }

    /// ANSI color escape for this severity's header.
    pub(crate) fn ansi(self) -> &'static str {
        match self {
            Severity::Bug => "\x1b[1;35m",     // bold magenta
            Severity::Error => "\x1b[1;31m",   // bold red
            Severity::Warning => "\x1b[1;33m", // bold yellow
            Severity::Note => "\x1b[1;36m",    // bold cyan
            Severity::Help => "\x1b[1;36m",    // bold cyan
        }
    }
}

/// A trailing `= note:` / `= help:` line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NoteKind {
    Note,
    Help,
}

/// A pointer at a span of source, with a message. Exactly the spans the user should look at.
#[derive(Clone, Debug)]
pub struct Label {
    pub span: Span,
    pub message: String,
    /// Primary labels get `^` underlines; secondary labels get `-`.
    pub primary: bool,
}

/// A single diagnostic. Build it fluently:
///
/// ```
/// # use mercury_diag::Diagnostic;
/// # use mercury_span::{SourceId, Span};
/// # let span = Span::new(SourceId(0), 0, 1);
/// let d = Diagnostic::error("shape mismatch")
///     .with_code("E0501")
///     .primary(span, "these dimensions disagree")
///     .help("transpose the second operand");
/// ```
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<&'static str>,
    pub message: String,
    pub labels: Vec<Label>,
    pub notes: Vec<(NoteKind, String)>,
}

impl Diagnostic {
    pub fn new(severity: Severity, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn error(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Warning, message)
    }

    pub fn bug(message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(Severity::Bug, message)
    }

    pub fn with_code(mut self, code: &'static str) -> Diagnostic {
        self.code = Some(code);
        self
    }

    pub fn primary(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.labels.push(Label { span, message: message.into(), primary: true });
        self
    }

    pub fn secondary(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.labels.push(Label { span, message: message.into(), primary: false });
        self
    }

    pub fn note(mut self, message: impl Into<String>) -> Diagnostic {
        self.notes.push((NoteKind::Note, message.into()));
        self
    }

    pub fn help(mut self, message: impl Into<String>) -> Diagnostic {
        self.notes.push((NoteKind::Help, message.into()));
        self
    }

    pub fn is_error(&self) -> bool {
        matches!(self.severity, Severity::Error | Severity::Bug)
    }

    /// The span the diagnostic is "about": the first primary label, else the first label.
    pub fn primary_span(&self) -> Option<Span> {
        self.labels
            .iter()
            .find(|l| l.primary)
            .or_else(|| self.labels.first())
            .map(|l| l.span)
    }
}

/// Collects diagnostics during a compilation and tracks whether any errors occurred.
#[derive(Default)]
pub struct DiagnosticSink {
    diagnostics: Vec<Diagnostic>,
    errors: usize,
}

impl DiagnosticSink {
    pub fn new() -> DiagnosticSink {
        DiagnosticSink::default()
    }

    pub fn emit(&mut self, d: Diagnostic) {
        if d.is_error() {
            self.errors += 1;
        }
        self.diagnostics.push(d);
    }

    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    pub fn error_count(&self) -> usize {
        self.errors
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Drain all collected diagnostics, resetting the sink.
    pub fn take(&mut self) -> Vec<Diagnostic> {
        self.errors = 0;
        std::mem::take(&mut self.diagnostics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_span::{SourceId, Span};

    #[test]
    fn builder_sets_fields() {
        let sp = Span::new(SourceId(0), 0, 3);
        let d = Diagnostic::error("boom")
            .with_code("E0001")
            .primary(sp, "here")
            .note("a note")
            .help("a help");
        assert_eq!(d.code, Some("E0001"));
        assert!(d.is_error());
        assert_eq!(d.labels.len(), 1);
        assert_eq!(d.notes.len(), 2);
        assert_eq!(d.primary_span(), Some(sp));
    }

    #[test]
    fn sink_counts_errors() {
        let mut sink = DiagnosticSink::new();
        sink.emit(Diagnostic::warning("w"));
        sink.emit(Diagnostic::error("e"));
        assert!(sink.has_errors());
        assert_eq!(sink.error_count(), 1);
        assert_eq!(sink.diagnostics().len(), 2);
    }
}
