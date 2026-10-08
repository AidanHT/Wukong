//! `wukong_diag` — compiler diagnostics, the stable error-code catalogue, and two renderers.
//!
//! A [`Diagnostic`] carries a severity, an optional stable error code, a headline message,
//! any number of source [`Label`]s, and trailing note/help lines. Producers (lexer, parser,
//! sema, MIR lowering, and the driver's import loader) build diagnostics with the fluent API
//! and push them into a [`DiagnosticSink`]; only `wukong_driver` renders, picking between the
//! [`Renderer`] — the familiar `error[E0501]: ...` terminal output — and [`to_json`], one JSON
//! object per line. [`explain`] backs `wukongc --explain <CODE>` from the `catalog.rs` table.
//!
//! The MIR verifier is *not* a producer: `wukong_mir` does not depend on this crate and reports
//! malformed IR as plain strings, which `wukong_driver::verify_or_ice` prints as
//! `internal compiler error (MIR verify): …` before stopping the pipeline. With `--verify-each`, the
//! driver wraps structured optimizer failures in [`Severity::Bug`] diagnostics so pass/function
//! context and verifier notes also appear in the selected human or JSON format.

mod catalog;
mod json;
mod render;
pub use catalog::{all as all_explanations, explain, Explanation};
pub use json::to_json;
pub use render::Renderer;

use wukong_span::Span;

/// How serious a diagnostic is. `Bug` denotes an internal compiler error (a Wukong bug),
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

/// How confident the compiler is that a [`Suggestion`] can be applied without human review.
///
/// This mirrors rustc's `Applicability`. Tooling (`wukongc --fix`, an editor's quick-fix) applies
/// only [`Applicability::MachineApplicable`] edits automatically; everything else is advisory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Applicability {
    /// The replacement is correct and complete; applying it will not change meaning in a surprising
    /// way (for example inserting a missing `mut`).
    MachineApplicable,
    /// The replacement is probably what the user meant but should be reviewed (a did-you-mean
    /// rename, where a different candidate might be intended).
    MaybeIncorrect,
    /// The replacement contains placeholders the user must fill in before it compiles.
    HasPlaceholders,
    /// It is unknown whether the replacement is safe to apply mechanically.
    Unspecified,
}

/// A concrete, machine-readable fix: replace the source covered by `span` with `replacement`.
///
/// Suggestions are rendered under the diagnostic as a `help:` line and emitted in the JSON stream
/// so an editor can offer a one-click fix. A deletion is a suggestion with an empty `replacement`;
/// an insertion is a zero-width `span` (its `lo == hi`) with a non-empty `replacement`.
#[derive(Clone, Debug)]
pub struct Suggestion {
    pub span: Span,
    pub replacement: String,
    pub message: String,
    pub applicability: Applicability,
}

/// A single diagnostic. Build it fluently:
///
/// ```
/// # use wukong_diag::Diagnostic;
/// # use wukong_span::{SourceId, Span};
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
    /// Structured, applicable fixes. Empty for most diagnostics; rendered after the notes.
    pub suggestions: Vec<Suggestion>,
}

impl Diagnostic {
    pub fn new(severity: Severity, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
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
        self.labels.push(Label {
            span,
            message: message.into(),
            primary: true,
        });
        self
    }

    pub fn secondary(mut self, span: Span, message: impl Into<String>) -> Diagnostic {
        self.labels.push(Label {
            span,
            message: message.into(),
            primary: false,
        });
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

    /// Attach a structured fix the user *probably* wants ([`Applicability::MaybeIncorrect`]), such
    /// as a did-you-mean rename. Use [`Diagnostic::suggest_with`] for any other applicability.
    pub fn suggest(
        self,
        span: Span,
        message: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Diagnostic {
        self.suggest_with(span, message, replacement, Applicability::MaybeIncorrect)
    }

    /// Attach a structured fix with an explicit [`Applicability`]. A `MachineApplicable` suggestion
    /// is one a tool may apply without review (for example inserting a missing `mut `).
    pub fn suggest_with(
        mut self,
        span: Span,
        message: impl Into<String>,
        replacement: impl Into<String>,
        applicability: Applicability,
    ) -> Diagnostic {
        self.suggestions.push(Suggestion {
            span,
            replacement: replacement.into(),
            message: message.into(),
            applicability,
        });
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

/// Levenshtein (edit) distance between two strings, counted in Unicode scalar values.
///
/// Classic two-row dynamic program, `O(a·b)` time and `O(min(a,b))` extra space. Used by
/// [`did_you_mean`]; kept here so every producer shares one "did you mean" implementation rather
/// than re-deriving it (sema's `nearest_generic` historically had the only copy).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    // Index the shorter string across the row so the row stays `min(|a|,|b|)+1` wide.
    let (a, b) = if a.len() < b.len() { (b, a) } else { (a, b) };
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// The candidate closest to `want` by [`levenshtein`] distance, if one is close enough to be worth
/// suggesting. Returns `None` when nothing is within the threshold.
///
/// The threshold is `max(1, want.len()/3)` scalar values, matching rustc's rule of thumb, so a
/// one-character typo in a short name is caught while unrelated names are not. The tie-break is
/// deterministic: the smallest distance wins, and on an exact tie the first candidate in iteration
/// order is kept (so callers get a stable suggestion regardless of hash ordering — feed a sorted
/// iterator for reproducibility).
pub fn did_you_mean<'a, I>(want: &str, candidates: I) -> Option<&'a str>
where
    I: IntoIterator<Item = &'a str>,
{
    let max_dist = (want.chars().count() / 3).max(1);
    let mut best: Option<(usize, &'a str)> = None;
    for cand in candidates {
        if cand == want {
            continue;
        }
        let d = levenshtein(want, cand);
        if d <= max_dist && best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, cand));
        }
    }
    best.map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{SourceId, Span};

    #[test]
    fn builder_sets_fields() {
        let sp = Span::new(SourceId(0), 0, 3);
        let d = Diagnostic::error("boom")
            .with_code("E0501")
            .primary(sp, "here")
            .note("a note")
            .help("a help");
        assert_eq!(d.code, Some("E0501"));
        assert!(d.is_error());
        assert_eq!(d.labels.len(), 1);
        assert_eq!(d.notes.len(), 2);
        assert_eq!(d.primary_span(), Some(sp));
    }

    #[test]
    fn suggest_records_a_structured_fix() {
        let sp = Span::new(SourceId(0), 4, 4);
        let d = Diagnostic::error("cannot assign twice to immutable binding `x`")
            .with_code("E0304")
            .primary(sp, "cannot assign")
            .suggest_with(
                sp,
                "consider making it mutable",
                "mut ",
                Applicability::MachineApplicable,
            );
        assert_eq!(d.suggestions.len(), 1);
        assert_eq!(d.suggestions[0].replacement, "mut ");
        assert_eq!(
            d.suggestions[0].applicability,
            Applicability::MachineApplicable
        );
    }

    #[test]
    fn levenshtein_basic_distances() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", "abd"), 1);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn did_you_mean_picks_the_nearest_within_threshold() {
        let cands = ["parallel", "generic", "tile", "inline"];
        assert_eq!(did_you_mean("parellel", cands), Some("parallel"));
        // Nothing close: a short, unrelated name is not suggested.
        assert_eq!(did_you_mean("zzzzzzzz", cands), None);
        // An exact match is never suggested against itself.
        assert_eq!(did_you_mean("tile", cands), None);
    }

    #[test]
    fn did_you_mean_tie_break_is_first_in_order() {
        // "ab" is distance 1 from both "ab_" candidates; the first in iteration order wins.
        let cands = ["abx", "aby"];
        assert_eq!(did_you_mean("ab", cands), Some("abx"));
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
