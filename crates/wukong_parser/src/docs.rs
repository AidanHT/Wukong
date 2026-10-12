//! Doc comments: finding them, attaching them to what they document, and rejecting the ones that
//! document nothing (`E0210`).
//!
//! The token stream carries no comments, so [`scan`] finds the doc comments once, when the parser
//! is built, by rescanning only the gaps between tokens (`wukong_lexer::trivia_comments`, the
//! lexer's own trivia scan). Each is recorded with the index of the token it precedes. A
//! documentable construct — an item, an `impl`/`trait`/`extern` member, a struct field, an enum
//! variant — claims the outer (`///`) comments in the gaps before its first token through the token
//! its kind is decided at (so before, between and after its attributes and `pub`). The file claims
//! the inner (`//!`) comments before its first item. Whatever is left unclaimed when the parse ends
//! documents nothing — it sits before a `}`, at the end of the file, before a statement or inside
//! an expression — and is `E0210`, rustc's E0585, one error per run of consecutive comments.

use super::Parser;
use wukong_ast::{Block, DocLine, Docs, Item};
use wukong_diag::{Applicability, Diagnostic};
use wukong_lexer::{Token, TokenKind};
use wukong_span::{Interner, Span};

/// What [`parse_doctest_tokens`] produces.
pub struct DoctestOutput {
    /// The example's items, in order.
    pub items: Vec<Item>,
    /// The statements after them (with the tail expression, when the example ends in one).
    pub block: Block,
    /// Parser errors (the lexer's diagnostics are the caller's).
    pub diags: Vec<Diagnostic>,
    /// The first `NodeId` after the last one allocated.
    pub next_node: u32,
}

/// Parse a doc example (`wukongc --doctest`): its items first — anything that starts like one
/// (`fn`, `struct`, `enum`, `const`, `type`, `trait`, `impl`, `extern`, `import`, `pub`, after any
/// attributes) — then statements to the end of input, the body of the `main` the runner wraps
/// them in. `NodeId`s are allocated from `first_node_id`, as for
/// [`crate::parse_module_tokens_from`].
pub fn parse_doctest_tokens(
    tokens: &[Token],
    src: &str,
    interner: &mut Interner,
    first_node_id: u32,
) -> DoctestOutput {
    let mut p = Parser::new(tokens, src, interner);
    p.next_node = first_node_id;
    let mut items = Vec::new();
    while p.at_item_start() {
        let before = p.pos;
        items.extend(p.parse_item());
        if p.pos == before {
            break;
        }
    }
    let block = p.stmts_to_eof();
    p.report_stray_docs();
    DoctestOutput {
        items,
        block,
        diags: std::mem::take(&mut p.diags),
        next_node: p.next_node,
    }
}

/// One doc comment of the source being parsed.
pub(crate) struct DocSlot {
    span: Span,
    /// `//!` (documents the file) rather than `///` (documents what follows).
    inner: bool,
    /// The index of the token the comment comes before.
    next_tok: usize,
    claimed: bool,
}

/// Every doc comment in `src`, in source order, from the gaps between `tokens` (which were lexed
/// from `src`). A gap with no `//` in it holds no doc comment and is not rescanned.
pub(crate) fn scan(tokens: &[Token], src: &str) -> Vec<DocSlot> {
    let Some(source) = tokens.first().map(|t| t.span.source) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut prev = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        let lo = t.span.lo as usize;
        if lo > prev && src.get(prev..lo).is_some_and(|gap| gap.contains("//")) {
            for c in wukong_lexer::trivia_comments(src, source, prev, lo) {
                if c.is_doc(src) {
                    out.push(DocSlot {
                        span: c.span,
                        inner: c.text(src).starts_with("//!"),
                        next_tok: i,
                        claimed: false,
                    });
                }
            }
        }
        prev = prev.max(t.span.hi as usize);
    }
    out
}

impl Parser<'_> {
    /// Claim the outer (`///`) doc comments before tokens `from..=self.pos`: those written before
    /// the construct that began at token `from`, up to the token the parser is at now. A caller
    /// takes `from` before the construct's attributes and calls this once it has read them and any
    /// `pub`, so a comment before, between or after them all belongs to the construct.
    pub(crate) fn claim_docs(&mut self, from: usize) -> Docs {
        let first = self.doc_slots.partition_point(|d| d.next_tok < from);
        let mut lines = Vec::new();
        for d in &mut self.doc_slots[first..] {
            if d.next_tok > self.pos {
                break;
            }
            if d.inner || d.claimed {
                continue;
            }
            d.claimed = true;
            lines.push(doc_line(self.src, d.span));
        }
        Docs { lines }
    }

    /// Claim the inner (`//!`) doc comments before the current token: the file's own documentation,
    /// taken at its first item (after the `module` header, if any).
    pub(crate) fn claim_inner_docs(&mut self) -> Docs {
        let mut lines = Vec::new();
        for d in &mut self.doc_slots {
            if d.next_tok > self.pos {
                break;
            }
            if d.inner && !d.claimed {
                d.claimed = true;
                lines.push(doc_line(self.src, d.span));
            }
        }
        Docs { lines }
    }

    /// `E0210` for every run of doc comments nothing claimed. Reported only when the parse is
    /// otherwise free of errors: error recovery skips tokens, and a doc comment in a skipped
    /// stretch was never offered to anything, so it would only add noise to the real error.
    pub(crate) fn report_stray_docs(&mut self) {
        if self.diags.iter().any(Diagnostic::is_error) {
            return;
        }
        let mut k = 0;
        while k < self.doc_slots.len() {
            if self.doc_slots[k].claimed {
                k += 1;
                continue;
            }
            // A run: consecutive unclaimed comments of one kind in the same gap.
            let (next_tok, inner) = (self.doc_slots[k].next_tok, self.doc_slots[k].inner);
            let mut end = k + 1;
            while end < self.doc_slots.len()
                && !self.doc_slots[end].claimed
                && self.doc_slots[end].next_tok == next_tok
                && self.doc_slots[end].inner == inner
            {
                end += 1;
            }
            let run: Vec<Span> = self.doc_slots[k..end].iter().map(|d| d.span).collect();
            let d = self.stray_doc(&run, next_tok, inner);
            self.diags.push(d);
            k = end;
        }
    }

    fn stray_doc(&self, run: &[Span], next_tok: usize, inner: bool) -> Diagnostic {
        let span = run[0].to(run[run.len() - 1]);
        let next = self.tokens.get(next_tok).copied();
        let d = if inner {
            Diagnostic::error("inner doc comment (`//!`) is not at the start of the file")
                .with_code("E0210")
                .primary(span, "this documents the file")
                .note(
                    "`//!` comments document the file they are in, so they must come before its \
                     first item (after the `module` header, if there is one)",
                )
                .help("to document what follows it instead, use `///`")
        } else {
            let after = match next.map(|t| t.kind) {
                Some(TokenKind::RBrace) => {
                    "nothing follows it before the `}` that closes this block".to_string()
                }
                Some(TokenKind::Eof) | None => {
                    "nothing follows it: it is at the end of the file".to_string()
                }
                Some(TokenKind::Module) => "it is followed by the `module` header, which takes no \
                                            documentation (document the file with `//!`)"
                    .to_string(),
                Some(_) => format!(
                    "it is followed by {}, which takes no documentation",
                    self.found_at(next_tok)
                ),
            };
            Diagnostic::error("doc comment does not document anything")
                .with_code("E0210")
                .primary(span, after)
                .note(
                    "a `///` comment documents the item, struct field, enum variant or method \
                     written after it; statements and expressions take none",
                )
        };
        // The fix, for a one-line comment: its marker becomes `//`. A run of lines would need one
        // edit per line, so it gets the advice instead.
        match run {
            [one] => d.suggest_with(
                Span::new(one.source, one.lo, one.lo + 3),
                "make it a plain comment",
                "//",
                Applicability::MachineApplicable,
            ),
            _ => d.help("make each line a plain comment: `//`"),
        }
    }

    /// Does an item start here: an item keyword (or `type Name`), after any `@attributes`?
    fn at_item_start(&self) -> bool {
        let kind = |i: usize| self.tokens.get(i).map_or(TokenKind::Eof, |t| t.kind);
        let mut i = self.pos;
        // Skip `@name` and `@name(..)`.
        while kind(i) == TokenKind::At {
            i += 2;
            if kind(i) == TokenKind::LParen {
                let mut depth = 0usize;
                loop {
                    match kind(i) {
                        TokenKind::LParen => depth += 1,
                        TokenKind::RParen => depth -= 1,
                        TokenKind::Eof => return false,
                        _ => {}
                    }
                    i += 1;
                    if depth == 0 {
                        break;
                    }
                }
            }
        }
        let word = |i: usize| {
            self.tokens
                .get(i)
                .map_or("", |t| &self.src[t.span.lo as usize..t.span.hi as usize])
        };
        match kind(i) {
            TokenKind::Fn
            | TokenKind::Struct
            | TokenKind::Enum
            | TokenKind::Const
            | TokenKind::Import
            | TokenKind::Extern
            | TokenKind::Trait
            | TokenKind::Impl
            | TokenKind::Pub => true,
            TokenKind::Ident => word(i) == "type" && kind(i + 1) == TokenKind::Ident,
            _ => false,
        }
    }

    /// The token at index `i` as the "found …" of a diagnostic (see [`Parser::found`]).
    fn found_at(&self, i: usize) -> String {
        let Some(t) = self.tokens.get(i) else {
            return "end of file".to_string();
        };
        match t.kind.glyph() {
            Some(g) => format!("`{g}`"),
            None if t.kind == TokenKind::Ident => format!(
                "identifier `{}`",
                &self.src[t.span.lo as usize..t.span.hi as usize]
            ),
            None => t.kind.describe().to_string(),
        }
    }
}

/// A doc comment's [`DocLine`]: its text after the three-byte marker, without a CR.
fn doc_line(src: &str, span: Span) -> DocLine {
    let t = src
        .get(span.lo as usize + 3..span.hi as usize)
        .unwrap_or_default();
    DocLine {
        text: t.strip_suffix('\r').unwrap_or(t).to_string(),
        span,
    }
}

#[cfg(test)]
#[path = "docs_tests.rs"]
mod tests;
