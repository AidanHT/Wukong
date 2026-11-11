//! Comment placement and blank lines.
//!
//! The comments (sorted by position, as the lexer found them) are walked with one cursor, so they
//! come out in exactly their source order. An *own-line element* — an item, an attribute line, a
//! statement, a block's tail, a field, a variant, a match arm, an `impl` method, a parameter /
//! argument / element of a vertical list, or a closing brace — is where comments are placed:
//!
//! - **Leading** ([`Printer::leading`]): before the element, every comment that starts before it
//!   goes on its own line at the current indent.
//! - **Trailing** ([`Printer::trailing`]): after the element's line, the comments that start
//!   before the next source line break (and before the next element), plus any comment inside the
//!   element that nothing has placed yet (a comment in the middle of an expression moves to the
//!   end of its line). A line carries at most one trailing line comment: once one is placed (or a
//!   block comment that spans lines), any further comments go on their own lines after it.
//! - **Closing** ([`Printer::close_at`]): before a `}`/`)`/`]`, the comments still inside.
//! - At end of file, everything left.
//!
//! A blank line is kept before an own-line element (comment or node) exactly when the source
//! has two or more line breaks between the end of the previous element and its start — never
//! right after an opening `{` and never before a closing one.
//!
//! In flat mode (measuring a one-line rendering) nothing here emits anything.

use crate::printer::Printer;
use wukong_lexer::{Comment, CommentKind};

impl<'a> Printer<'a> {
    fn comment_text(&self, c: &Comment) -> String {
        let t = c.text(self.s.src);
        match c.kind {
            CommentKind::Line => t.to_string(),
            // Line endings are re-applied to the whole output; inside a block comment they are
            // normalized to `\n` here so a CRLF file does not grow `\r\r\n`.
            CommentKind::Block => t.replace("\r\n", "\n"),
        }
    }

    fn newlines_between(&self, a: u32, b: u32) -> usize {
        if a >= b {
            return 0;
        }
        self.s
            .src
            .get(a as usize..b as usize)
            .map(|g| g.bytes().filter(|&c| c == b'\n').count())
            .unwrap_or(0)
    }

    /// Emit the blank line an own-line element starting at `lo` keeps, if any.
    fn blank_before(&mut self, lo: u32) {
        if self.blank_lines && !self.block_start && self.newlines_between(self.last_end, lo) >= 2 {
            self.out.push('\n');
        }
        self.block_start = false;
    }

    fn own_line_comment(&mut self, c: Comment) {
        self.blank_before(c.span.lo);
        let text = self.comment_text(&c);
        self.w(&text);
        self.nl();
        self.last_end = self.last_end.max(c.span.hi);
    }

    /// Before an own-line element that starts at source offset `lo`: every comment before it on
    /// its own line, then the element's blank line.
    pub(crate) fn leading(&mut self, lo: u32) {
        if self.flat {
            return;
        }
        while let Some(c) = self.s.comments.get(self.next_comment).copied() {
            if c.span.lo >= lo {
                break;
            }
            self.next_comment += 1;
            self.own_line_comment(c);
        }
        self.blank_before(lo);
    }

    /// After an own-line element that ends at source offset `end` (the next element starts at
    /// `limit`): its trailing comments, and the end of the line.
    pub(crate) fn trailing(&mut self, end: u32, limit: u32) {
        if self.flat {
            return;
        }
        let eol = match self.s.src.get(end as usize..).and_then(|r| r.find('\n')) {
            Some(i) => end + i as u32,
            None => self.s.src.len() as u32,
        };
        let bound = eol.min(limit.max(end));
        let mut closed = false;
        let mut after: Vec<Comment> = Vec::new();
        while let Some(c) = self.s.comments.get(self.next_comment).copied() {
            if c.span.lo >= bound {
                break;
            }
            self.next_comment += 1;
            if closed {
                after.push(c);
                continue;
            }
            let text = self.comment_text(&c);
            self.w(" ");
            self.w(&text);
            if c.kind == CommentKind::Line || text.contains('\n') {
                closed = true;
            }
            self.last_end = self.last_end.max(c.span.hi);
        }
        self.nl();
        for c in after {
            let text = self.comment_text(&c);
            self.w(&text);
            self.nl();
            self.last_end = self.last_end.max(c.span.hi);
        }
        self.last_end = self.last_end.max(end);
    }

    /// A closing `}`/`)`/`]` whose token starts at `close_lo`: the comments still inside the
    /// construct (at its indent), then the closer one level out. The caller continues the line.
    pub(crate) fn close_at(&mut self, close_lo: u32, glyph: &str) {
        if self.flat {
            self.failed = true;
            return;
        }
        while let Some(c) = self.s.comments.get(self.next_comment).copied() {
            if c.span.lo >= close_lo {
                break;
            }
            self.next_comment += 1;
            self.own_line_comment(c);
        }
        if !self.line_start {
            self.nl();
        }
        self.indent = self.indent.saturating_sub(1);
        self.w(glyph);
        self.last_end = self.last_end.max(close_lo + 1);
        self.block_start = false;
    }

    /// A closing `]` of a greedily filled array, which holds no comment: one placed before the
    /// array stays unplaced (it trails the statement), so it cannot move inside the brackets and
    /// change the next run's layout.
    pub(crate) fn close_bare(&mut self, close_lo: u32, glyph: &str) {
        if !self.line_start {
            self.nl();
        }
        self.indent = self.indent.saturating_sub(1);
        self.w(glyph);
        self.last_end = self.last_end.max(close_lo + 1);
        self.block_start = false;
    }

    /// End of file: every comment not yet placed, on its own line.
    pub(crate) fn flush_rest(&mut self) {
        while let Some(c) = self.s.comments.get(self.next_comment).copied() {
            self.next_comment += 1;
            self.own_line_comment(c);
        }
    }
}
