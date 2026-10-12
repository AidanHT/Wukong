//! `wukong_lexer` — turns source text into a flat stream of [`Token`]s.
//!
//! The lexer never aborts: unrecognized input produces a [`TokenKind::Error`] token plus a
//! diagnostic, and scanning continues. Literal values are not parsed here — tokens carry only a
//! kind and a [`Span`], and later stages read the covered text from the source map.
//!
//! A Unicode character that imitates ASCII punctuation or whitespace (`−` for `-`, a no-break
//! space) is `E0105`, naming the code point and the ASCII character it was mistaken for; where that
//! is one token the lexer produces it, so the parse continues as if it had been written (see
//! `confusables`). A leading byte-order mark is skipped silently.

mod confusables;
mod token;
pub use token::{Token, TokenKind};

use std::fmt::Write as _;
use wukong_diag::Diagnostic;
use wukong_span::{SourceId, Span};

/// Tokenize an entire source file. Returns the tokens (always ending in [`TokenKind::Eof`])
/// and any diagnostics produced during scanning.
pub fn tokenize(src: &str, source: SourceId) -> (Vec<Token>, Vec<Diagnostic>) {
    let mut lx = Lexer::new(src, source, None);
    let tokens = lx.run();
    (tokens, lx.diags)
}

/// [`tokenize`], plus every comment the scan skipped, in source order: the side channel a
/// formatter (or doc tooling) needs, since the token stream itself carries no trivia. The tokens
/// and diagnostics are exactly the ones [`tokenize`] returns for the same input; collecting the
/// comments is the only difference, and [`tokenize`] does not pay for it.
pub fn tokenize_with_comments(
    src: &str,
    source: SourceId,
) -> (Vec<Token>, Vec<Comment>, Vec<Diagnostic>) {
    let mut lx = Lexer::new(src, source, Some(Vec::new()));
    let tokens = lx.run();
    let comments = lx.comments.take().unwrap_or_default();
    (tokens, comments, lx.diags)
}

/// The comments in `src[lo..hi]`, which must be a run of trivia — what the scan skips between two
/// tokens of `src` (or before the first, or after the last). This is how the parser finds the doc
/// comments before an item without a second token stream: it rescans only the gaps between its
/// tokens, with the very scan [`tokenize`] used, so a `///` inside a string literal or a block
/// comment is never mistaken for one. The spans are positions in `src`. Diagnostics the rescan
/// would repeat (an unterminated block comment) are dropped: the first scan reported them.
pub fn trivia_comments(src: &str, source: SourceId, lo: usize, hi: usize) -> Vec<Comment> {
    let hi = hi.min(src.len());
    let mut lx = Lexer::new(src, source, Some(Vec::new()));
    lx.pos = lo.max(lx.pos);
    while lx.pos < hi {
        let before = lx.pos;
        lx.skip_trivia();
        if lx.pos == before {
            // Not whitespace or a comment: confusable whitespace the first scan reported and
            // skipped (`E0105`). Step over the whole character and keep going.
            let c = lx.bytes[lx.pos];
            for _ in 0..utf8_len(c) {
                lx.bump();
            }
        }
    }
    lx.comments.take().unwrap_or_default()
}

/// Which spelling a [`Comment`] used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommentKind {
    /// `// …` to the end of the line (including the doc forms `///` and `//!`).
    Line,
    /// `/* … */`, which nests: `/* a /* b */ c */` is one comment.
    Block,
}

/// One comment, as [`tokenize_with_comments`] found it.
///
/// A [`CommentKind::Line`] span runs from its `//` up to, but not including, the `\n` that ends
/// the line (a `\r` before that `\n` is inside the span; [`Comment::text`] drops it). A
/// [`CommentKind::Block`] span runs from its `/*` through the `*/` that closes it, counting
/// nesting; an unterminated block comment (E0103) runs to the end of the file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Comment {
    pub kind: CommentKind,
    pub span: Span,
}

impl Comment {
    /// The comment's text in `src` (the source it was lexed from), minus one trailing `\r`: the
    /// CR of a CRLF line ending is not part of a line comment's text.
    pub fn text<'s>(&self, src: &'s str) -> &'s str {
        let t = &src[self.span.lo as usize..self.span.hi as usize];
        t.strip_suffix('\r').unwrap_or(t)
    }

    /// Whether this is a doc comment: `///` (outer) or `//!` (inner). `////…` is an ordinary
    /// comment, as in Rust.
    pub fn is_doc(&self, src: &str) -> bool {
        if self.kind != CommentKind::Line {
            return false;
        }
        let t = self.text(src);
        t.starts_with("//!") || (t.starts_with("///") && !t.starts_with("////"))
    }
}

/// Render a token stream as one `Kind "text"` line per token. Used by `--emit=tokens` and
/// snapshot tests.
pub fn dump(tokens: &[Token], src: &str) -> String {
    let mut out = String::new();
    for t in tokens {
        let text = &src[t.span.lo as usize..t.span.hi as usize];
        writeln!(out, "{} {:?}", t.kind.name(), text).unwrap();
    }
    out
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    source: SourceId,
    diags: Vec<Diagnostic>,
    /// `Some` only for [`tokenize_with_comments`]: every comment [`Lexer::skip_trivia`] skips is
    /// recorded here. `None` (plain [`tokenize`]) records nothing.
    comments: Option<Vec<Comment>>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str, source: SourceId, comments: Option<Vec<Comment>>) -> Lexer<'a> {
        Lexer {
            src,
            bytes: src.as_bytes(),
            // A UTF-8 byte-order mark is an encoding signature some editors write, not source
            // text: skip it (as rustc does) rather than reporting the file's first character.
            pos: if src.starts_with('\u{FEFF}') {
                '\u{FEFF}'.len_utf8()
            } else {
                0
            },
            source,
            diags: Vec::new(),
            comments,
        }
    }

    /// Scan the whole input: every token, ending in [`TokenKind::Eof`].
    fn run(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let t = self.next_token();
            let eof = t.kind == TokenKind::Eof;
            tokens.push(t);
            if eof {
                break;
            }
        }
        tokens
    }

    /// Record a skipped comment that began at `start` and ends at the cursor, when comments are
    /// being collected.
    fn note_comment(&mut self, kind: CommentKind, start: usize) {
        if self.comments.is_some() {
            let span = self.span(start);
            if let Some(cs) = self.comments.as_mut() {
                cs.push(Comment { kind, span });
            }
        }
    }

    fn peek_at(&self, off: usize) -> Option<u8> {
        self.bytes.get(self.pos + off).copied()
    }

    fn bump(&mut self) {
        if self.pos < self.bytes.len() {
            self.pos += 1;
        }
    }

    fn span(&self, lo: usize) -> Span {
        Span::new(self.source, lo as u32, self.pos as u32)
    }

    fn error(&mut self, span: Span, code: &'static str, msg: impl Into<String>) {
        self.diags
            .push(Diagnostic::error(msg).with_code(code).primary(span, ""));
    }

    fn next_token(&mut self) -> Token {
        // A loop, not straight-line code, only for the confusable-whitespace case below: such a
        // character is reported and then skipped like whitespace, so the scan restarts — and a
        // pasted run of them must not grow the stack the way recursion would.
        loop {
            self.skip_trivia();
            let start = self.pos;
            let Some(c) = self.peek_at(0) else {
                return Token::new(TokenKind::Eof, self.span(start));
            };

            if c == b'_' || c.is_ascii_alphabetic() {
                let kind = self.lex_ident(start);
                return Token::new(kind, self.span(start));
            }
            if c.is_ascii_digit() {
                let kind = self.lex_number(start);
                return Token::new(kind, self.span(start));
            }
            if c == b'"' {
                let kind = self.lex_string(start);
                return Token::new(kind, self.span(start));
            }
            if c == b'\'' {
                // A `'name` not closed by a `'` is a loop label, not a char literal (`'a'`).
                if let Some(kind) = self.try_lex_label() {
                    return Token::new(kind, self.span(start));
                }
                let kind = self.lex_char(start);
                return Token::new(kind, self.span(start));
            }
            if let Some(kind) = self.lex_op() {
                return Token::new(kind, self.span(start));
            }

            // Unknown byte: consume the whole UTF-8 character, report, and continue.
            if let Some(tok) = self.unknown_char(start, c) {
                return tok;
            }
        }
    }

    /// Consume and report the character at `start` (first byte `c`), which begins no token.
    /// Returns the token to produce, or `None` when the character was confusable whitespace that
    /// the caller should skip.
    fn unknown_char(&mut self, start: usize, c: u8) -> Option<Token> {
        // `start` is a character boundary: every scan above stops on an ASCII byte or consumes
        // whole characters, so the decode cannot fail on well-formed (`&str`) input.
        let ch = self.src[start..].chars().next().unwrap_or('\u{FFFD}');
        for _ in 0..utf8_len(c) {
            self.bump();
        }
        let span = self.span(start);
        if let Some(conf) = confusables::lookup(ch) {
            self.confusable(span, conf);
            return match conf.recover {
                confusables::Recover::Token(kind) => Some(Token::new(kind, span)),
                confusables::Recover::Trivia => None,
                confusables::Recover::Error => Some(Token::new(TokenKind::Error, span)),
            };
        }
        let mut d = Diagnostic::error(format!("unexpected character {}", describe_char(ch)))
            .with_code("E0101")
            .primary(span, "this character cannot begin a token");
        if !ch.is_ascii() && ch.is_alphanumeric() {
            d = d.help(match confusables::greek_name(ch) {
                Some(name) => format!(
                    "identifiers are ASCII-only (letters, digits and `_`): spell it out, e.g. \
                     `{name}`"
                ),
                None => "identifiers are ASCII-only (letters, digits and `_`)".to_string(),
            });
        }
        self.diags.push(d);
        Some(Token::new(TokenKind::Error, span))
    }

    /// Report a confusable character (`E0105`) at `span`.
    fn confusable(&mut self, span: Span, conf: &confusables::Confusable) {
        let code = format!("U+{:04X} {}", conf.ch as u32, conf.name);
        let d = if conf.recover == confusables::Recover::Trivia {
            let help = if conf.ascii.is_empty() {
                "delete it: it is invisible and not whitespace to the compiler".to_string()
            } else {
                "replace it with an ASCII space".to_string()
            };
            Diagnostic::error(format!("unexpected Unicode whitespace character {code}"))
                .with_code("E0105")
                .primary(span, "not ASCII whitespace")
                .help(help)
        } else {
            let ascii = conf.ascii;
            Diagnostic::error(format!(
                "Unicode character `{}` ({code}) looks like `{ascii}`, but it is not",
                conf.ch
            ))
            .with_code("E0105")
            .primary(span, format!("not the ASCII `{ascii}`"))
            .help(format!("replace it with `{ascii}`"))
        };
        self.diags.push(d);
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek_at(0) {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.bump(),
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    let start = self.pos;
                    while let Some(c) = self.peek_at(0) {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
                    self.note_comment(CommentKind::Line, start);
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    let start = self.pos;
                    self.bump();
                    self.bump();
                    let mut depth = 1;
                    while depth > 0 {
                        match (self.peek_at(0), self.peek_at(1)) {
                            (Some(b'/'), Some(b'*')) => {
                                self.bump();
                                self.bump();
                                depth += 1;
                            }
                            (Some(b'*'), Some(b'/')) => {
                                self.bump();
                                self.bump();
                                depth -= 1;
                            }
                            (Some(_), _) => self.bump(),
                            (None, _) => {
                                let span = self.span(start);
                                self.error(span, "E0103", "unterminated block comment");
                                break;
                            }
                        }
                    }
                    self.note_comment(CommentKind::Block, start);
                }
                _ => break,
            }
        }
    }

    fn lex_ident(&mut self, start: usize) -> TokenKind {
        while let Some(c) = self.peek_at(0) {
            if c == b'_' || c.is_ascii_alphanumeric() {
                self.bump();
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        TokenKind::keyword(text).unwrap_or(TokenKind::Ident)
    }

    fn consume_digits(&mut self) {
        while let Some(c) = self.peek_at(0) {
            if c.is_ascii_digit() || c == b'_' {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn lex_number(&mut self, _start: usize) -> TokenKind {
        let mut is_float = false;

        // Radix-prefixed integers: 0x.., 0b.., 0o..
        if self.peek_at(0) == Some(b'0')
            && matches!(
                self.peek_at(1),
                Some(b'x' | b'X' | b'b' | b'B' | b'o' | b'O')
            )
        {
            self.bump();
            self.bump();
            while let Some(c) = self.peek_at(0) {
                if c.is_ascii_alphanumeric() || c == b'_' {
                    self.bump();
                } else {
                    break;
                }
            }
            return TokenKind::Int;
        }

        self.consume_digits();

        // Fractional part — only if a digit follows the dot (so `1..5` and `1.method` are not floats).
        if self.peek_at(0) == Some(b'.') && self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            self.bump();
            self.consume_digits();
        }

        // Exponent.
        if matches!(self.peek_at(0), Some(b'e' | b'E')) {
            let after = self.peek_at(1);
            let has_exp = after.is_some_and(|c| c.is_ascii_digit())
                || (matches!(after, Some(b'+' | b'-'))
                    && self.peek_at(2).is_some_and(|c| c.is_ascii_digit()));
            if has_exp {
                is_float = true;
                self.bump();
                if matches!(self.peek_at(0), Some(b'+' | b'-')) {
                    self.bump();
                }
                self.consume_digits();
            }
        }

        // Type suffix (i32, u8, usize, f32, bf16, ...). An `f` suffix implies float.
        if self.peek_at(0).is_some_and(|c| c.is_ascii_alphabetic()) {
            if self.peek_at(0) == Some(b'f') {
                is_float = true;
            }
            while let Some(c) = self.peek_at(0) {
                if c.is_ascii_alphanumeric() {
                    self.bump();
                } else {
                    break;
                }
            }
        }

        if is_float {
            TokenKind::Float
        } else {
            TokenKind::Int
        }
    }

    fn lex_string(&mut self, start: usize) -> TokenKind {
        self.bump(); // opening quote
        loop {
            match self.peek_at(0) {
                None | Some(b'\n') => {
                    let span = self.span(start);
                    self.error(span, "E0102", "unterminated string literal");
                    break;
                }
                Some(b'"') => {
                    self.bump();
                    break;
                }
                Some(b'\\') => self.consume_escape(),
                Some(_) => self.bump(),
            }
        }
        TokenKind::Str
    }

    /// Consume a backslash escape at the cursor (`\` is the current byte): `\xHH`, `\u{…}`, or a
    /// one-character escape (`\n`, `\\`, `\'`, …). Stops a `\u{…}` scan at a string/char terminator
    /// so a malformed escape can't swallow the closing quote. Validation of the escape *value* is
    /// the decoder's job (`decode_char_literal` in mir_build); the lexer only delimits the literal.
    fn consume_escape(&mut self) {
        self.bump(); // backslash
        match self.peek_at(0) {
            Some(b'x') => {
                self.bump();
                for _ in 0..2 {
                    if self.peek_at(0).is_some_and(|c| c.is_ascii_hexdigit()) {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
            Some(b'u') => {
                self.bump();
                if self.peek_at(0) == Some(b'{') {
                    self.bump();
                    while let Some(c) = self.peek_at(0) {
                        if c == b'}' {
                            self.bump();
                            break;
                        }
                        if matches!(c, b'"' | b'\'' | b'\n') {
                            break;
                        }
                        self.bump();
                    }
                }
            }
            Some(c) => {
                for _ in 0..utf8_len(c) {
                    self.bump();
                }
            }
            None => {}
        }
    }

    /// Try to lex a loop label `'ident` at the cursor (which is on the opening `'`). A label is a
    /// `'` followed by an identifier that is *not* immediately closed by a `'` — `'a'` is a char
    /// literal, but `'outer:`, `'a ` etc. are labels (the exact disambiguation Rust uses). Returns
    /// `Some(Label)` having consumed `'ident`, or `None` with the cursor unmoved (a char literal).
    fn try_lex_label(&mut self) -> Option<TokenKind> {
        let first = self.peek_at(1)?;
        if !(first.is_ascii_alphabetic() || first == b'_') {
            return None; // `'5'`, `'\n'`, `'!'`, … → char literal
        }
        // Length of the identifier following the quote (ASCII identifier chars).
        let mut len = 1;
        while self
            .peek_at(len)
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            len += 1;
        }
        // A closing quote right after the identifier means it was a char literal (`'a'`), not a label.
        if self.peek_at(len) == Some(b'\'') {
            return None;
        }
        // The lookahead above already proved that all `len` bytes exist.
        self.pos += len;
        Some(TokenKind::Label)
    }

    fn lex_char(&mut self, start: usize) -> TokenKind {
        self.bump(); // opening quote
        let body = self.pos;
        match self.peek_at(0) {
            Some(b'\\') => self.consume_escape(),
            Some(c) if c != b'\'' => {
                for _ in 0..utf8_len(c) {
                    self.bump();
                }
            }
            _ => {}
        }
        let empty = self.pos == body;
        if self.peek_at(0) == Some(b'\'') {
            self.bump();
            if empty {
                // `''` has no body; without this it decodes to 0, i.e. it is silently `'\0'`.
                let span = self.span(start);
                self.error(span, "E0104", "empty character literal");
            }
        } else if let Some(close) = self.close_quote_on_line() {
            // More than one codepoint. Consume through the literal's own closing quote so it is
            // not re-lexed as the opening quote of the next one, which would cost a second
            // diagnostic and swallow the token after it (`'ab';` losing its `;`).
            self.pos = close + 1;
            let span = self.span(start);
            self.error(
                span,
                "E0104",
                "character literal may only contain one codepoint",
            );
        } else {
            // No closing quote on this line: report and leave the cursor where it is, so the
            // rest of the line still lexes as itself.
            let span = self.span(start);
            self.error(span, "E0104", "unterminated character literal");
        }
        TokenKind::Char
    }

    /// Byte offset of the next `'` at or after the cursor, if one occurs before the end of the
    /// line. Used to resync after a malformed character literal.
    fn close_quote_on_line(&self) -> Option<usize> {
        let mut off = 0;
        loop {
            match self.peek_at(off) {
                Some(b'\'') => return Some(self.pos + off),
                Some(b'\n') | None => return None,
                Some(_) => off += 1,
            }
        }
    }

    /// Match a punctuation/operator token at the cursor, consuming it. Returns `None` if the
    /// current byte does not start an operator.
    fn lex_op(&mut self) -> Option<TokenKind> {
        use TokenKind::*;
        let c = self.peek_at(0)?;
        let n = self.peek_at(1);
        let n2 = self.peek_at(2);
        let (kind, len) = match c {
            b'(' => (LParen, 1),
            b')' => (RParen, 1),
            b'{' => (LBrace, 1),
            b'}' => (RBrace, 1),
            b'[' => (LBracket, 1),
            b']' => (RBracket, 1),
            b',' => (Comma, 1),
            b';' => (Semi, 1),
            b'@' => (At, 1),
            b'~' => (Tilde, 1),
            b'?' => (Question, 1),
            b':' if n == Some(b':') => (ColonColon, 2),
            b':' => (Colon, 1),
            b'.' if n == Some(b'.') && n2 == Some(b'=') => (DotDotEq, 3),
            b'.' if n == Some(b'.') => (DotDot, 2),
            b'.' => (Dot, 1),
            b'-' if n == Some(b'>') => (Arrow, 2),
            b'-' if n == Some(b'=') => (MinusEq, 2),
            b'-' => (Minus, 1),
            b'+' if n == Some(b'=') => (PlusEq, 2),
            b'+' => (Plus, 1),
            b'*' if n == Some(b'=') => (StarEq, 2),
            b'*' => (Star, 1),
            b'/' if n == Some(b'=') => (SlashEq, 2),
            b'/' => (Slash, 1),
            b'%' if n == Some(b'=') => (PercentEq, 2),
            b'%' => (Percent, 1),
            b'^' if n == Some(b'=') => (CaretEq, 2),
            b'^' => (Caret, 1),
            b'!' if n == Some(b'=') => (Ne, 2),
            b'!' => (Bang, 1),
            b'=' if n == Some(b'=') => (EqEq, 2),
            b'=' if n == Some(b'>') => (FatArrow, 2),
            b'=' => (Eq, 1),
            b'&' if n == Some(b'&') => (AmpAmp, 2),
            b'&' if n == Some(b'=') => (AmpEq, 2),
            b'&' => (Amp, 1),
            b'|' if n == Some(b'|') => (PipePipe, 2),
            b'|' if n == Some(b'=') => (PipeEq, 2),
            b'|' => (Pipe, 1),
            b'<' if n == Some(b'<') && n2 == Some(b'=') => (ShlEq, 3),
            b'<' if n == Some(b'<') => (Shl, 2),
            b'<' if n == Some(b'=') => (Le, 2),
            b'<' => (Lt, 1),
            b'>' if n == Some(b'>') && n2 == Some(b'=') => (ShrEq, 3),
            b'>' if n == Some(b'>') => (Shr, 2),
            b'>' if n == Some(b'=') => (Ge, 2),
            b'>' => (Gt, 1),
            _ => return None,
        };
        self.pos += len;
        Some(kind)
    }
}

/// How a diagnostic names a character: the character itself in backticks when it is visible ASCII
/// (`` `$` ``), the character plus its code point when it is a visible non-ASCII character
/// (`` `α` (U+03B1) ``), and the code point alone for a control, whitespace or invisible formatting
/// character, which would print as nothing — or, for a bidirectional override, reorder the rest
/// of the terminal line — inside backticks.
fn describe_char(ch: char) -> String {
    let invisible = ch.is_control()
        || ch.is_whitespace()
        || matches!(
            ch as u32,
            0x00AD
                | 0x061C
                | 0x180E
                | 0x200B..=0x200F
                | 0x202A..=0x202E
                | 0x2060..=0x206F
                | 0xFEFF
                | 0xFFF9..=0xFFFB
                | 0xE0000..=0xE007F
        );
    if ch.is_ascii_graphic() {
        format!("`{ch}`")
    } else if invisible || ch.is_ascii() {
        format!("U+{:04X}", ch as u32)
    } else {
        format!("`{ch}` (U+{:04X})", ch as u32)
    }
}

/// Length in bytes of the UTF-8 character beginning with `b`.
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        tokenize(src, SourceId(0))
            .0
            .into_iter()
            .map(|t| t.kind)
            .filter(|k| *k != TokenKind::Eof)
            .collect()
    }

    fn diags(src: &str) -> Vec<Diagnostic> {
        tokenize(src, SourceId(0)).1
    }

    #[test]
    fn keywords_and_identifiers() {
        use TokenKind::*;
        // `f32x8` must lex as a single identifier, not `f32` + `x8`.
        assert_eq!(
            kinds("fn main let x f32x8 Tensor"),
            vec![Fn, Ident, Let, Ident, Ident, Ident]
        );
    }

    #[test]
    fn numbers() {
        use TokenKind::*;
        assert_eq!(
            kinds("0 42 0xFF 0b1010 3.14 1e9 2.0f32 10i64"),
            vec![Int, Int, Int, Int, Float, Float, Float, Int]
        );
        // A range is not a float: `1..5` -> Int DotDot Int.
        assert_eq!(kinds("1..5"), vec![Int, DotDot, Int]);
        // Field-ish access keeps the int separate: `1.0` is a float but `0.5` too.
        assert_eq!(kinds("0.5"), vec![Float]);
    }

    #[test]
    fn operators() {
        use TokenKind::*;
        assert_eq!(
            kinds("+= -> :: ..= << >>= == != <= >= && || @ ? ="),
            vec![
                PlusEq, Arrow, ColonColon, DotDotEq, Shl, ShrEq, EqEq, Ne, Le, Ge, AmpAmp,
                PipePipe, At, Question, Eq
            ]
        );
    }

    #[test]
    fn strings_and_chars() {
        use TokenKind::*;
        assert_eq!(kinds(r#""hi\n" 'a' '\n'"#), vec![Str, Char, Char]);
        // Hex and Unicode escapes are single char tokens (multi-byte escape bodies), and an
        // escaped quote inside a string does not terminate it.
        assert_eq!(kinds(r#"'\x41' '\u{1F600}' "a\"b""#), vec![Char, Char, Str]);
        assert!(diags(r#"'\x41' '\u{1F600}'"#).is_empty());
    }

    #[test]
    fn comments_are_skipped() {
        use TokenKind::*;
        assert_eq!(
            kinds("a // line comment\n b /* block /* nested */ */ c"),
            vec![Ident, Ident, Ident]
        );
    }

    /// `tokenize_with_comments` is `tokenize` plus a side channel: on input that mixes comments
    /// with every lexer-error path (an unterminated block comment E0103, a stray character, an
    /// unterminated string), CRLF line endings and a BOM, its tokens and diagnostics are exactly
    /// `tokenize`'s.
    #[test]
    fn tokenize_with_comments_matches_tokenize() {
        let inputs = [
            "fn f() { // c\n  let x = 1; /* b */ }\n",
            "\u{FEFF}fn f() -> i32 {\r\n    return 1; // crlf\r\n}\r\n",
            "a # b // tail\n\"open\n/* never /* closed */",
            "/// doc\n//! inner\nfn g() {}\n",
            "",
        ];
        for src in inputs {
            let (t1, d1) = tokenize(src, SourceId(3));
            let (t2, _c, d2) = tokenize_with_comments(src, SourceId(3));
            assert_eq!(t1, t2, "tokens differ for {src:?}");
            let codes = |d: &[Diagnostic]| {
                d.iter()
                    .map(|x| (x.code, x.message.clone(), x.primary_span()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(codes(&d1), codes(&d2), "diagnostics differ for {src:?}");
        }
        let (_, _, d) = tokenize_with_comments("x /* a /* b */", SourceId(0));
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0103"));
    }

    /// Spans and kinds: a line comment stops before its `\n` (keeping a CR, which `text` drops), a
    /// nested block comment is one comment through its outer `*/`, doc comments are flagged, and a
    /// comment trailing code on the same line is found too.
    #[test]
    fn comment_spans_and_kinds() {
        let src = "/// doc\nlet x = 1; // trail\r\n/* a /* nested */ b */ //! inner\n//// not doc";
        let (toks, cs, d) = tokenize_with_comments(src, SourceId(0));
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(
            toks.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![
                TokenKind::Let,
                TokenKind::Ident,
                TokenKind::Eq,
                TokenKind::Int,
                TokenKind::Semi,
                TokenKind::Eof
            ]
        );
        let got: Vec<(CommentKind, &str, bool)> = cs
            .iter()
            .map(|c| (c.kind, c.text(src), c.is_doc(src)))
            .collect();
        assert_eq!(
            got,
            vec![
                (CommentKind::Line, "/// doc", true),
                (CommentKind::Line, "// trail", false),
                (CommentKind::Block, "/* a /* nested */ b */", false),
                (CommentKind::Line, "//! inner", true),
                (CommentKind::Line, "//// not doc", false),
            ]
        );
        // The raw span keeps the CR; only `text` strips it.
        let trail = cs[1].span;
        assert_eq!(&src[trail.lo as usize..trail.hi as usize], "// trail\r");
        assert_eq!(cs[0].span.lo, 0);
        assert_eq!(cs[0].span.hi, 7);
        // Spans are in source order.
        assert!(cs.windows(2).all(|w| w[0].span.hi <= w[1].span.lo));
        // An unterminated block comment runs to end of file.
        let (_, cs, _) = tokenize_with_comments("a /* open", SourceId(0));
        assert_eq!((cs[0].span.lo, cs[0].span.hi), (2, 9));
    }

    /// Rescanning every gap between tokens finds exactly the comments the full scan collected,
    /// in order, with the same spans: a `///` inside a string or a block comment is not one, and
    /// confusable whitespace in a gap is stepped over.
    #[test]
    fn trivia_comments_match_the_full_scan() {
        let inputs = [
            "/// doc\nfn f() { let s = \"/// no\"; /* /// no */ } // tail\n//! end",
            "\u{FEFF}//! top\r\nfn g() {}\r\n",
            "a\u{00A0}/// after nbsp\nb",
            "x /* open /// still",
            "",
        ];
        for src in inputs {
            let (toks, want, _) = tokenize_with_comments(src, SourceId(2));
            let mut got = Vec::new();
            let mut prev = 0usize;
            for t in &toks {
                got.extend(trivia_comments(src, SourceId(2), prev, t.span.lo as usize));
                prev = t.span.hi as usize;
            }
            assert_eq!(got, want, "{src:?}");
        }
    }

    #[test]
    fn attribute_sequence() {
        use TokenKind::*;
        assert_eq!(kinds("@tile(64)"), vec![At, Ident, LParen, Int, RParen]);
    }

    #[test]
    fn error_recovery_continues() {
        use TokenKind::*;
        // `#` is not a Wukong token; we emit Error and keep lexing.
        assert_eq!(kinds("a # b"), vec![Ident, Error, Ident]);
        assert_eq!(diags("a # b").len(), 1);
    }

    /// A minus sign pasted from a paper (U+2212) is one E0105 that names the code point and the
    /// ASCII character it imitates, and it lexes AS that character, so the parser sees `a - b`
    /// and reports nothing more. It used to be `E0101 unexpected character `\u{2212}`` followed by
    /// a parser error at the same place.
    #[test]
    fn confusable_minus_is_one_named_error_and_lexes_as_minus() {
        use TokenKind::*;
        let src = "a \u{2212} b";
        assert_eq!(kinds(src), vec![Ident, Minus, Ident]);
        let d = diags(src);
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].code, Some("E0105"));
        assert!(
            d[0].message.contains("U+2212 MINUS SIGN"),
            "{}",
            d[0].message
        );
        assert!(d[0].message.contains("looks like `-`"), "{}", d[0].message);
        // Two-character replacements produce the two-character token.
        assert_eq!(kinds("a \u{2264} b"), vec![Ident, Le, Ident]);
        assert_eq!(kinds("x \u{00D7} y"), vec![Ident, Star, Ident]);
        assert_eq!(
            kinds("f(a\u{FF0C} b)"),
            vec![Ident, LParen, Ident, Comma, Ident, RParen]
        );
    }

    /// No-break and zero-width spaces are reported and then skipped like whitespace; a run of them
    /// costs one diagnostic per character and no stack (the scan loops instead of recursing).
    #[test]
    fn confusable_whitespace_is_reported_and_skipped() {
        use TokenKind::*;
        assert_eq!(kinds("let\u{00A0}x"), vec![Let, Ident]);
        assert_eq!(diags("let\u{00A0}x")[0].code, Some("E0105"));
        let run = format!("a{}b", "\u{200B}".repeat(20_000));
        assert_eq!(kinds(&run), vec![Ident, Ident]);
        let d = diags(&run);
        assert_eq!(d.len(), 20_000);
        assert!(d.iter().all(|x| x.code == Some("E0105")));
    }

    /// A curly quote cannot open a string literal, so it is an error token (one E0105 each) —
    /// never silently a different literal.
    #[test]
    fn confusable_quote_is_an_error_token() {
        use TokenKind::*;
        assert_eq!(kinds("\u{201C}"), vec![Error]);
        let d = diags("\u{201C}");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0105"));
        assert!(d[0].message.contains("looks like `\"`"), "{}", d[0].message);
    }

    /// A non-ASCII letter that is not a confusable stays E0101, but the message names the
    /// character and its code point instead of `\u{3b1}`, and the help spells a Greek letter out.
    #[test]
    fn non_ascii_letter_names_its_code_point() {
        let d = diags("let \u{03B1} = 1;");
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].code, Some("E0101"));
        assert!(d[0].message.contains("(U+03B1)"), "{}", d[0].message);
        assert!(
            d[0].notes.iter().any(|(_, n)| n.contains("`alpha`")),
            "{:?}",
            d[0].notes
        );
        // An invisible formatting character is named by code point only: printing a bidi override
        // inside backticks would reorder the rest of the terminal line.
        let d = diags("a \u{202E} b");
        assert_eq!(d[0].message, "unexpected character U+202E");
        // ASCII punctuation is still shown as itself.
        assert_eq!(diags("a $ b")[0].message, "unexpected character `$`");
    }

    /// A leading UTF-8 byte-order mark is an encoding signature, not source.
    #[test]
    fn leading_bom_is_skipped() {
        use TokenKind::*;
        assert_eq!(kinds("\u{FEFF}fn f"), vec![Fn, Ident]);
        assert!(diags("\u{FEFF}fn f").is_empty());
        // Anywhere else it is a zero-width confusable.
        assert_eq!(diags("fn \u{FEFF}f")[0].code, Some("E0105"));
    }

    #[test]
    fn unterminated_string_reports() {
        let d = diags("\"abc");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0102"));
    }

    #[test]
    fn unterminated_block_comment_reports() {
        // Block comments nest, so the outer `/*` here is still open at EOF.
        let d = diags("/* a /* b */");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0103"));
    }

    #[test]
    fn unterminated_char_reports() {
        // `'5` cannot be a loop label (labels start with an identifier char), so it reaches
        // lex_char and hits EOF with no closing quote.
        let d = diags("'5");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0104"));
    }

    #[test]
    fn empty_char_literal_reports() {
        // `''` has no body: without a diagnostic it decodes to 0, indistinguishable from `'\0'`.
        let d = diags("''");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0104"));
    }

    #[test]
    fn multi_codepoint_char_literal_reports_once_and_resyncs() {
        use TokenKind::*;
        // One stray literal costs one diagnostic, and its own closing quote must not be re-lexed
        // as the opening quote of a new literal — the statement-terminating `;` stays a `;`.
        assert_eq!(kinds("let c = 'ab';"), vec![Let, Ident, Eq, Char, Semi]);
        let d = diags("let c = 'ab';");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].code, Some("E0104"));
    }

    #[test]
    fn keyword_tables_agree() {
        // `keyword()`, `is_keyword()` and `glyph()` are three separate spelling tables. Only
        // `glyph()` and `name()` are compile-time enforced (exhaustive matches); `is_keyword()`
        // is a `matches!` with an implicit `false`, so dropping a variant from it — or respelling
        // one in `keyword()` — is silent. Pin the round trip.
        const KEYWORDS: &[&str] = &[
            "fn", "let", "mut", "if", "else", "while", "for", "in", "loop", "match", "return",
            "break", "continue", "struct", "enum", "impl", "trait", "module", "import", "as",
            "const", "defer", "pub", "extern", "step", "where", "true", "false",
        ];
        for text in KEYWORDS {
            let kind = TokenKind::keyword(text).unwrap_or_else(|| panic!("`{text}` not a keyword"));
            assert!(kind.is_keyword(), "`{text}` is not in is_keyword()");
            assert_eq!(kind.glyph(), Some(*text), "glyph() disagrees for `{text}`");
            assert_eq!(
                kinds(text),
                vec![kind],
                "`{text}` does not lex as its keyword"
            );
        }
        assert!(TokenKind::keyword("fnx").is_none());
        assert!(!TokenKind::Ident.is_keyword());
    }

    #[test]
    fn dump_snapshot() {
        let (toks, _) = tokenize("fn f()", SourceId(0));
        let expected = "Fn \"fn\"\nIdent \"f\"\nLParen \"(\"\nRParen \")\"\nEof \"\"\n";
        assert_eq!(dump(&toks, "fn f()"), expected);
    }
}
