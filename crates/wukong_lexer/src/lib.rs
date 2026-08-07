//! `wukong_lexer` — turns source text into a flat stream of [`Token`]s.
//!
//! The lexer never aborts: unrecognized input produces a [`TokenKind::Error`] token plus a
//! diagnostic, and scanning continues. Literal values are not parsed here — tokens carry only a
//! kind and a [`Span`], and later stages read the covered text from the source map.

mod token;
pub use token::{Token, TokenKind};

use wukong_diag::Diagnostic;
use wukong_span::{SourceId, Span};

/// Tokenize an entire source file. Returns the tokens (always ending in [`TokenKind::Eof`])
/// and any diagnostics produced during scanning.
pub fn tokenize(src: &str, source: SourceId) -> (Vec<Token>, Vec<Diagnostic>) {
    let mut lx = Lexer::new(src, source);
    let mut tokens = Vec::new();
    loop {
        let t = lx.next_token();
        let eof = t.kind == TokenKind::Eof;
        tokens.push(t);
        if eof {
            break;
        }
    }
    (tokens, lx.diags)
}

/// Render a token stream as one `Kind "text"` line per token. Used by `--emit=tokens` and
/// snapshot tests.
pub fn dump(tokens: &[Token], src: &str) -> String {
    let mut out = String::new();
    for t in tokens {
        let text = &src[t.span.lo as usize..t.span.hi as usize];
        out.push_str(&format!("{} {:?}\n", t.kind.name(), text));
    }
    out
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    source: SourceId,
    diags: Vec<Diagnostic>,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str, source: SourceId) -> Lexer<'a> {
        Lexer {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            source,
            diags: Vec::new(),
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
        for _ in 0..utf8_len(c) {
            self.bump();
        }
        let text = self.src[start..self.pos].escape_default().to_string();
        let span = self.span(start);
        self.error(span, "E0101", format!("unexpected character `{text}`"));
        Token::new(TokenKind::Error, span)
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek_at(0) {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.bump(),
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    while let Some(c) = self.peek_at(0) {
                        if c == b'\n' {
                            break;
                        }
                        self.bump();
                    }
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
        for _ in 0..len {
            self.bump();
        }
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
