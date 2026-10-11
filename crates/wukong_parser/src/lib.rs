//! `wukong_parser` — turns a token stream into an AST.
//!
//! Hand-written recursive descent for items, statements, and types; a Pratt (precedence-climbing)
//! parser for expressions. The parser allocates [`NodeId`]s, interns identifiers/literals, and
//! recovers from errors so a single mistake doesn't abort the whole parse. This module covers
//! types, expressions, statements, patterns and attributes; the module header and the item forms
//! (`fn`/`struct`/`enum`/`const`/`import`/`extern`) live in `items.rs`, which also defines the three
//! `parse_module*` entry points re-exported below.

use wukong_ast::*;
use wukong_diag::{Applicability, Diagnostic};
use wukong_lexer::{Token, TokenKind};
use wukong_span::{Interner, SourceId, Span, Symbol};

use TokenKind as T;

mod items;
pub use items::{
    parse_module, parse_module_tokens, parse_module_tokens_from, parse_module_tokens_with_lints,
    ParseOutput,
};

/// Parse a standalone expression (for tests and a future REPL).
pub fn parse_expr_str(
    src: &str,
    source: SourceId,
    interner: &mut Interner,
) -> (Expr, Vec<Diagnostic>) {
    let (tokens, mut diags) = wukong_lexer::tokenize(src, source);
    let mut p = Parser::new(&tokens, src, interner);
    let e = p.parse_expr();
    diags.extend(p.diags);
    (e, diags)
}

/// Parse a standalone type (for tests).
pub fn parse_type_str(
    src: &str,
    source: SourceId,
    interner: &mut Interner,
) -> (TypeExpr, Vec<Diagnostic>) {
    let (tokens, mut diags) = wukong_lexer::tokenize(src, source);
    let mut p = Parser::new(&tokens, src, interner);
    let t = p.parse_type();
    diags.extend(p.diags);
    (t, diags)
}

pub(crate) struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    src: &'a str,
    interner: &'a mut Interner,
    next_node: u32,
    pub(crate) diags: Vec<Diagnostic>,
    /// When set, a `Path {` is NOT parsed as a struct literal — disambiguates the condition of
    /// `if`/`while`/`for`/`match` (where `{` opens the body block) from `Name { … }`. Cleared inside
    /// any delimited sub-expression (`(…)`, `[…]`, call args, a struct-literal body), so a
    /// parenthesized `(Point { x: 1 }).x` still parses.
    no_struct_lit: bool,
    /// Current nesting depth of the recursive grammar productions (grouping/prefix/cast/binary
    /// expressions, types, and patterns). Bounded by [`Parser::MAX_DEPTH`] so pathological input —
    /// thousands of nested `(` / `[`, or a 10k-long `1+1+…` chain — reports E0209 instead of
    /// overflowing the stack, either in the parser's own descent or in a later recursive walk over
    /// the resulting AST (sema, MIR lowering, even the tree's `Drop`).
    depth: u32,
    /// Latched once the depth limit is first hit. It keeps E0209 to a single diagnostic and (via
    /// [`Parser::error`]) silences the follow-on recovery cascade — the unmatched `)`/`]` and
    /// "expected …" errors that unwinding a half-parsed monster construct would otherwise spew.
    depth_exceeded: bool,
    /// Inside an `impl Type { … }` block: `Type`'s name, which every `Self` in the block denotes.
    pub(crate) self_ty: Option<Symbol>,
    /// The open delimiters consumed so far and not yet closed (innermost last), kept in step with
    /// [`Parser::bump`]. A failed `expect` of a closer labels its opener ("this `(` is not
    /// closed"), which for an unclosed `{` is the only pointer back to where the problem is: the
    /// failure itself is reported at end of file.
    delims: Vec<(TokenKind, Span)>,
    /// The tokens an `eat` has tried, and failed to find, at the current position (cleared by every
    /// [`Parser::bump`]). A failed `expect` reports them with its own token as the set the parser
    /// would have accepted — "expected one of `,` or `)`" — rather than naming only the last one.
    expected: Vec<TokenKind>,
    /// Lint findings (`W0030` missing `;`, `W0031` ambiguous line continuation): warnings, kept
    /// apart from `diags` so a caller that asserts a clean parse is not failed by a warning. Only
    /// [`parse_module_tokens_with_lints`] hands them out; the driver resolves their levels.
    pub(crate) lints: Vec<Diagnostic>,
    /// Parsing the target type of an `as` cast, where a bare `<` after a type name is the
    /// comparison `x as i32 < n`, never type arguments (those must be spelled `Name::<..>` there).
    in_cast: bool,
    /// A `>>` closed an inner type-argument list and the outer one's `>` with it: the outer list
    /// has been closed already (`Box<Mat<2>>`).
    split_gt: bool,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(tokens: &'a [Token], src: &'a str, interner: &'a mut Interner) -> Parser<'a> {
        Parser {
            tokens,
            pos: 0,
            src,
            interner,
            next_node: 0,
            diags: Vec::new(),
            no_struct_lit: false,
            depth: 0,
            depth_exceeded: false,
            in_cast: false,
            split_gt: false,
            self_ty: None,
            delims: Vec::new(),
            expected: Vec::new(),
            lints: Vec::new(),
        }
    }

    /// Parse an expression in a position where a trailing `{` opens a block (an `if`/`while`/`for`/
    /// `match` head), so a bare `Name { … }` must NOT be read as a struct literal.
    fn parse_cond(&mut self) -> Expr {
        let prev = std::mem::replace(&mut self.no_struct_lit, true);
        let e = self.parse_expr();
        self.no_struct_lit = prev;
        e
    }

    /// Parse `f()`-style content with struct literals re-enabled (a delimited context).
    fn allowing_struct_lit<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let prev = std::mem::replace(&mut self.no_struct_lit, false);
        let out = f(self);
        self.no_struct_lit = prev;
        out
    }

    // ---- Cursor & helpers ----

    fn tok(&self) -> Token {
        self.tokens[self.pos]
    }

    fn kind(&self) -> TokenKind {
        self.tok().kind
    }

    fn nth(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos + n)
            .map(|t| t.kind)
            .unwrap_or(T::Eof)
    }

    fn at(&self, k: TokenKind) -> bool {
        self.kind() == k
    }

    fn span(&self) -> Span {
        self.tok().span
    }

    fn prev_span(&self) -> Span {
        self.tokens[self.pos.saturating_sub(1)].span
    }

    fn bump(&mut self) -> Token {
        let t = self.tok();
        if t.kind != T::Eof {
            self.pos += 1;
            self.track_delim(t);
        }
        self.expected.clear();
        t
    }

    /// Keep [`Parser::delims`] in step with the token just consumed. A closer pops its innermost
    /// matching opener together with any unmatched openers above it — a recovery bump past a stray
    /// closer must not leave a stale opener for a later error to point at; a closer with no
    /// matching opener changes nothing.
    fn track_delim(&mut self, t: Token) {
        match t.kind {
            T::LParen | T::LBracket | T::LBrace => self.delims.push((t.kind, t.span)),
            T::RParen | T::RBracket | T::RBrace => {
                if let Some(open) = opener_of(t.kind) {
                    if let Some(i) = self.delims.iter().rposition(|(k, _)| *k == open) {
                        self.delims.truncate(i);
                    }
                }
            }
            _ => {}
        }
    }

    fn eat(&mut self, k: TokenKind) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            if !self.expected.contains(&k) {
                self.expected.push(k);
            }
            false
        }
    }

    /// The current token as the "found …" of a diagnostic: a keyword or punctuation in backticks
    /// (`` `;` ``), an identifier with its name (`` identifier `b` ``), any other category as its
    /// noun ("integer literal", "end of file").
    fn found(&self) -> String {
        let k = self.kind();
        match k.glyph() {
            Some(g) => format!("`{g}`"),
            None if k == T::Ident => {
                let sp = self.span();
                format!("identifier `{}`", &self.src[sp.lo as usize..sp.hi as usize])
            }
            None => k.describe().to_string(),
        }
    }

    /// Whether the current token starts on a later line than the previous token ends: the parser
    /// has already moved on to the next line, which is where a missing `;`/`)` leaves it.
    fn on_new_line(&self) -> bool {
        if self.pos == 0 {
            return false;
        }
        let (prev, cur) = (self.prev_span(), self.span());
        prev.hi <= cur.lo
            && self
                .src
                .get(prev.hi as usize..cur.lo as usize)
                .is_some_and(|gap| gap.contains('\n'))
    }

    /// Consume `k`, or report `E0200` without consuming anything. The message names every token
    /// the parser tried at this position ("expected one of `,` or `)`"). When the token found sits
    /// on a later line, the error is anchored just after the previous token — where the missing
    /// token goes — with the token it ran into as a secondary label, instead of on the next line's
    /// first token. A missing closer also labels its unclosed opener.
    fn expect(&mut self, k: TokenKind) -> bool {
        // The lexer has already reported an invalid token. Step over it (and any run of them) so
        // the token wanted here is still found right after it: `g(1 $)` costs the lexer's one
        // error, not that plus "expected `)`, found invalid token" plus a stray-`)` cascade.
        while self.at(T::Error) {
            self.bump();
        }
        if self.at(k) {
            self.bump();
            return true;
        }
        // Past the nesting limit every recovery diagnostic is suppressed (see `error`).
        if self.depth_exceeded {
            return false;
        }
        let mut want = std::mem::take(&mut self.expected);
        if !want.contains(&k) {
            want.push(k);
        }
        let want = describe_set(&want);
        let found = self.found();
        let cur = self.span();
        let mut d = Diagnostic::error(format!("expected {want}, found {found}")).with_code("E0200");
        d = if self.on_new_line() {
            d.primary(
                self.prev_span().shrink_to_hi(),
                format!("expected {want} here"),
            )
            .secondary(cur, format!("unexpected {found}"))
        } else {
            d.primary(cur, format!("expected {want}"))
        };
        if let Some(open) = opener_of(k) {
            if let Some(i) = self.delims.iter().rposition(|(o, _)| *o == open) {
                let glyph = open.glyph().unwrap_or("delimiter");
                d = d.secondary(self.delims[i].1, format!("this `{glyph}` is not closed"));
                // Reported once: the next failed closer labels the next opener out.
                self.delims.truncate(i);
            }
        }
        self.diags.push(d);
        false
    }

    fn error(&mut self, span: Span, code: &'static str, msg: impl Into<String>) {
        // Once the nesting limit is hit, a single E0209 is reported and every follow-on recovery
        // diagnostic (the unmatched delimiters and "expected …" errors produced while unwinding the
        // over-deep construct) is suppressed, keeping the output to one clean error.
        if self.depth_exceeded {
            return;
        }
        // A complaint about the current token when that token is the lexer's `Error` token: the
        // lexer already reported the character (E0101/E0105), so "expected …, found invalid token"
        // here would be a second error for one stray character.
        if self.at(T::Error) && span == self.span() {
            return;
        }
        self.diags
            .push(Diagnostic::error(msg).with_code(code).primary(span, ""));
    }

    // ---- Statement-terminator lints (`W0030`, `W0031`) ----

    /// End a statement: consume its `;`, or record `W0030` (see [`Parser::missing_semi`]).
    fn end_stmt(&mut self, start: usize, mark: usize, what: &str, before_brace_ok: bool) {
        if !self.eat(T::Semi) {
            self.missing_semi(start, mark, what, before_brace_ok);
        }
    }

    /// `W0030`: the statement that began at token `start` is not followed by `;`. Not reported
    /// before the block's closing `}` when `before_brace_ok` (every statement but a `let`, as in
    /// Rust), at end of file (the block's missing `}` is the error there), or for a statement that
    /// is already broken: one that pushed an error since `mark` (the error count when it began), or
    /// that a character rejected by the lexer cut short — its `;` is not what is missing.
    fn missing_semi(&mut self, start: usize, mark: usize, what: &str, before_brace_ok: bool) {
        if (before_brace_ok && self.at(T::RBrace))
            || self.at(T::Eof)
            || self.at(T::Error)
            || self.diags.len() > mark
            || self.tokens[start..self.pos]
                .iter()
                .any(|t| t.kind == T::Error)
        {
            return;
        }
        let at = self.prev_span().shrink_to_hi();
        let mut d = Diagnostic::warning(format!("missing `;` after {what}"))
            .with_code("W0030")
            .primary(at, "expected `;` here");
        if !self.on_new_line() {
            d = d.secondary(self.span(), "the next statement starts on the same line");
        }
        // The parse is the same with the `;`, so inserting it changes nothing but the warning.
        self.lints.push(
            d.help(
                "end the statement with `;`. It may be left out only after a block-like statement \
                 (`if`, `match`, `loop`, `while`, `for`, `{ … }`) and, for any statement but a \
                 `let`, before the `}` that closes the block",
            )
            .suggest_with(at, "add the `;`", ";", Applicability::MachineApplicable),
        );
    }

    /// `W0031`: the current token — a `(` or `[` about to be parsed as a call or index of the
    /// expression before it, or a binary `-` — starts a new line, so the expression on the line
    /// above continues into this one although the two read as separate statements (`acc = a[i]`
    /// then `(b + c)` on the next line parse as the call `a[i](b + c)`). Reported only in statement
    /// context — the innermost open delimiter is a `{` — where a new statement could have begun;
    /// inside `(…)` or `[…]` none can, so nothing is ambiguous.
    fn ambiguous_continuation(&mut self, reads_as: &str) {
        let statement_context = matches!(self.delims.last(), Some((T::LBrace, _)));
        if !(statement_context && self.on_new_line()) {
            return;
        }
        let glyph = self.kind().glyph().unwrap_or("token");
        let d = Diagnostic::warning(format!(
            "`{glyph}` at the start of a line continues the expression on the line above"
        ))
        .with_code("W0031")
        .primary(self.span(), format!("this `{glyph}` {reads_as}"))
        .secondary(self.prev_span(), "the expression it continues ends here")
        .help(
            "if a new statement starts here, end the previous one with `;`; if the continuation is \
             meant, move this token to the end of the line above",
        );
        self.lints.push(d);
    }

    /// Whether the current `-` (about to be read as a binary minus) starts a line and touches the
    /// operand after it — `x` then `-1` on the next line, which reads as a negated new statement.
    /// A `- y` with a space keeps the usual continued-subtraction style silent.
    fn minus_reads_as_prefix(&self) -> bool {
        let minus = self.span();
        self.tokens
            .get(self.pos + 1)
            .is_some_and(|next| next.span.lo == minus.hi)
    }

    /// Maximum nesting depth of the recursive grammar productions before the parser bails with
    /// E0209. Generous enough that no realistic program (hand-written or generated) comes close, yet
    /// bounded so that neither the parser's own descent nor any later recursive walk over the AST can
    /// overflow the stack. The compiler front-end runs on a large stack (see `wukongc::main`), so
    /// the actual overflow threshold sits far above this limit.
    const MAX_DEPTH: u32 = 1024;

    /// Report "nesting too deep" exactly once. Pushes the diagnostic directly (bypassing the now
    /// self-silencing [`Parser::error`]) and latches `depth_exceeded`, which dedups this code and
    /// quiets the recovery cascade that unwinding the over-deep construct triggers.
    fn too_deep(&mut self, span: Span) {
        if self.depth_exceeded {
            return;
        }
        self.diags.push(
            Diagnostic::error(format!(
                "expression or type nesting too deep (exceeds the limit of {})",
                Self::MAX_DEPTH
            ))
            .with_code("E0209")
            .primary(span, "the nesting becomes too deep here"),
        );
        self.depth_exceeded = true;
    }

    fn nid(&mut self) -> NodeId {
        let id = NodeId(self.next_node);
        self.next_node += 1;
        id
    }

    fn intern_span(&mut self, span: Span) -> Symbol {
        let text = &self.src[span.lo as usize..span.hi as usize];
        self.interner.intern(text)
    }

    fn ident(&mut self) -> Ident {
        if self.at(T::Ident) {
            let span = self.span();
            self.bump();
            let mut sym = self.intern_span(span);
            // Inside an `impl` block, `Self` names the implementing type — in a type, a path
            // (`Self::new`), a struct literal (`Self { .. }`) and a pattern alike.
            if let Some(ty) = self.self_ty {
                if self.interner.resolve(sym) == "Self" {
                    sym = ty;
                }
            }
            Ident { sym, span }
        } else {
            let span = self.span();
            self.error(
                span,
                "E0201",
                format!("expected identifier, found {}", self.found()),
            );
            Ident {
                sym: self.interner.intern("«error»"),
                span,
            }
        }
    }

    /// Like [`ident`], but also accepts the keyword `extern` by its text, so `@extern("C")` parses.
    ///
    /// Only `extern` is accepted. A blanket "any keyword" test let a stray or dangling `@` consume
    /// the `fn` / `let` that followed it as the attribute's name: the whole item or statement was
    /// then destroyed and the single diagnostic pointed at innocent code after it. `extern` is the
    /// only keyword any attribute in the corpus is spelled with (`@parallel`, `@simd`, `@export`,
    /// `@extern`), so every other keyword falls through to [`ident`]'s E0201 — which does not bump,
    /// leaving the keyword for the item/statement parser to resync on.
    fn ident_like(&mut self) -> Ident {
        if self.at(T::Ident) || self.at(T::Extern) {
            let span = self.span();
            self.bump();
            let sym = self.intern_span(span);
            Ident { sym, span }
        } else {
            self.ident()
        }
    }

    fn finish_expr(&mut self, start: Span, kind: ExprKind) -> Expr {
        Expr {
            id: self.nid(),
            kind,
            span: start.to(self.prev_span()),
        }
    }

    fn finish_type(&mut self, start: Span, kind: TypeKind) -> TypeExpr {
        TypeExpr {
            id: self.nid(),
            kind,
            span: start.to(self.prev_span()),
        }
    }

    // ---- Types ----

    pub(crate) fn parse_type(&mut self) -> TypeExpr {
        let start = self.span();
        // Bound type nesting (`[[[…; 1]; 1]`, `*****T`, deep tuples) so a pathological type cannot
        // overflow the stack here or in a later recursive walk.
        let saved = self.depth;
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            self.too_deep(start);
            self.depth = saved;
            return self.finish_type(start, TypeKind::Unit);
        }
        if self.at(T::Ident) {
            let sp = self.span();
            let sym = self.intern_span(sp);
            if self.interner.resolve(sym) == "shared" {
                self.bump();
                self.expect(T::LBracket);
                let elem = Box::new(self.parse_type());
                self.expect(T::Semi);
                let len = Box::new(self.parse_expr());
                self.expect(T::RBracket);
                self.depth = saved;
                return self.finish_type(
                    start,
                    TypeKind::Array {
                        elem,
                        len,
                        shared: true,
                    },
                );
            }
        }
        let kind = match self.kind() {
            T::Star => {
                self.bump();
                let mutable = self.eat(T::Mut);
                TypeKind::Pointer {
                    mutable,
                    pointee: Box::new(self.parse_type()),
                }
            }
            T::Amp => {
                self.bump();
                let mutable = self.eat(T::Mut);
                TypeKind::Ref {
                    mutable,
                    pointee: Box::new(self.parse_type()),
                }
            }
            T::LBracket => {
                self.bump();
                if self.eat(T::RBracket) {
                    TypeKind::Slice(Box::new(self.parse_type()))
                } else {
                    let elem = Box::new(self.parse_type());
                    self.expect(T::Semi);
                    let len = Box::new(self.parse_expr());
                    self.expect(T::RBracket);
                    TypeKind::Array {
                        elem,
                        len,
                        shared: false,
                    }
                }
            }
            T::LParen => {
                self.bump();
                if self.eat(T::RParen) {
                    TypeKind::Unit
                } else {
                    let first = self.parse_type();
                    if self.at(T::Comma) {
                        let mut items = vec![first];
                        while self.eat(T::Comma) {
                            if self.at(T::RParen) {
                                break;
                            }
                            items.push(self.parse_type());
                        }
                        self.expect(T::RParen);
                        TypeKind::Tuple(items)
                    } else {
                        self.expect(T::RParen);
                        first.kind
                    }
                }
            }
            T::Int => {
                let s = self.intern_span(self.span());
                self.bump();
                TypeKind::Int(s)
            }
            // `fn(A, B) -> R` — a function type (sema allows it only as a parameter's type).
            T::Fn => {
                self.bump();
                self.expect(T::LParen);
                let mut params = Vec::new();
                while !self.at(T::RParen) && !self.at(T::Eof) {
                    params.push(self.parse_type());
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RParen);
                let ret = if self.eat(T::Arrow) {
                    Some(Box::new(self.parse_type()))
                } else {
                    None
                };
                TypeKind::Fn { params, ret }
            }
            T::Ident => self.parse_named_type(),
            _ => {
                let sp = self.span();
                self.error(
                    sp,
                    "E0203",
                    format!("expected type, found {}", self.found()),
                );
                self.bump();
                TypeKind::Unit
            }
        };
        self.depth = saved;
        self.finish_type(start, kind)
    }

    fn parse_named_type(&mut self) -> TypeKind {
        let text = &self.src[self.span().lo as usize..self.span().hi as usize];
        if text == "vec" && self.nth(1) == T::LBracket {
            self.bump(); // vec
            self.bump(); // [
            let elem = Box::new(self.parse_type());
            self.expect(T::Comma);
            // A lane count that does not decode, or does not fit `u32`, is reported — never wrapped.
            // `vec[f32, 4294967304]` used to become `f32x8` and `vec[f32, 0x8]` a zero-lane `f32x0`.
            let lanes_span = self.span();
            let had_int = self.at(T::Int);
            let lanes = match self.parse_int_value().and_then(|n| u32::try_from(n).ok()) {
                Some(n) => n,
                None => {
                    if had_int {
                        self.error(
                            lanes_span,
                            "E0203",
                            "SIMD lane count must be an integer literal that fits in 32 bits",
                        );
                    }
                    0
                }
            };
            self.expect(T::RBracket);
            return TypeKind::Vector { elem, lanes };
        }
        if text == "Tensor" && self.nth(1) == T::LBracket {
            self.bump(); // Tensor
            self.bump(); // [
            let elem = Box::new(self.parse_type());
            let mut dims = Vec::new();
            let mut layout = None;
            while self.eat(T::Comma) {
                if self.at(T::Dot) {
                    layout = Some(self.parse_layout());
                    break;
                }
                dims.push(self.parse_dim());
            }
            self.expect(T::RBracket);
            return TypeKind::Tensor { elem, dims, layout };
        }
        if let Some((scalar, lanes)) = split_vector_ident(text) {
            let span = self.span();
            let sym = self.interner.intern(scalar);
            self.bump();
            let elem = Box::new(TypeExpr {
                id: self.nid(),
                kind: TypeKind::Path(Path {
                    segments: vec![Ident { sym, span }],
                    span,
                }),
                span,
            });
            return TypeKind::Vector { elem, lanes };
        }
        // A named type may be module-qualified either way: `geom.Point` (the item-level dotted
        // spelling) or `geom::Point` (the expression spelling, as in `geom::Point { .. }`).
        let start = self.span();
        let mut segments = vec![self.ident()];
        while matches!(self.kind(), T::Dot | T::ColonColon) && self.nth(1) == T::Ident {
            self.bump();
            segments.push(self.ident());
        }
        let path = Path {
            segments,
            span: start.to(self.prev_span()),
        };
        // `Name<args>` (not as a cast target, where `<` compares) or `Name::<args>` anywhere.
        if self.at(T::ColonColon) && self.nth(1) == T::Lt {
            self.bump(); // ::
            let args = self.parse_type_args();
            return TypeKind::Generic {
                path: Box::new(path),
                args,
            };
        }
        if self.at(T::Lt) && !self.in_cast {
            let args = self.parse_type_args();
            return TypeKind::Generic {
                path: Box::new(path),
                args,
            };
        }
        TypeKind::Path(path)
    }

    fn parse_dotted_path(&mut self) -> Path {
        let start = self.span();
        let mut segments = vec![self.ident()];
        while self.at(T::Dot) && self.nth(1) == T::Ident {
            self.bump();
            segments.push(self.ident());
        }
        Path {
            segments,
            span: start.to(self.prev_span()),
        }
    }

    /// One tensor dimension: an integer, a name, `?`, or arithmetic on those — `N*2`, `H*HD`,
    /// `(L - K)/S + 1`. A restricted Pratt loop over exactly `+ - * / %` (the expression precedences:
    /// `* / %` bind tighter than `+ -`, all left-associative) plus parentheses. Nothing else is an
    /// extent, so the grammar stays the closed algebra sema can put in normal form; a `?` may be
    /// written inside one, and sema rejects it there (it has no value to compute with).
    fn parse_dim(&mut self) -> Dim {
        self.parse_dim_bp(0)
    }

    fn parse_dim_bp(&mut self, min_bp: u8) -> Dim {
        let saved = self.depth;
        let mut lhs = self.parse_dim_atom();
        loop {
            let (op, bp) = match self.kind() {
                T::Plus => (BinOp::Add, 1),
                T::Minus => (BinOp::Sub, 1),
                T::Star => (BinOp::Mul, 2),
                T::Slash => (BinOp::Div, 2),
                T::Percent => (BinOp::Rem, 2),
                _ => break,
            };
            if bp < min_bp {
                break;
            }
            // Charge each fold, exactly as the expression Pratt loop does: the chain is built
            // iteratively, but the tree it leaves is deep, and sema folds it recursively.
            self.depth += 1;
            if self.depth > Self::MAX_DEPTH {
                let sp = self.span();
                self.too_deep(sp);
                break;
            }
            self.bump();
            let rhs = self.parse_dim_bp(bp + 1);
            let span = lhs.span.to(rhs.span);
            lhs = Dim {
                kind: DimKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        self.depth = saved;
        lhs
    }

    fn parse_dim_atom(&mut self) -> Dim {
        let span = self.span();
        // A parenthesized group recurses, so it is charged like any other nesting production.
        let saved = self.depth;
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            self.too_deep(span);
            self.depth = saved;
            return Dim {
                kind: DimKind::Dynamic,
                span,
            };
        }
        let kind = match self.kind() {
            // A dimension the decoder cannot represent is reported, not folded to 0: a 26-digit
            // extent used to become `Tensor[f32, 0]` with no diagnostic anywhere.
            T::Int => match self.parse_int_value() {
                Some(v) => DimKind::Int(v),
                None => {
                    self.error(
                        span,
                        "E0203",
                        "tensor dimension is not a valid integer literal (it must fit in 64 bits)",
                    );
                    DimKind::Dynamic
                }
            },
            T::Question => {
                self.bump();
                DimKind::Dynamic
            }
            T::Ident => {
                let sym = self.intern_span(span);
                self.bump();
                DimKind::Named(sym)
            }
            // Parentheses only group: the inner dimension is reused, its span widened over them.
            T::LParen => {
                self.bump();
                let inner = self.parse_dim_bp(0);
                self.expect(T::RParen);
                self.depth = saved;
                return Dim {
                    kind: inner.kind,
                    span: span.to(self.prev_span()),
                };
            }
            _ => {
                self.error(
                    span,
                    "E0203",
                    "expected a dimension (an integer, a name, `?`, or `+ - * / %` arithmetic \
                     on those)",
                );
                self.bump();
                DimKind::Dynamic
            }
        };
        self.depth = saved;
        Dim {
            kind,
            span: span.to(self.prev_span()),
        }
    }

    fn parse_layout(&mut self) -> Layout {
        self.bump(); // .
        let name_span = self.span();
        let name = self.ident();
        let text = self.interner.resolve(name.sym);
        match text {
            "contiguous" => Layout::Contiguous,
            "col_major" => Layout::ColMajor,
            "strided" => Layout::Strided,
            "tiled" => {
                let mut sizes = Vec::new();
                if self.eat(T::LParen) {
                    while !self.at(T::RParen) && !self.at(T::Eof) {
                        // An integer literal is consumed by `parse_int_value` whether or not it
                        // decodes, so the failure must be reported here rather than bumped again —
                        // the extra bump ate the `,`/`)` and moved the report onto a later token.
                        if self.at(T::Int) {
                            let sp = self.span();
                            match self.parse_int_value() {
                                Some(v) => sizes.push(v),
                                None => self.error(
                                    sp,
                                    "E0203",
                                    "tile extent is not a valid integer literal (it must fit in 64 \
                                     bits)",
                                ),
                            }
                        } else {
                            self.bump();
                        }
                        if !self.eat(T::Comma) {
                            break;
                        }
                    }
                    self.expect(T::RParen);
                }
                Layout::Tiled(sizes)
            }
            _ => {
                self.error(
                    name_span,
                    "E0204",
                    format!("unknown tensor layout `{text}`"),
                );
                Layout::Contiguous
            }
        }
    }

    /// Read an integer literal value at the cursor and consume it. Returns `None` when the cursor is
    /// not on an integer literal (nothing is consumed) or when the literal does not decode to a
    /// `u64` (the literal *is* consumed) — every caller must report that, never substitute a value.
    ///
    /// It used to keep only the leading run of decimal digits, so `0x10` decoded to `0` and each
    /// caller's `unwrap_or(0)` turned a written extent into a silent zero.
    fn parse_int_value(&mut self) -> Option<u64> {
        if !self.at(T::Int) {
            return None;
        }
        let span = self.span();
        let v = parse_u64_literal(&self.src[span.lo as usize..span.hi as usize]);
        self.bump();
        v
    }

    // ---- Expressions (Pratt) ----

    pub(crate) fn parse_expr(&mut self) -> Expr {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Expr {
        let saved = self.depth;
        let mut lhs = self.parse_cast();
        while let Some(op) = token_to_binop(self.kind()) {
            let bp = binop_bp(op);
            if bp < min_bp {
                break;
            }
            // Each fold deepens the left-leaning tree by one. A left-associative chain is built
            // *iteratively* (the loop, not recursion), so the parser itself never goes deep here —
            // but the resulting tree does, and a later recursive consumer (sema, lowering, `Drop`)
            // would overflow on it. Charging each fold to the shared depth budget caps that tree.
            self.depth += 1;
            if self.depth > Self::MAX_DEPTH {
                let sp = self.span();
                self.too_deep(sp);
                break;
            }
            if op == BinOp::Sub && self.minus_reads_as_prefix() {
                self.ambiguous_continuation("subtracts from the expression above, not negates");
            }
            self.bump();
            let rhs = self.parse_expr_bp(bp + 1);
            let span = lhs.span.to(rhs.span);
            lhs = Expr {
                id: self.nid(),
                kind: ExprKind::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        // Restores both this frame's folds and any cast-chain folds `parse_cast` charged above.
        self.depth = saved;
        lhs
    }

    /// Cast level: `as` binds looser than every prefix unary operator but tighter than every binary
    /// operator (the Rust precedence). Sitting it between `parse_expr_bp` and `parse_prefix` means
    /// `*p as T` is `(*p) as T` (not `*(p as T)` — which mis-typed as a `ptrtoint` then a load), and
    /// `-x as u8` is `(-x) as u8`. Chained `x as A as B` folds left.
    fn parse_cast(&mut self) -> Expr {
        let mut e = self.parse_prefix();
        while self.kind() == T::As {
            // A long cast chain (`x as A as B as …`) folds left iteratively, exactly like the binop
            // loop, so it gets the same per-fold depth charge. The enclosing `parse_expr_bp` restores
            // the budget (it snapshots `self.depth` before calling `parse_cast`).
            self.depth += 1;
            if self.depth > Self::MAX_DEPTH {
                let sp = self.span();
                self.too_deep(sp);
                break;
            }
            self.bump();
            let in_cast = std::mem::replace(&mut self.in_cast, true);
            let ty = self.parse_type();
            self.in_cast = in_cast;
            e = self.finish_expr(
                e.span,
                ExprKind::Cast {
                    expr: Box::new(e),
                    ty,
                },
            );
        }
        e
    }

    fn parse_prefix(&mut self) -> Expr {
        // One depth level per prefix expression. This is the choke point every operand passes
        // through (`parse_expr_bp` → `parse_cast` → here), so it bounds *both* a deeply nested
        // grouping descent (`((((…))))`) and a long unary chain (`----…x`, `****…p`), which recurses
        // straight back into `parse_prefix` without going through `parse_expr_bp`.
        let saved = self.depth;
        self.depth += 1;
        let out = if self.depth > Self::MAX_DEPTH {
            let sp = self.span();
            self.too_deep(sp);
            self.finish_expr(sp, ExprKind::TupleLit(Vec::new()))
        } else {
            self.parse_prefix_inner()
        };
        self.depth = saved;
        out
    }

    fn parse_prefix_inner(&mut self) -> Expr {
        let start = self.span();
        let op = match self.kind() {
            T::Minus => Some(UnOp::Neg),
            // `!` and `~` both lower to `UnOp::Not` (MIR `Op::Not`), which is bitwise complement on
            // an integer and logical negation on a `bool` (the result is masked to its width, so
            // `!true == false`) — the Rust-style polymorphic `!`. `~` is the conventional spelling
            // for the integer bitwise form; it is an alias here.
            T::Bang | T::Tilde => Some(UnOp::Not),
            T::Star => Some(UnOp::Deref),
            T::Amp => {
                self.bump();
                let mutable = self.eat(T::Mut);
                let expr = Box::new(self.parse_prefix());
                let kind = ExprKind::Unary {
                    op: if mutable { UnOp::RefMut } else { UnOp::Ref },
                    expr,
                };
                return self.finish_expr(start, kind);
            }
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let expr = Box::new(self.parse_prefix());
            return self.finish_expr(start, ExprKind::Unary { op, expr });
        }
        let primary = self.parse_primary();
        self.parse_postfix(primary)
    }

    fn parse_postfix(&mut self, mut lhs: Expr) -> Expr {
        loop {
            let start = lhs.span;
            // Each postfix operator folds one more level onto the left-leaning tree. The chain is
            // built *iteratively* (this loop, not recursion), so the parser's own descent stays
            // shallow — but the resulting tree does not, and a later recursive consumer (sema,
            // lowering, even the `Box` chain's `Drop`) overflows the stack on it. Charge every fold
            // to the shared budget, exactly like the binop fold in `parse_expr_bp` and the cast fold
            // in `parse_cast`; the `_ => break` arm below builds nothing, so it is not charged. No
            // restore is needed here: the sole caller `parse_prefix` snapshots and restores
            // `self.depth` around us.
            if matches!(
                self.kind(),
                T::LParen | T::LBracket | T::Dot | T::ColonColon
            ) {
                self.depth += 1;
                if self.depth > Self::MAX_DEPTH {
                    let sp = self.span();
                    self.too_deep(sp);
                    break;
                }
            }
            match self.kind() {
                T::LParen => {
                    self.ambiguous_continuation("calls the expression above");
                    let args = self.parse_args();
                    lhs = self.finish_expr(
                        start,
                        ExprKind::Call {
                            callee: Box::new(lhs),
                            generic_args: Vec::new(),
                            args,
                        },
                    );
                }
                T::LBracket => {
                    self.ambiguous_continuation("indexes the expression above");
                    let bracket = self.span();
                    self.bump();
                    let mut indices = Vec::new();
                    while !self.at(T::RBracket) && !self.at(T::Eof) {
                        indices.push(self.parse_expr());
                        if !self.eat(T::Comma) {
                            break;
                        }
                    }
                    // `a[]` is not a slice-of-everything or any other construct — it is a missing
                    // index. The loop above happily produced `Index { indices: [] }`, which sema's
                    // array arm returned the element type for unchecked and mir_build then refused
                    // with the generic `C0001 'expression' is not yet supported by codegen`: a
                    // message that tells the user the COMPILER is incomplete and names nothing they
                    // wrote. Say what is actually missing, where the grammar noticed it.
                    if indices.is_empty() {
                        self.error(
                            bracket,
                            "E0202",
                            "an index expression requires at least one index: `a[i]`, or \
                             `a[i, j]` for a rank-2 tensor",
                        );
                    }
                    self.expect(T::RBracket);
                    lhs = self.finish_expr(
                        start,
                        ExprKind::Index {
                            base: Box::new(lhs),
                            indices,
                        },
                    );
                }
                T::Dot => {
                    self.bump();
                    if self.at(T::Int) {
                        // A tuple index that does not decode, or does not fit `u32`, is reported —
                        // never narrowed. `t.0x1` used to read field 0 instead of field 1.
                        let isp = self.span();
                        let index = match self.parse_int_value().and_then(|n| u32::try_from(n).ok())
                        {
                            Some(n) => n,
                            None => {
                                self.error(
                                    isp,
                                    "E0202",
                                    "tuple field index is not a valid integer literal (it must fit \
                                     in 32 bits)",
                                );
                                0
                            }
                        };
                        lhs = self.finish_expr(
                            start,
                            ExprKind::TupleField {
                                base: Box::new(lhs),
                                index,
                            },
                        );
                    } else if self.at(T::Float)
                        && split_tuple_float(
                            &self.src[self.span().lo as usize..self.span().hi as usize],
                        )
                        .is_some()
                    {
                        // `t.0.0` — the lexer glues two adjacent tuple indices into a single float
                        // token (`0.0`). Split a plain `N.M` float into two consecutive tuple-field
                        // accesses (`(t.N).M`). A float with an exponent/suffix is not a tuple-index
                        // pair, so `split_tuple_float` declines and we fall through to the field-name
                        // error path.
                        let fsp = self.span();
                        let (a, b) =
                            split_tuple_float(&self.src[fsp.lo as usize..fsp.hi as usize]).unwrap();
                        self.bump();
                        let inner = self.finish_expr(
                            start,
                            ExprKind::TupleField {
                                base: Box::new(lhs),
                                index: a,
                            },
                        );
                        lhs = self.finish_expr(
                            start,
                            ExprKind::TupleField {
                                base: Box::new(inner),
                                index: b,
                            },
                        );
                    } else {
                        let name = self.ident();
                        lhs = self.finish_expr(
                            start,
                            ExprKind::Field {
                                base: Box::new(lhs),
                                name,
                            },
                        );
                    }
                }
                T::ColonColon if self.nth(1) == T::Lt => {
                    let generic_args = self.parse_turbofish();
                    let args = if self.at(T::LParen) {
                        self.parse_args()
                    } else {
                        let sp = self.span();
                        self.error(
                            sp,
                            "E0205",
                            "expected `(` after turbofish generic arguments",
                        );
                        Vec::new()
                    };
                    lhs = self.finish_expr(
                        start,
                        ExprKind::Call {
                            callee: Box::new(lhs),
                            generic_args,
                            args,
                        },
                    );
                }
                T::ColonColon => {
                    self.bump();
                    let name = self.ident();
                    lhs = self.finish_expr(
                        start,
                        ExprKind::Field {
                            base: Box::new(lhs),
                            name,
                        },
                    );
                }
                // `as` is NOT handled here: it is a cast level between binary and prefix
                // (`parse_cast`), so it binds looser than postfix `()`/`[]`/`.`/`::` (which stay on
                // the primary) but is applied after the whole prefix expression — fixing `*p as T`.
                _ => break,
            }
        }
        lhs
    }

    fn parse_args(&mut self) -> Vec<Expr> {
        self.bump(); // (
        let mut args = Vec::new();
        self.allowing_struct_lit(|p| {
            while !p.at(T::RParen) && !p.at(T::Eof) {
                args.push(p.parse_expr());
                if !p.eat(T::Comma) {
                    break;
                }
            }
        });
        self.expect(T::RParen);
        args
    }

    fn parse_turbofish(&mut self) -> Vec<TypeExpr> {
        self.bump(); // ::
        self.bump(); // <
        let mut args = Vec::new();
        let in_cast = std::mem::replace(&mut self.in_cast, false);
        while !self.at(T::Gt) && !self.at(T::Eof) {
            args.push(self.parse_type());
            if self.split_gt || !self.eat(T::Comma) {
                break;
            }
        }
        self.in_cast = in_cast;
        self.close_angle();
        args
    }

    /// Close a `<..>` list. A `>>` token closes this list and the enclosing one (`Box<Mat<2>>`):
    /// it is consumed here and [`Parser::split_gt`] tells the enclosing list it is closed already.
    fn close_angle(&mut self) {
        if self.split_gt {
            self.split_gt = false;
        } else if self.at(T::Shr) {
            self.bump();
            self.split_gt = true;
        } else {
            self.expect(T::Gt);
        }
    }

    /// `<A, B, ..>` after a type name (the opening `<` is the current token). Each argument is a
    /// type, or a dimension when only a dimension can be meant: an integer, `?`, or arithmetic
    /// (`N*2`, `(L - K)/S`). A bare name parses as a type; alias expansion reads it as either.
    fn parse_type_args(&mut self) -> Vec<TypeArg> {
        self.bump(); // <
        let in_cast = std::mem::replace(&mut self.in_cast, false);
        let mut args = Vec::new();
        while !self.at(T::Gt) && !self.at(T::Shr) && !self.at(T::Eof) {
            let dim_op =
                |k: TokenKind| matches!(k, T::Plus | T::Minus | T::Star | T::Slash | T::Percent);
            let is_dim = match self.kind() {
                T::Int | T::Question => true,
                T::Ident => dim_op(self.nth(1)),
                T::LParen => self.paren_is_dim(),
                _ => false,
            };
            if is_dim {
                args.push(TypeArg::Dim(self.parse_dim()));
            } else {
                args.push(TypeArg::Type(self.parse_type()));
            }
            if self.split_gt || !self.eat(T::Comma) {
                break;
            }
        }
        self.in_cast = in_cast;
        self.close_angle();
        args
    }

    /// At a `(` inside type arguments: is it a parenthesized dimension (`(L - K)/S`, `(N + 1)`)
    /// rather than a tuple or grouped type? True when an arithmetic operator follows the matching
    /// `)` or the first name or integer inside it.
    fn paren_is_dim(&self) -> bool {
        let dim_op =
            |k: TokenKind| matches!(k, T::Plus | T::Minus | T::Star | T::Slash | T::Percent);
        if matches!(self.nth(1), T::Ident | T::Int) && dim_op(self.nth(2)) {
            return true;
        }
        let mut depth = 0usize;
        let mut n = 0usize;
        loop {
            match self.nth(n) {
                T::LParen => depth += 1,
                T::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return dim_op(self.nth(n + 1));
                    }
                }
                T::Eof | T::Semi | T::LBrace | T::RBrace => return false,
                _ => {}
            }
            n += 1;
        }
    }

    fn parse_primary(&mut self) -> Expr {
        let start = self.span();
        match self.kind() {
            T::Int => {
                let s = self.intern_span(start);
                self.bump();
                self.finish_expr(start, ExprKind::Int(s))
            }
            T::Float => {
                let s = self.intern_span(start);
                self.bump();
                self.finish_expr(start, ExprKind::Float(s))
            }
            T::Str => {
                let s = self.intern_span(start);
                self.bump();
                self.finish_expr(start, ExprKind::Str(s))
            }
            T::Char => {
                let s = self.intern_span(start);
                self.bump();
                self.finish_expr(start, ExprKind::Char(s))
            }
            T::True => {
                self.bump();
                self.finish_expr(start, ExprKind::Bool(true))
            }
            T::False => {
                self.bump();
                self.finish_expr(start, ExprKind::Bool(false))
            }
            T::LParen => {
                self.bump();
                // A delimited context: struct literals are allowed inside `(…)` even within a
                // condition head, so `(Point { x: 1 }).x` parses.
                self.allowing_struct_lit(|p| {
                    if p.eat(T::RParen) {
                        return p.finish_expr(start, ExprKind::TupleLit(Vec::new()));
                    }
                    let first = p.parse_expr();
                    if p.at(T::Comma) {
                        let mut items = vec![first];
                        while p.eat(T::Comma) {
                            if p.at(T::RParen) {
                                break;
                            }
                            items.push(p.parse_expr());
                        }
                        p.expect(T::RParen);
                        p.finish_expr(start, ExprKind::TupleLit(items))
                    } else {
                        p.expect(T::RParen);
                        // Parenthesized expression: keep the inner node but extend its span.
                        Expr {
                            id: first.id,
                            kind: first.kind,
                            span: start.to(p.prev_span()),
                        }
                    }
                })
            }
            T::LBracket => {
                self.bump();
                self.allowing_struct_lit(|p| {
                    if p.eat(T::RBracket) {
                        return p.finish_expr(start, ExprKind::ArrayLit(Vec::new()));
                    }
                    let first = p.parse_expr();
                    if p.eat(T::Semi) {
                        let count = Box::new(p.parse_expr());
                        p.expect(T::RBracket);
                        p.finish_expr(
                            start,
                            ExprKind::ArrayRepeat {
                                value: Box::new(first),
                                count,
                            },
                        )
                    } else {
                        let mut items = vec![first];
                        while p.eat(T::Comma) {
                            if p.at(T::RBracket) {
                                break;
                            }
                            items.push(p.parse_expr());
                        }
                        p.expect(T::RBracket);
                        p.finish_expr(start, ExprKind::ArrayLit(items))
                    }
                })
            }
            T::LBrace => {
                let b = self.parse_block();
                self.finish_expr(start, ExprKind::Block(b))
            }
            T::If => self.parse_if(),
            T::Match => self.parse_match(),
            // `loop { … }` as a value-producing expression: its value is `break v`. A statement
            // `loop` reaches here too (the statement dispatcher routes bare/labeled `loop` through
            // the expression path), so this is the single loop-parsing site.
            T::Loop => {
                self.bump();
                let body = self.parse_block();
                self.finish_expr(start, ExprKind::Loop { label: None, body })
            }
            // A labeled loop in value position: `'l: loop { … }` (so `break 'l v` targets it). Only
            // `loop` is an expression; a labeled `while`/`for` is a statement handled in
            // `parse_block`, so a non-`loop` construct after a label here is a stray form — report
            // and recover with an empty tuple, like the fallthrough below.
            T::Label => {
                let label = self.label_ident();
                self.expect(T::Colon);
                if self.at(T::Loop) {
                    self.bump();
                    let body = self.parse_block();
                    self.finish_expr(
                        start,
                        ExprKind::Loop {
                            label: Some(label),
                            body,
                        },
                    )
                } else {
                    let sp = self.span();
                    self.error(
                        sp,
                        "E0200",
                        "expected `loop` after a label in expression position",
                    );
                    self.finish_expr(start, ExprKind::TupleLit(Vec::new()))
                }
            }
            T::Ident => {
                let text = &self.src[start.lo as usize..start.hi as usize];
                if text == "sizeof" && self.nth(1) == T::LBracket {
                    self.bump();
                    self.bump();
                    let ty = self.parse_type();
                    self.expect(T::RBracket);
                    return self.finish_expr(start, ExprKind::SizeOf(ty));
                }
                if text == "alignof" && self.nth(1) == T::LBracket {
                    self.bump();
                    self.bump();
                    let ty = self.parse_type();
                    self.expect(T::RBracket);
                    return self.finish_expr(start, ExprKind::AlignOf(ty));
                }
                // `Enum::Variant { field: value, … }` — a struct-payload enum-variant literal (a
                // multi-segment struct literal). Only when the `::`-path is *immediately* followed by
                // `{` (so a bare `Enum::Variant` value and an `Enum::Variant(..)` call still flow
                // through the postfix `::` path below), and only in a struct-literal-allowed context
                // (mirrors the single-segment `Name { … }` case just below).
                if !self.no_struct_lit && self.colon_path_before_brace() {
                    let path = self.parse_colon_path();
                    return self.parse_struct_lit(path, start);
                }
                let id = self.ident();
                let path = Path {
                    segments: vec![id],
                    span: start,
                };
                // `Name { field: value, … }` is a struct literal — unless we are parsing the head
                // of an `if`/`while`/`for`/`match`, where the `{` opens the body block instead.
                if !self.no_struct_lit && self.at(T::LBrace) {
                    return self.parse_struct_lit(path, start);
                }
                self.finish_expr(start, ExprKind::Path(path))
            }
            _ => {
                let sp = self.span();
                self.error(
                    sp,
                    "E0202",
                    format!("expected an expression, found {}", self.found()),
                );
                self.bump();
                self.finish_expr(start, ExprKind::TupleLit(Vec::new()))
            }
        }
    }

    /// Parse a struct literal `Path { name: value, … }` (the `{` is the current token). Field values
    /// are a delimited context, so struct literals nest freely inside them.
    fn parse_struct_lit(&mut self, path: Path, start: Span) -> Expr {
        self.bump(); // {
        let mut fields = Vec::new();
        self.allowing_struct_lit(|p| {
            while !p.at(T::RBrace) && !p.at(T::Eof) {
                // Functional update (`P { x: 9, ..base }`) has an AST slot (`StructLit.rest`) and a
                // printer, but no lowering: `mir_build`'s `StructLit` arm ignores `rest` and
                // `lower_struct_init` writes only the listed fields into an uninitialized `alloca`,
                // while sema's `check_struct_literal` suppresses its missing-field error when a
                // `rest` is present. Filling `rest` in here would therefore leave the un-listed
                // fields uninitialized with no diagnostic. Report once, consume the `..base` so the
                // closing `}` is still found, and keep `rest: None`. Before this arm the syntax cost
                // five errors (E0201, E0200, E0202, E0200, E0208), the last of them proving the
                // parser had fallen out of the function body into module scope.
                if p.at(T::DotDot) {
                    let sp = p.span();
                    p.error(
                        sp,
                        "E0201",
                        "expected a field name; struct functional update (`..base`) is not \
                         supported — list every field explicitly",
                    );
                    p.bump();
                    // `P { .. }` has no base to consume; asking for one would report a second
                    // error and eat the `}` that closes the literal.
                    if !p.at(T::RBrace) && !p.at(T::Eof) {
                        let _ = p.parse_expr();
                    }
                    p.eat(T::Comma);
                    break;
                }
                let name = p.ident();
                p.expect(T::Colon);
                let value = p.parse_expr();
                fields.push(FieldInit { name, value });
                if !p.eat(T::Comma) {
                    break;
                }
            }
        });
        self.expect(T::RBrace);
        self.finish_expr(
            start,
            ExprKind::StructLit {
                path,
                fields,
                rest: None,
            },
        )
    }

    fn parse_if(&mut self) -> Expr {
        let start = self.span();
        // Bound `if` / `else if` nesting. A long `else if` chain recurses straight back into
        // `parse_if` (the `else` arm below) *without* passing through the `parse_prefix` choke
        // point, so it must charge the shared depth budget here — otherwise a deep chain recurses
        // unbounded with no diagnostic, just an ever-slower descent and a tree a later recursive
        // walk (sema, lowering, even `Drop`) would overflow on. Mirrors the `parse_type` guard.
        let saved = self.depth;
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            self.too_deep(start);
            self.depth = saved;
            return self.finish_expr(start, ExprKind::TupleLit(Vec::new()));
        }
        self.bump(); // if
        if self.at(T::Let) {
            let out = self.parse_if_let(start);
            self.depth = saved;
            return out;
        }
        let cond = Box::new(self.parse_cond());
        let then_branch = self.parse_block();
        let else_branch = if self.eat(T::Else) {
            if self.at(T::If) {
                Some(Box::new(self.parse_if()))
            } else {
                let b = self.parse_block();
                let e = self.finish_expr(self.prev_span(), ExprKind::Block(b));
                Some(Box::new(e))
            }
        } else {
            None
        };
        let out = self.finish_expr(
            start,
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            },
        );
        self.depth = saved;
        out
    }

    /// The scrutinee of `if let P = e` / `while let P = e`: an expression that stops before a
    /// top-level `&&` or `||`, which continue the condition chain ([`Parser::parse_let_chain_guard`])
    /// rather than the scrutinee. A trailing `{` opens the block, never a struct literal.
    fn parse_let_scrutinee(&mut self) -> Expr {
        let prev = std::mem::replace(&mut self.no_struct_lit, true);
        let e = self.parse_expr_bp(binop_bp(BinOp::And) + 1);
        self.no_struct_lit = prev;
        e
    }

    /// The `&& c1 && c2 …` that may follow an `if let`/`while let` scrutinee, folded left into
    /// one `&&` expression: the guard of the `match` arm the statement becomes, so the conditions
    /// run only after the pattern matched, left to right with short-circuit, and see its bindings
    /// (Rust's let chains, one `let` per chain). `None` when no `&&` follows. A second `let` in the
    /// chain and a `||` after the condition are `E0200`; each is consumed to recover.
    fn parse_let_chain_guard(&mut self) -> Option<Expr> {
        let prev = std::mem::replace(&mut self.no_struct_lit, true);
        let operand_bp = binop_bp(BinOp::And) + 1;
        let mut guard: Option<Expr> = None;
        while self.eat(T::AmpAmp) {
            if self.at(T::Let) {
                let at = self.span();
                self.chain_error(
                    at,
                    "only one `let` per `if let`/`while let` chain is supported; nest a second \
                     `if let` in the block",
                    "a second `let` in this condition",
                );
                self.bump(); // let
                self.parse_pattern();
                if self.eat(T::Eq) {
                    self.parse_expr_bp(operand_bp);
                }
                continue;
            }
            let rhs = self.parse_expr_bp(operand_bp);
            guard = Some(match guard {
                None => rhs,
                Some(lhs) => {
                    let span = lhs.span.to(rhs.span);
                    Expr {
                        id: self.nid(),
                        kind: ExprKind::Binary {
                            op: BinOp::And,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                        span,
                    }
                }
            });
        }
        if self.at(T::PipePipe) {
            let at = self.span();
            self.chain_error(
                at,
                "`||` cannot follow an `if let` condition; parenthesize: `&& (a || b)`",
                "this `||` would apply to the whole condition, pattern included",
            );
            while self.eat(T::PipePipe) {
                self.parse_expr_bp(binop_bp(BinOp::Or) + 1);
            }
        }
        self.no_struct_lit = prev;
        guard
    }

    /// An `E0200` in an `if let`/`while let` condition chain, labelled at `span`.
    fn chain_error(&mut self, span: Span, msg: &str, label: &str) {
        if self.depth_exceeded {
            return;
        }
        self.diags.push(
            Diagnostic::error(msg.to_string())
                .with_code("E0200")
                .primary(span, label.to_string()),
        );
    }

    /// `if let P = e { A } else { B }` (the `if` consumed, the cursor on `let`), parsed as the
    /// `match e { P => { A }, _ => { B } }` it means. With no `else` the `_` arm does nothing and
    /// the `then` block becomes a statement, so the whole expression is `()` like an `else`-less
    /// `if`. An `else if …` chain becomes the `_` arm's body. `if let P = e && c1 && c2` puts the
    /// conditions on the pattern's arm as its guard: `match e { P if c1 && c2 => { A }, _ => { B } }`.
    fn parse_if_let(&mut self, start: Span) -> Expr {
        self.bump(); // let
        let pat = self.parse_pattern();
        self.expect(T::Eq);
        let scrutinee = Box::new(self.parse_let_scrutinee());
        let guard = self.parse_let_chain_guard();
        let then_block = self.parse_block();
        let then_span = then_block.span;
        let (then_body, else_body) = if self.eat(T::Else) {
            let else_body = if self.at(T::If) {
                self.parse_if()
            } else {
                let b = self.parse_block();
                self.block_expr(b)
            };
            (self.block_expr(then_block), else_body)
        } else {
            // `if let P = e { A }` is `()`: discard `A`'s value as a statement.
            let stmt = Stmt {
                id: self.nid(),
                attrs: Vec::new(),
                kind: StmtKind::Expr(self.block_expr(then_block)),
                span: then_span,
            };
            let body = Block {
                id: self.nid(),
                stmts: vec![stmt],
                tail: None,
                span: then_span,
            };
            let empty = Block {
                id: self.nid(),
                stmts: Vec::new(),
                tail: None,
                span: then_span,
            };
            (self.block_expr(body), self.block_expr(empty))
        };
        let else_span = else_body.span;
        let arms = vec![
            MatchArm {
                span: pat.span.to(then_span),
                pat,
                guard,
                body: then_body,
                origin: ArmOrigin::Source,
            },
            MatchArm {
                pat: Pattern {
                    id: self.nid(),
                    kind: PatKind::Wildcard,
                    span: else_span,
                },
                guard: None,
                body: else_body,
                span: else_span,
                origin: ArmOrigin::LetFallthrough,
            },
        ];
        self.finish_expr(start, ExprKind::Match { scrutinee, arms })
    }

    /// A block as an expression, spanning the block.
    fn block_expr(&mut self, b: Block) -> Expr {
        let span = b.span;
        Expr {
            id: self.nid(),
            kind: ExprKind::Block(b),
            span,
        }
    }

    fn parse_match(&mut self) -> Expr {
        let start = self.span();
        // Bound `match` nesting (an arm body that is itself a `match`, nested deep). The arm body
        // already passes through `parse_prefix` (which charges depth and reports E0209), but charge
        // it here too so the limit is enforced from the structural recursion itself — and, more
        // importantly, once that guard trips the over-deep body parse returns a placeholder
        // *without consuming its token*, so the arm loop below must also guarantee forward progress
        // or it would spin forever on the stuck token.
        let saved = self.depth;
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            self.too_deep(start);
            self.depth = saved;
            return self.finish_expr(start, ExprKind::TupleLit(Vec::new()));
        }
        self.bump(); // match
        let scrutinee = Box::new(self.parse_cond());
        self.expect(T::LBrace);
        let mut arms = Vec::new();
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let before = self.pos;
            let arm_start = self.span();
            let pat = self.parse_pattern();
            // Optional `if <expr>` guard between the pattern and `=>`. The guard head disallows a
            // bare struct literal (like other condition heads) so `n if Foo { .. }` isn't ambiguous.
            let guard = if self.eat(T::If) {
                Some(self.parse_cond())
            } else {
                None
            };
            self.expect(T::FatArrow);
            let body = self.parse_expr();
            arms.push(MatchArm {
                pat,
                guard,
                body,
                span: arm_start.to(self.prev_span()),
                origin: ArmOrigin::Source,
            });
            self.eat(T::Comma);
            // Guarantee forward progress (mirrors `module()`): if the depth limit has tripped and
            // every sub-parse above consumed nothing, bump so this loop can't spin on a stuck token.
            if self.pos == before && !self.at(T::RBrace) && !self.at(T::Eof) {
                self.bump();
            }
        }
        self.expect(T::RBrace);
        let out = self.finish_expr(start, ExprKind::Match { scrutinee, arms });
        self.depth = saved;
        out
    }

    // ---- Statements & blocks ----

    pub(crate) fn parse_block(&mut self) -> Block {
        let start = self.span();
        let id = self.nid();
        // Bound block nesting. Every nested body funnels through here — `{ … }` blocks, the
        // branches of `if`/`else`, and the bodies of `while`/`for`/`loop` — and several of those
        // paths recurse without otherwise charging the depth budget (a loop body via
        // `parse_while` → `parse_block`; a brace block via `parse_prefix`, whose own guard trips but
        // then leaves this loop spinning on an unconsumed `{`). Charge it here so a pathologically
        // deep nest reports E0209 once and stops, instead of overflowing the stack or stalling a
        // later recursive walk. Mirrors the `parse_type` guard; the loop below adds the matching
        // forward-progress guarantee.
        let saved = self.depth;
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            self.too_deep(start);
            self.depth = saved;
            return Block {
                id,
                stmts: Vec::new(),
                tail: None,
                span: start.to(self.prev_span()),
            };
        }
        self.expect(T::LBrace);
        let mut stmts = Vec::new();
        let mut tail = None;
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let before = self.pos;
            // The error count when this statement began: `missing_semi` stays quiet about a
            // statement that has already been reported as broken.
            let mark = self.diags.len();
            let stmt_start = self.span();
            let attrs = self.parse_attrs();
            match self.kind() {
                T::Let => {
                    let k = self.parse_let();
                    self.end_stmt(before, mark, "a `let` statement", false);
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                T::Const => {
                    // A local `const X: T = v;` is parsed as an immutable binding.
                    self.bump();
                    let name = self.ident();
                    let pat = Pattern {
                        id: self.nid(),
                        kind: PatKind::Ident(name.sym),
                        span: name.span,
                    };
                    self.expect(T::Colon);
                    let ty = Some(self.parse_type());
                    self.expect(T::Eq);
                    let init = Some(self.parse_expr());
                    self.end_stmt(before, mark, "a `const` statement", false);
                    let k = StmtKind::Let {
                        pat,
                        mutable: false,
                        ty,
                        init,
                    };
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                T::Return => {
                    self.bump();
                    let e = if self.at(T::Semi) || self.at(T::RBrace) {
                        None
                    } else {
                        Some(self.parse_expr())
                    };
                    self.end_stmt(before, mark, "`return`", true);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Return(e), stmt_start));
                }
                T::Break => {
                    self.bump();
                    // Optional target label: `break 'outer;` / `break 'outer v;`.
                    let label = if self.at(T::Label) {
                        Some(self.label_ident())
                    } else {
                        None
                    };
                    // Optional break value: `break v;` yields `v` from a value-producing `loop`.
                    // Absent before `;`/`}` (a plain `break`); otherwise the trailing expression.
                    let value = if self.at(T::Semi) || self.at(T::RBrace) {
                        None
                    } else {
                        Some(self.parse_expr())
                    };
                    self.end_stmt(before, mark, "`break`", true);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Break(label, value), stmt_start));
                }
                T::Continue => {
                    self.bump();
                    let label = if self.at(T::Label) {
                        Some(self.label_ident())
                    } else {
                        None
                    };
                    self.end_stmt(before, mark, "`continue`", true);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Continue(label), stmt_start));
                }
                T::Defer => {
                    self.bump();
                    let target = self.parse_expr();
                    // `defer { … }` (any block-like deferred expression) ends its own statement,
                    // as a block-like expression statement does; every other `defer` needs its `;`.
                    let block_like = is_block_like(&target) && cur_assign_op(self.kind()).is_none();
                    // `defer x += 1;` defers an assignment statement: wrap it in a block so the
                    // deferred thing is still one expression (`defer { x += 1; }`).
                    let e = if let Some(op) = cur_assign_op(self.kind()) {
                        let assign_start = target.span;
                        self.bump();
                        let value = self.parse_expr();
                        let assign = self.mk_stmt(
                            Vec::new(),
                            StmtKind::Assign { target, op, value },
                            assign_start,
                        );
                        let block = Block {
                            id: self.nid(),
                            stmts: vec![assign],
                            tail: None,
                            span: assign_start.to(self.prev_span()),
                        };
                        self.finish_expr(assign_start, ExprKind::Block(block))
                    } else {
                        target
                    };
                    if block_like {
                        self.eat(T::Semi);
                    } else {
                        self.end_stmt(before, mark, "a `defer` statement", true);
                    }
                    stmts.push(self.mk_stmt(attrs, StmtKind::Defer(e), stmt_start));
                }
                T::While => {
                    let k = self.parse_while(None);
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                T::For => {
                    let k = self.parse_for(None);
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                // `'label: while/for { … }` — a labeled statement loop. A labeled `loop` (which is a
                // value expression) is *not* handled here: the `nth(2) == Loop` guard lets it fall
                // through to the expression path below, so `'l: loop {…}` can be a value / block tail.
                T::Label if self.nth(2) != T::Loop => {
                    let label = self.label_ident();
                    self.expect(T::Colon);
                    let k = match self.kind() {
                        T::While => self.parse_while(Some(label)),
                        T::For => self.parse_for(Some(label)),
                        _ => {
                            let sp = self.span();
                            self.error(
                                sp,
                                "E0200",
                                "expected `loop`, `while`, or `for` after a label",
                            );
                            continue;
                        }
                    };
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                _ => {
                    let e = self.parse_expr();
                    if let Some(op) = cur_assign_op(self.kind()) {
                        self.bump();
                        let value = self.parse_expr();
                        self.end_stmt(before, mark, "an assignment", true);
                        stmts.push(self.mk_stmt(
                            attrs,
                            StmtKind::Assign {
                                target: e,
                                op,
                                value,
                            },
                            stmt_start,
                        ));
                    } else if self.eat(T::Semi) {
                        stmts.push(self.mk_stmt(attrs, StmtKind::Expr(e), stmt_start));
                    } else if self.at(T::RBrace) {
                        tail = Some(Box::new(e));
                        break;
                    } else {
                        // A block-like expression ends its own statement (Rust's rule).
                        if !is_block_like(&e) {
                            self.missing_semi(before, mark, "an expression statement", true);
                        }
                        stmts.push(self.mk_stmt(attrs, StmtKind::Expr(e), stmt_start));
                    }
                }
            }
            // Guarantee forward progress (mirrors `module()`): once the depth limit trips, an
            // over-deep operand parse (a nested `{` or `match`) returns a placeholder without
            // consuming its opening token, which would otherwise spin this loop forever on it. The
            // bump never fires for well-formed input — every statement form above consumes a token.
            if self.pos == before && !self.at(T::RBrace) && !self.at(T::Eof) {
                self.bump();
            }
        }
        self.expect(T::RBrace);
        self.depth = saved;
        Block {
            id,
            stmts,
            tail,
            span: start.to(self.prev_span()),
        }
    }

    fn mk_stmt(&mut self, attrs: Vec<Attr>, kind: StmtKind, start: Span) -> Stmt {
        Stmt {
            id: self.nid(),
            attrs,
            kind,
            span: start.to(self.prev_span()),
        }
    }

    fn parse_let(&mut self) -> StmtKind {
        self.bump(); // let
        let mutable = self.eat(T::Mut);
        let pat = self.parse_pattern();
        let ty = if self.eat(T::Colon) {
            Some(self.parse_type())
        } else {
            None
        };
        let init = if self.eat(T::Eq) {
            Some(self.parse_expr())
        } else {
            None
        };
        if self.at(T::Else) {
            if let Some(init) = init {
                if let Some(t) = &ty {
                    let span = t.span;
                    self.error(
                        span,
                        "E0200",
                        "a `let … else` takes no type annotation; annotate the value it matches \
                         instead",
                    );
                }
                return self.parse_let_else(pat, mutable, init);
            }
        }
        // The terminating `;` is the caller's (`parse_block`'s `end_stmt`).
        StmtKind::Let {
            pat,
            mutable,
            ty,
            init,
        }
    }

    /// `let P = e else { B };` (the cursor on `else`), parsed as what it means: a `let` of the
    /// names `P` binds, initialized from `match e { P => (names), _ => { B } }`, whose `_` arm sema
    /// requires to diverge. One name binds directly (`let x = match …`); none makes the whole
    /// statement the `match` (`let Some(_) = e else { return; }`). The terminating `;` is the
    /// caller's.
    fn parse_let_else(&mut self, pat: Pattern, mutable: bool, init: Expr) -> StmtKind {
        self.bump(); // else
        let else_block = self.parse_block();
        let else_span = else_block.span;
        let else_body = self.block_expr(else_block);
        let mut names: Vec<(Symbol, Span)> = Vec::new();
        pattern_bindings(&pat, &mut names);
        let name_expr = |p: &mut Self, (sym, span): (Symbol, Span)| Expr {
            id: p.nid(),
            kind: ExprKind::Path(Path {
                segments: vec![Ident { sym, span }],
                span,
            }),
            span,
        };
        let name_pat = |p: &mut Self, (sym, span): (Symbol, Span)| Pattern {
            id: p.nid(),
            kind: PatKind::Ident(sym),
            span,
        };
        let pat_span = pat.span;
        let (value, let_pat) = match names.as_slice() {
            [] => {
                let unit = Block {
                    id: self.nid(),
                    stmts: Vec::new(),
                    tail: None,
                    span: pat_span,
                };
                (self.block_expr(unit), None)
            }
            [one] => (name_expr(self, *one), Some(name_pat(self, *one))),
            many => {
                let many = many.to_vec();
                let vals = many.iter().map(|n| name_expr(self, *n)).collect();
                let pats = many.iter().map(|n| name_pat(self, *n)).collect();
                let value = Expr {
                    id: self.nid(),
                    kind: ExprKind::TupleLit(vals),
                    span: pat_span,
                };
                let let_pat = Pattern {
                    id: self.nid(),
                    kind: PatKind::Tuple(pats),
                    span: pat_span,
                };
                (value, Some(let_pat))
            }
        };
        let span = init.span.to(else_span);
        let arms = vec![
            MatchArm {
                span: pat_span,
                pat,
                guard: None,
                body: value,
                origin: ArmOrigin::Source,
            },
            MatchArm {
                pat: Pattern {
                    id: self.nid(),
                    kind: PatKind::Wildcard,
                    span: else_span,
                },
                guard: None,
                body: else_body,
                span: else_span,
                origin: ArmOrigin::LetElse,
            },
        ];
        let m = Expr {
            id: self.nid(),
            kind: ExprKind::Match {
                scrutinee: Box::new(init),
                arms,
            },
            span,
        };
        match let_pat {
            Some(pat) => StmtKind::Let {
                pat,
                mutable,
                ty: None,
                init: Some(m),
            },
            None => StmtKind::Expr(m),
        }
    }

    fn parse_while(&mut self, label: Option<Ident>) -> StmtKind {
        let start = self.span();
        self.bump(); // while
        if self.at(T::Let) {
            return self.parse_while_let(label, start);
        }
        let cond = self.parse_cond();
        let body = self.parse_block();
        StmtKind::While { label, cond, body }
    }

    /// `[label:] while let P = e { A }` (the cursor on `let`), parsed as the
    /// `[label:] loop { match e { P => { A }, _ => { break; } } }` it means: `e` is evaluated afresh
    /// on every iteration, a `continue` re-tests it, and a `break` (or the first non-match) leaves.
    /// `while let P = e && c` guards the pattern's arm with `c`, as `if let` does.
    fn parse_while_let(&mut self, label: Option<Ident>, start: Span) -> StmtKind {
        self.bump(); // let
        let pat = self.parse_pattern();
        self.expect(T::Eq);
        let scrutinee = Box::new(self.parse_let_scrutinee());
        let guard = self.parse_let_chain_guard();
        let body = self.parse_block();
        let span = start.to(self.prev_span());
        let body_span = body.span;
        let body = self.block_expr(body);
        let brk = Stmt {
            id: self.nid(),
            attrs: Vec::new(),
            kind: StmtKind::Break(None, None),
            span: start,
        };
        let exit = Block {
            id: self.nid(),
            stmts: vec![brk],
            tail: None,
            span: start,
        };
        let exit = self.block_expr(exit);
        let arms = vec![
            MatchArm {
                span: pat.span.to(body_span),
                pat,
                guard,
                body,
                origin: ArmOrigin::Source,
            },
            MatchArm {
                pat: Pattern {
                    id: self.nid(),
                    kind: PatKind::Wildcard,
                    span: start,
                },
                guard: None,
                body: exit,
                span: start,
                origin: ArmOrigin::LetFallthrough,
            },
        ];
        let m = Expr {
            id: self.nid(),
            kind: ExprKind::Match { scrutinee, arms },
            span,
        };
        let m = Stmt {
            id: self.nid(),
            attrs: Vec::new(),
            kind: StmtKind::Expr(m),
            span,
        };
        let body = Block {
            id: self.nid(),
            stmts: vec![m],
            tail: None,
            span,
        };
        StmtKind::Expr(Expr {
            id: self.nid(),
            kind: ExprKind::Loop { label, body },
            span,
        })
    }

    fn parse_for(&mut self, label: Option<Ident>) -> StmtKind {
        self.bump(); // for
        let pat = self.parse_pattern();
        self.expect(T::In);
        let prev = std::mem::replace(&mut self.no_struct_lit, true);
        let iter = self.parse_for_iter();
        self.no_struct_lit = prev;
        let body = self.parse_block();
        StmtKind::For {
            label,
            pat,
            iter,
            body,
        }
    }

    /// Read a loop-label token `'name` at the cursor, interning the name without the leading `'`.
    fn label_ident(&mut self) -> Ident {
        let span = self.span();
        self.bump();
        let text = &self.src[span.lo as usize + 1..span.hi as usize];
        let sym = self.interner.intern(text);
        Ident { sym, span }
    }

    fn parse_for_iter(&mut self) -> ForIter {
        let start = self.parse_expr();
        if self.at(T::DotDot) || self.at(T::DotDotEq) {
            let inclusive = self.at(T::DotDotEq);
            self.bump();
            let end = if self.at(T::Step) || self.at(T::LBrace) {
                None
            } else {
                Some(self.parse_expr())
            };
            let step = if self.eat(T::Step) {
                Some(self.parse_expr())
            } else {
                None
            };
            ForIter::Range {
                start,
                end,
                inclusive,
                step,
            }
        } else {
            ForIter::Expr(start)
        }
    }

    /// Parse a pattern, including an or-pattern `A | B | C` at the top level (each alternative may
    /// itself be a range or path pattern).
    fn parse_pattern(&mut self) -> Pattern {
        let start = self.span();
        let first = self.parse_pattern_range();
        if !self.at(T::Pipe) {
            return first;
        }
        let mut alts = vec![first];
        while self.eat(T::Pipe) {
            alts.push(self.parse_pattern_range());
        }
        Pattern {
            id: self.nid(),
            kind: PatKind::Or(alts),
            span: start.to(self.prev_span()),
        }
    }

    /// A primary pattern optionally followed by a range tail `..hi` / `..=hi`. A range is recognized
    /// only after an integer- or char-literal lower bound, so it never shadows `_`/identifier/tuple
    /// patterns.
    fn parse_pattern_range(&mut self) -> Pattern {
        let start = self.span();
        let lo = self.parse_pattern_primary();
        if matches!(lo.kind, PatKind::Int { .. } | PatKind::Char(_))
            && (self.at(T::DotDot) || self.at(T::DotDotEq))
        {
            let inclusive = self.at(T::DotDotEq);
            self.bump();
            let hi = self.parse_pattern_primary();
            return Pattern {
                id: self.nid(),
                kind: PatKind::Range {
                    lo: Box::new(lo),
                    hi: Box::new(hi),
                    inclusive,
                },
                span: start.to(self.prev_span()),
            };
        }
        lo
    }

    /// Peek (no consumption) whether the tokens from the current position form
    /// `Ident (:: Ident)+ {` — a multi-segment path immediately followed by a struct-literal brace,
    /// i.e. an `Enum::Variant { … }` struct-payload literal. Used to disambiguate it from a bare
    /// `Enum::Variant` value / `Enum::Variant(..)` call, which take the postfix `::` path instead.
    /// Requires at least one `:: Ident` segment (a single `Name {` is already handled directly).
    fn colon_path_before_brace(&self) -> bool {
        // Current token is the leading `Ident`.
        if self.nth(1) != T::ColonColon || self.nth(2) != T::Ident {
            return false;
        }
        let mut k = 3;
        while self.nth(k) == T::ColonColon && self.nth(k + 1) == T::Ident {
            k += 2;
        }
        self.nth(k) == T::LBrace
    }

    /// Parse a `::`-separated path (`Enum::Variant`), used by enum-variant patterns.
    fn parse_colon_path(&mut self) -> Path {
        let start = self.span();
        let mut segments = vec![self.ident()];
        while self.at(T::ColonColon) && self.nth(1) == T::Ident {
            self.bump(); // ::
            segments.push(self.ident());
        }
        Path {
            segments,
            span: start.to(self.prev_span()),
        }
    }

    /// The field list of a struct or struct-variant pattern, after its `{`: `f` (shorthand for
    /// `f: f`), `f: sub`, comma-separated, optionally closed by `..`. Returns the fields and whether
    /// a `..` was written. The `}` is the caller's.
    fn parse_field_pats(&mut self) -> (Vec<FieldPat>, bool) {
        let mut fps = Vec::new();
        let mut rest = false;
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            if self.eat(T::DotDot) {
                rest = true;
                break;
            }
            let fname = self.ident();
            let pat = if self.eat(T::Colon) {
                self.parse_pattern()
            } else {
                Pattern {
                    id: self.nid(),
                    kind: PatKind::Ident(fname.sym),
                    span: fname.span,
                }
            };
            fps.push(FieldPat {
                name: fname.sym,
                pat,
            });
            if !self.eat(T::Comma) {
                break;
            }
        }
        (fps, rest)
    }

    fn parse_pattern_primary(&mut self) -> Pattern {
        let start = self.span();
        // Bound deeply nested tuple patterns (`((((…))))`), which recurse through here via the
        // `LParen` arm, the same way expressions and types are bounded.
        let saved = self.depth;
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            self.too_deep(start);
            self.depth = saved;
            return Pattern {
                id: self.nid(),
                kind: PatKind::Wildcard,
                span: start,
            };
        }
        let kind = match self.kind() {
            T::Ident => {
                let text = &self.src[start.lo as usize..start.hi as usize];
                if text == "_" {
                    self.bump();
                    PatKind::Wildcard
                } else if self.nth(1) == T::ColonColon {
                    // `Enum::Variant` — an enum-variant pattern. A following `(` / `{` destructures a
                    // data-carrying (tagged-union) payload; a bare path is a unit variant / C-style
                    // enum, matched by discriminant.
                    let path = self.parse_colon_path();
                    if self.at(T::LParen) {
                        // `Enum::Variant(p0, p1, …)` — positional (tuple) payload sub-patterns.
                        self.bump(); // (
                        let mut subs = Vec::new();
                        while !self.at(T::RParen) && !self.at(T::Eof) {
                            subs.push(self.parse_pattern());
                            if !self.eat(T::Comma) {
                                break;
                            }
                        }
                        self.expect(T::RParen);
                        PatKind::Variant {
                            path,
                            fields: VariantPat::Tuple(subs),
                        }
                    } else if self.at(T::LBrace) {
                        // `Enum::Variant { f0, f1: sub, .. }` — named (struct) payload sub-patterns.
                        // Bare `{ r }` is field shorthand binding `r`; `{ r: sub }` binds via `sub`.
                        // An unlisted field is unconstrained, so a closing `..` changes nothing.
                        self.bump(); // {
                        let (fps, _rest) = self.parse_field_pats();
                        self.expect(T::RBrace);
                        PatKind::Variant {
                            path,
                            fields: VariantPat::Struct(fps),
                        }
                    } else {
                        PatKind::Path(path)
                    }
                } else if self.nth(1) == T::At {
                    // `name @ sub` — bind the whole value and match it against `sub`.
                    let name = self.intern_span(start);
                    self.bump(); // name
                    self.bump(); // @
                    let sub = self.parse_pattern_range();
                    PatKind::Binding {
                        name,
                        sub: Box::new(sub),
                    }
                } else if self.nth(1) == T::LBrace {
                    // `Point { x, y: 0, .. }` — a struct pattern. `Name {` cannot begin anything
                    // else in pattern position (a binding is followed by `=>`, `=`, `:`, `in`, `|`,
                    // `@` or `if`). `Self` names the implementing type inside an `impl`.
                    let name = self.ident();
                    let path = Path {
                        segments: vec![name],
                        span: name.span,
                    };
                    self.bump(); // {
                    let (fields, rest) = self.parse_field_pats();
                    self.expect(T::RBrace);
                    PatKind::Struct { path, fields, rest }
                } else {
                    let sym = self.intern_span(start);
                    self.bump();
                    PatKind::Ident(sym)
                }
            }
            // `[a, b, .., z]` — an array pattern: the first elements, optionally a `..` standing
            // for the unconstrained middle, and the last elements.
            T::LBracket => {
                self.bump(); // [
                let mut prefix = Vec::new();
                let mut suffix = Vec::new();
                let mut rest = false;
                while !self.at(T::RBracket) && !self.at(T::Eof) {
                    if !rest && self.eat(T::DotDot) {
                        rest = true;
                    } else if rest {
                        suffix.push(self.parse_pattern());
                    } else {
                        prefix.push(self.parse_pattern());
                    }
                    if !self.eat(T::Comma) {
                        break;
                    }
                }
                self.expect(T::RBracket);
                PatKind::Slice {
                    prefix,
                    rest,
                    suffix,
                }
            }
            T::LParen => {
                self.bump();
                if self.eat(T::RParen) {
                    PatKind::Unit
                } else {
                    let mut subs = vec![self.parse_pattern()];
                    while self.eat(T::Comma) {
                        if self.at(T::RParen) {
                            break;
                        }
                        subs.push(self.parse_pattern());
                    }
                    self.expect(T::RParen);
                    PatKind::Tuple(subs)
                }
            }
            // Integer literal pattern (a `match` arm like `0 =>` / `1 =>`).
            T::Int => {
                let sym = self.intern_span(start);
                self.bump();
                PatKind::Int { sym, neg: false }
            }
            // Char-literal pattern (`'a' =>`). `char` is comparable / usable in arithmetic, so it is
            // a valid literal pattern; it decodes to a code point and matches like an integer.
            T::Char => {
                let sym = self.intern_span(start);
                self.bump();
                PatKind::Char(sym)
            }
            // Negative integer literal pattern (`-1 =>`); fold the sign into the pattern since a
            // literal pattern has no sub-expression to negate.
            T::Minus => {
                self.bump();
                if self.at(T::Int) {
                    let isp = self.span();
                    let sym = self.intern_span(isp);
                    self.bump();
                    PatKind::Int { sym, neg: true }
                } else {
                    self.error(
                        start,
                        "E0206",
                        format!("expected an integer after `-`, found {}", self.found()),
                    );
                    PatKind::Wildcard
                }
            }
            T::True => {
                self.bump();
                PatKind::Bool(true)
            }
            T::False => {
                self.bump();
                PatKind::Bool(false)
            }
            _ => {
                self.error(
                    start,
                    "E0206",
                    format!("expected a pattern, found {}", self.found()),
                );
                self.bump();
                PatKind::Wildcard
            }
        };
        self.depth = saved;
        Pattern {
            id: self.nid(),
            kind,
            span: start.to(self.prev_span()),
        }
    }

    // ---- Attributes (also used by the item level in `items.rs`) ----

    pub(crate) fn parse_attrs(&mut self) -> Vec<Attr> {
        let mut attrs = Vec::new();
        while self.at(T::At) {
            attrs.push(self.parse_attr());
        }
        attrs
    }

    fn parse_attr(&mut self) -> Attr {
        let start = self.span();
        self.bump(); // @
        let name = self.ident_like();
        let mut args = Vec::new();
        if self.eat(T::LParen) {
            while !self.at(T::RParen) && !self.at(T::Eof) {
                args.push(self.parse_attr_arg());
                if !self.eat(T::Comma) {
                    break;
                }
            }
            self.expect(T::RParen);
        }
        Attr {
            name,
            args,
            span: start.to(self.prev_span()),
        }
    }

    fn parse_attr_arg(&mut self) -> AttrArg {
        let span = self.span();
        match self.kind() {
            T::Int => {
                let s = self.intern_span(span);
                self.bump();
                AttrArg::Int(s, span)
            }
            T::Str => {
                let s = self.intern_span(span);
                self.bump();
                AttrArg::Str(s, span)
            }
            T::Ident => {
                let key = self.ident();
                if self.eat(T::Eq) {
                    let value = self.parse_attr_val();
                    AttrArg::KeyValue { key, value }
                } else {
                    AttrArg::Word(key.sym, span)
                }
            }
            _ => {
                self.error(span, "E0207", "expected an attribute argument");
                self.bump();
                AttrArg::Word(self.interner.intern("«error»"), span)
            }
        }
    }

    fn parse_attr_val(&mut self) -> AttrVal {
        let span = self.span();
        match self.kind() {
            T::Int => {
                let s = self.intern_span(span);
                self.bump();
                AttrVal::Int(s)
            }
            T::Str => {
                let s = self.intern_span(span);
                self.bump();
                AttrVal::Str(s)
            }
            T::True => {
                self.bump();
                AttrVal::Bool(true)
            }
            T::False => {
                self.bump();
                AttrVal::Bool(false)
            }
            // A missing value (`@parallel(grain = )`) must not consume the delimiter that ends the
            // argument list: bumping the `)` stored it as the attribute's value and pushed the
            // "expected `)`" report onto the next line's token, so the user saw an error about the
            // following statement. Report here and leave the delimiter for `parse_attr`.
            T::RParen | T::Comma | T::Eof => {
                self.error(
                    span,
                    "E0207",
                    "expected a value after `=` in this attribute argument",
                );
                AttrVal::Word(self.interner.intern("«error»"))
            }
            _ => {
                let s = self.intern_span(span);
                self.bump();
                AttrVal::Word(s)
            }
        }
    }
}

/// Decode an integer literal's source text as an unsigned value. Handles the radix prefixes
/// `0x`/`0o`/`0b` (case-insensitive), digit separators `_`, and an explicit integer type suffix
/// (`16usize`). Returns `None` for a negative, malformed, or out-of-`u64`-range literal.
///
/// MIRROR: must accept exactly the literal grammar `wukong_sema::parse_u64_text` and
/// `wukong_mir_build::parse_int` accept. A decoder that silently drops a prefix here turns the
/// extent the user wrote into a different one with no diagnostic at all.
fn parse_u64_literal(text: &str) -> Option<u64> {
    let mut s = text.trim();
    if s.starts_with('-') {
        return None;
    }
    if s.starts_with('+') {
        s = &s[1..];
    }
    // Longest-first so `usize`/`isize` win over a shorter prefix. Every suffix letter is a
    // non-hex-digit, so this cannot truncate a hex literal's digits.
    for suf in [
        "usize", "isize", "u128", "i128", "u64", "i64", "u32", "i32", "u16", "i16", "u8", "i8",
    ] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
            break;
        }
    }
    // Extents without digit separators can keep their source slice borrowed;
    // only normalized spellings need their own storage.
    let normalized;
    let body = if s.contains('_') {
        normalized = s.replace('_', "");
        normalized.as_str()
    } else {
        s
    };
    if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()
    } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
        u64::from_str_radix(o, 8).ok()
    } else if let Some(b) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        u64::from_str_radix(b, 2).ok()
    } else {
        body.parse::<u64>().ok()
    }
}

/// The names `p` binds, each once, in the order they first appear (an or-pattern's alternatives
/// bind the same names, so the first is read), with the span to cite each by.
fn pattern_bindings(p: &Pattern, out: &mut Vec<(Symbol, Span)>) {
    fn push(sym: Symbol, span: Span, out: &mut Vec<(Symbol, Span)>) {
        if !out.iter().any(|(s, _)| *s == sym) {
            out.push((sym, span));
        }
    }
    match &p.kind {
        PatKind::Ident(s) => push(*s, p.span, out),
        PatKind::Binding { name, sub } => {
            push(*name, p.span, out);
            pattern_bindings(sub, out);
        }
        PatKind::Tuple(subs) => subs.iter().for_each(|s| pattern_bindings(s, out)),
        PatKind::Or(alts) => {
            if let Some(first) = alts.first() {
                pattern_bindings(first, out);
            }
        }
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => subs.iter().for_each(|s| pattern_bindings(s, out)),
            VariantPat::Struct(fps) => fps.iter().for_each(|fp| pattern_bindings(&fp.pat, out)),
        },
        PatKind::Struct { fields, .. } => {
            fields.iter().for_each(|fp| pattern_bindings(&fp.pat, out))
        }
        PatKind::Slice { prefix, suffix, .. } => prefix
            .iter()
            .chain(suffix)
            .for_each(|s| pattern_bindings(s, out)),
        PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => {}
    }
}

/// Split a float-token text of the form `N.M` (two non-empty runs of ASCII digits separated by a
/// single `.`, with no exponent or type suffix) into the tuple-index pair `(N, M)`. This recovers
/// the two indices the lexer glues together in a nested tuple-field access like `t.0.0` (it lexes
/// the trailing `0.0` as one float literal). Returns `None` for any genuine float (one with an
/// exponent, a suffix, or a missing side), which is not a valid tuple-index pair.
fn split_tuple_float(text: &str) -> Option<(u32, u32)> {
    let (a, b) = text.split_once('.')?;
    if a.is_empty() || b.is_empty() {
        return None;
    }
    if !a.bytes().all(|c| c.is_ascii_digit()) || !b.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn token_to_binop(k: TokenKind) -> Option<BinOp> {
    use BinOp as B;
    Some(match k {
        T::Plus => B::Add,
        T::Minus => B::Sub,
        T::Star => B::Mul,
        T::Slash => B::Div,
        T::Percent => B::Rem,
        T::Amp => B::BitAnd,
        T::Pipe => B::BitOr,
        T::Caret => B::BitXor,
        T::Shl => B::Shl,
        T::Shr => B::Shr,
        T::EqEq => B::Eq,
        T::Ne => B::Ne,
        T::Lt => B::Lt,
        T::Le => B::Le,
        T::Gt => B::Gt,
        T::Ge => B::Ge,
        T::AmpAmp => B::And,
        T::PipePipe => B::Or,
        _ => return None,
    })
}

fn binop_bp(op: BinOp) -> u8 {
    use BinOp::*;
    match op {
        Or => 1,
        And => 2,
        Eq | Ne | Lt | Le | Gt | Ge => 3,
        BitOr => 4,
        BitXor => 5,
        BitAnd => 6,
        Shl | Shr => 7,
        Add | Sub => 8,
        Mul | Div | Rem => 9,
    }
}

/// An expression that ends its own statement, as in Rust: a `{ … }` block, `if`, `match` or
/// `loop`. Statement position needs no `;` after one.
fn is_block_like(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::Block(_) | ExprKind::If { .. } | ExprKind::Match { .. } | ExprKind::Loop { .. }
    )
}

/// The opening delimiter a closing one matches (`)` → `(`), or `None` for any other token.
fn opener_of(k: TokenKind) -> Option<TokenKind> {
    Some(match k {
        T::RParen => T::LParen,
        T::RBracket => T::LBracket,
        T::RBrace => T::LBrace,
        _ => return None,
    })
}

/// "`)`", "one of `,` or `)`", "one of `,`, `.` or `)`" — a set of tokens for an "expected …"
/// message, in the order the parser tried them. A lexical category reads as its noun
/// ("identifier"), a fixed token as its glyph in backticks.
fn describe_set(kinds: &[TokenKind]) -> String {
    let names: Vec<String> = kinds
        .iter()
        .map(|k| match k.glyph() {
            Some(g) => format!("`{g}`"),
            None => k.describe().to_string(),
        })
        .collect();
    match names.as_slice() {
        [] => "a token".to_string(),
        [one] => one.clone(),
        [init @ .., last] => format!("one of {} or {last}", init.join(", ")),
    }
}

fn cur_assign_op(k: TokenKind) -> Option<AssignOp> {
    use AssignOp as A;
    Some(match k {
        T::Eq => A::Assign,
        T::PlusEq => A::Add,
        T::MinusEq => A::Sub,
        T::StarEq => A::Mul,
        T::SlashEq => A::Div,
        T::PercentEq => A::Rem,
        T::AmpEq => A::BitAnd,
        T::PipeEq => A::BitOr,
        T::CaretEq => A::BitXor,
        T::ShlEq => A::Shl,
        T::ShrEq => A::Shr,
        _ => return None,
    })
}

/// Recognize a SIMD vector identifier like `f32x8` -> ("f32", 8).
fn split_vector_ident(s: &str) -> Option<(&str, u32)> {
    let idx = s.rfind('x')?;
    if idx == 0 || idx + 1 >= s.len() {
        return None;
    }
    let (left, right) = (&s[..idx], &s[idx + 1..]);
    if !right.bytes().all(|b| b.is_ascii_digit()) || !is_scalar_name(left) {
        return None;
    }
    Some((left, right.parse().ok()?))
}

fn is_scalar_name(s: &str) -> bool {
    matches!(
        s,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "isize"
            | "f16"
            | "bf16"
            | "f32"
            | "f64"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_ast::print;

    fn expr(src: &str) -> String {
        let mut i = Interner::new();
        let (e, diags) = parse_expr_str(src, SourceId(0), &mut i);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
        // Render via a tiny module wrapper-free path using the inline expr printer.
        print::print_expr(&e, &i)
    }

    fn ty(src: &str) -> String {
        let mut i = Interner::new();
        let (t, diags) = parse_type_str(src, SourceId(0), &mut i);
        assert!(diags.is_empty(), "unexpected diagnostics: {diags:?}");
        print::print_type(&t, &i)
    }

    #[test]
    fn empty_index_is_rejected() {
        // `a[]` built `Index { indices: [] }`, which sema's array arm returned the element type for
        // unchecked and mir_build then refused with the generic
        // `C0001 'expression' is not yet supported by codegen` — the compiler reporting itself as
        // incomplete for what is a missing index.
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str("a[]", SourceId(0), &mut i);
        assert!(
            d.iter().any(|x| x.code == Some("E0202")),
            "`a[]` must be an E0202: {d:?}"
        );
        // A real index, and a multi-index, still parse clean.
        for src in ["a[0]", "a[i, j]", "a[i][j]"] {
            let mut i = Interner::new();
            let (_e, d) = parse_expr_str(src, SourceId(0), &mut i);
            assert!(d.is_empty(), "`{src}` must parse clean: {d:?}");
        }
    }

    #[test]
    fn precedence_and_assoc() {
        // `*` binds tighter than `+`; both left-assoc.
        assert_eq!(
            expr("1 + 2 * 3").trim(),
            "binary +\n  int 1\n  binary *\n    int 2\n    int 3"
        );
    }

    #[test]
    fn unary_and_postfix() {
        // -a.b parses as -(a.b); calls and indices chain.
        assert_eq!(expr("-a.b").lines().next().unwrap(), "unary -");
        let s = expr("f32x8::load(p)");
        assert!(s.contains("call"));
        assert!(s.contains("field load"));
    }

    #[test]
    fn cast_binds_looser_than_prefix_tighter_than_binary() {
        // `*p as i32` is `(*p) as i32`, NOT `*(p as i32)` (the old bug, which lowered to a
        // ptrtoint + load). The cast must be the outermost node, with the deref nested inside.
        let s = expr("*p as i32");
        assert_eq!(s.lines().next().unwrap(), "cast i32");
        assert!(s.contains("unary *"), "deref must nest under the cast: {s}");
        // `a + b as i32` is `a + (b as i32)` — `as` binds tighter than `+`.
        assert_eq!(expr("a + b as i32").lines().next().unwrap(), "binary +");
        // `-x as i32` is `(-x) as i32`.
        assert_eq!(expr("-x as i32").lines().next().unwrap(), "cast i32");
    }

    #[test]
    fn turbofish_call() {
        let s = expr("matmul::<512, 512, 512>(a, b, c)");
        assert!(s.contains("call"));
        assert!(s.contains("generic 512"));
    }

    #[test]
    fn multi_index() {
        let s = expr("a[i, k]");
        assert!(s.starts_with("index"));
    }

    #[test]
    fn types_render() {
        assert_eq!(ty("*mut f32"), "*mut f32");
        assert_eq!(ty("f32x8"), "f32x8");
        assert_eq!(ty("Tensor[f32, M, N]"), "Tensor[f32, M, N]");
        assert_eq!(
            ty("Tensor[f32, 512, 512, .col_major]"),
            "Tensor[f32, 512, 512, .col_major]"
        );
        assert_eq!(ty("[]f32"), "[]f32");
    }

    /// Extents written in a type annotation are decoded with the full integer-literal grammar, and a
    /// literal the decoder cannot represent is a diagnostic — never a silent 0. `Tensor[f32, 0x10]`
    /// used to parse as a 0-element tensor and `vec[f32, 4294967304]` as `f32x8`.
    #[test]
    fn radix_and_out_of_range_extents() {
        assert_eq!(
            ty("Tensor[f32, 0x10, 0b101, 0o17]"),
            "Tensor[f32, 16, 5, 15]"
        );
        assert_eq!(ty("Tensor[f32, 1_000, 16usize]"), "Tensor[f32, 1000, 16]");
        assert_eq!(ty("vec[f32, 0x8]"), "f32x8");

        let bad = |src: &str| -> Vec<Diagnostic> {
            let mut i = Interner::new();
            parse_type_str(src, SourceId(0), &mut i).1
        };
        // A 26-digit dimension and a lane count past `u32` are rejected, not truncated.
        for src in [
            "Tensor[f32, 99999999999999999999999999]",
            "vec[f32, 4294967304]",
            "Tensor[f32, 8, .tiled(99999999999999999999999999)]",
        ] {
            let d = bad(src);
            assert!(
                d.iter().any(|x| x.code == Some("E0203")),
                "`{src}` should report E0203, got {d:?}"
            );
        }
    }

    /// A dimension may be arithmetic on dimensions, with the expression precedences (`* / %` over
    /// `+ -`, left-associative) and parentheses that only group. The printer makes the association
    /// explicit, so a mis-associated `(L - K)/S + 1` would show here.
    #[test]
    fn dimension_arithmetic_parses_with_expression_precedence() {
        assert_eq!(ty("Tensor[f32, N*2]"), "Tensor[f32, (N * 2)]");
        assert_eq!(
            ty("Tensor[f32, S, H*HD, 3*D]"),
            "Tensor[f32, S, (H * HD), (3 * D)]"
        );
        assert_eq!(
            ty("Tensor[f32, (L - K)/S + 1]"),
            "Tensor[f32, (((L - K) / S) + 1)]"
        );
        assert_eq!(ty("Tensor[f32, A - B - C]"), "Tensor[f32, ((A - B) - C)]");
        assert_eq!(ty("Tensor[f32, A + B * C]"), "Tensor[f32, (A + (B * C))]");
        assert_eq!(ty("Tensor[f32, N % 4, ?]"), "Tensor[f32, (N % 4), ?]");
        // The layout suffix still terminates the dim list.
        assert_eq!(
            ty("Tensor[f32, M*N, 2, .col_major]"),
            "Tensor[f32, (M * N), 2, .col_major]"
        );
        // Anything outside the closed algebra is still E0203.
        for src in [
            "Tensor[f32, N << 1]",
            "Tensor[f32, f(N)]",
            "Tensor[f32, N*]",
        ] {
            let mut i = Interner::new();
            let d = parse_type_str(src, SourceId(0), &mut i).1;
            assert!(
                d.iter()
                    .any(|x| x.code == Some("E0203") || x.code == Some("E0200")),
                "`{src}` should be rejected, got {d:?}"
            );
        }
    }

    /// A radix-prefixed tuple index selects the field the user wrote: `t.0x1` is field 1, not the
    /// field 0 the leading-decimal-digits decoder used to produce.
    #[test]
    fn radix_tuple_field_index() {
        assert_eq!(expr("t.0x1").lines().next().unwrap(), "tuple-field 1");
    }

    #[test]
    fn recovers_after_a_bad_item() {
        // A garbage token at module scope must not swallow the following valid functions: the
        // parser should report an error and still recover to parse both `a` and `b`.
        let mut i = Interner::new();
        let src = "module t\nfn a() -> i32 { return 1; }\n@#$\nfn b() -> i32 { return 2; }";
        let (module, diags) = parse_module(src, SourceId(0), &mut i);
        assert!(!diags.is_empty(), "expected at least one diagnostic");
        let fn_names: Vec<String> = module
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Fn(f) => Some(i.resolve(f.name.sym).to_string()),
                _ => None,
            })
            .collect();
        assert!(
            fn_names.contains(&"a".to_string()),
            "lost `a`: {fn_names:?}"
        );
        assert!(
            fn_names.contains(&"b".to_string()),
            "lost `b`: {fn_names:?}"
        );
    }

    /// Pathological deeply-nested input must report E0209 rather than overflow the stack. Each shape
    /// mirrors one of the historical crash repros. The work runs on a roomy stack so the test itself
    /// can build and drop the (depth-bounded) AST without overflowing — exactly as the real compiler
    /// front-end runs (see `wukongc::main`); the guard is what keeps the depth bounded.
    #[test]
    fn deeply_nested_input_reports_e0209_not_stack_overflow() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let has_e0209 =
                    |diags: &[Diagnostic]| diags.iter().any(|d| d.code == Some("E0209"));

                // Case 3: a 5000-long left-associative `+` chain (iteratively built deep tree).
                let chain = format!("1{}", "+1".repeat(5000));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&chain, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "long `+` chain should report E0209, got {d:?}"
                );

                // Case 1: 4000 nested parentheses (recursive descent).
                let parens = format!("{}1{}", "(".repeat(4000), ")".repeat(4000));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&parens, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "nested parens should report E0209, got {d:?}"
                );

                // Case 2: 4000 nested array types (recursive `parse_type`).
                let ty = format!("{}i32{}", "[".repeat(4000), "; 1]".repeat(4000));
                let mut i = Interner::new();
                let (_t, d) = parse_type_str(&ty, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "nested array type should report E0209, got {d:?}"
                );

                // A single clean diagnostic, not a cascade: the parens case reports E0209 once.
                let parens = format!("{}1{}", "(".repeat(4000), ")".repeat(4000));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&parens, SourceId(0), &mut i);
                assert_eq!(
                    d.iter().filter(|x| x.is_error()).count(),
                    1,
                    "depth overflow should produce exactly one error, got {d:?}"
                );
            })
            .expect("spawn parser thread")
            .join()
            .expect("the parser must not overflow its stack on deeply nested input");
    }

    /// A long *postfix* chain (`a.b.b.b…`, `a[0][0]…`, `f()()…`, `a::b::b…`) builds the same kind of
    /// deep left-leaning tree the binop fold does, and used to be the one fold that never charged
    /// the depth budget: 400000 `.b` overflowed the compiler's stack (exit 127, zero diagnostics)
    /// instead of reporting E0209.
    #[test]
    fn deep_postfix_chain_reports_e0209() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let has_e0209 =
                    |diags: &[Diagnostic]| diags.iter().any(|d| d.code == Some("E0209"));
                let n = 5000;
                for chain in [
                    format!("a{}", ".b".repeat(n)),
                    format!("a{}", "[0]".repeat(n)),
                    format!("f{}", "()".repeat(n)),
                    format!("a{}", "::b".repeat(n)),
                ] {
                    let mut i = Interner::new();
                    let (_e, d) = parse_expr_str(&chain, SourceId(0), &mut i);
                    assert!(
                        has_e0209(&d),
                        "long postfix chain should report E0209, got {d:?}"
                    );
                }

                // A realistic postfix chain stays far under the limit.
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str("a.b.c[0].d(1).e", SourceId(0), &mut i);
                assert!(
                    d.is_empty(),
                    "short postfix chain must parse cleanly: {d:?}"
                );
            })
            .expect("spawn parser thread")
            .join()
            .expect("the parser must not overflow its stack on a long postfix chain");
    }

    /// Pathological *structural* nesting — deep `{ … }` blocks, `if`/`else if` chains, `match` arm
    /// bodies, and loop bodies — must report E0209 quickly instead of hanging (the depth guard used
    /// to leave these recursions unbounded: a brace/match nest spun the block/arm loop forever on an
    /// unconsumed token, while an `else if` chain recursed with no depth charge at all).
    #[test]
    fn deeply_nested_structural_input_reports_e0209() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let has_e0209 =
                    |diags: &[Diagnostic]| diags.iter().any(|d| d.code == Some("E0209"));
                let n = 3000;

                // Nested blocks `{ { { … 0 … } } }`.
                let blocks = format!("{}0{}", "{".repeat(n), "}".repeat(n));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&blocks, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "nested blocks should report E0209, got {d:?}"
                );

                // A long `if … else if … else if …` chain (direct `parse_if` recursion).
                let elifs = format!("{}{{ 0 }}", "if true { 0 } else ".repeat(n));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&elifs, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "deep else-if chain should report E0209, got {d:?}"
                );

                // A `match` whose arm body is another `match`, nested deep.
                let matches = format!("{}0{}", "match 0 { _ => ".repeat(n), " }".repeat(n));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&matches, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "nested match arms should report E0209, got {d:?}"
                );

                // Nested loop bodies `while … { while … { … } }` (statement form, via parse_module).
                let whiles = format!(
                    "fn f() {{ {}{} }}",
                    "while true { ".repeat(n),
                    "}".repeat(n)
                );
                let mut i = Interner::new();
                let (_m, d) = parse_module(&whiles, SourceId(0), &mut i);
                assert!(
                    has_e0209(&d),
                    "nested loop bodies should report E0209, got {d:?}"
                );
            })
            .expect("spawn parser thread")
            .join()
            .expect("the parser must not overflow its stack on deeply nested structural input");
    }

    /// A failed `expect` names every token the parser tried at that position, not just the last.
    #[test]
    fn expected_token_sets_name_every_alternative() {
        let first = |src: &str| -> String {
            let mut i = Interner::new();
            let (_e, d) = parse_expr_str(src, SourceId(0), &mut i);
            assert_eq!(d[0].code, Some("E0200"), "{src}: {d:?}");
            d[0].message.clone()
        };
        assert_eq!(
            first("f(a b)"),
            "expected one of `,` or `)`, found identifier `b`"
        );
        assert_eq!(
            first("[1, 2 3]"),
            "expected one of `,` or `]`, found integer literal"
        );
        assert_eq!(
            first("(1, 2 3)"),
            "expected one of `,` or `)`, found integer literal"
        );
        // A lone requirement still reads as one token.
        assert_eq!(first("(1 + 2;"), "expected `)`, found `;`");
    }

    /// A missing closer at the end of a line is reported where it is missing — just after the last
    /// token of that line — not on the first token of the next line, which is what the parser had
    /// moved on to. The token it ran into and the unclosed opener are secondary labels.
    #[test]
    fn missing_closer_is_anchored_after_the_previous_line() {
        let src = "fn main() -> i32 {\n    let x: i32 = f(1, 2\n    return x;\n}\n";
        let mut i = Interner::new();
        let (_m, d) = parse_module(src, SourceId(0), &mut i);
        let e = &d[0];
        assert_eq!(e.code, Some("E0200"), "{d:?}");
        assert_eq!(e.message, "expected one of `,` or `)`, found `return`");
        let after_two = (src.find("2\n").unwrap() + 1) as u32;
        let primary = e.primary_span().unwrap();
        assert_eq!((primary.lo, primary.hi), (after_two, after_two));
        let at = |needle: &str| src.find(needle).unwrap() as u32;
        let secondary: Vec<u32> = e
            .labels
            .iter()
            .filter(|l| !l.primary)
            .map(|l| l.span.lo)
            .collect();
        assert!(secondary.contains(&at("return")), "{e:?}");
        assert!(
            secondary.contains(&at("(1, 2")),
            "the opener must be labelled: {e:?}"
        );
    }

    /// An unclosed `{` used to be "expected `}`, found end of file" pointing at the end of the
    /// file and nowhere else; the brace that was never closed is now labelled.
    #[test]
    fn unclosed_brace_labels_its_opener() {
        let src = "fn main() -> i32 {\n    if true {\n        return 1;\n    }\n    return 0;\n";
        let mut i = Interner::new();
        let (_m, d) = parse_module(src, SourceId(0), &mut i);
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].message, "expected `}`, found end of file");
        let opener = src.find('{').unwrap() as u32;
        assert!(
            d[0].labels
                .iter()
                .any(|l| !l.primary && l.span.lo == opener && l.message.contains("not closed")),
            "{:?}",
            d[0]
        );
    }

    /// A character the lexer rejects costs exactly one diagnostic: the parser no longer adds
    /// "expected an expression, found invalid token" (or a type / item / pattern variant of it) at
    /// the same place.
    #[test]
    fn a_stray_character_costs_one_diagnostic() {
        for src in [
            "fn main() -> i32 {\n    let x: i32 = 1 $ 2;\n    return x;\n}\n",
            "fn f(x: $) {}\n",
            "$\nfn f() {}\n",
            "fn f() -> i32 { match 1 { $ => 0, _ => 1 } }\n",
            "fn f() -> i32 { return g(1 $); }\n",
        ] {
            let mut i = Interner::new();
            let (_m, d) = parse_module(src, SourceId(0), &mut i);
            assert_eq!(
                d.iter().map(|x| x.code).collect::<Vec<_>>(),
                vec![Some("E0101")],
                "{src}"
            );
        }
    }

    /// Moderately nested but entirely realistic input stays well under the limit and parses cleanly.
    #[test]
    fn moderate_nesting_parses_cleanly() {
        // A 100-term sum.
        let sum = format!("1{}", "+1".repeat(99));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&sum, SourceId(0), &mut i);
        assert!(d.is_empty(), "100-term sum should parse cleanly, got {d:?}");

        // 50-deep parentheses.
        let parens = format!("{}1{}", "(".repeat(50), ")".repeat(50));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&parens, SourceId(0), &mut i);
        assert!(
            d.is_empty(),
            "50-deep parens should parse cleanly, got {d:?}"
        );

        // A 16-deep array type.
        let ty = format!("{}i32{}", "[".repeat(16), "; 1]".repeat(16));
        let mut i = Interner::new();
        let (_t, d) = parse_type_str(&ty, SourceId(0), &mut i);
        assert!(
            d.is_empty(),
            "16-deep array type should parse cleanly, got {d:?}"
        );

        // 64-deep blocks, a 64-arm-deep else-if chain, a 64-deep match nest, and 64-deep loop
        // bodies all sit far under the limit — the new structural guards must not reject them.
        let blocks = format!("{}0{}", "{".repeat(64), "}".repeat(64));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&blocks, SourceId(0), &mut i);
        assert!(
            d.is_empty(),
            "64-deep blocks should parse cleanly, got {d:?}"
        );

        let elifs = format!("{}{{ 0 }}", "if true { 0 } else ".repeat(64));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&elifs, SourceId(0), &mut i);
        assert!(
            d.is_empty(),
            "64-deep else-if chain should parse cleanly, got {d:?}"
        );

        let matches = format!("{}0{}", "match 0 { _ => ".repeat(64), " }".repeat(64));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&matches, SourceId(0), &mut i);
        assert!(
            d.is_empty(),
            "64-deep match nest should parse cleanly, got {d:?}"
        );

        // A wide-but-shallow block (many sequential statements) must not accumulate depth.
        let wide = format!("{{ {} 0 }}", "let x = 1; ".repeat(500));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&wide, SourceId(0), &mut i);
        assert!(
            d.is_empty(),
            "wide shallow block should parse cleanly, got {d:?}"
        );
    }

    /// Parse `src` as a module through the driver's entry point; return the error codes and the
    /// lint codes separately.
    fn lint_codes(src: &str) -> (Vec<&'static str>, Vec<&'static str>) {
        let mut i = Interner::new();
        let (tokens, lexed) = wukong_lexer::tokenize(src, SourceId(0));
        let out = parse_module_tokens_with_lints(&tokens, src, &mut i, 0);
        let errors = lexed
            .iter()
            .chain(&out.diags)
            .filter_map(|d| d.code)
            .collect();
        assert!(out
            .lints
            .iter()
            .all(|d| d.severity == wukong_diag::Severity::Warning));
        (errors, out.lints.iter().filter_map(|d| d.code).collect())
    }

    /// `W0030`: a statement's missing `;` is a warning (the parse is unchanged), anchored just
    /// after the statement; the Rust exemptions hold — no `;` is needed after a block-like
    /// statement, nor before the closing `}` for anything but a `let`.
    #[test]
    fn missing_semicolon_is_w0030_with_rust_exemptions() {
        let src =
            "fn main() -> i32 {\n    let a: i32 = 3\n    print(a)\n    a = 4\n    return a\n}\n";
        let (errors, lints) = lint_codes(src);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            lints,
            vec!["W0030", "W0030", "W0030"],
            "let, print(a), a = 4"
        );
        let mut i = Interner::new();
        let (tokens, _) = wukong_lexer::tokenize(src, SourceId(0));
        let out = parse_module_tokens_with_lints(&tokens, src, &mut i, 0);
        let after_three = (src.find("3\n").unwrap() + 1) as u32;
        let at = out.lints[0].primary_span().unwrap();
        assert_eq!((at.lo, at.hi), (after_three, after_three));
        // Block-like statements, statements before `}`, and the old terminated forms are clean.
        let clean = "fn main() -> i32 {\n    let x: i32 = 1;\n    if x > 0 { print(x); }\n    \
                     loop { break }\n    match x { _ => { } }\n    while x < 0 { }\n    { print(2) }\n    \
                     return x\n}\n";
        assert_eq!(lint_codes(clean), (Vec::<&str>::new(), Vec::<&str>::new()));
        // A `let` needs its `;` even before `}`.
        assert_eq!(lint_codes("fn f() { let y: i32 = 2 }").1, vec!["W0030"]);
        // Two statements on one line: still a missing `;`.
        assert_eq!(
            lint_codes("fn f() { let y: i32 = 2 let z: i32 = 3; }").1,
            vec!["W0030"]
        );
    }

    /// N8: a `defer` statement ends with `;` like the other simple statements — it used to eat an
    /// optional `;` and report nothing — while a deferred block ends its own statement.
    #[test]
    fn defer_statement_is_checked_for_its_semicolon() {
        let src = "fn main() -> i32 {\n    defer print(1)\n    defer x += 1\n    defer { print(2); }\n    \
                   defer print(3);\n    return 0;\n}\n";
        let (errors, lints) = lint_codes(src);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            lints,
            vec!["W0030", "W0030"],
            "`defer print(1)`, `defer x += 1`"
        );
        // Before the closing `}` a `defer` (not a `let`) may leave its `;` out.
        assert_eq!(
            lint_codes("fn f() { defer print(1) }"),
            (Vec::<&str>::new(), Vec::<&str>::new())
        );
    }

    /// `W0030` carries the fix: a machine-applicable `;` inserted where the lint points.
    #[test]
    fn missing_semicolon_suggests_the_semicolon() {
        let src = "fn main() -> i32 {\n    let a: i32 = 3\n    return a;\n}\n";
        let mut i = Interner::new();
        let (tokens, _) = wukong_lexer::tokenize(src, SourceId(0));
        let out = parse_module_tokens_with_lints(&tokens, src, &mut i, 0);
        let d = &out.lints[0];
        assert_eq!(d.suggestions.len(), 1, "{d:?}");
        let s = &d.suggestions[0];
        assert_eq!(s.replacement, ";");
        assert_eq!(s.applicability, Applicability::MachineApplicable);
        let after_three = (src.find("3\n").unwrap() + 1) as u32;
        assert_eq!((s.span.lo, s.span.hi), (after_three, after_three));
    }

    /// A statement that is already broken — a parse error inside it, or a character the lexer
    /// rejected — is not also reported as missing its `;`.
    #[test]
    fn broken_statement_draws_no_w0030() {
        let (errors, lints) =
            lint_codes("fn main() -> i32 {\n    let x: i32 = 1 $ 2;\n    return x;\n}\n");
        assert_eq!(errors, vec!["E0101"]);
        assert_eq!(lints, Vec::<&str>::new());
        let (errors, lints) =
            lint_codes("fn main() -> i32 {\n    let x: i32 = f(1, 2\n    return x;\n}\n");
        assert_eq!(errors, vec!["E0200"]);
        assert_eq!(lints, Vec::<&str>::new());
    }

    /// `W0031`: a `(`, `[` or touching `-` that starts a line continues the expression above it.
    /// Inside `(…)`/`[…]` no statement can begin, so nothing is reported there; a `- y` with a space
    /// is the ordinary continued-subtraction style.
    #[test]
    fn line_leading_continuation_is_w0031() {
        let src = "fn main() -> i32 {\n    let x: i32 = f\n    (2);\n    let y: i32 = a\n    [0];\n    \
                   let z: i32 = x\n    -1;\n    let w: i32 = x * 10\n        - y;\n    return 0;\n}\n";
        let (errors, lints) = lint_codes(src);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(lints, vec!["W0031", "W0031", "W0031"], "`(`, `[`, `-1`");
        // The call still parses as a call: the lint changes no parse.
        let mut i = Interner::new();
        let (m, _) = parse_module(src, SourceId(0), &mut i);
        let ItemKind::Fn(f) = &m.items[0].kind else {
            panic!("expected a fn item")
        };
        let StmtKind::Let {
            init: Some(init), ..
        } = &f.body.as_ref().unwrap().stmts[0].kind
        else {
            panic!("expected `let x = …`")
        };
        assert!(matches!(init.kind, ExprKind::Call { .. }), "{init:?}");
        // Inside parentheses a line-leading `(` or `-1` is unambiguous.
        let inside =
            "fn main() -> i32 {\n    let x: i32 = g(f\n        (2),\n        3\n        -1);\n    \
                      return x;\n}\n";
        assert_eq!(lint_codes(inside), (Vec::<&str>::new(), Vec::<&str>::new()));
    }

    /// `if let P = e { … }` as the expression it parses to: its scrutinee and the first arm's guard.
    fn if_let_parts(src: &str) -> (Expr, Option<Expr>) {
        let mut i = Interner::new();
        let (e, d) = parse_expr_str(src, SourceId(0), &mut i);
        assert!(d.is_empty(), "{src}: {d:?}");
        let ExprKind::Match { scrutinee, arms } = e.kind else {
            panic!("`if let` parses to a match: {src}")
        };
        (*scrutinee, arms[0].guard.clone())
    }

    /// The `&&` operands of a left-folded chain, in order.
    fn and_operands(e: &Expr, out: &mut Vec<String>, i: &Interner) {
        match &e.kind {
            ExprKind::Binary {
                op: BinOp::And,
                lhs,
                rhs,
            } => {
                and_operands(lhs, out, i);
                out.push(print::print_expr(rhs, i));
            }
            _ => out.push(print::print_expr(e, i)),
        }
    }

    #[test]
    fn if_let_chains_become_the_arm_guard() {
        let mut i = Interner::new();
        let src = "if let E::A(x) = s && x > 2 && x < 9 && ok { 1 } else { 0 }";
        let (e, d) = parse_expr_str(src, SourceId(0), &mut i);
        assert!(d.is_empty(), "{d:?}");
        let ExprKind::Match { scrutinee, arms } = &e.kind else {
            panic!("`if let` parses to a match")
        };
        assert!(matches!(scrutinee.kind, ExprKind::Path(_)), "{scrutinee:?}");
        let guard = arms[0]
            .guard
            .as_ref()
            .expect("the chain is the first arm's guard");
        // Folded left: `((x > 2) && (x < 9)) && ok`.
        let ExprKind::Binary {
            op: BinOp::And,
            lhs,
            ..
        } = &guard.kind
        else {
            panic!("{guard:?}")
        };
        assert!(matches!(lhs.kind, ExprKind::Binary { op: BinOp::And, .. }));
        let mut ops = Vec::new();
        and_operands(guard, &mut ops, &i);
        assert_eq!(ops.len(), 3, "{ops:?}");
        assert!(arms[1].guard.is_none(), "the fall-through arm is unguarded");
        // No chain, no guard.
        assert!(if_let_parts("if let E::A(x) = s { x } else { 0 }")
            .1
            .is_none());
    }

    #[test]
    fn an_if_let_scrutinee_stops_before_and_and_or() {
        // `a == b` binds tighter than `&&`: it is the scrutinee, and `c` the guard.
        let (scrutinee, guard) = if_let_parts("if let true = a == b && c { 1 } else { 0 }");
        assert!(
            matches!(scrutinee.kind, ExprKind::Binary { op: BinOp::Eq, .. }),
            "{scrutinee:?}"
        );
        assert!(matches!(guard.map(|g| g.kind), Some(ExprKind::Path(_))));
        // `while let` takes the same chain.
        let mut i = Interner::new();
        let (_m, d) = parse_module(
            "fn f(n: i32) { let mut k = n; while let 1..=9 = k && k != 5 { k = k + 1; } }",
            SourceId(0),
            &mut i,
        );
        assert!(d.is_empty(), "{d:?}");
    }

    #[test]
    fn or_and_second_let_in_an_if_let_chain_are_e0200() {
        let codes = |src: &str| -> Vec<&'static str> {
            let mut i = Interner::new();
            let (_e, d) = parse_expr_str(src, SourceId(0), &mut i);
            d.into_iter().filter_map(|d| d.code).collect()
        };
        assert_eq!(codes("if let x = a || b { 1 } else { 0 }"), vec!["E0200"]);
        assert_eq!(
            codes("if let x = a && c || b || d { 1 } else { 0 }"),
            vec!["E0200"],
            "one report, and the parse recovers"
        );
        assert_eq!(
            codes("if let x = a && let y = b { 1 } else { 0 }"),
            vec!["E0200"]
        );
    }

    /// The entry points other than `parse_module_tokens_with_lints` never report a lint.
    #[test]
    fn plain_entry_points_drop_lints() {
        let mut i = Interner::new();
        let (_m, d) = parse_module("fn f() { let y: i32 = 2 }", SourceId(0), &mut i);
        assert!(d.is_empty(), "{d:?}");
    }
}
