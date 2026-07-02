//! `mercury_parser` — turns a token stream into an AST.
//!
//! Hand-written recursive descent for items, statements, and types; a Pratt (precedence-climbing)
//! parser for expressions. The parser allocates [`NodeId`]s, interns identifiers/literals, and
//! recovers from errors so a single mistake doesn't abort the whole parse. Items and the module
//! header are added in a later commit; this module covers types, expressions, and statements.

use mercury_ast::*;
use mercury_diag::Diagnostic;
use mercury_lexer::{Token, TokenKind};
use mercury_span::{Interner, SourceId, Span, Symbol};

use TokenKind as T;

mod items;
pub use items::{parse_module, parse_module_tokens};

/// Parse a standalone expression (for tests and a future REPL).
pub fn parse_expr_str(
    src: &str,
    source: SourceId,
    interner: &mut Interner,
) -> (Expr, Vec<Diagnostic>) {
    let (tokens, mut diags) = mercury_lexer::tokenize(src, source);
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
    let (tokens, mut diags) = mercury_lexer::tokenize(src, source);
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
        }
        t
    }

    fn eat(&mut self, k: TokenKind) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, k: TokenKind) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            let found = self.kind().describe();
            let want = k.glyph().unwrap_or_else(|| k.describe());
            let sp = self.span();
            self.error(sp, "E0200", format!("expected `{want}`, found {found}"));
            false
        }
    }

    fn error(&mut self, span: Span, code: &'static str, msg: impl Into<String>) {
        // Once the nesting limit is hit, a single E0209 is reported and every follow-on recovery
        // diagnostic (the unmatched delimiters and "expected …" errors produced while unwinding the
        // over-deep construct) is suppressed, keeping the output to one clean error.
        if self.depth_exceeded {
            return;
        }
        self.diags
            .push(Diagnostic::error(msg).with_code(code).primary(span, ""));
    }

    /// Maximum nesting depth of the recursive grammar productions before the parser bails with
    /// E0209. Generous enough that no realistic program (hand-written or generated) comes close, yet
    /// bounded so that neither the parser's own descent nor any later recursive walk over the AST can
    /// overflow the stack. The compiler front-end runs on a large stack (see `mercuryc::main`), so
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
            let sym = self.intern_span(span);
            Ident { sym, span }
        } else {
            let span = self.span();
            self.error(
                span,
                "E0201",
                format!("expected identifier, found {}", self.kind().describe()),
            );
            Ident {
                sym: self.interner.intern("«error»"),
                span,
            }
        }
    }

    /// Like [`ident`], but also accepts keyword tokens by their text. Used for attribute names
    /// such as `@extern` where the name collides with a keyword.
    fn ident_like(&mut self) -> Ident {
        if self.at(T::Ident) || self.kind().is_keyword() {
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
                    TypeKind::Array { elem, len }
                }
            }
            T::LParen => {
                self.bump();
                if self.eat(T::RParen) {
                    TypeKind::Unit
                } else {
                    let mut items = vec![self.parse_type()];
                    let mut trailing = false;
                    while self.eat(T::Comma) {
                        if self.at(T::RParen) {
                            trailing = true;
                            break;
                        }
                        items.push(self.parse_type());
                    }
                    self.expect(T::RParen);
                    if items.len() == 1 && !trailing {
                        items.pop().unwrap().kind
                    } else {
                        TypeKind::Tuple(items)
                    }
                }
            }
            T::Int => {
                let s = self.intern_span(self.span());
                self.bump();
                TypeKind::Int(s)
            }
            T::Ident => self.parse_named_type(),
            _ => {
                let sp = self.span();
                self.error(
                    sp,
                    "E0203",
                    format!("expected type, found {}", self.kind().describe()),
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
            let lanes = self.parse_int_value().unwrap_or(0) as u32;
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
        TypeKind::Path(self.parse_dotted_path())
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

    fn parse_dim(&mut self) -> Dim {
        let span = self.span();
        let kind = match self.kind() {
            T::Int => {
                let v = self.parse_int_value().unwrap_or(0);
                DimKind::Int(v)
            }
            T::Question => {
                self.bump();
                DimKind::Dynamic
            }
            T::Ident => {
                let sym = self.intern_span(span);
                self.bump();
                DimKind::Named(sym)
            }
            _ => {
                self.error(
                    span,
                    "E0203",
                    "expected a dimension (integer, name, or `?`)",
                );
                self.bump();
                DimKind::Dynamic
            }
        };
        Dim {
            kind,
            span: span.to(self.prev_span()),
        }
    }

    fn parse_layout(&mut self) -> Layout {
        self.bump(); // .
        let name_span = self.span();
        let name = self.ident();
        let text = self.interner.resolve(name.sym).to_string();
        match text.as_str() {
            "contiguous" => Layout::Contiguous,
            "col_major" => Layout::ColMajor,
            "strided" => Layout::Strided,
            "tiled" => {
                let mut sizes = Vec::new();
                if self.eat(T::LParen) {
                    while !self.at(T::RParen) && !self.at(T::Eof) {
                        if let Some(v) = self.parse_int_value() {
                            sizes.push(v);
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

    /// Read a (decimal) integer literal value at the cursor and consume it.
    fn parse_int_value(&mut self) -> Option<u64> {
        if !self.at(T::Int) {
            return None;
        }
        let text = &self.src[self.span().lo as usize..self.span().hi as usize];
        let digits: String = text
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '_')
            .collect();
        let v = digits.replace('_', "").parse().ok();
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
        loop {
            let Some(op) = token_to_binop(self.kind()) else {
                break;
            };
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
            let ty = self.parse_type();
            e = self.finish_expr(e.span, ExprKind::Cast { expr: Box::new(e), ty });
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
            match self.kind() {
                T::LParen => {
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
                    self.bump();
                    let mut indices = Vec::new();
                    while !self.at(T::RBracket) && !self.at(T::Eof) {
                        indices.push(self.parse_expr());
                        if !self.eat(T::Comma) {
                            break;
                        }
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
                        let index = self.parse_int_value().unwrap_or(0) as u32;
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
        while !self.at(T::Gt) && !self.at(T::Eof) {
            args.push(self.parse_type());
            if !self.eat(T::Comma) {
                break;
            }
        }
        self.expect(T::Gt);
        args
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
                    format!("expected an expression, found {}", self.kind().describe()),
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
            let stmt_start = self.span();
            let attrs = self.parse_attrs();
            match self.kind() {
                T::Let => {
                    let k = self.parse_let();
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
                    self.eat(T::Semi);
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
                    self.eat(T::Semi);
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
                    self.eat(T::Semi);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Break(label, value), stmt_start));
                }
                T::Continue => {
                    self.bump();
                    let label = if self.at(T::Label) {
                        Some(self.label_ident())
                    } else {
                        None
                    };
                    self.eat(T::Semi);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Continue(label), stmt_start));
                }
                T::Defer => {
                    self.bump();
                    let e = self.parse_expr();
                    self.eat(T::Semi);
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
                        self.eat(T::Semi);
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
        self.eat(T::Semi);
        StmtKind::Let {
            pat,
            mutable,
            ty,
            init,
        }
    }

    fn parse_while(&mut self, label: Option<Ident>) -> StmtKind {
        self.bump(); // while
        let cond = self.parse_cond();
        let body = self.parse_block();
        StmtKind::While {
            label,
            cond,
            body,
        }
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
    /// only after an integer-literal lower bound, so it never shadows `_`/identifier/tuple patterns.
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
                    // `Enum::Variant` — an enum-variant pattern (resolved to its discriminant).
                    PatKind::Path(self.parse_colon_path())
                } else {
                    let sym = self.intern_span(start);
                    self.bump();
                    PatKind::Ident(sym)
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
                        format!("expected an integer after `-`, found {}", self.kind().describe()),
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
                    format!("expected a pattern, found {}", self.kind().describe()),
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

    // ---- Attributes (also used by items in a later commit) ----

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
            _ => {
                let s = self.intern_span(span);
                self.bump();
                AttrVal::Word(s)
            }
        }
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
    use mercury_ast::print;

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
    /// front-end runs (see `mercuryc::main`); the guard is what keeps the depth bounded.
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
                assert!(has_e0209(&d), "long `+` chain should report E0209, got {d:?}");

                // Case 1: 4000 nested parentheses (recursive descent).
                let parens = format!("{}1{}", "(".repeat(4000), ")".repeat(4000));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&parens, SourceId(0), &mut i);
                assert!(has_e0209(&d), "nested parens should report E0209, got {d:?}");

                // Case 2: 4000 nested array types (recursive `parse_type`).
                let ty = format!("{}i32{}", "[".repeat(4000), "; 1]".repeat(4000));
                let mut i = Interner::new();
                let (_t, d) = parse_type_str(&ty, SourceId(0), &mut i);
                assert!(has_e0209(&d), "nested array type should report E0209, got {d:?}");

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
                assert!(has_e0209(&d), "nested blocks should report E0209, got {d:?}");

                // A long `if … else if … else if …` chain (direct `parse_if` recursion).
                let elifs = format!("{}{{ 0 }}", "if true { 0 } else ".repeat(n));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&elifs, SourceId(0), &mut i);
                assert!(has_e0209(&d), "deep else-if chain should report E0209, got {d:?}");

                // A `match` whose arm body is another `match`, nested deep.
                let matches = format!("{}0{}", "match 0 { _ => ".repeat(n), " }".repeat(n));
                let mut i = Interner::new();
                let (_e, d) = parse_expr_str(&matches, SourceId(0), &mut i);
                assert!(has_e0209(&d), "nested match arms should report E0209, got {d:?}");

                // Nested loop bodies `while … { while … { … } }` (statement form, via parse_module).
                let whiles = format!(
                    "fn f() {{ {}{} }}",
                    "while true { ".repeat(n),
                    "}".repeat(n)
                );
                let mut i = Interner::new();
                let (_m, d) = parse_module(&whiles, SourceId(0), &mut i);
                assert!(has_e0209(&d), "nested loop bodies should report E0209, got {d:?}");
            })
            .expect("spawn parser thread")
            .join()
            .expect("the parser must not overflow its stack on deeply nested structural input");
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
        assert!(d.is_empty(), "50-deep parens should parse cleanly, got {d:?}");

        // A 16-deep array type.
        let ty = format!("{}i32{}", "[".repeat(16), "; 1]".repeat(16));
        let mut i = Interner::new();
        let (_t, d) = parse_type_str(&ty, SourceId(0), &mut i);
        assert!(d.is_empty(), "16-deep array type should parse cleanly, got {d:?}");

        // 64-deep blocks, a 64-arm-deep else-if chain, a 64-deep match nest, and 64-deep loop
        // bodies all sit far under the limit — the new structural guards must not reject them.
        let blocks = format!("{}0{}", "{".repeat(64), "}".repeat(64));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&blocks, SourceId(0), &mut i);
        assert!(d.is_empty(), "64-deep blocks should parse cleanly, got {d:?}");

        let elifs = format!("{}{{ 0 }}", "if true { 0 } else ".repeat(64));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&elifs, SourceId(0), &mut i);
        assert!(d.is_empty(), "64-deep else-if chain should parse cleanly, got {d:?}");

        let matches = format!("{}0{}", "match 0 { _ => ".repeat(64), " }".repeat(64));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&matches, SourceId(0), &mut i);
        assert!(d.is_empty(), "64-deep match nest should parse cleanly, got {d:?}");

        // A wide-but-shallow block (many sequential statements) must not accumulate depth.
        let wide = format!("{{ {} 0 }}", "let x = 1; ".repeat(500));
        let mut i = Interner::new();
        let (_e, d) = parse_expr_str(&wide, SourceId(0), &mut i);
        assert!(d.is_empty(), "wide shallow block should parse cleanly, got {d:?}");
    }
}
