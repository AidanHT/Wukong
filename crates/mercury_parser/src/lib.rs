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
        }
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
        self.diags
            .push(Diagnostic::error(msg).with_code(code).primary(span, ""));
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
        let mut lhs = self.parse_cast();
        loop {
            let Some(op) = token_to_binop(self.kind()) else {
                break;
            };
            let bp = binop_bp(op);
            if bp < min_bp {
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
        lhs
    }

    /// Cast level: `as` binds looser than every prefix unary operator but tighter than every binary
    /// operator (the Rust precedence). Sitting it between `parse_expr_bp` and `parse_prefix` means
    /// `*p as T` is `(*p) as T` (not `*(p as T)` — which mis-typed as a `ptrtoint` then a load), and
    /// `-x as u8` is `(-x) as u8`. Chained `x as A as B` folds left.
    fn parse_cast(&mut self) -> Expr {
        let mut e = self.parse_prefix();
        while self.kind() == T::As {
            self.bump();
            let ty = self.parse_type();
            e = self.finish_expr(e.span, ExprKind::Cast { expr: Box::new(e), ty });
        }
        e
    }

    fn parse_prefix(&mut self) -> Expr {
        let start = self.span();
        let op = match self.kind() {
            T::Minus => Some(UnOp::Neg),
            T::Bang => Some(UnOp::Not),
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
        while !self.at(T::RParen) && !self.at(T::Eof) {
            args.push(self.parse_expr());
            if !self.eat(T::Comma) {
                break;
            }
        }
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
                if self.eat(T::RParen) {
                    return self.finish_expr(start, ExprKind::TupleLit(Vec::new()));
                }
                let first = self.parse_expr();
                if self.at(T::Comma) {
                    let mut items = vec![first];
                    while self.eat(T::Comma) {
                        if self.at(T::RParen) {
                            break;
                        }
                        items.push(self.parse_expr());
                    }
                    self.expect(T::RParen);
                    self.finish_expr(start, ExprKind::TupleLit(items))
                } else {
                    self.expect(T::RParen);
                    // Parenthesized expression: keep the inner node but extend its span.
                    Expr {
                        id: first.id,
                        kind: first.kind,
                        span: start.to(self.prev_span()),
                    }
                }
            }
            T::LBracket => {
                self.bump();
                if self.eat(T::RBracket) {
                    return self.finish_expr(start, ExprKind::ArrayLit(Vec::new()));
                }
                let first = self.parse_expr();
                if self.eat(T::Semi) {
                    let count = Box::new(self.parse_expr());
                    self.expect(T::RBracket);
                    self.finish_expr(
                        start,
                        ExprKind::ArrayRepeat {
                            value: Box::new(first),
                            count,
                        },
                    )
                } else {
                    let mut items = vec![first];
                    while self.eat(T::Comma) {
                        if self.at(T::RBracket) {
                            break;
                        }
                        items.push(self.parse_expr());
                    }
                    self.expect(T::RBracket);
                    self.finish_expr(start, ExprKind::ArrayLit(items))
                }
            }
            T::LBrace => {
                let b = self.parse_block();
                self.finish_expr(start, ExprKind::Block(b))
            }
            T::If => self.parse_if(),
            T::Match => self.parse_match(),
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
                self.finish_expr(
                    start,
                    ExprKind::Path(Path {
                        segments: vec![id],
                        span: start,
                    }),
                )
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

    fn parse_if(&mut self) -> Expr {
        let start = self.span();
        self.bump(); // if
        let cond = Box::new(self.parse_expr());
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
        self.finish_expr(
            start,
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            },
        )
    }

    fn parse_match(&mut self) -> Expr {
        let start = self.span();
        self.bump(); // match
        let scrutinee = Box::new(self.parse_expr());
        self.expect(T::LBrace);
        let mut arms = Vec::new();
        while !self.at(T::RBrace) && !self.at(T::Eof) {
            let arm_start = self.span();
            let pat = self.parse_pattern();
            self.expect(T::FatArrow);
            let body = self.parse_expr();
            arms.push(MatchArm {
                pat,
                body,
                span: arm_start.to(self.prev_span()),
            });
            self.eat(T::Comma);
        }
        self.expect(T::RBrace);
        self.finish_expr(start, ExprKind::Match { scrutinee, arms })
    }

    // ---- Statements & blocks ----

    pub(crate) fn parse_block(&mut self) -> Block {
        let start = self.span();
        let id = self.nid();
        self.expect(T::LBrace);
        let mut stmts = Vec::new();
        let mut tail = None;
        while !self.at(T::RBrace) && !self.at(T::Eof) {
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
                    self.eat(T::Semi);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Break(None), stmt_start));
                }
                T::Continue => {
                    self.bump();
                    self.eat(T::Semi);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Continue(None), stmt_start));
                }
                T::Defer => {
                    self.bump();
                    let e = self.parse_expr();
                    self.eat(T::Semi);
                    stmts.push(self.mk_stmt(attrs, StmtKind::Defer(e), stmt_start));
                }
                T::While => {
                    let k = self.parse_while();
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                T::For => {
                    let k = self.parse_for();
                    stmts.push(self.mk_stmt(attrs, k, stmt_start));
                }
                T::Loop => {
                    let k = self.parse_loop();
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
        }
        self.expect(T::RBrace);
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

    fn parse_while(&mut self) -> StmtKind {
        self.bump(); // while
        let cond = self.parse_expr();
        let body = self.parse_block();
        StmtKind::While {
            label: None,
            cond,
            body,
        }
    }

    fn parse_for(&mut self) -> StmtKind {
        self.bump(); // for
        let pat = self.parse_pattern();
        self.expect(T::In);
        let iter = self.parse_for_iter();
        let body = self.parse_block();
        StmtKind::For {
            label: None,
            pat,
            iter,
            body,
        }
    }

    fn parse_loop(&mut self) -> StmtKind {
        self.bump(); // loop
        let body = self.parse_block();
        StmtKind::Loop { label: None, body }
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

    fn parse_pattern(&mut self) -> Pattern {
        let start = self.span();
        let kind = match self.kind() {
            T::Ident => {
                let text = &self.src[start.lo as usize..start.hi as usize];
                if text == "_" {
                    self.bump();
                    PatKind::Wildcard
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
}
