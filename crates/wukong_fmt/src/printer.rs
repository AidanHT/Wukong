//! The pretty-printer: AST (+ the source it came from) → canonical text.
//!
//! Every `match` over an AST enum here is exhaustive with no `_ =>` arm, so a new variant fails
//! this crate's build until it is printed. Where the AST does not record how something was
//! spelled (`::` vs `.`, `~` vs `!`, `f32x8` vs `vec[f32, 8]`, a parenthesized type, …) the
//! spelling is read back from the token stream; where the parser desugared a construct (`impl`,
//! receivers, `if let`, `while let`, `let … else`, a local `const`, `defer x += 1`, `where`, a
//! range index) the printer recognizes the desugared shape and prints the surface form again.
//! Anything it cannot print faithfully is refused (`Err`), and `check.rs` refuses whatever slips
//! through.

use crate::FmtOptions;
use wukong_ast::*;
use wukong_lexer::{Comment, Token, TokenKind};
use wukong_span::{Interner, Span, Symbol};

use TokenKind as T;

/// The source a module was parsed from: its text, tokens, comments and interner.
pub(crate) struct Src<'a> {
    pub(crate) src: &'a str,
    pub(crate) tokens: &'a [Token],
    pub(crate) comments: &'a [Comment],
    pub(crate) interner: &'a Interner,
}

impl<'a> Src<'a> {
    pub(crate) fn new(
        src: &'a str,
        tokens: &'a [Token],
        comments: &'a [Comment],
        interner: &'a Interner,
    ) -> Src<'a> {
        debug_assert!(comments.windows(2).all(|w| w[0].span.lo < w[1].span.lo));
        Src {
            src,
            tokens,
            comments,
            interner,
        }
    }

    pub(crate) fn text(&self, sp: Span) -> &'a str {
        self.src
            .get(sp.lo as usize..sp.hi as usize)
            .unwrap_or_default()
    }

    fn sym(&self, s: Symbol) -> &'a str {
        self.interner.resolve(s)
    }

    /// Index of the first token whose start is at or after `pos`.
    fn tok_from(&self, pos: u32) -> usize {
        self.tokens.partition_point(|t| t.span.lo < pos)
    }

    /// The kind of the token that starts exactly at `pos`.
    fn kind_at(&self, pos: u32) -> Option<TokenKind> {
        let t = self.tokens.get(self.tok_from(pos))?;
        (t.span.lo == pos && t.kind != T::Eof).then_some(t.kind)
    }

    /// The last token that starts before `pos`.
    fn tok_before(&self, pos: u32) -> Option<Token> {
        let i = self.tok_from(pos);
        (i > 0).then(|| self.tokens[i - 1])
    }

    fn kind_before(&self, pos: u32) -> Option<TokenKind> {
        self.tok_before(pos).map(|t| t.kind)
    }

    /// The first real token at or after `pos`.
    fn tok_after(&self, pos: u32) -> Option<Token> {
        self.tokens
            .get(self.tok_from(pos))
            .copied()
            .filter(|t| t.kind != T::Eof)
    }

    /// The index of the closer matching the opener at token index `i`.
    fn matching(&self, i: usize) -> Option<usize> {
        let mut depth = 0usize;
        for (j, t) in self.tokens.iter().enumerate().skip(i) {
            match t.kind {
                T::LParen | T::LBracket | T::LBrace => depth += 1,
                T::RParen | T::RBracket | T::RBrace => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(j);
                    }
                }
                T::Eof => return None,
                _ => {}
            }
        }
        None
    }

    /// Strip balanced parentheses wrapped around the whole of `sp`: how many layers there were,
    /// and the span inside them.
    pub(crate) fn strip_parens(&self, sp: Span) -> (usize, Span) {
        let mut n = 0;
        let mut cur = sp;
        loop {
            let i = self.tok_from(cur.lo);
            let Some(t) = self.tokens.get(i) else { break };
            if t.span.lo != cur.lo || t.kind != T::LParen {
                break;
            }
            let Some(j) = self.matching(i) else { break };
            if self.tokens[j].span.hi != cur.hi {
                break;
            }
            n += 1;
            if j == i + 1 {
                break;
            }
            cur = Span::new(
                cur.source,
                self.tokens[i + 1].span.lo,
                self.tokens[j - 1].span.hi,
            );
        }
        (n, cur)
    }

    /// Whether any comment lies strictly inside `(lo, hi)`.
    pub(crate) fn has_comment_in(&self, lo: u32, hi: u32) -> bool {
        let i = self.comments.partition_point(|c| c.span.lo <= lo);
        self.comments.get(i).is_some_and(|c| c.span.lo < hi)
    }
}

/// An expression context: whether a struct literal would be misread here (the head of an
/// `if`/`while`/`for`/`match`, …, which the parser parses with struct literals off), and the
/// lowest precedence level an expression may print at without parentheses.
#[derive(Clone, Copy)]
pub(crate) struct Cx {
    pub(crate) ns: bool,
    pub(crate) min: u8,
}

impl Cx {
    fn top(ns: bool) -> Cx {
        Cx { ns, min: 0 }
    }
    fn at(self, min: u8) -> Cx {
        Cx { ns: self.ns, min }
    }
}

/// A closure's body runs to the end of the expression: the loosest level.
const LVL_CLOSURE: u8 = 0;
/// `e as T`: tighter than every binary operator, looser than every prefix operator.
const LVL_CAST: u8 = 10;
const LVL_UNARY: u8 = 11;
const LVL_POSTFIX: u8 = 12;

/// The parser's binding power (`wukong_parser::binop_bp`). Every operator is left-associative.
pub(crate) fn binop_bp(op: BinOp) -> u8 {
    match op {
        BinOp::Or => 1,
        BinOp::And => 2,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 3,
        BinOp::BitOr => 4,
        BinOp::BitXor => 5,
        BinOp::BitAnd => 6,
        BinOp::Shl | BinOp::Shr => 7,
        BinOp::Add | BinOp::Sub => 8,
        BinOp::Mul | BinOp::Div | BinOp::Rem => 9,
    }
}

fn level(e: &Expr) -> u8 {
    match &e.kind {
        ExprKind::Closure { .. } => LVL_CLOSURE,
        ExprKind::Binary { op, .. } => binop_bp(*op),
        ExprKind::Cast { .. } => LVL_CAST,
        ExprKind::Unary { .. } => LVL_UNARY,
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::Call { .. }
        | ExprKind::Index { .. }
        | ExprKind::Field { .. }
        | ExprKind::TupleField { .. }
        | ExprKind::StructLit { .. }
        | ExprKind::ArrayLit(_)
        | ExprKind::ArrayRepeat { .. }
        | ExprKind::TupleLit(_)
        | ExprKind::Block(_)
        | ExprKind::If { .. }
        | ExprKind::Match { .. }
        | ExprKind::Loop { .. }
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => LVL_POSTFIX,
    }
}

/// The parser's `is_block_like`: an expression that ends its own statement.
pub(crate) fn is_block_like(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::Block(_) | ExprKind::If { .. } | ExprKind::Match { .. } | ExprKind::Loop { .. }
    )
}

/// The two arms of a parser-desugared `if let` / `while let` / `let … else` match.
fn desugared_arms(e: &Expr, origin: ArmOrigin) -> Option<(&Expr, &[MatchArm])> {
    match &e.kind {
        ExprKind::Match { scrutinee, arms } if arms.len() == 2 && arms[1].origin == origin => {
            Some((scrutinee, arms))
        }
        _ => None,
    }
}

pub(crate) struct Printer<'a> {
    pub(crate) s: &'a Src<'a>,
    pub(crate) opts: &'a FmtOptions,
    pub(crate) out: String,
    pub(crate) indent: usize,
    /// Nothing has been written on the current output line yet.
    pub(crate) line_start: bool,
    /// The next comment (index into `s.comments`) not yet emitted.
    pub(crate) next_comment: usize,
    /// The source end of the last own-line element emitted (blank-line decisions).
    pub(crate) last_end: u32,
    /// The last line written opened a block: no blank line may follow.
    pub(crate) block_start: bool,
    /// Flat mode (`flat_of`): one line, no comments; a block-like sub-expression fails it.
    pub(crate) flat: bool,
    pub(crate) failed: bool,
    pub(crate) err: Option<String>,
}

/// Print a whole module, or say why it cannot be printed faithfully.
pub(crate) fn print_module(s: &Src<'_>, opts: &FmtOptions, m: &Module) -> Result<String, String> {
    let mut p = Printer::new(s, opts, false);
    p.module(m);
    if let Some(e) = p.err {
        return Err(e);
    }
    let mut out = p.out;
    while out.ends_with('\n') {
        out.pop();
    }
    if !out.is_empty() {
        out.push('\n');
    }
    Ok(out)
}

impl<'a> Printer<'a> {
    pub(crate) fn new(s: &'a Src<'a>, opts: &'a FmtOptions, flat: bool) -> Printer<'a> {
        Printer {
            s,
            opts,
            out: String::new(),
            indent: 0,
            line_start: true,
            next_comment: 0,
            last_end: 0,
            block_start: true,
            flat,
            failed: false,
            err: None,
        }
    }

    // ---- Output ----

    pub(crate) fn w(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.line_start {
            for _ in 0..self.indent {
                self.out.push_str("    ");
            }
            self.line_start = false;
        }
        self.out.push_str(text);
    }

    pub(crate) fn nl(&mut self) {
        if self.flat {
            self.failed = true;
            return;
        }
        self.out.push('\n');
        self.line_start = true;
    }

    /// The current output column (in characters).
    fn col(&self) -> usize {
        if self.line_start {
            return self.indent * 4;
        }
        let line = match self.out.rfind('\n') {
            Some(i) => &self.out[i + 1..],
            None => &self.out[..],
        };
        line.chars().count()
    }

    pub(crate) fn unsupported(&mut self, msg: impl Into<String>) {
        if self.err.is_none() {
            self.err = Some(msg.into());
        }
        self.failed = true;
    }

    fn sym(&self, s: Symbol) -> &'a str {
        self.s.sym(s)
    }

    /// An identifier: its symbol, except that a `Self` the parser resolved to an `impl`'s type is
    /// printed as the `Self` that was written.
    fn ident(&mut self, id: &Ident) {
        if self.s.text(id.span) == "Self" {
            self.w("Self");
        } else {
            let t = self.sym(id.sym);
            self.w(t);
        }
    }

    /// One line with no comments: the text `f` prints, or `None` when it would need more than one
    /// line (a block-like sub-expression) or cannot be printed.
    fn flat_of(&self, f: impl FnOnce(&mut Printer<'a>)) -> Option<String> {
        let mut p = Printer::new(self.s, self.opts, true);
        f(&mut p);
        (!p.failed && p.err.is_none()).then_some(p.out)
    }

    // ---- Module and items ----

    fn module(&mut self, m: &Module) {
        let first_item_lo = m.items.first().map(|i| i.span.lo);
        if let Some(name) = &m.name {
            let kw = self
                .s
                .tok_before(name.span.lo)
                .map(|t| t.span.lo)
                .unwrap_or(m.span.lo);
            self.leading(kw);
            self.w("module ");
            self.dotted_path(name);
            let end = match self.s.tok_after(name.span.hi) {
                Some(t) if t.kind == T::Semi => t.span.hi,
                _ => name.span.hi,
            };
            self.trailing(end, first_item_lo.unwrap_or(u32::MAX));
        }
        let mut i = 0;
        while i < m.items.len() {
            let it = &m.items[i];
            let consumed = match &it.kind {
                ItemKind::Impl(d) => d.methods.len(),
                ItemKind::Fn(_)
                | ItemKind::Struct(_)
                | ItemKind::Enum(_)
                | ItemKind::Const(_)
                | ItemKind::Import(_)
                | ItemKind::Extern(_)
                | ItemKind::Trait(_)
                | ItemKind::TypeAlias(_) => 0,
            };
            let end = (i + 1 + consumed).min(m.items.len());
            let next_lo = m.items.get(end).map(|n| n.span.lo).unwrap_or(u32::MAX);
            self.item(it, &m.items[i + 1..end], next_lo);
            if self.err.is_some() {
                return;
            }
            i = end;
        }
        self.flush_rest();
    }

    /// The attribute line above a node (all of them, space-separated), as its own element.
    /// Returns where the node itself starts in the source.
    fn attrs_line(&mut self, attrs: &[Attr], node_lo: u32) -> u32 {
        let Some(last) = attrs.last() else {
            return node_lo;
        };
        for (k, a) in attrs.iter().enumerate() {
            if k > 0 {
                self.w(" ");
            }
            self.attr(a);
        }
        let body_lo = self
            .s
            .tok_after(last.span.hi)
            .map(|t| t.span.lo)
            .unwrap_or(last.span.hi);
        self.trailing(last.span.hi, body_lo);
        self.leading(body_lo);
        body_lo
    }

    fn attr(&mut self, a: &Attr) {
        self.w("@");
        self.ident(&a.name);
        if a.args.is_empty() {
            return;
        }
        self.w("(");
        for (k, arg) in a.args.iter().enumerate() {
            if k > 0 {
                self.w(", ");
            }
            match arg {
                AttrArg::Word(s, _) | AttrArg::Int(s, _) | AttrArg::Str(s, _) => {
                    let t = self.sym(*s);
                    self.w(t);
                }
                AttrArg::KeyValue { key, value } => {
                    self.ident(key);
                    self.w(" = ");
                    match value {
                        AttrVal::Int(s) | AttrVal::Str(s) | AttrVal::Word(s) => {
                            let t = self.sym(*s);
                            self.w(t);
                        }
                        AttrVal::Bool(b) => self.w(if *b { "true" } else { "false" }),
                    }
                }
            }
        }
        self.w(")");
    }

    fn item(&mut self, it: &Item, methods: &[Item], next_lo: u32) {
        self.leading(it.span.lo);
        let body_lo = self.attrs_line(&it.attrs, it.span.lo);
        match &it.kind {
            ItemKind::Fn(f) => self.fn_decl(f, &f.name),
            ItemKind::Struct(d) => self.struct_decl(d, it.span),
            ItemKind::Enum(d) => self.enum_decl(d, it.span),
            ItemKind::Const(c) => {
                self.pub_kw(c.is_pub);
                self.w("const ");
                self.ident(&c.name);
                self.w(": ");
                self.ty(&c.ty);
                self.w(" = ");
                self.root_expr(&c.value, Cx::top(false), ";");
                self.w(";");
            }
            ItemKind::Import(im) => {
                self.w("import ");
                self.dotted_path(&im.path);
                if let Some(a) = &im.alias {
                    self.w(" as ");
                    self.ident(a);
                }
                if let Some(names) = &im.items {
                    self.w(".{");
                    for (k, n) in names.iter().enumerate() {
                        if k > 0 {
                            self.w(", ");
                        }
                        self.ident(n);
                    }
                    self.w("}");
                }
                self.w(";");
            }
            ItemKind::Extern(x) => self.extern_block(x, it.span, body_lo),
            ItemKind::Trait(t) => self.trait_decl(t, it.span),
            ItemKind::Impl(d) => self.impl_block(d, methods, it.span),
            ItemKind::TypeAlias(a) => {
                self.pub_kw(a.is_pub);
                self.w("type ");
                self.ident(&a.name);
                self.generics(&a.generics);
                self.w(" = ");
                self.ty(&a.ty);
                self.w(";");
            }
        }
        self.trailing(it.span.hi, next_lo);
    }

    fn pub_kw(&mut self, is_pub: bool) {
        if is_pub {
            self.w("pub ");
        }
    }

    fn struct_decl(&mut self, d: &StructDecl, span: Span) {
        self.pub_kw(d.is_pub);
        self.w("struct ");
        self.ident(&d.name);
        self.generics(&d.generics);
        self.w(" ");
        let close = span.hi - 1;
        let open = self.open_brace_before_fields(d.generics.last().map(|g| g.span.hi), d.name);
        if d.fields.is_empty() && !self.s.has_comment_in(open, close) {
            self.w("{}");
            return;
        }
        self.w("{");
        self.open();
        for (k, f) in d.fields.iter().enumerate() {
            let next = d.fields.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
            self.leading(f.span.lo);
            self.field_decl(f);
            self.w(",");
            self.trailing(f.span.hi, next);
        }
        self.close_at(close, "}");
    }

    /// The `{` that opens a struct's or enum's body.
    fn open_brace_before_fields(&self, after_generics: Option<u32>, name: Ident) -> u32 {
        let from = after_generics.unwrap_or(name.span.hi).max(name.span.hi);
        let mut i = self.s.tok_from(from);
        while let Some(t) = self.s.tokens.get(i) {
            if t.kind == T::LBrace || t.kind == T::Eof {
                return t.span.lo;
            }
            i += 1;
        }
        from
    }

    fn field_decl(&mut self, f: &Field) {
        self.pub_kw(f.is_pub);
        self.ident(&f.name);
        self.w(": ");
        self.ty(&f.ty);
    }

    fn enum_decl(&mut self, d: &EnumDecl, span: Span) {
        self.pub_kw(d.is_pub);
        self.w("enum ");
        self.ident(&d.name);
        self.generics(&d.generics);
        self.w(" ");
        let close = span.hi - 1;
        let open = self.open_brace_before_fields(d.generics.last().map(|g| g.span.hi), d.name);
        if d.variants.is_empty() && !self.s.has_comment_in(open, close) {
            self.w("{}");
            return;
        }
        self.w("{");
        self.open();
        for (k, v) in d.variants.iter().enumerate() {
            let next = d.variants.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
            self.leading(v.span.lo);
            self.ident(&v.name);
            match &v.data {
                VariantData::Unit => {}
                VariantData::Tuple(tys) => {
                    self.w("(");
                    for (j, t) in tys.iter().enumerate() {
                        if j > 0 {
                            self.w(", ");
                        }
                        self.ty(t);
                    }
                    self.w(")");
                }
                VariantData::Struct(fs) => {
                    if fs.is_empty() {
                        self.w(" {}");
                    } else {
                        self.w(" { ");
                        for (j, f) in fs.iter().enumerate() {
                            if j > 0 {
                                self.w(", ");
                            }
                            self.field_decl(f);
                        }
                        self.w(" }");
                    }
                }
            }
            if let Some(disc) = &v.discriminant {
                self.w(" = ");
                self.expr(disc, Cx::top(false));
            }
            self.w(",");
            self.trailing(v.span.hi, next);
        }
        self.close_at(close, "}");
    }

    fn trait_decl(&mut self, t: &TraitDecl, span: Span) {
        self.pub_kw(t.is_pub);
        self.w("trait ");
        self.ident(&t.name);
        self.w(" ");
        let close = span.hi - 1;
        if t.methods.is_empty() && !self.s.has_comment_in(t.name.span.hi, close) {
            self.w("{}");
            return;
        }
        self.w("{");
        self.open();
        let starts: Vec<u32> = t.methods.iter().map(|m| self.fn_kw_lo(m)).collect();
        for (k, m) in t.methods.iter().enumerate() {
            let next = starts.get(k + 1).copied().unwrap_or(close);
            self.leading(starts[k]);
            self.fn_decl(m, &m.name);
            let end = self.end_before(next);
            self.trailing(end, next);
        }
        self.close_at(close, "}");
    }

    /// Where a span-less `fn` declaration (a trait method, an extern declaration) starts: its
    /// `fn` keyword, or the `pub` before it.
    fn fn_kw_lo(&self, f: &FnDecl) -> u32 {
        let Some(fn_kw) = self.s.tok_before(f.name.span.lo) else {
            return f.name.span.lo;
        };
        match self.s.tok_before(fn_kw.span.lo) {
            Some(p) if p.kind == T::Pub => p.span.lo,
            _ => fn_kw.span.lo,
        }
    }

    /// The end of the last token before `pos`.
    fn end_before(&self, pos: u32) -> u32 {
        self.s.tok_before(pos).map(|t| t.span.hi).unwrap_or(pos)
    }

    fn impl_block(&mut self, d: &ImplDecl, methods: &[Item], span: Span) {
        if methods.len() != d.methods.len() {
            self.unsupported("an `impl` block whose methods are not all present");
            return;
        }
        self.w("impl ");
        if let Some(tr) = &d.trait_ref {
            self.dotted_path(tr);
            self.w(" for ");
        }
        self.dotted_path(&d.self_ty);
        self.w(" ");
        let close = span.hi - 1;
        if methods.is_empty() && !self.s.has_comment_in(d.self_ty.span.hi, close) {
            self.w("{}");
            return;
        }
        self.w("{");
        self.open();
        for (k, (mi, m)) in methods.iter().zip(&d.methods).enumerate() {
            let ItemKind::Fn(f) = &mi.kind else {
                self.unsupported("an `impl` block followed by a non-method item");
                return;
            };
            if f.name.sym != m.fn_name {
                self.unsupported("an `impl` method whose function is out of place");
                return;
            }
            let next = methods.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
            self.leading(mi.span.lo);
            self.attrs_line(&mi.attrs, mi.span.lo);
            self.fn_decl(f, &m.name);
            self.trailing(mi.span.hi, next);
        }
        self.close_at(close, "}");
    }

    fn extern_block(&mut self, x: &ExternBlock, span: Span, body_lo: u32) {
        // The `extern` keyword: after the attributes, and after a `pub`.
        let mut kw = self.s.tok_after(body_lo);
        let mut outer_pub = false;
        if kw.is_some_and(|t| t.kind == T::Pub) {
            outer_pub = true;
            kw = kw.and_then(|t| self.s.tok_after(t.span.hi));
        }
        let Some(kw) = kw.filter(|t| t.kind == T::Extern) else {
            self.unsupported("an `extern` item whose keyword was not found");
            return;
        };
        self.pub_kw(outer_pub);
        self.w("extern ");
        let mut after = self.s.tok_after(kw.span.hi);
        if let Some(t) = after.filter(|t| t.kind == T::Str) {
            let abi = self.sym(x.abi);
            self.w(abi);
            self.w(" ");
            after = self.s.tok_after(t.span.hi);
        }
        match after.map(|t| t.kind) {
            Some(T::LBrace) => {
                let open = after.map(|t| t.span.lo).unwrap_or(kw.span.hi);
                let close = span.hi - 1;
                if x.items.is_empty() && !self.s.has_comment_in(open, close) {
                    self.w("{}");
                    return;
                }
                self.w("{");
                self.open();
                let starts: Vec<u32> = x.items.iter().map(|f| self.fn_kw_lo(f)).collect();
                for (k, f) in x.items.iter().enumerate() {
                    let next = starts.get(k + 1).copied().unwrap_or(close);
                    self.leading(starts[k]);
                    let mut g = f.clone();
                    if outer_pub {
                        // The block's `pub` exports every declaration in it already.
                        g.is_pub = false;
                    }
                    self.fn_decl(&g, &f.name);
                    let end = self.end_before(next);
                    self.trailing(end, next);
                }
                self.close_at(close, "}");
            }
            Some(T::Fn) => match x.items.as_slice() {
                [f] => {
                    let mut g = f.clone();
                    g.is_pub = false;
                    self.fn_decl(&g, &f.name);
                }
                _ => self.unsupported("a one-declaration `extern` with no declaration"),
            },
            _ => {
                if !x.items.is_empty() {
                    self.unsupported("an `extern` item of an unknown shape");
                }
            }
        }
    }

    // ---- Functions ----

    fn generics(&mut self, gs: &[GenericParam]) {
        if gs.is_empty() {
            return;
        }
        self.w("<");
        for (k, g) in gs.iter().enumerate() {
            if k > 0 {
                self.w(", ");
            }
            match &g.kind {
                GenericParamKind::Type(name) => {
                    self.ident(name);
                    // Bounds written inline; one after the parameter came from a `where` clause.
                    let inline: Vec<&Path> =
                        g.bounds.iter().filter(|b| b.span.lo < g.span.hi).collect();
                    for (j, b) in inline.iter().enumerate() {
                        self.w(if j == 0 { ": " } else { " + " });
                        self.dotted_path(b);
                    }
                }
                GenericParamKind::Const { name, ty } => {
                    self.w("const ");
                    self.ident(name);
                    self.w(": ");
                    self.ty(ty);
                }
            }
        }
        self.w(">");
    }

    /// The `where` clause: the bounds that came from it and the dimension predicates, merged in
    /// source order. `T: A, T: B` stays two entries; `T: A + B` one.
    fn where_clause(&mut self, f: &FnDecl) {
        enum W<'e> {
            Bound(Ident, &'e Path),
            Pred(&'e Expr),
        }
        let mut entries: Vec<(u32, W<'_>)> = Vec::new();
        for g in &f.generics {
            if let GenericParamKind::Type(name) = &g.kind {
                for b in g.bounds.iter().filter(|b| b.span.lo >= g.span.hi) {
                    entries.push((b.span.lo, W::Bound(*name, b)));
                }
            }
        }
        for p in &f.where_preds {
            entries.push((p.span.lo, W::Pred(p)));
        }
        if entries.is_empty() {
            return;
        }
        entries.sort_by_key(|(lo, _)| *lo);
        self.w(" where ");
        for (k, (lo, e)) in entries.iter().enumerate() {
            match e {
                W::Bound(name, b) => {
                    if k > 0 && self.s.kind_before(*lo) == Some(T::Plus) {
                        self.w(" + ");
                    } else {
                        if k > 0 {
                            self.w(", ");
                        }
                        self.ident(name);
                        self.w(": ");
                    }
                    self.dotted_path(b);
                }
                W::Pred(p) => {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.expr(p, Cx::top(true));
                }
            }
        }
    }

    /// The `(` and `)` of a function's parameter list.
    fn param_parens(&self, f: &FnDecl) -> Option<(u32, u32)> {
        let from = f
            .generics
            .last()
            .map(|g| g.span.hi)
            .unwrap_or(f.name.span.hi)
            .max(f.name.span.hi);
        let mut i = self.s.tok_from(from);
        while let Some(t) = self.s.tokens.get(i) {
            match t.kind {
                T::LParen => {
                    let j = self.s.matching(i)?;
                    return Some((t.span.lo, self.s.tokens[j].span.lo));
                }
                T::Eof | T::LBrace | T::Semi => return None,
                _ => i += 1,
            }
        }
        None
    }

    fn fn_head(&mut self, f: &FnDecl, name: &Ident, vertical: Option<u32>) {
        self.pub_kw(f.is_pub);
        self.w("fn ");
        self.ident(name);
        self.generics(&f.generics);
        self.w("(");
        match vertical {
            Some(close) => {
                self.open();
                for (k, p) in f.params.iter().enumerate() {
                    let next = f.params.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
                    self.leading(p.span.lo);
                    self.param(p);
                    self.w(",");
                    self.trailing(p.span.hi, next);
                }
                self.close_at(close, ")");
            }
            None => {
                for (k, p) in f.params.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.param(p);
                }
                self.w(")");
            }
        }
        if let Some(r) = &f.ret {
            self.w(" -> ");
            self.ty(r);
        }
        self.where_clause(f);
    }

    fn fn_decl(&mut self, f: &FnDecl, name: &Ident) {
        let parens = self.param_parens(f);
        let mut vertical = None;
        if let Some((open, close)) = parens {
            let commented = !self.flat && self.s.has_comment_in(open, close);
            let too_long = || {
                self.flat_of(|p| p.fn_head(f, name, None))
                    .is_some_and(|h| self.col() + h.chars().count() + 2 > self.opts.max_width)
            };
            if !self.flat && (commented || too_long()) {
                vertical = Some(close);
            }
        }
        self.fn_head(f, name, vertical);
        match &f.body {
            None => self.w(";"),
            Some(b) => match b.tail.as_deref() {
                Some(t) if b.stmts.is_empty() && b.span == t.span => {
                    // `fn f() -> T = e;`: the parser's block around `e` takes its id after `e`'s.
                    self.w(" = ");
                    self.root_expr(t, Cx::top(false), ";");
                    self.w(";");
                }
                _ => {
                    self.w(" ");
                    self.block(b, false);
                }
            },
        }
    }

    fn param(&mut self, p: &Param) {
        for a in &p.attrs {
            self.attr(a);
            self.w(" ");
        }
        if self.sym(p.name.sym) == "self" && p.ty.span == p.name.span {
            // A receiver: `self`, `mut self`, `&self` or `&mut self`, as written.
            let i = self.s.tok_from(p.span.lo);
            let mut text = String::new();
            for t in self.s.tokens[i..].iter() {
                if t.span.lo >= p.span.hi {
                    break;
                }
                match t.kind {
                    T::Amp => text.push('&'),
                    T::Mut => text.push_str("mut "),
                    T::Ident => text.push_str("self"),
                    _ => {
                        self.unsupported("a receiver spelled with an unexpected token");
                        return;
                    }
                }
            }
            self.w(&text);
            return;
        }
        if p.mutable {
            self.w("mut ");
        }
        self.ident(&p.name);
        self.w(": ");
        self.ty(&p.ty);
    }

    // ---- Paths ----

    /// A path whose separators are always `.` (module, import, impl and bound paths).
    fn dotted_path(&mut self, p: &Path) {
        self.path_with(p, ".");
    }

    /// A path whose separators are read back from the source (`.` or `::`), `default` when the
    /// token before a segment is neither.
    fn path_with(&mut self, p: &Path, default: &str) {
        for (k, seg) in p.segments.iter().enumerate() {
            if k > 0 {
                let sep = match self.s.kind_before(seg.span.lo) {
                    Some(T::ColonColon) => "::",
                    Some(T::Dot) => ".",
                    _ => default,
                };
                self.w(sep);
            }
            self.ident(seg);
        }
    }

    // ---- Types ----

    pub(crate) fn ty(&mut self, t: &TypeExpr) {
        let (layers, inner) = self.s.strip_parens(t.span);
        // A parenthesized type keeps its parentheses: the parser allocates a node id for them.
        let layers = match &t.kind {
            TypeKind::Tuple(_) | TypeKind::Unit => layers.saturating_sub(1),
            TypeKind::Path(_)
            | TypeKind::Int(_)
            | TypeKind::Pointer { .. }
            | TypeKind::Ref { .. }
            | TypeKind::Slice(_)
            | TypeKind::Array { .. }
            | TypeKind::Vector { .. }
            | TypeKind::Tensor { .. }
            | TypeKind::Fn { .. }
            | TypeKind::Generic { .. } => layers,
        };
        for _ in 0..layers {
            self.w("(");
        }
        self.ty_kind(t, inner);
        for _ in 0..layers {
            self.w(")");
        }
    }

    fn ty_kind(&mut self, t: &TypeExpr, inner: Span) {
        match &t.kind {
            TypeKind::Path(p) => self.path_with(p, "."),
            TypeKind::Int(s) => {
                let x = self.sym(*s);
                self.w(x);
            }
            TypeKind::Unit => self.w("()"),
            TypeKind::Pointer { mutable, pointee } => {
                self.w(if *mutable { "*mut " } else { "*" });
                self.ty(pointee);
            }
            TypeKind::Ref { mutable, pointee } => {
                if *mutable {
                    self.w("&mut ");
                } else {
                    self.w("&");
                    // `& &T`, never `&&T` (one `&&` token).
                    if matches!(pointee.kind, TypeKind::Ref { .. })
                        && self.s.strip_parens(pointee.span).0 == 0
                    {
                        self.w(" ");
                    }
                }
                self.ty(pointee);
            }
            TypeKind::Slice(e) => {
                self.w("[]");
                self.ty(e);
            }
            TypeKind::Array { elem, len, shared } => {
                if *shared {
                    self.w("shared ");
                }
                self.w("[");
                self.ty(elem);
                self.w("; ");
                self.expr(len, Cx::top(true));
                self.w("]");
            }
            TypeKind::Tuple(items) => {
                self.w("(");
                for (k, it) in items.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.ty(it);
                }
                if items.len() == 1 {
                    self.w(",");
                }
                self.w(")");
            }
            TypeKind::Vector { elem, lanes } => {
                let text = self.s.text(inner);
                if self.s.tok_from(inner.lo) + 1 == self.s.tok_from(inner.hi)
                    && self.s.kind_at(inner.lo) == Some(T::Ident)
                {
                    // `f32x8`, one identifier.
                    self.w(text);
                } else {
                    self.w("vec[");
                    self.ty(elem);
                    self.w(", ");
                    match self.s.tok_before(inner.hi.saturating_sub(1)) {
                        Some(tk) if tk.kind == T::Int => {
                            let lt = self.s.text(tk.span);
                            self.w(lt);
                        }
                        _ => self.w(&lanes.to_string()),
                    }
                    self.w("]");
                }
            }
            TypeKind::Tensor { elem, dims, layout } => {
                self.w("Tensor[");
                self.ty(elem);
                for d in dims {
                    self.w(", ");
                    self.dim(d, 0);
                }
                if let Some(l) = layout {
                    self.w(", ");
                    let after = dims.last().map(|d| d.span.hi).unwrap_or(elem.span.hi);
                    self.layout(l, after);
                }
                self.w("]");
            }
            TypeKind::Fn { params, ret } => {
                self.w("fn(");
                for (k, p) in params.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.ty(p);
                }
                self.w(")");
                if let Some(r) = ret {
                    self.w(" -> ");
                    self.ty(r);
                }
            }
            TypeKind::Generic { path, args } => {
                self.path_with(path, ".");
                if self.s.kind_at(path.span.hi) == Some(T::ColonColon) {
                    self.w("::");
                }
                self.w("<");
                for (k, a) in args.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    match a {
                        TypeArg::Type(t) => self.ty(t),
                        TypeArg::Dim(d) => self.dim(d, 0),
                    }
                }
                self.w(">");
            }
        }
    }

    /// A tensor layout (`.contiguous`, `.tiled(64, 64)`, …), whose tile extents are read back
    /// from the source: the AST keeps only their values.
    fn layout(&mut self, l: &Layout, after: u32) {
        let mut name_tok = None;
        let mut i = self.s.tok_from(after);
        while let Some(t) = self.s.tokens.get(i) {
            if t.kind == T::Dot {
                name_tok = self.s.tokens.get(i + 1).copied();
                break;
            }
            if t.kind == T::Eof || t.kind == T::RBracket {
                break;
            }
            i += 1;
        }
        match l {
            Layout::Contiguous => self.w(".contiguous"),
            Layout::ColMajor => self.w(".col_major"),
            Layout::Strided => self.w(".strided"),
            Layout::Tiled(sizes) => {
                self.w(".tiled");
                let Some(nt) = name_tok else {
                    self.unsupported("a tiled layout whose extents were not found");
                    return;
                };
                let j = self.s.tok_from(nt.span.hi);
                if self.s.tokens.get(j).map(|t| t.kind) == Some(T::LParen) {
                    let Some(close) = self.s.matching(j) else {
                        self.unsupported("an unclosed tiled layout");
                        return;
                    };
                    let ints: Vec<&str> = self.s.tokens[j + 1..close]
                        .iter()
                        .filter(|t| t.kind == T::Int)
                        .map(|t| self.s.text(t.span))
                        .collect();
                    if ints.len() != sizes.len() {
                        self.unsupported("a tiled layout with a non-integer extent");
                        return;
                    }
                    self.w("(");
                    self.w(&ints.join(", "));
                    self.w(")");
                }
            }
        }
    }

    /// One tensor dimension, at precedence `min` (`+ -` are 1, `* / %` are 2, left-assoc).
    fn dim(&mut self, d: &Dim, min: u8) {
        match &d.kind {
            DimKind::Int(_) => {
                let (_, inner) = self.s.strip_parens(d.span);
                let t = self.s.text(inner);
                self.w(t);
            }
            DimKind::Named(s) => {
                let t = self.sym(*s);
                self.w(t);
            }
            DimKind::Dynamic => self.w("?"),
            DimKind::Binary { op, lhs, rhs } => {
                let bp = match op {
                    BinOp::Add | BinOp::Sub => 1,
                    BinOp::Mul | BinOp::Div | BinOp::Rem => 2,
                    BinOp::BitAnd
                    | BinOp::BitOr
                    | BinOp::BitXor
                    | BinOp::Shl
                    | BinOp::Shr
                    | BinOp::Eq
                    | BinOp::Ne
                    | BinOp::Lt
                    | BinOp::Le
                    | BinOp::Gt
                    | BinOp::Ge
                    | BinOp::And
                    | BinOp::Or => {
                        self.unsupported("a dimension with a non-arithmetic operator");
                        return;
                    }
                };
                let paren = bp < min;
                if paren {
                    self.w("(");
                }
                self.dim(lhs, bp);
                self.w(" ");
                self.w(op.glyph());
                self.w(" ");
                self.dim(rhs, bp + 1);
                if paren {
                    self.w(")");
                }
            }
        }
    }

    // ---- Patterns ----

    pub(crate) fn pat(&mut self, p: &Pattern) {
        match &p.kind {
            PatKind::Wildcard => self.w("_"),
            PatKind::Ident(s) => {
                let t = self.sym(*s);
                self.w(t);
            }
            PatKind::Tuple(subs) => {
                self.w("(");
                for (k, s) in subs.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.pat(s);
                }
                // `(p)` and `(p,)` are both a one-element tuple pattern; keep what was written.
                if subs.len() == 1
                    && self.s.kind_before(p.span.hi.saturating_sub(1)) == Some(T::Comma)
                {
                    self.w(",");
                }
                self.w(")");
            }
            PatKind::Unit => self.w("()"),
            PatKind::Int { sym, neg } => {
                if *neg {
                    self.w("-");
                }
                let t = self.sym(*sym);
                self.w(t);
            }
            PatKind::Char(s) => {
                let t = self.sym(*s);
                self.w(t);
            }
            PatKind::Bool(b) => self.w(if *b { "true" } else { "false" }),
            PatKind::Or(alts) => {
                for (k, a) in alts.iter().enumerate() {
                    if k > 0 {
                        self.w(" | ");
                    }
                    self.pat(a);
                }
            }
            PatKind::Path(path) => self.path_with(path, "::"),
            PatKind::Range { lo, hi, inclusive } => {
                self.pat(lo);
                self.w(if *inclusive { "..=" } else { ".." });
                self.pat(hi);
            }
            PatKind::Variant { path, fields } => {
                self.path_with(path, "::");
                match fields {
                    VariantPat::Tuple(subs) => {
                        self.w("(");
                        for (k, s) in subs.iter().enumerate() {
                            if k > 0 {
                                self.w(", ");
                            }
                            self.pat(s);
                        }
                        self.w(")");
                    }
                    VariantPat::Struct(fps) => {
                        // `..` is not stored (it changes nothing); keep it when it was written.
                        let rest =
                            self.s.kind_before(p.span.hi.saturating_sub(1)) == Some(T::DotDot);
                        self.field_pats(fps, rest);
                    }
                }
            }
            PatKind::Struct { path, fields, rest } => {
                self.path_with(path, "::");
                self.field_pats(fields, *rest);
            }
            PatKind::Binding { name, sub } => {
                let t = self.sym(*name);
                self.w(t);
                self.w(" @ ");
                self.pat(sub);
            }
            PatKind::Slice {
                prefix,
                rest,
                suffix,
            } => {
                self.w("[");
                let mut first = true;
                let mut sep = |p: &mut Self| {
                    if !first {
                        p.w(", ");
                    }
                    first = false;
                };
                for s in prefix {
                    sep(self);
                    self.pat(s);
                }
                if *rest {
                    sep(self);
                    self.w("..");
                }
                for s in suffix {
                    sep(self);
                    self.pat(s);
                }
                self.w("]");
            }
        }
    }

    fn field_pats(&mut self, fps: &[FieldPat], rest: bool) {
        if fps.is_empty() && !rest {
            self.w(" {}");
            return;
        }
        self.w(" { ");
        for (k, fp) in fps.iter().enumerate() {
            if k > 0 {
                self.w(", ");
            }
            let name = self.sym(fp.name);
            self.w(name);
            // Shorthand `{ r }` only where it was written: no `:` before the sub-pattern.
            if self.s.kind_before(fp.pat.span.lo) == Some(T::Colon) {
                self.w(": ");
                self.pat(&fp.pat);
            }
        }
        if rest {
            if !fps.is_empty() {
                self.w(", ");
            }
            self.w("..");
        }
        self.w(" }");
    }

    // ---- Blocks and statements ----

    /// Open a vertical construct: end the line, indent, and suppress a blank line after it.
    pub(crate) fn open(&mut self) {
        self.nl();
        self.indent += 1;
        self.block_start = true;
    }

    pub(crate) fn block(&mut self, b: &Block, ns: bool) {
        if self.flat {
            self.failed = true;
            return;
        }
        let close = b.span.hi.saturating_sub(1);
        if b.stmts.is_empty() && b.tail.is_none() && !self.s.has_comment_in(b.span.lo, b.span.hi) {
            self.w("{}");
            return;
        }
        self.w("{");
        self.open();
        self.block_body(b, ns, close);
        self.close_at(close, "}");
    }

    fn block_body(&mut self, b: &Block, ns: bool, close: u32) {
        for (k, st) in b.stmts.iter().enumerate() {
            let (next_lo, next_cont) = match b.stmts.get(k + 1) {
                Some(n) => (n.span.lo, self.stmt_starts_cont(n, ns)),
                None => match &b.tail {
                    Some(t) => (t.span.lo, self.starts_cont(t, Cx::top(ns))),
                    None => (close, false),
                },
            };
            let last_no_tail = k + 1 == b.stmts.len() && b.tail.is_none();
            self.stmt(st, ns, last_no_tail, next_cont, next_lo);
            if self.err.is_some() {
                return;
            }
        }
        if let Some(t) = &b.tail {
            self.leading(t.span.lo);
            self.root_expr(t, Cx::top(ns), "");
            self.trailing(t.span.hi, close);
        }
    }

    fn stmt(&mut self, st: &Stmt, ns: bool, last_no_tail: bool, next_cont: bool, next_lo: u32) {
        self.leading(st.span.lo);
        self.attrs_line(&st.attrs, st.span.lo);
        let cx = Cx::top(ns);
        match &st.kind {
            StmtKind::Let {
                pat,
                mutable,
                ty,
                init,
            } => {
                if let Some((scrut, arms)) = init
                    .as_ref()
                    .and_then(|i| desugared_arms(i, ArmOrigin::LetElse))
                {
                    if ty.is_some() {
                        self.unsupported("a `let … else` with a type annotation");
                        return;
                    }
                    self.let_else(*mutable, scrut, arms, ns);
                } else if self.s.kind_before(pat.span.lo) == Some(T::Const) {
                    // A local `const X: T = v;` (parsed as an immutable `let`).
                    let (PatKind::Ident(_), Some(t), Some(v)) = (&pat.kind, ty, init) else {
                        self.unsupported("a local `const` of an unexpected shape");
                        return;
                    };
                    self.w("const ");
                    self.pat(pat);
                    self.w(": ");
                    self.ty(t);
                    self.w(" = ");
                    self.root_expr(v, cx, ";");
                    self.w(";");
                } else {
                    self.w(if *mutable { "let mut " } else { "let " });
                    self.pat(pat);
                    if let Some(t) = ty {
                        self.w(": ");
                        self.ty(t);
                    }
                    if let Some(v) = init {
                        self.w(" = ");
                        self.root_expr(v, cx, ";");
                    }
                    self.w(";");
                }
            }
            StmtKind::Assign { target, op, value } => {
                self.expr(target, cx);
                self.w(" ");
                self.w(op.glyph());
                self.w(" ");
                self.root_expr(value, cx, ";");
                self.w(";");
            }
            StmtKind::Expr(e) => {
                if let Some((scrut, arms)) = desugared_arms(e, ArmOrigin::LetElse) {
                    // A `let … else` that binds no name: the statement is the match itself.
                    self.let_else(false, scrut, arms, ns);
                } else if self.is_while_let(e) {
                    self.while_let(e, ns);
                } else {
                    let semi = !is_block_like(e) || last_no_tail || next_cont;
                    self.root_expr(e, cx, if semi { ";" } else { "" });
                    if semi {
                        self.w(";");
                    }
                }
            }
            StmtKind::Return(v) => {
                self.w("return");
                if let Some(v) = v {
                    self.w(" ");
                    self.root_expr(v, cx, ";");
                }
                self.w(";");
            }
            StmtKind::Break(label, v) => {
                self.w("break");
                if let Some(l) = label {
                    self.w(" '");
                    let t = self.sym(l.sym);
                    self.w(t);
                }
                if let Some(v) = v {
                    self.w(" ");
                    if label.is_none() && self.leftmost_is_labeled_loop(v) {
                        self.w("(");
                        self.expr(v, Cx::top(false));
                        self.w(")");
                    } else {
                        self.root_expr(v, cx, ";");
                    }
                }
                self.w(";");
            }
            StmtKind::Continue(label) => {
                self.w("continue");
                if let Some(l) = label {
                    self.w(" '");
                    let t = self.sym(l.sym);
                    self.w(t);
                }
                self.w(";");
            }
            StmtKind::Defer(e) => {
                self.w("defer ");
                match &e.kind {
                    ExprKind::Block(b) if self.s.kind_at(e.span.lo) != Some(T::LBrace) => {
                        // `defer x += 1;`, parsed as `defer { x += 1; }`.
                        match (b.stmts.as_slice(), &b.tail) {
                            ([a], None) => match &a.kind {
                                StmtKind::Assign { target, op, value } => {
                                    self.expr(target, cx);
                                    self.w(" ");
                                    self.w(op.glyph());
                                    self.w(" ");
                                    self.expr(value, cx);
                                    self.w(";");
                                }
                                _ => {
                                    self.unsupported("a deferred assignment of an unexpected shape")
                                }
                            },
                            _ => self.unsupported("a deferred assignment of an unexpected shape"),
                        }
                    }
                    _ => {
                        let semi = !is_block_like(e) || next_cont;
                        self.root_expr(e, cx, if semi { ";" } else { "" });
                        if semi {
                            self.w(";");
                        }
                    }
                }
            }
            StmtKind::While { label, cond, body } => {
                self.label(label);
                self.w("while ");
                self.expr(cond, Cx::top(true));
                self.w(" ");
                self.block(body, ns);
            }
            StmtKind::For {
                label,
                pat,
                iter,
                body,
            } => {
                self.label(label);
                self.w("for ");
                self.pat(pat);
                self.w(" in ");
                match iter {
                    ForIter::Range {
                        start,
                        end,
                        inclusive,
                        step,
                    } => {
                        self.expr(start, Cx::top(true));
                        self.w(if *inclusive { "..=" } else { ".." });
                        if let Some(e) = end {
                            self.expr(e, Cx::top(true));
                        }
                        if let Some(s) = step {
                            self.w(" step ");
                            self.expr(s, Cx::top(true));
                        }
                    }
                    ForIter::Expr(e) => self.expr(e, Cx::top(true)),
                }
                self.w(" ");
                self.block(body, ns);
            }
        }
        self.trailing(st.span.hi, next_lo);
    }

    fn label(&mut self, label: &Option<Ident>) {
        if let Some(l) = label {
            self.w("'");
            let t = self.sym(l.sym);
            self.w(t);
            self.w(": ");
        }
    }

    /// `let [mut ]P = e else { B };` from the match the parser built for it.
    fn let_else(&mut self, mutable: bool, scrut: &Expr, arms: &[MatchArm], ns: bool) {
        let ExprKind::Block(b) = &arms[1].body.kind else {
            self.unsupported("a `let … else` whose `else` is not a block");
            return;
        };
        if arms[0].guard.is_some() {
            self.unsupported("a `let … else` with a guard");
            return;
        }
        self.w(if mutable { "let mut " } else { "let " });
        self.pat(&arms[0].pat);
        self.w(" = ");
        // An `else`-less `if` at the end of the value would take the `else` for its own.
        if self.ends_with_open_if(scrut) {
            self.w("(");
            self.expr(scrut, Cx::top(false));
            self.w(")");
        } else {
            self.expr(scrut, Cx::top(ns));
        }
        self.w(" else ");
        self.block(b, ns);
        self.w(";");
    }

    fn ends_with_open_if(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::If { else_branch, .. } => match else_branch {
                None => true,
                Some(eb) => self.ends_with_open_if(eb),
            },
            ExprKind::Match { arms, .. }
                if arms.len() == 2 && arms[1].origin == ArmOrigin::LetFallthrough =>
            {
                arms[1].body.span == arms[0].body.span || self.ends_with_open_if(&arms[1].body)
            }
            ExprKind::Binary { rhs, .. } => self.ends_with_open_if(rhs),
            ExprKind::Unary { expr, .. } => self.ends_with_open_if(expr),
            ExprKind::Closure {
                body, ret: None, ..
            } => self.ends_with_open_if(body),
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::Call { .. }
            | ExprKind::Index { .. }
            | ExprKind::Field { .. }
            | ExprKind::TupleField { .. }
            | ExprKind::Cast { .. }
            | ExprKind::StructLit { .. }
            | ExprKind::ArrayLit(_)
            | ExprKind::ArrayRepeat { .. }
            | ExprKind::TupleLit(_)
            | ExprKind::Block(_)
            | ExprKind::Match { .. }
            | ExprKind::Loop { .. }
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_)
            | ExprKind::Closure { .. } => false,
        }
    }

    /// Whether the leftmost primary of `e` (as printed) is a labeled `loop`, which after a
    /// label-less `break` would read as `break 'l` + garbage.
    fn leftmost_is_labeled_loop(&self, e: &Expr) -> bool {
        if let ExprKind::Loop { label, .. } = &e.kind {
            return label.is_some() && !self.is_while_let(e);
        }
        if let ExprKind::Binary { op, lhs, .. } = &e.kind {
            return level(lhs) >= binop_bp(*op) && self.leftmost_is_labeled_loop(lhs);
        }
        if let ExprKind::Cast { expr, .. } = &e.kind {
            return level(expr) >= LVL_CAST && self.leftmost_is_labeled_loop(expr);
        }
        false
    }

    fn is_while_let(&self, e: &Expr) -> bool {
        matches!(e.kind, ExprKind::Loop { .. }) && self.s.kind_at(e.span.lo) == Some(T::While)
    }

    /// `[label: ]while let P = e [&& g] { A }` from the `loop { match … }` the parser built.
    fn while_let(&mut self, e: &Expr, ns: bool) {
        let ExprKind::Loop { label, body } = &e.kind else {
            return;
        };
        let shape = match (body.stmts.as_slice(), &body.tail) {
            ([s], None) => match &s.kind {
                StmtKind::Expr(m) => desugared_arms(m, ArmOrigin::LetFallthrough),
                _ => None,
            },
            _ => None,
        };
        let Some((scrut, arms)) = shape else {
            self.unsupported("a `while let` of an unexpected shape");
            return;
        };
        let ExprKind::Block(user) = &arms[0].body.kind else {
            self.unsupported("a `while let` whose body is not a block");
            return;
        };
        self.label(label);
        self.w("while let ");
        self.pat(&arms[0].pat);
        self.w(" = ");
        self.let_head(scrut, arms[0].guard.as_ref());
        self.w(" ");
        self.block(user, ns);
    }

    /// The scrutinee of an `if let`/`while let` and its `&& g1 && g2` chain. The scrutinee and
    /// every chain operand are parsed above `&&`, so an `&&`/`||` among them is parenthesized.
    fn let_head(&mut self, scrut: &Expr, guard: Option<&Expr>) {
        self.expr(scrut, Cx { ns: true, min: 3 });
        if let Some(g) = guard {
            let mut ops = Vec::new();
            let mut cur = g;
            while let ExprKind::Binary {
                op: BinOp::And,
                lhs,
                rhs,
            } = &cur.kind
            {
                ops.push(&**rhs);
                cur = lhs;
            }
            ops.push(cur);
            for o in ops.iter().rev() {
                self.w(" && ");
                self.expr(o, Cx { ns: true, min: 3 });
            }
        }
    }

    /// `if let P = e [&& g] { A }[ else …]` from the match the parser built.
    fn if_let(&mut self, scrut: &Expr, arms: &[MatchArm], ns: bool) {
        let (a0, a1) = (&arms[0], &arms[1]);
        let no_else = a1.body.span == a0.body.span;
        let then_block = if no_else {
            match &a0.body.kind {
                ExprKind::Block(b) => match (b.stmts.as_slice(), &b.tail) {
                    ([s], None) => match &s.kind {
                        StmtKind::Expr(Expr {
                            kind: ExprKind::Block(inner),
                            ..
                        }) => Some(inner),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }
        } else {
            match &a0.body.kind {
                ExprKind::Block(b) => Some(b),
                _ => None,
            }
        };
        let Some(then_block) = then_block else {
            self.unsupported("an `if let` of an unexpected shape");
            return;
        };
        self.w("if let ");
        self.pat(&a0.pat);
        self.w(" = ");
        self.let_head(scrut, a0.guard.as_ref());
        self.w(" ");
        self.block(then_block, ns);
        if !no_else {
            self.else_branch(&a1.body, ns);
        }
    }

    fn else_branch(&mut self, eb: &Expr, ns: bool) {
        self.w(" else ");
        match &eb.kind {
            ExprKind::Block(b) => self.block(b, ns),
            ExprKind::If { .. } => self.expr_inner(eb, Cx::top(ns)),
            ExprKind::Match { .. } if desugared_arms(eb, ArmOrigin::LetFallthrough).is_some() => {
                self.expr_inner(eb, Cx::top(ns))
            }
            _ => self.unsupported("an `else` branch that is not a block or an `if`"),
        }
    }

    // ---- Statement-start analysis (when a block-like statement needs its `;`) ----

    /// Whether the statement `st` prints starting with a character that would continue a
    /// block-like expression statement before it (`(`, `[`, `-`, `*`, `&`, `!`, `|`, …).
    fn stmt_starts_cont(&self, st: &Stmt, ns: bool) -> bool {
        if !st.attrs.is_empty() {
            return false;
        }
        match &st.kind {
            StmtKind::Let { .. }
            | StmtKind::Return(_)
            | StmtKind::Break(..)
            | StmtKind::Continue(_)
            | StmtKind::Defer(_)
            | StmtKind::While { .. }
            | StmtKind::For { .. } => false,
            StmtKind::Assign { target, .. } => self.starts_cont(target, Cx::top(ns)),
            StmtKind::Expr(e) => {
                if desugared_arms(e, ArmOrigin::LetElse).is_some() || self.is_while_let(e) {
                    false
                } else {
                    self.starts_cont(e, Cx::top(ns))
                }
            }
        }
    }

    pub(crate) fn needs_parens(&self, e: &Expr, cx: Cx) -> bool {
        level(e) < cx.min || (cx.ns && matches!(e.kind, ExprKind::StructLit { .. }))
    }

    /// Whether `e`, printed in `cx`, starts with a continuation character.
    fn starts_cont(&self, e: &Expr, cx: Cx) -> bool {
        if self.needs_parens(e, cx) {
            return true;
        }
        match &e.kind {
            ExprKind::Unary { .. } | ExprKind::Closure { .. } => true,
            ExprKind::TupleLit(_) | ExprKind::ArrayLit(_) | ExprKind::ArrayRepeat { .. } => true,
            ExprKind::Binary { op, lhs, .. } => self.starts_cont(lhs, cx.at(binop_bp(*op))),
            ExprKind::Cast { expr, .. } => self.starts_cont(expr, cx.at(LVL_CAST)),
            ExprKind::Call { callee, args, .. } => match self.slice_call(callee, args) {
                Some((base, _, _)) => self.postfix_base_starts_cont(base, cx, false),
                None => self.postfix_base_starts_cont(callee, cx, false),
            },
            ExprKind::Index { base, .. } | ExprKind::Field { base, .. } => {
                self.postfix_base_starts_cont(base, cx, false)
            }
            ExprKind::TupleField { base, .. } => self.postfix_base_starts_cont(base, cx, true),
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::StructLit { .. }
            | ExprKind::Block(_)
            | ExprKind::If { .. }
            | ExprKind::Match { .. }
            | ExprKind::Loop { .. }
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_) => false,
        }
    }

    fn postfix_base_starts_cont(&self, base: &Expr, cx: Cx, tuple_field: bool) -> bool {
        if self.postfix_base_parens(base, cx, tuple_field) {
            return true;
        }
        self.starts_cont(base, cx.at(LVL_POSTFIX))
    }

    // ---- Expressions ----

    /// The expression at the root of a statement (or a `const`): the one place a call, a struct
    /// literal or an array literal may wrap. `suffix` is what follows it on the line.
    pub(crate) fn root_expr(&mut self, e: &Expr, cx: Cx, suffix: &str) {
        if self.flat || self.needs_parens(e, cx) {
            self.expr(e, cx);
            return;
        }
        let core = self.s.strip_parens(e.span).1;
        let fits = |p: &Self, flat: &str| {
            p.col() + flat.chars().count() + suffix.len() <= p.opts.max_width
        };
        match &e.kind {
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } if self.slice_call(callee, args).is_none() => {
                let Some(flat) = self.flat_of(|p| p.expr(e, cx)) else {
                    self.expr(e, cx);
                    return;
                };
                if self.s.has_comment_in(core.lo, core.hi) || !fits(self, &flat) {
                    self.postfix_base(callee, cx, false);
                    self.turbofish(generic_args);
                    self.w("(");
                    let close = core.hi - 1;
                    self.open();
                    for (k, a) in args.iter().enumerate() {
                        let next = args.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
                        self.leading(a.span.lo);
                        self.expr(a, Cx::top(false));
                        self.w(",");
                        self.trailing(a.span.hi, next);
                    }
                    self.close_at(close, ")");
                    return;
                }
            }
            ExprKind::StructLit { path, fields, rest } if !fields.is_empty() && rest.is_none() => {
                let Some(flat) = self.flat_of(|p| p.expr(e, cx)) else {
                    self.expr(e, cx);
                    return;
                };
                if self.s.has_comment_in(core.lo, core.hi) || !fits(self, &flat) {
                    self.path_with(path, "::");
                    self.w(" {");
                    let close = core.hi - 1;
                    self.open();
                    for (k, f) in fields.iter().enumerate() {
                        let next = fields.get(k + 1).map(|n| n.name.span.lo).unwrap_or(close);
                        self.leading(f.name.span.lo);
                        self.ident(&f.name);
                        self.w(": ");
                        self.expr(&f.value, Cx::top(false));
                        self.w(",");
                        self.trailing(f.value.span.hi, next);
                    }
                    self.close_at(close, "}");
                    return;
                }
            }
            ExprKind::ArrayLit(elems) if !elems.is_empty() => {
                let Some(flat) = self.flat_of(|p| p.expr(e, cx)) else {
                    self.expr(e, cx);
                    return;
                };
                let commented = self.s.has_comment_in(core.lo, core.hi);
                if commented || !fits(self, &flat) {
                    let close = core.hi - 1;
                    self.w("[");
                    self.open();
                    let simple = elems.iter().all(is_simple_elem);
                    if simple && !commented {
                        // Fill lines greedily.
                        let mut fresh = true;
                        for el in elems {
                            let text = format!(
                                "{},",
                                self.flat_of(|p| p.expr(el, Cx::top(false)))
                                    .unwrap_or_default()
                            );
                            if !fresh && self.col() + 1 + text.chars().count() > self.opts.max_width
                            {
                                self.nl();
                                fresh = true;
                            }
                            if !fresh {
                                self.w(" ");
                            }
                            self.w(&text);
                            fresh = false;
                        }
                        self.nl();
                        self.close_bare(close, "]");
                        return;
                    } else {
                        for (k, el) in elems.iter().enumerate() {
                            let next = elems.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
                            self.leading(el.span.lo);
                            self.expr(el, Cx::top(false));
                            self.w(",");
                            self.trailing(el.span.hi, next);
                        }
                    }
                    self.close_at(close, "]");
                    return;
                }
            }
            _ => {}
        }
        self.expr(e, cx);
    }

    pub(crate) fn expr(&mut self, e: &Expr, cx: Cx) {
        if self.failed {
            return;
        }
        if self.needs_parens(e, cx) {
            self.w("(");
            self.expr_inner(e, Cx::top(false));
            self.w(")");
        } else {
            self.expr_inner(e, cx);
        }
    }

    /// A range index `base[lo..hi]` the parser desugared to `$slice(base, lo, hi)` /
    /// `$slice_from(base, lo)`: its base, low bound (`None` when elided) and high bound.
    fn slice_call<'e>(
        &self,
        callee: &'e Expr,
        args: &'e [Expr],
    ) -> Option<(&'e Expr, Option<&'e Expr>, Option<&'e Expr>)> {
        let ExprKind::Path(p) = &callee.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        let name = self.sym(p.first().sym);
        let (base, lo, hi) = match (name, args) {
            ("$slice", [b, l, h]) => (b, l, Some(h)),
            ("$slice_from", [b, l]) => (b, l, None),
            _ => return None,
        };
        // An elided low bound is the synthesized `0` with a zero-width span at the `..`.
        let lo = match lo.kind {
            ExprKind::Int(_) if lo.span.lo == lo.span.hi => None,
            _ => Some(lo),
        };
        Some((base, lo, hi))
    }

    fn postfix_base_parens(&self, base: &Expr, cx: Cx, tuple_field: bool) -> bool {
        level(base) < LVL_POSTFIX
            || is_block_like(base)
            || (cx.ns && matches!(base.kind, ExprKind::StructLit { .. }))
            || (tuple_field && matches!(base.kind, ExprKind::Int(_) | ExprKind::Float(_)))
    }

    fn postfix_base(&mut self, base: &Expr, cx: Cx, tuple_field: bool) {
        if self.postfix_base_parens(base, cx, tuple_field) {
            self.w("(");
            self.expr_inner(base, Cx::top(false));
            self.w(")");
        } else {
            self.expr(base, cx.at(LVL_POSTFIX));
        }
    }

    /// The token that holds a tuple field's index: the last token of its (unparenthesized) span.
    fn index_token(&self, e: &Expr) -> Option<Token> {
        let core = self.s.strip_parens(e.span).1;
        self.s.tok_before(core.hi).filter(|t| t.span.hi == core.hi)
    }

    /// `base.N`, spelled with the source's tokens: `t.0.0` is one float token `0.0` the parser
    /// splits into two accesses, and `t.1 .2` is two integer tokens, which printed as `t.1.2`
    /// would lex as the float `1.2` — so the inner access is parenthesized, `(t.1).2`.
    fn tuple_field(&mut self, e: &Expr, base: &Expr, index: u32, cx: Cx) {
        let tok = self.index_token(e);
        let base_tok = match base.kind {
            ExprKind::TupleField { .. } => self.index_token(base),
            _ => None,
        };
        let glued = |t: Option<Token>| t.is_some_and(|t| t.kind == T::Float);
        let index_text = match tok {
            Some(t) if t.kind == T::Int => self.s.text(t.span).to_string(),
            Some(t) if t.kind == T::Float => {
                let text = self.s.text(t.span);
                let (a, b) = text.split_once('.').unwrap_or((text, text));
                // The outer access of the pair shares the token with its base.
                if base_tok == Some(t) { b } else { a }.to_string()
            }
            _ => index.to_string(),
        };
        let split_base = matches!(tok, Some(t) if t.kind == T::Int)
            && matches!(base_tok, Some(b) if b.kind == T::Int);
        if split_base && !glued(tok) {
            self.w("(");
            self.expr_inner(base, Cx::top(false));
            self.w(")");
        } else {
            self.postfix_base(base, cx, true);
        }
        self.w(".");
        self.w(&index_text);
    }

    fn turbofish(&mut self, generic_args: &[TypeExpr]) {
        if generic_args.is_empty() {
            return;
        }
        self.w("::<");
        for (k, t) in generic_args.iter().enumerate() {
            if k > 0 {
                self.w(", ");
            }
            self.ty(t);
        }
        self.w(">");
    }

    fn args(&mut self, args: &[Expr]) {
        self.w("(");
        for (k, a) in args.iter().enumerate() {
            if k > 0 {
                self.w(", ");
            }
            self.expr(a, Cx::top(false));
        }
        self.w(")");
    }

    pub(crate) fn expr_inner(&mut self, e: &Expr, cx: Cx) {
        match &e.kind {
            ExprKind::Int(s) | ExprKind::Float(s) | ExprKind::Str(s) | ExprKind::Char(s) => {
                let t = self.sym(*s);
                self.w(t);
            }
            ExprKind::Bool(b) => self.w(if *b { "true" } else { "false" }),
            ExprKind::Path(p) => {
                if p.segments.iter().any(|s| self.sym(s.sym).starts_with('$')) {
                    self.unsupported("a compiler-internal `$` name outside a range index");
                    return;
                }
                self.path_with(p, "::");
            }
            ExprKind::Unary { op, expr } => {
                match op {
                    UnOp::Neg => self.w("-"),
                    UnOp::Not => {
                        // `~` and `!` both parse to `Not`; keep the one written.
                        let tilde = self.s.kind_before(expr.span.lo) == Some(T::Tilde);
                        self.w(if tilde { "~" } else { "!" });
                    }
                    UnOp::Deref => self.w("*"),
                    UnOp::Ref => {
                        self.w("&");
                        // `& &x`, never `&&x` (one `&&` token).
                        if matches!(
                            expr.kind,
                            ExprKind::Unary {
                                op: UnOp::Ref | UnOp::RefMut,
                                ..
                            }
                        ) {
                            self.w(" ");
                        }
                    }
                    UnOp::RefMut => self.w("&mut "),
                }
                self.expr(expr, cx.at(LVL_UNARY));
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let bp = binop_bp(*op);
                self.expr(lhs, cx.at(bp));
                self.w(" ");
                self.w(op.glyph());
                self.w(" ");
                self.expr(rhs, cx.at(bp + 1));
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                if let Some((base, lo, hi)) = self.slice_call(callee, args) {
                    if !generic_args.is_empty() {
                        self.unsupported("a range index with type arguments");
                        return;
                    }
                    self.postfix_base(base, cx, false);
                    self.w("[");
                    if let Some(lo) = lo {
                        self.expr(lo, Cx::top(cx.ns));
                    }
                    self.w("..");
                    if let Some(hi) = hi {
                        self.expr(hi, Cx::top(cx.ns));
                    }
                    self.w("]");
                    return;
                }
                self.postfix_base(callee, cx, false);
                self.turbofish(generic_args);
                self.args(args);
            }
            ExprKind::Index { base, indices } => {
                self.postfix_base(base, cx, false);
                self.w("[");
                for (k, ix) in indices.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    // Index brackets do not re-enable struct literals.
                    self.expr(ix, Cx::top(cx.ns));
                }
                self.w("]");
            }
            ExprKind::Field { base, name } => {
                self.postfix_base(base, cx, false);
                let sep = if self.s.kind_before(name.span.lo) == Some(T::ColonColon) {
                    "::"
                } else {
                    "."
                };
                self.w(sep);
                self.ident(name);
            }
            ExprKind::TupleField { base, index } => self.tuple_field(e, base, *index, cx),
            ExprKind::Cast { expr, ty } => {
                self.expr(expr, cx.at(LVL_CAST));
                self.w(" as ");
                self.ty(ty);
            }
            ExprKind::StructLit { path, fields, rest } => {
                self.path_with(path, "::");
                if fields.is_empty() && rest.is_none() {
                    self.w(" {}");
                    return;
                }
                self.w(" { ");
                for (k, f) in fields.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.ident(&f.name);
                    self.w(": ");
                    self.expr(&f.value, Cx::top(false));
                }
                if let Some(r) = rest {
                    if !fields.is_empty() {
                        self.w(", ");
                    }
                    self.w("..");
                    self.expr(r, Cx::top(false));
                }
                self.w(" }");
            }
            ExprKind::ArrayLit(items) => {
                self.w("[");
                for (k, it) in items.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.expr(it, Cx::top(false));
                }
                self.w("]");
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.w("[");
                self.expr(value, Cx::top(false));
                self.w("; ");
                self.expr(count, Cx::top(false));
                self.w("]");
            }
            ExprKind::TupleLit(items) => {
                self.w("(");
                for (k, it) in items.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.expr(it, Cx::top(false));
                }
                if items.len() == 1 {
                    self.w(",");
                }
                self.w(")");
            }
            ExprKind::Block(b) => self.block(b, cx.ns),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                if self.flat {
                    self.failed = true;
                    return;
                }
                self.w("if ");
                self.expr(cond, Cx::top(true));
                self.w(" ");
                self.block(then_branch, cx.ns);
                if let Some(eb) = else_branch {
                    self.else_branch(eb, cx.ns);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                if self.flat {
                    self.failed = true;
                    return;
                }
                if let Some((scrut, arms)) = desugared_arms(e, ArmOrigin::LetFallthrough) {
                    self.if_let(scrut, arms, cx.ns);
                    return;
                }
                if arms.iter().any(|a| a.origin != ArmOrigin::Source) {
                    self.unsupported("a desugared `match` outside the construct that produced it");
                    return;
                }
                self.match_expr(e, scrutinee, arms, cx.ns);
            }
            ExprKind::Loop { label, body } => {
                if self.flat {
                    self.failed = true;
                    return;
                }
                if self.is_while_let(e) {
                    self.unsupported("a `while let` in expression position");
                    return;
                }
                if let Some(l) = label {
                    self.w("'");
                    let t = self.sym(l.sym);
                    self.w(t);
                    self.w(": ");
                }
                self.w("loop ");
                self.block(body, cx.ns);
            }
            ExprKind::SizeOf(t) => {
                self.w("sizeof[");
                self.ty(t);
                self.w("]");
            }
            ExprKind::AlignOf(t) => {
                self.w("alignof[");
                self.ty(t);
                self.w("]");
            }
            ExprKind::Closure { params, ret, body } => {
                self.w("|");
                for (k, p) in params.iter().enumerate() {
                    if k > 0 {
                        self.w(", ");
                    }
                    self.ident(&p.name);
                    if let Some(t) = &p.ty {
                        self.w(": ");
                        self.ty(t);
                    }
                }
                self.w("|");
                match ret {
                    Some(r) => {
                        self.w(" -> ");
                        self.ty(r);
                        self.w(" ");
                        match &body.kind {
                            ExprKind::Block(b) => self.block(b, cx.ns),
                            _ => self.unsupported("a closure with a return type and no block body"),
                        }
                    }
                    None => {
                        self.w(" ");
                        self.expr(body, Cx::top(cx.ns));
                    }
                }
            }
        }
    }

    fn match_expr(&mut self, e: &Expr, scrutinee: &Expr, arms: &[MatchArm], ns: bool) {
        self.w("match ");
        self.expr(scrutinee, Cx::top(true));
        self.w(" ");
        let (_, core) = self.s.strip_parens(e.span);
        let close = core.hi.saturating_sub(1);
        if arms.is_empty() && !self.s.has_comment_in(scrutinee.span.hi, close) {
            self.w("{}");
            return;
        }
        self.w("{");
        self.open();
        for (k, a) in arms.iter().enumerate() {
            let next = arms.get(k + 1).map(|n| n.span.lo).unwrap_or(close);
            self.leading(a.span.lo);
            self.pat(&a.pat);
            if let Some(g) = &a.guard {
                self.w(" if ");
                self.expr(g, Cx::top(true));
            }
            self.w(" => ");
            match &a.body.kind {
                ExprKind::Block(b) => {
                    self.block(b, ns);
                    // After a block body a comma is optional — unless the next arm's pattern
                    // would continue the block as a call, an index or a subtraction.
                    let cont = arms
                        .get(k + 1)
                        .is_some_and(|n| pat_starts_with_open(&n.pat));
                    if cont {
                        self.w(",");
                    }
                }
                _ => {
                    self.root_expr(&a.body, Cx::top(ns), ",");
                    self.w(",");
                }
            }
            self.trailing(a.body.span.hi.max(a.span.hi), next);
        }
        self.close_at(close, "}");
    }
}

/// Whether a pattern prints starting with `(`, `[` or `-`.
fn pat_starts_with_open(p: &Pattern) -> bool {
    match &p.kind {
        PatKind::Int { neg, .. } => *neg,
        PatKind::Tuple(_) | PatKind::Unit | PatKind::Slice { .. } => true,
        PatKind::Range { lo, .. } => pat_starts_with_open(lo),
        PatKind::Or(alts) => alts.first().is_some_and(pat_starts_with_open),
        PatKind::Wildcard
        | PatKind::Ident(_)
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Variant { .. }
        | PatKind::Struct { .. }
        | PatKind::Binding { .. } => false,
    }
}

/// An array element the greedy fill packs: a literal, a path, or a negated literal.
fn is_simple_elem(e: &Expr) -> bool {
    let lit = |k: &ExprKind| matches!(k, ExprKind::Int(_) | ExprKind::Float(_));
    matches!(
        e.kind,
        ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
    ) || matches!(&e.kind, ExprKind::Unary { op: UnOp::Neg, expr } if lit(&expr.kind))
}
