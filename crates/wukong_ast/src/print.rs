//! A deterministic, indented pretty-printer for the AST.
//!
//! Used by `--emit=ast` and by snapshot tests. The output is a stable two-space-indented tree —
//! leaves carry their payload inline (literal text exactly as written, names, type syntax as
//! written), while compound nodes break their children onto indented lines. It is deliberately
//! **lossy**: it is for humans and snapshots, not for round-tripping back to source.

use crate::*;
use wukong_span::{Interner, Symbol};

/// Render a whole module to a string.
pub fn print_module(m: &Module, interner: &Interner) -> String {
    let mut p = AstPrinter {
        interner,
        out: String::new(),
        depth: 0,
    };
    p.module(m);
    p.out
}

/// Render a single expression as an indented tree (used by parser tests).
pub fn print_expr(e: &Expr, interner: &Interner) -> String {
    let mut p = AstPrinter {
        interner,
        out: String::new(),
        depth: 0,
    };
    p.expr(e);
    p.out
}

/// Render a single type to its inline source-like form (used by parser tests).
pub fn print_type(t: &TypeExpr, interner: &Interner) -> String {
    let p = AstPrinter {
        interner,
        out: String::new(),
        depth: 0,
    };
    p.type_str(t)
}

struct AstPrinter<'a> {
    interner: &'a Interner,
    out: String,
    depth: usize,
}

impl AstPrinter<'_> {
    fn sym(&self, s: Symbol) -> &str {
        self.interner.resolve(s)
    }

    fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.depth += 1;
        f(self);
        self.depth -= 1;
    }

    // --- Paths & types (inline) ---

    fn path_str(&self, p: &Path, sep: &str) -> String {
        p.segments
            .iter()
            .map(|s| self.sym(s.sym))
            .collect::<Vec<_>>()
            .join(sep)
    }

    fn type_str(&self, t: &TypeExpr) -> String {
        match &t.kind {
            TypeKind::Path(p) => self.path_str(p, "::"),
            TypeKind::Int(s) => self.sym(*s).to_string(),
            TypeKind::Unit => "()".to_string(),
            TypeKind::Pointer { mutable, pointee } => {
                format!(
                    "*{}{}",
                    if *mutable { "mut " } else { "" },
                    self.type_str(pointee)
                )
            }
            TypeKind::Ref { mutable, pointee } => {
                format!(
                    "&{}{}",
                    if *mutable { "mut " } else { "" },
                    self.type_str(pointee)
                )
            }
            TypeKind::Slice(e) => format!("[]{}", self.type_str(e)),
            TypeKind::Array { elem, len } => {
                format!("[{}; {}]", self.type_str(elem), self.expr_inline(len))
            }
            TypeKind::Tuple(items) => {
                let parts: Vec<_> = items.iter().map(|i| self.type_str(i)).collect();
                format!("({})", parts.join(", "))
            }
            TypeKind::Vector { elem, lanes } => format!("{}x{}", self.type_str(elem), lanes),
            TypeKind::Tensor { elem, dims, layout } => {
                let mut parts = vec![self.type_str(elem)];
                for d in dims {
                    parts.push(match &d.kind {
                        DimKind::Int(n) => n.to_string(),
                        DimKind::Named(s) => self.sym(*s).to_string(),
                        DimKind::Dynamic => "?".to_string(),
                    });
                }
                let lay = match layout {
                    Some(Layout::Contiguous) => ", .contiguous".to_string(),
                    Some(Layout::ColMajor) => ", .col_major".to_string(),
                    Some(Layout::Strided) => ", .strided".to_string(),
                    Some(Layout::Tiled(extents)) => format!(
                        ", .tiled({})",
                        extents
                            .iter()
                            .map(|e| e.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    None => String::new(),
                };
                format!("Tensor[{}{}]", parts.join(", "), lay)
            }
        }
    }

    /// A compact inline form for the two places an expression appears inside a one-line rendering:
    /// an array length in `type_str` and an enum variant's explicit discriminant. Literals, paths and
    /// simple unary/binary expressions render; anything else is intentionally lossy and prints as
    /// `<expr>`.
    fn expr_inline(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Int(s) | ExprKind::Float(s) | ExprKind::Str(s) | ExprKind::Char(s) => {
                self.sym(*s).to_string()
            }
            ExprKind::Bool(b) => b.to_string(),
            ExprKind::Path(p) => self.path_str(p, "::"),
            ExprKind::Binary { op, lhs, rhs } => {
                format!(
                    "({} {} {})",
                    self.expr_inline(lhs),
                    op.glyph(),
                    self.expr_inline(rhs)
                )
            }
            ExprKind::Unary { op, expr } => {
                format!("{}{}", op.glyph().trim(), self.expr_inline(expr))
            }
            _ => "<expr>".to_string(),
        }
    }

    fn attr_str(&self, a: &Attr) -> String {
        if a.args.is_empty() {
            return format!("@{}", self.sym(a.name.sym));
        }
        let args: Vec<String> = a
            .args
            .iter()
            .map(|arg| match arg {
                AttrArg::Word(s, _) | AttrArg::Int(s, _) | AttrArg::Str(s, _) => {
                    self.sym(*s).to_string()
                }
                AttrArg::KeyValue { key, value } => {
                    let v = match value {
                        AttrVal::Int(s) | AttrVal::Str(s) | AttrVal::Word(s) => {
                            self.sym(*s).to_string()
                        }
                        AttrVal::Bool(b) => b.to_string(),
                    };
                    format!("{} = {}", self.sym(key.sym), v)
                }
            })
            .collect();
        format!("@{}({})", self.sym(a.name.sym), args.join(", "))
    }

    /// A one-line `fn name(a: T, b: U) -> R` signature, used where there is no body to hang a
    /// tree off (extern declarations) — printing only the name there hides the whole ABI surface.
    fn fn_sig_str(&self, f: &FnDecl) -> String {
        let params: Vec<String> = f
            .params
            .iter()
            .map(|p| {
                format!(
                    "{}{}: {}",
                    if p.mutable { "mut " } else { "" },
                    self.sym(p.name.sym),
                    self.type_str(&p.ty)
                )
            })
            .collect();
        let ret = match &f.ret {
            Some(t) => format!(" -> {}", self.type_str(t)),
            None => String::new(),
        };
        format!("fn {}({}){}", self.sym(f.name.sym), params.join(", "), ret)
    }

    /// An enum variant's payload in source form: `(i32, f32)`, `{ r: f32 }`, or empty for a unit
    /// variant. Dropping it made every variant render identically as `variant A`.
    fn variant_data_str(&self, d: &VariantData) -> String {
        match d {
            VariantData::Unit => String::new(),
            VariantData::Tuple(tys) => {
                let parts: Vec<String> = tys.iter().map(|t| self.type_str(t)).collect();
                format!("({})", parts.join(", "))
            }
            VariantData::Struct(fields) => {
                let parts: Vec<String> = fields
                    .iter()
                    .map(|f| format!("{}: {}", self.sym(f.name.sym), self.type_str(&f.ty)))
                    .collect();
                format!(" {{ {} }}", parts.join(", "))
            }
        }
    }

    // --- Module & items ---

    fn module(&mut self, m: &Module) {
        if let Some(name) = &m.name {
            self.line(format!("module {}", self.path_str(name, ".")));
        }
        for item in &m.items {
            self.item(item);
        }
    }

    fn item(&mut self, item: &Item) {
        for a in &item.attrs {
            self.line(self.attr_str(a));
        }
        match &item.kind {
            ItemKind::Fn(f) => {
                let vis = if f.is_pub { "pub " } else { "" };
                self.line(format!("fn {}{}", vis, self.sym(f.name.sym)));
                self.indented(|p| {
                    p.generics(&f.generics);
                    for param in &f.params {
                        p.param(param);
                    }
                    if let Some(ret) = &f.ret {
                        p.line(format!("ret {}", p.type_str(ret)));
                    }
                    match &f.body {
                        Some(b) => p.block(b),
                        None => p.line("(no body)"),
                    }
                });
            }
            ItemKind::Struct(s) => {
                self.line(format!("struct {}", self.sym(s.name.sym)));
                self.indented(|p| {
                    p.generics(&s.generics);
                    for field in &s.fields {
                        p.line(format!(
                            "field {}: {}",
                            p.sym(field.name.sym),
                            p.type_str(&field.ty)
                        ));
                    }
                });
            }
            ItemKind::Enum(e) => {
                self.line(format!("enum {}", self.sym(e.name.sym)));
                self.indented(|p| {
                    for v in &e.variants {
                        let disc = match &v.discriminant {
                            Some(d) => format!(" = {}", p.expr_inline(d)),
                            None => String::new(),
                        };
                        p.line(format!(
                            "variant {}{}{}",
                            p.sym(v.name.sym),
                            p.variant_data_str(&v.data),
                            disc
                        ));
                    }
                });
            }
            ItemKind::Const(c) => {
                self.line(format!(
                    "const {}: {}",
                    self.sym(c.name.sym),
                    self.type_str(&c.ty)
                ));
                self.indented(|p| p.expr(&c.value));
            }
            ItemKind::Import(i) => {
                let mut s = format!("import {}", self.path_str(&i.path, "."));
                if let Some(items) = &i.items {
                    let names: Vec<&str> = items.iter().map(|n| self.sym(n.sym)).collect();
                    s.push_str(&format!(".{{{}}}", names.join(", ")));
                }
                if let Some(alias) = &i.alias {
                    s.push_str(&format!(" as {}", self.sym(alias.sym)));
                }
                self.line(s);
            }
            ItemKind::Extern(e) => {
                self.line(format!("extern {}", self.sym(e.abi)));
                self.indented(|p| {
                    for f in &e.items {
                        p.line(p.fn_sig_str(f));
                    }
                });
            }
        }
    }

    fn generics(&mut self, gs: &[GenericParam]) {
        if gs.is_empty() {
            return;
        }
        self.line("generics");
        self.indented(|p| {
            for g in gs {
                match &g.kind {
                    GenericParamKind::Type(id) => p.line(format!("type-param {}", p.sym(id.sym))),
                    GenericParamKind::Const { name, ty } => p.line(format!(
                        "const-param {}: {}",
                        p.sym(name.sym),
                        p.type_str(ty)
                    )),
                }
            }
        });
    }

    fn param(&mut self, param: &Param) {
        for a in &param.attrs {
            self.line(self.attr_str(a));
        }
        let mt = if param.mutable { "mut " } else { "" };
        self.line(format!(
            "param {}{}: {}",
            mt,
            self.sym(param.name.sym),
            self.type_str(&param.ty)
        ));
    }

    // --- Statements & blocks ---

    fn block(&mut self, b: &Block) {
        self.line("block");
        self.indented(|p| {
            for s in &b.stmts {
                p.stmt(s);
            }
            if let Some(tail) = &b.tail {
                p.line("tail");
                p.indented(|p| p.expr(tail));
            }
        });
    }

    fn label(&self, l: &Option<Ident>) -> String {
        match l {
            Some(id) => format!(" :{}", self.sym(id.sym)),
            None => String::new(),
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        for a in &s.attrs {
            self.line(self.attr_str(a));
        }
        match &s.kind {
            StmtKind::Let {
                pat,
                mutable,
                ty,
                init,
            } => {
                self.line(format!("let{}", if *mutable { " mut" } else { "" }));
                self.indented(|p| {
                    p.pattern(pat);
                    if let Some(t) = ty {
                        p.line(format!("type {}", p.type_str(t)));
                    }
                    if let Some(i) = init {
                        p.line("init");
                        p.indented(|p| p.expr(i));
                    }
                });
            }
            StmtKind::Assign { target, op, value } => {
                self.line(format!("assign {}", op.glyph()));
                self.indented(|p| {
                    p.expr(target);
                    p.expr(value);
                });
            }
            StmtKind::Expr(e) => self.expr(e),
            StmtKind::Return(opt) => {
                self.line("return");
                if let Some(e) = opt {
                    self.indented(|p| p.expr(e));
                }
            }
            StmtKind::Break(l, val) => {
                self.line(format!("break{}", self.label(l)));
                if let Some(e) = val {
                    self.indented(|p| p.expr(e));
                }
            }
            StmtKind::Continue(l) => self.line(format!("continue{}", self.label(l))),
            StmtKind::Defer(e) => {
                self.line("defer");
                self.indented(|p| p.expr(e));
            }
            StmtKind::While { label, cond, body } => {
                self.line(format!("while{}", self.label(label)));
                self.indented(|p| {
                    p.line("cond");
                    p.indented(|p| p.expr(cond));
                    p.block(body);
                });
            }
            StmtKind::For {
                label,
                pat,
                iter,
                body,
            } => {
                self.line(format!("for{}", self.label(label)));
                self.indented(|p| {
                    p.pattern(pat);
                    p.for_iter(iter);
                    p.block(body);
                });
            }
        }
    }

    fn for_iter(&mut self, iter: &ForIter) {
        match iter {
            ForIter::Range {
                start,
                end,
                inclusive,
                step,
            } => {
                self.line(format!("range{}", if *inclusive { "=" } else { "" }));
                self.indented(|p| {
                    p.line("start");
                    p.indented(|p| p.expr(start));
                    if let Some(e) = end {
                        p.line("end");
                        p.indented(|p| p.expr(e));
                    }
                    if let Some(st) = step {
                        p.line("step");
                        p.indented(|p| p.expr(st));
                    }
                });
            }
            ForIter::Expr(e) => {
                self.line("iter");
                self.indented(|p| p.expr(e));
            }
        }
    }

    fn pattern(&mut self, pat: &Pattern) {
        match &pat.kind {
            PatKind::Wildcard => self.line("pat _"),
            PatKind::Ident(s) => self.line(format!("pat {}", self.sym(*s))),
            PatKind::Unit => self.line("pat ()"),
            PatKind::Tuple(subs) => {
                self.line("pat tuple");
                self.indented(|p| {
                    for s in subs {
                        p.pattern(s);
                    }
                });
            }
            PatKind::Int { sym, neg } => {
                let s = self.sym(*sym);
                self.line(format!("pat {}{}", if *neg { "-" } else { "" }, s));
            }
            PatKind::Char(sym) => {
                let s = self.sym(*sym);
                self.line(format!("pat {s}"));
            }
            PatKind::Bool(b) => self.line(format!("pat {b}")),
            PatKind::Or(alts) => {
                self.line("pat or");
                self.indented(|p| {
                    for a in alts {
                        p.pattern(a);
                    }
                });
            }
            PatKind::Path(path) => self.line(format!("pat {}", self.path_str(path, "::"))),
            PatKind::Range {
                lo,
                hi,
                inclusive,
            } => {
                self.line(format!("pat range{}", if *inclusive { "=" } else { "" }));
                self.indented(|p| {
                    p.pattern(lo);
                    p.pattern(hi);
                });
            }
            PatKind::Variant { path, fields } => {
                self.line(format!("pat variant {}", self.path_str(path, "::")));
                self.indented(|p| match fields {
                    VariantPat::Tuple(subs) => {
                        for s in subs {
                            p.pattern(s);
                        }
                    }
                    VariantPat::Struct(fps) => {
                        for fp in fps {
                            let nm = p.sym(fp.name);
                            p.line(format!("field {nm}"));
                            p.indented(|p| p.pattern(&fp.pat));
                        }
                    }
                });
            }
        }
    }

    // --- Expressions ---

    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Int(s) => self.line(format!("int {}", self.sym(*s))),
            ExprKind::Float(s) => self.line(format!("float {}", self.sym(*s))),
            ExprKind::Str(s) => self.line(format!("str {}", self.sym(*s))),
            ExprKind::Char(s) => self.line(format!("char {}", self.sym(*s))),
            ExprKind::Bool(b) => self.line(format!("bool {b}")),
            ExprKind::Path(p) => self.line(format!("path {}", self.path_str(p, "::"))),
            ExprKind::Unary { op, expr } => {
                self.line(format!("unary {}", op.glyph().trim()));
                self.indented(|p| p.expr(expr));
            }
            ExprKind::Binary { op, lhs, rhs } => {
                self.line(format!("binary {}", op.glyph()));
                self.indented(|p| {
                    p.expr(lhs);
                    p.expr(rhs);
                });
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                self.line("call");
                self.indented(|p| {
                    p.expr(callee);
                    for g in generic_args {
                        p.line(format!("generic {}", p.type_str(g)));
                    }
                    for a in args {
                        p.expr(a);
                    }
                });
            }
            ExprKind::Index { base, indices } => {
                self.line("index");
                self.indented(|p| {
                    p.expr(base);
                    for ix in indices {
                        p.expr(ix);
                    }
                });
            }
            ExprKind::Field { base, name } => {
                self.line(format!("field {}", self.sym(name.sym)));
                self.indented(|p| p.expr(base));
            }
            ExprKind::TupleField { base, index } => {
                self.line(format!("tuple-field {index}"));
                self.indented(|p| p.expr(base));
            }
            ExprKind::Cast { expr, ty } => {
                self.line(format!("cast {}", self.type_str(ty)));
                self.indented(|p| p.expr(expr));
            }
            ExprKind::StructLit { path, fields, rest } => {
                self.line(format!("struct-lit {}", self.path_str(path, "::")));
                self.indented(|p| {
                    for f in fields {
                        p.line(format!("field {}", p.sym(f.name.sym)));
                        p.indented(|p| p.expr(&f.value));
                    }
                    if let Some(r) = rest {
                        p.line("..rest");
                        p.indented(|p| p.expr(r));
                    }
                });
            }
            ExprKind::ArrayLit(items) => {
                self.line("array");
                self.indented(|p| {
                    for e in items {
                        p.expr(e);
                    }
                });
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.line("array-repeat");
                self.indented(|p| {
                    p.expr(value);
                    p.expr(count);
                });
            }
            ExprKind::TupleLit(items) => {
                self.line("tuple");
                self.indented(|p| {
                    for e in items {
                        p.expr(e);
                    }
                });
            }
            ExprKind::Block(b) => self.block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.line("if");
                self.indented(|p| {
                    p.line("cond");
                    p.indented(|p| p.expr(cond));
                    p.line("then");
                    p.indented(|p| p.block(then_branch));
                    if let Some(e) = else_branch {
                        p.line("else");
                        p.indented(|p| p.expr(e));
                    }
                });
            }
            ExprKind::Match { scrutinee, arms } => {
                self.line("match");
                self.indented(|p| {
                    p.expr(scrutinee);
                    for arm in arms {
                        p.line("arm");
                        p.indented(|p| {
                            p.pattern(&arm.pat);
                            if let Some(g) = &arm.guard {
                                p.line("guard");
                                p.indented(|p| p.expr(g));
                            }
                            p.expr(&arm.body);
                        });
                    }
                });
            }
            ExprKind::Loop { label, body } => {
                self.line(format!("loop{}", self.label(label)));
                self.indented(|p| p.block(body));
            }
            ExprKind::SizeOf(ty) => self.line(format!("sizeof {}", self.type_str(ty))),
            ExprKind::AlignOf(ty) => self.line(format!("alignof {}", self.type_str(ty))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::Span;

    #[test]
    fn empty_module_with_name() {
        let mut i = Interner::new();
        let a = i.intern("demo");
        let path = Path {
            segments: vec![Ident {
                sym: a,
                span: Span::dummy(),
            }],
            span: Span::dummy(),
        };
        let m = Module {
            name: Some(path),
            items: vec![],
            span: Span::dummy(),
        };
        assert_eq!(print_module(&m, &i), "module demo\n");
    }

    // --- helpers for the fidelity snapshot ---

    fn id(i: &mut Interner, s: &str) -> Ident {
        Ident {
            sym: i.intern(s),
            span: Span::dummy(),
        }
    }

    fn path(i: &mut Interner, segs: &[&str]) -> Path {
        Path {
            segments: segs.iter().map(|s| id(i, s)).collect(),
            span: Span::dummy(),
        }
    }

    fn named_ty(i: &mut Interner, s: &str) -> TypeExpr {
        TypeExpr {
            id: NodeId::DUMMY,
            kind: TypeKind::Path(path(i, &[s])),
            span: Span::dummy(),
        }
    }

    fn item(kind: ItemKind) -> Item {
        Item {
            id: NodeId::DUMMY,
            attrs: vec![],
            kind,
            span: Span::dummy(),
        }
    }

    /// `--emit=ast` is the parser's only user-facing observable, so the constructs a user would
    /// reach for it to debug must survive the rendering. Before this was pinned the printer
    /// dropped variant payloads, explicit discriminants, import aliases, selective import lists
    /// and extern signatures — an engineer inspecting `E::A = 5` saw a bare `variant A` and
    /// concluded the parser had lost the discriminant.
    #[test]
    fn payloads_aliases_and_signatures_survive_printing() {
        let mut i = Interner::new();

        let aliased = ItemKind::Import(Import {
            path: path(&mut i, &["std", "math"]),
            alias: Some(id(&mut i, "mm")),
            items: None,
        });
        let selective = ItemKind::Import(Import {
            path: path(&mut i, &["a", "b"]),
            alias: None,
            items: Some(vec![id(&mut i, "x"), id(&mut i, "y")]),
        });

        let five = i.intern("5");
        let e = ItemKind::Enum(EnumDecl {
            name: id(&mut i, "E"),
            is_pub: false,
            generics: vec![],
            variants: vec![
                Variant {
                    name: id(&mut i, "A"),
                    data: VariantData::Tuple(vec![
                        named_ty(&mut i, "i32"),
                        named_ty(&mut i, "f32"),
                    ]),
                    discriminant: Some(Expr {
                        id: NodeId::DUMMY,
                        kind: ExprKind::Int(five),
                        span: Span::dummy(),
                    }),
                    span: Span::dummy(),
                },
                Variant {
                    name: id(&mut i, "B"),
                    data: VariantData::Struct(vec![Field {
                        name: id(&mut i, "r"),
                        is_pub: false,
                        ty: named_ty(&mut i, "f32"),
                        span: Span::dummy(),
                    }]),
                    discriminant: None,
                    span: Span::dummy(),
                },
                Variant {
                    name: id(&mut i, "C"),
                    data: VariantData::Unit,
                    discriminant: None,
                    span: Span::dummy(),
                },
            ],
        });

        let u8_ty = named_ty(&mut i, "u8");
        let ext = ItemKind::Extern(ExternBlock {
            abi: i.intern("\"C\""),
            items: vec![FnDecl {
                name: id(&mut i, "puts"),
                is_pub: false,
                generics: vec![],
                params: vec![Param {
                    id: NodeId::DUMMY,
                    attrs: vec![],
                    mutable: false,
                    name: id(&mut i, "s"),
                    ty: TypeExpr {
                        id: NodeId::DUMMY,
                        kind: TypeKind::Pointer {
                            mutable: false,
                            pointee: Box::new(u8_ty),
                        },
                        span: Span::dummy(),
                    },
                    span: Span::dummy(),
                }],
                ret: Some(named_ty(&mut i, "i32")),
                body: None,
            }],
        });

        let m = Module {
            name: None,
            items: vec![item(aliased), item(selective), item(e), item(ext)],
            span: Span::dummy(),
        };

        assert_eq!(
            print_module(&m, &i),
            "import std.math as mm\n\
             import a.b.{x, y}\n\
             enum E\n\
             \x20 variant A(i32, f32) = 5\n\
             \x20 variant B { r: f32 }\n\
             \x20 variant C\n\
             extern \"C\"\n\
             \x20 fn puts(s: *u8) -> i32\n"
        );
    }

    /// A tiled tensor layout carries its extents; rendering them as `.tiled(..)` hid exactly the
    /// numbers a layout bug turns on.
    #[test]
    fn tiled_layout_prints_its_extents() {
        let mut i = Interner::new();
        let t = TypeExpr {
            id: NodeId::DUMMY,
            kind: TypeKind::Tensor {
                elem: Box::new(named_ty(&mut i, "f32")),
                dims: vec![
                    Dim {
                        kind: DimKind::Int(512),
                        span: Span::dummy(),
                    },
                    Dim {
                        kind: DimKind::Int(512),
                        span: Span::dummy(),
                    },
                ],
                layout: Some(Layout::Tiled(vec![64, 32])),
            },
            span: Span::dummy(),
        };
        assert_eq!(print_type(&t, &i), "Tensor[f32, 512, 512, .tiled(64, 32)]");
    }
}
