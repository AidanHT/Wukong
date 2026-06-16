//! A deterministic, indented pretty-printer for the AST.
//!
//! Used by `--emit=ast` and by snapshot tests. The output is a stable indented tree — leaves
//! carry their scalar payload inline (literals, names, fully-resolved type syntax), while
//! compound nodes break their children onto indented lines.

use crate::*;
use mercury_span::{Interner, Symbol};

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
                    Some(Layout::Contiguous) => ", .contiguous",
                    Some(Layout::ColMajor) => ", .col_major",
                    Some(Layout::Strided) => ", .strided",
                    Some(Layout::Tiled(_)) => ", .tiled(..)",
                    None => "",
                };
                format!("Tensor[{}{}]", parts.join(", "), lay)
            }
        }
    }

    /// A compact inline form for small expressions (array lengths, const generics).
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
                        p.line(format!("variant {}", p.sym(v.name.sym)));
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
                self.line(format!("import {}", self.path_str(&i.path, ".")));
            }
            ItemKind::Extern(e) => {
                self.line(format!("extern {}", self.sym(e.abi)));
                self.indented(|p| {
                    for f in &e.items {
                        p.line(format!("fn {}", p.sym(f.name.sym)));
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
        self.line(format!(
            "param {}: {}",
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
            StmtKind::Break(l) => self.line(format!("break{}", self.label(l))),
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
            StmtKind::Loop { label, body } => {
                self.line(format!("loop{}", self.label(label)));
                self.indented(|p| p.block(body));
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
                            p.expr(&arm.body);
                        });
                    }
                });
            }
            ExprKind::SizeOf(ty) => self.line(format!("sizeof {}", self.type_str(ty))),
            ExprKind::AlignOf(ty) => self.line(format!("alignof {}", self.type_str(ty))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_span::Span;

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
}
